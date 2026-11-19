# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long logical prefill parity; stream the memory-constrained single-chip reference.

Run baseline and mesh serially through the external run_recorded wrapper. Baseline
uploads/downloads <=32 rows per public prefill call; mesh makes ONE public prefill
call with the full logical input. Both continue the same state for three decodes.
Reported wall times include transfers and are correctness-run times, not speedups.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_decoder import (
    Collectives,
    MeshPackedEmbedding,
    MultichipDecoder,
    OptimizedDecoder,
    PackedEmbedding,
    load_ple_reference,
    load_weights,
    runtime_audit,
    token_rows,
)


def metadata(a):
    return dict(
        schema=1,
        probe="multichip_long_context",
        snapshot=str(a.snapshot.resolve()),
        layer=a.layer,
        length=a.length,
        capacity=a.capacity,
        token_policy="(123 + global_position) % 248320",
        baseline_prefill_chunk=32,
        mesh_prefill_calls=1,
        expert_dtype="bfloat4_b",
    )


def state_names(layer):
    if layer == 3:
        return ("position", "page_table", "k_cache", "v_cache", "raw_history", "compressed_keys")
    return ("conv", "recurrent") + (("ple_conv", "token_history") if layer == 1 else ())


def whole_metric(expected, actual):
    """Whole-tensor PCC from bounded FP64 sufficient statistics, never row PCC."""
    if expected.shape != actual.shape:
        raise ValueError(f"Shape mismatch: {expected.shape} != {actual.shape}")
    exact = torch.equal(expected, actual)
    left, right = expected.reshape(-1), actual.reshape(-1)
    sums = torch.zeros(5, dtype=torch.float64)
    finite, maximum = True, 0.0
    for start in range(0, left.numel(), 1024 * 1024):
        x, y = left[start : start + 1024 * 1024].double(), right[start : start + 1024 * 1024].double()
        finite &= bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        maximum = max(maximum, float(y.abs().max()))
        sums += torch.stack((x.sum(), y.sum(), x.square().sum(), y.square().sum(), (x * y).sum()))
    sx, sy, xx, yy, xy = sums.tolist()
    n = left.numel()
    denominator = math.sqrt(max(0.0, xx - sx * sx / n) * max(0.0, yy - sy * sy / n))
    correlation = 1.0 if exact and finite else (xy - sx * sy / n) / denominator if denominator else 0.0
    if not math.isfinite(correlation):
        correlation = 0.0
    return dict(exact=exact, pcc=correlation, finite=finite, max_absolute=maximum)


def mesh_state_reference(name, value):
    if name in ("k_cache", "v_cache"):
        return torch.cat([value[:, r // 2 : r // 2 + 1] for r in range(4)], dim=0)
    if name == "recurrent":
        return torch.cat(value.chunk(4, dim=1), dim=0)
    if name == "conv":
        q, k, v = value.split((2048, 2048, 6144), dim=-1)
        return torch.cat(
            [
                torch.cat(
                    (
                        q[..., r * 512 : (r + 1) * 512],
                        k[..., r * 512 : (r + 1) * 512],
                        v[..., r * 1536 : (r + 1) * 1536],
                    ),
                    dim=-1,
                )
                for r in range(4)
            ],
            dim=0,
        )
    if name == "ple_conv":
        return torch.cat(value.chunk(4, dim=-1), dim=0)
    return torch.cat([value] * 4, dim=0)


def validate_reference(a, saved):
    if saved.get("metadata") != metadata(a):
        raise ValueError("Reference provenance differs; regenerate the long-context baseline")
    lengths = [a.length, 1, 1, 1]
    for key in ("inputs", "outputs"):
        values = saved.get(key, [])
        if len(values) != 4:
            raise ValueError(f"Reference {key} must include prefill and three decodes")
        for value, length in zip(values, lengths, strict=True):
            if tuple(value.shape) != (1, 1, length, 10240) or value.dtype != torch.bfloat16:
                raise ValueError(f"Invalid reference {key} shape/dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite reference {key}")
    states = saved.get("states", [])
    if len(states) != 4 or any(set(s) != set(state_names(a.layer)) for s in states):
        raise ValueError("Reference must include all state buffers after every step")
    if len(saved.get("token_ids", [])) != 4:
        raise ValueError("Missing persisted token operands")
    position = 0
    for ids, length in zip(saved["token_ids"], lengths, strict=True):
        expected = (torch.arange(position, position + length).reshape(1, -1) + 123) % 248320
        if not torch.equal(ids, expected):
            raise ValueError("Reference token IDs do not preserve global order")
        position += length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "mesh"), required=True)
    parser.add_argument("--layer", type=int, choices=(0, 1, 3), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=32769)
    parser.add_argument("--capacity", type=int, default=32896)
    parser.add_argument(
        "--reserve-bytes",
        type=int,
        default=0,
        help="Mesh resident DRAM target before IO; full-stack candidate: 31203443520",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(
            "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa"
        ),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide")
    )
    a = parser.parse_args()
    if not 128 <= a.capacity <= 262144 or a.capacity % 128 or not 1 <= a.length <= a.capacity - 3:
        parser.error("Capacity must be a multiple of 128 in 128..262144 and cover length plus three decodes")
    if a.reserve_bytes < 0 or (a.reserve_bytes and a.mode != "mesh"):
        parser.error("DRAM reservation is nonnegative and mesh-only")
    if a.reference.resolve() == a.output.resolve():
        parser.error("Reference and JSON output must be different paths")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(2071 + a.layer)
    multi = a.mode == "mesh"
    if multi:
        saved = torch.load(a.reference, weights_only=False, map_location="cpu")
        validate_reference(a, saved)
    else:
        saved = dict(metadata=metadata(a), inputs=[], outputs=[], states=[], token_ids=[])
        position = 0
        for length in (a.length, 1, 1, 1):
            host = torch.empty(1, 1, length, 10240, dtype=torch.bfloat16)
            for start in range(0, length, 32):
                count = min(32, length - start)
                host[:, :, start : start + count] = (torch.randn(1, 1, count, 10240) * 0.02).bfloat16()
            saved["inputs"].append(host)
            saved["token_ids"].append((torch.arange(position, position + length).reshape(1, -1) + 123) % 248320)
            position += length
    report = dict(
        metadata=metadata(a),
        mode=a.mode,
        passed=False,
        rows=[],
        runtime_audit="Every public forward; transfers outside audit",
        baseline_streamed=not multi,
        mesh_public_prefill_calls=1 if multi else 0,
        timing_scope="Correctness wall time including transfers; not a baseline/mesh speedup",
    )
    mesh = None
    try:
        cfg = AutoConfig.from_pretrained(a.snapshot, local_files_only=True).text_config
        if a.layer == 1:
            reference, weights = load_ple_reference(a.snapshot, cfg)
            del reference
        else:
            weights = load_weights(a.snapshot, a.layer, "")
        if multi:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4 if multi else 1), trace_region_size=0, l1_small_size=32768)
        embedding = None
        if a.layer == 1:
            embedding = (
                MeshPackedEmbedding.from_cache(a.cache_dir, mesh, Collectives(mesh))
                if multi
                else PackedEmbedding.from_checkpoint(a.snapshot, cache_dir=a.cache_dir, mesh_device=mesh)
            )
        model = (MultichipDecoder if multi else OptimizedDecoder).from_state_dict(
            weights,
            hf_config=cfg,
            layer_idx=a.layer,
            mesh_device=mesh,
            packed_embedding=embedding,
            **({"expert_dtype": ttnn.bfloat4_b} if multi else {}),
        )
        page_table = torch.arange(a.capacity // 32 - 1, -1, -1, dtype=torch.int32).reshape(1, -1)
        state = model.create_state(capacity=a.capacity, page_table=page_table)

        def memory():
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            return dict(
                allocated_bytes=view.total_bytes_allocated_per_bank * view.num_banks,
                free_bytes=view.total_bytes_free_per_bank * view.num_banks,
                banks=view.num_banks,
            )

        report["memory_model_and_state"] = memory()
        reservation = []
        remaining = max(0, a.reserve_bytes - memory()["allocated_bytes"]) if a.reserve_bytes else 0
        while remaining > 0:
            size = min(256 * 1024**2, ((remaining + 2047) // 2048) * 2048)
            reservation.append(
                ttnn.empty(
                    (1, 1, 32, size // 64),
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            remaining -= size
        report["reservation"] = dict(
            target_bytes=a.reserve_bytes,
            buffers=len(reservation),
            actual=memory(),
            claim="DRAM reservation, not an instantiated full model",
        )
        mapper = ttnn.ShardTensorToMesh(mesh, dim=3) if multi else ttnn.ReplicateTensorToMesh(mesh)
        output_composer = ttnn.ConcatMeshToTensor(mesh, dim=3) if multi else None
        state_composer = ttnn.ConcatMeshToTensor(mesh, dim=0) if multi else None
        owner = state.attention if state.attention is not None else state

        def snapshot():
            return {
                name: ttnn.to_torch(
                    getattr(owner, name), **({"mesh_composer": state_composer} if multi else {})
                ).clone()
                for name in state_names(a.layer)
            }

        def forward(host, ids, prefill):
            # The baseline never has the whole long input, token table or output on device.
            x = ttnn.from_torch(
                host.contiguous(), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            tokens = (
                ttnn.from_torch(token_rows(ids), device=mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                if a.layer == 1
                else None
            )
            with runtime_audit():
                y = (model.prefill_forward if prefill else model.decode_forward)(x, state=state, token_ids=tokens)
            result = ttnn.to_torch(y, **({"mesh_composer": output_composer} if multi else {})).clone()
            # Readback is blocking; release all per-call device buffers before the next baseline chunk.
            del y, x, tokens
            return result

        for step, host in enumerate(saved["inputs"]):
            start_time = time.perf_counter()
            print(f"Starting {a.mode} step {step}, logical rows={host.shape[-2]}", flush=True)
            if step == 0 and not multi:
                actual = torch.empty_like(host)
                for start in range(0, a.length, 32):
                    end = min(start + 32, a.length)
                    actual[:, :, start:end] = forward(
                        host[:, :, start:end], saved["token_ids"][step][:, start:end], True
                    )
                    if start % 4096 == 0:
                        print(f"Baseline prefill consumed {end}/{a.length} logical rows", flush=True)
            else:
                actual = forward(host, saved["token_ids"][step], step == 0)
            observed = snapshot()
            row = dict(
                step=step,
                logical_position=a.length + step,
                wall_seconds=time.perf_counter() - start_time,
                public_forward_calls=math.ceil(a.length / 32) if step == 0 and not multi else 1,
                output=whole_metric(saved["outputs"][step] if multi else actual, actual),
                state_parity={},
            )
            for name, value in observed.items():
                expected = mesh_state_reference(name, saved["states"][step][name]) if multi else value
                row["state_parity"][name] = whole_metric(expected, value)
            exact_contract = {}
            if a.layer == 3:
                exact_contract["position"] = bool((observed["position"] == a.length + step).all())
                expected_pages = torch.cat([page_table] * (4 if multi else 1), dim=0)
                exact_contract["page_table"] = torch.equal(observed["page_table"], expected_pages)
            if a.layer == 1:
                from models.autoports.qwen_qwen3_8_flash_next.tt.ngram_hash import EOS

                consumed = torch.cat(saved["token_ids"][: step + 1], dim=1).flatten()
                history = torch.zeros(1, 16, dtype=torch.int64)
                history[0, :2] = torch.tensor([EOS, int(consumed[-1])]) if consumed.numel() == 1 else consumed[-2:]
                exact_contract["token_history"] = torch.equal(
                    observed["token_history"].long(), history.repeat(4 if multi else 1, 1)
                )
            row["exact_contract"] = exact_contract
            row["passed"] = row["output"]["finite"] and row["output"]["pcc"] >= 0.995 and all(exact_contract.values())
            row["passed"] &= all(
                m["finite"]
                and (m["exact"] if name in ("position", "page_table", "token_history") else m["pcc"] >= 0.995)
                for name, m in row["state_parity"].items()
            )
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            if not multi:
                saved["outputs"].append(actual)
                saved["states"].append(observed)
            del actual, observed
        report["memory_after_steps"] = memory()
        report["passed"] = len(report["rows"]) == 4 and all(row["passed"] for row in report["rows"])
        if not multi and report["passed"]:
            a.reference.parent.mkdir(parents=True, exist_ok=True)
            torch.save(saved, a.reference)
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    assert report["passed"], report


if __name__ == "__main__":
    main()
