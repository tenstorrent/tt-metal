# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""All-token maximum-context TP4 vs optimized TTNN; sampled output/cache oracle."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reserve-gib", type=int, default=0)
    p.add_argument("--policy-file", type=Path)
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(792)
    root = Path(__file__).resolve().parents[1]
    cfg = Qwen3_5TextConfig(**json.loads((root / "doc/hf_config.json").read_text())["text_config"])
    weights = load_layer(a.layer)
    x = torch.randn(1, a.length, 5120, dtype=torch.bfloat16) * 0.1
    d = torch.randn(1, 1, 5120, dtype=torch.bfloat16) * 0.1
    table = torch.randperm((a.length + 32) // 32).reshape(1, -1)
    rows = sorted(set([0, 1, 31, 32, 511, 512, a.length // 2, *range(a.length - 32, a.length)]))
    rows = [i for i in rows if 0 <= i < a.length]
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    metadata = dict(
        layer=a.layer,
        length=a.length,
        seed=792,
        baseline_sha256=hashlib.sha256((root / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
    )
    if not a.baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if a.baseline else 4),
        physical_device_ids=[0] if a.baseline else [],
        trace_region_size=32 * 1024 * 1024,
    )
    try:
        layer = (OptimizedDecoder if a.baseline else MultichipDecoder).from_state_dict(
            weights,
            hf_config=cfg,
            layer_idx=a.layer,
            mesh_device=mesh,
            **({"policy": json.loads(a.policy_file.read_text())} if a.policy_file and not a.baseline else {}),
        )
        upload = layer._upload if a.baseline else layer.upload_residual
        linear = layer.kind == "linear_attention"
        state = layer.create_recurrent_state() if linear else layer.create_paged_state(table)
        allocation_samples = []

        def sample_allocation(label):
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            entry = {
                name: getattr(view, name)
                for name in [
                    "num_banks",
                    "total_bytes_per_bank",
                    "total_bytes_allocated_per_bank",
                    "total_bytes_free_per_bank",
                    "largest_contiguous_bytes_free_per_bank",
                ]
            }
            entry["label"] = label
            entry["allocated_bytes_per_device"] = entry["num_banks"] * entry["total_bytes_allocated_per_bank"]
            allocation_samples.append(entry)

        sample_allocation("layer_and_cache")
        # Resident allocation on EVERY device, additional to real layer/cache/IO.
        reserve = [
            ttnn.empty(
                (1, 1, 1024, 131072),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(a.reserve_gib * 4)
        ]
        sample_allocation("with_reserve")
        tx = upload(x)
        rope = (
            None
            if linear
            else tuple(layer._upload(t.unsqueeze(1)) for t in rotary(x, torch.arange(a.length).reshape(1, -1)))
        )
        sample_allocation("with_input_and_rope")
        completed = 0
        finish = layer._finish

        def progress(chunk, attention, **kwargs):
            nonlocal completed
            out = finish(chunk, attention, **kwargs)
            completed += chunk.shape[1]
            if completed % 8192 == 0 or completed == a.length:
                print("ENQUEUED", completed, flush=True)
                sample_allocation("finish_" + str(completed))
            return out

        layer._finish = progress

        def read(t, dim=-1):
            values = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(t)]
            return values[0] if a.baseline else torch.cat(values, dim)

        start = time.perf_counter()
        out = layer.prefill_forward(tx, state=state, rope=rope)
        ttnn.synchronize_device(mesh)
        duration = time.perf_counter() - start
        sample_allocation("after_concat")
        result = {"metadata": metadata, "prefill": torch.cat([read(out[:, i : i + 1]) for i in rows], 1)}
        pages = sorted(set(int(table[0, i // 32]) for i in rows))
        if a.length < cfg.max_position_embeddings:
            pages = sorted(set(pages + [int(table[0, a.length // 32])]))

        def snapshot():
            if not linear:
                return {
                    name: torch.cat([read(getattr(state, name)[i : i + 1], 1) for i in pages], 0)
                    for name in ("key", "value")
                }
            values = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(state.conv)]
            conv = (
                values[0]
                if a.baseline
                else torch.cat(
                    [
                        torch.cat([v[..., lo:hi] for v in values], -1)
                        for lo, hi in [(0, 512), (512, 1024), (1024, 2560)]
                    ],
                    -1,
                )
            )
            return {"recurrent": read(state.recurrent, 1), "conv": conv}

        pref_state = snapshot()
        result.update(pref_state)
        continuation_checks = None
        if not linear:
            resume = a.length - 31
            continued = layer.prefill_forward(
                tx[:, resume:], state=state, start_pos=resume, rope=tuple(t[:, :, resume:] for t in rope)
            )
            result["continuation"] = read(continued)
            continuation_pcc = pcc(read(out[:, resume:]), result["continuation"])
            assert math.isfinite(continuation_pcc) and continuation_pcc >= 0.995
            after = snapshot()
            touched = torch.zeros((len(pages), 32), dtype=torch.bool)
            for position in range(resume, a.length):
                touched[pages.index(int(table[0, position // 32])), position % 32] = True
            for name, value in after.items():
                mask = (~touched)[:, None, :, None].expand_as(value)
                assert torch.equal(pref_state[name][mask], value[mask]), "Continuation changed untouched cache rows"
                result[name + "_after_continuation"] = value
            continuation_checks = {
                "output_pcc_to_original_tail": continuation_pcc,
                "untouched_sampled_cache_rows_bit_equal": True,
                "sampled_physical_pages": pages,
            }
        trace_cases = []
        if a.length < cfg.max_position_embeddings:
            td = upload(d)
            changed = upload(-d * 0.75)
            dk = (
                {}
                if linear
                else {
                    "current_pos": ttnn.from_torch(
                        torch.tensor([a.length], dtype=torch.int32),
                        device=mesh,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                    ),
                    "rope": tuple(layer._upload(t.unsqueeze(1)) for t in rotary(d, torch.tensor([[a.length]]))),
                }
            )
            tensors = [state.conv, state.recurrent] if linear else [state.key, state.value]
            saved = [ttnn.clone(t) for t in tensors]
            original = ttnn.clone(td)

            def restore():
                for src, dst in zip(saved, tensors):
                    ttnn.copy(src, dst)

            eager = {}
            for case, value in [("original", original), ("changed_input", changed)]:
                restore()
                ttnn.copy(value, td)
                eager[case] = {"decode": read(layer.decode_forward(td, state=state, **dk)), **snapshot()}
                for name, tensor in eager[case].items():
                    result[case + "_" + name] = tensor
            restore()
            ttnn.copy(original, td)
            warm = layer.decode_forward(td, state=state, **dk)
            del warm
            restore()
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            replay = layer.decode_forward(td, state=state, **dk)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            for case, value in [("original", original), ("changed_input", changed)]:
                for repeat in range(2):
                    restore()
                    ttnn.copy(value, td)
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    actual = {"decode": read(replay), **snapshot()}
                    assert all(torch.equal(eager[case][name], tensor) for name, tensor in actual.items())
                    trace_cases.append({"case": case, "repeat": repeat, "output_and_state_bit_equal": True})
            ttnn.release_trace(mesh, tid)
        report = dict(
            metadata=metadata,
            full_decoder_all_tokens=True,
            traced_decode_bit_equal=a.length < cfg.max_position_embeddings,
            selected_rows=rows,
            prefill_seconds=duration,
            reserved_bytes_per_device=a.reserve_gib * 1024**3,
            source_sha256=hashlib.sha256(
                (root / ("tt/optimized_decoder.py" if a.baseline else "tt/multichip_decoder.py")).read_bytes()
            ).hexdigest(),
        )
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        report["selected_policy"] = dict(layer.policy)
        report["continuation_checks"] = continuation_checks
        report["trace_cases"] = trace_cases
        assert all(torch.isfinite(v).all() for k, v in result.items() if k != "metadata")
        report["allocation_samples"] = allocation_samples
        report["largest_sampled_allocation_bytes_per_device"] = max(
            v["allocated_bytes_per_device"] for v in allocation_samples
        )
        report["concat_peak_additional_bytes_estimate"] = a.length * (5120 if a.baseline else 1280) * 2
        report["dram_view"] = {
            name: getattr(view, name)
            for name in [
                "num_banks",
                "total_bytes_per_bank",
                "total_bytes_allocated_per_bank",
                "total_bytes_free_per_bank",
                "largest_contiguous_bytes_free_per_bank",
            ]
        }
        if a.baseline:
            torch.save(result, a.reference)
        else:
            ref = torch.load(a.reference, weights_only=True)
            assert ref["metadata"] == metadata
            report["pcc"] = {k: pcc(ref[k], v) for k, v in result.items() if k != "metadata"}
            report["contribution_pcc"] = [
                pcc(ref["prefill"][:, i].float() - x[:, r].float(), result["prefill"][:, i].float() - x[:, r].float())
                for i, r in enumerate(rows)
            ]
            report["pass"] = all(
                math.isfinite(v) and v >= 0.995 for v in [*report["pcc"].values(), *report["contribution_pcc"]]
            ) and all(torch.isfinite(v).all() for k, v in ref.items() if k != "metadata")
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if not a.baseline:
            assert report["pass"], report
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
