# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture real B1/B2 MoE inputs for the existing reduction fixture consumer.

Serialized device-owner invocation, using the accepted lane/mesh environment::

  timeout 900 python_env/bin/python -m \
    models.autoports.qwen_qwen3_8_flash_next.tests.capture_moe_decode_geometry \
    --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/moe_decode_capture_v1.json

Loads reduced real layers0/1/3 once. Each request starts from the same prompt,
then receives forced42/43 with independent solo and B2 histories. Diagnostic
eager dispatch permits CPU snapshots between real native operations. It uses
the same model, prefill chunk body and decode body, with unchanged precision;
it is neither traced acceptance nor performance evidence. No production edit.
Exports --fixture-compatible B1-paired and B2 records for every layer/token.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import torch

from .probe_moe_reduction_rows import validate_fixture
from .probe_shared_pool_generator import decode_geometry_comparison, tensor_comparison


def owned_values(record):
    ids = record["ids"].to(torch.int32)
    return torch.stack(
        [
            torch.where(
                ((ids >= rank * 128) & (ids < (rank + 1) * 128)).unsqueeze(-1),
                record["combined"][rank],
                0,
            )
            for rank in range(4)
        ]
    )


def pair_solo(left, right):
    return {key: torch.cat([left[key], right[key]], dim=0 if key in ("weights", "ids") else 1) for key in left}


def export_fixture(record, provenance):
    fixture = {key: record[key] for key in ("combined", "weights", "ids")}
    fixture["provenance"] = provenance
    validate_fixture(fixture)
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixture-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/reference/moe_decode_geometry")
    )
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {"completed": False, "scope": "eager real reduced MoE capture; not acceptance/performance", "fixtures": []}
    mesh = gen = None
    started = time.monotonic()
    capture, active = {}, {}
    try:
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore the accepted mesh descriptor after lane-env"
        import ttnn

        from ..tt.generator import build_generator
        from ..tt.multichip_decoder import MeshMoe
        from .probe_full_model_serving import packed

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        options = dict(layer_indices=(0, 1, 3), max_seq_len=128, max_batch_size=2)
        if args.checkpoint:
            options["checkpoint"] = args.checkpoint
        if args.packed_cache:
            options["packed_cache"] = args.packed_cache
        gen = build_generator(Path(__file__).resolve().parents[1], mesh, **options)
        gen.prepare()
        layer_ids = {id(layer.mlp): layer.layer_idx for layer in gen.model.layers}
        prompts = ([785, 13458, 13], [785, 13, 13458, 13, 785] * 6 + [785, 13, 13458])
        tables = torch.arange(4, dtype=torch.int32).reshape(1, 4).repeat(2, 1)

        def read(value, *shape):
            return torch.stack([ttnn.to_torch(rank).reshape(*shape).clone() for rank in ttnn.get_device_tensors(value)])

        original_call, original_reduce, original_shared = MeshMoe.__call__, MeshMoe._reduce, MeshMoe.shared_forward

        def entry(moe):
            return capture[active["label"]][layer_ids[id(moe)]]

        def call(moe, x):
            if not active:
                return original_call(moe, x)
            n = x.shape[-2]
            capture[active["label"]][layer_ids[id(moe)]] = {"input": read(x, n, 2560)}
            result = original_call(moe, x)
            entry(moe)["output"] = read(result, n, 2560)
            return result

        def reduce(moe, combined, weights, ids, n):
            if not active:
                return original_reduce(moe, combined, weights, ids, n)
            record = entry(moe)
            record["combined"] = read(combined, n, 10, 2560)
            all_weights, all_ids = read(weights, n, 10), read(ids, n, 10).to(torch.int32)
            assert all(torch.equal(all_weights[0], row) for row in all_weights), "Weights differ across ranks"
            assert all(torch.equal(all_ids[0], row) for row in all_ids), "Expert IDs differ across ranks"
            record["weights"], record["ids"] = all_weights[0].clone(), all_ids[0].clone()
            result = original_reduce(moe, combined, weights, ids, n)
            record["reduced"] = read(result, n, 2560)
            return result

        def shared(moe, x):
            result = original_shared(moe, x)
            if active:
                entry(moe)["shared"] = read(result, x.shape[-2], 2560)
            return result

        def eager_prefill_chunk(count, slot):
            gen._prefill_body(count, slot)

        with (
            patch.object(MeshMoe, "__call__", call),
            patch.object(MeshMoe, "_reduce", reduce),
            patch.object(MeshMoe, "shared_forward", shared),
            patch.object(gen, "_prefill_chunk", eager_prefill_chunk),
        ):
            report["prefill"], report["decode"] = {}, {}
            for name, rows in (("solo0", [prompts[0]]), ("solo1", [prompts[1]]), ("batch", prompts)):
                gen.reset()
                slots = tuple(range(len(rows)))
                pref = gen.prefill_forward(
                    packed(rows),
                    page_table=tables[: len(rows)],
                    kv_cache=gen.caches[: len(rows)],
                    prompt_lens=list(map(len, rows)),
                    slots=slots,
                )
                report["prefill"][name] = pref
                for step, token in enumerate((42, 43)):
                    label = f"{name}_token{token}"
                    capture[label] = {}
                    positions = [len(row) + step for row in rows]
                    plan = gen._table_plan(
                        tables[: len(rows)], gen.caches[: len(rows)], slots, [p + 1 for p in positions]
                    )
                    gen._bind(plan, gen.caches[: len(rows)], slots, positions)
                    words = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                    words[..., : len(rows)] = token
                    gen._refresh(gen.decode_tokens, words)
                    active["label"] = label
                    try:
                        gen._decode_body(slots)
                    finally:
                        active.clear()
                    for cache in gen.caches[: len(rows)]:
                        cache.position += 1
                    report["decode"][label] = gen._read_logits(gen.sample_logits)[: len(rows)].clone()
                    print("CAPTURED_MOE", label, flush=True)

        assert not gen.traces and not gen.prefill_traces and not gen.sampling_traces, "Diagnostic unexpectedly captured"
        args.fixture_dir.mkdir(parents=True, exist_ok=True)
        report["source_sha256"] = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "tt/multichip_decoder.py").read_bytes()
        ).hexdigest()
        report["matched_prefill"] = tensor_comparison(
            torch.cat([report["prefill"]["solo0"], report["prefill"]["solo1"]]), report["prefill"]["batch"]
        )
        report["decode_cross_geometry"] = {
            str(token): decode_geometry_comparison(
                torch.cat([report["decode"][f"solo0_token{token}"], report["decode"][f"solo1_token{token}"]]),
                report["decode"][f"batch_token{token}"],
            )
            for token in (42, 43)
        }
        full_capture = args.fixture_dir / f"{args.output.stem}_capture.pt"
        torch.save({"moe": capture, "prefill": report.pop("prefill"), "decode": report.pop("decode")}, full_capture)
        report["capture_file"] = str(full_capture)
        report["comparisons"] = []
        for token in (42, 43):
            for layer in (0, 1, 3):
                solo = pair_solo(capture[f"solo0_token{token}"][layer], capture[f"solo1_token{token}"][layer])
                batch = capture[f"batch_token{token}"][layer]
                stages = {}
                for key in ("input", "ids", "weights", "combined_owned", "reduced", "shared", "output"):
                    left, right = (
                        (owned_values(solo), owned_values(batch))
                        if key == "combined_owned"
                        else (solo[key], batch[key])
                    )
                    stages[key] = tensor_comparison(left, right)
                report["comparisons"].append(
                    {
                        "token": token,
                        "layer": layer,
                        "stages": stages,
                        "first_nonexact_stage": next((k for k, v in stages.items() if not v["exact"]), None),
                    }
                )
                for kind, record in (("solo_paired", solo), ("batch", batch)):
                    provenance = {
                        "kind": "real_decode_pre_reduction",
                        "source_sha256": report["source_sha256"],
                        "token": token,
                        "layer": layer,
                        "geometry": kind,
                        "prompt_lengths": [3, 33],
                        "forced_history": [42] if token == 43 else [],
                        "capture_file": str(full_capture),
                    }
                    path = args.fixture_dir / f"{args.output.stem}_layer{layer}_token{token}_{kind}.pt"
                    torch.save(export_fixture(record, provenance), path)
                    report["fixtures"].append(
                        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **provenance}
                    )
        assert report["matched_prefill"]["exact"], "Diagnostic prompt prefixes differ before forced decode"
        report["completed"] = True
    except BaseException:
        # Avoid serializing partial CPU tensors into the JSON failure record.
        report.pop("prefill", None)
        report.pop("decode", None)
        report["error"] = traceback.format_exc()
        raise
    finally:
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
