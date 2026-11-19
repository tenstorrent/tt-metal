# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device NF4 lookup diagnostic; parent runs under a bounded timeout.

Default: compact random-weight operator test. --snapshot: real shard-0 rows.
--full: stream and retain the entire real checkpoint domain, and test all shards.
No CPU lookup or conversion occurs inside a measured/captured device forward.
This probe establishes embedding behavior, not decoder acceptance.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import NF4_VALUES, PackedEmbedding, _quantize_rows


def dequantize(weight, spec=None):
    spec = spec or {"quantization": "absmax", "group_size": 160, "scales_per_row": 1, "codebook_values": NF4_VALUES}
    packed, scales = _quantize_rows(
        weight, **{key: spec[key] for key in ("quantization", "group_size", "codebook_values")}
    )
    words = packed.to(torch.int64)
    codes = torch.stack([(words >> (4 * lane)) & 15 for lane in range(8)], dim=-1).reshape(-1, 160)
    scales = scales.reshape(-1, spec["scales_per_row"]).repeat_interleave(spec["group_size"], -1)
    return (torch.tensor(spec["codebook_values"])[codes] * scales.float()).bfloat16()


def pcc(a, b):
    return torch.corrcoef(torch.stack((a.float().flatten(), b.float().flatten())))[0, 1].item()


def selected_rows(snapshot, index, ids):
    result = {}
    groups = {}
    for row in sorted(set(ids.flatten().tolist())):
        shard, local = divmod(row, 2500012)
        groups.setdefault(shard, []).append((row, local))
    for shard, positions in groups.items():
        key = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        with safe_open(snapshot / index[key], framework="pt") as f:
            source = f.get_slice(key)
            for row, local in positions:
                result[row] = source[local : local + 1, :].clone()[0]
    return torch.stack([result[row] for row in ids.flatten().tolist()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--group-size", type=int, default=160)
    parser.add_argument("--quantization", choices=("absmax", "lsq"), default="absmax")
    parser.add_argument("--codebook-report", type=Path)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="CPU streaming cache construction only; no device import/open beyond module import",
    )
    args = parser.parse_args()
    torch.manual_seed(29)
    torch.set_num_threads(8)
    quantizer = {"quantization": args.quantization, "group_size": args.group_size}
    if args.codebook_report:
        quantizer["codebook_values"] = json.loads(args.codebook_report.read_text())["training"]["fixed_codebooks"][
            f"group{args.group_size}_affineFalse"
        ]
    if args.full and (args.snapshot is None or args.cache_dir is None):
        parser.error("--full requires --snapshot and --cache-dir")
    if args.prepare_only:
        from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import prepare_checkpoint_cache

        manifest = prepare_checkpoint_cache(args.snapshot, args.cache_dir, **quantizer)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
        return
    index = (
        json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        if args.snapshot
        else None
    )
    weight = None
    if not args.full:
        weight = (
            selected_rows(args.snapshot, index, torch.arange(4099))
            if args.snapshot
            else torch.randn(4099, 160).bfloat16() * 0.02
        )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = {"scope": "packed embedding diagnostic; not decoder acceptance", "full_domain": args.full, "cases": []}
    try:
        lookup = (
            PackedEmbedding.from_checkpoint(
                args.snapshot,
                cache_dir=args.cache_dir,
                mesh_device=mesh,
                **(
                    quantizer if args.codebook_report or args.group_size != 160 or args.quantization != "absmax" else {}
                ),
            )
            if args.full
            else PackedEmbedding.from_torch(weight, mesh_device=mesh, **quantizer)
        )
        report["quantization"] = lookup.spec

        def device(value, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, dtype=ttnn.uint32, layout=layout, device=mesh)

        large_rows = torch.tensor(
            [
                0,
                1,
                7,
                8,
                31,
                32,
                2500011,
                2500012,
                214748363,
                214748364,
                214748365,
                268435455,
                300000001,
                320001535,
                319999999,
                17,
            ]
        ).reshape(-1, 1)
        columns, stripes = lookup.word_addresses(device(large_rows))
        reference_addresses = large_rows * 20 + torch.arange(20)
        report["full_domain_addresses_exact"] = torch.equal(
            ttnn.to_torch(columns).long(), reference_addresses // 32
        ) and torch.equal(ttnn.to_torch(stripes).long(), reference_addresses % 32)
        assert report["full_domain_addresses_exact"]
        for sequence in (1, 7, 31, 32):
            ids = torch.randint(0, lookup.rows, (1, 1, sequence, 16))
            ids.flatten()[:8] = torch.tensor([0, 1, 7, 8, 31, 32, lookup.rows - 2, lookup.rows - 1])
            if args.full and sequence == 32:
                # Every shard is explicitly exercised, irrespective of RNG coverage.
                ids.flatten()[:128] = torch.arange(128) * 2500012 + 123
            source = selected_rows(args.snapshot, index, ids) if args.full else weight[ids.flatten()]
            expected = dequantize(source, lookup.spec).reshape(1, 1, sequence, 2560)
            tt_ids = device(ids, ttnn.ROW_MAJOR_LAYOUT)
            warm = lookup(tt_ids)
            ttnn.synchronize_device(mesh)
            initial = ttnn.to_torch(warm)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            output = lookup(tt_ids)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            trace_allocation_tracker.acknowledge_corruptible(output)
            ttnn.to_torch(output)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(output)
            entry = {
                "sequence": sequence,
                "eager_pcc": pcc(expected, initial),
                "traced_pcc": pcc(expected, actual),
                "traced_max_abs": (expected.float() - actual.float()).abs().max().item(),
            }
            # Refresh the same input allocation. Reuse this exact live trace.
            changed_ids = torch.flip(ids, dims=(-1,))
            host_ids = ttnn.from_torch(changed_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_ids, tt_ids)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            changed = ttnn.to_torch(output)
            changed_expected = expected.reshape(sequence, 16, 160).flip(1).reshape(expected.shape)
            entry["changed_input_pcc"] = pcc(changed_expected, changed)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            entry["repeated_replay_equal"] = torch.equal(changed, ttnn.to_torch(output))
            ttnn.release_trace(mesh, trace)
            assert min(entry["eager_pcc"], entry["traced_pcc"], entry["changed_input_pcc"]) >= 0.99999
            assert entry["repeated_replay_equal"]
            report["cases"].append(entry)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
