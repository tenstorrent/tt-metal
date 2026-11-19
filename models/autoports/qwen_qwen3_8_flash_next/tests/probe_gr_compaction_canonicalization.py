# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Enumerate legacy GR compaction BF16 transformations; no native candidate runs."""

import argparse
import hashlib
import json
import math
import os
import traceback
from collections import Counter
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_compact_gr_down import baseline, classify, read_ranks, upload
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit


def input_from_bits(bits, count):
    """Place 320 logical columns into four80/96 blocks, poisoning skipped lanes."""
    values = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, count, 320)
    source = torch.full((1, 1, count, 384), 123.0, dtype=torch.bfloat16)
    for rank in range(4):
        source[..., rank * 96 : rank * 96 + 80] = values[..., rank * 80 : rank * 80 + 80]
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), f"Refusing to overwrite evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    repo = root.parents[2]
    sources = [
        Path(__file__),
        root / "tests/probe_compact_gr_down.py",
        root / "tests/probe_linear_contract.py",
        repo / "ttnn/core/tensor/py_to_tt_tensor.cpp",
        repo / "ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp",
        repo / "ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp",
    ]
    report = {
        "passed": False,
        "scope": "Diagnostic only: existing four-slice/concat BF16 mapping. No native candidate, trace, speedup or model acceptance.",
        "mesh": [1, 4],
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("TT_", "TTNN_"))},
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "upload": "preserve_nan_values=True; all logical source bits checked on every rank before baseline",
        "source_observation": "concat.cpp build_untilize_rm_retilize_concat detects logical80 versus padded96, uses untilize_with_unpadding then tilize_with_val_padding. The observed mapping, not an assumed pack conversion rule, determines canonicalization.",
        "q32_passes": [],
        "q1_controls": [],
        "inconsistent_mappings": [],
    }
    mesh = None
    mapping = torch.full((65536,), -1, dtype=torch.int32)
    counts = torch.zeros((65536,), dtype=torch.int32)

    def save_mapping():
        known = mapping >= 0
        changed = known & (mapping != torch.arange(65536))
        ids = changed.nonzero().flatten().tolist()
        report["unique_patterns_observed"] = int(known.sum())
        report["mapping_sha256_int32_le"] = hashlib.sha256(mapping.numpy().tobytes()).hexdigest()
        report["exceptions"] = [
            {
                "input": f"0x{i:04x}",
                "output": f"0x{int(mapping[i]):04x}",
                "input_class": classify(i),
                "output_class": classify(int(mapping[i])),
            }
            for i in ids
        ]
        report["exception_histogram"] = dict(Counter(f"{classify(i)} -> {classify(int(mapping[i]))}" for i in ids))
        report["coverage_count_histogram"] = {str(k): v for k, v in Counter(counts.tolist()).items()}

    def run(bits_by_rank, count, label, destination, *, require_existing):
        report["last_phase"] = label
        print(f"GR_CANON_PHASE {label}", flush=True)
        hosts = [input_from_bits(bits, count) for bits in bits_by_rank]
        value = upload([(host,) for host in hosts], mesh)
        loaded = read_ranks(value)
        row = {"label": label, "count": count, "ranks": []}
        destination.append(row)
        for rank, (actual, expected) in enumerate(zip(loaded, hosts, strict=True)):
            bad = actual.view(torch.int16) != expected.view(torch.int16)
            row["ranks"].append({"rank": rank, "upload_mismatches": int(bad.sum())})
        assert all(r["upload_mismatches"] == 0 for r in row["ranks"]), row
        with runtime_audit():
            output = baseline(value)
        actual_ranks = read_ranks(output)
        for rank, (bits, actual) in enumerate(zip(bits_by_rank, actual_ranks, strict=True)):
            observed = actual.view(torch.int16).flatten().to(torch.int32) & 0xFFFF
            assert observed.numel() == bits.numel()
            changed = 0
            for source, target in zip(bits.tolist(), observed.tolist(), strict=True):
                prior = int(mapping[source])
                if require_existing:
                    assert prior >= 0
                if prior >= 0 and prior != target:
                    report["inconsistent_mappings"].append(
                        {
                            "phase": label,
                            "rank": rank,
                            "input": f"0x{source:04x}",
                            "prior_output": f"0x{prior:04x}",
                            "output": f"0x{target:04x}",
                        }
                    )
                elif prior < 0:
                    mapping[source] = target
                counts[source] += 1
                changed += source != target
            row["ranks"][rank]["output_transformations"] = changed
        save_mapping()
        assert not report["inconsistent_mappings"], report["inconsistent_mappings"][:16]
        del output, value

    try:
        report["last_phase"] = "open_mesh"
        print("GR_CANON_PHASE open_mesh", flush=True)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        for batch in range(2):
            bits = [
                (torch.arange(32 * 320, dtype=torch.int32) + (batch * 4 + rank) * 32 * 320) % 65536 for rank in range(4)
            ]
            run(bits, 32, f"q32:pass{batch}", report["q32_passes"], require_existing=False)
        assert bool((mapping >= 0).all())
        # Recheck every observed exception plus all exponent-zero and exponent255
        # patterns and boundaries on EVERY rank at Q1. No assumed NaN rule.
        special = set((mapping != torch.arange(65536)).nonzero().flatten().tolist())
        special.update(i for i in range(65536) if (i & 0x7F80) in (0, 0x7F80))
        special.update((0x0080, 0x8080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80))
        special = sorted(special)
        report["q1_unique_controls_per_rank"] = len(special)
        for batch in range(math.ceil(len(special) / 320)):
            bits = torch.tensor([special[(batch * 320 + i) % len(special)] for i in range(320)], dtype=torch.int32)
            run(
                [bits.roll(rank) for rank in range(4)],
                1,
                f"q1:control{batch}",
                report["q1_controls"],
                require_existing=True,
            )
        report["passed"] = True
        report["last_phase"] = "passed"
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        save_mapping()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
