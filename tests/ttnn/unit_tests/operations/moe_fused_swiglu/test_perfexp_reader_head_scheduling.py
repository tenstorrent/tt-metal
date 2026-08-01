# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Thin pytest entry point for the `reader_head_scheduling` isolated bake-off.

Everything real lives under
ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/reader_head_scheduling/
(that tree cannot host a `test_*.py` — pytest's --import-mode=importlib would re-execute
ttnn/ttnn/__init__.py under a second dotted name and crash on duplicate op registration).

    # the mechanism bake-off on the focus shape (all six head variants):
    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_reader_head_scheduling.py \
      -k bakeoff -s

    # a predicate sweep, cases and variants from the environment:
    MOE_HEAD_CASES="7168,5120,128,bf16_rm;6144,5120,256,bf16_rm" MOE_HEAD_VARIANTS="0;3" \
      scripts/run_safe_pytest.sh --run-all <this file> -k bakeoff -s

    # the /perf-ceiling-dm transaction accounting (no device):
    scripts/run_safe_pytest.sh --run-all <this file> -k ceiling -s
"""

import os

import pytest

from ttnn.operations.moe_fused_swiglu.perf_experiments.reader_head_scheduling import head_sched_bench as B

FOCUS = "7168,5120,256,bf16_rm"
DEFAULT_VARIANTS = "0;1;2;3;4;5"


def _cases():
    out = []
    for part in os.environ.get("MOE_HEAD_CASES", FOCUS).split(";"):
        part = part.strip()
        if not part:
            continue
        emb, capacity, count, fmt = part.split(",")
        out.append((int(emb), int(capacity), int(count), fmt.strip()))
    return out


def _variants():
    return [int(v) for v in os.environ.get("MOE_HEAD_VARIANTS", DEFAULT_VARIANTS).split(";") if v.strip()]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c[3]}_e{c[0]}_c{c[1]}_n{c[2]}")
def test_bakeoff(device, case):
    """One fresh device-kernel duration per (variant, case) — /perf-measure discipline, no trial
    loop. Correctness is the only pass/fail: every variant must be BIT-IDENTICAL to variant 0."""
    import torch

    emb, capacity, count, fmt = case
    wg_head = int(os.environ.get("MOE_HEAD_WG_HEAD_ROWS", 4))
    tt_x, tt_w, tt_counts, tt_idx = B.build_inputs(device, emb, capacity, count, fmt)
    ref = B.torch_reference(tt_x, tt_w, count) if count > 0 else None

    rows, base_out, base_ns = [], None, None
    for v in _variants():
        out, ns = B.run_and_measure(device, v, tt_x, tt_w, tt_counts, tt_idx, wg_head_rows=wg_head)
        got = B.ttnn.to_torch(out).to(torch.float32)[0, 0, :count, :] if count > 0 else None
        p = B.pcc(got, ref) if count > 0 else float("nan")
        bit_same = None
        if base_out is None:
            base_out, base_ns = got, ns
        elif count > 0:
            bit_same = bool(torch.equal(got, base_out))
        rows.append((v, B.VARIANTS.get(v, f"v{v}"), ns, p, bit_same))
        # `ns` is None under `--profile` (the tracy wrapper owns the device log) — that mode is used
        # for the per-STAGE zones, and the whole-op number comes from its ops_perf_results CSV.
        d = f"{100.0 * (ns - base_ns) / base_ns:+.2f}%" if (ns is not None and base_ns) else "n/a"
        print(
            f"[head] {fmt} emb={emb} cap={capacity} count={count} v{v}={B.VARIANTS.get(v, v)} "
            f"ns={ns if ns is None else round(ns)} delta={d} pcc={p:.6f} bit_identical={bit_same}"
        )

    print(f"\n[head-table] {fmt} emb={emb} cap={capacity} count={count}")
    for v, name, ns, p, bit_same in rows:
        d = f"{100.0 * (ns - base_ns) / base_ns:+7.2f}%" if (ns is not None and base_ns) else "    n/a"
        n = "       n/a" if ns is None else f"{ns:>10.0f}"
        print(f"[head-table]   v{v:<2} {name:<20} {n} ns  {d}  pcc={p:.6f}  bit={bit_same}")

    # ---- correctness gate (the ONLY pass/fail; perf is never asserted) ----
    if count > 0:
        assert rows[0][3] > 0.97, f"baseline PCC {rows[0][3]} — the harness itself is broken"
        for v, name, _ns, p, bit_same in rows[1:]:
            assert bit_same, (
                f"variant {v} ({name}) is NOT bit-identical to the baseline (pcc {p:.6f}). "
                "The head bake-off only re-orders and re-barriers identical transactions, so a "
                "value difference is a bug in the variant, not a precision trade."
            )


def test_ceiling(device):
    """/perf-ceiling-dm accounting for the W_gate serial head — printed, never asserted."""
    banks = int(B.ttnn._ttnn.device.GetMemoryView(device, B.ttnn.BufferType.DRAM).num_banks)
    for emb in (7168, 6144):
        info = B.wgate_head_transactions(emb, num_banks=banks)
        per = info["per_core"]
        txns = sorted(t for t, _b in per.values())
        byts = sorted(b for _t, b in per.values())
        print(
            f"[ceiling] emb={emb} banks={banks} KR_PAD={info['kr_pad']} HN_PAD={info['hn_pad']} "
            f"SLOTS_H={info['slots_h']} rows_kr={info['rows_kr']} "
            f"txn_per_krow_by_column={info['cols_txn_per_krow']}"
        )
        print(
            f"[ceiling] emb={emb} W_gate per-core transactions min/median/max = "
            f"{txns[0]}/{txns[len(txns) // 2]}/{txns[-1]}  per-core bytes = "
            f"{byts[0]}/{byts[len(byts) // 2]}/{byts[-1]}"
        )
        print(
            f"[ceiling] emb={emb} W_gate GRID totals: {info['grid_txn']} transactions, "
            f"{info['grid_bytes'] / 1e6:.3f} MB, mean transaction {info['grid_bytes'] / info['grid_txn']:.0f} B"
        )
