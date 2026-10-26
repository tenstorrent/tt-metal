# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the KV pair split's decode merge region (PairSplit.merge_heads, pair_merge_common.py) on Galaxy
row 0, after the global layer's split decode end to end (tests/test_kv_pair_split_tp8.py).

For 4, 8 and 32 decode users, eager and replayed from a trace, against the fp64 reference (alpha * o, heads
concatenated): every value finite, rows with no key on a chip exactly 0 there, every other value within one bf16 ulp
of the reference (|got - want| <= 2^-8 |want| + 1e-6: alpha in fp32, one bf16 rounding of the product), PCC >= 0.99999
per chip. The pytest of the split's merge, layer and compact-verify tests runs first and must pass. Prints
PAIR_MERGE_CORRECTNESS_PASS when all hold.
"""

import argparse
import os
import subprocess
import sys

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.common import pcc
from models.experimental.mimo_v2_6_flash.tests.perf import pair_merge_common as pm

ULP = 2.0**-8


def check(got, want, o, users, tag):
    assert torch.isfinite(got).all(), f"{tag}: non-finite output"
    zero_rows = o.reshape(pm.TP, users, -1).abs().sum(-1) == 0  # [TP, users]: no key on that chip, or idle
    assert torch.equal(got[zero_rows], torch.zeros_like(got[zero_rows])), f"{tag}: a no-key row is not 0"
    err = (got.double() - want).abs()
    bound = ULP * want.abs() + 1e-6
    bad = (err > bound).sum().item()
    assert bad == 0, f"{tag}: {bad} values off by more than one bf16 ulp (worst {(err - bound).max().item():.3g})"
    worst = min(pcc(got[c].double(), want[c]) for c in range(pm.TP))
    assert worst >= 0.99999, f"{tag}: PCC {worst:.6f}"
    print(f"{tag}: ok, worst PCC {worst:.7f}, max err {err.max().item():.3g}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.parse_args()
    # The layer tests first: this process holds the chips from its first open until it exits.
    test = "models/experimental/mimo_v2_6_flash/tests/test_kv_pair_split_tp8.py"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test, "-k", "merge or layer or compact"]
    result = subprocess.run(cmd, env=dict(os.environ), timeout=1500)
    assert result.returncode == 0, f"{test} failed (exit {result.returncode})"
    mesh, row = pm.open_row(trace_region_size=16 * 1024 * 1024)
    try:
        plan, pair, concat = pm.build(row)
        for users in (4, 8, 32):
            o, lse = pm.host_inputs(users, seed=100 + users)
            want = pm.reference(o, lse)
            o_tt, lse_tt = pm.device_inputs(plan, o, lse)
            out = pm.run(pair, concat, o_tt, lse_tt)
            check(pm.to_host(out, users), want, o, users, f"users {users} eager")
            out.deallocate(True)
            ttnn.synchronize_device(row)
            trace_id = ttnn.begin_trace_capture(row, cq_id=0)
            out = pm.run(pair, concat, o_tt, lse_tt)
            ttnn.end_trace_capture(row, trace_id, cq_id=0)
            for _ in range(2):
                ttnn.execute_trace(row, trace_id, cq_id=0, blocking=True)
            check(pm.to_host(out, users), want, o, users, f"users {users} traced")
            ttnn.release_trace(row, trace_id)
            for t in (out, o_tt, lse_tt):
                t.deallocate(True)
    finally:
        pm.close(mesh, row)
    print("PAIR_MERGE_CORRECTNESS_PASS", flush=True)


if __name__ == "__main__":
    main()
