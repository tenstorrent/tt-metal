# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Traced replay on fresh inputs: every bench timing and bench PCC goes through ``TracedForward``.

Trace keys (device rows, variate mode, pad cut P; R1-1), all prepared together and captured on their input A,
then each replayed on B and C (same key, different data) and on A again, twice round; every replay must equal the
eager forward on that input bit for bit. Keys:

    cut192      32 rows x 2048 (P = 192)
    cut128      32 rows x 4096 (P = 128)
    full        32 rows x 8192 (P = 0, full path)
    variate     32 rows in groups; C has ragged groups (5, 3, 8, 8, 6 rows) with its own mask
    paper1024   the 1024-row bench key (P = 192); B, C are permuted / negated rows

Both precisions (``accuracy`` and ``performance``). Run it with ``TT_METAL_TRACE_ALLOC_TRACKING=1`` (set before
ttnn is imported) to have ``ttnn.execute_trace`` refuse replay when a buffer allocated after a capture is alive.
"""

from __future__ import annotations

import os

import pytest
import torch

from models.experimental.tirex2_forecast.tests.golden.goldens import golden, have_goldens
from models.experimental.tirex2_forecast.tests.pcc.helpers import host_weights

pytestmark = pytest.mark.skipif(not have_goldens(), reason="goldens not generated")

TRACE_REGION = 6 * 1024**3  # four 32-row traces, a variate trace and the 1024-row trace


def _inputs():
    """``{key name: [(label, context, group_ids, target_mask), ...]}``, input A first."""
    m16, paper, mv4, mvb = golden("model16"), golden("bench_paper"), golden("multivar4"), golden("bench_multivariate")
    lc = golden("bench_long_context")["context"]

    def groups(ctx3):
        g, per, T = ctx3.shape
        return ctx3.reshape(-1, T), torch.arange(g).repeat_interleave(per), (torch.arange(g * per) % per) == 0

    sizes = [5, 3, 8, 8, 6]
    gid = torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes))
    first = torch.cat([torch.tensor([True] + [False] * (n - 1)) for n in sizes])
    ragged = mvb["context"][:4].reshape(-1, mvb["context"].shape[-1])[: sum(sizes)]
    perm = torch.randperm(1024, generator=torch.Generator().manual_seed(0))
    return {
        "cut192": [
            ("A model16 + paper 0..15", torch.cat([m16["context"], paper["context"][:16]]), None, None),
            ("B paper rows 16..47", paper["context"][16:48], None, None),
            ("C paper rows 48..79", paper["context"][48:80], None, None),
        ],
        "cut128": [
            ("A long_context 0..31 last 4096", lc[:, -4096:], None, None),
            ("B long_context 0..31 last 4096 negated", -lc[:, -4096:], None, None),
            ("C long_context 0..31 last 4096, other half first", lc[:, -4096:].roll(16, 0), None, None),
        ],
        "full": [
            ("A long_context", lc, None, None),
            ("B long_context negated", -lc, None, None),
            ("C long_context rolled", lc.roll(16, 0), None, None),
        ],
        "variate": [
            ("A multivar4", *groups(mv4["context"])),
            ("B multivariate groups 0..3", *groups(mvb["context"][:4])),
            ("C ragged groups 5/3/8/8/6", ragged, gid, first),
        ],
        "paper1024": [
            ("A paper", paper["context"], None, None),
            ("B paper permuted", paper["context"][perm], None, None),
            ("C paper negated", -paper["context"], None, None),
        ],
    }


EXPECTED_CUT = {"cut192": 192, "cut128": 128, "full": 0, "variate": 0, "paper1024": 192}


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
def test_trace_replays_fresh_inputs(precision):
    import ttnn

    from models.experimental.tirex2_forecast.tt.common import Precision
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2
    from models.experimental.tirex2_forecast.tt.trace import TracedForward

    tracking = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    print(f"\n[TRACE] TT_METAL_TRACE_ALLOC_TRACKING={'1' if tracking else 'unset'}")
    device = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=TRACE_REGION)
    try:
        tt = TtTiRex2(device, host_weights(), Precision.by_name(precision), variate=True)
        cases = {
            name: [(lbl, tt.prepare_inputs(c, 64, g, t)) for lbl, c, g, t in lst] for name, lst in _inputs().items()
        }
        for name, lst in cases.items():
            assert len({TracedForward.key(p) for _, p in lst}) == 1, f"{name}: inputs must share one trace key"
            assert lst[0][1].cut == EXPECTED_CUT[name], f"{name}: pad cut {lst[0][1].cut}"
        # eager references first: nothing may be allocated after a capture
        eager = {}
        for name, lst in cases.items():
            for lbl, prep in lst:
                inputs = tt.upload_inputs(prep)
                out = tt.forward_device(inputs)
                eager[lbl] = ttnn.to_torch(out).float()
                ttnn.deallocate(out)
                tt.deallocate_inputs(inputs)
        assert not torch.equal(eager["A paper"], eager["B paper permuted"])
        traced = TracedForward(tt)
        traced.prepare([lst[0][1] for lst in cases.values()])  # capture both keys on their input A
        for _ in range(2):
            for name, lst in cases.items():
                for lbl, prep in [*lst[1:], lst[0]]:  # B, C, then A again
                    got = ttnn.to_torch(traced(prep)).float()
                    diff = float((got - eager[lbl]).abs().max())
                    print(f"[TRACE] {name} / {lbl}: max |replay - eager| {diff:.3e}")
                    assert torch.equal(got, eager[lbl]), f"{lbl}: traced replay differs from eager by {diff}"
        traced.release()
        tt.release()
    finally:
        ttnn.close_device(device)
