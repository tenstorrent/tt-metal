# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Traced replay on fresh inputs: every bench timing and bench PCC goes through ``TracedForward``.

Two trace keys (32 univariate rows; 32 rows with variate attention) are prepared together and captured on input
A; each is then replayed on B and C (same key, different data and, for the variate key, its own mask) and on A
again, and every replay must equal the eager forward on that input bit for bit. Run it with
``TT_METAL_TRACE_ALLOC_TRACKING=1`` (set before ttnn is imported) to have ``ttnn.execute_trace`` refuse replay
when a buffer allocated after a capture is still alive.
"""

from __future__ import annotations

import os

import pytest
import torch

from models.experimental.tirex2_forecast.tests.golden.goldens import golden, have_goldens
from models.experimental.tirex2_forecast.tests.pcc.helpers import host_weights

pytestmark = pytest.mark.skipif(not have_goldens(), reason="goldens not generated")

TRACE_REGION = 3 * 1024**3  # two 32-row traces need more than 1 GB


def _inputs():
    """``{key name: [(label, context, group_ids, target_mask), ...]}``, input A first."""
    m16, paper, mv4, mvb = golden("model16"), golden("bench_paper"), golden("multivar4"), golden("bench_multivariate")

    def groups(ctx3):
        g, per, T = ctx3.shape
        return ctx3.reshape(-1, T), torch.arange(g).repeat_interleave(per), (torch.arange(g * per) % per) == 0

    return {
        "univariate": [
            ("A model16", m16["context"], None, None),
            ("B paper rows 0..15", paper["context"][:16], None, None),
            ("C paper rows 16..31", paper["context"][16:32], None, None),
        ],
        "variate": [
            ("A multivar4", *groups(mv4["context"])),
            ("B multivariate groups 0..3", *groups(mvb["context"][:4])),
            ("C multivariate groups 4..7", *groups(mvb["context"][4:8])),
        ],
    }


def test_trace_replays_fresh_inputs():
    import ttnn

    from models.experimental.tirex2_forecast.tt.common import Precision
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2
    from models.experimental.tirex2_forecast.tt.trace import TracedForward

    tracking = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    print(f"\n[TRACE] TT_METAL_TRACE_ALLOC_TRACKING={'1' if tracking else 'unset'}")
    device = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=TRACE_REGION)
    try:
        tt = TtTiRex2(device, host_weights(), Precision.accuracy(), variate=True)
        cases = {
            name: [(lbl, tt.prepare_inputs(c, 64, g, t)) for lbl, c, g, t in lst] for name, lst in _inputs().items()
        }
        for name, lst in cases.items():
            assert len({TracedForward.key(p) for _, p in lst}) == 1, f"{name}: inputs must share one trace key"
        # eager references first: nothing may be allocated after a capture
        eager = {}
        for name, lst in cases.items():
            for lbl, prep in lst:
                inputs = tt.upload_inputs(prep)
                out = tt.forward_device(inputs)
                eager[lbl] = ttnn.to_torch(out).float()
                ttnn.deallocate(out)
                tt.deallocate_inputs(inputs)
        assert not torch.equal(eager["A model16"], eager["B paper rows 0..15"])
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
