# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy-profileable perf harness for the Gaze-LLE TT-NN forward."""

import pytest
import torch

from models.experimental.gaze_lle.reference.torch_gaze_lle import build_gaze_lle
from models.experimental.gaze_lle.tt.tt_gaze_lle import TtGazeLLE


@pytest.mark.parametrize("variant", ["vitb14"])
def test_gaze_lle_forward_perf(device, variant):
    """Single-iteration forward suitable for Tracy profiling.

    Tracy captures every ttnn op issued inside ``tt_model(image, bboxes)``.
    Run with::

        python3 -m tracy --no-runtime-analysis --collect-noc-traces \
            --profiler-capture-perf-counters=all support-count=10000 -v -r -o \
            -m pytest models/experimental/gaze_lle/tests/test_perf.py

    After one dispatch for program caching we do a steady-state captured pass.
    """
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    ref = build_gaze_lle(variant=variant, inout=True).eval()
    tt_model = TtGazeLLE(ref, device, inout=True)

    image = torch.randn(1, 3, 448, 448)
    bboxes = [(0.3, 0.2, 0.6, 0.5)]

    # Warm-up pass: populate program cache + any lazy allocations.
    _ = tt_model(image, bboxes)

    # Captured pass.
    import ttnn

    if hasattr(ttnn, "synchronize_device"):
        ttnn.synchronize_device(device)
    out = tt_model(image, bboxes)
    if hasattr(ttnn, "synchronize_device"):
        ttnn.synchronize_device(device)

    assert out["heatmap"].shape == (1, 64, 64)
    if out["inout"] is not None:
        assert out["inout"].shape == (1,)
