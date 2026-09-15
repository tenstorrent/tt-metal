# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Non-parametrized GLX smoke test for Gemma4 disaggregated prefill.

Pins the canonical production shape covered by
``text_demo_prefill.py::test_prefill_long_context_traced``
(traced, 256k context, 8k chunk, final-chunk readback, real text, 8x4 = CP8/TP4)
so a single ``pytest models/demos/gemma4_d_p/tests/test_sanity_glx.py`` run is the
fast sanity check that this path still executes end to end on a Galaxy box.
"""

import torch

from models.demos.gemma4_d_p.demo.text_demo_prefill import TRACE_REGION_SIZE
from models.demos.gemma4_d_p.demo.text_demo_prefill import test_prefill_long_context_traced as _run_long_context_traced
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric


@torch.no_grad()
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_sanity_glx(mesh_device, reset_seeds, request):
    """Equivalent to test_prefill_long_context_traced[...-readback_final-ctx_256k-chunk8192-text-8x4]."""
    _run_long_context_traced(
        mesh_device=mesh_device,
        context_len=262144,
        chunk_size=8192,
        readback_all=False,
        token_source="text",
        reset_seeds=reset_seeds,
        request=request,
    )
