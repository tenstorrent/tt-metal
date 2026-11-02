# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Non-parametrized GLX smoke tests for Gemma4 disaggregated prefill.

Pins the canonical production shape covered by
``text_demo_prefill.py::test_prefill_long_context_traced``
(traced, 256k context, the service chunk size, real text, 8x4 = CP8/TP4), plus chunk
2048 at 32k context for the short-slab paths (sharded projections, L1 reduce-scatter,
sliding K split), so a single
``pytest models/demos/gemma4_d_p/tests/test_sanity_glx.py`` run is the
fast sanity check that these paths still execute end to end on a Galaxy box.
"""

import torch

from models.demos.gemma4_d_p.demo.text_demo_prefill import TRACE_REGION_SIZE
from models.demos.gemma4_d_p.demo.text_demo_prefill import test_prefill_long_context_traced as _run_long_context_traced
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4ServiceConfig


@torch.no_grad()
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_sanity_glx(mesh_device, reset_seeds, request):
    """Equivalent to test_prefill_long_context_traced at 256k context and Gemma4ServiceConfig.CHUNK_SIZE, real text,
    8x4."""
    _run_long_context_traced(
        mesh_device=mesh_device,
        context_len=262144,
        chunk_size=Gemma4ServiceConfig.CHUNK_SIZE,
        token_source="text",
        reset_seeds=reset_seeds,
        request=request,
    )


@torch.no_grad()
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_sanity_glx_short_slab(mesh_device, reset_seeds, request):
    """Equivalent to test_prefill_long_context_traced[blackhole-ctx_32k-chunk2048-text-8x4]."""
    _run_long_context_traced(
        mesh_device=mesh_device,
        context_len=32768,
        chunk_size=2048,
        token_source="text",
        reset_seeds=reset_seeds,
        request=request,
    )
