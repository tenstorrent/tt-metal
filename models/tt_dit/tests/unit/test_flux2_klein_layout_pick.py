# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only: the DiT layout choice of the FLUX.2 Klein pipeline (no device). The FLUX 3 Klein test
(``test_flux3_klein_layout_pick.py``) with the FLUX.2 defaults: the pipeline, the demo and the benchmark script
default to the single SP8 x TP4 layout; ``--layout dual`` keeps sp4tp8 primary with sp8tp4 from 16K tokens up."""
import inspect

import pytest

from models.tt_dit.pipelines.flux2_klein.pipeline_flux2_klein import (
    LARGE_LAYOUT_MIN_TOKENS,
    Flux2KleinPipeline,
    layout_name,
    pick_layout,
)


def test_layout_names_on_the_4x8_mesh():
    assert layout_name(0, 1) == "sp4tp8" and layout_name(1, 0) == "sp8tp4"
    assert layout_name(0, 1, (8, 4)) == "sp8tp4"


def test_pipeline_defaults_to_the_single_sp8tp4_layout():
    """Measured 2026-10-24 (4B distilled, warm eager, seed 123, bench_matrix_flux2.py): SP8 x TP4 is the faster or
    equal layout on every row of the 8-row matrix, so one resident layout is the default (no second weight set)."""
    params = inspect.signature(Flux2KleinPipeline.__init__).parameters
    assert layout_name(params["sp_axis"].default, params["tp_axis"].default) == "sp8tp4"
    assert params["large_layout"].default is None
    assert params["large_layout_min_tokens"].default == LARGE_LAYOUT_MIN_TOKENS == 16384


@pytest.mark.parametrize(
    "tokens, expected",
    [  # the dual pipeline (sp4tp8 primary, sp8tp4 resident for the large canvases) routes by the DiT row count
        (768 * 768 // 256 + 512, "sp4tp8"),
        (4096 + 512, "sp4tp8"),
        (4096 + 2 * 4096 + 512, "sp4tp8"),
        (LARGE_LAYOUT_MIN_TOKENS - 1, "sp4tp8"),
        (LARGE_LAYOUT_MIN_TOKENS, "sp8tp4"),
        (16384 + 512, "sp8tp4"),
        (16384 + 3 * 4096 + 512, "sp8tp4"),
    ],
)
def test_large_layout_from_16k_tokens(tokens, expected):
    assert pick_layout(tokens, primary="sp4tp8", large="sp8tp4") == expected


def test_single_layout_pipelines_never_switch():
    assert pick_layout(1 << 20, primary="sp4tp8", large=None) == "sp4tp8"
    assert pick_layout(10, primary="sp8tp4", large=None) == "sp8tp4"
    assert pick_layout(10, primary="sp4tp8", large="sp8tp4", min_tokens=0) == "sp8tp4"
