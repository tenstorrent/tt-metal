# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of what is H3's own in the step cache (First Block Cache): the pipeline refuses the switch on
the denoise paths the cache does not serve, and hands the shared `DBCache` a mask of the rung's true rows."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.tt_dit.pipelines.minimax_h3 import pipeline_minimax_h3 as pm

ENV = "MINIMAX_H3_STEP_CACHE"


def _step_cache(*, trace_denoise=False, dynamic_length=False):
    pipeline = object.__new__(pm.MiniMaxH3Pipeline)
    pipeline.trace_denoise, pipeline.dynamic_length = trace_denoise, dynamic_length
    pipeline.mesh_device, pipeline.sp_axis = None, 0
    return pipeline._step_cache(SimpleNamespace(num_blocks=4), 64, 40)


@pytest.mark.parametrize("path", [{"dynamic_length": True}, {"trace_denoise": True}])
def test_switch_is_refused_where_the_cache_does_not_run(monkeypatch, expect_error, path):
    """The pipeline's own 4x8 preset is dynamic-length and its 4x32 one traced: the switch must not be silently
    off there."""
    monkeypatch.setenv(ENV, "0.08807")
    with expect_error(RuntimeError, "MINIMAX_H3_DYNAMIC_LENGTH=0"):
        _step_cache(**path)
    monkeypatch.delenv(ENV)
    assert _step_cache(**path) == (None, None)


def test_padding_rows_are_masked_out_of_the_change_test(monkeypatch):
    monkeypatch.setenv(ENV, "0.08807")
    monkeypatch.setattr(pm, "from_torch", lambda tensor, **_: tensor)
    cache, valid_rows = _step_cache()
    assert cache.config.residual_diff_threshold == 0.08807
    assert valid_rows.shape == (1, 1, 64, 1)
    assert valid_rows.flatten().tolist() == [1.0] * 40 + [0.0] * 24
