# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for persistent KDA cache identities."""

from dataclasses import replace

from models.demos.deepseek_v3_d_p.tests.kda.utils import make_config
from models.demos.deepseek_v3_d_p.tt.kda.weights import _cache_stem


def test_tensor_cache_stem_identifies_config_and_placement() -> None:
    unbounded = make_config()
    bounded = replace(unbounded, gate_lower_bound=-5.0)
    baseline = _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (2, 4), 1)

    assert baseline == _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (2, 4), 1)
    assert baseline != _cache_stem("layer_0.kda", "decay_scale_flat", bounded, (2, 4), 1)
    assert baseline != _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (2, 4), 0)
    assert baseline != _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (4, 2), 1)
