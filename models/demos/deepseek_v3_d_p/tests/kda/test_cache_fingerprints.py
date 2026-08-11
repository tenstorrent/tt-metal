# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for persistent KDA cache identities."""

from dataclasses import replace

from models.demos.deepseek_v3_d_p.tests.kda.utils import make_config
from models.demos.deepseek_v3_d_p.tt.kda.weights import _cache_stem


def test_tensor_cache_stem_uses_only_parent_owned_layer_namespace() -> None:
    unbounded = make_config()
    bounded = replace(unbounded, gate_lower_bound=-5.0)

    assert _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (1, 1), 1) == "layer_0.kda.decay_scale_flat"
    assert _cache_stem("layer_0.kda", "decay_scale_flat", bounded, (2, 4), 0) == "layer_0.kda.decay_scale_flat"
