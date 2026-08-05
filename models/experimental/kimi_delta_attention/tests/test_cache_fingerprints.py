# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for persistent KDA cache identities."""

from dataclasses import replace

from models.experimental.kimi_delta_attention.tests.utils import make_config
from models.experimental.kimi_delta_attention.tt.weights import _cache_stem


def test_tensor_cache_stem_distinguishes_decay_mode() -> None:
    unbounded = make_config()
    bounded = replace(unbounded, gate_lower_bound=-5.0)

    unbounded_stem = _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (1, 1), 1)
    bounded_stem = _cache_stem("layer_0.kda", "decay_scale_flat", bounded, (1, 1), 1)

    assert ".bounded0." in unbounded_stem
    assert ".bounded1." in bounded_stem
    assert unbounded_stem != bounded_stem
