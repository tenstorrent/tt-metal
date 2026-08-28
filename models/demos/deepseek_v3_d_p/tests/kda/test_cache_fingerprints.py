# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for persistent KDA cache identities."""

from dataclasses import replace

import torch

from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_config
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights, _cache_stem


def test_checkpoint_identity_depends_on_weight_content() -> None:
    baseline = {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    changed = {"weight": torch.tensor([1.0, 3.0], dtype=torch.bfloat16)}

    assert kda_state_dict_sha256(baseline) == kda_state_dict_sha256(baseline)
    assert kda_state_dict_sha256(baseline) != kda_state_dict_sha256(changed)


def test_cache_completeness_is_false_for_missing_directory(tmp_path) -> None:
    assert not KDAWeights.check_cache_complete(tmp_path / "missing", "layer_0.kda", make_config(), object())


def test_tensor_cache_stem_identifies_config_and_placement() -> None:
    unbounded = make_config()
    bounded = replace(unbounded, gate_lower_bound=-5.0)
    baseline = _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (2, 4), 1)

    assert baseline == _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (2, 4), 1)
    assert baseline != _cache_stem("layer_0.kda", "decay_scale_flat", bounded, (2, 4), 1)
    assert baseline != _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (2, 4), 0)
    assert baseline != _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (4, 2), 1)
