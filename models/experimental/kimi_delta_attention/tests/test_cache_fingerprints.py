# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for persistent KDA cache identities."""

from dataclasses import replace
from pathlib import Path

import torch

from models.experimental.kimi_delta_attention.tests.perf.test_layer_perf import _cpu_reference_cache_path
from models.experimental.kimi_delta_attention.tests.utils import KimiK3TestCase, make_config, random_weights
from models.experimental.kimi_delta_attention.tt.weights import _cache_stem


def test_tensor_cache_stem_distinguishes_decay_mode() -> None:
    unbounded = make_config()
    bounded = replace(unbounded, gate_lower_bound=-5.0)

    unbounded_stem = _cache_stem("layer_0.kda", "decay_scale_flat", unbounded, (1, 1), 1)
    bounded_stem = _cache_stem("layer_0.kda", "decay_scale_flat", bounded, (1, 1), 1)

    assert ".bounded0." in unbounded_stem
    assert ".bounded1." in bounded_stem
    assert unbounded_stem != bounded_stem


def test_cpu_reference_cache_path_fingerprints_layer_weights(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    config = make_config()
    state_dict = random_weights(config)
    hidden = torch.zeros(1, 32, config.hidden_size, dtype=torch.bfloat16)

    original = KimiK3TestCase(config, state_dict, hidden, checkpoint_dir)
    copied = KimiK3TestCase(
        config, {name: tensor.clone() for name, tensor in state_dict.items()}, hidden, checkpoint_dir
    )
    changed_state_dict = {name: tensor.clone() for name, tensor in state_dict.items()}
    changed_state_dict["A_log"].flatten()[0] += 1
    changed = KimiK3TestCase(config, changed_state_dict, hidden, checkpoint_dir)

    assert _cpu_reference_cache_path(original) == _cpu_reference_cache_path(copied)
    assert _cpu_reference_cache_path(original) != _cpu_reference_cache_path(changed)
