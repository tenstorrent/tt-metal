# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Depth resolution: the tool must be able to learn a model's block count WITHOUT building it,
from whatever config dialect the model ships -- not just HF.

The alternative is expressing "all layers" as an absent env var, which a model that does
os.environ.setdefault("TT_PERF_LAYERS", "2") (xtts_v2, and every perf test the tool generated before
2026-07-26) silently converts into a 2-layer build.
"""
from __future__ import annotations

import json

from models.experimental.perf_automation.agent.layer_depth import (
    _depth_from_mapping,
    full_depth_from_config,
)


def test_reads_each_config_dialect():
    assert _depth_from_mapping({"num_hidden_layers": 32}) == 32  # HF
    assert _depth_from_mapping({"n_layers": 80}) == 80  # Meta params.json
    assert _depth_from_mapping({"n_layer": 12}) == 12  # GPT-2 lineage
    assert _depth_from_mapping({"gpt_layers": 30}) == 30  # XTTS
    assert _depth_from_mapping({"num_blocks": 6}) == 6


def test_reads_a_nested_text_config():
    """Multimodal wrappers declare the stack under text_config, not at the root."""
    assert _depth_from_mapping({"vision_config": {}, "text_config": {"num_hidden_layers": 28}}) == 28


def test_unknown_or_bogus_declares_nothing():
    """None, never a guess -- the caller must fall back to letting the builder reveal its depth."""
    assert _depth_from_mapping({}) is None
    assert _depth_from_mapping({"num_hidden_layers": 0}) is None  # 0 is not a depth
    assert _depth_from_mapping({"num_hidden_layers": "many"}) is None
    assert _depth_from_mapping({"num_hidden_layers": True}) is None  # bool is not a count
    assert _depth_from_mapping(None) is None


def test_reads_a_root_config_file(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 24}))
    assert full_depth_from_config(model_dir=tmp_path) == 24


def test_ignores_other_models_configs_in_subdirectories(tmp_path):
    """THE TRAP: a llama demo ships model_params/Qwen2.5-VL-72B-Instruct/config.json and ~40 more.
    A recursive scan would return a DIFFERENT model's depth, which looks perfectly plausible."""
    foreign = tmp_path / "model_params" / "Qwen2.5-VL-72B-Instruct"
    foreign.mkdir(parents=True)
    (foreign / "config.json").write_text(json.dumps({"num_hidden_layers": 80}))
    assert full_depth_from_config(model_dir=tmp_path) is None


def test_root_config_wins_over_a_foreign_one(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 32}))
    foreign = tmp_path / "model_params" / "other"
    foreign.mkdir(parents=True)
    (foreign / "config.json").write_text(json.dumps({"num_hidden_layers": 80}))
    assert full_depth_from_config(model_dir=tmp_path) == 32


def test_malformed_config_is_not_fatal(tmp_path):
    (tmp_path / "config.json").write_text("{ not json")
    assert full_depth_from_config(model_dir=tmp_path) is None
