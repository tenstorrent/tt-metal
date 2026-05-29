# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for 5 Hz LM memory-layout patches (prefill L1 + decode shard unification + P2/P3)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch

from models.demos.ace_step_v1_5.ttnn_impl.ace_step_lm_head_narrow import (
    ace_step_narrow_column_band,
    ace_step_split_column_ranges,
    ace_step_splits_for_band,
)
from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_lm_decode_qk_norm_sharded_enabled,
    ace_step_lm_narrow_audio_vocab_enabled,
    ace_step_lm_prefill_l1_enabled,
    ace_step_lm_sdpa_concat_width_enabled,
    ace_step_lm_unified_decode_shard_enabled,
)
from models.demos.ace_step_v1_5.ttnn_impl.qwen_decode_sdpa_layout import ace_step_patch_model_args_sdpa_gather_unified
from models.demos.ace_step_v1_5.ttnn_impl.qwen_decode_shard import ace_step_patch_model_args_decode_unified_shard
from models.tt_transformers.tt.common import Mode


def test_lm_mem_env_defaults():
    assert ace_step_lm_prefill_l1_enabled() is False
    assert ace_step_lm_unified_decode_shard_enabled() is True
    assert ace_step_lm_decode_qk_norm_sharded_enabled() is True
    assert ace_step_lm_sdpa_concat_width_enabled() is True
    assert ace_step_lm_narrow_audio_vocab_enabled() is True


def test_decode_unified_shard_routes_decode_getters_to_residual():
    residual = object()
    orig_mock = mock.Mock(return_value="orig")
    model_args = SimpleNamespace(
        is_galaxy=False,
        get_residual_mem_config=mock.Mock(return_value=residual),
        get_mlp_ff1_3_mem_config=orig_mock,
    )
    ace_step_patch_model_args_decode_unified_shard(model_args)

    assert model_args.get_mlp_ff1_3_mem_config(Mode.DECODE, None) is residual
    assert model_args.get_mlp_ff1_3_mem_config(Mode.PREFILL, None) == "orig"
    orig_mock.assert_called_once_with(Mode.PREFILL, None)


def test_decode_unified_shard_skips_when_prefetcher_set():
    residual = object()
    prefetcher = object()
    orig_mock = mock.Mock(return_value="ring")
    model_args = SimpleNamespace(
        is_galaxy=False,
        get_residual_mem_config=mock.Mock(return_value=residual),
        get_attn_qkv_mm_mem_config=orig_mock,
    )
    ace_step_patch_model_args_decode_unified_shard(model_args)

    assert model_args.get_attn_qkv_mm_mem_config(Mode.DECODE, prefetcher) == "ring"
    orig_mock.assert_called_once_with(Mode.DECODE, prefetcher)


def test_sdpa_gather_unified_routes_decode_to_residual():
    residual = object()
    orig = mock.Mock(return_value="gather")
    model_args = SimpleNamespace(
        is_galaxy=False,
        get_residual_mem_config=mock.Mock(return_value=residual),
        get_attn_gather_users_mem_config=orig,
    )
    ace_step_patch_model_args_sdpa_gather_unified(model_args)

    assert model_args.get_attn_gather_users_mem_config(Mode.DECODE, 1, None) is residual
    assert model_args.get_attn_gather_users_mem_config(Mode.PREFILL, 1, None) == "gather"


def test_narrow_column_band_and_split_hits():
    idx = torch.tensor([100, 500, 1200], dtype=torch.long)
    assert ace_step_narrow_column_band(idx) == (100, 1201)
    splits = [256, 256, 256, 256]
    assert ace_step_splits_for_band(splits, 100, 1201) == [0, 1, 2, 3]
    assert ace_step_splits_for_band(splits, 300, 400) == [1]
    ranges = ace_step_split_column_ranges(splits)
    assert ranges == [(0, 256), (256, 512), (512, 768), (768, 1024)]
