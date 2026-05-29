# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for 5 Hz LM memory-layout patches (prefill L1 + decode shard unification)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_lm_prefill_l1_enabled,
    ace_step_lm_unified_decode_shard_enabled,
)
from models.demos.ace_step_v1_5.ttnn_impl.qwen_decode_shard import ace_step_patch_model_args_decode_unified_shard
from models.tt_transformers.tt.common import Mode


def test_lm_mem_env_defaults():
    assert ace_step_lm_prefill_l1_enabled() is False
    assert ace_step_lm_unified_decode_shard_enabled() is True


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
