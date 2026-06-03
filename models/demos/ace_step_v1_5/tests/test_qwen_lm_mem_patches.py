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
    ace_step_lm_head_sharded_norm_enabled,
    ace_step_lm_narrow_audio_vocab_enabled,
    ace_step_lm_prefill_l1_enabled,
    ace_step_lm_prefill_qkv_sweep_enabled,
    ace_step_lm_sdpa_concat_width_enabled,
    ace_step_lm_unified_decode_shard_enabled,
)
from models.demos.ace_step_v1_5.ttnn_impl.qwen_decode_sdpa_layout import ace_step_patch_model_args_sdpa_gather_unified
from models.demos.ace_step_v1_5.ttnn_impl.qwen_decode_shard import ace_step_patch_model_args_decode_unified_shard
from models.demos.ace_step_v1_5.ttnn_impl.qwen_lm_head_sharded_norm import ace_step_apply_lm_head_sharded_norm
from models.demos.ace_step_v1_5.ttnn_impl.qwen_prefill_l1 import (
    ace_step_apply_qwen_prefill_l1,
    ace_step_patch_model_args_lm_prefill_qkv_matmul,
    ace_step_patch_model_args_lm_prefill_wo_matmul,
    ace_step_patch_model_args_prefill_l1,
    ace_step_promote_attention_wqkv_to_dram_interleaved,
)
from models.tt_transformers.tt.common import Mode


def test_lm_mem_env_defaults():
    assert ace_step_lm_prefill_qkv_sweep_enabled() is True
    assert ace_step_lm_prefill_l1_enabled() is True
    assert ace_step_lm_unified_decode_shard_enabled() is True
    assert ace_step_lm_decode_qk_norm_sharded_enabled() is True
    assert ace_step_lm_head_sharded_norm_enabled() is True
    assert ace_step_lm_sdpa_concat_width_enabled() is True
    assert ace_step_lm_narrow_audio_vocab_enabled() is True


def test_prefill_l1_patches_attention_getters_not_mlp(monkeypatch):
    import functools

    import ttnn

    monkeypatch.setenv("ACE_STEP_LM_PREFILL_MLP_SWEEP", "0")

    dram = ttnn.DRAM_MEMORY_CONFIG
    l1 = ttnn.L1_MEMORY_CONFIG

    @functools.lru_cache(maxsize=None)
    def attn_qkv(mode, _prefetcher=None):
        return dram if mode == Mode.PREFILL else l1

    @functools.lru_cache(maxsize=None)
    def mlp_ff1(mode, _prefetcher=None):
        return dram if mode == Mode.PREFILL else l1

    model_args = SimpleNamespace(
        get_attn_qkv_mm_mem_config=attn_qkv,
        get_mlp_ff1_3_mem_config=mlp_ff1,
    )
    ace_step_patch_model_args_prefill_l1(model_args)

    assert model_args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None) is l1
    assert model_args.get_mlp_ff1_3_mem_config(Mode.PREFILL, None) is dram


def test_prefill_l1_keeps_mlp_dram_memcfg_getters(monkeypatch):
    import ttnn

    monkeypatch.setenv("ACE_STEP_LM_PREFILL_MLP_SWEEP", "1")
    dram = ttnn.DRAM_MEMORY_CONFIG
    model_args = SimpleNamespace(
        get_mlp_ff1_3_mem_config=mock.Mock(return_value=dram),
        get_mlp_ff2_mem_config=mock.Mock(return_value=dram),
    )
    ace_step_patch_model_args_prefill_l1(model_args)

    assert model_args.get_mlp_ff1_3_mem_config(Mode.PREFILL, None) is dram
    assert model_args.get_mlp_ff2_mem_config(Mode.PREFILL, None) is dram


def test_lm_head_sharded_norm_patches_distributed_norm_and_apply():
    lm_head_mem = object()
    lm_head_cfg = {"sharded_output_config": lm_head_mem, "sharded_program_config": object()}
    hidden = mock.Mock()
    hidden.memory_config.return_value.is_sharded.return_value = False
    dram_hidden = mock.Mock()
    dram_hidden.memory_config.return_value.is_sharded.return_value = False
    sharded_hidden = mock.Mock()
    sharded_hidden.memory_config.return_value.is_sharded.return_value = True
    rms = mock.Mock(return_value=sharded_hidden)

    class FakeDistributedNorm:
        def __init__(self):
            self.args = SimpleNamespace(
                get_norm_config=mock.Mock(return_value=lm_head_cfg),
                get_lm_head_input_mem_config=mock.Mock(return_value=SimpleNamespace(is_sharded=lambda: True)),
                is_multichip=False,
                is_distributed_norm=mock.Mock(return_value=False),
            )
            self.prefetcher = None
            self.tt_ccl = None
            self.ag_config_key = None
            self.norm = rms

        def forward(self, x, mode, norm_config=None):
            return "stock"

        def __call__(self, x, mode, norm_config=None):
            return self.forward(x, mode, norm_config)

    dnorm = FakeDistributedNorm()
    tt_model = SimpleNamespace(
        norm=dnorm,
        args=dnorm.args,
        prefetcher=None,
        _apply_norm_and_lm_head=mock.Mock(return_value="orig_logits"),
        forward=mock.Mock(return_value="fwd"),
    )

    with mock.patch(
        "models.demos.ace_step_v1_5.ttnn_impl.qwen_lm_head_sharded_norm.ttnn.to_memory_config",
        side_effect=lambda x, memory_config=None: dram_hidden if x is hidden else x,
    ) as to_mem, mock.patch(
        "models.demos.ace_step_v1_5.ttnn_impl.qwen_lm_head_sharded_norm.ttnn.interleaved_to_sharded",
        return_value=sharded_hidden,
    ) as i2s, mock.patch(
        "models.demos.ace_step_v1_5.ttnn_impl.qwen_lm_head_sharded_norm.ttnn.DRAM_MEMORY_CONFIG",
        object(),
    ):
        ace_step_apply_lm_head_sharded_norm(tt_model, dnorm.args)

        out = dnorm(hidden, Mode.PREFILL, norm_config=lm_head_cfg)
        assert out is sharded_hidden
        to_mem.assert_called()
        i2s.assert_called()
        assert rms.call_count == 1
        assert rms.call_args.kwargs["in_sharded"] is True
        assert rms.call_args.kwargs["out_sharded"] is True

        tt_model.lm_head = mock.Mock(return_value="logits")
        logits = tt_model._apply_norm_and_lm_head(hidden)
        assert logits == "logits"
        assert i2s.call_count == 2
        assert rms.call_count == 2
        tt_model.lm_head.assert_called_once_with(sharded_hidden)


def test_sharded_prestep_skips_dram_on_sharded_input():
    target = object()
    sharded = mock.Mock()
    sharded.memory_config.return_value.is_sharded.return_value = True
    dnorm = SimpleNamespace(
        args=SimpleNamespace(is_multichip=False, is_distributed_norm=mock.Mock(return_value=False)),
        prefetcher=None,
        tt_ccl=None,
        ag_config_key=None,
    )
    norm_config = {"sharded_output_config": target}

    with mock.patch(
        "models.demos.ace_step_v1_5.ttnn_impl.qwen_lm_head_sharded_norm.ttnn.to_memory_config",
    ) as to_mem, mock.patch(
        "models.demos.ace_step_v1_5.ttnn_impl.qwen_lm_head_sharded_norm.ttnn.interleaved_to_sharded",
    ) as i2s, mock.patch(
        "models.demos.ace_step_v1_5.ttnn_impl.qwen_lm_head_sharded_norm._shard_for_lm_head",
        return_value=sharded,
    ) as shard:
        from models.demos.ace_step_v1_5.ttnn_impl.qwen_lm_head_sharded_norm import _distributed_norm_prestep

        out = _distributed_norm_prestep(
            dnorm,
            sharded,
            mode=Mode.PREFILL,
            norm_config=norm_config,
            target_sharded_mem_cfg=target,
        )
        assert out is sharded
        to_mem.assert_not_called()
        i2s.assert_not_called()
        shard.assert_called_once_with(sharded, target)


def test_decode_unified_shard_leaves_all_getters_stock():
    residual = object()
    orig_mlp = mock.Mock(return_value="stock_mlp")
    orig_wo = mock.Mock(return_value="stock_wo")
    model_args = SimpleNamespace(
        is_galaxy=False,
        get_residual_mem_config=mock.Mock(return_value=residual),
        get_mlp_ff2_mem_config=orig_mlp,
        get_attn_wo_output_mem_config=orig_wo,
        get_attn_all_gather_output_mem_config=orig_wo,
    )
    ace_step_patch_model_args_decode_unified_shard(model_args)

    assert model_args.get_mlp_ff2_mem_config(Mode.DECODE, None) == "stock_mlp"
    assert model_args.get_attn_wo_output_mem_config(Mode.DECODE, None) == "stock_wo"
    assert model_args.get_attn_all_gather_output_mem_config(Mode.DECODE, None) == "stock_wo"
    model_args.get_residual_mem_config.assert_not_called()


def test_decode_unified_shard_leaves_wide_matmul_getters_stock():
    residual = object()
    orig_mock = mock.Mock(return_value="stock")
    model_args = SimpleNamespace(
        is_galaxy=False,
        get_residual_mem_config=mock.Mock(return_value=residual),
        get_mlp_ff1_3_mem_config=orig_mock,
        get_attn_qkv_mm_mem_config=orig_mock,
    )
    ace_step_patch_model_args_decode_unified_shard(model_args)

    assert model_args.get_mlp_ff1_3_mem_config(Mode.DECODE, None) == "stock"
    assert model_args.get_attn_qkv_mm_mem_config(Mode.DECODE, None) == "stock"
    orig_mock.assert_has_calls(
        [
            mock.call(Mode.DECODE, None),
            mock.call(Mode.DECODE, None),
        ]
    )


def test_decode_unified_shard_skips_when_prefetcher_set():
    residual = object()
    prefetcher = object()
    orig_mock = mock.Mock(return_value="ring")
    model_args = SimpleNamespace(
        is_galaxy=False,
        get_residual_mem_config=mock.Mock(return_value=residual),
        get_mlp_ff2_mem_config=orig_mock,
    )
    ace_step_patch_model_args_decode_unified_shard(model_args)

    assert model_args.get_mlp_ff2_mem_config(Mode.DECODE, prefetcher) == "ring"
    orig_mock.assert_called_once_with(Mode.DECODE, prefetcher)


def test_sdpa_gather_unified_leaves_gather_getter_stock():
    residual = object()
    orig = mock.Mock(return_value="gather")
    model_args = SimpleNamespace(
        is_galaxy=False,
        get_residual_mem_config=mock.Mock(return_value=residual),
        get_attn_gather_users_mem_config=orig,
    )
    ace_step_patch_model_args_sdpa_gather_unified(model_args)

    assert model_args.get_attn_gather_users_mem_config(Mode.DECODE, 1, None) == "gather"
    orig.assert_called_once_with(Mode.DECODE, 1, None)
    model_args.get_residual_mem_config.assert_not_called()


def test_lm_prefill_qkv_sweep_patches_program_config(monkeypatch):
    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_lm_prefill_qkv_matmul_program_config

    monkeypatch.setenv("ACE_STEP_LM_PREFILL_QKV_SWEEP", "1")
    orig = mock.Mock(return_value="stock")
    device = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=4))
    model_args = SimpleNamespace(
        dim=2048,
        qkv_size=4096,
        get_attn_qkv_program_config=orig,
    )
    ace_step_patch_model_args_lm_prefill_qkv_matmul(model_args, device)

    pinned = ace_step_lm_prefill_qkv_matmul_program_config(device, seq_len=128, hidden_dim=2048, qkv_dim=4096)
    assert pinned is not None
    got = model_args.get_attn_qkv_program_config(Mode.PREFILL, 128, None)
    assert got is not pinned
    assert type(got) is type(pinned)
    assert got.in0_block_w == 8
    assert got.per_core_M == 4
    assert got.per_core_N == 4
    assert model_args.get_attn_qkv_program_config(Mode.DECODE, 1, None) == "stock"
    assert model_args.get_attn_qkv_program_config(Mode.PREFILL, 256, None) == "stock"


def test_lm_prefill_mlp_ff1_sweep_patches_program_config(monkeypatch):
    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_lm_prefill_mlp_ff1_3_matmul_program_config

    monkeypatch.setenv("ACE_STEP_LM_PREFILL_MLP_SWEEP", "1")
    orig = mock.Mock(return_value="stock")
    device = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=4))
    model_args = SimpleNamespace(
        dim=2048,
        hidden_dim=6144,
        get_mlp_ff1_3_prg_config=orig,
    )
    from models.demos.ace_step_v1_5.ttnn_impl.qwen_prefill_l1 import (
        ace_step_patch_model_args_lm_prefill_mlp_ff1_3_matmul,
    )

    ace_step_patch_model_args_lm_prefill_mlp_ff1_3_matmul(model_args, device)

    pinned = ace_step_lm_prefill_mlp_ff1_3_matmul_program_config(device, seq_len=128, k_dim=2048, n_dim=6144)
    assert pinned is not None
    got = model_args.get_mlp_ff1_3_prg_config(Mode.PREFILL, 128, None)
    assert type(got) is type(pinned)
    assert got.in0_block_w == 8
    assert got.per_core_M == 4
    assert got.per_core_N == 6
    assert got.out_subblock_h == 1
    assert got.out_subblock_w == 2
    assert model_args.get_mlp_ff1_3_prg_config(Mode.PREFILL, 512, None) == "stock"


def test_lm_prefill_mlp_ff2_sweep_patches_program_config(monkeypatch):
    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_lm_prefill_mlp_ff2_matmul_program_config

    monkeypatch.setenv("ACE_STEP_LM_PREFILL_MLP_SWEEP", "1")
    orig = mock.Mock(return_value="stock")
    device = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=4))
    model_args = SimpleNamespace(
        dim=2048,
        hidden_dim=6144,
        get_mlp_ff2_prg_config=orig,
    )
    from models.demos.ace_step_v1_5.ttnn_impl.qwen_prefill_l1 import ace_step_patch_model_args_lm_prefill_mlp_ff2_matmul

    ace_step_patch_model_args_lm_prefill_mlp_ff2_matmul(model_args, device)

    pinned = ace_step_lm_prefill_mlp_ff2_matmul_program_config(device, seq_len=128, k_dim=6144, n_dim=2048)
    assert pinned is not None
    got = model_args.get_mlp_ff2_prg_config(Mode.PREFILL, 128, None)
    assert type(got) is type(pinned)
    assert got.per_core_N == 2
    assert got.out_subblock_h == 1


def test_lm_prefill_wo_sweep_patches_program_config(monkeypatch):
    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_lm_prefill_wo_matmul_program_config

    monkeypatch.setenv("ACE_STEP_LM_PREFILL_QKV_SWEEP", "1")
    orig = mock.Mock(return_value="stock")
    device = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=4))
    model_args = SimpleNamespace(
        dim=2048,
        n_heads=16,
        head_dim=128,
        num_devices=1,
        get_attn_wo_program_config=orig,
    )
    ace_step_patch_model_args_lm_prefill_wo_matmul(model_args, device)

    pinned = ace_step_lm_prefill_wo_matmul_program_config(device, seq_len=128, k_dim=2048, n_dim=2048)
    assert pinned is not None
    got = model_args.get_attn_wo_program_config(Mode.PREFILL, 128, None)
    assert got is not pinned
    assert type(got) is type(pinned)
    assert got.in0_block_w == 8
    assert got.per_core_M == 4
    assert got.per_core_N == 2
    assert got.out_subblock_h == 1
    assert got.out_subblock_w == 2
    assert model_args.get_attn_wo_program_config(Mode.DECODE, 1, None) == "stock"


def test_apply_prefill_l1_keeps_decode_wqkv_sharded(monkeypatch):
    import ttnn

    monkeypatch.setenv("ACE_STEP_LM_PREFILL_L1", "1")
    sharded_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (2048, 512),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    wqkv = SimpleNamespace(memory_config=lambda: sharded_mc)
    attn = SimpleNamespace(
        wqkv=wqkv,
        wo=wqkv,
        forward_prefill=lambda *a, **k: None,
        attention_norm=None,
    )
    layer = SimpleNamespace(
        attention=attn,
        feed_forward=SimpleNamespace(w1=wqkv, w2=wqkv, w3=wqkv, forward=lambda *a, **k: None),
        forward=lambda *a, **k: None,
    )
    tt_model = SimpleNamespace(
        layers=[layer],
        embd=SimpleNamespace(forward=lambda *a, **k: None),
        norm=None,
    )
    model_args = SimpleNamespace(prefetcher=None)

    with mock.patch(
        "models.demos.ace_step_v1_5.ttnn_impl.qwen_prefill_l1.ace_step_linear_l1_memory_config",
        return_value=ttnn.L1_MEMORY_CONFIG,
    ), mock.patch(
        "models.demos.ace_step_v1_5.ttnn_impl.qwen_prefill_l1.ace_step_patch_model_args_prefill_l1",
    ), mock.patch(
        "models.demos.ace_step_v1_5.ttnn_impl.qwen_prefill_l1.ace_step_apply_qwen_prefill_matmul_configs",
    ), mock.patch(
        "models.demos.ace_step_v1_5.ttnn_impl.qwen_prefill_l1.ace_step_apply_qwen_prefill_gate_up_fusion",
    ):
        ace_step_apply_qwen_prefill_l1(tt_model, model_args)

    assert attn.wqkv is wqkv
    assert attn.wqkv.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED


def test_lm_prefill_qkv_sweep_promotes_sharded_wqkv(monkeypatch):
    import ttnn

    monkeypatch.setenv("ACE_STEP_LM_PREFILL_QKV_SWEEP", "1")
    dram = ttnn.DRAM_MEMORY_CONFIG
    sharded_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (2048, 512),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    interleaved_mc = dram
    device = object()
    wqkv_sharded = SimpleNamespace(
        memory_config=lambda: sharded_mc,
        device=lambda: device,
        dtype=ttnn.bfloat8_b,
    )
    wqkv_interleaved = SimpleNamespace(memory_config=lambda: interleaved_mc)
    promoted = object()
    attn = SimpleNamespace(wqkv=wqkv_sharded, wo=wqkv_sharded, forward_prefill=lambda *a, **k: None)
    tt_model = SimpleNamespace(layers=[SimpleNamespace(attention=attn)])
    torch_w = mock.Mock()

    with mock.patch.object(ttnn, "to_torch", return_value=torch_w) as to_torch:
        with mock.patch.object(ttnn, "ReplicateTensorToMesh", return_value="mapper"):
            with mock.patch.object(ttnn, "from_torch", return_value=promoted) as from_torch:
                ace_step_promote_attention_wqkv_to_dram_interleaved(tt_model)

    assert to_torch.call_count == 2
    assert from_torch.call_count == 2
    assert from_torch.call_args.kwargs["memory_config"] is dram
    assert attn.wqkv is wqkv_sharded
    assert attn.wqkv_prefill_interleaved is promoted
    assert attn._ace_step_prefill_sweep_patched is True

    attn.wqkv_prefill_interleaved = wqkv_interleaved
    with mock.patch.object(ttnn, "to_torch") as to_torch:
        ace_step_promote_attention_wqkv_to_dram_interleaved(tt_model)
    to_torch.assert_not_called()


def test_narrow_column_band_and_split_hits():
    idx = torch.tensor([100, 500, 1200], dtype=torch.long)
    assert ace_step_narrow_column_band(idx) == (100, 1201)
    splits = [256, 256, 256, 256]
    assert ace_step_splits_for_band(splits, 100, 1201) == [0, 1, 2, 3]
    assert ace_step_splits_for_band(splits, 300, 400) == [1]
    ranges = ace_step_split_column_ranges(splits)
    assert ranges == [(0, 256), (256, 512), (512, 768), (768, 1024)]
