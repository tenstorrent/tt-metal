# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import FlashAttentionProgramConfig, flash_attention
from ttnn.operations.scaled_dot_product_attention.program_descriptor import _make_mcast_groups


def _make_inputs(device, shape, *, kv_sequence=None):
    torch.manual_seed(23)
    batch, heads, q_sequence, head_dim = shape
    kv_sequence = q_sequence if kv_sequence is None else kv_sequence
    # Moderate values exercise online max rescaling without saturating BF16 exp.
    q = torch.randn(batch, heads, q_sequence, head_dim, dtype=torch.bfloat16) * 0.35
    k = torch.randn(batch, heads, kv_sequence, head_dim, dtype=torch.bfloat16) * 0.35
    v = torch.randn(batch, heads, kv_sequence, head_dim, dtype=torch.bfloat16) * 0.35

    def to_device(tensor):
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return (q, k, v), tuple(to_device(tensor) for tensor in (q, k, v))


def _reference(q, k, v, scale=None):
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    return torch.softmax(q.float() @ k.float().transpose(-2, -1) * scale, dim=-1) @ v.float()


def test_flash_attention_spreads_kv_readers(device):
    grid = device.compute_with_storage_grid_size()
    width = min(grid.x, 4)
    batch_heads = min(grid.y, width)
    groups = _make_mcast_groups(device, batch_heads, width, spread_senders=True)
    for bh, cores, sender in groups:
        assert sender == cores[bh % width]
        assert all(core.y == cores[0].y for core in cores)
    assert len({sender.x for _, _, sender in groups}) == batch_heads


@pytest.mark.parametrize("use_kv_multicast", [False, True], ids=["direct_dram", "kv_mcast"])
def test_flash_attention_single_block(device, use_kv_multicast):
    """One KV block checks transpose, row softmax, PV, scheduling, and both reader topologies."""
    shape = (1, 2, 64, 64)
    (q, k, v), (tt_q, tt_k, tt_v) = _make_inputs(device, shape)
    config = FlashAttentionProgramConfig(
        query_block_tiles=1,
        key_block_tiles=2,
        qk_output_subblock=(1, 2),
        pv_output_subblock=(1, 2),
        softmax_block_tiles=2,
        num_cores=4,
        q_parallel_group_size=2,
        use_kv_multicast=use_kv_multicast,
        exp_approx_mode="exact",
    )
    tt_actual = flash_attention(tt_q, tt_k, tt_v, program_config=config)
    assert tt_actual.dtype == ttnn.bfloat16
    assert tt_actual.layout == ttnn.TILE_LAYOUT
    assert tt_actual.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    actual = ttnn.to_torch(tt_actual).float()
    assert list(actual.shape) == list(q.shape)
    assert_with_pcc(_reference(q, k, v), actual, 0.985)


def test_flash_attention_online_multiblock(device):
    """Two online steps exercise max, denominator, and output-state rescaling."""
    shape = (1, 2, 256, 128)
    (q, k, v), (tt_q, tt_k, tt_v) = _make_inputs(device, shape)
    config = FlashAttentionProgramConfig(
        query_block_tiles=2,
        key_block_tiles=4,
        qk_output_subblock=(1, 4),
        pv_output_subblock=(1, 4),
        q_parallel_group_size=4,
        use_kv_multicast=True,
        exp_approx_mode="fast",
    )
    actual = ttnn.to_torch(ttnn.flash_attention(tt_q, tt_k, tt_v, program_config=config)).float()
    assert_with_pcc(_reference(q, k, v), actual, 0.975)


def test_flash_attention_bf16_dest(device):
    """BF16 DEST mode supports its eight-tile budget and remains numerically usable."""
    shape = (1, 2, 256, 128)
    (q, k, v), (tt_q, tt_k, tt_v) = _make_inputs(device, shape)
    config = FlashAttentionProgramConfig(
        query_block_tiles=2,
        key_block_tiles=4,
        qk_output_subblock=(2, 4),
        pv_output_subblock=(2, 4),
        softmax_block_tiles=8,
        q_parallel_group_size=4,
        use_kv_multicast=True,
        exp_approx_mode="accurate_fast",
        fp32_dest_acc_en=False,
    )
    actual = ttnn.to_torch(flash_attention(tt_q, tt_k, tt_v, program_config=config)).float()
    assert_with_pcc(_reference(q, k, v), actual, 0.96)


@pytest.mark.parametrize("head_dim", [32, 64])
def test_flash_attention_auto_subblocks(device, head_dim):
    """Default block/subblock selection adapts to smaller tile-aligned dimensions."""
    shape = (1, 1, 64, head_dim)
    (q, k, v), (tt_q, tt_k, tt_v) = _make_inputs(device, shape)
    actual = ttnn.to_torch(flash_attention(tt_q, tt_k, tt_v)).float()
    assert_with_pcc(_reference(q, k, v), actual, 0.975)


def test_flash_attention_cross_sequence_and_custom_scale(device):
    shape = (1, 1, 64, 64)
    (q, k, v), (tt_q, tt_k, tt_v) = _make_inputs(device, shape, kv_sequence=128)
    config = FlashAttentionProgramConfig(
        query_block_tiles=1,
        key_block_tiles=2,
        qk_output_subblock=(1, 2),
        pv_output_subblock=(1, 2),
        softmax_block_tiles=2,
        num_cores=2,
        use_kv_multicast=False,
        exp_approx_mode="exact",
    )
    scale = 0.075
    actual = ttnn.to_torch(flash_attention(tt_q, tt_k, tt_v, scale=scale, program_config=config)).float()
    assert_with_pcc(_reference(q, k, v, scale), actual, 0.985)


def test_flash_attention_validation(device):
    (_, _, _), (tt_q, tt_k, tt_v) = _make_inputs(device, (1, 1, 64, 64))
    with pytest.raises(ValueError, match="not divisible by query_block_tiles"):
        flash_attention(
            tt_q,
            tt_k,
            tt_v,
            program_config=FlashAttentionProgramConfig(
                query_block_tiles=3,
                key_block_tiles=2,
                qk_output_subblock=(1, 2),
                pv_output_subblock=(1, 2),
            ),
        )
    with pytest.raises(ValueError, match="exceeds the 4-tile FP32 DEST budget"):
        FlashAttentionProgramConfig(qk_output_subblock=(2, 4), fp32_dest_acc_en=True).validate_basic()
    FlashAttentionProgramConfig(
        qk_output_subblock=(2, 4), softmax_block_tiles=8, fp32_dest_acc_en=False
    ).validate_basic()
    with pytest.raises(ValueError, match="accurate_fast exponential requires fp32_dest_acc_en=False"):
        FlashAttentionProgramConfig(exp_approx_mode="accurate_fast", fp32_dest_acc_en=True).validate_basic()
    with pytest.raises(ValueError, match="must differ"):
        FlashAttentionProgramConfig(reader_noc="noc0", writer_noc="noc0").validate_basic()

    host = torch.randn((1, 1, 64, 64))
    tt_float32 = ttnn.from_torch(
        host,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="must have bfloat16 dtype"):
        flash_attention(tt_float32, tt_float32, tt_float32)

    tt_l1 = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="must be in DRAM"):
        flash_attention(tt_l1, tt_l1, tt_l1)


@pytest.mark.skipif(os.environ.get("TTNN_RUN_LONG_FLASH_ATTN") != "1", reason="set TTNN_RUN_LONG_FLASH_ATTN=1")
def test_flash_attention_prefill_8h_4k(device):
    """Primary-regime smoke: full 8-head, 4096-token launch; verify sampled query rows."""
    shape = (1, 8, 4096, 128)
    (q, k, v), (tt_q, tt_k, tt_v) = _make_inputs(device, shape)
    actual = ttnn.to_torch(flash_attention(tt_q, tt_k, tt_v)).float()

    # Avoid materializing an 8x4096x4096 CPU score tensor. Sample query rows
    # across every head while still comparing against every key/value row.
    sample = torch.tensor([0, 31, 32, 1023, 2048, 4095])
    expected = _reference(q[:, :, sample, :], k, v)
    assert_with_pcc(expected, actual[:, :, sample, :], 0.999)


@pytest.mark.skipif(
    os.environ.get("TTNN_RUN_FLASH_ATTN_PRECISION") != "1", reason="set TTNN_RUN_FLASH_ATTN_PRECISION=1"
)
def test_flash_attention_prefill_8h_4k_precision_sweep(device):
    """Report long-shape PCC across exp, fidelity, and DEST-accumulation choices."""
    shape = (1, 8, 4096, 128)
    (q, k, v), (tt_q, tt_k, tt_v) = _make_inputs(device, shape)
    sample = torch.tensor([0, 31, 32, 1023, 2048, 4095])
    expected = _reference(q[:, :, sample, :], k, v)
    configs = {
        "default_bf16_dest_hifi2_fast_probs_accurate_fast_rescale": FlashAttentionProgramConfig(),
        "fp32_dest_hifi2_fast": FlashAttentionProgramConfig(
            softmax_block_tiles=4,
            fp32_dest_acc_en=True,
            rescale_exp_approx_mode="fast",
        ),
        "fp32_dest_hifi2_exact": FlashAttentionProgramConfig(
            softmax_block_tiles=4,
            fp32_dest_acc_en=True,
            exp_approx_mode="exact",
        ),
        "fp32_dest_hifi4_fast": FlashAttentionProgramConfig(
            softmax_block_tiles=4,
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            rescale_exp_approx_mode="fast",
        ),
        "bf16_dest_hifi2_fast_same_geometry": FlashAttentionProgramConfig(
            qk_output_subblock=(2, 2),
            pv_output_subblock=(2, 2),
            fp32_dest_acc_en=False,
            rescale_exp_approx_mode="fast",
        ),
        "bf16_dest_hifi2_fast_8tile": FlashAttentionProgramConfig(
            qk_output_subblock=(2, 4),
            pv_output_subblock=(2, 4),
            softmax_block_tiles=8,
            fp32_dest_acc_en=False,
            rescale_exp_approx_mode="fast",
        ),
        "bf16_dest_hifi2_accurate_fast_same_geometry": FlashAttentionProgramConfig(
            qk_output_subblock=(2, 2),
            pv_output_subblock=(2, 2),
            fp32_dest_acc_en=False,
            exp_approx_mode="accurate_fast",
        ),
        "bf16_dest_hifi2_accurate_fast_8tile": FlashAttentionProgramConfig(
            qk_output_subblock=(2, 4),
            pv_output_subblock=(2, 4),
            softmax_block_tiles=8,
            fp32_dest_acc_en=False,
            exp_approx_mode="accurate_fast",
        ),
        "bf16_dest_hifi2_accurate_fast_probs_fast_rescale": FlashAttentionProgramConfig(
            qk_output_subblock=(2, 4),
            pv_output_subblock=(2, 4),
            softmax_block_tiles=8,
            fp32_dest_acc_en=False,
            exp_approx_mode="accurate_fast",
            rescale_exp_approx_mode="fast",
        ),
    }

    pccs = {}
    for name, config in configs.items():
        actual = ttnn.to_torch(flash_attention(tt_q, tt_k, tt_v, program_config=config)).float()
        _, pccs[name] = comp_pcc(expected, actual[:, :, sample, :], pcc=0.0)
        assert pccs[name] >= 0.95, f"{name} PCC={pccs[name]}"

    logger.info(
        "\n=== flash_attention 8-head/4K precision sweep ===\n"
        + "\n".join(f"{name}: PCC={pcc:.12f}" for name, pcc in pccs.items())
    )
