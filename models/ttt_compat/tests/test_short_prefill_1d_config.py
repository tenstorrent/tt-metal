# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Which program config the prefill projections (QKV, WO, FF1/FF3, FF2) pick at short prompts.

On a single card, prompts up to 128 tokens run the projections as a full-grid 1D mcast_in0 matmul. Every other
sequence length and device count keeps its existing config. These tests need no device: the config getters are
called on a stand-in for ModelArgs that carries only the attributes the short-prefill rule reads.
"""

from types import MethodType, SimpleNamespace

import pytest

import ttnn
from models.ttt_compat.tt.common import Mode
from models.ttt_compat.tt.model_config import MathFidelitySetting, ModelArgs, OpGroup

# Llama-3.1-8B shapes
DIM = 4096
HIDDEN_DIM = 14336
N_HEADS = 32
HEAD_DIM = 128
QKV_SIZE = HEAD_DIM * (2 * 8 + N_HEADS)


def _decoders(op_settings=None):
    """A stand-in for DecodersPrecision: one decoder, `op_settings` merged over the default dict."""
    op_fidelity = dict(op_settings or {})
    return SimpleNamespace(
        decoder_optimizations={0: SimpleNamespace(op_fidelity_settings=op_fidelity)},
    )


def _args(num_devices=1, is_galaxy=False, minimal_above_128=True, op_settings=None):
    args = SimpleNamespace(
        num_devices=num_devices,
        is_galaxy=is_galaxy,
        cluster_shape=[1, 1] if not is_galaxy else [4, 8],
        dim=DIM,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        qkv_size=QKV_SIZE,
        decoders_optimizations=_decoders(op_settings or {}),
        SHORT_PREFILL_1D_MAX_SEQ_LEN=ModelArgs.SHORT_PREFILL_1D_MAX_SEQ_LEN,
    )
    args.use_minimal_prefill_matmul = lambda seq_len: minimal_above_128 and seq_len > 128
    args.short_prefill_1d_matmul_config = MethodType(ModelArgs.short_prefill_1d_matmul_config, args)
    args.any_decoder_fp32_dest_acc = MethodType(ModelArgs.any_decoder_fp32_dest_acc, args)
    return args


def _getters(args):
    return {
        "ff1_3": lambda seq: ModelArgs.get_mlp_ff1_3_prg_config.__wrapped__(args, Mode.PREFILL, seq),
        "ff2": lambda seq: ModelArgs.get_mlp_ff2_prg_config.__wrapped__(args, Mode.PREFILL, seq),
        "qkv": lambda seq: ModelArgs.get_attn_qkv_program_config.__wrapped__(args, Mode.PREFILL, seq),
        "wo": lambda seq: ModelArgs.get_attn_wo_program_config.__wrapped__(args, Mode.PREFILL, seq),
    }


@pytest.mark.parametrize("seq_len", [32, 64, 96, 128])
@pytest.mark.parametrize("name", ["ff1_3", "ff2", "qkv", "wo"])
def test_single_card_short_prefill_uses_1d_mcast_in0(name, seq_len):
    pc = _getters(_args())[name](seq_len)
    assert isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig)
    assert pc.mcast_in0
    assert pc.fuse_batch
    assert (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y) == (8, 8)
    assert pc.per_core_M == seq_len // 32


@pytest.mark.parametrize(
    "name, k, n",
    [
        ("ff1_3", DIM, HIDDEN_DIM),
        ("ff2", HIDDEN_DIM, DIM),
        ("qkv", DIM, QKV_SIZE),
        ("wo", N_HEADS * HEAD_DIM, DIM),
    ],
)
def test_1d_config_tiles_the_whole_weight(name, k, n):
    pc = _getters(_args())[name](128)
    k_tiles, n_tiles = k // 32, -(-n // 32)
    assert k_tiles % pc.in0_block_w == 0
    assert pc.per_core_N * 64 >= n_tiles
    assert pc.per_core_N % pc.out_subblock_w == 0
    assert pc.out_subblock_h == 1


@pytest.mark.parametrize("name", ["ff1_3", "ff2", "qkv", "wo"])
def test_longer_prompts_keep_their_config(name, expect_error):
    args = _args()
    assert args.short_prefill_1d_matmul_config(256, DIM, DIM, 2, False) is None
    assert args.short_prefill_1d_matmul_config(129, DIM, DIM, 2, False) is None
    # Above the cutoff the getter must not return the 1D config. Its own branches need a full ModelArgs,
    # so stop at the first line they would reach.
    with expect_error(AttributeError, "has no attribute"):
        _getters(args)[name](256)


@pytest.mark.parametrize("name", ["ff1_3", "ff2", "qkv", "wo"])
def test_multi_device_keeps_its_config(name, expect_error):
    args = _args(num_devices=8)
    assert args.short_prefill_1d_matmul_config(128, DIM, DIM, 2, False) is None
    with expect_error(AttributeError, "has no attribute"):
        _getters(args)[name](128)


def test_galaxy_keeps_its_config():
    args = _args(num_devices=32, is_galaxy=True)
    assert args.short_prefill_1d_matmul_config(128, DIM, DIM, 2, False) is None


def test_minimal_matmul_prompts_are_not_rerouted():
    args = _args()
    args.use_minimal_prefill_matmul = lambda seq_len: True
    assert args.short_prefill_1d_matmul_config(128, DIM, DIM, 2, False) is None


def test_block_width_falls_back_to_a_divisor_of_k():
    args = _args()
    # K = 3 * 32 tiles: 4 does not divide 3, so the block width drops to 3.
    pc = args.short_prefill_1d_matmul_config(128, 96, 4096, 4, False)
    assert pc.in0_block_w == 3


# A 32-bit destination register halves the addressable register half-extents, so a matmul whose decoder
# accumulates in FP32 may not pick more than 4 out_subblock tiles. The MLP getters used to hardcode
# fp32_dest_acc_en=False, which let an FP32 decoder pick a 7-tile subblock and abort matmul validation.
@pytest.mark.parametrize(
    "name, op_group, fp32_fidelity",
    [
        ("ff1_3", OpGroup.LI_FF1_FF3, MathFidelitySetting.HIFI2),
        ("ff2", OpGroup.LI_FF2, MathFidelitySetting.HIFI2),
        # Any decoder accumulating in FP32 is enough: one program config serves them all.
        ("ff1_3", OpGroup.LI_FF1_FF3, MathFidelitySetting.HIFI4_FP32),
        ("ff2", OpGroup.LI_FF2, MathFidelitySetting.HIFI4_FP32),
    ],
)
def test_mlp_fp32_decoder_caps_the_subblock_at_four(name, op_group, fp32_fidelity):
    args = _args(op_settings={op_group: fp32_fidelity})
    pc = _getters(args)[name](128)
    assert isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig)
    # A 32-bit destination register allows at most 4 out_subblock tiles.
    assert 0 < pc.out_subblock_h * pc.out_subblock_w <= 4
    assert pc.out_subblock_w <= 4
    # The subblock stays a divisor of per_core_N, so matmul validation still accepts it.
    assert pc.per_core_N % pc.out_subblock_w == 0
    # It is the largest such divisor, i.e. nothing above 4 was chosen.
    assert pc.out_subblock_w == max(d for d in range(1, min(4, pc.per_core_N) + 1) if pc.per_core_N % d == 0)


@pytest.mark.parametrize("name, subblock_w", [("ff1_3", 7), ("ff2", 2)])
def test_mlp_default_fp16_decoder_keeps_the_wide_subblock(name, subblock_w):
    # The default is HIFI2_FP16 (fp32_dest_acc_en=False). FF1/FF3 fans 14336 N columns over 64 cores, so
    # per_core_N is 7 and the old hardcoded False picked all 7; FF2 fans 4096, so per_core_N is 2.
    args = _args(
        op_settings={
            OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16,
            OpGroup.LI_FF2: MathFidelitySetting.HIFI2_FP16,
        }
    )
    pc = _getters(args)[name](128)
    assert pc.per_core_N % pc.out_subblock_w == 0
    assert pc.out_subblock_w == subblock_w


def test_mlp_fp32_decoder_narrows_the_subblock_against_the_fp16_default():
    # Same 8B shapes, only the LI_FF1_FF3 fidelity differs: FP32 must not pick the FP16 subblock.
    fp16 = _getters(
        _args(
            op_settings={
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_FF2: MathFidelitySetting.HIFI2_FP16,
            }
        )
    )["ff1_3"](128)
    fp32 = _getters(_args(op_settings={OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2}))["ff1_3"](128)
    assert fp16.out_subblock_w == 7
    assert fp32.out_subblock_h * fp32.out_subblock_w <= 4


def test_mlp_fp32_setting_on_another_op_group_does_not_cap_the_subblock():
    # An FP32 accumulation the MLP matmuls do not use must not narrow their subblock.
    args = _args(op_settings={OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4})
    assert _getters(args)["ff1_3"](128).out_subblock_w == 7
    assert _getters(args)["ff2"](128).out_subblock_w == 2
