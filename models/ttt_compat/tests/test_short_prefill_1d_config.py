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
from models.ttt_compat.tt.model_config import ModelArgs

# Llama-3.1-8B shapes
DIM = 4096
HIDDEN_DIM = 14336
N_HEADS = 32
HEAD_DIM = 128
QKV_SIZE = HEAD_DIM * (2 * 8 + N_HEADS)


def _args(num_devices=1, is_galaxy=False, minimal_above_128=True):
    args = SimpleNamespace(
        num_devices=num_devices,
        is_galaxy=is_galaxy,
        cluster_shape=[1, 1] if not is_galaxy else [4, 8],
        dim=DIM,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        qkv_size=QKV_SIZE,
        SHORT_PREFILL_1D_MAX_SEQ_LEN=ModelArgs.SHORT_PREFILL_1D_MAX_SEQ_LEN,
    )
    args.use_minimal_prefill_matmul = lambda seq_len: minimal_above_128 and seq_len > 128
    args.short_prefill_1d_matmul_config = MethodType(ModelArgs.short_prefill_1d_matmul_config, args)
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
    assert tuple(pc.compute_with_storage_grid_size) == (8, 8)
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
