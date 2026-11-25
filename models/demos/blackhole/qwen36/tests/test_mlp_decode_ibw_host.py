# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the K blocks of the 1D decode MLP programs (tt/model_config.py Qwen36ModelArgs).

* At the Qwen3.8-27B shape (dim 5120, 160 K tiles) the gate (SILU fused) and up programs take 16-tile K blocks; their
  grid, per-core M and N and fused activation are the ones of the 8-tile programs.
* QWEN36_MLP_DECODE_IBW caps the gate and up K blocks (8 gives the 8-tile programs back).
* The down program and the other decode programs keep the 8-tile cap.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.model_config import MLP_DECODE_IBW_ENV, Qwen36ModelArgs

DIM = 5120
HIDDEN = 17408

# Qwen3.8-27B: the attributes Qwen36ModelArgs._init_tp_config reads.
_QWEN38 = dict(
    n_heads=24,
    n_kv_heads=4,
    head_dim=256,
    dim=DIM,
    hidden_dim=HIDDEN,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_num_value_heads=48,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    linear_q_dim=2048,
    linear_k_dim=2048,
    linear_v_dim=6144,
    max_batch_size=8,
    moe_num_experts=0,
)


@pytest.fixture
def progcfg_fields(monkeypatch):
    monkeypatch.setattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", lambda **kw: SimpleNamespace(**kw))


def _tp_args(tp, grid=(11, 10)):
    args = SimpleNamespace(num_devices=tp, **_QWEN38)
    mesh = SimpleNamespace(shape=(1, tp), compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid[0], y=grid[1]))
    Qwen36ModelArgs._init_tp_config(args, mesh)
    return args


def _eight_tile_program(tp, grid_w, **kw):
    return tpc.create_matmul_1d_decode_progcfg(1, DIM, HIDDEN // tp, num_cores=44, grid_w=grid_w, **kw)


@pytest.mark.parametrize("tp, grid", [(8, (11, 10)), (4, (11, 10)), (8, (8, 8))], ids=["tp8", "tp4", "tp8-wormhole"])
def test_the_gate_and_up_programs_take_16_tile_k_blocks(monkeypatch, progcfg_fields, tp, grid):
    monkeypatch.delenv(MLP_DECODE_IBW_ENV, raising=False)
    args = _tp_args(tp, grid)
    gate, up = args.mlp_w1_decode_1d_progcfg, args.mlp_w3_decode_1d_progcfg
    for pc, ref in (
        (gate, _eight_tile_program(tp, grid[0], fused_activation=ttnn.UnaryOpType.SILU)),
        (up, _eight_tile_program(tp, grid[0])),
    ):
        assert pc.in0_block_w == 16 and (DIM // 32) % pc.in0_block_w == 0
        assert ref.in0_block_w == 8
        assert {k: v for k, v in vars(pc).items() if k != "in0_block_w"} == {
            k: v for k, v in vars(ref).items() if k != "in0_block_w"
        }


def test_the_environment_variable_caps_the_gate_and_up_k_blocks(monkeypatch, progcfg_fields):
    monkeypatch.setenv(MLP_DECODE_IBW_ENV, "8")
    args = _tp_args(8)
    assert args.mlp_w1_decode_1d_progcfg.in0_block_w == 8
    assert args.mlp_w3_decode_1d_progcfg.in0_block_w == 8


def test_the_other_decode_programs_keep_the_8_tile_cap(monkeypatch, progcfg_fields):
    monkeypatch.delenv(MLP_DECODE_IBW_ENV, raising=False)
    args = _tp_args(8)
    down_ref = tpc.create_matmul_1d_decode_progcfg(1, HIDDEN // 8, DIM, num_cores=33, grid_w=11)
    assert vars(args.mlp_w2_decode_1d_progcfg) == vars(down_ref)
    for name in (
        "attn_qkv_decode_1d_progcfg",
        "gdn_qkvz_decode_1d_progcfg",
        "attn_wo_decode_1d_progcfg",
        "gdn_out_decode_1d_progcfg",
    ):
        assert getattr(args, name).in0_block_w <= 8, name
