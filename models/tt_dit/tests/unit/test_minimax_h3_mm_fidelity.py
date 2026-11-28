# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the matmul fidelities and all-gather matmul pins an H3 block and its attention pick.
Each is built with its sub-layers replaced by recorders, so only the module's own choices run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ttnn
from models.tt_dit.models.transformers.minimax_h3 import attention_minimax_h3 as am
from models.tt_dit.models.transformers.minimax_h3 import transformer_block_minimax_h3 as tb
from models.tt_dit.models.transformers.minimax_h3.agmm_config import AgmmMode

SDPA_FLAG = "MINIMAX_H3_SDPA_MM_FIDELITY"
FF_UP_FLAG = "MINIMAX_H3_FF_UP_MM_FIDELITY"
QKV_FLAG = "MINIMAX_H3_QKV_MM_FIDELITY"
AGMM_MODE_FLAG = "MINIMAX_H3_AGMM_MODE"

LOFI, HIFI2 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2
TINY = dict(hidden_size=8, num_heads=4, head_dim=2)
# The DiT's sizes, whose TP=4 layers are the pinned ones, and Blackhole's compute grid.
DIT = dict(hidden_size=5376, num_heads=56, head_dim=128)
DIT_FFN_DIM = 14336
BLACKHOLE = (12, 10)


def _stub(monkeypatch, module, *names, grid=(11, 10)):
    for name in names:
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ttnn, "init_device_compute_kernel_config", lambda arch, **kwargs: ttnn.WormholeComputeKernelConfig(**kwargs)
    )
    axis = SimpleNamespace(mesh_axis=1, factor=4)
    return dict(
        mesh_device=SimpleNamespace(arch=lambda: None, compute_with_storage_grid_size=lambda: ttnn.CoreCoord(*grid)),
        ccl_manager=SimpleNamespace(topology=ttnn.Topology.Ring, num_links=2),
        parallel_config=SimpleNamespace(tensor_parallel=axis, sequence_parallel=axis),
    )


def _block(monkeypatch, *, adaln_curve: bool = False, grid=(11, 10), sizes=TINY, ffn_dim: int = 16, **kwargs):
    """A block for a pruned (`adaln_curve`) or a full checkpoint, and the kwargs it gave its attention and its
    feed-forward."""
    attn, ff = {}, {}
    mesh = _stub(monkeypatch, tb, "DistributedRMSNorm", "ColParallelLinear", grid=grid)
    monkeypatch.setattr(tb, "MiniMaxH3Attention", lambda **kwargs: attn.update(kwargs))
    monkeypatch.setattr(tb, "ParallelFeedForward", lambda *args, **kwargs: ff.update(kwargs))
    block = tb.MiniMaxH3TransformerBlock(
        **sizes, ffn_dim=ffn_dim, time_embed_dim=8, adaln_curve=adaln_curve, **mesh, **kwargs
    )
    return block, attn, ff


def _attention(monkeypatch, *, grid=(11, 10), sizes=TINY, **kwargs):
    mesh = _stub(monkeypatch, am, "ColParallelLinear", "DistributedRMSNorm", "bf16_tensor", grid=grid)
    monkeypatch.setattr(am, "is_blackhole", lambda: False)
    return am.MiniMaxH3Attention(**sizes, **mesh, **kwargs)


@pytest.mark.parametrize(("adaln_curve", "expected"), [(True, LOFI), (False, None)])
def test_ring_sdpa_default(monkeypatch, adaln_curve, expected):
    """Pruned (10Eros) checkpoints default the ring SDPA matmuls to LoFi; the others leave the op's HiFi2."""
    assert _block(monkeypatch, adaln_curve=adaln_curve)[1]["sdpa_matmul_fidelity"] == expected


def test_ring_sdpa_flag_beats_default(monkeypatch):
    """hifi2 in the flag is the way back from the block's LoFi default."""
    monkeypatch.delenv(SDPA_FLAG, raising=False)
    assert _attention(monkeypatch, sdpa_matmul_fidelity=LOFI).sdpa_matmul_fidelity == LOFI
    monkeypatch.setenv(SDPA_FLAG, "hifi2")
    assert _attention(monkeypatch, sdpa_matmul_fidelity=LOFI).sdpa_matmul_fidelity == HIFI2


@pytest.mark.parametrize(
    ("adaln_curve", "flag", "expected"),
    [
        (True, None, AgmmMode.BALANCED),
        (False, None, AgmmMode.ACCURATE),
        (True, "accurate", AgmmMode.ACCURATE),
        (False, "balanced", AgmmMode.BALANCED),
    ],
)
def test_block_agmm_mode_on_blackhole(monkeypatch, adaln_curve, flag, expected):
    """On Blackhole a pruned (10Eros) checkpoint defaults to the BALANCED pins, which are the bucketed path's:
    ff1 without fp32 Dst accumulation, and the same mode handed to the attention. MINIMAX_H3_AGMM_MODE beats the
    default; ACCURATE leaves the bucketed path on its table."""
    monkeypatch.delenv(AGMM_MODE_FLAG, raising=False)
    if flag:
        monkeypatch.setenv(AGMM_MODE_FLAG, flag)
    block, attn, ff = _block(monkeypatch, adaln_curve=adaln_curve, grid=BLACKHOLE, sizes=DIT, ffn_dim=DIT_FFN_DIM)
    balanced = expected == AgmmMode.BALANCED
    assert attn["agmm_mode"] == expected
    assert (block._ff1_pins is not None) == balanced
    assert ff["ff1_compute_kernel_config"].fp32_dest_acc_en != balanced
    assert ff["ff2_compute_kernel_config"].fp32_dest_acc_en


def test_bucketed_attention_pins(monkeypatch):
    """On the bucketed path the DiT attention runs the mode's to_qkv pin and keeps to_out on the table; the token
    refiner's attention (replicated sequence, no mode from its caller) keeps the table for both."""
    dit = _attention(monkeypatch, grid=BLACKHOLE, sizes=DIT, agmm_mode=AgmmMode.BALANCED)
    assert not dit._qkv_pins["long"].compute_kernel_config.fp32_dest_acc_en
    assert dit._out_pins is None
    accurate = _attention(monkeypatch, grid=BLACKHOLE, sizes=DIT, agmm_mode=AgmmMode.ACCURATE)
    assert accurate._qkv_pins is None and accurate._out_pins is None
    monkeypatch.setenv(AGMM_MODE_FLAG, "balanced")
    refiner = _attention(monkeypatch, grid=BLACKHOLE, sizes=DIT, is_sequence_parallel=False)
    assert refiner._qkv_pins is None and refiner._out_pins is None


@pytest.mark.parametrize(
    ("adaln_curve", "value", "expected"),
    [(False, None, HIFI2), (False, "lofi", LOFI), (True, None, LOFI), (True, "hifi2", HIFI2)],
)
def test_ff_up_and_qkv_flags(monkeypatch, adaln_curve, value, expected):
    """On Blackhole a pruned (10Eros) checkpoint defaults the FF up and to_qkv matmuls to LoFi and a full one to
    HiFi2. The flags beat the default and move only these two matmuls, on a BALANCED pin too. The block reads both:
    an attention without a caller's fidelity (the token refiner's) keeps HiFi2."""
    for flag in (FF_UP_FLAG, QKV_FLAG):
        monkeypatch.delenv(flag, raising=False) if value is None else monkeypatch.setenv(flag, value)
    monkeypatch.setenv(AGMM_MODE_FLAG, "balanced")
    block, attn, ff = _block(monkeypatch, adaln_curve=adaln_curve, grid=BLACKHOLE, sizes=DIT, ffn_dim=DIT_FFN_DIM)
    assert ff["ff1_compute_kernel_config"].math_fidelity == attn["qkv_matmul_fidelity"] == expected
    assert ff["ff2_compute_kernel_config"].math_fidelity == block.mm_compute_kernel_config.math_fidelity == HIFI2
    assert _attention(monkeypatch, grid=BLACKHOLE, sizes=DIT).qkv_compute_kernel_config.math_fidelity == HIFI2
    attention = _attention(monkeypatch, grid=BLACKHOLE, sizes=DIT, qkv_matmul_fidelity=expected)
    kwargs = attention._agmm_kwargs(
        attention._qkv_pins,
        5376,
        5376,
        SimpleNamespace(padded_shape=[1, 1, 9600, 5376]),
        attention.qkv_compute_kernel_config,
    )
    assert kwargs["compute_kernel_config"].math_fidelity == expected
    assert attention.mm_compute_kernel_config.math_fidelity == HIFI2


def test_ff_up_and_qkv_default_off_blackhole(monkeypatch):
    """Off Blackhole a pruned checkpoint keeps both at HiFi2: LoFi was checked on Blackhole only."""
    for flag in (FF_UP_FLAG, QKV_FLAG):
        monkeypatch.delenv(flag, raising=False)
    _, attn, ff = _block(monkeypatch, adaln_curve=True)
    assert ff["ff1_compute_kernel_config"].math_fidelity == attn["qkv_matmul_fidelity"] == HIFI2
