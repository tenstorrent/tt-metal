# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the fused decode gate and switches (tt/fused_decode/config.py).

fused_decode_layout_supported decides which decoder layers use the fused decode path. It must accept exactly the
validated envelope (gpt-oss-20b on Blackhole 1x4 at TP=4 or one chip at TP=1, gpt-oss-120b on 1x4, one token per
device) and keep every other configuration on the original decode path. No device is needed.
"""

from unittest.mock import MagicMock

import pytest

from models.demos.gpt_oss.tt.fused_decode.config import (
    DENSE_MOE_TOKENS,
    FUSED_DECODE_DRAM_BANKS,
    FUSED_DECODE_MIN_GRID,
    dense_moe_decode_layout_supported,
    fused_decode_layout_supported,
    fused_decode_supported,
)

# gpt-oss-20b on a QuietBox 2: Blackhole, mesh 1x4 = the whole 4-device system, TP=4, EP=1, 32 experts, batch 1,
# 11x10 grid, 8 DRAM banks.
SUPPORTED = dict(
    is_blackhole=True,
    mesh_shape=(1, 4),
    tp=4,
    ep=1,
    num_experts=32,
    use_throughput_experts=False,
    tokens=1,
    grid=(11, 10),
    dram_banks=8,
    system_devices=4,
)


def test_supported_envelope():
    assert fused_decode_layout_supported(**SUPPORTED)


def test_minimum_grid_is_supported():
    assert fused_decode_layout_supported(**{**SUPPORTED, "grid": FUSED_DECODE_MIN_GRID})


# One P150-like chip: mesh 1x1, TP=1 (any system size: there is no fabric ring).
ONE_CHIP = {**SUPPORTED, "mesh_shape": (1, 1), "tp": 1, "system_devices": 4}


def test_one_chip_envelope(monkeypatch):
    monkeypatch.delenv("GPT_OSS_FUSED_DECODE", raising=False)
    assert fused_decode_layout_supported(**ONE_CHIP)
    assert fused_decode_layout_supported(**{**ONE_CHIP, "system_devices": 1})


def test_switch_keeps_original_path(monkeypatch):
    monkeypatch.setenv("GPT_OSS_FUSED_DECODE", "0")
    assert not fused_decode_layout_supported(**ONE_CHIP)
    assert not fused_decode_layout_supported(**SUPPORTED)
    assert not fused_decode_layout_supported(**{**SUPPORTED, "num_experts": 128})


@pytest.mark.parametrize(
    "change",
    [{"tp": 4}, {"tokens": 32}, {"num_experts": 128}, {"dram_banks": FUSED_DECODE_DRAM_BANKS - 1}],
    ids=["tp_4", "batch_32", "gpt_oss_120b_experts", "seven_dram_banks"],
)
def test_one_chip_outside_envelope(change):
    assert not fused_decode_layout_supported(**{**ONE_CHIP, **change})


@pytest.mark.parametrize(
    "change",
    [
        {"is_blackhole": False},
        {"mesh_shape": (1, 1), "tp": 4},
        {"mesh_shape": (1, 2), "tp": 2},
        {"mesh_shape": (1, 8), "tp": 8},
        {"mesh_shape": (4, 8), "tp": 8, "ep": 4},
        {"tp": 2},
        {"ep": 4},
        {"num_experts": 64},
        {"use_throughput_experts": True},
        {"tokens": 32},
        {"grid": (FUSED_DECODE_MIN_GRID[0] - 1, 10)},
        {"grid": (11, FUSED_DECODE_MIN_GRID[1] - 1)},
        {"dram_banks": FUSED_DECODE_DRAM_BANKS - 1},
        {"system_devices": 8},
        {"system_devices": 32},
    ],
    ids=[
        "wormhole",
        "mesh_1x1_tp4",
        "mesh_1x2",
        "mesh_1x8",
        "mesh_4x8",
        "tp_2",
        "ep_4",
        "64_experts",
        "throughput_experts",
        "batch_32",
        "grid_too_narrow",
        "grid_too_short",
        "seven_dram_banks",
        "slice_of_8_device_system",
        "slice_of_galaxy",
    ],
)
def test_outside_envelope_keeps_original_path(change):
    assert not fused_decode_layout_supported(**{**SUPPORTED, **change})


def test_gpt_oss_120b_on_quietbox2(monkeypatch):
    """gpt-oss-120b (128 experts) takes the fused decode on 1x4, its experts streamed from the ND-sharded tensors
    prefill reads (no decode-only copies); one chip stays on the original path (the experts do not fit)."""
    from models.demos.gpt_oss.tt.fused_decode.config import fused_expert_cache_tag, nd_expert_dtype_names, nd_experts_on

    assert fused_decode_layout_supported(**{**SUPPORTED, "num_experts": 128})
    assert not fused_decode_layout_supported(**{**ONE_CHIP, "num_experts": 128})
    assert nd_experts_on(4, 128)
    assert not nd_experts_on(4, 32)
    assert nd_experts_on(1, 32)
    for env in ("GPT_OSS_ND_EXPERT_DTYPE", "GPT_OSS_ND_DOWN_DTYPE", "GPT_OSS_ND_LAYER_DTYPES", "GPT_OSS_EXPERT_BFP4"):
        monkeypatch.delenv(env, raising=False)
    assert nd_expert_dtype_names() == ("bfp8", "bfp8")
    assert fused_expert_cache_tag(4, 128) == "nd_bfp8"
    assert fused_expert_cache_tag(1, 32) == "nd_bfp8"
    monkeypatch.setenv("GPT_OSS_ND_DOWN_DTYPE", "bfp4")
    assert nd_expert_dtype_names() == ("bfp8", "bfp4")
    assert fused_expert_cache_tag(4, 128) == "nd_bfp8_downbfp4_mse"
    monkeypatch.setenv("GPT_OSS_ND_EXPERT_DTYPE", "bfp4")
    monkeypatch.setenv("GPT_OSS_ND_DOWN_DTYPE", "bfp8")
    assert nd_expert_dtype_names() == ("bfp4", "bfp8")
    assert fused_expert_cache_tag(4, 128) == "nd_bfp4_downbfp8_mse"
    monkeypatch.setenv("GPT_OSS_EXPERT_BFP4", "max")
    assert fused_expert_cache_tag(4, 128) == "nd_bfp4_downbfp8"


def test_expert_formats(monkeypatch, expect_error):
    """gpt-oss-20b on 1x4 streams K-grouped least-squared-error BFP4 decode copies by default (BFP8 as an option); the
    ND experts (one chip, gpt-oss-120b) round BFP4 N-grouped, and the original decode path keeps ttnn's rounding."""
    import ttnn
    from models.demos.gpt_oss.tt.bfp4_quant import bfp4_cache_tag, expert_bfp4_rounding
    from models.demos.gpt_oss.tt.fused_decode.config import expert_decode_weight_dtype, fused_expert_cache_tag

    for env in ("GPT_OSS_FUSED_EXPERT_DTYPE", "GPT_OSS_EXPERT_BFP4"):
        monkeypatch.delenv(env, raising=False)
    assert expert_decode_weight_dtype() == ttnn.bfloat4_b
    assert expert_bfp4_rounding(nd=False) == "kmse" and expert_bfp4_rounding(nd=True) == "mse"
    assert fused_expert_cache_tag(4, 32) == "bfp4_kmse"
    assert bfp4_cache_tag(ttnn.bfloat4_b, "kmse") == "_bfp4kmse" and bfp4_cache_tag(ttnn.bfloat8_b, "kmse") == ""
    assert bfp4_cache_tag(ttnn.bfloat4_b, "max") == ""
    monkeypatch.setenv("GPT_OSS_FUSED_EXPERT_DTYPE", "bfp8")
    assert expert_decode_weight_dtype() == ttnn.bfloat8_b
    assert fused_expert_cache_tag(4, 32) == "bfp8_kmse"  # (the prefill copies stay BFP4)
    monkeypatch.setenv("GPT_OSS_EXPERT_BFP4", "kmse")
    assert expert_bfp4_rounding(nd=True) == "mse"  # ND tensors cannot be K-grouped
    monkeypatch.setenv("GPT_OSS_EXPERT_BFP4", "max")
    monkeypatch.setenv("GPT_OSS_FUSED_EXPERT_DTYPE", "bfp4")
    assert fused_expert_cache_tag(4, 32) == "bfp4"  # the original decode copies' marker
    monkeypatch.setenv("GPT_OSS_EXPERT_BFP4", "nearest")
    with expect_error(ValueError, "GPT_OSS_EXPERT_BFP4"):
        expert_bfp4_rounding(nd=False)


def test_bfp4_rounding():
    """mxfp4_to_bfp4 (numba) == bfp4_round (torch); max-exponent rounding reproduces ttnn's packer grid; the
    least-squared-error choice is never worse than it per group; the K-grouped form groups along K."""
    import torch

    from models.demos.gpt_oss.tt.bfp4_quant import ROUNDINGS, bfp4_round, mxfp4_to_bfp4, mxfp4_to_bfp4_k

    torch.manual_seed(0)
    fp4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    w = fp4[torch.randint(0, 8, (64, 64))] * torch.sign(torch.randn(64, 64))
    w = (w * torch.ldexp(torch.ones(64, 1), torch.randint(-3, 3, (64, 1)))).to(torch.bfloat16)  # one scale per row
    q = mxfp4_to_bfp4(w)
    assert torch.equal(q.float(), bfp4_round(w, ROUNDINGS["mse"]))
    err = lambda a: (a.float() - w.float()).square().reshape(64, 4, 16).sum(-1)  # noqa: E731
    assert (err(q) <= err(bfp4_round(w, (0,))) + 1e-12).all()
    assert torch.equal(mxfp4_to_bfp4_k(w), mxfp4_to_bfp4(w.T.contiguous()).T)


def test_vocab_sharded_embedding():
    """The vocab-sharded embedding table (fused_decode/embed.py): on for 128 experts at TP > 1; device d's rows
    then a zero row; local ids map foreign tokens to the zero row."""
    import torch

    from models.demos.gpt_oss.tt.fused_decode.config import embed_vshard_on
    from models.demos.gpt_oss.tt.fused_decode.embed import vshard_rows, vshard_table

    assert embed_vshard_on(4, 128) and not embed_vshard_on(4, 32) and not embed_vshard_on(1, 128)
    V, H, tp = 10, 3, 4
    w = torch.arange(V * H, dtype=torch.float32).reshape(V, H) + 1
    vd = vshard_rows(V, tp)
    t = vshard_table(w, tp).reshape(tp, vd + 1, H)
    for d in range(tp):
        rows = range(d * vd, min((d + 1) * vd, V))
        for i, v in enumerate(rows):
            assert torch.equal(t[d, i], w[v])
        assert not t[d, len(rows) :].any()  # padding rows of the last device and every zero row


def test_no_mesh_config_keeps_original_path():
    # Without a mesh config the device is never queried.
    mesh_device = MagicMock()
    assert not fused_decode_supported(mesh_device, None, MagicMock(), False, 1)
    mesh_device.compute_with_storage_grid_size.assert_not_called()


# Batched decode MoE (experts/decode.py decode_forward_dense): the same system with one 32-token tile per device.
BATCHED = {**SUPPORTED, "tokens": DENSE_MOE_TOKENS}


def test_batched_envelope_takes_dense_moe_not_fused_decode():
    assert dense_moe_decode_layout_supported(**BATCHED)
    assert not fused_decode_layout_supported(**BATCHED)


def test_one_token_takes_fused_decode_not_dense_moe():
    assert not dense_moe_decode_layout_supported(**SUPPORTED)


@pytest.mark.parametrize(
    "change",
    [
        {"is_blackhole": False},
        {"mesh_shape": (1, 8), "tp": 8},
        {"ep": 4},
        {"num_experts": 128},
        {"use_throughput_experts": True},
        {"tokens": 64},
        {"tokens": 8},
        {"dram_banks": FUSED_DECODE_DRAM_BANKS - 1},
        {"system_devices": 8},
    ],
    ids=[
        "wormhole",
        "mesh_1x8",
        "ep_4",
        "gpt_oss_120b_experts",
        "throughput_experts",
        "batch_64",
        "batch_8",
        "seven_dram_banks",
        "slice_of_8_device_system",
    ],
)
def test_outside_batched_envelope_keeps_original_moe(change):
    assert not dense_moe_decode_layout_supported(**{**BATCHED, **change})
