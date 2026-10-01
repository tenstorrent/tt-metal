# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""2D-mcast matmul with two in1 senders per column (MatmulMultiCoreReuseMultiCastProgramConfig.in1_dual_sender).

The top-row and the bottom-row core of every column each read and multicast half of the K rows of every in1 block.
Same blocks, same K order, same CB layout, so the result must be bit-identical to the single-sender program."""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32


def _cfg(gx, gy, bw, pcm, pcn, obh, obw, sh, sw, dual, **kw):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=obh,
        out_block_w=obw,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
        in1_dual_sender=dual,
        **kw,
    )


def _ckc(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _tensors(device, m, k, n, in0_dtype):
    torch.manual_seed(0)
    in0 = torch.randn(1, 1, m, k).bfloat16()
    in1 = (torch.randn(1, 1, k, n) * 0.05).bfloat16()
    in0_t = ttnn.from_torch(
        in0, dtype=in0_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    in1_t = ttnn.from_torch(
        in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return in0, in1, in0_t, in1_t


# (name, gx, gy, M, K, N, in0_block_w, per_core_N, out_block_w, out_subblock (h, w), out dtype)
# per_core_M = M / 32 / gy; out_block_h = per_core_M. out_block_w < per_core_N gives several out blocks per core (with a
# bf8 output the partials CB differs from the output CB, which also enables the out-block in1 lookahead).
CASES = [
    # the SP prefill shapes (11 columns, 8 rows, per_core_M 4)
    ("out_2048", 11, 8, 1024, 2048, 2048, 16, 6, 6, (1, 6), ttnn.bfloat16),
    ("down_6144", 11, 8, 1024, 6144, 2048, 16, 6, 6, (1, 6), ttnn.bfloat16),
    ("zba_n2112", 11, 8, 1024, 2048, 2112, 16, 6, 6, (1, 6), ttnn.bfloat16),  # 66 tiles = 11 x 6, no N padding
    ("qkv_n3072", 11, 8, 1024, 2048, 3072, 16, 9, 9, (2, 3), ttnn.bfloat16),
    ("in_6144", 11, 8, 1024, 2048, 6144, 16, 18, 18, (1, 6), ttnn.bfloat16),
    ("out_bw8_bf8", 11, 8, 1024, 2048, 2048, 8, 6, 6, (1, 6), ttnn.bfloat8_b),
    # several out blocks per core (lookahead active with a bf8 output)
    ("outblocks_bf8", 11, 8, 1024, 2048, 2048, 16, 6, 3, (1, 3), ttnn.bfloat8_b),
    ("outblocks_bf16", 11, 8, 1024, 2048, 2048, 16, 6, 3, (1, 3), ttnn.bfloat16),
    # other geometries: rows 3 / 4, columns 1 / 2 / 3, small in0_block_w
    ("g4x3", 4, 3, 384, 512, 512, 4, 4, 4, (1, 4), ttnn.bfloat16),
    ("g4x3_bw2", 4, 3, 384, 512, 512, 2, 4, 2, (1, 2), ttnn.bfloat8_b),
    ("g2x4", 2, 4, 512, 256, 256, 2, 4, 4, (1, 4), ttnn.bfloat16),
    ("g1x3", 1, 3, 384, 256, 128, 2, 4, 4, (1, 4), ttnn.bfloat16),
    ("g3x5", 3, 5, 640, 512, 384, 8, 4, 4, (2, 2), ttnn.bfloat16),
    ("g8x8_blocks", 8, 8, 512, 512, 1024, 4, 4, 2, (1, 2), ttnn.bfloat8_b),
]


def _skip_unless_grid(device, gx, gy):
    g = device.compute_with_storage_grid_size()
    if g.x < gx or g.y < gy:
        pytest.skip(f"needs a {gx}x{gy} worker grid, have {g.x}x{g.y}")


@pytest.mark.skipif(
    not is_blackhole(), reason="in1_dual_sender: the bottom sender's wrapped NOC_0 multicast is Blackhole-only"
)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_matmul_in1_dual_sender(device, case):
    name, gx, gy, m, k, n, bw, pcn, obw, (sh, sw), out_dtype = case
    _skip_unless_grid(device, gx, gy)
    pcm = m // TILE // gy
    in0, in1, in0_t, in1_t = _tensors(device, m, k, n, ttnn.bfloat16)
    outs = {}
    for dual in (False, True):
        cfg = _cfg(gx, gy, bw, pcm, pcn, pcm, obw, sh, sw, dual)
        assert cfg.in1_dual_sender == dual
        # twice: the second call is a program cache hit (runtime args of the descriptor are re-patched)
        for _ in range(2):
            out = ttnn.matmul(
                in0_t,
                in1_t,
                program_config=cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=out_dtype,
                compute_kernel_config=_ckc(device),
            )
            outs[dual] = ttnn.to_torch(out)
            ttnn.deallocate(out)
    assert torch.equal(outs[False], outs[True]), f"{name}: dual sender output differs from the single sender"
    assert_with_pcc(in0.float() @ in1.float(), outs[True].float(), 0.998)


@pytest.mark.skipif(
    not is_blackhole(), reason="in1_dual_sender: the bottom sender's wrapped NOC_0 multicast is Blackhole-only"
)
@pytest.mark.parametrize("glu_last_block", [True, False])
def test_matmul_in1_dual_sender_fuse_swiglu(device, glu_last_block):
    """fused SwiGLU gate|up (3 out blocks per core -> the out-block in1 lookahead): dual == single, bit for bit."""
    gx, gy, m, k, n = 11, 8, 1024, 2048, 12288
    _skip_unless_grid(device, gx, gy)
    in0, in1, in0_t, in1_t = _tensors(device, m, k, n, ttnn.bfloat16)
    outs = {}
    for dual in (False, True):
        cfg = _cfg(
            gx,
            gy,
            16,
            4,
            36,
            4,
            12,
            1,
            6,
            dual,
            fuse_swiglu=True,
            glu_last_block=glu_last_block,
            glu_sfpu_on_pack=glu_last_block,
        )
        out = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=_ckc(device),
        )
        outs[dual] = ttnn.to_torch(out)
        ttnn.deallocate(out)
    assert outs[True].shape[-1] == n // 2
    assert torch.equal(outs[False], outs[True]), "fuse_swiglu: dual sender output differs from the single sender"


@pytest.mark.skipif(not is_blackhole(), reason="in1_dual_sender: Blackhole-only")
def test_matmul_in1_dual_sender_rejected_configs(device, expect_error):
    gx, gy = 4, 4
    _skip_unless_grid(device, gx, gy)
    m, k, n = 512, 512, 512
    _, _, in0_t, in1_t = _tensors(device, m, k, n, ttnn.bfloat16)
    ckc = _ckc(device)

    def run(cfg, **kw):
        return ttnn.matmul(
            in0_t, in1_t, program_config=cfg, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckc, **kw
        )

    # odd in0_block_w
    with expect_error(RuntimeError, "even in0_block_w"):
        run(_cfg(gx, gy, 1, 4, 4, 4, 4, 1, 4, True))
    # only 2 core rows (M = 256 over per_core_M 4 -> 2 rows)
    _, _, in0_s, _ = _tensors(device, 256, k, n, ttnn.bfloat16)
    with expect_error(RuntimeError, "at least 3 core rows"):
        ttnn.matmul(in0_s, in1_t, program_config=_cfg(gx, 2, 4, 4, 4, 4, 4, 1, 4, True), compute_kernel_config=ckc)
    # bias
    bias = ttnn.from_torch(
        torch.randn(1, 1, 1, n).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(RuntimeError, "does not support bias"):
        ttnn.linear(
            in0_t, in1_t, bias=bias, program_config=_cfg(gx, gy, 4, 4, 4, 4, 4, 1, 4, True), compute_kernel_config=ckc
        )
