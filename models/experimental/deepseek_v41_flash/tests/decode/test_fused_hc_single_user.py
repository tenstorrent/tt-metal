# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC and timing of the single-user ``fused_hyperconnection`` program in the layouts V4.1 decode feeds it.

V4.1 decode (``DeepSeekV4HyperConnection(row_major_streams=True)``) hands the op ROW_MAJOR streams
width-sharded ``[hc, D / 8]`` over the 8 collapse cores, the ``fn`` output as a ROW_MAJOR ``[1, 32]``
row on core (0, 0), and the previous sublayer's ``pre`` as ``pre_mix``. The op is run ``ITERS`` times so
a device-profiler capture (``TT_METAL_DEVICE_PROFILER=1``) has several samples.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_fused_hc_single_user.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tests.decode.test_fused_hyperconnection_op import _torch_reference
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config

PCC = 0.999
ITERS = 8


@pytest.mark.parametrize("use_pre_mix", (True, False), ids=("pre_mix", "own_pre"))
@pytest.mark.parametrize("comb_bias_shape", ("row", "square"))
@pytest.mark.parametrize("hidden", (4096, 2048))
@pytest.mark.parametrize("bias_memory", ("dram", "l1"))
@pytest.mark.parametrize("collapsed_layout", (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), ids=("tile", "rm_out"))
def test_fused_hc_single_user_row_major(
    device, reset_seeds, use_pre_mix, comb_bias_shape, hidden, bias_memory, collapsed_layout
):
    hc, iters, eps = 4, 20, 1.0e-6
    pre_scale, post_scale, comb_scale = 0.7, 1.3, 2.0

    streams = torch.randn(1, 1, hc, hidden)
    fused_w = torch.zeros(1, 1, 1, ttnn.TILE_SIZE)
    fused_w[..., : (2 + hc) * hc] = torch.randn(1, 1, 1, (2 + hc) * hc)
    pre_b = torch.randn(1, 1, 1, hc) * 0.5
    post_b = torch.randn(1, 1, 1, hc) * 0.5
    comb_b = torch.randn(1, 1, 1, hc * hc) * 0.5
    pre_mix = torch.rand(1, 1, 1, hc) + 0.1 if use_pre_mix else None

    bf = lambda x: x.bfloat16().float()
    refs = _torch_reference(
        bf(streams),
        bf(fused_w[..., : (2 + hc) * hc]),
        bf(pre_b),
        bf(post_b),
        bf(comb_b),
        hc,
        iters,
        pre_scale,
        post_scale,
        comb_scale,
        eps,
        pre_mix=None if pre_mix is None else bf(pre_mix),
    )

    def to_tt(x, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)

    streams_tt = to_tt(
        streams,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=width_sharded_l1_config(hc, hidden, device, num_cores=8, tile_height=1),
    )
    fused_w_tt = to_tt(
        fused_w,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=width_sharded_l1_config(1, ttnn.TILE_SIZE, device, num_cores=1, tile_height=1),
    )
    bias_config = ttnn.DRAM_MEMORY_CONFIG if bias_memory == "dram" else ttnn.L1_MEMORY_CONFIG
    comb_b_tt = to_tt(comb_b if comb_bias_shape == "row" else comb_b.reshape(1, 1, hc, hc), memory_config=bias_config)
    kwargs = dict(
        fused_w=fused_w_tt,
        pre_bias=to_tt(pre_b, memory_config=bias_config),
        post_bias=to_tt(post_b, memory_config=bias_config),
        comb_bias=comb_b_tt,
        num_streams=hc,
        sinkhorn_iters=iters,
        pre_scale=pre_scale,
        post_scale=post_scale,
        comb_scale=comb_scale,
        eps=eps,
        pre_mix=None if pre_mix is None else to_tt(pre_mix),
        collapsed_layout=collapsed_layout,
    )

    for _ in range(ITERS):
        outputs = ttnn.experimental.deepseek.fused_hyperconnection(streams_tt, **kwargs)
    ttnn.synchronize_device(device)
    if collapsed_layout == ttnn.ROW_MAJOR_LAYOUT:
        # The layout sharded_rmsnorm writes, which attention's tokens gather reads.
        assert outputs[2].layout == ttnn.ROW_MAJOR_LAYOUT
        assert outputs[2].memory_config() == width_sharded_l1_config(1, hidden, device, num_cores=8, tile_height=1)

    names = ("post", "comb", "collapsed", "pre")
    num_outputs = 4 if use_pre_mix else 3
    assert len(outputs) == num_outputs
    failures = []
    for name, got_tt, ref in zip(names, outputs, refs[:num_outputs]):
        got_full = ttnn.to_torch(got_tt).float()
        got = got_full.reshape(-1)[: ref.numel()].reshape(ref.shape) if name == "collapsed" else got_full
        got = got.reshape(ref.shape)
        passing, pcc = comp_pcc(ref, got, PCC)
        max_err = (ref - got).abs().max().item()
        logger.info(f"{name}: PCC {pcc} max_abs_err {max_err:.3e}")
        if not passing:
            failures.append(f"{name}: {pcc}")
        if name == "comb":
            row_err = (got.sum(-1) - 1).abs().max().item()
            col_err = (got.sum(-2) - 1).abs().max().item()
            logger.info(f"comb row-sum err {row_err:.3e} col-sum err {col_err:.3e}")
    assert not failures, "; ".join(failures)

    if hasattr(ttnn, "ReadDeviceProfiler"):
        ttnn.ReadDeviceProfiler(device)
