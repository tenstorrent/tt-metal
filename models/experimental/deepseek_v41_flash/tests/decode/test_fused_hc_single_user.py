# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC and timing of the single-user ``fused_hyperconnection`` program in the layouts V4.1 decode feeds it.

V4.1 decode (``DeepSeekV4HyperConnection(row_major_streams=True)``) hands the op ROW_MAJOR streams
width-sharded ``[hc, D / 8]`` over the 8 collapse cores, the ``fn`` output as a ROW_MAJOR ``[1, 32]``
row on core (0, 0), and the previous sublayer's ``pre`` as ``pre_mix``. The op is run ``ITERS`` times so
a device-profiler capture (``TT_METAL_DEVICE_PROFILER=1``) has several samples.

``test_fused_hc_single_user_folded_mix`` checks the folded mix step (``mix_post`` / ``mix_comb`` /
``mix_sublayer_out``, with the ``fn`` dot in the program) against ``mix_streams`` followed by the same program.

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


# The decode layout's fn dot: hc * D / 256 cores in rows of 8 (or 10) below the HC row, fan-in one row.
_DOT_K_SLICE = 256


def _dot_grid(hc, hidden):
    num_dot_cores = hc * hidden // _DOT_K_SLICE
    width = 10 if num_dot_cores % 10 == 0 else 8
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(width - 1, num_dot_cores // width))})
    return grid, width


@pytest.mark.parametrize("sub_memory", ("dram", "l1_8", "l1_16"))
@pytest.mark.parametrize("collapsed_layout", (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), ids=("tile", "rm_out"))
def test_fused_hc_single_user_folded_mix(device, reset_seeds, sub_memory, collapsed_layout):
    """The previous sublayer's mix folded into the fn-dot program vs ``mix_streams`` then the same program: the
    mixed streams (fifth output) and post / comb / collapsed / pre. Twice, the second on the cached programs."""
    if device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip("the fused fn dot is Blackhole-only")
    hc, hidden, iters, eps, norm_eps = 4, 4096, 20, 1.0e-6, 1.0e-6
    pre_scale, post_scale, comb_scale = 0.7, 1.3, 2.0
    k, n = hc * hidden, ttnn.TILE_SIZE
    grid, fan_in = _dot_grid(hc, hidden)

    def to_tt(x, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)

    streams_config = width_sharded_l1_config(hc, hidden, device, num_cores=8, tile_height=1)
    if sub_memory == "dram":
        sub_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        sub_config = width_sharded_l1_config(1, hidden, device, num_cores=int(sub_memory[3:]), tile_height=1)

    # fn as the module stores it: [K, N], K in the collapse cores' (core, stream, column) order.
    w = torch.zeros(n, k)
    w[: (2 + hc) * hc] = torch.randn((2 + hc) * hc, k) * 0.02
    w_core_major = w.reshape(n, hc, 8, hidden // 8).permute(0, 2, 1, 3).reshape(n, k)
    fn_weight = to_tt(w_core_major.t().reshape(1, 1, k, n))
    biases = dict(
        pre_bias=to_tt(torch.randn(1, 1, 1, hc) * 0.5),
        post_bias=to_tt(torch.randn(1, 1, 1, hc) * 0.5),
        comb_bias=to_tt(torch.randn(1, 1, 1, hc * hc) * 0.5),
    )
    kwargs = dict(
        num_streams=hc,
        sinkhorn_iters=iters,
        pre_scale=pre_scale,
        post_scale=post_scale,
        comb_scale=comb_scale,
        eps=eps,
        collapsed_layout=collapsed_layout,
        fn_weight=fn_weight,
        fn_core_grid=grid,
        fn_reduce_fan_in=fan_in,
        fn_rms_norm_epsilon=norm_eps,
        **biases,
    )

    names = ("post", "comb", "collapsed", "pre", "streams")
    for step in range(2):
        streams = torch.randn(1, 1, hc, hidden).bfloat16().float()
        post = (torch.rand(1, 1, hc, 1) * 2).bfloat16().float()
        comb = torch.rand(1, 1, hc, hc)
        comb = (comb / comb.sum(-1, keepdim=True)).bfloat16().float()
        sub = torch.randn(1, 1, 1, hidden).bfloat16().float()
        pre_mix = (torch.rand(1, 1, 1, hc) + 0.1).bfloat16().float()
        expected_streams = torch.einsum("ji,jd->id", comb[0, 0], streams[0, 0]) + post[0, 0] * sub[0, 0]

        streams_tt = to_tt(streams, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=streams_config)
        post_tt, comb_tt = to_tt(post), to_tt(comb)
        sub_tt = to_tt(sub, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=sub_config)
        pre_mix_tt = to_tt(pre_mix)

        mixed_tt = ttnn.experimental.deepseek.mix_streams(post_tt, comb_tt, sub_tt, streams_tt)
        old = list(ttnn.experimental.deepseek.fused_hyperconnection(mixed_tt, pre_mix=pre_mix_tt, **kwargs))
        old.append(mixed_tt)
        new = ttnn.experimental.deepseek.fused_hyperconnection(
            streams_tt, pre_mix=pre_mix_tt, mix_post=post_tt, mix_comb=comb_tt, mix_sublayer_out=sub_tt, **kwargs
        )
        assert len(new) == 5, f"{len(new)} outputs, expected 5"
        assert new[4].memory_config() == streams_config and new[4].layout == ttnn.ROW_MAJOR_LAYOUT
        assert new[2].layout == collapsed_layout
        ttnn.synchronize_device(device)

        failures = []
        for name, got_tt, ref_tt in zip(names, new, old):
            got, ref = ttnn.to_torch(got_tt).float(), ttnn.to_torch(ref_tt).float()
            passing, pcc = comp_pcc(ref, got, PCC)
            logger.info(f"step {step} {name}: PCC vs mix_streams path {pcc}, max_abs_err {(ref - got).abs().max():.3e}")
            if not passing:
                failures.append(f"{name}: {pcc}")
        got_streams = ttnn.to_torch(new[4]).float().reshape(hc, hidden)
        passing, pcc = comp_pcc(expected_streams, got_streams, PCC)
        logger.info(f"step {step} streams vs torch: PCC {pcc}")
        if not passing:
            failures.append(f"streams vs torch: {pcc}")
        assert not failures, "; ".join(failures)


@pytest.mark.parametrize("hidden", (5120, 4096))
@pytest.mark.parametrize("fold_mix", (True, False), ids=("mix", "no_mix"))
@pytest.mark.parametrize("sharded", (False, True), ids=("dram", "sharded"))
def test_fused_hc_single_user_collapsed_norm(device, reset_seeds, hidden, fold_mix, sharded):
    """``collapsed_norm_weight`` (the FFN sublayer's RMSNorm fused onto the collapse) vs the same program without
    it followed by ``sharded_rmsnorm``, and vs torch. The other outputs must not change. Twice, the second on the
    cached programs."""
    if device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip("the fused fn dot and the fused norm are Blackhole-only")
    hc, iters, eps, norm_eps, rms_eps = 4, 20, 1.0e-6, 1.0e-6, 1.0e-6
    k, n = hc * hidden, ttnn.TILE_SIZE
    grid, fan_in = _dot_grid(hc, hidden)

    def to_tt(x, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)

    streams_config = width_sharded_l1_config(hc, hidden, device, num_cores=8, tile_height=1)
    w = torch.zeros(n, k)
    w[: (2 + hc) * hc] = torch.randn((2 + hc) * hc, k) * 0.02
    w_core_major = w.reshape(n, hc, 8, hidden // 8).permute(0, 2, 1, 3).reshape(n, k)
    kwargs = dict(
        num_streams=hc,
        sinkhorn_iters=iters,
        pre_scale=0.7,
        post_scale=1.3,
        comb_scale=2.0,
        eps=eps,
        fn_weight=to_tt(w_core_major.t().reshape(1, 1, k, n)),
        fn_core_grid=grid,
        fn_reduce_fan_in=fan_in,
        fn_rms_norm_epsilon=norm_eps,
        pre_bias=to_tt(torch.randn(1, 1, 1, hc) * 0.5),
        post_bias=to_tt(torch.randn(1, 1, 1, hc) * 0.5),
        comb_bias=to_tt(torch.randn(1, 1, 1, hc * hc) * 0.5),
    )
    gamma = (torch.rand(hidden) + 0.5).bfloat16().float()
    gamma_tile = to_tt(gamma.reshape(1, 1, 1, hidden))
    gamma_row = to_tt(gamma.reshape(1, 1, 1, hidden), layout=ttnn.ROW_MAJOR_LAYOUT)

    names = ("post", "comb", "collapsed", "pre", "streams")
    for step in range(2):
        # Streams with a large common scale, so a missing or wrong 1 / rms shows.
        streams = (torch.randn(1, 1, hc, hidden) * 3.0).bfloat16().float()
        pre_mix = (torch.rand(1, 1, 1, hc) + 0.1).bfloat16().float()
        streams_tt = to_tt(streams, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=streams_config)
        call = dict(kwargs, pre_mix=to_tt(pre_mix))
        if fold_mix:
            post = (torch.rand(1, 1, hc, 1) * 2).bfloat16().float()
            comb = torch.rand(1, 1, hc, hc)
            comb = (comb / comb.sum(-1, keepdim=True)).bfloat16().float()
            sub = torch.randn(1, 1, 1, hidden).bfloat16().float()
            call.update(
                mix_post=to_tt(post),
                mix_comb=to_tt(comb),
                mix_sublayer_out=to_tt(sub, layout=ttnn.ROW_MAJOR_LAYOUT),
            )

        old = list(ttnn.experimental.deepseek.fused_hyperconnection(streams_tt, **call))
        old_normed = ttnn.experimental.deepseek.sharded_rmsnorm(
            old[2],
            gamma_tile,
            epsilon=rms_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        new = ttnn.experimental.deepseek.fused_hyperconnection(
            streams_tt,
            collapsed_layout=ttnn.ROW_MAJOR_LAYOUT,
            collapsed_norm_weight=gamma_row,
            collapsed_norm_epsilon=rms_eps,
            collapsed_norm_sharded=sharded,
            **call,
        )
        assert len(new) == len(old), f"{len(new)} outputs, expected {len(old)}"
        want_config = (
            width_sharded_l1_config(1, hidden, device, num_cores=8, tile_height=1)
            if sharded
            else ttnn.DRAM_MEMORY_CONFIG
        )
        assert new[2].layout == ttnn.ROW_MAJOR_LAYOUT and new[2].memory_config() == want_config, new[2].memory_config()
        ttnn.synchronize_device(device)

        failures = []
        for i, (got_tt, ref_tt) in enumerate(zip(new, old)):
            if i == 2:
                continue
            got, ref = ttnn.to_torch(got_tt).float(), ttnn.to_torch(ref_tt).float()
            passing, pcc = comp_pcc(ref, got, PCC)
            logger.info(f"step {step} {names[i]}: PCC vs unfused {pcc}")
            if not passing or not torch.equal(got, ref):
                failures.append(f"{names[i]} changed: {pcc}, max_abs_err {(ref - got).abs().max():.3e}")
        got = ttnn.to_torch(new[2]).float().reshape(hidden)
        ref = ttnn.to_torch(old_normed).float().reshape(hidden)
        collapsed = ttnn.to_torch(old[2]).float()[..., 0, :].reshape(hidden)
        expected = collapsed * torch.rsqrt(collapsed.pow(2).mean() + rms_eps) * gamma
        for label, target in (("sharded_rmsnorm", ref), ("torch", expected)):
            passing, pcc = comp_pcc(target, got, PCC)
            logger.info(
                f"step {step} normed collapsed vs {label}: PCC {pcc}, max_abs_err {(target - got).abs().max():.3e}"
            )
            if not passing:
                failures.append(f"normed vs {label}: {pcc}")
        assert not failures, "; ".join(failures)
