# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The single-user ``fused_hyperconnection`` program (fn dot in the program) on a DSpark block of ``rows`` rows in one
call, vs the one-row program run once per row and vs torch.

Block layouts (see ``hc_block_rows`` in fused_single_user_types.hpp): streams TILE (rows, 32) ``[1, 1, hc * rows, D]``
stream-major (row s * rows + j), WIDTH_SHARDED on the 8 collapse cores; pre_mix / mix_post / mix_comb and the post /
comb / pre outputs one 32x32 tile per row (dim 1); mix_sublayer_out and the norm gain (rows, 32) blocks; collapsed a
(rows, 32) block on the collapse cores, or with the fused norm to DRAM a ROW_MAJOR ``[1, 1, 1, rows * D]`` row in
``tile_rows`` byte order; the replica one (rows, 32) block per core.

    pytest tests/ttnn/unit_tests/operations/experimental/deepseek/dspark_rows/test_fused_hyperconnection_rows.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tests.decode.test_fused_hyperconnection_op import _torch_reference
from models.experimental.deepseek_v41_flash.tt.decode.persistent_moe import tile_rows, untile_rows

HC, ITERS, EPS, NORM_EPS, RMS_EPS = 4, 20, 1.0e-6, 1.0e-6, 1.0e-6
PRE_SCALE, POST_SCALE, COMB_SCALE = 0.7, 1.3, 2.0
COLLAPSE = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
# attention_qkv_decode's token replica grid.
REPLICA_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9))})
DOT_K_SLICE = 256
PCC = 0.999


def _dot_grid(hidden):
    num = HC * hidden // DOT_K_SLICE
    width = 10 if num % 10 == 0 else 8
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(width - 1, num // width))}), width


def _width_sharded(h, w):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(COLLAPSE, [h, w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _bf(x):
    return x.bfloat16().float()


class Case:
    """Random inputs for `rows` rows: per row streams [hc, D], mix post / comb / sublayer output, pre_mix."""

    def __init__(self, hidden, rows, seed=0):
        g = torch.Generator().manual_seed(seed)
        n, k = ttnn.TILE_SIZE, HC * hidden
        self.hidden, self.rows = hidden, rows
        w = torch.zeros(n, k)
        w[: (2 + HC) * HC] = torch.randn((2 + HC) * HC, k, generator=g) * 0.02
        self.w = _bf(w)  # [N, K], K stream-major
        self.w_core_major = self.w.reshape(n, HC, 8, hidden // 8).permute(0, 2, 1, 3).reshape(n, k)
        self.pre_b = _bf(torch.randn(1, 1, 1, HC, generator=g) * 0.5)
        self.post_b = _bf(torch.randn(1, 1, 1, HC, generator=g) * 0.5)
        self.comb_b = _bf(torch.randn(1, 1, 1, HC * HC, generator=g) * 0.5)
        self.gamma = _bf(torch.rand(hidden, generator=g) + 0.5)
        # Rows with different scales, so a row mixed up with another shows.
        scale = (1.0 + torch.arange(rows, dtype=torch.float32)).reshape(rows, 1, 1)
        self.streams = _bf(torch.randn(rows, HC, hidden, generator=g) * scale)  # [rows, hc, D]
        self.post = _bf(torch.rand(rows, HC, 1, generator=g) * 2)
        comb = torch.rand(rows, HC, HC, generator=g)
        self.comb = _bf(comb / comb.sum(-1, keepdim=True))
        self.sub = _bf(torch.randn(rows, hidden, generator=g) * scale.reshape(rows, 1))
        self.pre_mix = _bf(torch.rand(rows, 1, HC, generator=g) + 0.1)

    def reference(self, j, mix, norm):
        """Torch: (post, comb, collapsed (normed with norm), pre, mixed streams) of row j."""
        streams = self.streams[j]
        if mix:
            streams = torch.einsum("ji,jd->id", self.comb[j], streams) + self.post[j] * self.sub[j]
        x = streams.reshape(-1)
        fused_w = (x * torch.rsqrt(x.pow(2).mean() + NORM_EPS)) @ self.w.t()
        post, comb, collapsed, pre = _torch_reference(
            streams.reshape(1, 1, HC, self.hidden),
            fused_w[: (2 + HC) * HC].reshape(1, 1, 1, -1),
            self.pre_b,
            self.post_b,
            self.comb_b,
            HC,
            ITERS,
            PRE_SCALE,
            POST_SCALE,
            COMB_SCALE,
            EPS,
            pre_mix=self.pre_mix[j].reshape(1, 1, 1, HC),
        )
        collapsed = collapsed.reshape(self.hidden)
        if norm:
            collapsed = collapsed * torch.rsqrt(collapsed.pow(2).mean() + RMS_EPS) * self.gamma
        return post.reshape(HC, 1), comb.reshape(HC, HC), collapsed, pre.reshape(1, HC), streams


def _common_kwargs(device, case, bias_memory=ttnn.DRAM_MEMORY_CONFIG):
    def to_tt(x, memory_config=bias_memory):
        return ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

    grid, fan_in = _dot_grid(case.hidden)
    k, n = HC * case.hidden, ttnn.TILE_SIZE
    return dict(
        pre_bias=to_tt(case.pre_b),
        post_bias=to_tt(case.post_b),
        comb_bias=to_tt(case.comb_b),
        num_streams=HC,
        sinkhorn_iters=ITERS,
        pre_scale=PRE_SCALE,
        post_scale=POST_SCALE,
        comb_scale=COMB_SCALE,
        eps=EPS,
        fn_weight=to_tt(case.w_core_major.t().reshape(1, 1, k, n), ttnn.DRAM_MEMORY_CONFIG),
        fn_core_grid=grid,
        fn_reduce_fan_in=fan_in,
        fn_rms_norm_epsilon=NORM_EPS,
    )


def _block_call(device, case, common, h, mix, norm, replica, sub_format="tile"):
    """The block op's kwargs and streams for `case`, padded to h rows (pad rows repeat the last row)."""
    rows, d = case.rows, case.hidden
    idx = [min(j, rows - 1) for j in range(h)]
    tile = ttnn.Tile((h, 32))

    def to_tt(x, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, tile=None):
        return ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config, tile=tile
        )

    streams = case.streams[idx].permute(1, 0, 2).reshape(1, 1, HC * h, d)  # stream-major
    streams_tt = to_tt(streams, memory_config=_width_sharded(HC * h, d // 8), tile=tile)
    kwargs = dict(common, pre_mix=to_tt(case.pre_mix[idx].reshape(1, h, 1, HC)))
    if mix:
        sub = case.sub[idx]
        if sub_format == "tile":
            sub_tt = to_tt(sub.reshape(1, 1, h, d), tile=tile)
        else:
            sub_tt = to_tt(tile_rows(sub).reshape(1, 1, 1, h * d), layout=ttnn.ROW_MAJOR_LAYOUT)
        kwargs.update(
            mix_post=to_tt(case.post[idx].reshape(1, h, HC, 1)),
            mix_comb=to_tt(case.comb[idx].reshape(1, h, HC, HC)),
            mix_sublayer_out=sub_tt,
        )
    if norm is not None:
        gamma = case.gamma.reshape(1, d).expand(h, d)
        kwargs.update(
            collapsed_norm_weight=to_tt(gamma.reshape(1, 1, h, d), tile=tile),
            collapsed_norm_epsilon=RMS_EPS,
            collapsed_norm_sharded=norm == "sharded",
        )
    if replica:
        kwargs["collapsed_replica_grid"] = REPLICA_GRID
    return streams_tt, kwargs


def _row_call(device, case, common, j, mix, norm, replica):
    """The one-row op's kwargs and streams for row j."""
    d = case.hidden

    def to_tt(x, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)

    streams_tt = to_tt(case.streams[j].reshape(1, 1, HC, d), ttnn.ROW_MAJOR_LAYOUT, _width_sharded(HC, d // 8))
    kwargs = dict(common, pre_mix=to_tt(case.pre_mix[j].reshape(1, 1, 1, HC)), collapsed_layout=ttnn.ROW_MAJOR_LAYOUT)
    if mix:
        kwargs.update(
            mix_post=to_tt(case.post[j].reshape(1, 1, HC, 1)),
            mix_comb=to_tt(case.comb[j].reshape(1, 1, HC, HC)),
            mix_sublayer_out=to_tt(case.sub[j].reshape(1, 1, 1, d), ttnn.ROW_MAJOR_LAYOUT),
        )
    if norm is not None:
        kwargs.update(
            collapsed_norm_weight=to_tt(case.gamma.reshape(1, 1, 1, d), ttnn.ROW_MAJOR_LAYOUT),
            collapsed_norm_epsilon=RMS_EPS,
            collapsed_norm_sharded=norm == "sharded",
        )
    if replica:
        kwargs["collapsed_replica_grid"] = REPLICA_GRID
    return streams_tt, kwargs


def _block_outputs(outs, case, h, mix, norm, replica):
    """Per row: (post, comb, collapsed, pre, mixed streams or None, replica rows [cores, D] or None)."""
    d = case.hidden
    post = ttnn.to_torch(outs[0]).float()
    comb = ttnn.to_torch(outs[1]).float()
    pre = ttnn.to_torch(outs[3]).float()
    if norm == "dram":
        assert outs[2].layout == ttnn.ROW_MAJOR_LAYOUT and list(outs[2].shape) == [1, 1, 1, h * d]
        collapsed = untile_rows(ttnn.to_torch(outs[2]).float().reshape(-1), h)
    else:
        assert outs[2].layout == ttnn.TILE_LAYOUT and tuple(outs[2].tile.tile_shape) == (h, 32)
        collapsed = ttnn.to_torch(outs[2]).float().reshape(h, d)
    mixed = ttnn.to_torch(outs[4]).float().reshape(HC, h, d) if mix else None
    rep = ttnn.to_torch(outs[-1]).float().reshape(-1, h, d) if replica else None
    return [
        (
            post[0, j],
            comb[0, j],
            collapsed[j],
            pre[0, j],
            None if mixed is None else mixed[:, j],
            None if rep is None else rep[:, j],
        )
        for j in range(case.rows)
    ]


CONFIGS = {
    # ATTN_HC: the previous layer's FFN mix folded, the input RMSNorm fused and kept on the collapse cores, the
    # token replica multicast onto attention_qkv_decode's grid.
    "attn": dict(mix=True, norm="sharded", replica=True),
    # FFN_HC: the attention mix folded, the post-attention RMSNorm fused, written to DRAM for the MoE request.
    "ffn": dict(mix=True, norm="dram", replica=False),
    "plain": dict(mix=False, norm=None, replica=False),
    "replica": dict(mix=True, norm=None, replica=True),
}


@pytest.mark.parametrize(
    "rows, h, hidden, config, sub_format",
    [
        (4, 4, 5120, "attn", "tile"),
        (4, 4, 5120, "ffn", "tile"),
        (4, 4, 5120, "ffn", "row"),
        (4, 4, 5120, "plain", "tile"),
        (4, 4, 4096, "replica", "tile"),
        (5, 8, 5120, "attn", "tile"),
        (8, 8, 5120, "ffn", "row"),
        (2, 2, 5120, "attn", "tile"),
        (2, 2, 4096, "ffn", "tile"),
        (2, 2, 5120, "ffn", "tile"),
        (2, 2, 4096, "attn", "tile"),
        (2, 2, 5120, "plain", "tile"),
    ],
    ids=lambda v: str(v),
)
def test_fused_hyperconnection_rows(device, rows, h, hidden, config, sub_format):
    if device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip("the fused fn dot is Blackhole-only")
    cfg = CONFIGS[config]
    mix, norm, replica = cfg["mix"], cfg["norm"], cfg["replica"]
    failures = []
    for step in range(2):  # the second call runs from the program cache
        case = Case(hidden, rows, seed=step)
        common = _common_kwargs(device, case)
        streams_tt, kwargs = _block_call(device, case, common, h, mix, norm, replica, sub_format)
        outs = ttnn.experimental.deepseek.fused_hyperconnection(streams_tt, **kwargs)
        assert len(outs) == 4 + int(mix) + int(replica), len(outs)
        if mix:
            assert outs[4].memory_config() == streams_tt.memory_config()
        got_rows = _block_outputs(outs, case, h, mix, norm, replica)

        for j in range(rows):
            row_streams, row_kwargs = _row_call(device, case, common, j, mix, norm, replica)
            single = ttnn.experimental.deepseek.fused_hyperconnection(row_streams, **row_kwargs)
            one = [
                ttnn.to_torch(single[0]).float()[0, 0],
                ttnn.to_torch(single[1]).float()[0, 0],
                ttnn.to_torch(single[2]).float().reshape(hidden),
                ttnn.to_torch(single[3]).float()[0, 0],
                ttnn.to_torch(single[4]).float()[0, 0] if mix else None,
                ttnn.to_torch(single[-1]).float()[0, 0] if replica else None,
            ]
            ref = case.reference(j, mix, norm is not None)
            for i, name in enumerate(("post", "comb", "collapsed", "pre", "streams")):
                got = got_rows[j][i]
                if got is None:
                    continue
                # post / comb / pre are 4- or 16-value vectors behind a LoFi fn dot: against torch (float fn) they
                # hold to ~0.99; against the one-row op (the same dot) to PCC.
                torch_pcc = PCC if name in ("collapsed", "streams") else 0.99
                for label, target, need in (("one-row op", one[i], PCC), ("torch", ref[i], torch_pcc)):
                    target = target.reshape(got.shape)
                    passing, pcc = comp_pcc(target, got, need)
                    err = (target - got).abs().max().item()
                    logger.info(f"step {step} row {j} {name} vs {label}: PCC {pcc}, max_abs_err {err:.3e}")
                    if not passing:
                        failures.append(f"step {step} row {j} {name} vs {label}: {pcc}")
            if replica:
                want = got_rows[j][2].reshape(1, hidden).expand(REPLICA_GRID.num_cores(), hidden)
                if not torch.equal(got_rows[j][5], want):
                    failures.append(f"step {step} row {j}: the replica is not collapsed on every core")
            if norm is None:
                # Without the norm, collapsed is exactly the one-row op's (same SFPU math, same order).
                passing, pcc = comp_pcc(one[2], got_rows[j][2], 0.9999)
                if not passing:
                    failures.append(f"step {step} row {j} collapsed vs one-row op (tight): {pcc}")
    assert not failures, "; ".join(failures)


def _prod_call(device, rows, config, common, case):
    cfg = CONFIGS[config]
    if rows == 1:
        streams_tt, kwargs = _row_call(device, case, common, 0, cfg["mix"], cfg["norm"], cfg["replica"])
    else:
        streams_tt, kwargs = _block_call(device, case, common, rows, cfg["mix"], cfg["norm"], cfg["replica"])
    return streams_tt, kwargs


@pytest.mark.parametrize("config", ["attn", "ffn"])
def test_fused_hyperconnection_rows_perf(device, config):
    """Production shapes (D = 5120, hc = 4, fn on 80 DRAM-weight cores below the HC row, fan-in 10, DRAM biases):
    s = 1, 4 and 8 rows, a warmup then a signpost then 20 timed calls each."""
    if device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip("the fused fn dot is Blackhole-only")
    from tracy import signpost

    hidden = 5120
    for rows in (1, 4, 8):
        case = Case(hidden, max(rows, 1))
        common = _common_kwargs(device, case)
        streams_tt, kwargs = _prod_call(device, rows, config, common, case)
        outs = ttnn.experimental.deepseek.fused_hyperconnection(streams_tt, **kwargs)
        ttnn.synchronize_device(device)
        for t in outs:
            ttnn.deallocate(t)
        signpost(f"hc_{config}_s{rows}")
        for _ in range(20):
            outs = ttnn.experimental.deepseek.fused_hyperconnection(streams_tt, **kwargs)
            for t in outs:
                ttnn.deallocate(t)
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
