# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""custom_ops/moe_dispatch_prefill (the model-local DeepSeek prefill dispatch / combine) on a (2,2) mesh over a
FABRIC_1D line: the mesh rows of each column are a dispatch group. Against a torch model of both programs built from
host-computed routing-setup tensors: the dispatch buffer rows and metadata written for every routed token (exact),
the combine output at every routed ``[token, slot]`` (exact; bfp8 expert outputs decoded as the packer does) and
the round trip ``combine(dispatch(x))[t, k] == x[t]`` on the slots whose expert is in the group; a second call on
fresh buffers (the program-cache hit), the padding config, a TILE dispatch input and the zeroed combine output.
When the shared ``ttnn.experimental.deepseek_prefill`` ops still build on the tree, the same inputs run through
them too: the written rows / slots must be bit-identical and the wall times are logged side by side.

    GEMMA4_TEST_MESH_SHAPES is not read: the (2,2) block is the case (a QB2-class 2x2 torus block on the Galaxy,
    TT_VISIBLE_DEVICES=8,9,12,13 MESH_DEVICE=P150x4 with the p150_x4 2-channel descriptor).
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants
from models.demos.gemma4.custom_ops.moe_dispatch_prefill import MoeCombinePrefill, MoeDispatchPrefill

MESH_2x2 = [pytest.param((2, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}, id="2x2")]
TILE = 32


def _round_up(v, a):
    return -(-v // a) * a


class Case:
    """Per-chip host tensors of one dispatch over a ``rows x cols`` mesh (axis 0 = the dispatch group)."""

    def __init__(self, rows, cols, S, E, K, H, real_tokens=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.rows, self.cols, self.S, self.E, self.K, self.H = rows, cols, S, E, K, H
        self.dgs, self.ndg = rows, cols
        self.epc = E // (rows * cols)
        _, self.metadata_len, self.max_buf, _ = compute_constants(S, E, K, rows * cols, rows, K)
        self.table = ExpertMapping.create_dispatch_table(E, rows, cols)  # [ndg, E + 1]
        self.real = real_tokens  # per row: the unpadded rows (None = all real)
        self.x = {}
        self.idx = {}
        for r in range(rows):
            for c in range(cols):
                self.x[(r, c)] = ((torch.rand(S, H, generator=g) - 0.5) * 4).bfloat16()
                ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(S)]).to(torch.int64)
                if real_tokens is not None:
                    ids[real_tokens[r] :] = E  # the padding sentinel (table column E is -1)
                self.idx[(r, c)] = ids
        # host routing setup: counts per expert over the group, tile-aligned regions in local-expert order, each
        # source chip's run start per expert (what masked_bincount + offset_cumsum produce on device)
        self.counts, self.region, self.offsets, self.local = {}, {}, {}, {}
        for c in range(cols):
            per_row = [torch.bincount(self.idx[(r, c)].reshape(-1), minlength=E + 1)[:E] for r in range(rows)]
            total = sum(per_row)
            for r in range(rows):
                local = [e for e in range(E) if self.table[c, e] == r]
                region = torch.zeros(E, dtype=torch.int64)
                run = 0
                for e in local:
                    region[e] = run
                    run += _round_up(int(total[e]), TILE)
                self.local[(r, c)] = local
                self.counts[(r, c)] = total.clone()
                self.region[(r, c)] = region
            for r in range(rows):
                off = torch.zeros(E, dtype=torch.int64)
                for e in range(E):
                    d = int(self.table[c, e])
                    if d < 0:
                        continue
                    off[e] = self.region[(d, c)][e] + sum(int(per_row[rr][e]) for rr in range(r))
                self.offsets[(r, c)] = off

    def lin(self, r, c):
        return r * self.cols + c

    def golden_dispatch(self, c):
        """Per group position: ``(buffer, metadata, written)`` for column ``c``."""
        rows = range(self.rows)
        return MoeDispatchPrefill.golden(
            [self.x[(r, c)] for r in rows],
            [self.idx[(r, c)] for r in rows],
            [self.offsets[(r, c)] for r in rows],
            self.table[c],
            [self.lin(r, c) for r in rows],
            self.cols,
            self.max_buf,
            None if self.real is None else [self.real[r] for r in rows],
        )

    def stacked(self, per_chip, shape):
        """``[rows, cols, *shape]`` torch tensor from per-chip tensors (the 2D mesh sharding of dims (0, 1))."""
        out = torch.zeros(self.rows, self.cols, *shape, dtype=next(iter(per_chip.values())).dtype)
        for (r, c), t in per_chip.items():
            out[r, c] = t.reshape(shape)
        return out


def _shard(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    """``[rows, cols, ...]`` -> one ``[1, 1, ...]`` shard per chip."""
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(0, 1)),
    )


def _per_chip(t, rows, cols):
    devs = ttnn.get_device_tensors(t)
    return {(r, c): ttnn.to_torch(devs[r * cols + c]) for r in range(rows) for c in range(cols)}


def _device_inputs(mesh, case, x_layout):
    rows, cols = case.rows, case.cols
    x = _shard(
        mesh,
        case.stacked(case.x, (1, 1, case.S, case.H)).reshape(rows, cols, 1, 1, case.S, case.H),
        ttnn.bfloat16,
        x_layout,
    )
    x = ttnn.reshape(x, (1, 1, case.S, case.H))
    idx = _shard(mesh, case.stacked(case.idx, (1, 1, case.S, case.K)).to(torch.int16), ttnn.uint16)
    idx = ttnn.reshape(idx, (1, 1, case.S, case.K))
    off = _shard(mesh, case.stacked(case.offsets, (1, 1, case.E)).to(torch.int32), ttnn.int32)
    off = ttnn.reshape(off, (1, case.E))
    table = MoeDispatchPrefill.shard_expert_dispatch_table(mesh, case.table, dispatch_axis=0)
    counts = ttnn.reshape(
        _shard(mesh, case.stacked(case.counts, (1, 1, case.E)).to(torch.int32), ttnn.int32), (1, case.E)
    )
    region = ttnn.reshape(
        _shard(mesh, case.stacked(case.region, (1, 1, case.E)).to(torch.int32), ttnn.int32), (1, case.E)
    )
    padding = None
    if case.real is not None:
        pad = torch.tensor([[[case.real[r], 0] for c in range(cols)] for r in range(rows)], dtype=torch.int32)
        padding = ttnn.reshape(_shard(mesh, pad.reshape(rows, cols, 1, 1, 2), ttnn.uint32), (1, 2))
    return x, idx, off, table, counts, region, padding


def _check_dispatch(case, buffer, metadata, label):
    buf = _per_chip(buffer, case.rows, case.cols)
    meta = _per_chip(metadata, case.rows, case.cols)
    for c in range(case.cols):
        g_buf, g_meta, written = case.golden_dispatch(c)
        for r in range(case.rows):
            w = written[r]
            assert int(w.sum()) > 0
            got = buf[(r, c)].reshape(case.max_buf, case.H)[w]
            assert torch.equal(got.bfloat16(), g_buf[r][w]), f"{label}: dispatch buffer rows differ on chip {(r, c)}"
            got_meta = meta[(r, c)].reshape(case.max_buf, case.metadata_len)[w].long()
            assert torch.equal(got_meta, g_meta[r][w]), f"{label}: metadata differs on chip {(r, c)}"
    return buf, meta


def _combine_input(buffer, form):
    if form == "bf16_rm":
        return buffer
    tiled = ttnn.to_layout(buffer, ttnn.TILE_LAYOUT)
    if form == "bf16_tile":
        return tiled
    assert form == "bfp8_tile"
    return ttnn.typecast(tiled, ttnn.bfloat8_b)


def _check_combine(case, combine_in, meta, out, init_zeros, label):
    """The combine output against the torch model fed the host copies of the combine's own inputs."""
    cin = _per_chip(combine_in, case.rows, case.cols)  # bfp8 decoded exactly, as the packer does
    got = _per_chip(out, case.rows, case.cols)
    for c in range(case.cols):
        rows = range(case.rows)
        golden, written = MoeCombinePrefill.golden(
            [cin[(r, c)].reshape(case.max_buf, case.H) for r in rows],
            [meta[(r, c)].reshape(case.max_buf, case.metadata_len).long() for r in rows],
            [case.counts[(r, c)] for r in rows],
            [case.region[(r, c)] for r in rows],
            [case.local[(r, c)] for r in rows],
            case.S,
            case.K,
        )
        for r in rows:
            lin = case.lin(r, c)
            o = got[(r, c)].reshape(case.S, case.K, case.H)
            w = written[lin]
            assert int(w.sum()) > 0
            assert torch.equal(o[w].float(), golden[lin][w].float()), f"{label}: combine slots differ on chip {(r, c)}"
            if init_zeros:
                assert torch.count_nonzero(o[~w]) == 0, f"{label}: unrouted slots not zero on chip {(r, c)}"
            # round trip: a routed slot of this group carries the token's own row (bf16 paths exact)
            if combine_in.dtype == ttnn.bfloat16:
                x = case.x[(r, c)]
                ids = case.idx[(r, c)]
                for t in range(case.S if case.real is None else case.real[r]):
                    for k in range(case.K):
                        if case.table[c, int(ids[t, k])] >= 0:
                            assert bool(w[t, k]), f"{label}: slot [{t}, {k}] of chip {(r, c)} not written"
                            assert torch.equal(o[t, k].bfloat16(), x[t]), f"{label}: round trip differs at [{t}, {k}]"


def _shared_ops_available():
    try:
        return hasattr(ttnn.experimental.deepseek_prefill, "dispatch") and hasattr(
            ttnn.experimental.deepseek_prefill, "combine"
        )
    except AttributeError:
        return False


def _timed(fn, mesh, iters=5):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / iters * 1e3


@pytest.mark.parametrize(
    "S, E, K, H, x_layout, form, padded, init_zeros",
    [
        (64, 64, 2, 256, "rm", "bf16_tile", False, False),  # the smallest cross-chip case
        (64, 64, 2, 256, "rm", "bf16_rm", False, True),  # row-major combine input, zeroed output
        (64, 64, 2, 256, "tile", "bf16_tile", False, False),  # TILE dispatch input (the untilize compute kernel)
        (64, 128, 8, 2816, "rm", "bfp8_tile", False, False),  # the 26B shape: bfp8 expert outputs
        (256, 128, 8, 2816, "rm", "bfp8_tile", True, False),  # the 26B shape with a padding config (200 / 100 rows)
    ],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("mesh_device, device_params", MESH_2x2, indirect=True)
def test_moe_dispatch_prefill(mesh_device, device_params, S, E, K, H, x_layout, form, padded, init_zeros):
    rows, cols = (int(d) for d in mesh_device.shape)
    assert (rows, cols) == (2, 2)
    real = [200, 100] if padded else None
    case = Case(rows, cols, S, E, K, H, real_tokens=real)
    layout = ttnn.TILE_LAYOUT if x_layout == "tile" else ttnn.ROW_MAJOR_LAYOUT
    dispatch = MoeDispatchPrefill(
        mesh_device,
        dispatch_group_size=rows,
        experts_per_chip=case.epc,
        num_routed_experts=E,
        num_experts_per_tok=K,
        metadata_len=case.metadata_len,
        max_dispatch_buffer_token_size=case.max_buf,
        seq_len_per_chip=S,
        emb_dim=H,
        cluster_axis=0,
        num_links=1,
    )
    combine = MoeCombinePrefill(
        mesh_device,
        dispatch_group_size=rows,
        experts_per_chip=case.epc,
        num_experts_per_tok=K,
        seq_len_per_chip=S,
        emb_dim=H,
        cluster_axis=0,
        num_links=1,
        init_zeros=init_zeros,
    )
    for call in ("first", "cache-hit"):  # the second call on fresh buffers: the program-cache hit
        x, idx, off, table, counts, region, padding = _device_inputs(mesh_device, case, layout)
        buffer, metadata = dispatch.op(x, idx, off, table, padding_config=padding)
        _, meta = _check_dispatch(case, buffer, metadata, call)
        combine_in = _combine_input(buffer, form)
        out = combine.op(combine_in, metadata, counts, region)
        _check_combine(case, combine_in, meta, out, init_zeros, call)

    if not _shared_ops_available():
        logger.info("shared deepseek_prefill dispatch / combine not on this tree: no cross-check")
        return
    # the shared ops on the same inputs: bit-identical written rows / slots, wall time side by side
    x, idx, off, table, counts, region, padding = _device_inputs(mesh_device, case, layout)
    x3 = ttnn.reshape(x, (1, S, H))
    idx3 = ttnn.reshape(idx, (1, S, K))
    try:
        s_buffer, s_meta = ttnn.experimental.deepseek_prefill.dispatch(
            input_tensor=x3,
            indices_tensor=idx3,
            expert_offsets_tensor=off,
            expert_dispatch_table_tensor=table,
            padding_config=padding,
            dispatch_group_size=rows,
            experts_per_chip=case.epc,
            num_routed_experts=E,
            num_experts_per_tok=K,
            metadata_len=case.metadata_len,
            max_dispatch_buffer_token_size=case.max_buf,
            cluster_axis=0,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
    except Exception as e:  # noqa: BLE001 - the shared op no longer builds on this tree: that is what this op is for
        logger.warning(f"shared dispatch op failed on this tree ({type(e).__name__}: {str(e)[:200]}): no cross-check")
        return
    buffer, metadata = dispatch.op(x, idx, off, table, padding_config=padding)
    buf, meta = _check_dispatch(case, buffer, metadata, "cross-check")
    s_buf, s_meta_t = _check_dispatch(case, s_buffer, s_meta, "shared")
    for key in buf:
        g_buf, g_meta, written = case.golden_dispatch(key[1])
        w = written[key[0]]
        assert torch.equal(buf[key].reshape(case.max_buf, H)[w], s_buf[key].reshape(case.max_buf, H)[w])
        assert torch.equal(meta[key].reshape(case.max_buf, -1)[w], s_meta_t[key].reshape(case.max_buf, -1)[w])
    combine_in = _combine_input(buffer, form)
    out = combine.op(combine_in, metadata, counts, region)
    s_out = ttnn.experimental.deepseek_prefill.combine(
        combine_in,
        metadata,
        counts,
        region,
        dispatch_group_size=rows,
        experts_per_chip=case.epc,
        num_experts_per_tok=K,
        seq_len_per_chip=S,
        cluster_axis=0,
        num_links=1,
        topology=ttnn.Topology.Linear,
        init_zeros=init_zeros,
    )
    got, s_got = _per_chip(out, rows, cols), _per_chip(s_out, rows, cols)
    cin = _per_chip(combine_in, rows, cols)
    for c in range(cols):
        _, written = MoeCombinePrefill.golden(
            [cin[(r, c)].reshape(case.max_buf, H) for r in range(rows)],
            [meta[(r, c)].reshape(case.max_buf, -1).long() for r in range(rows)],
            [case.counts[(r, c)] for r in range(rows)],
            [case.region[(r, c)] for r in range(rows)],
            [case.local[(r, c)] for r in range(rows)],
            S,
            K,
        )
        for r in range(rows):
            w = written[case.lin(r, c)]
            a = got[(r, c)].reshape(S, K, H)[w]
            b = s_got[(r, c)].reshape(S, K, H)[w]
            assert torch.equal(a, b), f"combine differs from the shared op on chip {(r, c)}"
    t_local = _timed(lambda: dispatch.op(x, idx, off, table, padding_config=padding), mesh_device)
    t_shared = _timed(
        lambda: ttnn.experimental.deepseek_prefill.dispatch(
            input_tensor=x3,
            indices_tensor=idx3,
            expert_offsets_tensor=off,
            expert_dispatch_table_tensor=table,
            padding_config=padding,
            dispatch_group_size=rows,
            experts_per_chip=case.epc,
            num_routed_experts=E,
            num_experts_per_tok=K,
            metadata_len=case.metadata_len,
            max_dispatch_buffer_token_size=case.max_buf,
            cluster_axis=0,
            num_links=1,
            topology=ttnn.Topology.Linear,
        ),
        mesh_device,
    )
    c_local = _timed(lambda: combine.op(combine_in, metadata, counts, region), mesh_device)
    c_shared = _timed(
        lambda: ttnn.experimental.deepseek_prefill.combine(
            combine_in,
            metadata,
            counts,
            region,
            dispatch_group_size=rows,
            experts_per_chip=case.epc,
            num_experts_per_tok=K,
            seq_len_per_chip=S,
            cluster_axis=0,
            num_links=1,
            topology=ttnn.Topology.Linear,
            init_zeros=init_zeros,
        ),
        mesh_device,
    )
    logger.info(
        f"[moe_dispatch_prefill] S {S} E {E} K {K} H {H} {form}: dispatch local {t_local:.2f} ms vs shared "
        f"{t_shared:.2f} ms; combine local {c_local:.2f} ms vs shared {c_shared:.2f} ms (wall, 5 calls each)"
    )
