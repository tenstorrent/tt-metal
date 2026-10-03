# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where the MoE core time goes: prefix timings of TTMoEDecode.forward (inside a long trace, (t(3)-t(1))/2).
Stages: format inputs+dispatch | + moe_compute | + unsqueeze/pad/tilize | + fast_reduce | + reduce_scatter (= full)."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer

T = 4


def chain_ms(md, fn):
    def run(k):
        def f():
            for _ in range(k):
                fn()

        fn()
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        f()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(20):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        r = (time.perf_counter() - t) / 20 * 1e3
        ttnn.release_trace(md, tid)
        return r

    return (run(3) - run(1)) / 2


def stages(dec, tt_x, tt_scores, tt_indices, upto):
    (in_t, d_in), (idx_t, d_idx), (sc_t, d_sc) = dec._format_dispatch_inputs(tt_x, tt_indices, tt_scores)
    sparse, o_idx, o_sc = ttnn.experimental.all_to_all_dispatch_metadata(
        in_t,
        idx_t,
        sc_t,
        dec.expert_state.tt_expert_mapping,
        **dec.config.dispatch.model_dump(),
        output_tensors=dec.buffers.tt_dispatch_output_tensors,
        cross_device_semaphore=dec.buffers.dispatch_global_semaphore,
    )
    if d_in:
        ttnn.deallocate(in_t)
    if d_sc:
        ttnn.deallocate(sc_t)
    if upto == 0:
        return sparse
    _, _, _, l1_out, _, combine = ttnn.experimental.moe_compute(
        sparse,
        o_idx,
        o_sc,
        dec.expert_state.tt_expert_mapping,
        dec.expert_state.tt_w0_w1,
        dec.expert_state.tt_w2,
        layer_id=0,
        **dec.config.compute.model_dump(),
        optional_output_tensor=dec.buffers.tt_combine_output,
        optional_cross_device_semaphore=dec.buffers.combine_global_semaphore,
    )
    ttnn.deallocate(l1_out)
    if upto == 1:
        return combine
    un = dec._pad_for_fast_reduce(ttnn.unsqueeze(combine, dim=1))
    if dec.config.use_post_combine_tilize:
        til = ttnn.experimental.deepseek_moe_post_combine_tilize(un, **dec.config.post_combine_tilize.model_dump())
    else:
        shp = list(un.shape)
        shp[2] = ((shp[2] + 31) // 32) * 32
        til = ttnn.tilize_with_val_padding(
            un, output_tensor_shape=shp, pad_value=0.0, **dec.config.tilize_with_val_padding.model_dump()
        )
    if upto == 2:
        return til
    fr = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
        til, idx_t, dec.expert_state.tt_expert_mapping, **dec.config.reduce.model_dump(), scores_tensor=tt_scores
    )
    ttnn.deallocate(til)
    if d_idx:
        ttnn.deallocate(idx_t)
    if upto == 3:
        return fr[0]
    out = ttnn.reduce_scatter(fr[0], **dec.config.reduce_scatter.model_dump())
    for t in fr:
        ttnn.deallocate(t)
    return out


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 300_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_moe_stages(mesh_device):
    md = mesh_device
    w = load_moe_layer(2)
    blk = DSV41MoEBlock(md, w, batch_per_device=T, gate_bias_shift=0.0)
    blk.warmup()
    h = ttnn.from_torch(
        torch.randn(1, 1, T, 5120).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    h_tok = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, 5120])
    sc, idx = blk.gate.forward(h)
    prev, names = 0.0, ["format+dispatch", "moe_compute", "unsqueeze/pad/tilize", "fast_reduce", "reduce_scatter"]
    for i, n in enumerate(names):
        ms = chain_ms(md, lambda: stages(blk.decode, h_tok, sc, idx, i))
        print(f"MST {n:24s} cumulative {ms * 1e3:7.1f} us   stage {(ms - prev) * 1e3:7.1f} us", flush=True)
        prev = ms
    print(
        f"MST full blk.forward (forced routing) {chain_ms(md, lambda: blk.forward(h, h_tok, (sc, idx))) * 1e3:.1f} us",
        flush=True,
    )

    # ---- how does moe_compute scale with the busiest device's expert count? (all 16 users pick the same 6 experts)
    mapping = blk.decode.expert_state  # noqa: F841
    rep_rows = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
    mk = lambda ids: ttnn.from_torch(
        torch.tensor(ids, dtype=torch.int32).repeat(T * 4, 1).reshape(T * 4, 1, 1, 6),
        device=md,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep_rows,
    )
    wts = ttnn.from_torch(
        torch.full((T * 4, 1, 1, 6), 0.25).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep_rows,
    )
    for label, ids in (
        ("6 experts on 6 different devices ", [0, 12, 24, 36, 48, 60]),
        ("6 experts on 1 device (ids 0-5)   ", [0, 1, 2, 3, 4, 5]),
        ("3 experts on 1 device, 3 elsewhere", [0, 1, 2, 12, 24, 36]),
        ("1 expert only (x6 same id)         ", [7, 7, 7, 7, 7, 7]),
    ):
        try:
            i_t = mk(ids)
            ms = chain_ms(md, lambda: stages(blk.decode, h_tok, wts, i_t, 1))
            print(f"MSTX {label} moe_compute stage total {ms * 1e3:7.1f} us", flush=True)
        except Exception as e:
            print(f"MSTX {label} FAILED {str(e)[:120]}", flush=True)
