# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RING decode MoE (TTMoEDecode = all_to_all_dispatch_metadata -> moe_compute (ring matmul + fused combine) -> deepseek_moe_fast_reduce_nc_fused -> reduce_scatter) at MiMo dims on the
REAL 4x8 mesh with a real layer (MIMO_LAYER=5: real gate/bias/256 experts, dump hidden states), the DSV4 decode MoE block pattern. Needs the patched library and
MOE_COMPUTE_BFP8_WEIGHTS=1 for bfp8 weights (MOE_COMPUTE_FIDELITY=LoFi|HiFi2|HiFi3|HiFi4, MOE_COMPUTE_FP32_ACC=0|1 set the matmul numerics).
Per U in MIMO_UVALS (1,8,32 tokens per mesh row): output vs dump ffn_out (alpha/rel/PCC on live tokens, hidden-sharded output gathered on host), 3-run bit identity, trace == eager,
cumulative stage times (router, dispatch, moe_compute, tilize+fast_reduce, reduce_scatter) in a trace (host-timed, 30 replays)."""
import os
import time

import pytest
import torch

import ttnn
from models.common.modules.moe.tt_moe_decode import TTMoEDecode
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tt.moe_prefill import COLS, HIDDEN, ROWS
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "mimo_v26.yaml")
D1 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
LAYER = int(os.environ.get("MIMO_LAYER", "5"))
UVALS = [int(v) for v in os.environ.get("MIMO_UVALS", "1,8,32").split(",")]
REPLAYS = int(os.environ.get("MIMO_REPLAYS", "30"))


def stats(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float((a @ b) / (b @ b)), float((a - b).norm() / b.norm()), float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def forward_staged(dec, tt_x, tt_scores, tt_indices, upto=99):
    """TTMoEDecode.forward (use_post_combine_tilize or not) cut after a stage: 1 dispatch, 2 moe_compute, 3 tilize+fast_reduce, 99 reduce_scatter (full)."""
    (xi, dx), (ii, di), (si, ds) = dec._format_dispatch_inputs(tt_x, tt_indices, tt_scores)
    sp, oi, osc = ttnn.experimental.all_to_all_dispatch_metadata(
        xi,
        ii,
        si,
        dec.expert_state.tt_expert_mapping,
        **dec.config.dispatch.model_dump(),
        output_tensors=dec.buffers.tt_dispatch_output_tensors,
        cross_device_semaphore=dec.buffers.dispatch_global_semaphore,
    )
    if upto == 1:
        return sp
    _, _, _, l1, _, comb = ttnn.experimental.moe_compute(
        sp,
        oi,
        osc,
        dec.expert_state.tt_expert_mapping,
        dec.expert_state.tt_w0_w1,
        dec.expert_state.tt_w2,
        layer_id=0,
        **dec.config.compute.model_dump(),
        optional_output_tensor=dec.buffers.tt_combine_output,
        optional_cross_device_semaphore=dec.buffers.combine_global_semaphore,
    )
    if upto == 2:
        return comb
    un = dec._pad_for_fast_reduce(ttnn.unsqueeze(comb, dim=1))
    if dec.config.use_post_combine_tilize:
        til = ttnn.experimental.deepseek_moe_post_combine_tilize(un, **dec.config.post_combine_tilize.model_dump())
    else:
        shp = list(un.shape)
        shp[2] = ((shp[2] + 31) // 32) * 32
        til = ttnn.tilize_with_val_padding(
            un, output_tensor_shape=shp, pad_value=0.0, **dec.config.tilize_with_val_padding.model_dump()
        )
    red = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
        til, ii, dec.expert_state.tt_expert_mapping, **dec.config.reduce.model_dump(), scores_tensor=tt_scores
    )
    if upto == 3:
        return red[0]
    out = ttnn.reduce_scatter(red[0], **dec.config.reduce_scatter.model_dump())
    return dec._unpad_after_reduce_scatter(out)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_moe_ring_decode(mesh_device):
    md = mesh_device
    a = torch.load(f"{D1}/layer_{LAYER}.pt", mmap=True)["prefill"]
    sel = torch.arange(0, 512, 16)[:32]
    xs = a["ffn_in"].reshape(ROWS, 512, HIDDEN)[:, sel].clone()
    ref = a["ffn_out"].reshape(ROWS, 512, HIDDEN)[:, sel].float()
    W = load_layer_experts(LAYER)
    w0 = torch.stack([W[e][0] for e in range(256)])[None]
    w1 = torch.stack([W[e][1] for e in range(256)])[None]
    w2 = torch.stack([W[e][2] for e in range(256)])[None]
    del W
    gw, gb = load_gate(LAYER)
    gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
    for n in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
        ttnn.deallocate(getattr(gate, n))
    text = open(CFG).read()
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    for U in UVALS:
        cfg = TTMoEDecodeConfig.from_yaml(
            text.replace("batch_per_device: 4\n", f"batch_per_device: {U}\n", 1), topology=ttnn.Topology.Linear
        )
        if cfg.num_fast_reduce_outputs == 1:
            cfg = cfg.model_copy(
                update={"reduce": cfg.reduce.model_copy(update={"output_memory_config": ttnn.DRAM_MEMORY_CONFIG})}
            )
        t0 = time.time()
        dec = TTMoEDecode(md, cfg, w0, w1, w2)
        print(
            f"RESULT U={U}: TTMoEDecode built in {time.time() - t0:.0f}s (fast_reduce_outputs={cfg.num_fast_reduce_outputs}, post_combine_tilize={cfg.use_post_combine_tilize})",
            flush=True,
        )
        x_tile = ttnn.from_torch(
            xs[:, :U].reshape(ROWS, 1, U, HIDDEN),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m2,
        )
        x_tok = ttnn.from_torch(
            xs[:, :U].reshape(ROWS * U, 1, 1, HIDDEN),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS)),
        )

        def route():
            s, i = gate.route_v(x_tile, "sel_mm1x8")
            return s, i

        def run(upto=99):
            s, i = route()
            if upto == 0:
                return s
            return forward_staged(dec, x_tok, s, i, upto)

        o = run()
        ttnn.synchronize_device(md)
        devs = ttnn.get_device_tensors(o)
        outs = [
            torch.cat(
                [ttnn.to_torch(devs[r * COLS + c]).reshape(-1, HIDDEN // COLS)[:U].float() for c in range(COLS)], dim=-1
            )
            for r in range(ROWS)
        ]
        out = torch.stack(outs)
        al, rel, pc = stats(out, ref[:, :U])
        ident = True
        for _ in range(2):
            o2 = run()
            ttnn.synchronize_device(md)
            d2 = ttnn.get_device_tensors(o2)
            ident &= all(
                bool(
                    torch.equal(
                        torch.cat(
                            [
                                ttnn.to_torch(d2[r * COLS + c]).reshape(-1, HIDDEN // COLS)[:U].float()
                                for c in range(COLS)
                            ],
                            dim=-1,
                        ),
                        outs[r],
                    )
                )
                for r in range(ROWS)
            )
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        ot = run()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        dt_ = ttnn.get_device_tensors(ot)
        same = all(
            bool(
                torch.equal(
                    torch.cat(
                        [ttnn.to_torch(dt_[r * COLS + c]).reshape(-1, HIDDEN // COLS)[:U].float() for c in range(COLS)],
                        dim=-1,
                    ),
                    outs[r],
                )
            )
            for r in range(ROWS)
        )
        ttnn.release_trace(md, tid)
        print(
            f"RESULT U={U} ring MoE vs dump ffn_out (live tokens, L{LAYER}): alpha {al:.4f} rel {rel:.4f} PCC {pc:.5f}; 3 runs identical {ident}; trace==eager {same}; env bfp8={os.environ.get('MOE_COMPUTE_BFP8_WEIGHTS', '0')} fid={os.environ.get('MOE_COMPUTE_FIDELITY', 'LoFi')} fp32acc={os.environ.get('MOE_COMPUTE_FP32_ACC', '0')}",
            flush=True,
        )
        prev = 0.0
        names = {
            0: "router(route_v)",
            1: "+dispatch",
            2: "+moe_compute(+combine)",
            3: "+tilize+fast_reduce",
            99: "+reduce_scatter (full)",
        }
        row = []
        for st in (0, 1, 2, 3, 99):
            run(st)
            ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            run(st)
            ttnn.end_trace_capture(md, tid, cq_id=0)
            for _ in range(5):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            t0 = time.time()
            for _ in range(REPLAYS):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            v = (time.time() - t0) / REPLAYS * 1e3
            ttnn.release_trace(md, tid)
            row.append(f"{names[st]} {v - prev:+.3f}")
            prev = v
        print(f"RESULT STAGES U={U} ring path (ms per layer, deltas; total {prev:.3f}): " + "; ".join(row), flush=True)
        del dec
