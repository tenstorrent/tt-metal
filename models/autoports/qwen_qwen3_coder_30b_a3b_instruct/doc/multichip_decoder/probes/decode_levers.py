# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Post-profile sweep: the levers the multichip decode profile actually suggests.

Re-audited after the collective change (`allreduce_ab.py`), because each round
here has promoted a new top op. The decode profile now reads, over the 64-op
decode layer (`ops_perf_multichip_decode.csv` sorted by HOST START TS, device 0,
rows 134-197 -- the second of the two decode iterations `profile_layer.py`
runs), 414.66 us of device time:

    SparseMatmul   82.65 us  19.9%
    LayerNorm      47.92     11.6    (replicated, and 40.21 of it is the two
    Matmul         46.04     11.1     residual-stream RMSNorms)
    ReshapeView    44.80     10.8
    ReduceScatter  38.36      9.3
    AllGather      28.40      6.8
    TopK           26.32      6.3    (one core, a single 128-wide row)

Collectives are 66.76 us, 16.1% -- the third-largest cost after the expert
matmuls and the replicated norms, and 1.67x the 39.92 us the plan budgeted. So
the legs below are about the collective, plus the one non-collective question
the L1 budget sweep left open.

Each leg is a warmed traced replay of the whole layer at ctx 128, median of 100,
so the number is comparable to `perf_decode.csv`.

    python decode_levers.py

Prints ``P|`` lines only.
"""
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tests.reference import layer_state_dict, load_config
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import functional_decoder as F
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.weight_mapping import convert_layer_weights

CTX = 128
hf = load_config()
tw = convert_layer_weights(layer_state_dict(0), hf)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=90_000_000, l1_small_size=32768)
_all_reduce = MC.all_reduce
_budget = MC._DECODE_EXPERT_L1_BUDGET_BYTES


def one_link(x, ctx):
    return _all_reduce(x, MC.MeshContext(ctx.mesh, ctx.ccl, ctx.num_devices, 1, ctx.topology))


def bf8_payload(x, ctx):
    """Halve the bytes on the wire. Decode is latency-bound, so this should not
    help -- and it costs precision, so it has to be clearly better to be worth
    considering at all."""
    small = ttnn.typecast(x, ttnn.bfloat8_b)
    out = _all_reduce(small, ctx)
    ttnn.deallocate(small)
    return ttnn.typecast(out, ttnn.bfloat16)


def linear_topology(x, ctx):
    """The repo default for a 4-device mesh, for the record."""
    return _all_reduce(x, MC.MeshContext(ctx.mesh, ctx.ccl, ctx.num_devices, ctx.num_links, ttnn.Topology.Linear))


try:
    cfg = MC.MeshDecoderConfig.from_hf(hf)
    ctx = MC.mesh_context(mesh)
    weights = MC.upload_multichip_weights(tw, mesh, cfg)
    cos, sin = F.build_rope_cache(hf, 1024, mesh)
    kv = MC.create_mesh_kv_cache(mesh, cfg, 1, 1024, block_size=32)

    torch.manual_seed(0)
    tok = ttnn.from_torch(
        torch.randn(1, 1, 1, hf.hidden_size) * 0.02,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    pos = ttnn.from_torch(
        torch.tensor([CTX - 1], dtype=torch.int32),
        dtype=ttnn.int32,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    LEGS = [
        ("shipped", _all_reduce, _budget),
        ("expert intermediates in DRAM", _all_reduce, 0),
        ("num_links=1", one_link, _budget),
        ("bfloat8_b collective payload", bf8_payload, _budget),
        ("Topology.Linear (repo default)", linear_topology, _budget),
    ]

    reference = None
    for name, fn, budget in LEGS:
        MC.all_reduce = MC.all_reduce_decode = fn
        MC._DECODE_EXPERT_L1_BUDGET_BYTES = budget

        def step():
            return MC.decoder_layer_decode_multichip(tok, weights, cfg, ctx, cos, sin, kv, pos, CTX - 1)

        try:
            out = step()
            ttnn.synchronize_device(mesh)
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1].float()
            if reference is None:
                reference = got
                agree = "reference"
            else:
                agree = f"max|diff| {(got - reference).abs().max().item():.3e}"
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            step()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            for _ in range(10):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            samples = []
            for _ in range(100):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                samples.append((time.perf_counter() - t0) * 1e3)
            ttnn.release_trace(mesh, tid)
            print(f"P|{name:32s} {statistics.median(samples):.4f} ms   ({agree})", flush=True)
        except Exception as exc:
            print(f"P|{name:32s} FAILED {str(exc)[:130]}", flush=True)
finally:
    MC.all_reduce = MC.all_reduce_decode = _all_reduce
    MC._DECODE_EXPERT_L1_BUDGET_BYTES = _budget
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("P|done")
