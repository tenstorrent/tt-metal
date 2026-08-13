# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A/B the decode all-reduce spelling on the real layer, not on a bare tensor.

The design phase picked AG-of-partials for decode from a standalone sweep
(`ar_probe.py`): 19.96 us against RS+AG's 23.69 at `[1,1,32,2048]`. The decode
profile says that sweep measured the wrong shape. The shipped decode tensor is
`[1,1,1,2048]` -- logical *one* row, padded to a tile -- and `ttnn.sum` over a
tensor whose last two dims are not tile-aligned drags a `FillPad` behind it,
which is the exact hazard stage 02 removed from the router. Measured in the
layer:

    AllGatherAsync 24.16 us + FillPad 5.78 + FastReduceNC 2.48 + Slice 1.35
        = 33.8 us for the attention all-reduce
    AllGatherAsync 19.11 + FillPad 5.70 + FastReduceNC 2.41 + Slice 1.36
        = 28.6 us for the expert all-reduce

against the 19.96 the probe promised. `ar_probe.py` used 32 logical rows, where
no FillPad appears.

Three legs, each a warmed traced replay of the whole decode layer at ctx 128, so
the number is the one the perf test reports:

  agsum  AG(dim 0) then ttnn.sum      -- what the plan chose
  rsag   reduce-scatter then all-gather -- what prefill uses
  agpad  AG(dim 0), reshape the logical shape up to the padded 32 rows, sum,
         reshape back -- keeps one collective but takes the FillPad away

Run: python allreduce_ab.py
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
_original = MC.all_reduce_decode


def ar_rsag(x, ctx):
    return MC.all_reduce_prefill(x, ctx)


def ar_agpad(x, ctx):
    rows, width = int(x.shape[-2]), int(x.shape[-1])
    padded = max(32, rows)
    gathered = ttnn.experimental.all_gather_async(
        x,
        dim=0,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=ctx.num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ctx.topology,
    )
    # Declare the tile row-padding as logical, so ttnn.sum's last two dims are
    # both tile-aligned and fill_pad.cpp:17-24 does not fire.
    aligned = ttnn.reshape(gathered, (ctx.num_devices, 1, padded, width))
    reduced = ttnn.sum(aligned, dim=0)
    ttnn.deallocate(gathered)
    return ttnn.reshape(ttnn.unsqueeze_to_4D(reduced), (1, 1, rows, width), (1, 1, padded, width))


LEGS = {"agsum": _original, "rsag": ar_rsag, "agpad": ar_agpad}

try:
    cfg = MC.MeshDecoderConfig.from_hf(hf)
    ctx = MC.mesh_context(mesh)
    weights = MC.upload_multichip_weights(tw, mesh, cfg)
    cos, sin = F.build_rope_cache(hf, 1024, mesh)
    sparsity = MC.build_local_sparsity(mesh, cfg.local_moe)
    kv = MC.create_mesh_kv_cache(mesh, cfg, 1, 1024, block_size=32)

    def rep(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    torch.manual_seed(0)
    tok = rep(torch.randn(1, 1, 1, hf.hidden_size) * 0.02)
    pos = ttnn.from_torch(
        torch.tensor([CTX - 1], dtype=torch.int32),
        dtype=ttnn.int32,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    reference = None
    for name, fn in LEGS.items():
        MC.all_reduce_decode = fn

        def step():
            return MC.decoder_layer_decode_multichip(tok, weights, cfg, ctx, cos, sin, kv, pos, CTX - 1)

        out = step()
        ttnn.synchronize_device(mesh)
        got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1].float()
        if reference is None:
            reference = got
            agree = "reference"
        else:
            agree = f"max|diff| vs agsum = {(got - reference).abs().max().item():.3e}"
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
        print(f"P|{name:6s} traced decode ctx{CTX} = {statistics.median(samples):.4f} ms   ({agree})", flush=True)
finally:
    MC.all_reduce_decode = _original
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("P|done")
