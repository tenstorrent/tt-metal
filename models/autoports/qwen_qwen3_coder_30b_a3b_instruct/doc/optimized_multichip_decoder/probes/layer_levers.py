# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-04 layer A/B: every candidate measured on the whole warmed traced layer.

Each leg is a warmed traced replay of one multichip decode layer at ctx 128,
median of 100 blocking replays, so the numbers are directly comparable with
``../multichip_decoder/perf_decode.csv`` and with stage 03's
``probes/decode_levers.py``.

The persistent-collective-buffer leg that used to sit here is gone: it is the
shipped default now, so the "04" leg *is* it, and the A/B against the allocating
path lives in ``layer_levers2.py`` and ``layer_levers3.py``.

Leg ``stage 03`` re-runs the *committed stage-03 layer body* verbatim (a copy of
it lives in this file), so the before/after pair is measured in one process, on
one mesh, against one set of weights, minutes apart.

Correctness column: ``max|diff|`` of the layer output against the stage-03 leg,
all four dies, so a leg that is fast and wrong cannot pass unnoticed.

    python layer_levers.py

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


# --- the stage-03 layer, verbatim, as the "before" leg ------------------------
def layer_stage03(x, weights, config, ctx, cos, sin, kv, pos, tok_idx):
    eps = config.global_config.rms_norm_eps
    normed = ttnn.rms_norm(x, weight=weights.input_layernorm, epsilon=eps)
    attn_partial = MC.attention_decode_optimized(
        normed,
        weights.experts,
        config.local_attention,
        cos,
        sin,
        kv,
        pos,
        tok_idx,
        sdpa_program_config=None if kv.is_paged else MC._sdpa_program_config(x.device()),
    )
    ttnn.deallocate(normed)
    attn_out = MC.all_reduce(attn_partial, ctx)
    ttnn.deallocate(attn_partial)
    hidden = ttnn.add(x, attn_out)
    ttnn.deallocate(attn_out)

    normed = ttnn.rms_norm(hidden, weight=weights.post_attention_layernorm, epsilon=eps)
    routing = MC.router_forward_multichip(
        normed, weights.router, weights.expert_window, config.global_config.moe, config.local_moe
    )
    moe_partial = MC.moe_decode_multichip(normed, routing, weights.experts, config.local_moe)
    ttnn.deallocate(normed)
    ttnn.deallocate(routing)
    moe_out = MC.all_reduce(moe_partial, ctx)
    ttnn.deallocate(moe_partial)
    out = ttnn.add(hidden, moe_out)
    ttnn.deallocate(hidden)
    ttnn.deallocate(moe_out)
    return out


# --- collective variants ------------------------------------------------------
def all_reduce_l1(x, ctx):
    """Same RS+AG, but staged in L1 rather than DRAM. Decode's payload is
    128 KB/die, which fits, and the collective is latency-bound there.

    **The archived ``layer_levers.log`` measured this leg confounded**: it read
    ``ctx.num_links`` (2) where the shipped ``all_reduce`` reads ``_links``,
    which is 1 at decode, so the L1 leg carried a second, independent penalty.
    Corrected here to ``MC._links``. The rejection survives the confound --
    ``layer_levers2.py`` and ``layer_levers3.py`` measure the same lever with
    matched link counts and also reject it -- but the 0.4377 in the log is not
    a clean single-variable number and is labelled as such in ``README.md``."""
    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        x,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=MC._links(x, ctx),
        memory_config=ttnn.L1_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=ctx.topology,
    )
    gathered = ttnn.experimental.all_gather_async(
        scattered,
        dim=3,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=MC._links(x, ctx),
        memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=ctx.topology,
    )
    ttnn.deallocate(scattered)
    return gathered


_ONE_LINK = {}


def one_link(x, ctx):
    """One ethernet link instead of two.

    The alternate ``MeshContext`` is **cached**, not rebuilt per call. It owns
    its own persistent collective buffers, and those are allocated on the first
    call at each shape; a fresh context per call means the allocation lands
    inside ``begin_trace_capture``, where ``ttnn.from_torch`` raises "Writes are
    not supported during trace capture" and leaves the trace open -- which is a
    hung mesh. Cached, the eager warm-up call allocates and the traced call
    reuses, which is exactly the discipline the shipped path relies on."""
    c = _ONE_LINK.get(id(ctx))
    if c is None:
        c = _ONE_LINK[id(ctx)] = MC.MeshContext(ctx.mesh, ctx.ccl, ctx.num_devices, 1, ctx.topology, decode_num_links=1)
    return _all_reduce(x, c)


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

    _scatter_router = MC.router_forward_multichip
    _threshold_router = MC.router_forward_threshold

    #  name, layer fn, router, all_reduce
    LEGS = [
        ("stage 03 (before)", layer_stage03, _scatter_router, _all_reduce),
        ("04: sharded norms + router mm", MC.decoder_layer_decode_multichip, _scatter_router, _all_reduce),
        ("04: + threshold router tail", MC.decoder_layer_decode_multichip, _threshold_router, _all_reduce),
        ("04: + CCL in L1", MC.decoder_layer_decode_multichip, _threshold_router, all_reduce_l1),
        ("04: + num_links=1", MC.decoder_layer_decode_multichip, _scatter_router, one_link),
    ]

    reference = None
    for name, layer_fn, router_fn, ar in LEGS:
        MC.router_forward_multichip = router_fn
        MC.all_reduce = MC.all_reduce_decode = ar

        def step():
            return layer_fn(tok, weights, cfg, ctx, cos, sin, kv, pos, CTX - 1)

        try:
            out = step()
            ttnn.synchronize_device(mesh)
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            if reference is None:
                reference = got
                agree = "reference"
            else:
                agree = f"max|diff| {(got - reference).abs().max().item():.3e}"
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                step()
            finally:
                # A leg that raises mid-capture leaves the trace open, and the
                # next begin_trace_capture then dies on
                # ``fd_mesh_command_queue.cpp:760 !trace_id_.has_value()`` --
                # which is a hung mesh and a tt-smi -r, not a failed leg.
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            for _ in range(10):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            samples = []
            for _ in range(100):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                samples.append((time.perf_counter() - t0) * 1e3)
            ttnn.release_trace(mesh, tid)
            print(f"P|{name:34s} {statistics.median(samples):.4f} ms   ({agree})", flush=True)
        except Exception as exc:
            print(f"P|{name:34s} FAILED {str(exc)[:180]}", flush=True)
finally:
    MC.router_forward_multichip = _scatter_router
    MC.all_reduce = MC.all_reduce_decode = _all_reduce
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("P|done")
