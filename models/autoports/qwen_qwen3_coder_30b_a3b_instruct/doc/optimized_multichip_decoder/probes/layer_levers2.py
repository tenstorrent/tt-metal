# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-04 layer A/B, round 2: the collective family, and a repeatability pass.

Round 1 (``layer_levers.py``) put the stage-04 default at 0.4346 ms against
stage 03's 0.4764, and left three things unsettled. **Both of those are an
early run of that probe and neither is a published figure**: ``layer_levers.log``
as it stands reads 0.4700 -> 0.4282, and stage 03's own frozen CSV reads 0.4767,
not 0.4764. They are kept because the three questions below are what they
motivated:

* the threshold routing tail measured 0.4378, i.e. *slower* by 0.7%, against a
  predicted 15 us saving -- close enough to the run-to-run spread that it needed
  a repeat before being believed either way;
* persistent CCL buffers raised, because this probe passed
  ``[intermediate, penult, output]`` where the op wants
  ``[intermediate, output, penult]`` (``test_minimal_reduce_scatter_async.py``
  :200). A first API error is not a rejection;
* ``matmul_reduce_scatter_async`` on the ``wo`` -> reduce-scatter edge, named as
  untried and bounded at 5.3% by stage 03, had not been built at all.

Every leg is run **twice**, in two interleaved passes over the same list, so the
spread of a leg against itself is on the page next to the differences between
legs.

    python layer_levers2.py

Prints ``P|`` lines only.
"""
import math
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tests.reference import layer_state_dict, load_config
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import functional_decoder as F
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import optimized_decoder as O
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.weight_mapping import convert_layer_weights

CTX = 128
hf = load_config()
tw = convert_layer_weights(layer_state_dict(0), hf)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=90_000_000, l1_small_size=32768)

_all_reduce = MC.all_reduce
_scatter_router = MC.router_forward_multichip
_threshold_router = MC.router_forward_threshold

_PERSIST = {}


def all_reduce_persistent(x, ctx):
    """RS+AG with caller-owned persistent buffers, so no collective allocates
    inside the trace. Buffer order is [intermediate, output, penult]."""
    key = (tuple(x.shape), str(x.dtype))
    if key not in _PERSIST:
        interm, penult = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
            x, dim=3, topology=ctx.topology, cluster_axis=None
        )
        shape = list(x.shape)
        shape[3] //= ctx.num_devices
        mk = lambda s: ttnn.from_torch(
            torch.zeros(s),
            device=ctx.mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=x.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(ctx.mesh),
        )
        _PERSIST[key] = (interm, penult, mk(shape), mk(list(x.shape)))
    interm, penult, rs_out, ag_out = _PERSIST[key]
    bufs = [interm, rs_out] + ([penult] if penult is not None else [])
    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        x,
        persistent_output_buffers=bufs,
        dim=3,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=MC._links(x, ctx),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ctx.topology,
    )
    gathered = ttnn.experimental.all_gather_async(
        scattered,
        persistent_output_buffer=ag_out,
        dim=3,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=MC._links(x, ctx),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ctx.topology,
    )
    # The persistent buffers must survive the caller's deallocate.
    return ttnn.clone(gathered, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# --- fused matmul + reduce-scatter on the wo edge ------------------------------
_MMRS = {}


def _attention_to_concat_heads(x, weights, config, cos, sin, kv, pos, tok_idx):
    """``attention_decode_optimized`` stopped one op early: everything up to and
    including ``nlp_concat_heads_decode``, returning the ``[1,1,32,1024]``
    activation that ``wo`` consumes."""
    k_cache, v_cache, page_table = kv.k, kv.v, kv.page_table
    k_qkv, n_qkv = int(weights.wqkv_decode.shape[-2]), int(weights.wqkv_decode.shape[-1])
    x_sharded = ttnn.to_memory_config(x, O._width_sharded_l1(k_qkv))
    xqkv = ttnn.linear(
        x_sharded,
        weights.wqkv_decode,
        program_config=O._dram_sharded_program_config(k_qkv, n_qkv),
        memory_config=O._width_sharded_l1(n_qkv),
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(x_sharded)
    xqkv = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(xqkv)
    kv_sharded_mem = k.memory_config()
    q = O._per_head_rms_norm(
        ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), weights.attention.q_norm, config.rms_norm_eps
    )
    k = O._per_head_rms_norm(
        ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG), weights.attention.k_norm, config.rms_norm_eps
    )
    q = O._apply_rope(q, cos, sin, token_index=tok_idx)
    k = ttnn.to_memory_config(O._apply_rope(k, cos, sin, token_index=tok_idx), kv_sharded_mem)
    ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=pos, page_table=page_table)
    ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=pos, page_table=page_table)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        k_cache,
        v_cache,
        page_table_tensor=page_table,
        cur_pos_tensor=pos,
        scale=config.head_dim**-0.5,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=None,
    )
    ttnn.deallocate(q)
    return O._concat_heads_decode(attn, config)


def layer_fused_wo_rs(x, weights, config, ctx, cos, sin, kv, pos, tok_idx):
    """The stage-04 layer with the ``wo`` projection and the first reduce-scatter
    replaced by one ``matmul_reduce_scatter_async``.

    This is the fourth collective edge, the one stage 03's "the neighbour is a
    norm or a residual add" argument does not cover. Two costs come with it:
    ``wo`` gives up its DRAM-sharded program config (the fused op takes a 2D
    ``MatmulMultiCoreReuseMultiCast`` one) and the RS workers must be given
    disjoint rows of the grid, which shrinks the matmul's.
    """
    eps = config.global_config.rms_norm_eps
    normed = MC.decode_residual_norm(x, weights.input_layernorm_rm, eps)
    attn = _attention_to_concat_heads(normed, weights.experts, config.local_attention, cos, sin, kv, pos, tok_idx)
    ttnn.deallocate(normed)

    wo = weights.experts.attention.wo  # [1024, 2048] interleaved, K-sharded across dies
    m, k_local = 32, int(wo.shape[-2])
    n = int(wo.shape[-1])
    key = (m, n, k_local)
    if key not in _MMRS:
        grid = (8, 6)
        per_core_n = max(1, math.ceil(n / 32 / grid[0]))
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=min(4, max(1, k_local // 32 // grid[0])),
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=per_core_n,
            out_block_w=max(1, per_core_n // 2),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
            allowed_worker_cores=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))}
            ),
        )
        mk = lambda w: ttnn.from_torch(
            torch.zeros(1, 1, m, w),
            device=ctx.mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(ctx.mesh),
        )
        _MMRS[key] = (pc, mk(n), mk(n // ctx.num_devices))
    pc, interm_buf, out_buf = _MMRS[key]

    attn_i = ttnn.to_memory_config(attn, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn)
    _mm, rs_out = ttnn.experimental.matmul_reduce_scatter_async(
        attn_i,
        wo,
        persistent_intermediate_buffer=interm_buf,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=(0, 6),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=MC._links(x, ctx),
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=ctx.topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        compute_kernel_config=None,
    )
    ttnn.deallocate(attn_i)
    attn_out = ttnn.experimental.all_gather_async(
        rs_out,
        dim=3,
        multi_device_global_semaphore=ctx.ccl.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=ctx.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=MC._links(x, ctx),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ctx.topology,
    )
    hidden = ttnn.add(x, attn_out)
    ttnn.deallocate(attn_out)

    normed_sharded = MC.decode_residual_norm(hidden, weights.post_attention_layernorm_rm, eps)
    routing = MC.router_forward_multichip(
        normed_sharded, weights.router, weights.expert_window, config.global_config.moe, config.local_moe
    )
    normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(normed_sharded)
    moe_partial = MC.moe_decode_multichip(normed, routing, weights.experts, config.local_moe)
    ttnn.deallocate(normed)
    ttnn.deallocate(routing)
    moe_out = MC.all_reduce(moe_partial, ctx)
    ttnn.deallocate(moe_partial)
    out = ttnn.add(hidden, moe_out)
    ttnn.deallocate(hidden)
    ttnn.deallocate(moe_out)
    return out


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
        ("04 default (scatter tail)", MC.decoder_layer_decode_multichip, _scatter_router, _all_reduce),
        ("04 + threshold router tail", MC.decoder_layer_decode_multichip, _threshold_router, _all_reduce),
        ("04 + persistent CCL buffers", MC.decoder_layer_decode_multichip, _scatter_router, all_reduce_persistent),
        ("04 + fused matmul-RS on wo", layer_fused_wo_rs, _scatter_router, _all_reduce),
    ]

    reference = None
    for pass_no in (1, 2):
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
                print(f"P|pass{pass_no} {name:30s} {statistics.median(samples):.4f} ms   ({agree})", flush=True)
            except Exception as exc:
                print(f"P|pass{pass_no} {name:30s} FAILED {str(exc)[:200]}", flush=True)
finally:
    MC.router_forward_multichip = _scatter_router
    MC.all_reduce = MC.all_reduce_decode = _all_reduce
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("P|done")
