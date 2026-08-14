# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-04 layer A/B, round 3: re-audit of the *stage-04* profile.

Every round of this model's optimization has promoted a new top op that the
previous audit dismissed, so the profile was re-read after the norm and router
changes landed. That profile was taken *before* the persistent collective
buffers of section 5 were adopted, so its window is 66 ops / **367.624 us** --
a **superseded, intermediate** capture, not the shipped one, which is 68 ops /
**362.828 us** (device 0, rows 154-221, re-derived by ``window.py`` and asserted
by ``check_published_figures.py``). The two findings it produced are unchanged
by that and the shipped figures are given beside them.

Note: 367.624 is the number stage 04's forward pointer in
``../../multichip_decoder/README.md`` once published as the *shipped* device
time. It came from here. Review caught it; both are now labelled for what they
are.

* **The two reduce-scatters cost 22.218 and 15.018 us for the same shape**
  (**18.871 and 15.018** in the shipped profile, rows 176 and 218). The only difference between them
  is where their input sits: the first reads the DRAM-interleaved tensor
  ``attention_decode_optimized`` returns, the second reads the L1-interleaved
  expert sum. Over 7 us on one op, and the same asymmetry is in the stage-03
  profile (20.413 vs 16.322) where nobody looked at it.
* **``TopK`` is 26.400 us on one core and is now the largest single op in the
  layer outside the expert matmuls** -- 7.2% of it, with a 4.181 us ``FillPad``
  in front of it (**26.356 and 4.190** in the shipped profile, rows 184 and 183).

Legs, each run twice in interleaved passes:

    04 default                     the current shipped path
    04 + attention out in L1       feed the first reduce-scatter from L1
    04 + persistent CCL buffers    with and without the clone
    04 + topk input in L1

    python layer_levers3.py

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
_router = MC.router_forward_multichip


def all_reduce_l1_in(x, ctx):
    """Stage that the caller's tensor into L1 before the reduce-scatter reads it."""
    xl = x if x.memory_config().buffer_type == ttnn.BufferType.L1 else ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    out = _all_reduce(xl, ctx)
    if xl is not x:
        ttnn.deallocate(xl)
    return out


_PERSIST = {}


def _persist_bufs(x, ctx):
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
    return _PERSIST[key]


def _persist_ar(x, ctx, clone):
    interm, penult, rs_out, ag_out = _persist_bufs(x, ctx)
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
    return ttnn.clone(gathered, memory_config=ttnn.DRAM_MEMORY_CONFIG) if clone else gathered


def ar_persist_clone(x, ctx):
    return _persist_ar(x, ctx, True)


def ar_persist_l1_in(x, ctx):
    xl = x if x.memory_config().buffer_type == ttnn.BufferType.L1 else ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    out = _persist_ar(xl, ctx, True)
    if xl is not x:
        ttnn.deallocate(xl)
    return out


def router_topk_l1(x, w_router, window, config, local_moe):
    """The shipped router with the logits staged in L1 before ``topk``."""
    logits = ttnn.linear(x, w_router, dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    return MC._router_tail(logits, window, config, local_moe, x.device())


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
        ("04 default", _router, _all_reduce),
        ("04 + attn out in L1", _router, all_reduce_l1_in),
        ("04 + persistent CCL", _router, ar_persist_clone),
        ("04 + persistent CCL + L1 in", _router, ar_persist_l1_in),
        ("04 + topk logits in L1", router_topk_l1, all_reduce_l1_in),
    ]

    reference = None
    for pass_no in (1, 2):
        for name, router_fn, ar in LEGS:
            MC.router_forward_multichip = router_fn
            MC.all_reduce = MC.all_reduce_decode = ar

            def step():
                return MC.decoder_layer_decode_multichip(tok, weights, cfg, ctx, cos, sin, kv, pos, CTX - 1)

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
    MC.router_forward_multichip = _router
    MC.all_reduce = MC.all_reduce_decode = _all_reduce
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("P|done")
