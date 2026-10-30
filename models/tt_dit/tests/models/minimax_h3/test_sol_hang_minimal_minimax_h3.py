"""Smallest reproducer of the multi-pass resident hang (jobs 162394, 163714, 163912, 164002; root cause 2026-10-29).

Synthetic Q/K/V, then the two ops that were on the device when it stuck, back to back and without any host
synchronisation in between, many times: the fused pre-pass (generic_op pool + route; the route kernel's step-9
constants multicast is where the stuck core sat) followed by the resident ring joint SDPA, once per Q pass (the ring
peers were already in the next pass's ring op while one chip was still in the route program). The 10 s / 15 s
geometries hung; 5 s never did (there every constant block fits one 16 KB NoC burst).

The fix (commit 13287576827: per-core constant reads) stays in the tree. This test flips the kept switch instead:
    SOL_HANG_MCAST=1 (default here)  route constants via the NoC multicast -> the old, hanging path
    SOL_HANG_MCAST=0                 per-core reads (production default)   -> must run clean
Env: SOL_HANG_GEOM (r9184k256p2 = 10 s / 2 passes, r13664k256p3 = 15 s / 3 passes, r4736k384p1 = 5 s control),
SOL_HANG_LAYERS (1000; the old tree hung after 350-400 and 850-900 layers). Run with TT_METAL_OPERATION_TIMEOUT_SECONDS
set so a hang trips the op timeout instead of blocking for ever.
"""
from __future__ import annotations

import os
import time

import torch

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import SolGeometry
from ....models.transformers.minimax_h3.sol_prepass_fused import SolPrepassFused
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor, from_torch
from .common import GALAXY_RING
from .test_sol_compact_minimax_h3 import D, H, Q1_CHUNK

GEOMS = {  # rows per shard, logical_n, k_chunk, passes
    "r4736k384p1": (4736, 37888, 384, 1),
    "r9184k256p2": (9184, 73472, 256, 2),
    "r13664k256p3": (13664, 109312, 256, 3),
}


@GALAXY_RING
def test_sol_hang_minimal(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    rows, logical_n, k_chunk, passes = GEOMS[os.environ.get("SOL_HANG_GEOM", "r9184k256p2")]
    layers = int(os.environ.get("SOL_HANG_LAYERS", "1000"))
    mcast = os.environ.get("SOL_HANG_MCAST", "1")
    os.environ["SOL_ROUTE_CONST_MCAST"] = mcast  # read by SolPrepassFused.prepare()
    SolPrepassFused.new_generation()  # drop any shared state prepared with the other setting
    S = tuple(mesh_device.shape)[sp_axis]
    assert S == 8
    scale, sink_tokens, qc = D**-0.5, 849, Q1_CHUNK

    # synthetic inputs: one Q/K/V set, uploaded once
    g = torch.Generator().manual_seed(11)
    q, k, v = (torch.randn((S, H, rows, D), generator=g) * s for s in (0.5, 0.5, 1.0))
    mesh_axes = [sp_axis, None, None, None]
    tq, tk, tv = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
        for x in (q, k, v)
    )
    empty_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)

    # op 1: fused pre-pass (pool + route)
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    pre = SolPrepassFused(mesh_device=mesh_device, ccl_manager=ccl, sp_axis=sp_axis, tau=1.0, mode="compact")
    pre.prepare(SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=qc, k_chunk=k_chunk), sink_tokens, scale)

    # op 2: resident ring joint SDPA, one call per Q pass
    grid = mesh_device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=qc,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    nqc = -(-rows // qc)

    def layer():
        sol = pre(tq, tk, tv, scale=scale, report_density=False)
        mask = sol["block_mask"]
        outs = []
        for p in range(passes):
            c0, c1 = nqc * p // passes, nqc * (p + 1) // passes
            q_p = ttnn.slice(tq, [0, 0, c0 * qc, 0], [1, H, min(rows, c1 * qc), D])
            mask_p = ttnn.slice(mask, [0, 0, c0, 0], [1, mask.shape[1], c1, mask.shape[3]])
            out_p, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_p,
                tk,
                tv,
                empty_joint,
                empty_joint,
                empty_joint,
                persistent_output_buffer_k=ccl.get_ag_ping_pong_buffer(tk.shape, 2, sp_axis, dtype=ttnn.bfloat16),
                persistent_output_buffer_v=ccl.get_ag_ping_pong_buffer(tk.shape, 2, sp_axis, dtype=ttnn.bfloat16),
                joint_strategy="rear",
                logical_n=int(logical_n),
                program_config=program_config,
                compute_kernel_config=ckc,
                dim=2,
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(sp_axis),
                num_links=num_links,
                cluster_axis=sp_axis,
                mesh_device=mesh_device,
                topology=topology,
                subdevice_id=ccl.ccl_sub_device_id,
                ccl_core_grid_offset=(grid.x - 1, 0),
                use_column_major_ccl=True,
                sol_block_mask=mask_p,
                sol_pooled_k=sol["pooled_k"],
                sol_pooled_v=sol["pooled_v"],
                sol_mode="resident",
                is_cross=True,
            )
            ttnn.deallocate(q_p)
            ttnn.deallocate(mask_p)
            outs.append(out_p)
        out = ttnn.concat(outs, dim=2)
        for t in outs + [mask, sol["pooled_k"], sol["pooled_v"]]:
            ttnn.deallocate(t)
        return out

    def host(out):
        dims = [1, 2] if sp_axis == 1 else [2, 1]
        return ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
        )[0, :H]

    ref = layer()
    ttnn.synchronize_device(mesh_device)
    ref_h = host(ref)
    ttnn.deallocate(ref)
    print(
        f"[hang] SOL_ROUTE_CONST_MCAST={mcast} rows {rows} logical_n {logical_n} k_chunk {k_chunk} passes {passes}; "
        f"{layers} layers, no host sync between layers; start {time.strftime('%H:%M:%S')}",
        flush=True,
    )
    t0 = time.perf_counter()
    for i in range(layers):
        out = layer()
        if (i + 1) % 100 == 0 or i + 1 == layers:
            ttnn.synchronize_device(mesh_device)
            mism = int((host(out) != ref_h).sum())
            print(f"[hang] layers {i + 1:5d} ok; {(time.perf_counter() - t0) * 1e3 / (i + 1):.1f} ms/layer; mismatches vs layer 0: {mism}", flush=True)
            assert mism == 0, f"layer {i + 1}: {mism} output elements differ from layer 0 (same inputs)"
        ttnn.deallocate(out)
    print(f"[hang] DONE {layers} layers in {time.perf_counter() - t0:.0f} s with SOL_ROUTE_CONST_MCAST={mcast}", flush=True)
