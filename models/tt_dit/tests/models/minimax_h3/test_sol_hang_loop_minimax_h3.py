"""Hang-rate loop for the multi-pass resident Sol-Attn attention path (written 2026-10-29 after hangs in jobs 162394 and
163714, both mid-generation after ~1900-2000 layer calls, both stuck on logical core (9,8)).

Mirrors what one transformer layer does on the attention path, thousands of times, without the rest of the model:
fresh Q/K/V device buffers per layer (allocation churn like the projections), the FUSED pre-pass (pool + two all-gathers
with the module's default gather configuration, or SOL_AG_KW from the environment + route), then the resident ring op
once per pass over 64-row-aligned Q ranges with that range's rows of the block mask, per-pass Q/mask slices freed right
after the enqueue, outputs concatenated and freed - exactly AttentionMiniMaxH3._sol_resident_multipass. The host is NOT
synchronised per layer (it runs ahead as in the generation loop); it synchronises every SOL_LOOP_SYNC_EVERY layers,
reads one output back and compares it with layer 0's (same inputs -> must be bit-identical: silent corruption shows up
as mismatching elements), and prints progress. Host-side aliasing check (v2): the bank-local address ranges of the
per-pass Q / mask slices freed while their ring op may still be in flight are kept per layer, and the next layer's
pre-pass outputs (gathered pooled K / V, which remote chips write into, and the block mask) are checked against them;
an overlap means the allocator handed a just-freed in-flight address to a buffer another chip writes into (the
all-gather DRAM aliasing hazard) and is counted as an alias event. A hang trips the op timeout (TT_METAL_OPERATION_TIMEOUT_SECONDS) and
the job's triage hook.
Env: SOL_LOOP_GEOM (r9184k256p2 = 10 s / 2 passes; r13664k256p3 = 15 s / 3 passes; r15680k256p3 = the client's shard;
r4736k384p1 = 5 s single pass), SOL_LOOP_LAYERS (6000), SOL_LOOP_SYNC_EVERY (50), SOL_LOOP_TAU (1.0),
SOL_LOOP_SEEDS (2: distinct Q/K/V sets cycled), SOL_AG_KW / TT_SOL_KV_PAGES as in production."""
from __future__ import annotations

import os
import time

import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import SolGeometry
from ....models.transformers.minimax_h3.sol_prepass_fused import SolPrepassFused
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor, from_torch
from .common import GALAXY_RING
from .test_sol_compact_minimax_h3 import D, H, Q1_CHUNK, _qkv

GEOMS = {
    "r4736k384p1": (4736, 37888, 384, 1),
    "r9184k256p2": (9184, 73472, 256, 2),
    "r13664k256p3": (13664, 109312, 256, 3),
    "r15680k256p3": (15680, 125440, 256, 3),
}


@GALAXY_RING
def test_sol_hang_loop(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    geom = os.environ.get("SOL_LOOP_GEOM", "r9184k256p2")
    rows, logical_n, k_chunk, passes = GEOMS[geom]
    layers = int(os.environ.get("SOL_LOOP_LAYERS", "6000"))
    sync_every = int(os.environ.get("SOL_LOOP_SYNC_EVERY", "50"))
    tau = float(os.environ.get("SOL_LOOP_TAU", "1.0"))
    n_seeds = int(os.environ.get("SOL_LOOP_SEEDS", "2"))
    S = tuple(mesh_device.shape)[sp_axis]
    assert S == 8
    scale = D**-0.5
    sink_tokens = 849
    qc = Q1_CHUNK
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=qc, k_chunk=k_chunk)
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    mesh_axes = [sp_axis, None, None, None]
    pre = SolPrepassFused(mesh_device=mesh_device, ccl_manager=ccl, sp_axis=sp_axis, tau=tau, mode="compact")
    pre.prepare(geo, sink_tokens, scale)
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
    empty_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)
    nqc = -(-rows // qc)
    base = []
    for s in range(n_seeds):
        q, k, v = _qkv(S, rows, 11 + s)
        base.append(
            tuple(
                from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
                for x in (q, k, v)
            )
        )
    print(
        f"[loop] geom {geom}: rows {rows} logical_n {logical_n} k_chunk {k_chunk} passes {passes}; layers {layers}, sync every {sync_every}, "
        f"tau {tau}, seeds {n_seeds}, gather kwargs {pre.gather_kwargs}, persistent gathers {pre.gather_persistent}, own gather sems {getattr(pre, 'gather_own_sems', None)}, "
        f"TT_SOL_KV_PAGES={os.environ.get('TT_SOL_KV_PAGES', 'unset')}",
        flush=True,
    )

    n_banks = mesh_device.num_dram_channels() if hasattr(mesh_device, "num_dram_channels") else 8

    def rng(t):
        # bank-local [base, base + bytes_per_bank) of an interleaved DRAM buffer (the same range in every bank)
        nbytes = 1
        for d in t.shape:
            nbytes *= int(d)
        nbytes *= 2 if t.get_dtype() == ttnn.bfloat16 else 4
        return (t.buffer_address(), t.buffer_address() + -(-nbytes // n_banks))

    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    state = {"freed_prev": [], "alias_events": 0, "alias_first": None}

    def layer(tq0, tk0, tv0, idx=-1):
        # fresh buffers per layer, as the projections produce them in the model
        tq = ttnn.multiply(tq0, 1.0)
        tk = ttnn.multiply(tk0, 1.0)
        tv = ttnn.multiply(tv0, 1.0)
        sol = pre(tq, tk, tv, scale=scale, report_density=False)
        mask = sol["block_mask"]
        # host-side aliasing check against the previous layer's freed in-flight slices
        for name, t in (("pooled_k", sol["pooled_k"]), ("pooled_v", sol["pooled_v"]), ("mask", mask)):
            r = rng(t)
            for fname, fr in state["freed_prev"]:
                if overlaps(r, fr):
                    state["alias_events"] += 1
                    if state["alias_first"] is None:
                        state["alias_first"] = (idx, name, fname, hex(r[0]), hex(fr[0]))
        freed_now = []
        outs = []
        for p in range(passes):
            c0, c1 = nqc * p // passes, nqc * (p + 1) // passes
            r0, r1 = c0 * qc, min(rows, c1 * qc)
            q_p = ttnn.slice(tq, [0, 0, r0, 0], [1, H, r1, D])
            mask_p = ttnn.slice(mask, [0, 0, c0, 0], [1, mask.shape[1], c1, mask.shape[3]])
            pk = ccl.get_ag_ping_pong_buffer(tk.shape, 2, sp_axis, dtype=ttnn.bfloat16)
            pv = ccl.get_ag_ping_pong_buffer(tk.shape, 2, sp_axis, dtype=ttnn.bfloat16)
            sems = ccl.get_ag_ping_pong_semaphore(sp_axis)
            out_p, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_p,
                tk,
                tv,
                empty_joint,
                empty_joint,
                empty_joint,
                persistent_output_buffer_k=pk,
                persistent_output_buffer_v=pv,
                joint_strategy="rear",
                logical_n=int(logical_n),
                program_config=program_config,
                compute_kernel_config=ckc,
                dim=2,
                multi_device_global_semaphore=sems,
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
            freed_now.append((f"q_p{p}", rng(q_p)))
            freed_now.append((f"mask_p{p}", rng(mask_p)))
            ttnn.deallocate(q_p)
            ttnn.deallocate(mask_p)
            outs.append(out_p)
        state["freed_prev"] = freed_now
        out = ttnn.concat(outs, dim=2)
        for t in outs:
            ttnn.deallocate(t)
        for t in (mask, sol["pooled_k"], sol["pooled_v"]):
            ttnn.deallocate(t)
        ttnn.deallocate(tq)
        ttnn.deallocate(tk)
        ttnn.deallocate(tv)
        return out

    def host(out):
        return ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=[1, 2] if sp_axis == 1 else [2, 1], mesh_shape=tuple(mesh_device.shape)
            ),
        )[0, :H]

    # reference outputs per seed (layer 0..n_seeds-1), compiled calls
    refs = []
    for s in range(n_seeds):
        out = layer(*base[s])
        ttnn.synchronize_device(mesh_device)
        refs.append(host(out))
        ttnn.deallocate(out)
    print(f"[loop] reference outputs taken (compile done); starting {layers} layers at {time.strftime('%H:%M:%S')}", flush=True)
    t_start = time.perf_counter()
    t_last = t_start
    mism_total = 0
    for i in range(layers):
        s = i % n_seeds
        out = layer(*base[s], idx=i)
        if (i + 1) % sync_every == 0 or i + 1 == layers:
            ttnn.synchronize_device(mesh_device)
            got = host(out)
            mism = int((got != refs[s]).sum())
            mism_total += mism
            now = time.perf_counter()
            print(
                f"[loop] layers {i + 1:6d} ok; {(now - t_last) * 1e3 / sync_every:.1f} ms/layer over the last {sync_every}; "
                f"mismatching elements vs layer-0 output: {mism} (total {mism_total}); host alias events {state['alias_events']}"
                f"{'' if state['alias_first'] is None else ' first ' + str(state['alias_first'])}; {time.strftime('%H:%M:%S')}",
                flush=True,
            )
            t_last = now
        ttnn.deallocate(out)
    total = time.perf_counter() - t_start
    print(
        f"[loop] DONE {layers} layers x {passes} passes in {total:.0f} s ({total * 1e3 / layers:.1f} ms/layer), mismatching elements total {mism_total}, "
        f"host alias events {state['alias_events']} (first {state['alias_first']})",
        flush=True,
    )
    assert mism_total == 0, f"{mism_total} output elements differed from the layer-0 output across the loop"
