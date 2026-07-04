# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""OPT-004: device verify + bench for the tuned matmul geometry of the 5 sparse-MoE matmuls.

WRITE-ONLY — do NOT run here; the QB2 box is owned by another agent. Run on QB2 when the device is free:

    DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
      python models/experimental/diffusion_gemma/doc/optimize_perf/bench_opt004_matmul_geometry.py

What it does (see doc/optimize_perf/opt004_matmul_geometry.md for the geometry rationale):
  1. Builds the real 26B MoE layer-0 (mesh (1,4), TP=4) and prints the device compute grid + the 5
     chosen OPT-004 program configs.
  2. Per matmul (gate/up, down, gather, combine): times UNTUNED (program_config=None, the current
     prototype) vs TUNED on the REAL per-device shapes, and checks tuned-vs-untuned PCC ≈ 1.0 (pure
     geometry change -> same math).
  3. Sweeps candidate geometries per matmul (grid variants, in0_block_w values, + a 1D candidate for
     combine) and reports the fastest LEGAL one per role (illegal candidates log their TT_FATAL).
  4. Times the whole sparse_experts_forward with DG_SPARSE_MOE_TUNED off vs on, plus MoE-output PCC vs
     the dense path — the layer-level win and correctness as one number each.

Markers (grep these):
  RESULT_GRID gx=.. gy=.. cores=..
  RESULT_CFG role=.. <repr>
  RESULT_MATMUL role=.. untuned_ms=.. tuned_ms=.. speedup=.. pcc=..
  RESULT_SWEEP role=.. cand=.. ms=.. legal=..   (one per candidate)
  RESULT_SWEEP_BEST role=.. cand=.. ms=..
  RESULT_FULL_MOE untuned_ms=.. tuned_ms=.. speedup=.. pcc_untuned_vs_dense=.. pcc_tuned_vs_dense=..

Precision/fidelity is held fixed (bf16 act / bfp8 weight, HiFi2) across untuned vs tuned — OPT-004 is a
geometry sweep, so any dtype/fidelity change would invalidate the comparison (per the optimize skill).
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.experts.operations import apply_geglu
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt import sparse_moe as SM

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")
TILE = 32


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0 if a.norm() == b.norm() else 0.0
    return (torch.dot(a, b) / denom).item()


def _time(fn, iters, mesh):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) * 1e3 / iters


def _to_host(t):
    dev = t.device()
    if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return ttnn.to_torch(t).float()


def _matmul_ms_pcc(a, b, pc, ckcfg, iters, mesh):
    """Time a single matmul with the given program_config; return (ms, host_output)."""

    def call():
        out = ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg, program_config=pc)
        out.deallocate(True)

    ms = _time(call, iters, mesh)
    out = ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg, program_config=pc)
    host = _to_host(out)
    out.deallocate(True)
    return ms, host


def _try_matmul_ms(a, b, pc, ckcfg, iters, mesh):
    """Time a matmul, catching illegal configs. Returns (ms or None, legal:bool, err:str)."""
    try:
        ms, host = _matmul_ms_pcc(a, b, pc, ckcfg, iters, mesh)
        del host
        return ms, True, ""
    except Exception as e:  # noqa: BLE001 - a candidate FATAL is a legitimate sweep outcome
        return None, False, f"{type(e).__name__}: {str(e)[:160]}"


def _batched_cfg(mesh, m_t, k_t, n_t, in0_block_w, per_core_M=None):
    """A batched MatmulMultiCoreReuse config candidate (per_core_N forced to Nt)."""
    gx, gy = SM._device_grid(mesh)
    pm = per_core_M if per_core_M is not None else m_t
    sh, sw = SM._pick_out_subblock(pm, n_t)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=n_t,
    )


def _2d_cfg(mesh, m_t, k_t, n_t, in0_block_w, grid=None):
    gx, gy = grid if grid is not None else SM._device_grid(mesh)
    per_core_M = math.ceil(m_t / gy)
    per_core_N = math.ceil(n_t / gx)
    sh, sw = SM._pick_out_subblock(per_core_M, per_core_N)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def _1d_cfg(mesh, m_t, k_t, n_t, in0_block_w, grid=None):
    """A 1D mcast_in0 candidate (narrow-M): whole M in one per_core_M block, N tiled across cores."""
    gx, gy = grid if grid is not None else SM._device_grid(mesh)
    num_cores = gx * gy
    per_core_N = math.ceil(n_t / num_cores)
    sh, sw = SM._pick_out_subblock(m_t, per_core_N)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=m_t,
        out_block_w=per_core_N,
        per_core_M=m_t,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=0,
        untilize_out=False,
    )


def run(num_layers, canvas_length, iters, max_seq_len, capacity):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1300000000)
    try:
        mi = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=max_seq_len, num_layers=num_layers, create_kv_cache=True
        )
        tt_model = mi.tt_model
        H = tt_model.hf_config.hidden_size
        moe = None
        for layer in tt_model.layers:
            if getattr(layer, "enable_moe_block", False):
                moe = layer.moe
                break
        experts = moe.experts
        weights = experts.weights
        cfg = experts.config
        mesh_config = experts.mesh_config
        ccl = experts.ccl_manager
        E = cfg.num_experts
        I = weights.intermediate_size_per_device
        C = capacity
        S = canvas_length
        EC = E * C
        gx, gy = SM._device_grid(mesh)
        print(f"RESULT_GRID gx={gx} gy={gy} cores={gx*gy}", flush=True)
        logger.info(f"E={E} top_k={cfg.top_k} H={H} I/dev={I} S={S} C={C} EC={EC}")

        ckcfg = SM.default_sparse_moe_compute_kernel_config()

        # Print the 5 chosen configs.
        tuned = SM.build_tuned_configs(mesh, E, C, H, I, S)
        for role in ("gate_up", "down", "gather", "combine"):
            print(f"RESULT_CFG role={role} {tuned[role]}", flush=True)

        def mk_hidden(scale=0.1):
            host = torch.randn(1, 1, S, H, dtype=torch.float32) * scale
            return ttnn.from_torch(
                host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        # ---- real routing + a realistic gathered/down buffer for the per-matmul microbench ----
        ri = mk_hidden()
        dense_routing = DF._denoise_router_forward(moe.router, ri)  # [1,1,S,E]
        ri.deallocate(True)
        hidden = mk_hidden()

        disp, comb = SM.build_capacity_dispatch(dense_routing, E, C, cfg.top_k)
        disp_t = ttnn.transpose(disp, 2, 3)  # [1,1,EC,S]
        disp.deallocate(True)
        dispatched = ttnn.matmul(disp_t, hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg)
        gathered = ttnn.reshape(dispatched, (1, E, C, H))  # [1,E,C,H]
        dispatched.deallocate(True)
        # A down_input-shaped tensor [1,E,C,I] for the down microbench.
        gate = ttnn.matmul(
            gathered, weights.gate_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg
        )
        up = ttnn.matmul(gathered, weights.up_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg)
        down_input = apply_geglu(gate, up)  # [1,E,C,I]
        gate.deallocate(True)
        up.deallocate(True)
        down = ttnn.matmul(
            down_input, weights.down_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckcfg
        )
        down_flat = ttnn.reshape(down, (1, 1, EC, H))  # [1,1,EC,H]
        down.deallocate(True)

        Ct, Ht, It, St, ECt = C // TILE, H // TILE, I // TILE, S // TILE, EC // TILE

        # ================= per-matmul untuned vs tuned + PCC =================
        # role -> (a, b, tuned_cfg, family)
        cases = {
            "gate_up": (gathered, weights.gate_proj, tuned["gate_up"], "batched", (Ct, Ht, It)),
            "down": (down_input, weights.down_proj, tuned["down"], "batched", (Ct, It, Ht)),
            "gather": (disp_t, hidden, tuned["gather"], "2d", (ECt, St, Ht)),
            "combine": (comb, down_flat, tuned["combine"], "2d", (St, ECt, Ht)),
        }
        for role, (a, b, tcfg, fam, (m_t, k_t, n_t)) in cases.items():
            un_ms, un_host = _matmul_ms_pcc(a, b, None, ckcfg, iters, mesh)
            tu_ms, tu_host = _matmul_ms_pcc(a, b, tcfg, ckcfg, iters, mesh)
            pcc = _pcc(un_host, tu_host)
            print(
                f"RESULT_MATMUL role={role} untuned_ms={un_ms:.3f} tuned_ms={tu_ms:.3f} "
                f"speedup={un_ms/max(tu_ms,1e-6):.2f} pcc={pcc:.5f}",
                flush=True,
            )
            del un_host, tu_host

            # ---- candidate sweep for this role (builders are lazy so an illegal *construction*
            #      is logged like an illegal run, not a crash) ----
            builders = {}
            if fam == "batched":
                # sweep in0_block_w over divisors of Kt, and per_core_M in {Mt, 2*Mt} (batch pack)
                for bw in [d for d in SM._divisors(k_t) if d <= min(k_t, 44)]:
                    builders[f"pm{m_t}_bw{bw}"] = lambda bw=bw: _batched_cfg(mesh, m_t, k_t, n_t, bw, per_core_M=m_t)
                if (E * m_t) % (2 * m_t) == 0:
                    bw0 = SM._pick_in0_block_w(k_t, n_t)
                    builders[f"pm{2*m_t}_bw{bw0}"] = lambda bw0=bw0: _batched_cfg(
                        mesh, m_t, k_t, n_t, bw0, per_core_M=2 * m_t
                    )
            else:
                for grid in [(gx, gy), (8, 8), (11, gy)]:
                    if grid[0] > gx or grid[1] > gy:
                        continue
                    for bw in [d for d in SM._divisors(k_t) if d <= 16]:
                        builders[f"g{grid[0]}x{grid[1]}_bw{bw}"] = lambda grid=grid, bw=bw: _2d_cfg(
                            mesh, m_t, k_t, n_t, bw, grid=grid
                        )
                if role == "combine":  # narrow-M: try a 1D mcast candidate
                    bw = [d for d in SM._divisors(k_t) if d <= 16][-1]
                    builders["1d_mcast"] = lambda bw=bw: _1d_cfg(mesh, m_t, k_t, n_t, bw)

            best_name, best_ms = None, None
            for name, builder in builders.items():
                try:
                    pc = builder()
                except Exception as e:  # noqa: BLE001 - illegal config construction is a sweep outcome
                    print(
                        f"RESULT_SWEEP role={role} cand={name} ms=None legal=False err=build:{type(e).__name__}",
                        flush=True,
                    )
                    continue
                ms, legal, err = _try_matmul_ms(a, b, pc, ckcfg, iters, mesh)
                print(
                    f"RESULT_SWEEP role={role} cand={name} ms={ms if ms is None else round(ms,3)} legal={legal}"
                    + (f" err={err}" if not legal else ""),
                    flush=True,
                )
                if legal and (best_ms is None or ms < best_ms):
                    best_name, best_ms = name, ms
            if best_name is not None:
                print(f"RESULT_SWEEP_BEST role={role} cand={best_name} ms={best_ms:.3f}", flush=True)

        for t in (disp_t, comb, down_input, down_flat, gathered):
            t.deallocate(True)

        # ================= full sparse MoE: untuned vs tuned + PCC vs dense =================
        expert_input = mk_hidden()
        dense_out = moe.experts(expert_input, dense_routing)
        dense_host = _to_host(dense_out)
        dense_out.deallocate(True)

        def full(tuned_flag):
            os.environ["DG_SPARSE_MOE_TUNED"] = "1" if tuned_flag else "0"
            out = SM.sparse_experts_forward(experts, expert_input, dense_routing, capacity=C)
            return out

        os.environ["DG_SPARSE_MOE_TUNED"] = "0"
        un_ms = _time(lambda: full(False).deallocate(True), iters, mesh)
        os.environ["DG_SPARSE_MOE_TUNED"] = "1"
        tu_ms = _time(lambda: full(True).deallocate(True), iters, mesh)

        un_out = full(False)
        un_pcc = _pcc(dense_host, _to_host(un_out))
        un_out.deallocate(True)
        tu_out = full(True)
        tu_pcc = _pcc(dense_host, _to_host(tu_out))
        tu_out.deallocate(True)
        os.environ["DG_SPARSE_MOE_TUNED"] = "0"

        print(
            f"RESULT_FULL_MOE untuned_ms={un_ms:.3f} tuned_ms={tu_ms:.3f} speedup={un_ms/max(tu_ms,1e-6):.2f} "
            f"pcc_untuned_vs_dense={un_pcc:.5f} pcc_tuned_vs_dense={tu_pcc:.5f}",
            flush=True,
        )

        expert_input.deallocate(True)
        dense_routing.deallocate(True)
        hidden.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--capacity", type=int, default=32)
    args = ap.parse_args()
    run(args.num_layers, args.canvas_length, args.iters, args.max_seq_len, args.capacity)


if __name__ == "__main__":
    main()
