# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage single-mesh + point_to_point + trace prototype for the DENOISE stage.

Runs the 18 real AdaRMS expert layers as ONE SPMD computation on a 6-chip mesh
(each chip holds 3 layers as sharded weights), chaining chips with
ttnn.point_to_point, captured as a single trace. Validates:
  - non-empty trace captured on the single mesh
  - traced replay == eager
  - eager output PCC vs torch AdaRMSGemmaBlock chain (same weights)

v1 simplification: SELF-attention only (past_key_value=None) — no prefix KV
cross-attention yet. That isolates the single-mesh + p2p + trace mechanism on
real expert ops before adding KV sharding.

Run (FABRIC_1D; the chain is a 6-chip column line):
  tt-smi -glx_reset
  PI05_CHECKPOINT_DIR=.../weights/pi05_libero_upstream TT_TRACE_POPULATE_DEBUG=1 \
    timeout 400 env PYTHONPATH=$PWD TT_METAL_HOME=$PWD \
    python_env/bin/python models/experimental/pi0_5/tests/perf/_trace_denoise_stage_repro.py
"""

import os
import sys

import torch
import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_gemma import AdaRMSGemmaBlock, precompute_freqs_cis
from models.experimental.pi0_5.tt.ttnn_gemma import (
    AdaRMSGemmaBlockTTNN,
    ada_rms_norm_no_gate_ttnn,
    precompute_freqs_cis_meta_format,
)
from models.experimental.pi0_5.tt.tt_bh_glx.expert_slice import _inject_adarms_weights_to_submesh
from models.experimental.pi0_5.tt.tt_bh_glx.suffix_slice import SuffixSlice
from models.experimental.pi0_5.tt.tt_bh_glx.vlm_slice import _load_block_weights_to_submesh

RUN_FULL = os.environ.get("REPRO_FULL", "1") == "1"  # full Euler loop (suffix+head+loop)
N_STEPS = int(os.environ.get("REPRO_STEPS", "5"))

CKPT = os.environ.get(
    "PI05_CHECKPOINT_DIR",
    "/home/tt-admin/sdawle/tt-metal/models/experimental/pi0_5/weights/pi05_libero_upstream",
)
N_CHIPS = 6
N_PER = 3
DEPTH = N_CHIPS * N_PER  # 18
SUFFIX = 32  # padded action-horizon (one tile)
PREFIX = int(os.environ.get("REPRO_PREFIX", "64"))  # prefix-KV length (0 = self-attn only)
SEED = 0


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return (torch.dot(a, b).item() / denom) if denom > 0 else 0.0


def main():
    def log(m):
        print(f"[denoise-repro] {m}", flush=True)

    # action_horizon MUST come from the checkpoint (this base ships 10 -> pad 32);
    # the bare-config default is 50 -> pad 64, which is wrong for this checkpoint.
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint

    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CKPT))
    ecfg = cfg.expert_config
    # Suffix length = padded action horizon (the real denoise suffix length).
    SUFFIX = ((cfg.action_horizon + 31) // 32) * 32
    log(
        f"expert width={ecfg.width} heads={ecfg.num_heads} kv={ecfg.num_kv_heads} head_dim={ecfg.head_dim} depth={ecfg.depth} suffix={SUFFIX}"
    )
    loader = Pi0_5WeightLoader(CKPT)
    expert_weights = loader.categorized_weights["action_expert"]

    torch.manual_seed(SEED)
    hidden0 = torch.randn(1, SUFFIX, ecfg.width, dtype=torch.float32) * 0.1
    cond0 = torch.randn(1, ecfg.width, dtype=torch.float32) * 0.1

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    submeshes = []
    try:
        denoise = parent.create_submesh(ttnn.MeshShape(N_CHIPS, 1), ttnn.MeshCoordinate(0, 0))
        submeshes.append(denoise)
        tmp = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(7, 0))  # weight-build scratch
        submeshes.append(tmp)
        log(f"denoise mesh shape={denoise.shape}")

        # 1) Build each layer's processed weights via the existing loader on a 1x1
        #    scratch mesh, pull to torch (+ record dtype/layout), then deallocate.
        per_layer = []
        meta = {}  # key -> (ttnn_dtype, ttnn_layout)
        for L in range(DEPTH):
            bw = _load_block_weights_to_submesh(expert_weights, L, tmp)
            _inject_adarms_weights_to_submesh(bw, expert_weights, L, tmp)
            d = {}
            for k, v in bw.items():
                d[k] = ttnn.to_torch(v)
                if k not in meta:
                    meta[k] = (v.dtype, v.layout)
                ttnn.deallocate(v)
            per_layer.append(d)
        log(f"gathered processed weights for {DEPTH} layers, {len(meta)} keys/layer")

        # 2) Build 3 "local-layer" expert blocks on the 6-chip mesh, each with
        #    per-chip-sharded weights: chip c holds global layer (c*N_PER + local).
        def build_block(local_idx):
            layers = [c * N_PER + local_idx for c in range(N_CHIPS)]
            sharded = {}
            for k in per_layer[0].keys():
                stacked = torch.stack([per_layer[L][k] for L in layers], dim=0)  # (6, *)
                dt, lay = meta[k]
                sharded[k] = ttnn.from_torch(
                    stacked,
                    dtype=dt,
                    layout=lay,
                    device=denoise,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(denoise, dim=0),
                )
            return AdaRMSGemmaBlockTTNN(ecfg, sharded, local_idx, denoise, None, None)

        blocks = [build_block(L) for L in range(N_PER)]
        log("built 3 local-layer blocks with sharded weights")

        # Synthetic prefix KV (one (1,1,PREFIX,head_dim) per layer), bf8_b to match
        # the real KV migration. Sharded so chip c holds layers 3c..3c+2's KV.
        kv_K = kv_V = None
        past_k_m = past_v_m = [None] * N_PER
        if PREFIX > 0:
            torch.manual_seed(SEED + 1)
            kv_K = [torch.randn(1, 1, PREFIX, ecfg.head_dim, dtype=torch.float32) * 0.1 for _ in range(DEPTH)]
            kv_V = [torch.randn(1, 1, PREFIX, ecfg.head_dim, dtype=torch.float32) * 0.1 for _ in range(DEPTH)]

            def shard_kv(kvlist, local_idx):
                layers = [c * N_PER + local_idx for c in range(N_CHIPS)]
                stacked = torch.cat([kvlist[L] for L in layers], dim=0)  # (6,1,PREFIX,hd)
                return ttnn.from_torch(
                    stacked,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=denoise,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(denoise, dim=0),
                )

            past_k_m = [shard_kv(kv_K, L) for L in range(N_PER)]
            past_v_m = [shard_kv(kv_V, L) for L in range(N_PER)]
            log(f"built sharded prefix KV (PREFIX={PREFIX}, bf8_b)")

        def torch_kv(L):
            return (kv_K[L], kv_V[L]) if PREFIX > 0 else None

        # cos/sin (replicated) + adarms_cond (replicated) + input (replicated).
        cos_m, sin_m = precompute_freqs_cis_meta_format(ecfg.head_dim, SUFFIX, denoise, base=ecfg.rope_base)

        def replicate(t, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=denoise,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(denoise),
            )

        # ttnn block wants adarms_cond as (B, 1, W) for clean broadcast over seq;
        # torch reference takes (B, W). (matches test_pcc_expert_block_drilldown.)
        cond_m = replicate(cond0.unsqueeze(1))
        hidden_m0 = replicate(hidden0)
        log("uploaded cos/sin + cond + input")

        # ---- single-block diagnostic: ttnn layer0 (chip0) vs torch layer0 ----
        cos_t, sin_t = precompute_freqs_cis(ecfg.head_dim, SUFFIX, ecfg.rope_base)
        sb_out, _ = blocks[0].forward(
            hidden_m0, cos_m, sin_m, cond_m, None, None, (past_k_m[0], past_v_m[0]) if PREFIX > 0 else None, False
        )
        ttnn.synchronize_device(denoise)
        sb_ttnn = ttnn.to_torch(sb_out, mesh_composer=ttnn.ConcatMeshToTensor(denoise, dim=0))[0]
        lw0 = {k[len("model.layers.0.") :]: v for k, v in expert_weights.items() if k.startswith("model.layers.0.")}
        sb_ref, _ = AdaRMSGemmaBlock(ecfg, lw0, 0).forward(hidden0, cos_t, sin_t, cond0, None, None, torch_kv(0), False)
        log(f"PCC single-block layer0 ttnn-vs-torch = {_pcc(sb_ttnn, sb_ref):.6f}")

        def chain(h):
            cur = h
            for c in range(N_CHIPS):
                for L in range(N_PER):
                    kv = (past_k_m[L], past_v_m[L]) if PREFIX > 0 else None
                    cur, _ = blocks[L].forward(cur, cos_m, sin_m, cond_m, None, None, kv, False)
                if c < N_CHIPS - 1:
                    # Forward hand-off c -> c+1. point_to_point(input, FROM, TO) — arg1
                    # is the source (the C++ receiver/sender names are misleading).
                    # Normalize to DRAM-interleaved TILE first or p2p silently writes
                    # NaN to the receiver (see SigLIP debug).
                    cur = ttnn.to_layout(cur, ttnn.TILE_LAYOUT)
                    cur = ttnn.to_memory_config(cur, ttnn.DRAM_MEMORY_CONFIG)
                    cur = ttnn.point_to_point(
                        cur,
                        ttnn.MeshCoordinate(c, 0),
                        ttnn.MeshCoordinate(c + 1, 0),
                        topology=ttnn.Topology.Linear,
                        output_tensor=cur,
                    )
            return cur

        log("warmup (eager) chain...")
        eager_out = chain(hidden_m0)
        ttnn.synchronize_device(denoise)
        eager_last = ttnn.to_torch(eager_out, mesh_composer=ttnn.ConcatMeshToTensor(denoise, dim=0))[N_CHIPS - 1]
        log("eager chain OK")

        log("begin_trace_capture")
        tid = ttnn.begin_trace_capture(denoise, cq_id=0)
        traced_out = chain(hidden_m0)
        log("end_trace_capture (HANG POINT in the old per-chip-submesh approach)")
        ttnn.end_trace_capture(denoise, tid, cq_id=0)
        log("END_TRACE_CAPTURE OK")
        ttnn.execute_trace(denoise, tid, cq_id=0, blocking=True)
        log("EXECUTE_TRACE OK")
        traced_last = ttnn.to_torch(traced_out, mesh_composer=ttnn.ConcatMeshToTensor(denoise, dim=0))[N_CHIPS - 1]

        # 3) torch reference: 18-layer AdaRMS chain with the same raw weights.
        cos_t, sin_t = precompute_freqs_cis(ecfg.head_dim, SUFFIX, ecfg.rope_base)
        ref = hidden0
        for L in range(DEPTH):
            lw = {
                k[len(f"model.layers.{L}.") :]: v
                for k, v in expert_weights.items()
                if k.startswith(f"model.layers.{L}.")
            }
            rb = AdaRMSGemmaBlock(ecfg, lw, L)
            ref, _ = rb.forward(ref, cos_t, sin_t, cond0, None, None, torch_kv(L), False)

        # Control: torch "only chip5's 3 layers (15,16,17) on the raw input" — if
        # eager_last matches THIS (not the full chain), p2p isn't propagating.
        tail = hidden0
        for L in (15, 16, 17):
            lw = {
                k[len(f"model.layers.{L}.") :]: v
                for k, v in expert_weights.items()
                if k.startswith(f"model.layers.{L}.")
            }
            tail, _ = AdaRMSGemmaBlock(ecfg, lw, L).forward(tail, cos_t, sin_t, cond0, None, None, torch_kv(L), False)

        log(f"PCC eager-vs-traced            = {_pcc(eager_last, traced_last):.6f}")
        log(f"PCC eager-vs-torch FULL chain  = {_pcc(eager_last, ref):.6f}")
        log(f"PCC eager-vs-torch TAIL(15-17) = {_pcc(eager_last, tail):.6f}  (high => p2p NOT propagating)")
        log(f"traced finite={torch.isfinite(traced_last).all().item()}")

        # ================= full traced denoise stage (task 5) =================
        if RUN_FULL:
            log("=== full denoise stage: suffix embed + adaRMS head + Euler loop ===")
            ah = cfg.action_horizon
            ah_pad = ((ah + 31) // 32) * 32
            adim = cfg.action_dim
            assert ah_pad == SUFFIX, f"ah_pad={ah_pad} must equal SUFFIX={SUFFIX} (cos table built for SUFFIX)"

            def repl(t, dtype=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG):
                return ttnn.from_torch(
                    t,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=denoise,
                    memory_config=mc,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(denoise),
                )

            suffix_cfg = SuffixConfig(action_dim=adim, action_horizon=ah, expert_width=ecfg.width, pi05=True)
            suffix = SuffixSlice(suffix_cfg, loader.categorized_weights["pi0_projections"], denoise)
            head_w = repl(expert_weights["model.norm.dense.weight"].T.contiguous())
            _mb = expert_weights.get("model.norm.dense.bias")
            head_b = repl(_mb.reshape(1, -1)) if _mb is not None else None
            dg = denoise.compute_with_storage_grid_size()
            head_grid = ttnn.CoreGrid(y=dg.y, x=dg.x)

            timesteps = [1.0 - i / N_STEPS for i in range(N_STEPS + 1)]
            dts = [timesteps[i + 1] - timesteps[i] for i in range(N_STEPS)]
            conds = [
                suffix.embed_adarms_cond(repl(torch.tensor([timesteps[i]], dtype=torch.float32)))
                for i in range(N_STEPS)
            ]

            torch.manual_seed(SEED + 7)
            noise = torch.zeros(1, ah_pad, adim, dtype=torch.float32)
            noise[:, :ah, :] = torch.randn(1, ah, adim)

            def euler_loop(x_t):
                for i in range(N_STEPS):
                    cond = conds[i]
                    x_bf16 = ttnn.typecast(x_t, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
                    cur = suffix.embed_actions(x_bf16)
                    for c in range(N_CHIPS):
                        for L in range(N_PER):
                            kv = (past_k_m[L], past_v_m[L]) if PREFIX > 0 else None
                            cur, _ = blocks[L].forward(cur, cos_m, sin_m, cond, None, None, kv, False)
                        if c < N_CHIPS - 1:
                            cur = ttnn.to_memory_config(ttnn.to_layout(cur, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                            cur = ttnn.point_to_point(
                                cur,
                                ttnn.MeshCoordinate(c, 0),
                                ttnn.MeshCoordinate(c + 1, 0),
                                topology=ttnn.Topology.Linear,
                                output_tensor=cur,
                            )
                    normed = ada_rms_norm_no_gate_ttnn(cur, cond, head_w, head_b, ecfg.rms_norm_eps, head_grid)
                    vel = suffix.project_output(normed)  # chip5 lane (1, ah_pad, adim)
                    vel = ttnn.to_memory_config(ttnn.to_layout(vel, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                    vel = ttnn.point_to_point(
                        vel,
                        ttnn.MeshCoordinate(N_CHIPS - 1, 0),
                        ttnn.MeshCoordinate(0, 0),
                        topology=ttnn.Topology.Linear,
                        output_tensor=vel,
                    )
                    v32 = ttnn.typecast(vel, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
                    v_scaled = ttnn.mul(v32, dts[i], memory_config=ttnn.L1_MEMORY_CONFIG)
                    ttnn.add(x_t, v_scaled, output_tensor=x_t)

            # eager
            xt_e = repl(noise, dtype=ttnn.float32, mc=ttnn.L1_MEMORY_CONFIG)
            euler_loop(xt_e)
            ttnn.synchronize_device(denoise)
            fe = ttnn.to_torch(xt_e, mesh_composer=ttnn.ConcatMeshToTensor(denoise, dim=0))[0][:ah, :]
            log(f"full-loop eager: finite={torch.isfinite(fe).all().item()} mean={fe.mean():.4f} std={fe.std():.4f}")

            # traced: persistent x_t, warmup -> reset -> capture -> reset -> execute
            xt_t = repl(noise, dtype=ttnn.float32, mc=ttnn.L1_MEMORY_CONFIG)
            euler_loop(xt_t)  # warmup (JIT)
            noise_host = ttnn.from_torch(
                noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(denoise)
            )
            ttnn.copy_host_to_device_tensor(noise_host, xt_t)
            tid_f = ttnn.begin_trace_capture(denoise, cq_id=0)
            euler_loop(xt_t)
            ttnn.end_trace_capture(denoise, tid_f, cq_id=0)
            ttnn.copy_host_to_device_tensor(noise_host, xt_t)
            ttnn.execute_trace(denoise, tid_f, cq_id=0, blocking=True)
            ft = ttnn.to_torch(xt_t, mesh_composer=ttnn.ConcatMeshToTensor(denoise, dim=0))[0][:ah, :]
            log(f"full-loop traced: finite={torch.isfinite(ft).all().item()} mean={ft.mean():.4f} std={ft.std():.4f}")
            log(f"PCC full-loop eager-vs-traced = {_pcc(fe, ft):.6f}")

        log("SUCCESS")
    finally:
        for sm in reversed(submeshes):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
    sys.exit(0)
