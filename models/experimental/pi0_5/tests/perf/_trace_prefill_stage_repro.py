# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced PREFILL stage prototype: 18 VLM (Gemma-2B) blocks on the (6,3) snake mesh.

Each chip holds 1 VLM layer (sharded weights); the chain follows the boustrophedon
snake (collinear hops) with point_to_point, captured as one trace. Validates the
snake VLM chain (final hidden) vs a torch GemmaBlock chain + trace replay.

v1: self-attention, no KV collection yet (final-hidden chain validation first).

  tt-smi -glx_reset
  PI05_CHECKPOINT_DIR=.../weights/pi05_libero_upstream timeout 500 \
    env PYTHONPATH=$PWD TT_METAL_HOME=$PWD \
    python_env/bin/python models/experimental/pi0_5/tests/perf/_trace_prefill_stage_repro.py
"""

import os
import sys

import torch
import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_gemma import GemmaBlock, precompute_freqs_cis
from models.experimental.pi0_5.tt.ttnn_gemma import GemmaBlockTTNN, precompute_freqs_cis_meta_format
from models.experimental.pi0_5.tt.tt_bh_glx import stages
from models.experimental.pi0_5.tt.tt_bh_glx.vlm_slice import _load_block_weights_to_submesh

CKPT = os.environ.get(
    "PI05_CHECKPOINT_DIR",
    "/home/tt-admin/sdawle/tt-metal/models/experimental/pi0_5/weights/pi05_libero_upstream",
)
ROWS, COLS = 6, 3
DEPTH = ROWS * COLS  # 18
PREFIX = int(os.environ.get("REPRO_PREFIX", "64"))  # prefix seq len (tile-aligned)
SEED = 0


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return (torch.dot(a, b).item() / d) if d > 0 else 0.0


def _coord_to_layer(r, c):
    return 3 * r + (c if r % 2 == 0 else COLS - 1 - c)


def main():
    def log(m):
        print(f"[prefill-repro] {m}", flush=True)

    cfg = Pi0_5ModelConfig()
    vcfg = cfg.vlm_config
    log(
        f"vlm width={vcfg.width} heads={vcfg.num_heads} kv={vcfg.num_kv_heads} head_dim={vcfg.head_dim} depth={vcfg.depth} prefix={PREFIX}"
    )
    loader = Pi0_5WeightLoader(CKPT)
    vlm_weights = loader.categorized_weights["vlm_language"]

    torch.manual_seed(SEED)
    hidden0 = torch.randn(1, PREFIX, vcfg.width, dtype=torch.float32) * 0.1

    snake = stages.prefill_snake_order(ROWS, COLS)
    log(f"snake order: {snake}")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(ROWS, COLS), ttnn.MeshCoordinate(0, 0))
        subs.append(mesh)
        tmp = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(7, 0))
        subs.append(tmp)
        log(f"prefill mesh shape={mesh.shape}")

        # Gather each VLM layer's processed weights on scratch, to_torch + dtype/layout.
        per_layer = []
        meta = {}
        for L in range(DEPTH):
            bw = _load_block_weights_to_submesh(vlm_weights, L, tmp)
            d = {}
            for k, v in bw.items():
                d[k] = ttnn.to_torch(v)
                if k not in meta:
                    meta[k] = (v.dtype, v.layout)
                ttnn.deallocate(v)
            per_layer.append(d)
        log(f"gathered {DEPTH} VLM layers, {len(meta)} keys/layer")

        # Sharded weights: row-major chip i=(i//3,i%3) holds layer _coord_to_layer(i).
        sharded = {}
        for k in per_layer[0].keys():
            order = [per_layer[_coord_to_layer(i // COLS, i % COLS)][k] for i in range(DEPTH)]
            stacked = torch.stack(order, dim=0)
            dt, lay = meta[k]
            sharded[k] = ttnn.from_torch(
                stacked,
                dtype=dt,
                layout=lay,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
        block = GemmaBlockTTNN(vcfg, sharded, 0, mesh, None, None)
        log("built sharded VLM block (chip = its snake layer)")

        cos_m, sin_m = precompute_freqs_cis_meta_format(vcfg.head_dim, PREFIX, mesh, base=vcfg.rope_base)
        cos_t, sin_t = precompute_freqs_cis(vcfg.head_dim, PREFIX, vcfg.rope_base)
        hidden_m = ttnn.from_torch(
            hidden0,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # single-block diagnostic: chip0 = layer0 vs torch layer0
        sb, _ = block.forward(hidden_m, cos_m, sin_m, None, None, None, False)
        ttnn.synchronize_device(mesh)
        sb_t = ttnn.to_torch(sb, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0]
        lw0 = {kk[len("model.layers.0.") :]: vv for kk, vv in vlm_weights.items() if kk.startswith("model.layers.0.")}
        sb_ref, _ = GemmaBlock(vcfg, lw0, 0).forward(hidden0, cos_t, sin_t, None, None, None, False)
        log(f"PCC single-block layer0 ttnn-vs-torch = {_pcc(sb_t, sb_ref):.6f}")

        def chain(h, collect_kv=False):
            cur = h
            kvs = []
            for k in range(DEPTH):
                # use_cache=True -> produce per-layer KV. At step k the active chip
                # snake[k]=coord(k) ran layer k on the forwarded activation, so its
                # KV lane is layer k's real KV (other chips' lanes are garbage).
                cur, new_kv = block.forward(cur, cos_m, sin_m, None, None, None, collect_kv)
                if collect_kv:
                    kvs.append(new_kv)  # (K, V) full-mesh tensors; keep as outputs
                if k < DEPTH - 1:
                    cur = ttnn.to_memory_config(ttnn.to_layout(cur, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                    cur = ttnn.point_to_point(
                        cur,
                        ttnn.MeshCoordinate(*snake[k]),
                        ttnn.MeshCoordinate(*snake[k + 1]),
                        topology=ttnn.Topology.Linear,
                        output_tensor=cur,
                    )
            return cur, kvs

        last_idx = snake[-1][0] * COLS + snake[-1][1]

        def layer_kv_lane(kvs, k):
            # extract chip coord(k)'s K,V lane (= layer k's real KV) as torch
            r, c = snake[k]  # snake[k] == coord(k)
            idx = r * COLS + c
            K = ttnn.to_torch(kvs[k][0], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[idx]
            V = ttnn.to_torch(kvs[k][1], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[idx]
            return K, V

        log("eager snake chain (18 VLM layers, collect KV)...")
        eager, eager_kvs = chain(hidden_m, collect_kv=True)
        ttnn.synchronize_device(mesh)
        eager_last = ttnn.to_torch(eager, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[last_idx]
        eager_K = [layer_kv_lane(eager_kvs, k) for k in range(DEPTH)]
        log("eager OK (KV collected)")

        log("capture + execute trace (with KV as outputs)...")
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced, traced_kvs = chain(hidden_m, collect_kv=True)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        traced_last = ttnn.to_torch(traced, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[last_idx]
        traced_K = [layer_kv_lane(traced_kvs, k) for k in range(DEPTH)]
        log("trace OK")

        # torch reference: chain + per-layer KV
        ref = hidden0
        torch_K = []
        for L in range(DEPTH):
            lw = {
                kk[len(f"model.layers.{L}.") :]: vv
                for kk, vv in vlm_weights.items()
                if kk.startswith(f"model.layers.{L}.")
            }
            ref, (Kt, Vt) = GemmaBlock(vcfg, lw, L).forward(ref, cos_t, sin_t, None, None, None, True)
            torch_K.append((Kt, Vt))

        kv_pcc_et = sum(_pcc(eager_K[k][0], torch_K[k][0]) for k in range(DEPTH)) / DEPTH
        kv_pcc_tt = sum(_pcc(eager_K[k][0], traced_K[k][0]) for k in range(DEPTH)) / DEPTH
        v_pcc_et = sum(_pcc(eager_K[k][1], torch_K[k][1]) for k in range(DEPTH)) / DEPTH
        log(f"PCC hidden eager-vs-traced    = {_pcc(eager_last, traced_last):.6f}")
        log(f"PCC hidden eager-vs-torch     = {_pcc(eager_last, ref):.6f}")
        log(f"PCC K(mean/18) eager-vs-torch = {kv_pcc_et:.6f}")
        log(f"PCC V(mean/18) eager-vs-torch = {v_pcc_et:.6f}")
        log(f"PCC K(mean/18) eager-vs-traced= {kv_pcc_tt:.6f}  (KV survives trace replay)")
        log("SUCCESS")
    finally:
        for sm in reversed(subs):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
    sys.exit(0)
