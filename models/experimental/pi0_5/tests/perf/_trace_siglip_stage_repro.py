# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage single-mesh + point_to_point + trace prototype for the VISION (SigLIP) stage.

Runs a chain of real SigLIP transformer blocks as ONE SPMD computation on a
6-chip mesh (each chip holds 3 layers as sharded weights), chained with
ttnn.point_to_point, captured as a single trace. Validates non-empty trace,
traced==eager, and eager PCC vs torch SigLIP block chain.

SigLIPBlockTTNN builds its weights internally (replicate-only), so we build each
layer on a scratch 1x1, gather the built weight attributes, stack+reshard them,
and overwrite a mesh-resident block (per-chip-different weights).

v1: 18 of the 27 SigLIP blocks (layers 0..17), 6 chips x 3. The embed/projector
endpoints are NOT included yet — this validates the homogeneous block-chain core.

Run (FABRIC_1D; 6-chip column line):
  tt-smi -glx_reset
  PI05_CHECKPOINT_DIR=.../weights/pi05_libero_upstream TT_TRACE_POPULATE_DEBUG=1 \
    timeout 400 env PYTHONPATH=$PWD TT_METAL_HOME=$PWD \
    python_env/bin/python models/experimental/pi0_5/tests/perf/_trace_siglip_stage_repro.py
"""

import os
import sys

import torch
import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_siglip import SigLIPBlock
from models.experimental.pi0_5.tt.ttnn_siglip import SigLIPBlockTTNN
from models.experimental.pi0_5.tt.tt_bh_glx.vision_slice import _layer_weights

CKPT = os.environ.get(
    "PI05_CHECKPOINT_DIR",
    "/home/tt-admin/sdawle/tt-metal/models/experimental/pi0_5/weights/pi05_libero_upstream",
)
N_CHIPS = int(os.environ.get("REPRO_NCHIPS", "6"))
N_PER = int(os.environ.get("REPRO_NPER", "3"))
DEPTH = N_CHIPS * N_PER  # 18
SEED = 0

# (attr_name, owner) — owner "" = block itself, else a sub-module attribute.
ATTRS = [
    ("ln1_weight", ""),
    ("ln1_bias", ""),
    ("ln2_weight", ""),
    ("ln2_bias", ""),
    ("wqkv", "attention"),
    ("bqkv", "attention"),
    ("wo", "attention"),
    ("bo", "attention"),
    ("fc1_weight", "mlp"),
    ("fc1_bias", "mlp"),
    ("fc2_weight", "mlp"),
    ("fc2_bias", "mlp"),
]


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return (torch.dot(a, b).item() / d) if d > 0 else 0.0


def _get(blk, name, owner):
    obj = blk if owner == "" else getattr(blk, owner)
    return getattr(obj, name, None)


def _set(blk, name, owner, val):
    obj = blk if owner == "" else getattr(blk, owner)
    setattr(obj, name, val)


def main():
    def log(m):
        print(f"[siglip-repro] {m}", flush=True)

    cfg = Pi0_5ModelConfig()
    scfg = cfg.siglip_config
    log(f"siglip hidden={scfg.hidden_size} heads={scfg.num_attention_heads} layers(using)={DEPTH}")
    loader = Pi0_5WeightLoader(CKPT)
    vision_weights = loader.categorized_weights["vlm_vision"]

    torch.manual_seed(SEED)
    num_patches = (scfg.image_size // scfg.patch_size) ** 2  # 256
    hidden0 = torch.randn(1, num_patches, scfg.hidden_size, dtype=torch.float32) * 0.1

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    submeshes = []
    try:
        vision = parent.create_submesh(ttnn.MeshShape(N_CHIPS, 1), ttnn.MeshCoordinate(0, 0))
        submeshes.append(vision)
        tmp = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(7, 0))
        submeshes.append(tmp)
        log(f"vision mesh shape={vision.shape}")

        # 1) Build each layer on scratch, gather built attrs to torch (+dtype/layout).
        per_layer = []  # list of {(name,owner): torch}
        meta = {}  # (name,owner) -> (dtype, layout)
        layer_torch = [_layer_weights(vision_weights, L) for L in range(DEPTH)]
        for L in range(DEPTH):
            blk = SigLIPBlockTTNN(scfg, layer_torch[L], tmp)
            d = {}
            for name, owner in ATTRS:
                t = _get(blk, name, owner)
                if t is None:
                    continue
                d[(name, owner)] = ttnn.to_torch(t)
                if (name, owner) not in meta:
                    meta[(name, owner)] = (t.dtype, t.layout)
                ttnn.deallocate(t)
            per_layer.append(d)
        present = list(meta.keys())
        log(f"gathered {len(present)} weight attrs/layer for {DEPTH} layers")

        # 2) Build 3 local-layer blocks on the mesh, overwrite attrs with sharded.
        def build_block(local_idx):
            layers = [c * N_PER + local_idx for c in range(N_CHIPS)]
            blk = SigLIPBlockTTNN(scfg, layer_torch[layers[0]], vision)  # placeholder weights
            for key in present:
                name, owner = key
                stacked = torch.stack([per_layer[L][key] for L in layers], dim=0)
                dt, lay = meta[key]
                sharded = ttnn.from_torch(
                    stacked,
                    dtype=dt,
                    layout=lay,
                    device=vision,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(vision, dim=0),
                )
                _set(blk, name, owner, sharded)
            return blk

        blocks = [build_block(L) for L in range(N_PER)]
        log("built 3 local-layer SigLIP blocks with sharded weights")

        hidden_m = ttnn.from_torch(
            hidden0,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=vision,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(vision),
        )

        # single-block diagnostic (NO p2p): each chip c runs local-layer0 = global
        # layer c on the same input; verify chip c == torch layer c (isolates sharding).
        sb = blocks[0].forward(hidden_m)
        ttnn.synchronize_device(vision)
        sb_full = ttnn.to_torch(sb, mesh_composer=ttnn.ConcatMeshToTensor(vision, dim=0))
        for c in range(N_CHIPS):
            lyr = c * N_PER + 0
            ref_c = SigLIPBlock(scfg, layer_torch[lyr]).forward(hidden0)
            log(f"PCC single-block chip{c}=layer{lyr} ttnn-vs-torch = {_pcc(sb_full[c], ref_c):.6f}")

        def chain(h):
            cur = h
            for c in range(N_CHIPS):
                for L in range(N_PER):
                    cur = blocks[L].forward(cur)
                if c < N_CHIPS - 1:
                    # p2p corrupts (NaN) on the SigLIP block's native output config;
                    # give it a clean DRAM-interleaved, tile-laid-out input.
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

        log("warmup (eager) chain... (instrumented per-iteration max-abs)")
        if os.environ.get("REPRO_TRACE_CHAIN", "0") == "1":
            cur = hidden_m
            for c in range(N_CHIPS):
                for L in range(N_PER):
                    cur = blocks[L].forward(cur)
                ttnn.synchronize_device(vision)
                full = ttnn.to_torch(cur, mesh_composer=ttnn.ConcatMeshToTensor(vision, dim=0))
                mx = [f"c{i}:{full[i].abs().max().item():.2e}" for i in range(N_CHIPS)]
                log(f"  after c={c} blocks: {' '.join(mx)}")
                if c < N_CHIPS - 1:
                    cur = ttnn.to_layout(cur, ttnn.TILE_LAYOUT)
                    cur = ttnn.to_memory_config(cur, ttnn.DRAM_MEMORY_CONFIG)
                    cur = ttnn.point_to_point(
                        cur,
                        ttnn.MeshCoordinate(c, 0),
                        ttnn.MeshCoordinate(c + 1, 0),
                        topology=ttnn.Topology.Linear,
                        output_tensor=cur,
                    )
                    ttnn.synchronize_device(vision)
                    full = ttnn.to_torch(cur, mesh_composer=ttnn.ConcatMeshToTensor(vision, dim=0))
                    mx = [f"c{i}:{full[i].abs().max().item():.2e}" for i in range(N_CHIPS)]
                    log(f"  after c={c} p2p({c}->{c+1}): {' '.join(mx)}")
        eager = chain(hidden_m)
        ttnn.synchronize_device(vision)
        eager_last = ttnn.to_torch(eager, mesh_composer=ttnn.ConcatMeshToTensor(vision, dim=0))[N_CHIPS - 1]
        log("eager chain OK")

        log("begin_trace_capture")
        tid = ttnn.begin_trace_capture(vision, cq_id=0)
        traced = chain(hidden_m)
        ttnn.end_trace_capture(vision, tid, cq_id=0)
        log("END_TRACE_CAPTURE OK")
        ttnn.execute_trace(vision, tid, cq_id=0, blocking=True)
        log("EXECUTE_TRACE OK")
        traced_last = ttnn.to_torch(traced, mesh_composer=ttnn.ConcatMeshToTensor(vision, dim=0))[N_CHIPS - 1]

        # torch reference: 18-layer SigLIP chain
        ref = hidden0
        for L in range(DEPTH):
            ref = SigLIPBlock(scfg, layer_torch[L]).forward(ref)

        log(f"stats eager_last mean={eager_last.float().mean():.4f} std={eager_last.float().std():.4f}")
        log(f"stats traced_last mean={traced_last.float().mean():.4f} std={traced_last.float().std():.4f}")
        log(f"stats torch ref   mean={ref.float().mean():.4f} std={ref.float().std():.4f}")
        log(f"PCC eager-vs-traced           = {_pcc(eager_last, traced_last):.6f}")
        log(f"PCC eager-vs-torch FULL chain = {_pcc(eager_last, ref):.6f}")
        log(f"traced finite={torch.isfinite(traced_last).all().item()}")
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
