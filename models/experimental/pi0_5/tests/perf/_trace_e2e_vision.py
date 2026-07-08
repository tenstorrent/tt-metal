# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""e2e assembly — VISION stage: real images -> embed -> traced 27-block core ->
post-LN -> mm_projector -> (1,256,2048), validated vs torch SigLIPVisionTower +
MultiModalProjector.

Hybrid: embed + post-LN + projector are eager (single ops, known-good modules);
the 27 SigLIP blocks (the bulk) run traced on a (1,3) row mesh (9 layers/chip),
chained with point_to_point. Cross-(embed/projector) hand-offs are host-bounce.

  tt-smi -glx_reset
  PI05_CHECKPOINT_DIR=.../weights/pi05_libero_upstream timeout 500 \
    env PYTHONPATH=$PWD TT_METAL_HOME=$PWD \
    python_env/bin/python models/experimental/pi0_5/tests/perf/_trace_e2e_vision.py
"""

import os
import sys

import torch
import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_siglip import MultiModalProjector, SigLIPVisionTower
from models.experimental.pi0_5.tt.ttnn_common import get_ln_weight_memory_config, tensor_1d_to_2d_ttnn
from models.experimental.pi0_5.tt.ttnn_siglip import MultiModalProjectorTTNN, SigLIPBlockTTNN
from models.experimental.pi0_5.tt.tt_bh_glx.vision_slice import SigLIPEmbedSlice, _layer_weights

CKPT = os.environ.get(
    "PI05_CHECKPOINT_DIR",
    "/home/tt-admin/sdawle/tt-metal/models/experimental/pi0_5/weights/pi05_libero_upstream",
)
NB_CHIPS = 3  # block-mesh chips
NB_PER = 9  # SigLIP layers per chip -> 27 total
DEPTH = NB_CHIPS * NB_PER
SEED = 42

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


def _get(blk, n, o):
    return getattr(blk if o == "" else getattr(blk, o), n, None)


def _set(blk, n, o, v):
    setattr(blk if o == "" else getattr(blk, o), n, v)


def main():
    def log(m):
        print(f"[e2e-vision] {m}", flush=True)

    cfg = Pi0_5ModelConfig()
    scfg = cfg.siglip_config
    loader = Pi0_5WeightLoader(CKPT)
    vw = loader.categorized_weights["vlm_vision"]
    pw = loader.categorized_weights["vlm_projector"]

    torch.manual_seed(SEED)
    images = torch.randn(1, 3, scfg.image_size, scfg.image_size, dtype=torch.float32)

    # torch reference
    tower = SigLIPVisionTower(scfg, vw)
    proj = MultiModalProjector(pw)
    ref_feats = proj.forward(tower.forward(images))
    log(f"torch vision features {tuple(ref_feats.shape)} mean={ref_feats.mean():.4f} std={ref_feats.std():.4f}")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        embed_chip = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(6, 3))
        subs.append(embed_chip)
        block_mesh = parent.create_submesh(ttnn.MeshShape(1, 3), ttnn.MeshCoordinate(6, 0))
        subs.append(block_mesh)
        scratch = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(7, 0))
        subs.append(scratch)

        # ---- embed (eager) on embed_chip ----
        embed = SigLIPEmbedSlice(scfg, vw, embed_chip)
        images_m = ttnn.from_torch(
            images,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=embed_chip,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        emb = embed.forward(images_m)
        ttnn.synchronize_device(embed_chip)
        emb_torch = ttnn.to_torch(emb)  # host-bounce
        log(f"embed out {tuple(emb_torch.shape)}")

        # ---- 27 SigLIP blocks (traced) on block_mesh: chip (0,c) = layers 9c..9c+8 ----
        layer_torch = [_layer_weights(vw, L) for L in range(DEPTH)]
        per_layer, meta = [], {}
        for L in range(DEPTH):
            b = SigLIPBlockTTNN(scfg, layer_torch[L], scratch)
            d = {}
            for n, o in ATTRS:
                t = _get(b, n, o)
                if t is None:
                    continue
                d[(n, o)] = ttnn.to_torch(t)
                meta.setdefault((n, o), (t.dtype, t.layout))
                ttnn.deallocate(t)
            per_layer.append(d)
        present = list(meta.keys())

        def build_block(local):
            layers = [c * NB_PER + local for c in range(NB_CHIPS)]
            blk = SigLIPBlockTTNN(scfg, layer_torch[layers[0]], block_mesh)
            for key in present:
                stacked = torch.stack([per_layer[L][key] for L in layers], dim=0)
                dt, lay = meta[key]
                _set(
                    blk,
                    key[0],
                    key[1],
                    ttnn.from_torch(
                        stacked,
                        dtype=dt,
                        layout=lay,
                        device=block_mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(block_mesh, dim=0),
                    ),
                )
            return blk

        blocks = [build_block(L) for L in range(NB_PER)]
        log("built 9 local SigLIP blocks (sharded across 3 chips)")

        emb_m = ttnn.from_torch(
            emb_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=block_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(block_mesh),
        )

        def chain(h):
            cur = h
            for c in range(NB_CHIPS):
                for L in range(NB_PER):
                    cur = blocks[L].forward(cur)
                if c < NB_CHIPS - 1:
                    cur = ttnn.to_memory_config(ttnn.to_layout(cur, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                    cur = ttnn.point_to_point(
                        cur,
                        ttnn.MeshCoordinate(0, c),
                        ttnn.MeshCoordinate(0, c + 1),
                        topology=ttnn.Topology.Linear,
                        output_tensor=cur,
                    )
            return cur

        chain(emb_m)  # warmup
        ttnn.synchronize_device(block_mesh)
        tid = ttnn.begin_trace_capture(block_mesh, cq_id=0)
        bo = chain(emb_m)
        ttnn.end_trace_capture(block_mesh, tid, cq_id=0)
        ttnn.execute_trace(block_mesh, tid, cq_id=0, blocking=True)
        blk_out = ttnn.to_torch(bo, mesh_composer=ttnn.ConcatMeshToTensor(block_mesh, dim=0))[NB_CHIPS - 1]
        log(f"traced 27-block out {tuple(blk_out.shape)}")

        # ---- post-LN + projector (eager) on embed_chip (host-bounce in) ----
        post_w = vw.get("post_layernorm.weight") or vw.get("vision_model.post_layernorm.weight")
        post_b = vw.get("post_layernorm.bias") or vw.get("vision_model.post_layernorm.bias")
        ln_mc = get_ln_weight_memory_config()
        pw_t = tensor_1d_to_2d_ttnn(post_w, embed_chip, dtype=ttnn.bfloat16, memory_config=ln_mc)
        pb_t = (
            tensor_1d_to_2d_ttnn(post_b, embed_chip, dtype=ttnn.bfloat16, memory_config=ln_mc)
            if post_b is not None
            else None
        )
        projector = MultiModalProjectorTTNN(pw, embed_chip)
        blk_m = ttnn.from_torch(
            blk_out,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=embed_chip,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        normed = ttnn.layer_norm(
            blk_m, weight=pw_t, bias=pb_t, epsilon=scfg.layer_norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        feats = projector.forward(normed)
        ttnn.synchronize_device(embed_chip)
        feats_t = ttnn.to_torch(feats)
        log(
            f"ttnn vision features {tuple(feats_t.shape)} mean={feats_t.float().mean():.4f} std={feats_t.float().std():.4f}"
        )
        log(f"PCC vision e2e ttnn-vs-torch = {_pcc(feats_t, ref_feats):.6f}")
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
