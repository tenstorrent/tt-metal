# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Profile a single Molmo2 sub-block with Tracy.

Called by run_block_profiles.sh for each block in turn so each block
gets its own ops_perf_results CSV. Use -n/--block-name to label the run.

Run with:
  python -m tracy -p -v -r -n text_attention \
      models/demos/molmo2/tests/profile_single_block.py --block text_attention

Blocks: text_attention  text_mlp  decoder_block  vit_encoder  image_pooling  image_projector
"""

import argparse
import pathlib

import torch
from transformers import AutoModelForImageTextToText

import ttnn
from models.demos.molmo2.tt.model_config import Molmo2Config
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, get_rot_transformation_mat, precompute_freqs

HF_PATH = (
    "/home/ttuser/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
)
WEIGHT_CACHE = pathlib.Path("/tmp/molmo2_weight_cache")

BLOCKS = ["text_attention", "text_mlp", "decoder_block", "vit_encoder", "image_pooling", "image_projector"]

parser = argparse.ArgumentParser()
parser.add_argument("--block", required=True, choices=BLOCKS)
parser.add_argument("--seq-len", type=int, default=128)
parser.add_argument("--n-crops", type=int, default=1)
parser.add_argument("--n-warmup", type=int, default=1)  # 1 warmup ensures JIT kernels are compiled before Tracy capture
args = parser.parse_args()

SEQ = args.seq_len

# ── Device setup ───────────────────────────────────────────────────────────
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
cfg = Molmo2Config(mesh_device=mesh)
cfg.max_batch_size = 1
cfg.max_seq_len = 4096

print(f"Loading weights for block={args.block}...")
hf = AutoModelForImageTextToText.from_pretrained(
    HF_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
)
sd = hf.state_dict()
del hf
ccl = TT_CCL(mesh)


def _tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


# Shared helpers
transformation_mats = {
    "prefill": _tt(get_rot_transformation_mat(dhead=cfg.head_dim)),
    "decode": _tt(get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)),
}
cos_raw, sin_raw = precompute_freqs(cfg.head_dim, cfg.max_seq_len * 2, cfg.rope_theta, None, None)
cos_hf = torch.cat([cos_raw[: cfg.max_seq_len], cos_raw[: cfg.max_seq_len]], dim=-1)
sin_hf = torch.cat([sin_raw[: cfg.max_seq_len], sin_raw[: cfg.max_seq_len]], dim=-1)
rot_mats = [
    _tt(cos_hf.unsqueeze(0).unsqueeze(0).bfloat16()),
    _tt(sin_hf.unsqueeze(0).unsqueeze(0).bfloat16()),
]

# Flush weight-loading ops from profiler DRAM buffers
ttnn.ReadDeviceProfiler(mesh)
print(f"Ready — profiling block: {args.block}\n")

# ── Block instantiation & forward functions ────────────────────────────────
if args.block == "text_attention":
    from models.demos.molmo2.tt.attention import TtMolmo2TextAttention

    blk = TtMolmo2TextAttention(
        mesh_device=mesh,
        tt_ccl=ccl,
        state_dict=sd,
        weight_cache_path=WEIGHT_CACHE,
        layer_num=0,
        dtype=ttnn.bfloat16,
        configuration=cfg,
        transformation_mats=transformation_mats,
    )

    def fwd():
        x = _tt(torch.randn(1, 1, SEQ, cfg.dim).bfloat16())
        out = blk.forward_prefill(x, rot_mats=rot_mats, user_id=0, mask=None)
        ttnn.deallocate(x)
        return out

elif args.block == "text_mlp":
    from models.demos.molmo2.tt.mlp import TtMolmo2TextMLP

    blk = TtMolmo2TextMLP(
        mesh_device=mesh,
        tt_ccl=ccl,
        state_dict=sd,
        weight_cache_path=WEIGHT_CACHE,
        layer_num=0,
        dtype=ttnn.bfloat16,
        configuration=cfg,
    )

    def fwd():
        x = _tt(torch.randn(1, 1, SEQ, cfg.dim).bfloat16())
        out = blk.forward(x, mode=Mode.PREFILL)
        ttnn.deallocate(x)
        return out

elif args.block == "decoder_block":
    from models.demos.molmo2.tt.model import TtMolmo2DecoderBlock

    blk = TtMolmo2DecoderBlock(
        mesh_device=mesh,
        tt_ccl=ccl,
        state_dict=sd,
        weight_cache_path=WEIGHT_CACHE,
        layer_num=0,
        dtype=ttnn.bfloat16,
        configuration=cfg,
        transformation_mats=transformation_mats,
    )

    def fwd():
        x = _tt(torch.randn(1, 1, SEQ, cfg.dim).bfloat16())
        out = blk.forward(x, rot_mats=rot_mats, user_id=0, mode="prefill", attn_mask=None)
        ttnn.deallocate(x)
        return out

elif args.block == "vit_encoder":
    from models.demos.molmo2.tt.vision_encoder import TtMolmo2ViTEncoder

    blk = TtMolmo2ViTEncoder(
        mesh_device=mesh,
        state_dict=sd,
        vit_cfg=cfg,
        weight_cache_path=WEIGHT_CACHE,
    )
    pv = ttnn.from_torch(
        torch.randn(args.n_crops, 1, 729, 588).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def fwd():
        return blk.forward(pv, patch_num=(27, 27))

elif args.block == "image_pooling":
    # Profile the full _run_chunked_ttnn_pooling path with realistic video shapes.
    # Uses n_crops=30, N_pooled=2430, k_pool=9 (default 30-frame video).
    # Chunk size = _POOL_CHUNK_WINDOWS=4096 → single chunk padded to 4096 windows.
    from models.demos.molmo2.tt.model import TtMolmo2Model
    from models.tt_transformers.tt.ccl import TT_CCL

    ccl = TT_CCL(mesh)
    model = TtMolmo2Model(
        mesh_device=mesh,
        tt_ccl=ccl,
        state_dict=sd,
        weight_cache_path=WEIGHT_CACHE,
        dtype=ttnn.bfloat16,
        configuration=cfg,
    )
    n_crops = args.n_crops if args.n_crops > 1 else 30
    k_pool = 9
    n_out_per_frame = 81
    N_pooled = n_crops * n_out_per_frame
    # Dummy ViT features and pooling indices (zeros are clamped/masked correctly)
    vit_cpu = torch.randn(1, n_crops, 729, cfg.vit_hidden * 2)
    pool_idx = torch.zeros(1, N_pooled, k_pool, dtype=torch.long)

    def fwd():
        return model._run_chunked_ttnn_pooling(vit_cpu, pool_idx)

elif args.block == "image_projector":
    from models.demos.molmo2.tt.image_projector import TtMolmo2ImageProjector

    blk = TtMolmo2ImageProjector(
        mesh_device=mesh,
        state_dict=sd,
        cfg=cfg,
        weight_cache_path=WEIGHT_CACHE,
    )
    x_proj = _tt(torch.randn(1, 81, cfg.vit_hidden).bfloat16())

    def fwd():
        return blk.forward(x_proj)


# ── Warmup + profiled pass ─────────────────────────────────────────────────
print(f"  Warmup ({args.n_warmup}x)...")
for _ in range(args.n_warmup):
    out = fwd()
    if isinstance(out, ttnn.Tensor):
        ttnn.deallocate(out)

ttnn.ReadDeviceProfiler(mesh)

print("  Profiling...")
out = fwd()
ttnn.ReadDeviceProfiler(mesh)
if isinstance(out, ttnn.Tensor):
    ttnn.deallocate(out)

ttnn.close_mesh_device(mesh)
print(f"=== {args.block} profiling complete ===")
