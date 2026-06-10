# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 real-HF-weights PCC harness for dots.ocr (integration skill).

One pytest case per block, parametrized; each block helper loads real
checkpoint weights through tt/weight_loader.py, runs the pure-PyTorch
reference on a production-distribution input, runs the TTNN block at the
production operating point (fp32, 1x4 mesh), and returns ONE float PCC.
Rows are appended per real_weights tick — vision_patch_embed first.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import ttnn

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


wl = _load_module("dots_ocr_weight_loader", MODEL_DIR / "tt" / "weight_loader.py")
ref = _load_module("dots_ocr_reference_functional", MODEL_DIR / "reference" / "functional.py")
_patch_embed_mod = _load_module("dots_ocr_tt_vision_patch_embed", MODEL_DIR / "tt" / "vision_patch_embed.py")
_vision_rmsnorm_mod = _load_module("dots_ocr_tt_vision_rmsnorm", MODEL_DIR / "tt" / "vision_rmsnorm.py")
_vision_attention_mod = _load_module("dots_ocr_tt_vision_attention", MODEL_DIR / "tt" / "vision_attention.py")
_vision_mlp_mod = _load_module("dots_ocr_tt_vision_mlp", MODEL_DIR / "tt" / "vision_mlp.py")
_vision_block_mod = _load_module("dots_ocr_tt_vision_block", MODEL_DIR / "tt" / "vision_block.py")
_patch_merger_mod = _load_module("dots_ocr_tt_patch_merger", MODEL_DIR / "tt" / "patch_merger.py")
_vision_transformer_mod = _load_module("dots_ocr_tt_vision_transformer", MODEL_DIR / "tt" / "vision_transformer.py")


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _t_vision_patch_embed(mesh_device) -> tuple[float, int]:
    sd = wl.vision_patch_embed_weights()
    # Production-distribution input: the HF preprocessor emits normalized,
    # pre-flattened patches. Reuse the golden's real preprocessed image patches.
    golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_patch_embed.pt")
    x = golden["input"]  # [num_patches, C*P*P]
    ref_out = ref.vision_patch_embed_forward(x, sd)

    block = _patch_embed_mod.TtVisionPatchEmbed(mesh_device, sd, dtype=ttnn.float32)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = block.forward(x_tt)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()
    assert out.shape == ref_out.shape, f"{out.shape} != {ref_out.shape}"
    return _pcc(ref_out, out), wl.count_params(sd)


def _t_vision_rmsnorm(mesh_device) -> tuple[float, int]:
    # Real weights from three sites (blocks.0.norm1, blocks.41.norm2,
    # post_trunk_norm) exercise the loader across the per-layer index and the
    # tower-level key; PCC is gated on each, the min is reported.
    # Production-distribution input: the golden's real residual-stream
    # activation (RMSNorm is its own first op, so a real activation suffices).
    golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_rmsnorm.pt")
    x, eps = golden["input"], golden["eps"]
    sites = [(0, "norm1"), (wl.VISION_NUM_BLOCKS - 1, "norm2"), (0, "post_trunk_norm")]
    pccs, n_params = [], 0
    for layer_idx, which in sites:
        sd = wl.vision_rmsnorm_weights(layer_idx=layer_idx, which=which)
        n_params += wl.count_params(sd)
        ref_out = ref.vision_rmsnorm_forward(x, sd["weight"], eps=eps)
        # Production operating point: the vision tower runs an fp32 residual
        # stream with fp32 TILE gammas (vision_transformer ttnn phase).
        block = _vision_rmsnorm_mod.TtVisionRMSNorm(mesh_device, sd, dtype=ttnn.float32, eps=eps)
        x_tt = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out = ttnn.to_torch(ttnn.get_device_tensors(block.forward(x_tt))[0]).float()
        assert out.shape == ref_out.shape, f"{which}: {out.shape} != {ref_out.shape}"
        pcc = _pcc(ref_out, out)
        print(f"  vision_rmsnorm[{which}@{layer_idx}] PCC = {pcc:.6f}")
        pccs.append(pcc)
    return min(pccs), n_params


def _t_vision_attention(mesh_device) -> tuple[float, int]:
    # Real fused-QKV / proj weights from two sites (blocks.0, blocks.41)
    # exercise the per-layer loader index; PCC gated on each, min reported.
    # Production-distribution input: the golden's real post-norm1 activation,
    # with the golden's real rope tables and cu_seqlens (block-diagonal mask).
    golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_attention.pt")
    x, rope, cu_seqlens = golden["input"], golden["rotary_pos_emb"], golden["cu_seqlens"]
    seq, dim = x.shape
    padded_seq = ((seq + 127) // 128) * 128
    x_pad = torch.cat([x, torch.zeros(padded_seq - seq, dim)], dim=0)
    pccs, n_params = [], 0
    for layer_idx in (0, wl.VISION_NUM_BLOCKS - 1):
        sd = wl.vision_attention_weights(layer_idx=layer_idx)
        n_params += wl.count_params(sd)
        ref_out = ref.vision_attention_forward(x, sd, cu_seqlens, rope)
        # Production operating point: fp32 high-precision path (explicit fp32
        # HF rope; bf16 only at the bf16-only windowed-SDPA kernel boundary),
        # as the 42-layer tower runs it.
        block = _vision_attention_mod.TtVisionAttention(mesh_device, sd, num_heads=12, dtype=ttnn.float32)
        x_tt = ttnn.from_torch(
            x_pad.reshape(1, 1, padded_seq, dim),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rot_mats = block.prepare_rope(rope, padded_seq)
        cu_tt = block.prepare_cu_seqlens(cu_seqlens)
        out_tt = block.forward(x_tt, rot_mats, cu_tt)
        out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float().reshape(padded_seq, dim)[:seq]
        assert out.shape == ref_out.shape, f"blocks.{layer_idx}: {out.shape} != {ref_out.shape}"
        pcc = _pcc(ref_out, out)
        print(f"  vision_attention[blocks.{layer_idx}] PCC = {pcc:.6f}")
        pccs.append(pcc)
    return min(pccs), n_params


def _t_vision_mlp(mesh_device) -> tuple[float, int]:
    # Real SwiGLU fc1/fc2/fc3 weights from two sites (blocks.0, blocks.41)
    # exercise the per-layer loader index; PCC gated on each, min reported.
    # Production-distribution input: the golden's real post-norm2 activation
    # (the MLP consumes the RMSNorm output directly, so this is exactly what
    # the block sees in production).
    golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_mlp.pt")
    x = golden["input"]  # [784, 1536]
    pccs, n_params = [], 0
    for layer_idx in (0, wl.VISION_NUM_BLOCKS - 1):
        sd = wl.vision_mlp_weights(layer_idx=layer_idx)
        n_params += wl.count_params(sd)
        ref_out = ref.vision_mlp_forward(x, sd)
        # Production operating point: the vision tower runs an fp32 residual
        # stream with fp32 weights/activations (vision_transformer ttnn phase).
        block = _vision_mlp_mod.TtVisionMLP(mesh_device, sd, dtype=ttnn.float32)
        x_tt = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out = ttnn.to_torch(ttnn.get_device_tensors(block.forward(x_tt))[0]).float()
        assert out.shape == ref_out.shape, f"blocks.{layer_idx}: {out.shape} != {ref_out.shape}"
        pcc = _pcc(ref_out, out)
        print(f"  vision_mlp[blocks.{layer_idx}] PCC = {pcc:.6f}")
        pccs.append(pcc)
    return min(pccs), n_params


def _t_vision_block(mesh_device) -> tuple[float, int]:
    # Real full-block weights (norm1/attn/norm2/mlp) from two sites (blocks.0,
    # blocks.41) exercise the composed per-layer loader; PCC gated on each,
    # min reported. Production-distribution input: the golden's real residual
    # stream entering blocks.0, with the golden's real rope tables and
    # cu_seqlens (block-diagonal mask).
    golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_block.pt")
    x, rope, cu_seqlens = golden["input"], golden["rotary_pos_emb"], golden["cu_seqlens"]
    seq, dim = x.shape
    padded_seq = ((seq + 127) // 128) * 128
    x_pad = torch.cat([x, torch.zeros(padded_seq - seq, dim)], dim=0)
    pccs, n_params = [], 0
    for layer_idx in (0, wl.VISION_NUM_BLOCKS - 1):
        sd = wl.vision_block_weights(layer_idx=layer_idx)
        n_params += wl.count_params(sd)
        ref_out = ref.vision_block_forward(x, sd, cu_seqlens, rope)
        # Production operating point: the 42-layer tower runs an fp32 residual
        # stream with fp32 weights and the high-precision attention path
        # (vision_transformer ttnn phase).
        block = _vision_block_mod.TtVisionBlock(mesh_device, sd, num_heads=12, dtype=ttnn.float32)
        x_tt = ttnn.from_torch(
            x_pad.reshape(1, 1, padded_seq, dim),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rot_mats = block.prepare_rope(rope, padded_seq)
        cu_tt = block.prepare_cu_seqlens(cu_seqlens)
        out_tt = block.forward(x_tt, rot_mats, cu_tt)
        out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float().reshape(padded_seq, dim)[:seq]
        assert out.shape == ref_out.shape, f"blocks.{layer_idx}: {out.shape} != {ref_out.shape}"
        pcc = _pcc(ref_out, out)
        print(f"  vision_block[blocks.{layer_idx}] PCC = {pcc:.6f}")
        pccs.append(pcc)
    return min(pccs), n_params


def _t_patch_merger(mesh_device) -> tuple[float, int]:
    # Real vision_tower.merger weights (LayerNorm-with-bias + two biased
    # Linears) through the consolidated loader. Production-distribution input:
    # the golden's real post-trunk-norm activation [784, 1536] (the merger
    # consumes the tower-level RMSNorm output directly). Production operating
    # point: fp32, exactly as TtVisionTransformer instantiates the merger.
    golden = torch.load(MODEL_DIR / "reference" / "golden" / "patch_merger.pt")
    x = golden["input"]  # [784, 1536]
    sd = wl.patch_merger_weights()
    ref_out = ref.patch_merger_forward(x, sd)

    block = _patch_merger_mod.TtPatchMerger(mesh_device, sd, dtype=ttnn.float32)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = ttnn.to_torch(ttnn.get_device_tensors(block.forward(x_tt))[0]).float()
    assert out.shape == ref_out.shape, f"{out.shape} != {ref_out.shape}"
    return _pcc(ref_out, out), wl.count_params(sd)


def _t_vision_transformer(mesh_device) -> tuple[float, int]:
    # Full 42-layer tower (patch_embed -> blocks -> post_trunk_norm -> merger)
    # through the composed vision_transformer_weights loader — the sub-model
    # gate runs at the PRODUCTION layer count since the fp32 tower already
    # cleared 0.99 in its ttnn-phase test (thin-margin block; a reduced
    # config would not exercise the compounding it is gated on).
    # Production-distribution input: the golden's real preprocessed image
    # patches + grid_thw; reference recomputed with the loader's weights so
    # the loader mapping itself is what is validated. Production operating
    # point: fp32 residual stream + fp32 weights, 1x4 mesh replicated.
    golden = torch.load(MODEL_DIR / "reference" / "golden" / "vision_transformer.pt")
    x, grid_thw = golden["input"], golden["grid_thw"]
    seq, patch_dim = x.shape
    num_heads, spatial_merge_size = 12, 2
    sd = wl.vision_transformer_weights()
    ref_out = ref.vision_transformer_forward(x, sd, grid_thw, num_layers=wl.VISION_NUM_BLOCKS)

    model = _vision_transformer_mod.TtVisionTransformer(
        mesh_device, sd, num_layers=wl.VISION_NUM_BLOCKS, num_heads=num_heads, dtype=ttnn.float32
    )
    # Host-side rope tables + UNPADDED window boundaries (hybrid_notes).
    head_dim = sd["blocks.0.attn.qkv.weight"].shape[-1] // num_heads
    rope = ref.vision_rot_pos_emb(grid_thw, head_dim=head_dim, spatial_merge_size=spatial_merge_size)
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    padded_seq = ((seq + 127) // 128) * 128
    x_pad = torch.cat([x, torch.zeros(padded_seq - seq, patch_dim)], dim=0)
    x_tt = ttnn.from_torch(
        x_pad.reshape(1, 1, padded_seq, patch_dim),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = model.prepare_rope(rope, padded_seq)
    cu_tt = model.prepare_cu_seqlens(cu_seqlens)
    out_tt = model.forward(x_tt, rot_mats, cu_tt)
    # Replicated output: one device's copy, sliced to the unpadded merged rows.
    merged_seq = seq // spatial_merge_size**2
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).float()[:merged_seq]
    assert out.shape == ref_out.shape, f"{out.shape} != {ref_out.shape}"
    return _pcc(ref_out, out), wl.count_params(sd)


BLOCKS = [
    ("vision_patch_embed", _t_vision_patch_embed),
    ("vision_rmsnorm", _t_vision_rmsnorm),
    ("vision_attention", _t_vision_attention),
    ("vision_mlp", _t_vision_mlp),
    ("vision_block", _t_vision_block),
    ("patch_merger", _t_patch_merger),
    ("vision_transformer", _t_vision_transformer),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("name,fn", BLOCKS, ids=[b[0] for b in BLOCKS])
def test_real_hf_weights(name, fn, mesh_device):
    pcc, n_params = fn(mesh_device)
    print(f"[{name}] real-HF PCC = {pcc:.6f} ({n_params} params loaded)")
    assert pcc > 0.99, f"{name}: PCC {pcc:.6f} <= 0.99"
