# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for Molmo2-8B TTNN text decoder components.

Tests attention, MLP, and full decoder block against golden outputs
saved by reference/test_functional.py.

Run:
    cd /home/ttuser/ssinghal/PR-fix/molmo2/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    MESH_DEVICE=T3K pytest models/demos/molmo2/tests/test_tt_text_decoder.py -v

Requirements:
  - T3K or N300 machine
  - models/demos/molmo2/reference/golden/ must exist
  - HF checkpoint at ~/.cache/huggingface/hub/models--allenai--Molmo2-8B/
"""

import os
import sys
from pathlib import Path

import pytest
import torch

import ttnn

GOLDEN_DIR = Path(__file__).parent.parent / "reference" / "golden"
HF_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/" "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
)
PCC_THRESHOLD = 0.99

_MESH_SHAPE = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ------------------------------------------------------------------ #
# Module-scoped mesh_device — opens the device once for all tests here.
# Overrides the function-scoped global conftest fixture to allow
# module-scoped fixtures (state_dict, molmo2_cfg, tt_ccl, etc.) that
# would otherwise hit a ScopeMismatch error.
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _MESH_SHAPE if isinstance(_MESH_SHAPE, tuple) else (1, _MESH_SHAPE)
    is_single = rows * cols == 1
    # Enable fabric for CCL on multi-chip (mirrors conftest device_params fabric_config=True)
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    yield device
    ttnn.close_mesh_device(device)
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ------------------------------------------------------------------ #
# Module-scoped heavy fixtures (loaded once, shared across all tests)
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def state_dict():
    sys.path.insert(0, HF_MODEL_PATH)
    from transformers import AutoModelForImageTextToText

    print("\nLoading Molmo2-8B state dict (fp32)...")
    model = AutoModelForImageTextToText.from_pretrained(
        HF_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
    )
    sd = model.state_dict()
    del model
    return sd


@pytest.fixture(scope="module")
def molmo2_cfg(mesh_device):
    from models.demos.molmo2.tt.model_config import Molmo2Config

    return Molmo2Config(mesh_device=mesh_device)


@pytest.fixture(scope="module")
def tt_ccl(mesh_device):
    from models.tt_transformers.tt.ccl import TT_CCL

    return TT_CCL(mesh_device)


@pytest.fixture(scope="module")
def transformation_mats(mesh_device, molmo2_cfg):
    from models.tt_transformers.tt.common import get_rot_transformation_mat

    def _tt(mat):
        return ttnn.as_tensor(
            mat,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return {
        "prefill": _tt(get_rot_transformation_mat(dhead=molmo2_cfg.head_dim)),
        "decode": _tt(get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)),
    }


@pytest.fixture(scope="module")
def rot_mats_for_seq(mesh_device, molmo2_cfg):
    """Returns a callable: seq_len -> [cos_ttnn, sin_ttnn] in HF rotate_half format.

    Uses ttnn.experimental.rotary_embedding (HF-style), matching Molmo2's rotate_half RoPE.
    Format: [c0,...,c63, c0,...,c63] (concatenated halves), shape [1, 1, S, head_dim].
    """
    from models.tt_transformers.tt.common import precompute_freqs

    cos_raw, sin_raw = precompute_freqs(
        molmo2_cfg.head_dim, molmo2_cfg.max_seq_len * 2, molmo2_cfg.rope_theta, None, None
    )
    # HF concatenated-halves format
    cos_hf = torch.cat(
        [cos_raw[: molmo2_cfg.max_seq_len], cos_raw[: molmo2_cfg.max_seq_len]], dim=-1
    )  # [max_seq, head_dim]
    sin_hf = torch.cat([sin_raw[: molmo2_cfg.max_seq_len], sin_raw[: molmo2_cfg.max_seq_len]], dim=-1)

    # Upload full matrix to device once (rotary_embedding needs the full matrix;
    # attention slices the output back to the actual seq_len after the op)
    cos_tt = ttnn.from_torch(
        cos_hf.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, max_seq, head_dim]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        sin_hf.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def _get(seq_len):
        return [cos_tt, sin_tt]  # full matrix; attention slices output to seq_len

    return _get


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


def test_prefill_mask(mesh_device):
    """Verify build_molmo2_prefill_mask: causal + image-bidir invariants."""
    from models.demos.molmo2.tt.prefill_mask import build_molmo2_prefill_mask

    B, S = 1, 20
    token_type_ids = torch.zeros(B, S, dtype=torch.long)
    token_type_ids[0, 5:15] = 1  # image tokens at positions 5..14

    mask_ttnn = build_molmo2_prefill_mask(S, token_type_ids, mesh_device, dtype=ttnn.bfloat16)
    mask = ttnn.to_torch(ttnn.get_device_tensors(mask_ttnn)[0]).float()
    ttnn.deallocate(mask_ttnn)

    assert mask[0, 0, 3, 4].item() == float("-inf"), "Text token should not see future"
    assert mask[0, 0, 3, 3].item() == 0.0, "Text token should see itself"
    assert mask[0, 0, 5, 14].item() == 0.0, "Image q=5 should see image kv=14"
    assert mask[0, 0, 14, 5].item() == 0.0, "Image q=14 should see image kv=5"
    assert mask[0, 0, 3, 10].item() == float("-inf"), "Text q=3 should not see image kv=10"
    assert mask[0, 0, 8, 17].item() == float("-inf"), "Image q=8 should not see text kv=17"
    print("  [PASS] Prefill mask invariants hold")


def test_text_attention_pcc(mesh_device, state_dict, molmo2_cfg, tt_ccl, transformation_mats, rot_mats_for_seq):
    """TtMolmo2TextAttention PCC > 0.99 against reference on block 0."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
    from functional import build_rope_cache, load_decoder_block_weights, rmsnorm, text_attention

    golden = torch.load(GOLDEN_DIR / "decoder_block0.pt", weights_only=False)
    hidden = golden["input"].to(torch.float32)
    B, S, H = hidden.shape

    w0 = load_decoder_block_weights(state_dict, 0)
    cos, sin = build_rope_cache(S, 128, 1e6, hidden.device)
    mask = torch.triu(torch.full((1, 1, S, S), float("-inf")), diagonal=1)
    normed = rmsnorm(hidden, w0["attn_norm_weight"])
    ref_out = text_attention(
        normed,
        w0["att_proj_weight"],
        w0["attn_out_weight"],
        w0["q_norm_weight"],
        w0["k_norm_weight"],
        cos,
        sin,
        mask,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )

    from models.demos.molmo2.tt.attention import TtMolmo2TextAttention

    attn = TtMolmo2TextAttention(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat16,
        transformation_mats=transformation_mats,
        configuration=molmo2_cfg,
    )

    x_ttnn = ttnn.from_torch(
        normed.to(torch.bfloat16).reshape(1, 1, S, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_ttnn = attn.forward_prefill(x_ttnn, rot_mats_for_seq(S), user_id=0)
    out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out_ttnn)[0]).float().reshape(1, S, H)
    ttnn.deallocate(out_ttnn)

    val = pcc(ref_out, out_cpu)
    print(f"  Text attention PCC = {val:.6f}")
    assert val >= PCC_THRESHOLD, f"Attention PCC {val:.4f} < {PCC_THRESHOLD}"


def test_text_mlp_pcc(mesh_device, state_dict, molmo2_cfg, tt_ccl):
    """TtMolmo2TextMLP PCC > 0.99 against reference on block 0."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
    from functional import load_decoder_block_weights, rmsnorm, text_mlp

    from models.tt_transformers.tt.common import Mode

    golden = torch.load(GOLDEN_DIR / "decoder_block0.pt", weights_only=False)
    hidden = golden["input"].to(torch.float32)
    B, S, H = hidden.shape

    w0 = load_decoder_block_weights(state_dict, 0)
    normed = rmsnorm(hidden, w0["ff_norm_weight"])
    ref_out = text_mlp(normed, w0["ff_proj_weight"], w0["ff_out_weight"])

    from models.demos.molmo2.tt.mlp import TtMolmo2TextMLP

    mlp = TtMolmo2TextMLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat16,
        configuration=molmo2_cfg,
    )

    x_ttnn = ttnn.from_torch(
        normed.to(torch.bfloat16).reshape(1, 1, S, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_ttnn = mlp.forward(x_ttnn, mode=Mode.PREFILL)
    out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out_ttnn)[0]).float().reshape(1, S, H)
    ttnn.deallocate(out_ttnn)

    val = pcc(ref_out, out_cpu)
    print(f"  Text MLP PCC = {val:.6f}")
    assert val >= PCC_THRESHOLD, f"MLP PCC {val:.4f} < {PCC_THRESHOLD}"


def test_decoder_block_pcc(mesh_device, state_dict, molmo2_cfg, tt_ccl, transformation_mats, rot_mats_for_seq):
    """Full TtMolmo2DecoderBlock PCC > 0.99 against reference on block 0."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
    from functional import build_rope_cache, load_decoder_block_weights, text_decoder_block

    golden = torch.load(GOLDEN_DIR / "decoder_block0.pt", weights_only=False)
    hidden = golden["input"].to(torch.float32)
    ref_out = golden["block0_output"].to(torch.float32)
    B, S, H = hidden.shape

    w0 = load_decoder_block_weights(state_dict, 0)
    cos, sin = build_rope_cache(S, 128, 1e6, hidden.device)
    mask = torch.triu(torch.full((1, 1, S, S), float("-inf")), diagonal=1)
    ref_out = text_decoder_block(
        hidden,
        w0["attn_norm_weight"],
        w0["ff_norm_weight"],
        w0["att_proj_weight"],
        w0["attn_out_weight"],
        w0["q_norm_weight"],
        w0["k_norm_weight"],
        w0["ff_proj_weight"],
        w0["ff_out_weight"],
        cos,
        sin,
        mask,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )

    from models.demos.molmo2.tt.model import TtMolmo2DecoderBlock

    block = TtMolmo2DecoderBlock(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat16,
        transformation_mats=transformation_mats,
        configuration=molmo2_cfg,
    )

    x_ttnn = ttnn.from_torch(
        hidden.to(torch.bfloat16).reshape(1, 1, S, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_ttnn = block.forward(x_ttnn, rot_mats=rot_mats_for_seq(S), mode="prefill")
    out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out_ttnn)[0]).float().reshape(1, S, H)
    ttnn.deallocate(out_ttnn)

    val = pcc(ref_out, out_cpu)
    print(f"  Decoder block 0 PCC = {val:.6f}")
    assert val >= PCC_THRESHOLD, f"Decoder block PCC {val:.4f} < {PCC_THRESHOLD}"


def test_vit_encoder_pcc(mesh_device, state_dict, molmo2_cfg):
    """TtMolmo2ViTEncoder PCC > 0.99 against golden (2 crops)."""
    golden = torch.load(GOLDEN_DIR / "vit_encode.pt", weights_only=False)
    pixel_values = golden["pixel_values"].to(torch.bfloat16)  # [2, 729, 588]
    ref_features = golden["ref_features"]  # [2, 729, 2304]

    from models.demos.molmo2.tt.vision_encoder import TtMolmo2ViTEncoder

    encoder = TtMolmo2ViTEncoder(
        mesh_device=mesh_device,
        state_dict=state_dict,
        vit_cfg=molmo2_cfg,
        weight_cache_path=None,
    )

    # TP ViT: replicate all crops to all devices; pre-shape [n_crops, 1, 729, 588]
    n_crops = pixel_values.shape[0]
    pv_ttnn = ttnn.from_torch(
        pixel_values.unsqueeze(1),  # [2, 1, 729, 588]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out_ttnn = encoder.forward(pv_ttnn, patch_num=(27, 27))
    # All devices hold same result after all_reduce — pull device 0
    out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out_ttnn)[0]).float()
    ttnn.deallocate(out_ttnn)
    out_cpu = out_cpu.reshape(n_crops, 729, 2304)  # [2, 729, 2304]

    val = pcc(ref_features, out_cpu)
    print(f"  ViT encoder PCC = {val:.6f}")
    assert val >= PCC_THRESHOLD, f"ViT encoder PCC {val:.4f} < {PCC_THRESHOLD}"


def test_image_projector_pcc(mesh_device, state_dict, molmo2_cfg):
    """TtMolmo2ImageProjector PCC > 0.99: SwiGLU [1152 → 12288 → 4096]."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
    from functional import image_projector

    # Use the valid pooled features from vision_adapter golden as input
    va = torch.load(GOLDEN_DIR / "vision_adapter.pt", weights_only=False)
    # ref_proj_out = w2(silu(w1(x)) * w3(x)) — we need the pre-projection input.
    # Compute it from the stored image_features_4d by running the reference pooling.
    # Simpler: use the stored ref_proj_out as reference, test against TTNN projector
    # using a random small input where we can compute the reference inline.
    torch.manual_seed(7)
    N = 64
    x_pt = torch.randn(N, 1152, dtype=torch.float32)

    ref = image_projector(
        x_pt,
        state_dict["model.vision_backbone.image_projector.w1.weight"],
        state_dict["model.vision_backbone.image_projector.w2.weight"],
        state_dict["model.vision_backbone.image_projector.w3.weight"],
    )

    from models.demos.molmo2.tt.image_projector import TtMolmo2ImageProjector

    proj = TtMolmo2ImageProjector(
        mesh_device=mesh_device, state_dict=state_dict, cfg=molmo2_cfg, weight_cache_path=None
    )

    x_ttnn = ttnn.from_torch(
        x_pt.to(torch.bfloat16).unsqueeze(0).unsqueeze(0),  # [1, 1, N, 1152]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_ttnn = proj.forward(x_ttnn)
    out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out_ttnn)[0]).float().squeeze(0).squeeze(0)
    ttnn.deallocate(out_ttnn)

    val = pcc(ref, out_cpu)
    print(f"  Image projector PCC = {val:.6f}")
    assert val >= PCC_THRESHOLD, f"Image projector PCC {val:.4f} < {PCC_THRESHOLD}"


def test_vision_adapter_pcc(mesh_device, state_dict, molmo2_cfg):
    """TtMolmo2ImagePooling2D (TTNN chunked) + TtMolmo2ImageProjector PCC > 0.99.

    Uses image_features_4d from vision_adapter golden and image_token_pooling
    from full_prefill golden. Exercises the full TTNN pooling path:
      ttnn.embedding gather → device masked mean → device attn mask → cross-attn.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
    va = torch.load(GOLDEN_DIR / "vision_adapter.pt", weights_only=False)
    fp = torch.load(GOLDEN_DIR / "full_prefill.pt", weights_only=False)

    image_features_4d = va["image_features_4d"].float()  # [1, 9, 729, 2304]
    ref_proj_out = va["ref_proj_out"]  # [1316, 4096]
    pool_idx = fp["inputs"]["image_token_pooling"].unsqueeze(0)  # [1, 1316, 4]

    from models.demos.molmo2.tt.image_pooling import TtMolmo2ImagePooling2D
    from models.demos.molmo2.tt.image_projector import TtMolmo2ImageProjector

    pooling = TtMolmo2ImagePooling2D(
        mesh_device=mesh_device, state_dict=state_dict, cfg=molmo2_cfg, weight_cache_path=None
    )
    projector = TtMolmo2ImageProjector(
        mesh_device=mesh_device, state_dict=state_dict, cfg=molmo2_cfg, weight_cache_path=None
    )

    # ---- TTNN pooling path ----
    # Build the 2D feature table and run _run_chunked_ttnn_pooling directly.
    # We reuse the same logic as TtMolmo2Model._run_chunked_ttnn_pooling but inline
    # it here so this test has no dependency on the full model being instantiated.
    B, n_crops, n_patches, feat_dim = image_features_4d.shape
    N_pooled = pool_idx.shape[1]
    k_pool = pool_idx.shape[2]
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    feat_tt = ttnn.from_torch(
        image_features_4d.reshape(-1, feat_dim).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    idx_flat = pool_idx[0].reshape(-1).clamp(min=0).to(torch.int32)
    valid_flat = (pool_idx[0].reshape(-1) >= 0).float()

    idx_tt = ttnn.from_torch(
        idx_flat.reshape(1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    valid_tt = ttnn.from_torch(
        valid_flat.reshape(1, 1, -1, 1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    gathered = ttnn.embedding(idx_tt, feat_tt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(feat_tt)
    ttnn.deallocate(idx_tt)
    gathered = ttnn.reshape(gathered, [1, 1, N_pooled * k_pool, feat_dim])
    gathered = ttnn.mul(gathered, valid_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    to_pool = ttnn.reshape(gathered, [1, N_pooled, k_pool, feat_dim])
    ttnn.deallocate(gathered)

    query_sum = ttnn.sum(to_pool, dim=2, keepdim=True)
    vm = ttnn.reshape(valid_tt, [1, N_pooled, k_pool, 1])
    denom = ttnn.clamp(ttnn.sum(vm, dim=2, keepdim=True), min=1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(vm)
    query = ttnn.div(query_sum, denom, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(query_sum)
    ttnn.deallocate(denom)

    valid_rm = ttnn.to_layout(valid_tt, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(valid_tt)
    valid_4d = ttnn.to_layout(ttnn.reshape(valid_rm, [N_pooled, 1, 1, k_pool]), ttnn.TILE_LAYOUT)
    ttnn.deallocate(valid_rm)
    zeros_buf = ttnn.zeros_like(valid_4d)
    neg_inf_buf = ttnn.from_torch(
        torch.full((1, 1, 1, 1), float("-inf"), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    threshold = ttnn.full_like(valid_4d, 0.5)
    cond = ttnn.gt(valid_4d, threshold, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn_mask = ttnn.where(cond, zeros_buf, neg_inf_buf, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in (valid_4d, zeros_buf, neg_inf_buf, threshold, cond):
        ttnn.deallocate(t)

    pooled_tt = pooling.forward(query, to_pool, attn_mask)
    ttnn.deallocate(query)
    ttnn.deallocate(to_pool)
    ttnn.deallocate(attn_mask)

    pooled_cpu = ttnn.to_torch(ttnn.get_device_tensors(pooled_tt)[0]).float()
    ttnn.deallocate(pooled_tt)
    pooled_cpu = pooled_cpu.squeeze(0).squeeze(1)  # [N_pooled, 1152]

    # Apply valid-token filter
    valid = (pool_idx[0] >= 0).any(dim=-1)
    pooled_valid = pooled_cpu[valid]  # [N_valid, 1152]

    # Project on device
    pooled_ttnn = ttnn.from_torch(
        pooled_valid.to(torch.bfloat16).unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    proj_out = projector.forward(pooled_ttnn)
    proj_cpu = ttnn.to_torch(ttnn.get_device_tensors(proj_out)[0]).float().squeeze(0).squeeze(0)
    ttnn.deallocate(proj_out)

    val = pcc(ref_proj_out, proj_cpu)
    print(f"  Vision adapter (pooling+projector) PCC = {val:.6f}")
    assert val >= PCC_THRESHOLD, f"Vision adapter PCC {val:.4f} < {PCC_THRESHOLD}"


def test_decoder_block_with_image_pcc(
    mesh_device, state_dict, molmo2_cfg, tt_ccl, transformation_mats, rot_mats_for_seq
):
    """Full decoder block 0 with image embeddings injected, PCC > 0.99.

    Uses block0_with_image golden: input already has ViT features additively
    injected at image_patch_id positions. token_type_ids for the bidir mask
    comes from full_prefill golden inputs.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))

    bi = torch.load(GOLDEN_DIR / "block0_with_image.pt", weights_only=False)
    fp = torch.load(GOLDEN_DIR / "full_prefill.pt", weights_only=False)

    hidden = bi["input_embeddings"].to(torch.bfloat16)  # [1, 1347, 4096]
    ref_out = bi["ref_block0_out"].to(torch.float32)
    token_type_ids = fp["inputs"]["token_type_ids"]  # [1, 1347]
    B, S, H = hidden.shape

    # Build combined prefill mask (causal + image-bidir)
    from models.demos.molmo2.tt.prefill_mask import build_molmo2_prefill_mask

    attn_mask = build_molmo2_prefill_mask(S, token_type_ids.long(), mesh_device, dtype=ttnn.bfloat8_b)

    from models.demos.molmo2.tt.model import TtMolmo2DecoderBlock

    block = TtMolmo2DecoderBlock(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat16,
        transformation_mats=transformation_mats,
        configuration=molmo2_cfg,
    )

    x_ttnn = ttnn.from_torch(
        hidden.reshape(1, 1, S, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_ttnn = block.forward(x_ttnn, rot_mats=rot_mats_for_seq(S), mode="prefill", attn_mask=attn_mask)
    ttnn.deallocate(attn_mask)

    out_cpu = ttnn.to_torch(ttnn.get_device_tensors(out_ttnn)[0]).float().reshape(1, S, H)
    ttnn.deallocate(out_ttnn)

    val = pcc(ref_out, out_cpu)
    print(f"  Decoder block 0 (with image embeddings) PCC = {val:.6f}")
    assert val >= PCC_THRESHOLD, f"Block-with-image PCC {val:.4f} < {PCC_THRESHOLD}"
