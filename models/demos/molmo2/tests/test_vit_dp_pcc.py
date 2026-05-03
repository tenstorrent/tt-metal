# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for ViT encoder with Data Parallel (DP) weights.

Instead of Tensor Parallel (TP=8):
  - wqkv: ShardTensorToMesh(dim=3) → column-parallel, n_local_heads=2/device
  - wo/w1/w2: ShardTensorToMesh(dim=2) → row-parallel + ttnn.all_reduce

DP uses:
  - ALL weights: ReplicateTensorToMesh → all 16 heads on each device
  - NO ttnn.all_reduce after wo or w2
  - Each device computes the full output independently (no CCL)

With sharded input (one future production step), each device would process a
different subset of crops — eliminating ALL per-block CCL (50 calls / batch).
This test uses replicated input to verify numerical correctness first.

Does NOT modify any model code.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

import ttnn

GOLDEN_DIR = Path(__file__).parent.parent / "reference" / "golden"
HF_PATH = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
    )
)

_MESH_SHAPE = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 8))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _MESH_SHAPE
    is_single = rows * cols == 1
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    yield device
    ttnn.close_mesh_device(device)
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def state_dict():
    sys.path.insert(0, str(HF_PATH))
    from transformers import AutoModelForImageTextToText

    hf = AutoModelForImageTextToText.from_pretrained(
        str(HF_PATH), trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    sd = hf.state_dict()
    del hf
    return sd


@pytest.fixture(scope="module")
def molmo2_cfg(mesh_device):
    from models.demos.molmo2.tt.model_config import Molmo2Config

    return Molmo2Config(mesh_device=mesh_device)


# ---------------------------------------------------------------------------
# DP ViT forward — no CCL, all weights replicated
# ---------------------------------------------------------------------------


def _tt(t, mesh_device, mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def vit_forward_dp(pixel_values_tt, state_dict, cfg, mesh_device):
    """
    Run 25-block ViT encoder with DATA PARALLEL weights — no CCL.

    All weights replicated on all devices. Each device runs full attention
    (all 16 heads) and full MLP (all 4304 channels) independently.
    No ttnn.all_reduce needed — each device already has the full output.

    Args:
        pixel_values_tt: [n_crops, 1, 729, 588] TILE_LAYOUT on device (replicated)
        state_dict: HF model state dict
        cfg: Molmo2Config

    Returns:
        image_features: [n_crops, 729, 2304] float32 CPU
    """
    import torch.nn.functional as F

    from models.demos.qwen3_vl.tt.vision_layernorm import LayerNorm

    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    hifi2 = cfg.compute_kernel_config_hifi2
    hifi4 = cfg.compute_kernel_config_hifi4

    vit_pfx = "model.vision_backbone.image_vit"
    n_heads = cfg.vit_n_heads  # 16
    head_dim = cfg.vit_head_dim  # 72
    padded_head_dim = cfg.vit_padded_head_dim  # 96
    hidden = cfg.vit_hidden  # 1152
    scale = head_dim**-0.5
    n_layers = cfg.vit_n_layers  # 25
    capture_layers = set(cfg.vit_capture_layers)  # {24, 18}

    def pad_w(w):
        """Pad head_dim 72→96."""
        w = w.reshape(n_heads, head_dim, -1)
        w = F.pad(w, (0, 0, 0, padded_head_dim - head_dim))
        return w.reshape(n_heads * padded_head_dim, -1)

    def pad_b(b):
        b = b.reshape(n_heads, head_dim)
        b = F.pad(b, (0, padded_head_dim - head_dim))
        return b.reshape(-1)

    # ---- Patch embedding ----
    patch_w = state_dict[f"{vit_pfx}.patch_embedding.weight"].T  # [588, 1152]
    patch_b = state_dict[f"{vit_pfx}.patch_embedding.bias"]
    patch_w_tt = _tt(patch_w.unsqueeze(0).unsqueeze(0), mesh_device, rep)
    patch_b_tt = _tt(patch_b, mesh_device, rep)
    x = ttnn.linear(
        pixel_values_tt, patch_w_tt, bias=patch_b_tt, compute_kernel_config=hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # ---- Positional embedding (bicubic for non-27×27 not needed for golden) ----
    pos_emb = state_dict[f"{vit_pfx}.positional_embedding"].to(torch.bfloat16)
    pos_emb_tt = _tt(pos_emb.reshape(1, 1, 729, hidden), mesh_device, rep)
    x = ttnn.add(x, pos_emb_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ---- 25 ViT blocks ----
    sdpa_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    captured = {}

    for i in range(n_layers):
        lpfx = f"{vit_pfx}.transformer.resblocks.{i}"

        # LayerNorm (replicated — same as current code)
        attn_norm = LayerNorm(
            device=mesh_device,
            dim=hidden,
            eps=cfg.vit_norm_eps,
            state_dict=state_dict,
            state_dict_prefix=f"{lpfx}.attention_norm",
            weight_cache_path=None,
            weight_dtype=ttnn.bfloat16,
        )
        ffn_norm = LayerNorm(
            device=mesh_device,
            dim=hidden,
            eps=cfg.vit_norm_eps,
            state_dict=state_dict,
            state_dict_prefix=f"{lpfx}.ffn_norm",
            weight_cache_path=None,
            weight_dtype=ttnn.bfloat16,
        )

        # --- Attention with DP weights (n_local_heads = ALL 16 heads) ---
        wq = pad_w(state_dict[f"{lpfx}.attention.wq.weight"])
        wk = pad_w(state_dict[f"{lpfx}.attention.wk.weight"])
        wv = pad_w(state_dict[f"{lpfx}.attention.wv.weight"])
        bq = pad_b(state_dict[f"{lpfx}.attention.wq.bias"])
        bk = pad_b(state_dict[f"{lpfx}.attention.wk.bias"])
        bv = pad_b(state_dict[f"{lpfx}.attention.wv.bias"])
        # Fuse QKV: [1, 1, hidden, 3*n_heads*padded_head_dim] — ALL heads, REPLICATED
        wqkv = torch.cat([wq.T, wk.T, wv.T], dim=-1)
        bqkv = torch.cat([bq, bk, bv])
        wqkv_tt = _tt(wqkv.unsqueeze(0).unsqueeze(0), mesh_device, rep)
        bqkv_tt = _tt(bqkv, mesh_device, rep)
        # wo: row-parallel in TP, but replicated here → [hidden, n_heads*padded] REPLICATED
        wo_raw = state_dict[f"{lpfx}.attention.wo.weight"]
        if head_dim != padded_head_dim:
            wo_raw = wo_raw.reshape(-1, n_heads, head_dim)
            wo_raw = F.pad(wo_raw, (0, padded_head_dim - head_dim))
            wo_raw = wo_raw.reshape(-1, n_heads * padded_head_dim)
        wo_tt = _tt(wo_raw.T.unsqueeze(0).unsqueeze(0), mesh_device, rep)
        bo_tt = _tt(state_dict[f"{lpfx}.attention.wo.bias"], mesh_device, rep)

        # Forward
        attn_in = attn_norm(x)
        xqkv = ttnn.add(
            ttnn.linear(attn_in, wqkv_tt, compute_kernel_config=hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            bqkv_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_in)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=n_heads,
            num_kv_heads=n_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=scale,
            program_config=sdpa_cfg,
            compute_kernel_config=hifi4,
        )
        for t in (q, k, v):
            ttnn.deallocate(t)

        attn_out = ttnn.reshape(attn_out, [attn_out.shape[0], n_heads, -1, padded_head_dim])
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # wo projection — NO ttnn.all_reduce (DP: each device has full output)
        wo_out = ttnn.add(
            ttnn.linear(attn_out, wo_tt, compute_kernel_config=hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            bo_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_out)

        h = ttnn.add(x, wo_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(wo_out)

        # --- MLP with DP weights (full 4304 channels, REPLICATED) ---
        w1_tt = _tt(state_dict[f"{lpfx}.feed_forward.w1.weight"].T.unsqueeze(0).unsqueeze(0), mesh_device, rep)
        b1_tt = _tt(state_dict[f"{lpfx}.feed_forward.w1.bias"], mesh_device, rep)
        w2_tt = _tt(state_dict[f"{lpfx}.feed_forward.w2.weight"].T.unsqueeze(0).unsqueeze(0), mesh_device, rep)
        b2_tt = _tt(state_dict[f"{lpfx}.feed_forward.w2.bias"], mesh_device, rep)

        ff_in = ffn_norm(h)
        ff_hidden = ttnn.linear(
            ff_in,
            w1_tt,
            bias=b1_tt,
            activation="gelu",
            compute_kernel_config=hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ff_in)

        # w2 — NO ttnn.all_reduce (DP: each device has full output)
        ff_out = ttnn.add(
            ttnn.linear(ff_hidden, w2_tt, compute_kernel_config=hifi2, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            b2_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ff_hidden)

        out = ttnn.add(h, ff_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        x = out

        if i in capture_layers:
            captured[i] = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ---- Concatenate captured features ----
    feats = [ttnn.to_layout(captured[l], ttnn.ROW_MAJOR_LAYOUT) for l in cfg.vit_capture_layers]
    image_features = ttnn.concat(feats, dim=-1)
    for f in feats:
        ttnn.deallocate(f)
    ttnn.deallocate(x)

    is_mesh = mesh_device.__class__.__name__ == "MeshDevice"

    if is_mesh:
        # Check if output is sharded (one crop per device) or replicated
        n_devices = mesh_device.get_num_devices()
        first_dev_shape = list(ttnn.get_device_tensors(image_features)[0].shape)
        if first_dev_shape[0] < image_features.shape[0]:
            # Sharded: collect crops from all devices
            all_parts = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(image_features)]
            out_cpu = torch.cat(all_parts, dim=0)
        else:
            # Replicated: pull from device 0
            out_cpu = ttnn.to_torch(ttnn.get_device_tensors(image_features)[0]).float()
    else:
        out_cpu = ttnn.to_torch(image_features).float()

    ttnn.deallocate(image_features)
    return out_cpu  # [n_crops, 1, 729, 2304] or [n_crops, 729, 2304]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_vit_dp_pcc(mesh_device, state_dict, molmo2_cfg):
    """
    ViT encoder PCC with DP weights (replicated, no CCL) vs golden.

    Confirms that eliminating all ttnn.all_reduce from ViT blocks
    (Data Parallel mode) gives numerically equivalent results to the
    current TP=8 implementation, before making any model code changes.
    """

    def pcc(a, b):
        a = a.to(torch.float32).flatten()
        b = b.to(torch.float32).flatten()
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    golden = torch.load(GOLDEN_DIR / "vit_encode.pt", weights_only=False)
    pixel_values = golden["pixel_values"].to(torch.bfloat16)  # [2, 729, 588]
    ref_features = golden["ref_features"].float()  # [2, 729, 2304]

    n_crops = pixel_values.shape[0]
    pv_tt = ttnn.from_torch(
        pixel_values.unsqueeze(1),  # [2, 1, 729, 588]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out = vit_forward_dp(pv_tt, state_dict, molmo2_cfg, mesh_device)
    ttnn.deallocate(pv_tt)

    # Reshape to [n_crops, 729, 2304]
    out = out.reshape(n_crops, 729, 2304)

    pcc_val = pcc(out, ref_features)
    print(f"\n  ViT DP (no CCL) PCC = {pcc_val:.6f}")
    print(f"  out  mean={out.mean():.4f}  std={out.std():.4f}")
    print(f"  ref  mean={ref_features.mean():.4f}  std={ref_features.std():.4f}")
    assert pcc_val > 0.99, f"PCC {pcc_val:.6f} < 0.99"
