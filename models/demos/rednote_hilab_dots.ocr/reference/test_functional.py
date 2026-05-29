# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify the standalone dots.ocr reference blocks against the HuggingFace modules.

The HF modeling lives in the trust_remote_code snapshot. We instantiate each HF
module with the verified dots_vit config and random-init weights (seeded), copy the
SAME weights into the standalone functional reference, run both, assert PCC > 0.99,
and save golden {input, output, state_dict-ish, config} tensors for the TTNN worker.

Run: pytest models/demos/rednote_hilab_dots.ocr/reference/test_functional.py -v
"""

import importlib.util
import os
import sys

import torch

REF_DIR = os.path.dirname(__file__)
GOLDEN_DIR = os.path.join(REF_DIR, "golden")
SNAPSHOT = (
    "/local/ttuser/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/"
    "snapshots/c0111ce6bc07803dbc267932ffef0ae3a51dc951"
)

sys.path.insert(0, REF_DIR)
import functional as fn  # noqa: E402


def _load_hf_vision_module():
    """Import modeling_dots_vision.py from the HF snapshot as a standalone module."""
    path = os.path.join(SNAPSHOT, "modeling_dots_vision.py")
    spec = importlib.util.spec_from_file_location("dots_modeling_vision", path)
    mod = importlib.util.module_from_spec(spec)
    # The module does `from .configuration_dots import DotsVisionConfig`; satisfy it.
    cfg_path = os.path.join(SNAPSHOT, "configuration_dots.py")
    cfg_spec = importlib.util.spec_from_file_location("configuration_dots", cfg_path)
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    pkg = type(sys)("dots_pkg")
    pkg.__path__ = [SNAPSHOT]
    sys.modules["dots_pkg"] = pkg
    sys.modules["dots_pkg.configuration_dots"] = cfg_mod
    cfg_spec.loader.exec_module(cfg_mod)
    mod.__package__ = "dots_pkg"
    sys.modules["dots_pkg.modeling_dots_vision"] = mod
    spec.loader.exec_module(mod)
    return mod, cfg_mod


VMOD, CFGMOD = _load_hf_vision_module()


def _vision_config():
    # Force eager attention so the HF reference path matches our standalone math.
    return CFGMOD.DotsVisionConfig(attn_implementation="eager")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# --------------------------------------------------------------------------- #
# vision_rmsnorm
# --------------------------------------------------------------------------- #
def test_vision_rmsnorm():
    torch.manual_seed(0)
    cfg = _vision_config()
    dim, eps = cfg.embed_dim, cfg.rms_norm_eps
    hf = VMOD.RMSNorm(dim, eps=eps)
    hf.weight.data.normal_(mean=1.0, std=0.05)
    x = torch.randn(256, dim)
    hf_out = hf(x)
    ref_out = fn.vision_rmsnorm_forward(x, hf.weight.data, eps=eps)
    p = pcc(hf_out, ref_out)
    torch.save(
        {"input": x, "output": hf_out, "weight": hf.weight.data, "eps": eps, "dim": dim},
        os.path.join(GOLDEN_DIR, "vision_rmsnorm.pt"),
    )
    assert p > 0.99, f"vision_rmsnorm PCC {p}"


# --------------------------------------------------------------------------- #
# vision_patch_embed
# --------------------------------------------------------------------------- #
def test_vision_patch_embed():
    torch.manual_seed(0)
    cfg = _vision_config()
    pre = VMOD.DotsViTPreprocessor(cfg)  # holds DotsPatchEmbed as .patchifier
    pe = pre.patchifier
    num_patches = 64
    flat = cfg.num_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    pixel_values = torch.randn(num_patches, flat)
    hf_out = pe(pixel_values)
    sd = {
        "proj.weight": pe.proj.weight.data,
        "proj.bias": pe.proj.bias.data,
        "norm.weight": pe.norm.weight.data,
    }
    ref_out = fn.vision_patch_embed_forward(
        pixel_values,
        sd,
        num_channels=cfg.num_channels,
        temporal_patch_size=cfg.temporal_patch_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        eps=cfg.rms_norm_eps,
    )
    p = pcc(hf_out, ref_out)
    torch.save(
        {
            "input": pixel_values,
            "output": hf_out,
            "state_dict": sd,
            "config": {
                "num_channels": cfg.num_channels,
                "temporal_patch_size": cfg.temporal_patch_size,
                "patch_size": cfg.patch_size,
                "embed_dim": cfg.embed_dim,
                "eps": cfg.rms_norm_eps,
            },
        },
        os.path.join(GOLDEN_DIR, "vision_patch_embed.pt"),
    )
    assert p > 0.99, f"vision_patch_embed PCC {p}"


# --------------------------------------------------------------------------- #
# vision_mlp
# --------------------------------------------------------------------------- #
def test_vision_mlp():
    torch.manual_seed(0)
    cfg = _vision_config()
    hf = VMOD.DotsSwiGLUFFN(cfg)
    x = torch.randn(256, cfg.embed_dim)
    hf_out = hf(x)
    sd = {
        "fc1.weight": hf.fc1.weight.data,
        "fc2.weight": hf.fc2.weight.data,
        "fc3.weight": hf.fc3.weight.data,
    }
    ref_out = fn.vision_mlp_forward(x, sd, bias=cfg.use_bias)
    p = pcc(hf_out, ref_out)
    torch.save(
        {
            "input": x,
            "output": hf_out,
            "state_dict": sd,
            "config": {
                "embed_dim": cfg.embed_dim,
                "intermediate_size": cfg.intermediate_size,
                "use_bias": cfg.use_bias,
            },
        },
        os.path.join(GOLDEN_DIR, "vision_mlp.pt"),
    )
    assert p > 0.99, f"vision_mlp PCC {p}"


# --------------------------------------------------------------------------- #
# vision_attention
# --------------------------------------------------------------------------- #
def test_vision_attention():
    torch.manual_seed(0)
    cfg = _vision_config()
    dim = cfg.embed_dim
    num_heads = cfg.num_attention_heads
    head_dim = dim // num_heads

    hf = VMOD.VisionAttention(cfg, dim, num_heads=num_heads, bias=cfg.use_bias)

    # Two packed "images" -> block-diagonal attention over 96 + 160 = 256 patches.
    cu_seqlens = torch.tensor([0, 96, 256], dtype=torch.int32)
    seq_length = int(cu_seqlens[-1])
    x = torch.randn(seq_length, dim)

    # rot_pos_emb produces a [seq_length, head_dim//2] freqs table; emulate with a
    # VisionRotaryEmbedding over the flattened (h,w) ids -> here just a contiguous
    # freqs table of the right width, identical for HF and ref.
    rotary = VMOD.VisionRotaryEmbedding(head_dim // 2)
    freqs_full = rotary(seq_length)  # [seq_length, head_dim//4]
    rotary_pos_emb = torch.cat([freqs_full, freqs_full], dim=-1)  # [seq_length, head_dim//2]

    hf_out = hf(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
    sd = {"qkv.weight": hf.qkv.weight.data, "proj.weight": hf.proj.weight.data}
    ref_out = fn.vision_attention_forward(x, sd, cu_seqlens, rotary_pos_emb, num_heads=num_heads, bias=cfg.use_bias)
    p = pcc(hf_out, ref_out)
    torch.save(
        {
            "input": x,
            "output": hf_out,
            "state_dict": sd,
            "cu_seqlens": cu_seqlens,
            "rotary_pos_emb": rotary_pos_emb,
            "config": {"embed_dim": dim, "num_heads": num_heads, "head_dim": head_dim, "use_bias": cfg.use_bias},
        },
        os.path.join(GOLDEN_DIR, "vision_attention.pt"),
    )
    assert p > 0.99, f"vision_attention PCC {p}"


# --------------------------------------------------------------------------- #
# vision_block (one full DotsVisionBlock layer)
# --------------------------------------------------------------------------- #
def test_vision_block():
    torch.manual_seed(0)
    cfg = _vision_config()
    dim = cfg.embed_dim
    num_heads = cfg.num_attention_heads
    head_dim = dim // num_heads

    hf = VMOD.DotsVisionBlock(cfg, attn_implementation="eager")

    cu_seqlens = torch.tensor([0, 96, 256], dtype=torch.int32)
    seq_length = int(cu_seqlens[-1])
    x = torch.randn(seq_length, dim)

    rotary = VMOD.VisionRotaryEmbedding(head_dim // 2)
    freqs_full = rotary(seq_length)
    rotary_pos_emb = torch.cat([freqs_full, freqs_full], dim=-1)

    hf_out = hf(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
    sd = {
        "norm1.weight": hf.norm1.weight.data,
        "attn.qkv.weight": hf.attn.qkv.weight.data,
        "attn.proj.weight": hf.attn.proj.weight.data,
        "norm2.weight": hf.norm2.weight.data,
        "mlp.fc1.weight": hf.mlp.fc1.weight.data,
        "mlp.fc2.weight": hf.mlp.fc2.weight.data,
        "mlp.fc3.weight": hf.mlp.fc3.weight.data,
    }
    ref_out = fn.vision_block_forward(
        x, sd, cu_seqlens, rotary_pos_emb, num_heads=num_heads, eps=cfg.rms_norm_eps, bias=cfg.use_bias
    )
    p = pcc(hf_out, ref_out)
    torch.save(
        {
            "input": x,
            "output": hf_out,
            "state_dict": sd,
            "cu_seqlens": cu_seqlens,
            "rotary_pos_emb": rotary_pos_emb,
            "config": {
                "embed_dim": dim,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "rms_norm_eps": cfg.rms_norm_eps,
                "use_bias": cfg.use_bias,
            },
        },
        os.path.join(GOLDEN_DIR, "vision_block.pt"),
    )
    assert p > 0.99, f"vision_block PCC {p}"


# --------------------------------------------------------------------------- #
# vision_patch_merger (PatchMerger: LayerNorm + GELU MLP)
# --------------------------------------------------------------------------- #
def test_vision_patch_merger():
    torch.manual_seed(0)
    cfg = _vision_config()
    context_dim = cfg.embed_dim
    out_dim = cfg.hidden_size
    merge = cfg.spatial_merge_size

    hf = VMOD.PatchMerger(
        dim=out_dim,
        context_dim=context_dim,
        spatial_merge_size=merge,
        init_merger_std=cfg.init_merger_std,
    )
    # init_merger_std zeros the mlp biases; perturb them so the test exercises bias too.
    hf.mlp[0].bias.data.normal_(std=0.02)
    hf.mlp[2].bias.data.normal_(std=0.02)

    # num_patches must be a multiple of merge**2 (=4).
    num_patches = 64
    x = torch.randn(num_patches, context_dim)

    hf_out = hf(x)
    sd = {
        "ln_q.weight": hf.ln_q.weight.data,
        "ln_q.bias": hf.ln_q.bias.data,
        "mlp.0.weight": hf.mlp[0].weight.data,
        "mlp.0.bias": hf.mlp[0].bias.data,
        "mlp.2.weight": hf.mlp[2].weight.data,
        "mlp.2.bias": hf.mlp[2].bias.data,
    }
    ref_out = fn.vision_patch_merger_forward(x, sd, context_dim=context_dim, spatial_merge_size=merge, ln_eps=1e-6)
    p = pcc(hf_out, ref_out)
    torch.save(
        {
            "input": x,
            "output": hf_out,
            "state_dict": sd,
            "config": {
                "context_dim": context_dim,
                "out_dim": out_dim,
                "spatial_merge_size": merge,
                "hidden_size": context_dim * (merge**2),
                "ln_eps": 1e-6,
            },
        },
        os.path.join(GOLDEN_DIR, "vision_patch_merger.pt"),
    )
    assert p > 0.99, f"vision_patch_merger PCC {p}"


# --------------------------------------------------------------------------- #
# vision_tower (full DotsVisionTransformer, reduced layer count)
# --------------------------------------------------------------------------- #
def test_vision_tower():
    torch.manual_seed(0)
    cfg = _vision_config()
    # Reduce depth for a small golden; full 42-layer check happens in real_weights.
    REDUCED_LAYERS = 2
    cfg.num_hidden_layers = REDUCED_LAYERS

    hf = VMOD.DotsVisionTransformer(cfg)
    hf.eval()

    # Single image grid: t=1, h=4, w=4 -> 16 patches (multiple of merge**2=4).
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    num_patches = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
    flat = cfg.num_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    pixel_values = torch.randn(num_patches, flat)

    # HF forward (fp32 path: bf16=False) to match the fp32 reference exactly.
    with torch.no_grad():
        hf_out = hf(pixel_values, grid_thw, bf16=False)

    sd = {
        "patch_embed.proj.weight": hf.patch_embed.patchifier.proj.weight.data,
        "patch_embed.proj.bias": hf.patch_embed.patchifier.proj.bias.data,
        "patch_embed.norm.weight": hf.patch_embed.patchifier.norm.weight.data,
        "post_trunk_norm.weight": hf.post_trunk_norm.weight.data,
        "merger.ln_q.weight": hf.merger.ln_q.weight.data,
        "merger.ln_q.bias": hf.merger.ln_q.bias.data,
        "merger.mlp.0.weight": hf.merger.mlp[0].weight.data,
        "merger.mlp.0.bias": hf.merger.mlp[0].bias.data,
        "merger.mlp.2.weight": hf.merger.mlp[2].weight.data,
        "merger.mlp.2.bias": hf.merger.mlp[2].bias.data,
    }
    for i in range(REDUCED_LAYERS):
        blk = hf.blocks[i]
        sd[f"blocks.{i}.norm1.weight"] = blk.norm1.weight.data
        sd[f"blocks.{i}.attn.qkv.weight"] = blk.attn.qkv.weight.data
        sd[f"blocks.{i}.attn.proj.weight"] = blk.attn.proj.weight.data
        sd[f"blocks.{i}.norm2.weight"] = blk.norm2.weight.data
        sd[f"blocks.{i}.mlp.fc1.weight"] = blk.mlp.fc1.weight.data
        sd[f"blocks.{i}.mlp.fc2.weight"] = blk.mlp.fc2.weight.data
        sd[f"blocks.{i}.mlp.fc3.weight"] = blk.mlp.fc3.weight.data

    ref_out = fn.vision_tower_forward(
        pixel_values,
        grid_thw,
        sd,
        num_layers=REDUCED_LAYERS,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_attention_heads,
        num_channels=cfg.num_channels,
        temporal_patch_size=cfg.temporal_patch_size,
        patch_size=cfg.patch_size,
        spatial_merge_size=cfg.spatial_merge_size,
        rms_norm_eps=cfg.rms_norm_eps,
        ln_eps=1e-6,
        post_norm=cfg.post_norm,
        bias=cfg.use_bias,
        hidden_size=cfg.hidden_size,
    )
    p = pcc(hf_out, ref_out)
    torch.save(
        {
            "input": pixel_values,
            "grid_thw": grid_thw,
            "output": hf_out,
            "state_dict": sd,
            "config": {
                "num_layers": REDUCED_LAYERS,
                "embed_dim": cfg.embed_dim,
                "num_heads": cfg.num_attention_heads,
                "num_channels": cfg.num_channels,
                "temporal_patch_size": cfg.temporal_patch_size,
                "patch_size": cfg.patch_size,
                "spatial_merge_size": cfg.spatial_merge_size,
                "rms_norm_eps": cfg.rms_norm_eps,
                "ln_eps": 1e-6,
                "post_norm": cfg.post_norm,
                "use_bias": cfg.use_bias,
                "hidden_size": cfg.hidden_size,
                "full_num_hidden_layers": 42,
            },
        },
        os.path.join(GOLDEN_DIR, "vision_tower.pt"),
    )
    assert p > 0.99, f"vision_tower PCC {p}"


# --------------------------------------------------------------------------- #
# embedding (Qwen2 token embedding)
# --------------------------------------------------------------------------- #
def test_embedding():
    torch.manual_seed(0)
    vocab_size = 151936
    hidden_size = 1536
    hf = torch.nn.Embedding(vocab_size, hidden_size)
    hf.weight.data.normal_(mean=0.0, std=0.02)

    input_ids = torch.randint(0, vocab_size, (1, 128))
    hf_out = hf(input_ids)
    ref_out = fn.embedding_forward(input_ids, hf.weight.data)
    p = pcc(hf_out, ref_out)
    torch.save(
        {
            "input": input_ids,
            "output": hf_out,
            "weight": hf.weight.data,
            "config": {"vocab_size": vocab_size, "hidden_size": hidden_size},
        },
        os.path.join(GOLDEN_DIR, "embedding.pt"),
    )
    assert p > 0.99, f"embedding PCC {p}"


if __name__ == "__main__":
    test_vision_rmsnorm()
    test_vision_patch_embed()
    test_vision_mlp()
    test_vision_attention()
    test_vision_block()
    test_vision_patch_merger()
    test_vision_tower()
    test_embedding()
    print("all reference blocks pass")
