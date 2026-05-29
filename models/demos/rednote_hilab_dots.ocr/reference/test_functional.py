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


if __name__ == "__main__":
    test_vision_rmsnorm()
    test_vision_patch_embed()
    test_vision_mlp()
    test_vision_attention()
    print("all reference blocks pass")
