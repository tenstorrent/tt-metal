# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end demo for LiquidAI/LFM2.5-VL-1.6B on Tenstorrent hardware.

Preserves Liquid/HF chat-template image-placeholder semantics (``image_token_id=396``)
and variable-resolution patch preprocessing compatible with SigLIP2 NaFlex inputs.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ttnn
from models.demos.lfm2_vl.tt.model import TtLfm2VlForConditionalGeneration, TtLfm2VlModel
from models.demos.lfm2_vl.tt.model_config import HF_MODEL_ID, create_model_config


def preprocess_liquid_image(
    image_path: str,
    *,
    patch_size: int = 16,
    min_image_tokens: int = 64,
    max_image_tokens: int = 256,
    downsample_factor: int = 2,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Variable-resolution image preprocessing for LFM2.5-VL.

    Resizes so patch grid dimensions are multiples of ``downsample_factor`` (required by
    the multimodal pixel-unshuffle projector), then returns flattened patch tokens.

    Returns:
        pixel_values: ``[1, num_patches, 3 * patch_size * patch_size]``
        spatial_shapes: ``(height_patches, width_patches)`` before projector downsampling
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for real image preprocessing") from exc

    img = Image.open(image_path).convert("RGB")
    aspect = img.width / max(img.height, 1)

    # Choose a patch grid within [min_image_tokens, max_image_tokens] after unshuffle.
    # After downsample_factor=2, tokens = (H/f)*(W/f); target mid-range.
    target_tokens = int(math.sqrt(min_image_tokens * max_image_tokens))
    target_side = max(downsample_factor, int(round(math.sqrt(target_tokens))) * downsample_factor)

    if aspect >= 1.0:
        width_patches = target_side
        height_patches = max(downsample_factor, round(width_patches / aspect / downsample_factor) * downsample_factor)
    else:
        height_patches = target_side
        width_patches = max(downsample_factor, round(height_patches * aspect / downsample_factor) * downsample_factor)

    # Clamp projected token count into [min, max]
    proj_tokens = (height_patches // downsample_factor) * (width_patches // downsample_factor)
    if proj_tokens > max_image_tokens:
        scale = math.sqrt(max_image_tokens / proj_tokens)
        height_patches = max(downsample_factor, int(height_patches * scale) // downsample_factor * downsample_factor)
        width_patches = max(downsample_factor, int(width_patches * scale) // downsample_factor * downsample_factor)
    elif proj_tokens < min_image_tokens:
        scale = math.sqrt(min_image_tokens / max(proj_tokens, 1))
        height_patches = max(downsample_factor, int(math.ceil(height_patches * scale / downsample_factor)) * downsample_factor)
        width_patches = max(downsample_factor, int(math.ceil(width_patches * scale / downsample_factor)) * downsample_factor)

    new_w = width_patches * patch_size
    new_h = height_patches * patch_size
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    arr = (arr - mean) / std

    patches = arr.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # [C, H_p, W_p, P, P] -> [H_p * W_p, C * P * P]
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    patches = patches.view(height_patches * width_patches, 3 * patch_size * patch_size)
    pixel_values = patches.unsqueeze(0)
    return pixel_values, (height_patches, width_patches)


def build_chat_input_ids(
    prompt: str,
    num_image_tokens: int,
    *,
    image_token_id: int = 396,
    vocab_size: int = 65536,
    tokenizer=None,
) -> torch.Tensor:
    """Build ``input_ids`` with HF-style image placeholders for a single image.

    When a HF tokenizer is available it is preferred; otherwise a compact mock
    sequence is used that still places the correct number of ``image_token_id``s.
    """
    if tokenizer is not None:
        # Liquid chat template uses special image delimiters; placeholders are expanded
        # by the processor. Here we insert the expanded image token run manually.
        text = f"<|image_start|>{'<image>' * num_image_tokens}<|image_end|>\n{prompt}"
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        ids = encoded["input_ids"]
        # Force image placeholders to the configured id when tokenizer maps <image>
        if hasattr(tokenizer, "image_token_id") and tokenizer.image_token_id is not None:
            ids = ids.clone()
            ids[ids == tokenizer.image_token_id] = image_token_id
        return ids

    # Mock tokenization: BOS, image block, prompt filler, EOS-like pad
    bos = torch.tensor([[1]], dtype=torch.long)
    image_block = torch.full((1, num_image_tokens), image_token_id, dtype=torch.long)
    # Simple deterministic text tokens from prompt chars
    text_ids = torch.tensor([[(ord(c) % (vocab_size - 100)) + 10 for c in prompt[:64]]], dtype=torch.long)
    if text_ids.numel() == 0:
        text_ids = torch.tensor([[42]], dtype=torch.long)
    return torch.cat([bos, image_block, text_ids], dim=1)


def create_rotary_cache(device, config: Dict, max_seq_len: int = 4096):
    head_dim = config["head_dim"]
    base = config.get("rope_theta", 1_000_000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)[..., :head_dim]
    cos = ttnn.from_torch(
        emb.cos()[None, None, :, :],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    sin = ttnn.from_torch(
        emb.sin()[None, None, :, :],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    return cos, sin


def _tt_param(tensor: torch.Tensor, device):
    layout = ttnn.TILE_LAYOUT
    if tensor.ndim == 1 or tensor.shape[-1] % 32 != 0 or (tensor.ndim >= 2 and tensor.shape[-2] % 32 != 0):
        layout = ttnn.ROW_MAJOR_LAYOUT
    return ttnn.from_torch(tensor, device=device, dtype=ttnn.bfloat16, layout=layout)


def _linear_mock(out_f, in_f, device, bias=False):
    obj = type("L", (), {})()
    # Store as [in, out] for ttnn.linear
    obj.weight = _tt_param(torch.randn(in_f, out_f) * 0.02, device)
    if bias:
        obj.bias = _tt_param(torch.zeros(out_f), device)
    return obj


def _norm_mock(dim, device, bias=False):
    obj = type("N", (), {})()
    obj.weight = _tt_param(torch.ones(dim), device)
    if bias:
        obj.bias = _tt_param(torch.zeros(dim), device)
    return obj


def create_mock_parameters(config: Dict, device):
    """Structured mock weights matching convert_lfm2_weights output."""
    vcfg = config["vision_config"]
    hs = config["hidden_size"]
    vhs = vcfg["hidden_size"]
    patch_dim = vcfg["num_channels"] * vcfg["patch_size"] * vcfg["patch_size"]
    proj_in = vhs * (config["downsample_factor"] ** 2)

    vision_layers = []
    for _ in range(vcfg["num_hidden_layers"]):
        vision_layers.append(
            type(
                "VL",
                (),
                {
                    "layer_norm1": _norm_mock(vhs, device, bias=True),
                    "layer_norm2": _norm_mock(vhs, device, bias=True),
                    "self_attn": type(
                        "A",
                        (),
                        {
                            "q_proj": _linear_mock(vhs, vhs, device, bias=True),
                            "k_proj": _linear_mock(vhs, vhs, device, bias=True),
                            "v_proj": _linear_mock(vhs, vhs, device, bias=True),
                            "out_proj": _linear_mock(vhs, vhs, device, bias=True),
                        },
                    )(),
                    "mlp": type(
                        "M",
                        (),
                        {
                            "fc1": _linear_mock(vcfg["intermediate_size"], vhs, device, bias=True),
                            "fc2": _linear_mock(vhs, vcfg["intermediate_size"], device, bias=True),
                        },
                    )(),
                },
            )()
        )

    layers = []
    for layer_type in config["layer_types"]:
        base = {
            "operator_norm": _norm_mock(hs, device),
            "ffn_norm": _norm_mock(hs, device),
            "feed_forward": type(
                "FF",
                (),
                {
                    "w1": _linear_mock(config["intermediate_size"], hs, device),
                    "w2": _linear_mock(hs, config["intermediate_size"], device),
                    "w3": _linear_mock(config["intermediate_size"], hs, device),
                },
            )(),
        }
        if layer_type == "full_attention":
            base["self_attn"] = type(
                "SA",
                (),
                {
                    "q_proj": _linear_mock(config["num_heads"] * config["head_dim"], hs, device),
                    "k_proj": _linear_mock(config["num_key_value_heads"] * config["head_dim"], hs, device),
                    "v_proj": _linear_mock(config["num_key_value_heads"] * config["head_dim"], hs, device),
                    "out_proj": _linear_mock(hs, config["num_heads"] * config["head_dim"], device),
                    "q_layernorm": _norm_mock(config["head_dim"], device),
                    "k_layernorm": _norm_mock(config["head_dim"], device),
                },
            )()
        else:
            base["conv"] = type(
                "C",
                (),
                {
                    "in_proj": _linear_mock(3 * hs, hs, device),
                    "conv": type("CW", (), {"weight": _tt_param(torch.randn(hs, 1, 3) * 0.02, device)})(),
                    "out_proj": _linear_mock(hs, hs, device),
                },
            )()
        layers.append(type("Layer", (), base)())

    return type(
        "Params",
        (),
        {
            "embed_tokens": type("E", (), {"weight": _tt_param(torch.randn(config["vocab_size"], hs) * 0.02, device)})(),
            "embedding_norm": _norm_mock(hs, device),
            "vision_tower": type(
                "VT",
                (),
                {
                    "patch_embedding": _linear_mock(vhs, patch_dim, device, bias=True),
                    "position_embedding": type(
                        "P", (), {"weight": _tt_param(torch.randn(1, vcfg["num_patches"], vhs) * 0.02, device)}
                    )(),
                    "layers": vision_layers,
                    "post_layernorm": _norm_mock(vhs, device, bias=True),
                },
            )(),
            "multi_modal_projector": type(
                "P",
                (),
                {
                    "linear_1": _linear_mock(config["projector_hidden_size"], proj_in, device, bias=True),
                    "linear_2": _linear_mock(hs, config["projector_hidden_size"], device, bias=True),
                },
            )(),
            "layers": layers,
            "lm_head": type("H", (), {"weight": _tt_param(torch.randn(hs, config["vocab_size"]) * 0.02, device)})(),
        },
    )()


def run_lfm2_vl_demo(
    image_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    prompt: str = "Describe this image for OCR/document comprehension.",
    use_mock: bool = False,
    device_id: int = 0,
):
    """Run LFM2.5-VL forward pass.

    Args:
        image_path: Optional path to a real image. When omitted, a synthetic
            variable-resolution patch grid is used.
        weights_path: Path to HF ``model.safetensors`` (or directory). When omitted,
            structured mock weights are used.
        prompt: User text prompt (OCR / caption / multi-image style tasks).
        use_mock: Force mock weights even if ``weights_path`` is set.
        device_id: TT device id.
    """
    device = ttnn.open_device(device_id=device_id)
    config = create_model_config()

    try:
        if weights_path and not use_mock:
            from models.demos.lfm2_vl.tt.convert_weights import convert_lfm2_weights

            parameters = convert_lfm2_weights(weights_path, device, config)
            print(f"Loaded weights from {weights_path}")
        else:
            parameters = create_mock_parameters(config, device)
            print("Using structured mock parameters")

        model = TtLfm2VlForConditionalGeneration(device, config, parameters)
        cos, sin = create_rotary_cache(device, config)
        model.set_rope_cache((cos, sin))

        if image_path:
            pixel_values, spatial_shapes = preprocess_liquid_image(
                image_path,
                patch_size=config["vision_config"]["patch_size"],
                min_image_tokens=config["min_image_tokens"],
                max_image_tokens=config["max_image_tokens"],
                downsample_factor=config["downsample_factor"],
            )
            print(f"Image: {image_path} spatial_shapes={spatial_shapes} patches={pixel_values.shape[1]}")
        else:
            # Synthetic 16x16 patch grid (256 patches -> 64 projector tokens with factor=2)
            h_p = w_p = 16
            patch_dim = 3 * config["vision_config"]["patch_size"] ** 2
            pixel_values = torch.randn(1, h_p * w_p, patch_dim)
            spatial_shapes = (h_p, w_p)
            print(f"Synthetic image grid {spatial_shapes}")

        # Projector reduces tokens by downsample_factor**2
        num_image_tokens = (spatial_shapes[0] // config["downsample_factor"]) * (
            spatial_shapes[1] // config["downsample_factor"]
        )
        input_ids = build_chat_input_ids(
            prompt,
            num_image_tokens,
            image_token_id=config["image_token_id"],
            vocab_size=config["vocab_size"],
        )
        n_placeholders = int((input_ids == config["image_token_id"]).sum().item())
        assert n_placeholders == num_image_tokens, (
            f"placeholder count {n_placeholders} != image features {num_image_tokens}"
        )

        tt_pixels = ttnn.from_torch(
            pixel_values,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
            if pixel_values.shape[-1] % 32 == 0 and pixel_values.shape[-2] % 32 == 0
            else ttnn.ROW_MAJOR_LAYOUT,
        )
        # Ensure tile layout for matmul when dims allow; otherwise model path tilizes after embed
        if pixel_values.shape[-1] % 32 == 0:
            tt_pixels = ttnn.to_layout(tt_pixels, ttnn.TILE_LAYOUT)

        tt_ids = ttnn.from_torch(input_ids, device=device, dtype=ttnn.uint32)

        print(f"Running {HF_MODEL_ID} on device {device_id}...")
        print(f"  prompt: {prompt!r}")
        print(f"  seq_len={input_ids.shape[1]} image_tokens={num_image_tokens}")
        logits = model(
            pixel_values=tt_pixels,
            input_ids=tt_ids,
            spatial_shapes=spatial_shapes,
        )
        logits_torch = ttnn.to_torch(logits)
        print(f"Inference complete. Logits shape: {tuple(logits_torch.shape)}")
        return logits_torch
    finally:
        ttnn.close_device(device)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="LFM2.5-VL-1.6B demo (LiquidAI)")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--weights", type=str, default=None, help="Path to HF weights (.safetensors or dir)")
    parser.add_argument("--prompt", type=str, default="What text is visible in this document?")
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use mock weights (default: true when --weights is omitted)",
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)

    use_mock = args.mock if args.mock is not None else args.weights is None
    run_lfm2_vl_demo(
        image_path=args.image,
        weights_path=args.weights,
        prompt=args.prompt,
        use_mock=use_mock,
        device_id=args.device,
    )


if __name__ == "__main__":
    main()
