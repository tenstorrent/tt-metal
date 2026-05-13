# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Small-4-119B multimodal generation demo (phase-based).

Drives the full vision + projector + language model pipeline end to end:

    1. Build the orchestrator (lightweight — config only).
    2. Phase 1 — ``encode_image``: load vision tower + projector, run forward
                  on pixel_values, pull the small image embeddings to host,
                  free vision/projector weights from DRAM.
    3. Phase 2 — ``load_text``: build the text language model on device.
    4. Phase 3 — ``prefill_multimodal`` then a decode loop.

The default 2 text + 2 vision layers makes this fast (~minutes total) and is
intended for plumbing validation — the output will be gibberish. Bump to
``--n-text-layers 36 --n-vision-layers 24`` for the full model.

Run::

    export MESH_DEVICE=T3K          # or P150x4, single, etc.
    python models/experimental/mistral_small_4_119b/demo_multimodal.py \
        --prompt "What's in this picture?" \
        --max-new-tokens 16

    # With a real image file:
    python models/experimental/mistral_small_4_119b/demo_multimodal.py \
        --image /path/to/image.jpg \
        --prompt "Describe the scene."

If ``--image`` is omitted, the demo generates a random pixel_values tensor so
the pipeline still exercises end-to-end (the model can't see anything real,
so don't expect a coherent answer — and at the default 2 layers, you won't
get one even with a real image).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from loguru import logger

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_NUM_LAYERS,
    HF_MODEL_ID,
    MMP_SPATIAL_MERGE_SIZE,
    VISION_NUM_LAYERS,
    VISION_PATCH_SIZE,
    text_decoder_layer_state_dict_prefix,
    vision_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral3_for_conditional_generation import (
    TtMistral3ForConditionalGeneration,
)
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


# ── Argument parsing ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mistral-Small-4-119B multimodal demo")
    p.add_argument("--image", type=str, default=None, help="Image file path (uses a random image if omitted)")
    p.add_argument("--prompt", type=str, default="Describe this image.", help="Text prompt accompanying the image")
    p.add_argument("--max-new-tokens", type=int, default=16, help="Tokens to generate after prefill")
    p.add_argument(
        "--n-text-layers",
        type=int,
        default=2,
        help=f"Text decoder layers (1..{EXPECTED_NUM_LAYERS}); default 2 for fast plumbing test",
    )
    p.add_argument(
        "--n-vision-layers",
        type=int,
        default=2,
        help=f"Pixtral vision layers (1..{VISION_NUM_LAYERS}); default 2 for fast plumbing test",
    )
    p.add_argument(
        "--img-patches",
        type=int,
        default=10,
        help="(Only used in random-image fallback) patches per side. Image is sized to img_patches × 14 pixels.",
    )
    p.add_argument(
        "--image-max-side",
        type=int,
        default=224,
        help="Max H/W of the input image, in pixels, before the HF processor sees it. "
        "Used only with --image. Larger = better understanding, slower vision forward.",
    )
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Skip HF chat template / processor (use raw [image]+[prompt] layout). "
        "Output will likely be EOS-only — useful only for plumbing tests.",
    )
    return p.parse_args()


# ── State-dict prefixes ────────────────────────────────────────────────────────


def _state_dict_prefixes(n_text: int, n_vision: int) -> tuple:
    p = ["language_model.model.embed_tokens."]
    for i in range(n_text):
        p.append(text_decoder_layer_state_dict_prefix(i))
    p.append("language_model.model.norm.")
    p.append("language_model.lm_head.")
    p.append("vision_tower.patch_conv.")
    p.append("vision_tower.ln_pre.")
    for i in range(n_vision):
        p.append(vision_layer_state_dict_prefix(i))
    p.append("multi_modal_projector.")
    return tuple(p)


# ── Mesh device open/close ─────────────────────────────────────────────────────


def _open_mesh_device() -> ttnn.MeshDevice:
    rows, cols = mesh_device_request_param()
    if (rows, cols) != (1, 1):
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
        )
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(rows, cols),
        dispatch_core_config=ttnn.DispatchCoreConfig(),
        trace_region_size=30_000_000,
        num_command_queues=1,
    )
    logger.info(f"Opened {rows}×{cols} mesh device ({device.get_num_devices()} chips)")
    return device


# ── Input building: HF chat template + image processor ────────────────────────


def _build_chat_inputs(
    image_path: str,
    prompt: str,
    image_max_side: int,
):
    """
    Build pixel_values + input_ids using HF ``AutoProcessor.apply_chat_template``.

    The processor handles everything the model was trained to expect:
      - chat-template framing: ``<s>[INST][IMG]…[IMG_BREAK]…[IMG_END] prompt [/INST]``
      - CLIP-style image normalization, resize, and patch-aligned dimensions
      - inserting the correct number of ``image_token_id`` slots into ``input_ids``
        (with ``[IMG_BREAK]`` between patch rows — the orchestrator's
        ``contiguous_runs`` helper handles those gaps)
      - ``add_generation_prompt=True`` so the model knows it's the assistant's turn

    Returns ``(pixel_values [1,3,H,W] bf16, input_ids [1, seq_len] long, processor)``.
    """
    from PIL import Image
    from transformers import AutoProcessor

    img = Image.open(image_path).convert("RGB")
    if max(img.size) > image_max_side:
        scale = image_max_side / max(img.size)
        new_w = max(VISION_PATCH_SIZE, int(round(img.size[0] * scale)))
        new_h = max(VISION_PATCH_SIZE, int(round(img.size[1] * scale)))
        img = img.resize((new_w, new_h))
        logger.info(f"Resized input image to {img.size} (max side ≤ {image_max_side})")

    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    input_ids = inputs["input_ids"]
    return pixel_values, input_ids, processor


def _build_random_inputs(img_patches: int, prompt: str, tokenizer, image_token_id: int):
    """
    Random-image fallback for plumbing tests (no chat template).

    ``input_ids`` = ``[image_token_id × N] + [prompt tokens]`` — the model will
    almost certainly emit EOS right away since this isn't the trained format.
    """
    side = img_patches * VISION_PATCH_SIZE
    logger.warning(f"No --image provided; using random {side}×{side} pixel_values + raw [IMG]+text layout")
    pixel_values = torch.rand(1, 3, side, side, dtype=torch.bfloat16) * 2 - 1
    num_image_tokens = (img_patches // MMP_SPATIAL_MERGE_SIZE) ** 2
    text_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    img_ids = torch.full((num_image_tokens,), image_token_id, dtype=torch.long)
    input_ids = torch.cat([img_ids, text_ids]).unsqueeze(0)
    return pixel_values, input_ids


# ── RoPE precompute (same trick as the text demo) ──────────────────────────────


def _precompute_rope_table(rotary_cls, text_config, max_seq_len: int):
    rotary = rotary_cls(text_config).eval().to(torch.bfloat16)
    dummy = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    pos_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
    return rotary(dummy, pos_ids)


def _slice_rope(cos_full, sin_full, start, end):
    if cos_full.dim() == 3:
        return cos_full[:, start:end, :].contiguous(), sin_full[:, start:end, :].contiguous()
    return cos_full[:, :, start:end, :].contiguous(), sin_full[:, :, start:end, :].contiguous()


# ── Generation loop ────────────────────────────────────────────────────────────


def generate(
    model: TtMistral3ForConditionalGeneration,
    tokenizer,
    rotary_cls,
    text_config,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    prompt: str,
    image_token_id: int,
    max_new_tokens: int,
) -> str:
    seq_len = input_ids.shape[-1]
    num_image_tokens = int((input_ids[0] == image_token_id).sum().item())
    logger.info(
        f"Prompt: {prompt!r} → {seq_len} tokens total "
        f"({num_image_tokens} image-token slots + {seq_len - num_image_tokens} text/template tokens)"
    )

    # Precompute RoPE for prefill + decode steps once on host.
    total_positions = seq_len + max_new_tokens
    cos_full, sin_full = _precompute_rope_table(rotary_cls, text_config, total_positions)

    # Phase 1: encode_image (loads vision, runs, frees, returns host img_embeds).
    logger.info("Phase 1 — encode_image (vision tower + projector → host img_embeds)…")
    t0 = time.perf_counter()
    img_embeds_host = model.encode_image(pixel_values)
    logger.info(
        f"Phase 1 done in {time.perf_counter() - t0:.1f}s "
        f"→ image embeddings {tuple(img_embeds_host.shape)} bf16 on host"
    )
    assert img_embeds_host.shape[0] == num_image_tokens, (
        f"projector produced {img_embeds_host.shape[0]} image tokens but input_ids has "
        f"{num_image_tokens} image_token_id={image_token_id} slots — check that the HF "
        f"processor agrees with the image dimensions you fed encode_image."
    )

    # Phase 2: load text model on device.
    logger.info("Phase 2 — load_text (text model construction on device)…")
    t0 = time.perf_counter()
    model.load_text()
    logger.info(f"Phase 2 done in {time.perf_counter() - t0:.1f}s")

    # Phase 3: prefill + decode loop.
    logger.info(f"Phase 3 — prefill_multimodal (seq_len={seq_len})…")
    t0 = time.perf_counter()
    next_id = model.prefill_multimodal(img_embeds_host, input_ids, _slice_rope(cos_full, sin_full, 0, seq_len))
    logger.info(f"Prefill done in {(time.perf_counter() - t0) * 1000:.0f} ms")

    generated_ids = [next_id]
    print(f"\n[{prompt}]\n→ ", end="", flush=True)
    print(tokenizer.decode([next_id], skip_special_tokens=True), end="", flush=True)

    decode_times = []
    cur = torch.tensor([[next_id]], dtype=torch.long)
    for step in range(1, max_new_tokens):
        current_pos = seq_len + step - 1
        t_dec = time.perf_counter()
        tok_id = model.decode_next_token(
            cur, _slice_rope(cos_full, sin_full, current_pos, current_pos + 1), current_pos
        )
        decode_times.append((time.perf_counter() - t_dec) * 1000)

        generated_ids.append(tok_id)
        print(tokenizer.decode([tok_id], skip_special_tokens=True), end="", flush=True)
        if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id:
            break
        cur = torch.tensor([[tok_id]], dtype=torch.long)
    print()

    if decode_times:
        avg = sum(decode_times) / len(decode_times)
        logger.info(f"Generated {len(generated_ids)} tokens | decode avg {avg:.0f} ms/tok ({1000/avg:.1f} tok/s)")

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    assert (
        args.img_patches % MMP_SPATIAL_MERGE_SIZE == 0
    ), f"--img-patches ({args.img_patches}) must be even for 2x2 spatial merge"

    try:
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding
    except ImportError as e:
        sys.exit(f"transformers with Mistral4/Pixtral support required: {e}")

    # HF config + image_token_id
    logger.info(f"Loading HF config from {HF_MODEL_ID!r}…")
    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as e:
        sys.exit(f"Failed to load HF config: {e}")
    text_cfg = cfg.text_config
    image_token_id = int(getattr(cfg, "image_token_index", 10))

    # State dict (filtered to what we actually need)
    logger.info(f"Loading state dict (text_layers={args.n_text_layers}, vision_layers={args.n_vision_layers})…")
    try:
        state_dict = load_hf_state_dict_filtered(
            HF_MODEL_ID, _state_dict_prefixes(args.n_text_layers, args.n_vision_layers)
        )
    except (FileNotFoundError, OSError) as e:
        sys.exit(f"Checkpoint load failed: {e}")

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except Exception as e:
        sys.exit(f"Tokenizer load failed: {e}")

    # Build pixel_values + input_ids.
    #  - With --image: HF chat template + image processor (the format the model was trained on).
    #  - Without --image, or with --no-chat-template: random/raw fallback (plumbing only).
    if args.image and not args.no_chat_template:
        logger.info(f"Building chat-template inputs via HF AutoProcessor for {args.image!r}…")
        pixel_values, input_ids, _ = _build_chat_inputs(args.image, args.prompt, args.image_max_side)
    else:
        if args.image and args.no_chat_template:
            logger.warning("--no-chat-template set: skipping HF chat template, expect EOS-only output.")
        pixel_values, input_ids = _build_random_inputs(args.img_patches, args.prompt, tokenizer, image_token_id)
    logger.info(f"pixel_values: {tuple(pixel_values.shape)} bf16, input_ids: {tuple(input_ids.shape)} long")
    num_image_tokens = int((input_ids[0] == image_token_id).sum().item())

    # Mesh device + orchestrator
    mesh_device = _open_mesh_device()
    try:
        # KV cache must cover the prefill prompt + every decode token.
        max_seq_len = input_ids.shape[-1] + args.max_new_tokens + 16

        logger.info(
            f"Building orchestrator (text_layers={args.n_text_layers}, "
            f"vision_layers={args.n_vision_layers}, max_seq_len={max_seq_len})…"
        )
        model = TtMistral3ForConditionalGeneration(
            mesh_device=mesh_device,
            state_dict=state_dict,
            text_config=text_cfg,
            image_token_id=image_token_id,
            num_text_layers=args.n_text_layers,
            num_vision_layers=args.n_vision_layers,
            max_seq_len=max_seq_len,
        )

        with torch.no_grad():
            generate(
                model=model,
                tokenizer=tokenizer,
                rotary_cls=Mistral4RotaryEmbedding,
                text_config=text_cfg,
                pixel_values=pixel_values,
                input_ids=input_ids,
                prompt=args.prompt,
                image_token_id=image_token_id,
                max_new_tokens=args.max_new_tokens,
            )
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
