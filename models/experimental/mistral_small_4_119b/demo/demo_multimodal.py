# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Small-4-119B multimodal generation demo.

Drives the full vision + projector + language model pipeline end to end. Loading
is unified: vision + projector + text are lazy-loaded together on the first
``encode_image`` call, then ``prefill_multimodal`` runs, then the decode loop.

Supported hardware: P150x8 or a Blackhole Loud Box only. Set ``MESH_DEVICE``
accordingly before running.

Defaults run a real image (a sample battle scene) and prompt at 2 text + 2
vision layers — fast (~minutes) but for plumbing only, so the output is
gibberish. Pass ``--n-text-layers 36 --n-vision-layers 24`` for the full model.

The input image is fixed to the built-in ``DEFAULT_IMAGE_URL`` sample (a battle
scene); it is not user-supplied. Only the prompt is configurable::

    export MESH_DEVICE=P150x8
    python models/experimental/mistral_small_4_119b/demo_multimodal.py

    # Change the prompt:
    python models/experimental/mistral_small_4_119b/demo_multimodal.py \
        --prompt "Describe the scene."

Pass ``--random-image`` to use a random pixel_values tensor instead of the
sample image — the pipeline still runs end to end, but the model can't see
anything real.
"""

from __future__ import annotations

import argparse
import copy
import gc
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
from models.experimental.mistral_small_4_119b.tt.mistral3_for_conditional_generation_unified import (
    TtMistral3ForConditionalGenerationUnified,
)
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_IMAGE_URL = (
    "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/" "revision/latest?cb=20220523172438"
)
DEFAULT_PROMPT = (
    "What action do you think I should take in this situation? List all the "
    "possible actions and explain why you think they are good or bad."
)

# ── Argument parsing ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mistral-Small-4-119B multimodal demo")
    p.add_argument(
        "--random-image",
        action="store_true",
        help="Skip the real sample image and use a random pixel_values tensor (plumbing only — "
        "the pipeline runs end to end but the model can't see anything real).",
    )
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt accompanying the image")
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
        "Ignored with --random-image. Larger = better understanding, slower vision forward.",
    )
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Skip HF chat template / processor (use raw [image]+[prompt] layout). "
        "Output will likely be EOS-only — useful only for plumbing tests.",
    )
    p.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable decode-step trace capture. Every decode step stays on the eager dispatch path "
        "(much slower steady-state — useful for debugging or comparing trace vs eager latency).",
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help="Run one untimed warmup pass (prefill + a few decode steps + trace capture) before the "
        "measured generation, so reported TTFT / decode tok/s reflect steady state instead of "
        "including one-time compile + trace-capture cost on the first run.",
    )
    p.add_argument(
        "--backend",
        choices=("ttnn", "hf", "both"),
        default="ttnn",
        help="Generation backend to run. Use 'both' to compare TTNN output against HF Torch output.",
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


def _hf_param_to_sd_key(name: str) -> str:
    if name.startswith("model.vision_tower."):
        return name[len("model.") :]
    if name.startswith("model.multi_modal_projector."):
        return name[len("model.") :]
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model.") :]
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    return name


def _build_hf_mm_ref(full_config, state_dict: dict, n_text: int, n_vision: int):
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

    cfg = copy.deepcopy(full_config)
    cfg.text_config.num_hidden_layers = n_text
    cfg.vision_config.num_hidden_layers = n_vision
    for sub in (cfg.text_config, cfg.vision_config):
        for attr in ("attn_implementation", "_attn_implementation"):
            if hasattr(sub, attr):
                setattr(sub, attr, "eager")

    if n_text >= EXPECTED_NUM_LAYERS and n_vision >= VISION_NUM_LAYERS:
        logger.info("Enabling activation checkpointing for full-layer HF model (36+24)…")
        cfg.text_config.gradient_checkpointing = True
        cfg.vision_config.gradient_checkpointing = True

    with init_empty_weights():
        model = Mistral3ForConditionalGeneration(cfg)

    missing = []
    for name, _ in model.named_parameters():
        sd_key = _hf_param_to_sd_key(name)
        if sd_key not in state_dict:
            missing.append(name)
            continue
        v = state_dict[sd_key]
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale_inv = state_dict.get(sd_key + "_scale_inv")
            if scale_inv is None:
                scale_inv = state_dict.get(sd_key.replace(".weight", ".weight_scale_inv"))
            v_cast = v.to(torch.float32)
            if scale_inv is not None:
                s = scale_inv.to(torch.float32)
                while s.dim() < v_cast.dim():
                    s = s.unsqueeze(-1)
                v_cast = v_cast * s
            tensor = v_cast.to(torch.bfloat16)
            del v_cast
        else:
            tensor = v.to(torch.bfloat16)
        set_module_tensor_to_device(model, name, "cpu", value=tensor)
        del tensor

    if missing:
        logger.warning(f"HF model missing keys (first 5): {missing[:5]}")
    return model.eval()


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
        trace_region_size=128 * 1024 * 1024,  # 36-layer decode trace measures ~66 MB
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
    Build pixel_values + input_ids via HF ``AutoProcessor.apply_chat_template``,
    which applies the chat-template framing, image normalization/resize, and the
    ``image_token_id`` slots the model was trained on.

    ``image_path`` may be a local file or an http(s) URL.

    Returns ``(pixel_values [1,3,H,W] bf16, input_ids [1, seq_len] long, processor)``.
    """
    from PIL import Image
    from transformers import AutoProcessor

    if image_path.startswith(("http://", "https://")):
        import io
        import urllib.request

        logger.info(f"Fetching image from URL: {image_path}")
        with urllib.request.urlopen(image_path) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    else:
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
    logger.warning(f"Using random {side}×{side} pixel_values + raw [IMG]+text layout")
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


# ── Generation loop ────────────────────────────────────────────────────────────


def generate(
    model: TtMistral3ForConditionalGenerationUnified,
    tokenizer,
    rotary_cls,
    text_config,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    prompt: str,
    image_token_id: int,
    max_new_tokens: int,
    use_trace: bool = True,
    warmup: bool = False,
) -> str:
    seq_len = input_ids.shape[-1]
    num_image_tokens = int((input_ids[0] == image_token_id).sum().item())
    logger.info(
        f"Prompt: {prompt!r} → {seq_len} tokens total "
        f"({num_image_tokens} image-token slots + {seq_len - num_image_tokens} text/template tokens)"
    )

    # Precompute RoPE for prefill + decode steps once on host. Uploaded to
    # device once after load_text; per-step lookups are an on-device slice.
    total_positions = seq_len + max_new_tokens
    cos_full, sin_full = _precompute_rope_table(rotary_cls, text_config, total_positions)

    logger.info(
        "Unified load — encode_image triggers one-time load of "
        f"vision_layers={model.num_vision_layers} + text_layers={model.num_text_layers} "
        "together (vision_dtype=bfloat8_b), then runs vision → host img_embeds…"
    )
    t0 = time.perf_counter()
    img_embeds_host = model.encode_image(pixel_values)
    logger.info(
        f"encode_image done in {time.perf_counter() - t0:.1f}s "
        f"→ image embeddings {tuple(img_embeds_host.shape)} bf16 on host"
    )
    assert img_embeds_host.shape[0] == num_image_tokens, (
        f"projector produced {img_embeds_host.shape[0]} image tokens but input_ids has "
        f"{num_image_tokens} image_token_id={image_token_id} slots — check that the HF "
        f"processor agrees with the image dimensions you fed encode_image."
    )

    logger.info("Unified load_text — idempotent/no reload; text is already resident. Caching RoPE tables…")
    t0 = time.perf_counter()
    model.load_text()
    model.cache_rope_tables(cos_full, sin_full)
    logger.info(f"load_text/cache_rope done in {time.perf_counter() - t0:.1f}s")

    # Optional warmup: run prefill + a couple decode steps + trace capture once, untimed, so the measured pass below hits the program cache / replays thedecode trace from token 1.
    # Without this the first run's TTFT and decode tok/s include the one-time compile + trace-capture cost.
    if warmup:
        logger.info("Warmup pass (untimed): prefill + decode compile + trace capture…")
        t0 = time.perf_counter()
        warm_next = model.prefill_multimodal(img_embeds_host, input_ids)
        warm_cur = torch.tensor([[warm_next]], dtype=torch.long)
        # Two decode steps: step 1 compiles the decode kernels eagerly; the second confirms a cache hit.
        # Capture the trace after the eager step so the measured loop replays it.
        for warm_step in range(2):
            warm_next = model.decode_next_token(warm_cur, seq_len + warm_step)
            if warm_step == 0 and use_trace:
                model.capture_decode_trace()
            warm_cur = torch.tensor([[warm_next]], dtype=torch.long)
        logger.info(f"Warmup done in {time.perf_counter() - t0:.1f}s — measured pass is now steady-state")

    logger.info(f"Inference — prefill_multimodal (seq_len={seq_len}) then decode loop…")
    t0 = time.perf_counter()
    next_id = model.prefill_multimodal(img_embeds_host, input_ids)
    logger.info(f"Prefill done in {(time.perf_counter() - t0) * 1000:.0f} ms")

    generated_ids = [next_id]
    print(f"\n[{prompt}]\n→ ", end="", flush=True)
    print(tokenizer.decode([next_id], skip_special_tokens=True), end="", flush=True)

    decode_times = []
    cur = torch.tensor([[next_id]], dtype=torch.long)
    for step in range(1, max_new_tokens):
        current_pos = seq_len + step - 1
        t_dec = time.perf_counter()
        tok_id = model.decode_next_token(cur, current_pos)
        decode_times.append((time.perf_counter() - t_dec) * 1000)

        # Step 1 compiles the decode kernels eagerly; capture the trace right after so steps 2+ replay it instead of re-dispatching from host.
        if step == 1 and use_trace:
            model.capture_decode_trace()

        generated_ids.append(tok_id)
        print(tokenizer.decode([tok_id], skip_special_tokens=True), end="", flush=True)
        if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id:
            break
        cur = torch.tensor([[tok_id]], dtype=torch.long)
    print()

    if decode_times:
        avg = sum(decode_times) / len(decode_times)
        logger.info(f"Generated {len(generated_ids)} tokens | decode avg {avg:.0f} ms/tok ({1000/avg:.1f} tok/s)")

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # Log the full decoded text as one line so it is visible in non-TTY logs (CI);
    # the streamed print() above renders only on an interactive terminal.
    logger.info(f"Generated: {full_text!r}")
    return full_text


def generate_hf(
    full_config,
    state_dict: dict,
    tokenizer,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    prompt: str,
    max_new_tokens: int,
    n_text_layers: int,
    n_vision_layers: int,
) -> str:
    logger.info(
        f"Building HF Torch model (text_layers={n_text_layers}, vision_layers={n_vision_layers}, dtype=bf16 CPU)…"
    )
    t0 = time.perf_counter()
    model = _build_hf_mm_ref(full_config, state_dict, n_text_layers, n_vision_layers)
    logger.info(f"HF model built in {time.perf_counter() - t0:.1f}s")

    image_sizes = torch.tensor([[pixel_values.shape[-2], pixel_values.shape[-1]]], dtype=torch.long)
    generated_ids = []
    cur_input_ids = input_ids

    print(f"\n[HF Torch | {prompt}]\n→ ", end="", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(pixel_values=pixel_values, input_ids=cur_input_ids, image_sizes=image_sizes)
            next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
            generated_ids.append(next_id)
            print(tokenizer.decode([next_id], skip_special_tokens=True), end="", flush=True)
            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break
            cur_input_ids = torch.cat([cur_input_ids, torch.tensor([[next_id]], dtype=torch.long)], dim=-1)
    print()

    elapsed = time.perf_counter() - t0
    if generated_ids:
        logger.info(f"HF generated {len(generated_ids)} tokens | avg {elapsed * 1000 / len(generated_ids):.0f} ms/tok")
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    del model
    gc.collect()
    return text


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
    #  - default: fetch the built-in DEFAULT_IMAGE_URL sample, then HF chat template
    #    + image processor (the trained format).
    #  - --random-image or --no-chat-template: random/raw fallback (plumbing only).
    if not args.random_image and not args.no_chat_template:
        logger.info(f"Building chat-template inputs via HF AutoProcessor for {DEFAULT_IMAGE_URL!r}…")
        pixel_values, input_ids, _ = _build_chat_inputs(DEFAULT_IMAGE_URL, args.prompt, args.image_max_side)
    else:
        if args.no_chat_template:
            logger.warning("--no-chat-template set: skipping HF chat template, expect EOS-only output.")
        pixel_values, input_ids = _build_random_inputs(args.img_patches, args.prompt, tokenizer, image_token_id)
    logger.info(f"pixel_values: {tuple(pixel_values.shape)} bf16, input_ids: {tuple(input_ids.shape)} long")

    tt_text = None
    hf_text = None

    if args.backend in ("ttnn", "both"):
        mesh_device = _open_mesh_device()
        try:
            # KV cache must cover the prefill prompt + every decode token.
            max_seq_len = input_ids.shape[-1] + args.max_new_tokens + 16

            logger.info(
                f"Building unified orchestrator (text_layers={args.n_text_layers}, "
                f"vision_layers={args.n_vision_layers}, max_seq_len={max_seq_len}, vision_dtype=bfloat8_b)…"
            )
            logger.info(
                "Unified mode keeps vision + projector + text resident together; "
                "the first encode_image() call performs the one-time combined load."
            )
            model = TtMistral3ForConditionalGenerationUnified(
                mesh_device=mesh_device,
                state_dict=state_dict,
                text_config=text_cfg,
                image_token_id=image_token_id,
                num_text_layers=args.n_text_layers,
                num_vision_layers=args.n_vision_layers,
                max_seq_len=max_seq_len,
                vision_dtype=ttnn.bfloat8_b,
            )

            with torch.no_grad():
                tt_text = generate(
                    model=model,
                    tokenizer=tokenizer,
                    rotary_cls=Mistral4RotaryEmbedding,
                    text_config=text_cfg,
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    prompt=args.prompt,
                    image_token_id=image_token_id,
                    max_new_tokens=args.max_new_tokens,
                    use_trace=not args.no_trace,
                    warmup=args.warmup,
                )
            del model
        finally:
            ttnn.close_mesh_device(mesh_device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        gc.collect()

    if args.backend in ("hf", "both"):
        hf_text = generate_hf(
            full_config=cfg,
            state_dict=state_dict,
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            input_ids=input_ids,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            n_text_layers=args.n_text_layers,
            n_vision_layers=args.n_vision_layers,
        )

    if args.backend == "both":
        logger.info("=" * 80)
        logger.info("Generation comparison")
        logger.info(f"TTNN: {tt_text!r}")
        logger.info(f"HF  : {hf_text!r}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
