# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Dots OCR demo (CLI).

Two backends, selected with ``--backend``:

- ``hf``:   HuggingFace reference path (torch). Always produces full generations. Useful for
            validating output text and for comparing against the TT path.
- ``ttnn``: Builds the qwen25_vl-style Dots TT stack (``DotsTransformer`` +
            ``DropInVisionTransformer``) and runs prefill + decode via
            :class:`models.tt_transformers.tt.generator.Generator`. This mirrors
            ``models/demos/qwen25_vl/demo/demo.py`` but with the simpler single-user /
            single-image loop Dots needs. Requires ``MESH_DEVICE`` to be set.

Real checkpoint weights are loaded through ``DotsModelArgs.load_real_state_dict`` when
available; the loader transparently falls back to the dummy state_dict
(``model_args.load_state_dict()``) so the demo stays runnable for bring-up.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from loguru import logger
from PIL import Image

from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec, get_hf_model_id
from models.demos.dots_ocr.reference.model import DotsOCRReference

# ---------------------------------------------------------------------------
# HF backend (reference generation)
# ---------------------------------------------------------------------------


def run_hf_backend(model_id: str, image_path: str, prompt: str, max_new_tokens: int) -> str:
    logger.info(f"[HF] Loading {model_id}")
    ref = DotsOCRReference(HFLoadSpec(model_id=model_id))
    image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None
    inputs = ref.preprocess_image_and_prompt(image, prompt)

    t0 = time.perf_counter()
    out = ref.forward(inputs, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - t0

    generated_ids = out["generated_ids"]
    text = ref.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("\n=== HF ===")
    print(f"Backend:   hf")
    print(f"Latency:   {elapsed:.2f}s for up to {max_new_tokens} tokens")
    print(f"Output:    {text}")
    return text


# ---------------------------------------------------------------------------
# TTNN backend — mirrors qwen25_vl/demo/demo.py end-to-end flow
# ---------------------------------------------------------------------------


def _load_state_dict_with_real_weights_fallback(model_args, dummy_only: bool):
    """Try real HF weights first; fall back to dummy on any failure so bring-up stays runnable."""
    if dummy_only:
        logger.info("[TTNN] Using dummy state_dict (--dummy-weights).")
        return model_args.load_state_dict()
    try:
        logger.info("[TTNN] Attempting to load real Dots weights via DotsModelArgs.load_real_state_dict()")
        real_sd = model_args.load_real_state_dict()
        logger.info(f"[TTNN] Loaded real Dots state_dict with {len(real_sd)} tensors.")
        return real_sd
    except Exception as exc:
        logger.warning(f"[TTNN] Real weight load failed ({exc!r}); falling back to dummy state_dict.")
        return model_args.load_state_dict()


def _decode_loop(
    *,
    generator,
    tokenizer,
    ref_model,
    prefilled_logits: torch.Tensor,
    decoding_pos: list[int],
    max_new_tokens: int,
    stop_at_eos: bool = True,
) -> list[list[int]]:
    """
    Minimal qwen25_vl-style decode loop.

    Host-side argmax sampling, deallocates nothing fancy, one user per batch (Dots OCR is
    typically batch=1). Returns per-user generated token id lists.
    """
    batch_size = prefilled_logits.shape[0]
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)], dtype=torch.int32)

    # argmax over vocab dim on the final prefill token
    prefilled_token = torch.argmax(prefilled_logits, dim=-1)  # [B, 1]
    out_tok = prefilled_token
    all_outputs: list[list[int]] = [[int(prefilled_token[b].item())] for b in range(batch_size)]

    # Build a conservative stop-token set; HF tokenizers may not expose ``stop_tokens``.
    stop_ids = set()
    tok_stop = getattr(tokenizer, "stop_tokens", None)
    if tok_stop:
        stop_ids.update(int(t) for t in tok_stop)
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    user_done = [False] * batch_size
    logger.info(f"[TTNN] Starting decode loop (max_new_tokens={max_new_tokens}, stop_at_eos={stop_at_eos})")

    for iteration in range(max_new_tokens):
        t0 = time.perf_counter()
        logits, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=False,
            page_table=None,
            kv_cache=None,
            sampling_params=None,
        )
        out_tok = torch.argmax(logits, dim=-1)
        if out_tok.dim() == 1:
            out_tok = out_tok.unsqueeze(-1)
        elapsed = time.perf_counter() - t0

        current_pos += 1
        any_new = False
        for b in range(batch_size):
            tok = int(out_tok[b].flatten()[0].item())
            if user_done[b]:
                continue
            if stop_at_eos and tok in stop_ids:
                user_done[b] = True
                logger.info(f"[TTNN] User {b} EOS at iter {iteration}")
                continue
            all_outputs[b].append(tok)
            any_new = True

        if iteration == 0 or (iteration + 1) % 8 == 0:
            logger.info(f"[TTNN] iter {iteration + 1}/{max_new_tokens}: {elapsed * 1000:.0f}ms")

        if stop_at_eos and all(user_done):
            break
        if not any_new:
            break

    _ = ref_model  # unused, kept for future reference-mode consistency checks
    return all_outputs


def run_ttnn_backend(
    model_id: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    *,
    dummy_weights: bool = False,
    stop_at_eos: bool = True,
) -> str:
    logger.info(
        f"[TTNN] Loading {model_id} and building Dots TT stack (MESH_DEVICE={os.environ.get('MESH_DEVICE', 'N150')})"
    )
    try:
        import ttnn
    except Exception as exc:  # pragma: no cover
        print(f"ttnn not importable: {exc}")
        return ""

    from models.demos.dots_ocr.tt.common import (
        PagedAttentionConfig,
        merge_vision_tokens,
        preprocess_inputs_prefill,
        text_rope_from_hf,
    )
    from models.demos.dots_ocr.tt.generator import Generator
    from models.demos.dots_ocr.tt.mesh import get_max_seq_len_cap, open_mesh_device
    from models.demos.dots_ocr.tt.model import DotsTransformer, DropInVisionTransformer
    from models.demos.dots_ocr.tt.model_config import DotsModelArgs
    from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs

    mesh_device = open_mesh_device()
    try:
        os.environ.setdefault("HF_MODEL", model_id)

        # HF reference for processor + token embeddings + RoPE helper + vision drop-in parent.
        ref = DotsOCRReference(HFLoadSpec(model_id=model_id))

        model_args = DotsModelArgs(
            mesh_device=mesh_device,
            max_batch_size=1,
            max_seq_len=get_max_seq_len_cap() or 4096,
            dummy_weights=dummy_weights,
        )
        state_dict = _load_state_dict_with_real_weights_fallback(model_args, dummy_weights)
        paged_cfg = PagedAttentionConfig(block_size=32, max_num_blocks=1024)

        tt_model = DotsTransformer(
            args=model_args,
            dtype=ttnn.bfloat8_b,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
            paged_attention_config=paged_cfg,
        )
        generator = Generator(tt_model, model_args, mesh_device, processor=ref.processor, tokenizer=ref.tokenizer)

        # Vision drop-in wrapping the HF reference's vision tower (same contract as qwen25_vl).
        visual = None
        if hasattr(ref.model, "vision_tower") or hasattr(ref.model, "visual"):
            vision_model_args = DotsVisionModelArgs(mesh_device=mesh_device, hf_config=ref.model.config)
            visual = DropInVisionTransformer(ref.model, vision_model_args, debug=False)

        # --- Build inputs ----------------------------------------------------
        image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None
        inputs = ref.preprocess_image_and_prompt(image, prompt)

        # Vision prefill (host fusion, same as qwen25_vl demo)
        if visual is not None and getattr(inputs, "pixel_values", None) is not None:
            t0 = time.perf_counter()
            image_embeds = visual(inputs.pixel_values, getattr(inputs, "image_grid_thw", None))
            logger.info(f"[TTNN] Vision forward: {time.perf_counter() - t0:.2f}s, {image_embeds.shape} tokens")
        else:
            image_embeds = torch.tensor([], dtype=torch.bfloat16)

        text_embeds = ref.model.get_input_embeddings()(inputs.input_ids)
        input_embeds = (
            merge_vision_tokens(inputs.input_ids, text_embeds, image_embeds, ref.model.config)
            if image_embeds.numel() > 0
            else text_embeds
        )

        pad_token_id = ref.tokenizer.pad_token_id or 0
        pad_embedding = ref.model.get_input_embeddings()(torch.tensor([pad_token_id])).squeeze(0)

        input_prefill, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            [input_embeds[0]],
            model_args,
            inputs.attention_mask,
            pad_embedding,
        )

        cos, sin = text_rope_from_hf(inputs, input_embeds, ref.model, model_args, pad_token_id)

        # --- Prefill ---------------------------------------------------------
        logger.info(f"[TTNN] Prefill: seq_len={prefill_lens[0]}")
        t0 = time.perf_counter()
        logits = generator.prefill_forward_text(
            input_prefill,
            rot_mats=(cos, sin),
            prompt_lens=torch.tensor(decoding_pos),
        )
        ttft = time.perf_counter() - t0

        # --- Decode ----------------------------------------------------------
        t0 = time.perf_counter()
        all_outputs = _decode_loop(
            generator=generator,
            tokenizer=ref.tokenizer,
            ref_model=ref.model,
            prefilled_logits=logits,
            decoding_pos=decoding_pos,
            max_new_tokens=max_new_tokens,
            stop_at_eos=stop_at_eos,
        )
        decode_time = time.perf_counter() - t0
        decoded = ref.tokenizer.decode(all_outputs[0], skip_special_tokens=True)

        print("\n=== TTNN ===")
        print(f"Backend:          ttnn")
        print(f"Prefill tokens:   {prefill_lens[0]}")
        print(f"Generated tokens: {len(all_outputs[0])}")
        print(f"TTFT:             {ttft:.2f}s")
        print(f"Decode time:      {decode_time:.2f}s")
        if len(all_outputs[0]) > 0 and decode_time > 0:
            print(f"Decode t/s:       {len(all_outputs[0]) / decode_time:.1f}")
        print(f"Output:           {decoded}")
        return decoded
    finally:
        ttnn.close_mesh_device(mesh_device)


# ---------------------------------------------------------------------------
# Prompt sources
# ---------------------------------------------------------------------------


def _load_prompts_from_json(path: str) -> list[tuple[str | None, str]]:
    """Load ``[{image, prompt}, ...]`` entries. Same shape as ``qwen25_vl/demo/sample_prompts/``."""
    with open(path, "r") as f:
        data = json.load(f)
    out: list[tuple[str | None, str]] = []
    for entry in data:
        out.append((entry.get("image"), entry.get("prompt", "")))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Dots OCR demo (HF reference or Dots TT stack)")
    parser.add_argument("--image", type=str, default=None, help="Path to an input image (optional for text-only)")
    parser.add_argument("--prompt", type=str, default="Read the text in this document.", help="OCR prompt")
    parser.add_argument(
        "--prompts-json",
        type=str,
        default=None,
        help="JSON file with `[{image, prompt}, ...]`; overrides --image/--prompt when set.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--hf-model", type=str, default=None, help="Override HF model id")
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "ttnn", "both"],
        help="'hf', 'ttnn', or 'both' (runs hf then ttnn).",
    )
    parser.add_argument("--dummy-weights", action="store_true", help="Force dummy weights in the TTNN backend.")
    parser.add_argument("--no-eos", action="store_true", help="Disable EOS stopping in the TTNN decode loop.")
    args = parser.parse_args()

    model_id = args.hf_model or get_hf_model_id()

    # Resolve prompt set
    if args.prompts_json:
        entries = _load_prompts_from_json(args.prompts_json)
    else:
        entries = [(args.image or "", args.prompt)]

    for image_path, prompt in entries:
        image_path = image_path or ""
        if args.backend in ("hf", "both"):
            run_hf_backend(model_id, image_path, prompt, args.max_new_tokens)
        if args.backend in ("ttnn", "both"):
            run_ttnn_backend(
                model_id,
                image_path,
                prompt,
                args.max_new_tokens,
                dummy_weights=args.dummy_weights,
                stop_at_eos=not args.no_eos,
            )


if __name__ == "__main__":
    main()
