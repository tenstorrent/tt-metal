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

The TTNN path loads **real** checkpoint tensors via ``DotsModelArgs.load_real_state_dict`` /
``models.demos.dots_ocr.tt.load.load_dots_full_state_dict`` (filtered HF loads; no dummy weights).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch
from loguru import logger
from PIL import Image

from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec, get_hf_model_id
from models.demos.dots_ocr.reference.model import DotsOCRReference


def _apply_repetition_penalty_to_logits(
    logits: torch.Tensor, *, token_ids_to_penalize: set[int], penalty: float
) -> torch.Tensor:
    """
    HuggingFace-style repetition penalty on the vocabulary row used for argmax.

    Reduces re-selection of tokens already seen in the running decode stream (helps when TT
    logits are imperfect and greedy decode loops on markdown / punctuation).
    """
    if penalty <= 1.0 or not token_ids_to_penalize:
        return logits
    out = logits.clone()
    if out.dim() == 3:
        row = out[0, -1, :]
    elif out.dim() == 2:
        row = out[0, :]
    else:
        return logits
    rf = row.float()
    for tid in token_ids_to_penalize:
        if 0 <= tid < rf.numel():
            v = rf[tid]
            if v < 0:
                rf[tid] = v * penalty
            else:
                rf[tid] = v / penalty
    row.copy_(rf.to(row.dtype))
    return out


def _clear_tt_model_args_mesh_refs(*args_objs) -> None:
    """
    Drop ``mesh_device`` from tt_transformers ``ModelArgs``-like objects.

    ``run_ttnn_backend`` keeps ``model_args`` / ``vision_model_args`` alive until the function
    returns; they still reference the open mesh after ``del tt_model`` / ``del generator``, which
    prevents nanobind from destroying ``MeshDevice`` wrappers at shutdown.
    """
    for obj in args_objs:
        if obj is None:
            continue
        try:
            obj.mesh_device = None
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HF backend (reference generation)
# ---------------------------------------------------------------------------


def _decode_ttnn_new_tokens(ref: DotsOCRReference, token_ids: list[int]) -> str:
    """
    Decode **only** generated token ids using the same path as HF ``decode_generated_suffix``:
    ``processor.batch_decode`` (multimodal templates), not ``tokenizer.decode`` alone.
    """
    if not token_ids:
        return ""
    row = torch.tensor([token_ids], dtype=torch.long)
    proc = getattr(ref, "processor", None)
    if proc is not None and hasattr(proc, "batch_decode"):
        return proc.batch_decode(row, skip_special_tokens=True)[0]
    return ref.tokenizer.decode(token_ids, skip_special_tokens=True)


def _strip_echoed_user_instruction(decoded: str, user_prompt: str) -> str:
    """
    When TT logits are weak, the model can regurgitate the OCR instruction. If the decoded
    string starts with the same user prompt we prefilled, drop that prefix for a cleaner readout.

    If stripping would remove *all* visible text (model echoed only the instruction), keep the
    original string so the demo does not print a blank ``Output:`` line.
    """
    d = decoded.strip()
    p = (user_prompt or "").strip()
    if p and d.startswith(p):
        rest = d[len(p) :].strip()
        if rest:
            return rest
        logger.info("[TTNN] Decoded text matched the user prompt only; skipping echo strip so output is not empty.")
        return d
    return d


def _strip_trailing_chat_markers(s: str) -> str:
    """Remove trailing ``<|...|>`` fragments that may remain after ``skip_special_tokens``."""
    return re.sub(r"(?:<\|[^|]+\|>\s*)+$", "", s).strip()


# ---------------------------------------------------------------------------
# HF backend (reference generation)
# ---------------------------------------------------------------------------


def run_hf_backend(
    model_id: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    *,
    repetition_penalty: float | None = None,
    use_slow_processor: bool = False,
) -> str:
    logger.info(f"[HF] Loading {model_id}")
    ref = DotsOCRReference(HFLoadSpec(model_id=model_id, use_fast_processor=not use_slow_processor))
    image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None
    inputs = ref.preprocess_image_and_prompt(image, prompt)

    gen_extras: dict = {}
    rpen = repetition_penalty
    if rpen is None:
        env_r = os.environ.get("DOTS_HF_REPETITION_PENALTY", "").strip()
        if env_r:
            rpen = float(env_r)
    if rpen is not None:
        gen_extras["repetition_penalty"] = rpen

    t0 = time.perf_counter()
    out = ref.forward(inputs, max_new_tokens=max_new_tokens, **gen_extras)
    elapsed = time.perf_counter() - t0

    # ``generate`` returns prompt + new tokens; decode only the continuation (not the instruction).
    text = ref.decode_generated_suffix(out["generated_ids"], inputs.input_ids)
    print("\n=== HF ===")
    print(f"Backend:   hf")
    print(f"Latency:   {elapsed:.2f}s for up to {max_new_tokens} tokens")
    print(f"Output:    {text}")
    return text


# ---------------------------------------------------------------------------
# TTNN backend — mirrors qwen25_vl/demo/demo.py end-to-end flow
# ---------------------------------------------------------------------------


def _load_dots_ttnn_state_dict(model_args, *, text_qkv_permute: bool) -> dict:
    """Load real Dots checkpoint tensors for the TT text stack and validate critical keys."""
    logger.info(f"[TTNN] Loading real Dots weights (text_qkv_permute={int(text_qkv_permute)})")
    real_sd = model_args.load_real_state_dict(qkv_permute=text_qkv_permute)
    required_keys = ("tok_embeddings.weight", "output.weight")
    missing = [k for k in required_keys if k not in real_sd]
    if missing:
        raise KeyError(f"Real state_dict missing required keys: {missing}")
    logger.info(f"[TTNN] Loaded real Dots state_dict with {len(real_sd)} tensors.")
    return real_sd


def _build_tt_stack(model_id: str, mesh_device, *, max_seq_len: int, text_qkv_permute: bool = True):
    """
    Shared TT stack builder for demos + perf benchmark (mirrors qwen25_vl structure).

    Returns: (ref, model_args, tt_model, generator, visual)
    """
    import ttnn
    from models.demos.dots_ocr.tt.generator import Generator
    from models.demos.dots_ocr.tt.model import DotsTransformer
    from models.demos.dots_ocr.tt.model_config import DotsModelArgs
    from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs
    from models.demos.dots_ocr.tt.vision_qwenstyle import DropInVisionTransformerQwenStyle

    os.environ.setdefault("HF_MODEL", model_id)
    ref = DotsOCRReference(HFLoadSpec(model_id=model_id))

    model_args = DotsModelArgs(
        mesh_device=mesh_device,
        hf_config=ref.model.config,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )
    state_dict = _load_dots_ttnn_state_dict(model_args, text_qkv_permute=text_qkv_permute)

    # Dense KV cache (max_seq_len along seq dim). Paged KV + prefill without page_table uses
    # fill_cache, which requires cache seq >= padded prefill; paged tensors only expose
    # block_size (e.g. 32) on that axis — use paged attention only when wiring page_table
    # like qwen25_vl/demo (create_tt_page_table + paged_fill_cache).
    tt_model = DotsTransformer(
        args=model_args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=None,
    )
    generator = Generator(tt_model, model_args, mesh_device, processor=ref.processor, tokenizer=ref.tokenizer)

    visual = None
    if hasattr(ref.model, "vision_tower") or hasattr(ref.model, "visual"):
        vision_model_args = DotsVisionModelArgs(mesh_device=mesh_device, hf_config=ref.model.config)
        visual = DropInVisionTransformerQwenStyle(ref.model, vision_model_args, debug=False)

    return ref, model_args, tt_model, generator, visual


def _decode_loop(
    *,
    generator,
    tokenizer,
    ref_model,
    prefilled_logits: torch.Tensor,
    decoding_pos: list[int],
    max_new_tokens: int,
    stop_at_eos: bool = True,
    repetition_penalty: float | None = None,
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
    logger.info(
        f"[TTNN] Starting decode loop (max_new_tokens={max_new_tokens}, stop_at_eos={stop_at_eos}, "
        f"repetition_penalty={repetition_penalty})"
    )

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
        if repetition_penalty is not None and repetition_penalty > 1.0 and batch_size == 1:
            seen = set(all_outputs[0])
            logits = _apply_repetition_penalty_to_logits(logits, token_ids_to_penalize=seen, penalty=repetition_penalty)
        elif repetition_penalty is not None and repetition_penalty > 1.0 and batch_size > 1:
            logger.warning("[TTNN] repetition_penalty is only applied when batch_size==1 (Dots OCR demo is batch=1).")
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
                logger.info(f"[TTNN] User {b} EOS at iter {iteration} (token_id={tok})")
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
    stop_at_eos: bool = True,
    use_slow_processor: bool = False,
    ttnn_dtype: str = "bf16",
    vision_backend: str = "hf",
    text_qkv_permute: bool = True,
    use_host_rope: bool = False,
    sanity_text_only: bool = False,
    ttnn_repetition_penalty: float | None = None,
) -> str:
    logger.info(
        f"[TTNN] Loading {model_id} and building Dots TT stack (MESH_DEVICE={os.environ.get('MESH_DEVICE', 'N150')})"
    )
    try:
        import ttnn
    except Exception as exc:  # pragma: no cover
        print(f"ttnn not importable: {exc}")
        return ""

    from models.demos.dots_ocr.tt.common import merge_vision_tokens, preprocess_inputs_prefill, text_rope_from_hf
    from models.demos.dots_ocr.tt.generator import Generator
    from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, get_max_seq_len_cap, open_mesh_device
    from models.demos.dots_ocr.tt.model import DotsTransformer
    from models.demos.dots_ocr.tt.model_config import DotsModelArgs
    from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs
    from models.demos.dots_ocr.tt.vision_qwenstyle import DropInVisionTransformerQwenStyle

    # Must stay None until assigned so ``finally`` can clear mesh refs safely.
    model_args = None
    vision_model_args = None
    parent_mesh_ref = None
    mesh_device = open_mesh_device()
    parent_mesh_ref = getattr(mesh_device, "_dots_parent_mesh_device", None)
    # Ensure objects holding TT resources are released before closing the mesh device.
    generator = None
    tt_model = None
    visual = None
    ref = None
    try:
        os.environ.setdefault("HF_MODEL", model_id)

        # HF reference for processor + token embeddings + RoPE helper + vision drop-in parent.
        ref = DotsOCRReference(HFLoadSpec(model_id=model_id, use_fast_processor=not use_slow_processor))

        model_args = DotsModelArgs(
            mesh_device=mesh_device,
            hf_config=ref.model.config,
            max_batch_size=1,
            max_seq_len=get_max_seq_len_cap() or 4096,
        )
        # For meaningful text output, avoid LM head output quantization.
        model_args.lm_head_dtype = ttnn.bfloat16
        state_dict = _load_dots_ttnn_state_dict(model_args, text_qkv_permute=text_qkv_permute)

        # Prefer bf16 for correctness. bf8 is faster but can be much less stable for text quality.
        tt_dtype = ttnn.bfloat16 if ttnn_dtype == "bf16" else ttnn.bfloat8_b

        # See _build_tt_stack: dense KV for prefill/decode without page_table.
        tt_model = DotsTransformer(
            args=model_args,
            dtype=tt_dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(tt_dtype),
            paged_attention_config=None,
        )
        generator = Generator(tt_model, model_args, mesh_device, processor=ref.processor, tokenizer=ref.tokenizer)

        # Vision embeddings:
        # - `hf`: match the working reference demo exactly (torch vision_tower)
        # - `ttnn`: run the TT vision stack (bring-up; may not match HF yet)
        visual = None
        if vision_backend == "ttnn" and (hasattr(ref.model, "vision_tower") or hasattr(ref.model, "visual")):
            vision_model_args = DotsVisionModelArgs(mesh_device=mesh_device, hf_config=ref.model.config)
            visual = DropInVisionTransformerQwenStyle(ref.model, vision_model_args, debug=False)
        else:
            vision_model_args = None

        # --- Build inputs ----------------------------------------------------
        image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None
        inputs = ref.preprocess_image_and_prompt(image, prompt)

        if sanity_text_only:
            logger.info("[TTNN] Sanity run: text-only (no image) using token-id prefill")
            inputs = ref.preprocess_image_and_prompt(None, prompt)

        # Vision forward
        image_embeds = torch.tensor([], dtype=torch.bfloat16)
        if not sanity_text_only and getattr(inputs, "pixel_values", None) is not None:
            t0 = time.perf_counter()
            if vision_backend == "hf":
                image_embeds = ref.vision_forward(inputs.pixel_values, getattr(inputs, "image_grid_thw", None))
            else:
                if visual is None:
                    raise RuntimeError("vision_backend=ttnn requested but TT vision stack was not constructed")
                image_embeds = visual(inputs.pixel_values, getattr(inputs, "image_grid_thw", None))
            logger.info(
                f"[TTNN] Vision forward ({vision_backend}): {time.perf_counter() - t0:.2f}s, {tuple(image_embeds.shape)}"
            )
            image_embeds = image_embeds.to(torch.bfloat16)

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

        rot_mats = None
        if use_host_rope:
            cos, sin = text_rope_from_hf(inputs, input_embeds, ref.model, model_args, pad_token_id)
            rot_mats = (cos, sin)
        else:
            logger.info("[TTNN] Using device-side RoPE caches (no host rot_mats)")

        # --- Prefill ---------------------------------------------------------
        logger.info(f"[TTNN] Prefill: seq_len={prefill_lens[0]}")
        t0 = time.perf_counter()
        logits = generator.prefill_forward_text(
            input_prefill, rot_mats=rot_mats, prompt_lens=torch.tensor(decoding_pos)
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
            repetition_penalty=ttnn_repetition_penalty,
        )
        decode_time = time.perf_counter() - t0
        decoded = _decode_ttnn_new_tokens(ref, all_outputs[0])
        decoded = _strip_echoed_user_instruction(decoded, prompt)
        decoded = _strip_trailing_chat_markers(decoded)

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
        # Explicitly drop objects that may hold MeshDevice references before closing the mesh.
        try:
            if generator is not None:
                del generator
            if visual is not None:
                del visual
            if tt_model is not None:
                del tt_model
            if ref is not None:
                del ref
        except Exception:
            pass
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        # ``model_args`` / ``vision_model_args`` outlive ``tt_model``; clear mesh refs before close.
        _clear_tt_model_args_mesh_refs(model_args, vision_model_args)
        try:
            ttnn.synchronize_device(mesh_device)
        except Exception:
            pass
        try:
            close_dots_mesh_device(mesh_device)
        finally:
            # Drop Python refs to mesh wrappers so nanobind can tear down cleanly at interpreter exit.
            try:
                del mesh_device
            except Exception:
                pass
            try:
                if parent_mesh_ref is not None:
                    del parent_mesh_ref
            except Exception:
                pass
            try:
                import gc

                gc.collect()
            except Exception:
                pass


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
        "--ocr-preset",
        type=str,
        default=None,
        choices=["en", "zh"],
        help="Override --prompt with a known-good OCR prompt style (English or Chinese).",
    )
    parser.add_argument(
        "--prompts-json",
        type=str,
        default=None,
        help="JSON file with `[{image, prompt}, ...]`; overrides --image/--prompt when set.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--hf-repetition-penalty",
        type=float,
        default=None,
        help="HF reference only: passed to generate() to reduce repeated tokens (also DOTS_HF_REPETITION_PENALTY).",
    )
    parser.add_argument("--hf-model", type=str, default=None, help="Override HF model id")
    parser.add_argument(
        "--use-slow-processor",
        action="store_true",
        help="Force the slow processor (`use_fast=False`). Can improve OCR tokenization/preprocessing stability.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "ttnn", "both"],
        help="'hf', 'ttnn', or 'both' (runs hf then ttnn).",
    )
    parser.add_argument(
        "--ttnn-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "bf8"],
        help="TTNN dtype for text decoder weights/activations. bf16 is recommended for correctness.",
    )
    parser.add_argument(
        "--vision-backend",
        type=str,
        default="hf",
        choices=["hf", "ttnn"],
        help=(
            "Vision embeddings for TTNN: 'hf' uses the HF vision tower (matches reference OCR; recommended). "
            "'ttnn' uses the on-device vision stack (bring-up; logits/OCR often differ until PCC matches HF)."
        ),
    )
    parser.add_argument(
        "--text-qkv-permute",
        action="store_true",
        default=True,
        help="Use HF->Meta Q/K permute in text weight conversion (improves correctness for dots.mocr).",
    )
    parser.add_argument(
        "--no-text-qkv-permute",
        dest="text_qkv_permute",
        action="store_false",
        help="Disable HF->Meta Q/K permute for text weight conversion.",
    )
    parser.add_argument(
        "--use-host-rope",
        action="store_true",
        help="Use HF-derived host RoPE cos/sin (default is device-side RoPE caches).",
    )
    parser.add_argument(
        "--sanity-text-only",
        action="store_true",
        help="Sanity check TTNN text stack using text-only token-id prefill (no image).",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Disable EOS stopping in the TTNN decode loop (try if generation stops after a few tokens).",
    )
    parser.add_argument(
        "--ttnn-repetition-penalty",
        type=float,
        default=None,
        help="TTNN decode only: HF-style repetition penalty (>1.0, e.g. 1.12–1.2) to reduce **/token loops. "
        "Also DOTS_TTNN_REPETITION_PENALTY.",
    )
    args = parser.parse_args()

    model_id = args.hf_model or get_hf_model_id()
    preset = args.ocr_preset

    ttnn_repetition_penalty = args.ttnn_repetition_penalty
    if ttnn_repetition_penalty is None:
        _rp = os.environ.get("DOTS_TTNN_REPETITION_PENALTY", "").strip()
        if _rp:
            try:
                ttnn_repetition_penalty = float(_rp)
            except ValueError:
                ttnn_repetition_penalty = None

    # Resolve prompt set
    if args.prompts_json:
        entries = _load_prompts_from_json(args.prompts_json)
    else:
        prompt = args.prompt
        if preset == "en":
            prompt = "OCR: transcribe the text in the image exactly. Output only the transcription."
        elif preset == "zh":
            prompt = "请识别图片中的文字，逐字输出，不要解释，不要重复题目。"
        entries = [(args.image or "", prompt)]

    for image_path, prompt in entries:
        image_path = image_path or ""
        if args.backend in ("hf", "both"):
            run_hf_backend(
                model_id,
                image_path,
                prompt,
                args.max_new_tokens,
                repetition_penalty=args.hf_repetition_penalty,
                use_slow_processor=args.use_slow_processor,
            )
        if args.backend in ("ttnn", "both"):
            run_ttnn_backend(
                model_id,
                image_path,
                prompt,
                args.max_new_tokens,
                stop_at_eos=not args.no_eos,
                use_slow_processor=args.use_slow_processor,
                ttnn_dtype=args.ttnn_dtype,
                vision_backend=args.vision_backend,
                text_qkv_permute=args.text_qkv_permute,
                use_host_rope=args.use_host_rope,
                sanity_text_only=args.sanity_text_only,
                ttnn_repetition_penalty=ttnn_repetition_penalty,
            )


if __name__ == "__main__":
    main()
