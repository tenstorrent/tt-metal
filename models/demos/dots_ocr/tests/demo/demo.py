# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Dots OCR demo (CLI).

Two backends, selected with ``--backend``:

- ``hf``:   HuggingFace reference path (torch). Always produces full generations. Useful for
            validating output text and for comparing against the TT path.
- ``ttnn``: Builds the Dots TT stack (``DotsTransformer`` + optional
            ``DropInVisionTransformer`` for ``--vision-backend ttnn``) and runs prefill + decode via
            :class:`models.tt_transformers.tt.generator.Generator`. Requires ``MESH_DEVICE`` to be set.
            By default, device-side fusion/prefill padding is ON (disable with ``--no-device-fusion``). Greedy
            token selection uses TTNN argmax (with host ``torch.argmax`` only
            when ``--ttnn-repetition-penalty`` is set, since penalty is applied on host logits). Optional
            ``--fixed-decode-steps`` runs Option A fixed-length decode.

The TTNN path loads **real** checkpoint tensors via ``DotsModelArgs.load_real_state_dict`` /
``models.demos.dots_ocr.tt.load.load_dots_full_state_dict`` (filtered HF loads; no dummy weights).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from PIL import Image

from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec
from models.demos.dots_ocr.reference.model import DotsOCRReference


def configure_dots_ocr_console_logging() -> None:
    """
    Point loguru at stderr with a sane default level so noisy TTNN DEBUG lines
    (e.g. tensor ``.tensorbin`` cache load/generate in ``ttnn.from_torch``) stay hidden.

    Override with ``DOTS_LOG_LEVEL`` (e.g. ``DEBUG`` to see those messages again).
    """
    raw = (os.environ.get("DOTS_LOG_LEVEL") or "INFO").strip().upper()
    allowed = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR")
    level = raw if raw in allowed else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level)


def _ensure_local_ttnn_import() -> None:
    """
    Ensure we import TTNN from *this* repo checkout.

    When multiple tt-metal checkouts or a pip-installed `ttnn` exist, Python can import a namespace
    `ttnn` package (no `__file__`) or load `_ttnn.so` from a different location. That can produce
    "works but wrong output" behavior for Dots OCR. Force the repo root onto `sys.path` early.
    """
    # demo.py lives at: <repo>/models/demos/dots_ocr/tests/demo/demo.py
    repo_root = str(Path(__file__).resolve().parents[5])
    # TTNN python package lives at: <repo>/ttnn/ttnn/__init__.py, so we must add <repo>/ttnn to sys.path.
    ttnn_py_root = os.path.join(repo_root, "ttnn")
    for p in (ttnn_py_root, repo_root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _decode_pcc_value(
    *,
    ref: DotsOCRReference,
    inputs,
    tt_logits: torch.Tensor,
    fused_input_embeds: torch.Tensor | None,
) -> float | None:
    """Return PCC for last-token vocab row (or None on failure)."""
    try:
        if fused_input_embeds is None:
            return None
        attn = getattr(inputs, "attention_mask", None)
        if attn is None:
            return None
        position_ids = (attn.long().cumsum(dim=-1) - 1).clamp_min(0)
        with torch.no_grad():
            lm = getattr(getattr(ref.model, "model", ref.model), "language_model", None)
            if lm is None:
                lm = getattr(ref.model, "model", None)
            out_lm = lm(inputs_embeds=fused_input_embeds, attention_mask=attn, position_ids=position_ids)
            hidden = out_lm[0] if isinstance(out_lm, (tuple, list)) else out_lm.last_hidden_state
            hf_logits = ref.model.lm_head(hidden)
        if tt_logits.dim() == 3:
            tt_row = tt_logits[:, -1, :]
        elif tt_logits.dim() == 2:
            tt_row = tt_logits
        else:
            return None
        hf_row = hf_logits[:, -1, :]
        v = min(int(tt_row.shape[-1]), int(hf_row.shape[-1]))
        a = tt_row[..., :v]
        b = hf_row[..., :v]
        a0 = a.float() - a.float().mean()
        b0 = b.float() - b.float().mean()
        denom = (a0.norm() * b0.norm()).clamp_min(1e-8)
        return float((a0.flatten() @ b0.flatten()) / denom)
    except Exception:
        return None


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
        obj.mesh_device = None


# ---------------------------------------------------------------------------
# HF backend (reference generation)
# ---------------------------------------------------------------------------


def _decode_ttnn_new_tokens(
    ref: DotsOCRReference, token_ids: list[int], *, prompt_input_ids: torch.Tensor | None
) -> str:
    """
    Decode **only** generated token ids using the same path as HF ``decode_generated_suffix``:
    ``processor.batch_decode`` (multimodal templates), not ``tokenizer.decode`` alone.
    """
    if not token_ids:
        return ""
    if prompt_input_ids is not None and isinstance(prompt_input_ids, torch.Tensor) and prompt_input_ids.dim() == 2:
        full = torch.cat([prompt_input_ids.to(torch.long), torch.tensor([token_ids], dtype=torch.long)], dim=1)
        return ref.decode_generated_suffix(full, prompt_input_ids.to(torch.long))

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
        return d
    return d


def _strip_trailing_chat_markers(s: str) -> str:
    """Remove trailing ``<|...|>`` fragments that may remain after ``skip_special_tokens``."""
    return re.sub(r"(?:<\|[^|]+\|>\s*)+$", "", s).strip()


def _strip_generic_image_boilerplate(s: str) -> str:
    """
    Drop common non-OCR boilerplate loops (e.g. "The image shows the text.").

    Keep any earlier OCR-like content and remove the trailing boilerplate block.
    """
    s = (s or "").strip()
    if not s:
        return ""
    lines = [ln.rstrip() for ln in s.splitlines()]
    lower = [ln.strip().lower() for ln in lines]
    idxs = [i for i, ln in enumerate(lower) if ln.startswith("the image")]
    if len(idxs) >= 3:
        cut = idxs[0]
        kept = "\n".join(lines[:cut]).strip()
        return kept if kept else s
    return s


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
    if image_path and not os.path.exists(image_path):
        raise FileNotFoundError(f"--image path does not exist: {image_path}")
    image = Image.open(image_path).convert("RGB") if image_path else None
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
# TTNN backend (prefill + host decode, same general pattern as other tt_transformers VLM demos)
# ---------------------------------------------------------------------------


def _load_dots_ttnn_state_dict(model_args, *, text_qkv_permute: bool) -> dict:
    """Load real Dots checkpoint tensors for the TT text stack and validate critical keys."""
    real_sd = model_args.load_real_state_dict(qkv_permute=text_qkv_permute)
    required_keys = ("tok_embeddings.weight", "output.weight")
    missing = [k for k in required_keys if k not in real_sd]
    if missing:
        raise KeyError(f"Real state_dict missing required keys: {missing}")
    return real_sd


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
    fixed_steps: bool = False,
) -> list[list[int]]:
    """
    Greedy decode: one user per batch (typical for Dots OCR), returns per-user new token id lists.
    Uses TTNN argmax on device when possible; host ``torch.argmax`` when repetition penalty is applied
    (penalty runs on host logits).
    """
    from models.demos.dots_ocr.tt.common import (
        argmax_token_id_host_via_ttnn,
        argmax_token_id_ttnn,
        token_ids_ttnn_to_torch,
    )

    batch_size = prefilled_logits.shape[0]
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)], dtype=torch.int32)

    # Next-token selection for the final prefill token (TTNN; matches host argmax on the same rows).
    prefilled_token = argmax_token_id_host_via_ttnn(prefilled_logits, mesh_device=generator.mesh_device)
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
    if stop_at_eos and stop_ids:
        for b in range(batch_size):
            if int(prefilled_token[b].item()) in stop_ids:
                user_done[b] = True
        if all(user_done):
            return all_outputs

    # Option A: fixed-step decode. We still run a Python loop (TTNN does not provide a dynamic
    # device-side loop primitive here), but we eliminate per-token EOS bookkeeping and only
    # decode/tokenize after the loop.
    if fixed_steps:
        out_buf = torch.empty((batch_size, max_new_tokens), dtype=torch.int64)
        out_buf[:, 0] = prefilled_token.view(-1).to(torch.int64)
        for iteration in range(1, max_new_tokens):
            use_device_pick = not (repetition_penalty is not None and repetition_penalty > 1.0)
            if use_device_pick:
                tt_out = generator.decode_forward(
                    out_tok,
                    current_pos,
                    enable_trace=False,
                    page_table=None,
                    kv_cache=None,
                    sampling_params=None,
                    read_from_device=False,
                )
                if isinstance(tt_out, list):
                    tt_logits = tt_out[0][0]
                elif isinstance(tt_out, tuple):
                    tt_logits = tt_out[0]
                else:
                    tt_logits = tt_out
                vocab_size = int(
                    getattr(generator.model_args, "vocab_size", 0) or getattr(tokenizer, "vocab_size", 0) or 0
                )
                out_tok_tt = argmax_token_id_ttnn(
                    tt_logits,
                    mesh_device=generator.mesh_device,
                    batch_size=batch_size,
                    layout="decode",
                    vocab_size=(vocab_size if vocab_size > 0 else None),
                )
                out_tok = token_ids_ttnn_to_torch(out_tok_tt, mesh_device=generator.mesh_device)
                import ttnn

                ttnn.deallocate(tt_logits)
                ttnn.deallocate(out_tok_tt)
            else:
                logits, _ = generator.decode_forward(
                    out_tok,
                    current_pos,
                    enable_trace=False,
                    page_table=None,
                    kv_cache=None,
                    sampling_params=None,
                    read_from_device=True,
                )
                if repetition_penalty is not None and repetition_penalty > 1.0 and batch_size == 1:
                    seen = set(int(x) for x in out_buf[0, :iteration].tolist())
                    logits = _apply_repetition_penalty_to_logits(
                        logits, token_ids_to_penalize=seen, penalty=repetition_penalty
                    )
                out_tok = torch.argmax(logits, dim=-1)
            if out_tok.dim() == 1:
                out_tok = out_tok.unsqueeze(-1)
            out_buf[:, iteration] = out_tok.view(-1).to(torch.int64)
            current_pos += 1

        # Host-side trim at EOS/stop tokens (Option A still avoids per-step EOS bookkeeping).
        out_lists = [out_buf[b].tolist() for b in range(batch_size)]
        if stop_at_eos and stop_ids:
            trimmed: list[list[int]] = []
            for b in range(batch_size):
                seq = out_lists[b]
                cut = None
                for i, t in enumerate(seq):
                    if int(t) in stop_ids:
                        cut = i
                        break
                trimmed.append(seq[:cut] if cut is not None else seq)
            out_lists = trimmed
        return out_lists

    # Legacy EOS-aware path (kept for debugging or very short runs).
    # Account for the prefill argmax token already appended above.
    for iteration in range(1, max_new_tokens):
        use_device_pick = not (repetition_penalty is not None and repetition_penalty > 1.0)
        if use_device_pick:
            tt_out = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=False,
                page_table=None,
                kv_cache=None,
                sampling_params=None,
                read_from_device=False,
            )
            # tt_out is typically [(tt_logits, tt_log_probs)] for data_parallel=1.
            if isinstance(tt_out, list):
                tt_logits = tt_out[0][0]
            elif isinstance(tt_out, tuple):
                tt_logits = tt_out[0]
            else:
                tt_logits = tt_out
            vocab_size = int(getattr(generator.model_args, "vocab_size", 0) or getattr(tokenizer, "vocab_size", 0) or 0)
            out_tok_tt = argmax_token_id_ttnn(
                tt_logits,
                mesh_device=generator.mesh_device,
                batch_size=batch_size,
                layout="decode",
                vocab_size=(vocab_size if vocab_size > 0 else None),
            )
            out_tok = token_ids_ttnn_to_torch(out_tok_tt, mesh_device=generator.mesh_device)
            import ttnn

            ttnn.deallocate(tt_logits)
            ttnn.deallocate(out_tok_tt)
        else:
            logits, _ = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=False,
                page_table=None,
                kv_cache=None,
                sampling_params=None,
                read_from_device=True,
            )
            if repetition_penalty is not None and repetition_penalty > 1.0 and batch_size == 1:
                seen = set(all_outputs[0])
                logits = _apply_repetition_penalty_to_logits(
                    logits, token_ids_to_penalize=seen, penalty=repetition_penalty
                )
            elif repetition_penalty is not None and repetition_penalty > 1.0 and batch_size > 1:
                logger.warning(
                    "[TTNN] repetition_penalty is only applied when batch_size==1 (Dots OCR demo is batch=1)."
                )
            out_tok = torch.argmax(logits, dim=-1)
        if out_tok.dim() == 1:
            out_tok = out_tok.unsqueeze(-1)

        current_pos += 1
        any_new = False
        for b in range(batch_size):
            tok = int(out_tok[b].flatten()[0].item())
            if user_done[b]:
                continue
            if stop_at_eos and tok in stop_ids:
                user_done[b] = True
                continue
            all_outputs[b].append(tok)
            any_new = True

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
    vision_backend: str = "ttnn",
    text_qkv_permute: bool = True,
    use_host_rope: bool = False,
    sanity_text_only: bool = False,
    ttnn_repetition_penalty: float | None = None,
    device_fusion: bool = True,
    fixed_decode_steps: bool = False,
) -> str:
    logger.debug(
        f"[TTNN] Loading {model_id} and building Dots TT stack (MESH_DEVICE={os.environ.get('MESH_DEVICE', 'N150')})"
    )
    try:
        _ensure_local_ttnn_import()
        import ttnn

        try:
            import ttnn._ttnn  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "TTNN is not built/installed for this checkout (missing `ttnn._ttnn`). "
                "Build tt-metal/ttnn (or activate the environment that provides `_ttnn.so`) and retry."
            ) from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"ttnn not importable (TTNN backend was requested): {exc}") from exc

    from models.demos.dots_ocr.tt.common import (
        fused_ttnn_embeddings_to_torch,
        merge_vision_tokens,
        merge_vision_tokens_ttnn,
        pad_embedding_ttnn,
        pad_embedding_ttnn_tensor,
        preprocess_inputs_prefill,
        preprocess_inputs_prefill_ttnn,
        text_embeds_from_ttnn_embedding,
        text_embeds_from_ttnn_embedding_ttnn,
        text_rope_from_hf,
        ttnn_fused_batch_to_user_list,
    )
    from models.demos.dots_ocr.tt.generator import Generator
    from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, get_max_seq_len_cap, open_mesh_device
    from models.demos.dots_ocr.tt.model import DotsTransformer, DropInVisionTransformer
    from models.demos.dots_ocr.tt.model_config import DotsModelArgs
    from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs

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
        # HF reference for processor + token embeddings + config access.
        ref = DotsOCRReference(HFLoadSpec(model_id=model_id, use_fast_processor=not use_slow_processor))

        # Prefer bf16 for correctness. bf8 is faster but can be much less stable for text quality.
        tt_dtype = ttnn.bfloat16 if ttnn_dtype == "bf16" else ttnn.bfloat8_b

        def _build_text_stack(*, qkv_permute: bool):
            nonlocal model_args, tt_model, generator
            model_args = DotsModelArgs(
                mesh_device=mesh_device,
                hf_config=ref.model.config,
                max_batch_size=1,
                max_seq_len=get_max_seq_len_cap() or 4096,
            )
            # Ensure tensorbin cache keys include conversion choices.
            model_args.dots_text_qkv_permute = bool(qkv_permute)
            model_args.dots_use_host_rope = bool(use_host_rope)
            # For meaningful text output, avoid LM head output quantization.
            model_args.lm_head_dtype = ttnn.bfloat16
            state_dict = _load_dots_ttnn_state_dict(model_args, text_qkv_permute=qkv_permute)
            tt_model = DotsTransformer(
                args=model_args,
                dtype=tt_dtype,
                mesh_device=mesh_device,
                state_dict=state_dict,
                weight_cache_path=model_args.weight_cache_path(tt_dtype),
                paged_attention_config=None,
            )
            generator = Generator(tt_model, model_args, mesh_device, processor=ref.processor, tokenizer=ref.tokenizer)

        _build_text_stack(qkv_permute=bool(text_qkv_permute))

        # Vision embeddings:
        # - `hf`: `ref.vision_forward` (full HF `vision_tower` on host)
        # - `ttnn`: `DropInVisionTransformer` (TTNN QKV/MLP, torch RoPE+SDPA, TTNN merger; weights from ref state_dict)
        visual = None
        if vision_backend == "ttnn" and (hasattr(ref.model, "vision_tower") or hasattr(ref.model, "visual")):
            vision_model_args = DotsVisionModelArgs(mesh_device=mesh_device, hf_config=ref.model.config)
            visual = DropInVisionTransformer(ref.model, vision_model_args, debug=False)
        else:
            vision_model_args = None

        # --- Build inputs ----------------------------------------------------
        if image_path and not os.path.exists(image_path):
            raise FileNotFoundError(f"--image path does not exist: {image_path}")
        image = Image.open(image_path).convert("RGB") if image_path else None
        inputs = ref.preprocess_image_and_prompt(image, prompt)

        if sanity_text_only:
            inputs = ref.preprocess_image_and_prompt(None, prompt)

        # Vision forward
        image_embeds = torch.tensor([], dtype=torch.bfloat16)
        if not sanity_text_only and getattr(inputs, "pixel_values", None) is not None:
            if vision_backend == "hf":
                image_embeds = ref.vision_forward(inputs.pixel_values, getattr(inputs, "image_grid_thw", None))
            else:
                if visual is None:
                    raise RuntimeError("vision_backend=ttnn requested but TT vision stack was not constructed")
                image_embeds = visual(inputs.pixel_values, getattr(inputs, "image_grid_thw", None))
            # Vision stack may return either torch.Tensor or ttnn.Tensor depending on backend/module wrappers.
            # Normalize to torch here because downstream host fusion path requires torch semantics.
            if isinstance(image_embeds, ttnn.Tensor):
                image_embeds = ttnn.to_torch(image_embeds)
            image_embeds = image_embeds.to(torch.bfloat16)

        pad_token_id = ref.tokenizer.pad_token_id or 0

        if device_fusion:
            text_ttnn = text_embeds_from_ttnn_embedding_ttnn(tt_model, inputs.input_ids)
            if image_embeds.numel() > 0:
                fused_ttnn = merge_vision_tokens_ttnn(
                    inputs.input_ids, text_ttnn, image_embeds, ref.model.config, mesh_device=mesh_device
                )
            else:
                fused_ttnn = text_ttnn
            pad_tt = pad_embedding_ttnn_tensor(tt_model, int(pad_token_id))
            per_user = ttnn_fused_batch_to_user_list(fused_ttnn)
            input_prefill, decoding_pos, prefill_lens = preprocess_inputs_prefill_ttnn(
                per_user, model_args, inputs.attention_mask, pad_tt
            )
            prefill_for_rope: torch.Tensor | None = None
            if use_host_rope:
                prefill_for_rope = fused_ttnn_embeddings_to_torch(fused_ttnn, mesh_device)
        else:
            text_embeds = text_embeds_from_ttnn_embedding(tt_model, inputs.input_ids)
            input_embeds = (
                merge_vision_tokens(inputs.input_ids, text_embeds, image_embeds, ref.model.config)
                if image_embeds.numel() > 0
                else text_embeds
            )
            pad_embedding = pad_embedding_ttnn(tt_model, int(pad_token_id))
            input_prefill, decoding_pos, prefill_lens = preprocess_inputs_prefill(
                [input_embeds[0]],
                model_args,
                inputs.attention_mask,
                pad_embedding,
            )
            prefill_for_rope = input_embeds

        rot_mats = None
        if use_host_rope:
            if prefill_for_rope is None:
                raise RuntimeError("use_host_rope: internal error (no embeddings for RoPE)")
            # Use HF rotary_emb forward for exact RoPE match (host-only; TTNN still runs prefill/decode).
            cos, sin = text_rope_from_hf(inputs, prefill_for_rope, ref.model, model_args, pad_token_id)
            rot_mats = (cos, sin)
        # --- Prefill ---------------------------------------------------------
        t0 = time.perf_counter()
        logits = generator.prefill_forward_text(
            input_prefill, rot_mats=rot_mats, prompt_lens=torch.tensor(decoding_pos)
        )
        # If decode PCC is poor, try both qkv_permute settings once and keep the better one.
        if isinstance(logits, torch.Tensor):
            p_base = _decode_pcc_value(ref=ref, inputs=inputs, tt_logits=logits, fused_input_embeds=prefill_for_rope)
            do_try = p_base is not None and p_base < 0.85
            if do_try:
                results: dict[bool, tuple[float | None, torch.Tensor | None]] = {}
                # Evaluate both permutations deterministically.
                for perm in (False, True):
                    try:
                        # If the current stack already matches this perm, reuse `logits`.
                        if perm == bool(text_qkv_permute):
                            p = p_base
                            results[perm] = (p, logits)
                            continue

                        # Otherwise rebuild just the text stack and re-run prefill; keep vision/fusion as-is.
                        if generator is not None:
                            del generator
                        if tt_model is not None:
                            del tt_model
                        _build_text_stack(qkv_permute=perm)
                        logits_try = generator.prefill_forward_text(
                            input_prefill, rot_mats=rot_mats, prompt_lens=torch.tensor(decoding_pos)
                        )
                        if not isinstance(logits_try, torch.Tensor):
                            results[perm] = (None, None)
                            continue
                        p_try = _decode_pcc_value(
                            ref=ref, inputs=inputs, tt_logits=logits_try, fused_input_embeds=prefill_for_rope
                        )
                        results[perm] = (p_try, logits_try)
                    except Exception as e:
                        results[perm] = (None, None)
                        logger.warning(f"[TTNN] qkv_permute try({int(perm)}) failed: {type(e).__name__}: {e}")

                # Pick best PCC; require the winner to be valid.
                p0, l0 = results.get(False, (None, None))
                p1, l1 = results.get(True, (None, None))
                best_perm = bool(text_qkv_permute)
                best_p = p_base
                best_logits = logits
                if p0 is not None and (best_p is None or p0 > best_p):
                    best_perm, best_p, best_logits = False, p0, l0 if l0 is not None else best_logits
                if p1 is not None and (best_p is None or p1 > best_p):
                    best_perm, best_p, best_logits = True, p1, l1 if l1 is not None else best_logits

                if best_perm != bool(text_qkv_permute):
                    logger.debug(
                        f"[TTNN] Switching text_qkv_permute {int(bool(text_qkv_permute))} -> {int(best_perm)} "
                        f"(PCC {p_base} -> {best_p})"
                    )
                    logits = best_logits

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
            fixed_steps=fixed_decode_steps,
        )
        decode_time = time.perf_counter() - t0
        decoded = _decode_ttnn_new_tokens(ref, all_outputs[0], prompt_input_ids=inputs.input_ids)
        decoded = _strip_echoed_user_instruction(decoded, prompt)
        decoded = _strip_trailing_chat_markers(decoded)
        decoded = _strip_generic_image_boilerplate(decoded)

        print("\n=== TTNN ===")
        print(f"Backend:          ttnn")
        print(f"Prefill tokens:   {prefill_lens[0]}")
        print(f"Generated tokens: {len(all_outputs[0])}")
        print(f"TTFT:             {ttft:.2f}s")
        print(f"Decode time:      {decode_time:.2f}s")
        if len(all_outputs[0]) > 0 and decode_time > 0:
            print(f"Decode t/s:       {len(all_outputs[0]) / decode_time:.1f}")
        print(f"Output:           {decoded}")
        if device_fusion:
            print(f"Device fusion:   on (ttnn merge + ttnn prefill pad)")
        return decoded
    finally:
        # Explicitly drop objects that may hold MeshDevice references before closing the mesh.
        if generator is not None:
            del generator
        if visual is not None:
            del visual
        if tt_model is not None:
            del tt_model
        if ref is not None:
            del ref
        import gc

        gc.collect()
        # ``model_args`` / ``vision_model_args`` outlive ``tt_model``; clear mesh refs before close.
        _clear_tt_model_args_mesh_refs(model_args, vision_model_args)
        ttnn.synchronize_device(mesh_device)
        try:
            close_dots_mesh_device(mesh_device)
        finally:
            # Drop Python refs to mesh wrappers so nanobind can tear down cleanly at interpreter exit.
            if "mesh_device" in locals():
                del mesh_device
            if parent_mesh_ref is not None:
                del parent_mesh_ref
            import gc

            gc.collect()


# ---------------------------------------------------------------------------
# Prompt sources
# ---------------------------------------------------------------------------


def _load_prompts_from_json(path: str) -> list[tuple[str | None, str]]:
    """Load ``[{image, prompt}, ...]`` entries (list of dicts with optional ``image`` and ``prompt``)."""
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
    configure_dots_ocr_console_logging()
    parser = argparse.ArgumentParser(description="Dots OCR demo (HF reference or Dots TT stack)")
    parser.add_argument("--image", type=str, default=None, help="Path to an input image (optional for text-only)")
    parser.add_argument(
        "--prompt",
        type=str,
        default="OCR: transcribe the text in the image exactly. Output only the transcription.",
        help="OCR prompt",
    )
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
            "Vision for TTNN run: 'hf' uses ``ref.vision_forward`` (full HF vision_tower on host). "
            "'ttnn' uses ``DropInVisionTransformer``: patchify on host, TTNN QKV/MLP linears, "
            "torch RoPE+SDPA, TTNN patch merger (no HF block forward)."
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
        dest="use_host_rope",
        action="store_true",
        default=True,
        help=(
            "Use HF-derived host RoPE cos/sin for TTNN run (default: ON; matches current demo behavior). "
            "Disable with --no-host-rope to exercise device-side RoPE caches."
        ),
    )
    parser.add_argument(
        "--no-host-rope",
        dest="use_host_rope",
        action="store_false",
        help="Disable host RoPE and use device-side RoPE caches for TTNN run.",
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
        help="TTNN decode only: HF-style repetition penalty (>1.0, e.g. 1.12–1.2) to reduce **/token loops.",
    )
    parser.add_argument(
        "--no-device-fusion",
        action="store_true",
        help="Disable TTNN device-side fusion/prefill padding (defaults to ON).",
    )
    parser.add_argument(
        "--fixed-decode-steps",
        action="store_true",
        help="Option A: run exactly --max-new-tokens decode steps, then trim at EOS. Default is EOS-aware per-step loop.",
    )
    args = parser.parse_args()

    model_id = args.hf_model or "rednote-hilab/dots.mocr"
    preset = args.ocr_preset

    ttnn_repetition_penalty = args.ttnn_repetition_penalty

    # No env-var dependency: drive behavior via CLI only.
    device_fusion = not bool(args.no_device_fusion)
    fixed_decode_steps = bool(args.fixed_decode_steps)

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
                device_fusion=device_fusion,
                fixed_decode_steps=fixed_decode_steps,
            )


if __name__ == "__main__":
    main()
