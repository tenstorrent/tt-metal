# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Official ACE-Step LM + handler conditioning for TTNN demos.

Host preprocessing helpers for the TTNN demo (5 Hz LM / CoT Phase 1 and handler kwargs).
passed into ``AceStepHandler.generate_music``, then runs only:

  ``_normalize_service_generate_inputs`` → ``_prepare_batch`` → ``preprocess_batch``
  → ``model.prepare_condition``

so TTNN can own DiT diffusion. The DiT-caption branch can swap in a TTNN Qwen encoder via
``attach_infer_text_embeddings_ttnn``. Does not call ``service_generate`` / PyTorch DiT.
"""

from __future__ import annotations

import inspect
import math
import os
import re
import sys
from typing import Any, Callable

import numpy as np
import torch
from loguru import logger


def release_preprocess_device_traces(
    *,
    device: Any,
    tt_qwen_encoder: Any | None = None,
    tt_audio_detokenizer: Any | None = None,
) -> None:
    """Drop optional preprocess traces so the next module can capture safely."""
    import ttnn

    ttnn.synchronize_device(device)
    if tt_qwen_encoder is not None and hasattr(tt_qwen_encoder, "release_trace"):
        tt_qwen_encoder.release_trace()
    if tt_audio_detokenizer is not None and hasattr(tt_audio_detokenizer, "release_trace"):
        tt_audio_detokenizer.release_trace()


def qwen_caption_encode_tt(
    tt_qwen_encoder: Any,
    ids_np: np.ndarray,
    attn_np: np.ndarray | None,
    *,
    use_trace: bool,
) -> Any:
    """Run ``AceStepQwen3Encoder`` caption prefill; trace replay when ``use_trace`` and ``B == 1``.

    ``forward_traced`` requires ``num_command_queues=2`` on the device (same as DiT trace).
    Batch sizes other than 1 fall back to eager ``forward()`` with a one-time warning.
    """
    b = int(ids_np.shape[0])
    if use_trace:
        if b != 1:
            logger.warning(
                "[ace_step_v1_5] --use-trace: Qwen3 trace supports batch_size=1 only; " "using eager forward for B={}.",
                b,
            )
            return tt_qwen_encoder.forward(ids_np, attn_np)
        return tt_qwen_encoder.forward_traced(ids_np)
    return tt_qwen_encoder.forward(ids_np, attn_np)


def condition_encode_tt(
    condition_encoder: Any,
    text_hidden_tt: Any,
    attn_mask_np: np.ndarray,
    *,
    use_trace: bool,
) -> tuple[Any, np.ndarray, Any]:
    """Run ``TtAceStepInstrumentalConditionEncoder`` fast instrumental path.

    When ``use_trace`` is True, uses :meth:`forward_traced` (2 CQs, persistent output buffer).
    Falls back to eager :meth:`forward` when trace is disabled or the build lacks trace APIs.
    """
    attn = np.asarray(attn_mask_np, dtype=np.float32).reshape(1, -1)
    s = int(text_hidden_tt.shape[2])
    if int(attn.shape[1]) != s:
        raise ValueError(f"attention_mask length {attn.shape[1]} != text sequence length {s}")
    valid = int(attn.sum())
    if use_trace and hasattr(condition_encoder, "forward_traced"):
        enc_tt, null_tt = condition_encoder.forward_traced(text_hidden_tt, valid)
        enc_mask_np = condition_encoder._enc_mask_np(valid, s)
        return enc_tt, enc_mask_np, null_tt
    return condition_encoder.forward(text_hidden_tt, attn_mask_np)


def split_prompt_instruments(text: str) -> list[str]:
    """Split comma/and-separated instrument phrases from a user caption."""
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = re.split(r"[,;&]|\band\b", raw, flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        p = part.strip()
        if len(p) > 1:
            out.append(p)
    return out


def _instrument_aliases(token: str) -> set[str]:
    t = token.lower().strip()
    aliases = {t}
    if t.endswith("s"):
        aliases.add(t[:-1])
    else:
        aliases.add(f"{t}s")
    if "sax" in t:
        aliases.update({"sax", "saxophone", "saxaphone"})
    if "drum" in t or t == "percussion":
        aliases.update({"drum", "drums", "percussion", "kit"})
    if "guitar" in t:
        aliases.update({"guitar", "guitars"})
    return aliases


def instrument_mentioned_in_caption(instrument: str, caption: str) -> bool:
    cap = str(caption or "").lower()
    return any(alias in cap for alias in _instrument_aliases(instrument))


def instruments_missing_from_caption(user_caption: str, final_caption: str) -> list[str]:
    parts = split_prompt_instruments(user_caption)
    if len(parts) < 2:
        return []
    return [p for p in parts if not instrument_mentioned_in_caption(p, final_caption)]


def merge_user_instruments_into_caption(user_caption: str, cot_caption: str) -> tuple[str, list[str]]:
    """Keep CoT prose but re-inject user-listed instruments dropped from LM caption."""
    user = str(user_caption or "").strip()
    cot = str(cot_caption or "").strip()
    if not user or not cot:
        return cot or user, []
    missing = instruments_missing_from_caption(user, cot)
    if not missing:
        return cot, []
    addon = ", ".join(missing)
    if cot.rstrip().endswith("."):
        merged = f"{cot.rstrip()} Also featuring {addon}."
    else:
        merged = f"{cot.rstrip()}, featuring {addon}"
    return merged, missing


def reconcile_lm_caption_with_user(
    *,
    user_caption: str,
    metadata: dict[str, Any],
    use_cot_caption: bool,
) -> dict[str, Any]:
    """Ensure Phase-2 YAML + DiT caption retain explicit multi-instrument user prompts."""
    user = str(user_caption or "").strip()
    if not user:
        return metadata
    cot_cap = str(metadata.get("caption") or "").strip()
    if cot_cap and use_cot_caption:
        merged, missing = merge_user_instruments_into_caption(user, cot_cap)
        if missing:
            metadata = dict(metadata)
            metadata["caption"] = merged
    elif not cot_cap:
        metadata = dict(metadata)
        metadata["caption"] = user
    return metadata


def log_caption_conditioning_trace(
    *,
    user_caption: str,
    lm_cot_caption: str | None,
    final_caption: str,
    use_cot_caption: bool,
) -> None:
    """Print how user prompt flows into LM CoT and DiT text conditioning."""
    user = str(user_caption or "").strip()
    final = str(final_caption or "").strip()
    cot = str(lm_cot_caption or "").strip() if lm_cot_caption else ""
    parts = split_prompt_instruments(user)
    missing = instruments_missing_from_caption(user, final) if user else []
    print(
        f"[ace_step_v1_5] caption trace: user={user!r} use_cot_caption={use_cot_caption}",
        flush=True,
    )
    if cot and cot != user:
        print(f"[ace_step_v1_5] caption trace: lm_cot={cot!r}", flush=True)
    print(f"[ace_step_v1_5] caption trace: dit_final={final!r}", flush=True)
    if len(parts) >= 2:
        present = [p for p in parts if instrument_mentioned_in_caption(p, final)]
        print(
            f"[ace_step_v1_5] caption trace: instruments listed={parts} "
            f"present_in_final={present} missing={missing}",
            flush=True,
        )
        if missing:
            print(
                f"[ace_step_v1_5] WARNING: user listed {missing} but final caption may omit them — "
                "try --no-use-cot-caption or a more explicit prompt",
                flush=True,
            )


def _code_window_unique_ratio(codes: list[int], start: int, window: int) -> float:
    chunk = codes[start : start + window]
    if not chunk:
        return 0.0
    return float(len(set(chunk))) / float(len(chunk))


def find_degenerate_code_prefix_len(codes: list[int], *, window: int = 25) -> int | None:
    """Return code index where the stream collapses into filler/repeats; ``None`` if healthy."""
    if len(codes) < 2 * window:
        return None
    head_ratio = _code_window_unique_ratio(codes, 0, window)
    if head_ratio < 0.12:
        return None
    collapse_ratio = max(0.06, head_ratio * 0.35)
    step = max(1, window // 2)
    for start in range(window, len(codes) - window + 1, step):
        cur = codes[start : start + window]
        prev = codes[start - window : start]
        if cur == prev:
            return start
        ratio = _code_window_unique_ratio(codes, start, window)
        if ratio >= collapse_ratio:
            continue
        next_start = start + window
        if next_start + window <= len(codes):
            next_ratio = _code_window_unique_ratio(codes, next_start, window)
            if next_ratio >= collapse_ratio:
                continue
        return max(window, start)
    return None


def repair_degenerate_lm_hint_tail(
    payload: dict[str, Any],
    code_string: str,
    target_frames: int,
    *,
    min_good_codes: int = 50,
    tt_audio_detokenizer: Any | None = None,
) -> None:
    """Silence-pad hint tail when LM fills to target with low-quality / repetitive codes."""
    from models.experimental.ace_step_v1_5.utils.device_lm_hints import (
        ace_step_device_native_detok_hints,
        repair_degenerate_lm_hint_tail_tt,
    )

    if (
        ace_step_device_native_detok_hints()
        and payload.get("precomputed_lm_hints_25Hz_tt") is not None
        and tt_audio_detokenizer is not None
    ):
        import ttnn

        if repair_degenerate_lm_hint_tail_tt(
            payload,
            code_string,
            target_frames,
            device=tt_audio_detokenizer.device,
            mem=getattr(tt_audio_detokenizer, "mem", None),
            dtype=getattr(tt_audio_detokenizer, "dtype", ttnn.float32),
            min_good_codes=min_good_codes,
        ):
            return

    from models.experimental.ace_step_v1_5.ttnn_impl.audio_code_detokenizer import parse_audio_code_string

    hints = payload.get("precomputed_lm_hints_25Hz")
    sil = payload.get("silence_latent")
    if hints is None or sil is None:
        return
    codes = parse_audio_code_string(str(code_string or ""))
    window = 25
    if len(codes) < min_good_codes + window:
        return
    prefix_len = find_degenerate_code_prefix_len(codes)
    if prefix_len is None or prefix_len < min_good_codes:
        return
    if prefix_len >= int(len(codes) * 0.85):
        return

    target = int(target_frames)
    good_frames = int(prefix_len * 5)
    if good_frames <= 0 or good_frames >= target:
        return

    if hints.dim() == 2:
        hints = hints.unsqueeze(0)
    pad = sil
    if pad.dim() == 2:
        pad = pad.unsqueeze(0)
    pad_chunk = pad[:, : target - good_frames, :]
    if pad_chunk.device != hints.device or pad_chunk.dtype != hints.dtype:
        pad_chunk = pad_chunk.to(device=hints.device, dtype=hints.dtype)
    repaired = torch.cat([hints[:, :good_frames, :], pad_chunk], dim=1)
    if hasattr(repaired, "contiguous"):
        repaired = repaired.contiguous()
    payload["precomputed_lm_hints_25Hz"] = repaired
    print(
        f"[ace_step_v1_5] LM code stream degenerates after ~{prefix_len} codes "
        f"(≈{prefix_len / 5.0:.1f}s): hint tail silence-padded "
        f"({good_frames}→{target} frames) — garbage tail codes sound like noise in ctx",
        flush=True,
    )


def build_non_cover_condition_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Mirror official turbo ``prepare_condition`` non-cover branch (silence ctx, SFT text prompt)."""
    nc_text = payload.get("non_cover_text_hidden_states")
    if nc_text is None:
        return None
    sil = payload.get("silence_latent")
    src = payload["src_latents"]
    if sil is None:
        return None
    sil_exp = sil[:, : src.shape[1], :].expand(src.shape[0], -1, -1)
    if hasattr(sil_exp, "clone"):
        sil_exp = sil_exp.clone()
    is_covers = payload["is_covers"]
    if hasattr(is_covers, "new_zeros"):
        non_is_covers = is_covers.new_zeros(is_covers.shape)
    else:
        non_is_covers = np.zeros_like(np.asarray(is_covers))
    nc_payload = dict(payload)
    nc_payload["text_hidden_states"] = nc_text
    nc_payload["text_attention_mask"] = payload["non_cover_text_attention_masks"]
    nc_payload["src_latents"] = sil_exp
    nc_payload["is_covers"] = non_is_covers
    nc_payload["precomputed_lm_hints_25Hz"] = None
    nc_payload["precomputed_lm_hints_25Hz_tt"] = None
    return nc_payload


def condition_encode_payload_tt(
    condition_encoder: Any,
    payload: dict,
    *,
    use_trace: bool,
) -> tuple[Any, np.ndarray, Any, Any]:
    """Run ``TtAceStepInstrumentalConditionEncoder.forward_payload`` (handler demo path).

    When ``use_trace`` is True, uses :meth:`forward_payload_traced` when available.
    """
    if use_trace and hasattr(condition_encoder, "forward_payload_traced"):
        return condition_encoder.forward_payload_traced(payload)
    return condition_encoder.forward_payload(payload)


def attach_infer_text_embeddings_ttnn(
    dit_handler: Any,
    *,
    tt_qwen_encoder: Any,
    max_seq_len: int = 256,
    use_trace: bool = False,
) -> Callable[[], None]:
    """Replace ``dit_handler.infer_text_embeddings`` with ``TtQwen3EmbeddingEncoder`` on TTNN.

    AceStep pads captions to ``max(text lengths in batch)`` (≤256); the TTNN encoder requires a fixed
    width of ``max_seq_len``, so IDs and masks are padded to ``max_seq_len`` for device forward and the
    result is cropped back to the original padded width ``S`` before ``prepare_condition``.
    Lyrics still use Torch ``embed_tokens`` from the loaded HF checkpoint.
    """
    import ttnn

    pad_id = getattr(dit_handler.text_tokenizer, "pad_token_id", None)
    dtype = getattr(dit_handler, "dtype", torch.float32)
    orig = dit_handler.infer_text_embeddings

    def _replacement(text_token_idss: torch.Tensor) -> torch.Tensor:
        ids = text_token_idss.detach()
        ids_cpu = ids.cpu()
        b, seq = int(ids_cpu.shape[0]), int(ids_cpu.shape[1])
        if seq > int(max_seq_len):
            raise ValueError(f"Text sequence length {seq} exceeds max_seq_len={max_seq_len}")
        attn_cpu = (
            ids_cpu.ne(pad_id).to(dtype=torch.float32)
            if pad_id is not None
            else torch.ones_like(ids_cpu, dtype=torch.float32)
        )
        ids_np = ids_cpu.numpy().astype(np.uint32)
        attn_np = attn_cpu.numpy().astype(np.float32)
        if seq < int(max_seq_len):
            pad_val = int(pad_id) if pad_id is not None else 0
            pad_w = int(max_seq_len) - seq
            ids_np = np.pad(ids_np, ((0, 0), (0, pad_w)), constant_values=pad_val).astype(np.uint32)
            attn_np = np.pad(attn_np, ((0, 0), (0, pad_w)), constant_values=np.float32(0.0)).astype(np.float32)
        hid_tt = qwen_caption_encode_tt(tt_qwen_encoder, ids_np, attn_np, use_trace=use_trace)
        out = ttnn.to_torch(hid_tt).float()
        if use_trace:
            ttnn.synchronize_device(tt_qwen_encoder.device)
            if hasattr(tt_qwen_encoder, "release_trace"):
                tt_qwen_encoder.release_trace()
        if out.ndim == 4:
            out = out.squeeze(1)
        out = out[:, :seq, :].contiguous()
        target_device = ids.device if ids.device.type != "meta" else torch.device("cpu")
        return out.to(device=target_device, dtype=dtype)

    dit_handler.infer_text_embeddings = _replacement

    def _restore() -> None:
        dit_handler.infer_text_embeddings = orig

    return _restore


def attach_payload_preprocess_ttnn(
    dit_handler: Any,
    *,
    tt_qwen_encoder: Any,
    tt_audio_detokenizer: Any | None = None,
    max_seq_len: int = 256,
    use_trace: bool = False,
) -> Callable[[], None]:
    """Move supported ``preprocess_batch`` tensor kernels to TTNN.

    The ACE handler still owns prompt formatting, tokenization, masks, and other
    Python control flow. This hook replaces the heavy tensor kernels it calls:

    * prompt Qwen forward via ``TtQwen3EmbeddingEncoder``;
    * lyric token embedding lookup via the same TTNN embedding table;
    * audio-code -> 25 Hz latent hints via TTNN detokenizer when provided.
    """
    import ttnn

    pad_id = getattr(dit_handler.text_tokenizer, "pad_token_id", None)
    dtype = getattr(dit_handler, "dtype", torch.float32)
    from models.experimental.ace_step_v1_5.utils.device_lm_hints import (
        ace_step_device_native_detok_hints,
        pad_lm_hints_tt_to_frames,
    )

    orig_text = dit_handler.infer_text_embeddings
    orig_lyric = dit_handler.infer_lyric_embeddings
    orig_decode = getattr(dit_handler, "_decode_audio_codes_to_latents", None)
    orig_prepare_hints = getattr(dit_handler, "_prepare_precomputed_lm_hints", None)
    dit_handler._ace_step_tt_audio_detokenizer = tt_audio_detokenizer
    dit_handler._ace_step_lm_hints_tt_batch = None

    def _text_replacement(text_token_idss: torch.Tensor) -> torch.Tensor:
        ids = text_token_idss.detach()
        ids_cpu = ids.cpu()
        b, seq = int(ids_cpu.shape[0]), int(ids_cpu.shape[1])
        if seq > int(max_seq_len):
            raise ValueError(f"Text sequence length {seq} exceeds max_seq_len={max_seq_len}")
        attn_cpu = (
            ids_cpu.ne(pad_id).to(dtype=torch.float32)
            if pad_id is not None
            else torch.ones_like(ids_cpu, dtype=torch.float32)
        )
        ids_np = ids_cpu.numpy().astype(np.uint32)
        attn_np = attn_cpu.numpy().astype(np.float32)
        if seq < int(max_seq_len):
            pad_val = int(pad_id) if pad_id is not None else 0
            pad_w = int(max_seq_len) - seq
            ids_np = np.pad(ids_np, ((0, 0), (0, pad_w)), constant_values=pad_val).astype(np.uint32)
            attn_np = np.pad(attn_np, ((0, 0), (0, pad_w)), constant_values=np.float32(0.0)).astype(np.float32)
        hid_tt = qwen_caption_encode_tt(tt_qwen_encoder, ids_np, attn_np, use_trace=use_trace)
        out = ttnn.to_torch(hid_tt).float()
        if use_trace:
            ttnn.synchronize_device(tt_qwen_encoder.device)
            if hasattr(tt_qwen_encoder, "release_trace"):
                tt_qwen_encoder.release_trace()
        if out.ndim == 4:
            out = out.squeeze(1)
        out = out[:, :seq, :].contiguous()
        target_device = ids.device if ids.device.type != "meta" else torch.device("cpu")
        return out.to(device=target_device, dtype=dtype)

    def _lyric_replacement(lyric_token_ids: torch.Tensor) -> torch.Tensor:
        ids = lyric_token_ids.detach()
        ids_np = ids.cpu().numpy().astype(np.uint32)
        if use_trace and hasattr(tt_qwen_encoder, "embed_tokens_traced"):
            hid_tt = tt_qwen_encoder.embed_tokens_traced(ids_np)
            ttnn.synchronize_device(tt_qwen_encoder.device)
        else:
            hid_tt = tt_qwen_encoder.embed_tokens(ids_np)
        out = ttnn.to_torch(hid_tt).float()
        if use_trace and hasattr(tt_qwen_encoder, "_release_embed_trace"):
            tt_qwen_encoder._release_embed_trace()
        target_device = ids.device if ids.device.type != "meta" else torch.device("cpu")
        return out.to(device=target_device, dtype=dtype)

    def _decode_replacement(code_str: str):
        from models.experimental.ace_step_v1_5.ttnn_impl.audio_code_detokenizer import parse_audio_code_string
        from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_detok_chunk_n

        from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
            ace_step_detok_chunk_n,
            ace_step_use_pytorch_detok,
        )

        n_codes = len(parse_audio_code_string(code_str or ""))
        chunk_n = ace_step_detok_chunk_n()
        use_pytorch_detok = ace_step_use_pytorch_detok(n_codes=n_codes)
        if use_pytorch_detok and orig_decode is not None:
            logger.info(
                "[detokenizer] ACE_STEP_PYTORCH_DETOK=1: HF PyTorch detokenizer for {} codes",
                n_codes,
            )
            return orig_decode(code_str)
        if tt_audio_detokenizer is None:
            if orig_decode is None:
                return None
            return orig_decode(code_str)
        if n_codes > chunk_n:
            n_forwards = (n_codes + chunk_n - 1) // chunk_n
            logger.info(
                "[detokenizer] {} audio codes → TTNN chunked detokenizer ({} forwards, {} codes/forward)",
                n_codes,
                n_forwards,
                chunk_n,
            )
        try:
            if use_trace and hasattr(tt_audio_detokenizer, "forward_traced"):
                hid_tt = tt_audio_detokenizer.forward_traced(code_str)
            else:
                hid_tt = tt_audio_detokenizer.forward(code_str)
        except Exception as exc:
            raise RuntimeError(
                f"TTNN detokenizer failed for {n_codes} audio codes "
                f"(chunk={chunk_n}; set ACE_STEP_PYTORCH_DETOK=1 for HF fallback): {exc}"
            ) from exc
        if hid_tt is None:
            return None
        dit_handler._ace_step_last_detok_hints_tt = hid_tt
        if ace_step_device_native_detok_hints() and tt_audio_detokenizer is not None:
            if use_trace and hasattr(tt_audio_detokenizer, "release_trace"):
                ttnn.synchronize_device(tt_audio_detokenizer.device)
                tt_audio_detokenizer.release_trace()
            return None
        out = ttnn.to_torch(hid_tt).float()
        if use_trace and hasattr(tt_audio_detokenizer, "release_trace"):
            ttnn.synchronize_device(tt_audio_detokenizer.device)
            tt_audio_detokenizer.release_trace()
        return out.to(device=getattr(dit_handler, "device", torch.device("cpu")), dtype=dtype)

    def _prepare_hints_replacement(
        self,
        batch_size: int,
        audio_code_hints: list,
        max_latent_length: int,
        silence_latent_tiled: torch.Tensor,
    ):
        if not ace_step_device_native_detok_hints() or tt_audio_detokenizer is None or orig_prepare_hints is None:
            return orig_prepare_hints(
                batch_size,
                audio_code_hints,
                max_latent_length,
                silence_latent_tiled,
            )

        tt_dev = tt_audio_detokenizer.device
        tt_mem = getattr(tt_audio_detokenizer, "mem", None)
        tt_dtype = getattr(tt_audio_detokenizer, "dtype", ttnn.float32)
        hints_tt_list: list[Any | None] = []
        for i in range(batch_size):
            if audio_code_hints[i] is None:
                hints_tt_list.append(None)
                continue
            logger.info(f"[generate_music] Decoding audio codes for LM hints for item {i}...")
            self._decode_audio_codes_to_latents(audio_code_hints[i])
            hid_tt = getattr(self, "_ace_step_last_detok_hints_tt", None)
            if hid_tt is None:
                hints_tt_list.append(None)
                continue
            hid_tt = pad_lm_hints_tt_to_frames(
                hid_tt,
                max_frames=int(max_latent_length),
                silence_latent=silence_latent_tiled,
                device=tt_dev,
                mem=tt_mem,
                dtype=tt_dtype,
            )
            hints_tt_list.append(hid_tt)

        if not any(h is not None for h in hints_tt_list):
            self._ace_step_lm_hints_tt_batch = None
            return None

        if batch_size != 1:
            logger.warning(
                "[ace_step_v1_5] device-native LM hints support B=1 only; falling back to torch hints",
            )
            return orig_prepare_hints(
                batch_size,
                audio_code_hints,
                max_latent_length,
                silence_latent_tiled,
            )

        self._ace_step_lm_hints_tt_batch = hints_tt_list[0]
        return None

    dit_handler.infer_text_embeddings = _text_replacement
    dit_handler.infer_lyric_embeddings = _lyric_replacement
    if orig_decode is not None:
        dit_handler._decode_audio_codes_to_latents = _decode_replacement
    if orig_prepare_hints is not None:
        dit_handler._prepare_precomputed_lm_hints = _prepare_hints_replacement.__get__(dit_handler, type(dit_handler))

    def _restore() -> None:
        dit_handler.infer_text_embeddings = orig_text
        dit_handler.infer_lyric_embeddings = orig_lyric
        if orig_decode is not None:
            dit_handler._decode_audio_codes_to_latents = orig_decode
        if orig_prepare_hints is not None:
            dit_handler._prepare_precomputed_lm_hints = orig_prepare_hints
        if hasattr(dit_handler, "_ace_step_tt_audio_detokenizer"):
            delattr(dit_handler, "_ace_step_tt_audio_detokenizer")
        if hasattr(dit_handler, "_ace_step_lm_hints_tt_batch"):
            delattr(dit_handler, "_ace_step_lm_hints_tt_batch")
        if hasattr(dit_handler, "_ace_step_last_detok_hints_tt"):
            delattr(dit_handler, "_ace_step_last_detok_hints_tt")

    return _restore


def configure_acestep_logging(
    *,
    level: str | None = None,
    show_ttnn_tensor_cache: bool = False,
) -> None:
    """Match CLI-style loguru lines (time | level | module:function:line - message).

    Default is ``INFO`` so TTNN ``core:as_tensor`` per-tensor flatbuffer cache lines stay hidden.
    Pass ``show_ttnn_tensor_cache=True`` (demo ``--verbose``) to include those DEBUG lines.
    """
    if level is None:
        level = os.environ.get("ACE_STEP_LOG_LEVEL", "INFO")

    def _filter(record: dict) -> bool:
        if show_ttnn_tensor_cache:
            return True
        if record["name"] == "core" and (
            "Loaded cache for" in record["message"] or "Generating cache for" in record["message"]
        ):
            return False
        return True

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        filter=_filter,
    )


def build_filtered_dit_kwargs_for_handler(
    dit_handler: Any,
    llm_handler: Any,
    params: Any,
    config: Any,
    progress: Any = None,
) -> dict[str, Any]:
    """
    Same LM + ``dit_generate_kwargs`` filtering as the upstream ACE-Step CLI Phase 1,
    without invoking ``dit_handler.generate_music``.
    """
    # --- Phase 1: LM metadata + kwargs (adapted from upstream ACE-Step inference) ---
    if params.task_type == "text2music" and params.flow_edit_morph:
        audio_code_string_to_use = ""
    else:
        audio_code_string_to_use = params.audio_codes
    lm_generated_metadata = None
    lm_generated_audio_codes_list: list[Any] = []
    lm_total_time_costs = {"phase1_time": 0.0, "phase2_time": 0.0, "total_time": 0.0}
    lm_total_tokens = 0

    bpm = params.bpm
    key_scale = params.keyscale
    time_signature = params.timesignature
    audio_duration = params.duration
    dit_input_caption = params.caption
    dit_input_vocal_language = params.vocal_language
    dit_input_lyrics = params.lyrics

    from models.experimental.ace_step_v1_5.utils.acestep_preprocess_shim import (
        _load_cached_repaint_source,
        _resample_matching_source_seeds,
        _update_metadata_from_lm,
    )

    cached_repaint_source = _load_cached_repaint_source(params.src_audio) if params.task_type == "repaint" else None
    source_repaint_latents = cached_repaint_source.latents if cached_repaint_source is not None else None

    user_provided_audio_codes = bool(params.audio_codes and str(params.audio_codes).strip())
    need_audio_codes = not user_provided_audio_codes
    actual_batch_size = config.batch_size if config.batch_size is not None else 1

    seed_for_generation = ""
    if config.seeds is not None:
        if isinstance(config.seeds, list) and len(config.seeds) > 0:
            seed_for_generation = ",".join(str(s) for s in config.seeds)
        elif isinstance(config.seeds, int):
            seed_for_generation = str(config.seeds)

    actual_seed_list, _ = dit_handler.prepare_seeds(actual_batch_size, seed_for_generation, config.use_random_seed)
    use_random_seed_for_dit = config.use_random_seed
    if cached_repaint_source is not None:
        actual_seed_list = _resample_matching_source_seeds(actual_seed_list, cached_repaint_source.source_seed)
        seed_for_generation = ",".join(str(seed) for seed in actual_seed_list)
        use_random_seed_for_dit = False

    skip_lm_tasks = {"cover", "cover-nofsq", "repaint", "extract"}
    morph_on_text2music = params.task_type == "text2music" and params.flow_edit_morph
    need_lm_for_cot = params.use_cot_caption or params.use_cot_language or params.use_cot_metas
    skip_lm = params.task_type in skip_lm_tasks or morph_on_text2music
    use_lm = (
        (params.thinking or need_lm_for_cot) and llm_handler is not None and llm_handler.llm_initialized and not skip_lm
    )

    if skip_lm:
        reason = params.task_type if params.task_type in skip_lm_tasks else f"{params.task_type}+flow_edit_morph"
        logger.info(f"Skipping LM for task_type='{reason}' - using DiT directly")

    logger.info(
        f"[generate_music] LLM usage decision: thinking={params.thinking}, "
        f"use_cot_caption={params.use_cot_caption}, use_cot_language={params.use_cot_language}, "
        f"use_cot_metas={params.use_cot_metas}, need_lm_for_cot={need_lm_for_cot}, "
        f"llm_initialized={llm_handler.llm_initialized if llm_handler else False}, use_lm={use_lm}"
    )

    if use_lm:
        top_k_value = None if not params.lm_top_k or params.lm_top_k == 0 else int(params.lm_top_k)
        top_p_value = None if not params.lm_top_p or params.lm_top_p >= 1.0 else params.lm_top_p

        user_metadata = {}
        if bpm is not None:
            try:
                bpm_value = float(bpm)
                if bpm_value > 0:
                    user_metadata["bpm"] = int(bpm_value)
            except (ValueError, TypeError):
                pass
        if key_scale and key_scale.strip():
            key_scale_clean = key_scale.strip()
            if key_scale_clean.lower() not in ["n/a", ""]:
                user_metadata["keyscale"] = key_scale_clean
        if time_signature and time_signature.strip():
            time_sig_clean = time_signature.strip()
            if time_sig_clean.lower() not in ["n/a", ""]:
                user_metadata["timesignature"] = time_sig_clean
        requested_duration_sec: float | None = None
        if audio_duration is not None:
            try:
                duration_value = float(audio_duration)
                if duration_value > 0:
                    requested_duration_sec = float(duration_value)
                    # Pin duration only — do not pre-fill bpm/key/timesig or Phase 1 CoT is skipped
                    # (has_all_metas) and caption/language refinement is lost vs the June benchmark path.
                    user_metadata["duration"] = int(round(duration_value))
            except (ValueError, TypeError):
                pass
        user_metadata_to_pass = user_metadata if user_metadata else None

        infer_type = "llm_dit" if need_audio_codes and params.thinking else "dit"
        max_inference_batch_size = (
            int(config.lm_batch_chunk_size) if config.lm_batch_chunk_size > 0 else actual_batch_size
        )
        num_chunks = math.ceil(actual_batch_size / max_inference_batch_size)

        all_metadata_list: list[Any] = []
        all_audio_codes_list: list[Any] = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * max_inference_batch_size
            chunk_end = min(chunk_start + max_inference_batch_size, actual_batch_size)
            chunk_size = chunk_end - chunk_start
            chunk_seeds = actual_seed_list[chunk_start:chunk_end] if chunk_start < len(actual_seed_list) else None

            logger.info(
                f"LM chunk {chunk_idx+1}/{num_chunks} (infer_type={infer_type}) "
                f"(size: {chunk_size}, seeds: {chunk_seeds})"
            )

            result = llm_handler.generate_with_stop_condition(
                caption=params.caption or "",
                lyrics=params.lyrics or "",
                infer_type=infer_type,
                temperature=params.lm_temperature,
                cfg_scale=params.lm_cfg_scale,
                negative_prompt=params.lm_negative_prompt,
                top_k=top_k_value,
                top_p=top_p_value,
                repetition_penalty=float(getattr(params, "lm_repetition_penalty", 1.0)),
                target_duration=requested_duration_sec if requested_duration_sec is not None else audio_duration,
                user_metadata=user_metadata_to_pass,
                use_cot_caption=params.use_cot_caption,
                use_cot_language=params.use_cot_language,
                use_cot_metas=params.use_cot_metas,
                use_constrained_decoding=params.use_constrained_decoding,
                constrained_decoding_debug=config.constrained_decoding_debug,
                batch_size=chunk_size,
                seeds=chunk_seeds,
                progress=progress,
            )

            if not result.get("success", False):
                error_msg = result.get("error", "Unknown LM error")
                raise RuntimeError(f"LM generation failed: {error_msg}")

            if chunk_size > 1:
                metadata_list = result.get("metadata", [])
                audio_codes_list = result.get("audio_codes", [])
                all_metadata_list.extend(metadata_list)
                all_audio_codes_list.extend(audio_codes_list)
            else:
                metadata = result.get("metadata", {})
                audio_codes = result.get("audio_codes", "")
                all_metadata_list.append(metadata)
                all_audio_codes_list.append(audio_codes)

            if requested_duration_sec is not None and infer_type == "llm_dit":
                from models.experimental.ace_step_v1_5.ttnn_impl.audio_code_detokenizer import parse_audio_code_string

                codes_str = (
                    all_audio_codes_list[-1]
                    if all_audio_codes_list and isinstance(all_audio_codes_list[-1], str)
                    else ""
                )
                n_codes = len(parse_audio_code_string(str(codes_str or "")))
                expected_codes = max(1, int(round(requested_duration_sec * 5.0)))
                print(
                    f"[ace_step_v1_5] LM audio codes: {n_codes} (expect~{expected_codes} for "
                    f"{requested_duration_sec:g}s)",
                    flush=True,
                )
                if n_codes + 2 < expected_codes:
                    print(
                        f"[ace_step_v1_5] WARNING: LM produced only {n_codes}/{expected_codes} audio codes for "
                        f"{requested_duration_sec:g}s — DiT duration will be clamped to ~{n_codes / 5.0:.1f}s. "
                        "Check logs for 'Phase 1: Using user-provided metadata' and "
                        f"'Phase 2 audio codes: target_duration={requested_duration_sec:g}s'.",
                        flush=True,
                    )

            lm_extra = result.get("extra_outputs", {})
            lm_chunk_time_costs = lm_extra.get("time_costs", {})
            if lm_chunk_time_costs:
                for key in ["phase1_time", "phase2_time", "total_time"]:
                    if key in lm_chunk_time_costs:
                        lm_total_time_costs[key] += lm_chunk_time_costs[key]
            lm_total_tokens += int(lm_extra.get("num_tokens", 0) or 0)

        lm_generated_metadata = all_metadata_list[0] if all_metadata_list else None
        lm_generated_audio_codes_list = all_audio_codes_list

        if infer_type == "llm_dit":
            if actual_batch_size > 1:
                audio_code_string_to_use = all_audio_codes_list
            else:
                audio_code_string_to_use = all_audio_codes_list[0] if all_audio_codes_list else ""
        else:
            audio_code_string_to_use = params.audio_codes

        if lm_generated_metadata:
            bpm, key_scale, time_signature, audio_duration, vocal_language, caption, lyrics = _update_metadata_from_lm(
                metadata=lm_generated_metadata,
                bpm=bpm,
                key_scale=key_scale,
                time_signature=time_signature,
                audio_duration=audio_duration,
                vocal_language=dit_input_vocal_language,
                caption=dit_input_caption,
                lyrics=dit_input_lyrics,
            )
            if (not params.bpm or params.bpm <= 0) and bpm and int(bpm) > 0:
                params.cot_bpm = bpm
            if not params.keyscale:
                params.cot_keyscale = key_scale
            if not params.timesignature:
                params.cot_timesignature = time_signature
            if (not params.duration or params.duration <= 0) and audio_duration and float(audio_duration) > 0:
                params.cot_duration = audio_duration
            if not params.vocal_language:
                params.cot_vocal_language = vocal_language
            if not params.caption:
                params.cot_caption = caption
            if not params.lyrics:
                params.cot_lyrics = lyrics

        lm_cot_caption: str | None = None
        if lm_generated_metadata is not None:
            lm_cot_caption = str(lm_generated_metadata.get("caption") or "").strip() or None
            if params.use_cot_caption and lm_cot_caption:
                dit_input_caption = lm_cot_caption
            if params.use_cot_language:
                dit_input_vocal_language = lm_generated_metadata.get("vocal_language", dit_input_vocal_language)

        user_cap = str(params.caption or "").strip()
        if user_cap:
            merged_cap, missing = merge_user_instruments_into_caption(user_cap, str(dit_input_caption or ""))
            if missing:
                dit_input_caption = merged_cap
                if lm_generated_metadata is not None:
                    lm_generated_metadata = dict(lm_generated_metadata)
                    lm_generated_metadata["caption"] = merged_cap
                print(
                    f"[ace_step_v1_5] caption: merged user instruments {missing} into CoT/DiT caption",
                    flush=True,
                )
            log_caption_conditioning_trace(
                user_caption=user_cap,
                lm_cot_caption=lm_cot_caption if lm_generated_metadata else None,
                final_caption=str(dit_input_caption or ""),
                use_cot_caption=bool(params.use_cot_caption),
            )

        if llm_handler is not None:
            llm_handler.last_lm_perf = {
                "lm_num_tokens": int(lm_total_tokens),
                "lm_gen_time_s": float(lm_total_time_costs.get("total_time", 0.0)),
                "lm_phase1_time_s": float(lm_total_time_costs.get("phase1_time", 0.0)),
                "lm_phase2_time_s": float(lm_total_time_costs.get("phase2_time", 0.0)),
                "lm_session_warmup_time_s": float(getattr(llm_handler, "_lm_session_warmup_time_s", 0.0)),
                "lm_init_warmup_time_s": float(getattr(llm_handler, "_lm_init_warmup_time_s", 0.0)),
            }

    if params.task_type in ("repaint", "cover", "cover-nofsq", "extract"):
        dit_input_caption = params.caption or dit_input_caption
        dit_input_lyrics = params.lyrics if params.lyrics is not None else dit_input_lyrics
        logger.info(
            f"[generate_music] {params.task_type} task: using params.caption='{params.caption}', params.lyrics='{params.lyrics}'"
        )
        logger.info(
            f"[generate_music] Final inputs: dit_input_caption='{dit_input_caption}', dit_input_lyrics='{dit_input_lyrics}'"
        )

    if params.task_type in ("cover", "cover-nofsq", "repaint", "lego", "extract"):
        audio_duration = None

    dit_generate_kwargs = {
        "captions": dit_input_caption,
        "global_caption": params.global_caption,
        "lyrics": dit_input_lyrics,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "vocal_language": dit_input_vocal_language,
        "inference_steps": params.inference_steps,
        "guidance_scale": params.guidance_scale,
        "use_random_seed": use_random_seed_for_dit,
        "seed": seed_for_generation,
        "reference_audio": params.reference_audio,
        "audio_duration": audio_duration,
        "batch_size": config.batch_size if config.batch_size is not None else 1,
        "src_audio": (params.src_audio if params.task_type != "text2music" or params.flow_edit_morph else None),
        "audio_code_string": audio_code_string_to_use,
        "repainting_start": params.repainting_start,
        "repainting_end": params.repainting_end,
        "chunk_mask_mode": params.chunk_mask_mode,
        "repaint_latent_crossfade_frames": params.repaint_latent_crossfade_frames,
        "repaint_wav_crossfade_sec": params.repaint_wav_crossfade_sec,
        "repaint_mode": params.repaint_mode,
        "repaint_strength": params.repaint_strength,
        "source_repaint_latents": source_repaint_latents,
        "retake_seed": params.retake_seed,
        "retake_variance": params.retake_variance,
        "flow_edit_morph": params.flow_edit_morph,
        "flow_edit_source_caption": params.flow_edit_source_caption,
        "flow_edit_source_lyrics": params.flow_edit_source_lyrics,
        "flow_edit_n_min": params.flow_edit_n_min,
        "flow_edit_n_max": params.flow_edit_n_max,
        "flow_edit_n_avg": params.flow_edit_n_avg,
        "instruction": params.instruction,
        "audio_cover_strength": params.audio_cover_strength,
        "cover_noise_strength": params.cover_noise_strength,
        "task_type": params.task_type,
        "use_adg": params.use_adg,
        "cfg_interval_start": params.cfg_interval_start,
        "cfg_interval_end": params.cfg_interval_end,
        "shift": params.shift,
        "infer_method": params.infer_method,
        "sampler_mode": params.sampler_mode,
        "velocity_norm_threshold": params.velocity_norm_threshold,
        "velocity_ema_factor": params.velocity_ema_factor,
        "dcw_enabled": params.dcw_enabled,
        "dcw_mode": params.dcw_mode,
        "dcw_scaler": params.dcw_scaler,
        "dcw_high_scaler": params.dcw_high_scaler,
        "dcw_wavelet": params.dcw_wavelet,
        "timesteps": params.timesteps,
        "latent_shift": params.latent_shift,
        "latent_rescale": params.latent_rescale,
        "progress": progress,
    }
    supported_generate_keys = set(inspect.signature(dit_handler.generate_music).parameters.keys())
    filtered = {k: v for k, v in dit_generate_kwargs.items() if k in supported_generate_keys}
    dropped = sorted(set(dit_generate_kwargs.keys()) - supported_generate_keys)
    if dropped:
        logger.warning(f"[generate_music] Skipping unsupported generate_music kwargs: {dropped}")
    return filtered


def handler_prepare_condition_payload(
    dit_handler: Any,
    filtered_generate_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """
    Run the same early path as ``AceStepHandler.generate_music`` through
    ``preprocess_batch`` and return the unpacked payload before
    ``model.prepare_condition`` (no diffusion).
    """
    progress = dit_handler._resolve_generate_music_progress(filtered_generate_kwargs.get("progress"))
    readiness = dit_handler._validate_generate_music_readiness()
    if readiness is not None:
        raise RuntimeError(readiness.get("error", "handler not ready"))

    task_type, instruction = dit_handler._resolve_generate_music_task(
        task_type=filtered_generate_kwargs["task_type"],
        audio_code_string=filtered_generate_kwargs["audio_code_string"],
        instruction=filtered_generate_kwargs["instruction"],
    )
    guidance_scale = float(filtered_generate_kwargs["guidance_scale"])
    if dit_handler.is_turbo_model() and guidance_scale != 1.0:
        logger.info(
            "[generate_music] Turbo model detected: overriding "
            "guidance_scale {:.1f} -> 1.0 (turbo does not use CFG).",
            guidance_scale,
        )
        guidance_scale = 1.0

    if getattr(dit_handler, "lora_loaded", False) and getattr(dit_handler, "use_lora", False):
        dit_handler._verify_decoder_device_dtype()

    logger.info("[generate_music] Starting generation...")
    if progress:
        progress(0.51, desc="Preparing inputs...")
    logger.info("[generate_music] Preparing inputs...")

    runtime = dit_handler._prepare_generate_music_runtime(
        batch_size=filtered_generate_kwargs.get("batch_size"),
        audio_duration=filtered_generate_kwargs.get("audio_duration"),
        repainting_end=filtered_generate_kwargs.get("repainting_end"),
        seed=filtered_generate_kwargs.get("seed"),
        use_random_seed=filtered_generate_kwargs.get("use_random_seed", True),
        retake_seed=filtered_generate_kwargs.get("retake_seed"),
        retake_variance=filtered_generate_kwargs.get("retake_variance", 0.0),
    )
    actual_batch_size = runtime["actual_batch_size"]
    actual_seed_list = runtime["actual_seed_list"]
    audio_duration = runtime["audio_duration"]
    repainting_end = runtime["repainting_end"]

    refer_audios, processed_src_audio, audio_error = dit_handler._prepare_reference_and_source_audio(
        reference_audio=filtered_generate_kwargs.get("reference_audio"),
        src_audio=filtered_generate_kwargs.get("src_audio"),
        audio_code_string=filtered_generate_kwargs["audio_code_string"],
        actual_batch_size=actual_batch_size,
        task_type=task_type,
        flow_edit_morph=filtered_generate_kwargs.get("flow_edit_morph", False),
    )
    if audio_error is not None:
        raise RuntimeError(audio_error.get("error", str(audio_error)))

    if processed_src_audio is not None and (
        task_type in ("cover", "cover-nofsq", "repaint", "lego", "extract")
        or (task_type == "text2music" and filtered_generate_kwargs.get("flow_edit_morph"))
    ):
        audio_duration = processed_src_audio.shape[-1] / dit_handler.sample_rate

    if filtered_generate_kwargs.get("flow_edit_morph") and task_type not in ("text2music", "cover", "cover-nofsq"):
        logger.warning(
            "[generate_music] flow_edit_morph=True but task_type={!r}; "
            "v1 overlay only applies to text2music / cover / cover-nofsq, ignoring.",
            task_type,
        )

    service_inputs = dit_handler._prepare_generate_music_service_inputs(
        actual_batch_size=actual_batch_size,
        processed_src_audio=processed_src_audio,
        audio_duration=audio_duration,
        captions=filtered_generate_kwargs["captions"],
        global_caption=filtered_generate_kwargs.get("global_caption") or "",
        lyrics=filtered_generate_kwargs["lyrics"],
        vocal_language=filtered_generate_kwargs["vocal_language"],
        instruction=instruction,
        bpm=filtered_generate_kwargs.get("bpm"),
        key_scale=filtered_generate_kwargs.get("key_scale") or "",
        time_signature=filtered_generate_kwargs.get("time_signature") or "",
        task_type=task_type,
        audio_code_string=filtered_generate_kwargs["audio_code_string"],
        repainting_start=filtered_generate_kwargs.get("repainting_start", 0.0),
        repainting_end=repainting_end,
        chunk_mask_mode=filtered_generate_kwargs.get("chunk_mask_mode", "auto"),
    )

    if not getattr(dit_handler, "preprocess_only", False):
        vram_error = dit_handler._vram_preflight_check(
            actual_batch_size=actual_batch_size,
            audio_duration=audio_duration,
            guidance_scale=guidance_scale,
        )
        if vram_error is not None:
            raise RuntimeError(vram_error.get("error", "VRAM preflight failed"))

    normalized = dit_handler._normalize_service_generate_inputs(
        captions=service_inputs["captions_batch"],
        lyrics=service_inputs["lyrics_batch"],
        keys=None,
        metas=service_inputs["metas_batch"],
        vocal_languages=service_inputs["vocal_languages_batch"],
        repainting_start=service_inputs["repainting_start_batch"],
        repainting_end=service_inputs["repainting_end_batch"],
        instructions=service_inputs["instructions_batch"],
        audio_code_hints=service_inputs["audio_code_hints_batch"],
        infer_steps=int(filtered_generate_kwargs["inference_steps"]),
        seed=actual_seed_list,
        return_intermediate=service_inputs["should_return_intermediate"],
    )
    batch = dit_handler._prepare_batch(
        captions=normalized["captions"],
        global_captions=service_inputs.get("global_captions_batch"),
        lyrics=normalized["lyrics"],
        keys=normalized["keys"],
        target_wavs=service_inputs["target_wavs_tensor"],
        refer_audios=refer_audios,
        metas=normalized["metas"],
        vocal_languages=normalized["vocal_languages"],
        repainting_start=normalized["repainting_start"],
        repainting_end=normalized["repainting_end"],
        instructions=normalized["instructions"],
        audio_code_hints=normalized["audio_code_hints"],
        audio_cover_strength=filtered_generate_kwargs.get("audio_cover_strength", 1.0),
        cover_noise_strength=filtered_generate_kwargs.get("cover_noise_strength", 0.0),
        chunk_mask_modes=service_inputs.get("chunk_mask_modes_batch"),
        task_type=task_type,
        source_repaint_latents=filtered_generate_kwargs.get("source_repaint_latents"),
    )
    processed = dit_handler.preprocess_batch(batch)
    payload = dit_handler._unpack_service_processed_data(processed)
    payload["silence_latent"] = dit_handler.silence_latent
    hints_tt = getattr(dit_handler, "_ace_step_lm_hints_tt_batch", None)
    if hints_tt is not None:
        payload["precomputed_lm_hints_25Hz_tt"] = hints_tt
        payload["precomputed_lm_hints_25Hz"] = None

    frames = int(payload["src_latents"].shape[1])
    repair_degenerate_lm_hint_tail(
        payload,
        str(filtered_generate_kwargs.get("audio_code_string") or ""),
        int(frames),
        tt_audio_detokenizer=getattr(dit_handler, "_ace_step_tt_audio_detokenizer", None),
    )
    return payload, frames


def handler_prepare_condition_from_payload(
    dit_handler: Any,
    payload: dict[str, Any],
    *,
    frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """PyTorch ``prepare_condition`` from an existing preprocess payload (no second LM pass)."""
    frames_i = int(frames if frames is not None else payload["src_latents"].shape[1])
    with torch.inference_mode():
        enc_hs, enc_mask, ctx = dit_handler.model.prepare_condition(
            text_hidden_states=payload["text_hidden_states"],
            text_attention_mask=payload["text_attention_mask"],
            lyric_hidden_states=payload["lyric_hidden_states"],
            lyric_attention_mask=payload["lyric_attention_mask"],
            refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
            refer_audio_order_mask=payload["refer_audio_order_mask"],
            hidden_states=payload["src_latents"],
            attention_mask=torch.ones(
                payload["src_latents"].shape[0],
                payload["src_latents"].shape[1],
                device=payload["src_latents"].device,
                dtype=payload["src_latents"].dtype,
            ),
            silence_latent=dit_handler.silence_latent,
            src_latents=payload["src_latents"],
            chunk_masks=payload["chunk_mask"],
            is_covers=payload["is_covers"],
            precomputed_lm_hints_25Hz=payload["precomputed_lm_hints_25Hz"],
        )

    null_emb = getattr(dit_handler.model, "null_condition_emb", None)
    if null_emb is None:
        raise RuntimeError("null_condition_emb missing on handler.model")

    enc_hs = enc_hs.detach().float().cpu()
    enc_mask = enc_mask.detach().float().cpu()
    ctx = ctx.detach().float().cpu()
    null_emb = null_emb.detach().float().cpu()

    return enc_hs, enc_mask, ctx, frames_i, null_emb


def handler_prepare_condition_tensors(
    dit_handler: Any,
    filtered_generate_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Run the same early path as ``AceStepHandler.generate_music`` through
    ``preprocess_batch`` and ``model.prepare_condition`` (no diffusion).
    """
    payload, frames = handler_prepare_condition_payload(dit_handler, filtered_generate_kwargs)

    dit_backend = (
        "MLX (native)"
        if (getattr(dit_handler, "use_mlx_dit", False) and getattr(dit_handler, "mlx_decoder", None) is not None)
        else f"PyTorch ({dit_handler.device})"
    )
    logger.info(f"[service_generate] Generating audio... (DiT backend: {dit_backend})")
    return handler_prepare_condition_from_payload(dit_handler, payload, frames=int(frames))
