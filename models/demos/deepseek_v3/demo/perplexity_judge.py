# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Perplexity scoring for *text* from another model using a compact HF causal LM.

Used to sanity-check Tenstorrent DeepSeek decode: teacher-forced accuracy can
stay high while autoregressive sampling drifts into garbage, because forcing
injects the HF token stream each step. A separate pretrained judge assigns
high perplexity to incoherent continuations; comparing TT autoregressive text
to the HF reference continuation (same prompt, same tokenizer decode) yields
a stable relative signal without loading DeepSeek on CPU again.

Default judge is a recent mid-scale model (see ``load_judge_lm``).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"


def _resolve_trust_remote_code() -> bool:
    env = os.getenv("DEEPSEEK_PPL_JUDGE_TRUST_REMOTE_CODE")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return True


@lru_cache(maxsize=16)
def _load_judge_cached(
    model_id: str, dtype_name: str, trust_remote: str
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    trust = trust_remote in ("1", "true", "yes")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    _ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust,
    )
    model.eval()
    return model, tokenizer


def load_judge_lm(
    model_id: str | None = None,
    *,
    judge_model_id: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
    """Load the HF judge. ``judge_model_id`` is an alias for the first positional ``model_id``."""
    if model_id is not None and judge_model_id is not None:
        raise TypeError("load_judge_lm() accepts at most one of model_id and judge_model_id")
    chosen = model_id if model_id is not None else judge_model_id
    resolved_id = chosen or os.getenv("DEEPSEEK_PPL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    dev = torch.device(device or os.getenv("DEEPSEEK_PPL_JUDGE_DEVICE", "cpu"))
    trust_remote = _resolve_trust_remote_code()
    trust_key = "true" if trust_remote else "false"
    dtype_env = os.getenv("DEEPSEEK_PPL_JUDGE_DTYPE")
    if dtype is None and dtype_env:
        if not hasattr(torch, dtype_env):
            raise ValueError(
                f"Unsupported DEEPSEEK_PPL_JUDGE_DTYPE='{dtype_env}'. "
                "Use a torch dtype name such as float32, float16, or bfloat16."
            )
        dtype = getattr(torch, dtype_env)
    if dtype is None:
        # Perplexity is a ranking metric; prefer numerically stable defaults.
        # Override with DEEPSEEK_PPL_JUDGE_DTYPE when memory/throughput requires.
        dtype = torch.float32 if dev.type == "cpu" else torch.bfloat16
    if dev.type == "cpu" and dtype == torch.float16:
        logger.warning(
            "Using float16 on CPU for judge scoring can reduce perplexity stability. "
            "Prefer float32 unless memory is constrained."
        )
    dtype_name = str(dtype).split(".")[-1]
    model, tokenizer = _load_judge_cached(resolved_id, dtype_name, trust_key)
    model = model.to(dev)
    logger.info(
        "Loaded perplexity judge '{}' on {} ({}, trust_remote_code={})",
        resolved_id,
        dev,
        dtype,
        trust_remote,
    )
    return model, tokenizer, dev


def _encode_ids(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _prompt_token_len_in_full(prompt_ids: list[int], full_ids: list[int]) -> int:
    if not prompt_ids:
        return 0
    if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    n = min(len(prompt_ids), len(full_ids))
    i = 0
    while i < n and prompt_ids[i] == full_ids[i]:
        i += 1
    return i


def mean_nll_continuation_with_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    *,
    prompt: str,
    continuation: str,
    max_length: int | None = None,
) -> float:
    """
    Average next-token negative log-likelihood over continuation tokens only,
    conditioned on prompt tokens in the same encoding window.

    When the concatenated sequence exceeds ``max_length``, the **oldest** tokens
    are dropped (tail window) so the score still emphasizes recent continuation
    while keeping prompt masking consistent for HF vs TT comparisons.

    If ``max_length`` is None, reads ``DEEPSEEK_PPL_JUDGE_MAX_LENGTH`` (default 2048).
    """
    if max_length is None:
        max_length = int(os.getenv("DEEPSEEK_PPL_JUDGE_MAX_LENGTH", "2048"))
    full_text = f"{prompt}{continuation}"
    p_ids = _encode_ids(tokenizer, prompt)
    f_ids = _encode_ids(tokenizer, full_text)
    if len(f_ids) < 2:
        return float("nan")

    plen = _prompt_token_len_in_full(p_ids, f_ids)
    if len(f_ids) > max_length:
        dropped = len(f_ids) - max_length
        f_ids = f_ids[-max_length:]
        plen = max(0, plen - dropped)

    input_ids = torch.tensor([f_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    if 0 < plen < input_ids.shape[1]:
        labels[:, :plen] = -100

    with torch.inference_mode():
        out = model(input_ids=input_ids, labels=labels)

    loss = out.loss
    if loss is None or not math.isfinite(float(loss)):
        return float("nan")
    return float(loss)


def perplexity_from_mean_nll(mean_nll: float) -> float:
    if not math.isfinite(mean_nll):
        return float("nan")
    return float(math.exp(mean_nll))
