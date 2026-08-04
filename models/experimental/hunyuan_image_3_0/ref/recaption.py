# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Host orchestration for HunyuanImage-3.0 Instruct autoregressive recaption / think.
#
# Step 1 (tokenizer prefix): ``prepare_recaption_inputs`` in gen_image_inputs.py
# Step 2 (this module): stage transitions + AR decode + ``cot_text`` extraction
#
# Mirrors upstream ``HunyuanImage3ForCausalMM.generate_image`` recaption block
# (modeling_hunyuan_image_3.py ~3283–3371).

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import torch

from models.experimental.hunyuan_image_3_0.ref.generate import (
    ConditionalSliceVocabLogitsProcessor,
    SamplingConfig,
    generate_text,
)
from models.experimental.hunyuan_image_3_0.ref.model_config import IMAGE_BASE_SIZE
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import GenImageHostInputs
from models.experimental.hunyuan_image_3_0.ref.tokenizer.hunyuan_tokenizer import HunyuanTokenizer
from models.experimental.hunyuan_image_3_0.ref.tokenizer.special_tokens import SpecialTokens


@dataclass(frozen=True)
class RecaptionStageParams:
    """Stage forcing knobs for ``mode='gen_text'`` AR decode."""

    first_bot_task: str
    stage_transitions: list[tuple[int, list[int]]]
    final_stop_tokens: list[int]
    need_ratio: bool


@dataclass
class RecaptionResult:
    """Output of a recaption / think AR stage."""

    cot_text: list[str]
    sequences: torch.Tensor
    new_tokens: list[list[int]]
    input_length: int
    image_size: str | int | None = None
    stage_params: RecaptionStageParams | None = None
    ttft: float | None = None  # seconds to first generated token's logits
    tps: float | None = None  # tokens/sec over tokens generated after the first
    total_seconds: float | None = None  # wall-clock for the whole AR decode (end-of-CoT time)


def _ratio_stop_token_ids(sp: SpecialTokens) -> list[int]:
    """All ratio token ids that can terminate the auto image-size stage."""
    ids = sp.all_ratio_token_ids()
    extra_33 = sp.special_token_map.get("<img_ratio_33>")
    if extra_33 is not None:
        extra_36 = sp.special_token_map.get("<img_ratio_36>")
        if extra_36 is not None:
            ids.extend(range(extra_33, extra_36 + 1))
        else:
            ids.append(extra_33)
    return ids


def build_recaption_stage_params(
    tok: HunyuanTokenizer,
    bot_task: str,
    *,
    image_size: str | int = IMAGE_BASE_SIZE,
    sequence_template: str | None = None,
    image_base_size: int | None = None,
) -> RecaptionStageParams:
    """Build ``stage_transitions`` and ``final_stop_tokens`` for recaption AR decode.

    ``bot_task`` may be ``'recaption'``, ``'think'``, or composite ``'think_recaption'``.
    """
    if bot_task not in ("think", "recaption", "think_recaption", "img_ratio"):
        raise ValueError(
            f"bot_task={bot_task!r} not a recaption task; "
            "use 'recaption', 'think', 'think_recaption', or 'img_ratio'"
        )
    sp = tok.special
    if sequence_template is None:
        sequence_template = tok.sequence_template
    if image_base_size is None:
        image_base_size = tok.config.image_base_size

    first_bot_task = bot_task.split("_")[0]
    need_ratio = image_size == "auto" or bot_task == "img_ratio"
    stage_transitions: list[tuple[int, list[int]]] = []

    if first_bot_task == "think" and "recaption" in bot_task:
        stage_transitions.append((sp.end_think_token_id, [sp.recaption_token_id]))

    if need_ratio:
        answer_prefix = [sp.answer_token_id] if sequence_template == "instruct" else []
        transition_id = sp.end_recaption_token_id if "recaption" in bot_task else sp.end_think_token_id
        stage_transitions.append(
            (
                transition_id,
                answer_prefix + [sp.boi_token_id, sp.size_token_id(image_base_size)],
            )
        )
        final_stop_tokens = _ratio_stop_token_ids(sp)
    elif "recaption" in bot_task:
        final_stop_tokens = [sp.end_recaption_token_id]
    else:
        final_stop_tokens = [sp.end_think_token_id, sp.end_recaption_token_id]

    return RecaptionStageParams(
        first_bot_task=first_bot_task,
        stage_transitions=stage_transitions,
        final_stop_tokens=final_stop_tokens,
        need_ratio=need_ratio,
    )


def default_recaption_sampling_config(
    gen_config: Any | None = None,
) -> SamplingConfig:
    """Default sampling knobs for Instruct recaption (matches generation_config.json)."""
    if gen_config is None:
        return SamplingConfig(
            do_sample=True,
            temperature=0.6,
            top_k=1024,
            top_p=0.95,
            repetition_penalty=1.0,
            max_new_tokens=2048,
        )
    return SamplingConfig(
        do_sample=bool(getattr(gen_config, "do_sample", True)),
        temperature=float(getattr(gen_config, "temperature", 0.6)),
        top_k=int(getattr(gen_config, "top_k", 1024)),
        top_p=float(getattr(gen_config, "top_p", 0.95)),
        repetition_penalty=float(getattr(gen_config, "repetition_penalty", 1.0)),
        max_new_tokens=int(getattr(gen_config, "max_new_tokens", 2048)),
    )


_QUAD_RE = re.compile(r"<quad><pos_x_(-?\d+)><pos_y_(-?\d+)><pos_x_(-?\d+)><pos_y_(-?\d+)></quad>")
_DEFAULT_QUAD = "<quad><pos_x_0><pos_y_0><pos_x_999><pos_y_999></quad>"


def _recaption_tail_looks_like_garbage(text: str) -> bool:
    if not text:
        return False
    if "<table>" in text or "<tr>" in text:
        return True
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if cjk >= 8:
        return True
    # Long English rewrites are valid Instruct cot — do not treat length alone as junk.
    prose = "".join(ch for ch in text if ch.isalnum() or ch.isspace()).strip()
    if len(prose) >= 20:
        return False
    return len(text) > 300


def sanitize_recaption_cot_text(
    cot_text: str,
    *,
    recaption_open: str,
    recaption_close: str,
) -> str:
    """Clamp malformed device AR recaption before it is injected into image gen."""
    if recaption_open not in cot_text:
        return cot_text

    body = cot_text.split(recaption_open, 1)[1]
    if recaption_close in body:
        body = body.split(recaption_close, 1)[0]
        return recaption_open + body + recaption_close

    for junk in ("<|endoftext|>", "<|pad|>"):
        if junk in body:
            body = body.split(junk, 1)[0]
    body = body.strip()

    quad_match = _QUAD_RE.search(body)
    if quad_match:
        quad = quad_match.group(0)
        tail = body[quad_match.end() :].strip()
        if tail and not _recaption_tail_looks_like_garbage(tail):
            body = f"{quad}{tail}"
        else:
            body = quad
    else:
        body = re.sub(r"<quad>.*", "", body, flags=re.DOTALL).strip()
        if body and not _recaption_tail_looks_like_garbage(body):
            body = f"{_DEFAULT_QUAD}{body}"
        else:
            body = _DEFAULT_QUAD

    if _recaption_tail_looks_like_garbage(body):
        body = _DEFAULT_QUAD

    return recaption_open + body + recaption_close


def _strip_recaption_body(cot_text: str, *, recaption_open: str, recaption_close: str) -> str:
    if recaption_open not in cot_text:
        return ""
    body = cot_text.split(recaption_open, 1)[1]
    if recaption_close in body:
        body = body.split(recaption_close, 1)[0]
    return body.strip()


def extract_recaption_written_prompt(
    cot_text: str,
    *,
    recaption_open: str,
    recaption_close: str,
) -> str:
    """Human-readable prose inside ``<recaption>`` (after optional ``<quad>`` block)."""
    body = _strip_recaption_body(cot_text, recaption_open=recaption_open, recaption_close=recaption_close)
    if not body:
        return ""
    quad_match = _QUAD_RE.search(body)
    if quad_match:
        body = body[quad_match.end() :].strip()
    body = re.sub(r"<[^>]+>", "", body)
    body = re.sub(r"<\|[^|]+\|>", "", body)
    return body.strip()


def extract_think_prose(
    cot_text: str,
    *,
    think_open: str,
    think_close: str | None = None,
    recaption_open: str | None = None,
) -> str:
    """Thinking prose before ``<recaption>`` when ``bot_task`` includes think."""
    if think_open not in cot_text:
        return ""
    body = cot_text.split(think_open, 1)[1]
    if recaption_open and recaption_open in body:
        body = body.split(recaption_open, 1)[0]
    elif think_close and think_close in body:
        body = body.split(think_close, 1)[0]
    body = re.sub(r"<[^>]+>", "", body)
    body = re.sub(r"<\|[^|]+\|>", "", body)
    return body.strip()


def is_meager_recaption_cot(
    cot_text: str,
    *,
    recaption_open: str,
    recaption_close: str,
    min_prose_chars: int = 20,
) -> bool:
    """True when AR output has no meaningful rewritten prompt (e.g. quad-only)."""
    prose = extract_recaption_written_prompt(cot_text, recaption_open=recaption_open, recaption_close=recaption_close)
    prose = "".join(ch for ch in prose if ch.isalnum() or ch.isspace()).strip()
    return len(prose) < min_prose_chars


def prompt_fallback_recaption_cot(tok: HunyuanTokenizer, prompt: str) -> str:
    """Wrap the user prompt when on-device AR cannot produce prose."""
    sp = tok.special
    recaption_open = tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    recaption_close = tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    return f"{recaption_open}{prompt}{recaption_close}"


def decode_cot_text(
    tok: HunyuanTokenizer,
    sequences: torch.Tensor,
    input_length: int,
    bot_task: str,
    *,
    drop_think: bool = False,
) -> list[str]:
    """Extract ``cot_text`` strings from generated token sequences."""
    sp = tok.special
    first_bot_task = bot_task.split("_")[0]
    think_str = tok.tokenizer.convert_ids_to_tokens(sp.think_token_id)
    recaption_str = tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    end_recaption_str = tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)

    if "recaption" in bot_task:
        end_token_id = sp.end_recaption_token_id
    else:
        end_token_id = sp.end_think_token_id

    generated = sequences[:, input_length:]
    cot_texts: list[str] = []
    for row in range(sequences.shape[0]):
        gen = generated[row]
        end_positions = (gen == end_token_id).nonzero(as_tuple=False)
        if end_positions.numel() > 0:
            cot_tokens = gen[: end_positions[0].item() + 1]
        else:
            cot_tokens = gen
        cot_text_gen = tok.decode(cot_tokens.tolist(), skip_special_tokens=False)

        if first_bot_task == "think":
            cot_text = think_str + cot_text_gen
        else:
            cot_text = recaption_str + cot_text_gen

        if drop_think and think_str in cot_text and recaption_str in cot_text:
            recaption_part = cot_text.split(recaption_str, 1)[1]
            if end_recaption_str in recaption_part:
                recaption_part = recaption_part.split(end_recaption_str, 1)[0]
            cot_text = recaption_str + recaption_part + end_recaption_str

        if "recaption" in bot_task:
            cot_text = sanitize_recaption_cot_text(
                cot_text,
                recaption_open=recaption_str,
                recaption_close=end_recaption_str,
            )

        cot_texts.append(cot_text)
    return cot_texts


def build_ratio_logits_processor(
    tok: HunyuanTokenizer,
    *,
    image_base_size: int | None = None,
    force_greedy: bool = True,
) -> ConditionalSliceVocabLogitsProcessor:
    """Ratio-token vocab slice processor for ``image_size='auto'`` recaption."""
    sp = tok.special
    if image_base_size is None:
        image_base_size = tok.config.image_base_size
    other_slices: list = []
    extra_33 = sp.special_token_map.get("<img_ratio_33>")
    extra_36 = sp.special_token_map.get("<img_ratio_36>")
    if extra_33 is not None:
        if extra_36 is not None:
            other_slices.append((extra_33, extra_36 + 1))
        else:
            other_slices.append(extra_33)
    return ConditionalSliceVocabLogitsProcessor(
        trigger_token_ids=[sp.size_token_id(image_base_size)],
        vocab_start=sp.ratio_token_id(0),
        vocab_end=sp.ratio_token_id(sp.num_ratio_tokens - 1) + 1,
        other_slices=other_slices,
        force_greedy=force_greedy,
    )


def resolve_image_size_from_ratio(
    tok: HunyuanTokenizer,
    processor,
    ratio_token_id: int,
) -> str | int:
    """Map a generated ratio token id to ``'HxW'`` using the VAE resolution group."""
    sp = tok.special
    try:
        ratio_index = sp.all_ratio_token_ids().index(ratio_token_id)
    except ValueError:
        extra_ids = _ratio_stop_token_ids(sp)
        ratio_index = extra_ids.index(ratio_token_id)
    reso = processor.vae_reso_group[ratio_index]
    return f"{reso.height}x{reso.width}"


def system_prompt_for_bot_task(bot_task: str) -> tuple[str, str]:
    """Return ``(system_prompt_key, system_prompt_subkey)`` for upstream ``get_system_prompt``."""
    if bot_task in ("think", "think_recaption"):
        return ("en_think_recaption", "think_recaption")
    if bot_task == "recaption":
        return ("en_recaption", "recaption")
    return ("en_unified", "image")


def run_recaption_ar(
    forward_logits_fn: Callable[[torch.Tensor], torch.Tensor],
    bundle: GenImageHostInputs,
    tok: HunyuanTokenizer,
    bot_task: str,
    *,
    image_size: str | int = IMAGE_BASE_SIZE,
    sequence_template: str | None = None,
    config: SamplingConfig | None = None,
    generator: torch.Generator | None = None,
    drop_think: bool = False,
) -> RecaptionResult:
    """Run AR recaption using our host ``generate_text`` loop.

    ``forward_logits_fn`` maps ``input_ids [B, S]`` → next-token logits ``[B, V]``.
    Intended for ref/TT adapters once backbone + LM head are wired.
    """
    params = build_recaption_stage_params(
        tok,
        bot_task,
        image_size=image_size,
        sequence_template=sequence_template,
    )
    config = config or default_recaption_sampling_config()
    input_length = int(bundle.input_ids.shape[1])

    logits_processors = None
    if params.need_ratio:
        logits_processors = [
            build_ratio_logits_processor(tok, image_base_size=tok.config.image_base_size, force_greedy=True)
        ]

    out = generate_text(
        forward_logits_fn,
        bundle.input_ids,
        config=config,
        stage_transitions=params.stage_transitions or None,
        final_stop_tokens=params.final_stop_tokens,
        logits_processors=logits_processors,
        generator=generator,
    )
    cot_text = decode_cot_text(
        tok,
        out["sequences"],
        input_length,
        bot_task,
        drop_think=drop_think,
    )
    resolved_size = image_size
    if params.need_ratio:
        resolved_size = _resolve_ratio_token(tok, int(out["sequences"][0, -1].item()))

    return RecaptionResult(
        cot_text=cot_text,
        sequences=out["sequences"],
        new_tokens=out["new_tokens"],
        input_length=input_length,
        image_size=resolved_size,
        stage_params=params,
    )


def _resolve_ratio_token(tok: HunyuanTokenizer, ratio_token_id: int, *, model=None) -> str | int:
    """Map a generated ratio token id back to ``'HxW'`` when possible."""
    if model is not None:
        tkw = model._tokenizer
        ratio_index = model._get_ratio_index_from_token(ratio_token_id, tkw)
        reso = model.image_processor.vae_reso_group[ratio_index]
        return f"{reso.height}x{reso.width}"

    sp = tok.special
    try:
        ratio_index = sp.all_ratio_token_ids().index(ratio_token_id)
        return ratio_index
    except ValueError:
        return ratio_token_id


def run_recaption(
    model,
    prompt: str,
    *,
    bot_task: str = "recaption",
    system_prompt: str | None = None,
    image=None,
    image_size: str | int = IMAGE_BASE_SIZE,
    max_new_tokens: int | None = None,
    seed: int | None = None,
    drop_think: bool = False,
    verbose: int = 1,
    sequence_template: str | None = None,
) -> RecaptionResult:
    """Run the full recaption / think stage on an upstream ``HunyuanImage3ForCausalMM``.

      Uses upstream ``prepare_model_inputs`` + ``generate`` for the AR forward (cond encode,
      KV cache, attention mask) and our stage-param builder + ``decode_cot_text`` for
    parity with the port's golden sampling path.
    """
    if bot_task not in ("think", "recaption", "think_recaption", "img_ratio"):
        raise ValueError(f"bot_task={bot_task!r} not supported for run_recaption")

    tok = _hf_tokenizer_as_port(model)
    params = build_recaption_stage_params(
        tok,
        bot_task,
        image_size=image_size,
        sequence_template=sequence_template or getattr(model.generation_config, "sequence_template", "instruct"),
        image_base_size=getattr(model.image_processor.vae_reso_group, "base_size", tok.config.image_base_size),
    )

    gen_kwargs: dict[str, Any] = {}
    if seed is not None:
        gen_kwargs["seed"] = seed
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens

    model_inputs = model.prepare_model_inputs(
        prompt=prompt,
        image=image,
        mode="gen_text",
        bot_task=params.first_bot_task,
        system_prompt=system_prompt,
        image_size=image_size,
        **gen_kwargs,
    )
    input_length = int(model_inputs["input_ids"].shape[1])

    logits_processor = None
    if params.need_ratio:
        from transformers import LogitsProcessorList

        tkw = model._tokenizer
        image_base_size = model.image_processor.vae_reso_group.base_size
        logits_processor = LogitsProcessorList(
            [
                model._ConditionalSliceVocabLogitsProcessor(
                    trigger_token_ids=[tkw.size_token_id(image_base_size)],
                    vocab_start=tkw.start_ratio_token_id,
                    vocab_end=tkw.end_ratio_token_id + 1,
                    other_slices=getattr(tkw, "ratio_token_other_slices", []),
                    force_greedy=True,
                )
            ]
        )

    generate_kwargs = dict(
        **model_inputs,
        decode_text=False,
        logits_processor=logits_processor,
        verbose=verbose,
    )
    if params.stage_transitions:
        generate_kwargs["stage_transitions"] = params.stage_transitions
        generate_kwargs["final_stop_tokens"] = params.final_stop_tokens

    outputs = model.generate(**generate_kwargs)
    cot_text = decode_cot_text(tok, outputs, input_length, bot_task, drop_think=drop_think)

    resolved_size = image_size
    if params.need_ratio:
        resolved_size = _resolve_ratio_token(tok, int(outputs[0, -1].item()), model=model)

    new_tokens = outputs[:, input_length:].tolist()
    return RecaptionResult(
        cot_text=cot_text,
        sequences=outputs,
        new_tokens=[list(row) for row in new_tokens],
        input_length=input_length,
        image_size=resolved_size,
        stage_params=params,
    )


def _hf_tokenizer_as_port(model) -> HunyuanTokenizer:
    """Lightweight adapter so stage/decode helpers can use ``HunyuanTokenizer`` API."""
    tkw = model._tokenizer
    hf_tok = tkw.tokenizer if hasattr(tkw, "tokenizer") else tkw
    from models.experimental.hunyuan_image_3_0.ref.tokenizer.hunyuan_tokenizer import HunyuanConfig
    from models.experimental.hunyuan_image_3_0.ref.tokenizer.special_tokens import build_special_tokens

    config = HunyuanConfig.from_dict(model.config.to_dict())
    special = build_special_tokens(hf_tok, model_version=config.model_version)
    seq_tpl = getattr(model.generation_config, "sequence_template", "instruct")
    return HunyuanTokenizer(config, hf_tok, special, sequence_template=seq_tpl)
