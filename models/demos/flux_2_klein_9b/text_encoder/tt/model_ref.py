# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Source-A side of the FLUX.2-klein-9B text-encoder bring-up.

Everything that touches HuggingFace lives here: the tokenizer, the reference
model, and the GOLDEN outputs the TT pipeline is scored against. `tt/pipeline.py`
imports this module for setup and for reference values only -- no HF forward is
ever called from the TT forward path.

The checkpoint under `black-forest-labs/FLUX.2-klein-9B (subfolder `text_encoder`)`
is the `text_encoder` folder of `black-forest-labs/FLUX.2-klein-9B`: a
`Qwen3ForCausalLM` (36 layers, 4096 hidden, 32 q / 8 kv heads, vocab 151936).
It ships no tokenizer files of its own -- the pipeline's tokenizer is a SIBLING
folder of the same hub repo, declared in `model_index.json` as
`Qwen2TokenizerFast`, so that is what `load_tokenizer` resolves.
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import torch

HF_MODEL_ID = os.environ.get("FLUX2_KLEIN_TEXT_ENCODER", "black-forest-labs/FLUX.2-klein-9B (subfolder `text_encoder`)")
HUB_REPO = "black-forest-labs/FLUX.2-klein-9B"
TOKENIZER_SUBFOLDER = "tokenizer"

# The prompt the demo, the e2e test and the trace seams all share, so the three
# exercise byte-identical input. A FLUX prompt is what this encoder sees in
# production; it is also a perfectly ordinary causal-LM prompt.
DEFAULT_PROMPT = "A photograph of a rusted lighthouse at dawn, waves breaking against the rocks."

# Safety cap for the on-device gate. `generation_config.json` states NO
# max_new_tokens and no max_length, so there is no model-declared generation
# length; the decode horizon is therefore STOP-TOKEN driven (eos_token_id
# 151645 / 151643) and this number only bounds a run that never emits one.
# It is applied IDENTICALLY to the HF reference and to the TT loop.
E2E_GATE_MAX_NEW_TOKENS = 40

_MODEL_CACHE: dict = {}
_TOKENIZER_CACHE: dict = {}


# --------------------------------------------------------------------- inputs


def _tokenizer_dir() -> str:
    override = os.environ.get("FLUX2_KLEIN_TOKENIZER")
    if override:
        return override
    if os.path.isfile(os.path.join(HF_MODEL_ID, "tokenizer_config.json")):
        return HF_MODEL_ID
    from huggingface_hub import snapshot_download

    for local_only in (True, False):
        try:
            snap = snapshot_download(HUB_REPO, allow_patterns=[f"{TOKENIZER_SUBFOLDER}/*"], local_files_only=local_only)
            cand = os.path.join(snap, TOKENIZER_SUBFOLDER)
            if os.path.isfile(os.path.join(cand, "tokenizer_config.json")):
                return cand
        except Exception:  # noqa: BLE001 - fall through to the networked attempt
            continue
    raise RuntimeError(f"could not resolve the {HUB_REPO} tokenizer; set FLUX2_KLEIN_TOKENIZER to a local copy")


def load_tokenizer():
    if "tok" not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE["tok"] = AutoTokenizer.from_pretrained(_tokenizer_dir())
    return _TOKENIZER_CACHE["tok"]


def encode_prompt(prompt: str, *, chat: bool = False) -> torch.Tensor:
    """The real HF tokenizer turning text into `input_ids [1, S]`."""
    tok = load_tokenizer()
    if chat:
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            return tok(text, return_tensors="pt").input_ids
        except Exception:  # noqa: BLE001 - template is optional; raw text is still valid input
            pass
    return tok(prompt, return_tensors="pt").input_ids


def decode_tokens(ids: Sequence[int]) -> str:
    return load_tokenizer().decode(list(ids), skip_special_tokens=True)


# ------------------------------------------------------------------ reference


def load_hf_model(dtype=torch.float32):
    """The Source-A reference model. Also the weight source for the TT build."""
    key = str(dtype)
    if key not in _MODEL_CACHE:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
        )
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def stop_token_ids(model) -> list:
    """The model's own end tokens, from generation_config then config."""
    out = []
    for cfg in (getattr(model, "generation_config", None), model.config):
        eos = getattr(cfg, "eos_token_id", None) if cfg is not None else None
        if eos is None:
            continue
        out.extend(eos if isinstance(eos, (list, tuple)) else [eos])
    return sorted({int(e) for e in out})


def resolve_max_new_tokens(model, prompt_len: int, gate_cap: int | None = E2E_GATE_MAX_NEW_TOKENS) -> int:
    """The decode horizon, in the priority order the bring-up contract sets.

    1. a stop token exists (it does: eos_token_id), so decoding is stop-driven and
       everything below is only the SAFETY bound that keeps a non-terminating run
       from hanging;
    2. `generation_config.max_new_tokens` / `max_length` if the model states one --
       this one does not;
    3. the gate cap, which is a bound, not a scope reduction: both the TT loop and
       the HF reference are given the same number and both break on the same id.
    """
    gc = getattr(model, "generation_config", None)
    stated = getattr(gc, "max_new_tokens", None) if gc is not None else None
    if stated is None and gc is not None and getattr(gc, "max_length", None):
        stated = int(gc.max_length) - prompt_len
    context_room = int(model.config.max_position_embeddings) - prompt_len
    bounds = [context_room]
    if stated is not None:
        bounds.append(int(stated))
    if gate_cap is not None:
        bounds.append(int(gate_cap))
    return max(1, min(bounds))


@torch.no_grad()
def hf_reference_text_generation(model, input_ids: torch.Tensor, max_new_tokens: int):
    """The golden for Call 1: `model.generate()` itself, with per-step logits.

    Greedy (`do_sample=False`) so the comparison is deterministic, capped at the
    SAME horizon the TT loop is given, and stopped by the SAME eos ids.
    """
    stop = stop_token_ids(model)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=stop or None,
        pad_token_id=int(getattr(model.generation_config, "pad_token_id", 0) or 0),
        return_dict_in_generate=True,
        output_logits=True,
    )
    prompt_len = int(input_ids.shape[-1])
    new_ids = out.sequences[0, prompt_len:].tolist()
    step_logits = torch.stack([lg[0].float() for lg in out.logits], dim=0)  # [steps, vocab]
    return {"token_ids": new_ids, "step_logits": step_logits, "text": decode_tokens(new_ids)}


@torch.no_grad()
def hf_reference_step_logits(model, input_ids: torch.Tensor, generated_ids: Sequence[int]) -> torch.Tensor:
    """The golden logits for the SAME contexts the TT loop actually decoded.

    Step `i` of the TT loop conditions on `prompt + generated[:i]`; this scores the
    reference on exactly those prefixes, so the comparison measures THIS pipeline's
    arithmetic rather than re-measuring the greedy tie-breaks a token sequence
    inherits once two runs have parted. One causal forward over
    `prompt + generated[:-1]` yields every step at once and is identical to running
    them incrementally, because the model is causal.

    Returns `[len(generated), vocab]`, aligned step-for-step with the TT logits.
    """
    tokens = list(generated_ids)
    if not tokens:
        return torch.empty(0)
    prompt_len = int(input_ids.shape[-1])
    context = torch.zeros(1, prompt_len + len(tokens) - 1, dtype=torch.long)
    context[0, :prompt_len] = input_ids[0, :prompt_len]
    if len(tokens) > 1:
        context[0, prompt_len:] = torch.tensor(tokens[:-1], dtype=torch.long)
    logits = model(input_ids=context).logits
    # position p predicts token p+1, so the step that emitted `tokens[i]` is read
    # off position `prompt_len - 1 + i`.
    return logits[0, prompt_len - 1 :, :].float()


@torch.no_grad()
def hf_reference_prompt_encoding(model, input_ids: torch.Tensor) -> torch.Tensor:
    """The golden for Call 2: the inner Qwen3Model's encoded hidden states.

    That inner module is exactly what `encoder_stack` is bound to, and it is what
    `Flux2Transformer2DModel` consumes as the prompt embedding.
    """
    return model.model(input_ids=input_ids).last_hidden_state.float()


@torch.no_grad()
def hf_rope_tables(model, position_ids: torch.Tensor, dtype=torch.float32):
    """(cos, sin) straight out of the reference's own `rotary_emb`.

    Used to seed the trace stages' persistent constants, so the traced shapes
    carry the golden's values rather than a re-derivation of them.
    """
    hidden = torch.zeros(1, int(position_ids.shape[-1]), model.config.hidden_size, dtype=next(model.parameters()).dtype)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    return cos.to(dtype), sin.to(dtype)


# ----------------------------------------------------------------------- pcc


def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    """Pearson correlation of two flattened tensors -- the bring-up metric.

    Accumulated in float64: the decode comparison is ~6M elements, where a float32
    dot product and norm carry enough rounding (~1e-4 relative) to report a
    correlation just ABOVE 1, which is not a number a correlation can take.
    """
    a = golden.detach().double().flatten()
    b = actual.detach().double().flatten()
    if a.numel() != b.numel():
        raise ValueError(f"pcc shape mismatch: {tuple(golden.shape)} vs {tuple(actual.shape)}")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b) / denom)


def first_divergence(a: Iterable[int], b: Iterable[int]):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None
