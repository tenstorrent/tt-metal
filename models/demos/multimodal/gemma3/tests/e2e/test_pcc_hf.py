# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF-referenced end-to-end PCC gate for the gemma-3 TEXT pipeline.

WHY THIS EXISTS
    gemma3 ships per-module PCC tests but no END-TO-END one, and the optimize loop needs a
    whole-pipeline correctness gate: it reverts any edit whose PCC drops, so the gate is the only
    thing standing between a perf lever and a silently degraded model.

    It must not be a token-accuracy gate. Quantization error is proportional to magnitude, so it
    preserves the argmax ordering while corrupting the values: measured on realistic logits, a
    bf4_b-style weight lever sat at PCC 0.513 with 100% top-1 match. optimize walks
    knob:dtype bf16 -> bf8_b -> bf4_b, so that is not hypothetical -- and gemma3's own
    "performance" preset already puts FF1_FF3 at BFP4.

WHAT IT MEASURES
    Raw decode LOGITS from the resident TT generator vs the same logits from HuggingFace, on a
    fixed greedy prompt. Logits are captured BEFORE argmax -- capturing after is exactly what makes
    a token-accuracy gate blind. Decode is TEACHER-FORCED from the reference tokens so a single
    divergence does not make every later step compare different contexts.

REFERENCE CACHING
    This gate runs after EVERY edit, so HF must not run every time: the reference logits are
    computed once and cached OUTSIDE the repo (optimize runs each iteration from a fresh temp
    worktree, so a cache beside this file would be absent every time and force a full HF CPU run
    per iteration). Delete the cache to regenerate.

    Emits ``PCC: <float>`` on stdout, which is the contract the optimize loop parses.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HF_MODEL", "google/gemma-3-12b-it")
HF_MODEL_ID = os.environ.get("HF_MODEL", "google/gemma-3-12b-it")

# The ONE correctness floor for this gate (the optimize loop lifts this number out of the source).
GEMMA3_PCC_MIN = 0.95

# Fixed, deterministic prompt: the only thing allowed to vary between runs is the model math.
PROMPT = "The capital of France is"
N_DECODE_STEPS = 4

# Match the demo's device configuration so the gate exercises the path optimize measures.
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
PAGE_PARAMS = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}

_REF_CACHE = Path(
    os.environ.get("GEMMA3_PCC_REF_CACHE")
    or (Path.home() / ".cache" / "tt_pcc_ref" / ("%s_logits.pt" % HF_MODEL_ID.replace("/", "_")))
)


def _hf_text_model():
    """The TEXT decoder of gemma-3, whichever class this transformers version exposes.

    gemma-3-12b-it is multimodal (Gemma3ForConditionalGeneration), so AutoModelForCausalLM may not
    map it. Fall back to the conditional-generation class and take its language_model.
    """
    import transformers

    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID, torch_dtype=torch.float32, device_map="cpu"
        )
    except (ValueError, KeyError, OSError):
        pass
    full = transformers.AutoModelForImageTextToText.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.float32, device_map="cpu"
    )
    return getattr(full, "language_model", full)


def _reference_logits(tokenizer_ids: torch.Tensor) -> torch.Tensor:
    """Reference decode logits from HF, cached on disk after the first computation."""
    if _REF_CACHE.is_file():
        blob = torch.load(_REF_CACHE, map_location="cpu")
        if blob.get("prompt") == PROMPT and blob.get("steps") == N_DECODE_STEPS and blob.get("model") == HF_MODEL_ID:
            return blob["logits"]

    model = _hf_text_model()
    model.eval()
    ids = tokenizer_ids.clone()
    steps = []
    with torch.no_grad():
        for _ in range(N_DECODE_STEPS):
            out = model(ids).logits[:, -1, :]  # logits for the NEXT token
            steps.append(out.float().cpu())
            ids = torch.cat([ids, out.argmax(-1, keepdim=True)], dim=1)  # greedy teacher-forcing
    logits = torch.stack(steps, dim=1)  # [batch, steps, vocab]
    _REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt": PROMPT, "steps": N_DECODE_STEPS, "model": HF_MODEL_ID, "logits": logits}, _REF_CACHE)
    return logits


def _as_logits(out):
    """prefill_forward_text / decode_forward may hand back a bare tensor or a tuple; take the tensor."""
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
        raise TypeError("no tensor in generator output: %r" % (type(out),))
    return out


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation over the flattened tensors.

    A zero denominator means one side is CONSTANT after centering. The reference is never constant,
    so that indicates a degenerate device output -- report 0.0, never 1.0.
    """
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 0.0
    return float((a @ b).item() / denom)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_e2e_pcc_hf(mesh_device, reset_seeds):
    """TT decode logits vs HF decode logits on a fixed prompt; gate on PCC."""
    from transformers import AutoTokenizer

    from models.demos.multimodal.gemma3.demo.text_demo import prepare_generator_args
    from models.demos.multimodal.gemma3.tt.gemma_multimodal_generator import GemmaMultimodalGenerator as Generator
    from models.tt_transformers.tt.model_config import DecodersPrecision

    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    ids = tok(PROMPT, return_tensors="pt").input_ids

    ref = _reference_logits(ids)
    # TEACHER FORCING: feed the REFERENCE's tokens into the TT decode, not TT's own predictions.
    # Letting each side continue from its own argmax means one divergence makes every later step
    # compare different contexts, so the PCC measures drift in the prompt rather than in the math.
    ref_tokens = ref.argmax(-1)  # [batch, steps]

    model_args, model, page_table, tt_kv_cache, tokenizer = prepare_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        instruct=True,
        global_batch_size=BATCH_SIZE,
        # A CALLABLE of model_args, exactly as the demo's parametrize supplies it.
        optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        max_seq_len=MAX_SEQ_LEN,
        page_params=PAGE_PARAMS,
        paged_attention=True,
    )
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)

    prompt = ids
    decoding_pos = [int(prompt.shape[1])]

    # sampling_params=None keeps RAW LOGITS: with greedy params the generator samples on device and
    # hands back a token, which is precisely the information a correctness gate must not lose.
    prefill_out = generator.prefill_forward_text(
        prompt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        sampling_params=None,
        warmup_prefill=False,
    )
    prefill_logits = _as_logits(prefill_out)
    got = [prefill_logits.float().cpu().reshape(1, -1)]
    cur_tok = ref_tokens[:, 0].reshape(1, 1)  # teacher-forced, not TT's own argmax
    cur_pos = torch.tensor(decoding_pos)

    for i in range(N_DECODE_STEPS - 1):
        out = generator.decode_forward(
            cur_tok,
            cur_pos,
            enable_trace=True,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=None,
        )
        step_logits = _as_logits(out)
        got.append(step_logits.float().cpu().reshape(1, -1))
        cur_tok = ref_tokens[:, i + 1].reshape(1, 1)  # teacher-forced
        cur_pos = cur_pos + 1

    tt_logits = torch.stack(got, dim=1)
    n = min(tt_logits.shape[1], ref.shape[1])
    v = min(tt_logits.shape[-1], ref.shape[-1])
    pcc = _pcc(tt_logits[:, :n, :v], ref[:, :n, :v])

    print(f"PCC: {pcc:.6f}")  # the contract the optimize loop parses
    assert pcc >= GEMMA3_PCC_MIN, f"e2e logits PCC {pcc:.6f} < floor {GEMMA3_PCC_MIN}"
