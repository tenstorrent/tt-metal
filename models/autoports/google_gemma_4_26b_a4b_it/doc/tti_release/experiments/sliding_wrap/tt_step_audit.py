# SPDX-License-Identifier: Apache-2.0
"""Per-step greedy audit of the Gemma-4 26B autoport.

Free-runs with an exact host argmax over all-gathered logits and records, for
every decode step, the top-64 candidate values/ids plus (best effort) the token
the shipped on-device sampler would have returned from the identical logits.

That yields two things the bringup pipeline never produced:
  * TT's own top1/top2 margin at each step -- how close each greedy decision was
  * a same-logits A/B of the shipped sampler against an exact argmax
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path("/home/mvasiljevic/tt-metal")
sys.path.insert(0, str(REPO))

from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

import ttnn  # noqa: E402
from models.common.readiness_check.mesh_device import (  # noqa: E402
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)

MODEL_DIR = REPO / "models/autoports/google_gemma_4_26b_a4b_it"
HF_MODEL = "google/gemma-4-26B-A4B-it"

GPQA_DOC_TO_TEXT = (
    "What is the correct answer to this question:{question}\n"
    "Choices:\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n"
    "Please reason step by step, and put your final answer (only the letter A, B, C, or D) "
    "within \\boxed{{}}.\nAnswer:"
)
QUESTION = {
    "question": (
        " A particle of mass m is confined to a one-dimensional box of width L with impenetrable walls. "
        "A weak perturbation V(x) = V0 * sin(pi x / L) is switched on. To first order in perturbation "
        "theory, what is the shift in the energy of the ground state?"
    ),
    "a": "8 V0 / (3 pi)",
    "b": "V0 / 2",
    "c": "0",
    "d": "2 V0 / pi",
}
# The two prompts whose committed HF/TT streams diverge earliest (index 0 and 3).
FRENCH = 'Translate the following to French: "Hello, how are you today?"'
EXPLANATION = "Explain the difference between supervised and unsupervised learning."

PROMPTS = {"gpqa": None, "french": FRENCH, "explanation": EXPLANATION}


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tokenizer():
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    if not getattr(tok, "chat_template", None):
        processor = AutoProcessor.from_pretrained(HF_MODEL, trust_remote_code=True)
        template = getattr(processor, "chat_template", None)
        if template:
            tok.chat_template = template
    return tok


def _render(tok, which: str, thinking: bool) -> tuple[list[int], str]:
    text = GPQA_DOC_TO_TEXT.format(**QUESTION) if which == "gpqa" else PROMPTS[which]
    kwargs = {"enable_thinking": True} if thinking else {}
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, **kwargs
    )
    return list(tok(rendered, add_special_tokens=False)["input_ids"]), rendered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--prompt", default="gpqa", choices=sorted(PROMPTS))
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--mesh-device", default="P300X2")
    ap.add_argument("--fabric-config", default="FABRIC_1D_RING")
    ap.add_argument("--precision-config", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tok = _tokenizer()
    prompt_ids, rendered = _render(tok, args.prompt, args.thinking)
    print(f"prompt={args.prompt} tokens={len(prompt_ids)} thinking={args.thinking}", flush=True)

    gen_mod = _import_module(MODEL_DIR / "tt" / "generator.py", "_gemma4_gen")
    mesh = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    try:
        generator = gen_mod.build_generator(
            MODEL_DIR,
            mesh,
            hf_model=HF_MODEL,
            max_seq_len=args.max_seq_len,
            max_batch_size=1,
            sampling_mode="host",
            precision_config_path=args.precision_config,
        )
        model = generator.model
        state = model.allocate_state(max_batch_size=1)
        logical_len = len(prompt_ids)
        physical = gen_mod._padded_prefill_len(logical_len)
        prompt = torch.tensor(prompt_ids, dtype=torch.long).reshape(1, logical_len)
        padded = torch.nn.functional.pad(prompt, (0, physical - logical_len))

        t0 = time.perf_counter()
        sharded = generator.prefill_forward(
            padded, page_table=state.page_tables, kv_cache=state, prompt_lens=[logical_len]
        )
        host_logits = generator._gather_logits_to_torch(sharded).float().reshape(-1)
        records: list[dict] = []
        tokens: list[int] = []
        eos_ids = set(model.eos_token_ids)
        sampler_mismatches: list[dict] = []
        sampler_ok = True

        def record(step: int, vec: torch.Tensor, sharded_logits) -> int:
            nonlocal sampler_ok
            top = torch.topk(vec, k=args.topk)
            chosen = int(top.indices[0])
            entry = {
                "i": step,
                "token": chosen,
                "top1_top2_margin": float(top.values[0] - top.values[1]),
                "top_ids": [int(x) for x in top.indices[:8]],
                "top_vals": [float(x) for x in top.values[:8]],
                "topk_ids": [int(x) for x in top.indices],
                "topk_vals": [float(x) for x in top.values],
            }
            if sampler_ok and sharded_logits is not None:
                try:
                    spec = gen_mod.SamplingSpec(greedy=True)
                    k, p, temp, seeds = generator._sampling_params(spec)
                    padded_logits = generator._pad_sampling_logits(sharded_logits)
                    tok_tensor, _ = generator.sampler.decode_forward(padded_logits, k=k, p=p, temp=temp, seeds=seeds)
                    device_token = int(generator._read_tokens(tok_tensor, 1)[0])
                    entry["device_sampler_token"] = device_token
                    if device_token != chosen:
                        sampler_mismatches.append(
                            {
                                "i": step,
                                "exact_argmax": chosen,
                                "device_sampler": device_token,
                                "margin": entry["top1_top2_margin"],
                            }
                        )
                except Exception as error:  # diagnostic only; never abort the audit
                    print(f"device-sampler A/B disabled at step {step}: {type(error).__name__}: {error}", flush=True)
                    sampler_ok = False
            records.append(entry)
            return chosen

        token = record(0, host_logits, sharded)
        tokens.append(token)
        for step in range(1, args.steps):
            if token in eos_ids:
                break
            sharded = generator.decode_forward(
                torch.tensor([[token]], dtype=torch.long),
                torch.tensor([logical_len + step - 1], dtype=torch.int32),
                page_table=state.page_tables,
                kv_cache=state,
                sampling_mode="host",
                enable_trace=False,
            )
            vec = generator._gather_logits_to_torch(sharded).float().reshape(-1)
            token = record(step, vec, sharded)
            tokens.append(token)
            if step % 128 == 0:
                print(
                    f"  step {step}  {time.perf_counter() - t0:.0f}s  sampler_mismatches={len(sampler_mismatches)}",
                    flush=True,
                )

        elapsed = time.perf_counter() - t0
        text = tok.decode(tokens, skip_special_tokens=False)
        payload = {
            "prompt": args.prompt,
            "thinking": args.thinking,
            "prompt_token_ids": prompt_ids,
            "prompt_text": rendered,
            "generated_token_ids": tokens,
            "num_generated": len(tokens),
            "requested": args.steps,
            "stopped_on_eos": bool(tokens) and tokens[-1] in eos_ids,
            "elapsed_s": elapsed,
            "selection": "exact host argmax over all-gathered logits",
            "device_sampler_ab": {
                "enabled": sampler_ok,
                "mismatches": sampler_mismatches,
                "num_mismatches": len(sampler_mismatches),
            },
            "precision_config": str(model.precision_config_path),
            "text": text,
            "steps": records,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload), encoding="utf-8")
        args.out.with_suffix(".txt").write_text(text, encoding="utf-8")
        print(
            f"generated {len(tokens)} tokens in {elapsed:.0f}s, eos={payload['stopped_on_eos']}, "
            f"sampler mismatches={len(sampler_mismatches)}",
            flush=True,
        )
        print(f"wrote {args.out}", flush=True)
    finally:
        close_readiness_mesh_device(mesh, args.fabric_config)


if __name__ == "__main__":
    main()
