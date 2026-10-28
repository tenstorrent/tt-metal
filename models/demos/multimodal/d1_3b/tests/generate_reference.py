# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden d1-3B decisions from the Hugging Face reference (`trust_remote_code=True`), on CPU.

Needs transformers>=5.14 (d1's remote code) and no ttnn:

    python models/demos/multimodal/d1_3b/tests/generate_reference.py --dtype float32
    python models/demos/multimodal/d1_3b/tests/generate_reference.py --dtype bfloat16

Every question is run as its own chain (state then question, read at the last token), which is what the
TT model reproduces. HF's `system_one` packs several questions over one state into one tree pass; its
answers are recorded alongside as `api_probabilities`, mathematically the same as the chains.

The JSON (committed) holds token ids, option-token log-probabilities, probabilities and answers. The sidecar
(`D1_SIDECAR_DIR`, not committed) holds the final hidden state and full logits at the answer slot, and the
processor's pixel inputs, for the TT PCC tests.
"""

import argparse
import json
import os
import platform
import sys
import time

import torch
from loguru import logger

from models.demos.multimodal.d1_3b.tests.common import (
    D1_HF_MODEL,
    D1_REVISION,
    REFERENCE_DIR,
    SIDECAR_DIR,
    ids_sha256,
    load_examples,
    load_images,
    load_state,
    remote_modules,
    resolve_checkpoint,
    text_sha256,
)

TOP_K = 10


def chain_inputs(engine, prompt, state, q, images):
    """The token ids (and pixel inputs) of one question over its state, exactly as HF's SystemOne builds them."""
    if not images:
        text = engine.render(state, q)
        ids = engine.tokenizer.encode(text, add_special_tokens=False)
        return text, {"input_ids": torch.tensor([ids])}
    markup = engine._image_markup(len(images))
    text = prompt.render(
        engine.tokenizer,
        state,
        q,
        engine.bos,
        engine.lead,
        engine.state_style,
        engine.system,
        engine.option_style,
        images=markup,
    )
    cap_pixels = sys.modules[type(engine).__module__].cap_pixels
    return text, engine._image_inputs(text, [cap_pixels(im) for im in images])


def main():
    parser = argparse.ArgumentParser(description="Generate CPU reference decisions for d1-3B.")
    parser.add_argument("--hf-model", default=os.environ.get("HF_MODEL", D1_HF_MODEL))
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--only", nargs="*", help="example names to run (default: all)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    ckpt_dir = resolve_checkpoint(args.hf_model)

    import transformers
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(ckpt_dir), trust_remote_code=True, dtype=dtype).eval()
    engine = model.engine
    prompt, api = remote_modules(anchor=type(engine).__module__)
    tok = engine.tokenizer
    if engine.calibration is not None:
        raise RuntimeError("d1-3B ships no calibration; the reference assumes calibration=None")

    captured = {}

    def keep_hidden(_module, _inputs, output):
        captured["hidden"] = output.last_hidden_state[0, -1].detach().float().clone()

    hook = model.model.language_model.register_forward_hook(keep_hidden)

    examples = load_examples()
    if args.only:
        examples = [e for e in examples if e["name"] in set(args.only)]

    results, sidecar = [], {}
    with torch.inference_mode():
        for example in examples:
            state, images = load_state(example), load_images(example)
            questions = {n: prompt.as_question(q) for n, q in example["questions"].items()}
            entry = {"name": example["name"], "questions": {}}
            for qname, q in questions.items():
                text, inputs = chain_inputs(engine, prompt, state, q, images)
                ids = inputs["input_ids"][0].tolist()
                start = time.perf_counter()
                logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
                elapsed = time.perf_counter() - start
                logz = logits - torch.logsumexp(logits, dim=-1)
                groups = prompt.readout_ids(tok, q)
                probs = prompt.readout(tok, q, logz, engine.calibration)
                top = torch.topk(logz, TOP_K)
                entry["questions"][qname] = {
                    "type": q.type,
                    "num_tokens": len(ids),
                    "text_sha256": text_sha256(text),
                    "input_ids_sha256": ids_sha256(ids),
                    "readout_ids": groups,
                    "option_logz": [[float(logz[i]) for i in g] for g in groups],
                    "probabilities": probs,
                    "answer": api.answer(q, probs),
                    "top_ids": top.indices.tolist(),
                    "top_logz": top.values.tolist(),
                    "cpu_seconds": elapsed,
                }
                side = {"input_ids": ids, "hidden": captured.pop("hidden"), "logits": logits.clone()}
                if images:
                    side["pixel_inputs"] = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
                sidecar[f"{example['name']}/{qname}"] = side

            api_out = model.system_one(state, example["questions"], images or None)
            entry["api_input_tokens"] = api_out["usage"]["input_tokens"]
            entry["api_answers"] = api_out["answers"]
            results.append(entry)
            for qname, r in entry["questions"].items():
                best = max(range(len(r["probabilities"])), key=r["probabilities"].__getitem__)
                logger.info(
                    f"{example['name']:<26}{qname:<10}{r['type']:<7}{r['num_tokens']:>6} tok  "
                    f"p[{best}]={r['probabilities'][best]:.4f}  {r['cpu_seconds']:.2f}s"
                )
    hook.remove()

    reference = {
        "hf_model": D1_HF_MODEL,
        "revision": D1_REVISION,
        "dtype": args.dtype,
        "device": "cpu",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "host": platform.machine(),
        "calibration": None,
        "examples": results,
    }
    suffix = "" if args.dtype == "float32" else f"_{args.dtype}"
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = REFERENCE_DIR / f"decision_reference{suffix}.json"
    if args.only:
        out = out.with_name(out.stem + "_partial.json")
    out.write_text(json.dumps(reference, indent=1, ensure_ascii=False) + "\n")
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(sidecar, SIDECAR_DIR / f"{out.stem}.pt")
    logger.info(f"wrote {out} and {SIDECAR_DIR / (out.stem + '.pt')}")


if __name__ == "__main__":
    main()
