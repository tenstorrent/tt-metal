# SPDX-License-Identifier: Apache-2.0
"""Free-running greedy generation on the Gemma-4 26B autoport, at GPQA length.

Measures the regime no bringup stage ever measured: sustained self-fed decode.
Records token ids, stop reason, and text so an HF control can score every step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/mvasiljevic/tt-metal")
sys.path.insert(0, str(REPO))

from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

from models.common.readiness_check.mesh_device import (  # noqa: E402
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)

MODEL_DIR = REPO / "models/autoports/google_gemma_4_26b_a4b_it"
HF_MODEL = "google/gemma-4-26B-A4B-it"

# Exact doc_to_text of gpqa_diamond_cot_zeroshot, as recorded in
# doc/tti_release/tti_eval_gpqa_cot.json. The graded documents themselves are in
# a gated dataset, so the question below is a stand-in of the same shape.
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


def _import_build_generator(model_dir: Path):
    path = model_dir / "tt" / "generator.py"
    spec = importlib.util.spec_from_file_location(f"_gen_{model_dir.name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_generator


def _tokenizer():
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    if not getattr(tok, "chat_template", None):
        processor = AutoProcessor.from_pretrained(HF_MODEL, trust_remote_code=True)
        template = getattr(processor, "chat_template", None)
        if template:
            tok.chat_template = template
    return tok


FILLER = (
    "Background note: the following context is not needed to answer the question. "
    "It is included only to change the prompt length. "
)


def _render(tok, thinking: bool, filler_repeats: int = 0) -> tuple[list[int], str]:
    text = GPQA_DOC_TO_TEXT.format(**QUESTION)
    if filler_repeats:
        text = FILLER * filler_repeats + text
    kwargs = {} if not thinking else {"enable_thinking": True}
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
        **kwargs,
    )
    # The template emits <bos> itself; do not let the tokenizer add a second one.
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    return list(ids), rendered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--thinking", action="store_true", help="Render with enable_thinking=True.")
    ap.add_argument(
        "--sampling-mode",
        default="device",
        choices=("device", "host"),
        help="'device' is the shipped sharded chunked-topk greedy path; 'host' all-gathers "
        "logits and takes an exact torch.argmax. Differences localise the sampler.",
    )
    ap.add_argument("--mesh-device", default="P300X2")
    ap.add_argument("--fabric-config", default="FABRIC_1D_RING")
    ap.add_argument(
        "--filler-repeats",
        type=int,
        default=0,
        help="Prepend N copies of a neutral filler sentence to move the prompt length, "
        "so a failure can be attributed to absolute position vs generated count.",
    )
    ap.add_argument("--precision-config", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tok = _tokenizer()
    prompt_ids, rendered = _render(tok, args.thinking, args.filler_repeats)
    print(f"prompt tokens: {len(prompt_ids)}  thinking={args.thinking}", flush=True)
    print("rendered prompt tail:", repr(rendered[-220:]), flush=True)

    mesh = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    try:
        build_generator = _import_build_generator(MODEL_DIR)
        t0 = time.perf_counter()
        generator = build_generator(
            MODEL_DIR,
            mesh,
            hf_model=HF_MODEL,
            max_seq_len=args.max_seq_len,
            max_batch_size=1,
            sampling_mode=args.sampling_mode,
            precision_config_path=args.precision_config,
        )
        print(f"generator built in {time.perf_counter() - t0:.1f}s", flush=True)

        t1 = time.perf_counter()
        out_ids = generator.generate(
            prompt_token_ids=list(prompt_ids),
            max_new_tokens=args.max_new_tokens,
            enable_trace=True,
        )
        elapsed = time.perf_counter() - t1

        eos_ids = list(generator.model.eos_token_ids)
        stopped_on_eos = bool(out_ids) and out_ids[-1] in eos_ids
        text = tok.decode(out_ids, skip_special_tokens=False)
        payload = {
            "thinking": args.thinking,
            "prompt_token_ids": list(prompt_ids),
            "prompt_text": rendered,
            "generated_token_ids": out_ids,
            "num_generated": len(out_ids),
            "requested": args.max_new_tokens,
            "stopped_on_eos": stopped_on_eos,
            "eos_token_ids": eos_ids,
            "elapsed_s": elapsed,
            "t_s_u": len(out_ids) / max(elapsed, 1e-9),
            "perf": generator.last_perf,
            "max_seq_len": args.max_seq_len,
            "precision_config": str(generator.model.precision_config_path),
            "text": text,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        args.out.with_suffix(".txt").write_text(text, encoding="utf-8")
        print(
            f"generated {len(out_ids)}/{args.max_new_tokens} tokens in {elapsed:.1f}s "
            f"({payload['t_s_u']:.2f} t/s/u), stopped_on_eos={stopped_on_eos}",
            flush=True,
        )
        print(f"wrote {args.out}", flush=True)
    finally:
        close_readiness_mesh_device(mesh, args.fabric_config)


if __name__ == "__main__":
    main()
