# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pinned, chat-rendered shared suite. HF mode never imports TTNN."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "IFM/K2-Horizon-MoVA-36B-A4B"
REVISION = "e9ce717642b2074082875be233c02c729ba47960"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "doc/full_model/qualitative"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hf", "tt"), required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=6)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/vllm_prompts.txt"
    prompts = source.read_text().strip().split("\n\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        model=MODEL,
        revision=REVISION,
        tokenizer=type(tokenizer).__name__,
        chat_template_present=bool(tokenizer.chat_template),
        prompt_mode="chat",
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        max_new_tokens=args.tokens,
        do_sample=False,
        rendering="apply_chat_template(add_generation_prompt=True)",
    )
    (output_root / "prompt_format.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.mode == "hf":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval()
    else:
        import ttnn

        from ..tt.generator import build_generator

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        model = build_generator(ROOT, mesh, max_seq_len=1024)
        model.capture()
    try:
        for index, prompt in enumerate(prompts):
            if index < args.start or index >= args.stop:
                continue
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, return_dict=False, add_generation_prompt=True
            )
            assert tokenizer.encode(rendered, add_special_tokens=False) == ids
            directory = output_root / f"prompt_{index}"
            directory.mkdir(exist_ok=True)
            (directory / "prompt.txt").write_text(rendered)
            started = time.perf_counter()
            if args.mode == "hf":
                with torch.inference_mode():
                    result = model.generate(
                        torch.tensor([ids]),
                        max_new_tokens=args.tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )[0, len(ids) :].tolist()
            else:
                model.reset()
                result = model.generate(ids, args.tokens, enable_trace=True)
            text = tokenizer.decode(result, skip_special_tokens=False)
            evidence = dict(
                **metadata,
                prompt_id=index,
                prompt_token_ids=ids,
                token_ids=result,
                text=text,
                elapsed_s=time.perf_counter() - started,
                cpu_threads=args.threads,
            )
            (directory / f"{args.mode}.json").write_text(json.dumps(evidence, indent=2) + "\n")
            (directory / f"{args.mode}_completion.txt").write_text(text)
            print(f"PROMPT {index} {args.mode} {len(result)} tokens {evidence['elapsed_s']:.2f}s\n{text}", flush=True)
    finally:
        if args.mode == "tt":
            model.close()
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
