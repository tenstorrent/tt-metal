# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Read the model's output on the **shared** qualitative prompt suite.

Stage 05's free-running evidence was one prompt -- the story continuation in
``models/common/readiness_check/autoregressive_prompt.txt``. The qualitative
check asks for several, across kinds, because a single narrative prompt cannot
show whether the model holds up on code, on translation, on factual recall or on
short-form instruction following, and those are the shapes a coder model is
actually asked for.

This runs the repository's shared suite,
``models/common/readiness_check/vllm_prompts.txt`` (six prompts: haiku, an
explanation, a story continuation, a factual list, a translation, and a Python
function), through the real generator on the real mesh -- **both** greedy and a
sampled leg per prompt, which also exercises the top-k/top-p path end to end on
the full 48-layer model rather than only in the unit tests.

Output goes two places:

* ``doc/full_model/qualitative_check.log`` -- the completions, to be read;
* ``<model-dir>/readiness_qualitative/vllm_qualitative_outputs.json`` -- the
  schema ``models/common/readiness_check/check_degenerate_output.py --scope
  vllm`` already knows how to score, so the suite is machine-gated as well as
  read.

    python .../probes/qualitative_probe.py --layers 48 --gen-len 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import build_generator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import DEFAULT_TRACE_REGION_SIZE  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[3]
PROMPTS = Path(__file__).resolve().parents[6] / "models" / "common" / "readiness_check" / "vllm_prompts.txt"


def read_prompts() -> list[str]:
    text = PROMPTS.read_text(encoding="utf-8")
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    prompts = read_prompts()
    print(f"{len(prompts)} prompts from {PROMPTS}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MC.MESH_SHAPE), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    results = []
    try:
        t0 = time.perf_counter()
        gen = build_generator(
            str(MODEL_DIR), mesh, override_num_layers=args.layers, max_context_len=args.context, max_batch_size=1
        )
        print(f"weight load {time.perf_counter() - t0:.1f} s", flush=True)

        for index, prompt in enumerate(prompts, 1):
            rendered = gen.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
            ids = gen.tokenizer(rendered, add_special_tokens=False)["input_ids"]
            print(f"\n=== prompt {index}/{len(prompts)}: {prompt!r} ({len(ids)} tokens) ===", flush=True)

            row = {"prompt": prompt}
            for label, kwargs in (
                ("greedy", dict(top_k=1, top_p=0.0, temperature=1.0)),
                ("sampled", dict(top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)),
            ):
                gen.reset()
                out = gen.generate(
                    ids, args.gen_len, enable_trace=True, sampling_mode="device", stop_on_eos=True, **kwargs
                )
                text = gen.tokenizer.decode(out, skip_special_tokens=True)
                row[f"{label}_completion"] = text
                print(f"--- completion ({label}, {len(out)} tokens) ---\n{text}", flush=True)
            results.append(row)

        gen.teardown()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    out_dir = MODEL_DIR / "readiness_qualitative"
    out_dir.mkdir(exist_ok=True)
    target = out_dir / "vllm_qualitative_outputs.json"
    target.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} prompts x 2 completions to {target}", flush=True)


if __name__ == "__main__":
    main()
