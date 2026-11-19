# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collect long sampled reasoning traces, retaining each answer through first EOS."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-kwargs-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-texts", type=int, default=4)
    args = parser.parse_args()
    import readiness_check.run_precision_reasoning as p1
    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.common.sampling.generator import SamplingParams

    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    kwargs = json.loads(args.build_kwargs_file.read_text())
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(root, mesh, **kwargs)
        texts = []
        # The existing matched quality controls use these checkpoint/card settings.
        # Keep the P1 answer cap of6144; reduce sample count, not answer length.
        for i, prompt in enumerate(p1.load_prompts(p1.DEFAULT_PROMPTS)):
            if len(texts) == args.num_texts:
                break
            prompt_ids = p1.chat_prompt_ids(gen.tokenizer, prompt["messages"], p1.DEFAULT_CHAT_TEMPLATE_KWARGS)
            params = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, seed=1000 + i)
            answer = gen.generate(prompt_ids, 6144, sampling_params=params)
            eos = gen.tokenizer.eos_token_id
            if eos in answer:
                answer = answer[: answer.index(eos) + 1]
            if len(answer) < p1.DEFAULT_MIN_ANSWER_TOKENS or p1.is_degenerate(answer):
                print("DROP", prompt["id"], len(answer), flush=True)
                continue
            row = {
                "id": prompt["id"],
                "messages": prompt["messages"],
                "prompt_token_ids": prompt_ids,
                "completion_token_ids": answer,
                "response": gen.tokenizer.decode(answer),
                "rendered_prompt": gen.tokenizer.decode(prompt_ids),
                "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "seed": 1000 + i},
                "max_new_tokens": 6144,
                "stop": "first EOS" if answer[-1] == eos else "cap",
                "precision_config": gen.model.precision_policy,
                "prompt_mode": "checkpoint chat template; thinking enabled",
            }
            texts.append(row)
            args.output.write_text("".join(json.dumps(t) + "\n" for t in texts))
            print("TEXT", prompt["id"], len(answer), row["stop"], flush=True)
        assert len(texts) == args.num_texts
        assert sum(len(t["completion_token_ids"]) for t in texts) >= 1000
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
