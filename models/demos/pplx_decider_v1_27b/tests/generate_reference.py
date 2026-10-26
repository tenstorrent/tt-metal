# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3_5Model

from models.demos.pplx_decider_v1_27b.tests.common import ids_sha256, load_example_state
from models.demos.pplx_decider_v1_27b.tt.decision import (
    DecisionConfig,
    answer,
    decision_probabilities,
    options,
    render_input_ids,
)
from models.demos.pplx_decider_v1_27b.tt.weight_mapping import PPLX_DECIDER_HF_MODEL, resolve_checkpoint

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Generate CPU fp32 reference outputs for pplx-decider-v1-27b.")
    parser.add_argument("--hf-model", default=os.environ.get("HF_MODEL", PPLX_DECIDER_HF_MODEL))
    parser.add_argument("--output", default=str(HERE / "reference" / "decision_reference.json"))
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    ckpt_dir = resolve_checkpoint(args.hf_model)
    config = DecisionConfig.from_checkpoint(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    readout = load_file(str(ckpt_dir / "readout.safetensors"))["weight"].float()
    # Qwen3_5Model.last_hidden_state is already after the final norm, which is where the readout applies.
    model = Qwen3_5Model.from_pretrained(ckpt_dir, dtype=torch.float32).eval()
    examples = json.loads((HERE / "examples.json").read_text())

    results, tops = [], []
    with torch.inference_mode():
        for example in examples:
            question = example["question"]
            state = load_example_state(example, ckpt_dir)
            ids = render_input_ids(tokenizer, state, question, config.codes)
            hidden = model(input_ids=torch.tensor(ids)[None], use_cache=False).last_hidden_state[0, -1]
            readout_logits = readout @ hidden.float()
            keys = options(question)[0]
            probabilities = decision_probabilities(readout_logits, len(keys), config.temperature)
            results.append(
                {
                    "name": example["name"],
                    "num_tokens": len(ids),
                    "input_ids_sha256": ids_sha256(ids),
                    "readout_logits": readout_logits.tolist(),
                    "probabilities": probabilities.tolist(),
                    "answer": answer(question, probabilities.tolist()),
                }
            )
            top = int(probabilities.argmax())
            tops.append((example["name"], len(ids), keys[top], float(probabilities[top])))

    reference = {
        "hf_model": PPLX_DECIDER_HF_MODEL,
        "revision": ckpt_dir.name,
        "dtype": "float32",
        "temperature": config.temperature,
        "examples": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reference, indent=2) + "\n")

    logger.info(f"{'example':<14}{'tokens':>8}  {'top option':<20}{'top prob':>9}")
    for name, num_tokens, key, prob in tops:
        logger.info(f"{name:<14}{num_tokens:>8}  {key:<20}{prob:>9.5f}")
    logger.info(f"wrote {output}")


if __name__ == "__main__":
    main()
