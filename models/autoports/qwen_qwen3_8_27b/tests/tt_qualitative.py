# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run exact-template shared controls through the already-loaded full generator."""

import json


def run(gen, root, control_name="hf_qualitative.json", max_new_tokens=None, output_name="tt_qualitative.json"):
    control = json.loads((root / "doc/full_model" / control_name).read_text())
    generation_length = max_new_tokens or control["max_new_tokens"]
    stops = json.loads((gen.model.snapshot / "generation_config.json").read_text())["eos_token_id"]
    stops = set(stops if isinstance(stops, list) else [stops])
    rows = []
    for reference in control["outputs"]:
        ids = gen.tokenizer(reference["rendered"], add_special_tokens=False)["input_ids"]
        assert ids == reference["prompt_tokens"]
        tokens = gen.generate(ids, generation_length)
        end = next((j + 1 for j, token in enumerate(tokens) if token in stops), len(tokens))
        tokens = tokens[:end]
        mismatch = next((i for i, (a, b) in enumerate(zip(tokens, reference["tokens"])) if a != b), None)
        row = dict(
            prompt_id=reference["prompt_id"],
            prompt=reference["prompt"],
            rendered=reference["rendered"],
            prompt_tokens=ids,
            tokens=tokens,
            text=gen.tokenizer.decode(tokens),
            hf_text=reference["text"],
            first_divergence=mismatch,
            perf=gen.last_perf,
        )
        rows.append(row)
        print("QUALITATIVE", row["prompt_id"], row["text"], flush=True)
    result = {
        k: v
        for k, v in control.items()
        if k not in ("outputs", "seconds", "command", "control_batch_size", "left_padding")
    }
    result.update(outputs=rows, hf_control=control_name, implementation="full TP4, canonical split sampling")
    result["max_new_tokens"] = generation_length
    result["hf_control_max_new_tokens"] = control["max_new_tokens"]
    target = root / "doc/full_model" / output_name
    target.write_text(json.dumps(result, indent=2) + "\n")
    return target
