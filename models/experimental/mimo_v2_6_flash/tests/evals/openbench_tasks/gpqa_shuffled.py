# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPQA-Diamond as OpenRouter's benchmark harness runs it: openbench's prompt, data (nmayorga7/gpqa_diamond), answer
extraction and settings (temperature 0.5, 10 epochs), but the four options shuffled deterministically PER QUESTION.

openbench's own gpqa_diamond reseeds random with 0 before every record, so every question gets the same permutation
and the correct answer is always option B; OpenRouter's docs say their harness shuffles by question instead and that
its numbers are therefore not comparable with openbench's. OpenRouter does not publish its exact shuffle; this one
seeds from a hash of the question text, so it is stable across runs, machines and endpoints.

  inspect eval gpqa_shuffled.py@gpqa_diamond_shuffled --model openai-api/tt/<model> ...   (see run_eval.sh)
"""

import hashlib
import random

from inspect_ai import Epochs, task
from inspect_ai.model import GenerateConfig
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE, SIMPLE_EVALS_SYSTEM_MESSAGE


def record_to_mcq_sample(record: dict) -> MCQSample:
    options = [
        record["Correct Answer"],
        record["Incorrect Answer 1"],
        record["Incorrect Answer 2"],
        record["Incorrect Answer 3"],
    ]
    # A seeded generator on purpose: the permutation must be the same on every run, machine and endpoint, or the
    # runs are not comparable (module docstring). Nothing secret or security-relevant is derived from it, and the
    # system's unseedable source (secrets / SystemRandom) cannot give a reproducible shuffle.
    seed = int.from_bytes(hashlib.sha256(record["Question"].encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(options)
    return MCQSample(
        input=MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=record["Question"],
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3],
        ),
        target="ABCD"[options.index(record["Correct Answer"])],
    )


@task
def gpqa_diamond_shuffled(epochs: int = 10, temperature: float = 0.5):
    return MCQEval(
        name="gpqa_diamond_shuffled",
        dataset_path="nmayorga7/gpqa_diamond",
        record_to_mcq_sample=record_to_mcq_sample,
        split="train",
        auto_id=True,
        prompt_template=SIMPLE_EVALS_SYSTEM_MESSAGE,
        config=GenerateConfig(temperature=temperature),
        epochs=Epochs(epochs),
    )
