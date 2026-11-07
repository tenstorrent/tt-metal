# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""inspect-ai tasks for the Artificial Analysis protocols and the tau2 benchmarks with a Claude user simulator.

Artificial Analysis (artificialanalysis.ai/methodology/intelligence-benchmarking, read 2026-11-07):

- gpqa_diamond_aa: GPQA Diamond, 198 questions, the dataset AA cites (openai/simple-evals' gpqa_diamond.csv, options
  permuted as simple-evals does: one random.Random(0) over the file), AA's 4-option instruction prompt, no system
  prompt, AA's multi-stage answer regex (last match), pass@1. AA does not state repeats for GPQA (default 1 here).
- tau2_telecom_aa: tau2-bench telecom, the 114 default tasks, 3 trials each (AA: "3 repeats per task ... pass@1 as the
  average of the 3 attempts"), at most 100 steps per trial, default dual-control mode. AA's user simulator is
  Qwen3 235B A22B 2507 (non-reasoning); here it is ``user_model`` (default Claude Haiku 5.5): a confound.
- Both: AA samples reasoning models at temperature 0.6 unless the lab recommends another; MiMo's card recommends
  1.0 / top_p 0.95, the defaults here (an inference: AA does not publish per-model settings).

tau2_airline_openrouter: OpenRouter's Auto Exacto tau2 (airline, temperature 0, 1 trial, openbench's 200-step cap) with
the same user simulator choice.

Claude as the user simulator: current Claude models reject sampling parameters and think adaptively by default, so
``claude_user_model_guard`` (applied on import) drops temperature / top_p / top_k from every Anthropic request of a
model newer than Claude 3.x, and raises the default output budget from inspect's 4096 (thinking shares it) to
``CLAUDE_USER_MAX_TOKENS``. The model id is passed as given (``anthropic/claude-haiku-5-5``, no date suffix); inspect's
Anthropic provider reads ANTHROPIC_API_KEY.
"""

import copy
import csv
import io
import os
import random
import re
import urllib.request

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

GPQA_CSV = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
CLAUDE_USER_MAX_TOKENS = int(os.environ.get("MM_CLAUDE_USER_MAX_TOKENS", "16000"))
DEFAULT_USER_MODEL = "anthropic/claude-haiku-5-5"

# ---- Artificial Analysis multiple-choice prompt and extraction (quoted from the methodology page)

PRIMARY = re.compile(r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])")
FALLBACKS = [
    re.compile(r"\\boxed\{[^}]*([A-Z])[^}]*\}"),
    re.compile(r"answer is ([a-zA-Z])"),
    re.compile(r"answer is \(([a-zA-Z])"),
    re.compile(r"([A-Z])\)\s*[^A-Z]*"),
    re.compile(r"([A-Z])\s+is\s+the\s+correct\s+answer"),
    re.compile(r"([A-Z])\s*$"),
    re.compile(r"([A-Z])\s*\."),
    re.compile(r"([A-Z])\s*[^\w]"),
]


def aa_prompt(question, options):
    letters = "ABCDEFGHIJ"[: len(options)]
    head = (
        "Answer the following multiple choice question. The last line of your response should be in the following "
        f"format: 'Answer: {'/'.join(letters)}' (e.g. 'Answer: A')."
    )
    return "\n".join([head, question] + [f"{letter}) {option}" for letter, option in zip(letters, options)])


def aa_extract(response):
    text = (response or "").strip()
    if len(text) == 1 and text.isalpha():
        return text.upper()
    for pattern in [PRIMARY] + FALLBACKS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    return None


@scorer(metrics=[accuracy(), stderr()])
def aa_choice():
    async def score(state: TaskState, target: Target) -> Score:
        answer = aa_extract(state.output.completion)
        return Score(value=CORRECT if answer == target.text else INCORRECT, answer=answer)

    return score


def gpqa_samples(path=None):
    """GPQA Diamond from simple-evals' CSV (MM_GPQA_CSV or a local path overrides the URL), options permuted as
    simple-evals permutes them."""
    path = path or os.environ.get("MM_GPQA_CSV") or GPQA_CSV
    if path.startswith("http"):
        with urllib.request.urlopen(path, timeout=60) as resp:
            text = resp.read().decode("utf-8")
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    rows = list(csv.DictReader(io.StringIO(text)))
    rng = random.Random(0)
    samples = []
    for i, row in enumerate(rows):
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        permutation = rng.sample(range(4), 4)
        choices = [choices[j] for j in permutation]
        target = "ABCD"[choices.index(row["Correct Answer"])]
        samples.append(Sample(id=i, input=aa_prompt(row["Question"], choices), target=target))
    return samples


@task
def gpqa_diamond_aa(epochs: int = 1, temperature: float = 1.0, top_p: float = 0.95, max_tokens: int = 131072):
    return Task(
        dataset=MemoryDataset(gpqa_samples(), name="gpqa_diamond_simple_evals"),
        solver=[generate()],
        scorer=aa_choice(),
        config=GenerateConfig(temperature=temperature, top_p=top_p, max_tokens=max_tokens),
        epochs=Epochs(epochs),
        name="gpqa_diamond_aa",
    )


# ---- tau2 with a Claude user simulator


def claude_user_model_guard():
    """Patch inspect's Anthropic provider for current Claude models: no sampling parameters (they answer 400), and a
    default output budget of CLAUDE_USER_MAX_TOKENS instead of 4096 (adaptive thinking shares it)."""
    try:
        from inspect_ai.model._providers.anthropic import AnthropicAPI
    except ImportError:  # the anthropic SDK is not installed: no Claude model can be used anyway
        return False
    if getattr(AnthropicAPI, "_mm_guarded", False):
        return True
    original_config, original_max = AnthropicAPI.completion_config, AnthropicAPI.max_tokens

    def current(api):
        return not re.match(r"claude-(3|instant|2)", api.service_model_name())

    def completion_config(self, config):
        params, headers, betas = original_config(self, config)
        if current(self):
            for key in ("temperature", "top_p", "top_k"):
                params.pop(key, None)
        return params, headers, betas

    def max_tokens(self):
        return CLAUDE_USER_MAX_TOKENS if current(self) else original_max(self)

    AnthropicAPI.completion_config, AnthropicAPI.max_tokens = completion_config, max_tokens
    AnthropicAPI._mm_guarded = True
    return True


def patch_tau2_tool_descriptions():
    """openbench 0.5.3 turns tau2's tool schemas into inspect tools, and inspect refuses a parameter without a
    description; tau2@558e6cd's telecom user tools have some (set_network_mode_preference's ``mode``), so every
    telecom sample fails before the first user turn. Give such a parameter its name as description."""
    import openbench.solvers.tau_bench as ob

    if getattr(ob, "_mm_patched", False):
        return
    original = ob._tool_from_schema

    def tool_from_schema(schema):
        schema = copy.deepcopy(schema)
        params = (schema.get("function") or {}).get("parameters") or {}
        for pname, prop in (params.get("properties") or {}).items():
            if isinstance(prop, dict) and not prop.get("description"):
                prop["description"] = pname.replace("_", " ")
        return original(schema)

    ob._tool_from_schema = tool_from_schema
    ob._mm_patched = True


def _tau2_task(domain, name, user_model, max_steps, trials, temperature, top_p):
    from openbench.datasets.tau_bench import ensure_tau2_data_dir, get_tau_bench_dataset

    ensure_tau2_data_dir()
    patch_tau2_tool_descriptions()
    from openbench.scorers.tau_bench import tau_bench_scorer
    from openbench.solvers.tau_bench import tau_bench_solver

    config = GenerateConfig(temperature=temperature)
    if top_p is not None:
        config = GenerateConfig(temperature=temperature, top_p=top_p)
    return Task(
        dataset=get_tau_bench_dataset(domain, num_trials=trials),
        solver=[tau_bench_solver(user_model=user_model, max_steps=max_steps)],
        scorer=tau_bench_scorer(),
        config=config,
        name=name,
    )


@task
def tau2_telecom_aa(
    user_model: str = DEFAULT_USER_MODEL,
    trials: int = 3,
    max_steps: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.95,
):
    return _tau2_task("telecom", "tau2_telecom_aa", user_model, max_steps, trials, temperature, top_p)


@task
def tau2_airline_openrouter(user_model: str = DEFAULT_USER_MODEL, trials: int = 1, max_steps: int = 200):
    return _tau2_task("airline", "tau2_airline_openrouter", user_model, max_steps, trials, 0.0, None)


claude_user_model_guard()
