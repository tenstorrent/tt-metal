# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the efficiency of LLM inference with traced prefill and prefix caching.

Loads a ``vllm.LLM`` instance with given options, collects prompts with a shared prefix
from local text files, and runs inference. Evaluation results are logged and/or saved.
The main goal is to investigate if traced (trace_mode="all") prefill helps reduce
monitored metrics: TTFT and prefill latency (and hence end-to-end latency).

Usage:
    $ python3 run.py \
        -m Qwen/Qwen3-4B-Instruct-2507 \
        -p 20 -w 3 -v 1 \
        -t none decode_only all
    $ # You'll see something like this as an output:
    Cross-Mode Summary (lower is better):
    +-------------+-----------------+------------------------------+
    | trace_mode  | avg_ttft (sec.) | ... | avg_e2e_latency (sec.) |
    +-------------+-----------------+-...-+------------------------+
    | none        |          *1.567 | ... |                  6.397 |
    | decode_only |           1.568 | ... |                  3.591 |
    | all         |           1.568 | ... |                 *3.581 |
    +-------------+-----------------+-...-+------------------------+
    * marks the lowest value per metric (ties allowed).
    Saved benchmark results to: results.json
"""

import argparse
import collections
import dataclasses
import functools
import json
import os
import pprint
import statistics
import textwrap
import time
from pathlib import Path
from typing import Literal

LOGGER_LEVEL = os.getenv("TRACED_PREFILL_BENCHMARK_LOGGER_LEVEL") or "ERROR"
os.environ["TT_LOGGER_LEVEL"] = LOGGER_LEVEL
os.environ["TT_METAL_LOGGER_LEVEL"] = LOGGER_LEVEL
os.environ["LOGURU_LEVEL"] = LOGGER_LEVEL
os.environ["VLLM_LOGGING_LEVEL"] = LOGGER_LEVEL

from tqdm import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


TraceModesT = Literal["all", "decode_only", "none"]

RANDOM_STATE: int = 43165

PARENT_DIR = Path(__file__).parent
DEFAULT_SYSTEM_PROMPT_PATH = PARENT_DIR / "system-prompt.txt"
DEFAULT_CONTEXT_PATH = PARENT_DIR / "context.txt"
DEFAULT_QUESTIONS_PATH = PARENT_DIR / "questions.txt"


def main() -> None:
    """Main function: parses CLI, benchmarks, and saves / reports results."""
    options = parse_options()
    sampling_params = load_sampling_params(options)
    prompts = build_prompts(options)
    results = benchmark_everything(sampling_params, prompts, options)
    maybe_print_metrics_table(results, options)
    maybe_save_results(results, options)


def parse_options() -> argparse.Namespace:
    """Parses CLI and returns parsed options such as model name."""

    class RawTextDefaultsFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawTextHelpFormatter,
    ):
        """Preserves CLI text formatting and also shows option defaults."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the efficiency of combination(s) of automatic prefix caching \n"
            "and traced prefill on a TT transformer.\n"
            "Collects shared context and multiple prompt tails from ./*.txt files, \n"
            "loads an LLM with provided options, generates responses on the prompts, \n"
            "and saves & reports the inference-evaluation results. The main goal is to \n"
            "make sure that traced prefill reduces TTFT and prefill-latency metrics.\n"
            "See also https://github.com/tenstorrent/tt-metal/pull/43165"
        ),
        formatter_class=RawTextDefaultsFormatter,
        usage="python %(prog)s [LLM options] [sampling options] [I/O options]"
    )
    parser.add_argument(
        "-m",
        "--model",
        help=(
            "Huggingface model name, format 'org/name'. \n"
            "Currently support text-generation models such as Qwen3.\n"
        ),
        default="Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument(
        "-b",
        "--block_size",
        help="KV-cache block size.\n",
        type=int,
        default=64,
    )
    parser.add_argument(
        "-D",
        "--disable_prefix_caching",
        help="Whether to disable automatic prefix caching through vLLM.\n",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--trace_mode",
        help=(
            "TT tracing mode(s) to benchmark.\nPass one or more values, for example: "
            "--trace_mode none decode_only all\n"
        ),
        nargs="+",
        default=["all"],
        choices=["none", "decode_only", "all"],
    )
    parser.add_argument(
        "-s",
        "--max_num_seqs",
        help="Maximum number of sequences processed simultaneously.\n",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-c",
        "--context_multiply",
        help=(
            "Context-length multiplier: shared context (excluding system prompt) will\n"
            "be multiplied by this number; use for experiments with longer context.\n"
            "Default context has 1,662 tokens and system prompt 730 tokens.\n"
        ),
        type=int,
        default=1,
    )
    parser.add_argument(
        "-g",
        "--max_tokens",
        help="Maximum number of tokens to generate.\n",
        type=int,
        default=96,
    )
    parser.add_argument(
        "-w",
        "--warmup_runs",
        help="Number of warmup runs before timed runs.\n",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-p",
        "--num_prompts",
        help=(
            "Number of realistic prompts to use from the built-in set.\n"
            "If greater than the prompts from the tails file, repeated.\n"
        ),
        type=int,
        default=32,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional path to save detailed benchmark rows and summary.\n",
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help=(
            "Verbosity level (0 = no outputs, 1 = only summary,\n"
            "2 = inference results such as prompts, responses, and metrics).\n"
        ),
        type=int,
        default=1,
        choices=(0, 1, 2),
    )
    return parser.parse_args()


def load_sampling_params(options: argparse.Namespace) -> SamplingParams:
    """Loads ``vllm.SamplingParams`` with parsed CLI arguments from ``options``."""
    if options.verbose > 0:
        print(
            f"\n🎲 LLM will generate maximum {options.max_tokens} tokens with "
            f"temperature = 1.0 and no top-{{p,k}} sampling."
        )
    return SamplingParams(
        max_tokens=options.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        seed=RANDOM_STATE,
    )


def build_prompts(options: argparse.Namespace) -> list[str]:
    """Builds ``options.num_prompts`` number of realistic prompts with shared prefix.

    Context (excluding system prompt) can be multiplied as requested by
    ``options.context_multiply``. If ``options.verbose``, prints the number of tokens
    in obtained shared prefix.
    """
    # region Load local system, prefix, and user prompts
    system = DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    context = (
        DEFAULT_CONTEXT_PATH.read_text(encoding="utf-8").strip()
        * options.context_multiply
    )
    shared_prefix = f"{system}\n---\n{context}"
    prompt_tails = DEFAULT_QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    # endregion

    # region Maybe print token stats
    if options.verbose > 0:
        tokenizer = AutoTokenizer.from_pretrained(options.model)
        shared_tokens_map = tokenizer(shared_prefix, return_length=True)
        n_shared_tokens = shared_tokens_map["length"][0]
        print(f"\n📚 Shared context has {n_shared_tokens:,} tokens.")
    # endregion

    # If requested number of prompts is smaller than available prompt tails,
    # return the subset. Otherwise, just repeat the prompts.
    num_prompt_tails = len(prompt_tails)
    num_repeat_prompt_tails = options.num_prompts // num_prompt_tails + 1
    return [
        (
            f"<assistant>\n{shared_prefix}\n</assistant>\n\n"
            f"<user>\n{tail}\n</user>\n\n"
            f"<response/>\n"
        )
        for tail in (prompt_tails * num_repeat_prompt_tails)[: options.num_prompts]
    ]


def benchmark_everything(
    sampling_params: SamplingParams,
    prompts: list[str],
    options: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    """Loads LLM from ``options`` w/ 1 ore more trace modes. Evals and returns summary.

    If ``options`` requests warmup, performs it on the first prompt requested number of
    times. Metrics are returned as name-to-value mapping.
    """
    return {
        t: benchmark(sampling_params, prompts, options, t)
        for t in options.trace_mode
    }


def benchmark(
    sampling_params: SamplingParams,
    prompts: list[str],
    options: argparse.Namespace,
    trace_mode: TraceModesT,
) -> dict[str, float]:
    """Loads LLM from ``options`` w/ ``trace_mode``. Evals & returns inference metrics.

    If ``options`` requests warmup, performs it on the first prompt requested number of
    times. Metrics are returned as name-to-value mapping.
    """
    llm = load_llm(options, trace_mode)
    warmup_prompts(llm, sampling_params, prompts, options.warmup_runs)
    metrics_list = run_prompts(llm, sampling_params, prompts, options.verbose)
    metrics_summary = summarize_metrics(metrics_list, options.verbose)
    return metrics_summary


def load_llm(options: argparse.Namespace, trace_mode: TraceModesT) -> LLM:
    """Loads ``vllm.LLM`` with parsed CLI arguments from ``options``."""
    if options.verbose > 0:
        apc_text = "disabled" if options.disable_prefix_caching else "enabled"
        print(
            f"\n🤖 Preparing LLM '{options.model}' for inference with {apc_text} "
            f"prefix caching,\n   block size {options.block_size}, "
            f"{options.max_num_seqs} simultaneously processed sequences, "
            f"and with trace mode '{trace_mode}'.\n"
        )
    return LLM(
        model=options.model,
        block_size=options.block_size,
        max_num_seqs=options.max_num_seqs,
        enable_prefix_caching=not options.disable_prefix_caching,
        override_tt_config={"trace_mode": trace_mode},
        use_tqdm_on_load=False,
        disable_log_stats=False,
        seed=RANDOM_STATE,
    )


def warmup_prompts(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[str],
    warmup_runs: int = 1,
) -> None:
    """Warms up tracing and kernel compilation with a single prompt.

    Runs the first prompt ``warmup_runs`` times to initialize caches and compile
    traces, without recording metrics. This stabilizes measurements but uses a
    single warmup prompt to avoid contaminating the prefix-cache state of the
    benchmark prompts.
    """
    if warmup_runs <= 0:
        return

    warmup_prompt = prompts[0]
    for _ in tqdm(range(warmup_runs), desc="Warming up"):
        llm.generate(warmup_prompt, sampling_params=sampling_params, use_tqdm=False)


def run_prompts(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[str],
    verbose: bool = False,
) -> list["InferenceMetrics"]:
    """Runs ``llm`` inference on ``prompts`` with ``sampling_params`` configuration.

    Returns a list of ``InferenceMetrics`` objects each containing vLLM metrics and
    texts.
    """
    metrics_list = []

    for i, prompt in enumerate(tqdm(prompts, desc="Running inference on prompts")):
        metrics = InferenceMetrics.from_llm(llm, sampling_params, prompt, id_=i)
        metrics_list.append(metrics)

        if verbose == 2:
            print(f"💬 Example {i}:\n{metrics!r}")

    return metrics_list


@dataclasses.dataclass
class InferenceMetrics:
    """Stores inference-evaluation metrics such as TTFT and e2e latency.

    Examples
    --------
    >>> # If you've already collected every metric, initialize the class:
    >>> metrics = InferenceMetrics(
    ...     id=1,
    ...     prompt="You are..",
    ...     response="The release...",
    ...     ttft=0.125,
    ...     prefill_latency=0.122,
    ...     decode_latency=3.229,
    ...     e2e_latency=3.355)
    >>> print(metrics)
    {'id': 1,
     'prompt': 'You are a production assistant for a cloud deployment team. Follow '
               'these rules: give concise, actionable answers; include assumptions '
               'and risks; prefer numbered steps for runbooks; ask one clarifying '
               'question if data is missing. Context: Region=us-east-1, Service '
               'tier=enterprise, Incident policy=P1 response [...]',
     'response': 'The release is failing due to a misconfigured load balancer. The '
                 'canary release is currently at 20% traffic. The user has not yet '
                 'confirmed the rollback. The incident is currently in P1 status. '
                 'The user has not yet confirmed the rollback. The incident is '
                 'currently in P1 status. The user has not yet confirmed [...]',
     'ttft (sec.)': 0.125,
     'prefill_latency (sec.)': 0.122,
     'decode_latency (sec.)': 3.229,
     'e2e_latency (sec.)': 3.355}
    >>> # otherwise, get from vLLM's ``LLM`` and ``SamplingParams``:
    >>> metrics_from_llm = InferenceMetrics.from_llm(llm, sampling_params, "You are...")
    """

    id: int
    prompt: str
    response: str
    ttft: float
    prefill_latency: float
    decode_latency: float
    e2e_latency: float

    def __post_init__(self) -> None:
        self.prompt = self._wrap_text(self.prompt)
        self.response = self._wrap_text(self.response)

    @staticmethod
    def _wrap_text(text: str) -> str:
        return textwrap.shorten(
            "\n".join(textwrap.wrap(text, width=79)),
            width=316,
            placeholder=" [...]",
        )

    def __repr__(self) -> str:
        """Pretty-prints all metrics and metadata as a dictionary."""
        return pprint.pformat(self.mapping, sort_dicts=False)

    @functools.cached_property
    def mapping(self) -> dict[str, int | float | str]:
        """Returns attribute-to-value mapping of class.

        This property is more readable and for printing and serialization purposes.
        """
        metrics_dict = dataclasses.asdict(self)
        metrics_ordered_dict = collections.OrderedDict()

        for name in ("id", "prompt", "response"):
            metrics_ordered_dict[name] = metrics_dict[name]

        for name in ("ttft", "prefill_latency", "decode_latency", "e2e_latency"):
            metrics_ordered_dict[f"{name} (sec.)"] = metrics_dict[name]

        return dict(**metrics_ordered_dict)

    @classmethod
    def from_llm(
        cls,
        llm: LLM,
        sampling_params: SamplingParams,
        prompt,
        id_: int,
    ) -> "InferenceMetrics":
        """Runs ``llm`` inference on ``prompt`` and collects inference metrics."""
        start_time = time.perf_counter()
        output = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
        e2e_latency = round(time.perf_counter() - start_time, 3)

        response = output[0].outputs[0].text
        metrics = output[0].metrics
        ttft = round(metrics.first_token_latency, 3)
        prefill_latency = round(metrics.first_token_ts - metrics.scheduled_ts, 3)
        decode_latency = round(metrics.last_token_ts - metrics.first_token_ts, 3)

        return cls(
            id=id_,
            prompt=prompt,
            response=response,
            ttft=ttft,
            prefill_latency=prefill_latency,
            decode_latency=decode_latency,
            e2e_latency=e2e_latency,
        )


def summarize_metrics(
    metrics_list: list[InferenceMetrics],
    verbose: int = 1,
) -> dict[str, float]:
    """Collects metric statistics for each inference metric per prompt.

    Returns metric_name-to-average_score mapping.
    """
    ttfts = []
    prefill_latencies = []
    decode_latencies = []
    e2e_latencies = []

    for metrics in metrics_list:
        ttfts.append(metrics.ttft)
        prefill_latencies.append(metrics.prefill_latency)
        decode_latencies.append(metrics.decode_latency)
        e2e_latencies.append(metrics.e2e_latency)

    summary = {
        "avg_ttft (sec.)": round(statistics.median(ttfts), 3),
        "avg_prefill_latency (sec.)": round(statistics.median(prefill_latencies), 3),
        "avg_decode_latency (sec.)": round(statistics.median(decode_latencies), 3),
        "avg_e2e_latency (sec.)": round(statistics.median(e2e_latencies), 3),
    }
    if verbose > 0:
        print("\n📉 Inference-Evaluation Summary:")
        pprint.pprint(summary, sort_dicts=False)

    return summary


def maybe_print_metrics_table(
    summaries: dict[str, dict[str, object]],
    options: argparse.Namespace,
) -> None:
    """Prints a compact summary table with per-metric minima highlighted.

    The output is intentionally close to a polars-like pretty table while using
    plain ASCII for broad terminal compatibility. If ``options.verbose`` is 0,
    nothing happens.
    """
    if options.verbose == 0:
        return

    first_summary = next(iter(summaries.values()))
    if not isinstance(first_summary, dict):
        return

    metric_names = list(first_summary.keys())
    minima = {
        metric_name: min(summary[metric_name] for summary in summaries.values())
        for metric_name in metric_names
    }

    headers = ["trace_mode", *metric_names]
    rows = []
    for trace_mode, summary in summaries.items():
        row = [trace_mode]
        for metric_name in metric_names:
            value = float(summary[metric_name])
            marker = "*" if value == minima[metric_name] else " "
            row.append(f"{marker}{value:.3f}")
        rows.append(row)

    col_widths = []
    for idx, header in enumerate(headers):
        max_cell_width = max(len(row[idx]) for row in rows)
        col_widths.append(max(len(header), max_cell_width))

    border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print("\n🧮 Cross-Mode Summary (lower is better):")
    print(border)

    header_cells = [f" {headers[i].ljust(col_widths[i])} " for i in range(len(headers))]
    print("|" + "|".join(header_cells) + "|")
    print(border)

    for row in rows:
        cells = [f" {row[0].ljust(col_widths[0])} "]
        for i in range(1, len(row)):
            cells.append(f" {row[i].rjust(col_widths[i])} ")
        print("|" + "|".join(cells) + "|")

    print(border)
    print("* marks the lowest value per metric (ties allowed).")

    print(
        "\n🎸 Reminder: benchmarking was done with the following configuration\n"
        "   (if not specified explicitly, other parameters were set to defaults):"
    )
    pprint.pprint(
        {
            "model": options.model,
            "block_size": options.block_size,
            "enable_prefix_caching": not options.disable_prefix_caching,
            "max_num_seqs": options.max_num_seqs,
        },
        sort_dicts=False,
    )


def maybe_save_results(
    results: dict[str, dict[str, float]],
    options: argparse.Namespace,
) -> None:
    """Saves benchmark ``results`` in file ``options.output`` if filename is provided."""
    if options.output is None:
        return

    with open(options.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    if options.verbose > 0:
        print(f"Saved benchmark results to: {options.output}")


if __name__ == "__main__":
    main()
