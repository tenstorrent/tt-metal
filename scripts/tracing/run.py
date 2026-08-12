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
        -p 20 -w 0 -v 1 \
        -t none decode_only all \
        -r 0.0 33.0 67.0 100.0
        -c 0.5 1.0 2.0
    $ # You'll see something like this as an output:
    🧮 Aggregate Cross-Mode Summary (lower is better):
    +-------------+--------------------+-...-+---------------------------+
    | trace_mode  | median_ttft (sec.) | ... | median_e2e_latency (sec.) |
    +-------------+--------------------+-...-+---------------------------+
    | none        |              1.567 | ... |                     6.397 |
    | decode_only |              0.968 | ... |                     3.591 |
    | all         |             *0.755 | ... |                    *2.551 |
    +-------------+--------------------+-...-+---------------------------+
    * marks the lowest value per metric (ties allowed).
    Saved benchmark results to: results.json
"""

import argparse
import collections
import dataclasses
import functools
import itertools
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
    results = benchmark_everything(sampling_params, options)
    maybe_print_metrics_tables(results, options)
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
        usage="python %(prog)s [LLM options] [sampling options] [I/O options]",
    )
    parser.add_argument(
        "-m",
        "--model",
        help=(
            "Huggingface model name, format 'org/name'. \n" "Currently support text-generation models such as Qwen3.\n"
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
        "-r",
        "--prefix_caching_ratio",
        help=(
            "Percent of the entire shared prompt prefix (system + context) fixed\n"
            "for prefix caching. 0.0 means no prefix caching, 50.0 means half of\n"
            "all prefix tokens are fixed, and 100.0 means full prefix caching.\n"
        ),
        nargs="+",
        default=[0.0, 50.0, 100.0],
        type=float,
    )
    parser.add_argument(
        "-t",
        "--trace_mode",
        help=(
            "TT tracing mode(s) to benchmark.\nPass one or more values, for example: "
            "--trace_mode none decode_only all\n"
        ),
        nargs="+",
        default=["decode_only", "all"],
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
            "Context-length multiplier (real-valued): shared context (excluding\n"
            "system prompt) is scaled by token count. For example, 0.4 keeps 40%%\n"
            "of context tokens and 1.5 means full context plus half of it.\n"
            "Default context has 1,662 tokens and system prompt 730 tokens.\n"
        ),
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0],
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
        "--winner_delta_threshold",
        help=(
            "Minimum prefill-latency gap (seconds) required to declare a strict winner\n"
            "in per-setting comparison. If mode differences are within this threshold,\n"
            "they are treated as a tie.\n"
        ),
        type=float,
        default=0.001,
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


def benchmark_everything(
    sampling_params: SamplingParams,
    options: argparse.Namespace,
) -> dict[str, list[dict[str, object]]]:
    """Runs benchmarks on the full grid and returns one row per setting/mode pair.

    The grid is ``trace_mode`` x ``prefix_caching_ratio`` x ``context_multiply``.
    """
    rows: list[dict[str, object]] = []

    for context_multiply, prefix_caching_ratio in itertools.product(
        options.context_multiply, options.prefix_caching_ratio
    ):
        prompts, context_stats = build_prompts(options, context_multiply, prefix_caching_ratio)
        for trace_mode in options.trace_mode:
            metrics_summary = benchmark(
                sampling_params=sampling_params,
                prompts=prompts,
                options=options,
                trace_mode=trace_mode,
                prefix_caching_ratio=prefix_caching_ratio,
            )
            rows.append(
                {
                    "trace_mode": trace_mode,
                    "prefix_caching_ratio (%)": prefix_caching_ratio,
                    "context_multiply": context_multiply,
                    "context_tokens_cached": context_stats["context_tokens_cached"],
                    "context_tokens_total": context_stats["context_tokens_total"],
                    **metrics_summary,
                }
            )

    return {"rows": rows}


def build_prompts(
    options: argparse.Namespace,
    context_multiply: float,
    prefix_caching_ratio: float,
) -> tuple[list[str], dict[str, int]]:
    """Builds ``options.num_prompts`` prompts using one context/cache configuration.

    Context (excluding system prompt) can be multiplied as requested by
    ``options.context_multiply``. If ``options.verbose``, prints the number of tokens
    in obtained shared prefix.
    """
    # region Load local system, context, and user prompts
    tokenizer = AutoTokenizer.from_pretrained(options.model)
    system = DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    base_context = DEFAULT_CONTEXT_PATH.read_text(encoding="utf-8").strip()
    prompt_tails = DEFAULT_QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()

    base_context_token_ids = tokenizer.encode(base_context, add_special_tokens=False)
    n_base_context_tokens = len(base_context_token_ids)
    n_target_context_tokens = max(0, int(n_base_context_tokens * context_multiply))

    if n_target_context_tokens == 0 or n_base_context_tokens == 0:
        context_token_ids: list[int] = []
    elif n_target_context_tokens <= n_base_context_tokens:
        context_token_ids = base_context_token_ids[:n_target_context_tokens]
    else:
        repeat_count, remainder = divmod(n_target_context_tokens, n_base_context_tokens)
        context_token_ids = base_context_token_ids * repeat_count + base_context_token_ids[:remainder]

    # Combine system and context to apply ratio to entire shared prefix
    system_token_ids = tokenizer.encode(system, add_special_tokens=False)

    full_prefix_token_ids = system_token_ids + context_token_ids
    n_full_prefix_tokens = len(full_prefix_token_ids)

    # Apply ratio to entire prefix
    n_fixed_total_tokens = int(n_full_prefix_tokens * prefix_caching_ratio / 100.0)

    shared_prefix = tokenizer.decode(full_prefix_token_ids[:n_fixed_total_tokens], skip_special_tokens=False).strip()
    nonfixed_prefix = tokenizer.decode(full_prefix_token_ids[n_fixed_total_tokens:], skip_special_tokens=False).strip()
    nonfixed_context = nonfixed_prefix
    # endregion

    # region Maybe print token stats
    if options.verbose > 0:
        n_nonfixed_total_tokens = n_full_prefix_tokens - n_fixed_total_tokens
        print(
            f"\n📚 Prefix token stats: full={n_full_prefix_tokens:,}, "
            f"cached={n_fixed_total_tokens:,}, non-cached={n_nonfixed_total_tokens:,} "
            f"(prefix_caching_ratio={prefix_caching_ratio:.1f}%)."
        )
    # endregion

    # If requested number of prompts is smaller than available prompt tails,
    # return the subset. Otherwise, just repeat the prompts.
    num_prompt_tails = len(prompt_tails)
    num_repeat_prompt_tails = options.num_prompts // num_prompt_tails + 1
    prompts = []
    for tail in (prompt_tails * num_repeat_prompt_tails)[: options.num_prompts]:
        user_body = tail if not nonfixed_context else f"{tail}\n\nAdditional context:\n{nonfixed_context}"
        prompt = f"<assistant>\n{shared_prefix}\n</assistant>\n\n<user>\n{user_body}\n</user>\n\n<response/>\n"
        prompts.append(prompt)

    return prompts, {
        "context_tokens_cached": n_fixed_total_tokens,
        "context_tokens_total": n_full_prefix_tokens,
    }


def benchmark(
    sampling_params: SamplingParams,
    prompts: list[str],
    options: argparse.Namespace,
    trace_mode: TraceModesT,
    prefix_caching_ratio: float,
) -> dict[str, float]:
    """Loads LLM from ``options`` w/ ``trace_mode``. Evals & returns inference metrics.

    If ``options`` requests warmup, performs it on the first prompt requested number of
    times. Metrics are returned as name-to-value mapping.
    """
    llm = load_llm(options, trace_mode, prefix_caching_ratio)
    warmup_prompts(llm, sampling_params, prompts, options.warmup_runs)
    metrics_list = run_prompts(llm, sampling_params, prompts, options.verbose)
    metrics_summary = summarize_metrics(metrics_list, options.verbose)
    return metrics_summary


def load_llm(options: argparse.Namespace, trace_mode: TraceModesT, prefix_caching_ratio: float) -> LLM:
    """Loads ``vllm.LLM`` with parsed CLI arguments from ``options``."""
    if options.verbose > 0:
        apc_text = "enabled" if prefix_caching_ratio > 0.0 else "disabled"
        print(
            f"\n🧠 Preparing LLM '{options.model}' for inference with {apc_text} "
            f"prefix caching ({prefix_caching_ratio:.1f}% fixed context),\n"
            f"   block size {options.block_size}, "
            f"{options.max_num_seqs} simultaneously processed sequences, "
            f"and with trace mode '{trace_mode}'.\n"
        )
    return LLM(
        model=options.model,
        block_size=options.block_size,
        max_num_seqs=options.max_num_seqs,
        enable_prefix_caching=prefix_caching_ratio > 0.0,
        additional_config={"tt": {"trace_mode": trace_mode}},
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
        "median_ttft (sec.)": round(statistics.median(ttfts), 3),
        "median_prefill_latency (sec.)": round(statistics.median(prefill_latencies), 3),
        "median_decode_latency (sec.)": round(statistics.median(decode_latencies), 3),
        "median_e2e_latency (sec.)": round(statistics.median(e2e_latencies), 3),
    }
    if verbose > 0:
        print("\n📉 Inference-Evaluation Summary:")
        pprint.pprint(summary, sort_dicts=False)

    return summary


def maybe_print_metrics_tables(
    results: dict[str, list[dict[str, object]]],
    options: argparse.Namespace,
) -> None:
    """Prints aggregate and per-setting summary tables for a parameter grid."""
    if options.verbose == 0:
        return

    rows = results.get("rows", [])
    if not rows:
        return

    trace_modes = list(options.trace_mode)
    by_setting: dict[tuple[float, float], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = _setting_key(row)
        by_setting.setdefault(key, {})[str(row["trace_mode"])] = row

    # Table 1: Aggregate summary over all settings.
    aggregate_headers = [
        "trace_mode",
        "median_ttft (sec.)",
        "median_prefill (sec.)",
        "median_e2e (sec.)",
    ]
    aggregate_rows_data: list[dict[str, float | str]] = []
    for mode in trace_modes:
        mode_rows = [row for row in rows if row["trace_mode"] == mode]
        ttfts = [float(row["median_ttft (sec.)"]) for row in mode_rows]
        prefills = [float(row["median_prefill_latency (sec.)"]) for row in mode_rows]
        e2es = [float(row["median_e2e_latency (sec.)"]) for row in mode_rows]
        aggregate_rows_data.append(
            {
                "mode": mode,
                "ttft": statistics.median(ttfts),
                "prefill": statistics.median(prefills),
                "e2e": statistics.median(e2es),
            }
        )

    # Find minima for each metric.
    min_ttft = min(d["ttft"] for d in aggregate_rows_data)
    min_prefill = min(d["prefill"] for d in aggregate_rows_data)
    min_e2e = min(d["e2e"] for d in aggregate_rows_data)

    # Build rows with `*` markers on best values.
    aggregate_rows: list[list[str]] = []
    for data in aggregate_rows_data:
        ttft_marker = "*" if data["ttft"] == min_ttft else " "
        prefill_marker = "*" if data["prefill"] == min_prefill else " "
        e2e_marker = "*" if data["e2e"] == min_e2e else " "
        aggregate_rows.append(
            [
                str(data["mode"]),
                f"{ttft_marker}{data['ttft']:.3f}",
                f"{prefill_marker}{data['prefill']:.3f}",
                f"{e2e_marker}{data['e2e']:.3f}",
            ]
        )

    _print_ascii_table(
        title="🧮 Aggregate Cross-Mode Summary (lower is better)",
        headers=aggregate_headers,
        rows=aggregate_rows,
        right_align_from=1,
    )
    print("* marks the lowest value per metric (ties allowed).")

    # Table 2: Per-setting prefill comparison with winner and delta against "all".
    per_setting_headers = [
        "prefix_ratio (%)",
        "cached_ctx_toks (#)",
        "full_ctx_toks (#)",
        *[f"prefill_{mode} (sec.)" for mode in trace_modes],
        "winner",
        "all_vs_best_other (sec.)",
        "all_vs_best_other (%)",
    ]

    per_setting_rows: list[list[str]] = []
    for (prefix_ratio, context_multiply), mode_rows in sorted(
        by_setting.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        prefill_values = {
            mode: float(mode_rows[mode]["median_prefill_latency (sec.)"]) for mode in trace_modes if mode in mode_rows
        }
        setting_sample = next(iter(mode_rows.values()))
        min_prefill = min(prefill_values.values())
        winners = "/".join(
            mode
            for mode in trace_modes
            if mode in prefill_values and abs(prefill_values[mode] - min_prefill) <= options.winner_delta_threshold
        )

        delta_sec = "n/a"
        delta_pct = "n/a"
        if "all" in prefill_values and len(prefill_values) > 1:
            best_other = min(value for mode, value in prefill_values.items() if mode != "all")
            all_value = prefill_values["all"]
            delta_value = all_value - best_other
            delta_sec = f"{delta_value:+.3f}"
            delta_pct = f"{(100.0 * delta_value / best_other):+.1f}%"

        per_setting_rows.append(
            [
                f"{prefix_ratio:.1f}",
                str(int(setting_sample["context_tokens_cached"])),
                str(int(setting_sample["context_tokens_total"])),
                *[f"{prefill_values[mode]:.3f}" if mode in prefill_values else "n/a" for mode in trace_modes],
                winners,
                delta_sec,
                delta_pct,
            ]
        )

    _print_ascii_table(
        title="📊 Per-Setting Prefill Comparison",
        headers=per_setting_headers,
        rows=per_setting_rows,
        right_align_from=0,
    )

    print("\n🎸 Reminder: benchmarking used these fixed options (unless overridden on CLI):")
    pprint.pprint(
        {
            "model": options.model,
            "block_size": options.block_size,
            "max_num_seqs": options.max_num_seqs,
            "trace_mode": options.trace_mode,
            "prefix_caching_ratio (%)": options.prefix_caching_ratio,
            "context_multiply": options.context_multiply,
            "winner_delta_threshold": options.winner_delta_threshold,
        },
        sort_dicts=False,
    )


def _print_ascii_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    right_align_from: int = 1,
) -> None:
    """Prints an ASCII table with configurable numeric alignment."""
    col_widths = []
    for idx, header in enumerate(headers):
        max_cell_width = max(len(row[idx]) for row in rows)
        col_widths.append(max(len(header), max_cell_width))

    border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(f"\n{title}")
    print(border)
    header_cells = [f" {headers[i].ljust(col_widths[i])} " for i in range(len(headers))]
    print("|" + "|".join(header_cells) + "|")
    print(border)

    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            formatted = cell.rjust(col_widths[i]) if i >= right_align_from else cell.ljust(col_widths[i])
            cells.append(f" {formatted} ")
        print("|" + "|".join(cells) + "|")
    print(border)


def _setting_key(row: dict[str, object]) -> tuple[float, float]:
    return (round(float(row["prefix_caching_ratio (%)"]), 6), round(float(row["context_multiply"]), 6))


def maybe_save_results(
    results: dict[str, list[dict[str, object]]],
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
