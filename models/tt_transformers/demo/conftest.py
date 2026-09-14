# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json

from models.tt_transformers.tt.model_config import parse_optimizations


# These inputs override the default inputs used by simple_text_demo.py. Check the main demo to see the default values.
def pytest_addoption(parser):
    parser.addoption("--input_prompts", action="store", help="input prompts json file")

    # Options for long_context_demo.py (moved here with that file).
    parser.addoption(
        "--page_block_size",
        action="store",
        type=int,
        default=None,
        help="KV-cache tokens per block for long_context_demo.py. Default 256, the value measured "
        "fastest for long prompts on Qwen3-8B / N300. A block is the allocation unit, so smaller "
        "values waste less memory on many short conversations. Must be a multiple of 32 that "
        "divides the row's max_seq_len.",
    )
    parser.addoption(
        "--accuracy",
        action="store_true",
        default=False,
        help="Run long_context_demo as a token-accuracy test instead of a performance test. "
        "Teacher-forces the model through a precomputed full-precision reference and reports "
        "top-1/top-5 agreement instead of tokens per second. Requires a reference sized for the "
        "context under test (see --accuracy_ref); disables Metal trace, which teacher forcing "
        "is incompatible with.",
    )
    parser.addoption(
        "--accuracy_ref",
        action="store",
        default=None,
        help="Path to the .refpt reference for --accuracy. Defaults to "
        "models/tt_transformers/tests/reference_outputs/<model>_<ctx label>.refpt, e.g. "
        "Qwen3-8B_32k.refpt, so each context length gets its own reference.",
    )
    parser.addoption(
        "--tracy_decode",
        action="store_true",
        default=False,
        help="Configure the run for Tracy profiling of the decode phase: disables Metal "
        "trace and caps generation at 2 tokens. Both are required -- trace replay under "
        "the profiler raises 'Device data mismatch', and a long decode loop overflows the "
        "device marker buffer. Slice the report to decode with `tt-perf-report "
        "--start-signpost DECODE_START`. Not for measurement: the decode average degrades to "
        "one sample.",
    )
    parser.addoption("--instruct", action="store", type=int, help="Use instruct weights")
    parser.addoption("--repeat_batches", action="store", type=int, help="Number of consecutive batches of users to run")
    parser.addoption("--max_seq_len", action="store", type=int, help="Maximum context length supported by the model")
    parser.addoption("--batch_size", action="store", type=int, help="Number of users in a batch ")
    parser.addoption(
        "--max_generated_tokens", action="store", type=int, help="Maximum number of tokens to generate for each user"
    )
    parser.addoption("--data_parallel", action="store", type=int, help="Number of data parallel workers")
    parser.addoption(
        "--paged_attention", action="store", type=bool, help="Whether to use paged attention or default attention"
    )
    parser.addoption("--page_params", action="store", type=dict, help="Page parameters for paged attention")
    # type=dict cannot parse a command-line string, so the option was write-only until now;
    # accept a JSON object, e.g. --sampling_params '{"temperature": 1.0, "top_k": 1, "top_p": 0.5}'
    parser.addoption("--sampling_params", action="store", type=json.loads, help="Sampling parameters for decoding")
    parser.addoption(
        "--stop_at_eos", action="store", type=int, help="Whether to stop decoding when the model generates an EoS token"
    )
    parser.addoption(
        "--optimizations",
        action="store",
        default=None,
        type=parse_optimizations,
        help="Precision and fidelity configuration diffs over default (i.e., accuracy)",
    )
    parser.addoption(
        "--decoder_config_file",
        action="store",
        default=None,
        type=str,
        help="Provide a JSON file defining per-decoder precision and fidelity settings",
    )
    parser.addoption(
        "--token_accuracy",
        action="store",
        default=False,
        type=bool,
        help="Whether to compute top1 and top5 exact token matching accuracy",
    )
    parser.addoption(
        "--stress_test",
        action="store",
        default=False,
        type=bool,
        help="Run stress test (same decode iteration over a large number of iterations",
    )
    parser.addoption("--enable_trace", action="store_true", default=None, help="Enable tracing")
    parser.addoption("--disable_trace", action="store_false", dest="enable_trace", default=None, help="Disable tracing")
    parser.addoption(
        "--num_layers",
        action="store",
        default=None,
        type=int,
        help="Number of layers to use",
    )
    parser.addoption(
        "--mode",
        action="store",
        default="full",
        type=str,
        help="Mode to use for full model demo tests (values can be 'prefill','decode','full')",
    )
    parser.addoption(
        "--use_prefetcher",
        action="store",
        default=False,
        type=bool,
        help="Whether to use DRAM prefetcher for prefetching weights into L1 during decode (only available on BH)",
    )
    parser.addoption(
        "--use_hf_rope",
        action="store_true",
        default=False,
        help="Whether to use HF-style rope, if not passed, the default mllama will be used",
    )
    parser.addoption(
        "--skip_perf_report",
        action="store_true",
        default=False,
        help=(
            "Skip writing the perf benchmark JSON and the CI perf-target check for this run. "
            "Use when the same test is run in more than one configuration and only one of them "
            "should report/validate perf (e.g. Llama-8B runs ci-eval-32 both without the prefetcher "
            "for repeat-batch coverage and with the prefetcher on a single batch for perf; only the "
            "latter should report perf). See issue #47820."
        ),
    )
