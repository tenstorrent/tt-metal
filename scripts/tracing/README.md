# Benchmark

This directory contains a benchmark for measuring whether traced prefill helps TT transformer inference when automatic prefix caching (APC) is enabled.

The benchmark was added alongside PR [#43165](https://github.com/tenstorrent/tt-metal/pull/43165), which enables traced prefill for APC in TT transformers. The point of this script is to compare `trace_mode` settings on the same prompts and confirm that traced prefill reduces time-to-first-token (TTFT), prefill latency, and end-to-end latency as well.

## Purpose

Without a focused benchmark, it is hard to tell whether APC + traced prefill is actually helping or just changing the code path. This script keeps the experiment narrow:

- load the same `vllm.LLM` model with different `trace_mode` values
- vary shared-prefix size and cached-prefix ratio
- run the same prompt set for each configuration
- report TTFT, prefill, decode, and end-to-end latency summaries

This was used during PR [#43165](https://github.com/tenstorrent/tt-metal/pull/43165) to validate the traced-prefill work and inspect where `trace_mode="all"` helps most.

## Files

- `run.py`: benchmark driver
- `system-prompt.txt`: shared system prompt
- `context.txt`: shared context that can be scaled up or down
- `questions.txt`: prompt tails appended to the shared prefix

## Usage

Run from this directory:

```bash
python run.py --help
python run.py \
    -m Qwen/Qwen3-4B-Instruct-2507 \
    -p 20 -w 0 -v 1 \  # 20 prompts for each mode, cached ratio, and context size.
    -t decode_only all \  # Two tracing modes.
    -r 0.0 50.0 100.0 \  # Three ratios of cached tokens in shared prefix
    -c 0.5 1.0 1.5 2.0  # Three context sizes (half of the context, full context, ...)
```

The script builds prompts from the local text assets, sweeps this grid,

`trace_mode × prefix_caching_ratio × context_multiply`

and then prints aggregate and per-setting tables.

## Example Logs

You will see prefix statistics and model setup for each run:

```text
📚 Prefix token stats: full=2,392, cached=1,000, non-cached=1,392 (prefix_caching_ratio=42.0%).

🧠 Preparing LLM 'Qwen/Qwen3-4B-Instruct-2507' for inference with enabled prefix caching (42.0% fixed context),
   block size 64, 1 simultaneously processed sequences, and with trace mode 'all'.
```

Then the script prints summary tables similar to the ones discussed in PR [#43165](https://github.com/tenstorrent/tt-metal/pull/43165):

```text
🧮 Aggregate Cross-Mode Summary (lower is better)
+-------------+--------------------+-----------------------+-------------------+
| trace_mode  | median_ttft (sec.) | median_prefill (sec.) | median_e2e (sec.) |
+-------------+--------------------+-----------------------+-------------------+
| decode_only |              0.184 |                 0.182 |             2.722 |
| all         |             *0.179 |                *0.178 |            *2.706 |
+-------------+--------------------+-----------------------+-------------------+
* marks the lowest value per metric (ties allowed).
```

And a per-setting comparison that makes APC impact easier to inspect:

```text
📊 Per-Setting Prefill Comparison
+------------------+-...-+----------------------------+--------------------+-----------------+-...-+
| prefix_ratio (%) | ... | prefill_decode_only (sec.) | prefill_all (sec.) | winner          | ... |
+------------------+-...-+----------------------------+--------------------+-----------------+-...-+
|             50.0 | ... |                      0.266 |              0.256 |             all | ... |
|            100.0 | ... |                      0.064 |              0.061 |             all | ... |
+------------------+-...-+----------------------------+--------------------+-----------------+-...-+
```

One of the useful patterns from PR [#43165](https://github.com/tenstorrent/tt-metal/pull/43165) was that `trace_mode="all"` often shows the clearest benefit once some prefix is cached, while cold-cache runs can be close to ties.

## Notes

- This benchmark is intended for local performance investigation, not correctness validation.
- Results may depend on model support, machine type, cache hit ratio, and shared-prefix length.
- The original traced-prefill work is tracked in PR [#43165](https://github.com/tenstorrent/tt-metal/pull/43165), with background tied to vLLM issue `#268` for APC support.
