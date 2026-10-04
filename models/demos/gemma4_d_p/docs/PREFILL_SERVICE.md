# Gemma4 prefill service

The service supports **Gemma4-31B-it, Blackhole 8×4, 262144 tokens per slot, 8192-token chunks, and batch 1**, with up to six resident KV slots. Defaults are in [the model manifest](../tt/runners/manifest.json).

Run this setup from the repository root in both terminals:

```bash
source python_env/bin/activate
export PYTHONPATH=$PWD
export TT_METAL_HOME=$PWD
export \
       HF_MODEL=google/gemma-4-31B-it \
       HF_HOME=/mnt/models/huggingface \
       TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
       HF_HUB_OFFLINE=1
```

Start the service:

```bash
python -m models.demos.gemma4_d_p.tt.runners.prefill_runner
```

Prepare six 256K prompts, one different Gutenberg book per slot, then send them with the shared producer:

```bash
python -m models.demos.gemma4_d_p.tt.runners.prepare_prefill_inputs
python -m models.demos.common.prefill.runners.prefill_producer --manifest /tmp/gemma4_prefill_inputs/producer.json
```

The preparation helper downloads and caches the books under `/tmp/gemma4_prefill_text`, then writes per-slot `metadata.json` token files and a producer manifest under `/tmp/gemma4_prefill_inputs`. Use one `--text /path/to/book.txt` per slot for local text, `--slots` for fewer prompts, or `--tokens 8193` to exercise a partial final chunk.

The shared producer uses round-robin scheduling and leaves the service running by default. Set `PREFILL_SEND_SHUTDOWN=1` when launching it to stop the service after the requests. These inputs contain tokens only, so golden-KV verification is disabled. In this mode the shared producer does not wait for layer acknowledgments; the hardware test below checks them separately.

Set `PREFILL_NUM_USERS` to 1–6 for the runner and prepare the same number of prompts with `--slots`. `PREFILL_H2D_SERVICE_ID` selects the shared service name and must match in both terminals. The service uses `google/gemma-4-31B-it`. `PREFILL_TTNN_CACHE` overrides the `TT_CACHE_PATH` root. One of these cache variables must be set. The runner reuses `tensor_cache_bf16_mesh8x4` beneath that root; no cache path is derived from the HF model ID or `HF_HOME`. Starting at position zero replaces a slot's prompt; subsequent chunks must be contiguous.

The service captures one trace and stages tokens, slot metadata, and absolute RoPE positions before each replay. Device acknowledgments follow each layer's KV writes. The engine owns the caches and sockets. The populated caches remain resident until shutdown.

Run the end-to-end hardware check, which starts both the runner and producer:

```bash
pytest models/demos/gemma4_d_p/tests/test_prefill_service.py -sv --basetemp=/tmp/gemma4-service-test
```

It checks all six full-context prompts, final-chunk trace/eager PCC above 0.999, finite hidden states, nonzero first/last KV rows from every layer, distinct slot contents, and preservation of completed slots. It then reuses all six slots with 8193-token prompts to check partial final chunks. Producer logs, including timings, are saved in the pytest temporary directory. For the canonical demo, add `--timeout=3600` if loading weights exceeds the repository's default 300-second timeout.

For source KV correctness and real loopback migration, see [Migration tests](PREFILL_MIGRATION.md).

## Long-running prefill stress test

The [stress launcher](../scripts/launch.sh) adapts the monitor and outer loop from
[PR #58456](https://github.com/tenstorrent/tt-metal/pull/58456). From the repository root:

```bash
HF_MODEL=google/gemma-4-31B-it \
HF_HOME=/mnt/models/huggingface \
TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
HF_HUB_OFFLINE=1 \
PREFILL_TRACE_DIR=/mnt/models/huggingface/gpu_traces/gemma4_d_p/gutenberg-135 \
ITERS_ID=iters600 CHUNKS_ID=chunks20 TRACE_ID=traced RESET_BETWEEN_RUNS=0 \
models/demos/gemma4_d_p/scripts/launch.sh 5 gemma4_8k_no_reset
```

This starts a detached tmux session. Each of five runs loads all 60 layers, allocates six 256K KV slots,
captures the production runtime's forward once, then executes 600 requests of 20 × 8192 tokens (160K).
Requests rotate across slots; a reused slot starts at position zero and overwrites its previous prompt.
There are 60,000 measured chunk forwards in total. Weights, caches, and the trace remain allocated for
all 600 iterations of each run. The mesh is closed/reopened between runs, with no `tt-smi` reset.
A failed run stops the no-reset outer loop.

The test stages token tensors and request metadata directly into the traced runtime, without the
runner/producer sockets or migration acknowledgments. Each slot uses a different rotation of the
token stream from `metadata.json`; no GPU KV files are needed. It checks finite, nonconstant final
hidden states on one chip and bit-exact output repeatability when reusing a slot. These are soak
checks, not GPU-reference accuracy validation. Device time excludes host staging and output checks.

By default, artifacts go under `generated/gemma4_stress/<log_name>/` (`LOG_ROOT` overrides the root):

```bash
tail -F generated/gemma4_stress/gemma4_8k_no_reset/stress.log
tmux attach -r -t "gemma4_stress_${HOSTNAME}"
```

`log_01` … `log_05` contain per-run pytest output and `TEST_DONE_EXIT`; `timings_01.csv` …
`timings_05.csv` stream per-chunk device/staging times, and `result_01.json` … `result_05.json`
record completed forward loops. `host_stats.tsv` records host resource samples. The usual pytest
wall-time timeout is disabled for this long test; optional `TRIAGE=1` enables the dispatch watchdog.
For a shorter run that exercises every slot twice, use `ITERS_ID=iters12` and one outer run.

For the reset variant, keep the same model paths and select 100 runs of 20 iterations:

```bash
ITERS_ID=iters20 CHUNKS_ID=chunks20 TRACE_ID=traced RESET_BETWEEN_RUNS=1 \
models/demos/gemma4_d_p/scripts/launch.sh 100 gemma4_8k_reset
```

This executes `tt-smi -r` on all 32 Blackhole devices before every run, including the first,
then loads the model and captures a new trace. The original PR's `-glx_reset` command uses
the Wormhole Galaxy IPMI tray reset path. Each run executes 400 chunks; the complete workload is 40,000 chunks and
100 hardware resets. Reset output and exit status are saved in `reset_NN.log`; a failed reset
stops the loop before launching pytest. Set `TT_SMI=/absolute/path/to/tt-smi` to use a specific
reset-tool installation. A completed failed test is recorded and followed by the next reset/run.
