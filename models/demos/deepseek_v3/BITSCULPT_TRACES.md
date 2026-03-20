# DeepSeek BitSculpt Trace Repro

This note explains how to reproduce the TT DeepSeek BitSculpt trace work that was done against `DeepSeek-R1-0528` on multi-host Galaxy (`CLUSTER=B5`), how to compare the resulting tensors against the reference `aho_sus` trace in `bit_sculpt`, and which branches/files contain the relevant code.

The intended audience is someone taking this work over without the original context.

## Scope

This document covers three related runs:

1. `yieldthought/ds-tensor-capture`
   - baseline TT trace capture plumbing
   - BFP8-ish cache / original fidelity
2. `yieldthought/ds-lofi-trace`
   - same weights as baseline
   - LoFi fidelity everywhere
3. `yieldthought/ds-bfp4-cache`
   - mostly-BFP4 weight cache
   - LoFi fidelity everywhere
   - FP32 accumulation disabled everywhere in the model path

The final trace artifact from this work is:

- `bit_sculpt/results/deepseek-r1-0528/debug_trace/tt_moconnor_bfp4_lofi`

The reference trace used for comparison is:

- `bit_sculpt/results/deepseek-r1-0528/debug_trace/aho_sus`

## Repo / Branch Map

Two sibling repos were used:

- `tt-metal`
- `bit_sculpt`

The relevant branches are:

| Repo | Branch | Purpose |
| --- | --- | --- |
| `tt-metal` | `yieldthought/ds-tensor-capture` | Add TT-side BitSculpt-compatible trace capture |
| `bit_sculpt` | `yieldthought/ds-tensor-capture` | Add comparison script + initial TT artifacts |
| `tt-metal` | `yieldthought/ds-lofi-trace` | LoFi-only runtime experiment |
| `bit_sculpt` | `yieldthought/ds-lofi-trace` | LoFi-only trace artifacts |
| `tt-metal` | `yieldthought/ds-bfp4-cache` | Final BFP4 + LoFi + no-FP32 runtime |
| `bit_sculpt` | `yieldthought/ds-bfp4-cache` | Final BFP4 + LoFi trace artifacts |

At the time this note was written:

- `tt-metal` `yieldthought/ds-bfp4-cache` included commit `12b86924760e3bf47c360d315d0946195846a04f`
- `bit_sculpt` `yieldthought/ds-bfp4-cache` included commit `f0bac2f8`

## What Changed In `tt-metal`

The BitSculpt trace plumbing lives in these files:

- `models/demos/deepseek_v3/tests/test_bitsculpt_trace.py`
  - manual multihost harness that drives a single prefill and saves BitSculpt-format outputs
- `models/demos/deepseek_v3/utils/bitsculpt_trace.py`
  - collector that writes `metadata.json`, `expert_routing.safetensors`, `hidden_states.safetensors`, `kv_cache.safetensors`
- `models/demos/deepseek_v3/conftest.py`
  - adds `--bitsculpt-trace*` CLI flags
- `models/demos/deepseek_v3/tt/mla/mla1d.py`
  - captures prefill KV at the correct semantic point
  - also contains some inline attention compute-kernel settings
- `models/demos/deepseek_v3/tt/moe.py`
  - exposes routed expert IDs / weights to the trace collector
- `models/demos/deepseek_v3/tt/decoder_block/decoder_block_2d_base.py`
  - hooks full-layer output and post-attention states
- `models/demos/deepseek_v3/tt/embedding/embedding1d.py`
  - lets the trace harness keep padded prefill outputs

The BFP4 cache work lives in these files:

- `models/demos/deepseek_v3/tt/experts.py`
  - expert weights switched to BFP4
- `models/demos/deepseek_v3/tt/mlp/mlp_dequant.py`
  - dense/shared MLP weights switched to BFP4
- `models/demos/deepseek_v3/tt/lm_head.py`
  - LM head weights switched to BFP4
- `models/demos/deepseek_v3/tt/lm_head1d.py`
  - LM head weights switched to BFP4

The final BFP4 + LoFi runtime settings live in:

- `models/demos/deepseek_v3/utils/config_helpers.py`
- `models/demos/deepseek_v3/tt/mla/mla1d.py`

Those two files are the ones that make the final branch:

- LoFi everywhere
- SDPA LoFi
- gate-MM LoFi
- FP32 accumulation disabled everywhere in the model path

## What Changed In `bit_sculpt`

The comparison tooling lives in:

- `scripts/compare_debug_trace.py`

Important behavior of the current comparison script:

- outputs a whitespace-padded markdown table for terminal viewing
- orders columns by model execution order
- defaults to order-invariant routing comparison
- includes both:
  - `expert_weights`
    - full expert-space, aligned by expert id
  - `expert_weights_shared_ids`
    - weights compared only on experts selected by both sides
- supports `--tokens` selectors such as:
  - `--tokens 0`
  - `--tokens 2-5`
  - `--tokens 2:5`
  - `--tokens 0,3,5-6`

## Environment And Layout

The concrete layout used here was:

```text
/data/moconnor/tt-metal
/data/moconnor/bit_sculpt
/data/deepseek/DeepSeek-R1-0528
/data/deepseek/DeepSeek-R1-0528-Cache-BFP8
/data/deepseek/DeepSeek-R1-0528-Cache-BFP4
```

The runs were performed on:

- `CLUSTER=B5`
- hosts:
  - `UF-MN-B5-GWH01`
  - `UF-MN-B5-GWH02`

Key environment variables:

```bash
export CLUSTER=B5
export TT_METAL_HOME=/data/<user>/tt-metal
export DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528
```

For baseline / LoFi-only runs:

```bash
export DEEPSEEK_V3_CACHE=/data/deepseek/DeepSeek-R1-0528-Cache-BFP8
```

For the BFP4 + LoFi run:

```bash
export DEEPSEEK_V3_CACHE=/data/deepseek/DeepSeek-R1-0528-Cache-BFP4
```

## Prompt Convention

The BitSculpt trace harness sends the prompt to the tokenizer exactly as-is.

Important:

- no Hugging Face chat template
- no instruct wrapper
- no `<|User|>` / `<|Assistant|>` wrapper added by the trace harness

The default trace prompt is:

```text
why is aho so sus
```

That default is defined in:

- `models/demos/deepseek_v3/conftest.py`

## Preferred Launch Path

If your environment is healthy, prefer:

```bash
CLUSTER=B5 ds-run ...
```

That is what was originally intended.

In the `/data/moconnor/tt-metal` worktree used during this effort, `ds-run` could not find `tt-run` from `PATH`, so direct invocation of `ttnn/distributed/ttrun.py` was used as a fallback. See the troubleshooting section for that exact wrapper.

## Repro 1: Baseline TT Trace Capture

Checkout:

```bash
cd /data/<user>/tt-metal
git checkout yieldthought/ds-tensor-capture

cd /data/<user>/bit_sculpt
git checkout yieldthought/ds-tensor-capture
```

Run the trace:

```bash
cd /data/<user>/tt-metal

CLUSTER=B5 \
DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528 \
DEEPSEEK_V3_CACHE=/data/deepseek/DeepSeek-R1-0528-Cache-BFP8 \
ds-run pytest -q models/demos/deepseek_v3/tests/test_bitsculpt_trace.py \
  --bitsculpt-trace \
  --bitsculpt-trace-run-tag tt_<user> \
  --bitsculpt-trace-output-dir /data/<user>/bit_sculpt/results/deepseek-r1-0528/debug_trace
```

Expected output:

```text
/data/<user>/bit_sculpt/results/deepseek-r1-0528/debug_trace/tt_<user>/
  metadata.json
  expert_routing.safetensors
  hidden_states.safetensors
  kv_cache.safetensors
```

Compare against `aho_sus`:

```bash
cd /data/<user>/bit_sculpt

python scripts/compare_debug_trace.py \
  results/deepseek-r1-0528/debug_trace/aho_sus \
  results/deepseek-r1-0528/debug_trace/tt_<user> \
  > results/deepseek-r1-0528/debug_trace/tt_<user>/COMPARISON.md
```

## Repro 2: LoFi-Only Trace

Checkout:

```bash
cd /data/<user>/tt-metal
git checkout yieldthought/ds-lofi-trace

cd /data/<user>/bit_sculpt
git checkout yieldthought/ds-lofi-trace
```

This branch reuses the existing BFP8 cache. Do not make a new cache directory for the LoFi-only run.

Run:

```bash
cd /data/<user>/tt-metal

CLUSTER=B5 \
DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528 \
DEEPSEEK_V3_CACHE=/data/deepseek/DeepSeek-R1-0528-Cache-BFP8 \
ds-run pytest -q models/demos/deepseek_v3/tests/test_bitsculpt_trace.py \
  --bitsculpt-trace \
  --bitsculpt-trace-run-tag tt_<user>_lofi \
  --bitsculpt-trace-output-dir /data/<user>/bit_sculpt/results/deepseek-r1-0528/debug_trace
```

Compare:

```bash
cd /data/<user>/bit_sculpt

python scripts/compare_debug_trace.py \
  results/deepseek-r1-0528/debug_trace/aho_sus \
  results/deepseek-r1-0528/debug_trace/tt_<user>_lofi \
  > results/deepseek-r1-0528/debug_trace/tt_<user>_lofi/COMPARISON.md
```

## Repro 3: BFP4 + LoFi + No-FP32 Trace

Checkout:

```bash
cd /data/<user>/tt-metal
git checkout yieldthought/ds-bfp4-cache

cd /data/<user>/bit_sculpt
git checkout yieldthought/ds-bfp4-cache
```

This branch expects:

- expert weights BFP4
- dense/shared MLP weights BFP4
- gate MM weights BF16
- embeddings BF16
- LM head weights BFP4
- all other MM weights BFP8
- RMSNorm weights BF16
- LoFi everywhere
- FP32 accumulation disabled everywhere in the model path

### Cold-cache run

Use a fresh cache directory:

```bash
export DEEPSEEK_V3_CACHE=/data/deepseek/DeepSeek-R1-0528-Cache-BFP4
```

Then run the same trace harness:

```bash
cd /data/<user>/tt-metal

CLUSTER=B5 \
DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528 \
DEEPSEEK_V3_CACHE=/data/deepseek/DeepSeek-R1-0528-Cache-BFP4 \
ds-run pytest -q models/demos/deepseek_v3/tests/test_bitsculpt_trace.py \
  --bitsculpt-trace \
  --bitsculpt-trace-run-tag tt_<user>_bfp4_lofi \
  --bitsculpt-trace-output-dir /data/<user>/bit_sculpt/results/deepseek-r1-0528/debug_trace
```

Observed behavior on B5:

- cold cache generation was slow but finite
- a conservative timeout of 4 hours was used
- the actual MoE conversion pace was about 74 seconds per layer on this machine
- the first cold-cache run may end in a stale multihost cache-visibility wait after the conversion is already complete

### Warm-cache rerun

Once `/data/deepseek/DeepSeek-R1-0528-Cache-BFP4` is populated on both B5 hosts, rerun the same command. The warm-cache rerun is the one that reliably produced the final trace artifact.

That is how `tt_moconnor_bfp4_lofi` was generated.

Compare:

```bash
cd /data/<user>/bit_sculpt

python scripts/compare_debug_trace.py \
  results/deepseek-r1-0528/debug_trace/aho_sus \
  results/deepseek-r1-0528/debug_trace/tt_<user>_bfp4_lofi \
  > results/deepseek-r1-0528/debug_trace/tt_<user>_bfp4_lofi/COMPARISON.md
```

## Comparing Only A Token Or Token Range

Examples:

```bash
python scripts/compare_debug_trace.py --tokens 0 \
  results/deepseek-r1-0528/debug_trace/aho_sus \
  results/deepseek-r1-0528/debug_trace/tt_<user>_bfp4_lofi

python scripts/compare_debug_trace.py --tokens 2-4 \
  results/deepseek-r1-0528/debug_trace/aho_sus \
  results/deepseek-r1-0528/debug_trace/tt_<user>_bfp4_lofi

python scripts/compare_debug_trace.py --tokens 2:5 \
  results/deepseek-r1-0528/debug_trace/aho_sus \
  results/deepseek-r1-0528/debug_trace/tt_<user>_bfp4_lofi
```

## Important Known Behaviors

### 1. KV capture must be pre-norm / pre-RoPE

The correct BitSculpt `compressed_kv_layer_*` tensor is the raw fused KV projection output before:

- KV RMSNorm
- RoPE on the positional slice

If you capture after those transforms, `compressed_kv` PCC collapses even when the rest of the residual stream matches well.

### 2. Expert order is arbitrary

Raw `expert_ids` PCC is misleading when both sides select the same experts in a different order.

The current `compare_debug_trace.py` defaults to order-invariant routing comparison and adds:

- `expert_weights_shared_ids`
  - only compare weights for experts selected by both traces

### 3. `run_tag` is not currently written by the collector

`BitSculptTraceCollector.save()` writes:

- prompt
- token ids / strings
- timestamp
- username
- hostname
- command line
- git branch / commit

but not `run_tag`.

For the saved `tt_moconnor_bfp4_lofi` artifact, `metadata.json` was patched manually afterward to add:

```json
"run_tag": "tt_moconnor_bfp4_lofi"
```

If you care about `run_tag` in metadata, either:

- patch the saved `metadata.json` after the run
- or add `run_tag` to `BitSculptTraceCollector.save()`

## Optional: Run The 128-Token Demo To Inspect Outputs

This is not needed for BitSculpt traces, but it is useful to see qualitative model output under the same settings.

Example:

```bash
cd /data/<user>/tt-metal

CLUSTER=B5 \
DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528 \
DEEPSEEK_V3_CACHE=/data/deepseek/DeepSeek-R1-0528-Cache-BFP4 \
ds-run python models/demos/deepseek_v3/demo/demo.py \
  --model-path /data/deepseek/DeepSeek-R1-0528 \
  --cache-dir /data/deepseek/DeepSeek-R1-0528-Cache-BFP4 \
  --prompts-file models/demos/deepseek_v3/demo/test_prompts.json \
  --num-prompts 1 \
  --output-path /tmp/deepseek_bfp4_lofi_demo_128.json \
  --max-new-tokens 128 \
  --no-stop-at-eos
```

The run done during this work used the first prompt in `test_prompts.json`:

```text
When did World War II officially end?
```

and wrote:

- `/tmp/deepseek_bfp4_lofi_demo_128.json`
- `/tmp/deepseek_bfp4_lofi_demo_128.log`

## Troubleshooting

### `ds-run` cannot find `tt-run`

In the `/data/moconnor/tt-metal` worktree used during this project, `ds-run` was not sufficient by itself because `tt-run` was missing from `PATH`.

The fallback that worked was:

1. source `/data/deepseek/scripts/cluster-config.sh`
2. call `ttnn/ttnn/distributed/ttrun.py` directly
3. export `PYTHONPATH` explicitly inside the rank command

If you hit import problems in this source-tree mode, the extra `PYTHONPATH` used here was:

```bash
$TT_METAL_HOME/.codex_pydeps:$TT_METAL_HOME/tools:$TT_METAL_HOME/ttnn:$TT_METAL_HOME
```

`graphviz` was installed into `$TT_METAL_HOME/.codex_pydeps` because it was missing from the active environment.

### Cold BFP4 cache run hangs after conversion

Observed symptom:

- the cold run prints `Done converting weights to TTNN SavedWeight format`
- then one host keeps logging `_maybe_log_wait(...)` messages about a tensorbin not becoming visible

What to do:

1. Verify the missing path on both hosts with `ssh`
2. If the file really exists on both hosts, kill the stale run
3. Rerun warm-cache with the same command

That warm rerun succeeded for `tt_moconnor_bfp4_lofi`.

### Devices or ranks get stuck in init

Use:

```bash
CLUSTER=B5 tt-reset
```

if device state looks wedged.

### Need to monitor a long run

Useful tools:

- `py-spy`
- `mpitop`

Practical checks:

- `ps -eo pid,etimes,%cpu,%mem,cmd | rg 'test_bitsculpt_trace.py|demo.py|ttrun.py'`
- `find /data/deepseek/DeepSeek-R1-0528-Cache-BFP4 -type f | wc -l`
- `du -sh /data/deepseek/DeepSeek-R1-0528-Cache-BFP4`

## Sanity Checklist

Before handing results to someone else, verify:

- `metadata.json` exists
- `expert_routing.safetensors` exists
- `hidden_states.safetensors` exists
- `kv_cache.safetensors` exists
- `COMPARISON.md` exists
- `metadata.json` records:
  - prompt
  - token ids
  - token strings
  - username
  - hostname
  - command line
  - git branch
  - git commit

## Suggested Repro Order

If someone is doing this from scratch, the shortest path is:

1. reproduce the baseline trace plumbing on `yieldthought/ds-tensor-capture`
2. verify comparison against `aho_sus`
3. reproduce LoFi-only on `yieldthought/ds-lofi-trace`
4. build the BFP4 cache on `yieldthought/ds-bfp4-cache`
5. rerun warm-cache and save `tt_<user>_bfp4_lofi`
6. generate `COMPARISON.md`

If time is limited and only the final result matters, skip directly to:

- `yieldthought/ds-bfp4-cache`
- `/data/deepseek/DeepSeek-R1-0528-Cache-BFP4`
- `tt_<user>_bfp4_lofi`
