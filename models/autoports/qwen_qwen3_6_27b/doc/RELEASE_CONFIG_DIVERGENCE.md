# The release flow will test this model in a materially different configuration

Checked 2026-08-17 against `tt-inference-server` branch
`vvukoman/add-8-models-to-release-flow` (tip `60f80c4b`, one commit ahead of `main`)
and its `workflows/model_specs/dev/llm.yaml` entry for `Qwen/Qwen3.6-27B`.

That branch adds five *new* models (`google/gemma-4-26B-A4B-it`, `Qwen/Qwen3.8-27B`,
`Qwen/Qwen3.6-35B-A3B`, `meta-models/Muse-Glimmer-30B`,
`google/diffusiongemma-26B-A4B-it`). **Qwen3.6-27B is already in both the dev and prod
specs**, so what matters here is its existing release entry, not the diff.

## How `vllm_args` reach the server

`workflows/model_spec.py:420-430` merges the spec's `vllm_args` over defaults and
passes them as server flags:

```python
default_vllm_args = {
    "block_size": "64",
    "max_model_len": str(self.max_context),
    "max_num_seqs": str(max_concurrency),
    "max_num_batched_tokens": str(self.max_context),
    "max-log-len": "32",
    "seed": "9472",
    "additional_config": json.dumps({"tt": self.override_tt_config}),
}
merged_vllm_args = {**default_vllm_args, **self.vllm_args}
```

So everything in the spec's `vllm_args` block becomes a real CLI flag.

## The divergences

| setting | stage-11 run (and my re-runs) | release spec (P300X2) |
|---|---|---|
| `impl` | `qwen36_autoport` | **`qwen36_blackhole`** |
| `reasoning_parser` | **absent** | **`qwen3`** |
| `tool_call_parser` | absent | `qwen3_coder` |
| `enable_auto_tool_choice` | absent | `true` |
| `max_num_seqs` / `max_concurrency` | **1** | **32** |
| `sample_on_device_mode` | unset (model default) | **`decode_only`** |
| `fabric_config` | `FABRIC_1D_RING` | **`FABRIC_1D`** |
| `trace_region_size` | 200,000,000 | **1,073,741,824** |
| `l1_small_size` | unset | **24576** |
| `max_tokens_all_users_override` | unset | **525312** |
| `seed` | 42 (my re-runs) | **9472** |
| env vars | `{}` | `TT_QWEN35_TEXT_VER=qwen36_blackhole`, `MESH_DEVICE="(1, 4)"`, `TT_MESH_GRAPH_DESC_PATH=...p300_x2_mesh_graph_descriptor.textproto`, `QWEN36_MAX_TOKENS_ALL_USERS=525312` |

The stage-11 spec is unambiguous about the parser: its `device_model_spec.vllm_args`
is exactly

```json
{"model": "Qwen/Qwen3.6-27B", "block_size": "64", "max_model_len": "262144",
 "max_num_seqs": "1", "max_num_batched_tokens": "262144"}
```

with `reasoning_parser_name: "qwen3"` appearing only under `metadata`, which is
informational and never becomes a flag. My own server startup dump confirms the
consequence: `reasoning_parser=''`.

## Why the reasoning parser is the important one

Without a reasoning parser, vLLM returns the **entire** `<think>` block in
`choices[0].message.content`. lm-eval grades `content`. So every recorded number on
this branch was computed over reasoning text with the answer, if any, at the end.

With `reasoning_parser: qwen3`, vLLM splits the response: the think block goes to
`reasoning_content` and **`content` holds only what follows `</think>`**. That changes
grading directly:

- **GPQA** (`exact_match,flexible-extract`, via the patched `boxed_choice` filter):
  the filter would see a short clean answer rather than a long chain. It finds
  `\boxed{}` either way, so the *extraction* is not the issue — but a truncated
  response now yields **empty** content rather than a partial chain, which is a
  cleaner failure and a different one.
- **IFEval** (mean of the four prompt/inst x strict/loose keys): its instruction
  checks inspect response shape — "respond in all lowercase", "wrap in quotes",
  "include keyword X". Grading those against a reasoning chain instead of the answer
  is close to meaningless. The recorded 15/43 instruction-level score was measured
  that way. **With the parser, this number could move substantially, in either
  direction, for reasons that have nothing to do with the port.**

So the IFEval and GPQA figures recorded on this branch are not the numbers the
release flow will produce, and the gap is a configuration difference rather than a
model change.

## The second important one: the release serves at batch 32

`max_concurrency: 32` means the release server runs `--max_num_seqs 32`. Per
`SERVING_BATCH_LATENCY.md`, decode cost on this port follows the **allocated** batch,
not the active rows: one active request on a 32-slot server costs ~270 ms/token
against ~56 ms at `max_num_seqs=1`.

Consequently the headline single-user figures on this branch — `TPOT 61.893 ms`,
`ITL P50/P99 55.840/56.850 ms`, decode `16.157 t/s/u`, all measured at
`max_num_seqs=1` — **are not what the release configuration delivers**. At
`max_num_seqs=32` a single user should see roughly `3.7 t/s/u`. That is worth
resolving before any single-user latency claim is attached to the release.

## The third: the release runs a different code tree

`impl: qwen36_blackhole` with `TT_QWEN35_TEXT_VER: qwen36_blackhole` points at
`models/demos/blackhole/qwen36`. That directory **exists on tt-metal `origin/main`**
(`f3cfc53ef81`) but **not at this branch's base pin**, and this branch's work lives at
`models/autoports/qwen_qwen3_6_27b` under `impl_id: qwen36_autoport`.

So findings on this branch transfer to the release only insofar as the two trees share
code. Anything specific to the autoport — the precision policy in
`doc/datatype_sweep/selected_precision_config.json`, `LINEAR_PREFILL_CHUNK_SIZE` and
the new `QWEN36_LINEAR_PREFILL_CHUNK_SIZE` hook, the prefill scan implementation whose
op mix is analysed in `PREFILL_CHUNK_LEVER.md` — must be re-checked against
`models/demos/blackhole/qwen36` before being claimed of the release. I have not done
that comparison; the demo tree is not present at this pin.

This is the single most important caveat on everything else in this directory.

## What to test locally, in order

1. **Reasoning parser, everything else unchanged.** Add `--reasoning_parser qwen3` to
   the known-good server (`FABRIC_1D_RING`, 200 MB trace, `max_num_seqs 1`) and re-run
   the Diamond probe plus IFEval. Isolates the grading effect of the parser from every
   other difference. Lowest risk: server-side only, no device-config change.
2. **Release `override_tt_config`.** Switch to `FABRIC_1D`, `trace_region_size`
   1 GB, `l1_small_size` 24576, `sample_on_device_mode decode_only`. Each of these can
   plausibly fail on the autoport: the port's own multichip documentation justifies
   `FABRIC_1D_RING`, and `sample_on_device_mode` interacts with the TP4 vocabulary
   shard — the sibling gemma-4 entry in the same spec carries the comment "Required on
   the TP mesh so token ids >= 65536 are reachable (host sampling only sees device 0's
   vocab shard)", and this model's EOS ids are **248,044 and 248,046**, far above that
   threshold.
3. **`max_num_seqs 32`.** Confirms the batch-32 latency penalty in the release
   configuration and, incidentally, makes long reasoning evals faster in wall clock by
   overlapping documents.
