# Re-running the vLLM full model on the rebased tree

Goal: serve the full model under vLLM on `mvasiljevic/qwen38-deltanet-kda` and
measure TSU at batch 32, ISL 128, OSL 128.

**Status: the model serves; the batch-32 measurement did not complete.** What
blocked it is a property of the implementation, recorded below with what it
cost, so the next attempt can start from the right configuration.

## Environment built from scratch

Nothing was in place. Everything below was installed or downloaded for this run:

| Piece | Detail |
|---|---|
| Checkpoint | `Qwen/Qwen3.8-27B`, 52 GB, 18 safetensors, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` — the revision `doc/qwen38_checkpoint_swap` records for 3.8. The HF cache held only a 32 KB `config.json`. |
| vLLM | `0.26.0+empty`, built from source via `vllm-tt-plugin/docs/install-vllm-tt.sh` |
| Plugin | `vllm-tt-plugin` `main` (standalone repo, cloned to `~/vllm-tt-plugin`) |
| Tooling | `tt-smi` 6.3.0, `tt-perf-report` 1.2.9, `py-spy` 0.4.2 |

The install left `transformers 5.12.1` and `torch 2.11.0+cpu` untouched — only
five minor packages moved (`lark`, `opencv-python-headless`, `pydantic`,
`pydantic-core`, `sse-starlette`) and nothing was removed. Verified before and
after with `uv pip freeze`.

## Three rebase gaps had to be fixed first

All three are the A1 failure mode in `doc/rebase_to_new_base`: `-X ours` kept
the base's version of the region holding a *definition* while the old branch's
*call site* applied without conflicting, so nothing surfaced as a conflict.

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: Can't instantiate abstract class Qwen36Generator without ... 'prefill_logits'` | The base's readiness contract added an abstract `prefill_logits`; the old branch's `contract.py` has zero occurrences of it | Implemented on the existing `prefill_forward` with `return_all_logits` |
| `AttributeError: 'TTSampling' object has no attribute '_plan_local_topk_chunks'` — server died at startup | Call site at `tt_sampling.py:227` and its test survived; the definition did not | Restored from the old branch |
| `NameError: name '_log_sampling_debug' is not defined` — every request 500'd | One orphaned call; the helper, `_compact_debug_list` and `_sampling_debug_enabled` are all absent, the base dropped that debug facility wholesale | Dropped the call, matching how the dispatch kernels were resolved |

Plus `test_plugin_registration_targets_autoport`, which read a patched
`platform.py` out of a sibling vLLM checkout. The plugin is a standalone
installed package now and the registration lives in
`models/autoports/vllm_bundles`; the test asserts against the bundle instead.

After these, all 17 contract tests pass and the server reaches
`Application startup complete`.

## The environment contract that actually works

`doc/qwen38_checkpoint_swap/tt-inference-server-onboard-qwen38-autoport.patch`
is the source of truth; missing any of it fails at load:

```bash
export QWEN_AUTOPORT_MODEL_ID=Qwen/Qwen3.8-27B
export QWEN_AUTOPORT_MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0  # default is 3.6's
export HF_MODEL=Qwen/Qwen3.8-27B
export QWEN36_MAX_TOKENS_ALL_USERS=525312
export EXTRA_MODELS_DIR=$PWD/models/autoports/vllm_bundles                     # serves the autoport, not the demo
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto

python -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --hf-model Qwen/Qwen3.8-27B --mesh-device P300x2 \
  --max-num-seqs 32 --max-model-len 262144 --sampling-profile full \
  --benchmark-prompt-len 128 --benchmark-output-len 128 \
  --benchmark-num-requests 32 --no-benchmark-ci-serving \
  --tt-config '{"trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}'
```

Without `QWEN_AUTOPORT_MODEL_REVISION` the loader looks for 3.6's revision and
fails with `No local snapshot ... Tried hf_config._name_or_path=...`. The mesh
label is `P300x2`, lowercase x.

## What was measured, and what was not

**Serving works.** The model loads in ~3 minutes and answers requests.

**Concurrency 1, ISL/OSL 128/128: 138.5 s per request**, steady across the five
requests that completed before the run was stopped
(`4/32 [09:16<1:04:38, 138.53s/it]`, `5/32 [11:34<1:02:14, 138.31s/it]`).

**Concurrency 32 did not complete.** Two attempts, 100 and 104 minutes, neither
finishing a single wave. Not hung — `py-spy` showed it inside
`_linear_attention_prefill_chunk` throughout, and EngineCore accumulated CPU
time 1:1 with wall clock (`01:26:35 -> 01:36:27` over one 10-minute window).

## Why: prefill runs at the full slot width, once per request

At concurrency 1 a request costs 138.5 s while its decode is only 128 tokens.
At the recorded batch-1 rate of 17.9 tok/s/user that decode is ~7 s, so **~130 s
of every request is prefill**.

That is the fixed-slot cost: `prefill_forward` runs at the generator's full
`batch` width — 32 with `--max-num-seqs 32` — no matter how many rows are
actually active. And `platform.py` disables chunked prefill for `model_type=qwen3_5`
(`Chunked prefill is not validated for model_type=qwen3_5; disabling it`), so
vLLM prefills **one request per scheduler step**. Thirty-two requests therefore
pay thirty-two full-width prefills, ~69 minutes before the first token of the
last one.

This is consistent with the single-chip measurement in `doc/kda_conv_swap`: a
128-token linear-attention prefill is 272 ms per layer at batch 1, and the
affine scan scales with `batch * value_heads`, so batch-32 slots cost ~32x that
across 48 linear layers.

It also explains the shape of the recorded evidence: the headline
`vllm_benchmark.json` is **concurrency 1**, and `vllm_ci_serving_benchmark.json`
is 32 requests at 100/100 in 187 s — a rate that is not reachable if each of
those requests pays a full-width prefill, so that run cannot have been paying
one. Reconciling the 187 s recorded against ~130 s per prefill measured here is
open, and is the first thing to check before trusting either number.

**Do not pass `--benchmark-concurrency` expecting it to help.** Omitting it
makes the harness default to concurrency 1, which completes; setting it to 32
serialises 32 full-width prefills.

## Open

1. Reconcile the recorded 187 s / 32-request CI-serving figure with the ~130 s
   per-request prefill measured here. One of them is not what it looks like.
2. Make prefill cost track active rows rather than slot count. That is the
   difference between a batch-32 serving benchmark taking ~70 minutes and taking
   ~2, and it is worth more than any decode-side win measured so far.
3. Then re-run this configuration for the TSU number.
