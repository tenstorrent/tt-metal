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

## What was measured

**Serving works.** The model loads in ~3 minutes and answers requests.

### Full model, non-vLLM, batch 1 (`tests/full_model_perf_warm.py`)

| | ms |
|---|---:|
| Cold TTFT, S=128 | 7517.2 |
| Warm TTFT, S=128 (median of 3) | **3612.3** (3611.8 / 3612.3 / 3612.6) |
| Cold overhead | 3904.9 |

So the model's own 128-token prefill is 3.6 s, and it is stable to a millisecond.
That is in line with the TTFT the stage 09/10 vLLM benchmarks recorded
(4139 ms and 3784 ms), so nothing about the model's prefill regressed.

### Stacked TP4 decoder, batch 32 (`tests/multichip_stacked_traced_decode.py`)

Trace replay median **5.032 ms** over six steps (5.029-5.041), replicated
residual. Batch 32 is healthy on the real TP4 decoder stack.

### Under vLLM

Concurrency 1, ISL/OSL 128/128: **138.5 s per request**, steady
(`4/32 [09:16<1:04:38, 138.53s/it]`, `5/32 [11:34<1:02:14, 138.31s/it]`).

Concurrency 32 did not complete: two attempts, 100 and 104 minutes, neither
finishing a wave. Not hung -- `py-spy` showed it inside
`_linear_attention_prefill_chunk` throughout, and EngineCore accumulated CPU
time 1:1 with wall clock.

## Why: prefill scales with the slot count, not the active rows

Measured directly on one linear-attention layer, S=128, single chip
(`linear_attention_synthetic_pcc.py --mode prefill --iterations 3`):

| slot count | prefill |
|---|---:|
| batch 1 | 274.16 ms |
| batch 32 | 8577.07 ms |

**31.3x for 32x the slots**, i.e. linear in the fixed slot count regardless of
how many rows carry a real prompt. The affine scan runs over
`batch * value_heads` groups, so a batch-32 generator pays 32 sequences' worth of
scan to prefill one.

That closes the chain:

| step | value |
|---|---:|
| full model, batch-1 slot width, warm | 3612 ms (measured) |
| x 31.3 slot-width factor | ~113 s (predicted) |
| observed under vLLM at `--max-num-seqs 32` | ~130 s/request (measured) |

And it explains the recorded evidence rather than contradicting it: a ~4 s TTFT
is what a **batch-1-slot** server pays, so the recorded stage 09/10 runs were not
paying the 32-slot cost. The discrepancy noted in the first version of this file
was mine, not theirs.

`platform.py` also disables chunked prefill for `model_type=qwen3_5`
(`Chunked prefill is not validated for model_type=qwen3_5; disabling it`), so
vLLM prefills one request per scheduler step. Thirty-two requests at
`--max-num-seqs 32` therefore pay thirty-two 32-slot prefills, ~70 minutes
before the last one's first token.

**Do not pass `--benchmark-concurrency` expecting it to help.** Omitting it makes
the harness default to concurrency 1, which completes; setting it to 32
serialises 32 full-width prefills.

## Open

1. **Make prefill cost track active rows rather than slot count.** This is the
   one that matters: it is the difference between a batch-32 serving benchmark
   taking ~70 minutes and taking ~2, and it is worth more than any decode-side
   win measured on this branch. The fused KDA conv cut the decode layer nearly
   in half; this is a 31x factor sitting in prefill.
2. Then re-run this configuration for the batch-32 TSU number. Until (1), a
   batch-32 TSU costs ~90 minutes of prefill to obtain.
