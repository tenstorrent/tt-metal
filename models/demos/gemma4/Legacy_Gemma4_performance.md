# Gemma4 31B — Speculative Decode & Server Work Summary

**Document date:** 2026-07-06 17:26:48 UTC
**Last refined:** 2026-07-09 — Prefetcher MLP A/B: **16.0 tok/s** (regresses vs interleaved 20.5 / DRAM_SHARDED ~21–22); keep off
**Platform:** QB2 (`MESH_DEVICE=P150x4`, `ARCH_NAME=blackhole`, TP=4, `FABRIC_1D`)
**Target model:** `google/gemma-4-31B-it`
**Drafter:** `google/gemma-4-31B-it-assistant` (EAGLE/MTP head, 4 layers)
**Primary config:** `GEMMA4_SPECULATIVE=1`, `GEMMA4_SPEC_DRAFT_LEN=16`, `GEMMA4_SPEC_ADAPTIVE_K=1`, `GEMMA4_MAX_SEQ_LEN=32768`, `GEMMA4_BOUNDED_SLIDING=0`, `GEMMA4_DRAM_SHARDED=1`
**Plain-floor experiments:** `GEMMA4_FUSED_GREEDY` / `GEMMA4_PREFETCHER` / `GEMMA4_LM_HEAD_MULTI_SPLIT` — all wash or regress; keep off

---

## Performance at a glance (QB2 · P150x4 · TP=4 · 31B-it)

Bottom-line comparison of the configs characterized in this doc. Throughput is **tok/s/user** (batch=1); "lossless" = token-identical / PCC-clean vs the bf16 reference.

| Config | Flags | tok/s/user | vs baseline | Quality | Verdict |
|--------|-------|-----------:|:-----------:|---------|---------|
| bf16 reference | `GEMMA4_PRECISION_ALL=bf16` | ~slower | — | highest (PCC ref) | debug/reference only |
| **Plain decode baseline** | bfp8 (default), interleaved | 19.2 | — | lossless | reference floor |
| **Plain decode + DRAM-sharded** | `GEMMA4_DRAM_SHARDED=1` | **21.1** | **+9.4%** | lossless (PCC pass) | ✅ best non-spec, safe |
| Plain decode + DRAM-sharded + split bfp4 | `GEMMA4_DRAM_SHARDED=1` `GEMMA4_PRECISION_SHARED_MLP=bfp4` | 22.7 | +18% | PCC 0.974, degraded gen | ⚠️ experiment only |
| Plain decode + on-device sampling | `GEMMA4_SAMPLE_ON_DEVICE=1` | 16.3 | −17% | lossless | ❌ do not use |
| **Speculative K=16 (+ DRAM-sharded)** | `GEMMA4_SPECULATIVE=1` `GEMMA4_SPEC_DRAFT_LEN=16` `GEMMA4_DRAM_SHARDED=1` | **14–81** (workload) | up to **+4.2×** | lossless (token-identical) | ✅ high-accept workloads |
| **Spec + adaptive K (+ DRAM-sharded)** | above + `GEMMA4_SPEC_ADAPTIVE_K=1` (default on) | **raises creative floor** (K→6/8) | acceptance-aware | lossless | ✅ **recommended default** |

**Speculative throughput is acceptance-limited** (`tok/s ≈ (mean_accept + 1) / iter_time`): ~81 tok/s on structured/repetitive text, ~52 on long-context analytical, ~44 on code, ~14 on short creative prompts at fixed K=16. **Adaptive K** (2026-07-09) picks K∈{16,8,6} per request so creative/summarize prompts stop paying for unused draft slots. It is lossless and stacks on top of the DRAM-sharded target forward. Full flag/setup detail is in the [Quick reference](#quick-reference-setup-flags--latest-results).

---

## Executive summary

We brought up **Gemma4 31B speculative decoding at K=16** on QB2, validated parity against greedy decode, integrated a **FastAPI server**, and fixed several correctness and stability issues (KV cache pollution, CCL trace deadlock, missing stop tokens). **Peak measured throughput is ~52 tok/s/user** on long-context, low-entropy workloads; **server throughput scales with draft acceptance** (~13–81 tok/s depending on prompt type). On top of the baseline **~19 tok/s/user** plain decode we shipped a **lossless DRAM-sharded matmul path** (`GEMMA4_DRAM_SHARDED=1`, +9.4%, PCC-clean) and characterized a bfp4 precision path (kept **opt-in only** — it fails the PCC gate and degrades generation). See [Quick reference](#quick-reference-setup-flags--latest-results) for the recommended production config; the per-phase optimization detail is under [Model implementation map](#model-implementation-map-for-baseline-optimization).

---

## Quick reference: setup, flags & latest results

> This is the consolidated "what to actually run" section. Everything below it is the deeper per-phase history and rationale.

### A. Model + inference server setup (QB2 / P150x4)

```bash
# --- environment (once per shell) ---
source ~/tt-metal/python_env/bin/activate
cd ~/tt-metal
export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal
export ARCH_NAME=blackhole MESH_DEVICE=P150x4          # QB2 = 4× P150, TP=4
export HF_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main
unset TT_CACHE_PATH                                    # use the HF-snapshot tensor cache (bfp8)
# Optional explicit drafter (defaults to <HF_MODEL>-assistant):
# export GEMMA4_ASSISTANT_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant/snapshots/main

# --- recommended production config (fast + lossless) ---
export GEMMA4_SPECULATIVE=1        # biggest throughput lever; token-identical to greedy
export GEMMA4_SPEC_DRAFT_LEN=16    # high-band K (code / long-ctx / structured)
export GEMMA4_SPEC_ADAPTIVE_K=1    # per-request K: 16 / 8 / 6 by workload (default ON with spec)
# export GEMMA4_SPEC_DRAFT_LEN_MID=8
# export GEMMA4_SPEC_DRAFT_LEN_LOW=6
export GEMMA4_MAX_SEQ_LEN=32768    # total context window (prompt + completion)
export GEMMA4_BOUNDED_SLIDING=0    # REQUIRED for speculative decode
export GEMMA4_DRAM_SHARDED=1       # +9.4% lossless decode matmuls (safe; auto-falls back)
unset  GEMMA4_SAMPLE_ON_DEVICE     # host sampling is faster for batch-1 greedy here

# --- launch ---
tt-smi -r                          # only if recovering from a hang/force-kill
bash models/demos/gemma4/demo/run_server.sh 2>&1 | tee gemma4_server.log
```

Weights are **bfp8** (`attention` + `shared_mlp`) via `precision_overrides.json`; the first run builds the tensor cache under the HF snapshot dir and later runs reuse it. The server exposes `/health`, `/v1/models`, `/v1/chat/completions` (OpenAI-style), and `/generate`.

**Smoke test once "Gemma4 server ready" logs:**

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the capital of France? Answer in one word.","max_tokens":8,"instruct":true}' \
  | python3 -m json.tool     # expect "Paris"; check metrics.decode_tok_s_per_user / mean_accept
```

### B. Main flags — perf vs. accuracy

**Throughput levers (little or no quality cost):**

| Flag | Default | Effect | Measured | Quality |
|------|---------|--------|----------|---------|
| `GEMMA4_SPECULATIVE=1` | 0 | EAGLE drafter + fused-trace verify | 19 → **~52 tok/s** (accept-limited) | **lossless** — token-identical to greedy |
| `GEMMA4_SPEC_DRAFT_LEN=K` | 4 (16 on QB2) | High-band draft candidates/iter | K=16 best on long ctx; K=32 collapses (accept 2.87/32) | lossless |
| `GEMMA4_SPEC_ADAPTIVE_K` | 1 when speculative | Per-request K from prompt class (high/mid/low) | creative→K=6, summarize→K=8, code/long→K=16 | lossless |
| `GEMMA4_SPEC_DRAFT_LEN_MID` / `_LOW` | 8 / 6 | Mid/low bands used by adaptive K | — | lossless |
| `GEMMA4_DRAM_SHARDED=1` | off | DRAM-width-sharded decode matmuls (MLP+ATTN) | +9.4% (47.5 ms/tok) | **lossless** (PCC unchanged); auto-falls back |
| ↳ `GEMMA4_DRAM_SHARDED_MLP` / `_ATTN` / `_LMHEAD` | off | Per-module opt-in of the above | MLP +6.8%, ATTN +~2.4% more; lmhead auto-skips (L1 budget) | lossless |
| `GEMMA4_FUSED_GREEDY=1` | 0 | Plain fused greedy (embed→layers→lm_head→argmax in one CCL trace) | **wash** (22.44→22.48) | lossless (greedy only) |
| `GEMMA4_FUSED_GREEDY_TRACE` | 1 | Metal trace for fused greedy (0 = eager fallback) | — | — |
| `GEMMA4_LM_HEAD_MULTI_SPLIT=1` | 0 | Multi-split DRAM-sharded lm_head (K-grid 56 cores) | device A/B pending | lossless |
| `GEMMA4_LM_HEAD_MAX_COLUMNS` | 8192 | Split width for multi-split lm_head | — | — |
| `GEMMA4_PREFETCHER=1` | 0 | Weight prefetcher (MLP ring matmul); mutually exclusive w/ DRAM_SHARDED | **−22%** (16.0 vs 20.5 interleaved) — keep off | end-to-end green |
| `GEMMA4_MAX_SEQ_LEN` | 1024 (server) | Total context; power-of-2 for long prompts | — | — |
| `GEMMA4_ENABLE_TRACE=1` | 1 | On-device decode/prefill trace (disabled for fused-greedy prefill) | large | — |

**Precision levers (real quality trade-off):**

| Setting | Perf | Quality | Recommendation |
|---------|------|---------|----------------|
| bf16 (`GEMMA4_PRECISION_ALL=bf16`) | slowest | highest | reference / debugging only |
| **bfp8 (default: `shared_mlp`+`attention`)** | baseline | PCC-clean | **shipped default** |
| split bfp4 (`GEMMA4_PRECISION_SHARED_MLP=bfp4`, `down` stays bfp8) | +bfp4 ≈ 44 ms/tok (22.7 tok/s) | PCC ~0.974 (below 0.99 gate) | **opt-in experiment only** |
| full bfp4 MLP | fastest | PCC 0.94, garbage generation | **do not use** |

Per-module precision overrides (applied on top of `precision_overrides.json`, no JSON edit needed): `GEMMA4_PRECISION_SHARED_MLP`, `GEMMA4_PRECISION_SHARED_MLP_DOWN`, `GEMMA4_PRECISION_ATTENTION`, `GEMMA4_PRECISION_LM_HEAD`, `GEMMA4_PRECISION_ALL` — values `bf16` / `bfp8` / `bfp4` / `fp32`.

**Per-request sampling knobs (`/generate` and `/v1/chat/completions` JSON body):**

| Field | Default | Notes |
|-------|---------|-------|
| `instruct` | `true` | **Required** for coherent output on the IT model (chat template). `false` → greedy loops. |
| `temperature` | `0.0` | `0.0` = greedy (only mode spec decode uses). `>0` enables sampling (+`top_k=32`). |
| `top_p` | `1.0` | Set e.g. `0.95` with `temperature>0`. |
| `repetition_penalty` | `1.15` | `1.0` disables. Host-side. |
| `repetition_max_streak` | `12` | Stop after N identical tokens. |
| `max_tokens` | 128 (server default) | Cap creative prompts at 128–256, not 2048. |
| `draft_len` | unset | Pin speculative K for this request (skips adaptive). Metrics echo `draft_len` + `adaptive_workload`. |

**Do NOT use for perf:** `GEMMA4_SAMPLE_ON_DEVICE=1` (−17% for batch-1 greedy here — see Phase 1), full-MLP `bfp4`, `GEMMA4_SPEC_FUSED_RESEED=0` (hangs), `GEMMA4_ATTN_L1_NORM=1` (+0.2%, noise).

### C. Latest consolidated results (QB2, P150x4, TP=4, 31B-it, bfp8)

**Plain greedy decode (non-spec), long-context-64k:**

| Config | ms/token | tok/s/user | Δ | PCC |
|--------|----------|-----------|---|-----|
| Baseline (bfp8, interleaved) | 51.98 | 19.24 | — | pass |
| **+ `GEMMA4_DRAM_SHARDED=1`** (recommended) | **47.50** | **21.05** | **+9.4%** | pass (lossless) |
| + split bfp4 gate/up + DRAM-sharded (opt-in) | 44.04 | 22.71 | +18% | 0.974 (below gate) |

**Speculative decode K=16 by workload (acceptance-limited):**

| Workload | mean accept / K | tok/s/user |
|----------|-----------------|-----------|
| Structured / repetitive | 15.6 / 16 | **~81** |
| Long-context analytical continuation | 10.77 / 16 | **~52** |
| Code (merge_sort, instruct) | 7.8 / 16 | ~44 |
| Summarize passage | 2.9 / 16 | ~20 |
| Creative story (short prompt) | 1.6 / 16 | ~14 |

Formula: `tok/s ≈ (mean_accept + 1) / iteration_time`; iteration time ~190–230 ms on QB2 — throughput is **acceptance-limited**, not per-step latency. Spec decode is lossless, so it stacks with `GEMMA4_DRAM_SHARDED=1` on the target forward.

---

## Work completed

### 1. Speculative decode validation (K=16)

| Test | Result |
|------|--------|
| `test_spec_decode_matches_greedy` | PASSED — token-identical vs greedy at K=16 |
| `test_verify_batchsize_invariance` | PASSED — batch=17 (K+1) with concat_heads padding fix |
| `test_spec_decode_traced` | PASSED — TOKEN-IDENTICAL, ~13.6 tok/s wall at 64 tokens |
| `test_full_model_decode` 1×4 | PASSED |

**Fixes in tests / attention:**

- **`tests/unit/test_spec_decode.py`** — explicit `warmup_model_prefill(..., can_sample_on_device=False, greedy_only=True)` before spec runs; `warmup_prefill=False` on request prefills.
- **`tt/attention/operations.py`** — pad unshardable verify batches (e.g. 17→18) before sharded decode `concat_heads`, slice back.

### 2. Precision (bfp8 overrides)

- **`tt/precision.py`** — fixed `model_key_from_path()` for HF cache snapshot paths (`main` → model name).
- **`precision_overrides.json`** — `attention` + `shared_mlp` → **bfp8** for `gemma-4-31B-it` on QB2.

### 3. Spec decode runtime (fused trace)

- **`tt/spec_decode.py`**
  - Reuse single **fused CCL trace** across requests (capture once, replay many).
  - **`release_fused_trace()`** for clean teardown (avoid recapture deadlock).
  - Documented constraint: **no interleaving** of prefill CCL trace replay with fused trace replay.

### 4. FastAPI server (`demo/serve_fastapi.py`)

| Issue | Fix |
|-------|-----|
| Plain "Hello" warmup before assistant load polluted KV | Reorder: `_init_spec_decoder()` → `_warmup_spec_decode()`; no plain decode warmup when `GEMMA4_SPECULATIVE=1` |
| Stale KV between requests | `_reset_paged_kv_cache()` + `_per_layer_kv_cache()` unwrap before each prefill |
| Second request hang (CCL trace interleave) | Spec path prefill: **`enable_trace=False`** (eager prefill); fused trace only CCL trace on queue |
| Startup crash on KV reset | Unwrap `[[[k,v],…]]` model wrapper before `ttnn.fill` |
|| Looping / garbage after correct answer | **`_augment_gemma4_stop_tokens()`** in `tt/generator.py` — add `<turn|>` (106) in addition to `<eos>` (1) |

### 5. Reference demo (working spec path)

- **`demo/text_demo_v2.py`** — load assistant **after** target prefill; single `spec.generate()` per run; long-context book prompt (~30K tokens).

---

## Performance achieved

### Common environment (QB2)

Use this block for all pytest and server runs below:

```bash
source ~/tt-metal/python_env/bin/activate
cd ~/tt-metal
export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal
export ARCH_NAME=blackhole MESH_DEVICE=P150x4
export HF_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main
# Optional: explicit drafter (defaults to <HF_MODEL>-assistant)
# export GEMMA4_ASSISTANT_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it-assistant/snapshots/main
unset TT_CACHE_PATH   # use HF snapshot tensor cache under model dir
# After force-kill / hang: tt-smi -r
```

---

## Recorded performance & how to reproduce

All numbers below were measured on **QB2 (P150x4, TP=4)**, **31B-it + assistant**, **bfp8** on attention + shared_mlp unless noted. Throughput is **tok/s/user** (batch=1). Spec numbers use **steady-state decode** from demo logs (`Decode:` line) or server `metrics.decode_tok_s_per_user`.

| # | Benchmark | Mode | tok/s/user | accept / K | tokens/iter | Reproduce | Log |
|---|-----------|------|------------|------------|-------------|-----------|-----|
| 1 | Long context plain decode | Greedy, no spec | **19.2** | — | 1.0 | [§1 Plain long-context](#1-plain-greedy-decode-long-context) | `plain_longwin.log` |
| 2 | Long context spec **K=16** | Fused trace, ~30K ctx | **51.6** | 10.77 / 16 | 11.77 | [§2 Spec K=16 long-context](#2-speculative-decode-k16-long-context) | `spec_k16.log` |
| 3 | Long context spec **K=8** | Fused trace, ~30K ctx | **44.6** | 6.48 / 8 | 7.48 | [§3 Spec K=8 long-context](#3-speculative-decode-k8-long-context) | `spec_k8.log` |
| 4 | Long context spec **K=32** | Fused trace, ~30K ctx | **11.2** | 2.87 / 32 | 3.87 | Same as §2 with `--spec-draft-len 32` | `spec_k32.log` |
| 5 | Spec traced unit test | Fused trace, 64 tok | **13.6** (wall) | 2.76 / 16 | 3.76 | [§4 Unit: traced spec](#4-unit-tests-spec-decode) | `spec_k16_traced.log` |
| 6 | Server — code gen | instruct, merge_sort | **43.9** | 7.8 / 16 | 8.8 | [§5 FastAPI server](#5-fastapi-server) + [curl A](#server-curl-examples) | `gemma4_server.log` |
| 7 | Server — structured repeat | instruct, numbered lines | **80.9** | 15.6 / 16 | 16.6 | [§5](#5-fastapi-server) + [curl B](#server-curl-examples) | `gemma4_server.log` |
| 8 | Server — creative story | instruct, short prompt | **13.7** | 1.6 / 16 | 2.6 | [§5](#5-fastapi-server) + [curl C](#server-curl-examples) | `gemma4_server.log` |
| 9 | Server — Q&A one word | instruct, “capital of France” | **~7** (2 tok) | 3.0 / 16 | 4.0 | [§5](#5-fastapi-server) + [curl D](#server-curl-examples) | `gemma4_server.log` |
| 10 | CI reference (different HW) | Plain decode | **9.5** | — | 1.0 | T3K 1×8 — see `README.md` | CI logs |

**Formula:** `tok/s ≈ (mean_accept + 1) / iteration_time`. Low acceptance (creative/open-ended) → ~13 tok/s even when the stack is healthy.

---

### Reproduction commands

#### 1. Plain greedy decode (long context)

~**19.2 tok/s** at ~30K-token context window (bounded sliding KV enabled by demo default).

```bash
export GEMMA4_MAX_SEQ_LEN=32768
pytest models/demos/gemma4/demo/text_demo_v2.py \
  -k "long-context-64k" -sv 2>&1 | tee plain_longwin.log
# Grep: Decode: ... tok/s/user
grep "Decode:" plain_longwin.log | tail -1
```

#### 2. Speculative decode K=16 (long context)

~**51.6 tok/s**, mean accept **10.77/16** — best measured spec throughput on QB2.

```bash
export GEMMA4_MAX_SEQ_LEN=32768 GEMMA4_BOUNDED_SLIDING=0
pytest models/demos/gemma4/demo/text_demo_v2.py \
  -k "long-context-64k" --speculative --spec-draft-len 16 -sv \
  2>&1 | tee spec_k16.log
grep -E "mean accepted|Decode:" spec_k16.log | tail -3
```

#### 3. Speculative decode K=8 (long context)

~**44.6 tok/s**, mean accept **6.48/8** — good when acceptance is moderate.

```bash
export GEMMA4_MAX_SEQ_LEN=32768 GEMMA4_BOUNDED_SLIDING=0
pytest models/demos/gemma4/demo/text_demo_v2.py \
  -k "long-context-64k" --speculative --spec-draft-len 8 -sv \
  2>&1 | tee spec_k8.log
```

Alternative entry point (shorter default prompt, 4096 ctx):

```bash
export GEMMA4_SPEC_DRAFT_LEN=16 GEMMA4_BOUNDED_SLIDING=0 \
       GEMMA4_MAX_SEQ_LEN=4096 GEMMA4_MAX_NEW_TOKENS=200
pytest models/demos/gemma4/demo/text_demo_v2.py::test_demo_spec_decode -sv \
  2>&1 | tee spec_demo_short.log
```

Long ~50K prefill variant:

```bash
export GEMMA4_MAX_SEQ_LEN=65536 GEMMA4_MAX_NEW_TOKENS=64 GEMMA4_BOUNDED_SLIDING=0
# test_demo_spec_decode is env-configured (GEMMA4_MAX_SEQ_LEN / GEMMA4_SPEC_DRAFT_LEN);
# a larger GEMMA4_MAX_SEQ_LEN drives the long-prefill variant.
export GEMMA4_SPEC_DRAFT_LEN=16
pytest models/demos/gemma4/demo/text_demo_v2.py::test_demo_spec_decode -sv \
  2>&1 | tee spec_long50k.log
```

#### 4. Unit tests (spec decode)

Correctness / parity (no perf assertion, but logs accept rate):

```bash
export GEMMA4_SPEC_DRAFT_LEN=16 GEMMA4_BOUNDED_SLIDING=0

# Token-identical vs greedy (K=16), mesh 1x4
pytest models/demos/gemma4/tests/unit/test_spec_decode.py::test_spec_decode_matches_greedy \
  -k "1x4" -sv 2>&1 | tee spec_k16_parity.log

# Batch=17 verify path (K+1 padding fix)
export GEMMA4_SPEC_VERIFY_PAD=16 GEMMA4_SPEC_TEST_TOKENS=128
pytest models/demos/gemma4/tests/unit/test_spec_decode.py::test_verify_batchsize_invariance \
  -k "1x4" -sv 2>&1 | tee spec_k16_batch_inv.log

# Fused trace correctness + ~13.6 tok/s wall on 64 tokens
pytest models/demos/gemma4/tests/unit/test_spec_decode.py::test_spec_decode_traced \
  -sv 2>&1 | tee spec_k16_traced.log
grep -E "tok/s/u|mean accepted" spec_k16_traced.log | tail -3
```

Full-model decode PCC (not a throughput test):

```bash
pytest models/demos/gemma4/tests/unit/test_model.py::test_full_model_decode \
  -k "1x4" -sv
```

Optional env overrides for unit tests:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMMA4_SPEC_DRAFT_LEN` | 4 (16 for QB2 tuning) | Draft length K |
| `GEMMA4_SPEC_TEST_TOKENS` | 24–48 | Generated tokens in parity tests |
| `GEMMA4_SPEC_PROMPT` | test-specific | Override prompt string |
| `GEMMA4_SPEC_VERIFY_PAD` | 4 | Verify batch padding for batch-inv test |
| `GEMMA4_BOUNDED_SLIDING` | 0 for spec | Must be 0 for spec decode |

#### 5. FastAPI server

End-to-end latency + throughput (includes TTFT ~95 ms on short prompts).

```bash
export GEMMA4_SPECULATIVE=1 GEMMA4_SPEC_DRAFT_LEN=16 \
       GEMMA4_MAX_SEQ_LEN=65536 GEMMA4_BOUNDED_SLIDING=0
unset GEMMA4_SAMPLE_ON_DEVICE
bash models/demos/gemma4/demo/run_server.sh 2>&1 | tee gemma4_server.log
```

##### Server curl examples

Use `"instruct": true` for chat (required for quality on this IT model).

**A — High throughput code (~44 tok/s):**
```bash
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a Python function called merge_sort that sorts a list of integers using the merge sort algorithm. Include a docstring and type hints.","max_tokens":220,"instruct":true}' \
  | python3 -m json.tool
```

**B — Structured / repetitive (~81 tok/s):**
```bash
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Repeat the following sentence 10 times, numbering each line: The quick brown fox jumps over the lazy dog.","max_tokens":180,"instruct":true}' \
  | python3 -m json.tool
```

**C — Creative story (~14 tok/s, low accept):**
```bash
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"tell me a story in 500 words","max_tokens":2048,"instruct":true}' \
  | python3 -m json.tool
```

**D — Short factual Q&A (quality check, low tok count):**
```bash
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the capital of France? Answer in one word.","max_tokens":24,"instruct":true}' \
  | python3 -m json.tool
```

Check metrics in the JSON response: `metrics.decode_tok_s_per_user`, `metrics.mean_accept`, `metrics.tokens_per_iter`.

---

### Baseline Server setup
```bash
source ~/tt-metal/python_env/bin/activate
cd ~/tt-metal
export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal
export ARCH_NAME=blackhole MESH_DEVICE=P150x4
export HF_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main
# Throughput path (spec decode + adaptive K + DRAM-sharded)
export GEMMA4_SPECULATIVE=1
export GEMMA4_SPEC_DRAFT_LEN=16
export GEMMA4_SPEC_ADAPTIVE_K=1    # creative→6, summarize→8, code/long→16
export GEMMA4_DRAM_SHARDED=1
export GEMMA4_MAX_SEQ_LEN=32768
export GEMMA4_BOUNDED_SLIDING=0    # required for spec
unset GEMMA4_SAMPLE_ON_DEVICE      # host sampling is faster for batch-1 greedy
unset TT_CACHE_PATH                # use HF snapshot tensor cache (bfp8)
tt-smi -r   # if restarting after a hang/kill
bash models/demos/gemma4/demo/run_server.sh
```

### Verify Server

```bash
# 0. Health — expect speculative=true, spec_adaptive_k=true, spec_draft_len=16
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

# 1. Sanity — should be "Paris" in ~1 token
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the capital of France? Answer in one word.","max_tokens":32,"instruct":true}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"

# 2. Throughput + quality — code (expect K=16, ~40+ tok/s with spec)
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a Python merge_sort function with type hints.","max_tokens":220,"instruct":true}' \
  | python3 -c "import sys,json; m=json.load(sys.stdin)['metrics']; print(m)"
# expect metrics.draft_len=16, adaptive_workload=high_accept

# 3. Creative — adaptive K=6 (cap max_tokens 128–256, NOT 2048)
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"tell me a story in 500 words","max_tokens":128,"instruct":true}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['metrics']); print(d['text'][:500])"
# expect metrics.draft_len=6, adaptive_workload=low_accept

# 4. Pin K (skip adaptive) — control A/B
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"tell me a story in 500 words","max_tokens":128,"instruct":true,"draft_len":16}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['metrics'])"
```


### Baseline (plain greedy decode, no spec)

| Context / setup | tok/s/user | Notes |
|-----------------|------------|--------|
| ~30K context, QB2 bfp8 | **~19.2** | Long-window plain decode benchmark |
| CI 31B T3K 1×8 | ~9.5 | README reference (different HW) |

### Speculative decode K=16 (QB2, bfp8) — by workload

| Workload | mean accept / K | tokens/iter | tok/s/user | Notes |
|----------|-----------------|-------------|------------|--------|
| Long context (~30K), analytical continuation | **10.77 / 16** | 11.77 | **~52** | `text_demo_v2` + `spec_k16.log` |
| Code (merge_sort, instruct) | 7.8 / 16 | 8.8 | **~44** | Server, coherent output |
| Structured repeat (numbered lines) | 15.6 / 16 | 16.6 | **~81** | Server; highly predictable |
| Creative story (short prompt) | 1.6 / 16 | 2.6 | **~14** | Open-ended; low drafter accept |
| Summarize passage | 2.9 / 16 | 3.9 | **~20** | Moderate entropy |

**Key formula:** `tok/s ≈ (mean_accept + 1) / iteration_time`. Iteration time is ~190–230 ms on QB2; throughput is **acceptance-limited**, not a regression in per-step latency.

### Server (end-to-end, after fixes)

- **TTFT:** ~95 ms (short prompt, traced prefill warmup already captured).
- **Healthy spec path:** no hang across multiple sequential requests.
- **Quality:** use **`instruct=true`** for chat; `instruct=false` on this IT model yields degenerate greedy loops on short prompts (expected — model expects chat template).

### Quick server start (reference)

```bash
# Spec path (recommended) — adaptive K ON
source ~/tt-metal/python_env/bin/activate
cd ~/tt-metal
export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal
export ARCH_NAME=blackhole MESH_DEVICE=P150x4
export HF_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main
export GEMMA4_SPECULATIVE=1 GEMMA4_SPEC_DRAFT_LEN=16 GEMMA4_SPEC_ADAPTIVE_K=1
export GEMMA4_DRAM_SHARDED=1 GEMMA4_MAX_SEQ_LEN=32768 GEMMA4_BOUNDED_SLIDING=0
unset GEMMA4_SAMPLE_ON_DEVICE TT_CACHE_PATH
bash models/demos/gemma4/demo/run_server.sh

# Plain greedy only (no drafter) — for floor A/B:
# unset GEMMA4_SPECULATIVE; export GEMMA4_MAX_SEQ_LEN=65536
```

Example high-throughput prompt (instruct, code):

```json
{
  "prompt": "Write a Python function called merge_sort that sorts a list of integers using the merge sort algorithm. Include a docstring and type hints.",
  "max_tokens": 220,
  "instruct": true
}
```

---

## Known limitations & do-not-use

- **`GEMMA4_SPEC_FUSED_RESEED=0`** (shift mode) — hangs on fused replay #2 when `draft_len >= 2`.
- **Recapturing fused trace per request without releasing** — deadlocks at compile run.
- **Prefill trace replay immediately before fused spec / plain-fused replay** — deadlocks; server/demo use eager prefill for those paths.
- **`GEMMA4_FUSED_GREEDY` + `GEMMA4_SPECULATIVE=1`** — speculative path wins; fused greedy is ignored.
- **`GEMMA4_FUSED_GREEDY` + bounded sliding / PLI / temp>0** — skipped (falls back to host loop).
- **Force-kill server** — run `tt-smi -r` before restart; only one server instance at a time.
- **12B as drafter for 31B** — not supported (EAGLE head is target-specific); use 12B as **target** with `gemma-4-12B-it-assistant` instead.

---

## Model implementation map (for baseline optimization)

Use this map for the **next round: increase full-model (non-spec) decode throughput**. Spec decode builds on the same target forward; wins there lift both paths.

### Entry points & orchestration

| Path | Role |
|------|------|
| **`tt/generator.py`** | `Gemma4Generator.from_pretrained`, model args patching, encode/stop tokens, entry for demos/server |
| **`tt/generator_trace.py`** | Prefill trace policy, `warmup_gemma4_batched_prefill_traces`, trace ISL caps (128…4096) |
| **`tt/model.py`** | **Main model** — `Gemma4Model`: RoPE caches, embedding, `ttnn_prefill_forward` / `ttnn_decode_forward`, LM head, verify forward for spec |
| **`tt/common.py`** | `create_tt_model`, assistant factory, mesh/CCL setup |
| **`config.py`** / **`configs/gemma-4-31B-it/`** | HF config stubs |

### Layer stack (decode hot path)

| Path | Role |
|------|------|
| **`tt/layer.py`** | `Gemma4DecoderLayer` — prefill/decode routing, norms, attn + MLP/MoE |
| **`tt/attention/decode.py`** | **Decode attention** — paged SDPA, `paged_update_cache`, sliding/global |
| **`tt/attention/prefill.py`** | Prefill attention, `paged_fill_cache`, chunked SDPA |
| **`tt/attention/operations.py`** | Shared attn ops, batched decode concat, sliding-window SDPA |
| **`tt/attention/kv_cache.py`** | KV cache allocation (paged block layout) |
| **`tt/attention/kv_cache_hybrid.py`** | Bounded sliding page tables (not used for spec; unbounded required) |
| **`tt/attention/weights.py`** | QKV/O-proj load, TP sharding |
| **`tt/shared_mlp.py`** | Dense MLP matmuls (31B uses shared MLP on non-MoE path) |
| **`tt/experts/decode.py`** | MoE expert decode (26B-A4B; 31B is dense in this demo path) |
| **`tt/experts/prefill.py`** | MoE expert prefill |
| **`tt/moe.py`** / **`tt/router.py`** | MoE routing (26B variant) |
| **`tt/rms_norm.py`** | RMSNorm |
| **`tt/ccl.py`** | TP all-gather / reduce-scatter (major decode cost on mesh) |

### Precision & weights

| Path | Role |
|------|------|
| **`tt/precision.py`** | Dtype overrides per module; `model_key_from_path` |
| **`precision_overrides.json`** | bfp8 on `attention`, `shared_mlp` for 31B/12B |

### Speculative decode (downstream of target forward)

| Path | Role |
|------|------|
| **`tt/spec_decode.py`** | Fused greedy spec loop, trace capture/replay, verify batching |
| **`tt/assistant/model.py`** | 4-layer EAGLE drafter, cross-attn into target KV |

### Demos, server, tests

| Path | Role |
|------|------|
| **`demo/text_demo_v2.py`** | Reference spec + plain decode demo |
| **`demo/serve_fastapi.py`** | Production FastAPI server |
| **`demo/run_server.sh`** | Server launcher |
| **`tests/unit/test_spec_decode.py`** | Spec parity & batch invariance |
| **`tests/unit/test_model.py`** | Full model PCC / decode |
| **`tests/unit/test_attention.py`** | Attention decode/prefill parity |
| **`tests/unit/test_prefill_trace_perf.py`** | Prefill trace perf |

### Shared tt_transformers integration

| Path | Role |
|------|------|
| **`models/tt_transformers/tt/generator.py`** | Generic prefill/decode loop, trace capture |
| **`models/tt_transformers/tt/common.py`** | `preprocess_inputs_prefill`, chat templates |

---

## Next steps after adaptive K (2026-07-09)

> **Stale note:** the older “Suggested next-round focus” that ranked SDPA/CCL first is superseded by Phase 0 profiling (matmul 46%, elementwise+copy ~31%, CCL ~9%, SDPA+KV ~3%). Trust this section + the Phase 0–5 roadmap below.

### Done — Phase A: Adaptive speculative draft length K ✅

**Goal:** raise the **creative / low-accept floor** without hurting high-accept workloads. Fixed K=16 wastes draft+verify work when mean accept is ~1.6/16 (~14 tok/s).

**Shipped (2026-07-09):**

| Piece | Path | Behavior |
|-------|------|----------|
| Classifier + bands | `tt/adaptive_draft_len.py` | high→`SPEC_DRAFT_LEN` (16), mid→8 (summarize), low→6 (creative/open-ended) |
| Trace-safe K change | `tt/spec_decode.py` `set_draft_len()` | Releases fused CCL trace + verify traces when K changes; next `generate` recaptures |
| Server wiring | `demo/serve_fastapi.py` | Per-request adaptive K (default ON with speculative); JSON `draft_len` override; metrics expose `draft_len` + `adaptive_workload` |
| Demo wiring | `demo/text_demo_v2.py` | Same selection when `--spec-draft-len` / env not pinned |
| Host unit tests | `tests/unit/test_adaptive_draft_len.py` | **14 passed** (classifier, env bands, override, `set_draft_len` release) |

**Env knobs:**

```bash
export GEMMA4_SPEC_ADAPTIVE_K=1          # default ON when GEMMA4_SPECULATIVE=1; set 0 to pin K
export GEMMA4_SPEC_DRAFT_LEN=16          # high band
export GEMMA4_SPEC_DRAFT_LEN_MID=8       # summarize-like
export GEMMA4_SPEC_DRAFT_LEN_LOW=6       # creative / short+large max_tokens
# Per-request pin (skips classifier):
# curl ... -d '{"prompt":"...","draft_len":8,"instruct":true}'
```

**Constraints (do not violate):**

- Mid-request K changes are **unsafe** (fused capture writes KV). Only call `set_draft_len` between requests / before `generate`.
- Changing K releases the fused trace → first iter of the new-K request pays recapture cost (same as cold start). Same-K requests reuse the live fused trace.
- Still respect fused-trace teardown: no prefill CCL trace interleaved with fused replay (`enable_trace=False` on spec prefill).

**Device A/B still recommended on QB2** (host tests do not measure tok/s):

```bash
# Creative floor — expect adaptive K=6 vs pinned K=16
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"tell me a story in 500 words","max_tokens":256,"instruct":true}' \
  | python3 -c "import sys,json; m=json.load(sys.stdin)['metrics']; print(m)"
# Code / high-accept — expect K=16 unchanged
curl -s -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a Python function called merge_sort ...","max_tokens":220,"instruct":true}' \
  | python3 -c "import sys,json; m=json.load(sys.stdin)['metrics']; print(m)"
```

Record `metrics.draft_len`, `metrics.adaptive_workload`, `metrics.mean_accept`, `metrics.decode_tok_s_per_user` for both. Pin with `"draft_len":16` on the creative prompt as the control.

### Done — Phase B: Fully fused plain greedy decode loop (was Phase 5) ✅ code

**Why:** Phase 0 showed matmul BW already attacked (+9.4% DRAM-sharded). Plain decode still did **full-vocab lm_head AG + D2H + host argmax every token**, and `_tt_vllm_always_refresh_decode_trace_inputs=True` made naive on-device sampling regress (−17%). Spec already had the pattern; Phase B ports it to non-spec greedy.

**Shipped (2026-07-09):**

| Piece | Path | Behavior |
|-------|------|----------|
| Shared argmax | `tt/device_greedy_argmax.py` | Pad-to-32 multicore untilize+argmax; used by spec + plain |
| Fused greedy decoder | `tt/plain_fused_decode.py` | One CCL trace/token: `ttnn_verify_forward`(batch=1) → argmax; host reads 1× uint32 |
| Spec refactor | `tt/spec_decode.py` | Delegates argmax helpers to shared module (behavior unchanged) |
| Demo | `demo/text_demo_v2.py` | `GEMMA4_FUSED_GREEDY=1` → fused path; eager prefill; releases trace |
| Server | `demo/serve_fastapi.py` | Same for non-spec greedy; skips when speculative / bounded sliding / PLI |
| Host tests | `tests/unit/test_plain_fused_decode.py` | **4 passed** (env gate + release teardown) |

**Env / constraints:**

```bash
export GEMMA4_FUSED_GREEDY=1           # opt-in; default OFF
export GEMMA4_FUSED_GREEDY_TRACE=1     # default ON; set 0 for eager fallback
export GEMMA4_BOUNDED_SLIDING=0        # required (hybrid page tables not wired)
unset GEMMA4_SPECULATIVE               # fused greedy is the *plain* path; spec wins if both set
export GEMMA4_DRAM_SHARDED=1           # stack with Phase 2a
# Prefill is forced eager when fused greedy is on (CCL-trace interleave rule).
# Host repetition_penalty is ignored on the fused path (streak guard still active).
```

**Does NOT flip `always_refresh`** — fused path bypasses `Generator.decode_forward` entirely (avoids Phase 1 regression).

### Phase B — 1:1 fused vs host A/B (copy-paste)

**Goal:** measure only `GEMMA4_FUSED_GREEDY` impact. One variable; everything else pinned.

| Pin (both legs) | Value | Why |
|-----------------|-------|-----|
| Spec | **OFF** (`unset GEMMA4_SPECULATIVE`) | Spec would dominate / skip fused |
| DRAM-sharded | **ON** | Match production plain floor |
| Bounded sliding | **OFF** | Fused path skips if ON |
| Case | **`ci-1`** | `stop_at_eos=False`, **512** tokens — same decode length both legs |
| Sampling | greedy (`temperature=0`) | Fused is greedy-only |

**Do not use `long-context-64k` for the first A/B** — early `<turn|>` can stop after ~2 tokens and make tok/s noise. Use `ci-1` first; optional long-context follow-up after.

#### One-shot script (recommended)

```bash
source ~/tt-metal/python_env/bin/activate
cd ~/tt-metal
export HF_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main
# optional: ARCH_NAME=blackhole MESH_DEVICE=P150x4  (script defaults these)

bash models/demos/gemma4/demo/ab_fused_greedy.sh
# logs: /tmp/gemma4_ab_fused_off.log  /tmp/gemma4_ab_fused_on.log
```

Alt cases: `bash models/demos/gemma4/demo/ab_fused_greedy.sh --case batch-1`

#### Manual 1:1 (same as the script)

```bash
source ~/tt-metal/python_env/bin/activate
cd ~/tt-metal
export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal
export ARCH_NAME=blackhole MESH_DEVICE=P150x4
export HF_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main
unset TT_CACHE_PATH GEMMA4_SPECULATIVE GEMMA4_SAMPLE_ON_DEVICE
export GEMMA4_DRAM_SHARDED=1 GEMMA4_BOUNDED_SLIDING=0 GEMMA4_FUSED_GREEDY_TRACE=1

# ── OFF: host greedy baseline ──
tt-smi -r; sleep 2
export GEMMA4_FUSED_GREEDY=0
pytest models/demos/gemma4/demo/text_demo_v2.py -k "ci-1" -sv 2>&1 | tee /tmp/gemma4_ab_fused_off.log
grep -E "Decode:|generated tokens:" /tmp/gemma4_ab_fused_off.log | tail -3

# ── ON: fused greedy ──
tt-smi -r; sleep 2
export GEMMA4_FUSED_GREEDY=1
pytest models/demos/gemma4/demo/text_demo_v2.py -k "ci-1" -sv 2>&1 | tee /tmp/gemma4_ab_fused_on.log
grep -E "plain-fused|Decode:|generated tokens:|GEMMA4_FUSED_GREEDY=1" /tmp/gemma4_ab_fused_on.log | tail -8
```

#### What to record

| Leg | `GEMMA4_FUSED_GREEDY` | Path banner | ms/token | tok/s/user | generated |
|-----|----------------------|-------------|----------|------------|-----------|
| OFF | `0` | no fused banner | | | expect **512** |
| ON | `1` | `GEMMA4_FUSED_GREEDY=1` + `[plain-fused]` | | | expect **512** |
| **Δ** | — | — | | | must match |

- **Primary metric:** `Decode: … ms/token @ … tok/s/user` (steady-state; fused excludes capture).
- **Ignore TTFT** for this A/B — fused forces eager prefill (CCL-trace rule); that is intentional and not the decode win.
- **Path check:** ON log must contain `GEMMA4_FUSED_GREEDY=1 — plain fused greedy decode`. If missing, fused was skipped (bounded sliding / PLI / temp).
- **Token count:** both legs must report the same `generated tokens` (512 on `ci-1`). Mismatch → invalid A/B.

Paste numbers back into this doc under Phase B when done. Host unit tests do **not** measure throughput.

#### Phase B A/B results (2026-07-09, QB2, `ci-1`, 512 tokens) — NO WIN

| Leg | `FUSED_GREEDY` | Path | ms/token | tok/s/user | generated |
|-----|----------------|------|----------|------------|-----------|
| OFF | 0 | host + Generator decode trace | **44.57** | **22.44** | 512 |
| ON | 1 | plain fused (capture OK) | **44.49** | **22.48** | 512 |
| **Δ** | — | — | −0.08 | +0.04 | match |

**Verdict: fused path is live and correct (same token count, capture/replay ran) but is a wash (±noise).** Not a throughput lever on this stack.

**Why (root cause):** the host baseline already runs a **traced** model body (`Generator.decode_forward` + `enable_trace=True`). The hypothesized win was removing full-vocab D2H + host argmax. In practice:

1. Host decode with `self.sampling is not None` **skips lm_head TP all-gather** (`_apply_lm_head` `is_decode=True`) and only D2Hs the local vocab shard — cheap vs ~44 ms forward.
2. Fused uses `ttnn_verify_forward` → **forces full-vocab AG** before on-device untilize+argmax (~1.6 ms) — extra device work the host path avoids.
3. Fused still does per-token `synchronize_device` + host token/pos bind + 1× uint32 readback — so the host barrier is not eliminated, only shrunk.

**Decision:** keep `GEMMA4_FUSED_GREEDY` as opt-in experiment; **do not enable by default**. Next plain-floor work should target weight BW / lm_head / elementwise (not another host-sync wrapper around the same forward).

### After Phase B (ordered) — superseded by matmul-BW path below

---

## Matmul bandwidth optimization path (2026-07-09)

> **Latest finding:** batch=1 decode is **weight-DRAM-bound**. Phase 0: matmul ~46% of device FW. DRAM-sharded (+9.4%) improved utilization; fused greedy (Phase B) was a wash. Remaining headroom is **effective weight throughput** (bytes/s to compute), not host sync.

### Floor math (31B, TP=4, bfp8)

| Quantity | Value |
|----------|-------|
| Weights / device / token | ~7.8 GB |
| Ideal GDDR floor | ~15–20 ms/tok |
| Measured (DRAM-sharded, `ci-1`) | **~44.5 ms/tok @ 22.4 tok/s** |
| Gap | CCL + elementwise/copy + **sub-peak BW** + lm_head interleaved |

### What already shipped

| Lever | Flag | Result |
|-------|------|--------|
| DRAM-sharded MLP+ATTN | `GEMMA4_DRAM_SHARDED=1` | **+9.4%** lossless |
| bfp8 attention+MLP | `precision_overrides.json` | default |
| Spec + adaptive K | `GEMMA4_SPECULATIVE=1` + `SPEC_ADAPTIVE_K=1` | up to ~4× tok/s/user (accept-limited) |
| Fused plain greedy | `GEMMA4_FUSED_GREEDY=1` | **wash** (22.44→22.48) — keep off |
| Split bfp4 MLP | `PRECISION_SHARED_MLP=bfp4` | +18% but quality fail |

### Active work — #1 Prefetcher + #2 LM-head multi-split

#### #1 — Weight prefetcher (Phase 2b) — END-TO-END GREEN, **REGRESSES** (keep off)

**Goal:** overlap next-layer weight DRAM reads with current matmul/CCL via `tt_transformers` `Prefetcher` + global CB.

| Item | Detail |
|------|--------|
| Peer ref | `models/tt_transformers/tt/prefetcher.py`, wired in `mlp.py` / `attention.py` |
| Size check | Gemma4 31B **passes** at `ring_size=64`: `tiles_per_core=576`, ~627 KB/core &lt; 1 MB L1 |
| Whitelist | ✅ `gemma-4-31B` in `VERIFIED_MODEL_CONFIGS` |
| Weight pad | Hidden + MLP N/K **5376 → 6144** (ring tile-aligned shards) |
| Mutual exclusion | **Incompatible with `GEMMA4_DRAM_SHARDED`** — `common.py` force-offs DRAM flags when PREFETCHER=1 |
| Env | `GEMMA4_PREFETCHER=1` (requires `HF_MODEL` containing `gemma-4-31B`) |
| Code landed | `tt/prefetcher_ring.py`; `common.py`; `model.py` switch_mode + receivers-only SD reload; `shared_mlp.py` ring path; `prefetcher_utils.py` (worker grids, split lm_head); attention/SDPA worker constraints |
| Scope | **MLP only** (gate/up/down); attention/lm_head stay on worker SD without ring GCB |
| **ci-1 A/B (2026-07-09)** | OFF interleaved **20.54 tok/s**; ON Prefetcher **16.04 tok/s** (−22%) |
| Verdict | **Keep off.** Prefer `GEMMA4_DRAM_SHARDED=1` (~21–22). Overhead from dual SD + ring pad + split lm_head + untilize/pad path dominates any BW overlap. |

**Hard bugs fixed to get green (do not regress):**
1. Receivers-only worker sub-device (GCB bbox hole fill).
2. Ring-pad hidden K/N 5376→6144; skip mid crop/re-pad (gate/up N == down K).
3. SDPA 16-core pick from fragmented receivers (64 fails `num_cores_to_corerangeset`).
4. TILE `pad` → untilize + `tilize_with_val_padding` on worker grids.
5. Prefetcher lm_head: column-split interleaved matmuls (full vocab OOMs 16-core L1).
6. Idempotent SD reload (second reload invalidated decode trace).

```bash
# Prefetcher A/B — compare to interleaved, NOT DRAM_SHARDED
unset GEMMA4_DRAM_SHARDED GEMMA4_SPECULATIVE GEMMA4_FUSED_GREEDY
export GEMMA4_BOUNDED_SLIDING=0
export GEMMA4_PREFETCHER=0
pytest models/demos/gemma4/demo/text_demo_v2.py -k "ci-1" -sv 2>&1 | tee /tmp/pf_off.log
export GEMMA4_PREFETCHER=1
pytest models/demos/gemma4/demo/text_demo_v2.py -k "ci-1" -sv 2>&1 | tee /tmp/pf_on.log
grep "Decode:" /tmp/pf_off.log /tmp/pf_on.log
# Measured: OFF 20.54 tok/s · ON 16.04 tok/s
```

#### #2 — LM-head multi-split DRAM-shard — CORRECTNESS BLOCKER (2026-07-09)

**Goal:** accelerate the tied lm_head (largest single weight tensor: ~704 MB/device bf16 at vocab/tp=65536) that previously **auto-skipped** `DramShardedMatmul` (gcd→8 cores → L1 CB overflow).

| Item | Detail |
|------|--------|
| Peer ref | `models/tt_transformers/tt/lm_head.py` — split vocab columns + K-derived core grid |
| Why single-shard fails | `gcd(168,2048)=8` → `per_core_N=256` → ~8.4 MB CB &gt; 1.3 MB budget |
| Code | `MultiSplitDramShardedMatmul` in `tt/dram_sharded.py`; default `max_columns=7168` → 9×7168 + 1024; **unified 56-core** grid with pad+crop on last split; wired in `model.py` |
| Env | `GEMMA4_LM_HEAD_MULTI_SPLIT=1` (+ `GEMMA4_LM_HEAD_MAX_COLUMNS`, `GEMMA4_DRAM_MATH_FIDELITY`) |
| Prefill | Unchanged (interleaved weight kept) |
| Host/device tests | `tests/unit/test_dram_sharded_splits.py`, `tests/unit/test_dram_sharded_lmhead_pcc.py` |
| Est. impact | **~0% measured** on ci-1 (22.1 tok/s both ON/OFF) — not a win until correct |

**Correctness dig (keep flag OFF for production):**

| Path | vs OFF (interleaved full lm_head) |
|------|-----------------------------------|
| Mesh-slice + **interleaved** multi-split | **byte-identical** |
| Mesh-slice + interleaved + HiFi2 CK | **byte-identical** (fidelity alone is fine) |
| Mesh-slice + **DRAM** multi-split (HiFi2) | Coherent but diverges (~251-char common prefix); more late `\textsf` |
| DRAM HiFi4 | Fewer random argmax flips (0/32) but **still diverges earlier** (~130-char); not a tok/s win |

**Root cause (isolated):** not concat-alias, not host re-shard, not mixed core grids, not activation reuse, not decode-trace. Weight DRAM roundtrip is exact (`max_abs=0`). Per-split PCC ≈ **0.9999** / `max_abs≈0.004` vs interleaved — enough to **swap near-tie greedy tokens** (top-2 margins ~0.001–0.003; ~6% flip rate at HiFi2 vs interleaved HiFi2 on real weights). Compounding over 256 tokens → demo drift / `\textsf` loops.

**Hard bugs already fixed (do not regress):**
1. Do not deallocate `concat` inputs (alias → freed logits → garbage/`\textsf`).
2. Slice mesh `lm_head_weight` with `ttnn.slice` — host cat+mapper under TP was wrong.
3. Pad N for core alignment and crop logits (silent `per_core_N` ceil scrambled vocab).
4. Unified K-grid (56) for every split (peer pattern); last 1024 → pad 1792 + crop.

**Next if revisiting:** match peer weight *load* path (`as_tensor` DRAM-width-sharded, no `to_memory_config` convert); or accept non-bitexact and gate on a quality metric that tolerates near-tie flips. Prefetcher MLP remains the better BW bet.

### Priority after Prefetcher A/B (2026-07-09)

| Rank | Path | Est. | Notes |
|------|------|------|-------|
| 1 | Ship **spec + adaptive K + DRAM_SHARDED** | ceiling | Best tok/s/user today |
| 2 | Re-profile one decode step w/ DRAM_SHARDED | — | Confirm matmul vs elementwise/copy |
| 3 | Elementwise + copy fusion (~31% Phase 0) | +5–10%? | Residual / gate×up / DRAM bounces |
| 4 | QKV emit L1 (#16667) | small–med | |
| — | ~~Prefetcher MLP~~ | **−22%** | keep off; revisit only if ring path sheds SD/pad tax |
| — | ~~Prefetcher + attention~~ | — | blocked by MLP wash |
| — | ~~LM-head multi-split~~ | 0% + quality | keep off |
| — | ~~Fused greedy default~~ | wash | keep opt-in off |

**Do not pursue on P150x4:** on-device sampling default, bfp4 default, `GEMMA4_ATTN_L1_NORM=1`, `GEMMA4_FUSED_GREEDY` as production default.

---

## Single-user (batch=1) decode optimization roadmap

**Goal:** raise **non-spec plain greedy decode** from **~19.2 tok/s/user** toward the hardware floor, on QB2 (`P150x4`, blackhole, TP=4), model `gemma-4-31B-it`. These wins also lift the **spec target-forward** (spec builds on the same per-token forward), so they compound with the ~52 tok/s spec path.

> **Read this first (shared context for every phase).** This section is written to be executed by an AI agent with no prior context. Read the referenced files before editing. Do not change spec-decode correctness (`tt/spec_decode.py`) or the fused-trace teardown contract described under [Known limitations](#known-limitations--do-not-use).

### Why single-user decode is slow (root cause)

Batch=1 decode is **latency- and bandwidth-bound**, NOT compute-bound: each matmul multiplies 1 row of activations by the full weights, so the systolic arrays are idle most of the step. Model facts (from `configs/gemma-4-31B-it/config.json`): `num_hidden_layers=60`, `hidden_size=5376`, `vocab_size=262144`, `num_attention_heads=32`, `num_key_value_heads=16`, sliding `head_dim=256` / global `head_dim=512`, `sliding_window=1024`, TP=4.

Measured ~52 ms/token (19.2 tok/s) decomposes roughly into:

| Cost | Where | ~Share | Notes |
|------|-------|--------|-------|
| Weight DRAM reads | all matmuls | hard floor (~15 ms) | ~31 GB weights / 4 dev ≈ 7.8 GB/token/device |
| TP collectives | `tt/ccl.py` | large | 2 `all_reduce`/layer × 60 + embedding AG + lm_head AG, all latency-bound on tiny decode tensors |
| Per-token host sync | `demo/text_demo_v2.py:488` `_host_sample` | large | all-gathers 262k-vocab logits → CPU argmax → token back to device *every token*; breaks pipelining |
| Attention memory bounces | `tt/attention/decode.py` | medium | per-head-norm DRAM round-trips (×3/layer) + DRAM→L1 QKV copy |

The >50% of the step that is NOT weight bandwidth is the exploitable headroom.

### Phase 0 — Measure first (device op profile)

**Context:** Do not optimize blind. Confirm the time split on *this* HW before touching code.

- **Action:** Capture an op-level device profile of one steady-state decode step (Tracy / `tt_metal` profiler; `TT_METAL_DEVICE_PROFILER=1`). Attribute time to: weight matmuls, `ttnn.all_reduce` (CCL), per-head `rms_norm` + `to_memory_config` bounces, lm_head, and the host gap between `execute_trace` calls.
- **Reproduce baseline:** the `#1 Plain greedy decode (long context)` command in this doc (`long-context-64k`, `enable_trace=True`) → grep `Decode:` for ms/token.
- **Deliverable:** a per-op ms breakdown appended here as a table. Use it to confirm the ordering of Phases 1–4 for the observed hardware.
- **Expected impact:** none directly; de-risks all later phases.

#### Phase 0 results (2026-07-06, QB2)

**Baseline tok/s — REPRODUCED ✓**

| Run | Prompt tokens | Generated | TTFT | Decode | Log |
|-----|---------------|-----------|------|--------|-----|
| Reference (2026-07-02) | 30720 | 166 | 12537 ms | **51.98 ms/tok @ 19.24 tok/s** | `plain_longwin.log` |
| Phase 0 (2026-07-06) | 32568 | 122 | 12922 ms | **52.26 ms/tok @ 19.13 tok/s** | `plain_longwin_phase0.log` |

Within ~0.6% of reference — baseline confirmed. Safe to proceed to Phase 1.

**Short-context sanity (profile run, batch-1):** 87 prompt tokens, 16 generated → **49.88 ms/tok @ 20.05 tok/s** (`phase0_profile.log`). Consistent with long-context (~52 ms).

**Device op profile — attempt 1 FAILED (overflow), clean procedure below**

First Tracy attempt (`batch-1`, 16 decode tokens) passed pytest but post-processing failed:

```
AssertionError: Device data missing: Op 1096705 not present in cpp_device_perf_report.csv
```

Root cause: **profiler DRAM buffer overflow** — hundreds of `Profiler DRAM buffers were full, markers were dropped!` at teardown (60 layers × 16 tokens + prefill ≫ 12000 markers/core). The salvaged device-only CSV had timing rows but **no OP labels**, so bucketing was impossible.

**Fix (implemented 2026-07-06): signpost + buffer-flush hook in `text_demo_v2.py`.** The decode loop now (when `GEMMA4_PROFILE_DECODE=1`) drains the profiler buffer (`ttnn.ReadDeviceProfiler`), emits a tracy `signpost("start")`, runs **one** steady-state decode step, emits `signpost("stop")`, and breaks. Only that single token's device ops accumulate → no overflow, and the region is cleanly delimited in the CSV (rows with `OP TYPE == "signpost"`, `OP CODE == start/stop`). Mirrors the proven `tests/unit/tracy_prefill_common.py` workflow.

Layer pattern is **5 sliding : 1 full, repeating** (`config.json:37-97`), so the single captured step already contains a representative mix; per-layer op costs scale ×60.

**Clean profile — run this:**

```bash
source ~/tt-metal/python_env/bin/activate
cd ~/tt-metal
export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal
export ARCH_NAME=blackhole MESH_DEVICE=P150x4
export HF_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main
unset TT_CACHE_PATH

# Short prompt; the hook stops after the first steady-state decode step anyway.
export GEMMA4_MAX_SEQ_LEN=1024
export GEMMA4_MAX_NEW_TOKENS=8
export GEMMA4_PROFILE_DECODE=1          # enable signpost/flush hook
export GEMMA4_PROFILE_DECODE_ITER=1     # capture first post-compile step
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000

python -m tracy -p -r -v -n phase0_decode_clean -m \
  pytest models/demos/gemma4/demo/text_demo_v2.py \
  -k "batch-1" -sv 2>&1 | tee phase0_profile_clean.log
```

Verify no drops + report generated:

```bash
grep -c "buffers were full" phase0_profile_clean.log   # want 0
ls -t generated/profiler/reports/*/ops_perf_results*.csv | head -1
```

**Bucket device time between the signposts:**

```bash
CSV=$(ls -t generated/profiler/reports/*/ops_perf_results*.csv | head -1)
CSV="$CSV" python3 - <<'PY'
import os, pandas as pd
df = pd.read_csv(os.environ["CSV"])
name_col = next(c for c in ["OP CODE","OP TYPE","OP NAME"] if c in df.columns)
# Slice rows between the start/stop signposts (decode-only region)
sp = df.index[(df.get("OP TYPE")=="signpost")].tolist()
if len(sp) >= 2:
    df = df.loc[sp[0]+1:sp[1]-1]
dur = pd.to_numeric(df["DEVICE FW DURATION [ns]"], errors="coerce").fillna(0)
df = df.assign(_d=dur)
def bucket(op):
    o=str(op).lower()
    if "reduce" in o and "scatter" not in o: return "CCL all_reduce"
    if "all_gather" in o or "allgather" in o: return "CCL all_gather"
    if "scaled_dot_product" in o or "sdpa" in o: return "SDPA"
    if "update_cache" in o: return "KV update"
    if "matmul" in o or "linear" in o: return "Matmul (weights)"
    if "rms" in o or "norm" in o: return "RMSNorm"
    if "embedding" in o or "rotary" in o: return "Embed/RoPE"
    if "concat_heads" in o or "to_memory" in o or "reshard" in o or "interleaved" in o: return "Layout/mem"
    if "sampling" in o or "topk" in o or "argmax" in o: return "Sampling"
    return f"Other:{op}"
df["b"]=df[name_col].map(bucket)
g=df.groupby("b")["_d"].agg(["sum","count"]).sort_values("sum",ascending=False)
tot=g["sum"].sum()
print(f"name_col={name_col}  region ops={len(df)}  total device FW={tot/1e6:.3f} ms\n")
for b,r in g.iterrows():
    print(f"  {b:20s} {r['sum']/1e6:7.3f} ms  ({100*r['sum']/tot:5.1f}%)  n={int(r['count'])}")
PY
```

**Per-op breakdown — CLEAN CAPTURE ✓ (2026-07-06).** One steady-state decode step, summed device FW across all 4 TP devices (op counts confirm scope: Matmul n=1204 ≈ 301/device = 5 matmuls × 60 layers + lm_head; CCL n=480 = 2 all-reduce/layer × 60 × 4).

| Bucket | Device FW (sum, 4 dev) | Share | n (ops) | Notes |
|--------|------------------------|-------|---------|-------|
| **Matmul (weights)** | **152.4 ms** | **46.4%** | 1204 | qkv/o_proj/gate/up/down + lm_head — weight-BW bound |
| Elementwise binary (`BinaryNg`) | 39.4 ms | 12.0% | 960 | residual adds, gate×up, RoPE mul (~4/layer) |
| Elementwise unary | 34.5 ms | 10.5% | 240 | ~1/layer (layer_scalar mul / activation) |
| Copy (`CopyDeviceOperation`) | 26.9 ms | 8.2% | 240 | ~1/layer — memory bounce |
| CCL all_gather | 16.4 ms | 5.0% | 480 | all_reduce decomposes → RS + AG |
| Layout/mem (`to_memory`/reshard) | 16.4 ms | 5.0% | 3608 | many tiny resharding ops |
| CCL reduce_scatter | 12.3 ms | 3.8% | 480 | — |
| RMSNorm | 12.0 ms | 3.6% | 1684 | per-head + layer norms |
| Embed/RoPE | 6.6 ms | 2.0% | 960 | — |
| SDPA | 5.7 ms | 1.7% | 240 | decode attention |
| KV update | 3.0 ms | 0.9% | 480 | `paged_update_cache` |
| QKV-heads/slice/concat | ~3.4 ms | ~1.0% | 720 | — |
| **Total** | **328.8 ms** | 100% | 11300 | sum across 4 dev (wall ≈ 52 ms; overlap ~1.6×) |

**Interpretation (drives phase priority):**
1. **Matmul = 46% → weight bandwidth is the #1 target, confirmed.** → **Phase 2a (DRAM-sharded matmuls, lossless)** then optional **Phase 2c (bfp4)**. This is the primary path.
2. **CCL is only ~8.8% (AG 5.0% + RS 3.8%)** — much smaller than hypothesized. **Phase 3 (async CCL) is DEPRIORITIZED** — cap the possible win at <9%.
3. **Elementwise + Copy ≈ 30.7%** (binary 12% + unary 10.5% + copy 8.2%) — **surprisingly large, and a new opportunity.** These are residual adds, gate×up mul, layer_scalar, and per-layer memory copies. Candidates: fuse into matmul epilogues / eliminate the attention DRAM bounces (**Phase 4**), and drill into exact OP CODEs + shapes. Potentially a bigger, lower-risk win than bfp4.
4. Layout/mem 5% across **3608 tiny ops** — dispatch/resharding churn; partially Phase 4.

**Caveat:** numbers are **summed op FW across 4 devices**, not critical path — good for *relative attribution / prioritization*, not wall-clock (wall ≈ 52 ms). Ratios are the actionable signal.

**Recommended order (updated by this data):** Phase 2a → Phase 2c (optional bfp4) → **investigate elementwise/copy ~31%** (Phase 4 + fusion) → Phase 2b (prefetcher). Phase 3 (CCL) demoted.

**Follow-up drill-down (optional):** re-run the bucket script but group by raw `OP CODE` (no bucketing) and add mean shape, to see which specific binary/unary/copy ops make up the 31% — decides what's fusable.

**Notes / fallbacks:**
- The `-p` flag (partial zones) keeps the capture to signposted regions. If post-processing still hits the missing-op assert, re-run and salvage with `python tools/tracy/process_ops_logs.py --device-only --force-legacy-device-logs -n phase0_decode_clean --date`.
- lm_head + embedding + final norm run **once per token** (not per layer) — they appear once in the captured step; don't scale them ×60.

### Phase 1 — Eliminate the per-token host sync (on-device greedy sampling)

**Status:** IMPLEMENTED in `serve_fastapi.py`, but **NOT wired in `demo/text_demo_v2.py`** and **NOT a perf win on Gemma4 QB2** (2026-07-06). Default remains **host sampling** (`GEMMA4_SAMPLE_ON_DEVICE=0`).

**Context / why:** `demo/text_demo_v2.py` decode loop calls `generator.decode_forward(..., sampling_params=None)` then `_host_sample(logits, ...)` (argmax on CPU). That forces (a) the lm-head TP all-gather of the full 262k vocab and (b) a device→host→device barrier every token. The machinery to avoid it already exists:
- `Gemma4Model.sampling` on-device sampler (`tt/model.py`; `_supports_on_device_sampling = True`).
- `ttnn_decode_forward(..., on_device_logits=True)` returns sharded logits and skips the gather when `_apply_lm_head` skips the all-gather for on-device sampling.
- Env flag `GEMMA4_SAMPLE_ON_DEVICE` is read by `serve_fastapi.py` (it does not affect `text_demo_v2.py`).

**Actions:**
1. Add a greedy on-device path to the plain decode loop in `demo/text_demo_v2.py`: when `temperature == 0`, request device sampling (pass `sampling_params` for greedy/argmax and/or `on_device_logits=True`) so the next token is produced on device and never round-trips to CPU.
2. Ensure the sampler is constructed for the plain demo run (set `GEMMA4_SAMPLE_ON_DEVICE=1`; confirm `self.sampling` is non-None before use).
3. Keep the traced step (`enable_trace=True`) so consecutive tokens run back-to-back with no host op between them.

**Validation:**
- Token-identical vs current host-argmax greedy on a fixed prompt (temperature=0): compare generated token IDs.
- Re-run `#1 Plain greedy decode` and record new `Decode:` ms/token.

**Expected impact:** removes one full-vocab all-gather + one host sync per token — likely the largest single non-spec win. **Risk:** low (greedy argmax is exact).

**What changed:** `serve_fastapi.py` now reads `GEMMA4_SAMPLE_ON_DEVICE` (default **0**) to enable/disable on-device sampling. `text_demo_v2.py` does **not** read this flag; it always uses host sampling.

**Validate Phase 1 (run both, compare `Decode:` line):**

```bash
source ~/tt-metal/python_env/bin/activate
cd ~/tt-metal
export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal
export ARCH_NAME=blackhole MESH_DEVICE=P150x4
export HF_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main
export GEMMA4_MAX_SEQ_LEN=32768
unset TT_CACHE_PATH

# Host baseline (Phase 0 reference path) — `text_demo_v2.py` always uses host sampling
export GEMMA4_SAMPLE_ON_DEVICE=0
pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-64k" -sv \
  2>&1 | tee phase1_host_sample.log
grep -E "Sampling path:|Decode:" phase1_host_sample.log | tail -3
```

The run above tests the host-sampling baseline in `text_demo_v2.py`. The server section describes how to exercise on-device sampling.

**Phase 1 A/B results (2026-07-06, long-context-64k, `GEMMA4_MAX_SEQ_LEN=32768`):**

| Path | Sampling log | Generated | Decode | vs Phase 0 baseline |
|------|--------------|-----------|--------|---------------------|
| Host (`SAMPLE_ON_DEVICE=0`) | `host (full logits readback)` | 2 tok (early `<turn|>`) | **52.41 ms/tok @ 19.08 tok/s** | ~matches Phase 0 (19.13) |
| Device (`SAMPLE_ON_DEVICE=1`) | `on-device greedy` | 2 tok | **61.22 ms/tok @ 16.34 tok/s** | **−17% regression** |

**Why device sampling lost:** Gemma4 sets `_tt_vllm_always_refresh_decode_trace_inputs = True` (`tt/model.py:150`). The traced decode path therefore **still `copy_host_to_device`s token + position every iteration** (see `models/tt_transformers/tt/generator.py`, where the always-refresh flag is checked) — no elimination of the host sync. On top of that, `sample_decode_on_device` runs a **second** post-forward sampling pass (top-k + TP gather on sharded logits, separate sampling trace). Net: extra work, not less. `serve_fastapi.py` already documented this: *"host is faster for batch-1 greedy on Gemma4 31B @ 1×4"*.

**Caveat:** both A/B runs stopped after **2 generated tokens** (hit `<turn|>`), so each reports only **one** steady-state decode iteration — noisier than Phase 0's 122-token average. Re-run with `GEMMA4_MAX_NEW_TOKENS=200` if you want a longer steady-state confirm; direction is unlikely to flip.

**Real Phase 1 follow-up (if revisiting):** disable always-refresh and wire on-device token feedback into the decode trace (see `_decode_token_feedback_buffer`) so consecutive tokens need no host copy — higher risk, needs token-identical validation.

**Next:** skip Phase 1 as a throughput lever → **Phase 2a** (lossless DRAM-sharded weight reads) then **Phase 2c** (optional bfp4 MLP).

### Phase 2a — DRAM-sharded matmuls (lossless weight-bandwidth path) — DO BEFORE optional bfp4

**Motivation (from 2026-07-06 review):** before reducing precision, close the gap to peak DRAM bandwidth. Single-user decode is memory-bound at arithmetic intensity ~1 (each weight byte used once/token), and pinning weights in SRAM is **impossible** here: 31B is **dense** (every one of 60 layers runs every token — no hot/cold skew like MoE), and per-device weights (~7.75 GB bfp8) are ~40× the total on-chip L1 (~200 MB/device). So the only levers are (i) fewer bytes (bfp4, Phase 2c) or (ii) read the same bytes closer to peak BW + overlap them — this phase.

**Gap evidence:** ideal weight-read floor ≈ 7.75 GB ÷ (hundreds of GB/s GDDR6) ≈ ~15-20 ms/token vs measured ~52 ms. The excess is CCL + host + **sub-peak BW utilization**. Gemma4 currently reads weights **DRAM-interleaved** via plain `ttnn.linear` with **no program config** (`tt/shared_mlp.py:67-94`, `ttnn.linear` at `:106-117`) and has **no prefetcher** (`tt/model.py:1248-1249` "no prefetcher to reinitialize").

**Reference implementations to port from (already in-repo, llama-class):**
- DRAM-sharded matmul program configs: `models/tt_transformers/tt/mlp.py`, `lm_head.py`, `attention.py` (grep `dram_sharded` / `DRAM_SHARDED`).
- Program-config builders: `models/tt_transformers/tt/model_config.py` (`get_mlp_ff1_3_prg_config`, `get_mlp_ff2_prg_config`, `get_mlp_ff1_3_mem_config`).

**Actions:**
1. Store `gate/up/down` (and `wqkv`, `o_proj`, `lm_head`) as **DRAM-sharded** weights and pass the matching DRAM-sharded matmul program config to `ttnn.linear` in `shared_mlp.py` / `attention/operations.py`.
2. A/B against the current interleaved path; keep an env flag to fall back.

**Validation:** `test_full_model_decode` PCC **unchanged** (lossless); re-run `#1 Plain greedy decode`; confirm `Decode:` ms/token drops.

**Expected impact:** medium, **zero quality risk** — gated by how much of the 52 ms is weight-read (Phase 0 profile decides). **Risk:** medium — per-matmul program configs, shard-spec correctness, trace interaction.

#### Implemented 2026-07-06 — SharedMLP DRAM-sharded decode path (opt-in) ✓

Landed in `tt/shared_mlp.py`, gated by **`GEMMA4_DRAM_SHARDED_MLP=1`** (default off → unchanged interleaved path). Started with the MLP because it is ~70% of per-layer weight bytes and every per-device matmul is a clean **5376×5376** (`hidden=5376`, `intermediate/tp = 21504/4 = 5376`), which shards perfectly.

**Weight-storage design (revised 2026-07-06 after two prefill failures):** the weights stay **DRAM-interleaved** (prefill path = untouched baseline) and the flag additionally builds **separate persistent DRAM-width-sharded copies** (`ttnn.to_memory_config` at init) that only the decode path reads. This decouples the two modes at the cost of **~2× MLP weight DRAM while enabled** (~+5.5 GB/device on 31B). The copy allocation is wrapped in try/except → an OOM logs a warning and falls back to interleaved decode (never crashes). *Rationale:* a single DRAM-sharded copy shared with prefill needs a correct per-seq-len prefill program config; without one the prefill matmul either rejects the fused-activation kwarg or asserts "Only L1 buffers can have an associated circular buffer" (see failure history below). Single-copy + prefill program config is the eventual production form; two-copy is the low-risk path to *measure the decode win* first.

What the flag turns on (decode only, `seq_len ≤ 32`):
- **Decode weights** are the DRAM-width-sharded copies across all DRAM banks (`dram_grid_size().x`; 8 on P150 → `[5376, 672]`/bank).
- **Matmuls** use `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`. `find_grid_k_n(168,168) → 56 compute cores (7×8)`; `in0_block_w=3`, `per_core_M=1`, `per_core_N=3`. GELU is fused into the gate epilogue via `UnaryWithParam(GELU, 1.0)` (matches `gelu_approx`).
- **Activation dataflow:** `normed` is resharded once to **L1 width-sharded** `(32, 96)×56`; gate/up outputs stay L1 width-sharded and feed the down matmul (explicitly re-pinned to the down input layout for safety); down output → `sharded_to_interleaved(DRAM)` before the TP all-reduce. So the only added ops are 1 input reshard + 1 output interleave/token — the gate·up→down hop stays in L1.
- **Safety:** all shard-divisibility is asserted in `_setup_dram_sharded`; any failure logs a warning and **falls back to the interleaved path**, so enabling the flag can never break a run — worst case it's a no-op.
- **Prefill failure history (why the design is two-copy):**
  1. *Attempt 1 — weights DRAM-sharded for all modes:* prefill `ttnn.linear(..., activation="gelu_approx")` became a sharded matmul → `matmul.cpp:175 TT_FATAL !user_fused_activation` (a sharded matmul needs the activation in the program config, not the kwarg). `plain_dramshard.log:803-1208`.
  2. *Attempt 2 — same, but GELU fused into the `mul` instead:* got past the activation assert, then the plain 2D prefill matmul tried to open a circular buffer on the DRAM-sharded weight → `assert.hpp:104 "Only L1 buffers can have an associated circular buffer"`. `plain_dramshard.log:799-1139`.
  3. *Resolution:* stop DRAM-sharding the prefill weights entirely — keep interleaved weights for prefill (baseline) and give decode its own DRAM-sharded copies (above). Prefill and the flag-off path are byte-for-byte the validated baseline.

**Measured result (2026-07-06, long-context-64k, plain greedy, host sampling) ✓:**

| Config | ms/token | tok/s | Δ vs baseline | source |
|---|---|---|---|---|
| Interleaved baseline | 51.98 | 19.24 | — | `plain_interleaved.log:796` |
| MLP only (`GEMMA4_DRAM_SHARDED_MLP=1`) | 48.66 | 20.55 | **+6.8%** | `plain_dramshard.log:856` |
| **MLP + ATTN** (`GEMMA4_DRAM_SHARDED=1`, lm_head auto-skipped) | **47.50** | **21.05** | **+9.4%** | `ds_all.log:857` |

Attention adds ~**1.2 ms/token** on top of the MLP win (~2.4% more). lm_head correctly auto-fell back to interleaved (`DramShardedMatmul[lm_head] disabled … L1 CBs 8417280 B > budget`, `ds_all.log:68`). Lossless (same weights, different DRAM layout for decode matmuls only).

**Extended to attention + lm_head (2026-07-06):** the same recipe is now packaged in `tt/dram_sharded.py` (`DramShardedMatmul`, `try_build`) and applied to `wqkv` + `o_proj` (`tt/attention/{weights,operations}.py`) and the tied `lm_head` (`tt/model.py`). Independent opt-in flags so each can be A/B'd or disabled:

| Flag | Covers | Notes |
|---|---|---|
| `GEMMA4_DRAM_SHARDED_MLP` | gate/up/down | validated above (+6.8%) |
| `GEMMA4_DRAM_SHARDED_ATTN` | wqkv, o_proj | per-layer sliding/global dims; all shard to 8 cores |
| `GEMMA4_DRAM_SHARDED_LMHEAD` | lm_head | fat tied weight; big per-core N (watch L1) |
| `GEMMA4_DRAM_SHARDED` | master — turns on all three | |

All follow the two-copy pattern (interleaved weight kept for prefill; separate DRAM-sharded copy for decode; `try_build` → `None` on unshardable dims / OOM / L1-overflow → interleaved fallback). Decode-only (guarded by `seq_len ≤ 32`); prefill and the flag-off path are byte-for-byte baseline.

**lm_head does NOT fit this matmul (confirmed 2026-07-06).** The DRAM-sharded matmul infers its compute-core count from `gcd(k_tiles, n_tiles)` (in0 must shard evenly over k). For lm_head `k=5376 (168 tiles)`, `n=vocab/tp=65536 (2048 tiles)`, `gcd=8` → only 8 cores → `per_core_N=256` → the weight circular buffer alone is ~11.7 MB vs 1.5 MB L1 (`ds_all.log:1204-1206`). `DramShardedMatmul.__init__` now estimates the CB footprint and raises when it exceeds `_L1_CB_BUDGET_BYTES` (1.3 MB), so `try_build` returns `None` and lm_head **auto-falls back to interleaved** — the master flag is safe. To actually accelerate lm_head we'd need a different program config (1D systolic / width-sharded matmul with more cores + explicit n-padding handling); deferred. **MLP + ATTN are the shipped Phase 2a wins.**

**A/B to validate (run both, compare `Decode:` tok/s):**
```bash
# quality gate first — PCC must be identical to interleaved (lossless change)
GEMMA4_DRAM_SHARDED_MLP=1 \
  pytest models/demos/gemma4/tests/unit/test_model.py::test_full_model_decode -k "1x4" -sv

# baseline (interleaved) vs DRAM-sharded, same long-context decode benchmark
export GEMMA4_MAX_SEQ_LEN=32768
pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-64k" -sv 2>&1 | tee plain_interleaved.log
GEMMA4_DRAM_SHARDED_MLP=1 \
  pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-64k" -sv 2>&1 | tee plain_dramshard.log
grep "Decode:" plain_interleaved.log | tail -1
grep "Decode:" plain_dramshard.log | tail -1
```
Optional: re-run the clean single-step profile (`GEMMA4_PROFILE_DECODE=1`) with the flag on and confirm the Matmul bucket shrinks.

**PCC gate PASSED (2026-07-06):** `GEMMA4_DRAM_SHARDED=1 … test_full_model_decode -k 1x4` passed with matching HF/TT argmax → confirmed lossless. Phase 2a (MLP+ATTN) is validated end-to-end.

### Phase 2b — Weight prefetcher (overlap DRAM reads)

**Context:** `tt_transformers` has a `Prefetcher` (`tt/prefetcher.py`, wired through `model_config.py` ring/global-CB configs) that streams next-layer weights through a global circular buffer, hiding DRAM read latency behind compute/CCL. Gemma4 does not use it. Larger lift than 2a; do only if 2a + profile show weight-read latency is still exposed (not overlapped).

**Expected impact:** medium; **Risk:** high (prefetcher sub-device lifecycle + trace/CCL interaction).

### Phase 2c — Weight precision: bfp8 baseline, optional bfp4 MLP

**Context / why:** the shared MLP (`gate` + `up` + `down`, `hidden=5376`) is ~70% of per-layer weight bytes and dominates the bandwidth floor; the tied `lm_head`/embedding (`262144×5376` ≈ 1.4B params) is the second-fattest tensor. Precision is controlled by `precision_overrides.json` + `tt/precision.py` (per-module dtype; `model_key_from_path`). Currently `attention` + `shared_mlp` = **bfp8** for `gemma-4-31B-it`.

**Composes with Phase 2a (orthogonal):** bfp4 changes the *byte count* per weight; DRAM-sharding changes the *DRAM layout*. A bfp4 weight is still DRAM-sharded, and `DramShardedMatmul`'s L1-budget estimator already accounts for the smaller bfp4 tile. Cache filenames embed the dtype (`_bfp4` vs `_bfp8`), so the two precisions coexist on disk and never reuse a stale cache. So bfp4 stacks on top of the +9.4% already banked.

**Runtime bfp4 toggle — keeps bfp8 as the checked-in default.** `Gemma4Precision.load` now applies env overrides on top of `precision_overrides.json`, so you can A/B bfp4 without editing the JSON:

| Env var | Effect |
|---|---|
| `GEMMA4_PRECISION_SHARED_MLP=bfp4` | shared MLP weights → bfp4 |
| `GEMMA4_PRECISION_LM_HEAD=bfp4` | lm_head → bfp4 |
| `GEMMA4_PRECISION_ATTENTION=bfp4` | attention (qkv/o_proj) → bfp4 |
| `GEMMA4_PRECISION_ALL=bfp4` | every known module → bfp4 (per-module var still wins) |

Values: `bf16` / `bfp8` / `bfp4` / `fp32`. A typo raises loudly. This is the "keep DRAM-sharded + add a 4-bit version" path — flip the env var, both weight caches build side by side.

**Shipped default (2026-07-06):** `precision_overrides.json` for `gemma-4-31B-it` keeps **shared_mlp and attention at bfp8**. A split bfp4 stack (`gate`/`up` bfp4, `down_proj` bfp8) had yielded acceptable PCC (~0.974), but real FastAPI generation with any bfp4 default produced catastrophic garbage (numbers → `//`/`****`, LaTeX explosion, wrong facts). The 0.974 PCC (below the 0.99 gate) was the early warning — bfp4's ~+18% decode gain is **not** worth the correctness loss. The bfp4 path is preserved as an opt-in env override (`GEMMA4_PRECISION_SHARED_MLP=bfp4`) for experiments only.

**Measured stack (long-context-64k, plain greedy):**

| Stack | ms/token | tok/s | PCC |
|---|---|---|---|
| Baseline interleaved bfp8 | 51.98 | 19.24 | pass |
| + Phase 2a (`GEMMA4_DRAM_SHARDED=1`) | 47.50 | 21.05 | pass |
| + split bfp4 (`gate`/`up` bfp4, `down` bfp8) + Phase 2a | **44.04** | **22.71** | ~0.974 |

Phase 2a remains **opt-in** via `GEMMA4_DRAM_SHARDED=1` (or per-module flags). Recommended production command for max throughput:

```bash
export GEMMA4_DRAM_SHARDED=1 GEMMA4_MAX_SEQ_LEN=32768
pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-64k" -sv
```

The optional Phase 2c bfp4 stack is **not** recommended as a default; use `GEMMA4_PRECISION_SHARED_MLP=bfp4` only for controlled experiments after gating on PCC and generation sanity.

**Result — full-MLP bfp4 FAILS the PCC gate (2026-07-06):** `GEMMA4_PRECISION_SHARED_MLP=bfp4 GEMMA4_DRAM_SHARDED=1 … test_full_model_decode` → **PCC 0.9398 vs 0.99 threshold** (argmax happened to match, `HF=563 TT=563 ' is'`, but 0.94 across 60 layers will drift on longer gens). So bfp4 on all three MLP weights is too lossy. Root cause matches the reference note (`tt_transformers/tt/mlp.py:99` "bfp4 … sub .99 pcc"): the `down_proj` row-parallel reduction is the precision-sensitive one.

**Split-precision (opt-in):** `shared_mlp_down` overrides only the `down_proj` dtype (defaults to `shared_mlp`). Split bfp4 gate/up + bfp8 down → **PCC 0.974**, **44.04 ms/token @ 22.71 tok/s** with Phase 2a.

**Validation (quality gate — REQUIRED before using):**
- `pytest models/demos/gemma4/tests/unit/test_model.py::test_full_model_decode -k "1x4" -sv` → PCC must stay within tolerance vs bf16 reference.
- Short generation sanity: server curl A (merge_sort) + curl D (capital of France) must stay coherent/correct.
- Re-run `#1 Plain greedy decode`; record ms/token.

**Expected impact:** roughly halves the largest bandwidth term (~5–8 ms/token plausible for MLP bfp4). **Risk:** medium — quality regression; gate strictly on PCC + sanity gen. Roll back per-tensor if PCC drops.

### Phase 3 — Reduce TP collective cost

**Context / why:** `tt/ccl.py` uses simple `ttnn.all_reduce` for both per-layer reductions (attn o_proj + MLP down) via `ccl_allreduce` (`tt/ccl.py:75-91`, called from `tt/attention/operations.py:447-449` and the shared MLP). The experimental `reduce_scatter_minimal_async` + `all_gather_async` path is already written but commented out (`tt/ccl.py:93-120`, `142-156`), and `CCLManager` already allocates the required semaphores (`tt/ccl.py:43-54`). `num_links=2` is already set for blackhole (`tt/ccl.py:8-17`).

**Actions:**
1. Sweep the experimental async `reduce_scatter_minimal_async` + `all_gather_async` decode path (uncomment + wire the semaphore getters `get_rs_semaphore`/`get_ag_semaphore`/`get_barrier_semaphore`). Compare against `ttnn.all_reduce`.
2. Where layout allows, keep activations **reduce-scatter-sharded** out of `o_proj`/`down_proj` and defer/avoid the following all-gather so each reduction is one collective instead of a full all-reduce.
3. Confirm Phase 1 already removed the lm_head all-gather (`tt/model.py:794`); if so, only the embedding all-gather (`tt/model.py:810-814`) and per-layer reductions remain.

**Validation:** `test_full_model_decode` PCC unchanged; re-run `#1 Plain greedy decode`. Watch for CCL deadlock/hang (run `tt-smi -r` between attempts as per this doc's ops notes).

**Expected impact:** medium — 120 latency-bound collectives/token; halving count/latency is material. **Risk:** medium — async CCL can deadlock; test incrementally, keep the simple `all_reduce` as a fallback flag.

### Phase 4 — Trim attention-decode memory bounces

**Context / why:** in `tt/attention/decode.py`, each token moves Q/K/V to DRAM for the per-head RMSNorm then back to the sharded layout (`decode.py:71-84`, and restore at `decode.py:133-135`) — 3 DRAM round-trips/layer × 60. Plus `split_qkv_heads_decode` copies the fused QKV DRAM→L1 as a blackhole `nlp_create_qkv_heads_decode` workaround for Metal issue #16667 (`tt/attention/operations.py:56-69`).

**Actions:**
1. **DONE (opt-in, `GEMMA4_ATTN_L1_NORM=1`; independent of `GEMMA4_DRAM_SHARDED` so it can be A/B'd on top of the current default).** Stage the per-head RMSNorm through **L1** instead of DRAM. rms_norm still needs an interleaved (non-height-sharded) input, so we can't run it fully in place, but K/V no longer bounce out to DRAM and back — the reshard is now L1↔L1. This mirrors the proven `norm_reshard` "hack until RMSNorm supports height-sharded output" in `tt_transformers/tt/attention.py:273-281`. Implementation:
   - `operations.py:apply_per_head_norm` gained an optional `memory_config=` passed to `ttnn.rms_norm` (decode passes `L1_MEMORY_CONFIG`; prefill/None keeps the op default).
   - `decode.py` per-head-norm block: `kv_stage_mem`/`kv_out_mem` switch DRAM→L1 for **K and V** when the flag is on. K/V are still restored to the height-sharded config for the cache update.
   - **Q stays on the DRAM path (baseline).** First device run failed with `TT_FATAL … Q tensor buffer type must be DRAM when not sharded but got BufferType::L1` (`sdpa_decode_device_operation.cpp:93`): after norm+RoPE, Q feeds `sdpa_decode`, which asserts a non-sharded Q buffer is DRAM. K/V don't hit SDPA as interleaved tensors, so L1 is only safe for them.
   - Removes ~2 DRAM writes + 2 DRAM reads per layer × 60 layers/token (K + V; Q's round-trip is required by SDPA).
2. **Investigated — kept as-is.** The Metal issue #16667 DRAM→L1 copy in `split_qkv_heads_decode` (`operations.py:60-73`) is a **correctness workaround** for a Blackhole `nlp_create_qkv_heads_decode` alignment bug that zeros odd Q rows on DRAM input. It's already a no-op when the fused QKV is in L1, and only fires when the QKV projection returns DRAM (which the DRAM-sharded matmul path does). Dropping it risks silent wrong-token corruption on Blackhole and can't be verified without hardware confirmation the bug is fixed, so it stays. A cleaner future win is making the QKV matmul emit L1 directly (then this branch never fires).
3. RoPE for batch=1 already uses the cheap fused single-position `rotary_embedding` (`decode.py:100-104`) — leave it; the per-user elementwise path (`apply_rope_decode_peruser`) is only for batch>1, not the single-user target.

**Validation:** `test_full_model_decode` PCC (must stay ≥ prior baseline) + re-run `#1 Plain greedy decode`. Flag is opt-in so a straight A/B against the current default isolates the norm change.

**Expected impact:** medium — removes ~120 DRAM round-trips/token (K+V; Q must stay DRAM for SDPA). **Risk:** low — same numerics (interleaved rms_norm, just L1-resident for K/V); the #16667 workaround is untouched.

**RESULT (2026-07-06, device A/B on `long-context-64k`, both on top of `GEMMA4_DRAM_SHARDED=1`):**

| Config | ms/token | tok/s | PCC |
|--------|----------|-------|-----|
| default | 44.07 | 22.69 | — |
| `+ GEMMA4_ATTN_L1_NORM=1` | 43.98 | 22.74 | **0.9754** |

**No measurable win (+0.2%, within noise).** Same story as Phase 1: numerically clean but not worth shipping. PCC 0.9754 is *slightly better* than the opt-in split-bfp4 stack (0.9743), so the `test_full_model_decode` red is just the unchanged 0.99 gate that the bfp4 experiment already breaches — **not** a Phase-4 regression. **Decision: keep `GEMMA4_ATTN_L1_NORM` opt-in and OFF by default.** The per-head-norm copies are only a small slice of the ~31% "elementwise+copy" bucket; the bulk is the MLP `mul` / residual adds / the per-token host-sampling barrier. **Re-profile before further attention-memory work** — the profile has shifted a lot since baseline (matmul dropped ~9% via 2a and ~18% via bfp4).

### Phase 5 — Fully fused device-resident greedy decode loop

**Context / why:** the plain path already traces the model body (`decode_forward(..., enable_trace=True)`), but even after Phase 1 the loop is host-driven per token. The spec path already runs an entire step as one `ttnn.execute_trace` with persistent I/O buffers and on-device `argmax`+re-embed (`tt/spec_decode.py` — see `execute_trace` usages at `spec_decode.py:313, 413, 967`; and the "argmax + re-embed on device" note at `text_demo_v2.py:690`). This is doc next-round item #4.

**Actions:**
1. Mirror the spec fused-iteration for **plain greedy**: keep token → embed → 60 layers → norm → lm_head → argmax → re-embed entirely on device inside a single captured trace, so consecutive tokens need zero host ops (only a periodic host read to check stop tokens / stream output).
2. Reuse the persistent-buffer + `execute_trace` pattern; respect the fused-trace teardown contract (`release_fused_trace()`, no interleaving with prefill CCL trace) from [Known limitations](#known-limitations--do-not-use).

**Validation:** token-identical greedy vs Phase-1 output; re-run `#1 Plain greedy decode`.

**Expected impact:** removes the last per-token host latency; best done AFTER Phases 1–4. **Risk:** medium-high — trace lifecycle/deadlock (see spec-decode do-not-use notes).

### Suggested execution order & tracking

1. **Phase 0** (profile) → append breakdown table. **← DONE** (clean signpost capture; baseline tok/s reproduced).
2. ~~**Phase 1** (on-device sampling)~~ → **DONE, no win** (−17%); reverted to host default. Root cause documented above.
3. **Phase 2a** (DRAM-sharded matmuls) → **DONE** (`GEMMA4_DRAM_SHARDED=1`: +9.4% lossless; opt-in).
4. **Phase 2c** (split bfp4 MLP) → **OPT-IN ONLY** (2026-07-06). Default reverted to bfp8. Available via `GEMMA4_PRECISION_SHARED_MLP=bfp4`. Real generation showed catastrophic quality loss at the bfp4 default, so bfp8 remains the checked-in default.
5. **Phase 2b** (weight prefetcher) → deferred (high risk; do only if re-profile shows exposed read latency).
6. **Phase 4** (attention decode memory bounces) → **DONE, no win** (+0.2%, noise); `GEMMA4_ATTN_L1_NORM` kept opt-in/OFF. Q must stay DRAM for SDPA; K/V L1-staging alone doesn't move the needle. Re-profile before more attention-memory work.
7. **Phase 3** (async/scatter CCL ~8.8%) → after Phase 4 or parallel if CCL re-emerges in profile.
8. **Phase 5** (fully fused device loop) → after Phases 3–4.

**Phase 0 decided the order:** Matmul-weights dominated the captured decode step, so DRAM-sharded matmuls (Phase 2a) were prioritized over CCL work (Phase 3).

After each phase, record `Decode: … tok/s/user` from the `long-context-64k` plain run and note PCC status. **Fundamental ceiling reminder:** batch=1 cannot saturate the array; the highest-leverage single-user lever remains **speculative decode** (already ~52 tok/s) — these baseline phases raise both the non-spec floor and the spec target-forward.

---

## Files changed in this effort (reference)

- `models/demos/gemma4/demo/serve_fastapi.py`
- `models/demos/gemma4/tt/spec_decode.py`
- `models/demos/gemma4/tt/generator.py`
- `models/demos/gemma4/tt/attention/operations.py`
- `models/demos/gemma4/tt/precision.py`
- `models/demos/gemma4/precision_overrides.json`
- `models/demos/gemma4/tests/unit/test_spec_decode.py`

---

## Log artifacts (on QB2)

| Log | Content |
|-----|---------|
| `spec_k16.log` | K=16 demo, ~52 tok/s, 30K context |
| `gemma4_server.log` | Server startup + request traces |

---

*Generated after QB2 bring-up session: spec K=16 validation, FastAPI server, hang/quality fixes, and throughput characterization.*
