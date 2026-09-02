<!-- template-version: 4 -->

# Gemma-4

Gemma 4 is the next-generation open-weights model family from Google, extending the Gemma line with mixed sliding-window/global attention, partial RoPE on global layers, per-layer-input embeddings on the smaller variants, and a sparse mixture-of-experts block on the larger ones. This directory implements text-only inference for five checkpoints — E2B and E4B (dense, with per-layer-input embeddings), 12B, 26B-A4B, and 31B — running on TT-NN with tensor parallelism across Tenstorrent meshes.

## Variants

| Variant | HuggingFace card |
|---|---|
| E2B | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) |
| E4B | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
| 12B | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |
| 26B-A4B | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| 31B | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |

## Tested Configurations

Wormhole rows are from CI (cold-start TTFT, including one-time program compile). Blackhole rows are measured metal demos on this branch (batch-1, warm, ISL=4k) — e2e CI exercises the same SKUs (`bh_p150` / `bh_quietbox_2`) but does not yet publish the same perf table.

| Variant | System | Mesh | Tokens/s | Tokens/s/user | TTFT (ms) | Source |
|---|---|---|---:|---:|---:|---|
| E2B     | N150 | 1×1 | 12.24 | 12.24 | 38714.7 [^ttft] | [CI](https://github.com/tenstorrent/tt-metal/actions/runs/25099500256/job/73545762105) |
| E4B     | N150 | 1×1 |  7.95 |  7.95 | 36832.1 [^ttft] | [CI](https://github.com/tenstorrent/tt-metal/actions/runs/25099500256/job/73545762123) |
| 26B-A4B | T3K  | 1×8 | 11.68 | 11.68 | 64186.5 [^ttft] | [CI](https://github.com/tenstorrent/tt-metal/actions/runs/25099500256/job/73545762080) |
| 31B     | T3K  | 1×8 |  9.48 |  9.48 | 44772.1 [^ttft] | [CI](https://github.com/tenstorrent/tt-metal/actions/runs/25099500256/job/73545762120) |
| E2B     | P150 | 1×1 | 22.82 | 22.82 | 580 [^bh-perf] | measured |
| E4B     | P150 | 1×1 | 13.97 | 13.97 | 950 [^bh-perf] | measured |
| 12B     | P150 | 1×1 | 15.57 | 15.57 | 1190 [^bh-perf] | measured |
| 12B     | LoudBox | 1×8 | 47.59 | 47.59 | 380 [^bh-perf] | measured |
| 31B     | LoudBox | 1×8 | 31.69 | 31.69 | 660 [^bh-perf] | measured |
| 31B     | QB2 | 1×4 | 22.68 | 22.68 | 1060 [^bh-perf] | measured |

[^ttft]: CI TTFT is cold-start, including one-time device program compile.
[^bh-perf]: Steady-state metal demo decode (batch-1 @ ISL=4k) from `isl_sweep_logs/defaults_scoreboard` (LoudBox) and `isl_sweep_logs/full_matrix_20260722` (QB2 / single-chip P150 aliased as N150 in that sweep). Long-context decode is lower (e.g. ~26–27 tok/s for 31B / P150x8 @ 128k; ~13–15 tok/s for 12B / P150 @ 128k–256k bounded).

E4B on N300 (1×2) is currently disabled in CI due to reduced N300 runner availability — see the Code Support Matrix below. Galaxy e2e entries are temporarily disabled (fabric / ethernet bring-up hang).

## Code Support Matrix

What the *code* in this directory supports, independent of what CI exercises.

Legend: 🟢 fully supported · 🟡 supported with known issues / limitations · 🔴 not supported · — not applicable

| Variant | N150 (1×1) | N300 (1×2) | T3K (1×8) | P150 (1×1) | QB2 / P150x4 (1×4) | LoudBox / P150x8 (1×8) | Galaxy (4×8) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| E2B     | 🟢 | 🟢 | 🟢 | 🟢 [^gemma4-p150] | 🟢 [^gemma4-lc] | 🟢 [^gemma4-lc] | 🔴 [^galaxy] |
| E4B     | 🟢 | 🟡 [^e4b-n300] | 🟢 | 🟢 [^gemma4-p150] | 🟢 [^gemma4-lc] | 🟢 [^gemma4-lc] | 🔴 [^galaxy] |
| 12B     | 🟡 [^gemma4-12b-wh] | 🟢 [^gemma4-wh-lc] | 🟢 [^gemma4-wh-lc] | 🟢 [^gemma4-p150-12b] | 🟢 [^gemma4-lc] | 🟢 [^gemma4-lc] | 🔴 [^galaxy] |
| 26B-A4B | 🔴 | 🔴 [^gemma4-wh-nofit] | 🟡 [^gemma4-wh-lc] | 🔴 | 🟢 [^gemma4-lc] | 🟢 [^gemma4-lc] | 🔴 [^galaxy] |
| 31B     | 🔴 | 🔴 [^gemma4-wh-nofit] | 🟢 [^gemma4-wh-lc] | 🔴 | 🟢 [^gemma4-lc] | 🟢 [^gemma4-lc] | 🔴 [^galaxy] |

[^e4b-n300]: E4B on N300 is exercised by the test suite locally but the CI entry is commented out due to runner availability. See `tests/pipeline_reorg/models_{unit,e2e}_tests.yaml`.
[^gemma4-12b-wh]: 12B on Wormhole N150 is short-context only (does not fit long-context KV).
[^gemma4-p150]: Single Blackhole P150 — **E2B / E4B fully supported through HF 256k** with default multi-chunk prefill (4096), unbounded KV (`text_demo_v2.py -k long-context-{4k,…,256k}`). CI: `bh_p150` e2e for E2B/E4B.
[^gemma4-p150-12b]: Single Blackhole P150 — **12B through full ISL 256k** with auto **bounded sliding + chunked prefill (4096)** above 32k (unbounded KV OOMs). Measured PASS + coherent gen at 64k/128k/256k (`isl_sweep_logs/full_matrix`). CI short-demo e2e on `bh_p150`; long-context via `MESH_DEVICE=P150 … text_demo_v2.py -k long-context-256k` (reports TTFT + decode tok/s). Prefill at 256k is slow (~6 min TTFT).
[^gemma4-lc]: Long-context via `models/demos/gemma4/demo/text_demo_v2.py` (`-k long-context-{4k,32k,64k,128k,256k}`) — reports **TTFT and decode tok/s** (long-context `text_demo.py` only logs TTFT). Same `GEMMA4_LONG_CONTEXT_POLICY` in both demos. **Coherence target on QB2 + LoudBox for 12B and 31B: 4k–128k** (256k may allocate; 31B/26B quality not a target). QB2 — E2B/E4B unbounded through 256k; 12B/26B-A4B unbounded through 128k, bounded(+chunked) at 256k; 31B bounded from 64k (chunk=2048 at ≥128k). LoudBox / P150x8 — E2B/E4B/12B unbounded through 256k; **31B/26B-A4B auto-bounded at 128k** with chunk=2048. Defaults: `GEMMA4_HOST_SAMPLE=1`; do not set `GEMMA4_DEMO_SINGLE_CHUNK=1`.
[^galaxy]: Galaxy (4×8) is not wired for Gemma4 yet; BH Galaxy e2e is disabled pending fabric / ethernet bring-up.

[^gemma4-wh-lc]: Wormhole long context, measured batch-1 on a real T3K with `text_demo_v2.py` (`GEMMA4_LONG_CONTEXT_POLICY` T3K/N300 entries; bounded sliding + multi-chunk prefill **2048** above 8–32k — WH carries 12 GB per ASIC vs Blackhole P150's 32 GB). **12B**: N300 (1×2, 24 GB) through **128k** (4k 11.4 s / 12.0 tok/s → 128k 171 s / 11.7 tok/s; 256k OOMs), T3K (1×8) through the **full 256k** (4k 9.9 s / 24.0 tok/s → 256k 468 s / 14.3 tok/s). **31B**: T3K through the **full 256k** (4k 8.5 s / 15.7 tok/s, 32k 55 s / 14.8 tok/s, 128k 188 s / 12.9 tok/s, 256k 564 s / 10.1 tok/s). **26B-A4B**: T3K functional through 128k but marked 🟡 — MoE prefill on WH is far slower than the dense 31B (TTFT 449 s @32k and 1506 s @128k), so ~32k is the practical serving ceiling.

[^gemma4-wh-nofit]: 26B-A4B and 31B do not fit a single N300 (1×2, 24 GB) at any ISL — both OOM during weight/KV allocation even at 4k. Use T3K (1×8) on Wormhole.

The 26B-A4B and 31B variants are too large for single-device N150/P150 or N300. Prefer Blackhole QB2 (TP=4, 128 GB GDDR) or LoudBox (TP=8, 256 GB GDDR) for long-context serve; Wormhole T3K (TP=8, ~12 GB/ASIC) remains the supported WH multi-chip path — metal Direct reaches the full 256k ISL for 12B and 31B there with bounded sliding + chunked prefill=2048, though TTFT is long (~8–9 min at 256k). **12B is the only variant that fits a single N300** (1×2, 24 GB), and only through 128k. Server specs stay well below the Blackhole 256k / 32-user shape: `workflows/model_specs/dev/llm.yaml` serves WH at `max_concurrency: 1` (12B T3K 256k, 12B N300 128k, 31B T3K 128k, 26B-A4B T3K 32k). On single **P150**, run E2B / E4B / 12B through 256k (12B uses bounded+chunked automatically).

## Prerequisites

- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) with submodules.
- TT-Metalium / TT-NN installed: see [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
- Model-specific dependencies (tt-metal uses [`uv`](https://docs.astral.sh/uv/) for Python package management — install via `uv pip`, not plain `pip`):
  ```
  uv pip install -r models/demos/gemma4/requirements.txt
  ```
- HuggingFace cache populated with `HF_HOME` set, plus `HF_HUB_OFFLINE=1` to skip network access.

## How to Run

### E2B on N150 (1×1)

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-E2B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-E2B-it
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x1"
```

### E4B on N150 (1×1)

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-E4B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-E4B-it
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x1"
```

### E4B on N300 (1×2)

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-E4B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-E4B-it
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x2"
```

### 26B-A4B on T3K (1×8)

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-26B-A4B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-26B-A4B-it
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x8"
```

### 31B on T3K (1×8)

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-31B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-31B-it
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x8"
```

### P150 (1×1) — E2B / E4B / 12B (full ISL through 256k)

Use **`text_demo_v2.py`** for long-context / perf (logs TTFT **and** decode tok/s). `text_demo.py` remains the CI short-demo path.

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-12B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-12B-it \
       MESH_DEVICE=P150
# Short demo (CI path)
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x1" -s --timeout 1500
# Full HF context — 12B auto-bounds + chunk=4096 above 32k; E2B/E4B stay unbounded
pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-256k" -s --timeout 3600
```

E2B/E4B: unbounded multi-chunk through 256k. 12B: bounded sliding + chunked prefill through 256k (required for DRAM).

### 12B / 26B-A4B / 31B on QB2 (P150x4) or LoudBox (P150x8)

Set `MESH_DEVICE` to the board mesh. Prefer **`text_demo_v2.py`** for long-context (TTFT + decode tok/s). Short CI demo and batch-32 stay on `text_demo.py`. Long-context rows are **not** gated by `--max-prefill`:

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=google/gemma-4-31B-it \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-31B-it \
       MESH_DEVICE=P150x8   # or P150x4 / P300x2 for QB2

pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x8"
pytest models/demos/gemma4/demo/text_demo_v2.py -k "long-context-4k or long-context-32k or long-context-128k" -s --timeout 1800
pytest models/demos/gemma4/demo/text_demo.py::test_demo_batch_32 -k "prefill_128 and 1x8" -v
```

For 12B on QB2, use `HF_MODEL=google/gemma-4-12B-it` and `MESH_DEVICE=P150x4` (filter `-k "1x4"`).

Keep defaults for quality: `GEMMA4_HOST_SAMPLE=1` (default). Prefill chunk follows `GEMMA4_LONG_CONTEXT_POLICY` (usually 4096; **31B/26B-A4B @ ≥128k on QB2/LB → 2048 + bounded**). Avoid `GEMMA4_DEMO_SINGLE_CHUNK=1` on long ISL (known “la la / lapped” collapse). Optional overrides: `GEMMA4_BOUNDED_SLIDING`, `GEMMA4_GEN_PREFILL_CHUNK`, `GEMMA4_MAX_SEQ_LEN`, `GEMMA4_MAX_NEW_TOKENS`.

### Short / batch demos (`text_demo_v2.py` batch-1)

```bash
HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=P150x8 \
  pytest models/demos/gemma4/demo/text_demo_v2.py -k "batch-1" -sv
```

### Single-layer smoke test

```bash
HF_MODEL=<path-or-id> pytest models/demos/gemma4/demo/text_demo.py::test_demo_single_layer
```

### 31B prefill KV migration loopback on Blackhole Galaxy

This bring-up test exercises the common prefill runner's mixed global/sliding KV chunk
table through the real migration data mover. It prefills source slot 0 through the 64K
cache, loops the bytes back through endpoint 1 into slot 1 on the same galaxy, and
byte-compares every migrated chunk. It needs no golden KV; the checked-in token-only trace
is padded to the configured input length.

Prerequisites:

- One Blackhole Galaxy available as a single CP8/TP4 rank.
- Gemma 4 31B weights and TTNN weight cache accessible through `GEMMA4_MODEL_PATH`.
- A tt-llm-engine checkout with the migration endpoint and Python client built under
  `disaggregation/migration/build_RelWithDebInfo`.

Set paths once in all three terminals:

```bash
export TT_METAL_HOME=/path/to/tt-metal
export ENGINE=/path/to/tt-llm-engine
export HOST=$(hostname)
export GEMMA4_MODEL_PATH=/path/to/gemma-4-31b-it
export PREFILL_MIGRATION_CLIENT_DIR="$ENGINE/disaggregation/migration/build_RelWithDebInfo/python"
```

Start the endpoint first:

```bash
cd "$ENGINE/disaggregation/migration"
./launch_migration_endpoints.sh \
  --name_server_host "$HOST" --prefill_hosts "$HOST" --prefill_endpoint_id 1
```

Start the runner next:

```bash
cd "$TT_METAL_HOME"
./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  models/demos/gemma4/tt/runners/manifests/gemma4_binding_loopback_migration_1rank.yaml \
  "$HOST:1"
```

Wait for `[migration] WORKER_READY`, then run the prefill and migration driver:

```bash
cd "$TT_METAL_HOME"
./models/demos/common/prefill/runners/run_migration_driver.sh \
  models/demos/gemma4/tt/runners/manifests/gemma4_producer_loopback_migration.yaml
```

The test passes when the driver prints `verify bytes PASSED` and exits with status 0.
The driver then sends the shutdown sentinel so the runner exits cleanly.

This proves that the published 36-config address table resolves the mixed global/sliding
cache correctly and that the local DRAM-to-DRAM migration preserves its bytes. It does
not establish model accuracy against a golden cache or exercise cross-host transport.

## Details

- **Entry points:**
  - `models/demos/gemma4/demo/text_demo_v2.py` — **preferred for long-context / perf**: Generator path; reports TTFT **and** decode tok/s (batch-1 / batch-32 / long-context). Same `GEMMA4_LONG_CONTEXT_POLICY` as v1.
  - `models/demos/gemma4/demo/text_demo.py` — CI short demo + batched prefill (`test_demo_batch_prefill` / `test_demo_batch_32`). Long-context Generator path works but currently logs TTFT only.
- **Batch size:** batch 1 by default; batched tests support up to batch-32 (override with `GEMMA4_BATCH_DEMO_SIZE`). On P150x8, batched-prefill microbatch is capped at ≤4 to avoid hangs.
- **Sequence length:** short demo buckets are 128 / 4096; long-context tests cover 4k–256k with per-(model, device) bounded-sliding / chunked-prefill cutovers (`GEMMA4_LONG_CONTEXT_POLICY` in `tt/generator_trace.py`). On **P150**, E2B/E4B/12B support full ISL **256k** (12B via bounded+chunked). On QB2/LB, coherence target for 12B/31B is 4k–128k. Batched prefill uses a **128k** virtual-token ceiling (`GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN`) with chunking above that.
- **Prefill trace:** enabled for MoE models on ISL buckets up to 4096 when `padded_batch × kernel < 32k`. Above 4k ISL or at/above 32k batched virtual tokens, prefill trace is disabled automatically (no perf gain, OOM risk).
- **Architecture:**
  - Mixed attention pattern: `sliding_attention` and `full_attention` layers interleaved per `hf_config.layer_types`.
  - Partial RoPE (factor 0.25) on global layers, full RoPE on sliding-window layers.
  - Per-layer-input embeddings on E2B/E4B; disabled on the MoE variants.
  - Optional `K=V` tying on global layers and KV-sharing across layer groups.
  - Sparse MoE block on 26B-A4B and 31B; dense MLP on E2B/E4B.
- **Pre/post-processing:** tokenization via the upstream HF tokenizer on host; logit softcapping (`final_logit_softcapping=30.0`) applied on device.

## Notes

- Weight cache is created on first run under `$TT_CACHE_PATH/tensor_cache_<dtype>/`. Subsequent runs reuse it; finetuned weights or a new dtype need a fresh cache. The cache directory must be writable (CI mounts of `/mnt/MLPerf` are often read-only — set `mlperf-read-only=false` or point `TT_CACHE_PATH` at a writable path for first-use).
