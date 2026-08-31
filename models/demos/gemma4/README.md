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
| E2B     | P150 | 1×1 | 22.82 | 22.82 | 580 [^bh-perf] | measured |
| E4B     | P150 | 1×1 | 13.97 | 13.97 | 950 [^bh-perf] | measured |
| 12B     | P150 | 1×1 | 15.57 | 15.57 | 1190 [^bh-perf] | measured |
| 12B     | LoudBox | 1×8 | 47.59 | 47.59 | 380 [^bh-perf] | measured |
| 31B     | LoudBox | 1×8 | 31.69 | 31.69 | 660 [^bh-perf] | measured |
| 31B     | QB2 | 1×4 | 22.68 | 22.68 | 1060 [^bh-perf] | measured |

[^ttft]: CI TTFT is cold-start, including one-time device program compile.
[^bh-perf]: Steady-state metal demo decode (batch-1 @ ISL=4k), measured via `tests/e2e/test_isl_sweep.py` (QB2 / single-chip P150 aliased as N150 in that sweep). Long-context decode is lower (e.g. ~26–27 tok/s for 31B / P150x8 @ 128k; ~13–15 tok/s for 12B / P150 @ 128k–256k bounded).

E4B on N300 (1×2) is currently disabled in CI due to reduced N300 runner availability — see the Code Support Matrix below. Galaxy e2e entries are temporarily disabled (fabric / ethernet bring-up hang).

### Per-ISL performance on Wormhole T3K

The table below comes from **`models/demos/gemma4/tests/e2e/test_isl_sweep.py`** — one pytest row per ISL bucket (`batch-1`, `batch-8`, `batch-32`, `long-context-4k` … `long-context-256k`). Each row exercises the Generator path in `text_demo_v2.py` (`run_demo_text`) and logs TTFT plus steady-state decode tok/s.

Measurements are on **Wormhole B0 T3K / LoudBox, `MESH_DEVICE=T3K` or `1x8`, TP=8**, at branch defaults (on-device sampling, `enable_trace=True`, paged attention, greedy `temperature=0` / `top_p=0.08`). Use **warm** TTFT — traces and program cache primed by a preceding run — so rows stay comparable to each other and are *not* mixed with the cold-start CI TTFT in the table above.

**Run every bucket:**

```bash
export HF_MODEL=google/gemma-4-31B-it   # 12B: see "12B checkpoint layout" below — the hub id will not load
export MESH_DEVICE=T3K                  # or 1x8 on T3K
pytest models/demos/gemma4/tests/e2e/test_isl_sweep.py -s --timeout=3600 -k "not ci-1"
```

Run it twice for numbers comparable to the table below — the first pass primes traces and the program
cache, so only the second is warm.

**Run one row:**

```bash
export HF_MODEL=google/gemma-4-31B-it   # 12B: see "12B checkpoint layout" below — the hub id will not load
MESH_DEVICE=T3K pytest models/demos/gemma4/tests/e2e/test_isl_sweep.py -k "long-context-128k" -s --timeout 1800
```

Sweep results, 12B and 31B:

| ISL bucket (`-k`) | ISL | Batch | 12B TTFT (ms) | 12B ms/tok | 31B TTFT (ms) | 31B ms/tok |
|---|---:|:-:|---:|---:|---:|---:|
| `batch-1`           |   128 |  1 |     58.5 | 26.21 |     93.5 | 42.88 |
| `batch-8`           |   128 |  8 |   4706.8 | 34.21 |   6977.7 | 57.12 |
| `batch-32`          |   128 | 32 |  13452.1 | 45.37 |  13801.5 [^t3k-b32] | 71.82 |
| `long-context-4k`   |    4k |  1 |    841.6 | 27.62 |   2032.2 | 45.05 |
| `long-context-32k`  |   32k |  1 |   7229.8 | 30.48 |  15067.5 | 48.68 |
| `long-context-64k`  |   64k |  1 |  14750.2 | 33.66 |  32943.4 | 52.74 |
| `long-context-128k` |  128k |  1 |  28925.7 | 38.01 |  61584.4 | 58.15 |
| `long-context-256k` |  256k |  1 | 105341.0 | 55.33 | 241697.9 | 80.08 |

`tok/s/user` is `1000 / ms-per-tok`, and aggregate `tok/s` is that times batch — both omitted as arithmetic.

[^t3k-b32]: 31B `batch-32` needs the batch-32 L1 fix (host batched extract + auto chunk-span warmup) — without it the row fails on a circular-buffer clash rather than running slowly.


## Code Support Matrix

What the *code* in this directory supports, independent of what CI exercises.

Legend: 🟢 fully supported · 🟡 supported with known issues / limitations · 🔴 not supported · — not applicable

| Variant | N150 (1×1) | N300 (1×2) | T3K (1×8) | P150 (1×1) | QB2 / P150x4 (1×4) | LoudBox / P150x8 (1×8) | Galaxy (4×8) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| E2B     | 🟢 | 🟢 | 🟢 | 🟢 [^gemma4-p150] | 🟢 [^gemma4-lc] | 🟢 [^gemma4-lc] | 🔴 [^galaxy] |
| E4B     | 🟢 | 🟡 [^e4b-n300] | 🟢 | 🟢 [^gemma4-p150] | 🟢 [^gemma4-lc] | 🟢 [^gemma4-lc] | 🔴 [^galaxy] |
| 12B     | 🟡 [^gemma4-12b-wh] | 🟢 [^gemma4-wh-lc] | 🟢 [^gemma4-12b-t3k] [^gemma4-wh-lc] | 🟢 [^gemma4-p150-12b] | 🟢 [^gemma4-lc] | 🟢 [^gemma4-lc] | 🔴 [^galaxy] |
| 26B-A4B | 🔴 | 🔴 [^gemma4-wh-nofit] | 🟡 [^gemma4-wh-lc] | 🔴 | 🟢 [^gemma4-lc] | 🟢 [^gemma4-lc] | 🔴 [^galaxy] |
| 31B     | 🔴 | 🔴 [^gemma4-wh-nofit] | 🟢 [^gemma4-wh-lc] | 🔴 | 🟢 [^gemma4-lc] | 🟢 [^gemma4-lc] | 🔴 [^galaxy] |

[^e4b-n300]: E4B on N300 is exercised by the test suite locally but the CI entry is commented out due to runner availability. See `tests/pipeline_reorg/models_{unit,e2e}_tests.yaml`.
[^gemma4-12b-wh]: 12B on Wormhole N150 is short-context only (does not fit long-context KV).
[^gemma4-12b-t3k]: 12B on Wormhole T3K (1×8) runs the full demo and unit suite, and is gated in CI (`wh_llmbox` unit tier 1, `wh_llmbox_perf` e2e tier 2) including the teacher-forced token/logit e2e that both 12B and 31B now run. A handful of `pcc_thresholds` nodes sit below the 0.99 default — fewer than 31B's on the same mesh — because 12B was Blackhole-only until this branch and its WH numerics were never gated. The one place 12B is genuinely behind 31B is `test_full_model[wormhole_b0-1x8]` at 0.935 against 31B's 0.98. Note that `GEMMA4_ATTN_WEIGHT_DTYPE=bf16` lifts several of those single-node PCCs but measures **worse** accumulated end to end, so do not read it as the fix (see `GEMMA4_PRECISION_OVERRIDE`). Per-ISL perf: see [Per-ISL performance on Wormhole T3K](#per-isl-performance-on-wormhole-t3k).
[^gemma4-p150]: Single Blackhole P150 — **E2B / E4B fully supported through HF 256k** with default multi-chunk prefill (4096), unbounded KV (`tests/e2e/test_isl_sweep.py -k long-context-{4k,…,256k}`). CI: `bh_p150` e2e for E2B/E4B.
[^gemma4-p150-12b]: Single Blackhole P150 — **12B through full ISL 256k** with auto **bounded sliding + chunked prefill (4096)** above 32k (unbounded KV OOMs). Measured PASS + coherent gen at 64k/128k/256k. CI short-demo e2e on `bh_p150`; long-context via `MESH_DEVICE=P150 … tests/e2e/test_isl_sweep.py -k long-context-256k` (reports TTFT + decode tok/s). Prefill at 256k is slow (~6 min TTFT).
[^gemma4-lc]: Long-context via `models/demos/gemma4/tests/e2e/test_isl_sweep.py` (`-k long-context-{4k,32k,64k,128k,256k}`) — reports **TTFT and decode tok/s** (long-context `text_demo.py` only logs TTFT). Same `GEMMA4_LONG_CONTEXT_POLICY` in both demos. **Coherence target on QB2 + LoudBox for 12B and 31B: 4k–128k** (256k may allocate; 31B/26B quality not a target). QB2 — E2B/E4B unbounded through 256k; 12B/26B-A4B unbounded through 128k, bounded(+chunked) at 256k; 31B bounded from 64k (chunk=2048 at ≥128k). LoudBox / P150x8 — E2B/E4B/12B unbounded through 256k; **31B/26B-A4B auto-bounded at 128k** with chunk=2048. Defaults: `GEMMA4_HOST_SAMPLE=0` (on-device sampling; set `=1` to force host sampling); do not set `GEMMA4_DEMO_SINGLE_CHUNK=1`.
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

### 12B checkpoint layout (`gemma4_unified`)

The other four variants load straight from their hub id. **12B does not.** The released 12B checkpoint is the encoder-free *unified* multimodal model (`model_type: gemma4_unified` / `Gemma4UnifiedForConditionalGeneration`), which the pinned `transformers` does not recognise, so `HF_MODEL=google/gemma-4-12B-it` fails at load with:

```
ValueError: The checkpoint you are trying to load has model type `gemma4_unified`
but Transformers does not recognize this architecture.
```

The TT model is text-only and reads `text_config` generically, so 12B needs a directory holding an **adapted text config next to the checkpoint's safetensors**:

| File | Purpose |
|---|---|
| `config.json` | Adapted text-only config — copy from [`models/demos/gemma4/configs/gemma-4-12B-it/`](configs/gemma-4-12B-it/). Mirrors the `gemma4`/`gemma4_text` schema with the 12B text dimensions. |
| `*.safetensors` (+ `*.safetensors.index.json`) | The released 12B weights. Symlink from your HF cache snapshot — no need to copy. |
| `tokenizer.json`, `tokenizer_config.json` | From the same snapshot. |
| `config.json.unified` *(optional)* | The **original** HF config. Only needed to load the HF *reference* model in accuracy tests — `test_factory.from_pretrained_gemma4_causal_lm` prefers this sidecar plus `Gemma4UnifiedForConditionalGeneration`. Demos do not need it. |

```bash
SNAP=$(ls -d "$HF_HOME"/hub/models--google--gemma-4-12B-it/snapshots/*/ | head -1)
DEST=/path/to/gemma-4-12B-it-text
mkdir -p "$DEST"
for f in model.safetensors tokenizer.json tokenizer_config.json generation_config.json; do
  ln -sf "$SNAP/$f" "$DEST/$f"
done
cp models/demos/gemma4/configs/gemma-4-12B-it/config.json "$DEST/config.json"
# Optional, for HF-reference accuracy tests only:
# cp "$SNAP/config.json" "$DEST/config.json.unified"

export HF_MODEL="$DEST"
```

The config directory alone is **not** a valid `HF_MODEL` — it holds only `config.json`, so loading it fails with `OSError: Error no file named model.safetensors`. Point `HF_MODEL` at a directory that has both.

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

### 12B on T3K (1×8)

Requires the 12B directory from [12B checkpoint layout](#12b-checkpoint-layout-gemma4_unified) — the hub id will not load. Gated in CI as `wh_llmbox` (unit, tier 1) and `wh_llmbox_perf` (e2e, tier 2); see footnote [^gemma4-12b-t3k] for the outstanding PCC nodes.

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=/path/to/gemma-4-12B-it-text \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-12B-it \
       MESH_DEVICE=T3K
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x8"
pytest models/demos/gemma4/tests/e2e/test_isl_sweep.py -k "batch-1" -sv
pytest models/demos/gemma4/tests/e2e/test_isl_sweep.py -k "long-context-4k" -s --timeout 1800
```

### P150 (1×1) — E2B / E4B / 12B (full ISL through 256k)

Use **`text_demo_v2.py`** for long-context / perf (logs TTFT **and** decode tok/s). `text_demo.py` remains the CI short-demo path.

```bash
export HF_HUB_OFFLINE=1 \
       HF_HOME=/path/to/huggingface \
       HF_MODEL=/path/to/gemma-4-12B-it-text \
       TT_CACHE_PATH=/path/to/huggingface/tt_cache/google--gemma-4-12B-it \
       MESH_DEVICE=P150
# Short demo (CI path)
pytest models/demos/gemma4/demo/text_demo.py::test_demo -k "1x1" -s --timeout 1500
# Full HF context — 12B auto-bounds + chunk=4096 above 32k; E2B/E4B stay unbounded
pytest models/demos/gemma4/tests/e2e/test_isl_sweep.py -k "long-context-256k" -s --timeout 3600
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
pytest models/demos/gemma4/tests/e2e/test_isl_sweep.py -k "long-context-4k or long-context-32k or long-context-128k" -s --timeout 1800
pytest models/demos/gemma4/demo/text_demo.py::test_demo_batch_32 -k "prefill_128 and 1x8" -v
```

For 12B on QB2, use `MESH_DEVICE=P150x4` (filter `-k "1x4"`) with the 12B directory from
[12B checkpoint layout](#12b-checkpoint-layout-gemma4_unified) as `HF_MODEL` — the hub id will not load.

Keep defaults for quality: `GEMMA4_HOST_SAMPLE=0` (default; on-device sampling — measured token-for-token identical to host sampling at batch-1 and 128k, and ~+29% / ~+18% decode tok/s respectively. `=1` forces the host path). Prefill chunk follows `GEMMA4_LONG_CONTEXT_POLICY` (usually 4096; **31B/26B-A4B @ ≥128k on QB2/LB → 2048 + bounded**). Avoid `GEMMA4_DEMO_SINGLE_CHUNK=1` on long ISL (known “la la / lapped” collapse). Optional overrides: `GEMMA4_BOUNDED_SLIDING`, `GEMMA4_GEN_PREFILL_CHUNK`, `GEMMA4_MAX_SEQ_LEN`, `GEMMA4_MAX_NEW_TOKENS`.

On-device sampling runs the top-k pipeline by default. `GEMMA4_TT_FORCE_ARGMAX=1` lets a fully-greedy batch (`top_k=1`, `top_p=1.0`, `temperature=0`) take TTSampling's single all-gather + `ttnn.argmax` path instead, and wires a dedicated CCL for that gather on multi-device meshes; unset or `=0` is the default. Verified token-exact on WH T3K at 1x1 / 1x2 / 1x8 (vocab 262144, batch-32), but **slower than the default** on the 31B batch-32 demo: 79.05 vs 73.34 ms/token (12.65 vs 13.63 tok/s/user). That is expected at this vocab — argmax gathers the full 262144-wide bf16 row (~16.7 MB) where the top-k path gathers only 32-wide results — so the flag is for experiments, not a perf default.

### Short / batch demos (ISL sweep `batch-1`)

```bash
HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=P150x8 \
  pytest models/demos/gemma4/tests/e2e/test_isl_sweep.py -k "batch-1" -sv
```

### Single-layer smoke test

```bash
HF_MODEL=<path-or-id> pytest models/demos/gemma4/demo/text_demo.py::test_demo_single_layer
```

## Details

- **Entry points:**
  - `models/demos/gemma4/demo/text_demo_v2.py` — Generator demo implementation (`run_demo_text`); spec-decode smoke test remains here.
  - `models/demos/gemma4/tests/e2e/test_isl_sweep.py` — **ISL perf sweep** (batch-1 / batch-32 / long-context): reports TTFT **and** decode tok/s. Same `GEMMA4_LONG_CONTEXT_POLICY` as v1.
  - `models/demos/gemma4/demo/text_demo.py` — CI short demo + batched prefill (`test_demo_batch_prefill` / `test_demo_batch_32`). Long-context Generator path works but currently logs TTFT only.
- **Decode loops:** the two demos remove the per-token host round trip by different means, so do not assume a flag from one applies to the other.
  - `text_demo_v2.py`, and `text_demo.py`'s Generator long-context path (`_run_generation_via_generator`), **pipeline the token readback one step deep** (`GEMMA4_DECODE_PIPELINE`, default on): the sampled token's DMA overlaps the next decode submit. Requires on-device sampling *and* a captured decode trace — the device then owns the token buffer between steps, so the host loop needs nothing back before it can submit again. The loop runs one step behind and drains in-flight reads on exit, so emitted text is unchanged.
  - `text_demo.py`'s hand-rolled `run_generation` loop instead **binds the sampler's output straight to the trace's token input** (`GEMMA4_DEMO_TOKEN_FEEDBACK`, default on), so no token is restaged from host at all. It also drops position staging on traced replay when the trace advances positions itself (`device_tracks_decode_on_device`).
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

## Environment Variables

Every performance change on this path ships behind a switch, so any one of them can be reverted at runtime without a rebuild. **The defaults are the supported configuration** — these are for bisecting a regression or reproducing a measurement, not for tuning a deployment.

### Model / run selection

| Variable | Default | Effect |
|---|---|---|
| `HF_MODEL` | — | Hub id or local directory. 12B needs a local directory ([layout](#12b-checkpoint-layout-gemma4_unified)). |
| `MESH_DEVICE` | `(1, N)` over visible devices | Mesh shape. Named SKUs (`T3K`, `P150x8`, `P300x2`, …) map explicitly; unset opens a full line mesh, so a LoudBox gets 1×8 rather than a 4-chip subset. |
| `GEMMA4_NUM_LAYERS` | all | Truncate the model for quick smoke runs. |
| `GEMMA4_MAX_SEQ_LEN`, `GEMMA4_MAX_NEW_TOKENS` | per test case | Override context / generation length. |
| `GEMMA4_BATCH_DEMO_SIZE` | 32 | Batch for `text_demo.py::test_demo_batch_32`. |

### Long context / prefill

| Variable | Default | Effect |
|---|---|---|
| `GEMMA4_LONG_CONTEXT_POLICY` | per (model, device) | Bounded-sliding / chunk cutovers; see `tt/generator_trace.py`. |
| `GEMMA4_BOUNDED_SLIDING`, `GEMMA4_GEN_PREFILL_CHUNK` | policy | Force bounded KV / prefill chunk size. |
| `GEMMA4_DEMO_SINGLE_CHUNK` | off | **Leave off.** Known "la la / lapped" collapse on long ISL. |
| `GEMMA4_PREFILL_TRACE_MAX_SEQ` | 4096 | Above this, prefill trace is off and decode stays traced. |
| `GEMMA4_CHUNKED_PREFILL_TRACE` | auto | Trace multi-chunk prefill on long unbounded demos. |
| `GEMMA4_WARMUP_CHUNK_SPANS` | `all` for 31B at batch ≥ 32, else off | Warm the *row spans* chunked batched prefill lands on. Traces are keyed on row span, not user count, so without this every chunk past the first compiles its trace inside the measured prefill. 31B batch-32 auto-enables `all` because at row span ≥ 16 the in-band capture collides with L1 CBs on WH T3K — there the warmup is a correctness prerequisite, not a perf option. Any explicit value (including `0`) suppresses the auto path; the run logs which way it went. `1` warms only the first chunk boundary. |
| `GEMMA4_TRACE_REGION_SIZE` | 192 MB (WH) / 256 MB (BH) | Must cover the **cumulative** size of every captured trace, not the largest one. |

### Sampling

| Variable | Default | Effect |
|---|---|---|
| `GEMMA4_HOST_SAMPLE` | `0` | `1` forces host sampling, which all-gathers the full 262144-wide vocab and reads it to CPU every token. On-device sampling is the default and measured token-for-token identical. |
| `GEMMA4_DECODE_PIPELINE` | `1` | `0` restores the blocking decode loop (see [Decode loops](#details)). |
| `GEMMA4_DEMO_TOKEN_FEEDBACK` | `1` | `0` restores host token staging in `run_generation`. |
| `GEMMA4_TT_FORCE_ARGMAX` | `0` | Single all-gather + `ttnn.argmax` for fully-greedy batches. Token-exact but slower than the top-k path at this vocab — for experiments, not perf. |

### Matmul / precision

| Variable | Default | Effect |
|---|---|---|
| `GEMMA4_QKV_DECODE_PROGCFG` | `1` | Swept narrow-N program config for the decode QKV matmul. **Not bit-exact** against auto (it re-chooses the blocking, hence accumulation order) but closer to an fp32 reference. `0` restores auto. |
| `GEMMA4_PREFILL_MATMUL_LOFI` | `0` | LoFi on tall prefill matmuls. **Do not default this on** — it corrupts long-context generation, and a single-layer PCC cannot see the error accumulating through 60 layers × 64 prefill chunks into a 131072-token KV cache. Needs a full 128k demo gate. |
| `GEMMA4_OPROJ_TUNED` | `0` | Tuned prefill `o_proj`. Shape-specific: a large 31B win and a 12B regression, because the pinned config's block-sharded output grid depends on `n`. Needs a per-variant gate before it can be a default. |
| `GEMMA4_PREFILL_LONG_2D` | `1` | Cutoff-reshape 2D path above the tuned band. |
| `GEMMA4_PREFILL_L1_TENSOR_MAX_BYTES` | 4 MiB | L1 budget for a single short-lived prefill activation; `0` forces DRAM. |
| `GEMMA4_DECODE_IN0_L1`, `GEMMA4_DECODE_QKV_L1` | `1` | Keep decode activations in L1 instead of staging through DRAM. |
| `GEMMA4_SHARDED_NORM`, `GEMMA4_NORM_KEEP_SHARDED`, `GEMMA4_PREFILL_ISLAND` | `1` | Width-sharded RMSNorm and the sharded residual islands. |
| `GEMMA4_LM_HEAD_FIDELITY` | `hifi3_destacc` | `hifi4_destacc` restores the previous default but exposes a documented WH B0 hardware bug (#38306); HiFi3 + fp32 dest-acc measured equivalent and is silent. |
| `GEMMA4_DECODE_SDPA_FIDELITY` | `hifi2` (op default) | Set `hifi4` to enable the measured optimization. **Do not add fp32 dest-acc to this op** — it collapses batch-1 decode PCC. |
| `GEMMA4_PREFILL_SDPA_FIDELITY` | `hifi4` | Main's validated HiFi4 + fp32 dest-acc policy. `hifi4_nodest` enables the measured alternative that avoids #38306. |
| `GEMMA4_ATTN_WEIGHT_DTYPE` | per `precision_overrides.json` | **Unit-test sweep knob only** (`tests/unit/test_attention.py`) — it does not affect the demos. `bf16` lifts most of the outstanding 12B WH decode PCC nodes over 0.99, which is how the residual loss was attributed to bfp8 attention weights. |

### Collectives

| Variable | Default | Effect |
|---|---|---|
| `GEMMA4_CCL_TOPOLOGY` | `ring` on BH ≥8 / `linear` otherwise | Set `ring` explicitly for the measured WH dense optimization; WH keeps main's validated Linear default. |
| `GEMMA4_CCL_SPLIT` | `1` | Split the TP all-reduce into `reduce_scatter` + `all_gather` so the worker knobs below are reachable. |
| `GEMMA4_CCL_SYNC_RS_WORKERS`, `GEMMA4_CCL_SYNC_RS_CHUNKS` | height-aware: `2` at padded height ≥ 2048, else `1` | Reduce-scatter workers / chunks-per-sync. Decode and short prefill want 1; the tall prefill chunk wants 2. **`w=4` is a 1.5x cliff, not a plateau** — with one link, extra workers contend. Do not raise without re-sweeping, and do not confuse these with the async path's `GEMMA4_CCL_NUM_WORKERS`. |
| `GEMMA4_CCL_ASYNC` | auto (tall prefill only) | Async RS+AG loses to sync in every decode / short-prefill arm. |
| `GEMMA4_CCL_L1_GATHER` | `1` | All-gather writes width-sharded L1 directly instead of DRAM + reshard. |
| `GEMMA4_CCL_PACKET_BYTES` | per model/arch | Fabric max packet payload; `0`/`none`/`default` keeps Fabric's own default. |

> **`num_links=2` is unusable on Wormhole.** It raises `TT_FATAL: Event Order Issue` and, because it fires inside trace capture, wedges the ETH links until a board reset. Keep `num_links=1`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ValueError: ... model type gemma4_unified but Transformers does not recognize this architecture` | 12B loaded from its hub id. Build a local directory per [12B checkpoint layout](#12b-checkpoint-layout-gemma4_unified). |
| `OSError: Error no file named model.safetensors` pointing at `configs/gemma-4-12B-it` | That directory holds only `config.json`. `HF_MODEL` must point at a directory with the config **and** the safetensors. |
| `TT_FATAL ... get_trace_buffers_size() <= trace_region_size`, or `Creating trace buffers of size N ... but only M is allocated` | `trace_region_size` must cover the **cumulative** footprint of every captured trace — batch-1 warms one prefill trace per ISL bucket plus the decode trace, and the failure surfaces at the *last* capture (usually decode) even when the added ops are in the prefill traces. Raise `GEMMA4_TRACE_REGION_SIZE`. |
| `Statically allocated circular buffers ... beyond max L1 size` in gate_up / down_proj / QKV prefill | A tuned prefill program config swept at TP=8 does not fit at TP=1/2, where the weight is not fractured and `N` is up to 8× wider. `linear_l1_safe` should catch this and fall back to auto per `(M, K, N)`; if it throws, that fallback was bypassed. |
| Repeated `"la"` / `lapped → la` at long context | Historically two distinct causes: host sampling receiving one TP vocab shard instead of the full vocabulary, and LoFi on tall prefill matmuls. Both are fixed/defaulted-off — check `GEMMA4_PREFILL_MATMUL_LOFI` is not set, and do not set `GEMMA4_DEMO_SINGLE_CHUNK=1`. |
| Fabric page-size warnings on every AllGather | Fabric's default packet is not a multiple of the CCL page size for this model. Handled by the `GEMMA4_CCL_PACKET_BYTES` default; a warning means the router config did not reach mesh open. |
| Batched demo hangs above batch 4 | Known AllGather shard-height validation failure; `user_cap=4` exists for this ceiling. |
| Two runs of identical code produce different completions | Expected — `text_demo_v2` batch-1 is **not** reproducible run to run. Gate on `test_full_model_decode` PCC (reproduces to 16 digits) plus an isolation `torch.equal`, and use the demo only for coherence and ms/token. A single-run text diff both raises false alarms and can pass a real regression. |

## Notes

- Weight cache is created on first run under `$TT_CACHE_PATH/tensor_cache_<dtype>/`. Subsequent runs reuse it; finetuned weights or a new dtype need a fresh cache. The cache directory must be writable (CI mounts of `/mnt/MLPerf` are often read-only — set `mlperf-read-only=false` or point `TT_CACHE_PATH` at a writable path for first-use).
