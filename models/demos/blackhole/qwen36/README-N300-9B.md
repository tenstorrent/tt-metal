# Qwen3.5-9B on Wormhole N300

This document covers **Qwen3.5-9B running on a single Wormhole N300 card** (2 Wormhole chips,
`MESH_DEVICE=N300`, tensor-parallel degree 2). It is a device-specific companion to the primary
[README.md](README.md), which documents this same code base's **Blackhole** targets (P150 / P150x4 /
P150x8) and the 27B checkpoints. Everything here — code paths, modules, tests — is shared with that
document; only the device and checkpoint are fixed.

On Wormhole, Qwen3.5-9B runs on **N300** — a `(1, 2)` mesh with **TP=2** across two chips on one
card (`MESH_DEVICE=N300`; see `tt/tp_common.py`, `tests/test_factory.py:204`).

## About the model

Qwen3.5-9B is a hybrid **Gated DeltaNet (linear attention) + Gated Full Attention** decoder-only LM.
The checkpoint is multimodal: a 27-block vision tower feeds image embeddings into the text stack.
Everything below is read from the parsed Hugging Face config (`HF_MODEL=Qwen/Qwen3.5-9B`).

The text model has hidden size 4096, vocab 248,320, and 32 decoder layers — 24 Gated DeltaNet and 8
Gated Full Attention, interleaved per `layer_types`. Full attention uses 16 query heads and 4 KV
heads (GQA) with head dim 256; partial RoPE rotates 64 of those 256 dims per head. The SwiGLU MLP
has intermediate size 12,288. Gated DeltaNet uses 16 key heads and 32 value heads (128-dim each),
with a causal conv kernel of 4. Norm is zero-centered RMSNorm (the "+1" fold) everywhere.

## Supported device: N300

| | |
| --- | --- |
| `HF_MODEL` | `Qwen/Qwen3.5-9B` |
| `MESH_DEVICE` | `N300` → mesh shape `(1, 2)` |
| Parallelism | **2-way** tensor parallel (TP=2) |
| Chips | 2 Wormhole chips (one N300 card) |

This is the Wormhole configuration this codebase validates the 9B against. The Blackhole target
for the 9B is the single-chip **P150** (see [README.md](README.md)); N300 and P150 exercise
different code paths (TP=2 collectives vs. single-device).

## Model modules

| Module | Path | Description |
| --- | --- | --- |
| Top-level model | `tt/model.py` | Embedding, 32-layer stack, final norm, LM head, KV/GDN state, generation loop |
| Model config | `tt/model_config.py` | Dims, dtypes, per-device tuning knobs, weight loading |
| Decoder layer | `tt/layer.py` | Norm + (GDN or attention) + norm + MLP per `layer_types` |
| Gated full attention | `tt/attention/` | GQA, partial RoPE, paged KV cache (prefill/decode/TP) |
| Gated DeltaNet | `tt/gdn/` | Causal conv, chunked/recurrent gated delta rule, recurrent state |
| MLP | `tt/mlp.py` | SwiGLU feed-forward |
| RMSNorm | `tt/rms_norm.py` | Zero-centered RMSNorm (the "+1" fold) |
| RoPE | `tt/rope.py` | Partial-rotary position embeddings (host freq table + on-device lookup) |
| Vision tower | `tt/vision/` | Patch embed, 27 blocks, merger, multimodal embeddings |
| Weight remapping | `tt/weight_mapping.py` | HF checkpoint key → internal module key mapping |
| TP helpers | `tt/tp_common.py` | Wormhole/N300 gates (`wh_9b_n300`), weight fracturing |
| Wormhole shims | `tt/chunk_seq_wh.py`, `tt/wh_compat.py`, `tt/prefill_norm_tuned.py` | Wormhole-only numerics/perf overrides |
| Generator interface | `tt/generator_interface.py`, `tt/qwen36_vllm.py` | vLLM-compatible generation contract |

## File paths

```
models/demos/blackhole/qwen36/
├── tt/                        # TTNN implementation (imported as models.demos.blackhole.qwen36.tt.*)
│   ├── model.py
│   ├── model_config.py
│   ├── layer.py
│   ├── mlp.py, rms_norm.py, rope.py
│   ├── weight_mapping.py, tp_common.py
│   ├── chunk_seq_wh.py, wh_compat.py, prefill_norm_tuned.py
│   ├── generator_interface.py, qwen36_vllm.py
│   ├── attention/
│   ├── gdn/
│   └── vision/
├── tests/
│   ├── unit/
│   ├── e2e/
│   ├── perf/
│   ├── pcc_thresholds.json
│   └── test_*.py
├── demo/
│   ├── text_demo.py
│   ├── vision_demo.py
│   ├── benchmark_vision.py
│   └── sample_prompts/
├── utils/                     # substate helper (weight key slicing)
├── README.md
├── README-N300-9B.md
└── VISION_TOWER_PERF.md
```

## Dependencies

| Dependency | Required for |
| --- | --- |
| `transformers` ≥ 5.12.1 | HF reference classes (`Qwen3_5ForCausalLM`, `Qwen3_5VisionModel`) used by PCC tests |
| tt-metal / TTNN | On-device inference |
| `HF_MODEL=Qwen/Qwen3.5-9B` | Checkpoint weights (Hub id or local directory) |

## Quick start

```bash
export HF_MODEL=Qwen/Qwen3.5-9B
export MESH_DEVICE=N300

# Full-depth correctness (32 layers, real weights, vs HF) -- the primary N300 correctness gate
pytest models/demos/blackhole/qwen36/tests/unit/test_prefill.py \
       models/demos/blackhole/qwen36/tests/unit/test_decode.py -v -s

# Demo: short-context traced generation
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128"
```

## Environment variables

Set these before running tests or demos on N300. Only `HF_MODEL` and `MESH_DEVICE` need to be
exported explicitly for the standard N300/9B runs; the rest use their defaults unless you override
them.

| Variable | Default | Required for | N300 test value |
| --- | --- | --- | --- |
| `HF_MODEL` | `Qwen/Qwen3.6-27B` in `*_tp` tests if unset | Checkpoint weights | `Qwen/Qwen3.5-9B` |
| `MESH_DEVICE` | Device-count fallback if unset | Device mesh selection | `N300` |
| `QWEN_SDPA_BF8` | `1` on N300, `0` elsewhere | SDPA bf8 Q/KV path | unset (on) |
| `QWEN_GDN_PHASED` | hardcoded on | Demo doc only (not env-controlled in code) | unset (on) |
| `QWEN_GDN_FLAT_QKV` | hardcoded on | Demo doc only (not env-controlled in code) | unset (on) |
| `QWEN36_FULL_DEPTH_PROMPT_LEN` | `128` | Prompt length in full-depth HF parity tests | `128` |
| `QWEN36_FULL_DEPTH_DECODE_STEPS` | `5` | Decode steps in full-depth HF parity tests | `5` |
| `QWEN36_FULL_DEPTH_REF_DTYPE` | `bfloat16` | HF reference dtype in full-depth tests | `bfloat16` |
| `QWEN36_TF_PREFILL_LEN` | `128` | Prefill length in teacher-forced e2e | `128` |
| `QWEN36_TF_MAX_NEW_TOKENS` | `128` | Decode length in teacher-forced e2e | `128` |
| `QWEN36_TF_TEXT_FILE` | Tale of Two Cities corpus | Ground-truth text in teacher-forced e2e | unset |

```bash
export HF_MODEL=Qwen/Qwen3.5-9B
export MESH_DEVICE=N300

# Full N300 suite (long-context prefill paths)
pytest <test_file> --max-prefill 131072 -v -s -p no:cacheprovider
```

| Pytest option | Default | N300 test value |
| --- | --- | --- |
| `--max-prefill` | `8192` | `131072` (full suite; quick runs omit it) |

## Test cases

Three tiers matter for N300: **full-depth HF parity** (`tests/unit/test_prefill.py`, `tests/unit/test_decode.py`),
**teacher-forced e2e** (`tests/e2e/test_teacher_forcing_e2e.py`), and **multi-device TP contract**
(`test_*_tp.py`, `test_model_tp.py`, plus vision and utility tests below). The primary
[README.md](README.md) covers Blackhole P150 / P150x4 only; N300/9B specifics live here.

Blackhole-only `tests/unit/*` component tests (attention, GDN, MLP, …) and top-level
`tests/test_prefill.py` / `tests/test_decode_bucketing.py` do **not** run the 9B on N300.

### Full-depth HF parity — `tests/unit/`

Every layer, real checkpoint, vs `Qwen3_5ForCausalLM` (not another TT path). Shared harness:
`tests/unit/full_depth_pcc_common.py`. `build_full_depth_model` asserts `n_layers == num_hidden_layers`.

| Test | One-line detail |
| --- | --- |
| `test_prefill.py::test_full_depth_prefill_logits_pcc` | 32-layer `prefill_paged` last-position logits vs HF |
| `test_decode.py::test_full_depth_decode_logits_pcc` | 32-layer teacher-forced decode (vLLM contract) vs HF after prefill |

Each decode step uses HF's argmax token so PCC is not inherited from an earlier greedy divergence.

### Teacher-forced e2e — `tests/e2e/`

Real `prefill_paged` + decode chain; ground-truth token fed every step (*A Tale of Two Cities* by
default, override with `QWEN36_TF_TEXT_FILE`). Runs 128 decode steps and reports position trends the
5-step full-depth test cannot see.

| Test | One-line detail |
| --- | --- |
| `test_teacher_forcing_e2e.py::test_teacher_forcing_e2e` | Top-1 / top-5 argmax agreement vs HF |
| `test_teacher_forcing_e2e.py::test_teacher_forcing_logits_pcc` | Full-vocab logit PCC at every step |

### Multi-device TP contract — `tests/` (set `HF_MODEL` / `MESH_DEVICE` for 9B/N300)

| Test | One-line detail |
| --- | --- |
| `test_mlp_tp.py` | TP SwiGLU MLP: prefill + decode-shape, single layer |
| `test_attention_tp.py` | TP gated full attention: decode / prefill / paged-KV, single layer |
| `test_gdn_tp.py` | TP GDN: decode, chunk-prefill, batched prefill (batch capped at 2/4), state R/W |
| `test_rope_tp.py` | TP partial-rotary RoPE, prefill + decode (synthetic q/k, real config dims) |
| `test_model_tp.py` | Full-model TP contract — **8 layers max** (not 32); prefill/decode/batched/traced |
| `test_generate_tp.py` | All 32 layers, functional generate check only (no PCC gate) |
| `test_weight_mapping.py` | HF → internal weight-key remapping, CPU only |
| `test_sampling.py` | On-device RNG sampling (skips on single device) |
| `test_batched_row_agreement.py` | Host-only batched-decode row agreement at long context |

### Vision tower — `tests/`

| Test | One-line detail |
| --- | --- |
| `test_vision_attention.py` | Single vision-attention block vs checkpoint, layer 0 |
| `test_vision_block.py` | All 27 blocks vs checkpoint, each checked independently |
| `test_vision_patch_embed.py` | Patch embed vs checkpoint, two grid sizes |
| `test_vision_tower_pcc.py` | Config-init smoke test: one block + full 27-block tower |
| `test_patch_merger.py` | Patch merger vs checkpoint |
| `test_model.py` / `test_wrapped_model.py` | Full tower at 1/2/27 layers — CI runs **2-layer only** (see **CI coverage**) |

Measured gates and PCC values: **PCC results** below.

## Commands

```bash
export HF_MODEL=Qwen/Qwen3.5-9B
export MESH_DEVICE=N300

# Full-depth parity (primary N300 correctness gate)
pytest models/demos/blackhole/qwen36/tests/unit/test_prefill.py \
       models/demos/blackhole/qwen36/tests/unit/test_decode.py -v -s

# Teacher-forced e2e
pytest models/demos/blackhole/qwen36/tests/e2e/test_teacher_forcing_e2e.py -sv --timeout=0

# TP component tests
pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_rope_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_model_tp.py -svq

# Vision tower
pytest models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_vision_block.py -v -s

# Demos
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced"      # all ISLs
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128" # one ISL
pytest models/demos/blackhole/qwen36/demo/vision_demo.py -v -s
```

Full suite: add `--max-prefill 131072 -p no:cacheprovider` per file (see **Environment variables**).

## Demos

Both demos support N300 (`"N300": (1, 2)` in code). Docstrings still say "Blackhole" only — stale.
Demos check output quality / degeneracy, **not** HF parity; use the tests above for reference validation.

### Text demo (`demo/text_demo.py`)

Prompts are built per ISL (`_get_prompt`); no input file on the command line.

| ISL | Prompt source | Task |
| --- | --- | --- |
| 128 (≤256) | `models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json` | open continuation |
| 4k | `demo/sample_prompts/input_data_long_4k.json` | summarize AI-history text |
| 8k – 128k | `demo/sample_prompts/eval_frankenstein_long.json` → Gutenberg *Frankenstein* | "Based on the above text: …" |
| 256k | same file, entry 4 → Gutenberg *War and Peace* | "Based on the above text: …" |

Long-context entries download on first run into `demo/sample_prompts/.context_cache`. Prompts are
clipped to the ISL (never padded). `QWEN35_REF_PROMPT=1` swaps ISLs ≥4k for the 27B extractive task;
`QWEN35_NO_THINK=1` emits an empty `<think>` block.

Greedy decode (`QWEN35_TEMP` unset); exact strings are not a contract — treat logged output as
sanity references. Long ISLs (≥8k) must mention source-corpus terms; the demo logs matches.

### Vision demo (`demo/vision_demo.py`)

| Prompt file | Media | Question |
| --- | --- | --- |
| `vision_demo.json` | 1 image (beach / dog) | "Describe this image." |
| `vision_multi_image.json` | same image twice | "Identify the differences…" |
| `vision_video.json` | `space_woaudio.mp4`, 16 frames | "Describe this video." |
| `vision_text_only.json` | none (tower skipped) | "Who are you…" |

Video via `transformers` `load_video` (pyav), not `qwen_vl_utils`. `_assert_describes_input` checks
N-of-M expected terms per case and rejects forbidden "corrupted/noise" phrases — stricter than the
text demo's degeneracy checks, but still not numeric HF comparison.

## PCC results — N300, Qwen3.5-9B

Gates from `tests/pcc_thresholds.json` unless noted; teacher-forcing top-1/top-5 floors are in
`tests/e2e/test_teacher_forcing_e2e.py` (`_MEASURED_FLOORS`). Gates sit **below** measured values.

| Test | Gate | Measured (N300) |
| --- | --- | --- |
| `test_full_depth_prefill_logits_pcc` (32 layers, 128-token prompt) | 0.98 | 0.998400 |
| `test_full_depth_decode_logits_pcc` (32 layers, 5 steps) | 0.95 | 0.994791 (mean) |
| `test_teacher_forcing_logits_pcc` (128 decode steps) | — | 0.992495 (mean), 0.933855 (worst step) |
| `test_attention_tp` (single layer, decode pos0) | 0.97 | 0.99993 – 0.99996 |
| `test_attention_tp_prefill` (single layer, S=64) | 0.95 | 0.999231 |
| `test_attention_tp_paged_peruser` (B=8 / B=32) | 0.97 | 1.00000 |
| `test_gdn_tp` (single layer, decode pos0) | 0.92 | 0.99988 |
| `test_gdn_tp` per-user state (B=8) | 0.92 | 1.00000 |
| `test_mlp_tp` (single layer, decode) | 0.97 | 0.998515 |
| `test_mlp_tp` prefill T=2048, bf16 / bf8 / bf4 in | 0.97 | 0.999394 / 0.999268 / 0.985412 |
| `test_rope_tp` partial RoPE, prefill / decode (q, k) | 0.99 | ≥ 0.9999991 |
| Vision tower, depth 27 (real checkpoint) | 0.985 | 0.98850 |
| Vision tower, depth 27 (after sequence-padding fix) | — | 0.99929 |
| Vision tower, depth 1 (real checkpoint) | 0.998 | 0.99981 |



## Performance

End-to-end numbers from demo runs on N300/9B, all cases `rc=0`. Throughput is logged live by
`text_demo.py` / `vision_demo.py` (no CI perf job — see **CI coverage**). Op-level micro-benchmarks
are inline in `tt/mlp.py`, `tt/gdn/tp.py`, `tt/attention/tp.py`, `tt/gdn/conv_fir_wh.py`.

### Text generation

All 8 traced ISLs, batch 1, greedy. Cold-`.rp`-cache run; reproduces earlier ladders within 0.02 s
TTFT and ±0.4 tok/s.

| ISL | Gen tokens | TTFT | Decode | ms/tok |
| --- | --- | --- | --- | --- |
| 128 | 50 | 0.24 s | 22.84 tok/s | 43.8 |
| 4k | 100 | 0.95 s | 22.58 tok/s | 44.3 |
| 8k | 100 | 1.95 s | 22.37 tok/s | 44.7 |
| 16k | 100 | 3.97 s | 22.41 tok/s | 44.6 |
| 32k | 100 | 8.33 s | 22.71 tok/s | 44.0 |
| 64k | 500 | 18.66 s | 22.12 tok/s | 45.2 |
| 128k | 100 | 33.50 s | 20.76 tok/s | 48.2 |
| 256k | 100 | 125.43 s | 18.15 tok/s | 55.1 |

Decode holds ~flat to 128k (~21% drop by 256k as paged-KV attention grows). TTFT near-linear through
128k, then steepens.

### Vision

**Demo throughput** (`demo/vision_demo.py`, all 5 cases):

| Case | Prompt tokens | TTFT | Decode | Gen |
| --- | --- | --- | --- | --- |
| `traced_single_image` | 2770 (2752 image) | 4.52 s | 24.8 tok/s | 419 |
| `paged_single_image` | 2770 (2752 image) | 2.14 s | 24.7 tok/s | 300 |
| `traced_multi_image` | 5529 (5504 image) | 6.64 s | 24.6 tok/s | 124 |
| `traced_video` | 820 (728 video) | 0.76 s | 24.9 tok/s | 300 |
| `traced_text_only` | 21 | 0.31 s | 24.9 tok/s | 100 |

Traced TTFT exceeds untraced for the 2770-token single-image case: prompt spans `PREFILL_CHUNK=2048`,
so traced prefill runs two passes vs one untraced pass. Trace capture is outside the TTFT timer.

**Tower kernel tuning** (full writeup: `VISION_TOWER_PERF.md`):

| Stage | Before | After (N300) |
| --- | --- | --- |
| Depth-1 window (one block, device time) | 11,876 µs | **6,887 µs** (1.72x) |
| Full tower device time | 52,477 µs | 45,343 µs (−13.6%) |
| SDPA (8 heads/device, 256/512) | 20.62 ms | **15.29 ms** (1.35x) |
| Redundant data-movement pass | 18,119 µs device | 79.46 ms → 71.81 ms wall (−9.6%) |
| CCL workers (`wpl=4` vs `wpl=2`) | 5.77 ms | **4.69 ms** (−19%) |

## Known limitations (N300, 9B)

- **Batched GDN prefill is capped at batch size 2–4**, not the model's full serving batch — the
  `gated_delta_attn_seq` kernel maps one row to `B * Nv_tp`, which bounds how large a batch can be
  processed in one prefill call (`tests/test_gdn_tp.py:351`).
- **GDN decode batch-splits above B=16 at fp32 recurrent state on N300** specifically (more headroom
  at bf16/bf8) — `tt/gdn/tp.py:350-357` (re-verified against current HEAD `a8233578aec`; shifted by
  2 lines from an earlier rebase).
- **`test_model_tp.py` is capped at 8 layers**, not the real 32 — see **Test cases**.
