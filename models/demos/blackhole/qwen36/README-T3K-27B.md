# Qwen3.6-27B on Wormhole T3K

This document covers **Qwen3.6-27B running on a Wormhole T3K** (8 Wormhole chips,
`MESH_DEVICE=T3K`, tensor-parallel degree 8). It is a device-specific companion to the primary
[README.md](README.md), which documents this same code base's **Blackhole** targets (P150 /
P150x4). Everything here — code paths, modules, tests — is shared with that document; only the
device and checkpoint are fixed.

> **T3K is not the shipping topology for the 27B.** The supported Blackhole mesh is **P150x4
> (TP=4)**; T3K is TP=8. The two differ in shard width and in how many collective hops each block
> pays, so the numbers here do **not** transfer to P150x4 by a simple scale factor. This document
> exists because T3K is the hardware these measurements were actually taken on — treat it as the
> Wormhole reference point, not as a Blackhole projection.

The same code serves Qwen3.5-27B on the same mesh; set `HF_MODEL=Qwen/Qwen3.5-27B`. Only
Qwen3.6-27B was measured for this document.

## About the model

Qwen3.6-27B is a hybrid **Gated DeltaNet (linear attention) + Gated Full Attention** decoder-only
LM. The checkpoint is multimodal: a 27-block vision tower feeds image embeddings into the text
stack. Everything below is read from the parsed Hugging Face config (`HF_MODEL=Qwen/Qwen3.6-27B`).

The text model has hidden size 5120, vocab 248,320, and 64 decoder layers — 48 Gated DeltaNet and
16 Gated Full Attention, interleaved per `layer_types`. Full attention uses 24 query heads and 4 KV
heads (GQA) with head dim 256; partial RoPE (`partial_rotary_factor = 0.25`) rotates 64 of those 256
dims per head. The SwiGLU MLP has intermediate size 17,408. Gated DeltaNet uses 16 key heads and 48
value heads (128-dim each), with a causal conv kernel of 4. Norm is zero-centered RMSNorm (the "+1"
fold) everywhere. `max_position_embeddings` is 262,144 (256k).

Against the 9B: 2× the layers, 1.25× the hidden size, 2× the KV-cache-carrying layers (16 vs 8).

## Supported device: T3K

| | |
| --- | --- |
| `HF_MODEL` | `Qwen/Qwen3.6-27B` |
| `MESH_DEVICE` | `T3K` → mesh shape `(1, 8)` |
| Parallelism | **8-way** tensor parallel (TP=8) |
| Chips | 8 Wormhole chips |
| Fabric | `FABRIC_1D` (wired from `MESH_DEVICE` by the test fixtures) |

The Blackhole target for the 27B is **P150x4** (a `(1, 4)` mesh, TP=4) — see [README.md](README.md).
T3K and P150x4 exercise the same TP code path at different widths.

## Model modules

| Module | Path | Description |
| --- | --- | --- |
| Top-level model | `tt/model.py` | Embedding, 64-layer stack, final norm, LM head, KV/GDN state, generation loop |
| Model config | `tt/model_config.py` | Dims, dtypes, per-device tuning knobs, weight loading |
| Decoder layer | `tt/layer.py` | Norm + (GDN or attention) + norm + MLP per `layer_types` |
| Gated full attention | `tt/attention/` | GQA, partial RoPE, paged KV cache (prefill/decode/TP) |
| Gated DeltaNet | `tt/gdn/` | Causal conv, chunked/recurrent gated delta rule, recurrent state |
| MLP | `tt/mlp.py` | SwiGLU feed-forward |
| RMSNorm | `tt/rms_norm.py` | Zero-centered RMSNorm (the "+1" fold) |
| RoPE | `tt/rope.py` | Partial-rotary position embeddings (host freq table + on-device lookup) |
| Vision tower | `tt/vision/` | Patch embed, 27 blocks, merger, multimodal embeddings |
| Weight remapping | `tt/weight_mapping.py` | HF checkpoint key → internal module key mapping |
| TP helpers | `tt/tp_common.py` | Mesh/grid/link derivation, weight fracturing |
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
└── README-T3K-27B.md
```

## Dependencies

| Dependency | Required for | Version validated |
| --- | --- | --- |
| tt-metal / TTNN | On-device inference | `v0.77.0-dev20260808`, commit `4724ad6dc08` (built in-tree) |
| `transformers` | HF reference classes (`Qwen3_5ForCausalLM`, `Qwen3_5VisionModel`) used by PCC tests | 5.12.1 (5.x required; 4.x lacks the architecture) |
| PyTorch | Reference model + host-side weight prep only | 2.11.0 (CPU build) |
| Python | — | 3.10 (`python_env/`, from `create_venv.sh`) |
| `HF_MODEL=Qwen/Qwen3.6-27B` | Checkpoint weights (Hub id or local directory) | — |

## Quick start

```bash
export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=T3K

# Full-depth correctness (64 layers, real weights, vs HF) -- the primary T3K correctness gate
pytest models/demos/blackhole/qwen36/tests/unit/test_prefill.py \
       models/demos/blackhole/qwen36/tests/unit/test_decode.py -v -s

# Demo: short-context traced generation
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128"
```

## Environment variables

Only `HF_MODEL` and `MESH_DEVICE` need to be exported for standard T3K/27B runs; the rest use their
defaults unless overridden.

| Variable | Default | Required for | T3K test value |
| --- | --- | --- | --- |
| `HF_MODEL` | `Qwen/Qwen3.6-27B` in `*_tp` tests if unset | Checkpoint weights | `Qwen/Qwen3.6-27B` |
| `MESH_DEVICE` | Device-count fallback if unset | Device mesh selection | `T3K` |
| `QWEN_SDPA_BF8` | off on T3K | SDPA bf8 Q/KV path (faster, slightly lower precision) | unset (off) |
| `QWEN_GDN_PHASED` | on | Opt-out only: `0` falls back to the monolithic fused GDN kernel | unset (on) |
| `QWEN_GDN_FLAT_QKV` | on | Opt-out only: `0` falls back to head-split q/k/v + host `l2_norm` | unset (on) |
| `QWEN36_FULL_DEPTH_PROMPT_LEN` | `128` | Prompt length in full-depth HF parity tests (keep a multiple of 128) | `128` |
| `QWEN36_FULL_DEPTH_DECODE_STEPS` | `5` | Decode steps in full-depth HF parity tests | `5` |
| `QWEN36_FULL_DEPTH_REF_DTYPE` | `bfloat16` | HF reference dtype in full-depth tests | `bfloat16` |
| `QWEN36_TF_PREFILL_LEN` | `128` | Prefill length in teacher-forced e2e | `128` |
| `QWEN36_TF_MAX_NEW_TOKENS` | `128` | Decode length in teacher-forced e2e | `128` |
| `QWEN36_TF_TEXT_FILE` | Tale of Two Cities corpus | Ground-truth text in teacher-forced e2e | unset |
| `QWEN35_TEMP` | `0` (greedy) | Sampling temperature; `>0` moves sampling to host | unset |
| `QWEN35_REF_PROMPT` | unset | Swaps ISLs ≥4k for the 64k extractive reference task | unset |
| `QWEN35_NO_THINK` | unset | Emits an empty `<think>` block | unset |

```bash
export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=T3K

# Full T3K suite (long-context prefill paths)
pytest <test_file> --max-prefill 131072 -v -s -p no:cacheprovider
```

| Pytest option | Default | T3K test value |
| --- | --- | --- |
| `--max-prefill` | `8192` | `131072` (full suite; quick runs omit it) |

## Test cases

Three tiers matter for T3K: **full-depth HF parity** (`tests/unit/test_prefill.py`,
`tests/unit/test_decode.py`), **teacher-forced e2e** (`tests/e2e/test_teacher_forcing_e2e.py`), and
the **multi-device TP contract** (`test_*_tp.py`, `test_model_tp.py`, plus vision and utility tests
below). PCC gates live in `tests/pcc_thresholds.json`.

The single-device `tests/unit/*` component tests (attention, GDN, MLP, embedding, RMSNorm, RoPE, LM
head) default to the **9B** checkpoint on a single device and are not part of the T3K/27B path.

### Full-depth HF parity — `tests/unit/`

Every layer, real checkpoint, vs `Qwen3_5ForCausalLM` (not another TT path). Shared harness:
`tests/unit/full_depth_pcc_common.py`. `build_full_depth_model` asserts `n_layers == num_hidden_layers`.

| Test | One-line detail |
| --- | --- |
| `test_prefill.py::test_full_depth_prefill_logits_pcc` | 64-layer `prefill_paged` last-position logits vs HF |
| `test_decode.py::test_full_depth_decode_logits_pcc` | 64-layer teacher-forced decode (vLLM contract) vs HF after prefill |

Each decode step uses HF's argmax token, so PCC is not inherited from an earlier greedy divergence —
while still carrying the paged KV and GDN recurrent/conv state the previous steps advanced.

### Teacher-forced e2e — `tests/e2e/`

Real `prefill_paged` + decode chain; ground-truth token fed every step (*A Tale of Two Cities* by
default, override with `QWEN36_TF_TEXT_FILE`). Runs 128 decode steps and reports the position trend
the 5-step full-depth test structurally cannot see.

| Test | One-line detail |
| --- | --- |
| `test_teacher_forcing_e2e.py::test_teacher_forcing_e2e` | Top-1 / top-5 argmax agreement vs HF, plus flip classification and Wilson intervals |
| `test_teacher_forcing_e2e.py::test_teacher_forcing_logits_pcc` | Full-vocab logit PCC at every step, plus KL and max\|Δlogit\| over HF's top-32 |

### Multi-device TP contract — `tests/`

| Test | One-line detail |
| --- | --- |
| `test_mlp_tp.py` | TP SwiGLU MLP: column/row-parallel + reduce-scatter, single layer |
| `test_attention_tp.py` | TP gated full attention: decode / prefill / paged-KV, single layer |
| `test_gdn_tp.py` | TP GDN: decode, chunk-prefill, batched prefill, per-user state R/W |
| `test_rope_tp.py` | TP partial-rotary RoPE, prefill + decode (synthetic q/k, real config dims) |
| `test_model_tp.py` | Full-model TP contract — **8 layers max** (not 64); prefill/decode/batched/traced |
| `test_generate_tp.py` | All 64 layers, functional generate check against an answer oracle (no PCC gate) |
| `test_weight_mapping.py` | HF → internal weight-key remapping, CPU only (shape constants assume the 9B) |
| `tests/unit/test_substate.py` | Weight `substate` slicing helper, CPU only |

### Vision tower — `tests/`

| Test | One-line detail |
| --- | --- |
| `test_vision_patch_embed.py` | Patch embed + interpolated positional embedding vs checkpoint, two grid sizes |
| `test_vision_attention.py` | Single vision-attention block vs checkpoint, layer 0 |
| `test_vision_block.py` | All 27 blocks vs checkpoint, each checked independently |
| `test_patch_merger.py` | 2×2 patch merger → text hidden width, vs checkpoint |
| `test_vision_tower_pcc.py` | Config-init smoke gate: one block + full 27-block tower (**random init**, not the checkpoint) |
| `test_model.py` / `test_wrapped_model.py` | Full tower assembly at reduced depth |

Measured gates and PCC values: **PCC results** below.

## Commands

```bash
export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=T3K

# Full-depth parity (primary T3K correctness gate)
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
pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py -v -s

# Vision tower
pytest models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_vision_block.py -v -s

# Demos
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced"     # all ISLs
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128" # one ISL
pytest models/demos/blackhole/qwen36/demo/vision_demo.py -v -s
```

Full suite: add `--max-prefill 131072 -p no:cacheprovider` per file (see **Environment variables**).

## Demos

Demos check output quality / degeneracy, **not** HF parity; use the tests above for reference
validation.

### Text demo (`demo/text_demo.py`)

Prompts are built per ISL (`_get_prompt`); no input file on the command line.

| ISL | Prompt source | Task |
| --- | --- | --- |
| 128 (≤256) | `models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json` | open continuation (repeated to fill, then clipped) |
| 4k | `demo/sample_prompts/input_data_long_4k.json` | summarize AI-history text |
| 8k – 128k | `demo/sample_prompts/eval_frankenstein_long.json` → Gutenberg *Frankenstein* (pg84) | "Based on the above text: …" |
| 256k | same file, entry 4 → Gutenberg *War and Peace* (pg2600) | "Based on the above text: …" |

Long-context entries download on first run into `demo/sample_prompts/.context_cache` (gitignored).
Prompts are clipped to the ISL, never padded. `QWEN35_REF_PROMPT=1` swaps ISLs ≥4k for the 64k
extractive task; `QWEN35_NO_THINK=1` emits an empty `<think>` block.

Greedy decode (`QWEN35_TEMP` unset); exact strings are not a contract — treat logged output as a
sanity reference. Long ISLs (≥8k) must mention source-corpus terms; the demo logs the matches.

Beyond the 8 single-user traced ISLs, the demo also covers batched serving (`batched_*_b8`,
`batched_*_b32`, up to 64k at B=8), a determinism case (`determinism_128`, same prompt twice, output
must be identical), and non-traced `paged_*` variants.

### Vision demo (`demo/vision_demo.py`)

| Prompt file | Media | Case id |
| --- | --- | --- |
| `vision_demo.json` | 1 image | `traced_single_image` (500 tokens), `paged_single_image` (300) |
| `vision_multi_image.json` | several images in one prompt | `traced_multi_image` (300) |
| `vision_video.json` | multi-frame video (temporal patch 2) | `traced_video` (300) |
| `vision_text_only.json` | none (tower skipped) | `traced_text_only` (100) |

`_assert_describes_input` checks N-of-M expected terms per case and rejects "corrupted/noise"
phrasing — stricter than the text demo's degeneracy checks, but still not a numeric HF comparison.

## PCC results — T3K, Qwen3.6-27B

Gates from `tests/pcc_thresholds.json`; teacher-forcing floors are in
`tests/e2e/test_teacher_forcing_e2e.py` (`_MEASURED_FLOORS`). Gates sit **below** measured values.

| Test | Gate | Measured (T3K) |
| --- | --- | --- |
| `test_full_depth_prefill_logits_pcc` (64 layers, 128-token prompt) | 0.98 | 0.995684 |
| `test_full_depth_decode_logits_pcc` (64 layers, 5 steps) | 0.95 | 0.993884 (mean), 0.990054 (worst step) |
| `test_teacher_forcing_e2e` top-1 / top-5 vs HF (128 steps) | 0.87 / 0.93 | **93.02% / 97.67%** |
| `test_teacher_forcing_logits_pcc` (128 decode steps) | 0.75 | 0.967950 (mean), 0.876796 (worst step) |

Component TP tests, measured on this T3K in one pass (all passing):

| Test | Gate | Measured (T3K, TP=8) |
| --- | --- | --- |
| `test_mlp_tp` (single layer, decode) | 0.97 | 0.999192 |
| `test_mlp_tp` prefill T=2048, bf16 / bf8 / bf4 in | 0.97 | 0.998952 / 0.998944 / 0.997938 |
| `test_attention_tp` (single layer, decode pos0) | 0.97 | 0.99993 – 0.99996 |
| `test_attention_tp_prefill` (single layer, S=64) | 0.95 | 0.999156 |
| `test_attention_tp_paged` **prefill** paged-vs-concat | 0.97 | 0.999686 |
| `test_attention_tp_paged` **decode** paged-vs-concat (B=1) | 0.97 | 1.000000 |
| `test_attention_tp_paged_peruser` (B=8) | 0.97 | 1.00000 |
| `test_gdn_tp` (single layer, decode pos0) | 0.92 | 0.99897 – 0.99999 |
| `test_gdn_tp` prefill-vs-decode (T=128) | 0.95 | 0.999300 / 0.999401 |
| `test_gdn_tp` fused-chunk vs step-decode (T=256) | 0.92 | 0.998750 |
| `test_gdn_tp` fused-chunk vs seq-adapter prefill (T=256) | 0.92 | 0.999151 |
| `test_gdn_tp` batched prefill (B=2 / B=4) | 0.92 | 0.99999 / 0.99995 |
| `test_gdn_tp` per-user state (B=8 / B=32) | 0.92 | 1.00000 / 1.00000 |
| `test_rope_tp` partial RoPE, prefill (q / k) | 0.99 | 0.9999992 / 0.9999992 |
| `test_rope_tp` partial RoPE, decode (q / k) | 0.99 | 0.9999993 / 0.9999995 |

> **Resolved: a missing paged KV-cache write.** Until recently the single-user paged decode path
> scored 0.825922 here against a 0.97 gate, and the 27B agreed with HF on only **75.97%** of
> teacher-forced top-1 tokens against the 9B's 90.70% — a gap that had no explanation and was
> tracked as an open finding.
>
> Both had the same cause. In `TPAttention.forward_decode` (`tt/attention/tp.py`), the paged branch
> taken by every config except Wormhole-N300-9B — **Blackhole P150x4 included** — prepared the
> sharded `k_sh`/`v_sh` and then never called `paged_update_cache`. Its two sibling branches both
> issue their own write; this one fell straight through to `deallocate` and SDPA. Decode therefore
> attended over a cache holding only the prefill tokens, with every generated token's slot left at
> zero. A second bug sat behind it in the same branch: the pad's input was freed before the reshard
> consumed the pad's output, which `_WH_KV_PAD_NOTE` had already measured as corrupting B=32
> (10-13/32 users correct) while leaving B=8 clean. Neither was a recent regression — before the
> `wh_9b_n300` narrowing the guard read `is_blackhole()` with the identical structure.
>
> This also explains the position trend that made the old finding so puzzling: agreement fell
> 96.97% → 43.33% across position bins because each successive step was missing more history, while
> the *reference* grew more decisive. With the write restored the trend is gone entirely —
> 93.94 / 93.94 / 87.88 / **96.67%**, the last bin the best — and the 27B now exceeds the 9B.
> TT-vs-truth (89.84%) sits just under HF-vs-truth (93.75%), so the residual gap is ordinary
> numerics rather than a structural defect.
>
> | | before | after |
> | --- | --- | --- |
> | `test_attention_tp_paged` decode PCC | 0.825922 | **1.000000** |
> | full-depth decode worst step | 0.9595 | **0.990054** |
> | teacher-forced top-1 / top-5 vs HF | 75.97% / 84.50% | **93.02% / 97.67%** |
> | teacher-forced logit PCC mean / worst | 0.8403 / 0.3957 | **0.967950 / 0.876796** |
> | position trend (first → last bin) | 96.97% → 43.33% | 93.94% → 96.67% |
>
> `_MEASURED_FLOORS` in `tests/e2e/test_teacher_forcing_e2e.py` was raised from 0.67/0.77/0.35 to
> **0.87/0.93/0.75** to lock the recovered state in.

## Performance

End-to-end numbers from demo runs on T3K/27B, all cases `rc=0`. Throughput is logged live by
`text_demo.py` (no CI perf job). Prefill and decode are both traced and replayed.

**TTFT** is time to first token, wall clock, including prefill trace replay and the first decode
step. **Decode** is per-user throughput in the steady state. Both are the values the demo writes to
`generated/benchmark_data/`.

### Text generation

All 8 traced ISLs, batch 1, greedy. **8/8 passed in 28m17s** (tt-metal `4724ad6dc08`,
`HF_MODEL=Qwen/Qwen3.6-27B`, `MESH_DEVICE=T3K`).

| ISL | Gen tokens | TTFT | Decode | ms/tok |
| --- | --- | --- | --- | --- |
| 128 | 50 | 0.73 s | 17.87 tok/s | 56.0 |
| 4k | 100 | 1.31 s | 17.67 tok/s | 56.6 |
| 8k | 100 | 1.97 s | 17.69 tok/s | 56.5 |
| 16k | 100 | 4.07 s | 17.32 tok/s | 57.7 |
| 32k | 100 | 8.83 s | 16.99 tok/s | 58.9 |
| 64k | 500 | 20.64 s | 16.05 tok/s | 62.3 |
| 128k | 100 | 38.68 s | 14.86 tok/s | 67.3 |
| 256k | 100 | 156.97 s | 11.79 tok/s | 84.8 |

Decode holds within 3% out to 16k and degrades 34% by 256k as paged-KV attention grows. TTFT is
sub-linear through 128k — 1024× the tokens for 53× the TTFT — then steepens sharply into 256k
(4.1× TTFT for 2× the tokens), where the quadratic term in the 16 full-attention layers finally
dominates. The 128 → 4k step costs only 1.8× for 32× the tokens: below ~4k, TTFT is dominated by
fixed per-call overhead (trace replay, embedding gather, LM-head readback), not by sequence length.

**Optimized sequence length: 4k–32k.** Prefill runs chunk-outer — the DeltaNet layers consume a
2048-token chunk (the chunk-seq kernel's L1-resident relayout caps it there) while the full-attention
layers take `max(chunk, 4096)`. 4k is therefore the first ISL that fills a whole attention chunk, and
it is the shape the matmul program configs were swept at. Below it you pay the fixed-overhead floor;
above 32k the quadratic term takes over.

### Against the 9B

The 9B was measured on N300 (TP=2) in the same pass — see the 9B's own device document for the full
ladder. At 4k the 27B costs ~1.3× the 9B's TTFT despite 2× the layers and a 1.25× wider hidden size,
because TP=8 splits each layer eight ways against the 9B's two. Decode is ~0.79× the 9B's and
degrades roughly twice as fast with context (−34% vs −19% from 128 to 256k) — the 27B carries 16
KV-cache layers to the 9B's 8, so twice as much of each decode step grows with context.

### Vision

Not measured on T3K in this pass. The vision tower's device-performance work — L1 budget, matmul and
SDPA sweeps, fidelity choices, and a list of **negative results not to re-try** — is written up in
[VISION_TOWER_PERF.md](VISION_TOWER_PERF.md); that tuning is gated to Wormhole in code
(`VisionModelArgs.vision_mm_tuned` is `is_wormhole_b0()`).

### Reproducing

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
export HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=T3K

# the full 8-ISL ladder (~28 min)
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced" --timeout=0
```

Each case logs a `ttft=... decode=... tok/s` line and appends to `generated/benchmark_data/`. Drop
`-k "traced"` to include the batched (B=8 / B=32) and non-traced `paged_*` cases.
