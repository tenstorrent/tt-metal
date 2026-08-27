# Qwen3.5-9B on Wormhole N300

This document covers **Qwen3.5-9B running on a single Wormhole N300 card** (2 Wormhole chips,
`MESH_DEVICE=N300`, tensor-parallel degree 2). It is a device-specific companion to the primary
[README.md](README.md), which documents this same code base's **Blackhole** targets (P150 / P150x4 /
P150x8) and the 27B checkpoints. Everything here — code paths, modules, tests — is shared with that
document; only the device and checkpoint are fixed.

N300 cannot run the 9B on a single chip: the checkpoint does not fit on one N150, so N300's 2-chip
mesh with TP=2 is the smallest Wormhole configuration that holds it
(`tt/tp_common.py`, `tests/test_factory.py:204`).

## About the model

Qwen3.5-9B is a hybrid **Gated DeltaNet (linear attention) + Gated Full Attention** decoder-only LM,
read from the parsed Hugging Face config so the same code adapts across checkpoints (this document is
scoped to the 9B specifically).

| | Value | Source |
| --- | --- | --- |
| Hidden size (`dim`) | 4096 | `hf_config.hidden_size` |
| Layers | 32 (24 Gated DeltaNet + 8 Gated Full Attention, interleaved per `layer_types`) | `hf_config.layer_types` |
| Attention heads / KV heads | 16 / 4 (GQA) | `hf_config.num_attention_heads` / `num_key_value_heads` |
| Head dim | 256 | `hf_config.head_dim` |
| RoPE | Partial rotary, factor 0.25 → 64 of 256 dims rotated | `hf_config.partial_rotary_factor` |
| Vocab size | 248,320 | `hf_config.vocab_size` |
| MLP intermediate size | 12,288 (SwiGLU) | `hf_config.intermediate_size` |
| GDN key / value heads | 16 / 32 | `hf_config.linear_num_key_heads` / `linear_num_value_heads` |
| GDN key / value head dim | 128 / 128 | `hf_config.linear_key_head_dim` / `linear_value_head_dim` |
| GDN causal conv kernel | 4 | `hf_config.linear_conv_kernel_dim` |
| Norm | Zero-centered RMSNorm (the "+1" fold) everywhere | `tt/rms_norm.py` |
| Vision tower | Present (this checkpoint is multimodal) — 27 vision-transformer blocks | `tt/vision/`, `VISION_TOWER_PERF.md` |

## Supported device: N300

| | |
| --- | --- |
| `HF_MODEL` | `Qwen/Qwen3.5-9B` |
| `MESH_DEVICE` | `N300` → mesh shape `(1, 2)` |
| Parallelism | 4-way→**2-way** tensor parallel (TP=2), the minimum that fits the 9B on Wormhole |
| Chips | 2 Wormhole chips (one N300 card) |

This is the only Wormhole configuration this codebase validates the 9B against — a single N150
cannot hold it, and this repo does not exercise a 4- or 8-chip Wormhole mesh for the 9B. The
Blackhole target for the 9B is the single-chip **P150** (see [README.md](README.md)); N300 and P150
exercise genuinely different code paths (TP=2 collectives vs. single-device), not just different
hardware.

## Architecture

```
                              input_ids [B, T]
                                    |
                              tok_embeddings                 (embedding, dim=4096)
                                    |
                                    v
      +-----------------------------------------------------------+
      |  Qwen36DecoderLayer x 32   (24 x GDN-layer, 8 x Attn-layer,|
      |  order fixed by hf_config.layer_types, per-layer identical|
      |  MLP + norm wiring)                                        |
      |                                                             |
      |   residual -> RMSNorm -> [ GatedDeltaNet | GatedAttention ] |
      |            -> + residual -> RMSNorm -> SwiGLU MLP -> +      |
      +-----------------------------------------------------------+
                                    |
                              RMSNorm (final)
                                    |
                                 LM Head            (dim=4096 -> vocab=248,320)
                                    |
                                 logits [B, T, 248320]

  Gated DeltaNet branch (24/32 layers)          Gated Full Attention branch (8/32 layers)
  --------------------------------------        ------------------------------------------
  x [B,T,4096]                                  x [B,T,4096]
    -> in_proj -> q,k,v,beta,g                    -> q,k,v proj (16 heads / 4 KV heads, GQA)
    -> causal conv1d (kernel=4) + SiLU             -> partial RoPE (64 of 256 dims/head)
    -> per-head L2-norm (q,k)                      -> paged KV cache (decode) / masked SDPA (prefill)
    -> chunked (prefill) or recurrent (decode)      -> attn_out_proj -> [B,T,4096]
       gated delta rule scan -> state [BH,K,V]
    -> RMSNorm(zero-centered) -> out_proj -> [B,T,4096]
```

Tensor shapes above are the logical (pre-sharding) shapes; on N300 (TP=2) each device holds half the
attention/MLP/GDN channel dimensions, replicating the embedding/LM-head columns per the model's
weight-fracturing scheme (`tt/tp_common.py`, `tt/attention/tp.py`, `tt/gdn/tp.py`, `tt/mlp.py`).

## Model modules

| Module | TTNN implementation | Reference | What it does |
| --- | --- | --- | --- |
| Top-level model | `tt/model.py` (`Qwen36Model`) | `transformers.models.qwen3_5.Qwen3_5ForCausalLM` | Embedding, 32-layer stack, final norm, LM head, KV/GDN-state management, generation loop |
| Model config | `tt/model_config.py` (`Qwen36ModelArgs`) | `hf_config.json` at `CKPT_DIR` | Central config: dims, dtypes, per-device tuning knobs, weight loading |
| Decoder layer | `tt/layer.py` (`Qwen36DecoderLayer`) | `Qwen3_5DecoderLayer` | Wires norm + (GDN or attention) + norm + MLP per `layer_types` |
| Gated full attention | `tt/attention/` (`tp.py`, `prefill.py`, `decode.py`, `rope_tp.py`, `weights.py`, `gated_attention.py`, `config.py`) | `Qwen3_5Attention` | GQA attention with partial RoPE, paged KV cache |
| Gated DeltaNet | `tt/gdn/` (`tp.py`, `gated_deltanet.py`, `decode.py`, `state.py`, `conv_fir_wh.py`, `fused_chunk.py`, `recurrent_decode_wh.py`, `weights.py`, `config.py`) | `Qwen3_5GatedDeltaNet` | Linear-attention branch: causal conv, chunked/recurrent gated delta rule, recurrent state |
| MLP | `tt/mlp.py` | `Qwen3_5MLP` | SwiGLU feed-forward |
| RMSNorm | `tt/rms_norm.py` | `Qwen3_5RMSNorm` | Zero-centered ("+1") RMSNorm |
| RoPE | `tt/rope.py` | `Qwen3_5RotaryEmbedding` | Partial-rotary position embeddings (host freq table + on-device lookup) |
| Vision tower | `tt/vision/` (`model.py`, `patch_embed.py`, `patch_merger.py`, `vision_attention.py`, `vision_block.py`, `vision_mlp.py`, `vision_layernorm.py`, `vision_ccl.py`, `vision_model_config.py`) | `Qwen3_5VisionModel` | 27-block ViT-style tower + patch merger producing image embeddings |
| Weight remapping | `tt/weight_mapping.py` | — | HF checkpoint key → internal module key mapping |
| TP / device-scoping helpers | `tt/tp_common.py` | — | `is_blackhole()`, Wormhole/N300-specific gates (`wh_9b_n300`, etc.), weight fracturing helpers |
| Wormhole compatibility shims | `tt/chunk_seq_wh.py`, `tt/wh_compat.py`, `tt/prefill_norm_tuned.py` | — | Wormhole-only numerics/perf overrides layered onto shared kernels (documented per-file as narrow, reversible patches) |
| Generator / serving interface | `tt/generator_interface.py`, `tt/qwen36_vllm.py` | — | vLLM-compatible generation contract |

There is no locally vendored PyTorch reference implementation — the reference for every PCC test is
the installed `transformers` package's `Qwen3_5*` classes, loaded from the same checkpoint directory
as the TT model (see **Dependency versions** below).

## File paths

```
models/demos/blackhole/qwen36/
├── tt/                        # TTNN implementation (imported as models.demos.blackhole.qwen36.tt.*)
│   ├── model.py                   Top-level Qwen36Model
│   ├── model_config.py            Qwen36ModelArgs (central config)
│   ├── layer.py                   Qwen36DecoderLayer
│   ├── mlp.py, rms_norm.py, rope.py
│   ├── weight_mapping.py, tp_common.py
│   ├── chunk_seq_wh.py, wh_compat.py, prefill_norm_tuned.py   # Wormhole-only shims
│   ├── generator_interface.py, qwen36_vllm.py
│   ├── attention/                 Gated full attention (prefill/decode/TP/RoPE/weights)
│   ├── gdn/                       Gated DeltaNet (prefill/decode/TP/state/weights)
│   └── vision/                    Vision tower (patch embed, blocks, merger, TP/CCL)
├── tests/                     # pytest suite (see Test cases below)
│   ├── unit/                      Single-device component PCC tests (Blackhole P150 only, see note)
│   ├── e2e/                       Full end-to-end, real-checkpoint parity tests
│   ├── perf/                      Internal perf-tuning sweep scripts (not part of the supported
│   │                              test surface; not covered by this document)
│   ├── pcc_thresholds.json        Central PCC gate table (test name -> threshold)
│   └── test_*.py                  Multi-device TP tests + single-device text/vision tests
├── demo/
│   ├── text_demo.py               Text-only generation demo/perf harness
│   ├── vision_demo.py             Multimodal (vision + text) generation demo
│   ├── benchmark_vision.py        Vision-tower standalone latency benchmark
│   └── sample_prompts/            Demo input fixtures (incl. long-context corpora, cached on first run)
├── README.md                  Primary README (Blackhole P150/P150x4/P150x8, all checkpoints)
├── README-N300-9B.md          This document
└── VISION_TOWER_PERF.md       Vision-tower device-performance optimisation writeup (N300 + T3K)
```

`models/` has no `__init__.py` files (namespace packages) — imports resolve via the repo root on
`sys.path`, e.g. `from models.demos.blackhole.qwen36.tt.model import Qwen36Model`.

## Dependency versions

Verified in this checkout's own environment (`python_env/`), not from memory:

| Dependency | Version / commit | Notes |
| --- | --- | --- |
| `transformers` | **5.12.1** | `transformers.models.qwen3_5` (the `Qwen3_5ForCausalLM` / `Qwen3_5VisionModel` reference classes every PCC test imports) is only present at this version — it is not yet in older/mainline `transformers` releases. Confirmed by import check in `python_env`. |
| tt-metal / TTNN | `v0.78.0-dev20260827-48-ga8233578aec` | This checkout's own `git describe`; TTNN has no separate `__version__` attribute — the tt-metal build tag is the reference point. |
| HF checkpoint | Resolved from `HF_MODEL=Qwen/Qwen3.5-9B` via `snapshot_download`, or a local directory | `HF_MODEL` is the single source of truth for checkpoint identity; no specific revision/commit is pinned in this repo — pin one yourself if reproducibility across HF Hub updates matters. |

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

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_MODEL` | — (required) | Checkpoint identity: HF Hub id or local directory. Set to `Qwen/Qwen3.5-9B` for everything in this document. |
| `MESH_DEVICE` | — (required) | Mesh shape selector. Set to `N300` → `(1, 2)`. |
| `QWEN_SDPA_BF8` | on by default for N300 (`tt/tp_common.py`) | Runs SDPA in bf8 Q + bf8 KV with HiFi2 fidelity; can be overridden `0`/`1` in either direction. |
| `QWEN_GDN_PHASED` | on (fused phase-split GDN prefill) | Set `0` to fall back to the monolithic single-kernel GDN prefill op (debug/benchmark only). |
| `QWEN_GDN_FLAT_QKV` | on (flat token-major q/k/v reads) | Set `0` to fall back to head-split q/k/v + host L2-norm (debug/benchmark only). |
| `QWEN36_FULL_DEPTH_PROMPT_LEN` | 128 | Prompt length for `unit/test_prefill.py` / `unit/test_decode.py` (keep a multiple of 128). |
| `QWEN36_FULL_DEPTH_DECODE_STEPS` | 5 | Decode steps checked by `unit/test_decode.py`. |
| `QWEN36_FULL_DEPTH_REF_DTYPE` | `bfloat16` | HF reference dtype for the full-depth parity tests. |
| `QWEN36_TF_TEXT_FILE` | *A Tale of Two Cities* (tt_transformers corpus) | Ground-truth text for `e2e/test_teacher_forcing_e2e.py`. |
| `QWEN36_TF_MAX_NEW_TOKENS` | 128 | Decode horizon for the teacher-forced e2e test. |
| `--max-prefill` (pytest flag) | 8192 | `test_prefill.py`'s bucket test auto-skips ISLs above this (Blackhole-only test, see note below — not applicable to N300 runs). |

This is not exhaustive — `tests/perf/` sweep scripts read many more tuning-only env vars not
documented here, since that directory is internal tuning scaffolding, not the supported test/demo
surface.

## Test cases

**Important device note, not stated in the primary README:** the single-device component tests under
`tests/unit/` (attention, GDN, MLP, embedding, RMSNorm, RoPE, LM head, decoder-layer) open a
**single-chip** `device` fixture (`ttnn.CreateDevice`), not a mesh. Since the 9B does not fit on one
Wormhole chip, **these tests only run on Blackhole P150** — they cannot run on N300 at all. The tests
below are the ones that actually exercise N300.

### Full-depth HF parity (real weights, all 32 layers, vs `transformers`) — `tests/unit/`

| Test | One-line detail |
| --- | --- |
| `test_prefill.py::test_full_depth_prefill_logits_pcc` | Full 32-layer `prefill_paged` last-position logits vs HF, real checkpoint |
| `test_decode.py::test_full_depth_decode_logits_pcc` | Full 32-layer teacher-forced decode steps (vLLM contract chain) vs HF, after prefill |

### Teacher-forced end-to-end — `tests/e2e/`

| Test | One-line detail |
| --- | --- |
| `test_teacher_forcing_e2e.py::test_teacher_forcing_e2e` | Top-1 / top-5 argmax agreement, TT vs HF, over a long teacher-forced run (default 128 decode steps) |
| `test_teacher_forcing_e2e.py::test_teacher_forcing_logits_pcc` | Full-vocab logit PCC at every teacher-forced step |

### Multi-device TP contract — `tests/` (top-level; default to 27B — set `HF_MODEL`/`MESH_DEVICE` for the 9B/N300)

| Test | One-line detail |
| --- | --- |
| `test_mlp_tp.py` | TP SwiGLU MLP: prefill + decode-shape, single layer, column/row-parallel + reduce-scatter |
| `test_attention_tp.py` | TP gated full attention: decode / prefill / paged-KV contract, single layer |
| `test_gdn_tp.py` | TP Gated DeltaNet: decode, chunk-prefill, batched prefill (batch capped at 2/4 by kernel row-mapping limit), state read/write, single layer |
| `test_rope_tp.py` | TP partial-rotary RoPE, prefill + decode lookup paths (op-level; synthetic q/k, real config dims) |
| `test_model_tp.py` | Full-model TP contract — **capped at 8 layers** (not the real 32), covers prefill/decode/batched/traced/bucketed variants |
| `test_generate_tp.py` | Full real model (all 32 layers), bespoke `generate_tp`, functional check only (first-token string match, non-degeneracy) — no PCC gate |
| `test_weight_mapping.py` | HF → internal weight-key remapping, pure CPU, no device |
| `test_sampling.py` | On-device RNG-seeded sampling correctness (mesh-parametrized; skips on single device) |
| `test_batched_row_agreement.py` | Host-only: batched-decode row-agreement logic at long context (pins the exact-match-vs-relaxed-check branch observed on 9B/N300 B=8) |

### Vision tower — `tests/` (mesh-parametrized, includes N300)

| Test | One-line detail |
| --- | --- |
| `test_vision_attention.py` | Single vision-attention block vs real checkpoint, layer 0 |
| `test_vision_block.py` | Every one of the 27 vision blocks vs real checkpoint, each checked independently (not one cascaded forward) |
| `test_vision_patch_embed.py` | Patch embedding vs real checkpoint weights, two grid sizes; plus a host/device-path self-consistency check |
| `test_vision_tower_pcc.py` | Checkpoint-free (config-init weights) smoke test: single block and the full cascaded 27-block tower |
| `test_patch_merger.py` | Patch merger vs real checkpoint |
| `test_model.py` / `test_wrapped_model.py` | Full vision tower vs real checkpoint at 1/2/27 layers — **the 27-layer case is skipped in CI**, only the 2-layer case runs there |

### Not applicable to N300

`tests/unit/*` component tests (Blackhole-only, see note above); `tests/test_prefill.py` and
`tests/test_decode_bucketing.py` (single-device / hardcoded to the 27B checkpoint respectively —
neither runs the 9B on N300).

## Commands

```bash
export HF_MODEL=Qwen/Qwen3.5-9B
export MESH_DEVICE=N300

# --- PCC / correctness ---

# Full-depth parity (the primary N300 correctness gate)
pytest models/demos/blackhole/qwen36/tests/unit/test_prefill.py \
       models/demos/blackhole/qwen36/tests/unit/test_decode.py -v -s

# Teacher-forced e2e (longer horizon, trend rather than a single sample)
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

# --- Demo ---

# All traced ISLs (128 .. 256k)
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced"

# A single short ISL
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128"
```

`text_demo.py`'s own docstring header still says "on Blackhole (P150 / P150x4)" and
`vision_demo.py`'s says "Qwen3.6-27B ... on Blackhole P150" — both are stale/incomplete: the code in
both files contains explicit N300 mesh-shape mappings (`"N300": (1, 2)`), and the model-level N300
results below were produced through this same prefill/decode code path.

Both demos **have** now been run end to end on N300/9B, every case `rc=0` — `text_demo.py` across
all 8 traced ISLs and `vision_demo.py` across all 5 multimodal cases. See "End-to-end demo run" and
"End-to-end vision demo run" below.

## End-to-end demo run — N300, Qwen3.5-9B

All 8 single-user traced ISLs, batch 1, greedy, one `pytest -k <case>` per row. Every case exited
`rc=0` with its output-quality assertions passing. Figures below are from the cold-`.rp`-cache run;
they reproduce two earlier full ladders to within 0.02 s TTFT and ±0.4 tok/s.

| ISL  | Gen tokens | TTFT     | Decode      | ms/tok |
| ---- | ---------- | -------- | ----------- | ------ |
| 128  | 50         | 0.24 s   | 22.84 tok/s | 43.8   |
| 4k   | 100        | 0.95 s   | 22.58 tok/s | 44.3   |
| 8k   | 100        | 1.95 s   | 22.37 tok/s | 44.7   |
| 16k  | 100        | 3.97 s   | 22.41 tok/s | 44.6   |
| 32k  | 100        | 8.33 s   | 22.71 tok/s | 44.0   |
| 64k  | 500        | 18.66 s  | 22.12 tok/s | 45.2   |
| 128k | 100        | 33.50 s  | 20.76 tok/s | 48.2   |
| 256k | 100        | 125.43 s | 18.15 tok/s | 55.1   |

Decode holds flat to 128k and costs ~21% by 256k (22.8 → 18.2 tok/s), the paged-KV attention term
growing with context. TTFT is near-linear in ISL through 128k, then steepens.

### Example inputs

The demo builds each ISL's prompt itself — there is no input file to pass on the command line.
Which corpus it draws from depends on the ISL (`_get_prompt` in `demo/text_demo.py`):

| ISL        | Prompt source                                                                              | Task given to the model            |
| ---------- | ------------------------------------------------------------------------------------------ | ---------------------------------- |
| 128 (≤256) | `models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json` | open question, free continuation   |
| 4k         | `demo/sample_prompts/input_data_long_4k.json`                                              | history-of-AI text, then summarize |
| 8k – 128k  | `demo/sample_prompts/eval_frankenstein_long.json` → Gutenberg *Frankenstein* (epub 84)     | "Based on the above text: …"       |
| 256k       | same file, entry 4 → Gutenberg *War and Peace* (epub 2600)                                 | "Based on the above text: …"       |

The 8k–256k entries hold a URL, not the text; the corpus downloads on first run and caches under
`demo/sample_prompts/.context_cache`. Prompts are clipped (never padded) to the ISL, so the
last-token logit is always a real token.

The ISL-128 prompt is entry 0 of the shared questions file, in full:

```text
What is your favorite condiment? There are so many condiments to choose from, each
bringing its unique flavor and texture to enhance different dishes. Do you prefer the
classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard,
or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy the
tangy zest of salsa or the smooth and savory taste of aioli. Share what your favorite
condiment is and why you love it. Does it remind you of a specific dish or meal?
```

Two env knobs change the prompt rather than the model: `QWEN35_REF_PROMPT=1` swaps ISLs ≥4k for the
reference 27B 64k extractive task, and `QWEN35_NO_THINK=1` emits an empty `<think>` block instead of
seeding one.

### Expected output

Decode is greedy (`QWEN35_TEMP` unset), so a given build is deterministic — the `determinism_128`
case asserts two runs of the same ISL emit identical tokens. Exact strings are **not** a contract:
they move with dtype, fidelity and kernel changes. Treat these as shape-and-sanity references.

ISL 128 (50 tokens) continues the prompt rather than answering it, which is correct for a base
continuation with no chat template applied:

```text
 bringing its unique flavor and texture to enhance different dishes. Do you prefer the
 classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard,
 or perhaps something more exotic like sriracha or hoisin sauce? Maybe
```

ISL 4k (100 tokens) opens a reasoning block, because the long-context prompts seed `<think>`:

```text
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A text about the history of artificial intelligence.
    *   Task: Provide a brief summary of the key milestones in the history of AI.
```

At 8k and above the expectation is topical, not literal — the output must mention the source text.
The demo logs which terms matched:

```text
64k :  output content check OK (Frankenstein, matched ['frankenstein', 'victor',
       'creature', 'monster', 'elizabeth', 'geneva', 'walton', 'margaret', …])
256k:  output content check OK (War and Peace, matched ['pierre', 'war and peace'])
```

## End-to-end vision demo run — N300, Qwen3.5-9B

`demo/vision_demo.py`, all 5 parametrized cases, every one `rc=0`. Images and the video are fetched
from public URLs declared in the prompt JSONs; the vision tower runs on device
(`DropInVisionTransformer`) and its embeddings are spliced into the text embeddings before prefill.

| Case                  | Prompt file                | Prompt tokens          | TTFT    | Decode      | Gen |
| --------------------- | -------------------------- | ---------------------- | ------- | ----------- | --- |
| `traced_single_image` | `vision_demo.json`         | 2770 (2752 image)      | 4.52 s  | 24.8 tok/s  | 419 |
| `paged_single_image`  | `vision_demo.json`         | 2770 (2752 image)      | 2.14 s  | 24.7 tok/s  | 300 |
| `traced_multi_image`  | `vision_multi_image.json`  | 5529 (5504 image)      | 6.64 s  | 24.6 tok/s  | 124 |
| `traced_video`        | `vision_video.json`        | 820 (728 video, 16 fr) | 0.76 s  | 24.9 tok/s  | 300 |
| `traced_text_only`    | `vision_text_only.json`    | 21 (tower skipped)     | 0.31 s  | 24.9 tok/s  | 100 |

Unlike the text ladder, `vision_demo.py` **does** carry a `paged_*` (untraced) case, and it is a
useful cross-check: traced and untraced produce the same description of the same image.

> **Traced TTFT is higher than untraced at this prompt length, and that is real, not noise.** The
> single-image prompt is 2770 tokens against `PREFILL_CHUNK = 2048`, so the traced path replays one
> full 2048-token chunk and then runs a 722-token tail through a padded masked bucket — two passes —
> where the untraced path prefills all 2770 in one. Just above a chunk boundary, chunk-outer traced
> prefill loses to a single-pass prefill. Trace capture itself is outside the TTFT timer in both
> paths, so it is not a measurement artifact.

### Example inputs (vision)

Each JSON holds one conversation in HF chat format. Media are remote URLs, fetched at run time:

| Prompt file               | Media                                                        | Question                                        |
| ------------------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| `vision_demo.json`        | 1 image — Qwen `demo.jpeg` (woman + golden retriever, beach)  | "Describe this image."                          |
| `vision_multi_image.json` | the **same** image twice                                      | "Identify the differences between these two images." |
| `vision_video.json`       | `space_woaudio.mp4`, `num_frames: 16`                         | "Describe this video."                          |
| `vision_text_only.json`   | none                                                          | "Who are you and what can you do?"              |

Video is decoded the way the HF reference does — `transformers`' `load_video` (pyav backend) into
`Qwen3VLProcessor` — not via `qwen_vl_utils`, whose video backends are unavailable here.

### Expected output (vision)

`traced_single_image` — the model describes the actual photograph in detail:

```text
This is a heartwarming, sun-drenched photograph capturing a tender moment between a young
woman and her golden retriever on a sandy beach at sunset.

**Scene & Setting:**
The setting is a wide, open beach with soft, undulating sand in the foreground. Gentle waves
roll in from the ocean in the background, meeting the shore under a bright, hazy sky...
```

`traced_multi_image` — both inputs are the same file, so "identical" is the correct answer; a model
handed bad embeddings would invent differences instead:

```text
After a careful comparison, it appears that the two images you have provided are identical.
```

`traced_video` — 16 frames through the same tower:

```text
This video features two astronauts inside the International Space Station (ISS), conducting a
live broadcast or educational session. The setting is a well-equipped module of the ISS, filled
with various pieces of scientific equipment and monitors displaying data and graphics.
```

`traced_text_only` — vision tower skipped entirely (the control case):

```text
I am **Qwen3.5**, the latest large language model developed by Tongyi Lab. I am designed to
assist with a wide range of tasks, from answering questions and creating content to solving
complex problems and analyzing data.
```

### What the demos do and do not validate

**`text_demo.py`** — `_assert_output_quality` is a **degeneracy detector, not a reference
comparison**. It asserts the text is non-empty, that no single token is more than 60% of the words,
that no 8-gram repeats more than 10 times, and — for the long-context ISLs only — that at least one
term from the source corpus appears. It never compares against HuggingFace.

**`vision_demo.py`** — meaningfully stricter, because a broken vision tower produces fluent text
about the wrong thing, which no degeneracy check would catch. `_assert_describes_input` requires the
output to actually describe the input (N-of-M expected terms per prompt file: 2 of
`beach/dog/woman/sand/sunset/shore/ocean/sea` for the single image, 1 of
`identical/same/no differences` for multi-image, 1 of `space/earth/planet/orbit/astronaut/star/spacecraft`
for video), **and** rejects a forbidden-phrase list — `corrupted`, `improperly rendered`,
`digital artifact`, `not a photograph`, `random noise` — the hallmarks of a model handed noise
instead of image embeddings. Measured matches on this run: single image 7 of 8 terms, multi-image
`identical`, video `space` + `astronaut`. It is still not a numeric comparison against HF.

Reference validation lives in the HF-parity tests, which drive the **same** `prefill_paged` + decode
chain the demo uses. Run those, not the demo, to answer "does this match the reference":

| Test                                    | Compares                                                     |
| --------------------------------------- | ------------------------------------------------------------ |
| `tests/unit/test_prefill.py`            | full-depth prefill logits vs `Qwen3_5ForCausalLM`             |
| `tests/unit/test_decode.py`             | full-depth teacher-forced decode logits vs HF                 |
| `tests/e2e/test_teacher_forcing_e2e.py` | top-1/top-5 token agreement + logit PCC vs HF over 128 steps  |

## PCC results — N300, Qwen3.5-9B

Gates in `tests/pcc_thresholds.json` unless noted; measured values are logged, not just gated.

| Test | Gate | Measured (N300) |
| --- | --- | --- |
| `test_full_depth_prefill_logits_pcc` (32 layers, 128-token prompt) | 0.98 | 0.998400 |
| `test_full_depth_decode_logits_pcc` (32 layers, 5 steps) | 0.95 | 0.9981, 0.9971, 0.9942, 0.9890, 0.9955 → min 0.989032, mean 0.994791 |
| `test_attention_tp` (single layer, decode pos0) | 0.97 | 0.99993 – 0.99996 |
| `test_attention_tp_prefill` (single layer, S=64) | 0.95 | 0.999231 |
| `test_attention_tp_paged_peruser` (B=8 / B=32) | 0.97 | 1.00000 |
| `test_gdn_tp` (single layer, decode pos0) | 0.92 | 0.99988 |
| `test_gdn_tp` per-user state (B=8) | 0.92 | 1.00000 |
| `test_mlp_tp` (single layer, decode) | 0.97 | 0.998515 |
| `test_mlp_tp` prefill T=2048, bf16 / bf8 / bf4 in | 0.97 | 0.999394 / 0.999268 / 0.985412 |
| `test_rope_tp` partial RoPE, prefill / decode (q, k) | 0.99 | ≥ 0.9999991 |
| Vision tower, depth 27 (real checkpoint, config gate) | 0.985 (`test_wrapped_model.py`) | 0.98850 |
| Vision tower, depth 27 (real checkpoint, after sequence-padding fix) | — | 0.99929 (`VISION_TOWER_PERF.md`) |
| Vision tower, depth 1 (real checkpoint) | 0.998 | 0.99981 |

The two full-depth gates (0.98 / 0.95) sit **below** the measured 9B/N300 numbers above — per the
primary README, they are regression detectors at this prompt length, not accuracy targets.

**Teacher-forced e2e, 9B / N300 (TP=2), 32 layers, 128-token prompt / 128 decode steps:**

| | top-1 vs HF | top-5 vs HF | logit PCC mean | worst step |
| --- | --- | --- | --- | --- |
| **Measured** | **93.80%** (CI 88.24–96.82) | **99.22%** (CI 95.74–99.86) | **0.992495** | **0.933855** |
| Floor in `_MEASURED_FLOORS` | 84.00% | 91.00% | — | — |

No gate is registered for this test in `pcc_thresholds.json` — it uses a separate, file-local
`_MEASURED_FLOORS` table instead (`tests/e2e/test_teacher_forcing_e2e.py`). That table still carries
an **older** 9B measurement in its provenance string (top-1 90.70%, top-5 96.12%, worst-step PCC
0.5763); the run above beats all three, worst-step by a wide margin. The floors are deliberately not
auto-updated — remeasure and add a row rather than editing one to make a case pass.

Of the 8 top-1 disagreements in 129 predictions, the test classifies **all 8 as near-ties** (the
reference's own top1−top2 margin averages 0.328 logits on them) and **zero** as confident flips.
Against ground truth the device is marginally ahead of the reference it is compared to: TT 72.09%
top-1, HF 71.32%.

> **All PCC numbers above are measured, not quoted from source docstrings.** They come from **three**
> completed 27-file suite runs on this N300 and reproduce identically across all three — the last of
> which was run with the `.rp` weight cache cleared, so the permuted-RoPE weights were rebuilt from
> the checkpoint rather than loaded (see Known limitations for why that distinction matters). That
> cold run was 27 files / 167 tests / 0 failures. Note that `full_depth_pcc_common.py`'s module
> docstring still records the older 0.9913 / min-0.9827 figures — that docstring is stale relative to
> the measured values in this table.

## Performance

**Vision tower — the one component with a committed, N300-specific performance writeup**
(`VISION_TOWER_PERF.md`), after the tuning pass documented there:

| Stage | Before | After (N300) |
| --- | --- | --- |
| Depth-1 window (one vision block, device time) | 11,876 µs | **6,887 µs** (1.72x) |
| Full tower device time | 52,477 µs | 45,343 µs (−13.6%, depth-1 window) |
| SDPA (8 heads/device, 256/512) | 20.62 ms | **15.29 ms** (1.35x) |
| Redundant data-movement pass | 18,119 µs → 14,430 µs (1.26x device) | 79.46 ms → 71.81 ms (−9.6% wall) |
| CCL workers (`wpl=4` vs `wpl=2`) | 5.77 ms | **4.69 ms** (−19%) |

See `VISION_TOWER_PERF.md` for the full per-pass writeup, including two negative results (things
tried and reverted) and the accuracy-vs-perf tradeoff analysis.

**Text decode (prefill/decode tok/s, TTFT) — see the "End-to-end demo run" table above** for the
measured, per-ISL N300/9B numbers (22.75 tok/s at ISL 128, holding flat to 128k, down to 18.43 tok/s
at 256k). Those are the first committed end-to-end throughput numbers for this device/checkpoint;
`text_demo.py` computes `ttft_s`/`decode_tok_s`/`agg_tok_s` live for every run
(`demo/text_demo.py:948-950`) but does not persist them itself — the table above was captured from
those logs. There is also a `validate_perf_targets.py` CI hook referenced in comments, but no CI runs
this model/device (see Known limitations), so no target file backs these numbers today. Below the
demo level, several individual op-level micro-benchmarks (e.g. specific matmul/conv timings measured
on N300) are documented inline in `tt/mlp.py`, `tt/gdn/tp.py`, `tt/attention/tp.py`, and
`tt/gdn/conv_fir_wh.py`.

## Known limitations (N300, 9B)

- **A warm weight cache hides the permuted-RoPE weight construction from every full-model test.**
  N300/9B enables permuted full-width RoPE (`tp_common.rope_permuted_enabled`, gated to
  `wh_9b_n300`), which folds a head_dim channel permutation into `q_proj`/`k_proj`/`q_norm`/`k_norm`
  at load time. The result is cached on disk as `*.rp*` tensorbins under
  `<snapshot>/N300/tensor_cache_bfp8_mesh1x2/layers.{3,7,11,15,19,23,27,31}/tp/`. When those files
  exist the model loads them and **never re-runs the permutation that produced them**, so a green
  full-model run does not prove that construction is correct. Only `tests/test_attention_tp.py`
  rebuilds it every run (it calls `load_attention_weights_tp` with `cache_dir=None`) — it is the
  single test that can catch a regression there, and it did catch one. To validate cold, move or
  delete those 16 `*.rp*` files (~520 MB) before running; they regenerate automatically.

- **On-device top-k/top-p sampling is unavailable on N300, and non-greedy sampling falls back to host
  more broadly than that.** The framework's `SamplingGenerator` requires
  `vocab / num_devices <= 65536`; at vocab 248,320 and 2 devices that's 124,160, so `self.sampling`
  is `None` on N300 and only greedy (`ttnn.argmax`) decoding runs on device (`tt/model.py:520-528`).
  This does not apply to Blackhole P150 (single device, same vocab, so the same math forces the same
  limit there too). Because `model.sampling is None` on N300, `demo/text_demo.py` unconditionally
  takes its host sampling path for **any** temperature/top-k/top-p request there (`_pick()`:
  `torch.where`/`softmax`/`topk`/`sort`/`multinomial`, every decode step) — not just when
  repetition-penalty or no-repeat-ngram are requested, which is also true on the Blackhole configs
  that otherwise have an on-device sampler. Batched decode (`QWEN36_BATCHED_DECODE_MODE`) likewise
  auto-downgrades from on-device per-shard argmax to full host `torch.argmax` whenever
  `model.sampling is None`, i.e. automatically on N300, not just when explicitly requested.
- **Batched GDN prefill is capped at batch size 2–4**, not the model's full serving batch — the
  `gated_delta_attn_seq` kernel maps one row to `B * Nv_tp`, which bounds how large a batch can be
  processed in one prefill call (`tests/test_gdn_tp.py:351`).
- **GDN decode batch-splits above B=16 at fp32 recurrent state on N300** specifically (more headroom
  at bf16/bf8) — `tt/gdn/tp.py:350-357` (re-verified against current HEAD `a8233578aec`; shifted by
  2 lines from an earlier rebase).
- **Batched demo generation can diverge above ~32k tokens of shared context.** All B users in a
  batched demo run share the same prompt and must decode identically up to the point where each
  user's KV cache position in the paged store causes SDPA's reduction order to differ per user; a
  near-tied logit can then flip argmax differently per user. Observed on 9B/N300 B=8: 64k context
  failed to match exactly in 1 of 6 runs (diverging after 29 of 50 compared tokens); 8k/16k/32k
  matched exactly and reproducibly across the runs checked (`tests/test_batched_row_agreement.py`).
- **There is no CI coverage at all for this device/checkpoint combination.** Checked
  `.github/workflows/` and `tests/pipeline_reorg/*.yaml` directly: the only CI wired for this model
  is "Qwen3.6 unit/e2e tests" on the `bh_quietbox_2` SKU (Blackhole P150x4), and it only ever sets
  `HF_MODEL=Qwen/Qwen3.6-27B` — there is no CI entry anywhere for `Qwen/Qwen3.5-9B`, and no CI entry
  for N300, T3K, single-card P150, or P150x8. Its e2e job is also marked `release_ready: false`.
  Everything in this document (including the vision full-27-layer real-checkpoint test being
  skipped in CI — it only runs the 2-layer case, `tests/test_model.py` / `test_wrapped_model.py`,
  `is_ci_env` skip) has been validated by manual/local runs on this N300, not by CI.
- **`test_model_tp.py` (the "full-model TP contract" suite) is capped at 8 layers**, not the real 32
  — see the Test cases table above.
- Two demo docstrings (`text_demo.py`, `vision_demo.py`) still say "Blackhole" only; treat that as
  stale documentation, not as evidence N300 is unsupported (see the Commands section note above).
