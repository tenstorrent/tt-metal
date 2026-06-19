# Mistral-Small-4-119B (Multimodal) on Tenstorrent

TTNN implementation of [`mistralai/Mistral-Small-4-119B-2603`](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) — a 119B-parameter (6.5B active/token) Mixture-of-Experts, vision-language model — running end to end on Tenstorrent Blackhole hardware.

The model takes **text + image** input and generates text. It pairs a Pixtral vision tower with a Multi-head Latent Attention (MLA) + MoE language decoder.

## Model architecture

```
Image ─▶ Pixtral Vision Tower (24 layers) ─▶ Multimodal Projector ─┐
                                              (spatial-merge ×2,    │
                                               GELU)                ▼
                                                         ┌──────────────────────┐
Text tokens ────────────────────── embed ──────────────▶│  Language Decoder     │──▶ logits ─▶ token
                                                         │  36 layers:           │
                                                         │   • MLA self-attn     │
                                                         │   • MoE (128 experts, │
                                                         │     4 active + 1      │
                                                         │     shared)           │
                                                         └──────────────────────┘
```

| Component | Details |
|-----------|---------|
| **Language decoder** | 36 layers, hidden size 4096, 32 attention heads, vocab 131072 |
| **Attention** | MLA — `q_lora_rank=1024`, `kv_lora_rank=256`, `qk_rope=64`, `qk_nope=64`, `v_head_dim=128`; latent (compressed) KV cache |
| **MoE** | 128 experts, 4 active + 1 always-on shared expert; expert intermediate size 2048 |
| **Vision tower** | Pixtral-style ViT, 24 layers, hidden 1024, 16 heads, patch size 14, 2D RoPE |
| **Projector** | Spatial-merge (size 2) + GELU projection to the text hidden dim |
| **RoPE θ** | text `1e6`, vision `1e4` |
| **License** | Apache 2.0 |

Architecture constants live in [constants.py](constants.py).

## Supported hardware

**P150×8 only** (a **Blackhole Loud Box** — 8× P150, `MeshShape(1, 8)`). Set `MESH_DEVICE=P150x8` before running:

| `MESH_DEVICE` | Mesh shape | Notes |
|---------------|-----------|-------|
| `P150x8` | (1, 8) | the only supported target — 8× BlackHole P150, tensor-parallel |

## Setup

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
```

**Dependencies:** Mistral-4 support is not in a released `transformers` yet — install from source (the `mistral4` / `mistral3` / `pixtral` modeling files the model imports come from this):

```bash
uv pip install git+https://github.com/huggingface/transformers.git
```

The model is gated on Hugging Face — request access on the model page, then authenticate so weights download on first run:

```bash
huggingface-cli login
```

Weights, config, tokenizer, and image processor are pulled from `HF_MODEL_ID` automatically (no manual repack step).

## Running the demo

[demo/demo_multimodal.py](demo/demo_multimodal.py) drives the full vision → projector → prefill → decode pipeline. Loading is unified: vision + projector + text are lazy-loaded together on the first `encode_image` call.

The input image is fixed to a built-in sample (a battle scene) and is not user-supplied; only the prompt is configurable.

```bash
export MESH_DEVICE=P150x8

# Full model (36 text + 24 vision layers) on the built-in sample image
python models/experimental/mistral_small_4_119b/demo/demo_multimodal.py \
    --image-max-side 512 \
    --n-text-layers 36 \
    --n-vision-layers 24 \
    --max-new-tokens 512

# Change the prompt
python models/experimental/mistral_small_4_119b/demo/demo_multimodal.py \
    --image-max-side 512 \
    --n-text-layers 36 \
    --n-vision-layers 24 \
    --max-new-tokens 512 \
    --prompt "Describe the scene."
```

> The defaults run only **2 text + 2 vision layers** — fast (~minutes) but a plumbing check only, so the output is gibberish. Pass the full-layer flags above for real generation.

Key flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--random-image` | off | Use a random pixel_values tensor instead of the built-in sample (pipeline-only) |
| `--prompt` | (built-in) | Text prompt accompanying the image |
| `--max-new-tokens` | 16 | Tokens to generate after prefill |
| `--n-text-layers` | 2 | Text decoder layers (1–36) |
| `--n-vision-layers` | 2 | Pixtral vision layers (1–24) |
| `--image-max-side` | 224 | Max image H/W (px) before the processor; larger = better, slower |
| `--no-chat-template` | off | Use raw `[image]+[prompt]` layout (plumbing only) |
| `--no-trace` | off | Disable decode-step trace capture (eager path; slower, for debugging) |
| `--backend` | `ttnn` | `ttnn`, `hf` (Torch reference), or `both` (compare) |

## Tests

Tests need hardware and download weights. Each parametrizes the `mesh_device` fixture, so set `MESH_DEVICE=P150x8` too. The text **prefill** and **decode** PCC tests run the **full 36-layer** decoder by default and are marked `@pytest.mark.slow`; the other PCC/smoke harnesses are gated behind per-test env flags.

**Correctness (PCC vs HF Torch reference):**

```bash
# Full 36-layer text decoder — no env gate, marked @pytest.mark.slow.
# Override layer count for fast local iteration: MISTRAL4_PREFILL_N_LAYERS=2 / MISTRAL4_DECODE_N_LAYERS=2
pytest models/experimental/mistral_small_4_119b/tests/test_text_prefill_pcc.py
pytest models/experimental/mistral_small_4_119b/tests/test_text_decode_pcc.py

MISTRAL4_VISION_PCC=1  pytest models/experimental/mistral_small_4_119b/tests/test_vision_tower_pcc.py
MISTRAL4_MMP_PCC=1     pytest models/experimental/mistral_small_4_119b/tests/test_multimodal_projector_pcc.py
MISTRAL4_MM_PCC=1      pytest models/experimental/mistral_small_4_119b/tests/test_multimodal_pcc_unified.py
```

**Measured PCC (BH Loudbox / P150×8, prompt `"The capital of France is"`):**

| Test | Layers | PCC | Notes |
|------|-------:|----:|-------|
| Text prefill (vs HF) | 36 | **0.933** | overall flattened; mean per-position 0.911; pos 0 = 0.717 (degenerate causal context), pos 1–4 ≥ 0.949; greedy token match 4/5 (only pos 0 differs) |
| Text decode (vs HF) | 36 | **0.961** | decode logits vs HF reference at pos 4; greedy token matches (HF = prefill = decode = `6993`) |
| Text decode (self-consistency) | 36 | **0.992** | decode-vs-prefill logits at pos 4 — validates the decode path reads/uses the KV cache that prefill wrote |
| Unified multimodal (vs HF) | 36 + 24 | **~0.855** | full vision → projector → language path (consistent with aggressive `bfloat4_b`/`bfloat8_b` weight quantization); PCC ≈ 0.839 / 0.842 at 4K / 16K context (**16K is the highest verified** — see below) |

The decode test runs **two** checks: decode-vs-HF (ground truth, 0.961) and decode-vs-prefill self-consistency (0.992). The self-consistency check isolates the decode path itself — in bfloat16, TTNN matmul kernels are shape-specific, so a 1-token decode kernel produces slightly different K/V at a position than the seq-len prefill kernel; the self-consistency PCC is free of that effect, while the HF PCC (floor 0.90, measured 0.961) folds it in on top of quantization loss.

`test_multimodal_pcc_unified.py` loads vision (bf8) + text (bf16) co-resident on device, runs `encode_image` → `prefill_multimodal_full_logits`, and compares per-position + flattened logits against a chunked HF Torch reference (`MISTRAL4_MM_PREFILL_CHUNK`, default 2048; DynamicCache keeps attention memory at O(chunk × past)). It is parametrized over `(num_text_layers, num_vision_layers, max_text_tokens)` — ids `L1V1`, `L36V24`, `L36V24_4096`, `L36V24_16384`, `L36V24_65536`, `L36V24_131072`, `L36V24_262144` — selected with `-k` (e.g. `-k L36V24_16384`); `max_text_tokens` pads the prompt with coherent English filler up to that count (image tokens on top). The pass/fail gate is flattened-logits PCC ≥ 0.80; greedy token match is logged for diagnostics only. **16K context is the highest that completes verification** — `L36V24_65536` and larger are kept (marked `@pytest.mark.slow`) as a safeguard but in practice do not finish on a CPU host: a 64K run was killed after **> 1 h without completing**, bottlenecked on the chunked HF reference forward. On-device prefill/decode scale further (see the perf table) but are not PCC-verified past 16K.

**Smoke / plumbing (no PCC, fast):**

```bash
MISTRAL4_VISION_SMOKE=1    pytest models/experimental/mistral_small_4_119b/tests/test_vision_tower_smoke.py
MISTRAL4_PREFILL_SMOKE=1   pytest models/experimental/mistral_small_4_119b/tests/test_text_prefill_smoke.py
MISTRAL4_DECODE_SMOKE=1    pytest models/experimental/mistral_small_4_119b/tests/test_text_decode_smoke.py
MISTRAL4_LANG_DEMO_SMOKE=1 pytest models/experimental/mistral_small_4_119b/tests/test_language_demo_smoke.py
```

**Performance:**

```bash
# End-to-end wall-clock (TTFT, prefill tok/s, steady-state decode tok/s/user)
pytest models/experimental/mistral_small_4_119b/tests/perf/test_e2e_performant.py -m models_performance_bare_metal -k L36V24

# Per-op device performance (single layer)
pytest models/experimental/mistral_small_4_119b/tests/perf/test_perf.py -m models_device_performance_bare_metal

# Single-layer prefill+decode device perf (Tracy per-op CSV/JSON; text decoder only, no vision)
pytest models/experimental/mistral_small_4_119b/tests/perf/test_device_perf_single_layer_prefill_decode.py \
    -m models_device_performance_bare_metal
```

The e2e perf test isolates steady-state timing (compile/load passes excluded). Decode is trace-captured and replayed over 2 command queues (`decode_next_token_2cq`); prefill is not trace-captured.

**Single-layer prefill+decode device perf** ([test_device_perf_single_layer_prefill_decode.py](tests/perf/test_device_perf_single_layer_prefill_decode.py)) profiles **one text decoder layer** (the **vision tower is not involved** — it builds `TtMistral4TextModel` directly on text-only weights). It holds two tests:

- `test_profile_single_layer_prefill_decode` — the Tracy *workload*: builds a 1-layer model, runs one warmup iteration, then prefills 128 tokens and runs one decode step at position 128 inside a `start`/`stop` signpost window (prefill before decode so per-op tooling can split the two phases). No PCC assertion.
- `test_device_perf_single_layer_prefill_decode` — the *wrapper* (the normal entry point above): runs the workload under Tracy via `run_device_perf` (shells out to the profile test by node id), then writes `device_perf_*.csv` + a partial benchmark JSON via `prep_device_perf_report`. No golden perf assertion.

Capture the workload standalone under Tracy:

```bash
python -m tracy -p -v -r --dump-device-data-mid-run \
    pytest models/experimental/mistral_small_4_119b/tests/perf/test_device_perf_single_layer_prefill_decode.py::test_profile_single_layer_prefill_decode -v
```

Analyze the generated ops CSV with `tt-perf-report`. The file carries `start`/`stop` signposts; pass both explicitly (default mode anchors only on the last signpost and shows no device ops):

```bash
tt-perf-report generated/profiler/mistral_small_4_119b_L1_prefill_decode/reports/.../ops_perf_results_*.csv \
    --start-signpost start --end-signpost stop
```

**Measured end-to-end performance (BH Loudbox / P150×8, 1×8 mesh, full 36 text + 24 vision layers, 2CQ traced decode):**

| ISL (tokens) | TTFT (ms) | Prefill tok/s | Decode tok/s/user |
|-------------:|----------:|--------------:|------------------:|
| 128 | 475.8 | 301.5 | 16.99 |
| 4096 | 3,392.5 | 1,225.5 | 16.32 |
| 16384 | 12,609.6 | 1,305.2 | 15.59 |
| 65536 | 93,708.9 | 699.8 | 13.32 |
| 131072 | 312,653.4 (5.2 min) | 419.3 | 11.27 |
| 262144 | 1,218,595.8 (20.3 min) | 215.1 | 8.50 |

All rows: L36V24, 25 image tokens in the prompt.

- **ISL** — input sequence length (prompt tokens, including the 25 image tokens)
- **TTFT** — vision replay + one prefill replay (time to first decode logits)
- **Prefill tok/s** — `prompt_len / prefill_trace_replay_time` (compile pass excluded)
- **Decode tok/s/user** — steady-state decode: `decode_iters / decode_trace_replay_total` (compile + capture excluded)

## Directory layout

```
mistral_small_4_119b/
├── README.md                       # this file
├── constants.py                    # architecture dimensions / HF model id
├── demo/
│   └── demo_multimodal.py          # end-to-end multimodal generation demo
├── tt/                             # TTNN (Tenstorrent) implementation
│   ├── mistral3_for_conditional_generation_unified.py  # top-level orchestrator (unified load)
│   ├── mistral3_for_conditional_generation_helpers.py
│   ├── mistral4_text_model.py      # text decoder: embed, decode path, lm_head
│   ├── mistral4_text_prefill.py    # text prefill path
│   ├── mistral4_self_attention.py  # MLA self-attention (latent KV cache)
│   ├── mistral4_moe.py             # MoE: 128 experts, 4 active + shared
│   ├── mistral4_multimodal_projector.py
│   ├── mistral4_vision_tower.py    # Pixtral vision tower
│   ├── mistral4_vision_attention.py
│   ├── mistral4_vision_mlp.py
│   ├── mistral4_vision_rope.py     # vision 2D RoPE
│   └── vision_matmul_config.py     # vision matmul program configs
└── tests/
    ├── mesh_param.py               # MESH_DEVICE → mesh-shape resolution
    ├── test_*_pcc.py               # PCC correctness vs HF Torch
    ├── test_*_smoke.py             # fast plumbing checks
    └── perf/                       # e2e + device-perf tests
```
