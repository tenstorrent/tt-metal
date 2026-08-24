# Qwen3.5 / Qwen3.6 on Blackhole

This directory implements Tenstorrent Blackhole inference for the hybrid
**Gated DeltaNet + Gated Full Attention** Qwen3.5/3.6 family. The same code path serves three checkpoints:

| Model            | `HF_MODEL`             | Mesh / `MESH_DEVICE` | Parallelism            |
| ---------------- | ---------------------- | -------------------- | ---------------------- |
| Qwen3.5-9B       | `Qwen/Qwen3.5-9B`      | single P150 — `P150` | single device          |
| Qwen3.5-27B      | `Qwen/Qwen3.5-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x8 — `P150x8`    | 8-way tensor parallel  |

- The **9B** runs on a **single Blackhole P150** device. It uses the validated
  single-device forward path (no collectives).
- The **27B** variants (both Qwen3.5-27B and Qwen3.6-27B) run on a **P150x4**
  (a `(1, 4)` Blackhole mesh) using **4-way tensor parallelism (TP)**. The TP
  path needs `FABRIC_1D` for the cross-device collectives (all-reduce /
  reduce-scatter) and a trace region for the captured chunk-outer prefill trace.
- **Qwen3.6-27B additionally runs at TP=8** on a `(1, 8)` mesh (`P150x8`).
  Because it has only **4 KV heads**, TP=8 cannot give each device its own head:
  each head is instead **replicated across the device pair holding its GQA query
  group** (devices 0-1 share KV head 0, 2-3 head 1, and so on), so
  `n_local_kv_heads` is 1 at both TP=4 and TP=8 and the whole runtime KV path is
  unchanged. See `tp_common.replicate_kv_weight` and
  `ModelArgs.SUPPORTS_KV_REPLICATION`.

Everything model-specific (hybrid layer dispatch, DeltaNet head/conv dims,
partial rotary factor, vocab, layer count) is read from the parsed HF config, so
the single code base adapts to each checkpoint. The device count alone
(`num_devices > 1`) switches between the single-device and TP code paths — see
`tt/model_config.py` and `tt/tp_common.py`.

## Architecture

Assembly: `tok_embeddings → N × Qwen36DecoderLayer → RMSNorm → LM Head`.

Each model interleaves two attention block types (read from the HF
`layer_types`): **Gated DeltaNet** (linear-attention, recurrent + causal conv
state) layers and **Gated Full Attention** (paged KV cache) layers. The 9B has
32 layers (24 DeltaNet + 8 full-attention). Qwen3.5 uses zero-centered RMSNorm
everywhere and **partial** RoPE (only a fraction of each head is rotated).

## Environment setup

Before running **any** test, export the two environment variables that select
the checkpoint and the device mesh.

**9B (single P150):**

```bash
export HF_MODEL=Qwen/Qwen3.5-9B
export MESH_DEVICE=P150
```

**27B (P150x4):**

```bash
# Qwen3.6-27B
export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=P150x4

# …or Qwen3.5-27B
export HF_MODEL=Qwen/Qwen3.5-27B
export MESH_DEVICE=P150x4
```

`HF_MODEL` is the single source of truth for the checkpoint — it may be a Hugging
Face hub id (resolved via `snapshot_download`) or a local checkpoint directory.
`MESH_DEVICE` selects the mesh shape (`P150` → `(1,1)`, `P150x4` → `(1,4)`,
`P150x8` → `(1,8)`).

Optional flags:

```bash
# Run SDPA in BF8 (faster; slightly lower precision).
export QWEN_SDPA_BF8=1
```


## End-to-end demo test (`demo/text_demo.py`)

The e2e text-generation test lives in `demo/text_demo.py`. It is a single
parametrized test (`test_demo_text`) covering a range of input sequence lengths
(ISLs): 128, 4k, 8k, 16k, 32k, 64k, 128k, and 256k tokens. Each ISL runs prefill
+ decode and validates output (non-degenerate generation) and per-ISL
performance gates (TTFT and decode tok/s).

Two execution variants exist per ISL, identified by the test id prefix:

- **`traced_*`** — captures the prefill (chunk-outer) and decode forward passes
  as device traces and replays them. This is the **preferred** path and the one
  vLLM serves; run these by default.
- **`paged_*`** — non-traced paged path, useful as an eager reference/fallback.

Run the preferred traced cases (the env vars above must already be exported):

```bash
# All traced ISLs
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced"

# A single ISL, e.g. the short 128-token traced case
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128"

# Medium / long traced ISLs
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_4k"
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_64k"
```

The **same command works for both 9B and 27B** — only the exported `HF_MODEL` /
`MESH_DEVICE` differ. On a single device the test takes the validated 9B path; on
the `(1,4)` mesh it routes through the TP chunk-outer traced prefill + paged
traced decode path automatically.

> Long-context cases (64k+) download a public-domain corpus (Frankenstein, War
> and Peace) on first run and cache it under `demo/sample_prompts/.context_cache`.

## Tests

There are three tiers of tests under `tests/`: single-device component PCC, the
multi-device TP contract, and full-depth parity against HuggingFace.

### Single-device unit / component tests — **9B**

These run on a single P150 against the 9B checkpoint (the conftests
`setdefault HF_MODEL=Qwen/Qwen3.5-9B`). They validate each component's forward
pass against a torch reference (PCC) — see `tests/pcc_thresholds.json` for the
per-test thresholds.

`tests/unit/` (component PCC vs torch):

| Test                       | Validates                                            |
| -------------------------- | ---------------------------------------------------- |
| `test_embedding.py`        | token embedding                                      |
| `test_rms_norm.py`         | zero-centered RMSNorm (the "+1" fold)                |
| `test_rope.py`             | partial-rotary RoPE (host freqs + on-device lookup)  |
| `test_mlp.py`              | single-device SwiGLU MLP (layer 0)                   |
| `test_attention.py`        | single-device gated full attention (layer 3)         |
| `test_gdn.py`              | single-device Gated DeltaNet (layer 0)               |
| `test_lm_head.py`          | LM head logits (bf8 vs bf16)                          |
| `test_layer.py`            | full decoder-block sanity (no NaN/Inf, non-constant) |
| `test_model.py`            | Generator decode contract: traced vs paged decode    |
| `test_prefill.py`          | full-depth prefill logits vs HF (see below)          |
| `test_decode.py`           | full-depth decode logits vs HF (see below)           |
| `test_substate.py`         | weight `substate` helper (pure CPU, no device)       |

`tests/` (single-device, also 9B):

| Test                     | Validates                                                  |
| ------------------------ | ---------------------------------------------------------- |
| `test_prefill.py`        | masked fixed-bucket + chunk-outer prefill vs `prefill_paged` |
| `test_weight_mapping.py` | HF → internal weight key remapping (pure CPU)              |

Run the 9B unit suite (with `HF_MODEL=Qwen/Qwen3.5-9B`, `MESH_DEVICE=P150`):

```bash
pytest models/demos/blackhole/qwen36/tests/unit/ -v -s
pytest models/demos/blackhole/qwen36/tests/test_prefill.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_weight_mapping.py -v -s
```

> `test_prefill.py` auto-skips cases longer than `--max-prefill` (default 8192).
> Raise it to exercise long-context prefill, e.g. `--max-prefill 131072`.

### Tensor-parallel tests — **27B (P150x4)**

The `*_tp` tests exercise the multi-device TP path and default to the 27B
checkpoint. They must run on the `(1,4)` mesh with `FABRIC_1D` (the
`parametrize_mesh_tp` helper wires this from `MESH_DEVICE`). PCC thresholds are in
`tests/pcc_thresholds.json`.

| Test                  | Validates                                                            |
| --------------------- | ------------------------------------------------------------------- |
| `test_mlp_tp.py`      | TP SwiGLU MLP (column/row-parallel + reduce-scatter)                |
| `test_attention_tp.py`| TP gated full attention: decode / prefill / paged-KV contract       |
| `test_gdn_tp.py`      | TP Gated DeltaNet: decode + chunk-prefill                           |
| `test_model_tp.py`    | full-model TP contract: paged+traced path matches the bespoke oracle |
| `test_generate_tp.py` | full-model bespoke `generate_tp` on a real prompt (answer oracle)   |

Run the 27B TP suite (with `HF_MODEL=Qwen/Qwen3.6-27B` or `Qwen/Qwen3.5-27B`,
`MESH_DEVICE=P150x4`):

```bash
pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_model_tp.py -svq
pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py -v -s
```

### Full-depth HF parity — **both models**

`tests/unit/test_prefill.py` and `tests/unit/test_decode.py` are the only tests that
run **every** layer with real weights and compare against the HuggingFace reference
(`Qwen3_5ForCausalLM`) instead of against another TT path. Both share
`tests/unit/full_depth_pcc_common.py` (mesh + device params, full-depth model build,
paged KV, prompt, HF forward) and both cover either checkpoint — `HF_MODEL` picks the
model, `MESH_DEVICE` the mesh.

| Test                   | Validates                                                                        |
| ---------------------- | -------------------------------------------------------------------------------- |
| `unit/test_prefill.py` | full-depth `prefill_paged` last-position logits vs HF                            |
| `unit/test_decode.py`  | full-depth teacher-forced decode steps (vLLM contract chain) vs HF, after prefill |

Each decode step is fed HF's own argmax token, so one step's PCC does not inherit an
earlier greedy divergence — while still carrying the paged KV and GDN recurrent/conv
state the previous steps advanced.

```bash
# 9B (32 layers)
HF_MODEL=Qwen/Qwen3.5-9B MESH_DEVICE=P150 \
  pytest models/demos/blackhole/qwen36/tests/unit/test_prefill.py \
         models/demos/blackhole/qwen36/tests/unit/test_decode.py -v -s

# 27B (64 layers)
HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=P150x4 \
  pytest models/demos/blackhole/qwen36/tests/unit/test_prefill.py \
         models/demos/blackhole/qwen36/tests/unit/test_decode.py -v -s
```

Measured on Wormhole (128-token prompt, bf16 reference): 9B/N300 prefill 0.9913,
decode min 0.9827; 27B/T3K prefill 0.9957, decode min 0.9595. The gates
(`test_full_depth_prefill_logits_pcc` 0.98, `test_full_depth_decode_logits_pcc` 0.95
in `pcc_thresholds.json`) sit below the worse of the two — they are regression
detectors at this prompt length, not accuracy targets; the per-step PCC and
argmax/top-5 lines the tests log are the finer signal. Knobs:
`QWEN36_FULL_DEPTH_PROMPT_LEN` (default 128, keep it a multiple of 128),
`QWEN36_FULL_DEPTH_DECODE_STEPS` (5), `QWEN36_FULL_DEPTH_REF_DTYPE` (bfloat16).

### Teacher-forced e2e — **both models**

`tests/e2e/test_teacher_forcing_e2e.py` runs the real generation path — `prefill_paged`
then the demo's decode chain — but feeds the **ground-truth** token at every step, so
each step is an independent measurement instead of one diverged sample. Ground truth is
*A Tale of Two Cities* (the tt_transformers corpus); override with `QWEN36_TF_TEXT_FILE`.

| Test                             | Validates                                                     |
| -------------------------------- | ------------------------------------------------------------- |
| `test_teacher_forcing_e2e`       | top-1 / top-5 token accuracy, TT vs HF and both vs truth      |
| `test_teacher_forcing_logits_pcc`| full-vocab logit PCC at every teacher-forced step             |

The unit tests gate 5 decode steps; this runs 128+ and reports the **trend**, which is
the failure mode a short test structurally cannot see. Beyond the raw rates it prints a
flip classification (by the reference's own top1−top2 margin), position-trend bins beside
HF's own accuracy/margin/entropy, Wilson intervals next to each floor, and KL +
max|Δlogit| over HF's top-32.

```bash
# 9B
HF_MODEL=Qwen/Qwen3.5-9B MESH_DEVICE=P150 \
  pytest models/demos/blackhole/qwen36/tests/e2e/test_teacher_forcing_e2e.py -sv --timeout=0

# 27B
HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=P150x4 \
  pytest models/demos/blackhole/qwen36/tests/e2e/test_teacher_forcing_e2e.py -sv --timeout=0
```

Measured on Wormhole at prefill_128 / max_new_tokens_128 (bf16 reference):

| Model | top-1 vs HF | top-5 vs HF | logit PCC mean | worst step |
| ----- | ----------- | ----------- | -------------- | ---------- |
| 9B / N300 (TP=2)  | 90.70% | 96.12% | 0.9674 | 0.5763 |
| 27B / T3K (TP=8)  | 75.97% | 84.50% | 0.8403 | 0.3957 |

> **Open finding.** The 27B agrees with HF markedly less than the 9B here (75.97% vs
> 90.70% top-1) and this is **not yet explained**. Ruled out by measurement: the reference
> method (HF chunked vs recurrent agree to 0.9997, no position trend), TP width (TP=4 and
> TP=8 identical bin for bin), the stale-GDN-state bug in `prefill_paged` (numbers unchanged
> after that fix), and sequence length (a 128-token prefill scores *worse* than a 256-token
> one — PCC here tracks which row is scored, not length). The corpus explains ~4 points:
> starting past the book's front matter (`QWEN36_TF_PREFILL_LEN=512`) gives 79.84%. The
> unexplained part is the trend — agreement falls 96.97% → 43.33% across position bins while
> the *reference* grows more decisive (margin 7.75 → 9.75, entropy → 0.05), the opposite of a
> text-difficulty explanation. Floors in `_MEASURED_FLOORS` record this state so a change is
> visible; they are not a statement that it is acceptable.

> `test_substate.py` and `test_weight_mapping.py` are pure-CPU and need no device.
> `test_weight_mapping.py`'s shape constants assume the 9B checkpoint.
