# Qwen3.5 / Qwen3.6 on Blackhole

This directory implements Tenstorrent Blackhole inference for the hybrid
**Gated DeltaNet + Gated Full Attention** Qwen3.5/3.6 family. A single code
path serves four checkpoints:

| Model            | `HF_MODEL`             | Mesh / `MESH_DEVICE` | Parallelism            |
| ---------------- | ---------------------- | -------------------- | ---------------------- |
| Qwen3.5-9B       | `Qwen/Qwen3.5-9B`      | single P150 — `P150` | single device          |
| Qwen3.5-27B      | `Qwen/Qwen3.5-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x8 — `P150x8`    | 8-way tensor parallel  |
| Qwen3.6-35B-A3B  | `Qwen/Qwen3.6-35B-A3B` | P150x4 — `P150x4`    | 4-way TP + sparse MoE  |

The **35B-A3B** is the sparse Mixture-of-Experts member of the family (`qwen3_5_moe`:
256 routed experts, top-8, plus a gated shared expert on every layer). Every layer's
dense SwiGLU MLP is replaced by the sparse MoE block in `tt/moe/`; dispatch is
config-driven (`args.is_moe_layer`), so on the dense 9B/27B `num_experts == 0` and the
dense MLP path is byte-for-byte unchanged.

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

**35B-A3B (P150x4, sparse MoE):**

```bash
export HF_MODEL=Qwen/Qwen3.6-35B-A3B
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

The **same command works for 9B, 27B, and 35B-A3B** — only the exported `HF_MODEL` /
`MESH_DEVICE` differ. On a single device the test takes the validated 9B path; on
the `(1,4)` mesh it routes through the TP chunk-outer traced prefill + paged
traced decode path automatically (the sparse MoE block is selected per layer from
the config, transparent to the demo).

> Long-context cases (64k+) download a public-domain corpus (Frankenstein, War
> and Peace) on first run and cache it under `demo/sample_prompts/.context_cache`.

## Tests

There are two tiers of tests under `tests/`.

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
| `test_moe.py`              | single-device sparse MoE MLP (MoE checkpoint only; decode + prefill) |
| `test_attention.py`        | single-device gated full attention (layer 3)         |
| `test_gdn.py`              | single-device Gated DeltaNet (layer 0)               |
| `test_lm_head.py`          | LM head logits (bf8 vs bf16)                          |
| `test_layer.py`            | full decoder-block sanity (no NaN/Inf, non-constant) |
| `test_model.py`            | Generator decode contract: traced vs paged decode    |
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
| `test_moe_tp.py`      | TP sparse MoE MLP (router + experts + shared; MoE checkpoint only; decode + prefill) |
| `test_attention_tp.py`| TP gated full attention: decode / prefill / paged-KV contract       |
| `test_gdn_tp.py`      | TP Gated DeltaNet: decode + chunk-prefill                           |
| `test_model_tp.py`    | full-model TP contract: paged+traced path matches the bespoke oracle |
| `test_generate_tp.py` | full-model bespoke `generate_tp` on a real prompt (answer oracle)   |
| `test_mtp_tp.py`      | MTP drafter head PCC vs torch (prefill and one decode step)         |
| `test_spec_lossless.py` | spec decode matches plain greedy, token for token                 |
| `test_spec_determinism.py` | spec decode is identical across repeat runs                     |
| `test_mtp_torch_ref.py` | host MTP reference (pure CPU)                                      |

Run the 27B TP suite (with `HF_MODEL=Qwen/Qwen3.6-27B` or `Qwen/Qwen3.5-27B`,
`MESH_DEVICE=P150x4`):

```bash
pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_model_tp.py -svq
pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_mtp_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_spec_lossless.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_spec_determinism.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_mtp_torch_ref.py -v -s
```

> The MoE-specific tests (`test_moe_tp.py`, and the MoE path in `test_model_tp.py` /
> `test_generate_tp.py`) require the sparse checkpoint — run them with
> `HF_MODEL=Qwen/Qwen3.6-35B-A3B MESH_DEVICE=P150x4`. On the dense 27B they are inert
> (`num_experts == 0`), and the dense/MoE checkpoints must not be mixed in one run
> (each test file `setdefault`s or expects a single `HF_MODEL`).
> `test_substate.py` and `test_weight_mapping.py` are pure-CPU and need no device.
> `test_weight_mapping.py`'s shape constants assume the 9B checkpoint.

## Wormhole

The same code runs on Wormhole, with two validated configurations. Blackhole-only fusions are
gated off (`is_blackhole()`), and the GDN kernels pick Wormhole-appropriate defaults for the
chunk-seq activation memory, the chunk output dtype, and the conv FIR padding layout.

| Config | Device | Mesh | `HF_MODEL` | `MESH_DEVICE` |
| --- | --- | --- | --- | --- |
| 9B | N300 | 1x2 | `Qwen/Qwen3.5-9B` | `N300` |
| 27B | T3K | 1x8 | `Qwen/Qwen3.6-27B` | `T3K` |

```bash
# 9B on N300 (no MTP: permuted RoPE falls back to plain decode)
HF_MODEL=Qwen/Qwen3.5-9B MESH_DEVICE=N300 \
  pytest models/demos/blackhole/qwen36/demo/text_demo.py -k traced_128 -v -s

# 27B on T3K, plain decode
HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=T3K QWEN36_SPEC=0 \
  pytest models/demos/blackhole/qwen36/demo/text_demo.py -k "traced_128 and not traced_128k" -v -s

# 27B on T3K, MTP speculative decode (demo default is K=7)
HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=T3K QWEN36_SPEC=1 \
  pytest models/demos/blackhole/qwen36/demo/text_demo.py -k "traced_128 and not traced_128k" -v -s
```

### Performance

Warm trace replay, batch 1 greedy. TTFT is wall clock including prefill and the first token.
The 9B has no MTP path. The 27B MTP columns are speculative decode at K=7.

| ISL | 9B / N300 TTFT | 9B decode | 27B / T3K TTFT | 27B decode | 27B MTP TTFT | 27B MTP decode | speedup | acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 0.22 s | 22.65 tok/s | 0.38 s | 17.08 tok/s | 1.10 s | 43.01 tok/s | 2.52x | 4.27/7 |
| 4k | 0.85 s | 23.02 tok/s | 1.10 s | 16.98 tok/s | 1.81 s | 40.67 tok/s | 2.39x | 4.05/7 |
| 8k | 1.94 s | 22.46 tok/s | 2.09 s | 16.63 tok/s | 2.79 s | 32.16 tok/s | 1.93x | 2.85/7 |
| 16k | 4.03 s | 22.13 tok/s | 4.44 s | 16.45 tok/s | 5.35 s | 33.89 tok/s | 2.06x | 3.44/7 |
| 32k | 8.82 s | 21.92 tok/s | 10.32 s | 16.00 tok/s | 11.65 s | 29.19 tok/s | 1.82x | 2.75/7 |
| 64k | 20.96 s | 21.56 tok/s | 26.74 s | 15.26 tok/s | 28.66 s | 24.28 tok/s | 1.59x | 2.22/7 |
| 128k | 39.63 s | 20.43 tok/s | 54.49 s | 14.35 tok/s | 58.61 s | 40.92 tok/s | 2.85x | 4.94/7 |
| 256k | 165.97 s | 18.09 tok/s | 256.73 s | 11.57 tok/s | 274.40 s | 15.51 tok/s | 1.34x | 2.53/7 |

Decode falls off at 256k as paged-KV attention grows (9B −20%, 27B −32%). TTFT is near-linear in ISL.

### Accuracy

Full-depth logits vs the HuggingFace reference, real weights:

| Gate | 9B / N300 | 27B / T3K |
| --- | --- | --- |
| prefill logits PCC (≥ 0.98) | 0.9984 | 0.9957 |
| decode logits PCC, 5 steps (≥ 0.95) | 0.9948 | 0.9939 |

```bash
pytest models/demos/blackhole/qwen36/tests/unit/test_prefill.py \
       models/demos/blackhole/qwen36/tests/unit/test_decode.py -v -s
```

### Known limitations

- Greedy decode runs on device; temperature sampling does not (`QWEN35_TEMP=0`).
- Batched GDN prefill is capped at batch 2–4, below the serving batch.
- GDN decode batch-splits above B=16 at fp32 recurrent state on N300.
- Run single-device and multi-device test files in **separate pytest processes**: one process
  opening both layouts mis-sizes the chunk-seq L1 reservation.
- Qwen3.5-9B on N300 does not run MTP. Permuted RoPE makes the demo fall back to plain decode.
- MTP is batch 1, draft length K=7 (`QWEN36_SPEC_DRAFT_LEN` overrides). `QWEN36_SPEC=0` is plain decode.
- No-repeat-ngram (`QWEN35_NO_REPEAT_NGRAM`) skips MTP and uses plain decode.
- MTP greedy picks on device. Temperature sampling stays on the host.
