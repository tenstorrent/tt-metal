# Qwen3.5 / Qwen3.6 on Blackhole and Wormhole

This directory implements Tenstorrent inference for the hybrid
**Gated DeltaNet + Gated Full Attention** Qwen3.5/3.6 family. Despite the
`blackhole/` path (kept for now to avoid churn — the code is arch-aware, not
Blackhole-only), the same code path serves these checkpoints:

| Model            | `HF_MODEL`             | Mesh / `MESH_DEVICE` | Parallelism            |
| ---------------- | ---------------------- | -------------------- | ---------------------- |
| Qwen3.5-9B       | `Qwen/Qwen3.5-9B`      | single P150 — `P150` | single device          |
| Qwen3.5-27B      | `Qwen/Qwen3.5-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | T3K — `T3K`          | 8-way tensor parallel  |

- The **9B** runs on a **single Blackhole P150** device. It uses the validated
  single-device forward path (no collectives).
- The **27B** variants (both Qwen3.5-27B and Qwen3.6-27B) run on a **P150x4**
  (a `(1, 4)` Blackhole mesh) using **4-way tensor parallelism (TP)**. The TP
  path needs `FABRIC_1D` for the cross-device collectives (all-reduce /
  reduce-scatter) and a trace region for the captured chunk-outer prefill trace.
- The **27B** also runs on a **T3K** (a `(1, 8)` Wormhole mesh, 8 n300 chips) at
  **TP=8**. Nothing selects this but the mesh width; the same `FABRIC_1D` +
  trace-region requirements apply. Two things differ from Blackhole and are
  handled automatically:
  - **KV replication.** Qwen3.6-27B has only 4 KV heads but T3K has 8 devices,
    so each device replicates a whole KV head rather than owning a fraction of
    one (`tp_common.replicate_kv_weight`; device *d* takes KV head
    `d*n_kv_heads//TP`, which matches HF's GQA grouping). It costs 2x KV-cache
    memory versus a perfect shard, which the halved per-device weight footprint
    more than pays for.
  - **A smaller worker grid** (8x8 = 64 cores, vs ~11x10 on a P150) and **one
    ethernet link instead of two**. The fused-CCL grids, link counts, matmul
    `K_block_size`, and the L1-vs-DRAM choice for prefill matmul outputs are all
    derived from the device rather than hardcoded — see `tt/tp_common.py`.
  - **A tighter L1 budget.** Wormhole has 1464 KB of L1 per core to Blackhole's
    1536 KB, and spreads every L1-interleaved tensor over 64 cores instead of
    ~110 — so a per-core share is ~1.7x larger. Two prefill placements are
    therefore chosen from the device rather than fixed: a matmul's `in0_block_w`
    shrinks when its circular buffers would not fit, and the ~20 MB `[S, dim]`
    out-projection outputs (MLP down-proj, attention `wo`) fall back to DRAM
    instead of staying L1-resident. Blackhole keeps its measured choices.
  - **A replicated-activation vision tower.** The vision tower's `dim` is 1152 =
    36 tiles, which 8 devices cannot split into whole tiles (4.5 each; it is 9 at
    TP=4). Since the block I/O contract restores its fracture with a
    `reduce_scatter` along `dim=3`, and a tile cannot be split, the tower instead
    keeps **activations replicated** at full `dim` on T3K and all-reduces the
    row-parallel out-projections (`tt/vision/vision_ccl.py`) rather than
    reduce-scattering them. Weights stay sharded, so no TP compute is given up —
    the tower is not run redundantly. The PatchMerger still fractures its
    **output** for the LLM, which is safe because that splits `out_hidden_size`
    (5120 -> 20 tiles/device), not `dim`. Selected automatically by
    `vision_replicated_acts`; `P150x4` is unaffected.

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

**27B on a T3K (8 Wormhole n300 chips, TP=8):**

```bash
export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=T3K
```

`HF_MODEL` is the single source of truth for the checkpoint — it may be a Hugging
Face hub id (resolved via `snapshot_download`) or a local checkpoint directory.
`MESH_DEVICE` selects the mesh shape — `P150` → `(1,1)`, `P150x4` → `(1,4)`,
`T3K` → `(1,8)`; see `MESH_SHAPES` in `tt/model_config.py`, which is the single
table the tests and both demos read.

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
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "traced"

# A single ISL, e.g. the short 128-token traced case
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "traced_128"

# Medium / long traced ISLs
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "traced_4k"
pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "traced_64k"
```

The **same command works for both 9B and 27B** — only the exported `HF_MODEL` /
`MESH_DEVICE` differ. On a single device the test takes the validated 9B path; on
the `(1,4)` mesh it routes through the TP chunk-outer traced prefill + paged
traced decode path automatically.

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
pytest models/demos/blackhole/qwen3_5_9b/tests/unit/ -v -s
pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill.py -v -s
pytest models/demos/blackhole/qwen3_5_9b/tests/test_weight_mapping.py -v -s
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
pytest models/demos/blackhole/qwen3_5_9b/tests/test_mlp_tp.py -v -s
pytest models/demos/blackhole/qwen3_5_9b/tests/test_attention_tp.py -v -s
pytest models/demos/blackhole/qwen3_5_9b/tests/test_gdn_tp.py -v -s
pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_tp.py -svq
pytest models/demos/blackhole/qwen3_5_9b/tests/test_generate_tp.py -v -s
```

> `test_substate.py` and `test_weight_mapping.py` are pure-CPU and need no device.
> `test_weight_mapping.py`'s shape constants assume the 9B checkpoint.
