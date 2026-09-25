# Qwen3.5 / Qwen3.6 on Blackhole and Wormhole

This directory implements Tenstorrent inference for the hybrid
**Gated DeltaNet + Gated Full Attention** Qwen3.5/3.6 family. A single code path serves four
checkpoints. Everything runs on **Blackhole** (P150); the sparse-MoE **Qwen3.6-35B-A3B**
additionally runs on a **Wormhole LoudBox** (4 × n300), as a `(1, 4)` mesh (`N150x4`):

| Model            | `HF_MODEL`             | Arch      | Mesh / `MESH_DEVICE` | Parallelism            |
| ---------------- | ---------------------- | --------- | -------------------- | ---------------------- |
| Qwen3.5-9B       | `Qwen/Qwen3.5-9B`      | Blackhole | single P150 — `P150` | single device          |
| Qwen3.5-27B      | `Qwen/Qwen3.5-27B`     | Blackhole | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | Blackhole | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | Blackhole | P150x8 — `P150x8`    | 8-way tensor parallel  |
| Qwen3.6-35B-A3B  | `Qwen/Qwen3.6-35B-A3B` | Blackhole | P150x4 — `P150x4`    | 4-way TP + sparse MoE  |
| Qwen3.6-35B-A3B  | `Qwen/Qwen3.6-35B-A3B` | Wormhole  | WH LoudBox — `N150x4` | 4-way TP + sparse MoE |

The same `(1, 4)` TP code path serves `P150x4` and `N150x4` — what differs is hardware geometry,
and every grid-, bank- and link-shaped constant is now derived from the mesh device rather than
hardcoded. See [Running on the Wormhole LoudBox](#running-on-the-wormhole-loudbox).

> **Wormhole is for the 35B-A3B only.** The 9B / 27B checkpoints remain Blackhole-only: their
> program configs and memory budgets were tuned for a P150 (32 GB, 11x10 grid) and neither has
> been brought up on a Wormhole chip (12 GB, 8x8). `demo/text_demo.py` skips them on Wormhole
> rather than running something unvalidated.
>
> **Blackhole runs the same code as before Wormhole support was added.** Every Wormhole-specific
> change is behind an `is_blackhole()` check, and the device-derived constants reproduce the P150
> values they replaced (`agmm_grid` → grid `(8,9)`, 2 links, 4 workers; GDN conv chunks 2 for the
> MoE and 1 for the dense checkpoints; 8 DRAM banks). The one intentional exception is
> `tp_common.pad_and_free`, a use-after-free fix that applies to both arches (see below).

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

**35B-A3B (n150x4 — Wormhole, sparse MoE):**

```bash
export HF_MODEL=Qwen/Qwen3.6-35B-A3B
export MESH_DEVICE=N150x4
```

`HF_MODEL` is the single source of truth for the checkpoint — it may be a Hugging
Face hub id (resolved via `snapshot_download`) or a local checkpoint directory.
`MESH_DEVICE` selects the mesh shape for `demo/text_demo.py` and the `*_tp` tests:
`P150` → `(1,1)`, `P150x4` / `N150x4` → `(1,4)`, `P150x8` → `(1,8)`.

Optional flags:

```bash
# Run SDPA in BF8 (faster; slightly lower precision).
export QWEN_SDPA_BF8=1
```

## Running on the Wormhole LoudBox

A Wormhole LoudBox holds four n300 cards (8 Wormhole chips). `MESH_DEVICE=N150x4` opens four of
those chips as a `(1, 4)` mesh and runs the **Qwen3.6-35B-A3B** through the same TP code path as
`P150x4`. No other checkpoint in this directory is supported on Wormhole.

```bash
export HF_MODEL=Qwen/Qwen3.6-35B-A3B
export MESH_DEVICE=N150x4
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128 and not 128k" --timeout=5000
```

Five hardware properties differ from a P150, and each one is read off the mesh device at config
time instead of being hardcoded:

| Property                  | BH P150 | WH chip | Where it is read                              |
| ------------------------- | ------- | ------- | --------------------------------------------- |
| Tensix worker grid        | 11 × 10 | 8 × 8   | `tp_common.worker_grid`                       |
| Ethernet links per TP hop | 2       | 1       | `tp_common.ccl_num_links`                     |
| DRAM banks                | 8       | 12      | `ModelArgs.num_dram_banks`                    |
| DRAM per device           | 32 GB   | 12 GB   | budget only — see below                       |
| L1 per core               | 1536 KB | 1464 KB | `tp_common.prefill_l1_output_ok`              |

### What differs on Wormhole

- **Fused all-gather + matmul.** `all_gather_matmul_prefill` / `all_gather_swiglu_prefill` place
  their `2 × num_links` fabric mux cores on the *last row* of the worker grid, and assert
  `ceil(grid.x / workers_per_link) == num_links`. `tp_common.agmm_grid` derives the grid height
  (9 on BH, 7 on WH) and the worker count (4 on BH's 2 links, 8 on WH's 1) from the device. PR
  #54572 disables this fusion on Wormhole because a grid height of 8 makes the op build
  overlapping sender/receiver core ranges; the height here is `gy - 1` = 7, which leaves the mux
  row free. Measured on WH at M=128, K=2048, N=1024: PCC 0.99997.
- **Shared-GDN Wormhole compat layer** (`tt/wh_compat.py` + `tt/chunk_seq_wh.py`, taken from
  PR #54572). The chunk-seq kernels in `models/experimental/gated_attention_gated_deltanet` were
  tuned for Blackhole's much larger total L1; on Wormhole their activations collide with the
  kernel's circular buffers. The compat layer sends chunk-seq activations to DRAM and switches the
  `[BH, L, V]` output relayout to bf16 (33.5 MB -> 16.8 MB at L=2048). Both overrides delegate to
  upstream whenever `is_blackhole()`. Without this the GDN prefill does not complete on Wormhole.
- **Fused matmul + reduce-scatter is Blackhole-only.** `matmul_reduce_scatter_async` (the GDN
  out-projection) *enqueues but never completes* on the `(1, 4)` WH mesh: the 1-link Linear hop
  asks for 8 RS workers per direction (18 cores) and no split of the 8x8 grid lets it finish.
  GDN prefill takes the **unfused** arm on Wormhole (`ttnn.linear` + `tt_all_reduce`); see
  `tp_common.mmrs_prefill_supported`.
- **L1-resident prefill outputs.** Keeping a tuned prefill matmul's `[seq, N]` output in L1 is a
  Blackhole-only win; on WH the same program config's circular buffers plus that output overflow
  L1. Those outputs go to DRAM on WH — program configs are unchanged.
- **GDN depthwise conv chunking.** The prefill `ttnn.conv1d` is height-sharded, so per-core L1
  scales with `channels / chunks / num_cores`. The chunk count is scaled by the core-count ratio
  against the BH reference (110 cores): 4 chunks for the MoE checkpoint on WH versus 2 on BH. The
  split is exact (depthwise is per-channel-independent).
- **MoE prefill** (`tt/moe/prefill.py::_process_prefill_chunk_wh`). The routed-expert
  `sparse_matmul`s were 82% of WH prefill device time. The WH path runs both expert matmuls per
  (32-token tile, expert) pair gated by the same tile mask, where Blackhole's down_proj computes
  every local expert over the whole chunk. It also uses the decode-swept block widths
  (`in0_block_w = K/2`, `per_core_n = 2`), applies the routing weights on the 512-wide
  intermediate before down_proj, and writes bfloat8_b matmul outputs. Measured on the first 4
  layers at T=2048: 926 -> 234 ms of prefill; MoE prefill PCC moves by -0.00025.
- **Decode.** The WH decode path keeps the attention head tensors height-sharded from the QKV
  head split through SDPA, uses a local fork of the GDN recurrent step
  (`tt/gdn/recurrent_decode_wh.py`, taken only when its `[B,H,K,V]` intermediate fits L1, i.e.
  up to B=8), and uses the fused `generalized_moe_gate` router with tuned sparse-matmul configs.

### Memory budget

Each Wormhole chip has **12 GB of DRAM** against a P150's 32 GB, and that, not compute, is the
binding constraint for the 35B-A3B. With the shipped dtypes (routed-expert gate/up `bfloat4_b`,
down `bfloat8_b`, everything else `bfloat8_b`) and expert-parallel sharding of 256 experts over
4 devices, the weights come to roughly **6–7 GB per device**, leaving ~5 GB for the 1 GiB trace
region, the paged KV cache, the GDN recurrent/conv state and activations. The KV cache is
~5.4 KB per token per device (10 full-attention layers × 1 local KV head × 256 head-dim × K and V
in bf8), and the GDN state adds ~15 MB per batch row.

Batch-1 prefill fits up to and including the `traced_128k` case (a 103,351-token prompt, see
below). The `traced_256k` ISL and the `batched_*_b8` / `b32` cases past 128 tokens have not been
run on this mesh; they were sized for the P150x4's 32 GB.

### Validated results

Measured on the Wormhole LoudBox, `MESH_DEVICE=N150x4`, `HF_MODEL=Qwen/Qwen3.6-35B-A3B`, traced,
model load 12–18 s from a warm weight cache:

```bash
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s --timeout=5000 \
    -k "traced_128 or traced_4k or traced_8k or traced_64k or traced_128k or batched_128_b8 or batched_128_b32"
```

| Case | Prompt tokens | TTFT | Decode | Result |
| ---- | ------------- | ---- | ------ | ------ |
| `traced_128`      | 128     | 0.47 s   | 25.46 tok/s | PASSED |
| `traced_4k`       | 2,642   | 3.00 s   | 25.27 tok/s | PASSED |
| `traced_8k`       | 8,192   | 7.86 s   | 24.99 tok/s | PASSED |
| `traced_64k`      | 65,536  | 71.24 s  | 23.13 tok/s | PASSED |
| `traced_128k`     | 103,351 | 120.89 s | 21.51 tok/s | PASSED |
| `batched_128_b8`  | 128 × 8  | 16.69 s | 20.12 tok/s/user (161.0 aggregate) | PASSED |
| `batched_128_b32` | 128 × 32 | 7.59 s  | 9.37 tok/s/user (299.9 aggregate)  | PASSED |

`traced_4k` uses a 2,642-token prompt, and `traced_128k` runs the whole Frankenstein text, which
is 103,351 tokens (its KV block budget is sized for 128k). The generated text is coherent and
on-topic in every case (the test's non-degeneracy gate passes).

Component PCC suite on the same mesh with the 35B-A3B checkpoint — `test_moe_tp.py`,
`test_rope_tp.py`, `test_attention_tp.py`, `test_gdn_tp.py`, and `test_model_tp.py`'s
`test_model_tp_contract` / `test_model_tp_long_prefill` / `test_model_tp_long_prefill_traced` —
all pass (`test_mlp_tp` skips on a MoE checkpoint, which has no dense MLP):

| Case | PCC vs reference |
| ---- | ---------------- |
| MoE decode (torch)                         | 0.98747 |
| MoE decode, batch 8 (torch)                | 0.98884 |
| MoE prefill, seq 32 / 256 / 512 (torch)    | 0.98949 / 0.99016 / 0.98999 |
| Model contract, prefill logits (8 layers)  | 0.99845 |
| Long prefill T=2304, chunked vs single-pass | 0.99745 |
| Traced vs eager chunked prefill, T=4096 / 4352 | 0.99823 / 0.99928 |

Full component suites, same mesh and checkpoint. The first group was re-run with the current code
(including with the watcher on, as the CI leg runs it); the second group was measured before the MoE
prefill change and has not been re-run since:

| Suite | Result | Measured |
| ----- | ------ | -------- |
| `test_gdn_tp.py` | 17/17 | current code |
| `test_attention_tp.py` | 7/7 | current code |
| `test_moe_tp.py` | 6/6 | current code |
| `test_rope_tp.py` | 2/2 | current code |
| `demo/text_demo.py` `determinism_128` | PASSED (two runs, identical output) | current code |
| `test_generate_tp.py`, `test_sampling.py` | 1/1 each | before the MoE prefill change |
| `test_decode_bucketing.py` | 16/17 (the remaining one imports vLLM) | before the MoE prefill change |
| `test_model_tp.py` | 13/14 | before the MoE prefill change |

The one `test_model_tp` failure is `prefill_paged_slots_long[eager-2chunks_plus_tail]`: for one
user of eight, first-step decode logits land at PCC 0.96 against the B=1 chunk-outer reference.
Both per-slot prefill variants (eager and traced) show the same user, so it is a real
per-slot-vs-reference numerical delta and not a cascade — the prefill logits and the GDN recurrent
state round-trip are both bit-exact for that user.

> **Fixed while porting:** the decode KV-cache update did `ttnn.pad(...)` and then
> deallocated the pad's *source*. `ttnn.pad` returns a metadata-only view aliasing
> its input when the requested pad already fits inside the tile padding — which is
> exactly the `[1, B, 1, HD] → [1, B, 32, HD]` case here — so that freed the padded
> tensor's own storage and the next L1 allocation clobbered it. On WH at B=32 this
> dropped attention PCC to 0.09; on BH the larger L1 happens not to recycle the block
> before the read, so it was latent. `tp_common.pad_and_free` frees the source only
> when the pad really allocated, and is used on both arches.

### Known limitations on Wormhole

- **MoE prefill still does redundant work.** TTFT is about 1.0–1.2 ms per token. The WH expert
  path still streams each expert's weights once per 32-token tile and runs swiglu over
  zero-filled expanded tensors; grouping tokens by expert before the matmuls is the next step.
- **No fused matmul + reduce-scatter** in GDN prefill (it hangs on this mesh, see above).
- **Untested corners:** `traced_256k`, and batched cases with prompts longer than 128 tokens.

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
`parametrize_mesh_tp` helper wires this from `MESH_DEVICE`; it also accepts
`N150x4`, the Wormhole LoudBox mesh — see
[Validated results](#validated-results)). PCC thresholds
are in `tests/pcc_thresholds.json`.

| Test                  | Validates                                                            |
| --------------------- | ------------------------------------------------------------------- |
| `test_mlp_tp.py`      | TP SwiGLU MLP (column/row-parallel + reduce-scatter)                |
| `test_moe_tp.py`      | TP sparse MoE MLP (router + experts + shared; MoE checkpoint only; decode + prefill) |
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

> The MoE-specific tests (`test_moe_tp.py`, and the MoE path in `test_model_tp.py` /
> `test_generate_tp.py`) require the sparse checkpoint — run them with
> `HF_MODEL=Qwen/Qwen3.6-35B-A3B MESH_DEVICE=P150x4`. On the dense 27B they are inert
> (`num_experts == 0`), and the dense/MoE checkpoints must not be mixed in one run
> (each test file `setdefault`s or expects a single `HF_MODEL`).

> `test_substate.py` and `test_weight_mapping.py` are pure-CPU and need no device.
> `test_weight_mapping.py`'s shape constants assume the 9B checkpoint.

### Running the tests on Wormhole — **35B-A3B (N150x4)**

Wormhole runs the sparse-MoE `Qwen3.6-35B-A3B` only, and only on the `(1,4)` mesh. The TP tests
above are the applicable set; everything single-device is Blackhole-only in practice:

* `tests/unit/*` skip themselves (`run_for_blackhole`, "only runs for Blackhole"). `test_substate.py`
  is pure CPU and passes anywhere.
* `test_prefill.py`, `test_weight_mapping.py` and the vision tests (`test_patch_merger.py`,
  `test_vision_attention.py`, `test_vision_block.py`, `test_wrapped_model.py`, `test_mlp.py`,
  `test_model.py`) target the 9B/27B/vision checkpoints and fail on 35B-A3B with
  `KeyError: 'Qwen3.6-35B-A3B'` or an empty state dict — not a regression.
* `test_prefill.py` additionally builds a **single-device** model. The chunk-seq kernel's circular
  buffers want nearly a whole Wormhole L1 bank at this model's head count, so any resident L1
  buffer collides; it needs the larger Blackhole L1.

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=Qwen/Qwen3.6-35B-A3B MESH_DEVICE=N150x4

pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py         -q --timeout=900
pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py   -q --timeout=900
pytest models/demos/blackhole/qwen36/tests/test_moe_tp.py         -q --timeout=900
pytest models/demos/blackhole/qwen36/tests/test_rope_tp.py        -q --timeout=600
pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py    -q --timeout=900
pytest models/demos/blackhole/qwen36/tests/test_sampling.py       -q --timeout=600
pytest models/demos/blackhole/qwen36/tests/test_decode_bucketing.py -q --timeout=900
pytest models/demos/blackhole/qwen36/tests/test_model_tp.py       -q --timeout=2400
```

> Run these **module by module**, not as one pytest session: a single session collecting the whole
> directory has been seen to die with a `Fatal Python error: Bus error` partway through and take
> the remaining modules with it. `tt-smi -r` between the heavy modules clears a wedged card.

> `test_decode_bucketing.py` needs vLLM for one case; without it that case skips.

> **Watcher on Wormhole:** set `TT_METAL_WATCHER_DISABLE_ETH=1` alongside `TT_METAL_WATCHER`. With the
> watcher on the ethernet cores the Wormhole `fabric_erisc_router` fails to link (`.data` overlaps
> `.text`), so every test errors at mesh open, and the half-initialized fabric leaves the ethernet cores
> wedged (`Timed out waiting for ETH heartbeat`) until `tt-smi -r`. Worker cores stay watched.

## CI

The Qwen3.6 CI legs live in `tests/pipeline_reorg/` (unit: `models_unit_tests.yaml`, e2e:
`models_e2e_tests.yaml`, vLLM: `vllm_model_tests.yaml`); perf targets for the e2e legs are in
`models/model_targets.yaml`. The 35B-A3B unit and e2e entries each serve both arches from one
command: `MESH_DEVICE` (and, for the unit leg, the watcher setup) come from per-SKU fields.

| Model | Leg | Arch / SKU | Tier | Runs |
| ----- | --- | ---------- | ---- | ---- |
| Qwen3.6-35B-A3B | unit | BH QuietBox 2 — `bh_quietbox_2`, `MESH_DEVICE=P150x4` | 1 | `test_moe_tp`, `unit/test_moe`, `test_gdn_tp`, `test_attention_tp` (watcher on) |
| Qwen3.6-35B-A3B | unit | WH LoudBox — `wh_llmbox`, `MESH_DEVICE=N150x4` | 2 | same files (`unit/test_moe` skips; watcher on, ethernet cores excluded) |
| Qwen3.6-35B-A3B | e2e | BH QuietBox 2 — `bh_quietbox_2` | 1 | `text_demo.py` `traced_128`, `traced_4k`, `determinism_128` |
| Qwen3.6-35B-A3B | e2e | WH LoudBox — `wh_llmbox_perf` | 2 | same cases |
| Qwen3.6-27B | unit, e2e, vLLM | BH QuietBox 2 — `bh_quietbox_2` | 1 | see the yamls |

Scheduled runs pick every leg up automatically: the Tier 1 workflows run the Blackhole legs and the
Tier 2 workflows run the Wormhole ones. To run one leg by hand (Actions tab, or `gh`), select the
model `qwen3.6-35b-a3b` and the SKU:

```bash
# Blackhole (Tier 1)
gh workflow run "(Tier 1) Models Unit Tests"       --ref <branch> -f model=qwen3.6-35b-a3b -f sku="bh_quietbox_2 (BH QB2)"
gh workflow run "(Tier 1) Models End-To-End Tests" --ref <branch> -f model=qwen3.6-35b-a3b -f sku="bh_quietbox_2 (BH QB2)"
# Wormhole LoudBox (Tier 2)
gh workflow run "(Tier 2) Models Unit Tests"       --ref <branch> -f model=qwen3.6-35b-a3b -f sku="wh_llmbox (T3000)"
gh workflow run "(Tier 2) Models End-To-End Tests" --ref <branch> -f model=qwen3.6-35b-a3b -f sku="wh_llmbox_perf (T3000 perf)"
```

The Wormhole runners read the checkpoint from `/mnt/MLPerf/huggingface` with the HF hub offline, and
the shared tensor cache is mounted read-only by default. A first Wormhole run therefore needs the
checkpoint on that share and `-f mlperf-write-access=true`, so it can write the Wormhole tensor cache
(kept separate from Blackhole's: the cache path includes the device name). The `wh_llmbox_perf`
targets were measured on a development LoudBox, not a CI runner; re-baseline them from the first
CI run.
