# Qwen3.5 / Qwen3.6 on Blackhole and Wormhole

This directory implements Tenstorrent inference for the hybrid
**Gated DeltaNet + Gated Full Attention** Qwen3.5/3.6 family. A single code path serves four
checkpoints. Everything runs on **Blackhole** (P150); the sparse-MoE **Qwen3.6-35B-A3B**
additionally runs on **Wormhole** (n150x4):

| Model            | `HF_MODEL`             | Arch      | Mesh / `MESH_DEVICE` | Parallelism            |
| ---------------- | ---------------------- | --------- | -------------------- | ---------------------- |
| Qwen3.5-9B       | `Qwen/Qwen3.5-9B`      | Blackhole | single P150 — `P150` | single device          |
| Qwen3.5-27B      | `Qwen/Qwen3.5-27B`     | Blackhole | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | Blackhole | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | Blackhole | P150x8 — `P150x8`    | 8-way tensor parallel  |
| Qwen3.6-35B-A3B  | `Qwen/Qwen3.6-35B-A3B` | Blackhole | P150x4 — `P150x4`    | 4-way TP + sparse MoE  |
| Qwen3.6-35B-A3B  | `Qwen/Qwen3.6-35B-A3B` | Wormhole  | n150x4 — `N150x4`    | 4-way TP + sparse MoE  |

The same `(1, 4)` TP code path serves `P150x4` and `N150x4` — what differs is hardware geometry,
and every grid-, bank- and link-shaped constant is now derived from the mesh device rather than
hardcoded. See [Running on Wormhole (n150x4)](#running-on-wormhole-n150x4).

> **Wormhole is for the 35B-A3B only.** The 9B / 27B checkpoints remain Blackhole-only: their
> program configs and memory budgets were tuned for a P150 (32 GB, 11x10 grid) and neither has
> been brought up on a Wormhole n150 (12 GB, 8x8). `demo/text_demo.py` skips them on Wormhole
> rather than running something unvalidated. **No Blackhole behaviour changes** — every derived
> constant reproduces the value it replaced on a P150 (`agmm_grid` → grid `(8,9)`, 2 links,
> 4 workers; `mmrs_prefill_grid` → grid `(8,8)`, RS offset `(0,8)`; GDN conv chunks 2 for the MoE
> and 1 for the dense checkpoints; 8 DRAM banks).

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
`MESH_DEVICE` selects the mesh shape. Blackhole and Wormhole names for the same
shape are interchangeable: `P150`/`N150` → `(1,1)`, `N300` → `(1,2)`,
`P150x4`/`N150x4` → `(1,4)`, `P150x8`/`T3K` → `(1,8)`.

Optional flags:

```bash
# Run SDPA in BF8 (faster; slightly lower precision).
export QWEN_SDPA_BF8=1
```

## Running on Wormhole (n150x4)

`MESH_DEVICE=N150x4` runs the **Qwen3.6-35B-A3B** on a `(1, 4)` Wormhole mesh through the same TP
code path as `P150x4`. No other checkpoint in this directory is supported on Wormhole. Five
hardware properties differ, and each one is read off the mesh device at config time instead of
being hardcoded:

| Property                  | BH P150 | WH N150 | Where it is read                              |
| ------------------------- | ------- | ------- | --------------------------------------------- |
| Tensix worker grid        | 11 × 10 | 8 × 8   | `tp_common.worker_grid`                       |
| Ethernet links per TP hop | 2       | 1       | `tp_common.ccl_num_links`                     |
| DRAM banks                | 8       | 12      | `ModelArgs.num_dram_banks`                    |
| DRAM per device           | 32 GB   | 12 GB   | budget only — see below                       |
| L1 per core               | 1536 KB | 1464 KB | `tp_common.prefill_l1_output_ok`              |

The consequences, all handled in code:

- **Fused all-gather + matmul — enabled here, unlike PR #54572.** That PR disables this fusion on
  Wormhole (`mlp_gateup_agmm_enabled` -> `is_blackhole()`), reporting that forcing the grid height
  to 8 makes `all_gather_minimal_matmul_async` build overlapping in0/in1 sender/receiver core
  ranges (it derives them from `grid_size.y-1/-2/-3`) and die with
  `local_noc0_in_use and local_noc1_in_use`. The height used here is `gy - 1` = **7**, not 8, which
  leaves the op's mux row free and avoids that overlap — measured working on a (1,4) WH mesh at
  op-level PCC 0.99997, with `test_mlp_tp_prefill` passing with the swiglu fusion live. Worth
  feeding back: that PR estimates the fusion hides ~2.5 ms of a 21.7 ms single-layer GDN prefill.
- **Fused all-gather + matmul.** `all_gather_matmul_prefill` / `all_gather_swiglu_prefill` place
  their `2 × num_links` fabric mux cores on the *last row* of the worker grid, and assert
  `ceil(grid.x / workers_per_link) == num_links`. `tp_common.agmm_grid` derives the grid height
  (9 on BH, 7 on WH) and the worker count (4 on BH's 2 links, 8 on WH's 1) from the device.
  Measured on WH at M=128, K=2048, N=1024: PCC 0.99997.
- **Shared-GDN Wormhole compat layer** (`tt/wh_compat.py` + `tt/chunk_seq_wh.py`, taken from
  PR #54572). The chunk-seq kernels in `models/experimental/gated_attention_gated_deltanet` were
  tuned for Blackhole's much larger total L1; on Wormhole their activations collide with the
  kernel's circular buffers. The compat layer sends chunk-seq activations to DRAM and switches the
  `[BH, L, V]` output relayout to bf16 (33.5 MB -> 16.8 MB at L=2048). Both overrides delegate to
  upstream whenever `is_blackhole()`, so Blackhole is bit-for-bit unchanged. **Without this the GDN
  prefill does not complete on Wormhole**; with it, `test_gdn_tp_prefill` passes at PCC 0.99996.
- **Fused matmul + reduce-scatter is Blackhole-only.** `matmul_reduce_scatter_prefill`
  (`matmul_reduce_scatter_async`, the GDN out-projection) *enqueues but never completes* on a WH
  N150x4 — the 1-link Linear hop asks for 8 RS workers per direction (18 cores) and no split of
  the 8x8 grid between the matmul and those cores lets it finish; the host spins in the readback
  indefinitely. GDN prefill therefore takes the **unfused** arm on Wormhole (`ttnn.linear` +
  `tt_all_reduce`, the arm the out-sharded config already uses), measured at PCC 0.99999 in 2.8 s
  against the fused op's hang on the identical shape. Blackhole keeps the fusion untouched — see
  `tp_common.mmrs_prefill_supported`. This costs Wormhole the matmul/RS overlap, so GDN-layer TTFT
  will be somewhat worse there than a naive scaling from Blackhole would suggest.
- **L1-resident prefill outputs.** Keeping a tuned prefill matmul's `[seq, N]` output
  in L1 is a Blackhole-only win; on WH the same program config's circular buffers
  plus that output overflow L1 (the MLP down-projection trips it even on the 27B).
  Those outputs go to DRAM on WH — program configs are unchanged.
- **GDN depthwise conv chunking.** The prefill `ttnn.conv1d` is height-sharded, so
  per-core L1 scales with `channels / chunks / num_cores`. The chunk count is scaled
  by the core-count ratio against the BH reference (110 cores), giving 4 chunks for
  the MoE checkpoint on WH versus 2 on BH. The split is exact (depthwise is
  per-channel-independent), so it is a placement change only.

### Memory budget

A Wormhole n150 has **12 GB of DRAM per device** against a P150's 32 GB, and that,
not compute, is the binding constraint for the 35B-A3B. With the shipped dtypes
(routed-expert gate/up `bfloat4_b`, down `bfloat8_b`, everything else `bfloat8_b`)
and expert-parallel sharding of 256 experts over 4 devices, the weights come to
roughly **6–7 GB per device**, leaving ~5 GB for the 1 GiB trace region, the paged
KV cache, the GDN recurrent/conv state and activations.

That is comfortable for short and medium ISLs at batch 1, and it is the reason the
long-context and large-batch corners of `demo/text_demo.py` (the 128k/256k ISLs, and
the `batched_*_b8` / `b32` ladder past ~8k) are **not expected to fit on n150x4** —
they were sized against the P150x4's 32 GB. Run them only with a reduced block
budget. The KV cache alone is ~5.4 KB per token per device (10 full-attention layers
× 1 local KV head × 256 head-dim × K and V in bf8), and the GDN state adds ~15 MB per
batch row.

### What was validated on Wormhole

**End-to-end, with the real checkpoint.** `Qwen/Qwen3.6-35B-A3B` on a `(1, 4)` Wormhole mesh:

```bash
MESH_DEVICE=N150x4 HF_MODEL=Qwen/Qwen3.6-35B-A3B \
    pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s \
           -k "traced_128 and not 128k" --timeout=5000
```

| Case | TTFT | Decode / user | Aggregate | Result |
| ---- | ---- | ------------- | --------- | ------ |
| `traced_128` | 0.67 s | 25.48 tok/s | — | PASSED |
| `traced_4k` | 12.6 s | 25.25 tok/s | — | PASSED |
| `determinism_128` | — | — | — | PASSED (two runs, identical output) |
| `batched_128_b8` | — | 20.13 tok/s | 161.1 tok/s | PASSED |
| `batched_128_b32` | 45.2 s | 9.31 tok/s | 298.0 tok/s | PASSED |

Model load 30-32 s from a warm bf8 cache; generated text is fluent and on-topic (the test's
non-degeneracy gate passes). The 40-layer model's weights fit the 4 x 12 GB budget with the 1 GiB
trace region resident.

Decode is the tuned path. Prefill TTFT still scales at roughly 3 ms/token at long ISL: the fused
matmul + reduce-scatter is off here (it hangs on Wormhole, see `tp_common.mmrs_prefill_supported`),
and the `wh_9b_n300` prefill passes from PR #54572 were swept on an N300 at TP=2 with dim<=4096, so
they need re-measuring for this (1,4) MoE config rather than copying.

**Component tests**, same mesh, same real checkpoint:

| Suite | Result |
| ----- | ------ |
| `test_gdn_tp.py` | 17/17 |
| `test_attention_tp.py` | 7/7 |
| `test_moe_tp.py` | 6/6 |
| `test_rope_tp.py` | 2/2 |
| `test_generate_tp.py`, `test_sampling.py` | 1/1 each |
| `test_decode_bucketing.py` | 16/17 (the remaining one imports vLLM) |
| `test_model_tp.py` | 13/14 |

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
> when the pad really allocated.


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
`N150x4`, which is how the shared TP surface was exercised on Wormhole — see
[What was validated on Wormhole](#what-was-validated-on-wormhole)). PCC thresholds
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
