# Qwen3.5 / Qwen3.6 / Qwen3.8 on Blackhole and Wormhole

This directory implements Tenstorrent inference for the hybrid
**Gated DeltaNet + Gated Full Attention** Qwen3.5/3.6/3.8 family, on Blackhole and
(text-only) on Wormhole. A single code path serves five checkpoints:

| Model            | `HF_MODEL`             | Mesh / `MESH_DEVICE` | Parallelism            |
| ---------------- | ---------------------- | -------------------- | ---------------------- |
| Qwen3.5-9B       | `Qwen/Qwen3.5-9B`      | single P150 — `P150` | single device          |
| Qwen3.5-27B      | `Qwen/Qwen3.5-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | P150x8 — `P150x8`    | 8-way tensor parallel  |
| Qwen3.6-35B-A3B  | `Qwen/Qwen3.6-35B-A3B` | P150x4 — `P150x4`    | 4-way TP + sparse MoE  |
| Qwen3.8-27B      | `Qwen/Qwen3.8-27B`     | T3K — `T3K`          | 8-way tensor parallel  |

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


## Wormhole (T3K / LoudBox / QuietBox) — text-only

A TT-LoudBox or Wormhole QuietBox is **4x n300 = 8 WH_B0 chips**, which tt-metal calls
**`T3K`**. The 27B runs there at **TP=8** on the same `(1, 8)` mesh and KV-replication path
as `P150x8`. **Qwen3.8-27B** is architecturally identical to Qwen3.6-27B — the two HF
`config.json` files differ only in `transformers_version` — so it needs no model-specific
code, only `HF_MODEL`.

```bash
export HF_MODEL=Qwen/Qwen3.8-27B     # or Qwen/Qwen3.6-27B
export MESH_DEVICE=T3K
export TT_CACHE_PATH=$HOME/tt_cache/Qwen3.8-27B
```

> **Do not set `WH_ARCH_YAML`.** It is obsolete and no longer read by the runtime; ttnn
> selects Ethernet dispatch automatically for `ClusterType::T3K`. Do not set
> `TT_MESH_GRAPH_DESC_PATH` either — a mismatched value can *silently hang* in the first
> collective rather than erroring.

Sanity-check the mesh before a long run; 8 chips with untrained inter-board ethernet report
`ClusterType.N300` and everything T3K-shaped then degrades or hangs:

```bash
python3 -c "
import ttnn
print(ttnn.get_num_devices(), ttnn.cluster.get_cluster_type())   # expect 8 ClusterType.T3K
print(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())     # expect MeshShape([2, 4])"
```
`tt-smi -r 0,1,2,3` recovers a wedged ETH core (`Timed out waiting for ETH heartbeat`).

### Measured perf (Qwen3.8-27B, T3K, batch 1)

Targets live in `models/model_targets.yaml` under `qwen3.8-27b` / `wh_llmbox_perf`. The
Blackhole QuietBox (`bh_quietbox_2`, TP=4) column is the cross-arch reference.

| ISL  | TTFT T3K | TTFT BH ref | t/s/u T3K | t/s/u BH ref |
| ---- | -------- | ----------- | --------- | ------------ |
| 128  | 0.57 s   | 0.15 s      | 16.8      | 26.0         |
| 4k   | 1.47 s   | 2.03 s      | 16.8      | 18.1         |
| 8k   | 3.18 s   | 4.62 s      | 16.4      | 18.0         |
| 16k  | 6.44 s   | 9.47 s      | 16.3      | 17.9         |
| 32k  | 13.76 s  | 19.96 s     | 16.0      | 17.6         |
| 128k | 56.45 s  | 77.61 s     | 14.2      | 16.7         |

### Wormhole-specific notes

Wormhole exposes **8x8 = 64 worker cores** (vs ~110 on a BH P150) and ~72 KB less L1 per
core, and a T3K has **one usable ethernet link** per chip pair (the other is reserved for
the dispatcher). The code adapts automatically; the pieces worth knowing:

* `tp_common.fused_ccl_num_links()` returns 1 on WH, 2 on BH.
* `agmm_prefill_grid_default()` / `mmrs_prefill_grid_default()` derive the fused-CCL grids
  from `compute_with_storage_grid_size()`, reserving the rows the all-gather muxes and
  reduce-scatter workers need. They reproduce the BH constants exactly on an 11x10 grid.
* `prefill_l1_output_ok()` is False on WH: the tuned prefill matmuls write their output to
  DRAM instead of L1, because on 64 cores the output block plus the matmul's own circular
  buffers do not fit. `_PREFILL_TUNING` also drops `in0_block_w_cap` to 2.
* `QWEN36_GDN_FUSED=0` routes GDN prefill through the composite
  `chunk_gated_delta_rule_seq_adapter` instead of the fused op — the fallback if the fused
  op regresses. It costs roughly 2x on TTFT.

**Run one pytest process per demo case.** Chaining demo configs in a single process on the
8-device mesh has wedged an ETH core. Beware that `-k` matches substrings, so
`-k "traced_128"` also selects `traced_128k`; use `-k "traced_128 and not traced_128k"`.

**Known limitation:** attention decode at **B=32 with TP=8** returns wrong results on
Wormhole (worst per-user PCC 0.019). B=1/2/4/8/16 are all ~0.9999 and the GDN state is
exact even at B=32, so it is attention-specific. Blackhole CI only exercises B=32 at TP=4.
Batch 1 and 8 are unaffected.

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

# A single ISL, e.g. the short 128-token traced case.
# NOTE: -k matches SUBSTRINGS, so a bare "traced_128" also selects traced_128k (and would run
# both in one process). Exclude the longer id explicitly:
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128 and not traced_128k"

# Medium / long traced ISLs (these ids are not prefixes of another, so they need no guard)
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_4k"
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_64k"
```

> Prefer **one pytest process per ISL** over `-k "traced"`: chaining several demo
> configs in a single process has wedged an ETH core on an 8-device mesh (the same reason
> `tests/pipeline_reorg/models_e2e_tests.yaml` gives each case its own process).

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

Run the 27B TP suite (with `HF_MODEL=Qwen/Qwen3.6-27B`, `Qwen/Qwen3.5-27B` or
`Qwen/Qwen3.8-27B`, and `MESH_DEVICE=P150x4`, `P150x8` or `T3K`):

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
