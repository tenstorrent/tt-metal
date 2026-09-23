<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8-27B on Wormhole (T3K / LoudBox / QuietBox)

Text-only inference for **Qwen3.8-27B**, the hybrid **Gated DeltaNet + Gated Full
Attention** member of the Qwen3.5/3.6/3.8 family, on a Wormhole **T3K** — 4x n300 =
8 WH_B0 chips — at **8-way tensor parallelism**.

The model implementation is shared with the Blackhole wrapper and lives in
[`models/demos/qwen36/tt`](../../qwen36/tt). This directory holds only the Wormhole
entry points (`demo/`, `tests/`). The Blackhole counterpart is
[`models/demos/blackhole/qwen36`](../../blackhole/qwen36).

Qwen3.8-27B is **architecturally identical to Qwen3.6-27B** — the two HF `config.json`
files differ only in `transformers_version`, including the full 64-entry `layer_types`
list — so the shared core needs no model-specific code, only `HF_MODEL`.

## Architecture

`tok_embeddings -> 64 x DecoderLayer -> RMSNorm -> LM Head`, where `layer_types` from the
HF config interleaves **48 Gated DeltaNet** layers (linear attention; fixed-size recurrent
state + causal conv, no KV growth) with **16 Gated Full Attention** layers (paged KV
cache), in a 3:1 pattern. Zero-centered RMSNorm throughout and **partial** RoPE (64 of
each 256-wide head is rotated). Only the 16 full-attention layers hold a KV cache, which
is what makes 262k context affordable on 12 GB/chip.

## Environment

```bash
export HF_MODEL=Qwen/Qwen3.8-27B
export MESH_DEVICE=T3K
export TT_CACHE_PATH=$HOME/tt_cache/Qwen3.8-27B
```

> **Do not set `WH_ARCH_YAML`** — it is obsolete and no longer read; ttnn selects Ethernet
> dispatch automatically for `ClusterType::T3K`. **Do not set `TT_MESH_GRAPH_DESC_PATH`**
> unless you deliberately want the 1x8 ring descriptor; a mismatched value can *silently
> hang* in the first collective rather than erroring.

Pre-flight — 8 chips whose inter-board ethernet has not trained report `ClusterType.N300`,
and everything T3K-shaped then degrades or hangs:

```bash
python3 -c "
import ttnn
print(ttnn.get_num_devices(), ttnn.cluster.get_cluster_type())   # expect 8 ClusterType.T3K
print(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())     # expect MeshShape([2, 4])"
```

`tt-smi -r 0,1,2,3` recovers a wedged ETH core (`Timed out waiting for ETH heartbeat`).

## End-to-end demo

```bash
# NOTE: -k matches SUBSTRINGS, so a bare "traced_128" also selects traced_128k and runs
# both in one process. Exclude the longer id explicitly:
pytest models/demos/wormhole/qwen38/demo/text_demo.py -v -s -k "traced_128 and not traced_128k"

# Longer ISLs (these ids are not prefixes of another, so they need no guard)
pytest models/demos/wormhole/qwen38/demo/text_demo.py -v -s -k "traced_4k"
pytest models/demos/wormhole/qwen38/demo/text_demo.py -v -s -k "traced_32k"

# Two runs, asserts identical token streams (detects a raced GDN recurrent state)
pytest models/demos/wormhole/qwen38/demo/text_demo.py -v -s -k "determinism_128"
```

> Run **one pytest process per ISL**. Chaining several demo configs in a single process
> has wedged an ETH core on the 8-device mesh, which is why
> `tests/pipeline_reorg/models_e2e_tests.yaml` gives each case its own process.

## Measured performance (batch 1)

Two different things get measured here, and they are **not** interchangeable.

**Demo** (`demo/text_demo.py`) times the model forward plus first-token sampling:

| ISL  | TTFT    | t/s/u | BH QuietBox ref (TP=4) |
| ---- | ------- | ----- | ---------------------- |
| 128  | 0.57 s  | 16.8  | 0.15 s / 26.0          |
| 4k   | 1.47 s  | 16.8  | 2.03 s / 18.1          |
| 8k   | 3.18 s  | 16.4  | 4.62 s / 18.0          |
| 16k  | 6.44 s  | 16.3  | 9.47 s / 17.9          |
| 32k  | 13.76 s | 16.0  | 19.96 s / 17.6         |
| 128k | 56.45 s | 14.2  | 77.61 s / 16.7         |

**vLLM serving** (tt-inference-server benchmarks workflow, concurrency 1) times a full request.
This is what `models/model_targets.yaml` grades against:

| ISL  | TTFT       | t/s/u (1000/TPOT) | t/s (OSL/E2EL) |
| ---- | ---------- | ----------------- | -------------- |
| 128  | 1298.3 ms  | 16.89             | 14.5           |
| 1024 | 1395.6 ms  | 16.78             | 14.3           |
| 4k   | 2465.6 ms  | 16.50             | 12.6           |
| 32k  | 14856.3 ms | 15.90             | 5.6            |

Serving adds a roughly fixed **0.7-1.1 s per request** on top of the demo TTFT, and that cost does
not grow with prompt length (2.3x the demo at ISL 128, only 1.08x at 32k). It is **not** the model
forward: the demo takes the identical eager `prefill_masked_bucket` path at ISL 128 and still
reaches 0.57 s. Look for it in the serving layer -- tokenization, scheduling, page-table
construction, host<->device copies -- not in `tt/`.

Decode is healthy and agrees across both harnesses: 16.89 t/s/u through vLLM vs 16.8 in the demo.
Only `t/s` differs, because `output_throughput` divides by E2EL and therefore includes TTFT; do not
expect it to equal the per-token rate even at batch 1.

Concurrency >1 is measured but deliberately left ungraded: prefill runs one user at a time, so TTFT
at 8 users rises to ~9.9 s, roughly 8x the single-user figure.

## Tensor-parallel tests

All of these run on the `(1, 8)` mesh with `FABRIC_1D`, wired automatically by
`tests/test_factory.py::parametrize_mesh_tp` from `MESH_DEVICE`. PCC thresholds are in
`tests/pcc_thresholds.json`, keyed by test function name.

```bash
pytest models/demos/wormhole/qwen38/tests/test_rope_tp.py      -v -s   # partial RoPE
pytest models/demos/wormhole/qwen38/tests/test_mlp_tp.py       -v -s   # SwiGLU + reduce-scatter
pytest models/demos/wormhole/qwen38/tests/test_attention_tp.py -v -s   # gated attn, paged KV
pytest models/demos/wormhole/qwen38/tests/test_gdn_tp.py       -v -s   # Gated DeltaNet
pytest models/demos/wormhole/qwen38/tests/test_model_tp.py     -svq    # full 64-layer model
pytest models/demos/wormhole/qwen38/tests/test_generate_tp.py  -v -s   # answer oracle
```

## Wormhole-specific notes

Wormhole exposes **8x8 = 64 worker cores** (vs ~110 on a BH P150) and ~72 KB less L1 per
core, and a T3K has **one usable ethernet link** per chip pair (the other is reserved for
the dispatcher datapath). The shared core adapts automatically:

* `tp_common.fused_ccl_num_links()` -> 1 on WH, 2 on BH.
* `agmm_prefill_grid_default()` / `mmrs_prefill_grid_default()` derive the fused-CCL grids
  from `compute_with_storage_grid_size()`, reserving the rows the all-gather muxes and
  reduce-scatter workers need. They reproduce the BH constants exactly on an 11x10 grid.
* `prefill_l1_output_ok()` is False on WH: the tuned prefill matmuls write their output to
  DRAM, because on 64 cores the output block plus the matmul's own circular buffers do not
  fit. `_PREFILL_TUNING` also drops `in0_block_w_cap` to 2.
* `QWEN36_GDN_FUSED=0` routes GDN prefill through the composite
  `chunk_gated_delta_rule_seq_adapter` instead of the fused op — the documented fallback.
  It costs roughly 2x on TTFT.

Device profiling shows cross-device collectives dominate: ~74% of MLP device time
(all-gather-matmul 48%, reduce-scatter 26%) against 25% for the matmuls themselves. That is
the single-link constraint, and the first place to look for decode headroom.

## Known limitations

* **Batch 32 at TP=8 returns wrong results.** Attention decode worst per-user PCC is 0.019,
  while batch 1/2/4/8/16 are all ~0.9999 and the GDN state is exact even at B=32 — so it is
  attention-specific. Blackhole CI only exercises B=32 at TP=4. **Batch 1 and 8 are
  unaffected**; use those.
* **Text only.** The vision tower and the sparse MoE (`Qwen3.6-35B-A3B`) paths are validated
  on Blackhole only — see [`models/demos/blackhole/qwen36`](../../blackhole/qwen36).
* **256k context is unvalidated** on Wormhole; 128k is the longest verified.
