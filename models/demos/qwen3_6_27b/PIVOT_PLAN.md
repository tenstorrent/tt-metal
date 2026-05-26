# Pivot Plan — Fork `llama3_70b_galaxy` into `models/demos/qwen3_6_galaxy/`

**Pattern**: same as `ssinghal/olmo3-galaxy-port`'s `models/demos/olmo_galaxy/`. Fork the production-grade BH-Galaxy infrastructure into a self-contained directory so changes don't risk regressing Qwen3-32B / Llama3-70B, and use **pad-and-slice** to handle the non-Qwen3-32B dimensions without rederiving every shard config.

## Directory layout (mirrors `olmo_galaxy`)

```
models/demos/qwen3_6_galaxy/
├── ARCHITECTURE.md              ← cross-reference qwen3_6_27b/ARCHITECTURE.md
├── BRINGUP_LOG.md
├── README.md
├── STATUS.md
├── conftest.py                  ← fabric init pattern from olmo_galaxy
├── tt/
│   ├── __init__.py
│   ├── llama_attention.py       ← FORKED, +output gate, +partial RoPE
│   ├── llama_mlp.py             ← FORKED, +pad/slice for intermediate
│   ├── llama_decoder.py         ← FORKED, +layer-type dispatch
│   ├── llama_model.py           ← FORKED, +dual cache (KV + DeltaNet state)
│   ├── llama_ccl.py             ← FORKED (or imported from llama3_70b_galaxy)
│   ├── llama_embedding.py       ← FORKED (or imported)
│   ├── lm_head.py               ← FORKED, +vocab pad to 248832
│   ├── llama_rope.py            ← FORKED, +MRoPE
│   ├── distributed_norm.py      ← FORKED, +zero-centered flag
│   ├── load_checkpoints.py      ← FORKED, +qwen3.6 weight remap (wq_gate)
│   ├── qwen36_model_config.py   ← NEW, based on qwen_model_config.py
│   ├── qwen36_deltanet.py       ← NEW, wraps validated kernels
│   └── prefetcher_common.py     ← FORKED (or imported)
├── reference/
│   ├── __init__.py
│   └── qwen36.py                ← standalone PyTorch oracle for module tests
├── tests/
│   ├── conftest.py              ← fabric init
│   ├── module_tests/
│   │   ├── test_qwen36_attention_ttt.py
│   │   ├── test_qwen36_deltanet_ttt.py
│   │   ├── test_qwen36_mlp_ttt.py
│   │   ├── test_qwen36_qk_norm.py
│   │   ├── test_qwen36_decoder_ttt.py
│   │   └── test_qwen36_transformer_ttt.py
│   └── test_qwen36_accuracy.py
└── demo/
    ├── text_qwen36_demo.py
    └── qwen36_text_demo_targets.json
```



## Update: `origin/ssinghal/olmo3-galaxy-port` validates the **pad-and-slice** approach

The Olmo3-32B port (40 Q-heads / 8 KV / intermediate=27648, also forked from `llama3_70b_galaxy`) handles
non-Qwen3-32B dimensions by **padding up to the existing tiled dimensions and slicing the result**:

| Olmo dim | Padding target | Mechanism |
|---|---|---|
| `n_local_heads=5` per chip | 8 (Qwen3-32B match) | Zero-pad Q proj along head axis; corresponding WO input padded `5120 → 8192`; output is mathematically identical because zero heads contribute nothing |
| `intermediate_dim_per_tp=3456` | 3840 (24-core aligned) | Pad FF1/FF3 weight output, FF2 input; slice FF2 output back to 3456 |
| AGMM variant | 3584 = 112 tiles | Alternate alignment for fused all-gather-matmul |

This **preserves every memcfg, program config, and matmul shard spec** from the Qwen3-32B path. The configs see the same tiled shapes; only the zero-padding regions waste compute (~ratio of pad to native size).

Applying to Qwen3.6-27B:

| Dim | Qwen3.6-27B native | Qwen3-32B target | Pad amount per chip |
|---|---:|---:|---:|
| n_q per col (4 cols) | 6 | 16 | +10 zero heads/col |
| n_kv per col | 1 | 2 | +1 zero head/col |
| intermediate per row (8 rows) | 2176 | 3200 (or 3840) | +1024 (or +1664) |
| vocab | 248320 | n/a (larger; pad to 248832 for 32×tile alignment) | +512 |
| **head_dim** | **256** | **128** (cannot pad down) | **isolated re-derivation** |

`head_dim=256` is the one dim we can't pad to match Qwen3-32B. The ~10 hardcoded sites in `llama_attention.py` (e.g. `shape=(64, 128)`, `[1, 1, 64, 128]` reshapes) need editing. Localized — not systemic.

This **eliminates risks #1, #2, #3** from the foresight report. The remaining risks (output gate breaking AGMM fusion, MRoPE complexity, DeltaNet plumbing, regression discipline) stand.

**Revised effort estimate: 6-9 weeks for correctness** (was 8-12), still 2-4 weeks for performance. Total ~3 months.

---

## Why

Hand-rolling mesh-sharded blocks under `models/demos/qwen3_6_27b/tt/` hit a wall at PCC=0.78 on the 8×4 DeltaNet block. `models/demos/llama3_70b_galaxy/` already has a **production-grade BH-Galaxy infrastructure** with:

- Distributed RMSNorm (4-way cross-col stats)
- TP-fractured SwiGLU MLP with line reduce-scatter / all-gather
- Fused QKV with QK-norm + paged KV + SDPA + line all-reduce
- Centralized CCL primitives that switch between WH 6U Ring (4 links) and BH GLX Linear (1 link)
- DRAM-sender / L1-receiver prefetcher pipeline (active on TG, bypassed on BH)
- BH_GLX detection (`is_blackhole()`) and tuned `GALAXY_NUM_LINKS=1`, `CCL_TOPOLOGY=Linear`
- A full Qwen3-32B variant in `qwen_model_config.py` (`TtQwenModelArgs`)
- Test harness: per-module `test_qwen_*_ttt.py` PCC tests, full-model accuracy test on 32-batch 8×4 mesh
- Demo runner with BH/6U-aware expected outputs

We **reuse** all of this and add only the qwen3.6-next-specific deltas.

## What to reuse as-is

| File | Use |
|---|---|
| `tt/llama_ccl.py` | All collectives (line reduce-scatter, line all-gather, line all-reduce, fused RS+create_heads). No changes. |
| `tt/distributed_norm.py` | 4-way distributed RMSNorm. Will need zero-centered (1+w) hook (see below). |
| `tt/llama_embedding.py` | TP-sharded embedding. Zero changes. |
| `tt/prefetcher_common.py` | Already has `is_blackhole()` BH GLX branch. No changes. |
| `tt/lm_head.py` | TP-split LM head with line all-reduce. Verify vocab size; otherwise reuse. |
| `tt/load_checkpoints.py` | HF→meta name remap. Cherry-pick `convert_hf_to_meta_qwen3_5` / `wq_gate` / linear_attn keys from `origin/ssinghal/qwen3.5-27B` already in our branch. |
| Test harness scaffold | Module test fixtures, accuracy test fixture, mesh setup, paged KV setup. Reuse signatures; swap reference module + thresholds. |

## What to lift-and-extend

### 1. `qwen_model_config.py` → add `TtQwen36ModelArgs`

Create a sibling class (or new file `qwen36_model_config.py`) that inherits and overrides:

| Param | Qwen3-32B | Qwen3.6-27B |
|---|---|---|
| `dim` | 5120 | 5120 |
| `n_q_heads` | 64 | **24** |
| `n_kv_heads` | 8 | **4** |
| `head_dim` | 128 | **256** |
| `intermediate_dim` | 25600 | **17408** |
| `padded_vocab_size` | 155648 | **248320** (or 248352 = 32×7761) |
| `n_layers` | 64 | 64 |
| `attn_output_gate` | False | **True** |
| `partial_rotary_factor` | 1.0 | **0.25** (rotary_dim=64) |
| `linear_attention_pattern` | n/a | **`hf_cfg.text_config.layer_types`** |

**Replace hardcoded literals** in `qwen_model_config.py` (`1280`=dim/4, `3200`=intermediate_dim/8, `128`=head_dim, `8`=n_kv_heads): refactor as computed `self.dim // self.dim_tp_factor` etc. The 64 Q-head reshape constants in `XQKV/QK-norm` paths must be re-derived for n_q=24.

**Critical divisibility recheck:**
- n_q=24 across `cluster_shape[1]=4` → 6 Q-heads/col ✓
- n_kv=4 across `cluster_shape[1]=4` → 1 KV-head/col ✓
- intermediate=17408 across `cluster_shape[0]=8` → 2176/row ✓
- head_dim=256 = 8 tiles ✓
- `num_devices_per_group = n_kv_heads if TG else num_devices` (line 278) → `4 if TG else 32`; verify all `n_kv_heads // cluster_shape[1]` arithmetic still works.

**QKV size with output gate:**
```
qkv_size_no_gate   = head_dim * (2*n_kv_heads + n_heads)        = 256*(2*4+24) = 8192
qkv_size_with_gate = head_dim * (2*n_kv_heads + 2*n_heads)      = 256*(2*4+2*24) = 14336
```
Use the gated variant; treat the second n_heads slice as `wq_gate` in the create-heads helper.

### 2. `llama_attention.py` → add output-gate + partial-RoPE branch

Add `attn_output_gate` and `partial_rotary_factor` configuration paths. Two routes — pick whichever is cleaner:

**Option A** (preferred): keep one fused QKV+gate weight `[Q|Q_gate|K|V]`, extend `create_qkv_heads` to emit `(q, q_gate, k, v)`. Use existing fused RoPE kernel on `(q, k)` over the *full* head_dim but only build cos/sin tables for the rotary_dim=64 slice (pass-through last 192 dims). Apply `sigmoid(q_gate) * attn_out` element-wise before `wo`.

**Option B**: keep `wq`, `wk`, `wv` fused as before AND a separate `wq_gate` matmul. Less ideal because it costs an extra projection but cleanly separates the gate path.

Reshape/shard constants `[1,8,8,128]`, `(64,128)` baked at lines 239–483 need to be re-derived from `(n_q, head_dim) = (24, 256)`.

### 3. `llama_rope.py` → MRoPE [11,11,10] + partial-rotary slicer

Replace the current fused full-RoPE call with:
- Precompute three sets of cos/sin tables (one per axis T/H/W) at sizes `(11, 11, 10)` of the 64-dim rotary slice.
- At apply time: slice q[..., :64], do per-axis `mul`-based rotation, concat with q[..., 64:] pass-through. Same for k.

This is the biggest net-new kernel work but straightforward — no flash-attn or fused kernel needed at this scale.

### 4. `llama_decoder.py` → layer-type dispatch

In `TtTransformerBlock.__init__`:
```
if config.linear_attention_pattern[layer_idx] == "full_attention":
    self.attention = TtQwen36GatedAttention(...)
else:
    self.attention = TtQwen36DeltaNet(...)
```

Forward signature gains a state-cache plumbing parallel to `kv_cache` (DeltaNet stores recurrent state, not KV).

### 5. `llama_model.py` → dual cache + MRoPE position triples

- Layer construction: pass `linear_attention_pattern` so each layer self-selects.
- Cache allocation: build `kv_caches` only for full-attention indices, `state_caches` for linear-attention indices. Or unify both as a generic per-layer-typed cache list.
- MRoPE: scalar `current_pos` → triple `(pos_t, pos_h, pos_w)`. Vision token positions need (T,H,W) coords; text tokens get `(t, 0, 0)`. Build position table once at prefill.

### 6. `distributed_norm.py` → zero-centered weight option

Qwen3-Next uses `(1+w) * norm(x)` for `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, and the final model `norm`. Add a `zero_centered: bool` constructor flag — when true, pre-add 1 to the loaded weight before storing. Drop-in for the existing API.

## Net-new components

### A. `tt/qwen36_deltanet.py`

The only block with no analog. Wraps the validated kernels from `models/experimental/gated_attention_gated_deltanet/tt/` (already cherry-picked in our branch):

- Input projections (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`): mesh-sharded across rows on the V-head axis — **use the same TP fracturing helpers** that `llama_mlp.py` uses (`ttnn.linear` + reduce-scatter pattern) rather than rolling our own `MeshMapperConfig` calls. Look at how `llama_mlp.py` lines 30–66 load `w1`, `w3` and follow that pattern exactly.
- Conv1d (depthwise, k=4): apply the **per-row interleaved layout** fix already discovered (the naive `cat([Q_conv, K_conv, V_conv])` doesn't shard cleanly because Q/K/V have different widths; need per-row chunks before stacking).
- Per-row kernel call: invoke the validated `chunk_gated_delta_rule_ttnn` (or `recurrent_gated_delta_rule_ttnn` for decode) — proven at PCC=0.999985 on full 8×4 mesh (T3.1).
- GQA expand (K from 16→48 V-heads): TTNN doesn't have `repeat_interleave`; emulate via `unsqueeze` + `repeat` + `reshape` (on device).
- GroupRMSNormGated (per-head_dim RMSNorm gated by SiLU(z)): construct from `ttnn.rms_norm` + `ttnn.silu` + `ttnn.multiply` — see `models/experimental/gated_attention_gated_deltanet/torch_functional/gated_deltanet.py::rms_norm_gated` for the math.
- `out_proj`: row-parallel matmul (input dim sharded across rows), followed by **`llama_ccl.line_all_reduce(cluster_axis=0)`** for the reduce step. **Use the centralized CCL helper, not raw `ttnn.all_reduce`** — that's where my earlier attempt went wrong.

State cache: per-layer `[B, n_v_heads, head_k_dim, head_v_dim]` FP32 tensor sharded along V-heads. Persistent across decode steps. Reset between requests.

### B. `tt/qwen36_gated_attention.py` (or fork of `llama_attention.py`)

If we choose to fork rather than extend `llama_attention.py`, this is the file. Contains the gate-split fused-QKV path, partial-RoPE branch, and MRoPE position handling. Recommendation: **extend `llama_attention.py` with a flag** rather than fork — the existing file is 1000+ LOC of well-tested mesh primitives.

### C. `reference/qwen36.py`

The existing `reference/qwen.py` (used by `tests/module_tests/test_qwen_attention_ttt.py`) is a standalone PyTorch Qwen3-32B reference. Need a sibling `reference/qwen36.py` (or `qwen3_5.py`) with:
- `GatedAttention` (Q+gate split, partial RoPE, zero-centered QK-norm)
- `GatedDeltaNet` (the FLA-style recurrence — already cherry-picked from user's branch at `models/demos/qwen3_6_27b/reference/gated_delta_net.py` — refactor into this file)
- `HybridDecoderLayer` with type dispatch
- `Qwen36Model` end-to-end

This is the PCC oracle for all module tests.

## Test plan (mirrors existing `test_qwen_*_ttt.py` pattern)

| Test file | What | Reuse from |
|---|---|---|
| `tests/module_tests/test_qwen36_attention_ttt.py` | Gated-Attention block PCC vs `reference.qwen36.GatedAttention` | Adapt `test_qwen_attention_ttt.py` |
| `tests/module_tests/test_qwen36_deltanet_ttt.py` | DeltaNet block PCC vs `reference.qwen36.GatedDeltaNet` | New — pattern from above |
| `tests/module_tests/test_qwen36_mlp_ttt.py` | Same as `test_qwen_mlp_ttt.py` but with intermediate=17408 | Existing |
| `tests/module_tests/test_qwen36_qk_norm.py` | Zero-centered QK-norm PCC | Adapt `test_qwen_qk_norm.py` |
| `tests/module_tests/test_qwen36_decoder_ttt.py` | Hybrid layer dispatch + forward | Adapt `test_qwen_decoder_ttt.py` |
| `tests/module_tests/test_qwen36_transformer_ttt.py` | Stacked layers PCC | Adapt `test_qwen_transformer_ttt.py` |
| `tests/test_qwen36_accuracy.py` | Full 32-batch end-to-end accuracy on 8×4 BH GLX | Adapt `test_qwen_accuracy.py` |
| `demo/text_qwen36_demo.py` | Generation demo with BH/6U-aware target outputs | Adapt `demo/text_qwen_demo.py` |
| `demo/qwen36_text_demo_targets.json` | Expected outputs (capture on first known-good run) | New |

## Execution order

1. **Branch hygiene**: new work lives in `models/demos/qwen3_6_galaxy/` (standalone fork, mirror of `olmo_galaxy`). The existing `models/demos/qwen3_6_27b/` dir keeps documentation only (ARCHITECTURE.md, BRINGUP_LOG.md, TEST_PLAN.md, QUALIFICATION_PLAN.md, PIVOT_PLAN.md). The single-chip prototype code under `qwen3_6_27b/tt/` is kept as historical reference; production code goes into `qwen3_6_galaxy/`.
2. **Reference module** (`reference/qwen36.py`): port Qwen3.6 reference here from our existing `models/demos/qwen3_6_27b/reference/` and the cherry-picked PyTorch DeltaNet kernels. CPU only.
3. **`qwen36_model_config.py`**: clone `TtQwenModelArgs`, parametrize over the dim/head differences, add the new flags. Verify divisibility asserts pass.
4. **`distributed_norm.py` zero-centered flag**: ~10 line addition. PCC unit test.
5. **`llama_attention.py` output-gate + partial-RoPE branch**: gated by `attn_output_gate` flag. Test with `test_qwen36_attention_ttt.py` (output-gated Qwen3.6 layer-3 weights). Target PCC > 0.99.
6. **`llama_rope.py` MRoPE [11,11,10]**: new precompute + apply. Unit test against HF `Qwen3VLRotaryEmbedding`.
7. **`qwen36_deltanet.py`**: wrap the validated kernels with the **`llama_ccl` primitives**. Test with `test_qwen36_deltanet_ttt.py`. Target PCC > 0.99.
8. **`llama_decoder.py` dispatch + `llama_model.py` dual cache**: small surgical changes. Test hybrid 4-layer slice on full 8×4 mesh.
9. **Full 64-layer accuracy** on 8×4 mesh with 32-batch: `test_qwen36_accuracy.py`. Target same thresholds as Qwen3-32B (min_top1=81, min_top5=98).
10. **Demo**: `text_qwen36_demo.py` produces " Paris" for "The capital of France is" on full 64 layers, properly sharded.
11. **Performance**: with everything on-device + sharded, target ≥10 tok/s decode initial (still without trace); then add trace capture and aim for 25 tok/s.

## Key risks

| Risk | Mitigation |
|---|---|
| Hardcoded n_q=64 / head_dim=128 reshapes throughout `llama_attention.py` | Replace with `self.n_q` / `self.head_dim`; touched lines must all be rederived. Plan a single audit pass before writing new tests. |
| Qwen3-32B baseline tests still need to pass (don't regress) | All Qwen3.6-specific behavior gated behind `attn_output_gate`/`partial_rotary_factor < 1.0`/`linear_attention_pattern is not None` flags. Default-off keeps Qwen3-32B path identical. |
| MRoPE kernel performance | First pass: build from primitives (mul-rotate-concat). Optimize later if needed; not critical-path for first PCC. |
| DeltaNet state cache layout interacts with paging | DeltaNet has no KV cache, only recurrent state. Cache plumbing in `llama_model.py` needs a per-layer-type branch. |
| ccl `all_reduce` initialization (fabric setup) | Use `llama3_70b_galaxy`'s existing test conftest pattern — `set_fabric_config(FABRIC_1D_RING, …)` before opening the full 8×4 system mesh, then create submeshes if needed. That's what the existing Qwen3-32B tests use successfully. |
| Vocab padding mismatch (Qwen3.6 = 248320 vs Qwen3-32B = 155648) | Update `padded_vocab_size` in `_set_params_from_dict`. May require re-tiling LM head; vocab is a power of 2 multiple of 32 (248320 = 32×7760) so it should divide cleanly across 32 chips for vocab-parallel. |

## What we keep from the previous (single-chip) work

- ✅ **PyTorch reference** (`reference/gated_delta_net.py`, `reference/model.py`): port into `models/demos/llama3_70b_galaxy/reference/qwen36.py`.
- ✅ **HF weight remap** (`load_checkpoints.py` qwen3_5 helpers): already cherry-picked and proven; just merge into `llama3_70b_galaxy/tt/load_checkpoints.py` (or call from there).
- ✅ **DeltaNet kernels** (`models/experimental/gated_attention_gated_deltanet/`): drop-in.
- ✅ **All ARCHITECTURE.md / QUALIFICATION_PLAN.md / TEST_PLAN.md insights**: still valid; just adjust file paths to point to galaxy demo for production code.
- ✅ **"Paris" prediction validation**: the single-chip path proves the math; we're now changing the infrastructure, not the math.
- ❌ **`models/demos/qwen3_6_27b/tt/*.py`**: superseded. Keep as historical reference in the commit but don't extend.

## Estimated effort (single engineer)

| Item | Days |
|---|---|
| Reference module + tests | 1-2 |
| `qwen36_model_config.py` | 2-3 |
| Distributed-norm zero-centered flag | 0.5 |
| `llama_attention.py` output gate + partial RoPE | 3-4 |
| MRoPE in `llama_rope.py` | 2-3 |
| `qwen36_deltanet.py` block | 4-5 |
| Decoder dispatch + dual cache plumbing | 2 |
| Module tests (one per block) | 2 |
| Full accuracy test + debug iteration | 3-5 |
| Demo wiring + targets capture | 1 |
| Performance: trace + Tracy profile per block | 5-7 |
| **Total** | **~5-7 weeks** |

Vision, MTP, and tt-inference-server integration are still deferred to a phase 2 after the text path lands.
