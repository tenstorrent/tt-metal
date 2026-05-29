# Width-sharded prefill embedding (Devstral2 / Ministral3)

This note documents the prefill `ttnn.embedding` memory-layout optimization: **WIDTH-sharded L1 output** aligned with **width-sharded RMSNorm**, replacing an earlier **HEIGHT-sharded** approach that caused L1 circular-buffer clashes on Blackhole.

## Problem (HEIGHT-sharded embed)

The first attempt used **HEIGHT-sharded** embedding on prefill:

| Setting | Value |
|--------|--------|
| Strategy | `HEIGHT_SHARDED` |
| Cores | 4 (single row above the 8×8 norm grid) |
| Per-core shard (seq=128, hidden=12288) | `[32, 12288]` ≈ **768 KiB** (BF16) |
| Placement | Row 8 (cols 0–3), outside RMSNorm rows 0–7 |

Issues:

1. **Huge per-core shards** — most of the hidden dimension on each of only 4 cores.
2. **Layout mismatch** — RMSNorm prefill expects **WIDTH-sharded** activations on an **8×8** grid; embed output needed `InterleavedToSharded` / DRAM staging before norm.
3. **L1 clashes** — `validate_circular_buffer_region` during `LayerNormDeviceOperation` when full hidden states were materialized in L1 interleaved on norm cores.

## Solution (WIDTH-sharded embed)

Prefill embedding output now uses the **same** memory config as prefill RMSNorm activations:

```python
get_prefill_width_sharded_embedding_mem_config(seq_len, hidden_size)
# → get_prefill_width_sharded_activation_mem_config(seq_len, hidden_size)
```

Wiring in `TtMinistral3Model`:

```python
embed_mem = self.args.get_embedding_output_mem_config(mode, mesh_device, batch_size=..., seq_len=...)
hidden_states = self.embed_tokens(input_ids, memory_config=embed_mem)
hidden_states = self._reshape_embeddings(hidden_states, input_ids, mode=mode)
# first layer: input_layernorm — skips interleaved_to_sharded when layout already matches
```

### Prefill config (seq=128, hidden=12288)

| Parameter | Value |
|-----------|--------|
| Op | `ttnn.embedding` (`EmbeddingsDeviceOperation`, tilized) |
| Output layout | `TILE_LAYOUT` |
| Memory layout | `WIDTH_SHARDED` |
| Buffer type | L1 |
| Core grid | **8×8** (64 cores), from `_pick_prefill_width_shard_grid` |
| Shard shape (per core) | `(M_padded, hidden_padded / 64)` = **`(128, 192)`** |
| Per-core size (BF16) | 128 × 192 × 2 ≈ **49 KiB** |
| Norm program config | `get_prefill_width_sharded_norm_program_config(128, 12288)` — `block_h=4`, `block_w=6`, `subblock_w=3` |
| Norm compute kernel | HiFi2, `fp32_dest_acc_en=True`, `packer_l1_acc=False` |

**Gating:** WIDTH embed is used when `mode == "prefill"` and `seq_len <= kv_block_size` (default **128**). Longer prefill chunks fall back to `L1_MEMORY_CONFIG` (interleaved).

### Decode config (unchanged)

| Parameter | Value |
|-----------|--------|
| Embed output | `ttnn.L1_MEMORY_CONFIG` (interleaved) |
| Reason | Tilized embed validation requires `input_volume % shard_height == 0`. Decode has **M=1** token while width-sharded norm uses **M_padded = TILE_SIZE (32)**. |

Decode RMSNorm still uses `get_decode_width_sharded_activation_mem_config` → shard **`(32, 384)`** on a **4×8** (32-core) grid.

## Why WIDTH fits better than HEIGHT

| | HEIGHT (old) | WIDTH (current) |
|---|-------------|-----------------|
| Cores | 4 | 64 (same as prefill norm) |
| Per-core shard | `[32, 12288]` | `[128, 192]` |
| Per-core L1 | ~768 KiB | ~49 KiB |
| Matches RMSNorm? | No | **Yes** — same `MemoryConfig` |
| Extra DM ops before norm | Often `InterleavedToSharded` + staging | **None** on prefill token embed |

TTNN supports WIDTH-sharded tilized embedding outputs (`tests/ttnn/unit_tests/operations/data_movement/test_embedding.py::test_embedding_tiled_sharded_output`).

## Prefill norm → matmul (width-sharded, QKV + gate + up)

Default for prefill when `seq_len <= kv_block_size` (default **128**). Linears with
`K == hidden_size` consume **WIDTH-sharded** RMSNorm output on the **same 8×8 grid** as embed/norm,
run `MatmulMultiCoreReuseMultiCast1DProgramConfig` (`mcast_in0=True`), write **WIDTH-sharded** L1
output, then `sharded_to_interleaved` for the next op (QKV heads, SiLU/mul). Down proj, o_proj, and
decode paths unchanged.

### Matmul shapes (local, TP=8)

| Matmul | Shape | Fused activation |
|--------|-------|------------------|
| QKV | `128 × 12288 × 1792` | — |
| gate | `128 × 12288 × 3584` | SiLU (program config only) |
| up | `128 × 12288 × 3584` | — |

### Memory layout

| Tensor | Layout | Buffer | Notes |
|--------|--------|--------|-------|
| in0 (activations) | `WIDTH_SHARDED` | L1 | Same shard as norm: `(128, 192)` per core on 8×8 |
| in1 (weights) | interleaved | DRAM | Unchanged |
| out | `WIDTH_SHARDED` | L1 | `get_prefill_width_sharded_matmul_output_mem_config()` |

Gating: `use_width_sharded_prefill_norm_matmul(args, mode, seq_len)` → prefill and
`seq_len <= kv_block_size`.

### Program config (`get_prefill_width_sharded_matmul_program_config`)

Family: **1D multicast** (`MatmulMultiCoreReuseMultiCast1DProgramConfig`), same grid as prefill norm
via `_pick_prefill_width_shard_grid` → **8×8 (64 cores)**. Uses 64 cores (not 56) so
`Kt = 384` divides `num_cores` for width-sharded in0 (`384 / 64 = 6` K-tiles per core).

Shared fields (seq=128, hidden=12288):

| Field | Value |
|-------|--------|
| `compute_with_storage_grid_size` | **8 × 8** |
| `mcast_in0` | `True` |
| `fuse_batch` | `True` |
| `per_core_M` | **4** (`Mt`, full M on each core) |
| `in0_block_w` | **6** (largest divisor of `Kt/num_cores` ≤ L1 CB cap) |
| `out_subblock_h` | **1** (required for width-sharded output) |
| Compute kernel | LoFi, `fp32_dest_acc_en=True`, `packer_l1_acc=True` (`get_compute_kernel_config_hifi4`) |

Per-matmul `per_core_N` / `out_subblock_w` (Nt sharded across 64 cores):

| Matmul | `Nt` | `per_core_N` | `out_subblock_w` |
|--------|------|--------------|------------------|
| QKV | 56 | 1 | 1 |
| gate / up | 112 | 2 | 2 |

Gate SiLU: pass `fused_activation=UnaryOpType.SILU` in the program config only — do **not** pass
`activation=` to `ttnn.linear` on sharded matmul.

Sweep reference (`matmul_mcast1D_width.py`, same QKV shape): best isolated kernel was
**1D l1/dram/ws, 8×7, in0_block_w=8** (~102 μs). Production uses **ws/dram/ws on 8×8** so in0 matches
norm without `ShardedToInterleaved`; grid 8×8 is required for `Kt % num_cores == 0`.

### Wiring

| File | Role |
|------|------|
| `tt/mem_config.py` | `get_prefill_width_sharded_matmul_program_config`, `use_width_sharded_prefill_norm_matmul` |
| `tt/tt_ministral3_decoder_layer.py` | `ws_norm_out_mem` for input + post-attention norm when gating is true |
| `tt/tt_ministralattn.py` | `_project_qkv_prefill` |
| `tt/tt_ministralmlp.py` | gate/up `_prefill_width_sharded_linear` |

PCC: `test_ministral3_decoder_layer.py::test_decoder_layer_prefill_pcc_real_weights`.

### Prefill matmul device time (one-layer trace, Blackhole 1×8)

From `decode_latest6.txt` / `decode_latest7.txt` (width-sharded path on, profiler):

| Matmul | Device time (typical) | Cores | in0 |
|--------|----------------------|-------|-----|
| QKV `128×12288×1792` | ~113 μs | 64 | width_sharded |
| gate `128×12288×3584` | ~131 μs | 64 | width_sharded |
| up `128×12288×3584` | ~115–117 μs | 64 | width_sharded |

Stacked summary: **3** `MatmulDeviceOperation (in0:width_sharded)` ≈ **360 μs** total; removes
norm→QKV / norm→gate `ShardedToInterleaved` (~6 μs each) vs interleaved-in0 path.

## Code locations

| File | Symbols |
|------|---------|
| `tt/mem_config.py` | `get_prefill_width_sharded_embedding_mem_config`, `get_embedding_output_mem_config` |
| `tt/model_args.py` | `Devstral2Args.get_embedding_output_mem_config` |
| `tt/tt_ministral3_model.py` | `TtEmbedTokens`, embed call in `TtMinistral3Model.__call__` |
| `tt/tt_ministralrmsnorm.py` | `get_prefill_width_sharded_activation_mem_config` in `_forward_prefill` |

## Measurement setup

- **Hardware:** Blackhole, **8-device** mesh (1×8), fabric 1D
- **Test:** `test_ministral3_model_pcc_devstral2_123B_instruct_partial_weights_one_layer_decode` — prefill **128** tokens, then **1** decode step, **one** decoder layer
- **Profiler:** Tracy (`python -m tracy -r -p -v -m pytest ...`)
- **Reports:** repo-root `decode_latest.txt` (before / interleaved+convert path) vs `decode_latest2.txt` (after / width-sharded embed)

### Device time (full trace)

| Metric | Before | After |
|--------|--------|-------|
| **Total device time** | **1982 μs** | **1986 μs** |

End-to-end device time is essentially flat (**+4 μs**, ~0.2%). The win is fewer layout conversions and correct L1 usage, not a large wall-clock drop on this short one-layer trace.

### Embedding-related ops (stacked Tracy summary)

| Op | Before (`decode_latest.txt`) | After (`decode_latest2.txt`) |
|----|----------------------------|------------------------------|
| `EmbeddingsDeviceOperation` (device time sum) | 55.32 μs (6 ops) | **29.78 μs** (6 ops) |
| `InterleavedToShardedDeviceOperation` (sum) | 20.84 μs (11 ops) | **16.31 μs** (10 ops) |

Prefill token embed in the detailed report moves from **4 cores / 49 μs** with a following **`InterleavedToSharded`** to **64 cores / ~23 μs** with **direct** width-sharded output into `LayerNormDeviceOperation`.

## How to reproduce

```bash
python -m tracy -r -p -v -m pytest \
  models/experimental/devstral2_123B_instruct/tests/test_ministral3_single_layer.py::test_ministral3_model_pcc_devstral2_123B_instruct_partial_weights_one_layer_decode
```

Inspect `generated/profiler/reports/*/ops_perf_results_*.csv` or the Tracy stacked report for `EmbeddingsDeviceOperation` and total device time.
