# DFlash drafter: device operator mapping (T3K)

Maps every device op in one step of the ttnn DFlash drafter ([tt/dflash/drafter.py](tt/dflash/drafter.py),
`TtDFlashDrafter`) back to the ttnn call that emits it. The 27B target is not in the capture; the
draft-logit projection runs on the target's LM head (`TtTarget.lm_head_device`) and is outside the
window as well.

## How to produce the report

```bash
MESH_DEVICE=T3K DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \
  python -m tracy -p --op-support-count 100000 -r -v -m \
    pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_dflash_drafter.py::test_profile_dflash_drafter[wormhole_b0-device_params0-1x8-ctx64]"

D=<new report dir under generated/profiler/reports/>
tt-perf-report generated/profiler/reports/$D/ops_perf_results_$D.csv \
  --start-signpost start --end-signpost stop --no-color
```

[test_profile_dflash_drafter.py](tests/perf/test_profile_dflash_drafter.py) warms the drafter to
`ctx` rows of history, uploads the step's taps and noise block outside the window, and signposts
`project_taps` + `forward`. It builds the drafter without `ctx_capacity`, so the capture is the
growing-history KV path described below.

A capture of the drafter can be recognised by its op-count signature, which follows from the source:

| Expected from source | Count |
| --- | --- |
| `fc` + 6 projections x 5 layers (`k` and `v` are one fused `kv_proj`) | 31 Matmul |
| `hidden_norm` + (`input_layernorm`, `q_norm`, `k_norm`, `post_attention_layernorm`) x 5 + final `norm` | 22 LayerNorm |
| the tap gather (the drafter is replicated) | 1 AllGatherAsync |
| one per layer | 5 SDPA |
| q and k per layer | 10 RotaryEmbeddingHf |

## Run configuration

| | | Source |
| --- | --- | --- |
| Mesh | T3K, `(1, 8)` | `parametrize_mesh_tp()` |
| Parallelism | none; replicated on all 8 devices | [tt/dflash/weights.py](tt/dflash/weights.py) |
| Block | 16 slots (1 anchor + 15 drafted) | `cfg.block_size` |
| New context rows per step | 16 | test `n_new` |
| Committed KV history | 64 rows, so SDPA k/v length is 96 (64 + 16 + 16) | `ctx64` param |
| Hidden / heads | 5120; 32 q heads, 8 kv heads, head_dim 128 | GQA 4:1, all local |
| MLP intermediate | 17408 | SwiGLU |
| Layers | 5: 4 causal sliding-window (2048) + 1 bidirectional | `cfg.layer_types` |
| Weight dtype | `BFLOAT8_B`, except `BFLOAT4_B` for MLP gate/up; DRAM-interleaved | `PROJ_DTYPE`, `MLP_DTYPE`, `MLP_DOWN_DTYPE` in [weights.py](tt/dflash/weights.py) |
| Activation dtype | `BFLOAT16`, DRAM-interleaved unless noted | |
| Math fidelity | matmul / SDPA HiFi2 (`fp32_dest_acc_en`), norms / RoPE HiFi4 | `compute_cfg` |

Every activation matmul runs at M = 32 padded for 16 logical rows (`32[16]` in the report), except
`kv_proj`, whose input is 16 context + 16 block rows.

## Prologue: `project_taps`

| Op code | ttnn op: block / purpose / config | In shape | Out shape | Weight dt / fid |
| --- | --- | --- | --- | --- |
| `Concat` | `ttnn.concat(taps, dim=-1)`: joins this device's 5 tap slices | 5 x `16x640` | `16x3200` | — |
| `AllGatherAsync` | `tt_all_gather(cluster_axis=None, dim=3)`: the drafter's only collective. Produces device-major columns, which is why `fc`'s rows are permuted at load time (`reorder_fc_rows`) | `16x3200` DRAM | `16x25600` DRAM | — |
| `Matmul 32 x 25600 x 5120` | `ttnn.linear(gathered, weights.fc)`: the tap projection, auto program config | `16x25600` BF16 | `16x5120` BF16 | BFP8 / HiFi2 |
| `LayerNorm` | `hidden_norm`: standard RMSNorm, gain used as-is (not the target's zero-centred form) | `16x5120` | `16x5120` | BF16 / HiFi4 |

## Per-step setup: RoPE slice and mask (`forward`)

| Op code | ttnn op: block / purpose / config |
| --- | --- |
| `Slice` x2 | `_rope_slice`: cos/sin for `[start - n_new, start + q_len)` sliced from the resident ROW_MAJOR tables |
| `Tilize` x2 | `ttnn.to_layout(TILE)` on the sliced cos/sin; the slice is not tile-aligned, so it is taken in ROW_MAJOR |
| `Slice` x2, `TilizeWithValPadding` x2 | Q's cos/sin for `[start, start + q_len)`; q_len 16 pads to a tile |
| `UntilizeWithUnpadding` | `ctx_rm = ttnn.to_layout(kv_source, ROW_MAJOR)`, once per step, shared by all five layers |
| `TilizeWithValPadding` | `_sliding_mask` upload. Built once per step: all four sliding layers see the same `(q_len, kv_len)` because their histories commit in lockstep |

## Repeating layer (x5)

Source: `_layer_attention` and `_layer_mlp`, called from `forward`.

### Attention (`_layer_attention`)

| Op code | ttnn op: block / purpose / config | In shape | Out shape | Weight dt / fid |
| --- | --- | --- | --- | --- |
| `LayerNorm` | `input_layernorm` on the noise branch only; the K/V source goes to `kv_proj` un-normed, matching the reference drafter | `16x5120` | `16x5120` | BF16 / HiFi4 |
| `UntilizeWithUnpadding`, `Concat`, `Tilize` | `ttnn.concat([ctx_rm, hidden_rm], dim=-2)`: K/V source is context then block. Both are 16 rows (part-tile), so the row concat is done in ROW_MAJOR | `16x5120` + `16x5120` | `32x5120` | — |
| `Matmul 32 x 5120 x 4096` | `q_proj`: Q from the block only. Explicit 1D (mcast_in0) program config on the 8x8 grid (`_proj_pc`), L1 output | `16x5120` | `16x4096` L1 | BFP8 / HiFi2 |
| `Matmul 32 x 5120 x 2048` | fused `kv_proj`: k and v weights concatenated on the output dim at load time (`_kv_proj` in weights.py). Same explicit 1D config, L1 output | `32x5120` | `32x2048` L1 | BFP8 / HiFi2 |
| `ReshapeView`, `Transpose`, `Copy` | `_heads` for Q: `[1,1,16,4096] -> [1,32,16,128]`, then `to_memory_config(DRAM)` | `16x4096` | `32x16x128` | — |
| `NlpCreateHeads` | `_kv_heads`: `nlp_create_qkv_heads(kv_tied=True)` splits the fused `[k\|v]` block into head-major K and V in one op | `32x2048` | 2 x `8x32x128` | — |
| `LayerNorm` | `q_norm` over head_dim | `32x16x128` | same | BF16 / HiFi4 |
| `LayerNorm` | `k_norm` over head_dim | `8x32x128` | same | BF16 / HiFi4 |
| `Copy`, `RotaryEmbeddingHf` | `apply_partial_rope_prefill(k, ...)`: full 128-dim rotation, HF half-split. `rope_dim == head_dim`, so no pass-through concat | `8x32x128` | same | — / HiFi4 |
| `Copy`, `RotaryEmbeddingHf` | `apply_partial_rope_prefill(q, ...)`: Q at the block's trailing `q_len` positions | `32x16x128` | same | — / HiFi4 |
| `Concat` x2, `Slice` x2 | KV history: `concat([hist, k])` / `concat([hist, v])` for SDPA, then `slice` to `hist_len + n_new` to persist only the context rows; the block's own K/V is scratch | `8x64x128` + `8x32x128` | `8x96x128`; history `8x80x128` | — |
| `SDPA` | `scaled_dot_product_attention(attn_mask=mask if sliding else None, is_causal=False)`. `is_causal=False` because Q is the block only, so SDPA's own causal alignment would be wrong | q `32x16x128`, k/v `8x96x128` | `32x16x128` | — / HiFi2 |
| `Transpose`, `ReshapeView` | head concat back to `[1,1,16,4096]` | `32x16x128` | `16x4096` | — |
| `Matmul 32 x 4096 x 5120` | `o_proj`: plain local matmul (replicated, so no all-reduce), auto program config | `16x4096` | `16x5120` | BFP8 / HiFi2 |
| `BinaryNg` | `ttnn.add(residual, attn)` (in `forward`) | `16x5120` | `16x5120` | — |

### MLP (`_layer_mlp`)

| Op code | ttnn op: block / purpose / config | In shape | Out shape | Weight dt / fid |
| --- | --- | --- | --- | --- |
| `LayerNorm` | `post_attention_layernorm` (in `forward`) | `16x5120` | `16x5120` | BF16 / HiFi4 |
| `Matmul 32 x 5120 x 17408` | `gate_proj` with an explicit 1D config (`_proj_pc`) whose `fused_activation=SILU` fuses the SiLU into the matmul | `16x5120` | `16x17408` | BFP4 / HiFi2 |
| `Matmul 32 x 5120 x 17408` | `up_proj`, auto program config | `16x5120` | `16x17408` | BFP4 / HiFi2 |
| `BinaryNg` | `ttnn.mul(gate, up)`: SwiGLU gate | 2 x `16x17408` | `16x17408` | — |
| `Matmul 32 x 17408 x 5120` | `down_proj`: plain local matmul, no all-reduce, auto program config | `16x17408` | `16x5120` | BFP8 / HiFi2 |
| `BinaryNg` | `ttnn.add(residual, mlp)` (in `forward`) | `16x5120` | `16x5120` | — |

## Epilogue

| Op code | ttnn op: block / purpose / config |
| --- | --- |
| `LayerNorm` | final `norm`. Its output feeds the target's LM head (`TtTarget.lm_head_device` / `draft_ids_device`), which is outside this window |

## Fixed-capacity path

With `ctx_capacity=C` (the demo's configuration), `forward` takes the fixed-capacity branch. The
per-layer ops are the same except for the KV history: the accepted context rows are written in
place into a persistent `[1, 8, C, 128]` buffer (`Slice` + `slice_write`), and SDPA reads
`concat([buffer, k])` with a mask that hides the unwritten tail, instead of the history being
re-concatenated and re-sliced every step. The context is padded to a constant 16 rows, and RoPE and
both masks come from `_fixed_rope` / `_fixed_mask_tensors` at constant shapes. Every per-step shape is then
fixed except the block width.
