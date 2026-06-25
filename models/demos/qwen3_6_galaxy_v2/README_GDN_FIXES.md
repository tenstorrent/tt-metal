# Qwen3.6-27B GDN Bug Fixes

This document summarises the bugs found and fixes applied to the Qwen3.6-27B
GatedDeltaNet (GDN / linear_attention) prefill path on P150x4 (TP=4, 4-chip
Blackhole).

---

## Bug 1 — Forward substitution diagonal not zeroed (strict lower triangular)

**File:** `models/demos/qwen3_6_galaxy_v2/tt/qwen35_chunk_delta_rule_ops.py`

### Root cause

The chunk-parallel prefill kernel computes the GDN recurrence via a Neumann
series / forward substitution:

```
A = -(k_beta) × k × L_mask          # [chunk, chunk] lower-triangular "influence" matrix
v_corrected = (I - A)^{-1} v_beta   # solve by forward substitution
```

`L_mask[i, i] = exp(0) = 1`, so `A[i, i] = -(k_beta_i · k_i) ≠ 0`.  The delta
rule requires `A` to be **strictly** lower triangular (zero diagonal) so that
the `i`-th equation reads `v_c[i] = v_b[i] + A[i, :i] @ v_c[:i]` — no
self-referential term.  With the diagonal present every `v_corrected[t]` is
silently divided by `(1 + k_beta_t · k_t)`, a spurious ≈ 0.5–1 factor.

`L_mask` itself retains its diagonal because it is also used for the separate
**intra-chunk output attention** path (`attn_raw = qk × L_mask × lower_causal`),
which needs `exp(0) = 1`.

### Fix

```python
# After: A = ttnn.to_torch(attn_raw, ...).float()
A.diagonal(dim1=-2, dim2=-1).zero_()   # strict lower triangular before solve
```

A dedicated `strict_lower` TTNN mask is created (and cached alongside
`triu_ones` / `tril_mask`) and used as the forward-substitution mask in place
of `L_mask`.

### Impact

| Metric | Before fix | After fix |
|---|---|---|
| Primed-decode PCC | 0.64 | **0.9991** |
| 64L ISL=128 PCC | 0.9877 | **0.9996** |
| GDN min-layer PCC (ISL=128) | 0.97 | **0.9989** |

---

## Bug 2 — Decay applied after read instead of before (decode recurrence)

**File:** `models/demos/qwen3_6_galaxy_v2/tt/ttnn_delta_rule_ops_fp32.py`

### Root cause

The decode recurrent step has two sub-functions:

* `_fused_decay_and_write_fp32` — decays `h` and then writes the new
  outer-product update: `h_new = h * decay + outer * beta`
* `_recurrent_delta_rule_step_fp32` — reads the current state to compute the
  output: `v_read = k @ h`

The original code computed `v_read = k @ h` (un-decayed `h`) while
`_fused_decay_and_write_fp32` was writing `h * decay`.  When `h ≠ 0` the
read and write used inconsistent versions of `h`.

### Fix

```python
h_decayed = h * decay
v_read = k @ h_decayed          # read from decayed state (consistent with write)
h_new   = h_decayed + outer * beta
```

A `pre_decayed` flag was added to `_fused_decay_and_write_fp32` so that when
the caller already holds `h_decayed` the internal multiply is skipped.

---

## Bug 3 — BFP8 decay noise accumulates over long context (prefill PCC)

**File:** `models/tt_transformers/tt/qwen36_gdn_attention.py`

### Root cause

The GDN decay factor is computed as:

```
g = -exp(A_log) * softplus(a + dt_bias)
```

`A_log` and `dt_bias` were loaded with `dtype = self.dtype = bfloat8_b`.
BFP8 has only 2-3 mantissa bits; each chunk introduces a small quantisation
error in `g`, and these errors **accumulate multiplicatively** in the
recurrent state `S` over `T / chunk_size` chunks.

At `T = 128` (1 chunk) the error is negligible.  At `T = 8192` (32 chunks)
the accumulated noise degrades PCC from **0.9999 → 0.9874** — below the
mandatory 0.99 threshold.

All five GDN weight tensors were affected: `A_log`, `dt_bias`, `w_qkvz`,
`w_ba`, `w_out`.  The projection weights (`w_qkvz`, `w_ba`, `w_out`) are less
sensitive but were also upgraded to BF16 for consistency and to eliminate
related activation-dtype quantisation in the projection outputs.

### Fix

```python
# _build_weights — force BF16 regardless of self.dtype
self.w_qkvz = self._to_device(QKVZ_w_T, shard_out_dim1, dtype=ttnn.bfloat16)
self.w_ba   = self._to_device(BA_w_T,   shard_out_dim1, dtype=ttnn.bfloat16)
self.A_log  = self._to_device(A_log_3d, shard_head, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
self.dt_bias= self._to_device(dt_bias_3d, shard_head, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
self.w_out  = self._to_device(out_proj_w_T, shard_in_dim0, dtype=ttnn.bfloat16)

# _project_inputs — projection outputs also BF16
qkvz = ttnn.linear(x, self.w_qkvz, dtype=ttnn.bfloat16, ...)
ba   = ttnn.linear(x, self.w_ba,   dtype=ttnn.bfloat16, ...)
```

### Impact

| Metric | Before fix | After fix |
|---|---|---|
| Single GDN layer prefill PCC (T=128) | 0.9999 | 0.9999 |
| Single GDN layer prefill PCC (T=8192) | **0.9874** (FAIL) | **0.9999** ✓ |
| 64L composite PCC (ISL=8192) | — | **0.9999** ✓ |
| GDN min-layer PCC (ISL=8192) | — | **0.9998** ✓ |
| Memory overhead | — | ~30 MB/layer/chip extra (negligible on 128 GB DRAM) |

---

## New test: 64-layer hybrid prefill PCC

**File:** `models/tt_transformers/tests/test_qwen36_64layer_prefill_pcc.py`

A new end-to-end PCC test runs all 64 hybrid layers (GDN + full-attention) on
device against a float32 CPU reference, loading weights from the Qwen3.5-27B
snapshot (architecture-compatible; all shards available).

```bash
MESH_DEVICE=P150x4 QWEN36_64L_T=128 \
  python_env/bin/python -m pytest \
    models/tt_transformers/tests/test_qwen36_64layer_prefill_pcc.py -s -x

# Long-context variant
MESH_DEVICE=P150x4 QWEN36_64L_T=8192 \
  python_env/bin/python -m pytest \
    models/tt_transformers/tests/test_qwen36_64layer_prefill_pcc.py -s -x
```

| ISL | Final hidden-state PCC | GDN min-layer PCC |
|---|---|---|
| 128 | 0.9996 | 0.9989 |
| 8192 | **0.9999** | **0.9998** |

---

## Summary of PCC thresholds (all pass after fixes)

| Test | ISL | PCC | Threshold |
|---|---|---|---|
| Single GDN prefill | 128 | 0.9999 | 0.99 |
| Single GDN prefill | 512 | 0.9998 | 0.99 |
| Single GDN prefill | 1024 | 0.9998 | 0.99 |
| Single GDN prefill | 2048 | 0.9998 | 0.99 |
| Single GDN prefill | 4096 | 0.9999 | 0.99 |
| Single GDN prefill | 8192 | **0.9999** | 0.99 |
| 64L composite (ISL=128) | 128 | 0.9996 | 0.98 |
| 64L composite (ISL=8192) | 8192 | **0.9999** | 0.98 |
