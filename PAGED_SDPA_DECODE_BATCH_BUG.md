# Bug: `paged_scaled_dot_product_attention_decode` produces incorrect per-batch output for OLMo on TG

## Summary

`ttnn.transformer.paged_scaled_dot_product_attention_decode` produces incorrect
per-batch outputs when invoked with the shape configuration used by OLMo-3.1-32B
on a Galaxy (TG, 8×4) mesh: `Q=[1, B=8, n_q_heads=8, D=128]`,
`KV=[max_blocks, n_kv=1, block_size=64, D=128]` (GQA ratio 8:1), batch>1.

The kernel returns batch-correlated outputs that depend only on **col-local
batch index 0** for every batch position, instead of computing a distinct
output per batch position. Concretely:

- **Output mem cfg = HEIGHT_SHARDED on OLMo's `sub_core_grids` (50-core
  layout that skips col 4):** 4 of 8 batch slots come back **all zeros**;
  the populated slots show paired duplicates (slot 0 == slot 4).
- **Output mem cfg = `DRAM_MEMORY_CONFIG`:** 8 batch positions split into
  2 groups of 4. Within each group all 4 positions are bit-identical.
  The two groups produce different (but both incorrect) values even though
  the inputs are identical.

In OLMo's full decode pipeline this manifests as: only **1 user per
cluster_axis=1 column** decodes coherently. With `batch=32` (4 cols × 8
col-local users), only users at slots `K%8==0` (i.e. 0, 8, 16, 24) produce
coherent output — the other 28 users produce garbage **regardless of
whether their prompts are identical or distinct**.

## Reproducer

`tests/ttnn/nightly/unit_tests/operations/sdpa/test_paged_sdpa_decode_batch_collapse.py`

```bash
pytest tests/ttnn/nightly/unit_tests/operations/sdpa/test_paged_sdpa_decode_batch_collapse.py -xvs
```

The test sets up paged KV cache and Q with **identical data across all 8
batch positions**, runs `paged_scaled_dot_product_attention_decode`, and
asserts that all 8 output rows are bit-identical. The expectation is that
identical inputs should produce identical outputs; the kernel currently
fails this invariant.

**Reproduction status:**

| Test | Result | Evidence |
|---|---|---|
| Single device, DRAM output | ✅ PASS | identical Q/KV ⇒ identical output |
| Galaxy mesh, contiguous output cores | ✅ PASS | identical Q/KV ⇒ identical output |
| **Galaxy mesh, OLMo `sub_core_grids` output cores** | ❌ **FAIL** | rows 0-4 identical (sum=114.911, correct); **rows 5,6,7 corrupted** — row 6 has `max_diff=2.2e+37` (uninitialized memory) |

The repro confirms the bug is specifically triggered by HEIGHT_SHARDED
output sharding on the **non-contiguous** `sub_core_grids` layout that
starts at `CoreCoord(1, 0)` and skips col 4. With a contiguous output
layout starting at `CoreCoord(0, 0)`, the kernel works correctly.

## Verified-non-causes (from session diagnostics)

The following were instrumented and confirmed to be **correct** before the
SDPA call, so the bug is downstream of all of these:

| Signal | Verified |
|---|---|
| KV cache after prefill: identical for all 32 slots on every device | ✅ |
| Cache update during decode iter 0: col 0 slots 0..7 identical at pos 128 | ✅ |
| `current_pos` device-side: uniform `[130]*8` per col | ✅ |
| `q_heads_1BQD` per col-local user: identical sums (4.7684 / 14.6837 / 318.2634 on dev0/4/8) | ✅ |
| Decode iter 0 output token (',' id 11) for all 32 users | ✅ |
| Decode iter 1 output token: differs per col-local index, identical across cols | ❌ |

So the divergence enters specifically at the `paged_scaled_dot_product_attention_decode`
call when `batch>1` is used in OLMo's configuration.

## Why this didn't surface in earlier batch=32 testing

- Llama 3 70B Galaxy uses GQA ratio 1:1 per device (`n_q=8, n_kv=8` per
  device) and likely doesn't hit the same kernel path.
- OLMo's batched-prefill path was previously broken (single
  `paged_fill_cache(batch_idx_tensor=[0])` only wrote slot 0). With slots
  1..31 left at zero, decode-side garbage was attributed to "missing KV"
  rather than a decode-kernel bug. After the prefill fix
  (`a02d38e52b7`), all 32 slots are populated correctly, exposing the
  decode kernel issue.
- `batch=1` works perfectly (verified at 32k decode tokens) because the
  batch>1 code path isn't exercised.

## OLMo-specific shape & sharding

```
n_q_heads_global  = 64
n_q_heads_local   = 5  (padded to 8 for fused RoPE)
n_kv_heads_global = 8
n_kv_heads_local  = 1  (replicated across cluster_axis=1 cols, sharded
                        across cluster_axis=0 rows)
GQA ratio per device = 8:1
batch_size_per_device_group = 8 (col-local)
sub_core_grids = (1,0)-(3,9) ∪ (5,0)-(6,9)  // 50 cores, skips col 4
SCORES_BATCHED_MM_OUTPUT_MEMCFG: 8 cores from sub_core_grids row-wise
                                  starting at (1,0).
PAGED_SDPA_DECODE_PROGCFG: compute_with_storage_grid_size=(7, 6),
                           sub_core_grids=42 cores, q_chunk_size=0,
                           k_chunk_size=0
```

## Workarounds tried (none succeeded)

1. **Match `compute_with_storage_grid_size` to llama70b's `(8, 6)`:** no change.
2. **`memory_config=ttnn.DRAM_MEMORY_CONFIG` for SDPA + `to_memory_config`
   reshard after** (matches `tt_transformers/tt/attention.py:763`): hangs
   in trace replay; the rest of the OLMo decode pipeline expects a specific
   sharded layout from start.
3. **Contiguous core range for `SCORES_BATCHED_MM_OUTPUT_MEMCFG` (via
   `num_to_corerange(8)` = `((0,0),(7,0))`):** changes the per-user error
   pattern but breaks user 0 (now produces "Okay, the C. 3. 3. 3...").
   Likely conflicts with cores already claimed by dispatch / prefetcher.

## Probable fix locations (need kernel-level investigation)

- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/`
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/`

The kernel must:
- Iterate batch dim correctly when `n_q_heads_per_device != n_kv_heads_per_device`
  (GQA ratio > 1).
- Map batch positions to its compute grid such that each batch produces a
  distinct output even when Q is bit-identical across batches.
- Place per-batch output into the correct shard of the output memory
  config, including non-contiguous `sub_core_grids`-style layouts.

## Related

- Earlier OLMo session note acknowledging "varying batch size will result
  in slightly different outputs" in `models/demos/olmo_galaxy/tt/llama_attention.py:909-912`
  describes a **separate** numerical-precision issue (small PCC loss).
  The bug documented here is qualitatively different — outputs are zero
  or duplicated, not "slightly different".
- Prefill-side fix (commit `a02d38e52b7`) is required for the
  `paged_fill_cache` problem; this issue is purely on the decode side.
