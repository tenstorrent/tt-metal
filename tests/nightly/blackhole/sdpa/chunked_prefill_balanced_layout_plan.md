# Chunked-prefill: balanced K/V layout — plan

Follow-up to [chunked_prefill_test_plan.md](chunked_prefill_test_plan.md). Replaces
v1's contiguous chunked-prefill codepath with balanced as the **only** chunked
layout — no flag, no opt-in.

## Goal

Make chunked-prefill ring SDPA use a balanced K/V cache layout: each device's local
K shard holds K rows for global positions corresponding to *its own Q slice* across
all chunks. Cache updates are local appends with zero cross-device traffic, and
every device has the same real-data extent at every chunk.

Non-chunked calls (`q_start_idx` is `nullopt`) keep the existing contiguous
sharding — unchanged.

## Why

v1 chunked-prefill uses contiguous-by-global-position sharding (the kernel's
existing convention). Sequential prefill → chunk 0 lands entirely in device 0's
slot → 3 of 4 devices hold all-padding K on early chunks. Fine for proving
kernel correctness; wrong for a production MLA layer:

- Cache update under contiguous requires cross-device shuffles per chunk to put
  each new K row in the right device's slot.
- Cache update under balanced is local: each device writes its own projected K
  rows in place. No NoC traffic.
- Decode (one token per step) is essentially impossible under contiguous.

Balanced is the production layout. Keeping two layouts (contiguous + balanced)
for chunked just doubles the kernel surface to maintain. Make it one — balanced.

## Layout shift in one diagram

```
Contiguous (current v1 chunked):           Balanced (new default for chunked):
device d's local row r maps to             device d's local row r maps to
  global K position d*KV_SHARD + r           global K position
                                             (r / slab_rows) * chunk_size
                                           + d * slab_rows
                                           + (r % slab_rows)
                                           where slab_rows = chunk_size / sp_size

After chunk 2 (logical_n=15K):
   dev0: [■■■■■■■■■■■■·]                   dev0: [■■■···········]
   dev1: [■············]                   dev1: [■■■···········]
   dev2: [··············]                  dev2: [■■■···········]
   dev3: [··············]                  dev3: [■■■···········]
```

## Kernel changes

All under `ttnn/cpp/ttnn/operations/transformer/sdpa/device/`.

1. **Position mapping function** in compute / reader / writer:
   ```cpp
   constexpr uint32_t kv_global_tile_for_local(uint32_t ring_id, uint32_t local_tile_idx) {
       if constexpr (chunked_prefill_enabled) {
           // Balanced: each shard has one slab per chunk, slab_tiles each.
           return (local_tile_idx / slab_tiles) * chunk_size_t
                + ring_id * slab_tiles
                + (local_tile_idx % slab_tiles);
       } else {
           // Non-chunked: unchanged contiguous addressing.
           return ring_id * kv_local_padded_Nt + local_tile_idx;
       }
   }
   ```
   New CT args: `chunk_size_t`, `slab_tiles = chunk_size_t / sp_size`. Gated on
   `chunked_prefill_enabled` so non-chunked binary is unchanged.

2. **Replace the 4 sites** that compute `ring_id * kv_local_padded_Nt + k_chunk * Sk_chunk_t`
   with calls to the new mapping. Today they live at:
   - `ring_joint_reader.cpp:351` (tile read addresses)
   - `ring_joint_writer.cpp` (skip predicate + intra-shard offsets)
   - `compute/ring_joint_sdpa.cpp:184` (`ring_iter_kv_start_tile`)
   - `compute_streaming.hpp:1795` (`step_k_start_tile` fed to diag stamp — must
     stay the true absolute global K position)

3. **Skip predicate becomes per-tile, not per-iter.** Under balanced, every iter's
   shard contributes one slab per chunk → no whole iters skip. Check
   `kv_global_tile_for_local(...) >= logical_nt` per tile inside the K loop.

4. **Revert `is_first_active_iter` for chunked.** Under balanced every iter is
   active → `ring_iter == 0` is always-active-first again. Drop `active_iter_idx`
   tracking from compute and writer in the chunked path. (Keep the boundary fix
   in `ring_iter_does_work` / `find_last_active_ring_iter` — that's correct
   regardless of layout.)

5. **`try_skip_causal_above_diag`** assumes monotonic K position inside the K
   loop. True within a slab, false across slabs. Restructure to slab-outer /
   within-slab-inner, OR disable when chunked.

6. **Mask CB layout** simplifies — each slab is `chunk_size / sp_size` rows
   (always tile-aligned by construction), no partial-tile masks at slab
   boundaries. Recheck `global_n_partial_col` math.

7. Watch the `mla_100k-q160-k256` binary-size canary. Two new CT args
   (`chunk_size_t`, `slab_tiles`) need slots — reuse where possible.

## Host changes

- `ring_joint_sdpa_device_operation_types.hpp`: add `chunk_size` to params (needed
  to derive `chunk_size_t` and `slab_tiles` for CT args). Add a TT_FATAL that
  `chunk_size % (sp_size * TILE_HEIGHT) == 0`. Exclude from `attributes()` —
  same program-cache hash across chunks of the same size.
- `ring_joint_sdpa_program_factory.cpp`: push `chunk_size_t`, `slab_tiles` to all
  3 kernels in the `chunked_prefill_enabled` branch.
- `sdpa.hpp` / `sdpa.cpp` / `sdpa_nanobind.cpp`: add `chunk_size` arg next to
  `q_start_idx`. Required when `q_start_idx` is set; `nullopt` otherwise.

No new public flag — `q_start_idx.has_value()` IS the balanced switch.

## Test changes

`tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`:

- **Update** `run_ring_joint_sdpa_chunked` (the existing v1 driver) to use the new
  balanced upload: per chunk, each device writes only its own
  `slab_rows = chunk_size / sp` K rows into its local cache slot at offset
  `chunk_idx * slab_rows`. No full-cache zero-pad host tensor. Pass `chunk_size`
  to the SDPA call alongside `q_start_idx`.
- Reference oracle unchanged (`torch_joint_sdpa_reference` on the full sequence —
  layout-independent).
- **Single test config**: keep `mla_100k` only at the 20K dev shape (4 chunks of
  5K). Don't sweep chunk sizes, total_seq, or models. Re-add mla_128k or scale
  to 55K only when this is green.

## How to verify

```bash
source python_env/bin/activate

# 1. Non-chunked regression — confirms the non-chunked codepath is untouched.
scripts/run_safe_pytest.sh \
    tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy \
    -k mla_100k

# 2. Chunked-prefill — now uses balanced layout under the hood.
scripts/run_safe_pytest.sh \
    tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_chunked_accuracy
```

Acceptance:

1. (1) passes at PCC ≥ 0.999 on mla_100k all configs — proves no regression on
   the non-chunked path.
2. (2) passes at per-chunk PCC ≥ 0.99 on all 4 chunks of the 20K dev shape.
3. `num_program_cache_entries == 1` across all chunks (already asserted in the
   existing test).
4. Chunk 0 wall-clock under balanced should be roughly **4× faster** than v1
   contiguous (under contiguous only 1 of 4 devices did real work on chunk 0;
   under balanced all 4 do). Compare against a v1 baseline if needed.

## Risks

- **Joint K/V tail** (`L > 0`): layout under balanced needs its own convention.
  v1 chunked test uses `joint_seq_len = 0`, so the existing test won't catch
  joint-tail bugs. Re-check the non-chunked joint-tail configs in (1).
- **NHK != 1** (WAN-style): balanced + K-broadcast interaction. Keep gated to
  `NHK == 1` (TT_FATAL); WAN variant later.
- **AllGather output**: standard ring AllGather concatenates device-index order
  → gathered K under balanced is row-permuted relative to global K. The kernel
  handles via the mapping; verify `persistent_output_buffer_k` writer doesn't
  assume contiguous on its end.

## Out of scope

- Decode (separate work).
- WAN-style `NHK == NHQ` chunked variant.
- Full MLA layer integration —
  `models/demos/deepseek_v3_d_p/tests/torch/test_kimi_k26_mla_chunked_prefill.py`
  re-targeted onto the device op, later.
