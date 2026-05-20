# Chunked-prefill: balanced K/V layout — findings

Follow-up to [chunked_prefill_balanced_layout_plan.md](chunked_prefill_balanced_layout_plan.md).
Status: **GREEN** on the dev shape (mla_100k, 20K total, 4 chunks of 5K, sp=4 BH).

## What landed

Replaced v1's contiguous chunked-prefill layout with balanced as the only chunked
layout — no flag, no opt-in. `q_start_idx.has_value()` is the balanced switch.

### Per-chunk PCC (mla_100k, sp=4 BH, q_chunk=k_chunk=160, 20K dev shape)

| Chunk | logical_n | PCC    | RMSE   |
|------:|----------:|-------:|-------:|
|     0 |      5120 | 0.9997 | 0.0047 |
|     1 |     10240 | 0.9996 | 0.0069 |
|     2 |     15360 | 0.9994 | 0.0087 |
|     3 |     20480 | 0.9994 | 0.0099 |

`num_program_cache_entries == 1` across all 4 chunks. Non-chunked
`test_ring_joint_attention_sdpa_accuracy` regression: PCC ≈ 0.9996 across all 3
mla_100k k_chunk_sizes (unchanged from prior baseline).

PCC / RMSE numerics match the v1 contiguous-layout numbers from the original
test plan — the layout switch preserves correctness.

## Why balanced

Under contiguous (v1), chunk 0 of a sequential prefill lands entirely in dev 0's
slot — 3 of 4 devices hold all-padding K on early chunks. Production MLA cache
updates would then require a cross-device shuffle per chunk to place each new K
row in the right device's slot. Under balanced, each device holds **one slab per
chunk** of size `slab_rows = chunk_size / sp_size`, so cache updates are local
appends with zero cross-device traffic, and every device has the same real-data
extent at every chunk. Decode (one token per step) also becomes feasible.

## Layout

Local cache row `r` on device `d` maps to global K position:

```
global_K_pos = (r / slab_tiles) * chunk_size + d * slab_rows + (r % slab_rows)
             where slab_tiles = chunk_size_t / sp_size  (in tiles)
```

DRAM read addresses for gathered K still use the contiguous formula
`ring_id * kv_local_padded_Nt + k_chunk * Sk_chunk_t` — the AllGather still
concatenates device slices in ring order. The mapping function is only used for
the *global attention K position* (for the diag-stamp mask and the `logical_n`
skip).

## Changes by surface

**Host plumbing**
- `ring_joint_sdpa_device_operation_types.hpp`: added `chunk_size` to params,
  excluded from `attributes()` so all chunks share one program-cache entry.
- `ring_joint_sdpa_device_operation.cpp`: `TT_FATAL`s requiring `chunk_size`
  when `q_start_idx` is set, and `chunk_size % (sp_size * TILE_HEIGHT) == 0`
  to keep each slab tile-aligned.
- `sdpa.hpp` / `sdpa.cpp` / `sdpa_nanobind.cpp`: plumbed `chunk_size` through to
  the Python API (`nb::arg("chunk_size") = nb::none()` — required when
  `q_start_idx` set, ignored otherwise).
- `ring_joint_sdpa_program_factory.cpp`: derived `chunk_size_t` and
  `slab_tiles = chunk_size_t / sp_size`, pushed as CT args at slots 26/27
  (reader), 30/31 (writer), 41/42 (compute). Zero on the non-chunked path.
  Added `TT_FATAL slab_tiles % Sk_chunk_t == 0` so every K chunk lives in
  exactly one slab.

**Kernel changes**
- `dataflow/ring_utils.hpp`: added `kv_global_tile_for_local(...)` helper that
  threads the slab arithmetic; updated `find_last_active_ring_iter` and
  `count_valid_kv_chunks` to be chunked-aware.
- Reader / writer / compute, chunked path:
  - `ring_iter_processes_KV_chunks = true` (every iter is active under
    balanced — every device contributes one slab per chunk of real K).
  - `is_first_active_iter = (ring_iter == 0)` (no `active_iter_idx` tracking
    needed — plan §4 "Revert is_first_active_iter for chunked").
  - Per-k_chunk skip uses `kv_global_tile_for_local(...) >= logical_nt`
    instead of the contiguous formula. The `slab_tiles % Sk_chunk_t == 0`
    TT_FATAL guarantees the first-tile-of-chunk check is sufficient (no
    straddling).
  - `global_n_partial_col` partial-tile mask machinery disabled — slab
    boundaries are tile- and k_chunk-aligned by construction, so the
    per-k_chunk skip is exact.
- `compute_streaming.hpp`:
  - `try_skip_oob_kv` uses the helper.
  - `step_k_start_tile` fed to the diagonal stamp uses the global K position
    via the helper — under balanced this gives the correct relative
    (Q − K) tile offset for every (own-slab / cross-device-slab) case:
    - K in slab from dev `d' < d` (own slab index): all valid (mask off
      the left).
    - K in slab from dev `d' == d`: standard causal diag.
    - K in slab from dev `d' > d`: all -inf (mask off the right, clamped
      to TILE_WIDTH).
- Reader's `end_seq_tile` clamp: under chunked drops the `logical_nt` ceiling
  (the global-K-position skip already handles "past logical_n"; the host
  pre-zeros padding slabs, so reading the full device slice is correct).

**Test driver**
- `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`: builds a permuted
  full-K tensor where contiguous dim-2 sharding produces the balanced
  per-device layout. For chunk `i`, only slabs `[0..i]` are populated; the
  rest is zero. Passes `chunk_size` to the SDPA call alongside `q_start_idx`.

## Bug found during bring-up

Initial run failed with PCC ≈ 0.81 across all chunks. Root cause: the reader's
`end_seq_tile` clamp `std::min(logical_nt, kv_local_padded_Nt * (ring_id + 1))`
compared gathered-K coords against `logical_nt`, which is a *global* K coord.
Under contiguous they're the same; under balanced they're not. For chunk 0
(logical_nt = 160) and ring_id ≥ 1, every read into gathered K's slice
[160, 320), [320, 480), [480, 640) hit `d2 ≥ end_seq_tile` and zero-filled
instead of reading real data. Fix: under chunked, drop the `logical_nt` ceiling
from `end_seq_tile` and rely on the per-k_chunk global-K skip + host-side
zeroing of padded slabs.

## Out of scope (deferred)

- Joint K/V tail (`L > 0`): the chunked test runs with `joint_seq_len = 0`, so
  the balanced layout for the joint tail is not exercised. Re-check non-chunked
  joint-tail configs as a guard against host-side regressions (which the
  passing non-chunked regression test already covers for our shapes).
- `NHK != 1` (WAN-style): balanced + K-broadcast interaction not exercised —
  test gates to `NHK == 1` via the existing `mla_100k` config.
- Decode (separate work).
- Scale `total_seq` back up to 55K (currently 20K dev shape).
- In-place device K/V writes instead of full-cache re-upload per chunk (test
  driver still re-uploads the full balanced cache per call; correct but slow).
- Full MLA layer integration (`test_kimi_k26_mla_chunked_prefill.py`
  re-targeting onto the device op).

## How to verify

```bash
source python_env/bin/activate

# Non-chunked regression — confirms the non-chunked codepath is untouched.
scripts/run_safe_pytest.sh \
    tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy \
    -k mla_100k

# Chunked-prefill — balanced layout.
scripts/run_safe_pytest.sh \
    tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_chunked_accuracy
```
