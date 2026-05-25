# KV Pad-Aware Rotation Multi-Slab Design

## Goal

Extend `ttnn.transformer.ring_joint_scaled_dot_product_attention` KV-pad-aware rotation so it supports a growing chunked-prefill history.

The current rotated path only accepts:

```cpp
N_local_kv == 2 * N_local_q
```

That means one per-device `OLD` slab plus one per-device `NEW` slab. It covers the first small-ISL transition, but it does not cover later chunked-prefill calls where K/V contain many previous slabs.

This design supports:

- Arbitrary prior cache length up to the physical K/V cache capacity.
- A current fixed-size Q chunk, with `new_actual_isl <= chunk_size_global`.
- K/V tensors whose per-device sequence dimension contains any number of slabs.
- The same public API: `logical_n` is total valid KV after the current write, and `kv_actual_isl` is valid KV before the current write.

The first implementation should still reject nonzero joint tokens and sub-tile valid lengths, matching the current rotated path.

## Current Limitation

The current implementation derives per-device Q rotation metadata in `ring_joint_sdpa_program_factory.cpp` using only one global chunk as the old capacity:

```cpp
old_capacity_nt = ring_size * q_local_padded_Nt;
```

It then validates:

```cpp
N_local_kv == 2 * N_local_q
kv_actual_isl <= old_capacity
logical_n >= old_capacity
new_actual_isl <= old_capacity
```

Those conditions assume the prior valid KV can only live in the first global chunk. For growing history, the prior valid KV may be in chunk group 3, 10, or any later group. The physical mapping is still regular, but the Q metadata must be based on the tail group containing `kv_actual_isl`, not group 0.

## Terminology

All core mapping should operate in tile units.

```text
TILE               = 32
R                  = ring_size
Q                  = q_local_padded_Nt
C                  = R * Q
K                  = kv_local_padded_Nt
S                  = K / Q
kv0                = kv_actual_isl / TILE
kv1                = logical_n / TILE
new_nt             = kv1 - kv0
cache_capacity_nt  = S * C
```

Definitions:

- `Q`: one per-device slab size in tiles.
- `C`: one global chunk group size in tiles.
- `S`: number of per-device K/V slabs present in the input/cache tensor.
- `kv0`: valid global KV length before this call.
- `kv1`: valid global KV length after this call.
- `new_nt`: number of current valid Q/K/V tiles.

For physical local K/V tile `local_k_tile` on device/ring id `chip`:

```cpp
slab = local_k_tile / Q;
cell = local_k_tile % Q;
global_k_tile = slab * C + chip * Q + cell;
```

This is the multi-slab form of the existing chunked K mapping. The current helper already has this shape for chunked mode, but validation prevents using it beyond two slabs in rotated mode.

## Layout Invariant

The cache is slab-major by global chunk group:

```text
global group 0:
  chip 0 slab 0 -> global [0, Q)
  chip 1 slab 0 -> global [Q, 2Q)
  ...
  chip R-1 slab 0 -> global [(R-1)Q, C)

global group 1:
  chip 0 slab 1 -> global [C, C+Q)
  chip 1 slab 1 -> global [C+Q, C+2Q)
  ...
```

Valid K/V tiles are the dense global prefix:

```text
[0, kv1)
```

The current Q rows are the new global interval:

```text
[kv0, kv1)
```

For each chip, Q rows are packed from row 0 in the same natural order as the new tokens whose destination chip is that chip. Padding rows are a suffix of that chip's Q slab.

Because the first implementation keeps `new_nt <= C`, each chip receives at most `Q` valid Q rows, and each chip's new interval intersects at most two global groups.

## Public API

No new public parameter is required for the first multi-slab implementation.

Existing inputs are sufficient:

- `kv_actual_isl`: prior valid global KV length.
- `logical_n`: total valid global KV length after this call.
- `input_tensor_q.shape[2]`: per-device fixed current chunk size.
- `input_tensor_k.shape[2]` or cache shape: number of physical K/V slabs available per device.

The activation rule remains:

```text
kv_actual_isl.has_value() -> KV-pad-aware rotation mode
```

The docstring should be updated to say that K/V may contain more than one historical slab; it is not limited to one `OLD` slab plus one `NEW` slab.

## Validation Changes

Keep:

```cpp
is_chunked
is_causal
L == 0
logical_n >= kv_actual_isl
logical_n > kv_actual_isl
kv_actual_isl % TILE == 0
(logical_n - kv_actual_isl) % TILE == 0
new_actual_isl <= q_local_padded_N * ring_size
```

Add:

```cpp
kv_local_padded_Nt % q_local_padded_Nt == 0
logical_nt <= (kv_local_padded_Nt / q_local_padded_Nt) * ring_size * q_local_padded_Nt
```

Remove or replace:

```cpp
N_local_kv == 2 * N_local_q
kv_actual_isl <= old_capacity
```

Rename the remaining capacity concept:

```cpp
chunk_capacity = N_local_q * ring_size
```

`new_actual_isl <= chunk_capacity` should remain for the first implementation because Q has only one fixed global chunk of capacity.

`logical_n > kv_actual_isl` rejects no-op calls where the current Q slab contains no valid tokens. If that case becomes useful later, handle it explicitly before computing `last_group = (kv1 - 1) / C`.

`logical_n < chunk_capacity` is allowed once reader, compute, and writer share the same no-K ring-iteration skip predicate.

## Host-Derived Q Metadata

Replace the current group-0-only Q metadata with group-aware metadata.

Current fields:

```cpp
kv_pad_q_old_start_nt
kv_pad_q_old_count_nt
kv_pad_q_new_start_nt
kv_pad_q_valid_nt
```

They can be kept for a minimal patch, but their semantics should become generic two-segment metadata:

```text
q_seg0_start_nt
q_seg0_count_nt
q_seg1_start_nt
q_valid_nt
```

For chip `chip`, compute intersections of `[kv0, kv1)` with that chip's block in the first and second possible global groups.

```cpp
struct Segment {
    uint32_t start = 0;
    uint32_t count = 0;
};

Segment intersect_chip_group(uint32_t group, uint32_t chip) {
    uint32_t block_start = group * C + chip * Q;
    uint32_t block_end = block_start + Q;
    uint32_t lo = std::max(kv0, block_start);
    uint32_t hi = std::min(kv1, block_end);
    if (hi <= lo) {
        return {};
    }
    return {.start = lo, .count = hi - lo};
}

uint32_t first_group = kv0 / C;
uint32_t last_group = (kv1 - 1) / C;
TT_FATAL(last_group <= first_group + 1, "new_actual_isl must fit in one fixed global chunk");

Segment seg0 = intersect_chip_group(first_group, chip);
Segment seg1 = {};
if (last_group != first_group) {
    seg1 = intersect_chip_group(first_group + 1, chip);
}

q_seg0_start_nt = seg0.start;
q_seg0_count_nt = seg0.count;
q_seg1_start_nt = seg1.start;
q_valid_nt = seg0.count + seg1.count;
TT_FATAL(q_valid_nt <= Q, "A chip received more valid Q tiles than one local Q slab can hold");
```

The kernel Q-row mapping becomes:

```cpp
uint32_t q_global_tile_or_invalid(uint32_t q_row) {
    if (q_row < q_seg0_count_nt) {
        return q_seg0_start_nt + q_row;
    }
    if (q_row < q_valid_nt) {
        return q_seg1_start_nt + (q_row - q_seg0_count_nt);
    }
    return INVALID_TILE;
}
```

This handles:

- Cold start.
- Exact chunk-boundary prior length.
- Prior length in any later group.
- Current interval that fills the prior tail group and wraps into the next group.

## K Mapping And Skips

Add explicit rotated helpers in `chunked_prefill_utils.hpp` even if they share the existing formula. The goal is to make reader, compute, and writer use the same contract.

```cpp
template <uint32_t Q, uint32_t C>
uint32_t kv_pad_k_global_tile(uint32_t chip, uint32_t local_k_tile) {
    uint32_t slab = local_k_tile / Q;
    uint32_t cell = local_k_tile % Q;
    return slab * C + chip * Q + cell;
}

template <uint32_t Q, uint32_t C, uint32_t Sk_chunk_t>
bool kv_pad_k_chunk_has_valid_cols(uint32_t chip, uint32_t local_k_start, uint32_t logical_nt) {
    for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
        if (kv_pad_k_global_tile<Q, C>(chip, local_k_start + col) < logical_nt) {
            return true;
        }
    }
    return false;
}
```

Use `kv_pad_k_chunk_has_valid_cols` in:

- `ring_joint_reader.cpp`
- host ring-work planning in `ring_joint_sdpa_program_factory.cpp`
- `compute_streaming.hpp` K pre-scan
- `compute_streaming.hpp` K loop
- `compute_common.hpp` if non-streaming support is restored

Reader and compute must skip the exact same K chunks. A mismatch can deadlock circular buffers.

## Masking

The rotated mask should be position based:

```text
mask = -inf if Q is padding
mask = -inf if K is padding
mask = -inf if K_global_tile > Q_global_tile
diag tile if K_global_tile == Q_global_tile
0 otherwise
```

For multi-slab support, prefer passing the local K start tile and chip/ring id into the mask helper, then deriving `K_global_tile` per column:

```cpp
k_global = kv_pad_k_global_tile(chip, local_k_start + col);
k_valid = k_global < logical_nt;
```

This removes the current single-straddle assumption:

```cpp
straddle_col
straddle_jump
```

If implementation time is tight, an initial patch can keep the existing straddle mechanism and validate:

```cpp
Sk_chunk_t <= q_local_padded_Nt
```

That guarantees a K chunk crosses at most one slab boundary. The more robust endpoint is per-column K mapping for the rotated path.

## Program Factory Wiring

Compute these values in `ring_joint_sdpa_program_factory.cpp`:

```cpp
kv_pad_rotation_enabled
kv_actual_nt
new_actual_nt
q_seg0_start_nt
q_seg0_count_nt
q_seg1_start_nt
q_valid_nt
```

Pass the segment metadata to compute and writer. If reader only needs K validity, pass:

```cpp
kv_pad_rotation_enabled
logical_nt
q_local_padded_Nt
chunk_size_t
```

`kv_actual_nt` does not need to be a reader argument unless reader starts making Q-row decisions.

Program cache safety:

- If segment metadata remains compile-time args, the existing hash must include `kv_actual_isl` and `logical_n`.
- If moved to runtime args later, descriptor cache-hit patching must update these scalar runtime args, not just buffer addresses.

## Writer Behavior

Valid Q rows are a contiguous prefix on each chip:

```text
[0, q_valid_nt)
```

For rotated mode, writer should use `q_valid_nt` to avoid restoring, saving, or writing padding rows when practical.

Acceptable first behavior:

- Output remains in local rotated Q order.
- Padding output rows are unspecified.
- Tests gather only valid rows and unrotate them to natural order.

Preferred behavior:

- Writer clamps output and stats writes to `q_valid_nt`.
- Compute avoids all-`-inf` softmax rows for padding Q rows by skipping those rows or forcing zero output.

The preferred behavior reduces NaN risk if a caller accidentally consumes padding rows.

## Indexed Cache Mode

For `cache_batch_idx`, K/V may be full-cache tensors. Multi-slab rotation should work the same way as pre-sliced K/V:

```text
physical K local seq = cache maximum per device
logical_n = valid global length after this call
kv_actual_isl = valid global length before this call
```

K chunks beyond `logical_n` are skipped or masked. The physical cache capacity is derived from the K/V shape and must satisfy:

```text
logical_nt <= cache_capacity_nt
```

Do not derive Q start from full cache shape. In rotated mode, Q positions come only from `[kv0, kv1)`.

## Test Plan

Add a pure torch spec that replaces `_kv_pad_rotation_layout` with a group-aware version:

```python
def _kv_pad_rotation_layout_multi_slab(kv_actual_isl, new_actual_isl, sp_factor, chunk_size_local, max_slabs):
    # For each natural new token global position g in [kv0, kv1):
    #   group = g // chunk_size_global
    #   within = g % chunk_size_global
    #   chip = within // chunk_size_local
    #   cell = within % chunk_size_local
    #   local_k_row = group * chunk_size_local + cell
    #   q row is the next packed row for that chip
```

Device cases:

- `history_partial_tail`: `kv_actual = 3*C + 96`, `new = 64`.
- `history_exact_boundary`: `kv_actual = 4*C`, `new = 64`.
- `history_wrap_to_next_group`: `kv_actual = 2*C + C - 32`, `new = 128`.
- `history_full_current_chunk`: `kv_actual = 5*C + 32`, `new = C`.
- `cold_start`: `kv_actual = 0`, `new = 128`.
- `indexed_full_cache`: cache has more slabs than `ceil(logical_nt / C)`.

Negative cases:

- `new_actual_isl > C * TILE`.
- `logical_n < kv_actual_isl`.
- `logical_n` exceeds physical cache capacity.
- `kv_local_padded_Nt % q_local_padded_Nt != 0`.
- `kv_actual_isl` or `new_actual_isl` not tile aligned.
- nonzero `L` while joint support remains out of scope.

Regression tests to keep:

```bash
pytest models/demos/deepseek_v3_d_p/tests/test_ring_joint_sdpa_handoff.py -k "kv_pad_aware_rotation"
pytest models/demos/deepseek_v3_d_p/tests/test_ring_joint_sdpa_handoff.py -k "index_based_kv_access or chunked_q_start_idx or persistent_buffer"
pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py -k ring_joint
```

## Implementation Order

1. Replace rotated-mode validation with multi-slab validation.
2. Add group-aware Q segment derivation in the program factory.
3. Rename or reinterpret `kv_pad_q_old_*` compile-time args as `q_seg*` metadata.
4. Add explicit rotated K mapping and K-chunk validity helpers in `chunked_prefill_utils.hpp`.
5. Switch reader, writer valid-K counting, and compute K pre-scan to the shared K validity helper.
6. Update the rotated mask to derive K global tiles per column, or temporarily validate `Sk_chunk_t <= q_local_padded_Nt`.
7. Clamp writer restore/save/write behavior with `q_valid_nt`, or document padding rows as ignored for the first patch.
8. Add torch and device tests for history lengths greater than one global chunk.
9. Wire the chunked MLA caller to pass `kv_actual_isl` and `logical_n` from the real cache state.

## Cold-Start / No-K Ring Iterations

True cold-start calls have a total populated cache after the current write that is smaller than one full global chunk. In those calls, at least one ring slot has no valid K/V work.

Example with `ring_size = 4` and `N_local_q = 64`:

```text
chunk_capacity = 256
logical_n      = 96

ring_id 0 owns global K [0, 64)    -> valid K/V exists
ring_id 1 owns global K [64, 128)  -> partial K/V exists
ring_id 2 owns global K [128, 192) -> no K/V
ring_id 3 owns global K [192, 256) -> no K/V
```

The per-K-chunk skip is not enough for this case. Reader, compute, and writer also need to agree when a whole ring iteration has zero valid K/V chunks. Otherwise one kernel may skip all K/V pushes while another waits for K/V, staging, or completion signals for that ring iteration.

### Host-Computed Active-Iteration Plan

The active ring-iteration plan is pure launch geometry: it depends on ring order, `logical_nt`, local K/V shape, chunk size, and the current device index. It should be computed once in the program factory and passed to kernels as compile-time metadata:

```text
active_ring_iter_mask
last_active_ring_iter
single_valid_kv_chunk_mask
```

For nonzero joint tokens, the active predicate becomes:

```text
ring_iter_is_active = ring_iter_has_kv_work || joint_contributes
```

KV-pad-aware rotation currently rejects joint tokens, but the shared host plan keeps the generic joint predicate so the reader, compute kernel, and writer use the same ring-iteration schedule.

The writer also receives `single_valid_kv_chunk_mask` from host because that staging decision is pure launch geometry. Compute still needs per-column global K coordinates for causal and padding masks.

Per-K-chunk logical-N skips remain in reader and compute. They are cheap loop-local checks, and moving them to host would require passing a wider per-ring, per-K-chunk mask rather than the compact ring-iteration masks above.

### Reader Contract

For a no-K ring iteration:

- Do not push K/V chunks.
- Do not push Q solely for that skipped ring iteration.
- Do not push per-K completion signals.
- Advance to the next active ring iteration using the same predicate as compute and writer.

Reader and compute must still agree on per-K-chunk skips inside partially valid ring iterations. That continues to use `kv_chunk_has_valid_cols`.

### Compute Contract

For a no-K ring iteration:

- Do not wait on K/V circular buffers.
- Do not update softmax accumulators.
- Do not emit partial output, staging, or completion signals for the skipped iteration.

The first active ring iteration initializes the accumulators. Later active ring iterations rescale and accumulate normally. Skipped ring iterations leave the accumulator state unchanged.

This requires replacing any `chunked_enabled ? true : ...` active-iteration shortcut with the host-provided active mask. In particular, "chunked mode" does not imply "this ring iteration has valid K/V."

### Writer Contract

Writer must use the same host-provided active iteration for:

- `seen_active_iter`
- `is_first_active_iter`
- `last_active_ring_iter`
- deferred-normalization restore/save decisions
- output finalization
- prefetches that can block on compute-produced CBs

For no-K ring iterations, writer should not wait for compute signals and should not restore/save staging state. The final output should be written on the last active ring iteration, not the last ring iteration in the ring sequence.

### Masking

No new mask semantics are needed for valid active iterations. The rotated mask already treats K positions `>= logical_nt` as padding and Q rows without a global position as padding.

The cold-start change is about skipping whole ring iterations whose K chunks are all padding. Partially valid iterations still use the existing per-column K global-position mask.

### Validation Change

Keep:

```cpp
logical_n > 0
logical_n > kv_actual_isl
new_actual_isl <= chunk_capacity
logical_n <= cache_capacity
```

The operation should then support:

```text
0 < logical_n < chunk_capacity
```

as long as lengths are tile-aligned and the current valid Q fits in one global chunk.

### Cold-Start Tests

Torch-only tests:

- `kv_actual_isl = 0`, `new_actual_isl = 64`: only ring slot 0 has valid K/V.
- `kv_actual_isl = 32`, `new_actual_isl = 64`: ring slot 0 has valid K/V; the next slot is partial.
- `kv_actual_isl = 96`, `new_actual_isl = 64`: multiple ring slots are active, but at least one later slot is no-K.

These tests should compare rotated layout attention against natural-order causal attention and assert which ring slots have valid K/V work.

TTNN positive tests:

- Build rotated-mode calls with `logical_n < N_local_q * ring_size`.
- Compare gathered valid output rows against natural-order causal torch reference.
  The cold-start cases should include at least one ring slot with zero valid K/V work.

## Open Decisions

- Whether to rename compile-time args now or keep old names for a smaller patch.
- Whether to support `new_actual_isl > chunk_size_global` in one op call. This would require Q to represent more than one global chunk and needs more than two Q segments per chip.
- Whether to implement per-column K mapping immediately or stage it behind `Sk_chunk_t <= q_local_padded_Nt`.
- Whether padding Q rows should be guaranteed zero, or remain unspecified with tests and callers selecting only valid rows.
- Whether nonzero joint tokens should remain rejected until after multi-slab rotation is stable.
