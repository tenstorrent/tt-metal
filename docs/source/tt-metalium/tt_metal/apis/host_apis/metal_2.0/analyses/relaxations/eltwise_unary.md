# TensorParameter relaxations — `eltwise/unary`

**Author:** Anasuya and Claude

**Purpose:** State the Metal 2.0 `TensorSpecRelaxations` declaration this op requires, per `TensorParameter`, so that neither the auditor nor the porter has to derive it. The readiness sheet's `TensorParameter relaxation` cell points here.

**Covers one sheet row:** `eltwise/unary` · `UnaryDeviceOperation` · `ProgramFactory`.

---

## Contract — read this before using the document

This is **not** the same contract as the [offset-base-pointer](../2026-07-19_offset_base_pointers.md) and [3rd-argument](../2026-07-06_tensor_accessor_3rd_arg_triage.md) triage docs. Those are *priors* layered on a scan the auditor runs anyway, so a disagreement means "the doc is stale, trust your own scan."

Here there is no scan to fall back on: the audit recipe forbids re-deriving a relaxation, because deriving one is the expert work the sheet column exists to record. So this document is **authoritative but perishable**.

> **If any validity check fails, do not substitute your own judgement. Stop, and report the relaxation verdict as UNCONFIRMED.**

### Validity checks — confirm all six before applying anything below

Each is one grep. A commit stamp is deliberately not used: this op's sources move, and a stamp fires on every unrelated commit until its reader learns to ignore it.

1. **The cache key pins `tensor_layout`.** Read `compute_program_hash` in `unary_device_operation.cpp`: **no two tensors differing in dtype, page config (*including* `Tile`), memory config, or `Alignment` may share a cache entry.** Check that property, not any particular implementation of it — hashing the field, normalising it, or rejecting the divergent domain all satisfy it equally.

   **At the time of analysis this check passes.** The key hashes `input_tensor.tensor_spec().tensor_layout()` and `output_spec.tensor_layout()`. `TensorLayout::attribute_values()` is `(dtype, page_config, memory_config, alignment)`, so both `Tile` and `Alignment` are pinned per slot, which is exactly what `tensorspecs_match_with_relaxation` requires to be exactly equal. The output needs its own term because `compute_output_specs` can hand back a caller-supplied preallocated spec, and validation compares only its `Layout` enum against the input's.

   This was not always so. The key previously carried `input_tensor.layout()` — the `Layout` **enum**, not the page config — so `Tile` was unpinned and `Alignment` absent entirely, and two `bfloat16` TILE tensors with identical padded shape and memory config but tiles `32x32` and `16x32` were confirmed on silicon to share one cache entry. **If two such tensors can collide again, STOP** and report UNCONFIRMED: every relaxation requires exact `tensor_layout` equality and no flag reaches inside it, so a declaration against a key that does not pin it turns a working `ttnn.relu` into a hard `TT_FATAL` on the second dispatch.

   Note what pinning `tensor_layout` does **not** fix, and deliberately so: unary mis-sizes its buffers for a non-`32x32` tile regardless of caching, and pinning `Tile` gives each tile its own equally-wrong entry. That is a live bug, but it is **not this port's to fix and not a reason to hold the port** — see [§4](#4-not-covered), where the same bug is confirmed in an already-ported sibling.

2. **The key pins the sharded distribution geometry.** `compute_program_hash` carries a `distribution_key` term for both slots, hashing `shard_shape_in_pages()` and `cores()` off the distribution spec. This is load-bearing for rows 3–5 of §3: shard shapes squeeze jointly with tensor shape, so one shard spec resolves to different geometry at different shapes, and `GRID_2D` trims the bank list from the unsqueezed shape. Without this term, two shapes agreeing on the shard spec could share an entry and then fail the relaxation's geometry comparison. **Read the source it keys off** — the buffer's or the spec's — and carry the answer into §2, which depends on it.

3. **The dataflow kernels compile the accessor away when the slot is sharded.** `SRC_SHARDED` / `DST_SHARDED`, set from `has_sharding && is_sharded()`. In `reader_unary.cpp` the `TensorAccessorArgs<0, 0>()` and `TensorAccessor` declarations sit inside the `#else` of `#if SRC_SHARDED`, and `writer_unary.cpp` mirrors it. This is what makes the native-sharded rows safe — the sharded accessor payload is emitted by the host and never read by the device.

4. **`has_sharding` is itself pinned by the key.** `dst_shard_vol.has_value()` is `true` exactly when `get_shard_specs(...)` returned a value, and both shard-volume optionals are hashed. This is load-bearing: `SRC_SHARDED` / `RM_INTERLEAVED` are *compile-time defines* and the per-core runtime-arg slot count differs between the sharded (3) and interleaved (8) forms, so a code path that could flip on a cache hit would be unfixable by any override. It cannot flip.

5. **The op still has exactly one factory** (`std::variant<ProgramFactory>`, no `select_program_factory`), so the declaration is unconditional across factory choice.

6. **The TILE-path key still omits shape, and the override still re-applies the whole split.** `compute_program_hash` hashes `padded_shape` only on the `ROW_MAJOR` branch, and `override_runtime_arguments` re-enumerates the work split through the same `enumerate_core_rt_args` the miss path uses. If either changes, the regime table in §3 is reasoning about an op that no longer exists — in particular, a key that gains shape makes most of this document inert rather than wrong.

*Provenance, not a gate:* analysed against `origin/main` at the tree where `compute_program_hash` hashes both slots' `tensor_layout()` plus a `distribution_key`, and `enumerate_core_rt_args` is shared between `create_descriptor` and `override_runtime_arguments`.

---

## 1. For the auditor

**Relaxation verdict: `dynamic`.**

This clears the relaxation conjunct only — the op's other gate conjuncts are the sheet's to answer, as usual. Note that the sheet's `Known op issues` cell for this op is a second, independent block, and it is **not** cleared by this document.

The verdict is unconditional, but the declaration is not free-standing: §2 carries one code change the porter must make **in the same edit**. An auditor reading only this section should not conclude that the port is a pure addition.

### The rule behind validity checks 1 and 2

> **An op's `compute_program_hash` must pin at least everything `tensorspecs_match_with_relaxation` requires to be exactly equal** — the whole `tensor_layout` (dtype, page config *including* `Tile`, memory config, alignment) per slot, plus the sharded distribution geometry for sharded slots.
>
> Where the key is looser than the declaration, you get spurious throws. Where the key is looser than what the factory *bakes*, you get silent corruption. The declaration can only ever fix the second.

Unary satisfies both halves today, and did not before: the prerequisite change that added the `tensor_layout` and `distribution_key` terms exists precisely so this declaration can be made. What remains open is *which resolution* the geometry term keys off — see §2.

---

## 2. For the porter — what to write

The declaration is one line, applied to **both** `TensorParameter`s — input and output, unconditionally:

```cpp
.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
```

- **`dynamic_tensor_shape`** is mandatory, not optional: the TILE-path key omits `padded_shape` entirely, so one cache entry legitimately serves many shapes. Without it the *first* cache hit at a different shape throws.
- **`relax_logical_rank`** is required for the same reason — the TILE key omits rank along with the rest of the shape, so two tensors of different logical rank reach the same entry. Hashing `tensor_layout` does not change this: rank lives in `logical_shape`, on the other side of the split.
- **Do not** set `match_page_size`, and note the reasoning differs from [binary_ng](eltwise_binary_ng.md)'s even though the answer matches. binary_ng declines it because its row-major kernels override the accessor's page size per dispatch. Unary would be *entitled* to set it — it hashes `padded_shape` on the `ROW_MAJOR` branch, so it does independently pin the last-dim width, which is exactly the condition binary_ng's rule names — but it is declined here on precedent grounds: **no shipped factory in the tree sets `match_page_size`**, and unary should not be the op that introduces an untravelled flag while also being the first shipped op to declare a relaxation at all (see below). Revisit if a second op needs it.
- **Do not** set `match_padded_shape_only`. It is strictly weaker than `dynamic_tensor_shape` and pins nothing this op needs.

### The one code change that ships with the declaration

`compute_program_hash`'s `distribution_key` keys off the **Buffer's** `buffer_distribution_spec()`, falling back to the spec only when the output has no buffer yet. That is correct today, because `create_descriptor` bakes the buffer's resolution via `TensorAccessorArgs(*src_buffer)`. But `tensorspecs_match_with_relaxation` compares `relaxation_fields::shard_distribution_of`, which reads **`spec.compute_buffer_sharding_args()`** — the spec's resolution, not the buffer's. And `hash_tensorspec_with_relaxation` has no production callers; it is a helper for ops, not something the framework folds into the program key. So after the port the custom hash is still the only gate, and it has to be at least as strict as what validation compares.

**Swap the Buffer branch for `spec.compute_buffer_sharding_args()` on both sides, in the same edit as the declaration.** The code carries a `TODO(port)` at that line naming the swap.

**The swap is not cost-neutral.** An earlier revision of this section claimed it was, and that claim was wrong. During the prerequisite work, hashing *both* sources was measured at **+16%** on a 64-core `BLOCK_SHARDED` dispatch (139.7 µs → 162.5 µs, interleaved unchanged). But that measurement isolated the cost of the spec-side call, and the swap keeps exactly that call:

- `buffer->buffer_distribution_spec()` returns a stored value.
- `spec.compute_buffer_sharding_args()` recomputes the physical shape and page shape and builds a fresh `BufferDistributionSpec`, including the core enumeration, on every call.

Measured at the port, calling `compute_program_hash` directly on a 64-core `BLOCK_SHARDED` `[1,1,512,512]` bf16 TILE tensor (shard `64×64`, Wormhole n150 host, best of 5 × 200k calls, two runs each):

| | Buffer source (pre-swap) | spec source (post-swap) |
|---|---|---|
| `spec.compute_buffer_sharding_args()` alone | 27.6–28.0 µs | 26.7–27.2 µs |
| `buffer->buffer_distribution_spec()` alone | 0.002 µs | 0.002 µs |
| hash, fresh output (common path) | 31.6–32.3 µs | 58.6–60.6 µs |
| hash, preallocated output | 2.4 µs | 56.7–57.5 µs |
| hash, interleaved | 1.0 µs | 1.0 µs |

The Buffer-source key already paid one recompute on the common path, because a fresh output has no buffer yet and falls back to the spec. So the swap adds **one recompute (~27 µs) per sharded dispatch on the common path, and two (~55 µs) when the output is preallocated**. Against the prerequisite work's 139.7 µs baseline, the common-path addition is ≈ +19%. That is the whole of the +16% "both sources" figure (which added the same single recompute), not none of it. Interleaved dispatches are unaffected.

End-to-end Python dispatch timing on the port host was too noisy to resolve the difference: the interleaved control drifted 152 → 220 µs between runs. So the numbers above are hash-isolated.

This does not change whether the swap is required; it is. It changes what the swap costs, and that cost has to be either accepted explicitly or reduced, for example by making the spec's sharding resolution cheaper or cached. Either way it is outside the port.

The two resolutions agree for a freshly allocated tensor, since `tt_metal/impl/tensor/tensor_impl.cpp` builds the buffer with `.sharding_args = tensor_spec.compute_buffer_sharding_args()`. The only decoupler found is `view()` / `reshape`: for a TILE-layout sharded tensor `ttnn/core/tensor/tensor_ops.cpp` reuses the parent buffer's `device_local_config`, `sharding_args` included, under a freshly computed `TensorSpec`. No divergent pair could actually be constructed — `squeeze_shape_ranks` normalises aggressively, and every hand-worked candidate (`[1,1,128,64]` vs `[1,2,64,64]` vs `[2,1,64,64]` vs `[1,3,64,64]`, and the last-dim-changing reshape that does rewrite the shard spec) collapsed to the same `[8]` / `[4]` geometry. Make the swap anyway: unreachability by hand-search is not a guarantee, and after the declaration a divergence is a hard `TT_FATAL` rather than a missed cache split.

**Precedent, so the porter knows how much road is ahead.** At the time of analysis, **zero** non-experimental shipped factories declare any `TensorSpecRelaxations`; the only in-tree factory precedent is `experimental/quasar/transpose`, where five factories set `.relaxations = {.dynamic_tensor_shape = true}` unconditionally on both input and output, each with a comment recording that it mirrors a legacy `RuntimeTensorShape` accessor. Unary matches that shape and adds `relax_logical_rank`, which no shipped factory sets yet. Treat a validation throw during the port as a plausible framework-side gap, not automatically a mistake in the declaration.

---

## 3. Why — the derivation

Declarations are per slot, per cache entry, written by the factory at cache **miss** while it holds the actual tensors. The rows below are conditioned on the runtime tensor to show the reasoning is regime-complete.

| # | Runtime condition | `dynamic_tensor_shape` | `relax_logical_rank` | Confidence |
|---|---|---|---|---|
| 1 | interleaved, TILE | true | true | **High** |
| 2 | interleaved, ROW_MAJOR | true | true | **High** |
| 3 | native-L1 sharded, TILE | true | true | **High**, after the §2 swap |
| 4 | native-L1 sharded, ROW_MAJOR | true | true | **High**, after the §2 swap |
| 5 | sharded buffer, interleaved code path | true | true | Medium, after the §2 swap |

All five land on the same declaration; the regimes differ only in *why* it is safe.

**Row 1 — the case the relaxation exists for.** The key omits shape and rank, the override re-applies the split, and the only interleaved-TILE quantity outside per-core RTAs is `aligned_page_size`, which is `f(dtype, Tile)`. So with validity check 1 satisfied, the match collapses to bare `tensor_layout` equality and the shard term is `nullopt` on both sides. This is the regime that carries almost all of unary's traffic.

**Row 2 — nearly inert.** The `ROW_MAJOR` branch of the key hashes `padded_shape`, so shape barely varies within an entry to begin with; the relaxation is declared for uniformity rather than need. Because the key pins the width, `dynamic_tensor_shape`'s dynamic `aligned_page_size` common runtime argument is re-derived to the same value on every dispatch — which is why declining `match_page_size` costs nothing here beyond one runtime word.

**Rows 3 and 4 — safe on dead code, and now also on a pinned key.** Per validity check 3 the accessor does not exist in the compiled kernel on a sharded slot, so the distribution geometry the relaxation pins is not read by anything. The remaining exposure was never corruption but a spurious throw: `tensorspecs_match_with_relaxation` runs in `UpdateTensorArgs` regardless of what the kernel reads, so a squeeze-induced geometry mismatch would throw even though no kernel would have mis-addressed. Validity check 2's `distribution_key` closes that route — divergent geometry can no longer share an entry — provided the key and validation read the *same* resolution, which is what the §2 swap ensures. Absent the swap, treat these rows as Medium.

**Row 5 — the one to keep an eye on.** A tensor can be sharded while the op runs its interleaved path, because `get_shard_specs` returns `nullopt` on three separate fallbacks: `is_native_L1_sharding` failing (DRAM, mismatched grids, or an uneven input), an uneven *output*, and a `ROW_MAJOR` shard whose element count is not tile-aligned — that last one even emits a `log_warning`. In that state `has_sharding` is false, so `SRC_SHARDED` is `0` and **the accessor is live over a sharded buffer**. This is the row where the geometry term does real work rather than pinning dead code, and unlike binary_ng's equivalent it has documented, warning-logged routes into it, so "probably dead in practice" is not available as a defence. It is Medium rather than High only because that combination has no dedicated test: the mechanism is now sound on both sides of the match.

### Two things worth recording so they are not re-derived

- **`has_sharding` cannot flip on a cache hit**, so the compile-time `SRC_SHARDED` / `RM_INTERLEAVED` defines and the 3-vs-8 runtime-arg slot count are safe. This is validity check 4, and it is the same coincidence binary_ng relies on: a hashed optional whose `has_value()` tracks the code path exactly. It is a coincidence of the current key, not a design invariant — it breaks if anyone replaces the shard-volume optionals with plain integers.
- **The override's accessor-refresh loops are bounded by the cached buffer length** (`for (i = 0; i < common_args.size() && i < reader_common.size(); ++i)`), so a fresh tensor needing *more* accessor words would be silently truncated rather than caught. Under `dynamic_tensor_shape` the word count varies with rank for sharded slots (`tensor_accessor_args.cpp` computes `n_args` with a `rank * add_tensor_shape` term). Whether a mismatch is reachable was **not** established — the distribution spec squeezes rank before that point, which may well preclude it. Flagged because the `&&` guard would mask it either way.

---

## 4. Not covered

- **Non-`32x32` tile support, as distinct from `Tile` in the key. This is a family-wide bug, and it does not gate this port.** `create_descriptor` sizes its buffers with `tile_size(cb_data_format)`, which takes only a `DataFormat` and therefore assumes `32x32`, while `enumerate_core_rt_args` reads the real `tensor_spec().tile()` for the work split. The two disagree, and nothing in `eltwise/` guards the tile. Confirmed on silicon: `ttnn.relu` on a `16x32`-tile `bfloat16` tensor returns wrong data **in isolation**, with a clean `from_torch`/`to_torch` round-trip at the same tile — an op bug, not a caching artifact.

  The reason it does not gate: **`ttnn.typecast` fails identically** (same shape, same tile, isolation, `max_abs_err` ≈ 0.98), and `copy/typecast` is **already ported** — all four of its factories are on `create_program_artifacts`. It sizes every TILE-layout buffer with `tile_size(DataFormat)` and never reads the tensor's real tile, and it carries no tile guard. So a ported sibling already ships this exact defect, which settles that the defect is orthogonal to Metal 2.0 rather than something a port must clear. Fixing it inside the unary port would also violate the porting invariant, since it changes behaviour the sentinels are supposed to hold fixed.

  Worth noting how narrow real support is elsewhere: `data_movement/tilize` is the only op in this neighbourhood that engages with the question deliberately — it sizes buffers from the *real* tile (`operation_attributes.tile.get_tile_size(df)`) and `TT_FATAL`s that the tile **width** is 32 while permitting a smaller height, with a comment saying exactly that. `untilize` has unary's own split personality (real tile for work partitioning, `tile_size(DataFormat)` for buffers) and its codegen path rejects non-`32x32` outright. So "the ported ops support non-`32x32` tiles" is not the state of the tree; only tilize does, partially and on purpose. This belongs to the eltwise team as one issue spanning at least `unary` and `typecast`; the same gap is recorded against [binary_ng](eltwise_binary_ng.md).
- **`nd_shard_spec` configurations.** `MemoryConfig` can carry one, and the op's sharding helpers read only `shard_spec()`. Not analysed.
- **The independent Quasar clone** under `experimental/quasar/`, which was not consulted.
- The port's own mechanics. Nothing in this document is a porting instruction beyond §2.
