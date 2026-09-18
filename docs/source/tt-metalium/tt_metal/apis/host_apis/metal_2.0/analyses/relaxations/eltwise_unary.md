# TensorParameter relaxations — `eltwise/unary`

**Author:** Anasuya and Claude

**Purpose:** State the Metal 2.0 `TensorSpecRelaxations` declaration this op requires, per `TensorParameter`, so that neither the auditor nor the porter has to derive it. The readiness sheet's `TensorParameter relaxation` cell points here.

**Covers one sheet row:** `eltwise/unary` · `UnaryDeviceOperation` · `ProgramFactory`.

---

## Contract — read this before using the document

This is **not** the same contract as the [offset-base-pointer](../2026-07-19_offset_base_pointers.md) and [3rd-argument](../2026-07-06_tensor_accessor_3rd_arg_triage.md) triage docs. Those are *priors* layered on a scan the auditor runs anyway, so a disagreement means "the doc is stale, trust your own scan."

Here there is no scan to fall back on: the audit recipe forbids re-deriving a relaxation, because deriving one is the expert work the sheet column exists to record. So this document is **authoritative but perishable**.

> **If any validity check fails, do not substitute your own judgement. Stop, and report the relaxation verdict as UNCONFIRMED.**

### Validity checks — confirm all five before applying anything below

Each is one grep. A commit stamp is deliberately not used: this op's sources move, and a stamp fires on every unrelated commit until its reader learns to ignore it.

1. **The cache key pins `tensor_layout`.** Read `compute_program_hash` and `operation_attributes_t::to_hash()` in `unary_device_operation.cpp`: **no two tensors differing in dtype, page config (*including* `Tile`), memory config, or `Alignment` may share a cache entry.** Check that property, not any particular implementation of it — hashing the field, normalising it, or rejecting the divergent domain all satisfy it equally.

   **At the time of analysis this check FAILED, and it is the reason this document does not yet clear.** `TensorLayout::attribute_values()` is `(dtype, page_config, memory_config, alignment)`; the key carries `input_tensor.layout()`, which is the `Layout` **enum** (`TILE` / `ROW_MAJOR`) and not the page config, so **`Tile` is unpinned**, and **`Alignment` is absent entirely**. Two `bfloat16` TILE tensors with identical padded shape and memory config but tiles `32x32` and `16x32` were confirmed on silicon to share one cache entry. **If two such tensors can still collide, STOP** and report UNCONFIRMED: every relaxation requires exact `tensor_layout` equality and no flag reaches inside it, so a declaration against a key that does not pin it turns a working `ttnn.relu` into a hard `TT_FATAL` on the second dispatch.

   **The resolution is one line, and it is the minimal one.** `TensorSpec::attribute_values()` is `(logical_shape, tensor_layout)` and `TensorLayout` holds no shape, so the two are cleanly separable: hashing `input_tensor.tensor_spec().tensor_layout()` in place of the present `dtype()` / `layout()` / `memory_config()` triple pins exactly what `tensorspecs_match_with_relaxation` requires — and nothing more, leaving the deliberate omission of shape intact. It is a strict superset of what the key carries today, so it can only split cache entries that are currently shared, never merge ones that are currently separate.

   Note what this does **not** fix, and deliberately so: unary mis-sizes its buffers for a non-`32x32` tile regardless of caching, and pinning `Tile` gives each tile its own equally-wrong entry. That is a live bug, but it is **not this port's to fix and not a reason to hold the port** — see [§4](#4-not-covered), where the same bug is confirmed in an already-ported sibling.

2. **The dataflow kernels compile the accessor away when the slot is sharded.** `SRC_SHARDED` / `DST_SHARDED`, set from `has_sharding && is_sharded()`. In `reader_unary.cpp` the `TensorAccessorArgs<0, 0>()` and `TensorAccessor` declarations sit inside the `#else` of `#if SRC_SHARDED`, and `writer_unary.cpp` mirrors it. This is what makes the native-sharded rows safe — the sharded accessor payload is emitted by the host and never read by the device.

3. **`has_sharding` is itself pinned by the key.** `dst_shard_vol.has_value()` is `true` exactly when `get_shard_specs(...)` returned a value, and both shard-volume optionals are hashed. This is load-bearing: `SRC_SHARDED` / `RM_INTERLEAVED` are *compile-time defines* and the per-core runtime-arg slot count differs between the sharded (3) and interleaved (8) forms, so a code path that could flip on a cache hit would be unfixable by any override. It cannot flip.

4. **The op still has exactly one factory** (`std::variant<ProgramFactory>`, no `select_program_factory`), so the declaration is unconditional across factory choice.

5. **The TILE-path key still omits shape, and the override still re-applies the whole split.** `compute_program_hash` hashes `padded_shape` only on the `ROW_MAJOR` branch, and `override_runtime_arguments` re-enumerates the work split through the same `enumerate_core_rt_args` the miss path uses. If either changes, the regime table in §3 is reasoning about an op that no longer exists — in particular, a key that gains shape makes most of this document inert rather than wrong.

*Provenance, not a gate:* analysed against `origin/main` at the tree containing `unary_program_factory.cpp` with `enumerate_core_rt_args` shared between `create_descriptor` and `override_runtime_arguments`.

---

## 1. For the auditor

**Relaxation verdict: `dynamic` — conditional on validity check 1, which currently fails.**

Report **UNCONFIRMED** while the `Tile` gap is open. Once it is closed by either route, the verdict is `dynamic` and the declaration in §2 applies as written; nothing else in this document changes with the choice of route.

This clears the relaxation conjunct only — the op's other gate conjuncts are the sheet's to answer, as usual. Note that the sheet's `Known op issues` cell for this op is a second, independent block, and it is **not** cleared by this document.

### The rule behind validity check 1

> **An op's `compute_program_hash` must pin at least everything `tensorspecs_match_with_relaxation` requires to be exactly equal** — the whole `tensor_layout` (dtype, page config *including* `Tile`, memory config, alignment) per slot, plus the sharded distribution geometry for sharded slots.
>
> Where the key is looser than the declaration, you get spurious throws. Where the key is looser than what the factory *bakes*, you get silent corruption. The declaration can only ever fix the second.

Unary is currently on the wrong side of both halves of that rule for `Tile`: the factory bakes a `32x32` assumption into its buffer page sizes (`tile_size(DataFormat)` ignores tile dims) while reading the *real* tile for the work split, and the key separates neither. That is a live correctness bug today, independent of this port — see [§4](#4-not-covered).

---

## 2. For the porter — what to write

Once validity check 1 passes, the instruction is one line, applied to **both** `TensorParameter`s — input and output, unconditionally:

```cpp
.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
```

- **`dynamic_tensor_shape`** is mandatory, not optional: the TILE-path key omits `padded_shape` entirely, so one cache entry legitimately serves many shapes. Without it the *first* cache hit at a different shape throws.
- **`relax_logical_rank`** is required for the same reason — the TILE key omits rank along with the rest of the shape, so two tensors of different logical rank reach the same entry. Hashing `tensor_layout` per validity check 1 does not change this: rank lives in `logical_shape`, on the other side of the split.
- **Do not** set `match_page_size`, and note the reasoning differs from [binary_ng](eltwise_binary_ng.md)'s even though the answer matches. binary_ng declines it because its row-major kernels override the accessor's page size per dispatch. Unary would be *entitled* to set it — it hashes `padded_shape` on the `ROW_MAJOR` branch, so it does independently pin the last-dim width, which is exactly the condition binary_ng's rule names — but it is declined here on precedent grounds: **no shipped factory in the tree sets `match_page_size`**, and unary should not be the op that introduces an untravelled flag while also being the first shipped op to declare a relaxation at all (see below). Revisit if a second op needs it.
- **Do not** set `match_padded_shape_only`. It is strictly weaker than `dynamic_tensor_shape` and pins nothing this op needs.

**Precedent, so the porter knows how much road is ahead.** At the time of analysis, **zero** non-experimental shipped factories declare any `TensorSpecRelaxations`; the only in-tree factory precedent is `experimental/quasar/transpose`, where five factories set `.relaxations = {.dynamic_tensor_shape = true}` unconditionally on both input and output, each with a comment recording that it mirrors a legacy `RuntimeTensorShape` accessor. Unary matches that shape and adds `relax_logical_rank`, which no shipped factory sets yet. Treat a validation throw during the port as a plausible framework-side gap, not automatically a mistake in the declaration.

> **One stop condition.** If you are porting a configuration where the input tensor is **sharded but the op took the interleaved code path** — row 5 below — **stop and ask**. It is the one regime where the accessor is live *over a sharded buffer*, so the declaration's geometry term does real work instead of pinning dead code, and unlike binary_ng's equivalent row it is plainly reachable.

---

## 3. Why — the derivation

Declarations are per slot, per cache entry, written by the factory at cache **miss** while it holds the actual tensors. The rows below are conditioned on the runtime tensor to show the reasoning is regime-complete.

| # | Runtime condition | `dynamic_tensor_shape` | `relax_logical_rank` | Confidence |
|---|---|---|---|---|
| 1 | interleaved, TILE | true | true | **High** |
| 2 | interleaved, ROW_MAJOR | true | true | **High** |
| 3 | native-L1 sharded, TILE | true | true | Medium |
| 4 | native-L1 sharded, ROW_MAJOR | true | true | Medium |
| 5 | sharded buffer, interleaved code path | true | true | **Low — stop and ask** |

All five land on the same declaration; the regimes differ only in *why* it is safe.

**Row 1 — the case the relaxation exists for.** The key omits shape and rank, the override re-applies the split, and the only interleaved-TILE quantity outside per-core RTAs is `aligned_page_size`, which is `f(dtype, Tile)`. So with validity check 1 satisfied, the match collapses to bare `tensor_layout` equality and the shard term is `nullopt` on both sides. This is the regime that carries almost all of unary's traffic.

**Row 2 — nearly inert.** The `ROW_MAJOR` branch of the key hashes `padded_shape`, so shape barely varies within an entry to begin with; the relaxation is declared for uniformity rather than need. Because the key pins the width, `dynamic_tensor_shape`'s dynamic `aligned_page_size` common runtime argument is re-derived to the same value on every dispatch — which is why declining `match_page_size` costs nothing here beyond one runtime word.

**Rows 3 and 4 — safe on dead code, but still validated.** Per validity check 2 the accessor does not exist in the compiled kernel on a sharded slot, so the distribution geometry the relaxation pins is not read by anything. Two caveats keep this at Medium rather than High. First, that is a property of the current kernel sources, not of the framework — a future kernel that reads the accessor on the sharded path would move these rows onto row 5's argument. Second, `tensorspecs_match_with_relaxation` runs in `UpdateTensorArgs` regardless of what the kernel reads, so a squeeze-induced geometry mismatch would still *throw* even though no kernel would have mis-addressed. That is a spurious-throw risk, not a corruption risk, and it is bounded by how restrictive `is_native_L1_sharding` already is (even shards, identical in/out grids, L1 only).

**Row 5 — the row not to trust.** A tensor can be sharded while the op runs its interleaved path, because `get_shard_specs` returns `nullopt` on three separate fallbacks: `is_native_L1_sharding` failing (DRAM, mismatched grids, or an uneven input), an uneven *output*, and a `ROW_MAJOR` shard whose element count is not tile-aligned — that last one even emits a `log_warning`. In that state `has_sharding` is false, so `SRC_SHARDED` is `0` and **the accessor is live over a sharded buffer**. The key pins the shard spec (it hashes `input_tensor.memory_config()`, which carries it) but not the shape, and the distribution geometry's squeeze depends on shape *values* — so two shapes that agree on the shard spec can still resolve to different geometry, which `tensorspecs_match_with_relaxation` rejects rather than silently mis-addressing. Unlike binary_ng's equivalent row, this one has documented, warning-logged routes into it, so "probably dead in practice" is not available as a defence. It needs either a reachability argument or a key that separates the geometry.

### Two things worth recording so they are not re-derived

- **`has_sharding` cannot flip on a cache hit**, so the compile-time `SRC_SHARDED` / `RM_INTERLEAVED` defines and the 3-vs-8 runtime-arg slot count are safe. This is validity check 3, and it is the same coincidence binary_ng relies on: a hashed optional whose `has_value()` tracks the code path exactly. It is a coincidence of the current key, not a design invariant — it breaks if anyone replaces the shard-volume optionals with plain integers.
- **The override's accessor-refresh loops are bounded by the cached buffer length** (`for (i = 0; i < common_args.size() && i < reader_common.size(); ++i)`), so a fresh tensor needing *more* accessor words would be silently truncated rather than caught. Under `dynamic_tensor_shape` the word count varies with rank for sharded slots (`tensor_accessor_args.cpp` computes `n_args` with a `rank * add_tensor_shape` term). Whether a mismatch is reachable was **not** established — the distribution spec squeezes rank before that point, which may well preclude it. Flagged because the `&&` guard would mask it either way.

---

## 4. Not covered

- **Non-`32x32` tile support, as distinct from `Tile` in the key. This is a family-wide bug, and it does not gate this port.** `create_descriptor` sizes its buffers with `tile_size(cb_data_format)`, which takes only a `DataFormat` and therefore assumes `32x32`, while `enumerate_core_rt_args` reads the real `tensor_spec().tile()` for the work split. The two disagree, and nothing in `eltwise/` guards the tile. Confirmed on silicon: `ttnn.relu` on a `16x32`-tile `bfloat16` tensor returns wrong data **in isolation**, with a clean `from_torch`/`to_torch` round-trip at the same tile — an op bug, not a caching artifact.

  The reason it does not gate: **`ttnn.typecast` fails identically** (same shape, same tile, isolation, `max_abs_err` ≈ 0.98), and `copy/typecast` is **already ported** — all four of its factories are on `create_program_artifacts`. It sizes every TILE-layout buffer with `tile_size(DataFormat)` and never reads the tensor's real tile, and it carries no tile guard. So a ported sibling already ships this exact defect, which settles that the defect is orthogonal to Metal 2.0 rather than something a port must clear. Fixing it inside the unary port would also violate the porting invariant, since it changes behaviour the sentinels are supposed to hold fixed.

  Worth noting how narrow real support is elsewhere: `data_movement/tilize` is the only op in this neighbourhood that engages with the question deliberately — it sizes buffers from the *real* tile (`operation_attributes.tile.get_tile_size(df)`) and `TT_FATAL`s that the tile **width** is 32 while permitting a smaller height, with a comment saying exactly that. `untilize` has unary's own split personality (real tile for work partitioning, `tile_size(DataFormat)` for buffers) and its codegen path rejects non-`32x32` outright. So "the ported ops support non-`32x32` tiles" is not the state of the tree; only tilize does, partially and on purpose. This belongs to the eltwise team as one issue spanning at least `unary` and `typecast`; the same gap is recorded against [binary_ng](eltwise_binary_ng.md).
- **`Alignment`.** Absent from the key, like `Tile`, and fixed by the same one-line change in validity check 1. No python surface was found for constructing a tensor with a non-derived `Alignment` (only `ShardShapeAlignment` is exported), so it may be C++-reachable only — but this is moot once `tensor_layout` is hashed, which is why no reachability argument is needed for it. For contrast, every ported sibling checked (`typecast`, `tilize`, `untilize`, `transpose`) pins both `Tile` and `Alignment` **for free**, because none of them defines a custom `compute_program_hash` and the framework default hashes the whole `TensorSpec`. Unary's gap exists precisely because its custom hash drops fields the default would have kept.
- **`nd_shard_spec` configurations.** `MemoryConfig` can carry one, and the op's sharding helpers read only `shard_spec()`. Not analysed.
- **The independent Quasar clone** under `experimental/quasar/`, which was not consulted.
- The port's own mechanics. Nothing in this document is a porting instruction beyond §2.
