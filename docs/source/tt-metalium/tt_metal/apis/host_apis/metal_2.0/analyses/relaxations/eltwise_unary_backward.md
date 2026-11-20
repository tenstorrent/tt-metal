# TensorParameter relaxations — `eltwise/unary_backward`

**Author:** Anasuya and Claude

**Purpose:** State the Metal 2.0 `TensorSpecRelaxations` declaration this op requires, per `TensorParameter`, so that neither the auditor nor the porter has to derive it. The readiness sheet's `TensorParameter relaxation` cell points here.

**Covers one sheet row:** `eltwise/unary_backward` · `UnaryBackwardDeviceOperation` · `UnaryBackwardProgramFactory`.

**Supersedes** the earlier version of this document, whose verdict was `none` conditional on a whole-`TensorSpec` key (#63334). That verdict was correct when written. It was overtaken by #64663 and #63799; see §4.

---

## Contract — read this before using the document

This is **not** the same contract as the [offset-base-pointer](https://github.com/tenstorrent/tt-metal/blob/akertesz/op-porting-recipe/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/2026-07-19_offset_base_pointers.md) and [3rd-argument](https://github.com/tenstorrent/tt-metal/blob/akertesz/op-porting-recipe/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md) triage docs. Those are *priors* layered on a scan the auditor runs anyway, so a disagreement means "the doc is stale, trust your own scan."

Here there is no scan to fall back on: the audit recipe forbids re-deriving a relaxation. So this document is **authoritative but perishable**.

> **If any validity check fails, do not substitute your own judgement. Stop, and report the relaxation verdict as UNCONFIRMED.**

### Validity checks — confirm all six before applying anything below

Each is one read or one grep. A commit stamp is deliberately not used.

1. **The cache key pins each tensor exactly as the declaration matches it.** Read the body of `compute_program_hash` in `unary_backward_device_operation.cpp`. Today it keys `input`, `grad_output` and any preallocated output with `tt::tt_metal::hash_tensorspec_with_relaxation(spec, {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true})` (#63799). Check the property, not the implementation: **no two tensors that differ in padded shape, dtype, page config (*including* `Tile`) or memory config may share a cache entry, for any of the three slots.** A key that is *stricter* than that (for example, the whole `TensorSpec`) still passes; it only splits the cache more finely. **If any of those terms can diverge within one entry, STOP.**
2. **The factory never reads the logical shape or the `TensorLayout` alignment.** (`mesh_buffer().alignment()` is the buffer's allocator alignment, a function of the buffer type in the memory config, and is fine. Comments that say "logical page" are about page indexing and are fine too.)
3. **The derived output's padded shape still comes from the input's padded shape.** `compute_output_specs` builds the non-preallocated output with `tensor_layout_from_padded_shape(..., input.logical_shape(), input.padded_shape())`. Its dtype and memory config come from the hashed `args`. Its alignment is derived from the input's logical shape and therefore varies within one cache entry, which is why the output slot needs `relax_tensor_layout_alignment` as well.
4. **The op is still the base concept:** no `override_runtime_arguments`, no `get_dynamic_runtime_args` (after the port: `ProgramSpecFactoryConcept`). A cache hit refreshes only tensor bindings. Every per-core runtime arg is baked at the cache miss from the padded shape (`num_tiles` from `physical_volume()`, the row-major `row_elements`, `total_rows` and per-tensor paging). This is what rules out `dynamic_tensor_shape`.
5. **Validation still ties the slots together and runs on cache hits.** `validate_on_program_cache_miss` requires `grad_output` and any preallocated output to match the input's logical shape, padded shape and layout. The op declares no `validate_on_program_cache_hit`, so the framework runs the miss validator on every hit.
6. **The op still has exactly one factory** (`program_factory_t = std::variant<UnaryBackwardProgramFactory>`, no `select_program_factory`), so the declaration is unconditional across factory choice.

*Provenance, not a gate:* analysed against `main` at `f0d4c1fc7d0` (2026-11-20), which includes #64663 (`e8b68619482`) and #63799 (`2301d23c13b`).

---

## 1. For the auditor

**Relaxation verdict: `legality`.** The declaration is in §2; carry it into the brief. This clears the relaxation conjunct only. The op's other gate conjuncts are the sheet's to answer, as usual.

Why `legality` and not `none`: the key deliberately leaves each tensor's logical shape and `Alignment` free, so one cache entry legitimately serves tensors whose `TensorSpec`s differ in exactly those two terms. Under the default exact match, the second such call would throw on the cache hit. The declaration makes those calls legal. Nothing about the shape is dynamic: the padded shape is pinned, so this is not a `dynamic` verdict.

No pre-port fix is owed. #63799 already brought the key into this form.

### The rule behind validity check 1

> **An op's `compute_program_hash` must pin at least everything `tensorspecs_match_with_relaxation` requires to be exactly equal under the declared relaxation.** Under `{match_padded_shape_only, relax_tensor_layout_alignment}` that is each slot's `tensor_layout` *except* `Alignment` (dtype, page config including `Tile`, memory config), plus its padded shape, plus the sharded distribution geometry for sharded slots.
>
> Where the key is looser than the declaration, you get spurious throws. Where the key is looser than what the factory *bakes*, you get silent corruption. The declaration can only ever fix the second.

The recipe's general statement of this rule ("`tensor_layout` … is freed by **no** flag") predates #64663. `relax_tensor_layout_alignment` now frees `Alignment` when combined with `match_padded_shape_only` or `dynamic_tensor_shape`. Every other `tensor_layout` field is still freed by no flag.

---

## 2. For the porter — what to write

Once the checks in the contract pass, the entire instruction is one line, applied to **every** `TensorParameter` (`grad_output`, `input` and `output`), unconditionally:

```cpp
.relaxations = {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true},
```

- **Do not** leave it at `{}`. The key admits tensors with a different logical shape or `Alignment` into one entry, so a strict declaration throws on the second call.
- **Do not** set `match_padded_shape_only` alone. Plain `match_padded_shape_only` still requires equal `Alignment`.
- **Do not** set `dynamic_tensor_shape`. It frees the padded shape, and this op bakes padded-shape-derived runtime args that a base-concept cache hit never refreshes (validity check 4). A different padded shape would leave them stale: silent corruption, not a throw.
- **There is no conditional.** The declaration is identical for every slot in every layout and sharding regime, and for every op type. The regime analysis in §3 is *audit* material; it does not branch the code you write.

---

## 3. Why — the derivation

Per `tensor_spec_relaxations.hpp`, `match_padded_shape_only` frees the logical shape and requires the padded shape to match exactly. `relax_tensor_layout_alignment` additionally frees the layout's `Alignment`. Both leave the `TensorAccessor` configuration completely unchanged: alignment reaches the accessor only through the padded shape and the row-major page size, and `match_padded_shape_only` pins both.

| Term | Pinned by | Status |
|---|---|---|
| logical shape | nothing — freed by the declaration | free by design |
| `Alignment` | nothing — freed by the declaration | free by design |
| padded shape | the key, per slot; the derived output copies the input's (check 3) | pinned |
| dtype | the key, per slot; the derived output's comes from hashed `args` | pinned |
| page config (layout, `Tile`) | the key, per slot; validation also rejects any tile but 32×32 | pinned |
| memory config (incl. shard spec) | the key, per slot; the derived output's comes from hashed `args` | pinned |

The key computes each slot's term with the framework's own `hash_tensorspec_with_relaxation`, using the same flags as the declaration. The key and the declaration therefore admit exactly the same set of tensors, by construction.

Everything the factory turns into compile-time or runtime args is a function of pinned terms:

- **Work split:** `num_tiles = input.physical_volume() / TILE_HW`, a function of the padded shape.
- **Row-major paging:** `row_elements = input.padded_shape()[-1]`, `total_rows`, and each tensor's `page_size()` and buffer `alignment()`. These depend on the padded shape, the memory config and the buffer type, never on the layout's `Alignment` directly.
- **Sharding and aliasing:** the shard spec comes from the memory config. The aliasing preconditions (all three slots L1-sharded, even shards, one identical shard spec, a whole shard per core) use the padded shape, `physical_volume()` and the shard spec, so the mode cannot flip on a cache hit.
- **Accessors:** unchanged by either flag (see above).

Two specs that agree on the pinned terms therefore produce byte-identical programs. The freed terms are representational: a logical 40×40 inside a 96×96 padded shape and a logical 96×96 have different `TensorSpec`s and the same physical layout.

### Reachability

The op's own `compute_output_specs` comment documents the case: a tensor from `tilize_with_val_padding` can hold a logical 40×40 inside a 96×96 padded shape. A plain `from_torch` tensor of logical 96×96 has the same padded shape but a different logical shape and `Alignment`. The second case is any two logical shapes that tile-pad to the same padded shape (30×30 and 32×32). Both are covered by tests added in #63799.

### Regime matrix

All four rows land on the same declaration. The columns show that the reasoning is regime-complete, not that the code branches.

| # | Regime | Accessors in the compiled kernels | Padded-shape-derived bakes | Declaration | Confidence |
|---|---|---|---|---|---|
| 1 | interleaved or mixed, TILE | three live | per-core `units` / `units_done` | both flags | **High** — exercised by #63799's tests |
| 2 | L1-sharded aliased, TILE | none in the reader; the writer's is dead under `OUT_SHARDED` | shard grid and shard tile count | both flags | **High** |
| 3 | interleaved or mixed, ROW_MAJOR | three live | `row_elements`, `total_rows`, `chunks_per_row`, `rows_per_tile`, per-tensor paging | both flags | Medium |
| 4 | L1-sharded aliased, ROW_MAJOR | as row 2 | as row 2 | both flags | Medium |

**Why rows 3 and 4 are Medium.** They are reasoned from the current `rm_blocked` arithmetic, which reads only the padded shape and buffer properties. No test makes two row-major calls with different logical shapes or alignments share a cache entry.

---

### Claims that did not survive, recorded so they are not re-derived

- **"A relaxation flag cannot substitute for hashing the whole `TensorSpec`; no flag reaches inside `tensor_layout`."** True until #64663 (2026-11-17). That PR extended `relax_tensor_layout_alignment` (added for `dynamic_tensor_shape` by #64290) to `match_padded_shape_only`. Since then a declaration can free `Alignment`, and the op keys on exactly that (#63799). The same sentence in #59915 ("No `TensorSpecRelaxations` flag can relax a layout field") is now true for every layout field except `Alignment`.
- **"`none` is the verdict once the key hashes each tensor's whole `TensorSpec` (#63334)."** Correct for that key. #63799 deliberately loosened the key to restore the cache reuse the pre-#63334 key had (the legacy key hashed dtype, memory config, input padded shape and layout, and left the logical shape and `Alignment` free). With the looser key, `{}` throws on a hit.
- **"`match_padded_shape_only` alone is the minimal declaration."** It is not: it keeps `Alignment` pinned, and the key does not.
- **"Divergent `Alignment` corrupts the row-major page geometry."** It does not. With the padded shape pinned, the page shape, page size and baked `page_elements` are identical. The divergence is representational only.
- **"The loose key lets a mismatched operand skip validation."** It does not. With no `validate_on_program_cache_hit`, the miss validator runs on every hit (validity check 5). Check for a hit validator before concluding that validation is bypassable.

---

## 4. Not covered

- **The sheet reconciliation.** The sheet's `eltwise/unary_backward/gelu_bw`, `.../tanh_bw` and `experimental/unary_backward/gelu_backward` rows describe device operations deleted by #58352, #57832 and #54799. The live `UnaryBackwardDeviceOperation` has no row. That is a sheet matter, not a relaxation matter.
- **Edwin's other #63798 ops** (roll, reshape_on_device, interleaved_to_sharded_partial). They use the same pair of flags on their input slot, but are **not** analysed here.
- **`Tile` support**, as distinct from `Tile` in the key. Validation rejects non-32×32 tiles outright.
- **The port's own mechanics.** Nothing in this document is a porting instruction beyond §2.
