# TensorParameter relaxations — `eltwise/unary_backward`

**Author:** Anasuya and Claude

**Purpose:** State the Metal 2.0 `TensorSpecRelaxations` declaration this op requires, per `TensorParameter`, so that neither the auditor nor the porter has to derive it. The readiness sheet's `TensorParameter relaxation` cell points here.

**Covers one factory, not one sheet row.** `UnaryBackwardDeviceOperation` · `UnaryBackwardProgramFactory`, which serves **all four** values of `UnaryBackwardOpType` — `SIGMOID_BW`, `TANH_BW`, `GELU_BW`, `GELU_TANH_BW`. The sheet's granularity is one row per *(op, DeviceOperation, ProgramFactory variant)*, which does not model several ops sharing one factory, so this document deliberately covers a factory. The sheet's `eltwise/unary_backward/gelu_bw` and `.../tanh_bw` rows describe `GeluBwDeviceOperation` and `TanhBwDeviceOperation`, both **deleted** — by `e8cbbd6b7d9` (#57832) and `9ad6239dc29` (#58352) — when those ops were consolidated onto this factory. `SIGMOID_BW` and `GELU_TANH_BW` have no row at all. Reconciling that is a precondition for the audit and is not this document's job.

---

## Contract — read this before using the document

This is **not** the same contract as the [offset-base-pointer](../2026-07-19_offset_base_pointers.md) and [3rd-argument](../2026-07-06_tensor_accessor_3rd_arg_triage.md) triage docs. Those are *priors* layered on a scan the auditor runs anyway, so a disagreement means "the doc is stale, trust your own scan."

Here there is no scan to fall back on: the audit recipe forbids re-deriving a relaxation. So this document is **authoritative but perishable**.

> **If any validity check fails, do not substitute your own judgement. Stop, and report the relaxation verdict as UNCONFIRMED.**

### Validity checks — confirm all six before applying anything below

1. **The cache key pins the whole `TensorSpec` of every bound tensor.** Read the body of `compute_program_hash` in `unary_backward_device_operation.cpp` — **do not grep the file for `logical_shape`**, which matches eight times today, all of them in `validate_on_program_cache_miss` and `compute_output_specs` and none of them in the key. The property to confirm is that **no two tensors differing in logical shape, dtype, page config (*including* `Tile`), memory config, or `Alignment` may share a cache entry — for `grad_output` and any preallocated output as much as for `input`.** Check that property, not any particular implementation of it: hashing each tensor's `tensor_spec()` wholesale, hashing the fields individually, or rejecting the divergent domain all satisfy it equally. **This check does NOT pass on `main` at the provenance commit below** — it is the pre-port fix this verdict is conditional on, and §1 explains why that fix is owed regardless of the port. **If any bound tensor's spec can still diverge within one entry, STOP** and report UNCONFIRMED.
2. **The factory still never reads `logical_shape`.** One grep over `unary_backward_program_factory.cpp`. The verdict rests on the program built for one logical shape being byte-identical to the program built for another under the same padded shape; a factory that reads the logical shape destroys that premise.
3. **The op is still the base `ProgramSpecFactoryConcept`** — no `override_runtime_arguments`, no `get_dynamic_runtime_args`. A cache hit then refreshes *only tensor bindings*, so every per-core runtime arg is baked at cache miss and never refreshed. This is what makes `dynamic_tensor_shape` unsafe here, and it is the single biggest difference from the ops that do set it.
4. **The row-major blocked path still bakes padded-shape-derived geometry into per-core runtime args** — `row_elements = input.padded_shape()[-1]` and, through `buffer()->page_size()`, each tensor's `page_elements` / `pages_per_row` (`unary_backward_program_factory.cpp`, the `rm_blocked` branch). With check 3, these never refresh, so the padded shape must stay *exactly* pinned.
5. **Validation still rejects any tile other than 32×32 on a TILE-layout operand** (`validate_operand` in `unary_backward_device_operation.cpp`). A ROW_MAJOR operand has no tile to reject, because `RowMajorPageConfig` carries no `Tile`. With the fix the page config's `Tile` term is in the key anyway, via the spec, so this check is now belt-and-braces rather than load-bearing: it is what keeps the *pre-fix* key sound on this term, and it bounds the regimes this analysis had to consider.
6. **The op still has exactly one factory** (`program_factory_t = std::variant<UnaryBackwardProgramFactory>`, no `select_program_factory`), so the declaration is unconditional across factory choice.

*Provenance, not a gate:* analysed against `main` at `6d8a1032acc`; the fix described in §1 is implemented on `anasuya/unary_backward_preport_hash`. Re-validated at `42d828a3882` by the pre-port audit (2026-10-29). The only in-scope change since `6d8a1032acc` is #62567, which touches no check. The fix is draft PR #63334.

---

## 1. For the auditor

**Relaxation verdict: `none` — conditional on a pre-port hash fix.**

Two statements, and both matter:

1. **As the op stands, `none` is not safe.** The key names `input.padded_shape()` and `input.layout()`, but no logical shape and no alignment. So two calls agreeing on every hashed term can present `TensorSpec`s that differ in **logical shape and `Alignment` — those two only**, and the default exact match rejects the second on a cache hit. Padded shape and layout cannot diverge: the key pins the input's, and validation (which runs on hits, see below) ties the other slots to the input.
2. **With the fix, `none` is correct and no relaxation is needed.** Keying on each bound tensor's whole `tensor_spec()` pins every field the match compares, by construction: the spec carries the logical shape and the tensor layout, and the padded shape follows from the logical shape and the alignment. Exact matching then holds in every regime.

**The fix is a port prerequisite only — there is no live bug.** It is tempting to argue that because `validate_on_program_cache_miss` compares `grad_output` and any preallocated output against the input, and the old key did not pin their shapes, a mismatched operand could collide onto a valid entry and skip validation. **That argument is wrong, and the misleading function name is the trap.** An op that declares no `validate_on_program_cache_hit` has `validate_on_program_cache_miss` run on hits too — see the fallback in `device_operation.hpp` (`handle_mesh_adapter_cache_hit`), the same fallback in `mesh_device_operation_adapter.hpp`, and the comment on `HasValidateOnProgramCacheHit` in `operation_concepts.hpp`: "If not provided, the framework defaults to calling `validate_on_program_cache_miss`." This op declares no hit validator, so full validation runs on every dispatch. The behaviour dates to #37633 (2026-02-12).

So a mismatched `grad_output` is rejected whether or not it collides, nothing is read or written out of bounds, and the only consequence of the loose key is the one in point 1: two *valid* calls sharing an entry they must not share after the port.

Route it to the op owner before the port; the relaxation column can read `none` once it lands. This is the same class of pre-port key repair that `eltwise/ternary` needed.

### Why a relaxation flag cannot substitute for the fix

`match_padded_shape_only` is the flag that looks like it fits — it permits a differing logical shape while requiring the padded shape to match exactly, and it leaves `TensorAccessor` configuration untouched, which suits an op whose kernels never see a logical shape. It still does not work. Alignment lives *inside* `tensor_layout`, every relaxation requires `tensor_layout` equality, and **no flag reaches inside it**. The very pair that motivates the relaxation — same padded shape, different logical shapes — is also the pair whose alignments differ. The flag would relax the first difference and throw on the second.

### The rule behind validity check 1

> **An op's `compute_program_hash` must pin at least everything `tensorspecs_match_with_relaxation` requires to be exactly equal** — the whole `tensor_layout` (dtype, page config *including* `Tile`, memory config, alignment) per slot, plus the sharded distribution geometry for sharded slots.
>
> Where the key is looser than the declaration, you get spurious throws. Where the key is looser than what the factory *bakes*, you get silent corruption. The declaration can only ever fix the second.

This op is in the first case, not the second — see §4.

---

## 2. For the porter — what to write

Once the checks in the contract pass, leave the relaxations at their default on **every** `TensorParameter` — both operands and the output, unconditionally:

```cpp
// Exact match. compute_program_hash keys on each bound tensor's whole TensorSpec.
// A cache hit therefore always sees the same specs as the miss that built the program.
.relaxations = {},
```

- **Do not** set `dynamic_tensor_shape`. It permits the padded shape to vary, and this op bakes padded-shape-derived geometry into per-core runtime args that a base-concept cache hit does not refresh (validity checks 3 and 4). A varying padded shape would leave `row_elements` and each tensor's `page_elements` stale, which is silent corruption rather than a throw.
- **Do not** set `match_padded_shape_only`. With the hash fix in place it is inert — the logical shape it would relax is pinned anyway. Without the fix it is insufficient, for the alignment reason in §1. Either way it is the wrong answer.
- **There is no conditional.** The declaration is identical for every slot in every layout and sharding regime. The regime analysis in §3 is *audit* material; it does not branch the code you write.

> **One stop condition.** If validity check 1 fails — the key still does not pin the logical shape — **stop and report**, do not port with `.relaxations = {}` and do not invent a flag. A strict declaration against a looser key turns a working `ttnn.gelu_bw` on an overpadded tensor into a hard runtime error on the second call.

---

## 3. Why — the derivation

### What the exact match compares, and what pins each term

`TensorSpec::operator==` is `= default`, so it compares **all five** members: `logical_shape_`, `tensor_layout_`, and the three caches `cached_padded_shape_`, `cached_logical_2d_shape_` and `cached_physical_shape_`. Hashing walks a different and smaller set — `attribute_values()` is just `(logical_shape_, tensor_layout_)` — and that is sound only because the three caches are pure functions of those two. The framework asserts the same equivalence for its own whole-spec hash (`hash_tensorspec_with_relaxation`). `TensorLayout`'s attributes are `(dtype_, page_config_, memory_config_, alignment_)`, so six terms have to be pinned per slot:

| Term | Pinned by | Status |
|---|---|---|
| `logical_shape` | nothing today; each slot's whole `tensor_spec()` after the fix | **a gap today** |
| `dtype` | hashed per slot; also `output_dtype == input.dtype()` in validation | pinned |
| `page_config.layout` | `input.layout()` hashed; both other slots validated equal to it | pinned |
| `page_config.tile` | validation rejects anything but 32×32; also in the spec after the fix | pinned by rejection |
| `memory_config` | hashed per slot, operands and preallocated output alike | pinned |
| `alignment` | nothing today — see below | **a gap today** |

The three slots are tied together by validation — `grad_output` and any preallocated output must match the input's logical shape, padded shape and layout — and validation runs on cache hits too, so **a single logical-shape term in the key would cover the logical shape of all three.** The non-preallocated output is built from the input's shapes plus `args.output_dtype` / `args.output_memory_config`, which ride in the hashed `args`.

**It would not cover alignment, which is why the fix hashes each slot's whole spec rather than adding one shape term.** Alignment is usually a function of the shapes, but not necessarily: a caller may pass an explicit `Alignment` to `TensorLayout`, and `initialize_alignment` keeps it (rounded up against the layout default) instead of deriving one. Validation never compares alignment, so an explicitly aligned slot is pinned by nothing at all.

### Where alignment actually diverges

`legacyShapeToAlignment` returns `Alignment{}` whenever `logical_shape == padded_shape`, and otherwise derives values from the padded shape and from per-dimension over-padding predicates computed against the logical shape. Consequences, by regime:

*Read `Alignment{}` below as "no alignment supplied at this step", not as the value a spec ends up holding.* Every `TensorLayout` constructor runs `initialize_alignment`, which substitutes the layout's default alignment for an empty one (`tensor_layout.cpp`), so a constructed spec always carries a populated alignment. That changes none of the conclusions here: what matters is whether two specs carry the *same* alignment, and the substitution is a function of page config, dtype and memory config, all of which are already pinned.

- **Sharded, TILE** — `Alignment{}` on both branches, so alignment is *invariant* here even with a free logical shape.
- **Sharded, ROW_MAJOR** — `Alignment{}` when logical equals padded, `Alignment{shard_spec.shape[1]}` otherwise. Diverges.
- **Interleaved, either layout** — `Alignment{}` when logical equals padded, otherwise `{height_or_tile_h, width_or_tile_w}` selected by the over-padding predicates. Diverges, and can differ between two distinct over-padded logical shapes.

So three of the four regimes diverge, which is why the fix belongs in the key rather than in a per-regime declaration.

### Reachability

The op's own `compute_output_specs` documents the reachable case: a tensor from `tilize_with_val_padding` can carry a logical 40×40 inside a 96×96 padded shape, where tile-padding the logical shape alone would have given 64×64. A logical 40×40 and a logical 50×50 both sit in 96×96 with identical dtype, layout and memory config — identical under today's key, unequal as `TensorSpec` values.

### Regime matrix

All four rows land on the same declaration. The columns show that the reasoning is regime-complete, not that the code branches.

| # | Regime | Accessors in the compiled kernels | Padded-shape-derived bakes | Declaration | Confidence |
|---|---|---|---|---|---|
| 1 | interleaved or mixed, TILE | three live | per-core `units` / `units_done` | `{}` | **High** |
| 2 | L1-sharded aliased, TILE | reader has **none**; writer's is emitted but dead under `OUT_SHARDED` | shard grid and shard tile count | `{}` | **High** |
| 3 | interleaved or mixed, ROW_MAJOR | three live | `row_elements`, `total_rows`, `chunks_per_row`, `rows_per_tile`, per-tensor paging | `{}` | Medium |
| 4 | L1-sharded aliased, ROW_MAJOR | as row 2 | as row 2 | `{}` | Medium |

**Rows 1 and 3.** Nothing beyond the six terms above is at stake; with the key pinning all of them the match is exact and trivially satisfied. Row 3 is the row that forbids `dynamic_tensor_shape`, because its bakes are the richest and none of them refresh.

**Rows 2 and 4.** The aliased path is the strongest form of the "accessor is dead code" argument seen in sibling analyses: the reader does not merely ignore its accessor, it **has none compiled in at all**. The factory sets `IN0_SHARDED` / `IN1_SHARDED` and deliberately appends no `TensorAccessorArgs`, because "appending one for an aliased operand would shift the other's offset." The writer's accessor is appended unconditionally but is unreachable under `OUT_SHARDED`. The aliasing preconditions — all three slots L1-sharded, even shards, one identical `ShardSpec` including orientation, matching memory layout, and a grid holding exactly one shard per core — are all functions of hashed terms, so the mode cannot flip on a cache hit.

**Why rows 3 and 4 are Medium rather than High.** Both are reasoned from the current kernel sources and the current `rm_blocked` arithmetic rather than from a framework invariant, and neither was exercised on silicon for this document.

---

## 4. Claims that did not survive, recorded so they are not re-derived

- **"Divergent *alignment* corrupts the row-major page geometry today, pre-port."** It does not. `compute_physical_shape` rounds each logical dimension up to the corresponding `alignment_` entry, and `legacyShapeToAlignment` picks exactly the alignment that reproduces the legacy padded shape. So the spec derived for a logical 40×40 (alignment `{96,96}`) and the one derived for a logical 96×96 (alignment left empty, hence the layout default) yield the *same* physical shape, hence the same page shape, the same page size, and a correct baked `page_elements`. That divergence is **representational, not semantic** — invisible to the current `ProgramDescriptor` path and fatal only to a value-equality match.

- **"The loose key lets a mismatched operand skip validation, so this is a live bug."** It does not; see §1. `validate_on_program_cache_miss` runs on cache hits as well for any op that declares no `validate_on_program_cache_hit`, which is the case here. Do not reason from the function's name. **Any audit of a cache key that concludes "validation is bypassable" has to check for a hit validator first**, and this trap is not specific to this op.
- **"This op needs `dynamic_tensor_shape` because `eltwise/unary` sets it."** The sibling sets it, but the sibling is not analysed here and the premise does not transfer: this op has no mechanism to refresh the padded-shape-derived runtime args it bakes. Treat the contrast as a warning against copying the declaration, not as evidence about this op.
- **"`match_padded_shape_only` is the minimal correct declaration."** It is the right *shape* of answer and still wrong, for the alignment reason in §3: the flag relaxes the shape comparison but reaches nothing inside `tensor_layout`, where alignment sits. Recorded because it is the conclusion a reader reaches from the flag's doc comment alone, without following alignment into `tensor_layout`.

- **"Pinning the logical shape in the key is enough, so hash `input.logical_shape()`."** Too narrow, and it was this document's own first answer. Validation ties the three slots' logical shapes together and runs on hits, so one shape term really would cover the logical shape — but alignment can be supplied explicitly rather than derived (§3), and validation never compares it. Hashing each slot's whole `tensor_spec()` closes both terms at once and cannot go stale as the factory changes.

---

## 5. Not covered

- **The sheet reconciliation.** The two stale rows, the two gradients with no row, and the `Pointer patching perf issue` cell are audit/sheet matters, not relaxation matters.
- **`Tile` support**, as distinct from `Tile` in the key. Validation rejects non-32×32 tiles outright, so the question does not arise for caching; whether the op *should* support them is a separate correctness question.
- **The other relaxation rows** — `eltwise/unary`, `eltwise/binary_ng`, `eltwise/ternary` — are **not** analysed here.
- **The port's own mechanics.** Nothing in this document is a porting instruction beyond §2.
