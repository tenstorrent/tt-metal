# TensorParameter relaxations — `eltwise/ternary`

**Author:** Anasuya and Claude

**Purpose:** State the Metal 2.0 `TensorSpecRelaxations` declaration this op requires, per `TensorParameter`, so that neither the auditor nor the porter has to derive it. The readiness sheet's `TensorParameter relaxation` cell points here.

**Covers one sheet row:** `eltwise/ternary` · `TernaryDeviceOperation` · `TernaryProgramFactory`.

---

## Contract — read this before using the document

Same contract as [`eltwise_binary_ng.md`](eltwise_binary_ng.md), and for the same reason: the audit recipe forbids re-deriving a relaxation, because deriving one is the expert work the sheet column exists to record. There is no scan to fall back on, so this document is **authoritative but perishable**.

> **If any validity check fails, do not substitute your own judgement. Stop, and report the relaxation verdict as UNCONFIRMED.**

**This is the third revision of validity check 2, and the first in which the op is TILE-only by validation rather than by assumption.** The first revision asserted TILE-only and was wrong — row-major was reachable in several slots. The second admitted row-major conditionally and was also wrong, because the conditions were not exhaustive. Both are recorded in §5 so neither is re-derived; between them they cost five silently-wrong configurations, four of which the previous audit found. Row-major has now been removed from the op outright, which is what finally makes the check true. The declaration in §2 has never changed across the three revisions; every reason for it has.

### Prerequisites — this analysis is void without all three

The derivation was run against a tree carrying three fixes that are **not on `main`**; they are the base commit of `anasuya/metal2_eltwise_ternary` (`21d0f3c7ba9`, *Pre-port fixes*). Each one establishes a property a validity check asserts, so on a tree without them the corresponding check is false and the verdict reverts to UNCONFIRMED.

| Fix | Establishes |
|---|---|
| Every tensor slot must be TILE (`ternary_device_operation.cpp:383-404`, `:416`) | Validity check 2, the whole of it. Without it row-major reaches slots the kernels cannot address. Two narrower attempts at this rule were each defeated — see §5 — so the check is applied per slot against no condition. |
| A preallocated output's memory config wins over the `memory_config` argument (`:187-195`) | Validity check 1 for the output slot's memory config. Without it the key records the argument's memory while the writer stores to the output's (silicon-confirmed: 4095/4096 wrong). |
| The output's dtype, alignment and tile are hashed (`:644-652`) | The rest of validity check 1 for the output slot. Without it two preallocated outputs differing in alignment share one cache entry, which the relaxation check rejects. |

### Validity checks — confirm all five before applying anything below

Each is one grep against `ttnn/cpp/ttnn/operations/eltwise/ternary/`.

1. **The cache key pins `tensor_layout` for all four parameters.** For the three input slots, `compute_program_hash` hashes `dtype`, `memory_config`, `alignment` and `tensor_spec().tile()` independently per slot (`device/ternary_device_operation.cpp:624-638`). For the output it hashes `data_type()`, `get_alignment()` and `tile()` from the computed output spec (`:644-652`), with the memory config arriving through `args`. None of these terms is the layout. The key cannot tell a TILE tensor from a row-major one whose padding gives it the same alignment: a `[1,1,30,30]` row-major tensor padded to `[1,1,32,32]` hashes like its TILE twin. The property therefore holds only together with validity check 2, which rejects every row-major tensor on every dispatch. Check the property, not the implementation: **no two tensors differing in dtype, page config (including `Tile`), memory config or `Alignment` may share a cache entry, in any of the four slots.** **If two such tensors can still collide, STOP** — every relaxation below requires exact `tensor_layout` equality and no flag reaches inside it, so declaring against a key that does not pin it turns a working `ttnn.where` into a hard `TT_FATAL`.
2. **The op is TILE-only.** Every tensor slot is checked on its own, against no condition: `input_a`, `input_b`, `input_c` (`:398-404`) and a preallocated output (`:416`) must each be `Layout::TILE`. They run on every dispatch, not only on a cache miss. The op defines no `validate_on_program_cache_hit`, and the framework then runs `validate_on_program_cache_miss` on hits as well (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:228-233`). This is what keeps check 1 true, because layout is not a hash term. If a `validate_on_program_cache_hit` is ever added, it must keep the per-slot TILE checks. Without them, a row-major tensor that hashes like a TILE one would reach the cached TILE program. `compute_output_specs` then produces TILE in both branches, since the sharded branch takes `input_a`'s layout (`:492-494`) and that is now TILE by the same rule. **This check carries more weight than any other.** It is what makes `match_page_size` inert — see §2. Confirm the shape of the rule, not just its presence: **a per-slot check against no condition.** Both narrower forms of it have already been tried and defeated, each by a configuration that returned nearly all of its elements wrong (96.6–99.4% across the five probes; §5), so a check that is conditional on anything — the slot's sharding, another slot's layout, `input_a` — should be read as this check failing.
3. **`has_sharding` cannot flip on a cache hit.** `const bool has_sharding = shard_volumes.has_value()` (`device/ternary_program_factory.cpp:1053`), and `shard_volumes` is hashed (`ternary_device_operation.cpp:639`). The identity is exact, not approximate. This matters because `has_sharding` is pushed as a compile-time arg and gates every `SRC_SHARDED_*` / `DST_SHARDED` define, so a flip under a cached program would run a kernel compiled for the wrong addressing mode.
4. **The sharded distribution geometry is hashed for every slot.** `sharded_tensor_shape_in_pages` for all three inputs *and* the output (`ternary_device_operation.cpp:640-643`), plus `shard_volumes`, with the core grid, shard shape and orientation carried inside each slot's hashed `memory_config`.
5. **The op has exactly one factory** — `using program_factory_t = std::variant<TernaryProgramFactory>` (`device/ternary_device_operation.hpp:67`), no `select_program_factory` — so the declaration is unconditional across factory choice.

*Provenance, not a gate:* derived on branch `anasuya/metal2_eltwise_ternary`, now a single commit (`21d0f3c7ba9`, *Pre-port fixes*) **on `main` at `d764c9f5ce2`**, carrying the three fixes above. All line numbers are that tree's. They are not `main`'s: the citations into the three fixes point at code `main` does not carry, and because those fixes shift everything below them, the surrounding citations do not match `main` either even where the code is identical.

The derivation has survived two rebases. The first moved the base 478 commits to `26e6c5e2000b`; `compute_program_hash` and `resolve_mem_config_actual` are byte-identical across all of them and no upstream commit added layout validation, so only the line numbers changed, not the argument. The second moved it a further 253 commits to `d764c9f5ce2`. There the replayed patch is byte-identical to the pre-rebase one, and of the three upstream commits touching `ternary` or `binary_ng` in that span — #62492, #61721, #61591 — none touches any file carrying the three fixes. Two cited files did move: #61721 added 23 lines to `ternary_op_utils.cpp`, and `buffer_distribution_spec.cpp` shifted by nine. Citations into both were re-resolved by content, not by arithmetic. #61721 also added `has_empty_shards` as a new way for `is_native_L1_sharding` to come out false, which row 3 records.

Two things the first rebase did change, and both are followed up outside this document. #59392 refactored binary_ng's sharded gate and extended it to the tensor-tensor overload; that is binary_ng's analysis to absorb, not this one's, but the stop-condition bullet in §2 cites a row of it. #60282 landed. It closes the triggers of the pack-format and divide-by-zero suspicions that the pre-port audit had recorded as live; both former triggers run correctly on silicon, and the kernel hazards remain behind a host-side BF16 gate. ND-sharded tensors reach row 3 on this tree (#59788), and they are now probed on silicon — see §4.

---

## 1. For the auditor

**Relaxation verdict: `dynamic`.** The declaration is in §2; carry it into the brief. This clears the relaxation conjunct only — the op's other gate conjuncts are the sheet's to answer, as usual.

### The rule behind validity check 1

> **An op's `compute_program_hash` must pin at least everything `tensorspecs_match_with_relaxation` requires to be exactly equal** — the whole `tensor_layout` (dtype, page config *including* `Tile`, memory config, alignment) per slot, plus the sharded distribution geometry for sharded slots.
>
> Where the key is looser than the declaration, you get spurious throws. Where the key is looser than what the factory *bakes*, you get silent corruption. The declaration can only ever fix the second.

Both halves of that were live defects on this op, which is why the prerequisite table exists. The output slot was looser than the declaration in two ways at once: its memory config was taken from the wrong place, and its alignment and tile were not hashed at all. The first corrupts, the second throws once validation is switched on.

### Supersedes the issue-tracker conclusion

[Issue #54235](https://github.com/tenstorrent/tt-metal/issues/54235) recorded this row's answer as "`none`, with the caveat that `none` does not preserve current behaviour." **That conclusion is void and must not be carried forward.** It was derived while `override_runtime_arguments` still froze shape-dependent per-core arguments, so no declaration could be both safe and useful. #55167 removed that freeze (see §3), which is what makes `dynamic` reachable. The issue closed as completed on 2026-09-09; the sheet's `Known op issues` cell still citing it is stale.

---

## 2. For the porter — what to write

Once the checks in the contract pass, the entire instruction is one line, applied to **all four** `TensorParameter`s — `predicate`, `value_true`, `value_false` and the output, unconditionally:

```cpp
.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
```

- **Both flags are load-bearing, not belt-and-braces.** §3 gives a concrete pair of shapes, verified on silicon to share a cache entry, for each one. Dropping either produces a hard throw on a legitimate call, not merely a missed optimisation.
- **Do not set `match_page_size`.** The flag "has no effect" on anything other than an interleaved row-major tensor (`tensor_spec_relaxations.hpp:98-101`), and validity check 2 makes row-major unreachable in any slot, interleaved or sharded. So the flag is *inert*. Note what the argument rests on: an invariant held up by *validation*, not by an absence of row-major kernel code — the kernels can still address row-major on the native-sharded path, and that path is now closed deliberately rather than by accident. Re-check this whenever that validation moves. (Contrast [sparse SDPA](transformer_sdpa.md), which sets the flag because it independently pins its KV width. The general rule — set it iff the op independently pins the last-dim width — never gets a chance to apply.)
- **Do not set `match_padded_shape_only`.** Not just weaker: actively insufficient. The key lets two tensors with different *padded* leading dims share an entry, which is exactly what this flag refuses.
- **There is no conditional and no stop condition.** The declaration is identical for every slot in every regime ternary supports. binary_ng's stop condition is **non-native sharded row-major with a live accessor** ([`eltwise_binary_ng.md`](eltwise_binary_ng.md), row 4) — not, as an earlier revision of this document had it, a mixture of layouts across slots. Ternary cannot reach that regime, for the plain reason that validity check 2 admits no row-major tensor at all.

---

## 3. Why — the derivation

Declarations are per slot, per cache entry, written by the factory at cache **miss** while it holds the actual tensors. The question is always the containment:

> what the hash lets TTNN reuse the program for ⊆ what we declare ⊆ what the program can actually tolerate

### What the key leaves free

Per slot the key pins dtype, memory config, alignment, `Tile`, and the logical `(H, W)` pair — the last via `hw()`, which reads `s[-2]` and `s[-1]` only (`ternary_device_operation.cpp:614-617`). For an interleaved slot it pins **nothing else about the shape**. For a sharded slot the shape in pages (validity check 4) also pins the product of the leading dims, but not their order. Two consequences, each giving one of the two flags its job:

- **Leading dims are free.** The op's own sources give the canonical pair (`ternary_program_factory.cpp:688-694`): a predicate of `[4, 1, 32, 32]` and one of `[1, 4, 32, 32]` hash identically and share one cache entry, while needing different reader strides, because "`n_stride` / `c_stride` are functions of the individual dims, not of their product." Their logical *and* padded shapes differ. Admitting that needs `dynamic_tensor_shape`; `match_padded_shape_only` would reject it. Verified on silicon: the two calls produce one cache entry and both return exact results.

  (That comment describes the key as coarsening "each input tensor to its padded VOLUME," which is stale — the key now pins logical `(H, W)` via `hw()`, which is *tighter* than volume. The example still holds under the current key, and the direction of the staleness is harmless for this analysis, but the wording is worth fixing in the pre-port cleanup.)
- **Rank is free.** `hw()` is rank-tolerant by construction — `s.rank() >= 2 ? s[-2] : 1u` — so `[1, 32, 64]` and `[1, 1, 32, 64]` also collapse to one entry, likewise verified on silicon. `dynamic_tensor_shape` alone still requires constant logical rank, so this is what `relax_logical_rank` is for.

Adding the output's dtype, alignment and tile to the key (prerequisite 3) does not disturb either pair. For a freshly allocated output all three are functions of quantities the key already pins, so both pairs still share one entry; what the new terms split is a *preallocated* output carrying its own alignment.

### Why the program tolerates it

The right-hand containment rests on one property: **nothing that varies with the free dims is baked into the cached program.** Two things could have been, and neither is.

*Per-core work.* Strides and tile counts are functions of the individual dims, so the per-core argument rows genuinely vary across a cache hit. `override_runtime_arguments` re-derives **every** per-core arg from the same shared builder `create_descriptor()` uses, so the two paths are byte-identical by construction (`ternary_program_factory.cpp:1543-1604`), with a `TT_FATAL` on any width divergence. The builder even emits rows for all `num_cores_total` cores, because the work/noop partition itself shifts with the output volume. The op's sources name this exact corruption mode and its fix at `ternary_program_factory.cpp:685-695` — worth reading before trusting this document, since it is the property #55167 established and the reason #54235's verdict no longer applies.

*Accessor configuration.* The framework supplies this for free, and the pre-port code already anticipates it: every slot is declared `tensor_accessor::ArgConfig::RuntimeTensorShape` today (`ternary_program_factory.cpp:1279-1333`), which is the Device-1.0 spelling of what `dynamic_tensor_shape` does — it moves the shape out of compile-time args and into common runtime args. For an **interleaved TILED** tensor the framework leaves accessor configuration untouched, because page size is fixed by dtype and tile dims, both pinned. For a **sharded** tensor it makes shape-in-pages an implicit common runtime argument while holding the rest of the geometry — shard shape in pages, bank count, bank coordinates — fixed at `ProgramSpec` build. Validity check 4 is what makes that hold safe, and the framework *rejects* a geometry divergence rather than mis-addressing it.

### Regime table

Conditioned on the runtime tensor to show the reasoning is regime-complete. All rows land on the same two bools.

| # | Slot | Runtime condition | `dynamic_tensor_shape` | `relax_logical_rank` | `match_page_size` | Confidence |
|---|---|---|---|---|---|---|
| 1 | predicate, value_true, value_false, output | interleaved, TILE | true | true | false (inert) | **High** |
| 2 | same | sharded, TILE, `has_sharding` | true | true | false (inert) | **High** |
| 3 | same | sharded slot, `has_sharding == false` | true | true | false (inert) | Medium |
| — | — | any ROW_MAJOR, sharded or interleaved | *unreachable* — validity check 2 | | | |

**Row 1.** With both bools set the match collapses to bare `tensor_layout` equality, since the shard term is `nullopt` on both sides. The only interleaved-TILE quantity outside per-core RTAs is `aligned_page_size`, which is `f(dtype, Tile)` and so already pinned.

**Row 2.** Safe on two legs. Validity check 3 fixes the addressing mode, and validity check 4 pins the geometry the framework holds fixed.

**Row 3 — read carefully.** `SRC_SHARDED_*` is `(slot_sharded && has_sharding)` (`ternary_program_factory.cpp:206-212`), so a slot can be sharded while its define is `0` — whenever `is_native_L1_sharding` is false, for example for unequal input shapes, differing memory configs, a DRAM slot, a grid mismatch, uneven shards, empty shards (added by #61721) or an interleaved output (`ternary_op_utils.cpp:59-161`, `ternary_program_factory.cpp:319-326`). This is the only regime where a **sharded accessor is live** rather than compiled away, so the geometry term does real work instead of pinning dead code. The safety argument still holds, via validity check 4 plus the framework's reject-not-corrupt behaviour, and it is not contingent on a coincidence. Row 3 is the counterpart of binary_ng's row 4, the live sharded accessor. There the geometry term was safe only by a coincidence of the current key. Here every slot's shape in pages is hashed on purpose (`ternary_device_operation.cpp:560`, `:640-643`), and the output is included. Graded Medium only because it is the thinnest-tested path, not because a leg is missing. `test_ternary_accepts_tile_sharded`'s DRAM-output iteration now takes it on a cache miss.

**The unreachable row.** Row-major is excluded in every slot, so neither clause of `dynamic_tensor_shape` that depends on layout can be entered. Worth knowing *why* it is excluded, because it is not that the kernels cannot do it: row-major was exact on the native-sharded path, where every slot is a borrowed circular buffer and no accessor runs. It was wrong everywhere else, and nothing at the API distinguished the two, so the op was narrowed to TILE rather than asked to tell them apart. That makes this row an invariant of validation, not of the kernels — if row-major is ever readmitted, this analysis needs redoing from the regime table down, not patching.

---

## 4. Not covered

- **Non-32×32 `Tile` support**, as distinct from `Tile` in the key. Carried over from binary_ng's analysis as an open correctness question, not a caching one; not analysed here.
- **The divide-by-zero and pack-format suspicions** are closed at the trigger by #60282, which requires a BF16 output for the LLK row broadcast; both former triggers run correctly on silicon (pre-port audit, 2026-10-22). The kernel hazards remain behind that host-side gate. Neither is a relaxation question.
- **ND-sharded slots.** #59788 sends them down the non-native path, which makes them row 3 slots with a live sharded accessor. Validity check 4 covers them from the code — the 1D distribution strategies take the whole hashed grid as the core list (`buffer_distribution_spec.cpp:168-174`), and the shape in pages is hashed. Probed on silicon (true `ND_SHARDED` inputs, preallocated output): a miss, a hit, and a leading-dims pair sharing one entry, all exact. An ND-sharded `input_a` reaches row 3 reliably only with a preallocated output: without one, `resolve_mem_config_actual` dereferences an empty 2D shard spec (`ternary_device_operation.cpp:198-199`, `:206-218`; `ternary_op_utils.cpp:961`), which is an op bug, not a relaxation question.
- **Whether row-major should be readmitted.** It was removed as a product decision, on the grounds that the only configuration that worked was covered by no test or sweep, and reached by no caller in the repository. A composite that forwards row-major sharded user tensors to `ttnn.where` (for example `prelu`) now raises instead of taking it. Restoring it would mean supporting the native-sharded path specifically, and the relaxation question it reopens — whether a sharded row-major slot's page size can drift under a cache hit — is not analysed here.
- **The port's own mechanics.** Nothing in this document is a porting instruction beyond §2.
- The other `(dynamic - pending analysis)` rows are not analysed here.

---

## 5. Claims that did not survive, recorded so they are not re-derived

Note the first two are both earlier forms of validity check 2. Check 2 now reads "the op is TILE-only", which is the same sentence as the first failed claim — the difference is that it is now enforced per slot rather than merely believed. Read them as a record of how the sentence was made true, not as a contradiction of it.

- **"The op is TILE-only" — as an observation about code that did not enforce it.** Revision 1's check 2, and the audit that rejected it was right. Four readings went wrong the same way, matching on `Layout::TILE` without reading the enclosing scope. `input_a`'s TILE check sat inside `if (!input_a.is_sharded())`; the preallocated output's sat inside the same kind of guard; the all-operand TILE check belongs to SNAKE_BETA alone (`ternary_device_operation.cpp:420`, check at `:440-442`), not to the addc operands; and `compute_output_specs` sets `output_layout = Layout::TILE` only to fall back to `input_a.layout()` two lines later when the memory config is sharded. Most of all, **no check on `input_b` / `input_c` layout existed at all** — the claim was generalised from `input_a`. The sentence was made true later, by adding the checks it had assumed; it was not true when first written.
- **"Interleaved implies TILE, in every slot."** Revision 2's check 2, written after the above was rejected, and false in the same way: generalised from `input_a`'s guard without checking the slots it did not mention. It rested on operands matching `input_a`'s *layout* while nothing compared a slot's layout with its own *sharding*, so a row-major interleaved operand passed whenever `input_a` was row-major sharded. Three further configurations fell with it — a row-major sharded output beside TILE operands, an output whose layout differed from `input_a`'s, and a computed interleaved output over row-major operands — each returning nearly all of its elements wrong (96.6–99.4% across the five probes). The lesson, now twice learned: **for a layout claim, enumerate the slots and state the condition each check is guarded by.** A conditional rule needs exhaustive conditions, which is why the surviving rule has none.
- **"A preallocated output should be guarded on its own sharding rather than `input_a`'s."** True as a criticism of the old guard, false as a replacement for it. Swapping one condition for the other merely moved the hole: `input_a`-guarded admitted a row-major interleaved output, output-guarded admitted a row-major sharded one beside TILE operands. Either guard alone is insufficient, and the pair is unnecessary once an unconditional per-slot check covers both. This one also introduced a regression in the uncommitted working tree, briefly accepting a call `main` rejects.
- **"`match_page_size` is moot because there is no row-major path."** Conclusion now correct, but it was asserted before it was true — twice, while row-major was still reachable. Kept as a reminder that this flag's argument is downstream of check 2 and must be re-derived with it, never carried over.
- **"`relax_logical_rank` is redundant because ttnn normalises rank before reaching the device op."** False at the cache-key level, which is the only level that matters here. `hw()` is explicitly written to tolerate rank < 2, so the key cannot distinguish ranks that produce the same trailing pair, and the declaration is checked against the key's equivalence class rather than against front-end behaviour.
