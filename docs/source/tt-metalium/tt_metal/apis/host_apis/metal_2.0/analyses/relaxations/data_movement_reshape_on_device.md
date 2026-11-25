# TensorParameter relaxations — `data_movement/reshape_on_device`

**Author:** Anasuya and Claude

**Purpose:** State the Metal 2.0 `TensorSpecRelaxations` declaration this op requires, per `TensorParameter`, so that neither the auditor nor the porter has to derive it. The readiness sheet's `TensorParameter relaxation` cell points here.

**Covers two sheet rows:** `data_movement/reshape_on_device` · `ReshapeDeviceOperation` · `ReshapeRMProgramFactory`, and the same op with `ReshapeTileProgramFactory`. The declaration is the same for both.

---

## Contract — read this before using the document

This is **not** the same contract as the [offset-base-pointer](../2026-07-19_offset_base_pointers.md) and [3rd-argument](../2026-07-06_tensor_accessor_3rd_arg_triage.md) triage docs. Those are *priors* layered on a scan the auditor runs anyway, so a disagreement means "the doc is stale, trust your own scan."

Here there is no scan to fall back on: the audit recipe forbids re-deriving a relaxation. So this document is **authoritative but perishable**.

> **If any validity check fails, do not substitute your own judgement. Stop, and report the relaxation verdict as UNCONFIRMED.**

### Validity checks — confirm all six before applying anything below

Each is one read or one grep. A commit stamp is deliberately not used. Paths are under `ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/`.

1. **The cache key pins the input exactly as the declaration matches it.** Read the body of `compute_program_hash` in `reshape_op.cpp`. Today it keys the input with `tt::tt_metal::hash_tensorspec_with_relaxation(input_tensor.tensor_spec(), {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true})` (#63798). Check the property, not the implementation: **no two inputs that differ in padded shape, dtype, page config (*including* `Tile`) or memory config may share a cache entry.** A stricter key (for example, the whole `TensorSpec`) still passes; it only splits the cache more finely. **If any of those terms can diverge within one entry, STOP.**
2. **The output spec is a pure function of the key.** `compute_output_specs` builds the output from `logical_output_shape`, `padded_output_shape` and `output_mem_config`, which the key hashes, plus the input's dtype and page config, which check 1 pins. The op takes no preallocated output (`ReshapeOnDeviceInputs` holds only `input_tensor`). This is why the output slot stays strict.
3. **Neither factory nor any kernel reads the input's logical shape or its `TensorLayout` alignment.** The tile factory's `alignment` is `hal::get_dram_alignment()` or `hal::get_l1_alignment()`, chosen by buffer type, and is fine. So is the tile reader's `ALIGNMENT`, which is that same value as a compile-time arg.
4. **The op is still the base concept:** no `override_runtime_arguments`, no `get_dynamic_runtime_args` (after the port: `ProgramSpecFactoryConcept`). A cache hit refreshes only tensor bindings. Every per-core runtime arg is baked at the cache miss from padded shapes (§3). This is what rules out `dynamic_tensor_shape`.
5. **Validation reads only pinned terms and runs on cache hits.** `validate_on_program_cache_miss` reads storage, dtype, layout, both memory layouts, the tile's dimensions, the input's physical (padded) volume, the output's padded volume and the padded widths. Nothing in it reads the input's logical shape or alignment, so its outcome cannot differ within one cache entry. The op declares no `validate_on_program_cache_hit`, so the framework runs the miss validator on every hit.
6. **Both factories are still interleaved-only and chosen by the input's layout.** Validation rejects a sharded input and a sharded output, so no shard-geometry term is needed. `select_program_factory` picks by `input.layout()`, which the key pins through the page config (and `program_factory.index()` is hashed as well).

*Provenance, not a gate:* analysed against `main` at `d1a10279368` (2026-11-21), which includes #64663 (`e8b68619482`) and #63798 (`fa826ca2a9a`).

---

## 1. For the auditor

**Relaxation verdict: `legality`.** The declaration is in §2; carry it into the brief. This clears the relaxation conjunct only. The op's other gate conjuncts are the sheet's to answer, as usual.

Why `legality` and not `none`: the key deliberately leaves the input's logical shape and `Alignment` free, so one cache entry legitimately serves inputs whose `TensorSpec`s differ in exactly those two terms. Under the default exact match, the second such call would throw on the cache hit. The declaration makes those calls legal. Nothing about the shape is dynamic: the padded shape is pinned, so this is not a `dynamic` verdict.

No pre-port fix is owed. #63798 already brought the key into this form.

The sheet's cell currently reads `Padded Shape + Alignment`, which is not in the audit's vocabulary. Once this document lands, the cell should read `legality (see analysis)`.

### The rule behind validity check 1

> **An op's `compute_program_hash` must pin at least everything `tensorspecs_match_with_relaxation` requires to be exactly equal under the declared relaxation.** Under `{match_padded_shape_only, relax_tensor_layout_alignment}` that is the slot's `tensor_layout` *except* `Alignment` (dtype, page config including `Tile`, memory config), plus its padded shape.
>
> Where the key is looser than the declaration, you get spurious throws. Where the key is looser than what the factory *bakes*, you get silent corruption. The declaration can only ever fix the second.

---

## 2. For the porter — what to write

Once the checks in the contract pass, the instruction is two lines.

On the **input** `TensorParameter`:

```cpp
.relaxations = {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true},
```

On the **output** `TensorParameter`: no relaxation (leave `relaxations` at its default).

- **Do not** leave the input at `{}`. It throws on the second call in a cache entry (§1).
- **Do not** set `match_padded_shape_only` alone on the input. Plain `match_padded_shape_only` still requires equal `Alignment`, and the key does not pin it.
- **Do not** set `dynamic_tensor_shape`. It frees the padded shape, and both factories bake padded-shape-derived runtime args that a base-concept cache hit never refreshes (validity check 4). A different padded shape would leave them stale: silent corruption, not a throw.
- **Do not** copy the input's flags onto the output. Its spec is a function of the key (validity check 2), so strict is correct.
- **There is no conditional.** The declaration does not depend on the factory, the layout, the dtype or the buffer type. The regime analysis in §3 is *audit* material; it does not branch the code you write.

---

## 3. Why — the derivation

Per `tensor_spec_relaxations.hpp`, `match_padded_shape_only` frees the logical shape and requires the padded shape to match exactly. `relax_tensor_layout_alignment` additionally frees the layout's `Alignment`. Both leave the `TensorAccessor` configuration unchanged: alignment reaches the accessor only through the padded shape and the row-major page size, and `match_padded_shape_only` pins both.

**Input slot:**

| Term | Pinned by | Status |
|---|---|---|
| logical shape | nothing — freed by the declaration | free by design |
| `Alignment` | nothing — freed by the declaration | free by design |
| padded shape | the key | pinned |
| dtype | the key | pinned |
| page config (layout, `Tile`) | the key; validation also rejects any tile but 32×32 | pinned |
| memory config | the key; validation also requires INTERLEAVED | pinned |

The key computes this term with the framework's own `hash_tensorspec_with_relaxation`, using the same flags as the declaration. The key and the declaration therefore admit exactly the same set of inputs, by construction.

**Output slot:** every term of its spec comes from `logical_output_shape`, `padded_output_shape` and `output_mem_config` (all hashed) and from the input's dtype and page config (pinned above). Its alignment is derived by `tensor_layout_from_padded_shape` from the two hashed output shapes, so it is also fixed within an entry. Nothing is free, so nothing needs relaxing.

Everything the factories turn into compile-time or runtime args is a function of pinned terms:

- **Row-major factory:** old and new stick sizes from the padded widths (`reshape_rm_program_factory.cpp:44-45`), old and new stick counts from the padded shapes (`:40-42`), the work split by whichever stick is larger (`:61-65`), the per-core counts and stick-size ratio (`:134-139`), and the CB size (`:68-78`).
- **Tile factory:** `num_tiles` from the output's physical volume (`reshape_tile_program_factory.cpp:39`), and the reader's runtime args `in_Wt`, `out_N`, `out_C`, `out_Ht`, `out_Wt` from the input's and output's padded shapes (`:95-102`). The `alignment` compile-time arg and the conditional scratch CB depend on the buffer type and the architecture (`:61-71`), which the memory config pins.
Two inputs that agree on the pinned terms therefore produce byte-identical programs. The freed terms only say which part of the same padded storage holds logical data.

### Reachability

- **TILE:** easy to reach.
- **ROW_MAJOR:** no Python route to an RM input whose logical shape or `Alignment` differs at the same padded shape was found. On this path the declaration is therefore inert, and harmless.

### Regime matrix

The declaration is the same in every regime (§2). The rows differ only in how well the reasoning is tested.

| # | Regime | Confidence |
|---|---|---|
| 1 | TILE, interleaved DRAM | **High** — exercised by #63798's test |
| 2 | TILE, interleaved L1, or DRAM with the wide-alignment scratch CB (Blackhole) | **High** — same arithmetic as row 1; the extra terms are pinned by the memory config |
| 3 | ROW_MAJOR, interleaved DRAM or L1 | Medium — reasoned from the row-major arithmetic only; no test, and no Python route to such a pair (see Reachability) |

---

### Claims that did not survive, recorded so they are not re-derived

- **"The tile is absent from `compute_program_hash`"** (the comment at `reshape_op.cpp:51-53`). True before #63798. The key now hashes the input's page config, and a TILE page config carries its `Tile`. The comment is stale; the 32×32 rejection under it is still correct and still runs on every call.
- **"No flag can free `Alignment`, so the key fix must hash the whole input spec and the verdict is `none`."** #59915 framed the fix this way (the cache-hit check "requires exact `tensor_layout` equality"), and #63798's first commit did exactly that (`76b17270e9da`, "Hash whole TensorSpec in roll, reshape_on_device and interleaved_to_sharded_partial"). It was overtaken by #64663 (2026-11-17), which extended `relax_tensor_layout_alignment` to `match_padded_shape_only`, and by #63798's final key, which uses exactly that pair.
- **"`match_padded_shape_only` alone on the input."** This was an intermediate commit on #63798 (`77f124a3c03c`); a later commit freed `Alignment` (`89abbab50e4a`). The merged key frees it, so the flag alone would throw on #63798's own test pair.

---

## 4. Not covered

- **The op's reshape semantics.** The tile path reshapes the padded storage, so for an input whose logical shape is not tile-aligned the result differs from `torch.reshape` of the logical data. #63798's test says so ("a logical torch reshape is not a valid reference"). That is the same on a cache miss as on a hit, so it does not affect the declaration. Issue #15721 proposes removing the op.
- **Edwin's other #63798 ops** (roll, interleaved_to_sharded_partial). They use the same pair of flags on their input slot, but are **not** analysed here.
- **The port's own mechanics.** Nothing in this document is a porting instruction beyond §2.
