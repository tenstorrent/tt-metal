# TensorParameter relaxations — `eltwise/binary_ng`

**Author:** Audrey and Claude

**Purpose:** State the Metal 2.0 `TensorSpecRelaxations` declaration this op requires, per `TensorParameter`, so that neither the auditor nor the porter has to derive it. The readiness sheet's `TensorParameter relaxation` cell points here.

**Covers one sheet row:** `eltwise/binary_ng` · `BinaryNgDeviceOperation` · `ProgramFactory`.

---

## Contract — read this before using the document

This is **not** the same contract as the [offset-base-pointer](../2026-07-19_offset_base_pointers.md) and [3rd-argument](../2026-07-06_tensor_accessor_3rd_arg_triage.md) triage docs. Those are *priors* layered on a scan the auditor runs anyway, so a disagreement means "the doc is stale, trust your own scan."

Here there is no scan to fall back on: the audit recipe forbids re-deriving a relaxation, because deriving one is the expert work the sheet column exists to record. So this document is **authoritative but perishable**.

> **If any validity check fails, do not substitute your own judgement. Stop, and report the relaxation verdict as UNCONFIRMED.**

### Validity checks — confirm all four before applying anything below

Each is one grep. A commit stamp is deliberately not used: this op's sources move, and a stamp fires on every unrelated commit until its reader learns to ignore it.

1. **The cache key pins `tensor_layout`.** Read `tensor_args_t::to_hash()` and `attribute_values()` in `binary_ng_device_operation.{cpp,hpp}`: **no two tensors differing in dtype, page config (*including* `Tile`), memory config, or `Alignment` may share a cache entry.** Check that property, not any particular implementation of it — hashing the field, normalising it, or rejecting the divergent domain all satisfy it equally. **If two such tensors can still collide, STOP** and report UNCONFIRMED: every relaxation requires exact `tensor_layout` equality and no flag reaches inside it, so a declaration against a key that does not pin it turns a working `ttnn.add` into a hard `TT_FATAL`.
2. **The tiled dataflow kernels still `#if`-compile the accessor away when the slot is sharded.** `SRC_SHARDED` / `SRC_SHARDED_B` / `DST_SHARDED`, set from each slot's own `memory_config().is_sharded()`. This is what makes the sharded rows safe — the sharded accessor CTA payload is emitted by the host and never read by the device.
3. **The row-major readers still pass an explicit page size as the accessor's third constructor argument**, fed from a per-core RTA that `override_runtime_arguments` refreshes every dispatch (`kernels_ng/dataflow/reader_interleaved_rm_*`, `writer_interleaved_rm_no_bcast`). This is what makes row-major safe *without* `match_page_size`.
4. **The op still has exactly one factory** (`std::variant<ProgramFactory>`, no `select_program_factory`), so the declaration is unconditional across factory choice.

*Provenance, not a gate:* analysed at `28994778430`, whose `binary_ng` tree is byte-identical to `origin/main` and includes `cddcf95ca35` (#54233).

---

## 1. For the auditor

**Relaxation verdict: `dynamic`.** The declaration is in §2; carry it into the brief. This clears the relaxation conjunct only — the op's other gate conjuncts are the sheet's to answer, as usual.

### The rule behind validity check 1

> **An op's `compute_program_hash` must pin at least everything `tensorspecs_match_with_relaxation` requires to be exactly equal** — the whole `tensor_layout` (dtype, page config *including* `Tile`, memory config, alignment) per slot, plus the sharded distribution geometry for sharded slots.
>
> Where the key is looser than the declaration, you get spurious throws. Where the key is looser than what the factory *bakes*, you get silent corruption. The declaration can only ever fix the second.

---

## 2. For the porter — what to write

Once the checks in the contract pass, the entire instruction is one line, applied to **every** `TensorParameter` — both inputs and the output, unconditionally:

```cpp
.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
```

- **Do not** set `match_page_size`. binary_ng's last-dim width genuinely varies, and the row-major kernels already override the accessor's page size with a per-core RTA refreshed every dispatch. (Contrast [sparse SDPA](transformer_sdpa.md), which *does* set it, because that op independently pins its KV width. The rule: **set `match_page_size` iff the op independently pins the last-dim width.**)
- **Do not** set `match_padded_shape_only`. It is strictly weaker than what `dynamic_tensor_shape` buys here and pins nothing this op needs.
- **There is no conditional.** The declaration is identical for every slot in every layout regime. The regime analysis in §3 is *audit* material; it does not branch the code you write.

> **One stop condition.** If you find yourself porting a configuration that is **sharded *and* row-major** — see row 4 below — **stop and ask**. It is the one regime where the analysis is not confident, no caller was found that reaches it, and it is the only row where the declaration's geometry term does real work rather than pinning dead code.

---

## 3. Why — the derivation

Declarations are per slot, per cache entry, written by the factory at cache **miss** while it holds the actual tensors. The rows below are conditioned on the runtime tensor to show the reasoning is regime-complete; they all land on the same two bools.

| # | Slot | Runtime condition | `dynamic_tensor_shape` | `relax_logical_rank` | `match_page_size` | Confidence |
|---|---|---|---|---|---|---|
| 1 | a, b, c | interleaved, TILE | true | true | false | **High** |
| 2 | a, b, c | interleaved, ROW_MAJOR | true | true | false | Medium |
| 3 | a, b, c | sharded, TILE | true | true | false | Medium |
| 4 | a, b, c | sharded, ROW_MAJOR | true | true | false | **Low — stop and ask** |

**Row 1.** With both bools set, the match collapses to bare `tensor_layout` equality — the shard term is `nullopt` on both sides for an interleaved slot. The only interleaved-TILE quantity outside per-core RTAs is `aligned_page_size`, which is `f(dtype, Tile)` and therefore already pinned by `tensor_layout`.

**Row 2.** Safe because of validity check 3. The hedge: that is a property of the current kernel sources, not of the framework. A ported row-major kernel that drops the third argument would need `match_page_size = true` — which in turn needs a page-size term in the key, or it throws.

**Row 3.** Safe on two legs. The stronger is validity check 2 — the sharded accessor is dead code on the tiled path. The second is that `has_sharding` cannot flip on a cache hit, because `c_shard_volume.has_value() == has_sharding` exactly and `c_shard_volume` is a hashed attribute.

**Row 4 — the row not to trust.** It is the intersection of two exotic paths: non-native-L1 sharding with all layouts row-major. It is the **only** row where the accessors are live *and* sharded, so the geometry term does real work. No caller was found that reaches it — `ttnn.add`'s front end requires both inputs non-sharded for its row-major path and otherwise converts to TILE — so it may be dead in practice or prim-reachable and under-tested. The safety argument is also *contingent*: today the case is safe only because `to_hash()` happens to hash `sharded_tensor_shape_in_pages`, which is a coincidence of the current key rather than a design invariant, and it breaks if anyone drops that term as a perf win.

### Two claims that did not survive, recorded so they are not re-derived

- **"Freeing the shape flips `is_native_L1_sharding` via `is_uneven(padded_shape)`."** Not a cache defect. The pairs that flip it differ in `sharded_tensor_shape_in_pages`, which the key hashes, so they land in different cache entries and nothing is stale.
- **"A BLOCK_SHARDED pair with identical squeezed shape-in-pages but different `cores()` corrupts today."** The first two steps hold — `[1,1,128,128]` and `[1,1,64,256]` with shard `{32,64}` do collapse to the same squeezed shape while `GRID_2D` trims different core grids from the *unsqueezed* shape, and the key does not separate them. But the pair is sharded, so per validity check 2 the accessor carrying the stale bank list does not exist in the compiled kernel. Not a live bug. The row-major resurrection is closed too: row-major block/width-sharded page shape is `(1, physical_shard_width)`, so the squeeze's merge condition cannot fire while `shard_h > 1`, and `shard_h == 1` is `TT_FATAL`'d against any real grid.

---

## 4. Not covered

- **`Tile` support**, as distinct from `Tile` in the key: `binary_ng_program_factory.cpp` sizes CBs with `tt::tile_size(data_format)` — a function of the data format alone, ignoring the tensor's `Tile` — while `get_shard_volumes` *does* read `c.tile()`. That reads as "binary_ng does not support non-32×32 tiles at all," a correctness question rather than a caching one, and `ttnn.from_torch(..., tile=...)` makes it reachable. Not analysed here.
- The other `(dynamic - pending analysis)` rows — `eltwise/ternary` and `eltwise/unary` — are **not** analysed here.
- The port's own mechanics. Nothing in this document is a porting instruction beyond §2.
