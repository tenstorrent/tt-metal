# Metal 2.0 — `TensorAccessor` 3rd-argument porting triage

**Author:** Audrey and Claude

**Purpose:** Triage all uses of the `TensorAccessor` 3rd argument (page-size) into **ok (auto-port)** or **"ops team must fix"** buckets.

---

## The 3rd arg

**What does it do?** Probably not what you'd expect. There are two `TensorAccessor` specializations, and they have _different_ 3rd arg behavior:
 - **Sharded** uses the `aligned_page_size` directly to compute byte offsets. An incorrect page size mis-addresses.
 - **Interleaved** strides by `align_power_of_2(page_size, alignment)`, i.e., it silently realigns the passed page size up to the allocator alignment. Any page size of the right order of magnitude will work!

**Metal 2.0** handles the `aligned_page_size` implicitly. No explicit override API is provided. Why? Because no valid use case exists.


## Classification

Every kernel that passes a 3rd argument to `TensorAccessor(args, base_addr, page_size)` falls into one of four classes:


| # | Class | What the 3rd arg is | How to handle it | **Ops-team fix?** |
|---|---|---|---|---|
| **1** | **Dynamic page size** | genuinely *varies* with row width across cache-reused shapes (interleaved row-major) | set `dynamic_tensor_shape` when porting to Metal 2.0 — the page size is then auto-supplied | **No** — it's the feature |
| **2** | **Redundant / inert** | resolves to `buffer->aligned_page_size()` — either already equal, **or** a correct-magnitude value on an *interleaved* accessor (silently realigned, see above), **or** never used (`page_id ≡ 0`) | drop the 3rd arg (pure no-op) when porting to Metal 2.0 | **No** |
| **3** | **Latent bug** | a **wrong-magnitude** value (realignment can't repair it), but masked today (dead path / test config) | drop the 3rd arg | **YES** |
| **4** | **Live bug** | a **wrong-magnitude** (or sharded-verbatim-wrong) value that mis-addresses **in the default config** | analyze the op and determine the appropriate fix | **⚠ YES** |
| **S** | **Special** | a manual override the binding model **cannot express** — a sharded raw-pack page, or a sub-page base offset | stays on manual addressing; **the op owner handles the port** | **eval needed** |

 - **Classes 1 and 2** are fine; handled automatically by the Metal 2.0 porting recipe.
 - **Class 3 and 4** must be fixed by the ops team prior to Metal 2.0 porting. (They cannot be handled automatically, because the porting recipe will _not_ make any functional changes to an op.)
 - **Special** ops can't be auto-ported — the manual override has no binding-token equivalent; the op owner ports them by hand.

---

## Alphabetical lookup — op → class

Use this for the glance test. `⚠` = needs ops-team fix.

| Op | Class | Fix? |
|---|---|---|
| `all_gather_minimal_matmul_async` | 2 — Redundant | no |
| `all_to_all_dispatch_metadata` | 2 — Redundant | no |
| `binary_ng` (RM readers) | 1 — Dynamic page size | no |
| `deepseek_moe_fast_reduce_nc_fused` | 2 — Redundant | no |
| `deepseek_moe_post_combine_reduce` | 2 — Redundant | no |
| `deepseek_prefill/combine` (untilize reader) | 2 — Redundant | no |
| `deepseek_prefill/dispatch` (untilize reader) | 2 — Redundant | no |
| `embeddings_rm_writer_chunked` | 2 — Redundant | no |
| `extract` | 2 — Redundant | no |
| `fill_pad` | 2 — Redundant | no |
| `fill_rm` | 1 — Dynamic page size | no |
| `gated_delta_attn` | 2 — Redundant | no |
| `indexed_fill` | 2 — Redundant | no |
| `indexer_score` | 2 — Redundant | no |
| `insert` | 2 — Redundant | no |
| `isin` | 2 — Redundant | no |
| ~~`matmul` (×2 sites)~~ | ~~2 — Redundant~~ | ✅ DONE — already 2-arg on `main` |
| `minimal_matmul` | 2 — Redundant | no |
| `moe_grouped_topk` | 2 — Redundant | no |
| `moreh_fold` | 1 — Dynamic page size | no |
| `moreh_getitem` | 1 — Dynamic page size | no |
| `msda` (multi-scale deformable attn) | 2 — Redundant | no |
| `non_zero_indices` | **S — Special** | **⚠ → op eval needed** |
| `normalization_ln_rm_gb_post_allgather` | **3 — Latent bug** | **⚠ YES** |
| `padded_slice` | 1 — Dynamic page size | no |
| `point_to_point` | 1 — Dynamic page size | no |
| `quasar/binary_ng` | 1 — Dynamic page size | no |
| `quasar/slice` | 1 — Dynamic page size (+ **S** base-offset) | **no (but eval needed)** |
| `rotary_embedding` | 2 — Redundant | no |
| `rotary_embedding_hf` | 2 — Redundant | no |
| `sdpa` (page-table accessor) | 2 — Redundant | no |
| `sdpa_decode` (page-table) | 2 — Redundant | no |
| `slice` (interleaved RM path) | 1 — Dynamic page size (+ **S** base-offset) | **no (but eval needed)** |
| `slice_reshard_async` | 1 — Dynamic page size | no |
| `slice_write` | 1 — Dynamic page size | no |
| `topk_large_indices` (input reader) | 1 — Dynamic page size | no |
| `topk_router_gpt` | 2 — Redundant | no |
| `unified_routed_expert_ffn` | 2 — Redundant | no |
| `zero_padded_kv_cache` | 2 — Redundant | no |

---

## Class detail + complete op lists

### Class 1 — Dynamic page size (relaxation customer) · *no fix needed*

Interleaved row-major ops whose page size (`last_dim_width × element_size`) genuinely varies across shapes that reuse one compiled program. **These are the customers the page-size relaxation exists for** — port them by setting `dynamic_tensor_shape` on the interleaved-RM tensor parameter and dropping the manual override. (See the relaxation design doc.)

`binary_ng` (RM), `slice` (interleaved), `padded_slice`, `slice_write`, `moreh_getitem`, `moreh_fold`, `fill_rm`, `topk_large_indices` (input), `point_to_point` †, `slice_reshard_async` †, `quasar/binary_ng` ‖, `quasar/slice` §

- These carry **all** dimension extents as runtime args, so `dynamic_tensor_shape`'s all-dims relaxation is exactly what they want (no coarseness hazard).
- `topk_large_indices` also needs its accessor args migrated off the buffer-less `create_*_interleaved()` helpers (its compile-time `AlignedPageSize` is otherwise 0).
- † `point_to_point` is layout-agnostic and *shape-blind* (copies whole pages by `page_id`); `slice_reshard_async` is a CCL op slicing the **outer** dim (page constant across the slice). Both pass a runtime page and are robust — neither is a problem, but neither is a "page-size-tracks-width" case in the strict sense.
- ‖ `quasar/binary_ng` is a near-verbatim copy of `binary_ng`'s RM readers, but still written against the **legacy** accessor form (`TensorAccessorArgs<0>()` + positional runtime args), *not* Metal 2.0 binding tokens — i.e. not yet binding-ported despite living in the `quasar` tree. (Mainline `binary_ng` is equally unported.)
- § `quasar/slice` *is* binding-ported, but like mainline `slice` it also does sub-page **base-offset** addressing → see **Special**. Its page-size arg alone is Class 1 (dynamic for the sharded sub-case, inert for interleaved).

### Class 2 — Redundant / inert · *no fix needed*

Dropping the 3rd arg is a pure no-op. Three ways an override lands here:
- **(a) Already equals the aligned page.** The value is `tt::tile_size` / `get_tile_size(cb)` (TILE pinned 32×32, block-float-safe → bf8 = 1088 B) or the host `aligned_page_size()` directly.
- **(b) Correct magnitude on an *interleaved* accessor.** The value is the true logical page (`buffer->page_size()`) but *unaligned* — harmless, because the interleaved accessor silently realigns it up to the allocator alignment (see [The 3rd arg](#the-3rd-arg)). **This is what dissolved the bulk of the former "bug" flags.**
- **(c) Never used.** `page_id ≡ 0`, so the stride never multiplies a nonzero page index.

`msda`, `extract`, `insert`, `rotary_embedding`, `rotary_embedding_hf`, `moe_grouped_topk`, `unified_routed_expert_ffn`, `deepseek_prefill/combine` & `/dispatch` (untilize readers), `fill_pad`, `zero_padded_kv_cache`, `gated_delta_attn`, `minimal_matmul`, `all_gather_minimal_matmul_async`, `topk_router_gpt` ‡, `sdpa_decode` page-table ‡, `sdpa` page-table, `all_to_all_dispatch_metadata`, `embeddings_rm_writer_chunked`, `isin` ¶, `indexed_fill` ¶, `indexer_score`, `deepseek_moe_post_combine_reduce`, `deepseek_moe_fast_reduce_nc_fused`

- **(a)-type**: most pass `tt::tile_size`/`get_tile_size(cb)` (block-float-safe) or `aligned_page_size()` directly. `moe_grouped_topk` passes a raw `buffer->page_size()`, but every permitted dtype is TILE → a 64-aligned page.
- **(b)-type** — interleaved, correct-magnitude-but-unaligned, realigned by the addrgen: `all_to_all_dispatch_metadata` (`buffer->page_size()`), `embeddings_rm_writer_chunked`, `indexer_score` writer, and **the two deepseek MoE ops earlier sweeps flagged as Class-4 live bugs** — `deepseek_moe_post_combine_reduce`, `deepseek_moe_fast_reduce_nc_fused` — both pass `buffer->page_size()` (raw 16 B) on interleaved DRAM, which realigns to the true 64 B stride. **Not bugs;** their tests pass because there is no mis-addressing.
- ¶ `isin` and `indexed_fill` are additionally **(c)-type** — `page_id ≡ 0`.
- ‡ `topk_router_gpt` and `sdpa_decode` page-table pass a **runtime** arg whose value is **constant per compiled program** (width hashed / pinned by `B==32`), so they *look* like Class 1 but aren't. Still a no-op drop. Do **not** set `dynamic_tensor_shape` — their leading geometry is compile-time-pinned. (`sdpa`'s page-table, on the flexible branch, passes `max_blocks_per_seq*4` = the true page magnitude → realigned → inert.)
- `fill_pad` and `zero_padded_kv_cache` were **previously mis-flagged as bugs**; they use `tt::tile_size` (correct for block-float), so they are redundant.
- ~~`matmul`~~ — **already fixed on `main`**: every `matmul` `TensorAccessor` is now 2-arg (the override was dropped). Nothing to do.
- **No 3rd arg at all** (flagged by earlier sweeps, confirmed clean — *not* table rows, since the table is for ops that pass a 3rd arg): the main `data_movement/untilize` op, and `sdpa`'s **Q** accessor (though `sdpa` still appears above — its *page-table* accessor does pass one).

### Class 3 — Latent bug · *fix needed*

A **wrong-magnitude** value — one the interleaved realignment can *not* repair, because it differs from the true page by more than alignment rounding. It mis-addresses wherever it's used, but is masked today by a dead path or the default test config. **Exactly one op remains here** after the realignment mechanism dissolved the rest:

| Op | The bug, and why it doesn't bite today |
|---|---|
| `normalization_ln_rm_gb_post_allgather` | passes `element_size()*1024` on the TILE-layout block-float gamma/beta branch → **1024 B, dropping the bf8 exponent (true page 1088 B)**; used verbatim, so realignment can't save it. Reachable only with **bf8-TILE** gamma/beta — which validation permits but the docstring forbids; the default test passes gamma/beta as ROW_MAJOR bf16 (a different, correct branch). *Masked by test config.* (`layernorm_post_all_gather_program_factory.cpp:223,232`) |

*Formerly here, now Class 2:* `isin`, `embeddings_rm_writer_chunked`, `all_to_all_dispatch_metadata` — all interleaved + correct-magnitude, so their overrides are inert. (`all_to_all`'s old rationale — "hidden_size×2 = 14336, happens to be 64-aligned" — was doubly moot: the value is `buffer->page_size()` fetched at runtime, and the realignment makes alignment irrelevant.)

### Class 4 — Live bug · *⚠ ops-team fix needed*

**None on current `main`.** All three former entries dissolved once the interleaved-realignment mechanism was understood:
- `deepseek_moe_post_combine_reduce`, `deepseek_moe_fast_reduce_nc_fused` → **Class 2.** Both are interleaved with a correct-magnitude `buffer->page_size()` override; the interleaved addrgen realigns it to the true stride. The tests passed because there is **no** mis-addressing — not because of weak validation. (This retires the earlier "divergence reachable, detection unconfirmed" caveat.)
- `non_zero_indices` → **Special** (below), and in any case **already fixed on `main`** (#44572, merged Jun 11 — *predates* the original audit). The earlier "Borys-confirmed live bug" described the pre-fix code.

### Special — manual override the binding model can't express · *→ op owner*

Not bugs, but they can't be trivially ported: the manual override is load-bearing and has **no Metal 2.0 binding-token equivalent**. The op owner handles the port (the op stays on manual addressing).

| Op | Why it's special |
|---|---|
| `non_zero_indices` | **sharded raw-pack.** Correct on `main` (branches `last_dim × NUM_BYTES` for sharded vs `aligned_page_size` for interleaved), but the sharded path's raw override is **load-bearing** — the binding token supplies `aligned_page_size`, which overshoots the no-padding sharded L1 packing, so dropping it on the sharded path breaks addressing. Also gated on the unresolved **sharded-L1-padding** question. |
| `slice` / `quasar/slice` | **sub-page base offset.** Both do byte-granular base addressing (`base_addr + addr_offset`, op-semantic slice bounds) that the binding model can't express — a *2nd-arg* concern, separate from the page-size 3rd arg (which is Class 1 / inert). Stays on raw addressing. |

---

## How to classify a new op (if it isn't in the table)

For each accessor that passes a 3rd page-size argument, ask **two** questions — the classification hinges on them (see [The 3rd arg](#the-3rd-arg)):

1. **Sharded or interleaved?** (`src_args.is_sharded`, or trace the tensor's memory config.)
   - **Interleaved:** the accessor silently realigns the passed value up to the allocator alignment. Only the *magnitude* matters — an unaligned but correct-magnitude value is **inert**.
   - **Sharded:** the passed value is used **verbatim** as the stride. Any wrong value mis-addresses.
2. **Correct or wrong magnitude?** Resolve the expression to a host value (trace the CTA/RTA to the program factory) and compare to the true logical page (`buffer->page_size()`):
   - `tt::tile_size` / `get_tile_size(cb)`, `buffer->page_size()`, and `aligned_page_size()` are **correct magnitude**.
   - `element_size()*1024` (drops the block-float exponent — bf8 gives 1024, not 1088), a stale/hardcoded constant, or a sub-page fragment are **wrong magnitude**.

**Decision:**
- **Varies with row width across cache-reused shapes** (interleaved RM) → **Class 1** (set `dynamic_tensor_shape`; the runtime override is load-bearing across cache hits — dropping it strands a *stale* page, a wrong magnitude the realignment won't fix).
- **Interleaved + correct magnitude**, *or* value already `== aligned_page_size()`, *or* `page_id ≡ 0` → **Class 2** (drop, no-op). *Includes runtime args whose value is constant per program — hashed/pinned width.*
- **Wrong magnitude** (or sharded + wrong-for-packing), but masked by dead path / `page_id ≡ 0` / default test config → **Class 3**.
- **Wrong magnitude** (or sharded verbatim-wrong) and reachable in the default config → **Class 4**.
- **Manual override with no binding-token equivalent** (sharded raw-pack page, or sub-page base offset) → **Special** (→ op owner).

**Notes:** Evaluate alignment on **Blackhole/Quasar DRAM (64)** — the strictest, and this hardware's target. A **sharded** accessor never realigns, so it's the one place a raw override actually bites. Watch for args built from buffer-less `create_*_interleaved()` helpers (→ compile-time `AlignedPageSize` is 0).
