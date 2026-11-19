# Full-model physical capacity accounting

Source-only audit, 2026-11-14. No hardware run, allocation attempt, context reduction or new supported-capacity claim is made. This note derives the future serving budget from the current model/generator/helper allocation contract and identifies measurements still required. It is not a vLLM implementation.

**Keep the per-request context contract at262144.** Batch capacity32 does not mean32 simultaneous full-context requests. The default dense B32 allocation would require68GiB/device for K/V alone and cannot fit the measured approximately32GiB/device DRAM. B32 therefore requires an explicitly bounded physical pool, mapped through full-width virtual per-slot tables. A scheduler can admit/preempt requests according to that aggregate pool budget without reducing the maximum length of an individual request.

## Quantities and accounting convention

All byte values below are **per device/rank**, not sums across the four-device mesh. The selected TP4 lane has16 full-attention layers,48 GDN layers, one local KV head of dimension256,32 tokens per KV page, BF8_B K/V, BF16 published GDN state, and FP32 live-prefill recurrence. Vocabulary is248320 globally /62080 per rank; hidden ownership is1280 per rank.

Let:

- `B` be configured fixed slots,1..32.
- `C` be the requested virtual per-slot context, at most262144.
- `T = max(16, 16*ceil(C/512))` be table columns per slot; at full context `T=8192`.
- `P` be physical pages per full-attention layer/rank. Uniform pool shapes across all16 layers are assumed in the main formulas.
- `H(B) = {1}` for B1, otherwise `{1,B}`, the two bounded-prefill physical batch sizes.
- `A(shape,dtype,layout)` be eight-bank DRAM consumption, including bank/page alignment, rather than logical element payload.

The existing CPU-only `tests/optimized_multichip_capacity.py::tensor_bytes` provides the selected machine's allocation model: `A = 8*ceil(n_pages/8)*64*ceil(page_bytes/64)`. TILE pages are32x32: BF16=2048 bytes, FP32/int32/uint32=4096, BF8_B=1088. RM pages are full last-dimension rows; page bytes are row width times element size. For example BF16 RM `[1,3,2560]` has only15360 logical payload bytes but reserves40960 across eight banks. This distinction materially affects the1024 short ID/position tensors in each helper.

The C++ derivation is `tt_metal/impl/tensor/spec/layout/{tensor_layout,page_config}.cpp`: `compute_consumed_memory_bytes_per_bank` rounds the number of pages per bank and page alignment. The formula is a source allocation model, not a measurement of allocator fragmentation, program buffers or peak live temporaries. Existing DRAM-sharded weight allocation is taken from the measured loaded-model baseline rather than approximated with this interleaved formula.

## Loaded weights: use the measurement once

`doc/context_contract.json:full_model.loaded_model_dram_measurement` records:

| Quantity | Bytes/device |
| --- | ---: |
| Available general DRAM:8 banks x4259840384 | 34078723072 |
| Loaded all64 model:8 x1732213760 | 13857710080 |
| Free after loading:8 x2527626624 | 20221012992 |
| Largest sampled free block per bank | 2423779200 |

Evidence is `doc/full_model/teacher_forcing_first_all64.log`. This baseline includes all distinct real layer weights, embedding, LM head, shared RoPE and setup-owned auxiliary resources, before cache/generator/bounded-prefill buffers. It is a sampled resident observation, not a high-water mark. Its provenance must be rechecked if weights/native allocation policy change. Largest-free-block bytes are per bank; aggregate free bytes alone do not establish feasibility of every later allocation.

`capacity_plan.json` retains the older source inventory: matrices12344360960, future embedding/head/norm1275985920, auxiliary177758208, persistent CCL69206016, plus B1 state/cache and a4GiB reserve. That plan is not an additional allocation to add to the measured baseline. Its “future” head estimate and one-signature CCL bound do not describe every current helper/preparation allocation. Shared RoPE is67108864 bytes and already included in the measured loaded baseline; do not add it a second time. The full-model `measurement_scope` string in the context contract still says all64 measurement pending despite the nested all64 measurement; reconcile that stale description separately.

## Physical KV pool and reset-zero storage

One K or V page per rank is `[1,1,32,256]`:8 BF8_B tiles x1088 =8704 bytes. For uniform physical pool sizeP:

```
KV(P)       = 16 layers * 2 tensors * 8704 * P = 278528 * P
KV_zero(P)  = 8704 * P
KV_total(P) = 287232 * P
```

The zero is **one shared zero source**, not32 zeros: `generator.reset_zeros` deduplicates by `(shape,dtype,layout)`, and every K/V tensor has the same signature. The32 distinct K/V targets remain allocated. If layers have different physical extents, use `sum_l 2*A([P_l,1,32,256],BF8)` plus one `A` for each distinct zero signature. Aliasing physical storage between different layers is not allowed.

For a pool representing262144 cached tokens, `P=8192`: K/V=2281701376 bytes (2.125GiB), zero=71303168, total2353004544. Every32-token physical block added to the pool costs287232 bytes/device under this reset strategy; this is8976 bytes per aggregate cached-token slot, including the shared reset zero. The physical IDs are shared **as metadata across attention layers**, but each layer has its own tensor storage.

With default dense tables `P=B*T`. At B32/C262144, K/V=73014444032 bytes (68GiB), reset zero=2281701376. Those two terms already exceed the device budget. `max_seq_len=262144` does not force dense allocation when explicit external `kv_cache` and a bounded host `page_table` are supplied. A deliberately bounded host mapping can also guide `model.create_cache` when the generator allocates internally: that constructor sizes by `max(page_id)+1`, not virtual table element count. In either case the physical pool must be explicitly planned; the default sequential B*T mapping is unsuitable.

## Fixed B/C storage outside the physical pool

The table gives current source-derived allocator bytes for full virtual context C262144. These are required resident categories, not the complete process total. `uint32` tile size4096 was included when evaluating the existing CPU allocation formula; no TTNN import/device access was needed.

| Category and formula | B1 bytes | B32 bytes |
| --- | ---: | ---: |
| Published GDN: `48*(A([B,12,128,128],BF16)+A([B,3,2560],BF16,RM))` | 20840448 | 627572736 |
| Deduplicated GDN reset sources: one recurrence and one convolution shape | 434176 | 13074432 |
| FP32 live helper recurrence + BF16 helper convolution: sum over `h in H(B)` of `48*(A([h,12,128,128],FP32)+A([h,3,2560],BF16,RM))` | 39714816 | 1271267328 |
| Helper ID and RoPE-position banks: `sum_h 2*sum_(n=1..512) A([h,n],uint32,RM)` | 8650752 | 43253760 |
| Helper runtime row-gather indices: `sum_h A([h,1,1280],uint32,TILE)` | 163840 | 5406720 |
| Helper full/fill tables: `sum_h (A([h,T],int32,RM)+A([h,16],int32,RM))` | 262656 | 1313280 |
| Attention-state tables: `16*A([B,T],int32,RM)` | 4194304 | 16777216 |
| GDN remap indices, one per unique state shape: `A([B,12,128,128],uint32)+A([B,3,2560],uint32,RM)` | 868352 | 26148864 |
| Stable tiled logits + RM sampled-prefill staging, both `[1,1,32,62080]` BF16 | 7954432 | 7954432 |
| **Subtotal of these fixed categories** | **83083776** | **2012768768** |

The B32 helper cost includes both B1 and B32 worksets; it is not only32 times the B1 helper cost. The helper's attention state objects alias the existing K/V pool, so do not count a duplicate KV pool per helper or per state view. Its16 GDN-independent attention views share the helper's one full-table tensor. Conversely each actual attention layer's decode table is independent and counted16 times.

Small controls still require separate accounting: stable tokens, signed positions, unsigned RoPE positions, active masks, sampled-prefill previous mask, remap vector, helper start/head-position/batch-index/single-position scalars/vectors, sampling parameters/seeds/user IDs/greedy masks/index-offset tensors, and logprob helper setup tensors. These have bank/tile minimum allocations; summing only logical scalars undercounts them. Host tensors such as `page_host`, uploaded host wrappers and token histories are host RAM, not DRAM pool terms.

The common `SamplingGenerator` also constructs `TTPenalties` even though this generator's device penalty route is disabled. Five wide persistent arrays are therefore **not zero cost**:3 int32 vocabulary-local masks/count arrays `[32,62080]`, one full-vocabulary gathered count array `[32,248320]`, and one RM full-vocabulary zero array `[32,248320]`. Their combined allocation is87457792 bytes/device for both B1 and B32. Small penalty parameters/slice controls are additional. Include this already allocated common-module storage without claiming unsupported penalties are executed.

## CCL, traces and peak working set

The old69206016-byte CCL line cannot be reused as a bound for full1..512 preparation. `MultichipDecoder._collective` keys persistent buffers on `(reduce, logical_shape, dtype, memory_config)`. Short M=1..32 can retain different signatures despite sharing padded tile dimensions. Bounded preparation creates equivalent **private** resources on all matching layers, and terminal norm adds resources to layer0's owner. B1 and configured-B paths, statistics gathers, sampler value/index gathers and RS staging must all be inventoried.

The exact CCL resident term is the sum of `A` over distinct owned tensors in each layer's `_ccl_buffers`; count each layer's storage independently, but do not count the same storage twice through aliases. For scale, one BF16 hidden all-gather output plus one RS intermediate/output at a short-M signature has nominal allocations327680+655360+81920=1064960 bytes/layer. Retaining32 such signatures across64 layers would cost2181038080 bytes, before stats/sampler variants. This example is **not** a measured count or complete bound. Record the actual signature inventory after complete preparation.

Trace region requested by the probes is100000000 bytes/device. Current `AllocatorImpl::init_one_bank_per_channel` reserves `8*round_up(ceil(100000000/8),8192)=100007936` bytes across the banks, not100MB per bank. General-DRAM `total_bytes_per_bank` excludes this reserved region. Therefore do **not** subtract that same trace reservation again when budgeting from the measured34078723072 general-DRAM total above. If budgeting from nominal physical DRAM instead, subtract trace and other reserved regions once. Trace-owned intermediate buffers allocated in general DRAM and live-trace protection remain separate costs; the trace command-region reservation does not account for them. Measure command-region sufficiency as well as general memory.

Peak preparation/operation terms are also material:

| Example temporary | B1 bytes | B32 bytes |
| --- | ---: | ---: |
| One chunk's all-position BF16 logits `[B,512,62080]` | 63569920 | 2034237440 |
| Last-position prefill logits `[B,1,62080]` with each user's sequence tile | 3981312 | 127139840 |
|512 distinct head-concat preparation inputs `[B,1,1280]` BF16 | 41943040 | 1342177280 |

`BoundedPrefill.prepare` currently warms all-position terminal logits for every physical length, including512, and retains the distinct concat inputs until that helper preparation returns. That is a real B32 preparation peak even when production returns only last-position logits. Add layer projection/SDPA scratch, normalized full-hidden activations, concat output, masks/copies, and allocator fragmentation. Do not add every temporary across every layer as simultaneously live; derive actual lifetimes or measure high-water marks. Conversely snapshot memory after preparation can miss a much larger preparation peak.

As a sanity check only, loaded baseline + fixed-category subtotal + five wide penalty arrays + P8192 KV/reset total equals16381256192 bytes at B1 and18310941184 at B32. These are **incomplete lower inventories**, leaving17697466880 and15767781888 bytes of general DRAM respectively before CCL, small controls, programs, trace intermediates and peaks. They are not proofs of feasible pool size, safe reserves, or maximum aggregate tokens. The corresponding dense B32 lower inventory is91254082048 bytes/device and is physically impossible on the measured hardware.

## Future `get_max_tokens_all_users` mapping

The pinned plugin commit is `042a94863ad27d88207bcf71abdf67122c6f4c27`. `docs/MODEL_SETUP.md` and `src/vllm_tt_plugin/worker.py::get_num_available_blocks_tt` establish the exact boundary: the class-callable method returns the **aggregate KV token budget for one TP lane/rank before plugin-added headroom**, not context per request and not tokens summed over the four TP devices. The plugin does not measure free DRAM and defaults to131072 on `AttributeError`; relying on that default would leave the intended capacity unmapped.

Define `F(B,C)` as measured/derived fixed nonpool resident storage (loaded weights, all fixed rows above, actual prepared CCL, sampler, programs and general-DRAM trace-owned buffers), and `R_peak(B,C,P)` as a validated remaining peak/fragmentation reserve. For uniform P:

```
F(B,C) + 287232*P + R_peak(B,C,P) <= D_general
P_budget = largest physically validated P satisfying that constraint
```

If a constant conservative reserve has been validated over the relevant P interval, the preliminary arithmetic is `floor((D_general-F-R_peak)/287232)`. It is an estimate until the required allocation/operation evidence exists. A function called before model construction must use an evidence-backed hardware/policy/batch lookup or validated formula; it cannot invent an available-memory measurement.

For this autoregressive model, block size32 and `output_tokens_per_step=1`, the plugin adds exactly `32*B` tokens, i.e.B physical blocks, then computes `ceil(tokens/32)`. No sliding-window headroom is appropriate for GDN: recurrent states belong in fixed slot storage, not fake sliding-attention KV groups. Define `P_extra` for additional implementation-owned scratch/null/slack pages **outside** the plugin-counted pool, if any. Then a safe future return valueQ must satisfy:

```
ceil((Q + 32*B)/32) + P_extra <= P_budget
Q <= B*C
```

A convenient block-aligned candidate is `Q = min(32*floor(B*C/32), 32*(P_budget-P_extra-B))`, with nonpositive values rejected. Account explicitly for null/admission pages inside the plugin pool too: to admit one full-C request, its live pages plus null/in-flight requirements must fit the resulting pool. Do not claim single-request262144 from Q alone. The current plugin only conditionally raises its hybrid admission floor; do not rely on an automatic budget correction or enable hybrid sliding groups to hide an under-budgeted pool.

For B32, a Q smaller than32*C is an aggregate concurrency budget, **not** a context reduction. Serve mixed lengths so their active page allocations fit; permit an individual request up toC when enough aggregate pages are free. Keep `max_seq_len=C` and `[B,T]` virtual tables independent of P. Physical pages represent a pool shared among requests inside one generator; different independently live generators may not share this writable pool because `prepare/reset` clear it wholesale. Prefix sharing is not admitted merely because page IDs can alias: KV-only automatic prefix caching remains false for the GDN model.

## Current pool-to-prefill issue to resolve

At full virtual width, B32 tables contain262144 entries. Any feasible much smaller P requires repeated IDs in unused/unallocated entries, because the constructor requires all entries to be nonnegative and in range. `BoundedPrefill.run` currently decides bulk-fill eligibility by uniqueness of the **entire** submitted row table. It therefore disables the accepted batched bulk fill for such a bounded pool even when the16-page spans actually written by the current batch are disjoint. This is stricter than write ownership requires and affects the optimized serving path.

Also `prepare()` initially leaves `_bulk_fill_enabled=True` and directly warms the bulk operation against its supplied fill table without first proving those initial write spans disjoint. For an externally pooled/aliased initial mapping, `disjoint_write_pages=True` can be an unjustified warmup promise. Preparation state is later reset, but that does not make a concurrent-write ownership assertion true.

Preserving the optimized strategy with a bounded pool requires valid disjoint preparation mappings and sound certification of the physical pages actually written by each current chunk, with active-row/length ownership included; immutable/private runtime tables must remain trace-safe. For an aligned chunk with logical `count`, validate precisely `ceil(count/32)` page entries beginning at `absolute//32` for every active row: every written ID must be in range in every bound cache, and the flattened written-ID set must be unique before asserting bulk ownership. Padded tail tokens still write their final whole32-token tile. `paged_fill_cache_program_factory.cpp` derives work from `input_batch*num_heads*(padded_input_seq/32)`; its writer maps those sequence tiles through the table. For full512 warmup, a sufficient private mapping is first16 columns of rowr=`16*r+arange(16)`, remaining unread columns any valid aliases, requiring `P>=16*B`; external P64/virtualT96/B3 satisfies this with IDs0..47. Preserve the production mapping separately and ensure first real run refreshes the private runtime table. If P is smaller than16*B, do not assert that impossible simultaneous full-batch512 fill is disjoint: explicitly restrict which feasible bulk signatures are warmed or reject the undersized full-batch preparation contract without relabeling per-request context. This is implementation work identified by source, not a request to allocate dense B32 state or silently accept slower serving. The parent implementation owner was notified. No decoder/helper source was changed by this audit.

## Measurements and final contract updates still required

1. Record all64 loaded baseline, after external-pool binding/reset zeros/helper allocation, after complete B1/B32 preparation, after capture, and peak warmup/prefill/decode usage on the exact final policy/native build. Inventory CCL signatures and trace-owned buffers; include per-bank largest contiguous space, not only aggregate free bytes.
2. Test the external bounded physical pool while retaining virtual C262144 tables, request-slot reuse, noncontiguous page IDs, changed/unchanged mappings, mixed prompt lengths and legal simultaneous writes. Test initialization and full preparation on the actual pool layout; do not infer this from dense context2048 probes.
3. Execute the required maximum-context full-model acceptance once. If a hard physical limit is found, retain the one bracketing pass/fail pair and the largest feasible value; do not advertise an untested reduced maximum or repeatedly search context as an optimization shortcut. Aggregate B32 pool sizing is separate from the preserved per-request limit.
4. Update `doc/context_contract.json` with measured weight+KV+zero+helper+CCL+trace/peak accounting, exact P and budget scope. Reconcile the old plan rather than adding overlapping entries. Update the serving-contract `get_max_tokens_all_users` row with the validated Q formula/lookup, plugin output headroom, cache geometry, exclusive-pool ownership, admission/concurrency meaning and exact evidence. No numeric Q is certified by this source-only note.

## Followup: a conservative B32 pool, without reducing context

Source-only followup to the planned command (not an executed result):

```
probe_full_slots.py --batch 32 --all-layers --prefill-length 33 \
  --context 262144 --physical-pages-per-slot 257 --full-prepare \
  --output <parent-selected artifact>
```

This probe's shuffled initial mapping has `32*257=8224` physical pages and
virtual shape `[32,8192]`. Each row repeats its private257-page allocation in
unused virtual columns. The first16 pages of every row are disjoint, so the
full512-token batched preparation has enough real writable pages. Current
`BoundedPrefill.prepare` also constructs private warmup tables and restores
runtime mappings; the earlier “Current pool-to-prefill issue” section records
the pre-repair finding, not an outstanding claim about this revised source.
The full repertoire is the default `prepare()` path; the diagnostic subset
must not be used for this capacity acceptance.

Conditional on that exact all64/B32/full-table/full-preparation workload
succeeding with final source and its memory/CCL/trace evidence, the **configured
aggregate token budget** `Q=262144` has an exact pinned-plugin mapping:

```
B = 32
block_size = 32
output_tokens_per_step = 1
plugin_headroom_tokens = max(32,1) * 32 = 1024
plugin_allocated_pages = ceil((262144 + 1024) / 32) = 8224
```

Thus the proposed pool matches the plugin's requested allocation exactly.
There is no additional dense `B*context` allocation and no division by the
four TP ranks. The evidence would justify this conservative configured pool
for the measured policy/batch/hardware. It would **not** prove that8224 is the
largest physically feasible pool, nor that262144 is the largest possible
aggregate concurrency budget. The stage does not need an exhaustive pool
sweep to choose a conservative operating budget. Keep the independent
per-request advertised context at262144 and perform the required B1
262143-token prefill plus last-position decode acceptance once. A smaller
aggregate budget than32 full requests is not a context reduction and does
not trigger a fabricated pass/fail search for a lower context maximum.

The current slot probe supplies a host mapping to the native allocator rather
than constructing caller-owned storage in this particular command. This still
exercises the exact bounded pool allocation because `create_paged_state`
allocates `max(page_id)+1`; independent external-cache identity is established
by the mixed-prefill probe. Do not describe this command alone as proof of an
independently allocated cache object's binding.

### Null block and admission

Pinned `worker.py::_upstream_admission_blocks` explicitly adds one BlockPool
null block to the blocks needed by one maximum-length request. That null
block is **inside** the plugin's8224-page allocation, not an extra TT tensor
or an additional out-of-pool page. With one homogeneous full-attention group,
no speculation/lookahead, no sliding-window groups, and automatic prefix
caching disabled, one262144-token request needs8192 live pages. Including
the null block requires8193 pages, which fits8224 with31 pages remaining.
The hybrid admission-floor path is not needed and should not be enabled to
manufacture another budget. Future integration must preserve this exact
one-group cache specification and verify the pinned upstream admission result.

During actual serving, do not assign the reserved null ID to a writable active
request range. Unallocated virtual tails may alias a valid placeholder/null
ID because kernels must not access them until scheduler allocation. Setup
warmup may temporarily write the backing storage before the generator's full
reset; the restored zero null page and production mapping must be established
before requests. The proposed raw-generator probe has no vLLM BlockPool and
therefore does not independently test its null-ID convention.

There is no separate KV scratch/null allocation in the current generator:
helper attention views alias the bound pool, and private full/fill tables are
metadata. Reset-zero storage, helpers, program buffers and prefill scratch
are real DRAM costs included in the measured8224-page generator preparation,
not extra physical page IDs to subtract again. If a future adapter introduces
any additional physical scratch allocation outside that pool, it must account
for it and cannot claim this evidence unchanged.

The32 output-reservation blocks must not be sold as a guarantee that *every*
32-request mixture with aggregate prompt tokens Q can also append output
without scheduler pressure. Per-request rounding and the null block consume
capacity too. For example,32 prompts of8192 tokens have aggregate length262144 and
consume8192 live pages. Appending one cached token to each requires8224 live
pages, or8225 including the null block: one more than the proposed pool. The
scheduler must defer or preempt a request before that simultaneous allocation. More generally,
for any live prefix lengths `L_i`, admission must use
`1 + sum_i ceil(L_i/32)` plus genuinely needed in-flight allocations, not just
`sum_i L_i <= Q`. The conservative Q defines physical allocation sizing; the
scheduler remains responsible for actual block admission and preemption.
No promise of32 simultaneous maximum-context requests is made.

The fixed257-page-per-slot map is sufficient for this probe's short requests;
it is not the production scheduling policy. A long individual request must be
allowed to acquire pages across the shared pool and install them in its full
8192-column row. Repeating257 IDs for an active long prefix would overwrite
history and is forbidden. The B1 maximum-context acceptance proves the full
logical interval with its own valid map; it does not directly execute a full
context request while B32 state remains resident. Relate the two using the
bounded helper's fixed physical chunks/runtime positions and measured
preparation headroom, and retain that distinction in evidence. If a claimed
B32 workload needs additional position-dependent scratch beyond the validated
preparation envelope, measure or justify that cost before claiming coverage.

Finally, the formula is batch-specific. Returning Q262144 at B1 would ask the
plugin for8193 pages after its one-block reservation, whereas the plain B1
context allocation is8192 pages. A B1 maximum-context test alone does not
measure that extra plugin page or establish every batch's budget lookup.
Record the tested B32 operating point explicitly, and derive other supported
batch entries from their own validated allocation accounting. No success,
physical maximum, byte high-water mark, or completed admission test is asserted
by this addendum before the parent records the planned measurements.
