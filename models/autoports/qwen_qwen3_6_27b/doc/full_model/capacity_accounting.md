# Full-model physical capacity accounting

Current source accounting, 2026-11-14. All byte values are **per device/rank** on the selected 1x4 TP4 lane, not sums over four ranks. Both final all64 B32 bounded-pool and B1 maximum-context runs pass; measured resident totals and the non-extrapolating capacity scope are reconciled below. Earlier investigation is preserved in `capacity_accounting_initial.md` and the focused AutoFix reports.

**Preserve the per-request context of 262144.** A configured batch of 32 does not promise 32 simultaneous full-context requests. Virtual per-slot context and aggregate physical KV capacity are independent: a scheduler admits requests against the physical pool while retaining full-width virtual tables. No context reduction or largest-physical-pool claim is made here.

## Geometry and allocation model

The selected decoder has 16 full-attention layers and 48 GDN layers. Each rank owns one KV head of dimension 256, with 32 tokens per page and BF8_B K/V. Published GDN state is BF16; live bounded-prefill recurrence is FP32. Vocabulary width is 248320 globally / 62080 per rank; residual hidden width is 1280 per rank.

| Symbol | Meaning |
| --- | --- |
| `B` | Configured fixed slots, 1 through 32 |
| `C` | Virtual per-request context, at most 262144 |
| `T = max(16,16*ceil(C/512))` | Virtual page-table columns per slot; 8192 at full context |
| `P` | Physical pages per full-attention layer/rank; shared among requests within one generator |
| `H(B)` | Helper physical batches: `{1}` for B1; `{1,B}` otherwise |
| `A(shape,dtype,layout)` | Modeled eight-bank DRAM allocation, including page/bank alignment |
| `D_general` | Available general DRAM after reserved regions |

For interleaved allocations on this machine:

```text
A = 8 * ceil(n_pages / 8) * 64 * ceil(page_bytes / 64)
```

TILE pages are 32x32: BF16/UINT16 = 2048 bytes, FP32/INT32/UINT32 = 4096, BF8_B = 1088. ROW_MAJOR pages are last-dimension rows, with page bytes equal to row width times element size. Tile counts use padded shapes. For example, BF16 RM `[1,3,2560]` contains 15360 payload bytes but consumes 40960 modeled bytes across the banks.

The derivation is `tt_metal/impl/tensor/spec/layout/{tensor_layout,page_config}.cpp::compute_consumed_memory_bytes_per_bank`; `tests/optimized_multichip_capacity.py::tensor_bytes` supplies the source model. This arithmetic does not measure fragmentation, cached native metadata, programs, or simultaneous transient peaks. Use the measured loaded-model baseline for DRAM-sharded weights rather than applying this interleaved formula to them.

## Loaded-model baseline

`doc/context_contract.json:full_model.loaded_model_dram_measurement` and `teacher_forcing_first_all64.log` record:

| Measured quantity | Bytes |
| --- | ---: |
| General DRAM per device: 8 banks x 4259840384 | 34078723072 |
| Loaded all64 model per device: 8 x 1732213760 | 13857710080 |
| Free after load per device | 20221012992 |
| Largest free block per bank after load | 2423779200 |

This resident snapshot includes real layer weights, embeddings, LM head, shared RoPE, and model-owned setup resources before generator/cache/helper construction. Shared RoPE (67108864 bytes) is already inside it. Do not add separate historical matrix/head/RoPE inventories to this baseline. The final-native maximum-context artifact reproduces this baseline exactly. A largest-free-block value is per bank; aggregate free bytes alone do not prove a later allocation fits.

## Physical KV and reset-zero storage

One K or V page is `[1,1,32,256]`: eight BF8_B tiles, or 8704 bytes per rank. Uniform physical extents across all 16 attention layers give:

```text
KV(P)       = 16 * 2 * 8704 * P = 278528 * P
KV_zero(P)  = 8704 * P
KV_total(P) = 287232 * P
```

`generator.reset_zeros` deduplicates by shape/dtype/layout, so identical K/V target shapes share one zero source. The 32 K/V targets remain distinct. For heterogeneous layer extents use `sum_l 2*A([P_l,1,32,256],BF8)` and one reset source per distinct zero signature. Cross-layer physical tensor aliasing is not supported.

| Physical pool | K/V bytes | Shared reset-zero bytes | Total bytes |
| --- | ---: | ---: | ---: |
| 8192 pages, logical context alone | 2281701376 | 71303168 | 2353004544 |
| 8193 pages, validated B1 plugin pool | 2281979904 | 71311872 | 2353291776 |
| 8224 pages, validated B32 plugin pool | 2290614272 | 71581696 | 2362195968 |

Each physical page adds 287232 bytes/device including the reset source, or 8976 bytes per aggregate cached-token capacity. These coefficients are exact for the selected page geometry and uniform extents.

The default dense table uses `P=B*T`. At B32/full context, K/V alone is 73014444032 bytes (68 GiB/device), before the 2281701376-byte zero source. That cannot fit this machine. A supplied bounded host table makes `create_paged_state` allocate `max(page_id)+1`; externally allocated cache tensors can also supply a bounded pool. Neither route requires reducing virtual `C` or allocating dense `B*C` cache storage.

## Fixed B/C storage outside the pool

The following source-derived allocations use `C=262144`. They are a partial resident inventory, not the complete process total.

| Category and formula | B1 bytes | B32 bytes |
| --- | ---: | ---: |
| Published GDN: `48*(A([B,12,128,128],BF16)+A([B,3,2560],BF16,RM))` | 20840448 | 627572736 |
| Deduplicated GDN reset sources: one recurrence and one convolution shape | 434176 | 13074432 |
| Helper state: `sum_(h in H(B)) 48*(A([h,12,128,128],FP32)+A([h,3,2560],BF16,RM))` | 39714816 | 1271267328 |
| Helper IDs and RoPE positions: `sum_h 2*sum_(n=1..512) A([h,n],UINT32,RM)` | 8650752 | 43253760 |
| Helper row-gather indices: `sum_h A([h,1,1280],UINT32,TILE)` | 163840 | 5406720 |
| Helper full/fill tables: `sum_h (A([h,T],INT32,RM)+A([h,16],INT32,RM))` | 262656 | 1313280 |
| Attention decode tables: `16*A([B,T],INT32,RM)` | 4194304 | 16777216 |
| GDN remap indices: `A([B,12,128,128],UINT32)+A([B,3,2560],UINT32,RM)` | 868352 | 26148864 |
| Stable logits plus sampled-prefill RM staging, both `[1,1,32,62080]` BF16 | 7954432 | 7954432 |
| **Listed fixed subtotal** | **83083776** | **2012768768** |

B32 includes both B1 and B32 helper worksets. Helper attention states alias the existing K/V pool and share each helper's full table; they do not allocate another KV pool. Actual decoder attention layers retain their independent decode tables. Host tables, token histories, and upload wrappers are host RAM rather than extra device pool pages.

The common sampler also allocates five wide penalty-support arrays even though this generator disables device penalties: three local-vocabulary INT32 arrays `[32,62080]`, one gathered INT32 array `[32,248320]`, and one RM INT32 zero array `[32,248320]`. Their modeled total is **87457792 bytes/device** for both B1 and B32. Count their storage without claiming those unsupported penalties execute.

Additional controls must still be inventoried: stable tokens/positions/RoPE positions, active and sampling masks, prior-mask backup, remap vectors, helper start/head/batch controls, sampler parameters/seeds/user IDs/greedy masks/index offsets, and logprob setup resources. Their bank/tile minimum allocations cannot be replaced by logical scalar payload counts.

## Prepared CCL, native metadata, programs, and traces

Persistent decoder CCL is keyed by reduction flag, logical shape, dtype, and memory config. Both layer kinds select `flat_ccl=True`; the persistence test runs after packing `B*S` rows and retains outputs only for packed M<=32. Thus B1 preparation may retain 32 distinct short logical signatures despite common padded tile shapes; uniform B32 long sequences do not create a new persistent long-shape CCL buffer at every length. Each layer owns private equivalent resources, copied after representative preparation; terminal norm adds layer0 resources.

Use `tests/full_model_evidence.py::ccl_inventory` to sum distinct live layer-owned buffers, including UINT16 resources. The completed B32 run inventories **2206482432 bytes/device**, modeled from the actual live owned buffers: layer0 has34619392 bytes, the other47 GDN layers have34603008 each, and16 attention layers have34095104 each. This inventory is separate from the allocator's total resident measurement. Sampler-owned resources outside layer `_ccl_buffers` remain separate. Do not substitute a single-signature estimate or multiply all logical prefill lengths by a full CCL output size.

Native reshape owns cached segment-map tensors through `ProgramArtifacts.op_owned_tensors`. The repaired native tiled path stores one mapping page per output tile row when input/output widths are equal and tile-aligned. If `W_tiles=W/tile_width`, the mapping-page count changes from `N_output_tiles` to `N_output_tiles/W_tiles`; source-page column offsets are reconstructed in the reader. Other shapes retain the original mapping. Metadata page size and allocator alignment remain part of `A`. This preserves all shape signatures and cached ownership while removing repeated column metadata. The CPU proof and independent review are in `reshape_mapping/`; full64 final measurements must include the repaired build.

Cached program kernel binaries also own DRAM (`ProgramImpl::allocate_kernel_bin_buf_on_device`), allocated top-down. `ttnn._ttnn.reports.get_buffers(mesh)` can enumerate native-owned allocations; its records do not label owners. Reconcile simultaneous buffer addresses with tensor inventories and compare against allocator totals, rather than attributing every unknown buffer to binaries. JIT build-cache counters are not live TTNN program counts.

A requested 100000000-byte trace region reserves:

```text
8 * round_up(ceil(100000000 / 8),8192) = 100007936 bytes/device
```

General-DRAM capacity already excludes that command region. Do not subtract it again from `D_general=34078723072`. Trace-owned intermediate tensors in general DRAM are a separate resident cost; trace-region sufficiency is also a separate acceptance check.

## Temporary peaks and fragmentation

| Example simultaneous allocation | B1 bytes | B32 bytes |
| --- | ---: | ---: |
| All-position BF16 logits `[B,512,62080]` | 63569920 | 2034237440 |
| Last-position BF16 logits `[B,1,62080]`, with per-user sequence padding | 3981312 | 127139840 |
| 512 distinct concat preparation inputs `[B,1,1280]` BF16 | 41943040 | 1342177280 |

Current `BoundedPrefill.prepare` retains its distinct concat-input pool while warming terminal shapes. These are real preparation costs even when serving returns only last-position logits. Add the actually live normalized hidden, concat result, projection/attention scratch, masks, and copies. Do not add every layer's transient as if all layers retain it; conversely, an after-prepare snapshot can miss a much larger warmup peak.

Fragmentation must be checked per bank. The pre-repair B32 failure in `slots32_all64_pool.log` had more aggregate free space than the failed head allocation but insufficient largest contiguous space. Its source attribution and repaired metadata path are documented in `b32_capacity_hypothesis.md` and `reshape_mapping/`. That historical failure is not a context-capacity limit and must not be used to reduce context or as the final repaired peak measurement.

## Bounded write ownership and cache lifetime

Current bulk-fill eligibility certifies only the pages actually written: for an aligned chunk beginning at `absolute` with logical `count`, validate `ceil(count/32)` entries from column `absolute//32` for every active row. IDs must fit every bound physical cache, and the combined active write set must be disjoint. The final padded tile still owns a whole page. Repeated unused virtual-tail IDs do not invalidate an otherwise safe bulk chunk.

Preparation installs private safe warmup mappings, uses native skipped writes for unavailable fill destinations, warms both required branches, and restores runtime tables. It does not require an impossible unique mapping across every unused virtual entry. `active_fill_page_proof.md` records the write-range proof; mixed-prefill evidence covers independent externally allocated cache binding. Reset/prepare clear the complete bound writable pool, so one generator owns that pool exclusively for its lifetime. Different independently live generators may not share it. GDN-owned state also means KV-only automatic prefix caching remains disabled.

## Aggregate budget and pinned plugin admission

Pinned plugin `042a94863ad27d88207bcf71abdf67122c6f4c27`, `docs/MODEL_SETUP.md` and `worker.py::get_num_available_blocks_tt` define `get_max_tokens_all_users` as an aggregate token budget **before plugin headroom**, for one TP lane. It is neither per-request context nor a sum over TP ranks. The plugin does not measure free DRAM; its fallback default is not evidence for this model's capacity. The completed method mapping in `serving_contract.json` uses the validated operating points below.

Let `F(B,C)` cover measured/derived nonpool resident storage and `R_peak(B,C,P)` cover the validated remaining temporary/fragmentation reserve. A candidate pool must satisfy:

```text
F(B,C) + 287232*P + R_peak(B,C,P) <= D_general
```

With a validated constant reserve over the relevant interval, `floor((D_general-F-R_peak)/287232)` is a preliminary upper estimate. It is not a proved largest physical pool. Selecting a conservative feasible operating point does not require claiming that it is physically maximal.

For block size32, output_tokens_per_step=1, no speculation/lookahead and one homogeneous full-attention group, the plugin adds `32*B` tokens. If `P_extra` denotes genuine scratch pages outside its pool:

```text
P_plugin = ceil((Q + 32*B)/32)
P_plugin + P_extra <= P_validated
Q <= B*C
```

A block-aligned preliminary candidate is `Q=min(32*floor(B*C/32),32*(P_validated-P_extra-B))`, rejecting nonpositive values. GDN recurrence belongs in fixed slot storage; do not invent sliding-attention groups or sliding headroom for it. The current generator has no out-of-pool KV scratch allocation: helper views alias the pool and helper tables are metadata. Its reset zeros and other fixed resources are already byte costs above, not additional physical page IDs.

For the validated conservative **Q=262144** operating point:

| Configured batch | Plugin headroom tokens | Plugin physical pages |
| --- | ---: | ---: |
| B1 | 32 | 8193 |
| B32 | 1024 | 8224 |

The B32 allocation mapping is now exercised by the completed all64 bounded-pool run described below. The B1 mapping is exercised by the completed maximum-context acceptance. Intermediate batch entries require validated accounting/coverage before publication; neither endpoint alone establishes every batch.

The pinned upstream admission calculation reserves one BlockPool null block **inside** `P_plugin`. One full-context request needs8192 live pages; with the null block it needs8193. That fits the validated B1 pool exactly and leaves31 pages in the validated B32 pool. Do not assign the null ID to active writable history. Unallocated virtual tails may use placeholders until the scheduler allocates real pages. Warmup/reset must restore production contents and mappings before requests. The raw B32 slot probe has no vLLM BlockPool and does not itself test that null-ID convention.

Actual request admission uses page counts, not merely aggregate token sums:

```text
1 + sum_i ceil(L_i/32) + required_in_flight_pages <= P_plugin
```

For example, 32 prompts of8192 tokens occupy8192 live pages. Appending one cached token to each needs8224 live pages plus the null block, exceeding an8224-page pool by one. Defer/preempt before that allocation. Q sizes the configured pool; it does not guarantee every mixture summing toQ can append simultaneously. Scheduler admission/preemption is a future adapter responsibility, not additional vLLM implementation in this stage.

The B32 probe's shuffled257-pages-per-slot mapping has virtual shape `[32,8192]` and P8224. Its short active histories are valid; repeated257-page patterns are not a production long-request allocation policy. A long request must acquire distinct pages across the shared pool and install them in its full-width row. The B1 maximum-context test uses valid pages1..8192 in P8193. It does not directly execute full context with all B32 state resident; preserve that distinction when relating fixed-chunk preparation coverage to serving capability.

## Intermediate configured batches: bounded terms and remaining gap

The proposed general formula is algebraically consistent with the plugin contract:

```text
Q(B,C) = min(262144, 32*floor(B*C/32))
P(B,C) = ceil((Q(B,C) + 32*B)/32)
```

For `B<=32`, `C<=262144`, it never allocates more than8224 physical pages; at full context B2..31 use8194..8223 pages. Reject a configuration for which the block-aligned Q is nonpositive rather than presenting it as usable capacity. The admission/null rules above still apply.

Several terms have a source-backed B32 upper bound: every fixed category in the table is monotone inB at fixedC; the B1 helper is common and the configured-batch helper's state and tables grow withB; K/V plus reset-zero storage is bounded byP8224; the stable 32-row sampler buffers are fixed; and B<=32 all-position logits and concat-input pools are bounded by the B32 shapes. The tile-aligned equal-width native mapping repair also bounds the number of retained mapping pages for a particular unpack shape by its output tile-row count, rather than by vocabulary/hidden tile columns.

**These component bounds do not yet prove total prepared memory or peak/fragmentation for every intermediate configured batch.** B32 compact rows are `32*S`, hence tile-aligned for every length. Intermediate batches execute additional shape-dependent routes. For example, B31/S9 has279 compact rows; `multichip_decoder.py::_linear` pads to288 before MMRS and slices back to279 afterward. B32/S9 has288 rows and skips that pad/crop pair. Native reshape routing in `reshape_codegen_supported.cpp::plan_codegen/is_demoted` also depends on divisibility, geometry, and available L1, rather than only total element count. Its cached mapping/program resources therefore are not proved to form a subset of the B32 preparation cache.

The observed10861259264 bytes free after the B32 run is substantial operational headroom, but it is a resident snapshot after preparation temporaries have died. It supplies neither a numeric upper bound for intermediate-batch extra cached resources nor a measured peak reserve. No source-derived complete bound for those extra routes has been established by this audit. Do not convert "smaller batch" into a blanket hardware-capacity claim.

The stage's operating-budget lookup therefore covers the measured **physical fixed-slot B32 configuration**, and the separately accepted physical B1 configuration. A B32 generator still accepts **1 through32 live requests**, using mixed prompts, fixed slots and inactive rows without changing its physical configured batch. Choosing the measured B32 operating configuration does not require all32 requests to be active.

Other constructor batch sizes remain implemented, and some have focused functional evidence. They do not inherit Q262144 full-context prepared-capacity certification from the endpoints. This distinction concerns measured operating configurations; it does not reduce per-request context, the largest supported physical batch, or the number of live requests the B32 API accepts. No additional hardware sweep is required for the stage to use the measured B32 lookup. A future lookup advertising Q262144 for physical constructors B2..31 would need an actual resource bound or validation of their extra shape families. Keep that future extension separate from current stage acceptance.

## Completed full-model measurements

Both `slots32_all64_pool_fixed.json` and `max_context_final.json` pass on the seven recorded current model/native source hashes. The loaded-model baseline is reproduced on the final build. The following are resident snapshots, not allocator high-water marks.

| Configuration / phase | Allocated bytes/device | Free bytes/device | Largest free bytes/bank |
| --- | ---: | ---: | ---: |
| All64 loaded model | 13857710080 | 20221012992 | 2423779200 |
| B1/C262144/P8193 cache and setup | 16382112768 | 17696610304 | 2192940352 |
| B1 prepared, after max prefill, and after final decode (each identical) | 19072073728 | 15006649344 | 1870288192 |
| B32/C262144/P8224 after complete preparation and functional replay | 23217463808 | 10861259264 | 1184727168 |

B1 layer-owned CCL allocation is2206220288 bytes/device; B32 is2206482432. These are live-buffer allocator inventories, already included in the resident snapshots. The B1 partial nonoverlapping inventory (loaded baseline + listed fixed83,083,776 + KV/reset2,353,291,776 + wide sampler arrays87,457,792 + CCL2,206,220,288) is18,587,763,712 bytes, leaving484,310,016 unclassified among native maps/programs, small controls, trace/general resources and other live allocations. The corresponding B32 partial inventory is20,526,615,040, leaving2,690,848,768 unclassified. Neither residual is all kernel binaries, free memory, or a peak reserve.

The B32 artifact records full1..512 preparation for both B1 and configured B32, three captures, finite identical uniform33-token logits across32 slots, inactive-state preservation, resident replay, remap, release and final replay. Although its key is `prepared_memory`, the snapshot is taken after those functional assertions. The bounded host mapping allocates8224 pages with virtual table shape `[32,8192]`; independent externally allocated cache identity is established separately by `mixed_prefill_native_all64.json`.

The single maximum-context acceptance uses B1/P8193, reserved page0 and active pages1..8192. Public prefill consumes262143 nonaligned tokens; resident decode consumes position262143 and advances to262144. Both generated IDs are valid. Programs remain24456, the three traces remain unchanged, no full logits are read, and final decode makes zero token/position/RoPE/page-table host refreshes. Prepared, after-prefill and after-decode memory are identical. Its313.599s prefill and40.130ms final decode are allocation-tracked acceptance timings, not the uninstrumented primary performance result.

The selected pools passed complete preparation and their stated runtime workloads. **No allocator high-water mark or extrapolated maximum pool is claimed.** Preparation's transient admission is established by its completed execution, including every1..512 shape; the one B1 long-context run establishes its full logical interval. We do not use a guessed reserve to expand these measured pools or certify other physical batches. The B32 short-request/full-preparation result and B1 full-context result retain their separate scopes.

The operating lookup is Q262144 before plugin headroom for physical B1/P8193 and B32/P8224, with1..32 live/inactive requests supported by the fixed B32 API. `serving_contract.json` maps this to the existing generator pool/table construction and records null/headroom/admission semantics. `doc/context_contract.json:full_model` now records all memory phases, cache coefficients, source identities and acceptance scope. Context262144 is preserved, with no capability reduction and no hard-limit bracket. No vLLM implementation or server is claimed.
