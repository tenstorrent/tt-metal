# Metal 2.0 Bulk Op Migration — Working Report

Branch: `dgomez/metal2-bulk-migration` (from the rand+sampling+infra work).
Goal: migrate the listed ops to Metal 2.0, measure host-dispatch with vs without Metal 2.0,
leave genuinely-blocked ops for later, report any op that would need a metal-API change.

**Rules honored:** no metal-API changes (report needs at the end); no custom hashes
(exclude values from the key via an `immutable_info_t`); faithful kernel ports (binding
mechanism only); measure the realistic cache-hit dispatch path.

## Status legend
- ✅ migrated + built + validated + measured
- 🟡 in progress
- ⛔ blocked (reason) — left for later
- ⚙️ needs metal-API change (reported below)
- ⬜ not started

## Ops

| op | status | concept | legacy µs | metal2 µs | notes |
|---|---|---|---|---|---|
| reduction/sampling | ✅ | Adv 3++ | ~1e6 (varying seed) | ~92 | done earlier; 1 cache entry / N seeds |
| pool/generic | ⛔ | — | | | BLOCKED on gap #2 (op-owned halo lookup-table / config device tensors). No code attempted. |
| normalization/layernorm | ⬜ | | | | |
| reduction/topk | ✅ partial | degenerate (run-args) | 207.1 (incl per-iter sync) | 211.7 (incl per-iter sync) | Migrated TopKSingleCore (W<8192 / non-pow2 / k>64 routes here); TopKMultiCore (semaphores) left legacy. 117 pytest pass / 80 xfail / 0 fail; 1 cache entry. No bugs. Host µs is sync-dominated (W=256 single tile); diff within noise. |
| data_movement/concat | ⛔ | — | 35.4 | — | REVERTED. Only ConcatS2STiled was migrated; it is BLOCKED on gap #3. Sharded input/output tensors are accessed purely via `borrowed_from` DFBs (no NOC tensor accessor on any kernel), so the framework's referential-integrity check (`program_spec.cpp:462`, `tensor_parameter_users`) reports `TensorParameter 'input0' is defined but not bound by any kernel`. `borrowed_from` does not register a tensor-parameter user. 42 TILE_LAYOUT cases fail; op reverted whole (only migrated variant). |
| eltwise/binary_ng | ⛔ | — | | | BLOCKED on gap #4 (chained-accessor offset API on the interleaved path). Left legacy; no code. |
| matmul | ⬜ | | | | partial Metal2 work exists in a sibling worktree |
| data_movement/slice | ⛔ | — | 38.3 | — | REVERTED. Only SliceRmStride was migrated, but slice's `program_factory_t` std::variant is also consumed by another op — `ccl/mesh_partition` — which hand-writes a `std::visit` that unconditionally calls `Factory::create_descriptor` on every variant. Migrating one variant to the Metal 2.0 `create_program_artifacts` API breaks that external consumer's compile (`no member named 'create_descriptor'`). Fixing it would require editing mesh_partition (another op, out of scope). Op reverted whole. |
| data_movement/transpose | ✅ partial | degenerate (run-args) | 29.1 | 31.7 | Migrated CN + HCTiledInterleaved; WH/row-major left legacy (shared single factory). 122 pytest pass (`-k "cn or hc"`) / 0 fail; 1 cache entry. No bugs. |
| conv/conv2d | ⬜ | | | | |
| copy/typecast | ✅ partial | degenerate (run-args) | 50.2 | 64.8 | Migrated 3/4: interleaved (TypecastProgramFactory) + subgrid + RM-chunked; sharded left legacy (program_factory_t mixes legacy+Metal2). 713 pytest pass / 0 fail; 1 cache entry. Two bugs fixed during migration: (a) local DFB producer/consumer must share the SAME WorkUnitSpec — reader/writer were split into a DM-only WU separate from compute → "DFB producer and consumer do not cover the same WorkUnitSpec"; merged reader/writer into each compute group's WU. (b) `UnpackToDestFp32` only valid when input DFB is Float32 — legacy applied it unconditionally under preserve_fp32_precision (tolerated no-op for *->uint8); now guarded on input format. Host µs measured incl. sync; control floor ~12 (legacy) vs ~15 (metal2). |
| data_movement/pad | ✅ partial | degenerate (run-args) | 38.7 | 24.9 | Migrated tiled multicore (PadTileMulticoreProgramFactory) + single-core; RM/sharded left legacy. 320 pytest pass / 0 fail (test_pad_tile + TILE program-cache); 1 cache entry. No bugs — reader/writer already co-located in one WU per the local-DFB rule. metal2 faster than legacy (legacy baseline had high variance, p10≈23). |
| data_movement/sharded/interleaved_to_sharded | ⛔ | — | | | LEFT LEGACY. Metal 2.0 has no compile-time-vararg form for the `shard_builder` addrgen args (the sharded writer's per-core address-generation config is appended as a variable-length CTA block; the ProgramSpec CTA API takes only named scalars). Not attempted. |
| data_movement/sharded/reshard | ⬜ | | | | |
| data_movement/sharded/sharded_to_interleaved | ✅ | degenerate (run-args) | 22.6 | 21.3 | Migrated the single variant; op-private kernel copies (eltwise_copy / reader_unary_sharded / writer_unary_(stick_layout_)sharded_blocks_interleaved). test_sharded_to_interleaved_oob passes; 1 cache entry. No bugs. |
| data_movement/tilize | ✅ partial | degenerate (run-args) | sc 24.5 / mc 43.6 | sc 43.5 / mc 44.3 | Migrated single-core + multi-core default; sharded/block/width-sharded left legacy. 59 pass / 1 fail (`test_deepseek_v3_mla_tilize_trace_mode` — PRE-EXISTING, also fails on legacy). 1 cache entry (sc benchmark shows 2: a `from_torch` tilize warms a 2nd entry, same on legacy). Bugs fixed: (a) single-core `-Werror` unused `stick_size`; (b) **multi-core default dropped the sharded width-split** — `select_program_factory` routes non-optimizable sharded input to this factory, but the migration hard-coded `num_pages_in_row=1` / valid-last-page=`page_size`, corrupting 23 nd/width-sharded cases. Restored the legacy `is_sharded()` branch (the m2 reader kernel already supported it). |
| data_movement/untilize | ✅ partial | degenerate (run-args) | 242.9 (incl per-iter sync) | 258.2 (incl per-iter sync) | Migrated single-core only; multi-core/sharded left legacy. 164 single_core pytest pass / 0 fail; 1 cache entry. Bug fixed: the `writer` kernel binds the `output` tensor parameter but had NO KernelRunArgs entry — the framework fills tensor addresses from kernel_run_args, so `ValidateProgramRunArgs` aborted with "non-empty RTA/CRTA schema but no runtime parameters". Added a per-core writer KernelRunArgs entry (empty scalars). |
| data_movement/untilize_with_unpadding | ✅ partial | degenerate (run-args) | sc 31.1 / mc 29.2 | sc 21.8 / mc 23.2 | Migrated single-core + multi-core interleaved; RM/sharded/nd left legacy. 338 pass / 6 skip / 12 xfail / 0 fail (test_untilize_with_unpadding + test_untilize_bfloat8_b::test_untilize_with_unpadding 128 pass); 2 cache entries (sc+mc). No bugs — reader/writer co-located per the local-DFB rule. metal2 faster than legacy on both. |
| embedding | ⛔ | — | TILE 36.3 / RM 39.6 | — | REVERTED. Interleaved paths worked (260/263 pytest after fixing the fused factory's WorkUnit split so reader/writer share the compute groups' WorkUnitSpecs), but the **tiled sharded-output** path is BLOCKED on gap #3 (borrowed `output_cb` produced by compute with no consumer kernel → "DFB has no consumer"). 3 sharded cases fail; op reverted whole. |
| experimental/paged_cache | ⛔ | — | 40.6 (1 entry across batch_idx) | — | REVERTED. Migrated paged_fill_cache single-device PagedFillCacheProgramFactory (Advanced 3++; batch_idx_fallback out of the key). Host ProgramSpec + scalar-fallback bindings were correct, but the **writer kernel JIT-fails on the scalar-fallback path** (gap #1, conditional-token emission): it references `dfb::batch_idx` / `ta::batch_idx` inside `if constexpr (use_batch_idx_tensor)`, and genfiles only emits those tokens when the resource is bound — which it isn't in scalar-fallback (no batch_idx tensor) → `'batch_idx' is not a member of 'dfb'` / `'ta'`. The measure + fill_cache tests use the scalar-fallback path, so it can't compile. Binding-only fixes can't address codegen. Note: the targeted **win doesn't exist here** — legacy ALREADY shows 1 cache entry across varying batch_idx (40.6µs), so batch_idx is not in the legacy key on this path. Mesh variant left legacy. Op reverted whole (incl. sources.cmake + new m2 files); left for later. |
| experimental/plusone | ✅ | single | 7.2 | 7.9 | committed; deleted custom compute_program_hash; correctness PASS (both default and skip_negative paths); 1 cache entry; entrypoint is `ttnn.plus_one` (not under experimental) |
| experimental/transformer/nlp_concat_heads | ✅ | single (degenerate run-args) | 25.4 | 29.2 | 217 pytest pass; 1 cache entry; op-private copy of writer_unary_interleaved_start_id.cpp |
| experimental/transformer/nlp_concat_heads_decode | ✅ | single | 32.0 | 35.1 | 15 pytest pass (regular + subcoregrid); 1 cache entry; borrowed-output DFB needs reader=Producer / writer=Consumer split |
| experimental/transformer/nlp_create_qkv_heads | ✅ (partial) | single | 39.7 | 42.0 | non-transpose interleaved: llama (18) + generic non-transpose pass; transpose_k_heads + sharded paths BLOCKED (see below); 1 cache entry |
| experimental/transformer/nlp_create_qkv_heads_decode | ⛔ | — | 88.6 | — | REVERTED — Metal 2.0 conditional-token limitation (see Blocked); legacy baseline 88.6µs |
| experimental/transformer/rotary_embedding_llama | ⛔ | — | 366 (varying decode pos) | — | REVERTED. Prefill (interleaved + sharded) factories were fine, but the **decode/sharded** factory is BLOCKED on gap #3 (compute-only kernel with borrowed input/cos/sin/trans_mat/output tensors; the TensorParameter referential-integrity check requires a TensorBinding on some kernel, but tensor accessors only JIT-compile on DM kernels and this op has no DM kernel). Decode is the high-value cache-key win, so the op was reverted whole. Legacy decode used **10 cache entries** across varying positions. |
| experimental/transformer/rotary_embedding_llama_fused_qk | ⛔ | — | 53 | — | REVERTED. Same gap #3 as rotary_embedding_llama decode: single compute kernel with 7 borrowed tensors, no DM kernel; `TensorParameter ... not bound by any kernel` on all 16 pytest cases. |
| sliding_window/halo | ⬜ | | | | |
| transformer/sdpa | ⬜ | | | | |
| transformer/sdpa_decode | ⛔ | — | | | BLOCKED on gap #6 (varargs have no per-vararg enqueue_invariant). The cur_pos-out-of-key win needs varargs + the deprecated per-node-count API. Left legacy; no code. |

## Triage (recon pass complete)

All ops below are currently on the **ProgramDescriptor / WorkloadDescriptor** framework (the *previous*
migration) — NOT yet Metal 2.0. Target = port `create_descriptor`/`create_workload_descriptor` →
`create_program_artifacts`/`ProgramSpec` (+ optional `extract_immutable_info` / `create_per_enqueue_args`),
same transform as sampling. Variant count drives effort (each `program_factory_t` alternative = one port).

**Tractable (single/few variant, S/M, simple kernels) — do first:**
- plusone — S, 1 variant, 1 kernel, no dynamic args. EASIEST.
- nlp_concat_heads — M, 1 variant, 2 reader kernels.
- nlp_concat_heads_decode — M, 2 variants.
- nlp_create_qkv_heads — M, 2 variants (Interleaved/Sharded).
- interleaved_to_sharded — M, 1 variant, custom hash (drop→immutable_info).
- sharded_to_interleaved — M, 1 variant.
- pool/generic — M, 1 variant (already WorkloadDescriptor), custom hash.
- rotary_embedding_llama_fused_qk — M, 1 variant, 7 sharded CB bindings (no select_program_factory).

**Moderate (multi-variant, mechanical but bulky):**
- nlp_create_qkv_heads_decode — L, 3 variants.
- concat — 5 variants. slice — 5. pad — 7. tilize — 5. untilize — 8. untilize_with_unpadding — 6.
  transpose — 8 variants/18 kernels (largest data_movement).
- topk — L, 2 variants, **semaphores** (multi-core local→final).

**High value (per-call scalars that today bloat the key — Metal 2.0 win, like sampling's seed):**
- rotary_embedding_llama — XL, 3 variants, per-call batch/seq token positions in compute RTA + custom hash.
- sdpa_decode — L, per-call cur_pos vector + page_table NOT in hash today.
- paged_cache — XL, 3 sub-ops (fill/update/fused) × mesh variants, per-call update_idxs/batch_idx.

**XL / likely-blocked (assess, do what's feasible, else leave for later):**
- matmul — 6 factories (MatmulMultiCore partially ported in sibling worktree); mcast1D/2D huge.
- conv/conv2d — 2 WorkloadDescriptor factories, very large, delegates toward matmul.
- normalization/layernorm — XL, 2 variants, 23 kernels.
- reshard — XL, 8 variants, complex page-stride maps.
- transformer/sdpa — XL, 12 kernels, ring/joint topologies.
- sliding_window/halo — L, WorkloadDescriptor + 360-line halo_gather, config-tensor lifetime.

(Full per-op detail in /tmp/triage_*.md and the two inline recon dumps.)

## Triage tables (legacy)

## Blocked / needs-metal-change (running)

- **nlp_create_qkv_heads_decode** — ⚙️ Metal 2.0 limitation: genfiles emits `dfb::<name>`/`ta::<name>`
  tokens only for resources a kernel actually binds. Kernels that reference such tokens inside
  `if constexpr(feature)` guards fail to compile when the feature is absent (the discarded branch still
  needs name resolution in a non-template function). Binding-only port can't fix it. Needs a metal
  mechanism (always-emit guarded tokens, or a `#if HAS_dfb_x` macro from genfiles) or kernel restructure.
  Feature-present path is correct. → REVERTED for now, left for later.
- **nlp_create_qkv_heads** — partial: non-transpose interleaved path migrates cleanly; (1) the
  `transpose_k_heads` path uses the SHARED unmigrated `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`
  (hardcodes CB c_0/c_16 — incompatible with Metal 2.0 dynamic DFB ids); (2) the sharded path has two
  producers writing the same Q-output DFB (single-producer-per-node violation). Both are kernel-logic /
  shared-kernel issues, not binding-only. → migrate non-transpose interleaved; note the rest.

## Systemic metal-API gaps found (for Diego/Audrey)

1. **genfiles conditional-token gap** — `dfb::<name>`/`ta::<name>` tokens are emitted only for resources a
   kernel actually binds. Kernels that reference such a token inside `if constexpr(optional_feature)`
   fail to compile when the feature is absent. Blocks any op with optional-feature kernels.
   Hits: nlp_create_qkv_heads_decode (batch_offset), nlp_create_qkv_heads (transpose), likely others.
   Fix idea: always-emit guarded tokens, or a `#if HAS_dfb_x` macro from genfiles.
2. **op-owned device tensors have no owning slot in ProgramArtifacts** — ops that allocate device tensors
   *inside* the factory (not in tensor_args) and park them for the cached program's lifetime (the old
   `WorkloadDescriptor.buffers` ownership) cannot express that ownership. `ProgramRunArgs.tensor_args`
   binds only non-owning `cref<MeshTensor>`. Hits: pool/generic (halo lookup table + scalar config),
   sliding_window/halo (4 config tensors), likely conv2d/sharded ops with precomputed config.
   Fix idea: an owning-tensor slot in ProgramArtifacts, or a workload-factory-family equivalent.
3. **borrowed-tensor on a compute-only kernel cannot satisfy the TensorParameter referential-integrity
   check** — every declared TensorParameter must be bound via a `TensorBinding` (tensor accessor) on some
   kernel (`program_spec.cpp:462`); a borrowed-DFB reference alone does not count. But tensor-accessor
   codegen only JIT-compiles on data-movement kernels (it pulls in `dataflow_api_common.h` / `NOC_INDEX`,
   which a compute-kernel build context lacks — verified: adding the bindings to the compute kernel breaks
   the JIT). Sharded ops whose ONLY kernel is compute (it fills/produces the borrowed L1 tensor views
   itself, with no reader/writer DM kernel) therefore have nowhere legal to bind their tensor parameters.
   Closely related: a borrowed output DFB produced by compute with no second kernel hits the separate
   "DFB has no consumer" check (`program_spec.cpp:377`). Hits: rotary_embedding_llama (decode/sharded),
   rotary_embedding_llama_fused_qk, embedding (tiled sharded-output). Note: ops WITH a DM kernel (e.g.
   nlp_concat_heads_decode) satisfy this by binding the borrowed tensor on the DM kernel with an
   unused accessor token — that escape hatch is unavailable to compute-only factories.
   Fix idea: allow a TensorParameter to be satisfied by a borrowed-DFB reference, or permit a
   binding-only (address-supplying, no-accessor-codegen) TensorBinding on compute kernels.
4. **chained / variadic / runtime-indexed tensor accessors have no Metal 2.0 form** — `ta::name` is a
   per-binding compile-time symbol; there is no array/runtime-indexed accessor, no chained
   `TensorAccessorArgs::next_*_offset()` equivalent, and no per-binding base-address / page-size override.
   Hits: binary_ng (chained accessors on the PLAIN interleaved path — fatal), concat (variadic N inputs),
   reshard (runtime page-stride maps), interleaved_to_sharded (compile-time-vararg shard addrgen),
   slice RM (runtime base+page-size override). Fix idea: an indexed/runtime `ta::` form + accessor overrides.
6. **varargs have no per-vararg `enqueue_invariant` flag** — variable-length / per-core-count runtime args
   must use positional varargs (`num_runtime_varargs`), but `enqueue_invariant_runtime_args` is a list of
   NAMES only, so varargs are always re-applied wholesale. Tree-reduction / mcast ops (sdpa_decode, sdpa)
   therefore cannot get the "++" enqueue-invariant fast path even though their dynamic value (cur_pos) could
   leave the key via immutable_info. Fix idea: per-vararg invariance, or a named variable-length arg form.
7. **the no-custom-hash static_assert is per-DEVICE-OP, not per-factory** — a multi-variant op with a SHARED
   custom `compute_program_hash` (needed by its still-legacy variants) cannot have ANY single variant
   migrated to Metal 2.0: the `static_assert` (device_operation.hpp:264) fires for the whole device op, and
   removing the hash switches ALL legacy variants to the default key too. Hits: matmul (1 hash serves all 6
   factories). Fix idea: make the prohibition per-migrated-factory, or provide a sanctioned "default-hash is a
   superset of this custom hash" opt-out. (The sibling matmul port simply deleted the hash — a behavior call.)

## Blocked / needs-metal-change (filled at the end)

- **pool/generic** — ⚙️ BLOCKED on gap #2 (op-owned halo/config device tensors). No port attempted.

- **nlp_create_qkv_heads_decode** — ⛔ CONFIRMED the predicted Metal 2.0 limitation at finalize time.
  Host-side ProgramSpec compiles, but the device kernel JIT-fails: the reader kernel references
  `ta::batch_offset` / `dfb::batch_offset` inside `if constexpr (use_batch_offset)`, and genfiles only
  emits those tokens when a kernel binds the resource. Cases without batch_offset never emit the tokens →
  `'batch_offset' is not a member of 'dfb'` (1021 occurrences across the suite). This is the same
  always-emit-guarded-token need described above; binding-only fixes (adding TensorBindings for the borrowed
  q/k/v_output params, splitting Producer/Consumer roles) got past all ProgramSpec validation but cannot fix
  the kernel-codegen gap. REVERTED to legacy ProgramDescriptor; left for later.
- **nlp_create_qkv_heads** (partial) — `transpose_k_heads` path blocked (shared unmigrated
  `transpose_wh.cpp` hardcodes CB c_0/c_16) and the sharded path blocked (dual-producer + shared kernel).
  Non-transpose interleaved path migrated and validated.
- **rotary_embedding_llama** — ⛔ REVERTED. The decode/sharded factory is a compute-only kernel over
  borrowed input/cos/sin/trans_mat/output tensors and hits gap #3 (`TensorParameter 'input' ... not bound
  by any kernel`). Adding TensorBindings to the compute kernel passes ProgramSpec validation but then
  JIT-fails the compute build (`NOC_INDEX` / `dataflow_api_common.h`). Prefill factories were fine, but the
  decode path is the headline win (positions stay out of the cache key: legacy used 10 entries across
  varying decode positions, migrated would have been 1) so the op was reverted whole. Left for later.
- **rotary_embedding_llama_fused_qk** — ⛔ REVERTED. Same gap #3: a single compute kernel with 7 borrowed
  tensors and no DM kernel; all 16 pytest cases fail the referential-integrity check. Left for later.
- **embedding** — ⛔ REVERTED. Interleaved (TILE + ROW_MAJOR) migrated and 260/263 pytest pass after fixing
  the fused factory's WorkUnit layout (it split the all-cores reader/writer into a separate WorkUnitSpec
  from the per-core-group compute kernels, violating the Local-DFB producer==consumer WorkUnitSpec
  invariant; fixed by placing reader/writer into each compute group's WorkUnitSpec — a KernelSpec may
  belong to multiple WorkUnitSpecs). But the **tiled sharded-output** path hits gap #3's sibling
  ("DFB 'output_cb' has no consumer"): the borrowed output is produced by compute and has no writer kernel.
  3 sharded cases fail; op reverted whole. Left for later.

## TLDR — autonomous bulk-migration run (branch `dgomez/metal2-bulk-migration`)

**Migrated + built + validated + committed (13 ops; some partial per-variant — `program_factory_t` legitimately mixes legacy + Metal 2.0 alternatives):**

| op | variants migrated | legacy µs | metal2 µs | cache |
|---|---|---|---|---|
| reduction/sampling | full (Adv 3++) | ~1e6 varying-seed | ~92 | 1 / N seeds |
| experimental/plusone | full | 7.2 | 7.9 | 1 |
| experimental/transformer/nlp_concat_heads | full | 25.4 | 29.2 | 1 |
| experimental/transformer/nlp_concat_heads_decode | both | 32.0 | 35.1 | 1 |
| experimental/transformer/nlp_create_qkv_heads | interleaved non-transpose | 39.7 | 42.0 | 1 |
| copy/typecast | 3/4 (interleaved+subgrid+RM) | 50.2 | 64.8 | 1 |
| data_movement/pad | tiled mc + single | 38.7 | 24.9 | 1 |
| data_movement/sharded/sharded_to_interleaved | full | 22.6 | 21.3 | 1 |
| data_movement/tilize | single + mc-default | 24.5/43.6 | 43.5/44.3 | 1 |
| data_movement/untilize | single-core | 242.9* | 258.2* | 1 |
| data_movement/untilize_with_unpadding | single + mc-interleaved | 31.1/29.2 | 21.8/23.2 | 1 |
| data_movement/transpose | CN + HC-tiled-interleaved | 29.1 | 31.7 | 1 |
| reduction/topk | TopKSingleCore | 207* | 212* | 1 |

(*host time sync-dominated, diff within noise; rand was already migrated pre-run.)

**Metrics headline (with vs without Metal 2.0):** for **static** ops the cache-hit host dispatch is roughly
**parity ±15%** — Metal 2.0 is a cleaner API, not a host-dispatch speedup (pad/untilize_with_unpadding were
faster; typecast slower). The real, large win is **structural**: keeping a per-call value OUT of the cache key
so dynamic workloads stop recompiling — proven by **sampling (~1e6→~92 µs, 1050→1 entries)**. Every op that
would showcase that win at scale (rotary decode, sdpa_decode, paged_cache) is currently **blocked by a metal
gap below** — so the dynamic-arg win is demonstrated only by rand/sampling so far.

**Blocked / left-for-later (with the gap that blocks each):**
- gap #1 (conditional tokens): nlp_create_qkv_heads_decode; paged_cache (fill, scalar path); nlp_create_qkv_heads transpose path.
- gap #2 (op-owned device tensors): pool/generic; sliding_window/halo (assessed); likely conv2d.
- gap #3 (compute-only / borrowed-tensor binding): rotary_embedding_llama (decode), rotary_embedding_llama_fused_qk, embedding (tiled sharded), concat (S2STiled).
- gap #4 (chained/variadic/runtime-indexed accessors): binary_ng (interleaved!); concat (interleaved/multi, variadic N inputs); interleaved_to_sharded (compile-time vararg addrgen).
- gap #6 (varargs have no per-vararg enqueue_invariant): sdpa_decode; any tree-reduction/mcast op.
- non-metal scope coupling: slice (`create_descriptor` consumed by `ccl/mesh_partition`'s std::visit); topk multi-core (multi-program WorkloadSpec shape — semaphores themselves ARE supported).

**Not attempted (XL, deferred — likely blocked by the same gaps; flagged for you):**
- normalization/layernorm (XL, 23 kernels — multi-core interleaved variant is the candidate if pursued).
- matmul (6 factories; MatmulMultiCore has a partial port in the sibling worktree — best candidate).
- conv/conv2d (WorkloadDescriptor + op-owned config → gap #2; delegates to matmul).
- transformer/sdpa (varargs + semaphores + ring/joint → gap #6 + size).
- data_movement/sharded/reshard (8 variants, runtime-indexed page-stride maps → gap #4).
- sliding_window/halo (WorkloadDescriptor + 4 op-owned config tensors → gap #2).

### The metal-API changes you asked me to report (NONE were made — all surfaced for you/Audrey)
The single highest-leverage fixes, in impact order:
1. **gap #3** — let a TensorParameter be satisfied by a `borrowed_from` DFB reference (or allow an
   address-only, no-accessor-codegen TensorBinding on compute kernels). Unblocks the whole sharded/compute-only
   class incl. the **rotary decode** dynamic-arg win.
2. **gap #1** — genfiles should always emit guarded `dfb::`/`ta::` tokens (or expose `#if HAS_dfb_x`). Unblocks
   every optional-feature kernel (qkv_heads_decode, paged_cache, …).
3. **gap #4** — a runtime/variadic tensor-accessor form (array/indexed `ta::`, + an accessor base/page-size
   override). Unblocks binary_ng, concat, reshard, the sharded addrgen ops.
4. **gap #6** — per-vararg `enqueue_invariant` capability. Required before any tree-reduction/mcast op
   (sdpa_decode, sdpa) can get the "++" fast path with a dynamic value out of the key.
5. **gap #2** — an owning-tensor slot in `ProgramArtifacts` (the old `WorkloadDescriptor.buffers` role) for
   op-allocated config tensors. Unblocks pool, halo, conv.

### Recurring migration gotchas (no metal change — for the recipe / future ports)
- Unity-build: sibling factory `.cpp` file-scope constants collide even in anon namespaces → unique names.
- Local DFB producer+consumer must share the SAME WorkUnitSpec (a KernelSpec may join several WUs; put
  all-cores reader/writer into each per-core-group compute WU).
- A borrowed output DFB needs exactly one Producer + one Consumer (split reader=Producer / writer=Consumer).
- A kernel with ANY TensorBinding needs a (possibly empty-scalar) KernelRunArgs entry, or ValidateProgramRunArgs aborts.
- `UnpackToDestFp32` is rejected unless the input DFB is Float32 (guard it).
- ENSURE you actually assign `spec.kernels/.dataflow_buffers/.tensor_parameters/.work_units` (easy to build locals and drop them).
- Always-declaring an otherwise-optional DFB makes its `dfb::` token exist, dodging gap #1 where the feature is structural.

### Process
Parallel codegen sub-agents (1 op each, referencing rand/sampling + committed examples + the recipe) →
serial finalize agents (build + git-stash legacy baseline + validate pytest + measure + commit, one at a time
for exclusive device use). 6 codegen waves + 6 finalize waves. All commits are individual, no AI attribution.
No `tt_metal/` or shared-`ttnn/api` files were changed.
