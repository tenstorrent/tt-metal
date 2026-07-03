# In Place Halo — Re-implementation Project (single-source status)

Branch: `wransom/in_place_halo_redo` (based on `origin/main`).
Owner: wransom. This file is the single source of truth for the effort; older per-PR
notes and memories may be stale — **the code at current HEAD (and the archived
pre-removal source) is the oracle.**

> Status legend: ✅ done · 🔜 next · ⏳ in progress · ❓ open question

---

## 1. Goal and the bar to beat

Make the Halo op **in-place**: the output shard buffer *contains/overlaps* the input
shard buffer, so we do not pay L1 for a separate input buffer. This is a pure **L1
savings** feature (it is deliberately *not* a perf win — perf is traded away for memory).

In-place Halo shipped in 2025 (PRs below), then was **entirely removed** in
`#31516` (commit `f5964193d6a`, 2025-11-10). The removal PR's verdict is the bar we
must clear this time:

> *"In Place Halo is buggy, non-performant, and in most practical cases uses more
> memory than normal halo."*

So the redo must be: (a) **correct** (no sub-stick corruption / hangs across the full
matrix), and (b) actually deliver an **L1 net win** in practical cases — the third
clause ("uses more memory than normal halo") is the strategic failure and must be
understood, not just the corruption bugs.

### Why the old version could *use more* memory (the key strategic question ❓)
Even though the output subsumes the input, the in-place path adds auxiliary L1:
a **remote temp CB** (`max_ref_size * shard_width * nbytes`), a **pad CB**, an
**untilize temp CB** for wide tensors, and the config CBs. If those auxiliaries exceed
the input-shard size we would have saved, in-place is a net loss. **Quantifying the
break-even and gating in-place ON only when it truly wins is a first-class goal of the
redo** — the old version applied it too broadly. See §7.

---

## 2. History (the full PR arc — more than the 5 originally cited)

Chronological (oldest → newest), from `git log origin/main`:

| Commit | PR | What |
|---|---|---|
| `e8a36c2` | #19056 | **In Place Halo** (initial) |
| `f41a3fe` | #19787 | **Revert** #19056 — *hangs on BH for conv/pool* |
| `fe7b175` | #19834 | **Re-land**: fixed core-grid-shape assumptions + worker NOC coords for `noc_semaphore_inc` |
| `c046def` | #19988 | Fix missing in-place nightly tests (lost to a commit interaction) |
| `4ac4504` | #20225 | **Halo fix** — hang for some inputs: semaphore core count + `noc_async_atomic_barrier` before exit |
| `1bbe8ad` | #20473 | **In Place Untilize** — tiled inputs, untilize into dst buffer, temp CB for wide tensors (one tile-row at a time) |
| `b897b32` | #20878 | **Multicasting on WH/BH** — BH runtime-arg overflow for core list → multicast + rectangular grid + noop cores |
| `5c26e57` | #22852 | **Optimizations** — split reader NC/BR for remote→temp and local copies |
| `a3d7cd0` | #27333 | Max Pool fixes + **In Place Halo small stick sizes on BH** (alignment in `in_out_shard_size_delta`) |
| `1f02912` | #28437 | **Bounded core grids** + shard width > stick size: `num_cores_x` from bounding box (wrong NOC coords → hang) |
| `b1ad741` | #30644 | **New L1 alignment** — L1 alignment differentiated from DRAM; overlap check updated |
| `f596419` | #31516 | **Remove In Place Halo** (the final removal) |
| `f8855fe` | #33389 | Remove `in_place` from `Conv2dConfig` (dead-code cleanup) |

Also relevant (separate lineage, cautionary): `#23650` "Split halo gather padding between
BR/NC cores" was **reverted by #23818** — broke ttnn cpp tests on N300 (device FW init
failure). Split-reader work has a history of destabilizing; treat with care.

**Structural gap:** the halo op was migrated to the **ProgramDescriptor framework in
`#45075` (`b2af0cd`), which landed *after* the removal.** So the archived in-place code
predates ProgramDescriptor. The device kernel logic ports almost verbatim; the program
factory and config generation must be rewritten against the new framework.

Archived pre-removal source (the complete final implementation, all fixes applied) is at
commit `c852f3aa` (parent of the removal) and saved for study in the scratchpad.

---

## 3. What in-place Halo is (mechanism)

Normal halo: `input_shard` (L1 buffer A) → op writes `output_shard` (L1 buffer B),
where B = local sticks + halo (remote) sticks + padding. Two buffers live at once.

In-place halo: allocate B to **overlap** A. Host deallocates the sharded input tensor
so the output can be reallocated over the same L1 region. The invariant enforced on
host:

```
src_buffer->address() == dst_buffer->address() + delta
```

i.e. the input shard sits `delta` bytes *into* the (larger) output buffer. The output's
front `delta` bytes and its tail are filled with halo/padding; the input's own sticks
are moved (possibly in place, possibly to other cores) to their final positions.

- Row-major: `delta = in_out_shard_size_delta * out_stick_nbytes`
- Tiled: `delta = out_buffer.aligned_size_per_bank() - in_buffer.aligned_size_per_bank()`
  (untilize writes directly into the dst buffer, so the stick delta is 0)

Because input and output overlap, **the order of every stick copy matters** — this is
the entire source of difficulty.

---

## 4. The algorithm

### 4a. Device kernel — `halo_gather_in_place.cpp` (archived, 460 lines)

Two DM cores run the same kernel with a `main_thread` compile-time arg (BRISC=main,
NCRISC=secondary). Ordering discipline (from the kernel's own header comment):

1. **(both cores) copy remote sticks from my shard → my local `temp` CB.** Staging
   *first*, before any overwrite, is what protects a core's source data from being
   clobbered by another core's local copy. Remote entries alternate between the two DM
   cores (`remote_entry_count % 2 != main_thread`); the non-owning core just advances
   the temp pointer to keep the shared temp layout consistent.
2. **(both cores) copy local sticks from my shard → their destinations.** Overlapping
   in-place copy: bulk copy only when chunks provably don't overlap; otherwise
   **stick-by-stick** with direction chosen by
   `is_forward_copy = dst_local_idx > (src_local_idx + in_out_buffer_start_delta)`
   → copy **last-to-first** when dst is "in front" of src (memmove semantics), else
   first-to-last. The two cores split each stick front-half / back-half and stay in
   lockstep per config entry via a `sync_cb` ping-pong (order preservation).
3. **Global barrier**: each core's main thread increments a semaphore on the top-left
   core; the `cast_core` waits for `num_active_cores` increments then
   `noc_semaphore_set_multicast`es the release to the whole rectangular grid. Gates
   step 5 so no core writes into another core's buffer before that core has vacated its
   source sticks.
4. **(NCRISC, if padding) write padding sticks** — concurrent with step 5.
5. **(BRISC; or both if no padding) copy remote sticks from temp → final destinations**
   on other cores.
6. Exit with `noc_async_read_barrier()` + `noc_async_write_barrier()` +
   **`noc_async_atomic_barrier()`** (the #20225 hang fix — atomics must drain before exit).

Untilize-in-place for **wide tensors**: the compute kernel untilizes into an
`untilize_temp_cb`, and the reader restages it **one tile-row at a time** into the
in/out buffer (`TILE_SIZE_BYTES * tile_cols` per iteration) to avoid tilized vs
untilized data colliding in the shared buffer. Narrow tensors use `pack_untilize`
straight into the dst buffer.

### 4b. Host config generation — `sliding_window.cpp` (archived)

- **local config, in-place branch**: emit **forward-ordered** entries while
  `dst_start <= src_start + in_out_shard_size_delta` (the non-overlap region), then
  emit the remaining (overlap-region) entries in **reverse order** (highest src first).
  This ordering is what pairs with the kernel's per-stick direction to make overlapping
  moves safe. In-place does **not** pre-split local entries across the two cores in the
  config (all go to one vector); the split happens in the kernel via half-sticks.
- **remote config, in-place branch**: all transfers listed together (no config split),
  and `ref_size` accumulates transfer lengths per core → `max_ref_size` sizes the
  remote temp CB.
- **pad config, in-place branch**: entries kept together (not core-split).

### 4c. Program factory — `untilize_with_halo_program_factory.cpp` (archived)

- Output CB and untilize-out CB both `.set_globally_allocated_address(*dst_buffer)`.
- Overlap invariant `TT_FATAL(src == dst + delta)`; `remote_read` disallowed in-place.
- Remote temp CB sized `max_ref_size * output_shard_shape[1] * out_nbytes`.
- Rectangular core grid + noop cores for clean multicast; `semaphore_id` on the
  rectangular set; `cast_core = core 0` drives the multicast; per-core RT args
  `{noop_core, cast_core}` for main thread, `{noop_core, false}` for secondary.
- Two kernels created (main / secondary) on the rectangular core range.

---

## 5. Gotcha catalog (the corruption/hang classes)

From PR bodies, code, and the **mined review threads** (reviewer quotes cited). Ranked by
how much pain they caused historically. **Note the pre-history**: the *first* landing was
`#18329` (before #19056), reverted for BH+WH regressions and re-landed by fixing "core
grid shape assumptions and the worker NOC coordinates used for `noc_semaphore_inc`". That
one sentence names classes 3 and 4, and both recurred later. This feature has been
reverted/removed **three times** (#18329, #19787, #31516) — respect it.

1. **BH vs WH alignment breaks the skip-copy early-exit → silent corruption (#27333).**
   The single most dangerous bug. The local-copy path has an early exit: if
   `src + in_out_shard_size_delta == dst`, the stick is already in place, skip the copy.
   But BH aligns to **64B** vs WH's **32B**, so a delta of 11 sticks (32B sticks) rounds
   to 12 on BH; if the delta arg isn't recomputed per-arch the copy is wrongly skipped on
   *both* archs and "chaos ensues on blackhole since the data wasn't actually where it
   needed to be." **The delta must be computed with the actual per-arch L1 alignment.**
2. **L1 alignment distinct from DRAM alignment (#30644).** The overlap check
   `src == dst + delta` must use **L1** alignment; delta is `stick_delta` (row-major) vs
   `buffer_delta = out.aligned_size_per_bank() - in.aligned_size_per_bank()` (tiled).
   For **tiled inputs the stick delta is 0** because untilize writes into the dst buffer.
3. **Buffer-overlap invariant must be asserted (#19056 review → #30644).** The
   reallocated output must sit on top of the deallocated input; "there are scenarios
   where it might not, and we don't support this" — assert it, don't assume it.
4. **conv2d/pool must NOT deallocate the input when in-place is on.** In-place makes
   input and output the *same buffer*; a `deallocate_activation` after halo is a
   double-dealloc → PCC errors then OOM. (Root-caused by @pavlejosipovic on #22852.)
5. **Local overlapping-copy direction.** Copy stick-by-stick when chunks overlap;
   **reverse** (last-to-first) when `dst > src + delta` (dst "in front"), else forward.
   Direction depends on the *alignment-correct* delta — so a stale delta (class 1)
   manifests here as corruption. Host config emits forward-region entries then
   reverse-region entries. Test intent: "reversed reads on some cores, forward on others"
   (maxpool N=8 112×112, N=32 264×40).
6. **Remote must stage to temp before local copies.** A local copy that runs first can
   destroy a remote source stick. Global semaphore barrier gates the from-temp
   distribution until all cores have vacated their sources.
7. **Core-grid-shape / NOC-coordinate hang (#28437, and #18329's original revert).**
   Bounded/multi-row-partial grids (e.g. `({0,0},{2,2})`) give wrong NOC coords → hang.
   Compute `num_cores_x` from the **core bounding box**, never `compute_with_storage_grid_size()`.
8. **Runtime-arg overflow on BH → multicast (#20878).** Passing a per-core NOC/active-core
   list as runtime args hit the **256-arg limit** on large BH grids
   (`260 ... too large. Max allowable is 256`). Use a rectangular grid + noop cores +
   multicast release instead of a core list. Sender/receiver role can't be a constexpr
   (one kernel plays both), so noop is a runtime arg.
9. **Atomic barrier before kernel exit (#20225).** Omit `noc_async_atomic_barrier()` and
   the atomic-inc sync races the next program → hang. Confirmed *not* a DPRINT artifact.
10. **NC/BR single-slot sync-CB double-push race (#22852).** The intra-tensix NC↔BR sync
    CB has one slot; without a `cb_reserve` on the secondary thread before its push, both
    threads can push → 2 pages into a 1-slot CB → the main thread races ahead before
    local copies finish. Serialize with reserve-before-push. (A CB, not a semaphore, is
    used because NC/BR are on the *same* tensix.)
11. **Untilize tilized/untilized collision (#20473).** Wide tensors use the plain
    untilize kernel (not pack_untilize), which would overwrite not-yet-read tiles in the
    shared buffer — stage **one tile-row at a time** through a temp CB.
12. **Post-untilize dtype for scratch/temp (#27333).** After untilize, `bfloat8_b`
    becomes `bfloat16`; size the remote-temp / untilize-temp CBs for the **post-untilize
    (bf16) width**, not the bf8 input width.
13. **Do NOT split padding writes across BR/NC (#23650 → reverted #23818).** That split
    broke N300 device-FW init (`failed to initialize FW`). Splitting the *gather/copy*
    work is fine; splitting the *padding fill* destabilized N300.
14. **Net-memory regression (the removal verdict).** Auxiliary CBs (remote temp, pad,
    untilize temp, config) can exceed the saved input-shard size → in-place uses *more*
    L1. This is *why it was removed*. Gate in-place ON only when it net-saves (§7).

---

## 6. Test strategy

Port the archived `tests/ttnn/unit_tests/operations/pool/test_maxpool2d.py` structure —
its coverage was purpose-built to catch these bugs:

- `in_place: [True, False]` parametrization on **height / width / block** sharded, over
  `bfloat16` and `bfloat8_b`, so identical shapes run both ways and any divergence is a
  bug.
- Deliberate edge cases: partial grid → noop cores; ceil-mode output-shape adjustment;
  **specs that force reversed-on-some-cores / forward-on-others local reads**; wide vs
  normal in-place untilize; non-tile-multiple NHW; extreme channel counts (C=1, 7, 16,
  >800) to exercise stick-size / alignment paths.
- **Structured input tensors** (per user guidance): use inputs where each stick's value
  encodes its origin (core, stick index) so corruption localizes to a specific
  sub-stick/race, not just "PCC failed". This is how the original edge cases were found.
- Cross-arch: WH now, **BH later** (a BH machine session is planned). Corruption can be
  WH-clean but BH-broken (class 1, 7) — do not declare done on WH alone.

---

## 7. Current-main structure & the hook points (from the structure survey)

Halo is now built via **`WorkloadDescriptor`/`ProgramDescriptor`** (declarative
`CBDescriptor`/`KernelDescriptor`) — no `CreateProgram`, no `operation::run`, and **no
in-place remnant** (only `halo_gather.cpp` in `kernels/dataflow/`). Exact anchors
(HEAD `004d4778daf`):

- **Op struct** `HaloDeviceOperation` (`halo/device/halo_device_operation.hpp:19-31`):
  attrs = `HaloParams` (`..._types.hpp:13-27`); single-tensor `tensor_args_t = Tensor`;
  factory = `std::variant<UntilizeWithHaloProgramFactory>`.
- **Output allocation** `create_output_tensors` (`halo_device_operation.cpp:93-97`) →
  `create_device_tensor(...)` — **unconditionally a fresh buffer**. Output shard is
  generally **taller** than input (H grows via halo; **width preserved**,
  `compute_output_specs` `.cpp:73-75`); layout forced ROW_MAJOR (`.cpp:89`).
- **Factory** `UntilizeWithHaloProgramFactory::create_workload_descriptor`
  (`untilize_with_halo_program_factory.cpp:388-515`, `build_halo_program` `:83-384`).
  CBs via `add_cb(...)` helper (`.cpp:60-78`) with an optional `Buffer*` for
  global-allocation. **The two aliasing lines**: `src_cb`→`input_tensor.buffer()`
  (`.cpp:102,152`), `out_cb`→`output_tensor.buffer()` (`.cpp:103,166`).
  `output_tensor` arrives as **`Tensor&`** (mutable) at the adapter
  (`mesh_device_operation_adapter.hpp:488-489`) and factory (`.cpp:391`).
- **CBs today**: `src`, `out`, `pad0/1`, `untilize_out0/1` (double-buffered, only if
  `!skip_untilize`; `skip_untilize = input RM`), `padding_config0/1`, `gather_config0/1`.
  **Local and remote are now MERGED into one `gather_config`** (split internally by
  `src==dst` vs `src!=dst`) — different from the old 3-way pad/local/remote split.
- **Gather kernel** `halo_gather.cpp`: `run_halo_gather` (`:186-255`) walks a flat
  `gather_config` (`[num_segments]`, per route `[dst_noc_x,dst_noc_y,transfers]`, per
  transfer `[src_off,dst_off,size]`) and does `noc.async_write` src→dst
  (`write_stick_async` `:146-172`). **No semaphores / multicast / atomic barrier today.**
  Split reader via `block_start_offset` (0/1) + `block_stride=2` (interleave blocks
  between RISCV_0/RISCV_1), plus a shared-read-pointer discipline (`:339-347,367-372`).
  `static_assert(!remote_read)`.
- **Config generation** `sliding_window.cpp`: `generate_halo_kernel_config_tensors`
  (`:775-971`) builds `GatherConfig/GatherRoute/GatherHeader/GatherTransfer` (the record
  types reviewers once asked for now exist), then for tiled input
  `quantize_transfers_along_block_boundaries` → `reorder_transfers_globally` →
  `divide_blocks_between_cores` (`:886-903`). **This reorder/reblock stage is exactly
  where the in-place write-before-read ordering must be introduced.**
  `construct_on_host_config_tensor` (`:1218`) + `move_config_tensor_to_device` (`:1236`,
  height-sharded L1_SMALL) push configs to device; the factory parks them on
  `workload_descriptor.buffers` so they outlive cached programs (`.cpp:466-487`).
- **Host entry** `halo()` (`halo/halo.cpp:11-30`): `(input, config, compute_config,
  pad_val, remote_read, transpose_mcast, is_out_tiled, config_tensors_in_dram)` — **no
  in-place / output / memcfg param**. Add one and thread into `HaloParams`.
- **Callers**: conv2d (`conv2d.cpp:262-300`) and generic pools
  (`generic_pools.cpp:332-356`) both call `ttnn::halo`, then optionally
  `deallocate_activation`/`deallocate_input` and `reallocate_halo_output` (`ttnn::move`).
  These lifecycle knobs are what in-place interacts with (class 4).

## 7b. Porting plan (execution order)

1. **Host plumbing**: add `in_place` to `halo()` → `HaloParams`; thread `in_place`
   through `generic_pools` and `conv2d`. Suppress input-dealloc + `reallocate_halo_output`
   when in-place (class 4).
2. **Output aliasing**: make `create_output_tensors` (or the factory via the mutable
   `Tensor&`) return an output whose buffer *is* the input buffer at the right offset.
   The input must be **allocated at output size** up front (caller over-allocation, as in
   the original), then the output aliases it. Assert `src == dst + delta` with **L1
   alignment**, `stick_delta` (RM) vs `buffer_delta` (tiled, delta 0) (classes 1-3).
3. **Config ordering for in-place**: in `generate_halo_kernel_config_tensors`, add the
   in-place ordering — forward-region then reverse-region for local (src==dst) transfers,
   keep remote (src!=dst) transfers together sized for the temp CB (`max_ref_size`).
   Decide: extend the merged `gather_config`, or re-introduce a local/remote split for the
   in-place path only.
4. **In-place gather kernel** `halo_gather_in_place.cpp`: port the archived kernel;
   reconcile with the current split-reader model (block interleave vs half-stick) and the
   `experimental::CB` API. Include: stage-to-temp, overlap-direction copy, global
   semaphore+multicast barrier, padding after barrier, `noc_async_atomic_barrier` at exit,
   NC/BR sync-CB with reserve-before-push (classes 5-11).
5. **Factory (ProgramDescriptor)**: bind out/untilize-out CBs to `dst_buffer`; add
   remote-temp CB (post-untilize dtype, class 12); rectangular grid + noop cores +
   semaphore + multicast; two DM KernelDescriptors + compute; RT args
   `{noop_core, cast_core}` / `{noop_core, false}`; `num_cores_x` from **bounding box**
   (class 7).
6. **Net-L1 gate** (class 14): compute saved-input vs added-auxiliary-CB L1 and only
   engage in-place when it net-saves; otherwise transparently fall back to normal halo.
7. **Tests** (§6), WH first, then BH.

**Sequencing note**: do a **row-major, height-sharded, no-untilize** vertical slice
end-to-end first (simplest: delta is stick-based, no untilize temp, no block sharding
NOC subtleties), get it green with structured inputs, then widen to tiled/untilize,
width/block sharding, and the net-L1 gate.

---

## 9. L1 economics & the remote-temp elimination hypothesis (the path to the "good outcome")

**Mission (per user):** in-place halo was removed because it net-*cost* L1 in most
practical cases — but that was the team's best understanding, which the user themselves
argued. The redo's job is to **either confirm that (bad outcome) or refute it and produce
a better solution (good outcome)** — i.e. genuinely test whether L1 savings are
achievable, not just re-implement + gate. First target op: **MaxPool, height-sharded.**

### 9a. The accounting (in-place vs normal halo, per core)

Normal halo holds **two** L1 buffers: `input_shard` (size `in_nsticks * aligned_page`)
and `output_shard` (size `out_nsticks * aligned_stick`), plus config CBs + a pad CB
+ (if tiled) an untilize-out CB.

In-place halo holds **one** output-sized buffer with the input embedded in it. So:
- **Saved:** the separate `input_shard` buffer (≈ `in_nsticks * aligned_page`).
- **Added vs normal:** the **remote-temp CB** (`max_ref_size * shard_width * out_nbytes`,
  where `max_ref_size` = max outbound-halo sticks over all cores), and a possible second
  pad CB / wide-untilize temp CB.
- **Shared (not a delta):** the padding_config / gather_config CBs exist in both paths;
  the untilize-out CB is subsumed by writing untilize straight into the dst buffer.

So the net question reduces to: **is `remote_temp_CB` (+ minor extras) smaller than the
saved `input_shard`?** For large shards / small kernels (the ResNet/UNet 3×3 common case)
the outbound halo is a *small fraction* of the shard → temp ≪ input → in-place should
**win**. For small shards / large windows the halo fraction is large → temp can approach
or exceed the input → in-place **loses**. That fraction is the whole ballgame.

### 9b. First hypothesis (remote-temp is eliminable) — TESTED & REFUTED

Attractive idea: the temp only exists so a sender's outbound halo survives its own local
copies; if a remote halo write always landed in a region the receiver never reads, we
could reorder to **remote-first → barrier → local** with *no temp*, deleting the dominant
added cost. That requires halo-destinations ∩ receiver-input-region = ∅.

**It is not ∅.** Two facts from current-main code kill it:
- `generate_shard_boundaries` (`sliding_window.cpp:454-514`) makes each output shard a
  **contiguous window of the padded/haloed input space** → the output is **spatially
  ordered**: `[top halo/pad from core i-1][core i's own local sticks][bottom halo/pad from
  core i+1]`, *not* segregated `[all-halo][all-local]`.
- The delta is `in_out_shard_size_delta = aligned(out_nsticks) − aligned(in_nsticks)`
  (archived `halo_device_operation.cpp:141-147`), so the input is placed at the **tail**
  `[delta, out_nsticks)` of the output buffer.

Therefore the **bottom-halo region sits at the end of the output = inside `[delta,
out_nsticks)` = the receiver's live input region.** A remote-first write from core i+1
would overwrite core i's input before core i reads it → corruption. The original's
**stage-remote→temp, local, barrier, distribute-remote-after** ordering is genuinely
required. (This is the recurring theme: the "obvious" simplification has a real reason it
doesn't work. Verified before writing any kernel — cheap.)

### 9c. So where the "good outcome" actually lives (the real experiment)

The temp stays, but per-buffer accounting still favors in-place when the temp is small:
`remote_temp = max_ref_size · shard_width · nbytes`, and `max_ref_size` = max **outbound
halo** sticks per core ≈ a few rows for small kernels. Net saving ≈
`input_shard − remote_temp − extra_CBs`. For **large shards / small kernels** (ResNet/UNet
3×3) the outbound halo is a small fraction of the shard, so this should be **positive**.

So why did it net-*lose* "in most practical cases"? The suspects, to measure empirically:
1. **Config-CB bloat**: in-place reordering (`reorder_transfers_globally` "may add
   duplicate routes") can enlarge the gather/pad config tensors vs normal halo.
2. **Allocation lifetime / over-allocation**: in-place needs the *input* allocated at
   *output* size up front. If the producer of that input then holds an output-sized buffer
   for its whole lifetime (or the input must persist for a residual), the "saving" is paid
   back elsewhere — the win only materializes when the input is truly transient.
3. **Shape distribution**: models may shard into *small* per-core shards where the halo
   fraction (and thus temp) is large, erasing the saving.

**The experiment (needs the implementation):** build the MaxPool height-sharded slice
(temp-based, correct), then measure **actual peak L1** for in-place vs normal across the
`test_maxpool2d.py` shapes AND a couple of real conv layers, decomposed into
input/output/temp/config contributions. That win/lose table — not more reasoning — is
what confirms the bad outcome or reveals the good one. If in-place wins per-buffer but
loses in a model, the lever is suspect #2 (lifetime), which is a *scheduling/gating*
problem, not a kernel one.

### 9e. FIRST EMPIRICAL RESULT — host-only economics probe (2026-07-02)

Ran a host-only gtest (`InPlaceHaloEconomics.MaxPoolHeightSharded` in
`tests/ttnn/unit_tests/gtests/test_sliding_window_infra.cpp`) over the archived MaxPool
height-sharded shapes, computing from the real config generation the width-independent
verdict `maxref < in_nsticks_per_core` (SAVE). `in/core` = input sticks/core (saved
buffer); `maxref` = max outbound-halo sticks/core (remote-temp CB). Cores =
`min(64, output_nhw)`.

| shape | cores | in/core | maxref | ratio% | verdict |
|---|--:|--:|--:|--:|:--|
| resnet 2×2s2 150×150 (C128) | 64 | 352 | 176 | 50 | **SAVE** |
| C=16 25×23 2×2s2 | 64 | 9 | 199 | 2211 | LOSE |
| 3×3s2 pad1 ceil 28×28 (C480) | 64 | 13 | 65 | 500 | LOSE |
| dilation C=7 24×24 | 64 | 9 | 162 | 1800 | LOSE |
| dilation ceil C=1 59×59 | 64 | 55 | 1155 | 2100 | LOSE |
| massive 400×544 3×3s2 (C64) | 64 | 3400 | 1157 | 34 | **SAVE** |
| >800ch 4×4 14×14 | 36 | 6 | 66 | 1100 | LOSE |
| 15×15 kernel 30×30 (C160) | 64 | 15 | 467 | 3113 | LOSE |
| 8×8s6 20×20 (C224) | 12 | 34 | 202 | 594 | LOSE |
| 36×36 kernel 48×48 (C320) | 64 | 36 | 2268 | 6300 | LOSE |
| tiny 6×6 3×3 | 36 | 1 | 14 | 1400 | LOSE |
| ceil edge 13×8 | 12 | 9 | 63 | 700 | LOSE |
| N=8 112×112 3×3s2 (C64) | 64 | 1568 | 432 | 27.6 | **SAVE** |
| N=32 264×40 5×5 (C32) | 64 | 5280 | 200 | 3.8 | **SAVE** |

**Verdict: 5 SAVE / 9 LOSE — the premise is CONFIRMED for this (deliberately
adversarial) shape mix, but the win region is real and important.** SAVE = large feature
maps + small kernels = ResNet/UNet bread-and-butter early layers, saving **28–96 %** of the
input-shard buffer. LOSE = tiny per-core shards (few sticks/core) or large kernel/dilation
(halo depth ≫ shard). The physics: `maxref` ≈ halo depth `(dilated_window_h−1)·padded_W`
is ~independent of core count, while `in/core = input_nhw/num_cores` shrinks with more
cores — so many-core + small-spatial + big-window collapses the saving.

**Path to the good outcome is now concrete:** a cheap, exact host-side gate
`maxref < in_nsticks_per_core` (with a margin for the auxiliary costs below) turns a
blanket loss into a *targeted* win — enable in-place exactly on the large-shard/small-kernel
layers, fall back to normal halo elsewhere. The old feature's mistake was applying it too
broadly.

**Caveats (why this is a proxy, not the final word):** the probe counts only input-shard
(saved) vs remote-temp (added). It ignores (a) config-CB bloat, pad CB, wide-untilize temp
CB; (b) buffer/bank alignment rounding; (c) the over-allocation lifetime effect
(§9c-2). And borderline verdicts depend on the real `num_cores` (I used
`min(64, output_nhw)`); the SAVE cases (big shards) are robust, but small-output LOSE cases
could shift if the op uses fewer cores. Real peak-L1 needs the implementation — but the
SAVE region is large enough here that building it out for height-sharded is clearly warranted.

### 9d. Candidate levers for the good outcome (to try if the table is close)

- **Shrink the temp**: it only needs to hold *outbound* halo, not `shard_width` full
  width if channels are padded; size it from the exact `max_ref_size`, post-untilize dtype.
- **Shrink config CBs**: avoid duplicate-route bloat in the in-place ordering; reuse the
  normal-halo config where the ordering already happens to be safe.
- **Gate precisely** (net-savings check) and, for models, **only enable when the input is
  provably transient** so the over-allocation isn't paid back — turning a blanket
  loss into a targeted win.

---

## 10. Implementation design — LOCKED (per user direction 2026-07-02)

**Silent auto-activation, no user toggle.** There is NO `in_place` flag exposed to the
user (unlike the original, which put `in_place` on Conv2d/MaxPool). The halo op decides
internally, per invocation, using the exact runtime core count. In-place activates iff
**(a) it net-saves L1** (the §9 gate) **AND (b) it is corruption-safe** (a class the
kernel provably handles). Otherwise it silently falls back to normal halo.

**The aliasing mechanism (confirmed from archived `halo_device_operation.cpp:87-107`).**
Sharded L1 buffers allocate top-down, so:
1. Deallocate the input buffer (`input.mesh_buffer()->deallocate()`).
2. Allocate the (larger) output — it reuses the same top region, landing `delta` *below*
   the old input, so the input data survives at the tail `[delta, out_nsticks)` and
   `src_buffer->address() == dst_buffer->address() + delta` holds. Assert this.
Under the current framework this goes in `create_output_tensors`
(`halo_device_operation.cpp:93-97`): if in-place, dealloc input then
`create_device_tensor(output_spec)`; the factory binds `src_cb`→(old input addr) and
`out_cb`→output buffer, which now overlap.

**Caller-lifecycle adaptation (silent).** The pool/conv callers today call
`deallocate_input` and `reallocate_halo_output` (`ttnn::move`) after halo
(`generic_pools.cpp:350-356`). When in-place activates these MUST be skipped —
`deallocate` would double-free the shared buffer and `move` would copy the output to a
fresh buffer (spiking L1 and undoing the saving). Since there is no user flag, the caller
determines this by calling the SAME shared decision function the op uses. So the gate is a
**pure function of (config, input shard, output shard sizes)** — `should_halo_be_in_place(...)`
— computed identically at: (i) the pool caller (to skip dealloc/move), (ii)
`create_output_tensors` (to dealloc+alias), (iii) the factory (to build the in-place
program). No state stored; program-cache-consistent because it's a pure function of the
attributes.

**The gate** = `max_ref_size (outbound-halo sticks) + aux_margin < in_nsticks_per_core`
AND layout ∈ {validated classes} (start: height-sharded). `aux_margin` accounts for the
config-CB / pad-CB / untilize-temp deltas the §9 proxy ignores; start conservative and
tighten once real peak-L1 is measured.

**Build order (vertical slice first):**
1. ✅ DONE (commit fd939515345) `should_halo_be_in_place(...)` + `compute_max_outbound_halo_sticks(...)`
   shared pure fns; validated by gtest (4 SAVE activate / 9 LOSE don't).
2. ✅ DONE (commit 284cd6a7603) `generate_inplace_halo_kernel_config_tensors(...)` in
   `sliding_window.cpp` (local forward-then-reverse ordering, remote transfers kept together
   + `max_ref_size`, pad); max_ref_size cross-checked vs the gate (200k fuzz, 0 mismatch).
3. ✅ DONE (commit aacb4e1cdd8) Aliasing in `create_output_tensors` + overlap `TT_FATAL`.
4. ✅ DONE (aacb4e1cdd8) `halo_gather_in_place.cpp` — faithful port (only include paths changed).
5. ✅ DONE (aacb4e1cdd8) Factory in-place path `build_inplace_halo_program` (ProgramDescriptor).
6. ✅ DONE (aacb4e1cdd8) Caller adaptation in `generic_pools` (skip dealloc/move when gate true).
7. ⏳ IN PROGRESS Rigorous tests (§11). **WH: PASS** (commits 8f0f8effdd1, 5501d8a38c6) —
   **9 SAVE height-sharded RM shapes × {avg,max} × {bf16,bf8_b out} = 36/36 bitwise-exact**
   in-place vs normal (incl. fwd/rev-read edge shapes n8_112, n32_264, and stride-1). Proven
   in-place engaged every case. **Partial-grid + NOOP-core coverage CONFIRMED** via the bf8_b
   (tiled-output) variants (e.g. c256: 49 active / 8×7 rect / 7 noop) — exercises gotcha
   classes 7/8 (partial-grid NOC coords + multicast/noop-release) and passes bitwise; bf16
   variants all used the full 64-core grid. **Remaining:** wide untilize; even more SAVE
   shapes if desired; and **Blackhole** (alignment class 1 / NoC-order class 7 can be WH-clean
   but BH-broken). The factory has a grep-able grid-geometry `log_info` diagnostic (downgrade
   to `log_debug` before PR).

**STATUS — STEP 1 (full pool support on Wormhole) COMPLETE** (branch @ 3f46dbde9d1).
In-place halo silently auto-activates on SAVE shapes (28–96% input-shard L1 saved, zero
output difference), scoped pool-only (conv/others stay off until step 3 via `allow_in_place`).
WH bitwise-exact (`torch.eq`, in-place vs normal, avg+max pool, bf16 + bf8_b) across the FULL
pool matrix:
- HEIGHT-sharded: row-major (36/36) + tiled/untilize incl. wide untilize-temp path.
- WIDTH-sharded: all-local halo (max_ref==0, no remote-temp CB), RM + tiled (6/6).
- BLOCK-sharded: 2D grid incl. partial 8×6 grid + wide untilize (9/9).
Partial-grid + NOOP-core multicast path exercised (via bf8_b tiled output) and passing.
Known untested-but-unreachable-via-pool: column-major block (transpose_mcast) NOC path —
cover in step 3 (conv). The removal premise ("can't net-save L1") is REFUTED for pools.

**Sequence (user-set):** 1. full pool on WH ✅ · 2. full pool on **Blackhole ✅** · 3. extend
to conv / matmul / other ops on both. Plus the deferred DRAM-slicing L1-estimate.

**STEP 2 (full pool on Blackhole p100a) COMPLETE** (branch @ 461c55d3148). Suite on p100a
(11×10 = **110-core** grid): **20 passed / 21 skipped / 0 failed** — bitwise-exact in-place
vs normal across the shapes that activate (height RM+tiled, width all-local, block incl.
tiled buffer-delta path) × bf16 & bf8_b. **No corruption** in the alignment (class 1) or
NoC-ordering (class 7) classes. Findings:
- BH's 110-core grid **narrows the SAVE region** vs WH's 64: mid-size shapes (resnet_150x150,
  etc.) shard across more cores → per-core shard drops below halo depth → flip SAVE→LOSE, and
  the gate **correctly declines** them (the 21 skips). In-place pays off on *larger* feature
  maps on BH. This is the same arch-economics lesson the removal era hit — now handled by the
  auto-gate instead of a blanket flag.
- The small-stick alignment hazard (#27333, C=16→32B sticks on BH's 64B alignment) **cannot
  occur in practice**: such shapes are net-L1 LOSE so in-place never activates — the economics
  gate implicitly steers clear of the alignment danger zone.
- Test made arch-portable: non-activation → `pytest.skip` (not fail), since the gate
  legitimately declines per-arch.

**In-place halo is now verified correct on BOTH Wormhole and Blackhole for the full pool op
family. The removal premise ("can't net-save L1") is refuted for pools on both archs.**

**Earlier status (superseded):** the RM/height-sharded vertical slice was the first milestone.

**Test entry:** `tests/ttnn/unit_tests/operations/pool/test_inplace_halo.py`
(`TT_METAL_DISABLE_INPLACE_HALO=1` forces the normal golden). The economics/gate gtest is
`InPlaceHaloEconomics.MaxPoolHeightSharded` in `test_sliding_window_infra.cpp`.

**Deferred (tracked, non-blocking):** DRAM-slicing L1 estimation. Lowering L1 can make the
estimator *over*-estimate → unnecessary DRAM slicing (already happens today) → no
correctness risk. Only *under*-estimation is dangerous, and in-place never raises L1 when
active. Revisit after the op is solid.

## 11. Verification rigor (per user — this is how the original bugs were caught)

In-place halo "looks" correct extremely easily while harboring races that only corrupt
under specific shapes. Testing discipline:
- **Elementwise `allclose`, NOT just PCC.** Small corruption (a few wrong sticks) can pass
  a PCC threshold; use exact/near-exact elementwise comparison to catch it.
- **Structured inputs** where each stick's value encodes its (core, local index) origin, so
  a wrong value pinpoints the exact sub-stick and the race that produced it.
- **Run in-place vs normal on identical shapes** (the gate can be forced on for tests via a
  test-only hook) and compare outputs elementwise — any divergence is a bug.
- **Cover the edge classes** that historically broke: forward-reads-on-some-cores /
  reverse-on-others (N=8 112×112, N=32 264×40), noop/partial grids, small sticks (C=1,7,16),
  large kernels/dilation, wide untilize, non-tile-multiple NHW, both bf16 and bf8_b.
- **WH is not sufficient** — corruption can be WH-clean, BH-broken (alignment, NoC order).

## 8. Curriculum lessons to capture (running list)

Per the standing request to log novel/unexpected/useful findings via the curriculum
`record_lesson.sh` inbox mechanism (never edit canon directly):

- In-place data movement where output overlaps input needs a **stage-to-temp → global
  barrier → distribute** discipline + **memmove-style per-stick direction** — a reusable
  kernel pattern.
- Cross-core release via **increment-to-cast-core → multicast** is how halo avoids a
  per-core runtime-arg core list (BH arg-count ceiling).
- The recurring hang root cause is **NOC-coordinate / core-grid-rectangle** computation,
  not the copy logic — worth its own note.
- (more as discovered)
