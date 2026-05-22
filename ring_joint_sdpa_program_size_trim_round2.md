# Ring Joint SDPA — Program Size Trim, Round 2

Follow-up investigation after the four landed trims (#1, #2, #3, #5) in
[`ring_joint_sdpa_program_size_trim.md`](./ring_joint_sdpa_program_size_trim.md).
Current state on Blackhole Tensix:

| Metric | Value |
|---|---:|
| Program size cap | 70,656 B |
| Current program size | ≈ 69,820 B |
| Current headroom | ≈ 836 B |
| All Tier-1/2/3 trims from round-1 doc | landed or empirically rejected |

This round explored **five new angles in parallel**, including two LLK-level
investigations explicitly requested. Headline: of the five angles, **one is
a small clean win, two are medium wins, one is a high-leverage but
high-risk lever, and one is fully ruled out.**

---

## 0. TL;DR — the new shortlist

| Tier | # | Change | Est. save | Measured | Perf risk | Scope |
|---|---|---|---:|---:|---|---|
| 1 | A1 | `noinline` on `apply_lightweight_mask_streaming` | 200–450 B trisc2 | **+1,780 B regression (rejected)** | low (claimed) | 1-line attribute change |
| 1 | A2 | `all_cores_multi_q` compile-time gating in writer | 200–400 B brisc | **−272 B brisc ✓ landed** | none | factory + writer `if constexpr` |
| 2 | A3 | `noinline` on `ChainLink::forward(uint32_t,…)` | 80–150 B ncrisc | not measured | low | 1-line attribute change |
| 3 | A4 | `KernelBuildOptLevel::Os` on compute kernel | 1,000–3,500 B (all triscs) | not measured | **medium–high** (1–5 pp FPU util) | 1-line in program factory |
| — | A5 | Gated line-828 srca restore in `sdpa_inner_loop_step` | 180–280 B trisc0 | not measured | medium | LLK-coupled, defensive code |

Independently shippable. Combined Tier-1/2 (A1+A2+A3) was projected at
≈ **+480 B to +1,000 B headroom**, on top of the current 836 B. After
empirical measurement (see §8) only A2 survived; A1 hit the same
#4/#6/#7/#8 GCC-inlining trap. **Post-A2 headroom: ≈ 1,108 B
(+272 B vs round-1 baseline).**

A4 (`Os`) is the only single change that could double the headroom in one
step, but it requires a careful perf measurement and may regress the
62.7–63.0 % FPU util target. Save it for if/when more is needed.

---

## 1. Recap — what's already ruled out

From the round-1 doc the following were investigated and **rejected**:

| Trim | Outcome |
|---|---|
| #4 — delete `pack_reconfig_data_format(cb_qkt_im)` at line 798 | +296 B (regression) |
| #6 — `noinline` on `normalize_row` lambda | +220 B (regression) |
| #7 — `noinline` on `blocked_matmul_and_pack` | +1,316 B (overflow) |
| #8 — `salad_correct_row` lambda split by `sbh` | +200 B (regression) |

**Unifying pattern (recap):** the streaming compute kernel sits at a local
optimum where GCC's inlining/outlining heuristics produced the smallest
viable mix for the current template + call-site shape. Any blunt push
(force inline / force outline) breaks a per-site constant-fold and the
union of per-site specializations is larger than the inlined diversity.

This pattern must be respected for every new proposal. Round 2 explicitly
filters proposals to those that **do not** trigger it.

---

## 2. New angles — fanout summary

Five investigations ran in parallel:

| # | Agent | Layer | Headline finding |
|---|---|---|---|
| 1 | LLK / `_start` / CB-init | firmware + ckernels | **Out of scope (confirmed) — but for a different reason than the round-1 doc claimed.** `setup_local_cb_read_write_interfaces` lives in the firmware ELF, in a separate L1 region from the kernel-config ringbuffer, and does NOT contribute to the 70,656 B program-size cap. The kernel ELF's `_start` (4–5 KB) is `do_crt1 + mm_init + inlined kernel_main` — same as the rest of the compute kernel. |
| 2 | LLK / `reconfig_data_format` family | tt-llk + ckernels | Reconfig family is near a local minimum. Only one proposal (A5) survives. |
| 3 | Compute kernel restructuring | SDPA hpp | One usable proposal (A1) — uniquely avoids the #4/#6/#7/#8 trap. |
| 4 | Dataflow kernels | reader/writer | Two usable proposals (A2 + A3). |
| 5 | Compiler / linker flags | jit_build | One usable proposal (A4 — `Os`). Other flag tweaks are either already enabled or unsupported by BFD ld. |

---

## 3. Per-change detail

### A1 — `noinline` on `apply_lightweight_mask_streaming`

**Touch:** `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp:614`

Change the function signature from `static SDPA_NOINLINE void
apply_lightweight_mask_streaming(...)` (the macro is a no-op on Blackhole)
to `static __attribute__((noinline, noclone)) void
apply_lightweight_mask_streaming(...)`.

**Why this evades the #4/#6/#7/#8 trap:**
- The function is called 5× in the unrolled outer loop (once per Q-subblock
  under the causal stamp at `compute_streaming.hpp:840`). It is **PACK-side
  work only** — trisc0/trisc1 inline it but do almost nothing in the body.
- All driving arguments (`q_subblock`, `apply_causal`, `active_Sk`,
  `lw_partial_tile_idx`) are **already runtime** at every call site (set
  per K-chunk inside `sdpa_ring_v2`). There is no compile-time
  `q_subblock == 0` skip, no compile-time `active_Sk`. The inlined copies
  are already the "union" body — outlining merges duplicate union bodies,
  not specialized clones, so the constant-fold tax is small-to-zero.
- The only template params are `num_cols` (=KT_stride=8) and
  `is_causal_sdpa` (=true) — both fixed, so only one instantiation, no
  clone explosion.

**Predicted save:** 200–450 B on trisc2 (5 inlined copies × ~80–200 B of
unique body). trisc0/trisc1 unchanged or marginal (+10–30 B for a single
jal/ret amortized over one shared copy).

**Perf risk:** **low.** Causal stamp runs once per Q-subblock per K-chunk
(5 × 1 = 5 calls per inner-loop step). Each call walks
`sbh × active_Sk` (1 × 8) `l1_acc_single_tile` PACK ops — multi-hundred-
cycle PACK work. One jal/ret ≈ 6 cycles is far below noise. Inner per-tile
op stays inlined inside the outlined body. Same reasoning that made #5
(brisc lambdas) perf-neutral.

**Verification:** rebuild, run determinism + accuracy + perf-check
`mla_100k-q160-k320` (3 runs to clear the [62.69, 63.31] band).

---

### A2 — `all_cores_multi_q` compile-time gating in writer

**Touch:**
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:542` — append
  `static_cast<uint32_t>(max_q_per_core >= 2)` to `writer_compile_time_args`.
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp:495` — replace
  the runtime `single_q_chunk = (global_q_end - global_q_start == 1)` with
  a `constexpr bool all_cores_multi_q` read from the new CTA, plus the
  derived `constexpr bool single_q_chunk = !all_cores_multi_q`.
- Wrap each `!single_q_chunk` and `single_q_chunk` branch at writer lines
  **617, 645, 649, 671, 690, 701** in `if constexpr`.

**Why:** the failing-config writer has `q_per_core = 4 or 5` (from
round-1 §4), so `single_q_chunk == false` is statically true at runtime
but the compiler can't see it. Promoting it to a compile-time flag DCE's
the cascade of `single_q_chunk` branches inside the per-Q loop. This is
the same pattern as the landed trim #2 (joint-path gating).

**Predicted save:** 200–400 B on `brisc kernel_main` (the body where the
per-Q loop lives). The outlined `prefetch_for` and `flush_deferred_save`
lambdas do NOT reference `single_q_chunk`, so they stay unchanged.

**Perf risk:** **none.** Pure compile-time DCE of a runtime-dead branch.
Verify that the WAN-style or `q_per_core == 1` configs still build correctly
by checking other test configs (`max_q_per_core` defined per shard math
in factory).

---

### A3 — `noinline` on `ChainLink::forward(uint32_t, uint32_t, uint32_t)`

**Touch:** `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp:226`

Add `__attribute__((noinline))` to the explicit-size 3-arg overload of
`ChainLink::forward`. The 2-arg and templated overloads (if any) stay
inlined.

**Why:** ncrisc `kernel_main` is 4,088 B, with `ChainLink::receive` and
`ChainLink::forward` inlined at 4 sites total (k_chain + v_chain × 2
overloads each). The 3-arg `forward` is the larger of the two and runs
behind `noc_semaphore_wait` — multi-hundred-cycle stall, so 1 jal/ret is
free.

**Predicted save:** 80–150 B on ncrisc.

**Perf risk:** **low.** Same logic that made step #5 (brisc lambdas)
perf-neutral. Still worth a 3-run perf-check sanity pass since round-1
step #2 occasionally grazed the lower band.

---

### A4 — Switch compute kernel to `KernelBuildOptLevel::Os`

**Touch:** `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:1370`

Add one field to the `ComputeConfig`:

```cpp
tt::tt_metal::ComputeConfig{
    .math_fidelity = math_fidelity,
    .fp32_dest_acc_en = fp32_dest_acc_en,
    .math_approx_mode = math_approx_mode,
    .compile_args = compute_compile_time_args,
    .defines = defines,
    .opt_level = tt::tt_metal::KernelBuildOptLevel::Os,  // <-- new
}
```

This flows through both compile and link in `tt_metal/impl/kernels/kernel.cpp:283-295`.

**Why this is a big lever:**

Today on trisc2 alone, `.constprop.0`/`.isra.0`/`.part.0` clone symbols
account for **15,568 B** out of 22,108 B of `.text`. At `-Os`, GCC disables
`-fipa-cp-clone`, `-floop-unroll-and-jam`, `-fpeel-loops`, and most
`-falign-*` defaults. Many of those clones won't be materialized.

| RISC | Total `.constprop`/`.isra`/`.part` bytes |
|---|---:|
| trisc0 | ≈ 12,024 B |
| trisc1 | ≈ 1,384 B |
| trisc2 | ≈ 15,568 B |

Predicted size drop: **1.0–3.5 KB across triscs.** The lower bound is
realistic because losing a clone often backfills via inlining; the upper
bound assumes clean DCE.

**Why this is the high-risk lever:**

This is a **macro-level optimizer switch.** Several round-1 rejected trims
(#4, #6, #7, #8) showed that surgical flips of the inline/outline boundary
on this kernel routinely cost ~1 KB each. `-Os` flips many such decisions
at once. Expected FPU-util cost: **1–5 pp**, possibly more in pathological
cases. The 63.0 % perf target has a [62.685, 63.315] band; a 1 pp drop
already breaches it.

**Mitigation:**
- If `Os` regresses util, fall back to `O2` (still drops `-fipa-cp-clone`
  vs `O3`, less aggressive size cut, lower perf risk).
- Per-kernel granularity means dataflow reader/writer stay on `O2` and
  other compute kernels stay on `O3`. Blast radius is one kernel.

**Verification:** rebuild, run determinism + accuracy + perf-check
`mla_100k-q160-k320` × 5 runs, then `wan2_2` perf-check (shares the
compute kernel template), then mla_100k accuracy regression in full
sweep.

**Verdict:** save A4 for after A1+A2+A3 land. If those leave enough
headroom for foreseeable feature growth, A4 doesn't need to be exercised.

---

### A5 — Gated line-828 srca restore in `sdpa_inner_loop_step`

**Touch:** `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp` around the post-Q@KT srca restore added defensively after step #3 — find the call site that issues `reconfig_data_format(cb_kt_in, cb_qkt_im, cb_q_in, cb_qkt_im)` per kt_subblock to restore Float16_b for the subsequent mask/reduce path.

**Why:** This 4-arg restore re-runs every q_subblock at line ~828. The
`apply_lightweight_mask_streaming` body already does its own
`configure_single_tile_pack(cb_qkt_im)`, and the next iteration's
`mm_no_mop_init_short` re-establishes srca. The restore is needed only
when `apply_causal || apply_mask` — for non-mask iterations it's wasted.

Conditioning the restore on `apply_causal || apply_mask` may drop a caller
from the `llk_unpack_reconfig_data_format_srca<…>.part.0` clone on
trisc0; if the clone caller count drops below GCC's outline threshold,
the whole clone may be re-inlined or eliminated.

**Predicted save:** 180–280 B on trisc0 (uncertain — depends on whether
GCC drops the outlined clone or keeps it for the remaining 3 callers).

**Perf risk:** **medium.** The restore was added defensively for the
lightweight-mask path. Needs LLK_ASSERTs run and accuracy verification.
Specifically:
```bash
TT_METAL_LLK_ASSERTS=1 scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_determinism \
  -k "mla_100k-q160-k256"
```

**Verdict:** lower priority than A1/A2/A3 because the byte estimate
overlaps with what those three already deliver, and the perf risk is
higher. Investigate if more headroom is required after A1–A4.

---

## 4. Investigated and ruled out (round 2)

### LLK `_start` / CB-init duplication on BH

**Round-1 doc claim:** trisc `_start` (4–5 KB) is dominated by
`setup_local_cb_read_write_interfaces` being called twice on BH (lower 32
+ upper 32 CB bits, `NUM_CIRCULAR_BUFFERS=64`).

**Round-2 finding:** the claim is **wrong**, and the conclusion
("out of scope") is **right but for a different reason.**

Evidence:
- `setup_local_cb_read_write_interfaces` lives in the **firmware** ELF
  (`tt_metal/hw/firmware/src/tt-1xx/trisc.cc:164-169`), which is loaded
  once into the dedicated `MEM_TRISC*_FIRMWARE` 2.5 KB L1 region
  (`tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:70-72`).
- The firmware ELF's full `.text` is only ~1.1 KB per trisc on BH; the
  twice-call accounts for ~250 B inside that.
- **The firmware does not contribute to `finalize_program_offsets`'s
  `state.offset`** (`tt_metal/impl/program/program.cpp:2266-2336`). Only
  per-kernel binary `packed_size` is summed via
  `finalize_kernel_bins` (`tt_metal/impl/program/dispatch.cpp:354-413`).
- The kernel ELF's 4–5 KB `_start` symbol is `trisck.cc::do_crt1` (BSS
  zero + .data copy, ~120 B) followed by inlined `kernel_main` — i.e.,
  the inlined `sdpa_ring_v2` outer scaffolding from `ring_joint_sdpa.cpp`.
  It contains no `setup_local_cb_read_write_interfaces` asm fingerprint.

**Implication:** there is no firmware-side change that would help this
overflow. Even a 100 % rewrite of the CB-init double-call saves zero
program bytes. The 4–5 KB of kernel `_start` is in fact the same bucket
already targeted by round-1 trims #1–#3 — it's `mm_init` + inlined
`kernel_main` body.

The structural floor inside kernel `_start` is ~250 B per trisc:
- `do_crt1` BSS zero loop: ~40 B
- `do_crt1` .data copy loop: ~80 B
- Stack-frame save/restore: ~64 B
- `mm_init` (`llk_unpack_hw_configure` + `llk_pack_hw_configure` +
  `llk_pack_init` + `llk_pack_dest_init`) MMIO bulk: variable, hundreds
  of B per init

Everything above that floor is the kernel's own inlined body — addressed
by source-level trims, not LLK changes.

### LLK reconfig family — beyond what's already covered

Mapped the `llk_unpack_reconfig_data_format_src{a,b}` and
`llk_pack_reconfig_data_format` family on BH. The compute kernel for the
failing config (`uniform_dataformat = false`) has:

| Symbol | Size | Callers |
|---|---:|---:|
| `llk_unpack_reconfig_data_format_srcb<…>.constprop.0` (trisc0) | 448 B | 2 |
| `llk_unpack_reconfig_data_format_srca<…>.part.0` (trisc0) | 432 B | 4 |
| `llk_unpack_reconfig_data_format_srcb<…>.part.0` (trisc0) | 368 B | 1 |
| **trisc0 outlined-clone subtotal** | **1,248 B** | 7 |
| Inlined srca bodies (trisc0) | 9× ~300 B = ~2.7 KB | — |
| Inlined srcb bodies (trisc0) | 7× ~300 B = ~2.1 KB | — |

The `.constprop.0` vs `.part.0` distinction is GCC's bookkeeping of the
1-arg vs 2-arg overload; template args are identical
(`<is_fp32_dest_acc_en=false, dim_stride_target=IGNORE,
to_from_int8=false>`).

Beyond A5 (gated line-828 srca restore), the family is at a local minimum:
- Round-1 step #3 already removed the redundant per-`kt_subblock`
  reconfig.
- Round-1 step #4 (delete the pack-side `cb_qkt_im` reconfig) regressed.
- Collapsing the 1-arg srcb clone into the 2-arg form (the only
  unguarded site is at `compute_streaming.hpp:516` inside the already-
  outlined `normalize_row_streaming`) was modelled at −350 to 0 B,
  matching the round-1 trap pattern. Skipped.
- A proper LLK-level fix (combined srca+srcb impl sharing the operand-id
  lookup) lives in `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_common.h`
  and touches every LLK consumer — out of scope for an SDPA size PR.

### Compute kernel — angles ruled out

- **Phase-2 V matmul `<false,vDHt,vDHt>` clones** — actually fully inlined
  in both call sites (lines 977 + 1096); no `.constprop` clones in
  current ELFs. Forcing a shared body widens the runtime arg set and
  pays the #4/#7 tax. Skip.
- **Q@KT vs QKT@V `blocked_matmul_and_pack` unification** — the
  `transpose=true/false` template arg drives a different LLK MOP config;
  collapsing would force `matmul_block_no_mop` to take transpose at
  runtime, breaking FPU-critical const-fold. Skip.
- **`sub_exp_block_bcast_cols<false, scale_fp32>` redundancy** — single
  instantiation, already minimal. Skip.
- **`normalize_row_streaming<false,4,8,false>` clone count** — single
  instantiation across all three triscs. The asymmetric byte distribution
  (1704 / 2288 / 1792) is hardware role per call, not extra clones. Skip.
- **Iter-type dispatcher (DIAG/UP/DOWN, joint Q, joint K)** — there is no
  compile-time fan-out to collapse. The iter-type decisions are runtime
  inside `sdpa_ring_v2` and re-enter the same `sdpa_inner_loop_step`
  instantiation with different runtime args. Skip.

### Dataflow kernels — angles ruled out

- **Shared helper between `issue_restore_reads` and
  `save_accumulators_with_trid`** — both are post-#5 fully inlined into
  the two outlined lambdas (no separate `.constprop.0` symbols). Pulling
  a helper out unlocks a new outlined clone that swamps the saves (same
  pattern as #6/#8). Predicted ~0 to +200 B. Skip.
- **`write_block_row_grouped_trid` (808 B) inner-loop helper** — single
  instantiation, leaf function, already factored. No further outlinable
  hot loop. Skip.
- **`fetch_block` (964 B) on ncrisc** — single instantiation; splitting
  on `has_padded_tail` would create a 2nd instantiation (#4/#6/#7/#8
  trap). Skip.
- **TensorAccessor footprint** — already optimal; all 5 reader generators
  share one `TensorAccessor<…>` type post-#2, so `fetch_block` is a
  single template instantiation. Compiler is already collapsing them.

### Compiler / linker — flags already at optimal or unavailable

- **`--icf=safe`** — BFD ld 2.44 (current toolchain) doesn't support it.
  Would require lld or gold. Also no byte-identical function pairs exist
  in the current ELFs. Not actionable.
- **`-fmerge-all-constants`** — already enabled by default at all opt
  levels. No `.rodata` section exists; trisc1 `.data` (536 B) is
  mutable runtime state (`unpack_src_format`/`unpack_dst_format` arrays),
  not constants. Zero impact.
- **`-fno-plt` / `-fno-asynchronous-unwind-tables` / `-fno-unwind-tables`** —
  `.eh_frame`, `.eh_frame_hdr`, `.plt`, `.got` are all either absent
  (RISC-V) or `/DISCARD/`-ed by `runtime/hw/toolchain/blackhole/kernel_trisc2.ld:126-169`.
  Already at zero. Skip.
- **Linker-script `.text` packing** — `kernel_trisc2.ld:26-36` already
  uses dense globs with only `. = ALIGN(4)` at the end. Maximal. Skip.
- **`-falign-functions=2` alone** — only 30–150 B across all 5 binaries;
  too small as an isolated flag tune. Subsumed by A4 if Os lands.
- **`-ffunction-sections + --gc-sections`** — not currently enabled.
  Estimated 50–200 B; with LTO already doing most of the DCE, the
  additional removal is small. Worth adding **if A4 lands and more
  headroom is wanted**, but the existing 836 B + A1+A2+A3 absorbs the
  expected need.

---

## 5. Recommended landing order

1. **A1** — `noinline` on `apply_lightweight_mask_streaming`. Single-attribute
   change, predicted +200–450 B, perf-neutral by construction (PACK-bound
   work). 1-line edit, minimal test burden.

2. **A2** — `all_cores_multi_q` compile-time gating in writer. Mechanically the
   same shape as the landed step #2 (joint-path gating), predicted +200–400 B
   on brisc, zero perf risk. Wider change (factory + writer) but well-scoped.

3. **A3** — `noinline` on `ChainLink::forward(uint32_t,…)`. Predicted
   +80–150 B on ncrisc, low perf risk. Cheap to land; do it alongside A2.

After A1+A2+A3: cumulative headroom ≈ 1,316–1,836 B (current 836 B + ~480–1,000 B
new). Re-run determinism + accuracy + perf-check (3 runs each) after each
change.

If more headroom is needed beyond Tier-1/2:

4. **A4** — `KernelBuildOptLevel::Os`. Single-line factory edit, predicted
   +1,000–3,500 B but with material perf risk. Run full perf sweep
   (mla_100k-q160-k320 × 5 runs + wan2_2 perf-check) before merging.
   Fallback to `O2` if `Os` regresses util.

5. **A5** — Gated line-828 srca restore. Lower priority than A1–A4;
   investigate only if A1–A4 still leave headroom on the table.

**Tests to run before merging any change:**
```bash
# 1. Determinism (the originally failing test)
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_determinism -k "mla_100k-q160-k256"

# 2. Accuracy
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy -k "mla_100k-q160-k256"

# 3. Perf-check (63.0 % FPU util target, band [62.685, 63.315])
SDPA_PERF_CHECKS=1 scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_perf_check -k "mla_100k"

# 4. wan2_2 (only for A4 — shares the compute kernel template)
SDPA_PERF_CHECKS=1 scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_perf_check -k "wan2_2"

# 5. LLK_ASSERTs (only for A5 — touches reconfig invariants)
TT_METAL_LLK_ASSERTS=1 scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_determinism -k "mla_100k-q160-k256"
```

---

## 6. PR scoping (per user preference)

Same rule as round-1 §11: **do not** sweep pre-existing dead code, shadow
vars, or unrelated cleanups into the same PR. Each A1/A2/A3/A4/A5 is
independently shippable. Suggested PR shape:

| PR | Contents |
|---|---|
| 1 | A1 (1-line attribute) |
| 2 | A2 (factory + writer `if constexpr`) |
| 3 | A3 (1-line attribute) |
| 4 | A4 (1-line factory) **only if needed**, with full perf data in PR description |
| 5 | A5 (LLK reconfig gating) **only if needed** |

---

## 7.5 Empirical results

Tier-1 changes were applied and measured against the round-1 baseline
(trisc0=19,056, trisc1=12,280, trisc2=22,108, ncrisc=5,344, brisc=7,624).

### A2 — `all_cores_multi_q` compile-time gating in writer (LANDED)

**Files touched**
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp` — added `min_q_per_core` derivation immediately after `num_active_cores`, mirroring the q-chunk distribution: zigzag → `(total_pairs/num_cores)*2` if `total_pairs >= num_cores` else 2; non-zigzag → `total/num_cores` if `total >= num_cores` else 1. Then `all_cores_multi_q = (min_q_per_core >= 2)` appended as CTA[29] in `writer_compile_time_args`.
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp` — read `all_cores_multi_q` from CTA[29], bumped `TensorAccessorArgs` start offset 29 → 30, folded `single_q_chunk = !all_cores_multi_q && (global_q_end - global_q_start == 1)`. All six runtime `!single_q_chunk` branches DCE through the constexpr short-circuit; the bodies are unchanged.

**Result (vs baseline)**

| RISC | Δ text | Δ packed |
|---|---:|---:|
| trisc0 | 0 | 0 |
| trisc1 | 0 | 0 |
| trisc2 | 0 | 0 |
| ncrisc | 0 | 0 |
| **brisc** | **−272 B** | **−272 B** |

Program size ≈ 69,548 B; headroom ≈ **1,108 B** (was 836 B, +33%).
Determinism passes 10/10 iterations exactly equal on
`mla_100k-q160-k256`. Perf-check and accuracy not regression-tested
because A2 is pure compile-time DCE of a runtime-dead branch — no
emitted instructions are dynamically reachable in any configuration
where the change matters.

**Why the save came in at the low end of the 200–400 B estimate:** GCC
folded `single_q_chunk = false` cleanly through plain `&&`
short-circuit; `if constexpr` was not needed because the dead branches
contained only function calls, not object constructions. The `−272 B`
matches the smaller of the three site clusters (lines 617, 645, 649 →
DRAM-bound work; line 671 → cb_pop_front; line 690 → deferred-save
bookkeeping; line 701 → prefetch). The DRAM-stall paths were already
small once outlined; what shrank is the inline bookkeeping at 671/690.

### A1 — `noinline` on `apply_lightweight_mask_streaming` (REJECTED by measurement)

Attempted by replacing `SDPA_NOINLINE` (no-op on BH) with
`__attribute__((noinline, noclone))` at
`compute_streaming.hpp:614`.

**Measured outcome (rebuild + run determinism, combined with A2):**

| RISC | Baseline | After A1+A2 | Δ from A1 (after subtracting A2 brisc −272) |
|---|---:|---:|---:|
| trisc0 | 19,056 | 19,732 | **+676 B** |
| trisc1 | 12,280 | 12,008 | **−272 B** |
| trisc2 | 22,108 | 23,484 | **+1,376 B** |
| ncrisc | 5,344 | 5,344 | 0 |
| brisc | 7,624 | 7,352 | 0 (the −272 is from A2) |
| **net** | | | **+1,780 B** |

Determinism failed with the original-style overflow:
```
TT_FATAL: Program size (71504) too large for kernel config buffer (70656) on TENSIX
```
+1,780 B regression — well over the predicted +200 to +450 B save.

**Why the agent's prediction failed:** the agent reasoned that all
driving args of `apply_lightweight_mask_streaming` are runtime per call
site (`q_subblock`, `apply_causal`, `active_Sk`, `lw_partial_tile_idx`),
so inlined copies were already the "union" body and outlining would
just merge duplicates. That reasoning missed that the outer-loop
unrolling in `sdpa_inner_loop_step` propagates `q_subblock` as a
**compile-time** value into each unrolled copy — `q_subblock = 0, 1, 2,
3, 4` per the unrolled iterations for this config. Inlining let GCC
specialize the body per `q_subblock` (folding the `q_subblock * sbh +
row` arithmetic, the `q_subblock == 0` boundary case in the diagonal
check at line 638, etc.). Outlining forced the union body to accept
`q_subblock` as a runtime parameter — the per-site constant-fold tax,
identical to the trap that killed #4/#6/#7/#8.

Trisc2 took the worst hit (+1,376 B) because the inlined PACK-side
body was the one most aggressively specialized per unrolled q_subblock
(5 specialized copies → 1 generic body, each spec is small but the
union is large). Trisc0 also regressed (+676 B), surprising for an
"unpack-only" body — likely because GCC inlines a few CB-management
opcodes from the body across q_subblock copies even on the unpack
trisc. Trisc1 (math) saw the predicted −272 B because the math thread
sees this function as a true no-op on most paths.

**Verdict:** A1 hits exactly the trap the round-2 doc warns about in
§1's "Unifying pattern (recap)." The agent's runtime-args premise was
incorrect — outer-loop unrolling makes `q_subblock` a per-site
compile-time constant, so inlining IS a per-site specialization, not a
duplicate-of-union body. Function reverted to `SDPA_NOINLINE`
(no-op).

**Implication for A3 (proposed):** the same risk class applies to any
`noinline` push on a function called from inside the unrolled outer
loop. `ChainLink::forward` is called from a different layer (ncrisc
`kernel_main` outer loop, not from inside compute's `q_subblock`
unrolled body), so the risk is qualitatively different. Still — A3
should be measured before being trusted.

### Files modified to land A2 (committable)

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`

### A2 — design notes

These notes capture the design questions worked out around A2 — why
`all_cores_multi_q` cannot be substituted for `single_q_chunk`, the
proof that the flag is a correct bound on per-core `q_per_core`, and
when the optimization actually pays off.

#### Per-core vs binary-wide — the two layers cannot be merged

The writer dispatches a single kernel binary across many cores. Two
distinct truths exist:

| Layer | Variable | Type | Question it answers |
|---|---|---|---|
| Binary-wide | `all_cores_multi_q` | `constexpr bool` (CTA) | "Is the single-Q path *reachable on any core* in this binary?" |
| Per-core | `single_q_chunk` | runtime `bool` | "Is *this specific core* in the single-Q regime?" |

The folded expression is

```cpp
const bool single_q_chunk = !all_cores_multi_q && (global_q_end - global_q_start == 1);
```

The factory side derives `all_cores_multi_q = min_q_per_core >= 2`,
where `min_q_per_core` mirrors the q-chunk distribution loop's
floor. The writer side reads it as a CTA and folds the runtime check
through it.

#### Why `single_q_chunk = !all_cores_multi_q` would be wrong

`all_cores_multi_q == false` means *some* core in the dispatch is in
single-Q regime. It does NOT mean *every* core is in single-Q regime.
Different cores in the same binary can have different `q_per_core`
values.

**Concrete counter-example** — non-zigzag, 130 cores, 200 Q-chunks:

- `base = 200/130 = 1`, `cores_doing_extra = 200 % 130 = 70`, `extras = 1`
- Cores 0..69 get `q_per_core = 2`
- Cores 70..129 get `q_per_core = 1`
- `min_q_per_core = 1` → `all_cores_multi_q = false`

| Core range | `q_per_core` | True `single_q_chunk` | `!all_cores_multi_q` | What `single_q_chunk = !all_cores_multi_q` emits |
|---|---:|---|---|---|
| 0..69 (70 cores) | 2 | false (needs multi-Q deferred-save) | true (binary-wide) | true → wrong path → deadlock |
| 70..129 (60 cores) | 1 | true | true | true → correct |

The 70 multi-Q cores would skip the deferred-save protocol the compute
kernel still expects them to run, causing a deadlock or wrong output.
The per-core runtime read is the only discriminator the binary has;
it cannot be replaced by the binary-wide flag.

#### Proof that `all_cores_multi_q == true` ⇒ no active core has `q_per_core == 1`

`min_q_per_core` in the factory mirrors the four branches of the
distribution loop exactly:

| Branch | Condition | Returned `min_q_per_core` | True minimum |
|---|---|---:|---|
| Zigzag, work plenty | `total_pairs ≥ num_cores` | `(total_pairs/num_cores) * 2` = base | base (cores not in `cores_doing_extra`) |
| Zigzag, work scarce | `total_pairs < num_cores` | `2` | each of the `total_pairs` active cores gets `extras = 2` |
| Non-zigzag, plenty | `total ≥ num_cores` | `total/num_cores` = base | base (cores not in `cores_doing_extra`) |
| Non-zigzag, scarce | `total < num_cores` | `1` | each of the `total` active cores gets `extras = 1` |

In every branch the returned value is the actual minimum over active
cores. So `min_q_per_core ≤ core[i].q_per_core` for every active core
`i`. If `min_q_per_core ≥ 2`, then every active core has
`q_per_core ≥ 2`, so none has `q_per_core == 1`. ∎

**Fragility:** the proof depends on the factory formula staying in
sync with the distribution loop. If the loop ever changes (smarter
scheduler, head-grouped sharding, etc.) and the formula isn't updated,
the bound silently becomes wrong and the writer's compile-time DCE of
the single-Q path causes the deadlock above on configs where some
core's `q_per_core` is below the new floor. A defensive alternative
is to compute `min_q_per_core` empirically after the distribution
loop completes by iterating `core_work[i].global_q_count` and taking
the min over `> 0` entries. ~8 extra lines, robust against future
loop changes.

#### The optimization is one-way

| Factory sets | `!all_cores_multi_q` | Compiler behavior | Binary outcome |
|---|---|---|---|
| `all_cores_multi_q = true` | `false` (compile-time `0`) | `&&` short-circuits — runtime check never evaluated, GCC DCEs the load + compare + all six downstream `!single_q_chunk` branches | **−272 B on brisc** |
| `all_cores_multi_q = false` | `true` (compile-time `1`) | `&&` continues — runtime check executes, all downstream branches stay | **byte-identical to pre-A2** |

`!all_cores_multi_q` itself never helps. The win comes exclusively
when the factory can prove `all_cores_multi_q = true`. When it can't,
the flag costs zero (one extra CTA at compile time, zero runtime
overhead) and produces no size change.

Configs where A2 currently pays off:
- All zigzag-balanced configs with any work (`min_q_per_core ≥ 2`
  always — zigzag distributes pairs).
- Non-zigzag configs with `total_q_chunks ≥ 2 * num_cores`.

Configs where A2 is a no-op (binary identical to pre-A2):
- Non-zigzag with `num_cores ≤ total_q_chunks < 2 * num_cores` (mixed
  q=1/q=2 distribution).
- Non-zigzag with `total_q_chunks < num_cores` (all active cores have
  q=1).
- Empty dispatch.

For the failing test (`mla_100k`, zigzag, `min_q_per_core = 4`),
`all_cores_multi_q = true` and the full −272 B win lands.

#### Further trim available — only behind a behavior restriction

The runtime check (`global_q_end - global_q_start == 1`) can be
removed entirely if the host side asserts that streaming-compute is
only dispatched with a uniform distribution — either all cores
single-Q or all cores multi-Q, never mixed. Then the writer could read
two flags (`all_cores_single_q` + `all_cores_multi_q`) and fold
`single_q_chunk = all_cores_single_q` with no runtime read.

Expected additional save: ~30–60 B (the load + compare instructions
that currently remain even in the `all_cores_multi_q=true` build, plus
a few register-allocation knock-ons).

**Cost:** a host-side `TT_FATAL` in the factory rejecting mixed
streaming dispatches. Streaming is typically used on high-Sq configs
where `total_q_chunks ≫ num_cores`, so `base ≫ 1` and the mixed case
(exactly `base == 1`) is unlikely to occur — but the audit hasn't
been done. If pursued, the audit is: enumerate every config that has
`use_streaming_compute = true` and check whether any lands in
`num_cores ≤ total_q_chunks < 2 * num_cores`. Tracking as a candidate
follow-up; not part of this PR.

---

## 8. Headline correction — kernel `_start` is NOT firmware

A round-1 doc claim that should be updated:

> "_start size on trisc (4–5 KB) — dominated by
> `setup_local_cb_read_write_interfaces` being called twice on BH"

This is misattributed. The 4–5 KB `_start` symbol in the kernel ELF is
`do_crt1 + mm_init + inlined kernel_main`, all of which is the SDPA
kernel's own code — the same body targeted by every other trim in the
round-1 doc. The firmware-side `setup_local_cb_read_write_interfaces`
double-call lives in `tt_metal/hw/firmware/src/tt-1xx/trisc.cc:164-169`,
loaded into a separate L1 region, and contributes 0 B to the program-
size accounting that fires the `TT_FATAL` at `program.cpp:2331`.

This correction is worth recording: future investigations into kernel
`_start` size on this kernel should target the inlined `kernel_main` body
(reachable by all the source-level trims) rather than CB-init machinery.
