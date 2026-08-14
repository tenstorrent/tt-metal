# tt-llk test strategy for the blaze→tt-metal `experimental/` promotions

> **Status (2026-08-14, latest).** Work is on branch
> `ldjurovic/llk-tests-blaze-promotions` (tt-metal), which merges #52709 + #52713 + #52727
> onto main so the promoted headers exist to compile against. Everything below was run on
> Blackhole p100a.
>
> | | |
> |---|---|
> | Verification tier (V1–V4) | **4 of 4 done, all green** |
> | New test items | **3 of 8 landed**, 1 attempted-and-reverted, 4 not started |
> | Tests added | 3 drivers + 3 python suites, **75 passing / 2 xfailed** |
> | Product findings | **1 defect** (§9 Correction 2, needs an owner decision) + 3 behavioural constraints |
>
> Landed: 🟩 `add_rsqrt` (§3.2), 🟩 `custom_mm`/`compressed_custom_mm` `block_uninit`
> (§5.2), 🟩 sort-header coexistence (§4.3).
> Reverted: 🟥 the `eltwise_mul_scalar` HiFi reproducer (§3.3) — it hangs the device and
> the theory behind it is disproved; **read §10 before re-attempting it**.
> Not started: `mul_reduce_scalar` chunked (§3.5), rmsnorm dest-reuse (§3.4),
> plain `custom_mm` matmul (§5.3), `top32_rm` (§4.2), sampling polluter (§2).
>
> **Read §9 and §10 before acting on §3–§5.** Four of their predictions were wrong, and
> the corrections are recorded there rather than edited into the original prose, so the
> reasoning that failed stays visible.
>
> 🟩 = done, landed, passing on BH p100a. 🟥 = attempted and reverted, reason recorded so
> it is not re-attempted the same way. Unmarked = outstanding.

**PRs reviewed:** tt-metal #52747, #52745, #52713, #52727, #52709
**Date:** 2026-08-14
**Scope rule used:** an entity counts as "newly promoted from blaze" if the PR adds or modifies a file under
`tt_metal/tt-llk/**/experimental/`, `tt_metal/hw/ckernels/<arch>/metal/llk_api/experimental/`, or
`tt_metal/hw/inc/api/compute/experimental/`. Deletions under `models/demos/deepseek_v3_b1/kernel_includes/`
are the demo shadow tree being retired and are not themselves test targets.

All five PRs target **Blackhole only** — every promoted compute-API header is wrapped in
`#if defined(ARCH_BLACKHOLE)` and every llk_lib/llk_api file lands only in the `tt_llk_blackhole` /
`hw/ckernels/blackhole` trees. Every test below therefore carries `blackhole_only` (or
`skip_for_wormhole` + `skip_for_quasar`).

---

## 1. Verdicts at a glance

| PR | Family | New `experimental/` entities | tt-llk coverage today | Action |
|----|--------|------------------------------|-----------------------|--------|
| #52747 | `deepseek_moe_gate` → `generalized_moe_gate` | **none** | full | **no new test** — run existing suites |
| #52745 | `ckernel_sfpu_sampling.h` | **none** | full | **no new test** — one optional hardening case |
| #52709 | `rmsnorm` / `add_rsqrt` / `eltwise_mul_scalar` | 8 promoted + 2 new | **zero** | 1 new file, 3 extensions — 🟩 `add_rsqrt` done |
| #52713 | `top32_rm` | 5 promoted + 2 new + 1 modified | **zero** (indirect: `topk_xl`) | 1 new file + 1 compile case — 🟩 compile case done |
| #52727 | `custom_mm` / `compressed_custom_mm` | 9 promoted + 1 new | partial (compressed llk_lib only) | 2 new files — 🟩 `block_uninit` done |

---

## 2. Cleanup PRs — checked, no tt-llk work needed

### #52747 — Retire the demo `deepseek_moe_gate` fork onto canonical `generalized_moe_gate`

Adds **nothing** under any `experimental/` path. 13 headers deleted from the demo shadow tree; the only
non-deletion is `unified_kernels/deepseek_moe_gate.hpp` re-pointing at
`api/compute/experimental/generalized_moe_gate.h` with `GMG_UNGROUPED_TOP8 = 0`.

The canonical target is already the best-covered family in the tt-llk suite:

- `test_generalized_moe_gate.py` / `sources/generalized_moe_gate_test.cpp` — 12 test functions, including
  the **grouped** DeepSeek path that `GMG_UNGROUPED_TOP8 = 0` selects: `test_generalized_moe_gate_grouped`,
  `test_generalized_moe_gate_sigmoid[grouped=True]` (the sigmoid + grouped combination is the DeepSeek gate
  exactly), `test_generalized_moe_gate_ties`, `test_generalized_moe_gate_shipping_config`.
- `test_sfpu_generic_moe_gate_topk.py` — the SFPU top-k functors underneath.

**Verdict: no new test.** Run `test_generalized_moe_gate.py` and `test_sfpu_generic_moe_gate_topk.py`
unchanged on the branch as the regression gate.

**One thing worth flagging to the author (not a blocker):** tt-llk's own `test_deepseek_moe_gate.py` /
`sources/deepseek_moe_gate_test.cpp` do **not** consume the tree this PR deletes. They include a *third*
fork living under `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate/device/kernel_includes/`,
reachable via the `-I../../../ttnn/cpp/ttnn/operations/experimental` entry in
`tests/python_tests/helpers/test_config.py` (carrying the TODO "remove this after kernels get moved into
Metal experimental (#52837)"). That is the fork the PR body defers to a later batch — and it is the one
whose retirement *will* require rewiring a tt-llk test source. Queue that as a known follow-up.

### #52745 — Retire the demo fork of `ckernel_sfpu_sampling.h`

Adds nothing under `experimental/`. The canonical
`hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sampling.h` landed with its suite in
#52163; this PR deletes the demo copy and rewires `unified_kernels/sampling.hpp`.

`test_sfpu_sampling.py` + `sources/sfpu_sampling_test.cpp` already cover **every entry point**, including the
two things this PR's call-site changes touch:

- `sampling_recip_init<legacy_compat>` — called at `sfpu_sampling_test.cpp:148`, swept both ways via
  `legacy_compat=[True, False]` (test_sfpu_sampling.py:212).
- `calculate_sampling_binary_first_column<SamplingBinaryOp::{add,sub,mul}>` — the collapsed dispatch,
  driven at `sfpu_sampling_test.cpp:122-126`.

**Verdict: no new test.** Run `test_sfpu_sampling.py` unchanged.

**Optional hardening (~1 hour, worth it).** The PR's stated motivation for `sampling_recip_init` is a
*cross-op* hazard: the `legacy_compat=false` reciprocal reads `vConstFloatPrgm0` for its Newton-Raphson
constant, and only `sfpu_reciprocal_init` writes the `2.0f` it expects — so a kernel that ran e.g.
`exp_tile_init` earlier computes silently wrong. The existing test always calls `sampling_recip_init`
immediately before the op, so it proves the init *works* but never proves it is *necessary*. Add a
`POLLUTER_INIT` template parameter to `sfpu_sampling_test.cpp` that runs `_init_exponential_` (or any
`vConstFloatPrgm0` writer) before the recip sequence, and cross it with a `SKIP_RECIP_INIT` switch:

| polluter | recip_init | expectation |
|----------|-----------|-------------|
| no | yes | pass (today's case) |
| yes | yes | pass — this is the case the PR exists for |
| yes | no | **must fail** — pins that the init is load-bearing |

This is the same "prove the restore is necessary" shape as the existing
`test_unpack_bcastA_B_uninit_restore.py`, so the pattern is already established in the repo.

---

## 3. #52709 — `rmsnorm` / `add_rsqrt` / `eltwise_mul_scalar`

### 3.1 Inventory

| Path | Status |
|------|--------|
| `tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h` | promoted (rename, 60% similarity) |
| `tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_A_rmsnorm.h` | **new file** |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse_api.h` | promoted |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_unpack_A_rmsnorm_api.h` | promoted |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h` | promoted (100% identical) |
| `hw/inc/api/compute/experimental/rmsnorm.h` | **new file** (blaze version + chunked mul-reduce) |
| `hw/inc/api/compute/experimental/add_rsqrt.h` | promoted |
| `hw/inc/api/compute/experimental/eltwise_mul_scalar.h` | promoted + **behavioral fix** |
| `hw/sources.cmake` → `HW_JIT_API_HEADERS` | +3 entries (ships in packaged metalium) |

**Public surface newly exposed** (`api/compute/experimental/rmsnorm.h`):
`rmsnorm_bcast_scalar_reuse_tiles_init` / `_tiles`, `rmsnorm_bcast_scalar_reuse_tiles_init_fidelity` /
`_tiles_fidelity` (explicit-fidelity + `unpack_full_transpose` axis), `rmsnorm_mul_bcast_scalar_reuse_tiles_init` /
`_tiles`, and `mul_reduce_scalar_chunked_tile<num_tiles, dst_capacity, reduce_type>`.

**Current tt-llk coverage: zero.** `grep -rl "rmsnorm|add_rsqrt|mul_scalar|dest_reuse" tests/sources/` returns
nothing for all four.

### 3.2 🟩 `add_rsqrt` — DONE (landed as its own file, not an extension)

> **What landed:** `tests/python_tests/test_sfpu_add_rsqrt.py` +
> `tests/sources/sfpu_add_rsqrt_test.cpp`, 42 passed / 14 skipped on BH p100a.
>
> Deviation from the recommendation below: a **dedicated file**, not an extension of
> `test_sfpu_binop_scalar.py`. `calculate_add_rsqrt` carries two template axes that suite
> has no notion of (`APPROX` selecting the sqrt body, `FAST_APPROX` gating the negative
> guard), and it lives in the metal `experimental/llk_sfpu/` tree, which needs the
> `#define DST_ACCUM_MODE` / `constexpr bool APPROX` preamble. `test_sfpu_sampling.py` is
> the exact precedent — a dedicated file for an `experimental/llk_sfpu` header — so that
> shape was followed instead. `SFPU_UNARY_SCALAR` is still reused for the addend.
>
> Two further departures, both forced by measurement:
> * The `FAST_APPROX` case asserts a **sign** predicate, not `isnan`. The guard's NaN
>   arrives as `+inf` on the Float16_b path, so an isnan assertion fails while the guard
>   works correctly. What holds in all six live configurations is: guard on → no negative
>   lane, guard off → negative lanes present.
> * Tolerances are measured envelopes per (body, output width), 1e-6 … 2.0e-2, replacing
>   the "loosen for approx" sketch below — which as written would have *tightened* the
>   bf16 cases 25x below the format default and failed them.

`calculate_add_rsqrt<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX>(uint32_t param0)` is a
unary SFPU op with one bit-packed float scalar — exactly the shape of the
`test_sfpu_binop_scalar.py` / `sources/sfpu_binop_scalar_test.cpp` suite, which already uses
`SFPU_UNARY_SCALAR(scalar_bits)` and a `_bits()` host helper and has precedent for a host-transformed
scalar (`ScalarDiv` inverts on the host).

Work items:
1. `helpers/llk_params.py` — add `MathOperation.AddRsqrt`.
2. `sources/sfpu_binop_scalar_test.cpp` — add an `#elif defined(SFPU_OP_ADD_RSQRT)` arm including
   `experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h` and calling `init_add_rsqrt<APPROX>()` +
   `calculate_add_rsqrt<...>` through `_llk_math_eltwise_unary_sfpu_params_`.
3. `helpers/golden_generators.py` — golden is `torch.rsqrt(x + eps)` in fp32, then
   `round_to_dest_width` for the `!fp32_dest_acc_en` `convert<vFloat16b>(Nearest)` store.
4. Add to `_SCALAR_OPS`.

Sweep axes: `eps ∈ {0.0, 1e-6, 1.0}` (`1e-6` is the production RMSNorm epsilon; `0.0` cross-checks against
the plain `MathOperation.Rsqrt` result in `test_sfpu_unary.py`), `dest_acc ∈ {No, Yes}` (drives both the
`ITERATIONS` count and the truncation branch), `APPROX ∈ {False, True}`, `FAST_APPROX ∈ {False, True}`.

**Domain note.** The binop-scalar suite does not consume `helpers/sfpu_domains.py` — it falls back to
`default_spec_for_format` = `uniform(0.1, 1.1)`, i.e. positive-only. That is fine as the default (it keeps
`x + eps > 0`), but pass an explicit `spec_A` for two extra cases worth having: `x` near `0` with `eps = 0`
(result → `+inf`, assert the inf rather than a tolerance) and large `x` (~1e4, exercises the
`_calculate_sqrt_body_` exponent path). Do **not** feed negatives — `rsqrt` of a negative is undefined for
this functor and would only test garbage.

### 3.3 🟥 `eltwise_mul_scalar` HiFi init fix → **extend `test_eltwise_binary.py`** (ATTEMPTED — see §10)

This is the PR's stated "one behavioral change" and it is **not** covered today, though it looks like it is.

`test_eltwise_binary_dest_reuse` already crosses `EltwiseBinaryReuseDestType.DEST_TO_SRCB` ×
`MathOperation.Elwmul` × `MathFidelity.{LoFi..HiFi4}` × `tile_dimensions ∈ {[32,32],[16,32],[8,32]}`. But
`sources/eltwise_binary_test.cpp` only ever calls the **general** init
`_llk_math_eltwise_binary_init_<...>(tensor_shape, ACC_TO_DEST)` (lines 113/123/142) — which is precisely the
branch the fix *switches to*. The buggy path is the llk_api **shorthand**
`llk_math_eltwise_binary_init<...>(icb0, icb0, acc_to_dest)`, and nothing in tt-llk drives it.

Reading `hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h:31-41` makes the failure mechanism
concrete and gives the test its shape:

- the **init** shorthand passes `get_operand_tensor_shape(get_operand_id(operand_A))` — the *CB's* tile
  geometry — to `_llk_math_eltwise_binary_init_`;
- the **execute** function `llk_math_eltwise_binary` (line 66) passes `ckernel::DEFAULT_TENSOR_SHAPE`.

So init configures the MOP for the CB's tile shape while execute assumes 32×32 / 4-face. Whenever the CB's
tile shape is not the default, the two disagree — and at HiFi the shape drives the inner-loop pass count, which
is why the corruption is HiFi-only and why the blaze fix hardcodes `DEFAULT_TENSOR_SHAPE` in the init.

Concrete addition: a `USE_SHORTHAND_INIT` template selector in `eltwise_binary_test.cpp` that routes through
`llk_math_eltwise_binary_init<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, REUSE_DEST_TYPE>(operand, operand, ACC_TO_DEST)`,
then cross it into `test_eltwise_binary_dest_reuse`. Expected result matrix:

| tile_dimensions | fidelity | shorthand init | expectation |
|-----------------|----------|----------------|-------------|
| `[32,32]` | any | yes | pass (shape happens to equal default) |
| `[16,32]`, `[8,32]` | LoFi | yes | pass (shape-insensitive branch) |
| `[16,32]`, `[8,32]` | HiFi2/HiFi4 | yes | **fail on main, pass after the blaze #1760 fix** |

That last row is the reproducer. It is worth writing even though the fix already landed in the promoted
header, because it converts an anecdotal model-accuracy observation (0.70 → 0.9996 on M2 MoE HiFi4) into a
deterministic LLK-level assertion, and it guards the shorthand for every other caller.

### 3.4 `rmsnorm` bcast-scalar dest-reuse → **new file**

- `tests/python_tests/test_rmsnorm_bcast_scalar_dest_reuse.py`
- `tests/sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp`

**Why a new file, not an extension.** The op is a `num_tiles`-templated MOP
(`rmsnorm_bcast_scalar_dest_reuse_configure_mop<eltwise_binary_type, num_tiles, math_fidelity>`) driven from a
*single* unpack, with SrcB sourced from DEST via `MOVD2B` under a `WAIT_SFPU | SRCB_VLD` stall
(`rmsnorm_bcast_scalar_reuse_dest_as_src`). No existing test file has that structure: `test_bcast.py` does
one-tile-per-unpack broadcasts, `test_eltwise_binary.py` has no MOP-over-N-tiles axis and no
`num_tiles`-as-template-argument plumbing.

Kernel structure — replicate `api/compute/experimental/rmsnorm.h`'s call sequence at the llk layer, which is
the established tt-llk convention (see §6):

```
UNPACK: _llk_unpack_A_rmsnorm_init_<num_tiles, SCALAR, true, DEST_TO_SRCB>(transpose, transpose, ...)
        _llk_unpack_A_<SCALAR, true, DEST_TO_SRCB>(...)          // ONE unpack for all num_tiles
MATH:   _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_<op, num_tiles, fidelity>(num_faces, acc_to_dest)
        _llk_math_rmsnorm_bcast_scalar_dest_reuse_<op, num_tiles, dest_acc, fidelity, clear_dest>(src, dst)
PACK:   pack num_tiles tiles
```

Golden: seed DEST with `num_tiles` known tiles (a datacopy pre-pass, same technique as
`_prepare_dest_reuse_inputs` in `test_eltwise_binary.py`), then apply the scalar at element `[0]` of the
unpacked operand element-wise across all `num_tiles` × all faces, in `ELWADD` or `ELWMUL`. Reuse
`helpers/golden_generators.py`'s binary golden with a bcast-scalar wrapper.

Sweep:

| axis | values | rationale |
|------|--------|-----------|
| `eltwise_binary_type` | `ELWADD`, `ELWMUL` | both instantiate distinct MOP branches (`TT_OP_ELWADD` vs `TT_OP_ELWMUL` at lines 31-64) |
| `num_tiles` | `1, 2, 3, 7, 8` (bf16) / `1..4` (fp32 DEST) | it is the MOP's outer-loop count; DEST half-sync capacity caps it |
| `math_fidelity` | `LoFi, HiFi2, HiFi4` | fidelity is a template arg on both init and execute |
| `clear_dest` | `False, True` | template arg with no other coverage |
| `dest_acc` | `No, Yes` | |
| `num_faces` | `1, 2, 4` | runtime arg to `_init_`; tiny-tile geometry |
| `unpack_full_transpose` | `False, True` | **the axis only blaze's version has** — the `_fidelity` init folds transpose into the unpack. This is new reachable surface, so it must be swept. |

Priority note: `unpack_full_transpose=True` and `clear_dest=True` are the two axes that exist *only* because
blaze's version won the reconciliation. They are the highest-value cells in this matrix.

### 3.5 `mul_reduce_scalar_chunked_tile` → **extend `test_mul_reduce_scalar.py`**

This is the single riskiest piece of new code in #52709: it is not a promoted header at all but a genuinely
new compute-API composition, with non-trivial host-side arithmetic:

```
batch_size     = dst_capacity - 1
accumulator    = batch_size
num_batches    = (num_tiles + batch_size - 1) / batch_size
last_batch_size = num_tiles - (num_batches - 1) * batch_size
```

plus a mid-loop `mul_reduce_scalar_init(icb0, icb1)` re-init for `batch > 0` and a one-shot
`llk_pack_reduce_mask_config` on `batch == 0`. Off-by-one in `last_batch_size`, a missing re-init, or a
double pack-mask config would all produce a plausible-looking but wrong scalar.

`test_mul_reduce_scalar.py` + `sources/mul_reduce_scalar_test.cpp` already exist and drive the non-chunked
op. Add a `CHUNKED` mode to the C++ source replicating the loop above, and a `test_mul_reduce_scalar_chunked`
function. Sweep `num_tiles` specifically at the boundaries the arithmetic can get wrong:

- `num_tiles = dst_capacity` (smallest legal — `static_assert(num_tiles > dst_capacity)` means start at `+1`)
- `num_tiles = dst_capacity + 1`
- `num_tiles = 2 * batch_size` (exact multiple → `last_batch_size == batch_size`)
- `num_tiles = 2 * batch_size + 1` (**ragged tail** → `last_batch_size == 1`, the off-by-one canary)
- `num_tiles = 3 * batch_size - 1`

crossed with `dst_capacity ∈ {2, 4, 8}` (the `static_assert` range is 2..8; `dst_capacity = 2` means
`batch_size = 1`, i.e. one product per batch — the degenerate case), `scaler ∈ {1.0, 0.5}`, and the existing
format/fidelity axes. Keep `B == 1.0` as the existing test does so the golden stays `scaler * sum(A)` and the
test isolates the chunking logic rather than re-testing the multiply.

---

## 4. #52713 — `top32_rm`

### 4.1 Inventory

| Path | Status |
|------|--------|
| `tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_top32_rm.h` | promoted (76%) |
| `tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_A_top32_rm.h` | promoted (57%) |
| `tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h` | promoted (79%) |
| `tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h` | **new file** (extracted shared helper) |
| `tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_topk_xl.h` | **modified** (helper removed, now includes the shared header) |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_top32_rm_api.h` | promoted |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_unpack_A_top32_rm_api.h` | promoted |
| `hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/llk_math_deepseek_top32_rm.h` | **new file** — 7 SFPU entry points |

> **Discrepancy to raise with the author.** The PR body states "Blaze's `llk_math_deepseek_top32_rm.h`
> wrapper is **not** promoted — the in-tree consumers drive the SFPU functors via the `SFPU_UNARY_CALL`
> macros already." The diff at the current head **does** add
> `hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/llk_math_deepseek_top32_rm.h` as a new file with
> seven entry points (`llk_math_deepseek_top32_rm_init`, `_local_sort`, `_merge`, `_rebuild`,
> `llk_math_deepseek_top32_of_1024_rm_pre_sorted_{prep,combine,final}`). Either the description is stale or the
> file is unintentionally included. It matters for testing: if it stays, it is public API with zero callers
> and zero tests, which is exactly the surface a tt-llk test should pin.

**Current tt-llk coverage: zero for `top32_rm`.** The only in-tree coverage anywhere is the tt-metal gtest
`tests/tt_metal/tt_metal/llk/test_top32_rm_dev.cpp` (`Top32RmDevPipelineCompletes`), which runs
`row_elements ∈ {64, 128, 160, 3232}` at a single seed and a single dest-acc setting.

`topk_xl_test.cpp` / `test_topk_xl.py` exist and are the regression net for the
`set_dst_write_addr_offset` extraction.

### 4.2 New file: `test_top32_rm.py` + `sources/top32_rm_test.cpp`

**Model it on `test_topk_xl.py`.** That is the closest analogue by a wide margin: same value+index DEST
layout discipline, same hand-built stimuli approach ("`helpers.stimuli_generator` is very awkward for these
tests"), same bf16-value / uint32-index output decoding, and it now literally shares the
`set_dst_write_addr_offset` helper. Reuse its `_decode_row_major` / `_bitcast_float32` helpers if the DEST
index layout matches; the `dst_indices_offset = 128` (2 tiles × 64 rows) convention in
`ckernel_sfpu_deepseek_top32_rm.h` suggests it will need a local variant.

**Two kernel modes**, mirroring the two tt-metal dev kernels and selected by a `TOP32_MODE` template
parameter (the gtest picks between them at `row_elements >= 1024`):

- `MODE_INCREMENTAL` (`row_elements < 1024`) — the 64-elements-at-a-time path:
  `_llk_unpack_A_top32_rm_` transposing load (16 elements into the first row of each of 4 faces, remainder
  padded to `-inf`) → `llk_math_deepseek_top32_rm_local_sort` (bitonic phases/steps) → per-chunk
  `_merge` + `_rebuild`.
- `MODE_PRESORTED` (`row_elements >= 1024`) — the whole-1024-chunk path:
  `llk_math_deepseek_top32_of_1024_rm_pre_sorted_prep<top_min>` → `_pre_sorted_combine` → `_pre_sorted_final`.

**Sweep.** The gtest's four sizes leave the interesting boundaries untested:

| axis | values | rationale |
|------|--------|-----------|
| `row_elements` | `32, 63, 64, 65, 128, 160, 1023, 1024, 1088, 2048, 3232` | `64`/`1024` are the chunk boundaries; `63`/`65`/`1023` are the ragged tails that exercise the `-inf` padding; `32` is fewer elements than the top-k width |
| `is_fp32_dest_acc_en` | `No, Yes` | **switches the index load/store `InstrModLoadStore` between `LO16` and `INT32`** (`bitonic_top32_load8`/`load16`). The gtest runs one setting; this axis can silently corrupt indices only. |
| `idir` / sort direction | both | `_local_sort(idir)`, `_merge<idir>`, `_rebuild(idir, skip_second)` all take it |
| `top_min` | `False, True` | template arg on `_pre_sorted_prep_`, no other coverage |
| `skip_second` | `False, True` | runtime arg on `_rebuild` |
| stimuli shape | shuffled-distinct, **all-ties**, **partial ties at the k=32 cut**, all-`-inf` row, single non-`-inf` element | ties are where a top-k index tie-break becomes non-deterministic; the gtest uses one shuffled seed only |

**Golden.** `argsort` descending over `row_elements`, take 32. Values compare **exactly** — the op is a
pure permutation of bf16 inputs with no arithmetic, so no tolerance is warranted, and a tolerance would mask
a wrong-lane bug. Indices compare exactly too, with the tie cases either restricted to distinct values or
asserted as a set rather than a sequence.

### 4.3 🟩 The `set_dst_write_addr_offset` extraction — DONE (one compile case)

The PR's stated reason for the new shared header is that `ckernel_sfpu_topk_xl.h` already defined an
identical `set_dst_write_addr_offset`, so a kernel including both headers would hit a redefinition error
(blaze papers over it with `#ifndef` guards). **Nothing in the tree compiles both headers into the same
TRISC-math translation unit**, so the redefinition this PR fixes is currently unreachable by any test.

Add a variant to `sources/top32_rm_test.cpp` (or a dedicated compile-only case) that includes **both**
`sfpu/experimental/ckernel_sfpu_topk_xl.h` and `sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h` in the
math TU and calls one entry point from each. Cheap, and it is the only thing that actually pins the
extraction. Also note the PR's own observation: the shared copy keeps the demo's `LLK_ASSERT`, so `topk_xl`
callers gain a Dst-offset bounds check — run this case with `ENABLE_LLK_ASSERT` set to exercise it, and pass
an out-of-range offset in a negative variant if the harness supports expected-assert tests.

Plus: run `test_topk_xl.py` unchanged on the branch. It is the direct regression check for the header edit.

---

## 5. #52727 — `custom_mm` / `compressed_custom_mm`

### 5.1 Inventory

| Path | Status | tt-llk coverage today |
|------|--------|-----------------------|
| `tt-llk/.../llk_lib/experimental/llk_math_custom_mm.h` | promoted | **none** |
| `tt-llk/.../llk_lib/experimental/llk_unpack_AB_custom_mm.h` | promoted (84%) | **none** |
| `tt-llk/.../llk_lib/experimental/llk_math_compressed_custom_mm.h` | promoted (51% — clang-format) | ✅ `matmul_custom_compressed_test.cpp` |
| `tt-llk/.../llk_lib/experimental/llk_unpack_AB_compressed_custom_mm.h` | **new file in-tree** (was vendored) | ✅ same |
| `llk_api/experimental/llk_math_custom_mm_api.h` | promoted | none |
| `llk_api/experimental/llk_math_compressed_custom_mm_api.h` | promoted | none |
| `llk_api/experimental/llk_unpack_AB_custom_mm_api.h` | promoted | none |
| `llk_api/experimental/llk_unpack_AB_compressed_custom_mm_api.h` | promoted | none |
| `hw/inc/api/compute/experimental/custom_mm.h` | promoted + `ARCH_BLACKHOLE` guard + **uninit change** | none |
| `hw/inc/api/compute/experimental/compressed_custom_mm.h` | same | none |
| `hw/sources.cmake` → `HW_JIT_API_HEADERS` | +2 entries | — |
| `tt-llk/tests/python_tests/test_matmul_custom_compressed.py` | **rewired** (drops the `VENDORED_LLK_LIB` fixture) | — |
| `tt-llk/tests/sources/matmul_custom_compressed_test.cpp` | **rewired** to `experimental/` includes | — |

Compute-API surface per family: `*_block_init`, `*_block_init_short`, `*_block`, `*_block_unpack`,
`*_block_math`, `*_block_uninit`.

**Good news first:** the compressed llk_lib pair is already covered. `matmul_custom_compressed_test.cpp`
drives `_llk_unpack_AB_compressed_custom_mm_init_` / `_llk_unpack_AB_compressed_custom_mm_` and
`_llk_math_compressed_custom_mm_init_` / `_llk_math_compressed_custom_mm_` across BFP0/2/4/8 and a shape
sweep. This PR only changes the include spelling and deletes the vendored-path fixture, so **running
`test_matmul_custom_compressed.py` on the branch is the direct validation** — and it is now a *better* test
than before, since it consumes the canonical headers instead of the demo tree.

### 5.2 🟩 DONE — `test_custom_mm_uninit_restore.py` + `sources/custom_mm_uninit_restore_test.cpp`

**This is the highest-value new test in the whole batch.** It is the only behavioral delta in #52727 and it
has zero coverage at any layer.

`custom_mm_block_uninit<dense_packing, restore_tile_pack_mop>` does two conditional state restores:
- `dense_packing` → `cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(...)` back to the 64-row stride;
- `restore_tile_pack_mop` → `_llk_pack_mop_config_<PackMode::Default>()`.

Both are classic cross-op state leaks, and tt-llk has an established pattern for exactly this shape:
`test_unpack_tilize_uninit_restore.py`, `..._block.py`, `..._tiny.py`, `test_unpack_bcastA_B_uninit_restore.py`.
Follow that pattern literally — its docstring even spells out the discipline ("NO existing test calls this
function at all").

Kernel shape:

```
run 0: custom_mm block, packer MOP replaced by pack_block_contiguous_init (± dense_packing stride)
       custom_mm_block_uninit<dense_packing, restore_tile_pack_mop>()
run 1: plain _llk_pack_<..., PackMode::Default> datacopy of a known tile, NO packer re-init
```

Assertion matrix — note that both polarities are assertions, not just the happy path:

| `restore_tile_pack_mop` | run-1 expectation |
|------------------------|-------------------|
| `true` | matches the datacopy golden — the restore works |
| `false` | **differs** from the datacopy golden — pins the documented "the MOP is owned by whichever init programmed it" contract, so a future accidental unconditional restore is caught |

Cross with `dense_packing ∈ {False, True}` to cover the `Wstride` RMW in the same kernel (same failure
shape, different config register — a `dense_packing` block followed by an unrestored default pack writes
tiles 32 rows apart instead of 64). Run the identical matrix for
`compressed_custom_mm_block_uninit<dense_packing, restore_tile_pack_mop>`.

> **Second discrepancy to raise with the author.** The PR body describes the fix as unconditional:
> "`*_block_uninit()` now restores the Default tile-pack MOP." At the current head it is **opt-in** —
> `template <bool dense_packing = false, bool restore_tile_pack_mop = false>`, defaulting to `false`, with a
> comment explaining that an unconditional `_llk_pack_mop_config_<Default>()` would install fixed 32×32
> geometry and clobber the 1×32 configuration this family targets. So the "all ten demo `*_block_uninit`
> callers are exercised" claim in the notes is about the *old* behavior unless those call sites were switched
> to `<..., true>`. Worth confirming which callers opt in — and it is a good argument for testing both
> polarities as above rather than assuming the restore is always on.

### 5.3 New file: `test_matmul_custom_mm.py` + `sources/matmul_custom_mm_test.cpp`

The plain (non-compressed) `custom_mm` llk_lib and llk_api pair has no tt-llk coverage. Note that the
similarly-named existing `test_matmul_custom.py` / `matmul_custom_test.cpp` drive a **different** family —
`experimental/llk_math_matmul_custom_no_mop.h` — so it cannot simply be extended, and the new file should be
named to avoid the confusion (`test_matmul_custom_mm.py`, not `test_custom_matmul.py`).

Mirror `matmul_custom_compressed_test.cpp`'s three-TRISC structure — it is the sibling family and is already
wired to the canonical `experimental/` headers, so the port is mostly mechanical.

Sweep:

| axis | values |
|------|--------|
| shapes `(rt, kt, ct)` | `ct ∈ {1..6, 8, 10, 12, 14, 16}` — the set blaze documents as tested |
| **`ct ∈ {7, 9, 11}`** | the doc-split holes (see below) |
| template bools | `transpose`, `split_acc`, `dense_packing`, `read_transposed`, `clear_src`, `finalize` |
| `math_fidelity` | `LoFi, HiFi2, HiFi4` |
| `dest_acc` | `No, Yes` |

**On the ct doc split.** The PR notes that blaze's comment lists the tested set `{1..6, 8, 10, 12, 14, 16}`
while the demo's says "any 1–16", that the difference is comment-only, and that "the code enforces neither" —
demo's wording was kept. That is an unresolved factual question sitting in a comment, and it is cheap to
settle: include `ct ∈ {7, 9, 11}` in the sweep. If they pass, the demo's wording is right and the doc is now
backed by a test. If they fail, the promoted header is shipping a documented-but-broken range and the comment
needs to become a `static_assert`. Either outcome is worth more than the comment.

Golden: reuse whatever `test_matmul_custom.py` uses from `helpers/golden_generators.py` (`MatmulGolden`);
`helpers/matmul_sweep.py` already exists for shape enumeration.

---

## 6. Shared infrastructure work

**The compute-API layer is not directly testable from tt-llk — replicate its sequence instead.**
`-I../../hw/inc` *is* on the tt-llk compile line (`helpers/test_config.py:530`), so
`api/compute/experimental/*.h` technically resolves. But those headers need the metal JIT environment: the
`UNPACK()`/`MATH()`/`PACK()` thread-elision macros, `ALWI`, the CB APIs, `get_compile_time_arg_val`. The
established convention — set by `sources/fast_tilize_metal_api_test.cpp` + `test_fast_tilize_metal_api.py`,
whose docstring reads "Replicates ttnn tilize compute kernel flow (`compute_kernel_hw_startup` +
`fast_tilize_init`/`block`/`uninit`) through the LLK test infra" — is to **reproduce the compute-API call
sequence using `llk_api` / `llk_lib` calls** inside the tt-llk kernel. Every plan above follows that. The
compute-API wrappers themselves stay covered by the tt-metal gtests and the demo tests.

`-I../../hw/ckernels/blackhole/metal/llk_api` is also on the path, so the new `experimental/llk_sfpu/...`
and `experimental/llk_*_api.h` headers are includable as `"experimental/<name>.h"` with no fixture — the
`VENDORED_LLK_LIB` include-path hack that #52727 deletes is exactly what promotion buys.

New helper plumbing needed:

- **`helpers/llk_params.py`** — `MathOperation.AddRsqrt`.
- **`helpers/test_variant_parameters.py`** — new `TemplateParameter`s: `NUM_TILES_TEMPLATE` (rmsnorm's
  `num_tiles` is a *template* arg, unlike the existing runtime `TILE_COUNT`), `CLEAR_DEST`,
  `UNPACK_FULL_TRANSPOSE`, `DST_CAPACITY`, `USE_SHORTHAND_INIT`, `RESTORE_TILE_PACK_MOP`, `DENSE_PACKING`,
  `TOP32_MODE`, `TOP32_TOP_MIN`, `SORT_DIRECTION`, `POLLUTER_INIT`.
- **`helpers/golden_generators.py`** — `AddRsqrtGolden` (or a branch in the binop-scalar golden),
  `RmsnormBcastScalarGolden`, `Top32RmGolden` (likely an extension of `TopKXLGolden`),
  `MulReduceScalarChunkedGolden`.
- **`conftest.py`** — all new tests are `blackhole_only`.
- **Test-isolation caution.** Two of these tests (`custom_mm` uninit-restore, the optional sampling polluter)
  deliberately leave hardware state dirty between runs. Per the tt-llk notes, HW state leaks between kernel
  reconfigurations and a `TENSIX TIMED OUT` there must **not** be masked with `tt-smi -r` — that would hide
  the very reconfig escape the test is designed to catch.

---

## 7. Sequencing

**Verify first (no new code, run on each branch):**

| # | Action | For PR | Status |
|---|--------|--------|--------|
| 🟩 V1 | `test_matmul_custom_compressed.py` — direct regression for the include rewiring | #52727 | **582 passed** |
| 🟩 V2 | `test_topk_xl.py` — regression for the `set_dst_write_addr_offset` extraction | #52713 | **71 passed** |
| 🟩 V3 | `test_sfpu_sampling.py` | #52745 | **51 passed, 93 skipped** |
| 🟩 V4 | `test_generalized_moe_gate.py` **89 passed**, `test_sfpu_generic_moe_gate_topk.py` **24 passed** | #52747 | **DONE** |

**P0 — real behavioral changes with zero coverage:**

| # | Work | PR | Est. / Status |
|---|------|----|---------------|
| 🟩 1 | `test_custom_mm_uninit_restore.py` + source (both polarities × `dense_packing` × both families) | #52727 | **DONE** — 30 passed, 2 xfailed, 32 skipped |
| 2 | `mul_reduce_scalar_chunked` extension to `test_mul_reduce_scalar.py` | #52709 | ~1 d |
| 🟥 3 | `USE_SHORTHAND_INIT` extension to `test_eltwise_binary.py` (the HiFi #1760 reproducer) | #52709 | **attempted, reverted** — see §10 |

**P1 — promoted-as-is code, no coverage; regression net for future edits:**

| # | Work | PR | Est. / Status |
|---|------|----|---------------|
| 🟩 4 | `add_rsqrt` — landed as `test_sfpu_add_rsqrt.py` + source | #52709 | **DONE** — 42 passed, 14 skipped, mutation-verified |
| 5 | `test_rmsnorm_bcast_scalar_dest_reuse.py` + source | #52709 | ~2 d |
| 6 | `test_matmul_custom_mm.py` + source (incl. the `ct ∈ {7,9,11}` doc question) | #52727 | ~2 d |
| 7 | `test_top32_rm.py` + source — two modes, index goldens, largest single effort | #52713 | ~3–4 d |
| 🟩 8 | Both-headers-in-one-TU compile case for `set_dst_write_addr_offset` | #52713 | **DONE** — 3 passed |

**P2 — optional hardening:**

| # | Work | PR |
|---|------|----|
| 9 | `POLLUTER_INIT` × `SKIP_RECIP_INIT` matrix in `test_sfpu_sampling.py` | #52745 |

---

## 8. Open questions for the PR authors

1. **#52713** — is `llk_api/experimental/llk_sfpu/llk_math_deepseek_top32_rm.h` intentionally promoted? The
   PR body says it is not. If it stays, it is seven public entry points with no in-tree caller and no test.
2. **#52727** — `restore_tile_pack_mop` defaults to `false`, but the PR body describes the restore as
   unconditional. Which of the ten demo `*_block_uninit()` call sites opt in to `true`? The +22-instruction
   pack-TRISC delta reported in the notes suggests at least one does.
3. **#52727** — is `ct ∈ {7, 9, 11}` actually supported? The comment says 1–16, the tested set skips them,
   and nothing enforces either. Settling this in a test is item 6 above.
4. **#52709** — `mul_reduce_scalar_chunked_tile` is new code rather than a reconciliation. Was it validated
   beyond the `test_rmsnorm` run cited (6 passed), and at which `(num_tiles, dst_capacity)` pairs? That
   determines how much of item 2 is confirmation versus first-time coverage.
5. **All** — `HW_JIT_API_HEADERS` now ships five more `experimental/` compute headers in packaged metalium.
   Is there a packaging/compile gate that catches a header added to `experimental/` but missing from
   `sources.cmake`? Neither #52713 nor #52745 touches `sources.cmake`; #52713 adds no compute-API header, so
   it looks correct, but a gate would make that verifiable rather than reviewed.


---

## 9. Implementation status and corrections from silicon

Branch: **`ldjurovic/llk-tests-blaze-promotions`** (pushed). It merges the three open PRs
onto `origin/main` (one trivial `sources.cmake` conflict, both sides' entries kept) so the
promoted headers are present. Rebase the test commits onto main once the PRs land — they
touch only `tt_metal/tt-llk/tests/`, so they carry cleanly.

### Done, verified on BH p100a

| Item | Files | Result |
|------|-------|--------|
| 🟩 P1-4 `add_rsqrt` | `tests/sources/sfpu_add_rsqrt_test.cpp`, `tests/python_tests/test_sfpu_add_rsqrt.py` | 42 passed, 14 skipped |
| 🟩 P0-1 `custom_mm` uninit-restore | `tests/sources/custom_mm_uninit_restore_test.cpp`, `tests/python_tests/test_custom_mm_uninit_restore.py` | 30 passed, 2 xfailed, 32 skipped |
| 🟩 P1-8 sort-header coexistence | `tests/sources/sort_headers_coexist_test.cpp`, `tests/python_tests/test_sort_headers_coexist.py` | 3 passed |

All four verification items (§7 V1–V4) run green on the merged branch:

| Suite | For | Result |
|---|---|---|
| `test_matmul_custom_compressed.py` | V1 / #52727 | 582 passed |
| `test_topk_xl.py` | V2 / #52713 | 71 passed |
| `test_sfpu_sampling.py` | V3 / #52745 | 51 passed, 93 skipped |
| `test_generalized_moe_gate.py` | V4 / #52747 | 89 passed |
| `test_sfpu_generic_moe_gate_topk.py` | V4 / #52747 | 24 passed |
| `test_eltwise_binary.py` | regression baseline | 4388 passed, 72 skipped |

V3 and V4 confirm §2's verdict that #52745 and #52747 need no new tt-llk tests: the
canonical targets they rewire onto are already fully covered.

### Not started

P0-2 (`mul_reduce_scalar` chunked), P0-3 (shorthand-init HiFi reproducer), P1-5 (rmsnorm
dest-reuse), P1-6 (plain `custom_mm` matmul), P1-7 (`top32_rm`). §3–§5 still describe the
intended shape for each, with the corrections below applied.

### Correction 1 — the `restore_tile_pack_mop` test needs mismatched tile geometry

§5.2 assumed a leftover block-contiguous MOP would corrupt any following
`_llk_pack_<Default>`. It does not. Measured: with the run-0 block MOP programmed at **4
faces** — the same geometry the restore installs — run 1 is byte-correct whether or not
the restore runs, so the flag is **unobservable**. The restore re-establishes *geometry*,
nothing broader.

Programming the run-0 MOP at **2 faces** (a 16x32 tiny tile) makes it observable, and is
the hazard the header's own comment names ("wrong for 1x32 follow-ons"): the un-restored
MOP then packs half of each tile, a 0.50 per-tile match. The landed test uses 2 faces for
the discriminating cases and keeps a 4-face test that pins the no-op-at-matching-geometry
behaviour, since that is *why* the flag is opt-in rather than unconditional.

### Correction 2 — a real defect: the `dense_packing` W-stride is not format-aware

Found while building P0-1, and the most substantive result so far.

`cpack_common.h set_packer_strides` — the canonical writer — computes

```
w_stride = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(pack_src_format)
```

but both `custom_mm.h` and `compressed_custom_mm.h` spell the same expression with a
literal `* 2`, i.e. hardcoded for a **16-bit** pack source, at four sites (init + uninit
in each family):

```
init   dense:   (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2  = 1024
uninit restore:  TILE_NUM_FACES      * FACE_C_DIM * FACE_R_DIM * 2  = 2048
```

With a Float32 pack source `datum_size_in_bytes` is 4, so the correct values are 2048 and
4096. The uninit therefore does not restore what `_llk_pack_init_` programmed. Measured
(Float32 in/out, `dest_acc=Yes`, `dense_packing=True`): run 1 matches on tile 0 only
(0.25 overall) **regardless of `restore_tile_pack_mop`** — the W-stride restore cannot
recover. The 16-bit path is fully correct.

Pre-existing rather than introduced by #52727 (the demo's structure was kept), but the
promotion ships it in packaged metalium via `HW_JIT_API_HEADERS`, widening the blast
radius. Landed as an `xfail` with the full explanation, so the suite stays green and
flips to XPASS when the constants become format-aware — or when a `static_assert`
restricts `dense_packing` to 16-bit pack sources, if that is the intended contract.
**This needs an owner decision; it is the one item here that is a product question, not a
test question.**

### Correction 3 — the two sort families cannot both be initialized in one kernel

§4.3 proposed calling an entry point from each header to prove they coexist. Compiling
both in one TU works (the extraction is sound — that is the assertion worth having), but
calling `_top32_rm_init_()` and `_topk_xl_init_<K, fused>()` in the same kernel **hangs
the math thread**: both program overlapping ADDR_MOD slots, the MOP and the REPLAY
buffer. Not a defect, and no real kernel does it, but it bounds the claim — the PR's
guarantee is translation-unit coexistence, not simultaneous liveness. Anyone fusing the
two families must re-init between them. The landed test therefore calls only the shared
helper, with the constraint documented in the driver.

### Correction 4 — P0-3's reproducer does not need the CB interface

§3.3 proposed driving the llk_api shorthand `llk_math_eltwise_binary_init` directly. That
needs `get_operand_tensor_shape` and the metal CB interface, which tt-llk kernels do not
have. It is also unnecessary: the shorthand's only relevant difference is *which tensor
shape reaches the init*, so the mismatch reproduces with no CB at all —

- pre-fix pair: `_llk_math_eltwise_binary_init_(tiny_shape)` + `_llk_math_eltwise_binary_(DEFAULT_TENSOR_SHAPE)`
- post-fix pair: `_llk_math_eltwise_binary_init_(DEFAULT_TENSOR_SHAPE)` + same execute

`eltwise_binary_test.cpp` currently passes `tensor_shape` to *both*, which is why the
existing dest-reuse sweep cannot see the bug. Two template flags selecting the shape
source for init and execute independently give the 2x2, with the pre-fix cell expected to
fail at HiFi on tiny tiles. Cheaper and more faithful than the §3.3 sketch.

### Note on tooling

Run tests through `tt-llk/.claude/scripts/run_test.sh` (`count` / `compile` / `run`), not
raw pytest. Two gotchas cost time: a `--k` expression containing brackets or commas
mangles the pytest args and surfaces as an opaque xdist worker crash (use `--test-id`, or
a bracket-free `--k`); and `tests/.venv` must be created with
`source ./setup_external_testing_env.sh` — `setup_testing_env.sh` alone only fetches SFPI
and assumes the Docker image's Python environment.


---

## 10. Item 3 (`eltwise_mul_scalar` HiFi init fix) — attempted and reverted

Worth recording in full, because §3.3 and Correction 4 in §9 are **both wrong** about the
mechanism, and the next person should not re-run this experiment.

### What was tried

A `BINARY_SHAPE_MODE` template switch in `eltwise_binary_test.cpp` selecting which
`TensorShape` reaches the binary init vs the binary execute, on the theory (Correction 4)
that the production llk_api pair is asymmetric —
`llk_math_eltwise_binary_init` forwards `get_operand_tensor_shape(operand)` while
`llk_math_eltwise_binary` forwards `DEFAULT_TENSOR_SHAPE` — so:

* mode 0 `init=real, exec=real` — the suite's existing, self-consistent behaviour
* mode 1 `init=real, exec=DEFAULT` — "production before the fix"
* mode 2 `init=DEFAULT, exec=DEFAULT` — "production after the fix"

The switch was built, defaulted to mode 0, and confirmed inert: the full
`test_eltwise_binary.py` sweep stayed at **4388 passed, 72 skipped**.

### Why it was reverted

Mode 1 on a tiny tile **hangs the math thread** (BH p100a, device reset required). The
cause is concrete: `_llk_math_eltwise_binary_` derives `num_faces` and `face_r_dim` from
the `tensor_shape` it is handed (`llk_math_eltwise_binary.h` ~line 600). Forcing it to
`DEFAULT_TENSOR_SHAPE` on a `[16, 32]` / 2-face tile makes the math thread issue four
faces' worth of ops against a packer configured for two, and the MATH_PACK handshake
deadlocks.

That is a hang, not the silent corruption the blaze report describes (M2 MoE HiFi4
accuracy 0.70 → 0.9996). So the shape-pairing theory does not reproduce the bug — it
produces a *different*, harsher failure that production evidently never hits. A test that
hangs the device is worse than no test, so the change was reverted in full and
`test_eltwise_binary.py` re-verified at 4388 passed.

### What this rules out, and where to look next

If production really ran `init=real / exec=DEFAULT` on a non-default CB it would hang, not
mis-compute. Two consequences:

1. **The demo's CBs for this call site are almost certainly 32x32 / 4-face.** In that case
   `get_operand_tensor_shape` returns `DEFAULT_TENSOR_SHAPE` anyway and the shape argument
   is identical on both sides — meaning the tensor shape is **not** the delta at all, and
   the "mis-specializes the tile shape" wording in #52709 is describing the symptom rather
   than the mechanism.

2. **The remaining candidate delta is fidelity, not shape.** The shorthand applies
   `get_effective_math_fidelity<eltwise_binary_type, math_fidelity>()`
   (`llk_math_binary_api.h:38`) before forwarding to `_llk_math_eltwise_binary_init_`,
   whereas the blaze fix calls `_llk_math_eltwise_binary_init_` directly with the **raw**
   `MATH_FIDELITY`. The execute path applies the effective-fidelity transform too
   (`:59`), so the fix arguably *introduces* an init/execute fidelity difference rather
   than removing one — which is exactly the kind of asymmetry that would perturb an
   ELWMUL dest-reuse accumulation at HiFi while leaving LoFi byte-identical (the fix's own
   `if constexpr (MATH_FIDELITY != LoFi)` gate is consistent with this).

So the next attempt should sweep **`get_effective_math_fidelity` applied vs not** on the
init, holding the shape fixed at `DEFAULT_TENSOR_SHAPE` on both sides, with dest-reuse
ELWMUL at HiFi2/HiFi4. That is a one-line switch in the same place and carries no
face-count mismatch, so it should not hang.

Before building it, read `get_effective_math_fidelity` for `ELWMUL` — if it is the
identity for ELWMUL then this theory is dead too, and the right next step is to ask
the #52709 author for the failing demo configuration (which CB geometry, which fidelity,
which call site) rather than to keep guessing from the LLK layer.

### Still not started

Items **2** (`mul_reduce_scalar` chunked), **5** (rmsnorm dest-reuse), **6** (plain
`custom_mm` matmul), **7** (`top32_rm`) and **9** (sampling polluter) are untouched. §3.4,
§3.5, §4.2, §5.3 and §2 describe them and, unlike §3.3, nothing measured so far
contradicts those plans.
