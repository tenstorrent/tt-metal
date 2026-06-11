# Handoff: Src zero-substitution flag → math-side state tracker (tt-llk #960, #966)

**Branch:** `ncvetkovic/disable-src-zero-flag-restorable` · **PR:** [#46511](https://github.com/tenstorrent/tt-metal/pull/46511) (DO NOT REVIEW / WIP)
**Status:** WH (Wormhole) fully validated on silicon. **One open blocker: a Blackhole-only regression in `test_layer_norm_sharded_single_stage` that does NOT reproduce on Wormhole.** This handoff is for an instance running on **Blackhole hardware** to root-cause and fix it.

---

## 1. What this change does

`ALU_ACC_CTRL_Zero_Flag_disabled_src` is only read by the **math** thread (MOVA2D/MOVB2D/ELWADD/MVMUL — `FlushDenormals = !flag`; a Src datum whose low 8 bits are zero is flushed to 0). So we moved ownership of the flag **off the unpack side** into a **math-thread state tracker**, modeled on Ana Mokan's Quasar "ALU data format config state" (PR #43540).

### #960 — why the flag matters
`MOVA2D.md`: `if (FlushDenormals && !(SrcAVal & 0xff)) SrcAVal = 0;`. For floats this is denormal flushing; for **UInt16** (HW "Integer 16", the only 16-bit-int dst format) a value like `256` (`0x0100`) has a zero low byte and is silently corrupted → flag must be 1 for UInt16. Documented in `ckernel::requires_disabled_src_zero_flag` (common `ckernel_defs.h`).

### #966 — the tracker
`ckernel::math::SrcZeroFlagState` in `cmath_common.h` (per-arch), states + skip-if-set:
- **DEFAULT** — flag follows operand formats (`UInt16→1`, else `0`). Established in `_llk_math_hw_configure_` + all `_llk_math_reconfig_data_format_*` (cached srcA/srcB formats make single-operand reconfig correct). Clears stale op-state before FP matmul/binary/reduce.
- **UNARY_PRESERVE** — flag `=1`. Asserted in metal `unary_op_init_common` **after** `llk_math_hw_configure` (so it's the last writer, not clobbered) to preserve bf16 `-0.0` (tt-metal #18346). **Scoped to eltwise-unary/SFPU only** — must NOT be on every datacopy (see §5).
- **MOV_OPS** — flag `=1`. Asserted in the **execute** of `transpose_dest` (32b) and the 32b unpack-to-dest `datacopy` branch (WH), for the hi16/lo16 MOVB2D.

Removed: unpack-side flag write in `configure_unpack_AB`, `_llk_unpack_reconfig_zero_src_flag_`, and the `disable_src_zero_flag` template param (`configure_unpack_AB` / `_llk_unpack_hw_configure_` / `llk_unpack_hw_configure` / `unary_op_init_common` / `transpose_wh_init` / `eltwise_unary.h`).

**Arch differences:** Blackhole `reduce` and `transpose_dest` use `ALU_ACC_CTRL_Fp32_enabled` for the MOVD2B/B2D workaround (Issue #449), **not** the zero flag. BH `reduce` is **unchanged** by this PR. BH 32b datacopy keeps its own #449 `Fp32_enabled` dance and `_invalidate_src_zero_flag_state_()`s the tracker afterward.

---

## 2. Commits on the branch

```
3d215470c94  [LLK] Scope Src zero-flag-disable to unary/SFPU + 32b datacopy, not all datacopy
32640ca965d  [LLK][BH] Mirror math-side Src zero-flag state tracker to Blackhole
96bb4dacf84  [LLK][WH] Math-side ALU data-format state tracker owns the Src zero flag [WIP]
33160c931ad  [LLK] Document and centralize the Src zero-substitution flag (#960)
```

## 3. Files changed (per arch — WH and BH mirror each other)
- `tt_metal/tt-llk/tt_llk_{wormhole_b0,blackhole}/common/inc/ckernel_defs.h` — `requires_disabled_src_zero_flag` predicate.
- `.../common/inc/cmath_common.h` — `SrcZeroFlagState` enum + `_configure_{default,unary_preserve,mov_ops}_zero_flag_state_` + `_invalidate_src_zero_flag_state_` + `_set_src_zero_flag_`.
- `.../common/inc/cunpack_common.h` — removed flag write + template param.
- `.../llk_lib/llk_unpack_common.h` — removed template param + `_llk_unpack_reconfig_zero_src_flag_`.
- `.../llk_lib/llk_math_common.h` — DEFAULT in hw_configure + 3 reconfig fns.
- `.../llk_lib/llk_math_eltwise_unary_datacopy.h` — MOV_OPS in 32b branch (WH); BH keeps #449 dance + invalidate.
- `.../llk_lib/llk_math_transpose_dest.h` — MOV_OPS in 32b execute.
- `tt_llk_wormhole_b0/llk_lib/llk_math_reduce.h` — WH only: MOV_OPS + DEFAULT-restore around the enforce_fp32 transpose. **(BH reduce untouched.)**
- `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_unpack_common_api.h` — removed template param + reconfig flag calls.
- `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h` — UNARY_PRESERVE after hw_configure; dropped `<…,true>`.
- `tt_metal/hw/inc/api/compute/transpose_wh.h` — collapsed int32 branch to one `hw_configure`.
- `tt_metal/tt-llk/tests/sources/{unpack_A,ttnn_where}_test.cpp`, `tests/python_tests/test_unpack_A.py` — drop removed template arg.

---

## 4. Build / test environment (on the BH host)

```bash
export TT_METAL_HOME=$(pwd)        # repo root
export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=blackhole         # (wormhole_b0 on WH)
./build_metal.sh --build-tests     # if not already built (build_Release/)
./create_venv.sh                   # if python_env/ missing
source python_env/bin/activate
```

**CRITICAL:** kernels are JIT-compiled and cached at `~/.cache/tt-metal-cache/`. After editing any LLK header you MUST clear it or your edit won't take effect:
```bash
rm -rf ~/.cache/tt-metal-cache/
```
(0/N JIT cache hits in the test output = clean recompile.)

---

## 5. THE OPEN BLOCKER — Blackhole layernorm regression

**Failing test (deterministic, BH-only):**
```bash
rm -rf ~/.cache/tt-metal-cache/
pytest "tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage[dtype=torch.bfloat16-tensor_type=ascending_values_repeated_rows-two_stage=False-use_welford=False-h=256-w=512-num_cores_h=4-num_cores_w=8-block_ht=2-block_wt=2-subblock_wt=2]" -q
```
Failure: `[FROBENIUS FAILED] Relative Frobenius norm 2.453613e-02 > threshold 0.014` (byte-identical across runs → deterministic; ~1.75× over threshold → a systematic numeric shift, not corruption).

**It PASSES on Wormhole n150** (ran the exact node locally). It is **green on `main` BH** (3 recent failed main BH post-commit runs fail only on `deepseek blitz`, never fused groups). So this PR introduces it, BH-only.

**Kernel:** `ttnn/.../normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp` — uses `reduce_init`/`reduce_tile` (×5) + `reconfig_data_format` (×19) + `binary_op_init_common`. **No** `copy_tile`, `transpose_dest`, or `unary_op_init_common`.

**Therefore the regression can only come from the BH `_llk_math_hw_configure_` / `_llk_math_reconfig_data_format_*` DEFAULT additions** (the only changed code that path touches; BH `reduce` is unchanged).

### Ruled out
- datacopy scoping (Frobenius identical before/after the scoping commit; layernorm doesn't use datacopy).
- BH reduce (verified `git diff main...HEAD` empty for `tt_llk_blackhole/llk_lib/llk_math_reduce.h`).
- flaky / pre-existing (green on main BH).

### The puzzle / leading hypotheses
For all-float layernorm the flag value should be `0` exactly as before (`requires_disabled_src_zero_flag` uses the same `==UInt16` condition the old `configure_unpack_AB` used). Two things are genuinely *new* on this path:
1. **Added `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU)` in `_set_src_zero_flag_`** — the original in-op zero-flag writes (datacopy/reduce/transpose) were **bare `cfg_reg_rmw_tensix`** with NO stall. **Leading suspect.**
2. The zero-flag=0 write **moved from the unpack thread** (`configure_unpack_AB`) **to the math thread** (`hw_configure`) — possible cross-thread cfg-ordering / ALU_ACC_CTRL RMW interaction with the reduce's `Fp32_enabled` toggles (same register).

Static reasoning couldn't explain a value change from either — hence the need for HW.

---

## 6. Plan for the BH instance

1. **Reproduce** §5 on BH (clear cache first). Confirm `2.45e-2` Frobenius.
2. **Bisect, in order (clear cache between each):**
   - **(a)** In `tt_llk_blackhole/common/inc/cmath_common.h`, make `_set_src_zero_flag_` a **bare** `cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(disable ? 1 : 0);` (drop the `TTI_STALLWAIT`). Re-run §5. ← most likely fix.
   - **(b)** If still failing: temporarily make BH `_llk_math_hw_configure_` / `reconfig` **not** call `_configure_default_zero_flag_state_` (revert to original no-op for float) and instead keep the operand-driven write on the unpack side (restore the `configure_unpack_AB` write for BH only) — confirms whether the unpack→math move is the cause.
   - **(c)** Add `dprint`/read-back of `ALU_ACC_CTRL_Zero_Flag_disabled_src` (and `Fp32_enabled`) inside the layernorm compute loop on BH to see the actual flag value vs `main`. `tt_metal/hw/inc/api/debug/dprint_tensix_pack.h` already prints `ALU_ACC_CTRL_Zero_Flag_disabled_src`.
3. Whatever the BH fix, **re-validate WH** (must stay green — see §7) since most code is shared, then mirror the fix to WH if applicable.
4. Note: `deepseek blitz` BH failures are **pre-existing on main** — ignore them (flag to the deepseek owner separately). The "Code Quality" CI check is a **broken/auth-failing CodeQL workflow** (fails on other PRs too, e.g. #46635; "cannot be rerun; workflow file may be broken") — not our code.

---

## 7. WH regression suite (must stay green — all pass at HEAD on WH n150)

```bash
rm -rf ~/.cache/tt-metal-cache/   # ARCH_NAME=wormhole_b0 on WH
pytest tests/ttnn/unit_tests/operations/eltwise/test_signbit.py -q            # incl -0.0 (#18346)
pytest "tests/ttnn/unit_tests/operations/pool/test_mpwi.py::test_mpwi_20_core_C_dims" -q   # uint16 indices
pytest tests/ttnn/unit_tests/operations/reduce/test_max.py -q                 # reduce (228)
pytest tests/ttnn/unit_tests/operations/reduce/test_argmax.py -q              # int indices
pytest tests/ttnn/unit_tests/base_functionality/test_reshape_transpose.py -q
pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -q -k "test_matmul_2d and not block_sharded"
# norm ops that were the WH-side issue (now fixed by scoping; must pass):
pytest "tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage" -q -k "ascending_values_repeated_rows"   # passes on WH
pytest tests/ttnn/unit_tests/operations/fused/test_group_norm.py -q -k "legacy"
```

**Negative control** (proves the path is exercised): force the flag off and maxpool indices corrupt:
- temporarily make `_configure_default_zero_flag_state_` / `_configure_unary_preserve_*` write `0`, run `test_mpwi_20_core_C_dims` → expect `indices_valid: False`, ~22k errors.

---

## 8. CI

Triggered via `gh workflow run <wf>.yaml --ref ncvetkovic/disable-src-zero-flag-restorable`:
- `tt-metal-l2-nightly.yaml` — **passed** (27271169676).
- `blackhole-post-commit.yaml` — fails on layernorm (§5) + pre-existing deepseek.
- `sanity-tests.yaml` — **passed** after the scoping fix (27287067569).

## 9. Memory / notes
Persistent notes for this task live in the agent memory dir under `disable-src-zero-flag-task.md`, `zero-flag-state-tracker-design.md`, `src-zero-flag-mechanism.md` (mechanism, design, status).
