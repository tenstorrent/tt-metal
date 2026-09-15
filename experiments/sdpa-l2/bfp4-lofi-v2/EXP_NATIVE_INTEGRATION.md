# One-pass native exp: private FP32 streaming integration

[exp_native.hpp](exp_native.hpp) is an isolated, opt-in API wrapper; existing
headers are unchanged. It requires `SDPA_LOFI_NATIVE_EXP` and the private
cheap-exp FP32 streaming path, not BF16 FAST or the accurate fused algorithms.

The current `calculate_sdpa_exp_grid_batch<32/128>` is already the native
LOADMACRO engine. Its behavior depends on the initialization, not its name.
The ordinary chunk-entry `exp_packthread_tile_init<true, scale_fp32,
InputClamping::None>()` loads the native eight-bit constants and records the
replay. The per-exp `init_sdpa_exp_grid` switches these to the custom ten-bit,
2^-96 grid. Skipping that switch and the refiner therefore leaves a complete
native exp, without a second pass or an exponent-restoration multiply.

## Minimal patch sites for the integrator

1. In the FP32 branch of each experimental compute wrapper, include
   `exp_native.hpp` after the selected frozen `compute_common.hpp` and before
   the private `streaming/compute_streaming.hpp`. Do not include another
   physical copy of the frozen SFPU header.
2. In `sub_exp_block_bcast_cols`, the four-tile branch currently has the
   `SDPA_DIAG_EXP_MODE == 2` case followed by an `else` containing
   init-grid + grid-batch + refiner. Insert before that `else`:

   ```cpp
   #elif defined(SDPA_LOFI_NATIVE_EXP)
       exp_native_packthread_tile<128>(0);
   ```

3. In the one-tile branch of the same function, make the analogous insertion:

   ```cpp
   #elif defined(SDPA_LOFI_NATIVE_EXP)
       exp_native_packthread_tile<iterations>(dst_index);
   ```

   Preserve the existing `++dst_index` outside the conditional.
4. Expose a mutually exclusive experimental host flag and add the define to
   its kernel cache key/defines. Leave the existing cubic path untouched when
   the flag is absent. Reject simultaneous polynomial-degree or accurate-exp
   selection rather than silently accepting a no-op flag.

No chunk-entry initialization change is needed. Both sites retain the existing
PACK/SFPU fence and packer ReLU. The four-tile path remains one batched SFPU
invocation; replacing it with four ordinary `exp_packthread_tile` calls would
unnecessarily repeat setup and pipeline drains.

## Lifetime and precision requirements

- Native constants L12/L13/L14, LOADMACRO configuration, and replay slots 0..7
  must remain intact across the QK/exp phase. The current cheap branch's QK,
  max reduction, and subtraction use the MATH thread; its PACK work does not
  overwrite those SFPU constants once the refiner and custom grid init are
  removed. Later correction/normalization may overwrite them: the next
  QK/exp phase must retain its existing native initialization.
- `calculate_sdpa_exp_grid_batch` resets ADDR_MOD_7 to destination increment
  two, replays the existing eight-instruction/four-vector cycle, and retains
  the existing six-instruction tail/drain. No new instruction scheduling,
  register allocation, or replay recording is introduced by this wrapper.
- Preserve the matched represented-P denominator, effective P truncation,
  FP32 P format, and FP32 accumulation. Do not replace online correction
  exponentials: a common multiplicative bias in those corrections compounds
  rather than cancelling through normalization.
- This is native exp, not the ten-bit direct grid. There is no 2^96 multiply.
  Very negative inputs require the existing packer ReLU. Post-subtraction
  scores must remain nonpositive, as in the current operator.

## Validation boundary

Source audit and host-stub C++ checks validate wrapper expansion, opt-in
guards, iteration restrictions, and FP32 restriction only. No device compile,
device execution, or performance claim is made here. First device validation
should compare a small FP32 native smoke against the cubic control and the
[CPU numerical expectations](DIRECT_EXP_GRID.md), then benchmark the resident
loop. Future PACK-thread SFPU changes require re-auditing the constant lifetime.
