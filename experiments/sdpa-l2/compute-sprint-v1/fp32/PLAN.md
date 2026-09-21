# FP32 C/D compute-only sprint

Prepared 2026-09-18. Owned subtree: this directory. No canonical source modifications.

## Numerical contract

`bench.py` imports `flux2-frontier-v1/device_attention.py:recipe` as the numerical authority. D retains HiFi4 QK/PV, FP32 score/numerator/denominator state, BF16 maximum, full-FP32 L1 subtraction, existing unbiased exp grid/cubic refiner and macro rounding, and denominator phases 0+2. C retains HiFi4 QK / HiFi2 PV, original cheaper subtraction and corrected-exp implementation, and matched effective-weight denominator. No input preprocessing changes.

Fixed Q256/K512/D128, one compute core, FP32 destination halves, original CB formats/counts, original single-slot FP32 K/V inputs. Reader and writer are unmodified `hybrid-mixed-v1` files. Resident tests initialize/reuse input tiles; distinct-KV tests stream independent K/V and are not no-DM performance measurements. FLOPs/core must not be represented as measured chip throughput.

## Source pins at preparation

| Source | SHA256 |
|---|---|
| bfp4-lofi-v2/streaming/compute_streaming.hpp | f795e49f09b34e388fdb3149b40656905fcee0ae496abdf8f7d04e221afcaac9 |
| bfp4-lofi-v2/exp_refiner.hpp | bfd75a1f702827ceefebc7255a47621b96d0438eefa5f79af60dbba62f791a00 |
| bfp4-lofi-v2/fullchip/compute.cpp | 39925d62be5e29f6f4a6770bcab272880d4b5029f424e8e319b970174d754be5 |
| flux2-frontier-v1/device_attention.py | ec477bb574bd1503320802a400ccfde938eaf8e0cf054c6da39541fd2d779afe |

The old bfp4 resident harness unconditionally removes SDPA_DENOM_PHASES, so it is not used as the recipe authority. Per-run provenance hashes the included frozen hybrid headers/SFPU, canonical adapter/streaming source, private candidates and helper headers.

## Independent candidates

1. `identity4`: when the existing exact maximum comparison reports identity rescale, load/copy four numerator tiles in one destination acquisition, replacing two pairs. All original L1 additions retained; no decision criterion changed.
2. `state_unpack`: combine adjacent FP32 numerator state loads into the existing multi-face unpack MOP. Keep Blackhole zero-flag clearing for every tile and restore the one-tile MOP. Same values and destination locations.
3. `denom_pack`: pack four independent denominator tiles with a blocked pack, replacing four per-tile packs. No denominator accumulation order or fidelity change.
4. `state_both` and `all`: test combinations only after independent results pass. `copy` verifies that an unchanged private schedule matches the canonical source.

Prior reports rejected fixed-half pipelining, MATH-owned QK pack issue and lighter direct-DST reload initialization. Those proposals are not being repeated.

## Qualification protocol

Each candidate is compared against a canonical-header eager oracle with exact full BF16 output comparison. JSON records mismatched elements and maximum difference, and `.pt` saves the actual output. Any mismatch rejects/stops the candidate; relaxed L2 is not acceptance. Timed variants additionally compare final traced output exactly with eager output. Every run uses an explicit source hash manifest.

First JIT smoke: `bench.py --label smoke-v1 --q-repeats 1 --k-chunks 4 --iters 0`. Then distinct KV tests with normal, scaled_qk, outliers, common_q, common_k, common_v, uniform and constant_v; several seeds/longer KV for candidates that survive. Use forward/reverse timed order after warmup, comparing C and D separately. Device/profiler access must use root's exclusive global lock. Never launch independently while another agent owns the device.

At preparation only Python syntax validation has run. JIT compilation, numerical equality, and speedups are unverified. No performance result is claimed.
