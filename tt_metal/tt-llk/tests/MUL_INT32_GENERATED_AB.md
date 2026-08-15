# MulInt32 handwritten/generated A/B

## Result

**NO-GO (performance).** The compiler-owned implementation is correct on WH
and BH CRAQ and on physical BH, but its BH math-isolate device time is 1.9816x
the handwritten SFPLOADMACRO implementation.

| BH measurement (device cycles/tile) | handwritten | generated | delta |
|---|---:|---:|---:|
| Math isolate, samples 1/2/3 | 283.9296875 / 283.9296875 / 283.9296875 | 562.625 / 562.625 / 562.625 | +278.6953125 (+98.16%) |
| L1-to-L1, samples 1/2/3 | 412.1796875 / 412.1796875 / 412.1796875 | 695.5625 / 695.5625 / 695.5625 | +283.3828125 (+68.75%) |
| Math ELF text | 2793 bytes | 2585 bytes | -208 bytes (-7.45%) |

The three samples are fresh independent profiler invocations. These are scoped
device-profiler cycles from `TILE_LOOP`, not pytest wall time.

The negative result is structurally expected, not a register-allocation loss:
the handwritten BH body drives four preconfigured SFPLOADMACRO sequences and
retires one visible `sfpmul24`/`sfpiadd` pair per unrolled row. The generated
body emits five explicit `sfpmul24`, four shifts, and four integer adds for the
same row. Closing this target therefore requires general compiler formation of
load-macro templates (including delayed multi-pipeline operations and the
scheduled store), not Welford- or MulInt-specific peepholes.

## Correctness and formulation

- WH CRAQ: PASS.
- BH CRAQ: PASS, using the reset-PC-capable CRAQ binary below.
- Physical BH handwritten: PASS.
- Physical BH generated: PASS.

BH uses typed `sfpi::fractional_mul` low/high results and the exact radix-23
identity used by the handwritten fallback. WH uses radix 2^10. Every chunk
product and accumulated coefficient remains below 2^23, so FP32 arithmetic is
integer-exact; adding 2^23 and extracting the mantissa performs an exact,
non-saturating conversion. Loads and stores are U32 so signed input bit
patterns retain the same modulo-2^32 product.

One useful harness finding: raw `TT_SFPLOAD` tile offsets use 64 units, while
typed `sfpi::dst_reg` tile offsets use 32 row indices. An initial 64-unit typed
offset compiled but selected the wrong operand tile; CRAQ caught it. The final
selector uses 32.

## Corpus discrepancy

The corpus manifest names common `ckernel_sfpu_mul_int.h` headers (the UInt16
operation), but its live correctness route dispatches `SfpuMulInt32` to metal
`ckernel_sfpu_mul_int32.h`. This probe follows the executable correctness and
perf route. The manifest should split or rename the UInt16 and Int32 targets.

## Provenance and archive

- TT-Metal base: `f46e98b5e3fd63d3850a1475f7bf6421b42d9417`
- SFPI source: `febf2e20dde3b771f62421586b6f0f3a9676108b`
- compiler SHA256: `f584dce043a26fef1ed8d2d11ef9fb6b4903a61c42b2e9af0c9b0701cca5f360`
- CRAQ BH library SHA256: `6c03401dee15ca74e7dab0e6c13cf60928314197a12cb17b1e858c68262d138a`
- external archive: `/localdev/nkapre/mul-int-validation-device`
- handwritten correctness ELF SHA256: `85ff0651eb7ab412238ebda5e42320d0732b0b79675bb7ab52a5724087ca59cc`
- generated correctness ELF SHA256: `d0d33ca4e8a672b8836297255e519e614faf4dacbcf0145d66bfe5f93f9742c3`
- generated math-isolate ELF SHA256: `fa3ae69d19be64de90b6781396694735f116e911b60265a7eaa2a4d7c1ae5962`

The archive contains the six raw/post CSV pairs, correctness and profiler
logs, ELFs, and disassemblies. No production LLK file is modified.
