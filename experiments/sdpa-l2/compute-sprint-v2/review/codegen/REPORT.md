# Independent A/B compiler-attribute screen

Private wrappers apply `#pragma GCC optimize("O2")` or an explicit O3 control
around the unchanged frozen v1 A/B compute source. The original source is the
baseline. Global build/link/fast-math flags, numerical recipes, readers/writers,
CB geometry and two input slots are unchanged. This is separate from the
early-identity scheduling candidate and is not combined with it.

Baseline A uses v1 `SDPA_BF16_EXP_BLOCK`; baseline B uses v1
`SDPA_BF16_BLOCK_STATE`, `SDPA_BF16_CORRECTION_REUSE` and
`SDPA_BF16_CORRECTION_FENCE`. Numeric definitions come directly from the
authoritative frontier recipe.

Both variants passed a distinct growing-max smoke with Q256/K1536, two repeated
Q jobs, raw eager equality and final raw-bit trace equality. This is not a
broad fullchip qualification and not multiple distinct Q inputs.

Resident screens fix Q256/K512/D128, sixteen Q repeats ×512 repeated K blocks,
twelve warmups and ten alternating original/O3/O2 measured rounds.

| Variant | Original v1 ms | O3 wrapper ms | O2 ms | O2 time change | v1 / O2 TFLOP/core |
|---|---:|---:|---:|---:|---:|
| A | 275.159238 | 275.167489 | 283.708605 | +3.107% | 1.997955 / 1.937748 |
| B | 337.255675 | 337.254510 | 344.651028 | +2.193% | 1.630086 / 1.595109 |

Both are exact but slower; reject O2 and stop expanded qualification. The O3
wrapper controls are neutral. Timing artifacts include source hashes, exact numerical and
CB configuration, and raw eager/final-trace equality gates. No production/v1
source files were changed. Both private wrappers were device-JIT compiled;
Python syntax checks passed.
