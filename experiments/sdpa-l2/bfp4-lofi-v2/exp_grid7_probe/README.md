# BF16 grid7 exp bit probe

## Qualification and attention outcome

The parent reports an exact primitive pass over524288 values, including all
17405 supported BF16 encodings in[-1000,0], integer-grid ties, underflow and
the required cleared positive-output mantissa bit. The implementation agent
has not yet received the primitive JSON and does not mark that raw record as
locally inspected.

The following separate attention records **have** been inspected locally;
all contain a final `complete` record. N32768/H10/110cores, Q256/K512, Q7 and
K/V RNE5-BFP8, BF16 destination, native versus grid7 exp:

| Compensation | Input | Native L2 | Grid7 L2 | Attention TFLOP/s native → grid7 |
|---|---|---:|---:|---:|
| Full FAST | normal | 3.15106% | 3.18388% | 198.10 → 198.13 |
| Full FAST | constant V | 0.51258% | 0.57939% | 197.98 → 198.14 |
| Denominator only | normal | 3.31841% | 3.34106% | 216.05 → 216.14 |
| Denominator only | constant V | 1.42771% | 1.42934% | 216.13 → 216.23 |

Evidence: [full native](../grid7-32768-full-native-v1.jsonl),
[full grid7](../grid7-32768-full-grid7-v1.jsonl),
[denominator native](../grid7-32768-denom-native-v1.jsonl),
[denominator grid7](../grid7-32768-denom-grid7-v1.jsonl).
Useful attention FLOPs exclude preprocessing; each record also reports combined
time. Operator reference covers all heads/KV and the recorded sampled Q rows.

This experiment did **not** improve accuracy, while timing was effectively
unchanged. Matching the stored P bits to LoFi consumption is therefore
insufficient to remove the remaining error. BF16 reduction, recurrent state
and normalization still contribute; these results do not individually isolate
which one dominates. Grid7 is a qualified numerical probe, not a recommended
replacement on the strength of these attention results.

## Primitive mechanics

Isolated copy → pack-thread exp → pack-ReLU → BF16 output. No attention,
max subtraction, denominator, PV, recurrence or calibration. The existing
`exp_grid7.hpp` is included unchanged. Default64-step grid uses its wrapper;
`--native` uses the same wrapper with the grid7 define absent, preserving the
ordinary256-step native grid. Scale is FP32-rounded `1/sqrt(128)`.

The independent oracle rounds coefficients exactly as C++ FP32 expressions,
evaluates the BF16 input and FP32 coefficients in FP64 before one FP32 rounding,
rounds to signed INT16 nearest/ties-away, shifts17 (native15), applies sign and
pack ReLU, flushes exponent-zero grid results at BF16 SFPU store, and truncates
the remaining low16 bits. All positive grid7 outputs must have BF16 bit0 clear.
Input is ordinary finite BF16 or signed zero in [-1000,0]; input subnormals,
NaN/Inf and positive score deltas are rejected, not silently filtered in the
oracle. Generated pools exclude those unsupported values explicitly.

`all` includes every supported BF16 encoding, BF16 neighbors surrounding
integer-grid half thresholds, underflow values, both zeros and seeded normal
negative values. The default524288 elements fit the complete pool in both
native and grid7 modes. Individual distributions can be selected. The JSON
reports unique source encodings, exact transformed half-tie counts, negative
transformed values, pre-store subnormal counts and zero/positive output counts.

On mismatch the probe prints input bits, FP32-transformed bits, rounded integer,
expected output bits and actual output bits **before** the exact-bit assertion.
No mismatch is waived at an integer threshold: SFPMAD is documented as only
partially fused, so a discrepancy would require inspecting that arithmetic
contract rather than relabeling a failing sample as a pass.

## Parent-run commands

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/exp_grid7_probe.py --label grid7-probe-all-v1
python experiments/sdpa-l2/bfp4-lofi-v2/exp_grid7_probe.py --label grid7-probe-native-all-v1 --native
python experiments/sdpa-l2/bfp4-lofi-v2/exp_grid7_probe.py --label grid7-probe-underflow-v1 --distribution underflow
python experiments/sdpa-l2/bfp4-lofi-v2/exp_grid7_probe.py --label grid7-probe-zeros-v1 --distribution zeros
```

Defaults: one core, four input tiles per batch, BF16 DST/output, no timing.
Optional `--cores 110 --iters 5 --trace-repeats 10` measures this standalone
primitive only; it is not SDPA throughput. Each passing record includes input,
expected/output and source SHA256; active source changes and trace-output changes
are rejected. Use a fresh label per invocation.

Local validation: Python syntax, all12 selected source paths and a standard-
library scalar-domain audit. Torch is unavailable locally; no TTNN CPU runtime,
JIT compilation or device execution was performed by the implementation agent.

## Primary ISA references

- [INT16 nearest/ties-away and saturation](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatInt.md).
- [BF16 store truncation and denormal flush](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPSTORE.md).
- [SFPMAD partially fused arithmetic](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md).

This oracle is intentionally independent of `exp_grid7_models.py`, which is an
attention attribution model and explicitly not a full device/store oracle.
