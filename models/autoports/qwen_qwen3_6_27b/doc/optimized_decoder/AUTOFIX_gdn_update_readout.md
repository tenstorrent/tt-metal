# Fuse exact recurrent-state update and Q readout

The final118 B32 profile supplied by the parent attributes 532.153 us (28.5%) to
recurrence, versus 59.887 us for untilize+convolution and 29.444 us for the later
reshape. Six matmuls already run near 89% nominal DRAM bandwidth. This makes
repeated SFPU state traversals a grounded next target. This source-only candidate
has no measured speedup or device correctness claim.

## Immutable artifacts

- `candidates/gdn_sfpu_update_readout.py`: standalone recurrence helper.
- `candidates/make_gdn_update_readout_candidate.py`: immutable integration generator.
- `candidates/update_readout_on_delivery118.py`: generated decoder using the new
  helper at the existing decay-predict dispatch; all defaults remain unchanged.
- `candidates/check_gdn_update_readout_cpu.py`, `gdn_update_readout_cpu.json`:
  source checks and arithmetic scheduling proof.
- `../../tests/run_recurrence_exact_comparison.py`: parent-owned device comparator.

No main/shared source or previous candidate was edited. The integrated helper
uses `_update_readout*` names. Standalone baseline `gdn_sfpu_decay_predict.py`
reader, writer and compute strings were checked equal to delivery118's three
`_decay107` kernel strings, establishing the actual incumbent being compared.

## Hypothesis and exact schedule

For each value-column tile n, the incumbent retains four decayed state tiles in
DST0..3, computes delta, then performs two loops over m=0..3:

1. `updated[m] = (delta * K[m]) + decayed[m]`, with separate rounded FP32 multiply
   and add, storing each updated tile to DST.
2. `term[m] = updated[m] * Q[m]`, folding terms in order as
   `acc = term[m] + acc`, then reducing rows with the existing SFPU SUM tree.

The new loop interleaves these two independent operations for each m. Updating
one state tile has no dependency on another state tile's update, and the readout
still encounters m=0,1,2,3 in the identical order. DST7's beta is dead after delta
is formed, so it can hold Q while DST5 holds K. DST4 continues to hold the
broadcast delta and DST6 continues to hold the ordered readout accumulator.

The fused SFPU primitive performs the exact old delta*K multiplication and
state addition. Updated FP32 values remain in LREG_DATA0..3. It then broadcasts
Q using only LREG_BCAST, LREG_TMP and the preserved column mask, multiplies the
live updated values by Q, and adds to the previous accumulator for m>0. No FMA,
reassociation, precision change, new reduction, or speculative compensation is
introduced. The existing row-reduction and BF16 persistence-rounding suffix is
unchanged: current readout consumes full FP32 updated state before persistence
rounding, as required by the incumbent.

The Q broadcast's final rotate-by-four stage has four dependent shifts, each
requiring a latency-filling instruction. Native `_broadcast_stage3_preserve_data_`
uses NOPs; `_broadcast_stage3_with_data_prefetch_` uses data loads. The new helper
uses the four required updated-state SFPSTOREs in those slots. Stores do not
modify the rotating scratch register or the retained DATA registers. This
preserves the native broadcast's issue spacing and final SFPADD/NOP pair.

The candidate removes the separate updated-state reload traversal and one SFPU
wrapper call per state tile (16/head), and schedules persistence stores into
otherwise-required broadcast latency slots. Some removed loads were already
latency-hidden in the incumbent's broadcast, so byte/count savings must not be
interpreted as proportional latency savings. No claim is made that this single
change can recover the approximately 200 us needed for the overall target.

## Contracts retained

Reader, writer, host descriptors, partitions, double-buffered CB sizes, direct
FP32 unpack modes, BF16 cache dtype, Q/K grouping, decay-predict fusion, native
exp and norm preparation are unchanged. Full eight-tile FP32 DST synchronization
remains required. The new path statically requires the incumbent's enabled
hoisted-delta and ordered-accumulator options. It is model-specific, not a new
general TTNN operation. The SFPU binary-call macro is variadic and forwards the
extra delta/Q destination indices; all five indices are in the existing 0..7
allocation and have distinct roles.

## Validation completed without hardware

Python syntax compilation passes for helper, generator, generated decoder and
comparison test. CPU checks confirm unchanged reader/writer and all host
functions, exact unchanged reduction/persistence suffix, absence of SFPMAD/FMA,
and update→store→readout multiplication ordering for each DATA register.
Ten host stress cases (BF16/FP32 state, scales 0,1e-10,.03,1,1e10) each run eight
updates and produce bit-identical FP32 updated states and ordered readout
accumulators under the interleaved schedule. These are source/algebra checks;
they do not prove device instruction scheduling or compilation.

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_gdn_update_readout_cpu.py > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/gdn_update_readout_cpu.json
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_gdn_update_readout_candidate.py --base delivery118.py --output update_readout_on_delivery118.py
```

## Required parent-owned device proof

```sh
python -m models.autoports.qwen_qwen3_6_27b.tests.run_recurrence_exact_comparison --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/update_readout_probe.json
```

Defaults cover B1/B3/B32, normal/zero-KV/zero-state inputs, three recurrent steps,
BF16 state, changed input addresses, persistent output reuse and stable state.
The test requires **zero mismatches in FP32 output bits and cache-state bits**
against the exact incumbent. It saves incremental JSON with both source hashes
and the first failing call's actual input state/data/output. Use
`--state-dtypes bf16 fp32` for both supported persistence modes and `--cores 1`
for repeated CB wraps. The parent owns all device execution.

Only after this strict comparator passes should the frozen decoder run the
unchanged 512-step gate, trace/cache checks, watcher and warmed profiling.
The new SFPU instruction schedule is the primary unresolved device risk.
