# Sparse output-subblock hypothesis

## Conclusion

The selected sparse geometry cannot legally use a larger output subblock without
also changing its role-specific core grid. All sparse calls have
`per_core_M=1`. Gate/up have `N=768 = 24` tiles on 24 cores, and down has
`N=2048 = 64` tiles on 64 cores, so all three also have `per_core_N=1`.
Consequently the current `(out_block, out_subblock)=(1x1, 1x1)` is the only
divisible choice on the selected grids.

The narrow hypothesis is: **using fewer N cores to make `per_core_N >= 2`, then
using a wider output block/subblock, will recover enough per-core efficiency to
outweigh the lower core count.** Keep expert weights BFP8, activations BF16,
LoFi, L1 intermediates, and the selected `in0_block_w` fixed while testing it.

## Legality

For the current sparse 1D-mcast kernel, require:

- `per_core_M=out_block_h=out_subblock_h=1`;
- `K_tiles % in0_block_w == 0`;
- `N_tiles % core_count == 0`, with
  `per_core_N=N_tiles/core_count`;
- `per_core_N % out_block_w == 0`;
- `out_block_w % out_subblock_w == 0`;
- `out_subblock_h * out_subblock_w <= 8` (the current TTNN destination-register
  search limit; this matrix deliberately stops at four);
- the work cores fill the rectangular grid exactly. This avoids the sparse
  factory's explicit non-rectangular receiver-grid rejection/hang guard.

`2x1` is not legal here: `per_core_M=1`, and the input has exactly one M tile.
The sparse factory does not implement a partial last M block, so setting
`per_core_M=2` merely to admit `2x1` risks an out-of-bounds read/write rather
than representing a valid candidate. `1x2`, `1x3`, and `1x4` are legal after
reducing the N-grid as below.

## Exact candidate matrix

Notation is `grid / per_core_MxN / out_block / out_subblock`. Gate and up must
be independently selectable even though their shapes match.

| ID | Changed role | Exact geometry | Purpose |
|---|---|---|---|
| `baseline` | none | gate `8x3 / 1x1 / 1x1 / 1x1`; up same; down `8x8 / 1x1 / 1x1 / 1x1` | Selected control |
| `gate_g12_b1_s1` | gate | `6x2 / 1x2 / 1x1 / 1x1`, `in0_block_w=16` | Isolate fewer cores |
| `gate_g12_b2_s1` | gate | `6x2 / 1x2 / 1x2 / 1x1`, `in0_block_w=16` | Isolate output block |
| `gate_g12_b2_s2` | gate | `6x2 / 1x2 / 1x2 / 1x2`, `in0_block_w=16` | Required area-2 subblock |
| `gate_g8_b3_s3` | gate | `4x2 / 1x3 / 1x3 / 1x3`, `in0_block_w=16` | Larger non-power-of-two area |
| `gate_g6_b4_s4` | gate | `3x2 / 1x4 / 1x4 / 1x4`, `in0_block_w=16` | Largest primary candidate |
| `up_g12_b1_s1` | up | `6x2 / 1x2 / 1x1 / 1x1`, `in0_block_w=16` | Isolate fewer cores |
| `up_g12_b2_s1` | up | `6x2 / 1x2 / 1x2 / 1x1`, `in0_block_w=16` | Isolate output block |
| `up_g12_b2_s2` | up | `6x2 / 1x2 / 1x2 / 1x2`, `in0_block_w=16` | Required area-2 subblock |
| `up_g8_b3_s3` | up | `4x2 / 1x3 / 1x3 / 1x3`, `in0_block_w=16` | Larger non-power-of-two area |
| `up_g6_b4_s4` | up | `3x2 / 1x4 / 1x4 / 1x4`, `in0_block_w=16` | Largest primary candidate |
| `down_g32_b1_s1` | down | `8x4 / 1x2 / 1x1 / 1x1`, `in0_block_w=12` | Isolate fewer cores |
| `down_g32_b2_s1` | down | `8x4 / 1x2 / 1x2 / 1x1`, `in0_block_w=12` | Isolate output block |
| `down_g32_b2_s2` | down | `8x4 / 1x2 / 1x2 / 1x2`, `in0_block_w=12` | Required area-2 subblock |
| `down_g16_b4_s4` | down | `8x2 / 1x4 / 1x4 / 1x4`, `in0_block_w=12` | Largest primary candidate |

After selecting each independent winner, test one cumulative
`gate winner + up winner + down winner` candidate. Do not infer a cumulative
win by summing isolated row times.

## Proposed patch

1. Replace the shared gate/up controls with independent
   `sparse_{gate,up,down}_{grid,in0_block_w,out_block_h,out_block_w,subblock_h,subblock_w}`
   fields in `OptimizationConfig`. Preserve the current values as defaults.
2. Extend `_sparse_program()` with the block/subblock arguments and the
   fail-fast legality checks above. Derive and record `per_core_N`; keep
   `per_core_M=1`.
3. Pass gate fields to the gate call, up fields to the up call, and down fields
   to both decode and grouped-prefill down calls.
4. Add the same role-specific flags to `optimized_decoder_perf.py`, and include
   every field in its candidate JSON.
5. Add static parametrized tests for every matrix row. Add an opt-in authentic
   candidate test named
   `test_optimized_real_weight_sparse_subblock_candidate`: use
   `_real_layer_state()` and `_real_hidden_at_layer()` for layers 1 and 4;
   cover traced b1 decode after a prefill-to-decode cache transition and b1
   prefill at sequences 33 and 128; emit PCC, per-role sparse row time,
   whole-layer time, exact config, and rejection/error text to JSON.

## Run contract

First run the static checks (no device):

```bash
pytest -q \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder_prefill_geometry.py \
  -k sparse_program
```

Run hardware candidates **one process at a time**, without watcher or profiler.
Invoke the following once for every ID in the matrix; the opt-in test should map
the ID to the exact geometry above:

```bash
timeout 1200 env \
  NORTH_MINI_SPARSE_CANDIDATE=gate_g12_b2_s2 \
  NORTH_MINI_SPARSE_LAYERS=1,4 \
  NORTH_MINI_SPARSE_MODES=decode,prefill \
  NORTH_MINI_SPARSE_PREFILL_SEQUENCES=33,128 \
  NORTH_MINI_SPARSE_SWEEP_OUTPUT_DIR=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/candidates/sparse_subblocks \
  pytest -q -s \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py::test_optimized_real_weight_sparse_subblock_candidate
```

Replace only `NORTH_MINI_SPARSE_CANDIDATE` for the other named rows. Stop after
any timeout/hang; capture `tt-triage`, recover the device, and do not continue
the matrix on a potentially wedged card.

Acceptance per role:

- PCC meets the existing authentic layer threshold for both layer kinds and
  all three workloads;
- traced decode whole-layer time improves, with no material prefill regression;
- the profiler row proves BFP8 weights, BF16 activation/output, LoFi,
  `in0_block_w=16/12`, the intended core count, and intended subblock;
- no non-finite, trace, routing, cache-transition, or repeated-run failure.

Promote only the cumulative winner, rerun the existing real-weight sparse
decode/non-aligned-prefill/repeated-trace tests, then run watcher separately:

```bash
pytest -q \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py \
  -k 'optimized_real_weight_moe_decode or optimized_non_aligned_sparse_prefill_exercises_active_experts or optimized_decode_determinism_and_repeated_trace'

TT_METAL_WATCHER=10 pytest -q \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py \
  -k 'optimized_real_weight_moe_decode or optimized_non_aligned_sparse_prefill_exercises_active_experts or optimized_decode_determinism_and_repeated_trace'
```

Finally profile the promoted default in a separate non-watcher run and retain
advice-enabled layer-1 and layer-4 b1 decode plus sequence-33/128 prefill
reports. The final report must show the three sparse roles separately and
reproduce the selected whole-layer timing; otherwise retain `1x1` with the
exact per-role failure/blocker.

## Resolution

The rereview matrix was implemented with an explicit all-role 1x1 control, so
every isolated candidate genuinely changed one role. One device session used
checkpoint layer-1 weights and the recorded checkpoint layer input, with three
warmups and twenty traced replays per candidate. Every row preserved PCC
`0.9993107116`.

| isolated role | candidate | traced decoder (ms) |
|---|---|---:|
| control | all-role 1x1 | 0.781748 |
| gate | G12 block/subblock 1/1 | 0.818704 |
| gate | G12 2/1 | 0.772062 |
| gate | G12 2/2 | **0.771717** |
| gate | G8 3/3 | 0.779254 |
| gate | G6 4/4 | 0.794333 |
| up | G12 1/1 | 0.819105 |
| up | G12 2/1 | **0.770802** |
| up | G12 2/2 | 0.775018 |
| up | G8 3/3 | 0.778011 |
| up | G6 4/4 | 0.793346 |
| down | G32 1/1 | 0.766479 |
| down | G32 2/1 | 0.739863 |
| down | G32 2/2 | 0.739675 |
| down | G16 4/4 | **0.731329** |

The cumulative matrix also used three warmups and twenty authentic traced
replays:

| cumulative candidate | traced decoder (ms) |
|---|---:|
| all-role 2/2 | 0.717877 |
| gate 2/2 + up 2/1 + down 4/4 | **0.704813** |
| gate 2/2 + up 2/2 + down 4/4 | 0.709911 |
| gate 2/1 + up 2/1 + down 4/4 | 0.711213 |

The winner is 9.84% faster than the authentic 1x1 control and 1.82% faster
than the best previously correct cumulative 2/2 policy. A final independent
reproduction measured `0.705102 ms` at the same PCC.

The decode winner regressed prefill: sequence 33 moved from
`4.099386/4.078837 ms` mean/min to `4.443987/4.252177 ms`, and sequence 128
from `13.598431/13.472163 ms` to `14.495682/14.213173 ms`. Therefore the
runtime now has phase-specific sparse programs. Decode selects gate G12 2/2,
up G12 2/1, and down G16 4/4; grouped prefill retains the prior 2/2 programs.
Final default prefill reproduced minima `4.077707 ms` (seq33) and
`13.491884 ms` (seq128).

Advice-enabled profiles under `tracy/review7_phase_specific/` prove the final
runtime rows: BFP8/LoFi, BF16 activation/output, `in0_block_w=16/16/12`,
decode subblocks 1x2/1x1/1x4, and prefill 1x2/1x2/1x2. The report calls the
decode gate/down and all prefill sparse subblocks good; the up 1x1 choice is
retained because its authentic isolated and cumulative measurements beat
legal 1x2, 1x3, and 1x4 alternatives.

Machine-readable evidence is in
`candidates/sparse_subblocks/{authentic_decode,authentic_cumulative_decode,final_default_decode}.json`
and the corresponding XML files.
