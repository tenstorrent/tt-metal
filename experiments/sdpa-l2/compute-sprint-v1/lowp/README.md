# E/G compute scheduling sprint

Final frozen winner: `combined_fence/compute_streaming.hpp`. Resident time improves **2.080% for E** and **2.171% for G**, with 14/14 distinct-input, multi-Q stress cases bitwise identical to the canonical implementation. See [final results and remaining bottlenecks](RESULTS.md); early screening notes below are historical, not the final qualification status.

Numerical authority: `../../flux2-frontier-v1/device_attention.py`, `recipe()` and `prepare()`. E uses Q RNE7/BF16 and K/V RNE5/native B8; G uses the same Q and native-group RNE+saturating B4 K/V. Both retain LoFi QK/PV, BF16 DST, BF16 score/P and compensated high/low recurrent state, accurate maximum-change correction, original approximate score exp, the Q256 address-modifier repair, and HiFi2 recurrent/final broadcast rescaling. No algorithmic accuracy changes.

Fixed Q256/K512/D128; original CB sizes, two Q slots and two K/V slots. Resident mode reuses the existing no-recurring-DM reader/writer. Distinct mode uses the unchanged standard fullchip reader/writer and compares against the canonical adapter. No production or canonical source is edited.

## Controls and qualification

`benchmark.py` imports the canonical recipe/preparation directly. It checks decoded prepared values exactly against the established Q/B8/B4 oracles, checks original and prepared tensors remain unchanged, pins selected sources before/after, compares every candidate output bit against baseline, and always performs two actual trace replays per kernel, even with zero timing iterations. Distinct mode also checks exact equality with canonical `device_attention.attention()`. Timing uses alternating AB/BA order in a single process with the same prepared tensors. Reference metrics use original BF16 inputs and FP64 attention, not the prepared tensors.

The first distinct N4096/max-changing tests pass for both E and G (`e-block-distinct-v1.json`, `g-block-distinct-v1.json`). These initial tests execute one Q256 job and are not complete stress qualification. No speedup is implied by correctness.

Principal frozen pins:

| Selected file | SHA256 |
|---|---|
| FAST `compute_streaming.hpp` | `b471e527b61f55f2c9f30573ee4cdd82f5cbad7d0b02e1c654814835d3d0916b` |
| FAST `compute_common.hpp` | `2a1955a9c655ffce2bd4cab35a05830277ade33e8832049b4552e26b99b1eae5` |
| FAST `ckernel_sfpu_sdpa.h` | `54c51c56aa457134aed0181202c6fd3cad66df0a507659f364857c77dca9bb42` |

Exact current preparation, wrapper and selected project/API hashes are in every run JSON. This is not the complete compiler/firmware closure. Current dirty shared sources are preserved.

## Isolated candidates

- `block_state`: arrange the same numerator DST values as `(hi0, hi1, lo0, lo1, chunk0, chunk1, correction)`. Three two-tile copies replace six scalar copies; two width-two packs replace four scalar packs. The original 15-instruction SFPU program changes only DST addresses. Both BF16 rounding points and the live FP32 residual stay identical. Denominator still uses the original eight-tile layout/program; pack width restores to one before it. Fixed even-D128 research prototype, not generalized production dispatch.
- `setup_cache`: attempted to retain numerator/denominator macro and replay programs between compensation groups within a K iteration. **Rejected:** G Q2048/K8192 with changing maxima produced 147,367 numerical output mismatches. The baseline first matched the canonical adapter. Macro/replay lifetime assumptions are therefore insufficient. No timing is accepted for this candidate; `g-cache-distinct-v1.log` retains the assertion and clean device shutdown. The coordinator cleared the dirty marker after inspection, without a reset.
- `correction_reuse`: replay the original arithmetic but skip repeated correction loads for the odd-column SFPU vector. `SFPLOAD` offsets `d`/`d+2` cover even/odd columns of the same four rows; a COL-broadcast correction is identical for those vectors. Separate numerator-only/denominator-only controls passed; the combined final candidate passed the qualification below. See [instruction accounting](REUSE_NOTES.md).
- `correction_fence`: remove only the redundant balanced PACK_DONE triplet after correction publication has already ordered the needed preceding PV packs. The remaining correction-CB publication barrier, waits, L1 mode toggles, and other fences stay intact. See [ordering proof and branch coverage](correction_fence/ORDERING.md).
- `combined_fence`: combines plane-layout copies/packs, correction-vector reuse, and that single redundant-fence removal. This is the only final winner.

First-column-only denominator compensation was rejected during source review: FAST keeps a full 32-column partial-sum vector and final normalization sums every column. Discarding those updates would change the algorithm.

Prior work already includes paired state SFPU chains, load+add / round+store macro fusion, and paired denominator rows. This sprint does not count those existing changes as new gains. A/B PACK-width-cache work is owned by a separate agent.

## Initial resident screen

`e-block-resident-v1.json` measures seven interleaved AB/BA pairs, Q-repeat16/K-chunks512. Baseline median293.050647ms (1.875975TF/core), `block_state`291.940297ms (1.883110TF/core): 0.379% less time. Ranges do not overlap in this run, but this small result needs repeat measurement. Output hashes match; G performance and multi-Q qualification remain pending. No chip-level speedup is inferred.

## Run

Run only through the coordinating agent's machine-global exclusive wrapper, never a nested lock or automatic reset:

```sh
bash experiments/sdpa-l2/compute-sprint-v1/run_locked.sh \
  experiments/sdpa-l2/compute-sprint-v1/lowp/benchmark.py \
  --variant E --mode distinct --k-chunks 8 --max-changing --iters 0 --warmup 0 \
  --candidate experiments/sdpa-l2/compute-sprint-v1/lowp/block_state/compute_streaming.hpp \
  --output experiments/sdpa-l2/compute-sprint-v1/lowp/new-smoke.json
```

For the primary resident screen use `--mode resident --k-chunks 512 --q-repeats 16 --warmup 5 --iters 10` and remove `--max-changing`. Use a fresh output path; records are never overwritten. Resident metrics are repeated-input diagnostics, not distinct-input accuracy qualification. Compute timing excludes original upload and canonical preprocessing; those inputs are held constant.
