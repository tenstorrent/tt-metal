# AutoDebug Report

Inspection only: this pass did not edit model code and did not run TT
hardware.  Hardware numbers below are from saved artifacts already present
in the workspace.

## Headline

The catastrophic `dram_mlp_bfp4_24c` / `dram_mlp_bfp4_48c`
collapse was most likely caused by using DRAM `WIDTH_SHARDED` weight
copies in large-M prefill matmuls.  The DRAM-sharded matmul mode is a
decode-path weight-layout contract, but the failing candidates let that
layout leak into prefill.  Prefill then corrupted the attention/KV path,
so decode later consumed a bad cache and also collapsed.

The current working tree appears to contain the minimal fix: all prefill
dense-matmul weights are materialized with conventional interleaved DRAM, while
`decode_matmul_mode="dram_sharded"` creates separate `*_decode_weight`
copies for decode.  See
`models/autoports/tiiuae_falcon3_7b_base/tt/optimized_decoder.py:390-477`.
The current prefill/decode call sites use the intended copies:
prefill uses `qkv_weight`, `o_weight`, `gate_weight`, `up_weight`,
`down_weight`; decode uses `qkv_decode_weight`, `o_decode_weight`,
`gate_decode_weight`, `up_decode_weight`, `down_decode_weight` when
present.

## Artifact State

The requester-named
`models/autoports/tiiuae_falcon3_7b_base/doc/optimized_decoder/results/candidates/candidate_sweep.json`
no longer contains the near-zero PCC values described in the prompt; it
has been overwritten by a newer passing sweep.  That newer sweep reports:

- `advisor_mlp_bfp4`: prefill `0.99998603`, decode `0.99999903`
- `dram_mlp_bfp4_24c`: prefill `0.99998603`, decode `0.99999904`
- `dram_mlp_bfp4_48c`: prefill `0.99998603`, decode `0.99999909`

An older preserved artifact still demonstrates the same failure class:
`doc/optimized_decoder/results/autofix/full_candidate/candidate_sweep.json`
reports `dram_mlp_bfp4_24c` prefill/decode `0.00797/0.02321` and
`dram_mlp_bfp4_48c` `0.00797/0.02352`, while `advisor_mlp_bfp4` passes.
The prompt also records the original near-zero values
`0.0025/0.0423` and `0.0008/0.0474`.

## Evidence

The failing candidates are not MLP-only changes.  In the candidate
sweep, `dram_mlp_bfp4_*` sets `decode_matmul_mode="dram_sharded"` plus
the MLP target core count, and the BFP4 policy default sets
`prefill_grid_x=11`, `prefill_in0_block_w=8`
(`tests/test_optimized_decoder.py:657-679`).  If all weights are uploaded
through `_device_weight(..., memory_config=None)`, they become DRAM
`WIDTH_SHARDED` via `_dram_sharded_memory_config`
(`optimized_decoder.py:175-207`).

The current source now explicitly prevents that for prefill:
`prefill_weight_memory_config = ttnn.DRAM_MEMORY_CONFIG`, then only the
decode copies use the default DRAM-sharded upload path
(`optimized_decoder.py:397-477`).  The newest focused artifacts validate
that split:

- `doc/optimized_decoder/results/autofix/prefill_stage_localization.json`
  shows advisor and both DRAM BFP4 candidates with identical passing
  prefill attention, full-output, key-cache, and value-cache PCC.
- `doc/optimized_decoder/results/autofix/candidate_sweep.json` shows the
  three-candidate contrast passing end to end after the split.
- The current overwritten `results/candidates/candidate_sweep.json` also
  shows the full candidate sweep passing for these cases.

The older failure is not supported as raw BFP4 weight materialization
corruption.  `doc/optimized_decoder/results/autofix/weight_materialization.json`
round-trips advisor BFP4 and DRAM BFP4 MLP weights with the same
quantization-level PCC, about `0.993` for gate/up/down.

The older failure is also not supported as an isolated decode MLP bug.
`doc/optimized_decoder/results/autofix/matmul_localization.json` shows
DRAM BFP4 24c/48c decode MLP gate/up/gated/down PCC close to advisor, and
`doc/optimized_decoder/results/autofix/stage_localization.json` shows
decode from an HF-filled cache passing at about `0.999999` full-output
PCC.  That explains why full candidate decode was bad: decode was
downstream of bad prefill state, not independently broken.

## Program-Contract Detail

The most suspicious lowered contract is the generic 2D prefill matmul
with DRAM `WIDTH_SHARDED` operand B.  For Falcon3 qkv, `N = 5120`, so
`Ntiles = 160`.  With BFP4-policy prefill defaults,
`_prefill_matmul_program_config(... grid_x_limit=11)` gives
`per_core_N = ceil(160 / 11) = 15` (`optimized_decoder.py:243-269`).
The C++ DRAM-width-sharded reader path computes storage width by DRAM
banks, `ceil(Ntiles / num_dram_banks)`, which is `20` on 8-bank devices
(`matmul_multicore_reuse_mcast_2d_program_factory.cpp:357-364` and
`:1840-1846`).  Validation skips the `per_core_N == shard_width` check
for DRAM-backed operand B (`matmul_device_operation.cpp:1583-1597`).

That contract mismatch is a strong explanation for the old failure and
for why `dram_packed_bfp8` with `prefill_grid_x=8` / `in0_block_w=1`
passed: qkv `per_core_N` becomes `20`, matching the 8-bank storage width.
It is still a hypothesis about the matmul kernel contract, because the
current model code avoids this path for prefill instead of proving the
kernel wrong.

## Claim Review

Kept as headline: prefill/decode weight-layout contract.  It predicts the
older prefill PCC cliff, explains the corrupted KV/cache-dependent decode,
matches the current source change, and is validated by newer passing
prefill and candidate artifacts.

Demoted: raw BFP4 MLP materialization.  Round-trip materialization PCC is
the same for advisor and DRAM BFP4 weights, so it does not explain a
near-zero cliff.

Demoted: DRAM-sharded decode MLP program or 24c/48c target-core choice.
Decode MLP localization and HF-cache decode pass.  The target-core choice
only distinguishes decode MLP geometry, while the old PCC cliff was
already present in prefill.

Demoted: tile-size/alignment theory.  No static evidence found that BFP4
tile byte size alone explains this, and the passing/failing boundary is
better explained by the prefill operand-B layout and per-core-N contract.

## Minimal Hardware Experiments

1. Preserve a clean full sweep artifact on current source.  Re-run the
   exact full `test_decode_candidate_sweep` matrix and save it under a new
   timestamped directory so the failing/passing chronology is not
   overwritten again.  Expected: `dram_mlp_bfp4_24c` and
   `dram_mlp_bfp4_48c` stay at about `0.999986/0.999999` PCC.

2. Single-op qkv A/B.  Run one qkv prefill matmul with interleaved BFP8
   qkv weights and one with DRAM `WIDTH_SHARDED` qkv weights, same input,
   same `grid_x=11`, `in0_block_w=8`.  Expected: interleaved passes; the
   forced DRAM-sharded generic prefill path reproduces the cliff.

3. Grid contract A/B for the forced DRAM-sharded qkv path.  Compare
   `grid_x=11` with `grid_x=8`.  Expected: if the per-core-N/storage-width
   hypothesis is right, `grid_x=11` fails and `grid_x=8` improves or
   passes.

4. Cache causality check.  Run decode twice for a forced-bad prefill
   model: once with its own prefill KV cache and once with an HF-filled
   cache.  Expected: bad with self-prefill cache, good with HF cache.

5. Add a regression guard after the current fix is accepted: in DRAM
   decode mode, assert prefill attention/MLP matmul weights are interleaved DRAM
   while decode weights are DRAM `WIDTH_SHARDED`, then run the focused
   prefill-stage localization for `advisor_bfp4`, `dram_bfp4_24c`, and
   `dram_bfp4_48c`.
