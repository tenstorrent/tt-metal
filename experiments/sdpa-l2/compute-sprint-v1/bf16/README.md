# A/B compute-only sprint

Private copies only; canonical snapshots are not modified. Numerical recipes come directly from `flux2-frontier-v1/device_attention.py:recipe` (A and B), not historical resident CLI mode derivation.

## Frozen source pins

- A streaming: `single-core-resident-v1/main/.../compute_streaming.hpp`, SHA256 `e531a51efc6c47c4e4634b088f3d0256354c2096419379b1861b6f565a085ef6`.
- A common: SHA256 `e216a923296e053f86c3aaa6bce3a097f79e323197023bc1a773e6ceef521a6d`.
- B streaming: `bf16-denom-pair-v3/candidate/.../compute_streaming.hpp`, SHA256 `b471e527b61f55f2c9f30573ee4cdd82f5cbad7d0b02e1c654814835d3d0916b`.
- B common: SHA256 `2a1955a9c655ffce2bd4cab35a05830277ade33e8832049b4552e26b99b1eae5`.
- B correction addrmod reset wrapper: `bfp4-lofi-v2/fast_correction.hpp`, SHA256 `1e6905b04d9afdea8498b4bbdbe43c0ae988896b5051dfa46137df1389d9f632`.
- Exp-refiner wrapper (no alternative-degree define active): SHA256 `bfd75a1f702827ceefebc7255a47621b96d0438eefa5f79af60dbba62f791a00`.

The runner records all active common/SFPU snapshot hashes and exact numerical/implementation defines. B always has both compensation terms and the Q256 correction-address reset. Q/K/V are raw BF16 for both. Q256/K512/D128, input double buffers and CB counts are unchanged.

## First candidate: PACK MOP width cache

`SDPA_BF16_PACK_WIDTH_CACHE` suppresses a repeated `llk_pack_init` only when its requested width equals the current width. Width transitions use the original initialization. Exp constants, subtraction, matmul recurrence, compensation arithmetic and pack destinations are unchanged.

Static scope audit: the streaming header centralizes all explicit PACK MOP initialization in `configure_pack_width`. Used copy/broadcast and no-MOP matmul short-inits touch UNPACK/MATH, not PACK; format reconfiguration changes formats/strides rather than the PACK MOP. This first experiment is restricted to full-tile BF16 A/B inputs and CBs. Do not extrapolate it to mixed formats, tilized outputs or other operations without a new audit.

## Second candidate: remove redundant B L1-acc toggles

`SDPA_BF16_SKIP_REDUNDANT_L1_ACC` removes the caller's enable immediately before the compensated-state helper disables accumulation, and the helper's trailing enable immediately before the caller disables accumulation. Both actual disable operations and the explicit PACK-to-UNPACK visibility fence remain. No pack operation occurs in either eliminated enable/disable interval. The Blackhole LLK has no internal equality check: every toggle stalls CFG against PACK and writes the hardware register. This candidate is B-only. CLI `--candidate l1` tests it alone and `--candidate cache_l1` combines both changes.

## Measurement contract

`bench.py` instantiates canonical, private-copy-with-candidate-disabled, and enabled candidate kernels in one device session. All use identical unchanged readers/writers. Disabled-copy output must match canonical; any candidate bit mismatch is saved then fails the run. Interleaved forward/reverse timing orders, warmups and eager/trace checks are built in.

The repeated-resident reader is unchanged `bfp4-lofi-v2/resident/reader.cpp`. Distinct-KV qualification uses unchanged `hybrid-mixed-v1/reader_distinct.cpp`; this is correctness testing with DM, not the compute throughput measurement. Both keep the same CB geometry. `--distribution growing_max` progressively increases K scale to force repeated max changes; normal/outliers/scaled/common/uniform/constant-V inputs are also accepted. Sampled reference max-change counts are recorded to make correction coverage visible.

Examples (only under root-provided exclusive-device runner):

```sh
python experiments/sdpa-l2/compute-sprint-v1/bf16/bench.py --variant B --label B-smoke --q-repeats 1 --k-chunks 8 --iters 0
python experiments/sdpa-l2/compute-sprint-v1/bf16/bench.py --variant B --label B-maxchanges --q-repeats 1 --k-chunks 16 --distinct-kv --distribution growing_max --iters 0
python experiments/sdpa-l2/compute-sprint-v1/bf16/bench.py --variant A --label A-steady
```

See `REPORT.md` for final results and `STATUS.md` for historical screening. Width-cache
and redundant B L1-toggle candidates are numerically exact but have no measured
gain. Retained B combines plane grouping, correction reuse and fence removal.
All candidates remain experimental and qualified only for the tested geometry.
