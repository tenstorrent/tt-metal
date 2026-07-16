# Candidate evidence index

Every XML file in this directory was emitted by pytest against real Hugging
Face layer-39 weights. Correctness XMLs prove the pytest PCC gate passed; the
refreshed final, packed-decode, and BFP8-cache artifacts also contain exact PCC
minima as JUnit properties. Older one-step geometry XMLs retain pass/fail and
timestamps rather than numeric PCC properties. Performance XMLs contain
`variant`, cache dtype, batch, logical sequence length, prefill latency, traced
decode latency, and repetition counts as JUnit properties. Candidate names
resolve to explicit policies in
`tests/test_optimized_decoder.py` so the retained artifacts are reproducible
after the default changed.

## Primary single-user adjudication

| Experiment | Prefill ms | Traced decode ms | Replays | Result |
| --- | ---: | ---: | ---: | --- |
| functional BF16 (`final_batch1_functional_bf16_500replay.xml`) | 5.058957 | 4.916922 | 500 | baseline |
| final BF16 (`final_batch1_optimized_bf16_500replay.xml`) | 3.179130 | 1.845752 | 500 | selected primary policy |
| final BFP8 (`final_batch1_optimized_bfp8_500replay.xml`) | 3.204054 | 1.846026 | 500 | supported; no batch-1 latency win |

This cumulative comparison holds the final all-BFP4/LoFi projection policy,
11x6 down grid, batch 1, sequence 18, cache layout, and 5/500 timing harness
fixed. Only the cache dtype changes between the two optimized rows.

## Final and long-run adjudications

| Experiment | Prefill ms | Traced decode ms | Replays | Result |
| --- | ---: | ---: | ---: | --- |
| final BF16 cache (`final_optimized_bf16_500replay.xml`) | 7.968504 | 2.113991 | 500 | primary-policy batch-32 control |
| final BFP8 cache (`final_optimized_bfp8_500replay.xml`) | 8.024863 | 2.078729 | 500 | supported faster batch-32 option |
| pre-grid control (`optimized_down_grid_adjudication_500replay.xml`) | 7.978771 | 2.120273 | 500 | superseded |
| 11x6 down grid (`final_down_grid11x6_down_grid_adjudication_500replay.xml`) | 7.981651 | 2.114013 | 500 | selected over 11x8 |
| split gate/up (`optimized_500replay_perf.xml`) | 7.959280 | 2.120094 | 500 | selected topology at that checkpoint |
| packed-decode gate/up, 100 cores, block 1 (`final_packed_decode_100_block1_500replay_perf.xml`) | 8.011141 | 2.122008 | 500 | rejected: fewer ops but slower |

`optimized_recurrent_8step_correctness.xml` and
`final_down_grid11x6_block4_8step_correctness.xml` independently cover eight
cache-consuming positions. Both keep output PCC at 0.999954--0.999960 and K/V
append PCC above 0.9928/0.9933. The refreshed final artifact records exact
minima of 0.9999541473 output, 0.9927964013 K, and 0.9933396936 V. The key
packed-decode rejection and BFP8 cache result likewise retain exact PCC
properties in `final_packed_decode_100_block1_correctness.xml` and
`final_bfp8_cache_correctness.xml`.

## Final-family geometry sweep

The 50-replay rows below were collected on the all-BFP4/LoFi advisor family.
All corresponding `*_correctness.xml` files pass the real-weight PCC gate.
The down block sweep uses the advisor 11x8 seed; the grid rows combine block 4
with the stated grid.

| Role / variant | Prefill ms | Decode ms | Decision |
| --- | ---: | ---: | --- |
| gate/up block 4 | 8.070413 | 2.264896 | reject |
| gate/up block 8 | 7.947870 | 2.451212 | reject |
| gate/up block 16 | 8.019620 | 2.376906 | reject |
| gate/up grid 11x9 | 7.967885 | 2.123379 | reject |
| gate/up grid 11x8 | 7.943177 | 2.130473 | reject |
| down block 2 | 7.994793 | 2.281115 | reject |
| down block 4, 11x8 control | 8.010782 | 2.119652 | block winner |
| down block 7 | 7.972216 | 2.134577 | reject |
| down block 8 | 7.982776 | 2.139644 | reject |
| down block 14 | 7.972988 | 2.164599 | reject |
| down block 16 | 7.994943 | 2.171021 | reject |
| down grid 11x6, block 4 | 7.936248 | 2.114164 | grid winner; confirmed by 500 replays |
| down grid 11x5, block 4 | 7.985081 | 2.168711 | reject |

## Packed projection evidence

The phase-specific packed-decode policies at 100 cores/block 2, 100
cores/block 1, and 106 cores all pass watcher correctness; their XMLs are
`final_packed_decode_*_correctness.xml`.  Their 50-replay decode times are
2.226319, 2.122340, and 2.253887 ms.  The block-1 policy was therefore the only
close candidate and lost the 500-replay split comparison above.  Full-phase
packed prefill remains impossible with the tested large-grid programs because
the matmul requests 2,263,296 bytes of circular buffers per affected core,
above Blackhole's 1,572,864-byte L1 limit; decode-only packing was the adapted
second trial, not a first-error rejection. The exact retained failure is
`packed_prefill_resource_failure.xml`.
