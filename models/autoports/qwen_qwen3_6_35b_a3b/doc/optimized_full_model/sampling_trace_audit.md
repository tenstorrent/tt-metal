# Sampling And Trace Audit

## Selected Path

| Area | Setting |
| --- | --- |
| LM head | BF8 flat 4-way vocab shard, `ShardTensorToMesh(dim=3)` |
| Logits contract | sampler-ready sharded logits, no full-vocab host materialization on token-out |
| Sampler | common `SamplingGenerator` top-1 split sampling, `max_top_k=32` |
| Greedy output | sampler writes into persistent tile-width TT output buffer through `tt_out_tok` |
| Feedback | captured device-side slice from sampler output slot 0 to 1-wide decode input |
| Position/RoPE | persistent TT current-position tensor, in-trace `plus_one`, RoPE lookup on device |
| Page table | cache-owned TT page table reused across replay; changed-only caller overrides are honored |

The captured decode body includes token embedding, all 40 multichip decoder
layers, final RMSNorm, vocab-sharded LM head, common sampler, token feedback,
and position advance. The optimized measurement enqueues trace replay
nonblocking for every steady-state token, then performs one final device sync
and optional one-token readback only after the measured loop.

## Host Boundaries

`measure_token_out_no_readback()` records the measured serving-style boundary
counters. For the warmed prompt-128/gen-128 run:

| Counter | Value |
| --- | ---: |
| trace replays | 127 |
| nonblocking `execute_trace` replays | true |
| steady-state token refreshes | 0 |
| steady-state position refreshes | 0 |
| steady-state RoPE refreshes | 0 |
| steady-state page-table refreshes | 0 |
| steady-state synchronizations | 0 |
| steady-state token readbacks | 0 |
| terminal validation synchronizations | 1 |
| terminal validation token readbacks | 1 |
| full-logits readbacks | 0 |

The comparison row in the same artifact uses the completed full-model traced
token-out path with per-token sync/readback: `127` steady-state syncs and
`127` token readbacks. Both rows produce the same final token.

## Greedy Sampler Choice

`scripts/profile_terminal_sampler_choices.py` benchmarks the valid greedy
choices on the target mesh.

| Candidate | Result |
| --- | --- |
| common top-k1 composite gather | selected; `10.901 ms` mean over 5 iterations, matches host argmax |
| force-argmax async full-vocab gather | rejected; the local 2x2 p300c fabric cannot route the requested full-mesh all-gather from `M0,D1` to `M0,D2` for this flat vocab layout |

The terminal-path Tracy report inherited from the completed full-model stage
measures final norm plus LM head at `0.510 ms`, sampler at `10.938 ms`, and
the combined terminal path at `11.464 ms`. `TopKDeviceOperation` is the
dominant sampler op, but it is about `19%` of optimized token-out decode
(`57.359 ms/token`) and does not dominate the full decode step.

## Capability

This stage keeps the canonical split-sampling contract from the completed
full-model path. The public generator still supports explicit cache,
page-table, position, prompt-length, and batch state; mixed prompts, fixed
slots, changed page tables, inactive rows, and non-aligned prompt lengths are
covered by synthetic hardware and watcher tests.
