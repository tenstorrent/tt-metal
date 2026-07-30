# Qwen3.6-27B fused decoder

Stage 02 delivers `tt/fused_decoder.py`, a distinct single-device runtime path
that inherits the completed functional decoder's setup, paged KV cache,
linear-attention state, RoPE, non-aligned sequence handling, and public tensor
contracts. No layout, sharding, dtype, cache allocation, or capability claim
changed, so `doc/context_contract.json` is unchanged.

## Retained fusion

Three fusions are retained. The common SwiGLU path changed from
`linear(gate) → linear(up) → silu(gate) → multiply → linear(down)` to
`linear(gate) → linear(up) → multiply(input-a=SILU) → linear(down)`.
TTNN's binary kernel applies SiLU while consuming the gate tensor, removing
one dispatch and its intermediate without changing BF16 operands. Full
attention packs Q+gate, K, and V weights into one setup-time RHS and slices one
runtime projection. Linear attention likewise packs QKV, Z, beta, and decay
into one RHS. Unequal output widths are preserved by exact slices before their
original consumers.

## Correctness

| Layer kind / mode | Batch / sequence | Functional PCC | Fused PCC |
|---|---:|---:|---:|
| full attention decode | 1 / 1 | 0.999960815 | 0.999960815 |
| full attention prefill | 1 / 33 | 0.999729385 | 0.999729385 |
| full attention prefill | 32 / 33 | — | 0.999725289 |
| linear attention decode | 1 / 1 | 0.999997956 | 0.999997956 |
| linear attention official weights decode | 1 / 1 | 0.999921858 | 0.999921858 |
| linear attention prefill | 1 / 65 | 0.999997842 | 0.999997842 |
| linear attention prefill | 32 / 5 | — | 0.999997894 |

All final acceptance runs set `throw_exception_on_fallback=true`; superseded
candidate logs are retained only as optimization history. The final fused page-routing test
performs sequence-65 prefill followed by position-65 decode using page table
`[[1,0]]`; PCC is 0.999905286 and physical key/value assertions distinguish
the two pages. Two sequential numerical trace replays validate changing inputs,
positions, and mutable state; an additional restore-and-replay check proves
bit-exact determinism from identical starting cache/state. Ten timed replays
provide stress coverage, and batch-32 rows remain distinct. Watcher-10 runs for
both layer kinds at batch 32 are clean.

## Performance

Device time is the sum of `tt-perf-report` rows between signposts. Decode rows
cover ten trace replays and are divided by ten.

| Mode / kind | Batch | Functional µs | Fused µs | Change |
|---|---:|---:|---:|---:|
| traced decode / full | 1 | 2381.374 | 2287.742 | -93.632 |
| traced decode / full | 32 | 2474.148 | 2386.759 | -87.389 |
| traced decode / linear | 1 | 3058.887 | 2921.279 | -137.608 |
| traced decode / linear | 32 | 21470.382 | 20884.852 | -585.530 |
| warmed prefill / full, seq33 | 1 | 2694.606 | 2584.460 | -110.146 |
| warmed prefill / full, seq33 | 32 | 72256.603 | 68521.011 | -3735.592 |
| warmed prefill / linear, seq5 | 1 | 10691.700 | 10577.316 | -114.384 |
| warmed prefill / linear, seq5 | 32 | 316313.463 | 312361.022 | -3952.441 |

The final packed path beats every correct traced-decode and warmed-prefill
baseline. It supersedes the earlier SiLU-only candidate, including its noisy
short linear-prefill b1 result.

An independent review required a trial of two-way MLP gate/up packing. The
AutoFix candidate was PCC-clean but regressed the decisive full-attention
batch-32 traced decode from 2386.759 to 2388.755 us/replay in `tt-perf-report`
(host median likewise regressed from 2.559 to 2.572 ms), so it was reverted.
The retained report is under `tracy/candidate_mlp_pack_full_b32/`.

## Artifacts

- `correctness/`: fused HF PCC, non-aligned prefill, and paged-cache logs.
- `trace_ab10/`: like-for-like ten-replay host A/B and PCC logs.
- `tracy/decode_*` and `tracy/prefill_*`: filtered `perf_report.csv`, human
  `perf_report.txt`, summary CSV/PNG, and console provenance. Duplicated raw
  profiler intermediates were removed after report generation; the retained
  filtered CSV is the exact input used for every reported sum.
- `watcher/final_{full,linear}_b32/`: watcher console plus generated watcher log.
- `work_log.md`: commands, topology audit, rejected candidates, and review/commit
  provenance.
