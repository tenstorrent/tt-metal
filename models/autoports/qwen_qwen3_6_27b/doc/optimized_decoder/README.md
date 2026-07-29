# Qwen3.6-27B optimized decoder

Status: optimized-decoder complete; independent rereview `clean-pass`.

## Result

`tt/optimized_decoder.py` is an independent TTNN runtime, not a functional
fallback. It preserves the prefill/decode and cache contracts while using
packed QKV and gate/up projections, BFP8 attention/down weights, BFP4 gate/up,
BFP8 paged KV, LoFi compute, and advisor-seeded decode configurations.

| Path | Functional | Final optimized | Change |
|---|---:|---:|---:|
| full decode batch 1, traced | 2.4741 ms | 1.2180 ms | -50.8% |
| full decode batch 32, traced | 2.6707 ms | 1.5182 ms | -43.2% |
| linear decode batch 1, traced | 3.1645 ms | 2.2907 ms | -27.6% |
| linear decode batch 32, traced | 21.5003 ms | 20.6059 ms | -4.2% |
| full prefill batch 1, seq 33 | 3.730 ms | 2.847 ms | -23.7% |
| full prefill batch 32, seq 33 | not previously recorded | 49.689 ms | new evidence |
| linear prefill batch 1, seq 5 | 11.629 ms | 11.005 ms | -5.4% |
| linear prefill batch 32, seq 5 | not previously recorded | 294.716 ms | new evidence |

Headline numbers are uninstrumented host trace-replay or warmed-forward wall
times from saved `candidates/default*.log` runs. Tracy console timing is
instrumented and is not mixed into this table.

## Correctness

| Gate | Result |
|---|---|
| full prefill seq 33 PCC, batch 1/32 | 0.999991614 / 0.999991089 |
| full traced decode PCC, batch 1 step 1/2 | 0.999002617 / 0.999580396 |
| full traced decode PCC, batch 32 step 1/2 | 0.999584361 / 0.999814878 |
| linear prefill seq 5 PCC, batch 1/32 | 0.999997419 / 0.999997010 |
| linear traced decode PCC, batch 1 step 1/2 | 0.999986797 / 0.999987234 |
| linear traced decode PCC, batch 32 step 1/2 | 0.999967587 / 0.999990453 |
| official-weight full layer 3 decode | 0.998368503 |
| official-weight linear layer 0 decode | 0.999310013 |
| optimized paged prefill to decode | 0.999993917 |
| watcher, both kinds and batches | clean at interval 10 |

The initial real full-layer diagnostic exposed an inherited semantic bug:
`q_proj` channels were split into contiguous Q/gate halves, but HF views the
projection by head and chunks each `2*head_dim` pair. Correct per-head splitting
raises official-weight PCC from 0.6906 to 0.998369. The valid oracle rejects
BFP4 attention at 0.987799, so the shipped BFP8 policy is evidence-backed.

The multi-step trace test changes inputs and positions, advances KV or recurrent
state, rejects stale outputs, and detects batch-row aliasing. Non-aligned public
prefill lengths 33 and 5 pass. Runtime methods contain no Torch conversion or
host fallback.

## Topology audit

| Existing topology | Candidate | Action and evidence |
|---|---|---|
| separate Q/K/V same-input matmuls | packed QKV | kept; split is 1.593/2.015 ms versus 1.218/1.518 |
| separate gate/up same-input matmuls | packed gate/up | kept in the same split control |
| dense DRAM-interleaved decode | L1 activation chain plus DRAM-sharded QKV/O/down | kept; packed interleaved is 1.501/1.913 ms |
| one weight layout for both phases | persistent phase-specific copies | kept; interleaved prefill plus sharded decode |
| manual attention decomposition | paged/chunked TTNN SDPA | kept |
| BF16 weights/cache | group BFP8/BFP4 and BFP8 KV | kept at real/synthetic PCC bars |

## Shard advisor hard gate

The required capture ran in a separate bootstrapped shell with
`TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir`. Mandatory artifacts:

- `shard_advise/report.json`
- `shard_advise/final_ir.mlir`

Batch-1 corroboration is under `shard_advise_batch1/`. Both captures contain 17
ops and zero spills. Applied: width-sharded dense inputs; DRAM-sharded packed
QKV (`in0_block_w=4`, `per_core_N=5`), O (`12`, `2`), and down (`4`, `2`);
packed gate/up 110-core 1D multicast (`2`, subblock `1x5`, `per_core_N=10`).
Whole-residual norm sharding was rejected because composite token mixers require
interleaved boundaries; the measured partial chain wins.

## Precision and geometry matrix

All full-decode rows are saved under `candidates/` and use both batches.

| Candidate (dtype/fidelity) | Batch 1 / 32 | Decision |
|---|---:|---|
| final BFP8 attention/down, BFP4 gate/up, LoFi | 1.218 / 1.518 ms | kept; real full PCC 0.998369 |
| packed interleaved, same precision | 1.501 / 1.913 ms | reject |
| split projections, same precision | 1.593 / 2.015 ms | reject |
| HiFi2, same dtypes | 1.483 / 1.782 ms | reject |
| BFP4 attention/LoFi | 1.183 / 1.479 ms | reject; real PCC 0.987799 |
| BFP4 down/LoFi | 1.215 / 1.512 ms | reject; no meaningful win |
| BF16 KV | 1.220 / 1.528 ms | reject; slower and doubles cache |
| DRAM-sharded gate/up `in0_block_w=2` | 1.219 / 1.518 ms | tie; retain advisor 1D |
| down `in0_block_w=17`, `per_core_N=2` | 1.223 / 1.519 ms | reject; slower |

Larger precision-locked blocks have exact hardware evidence: gate/up widths 10
and 5 need 2,780,928 and 1,585,408 L1 bytes; QKV width 10 with `per_core_N=7`
needs 2,094,848; O width 24 needs 1,819,392. The limit is 1,572,864 bytes.
These adapted trials and the legal down trial cover larger block sizes, a
different QKV core count/per-core-N, and both kept precision groups.

## Profiler conclusions

The post-fix final profile verifies BFP8/LoFi/DRAM-sharded QKV, O, and down plus
BFP4/LoFi packed gate/up. Full decode reaches 48.2% modeled DRAM roofline
(247 GB/s); linear decode reaches 37.4% (191 GB/s). Full/linear prefill reach
27.5%/12.1%. Filtered CSVs, text, summaries, and PNGs are under
`tracy/final_*`. Instrumented profiler-console wall times are expected to differ
from uninstrumented headline timings.

## Context and scope

`doc/context_contract.json` retains 262,144-token decode and the existing
192,511-token single-pass prefill limit. BFP8 paged KV halves batch-32
advertised-context KV from 32 GiB to 16 GiB; no capability is reduced. This
stage contains no multichip, full-model, or vLLM work.
