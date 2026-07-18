# Llama 3.1 8B optimized multichip decoder

This stage optimizes the completed decoder in place on the real four-chip
Blackhole P300c `MeshShape(1,4)` ring.  Full-model and vLLM work are not part
of this stage.

Status: implementation and hardware gates complete; independent stage review
returned `clean-pass`.  Local commit provenance is recorded below.

## Result

The final default replaces each decode-time composite
reduce-scatter/all-gather with the dedicated width-sharded minimal all-reduce,
uses the fresh TP4-local shard-advisor 1-D projection family, and tunes QKV to
24 cores with `in0_block_w=32` and O projection to `in0_block_w=32`, both
with a 1x2 output subblock.  It keeps separate
persistent weight layouts for the two phases: DRAM-sharded prefill weights and
interleaved 1-D decode weights.  Both decode-only weights and the one reusable
minimal-all-reduce buffer are allocated by `prepare_decode()` after prefill.

| TP4 path, batch 1 / logical seq 18 | warmed prefill | traced warmed decode |
| --- | ---: | ---: |
| legacy composite median, 100 / 1000 | 0.773172 ms | 0.320012 ms |
| final default median, 100 / 1000, 6 samples | 0.794058 ms | 0.246686 ms |
| change | +2.70% (within observed prefill process spread) | **-22.91%** |

The six exact-final-default decode samples were 0.246689, 0.246689, 0.246682,
0.246753, 0.246635, and 0.246679 ms; corresponding prefill samples were
0.841712, 0.828560, 0.726828, 0.748395, 0.790864, and 0.797251 ms.  The final
single-chip control medians were 1.242878 ms prefill and 0.581132 ms decode,
so TP4 decode speedup is 2.356x with 58.89% TP4 efficiency.  The fresh
pre-pass measurement was 0.746841 / 0.320079 ms; repeated process samples are
used above because prefill E2E varied materially despite the decode-only
policy change.  Device-profiler prefill time is unchanged at about 1.266 ms
for three executions before and after.

Final performance output PCC against the invariant optimized single-chip
control is `0.9999998070869076`.

## Final default policy

- TP=4, 1x4 ring, two fabric links.
- Local ownership: Q8, KV2, MLP intermediate 3584.
- Packed QKV; separate gate/up; row-parallel O/down.
- BFP4 projection weights with LoFi math; BF16 activations, norms, residual,
  and decode collective; caller-owned BFP8 or BF16 KV cache.
- QKV 1-D matmul: 24 cores on 8x3, `in0_block_w=32`, `per_core_N=2`, 1x2
  subblock.  O uses 64 active cores on the 11x6 advisor grid,
  `in0_block_w=32`, `per_core_N=2`, and a 1x2 subblock.  Gate/up/down retain
  the corresponding 56/56/64-core advisor layouts with `in0_block_w=8`.
- One mesh/stack-shared persistent decode minimal-all-reduce buffer, reused
  sequentially by attention, MLP, and all 32 decoder layers.  Default worker
  placement, two links, and the normal NOC policy won.

## Inter-layer residual contract

Decode returns replicated-across-ranks BF16 data in 16-core width-sharded L1,
logical shape `[1, batch, 1, 4096]` with internally tiled rows.  The next
decoder consumes the same contract.  There is no gather, reduce-scatter,
all-gather, or material reshard between decoder layers.  The two collectives
inside a layer consume and return this width-sharded residual directly.

The tested 64-core carried-residual family adapted the collective buffer grid,
norm layouts, gate/up input, down input, and layer output as one coherent
family.  It passed PCC but was slower (0.251809 ms BF16 and 0.272064 ms BFP8)
than the retained 16-core contract; it was not rejected via an immediate
restore to the old layout.

## Correctness, stress, and runtime audit

The final default passes with `throw_exception_on_fallback=true`; the stacked
hardware check uses two distinct decoder instances and verifies that both
acquire the same mesh-owned CCL buffer pool:

- TP4 prefill PCC `0.9999993670541333` at non-aligned logical seq 7;
- TP4 decode PCC `0.9999886769364014`;
- stacked decode boundary PCC `0.9999549274032734`;
- deterministic contiguous and paged caches, adversarial pages, and positions
  63, 64, and 65; all public-output PCC values exceed 0.99995;
- batch-32 paged watcher stress with ten deterministic trace replays, each PCC
  1.0;
- source audit rejects host tensor conversion or functional/single-chip
  fallback in either measured forward path.

The full watcher image exceeds the Blackhole ACTIVE_ETH config buffer before
mesh open (27,920 versus 25,600 bytes).  The prescribed retry with
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` passed; Tensix, dispatch, and NoC watcher
instrumentation remained enabled.  This is a watcher tooling limitation, not
a model failure.

## Profiler and roofline accounting

The final advice-backed `tt-perf-report` is filtered to ten warmed trace
replays only; eager warm-up and trace construction are outside the signposts:

| Class | Device time | Share |
| --- | ---: | ---: |
| five 1-D matmuls | 115.965 us/replay | 48.98% |
| two minimal all-reduces | 29.677 us/replay | 12.53% |
| norms | 14.608 us/replay | 6.17% |
| SDPA decode | 12.758 us/replay | 5.39% |

Total summed merged-device time is 236.764 us/replay.  The machine-retained
raw signpost timestamps bracket the ten profiler-enabled replays at 291.265
us/replay, leaving 54.501 us.  A targeted same-count run without profiler
instrumentation is 248.833 us, assigning 42.432 us of the gap to profiler and
signpost instrumentation; increasing to 100 unprofiled replays reaches
246.850 us by amortizing the final synchronization.  The remaining
approximately 10.086 us is nonblocking TTNN trace enqueue and cross-run
accounting overhead,
not a decoder host fallback.  Each device must read about
30,670,848 B of BFP4 projection weights plus roughly 10 KiB of KV data at
position 18.  At 512 GB/s/device, the four-device aggregate roofline is about
59.9 us/token.  The replay-only profile therefore reaches 25.30% of the
device-time roofline and 20.57% by profiler-enabled signpost E2E.  The measured rows confirm BF16 activation,
BFP4 weight, LoFi math, L1-interleaved matmul input, and width-sharded BF16
minimal-all-reduce input.

Prefill remains the composite DRAM path: matmul 40.28%, norm 23.62%,
reduce-scatter 11.05%, and all-gather 6.56%.  A non-aligned, internally padded
16-core L1 sharded-norm retry was correct but increased adjacent E2E prefill
from 0.711551 to 0.991491 ms, so it was rejected with direct evidence.

## Context contract

`../context_contract.json` still preserves 131072 tokens.  Phase-specific
decode weights add 981,467,136 B/device, raising the conservative BF16 plan to
14,847,836,160 B/device, below the physically probed 34,178,731,008 B/device
allocator capacity.  The single stack-shared collective buffer is
1,048,576 B/device, or 65,536 B on each of 16 workers versus 1,572,864 B/core;
it no longer scales with the 32-layer count.  KV dtype/layout, two local KV heads, 64-token pages, and
logical sequence semantics are unchanged.  Public inputs need not be tile- or
page-aligned.

## Candidate and artifact index

- `candidate_results.csv`: all material before/after candidate measurements.
- `topology_audit.md`: topology-first audit, family comparison, and final
  disposition.
- `geometry_sweep.md`: final-topology packed projection and independent
  role-by-role program/memory geometry matrix.
- `work_log.md`: exact commands, failures, adaptations, and gate results.
- `shard_advise/`: capture script, report, final IR, and provenance.
- `tracy/analysis/`: pre-review block16 per-op CSVs, summaries, and tables.
- `tracy_replay_final/`: authoritative exact-final replay-only profile and
  reconciled same-run accounting.  `tracy_replay/` is the preceding QKV32/O8
  review-repair profile, while `tracy/` is retained as pre-review block16
  topology provenance.
- `logs/`: JUnit/PCC/performance/watcher provenance for final gates and all
  review-driven candidates.
- `tracy_replay_final/reports/2026_07_18_14_22_36/`: losslessly compressed
  authoritative final raw ops CSV provenance.  The capture under
  `tracy/reports/2026_07_18_13_09_56/` is pre-review topology evidence.

## Stage review and commit

The final fresh `$stage-review` verdict is `clean-pass`; see
`stage_review.md` for the repair history and scope.  The stage-owned local
commit SHA is appended here immediately after the checkpoint is created.
Nothing is pushed.
