# Falcon3-10B-Base optimized multichip decoder

This stage optimizes the completed `tiiuae/Falcon3-10B-Base` tensor-parallel
decoder in place. The measured target is a fixed 1x4 Blackhole p300c mesh using
all four devices. Full-model assembly and vLLM are intentionally outside scope.

Status: implementation and hardware gates complete; the initial review and
first rereview findings are fixed, and final independent `$stage-review`
returned `clean-pass`. Stage-owned implementation and evidence are checkpointed
at `f6f3f0a11ef6bf6cfab2de9379a3a82b35c729cd`; nothing was pushed.

## Result

The final batch-32 default reduces 100-replay traced warmed decode latency from
0.576603 ms to 0.391425 ms, a 32.12% reduction (1.473x speedup), while preserving
the accepted real-weight PCC. The final path is TP=4 Ring/two-link execution,
not a replicated or single-chip fallback.

| Real layer 20, sequence 17 | Stage-entry rerun | Final default | Change |
|---|---:|---:|---:|
| Warmed prefill median | 3.003479 ms | 3.030677 ms | +0.91% |
| Traced warmed decode median | 0.576603 ms | 0.391425 ms | -32.12% |
| Prefill PCC vs single-chip | 0.999999505 | 0.999999505 | preserved |
| Decode PCC vs single-chip | 0.999999934 | 0.999999931 | preserved |
| K-cache PCC | 0.999995759 | 0.999995749 | preserved |
| V-cache PCC | 0.999998539 | 0.999998539 | preserved |

Each timing is the median of five samples. Each decode sample contains 100
trace replays. The changes are decode-only; nevertheless, the observed 0.91%
prefill regression is reported as measured. Against the strongest historical
input-stage numbers (2.771583 ms prefill, 0.576824 ms decode), final decode is
32.14% faster and prefill is 9.35% slower. Batch 1 changed from the historical
0.853517/0.370774 ms prefill/decode to 0.917850/0.318242 ms: decode is 14.17%
faster and prefill is 7.54% slower.

Exact final provenance:

- implementation SHA-256:
  `52d49bcf238ccd76ab2e763d9b5cd3c028ac12d40764eed84272aff5813f90ea`
- test SHA-256:
  `d0f4bed85fc6b33a922184f3b2a0a374258ba99634f4058b97e7f9e6896b8642`
- stage start HEAD: `3c69719127e668656ec77396bb042b024ff37659`
- completed input-stage checkpoint: `64e3199158e765e2558115bbfcc1ed2e4edcd68a`

The structured result is `results/perf_summary.json`; exact timing and PCC
sample arrays are in `results/final_default_*.json`.
The device artifacts were captured at implementation/test hashes
`19385bd701b70ad6072266fdcd68aeba2c9d5d9dbd4841915dc18f3ce83bd174` and
`32344521abf0a7da3afd5beac6c299b42aacca750ff9f7bc910d8c739c1ee871`;
the repository hooks then made formatting-only import/line-wrap changes to the
current hashes above. No executable expression or configuration changed.

## Selected implementation

The final default keeps the input stage's BFP4_B/LoFi projection weights,
BFP8_B KV cache, DRAM-sharded decode matmuls, and BF16 activation/residual
contract. It adds:

1. A mesh-wide `DecodeAllReduceResources` object with a sub-device manager,
   two persistent global semaphores, and an L1 intermediate buffer. Both
   row-parallel decode reductions use `ttnn.experimental.all_reduce_async`.
2. Dedicated `ttnn.experimental.rotary_embedding` operations for tile-aligned
   heterogeneous batches. Non-tile-aligned public batches retain the established
   internally padded explicit rotate-half path, so alignment is never imposed on
   callers.
3. Direct public DRAM materialization when the physical and logical row counts
   match. Padded smaller batches use the owned safe slice/staging path.
4. A stack-native residual API that can carry the layer output directly into the
   next layer without an inter-layer gather, reshard, or all-reduce. Sequential
   layers share the first layer's persistent CCL pool.

The stack residual contract is replicated values across the TP mesh, with each
device holding BF16 L1 width-sharded `[1,1,32,3072]` on 32 residual cores.
`decode_forward_to_residual`, `decode_forward_from_residual`, and
`materialize_decode_output` make this contract explicit. A real two-layer trace
uses two independently materialized decoder instances sharing one owner-managed
pool. It measures 0.710055 ms with the sharded boundary versus 0.778981 ms with
the old DRAM boundary, a 1.0971x speedup at PCC 1.0 (100 replays per sample,
five samples). There are zero inter-layer collectives; the two mathematically
required row-parallel reductions remain inside each layer.

## Operation-topology and candidate audit

`topology_audit.md` records the graph before knob tuning, including same-input
MLP matmuls, packed QKV, residual/layout conversions, collective boundaries,
fused CCL-matmul paths, and the proposed stack contract. The coherent candidate
families and final decisions are:

| Family/candidate | Prefill ms | Decode ms | Decision/evidence |
|---|---:|---:|---|
| Final: Ring, 2 links, BF16 persistent AR, dedicated RoPE | 3.030677 | **0.391425** | selected; current-hash exact rerun |
| Same final graph, no persistent AR | 2.873771 | 0.437456 | reject; persistent resource wins 10.56% |
| Same final graph, explicit RoPE | 2.897707 | 0.514621 | reject; dedicated op wins 23.97% |
| Linear, 2 links, BF16 | 3.044918 | 0.397344 | reject; Ring wins 1.53% |
| Ring, 1 link, BF16 | 3.054426 | 0.408873 | reject; two links win 4.31% |
| Ring, 2 links, BFP8 CCL | 2.944967 | 0.449578 | reject; BF16 wins 12.97% and PCC is already accepted |
| Corrected final-graph advisor, coherent 96-core residual | 2.892734 | 0.451892 | PCC and carried two-layer layout pass; reject |
| Earlier shard-advisor exact programs | 2.931702 | 0.641203 | PCC passed; superseded by final-graph closure |
| Advisor O projection only | 3.021012 | 0.580691 | reject |
| Adapted advisor O on legal 8x6 grid | 2.980914 | 0.577288 | first grid mismatch fixed and retried; still slower |
| Current graph packed gate/up, DRAM unpack | 3.319950 | 0.392700 | PCC passed; reject |
| Current graph packed gate/up, L1-sharded unpack | 3.073750 | 0.397105 | reject |
| BFP8/HiFi2 projections | 3.010753 | 0.493697 | reject |
| BF16/HiFi4, adapted down projection on 24 cores | 2.774082 | 0.689089 | reject |
| BFP4 with attention HiFi2 | 2.915361 | 0.400512 | reject |
| BFP4 with MLP HiFi2 | 3.023798 | 0.478435 | reject |

The initial BF16/HiFi4 down-projection attempt exceeded L1 at eight cores
(2,003,712 bytes required versus 1,572,864 available). It was not rejected at
the first API error: the weight/layout was adapted to 24 cores and the complete
layer was remeasured. Likewise, the 48-core advisor O candidate's first grid
mismatch was corrected to an exact legal 8x6 grid before timing.

The isolated persistent all-reduce probe measures 0.022101 ms versus 0.056970
ms for allocation-bearing default all-reduce, a 2.578x speedup at PCC
0.9999974. Both use five samples of 100 trace replays. The production decoder
candidate then improved from 0.576603 ms to 0.528400 ms before the other graph
rewrites were added.

### Matmul and sharding advice

Fresh final-graph `$shard-advise` output is retained in
`shard_advise/final_graph_corrected/`. It uses the production
`[1,1,batch,head_dim]` cosine/sine shape, includes dedicated RoPE, and returns
the stack-native residual: 25 operations, 22 choices, and two spills. Concat is
the only unfixable op because its input must be sharded. Every feasible choice,
including the rotary-adjacent block/height layouts, was applied together with
11-core block norms, 11x4 QKV, 11x9 O/gate/up/down, and a 96-core residual/add
chain with a matching persistent all-reduce buffer. The complete candidate
passed PCC at 0.451892 ms; its own carried two-layer boundary was 0.819200 ms
versus 0.888366 ms with DRAM, PCC 1.0 and zero inter-layer collectives. It is
15.45% slower than the selected default as a coherent family, so the selected
DRAM-sharded geometries remain 4/2/24/8 cores.

Final decode projection measurements from `tt-perf-report` are:

| Projection | Per-rank shape | Selected shard/program geometry | Device time/replay |
|---|---|---|---:|
| Packed QKV | 32x3072x1280 | grid 4x1, `[32,768] -> [32,320]`, K block 24 | ~10 us |
| O | 32x768x3072 | grid 2x1, `[32,384] -> [32,1536]`, K block 12 | ~8 us |
| Gate/up, each | 32x3072x6144 | grid 8x3, `[32,128] -> [32,256]`, K block 4 | ~41 us |
| Down | 32x6144x3072 | grid 8x1, `[32,768] -> [32,384]`, K block 24 | ~37 us |

The raw profiler records `CORE COUNT=80` launch metadata and 110 available
workers for these operations. The derived report's `Cores=12` is not an active
worker measurement: `tt-perf-report` hardcodes 12 for the FLOP model whenever it
recognizes a DRAM-sharded matmul. The program attributes and output shard specs
are authoritative for runtime geometry. The report's actionable advice was
tried through the coherent advisor 1D-multicast family; it lost at whole-layer
timing.

### Fused and lower-movement communication

The prior completed multichip stage already ran `$autofix` on the material
reduce-scatter/distributed-residual and fused matmul-CCL family. The coherent
lower-movement boundary was 1.00765x faster in isolation at PCC 0.999794, but
poisoned the Ethernet heartbeat after process exit even with explicit teardown.
Adapted one- and two-link fused matmul-reduce-scatter attempts hung; a standalone
matmul-reduce-scatter passed, isolating the integration failure. The fused
all-gather-matmul path had source-proven TP4 receiver accounting hardcoded for
four transfers. `doc/multichip_decoder/AUTOFIX.md`, `AUTOTRIAGE.md`, and
`results/graph_rewrite_*` contain the retries and failed AutoFix conclusion.

This pass therefore selected the clean persistent replicated-residual family,
then removed movement at the inter-layer boundary without immediately restoring
the old public DRAM contract. The two-layer measurement above is the deciding
whole-family evidence.

## Profiler result

Tracy capture and advice-enabled `tt-perf-report` were run separately from
watcher. The final raw CSV is
`tracy/final_default_post_advisor/reports/2026_07_18_22_20_27/ops_perf_results_2026_07_18_22_20_27.csv`;
filtered tables, console logs, CSV summaries, and plots are under
`tracy/final_default_post_advisor/derived/`.

| Decode profile metric, per replay | Entry | Final |
|---|---:|---:|
| Device operations | 68 | 45 |
| Summed device time | 518.003 us | 381.793 us |
| Op-to-op gaps | 82.751 us | 35.600 us |
| Matmuls | 137.09 us | 136.863 us |
| Material CCL | 76.97 us RS+AG | 27.991 us async all-reduce |
| Dedicated RoPE | primitive op cluster | 29.160 us |

The final dominant groups are matmul 35.85%, DRAM reshape views 17.32%, RoPE
7.64%, and async all-reduce 7.33%. In the same profile run, 431.963 us wall time
equals 381.793 us device time plus 35.600 us in-replay gaps plus 14.570 us
residual. One 90,574.710 us cross-iteration gap is excluded. The report's exact
31,457,280 B/rank projection model gives a 61.44 us ideal time at 512 GB/s and
82.39 GB/s achieved (16.09%). `results/roofline_accounting.json` also itemizes
known activation, KV, and Ring wire bytes. Profile timing (4.242417 ms prefill,
0.431963 ms decode) includes profiler overhead and is not substituted for the
warmed benchmark above.

## Correctness, context, and runtime gates

Falcon3-10B-Base is dense; real layer 20 covers its only meaningful decoder
layer kind. The final gates are:

- exact real-weight batch-32 prefill/decode/K/V PCC: pass;
- real paged non-aligned sequence 31 and sequence 1,025: pass;
- heterogeneous batch-2 positions with owned internal padding: pass;
- synthetic TP4 mapping/static runtime fallback audit: pass;
- strict `throw_exception_on_fallback=true` synthetic and real tests: 2 passed;
- maximum context: full 32,768-token batch-1 prefill plus decode at position
  32,767, finite replicated output, sampled K/V PCC 0.996420/0.994769: pass;
- watcher: real non-aligned paged traced decode passed with no watcher, kernel,
  fatal, assert, sanitizer, hang, or heartbeat diagnostic;
- final `tt-smi`: all four devices live, zero GDDR errors;
- trace stress: five samples x 100 replays for each performance candidate,
  persistent-all-reduce control, and two-layer contract: pass.

Ethernet watcher instrumentation was disabled because this platform's watcher
configuration buffer overflows when ETH instrumentation is enabled; Ring CCL
execution itself remained enabled. Profiler and watcher were never combined.

`doc/context_contract.json` now records the optimized runtime contract. KV
dtype/layout remains BFP8_B paged cache. The pool uses 786,432 bytes/device
(24,576 bytes on each of 32 residual cores) plus two 440-byte/device global
semaphores across all 110 worker cores. It is L1-only and does not reduce the
DRAM-backed 32,768-token context. Ownership, sharing, lifetime, cleanup, and the
196,608-byte/device stack residual are explicit. The largest advertised value
was physically executed rather than inferred.

## Reproduction and artifacts

Functional and benchmark hardware invocations use
`scripts/run_safe_pytest.sh`; Tracy uses `flock /tmp/tt-device.lock` around its
pytest wrapper. Thus only one TT process owns the mesh. Representative commands
and the full candidate chronology are in `work_log.md`.

Key artifacts:

- `results/perf_summary.json`: final structured decision summary;
- `results/final_default_perf.json`, `final_default_batch1_perf.json`, and
  `final_default_pcc.json`: exact default measurements;
- `results/two_layer_residual_contract.json`: stack-native boundary result;
- `results/max_context_batch1.json`: physical context validation;
- `results/final_family_*.json`, `precision_*.json`, `fidelity_*.json`,
  `advisor_*.json`, and `packed_mlp_*.json`: tried/rejected families;
- `tracy/*/reports`: raw profiler CSV provenance;
- `tracy/*/derived`: filtered reports, advice, summaries, and plots;
- `logs/final_broad_suite_post_advisor.xml`,
  `final_no_fallback_post_advisor.xml`, `final_watcher_post_advisor.{xml,log}`,
  and `final_tt_smi_post_advisor.json`: current-hash runtime gates;
- `shard_advise/final_graph_corrected/` and
  `final_graph_corrected_pipeline.log`: corrected production-shape dedicated-
  RoPE/stack graph compiler advice and provenance;
- `results/roofline_accounting.json`: same-run timing, byte roofline, collective
  traffic, and core-count semantics;
- `stage_review_initial.md`, `stage_review_second.md`, and
  `stage_review_final.md`: independent findings, remediation maps, and final
  clean-pass.

Limitations are explicit: the production configuration is fixed to the tested
1x4 p300c mesh; maximum context was physically tested at batch 1; prefill shows
the variance/regression reported above; ETH watcher instrumentation is not
available on this platform configuration. No decoder optimization from the
audit is deferred: candidates either became the default, lost with complete
before/after evidence, or were closed by the existing failed `$autofix` repair
loop. No full-model or vLLM work was started.
