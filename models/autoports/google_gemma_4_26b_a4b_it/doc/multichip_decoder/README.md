# Gemma4 multichip decoder — Stage 04

The selected decoder uses all four Blackhole P300c ASICs in a 1x4 Linear
FABRIC_1D mesh. It combines EP4 active-expert prefill with indexed TP4 active8
decode, local paged KV caches and a replicated BF16 layer interface. The
advertised context remains262144 tokens. The baseline is the unchanged
`OptimizedDecoder` (SHA256 `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`,
baseline commit `a9259624f2`). No full-model or vLLM work is included.

Acceptance: all final gates pass and [independent stage review](stage_review_final.md)
returns **clean-pass**. Local stage checkpoint: `ab2892308b8960b8652546bbaf070ad8b120996d`; never pushed.
The runtime used by the final-policy artifacts is
`a12a913cf752b765338736dc71f71151ab972af1529a0098455755dd4f499255`.

## Selected plan

[Mesh plan](mesh_plan.md) retains the preimplementation topology/shape decisions;
[selected policy](selected_policy_audit.md) records every precision, geometry and
collective choice. The minimum worker grid is11x10. Each rank owns Q4 and
sliding KV2/D256 or full KV1/D512 (full KV heads duplicated across rank pairs).
Global expert intermediate704 is padded to768 for local192 decode shards;
EP prefill owns32 complete experts/rank. Shared intermediate2112 is padded to2176
for local544 shards. Norm/router/page/position state is replicated as specified
in the plan. The public BF16 `[1,1,S,2816]` interface directly stacks both kinds.
Padding stays internal; logical lengths need no tile or chunk alignment.

Decode expert GU/down weights are BFP4 with BFP8 input and LoFi math. Sliding
GU uses raw-checkpoint host packing at setup. Shared decode GU is BFP4; down is
BFP8 sliding/BFP4 full. Prefill retains its separately validated EP and BF16
shared weights. Sliding uses BF16 sharded RoPE; full retains FP32 interleaved
RoPE. Attention CCL is BF16 sliding/BFP8 full; grouped shared/routed CCL is
BFP8 sliding/BF16 full. Independent branch norms are preserved after reduction.

[Memory plan](memory_capacity_plan.json) and [source audit](memory_audit.md)
budget the entire30-layer stack per device:9,231,600,640 weight bytes,
8,556,380,160 cache bytes, shared RoPE/tied embedding allowances,2GiB reserve,
and4,429,185,024 bytes for maximum-length live prefill buffers. Conservative
peak27,451,657,216 bytes leaves4,548,342,784 bytes against32GB. See the
[context contract](../context_contract.json). RoPE must be shared by kind/layout,
embedding and LM head tied, and prior-layer outputs released. Capacity tests
reserve other resident payloads anonymously; they do not claim full-model
allocation order or fragmentation validation. B32 short contexts do not imply
32 simultaneous maximum-length requests.

## Paired target workload and warmed latency

Real checkpoint weights and recorded activations:4096 input tokens,
128 advancing traced decode positions, batch1, one request. Results compare
TP4 against TP1 with the same logical inputs, pages and positions. PCC bar0.995
is unchanged. Each phase/layer kind is reported separately. The timings below
are warmed **host-wall** medians (3 prefill samples and128 first replay samples),
not device durations. Duplicate replay stress is outside timed intervals.
Speedup is TP1/TP4; efficiency divides speedup by four ASICs.

| Kind | Phase | Minimum PCC | TP1 host us | TP4 host us | Speedup | Efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sliding | prefill | 0.9999220421 | 221342.741 | 92826.410 | 2.384x | 59.61% |
| Sliding | decode | 0.9976112404 | 825.488 | 650.445 | 1.269x | 31.73% |
| Full | prefill | 0.9999453677 | 186661.849 | 78905.756 | 2.366x | 59.14% |
| Full | decode | 0.9994477563 | 876.380 | 724.702 | 1.209x | 30.23% |

Evidence: [sliding](sliding_final_policy_raw_stress.json),
[full](full_final_policy_stress.json). Each checks all four replicas, local
K/V cache PCC and128 positions times8 duplicate replays, with no runtime host
fallback. [Mixed stack](stack_final_policy.json) passes layer0→5 direct handoff,
independent caches and a joint trace at33-token prefill and positions33/34.
This is an interface test, not a consecutive six-layer model accuracy claim.

## Boundary, capacity and Watcher checks

Batch32 uses heterogeneous logical lengths32..63, partial-page continuations,
disjoint page ownership, preservation of unrelated slots and refreshed page
tables. Long-prefix controls continue31 cached tokens with1025 new tokens.
Maximum tests use repeated recorded target activations, compare to TP1, and
reserve other resident stack payloads before execution.262143+1 checks the
last decode position;262144+0 checks aligned maximum prefill.

| Kind | Check | Result | Minimum output PCC | Artifact |
| --- | --- | --- | ---: | --- |
| sliding | batch32 | pass | 0.9974726964 | [sliding_final_batch32.json](sliding_final_batch32.json) |
| sliding | long_prefix | pass | 0.9995432858 | [sliding_final_long_prefix.json](sliding_final_long_prefix.json) |
| sliding | max_262143 | pass | 0.9995173073 | [sliding_final_max_262143.json](sliding_final_max_262143.json) |
| sliding | max_262144 | pass | 0.9999230201 | [sliding_final_max_262144.json](sliding_final_max_262144.json) |
| sliding | watcher | pass | 0.9976112404 | [sliding_final_watcher.json](sliding_final_watcher.json) |
| full | batch32 | pass | 0.9994065931 | [full_final_batch32.json](full_final_batch32.json) |
| full | long_prefix | pass | 0.9996984816 | [full_final_long_prefix.json](full_final_long_prefix.json) |
| full | max_262143 | pass | 0.9999080184 | [full_final_max_262143.json](full_final_max_262143.json) |
| full | max_262144 | pass | 0.9999435500 | [full_final_max_262144.json](full_final_max_262144.json) |
| full | watcher | pass | 0.9994477563 | [full_final_watcher.json](full_final_watcher.json) |

Watcher uses `TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1`, independently
of profiling. NOINLINE reduces the instrumented fabric firmware code size;
checks remain enabled. Exact invocations and exits are in each `.command.json`.
Runtime conversion/trace/source findings are in the
[fallback audit](runtime_fallback_audit.md). Replay/state sharing assumes serial
execution on one queue or separate stateful decoder instances.

## Device profiling and rooflines

Final native profile accounting uses the complete layer window, first firmware
start through last firmware end, including every operation and intra-layer gap.
For each replay it takes the maximum per-device span, then averages128 replays;
device clocks are not assumed synchronized. Useful FLOPs count logical tokens
and active8 experts. DRAM is an explicit operand-transfer estimate with sparse
reads, BFP exponent bytes, local cache read windows and real CCL outputs; it
excludes unmeasured internal rereads/scratch traffic. The denominator uses all
four participating ASICs' theoretical120-core LoFi peak at1.35GHz and512GB/s
DRAM/ASIC. Mixed fidelity, padding and inactive prefill-union work are described
in [roofline basis](roofline_basis_audit.md). No percentages are clamped.

| Kind | Prefill device us | Decode device us | Useful prefill FLOPs/peak | Estimated decode DRAM/peak |
| --- | ---: | ---: | ---: | ---: |
| sliding | 325830.533 | 705.420 | 0.10204% | 9.425% |
| full | 309683.707 | 766.584 | 0.14787% | 10.709% |

Per-kind `profile_final_sliding/` and `profile_final_full/` contain whole_layer.json,
window CSV, capture integrity, tt-perf-report human tables (`*_table.txt.gz`),
report CSV (`*_perf_report.csv.gz`), commands and SHA256 provenance. Native
captures remain local; compact evidence is committed. Generic tool op subtotals
omit gaps and are not roofline denominators. Final tables use explicit phase
end signposts and exact native row counts (final_signpost_filter_check.json). Profiler overhead can inflate
prefill elapsed device windows; ordinary warmed host timings are separately
reported above, never substituted into device fields.

## Optimization and anomaly evidence

[Final matched controls](final_policy_alternatives.json) use the same selected
precision policy. Sliding wider expert GU N2/K44 orK88 costs681.535/683.980us;
down N2 costs652.010us; DRAM shared MLP costs666.306us versus650.445us selected.
The adapted sharded-residual/Ring/fused-AGMM family passes correctness while
consuming the sharded stream through subsequent operations, but costs810.761us
sliding/877.898us full. Boundary gathers are outside decoder timing. Broader
candidate/geometry/topology comparisons and per-op advice are retained in
[optimization audit](optimization_advice_audit.md), [fused plan](fused_ccl_plan.md)
and candidate_summary_resumed.json. Dense all-expert decode is not selected.

- [Router-placement AutoFix](AUTOFIX_full_router1_bfp8.md): repeatable paired
  full-layer divergence at the old router core was removed by placement(10,9),
  outside fixed compute grids; reallocation-only controls refuted a simple
  address explanation. Exact native state mechanism remains unproven.
- [RoPE AutoFix](AUTOFIX_sharded_rope.md): the initial FP32 sharded candidate
  violated precision/configuration contracts. Homogeneous BF16 D256 passes
  component and layer cache checks. Adapted D512 is correct but slower and unused.
- [Packing AutoFix](AUTOFIX_final_policy_packing.md): device-converted sliding
  BFP4 GU failed PCC; raw host packing restores the complete candidate PCC vector.
- [Profiler AutoFix](AUTOFIX_profile_selected_abort.md):250k records overflowed
  a native allocation-size limit at shutdown.100k captures close cleanly with
  complete per-device/replay metadata. This changes measurement capacity only.

Earlier candidate files and reports remain historical evidence; their source
hashes and options must not be read as final-default acceptance. Failed gates
and reset/list/smoke recovery are retained in the [work log](work_log.md).

## Reproduction

```sh
HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder \
  --layer 0 --length 4096 --steps 128 --trace --check-cache --duplicate-replays 8 --output <report.json>
```

Use `--layer 5` for full attention. Contract scripts are `test_multichip_contracts`
and `test_multichip_stack` in the same tests package; exact final commands are
saved beside their result JSONs. Native capture commands are in each profile
folder's capture_command.json, with Watcher unset. Hardware jobs must remain
serial. Full-model generation quality and allocation order belong to the next
stage; this stage validates the decoder stack baseline only.

The sliding prefill CSV is committed as `profile_final_sliding/prefill_perf_report.csv.xz`
as a compact lossless copy; its byte-identical raw CSV and gzip copy remain local.
