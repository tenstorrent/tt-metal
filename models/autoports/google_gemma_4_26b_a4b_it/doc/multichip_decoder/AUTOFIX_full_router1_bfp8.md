# AutoFix: full-attention replica divergence after router relocation

## Starting evidence

- Initial report: [AUTODEBUG_full_router1_bfp8_initial.md](AUTODEBUG_full_router1_bfp8_initial.md).
- Fresh independent source review: [AUTODEBUG_full_replica_fresh.md](AUTODEBUG_full_replica_fresh.md).
- Runtime: `070613ddc32b5cc8a22fd92a64cb541de9a9f27152852f1caad0843d1d06903d`.
- Original runner: `06f0a0573dbd14e852be4a27c88b72faa610a51961b90cf33365025747898581`.
- Runner with authorized failure diagnostics: `c531a9a2f00db1eb9b2135f9b4397cbdf5a388db50776c17b6d4c670899b36ff`.

The ordinary paired TP1/TP4 layer-5 run, using real 4096-token prefill and
128 changing decode positions, fails strict replica equality with BF8 attention
CCL. BF16 attention CCL passes output PCC, cache PCC and replay checks. The new
failure differs from the earlier sliding-layer indexed-expert failure: only a
few finite final output values differ between mesh ranks within one replay.

Commands use `HF_HUB_OFFLINE=1 timeout 180 python_env/bin/python -m` and the module
recorded in each artifact's `command`, followed by:

```text
--layer 5 --length 4096 --steps 128 --trace --check-cache
--fused-tail --hybrid-experts --optimized-shared --shared-geometry 1
--grouped-moe-reduce --qkv-fidelity LoFi --output-fidelity LoFi
--attention-ccl-dtype bfloat8_b --output <artifact>.json
```

## Hypothesis experiments

| Artifact stem | Isolated control | Result |
|---|---|---|
| `full_router1_ccl_bf16_control` | Ordinary paired BF16 attention CCL | Pass; min output PCC 0.9994762094, cache PCC 0.9999715022 |
| `full_router1_bfp8_failure_detail` | Ordinary paired BF8, detailed failure reporting | Fail step 41 first replay; rank 1, six finite values |
| `full_bfp8_outer_boundaries` | Standalone TP4, broad boundary retention | Pass; retention and paired-process differences prevent acceptance |
| `full_bfp8_output_only` | Standalone TP4 diagnostic, original decoder class | Pass 128 positions |
| `full_bfp8_ordinary_tp4` | Ordinary runner with only `--tp 4` added | Pass 128; no TP1 oracle by design |
| `full_bfp8_ordinary_paired2` | Repeat ordinary paired negative control | Fail step 65 repeat; rank 3, eight finite values |
| `full_bfp8_paired_cleanup` | Drop TP1 device references and collect cycles before closing TP1 | Fail step 12 repeat; cleanup is not a sufficient fix |
| `full_bfp8_paired_attention_ag` | Retain actual attention all-gather return only | AG replicas exact; output fails step 33, columns 39,47,58,63 |
| `full_bfp8_paired_postnorm` | Retain actual post-attention normalization return | Boundary exact; output fails step 0, columns 33,39 |
| `full_bfp8_paired_moe_reduced` | Retain actual shared and routed grouped-reduction returns | Both exact; output fails step 18, six columns within 32:64 |
| `full_bfp8_paired_tail_shared` | Retain actual first shared-branch tail RMSNorm return | First bad boundary: step 12, seven columns within 32:64 |
| `full_bfp8_paired_tail_shared_io` | Retain reduced input, actual sharded RMSNorm input, weight, and return | Exact physical inputs/weight; RMSNorm return differs at step 0 |

The last control is decisive localization. Reduced and sharded inputs have
identical full physical `[1,1,32,2816]` tiles across all four ranks, including
14,490 nonfinite padding values; logical values are finite. The weight is also
identical. The RMSNorm return differs at logical columns 42,44,60,63, with a
maximum difference of 0.0625. Its physical output differs in 24 words, all within
columns 32:64. Snapshots and bitwise analysis are preserved in
`full_bfp8_paired_tail_shared_io.failure.pt` and `.failure.summary.json`.

This localizes corruption to the first tail RMSNorm computation/output storage,
after its input resharding. It does not establish a specific LLK race or state
register defect. The repeated column range corresponds to normalization worker
(1,0), also the relocated generalized-router gate core.

## Refuted or limited explanations

- The active `models.demos.gemma4.config.MeshConfig.allreduce` does not force-free
  inputs/scattered results in this no-padding path. The proposed early-free
  helper was a different, unused implementation.
- Both ordinary and standalone diagnostic runners delete each prefill output.
- Cleaning up TP1 ownership before close does not fix the failure.
- Attention AG and grouped MoE replica divergence are absent in reproduced
  failures; this is not evidence of a universal BF8 collective limitation.
- Standalone TP4 passes suggest a context/timing sensitivity, but do not prove
  that paired execution itself is the cause.
- Existing source review found no proven stale SFPU constant/predicate defect.

## Current candidate and verification

The diagnostic setup-only router move to core (10,9) passes both the original-output-only paired run and the same RMSNorm input/output boundary probe. This core is
outside the 11x8 normalization/expert-down/projection grid, 6x2 expert-GU grid,
11x4 shared-MLP grid, 4x1 router projection, and 8x8 decode SDPA grid. Some
movement operations may use it, so it is not described as globally unused.
Only router memory placement and four persistent buffers move; math, precision,
weights, inputs and native kernels remain unchanged.

Artifacts `full_bfp8_router109_output.json` and `full_bfp8_router109_tail_io.json`
both pass all 128 positions with minimum output PCC 0.9994477563 and cache PCC
0.9999715022. Next gates: restored core-(1,0) negative control, 128 positions with
eight duplicate replays, and ordinary production-path paired accuracy/cache
plus mixed-stack checks.
`full_bfp8_router1_negative.failure.json` restores core (1,0) after a clean reset
and fails immediately at step 0 first replay: rank 3 has four finite changed
output values, maximum 0.0078125. `full_bfp8_router109_stress1024.json` then passes 128 changing positions with
eight duplicate replays per position: **1024 exact comparisons per TP**, all
TP4 replicas finite/equal. Output/cache PCC minima remain 0.9994477563 and
0.9999715022. The diagnostic records the count and scalar buffer addresses.

The relocation also allocates/copies four buffers, while a core-(1,0) override
is a no-op. Therefore a same-core copy-only control is prepared using `ttnn.clone`
with an identical shard specification. It records scalar before/after addresses.
No specific gate-state defect is claimed solely from the placement A/B.
The unapplied `router_dedicated_core_candidate.patch` changes only the router
coordinate and its comment in the production constructor. The `ttnn.clone` same-core control failed during setup with a native factory
error: its reader kernel was given runtime arguments on node (0,0), where it
was not placed for this core-(1,0) shard. This is not a model correctness result.
The supported replacement copies core1→(10,9)→core1, retaining all original and
intermediate buffers until every final buffer is installed. It records old,
intermediate and final addresses, then releases setup temporaries before warmup.
`full_bfp8_router1_roundtrip_control.failure.json` fails at step 13 repeat,
with one finite output value differing by 0.0001220703125. Every final buffer
address changed: bias 1561856→1557760, indices 1570048→1553664, output
1568000→1549568, output_indices 1565952→1545472. Thus fresh buffer allocation and
copies alone are insufficient to fix the failure while the gate stays on core1.
Combined with the output-tile/core mapping and dedicated-core passes, this
supports the placement workaround without identifying a particular native state
register or race. No production placement change has been
integrated during this diagnosis.

Every failed multi-chip run closed normally and was followed by serialized
reset/list/mesh-smoke recovery. Recovery sets `full_bfp8_{reset,list,smoke}1`
through `11` (set 9 has suffix `_negative`) completed with exit 0. Final set 12 also completed with exit 0 after the roundtrip negative control;
all four devices were visible and the mesh smoke printed MESH_SMOKE_OK. No hangs or Watcher/profiler runs occurred
in this new investigation. Python formatting/compilation checks pass; no C++
build is needed for the diagnostic-only changes.

## Handoff status

Hardware was released to the coordinator after recovery set 12. The smallest
candidate patch is prepared but unapplied. Runtime0706 was not
changed during the experiment ladder; the coordinator is responsible for
integrating the dedicated-core placement and validating the ordinary constructor
path, sliding/full layers, mixed stack, batch and other adjacent gates.
Historical diagnostic scripts and hashes are recorded by
`full_replica_diagnostic_provenance.json`; the ordinary runner may receive
unrelated dormant geometry options after the preserved c531 snapshot.

These diagnostic timings are not the final performance evidence. Use the
ordinary integrated accuracy/performance and stack runs for stage acceptance.
