# Current-arm AutoFix result

## Attention precision: fixed

QKV and O weight dtype/fidelity are now independent; non-matmul attention
fidelity stays fixed. A real-checkpoint sequence-33 prefill feeds a warmed
cache-consuming traced decode at batches 1 and 32 on the final topology.

All 16 BFP4 candidates failed the 0.995 decode bar: QKV-only 0.99458730,
O-only 0.99474771, and cumulative 0.99258065 at both batches across LoFi and
HiFi2. Selected BFP8/LoFi passed with decode PCC 0.99664833 and prefill PCC
0.99896813/0.99897853. Twenty-replay latency was 0.178979 ms b1 and
0.252752 ms b32. Evidence:
`candidates/review_attention_precision/{results,selected}.xml`.

## Sparse output subblocks: fixed and promoted

The legal cumulative 1x2 candidate uses 12 cores for gate/up and 32 for down.
Authentic layer-1/layer-4 decode passed at PCC 0.99931071/0.99974180.

| workload | 1x1 control | cumulative 1x2 | delta |
|---|---:|---:|---:|
| traced decode b1 | 0.791673 ms | 0.725390 ms | -8.37% |
| prefill b1 seq128 | 13.990679 ms | 13.544480 ms | -3.19% |

The cumulative geometry is now default. Evidence:
`candidates/sparse_subblock_{baseline,cumulative}_{b1,prefill_b1}.json` and
`candidates/sparse_subblock_cumulative.xml`.

## Batch-32 routed MoE: AutoFix exhausted

No model-local composition retains every contribution from the fast rolling
`moe_compute` buffer. Full-output sparse candidates remain 5–7x slower than
the selected dense path, while the complete fused consumer requires fabric.
Host combine violates the traced device-resident contract. The missing
capability is a shared-TTNN local-only combine or compact persistent routed
output, outside the user's authorized stage files.

AutoFix closed both model-local findings and failed on this third finding
after exhausting the legal model-local families. The stage cannot claim the
current optimize checklist or a clean review while that scope conflict
remains.

## Rereview sparse remediation

The first sparse remediation was superseded after rereview found that its
named isolated policies inherited cumulative defaults. The test now starts
every isolated row from an explicit all-role 1x1 control and retains the full
legal gate/up/down 1x1–1x4 matrix.

Authentic checkpoint evidence selected a phase-specific policy:

- decode: gate 2/2, up 2/1, down 4/4, `0.704813 ms`;
- grouped prefill: retain gate/up/down 2/2 because the decode winner regressed
  both sequence 33 and 128;
- correctness: PCC `0.99931071` layer 1 and `0.99974180` layer 4;
- final focused tests: seven passed, including ten bitwise-identical trace
  replays and non-aligned sequence 33/128 prefill;
- final normal and watcher suites: 41 passed, 17 intentional opt-in skips;
  the 3,247-line watcher log is fault-signature clean.

Full rows and profiler evidence are recorded in
`SPARSE_SUBBLOCK_HYPOTHESIS.md`. This closes the sparse rereview finding.
Batch-32 routed MoE remains the same exhausted shared-TTNN capability blocker.

Final review also flagged a 5.30% seq33 mean discrepancy between two separate
prefill processes. AutoFix added an alternating same-session harness and ran
five warmup plus fifty measured pairs at seq33 and seq128. Final versus
retained-S2 mean/median deltas were -0.47%/-0.08% and +0.18%/+0.49%,
respectively. The prefill programs are asserted identical, so the earlier
gap is explained as host scheduling noise. Evidence:
`candidates/sparse_subblocks/interleaved_prefill{33,128}.json`.

## Fresh resumed-stage AutoFix

`AUTODEBUG_B32_FRESH.md` re-audited the current API and kernel contracts from
scratch. It found no additional model-local family that preserves complete
routed output, trace safety, fabric-free execution, and the batch-32
no-regression requirement.

The missing capability has since landed upstream as commit `50c56281566`
(`Feature: Add single-device fused moe_compute support (#49886)`), which is
contained in `origin/main` but not in this checkout's HEAD. That change adds
`MoEComputePath::FullLocal` and a local selective-reduce-combine writer across
shared TTNN sources. It permits `compute_only=False, cluster_axis=None` on a
1x1 mesh and drains each rolling expert output while it is live, without
fabric.

This cannot be reproduced from `optimized_decoder.py`: current HEAD has only
`Full` and `ComputeOnly`, and the necessary producer/consumer work occurs
inside the shared device program. Advancing shared TTNN to `50c56281566` or
newer is therefore the concrete external dependency. The original goal
explicitly permits edits only to the model-local optimized decoder, tests,
and docs, so applying or backporting that shared change is not authorized in
this stage.
