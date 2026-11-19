# Qwen3.6-27B TP4 decoder

All 54 delivery gate artifacts pass; independent [round-two review](stage_review_round2.md) returns `clean-pass`. The implementation is a real tensor-parallel
subclass of the frozen single-chip `OptimizedDecoder`, targeting all four
Blackhole P300c chips on this machine. No full-model or vLLM implementation is
part of this stage. The accepted-source gate suite uses prefix `delivery_`;
earlier `final_safe_*` results predate the large-matrix crossover;
`final_perf`/`final_profile` results include the watcher-rejected
unchunked native MMRS path and are historical evidence.

The final source uses separate row matmul/reduce-scatter through 512 flattened
rows and above 8192 rows, with 128-row fused pieces for 513..8192 rows. The latter selects the native NoC0
reader route; the original tall-grid NoC1 path violates dedicated counter
ownership. [AutoDebug](debug_narrow_watcher/AUTODEBUG.md) and
[AutoFix](AUTOFIX.md) preserve the diagnosis, repair and verification. Ordinary
prefill uses 110 cores/K8; small native SwiGLU uses88 cores with a measured
configuration. Decode retains compact batch/head layouts and narrow correction.

The stack boundary is BF16 `[B,S,1280]` per rank, hidden-sharded from 5120.
Each rank owns six full-attention Q heads and one KV head, or four GDN K heads
and twelve V heads. The mesh is 1x4 FABRIC_1D, Linear axis 1, two links. The model
is dense, so routing and expert parallelism do not apply.

The public context remains 262144, with no capability reduction. Internal
chunk 512, tile 32, page 32 and DRAM-weight padding do not restrict logical
sequence lengths. Maximum context is tested at batch 1; short/medium contexts
are tested through batch 32. Aggregate cache allocation must fit device DRAM.

## Design and numerical contracts

- [Mesh plan](mesh_plan.md): initial decision before implementation, per-rank
  ownership, rejected alternatives, and selected phase policies.
- [Kernel/stack contracts](kernel_contracts.md): compact local-head kernels,
  QKV padding, GDN grouping, cache ownership, residual layout and trace rules.
- [Capacity plan](capacity_plan.json): all 64 layer weights, phase copies,
  embedding/head estimates, all 16 KV caches, all 48 recurrent states and padded
  auxiliary tensors. No all-token maximum-context logits allocation is assumed.
- [Optimization decisions](optimization_checklist.md): measurements, retained
  candidates, rejections, and inherited baseline decisions.

Delivered decoder SHA256 is
`082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`.
The baseline source SHA256 is
`315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`.
Checkpoint revision is `8f33b98937c6031bb49e713dee3e8e185aafd26f`.
All correctness comparisons use the same real weights, inputs, page mappings
and positions; recorded target activations are used for precision decisions.
PCC threshold is 0.995, including layer contribution and cache/state checks.

## Evidence and reproduction

Run from the tt-metal root after sourcing `python_env/bin/activate` and
`/home/ttuser/qwen36-27b/lane-env.sh`. Device jobs are serialized. Every
`run_job.sh` invocation records exact argv, UTC, environment and source hashes
in its `.command.json`; suite `.jsonl` files also record exit codes. A wrapper
exit code alone is insufficient: the script verifies the target JSON exists
and passed. Source hashes in intermediate candidate evidence describe those
historical implementations, not the final default.

```bash
bash models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/run_delivery_gates.sh
python -m models.autoports.qwen_qwen3_6_27b.tests.summarize_multichip --tag delivery_perf
python -m models.autoports.qwen_qwen3_6_27b.doc.multichip_decoder.finalize_context
python -m models.autoports.qwen_qwen3_6_27b.doc.multichip_decoder.render_profile_summary --profile-prefix delivery_profile_retry
python -m models.autoports.qwen_qwen3_6_27b.tests.collect_multichip_evidence --prefix delivery --profile-prefix delivery_profile_retry
```

The context runner computes the full decoder on every token, compares selected
output rows, full GDN state/history and selected physical KV pages against
fresh optimized TTNN references. At 262143 it decodes the last legal position
and checks changed-input trace output/state equality. At 262144 it tests full
prefill and full-attention continuation, without an out-of-contract next token.
Untouched sampled cache rows must remain bit-equal after continuation.
Every TP4 context run holds an additional 20 GiB allocation per device throughout
execution; this exceeds the estimated full-stack resident set. It validates
aggregate capacity, not identical future allocator fragmentation.

Stress runs 128 steps at B1 and 512 at B32, twice, with changed inputs and
advancing attention positions. Outputs and final mutable state must be exactly
repeatable and pass baseline PCC. Trace validation includes allocation tracking,
program-cache allocations and an unsafe-allocation positive control; there are
no allocation acknowledgments. Focused QKV/compaction tests additionally check
rank/user ownership and padded values exactly.

The 64-instance resource test shares representative weights but owns distinct
CCL managers, convolution exchange/semaphore resources and cache/state. It
warms every instance at B1/S65 and actual B32/S33 prefill in separate runs, then captures a 64-layer-equivalent decode sequence with
changed-input replay, complete output/state hashes and allocator/cache-growth
measurements. The separate five consecutive checkpoint-layer test checks numerical handoff
using recorded target-model activations
without boundary conversions. Neither test is full-model inference.

The delivery correctness, watcher and performance cases have minimum aggregate
prefill/decode PCC of 0.999133/0.998291 for GDN and 0.999871/0.999623 for full
attention. Minimum per-user PCC is 0.996482 and 0.999222 respectively; the
separately disclosed synthetic B31/S31 counterexample below is excluded from
those accepted-case minima. Both layer kinds pass repeated stress runs.

Uninstrumented warmed decode is 0.574 ms (B1) and 0.733 ms (B32) for GDN;
full attention is 0.490 ms (B1) and 0.701–0.748 ms (B32). Measured decode
speedup over the frozen baseline is 1.67–2.28x, or 41.9–57.0% four-chip
scaling efficiency. Prefill speedup is 1.22–3.14x. These speedups include the
validated precision and compact-kernel changes documented in
[performance.md](performance.md), as well as tensor parallelism.

## Limits and next-stage expectations

Only this 1x4 hardware target is supported. Execute layers serially on one
mesh/command queue: native fused-MMRS communication resources are shared per
mesh, and concurrent execution on independent queues is not established.
BFP4 candidates are recorded for the dedicated datatype stage; BFP8 remains
the default. Full-attention BFP4 fails the real-activation cache PCC threshold.
Future logits must be last-token or bounded-token-chunk allocations.

A disclosed random-activation GDN B31/S31 control reaches 0.994697 for one
user's decode output under LoFi. It passes with baseline HiFi4; disabling
fused MMRS leaves its decode/state scores unchanged. Recorded target inputs
at the same shape pass the unchanged 0.995 bar on both kinds, with exact trace
and watcher checks. See [matched controls](pcc31_control_summary.json) and
[anomaly ledger](anomaly_ledger.md). The delivery 31x31 watcher case explicitly
uses those recorded inputs. Full-model accuracy remains a subsequent-stage
requirement; this stage makes no end-to-end accuracy claim.

`work_log.md` records failures, AutoFix/triage, command history and review/commit
status. Raw tensor references, private agent transcripts and Tracy databases stay local; compact numerical
results, human profiler tables, CSV and provenance are stage artifacts. Large
logs/JSON/raw-ops archives are losslessly compressed; `evidence_archive.json`
indexes them and its reconstruction command restores original filenames.

## Delivery artifact index

`gate_summary.json` is the fail-closed aggregate of accepted-source evidence.
Its entries name each individual JSON artifact; suite command logs retain exact
arguments and exit codes.

| Gate | Artifacts |
|---|---|
| Layer PCC and non-aligned lengths | `delivery_correctness_commands.jsonl`, ten TP4 results and matching optimized references |
| Adjacent dispatch boundaries | `delivery_boundary_commands.jsonl`, eight TP4 watcher cases at B2/B32 × S256/S257, observed native calls and shapes |
| Watcher and tail shapes | `delivery_watcher_commands.jsonl`, ten TP4 results, per-job watcher logs; 31x31 recorded-input provenance is explicit |
| Warmed latency, speedup, efficiency | [performance.md](performance.md), [performance.csv](performance.csv), [performance.json](performance.json), `delivery_perf_commands.jsonl` |
| Compute, DRAM, CCL, movement | [profile_accounting.md](profile_accounting.md), `perf/delivery_profile_retry_l{0,3}_b{1,32}/` and large-prefill `_b32_s513/` tables/CSV/provenance |
| Sustained decode | `delivery_stress_commands.jsonl`, 128 steps at B1 and 512 at B32, repeated twice |
| Advertised and near-limit context | `delivery_context_commands.jsonl`,262143/262144 for both kinds with 20 GiB/device reservation |
| Consecutive real-layer stack | `delivery_stack_commands.jsonl`, checkpoint layers 0..4 at B2/S65 and B32/S33 |
| 64-owner resources and trace | `delivery_resources_b{1,32}.json`, actual B1/S65 and B32/S33 prefill, distinct resources/states |
| Exact local layouts | `delivery_qkv_watcher.json`, `delivery_compaction_watcher.json` |
| Runtime fallback and host contracts | [runtime_audit.json](runtime_audit.json), [host_validation.json](host_validation.json) |
| Ordinary pytest entrypoint | `delivery_pytest.log` |
| Controlled precision anomaly | [pcc31_control_summary.json](pcc31_control_summary.json), [anomaly_ledger.md](anomaly_ledger.md) |

The source hash in each accepted result must match the delivered decoder.
Historical candidate artifacts retain their own hashes and are not substituted
for accepted-source gates. Large artifacts can be reconstructed with the
command recorded in `evidence_archive.json`.
