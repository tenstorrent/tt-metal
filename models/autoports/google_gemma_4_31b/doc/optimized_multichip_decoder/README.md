# Gemma 4 31B optimized multichip decoder (Stage 05)

Status: complete.  The original Stage 05 review and the fresh 2026-07-17
current-checkout review both returned `clean-pass`.

## Outcome

The final default remains the real four-device `MeshShape(1, 4)` TP decoder.
It uses Linear fabric, two persistent BFP8 asynchronous TP reductions during
decode, native BF16 reductions during prefill, packed TP-local QKV with block
7, a packed BFP8 gate/up decode projection, and a 14-core DRAM-sharded local
MLP.  The public layer boundary remains replicated BF16, so there is no
gather, reshard, reduce-scatter, or all-reduce between decoder layers.
Full-model and vLLM work were not started.

The final path is dense because Gemma 4 31B is not an MoE model.  The MoE
active-expert requirement is therefore not applicable.

## Target and preserved contract

- Hardware: 4x Blackhole P150b, firmware bundle 19.9.0, TP=4, 1x4 mesh.
- Representative layers: layer 0 sliding attention and layer 5 full attention.
- Layer input/output: replicated BF16 in DRAM.
- Decode working activations: width-sharded L1; attention O and MLP down are
  resharded once at the collective boundary to the final 24 worker cores, then
  return once to replicated DRAM for the residual add.
- Attention weights: BFP8; MLP weights: BFP4; paged KV cache: BFP8;
  prefill collective input/output: BF16; decode collective input pages/output:
  BFP8 with restoration to the BF16 layer boundary; packed gate/up output:
  BFP8.
- Public sequence lengths remain logical and may be non-aligned.  Existing
  owned padding, causal/sliding masks, cache modulo, and output slicing remain
  inside the decoder.
- Advertised context remains 262,144.  The shared persistent CCL pool adds
  731,136 L1 bytes per device and does not change KV DRAM capacity.  See
  `../context_contract.json`.

### Inter-layer residual contract

Full-model bringup must pass each decoder's replicated BF16 DRAM output
directly to the next decoder's replicated BF16 input.  It must not insert an
all-gather, reduce-scatter, all-reduce, reshard, or layout conversion between
layers.  Each layer owns its two row-parallel reductions internally.  All
layers on a mesh share the same two global semaphores and one physical-capacity
scratch buffer on the tail 24-core grid; allocating a private pool per layer would multiply L1 use by the
number of layers and violates this stage's contract.

## Operation-topology audit and action

| Group | Audited starting topology/cost | Action and evidence | Final decision |
|---|---|---|---|
| input/QKV | replicated residual -> norm -> L1 reshard -> packed local QKV; packed matmul followed by head-layout conversion | Compared packed with legal split Q/K/V and swept 8/16/32-core packed geometries. Review remediation kept the declared 0.995 PCC gate and reran block 7 cumulatively: both kinds passed at 0.999802/0.999718. | packed QKV, 32 cores, block 7 |
| SDPA/cache | local heads, BFP8 paged cache, explicit q32/k64 exact-exp SDPA | Tried k32, k128, approximate exp, and full-grid full-attention. Alternatives were neutral/slower or reduced PCC; BFP4 KV fell to 0.996010 PCC. | q32/k64 exact exp, BFP8 cache |
| O projection | DRAM-sharded decode weight -> L1 width-sharded matmul -> DRAM conversion -> synchronous all-reduce | Kept the L1 result sharded into persistent `all_reduce_async`; block-16 was retried on full attention and collided with L1 after persistent scratch, while block-8 passes. Decode typecasts its BF16 partial to BFP8 pages; prefill remains BF16. | block 8, persistent BFP8 decode reduction |
| gate/up | two same-input BFP4 matmuls | Compared separate, packed BF16, and packed BFP8 in the full trace. Packed BF16 was 0.48807 ms; packed BFP8 was 0.46860 ms with accepted PCC. | packed BFP8 output |
| MLP down | DRAM-sharded BFP4 matmul -> DRAM conversion -> synchronous all-reduce, 24 cores | The review-corrected sweep preserved packed gate/up+BFP8 output for every 7/8/12/14/21/24/28/42/56/84 candidate and measured both MLP-only and whole-layer traces. Fourteen cores won at 0.21669/0.46436 ms versus 24 cores at 0.22198/0.46786 ms. The 7-core block-24 L1 failure was adapted to block 12 and passed but lost end-to-end. | 14 cores/block 12, native-BFP8 persistent reduction |
| collective dtype | BF16 two-link reductions in the starting path | Review found that the original BFP8 rejection used an undocumented stricter PCC rule. The current cumulative QKV7+BFP8 family passes the declared 0.995 gate. Applying it to both phases hurt prefill, so the final coherent phase policy uses BF16 for prefill and BFP8 for decode. | BF16 prefill; BFP8 decode |
| collective implementation | two material synchronous collectives plus two sharded-to-interleaved conversions | Adapted the first DRAM API failure to the Blackhole-required L1 input/scratch/output contract and shared preallocated buffers. | persistent async, 2 links |
| residual family | replicated BF16 boundary | The coherent fractured family was already 0.2734 vs 0.1616 ms decode and 0.8545 vs 0.6796 ms prefill. Fused MM+RS passed actual Gemma shape only on Ring, but the required distributed norm/delayed gather family remains slower even before its gather. | replicated BF16 boundary |
| topology/fused CCL | Linear default; the available fused MM+RS and AGMM implementations require Ring | Linear exact-shape MM+RS hung and was triaged/reset; Ring MM+RS passed at PCC 0.999963 but its complete decoder family lost. Exact TP4 AGMM then passed sliding/full O and adapted BFP8 down at PCC >=0.999964. A real-weight coherent H/TP-local spine passed at 0.999752/0.999695, but cost 925.49/1026.01 us versus 349.77/384.95 us for matching replicated endpoints. Immediate-gather compatibility variants also lost. | Linear; no fused MM+RS/AGMM |
| persistence | per-call temporary CCL state | Two semaphore epochs and one TP4-volume tail-24 scratch tensor are shared by every live decoder on the mesh. Tests assert stable one-buffer reuse after both layer kinds. | selected |
| fidelity/precision | LoFi BFP8 attention, LoFi BFP4 MLP | The current-topology BFP4-attention retry passed sliding at 0.997253 but failed full attention at 0.992305 below the declared 0.995 gate. KV BFP4 also failed; HiFi2 attention/gate/down all regressed latency. | BFP8 attention, BFP4 MLP, packed BFP8 output |

The fused all-gather+matmul review remediation used the exact Gemma TP4 shapes,
real checkpoint weights, and a coherent lower-movement endpoint.  The Ring
kernel used one link, dimension 3, four transfers, persistent gathered-input
buffers, DRAM-interleaved storage, an 8x6 grid, and a 1x1 subblock.  Sliding O
(K=8192, local N=1344) passed at PCC 0.999964, full O (K=16384, local N=1344)
at 0.999964, and down (K=21504, local N=1344) was adapted from a failing BFP4
PCC of 0.993139 to BFP8 PCC 0.999965.

The coherent real-weight family kept H/TP local through fused O, distributed
norms/residuals, fused packed gate/up, fused down, and the next fused QKV; it
did not immediately restore the old replicated contract.  It passed at PCC
0.999752/0.999695 but warmed at 0.95258/1.05563 ms versus matching baselines
of 0.38441/0.41347 ms (2.478x/2.553x slower).  Tracy device sums were
925.49/1026.01 us versus 349.77/384.95 us.  Separate immediate-gather
compatibility candidates were also slower: O 105.14 vs 49.21 us sliding and
188.84 vs 72.59 us full; down 221.11 vs 72.47 us sliding and 222.30 vs
72.85 us full.  This exact family measurement supersedes the earlier
source-backed bound and rejects fused AGMM without a mixed-family comparison.

## Original Stage 05 correctness and performance

PCC uses the accepted Stage 04 optimized single-chip oracle and covers all
four device replicas.  These historical completion numbers came from the
selected default, not an earlier candidate; the later current-checkout table
is the authoritative reproduction on the present source tree.

| Metric | Before Stage 05 | Final default | Change |
|---|---:|---:|---:|
| sliding prefill-128 PCC | 0.999846 | 0.999846 | preserved |
| full prefill-128 PCC | 0.999757 | 0.999757 | preserved |
| sliding traced decode PCC | declared gate 0.995 | 0.999802 | accepted |
| full traced decode PCC | declared gate 0.995 | 0.999718 | accepted |
| sliding warmed prefill-128 | 2.41804 ms | 2.40641 ms | 0.48% faster |
| full warmed prefill-128 | 2.18014 ms | 2.20546 ms | 1.16% slower (run variation) |
| sliding traced warmed decode | 0.52649 ms | 0.46381 ms | 11.91% faster |
| full traced warmed decode | 0.57565 ms | 0.51663 ms | 10.25% faster |

The final latency rerun and profiler tables are under `evidence/final_latency.*`
and `tracy/final_post_fused_review/`.  The standard suite covers prefill-33, non-aligned sliding
1025/1057 with internal wrap/padding, paged decode with a permuted page table,
eight identical trace replays, mutable token/position replay, batch-32
non-aligned prefill/decode, replica equality, cache/layout contract, and the
runtime source audit.  The advertised-position trace reuses unchanged BFP8 KV
layout while exercising the final default at position 262,143.

### Original Stage 05 decode performance accounting

The Tracy workload and warmed host workload both use logical batch 1, a
32-token populated cache, real layer shapes, and one traced replay.  BFP8 and
BFP4 physical payloads use 1.0625 and 0.5625 bytes/value respectively.  At the
Blackhole report model's 512 GB/s per-device DRAM ceiling, the mandatory local
weight plus K/V read is 57,614,336 bytes (0.11253 ms) for sliding attention and
61,966,336 bytes (0.12103 ms) for full attention.

| Layer kind | Roofline | `tt-perf-report` device-op sum | Warmed end-to-end | Remaining e2e minus device |
|---|---:|---:|---:|---:|
| sliding | 0.11253 ms | 0.42830 ms | 0.46381 ms | 0.03551 ms |
| full | 0.12103 ms | 0.48162 ms | 0.51663 ms | 0.03501 ms |

The remaining roughly 35 us is traced replay/synchronization overhead.  The
report's merged multi-device `Op-to-Op Gap` sum is not used as a critical-path
number because rows selected from different devices overlap; its per-op device
durations are valid and reproduce the host/device delta above.  Dominant decode
rows are the packed gate/up BFP4/LoFi matmul (103 us), down BFP4/LoFi matmul
(54 us), attention BFP8/LoFi projections (29-63 us), and the two persistent
BFP8 async reductions (about 12 us each).

## Candidate summary

| Candidate | Correctness | Measured result | Decision |
|---|---|---|---|
| synchronous BF16 starting path | accepted | 0.52649/0.57565 ms decode | replace |
| BFP8 synchronous CCL | PCC 0.999879/0.999825 | 0.52811/0.58001 ms | reject |
| persistent BF16-output CCL | PCC 0.999967/0.999896 | 0.48592/0.53729 ms | intermediate winner; superseded |
| persistent BFP8 CCL, historical topology | PCC 0.999906/0.999849 | 0.47667/0.52814 ms | reopen after review; old rejection invalid |
| QKV7 + BFP8 CCL in both phases | PCC 0.999802/0.999718 | prefill 2.60662/2.60277; decode 0.46441/0.51931 ms | adapt phase policy |
| QKV7 + BF16 prefill/BFP8 decode CCL | PCC 0.999802/0.999718 | final 2.40641/2.20546 prefill; 0.46381/0.51663 decode | select |
| BFP4 attention on current topology | 0.997253 sliding; 0.992305 full | full layer fails declared 0.995 gate | reject |
| packed-BFP8 MLP geometry, 14 cores | PCC 0.999802 | 0.21669 ms MLP / 0.46436 ms layer | select |
| packed-BFP8 MLP geometry, 24 cores | PCC 0.999803 | 0.22198 ms MLP / 0.46786 ms layer | reject slower |
| adapted Ring k64/MLP24 | PCC 0.999959/0.999890 | current-source rerun 0.47776/0.52786 ms; no decode win | reject |
| fractured residual | PCC >=0.99987 | slower prefill and decode boundary | reject |
| fused MM+RS actual shape on Ring | PCC 0.999963 | isolated op passes; coherent family loses | reject |
| fused AGMM exact O/down shapes on Ring | O PCC >=0.999964; down BFP4 0.993139, adapted BFP8 0.999965 | primitive API/shape path established; BFP4 down fails gate | adapt and measure coherent family |
| fused AGMM coherent H/TP-local residual spine | PCC 0.999752/0.999695 | 0.95258/1.05563 ms vs 0.38441/0.41347 ms; 925.49/1026.01 us device sums vs 349.77/384.95 us | reject slower |
| prefill larger-grid family | PCC >=0.999955 | 28/42/56 cores: 0.68133/0.71375/0.67339 ms vs 0.64982 ms default | reject |
| prefill L1 input family | PCC 1.0 after block-sharded adaptation | 0.80624 ms including two reshards vs 0.64982 ms | reject |

Machine-readable values are in `candidates/summary.csv`; raw logs and JUnit
XML remain beside it.  Failed first attempts are intentionally retained to
show each API/layout/padding adaptation rather than treating the first TTNN
error as a rejection.

## Runtime safety and limitations

- The hot forward methods contain no host fallback (`to_torch`, host math, or
  host cache update).  Persistent buffer creation is a one-time construction/
  warmup action before trace capture; replay is device-only.
- Watcher is run separately from Tracy.  Final watcher and device logs contain
  no watcher error, assert, hang, or stale-buffer report.
- One Linear fused-MMRS experiment hung.  `tt-triage` evidence was captured,
  the host was terminated, all four devices were reset with `tt-smi -r all`,
  and health was rechecked before further work.  The Ring retry passed.
- Persistent scratch addresses must remain stable while traces exist.  Do not
  clear or replace the shared pool to make room for a later prefill.
- After every trace is released, the mesh owner calls the terminal resource
  cleanup before `close_mesh_device`; this synchronizes, deallocates scratch,
  clears semaphore/mesh references, and removes the registry entry.  Cleanup
  is never performed per layer or while a trace can replay.
- The final public contract is intentionally replicated.  A future full model
  must preserve it; revisiting fractured residuals would be a new measured
  family, not an implicit layer-boundary conversion.

## Reproduction and artifacts

Core commands, environment, failures, recovery, and artifact paths are in
`work_log.md`.  The main evidence is:

- `evidence/baseline_latency.log` / `.xml`
- `evidence/final_suite.log` / `.xml`
- `evidence/final_latency.log` / `.xml`
- `evidence/exact_context.log` / `.xml`
- `evidence/watcher_device.log` and `watcher.xml`
- `candidates/summary.csv` and all raw candidate logs/XML
- `candidates/prefill_advice_sweep*.log` / `.xml`
- `candidates/ring_l1_AUTODEBUG.md` / `ring_l1_AUTOFIX.md`
- `candidates/fused_agmm*.log` / `.xml` exact primitive, adaptation, and
  source-current real-weight coherent-family runs
- `tracy/final_post_fused_review/<layer>_<mode>/` source CSV, filtered CSV, summary CSV/PNG,
  human-readable `report.txt`, and provenance README
- `tracy/fused_agmm_coherent/` enriched source CSV, 12 signpost-filtered
  advice/CSV/summary report families, source hash, and provenance README

The stage-owned local commit SHA(s) and independent review verdict are recorded
at the end of `work_log.md`.  Nothing is pushed.

## 2026-07-17 current-checkout revalidation

The optimized decoder was revalidated at repository HEAD
`b68b16df75d121dcbe0128d6fa4cea98f993b870` after later repo-local work had
touched shared decoder code.  This pass did not run full-model, generation,
vLLM, or serving paths.  It opened the target `MeshShape(1, 4)` over four
Blackhole P150b boards and measured `MultichipDecoder` directly.

The current operation-topology audit matches the reviewed Stage 05 winner:
packed local QKV; explicit SDPA with local BFP8 KV heads; row-local O plus a
persistent BFP8 async all-reduce; packed BFP4/LoFi gate-up with BFP8 output;
BFP4/LoFi down plus the second persistent reduction; and a replicated BF16
DRAM layer boundary.  The fresh reports have the same 33/30 prefill and 59/58
decode op counts as `tracy/final_post_fused_review`.  Current device sums are
1,153.2255/1,418.152 us for sliding/full prefill and
427.9625/481.827 us for sliding/full decode.  No inter-layer collective,
gather, or reshard is present; both reductions remain owned inside the layer.

The exact packed/split projection, residual-layout, CCL placement and dtype,
persistent-buffer, Ring/Linear, fused MMRS, and fused AGMM families remain
applicable because the measured production topology and real tensor contracts
are unchanged.  Their retained current-source evidence is in
`candidates/summary.csv`, including the exact real-weight fused AGMM coherent
spine that is 2.478x/2.553x slower without an immediate restore to the old
replicated residual.  The current profile confirms the selected policy reached
runtime: BFP8/LoFi QKV and O, BFP4/LoFi packed gate-up and down, BFP8 cache,
and two BFP8-input/BFP8-output async reductions restored to BF16 at the layer
boundary.

Current-head acceptance results (the authoritative final-default numbers for
this revalidation):

| Metric | Stage 05 starting multichip path | Current final default | Result |
|---|---:|---:|---:|
| sliding prefill-33 PCC | 0.999846 | 0.999846 | preserved |
| full prefill-33 PCC | 0.999757 | 0.999757 | preserved |
| sliding traced decode PCC | gate 0.995 | 0.999802416 | pass |
| full traced decode PCC | gate 0.995 | 0.999718188 | pass |
| sliding warmed prefill-128 | 2.418039 ms | 2.3350725 ms | 3.43% faster |
| full warmed prefill-128 | 2.180140 ms | 2.4464875 ms | 12.22% slower wall time |
| sliding traced warmed decode | 0.5264925 ms | 0.4635775 ms | 11.95% faster |
| full traced warmed decode | 0.575653 ms | 0.5181375 ms | 9.99% faster |

The full-prefill wall-time shift is classified rather than dismissed: its
fresh on-device sum is 1.418152 ms versus 1.42228 ms in the prior clean-pass
profile, with the same 30 operations and dominant rows.  The device graph is
0.29% faster, so the 12-sample wall difference is untraced host/harness
variance rather than added device work.  The same-run TP4 path still beats the
single-P150 full-prefill control by 1.396x.

The standard suite is 12/12, including both layer kinds, non-aligned 33-token
prefill, sliding 1,025/1,057 wrap, permuted page tables, mutable/repeated trace
replay, batch 32, BFP8 cache/layout, replica equality, persistent-pool teardown,
and a clean source fallback audit.  Both advertised-position traces pass at
262,143 with the unchanged 262,144 context contract.  A separate watcher run
passes four target-mesh tests with worker/NoC checks enabled; Ethernet watcher
instrumentation remains scoped off only for the documented firmware config
buffer limit.

Current artifacts are under `evidence/current_head_*` and
`tracy/current_head/`.  `perf_accounting.json` reconciles the current roofline,
device, and end-to-end decode numbers.  Raw multi-hundred-megabyte profiler
intermediates were removed after the compact enriched CSV, hash, reports,
tables, PNGs, and JUnit results were retained.

The independent current-checkout report is `stage_review_current_head.md`; its
verdict is `CLEAN PASS` with no required work.
