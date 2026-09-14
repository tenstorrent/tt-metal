# Qwen3.8-27B optimized full model — stage 7

| Warmed batch-1 measurement | Before | After (final default) |
| --- | ---: | ---: |
| TTFT, S128/G128 | 86.312 ms | 59.295 ms |
| Traced token-out, immediate delivery | 39.433 t/s/user | 40.328 t/s/user |
| Traced token-out, deferred complete delivery | Not implemented | 40.385 t/s/user |
| Traced teacher forcing with token delivery, S203/G100 | 39.300 t/s/user | 40.238 t/s/user |
| Queued token-out, final-token check outside timing | 39.536 t/s/user | 40.394 t/s/user |

Full64-layer measurements: [before_full.json](before_full.json) and
[after_full_final.json](after_full_final.json). The selected TTFT is 31.30%
lower. Deferred decode includes the final history transfer and output-list
construction; immediate decode includes every token read. The queued row is a
diagnostic and excludes final delivery. Teacher forcing explicitly refreshes
reference token inputs and reads sampled tokens, so its rate is separate from
autoregressive device feedback. Each run has a `.source.sha256`, `.commit`,
`.environment.json`, `.log`, and `.exit_status`; [commands.log](commands.log)
records exact invocation arguments.

**Stage status: validated; independent [stage review](stage_review.md) clean-pass.**
Final default measurements reproduce the selected optimizations. Accuracy,
qualitative, maximum-context and matched profiler evidence are complete.

## Selected implementation and mesh

Qwen/Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
64 layers (48 linear,16 full attention), four Blackhole p300c devices,
`MeshShape(1,4)`, TP4 Ring. Before opening the mesh, use the measured fabric
configuration:

```python
from models.autoports.qwen_qwen3_8_27b.tt.generator import configure_fabric
configure_fabric()  # Ring, 8192-byte payload; then open MeshShape(1, 4).
```

The selected Stage5 decoder policy is preserved: BFP4/LoFi projections with
FP32 accumulation, BF16 activation/residual/norm/CCL, BFP8 paged KV, FP32
recurrence, and BF16 convolution history. B1 residuals remain replicated across
TP ranks and width-sharded across40 L1 cores per chip; this is the inherited
measured winner, not a newly introduced replicated fallback. B2–32 retain the
inherited DRAM boundary. The decoder rejection ledger and inter-layer contract
are linked in [full_path_checklist.md](full_path_checklist.md).

- The vocabulary-sharded BF8/HiFi2 head uses four local chunks (16384/16384/
  16384/12928), two readers per bank and K block5. Selected head microtrace
  949.427us versus999.725us before. Corrected reader1/3 alternatives lose;
  block10 at the selected chunk size exceeds L1. A local LoFi trial misses the
  .999 PCC gate on one real-activation shard and is rejected.
- Common split sampling keeps physical32 candidates per rank, tiny candidate
  gathers and semantic greedy k1/p0/T1. Sampled k1–32/top-p remains available.
  The current parallel LargeIndices TopK route is retained; power-of-two pad
  and force-argmax controls are slower. All128 argmax-control tokens equal
  split greedy. There is no selected full-vocabulary all-gather.
- Model and sampling traces remain separate. Sampling writes `tt_out_tok`
  directly into the next decode input; position and RoPE advance on device.
  Persistent page tables refresh only when changed. Steady decode submits
  nonblocking traces without host conversion, upload, synchronization or read.
- Complete generation records all32 physical token lanes into persistent
  UINT32 history inside the sampling trace, then transfers history once.
  Explicit callbacks and host-compatibility controls retain immediate output.
- Owned batch1 generation caches one prefill graph for logical lengths1–4096.
  Stable tokens, absolute positions and padded logits precede capture; model
  and sample traces are recorded before prefill to preserve buffer lifetimes.
  Longer prompts use the existing4096-chunk path and remain valid through the
  advertised context. Public mixed-slot/continuation/all-logits prefill retains
  independent output ownership. [Prefill design](prefill_trace_design.md) and
  [intermediate source snapshots](source_snapshots/manifest.json) record details.
- Fabric payload8192 improves full decode about2% over4352 with topology,
  links, dtype and residual policy unchanged. `configure_fabric()` is used by
  the full-model harnesses; benchmark `--fabric-payload-bytes 4352` retains the
  control. Shared persistent CCL buffers and the host pass-through pool remain
  enabled.

## Correctness and capability evidence

Fresh Stage7 [standardized AIME24 checks](readiness_final.json) pass: prefill
top1=.99 and teacher-forcing decode top1=.98; both have top5=top100=1.0 across
100 positions. These equal the Stage6 full-model scores. The accuracy harness
reads full logits and computes rankings; its diagnostic throughput is not the
sampling-inclusive warmed benchmark above. All six shared qualitative prompts reach coherent EOS-complete answers; the
longer story completes at token1700 with an exact1024-token prefix match to
the shorter run. [Qualitative review](qualitative_review.md) records outputs
and controls. Quality uses intact templates and pinned HF token IDs.
The fixed S128 timing fixture is explicitly not qualitative evidence.

Current accepted contract evidence:

| Artifact | Verified scope |
| --- | --- |
| [contract_final_full_b32.json](contract_final_full_b32.json) | Full64, final fabric, slots31/0, mixed31/33 prompts, inactive state, changed-only pages, physical remapping, all32 history lanes and sampling-mode recapture; continuation PCC.99921447. |
| [prefill_contract_full.json](prefill_contract_full.json) | Full64, lengths1/31/32/33/4095/4096/4097, exact eager/traced token parity, changed prompts/modes, persistent addresses on all devices, retained public outputs and physical K/V remapping;50 guarded captures without a program-cache miss. |
| [prefill_deferred_reduced.json](prefill_deferred_reduced.json) | Delivery matrix on real layers0/3, sampled/greedy transitions, long-to-short history reuse and independent no-host steady-decode guards. |
| [watcher_prefill_final.json](watcher_prefill_final.json) | Real layers0/3 on all4 chips, selected8192 fabric, S33 and changed-input/mode/page/lifetime checks under watcher and allocation tracking; no ETH disablement. |
| [runtime_fallback_audit.md](runtime_fallback_audit.md) | Final counters, device-only guards, native sampler and trace evidence. |

The [context contract](../context_contract.json) preserves262144 tokens,
page32, logical non-aligned lengths, explicit cache/position/prompt-length/batch
state, fixed slots and inactive rows. Batch32 short prompts and batch1 maximum
context are separate coverage points. No capability was reduced. The updated
memory plan totals13,538,275,840 bytes per device, leaving20,600,412,672 bytes
of physical headroom, with conservative history/scratch and prefill-anchor
reserves. Final source hashes and full64 S262143/G2 plus S262144/G1 executions
are recorded in [readiness_final.json](readiness_final.json).

History capacity is a request-sized high-water mark. The existing indexed-fill
and copy sequence moves the allocation each step, and final delivery transfers
the whole allocation once before taking the valid prefix. It is not an in-place
single-row DMA or a partial host transfer. Storage is bounded by context (32MiB
history per chip plus conservative scratch); shorter requests reset the cursor
without clearing retained history. Cold captures and larger prior requests can
increase request cost; warmed metrics identify their exact fixture.

## Performance accounting and reports

[perf_summary.json](perf_summary.json) derives each device's accounting from
real layers0/3 plus the full embedding/norm/head/sampling/history path, with
cache256/page-table8/history127 matching the S128/G128 benchmark allocation.
Runtime model/generator/decoder/native-library provenance matches the final
full benchmark. The representative kernel-only stack expands to22.315–22.446ms
for48 linear plus16 full-attention layers. Adding entry and terminal kernels
once gives23.756–23.884ms, versus24.762ms/token for complete full-model delivery:
a3.7–4.2% margin over the kernel estimate, below the10–15% investigation threshold. This is an
empirical estimate, not a directly profiled all64-layer interval or measured
host overhead. Gap-inclusive expansion26.444–26.556ms overshoots full wall time.
The inherited isolated-layer sum25.156384ms includes per-layer synchronization.

The final terminal boundary costs.967–.984ms, sampling without history
.497–.512ms, and history append at capacity127 about.008ms including local gaps.
Parallel110-core LargeIndices TopK costs about.320ms (1.3% of full decode);
there is no dominant generic TopK, force-argmax or full-vocabulary gather.
Representative prefill gaps fall from3.518–3.568ms eager to.731–.735ms traced,
while kernels remain about3.05ms. This closes the large avoidable TTFT gap.

The mandatory stored-projection/head/minimum-KV read estimate is about3.928GB
per device/token at the measured context range, or7.671ms at nominal512GB/s.
It excludes other traffic, writes, rereads, recurrence, CCL and dispatch; it is
not an achievable full-model target or measured utilization. In the same
reduced profiler window, token-out device span2.274–2.279ms compares with host
signpost2.713ms including end synchronization. That difference is specific to
the instrumented reduced run, not full-model Python overhead.

All-device advice tables, phase CSVs and portable compressed op CSVs are under
[tracy/profile_final_buffers](tracy/profile_final_buffers); the eager control is
[tracy/before_prefill_trace](tracy/before_prefill_trace). Duplicate raw streams
are archived outside Git; each directory's `raw_archive_manifest.json` supplies
paths, sizes and hashes. Decompress `device*_ops.csv.gz` to restore uncompressed
CSV paths when needed. [The summarizer](../../tests/summarize_full_model_perf.py)
reads compressed sources directly. The perf tool's eight-worker matmul FLOPs
denominator is invalid for native16/24-worker programs; displayed percentages
are retained as raw tool output and excluded from utilization claims.
[Earlier terminal accounting](profile_analysis_terminal.md) remains historical.

## Reproduction, recovery and review

From the configured checkout, the stage launcher exports the packaged runtime
environment and records provenance. Representative commands:

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/run_optimized_full_model_experiment.sh \
  prefill_contract_full models.autoports.qwen_qwen3_8_27b.tests.check_prefill_tracing --full --lengths 1,31,32,33,4095,4096,4097
bash models/autoports/qwen_qwen3_8_27b/tests/run_optimized_full_model_experiment.sh \
  after_full_final models.autoports.qwen_qwen3_8_27b.tests.benchmark_full_model --full --qualitative-story
```

[work_log.md](work_log.md) records the exact chronology, failed/adapted controls,
commands and recovery. Startup ownership/sysmem faults were resolved by the
host operator; a later isolated initialization transfer stall was captured and
recovered. [AutoFix startup](AUTOFIX_startup.md), [transfer triage](AUTOTRIAGE_head_startup.md),
and [head geometry diagnosis](AUTODEBUG_head_geometry.md) preserve evidence.
These are historical incidents, not current blockers. Watcher uses the existing
NOINLINE/fabricO3 settings without disabling Ethernet checks; profiler and
watcher are separate. No C++ changes or new build are required by this stage.

Final default tests and host checks pass. Independent `$stage-review` returns
clean-pass; [work_log.md](work_log.md) records local checkpoint provenance. No vLLM integration or push is included.
