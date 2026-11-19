# Final capacity reconciliation

Final closure: [independent stage review round1](stage_review_round1.md) returned
**clean-pass**, with no required work or further hardware runs. The audit below
retains its pre-review provenance; pending-review statements are superseded by
that verdict. Local checkpoint records are in [commits.json](commits.json).

Independent CPU-only arithmetic/source review, 2026-11-14. No device run,
production change or context reduction. The advertised context remains
**262144**. Corrected full48 B32v2 and the once-only full48 B1 maximum-context
execution pass. All42 supported generator-equivalent rows now have scoped
evidence;21 unsupported declarations remain. Fresh independent review and local
stage commit remain pending; this is not a review verdict.

The current runtime envelope remains conservative for the measured workloads.
Charging the pinned worker's extra pool pages leaves calculated free memory of
**386,485,760 bytes/rank at B32/C512** and **1,384,932,352 bytes/rank at
B1/C262144**. These are translations of actual retained-memory snapshots, not
observations of the expanded pool configurations. Actual maximum-context
execution is established separately with the measured8192-block private pool. No additional standalone capacity sweep is required to
state this byte accounting. A literal executed545/8194-block claim would need
that actual setting; a separate repeated full-model run must not be invented
or substituted for the completed corrected B32 and once-only maximum-context tests.

## Units and allocator granularity

`tests/run_full_model_accuracy.py::memory` calls
`ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)`. The nanobind mesh overload
(`ttnn/cpp/ttnn-nanobind/device.cpp`, `GetMemoryView`) calls the mesh allocator;
`tt_metal/detail/reports/memory_reporter.cpp::get_memory_view` copies allocation,
free and largest-free-block statistics into the explicitly **per-bank** fields.
Each rank has8 DRAM banks, each with4,272,341,376 allocatable bytes: total
**34,178,731,008 bytes/rank**. Multiply the snapshot fields by8, not by the4 mesh
ranks. These scalar views are mesh allocator accounting, not four independent
physical-bank telemetry samples.

Accepted BF16 tiled K/V shape is `[N,1,32,256]` per tensor per rank. Each logical
32-token block contains8 tiles, each2048 bytes, exactly one tile per bank:
**16,384 bytes/block per K or V tensor**, **2,048 bytes/block/bank**. Each of12
QSA layers owns K and V, so a full-stack block costs **393,216 bytes/rank** or
**49,152 bytes/bank**. No extra round-up occurs for integer N:8N tile pages are
already divisible by8 banks, and2048-byte pages satisfy the64-byte alignment.
This follows `capacity_arithmetic.py::entry` and `cache_pool.py::PoolSpec`.

A single tensor allocation requires a contiguous extent on every bank. Total
free alone does not prove fit. Pool allocation happens before preparation;
there is no in-place resizing after capture. Enlarging an allocation also may
change compilation/setup details, so subtracting an exact payload increment
from a later snapshot is a **calculated translation**, not an observed new
allocator layout or a guarantee of unchanged transient peaks.

## Actual memory snapshots

All values in this table are observed **bytes per bank** from the named JSON.
The loaded B32 view is before global KV allocation (deferred setup); loaded B1
already includes its full private KV and generator persistent state.

| Artifact / point | Allocated | Free | Largest contiguous free |
| --- | ---: | ---: | ---: |
| `batch32_hf_factored_v2.json` (v1 identical), loaded | 3,821,554,944 | 450,786,432 | 450,785,728 |
| B32 immediately before bounded model capture | 3,852,325,824 | 420,015,552 | 416,349,312 |
| B32 prepared after bounded model capture | 4,126,660,544 | 145,680,832 | 142,014,592 |
| B32 final / at-exit | 4,222,408,640 | 49,932,736 | 46,266,496 |
| `acceptance_accuracy_resume_final_v1.json`, loaded | 3,978,515,648 | 293,825,728 | 292,945,216 |
| B1 prepared | 3,985,659,136 | 286,682,240 | 278,717,760 |
| B1 accuracy final / max run before finiteness | 4,099,118,336 | 173,223,040 | 165,258,560 |
| `acceptance_max_context_final_v1.json`, final after finiteness | 4,099,126,528 | 173,214,848 | 165,250,368 |

B32v1 allocated and used512 blocks per pool. It completed both2704-prediction
HF checks and resident feedback, retaining1,864 traces with no request releases.
Its overall verdict remains **failed** because final full-state hashes differ;
the confirmed scalar-seed harness error and corrected reruns are recorded in
`batch_hf_seed_contract.md`. A failed numerical/state gate does not erase actual
allocator observations, but those observations do not convert the run to a pass.

The corrected `batch32_hf_factored_v2.json` passes in1296.826s (process exit0),
with all48 layers and `acceptance_eligible=true`. Each baseline/migration run
scores2704 correlated predictions:2645 top1,2704 top5 and2704 top100. Same-history
logits/tokens/final request state are exact, final and dormant per-field mismatch
maps are empty, and all-inactive/noop and resident feedback gates pass. Explicit
request seed lists fix the harness while preserving production seed semantics.
All three allocator snapshots and the2,194,677,760-byte bounded model capture
growth are exactly equal to v1. Thus the current memory figures now also have
a successful full48 correctness run; the failed v1 artifact remains immutable.
The v2 test does not adjudicate the separate frozen cross-active-width PCC gate;
`batch_geometry_disposition.md` records the owner's scoped HF-matched limitation
for independent review.

The current B1 accuracy/resume artifact passes and constructs C262144 private
state, retaining8 traces (six prefill, model, sampling). It runs short accuracy
and continuation workloads, **not**262144-token prefill. Both artifacts bind
`tt/generator.py` SHA `f539343185c824cbc73b820c241b03d9d953fc73e0808fa421be0e21db4abc63`
and `tt/model.py` SHA `bc46bb7e0428947cb29d7fc312b3fe6c3fd147eb88c4e6fa847f13755c10352a`.

The once-only `acceptance_max_context_final_v1.json` and
`max_context_attempt.json` both record PASS: public nonaligned262143-token
prefill, two sampled outputs and one decode consume262144 context. Counters
record8196 prefill replays, one model replay and two sampling replays; all12 QSA
layers on all4 ranks reach262144. Final active-row logits are finite on all4
ranks, checked with scalar output only. This is execution/capacity evidence,
not text-quality evaluation. The log records MAX_CONTEXT_DONE,
ACCEPTANCE_DRIVER_DONE and clean UMD close. The outer monitor was interrupted
and retains START only; **no numeric process exit is claimed**.

The max run's post-finiteness `memory_final` allocates8192 bytes/bank more than
its earlier `max_context.memory_after` (and the accuracy final). Current B1
budget translations below use that later, more conservative snapshot, a65536-byte
per-rank difference. Loaded/prepared values are identical to current accuracy.

These points measure retained allocator occupancy, not continuously sampled
high-water marks. A successful phase proves its actual transient allocations
fit at that time, but their maximum size is not derivable from these snapshots.
Neither a post-phase free field nor capture net growth should be labeled an
instrumented transient peak. The B32 runtime arithmetic below applies identically
to current passingv2 and historical failedv1.

## B32/C512: measured runtime versus forecast

The pure planner currently lists **30,575,773,184 bytes/rank before pool**,
including33 private non-KV caches (32 requests plus scratch),32 compact
checkpoints, one shared prefill output and17,236,992 bytes of factored workspace.
The actual deferred-loaded snapshot is30,572,439,552: the conservative inventory
exceeds it by3,333,632. Do not adjust source constants from this net difference;
the inventory intentionally includes some later/prepared allocations.

The measured512-block full-stack pool costs201,326,592. Final retained allocated
memory is33,779,269,120. Therefore:

- Net non-pool growth from the actual loaded baseline is
  `33,779,269,120 - 30,572,439,552 - 201,326,592 = 3,005,502,976` bytes/rank.
- Of that, measured bounded model capture growth is **2,194,677,760**, exactly
  matching the selected per-slot attention/MoE-bucket/terminal forecast.
- Prepared-to-final growth is **765,984,768**, covering the six prefill traces,
  sampling capture and retained changes; it is not isolated command-buffer size.
- Remaining loaded-to-pre-capture non-pool growth is **44,840,448**.
- The selected runtime reserve is **3,229,294,592**. Against measured-baseline
  net runtime it retains **223,791,616**. Against the conservative listed tensor
  inventory it retains **227,125,248**. These use different baselines and must
  not be mixed or subtracted twice.

Thus the original forecast is supported for this observed C512 workload;
its long-context B>1 extrapolation remains unverified. The forecast includes
traces and non-trace allowance: subtracting the actual trace bytes again from
its residual would double-charge them.

## Worker-expanded pool accounting

Pinned plugin commit `042a94863ad27d88207bcf71abdf67122c6f4c27`,
`src/vllm_tt_plugin/worker.py::get_num_available_blocks_tt`, adds
`max(block_size, output_tokens_per_step) * per_lane_batch` tokens, then rounds
up to block_size. Here block_size32, one output token, TP4/DP1. This calculation
assumes the ordinary12-QSA full-attention specification and no enabled
sliding-window/adaptive-output adjustment. Any future adapter setting that
activates a hybrid admission-floor or different grouping must be reconciled
again; no adapter/server execution is claimed here.

| Quantity | B32/C512 | B1/C262144 |
| --- | ---: | ---: |
| API return, tokens (includes null block) | 16,416 | 262,176 |
| Plugin-added output tokens | 1,024 | 32 |
| Resulting physical blocks per pool | 545 | 8,194 |
| Actually measured blocks per pool | 512 | 8,192 |
| Extra full-stack blocks | 33 | 2 |
| Extra bytes/rank | 12,976,128 | 786,432 |
| Extra bytes/bank | 1,622,016 | 98,304 |
| Full expanded pool bytes/rank | 214,302,720 | 3,222,011,904 |
| Expanded individual K or V tensor bytes/bank | 1,116,160 | 16,781,312 |
| Measured final free bytes/rank | 399,461,888 | 1,385,718,784 |
| Calculated final free after payload growth | **386,485,760** | **1,384,932,352** |
| Same calculated value per bank | 48,310,720 | 173,116,544 |
| Measured final largest contiguous bytes/bank | 46,266,496 | 165,250,368 |
| Largest minus total payload growth, conditional unchanged-layout arithmetic | 44,644,480 | 165,152,064 |

The last row is a useful conservative *charge to the observed extent*, not an
actual post-resize free-block observation or proof that allocation order leaves
that exact hole. At B32 setup, all24 expanded tensors together consume only
26,787,840 bytes/bank versus450,785,728 observed largest free before pool. At B1,
borrowed/private tensors must not coexist with a duplicate full pool: the
production allocator frees owned private K/V before replacement or uses
`defer_kv_cache=True`. The full private allocation already fit; expanded
individual tensors grow by only4,096 bytes/bank each (24 tensors). The extra
aggregate98,304 bytes/bank is far below the observed final contiguous extent,
but the actual shared replacement allocation order remains a different setting.

A more conservative bound keeps the entire preselected runtime reserve instead
of replacing it with measured net growth:

| Planner calculation, bytes/rank | B32/C512 | B1/C262144 |
| --- | ---: | ---: |
| Listed non-pool resident | 30,575,773,184 | 28,610,184,192 |
| Expanded pool | 214,302,720 | 3,222,011,904 |
| Selected runtime reserve | 3,229,294,592 | 2,147,483,648 |
| Total including full reserve | 34,019,370,496 | 33,979,679,744 |
| Residual below allocatable limit | **159,360,512** | **199,051,264** |

B1 listed resident including measured8192-block private KV is31,831,409,664;
its actual loaded baseline is31,828,125,184 (3,284,480 lower). Actual loaded-to-
final growth is964,887,040, leaving **1,182,596,608** of the2GiB reserve versus
that measured baseline. Using the listed baseline gives961,602,560 runtime and
**1,185,881,088** reserve remainder. The final larger measured free is consistent
with the still-conservative199,051,264 planner residual after expanded KV; those
are not competing measurements.

## Acceptance implications

Actual full48 B1 maximum-context execution and full48 B32/C512 acceptance now
pass. The measured allocations and source arithmetic support retaining262144;
they do not establish a larger physical maximum or justify any reduction.
No failure bracket or new tuning/capacity sweep was needed. The expanded8194/545
pool settings remain explicitly calculated budgets with measured runtime
corroboration, not separately executed configurations. An additional run merely
to repeat this arithmetic is unnecessary.

`doc/context_contract.json::full_model` records current maximum-context final
memory, B1 accuracy/resume/clean performance and passing B32v2 evidence, plus
selected head/checkpoint/workspace/factored inventory and worker expansion.
Historical nested records retain prior full_accuracy_v2, failed B32v1 and the
pre-max B1 measurement/profile. `serving_contract.json` has42 supported and21
unsupported rows, none missing. Acceptance gates pass; fresh independent stage
review and local commit remain. No capacity implementation source changed.

Historical input SHA256 values at initial preparation time (the context contract
has since been updated as described above):

```json
{
  "doc/full_model/batch32_hf_factored_v1.json": "eebede318006a16fc7d2d762c9e671f641ba23bc9b3afc61a36c7c583c43240b",
  "doc/full_model/acceptance_accuracy_resume_final_v1.json": "0e71968dc38b99a6187d32068f363dafbf10f3c33e10a77575f41a96053aa57f",
  "tt/cache_budget.py": "28b2f5cdfe258281e5910eb08444f0721517b240ebdb38ccc4c1ea8e59f70a1f",
  "doc/full_model/capacity_arithmetic.py": "0ea3a279a72824af0d07f17d76d31ce9a14317a04b42600d441a3a7a94278168",
  "doc/full_model/capacity_plan.json": "ca7c1717c1416df8b1aeac6cd95b3566ee0d3d86173e7b52cf276afe8994189f",
  "doc/context_contract.json": "f47821ae2f7ca59b150060ecbeafb146c89538504155059e6c773b074678d354"
}
```

Current passing B32 artifact SHA256: `da78875d7a768a13e169afc9bf0f24740d2fe30b0e127679a58af5881dc4e370`.
