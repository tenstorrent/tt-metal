# AutoFix — stage 7 speed floor repaired

The required all40 batch1 128-in/128-out useful-byte floor is 21.631299 ms/token.
The starting selected path measured25.081181 ms. Its persistent split-sampling
trace loop has no per-token host boundary, and sampler-only latency is0.487795ms.
The historical layer estimate24.993502ms locates the repair primarily inside
the decoder. It is not an additive decomposition of the fresh workload.

[AutoDebug](AUTODEBUG.md) was checked against current Python dispatch and native
collective/reduction sources. Selected arithmetic, dtypes, KV and residual
contracts are immutable constraints. No rejected native numerical policy is
eligible for promotion, and no denominator inflation can repair this gate.

The independent hypotheses, in priority order, are:

1. Remove direct-AR logical padding/cropping while retaining physical shards,
   persistent scratch, semaphore and rank reduction order; remove only the
   output recurrence's inverse permutation pair. Exact component and integrated
   state/consumer equality decide acceptance.
2. Narrow exact projection activation reads and eliminate unused internal DST
   prefix stores. Keep products, K512 grouping, every addition, carry and reset;
   test NOP-preserving removal before instruction-spacing changes. Exact FP32
   partials are required, not just final BF16 PCC.
3. Contract native recurrent products directly, retaining ordered16-then8 FP32
   additions, to avoid fourfold padded HC materialization. Compare both output
   layouts and evolving trace state before any production replacement.
4. Retain the measured DRAM-sharded terminal-head win only after full-depth
   allocation and generator acceptance. It cannot close the floor alone.

Each hypothesis has isolated source/probes. Hardware runs are serialized by the
parent. CPU analyses and code construction may proceed concurrently. At this initial repair snapshot, structural, narrow-read and direct-sum repairs had component
and integrated exactness evidence; production integration and full-model gates
remain pending. Prefix-store elision was exact but did not improve latency and
was rejected. Coefficient caching initially regressed; removing unused padding
initialization passes a qNaN-padding control and improves the two large K512
projections, while router/K1024 caching remains rejected. A further isolated
product-plus-contraction candidate preserves FP32 multiplication before the
same ordered sums and targets product materialization, not numerical policy.
Stage acceptance and AutoFix success remain pending.

## Complete-path measurement after exact state-slab fusion

`candidate2_performance.json` passes the complete B1 128-in/128-out useful-byte
floor at21.424391835ms/token (46.67577t/s/u), achieved fraction0.100965754.
The denominator remains1,107,522,496 useful bytes/rank at512GB/s; no padding,
rereads, CCL or transient materialization was added to it. Original baseline
was25.081180984ms; candidate1 was22.272769457ms and failed the floor.

The final step retains each32-column recurrent slab on one of32 workers through
decay, ordered K contraction, delta, state update and ordered Q contraction.
FP32 product/add order and BF16 consumption boundaries remain unchanged. Four
128-step rounded/unrounded persistent/temporary component controls pass exact
all-rank state/output comparison with Watcher and allocation tracking; integrated
layer0/layer34 and nonaligned B3 controls also pass. Concrete production kernels
and descriptors match the qualified isolated probe. The selected policy and
rejection ledger are unchanged.

The full loop has127 nonblocking model+sampling replay pairs, stable program
cache8440, and zero resident refresh/readback/recapture. Teacher logits timing
is20.940451970ms separately; caller-observed token delivery21.493701142ms includes
readback. Fresh final-source accuracy, serving/context, qualitative, native
profile and independent review remain stage gates. A passing floor alone is
not stage completion.

### Final serving harness correction

The all40 command probe failed at a reduced-model expected-update constant: observed30, expected3. Inspection shows10 full-attention layers×3 changed slots; generator behavior is correct. The test now derives this count from actual cache groups and checks all saved page-table rows. No inference source changes. Original failure is final/serving_commands_all40.json; The v2 rerun then exposed a local test variable shadowing the state() helper; renaming it cache_slot fixes the harness without changing inference code. The fresh final/serving_commands_all40_v3.json passes all nine checkpoints, including30 changed and0 unchanged page-table updates. Reset/list/mesh recovery logs use serving_assert_* names. This is a test-scope repair, not an accuracy waiver or a reason to rerun unaffected completed gates.

### Frozen final confirmation

`final/performance.json` reproduces the repaired floor at21.416868244ms/token,
46.692167529t/s/u and10.10012225% useful bandwidth. Final all40 prefill and
teacher accuracy both reach top5=100%,top100=100%. All40 Watcher mixed replay,
B32 fixed slots, serving v3, correct greedy sampler, exact AIME generation and
six shared qualitative controls have finished with the bounded passes recorded
in `final/correctness_acceptance.json`. `final/prefill_boundaries.json` preserves
all six aligned/nonaligned public prompt lengths and resident feedback. The
performance root cause has been repaired. Native attribution and32769/context65536
smoke now pass; independent stage review remains a completion requirement.

## Final repair disposition before independent review

All required accuracy/runtime/performance/context gates are complete on frozen
production. The selected exact state-slab/scan/structural/head changes repair the
full-model speed floor without changing the useful-byte denominator or selected
precision/residual policies. The original failure and subsequent candidate1
failure remain above as chronological evidence. No AutoFix failure is asserted;
independent stage review and local checkpoints still govern stage completion.

## Accepted repair disposition

Independent round1 review returned **clean-pass**, with no required work
([report](stage_review_round1.md)). All required gates passed; the user cap
stops further optimization under Stop Rule3. Earlier pending entries above
record the investigation chronology. Local checkpoint details are in work_log.md.
