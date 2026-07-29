# AutoDebug: North Mini optimized-decoder review 3 (resolved)

## Scope and baseline

This is an inspection-only, no-hardware investigation of every required-work
finding in
`models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/STAGE_REVIEW_3.md`.
The inspected implementation baseline is
`c7e024e8faa54118c71771dc1629d8b75abeb022`; its parent is
`0085f30d237882107832133b682f3b4bda056efa`. The only change between those
commits is three optimization-checkpoint worklog lines, so all implementation
conclusions apply equally to both.

No device experiment was run. “Source fact” below means directly established
from code or preserved artifacts. “Prediction” means the expected result of a
future focused device experiment and must not be promoted to evidence until
that experiment runs.

## Headline verdict

The stage cannot pass yet, but several review claims need tightening:

1. **Dense-expert BFP4 selection:** the core finding is verified. The authentic
   layer-1/layer-4 real-weight matrix passes the required PCC in every tested
   mode, and the synthetic failure is not a valid veto under OPT-012. The
   existing performance comparison is useful screening evidence, not
   post-repair/current-code authentic timing. The smallest likely repair is a
   **mixed policy**: dense expert weights BFP4/LoFi, sparse expert weights still
   BFP8/LoFi.
2. **Batch-1 prefill topology:** verified. The selected default sends
   batch-1, sequence-128 prefill through all 128 experts, despite the
   documentation saying batch 1 uses active experts. A phase-aware active
   prefill path is required; merely raising the global dense threshold would
   regress the deliberately retained batch-32 decode exception.
3. **DRAM-sharded dense experts:** the reviewer is right that the current
   artifact does not reject the compatible family. It times disconnected
   synthetic projections and serially extrapolates 16 groups. However, the
   reviewer should not imply that an all-128-expert DRAM-sharded form is
   untested merely by accident: the specialized factory’s L1 input-shard
   requirement gives that particular geometry a concrete capacity blocker.
   The untested space is legal grouped geometry, fusion, packed projections,
   and on-chip carry/reduction.
4. **Final profiling:** verified in substance. The selected profiles predate
   the correctness repairs and final policy changes, and no post-repair profile
   set exists. The claim that every old profile exercised a later-proven wrong
   path is too broad: aligned batch-1 prefill did not exercise the repaired
   non-aligned multi-user path. The artifacts are nevertheless stale and
   incomplete, and their selected directories do not contain the “raw CSV”
   claimed by the documentation.
5. **Optimized 500k prefill:** verified. The optimized implementation advertises
   the inherited 500,000-token contract, but preserved optimized evidence
   covers only batch-32 decode at context 500,000, not optimized prefill.
   Capacity must be tested after the precision and active-prefill changes,
   because both change resident weights and long-context execution topology.

The disciplined repair order is:

1. select and validate the mixed dense-BFP4 policy;
2. repair selected-default batch-1 prefill topology and its long-sequence
   chunking;
3. exhaust the compatible grouped DRAM-sharded dense-expert family;
4. rerun the full correctness, watcher, and wall-latency gates on final code;
5. collect final signposted profiles and optimized 500k/499999 capacity
   evidence on that same final policy.

Do not batch these repairs into one patch. Each item below has a focused
experiment that can prove or refute its hypothesis in isolation.

## Finding 1: dense-expert BFP4/LoFi is not selected

### Source facts

`OptimizationConfig` in
`models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py`
defaults all four expert dtype fields to BFP8:

- `expert_gate_up_dtype` and `expert_down_dtype` control the sparse/active path;
- `dense_expert_gate_up_dtype` and `dense_expert_down_dtype` control the dense
  all-expert path.

Both expert paths already use LoFi fidelity. The loader can materialize a
separate dense BFP4 copy while retaining the sparse BFP8 copy; the dtype split
is therefore already structurally supported.

The real-weight precision matrix in
`tests/test_optimized_decoder.py::test_optimized_real_weight_moe_precision_matrix`
propagates authentic hidden states through prior Hugging Face layers and tests
layers 1 and 4, batch 1 and 32, prefill and cache-consuming decode. It forces
the dense path and changes only the dense expert dtypes. Preserved BFP4 results
are:

| Layer/mode | BFP4 PCC |
|---|---:|
| layer 1, prefill, batch 1/32 | 0.9994277 |
| layer 1, decode, batch 1/32 | 0.9979947 |
| layer 4, prefill, batch 1/32 | 0.9999415 |
| layer 4, decode, batch 1/32 | 0.9972343 |

All exceed the required 0.995 full-output threshold. The corresponding BFP8
values are approximately 0.9991–0.99999.

The synthetic selected-default test fails for BFP4 at approximately 0.9816 PCC
for batch-32 decode and batch-2/sequence-33 prefill. That test uses
recorded-stat random expert weights, engineered router behavior, and synthetic
activations. OPT-012 explicitly prevents synthetic failures from vetoing a
configuration that passes authentic representative correctness.

The existing candidate JSONs report these warmed device times:

| Mode | dense BFP4 | dense BFP8 |
|---|---:|---:|
| batch-32 decode | 2.2145 ms | 3.3911 ms |
| batch-1 prefill | 3.7623 ms | 4.7528 ms |
| batch-32 prefill | 139.965 ms | 141.142 ms |
| batch-1 decode | 0.7902 ms | 0.7912 ms |

These JSONs were created at older commit `03b1b0078f1`, use synthetic state,
and predate the current separate dense/sparse dtype schema. They establish a
strong shape-level performance hypothesis, not final timing for
`c7e024e8faa`. Batch-1 decode is sparse under the current default, so its
near-tie says nothing about the proposed dense-only dtype change. After the
batch-1 prefill topology repair, that mode will also cease to be evidence for
dense BFP4.

The exact PCC logs appear to exist only as ignored/untracked runtime logs. The
tracked JUnit XML records eight passing cases but not the dtype environment or
the exact values. This is a provenance gap for a clean checkout, even though
the recorded numerical result itself is clear in the present workspace.

### Challenge to the review

The reviewer is correct that BFP8 is retained solely because of an inadmissible
synthetic veto. It is too strong, however, to characterize the older latency
artifact as current authentic timing. It should be treated as the screening
result that justifies a small current-code rerun.

The right selection is not “make every expert tensor BFP4.” Sparse/active
experts have no authentic BFP4 acceptance evidence in this matrix, and their
performance is a different topology. The supported minimal policy change is
dense BFP4/LoFi plus sparse BFP8/LoFi.

### Focused experiment and predicted result

Run one-variable A/Bs on the current commit:

1. rerun the authentic real-weight matrix with dense BFP8;
2. rerun it with only
   `--dense-expert-gate-up-dtype bfp4
   --dense-expert-down-dtype bfp4`;
3. compare warmed current-code latency for layer 1 batch-32 decode and
   batch-32 prefill, using 3 warmups and 20 measured iterations;
4. retain batch-1 prefill only as a temporary diagnostic until the topology
   repair lands.

Prediction: the full-output PCC values remain above 0.995 and BFP4 retains a
material batch-32 decode win. If the current-code win disappears, the
precision selection is refuted on performance rather than by the synthetic
PCC test.

Because the residual can mask component error, a direct real-weight
MoE-only PCC comparison is a useful secondary diagnostic, but the decoder
full-output matrix remains the contractual gate.

### Smallest model-local repair

Change only the two `dense_expert_*_dtype` defaults to BFP4. Keep the two
`expert_*_dtype` defaults at BFP8. Preserve LoFi.

The selected synthetic correctness test must then be made semantically honest:

- keep an explicit BFP8 synthetic reference test for the random-stress PCC
  behavior;
- use the authentic full-output matrix as the BFP4 acceptance gate;
- retain a BFP4 synthetic stress test for finiteness, branch execution, and
  runtime legality, without treating its artificial PCC as the model
  acceptance metric.

Do not lower the synthetic PCC threshold and call that a pass, and do not hide
the test behind an unconditional xfail.

### Acceptance

- current-code real-weight layer-1/layer-4 matrix passes all eight relevant
  cases at PCC >= 0.995;
- current-code warmed timing shows BFP4 is faster for at least the selected
  dense modes and no selected-mode functional regression occurs;
- final policy artifacts explicitly record both sparse and dense dtype fields;
- tracked evidence contains the command, commit, environment, and exact PCC,
  not only anonymous passing XML cases.

## Finding 2: selected-default batch-1 prefill is all-expert

### Source facts

`_sparse_moe` selects `_dense_expert_moe_chunk` whenever
`batch * sequence_length >= dense_expert_batch_threshold`. The threshold
defaults to 32. Consequently:

- batch-1 decode (`1 * 1`) uses active experts;
- batch-1 sequence-128 prefill (`1 * 128`) uses all 128 experts;
- batch-32 decode (`32 * 1`) uses all 128 experts.

The dense chunk repeats every token over all 128 experts, runs three ordinary
interleaved matmuls, applies router weights, and sums experts. The selected
batch-1 prefill profile contains the expected `repeat` plus three dense
batch-128 expert matmuls. This directly contradicts the README’s unqualified
statement that “Batch 1 uses active experts.”

The existing active-expert branch proof is not selected-default evidence. It
forces `dense_expert_batch_threshold=1<<30`, forces a DRAM sparse
intermediate, uses synthetic state, and has no timing result. The performance
harness exposes dtype overrides but not the dense threshold, so a mechanical
CLI/config override is needed for the first A/B.

The active loop currently uses chunks of four tokens and retains each output
tensor in a Python list before concatenation. At 500,000 tokens that implies
125,000 chunks and output objects. This is a direct scalability defect in the
candidate topology even before device memory is measured.

### Challenge to the review

The reviewer’s branch diagnosis is correct. The repair must be more precise
than “force active experts for all batch-1 work” or “raise the threshold.”
Batch-32 decode is intentionally dense because the previously tested active
alternatives regressed, while prefill and decode can both have sequence length
one in different batching scenarios. The topology choice needs explicit phase
information.

Prior AutoFix measurements of batch-32 active variants—approximately 21 ms
dynamic, 17.83 ms static-nnz, 19.58 ms packed, and 1.64 ms for a fused compute
fragment that did not retain the required outputs—do not predict batch-1
prefill performance. They justify retaining a batch-32 decode exception, not
the current batch-1 prefill behavior.

### Focused experiment and predicted result

First add only a test/harness override for the threshold or phase policy. Then
compare on current code, layer 1, batch 1, sequence 128:

1. selected dense all-expert BFP8 baseline;
2. forced active BFP8 with chunk size 4 and the existing placement;
3. if legal but slower, active BFP8 with a DRAM intermediate and chunk sizes
   32 and 128.

The functional baseline recorded for this mode is 14.908 ms; the active
candidate must not regress it. Prediction: chunk 4 will be launch/object
overhead dominated; a larger prefill-specific chunk is more likely to be
viable. If neither 32 nor 128 is legal or competitive, test the exact blocker
before expanding the search to 256/512/1024.

Only after a viable layer-1 sequence-128 candidate exists, run:

- sequence 33, to cover the non-aligned prefill path;
- layers 1 and 4 with propagated real hidden states;
- a batch-1 active-path branch assertion without any override.

If chunk-size tuning is insufficient, the next single hypotheses are, in
order: static exact-nnz masks; packed sparse gate/up; fused SiLU-multiply; then
a compact/grouped active topology. Each must retain required outputs before
its timing is accepted.

### Smallest model-local repair

Pass explicit phase information into the MoE policy or split the prefill and
decode dispatch methods. Add a prefill-specific active chunk size and
intermediate-placement setting. Select active experts by default for batch-1
prefill while preserving the batch-32 decode dense exception.

Do not globally set `dense_expert_batch_threshold` to a huge value. That would
silently select a known-regressed batch-32 decode path.

For long context, replace the four-token/list-of-all-results implementation
with the largest verified legal streaming chunk, or with a hierarchical output
assembly that does not retain 125,000 tensors. This long-sequence behavior is
part of the same topology repair, not an optional later optimization.

Batch-32 prefill remains a separate potential all-expert optimization gap. It
is outside the reviewer’s explicit batch-1 finding and must not be silently
claimed fixed by this repair.

### Acceptance

- default batch-1 prefill reaches the active-expert branch at sequences 33 and
  128 without a threshold override;
- real propagated layer-1/layer-4 output PCC remains >= 0.995;
- sequence-128 warmed latency is no worse than the functional 14.908 ms gate;
- the selected chunking is demonstrably capable of streaming 499999/500000
  tokens without a list proportional to the number of four-token chunks;
- batch-32 decode remains on its explicitly justified dense exception;
- documentation describes prefill and decode policy separately and accurately.

## Finding 3: the compatible DRAM-sharded dense-expert family was not exhausted

### Source facts

The preserved experiment uses
`MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig` with eight
optimal bank workers. It constructs synthetic random activations and weights,
uses a batch/group size of eight, and times:

- gate/up-shaped projections with K=2048 and N=768, `in0_block_w=4`;
- a down-shaped projection with K=768 and N=2048, `in0_block_w=2`.

The “up” time reuses the gate output rather than performing a distinct
gate/up→SiLU×up chain. The down projection consumes an unrelated random input.
After down, each group is converted to DRAM interleaved, reduced, and added.
The result is multiplied by 16 to estimate all 128 experts. Recorded estimates
are 5.184765 ms for BFP8 and 4.170053 ms for BFP4.

This is neither a full MoE chain nor a lower bound:

- it omits real routing and real propagated activations;
- it does not carry the gate/up result into the down projection;
- it does not test fused activation or packed gate/up;
- it performs removable per-group layout traffic;
- it assumes strict serial execution of 16 identical eight-expert groups.

Projection PCC is approximately 0.99986 for BFP8 and 0.99384 for BFP4 under a
lowered 0.99 projection threshold. Those numbers do not substitute for the
required real full-decoder PCC.

The specialized factory does impose concrete constraints:

- input and output storage-core sets must match the optimal DRAM-bank workers;
- A is height-sharded in L1, weights are height-sharded in DRAM, and output is
  height-sharded in L1;
- batches per core are `ceil(B / 8)`;
- the program supports fused activation;
- K must be divisible by the chosen `in0_block_w`.

An all-128-expert tensor therefore has a real L1 blocker in the factory’s
required input shard. At BF16, 16 experts per bank require roughly
`16 * 32 * 2048 * 2 = 2,097,152` bytes for input A alone, exceeding the
reported approximately 1,461,504 available bytes before other circular
buffers. That refutes an all-128-at-once instance, not the family.

Legal grouped geometries remain untested. For group size G in
`{8,16,32,64}`, each bank carries `G/8` experts. The input-shard sizes are
approximately 128 KiB, 256 KiB, 512 KiB, and 1 MiB respectively. Group 64 may
be tight once other buffers are included; groups 16 and 32 are credible and
would reduce launches and reductions. The valid gate/up K-tile divisors are
`{1,2,4,8,16,32,64}` and down-projection divisors are
`{1,2,3,4,6,8,12,24}`.

### Challenge to the review

The reviewer is correct that the serial synthetic estimate cannot reject the
family. It is also important not to swing to the opposite unsupported claim
that the dense family obviously fits as one 128-expert operation. Source-level
capacity analysis refutes that geometry. The actionable missing work is a
bounded search of grouped legal geometries and end-to-end carry/fusion, with
exact blockers recorded.

### Focused experiment ladder and predictions

Use the final dense BFP4 policy as the baseline. This search is for the
batch-32 decode exception; do not reuse the same topology for batch-32 prefill
without a separate two-dimensional prefill evaluation.

1. **Geometry legality only.** For G=8,16,32,64, allocate real-shaped input and
   real BFP4 height-sharded DRAM weights. Sweep only the legal K-block divisors
   above. Record exact allocation/program-construction errors and L1 use.
   Prediction: G=16 and at least one G=32 configuration are legal; G=64 may
   fail capacity.
2. **Real projection A/B.** On every legal maximum-G geometry, compare split
   gate/up and packed gate-up (N=1536). Do not extrapolate a single group;
   execute the required `128/G` groups.
3. **Full local chain.** Feed the real gate and up results into
   `SiLU(gate) * up`, then into the matching down weights. Compare factory
   fused activation against multiply-side activation. Retain all required
   outputs.
4. **Carry and reduction.** Keep gate/up, multiply, down, and routing multiply
   height-sharded in L1. First try a direct sharded fast reduction. If it is
   unsupported or wrong, convert only immediately before the cross-bank
   reduction, preferably to L1 interleaved, keep a group accumulator in L1,
   and move to the consumer’s layout once at the boundary. Record the exact
   unsupported operation if a conversion is unavoidable.
5. **Authentic integration.** Run the complete router plus all `128/G` groups
   on propagated real layer-1 and layer-4 inputs. Validate full decoder PCC,
   trace capture/replay for batch-32 decode, wall latency, and profiler
   topology.

Prediction: larger legal groups plus on-chip carry will materially beat the
4.17 ms serial BFP4 estimate, because they remove per-eight-expert conversions
and reduce launches. This does not guarantee they beat the current dense BFP4
baseline; only the full-chain result decides selection.

### Smallest model-local repair

If a compatible candidate wins, isolate it behind an explicit
batch-32-decode policy and reuse the existing real weights in grouped
height-sharded DRAM form. Keep prefill dispatch independent. Select packed
gate/up or split projections solely from measured full-chain results.

If no candidate wins, retain the current dense path, but the rejection artifact
must include:

- all legal tested group sizes and block choices;
- exact capacity or unsupported-operation blockers;
- real full-chain PCC for every surviving candidate;
- end-to-end wall/device timing against the final BFP4 dense baseline;
- a topology trace proving required outputs and layout conversions were not
  optimized away.

A projection-only serial extrapolation is not sufficient rejection evidence.

### Acceptance

- all feasible G=8/16/32/64 candidates are either measured end to end or have
  exact source/runtime blockers;
- the selected/rejected result uses real weights and propagated layer-1/layer-4
  hidden states;
- full-output PCC is >= 0.995 and trace replay is correct;
- timing includes routing, gate, up, SiLU/multiply, down, routing multiply,
  reduction, and necessary layout transitions;
- the comparison baseline is current final dense BFP4, not the older BFP8
  3.39 ms number.

## Finding 4: selected profiles predate correctness repairs

### Source facts

Every file under the selected Tracy profile tree is timestamped before
`0085f30d237`, which fixed the non-aligned multi-user prefill corruption and
other correctness gaps. The selected batch-32 decode profile is definitely
invalid as final evidence because the old rectangular RoPE path corrupted
lanes 8–31. The old batch-1 and aligned sequence-128 prefill profiles did not
enter that particular changed branch, so it is inaccurate to say that every
row is known to describe an incorrect execution.

All selected profiles are still stale because:

- they predate the repairs and the pending BFP4/active-prefill policy changes;
- their dense rows are BFP8;
- batch-1 prefill visibly contains the now-rejected all-expert topology;
- there is no post-repair selected profile set;
- each selected directory contains only `filtered.csv`, `summary.csv`, and
  `summary.png`, not the raw operation CSV claimed by the README/review.

The post-review wall-latency JSONs are newer and useful, but they are not
profiles and still use the current BFP8/all-expert policy.

### Focused experiment

Profiling must be last, after all three topology/policy decisions are frozen.
Collect signposted profiles for:

- layers 0, 1, and 4;
- prefill and decode;
- batches 1 and 32.

That is 12 final profiles. Add one targeted layer-1, batch-1, sequence-33
prefill profile to demonstrate the non-aligned repaired path and selected
active experts. Decode must profile one trace replay, not capture/compile.
Prefill must be warmed. Run advice-enabled filtering, but preserve the raw
operations CSV as well as the filtered and human-readable summaries.

In the same artifact directory record the exact command, commit, final policy
JSON, environment, and signpost interpretation. Verify from the rows—not from
configuration intent—that:

- dense expert matmuls use BFP4 where selected;
- batch-1 prefill contains active-expert operations and no 128-expert repeat;
- batch-32 decode uses the final dense or compatible DRAM-sharded choice;
- no unexpected DRAM layout round trips remain;
- device time, wall time, and roofline conclusions refer to the same code and
  run family.

Watcher validation must be a separate run from profiling.

### Acceptance

- all 12 required current-commit final-policy profiles plus the sequence-33
  proof exist;
- raw, filtered, and summarized artifacts are present and reproducible;
- profile topology agrees with selected defaults;
- final normal and watcher correctness suites pass before profile conclusions
  are promoted;
- wall and device timing are reconciled without mixing artifacts from older
  policies.

## Finding 5: optimized prefill lacks 500k evidence

### Source facts

`doc/context_contract.json` advertises a 500,000-token context. Functional
evidence covers aligned 500000 prefill and near-limit 499999 prefill at layers
0, 1, and 4, plus decode. The only optimized capacity artifact is
`context500000_decode_b32.json`, layer 0, with a 32.768 GB cache and finite
decode. It does not test optimized prefill.

`tests/optimized_decoder_capacity.py` already supports prefill via the
functional capacity probe. The optimized class inherits the 500,000 maximum
and validates it, so the absence is an evidence gap, not a missing advertised
feature.

The earlier large-M nonfinite bug was in the multi-user token-packed QKV/O path
and was repaired with 512-row chunks. Batch-1 aligned 500000 and near-limit
499999 use the standard batch-1 attention path, so that historical failure
does not prove these cases fail. It does establish that large-M behavior must
be tested rather than inferred.

The capacity harness’s synthetic expert weights are zero by default. Its
attention weights remain nonzero, so it can catch attention large-M
nonfiniteness and allocation failures. Zero experts still allocate and execute
the topology but are not expert numerical-stress evidence.

The pending changes make final-policy sequencing important:

- retaining sparse BFP8 weights while adding separate dense BFP4 copies adds
  roughly 302 MB of expert storage;
- the present active-prefill chunk-4 loop would create 125,000 chunks at
  500,000 tokens.

Therefore a capacity pass on the current pre-repair default would not validate
the final implementation.

### Focused experiment and predicted result

After final dtype and prefill chunking are selected, run the optimized capacity
harness as single-pass finite/allocation evidence:

1. batch 1, layer 0, context 500000 aligned prefill;
2. batch 1, layers 0, 1, and 4, context 499999 near-limit prefill;
3. rerun the existing 500000 batch-32 decode smoke on the final weight policy.

A warmed benchmark is unnecessary for this capacity gate; the artifact must
record exact allocation, peak-memory information available from the harness,
output finiteness, shape, command, commit, and policy.

Prediction: the present four-token active implementation is impractical or
fails before meaningful completion. The repaired streaming chunk should pass
if attention allocations fit as functional evidence indicates. If it does not,
record whether the first blocker is host object growth, L1/DRAM allocation,
program legality, or nonfinite output; do not relabel decode-only evidence as
prefill support.

Layer 1 is important because it materializes the worst-case sparse and dense
expert policy. If resources permit, add batch-32 layer-1 capacity after the
required batch-1 matrix; the old layer-0 decode artifact does not cover
resident expert-weight pressure.

### Smallest model-local repair

The expected implementation prerequisite is the phase-specific streaming
active-expert chunk/assembly described in Finding 2. Do not add a
capacity-only bypass that silently switches 500k prefill back to all-expert
dense execution. If the maximum contract remains infeasible with the selected
topology, either produce a precise blocker and revise the advertised contract
through the normal stage process, or continue the isolated chunk/streaming
repair.

Update the optimized portion of the context contract only after the new
machine-readable artifacts exist.

### Acceptance

- optimized batch-1 prefill is finite and completes at aligned 500000;
- optimized near-limit 499999 prefill passes at layers 0, 1, and 4;
- evidence uses final defaults, including both sparse and dense weight copies;
- the profile/branch proof confirms active experts rather than a hidden dense
  capacity fallback;
- final context-contract entries point to tracked, reproducible artifacts.

## Minimal verification matrix after repairs

Run this only after each focused hypothesis has passed on its own:

| Gate | Required evidence |
|---|---|
| Precision | real-weight layers 1/4 × batch 1/32 × prefill/decode; dense BFP4 vs BFP8; PCC and current timing |
| Active prefill | selected-default batch 1 sequence 33/128 branch proof, real PCC, warmed latency, long-sequence streaming proof |
| Dense decode family | full real grouped DRAM-sharded chain for every surviving legal geometry, or exact blockers; trace replay |
| Correctness | full optimized normal suite and a separate watcher suite on final defaults |
| Performance | 3 warmups/20 runs, layers 0/1/4 × modes × batches; compare to the functional gates and current final baseline |
| Profiling | 12 signposted final profiles plus batch-1 sequence-33; raw and filtered CSVs; policy/commit provenance |
| Capacity | optimized 500000 aligned prefill and 499999 layers 0/1/4 near-limit prefill; final-policy decode smoke |

The stage is ready for another independent review only when all of these
artifacts describe the same commit and default policy. A clean review should
not need to infer branch choice from prose, reconstruct dtypes from filenames,
or combine old synthetic timings with new correctness logs.
