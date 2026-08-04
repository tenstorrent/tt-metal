# Plan for a Collective-Redundancy Static Analyzer for the DiT Team

> **This is the original design, kept as written.** It is not updated as work
> lands, so do not read it as a statement of what exists. Current status —
> which phases are done, what is next, and which of the 44 blockers are still
> open — lives in one place:
> [`DitStaticAnalyzerRoadmap.md` § "Where we are"](DitStaticAnalyzerRoadmap.md#where-we-are).
> In short, at the last update: the analyzer and the device-free dry-run front end
> are built and the v1 bar below is met; shape fidelity is the next phase, and
> on-device conformance is what stands between "the shim believes" and "this is
> true".

## Executive summary

Build a **forward-pass communication analyzer** that models device topology, tensor placement, tensor ownership, and per-op data requirements, then flags collectives that are **provably redundant**, **potentially reducible**, or **suspicious enough for human review**.

The tool should not start as a fully general compiler. It should start as a **restricted but high-confidence analyzer** for the subset of TTNN / DiT forward graphs that matter most to the team:

- inference-only forward passes
- tensor-parallel and sequence-sharded layouts already used in LTX / DiT pipelines
- collectives such as all-gather, reduce-scatter, all-reduce, and explicit reshard / replicate operations
- matmul-heavy subgraphs where communication dominates runtime

That scope is narrow enough to ship, but broad enough to catch the class of missed optimizations that cost the team months of kernel work.

## What the tool must answer

For every collective in a forward pass, the tool should answer:

1. **What data did each device have immediately before this collective?**
2. **What data does each downstream op actually require?**
3. **Is this collective necessary to satisfy those requirements?**
4. **If not fully redundant, can it be reduced?**
   - fewer participants
   - smaller gathered dimension
   - delayed materialization
   - fusion with a neighboring op
5. **What proof or rationale supports that conclusion?**

If the tool cannot prove redundancy, it should still explain why.

## Recommended product shape

Build this as an **offline static analysis pipeline** with three layers:

### 1\. Graph capture layer

Turns a model forward pass into an analyzable intermediate representation.

### 2\. Distributed state engine

Simulates tensor state across devices through compute ops and collectives.

### 3\. Redundancy checker

Compares required-vs-available data at each collective and emits findings.

A fourth optional layer can come later:

### 4\. Visualization and diff UI

Lets engineers inspect device states, shard maps, and “why this collective is redundant” traces.

## Core design principle

The tool should reason about **logical tensor regions**, not raw buffers.

Instead of saying “device 3 has tensor X,” the analyzer should say something like:

- device 0 owns rows `[0:1024)` of activation `A`
- device 1 owns rows `[1024:2048)` of `A`
- all devices hold a replicated copy of weight `W_qkv`
- device 2 has a stale but equivalent copy of gathered activation `A_full`

This is the right abstraction because redundant collectives are usually about **equivalent data availability**, not pointer identity.

## High-level architecture

### A. Intermediate representation (IR)

Define an IR for the forward pass with:

- **device mesh / topology**
- **ops**
- **tensor symbols**
- **sharding / replication annotations**
- **collective semantics**
- **data dependencies**

Each IR node should include:

- op type
- input tensors
- output tensors
- device participants
- mesh axis / communication group
- layout before and after
- semantic effect on tensor regions
- source location in model code if available

Example node categories:

- compute: matmul, elementwise, norm, attention pieces, reshape/view, concat, split
- communication: all-gather, reduce-scatter, all-reduce, broadcast, all-to-all, reshard
- metadata-only: transpose, view, alias, slice

### B. Device-state model

For every program point, track per-device state:

- which tensor symbols exist on the device
- what logical region of each tensor exists
- whether it is replicated, sharded, partial, reduced, or invalidated
- provenance: which earlier ops / collectives produced it
- equivalence classes: different tensor symbols known to contain the same logical data

A good minimum representation is:

`TensorState(device, tensor_id, region_set, layout, value_semantics, provenance)`

Where:

- `region_set` describes exact logical coverage
- `layout` captures shard/replicate/partial status
- `value_semantics` captures whether data is exact, partial-sum, transformed, or aliased
- `provenance` supports explanations and debugging

### C. Requirement engine

For each op, define a transfer function that answers:

- what data must exist on each participating device before the op
- what data exists after the op

This is the heart of the analyzer.

For example, a sequence-sharded AGMM may require:

- each device has its local shard of activation `A`
- each device receives enough remote activation regions to produce its assigned output region
- weights are already replicated or appropriately sharded

The analyzer should encode this as a **formal requirement rule**, not as ad hoc logic embedded in passes.

### D. Redundancy detection engine

At each collective, compare:

- **required data for all downstream consumers until the next state-changing boundary**
- against **data already present across participant devices before the collective**

Then classify the collective as:

- **provably necessary**
- **provably redundant**
- **partially redundant / reducible**
- **unknown due to missing semantics**

## Suggested analysis model

Use a **forward dataflow analysis** plus **backward demand analysis**.

### Forward pass: availability

Simulate what data each device has after every op.

### Backward pass: necessity

Starting from each op’s inputs, propagate backward what regions are truly needed.

A collective is redundant when the backward “needed set” is already satisfied by the forward “available set” without that collective.

This two-sided approach is much stronger than forward-only simulation because it catches cases where a gathered tensor exists but is never actually needed in full.

## Semantics the team must define explicitly

The tool will only be as good as its op semantics. Before coding, write a **semantics spec** for the operations that matter.

### Tier 1: must support first

- all-gather
- reduce-scatter
- all-reduce
- matmul / AGMM
- reshape / view / transpose
- split / concat
- layernorm / RMSNorm if they affect communication assumptions
- elementwise ops
- residual adds

### Tier 2: add next

- attention-specific ops
- KV-cache updates
- cross-device concat / split variants
- expert routing if MoE enters scope

Each op spec should define:

- participant devices
- required input regions per device
- produced output regions per device
- whether the op preserves equivalence
- whether the op changes sharding semantics

## How to model redundancy correctly

The most important rule: **redundancy is relative to downstream use, not merely to immediate equivalence**.

Examples of findings the tool should detect:

### 1\. Fully redundant all-gather

A full gather occurs, but the next compute only uses the local shard already present before the gather.

### 2\. Redundant duplicate gather

A tensor has already been gathered identically across the same participant set, and no intervening op invalidated that equivalence.

### 3\. Over-wide gather

A gather materializes the entire tensor, but only a subset of gathered regions is consumed.

### 4\. Redundant gather before fusion opportunity

Two neighboring AGMMs each gather overlapping activation regions that could be satisfied by one shared communication step or by eliminating one gather after fusion.

### 5\. Participant-set mismatch

A collective is executed over a broader device group than the actual consumer set requires.

## Implementation plan

### Phase 0: align on scope and success criteria (1 week)

Deliverables:

- one-page problem statement
- list of target models / forward passes
- first-class collectives to support
- definition of “redundant” vs “reducible” vs “suspicious”
- gold examples, including the 12-AGMM-to-6 case

This phase matters because a general static analyzer is too open-ended. The first version should be benchmarked on a small number of known communication patterns from DiT/LTX.

### Phase 1: build the semantic IR and graph capture (2–3 weeks)

Goal: export forward-pass graphs into a stable IR.

Options for graph capture:

- trace from TTNN graph construction if such hooks already exist
- instrument model-building APIs to emit structured op records
- import from an existing lowered graph if available

Requirements:

- deterministic graph serialization
- stable tensor IDs
- explicit device-mesh metadata
- explicit collective nodes
- source-code locations for debugging

Output:

- JSON or protobuf graph dump
- small graph viewer or text printer for sanity checks

### Phase 2: implement the device-state simulator (3–4 weeks)

Goal: accurately compute per-device tensor states after every node.

Tasks:

- implement region-set abstraction
- implement layout states: replicated, shard(axis, range), partial, gathered
- implement transfer functions for Tier 1 ops
- support equivalence/provenance tracking
- emit state snapshots at every collective boundary

Success criterion: For several known forward graphs, an engineer can inspect a collective and see a believable before/after state table for every device.

### Phase 3: add backward demand analysis (2–3 weeks)

Goal: determine what data is actually needed at each program point.

Tasks:

- define consumer requirements per op
- propagate region demand backward
- handle aliases/views conservatively
- add barrier semantics for ops that force materialization

Success criterion: For a chosen collective, the tool can print:

- what regions are available before it
- what regions are needed downstream
- why the collective is or is not required

### Phase 4: implement redundancy rules and reporting (2 weeks)

Goal: turn analysis into actionable findings.

Rules should include:

- identical-prior-gather detection
- unused-full-gather detection
- gather-width reduction opportunities
- removable collective after fusion / op reorder hints
- collective-group shrink hints

Output format per finding:

- severity
- confidence
- affected ops
- proof summary
- suggested optimization shape
- estimated bytes / latency impact if removable

### Phase 5: validate on historical wins and misses (2 weeks)

Use real examples from the team:

- the redundant all-gathers Kevin identified
- known necessary collectives the tool must not flag
- past AGMM bottleneck graphs across compute/fabric/DRAM-bound regimes

Metrics:

- precision of redundancy flags
- false-positive rate
- percentage of known wins rediscovered
- engineer time to inspect a finding

### Phase 6: integrate into optimization workflow (1–2 weeks)

Integrate as:

- a CLI run on captured forward graphs
- a CI check for selected model graphs
- an optional visualization view for debugging

The first shipped workflow could be as simple as:

1. export graph from model
2. run analyzer
3. print ranked redundancy findings
4. inspect proof trace for top findings

## Team roles

A lean team could do this with 2–3 engineers.

### Engineer 1: graph / compiler infra

Owns IR design, tracing, serialization, and graph normalization.

### Engineer 2: distributed semantics / analysis

Owns device-state model, transfer functions, backward demand analysis, and redundancy rules.

### Engineer 3: model integration and validation

Owns integration with DiT/LTX forward passes, gold test cases, and usability feedback from model engineers.

If staffing is tighter, two engineers can do it, but only if they **limit v1 scope aggressively**.

## Recommended v1 scope

To keep this build realistic, v1 should support only:

- inference forward pass
- one device topology family at a time
- one or two parallelism strategies already common in DiT
- AGMM-heavy subgraphs
- all-gather redundancy first

That is enough to catch high-value communication mistakes without getting trapped in full-framework generality.

## Concrete internal abstractions

### 1\. Device mesh

Represent as:

- device IDs
- mesh axes
- adjacency/topology metadata
- named communication groups

### 2\. Logical tensor region

Represent tensor contents symbolically by axis intervals, for example:

- full tensor
- rows `[a:b)`
- columns `[c:d)`
- Cartesian products for multi-axis shards

### 3\. Layout descriptor

Example enum-like structure:

- replicated
- shard(axis, interval, group)
- partial\_reduction(axis, contributors)
- gathered(group)
- unknown

### 4\. Equivalence relation

Track when two tensor instances are semantically equal over the same region. This is what lets the tool prove a collective is duplicate or unnecessary.

### 5\. Proof object

Every finding should include a machine-readable proof object:

- collective node ID
- participant set
- required regions
- available equivalent regions
- invalidation check
- conclusion

This is crucial. Engineers will trust the tool only if it explains itself.

## Key engineering choices

### Choice 1: conservative correctness over aggressive cleverness

If semantics are uncertain, emit “unknown” instead of a bad optimization recommendation.

### Choice 2: symbolic reasoning over numeric simulation

Do not simulate tensor values. Simulate ownership, layout, and dependency semantics.

### Choice 3: op-spec-driven analyzer

Store op semantics in a declarative or semi-declarative registry. That makes it easier to extend coverage without rewriting the engine.

### Choice 4: proofs before auto-fixes

Do not begin with automated rewrites. Begin with trustworthy diagnostics.

### Choice 5: state the trust level on every report ("the shim believes")

A finding is only as trustworthy as the graph it was computed on, and the
everyday graph now comes from a metadata-only `ttnn` shim, not from hardware. So
it is a **requirement**, not an option, that the tool label the provenance of
every result and say so explicitly — in the words "the shim believes" — whenever
a finding rests on shapes the shim *computed* rather than on real `ttnn`. A
shim-derived finding must never read as device-verified. Concretely: each graph
carries a `provenance` tag (`dry-run` / `hand-written` / `captured` / `unknown`)
that survives serialization, and `report.render_trust` turns it into a `trust:`
banner on every report and every `ditcheck dryrun`. On-device conformance
(`conform.py`, phase 11) is what promotes a `dry-run` finding out of "the shim
believes"; until then the banner says so. This generalises the existing
withhold-don't-guess and taint rules from *individual* assumptions to the
*whole-graph* assumption that the shim's shapes match ttnn's.

## Risks and mitigations

### Risk: semantic complexity explodes

**Mitigation:** scope v1 to AGMM-centric inference graphs and Tier 1 ops only.

### Risk: graph capture misses important runtime behavior

**Mitigation:** require explicit annotations for dynamic shape decisions, communication groups, and lowered collective variants.

### Risk: too many false positives

**Mitigation:** attach confidence levels and demand proof traces; validate against known-good pipelines before broad use.

### Risk: engineers ignore the tool

**Mitigation:** optimize for top-10 ranked findings with very short proof summaries and source links.

## What “done” looks like

A successful v1 can take an LTX / DiT forward graph and produce output like:

- “Collective 184: all-gather over devices 0–7 is likely redundant.”
- “Reason: downstream consumers use only local row shards already available on each device.”
- “Equivalent full gather already exists from collective 172 and is not invalidated.”
- “Potential savings: eliminate 1 collective across 12 calls in this block.”

If it can reliably rediscover Kevin’s finding, and do so early in bring-up, it is already worth the build cost.

## Suggested development cadence

### Month 1

- scope
- graph capture
- IR skeleton
- first Tier 1 op semantics

### Month 2

- forward state simulator
- backward demand engine
- proof traces
- first redundancy rules

### Month 3

- validation on real DiT/LTX graphs
- ranking / reporting polish
- CI / workflow integration

## Immediate next steps

1. Write a short design doc with the v1 scope above.
2. Collect 5–10 real forward graphs, including one with the redundant AGMM pattern Kevin found.
3. Define formal semantics for all-gather, AGMM, reshape/view, split, concat, and elementwise ops.
4. Build a text-mode prototype that prints per-device tensor state around collectives.
5. Only after the state model is trusted, add redundancy classification.

## Bottom line

The right tool is not a generic “communication optimizer.” It is a **distributed forward-pass state analyzer** with a demand engine and proof-based redundancy checks. If the team keeps the first version narrow, this is very buildable in roughly one quarter and could pay for itself the first time it catches a one-line collective-elimination win before months of kernel work are spent downstream.

*Co-authored with Glean*
