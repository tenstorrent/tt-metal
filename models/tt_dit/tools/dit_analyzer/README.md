# ditcheck — collective-redundancy static analyzer for DiT forward graphs

The tool described in [`DitStaticAnalyzerPlan.md`](DitStaticAnalyzerPlan.md):
a **distributed forward-pass state analyzer** with a backward demand engine and
proof-based redundancy checks. It answers, for every collective in a forward
pass: *what did each device already have, what does anything downstream actually
need, and is this collective doing any work?*

It is pure Python and imports no real `ttnn`, so both the analysis *and* the graph
it analyses come off a laptop: [`dryrun/`](dryrun/) runs the real model code
against a metadata-only `ttnn` and emits the graph as a side effect. Only
[`capture.py`](capture.py) needs a live device, and nothing in the daily loop
uses it.

**Status** — what is built, what is next, and how much of a finding to believe
today: [`DitStaticAnalyzerRoadmap.md` § "Where we are"](DitStaticAnalyzerRoadmap.md#where-we-are).
That section is the status of record; this file describes the tool as it stands and
the roadmap holds the remaining work and the 44-item blocker inventory.

```bash
# derive the graph from models/tt_dit source -- no device, no checkpoint, no capture
models/tt_dit/tools/ditcheck dryrun ltx_block --preset bh_4x8 --analyze --top 1

# ... and check it still agrees with the hand-written graph for the same block
models/tt_dit/tools/ditcheck dryrun ltx_block --preset bh_4x8 --check-oracle

# a second block from source: the SD3.5-large joint block (every collective load-bearing)
models/tt_dit/tools/ditcheck dryrun sd35_block --preset bh_2x4 --check-oracle

# a VAE block from source: SD3.5 VAE ResnetBlock (conv2d / group_norm), runs clean
models/tt_dit/tools/ditcheck dryrun sd35_vae_resnet --preset bh_2x4 --analyze

# LTX-2.3 A+V block on BH 4x8 (Ring): 6 provably duplicate TP gathers per block
models/tt_dit/tools/ditcheck analyze example:ltx_block_bh_4x8 --top 1

# the same source on BH 2x4 (Linear topology): nothing redundant
models/tt_dit/tools/ditcheck analyze example:ltx_block_bh_2x4

# every collective in the real SD3.5-large block is load-bearing -> no findings
models/tt_dit/tools/ditcheck analyze example:sd35_block

# the same block with fused AGMM added but the explicit pre-gathers left in place
models/tt_dit/tools/ditcheck analyze example:sd35_block_double_gather --top 3

# per-device state table around each collective
models/tt_dit/tools/ditcheck states example:sd35_block --node pre_attn

models/tt_dit/tools/ditcheck dryrun            # list dry-run targets and presets
models/tt_dit/tools/ditcheck examples          # list built-in graphs
models/tt_dit/tools/ditcheck ops               # list registered op semantics
models/tt_dit/tools/ditcheck ops --check g.json  # exit 1 if a graph uses an uncovered op
models/tt_dit/tools/ditcheck dump example:sd35_block > g.json
models/tt_dit/tools/ditcheck analyze g.json --json findings.json --fail-on provable
```

Tests (no device, no pytest needed):

```bash
python3 models/tt_dit/tools/dit_analyzer/tests/test_dit_analyzer.py   # analyzer, 24 tests
python3 models/tt_dit/tools/dit_analyzer/tests/test_dryrun.py         # dry run, 19 tests
# or: pytest models/tt_dit/tools/dit_analyzer/
# on a device: shape conformance vs real ttnn (phase 7b / 11)
python3 models/tt_dit/tools/dit_analyzer/conform.py --mesh 2 4
```

## The dry run

`ditcheck dryrun` runs a real `models/tt_dit` module against a `ttnn` whose
`Tensor` carries only metadata — per-device shape, logical shape, dtype, layout,
distribution — and whose ops compute output metadata and append IR nodes.
`LTXTransformerBlock.forward` is ordinary Python: if
`all_gather_minimal_matmul_async` returns a metadata tensor instead of doing work,
the forward runs on a laptop and emits the graph directly. No hardware, no
checkpoint download, no trace capture, and no hand-written model.

It derives the same findings the hand-written `examples/ltx.py` states, on both
shipped Blackhole configs — 31 vs 31 collectives and 6 provable duplicate gathers
on Ring, 25 vs 25 and none on Linear — which is what `--check-oracle` asserts. The
examples stay as oracles rather than scaffolding: a shim shape rule that drifts
does not perturb the graph, it *invents* redundancy, so the diff is the regression
test. [`spike/FINDINGS.md`](spike/FINDINGS.md) is the record of the prototype this
grew from, including the two shape bugs that produced 15 spurious findings.

What a dry run still needs stated, because it cannot be read off the source: mesh
shape, which mesh axis carries which parallel role, activation shapes, and
checkpoint-derived branch flags (`has_audio`, `has_gate`). Those live in one place
per target, [`dryrun/targets.py`](dryrun/targets.py), which is also what phase 12's
coverage matrix will sweep.

### The host environment

Weights are `torch.empty(..., device='meta')`: shapes with no bytes and no kernels,
so torch is only ever asked for metadata — no CUDA, no MPS, no compute backend. A
plain `pip install torch` is enough, including on an Apple Silicon laptop (Python
3.9 resolves to torch 2.8.0, the last release with a `cp39` macOS-arm64 wheel).

Every run prints what it had to stand in for, so it is never quietly less faithful
than it looks:

```
torch: real (device='meta')
  substituted loguru: silent logger
```

With torch, loguru, safetensors, numpy and pytest present the list is empty and
the run uses the real `models.common.utility_functions`. Without them the dry run
still works: there is a metadata-only torch in [`dryrun/hostfakes.py`](dryrun/hostfakes.py)
and stubs for the rest, because the point of this design is a redundancy check at
unit-test cost — a device-free run should not even need a torch install.
Order matters, and one thing to keep in mind when adding to `hostenv.py`: nothing
under `models.` may be imported before the shim is installed, or tt_dit's
module-level `import ttnn` drags in the real one and `install()` then refuses to
run at all.

```
dryrun/
  install.py    shadow `ttnn` in sys.modules; refuse to displace a real one
  tensor.py     the metadata Tensor: local vs logical shape
  ops.py        one shape/distribution rule per ttnn op
  fused.py      fused-kernel table: which kernels hide which collective (`ops --fused`)
  recorder.py   ops -> IR nodes, with a short tt_dit caller stack per node
  stubs.py      mesh device, mesh mappers, enums, program configs
  weights.py    load every Parameter from torch meta tensors, via the real path
  hostenv.py    host-side imports; prefers real torch, reports what it substituted
  targets.py    named targets and mesh presets
  verify.py     the four acceptance criteria against an examples/ oracle
```

Two honesty properties are worth knowing about before reading a report:

* **Nothing is invisible.** A ttnn call with no shape rule still runs, and still
  appears in the IR as an `unregistered` node with real inputs, outputs, shapes and
  source location. `ditcheck ops --missing <graph>` lists them; `--check` exits nonzero.
* **Analysis withholds, never guesses.** The output metadata of an `unregistered`
  node is *assumed* to match its first input, so any finding whose proof passes
  through one is not reported and not downgraded. It goes to a `withheld` queue
  that names the registration that would unlock it.

## Phase 5: what it found in LTX-2.3

`examples/ltx.py` models one `LTXTransformerBlock` (audio + video) branch-for-branch
from `attention_ltx.py` / `transformer_ltx.py`, for both shipped Blackhole
configs. They differ only in CCL topology, and that flips one line:

```python
use_nonfused_agmm = (self.ccl_manager.topology == ttnn.Topology.Linear) and tp_factor > 1
qkv_parallel_config = None if use_nonfused_agmm else self.parallel_config
gate = self._compute_gate(spatial_1BND, qkv_parallel_config)     # to_gate_logits(...)
q, k, v = self.to_qkv(spatial_1BND, parallel_config=qkv_parallel_config)
```

On **Ring** (BH 4x8) `qkv_parallel_config` is not None, so `to_gate_logits` *and*
`to_qkv`/`to_q` each take the fused `all_gather_minimal_matmul_async` path — and
each gathers the same activation over the TP axis. The analyzer reports one
`duplicate_gather` per attention instance, 6 per block, `provable`:

```
#1  [HIGH/provable]  duplicate_gather
    attn1.to_qkv_q_ag (all_gather fused in agmm:attn1.to_qkv_q) duplicates data already materialised by all_gather_30
    source: models/tt_dit/models/transformers/ltx/attention_ltx.py:428
    why:    attn1.to_gate_logits_ag_29 already holds every region this collective produces, on all 4 participants.
    why:    Both carry value_id v41c7e33351, so no compute between them changed the value.
    cost:   912.0 MiB per call x 48 calls = 42.8 GiB of link traffic per forward; x8 steps = 342.0 GiB per generation
    proof:  invalidation_check: SSA graph: attn1.to_gate_logits_ag_29 is still live and carries value_id
            v41c7e33351, identical to the collective's operand; no intervening node redefines it
```

The block's 6 attention instances issue **12 TP activation gathers** for their
gate and Q/QKV projections where **6** would do (22 -> 16 TP gathers per block
overall). The three video-sized duplicates — attn1, attn2 and a2v, all on the
38912x4096 activation — account for ~128 GiB of link traffic per forward pass
aggregated over the 32 devices (1.3 GiB per device each); the three audio-sized
ones are ~150 MiB each. The gate projection's *output* is tiny (`num_heads`
columns), so its gather is pure overhead.

Applies when the checkpoint carries gate weights (`has_gate` is detected from the
state dict, `transformer_ltx.py:1090`) — i.e. the audio+video checkpoints — and
only on Ring. On **Linear** (BH 2x4) the same source pre-gathers once and passes
`parallel_config=None`, and the analyzer reports **no findings** across all 25
collectives, which is the result that makes the Ring finding worth acting on.

Two candidate fixes, in preference order:

1. **Fuse the gate into the QKV projection.** `to_qkv` is already a chunked
   ColParallelLinear (`chunks=3`); making it `chunks=4` with `num_heads` extra
   output columns gives the gate its own chunk out of the *same* gather. The gate
   always consumes exactly the tensor `to_q`/`to_qkv` consumes (both use
   `query_input_dim`), so the fusion is always type-compatible. Removes the
   gather without giving up the AGMM's comm/compute overlap.
2. **Pre-gather once, like the Linear path** (explicit `all_gather` +
   `parallel_config=None` for both projections). Simpler, but loses the overlap
   on the large QKV matmul.

A caveat this exercise exposed: an earlier version of the model omitted the
per-direction adaLN modulations in the A↔V cross-attention, and the analyzer
correctly reported extra "duplicates" for what were then genuinely identical
values. Those disappeared once `video_q_a2v` / `video_kv_v2a` (and the audio
pair) got their own shift/scale, as in the real code. Findings are only as good
as the graph it is given — which is the argument for deriving the graph from the
model code instead of restating it by hand. See the roadmap for how: a
metadata-only `ttnn` shim runs the real forward on a laptop.

## What it found on the gold case

`sd35_block_double_gather` models a plausible mistake: switch the
`ColParallelLinear` calls to the fused `all_gather_minimal_matmul_async` path
without removing the explicit gathers in `transformer_block.py` /
`attention.py`. 16 collectives per block, 6 of them provably doing nothing:

```
#1  [HIGH/provable]  unused_gather
    attn.to_out_ag (all_gather fused in agmm:attn.to_out) is redundant: consumers only read data each device already had
    source: models/tt_dit/blocks/attention.py:328
    why:    Downstream consumers (attn.to_out (matmul)) demand only regions already present before the collective.
    why:    ag_attn_out already made the operand complete on this axis, so the pair is redundant.
    cost:   60.0 MiB per call x 38 calls = 2.2 GiB of link traffic per forward (all participants); x28 steps = 62.3 GiB per generation
    fix:    Drop one of the two: remove ag_attn_out (keep the fused/later collective, which usually overlaps
            communication with compute), or keep ag_attn_out and drop this one.
    proof:
      layout_before:      shard(dim1,sp), replicated(tp)
      value_id:           v71c5751f91
      invalidation_check: value-preserving collective: output value_id == input value_id
      available_before / materialised_after / needed_downstream: per device, per region
```

The same run leaves the 10 genuinely necessary collectives alone, which is the
property that matters most for adoption.

## How it works

```
DitStaticAnalyzerPlan.md      the plan this implements (scope, phases, rule classes)
DitStaticAnalyzerRoadmap.md   blockers to running on real pipelines, and phases 6-13
region.py     interval/box algebra over logical tensor axes (union, subtract, covers, volume)
ir.py         Mesh, TensorSymbol, Dist, Node, Graph (+ JSON), value identity
state.py      per-device SymbolState: region owned, layout, provenance, taint
semantics.py  op registry: forward `apply` + backward `demand` per op (Tier 1)
analysis.py   forward availability walk, backward demand walk
rules.py      redundancy rules -> Finding + machine-readable proof
report.py     text rendering: state tables, ranked findings, proofs, diagnostics
builder.py    DSL for writing/lifting graphs; expands fused ttnn ops into stages
capture.py    record a real ttnn forward pass -> trace -> graph
conform.py    on-device: diff the shim's per-device shapes against real ttnn (phase 7b/11)
link.py       link per-stage graphs into one multi-stage pipeline graph (`ditcheck link`, phase 10c)
dryrun/       real model code under a metadata-only ttnn -> graph, no device
dryrun/checkpoint.py  checkpoint-derived branch flags from a metadata-only index
examples/     gold graphs (LTX-2.3 block x2 topologies, SD3.5 block, synthetic patterns)
spike/        FINDINGS.md: what the dry-run prototype answered, and what it cost
```

Three ideas carry most of the weight:

1. **Logical regions, not buffers.** State is "device 2 owns rows `[0:2048)` x
   cols `[1216:1824)` of `spatial_normed`", so *partial* availability is
   expressible and over-wide collectives are measurable.
2. **`value_id` is the mathematical value; `Dist` is its materialisation.**
   Collectives propagate their input's `value_id` (they move data, they do not
   change it); compute ops mint a new one. "Was this invalidated between the two
   collectives?" then reduces to comparing two ids — that is the whole
   invalidation check, and it is exact because the IR is SSA.
3. **Two-sided analysis.** Forward gives *available*; backward gives *needed*.
   A collective is redundant exactly when needed ⊆ available-without-it. Forward
   simulation alone cannot see that a gathered tensor is never read in full.

Partial sums are tracked separately (`Dist.partial`), so a reduce-scatter is
never mistaken for redundant data movement: its input is not the same value as
its output. That gate is what keeps `RowParallelLinear` out of the report.

### Rules

| rule | confidence | claim |
|---|---|---|
| `dead_collective` | provable | nothing downstream demands any region of the result |
| `unused_gather` | provable | every participant already held everything its consumers read |
| `duplicate_gather` | provable | an equivalent, uninvalidated materialisation already exists |
| `overwide_gather` | likely | a measurable fraction of the moved volume is never read |
| `participant_shrink` | likely | the group is wider than the set that needs remote data |
| `invariant_collective` | likely | operand is constant across denoise steps — hoist/cache it |

Reported separately as **hints** (not redundancy, no provable byte savings):
`mergeable_collectives` — independent collectives on the same axis and group
that could be issued as one, saving fixed per-collective cost (semaphores,
barrier, ring warm-up).

Every finding carries `severity`, `confidence`, the affected nodes with source
locations, a per-device proof object, and a byte estimate scaled by how many
times the node runs (`calls`, e.g. 38 blocks) and by denoise steps.

Source locations are a short caller stack, not one line: a duplicate gather
reported at `layers/linear.py:250` is true but not actionable, so a finding leads
with the model frame that chose to gather and names the library frame underneath.

```
source: models/tt_dit/models/transformers/ltx/attention_ltx.py:428
        via models/tt_dit/layers/linear.py:250
```

### Honesty rules the analyzer follows

* **Every report declares its trust.** The header carries a `trust:` line stating
  how the graph's shapes were produced, and it says **"THE SHIM BELIEVES"** in as
  many words whenever a finding rests on shapes the metadata-only ttnn shim
  *computed* rather than on real ttnn (any `provenance: dry-run` graph). This is a
  requirement, not a nicety: a shim-derived finding must never be mistaken for a
  device-verified one. Hand-transcribed (`examples/`) graphs say so; a captured
  device trace says its shapes are ground truth. The tag rides in the graph JSON,
  so it survives `dump` → `analyze`.
* An op with unknown semantics gets pessimistic semantics **and** taints
  everything downstream; findings touching tainted values are demoted to
  `suspicious`.
* An `unregistered` op — one the dry run saw and had no shape rule for, so its
  output metadata is a guess — goes further: findings downstream of it are
  **withheld**, listed with the registration that would unlock them. A wrong shape
  does not weaken a finding, it invents one.
* Anything the analyzer cannot model is a `diagnostic` (`UNKNOWN_OP`,
  `GATHER_OF_PARTIAL`, `LAYOUT_MISMATCH`, `K_COVERAGE`, …), printed with the
  findings. Read those before trusting a report.
* Byte counts are a first-order ring model (`(g-1)/g` of the payload per
  device), independent of the region math, because a collective on
  already-replicated data still pays full fabric cost. Latency numbers appear
  only with `--link-bw` and are labelled as estimates.
* A verdict that holds on some participant groups but not others says so in the
  finding title instead of generalising.

## Capturing a real forward pass

Analysis needs the mesh layout of the tensors *entering* the traced region: a
ttnn tensor's `.shape` is per-device and nothing on it says which mesh axis
fractures which tensor axis. So capture is two steps, and undeclared placements
are recorded as assumptions rather than guessed.

```python
# on device
from dit_analyzer.capture import capture
with capture(mesh_device, name="sd35_block", steps=28) as cap:
    model(spatial, prompt, ...)
cap.write("sd35.trace.json")

# offline
from dit_analyzer.capture import Trace, trace_to_graph
from dit_analyzer.ir import Dist, Mesh
trace = Trace.read("sd35.trace.json")
print(trace.entry_summary())                    # which entries need a placement
mesh = Mesh(shape=(2, 4), axis_names=("sp", "tp"))
graph = trace_to_graph(trace, placements={"in0": Dist.make(mesh, {0: 1, 1: 2})}, params=["in3"])
```

`capture.HOOKS` lists the patched calls (all_gather_async,
reduce_scatter_minimal_async, all_gather_minimal_matmul_async,
minimal_matmul[_strided_reduce_scatter_async], dit_fused_distributed_*norm,
split_query_key_value_and_split_heads, concatenate_heads,
ring_joint_scaled_dot_product_attention, pointwise/reshape ops). Fused ops are
expanded into their communication and compute stages, tagged `fused_in`, so the
gather inside an AGMM is analyzable while the report still names the real kernel.

**Status:** the recorder half has not been run against hardware yet — it is
wired against the op names in this tree today. The offline half
(`trace_to_graph`, including per-device → logical shape lifting) is covered by
the tests. A sturdier long-term alternative is `ttnn.graph.begin_graph_capture`
/ `end_graph_capture_to_file`, which already records a full op/tensor graph in
C++ (`ttnn/ttnn/graph.py`, node types `function_start` / `tensor` / `buffer`
with `connections`); a converter from that report would replace the monkeypatch
path and pick up ops nobody remembered to hook.

## Scope, and what is deliberately not here

Built (plan phases 1–4 and roadmap phase 6, narrowed to the v1 scope in the plan):

* IR + JSON serialisation, device mesh, region algebra, value identity
* Tier-1 op semantics: all-gather, reduce-scatter, all-reduce, matmul (column /
  row parallel, partial sums), AGMM and matmul+RS as fused stages, distributed
  and local norms, pointwise, slice/concat, view/squeeze/reshape, fused-QKV
  split (the per-device `[n_dev][q|k|v]` layout this tree actually uses),
  merge-heads, SDPA (incl. ring-SDPA's internal K/V gathers), host readback
* forward availability + backward demand, proof objects, ranked text report,
  JSON output, `--fail-on` for a manual gate / scripts, per-device state tables
* the dry-run front end: `ditcheck dryrun` builds a real tt_dit module against the
  metadata-only `ttnn`, loads its weights from torch meta tensors through the real
  `Parameter` path, records a caller stack per node, and diffs against the
  hand-written oracle

Not built (and where it would go):

* **Tier-2 semantics**: `mesh_partition`, `pad`, `repeat`, KV-cache updates, MoE
  routing, conv/VAE spatial collectives (`neighbor_pad_async`,
  `slice_reshard_async`), cross-device concat variants. Under the dry run these
  become `unregistered` nodes → withheld findings, which is the intended failure
  mode. Add a shape rule in `dryrun/ops.py` and a spec in `semantics.py`; nothing
  else changes. Roadmap phase 8 merges the two into one registration.
* **Automated rewrites.** Diagnostics only, per the plan's "proofs before
  auto-fixes".
* **Shape fidelity — phase 7a, shipped and corroborated on 2×4.** Tile padding is
  real (`shape` vs `padded_shape` are distinct; byte/cost math uses a tile-padded
  volume), shard division reproduces ttnn's `torch.chunk` rule including uneven
  splits, block-float bytes carry their exponent overhead, and checkpoint flags
  (`has_gate`, `cross_attention_adaln`) are derived from a metadata-only index.
  [`conform.py`](conform.py) diffs the shim's per-device shapes against real ttnn
  on a 2×4 Blackhole mesh and they match. Remaining: a fused weight's column
  *interleave* is not modelled (the analyzer doesn't consume column order), and the
  4×8 Ring finding needs a 32-chip Galaxy to corroborate.
* **Whole pipelines.** One block on one mesh, not encoder → DiT → VAE across
  submeshes with carried latents (phase 10), and no branch/shape sweep (phase 12).
* **Trusting the shim.** Until per-op conformance runs on a device (phase 11),
  every dry-run finding should be read as "the shim believes".
* **On-device conformance.** `capture.py` remains for the device's two jobs in
  the new design -- per-op shape/layout conformance and one flat collective log to
  diff the dry run against -- and has not been run on hardware yet.
* **Multi-block / whole-pipeline graphs.** Examples model one block with a
  `calls` multiplier, so a redundancy that spans two *different* blocks is out of
  reach here (not in the analysis -- the rules are not block-scoped -- only in
  these examples). * Cost model beyond first-order bytes: no link contention, no overlap with
  compute, no per-op fixed cost. Rankings are byte-based; treat them as
  "look here first", not as a latency prediction.
