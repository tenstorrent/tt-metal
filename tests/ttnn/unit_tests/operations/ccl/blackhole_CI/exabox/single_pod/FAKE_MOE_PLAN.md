# Fake MoE traffic — design and execution plan

## Goal

Build a stand-in for the deepseek_v3_b1 `MoEDecoderStage` that reproduces the
MoE stage's **inter-device traffic pattern** without invoking the real
`MoeOp`. Once it works in isolation, drop it into the 16-stage single-pod
pipeline so `test_single_pod_pipeline_setup_and_decode` runs end-to-end
without the model bug we hit in `MoeOp` (`Tensor must be sharded to
automatically create a CBDescriptor`).

The fake MoE serves three purposes:

1. **Validate the exabox CCL stack at single-pod scale** — chains four
   primitives the real MoE depends on, on the same `(4, 2)` per-rank mesh,
   with the same shapes and `FABRIC_2D_TORUS_Y` setup.
2. **Unblock the single-pod pipeline test** — gives us a working 16-stage
   integration test that exercises the full pipeline scaffolding (socket d2d
   between ranks, fabric_router, slow dispatch, mesh-graph descriptor,
   weight-upload phase) without depending on the broken `MoeOp` synthetic
   path.
3. **Provide a reusable simulation harness** for future MoE-style traffic
   experiments without the demo dependency.

The fake MoE does **no MoE math**. It chains real CCL ops with random tensors
on the right shapes and topologies, reads back deterministic results to
verify correctness of each primitive, and forwards a token-shaped tensor
downstream. To anyone watching the network, it looks like an MoE stage.

## What MoE actually does (one iteration on one rank)

Reproduced here for reference; full breakdown is in the chat that produced
this plan. Each rank owns a `(4, 2)` BH submesh = 8 devices.

| # | Step | Pattern | CCL primitive (or substitute) | Payload |
|---|---|---|---|---|
| 1 | inbound socket recv | point-to-point | `PipelineBlock.upstream_d2d_socket` | `[batch, seq, hidden]` bf16 |
| 2 | broadcast within submesh | 1 → 8 | `ttnn.broadcast` (`cluster_axis=1`, FABRIC_2D) | same |
| 3 | attention SDPA reduce | 8 → 8 (axis 0 only) | `ttnn.all_reduce` (`cluster_axis=0`) | `[batch, heads, head_dim]` |
| 4 | expert dispatch | tokens → expert-owning devices | `ttnn.all_to_all_dispatch` (`cluster_axis=1`) | `[batch, K, hidden]` |
| 5 | expert compute (per-device) | local | none | — |
| 6 | expert combine | expert-results → originating device | `ttnn.all_to_all_combine` (`cluster_axis=1`) | `[batch, K, hidden]` |
| 7 | reduce-to-one | 8 → 1 device on this rank | `ReduceToOneB1` (or substitute via `ttnn.all_reduce` + read one) | `[batch, hidden]` |
| 8 | outbound socket send | point-to-point | `PipelineBlock.downstream_d2d_socket` | `[batch, hidden]` |

Steps 1, 5, 8 are not collectives — fake MoE does not need to simulate them
for the **CCL** side, but step 1 and step 8 *are* required when the fake MoE
is dropped into the pipeline (Phase 3). For Phase 2 (standalone) we can
ignore them.

## Phasing

### Phase 1 — Plan + supporting helper module (this doc + skeleton)

**Deliverables:**

- `FAKE_MOE_PLAN.md` (this file)
- `_fake_moe_helpers.py` — utility functions used by the standalone test and
  later by the pipeline-integration test. Empty or minimal at first; grows
  during Phase 2.

**Acceptance:** doc + skeleton merged; reviewer agrees on the overall plan
before code is written.

### Phase 2 — Standalone CCL chain test, no pipeline scaffolding

**File:** `test_fake_moe_traffic.py`

Open a `(4, 2)` mesh per rank, run the CCL chain (steps 2 → 7), verify
correctness against torch goldens. **Does not use sockets, does not depend
on the demo model, does not depend on slow dispatch.**

**Tiered targets** (each must pass before moving to the next):

1. **`test_fake_moe_traffic_8x4_singlebh`** — runs on `SINGLE_BH` 8×4 host,
   creates a `(4, 2)` submesh of 8 devices, exercises the chain on a single
   rank (no MPI). This is the easiest to debug; if anything breaks it's
   a CCL-or-shape issue, not a multi-host issue.

2. **`test_fake_moe_traffic_per_rank_dual_bh`** — runs under `tt-run` on
   DUAL_BH 16×4, with 2 MPI ranks, each rank gets a `(4, 2)` submesh. Each
   rank does the chain independently on its 8 devices. This validates the
   per-rank submesh slicing without yet involving the 16-rank single-pod
   topology.

3. **`test_fake_moe_traffic_single_pod`** — runs under `tt-run` with the
   16-rank single-pod rank-binding (the same one we generated and ran for
   `test_single_pod_pipeline.py`). Each of the 16 ranks runs the chain on
   its `(4, 2)` submesh. This validates that the chain composes with
   `FABRIC_2D_TORUS_Y` + `fabric_router_config(15232)` + `worker_l1_size`.

The chain itself is the same code in all three; only the `requires_device`
marker, `mesh_device` parametrize, and `device_params` change.

**Acceptance:** all 3 tiers pass on the cluster.

### Phase 3 — `FakeMoeDecoderStage` class wrapping the chain

**File:** `_fake_moe_helpers.py` grows a `FakeMoeDecoderStage` class.

Mirror `MoEDecoderStage`'s interface from
`models/demos/deepseek_v3_b1/demo/decoder_stage.py`:

```python
class FakeMoeDecoderStage(StageKind):
    def create_pipeline_block(self, ctx) -> PipelineBlock: ...
    def setup(self, ctx, pipeline_block) -> None: ...
    def launch_compute(self, ctx, pipeline_block) -> None: ...
```

- `create_pipeline_block`: identical to `DecoderStage.create_pipeline_block`
  — same socket FIFO sizes, same entry/exit node coords. We're imitating the
  network behavior of an MoE stage, so the socket wiring should match.
- `setup`: allocate the synthetic input/output tensors and any semaphores
  the chained CCL primitives need (the existing primitives have their own
  semaphore allocators we can call). Crucially, **no `MoeOp.create_…`
  calls** — the failure point.
- `launch_compute`: receive from upstream socket → run the CCL chain → send
  to downstream socket.

**Acceptance:** Phase 3 produces a class importable from
`_fake_moe_helpers`, exercised by a small unit test that constructs it on
a single rank and runs `launch_compute` once with a stub upstream socket
(if feasible without the full pipeline).

### Phase 4 — Pipeline integration

**File:** `test_single_pod_pipeline_fake_moe.py` (new test, doesn't replace
the original).

Build a custom pipeline configuration: replace the 10 `MoEDecoderStage`
factories (stages 4–13 in `single_pod_pipeline_configuration`) with
`FakeMoeDecoderStage`. The Embedding (stage 0), Dense decoder stages (1–3),
LMHead (stage 14), Passthrough (stage 15) factories are kept as-is.

```python
stage_factories: dict[int, Callable[[ttnn.MeshDevice], StageKind]] = {
    0: lambda d: EmbeddingStage(weight_provider.load_embedding(d)),
    1: _dense_stage(0),
    2: _dense_stage(1),
    3: _dense_stage(2),
    **{i: _fake_moe_stage() for i in range(4, 14)},   # ← change
    14: lambda d: LMHeadStage(...),
    15: lambda d: PassthroughStage(...),
}
```

Use `ModelPipeline._set_pipeline_config(...)` (or whatever injection point
exists) — if no such API exists, build the `Pipeline` object directly with
our own config map, bypassing `ModelPipeline`. Either way, the rest of the
test is mechanically identical to `test_single_pod_pipeline_setup_and_decode`.

**Acceptance:** the 16-rank pipeline reaches `Pipeline started, waiting for
all stages to complete` on every rank, and the 4 decode iterations actually
run end-to-end (one prefill + four decode steps for the prompt of length 1
or 8). Rank 0 returns 4 token IDs from `run_inference(...)`.

### Phase 5 — Documentation update

Update `AGENTS.md` (the runbook) with:

- A section on the fake-MoE pattern and when to use it
- The `tt-smi -glx_reset_auto` workflow (we already use it; document it
  explicitly)
- The mpirun-wrapper recipe for `generate_blitz_decode_pipeline_configs.py`
  (ulimit / SSH-agent / TT_METAL_HOME symlinks)

## Risks and open questions

1. **Reduce-to-one substitution.** The real MoE uses `ReduceToOneB1` (a 3-level
   tree). We have two options:
   - Reuse the `ReduceToOneB1` class directly. It's standalone and not tied
     to the demo, but expects a specific tensor layout. May require its own
     intermediate tensors and semaphore setup.
   - Substitute with `ttnn.all_reduce(cluster_axis=both?)` and read one
     device. Easier to set up; bandwidth is ~2× the tree but result is the
     same.
   - **Initial choice for Phase 2:** the substitute. Keeps Phase 2 small.
     Switch to `ReduceToOneB1` if traffic shape matters for the experiment.

2. **`ttnn.all_reduce` doesn't support `cluster_axis=both`** — we'd need
   two passes (`cluster_axis=0` then `cluster_axis=1`) for a true 8→8
   all-reduce, then read from one. The "read one" step is just
   `ttnn.get_device_tensors(...)` on the appropriate coord; no extra CCL.

3. **`ttnn.broadcast` requires `FABRIC_2D` (or `FABRIC_2D_TORUS_Y`).** Phase 2
   tier 1 (SINGLE_BH 8×4 mesh) needs the right fabric config. Existing
   `test_broadcast_exabox.py::test_broadcast_8x4` uses `FABRIC_2D`; reuse
   that.

4. **Per-rank submesh on tier 2/3.** When running under `tt-run` with the
   full 16x4 mesh device, each rank already gets its own submesh via the
   rank binding's `TT_VISIBLE_DEVICES` and the conftest's
   `bh_2d_mesh_device_context`. The test should not have to call
   `mesh_device.create_submesh(...)` — it just opens its `(4, 2)` mesh
   directly, like `test_single_pod_pipeline.py` already does. Confirm
   this assumption when wiring tier 2.

5. **Sub-device worker setup.** `ttnn.broadcast` needs a worker sub-device.
   `ttnn.all_to_all_combine` may need one (depends on op). The standalone
   test will need the same sub-device boilerplate as
   `test_broadcast_exabox.py`. Reuse that pattern.

6. **`test_all_to_all_dispatch_exabox.py` uses sharded inputs;** the chain
   in our test wants to thread one tensor through many ops with consistent
   sharding. May need to re-shard between steps. Plan: start with
   non-sharded interleaved memory configs everywhere and only add sharding
   if a primitive insists on it.

7. **Verifying correctness without MoE math.** Each CCL step has a
   well-defined torch golden:
   - `bcast`: every device should have the sender's tensor
   - `all_reduce`: every device should have the elementwise sum
   - `a2a_dispatch`: per-device output is the input pre-routed by
     `expert_indices`
   - `a2a_combine`: per-device output is the inverse of dispatch
   - `reduce-to-one` substitute (`all_reduce` then read one): one device has
     the full sum
   We chain these with checks between each step to localize regressions.

8. **Seeded randomness across MPI ranks.** Use `torch.manual_seed(0)` at the
   start of each helper that generates inputs (we already learned this
   lesson with the broadcast/all_reduce exabox tests).

## Roadmap dependency

| Phase | Depends on |
|---|---|
| 1 (plan + skeleton) | nothing |
| 2 tier 1 (SINGLE_BH) | Phase 1 |
| 2 tier 2 (DUAL_BH) | Phase 2 tier 1 passing |
| 2 tier 3 (single-pod) | Phase 2 tier 2 passing + ability to run 16-rank `tt-run` (already validated) |
| 3 (`FakeMoeDecoderStage`) | Phase 2 tier 3 passing |
| 4 (pipeline integration) | Phase 3 |
| 5 (docs) | Phase 4 |

## Definition of done

`test_single_pod_pipeline_fake_moe.py` runs on the 4-host single-pod
configuration with `tt-run`, exits 0, and passes both parametrize cases
(`prompt_len=1`, `prompt_len=8`). The end-to-end log shows:

- 16 ranks bring up the mesh
- `Pipeline started, waiting for all stages to complete` on every rank
- Rank 0 prints `Rank 0 decode complete; generated 4 tokens`
- All ranks rendezvous at `model_pipeline.terminate()`

Once that lands, the original `test_single_pod_pipeline.py` is left as-is
(still gated on the model team fixing the MoE CBDescriptor bug); the
fake-MoE test serves as both a CCL stress test and a runnable pipeline
smoke test in the meantime.
