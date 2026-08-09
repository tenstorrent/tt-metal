# Resuming AttnRes on Galaxy `(8, 4)`

Everything you need to pick this up on a Blackhole Galaxy without re-deriving it. The op is
green on LoudBox `(2, 4)`; `(8, 4)` has never run. Read §1 and §2 before touching anything —
§2 contains a trap that lets a Galaxy run *pass* while measuring the wrong topology.

Sibling docs: `PHASES.md` (status), `ROOFLINE.md` (the bandwidth model and its amendments),
`DISTRIBUTION.md` (why the mapping is what it is), `bringup_log.md` (every measurement and
refutation), `API_SPEC.md` (the tensor contract).

---

## 1. Where things stand

- **Branch:** `nmilicevic/bringup/kimi-k3-attnres-2026-07-30`, 13 commits off `main` @ `6d526e8d61d`.
- **State:** 239 tests collected; the 109 CPU cases pass. 11 of 12 phases closed; Phase 11
  (PP boundary) not started. The last full LoudBox run predates the reference rework — the
  device rungs need re-running before the pass count is quotable again.
- **Machine so far:** LoudBox, 8 × Blackhole. Meshes run: `(1,1)`, `(8,1)`, `(2,4)`. `(4,2)` skipped deliberately.
- **What the op is:** Kimi K3 attention residuals — `v = cat(block_residual, prefix_sum)`,
  RMS-normed keys against raw values, `α = softmax(rsqrt(mean(v²)+eps)·⟨q,v⟩)`, `out = Σ αᵢvᵢ`.
  Validated against `reference/attn_res_reference.py`, an unfolded fp64 ground truth pinned by
  closed forms rather than by any other implementation, which in turn is checked against
  upstream's own read vendored verbatim in `reference/hf_attn_res.py`. It is **the op only** —
  not wired into a model, no real K3 weights, prefill only.

Files that matter:

| path | what |
|---|---|
| `tt/attn_res.py` | the op — `TtAttnRes`, 577 lines |
| `tt/attn_res_stream.py` | the residual-stream bookkeeping the 93-layer walk drives |
| `torch_functional/attn_res.py` | the reference; `NUM_LAYERS=93`, `BLOCK_SIZE=12`, `EPS=1e-5` |
| `reference/attn_res_reference.py` | the unfolded fp64 ground truth — the root of the ladder |
| `reference/hf_attn_res.py` | upstream's read, verbatim — the external anchor. Kimi K3 License, see §8 |
| `tests/test_tt_attn_res_distributed.py` | **the file you edit for `(8,4)`** |
| `tests/perf/test_attn_res_perf.py` | the perf harness; logs, asserts nothing |
| `ttnn/cpp/ttnn/operations/experimental/reduction/fast_weighted_reduce_nc/` | the fused mixture kernel |

---

## 2. The topology trap — read this first

`DISTRIBUTION.md` §4 says "Galaxy prefill is `[LINE, RING]`" and cites
`test_prefill_block.py:666-673`. **That citation is for a `(4,4)` sub-torus**, which needs
`TT_VISIBLE_DEVICES` (16 chips) plus
`TT_MESH_GRAPH_DESC_PATH=…/single_bh_galaxy_subtorus_x4_graph_descriptor.textproto`. It is not
a full-Galaxy config. The full-Galaxy equivalents live in
`models/demos/deepseek_v3_d_p/tests/conftest.py:100-135` (`torus-x-8x4`, `torus-xy-8x4`).

**A topology tuple does not create a ring.** `ttnn.all_reduce` passes your request through
`get_usable_topology` (`ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp:42`), which
**silently demotes Ring → Linear** when the fabric cannot wrap that axis
(`ccl_common.cpp:149-164`). No warning, no error. So on Galaxy you can pass
`topology=[Linear, Ring]`, watch every PCC gate pass, and be measuring a line.

Whether axis 1 — AttnRes's TP axis, the only axis it ever communicates on — can wrap is
decided by `FabricConfig`, per `get_axis_topology` (`ccl_common.cpp:126-148`):

| `fabric_config` | axis 0 (SP, 8) | axis 1 (TP, 4) | does AttnRes benefit? |
|---|---|---|---|
| `FABRIC_1D` ← **what all current tests use** | Linear | Linear | — |
| `FABRIC_1D_RING` | Ring | Ring | yes |
| `FABRIC_2D` | Linear | Linear | — |
| `FABRIC_2D_TORUS_Y` | Ring | Linear | **no** — ring on the axis the op never uses |
| `FABRIC_2D_TORUS_X` | Linear | **Ring** | yes — this is the `[LINE, RING]` config |
| `FABRIC_2D_TORUS_XY` | Ring | Ring | yes |

Two consequences:

1. **`FABRIC_2D_TORUS_Y` is the wrong config for this op**, even though it is the analog's
   natural Galaxy choice — its ring is on the SP axis, and AttnRes moves zero bytes there.
   To ring the collective you need `TORUS_X` or `TORUS_XY`.
2. **Assert the topology you think you have.** `ttnn.get_usable_topology` is exposed to
   Python for exactly this (`ccl_nanobind.cpp:40-74`):

   ```python
   assert ttnn.get_usable_topology(stats, ttnn.Topology.Ring, cluster_axis=1) == ttnn.Topology.Ring
   ```

   Put that in the Galaxy perf test before any ring number is recorded. Without it, §4's
   "a ring halves the fabric term" is unfalsifiable.

---

## 3. What has to change in the code

**The op itself: probably nothing.** `TtAttnRes` was written for this and the defaults line up:

- `sp_axis=0, tp_axis=1` → on `(8,4)` that is SP=8, TP=4. Correct as-is.
- `topology` is already **one entry per mesh axis**, with an assert on its length
  (`tt/attn_res.py:155-159`). Default `[Linear] * len(mesh_shape)`.
- `hidden_size` is the **global** `d`; `shard_width = d // tp_factor`.

**The TP factor is 4 on both `(2,4)` and `(8,4)`.** That is the single most useful fact for
planning this port, and it means:

- `shard_width` stays **1792** — so the `ONE_PASS_SQUARES_MAX_WIDTH` threshold (177 tiles =
  5664, `tt/attn_res.py:61`) resolves the same way and the one-pass statistics path stays on.
- The all-reduce's `ring_size` is the extent of axis 1, so it stays **4**. Same payload per
  rank, same algorithm selection.
- Only the SP axis grows, 2 → 8, and SP carries no traffic.

**Prediction, stated so it can be wrong:** correctness ports unchanged, and the only genuinely
new variable is **8 concurrent TP rows contending for fabric instead of 2**. If a Galaxy
number regresses, contention is the first hypothesis, not the mapping.

**Test edits, in order:**

1. `tests/test_tt_attn_res_distributed.py:42` — `MESH = (2, 4)` is a module-level constant, and
   `on_mesh` (lines 49-54) is a single-entry `parametrize`. Either flip the constant or, better,
   make `on_mesh` multi-entry so LoudBox and Galaxy both collect. Note `test_tp_forward_matches_torch`
   asserts `(op.sp_factor, op.tp_factor) == MESH`, so the constant is load-bearing in two places.
2. `tests/perf/test_attn_res_perf.py:59-64` — `PLACEMENTS` / `PLACEMENT_IDS`. Add `((8,4), FABRIC)`.
   Several tests hard-code `[((2, 4), FABRIC)]` inline (lines 201-203, 282-284, 357-359) — grep
   `mesh-2x4` and fix all of them, not just the shared list.
3. **Gate collection on hardware.** `pytest.mark.requires_mesh_topology(mesh_shape=(8,4), topology="mesh-8x4")`
   is the repo idiom, but it is **not a repo-wide marker** — it is registered in
   `models/demos/deepseek_v3_d_p/tests/conftest.py:194` and implemented at `:249-270`. This module
   has no `conftest.py`, so using it means adding one that copies that hook. Without it, a Galaxy
   param collected on LoudBox fails at device open rather than skipping.

---

## 4. Commands

Absolute paths — this tree's venv is not on `PATH`:

```bash
cd /localdev/nmilicevic/tt-metal

# incremental rebuild (needed for the fused op; never invoke cmake/ninja directly)
bash build_metal.sh

# the whole suite — 183 tests on LoudBox
PYTHONPATH=/localdev/nmilicevic/tt-metal /localdev/nmilicevic/tt-metal/python_env/bin/python \
  -m pytest models/experimental/kimi_k3_attn_res/tests/ \
            tests/ttnn/unit_tests/operations/reduce/test_fast_weighted_reduce_nc.py -q

# just the mesh gates
… -m pytest models/experimental/kimi_k3_attn_res/tests/test_tt_attn_res_distributed.py -q

# perf: -s is REQUIRED. `_report` writes through loguru; without -s you get nothing.
… -m pytest models/experimental/kimi_k3_attn_res/tests/perf/test_attn_res_perf.py -s -k mesh-8x4
```

Gotchas that cost time on LoudBox:

- **Collectives need the fabric initialized.** Without
  `device_params={"fabric_config": ttnn.FabricConfig.FABRIC_1D}` the op dies in
  `control_plane.cpp:2186` — it does not return wrong numbers.
- **`TT_ENABLE_UNITY_BUILD` hides new filenames from the build log.** A new kernel that looks
  uncompiled may just be invisible.
- Stale `_ttnn.so` and uninitialized submodules cost Phase 0 a full cycle:
  `git submodule update --init --recursive` then a full build.
- `generated/` and `__pycache__` under this module are gitignored; don't try to commit them.

---

## 5. The verification ladder for `(8, 4)`

Run in this order. Each rung fails on something the one below it cannot see.

| # | gate | pass criterion | fails on |
|---|---|---|---|
| 1 | `test_tp_forward_matches_torch`, `S=0` | PCC ≥ 0.9999 | placement, not the reduction — `S=0` communicates nothing |
| 2 | same, `S ∈ {1,8}` × `d ∈ {256,7168}` | PCC ≥ 0.9999, rel err ≤ 2e-2 | a missing reduction, a wrong global `d`, an axis swap |
| 3 | `test_sequence_axis_communicates_nothing` | **max\|Δ\| == 0**, exactly | a collective pointed at the SP axis or spanning both. The only gate that distinguishes "reduced on the right axis" from "reduced on an axis" |
| 4 | `test_statistics_reduction_is_load_bearing` | PCC **< 0.9999** with `_reduce_stats` stubbed to identity | a blind gate. On `(2,4)` stubbing it gives 0.5757 |
| 5 | `test_tp_depth_walk`, `T ∈ {64,256}` × `fold_stats ∈ {F,T}` | device PCC ≥ torch-bf16 − slack | a reduction that is *close* rather than correct, compounded over 186 chained reads. A topology mismatch shows up here as a **hang**, not a number |
| 6 | `test_tp_split_matches_forward` | PCC ≥ 0.9999 at all 24 sites | the split form's 49 collectives vs the direct form's 24 |

Only after 1-6 are green does a perf number mean anything.

---

## 6. Numbers to check the predictions against

All LoudBox `(2,4)`, traced, `T = 5120`, two-run means. Galaxy has its own predictions in
`ROOFLINE.md` §3-§4; the point of listing both is that you can falsify them.

**Measured on `(2,4)`:**

| quantity | value |
|---|---|
| per forward, split form, fused mixture | **153.6 ms** (from 380 ms traced at the start of the perf loop) |
| per forward, direct form | 216.2 ms |
| DRAM floor, one pass over the 79 MiB candidate tensor | **228.2–229.1 µs**, four independent ways |
| the mixture, composed `mul`+`sum` | 687.1 µs (3.01× floor) |
| the mixture, `fast_weighted_reduce_nc` | **257.3 µs** (1.13× floor) |
| statistics all-reduce, `links=1`, `S=8` | 235.9 µs — **2.7× worse than §4 predicted** |
| depth PCC, 186 chained reads, `d=7168` | 0.9999500 (torch-bf16: 0.9999741) |

**Predicted for `(8,4)`, never run:**

| quantity | prediction | source |
|---|---|---|
| DRAM floor per read, `S=8`, `T=5120` | 248 MB → **484 µs** | `ROOFLINE.md` §3 |
| collective, RS+AG critical path, `links=2` | 44.2 µs | §4 — **and §4's LoudBox column was already off 2.7×, so treat this as an upper-bound sketch, not a target** |
| collective as a share of the DRAM floor | 9.1% (vs 4.6% on `(2,4)`) | §4 |
| statistics as a share of DRAM traffic | 0.595% | §4 |

Two live cautions on the predictions:

- **§4's fabric model is refuted on LoudBox.** A collective reaching 18-25% of fabric peak is
  **core**-limited, not link-limited: `all_reduce` runs on **two worker cores**. Expect the
  Galaxy column to be wrong in the same direction.
- **`num_links` stays 1**, and this was measured, not assumed. After the statistics fold there
  is no payload left for a second link — it buys 4.7 µs per read. On Galaxy the fabric is
  contended by dispatch/combine, which makes the layout the better trade there too. Re-measure,
  but don't start by raising `num_links`.

---

## 7. Known-open, in the order I'd do them

1. **`(8,4)` correctness** — §5's ladder. Cheap, and it derisks everything else.
2. **`(8,4)` perf, with the topology asserted** — §2. Includes the first honest ring measurement.
3. **Phase 11, the PP boundary** — the `(1+S)·d` canonical layout round-tripped through a
   `MeshSocket` pair. This is what the pipeline-of-Galaxies goal actually needs, and it has
   never run. `PIPELINE.md` doesn't exist yet.

   Priced (workbook, `PP BOUNDARY`): a plain boundary carries one activation plane, AttnRes
   carries `1+S`, because a read at layer 90 still mixes the snapshot sealed at layer 0. At a
   balanced 2-stage cut after layer 46, `S = 4` → 5 planes, i.e. **9.2 MB per chip / 294 MB per
   mesh of new traffic**. The payload therefore grows with cut *depth*, which standard PP
   schedulers do not assume.

   It is nonetheless not a problem, for one reason: a sealed snapshot is **write-once and
   immutable from the instant it is sealed**, so it can be sent eagerly and overlapped with the
   layers between its seal and the cut. Snapshots seal at 0/12/24/36 against a cut at 46, so
   their windows are 46/34/22/10 layers and the last one binds at 10/93 of a forward. Break-even
   forward time — no `T_fwd` estimate needed — is **0.85 ms per-device**, or **27.3 ms** if all 32
   chips funnel through one exit link the way the b1 decode path does. AttnRes alone measures
   68.4 ms, so even the funnelled path clears by ≥2.5×.

   The risk is not bandwidth, it is **scheduling**: dump all `S` planes *at* the cut instead of
   eagerly and the funnelled path serializes 11.7 ms per boundary, 17% of the op. Eager send is a
   requirement of the design. Decode is a non-issue either way (`T = 1` → 3.6 KB a plane), and
   snapshot residency is unchanged by PP — the op already holds `(1+S)` planes, ≤20.6 MB/chip.
4. **Hoist the collective's global semaphores** — 481 µs enqueue against a 152 µs baseline;
   per-call global-semaphore creation is the suspect. The analog hoists it with
   `create_global_semaphores` (`tt_ccl.py`). Untraced-only, so it moves none of §6's numbers.
5. **The remaining ~3× of `ROOFLINE.md` §7** — fold the two `d`-reductions and the
   cross-candidate softmax into the mixture's pass. Not a continuation of Phase 10: that op
   changed no arithmetic, this one owns the statistics all-reduce and the fp32 score chain that
   took four iterations to get right in composed form.
6. **Let the caller pick the read form per block.** The split form is **9% slower at `S=1`**
   (710.9 µs vs direct's 649.6); crossover at `S+1 = 2.30`. 24 of the schedule's 186 reads sit
   on the direct side. Currently a global choice.
7. **Decode (`T=1`), real K3 weights, real module outputs.** `apply_module` is an elementwise
   `h ⊙ w` — it exercises the residual bookkeeping, not KDA/MLA/MoE traffic.

---

## 8. Two things to raise with people, not with the device

- **The vendored upstream reference is not Apache-2.0.** `reference/hf_attn_res.py` is
  `_apply_attn_res` copied verbatim from `modeling_kimi_linear.py` (`moonshotai/Kimi-K3`,
  upstream sha256 `9e3564c7…fff44a`, lines 1075-1088 of 1314), with the upstream LICENSE
  alongside it as `reference/LICENSE-Kimi-K3`. Upstream dual-licenses: the DeepSeek-V3-adapted
  MLA / MoE-gating / sparse-MoE parts are Apache-2.0, everything else is the Kimi K3 License.
  `_apply_attn_res` is not DeepSeek-adapted, so it lands on the Kimi K3 arm — source-available,
  with MaaS revenue and attribution conditions above certain thresholds. The header and folder
  LICENSE follow the repo's existing pattern for this
  (`models/demos/t3000/llama2_70b/reference/llama/llama31_8b/*.py`, SPDX
  `LicenseRef-LICENSE-FILE`), and there is no license or SPDX pre-commit hook to consult.
  **Whether it can merge in this form is a call for people, not for me.** If it cannot, the
  fallback is cheap: delete the file and rung 0b, and the ladder still stands on
  `reference/attn_res_reference.py` — it loses the external anchor, not its root.
- **For mvasilijevic:** `modeling_kimi_linear.py:520-521` allocates `A_log` as
  `[num_heads] = [96]`, while the checkpoint stores `F32 [128]` = `head_dim`. Unrelated to
  AttnRes, found while reading KDA.
