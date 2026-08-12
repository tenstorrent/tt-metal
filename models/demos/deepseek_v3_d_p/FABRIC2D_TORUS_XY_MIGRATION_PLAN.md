# Kimi K2.6/K2.7, Kimi K3, and GLM-5.2 prefill migration to Fabric2D/TorusXY

Status: implementation in progress

Prepared: 2026-08-11

Repository baseline inspected: `076c3bf5ae997a702cac58ef061cec544f4d9535` (`main`)

CI baselines inspected: [Blaze prefill run 31461379870](https://github.com/tenstorrent/tt-metal/actions/runs/31461379870) at `16cb33980adddc6559fc9d2212a6fde22558fbfc`, and [Blackhole e2e run 31472546803](https://github.com/tenstorrent/tt-metal/actions/runs/31472546803) at `02df4aaef6b2908324ef7caa4261065542127dae`

## 1. Outcome and non-negotiable invariants

The migration is complete only when all of the following are true:

1. Every production Galaxy prefill launch for Kimi K2.6, Kimi K2.7, Kimi K3, and GLM-5.2 opens `ttnn.FabricConfig.FABRIC_2D_TORUS_XY` and uses a matching `[RING, RING]` mesh graph descriptor.
2. Production collectives derive `(SP topology, TP topology) = (Ring, Ring)` from the active fabric. No production call site silently defaults either axis to `Topology.Linear`.
3. Scoped tests do not open `FABRIC_1D` or `FABRIC_1D_RING`. Tests that need communication use Fabric2D; tests that are truly single-device/local use fabric disabled.
4. Every test running on hardware with a verified physical wrap uses a torus fabric and Ring on every wrapped collective axis. A test may use Linear over Fabric2D only when the target hardware or descriptor does not provide that wrap, or while a documented ring-specific deadlock is being fixed.
5. Production behavior is gated by production-shaped Galaxy tests. Op-level tests are not the release gate for topology, full-model correctness, determinism, trace replay, KV layout, or performance.
6. Fabric choice, mesh descriptor, and CCL topology fail closed on mismatch. A mismatch must raise before model construction rather than hang in fabric bring-up or a collective.
7. The CI log for each production Galaxy job records and asserts all three values: active FabricConfig, mesh descriptor/dim types, and per-axis CCL topology.
8. The migration adds no test files, test functions, or parameterized configurations. Existing cases may be reassigned across K2.6/K2.7, migrated one-for-one, or pruned when higher-level coverage supersedes them, but no matrix may grow.
9. FabricConfig is the only topology-selection input in tests and production. Tests may parameterize link count, but derive per-axis CCL topology from the active/configured fabric rather than carrying a separate topology parameter.

Terminology in this plan:

- **Fabric2D mesh** means `FABRIC_2D` with `(Linear, Linear)` collectives on an unwrapped 2D physical mesh. This is the LoudBox target.
- **TorusXY** means `FABRIC_2D_TORUS_XY` with `(Ring, Ring)` and a descriptor whose two device dimensions are both `RING`. This is the production Galaxy target.
- **Production-shaped test** means an 8×4 Galaxy module/full-model test using the same fabric, topology, link count, model dimensions, checkpoint/cache format, and important execution mode as serving.
- “Remove an op test” below means remove it from the production CI gate after equivalent higher-level coverage is proven. Small diagnostic tests may remain as developer tests, but must obey the no-Fabric1d invariant.

## 2. Baseline findings

### 2.1 Production runner is not fail-safe today

`models/demos/common/prefill/runners/runner_utils.py::open_mesh_device` currently defaults to:

```text
SP <= 8  -> FABRIC_1D
SP > 8   -> FABRIC_2D
```

This means a normal 8×4 single-Galaxy runner silently opens Fabric1d unless `PREFILL_FABRIC_MODE` is supplied by its launcher. This is visible in the referenced Blaze run: the Kimi producer/runner test opened `FABRIC_1D`.

The seven checked-in production rank bindings are `pipeline_prefill_request_{1,2,4,8}rank.yaml` plus `pipeline_prefill_request_{1,2,4}rank_d2h_ack.yaml`. They now all set `PREFILL_FABRIC_MODE=2d_torus_xy`; the one-rank forms use the single-Galaxy Ring/Ring descriptor and the multi-rank forms use the corresponding connected-Galaxy descriptors. Direct runner invocations and a missing environment value are also covered by the same fail-closed topology policy.

The runner also imports `DEFAULT_MODEL = "kimi_k2_7"`. A missing model field or a rank-local environment propagation bug can therefore silently select a scoped production model. Production must require an explicit model on every rank and fail closed; the default can remain only for an explicitly marked developer path. GLM-5.1 is registered, but is outside this K2.6/K2.7/K3/GLM-5.2 migration scope.

### 2.2 The referenced Blaze run passed, but many Kimi/GLM jobs exercised Fabric1d

All Kimi/GLM jobs listed below passed in run 31461379870. The active-fabric lines in their logs show that passing does not yet imply production-topology coverage.

| Blaze job | Fabric observed in log | Migration required |
|---|---:|---|
| Kimi MLA | `FABRIC_1D` | Replace the `line` selector with 8×4 TorusXY/Ring coverage. |
| Kimi MoE | `FABRIC_1D` | Add/select the TorusXY MoE param and Ring on SP and TP. |
| Kimi Prefill Block | `FABRIC_1D` | Select the TorusXY block case, including dense and MoE layers. |
| Kimi Chunked Padded Accuracy, trace/no-trace | `FABRIC_2D` | Promote to TorusXY; retain both trace modes. |
| Kimi Chunked Perf, trace/no-trace | `FABRIC_1D` | Move perf baseline to production TorusXY. |
| Kimi DFlash drafter | `FABRIC_1D` | Move to TorusXY or remove from this scope if DFlash is not a production Kimi prefill path. |
| Kimi Prefill Runner | `FABRIC_1D` | Make this the first full runner TorusXY gate. |
| GLM-5.2 sparse/DSA MLA | `FABRIC_2D` | Promote to TorusXY and Ring-aware per-axis gathers/reshards. |
| GLM-5.2 MoE | `FABRIC_1D` | Add/select TorusXY MoE coverage. |
| GLM-5.2 Prefill Block | `FABRIC_2D` | Promote to TorusXY, preserving full/shared indexer-layer coverage. |
| GLM-5.2 Chunked perf | `FABRIC_2D` | Promote and recalibrate on TorusXY. |
| GLM-5.2 Chunked accuracy | `FABRIC_1D` | Replace the implicit `mesh-8x4` case with explicit TorusXY. |
| KV cache table | `FABRIC_1D` | Move all model KV-table validation into the TorusXY producer/runner gate. |
| MoE Gate | `FABRIC_1D` | Replace broad `mesh-8x4` selector with the explicit TorusXY ID. |
| DeepSeek MLA / MLA chunked / block anchors | `FABRIC_2D` | Use as the initial TorusXY control while migrating Kimi/GLM. |

The overall Blaze run failed only in the unrelated MiniMax-M3 perf row. The Kimi/GLM results are therefore usable as pre-migration correctness/time baselines, not as evidence of TorusXY correctness.

### 2.3 LoudBox CI currently spends substantial time on Fabric1d variants

The relevant jobs in run 31472546803 all passed:

| LoudBox job | Result | Current problem |
|---|---:|---|
| `bh_lb_DeepSeek_PREFILL_PERF` | child perf tests passed | Main perf selections include Fabric1d MoE, MLA, and block cases; only one block sibling is Fabric2D. |
| `bh_lb_DeepSeek_PREFILL_OP_TESTS` | 20 host tests plus 159 device tests passed | Logs repeatedly initialize `FABRIC_1D` and `FABRIC_1D_RING`; the job is the largest source of obsolete topology coverage. |
| `bh_lb_DeepSeek_DSA` | 8 GLM + 7 DeepSeek cases passed | Already Fabric2D; retain as the local sparse-MLA gate and remove the line/ring modes from the suite. |
| `bh_lb_DeepSeek_PREFILL` | 84 PCC, 4 MLA, 13 block, and 5 block-loop cases passed | The PCC and loop selectors still include `linear-8`, `line`, and `not fabric2d`; convert to Fabric2D and reduce duplication once Galaxy coverage lands. |

This run is the concrete before-state for CI duration, selected tests, and expected skip counts.

Five additional Blackhole jobs in the same pipeline also need an explicit disposition:

| Job | Hardware/role | Required disposition |
|---|---|---|
| `bh_qb_DeepSeek_PREFILL` | 4-chip QuietBox PCC/cache | Convert every communicating case to a 2×2 Fabric2D profile; make host-only cache checks fabric-free. |
| `bh_p150_DeepSeek_PREFILL_OP_TESTS` | 1 chip | A communicating Fabric2D test is physically impossible. Move communicating coverage to the 4/8-chip jobs or Galaxy; retain only truly single-device tests with fabric disabled. |
| `bh_p300_DeepSeek_PREFILL_OP_TESTS` | 2 chips | A two-dimensional communicating mesh is not available. Move communicating coverage to the 4/8-chip jobs or Galaxy; retain only local tests with fabric disabled. |
| `bh_qb2_DeepSeek_PREFILL_OP_TESTS` | 4-chip QuietBox | Convert retained diagnostics to 2×2 Fabric2D and remove its 1D topology matrix. |
| `bh_qb2_DeepSeek_D2D_SOCKET_SYNC` | 4-chip QuietBox, currently `linear-2x2` | Keep as an LB Fabric2D program-cache diagnostic; it does not replace connected-Galaxy TorusXY D2D validation. |

The user invariant is not relaxed for small machines: the 1/2-chip jobs must stop running scoped Fabric1d tests rather than being grandfathered. The DeepSeek-V3 B1 `fabric_1d` row in `demo_sp_release_tests.yaml` is not one of the requested models and is out of scope for this plan.

### 2.4 This local machine

The current environment exposes eight Blackhole P150b devices: one LoudBox-class 8-chip machine. The release build and matching Python extension have now been rebuilt successfully. No Kimi/GLM/DeepSeek production model directories were found below `/mnt` in this container.

Consequences:

- This machine can validate 4×2 and 2×4 Fabric2D paths, random-weight module correctness, program cache behavior, and host/device references that do not require staged model assets.
- It cannot validate the 8×4 Galaxy topology, a two-axis Galaxy wrap, TorusXY, 32-device memory pressure, Galaxy link routing, or Galaxy performance.
- Real-weight/full-depth tests require the model/cache/trace mounts to be made available first.
- The original DFlash cases are 8×4. They remain Galaxy-only; do not create a 2×4 surrogate merely to run them here.
- The D2D socket diagnostic's retained 2×2 case is a four-device QuietBox case. Blackhole LoudBox requires tests to consume all eight visible devices, so this machine collects it as a hardware skip; it is not part of the LoudBox execution manifest.
- The GLM-5.2 shared-layer `ReuseIndexer` cache case retains its original 8×4 shape and is Galaxy-only. The sparse cache-only round-trip and missing-cache fallback diagnostics remain on their pre-existing 4×2 rows for local validation.
- Do not synthesize a local `FABRIC_2D_TORUS_XY` test by changing only the enum. There is no checked-in LoudBox Ring/Ring descriptor establishing that both physical axes wrap; an enum/descriptor mismatch is a hang risk.

## 3. Architecture decision: explicit topology profiles

Do not solve this by changing every `Topology.Linear` token to `Topology.Ring`. Fabric wiring and collective topology are coupled per axis, and several models communicate on both SP and TP.

Introduce one resolved topology profile, constructed before opening the device and passed/queried consistently:

```text
TopologyProfile
  fabric_config
  mesh_graph_descriptor
  sp_topology
  tp_topology
  num_links_by_axis
  reliability_mode
  production (bool)
```

Required profiles:

| Profile | Hardware | Fabric | SP topology | TP topology | Use |
|---|---|---|---|---|---|
| `bh_loudbox_fabric2d` | 8-chip P150b LB, 4×2 or 2×4 mesh | `FABRIC_2D` | Linear | Linear | Local correctness/bring-up only. |
| `bh_galaxy_torus_xy` | 32-chip BH Galaxy, 8×4 mesh | `FABRIC_2D_TORUS_XY` | Ring | Ring | All production and production-shaped Galaxy CI. |
| `disabled` | CPU/single-device local test | disabled | N/A | N/A | Pure local ops that do not communicate. |

Rules for the implementation:

- Resolve the effective descriptor path exactly as the launcher/device setup will: require `TT_MESH_GRAPH_DESC_PATH` for production, canonicalize it, open it, and parse the selected mesh's `device_topology.dim_types`. Do this before device creation. Do not rely on descriptor auto-discovery for a torus production job.
- The descriptor is authoritative about which axes wrap. The profile validator compares its `dim_types` with the selected FabricConfig before `open_mesh_device`: TorusXY requires `[RING, RING]`; TorusX/Y require their matching single wrapped axis; plain Fabric2D must not masquerade as TorusXY. A missing descriptor path in a production torus profile is a hard error.
- `per_axis_topology()` remains the single mapping from active FabricConfig to CCL topology, but production setup asserts it returns `(Ring, Ring)`.
- `PREFILL_FABRIC_MODE` remains an escape hatch for developer bring-up, not the source of a silent production default.
- Scoped adapters/manifests declare that their production profile is `bh_galaxy_torus_xy`. A production runner must reject `1d`, `1d_ring`, missing mode, or a non-ring descriptor for these models.
- Test IDs encode the real profile (`fabric2d-mesh-2x4`, `torus-xy-8x4`) so selectors cannot accidentally match a topology sibling.
- Use two links on both Galaxy axes unless a measured operation has a different production setting. Keep link count in the profile rather than scattering literals through tests.
- The checked-in `single_bh_galaxy_torus_xy_graph_descriptor.textproto` is 8×4 `[RING,RING]`, two channels, and `STRICT`. Use that reliability contract for production. Test params currently using `RELAXED_INIT` are diagnostics only; the release preflight and production rows must assert `STRICT` initialization so link failures cannot be hidden.
- Connected 2/4/8-Galaxy descriptors currently declare `RELAXED` mesh and inter-Galaxy channels for discovery-based placement. The runner must derive reliability from every descriptor channel: `STRICT_INIT` only when all mesh/connection policies are `STRICT`, otherwise `RELAXED_INIT`. It must never force the single-Galaxy strict policy onto a connected descriptor silently; promotion of connected descriptors to strict is separate Galaxy-validated work.

## 4. Production/runtime code migration inventory

### 4.1 Common runner and topology configuration

| File/area | Required change |
|---|---|
| `models/demos/common/prefill/runners/runner_utils.py` | Remove the `SP <= 8 -> FABRIC_1D` fallback for scoped production models. Resolve an explicit profile; add pre-open descriptor/fabric validation and post-open active-fabric assertion. Keep Fabric1d support only for unrelated models if necessary, behind a non-production compatibility path. |
| `models/demos/common/prefill/runners/prefill_runner.py` | Log the resolved profile once per rank and fail if ranks disagree. Persist fabric/topology fields in run summaries. Ensure cleanup still disables fabric after failure. |
| `models/demos/common/prefill/adapter.py` | Add a lightweight adapter capability/requirement for production topology, or an equivalent runner-side model policy table. Register Kimi K3 only after its full runtime exists. Eliminate silent production fallback to `DEFAULT_MODEL = "kimi_k2_7"`; require an explicit model on every production rank. Keep GLM-5.1 explicitly out of this migration scope. |
| `models/demos/deepseek_v3_d_p/tt/runners/manifests/kimi26.json` | Declare/require the production TorusXY profile; do not rely solely on the launcher. |
| `.../manifests/kimi27.json` | Same as K2.6; add a checkpoint-specific Galaxy runner gate. |
| `.../manifests/glm52.json` | Same; preserve 78 layers, full/shared indexer rules, and `KV_ONLY_LAST_LAYER=0`. |
| new `.../manifests/kimi3.json` | Add only after MLA, KDA, AttnRes, MoE, block, transformer, KV cache, and runtime contracts are integrated. Require TorusXY. |
| `pipeline_prefill_request_{1,2,4,8}rank.yaml` | TorusXY request-mode bindings: validate the single-Galaxy and connected-Galaxy forms in CI, including at least a two-rank D2D smoke. |
| `pipeline_prefill_request_{1,2,4}rank_d2h_ack.yaml` | TorusXY D2H-ack bindings: validate descriptor/profile agreement and acknowledgement flow at each retained rank count. |
| `pipeline_prefill_{2,4,8}galaxy_connected_mesh_graph_descriptor.textproto` | Verify every per-Galaxy mesh is `[RING,RING]`, intermesh links remain routable under TorusXY, and channel count matches two-link expectations. |
| `tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto` | Treat the existing 8×4 Ring/Ring, two-channel, `STRICT` descriptor as the single-Galaxy production contract; do not duplicate it in test data. Validate physical cabling separately. |
| `models/demos/common/prefill/docs/*.md` | Replace examples that pair a torus descriptor with `PREFILL_FABRIC_MODE=2d`; document the three supported profiles and the fail-closed rule. |

### 4.2 Model CCL plumbing

| File/area | Required change |
|---|---|
| `tt/tt_ccl.py` | Keep the Fabric2D Torus X/Y/XY per-axis map; delete production use of `default_topology`; add a strict helper that asserts TorusXY returns Ring/Ring. Validate every mapping and mismatched-profile rejection through existing host/import/config paths rather than adding tests. |
| `tt/runners/adapters/mla.py` | Continue deriving topology from the active fabric, but assert the adapter's production requirement. Pass a per-axis tuple to every runtime component. |
| `tt/tt_prefill_runtime.py` | Make the FabricConfig-derived topology/profile required in `TtPrefillRuntimeConfig`; remove the Linear default for production construction. |
| `tt/tt_prefill_transformer.py` | Remove the Linear default. Consume the adapter's per-axis topology, or derive it from the already active FabricConfig for non-runtime component use; select TP topology for LM head/norm and pass the full tuple to blocks. |
| `tt/tt_prefill_block.py` | Remove the Linear default, derive omitted component topology from the active FabricConfig, and validate tuple order `(SP axis 0, TP axis 1)`. Dense FFN, MLA, norm, MoE, and final all-gather must use the correct axis element. |
| `tt/mla/mla.py` | Exercise Ring on both the SP ring-attention path and TP q/kv/wo collectives. Preserve the axis-order assertion. Add a runtime assertion that a requested Ring axis is physically wrapped. |
| `tt/tt_ffn.py`, `tt/tt_distributed_rms_norm.py`, `tt/tt_lm_head.py` | Remove accidental Linear/Ring defaults; consume the passed TP topology or derive the correct axis from active FabricConfig. |
| `utils/transformer_helpers.py` | Remove helper APIs that silently inject Linear; omitted topology derives from active FabricConfig. |
| `tt/dflash_prefill/tt_dflash_drafter.py` | If DFlash is part of production Kimi prefill, pass the TP Ring topology from the profile; otherwise split it from this migration's release criteria and CI rows. |

### 4.3 MoE-specific TorusXY work

| File/area | Required change |
|---|---|
| `tt/moe/tt_moe.py` | Ring on SP dispatch/combine and TP gate/pre-gather/post-reduce. Track the existing overlapped shared-expert Ring reduce-scatter deadlock as an explicit exception, not silent fallback. |
| `tt/moe/tt_dispatch.py`, `tt/moe/tt_combine.py` | Verify ring-aware dispatch/combine across the SP wrap edge, including first/last rank traffic, uneven expert/token routing, and one/two-link cases. |
| `tt/moe/tt_reduce.py`, `tt/moe/tt_shared_expert.py` | Validate TP Ring reduce-scatter/all-gather and persistent semaphore reuse. |
| `tt/moe/tt_moe_gate_prefill.py` | Run the TP all-reduce on Ring and preserve Kimi/GLM/K3 expert-count-specific numeric bars. |
| `tt/moe/init_helpers.py` | Centralize router payload/profile inputs so K2.6/K2.7/K3/GLM do not construct inconsistent fabric settings. |

Current exception requiring a dedicated work item:

- `TtMoe` forces the shared-expert TP reduce-scatter to Linear when it overlaps SP dispatch on a TP Ring, because concurrent wrap traffic can deadlock on EDM credits.
- This workaround is reachable in no-trace serving. Trace capture sets overlap false, so a trace-only pass does not exercise or clear the exception.
- Correctness bring-up should first run TorusXY with overlap disabled, proving all collectives can Ring, and then explicitly run the no-trace production mode with overlap enabled.
- `tests/pcc/mesh_configs.py` independently documents a multi-hop-over-wrap MoE dispatch hang. Preserve a bounded diagnostic comparison of TorusY versus TorusXY until the fault is localized, but neither X/Y-only topology is a production alternative. This hang must be cleared before Phase 3 promotion.
- Then fix the router/collective interaction and re-enable overlap with Ring. Production performance sign-off is not complete while the forced-Linear workaround remains, unless the product owner explicitly accepts this one collective-level exception with an owner and expiry while the fabric itself remains TorusXY.

## 5. Kimi K3 integration dependencies

Kimi K3 is not one merged, production-runnable stack on the inspected `main`. The topology work should land in common code first and be consumed by these active feature branches/PRs rather than independently reimplemented in each:

| Component | Current branch / PR | Files/topology work to carry forward |
|---|---|---|
| MLA | merged to `main` by [PR #52068](https://github.com/tenstorrent/tt-metal/pull/52068); integrated here at `a2cc0716f3c` | `reference/kimi_k3*`, `tt/mla/*`, K3 adapter, `test_mla.py`, MLA cache/perf tests. Its retained LB rows now use Fabric2D, its existing Galaxy rows use TorusXY, and topology is derived from FabricConfig. |
| LatentMoE | `ianastasijevic/kimi_k3_moe`, [PR #52453](https://github.com/tenstorrent/tt-metal/pull/52453) | 896-expert gate, latent projection, routed/shared expert, MoE/block/runtime/transformer integration. Its current Galaxy K3 MoE is Fabric2D; promote it to TorusXY and Ring/Ring. |
| KDA/delta attention | `kda-split/05-kimi-k3-model`, [PR #52799](https://github.com/tenstorrent/tt-metal/pull/52799) and its dependency stack | `tt/kda/*`, distributed layer, halo/affine ops, model tests, checkpoint tests, perf. Reassign existing distributed-model cases to Fabric2D LB and TorusXY Galaxy; keep pure reference tests fabric-free. |
| AttnRes | `nmilicevic/bringup/kimi-k3-attnres-2026-07-30`, [PR #52676](https://github.com/tenstorrent/tt-metal/pull/52676) | `tt/attn_res/*` and model/perf tests. Migrate its sole existing 2×4 LoudBox gate in place to Fabric2D and derive both ordinary and fused-op TP topology from FabricConfig. The component owns no 8×4 row; production Galaxy coverage must come from the future composed K3 gate rather than adding or reshaping a component configuration. |
| Full block/transformer/runner | not yet a single reviewed production gate | Compose MLA + KDA + AttnRes + LatentMoE, allocate all caches, add K3 manifest, and run full depth/checkpoint before calling K3 migrated. |

Merge/validation order for K3: common topology contract → MLA → KDA and AttnRes (independent where possible) → LatentMoE → block → transformer/runtime → full producer/runner → performance. Module PRs may land earlier, but none should introduce new Fabric1d coverage.

As of 2026-08-12, only the K3 MLA component is merged into `main`. LatentMoE PR #52453, KDA PR #52799, and AttnRes PR #52676 remain separate draft integration surfaces, and there is still no single composed K3 block/transformer/runner gate. Therefore the available MLA work can be migrated and validated now, but it cannot be used to claim `LB-F2D-K3` or production K3 completion.

## 6. Test parameter and file migration inventory

### 6.1 Central parameter sources

| File | Migration |
|---|---|
| `models/demos/deepseek_v3_d_p/tests/conftest.py` | Make `FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS` the only communicating mesh family. Keep LB 4×2/2×4 Fabric2D and the existing Galaxy 8×4 TorusXY param. Narrow/remove the CI-wide Galaxy torus skip only on a cabling-certified allocation after the physical-link preflight passes. Retire X-only/Y-only and 4×4 sub-torus production siblings after resolving the stale comment that calls TorusX production; retain them only as explicitly unscheduled diagnostics. |
| `tests/pcc/mesh_configs.py` | Replace `ALL_MESH_CONFIGS` with profile-based params. Remove all `FABRIC_1D` and `FABRIC_1D_RING` entries. Map retained local 2/4/8-chip cases onto their existing 2×2, 4×2, or 2×4 Fabric2D shapes; use disabled fabric for non-communicating 1-chip cases. Promote the existing TorusXY 8×4/two-link slot; do not add another link-count variant. Remove TorusY/TorusX siblings once their equivalent TorusXY production workload is identified. |
| test IDs and `-k` contracts | Stop using ambiguous selectors such as `mesh-8x4`, `8x4`, `line`, or `not fabric2d`. Use `fabric2d-mesh-2x4` locally and the exact existing `fabric2d-torus-xy-8x4-*link`/canonical profile ID in Galaxy CI. Today `mesh-8x4` does not select the TorusXY ID, so a mechanical rename can silently collect zero cases. Update every `EXPECT_NUM_TESTS` after collection-only verification and fail the command if collected or executed count is zero. |
| sparse MLA fabric selection | Replace `DS_SPARSE_FABRIC=line|ring|fabric2d` and priority fallback with `fabric2d|torus_xy`; default local tests to Fabric2D and require TorusXY in Galaxy production CI. In `tests/sparse_mla/test_sparse_mla.py`, delete `_topology_from_device_params`—it returns Linear for every fabric except `FABRIC_1D_RING` and would silently misconfigure TorusXY—and use `tt_ccl.per_axis_topology()` with the correct sparse-MLA cluster axis. |

### 6.2 Core model/integration tests

| Files | Planned destination |
|---|---|
| `tests/test_mla.py` | Remove line and Fabric1d-ring params for DeepSeek/Kimi/K3. LB: Fabric2D 2×4 random-weight single/chunked accuracy. Galaxy: TorusXY 8×4 max-sequence, determinism, metadata/trace as supported, and perf. |
| `tests/sparse_mla/test_sparse_mla.py` | Remove line/ring modes. LB: GLM-5.2 Fabric2D 2×4, including full and reuse layers. Galaxy: TorusXY 8×4 with Ring-aware gather/reshard and determinism. |
| `tests/sparse_mla/test_sparse_mla_{cache,ccl_perf,perf,vs_trace}.py` | Fabric-free for pure host/cache-schema tests; Fabric2D for LB device diagnostics; TorusXY only in certified production perf/trace rows. Delete the redundant Fabric1d 8×1 SP proxy from CCL perf; the full sparse-MLA Galaxy gate owns that SP-path coverage. Keep the remaining unscheduled CCL diagnostics on stable-ID unwrapped Fabric2D and never infer TorusXY from device count. |
| `tests/test_prefill_block.py` | Retain the existing DeepSeek LB 2×4 Fabric2D random dense + MoE smokes. Kimi/GLM have only 8×4 composition rows: preserve those shapes and move their composition proof, plus pretrained, long prompt, determinism, full/shared indexer, K3 mixed-attention, and perf-shaped cases, to TorusXY Galaxy. |
| `tests/test_prefill_block_loop.py` | Convert current `not fabric2d`/line cases to Fabric2D locally. Use the loop only as a short LB diagnostic after Galaxy block/transformer gates exist. |
| `tests/test_prefill_block_chunked.py` | Convert to Fabric2D/TorusXY profiles; ensure cache update, per-chunk topology, and no new program compilation are covered by the production transformer gate. |
| `tests/test_prefill_transformer.py` | Remove Fabric1d 2×4/8×4 params. Galaxy TorusXY becomes the authoritative non-chunked full-transformer correctness/determinism test. |
| `tests/test_prefill_transformer_chunked.py` | Make TorusXY the Galaxy trace/no-trace accuracy and perf topology for K2.6, K2.7, GLM-5.2, and eventually K3. The current rows are all 8×4; do not manufacture a reduced 2×4 LB configuration. |
| `tests/test_kv_cache_table.py` | Remove line/ring device params. Keep small LB Fabric2D table/readback diagnostics, but move production acceptance to the full-depth producer/runner test that validates all cache configs and layers. |
| `models/demos/common/prefill/tests/test_producer_runner_e2e.py` | Reassign existing full-depth scenario slots across K2.6, K2.7, GLM-5.2, and K3 without increasing the scenario count. Launch via the canonical TorusXY rank binding instead of direct `mpirun` with runner defaults. |
| `tests/dflash_prefill/test_dflash*.py` | If in scope, reassign the existing small diagnostic to LB Fabric2D and the existing integration slot to Galaxy TorusXY; otherwise move them to a separately owned DFlash pipeline rather than keeping Fabric1d in the scoped suite. |
| `tests/test_disaggregation.py`, `test_{d2d,embedding,h2d}_socket_sync.py` | Use Fabric2D locally. Production D2D validation must use the 2-rank connected TorusXY MGD and assert intermesh socket traffic plus intramesh Ring/Ring collectives. |

`test_producer_runner_e2e.py` still uses direct `mpirun`; until Phase 4 moves it to the canonical rank-binding launcher, the Blaze row explicitly exports `PREFILL_MODEL`, `PREFILL_FABRIC_MODE=2d_torus_xy`, and the descriptor. This is an interim propagation contract, not evidence that the rank-binding migration is complete.

### 6.3 PCC/module tests

Migrate the following to profile-based params and remove their Fabric1d variants:

- `tests/pcc/test_moe_gate_prefill2d.py`
- `tests/pcc/test_moe_routing_setup.py`
- `tests/pcc/test_ttnn_moe.py`
- `tests/pcc/test_shared_expert.py`
- `tests/pcc/test_ffn.py`
- `tests/pcc/test_rmsnorm.py`
- `tests/pcc/test_lm_head.py`
- `tests/pcc/test_parallel_embedding.py`
- Apply the same retained MoE/gate/shared-expert slots to K3 on its integration branches; do not expand the migration matrix.

Target split:

- LB retains one small Fabric2D random-weight diagnostic per materially different kernel/configuration.
- Galaxy TorusXY owns model-size expert counts, maximum practical sequence, real weights, Ring wrap traffic, full block composition, and perf.
- Do not keep 1×N/`linear-8` shapes solely because an old op was written around Fabric1d. Move it to an existing two-dimensional profile and select the correct cluster axis. Most modules use 2×4; the pre-existing MoE module/routing diagnostics legitimately retain their 4×2 axis-sensitive rows. Do not clone those rows onto 2×4.

### 6.4 Cache tests

The cache-construction tests below currently open Fabric1d even though most are not testing fabric:

- `tests/cache/test_embedding_cache.py`
- `tests/cache/test_ffn_cache.py`
- `tests/cache/test_gate_cache.py`
- `tests/cache/test_lm_head_cache.py`
- `tests/cache/test_mla_cache.py`
- `tests/cache/test_moe_cache.py`
- `tests/cache/test_rms_norm_cache.py`
- `tests/cache/test_routed_expert_cache.py`
- `tests/cache/test_shared_expert_cache.py`

For each test:

1. If it only validates filenames/fingerprints/schema/host tensors, make it device-free.
2. If it validates device load/layout, use a 2×2 or 2×4 Fabric2D mesh on LB.
3. Do not duplicate full-checkpoint cache loading across many op tests. One per-module diagnostic plus a full Galaxy model cache-load gate is sufficient.
4. Reassign existing cache slots to cover K2.7 checkpoint identity and K3 schema differences while retaining a K2.6 anchor elsewhere; do not add parameter rows.

### 6.5 Perf tests

| File | Change |
|---|---|
| `tests/perf/test_mla_perf.py` | Keep LB numbers as diagnostic proxies only. Establish production baselines on 8×4 TorusXY for K2.6/K2.7/K3; keep model-specific configs separate. |
| `tests/perf/test_moe_perf.py` | Convert Kimi/GLM/K3 Galaxy params to TorusXY; capture Ring CCL time and total time. Do not compare a TorusXY result against a Fabric1d baseline. |
| `tests/perf/test_prefill_block_perf.py` | Remove line siblings from the production gate. Measure dense and MoE layers on TorusXY, including GLM full/reuse layers and K3 layer types. |
| removed `tests/perf/test_dispatch_combine_perf.py` | Delete the stale, unscheduled 8×1 linear/ring wrapper rather than carrying zero-selecting Fabric1d-era baselines. Existing production MoE/block Galaxy gates are authoritative; add no replacement op matrix. |
| `tests/sparse_mla/test_sparse_mla_perf.py` | Keep the unscheduled non-CI diagnostic on stable-ID Fabric2D; never infer TorusXY from device count. Scheduled production Galaxy perf rows own TorusXY after certification and record topology in profiler output. |
| K3 KDA/AttnRes perf files | Keep their existing local diagnostics on Fabric2D and sign off on full block/transformer TorusXY performance rather than adding a component Galaxy row or summing isolated op targets. |

### 6.6 Utilities and test-only leftovers

- Remove the redundant `utils/sanity_test_32x4_device.py`; the production runner's descriptor/profile preflight and post-open active-fabric assertion are the authoritative replacement.
- Remove Fabric1d defaults from test helpers rather than overriding them in individual CI commands.
- Use the existing collection hooks and repository lint/static-check machinery, scoped to `models/demos/common/prefill`, `models/demos/deepseek_v3_d_p`, and their pipeline entries, to reject `FABRIC_1D`, `FABRIC_1D_RING`, ambiguous production selectors, or a TorusXY mode paired with a non-ring/ring descriptor. Do not add a test file, test function, or parameter row for this policy check; allow narrowly documented compatibility exceptions outside the scoped models.

## 7. Move coverage from op tests to production Galaxy tests

Do this in two steps: first promote and prove an existing production test that covers the behavior and catches injected faults, then prune/demote the duplicate op matrix. Do not add a replacement configuration and do not delete the only diagnostic before the higher-level gate exists.

| Current op-level coverage | New production Galaxy coverage | End state of op test |
|---|---|---|
| `test_prefill_dispatch.py`, `test_prefill_combine.py`, `test_ttnn_dispatch_combine.py` | Full model-size MoE inside dense/MoE block and transformer on TorusXY, with routed-token metadata/recall checks. | Keep one small LB Fabric2D diagnostic per dtype/compression path; remove mesh/topology Cartesian product from release CI. |
| `test_reduce.py` | MoE post-combine reduction in full block with output PCC and Ring TP wrap. | Keep the generic and production-model Fabric2D diagnostics. Remove the newly merged top-k=1-only row: none of the scoped production models uses it, its premise requires a degenerate second axis, and retaining it would add a non-production configuration instead of moving confidence to the composed model gate. |
| `test_ring_joint_mla.py`, `test_mla_matmuls.py` | Kimi K2.6/K2.7/K3 MLA in chunked full transformer, TorusXY, long context, determinism, and KV PCC. | Keep small kernel diagnosis, fabric disabled or LB Fabric2D; no production topology matrix. |
| `test_deepseek_prefill_rotary_embedding_indexed.py`, `test_rope_prefill.py` | Chunked transformer across multiple chunks with golden KV/output PCC on TorusXY. | Retain boundary/index arithmetic unit cases only. |
| `test_deepseek_prefill_update_padded_kv_cache.py`, `test_zero_padded_kv_cache.py`, `test_fp8_kv_cache_gather.py` | Full producer/runner KV table and cache readback for K2.6/K2.7/GLM/K3, including GLM's second index cache. | Retain format/edge-case diagnostics; remove production-sized op sweeps. |
| `test_masked_bincount.py`, `test_offset_cumsum.py`, `test_moe_padding_config.py` | Full model-size MoE routing with zero/overflow/padding cases and end-to-end output PCC. | Keep host/single-device corner cases with fabric disabled. |
| `test_combine_subdevices.py`, `test_dispatch_combine_l1_small_semaphores.py`, `test_sub_device_load_clear_timing.py` | Traced chunked transformer with MoE shared-expert/dispatch overlap, repeated replay, no semaphore leak, and timeout/watchdog. | Keep one LB Fabric2D stress reproducer until TorusXY overlap is stable, then demote from release CI. |

Production Galaxy tests must validate outcomes that isolated ops cannot:

- active TorusXY plus Ring/Ring topology;
- both wrap edges actually carry traffic;
- all layers use the intended topology, not just layer 0;
- persistent semaphores/program cache survive repeated layers and chunks;
- trace capture/replay and non-trace mode agree;
- KV is correct after a full prompt and across all cache configs;
- real weights/cache loading and model-specific layer schedules work;
- no hang under production overlap/concurrency;
- end-to-end latency/throughput is measured on the production fabric.

## 8. Target Galaxy CI matrix

Use separate accuracy and performance jobs so a profiler/perf timeout does not hide correctness attribution. Every row below uses the single-Galaxy Ring/Ring MGD, `FABRIC_2D_TORUS_XY`, two links, and explicit `torus-xy-8x4` selection.

| Model | Fast module gate | Full production gate | Perf gate |
|---|---|---|---|
| Kimi K2.6 | Retained chunked MLA anchor, 384-expert MoE, dense+MoE block, KV table | 61-layer producer/runner; 55k prompt; trace and no-trace; KV PCC | Chunked transformer plus representative MLA/MoE/block signposts |
| Kimi K2.7 | Existing single-shot MLA and cache-identity anchors plus the K2.7 adapter/checkpoint | 61-layer producer/runner using `kimi27.json`; checkpoint-specific 55k golden; D2D smoke | Full chunked runner and only checkpoint-sensitive module targets |
| GLM-5.2 | Sparse MLA full layer + shared/reuse layer, 256-expert MoE, dense layer 0 + first full-indexer MoE layer 6 | 78-layer producer/runner; 11 chunks; both KVPE and compact index-K cache PCC; trace only when sparse metadata trace support is ready | Sparse MLA, MoE, block, and full chunked runner; recalibrate after Ring gathers |
| Kimi K3 | MLA, KDA distributed layer, AttnRes, 896-expert LatentMoE, representative composed blocks | Full K3 depth/checkpoint runner with all caches and mixed layer schedule; trace/no-trace as supported | KDA/MLA/MoE/AttnRes signposts plus full transformer throughput |

Run the topology-only preflight at the start of an existing Galaxy job before its expensive rows; do not add a job or test configuration:

1. run the existing physical-cabling check from `tests/pipeline_reorg/galaxy_health_tests.yaml`: `run_cluster_validation --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/bh_galaxy_xy_torus.textproto --hard-fail --send-traffic`;
2. resolve and open the exact existing `single_bh_galaxy_torus_xy_graph_descriptor.textproto` with `STRICT` reliability;
3. assert active FabricConfig is `FABRIC_2D_TORUS_XY`;
4. assert descriptor dimensions are `[RING,RING]` and `(sp,tp) == (Ring,Ring)`;
5. run a short collective over each axis and prove traffic crosses each wrap edge;
6. close/reopen once to catch teardown/reinit problems.

The production tests should run through the same `tt-run` rank-binding path as serving, not a pytest fixture that reconstructs a similar device setup.

For every production command, acceptance is based on exact execution, not merely exit code: record collection count, executed/passed count, skipped count, deselected count, and selected node IDs. Require the expected nonzero selected/executed count and zero topology/hardware skips. The current global Galaxy torus skip can otherwise turn an incompatible allocation into a green job with no useful coverage.

## 9. What can be tested on this LoudBox

### 9.1 Required environment repair first

The current `_ttnn` extension does not match the Python source. Build the code from the repository root with the release configuration before interpreting any pytest result:

```bash
./build_metal.sh --release
```

This is the mandatory build command for every implementation phase. Re-run it after changes that affect C++, generated bindings, TTNN, fabric, or build inputs; a phase cannot exit on a stale or differently configured build. Then confirm eight devices are idle and visible.

No destructive reset should be part of the normal test recipe. If a fabric hang leaves devices occupied, follow the existing lab recovery procedure rather than embedding resets in tests.

### 9.2 Local tests after the parameter migration

All tests launched manually on the local LoudBox must run through `scripts/run_safe_pytest.sh`, without `--dev`. Direct `pytest` is not an accepted local milestone result. The safe runner provides cooperative device handling and invokes `tt-triage` on dispatch timeout. CI scheduling continues to use the repository's normal direct `pytest` command form; the safe wrapper must not be inserted into pipeline YAML. These are the intended local command shapes; their final node IDs and exact counts must be locked in the `LB-F2D` collection manifest before milestone execution:

```bash
# Pure host/reference/config tests (fabric disabled or no device fixture)
EXPECT_NUM_TESTS=17 scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/reference/kda/tests -q
EXPECT_NUM_TESTS=15 scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/test_sparse_kv_cache_contract.py -q
EXPECT_NUM_TESTS=5 scripts/run_safe_pytest.sh models/demos/common/prefill/tests/test_prefill_producer_kv_decode.py -q

# K2.7 single-shot and K2.6 chunked MLA anchors: existing random-weight Fabric2D 2x4 rows
EXPECT_NUM_TESTS=1 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/test_mla.py::test_kimi_mla \
  -q -k "fabric2d-2x4 and random and scaled_sl and seq5k and check_pcc and sequential"
EXPECT_NUM_TESTS=1 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill \
  -q -k "fabric2d-2x4 and cpu and maxedge-1u and kimi and scalar"

# GLM-5.2 sparse MLA full/reuse anchors, LB Fabric2D 2x4
DS_SPARSE_FABRIC=fabric2d EXPECT_NUM_TESTS=8 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla.py \
  -k "glm_5_2 and 2x4 and fabric2d"

# Existing model-size/config-specific MoE diagnostics on LB Fabric2D
EXPECT_NUM_TESTS=1 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/pcc/test_moe_gate_prefill2d.py::test_forward_pass \
  -q -k "fabric2d-mesh-2x4 and kimi-device_fp32"
EXPECT_NUM_TESTS=1 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ds_moe \
  -q -k "fabric2d-mesh-4x2 and pcc-device-glm-256 and pad0"
EXPECT_NUM_TESTS=1 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/pcc/test_moe_routing_setup.py::test_prep_dispatch_combine \
  -q -k "fabric2d-mesh-2x4 and random and pad50"
EXPECT_NUM_TESTS=1 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/pcc/test_shared_expert.py \
  -q -k "fabric2d-2x4 and 3.2K"

# Existing DeepSeek 2x4 composition anchors. Kimi/GLM composition remains at its original 8x4
# production shape and is deliberately deferred to Galaxy rather than cloned or reshaped locally.
EXPECT_NUM_TESTS=1 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/test_prefill_block.py::test_ds_prefill_block \
  -q -k "fabric2d-mesh-2x4 and smoke-random and dense and balanced and not non_balanced and no_determinism and iter1 and random and not pretrained"
EXPECT_NUM_TESTS=1 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/test_prefill_block.py::test_ds_prefill_block \
  -q -k "fabric2d-mesh-2x4 and smoke-random and moe-gate_device and balanced and not non_balanced and no_determinism and iter1 and random and not pretrained"
EXPECT_NUM_TESTS=1 scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py::test_ds_prefill_transformer \
  -q -k "fabric2d-mesh-2x4 and 5_layers and e256_host and no_determinism and iter1 and random and not pretrained and balanced and not regular and 1024 and smoke-random-random and right_pad"

# The temporary low-level diagnostic set is the exact EXPECT_NUM_TESTS=68 selector owned by
# bh_lb_DeepSeek_PREFILL_OP_TESTS; copy that selector verbatim when executing the locked local manifest.
```

Do not add `--dev` to any invocation above or to their CI equivalents. Record the safe-runner log and any generated `tt-triage` artifacts with the phase evidence.

Local acceptance:

- no `FabricConfig::FABRIC_1D*` appears in logs;
- all communicating tests report `FABRIC_2D` and Linear/Linear;
- representative random-weight PCC bars remain at or above their existing thresholds;
- two consecutive runs are clean, including device close/reopen;
- program-cache/trace tests compile no unexpected programs after warmup;
- test runtime is recorded to resize the LoudBox jobs after duplicate op cases are removed.

### 9.3 Optional local real-weight work

If the required mounts are added, the LoudBox can run reduced-depth K2.6/K2.7/GLM/K3 checkpoint/cache-load tests on 2×4 Fabric2D. These validate weight naming and memory layout only. They do not sign off production topology or performance.

### 9.4 Local LoudBox milestone: `LB-F2D`

#### Goal

Complete and independently validate every migration change that does not require a physically wrapped Galaxy. From a clean release build, the scoped host and LoudBox suites must use either disabled fabric or Fabric2D, exercise the common topology plumbing and each available model component, and leave no Fabric1d dependency in scoped local tests or Blackhole local CI.

This milestone is the local completion gate for Phases 0–2. Passing it means the code is ready to enter TorusXY Galaxy bring-up; it does **not** mean the production topology is validated.

#### Fixed local topology policy

| Test type | Required local topology | Rule |
|---|---|---|
| Host/reference/schema test | Fabric disabled; preferably no device fixture | It must not open fabric merely because a shared fixture used to do so. |
| Single-device test | Fabric disabled | A one-chip Fabric1d compatibility case is not allowed. |
| Four-device communication | Fabric2D 2×2, Linear/Linear | Used for retained QuietBox diagnostics and quick local reproduction. |
| Eight-device communication | Fabric2D 2×4, Linear/Linear | Canonical LoudBox composition/MLA shape and the default local selector. |
| Axis-sensitive module diagnostics | Fabric2D 4×2, Linear/Linear | Retain only the existing MoE/routing and sparse-cache diagnostic families that already own 4×2; do not clone them onto 2×4 or expand either matrix. |

The LoudBox has no verified physical wrap, so local tests must not open TorusX, TorusY, or TorusXY. Torus parsing/mapping is validated through existing host/collection paths and static inspection; this milestone does not add tests or configurations.

#### Test-matrix freeze

- Do not add a test file, test function, or parameter value for this migration.
- Replace existing Fabric1d cases with Fabric2D/TorusXY profiles or remove redundant op-level cases after mapping them to an existing production Galaxy gate.
- Validate K2.7 by reassigning selected existing Kimi slots while retaining K2.6 coverage elsewhere; do not cross-product the model variants.
- Remove explicit topology parameters and derive topology from FabricConfig. This is a refactor of existing cases, not an added configuration axis.
- A migration must not change a row's mesh shape, link count, sequence length, model, trace mode, workload, or calibrated performance role merely to make it runnable on LoudBox. In particular, never repurpose an 8×4/32×4 production row as a 2×4 local row.
- A row is redundant only when its legacy fabric/topology choice is the sole remaining difference after migration, or when an explicitly named higher-level production Galaxy test covers the same workload. Record that replacement before deletion; distinct shapes and performance baselines are not implicitly redundant.

#### Entry criteria

1. `./build_metal.sh --release` succeeds from the current source revision.
2. `python -c "import ttnn"` succeeds against that release build; the current stale `_ttnn`/Python mismatch is gone.
3. `tt-smi -s` reports exactly eight expected Blackhole devices, all idle and healthy.
4. Required tests use random/generated weights and repository-owned references. Model mounts are optional and cannot be required for milestone completion.
5. A collection manifest is captured before execution. For every command it records the exact selected node IDs and a nonzero `EXPECT_NUM_TESTS`; there may be no `TBD` count at milestone sign-off.

#### Required work and local validation

| ID | Deliverable | Required local proof |
|---|---|---|
| `LB-01` | Common topology contract in `runner_utils.py`, runner setup, adapter/model selection, and `tt_ccl.py` | Existing host/import paths plus direct static/config checks cover Fabric2D and every torus enum mapping, Ring/Ring descriptor parsing, missing/mismatched descriptor rejection, explicit production model requirement, and no silent SP-size/default-model fallback. An existing device smoke opens/closes Fabric2D 2×4 twice and reports Linear/Linear. No new test/config is added. |
| `LB-02` | Central test params in `models/demos/deepseek_v3_d_p/tests/conftest.py` and `tests/pcc/mesh_configs.py` | Collection contains disabled, 2×2, canonical 2×4, and only the pre-existing axis-sensitive 4×2 diagnostic families for local execution. No scoped local node ID resolves to `FABRIC_1D` or `FABRIC_1D_RING`; torus params are not selected on LB. |
| `LB-03` | Sparse MLA topology selection | Delete `_topology_from_device_params` and its priority fallback; locally supported random-weight sparse-MLA/cache rows pass on their existing Fabric2D 2×4 or 4×2 shapes using the common per-axis topology helper. The GLM-5.2 shared-layer cache row remains at its original 8×4 shape and is Galaxy-only. |
| `LB-04` | Dense MLA | Preserve, rather than cross-product, the existing Kimi anchors: the K2.7 random-weight single-shot MLA PCC case and the K2.6 random-weight chunked MLA PCC case pass on Fabric2D 2×4. Each K3 component PR must run the equivalent existing K3 MLA gate on its branch once available. |
| `LB-05` | MoE/gate/shared expert | One representative Kimi and GLM route/gate/shared-expert PCC path passes on its existing Fabric2D row: MoE's axis-sensitive module cases remain 4×2, while gate/routing/shared-expert cases with existing 2×4 ownership remain 2×4. Include uneven/padded routing and program-cache reuse without cloning either shape. K3 LatentMoE runs the equivalent branch-local gate when integrated. |
| `LB-06` | Prefill block and transformer composition | Run the existing DeepSeek reduced random-weight dense block, MoE block, and short transformer/loop smoke on Fabric2D 2×4. Kimi and GLM composition rows exist only at their original 8×4 production shape, so they are explicit Galaxy-only requirements; do not clone or reshape them for LB. A K3 branch must run local composition only if that branch already owns a 2×4 row, otherwise its composed 8×4 gate is Galaxy-only. If trace is supported by an applicable existing local row, trace and no-trace are required; otherwise record it as deferred rather than collecting a skipped test. |
| `LB-07` | Cache construction and KV-table handling | Host-only cache schema/fingerprint tests run fabric-free. One device layout/load/readback diagnostic per materially different cache type runs on Fabric2D. Cover K2.7 identity and GLM-5.2's compact index-K contract; add K3 schema coverage on its integration branch. |
| `LB-08` | Socket and D2D diagnostics | Retain `test_d2d_socket_sync.py` as Fabric2D 2×2 QuietBox coverage and demonstrate second-dispatch program-cache reuse there. On this eight-device LoudBox, collection must show the expected hardware skip and the case is excluded from the required execution manifest. Connected-Galaxy transport remains Galaxy-only. |
| `LB-09` | DFlash scope and diagnostics | Treat the currently scheduled Kimi DFlash drafter as in scope, retain its original 8×4 shape, and migrate it one-for-one to TorusXY in Galaxy CI. It is explicitly outside local execution because weights and a 32-device mesh are unavailable; do not manufacture a 2×4 DFlash configuration. It may leave scope only through an explicit product-owner decision that removes it from all scoped production rows. |
| `LB-10` | Op-test reduction | Retain one small Fabric2D diagnostic per materially different kernel/dtype/compression path. Remove Fabric1d topology matrices from scoped op tests and document the production Galaxy replacement for every pruned release case. |
| `LB-11` | Local CI migration and policy enforcement | All `bh_lb_DeepSeek_*` and 4-chip QuietBox selectors are exact Fabric2D/disabled-fabric selectors. Communicating cases are removed from p150/p300 jobs. The scoped static validator rejects Fabric1d tokens, ambiguous selectors, zero collection, and topology/hardware skips. |
| `LB-12` | Repeatability and failure evidence | The complete required device command set passes twice, including fabric/device close and reopen, normally without manual reset. Any hang is diagnosed with `tt-triage`, classified under the reopen policy below, rebuilt if code changes, and rerun; unresolved migration-caused hangs fail the milestone. |
| `LB-13` | Iterative review | Claude reviews the local diff, release build, collection manifest, both test runs, topology logs, policy audit, and triage artifacts. Findings are fixed and affected work rebuilt/retested until Claude returns explicit `OK`. |

K3 is developed across unmerged branches, so `main` is not expected to collect unavailable K3 nodes. Use two explicit statuses:

- `LB-F2D-main` covers common code plus K2.6, K2.7, and GLM-5.2. It can pass and unblock those models' Phase 3 rows without collecting K3 selectors.
- `LB-F2D-K3` consists of linked `LB-04` through `LB-07` reports from the corresponding K3 component PRs plus `LB-09` if K3 uses DFlash, and a composed K3 branch passing `LB-06`. It must pass before any K3 Phase 3 row is promoted.

The program-wide `LB-F2D` milestone is complete only when both statuses pass. An unavailable K3 selector is absent—not counted as skipped—from `LB-F2D-main`; an empty K3 selection never counts toward `LB-F2D-K3`.

#### Current local evidence (2026-08-11)

- `./build_metal.sh --release` passed; `python_env/bin/python -c "import ttnn"` passed; `tt-smi -s` reported all eight Blackhole devices healthy (KMD 2.8.0, firmware 19.12.0).
- The locked host-only manifest contains the 17 existing KDA reference tests, 15 existing sparse-KV contract tests, and 5 existing producer-decode tests. After the latest code changes and release build it passed twice through the safe runner: 37/37 each run. The unrelated broad `tests/torch` directory is not part of this topology/config/schema milestone.
- Kimi K2.7 MLA on the existing Fabric2D 2×4 case passed twice (PCC 0.998947).
- GLM-5.2 KV-table readback on the existing Fabric2D 2×4 case passed twice (5120-token readback).
- In the corrected matrix, the three local 4×2 cases—DeepSeek-V3.2 and GLM-5.1 sparse cache-only round trips plus the DeepSeek-V3.2 missing-cache fallback—selected/executed 3/3 and passed twice. Both round trips reported PCC 1.0; the second complete run was fully warm (648/648 JIT-cache hits). The GLM-5.2 shared-layer `ReuseIndexer` case is restored to its original 8×4 Galaxy-only shape.
- Kimi K2.7 MLA selected/executed 1/1 and passed twice, GLM-5.2 KV-table selected/executed 1/1 and passed twice, and the safe runner's exact-count guard rejected intentionally incorrect expected counts before hardware execution.
- The retained block perf wrapper collects exactly two existing `block_2x4` rows. Execution requires the staged DeepSeek weights: without the mount it began a 17.1 GB Hugging Face download and was stopped immediately; this was an asset failure, not a hang.
- The retained K2.6 LB model anchor is `test_mla_chunked_prefill[...,kimi,...,fabric2d-2x4]`; K2.7 owns the existing single-shot MLA and KV-identity anchors. No K2.6/K2.7 cross-product was added.
- The D2D 2×2 diagnostic collects but skips on this machine because Blackhole requires all eight visible devices. It is QuietBox-only. DFlash and the restored shared-cache 8×4 case are Galaxy-only.
- The 2026-08-11 completion attempt did not close `LB-F2D-main`. A 2×2 cache sweep passed embedding cold/warm PCC 1.0, then saw one intermittent FFN cold-cache PCC failure; the isolated FFN rerun passed cold/warm PCC 1.0 with 100% JIT hits. More importantly, the next 8-device K2.7 open failed twice without reset or recovery at `Fabric Router Sync: Timeout ... Device 6`, with the router stuck in `STARTED` during the remote Ethernet handshake.
- A post-review allocation recheck on 2026-08-11 again found only 17 MiB free in `/dev/shm`. All eight device nodes and `tt-smi` telemetry entries were present, but a fresh read-only `tt-triage --dev=all` snapshot (`generated/tt-triage/lb-f2d-health-recheck-20260811.csv`) again inspected only seven devices, with device 3 absent as in the earlier pass, and again failed Ethernet status on devices 4, 5, and 6 with retrain count 1 on both reported links. No test/MPI process was active, and no reset or shared-memory deletion was performed. This is a repeated platform-health failure, so no Fabric2D device execution was launched from that allocation.
- An exploratory broad host run selected 47 existing tests but stopped at the pre-existing `test_moe[random-weights-gate]` assertion that `TorchMoe` requires `route_scale`; that unrelated test is not part of the locked topology/config/schema manifest and was not hidden by changing a test/config row.
- A direct topology-profile check proved local `2d -> (Linear, Linear)`, TorusXY descriptor parsing to `(Ring, Ring)`, and rejection of missing mode, scoped 1D modes, and TorusXY without a descriptor.
- A restricted four-device LoudBox run exposed a timeout while transferring the production-size parallel-embedding weight on a redundant 2×2 candidate row. The safe runner invoked `tt-triage` before recovery, saved `generated/tt-triage/triage.csv` plus the full triage log, and reset devices 0–3. That redundant row was removed rather than added to QuietBox; the existing 2×4 parallel-embedding row remains the diagnostic. The nonredundant migrated FFN and RMSNorm 2×2 rows then executed 4/4 and passed after reset. Any later hang retains the same mandatory triage-before-recovery policy.
- Collection-only validation produced exact nonzero contracts for every edited pipeline selector. Representative local counts are the LoudBox PCC suite 65 (including the restored shared-expert, LM-head, and parallel-embedding 2×4 rows), the narrowed LoudBox op-diagnostic suite 68, Kimi MLA 1, the release-filtered DeepSeek MLA selector 2, the broader LoudBox/T3K DeepSeek MLA selector 4, the three nonredundant CI-executable DeepSeek block anchors, each model-specific chunked selector 2, the combined chunked selector 4, block perf 2, the eight CI-executable block-loop rows, GLM sparse MLA 8, and DeepSeek sparse MLA 7. Representative Galaxy-only counts are MoE gate 16, DeepSeek MLA 2, chunked MLA 4, Kimi K2.7 MLA 2, KV base 1, Kimi KV mock 12, GLM KV 1, DeepSeek block 6, and GLM MoE 4. Sparse-MLA 8×4 counts remain Galaxy-collection-dependent because that test constructs its matrix from detected device count; they cannot be certified from an eight-device allocation.
- The corrected QuietBox PCC selector collects exactly 28 existing cases and is now count-guarded. FFN and RMSNorm own the four migrated 2×2 cases; shared expert, LM head, and parallel embedding reuse their existing 2×4 coverage instead of retaining/replacing redundant 1×4 rows.
- The sparse MLA vs-trace device selector now collects exactly 42 existing cases on this LoudBox, all with stable `fabric2d-sp4xtp2` or `fabric2d-sp2xtp4` profile IDs. The 21 cases contributed only by the legacy 8×1 line proxy are removed; QuietBox retains only 2×2 and unsupported device counts collect one skipped 2×2 sentinel without opening fabric.
- The test inventory did not grow: AST comparison reports zero added test functions and three removed functions—the redundant Fabric1d LoudBox MoE perf wrapper, the redundant 32×4 sanity test replaced by production runner preflight, and the redundant 8×1 sparse-SP CCL perf proxy now covered by the full sparse-MLA Galaxy gate. The independent mesh×fabric products were replaced by paired profiles, shrinking the principal DeepSeek MLA matrix from 9 profile combinations to 3, Kimi MLA from 6 to 2, and chunked MLA from 9 to 3; raw `pytest.param` call-site growth is only the representation of those pairs, not added configurations.

#### `LB-F2D-main` completion evidence (2026-08-12)

- Allocation health recovered without a migration-code change. `tt-smi` enumerated all eight Blackhole devices and every device reported the same 20-live-link Ethernet mask (`0x3ed03edf`). A fresh read-only `tt-triage` health pass succeeded and was saved as `generated/tt-triage/lb-f2d-health-recheck-20260812.csv`. `/dev/shm` remained constrained to approximately 17 MiB by pre-existing OpenMPI segments, so every safe-runner invocation emitted a 16 MiB allocation warning; no segment was removed, no per-test accumulation occurred, and this did not fail or hang a run.
- The current source revision `40248de405fcfc99c3b0a3e7dfd10fe5cf037643` on `pjosipovic/fabric2d-torus-xy-prefill` passed the exact `./build_metal.sh --release` build and `python_env/bin/python -c "import ttnn"`; the clean source build identified itself as `0.77.0-dev20260811+13.40248de405`. The exact release build and import were repeated successfully after the final CI-selector and evidence edits (`+m` denoting the expected tracked working-tree changes).
- Every locked selector was collection-checked before execution. One documentation-only selector error was found by execution: the five-layer transformer used `e64_host`, but the selected model owns 256 routed experts and rejected 64 supplied expert weights during construction. The selector was corrected to the already-existing, uniquely collected `e256_host` row. It still selects exactly one existing test/configuration; no test, parameter, mesh shape, or production 8×4 row was added or changed. The corrected 256-expert transformer passed on Fabric2D 2×4.
- The host manifest—17 KDA reference tests, 15 sparse-KV contract tests, and 5 producer-decode tests—passed twice: 37/37 on each run, fabric-free.
- The complete device/component manifest passed twice without reset or recovery between runs: K2.7 single-shot MLA (1), K2.6 chunked MLA (1), GLM-5.2 sparse MLA/cache (8), Kimi gate (1), GLM 256-expert integrated MoE (1), padded routing (1), shared expert (1), dense block (1), MoE block (1), and the five-layer/256-expert transformer (1). Every communicating run opened `FabricConfig::FABRIC_2D` on all eight devices; 2×4 and 4×2 local profiles derived Linear/Linear from FabricConfig rather than accepting an explicit topology argument.
- Representative second-pass accuracy remained above existing thresholds: K2.7 output PCC `0.998947` and KV/PE PCC `0.999876`/`0.999887`; K2.6 three chunk outputs `0.998560`, `0.997556`, `0.996935`, full output `0.997926`, and KV/PE `0.999877`/`0.999886`; GLM-5.2 sparse output PCC at least `0.993832` with KV/PE at least `0.999526`/`0.999916`; integrated MoE routed/final/reference PCC `0.974067`/`0.988244`/`0.988228`; shared expert PCC `0.999721`.
- The original 122-case op selector was not a valid fail-closed gate: 26 selected parameter combinations deliberately skipped in their test bodies. The selector was narrowed without changing tests or adding configurations. Iterative Claude review then found another 28 rows that pass locally but deliberately skip under `CI=true` (unsupported cache/rotary variants and row-major CI duplicates). The final shared local/CI selector therefore contains 68 existing executable diagnostics, preserves the supported PCC, dtype/compression, ring-joint, cache, routing, dispatch, and combine paths, and removes only redundant/unsupported skip rows. It passed locally twice as `68 passed, 1704 deselected`, zero skips/failures, in 427.63 s and 426.90 s; each pass reported `1730/1730` JIT-cache hits and a clean device close. Claude independently ran the same selector with `CI=true`; it passed `68/68`, zero skips/failures, in 448.18 s through the local safe runner. The broader 94-case local precursor also passed twice before the CI-only duplicates were removed, demonstrating that narrowing did not conceal failures.
- Across the completed component and op manifests there was no timeout or hang, every safe-runner invocation closed the devices normally, and no reset occurred. Consequently no new failure triage artifact was needed. A post-run `./tools/tt-triage.py --dev=all` health snapshot enumerated all eight devices and passed ARC, Ethernet, NOC-location, L1-inactive, and broken-component checks; it is saved as `generated/tt-triage/lb-f2d-postrun-health-20260812.csv`. The idle post-run masks reported 14 live links per device (`0x3ED016D0` on devices 0–3 and `0x3ED02AD0` on devices 4–7), while `check_eth_status.py` passed; the recovered-allocation and earlier failure artifacts are retained for comparison.
- All 8×4 production model composition, Ring/Ring/TorusXY cabling and traffic, production-depth/real-weight correctness, trace replay at production shape, and performance calibration remain Galaxy-only. K3 remains tracked separately as `LB-F2D-K3` on its component branches and is not falsely counted as skipped or passed by `LB-F2D-main`.

This evidence closed `LB-F2D-main` at commit `38414f258c1`: the locked host, component, and op manifests passed with exact nonzero counts and zero skips, the required release build passed, device reopen/health remained clean, and iterative Claude review ended in exact `OK`. Merging current `origin/main` at `a2cc0716f3c` reopened the local certification gate because upstream added K3 MLA and changed collected inventories. The current head retains that earlier evidence as a baseline. Its refreshed host manifest passed 37/37; its exact `CI=true` op gate passed twice as 68/68 with zero skips/xfails (`1710` deselected), with the second pass completing in 427.79 seconds at `1730/1730` JIT-cache hits and a clean device close; and the three narrowed DeepSeek block anchors passed 3/3 in 111.16 seconds with zero skips/xfails and a clean close. Safe-runner collection confirms exact CI contracts of 3 block, 8 block-loop, 14 QuietBox cache (including K3), and 205 QuietBox op cases. The eight real-weight block-loop cases cannot execute on this allocation because `/mnt/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528` is absent, so their physical gate remains assigned to mounted LoudBox CI rather than being represented as local evidence. The exact release build passed after these source changes. Iterative Claude `OK` remains the final post-merge closure requirement. This evidence does not close `LB-F2D-K3` or program-wide `LB-F2D`; all 8×4, Ring/Ring, wrap-link, production-depth, real-weight, and performance conclusions remain Galaxy-only.

#### `LB-F2D-K3` partial MLA evidence (2026-08-12)

- Current `origin/main` was merged into this branch as `a2cc0716f3c`, bringing in the already-merged K3 MLA component from PR #52068. The integration retained the migration's paired Fabric2D/TorusXY profiles and removed no production 8×4 shape.
- The merged K3 cache row's existing 2×4 parameter was migrated in place from Fabric1d to Fabric2D. Its already-existing 2×2 QuietBox cache variant remains in the 14-case cache job because it is the only scheduled cold/warm `g_proj` cache gate and caught the real merge regression fixed here. The existing K3 Galaxy functional and perf rows now use the shared 8×4 TorusXY profile; the perf call derives Ring/Ring from FabricConfig rather than passing `Topology.Linear`. The two existing Blaze K3 jobs select exact `torus-xy-8x4` IDs with `EXPECT_NUM_TESTS=2` and `1` respectively and explicitly export the single-Galaxy TorusXY mesh-graph descriptor before rank launch. No test function, parameter row, mesh shape, or configuration was added by this migration.
- Collection through the local safe runner confirms exactly two existing K3 Galaxy functional cases and one existing K3 Galaxy perf case. The perf child itself also resolves to exactly one `torus-xy-8x4`, scalar, non-determinism case. Their physical TorusXY execution and performance calibration are Galaxy-only.
- The merged K3 reference suite passed locally through the safe runner: 9/9 existing tests, including configuration/layer scheduling, absorbed-vs-unabsorbed MLA at sequence lengths 128 and 512, output-gate and NoPE load-bearing checks, and the pinned accuracy contract.
- The integration and follow-up K3 repair each passed the exact `./build_metal.sh --release` command. The scoped Fabric2D policy validator, pre-commit checks, Python compilation, YAML parsing, and diff checks pass.
- The existing K3 MLA cold/warm-cache case passed on Fabric2D 2×4: weights-to-cold and weights-to-warm PCC were both `1.0`, including `g_proj` cache construction/reload. The existing K3 chunked CPU-reference case passed on Fabric2D 2×4 with per-chunk PCC `0.998479`, `0.993904`, and `0.991073`, full-output PCC `0.996875`, and KV/PE PCC `0.999877`/`0.999894`. The existing deterministic functional case passed three repetitions at PCC `1.0`.
- Local execution found and fixed one real merge interaction: K3's output gate referenced a legacy `_all_gather` helper that had already been removed on `main` by PR #52606 before the K3 MLA merge reached this branch. Commit `017e8f00a6a` routes it through the topology-free high-bandwidth Fabric2D gather with a construction-time persistent output buffer and preserves that buffer across serial cache/model instances. The failing cache case was rerun after the exact release build and passed.
- Every K3 local invocation used `scripts/run_safe_pytest.sh`, opened `FabricConfig::FABRIC_2D` on all eight devices, and closed normally. There was no hang, reset, or new triage artifact.
- `LB-F2D-K3` remains **incomplete**. The linked sections below now close the local AttnRes, KDA, and LatentMoE component gates; a composed K3 block/transformer gate (`LB-06`) is still required. Their original production 8×4 rows must be validated as TorusXY in Galaxy CI; they must not be cloned or reshaped to make them runnable on LoudBox.

#### `LB-F2D-K3` linked AttnRes evidence (2026-08-12)

- Branch `pjosipovic/fabric2d-torus-xy-k3-attnres` is stacked on the reviewed base at `dfbed04cc68` and integrates the existing K3 AttnRes component from draft PR #52676 at merge commit `6d02ee5f90`. The component's sole device placement remains the original 2×4 LoudBox shape. Its existing model and fused-op rows were migrated one-for-one from Fabric1d to plain Fabric2D; no test function, parameter row, mesh shape, model configuration, or 8×4 component row was added.
- `TtAttnRes` no longer accepts a caller-selected topology. Its ordinary TP all-reduces derive the active axis topology from FabricConfig. The fused `attn_res_gather_softmax` public Python/C++ operation also removes its topology input and resolves the TP axis from the active FabricConfig before creating the internal primitive. Plain local Fabric2D resolves to Linear; a certified production TorusXY mesh resolves to Ring without a caller override.
- The first full Fabric2D run exposed a real imported-op hang in the first S=1 read. The safe runner invoked `tt-triage` before recovery, saved `generated/tt-triage/triage.csv`, and reset all eight devices. Triage showed the gather/fold BRISCs and Ethernet routers waiting indefinitely. The fused gather kernel was casting the active Fabric2D `HybridMeshPacketHeader` to the Fabric1d low-latency header and programming a hop-count route; on Fabric2D that malformed every peer route, so arrival semaphores never completed.
- The repair keeps topology ownership in FabricConfig and makes routing fabric-aware. The host supplies each existing peer's destination mesh/chip and derives its first-hop connection from the Fabric2D control plane, including a future torus wrap; the kernel programs Hybrid headers by destination node and retains hop-count routing only for the non-scoped Fabric1d compatibility path. The formerly hanging S=1 case then passed at PCC `0.9999952`, maximum relative error `7.906e-03`, with a normal eight-device close.
- The exact `./build_metal.sh --release` build and `python -c 'import ttnn'` passed after the repair. After that final build, the unchanged combined AttnRes job exact-collected 17 tests and passed twice consecutively through `scripts/run_safe_pytest.sh` with `EXPECT_NUM_TESTS=17`, zero skips/xfails, no reset between passes, and normal closes. Because the release rebuild does not clear the Metal JIT cache, both post-build passes were warm: `17/17` in 50.70 and 50.78 seconds, each with `603/603` JIT-cache hits. Durable evidence is retained in `generated/lb-f2d-evidence/attnres-pass1.log`/`.xml` and `attnres-pass2.log`/`.xml`; both JUnit sessions start after the final release binary. The 93-layer/186-read model walk reported PCC `0.9999895`; fused plain/settled reads reported PCC `0.9999975600432865` and `0.9999962988289343`, and the settled stream remained bit-identical to `ttnn.add`. Claude iterations 30 and 31 found no code defect but required fresh post-build runs and then durable per-pass artifacts; iteration 32 reviewed those retained artifacts and the complete staged diff and returned exact `OK`.
- This LoudBox has no checked-in descriptor proving a physical wrap, so the retained 2×4 row correctly uses unwrapped Fabric2D and cannot certify Ring traffic. The component owns no pre-existing 8×4 row to migrate. Physical TorusXY routing, wrap links, and production composition remain assigned to the future existing-shape K3 block/transformer Galaxy gate; no reduced or duplicate component configuration is manufactured locally.
- The linked AttnRes evidence alone does not close `LB-F2D-K3`; the following KDA and LatentMoE sections close their local component gates, while a composed K3 block/transformer/runner gate remains outstanding. The local AttnRes stage gate is closed by the exact release build, two retained post-build runs, and Claude iteration 32's exact `OK`; physical TorusXY remains a Galaxy CI obligation.

#### `LB-F2D-K3` linked KDA evidence (2026-08-12)

- Branch `pjosipovic/fabric2d-torus-xy-k3-kda` is stacked on the reviewed AttnRes branch at `1033e1a5a89` and integrates the existing K3 KDA implementation from draft PR #52799 at merge commit `41245da1330`. Its six existing distributed placements were migrated one-for-one from Fabric1d to plain Fabric2D. The existing performance matrix remains SP1×TP8, SP2×TP4, and SP4×TP2 at T=5120; no test file, test function, parameter value, mesh shape, checkpoint configuration, or performance row was added or reshaped.
- `KDAProgramConfig`, `kimi_k3_program_config`, and `ttKDA` no longer accept a caller-selected topology. KDA derives the TP-axis topology from the active FabricConfig at the CCL boundary through the common `per_axis_topology` helper. Direct CCL diagnostics use the same helper instead of hard-coded Linear. Plain local Fabric2D therefore resolves to Linear on each axis, while a certified production TorusXY configuration resolves to Ring on each axis without a caller override. A scoped source audit finds no Fabric1d `FabricConfig` selection in KDA production, reference, or test code; the remaining Fabric1d words are limited to the deliberately retained historical performance warning and baseline-provenance record.
- The K3 configuration-file merge preserves the already-integrated MLA/K3 dimensions and adds only the constants/configuration functions required by the imported KDA implementation; it does not introduce an alternative model shape. In particular, no production 8×4 composition row was changed to 2×4.
- The exact `./build_metal.sh --release` build passed at `41245da1330`, including the imported KDA C++ operations. `python -c 'import ttnn; import models.demos.deepseek_v3_d_p.tt.kda.kda'`, Python compilation, and the scoped Fabric2D policy validator also passed.
- Three locked manifests were collection-checked and executed twice after that release build through `scripts/run_safe_pytest.sh`, always with an exact nonzero `EXPECT_NUM_TESTS` and without `--dev`. The fabric-free host/reference/checkpoint-contract manifest passed 41/41 in 8.25 and 8.02 seconds. The existing single-device composed-layer/chunk manifest passed 21/21 in 50.22 and 18.45 seconds. The existing distributed Fabric2D manifest passed 23/23 in 159.96 and 25.24 seconds, exercising the retained 1×8 TP weight/layout and layer-PCC rows, both TP axes of the 2×4 distributed layer, affine, halo, output-placement, and determinism paths. The selector excludes seven wrong-shape parameter products whose test bodies deliberately skip; it changes no test or parameter matrix and turns the required execution gate into zero-skip evidence.
- All six executions ended with `SAFE_PYTEST_RESULT: PASS`; both distributed passes initialized `FabricConfig::FABRIC_2D` on all eight devices and closed normally. There was no hang, reset, recovery, or new `tt-triage` artifact. Durable logs and JUnit XML are retained under `generated/lb-f2d-evidence/kda-{host,single-device,distributed}-run{1,2}.{log,xml}`.
- `KIMI_K3_CKPT` is not set and no Kimi K3 checkpoint exists at the checked local mount locations. Consequently the unchanged real-weight and T=5120 performance rows are absent from the required local manifest rather than collected as skips. They require a checkpoint-mounted LoudBox or Galaxy CI allocation. Physical Ring/Ring traffic, wrap-link routing, the original 8×4 production composition, production depth/checkpoint correctness, and production performance remain Galaxy-only.
- The imported component had already recorded that an earlier Fabric2D T=5120 SP1×TP8 attempt hung with a device timeout and failed Ethernet-core recovery, while SP2×TP4/SP4×TP2 were correct but 0.05%/0.10% slower than Fabric1d. Migrating the fixture does not erase or clear that finding: it is restored beside the existing perf row with the KDA component stage as owner. The JSON targets now identify their exact `FABRIC_1D` provenance and are explicitly retained only as a conservative historical guardrail. The checkpoint-backed Fabric2D matrix must clear the SP1×TP8 hang and rebaseline all three unchanged rows before KDA performance or Galaxy promotion can pass; the small synthetic T=32 1×8 PCC result is not substituted for that gate.
- Claude iteration 1 verified the topology/configuration migration and all retained test artifacts, then required the unresolved SP1×TP8 Fabric2D hang and historical Fabric1d target provenance to remain explicit and the KDA instructions to distinguish local safe-runner use from direct-pytest CI. Iteration 2 confirmed those fixes and required the source-audit wording to distinguish executable Fabric1d selection from retained historical prose. Iteration 3 rechecked the complete diff at `7aef180e854` and returned exact `OK`; the final documentation-only head `212f46453e0` then received a separate exact-snapshot review and exact `OK`. The KDA local component gate and its publication record are closed.
- The linked KDA evidence does not close `LB-F2D-K3`: LatentMoE is tracked by the following linked stage, while a composed K3 block/transformer/runner gate remains outstanding; no local result is represented as physical TorusXY or checkpoint-backed performance evidence.

#### `LB-F2D-K3` linked LatentMoE evidence (2026-08-12)

- Branch `pjosipovic/fabric2d-torus-xy-k3-latent-moe` is stacked on the reviewed KDA branch at `212f46453e0` and integrates the existing K3 LatentMoE implementation from PR #52453 at merge/migration commit `e20161fae4e`. The migration adds no test function, parameter value, mesh shape, or model configuration beyond that imported feature. It removes the imported 8×1 Fabric1d line proxy, migrates the unchanged 4×2 K3 module row to plain Fabric2D, and migrates the unchanged 8×4 production row to the shared TorusXY profile.
- The removed 8×1 proxy is not replaced by a new configuration. Its meaningful distributed LatentMoE contract is owned by the retained 4×2 module PCC row (including gate, dispatch/combine, down/up projections, distributed latent norm, and both mesh axes) and the retained production-shape 8×4 Galaxy row. The generic `tp_factor == 1` implementation branches remain available to non-K3 callers but are no longer claimed as K3 production coverage. This records the replacement ownership required before deleting redundant Fabric1d coverage.
- `TtLatentMoeProjections` no longer accepts a caller-selected topology. It derives the TP-axis topology from the active FabricConfig through `per_axis_topology`; plain Fabric2D resolves to Linear and certified TorusXY resolves to Ring. The CCL APIs still receive that derived value because topology remains an operation argument, but tests and production callers do not select it. A scoped audit finds no executable Fabric1d selection introduced by the LatentMoE stage.
- K3 gate loading remains real-weight-capable when `KIMI_K3_HF_MODEL` or the known mounted `/mnt/models/moonshotai/Kimi-K3` checkpoint is present. When neither is present, it uses the existing seeded fallback and does not resolve the router through Hugging Face's 16.6 GB shard. This makes the local gate hermetic without weakening a checkpoint-mounted Galaxy run.
- The existing shared Galaxy gate admits the two imported K3 gate cases and is count-pinned at 18. Its 600-second per-test cap covers the measured 475-second cold K3 pair, while its 17-minute job bound covers the prior eight-minute inventory plus that cold pair with about one minute of margin. The broad LoudBox and QuietBox directory selectors explicitly exclude K3 and remain pinned to their pre-import 65/28 configurations, preventing the imported 4×2/2×4/2×2 cases—including the four expensive 5K/25K K3 MoE rows—from silently expanding unrelated CI jobs. The dedicated K3 functional and performance Galaxy jobs each select one existing 8×4 row and export the TorusXY mesh descriptor; the perf wrapper additionally receives `MESH_DEVICE=TG`, while `PREFILL_TORUS_XY_CERTIFIED=1` remains owned exclusively by the workflow's successful TorusXY traffic preflight. Both jobs use fail-fast shells and invoke pytest directly; the local safe runner is not used for CI scheduling.
- The broad LoudBox exclusion also removes the imported standalone 2×4 `k3-6144` shared-expert row. It is not left unowned: the retained 4×2 K3 MoE PCC row exercises the same 6144-wide shared expert and passed `shared_output` PCC `0.999670` twice, while the dedicated 8×4 K3 MoE Galaxy row owns its production placement. No replacement test or configuration is added.
- The K3 MoE performance row is record-only after migration to TorusXY. Its historical 12,924,852 ns result was measured on unwrapped Fabric2D and is retained only as provenance, not asserted against Ring/Ring traffic. Execution requires both `PREFILL_TORUS_XY_CERTIFIED=1` and an explicit mesh-graph descriptor; the unchanged 5K production shape must be recalibrated in Galaxy CI before a TorusXY threshold is restored.
- The exact `./build_metal.sh --release` build passed at the imported merge, the hermetic-fallback source head `85bf63d6196`, the first review-correction head `9539bfc57eb`, and the final reviewed source head `f69c5793d0a` (`0.77.0-dev20260811+34.f69c5793d0`). Python compilation, YAML parsing, diff checks, pre-commit hooks, exact time-budget verification, and the scoped Fabric2D policy audit passed.
- Three locked local manifests ran through `scripts/run_safe_pytest.sh` without `--dev`, with exact nonzero `EXPECT_NUM_TESTS` values and durable logs/JUnit XML under `generated/lb-f2d-evidence/latent-moe-{host,gate,composed}-run{1,2}.{log,xml}`. The host K3 reference manifest passed 3/3 twice in 2.02 and 2.47 seconds. The unchanged K3 gate row passed 2/2 twice on Fabric2D 2×4 in 18.26 and 9.28 seconds. The representative unchanged 5K PCC MoE row passed 1/1 twice on Fabric2D 4×2 in 754.09 and 325.44 seconds; the second run reported `931/931` JIT-cache hits.
- Both composed runs reported gate recall 4/4 and identical PCCs: shared output `0.999670`, latent input `0.999808`, latent routed output `0.973403`, routed output `0.973228`, and final output `0.995783`, all above their retained thresholds. All six local invocations ended with `SAFE_PYTEST_RESULT: PASS`; every device run initialized `FabricConfig::FABRIC_2D`, and both repetitions closed all eight devices normally without reset, recovery, timeout, or hang. Therefore no `tt-triage` failure artifact was needed for this stage.
- Claude iteration 1 corrected stale CI counts, prevented K3 from expanding the broad LoudBox/QuietBox jobs, made the migrated performance result record-only, assigned ownership for removed 8×1 coverage, and required complete local evidence. Iteration 2 corrected time-budget provenance, standalone shared-expert replacement ownership, and the retained `0.965` final-output threshold comment. Iterations 3 and 4 made the shared gate's 600-second per-test and 17-minute job limits consistent with its measured 475-second cold K3 pair, added the missing Galaxy environment marker to the K3 perf row, and aligned gate-bias precision with the actual bf16 path. Iteration 5 preserved the workflow-owned TorusXY traffic-certification gate and removed dead bias-dtype plumbing. Iteration 6 reviewed the complete source/CI/plan diff at `f69c5793d0a` and returned exact `OK`; a final documentation-only exact-snapshot review still covers this recorded history before publication closes.
- Physical Ring/Ring routing, wrap-link traffic, the unchanged 8×4 functional and profiler rows, mounted real K3 router weights, TorusXY performance calibration, and full production composition remain Galaxy-only. The linked LatentMoE evidence closes the local `LB-05` component gate but does not close `LB-F2D-K3`: a composed K3 block/transformer/runner gate remains outstanding.

#### Required-case and skip policy

The collection manifest contains only cases required for the applicable status/model capabilities. Every manifest case must execute in both runs with zero failures, skips, or xfails. Optional real-weight cases are kept in a separate optional manifest and do not affect the milestone. Unsupported capabilities, such as trace on a branch that does not implement it, are listed explicitly as deferred and are not collected. Outside the manifest, no topology/hardware skip is permitted to hide a required case. This same bar applies to the Section 9.4 definition of done and the Phase 2 exit gate.

#### Reopen-failure policy

The normal gate is two complete passes without reset, proving close/reopen works. A reset between repetitions is permitted only as a time-limited `PASS_WITH_PLATFORM_WAIVER` when all of the following evidence exists:

1. `tt-triage` and safe-runner logs classify the failure before recovery;
2. the same Ethernet-initialization/reopen failure reproduces on an unmodified release baseline with the same test shape, showing it is not introduced by this migration;
3. each repetition is otherwise complete and clean, with no reset or recovery inside a repetition;
4. a platform issue, owner, expiry date, affected firmware/KMD/hardware identity, and exact recovery command are recorded; and
5. the code owner and final Claude review explicitly accept the waiver.

An unclassified failure, a failure unique to the migration branch, or any reset inside a required repetition blocks the milestone. The waiver does not satisfy production teardown/reopen acceptance and must be rechecked in Galaxy CI.

#### Required execution sequence

1. Run the release build and environment/device preflight.
2. Run each required command once with `scripts/run_safe_pytest.sh ... --collect-only`; save exact node IDs and counts in the milestone evidence. Do not use `--dev`.
3. Run host/config/schema groups.
4. Run device groups from narrowest to widest: disabled/single-device → 2×2 → canonical 2×4 → the locked pre-existing 4×2 axis-sensitive diagnostic families.
5. Run the complete required device group a second time without reset to validate teardown/reopen and persistent state cleanup. Apply the documented platform-waiver path only after baseline reproduction and triage.
6. Run the blocking no-Fabric1d/selector/count policy audit.
7. If a timeout/hang occurs, preserve the automatically generated safe-runner triage output and run `./tools/tt-triage.py` before recovery where possible. Fix, rebuild with `./build_metal.sh --release`, and restart the affected sequence.
8. Submit the complete evidence to the iterative Claude review loop and repeat until explicit `OK`.

#### Milestone evidence report

Attach one report to the implementation PR containing:

- source commit, branch, hostname/allocation, date, KMD/firmware, and eight device IDs/health;
- exact `./build_metal.sh --release` command, exit status, and build artifact identity;
- every required and optional safe-runner command, selected node IDs, `EXPECT_NUM_TESTS`, passed/failed/skipped/xfailed/deselected counts, duration, and log path, with the manifests clearly separated;
- active fabric and per-axis topology extracted from every communicating test log;
- PCC thresholds and observed values, trace/program-cache compilation counts, and repeat-run comparison;
- static policy-audit result and scoped remaining Fabric1d occurrences, which must be zero in tests/local CI;
- every `tt-triage` artifact or an explicit statement that no hang occurred;
- optional real-weight results clearly separated from required random-weight evidence;
- deferred Galaxy-only items mapped to Section 10; and
- all Claude review iterations, resolved findings, the final explicit `OK`, and any accepted platform waiver.

#### Definition of done

`LB-F2D` passes only when all of the following are true:

- `LB-F2D-main` is complete before K2.6/K2.7/GLM Phase 3 promotion; `LB-F2D-K3` is complete before K3 Phase 3 promotion; both are required for program-wide `LB-F2D` completion;
- all `LB-01` through `LB-13` deliverables applicable to the relevant status are complete;
- the release build and import/device preflight pass;
- every required-manifest selector has an approved exact nonzero count and executes with zero failures, skips, or xfails in both complete runs; optional or unsupported cases follow the required-case policy and cannot create a false green;
- logs contain no scoped `FABRIC_1D`/`FABRIC_1D_RING` initialization, and every communicating case reports Fabric2D Linear/Linear on the intended 2D shape;
- existing PCC bars are preserved, trace/program-cache behavior is stable after warmup, and the second run needs no recovery;
- no unresolved migration-caused local hang, topology fallback, stale selector, or required K3 branch evidence remains; any baseline-proven reopen limitation meets every `PASS_WITH_PLATFORM_WAIVER` condition;
- the evidence report explicitly labels all TorusXY, physical-wrap, production-depth, real-weight production, and performance conclusions as deferred to Galaxy; and
- the final Claude review iteration returns explicit `OK`.

#### Explicitly out of scope for `LB-F2D`

- opening or performance-testing any torus fabric on this LoudBox;
- proving Ring collectives or physical wrap-edge traffic;
- 8×4/32-device sharding, full production model depth, long-prompt production memory pressure, or Galaxy performance;
- Kimi K2.6/K2.7 and GLM-5.2 block/transformer composition, because their existing composition rows are 8×4-only and the matrix freeze forbids a 2×4 clone;
- `STRICT` TorusXY initialization and cabling validation;
- connected multi-Galaxy request/D2D behavior; and
- removing a Galaxy-only Ring workaround based solely on a local Linear/Linear pass.

## 10. What can only be tested in Galaxy CI

The following require a BH Galaxy allocation with the intended MGD and staged assets:

- `FABRIC_2D_TORUS_XY` bring-up on an 8×4 Ring/Ring physical topology;
- Ring collectives on both SP and TP, including both wrap edges;
- 32-device model sharding and Galaxy L1/DRAM pressure;
- two-link production behavior on both axes;
- full-depth K2.6/K2.7 (61), GLM-5.2 (78), and eventual K3 layer schedules;
- Kimi K2.6/K2.7 and GLM-5.2 block and chunked-transformer composition at their preserved 8×4 shapes;
- pretrained checkpoint and production TTNN cache loading at Galaxy shape;
- 55k/56,320-token chunked prompt accuracy and all-layer KV PCC;
- GLM compact full-indexer cache across all 21 full layers;
- trace capture/replay with production model depth and overlap;
- overlapped shared-expert + dispatch behavior on Ring/Ring, including the current deadlock exception;
- stable production performance baselines and profiler attribution;
- multi-Galaxy D2D sockets and 2/4/8-rank connected descriptors;
- hang recovery/timeout behavior under bad topology or link health.

CI must use a topology- and cabling-aware allocation. An MGD describes the desired logical graph but does not create physical wrap links. The generic `bh_sc1` allocation is currently skipped for these torus cases because its physical sub-torus wrap cabling is not guaranteed. Add/use a dedicated single-Galaxy SKU that combines:

- a scheduler label guaranteeing the node is physically cabled as the 8×4 XY torus;
- the existing `single_bh_galaxy_torus_xy_graph_descriptor.textproto` supplied through `TT_MESH_GRAPH_DESC_PATH`; and
- the hard-fail traffic/cabling preflight above.

Do not remove the global conftest protection or promote Phase 3 on an MGD-only `bh_sc1` change. If no cabling-certified single Galaxy is allocatable, Phase 3 is blocked; plain Fabric2D Galaxy can remain bring-up coverage but cannot meet the production TorusXY acceptance criteria.

Production TorusXY CI is intentionally fail closed: after selectors become TorusXY-only and `EXPECT_NUM_TESTS` rejects skips, an allocation that lacks or fails the cabling/traffic preflight must be red, not green-with-skips. `PREFILL_TORUS_XY_CERTIFIED=1` is therefore set only by a successful hard-fail preflight (or immediately after the inline preflight in the legacy release job), never unconditionally in a test-matrix row. The dedicated scheduler-certified SKU in `.github/sku_config.yaml` remains a Phase 3 provisioning prerequisite and must not be guessed from this local migration.

## 11. CI file migration

| File | Changes |
|---|---|
| `tests/pipeline_reorg/blaze_models_prefill_tests.yaml` | Replace every scoped 8×4 `line`/ambiguous `mesh-8x4`/Fabric2D-mesh selector with `torus-xy-8x4`; launch full runner tests through TorusXY rank bindings; reassign existing Kimi slots across K2.6/K2.7/K3 without increasing row count; split accuracy/perf only within the existing jobs; update timeouts and expected counts. |
| `.github/workflows/blaze-models-prefill-tests.yaml` | Update dispatch help/test types for new K2.7/K3/full-runner rows and enable the topology-aware Galaxy SKU. |
| `.github/sku_config.yaml` | Add/select a cabling-certified single-Galaxy TorusXY SKU whose scheduler label guarantees physical XY wrap links and whose environment sets the existing descriptor path. An MGD-only variant of generic `bh_sc1` is insufficient. |
| `.github/time_budget.yaml` | Rebudget after measuring TorusXY jobs; recover time by pruning duplicated op matrices rather than hiding production tests in broad jobs. |
| `tests/pipeline_reorg/blackhole_e2e_tests.yaml` | Convert the four referenced LB jobs plus `bh_qb_DeepSeek_PREFILL`, `bh_qb2_DeepSeek_PREFILL_OP_TESTS`, and `bh_qb2_DeepSeek_D2D_SOCKET_SYNC` to Fabric2D-only selectors. Move all communicating coverage out of the 1-chip `bh_p150_*` and 2-chip `bh_p300_*` jobs; keep only fabric-disabled local cases there. Rename `PREFILL_OP_TESTS` to reflect diagnostic/local coverage after production gates move to Galaxy. Remove `line`, `linear-8`, `not fabric2d`, and Fabric1d perf cases. |
| `tests/pipeline_reorg/galaxy_health_tests.yaml` | Reuse the existing Blackhole XY-torus cabling validation as a required dependency of each production prefill allocation; retain hard-fail and traffic injection. |
| `tests/pipeline_reorg/demo_sp_release_tests.yaml` | Do not create a second matrix if this pipeline is being retired. Point release gating to the canonical Blaze production rows; until removal, update its remaining Kimi/GLM LB selectors to Fabric2D and its Galaxy selectors to TorusXY. |
| `tests/pipeline_reorg/t3k_e2e_tests.yaml` | Retain the model-level Fabric2D 2×4 gates, but remove the stale Wormhole broad op job: it mixed unsupported Blackhole-only rows and topology skips and is not a substitute for Blackhole Galaxy TorusXY sign-off. |
| existing collection/lint policy checks | Validate forbidden Fabric1d tokens, profile/descriptor pairing, exact CI selectors, matrix test types, and `EXPECT_NUM_TESTS` collection/execution counts without adding a test or configuration. Make empty selection a hard failure: a renamed selector such as `mesh-8x4` → `torus-xy-8x4` must have an explicitly verified nonzero expected count. Run it in the workflow's existing validation stage. |

## 12. Phased implementation and merge gates

### Mandatory per-phase execution and Claude approval loop

Every phase below follows the same gated loop; this is part of the phase definition, not an optional final review:

1. Implement only that phase's scoped changes and update its inventory/checklist.
2. Build from the repository root with `./build_metal.sh --release`. Do not substitute a debug, development, or pre-existing build.
3. Run all local phase tests through `scripts/run_safe_pytest.sh` **without `--dev`**. CI jobs use their normal direct `pytest` scheduling commands. Capture exact selected/executed/skipped counts, topology logs, local safe-runner logs, and any triage artifacts.
4. Ask Claude to review the phase plan, code diff, build result, test evidence, topology invariants, and unresolved risks using `claude --dangerously-skip-permissions ...`. The review prompt must require either an explicit `OK` or a concrete blocking finding, not a vague summary.
5. Address every blocking finding, rebuild, rerun affected safe tests, and ask Claude to review again. Repeat as many iterations as necessary until Claude returns explicit `OK` for that phase.
6. Save each review iteration and the final `OK` in the implementation PR or a linked artifact so reviewers can see what changed between iterations. Human/code-owner approval remains required; Claude's `OK` is an additional gate, not a replacement.

Suggested non-interactive review shape, with the phase number and evidence paths filled in:

```bash
claude --dangerously-skip-permissions --print \
  "Review Fabric2D/TorusXY migration Phase <N>. Inspect the plan, current diff, release-build result, safe-pytest logs, and tt-triage artifacts. Check correctness, production TorusXY invariants, missing tests, false-green skips, and regressions. Return OK only if there are no blocking findings; otherwise list concrete required changes."
```

No phase may start its successor until its implementation exit gate, build/test gate, and iterative Claude `OK` gate all pass.

### Phase 0 — Freeze the baseline and add observability

1. Save the referenced CI job/test counts and durations as the before-state.
2. Add structured logging of FabricConfig, MGD, dim types, per-axis topology, link count, model, mesh shape, and rank.
3. Add a cheap static CI audit that reports all scoped Fabric1d uses, initially non-blocking.
4. Add fail-fast operation timeouts to all new torus bring-up jobs.

Exit gate: logs make it unambiguous which topology every job ran; no behavior change yet; release build and safe-runner checks pass; Claude returns explicit `OK` after all review iterations.

### Phase 1 — Land the explicit topology contract

1. Implement `TopologyProfile` resolution/validation in common prefill code.
2. Make production manifests/rank bindings require `bh_galaxy_torus_xy`.
3. Remove production constructor defaults to Linear; pass the per-axis tuple through runtime, transformer, block, MLA, MoE, FFN, norm, LM head, and DFlash if applicable.
4. Validate profile mapping and descriptor mismatch rejection through existing host/import/config paths; add no test.

Exit gate: an omitted mode or mismatched descriptor fails before device/model initialization; existing TorusXY rank bindings pass preflight; release build and safe-runner checks pass; Claude returns explicit `OK` after all review iterations.

### Phase 2 — Make LoudBox coverage Fabric2D-only

Execute the complete `LB-F2D` milestone in Section 9.4:

1. Replace central Fabric1d params in `mesh_configs.py` and test `conftest.py`.
2. Convert the four referenced `bh_lb_DeepSeek_*` jobs and both 4-chip QuietBox suites to exact Fabric2D selectors; move communicating p150/p300 cases off the 1/2-chip jobs.
3. Convert sparse MLA, dense MLA, MoE, block, cache, DFlash, and op diagnostics.
4. Make the static no-Fabric1d audit blocking for the scoped test/CI directories.
5. Lock the applicable collection manifest, run the required suite twice, and publish the `LB-01` through `LB-13` evidence report.
6. Record new counts/durations; every required selector must execute its exact nonzero count with zero failures, skips, or xfails; remove skips that existed only for incompatible 1D shapes.

Exit gate: `LB-F2D-main` must pass before K2.6/K2.7/GLM Phase 3 begins, and `LB-F2D-K3` must pass before K3 Phase 3 begins. For each applicable status, all Section 9.4 definition-of-done items pass: zero Fabric1d/Fabric1d-ring initialization, exact nonzero required-manifest execution with zero failures/skips/xfails, matching reference PCC, a release build, and explicit Claude `OK` after all review iterations. Program-wide Phase 2 is complete when both statuses pass.

### Phase 3 — Shadow TorusXY Galaxy module gates

1. Provision a physically XY-wrap-cabled, scheduler-labelled single-Galaxy allocation, attach the existing TorusXY MGD, and require the cabling/traffic plus topology preflight. Do not proceed on MGD alone.
2. Reassign the existing K2.6/GLM module rows one-for-one to `torus-xy-8x4`; do not clone rows or grow the matrix. If a temporary comparison is required, use the retained bounded TorusY diagnostic manually rather than adding a CI configuration.
3. Fix ring-specific hangs/numerics one operation at a time, starting with the documented multi-hop-over-wrap MoE dispatch hang, shared-expert concurrency, sparse MLA's Linear fallback, and MLA SP/TP axis separation. Use TorusY only as a bounded differential diagnostic.
4. Reassign existing Kimi checkpoint slots to cover K2.7 while retaining a K2.6 anchor elsewhere.
5. Integrate K3 component rows as their dependency PRs land.
6. Recalibrate performance only after correctness/determinism is stable.

Exit gate: physical cabling validation passes; every row executes the exact expected nonzero count with zero topology/hardware skips; the multi-hop-over-wrap dispatch blocker is cleared; and three consecutive scheduled runs pass without hang on at least two cabling-certified Galaxy allocations. Trace/no-trace agree where supported, the release build passes, and Claude returns explicit `OK` after reviewing the CI and triage evidence over as many iterations as necessary.

### Phase 4 — Promote full production runners

1. K2.6 full-depth producer/runner on TorusXY.
2. K2.7 full-depth producer/runner on TorusXY.
3. GLM-5.2 full-depth/two-cache producer/runner on TorusXY.
4. K3 full-depth producer/runner after all components compose.
5. Add a two-rank D2D TorusXY smoke for at least K2.7 and GLM/K3 where product deployment needs it.

Exit gate: real-weight, full-depth KV/output correctness and request-mode transport pass through the same rank-binding path used by production; the release build passes; Claude returns explicit `OK` after all review iterations.

### Phase 5 — Move the release gate and prune op matrices

1. Mark the production Galaxy rows required.
2. Remove old Fabric2D-mesh Galaxy shadow rows and every Fabric1d production row.
3. For each mapping in section 7, prune/demote duplicated op params only after the higher-level test has demonstrated fault sensitivity.
4. Rename/rebudget LB jobs around their smaller diagnostic role.
5. Make topology-policy validation permanent.

Exit gate: production release status is determined by TorusXY Galaxy tests, while LB and op tests provide fast diagnosis only; the release build and safe-runner checks pass; Claude returns explicit `OK` after all review iterations.

### Phase 6 — Close the Ring overlap exception

1. Reproduce the shared-expert Ring reduce-scatter + dispatch deadlock in a focused TorusXY **no-trace production-mode** stress case; trace capture disables the overlap and is not evidence for this gate.
2. Fix EDM credit/scheduling or collective ordering so both can overlap without forcing Linear.
3. Run long repeated no-trace overlap stress, plus trace/non-trace correctness comparison, and compare throughput to the workaround.
4. Delete `force_shared_expert_linear` and its exception from the acceptance criteria.

Exit gate: every physically wrapped production collective uses Ring, including overlapped shared expert; release build, safe-runner, long-stress, and triage checks pass; Claude returns explicit `OK` after all review iterations.

## 13. Failure handling and rollback

- Never fall back from TorusXY to Fabric2D/Fabric1d inside a production job. A failure must remain visible.
- Roll back a phase by disabling the new required CI row or reverting that phase's call-site change; do not restore a silent runtime default.
- Topology bring-up and collective tests need bounded timeouts and per-layer/per-chunk progress markers so a hang identifies the last operation.
- Run locally launched tests through `scripts/run_safe_pytest.sh` without `--dev`, allowing its dispatch-timeout handler to invoke `tt-triage`. Keep CI scheduling on direct `pytest`. On any suspected hang, run/preserve `tt-triage` (`./tools/tt-triage.py`) diagnostics before device reset, process termination, or other recovery whenever the system is still inspectable.
- On a hang, preserve the safe-runner log, `tt-triage` report, call stacks, fabric/route/telemetry logs, active descriptor/profile, last layer/chunk/op, and allocation identity. Attribute descriptor mismatch, physical-link health, operation deadlock, and numerical failure separately. Attach the evidence to the phase's next Claude review iteration.
- Retune perf baselines after topology migration. Ring can legitimately improve or regress an isolated operation; use end-to-end throughput as the release metric and module signposts for diagnosis.
- Update time budgets in the same PR as CI matrix changes.

## 14. Acceptance checklist

- [ ] `LB-F2D-main` and its applicable `LB-01` through `LB-13` evidence are complete before K2.6/K2.7/GLM Phase 3; `LB-F2D-K3` is complete before K3 Phase 3.
- [ ] Every phase was built with `./build_metal.sh --release`; no stale/debug/development build was used as evidence.
- [ ] Every locally invoked scoped test used `scripts/run_safe_pytest.sh` without `--dev`; no CI schedule uses that wrapper.
- [ ] Every hang has a preserved `tt-triage` report and associated topology/fabric evidence captured before recovery where possible.
- [ ] Every phase has a saved multi-iteration Claude review record ending in explicit `OK`, with all earlier blocking findings resolved and affected code rebuilt/retested.
- [ ] No scoped test or CI entry contains/opens `FABRIC_1D` or `FABRIC_1D_RING`.
- [ ] Communicating LB tests use Fabric2D; pure local tests disable fabric.
- [ ] All K2.6/K2.7/K3/GLM-5.2 production manifests require TorusXY.
- [ ] Production runner cannot silently choose a fabric from SP size.
- [ ] Single-Galaxy CI allocation supplies the Ring/Ring MGD explicitly.
- [ ] That allocation is scheduler-certified for physical XY wrap cabling and passes hard-fail traffic validation; the MGD is not treated as proof of wiring.
- [ ] Galaxy logs assert `FABRIC_2D_TORUS_XY` and `(Ring, Ring)`.
- [ ] Production jobs use the `STRICT`, two-channel single-Galaxy descriptor contract.
- [ ] Every production selector has a checked nonzero `EXPECT_NUM_TESTS`, executes that exact count, and reports zero topology/hardware skips.
- [ ] K2.6 full runner passes real-weight full-depth KV/output checks.
- [ ] K2.7 full runner passes its checkpoint-specific full-depth checks.
- [ ] GLM-5.2 full runner validates KVPE plus compact index-K cache for all required layers.
- [ ] K3 MLA, KDA, AttnRes, LatentMoE, block, transformer, and full runner are integrated and pass.
- [ ] Trace/no-trace correctness is covered for every supported model.
- [ ] Production perf is baselined on TorusXY, not inherited from Fabric1d/Fabric2D mesh.
- [ ] At least a two-rank connected TorusXY D2D production smoke passes.
- [ ] Op-level production matrices have been pruned only after their Galaxy replacements are stable.
- [ ] Shared-expert overlap Ring exception is fixed or explicitly product-accepted with an owner and expiry.
- [ ] The no-trace serving path with overlap enabled has been exercised; trace-only coverage is not accepted for the Ring-overlap gate.
- [ ] The documented multi-hop-over-wrap MoE dispatch hang is resolved on TorusXY.
- [ ] Three consecutive scheduled Galaxy runs pass on more than one allocation.

## 15. Review record

Independent review completed with the requested Claude CLI (`claude --dangerously-skip-permissions ...`) on 2026-08-11. Its findings were checked against the repository and incorporated above. The material corrections were:

- physical XY wrap cabling and traffic validation are mandatory; an MGD-only `bh_sc1` change is unsafe;
- five additional Blackhole jobs, the D2H-ack rank bindings, the existing TorusXY param/descriptor, sparse MLA's topology fallback, and both known MoE Ring hangs are now explicitly inventoried;
- single-Galaxy production descriptor validation now fails closed through `TT_MESH_GRAPH_DESC_PATH`, Ring/Ring parsing, descriptor-derived `STRICT` reliability, and an explicit model on every rank; connected descriptors retain their declared relaxed channel contract;
- Phase 3 and acceptance now require exact nonzero execution with zero topology skips, so a broad conftest skip or stale `-k` selector cannot create a false green; and
- the overlap exception is correctly scoped to no-trace serving, because trace capture disables that overlap.

No review finding remains unresolved. Implementation details such as the final scheduler SKU name and the owner/expiry for any temporary no-trace overlap waiver must be assigned in the corresponding implementation PR.

For implementation, this one-time plan review is only the starting point. Section 12 requires a fresh Claude review after each phase and repeated review/fix/build/test iterations until that phase receives explicit `OK`.

The `LB-F2D` milestone received its own two-iteration Claude review on 2026-08-11:

1. The first completed review identified four blockers: inconsistent skip bars, ambiguous `main` versus K3 completion, unresolved DFlash scope, and an unachievable reopen gate if a baseline platform failure requires reset.
2. The milestone was revised with one required-manifest skip policy, separate `LB-F2D-main`/`LB-F2D-K3` statuses, an explicit DFlash deliverable/disposition, and a tightly evidenced `PASS_WITH_PLATFORM_WAIVER` path.
3. The second review returned explicit `OK` with no remaining blocking feasibility or acceptance ambiguity.

The implementation diff then entered its required iterative Claude review loop. The third implementation review found six concrete blockers: two sparse-cache mesh rows had been exchanged, a Fabric1d block-perf baseline was still asserted for Fabric2D, the CI torus guard admitted a TorusY diagnostic, LoudBox selectors admitted unsupported shapes, topology mapping was duplicated without a production Ring/Ring assertion, and dead perf code/imports remained. The implementation now restores the original 4×2 cache-only and 8×4 GLM shared-layer ownership, makes the migrated 2-link Fabric2D perf row record-only pending recalibration, keeps TorusY out of CI, count-guards exact supported LoudBox selections, centralizes FabricConfig-to-axis-topology resolution, asserts production TorusXY is `(Ring, Ring)` immediately after device open, and removes the dead code. A fresh exact `./build_metal.sh --release` and final Black/isort/autoflake checks pass. The next review iteration must inspect these resolutions and return explicit `OK` before this implementation milestone can close.

Implementation review iteration 4 found six further blockers. The fixes are now applied: perf wrappers remove and restore the outer `EXPECT_NUM_TESTS` around every tracy child; all remaining Fabric1d/unwrapped-Fabric2D performance numbers are record-only until calibrated on their new fabric; the LoudBox PCC selector again schedules the existing shared-expert, LM-head, and parallel-embedding 2×4 rows (65 exact cases, with the 11 restored component cases passing locally); descriptor parsing derives `STRICT_INIT` only when every mesh/connection channel is strict and preserves relaxed connected-Galaxy descriptors; the count hook is hoisted so the common full-runner gate is enforced; and certified CI admits only TorusXY per item while TorusX/Y stay skipped diagnostics. Local validation reproduced the formerly inert common-runner count failure and correct-count pass, verified strict single-Galaxy versus relaxed connected-Galaxy profile resolution, and showed the real block-perf child selects its one worker without inheriting the outer count before the unavailable local checkpoint caused the run to be stopped. Iteration 5 must review these corrections and still return explicit `OK` before closure.

Implementation review iteration 5 confirmed all six iteration-4 blockers closed and found one remaining duplicated transformer parameter: an explicit Fabric2D 2×4 row had become identical to the shared mesh-list row spliced into the same parametrization. The explicit duplicate is removed, along with its now-unused import. Recollection now produces 33,600 total transformer cases; the representative Fabric2D 2×4 filter selects 84 stable IDs with no pytest-generated `_0`/`_1` suffix. This reduces the matrix rather than adding coverage. Iteration 6 must confirm the correction and return explicit `OK`.

Implementation review iteration 6 confirmed the transformer duplication and all earlier blockers closed, then found two remaining enforcement gaps. First, both MoE Gate PCC entry points still inherited `TtMoEGateConfig`'s Linear TP-all-reduce default even for the certified TorusXY row. They now receive the existing indirect `device_params` fixture and derive the TP topology from the shared FabricConfig mapping; the unchanged TorusXY CI slice still collects exactly 16 cases, while representative existing Kimi device-gate and V4 hash-device cases each pass 1/1 locally on Fabric2D 2×4. Second, the required no-Fabric1d/exact-selector policy existed only in this plan. A repository pre-commit hook now runs `scripts/validate_prefill_fabric_policy.py`: it AST-checks the scoped Python source, narrowly allowlists only the established compatibility mapping sites, and rejects ambiguous scoped pipeline `-k` selectors. The sparse case IDs now give their existing mesh field the explicit `shape-` prefix; the two Galaxy commands select model, `shape-8x4`, and `torus_xy`, retain their exact 6/7 count contracts, and no longer depend on an ambiguous bare `8x4` token. This is an ID-only change to existing cases, not a new row or configuration. The hook passes through pre-commit, `git diff --check` passes, and a fresh exact `./build_metal.sh --release` succeeds. Iteration 7 must review these corrections and return explicit `OK` before closure.

Implementation review iteration 7 confirmed the iteration-6 fixes and found three further fail-closed/coverage gaps. The runner's cross-rank fingerprint now resolves the topology profile before mesh open and includes fabric mode, descriptor path, per-axis topology, and reliability mode; a host-injected divergent digest raises a named error containing all four fields. Sparse Galaxy generation now contains only the full 8×4 mesh, removing the unsafe/redundant 8×2 submesh that could not carry the descriptor's TP wrap edge. Finally, the five legacy 1×4 communicating rows are gone: FFN and RMSNorm were migrated one-for-one to 2×2, while shared expert, LM head, and parallel embedding reuse their existing 2×4/2×2 rows rather than adding duplicates. The exact QuietBox selector is 28 cases. Using `TT_VISIBLE_DEVICES=0,1,2,3` on LoudBox, the FFN/RMSNorm slice passed 4/4 after a safe-runner-diagnosed and recovered timeout in the now-removed redundant parallel-embedding candidate, with the triage artifact and reset recorded above. Iteration 8 must review these corrections and return explicit `OK` before closure.

Implementation review iteration 8 confirmed all iteration-7 fixes, every count-guarded selector, the no-new-test/no-expanded-matrix constraint, and the production descriptor preflight. It found one remaining false-green path: the Kimi prefill-block parameter had moved from K2.6 to K2.7 while the shared CI skip exemption still named only K2.6, and `EXPECT_NUM_TESTS` enforced collection but not execution. The exemption now follows both supported Kimi 2.x variant names, so the existing non-balanced K2.7 production row is not skipped. The root count guard now also requires every selected item to pass and rejects any skip, xfail, or xpass; collect-only validation remains available for scheduling checks. Existing local tests through `scripts/run_safe_pytest.sh` demonstrate a guarded 1/1 pass, a guarded existing skip failing with `passed=0, skipped=1`, and the same one-item collect-only invocation passing. The physical K2.7 8×4 TorusXY 2/2 execution remains a Galaxy-CI acceptance item, not local LoudBox evidence. Iteration 9 must review these corrections and return explicit `OK` before closure.

Implementation review iteration 9 confirmed the K2.7 predicate, execution-hook semantics (including xfail rejection and perf-child environment scrubbing), all guarded selector counts, TorusXY fail-closed behavior, 8×4 preservation, and plan accuracy. It found that two existing Blaze legs placed multiple guarded commands inside `bash -lc` without fail-fast semantics, allowing an earlier failed pytest or dependency install to be masked by a later successful command. The existing KV-cache-table and Kimi-prefill-block inner scripts now begin with `set -e`, so every guarded command determines the leg result. YAML parsing and a local shell fail-fast probe validate the scheduling fix; the scheduled commands remain direct pytest, not the local safe runner. The review also noted that generic multi-rank binding delivery of a shell-selected manifest should be made self-consistent before Phase 4. That is retained as a fail-closed Galaxy/multihost integration item rather than claimed as completed LoudBox work: current ranks cannot silently choose a default model, and no local single-Galaxy evidence can validate multi-context environment delivery. Iteration 10 must review the fail-fast correction and return explicit `OK` before closure.

Implementation review iteration 10 re-read the complete diff and plan, confirmed that `set -e` is placed only in the two affected multi-command Blaze blocks before every fallible command, and re-verified the earlier topology, selection, execution-count, 8×4-preservation, and no-expansion constraints. It returned exact `OK`. The local LoudBox milestone is therefore review-complete; the explicitly identified physical 8×4 TorusXY, production-checkpoint, and multihost binding items remain gated on Galaxy CI phases rather than being represented as local evidence.

The stage-9.4 completion audit reopened review after the complete locked manifest was assembled. Implementation review iteration 11 confirmed the earlier findings closed and accepted the repeated Device-6 router-handshake failure as an honestly recorded, unwaived external blocker. It found six new source/CI consistency issues: one unscheduled sparse-perf diagnostic inferred TorusXY from device count; the XY cabling probe had replaced the shared deployment health descriptor; the plan understated the pre-existing 4×2 MoE diagnostic ownership; descriptor dimensions were parsed but not checked; the policy validator had enum/config/workflow/multiline blind spots; and a test helper, redundant 32×4 sanity test, plus non-fail-fast single-pytest Blaze shells retained low-level drift. The fixes keep that diagnostic on stable-ID Fabric2D, separate deployment health from non-fatal XY certification (only a successful probe writes the certification environment), preserve/document the existing 4×2 MoE rows without adding configurations, reject mesh/descriptor dimension mismatch, harden the validator, derive all remaining scoped helper topologies from active FabricConfig, remove the redundant sanity test in favor of production preflight, and put every Blaze rank shell under `set -e`. A fresh exact release build, policy/pre-commit/YAML/compile checks, descriptor positive/negative probes, and the locked 37-test host manifest are rerun before iteration 12. Iteration 12 must inspect these corrections and return exact `OK`; the hardware blocker remains open and is not converted into milestone completion.

Implementation review iteration 12 confirmed all iteration-11 corrections and independently reproduced the descriptor/profile checks. It found the same device-count Torus inference in the sibling sparse CCL-perf diagnostic and found that its old Fabric1d SP proxy had only been re-enumed onto a degenerate 8×1 Fabric2D shape. The remaining CCL diagnostics are now stable-ID, unwrapped Fabric2D on their existing supported 2×4/8×4 mesh chosen only for capacity; no device count can select TorusXY. The redundant 8×1 SP proxy and its dead shape/path helpers are removed rather than replaced with a new configuration, with the full sparse-MLA production Galaxy gate owning that collective. Iteration 13 must confirm these final corrections and return exact `OK`; the external hardware blocker remains unchanged.

Implementation review iteration 13 confirmed the iteration-12 CCL fixes and the zero-growth inventory, then found the same legacy line shapes in sparse MLA vs-trace plus unreachable all-gather framework code left behind in the CCL diagnostic. The vs-trace mesh helper now exposes only 2×2 on QuietBox, 2×4/4×2 on LoudBox, and 8×4 on Galaxy, with stable Fabric2D IDs; the 8×1, 1×4, and 1×1 communicating rows are removed rather than replaced. LoudBox collection is locked at 42 device cases. The unreachable all-gather driver, semaphore setup/import, topology-dependent roofline arm, optional path fields, and unused resolver argument are deleted. The validator now rejects legacy topology-only pytest IDs and prevents degenerate shapes from returning to the central sparse mesh helper. Iteration 14 must confirm these corrections and return exact `OK`; the external hardware blocker remains unchanged.

Implementation review iteration 14 confirmed the iteration-13 vs-trace and CCL cleanup, then found a second sparse-MLA mesh table still carrying a QuietBox 1×4 row and synthesizing 1×N meshes for unknown device counts. The redundant 1×4 row is removed without replacement, leaving the fixed 2×2 QuietBox profile; unsupported boxes now collect a skipped 2×2 sentinel instead of opening a fabricated line mesh. The policy validator covers both sparse mesh shape tables so degenerate communicating profiles cannot return through either selector. Iteration 15 must confirm these corrections and return exact `OK`; the external hardware blocker remains unchanged.

Implementation review iteration 15 confirmed the iteration-14 selector fixes, then found the same QuietBox 1×4 and unsupported 1×1 patterns in the separate sparse-MLA perf selector. That existing perf row is migrated in place to 2×2, while unsupported boxes now carry a collection-time skipped 2×2 parameter so no fixture can open a degenerate mesh. The validator now audits every scoped `*BY_DEVICE_COUNT` mesh table plus the sibling sparse mesh table, instead of relying on two exact variable names. A stale removed-Galaxy-8×2 comment is corrected, and the masked-bincount Fabric2D row's physical requirement and ID now match its actual 2×4 mesh. Iteration 16 must confirm these corrections and return exact `OK`; the external hardware blocker remains unchanged.

Implementation review iteration 16 confirmed all iteration-15 topology and selector corrections, then found that the sparse-MLA perf harness still described and printed the migrated QuietBox 2×2 row as if it retained TP=4 head sharding. The workload now records the unchanged full model head counts actually consumed by the test, and reporting distinguishes the Galaxy-local head shard from the proxy-local shard. The plan and harness explicitly record that the required 2×2 profile doubles QuietBox's per-chip head-dependent compute/storage relative to TP=4, making it a heavier functional/perf diagnostic whose physical execution and memory headroom require a healthy QuietBox; it is not claimed as Galaxy-equal performance evidence. Stale QuietBox cache/sequence scaling statements are corrected. Iteration 17 must confirm these corrections and return exact `OK`; the external hardware blocker remains unchanged.

Implementation review iteration 17 independently verified the full-head accounting, per-box cache/chunk scaling, collection-time unsupported-box skip, validator coverage, unchanged 27-case perf inventory, and the plan's explicit QuietBox headroom limitation. It found no remaining actionable issue and returned `OK`. The source/review gate is complete; the external Device-6 router-handshake blocker remains open and prevents claiming the physical `LB-F2D-main` execution gate.

Final implementation sign-off iteration 18 re-read the complete working tree and recorded review history, found no actionable source, scope, inventory, CI, validation, or plan gap, and returned exact `OK`. The external Device-6 router-handshake failure remains an unwaived platform blocker rather than a source defect.

Post-sign-off allocation-evidence review iteration 19 confirmed the repeated seven-device triage result, Ethernet retrain failures, low shared-memory headroom, lack of active test/MPI processes, and absence of any reset or waiver. It found that the new evidence text incorrectly said device 3 was absent "this time," although the original timeout snapshot also omitted device 3. The evidence now states that device 3 was absent in both passes, strengthening the repeated external-fault diagnosis. Iteration 20 re-verified both artifacts and the corrected allocation-health account, then found this review-history omission. The history is now complete; iteration 21 must confirm it and return exact `OK`. The Device-6 router-handshake/allocation-health blocker remains open and unwaived.

Allocation-evidence review iteration 21 confirmed the corrected device-3 account, complete review history, unchanged source sign-off, and open unwaived platform blocker, then returned exact `OK`.

After allocation recovery, stage-9.4 execution review iterations 22–24 audited the locked local manifest while the hardware work ran. Iteration 22 found that the 122-case op selector collected 26 rows whose test bodies deliberately skipped, so the release gate was narrowed to executable existing rows rather than accepting skips. Iteration 23 found two additional combine rows that pass locally but skip under `CI=true`, then found 26 more CI-only cache/rotary/row-major duplicates; the shared local/CI selector was reduced to the exact 68 supported rows without adding or changing a test/configuration. Iteration 24 independently collected and executed that selector with `CI=true`, confirmed `68/68` passed with zero skips, verified the corrected transformer `e256_host` selector, all plan counts, and the retained supported PCC/perf/dtype/compression paths, and returned exact `OK`.

Merging current `origin/main` reopened the review and certification gate. Iteration 25 found that the newly merged K3 TorusXY Blaze commands did not explicitly export the required MGD, the policy validator did not enforce that co-location, six guarded selectors had drifted after upstream inventory growth, the cache 2×2 marker still said `linear`, and the plan overstated post-merge closure and `_all_gather` provenance. The K3 commands now fail closed with the single-Galaxy TorusXY descriptor; the validator requires that descriptor in the enclosing YAML command body; the existing Blaze, demo-SP, T3K, QuietBox cache, and QuietBox op schedules are narrowed back to their prior exact inventories without adding tests/configurations; and the marker/evidence text is corrected. Local safe-runner collection confirms the corrected 4-case T3K/Blaze-equivalent 2×4 chunked selector, 2-case Kimi and DeepSeek demo-SP selectors, 13-case non-K3 cache selector, 205-case pre-existing op selector, and K3 Galaxy counts of 2 functional plus 1 perf. Policy positive/negative probes and all pre-commit checks pass. The fresh exact release build passed. The affected `CI=true` op gate passed 68/68 with zero skips (`1710` deselected) in 584.64 seconds on Fabric2D, and all four corrected 2×4 DeepSeek/Kimi chunked rows passed CPU-reference output and KV-cache checks in 113.63 seconds; both safe-runner invocations closed all eight devices normally. Iteration 26 is reviewing these corrections and must still return exact `OK` before the post-merge gate closes.

Iteration 26 confirmed the iteration-25 topology, descriptor, selector-inventory, provenance, and 8×4-preservation fixes, then found five remaining blockers. Three execution-strict CI commands still selected guaranteed skip/xfail rows; K3 MLA perf and the old dispatch/combine perf wrapper carried zero-selecting embedded selectors; the newly merged top-k=1 reduce diagnostic had been reshaped incompatibly and hidden; the K3 cold/warm cache row was excluded from its only device-CI job; and eight sibling 2×2 cache markers still said `linear`. The LoudBox block job is narrowed to three nonredundant executable model anchors and eight executable block-loop rows, while the unsupported broad Wormhole op job is removed. K3 MLA perf now selects exact `torus-xy-8x4`, requires certification, and is record-only pending Ring/Ring recalibration. The stale unscheduled 8×1 dispatch/combine perf wrapper and non-production top-k=1 reduce row are removed rather than replaced with new matrices: no scoped production model uses top-k=1, the retained model-shaped reduce and composed production gates cover the relevant path, and preserving the diagnostic would require the same degenerate-axis configuration this migration removes. QuietBox retains all 14 cache cases including K3, every 2×2 cache marker names `mesh-2x2`, and the validator now audits Python-embedded pytest selectors as well as YAML. Exact safe-runner collection is 3/8/14/205 for the corrected block/block-loop/cache/op jobs and one for the K3 perf child. The three block anchors physically passed 3/3; the eight loop cases are collection-only locally because the required MLPerf checkpoint mount is absent. The refreshed host manifest passed 37/37, the post-merge `CI=true` op gate passed 68/68 twice with zero skips/xfails, and the exact release build passed. Iteration 27 must verify these corrections and zero-growth removals and return exact `OK` before the post-merge gate closes.

Iteration 27 independently re-derived the 3/8/14/205/1 collection contracts, verified selector substring safety and strict execution semantics, accepted the zero-growth test removals, and confirmed the Fabric2D/TorusXY topology and descriptor behavior. It found one policy-enforcement blind spot: the new embedded-selector audit used raw source for an implicitly concatenated f-string, so the quote boundary before `-k` prevented its regex from seeing the K3 perf selector. The audit now reconstructs a joined string from its literal AST values and placeholder tokens. An in-memory negative probe using the same implicit f-string concatenation proves that ambiguous `mesh-8x4` is rejected, while the corrected repository passes the policy validator, Python compilation, diff checks, and pre-commit. Iteration 28 must verify this final enforcement fix and return exact `OK`.

Iteration 28 verified the real joined-string AST shape, the negative probe, the corrected K3 selector, and the complete working-tree diff, found no remaining actionable stage blocker, and returned exact `OK`. A final exact `./build_metal.sh --release` also passed against the fully corrected tree. The post-merge local review gate is closed; mounted LoudBox real-weight loop execution and physical Galaxy TorusXY execution remain the explicitly assigned CI gates.

Iteration 29 reviewed merge commit `ada58b94d52`, which integrated six newer `origin/main` commits without changing any scoped model, common-prefill, pipeline, or policy file. The exact release build and import passed after the merge; the policy validator remained clean; and the existing K3 Fabric2D 2×4 cold/warm MLA cache gate exact-collected and passed 1/1 at PCC 1.0 with a normal eight-device close. The review confirmed that the merge does not invalidate the stage-9.4 evidence and returned exact `OK`. Program-wide `LB-F2D` remains open because the three named K3 component PRs are still draft/unmerged and no composed K3 block/transformer gate exists.
