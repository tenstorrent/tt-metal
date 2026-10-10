# Kimi K3 prefill: two 4x4 stages on one Blackhole galaxy

Status: plan v5.1. Scoped to a single galaxy, ordered bottom-up. Reviewed in two rounds by a separate
Claude instance and Codex gpt-6-astra (section 8). Not started.

## 1. Goals and success metrics

**Goal.** Run K3 prefill on one BH galaxy as **two 4x4 pipeline stages** (one process per stage,
12 + 12 of the 24 layers the single-galaxy config runs today) instead of one 8x4 stage. Each 4x4
runs one mesh axis as a ring and the other as a line. Decide whether the split is worth it.

**Baselines** (same galaxy, same build, `PREFILL_SYNC_PER_CHUNK` on and off):

- **B1**: today's single-galaxy K3 run, one 8x4 stage, 24 layers, 5120-token chunks
  (640 tokens/chip). `ci/run_multirank_pcc.sh kimi_k3 sc1`.
- **B2**: the same 8x4 run at 10240-token chunks (1280 tokens/chip, the same per-chip load as a 4x4
  stage at 5120). Without B2, a matmul-efficiency gain from larger per-chip tiles would be credited
  to the split.

| # | Goal | Success metric | Measured by |
|---|---|---|---|
| G1 | **Correct** | Per-layer PCC >= 0.98, KV-cache PCC >= 0.96, minimum PCC over layers 22-23 >= 0.99 (the existing split gate). Runner KV PCC >= 0.85 on the first 56320 tokens of the golden (the sc1 setting). No worse than B1. | `tests/kimi_k3/test_transformer_pipeline_split.py` (`LAYER_PCC`, `KV_CACHE_PCC`, `RECOVERY_PCC`, lines 51-55) at 4x4; runner PCC verdicts from the two-rank run |
| G2 | **Stable** | 3 back-to-back traced two-rank runs, no dispatch timeout and no hang. Each run: several chunks, a partial final chunk, slot reuse, every expected ACK received (K3 acks on its MLA layers only, 3 per chunk per rank, `prefill_runner.py:776-783`), clean shutdown. | runner logs |
| G3 | **Worth it** | Steady-state tokens/s **above both B1 and B2 by more than twice the run-to-run spread** (measure the spread over 3 runs of each). Time-to-first-token at 5k regresses by no more than an agreed bound (proposed: 25%; a short prompt now crosses two stages and a hop). Report TTFT at 5k, 56k and the longest supported sequence. | runner timing CSVs (`PREFILL_TIMING_DIR`) |
| G4 | **Explained** | Per-op time for one eager chunk of one 4x4 stage vs. B1 and B2: CCLs on the line axis, CCLs on the ring axis, matmuls, MoE at 56 experts/chip, D2D handoff. | `scripts/run_safe_pytest.sh --profile` on one eager chunk |

The G3 thresholds are proposals; adjust before Level 4's early go/no-go.

**Measurement contract** (applies to G3, G4 and all baselines):

- **TTFT** here means request arrival at the first stage to prefill completion (last KV write acked)
  at the last stage.
- **Throughput** is completed tokens over wall time across many chunks, measured end to end. The
  runner's per-chunk timing CSVs record compute only, start after an optional sync, and stop before
  the D2D send (`prefill_runner.py:421`), so they cannot be compared across one- and two-stage runs
  on their own. Also record D2D send time, per-stage idle time and per-chip memory peaks.
- Same instrumentation and settings for every configuration; 3 runs each, report mean and spread.
- Keep the timing artifacts. In CI, set `PREFILL_EXPECTED_TPS` and override `PREFILL_PERF_MARGIN`
  (default 0.15, `run_multirank_pcc.sh:38`) so the CI check matches G3.
- G1 validates correctness only on the first 56320 tokens (the K3 golden is not trusted past that);
  longer-context correctness stays unvalidated in this plan.

Why G3 can go either way: per chip, a 12-layer stage at 1280 tokens/chip does the same compute as a
24-layer 8x4 stage at 640 tokens/chip. Pipeline throughput is at best one chunk per slowest-stage
time, so the split wins only if 4-chip collectives save more than the line axis, the extra hop and
the bubble cost.

**Non-goals:** more than one galaxy, meshes spanning two hosts, production deployment.

## 2. Working practices

### Build

```bash
./build_metal.sh --release --build-tests   # Release is the default; Tracy is on by default
```

`--build-tests` is needed for the C++ fabric test binaries used in Level 6 (`test_tt_fabric`); tests
are off by default (`build_metal.sh:82`). Rebuild after pulling or after any C++ change (op, fabric,
nanobind). Python-only changes need no rebuild. Hang triage needs tt-exalens: `uv pip install -r tools/triage/requirements.txt`.

### Single-process tests: `scripts/run_safe_pytest.sh`

```bash
scripts/run_safe_pytest.sh <test_path>[::<test>] -rs [pytest args]
scripts/run_safe_pytest.sh --dev <test_path>       # watcher, NoC sanitizer, asserts, auto-triage
scripts/run_safe_pytest.sh --profile <test_path>   # Tracy per-op CSV
```

Exit codes: 0 pass, 1 test failure, 2 hang, 3 setup error. `--run-all` reports every test instead of
stopping at the first failure. Gotchas:

- **Skips count as PASS.** On Blackhole the conftest skips any test whose mesh is not all visible
  chips (`tests/conftest.py:331-334`), pytest returns 0, and the script prints PASS. Always pass
  `-rs`; a gate passes only with "N passed, 0 skipped" for the intended parametrization.
- **Opening a 4x4 in one process** needs `TT_VISIBLE_DEVICES` set to that half's 16 chips **and**
  `TT_MESH_GRAPH_DESC_PATH` set to a single-4x4 descriptor (as `perf/test_prefill_block_perf.py:20-27`
  does). Alternatively open the full 8x4 and carve 4x4 submeshes (Level 5).
- **5-second timeout.** The script hardcodes `DISPATCH_TIMEOUT=5` (line 52) and exports it over any
  value you set (line 186). It measures dispatch no-progress, not compile time, but legitimate long
  device waits (large first-chunk ops, CCL bring-up) can trip it, and the same value also caps fabric
  topology mapping (`control_plane.cpp:444-448`, otherwise 120 s). Level 0 makes it overridable
  (e.g. `DISPATCH_TIMEOUT=${SAFE_PYTEST_TIMEOUT:-5}`); when a run reports exit 2 during bring-up,
  rerun with a longer timeout before treating it as a hang.
- **Whole-galaxy lock and reset.** The lock is host-wide and a hang triggers a device reset of the
  whole galaxy, so the two halves cannot run tests (or weight conversion) in parallel under the script.
  The lock only coordinates cooperating users: multi-process `tt-run` launches do not take it, so do
  not start one while a `run_safe_pytest.sh` run is active.
- **`--profile` masks the test's exit code** (lines 29-31). Profile one eager chunk; a traced 12-layer
  chunk may overflow the profiler buffers.

### Hangs: use tt-triage output

- Under `run_safe_pytest.sh`, triage runs automatically when the dispatch timeout fires. Read the
  machine-readable report `generated/tt-triage/triage.csv` (kernel, go message, waypoint, PC,
  callstack per core) before guessing at causes. Rerun with `--dev` for watcher and waypoints.
- Multi-process runs (`tt-run`, `run_pipeline_prefill.sh`, `ci/run_multirank_pcc.sh`) have no
  dispatch timeout by default (`TT_METAL_OPERATION_TIMEOUT_SECONDS` defaults to 0,
  `tt_metal/llrt/rtoptions.hpp`). When a run stops making progress, run triage **per rank, while the
  ranks are still alive, before any reset**. Standalone tt-triage connects to rank 0's Inspector
  unless `TT_RUN_RANK` is set (`tools/triage/utils.py:17`):

  ```bash
  for r in 0 1; do
    TT_RUN_RANK=$r python3 tools/tt-triage.py --disable-progress --skip-version-check \
        --llm-output-path=generated/tt-triage/<run_name>_rank$r.csv
  done
  ```

  Attach the CSVs to the bug or PR.

## 3. Background: galaxy wiring as an 8x4 grid

Derived from the cluster descriptors in `tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole/`.
Tray comes from the bus id (`tt_metal/fabric/control_plane.cpp:124-148`), position from `asic_locations`.
Rev C layout shown; **rev A/B swaps trays 2 and 3** (top half = trays 1+3, bottom half = trays 2+4).

```
         col0   col1   col2   col3        each row: col3 -> col0 wrap (in-tray link)
row 0    T1L1   T2L1   T2L5   T1L5   ┐
row 1    T1L2   T2L2   T2L6   T1L6   │ stage 1 (top half)
row 2    T1L3   T2L3   T2L7   T1L7   │        ┐
row 3    T1L4   T2L4   T2L8   T1L8   ┘        │ subtorus links row 2 <-> row 5
row 4    T3L4   T4L4   T4L8   T3L8   ┐        │
row 5    T3L3   T4L3   T4L7   T3L7   │        ┘
row 6    T3L2   T4L2   T4L6   T3L6   │ stage 2 (bottom half)
row 7    T3L1   T4L1   T4L5   T3L5   ┘
         row 7 -> row 0 wrap inside the galaxy
```

- Each half is line x ring: the 4-wide in-tray axis wraps, the 4-row axis does not.
- Links between the halves: rows 3 <-> 4, plus the row 7 -> 0 wrap and the row 2 <-> 5 subtorus links
  on a subtorus galaxy. Standard galaxies have only rows 3 <-> 4; the split still works there.

## 4. Why top/bottom and not Middle/Edge

The Middle/Edge split (rows 2-5 ring x ring, rows 6,7,0,1 line x ring) is ruled out:

- **Not configurable.** `FabricConfig` replaces each mesh's MGD topology
  (`tt_metal/fabric/mesh_graph.cpp:457-476`): plain `FABRIC_2D` -> MESH for every mesh
  (`fabric_host_utils.cpp:61-78`); `TORUS_XY` throws for the LINE/RING mesh; `TORUS_X/Y` gives one
  ring per mesh, same as top/bottom. All ranks use one config (`control_plane.cpp:838-845`).
- **No throughput gain.** 24 layers split only 12 + 12 (multiple-of-12 rule), so the line x ring
  stage sets the pace either way.
- **Needs a subtorus galaxy.**

## 5. Plan, bottom-up

Each level depends only on the levels below it and has an exit gate. Do not start a level until the
one below passes. Single-process levels run under `scripts/run_safe_pytest.sh -rs`.

Fabric configurations to carry through the levels:

| Config | SP axis (0) | TP axis (1) | Status |
|---|---|---|---|
| `FABRIC_2D_TORUS_X` | line | ring | natural orientation of a half of the 8x4 |
| `FABRIC_2D_TORUS_Y` | ring | line | proven K3/DeepSeek path (`tt_moe.py:352-358`), needs the 4x4 transposed |
| `FABRIC_2D` | line | line | proven for 2 x [4,4] on one galaxy (MiniMax-M3); fallback and reference |

### Level 0: build, host, descriptors, device lists

1. `./build_metal.sh --release`; install tt-exalens.
2. Generate the cluster descriptor for `bh-glx-120-c02u02` and run the check script below. The host
   is not in the repo; its rack neighbours (e.g. `bh-glx-120-c02u14`) are in
   `SC36_32x4_revAB_subtorus_aisleC`, and the script run on `c02u14` reports rev A/B with both halves
   line x ring.
3. Generate the **`TT_VISIBLE_DEVICES` list for each half** (16 chips each) and the two-rank binding
   with a 4x4 variant of `models/demos/deepseek_v3_d_p/utils/gen_pipeline_binding.py` (today it probes
   `create_submeshes(MeshShape(8, 1))` columns). Do not hand-edit PCIe ids.
4. Write the descriptors:
   - **single 4x4, LINE/RING** for one-process tests (levels 2-4). The existing single-4x4 descriptors
     (`tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_single_4x4_subtorus_topology_mesh_graph_descriptor.textproto`,
     `models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_xy4_pinned_graph_descriptor.textproto`)
     are the RING/RING Middle carve, not a top/bottom half.
   - **two 4x4 meshes on one host** for levels 6-7, RING on the in-tray axis, inter-mesh channel count
     from the real crossing links (section 3). The existing
     `single_bh_galaxy_2x4x4_z_graph_descriptor.textproto` asks for 8 RELAXED channels and wrongly
     declares RING/RING; its comment ("trays 1+2 -> mesh 0 on Rev C") does not hold on rev A/B.
5. Decide how the `TORUS_Y` orientation is produced (mesh axis 0 on the in-tray ring).
6. Make `run_safe_pytest.sh`'s dispatch timeout overridable (section 2).

```python
import sys, yaml, networkx as nx
from networkx.algorithms import isomorphism as iso
TRAY = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}
d = yaml.safe_load(open(sys.argv[1]))
bus, loc = d["chip_to_bus_id"], d["asic_locations"]
pos = {c: (TRAY[(b if isinstance(b, int) else int(b, 16)) & 0xF0], loc[c]) for c, b in bus.items()}
G = nx.Graph((pos[a["chip"]], pos[b["chip"]]) for a, b in d["ethernet_connections"])
pairs = lambda a, b: sorted({p[1] for p, q in G.edges() if {p[0], q[0]} == {a, b}})
print("revC-style (1-2 full)" if len(pairs(1, 2)) == 8 else "revAB-style (1-3 full)",
      {k: pairs(*k) for k in [(1, 2), (1, 3), (2, 4), (3, 4)]})
torus = nx.cartesian_product(nx.cycle_graph(4), nx.cycle_graph(4))
line_ring = nx.cartesian_product(nx.path_graph(4), nx.cycle_graph(4))
top = {1, 2} if len(pairs(1, 2)) == 8 else {1, 3}
for name, f in {"top": lambda p: p[0] in top, "bottom": lambda p: p[0] not in top}.items():
    H = G.subgraph(p for p in G if f(p))
    print(name, "torus:", iso.GraphMatcher(H, torus).subgraph_is_monomorphic(),
          "line x ring:", iso.GraphMatcher(H, line_ring).subgraph_is_monomorphic())
```

Gate: build passes; board revision known; both halves line x ring; device lists, binding and
descriptors exist.

### Level 1: smoke test on one 4x4

`tests/test_prefill_block_loop.py -k torus-x-4x4` (DeepSeek block; it uses the
`FABRIC_2D_PREFILL_BLOCK_MESH_PARAMS` 4x4 params, `tests/conftest.py:84-106`) with the half's
`TT_VISIBLE_DEVICES` and the single-4x4 descriptor. This proves a 16-chip process opens, runs a full
prefill block with ring-on-one-axis CCLs, and that the skip trap is handled.

Gate: the 4x4 case runs (not skipped) and passes, for one half and then the other.

### Level 2: ops on one 4x4

1. **16-chip crash fix**: `_determine_device_name` (`models/demos/deepseek_v3_d_p/tt/tt_ccl.py:610-641`)
   has no Blackhole entry for 16 devices; reached via `get_num_links` (e.g. the KDA output
   reduce-scatter, `kda.py:421`). Add one with 2 links per axis.
2. **Add 4x4 cases** (none of these has one today: `test_ttnn_dispatch_combine.py` uses
   `ALL_MESH_CONFIGS` up to 8x1/4x2/2x4; `test_reduce`, `test_offset_cumsum`,
   `test_zero_padded_kv_cache`, `pcc/test_ttnn_moe` go up to 8x4; `test_mla.py` is 8x4 and Mistral-only).
   Prioritize what K3 changes at 4x4:
   - MoE dispatch/combine and offset_cumsum at **56 experts per chip**
     (`init_helpers.extract_mesh_config:44`);
   - ring joint SDPA with the SP axis as a line (Linear support: `ring_joint_sdpa_program_factory.cpp:455`);
   - KDA output reduce-scatter (the crash path).
3. `per_axis_topology` (`tt_ccl.py:543-578`) already maps `TORUS_X/Y` correctly; verify each CCL
   receives its axis' topology.

Gate: the new 4x4 cases pass (not skipped) under `TORUS_X`, then `TORUS_Y`.

### Level 3: single layers on one 4x4

1. Add 4x4 placements to `tests/kimi_k3/test_transformer_depth.py:100`.
2. The K3 depth and split tests hardcode `SEQ_LEN = 5120` (`test_transformer_depth.py:62`,
   `test_transformer_pipeline_split.py:46`). Parametrize it so they can run at 2560 (640 tokens/chip,
   where MLA/KDA configs are tuned; 2560 % (32*4) == 0).
3. One MLA layer, one KDA layer, one MoE block: PCC against the reference at 2560.

Gate: single-layer PCC passes for each layer type.

### Level 4: one 12-layer stage on one 4x4, then early go/no-go

1. **Memory budget** per chip for 12 layers on 16 chips: weights, MLA KV (per-layer per-device
   sequence storage doubles with SP 8 -> 4, layer count halves), KDA state (`adapters/kimi_k3.py:132`),
   AttnRes snapshots, D2D backing tensors, trace buffers. Expected: DRAM about the same as 24 layers
   on 32 chips; L1 and activations change.
2. **Test weight cache**: the K3 tests cache under `kimi_k3_bh_16dev/<ckpt>/mesh4x4.tpaxis1`
   (`kimi_k3/weights.py:236-242`). Convert layers 0-11 (one half at a time; see the lock gotcha).
   Before converting all 12 layers, convert **one** layer and prove it loads through both the test
   path and the runner path (Level 7.1); the two use different cache layouts.
3. 12-layer depth test at 2560, eager then traced. `test_transformer_depth.py` runs an eager forward
   only (line 274); add a traced variant.
4. 5120-token chunks (1280 tokens/chip): sweep MLA matmul/SDPA configs (`mla_config.py`;
   `_resolve_mm_cfg` falls back silently) and KDA projections (`kda/config.py:71`); check L1.
5. **Early go/no-go.** Time one traced 12-layer 4x4 chunk at 5120 against one traced 24-layer 8x4
   chunk at 5120 (B1) and at 10240 (B2), per chip-token. Pipeline throughput cannot beat one chunk
   per slowest-stage time, so if the 4x4 stage is not clearly faster per chunk, stop here and report
   (G4 profile). Levels 5-7 are only worth building if this passes.

Gate: 12-layer PCC passes traced at 2560 and 5120; early go/no-go passes.

### Level 5: the stage boundary, in one process, with real stage memory

`tests/kimi_k3/test_transformer_pipeline_split.py` builds both halves on **one** mesh
(`test_transformer_pipeline_split.py:119-120`). At 4x4 that puts all 24 layers on 16 chips: about
24 GB of routed experts (56 bfloat4_b experts/chip x 23 MoE layers) plus about 3 GB of attention and
KDA weights against 32 GB of DRAM per chip, twice a real stage's weights. Instead:

- open the 8x4 in one process under `FABRIC_2D_TORUS_X` and carve two 4x4 submeshes, top and bottom
  (`create_submesh`, as `tests/test_d2d_socket_sync.py:93-94` does), one half of the layers on each.
  This gives real per-stage memory and exercises the two-plane handoff at layer 12, including the
  sealed AttnRes snapshot (`adapters/kimi_k3.py:231`).

Gate: `LAYER_PCC` 0.98, `KV_CACHE_PCC` 0.96, `RECOVERY_PCC` 0.99 with the 4x4 submesh placement
(G1 part one).

### Level 6: fabric with two processes

Reuse existing coverage (MiniMax-M3 runs 2 x [4,4] on one galaxy,
`minimax_m3/docs/PIPELINE_PREFILL_TESTING.md:165-240`):

1. Fabric traffic test with `tools/scaleout/exabox/run_fabric_tests.sh --hosts <this host> --image none`.
   **`--config 2x4x4z` as-is tests the wrong split**: it resolves devices from 2x4 slices and builds
   the interleaved inner/outer pair (slices {1,2} / {0,3}, i.e. Middle/Edge,
   `run_fabric_tests.sh:1040-1045`, `tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py:166-176`),
   and its default suite includes TorusXY cases (`test_fabric_multi_mesh_sanity_common.yaml:126`).
   Supply the top/bottom device lists from Level 0, the new two-mesh descriptor, and a
   `TORUS_X`/`TORUS_Y`/`FABRIC_2D`-only test YAML.
2. Open both meshes at once under each config in the table above. Assert per axis that
   `ttnn.get_usable_topology(tensor, Ring, axis)` returns Ring only where expected. A fabric config
   that does not torus a declared RING axis only logs (`control_plane.cpp:2184-2195`).
3. all_gather / reduce_scatter on both axes, PCC; D2D transfer between the meshes repeated while
   CCLs run, with lease release/reclaim and shutdown.

Gate: expected per-axis topology on both meshes; CCLs and D2D pass; no hangs.

### Level 7: two-rank K3 run on one galaxy

1. **Runner weight cache.** The runner reads a different layout from the tests:
   `{name}_bh_{ttnn.get_num_devices()}dev/{sp}x{tp}` (`adapters/mla.py:83-85`, via
   `prefill_runner.py:667`), so Level 4's cache does not feed it. Convert explicitly before the first
   launch; inline conversion of 12 layers takes over an hour (`weights.py:191`) and can exceed the CI
   script's `TABLE_WAIT_SECS=3600` (`run_multirank_pcc.sh:298`).
2. **By hand first.** Copy `topology_configuration/pipeline_prefill_request_intragalaxy_2rank.yaml`
   for K3 (generated device lists, `PREFILL_SP=4`, `PREFILL_TP=4`, 24 layers as 12 + 12,
   `adapters/kimi_k3.py:174`; same SP/TP for the producer, `prefill_producer.py:98`) and launch with
   `run_pipeline_prefill.sh`. **Set `PREFILL_FABRIC_MODE` explicitly**: the runner defaults to
   `2d_torus_xy` (`prefill_runner.py:567`), which throws on a LINE/RING mesh. Start with `"2d"`
   (proven), then the torus configs. Include the two-ranks-on-one-host setup steps from the MiniMax
   doc (PRTE slots, `ulimit -u`).
3. **Then CI.** `ci/run_multirank_pcc.sh` runs kimi_k3 as a single rank; its pipeline path is
   Mistral4-specific (binding name and 8x1 cache checks, ~lines 235-255) and `CHUNK_SIZE=5120` is fixed
   (line 31). Add a K3 two-rank intragalaxy mode with configurable chunk size and generated bindings.
4. **Gates, in order**: eager then traced; several chunks, a partial final chunk, slot reuse; every
   expected ACK; MLA/KDA migration-table readback with local mock migration; runner KV PCC >= 0.85.

Gate: G1 and G2.

### Level 8: measure

B1, B2, and the two-stage run under each fabric config, 3 runs each, `PREFILL_SYNC_PER_CHUNK` on and
off. Record tokens/s and TTFT (G3) and the per-op profile (G4).

Gate: G3 and G4 reported; go/no-go decision.

## 6. Risks

1. L1 overflow or hangs at 1280 tokens per chip (levels 3-4 start at 2560-token chunks).
2. 56 experts per chip: buffer sizing in the fused MoE path (level 2).
3. Concurrent-CCL workarounds proven only on 8x4 (shared-expert/dispatch EDM deadlock workaround
   `tt_moe.py:351-370`, `attn_res_gather_softmax` hang #53318). Capture tt-triage per rank for any hang.
4. `TORUS_Y` on a 4x4 needs a transposed placement; if that is awkward, use `TORUS_X` and `FABRIC_2D`.
5. G3 may fail; the Level 4 early go/no-go is there to find out before the two-rank work.
6. False passes from skipped tests (section 2); every gate requires 0 skipped.

7. MoE accuracy margin: measured at 4x4 (seeded weights, 640 tokens/chip) the MoE block scores
   0.9695 (TORUS_X) / 0.9691 (TORUS_Y) against the 0.965 bar, the same as the 8x4 calibration
   (0.9696). The earlier expectation that smaller meshes score higher (2x4 0.9957) did not hold.
8. Flaky ethernet link training on the target galaxy (section 9): a 2-channel pair that comes up with
   one channel drops the whole mesh to one routing plane, and 2-link CCLs TT_FATAL.

## 7. Out of scope

- More than one galaxy, and the decode/migration contract for SP=4.
- Meshes spanning two hosts.
- Mixed per-mesh fabric topology (Middle/Edge split); needs a fabric change.

## 8. Review log

- **v1** reviewed by a separate Claude instance and Codex (gpt-6-astra, high effort). Both found that
  the Middle/Edge split cannot be configured (FabricConfig overrides MGD topology per mesh).
- **v2**: chose top/bottom; rev A/B handling; corrected silent-downgrade reasoning; reuse existing
  intragalaxy coverage; generated bindings; real inter-mesh channel count; memory budget, cache keying,
  K3 pipeline gates, CI gaps, 12-layer depth test.
- **v3**: scoped to one galaxy running two stages.
- **v4**: goals and success metrics; build, test and hang-triage practices; bottom-up order.
- **v5** (second Claude review): skip-as-pass trap and `TT_VISIBLE_DEVICES` for one-process 4x4;
  Level 2 is new 4x4 test cases, not reuse; Level 1 smoke via `test_prefill_block_loop`; Level 5 on two
  4x4 submeshes instead of 24 layers on one 4x4; separate runner cache conversion; explicit
  `PREFILL_FABRIC_MODE` and `FABRIC_2D` as a third config; G3 needs a margin and a 10240-chunk 8x4
  baseline; early stage-timing go/no-go after Level 4; two-process fabric moved next to the two-rank
  run; per-rank tt-triage; corrected timeout attribution, ACK count, recovery-gate wording;
  run_safe_pytest gotchas (5 s timeout also caps topology mapping, whole-galaxy lock and reset,
  `--profile` masks exit code).
- **v5.1** (second Codex review, written against v4; its overlapping findings were already fixed in
  v5): `--build-tests` for fabric binaries; `run_fabric_tests.sh --config 2x4x4z` tests the
  inner/outer (Middle/Edge) split and includes TorusXY cases, so supply top/bottom bindings and an
  X/Y-only YAML; overridable safe-runner timeout and the lock only coordinating cooperating users;
  measurement contract (TTFT definition, end-to-end throughput, D2D and idle time, memory peaks, CI
  perf margin); prove one cached layer loads through both cache paths before converting all;
  traced variant of the depth test; correctness validated only to 56320 tokens.

## 9. Execution log

Target host `bh-glx-120-c02u02` (single BH galaxy, runs inside a container). Branch
`pjosipovic/k3-prefill-4x4-single-galaxy`.

### Level 0 (done)

- Cluster descriptor dumped with UMD's `topology` tool: **rev A/B subtorus**. Halves are trays 1+3
  (top) and 2+4 (bottom); both line x ring. 12 link pairs / 28 channels between the halves.
- `TT_VISIBLE_DEVICES` integers are logical ids in PCIe-BDF order (the same numbering as UMD chip
  ids), not `/dev/tenstorrent` indices, and their order is ignored (UMD keeps a set). On this host the
  two differ for ids 16-31.
- `gen_pipeline_binding.py --stage-shape 4x4` (new) writes the 2-rank binding:
  top `0,4,28,24,1,5,29,25,2,6,30,26,3,7,31,27`, bottom `11,15,23,19,10,14,22,18,9,13,21,17,8,12,20,16`.
- Single-4x4 descriptors: the existing unpinned `experimental_descriptors/single_bh_galaxy_subtorus_x4`
  ([LINE, RING]) and `_y4` ([RING, LINE]) work for either half. Two-mesh descriptors added:
  `single_bh_galaxy_2x4x4_z_torus_{x,y}_graph_descriptor.textproto`.
- `TORUS_Y` orientation needs no extra work: with the y4 descriptor the topology mapper places the
  in-tray ring on mesh axis 0 by itself.
- `run_safe_pytest.sh` timeout is now overridable (`SAFE_PYTEST_TIMEOUT`); 16-chip entry added to
  `tt_ccl._determine_device_name`.
- Weight cache: the shared Weka cache is read-only here, so the 4x4 cache lives under `/home`
  (~14 GB per MoE layer, ~330 GB for 24 layers). The runner layout is bridged with a symlink
  `kimi_k3_bh_16dev/4x4 -> <ckpt>/mesh4x4.tpaxis1`, as the 8x4 cache already does; no second copy.

### Link reliability (affects every level)

- After a reset, 1-4 ethernet channels randomly fail to train. If one of the 2-channel pairs a mesh
  uses comes up with a single channel, the fabric drops that mesh to one routing plane and CCLs that
  request 2 links fail (`high_bw_all_gather requested 2 links, but only 1 usable links`).
- Pairs can also come up with **zero** channels. The first version of the check only looked at pairs
  that were present, so it missed those; it now compares against the bring-up descriptor.
- It got worse over the session: by 16:50, 16 consecutive `tt-smi -r` resets never produced a fully
  clean galaxy. The pairs that fail are mostly the ones crossing between the halves (tray-pair links at
  L1/L5, the row 7 -> row 0 wraps), so the 8x4 TorusXY baseline cannot map ("no embeddable MGD
  fallback") while each 4x4 half stays clean.
- `tt-smi -glx_reset` cannot run inside this container (no IPMI device); it has to be run on the host.
  `recover.sh` is reserved for the operator helper.
- Workaround: before every run, reset with `tt-smi -r` until the pairs the run needs are intact
  (`halves`: every pair inside both 4x4 halves; `x8x4`: plus the row 3 <-> 4 crossing pairs; `all`:
  every pair, needed by the 8x4 torus baseline).

### Level 1 (done)

- `test_prefill_block_loop.py` `torus-{x,y}-4x4`, `no_ref-layer3-gate_device_fp32`: **1 passed,
  0 skipped** on the top and the bottom half, under both TORUS_X (Y=LINE X=RING) and TORUS_Y
  (Y=RING X=LINE).
- The skip trap is real: the `with_ref` 4x4 cases are skipped (host gate forced at 4x4; reference
  path skipped above 1024 tokens) and `run_safe_pytest.sh` printed PASS for "1 skipped".

### Level 2 (MoE done)

- `pcc/test_ttnn_moe.py::test_kimi_k3_moe` new 4x4 cases, `kimi_k3-5k-pcc` (56 experts/chip,
  640 tokens/chip), top half: **passed** under TORUS_X (final 0.9695, latent routed 0.9700, shared
  0.9996) and TORUS_Y (final 0.9691). Bar 0.965.

### Level 3 (done)

- `test_transformer_depth.py` L1, L2, L5 at `torus-x-4x4`, `KIMI_K3_TEST_SEQ_LEN=2560`, top half:
  **3 passed, 0 skipped**. Per-layer PCC 0.99987 / 0.99808 / 0.99841 / 0.99693 / 0.99681 (bar 0.99);
  KV layer 3 lora 0.9986, rope 0.9996 (bar 0.96).

### Level 4 (12-layer stage)

- `test_transformer_depth.py` L12 at `torus-x-4x4`, top half: **passed** at 2560-token chunks (eager)
  and at 5120 (1280 tokens/chip, eager, untuned fallback configs, no L1/CB failures). Per-layer PCC
  >= 0.9967 (bar 0.98); KV layers 3/7/11 lora >= 0.9986, rope >= 0.9996 (bar 0.96).
- Cache for layers 0-11 at 4x4: 184 GB.
- **The runner needs the AttnRes queries in the cache** (it builds every weight from the cache; the depth
  test reads AttnRes from the checkpoint and never caches it). `test_build_stage_cache.py` writes them.
- **The runner cannot run 2560-token chunks**: `create_kv_chunk_address_table_block_cyclic` assumes a
  block-cyclic period of `PREFILL_CHUNK_TOKENS = 5120`. Runner-level 4x4 stages therefore run at
  1280 tokens per chip; the 2560 start applies to the tests only.
- Traced 12-layer stage through the real runner (one rank, top half, TORUS_X, 5120 chunks, 11 chunks of
  the 56,320-token golden): **KV PCC 0.994** (bar 0.85), runner and producer exit 0. Per-chunk compute
  196-268 ms (grows with context).
- Cache for layers 12-23 built with `test_build_stage_cache.py` on the bottom half (44 min). Full 4x4 cache:
  24 layers + AttnRes, 381 GB.

### Level 5 (done)

- `test_pipeline_split_on_stage_submeshes[torus-x-8x4-as-2x4x4]` at 5120: **1 passed, 0 skipped**. Layers 0-11
  on the top 4x4 submesh, 12-23 on the bottom one, boundary tensor handed across meshes. Worst layer 19
  0.98614 (bar 0.98), layers 22-23 min 0.99562 (bar 0.99), the same curve as the undivided L24 run
  (0.9864 at layer 19, 0.998 by 23); KV layers 3-23 lora >= 0.9974, rope >= 0.9992 (bar 0.96).
- Two silent skips had to be fixed first: the conftest's allowed-fabric table had no `TORUS_X` for 8x4 and
  no `FABRIC_2D` for 4x4 on a Blackhole galaxy.

### Level 6 (done)

- `two_stage_fabric_check.py` (new; two ranks via `tt-run`, one 4x4 half each): **both ranks pass** under
  `2d_torus_x` (Y=LINE X=RING), `2d_torus_y` (Y=RING X=LINE) and `2d` (LINE/LINE): realized per-axis ring
  matches what the model picks, all_gather on both axes matches torch, 10 socket transfers between the
  meshes interleaved with CCLs arrive bit-exact.

### Level 7 (two-rank K3 run on one galaxy)

All runs: 24 layers as 12 + 12, 5120-token chunks, traced, per-layer ACKs, mock migration, KV PCC
against the first 56,320 tokens of the golden (bar 0.85). Launch notes:

- The migration table must be on storage the runner does not classify as per-host (`/tmp` is rejected
  for 2 ranks even on one host).
- Reset (`tt-smi -r`) before every launch: after a killed run, bring-up fails with "Timed out while waiting
  for active ethernet core".
- The runner requires `max_seq_len % chunk_size == 0`; a partial final chunk comes from a request shorter
  than `max_seq_len`. K3's KDA requires 32-token-aligned request ends (`validate_kda_bounds`); the
  producer's mid-chunk end is now alignable (`PREFILL_PRODUCER_END_ALIGN`, default 1).

Results:

| Fabric mode | Outcome | KV PCC rank0 / rank1 |
|---|---|---|
| `2d` (LINE/LINE) | 3 of 3 runs pass | 0.99381 / 0.99668 (identical each run) |
| `2d` + workload (3 requests reusing the slot, 32-aligned mid-chunk ends) | pass, 35 chunks per rank | 0.99464 / 0.99674 |
| `2d_torus_y` (SP ring) | 3 of 3 runs pass | 0.99392 / 0.99669 |
| `2d_torus_x` (TP ring) | 2 of 5 pass; 3 hangs (see below) | 0.99417 / 0.99670 |
| `2d_torus_x`, D2D limited to 1 lane (now the runner default for this mode) | 12 of 12 valid runs pass (4 more were bring-up link flakes) | (per-run KV PASS) |

`2d_torus_x` hangs (triage per rank in `generated/tt-triage/`):
- try 1: rank 0 finished chunk 0 and sent it, but 4 of its 16 D2D senders never wrote; rank 1 waited in
  `InboundSocketServiceSync` on those 4 coordinates. Two ethernet ports on one of the stalled chips had
  retrain count 6.
- try 2: in rank 0's first chunk, rows 2-3 of mesh 0 (the chips that carry the inter-mesh links to the
  other half) stuck in a TP-ring `ReduceScatterMinimalAsync` while rows 0-1 moved on; all ports up, no
  retrains. Consistent with the documented hazard of inter-mesh links next to a fabric-config torus axis
  (`MGD_README.md`, #54650).

### Level 8 (measurements)

Baselines: the whole galaxy as one 8x4 stage, 24 layers, same runner, chunk size and instrumentation:
8x4 `FABRIC_2D` (LINE/LINE) and, once the wrap links trained, the production 8x4 TorusXY. Steady-state
throughput is measured from the last rank's chunk-start cadence over the 11 measured chunks.

| Configuration | Steady-state tok/s | Mean compute per chunk | One chunk through all stages |
|---|---|---|---|
| 8x4, one stage, `2d` (B1) | 17,472 (sync on) / 17,540 (sync off) | 292 ms | 298 ms |
| 2 x 4x4, `2d` | 16,697-16,976 (sync on, 3 runs) / 16,934 (sync off) | 286 / 255 ms | 478 ms |
| 2 x 4x4, `2d_torus_y` | 17,058-17,100 (3 runs) | 281 / 250 ms | 465 ms |
| 2 x 4x4, `2d_torus_x` (2 D2D lanes; hangs) | 17,874-17,940 | 274 / 243 ms | 443 ms |
| 2 x 4x4, `2d_torus_x`, 1 D2D lane | 16,959-17,553 (12 runs) | | |
| **8x4, one stage, TorusXY (production)** | **19,413** | 262 ms | 267 ms |

- Per chip, a 4x4 stage (12 layers, 1280 tokens/chip) takes about as long as the 8x4 stage (24 layers,
  640 tokens/chip), as the compute arithmetic predicts. The split cannot win on compute; only the
  collectives differ.
- The pipeline is paced by rank 0 (embedding, dense layer 0 and H2D input on top of its 12 layers).
- B2 (8x4 at 10240-token chunks) cannot run through the runner: the KV migration table fixes the
  block-cyclic period at 5120 tokens.

### Verdict against the goals

- **G1 Correct: met.** Split gate (Level 5) and runner KV PCC (Level 7) pass in every configuration.
- **G2 Stable: met** for `2d`, `2d_torus_y` and `2d_torus_x`. `2d_torus_x` hangs intermittently with two D2D lanes; with one lane (the runner default for that mode now) 12 of 12 valid runs were clean.
- **G3 Worth it: not met.** Against the production 8x4 TorusXY stage (19,413 tok/s), two 4x4 stages
  reach 17,058-17,100 tok/s with `2d_torus_y` (-12%), 17,874-17,940 with `2d_torus_x` (-8%, but it
  hangs with two D2D lanes) and about 17,230 with `2d_torus_x` on one D2D lane (-11%). Per-chunk time
  through the pipeline rises about 65-75% (two stages in series); throughput is the goal, so that is
  expected, but throughput does not improve.
- **G4 Explained: met.** Per-op profile below: per chunk, compute per chip is equal (a 12-layer 4x4 stage
  and the 24-layer 8x4 stage both spend about 120-125 ms in matmuls, experts, SDPA and KDA), and the
  4-chip collectives and MoE dispatch are only 1.1-1.7x cheaper per layer than on 8 chips, which does not
  pay for the extra stage.

### The `2d_torus_x` hang (investigation)

Triage per rank on every hang (`generated/tt-triage/`):

- Primary failure: the stage-to-stage D2D transfer. On rank 0, a subset of the 16 `persistent_d2d_sender`
  cores, always on the boundary rows 2-3 of mesh 0 (the chips with the inter-mesh links), never write
  (`wr_off=0`). Their second lane (ncrisc, link index 1) sits in `fabric_write_bytes` ->
  `WorkerToFabricEdmSenderBase::wait_for_empty_write_slot`: the local fabric router never frees a slot.
  Rank 1 waits in `InboundSocketServiceSync` on exactly those coordinates.
- Secondary: when the stall happens before rank 0's capture forward, the TP-ring
  `reduce_scatter_minimal_async` on the same boundary rows wedges behind it (the first hangs seen).
- Only under FABRIC_2D_TORUS_X; never under TORUS_Y or FABRIC_2D. Consistent with the inter-mesh-link /
  config-torus hazard in `MGD_README.md` (#54650).
- A no-model reproducer (`tests/kimi_k3/two_stage_torus_x_repro.py`) reproduces the signature, but
  rarely (1 in 12 runs).
- A startup fence (`PREFILL_WARMUP_BARRIER=1`, now opt-in) did not fix it. One D2D lane (new
  `max_sender_lanes` binding argument; runner default under `2d_torus_x`, override with
  `PREFILL_D2D_MAX_LANES`) ran 12 of 12 valid runs clean. At the observed two-lane hang rate (1 in 5 to
  3 in 5 runs) that is 0.2-7% likely by chance. It costs the throughput `2d_torus_x` had over
  `2d_torus_y`.
- Recommendation: run two stages per galaxy with `2d_torus_y`; report the two-lane D2D stall under
  TORUS_X to the fabric owners with the reproducer and triage.

### G4: per-op profile

`scripts/run_safe_pytest.sh --profile` on `test_transformer_depth.py` L12 (one eager 12-layer chunk,
5120 tokens), same 12 layers in each column. Device kernel time per device (mean over devices), by op
category. Eager profiles inflate collective and dispatch time with cross-device wait, so compare the
compute rows directly and the communication rows only as ratios.

| Category (ms per device) | 4x4 `2d` | 4x4 `TORUS_Y` | 8x4 `2d` | 4x4 / 8x4 |
|---|---|---|---|---|
| Matmul | 39.1 | 39.1 | 18.0 | 2.2x |
| MoE experts and gate | 61.4 | 61.4 | 32.0 | 1.9x |
| Attention (SDPA) | 8.2 | 8.2 | 4.5 | 1.8x |
| KDA | 11.4 | 11.4 | 8.2 | 1.4x |
| MoE dispatch/combine | 82.5 | 71.3 | 64.7 | 1.1-1.3x |
| Collectives | 55.3 | 71.2 | 41.7 | 1.3-1.7x |
| Other | 19.0 | 19.0 | 13.6 | 1.4x |
| Total | 276.8 | 281.5 | 182.8 | |

- Compute rows scale about 2x with 2x tokens per chip: per chunk, a 12-layer 4x4 stage (about 120 ms)
  and the 24-layer 8x4 (2 x 63 = about 125 ms) do the same compute per chip.
- Communication rows grow less than 2x (smaller groups), which is the only place the split can gain; in
  the traced runner it does not outweigh the stage hop (section above).
- CSVs: `generated/profiler/reports/2026_10_10_01_20_38`, `..._01_25_47`, `..._01_35_32`.
