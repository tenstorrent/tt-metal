// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  CollapsibleSection,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Link,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  computeDAGLayout,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";
import type { TableRowTone } from "cursor/canvas";

// ---------------------------------------------------------------------------
// The build, step by step
// ---------------------------------------------------------------------------

type Step = {
  id: string;
  title: string;
  where: string[];
  does: string[];
  imposes: string[];
  tierC: string;
};

const DESIGN_DOC_URL =
  "https://github.com/tenstorrent/tt-metal/blob/rsong/mgd-pipeline-tier-c/docs/scratch/mgd-pipeline-graph-design.md";

function docAnchor(heading: string): string {
  return heading
    .toLowerCase()
    .replace(/[^\w\- ]/g, "")
    .replace(/ /g, "-");
}

type MapperIntegration = { does: string[]; refs: string[] };

const MAPPER: Record<string, MapperIntegration> = {
  launch: {
    does: [
      "The topology mapper runs inside control-plane init, which this step triggers. When the MGD has `pipeline_descriptors`, it checks the pipeline for each candidate seating before the mapping commits, in four steps: build the position graph (C1), solve one carve and check it with the resolver (C2), publish the crossing channels `R` (C3), or reject the seating and draw the next one (C4).",
      "On success, `generate_rank_bindings` writes each rank's stage as `TT_PIPELINE_STAGE`. Nothing reads it yet: it is a witness that a placement exists.",
      "Without a descriptor every hook is off and the mapping is today's.",
    ],
    refs: [
      "2.3 Tier C - inside the topology mapper",
      "4. Pipeline mapping in the topology mapper (Tier C)",
      "4.6 Writing the stage plan into rank bindings (written, not used)",
      "4.7 Without a pipeline descriptor: every hook off",
    ],
  },
  entry: {
    does: [
      "The mapper sees the pipeline only as `pipeline_descriptors` in the MGD: stages with shapes, edges, and `loopback: true` on each edge that closes a cycle. Blaze does not read it; this entry builds its own graph.",
      "Steps 1a and 1b cover how the mapper sees the reload ring and the DFlash fork.",
    ],
    refs: ["1.2 The pipeline descriptor", "A fork example", "1.4 Edge policy: STRICT or RELAXED, per edge - RELAXED (later)"],
  },
  reload: {
    does: [
      "A reload ring is written as an ordinary ring descriptor: 4 stages of 4x2 and one loopback edge. The mapper checks it like any ring: positions on the one mesh, one carve, the resolver, then `R`.",
      "The design has no way yet to require two or more channels on each hop, or a ring along a stage's rows. Those are G2 and G4.",
    ],
    refs: ["1.2 The pipeline descriptor", "4.1 Step C1 - build the physical pipeline graph", "4.2 Step C2 - pipeline solve"],
  },
  dflash: {
    does: [
      "The fork is written as one router stage and two chains that return to it. The resolver already handles fork graphs: edges form a DAG once loopbacks are dropped, and host edges are left out.",
      "In the first version both returns must be STRICT, so the drafter's host return cannot be spelled. RELAXED edges (later) make it a soft target edge, solved with one resolver call per fabric chain.",
    ],
    refs: ["A fork example", "3.1 Reuse the C++ graph-layout resolver as-is", "1.4 Edge policy: STRICT or RELAXED, per edge - RELAXED (later)"],
  },
  harness: {
    does: [
      "Tier A checks the descriptor on its own when the MGD is parsed: stage ids, edges, loopback marks, shapes.",
      "Tier B checks it against the declared meshes and connections: which meshes each stage can bind to, whether the slices have the capacity, and whether a split onto meshes exists whose crossings ride declared connections.",
      "The design treats the harness loop and the descriptor as two spellings of one graph. The builder has no path that loads stages from MGD yet.",
    ],
    refs: ["How blaze builds the same graph", "2.1 Tier A - descriptor only", "2.2 Tier B - descriptor plus the declared fabric"],
  },
  pad: {
    does: [
      "The descriptor counts stages; blaze sizes the ring in ranks. A joint stage is written as one shape that spans several host slices, and its ranks co-own it.",
      "Slices that no stage takes become fillers and their ranks idle. Tier B's capacity check compares the stages' area with the slices available.",
    ],
    refs: ["A joint-stage example", "Capacity", "5. Stage-to-rank mapping"],
  },
  graph: {
    does: [
      "Tier C's target graph is the descriptor's stage graph, merged across MGD parts into `pipeline_stage_graph_`, with each stage's shape and loopback marks. Several pipelines are one target graph with one component each.",
      "The loopback marks are copied into the resolver's edges, so stage order and host I/O placement come out the same way blaze computes them.",
    ],
    refs: ["The two graphs", "1.3 Schema", "1.5 Merging MGDs"],
  },
  path: {
    does: [
      "C1 replaces blaze's three carve paths with one position graph: every rectangle of host slices that some stage shape fills, on every mesh, and which rectangles touch.",
      "That one graph covers the planned carve, per-rank tiling and single-mesh tiling. It does not cover a stage smaller than a host slice (open question 15).",
    ],
    refs: ["4.1 Step C1 - build the physical pipeline graph", "The position graph", "3.4 How blaze turns meshes into submeshes"],
  },
  resolve: {
    does: [
      "C2 runs one footprint solve on the position graph, which picks the rectangles, the stage order and the crossings together. It then checks that one carve with the resolver. Any failure fails the seating.",
      "On failure, C4 first retries the same seating with exit-node constraints that move cable ends onto other slices, then excludes the seating and draws the next one, up to a rejection cap.",
    ],
    refs: ["4.2 Step C2 - pipeline solve", "4.4 Step C4 - on failure: constraints for the next solve", "4.5 The whole loop on two examples"],
  },
  carve: {
    does: [
      "Blaze's carve planner is not ported. The design keeps it as the runtime reference and lists where it and the footprint solve disagree (open question 8).",
      "C1 links slices the way `discover_slice_adjacency` does: inside a mesh, the `device_topology` neighbours with RING wraps included; between meshes, the PSD channels the control plane could bind for each MGD connection.",
    ],
    refs: ["3.2 Blaze's carve planner: the runtime reference, not ported", "Links", "3.3 Inter-mesh cables: read from the PSD (policy plan later)"],
  },
  cpp: {
    does: [
      "C2 calls the same search blaze reaches here, `resolve_pipeline_placement`, with the chosen positions' chips and the links C1 found instead of asking the control plane. The mapper and runtime apply one set of layout rules.",
      "The resolver picks each crossing's two chips. C2 records that exact channel in `R` and checks that `R` fits each connection's channel count.",
    ],
    refs: ["3.1 Reuse the C++ graph-layout resolver as-is", "What the resolver checks", "Which cable each crossing uses (IMPORTANT)"],
  },
  layout: {
    does: [
      "C3 emits the mapping: each rank's slice, then its slot, then its stage. That stage-to-rank map is a witness; blaze carves again here and may land elsewhere.",
      "`R` is broadcast, and the control plane binds those channels first (phase 0 of the pairing), so the cables the accepted layout crosses on are the cables that carry traffic.",
    ],
    refs: ["4.3 Step C3 - on success: emit the mapping and publish the cables", "New: phase 0 in the pairing, bind `R` first", "5.2 In this design"],
  },
  run: {
    does: [
      "This is Tier D, blaze's own build on the committed mapping, which the design leaves unchanged. Tiers A to C move its failures to mapping time, where the mapper can still try another seating.",
      "Socket-level needs, such as two or more channels per hop (G2), are not in the mapper's model yet.",
    ],
    refs: ["2.4 Tier D - inside the pipeline builder", "Crossings must ride connections the control plane actually binds"],
  },
};

const STEPS: Step[] = [
  {
    id: "launch",
    title: "0. Launch and open the mesh",
    where: [
      "`blaze/models/cli.py` `main`, `run_demo`, `bring_up`, `open_mesh_device`, `get_worker_l1_size`",
      "`blaze/models/pipeline_graph_helpers.py` `local_mesh_shape`",
    ],
    does: [
      "`tt-run` starts one process per host slice with `TT_MESH_GRAPH_DESC_PATH` and rank bindings (generated by the automapper; no Kimi rank-binding YAML is checked in).",
      "`main` calls `ttnn.init_distributed_context()`. `_maybe_enable_hybrid_allocator` exports `TT_METAL_ALLOCATOR_MODE_HYBRID=1` because Kimi sets `REQUIRES_HYBRID_ALLOCATOR`.",
      "`open_mesh_device` opens this rank's mesh with `local_mesh_shape()` (the one mesh the MGD gives the rank), the fabric config read from the MGD (`_fabric_config_from_mgd`), and `worker_l1_size = get_worker_l1_size(model, num_procs, rank)`.",
      "Kimi: ranks 62 and up get the verifier-tail L1 (1461504 B) instead of 1441568 B, because the LM head and token-forward tail sit there in the 64 and 80 layouts.",
    ],
    imposes: [
      "Each rank owns exactly one local mesh.",
      "Per-rank L1 is fixed here, keyed by MPI rank, before any stage is placed.",
    ],
    tierC: "The topology mapper (and so Tier C) runs during control-plane init, before every step below. Its stage placement is a witness only; nothing here reads it.",
  },
  {
    id: "entry",
    title: "1. Kimi `entry.build_pipeline` picks a path",
    where: ["`blaze/models/kimi_k2_6/entry.py` `build_pipeline`, `build_kimi_dflash_router_pipeline_graph`", "`blaze/models/kimi_k2_6/reload.py` `build_reload_pipeline`"],
    does: [
      "`reload=True`: `build_reload_pipeline` builds a ring of ring stages (not layers) that reloads weight images in laps, through `ReloadPipeline.open`.",
      "`dflash_router=True`: needs exactly 80 processes (SC20). Router (4x2) + 63 target stages + 9 draft stages. `target62 -> router` is a fabric loopback; `draft8 -> router` is declared `transport=\"host\"`. Built by `build_model_pipeline_from_graph(..., submesh_shape=4x2, multi_mesh=True)`.",
      "Otherwise: `blaze.stages.stage.build_pipeline(model_name=..., double_torus, n_slots, mtp_level, bspm_*, dflash_options, placement)`.",
    ],
    imposes: [
      "The production Kimi fork has a declared host edge even under `--pipeline-builder strict`.",
      "Fork graphs are hand-written in `entry.py`; rings come from harnesses.",
    ],
    tierC: "The design describes the generic ring and fabric-only forks. The DFlash router's declared host return has no MGD spelling in the STRICT-only first version.",
  },
  {
    id: "reload",
    title: "1a. Reload ring (`build_reload_pipeline`)",
    where: [
      "`blaze/models/kimi_k2_6/reload.py` `build_reload_pipeline`, `check_reload_kwargs`, `FULL_SEQUENCE`",
      "`blaze/reload/ring.py` `check_ring`, `build_ring_stages`; `blaze/reload/plan.py` `RingPlan.deal`",
      "`blaze/reload/pipeline.py` `ReloadPipeline.open`, `_ring_graph`",
    ],
    does: [
      "`check_firmware_env` and `check_ring` refuse to run without slow dispatch, the blaze BRISC firmware and exactly 4 processes. `check_reload_kwargs` refuses every option the ring does not support.",
      "`RingPlan.deal` deals the 65-step `FULL_SEQUENCE` (embed, dense 0, MoE 1 to 60, two bypasses, LM head) over 4 stages: step i goes to stage i % 4, slot i // 4, giving 16 laps, and the head takes m0's extra slot.",
      "`build_ring_stages` makes one `ReloadStage` per stage with its reload images and weight budget. Only m0 reads from the loopback and writes D2H.",
      "`ReloadPipeline.open` builds `_ring_graph` (a chain plus one loopback into s0, `co_locate_ring_host_io=False`) and calls `build_topology(submesh_shape=4x2, is_loopback=True)`.",
    ],
    imposes: [
      "Four 4x2 stages on one mesh, each joined to the next by a direct link, and the last joined back to the first over the torus column wrap.",
      "Two or more channels on every hop, the wrap included.",
      "Every stage spans all 4 rows and needs the row wrap for MLA's `sdpa_reduce`.",
    ],
    tierC: "The design does not describe reload rings: a descriptor of ring stages rather than layers, with stage 0 keeping one pipeline core (G14).",
  },
  {
    id: "dflash",
    title: "1b. DFlash router fork (`build_kimi_dflash_router_pipeline_graph`)",
    where: [
      "`blaze/models/kimi_k2_6/entry.py` `build_kimi_dflash_router_pipeline_graph`, `_dflash_target_branch_factories`",
      "`blaze/models/kimi_k2_6/kimi_k2_6_blaze_config.py` `dflash_fork_worker_l1_size`",
      "`pipeline_builder/fork.py` `build_fork_layout`, `ForkLayout`",
    ],
    does: [
      "Requires `--weights real`, `--model-path`, `--cache-path`, `$TT_HF_DRAFT_MODEL` and exactly 80 processes.",
      "Builds the router S0, 63 target stages (the full Kimi harness list, LM head rebuilt with `emit_auxiliary_first=True`) and 9 drafter stages (pre-decode, 6 decoders, post-decode, winner relay).",
      "Edges: router to target0 through target62, back to the router as a fabric loopback; router to draft0 through draft8, back to the router declared `transport=\"host\"`.",
      "`build_model_pipeline_from_graph(..., submesh_shape=4x2, multi_mesh=True)` tiles per rank. `build_topology` sees a fork and returns a `ForkLayout`; the 7 unused ranks stay idle.",
    ],
    imposes: [
      "The router needs 6 pipeline cores and 3 fabric neighbours (4 if the drafter also returned over fabric).",
      "A declared host edge, even under `--pipeline-builder strict`.",
      "Worker L1 is chosen per resolved rank before topology runs.",
    ],
    tierC: "The drafter's declared host return has no MGD spelling in the STRICT-only first version, and fork capacity (router core slots, neighbour count) is not modelled (G5, G7).",
  },
  {
    id: "harness",
    title: "2. Harnesses from the process count",
    where: ["`blaze/stages/stage.py` `build_pipeline` (`_PIPELINE_BUILDERS`)", "`blaze/models/kimi_k2_6/kimi_k2_6_blaze_config.py` `get_pipeline_harnesses`, `make_blaze_config`"],
    does: [
      "`_PIPELINE_BUILDERS[model_name]` gives `(make_blaze_config, get_pipeline_harnesses)`. DFlash options are folded into the config with `with_dflash_config`.",
      "`num_stages = ttnn.distributed_context_get_size()`; `get_pipeline_harnesses(blaze_config, num_stages, mesh_device, ...)` returns one harness per stage.",
      "Kimi accepts 4, 16, 63, 64 and 80. Every Kimi harness is 4x2 and `num_ranks() == 1`; no harness overrides `pipeline_core_coords()`.",
    ],
    imposes: [
      "The stage list is a function of the world size, not of the MGD.",
      "Shapes, ranks and core pools come from harness methods.",
    ],
    tierC: "The MGD's `pipeline_descriptors` is written by hand next to the mesh. Nothing checks it against this list.",
  },
  {
    id: "pad",
    title: "3. Size the ring in ranks, pad, check boundaries",
    where: ["`blaze/stages/stage.py` `build_pipeline`", "`blaze_tests/.../pipeline_socket_passthrough_stage/harness.py` `PipelineSocketPassthroughStageHarness`"],
    does: [
      "`num_ranks_used = sum(h.num_ranks())`; asserts it does not exceed the world size.",
      "Pads with forward passthroughs until every rank has a stage. A pad takes the shape of the last single-rank harness and forwards the last harness's output size (and second stream).",
      "Asserts producer `sum(output_sizes_bytes)` equals consumer `input_size_bytes()` on every hop, and that both sides agree on a second stream.",
    ],
    imposes: [
      "On the generic path every rank carries a stage, so there are no idle ranks and no filler slots.",
      "Page sizes are fixed per hop; host hops reuse them (`host_hop_page_size_bytes`).",
    ],
    tierC: "A descriptor with fewer stages than ranks describes a ring blaze never builds on this path.",
  },
  {
    id: "graph",
    title: "4. Nodes, edges, `PipelineGraph`",
    where: ["`blaze/stages/stage.py` `_make_node`", "`pipeline_builder/graph.py` `Node`, `Edge`, `PipelineGraph.__init__`, `_validate`"],
    does: [
      "`Node(shape=harness.mesh_shape(), name=f\"s{i}\", pipeline_core_coords=harness.pipeline_core_coords(), factory=GenericStage(...), host_hop_page_size_bytes=..., secondary_host_hop_page_size_bytes=...)`.",
      "Edges: `s0 -> s1 -> ... -> s{N-1}`, then `s{N-1} -> s0` with `is_loopback=True`. Every edge is `transport=\"fabric\"`.",
      "`PipelineGraph` checks the metadata class is kernel-compatible, core pools are non-empty and unique, and edge endpoints exist. Default `co_locate_ring_host_io=True`.",
    ],
    imposes: ["Ring = chain plus one loopback into stage 0.", "Stage 0 owns H2D, the loopback entry and D2H."],
    tierC: "Same graph as the design's 1.2 harness loop. Matches the descriptor's stages and edges once padding is included.",
  },
  {
    id: "path",
    title: "5. Choose the carve path",
    where: ["`blaze/stages/stage.py` `build_pipeline`", "`blaze/models/pipeline_graph_helpers.py` `local_stage_shape`"],
    does: [
      "`multi_mesh = has_joint_stage or len(ttnn.get_all_fabric_mesh_ids()) > 1`.",
      "Joint stage: `submesh_shape=None`, so `build_topology` plans the carve from the fabric.",
      "Two distinct shapes: this rank's owned bounding box if it is a stage shape, else the big shape on mesh 0 and the small one elsewhere (FIXME tt-blaze#1455). More than two shapes is not handled here.",
      "Uniform shapes: `submesh_shape = harnesses[0].mesh_shape()`.",
      "Single mesh, no joint stage: `graph.build_topology(mesh_device, placement=...)`.",
    ],
    imposes: [
      "Without a joint stage, each mesh is tiled with one shape, chosen per rank.",
      "A single mesh needs every stage to have the same shape unless one is joint.",
    ],
    tierC: "The position graph can mix shapes on one mesh or put the big shape on mesh 1. Blaze takes the planned carve only when a stage is joint.",
  },
  {
    id: "resolve",
    title: "6. `build_topology` and `_resolve_topology`",
    where: ["`pipeline_builder/graph.py` `build_topology`, `_resolve_topology`, `_collect_submeshes`, `_resolve_chains_separately`", "`pipeline_builder/host_edge_planner.py` `plan_host_edges`, `plan_fork_host_edges`"],
    does: [
      "Resets any earlier planned host cuts. With `placement=\"flexible\"`, no declared host edges and a loopback: `plan_host_edges` (ring) or `plan_fork_host_edges` (fork; `ForkShapeUnsupported` falls back to the declared graph).",
      "`_collect_submeshes` carves, finds this rank's submesh, allgathers `(mesh_id, chip_id, row, col)` per rank, and collapses ranks with identical chip sets into one joint submesh.",
      "Edge tuples drop host edges. `node_chip_counts = rows*cols` for every node. `node_pipeline_core_counts` only for nodes with an explicit pool or a role default.",
      "Calls `resolve_graph_layout` once, or once per fabric chain when host edges split the graph. On failure rank 0 prints the fabric adjacency (unless `BLAZE_SKIP_CONNECTIVITY`) and suggests flexible.",
    ],
    imposes: [
      "At least as many distinct submeshes as nodes.",
      "A rank's chips must fall in exactly one planned slot.",
      "With host edges, stage order comes from the logical graph and endpoints of host hops go on the least-used chip.",
    ],
    tierC: "Tier C replaces the carve with one footprint solve and calls `resolve_pipeline_placement` directly, once. Per-chain calls appear only with RELAXED (P11).",
  },
  {
    id: "carve",
    title: "6a. `plan_carve` (planned carve only)",
    where: ["`pipeline_builder/carve_planner.py` `discover_host_slices`, `discover_slice_adjacency`, `plan_ring_carve`, `plan_fork_carve`, `plan_carve`"],
    does: [
      "`discover_host_slices`: each rank lists its local chips (`is_local`), allgathers them, and turns each rank into a `HostSlice(mesh_id, row0, col0, nrows, ncols)`, sorted by `(mesh_id, row0, col0)`. Every MGD mesh must have a rank.",
      "`discover_slice_adjacency`: for every chip and direction N, E, S, W, Z, a neighbour counts if `get_forwarding_direction` confirms it. Wrap links of a RING dim are included.",
      "`plan_ring_carve`: uniform slices; every stage a whole number of slices; stage 0 exactly one slice; prune leaves, contract chains, search cycles of exactly `need` slices (200,000 expansions, 64 cycles), sort by fewest mesh crossings, align stage rectangles (no wrap, no rotation), first fit, fillers, `validate_carve_plan`.",
      "`plan_fork_carve`: single-slice router, one cycle per fabric-return branch, one open path per host-return branch, 2,000 attempts.",
      "All ranks allgather a CRC32 of the plan and fail on any difference.",
    ],
    imposes: ["Needs a loopback edge, even for a fork.", "Uniform host slices only.", "Search budgets can reject a fabric that has a carve."],
    tierC: "Not ported; the design's 3.2 table lists where the footprint and the planner differ (open question 8).",
  },
  {
    id: "cpp",
    title: "7. C++ resolver",
    where: ["`tt_metal/fabric/pipeline_builder.cpp` `resolve_graph_layout_with_connections`, `topological_sort`, `discover_submesh_links`, `PlacementSearch::place_stages`, `LinkSearch::solve`, `place_host_endpoint`"],
    does: [
      "Validates names, overrides and edges, then sorts stages by Kahn on non-loopback edges.",
      "`discover_submesh_links`: a chip pair is a link only if `get_forwarding_direction` returns a direction that has active fabric routing planes, and the destination is a direct neighbour in that direction. Single hop only.",
      "`place_stages` backtracks over stage -> submesh: exact chip count; `links + 2 (stage 0, if it has a fabric edge) <= capacity x chips`; enough distinct neighbouring submeshes; linked to placed neighbours; lookahead reachability.",
      "`LinkSearch` picks one chip pair per edge, first with exclusive chips, then shared; places H2D near stage 0's forward exit and D2H near the loopback entry. Failed states are memoized; there is no time budget.",
    ],
    imposes: ["Default slots: 1 per chip on submeshes of 8+ chips, else 2.", "Each stage on a distinct submesh."],
    tierC: "Reused as-is by C2 with injected `DirectLinks` from C1.",
  },
  {
    id: "layout",
    title: "8. Layout: endpoints, ranks, host hops",
    where: ["`pipeline_builder/graph.py` `_layout_from_resolution`, `_resolve_host_io_endpoints`, `_co_locate_host_io`, `allgather_end_owners`, `_resolve_host_hops`", "`pipeline_builder/fork.py` `build_fork_layout`"],
    does: [
      "Moves H2D and D2H onto the chips of stage 0's forward exit and loopback entry (`co_locate_ring_host_io`), on free slots of a 4-core pool.",
      "Builds `pipeline_config`: one entry per stage plus entry N for stage 0's loopback entry and D2H. Maps slot i to `DEFAULT_PIPELINE_CORE_COORDS[i]`, starting at core (12, 8).",
      "`allgather_end_owners` finds which rank owns each stage's entry and exit chip. A stage 0 spanning several ranks raises `NotImplementedError`.",
      "Allgathers hostnames and `TT_VISIBLE_DEVICES`, builds `StageMetadata`, resolves host hops (tags follow host-edge declaration order), returns `PipelineLayout`. Forks go to `build_fork_layout` instead (two branches only).",
      "Rank 0 writes `pipeline_config.json`.",
    ],
    imposes: ["Stage 0 is driven by one rank on every path.", "Host edges only on a ring that loops back."],
    tierC: "The design's open question 14 keeps stage 0 on one slice only for the planner; here it is a runtime limit on every path.",
  },
  {
    id: "run",
    title: "9. Build sockets, programs and the model pipeline",
    where: ["`pipeline_builder/submesh_partition.py` `PipelineLayout.build_pipeline`", "`blaze/models/pipeline_block.py` `PipelineBlock`", "`blaze/models/pipeline_sockets/d2d_exchange.py` `SocketInterface`", "`blaze/stages/stage.py` `GenericStage.setup`", "`blaze/models/model_pipeline.py` `ModelPipeline.from_layout_pipeline`"],
    does: [
      "Host hops need persistent mode; a DFlash ring cannot return to stage 0 over host.",
      "`enable_asynchronous_slow_dispatch(my_submesh)`, then `layout.build_pipeline()` builds this rank's `PipelineBlock`: H2D and D2H on stage 0, a D2D `SocketInterface` per hop, optional second stream, host-hop pumps.",
      "Each D2D endpoint opens fabric connections on link indices 0 and 1 (one per RISC) and stripes a page over `NUM_FWD_LINKS = 2`. Across meshes the hop must be a direct single-hop link; inside a mesh it may be multi-hop.",
      "`GenericStage.setup` picks the socket getters for this stage's role and calls `harness.create_program`; passthroughs run no program.",
      "`ModelPipeline.from_layout_pipeline` needs slow dispatch and a process count in {4, 8, 16, 35, 64, 72, 80, 96, 160}.",
    ],
    imposes: ["Two routing planes per fabric endpoint (to verify).", "Process count whitelist."],
    tierC: "Tier C stops at the resolver. Socket-level link needs are not modelled.",
  },
];

// ---------------------------------------------------------------------------
// Call graph
// ---------------------------------------------------------------------------

const FLOW_NODES: Array<{ id: string; label: string; step: string }> = [
  { id: "cli", label: "cli.open_mesh_device", step: "launch" },
  { id: "entry", label: "kimi entry.build_pipeline", step: "entry" },
  { id: "reload", label: "build_reload_pipeline", step: "reload" },
  { id: "dflash", label: "dflash router graph", step: "dflash" },
  { id: "generic", label: "stage.build_pipeline", step: "harness" },
  { id: "pad", label: "rank sizing + padding", step: "pad" },
  { id: "ring", label: "PipelineGraph ring", step: "graph" },
  { id: "topo", label: "build_topology", step: "resolve" },
  { id: "planned", label: "planned carve", step: "carve" },
  { id: "tiling", label: "per-rank tiling", step: "path" },
  { id: "single", label: "single-mesh tiling", step: "path" },
  { id: "resolver", label: "resolve_graph_layout", step: "cpp" },
  { id: "linear", label: "PipelineLayout", step: "layout" },
  { id: "fork", label: "ForkLayout", step: "layout" },
  { id: "sockets", label: "PipelineBlock sockets", step: "run" },
  { id: "model", label: "ModelPipeline", step: "run" },
];

const FLOW_EDGES: Array<{ from: string; to: string }> = [
  { from: "cli", to: "entry" },
  { from: "entry", to: "reload" },
  { from: "entry", to: "dflash" },
  { from: "entry", to: "generic" },
  { from: "generic", to: "pad" },
  { from: "pad", to: "ring" },
  { from: "ring", to: "topo" },
  { from: "dflash", to: "topo" },
  { from: "reload", to: "topo" },
  { from: "topo", to: "planned" },
  { from: "topo", to: "tiling" },
  { from: "topo", to: "single" },
  { from: "planned", to: "resolver" },
  { from: "tiling", to: "resolver" },
  { from: "single", to: "resolver" },
  { from: "resolver", to: "linear" },
  { from: "resolver", to: "fork" },
  { from: "linear", to: "sockets" },
  { from: "fork", to: "sockets" },
  { from: "sockets", to: "model" },
];

type GraphKind = "dflash" | "reload";

function CallGraph({ selected, onSelect }: { selected: string; onSelect: (step: string) => void }) {
  const theme = useHostTheme();
  const layout = computeDAGLayout({
    nodes: FLOW_NODES.map((n) => ({ id: n.id })),
    edges: FLOW_EDGES,
    direction: "vertical",
    nodeWidth: 168,
    nodeHeight: 32,
    rankGap: 30,
    nodeGap: 18,
    padding: 12,
  });
  const byId = new Map(FLOW_NODES.map((n) => [n.id, n]));
  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={layout.width} height={layout.height} style={{ display: "block" }}>
        <defs>
          <marker id="flow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={theme.stroke.primary} />
          </marker>
        </defs>
        {layout.edges.map((e, i) => (
          <line
            key={i}
            x1={e.sourceX}
            y1={e.sourceY}
            x2={e.targetX}
            y2={e.targetY}
            stroke={theme.stroke.primary}
            strokeWidth={1}
            markerEnd="url(#flow-arrow)"
          />
        ))}
        {layout.nodes.map((n) => {
          const node = byId.get(n.id)!;
          const active = node.step === selected;
          return (
            <g key={n.id} onClick={() => onSelect(node.step)} style={{ cursor: "pointer" }}>
              <rect
                x={n.x}
                y={n.y}
                width={168}
                height={32}
                rx={6}
                fill={active ? theme.fill.secondary : theme.bg.elevated}
                stroke={active ? theme.accent.primary : theme.stroke.secondary}
                strokeWidth={active ? 1.5 : 1}
              />
              <text
                x={n.x + 84}
                y={n.y + 20}
                textAnchor="middle"
                fontSize={11}
                fill={active ? theme.text.primary : theme.text.secondary}
                style={{ fontFamily: "ui-monospace, monospace" }}
              >
                {node.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function Bullets({ items }: { items: string[] }) {
  return (
    <Stack gap={6}>
      {items.map((t, i) => (
        <Row key={i} gap={8} align="start">
          <Text tone="tertiary" size="small" style={{ minWidth: 10 }}>
            -
          </Text>
          <Text size="small">{t}</Text>
        </Row>
      ))}
    </Stack>
  );
}

function MapperIntegrationSection({ step }: { step: Step }) {
  const mapper = MAPPER[step.id];
  if (!mapper) return null;
  return (
    <CollapsibleSection title="Topology mapper integration" defaultOpen>
      <Stack gap={10}>
        <Bullets items={mapper.does} />
        <Text size="small" tone="secondary">
          <Text weight="semibold" size="small">
            {"Where blaze and the design differ: "}
          </Text>
          {step.tierC}
        </Text>
        <Stack gap={4}>
          <Text size="small" tone="tertiary">
            In the design doc (`docs/scratch/mgd-pipeline-graph-design.md`):
          </Text>
          {mapper.refs.map((ref) => (
            <Link key={ref} href={`${DESIGN_DOC_URL}#${docAnchor(ref)}`}>
              <Text size="small">{/^\d/.test(ref) ? `§${ref}` : ref}</Text>
            </Link>
          ))}
        </Stack>
      </Stack>
    </CollapsibleSection>
  );
}

const STEP_PANELS: Record<string, () => ReturnType<typeof Text>> = {
  entry: () => <EntryPathsPanel />,
  reload: () => <ReloadRingPanel />,
  dflash: () => <DFlashForkPanel />,
  harness: () => <ConfigToGraphPanel />,
  graph: () => <GenericRingPanel />,
};

function StepDetail({ step }: { step: Step }) {
  return (
    <Card>
      <CardHeader>{step.title.replace(/`/g, "")}</CardHeader>
      <CardBody>
        <Stack gap={12}>
          <Stack gap={4}>
            {step.where.map((w, i) => (
              <Text key={i} size="small" tone="secondary">
                {w}
              </Text>
            ))}
          </Stack>
          <Divider />
          <Text weight="semibold" size="small">
            What happens
          </Text>
          <Bullets items={step.does} />
          <Text weight="semibold" size="small">
            What it imposes on the topology
          </Text>
          <Bullets items={step.imposes} />
          {STEP_PANELS[step.id] && (
            <>
              <Divider />
              {STEP_PANELS[step.id]()}
            </>
          )}
          <Divider />
          <MapperIntegrationSection step={step} />
        </Stack>
      </CardBody>
    </Card>
  );
}

function BuildWalkthrough() {
  const [selected, setSelected] = useCanvasState<string>("step", "launch");
  const step = STEPS.find((s) => s.id === selected) ?? STEPS[0];
  return (
    <Stack gap={12}>
      <Row gap={6} wrap>
        {STEPS.map((s) => (
          <Pill key={s.id} size="sm" active={s.id === selected} onClick={() => setSelected(s.id)}>
            {s.title.split(" ")[0].replace(".", "")}
          </Pill>
        ))}
      </Row>
      <Stack gap={6} style={{ alignItems: "center" }}>
        <CallGraph selected={selected} onSelect={setSelected} />
        <Text size="small" tone="tertiary">
          Click a box or a step number. Boxes that share a step light up together.
        </Text>
      </Stack>
      <StepDetail step={step} />
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Carve paths and Kimi layouts
// ---------------------------------------------------------------------------

const CARVE_ROWS: string[][] = [
  [
    "Planned carve",
    "`multi_mesh` and no `submesh_shape` (a joint stage on the generic path)",
    "`plan_carve`, then `create_submesh(shape, offset)` per slot of my mesh",
    "uniform slices, stage 0 one slice, cycle of exactly `need` slices, rectangles without wrap, budgets",
    "the footprint solve (any mix of positions)",
  ],
  [
    "Per-rank tiling",
    "`multi_mesh` with `submesh_shape` (uniform, or two shapes picked per rank)",
    "`create_submeshes(submesh_shape)`; keep the tile whose (0,0) is local",
    "one shape per mesh; at most two shapes, big one on mesh 0 unless the rank's own box says otherwise",
    "positions of the tile shape; mixing on one mesh is not constrained",
  ],
  [
    "Single-mesh tiling",
    "one fabric mesh and no joint stage",
    "`create_submeshes(shape)`; every tile is local",
    "every stage the same shape; tiles >= nodes",
    "one mesh, one shape",
  ],
];

const KIMI_ROWS: string[][] = [
  ["4", "`single_bh_galaxy_4x2` (8x4, host 2x2) or `single_bh_galaxy_4x8_torus_y`", "embed, dense 0, MoE 1, LM head (`double_torus`) or embed, 2 passthroughs, LM head", "single mesh, uniform 4x2"],
  ["16", "`SC4_4x2` (16 meshes of 4x2, RING x LINE)", "embed, dense 0, MoE 1 to 12, LM head, 1 passthrough", "per-rank tiling, 4x2"],
  ["64", "`blitz_decode_mesh_graph_descriptor_superpod` (64 meshes of 4x2)", "embed, dense 0, MoE 1 to 60, LM head, 1 passthrough", "per-rank tiling, 4x2"],
  ["80", "SC20-class (5 meshes of 4x32, host 1x16)", "63 compute stages + 17 passthroughs", "per-rank tiling, 4x2"],
  ["80 (DFlash)", "`SC20_4x2` (5 meshes of 4x32, fully connected)", "router + 63 target + 9 draft; draft return over host", "fork, `multi_mesh=True`, 4x2"],
];

// ---------------------------------------------------------------------------
// Two Kimi graphs: the DFlash router fork and the reload ring
// ---------------------------------------------------------------------------

function tint(color: string, pct: number): string {
  return `color-mix(in srgb, ${color} ${pct}%, transparent)`;
}

function NodeBox({
  x,
  y,
  w,
  title,
  sub,
  color,
  stacked,
}: {
  x: number;
  y: number;
  w: number;
  title: string;
  sub: string;
  color?: string;
  stacked?: boolean;
}) {
  const theme = useHostTheme();
  const fill = color ? tint(color, 18) : theme.bg.elevated;
  const stroke = theme.stroke.primary;
  return (
    <g>
      {stacked && <rect x={x + 4} y={y - 4} width={w} height={56} rx={6} fill={fill} stroke={stroke} strokeWidth={1} />}
      <rect x={x} y={y} width={w} height={56} rx={6} fill={stacked ? theme.bg.editor : fill} stroke={stroke} strokeWidth={color ? 1 : 1.5} />
      {stacked && <rect x={x} y={y} width={w} height={56} rx={6} fill={fill} stroke="none" />}
      <text x={x + w / 2} y={y + 23} textAnchor="middle" fontSize={12} fill={theme.text.primary}>
        {title}
      </text>
      <text x={x + w / 2} y={y + 42} textAnchor="middle" fontSize={11} fill={theme.text.secondary}>
        {sub}
      </text>
    </g>
  );
}

function LegendItem({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  const theme = useHostTheme();
  return (
    <Row gap={6} align="center">
      {dashed !== undefined ? (
        <svg width={22} height={10}>
          <line x1={1} y1={5} x2={21} y2={5} stroke={color} strokeWidth={1.5} strokeDasharray={dashed ? "4 3" : undefined} />
        </svg>
      ) : (
        <div style={{ width: 10, height: 10, borderRadius: 2, background: tint(color, 40), border: `1px solid ${theme.stroke.secondary}` }} />
      )}
      <Text size="small" tone="secondary">
        {label}
      </Text>
    </Row>
  );
}

function DFlashForkDiagram() {
  const theme = useHostTheme();
  const fg = theme.stroke.primary;
  const blue = theme.category.blue;
  const purple = theme.category.purple;
  const host = theme.category.orange;
  const edge = { fill: "none", stroke: fg, strokeWidth: 1 } as const;
  return (
    <Stack gap={8}>
      <Row gap={16} wrap>
        <LegendItem color={blue} label="Target branch (Kimi verifier)" />
        <LegendItem color={purple} label="Draft branch (DFlash drafter)" />
        <LegendItem color={fg} label="Fabric edge" dashed={false} />
        <LegendItem color={host} label="Declared host edge (MPI)" dashed />
      </Row>
      <svg viewBox="0 0 736 374" width="100%" style={{ maxWidth: 820, display: "block" }}>
        <defs>
          <marker id="fork-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={fg} />
          </marker>
          <marker id="fork-arrow-host" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={host} />
          </marker>
        </defs>
        <path {...edge} d="M675 44 V20 H91 V156" markerEnd="url(#fork-arrow)" />
        <text x={383} y={12} textAnchor="middle" fontSize={11} fill={theme.text.secondary}>
          fabric loopback, S63 to router
        </text>
        <path d="M684 328 V352 H91 V216" fill="none" stroke={host} strokeWidth={1.5} strokeDasharray="5 4" markerEnd="url(#fork-arrow-host)" />
        <text x={383} y={370} textAnchor="middle" fontSize={11} fill={theme.text.secondary}>
          declared host edge, S72 to router
        </text>
        <path {...edge} d="M166 172 H198 V72 H226" markerEnd="url(#fork-arrow)" />
        <path {...edge} d="M166 200 H198 V300 H226" markerEnd="url(#fork-arrow)" />
        <NodeBox x={16} y={158} w={150} title="S0 router" sub="6 pipeline cores" />

        <path {...edge} d="M320 72 H338" markerEnd="url(#fork-arrow)" />
        <path {...edge} d="M436 72 H454" markerEnd="url(#fork-arrow)" />
        <path {...edge} d="M610 72 H624" markerEnd="url(#fork-arrow)" />
        <NodeBox x={228} y={44} w={92} title="S1 embed" sub="target0" color={blue} />
        <NodeBox x={340} y={44} w={96} title="S2 dense L0" sub="target1" color={blue} />
        <NodeBox x={456} y={44} w={150} title="S3–S62 MoE L1–60" sub="60 stages" color={blue} stacked />
        <NodeBox x={626} y={44} w={98} title="S63 LM head" sub="target62" color={blue} />

        <path {...edge} d="M340 300 H358" markerEnd="url(#fork-arrow)" />
        <path {...edge} d="M488 300 H502" markerEnd="url(#fork-arrow)" />
        <path {...edge} d="M624 300 H642" markerEnd="url(#fork-arrow)" />
        <NodeBox x={228} y={272} w={112} title="S64 pre-decode" sub="draft0" color={purple} />
        <NodeBox x={360} y={272} w={124} title="S65–S70 decoder" sub="6 layers" color={purple} stacked />
        <NodeBox x={504} y={272} w={120} title="S71 post-decode" sub="draft7" color={purple} />
        <NodeBox x={644} y={272} w={80} title="S72 relay" sub="draft8" color={purple} />
      </svg>
    </Stack>
  );
}

function ReloadRingDiagram() {
  const theme = useHostTheme();
  const fg = theme.stroke.primary;
  const stages = [0, 1, 2, 3];
  return (
    <svg viewBox="0 0 736 252" width="100%" style={{ maxWidth: 820, display: "block" }}>
      <defs>
        <marker id="ring-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill={fg} />
        </marker>
        <marker id="ring-arrow-start" markerWidth="8" markerHeight="8" refX="1" refY="4" orient="auto">
          <path d="M8,0 L0,4 L8,8 z" fill={fg} />
        </marker>
      </defs>
      {stages.map((j) => {
        const sx = 136 + j * 126;
        return (
          <g key={j}>
            <rect x={sx - 6} y={58} width={98} height={134} rx={6} fill="none" stroke={fg} strokeWidth={1.5} />
            <text x={sx + 43} y={48} textAnchor="middle" fontSize={12} fill={theme.text.primary}>
              {`m${j} · cols ${2 * j}–${2 * j + 1}`}
            </text>
            {[0, 1].flatMap((c) =>
              [0, 1, 2, 3].map((r) => (
                <rect
                  key={`${c}-${r}`}
                  x={sx + c * 46}
                  y={64 + r * 32}
                  width={40}
                  height={26}
                  rx={3}
                  fill={theme.fill.secondary}
                  stroke={theme.stroke.secondary}
                  strokeWidth={1}
                />
              )),
            )}
            {j < 3 && <path d={`M${sx + 92} 125 H${sx + 118}`} fill="none" stroke={fg} strokeWidth={1} markerEnd="url(#ring-arrow)" />}
          </g>
        );
      })}
      <path d="M606 125 H638" fill="none" stroke={fg} strokeWidth={1} markerEnd="url(#ring-arrow)" />
      <text x={646} y={129} fontSize={11} fill={theme.text.secondary}>
        wrap to m0
      </text>
      <path d="M98 125 H128" fill="none" stroke={fg} strokeWidth={1} markerEnd="url(#ring-arrow)" />
      <text x={90} y={129} textAnchor="end" fontSize={11} fill={theme.text.secondary}>
        wrap from m3
      </text>
      <path d="M179 194 V210" fill="none" stroke={fg} strokeWidth={1} markerStart="url(#ring-arrow-start)" markerEnd="url(#ring-arrow)" />
      <rect x={130} y={214} width={98} height={32} rx={6} fill={theme.bg.elevated} stroke={fg} strokeWidth={1} />
      <text x={179} y={234} textAnchor="middle" fontSize={11} fill={theme.text.primary}>
        host H2D, D2H
      </text>
    </svg>
  );
}

type ReloadKind = "embed" | "dense" | "sparse" | "bypass" | "head";
type ReloadItem = [ReloadKind, number | null];

const RELOAD_BODY: ReloadItem[] = [
  ["embed", null],
  ["dense", 0],
  ...Array.from({ length: 60 }, (_, i): ReloadItem => ["sparse", i + 1]),
  ["bypass", null],
  ["bypass", null],
];
const RELOAD_STAGES = 4;
const RELOAD_LAPS = RELOAD_BODY.length / RELOAD_STAGES;

function reloadGrid(): Array<Array<{ item: ReloadItem; step: number } | null>> {
  const grid = Array.from({ length: RELOAD_STAGES }, () =>
    Array.from({ length: RELOAD_LAPS + 1 }, (): { item: ReloadItem; step: number } | null => null),
  );
  RELOAD_BODY.forEach((item, step) => {
    grid[step % RELOAD_STAGES][Math.floor(step / RELOAD_STAGES)] = { item, step };
  });
  grid[0][RELOAD_LAPS] = { item: ["head", null], step: RELOAD_BODY.length };
  return grid;
}

function reloadLabel([kind, id]: ReloadItem): string {
  if (kind === "embed") return "E";
  if (kind === "dense") return `D${id}`;
  if (kind === "sparse") return `M${id}`;
  if (kind === "bypass") return "B";
  return "H";
}

function ReloadSlotGrid() {
  const theme = useHostTheme();
  const kindColor: Record<ReloadKind, string> = {
    embed: tint(theme.category.cyan, 28),
    dense: tint(theme.category.blue, 28),
    sparse: theme.fill.tertiary,
    bypass: tint(theme.category.orange, 26),
    head: tint(theme.category.pink, 30),
  };
  const grid = reloadGrid();
  const cols = `44px repeat(${RELOAD_LAPS + 1}, minmax(34px, 1fr))`;
  const cell = { padding: "5px 0", textAlign: "center" as const, fontSize: 11, borderRadius: 4 };
  return (
    <Stack gap={8}>
      <Row gap={16} wrap>
        <LegendItem color={theme.category.cyan} label="E embedding" />
        <LegendItem color={theme.category.blue} label="D dense layer" />
        <LegendItem color={theme.stroke.secondary} label="M MoE layer" />
        <LegendItem color={theme.category.orange} label="B bypass (pads the laps)" />
        <LegendItem color={theme.category.pink} label="H LM head" />
      </Row>
      <div style={{ overflowX: "auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: cols, gap: 3, minWidth: 680 }}>
          <div style={{ ...cell, color: theme.text.tertiary }}>slot</div>
          {Array.from({ length: RELOAD_LAPS + 1 }, (_, k) => (
            <div key={`h${k}`} style={{ ...cell, color: theme.text.tertiary }}>
              {k}
            </div>
          ))}
          {grid.map((row, s) => [
            <div key={`r${s}`} style={{ ...cell, textAlign: "left", color: theme.text.secondary, fontFamily: "ui-monospace, monospace" }}>
              {`m${s}`}
            </div>,
            ...row.map((c, k) => (
              <div
                key={`${s}-${k}`}
                title={c ? `m${s}, slot ${k}, model step ${c.step}` : undefined}
                style={{
                  ...cell,
                  background: c ? kindColor[c.item[0]] : "transparent",
                  color: c ? theme.text.primary : theme.text.quaternary,
                  fontFamily: "ui-monospace, monospace",
                }}
              >
                {c ? reloadLabel(c.item) : "–"}
              </div>
            )),
          ])}
        </div>
      </div>
    </Stack>
  );
}

const FORK_WIDE_REQS: string[][] = [
  ["Ranks", "80 processes (`_DFLASH_ROUTER_NUM_PROCS`). 73 stages occupy submeshes; the other 7 ranks build nothing but join every barrier."],
  ["Mesh graph", "SC20: 5 meshes of 4x32, both axes RING, host 1x16, 2 channels inside a mesh, 16 channels between every pair of meshes."],
  ["Placement", "`multi_mesh=True`, per-rank 4x2 tiling. The draft return is declared `transport=\"host\"` because a second fabric loopback makes the 63-stage branch too expensive for the resolver."],
  ["Fabric packet", "15232 B max payload (the default; the linear Kimi ring uses 8192)."],
  ["Inputs", "`--weights real`, `--model-path`, `--cache-path`, `$TT_HF_DRAFT_MODEL`. Drafter `block_size` must equal the router's (8)."],
  ["L1 by rank", "Budgets follow the resolved SC20 placement, which differs per cluster (110-A: router rank 2, drafters 32–38, post-decode 47). Deployments override with `KIMI_FORK_DRAFTER_RANKS`, `KIMI_FORK_DRAFTER_WORKER_L1`, `KIMI_FORK_WORKER_L1_OVERRIDES`."],
];

const FORK_STAGE_REQS: string[][] = [
  [
    "S0 router",
    "`BlockDiffusionRouterStageKind`, block size 8; one fused op owning six sockets, one per core",
    "Out to S1 and S64; in from S63 over fabric; S72's pages arrive by host-hop pump and enter through an H2D socket on the drafter-receive core. 3 fabric neighbours (4 if both returns used fabric)",
    "6 (2 + 2 per branch)",
    "1461504 (verifier size); `DFLASH_FORK_ROUTER_WORKER_L1_SIZE` = 1484032 is defined but no default entry uses it",
    "Only rank with host input and output; `router_hidden_dim` = Kimi hidden size",
  ],
  [
    "S1 embed",
    "`GenericStage` over the Kimi harness list, index 1 (index 0 is a never-selected router placeholder)",
    "In from S0, out to S2",
    "1",
    "1461504",
    "All 4 rows of its 4x2",
  ],
  [
    "S2–S62 dense L0, MoE L1–60",
    "MLA decoder layers, one per stage",
    "One in, one out",
    "1",
    "1461504; rank 15 gets 1458176 by default",
    "All 4 rows; MLA's `sdpa_reduce` rings over the row axis, so dim 0 must be RING",
  ],
  [
    "S63 LM head",
    "LM head on 128 matmul cores, rebuilt with `emit_auxiliary_first=True`",
    "In from S62; fabric loopback to S0",
    "1",
    "1461504",
    "Returns `[auxiliary | metadata]` to the router, not the host readout order",
  ],
  [
    "S64 pre-decode",
    "`DFlashPreDecodeStageKind`, `fork_mode=True`",
    "In from S0, out to S65",
    "1",
    "1474560 on drafter ranks",
    "Drafter ranks come from the resolved placement, not the graph",
  ],
  [
    "S65–S70 decoder",
    "`DFlashDecoderStageKind` layers 0–5; KV migration layer ids 61–66",
    "Chain",
    "1",
    "1474560",
    "Larger kernel binaries must fit the kernel-config ring buffer",
  ],
  [
    "S71 post-decode",
    "`DFlashPostDecodeStageKind`",
    "Chain",
    "1",
    "1484032 (rank 56 by default)",
    "Deferred semaphores need a common free slot across 130 cores after sockets are placed",
  ],
  [
    "S72 winner relay",
    "`PassthroughStage`, winner page, 16-page FIFOs both sides",
    "In from S71; declared host edge to S0: the branch resolves as an open path ending in its own D2H",
    "1 (8 chips), so its D2H goes on another chip",
    "1461504",
    "Its rank runs the host-hop pump to the router rank",
  ],
];

const RING_WIDE_REQS: string[][] = [
  ["Ranks", "Exactly 4, one per stage (`check_ring`). `ttrun` gives each rank eight chips of the galaxy."],
  ["Mesh graph", "`single_bh_galaxy_4x8_torus_xy`: one 4x8 mesh, both axes RING, host 1x4, `FABRIC_2D_TORUS_XY`."],
  ["Channels per hop", "2 on every edge, the column wrap included. Each D2D endpoint stripes a page over 2 links, and the galaxy has the links for it (12 live per chip after a good reset). Keep `channels count: 2`; lowering it hides a one-link edge instead of avoiding it."],
  ["Links", "Consecutive stages need a direct link; routing through another chip does not count. On a LINE column axis the resolver fails with \"no valid submesh assignment\"."],
  ["Inside a stage", "Every stage owns all 4 rows and needs the row wrap: MLA's `sdpa_reduce` rings over `(row + offset) % 4`, or it opens two connections on one router and blaze rejects the build."],
  ["Runtime", "Slow dispatch, hybrid allocator, blaze BRISC firmware (`TT_METAL_DISABLE_PRECOMPILED_FW=1`, `TT_METAL_FW_SRC_BRISC=blaze`, `TT_METAL_FW_HEADER_BRISC` = `runtime_reload.h`)."],
  ["Budgets", "Worker L1 1441568 B on every rank; 15232 B max packet payload."],
  ["Graph", "`co_locate_ring_host_io=False`, so stage 0 keeps its shape default of 1 pipeline core instead of 4. The 64 steps before the head must fill whole laps; two bypasses pad Kimi to 16."],
  ["Refused options", "`double_torus`, `n_slots` other than 1, `mtp_level`, `dflash_router`, `fc_matmul_enabled`, hot experts, layer id overrides, an I/O socket prefix, `persistent_mode=False`, synthetic weights."],
];

const RING_STAGE_REQS: string[][] = [
  ["m0", "cols 0–1", "17: embed, MoE 3, 7, …, 59, LM head", "488", "In from m3 over the col 7 to col 0 wrap; out to m1", "H2D in; D2H page = LM head output. The only stage whose endpoints change per reload image"],
  ["m1", "cols 2–3", "16: dense 0, MoE 4, 8, …, 60", "544", "In from m0, out to m2", "None"],
  ["m2", "cols 4–5", "16: MoE 1, 5, …, 57, bypass", "480", "In from m1, out to m3", "None"],
  ["m3", "cols 6–7", "16: MoE 2, 6, …, 58, bypass", "480", "In from m2; out to m0 over the wrap", "None"],
];

const TIER_C_CARRY: Array<{ for: GraphKind[]; row: string[] }> = [
  { for: ["dflash", "reload"], row: ["Two or more channels on every hop", "The D2D socket stripes each page over 2 links; the reload descriptor asks for 2 on every edge, wraps included", "`R` records one channel per crossing", "G2"] },
  { for: ["dflash", "reload"], row: ["A direct link between consecutive stages", "Every hop", "C1 also links positions over routed intra-mesh links", "G3"] },
  { for: ["reload"], row: ["A torus wrap as a stage-to-stage loopback", "m3 to m0 over col 7 to col 0", "Silent on wrap links between positions", "G4"] },
  { for: ["dflash", "reload"], row: ["A row ring inside each MLA stage", "Every MLA stage (SC20 and the 4x8 torus are RING on dim 0)", "Positions are rectangles; no intra-position topology", "G4"] },
  { for: ["dflash"], row: ["A declared host edge on one fork branch", "S72 to S0", "STRICT only; a host return is RELAXED, later", "G5, G6"] },
  { for: ["dflash"], row: ["Router with 6 pipeline cores and 3 or 4 fabric neighbours", "S0", "Fork capacity not modelled", "G7, G10"] },
  { for: ["dflash"], row: ["L1 budget by resolved rank", "Router, drafter and override ranks", "Stage-to-rank map is a witness only", "G1"] },
  { for: ["dflash", "reload"], row: ["Fixed process count", "80 for DFlash, 4 for reload", "Stage list comes from the MGD", "G12"] },
  { for: ["reload"], row: ["Ring host I/O kept off the D2D chips", "`co_locate_ring_host_io=False`", "Core-slot defaults assumed from graph shape", "G10, G14"] },
];

function ReqTable({ headers, rows }: { headers: string[]; rows: string[][] }) {
  return <Table headers={headers} rows={rows.map((r) => r.map((c, i) => <Text key={i} size="small">{c}</Text>))} />;
}

function TierCCarryTable({ kind }: { kind: GraphKind }) {
  return (
    <Stack gap={8}>
      <Text weight="semibold" size="small">
        What Tier C has to carry for this graph
      </Text>
      <ReqTable
        headers={["Requirement", "Needed by", "Tier C today", "Gap"]}
        rows={TIER_C_CARRY.filter((r) => r.for.includes(kind)).map((r) => r.row)}
      />
    </Stack>
  );
}

function EntryPathsDiagram() {
  const theme = useHostTheme();
  const fg = theme.stroke.primary;
  const lanes = [
    {
      x: 128,
      cond: "reload=True",
      boxes: [
        ["build_reload_pipeline", "mesh + sampling options"],
        ["FULL_SEQUENCE, deal", "65 steps to 4 stage lists"],
        ["ReloadPipeline.open", "4-node ring, ReloadModelPipeline"],
      ],
    },
    {
      x: 368,
      cond: "dflash_router=True",
      boxes: [
        ["dflash router graph", "PipelineGraph, 73 nodes"],
        ["build_model_pipeline_from_graph", "submesh 4x2, multi_mesh"],
        ["ForkLayout", "router, branches, 7 idle"],
      ],
    },
    {
      x: 608,
      cond: "otherwise",
      boxes: [
        ["stage.build_pipeline", "model name + options"],
        ["get_pipeline_harnesses", "N harnesses, N = world size"],
        ["PipelineGraph ring", "N nodes, then ModelPipeline"],
      ],
    },
  ];
  return (
    <svg viewBox="0 0 736 336" width="100%" style={{ maxWidth: 820, display: "block" }}>
      <defs>
        <marker id="entry-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill={fg} />
        </marker>
      </defs>
      <NodeBox x={218} y={0} w={300} title="entry.build_pipeline" sub="mesh_device + CLI options" />
      <path d="M368 56 V70 M128 70 H608" fill="none" stroke={fg} strokeWidth={1} />
      {lanes.map((lane) => (
        <g key={lane.cond}>
          <path d={`M${lane.x} 70 V108`} fill="none" stroke={fg} strokeWidth={1} markerEnd="url(#entry-arrow)" />
          <text x={lane.x + 8} y={94} fontSize={11} fill={theme.text.secondary} style={{ fontFamily: "ui-monospace, monospace" }}>
            {lane.cond}
          </text>
          {lane.boxes.map(([title, sub], i) => (
            <g key={title}>
              <NodeBox x={lane.x - 110} y={110 + i * 80} w={220} title={title} sub={sub} />
              {i < 2 && (
                <path
                  d={`M${lane.x} ${166 + i * 80} V${188 + i * 80}`}
                  fill="none"
                  stroke={fg}
                  strokeWidth={1}
                  markerEnd="url(#entry-arrow)"
                />
              )}
            </g>
          ))}
        </g>
      ))}
    </svg>
  );
}

const ENTRY_PASSED_IN: string[][] = [
  [
    "reload",
    "`build_reload_pipeline(mesh_device, model_name, cache_path, model_path, seed, top_k, top_p, temperature)`",
    "No graph. The model is the 65-step `FULL_SEQUENCE`, dealt into 4 stage lists; `ReloadPipeline.open` builds a 4-node ring from them",
    "`ReloadModelPipeline`, already launched",
  ],
  [
    "dflash_router",
    "`build_model_pipeline_from_graph(graph, mesh_device, submesh_shape=4x2, multi_mesh=True, placement=...)`",
    "A hand-written `PipelineGraph`: router, 63 target nodes, 9 draft nodes, one fabric loopback, one host loopback",
    "`ModelPipeline` over a `ForkLayout`",
  ],
  [
    "otherwise",
    "`stage.build_pipeline(mesh_device, model_name, weights_mode, ..., dflash_options, placement)`",
    "No graph. `get_pipeline_harnesses` returns one harness per rank; `build_pipeline` pads and walks them into a ring",
    "`ModelPipeline` over a `PipelineLayout`",
  ],
];

function EntryPathsPanel() {
  return (
    <Stack gap={12}>
      <Text weight="semibold" size="small">
        What each path is handed, and what it builds
      </Text>
      <EntryPathsDiagram />
      <ReqTable headers={["Path", "Call from entry.py", "Pipeline that is passed in", "Returns"]} rows={ENTRY_PASSED_IN} />
    </Stack>
  );
}

function GenericRingDiagram() {
  const theme = useHostTheme();
  const fg = theme.stroke.primary;
  const blue = theme.category.blue;
  const edge = { fill: "none", stroke: fg, strokeWidth: 1 } as const;
  return (
    <svg viewBox="0 0 736 150" width="100%" style={{ maxWidth: 820, display: "block" }}>
      <defs>
        <marker id="generic-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill={fg} />
        </marker>
      </defs>
      <NodeBox x={12} y={30} w={100} title="S0 embed" sub="H2D, D2H here" color={blue} />
      <path {...edge} d="M112 58 H130" markerEnd="url(#generic-arrow)" />
      <NodeBox x={132} y={30} w={104} title="S1 dense L0" sub="harness 1" color={blue} />
      <path {...edge} d="M236 58 H254" markerEnd="url(#generic-arrow)" />
      <NodeBox x={256} y={30} w={168} title="S2–S61 MoE L1–60" sub="60 stages" color={blue} stacked />
      <path {...edge} d="M428 58 H446" markerEnd="url(#generic-arrow)" />
      <NodeBox x={448} y={30} w={112} title="S62 LM head" sub="verifier tail L1" color={blue} />
      <path {...edge} d="M560 58 H578" markerEnd="url(#generic-arrow)" />
      <NodeBox x={580} y={30} w={144} title="S63 passthrough" sub="pads the ring" />
      <path {...edge} d="M652 86 V112 H62 V88" markerEnd="url(#generic-arrow)" />
      <text x={357} y={130} textAnchor="middle" fontSize={11} fill={theme.text.secondary}>
        fabric loopback, S63 to S0
      </text>
    </svg>
  );
}

function GenericRingPanel() {
  return (
    <Stack gap={12}>
      <Text weight="semibold" size="small">
        The ring the generic path builds for Kimi on 64 processes
      </Text>
      <Text size="small" tone="secondary">
        {"Every node is `s{i}` with the harness's 4x2 shape and a `GenericStage` factory. Other process counts give other lists; see the Kimi layouts table."}
      </Text>
      <GenericRingDiagram />
    </Stack>
  );
}

const CONFIG_ROWS: Array<{ harness: string; loads: string; node: string; stacked?: boolean }> = [
  { harness: "[0] Embedding", loads: "load_embedding", node: "s0" },
  { harness: "[1] Dense L0", loads: "load_dense_layer(0)", node: "s1" },
  { harness: "[2..61] MoE L1–60", loads: "load_moe_layer(i - 1)", node: "s2 … s61", stacked: true },
  { harness: "[62] LM head", loads: "load_lm_head", node: "s62" },
  { harness: "[63] Passthrough", loads: "pad, no weights", node: "s63" },
];

function ConfigToGraphDiagram() {
  const theme = useHostTheme();
  const fg = theme.stroke.primary;
  const blue = theme.category.blue;
  const purple = theme.category.purple;
  const edge = { fill: "none", stroke: fg, strokeWidth: 1 } as const;
  const rowY = (i: number) => 40 + i * 64;
  const heading = (x: number, label: string) => (
    <text x={x} y={16} textAnchor="middle" fontSize={11} fontWeight={600} fill={theme.text.secondary}>
      {label}
    </text>
  );
  return (
    <svg viewBox="0 0 736 410" width="100%" style={{ maxWidth: 820, display: "block" }}>
      <defs>
        <marker id="cfg-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill={fg} />
        </marker>
      </defs>
      {heading(75, "1. Config + world size")}
      {heading(271, "2. Ordered harness list")}
      {heading(467, "3. graph.py Node")}
      {heading(657, "4. Builder places it")}

      <NodeBox x={0} y={40} w={150} title="BlazeConfig" sub="dims, mesh_shape 4x2" />
      <NodeBox x={0} y={120} w={150} title="world size = 64" sub="MPI process count" />
      <NodeBox x={0} y={200} w={150} title="layout by count" sub="4, 16, 63, 64, 80" />
      <path {...edge} d="M75 96 V118 M75 176 V198" markerEnd="url(#cfg-arrow)" />
      <path {...edge} d="M150 228 H166 V68 M166 228 V324 M166 68 H184 M166 132 H184 M166 196 H184 M166 260 H184 M166 324 H184" />
      <text x={75} y={290} textAnchor="middle" fontSize={11} fill={theme.text.secondary}>
        get_pipeline_harnesses
      </text>
      <text x={75} y={306} textAnchor="middle" fontSize={11} fill={theme.text.tertiary}>
        position i = layer
      </text>

      {CONFIG_ROWS.map((r, i) => (
        <g key={r.node}>
          <NodeBox x={186} y={rowY(i)} w={170} title={r.harness} sub={r.loads} color={blue} stacked={r.stacked} />
          <path {...edge} d={`M${r.stacked ? 360 : 356} ${rowY(i) + 28} H390`} markerEnd="url(#cfg-arrow)" />
          <NodeBox x={392} y={rowY(i)} w={150} title={r.node} sub="shape 4x2, core pool" color={purple} />
          <path {...edge} d={`M542 ${rowY(i) + 28} H576`} markerEnd="url(#cfg-arrow)" />
        </g>
      ))}

      <rect x={578} y={40} width={158} height={312} rx={6} fill={theme.bg.elevated} stroke={fg} strokeWidth={1.5} />
      {[
        ["build_topology", 12, theme.text.primary, 150],
        ["resolve_graph_layout", 11, theme.text.secondary, 170],
        ["rank → submesh", 11, theme.text.secondary, 200],
        ["submesh → s{i}", 11, theme.text.secondary, 218],
        ["= my_stage_idx", 11, theme.text.secondary, 236],
        ["+ entry/exit socket cores", 11, theme.text.secondary, 266],
      ].map(([t, size, fill, y]) => (
        <text key={t as string} x={657} y={y as number} textAnchor="middle" fontSize={size as number} fill={fill as string}>
          {t}
        </text>
      ))}

      <path {...edge} strokeDasharray="4 3" d="M657 352 V380 H271 V362" markerEnd="url(#cfg-arrow)" />
      <text x={464} y={400} textAnchor="middle" fontSize={11} fill={theme.text.secondary}>
        on that rank, GenericStage takes harnesses[my_stage_idx]: weights, socket sizes, kernels
      </text>
    </svg>
  );
}

const CONFIG_TO_GRAPH_FIELDS: string[][] = [
  ["Stage order", "Position in the harness list", "Node name `s{i}` and the chain of edges", "Ring order; stage 0 holds H2D and D2H"],
  ["Mesh shape", "`harness.mesh_shape()`", "`Node.shape`", "Tiling or carving the submeshes"],
  ["Socket core pool", "`harness.pipeline_core_coords()` (None for every Kimi harness)", "`Node.pipeline_core_coords`", "Slot capacity per chip"],
  ["Host hop page", "`sum(harness.output_sizes_bytes())`", "`Node.host_hop_page_size_bytes`", "Only edges with `transport=\"host\"`"],
  ["Stage object", "`GenericStage(harnesses=<whole list>)`", "`Node.factory`", "Opaque; called once, on the rank that owns the stage"],
  [
    "Stays out of graph.py",
    "Layer id, weights, I/O sizes and cores, kernels, L1",
    "Nothing",
    "Read by `GenericStage` after placement is decided",
  ],
];

function ConfigToGraphPanel() {
  return (
    <Stack gap={12}>
      <Text weight="semibold" size="small">
        How the config becomes the graph, in simple terms
      </Text>
      <Bullets
        items={[
          "The config describes the model (dimensions, a 4x2 stage shape). It has no stage list and no ranks.",
          "The process count picks a hard-coded layout in `get_pipeline_harnesses`. Each harness is one stage, and its position in the list is its layer.",
          "`build_pipeline` pads and checks the list, then turns harness i into node `s{i}`. It copies the shape and core pool, gives every node the same factory, and chains them into a ring.",
          "`graph.py` never sees layers or kernels. It only decides which rank and submesh become `s{i}`, and which cores carry the sockets.",
          "On that rank the factory builds a `GenericStage`, which looks up `harnesses[my_stage_idx]` and loads the weights, sizes the sockets and compiles the kernels. The stage index is the only link between the two halves.",
        ]}
      />
      <ConfigToGraphDiagram />
      <ReqTable headers={["What", "From the config or harness", "In graph.py", "Builder uses it for"]} rows={CONFIG_TO_GRAPH_FIELDS} />
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// What a stage is: four objects
// ---------------------------------------------------------------------------

function StageObjectsDiagram() {
  const theme = useHostTheme();
  const fg = theme.stroke.primary;
  const edge = { fill: "none", stroke: fg, strokeWidth: 1 } as const;
  const layers: Array<{ title: string; sub: string; where: string; color: string; who: string }> = [
    { title: "Node  (graph.py)", sub: "shape, factory, name, group, core pool, host-hop pages", where: "every rank, all nodes", color: theme.category.purple, who: "pipeline_builder" },
    { title: "StageKind  (GenericStage, router, DFlash kinds)", sub: "factory(my_submesh) builds it; gets StageContext(my_stage_idx, …)", where: "this rank only", color: theme.category.green, who: "blaze stage" },
    { title: "Harness  (TestHarness subclass)", sub: "one layer: weights, socket sizes and cores, create_program", where: "one per stage or image", color: theme.category.blue, who: "model code" },
    { title: "Fused op  (registered by name)", sub: "BlazeCompiler.compile(\"dsv3_kimi2_6_sparse_layer_stage\", …)", where: "one kernel per RISC", color: theme.category.orange, who: "blaze ops" },
  ];
  const arrows = ["factory(submesh)", "picks its harness", "create_program"];
  const y = (i: number) => 14 + i * 84;
  return (
    <svg viewBox="0 0 760 350" width="100%" style={{ maxWidth: 820, display: "block" }}>
      <defs>
        <marker id="so-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill={fg} />
        </marker>
      </defs>
      {layers.map((l, i) => (
        <g key={l.title}>
          <text x={4} y={y(i) + 32} fontSize={11} fontWeight={600} fill={theme.text.secondary}>
            {l.who}
          </text>
          <NodeBox x={110} y={y(i)} w={430} title={l.title} sub={l.sub} color={l.color} />
          <text x={556} y={y(i) + 32} fontSize={11} fill={theme.text.tertiary}>
            {l.where}
          </text>
          {i < arrows.length && (
            <g>
              <path {...edge} d={`M325 ${y(i) + 56} V${y(i + 1) - 2}`} markerEnd="url(#so-arrow)" />
              <text x={333} y={y(i) + 72} fontSize={11} fill={theme.text.secondary}>
                {arrows[i]}
              </text>
            </g>
          )}
        </g>
      ))}
      <path d="M100 14 H104 V70 H100" fill="none" stroke={theme.category.purple} strokeWidth={1.5} />
      <text x={110} y={346} fontSize={11} fill={theme.text.tertiary}>
        graph.py only sees the top box. The stage index my_stage_idx is the one link down.
      </text>
    </svg>
  );
}

const NODE_FIELDS: string[][] = [
  ["`shape`", "`harness.mesh_shape()`, e.g. 4x2", "Tiling or carving the submeshes; the resolver's chip-count check", "acts"],
  ["`factory`", "A closure the model or stage code writes", "Collected into `PipelineLayout.stage_factories`; called once, on this rank", "acts"],
  ["`pipeline_core_coords`", "`harness.pipeline_core_coords()`; None for every Kimi harness", "`pipeline_core_pool()`: the socket cores per chip. None takes the graph-role default (2 or 4 at the ring head) or a default by shape", "acts"],
  ["`host_hop_page_size_bytes`", "`sum(harness.output_sizes_bytes())`", "Page size of a `transport=\"host\"` edge leaving this node (`host_edge_planner.py`). Required on ring host edges", "acts"],
  ["`secondary_host_hop_page_size_bytes`", "The harness's second output stream, or None", "The same, for a host hop's second stream", "acts"],
  ["`name`", "\"s{i}\", \"m0\", \"S0\" …", "`stage_labels` for logs and visualization only", "label"],
  ["`group`", "Empty, unless a wide stage is several narrow nodes", "`stage_groups`: visualization merges the slices into one stage; runtime ignores it", "label"],
];

const FACTORY_PATHS: string[][] = [
  ["Regular ring", "`blaze/stages/stage.py`, `build_pipeline` → `_make_node`", "The same `GenericStage(harnesses=<whole list>)` on every node; the stage picks `harnesses[my_stage_idx]`"],
  ["Reload", "`blaze/reload/pipeline.py`, `ReloadPipeline._ring_graph` → `make(i)`", "`GenericStage(run_persistently=False, reload_config=…)` carrying stage i's images"],
  ["DFlash", "`blaze/models/kimi_k2_6/entry.py`, `build_kimi_dflash_router_pipeline_graph`", "A different closure per node: `BlockDiffusionRouterStageKind`, target `GenericStage`, `DFlashPreDecodeStageKind`, `DFlashDecoderStageKind` ×6, `DFlashPostDecodeStageKind`, `PassthroughStage`"],
];

const RELOAD_OBJECTS: string[][] = [
  ["m0", "17", "s0.0-embed, s0.1-sparse3, s0.2-sparse7 … s0.15-sparse59, s0.16-head", "INGRESS, WRAP, TERMINAL"],
  ["m1", "16", "s1.0-dense0, s1.1-sparse4, s1.2-sparse8 … s1.15-sparse60", "LAYER"],
  ["m2", "16", "s2.0-sparse1, s2.1-sparse5 … s2.14-sparse57, s2.15-bypass", "LAYER"],
  ["m3", "16", "s3.0-sparse2, s3.1-sparse6 … s3.14-sparse58, s3.15-bypass", "LAYER"],
];

const STRING_LOOKUP: string[][] = [
  ["Fused op", "yes", "`register_fused_op(name, config)` and `get_fused_op_config(name)` in `blaze/blaze_op.py`; `blaze/registry.py` does the same for `OpSpec`"],
  ["Harness", "model-local only", "`make_harness_for` in `kimi_k2_6/reload.py` switches on \"embed\", \"bypass\", \"head\", \"sparse\", \"dense\""],
  ["StageKind / node factory", "no", "No registry. `Node.name` and stage labels are display strings; nothing dispatches on them"],
];

function Mono({ children }: { children: string }) {
  const theme = useHostTheme();
  return (
    <pre
      style={{
        margin: 0,
        padding: "10px 12px",
        background: theme.fill.quaternary,
        border: `1px solid ${theme.stroke.secondary}`,
        borderRadius: 6,
        fontSize: 12,
        lineHeight: 1.5,
        overflowX: "auto",
        color: theme.text.primary,
      }}
    >
      {children}
    </pre>
  );
}

function StageObjectsSection() {
  return (
    <Stack gap={16}>
      <Text tone="secondary">
        A stage is four plain Python objects, one per layer of code, linked only by the stage index. Each can be built and used on
        its own: a harness runs standalone in its fused-op test or `scripts/run_single_stage_perf.py`, and any code can build a
        `PipelineGraph` from its own nodes and call `build_topology`.
      </Text>
      <StageObjectsDiagram />

      <Stack gap={8}>
        <H3>What a Node holds</H3>
        <Text size="small" tone="secondary">
          Seven fields. Five change what the builder does; `name` and `group` are labels. No layer id, weights, kernels or L1 size.
        </Text>
        <Table
          headers={["Field", "Set from", "Builder uses it for", "Role"]}
          rows={NODE_FIELDS.map((r) => r.map((c, i) => <Text key={i} size="small">{c}</Text>))}
          rowTone={NODE_FIELDS.map((r) => (r[3] === "acts" ? undefined : "neutral"))}
        />
        <Text size="small" tone="tertiary">
          {"Edges are `Edge(src, dst, is_loopback=False, transport=\"fabric\" | \"host\")`. A host edge is left out of fabric resolution and sized from its source node's host-hop page."}
        </Text>
      </Stack>

      <Grid columns="minmax(0, 1fr) minmax(0, 1fr)" gap={20} align="start">
        <Stack gap={8}>
          <H3>Who writes the factory</H3>
          <Table
            headers={["Path", "Graph built in", "Each node's factory returns"]}
            rows={FACTORY_PATHS.map((r) => r.map((c, i) => <Text key={i} size="small">{c}</Text>))}
          />
        </Stack>
        <Stack gap={8}>
          <H3>Where factories are stored and called</H3>
          <Bullets
            items={[
              "Stored on `Node.factory`, typed `factory(submesh) -> stage instance`.",
              "`build_topology` collects `{stage_idx: node.factory}` into `PipelineLayout.stage_factories`.",
              "`PipelineLayout.build_pipeline` calls `stage_factories[my_stage_idx](my_submesh)` on this rank only, then `set_host_hops` if it has host edges. Idle ranks call nothing.",
              "Forks do the same in `fork.py` through `stage_factory=graph.nodes[my_node].factory`.",
              "`Pipeline.setup_and_run` then drives the stage: `prepare_for_pipeline_block`, `create_pipeline_block` (sockets), `setup` (compiles the kernels), `start_pipeline`, `launch_compute`.",
            ]}
          />
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H3>One stage, all four objects: reload stage m1</H3>
        <Mono>{`Node "s1"   shape=4x2, name="m1", factory=make(1)                 graph.py, every rank
  └─ factory(submesh) ─> GenericStage(run_persistently=False,     this rank only
                           reload_config=ReloadStageConfig(m1's 16 images))
        └─ ReloadStage "m1": images = [ReloadImage × 16]           blaze/reload/plan.py
              ReloadImage(slot=1, name="s1.1-sparse4",
                          harness=DeepseekKimiMoEDecoderStageHarness(layer_idx=4),
                          weights=StagedWeights.pinned("s1.1-sparse4"), iterations=1)
                    └─ create_program ─> BlazeCompiler.compile("dsv3_kimi2_6_sparse_layer_stage")`}</Mono>
        <Text size="small" tone="secondary">
          {"`FULL_SEQUENCE` is 65 `(kind, layer_id)` tuples. `RingPlan.deal(sequence, 4)` turns each into a `RingSlot(stage, slot, item, step, from_loopback, writes_exit_d2d)`, step i on stage `i % 4`; `make_harness_for` maps each item to a harness. The deal, run on CPU:"}
        </Text>
        <Table
          headers={["Stage", "Images", "Image names (slot order)", "Roles"]}
          rows={RELOAD_OBJECTS.map((r) => r.map((c, i) => <Text key={i} size="small">{c}</Text>))}
        />
      </Stack>

      <Stack gap={8}>
        <H3>Can you look one up by string?</H3>
        <Table
          headers={["Object", "By string?", "How"]}
          rows={STRING_LOOKUP.map((r) => r.map((c, i) => <Text key={i} size="small">{c}</Text>))}
          rowTone={["success", "warning", "danger"]}
        />
        <Callout tone="info" title="For the MGD design">
          An MGD stage label such as &quot;MoE12&quot; cannot become a stage today. That needs a new per-model lookup from a label or
          kind to a harness or factory, for example by lifting `make_harness_for`&apos;s switch into a shared registry.
        </Callout>
      </Stack>
    </Stack>
  );
}

const RELOAD_STAGE_HUES = ["blue", "purple", "cyan", "orange"] as const;

function ReloadSequenceStrip() {
  const theme = useHostTheme();
  const seq: ReloadItem[] = [...RELOAD_BODY, ["head", null]];
  const n = seq.length;
  const cw = 704 / n;
  const ticks: Array<[number, string, number]> = [
    [0, "E", 14],
    [2, "M1", 14],
    [31, "M30", 14],
    [61, "M60", 14],
    [64, "H", 14],
    [1, "D0", 64],
    [62.5, "B B", 64],
  ];
  return (
    <Stack gap={8}>
      <Row gap={16} wrap>
        {RELOAD_STAGE_HUES.map((hue, s) => (
          <LegendItem key={hue} color={theme.category[hue]} label={`dealt to m${s}`} />
        ))}
      </Row>
      <svg viewBox="0 0 736 74" width="100%" style={{ maxWidth: 820, display: "block" }}>
        {seq.map((item, step) => {
          const stage = step === n - 1 ? 0 : step % RELOAD_STAGES;
          const x = 16 + step * cw;
          return (
            <g key={step}>
              <rect
                x={x}
                y={22}
                width={cw - 1.5}
                height={26}
                rx={2}
                fill={tint(theme.category[RELOAD_STAGE_HUES[stage]], item[0] === "sparse" ? 30 : 60)}
                stroke={theme.stroke.secondary}
                strokeWidth={0.5}
              >
                <title>{`step ${step}: ${reloadLabel(item)} -> m${stage}, slot ${step === n - 1 ? RELOAD_LAPS : Math.floor(step / RELOAD_STAGES)}`}</title>
              </rect>
            </g>
          );
        })}
        {ticks.map(([step, label, y]) => (
          <text key={label} x={16 + step * cw + cw / 2} y={y} textAnchor="middle" fontSize={10} fill={theme.text.secondary}>
            {label}
          </text>
        ))}
      </svg>
    </Stack>
  );
}

const DFLASH_GRAPH_FIELDS: string[][] = [
  ["nodes", "`router` (S0), `target0` to `target62` (S1 to S63), `draft0` to `draft8` (S64 to S72); every shape `SUBMESH_SHAPE_4X2`"],
  ["edges", "`router -> target0 -> ... -> target62 -> router` (`is_loopback=True`); `router -> draft0 -> ... -> draft8 -> router` (`is_loopback=True, transport=\"host\"`)"],
  ["factories", "router: `BlockDiffusionRouterStageKind`; targets: `GenericStage` over `[router placeholder, 63 Kimi harnesses]`; drafts: pre-decode, 6 decoders, post-decode, `PassthroughStage` relay"],
  ["extras", "`router_hidden_dim` = Kimi hidden size; `io_socket_descriptor_prefix`"],
];

const RELOAD_OPEN_ARGS: string[][] = [
  ["stages", "4 `ReloadStage` specs, m0 to m3: reload images (one per slot, e.g. `s1.3-sparse16`, pinned staged weights), `max_weights`, `from_loopback` (m0 only), `d2h_output_page_size` (m0 only)"],
  ["harnesses", "4 lists in slot order: m0 has 17 (embed, 15 MoE, head), m1 to m3 have 16"],
  ["submesh_shape", "`MeshShape(4, 2)`"],
  ["loopback", "`True`: s3 returns to s0"],
  ["laps", "16, from `RingPlan.deal`"],
  ["mode", "`ReloadMode.FIRMWARE_PERSISTENT`"],
];

function ReloadGraphDiagram() {
  const theme = useHostTheme();
  const fg = theme.stroke.primary;
  const edge = { fill: "none", stroke: fg, strokeWidth: 1 } as const;
  const slots = [17, 16, 16, 16];
  return (
    <svg viewBox="0 0 736 150" width="100%" style={{ maxWidth: 820, display: "block" }}>
      <defs>
        <marker id="rgraph-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill={fg} />
        </marker>
      </defs>
      {slots.map((k, i) => {
        const x = 40 + i * 172;
        return (
          <g key={i}>
            <NodeBox
              x={x}
              y={30}
              w={140}
              title={`s${i} "m${i}"`}
              sub={`${k} reload images`}
              color={theme.category[RELOAD_STAGE_HUES[i]]}
            />
            {i < 3 && <path {...edge} d={`M${x + 140} 58 H${x + 170}`} markerEnd="url(#rgraph-arrow)" />}
          </g>
        );
      })}
      <path {...edge} d="M626 86 V112 H110 V88" markerEnd="url(#rgraph-arrow)" />
      <text x={368} y={130} textAnchor="middle" fontSize={11} fill={theme.text.secondary}>
        is_loopback=True, s3 to s0
      </text>
    </svg>
  );
}

function DFlashForkPanel() {
  return (
    <Stack gap={12}>
      <Text weight="semibold" size="small">
        What entry.py passes in: a hand-written graph
      </Text>
      <Text size="small" tone="secondary">
        {"Unlike the other two paths, `entry.py` builds the graph itself and passes it to `build_model_pipeline_from_graph(graph, mesh_device, submesh_shape=4x2, multi_mesh=True, placement=...)`. Every stage is 4x2. The drawing is logical: which slice each stage lands on is decided by `resolve_graph_layout` at runtime."}
      </Text>
      <DFlashForkDiagram />
      <ReqTable headers={["Graph field", "Value"]} rows={DFLASH_GRAPH_FIELDS} />
      <ReqTable headers={["Whole graph", "Requirement"]} rows={FORK_WIDE_REQS} />
      <ReqTable headers={["Stage", "Runs", "Edges", "Pipeline cores", "Worker L1 (B)", "Other needs"]} rows={FORK_STAGE_REQS} />
      <TierCCarryTable kind="dflash" />
    </Stack>
  );
}

function ReloadRingPanel() {
  return (
    <Stack gap={12}>
      <Stack gap={12}>
        <Text weight="semibold" size="small">
          What entry.py passes in: a sequence, not a graph
        </Text>
        <Text size="small" tone="secondary">
          {"`entry.build_pipeline` hands `build_reload_pipeline` only the mesh and options. It builds the whole model as `FULL_SEQUENCE`, 65 steps, and `RingPlan.deal` sends step i to stage i % 4. Hover a cell for its stage and slot."}
        </Text>
        <ReloadSequenceStrip />
        <ReqTable headers={["runner.open argument", "Value"]} rows={RELOAD_OPEN_ARGS} />
      </Stack>

      <Stack gap={12}>
        <Text weight="semibold" size="small">
          The graph ReloadPipeline.open builds from it
        </Text>
        <Text size="small" tone="secondary">
          {"`_ring_graph` makes 4 nodes with a chain and one loopback, and `co_locate_ring_host_io=False`. Each node's `GenericStage` carries its stage's layers through `reload_config`. Topology and the resolver see only these 4 nodes; after the resolver gives a rank its `my_stage_idx`, that rank captures only that stage's images."}
        </Text>
        <ReloadGraphDiagram />
      </Stack>

      <Stack gap={12}>
        <Text weight="semibold" size="small">
          Where it lands on the galaxy, and what each stage needs
        </Text>
        <Text size="small" tone="secondary">
          {"Every stage is 4x2. One mesh, so `build_topology` takes the single-mesh path: `create_submeshes(4x2)` cuts four tiles and the resolver assigns s0 to s3 to them. The drawing follows `single_bh_galaxy_4x8_torus_xy`, where stage j owns columns 2j and 2j+1; that is the resolver's first fit, not a pinned placement."}
        </Text>
        <ReloadRingDiagram />
        <Text size="small" tone="secondary">
          {"What each stage reloads: `RingPlan.deal` puts step i on stage i % 4, slot i // 4. A token walks the ring 16 times, one slot per stage per lap, then runs the LM head in m0's extra slot. Hover a cell for its model step."}
        </Text>
        <ReloadSlotGrid />
        <ReqTable headers={["Whole ring", "Requirement"]} rows={RING_WIDE_REQS} />
        <ReqTable
          headers={["Stage", "Columns", "Reload table slots", "Weight regions per core", "Fabric edges", "Host I/O"]}
          rows={RING_STAGE_REQS}
        />
        <Text size="small" tone="tertiary">
          {"Weight regions per core sum `REGIONS_PER_KIND` over a stage's slots: dense 64, MoE 32, head 8, embed and bypass 0."}
        </Text>
      </Stack>

      <TierCCarryTable kind="reload" />
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Gaps
// ---------------------------------------------------------------------------

type Severity = "blocker" | "mismatch" | "verify" | "note";
type Category = "Topology" | "Graph shapes" | "Capacity" | "Ranks and launch" | "Search";

type Gap = {
  id: string;
  cat: Category;
  sev: Severity;
  title: string;
  blaze: string;
  design: string;
  fix: string;
  oq?: string;
};

const GAPS: Gap[] = [
  {
    id: "G1",
    cat: "Ranks and launch",
    sev: "blocker",
    title: "Per-rank L1 is chosen before the stage is known",
    blaze:
      "`cli.get_worker_l1_size(model, num_procs, rank)` runs inside `open_mesh_device`, before `build_topology`. Kimi gives ranks >= 62 the verifier-tail size (1461504 B vs 1441568 B); Kimi K3 picks per rank by layer; the DFlash fork has a rank table; `BLAZE_WORKER_L1_SIZE_TABLE` is keyed by rank. All of it assumes stage i runs on rank i.",
    design:
      "Section 5.2: the mapper's carve is a witness, blaze carves again at runtime and may land elsewhere, and `TT_PIPELINE_STAGE` is written but not consumed. Nothing ties a stage to a rank.",
    fix:
      "Make the stage-to-rank map a real output. Either blaze keys L1 by stage after topology (open with the largest budget, or reopen), or the mapper pins stage to rank and blaze consumes `TT_PIPELINE_STAGE`. Until one lands, the mapper should keep the LM head and tail on the ranks the L1 table expects.",
  },
  {
    id: "G2",
    cat: "Topology",
    sev: "mismatch",
    title: "A fabric hop needs two or more channels, but R records one",
    blaze:
      "Each D2D endpoint core opens fabric connections on link indices 0 and 1, one per RISC (`for link_idx in range(2)` in `d2d_exchange.py`), and stripes every page over `NUM_FWD_LINKS = 2` (`pipeline_sockets/utils.py`). The reload ring states the requirement outright: its torus descriptor asks for `channels count: 2` on every edge, wraps included, the galaxy has the links for it, and `reload.md` warns that lowering the count hides a one-link edge the two-channel transport then runs on.",
    design:
      "C2 records one exact channel per crossing in `R`, and the demand check compares crossings with the connection's channel count. Nothing checks that the exit chip has a second routing plane toward the next mesh.",
    fix:
      "Make channels per hop a requirement of the edge: a crossing consumes at least two channels out of the same exit chip toward the next position, and `R` records all of them. Let a graph ask for more where the cabling allows it, as the reload ring does on every edge.",
  },
  {
    id: "G3",
    cat: "Topology",
    sev: "mismatch",
    title: "A link counts only if its direction has an active routing plane",
    blaze:
      "The C++ resolver's `pipeline_get_forwarding_direction` treats a direction as unreachable when `get_active_fabric_eth_routing_planes_in_direction` is empty. The carve planner and the connectivity dump use the nanobind `get_forwarding_direction` without that filter, so blaze's own planner and resolver can disagree.",
    design:
      "C1 links positions over PSD cables and routed intra-mesh links. Open question 1 mentions links with no active routing planes but leaves them open.",
    fix:
      "Close question 1 with the resolver's rule: a cable joins two positions only if the control plane will give that direction at least one routing plane (two, if G2 holds).",
    oq: "1",
  },
  {
    id: "G4",
    cat: "Topology",
    sev: "mismatch",
    title: "Torus wrap links: a loopback between stages, and a ring inside each stage",
    blaze:
      "`discover_slice_adjacency` keeps every confirmed neighbour, including wrap links of a RING dimension. `_as_rectangle` only builds a stage from slices with contiguous offsets, so a stage's rectangle never wraps. The reload ring uses the column wrap (col 7 to col 0) as its m3 to m0 loopback, and every MLA stage needs the row wrap inside its own rectangle: `sdpa_reduce` rings over `(row + offset) % 4`, and without the wrap blaze rejects the build (`_build_router_rings`). SC20 and the 4x8 torus are both RING on dim 0 for this reason.",
    design: "Section 2.1 rules out rotation. Nothing says whether wrap links join positions in C1, or that a position needs a ring along an axis inside itself.",
    fix: "State in 4.1 that positions never wrap but wrap links join positions like any other link. Add an intra-position topology requirement: a stage may require its rectangle to span a whole RING axis, so its own CCL ring closes.",
  },
  {
    id: "G5",
    cat: "Topology",
    sev: "mismatch",
    title: "A declared host edge is a third kind of edge",
    blaze:
      "`Edge(transport=\"host\")` is legal under `strict`; it is dropped from fabric resolution and never uses a cable. Kimi's production DFlash router declares `draft8 -> router` as a host loopback, and the GPT-OSS fork does the same. The carve then needs only one router neighbour for that branch (router degree is 2 x fabric returns + host returns).",
    design:
      "The first version is STRICT only, and RELAXED (later) means \"may go over host\". A declared host edge is neither: it must never use fabric, and it consumes no channel.",
    fix:
      "Give it a spelling (for example a HOST policy, or `transport: HOST`) that Tier B and C drop from the fabric graph as a hard rule, or list the DFlash router as out of scope. Writing it as STRICT asks Tier C for a fabric return blaze never builds, and changes the router's degree.",
  },
  {
    id: "G6",
    cat: "Topology",
    sev: "mismatch",
    title: "Host edges change how the resolver is called",
    blaze:
      "With any host edge, `_layout_from_resolution` takes stage order from the logical graph, `_resolve_chains_separately` resolves each fabric chain on its own pool of submeshes, the hop's endpoints go on the least-used chip (`_local_endpoint`), and H2D and D2H are not co-located.",
    design: "Per-chain resolution appears only with RELAXED edges (P11, section 3.4).",
    fix: "If G5 is accepted, the same per-chain loop is needed for declared host edges in the first version.",
  },
  {
    id: "G7",
    cat: "Graph shapes",
    sev: "mismatch",
    title: "Blaze builds fewer fork shapes than Tier A accepts",
    blaze:
      "`fork.py` asserts exactly two branches. The router is the one in-degree-0 node and a single host slice; no edge out of the router may be a host edge; a fused router with host returns needs all four router roles on one rank. Fan-in and other fan-outs are not supported (`builder.md`, Limits).",
    design: "Tier A checks ids, edges, loopback marks and one host I/O stage, so any fork with returns to the router passes.",
    fix: "Add these as Tier A rules, or as a \"not buildable by blaze\" diagnostic, so the mapper does not search for forks blaze cannot lay out.",
  },
  {
    id: "G8",
    cat: "Graph shapes",
    sev: "mismatch",
    title: "Stage 0 can never be co-owned, on any carve path",
    blaze:
      "`_layout_from_resolution` raises `NotImplementedError` when stage 0's submesh spans several ranks: on per-rank tiling and the single mesh too, not just in `plan_ring_carve`. Host I/O, the loopback entry and D2H must sit on one rank.",
    design: "Open question 14 treats \"stage 0 is one slice\" as a planner rule to keep \"for as long as blaze keeps the rule\".",
    fix: "Close question 14: restrict stage 0's positions in C1 to single slices. It is a runtime wiring limit, not a planner heuristic.",
    oq: "14",
  },
  {
    id: "G9",
    cat: "Graph shapes",
    sev: "mismatch",
    title: "Without a joint stage, a mesh holds one shape",
    blaze:
      "With no joint stage, the generic path never plans a carve. Each rank tiles its own mesh with one shape: its owned bounding box if that is a stage shape, else the big shape on mesh 0 and the small one elsewhere (FIXME tt-blaze#1455). Three or more distinct shapes are not handled.",
    design: "The position graph takes positions per mesh from every stage's shape, so it can mix shapes on a mesh or place the big shape on mesh 1.",
    fix: "When no stage is joint, limit C1 to one shape per mesh, equal to what that mesh's ranks tile, or have blaze plan the carve in that case too.",
  },
  {
    id: "G10",
    cat: "Capacity",
    sev: "mismatch",
    title: "Core-slot defaults depend on how blaze builds, not just on graph shape",
    blaze:
      "`_pipeline_core_defaults`: a ring that keeps host I/O beside its fabric links gives stage 0 four slots. Reload rings pass `co_locate_ring_host_io=False`, so stage 0 falls back to one slot on 8+ chips. An open chain gives head and tail two slots each; a fork router gets 2 + 2 x branches; a host-return tail on a small submesh gets two. Nodes without a default take the C++ default of the submesh they land on. The +2 host slots count only when stage 0 has a fabric edge.",
    design: "The 3.1 table lists only one slot per chip on 8+ chips, two on smaller, plus two for stage 0. Open question 7 has C2 compute blaze's defaults \"from the graph\".",
    fix: "Copy blaze's role table into 3.1, and carry the co-location flag (reload or generic) into Tier C, or add an optional per-stage core count to `PipelineStage`.",
    oq: "7, 17",
  },
  {
    id: "G11",
    cat: "Capacity",
    sev: "note",
    title: "Pipeline cores share a column with op grids",
    blaze:
      "Slots map to physical cores starting at (12, 8), then (12, 7) and down. Kimi's 128-core LM-head matmul uses (12, 0) to (12, 7) and leaves (12, 8) and (12, 9) free, so that stage can give up exactly one pipeline core per chip. A harness can override `pipeline_core_coords()`; no Kimi harness does today.",
    design: "Open question 7: add a core count only if a model needs one.",
    fix: "The ceiling comes from op core grids the mapper cannot see. Record a per-stage maximum in the descriptor as soon as any harness overrides its pool.",
    oq: "7",
  },
  {
    id: "G12",
    cat: "Ranks and launch",
    sev: "mismatch",
    title: "The process count defines the stage list, and the MGD is a second copy",
    blaze:
      "`build_pipeline` sizes the ring from `distributed_context_get_size()`. Kimi accepts 4, 16, 63, 64 and 80; `from_layout_pipeline` accepts 4, 8, 16, 35, 64, 72, 80, 96 and 160. Passthrough padding fills every remaining rank. `double_torus=False` swaps the 4-proc decoders for passthroughs.",
    design: "`pipeline_descriptors` is written by hand next to the mesh; the builder cannot load it (TODO in 1.2). Tier C proves the descriptor fits, not that blaze's graph fits.",
    fix: "Add a check in blaze, after building `PipelineGraph`, that its stage count, shapes, edges and loopback match `pipeline_descriptors` (exported from the harnesses, padding included). Fail before carving.",
  },
  {
    id: "G13",
    cat: "Ranks and launch",
    sev: "note",
    title: "Host-hop rules to carry into RELAXED (P11, P14)",
    blaze:
      "Host hops need persistent mode and a ring that loops back. A DFlash ring cannot return to stage 0 over host. A host loopback and every fork host hop reject a second stream. MPI tags follow the order host edges are declared. Flexible ring planning needs `host_hop_page_size_bytes` on every node and breaks ties by page bytes, then by cutting the loopback.",
    design: "Section 3.4 and open question 16 cover the tie-breaks only.",
    fix: "List these as RELAXED preconditions in 1.4 so a RELAXED edge never lands where blaze cannot run a host hop.",
    oq: "16",
  },
  {
    id: "G14",
    cat: "Ranks and launch",
    sev: "note",
    title: "Reload rings build a different graph",
    blaze:
      "`ReloadPipeline` builds a ring shorter than the model (stages reload weight images in laps), with `co_locate_ring_host_io=False`, a fixed `submesh_shape` (per-rank tiling), strict placement only, and an explicit `is_loopback`.",
    design: "Not mentioned.",
    fix: "A reload descriptor describes ring stages, not layers, and needs the stage-0 core count from G10, two channels on every edge including the wrap (G2), and a RING row axis inside each stage (G4). The per-stage requirements are in the Kimi graphs section above.",
  },
  {
    id: "G15",
    cat: "Topology",
    sev: "note",
    title: "Sockets accept more than the resolver; keep C1 at the resolver's rule",
    blaze:
      "`_assert_point_to_point_across_meshes` lets a D2D hop route multi-hop inside one mesh, but asserts a direct link across meshes. The resolver accepts only direct links everywhere.",
    design: "C1 uses direct links, which matches the resolver.",
    fix: "Note in 4.1 that loosening C1 to multi-hop links inside a mesh would accept layouts the resolver rejects.",
  },
  {
    id: "G16",
    cat: "Search",
    sev: "note",
    title: "Blaze's budgets and the resolver's missing one",
    blaze:
      "Ring carve: 200,000 expansions or 64 cycles. Fork carve: 2,000 attempts. Host-edge cover: 200,000 expansions and 16 covers per chain count. Fork walks: 4,000 attempts. The C++ resolver memoizes failed link states but has no time budget.",
    design: "Open question 8 covers planner false rejects; open question 22 covers total time.",
    fix: "Add the resolver's missing budget to question 22: one C2 resolver call is unbounded.",
    oq: "8, 22",
  },
];

const SEV_TONE: Record<Severity, TableRowTone> = {
  blocker: "danger",
  mismatch: "warning",
  verify: "info",
  note: "neutral",
};

const CATEGORIES: Array<Category | "All"> = ["All", "Topology", "Graph shapes", "Capacity", "Ranks and launch", "Search"];

function GapDetail({ gap }: { gap: Gap }) {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">{gap.sev}</Pill>}>{`${gap.id}  ${gap.title}`}</CardHeader>
      <CardBody>
        <Stack gap={12}>
          <Stack gap={4}>
            <Text weight="semibold" size="small">
              What blaze does
            </Text>
            <Text size="small">{gap.blaze}</Text>
          </Stack>
          <Stack gap={4}>
            <Text weight="semibold" size="small">
              What the design says
            </Text>
            <Text size="small">{gap.design}</Text>
          </Stack>
          <Divider />
          <Stack gap={4}>
            <Text weight="semibold" size="small">
              Proposed change
            </Text>
            <Text size="small">{gap.fix}</Text>
          </Stack>
          {gap.oq ? (
            <Text size="small" tone="tertiary">
              Related open questions: {gap.oq}
            </Text>
          ) : null}
        </Stack>
      </CardBody>
    </Card>
  );
}

function GapExplorer() {
  const [cat, setCat] = useCanvasState<string>("gap-cat", "All");
  const [sel, setSel] = useCanvasState<string>("gap-sel", "G1");
  const shown = GAPS.filter((g) => cat === "All" || g.cat === cat);
  const gap = GAPS.find((g) => g.id === sel) ?? GAPS[0];
  return (
    <Stack gap={12}>
      <Row gap={6} wrap>
        {CATEGORIES.map((c) => (
          <Pill key={c} size="sm" active={c === cat} onClick={() => setCat(c)}>
            {c === "All" ? `All (${GAPS.length})` : `${c} (${GAPS.filter((g) => g.cat === c).length})`}
          </Pill>
        ))}
      </Row>
      <Grid columns="minmax(0, 1fr) minmax(0, 1fr)" gap={16} align="start">
        <Table
          headers={["", "Gap", "Kind", "Area"]}
          rowTone={shown.map((g) => SEV_TONE[g.sev])}
          rows={shown.map((g) => [
            <Pill key={g.id} size="sm" active={g.id === sel} onClick={() => setSel(g.id)}>
              {g.id}
            </Pill>,
            <Text key="t" size="small" weight={g.id === sel ? "semibold" : "normal"}>
              {g.title}
            </Text>,
            <Text key="s" size="small" tone="secondary">
              {g.sev}
            </Text>,
            <Text key="c" size="small" tone="tertiary">
              {g.cat}
            </Text>,
          ])}
        />
        <GapDetail gap={gap} />
      </Grid>
      <Text size="small" tone="tertiary">
        Kinds: blocker means the design as written lets blaze run the wrong program or fail at build. Mismatch means Tier C accepts or rejects differently from what blaze builds. Verify means a code path to confirm before deciding. Note means a fact the design should record.
      </Text>
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Doc fixes
// ---------------------------------------------------------------------------

const DOC_FIXES: string[][] = [
  ["Status line", "Both the doc and the canvas say nothing is implemented, but P1 to P3 (schema, Tier A, Tier B) are on this branch."],
  ["Canvas link", "The canvas's \"open the doc\" button points at `/data/rsong/tt-metal3/...`."],
  ["4.2 vs P5", "4.2 calls a failed pre-check `pipeline_layout`; P5 uses `pipeline_carve`."],
  ["2.3 and 3.2", "\"Enumerate carves until one passes\" contradicts the decision of one carve per seating (open question 24)."],
  ["`inter_mesh_cables`", "The canvas puts it in P6; the doc puts it in P5."],
  ["`PipelineMappingResult (P6)`", "Named in the canvas; no such type in the plan."],
  ["3.1 capacity row", "Leaves out blaze's role defaults (stage 0 with four slots, and the rest of G10)."],
  ["1.2", "Says older models still declare forks in `entry.py` as cleanup; the Kimi DFlash fork there carries a declared host edge (G5), which is a format gap, not a cleanup."],
];

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function BlazePipelineBuilderDeepDive() {
  const blockers = GAPS.filter((g) => g.sev === "blocker").length;
  const mismatches = GAPS.filter((g) => g.sev === "mismatch").length;
  const verify = GAPS.filter((g) => g.sev === "verify").length;
  return (
    <Stack gap={24} style={{ padding: 4 }}>
      <Stack gap={8}>
        <H1>How blaze builds a pipeline today, and what Tier C misses</H1>
        <Text tone="secondary">
          A trace of `blaze.stages.stage.build_pipeline` from the Kimi K2.6 entry (`blaze/models/kimi_k2_6/entry.py`) through `pipeline_builder` and the C++ resolver to the sockets, read against the MGD pipeline design (`docs/scratch/mgd-pipeline-graph-design.md`, tt-metal#63260). tt-blaze `main` at 56b651a975; tt-metal `rsong/mgd-pipeline-tier-c` at b6d09448224.
        </Text>
      </Stack>

      <Row gap={24} wrap>
        <Stat value={String(GAPS.length)} label="gaps against the design" />
        <Stat value={String(blockers)} label="blocker" tone="danger" />
        <Stat value={String(mismatches)} label="Tier C and blaze disagree" tone="warning" />
        <Stat value={String(verify)} label="to verify in code" tone="info" />
      </Row>

      <Callout tone="warning" title="The two findings that change the design">
        <Text size="small">
          First, blaze picks each rank's L1 size by MPI rank when it opens the mesh, before topology decides which stage runs there (G1). A witness placement that moves the LM head off rank 62 and up builds a stage with the wrong L1 budget, so the stage-to-rank map has to become a real output. Second, every fabric hop needs two or more channels from its exit chip (G2): the D2D socket stripes each page over two links, and the reload ring asks for two channels on every edge, wraps included. `R` records one channel per crossing, so P6's demand check has to count channels per hop.
        </Text>
      </Callout>

      <Stack gap={12}>
        <H2>The build, step by step</H2>
        <Text tone="secondary">
          Every rank runs this whole path collectively. The resolver's outputs are identical on every rank because the chip lists are allgathered and the control plane is global.
        </Text>
        <BuildWalkthrough />
      </Stack>

      <Stack gap={12}>
        <H2>What a stage is: four objects, and where its kernels come from</H2>
        <StageObjectsSection />
      </Stack>

      <Stack gap={12}>
        <H2>Three ways blaze turns meshes into submeshes</H2>
        <Text tone="secondary">
          {"`PipelineGraph._collect_submeshes` picks one of three paths. The generic harness path reaches the planned carve only when a stage is joint (`num_ranks() > 1`), so most multi-mesh rings, Kimi's included, use per-rank tiling."}
        </Text>
        <Table
          headers={["Path", "Taken when", "Carve", "Rules blaze enforces", "Tier C position graph"]}
          rows={CARVE_ROWS.map((r) => r.map((c, i) => <Text key={i} size="small">{c}</Text>))}
        />
      </Stack>

      <Grid columns="minmax(0, 3fr) minmax(0, 2fr)" gap={24} align="start">
        <Stack gap={12}>
          <H2>Kimi K2.6 layouts</H2>
          <Table
            headers={["Procs", "MGD", "Stages", "Carve path"]}
            rows={KIMI_ROWS.map((r) => r.map((c, i) => <Text key={i} size="small">{c}</Text>))}
          />
          <Text size="small" tone="tertiary">
            Every Kimi stage is 4x2 with one rank. The LM head's 128-core grid lives on 4x2, not on a 4x4 joint stage. `mtp_level` adds no stages. `get_pipeline_harnesses` also accepts 63, which `from_layout_pipeline` rejects. Source: `kimi_k2_6_blaze_config.py`, `entry.py`, `.github/workflows/kimi-decode-ladder.yaml`.
          </Text>
        </Stack>
        <Stack gap={12}>
          <H3>What the resolver checks per stage</H3>
          <Table
            headers={["Check", "Rule"]}
            rows={[
              ["chip count", "submesh chips equal the stage's rows x cols"],
              ["core slots", "edges + 2 (stage 0 with a fabric edge) <= slots x chips"],
              ["neighbours", "distinct neighbouring submeshes >= distinct logical neighbours"],
              ["links", "a direct link, on a direction with an active routing plane, to every placed neighbour"],
              ["lookahead", "unplaced stages can still reach enough free submeshes"],
            ].map((r) => r.map((c, i) => <Text key={i} size="small">{c}</Text>))}
          />
        </Stack>
      </Grid>

      <Stack gap={12}>
        <H2>Gaps in the Tier C design</H2>
        <Text tone="secondary">
          Each gap names what blaze does today, where the design is silent or different, and a proposed change. Gaps that already overlap an open question in section 6 say which one.
        </Text>
        <GapExplorer />
      </Stack>

      <Stack gap={12}>
        <H2>Doc fixes spotted along the way</H2>
        <Table
          headers={["Where", "Problem"]}
          rows={DOC_FIXES.map((r) => r.map((c, i) => <Text key={i} size="small">{c}</Text>))}
        />
      </Stack>

      <Text size="small" tone="tertiary">
        {"Sources: tt-blaze `blaze/models/cli.py`, `blaze/models/kimi_k2_6/`, `blaze/stages/stage.py`, `pipeline_builder/` (`graph.py`, `carve_planner.py`, `host_edge_planner.py`, `fork.py`, `submesh_partition.py`), `blaze/models/pipeline_sockets/`, `docs/architecture/pipeline/builder.md` and `sockets.md`; tt-metal `tt_metal/fabric/pipeline_builder.cpp`."}
      </Text>
    </Stack>
  );
}
