// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  DiffView,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Table,
  Text,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";
import type { DiffLineData } from "cursor/canvas";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function codeLines(src: string, added: boolean): DiffLineData[] {
  return src
    .replace(/^\n/, "")
    .replace(/\n\s*$/, "")
    .split("\n")
    .map((content, i) => ({ type: added ? ("added" as const) : ("unchanged" as const), content, lineNumber: i + 1 }));
}

function CodeCard(props: { path: string; src: string; added?: boolean; language?: string }) {
  return (
    <Card>
      <CardHeader
        trailing={
          <Text size="small" tone="tertiary">
            {props.added ? "proposed" : "reference"}
          </Text>
        }
      >
        {props.path}
      </CardHeader>
      <CardBody style={{ padding: 0 }}>
        <DiffView
          lines={codeLines(props.src, !!props.added)}
          path={props.path}
          language={props.language ?? "cpp"}
          showAccentStrip={!!props.added}
          coloredLineNumbers={!!props.added}
        />
      </CardBody>
    </Card>
  );
}

function fit(minPx: number) {
  return `repeat(auto-fit, minmax(min(100%, ${minPx}px), 1fr))`;
}

function scrollToId(id: string) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------------------
// Flow diagram: today vs proposed
// ---------------------------------------------------------------------------

type BoxKind = "plain" | "problem" | "new";

function Flow(props: { title: string; boxes: Array<{ text: string; sub: string; kind: BoxKind }> }) {
  const t = useHostTheme();
  const w = 200;
  const gap = 34;
  const h = 64;
  const total = props.boxes.length * w + (props.boxes.length - 1) * gap;
  const stroke = (k: BoxKind) =>
    k === "problem" ? t.category.orange : k === "new" ? t.accent.primary : t.stroke.primary;
  return (
    <Stack gap={6}>
      <Text size="small" tone="secondary" weight="semibold">
        {props.title}
      </Text>
      <div style={{ overflowX: "auto" }}>
        <svg width="100%" viewBox={`0 0 ${total + 4} ${h + 8}`} style={{ minWidth: 0.7 * total }}>
          <defs>
            <marker id={`ar-${props.title.length}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill={t.text.tertiary} />
            </marker>
          </defs>
          {props.boxes.map((b, i) => {
            const x = 2 + i * (w + gap);
            return (
              <g key={b.text}>
                <rect x={x} y={4} width={w} height={h} rx={6} fill={t.fill.secondary} stroke={stroke(b.kind)} strokeWidth={b.kind === "plain" ? 1 : 1.6} />
                <text x={x + w / 2} y={30} textAnchor="middle" fontSize={12} fontWeight={600} fill={t.text.primary}>
                  {b.text}
                </text>
                <text x={x + w / 2} y={48} textAnchor="middle" fontSize={10} fill={t.text.secondary}>
                  {b.sub}
                </text>
                {i < props.boxes.length - 1 ? (
                  <line x1={x + w} y1={4 + h / 2} x2={x + w + gap - 2} y2={4 + h / 2} stroke={t.text.tertiary} markerEnd={`url(#ar-${props.title.length})`} />
                ) : null}
              </g>
            );
          })}
        </svg>
      </div>
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Worked example: three meshes, one STRICT seam and one RELAXED seam
// ---------------------------------------------------------------------------

type Seating = "both" | "noRelaxed" | "noStrict";

const SEATINGS: Record<Seating, { label: string; m01: number; m12: number }> = {
  both: { label: "links on both seams", m01: 2, m12: 2 },
  noRelaxed: { label: "no M1-M2 link", m01: 2, m12: 0 },
  noStrict: { label: "no M0-M1 link", m01: 0, m12: 2 },
};

const VERDICT_TODAY = "rejected at parse: Cannot mix STRICT and RELAXED policies in the same graph";

function verdictProposed(s: Seating): { ok: boolean; text: string } {
  if (s === "noStrict") return { ok: false, text: "rejected: M0-M1 is a required edge with no link" };
  if (s === "noRelaxed") return { ok: true, text: "accepted, warning: M1-M2 (RELAXED) is zero-link" };
  return { ok: true, text: "accepted: both seams realized, 2 channels each" };
}

function ThreeMeshExample() {
  const t = useHostTheme();
  const [seating, setSeating] = useCanvasState<Seating>("seating", "noRelaxed");
  const s = SEATINGS[seating];
  const next = verdictProposed(seating);
  const mesh = (x: number, label: string) => (
    <g>
      <rect x={x} y={30} width={90} height={56} rx={6} fill={t.fill.secondary} stroke={t.stroke.primary} />
      <text x={x + 45} y={63} textAnchor="middle" fontSize={13} fontWeight={600} fill={t.text.primary}>
        {label}
      </text>
    </g>
  );
  const seam = (x1: number, x2: number, links: number, policy: string) => (
    <g>
      {links > 0 ? (
        [0, 1].slice(0, links).map((k) => (
          <line key={k} x1={x1} y1={52 + k * 12} x2={x2} y2={52 + k * 12} stroke={t.text.secondary} strokeWidth={1.6} />
        ))
      ) : (
        <line x1={x1} y1={58} x2={x2} y2={58} stroke={t.category.orange} strokeDasharray="4 3" />
      )}
      <text x={(x1 + x2) / 2} y={20} textAnchor="middle" fontSize={10.5} fill={policy === "STRICT" ? t.text.primary : t.text.secondary}>
        {policy} count 2
      </text>
      <text x={(x1 + x2) / 2} y={104} textAnchor="middle" fontSize={10} fill={links > 0 ? t.text.tertiary : t.category.orange}>
        {links > 0 ? `${links} links` : "no link"}
      </text>
    </g>
  );
  return (
    <Grid columns={fit(360)} gap={20} align="start">
      <Stack gap={8}>
        <svg width="100%" viewBox="0 0 430 112" style={{ maxWidth: 430 }}>
          {seam(100, 170, s.m01, "STRICT")}
          {seam(260, 330, s.m12, "RELAXED")}
          {mesh(10, "M0")}
          {mesh(170, "M1")}
          {mesh(330, "M2")}
        </svg>
        <Row gap={6} wrap>
          {(Object.keys(SEATINGS) as Seating[]).map((k) => (
            <span key={k}>
              <Pill active={k === seating} onClick={() => setSeating(k)}>
                {SEATINGS[k].label}
              </Pill>
            </span>
          ))}
        </Row>
      </Stack>
      <Stack gap={10}>
        <Table
          headers={["", "Verdict"]}
          rows={[
            ["Today", VERDICT_TODAY],
            ["Proposed", next.text],
          ]}
          rowTone={["danger", next.ok ? "success" : "danger"]}
        />
        <CodeCard
          path="the MGD (hypothetical)"
          language="protobuf"
          src={`
connections { nodes { mesh { mesh_descriptor: "M" mesh_id: 0 } }
              nodes { mesh { mesh_descriptor: "M" mesh_id: 1 } }
              channels { count: 2 policy: STRICT } }
connections { nodes { mesh { mesh_descriptor: "M" mesh_id: 1 } }
              nodes { mesh { mesh_descriptor: "M" mesh_id: 2 } }
              channels { count: 2 policy: RELAXED } }`}
        />
      </Stack>
    </Grid>
  );
}

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

function OverviewTab() {
  return (
    <Stack gap={18}>
      <Text tone="secondary">
        The proto already stores a policy on every connection. The code collapses it into one boolean for the whole MGD (the
        first FABRIC connection's policy), and that boolean becomes two global solver arguments. The design carries the policy
        per connection to the solver as a constraint on each target edge.
      </Text>
      <Flow
        title="Today"
        boxes={[
          { text: "MGD connections", sub: "policy per connection", kind: "plain" },
          { text: "is_inter_mesh_policy_relaxed", sub: "first connection wins", kind: "problem" },
          { text: "solver arguments", sub: "ConnectionValidationMode + allow_unmatched", kind: "problem" },
          { text: "every seam", sub: "all hard or all soft", kind: "problem" },
        ]}
      />
      <Flow
        title="Proposed"
        boxes={[
          { text: "MGD connections", sub: "policy per connection", kind: "plain" },
          { text: "demand per mesh pair", sub: "{ strict, relaxed } channels", kind: "new" },
          { text: "edge constraints", sub: "required / preferred per edge", kind: "new" },
          { text: "each seam", sub: "hard or soft on its own", kind: "new" },
        ]}
      />
      <Callout tone="info" title="Answer: yes, it can be a required constraint">
        MappingConstraints already has node constraints (required, preferred, forbidden, cardinality). It gains edge
        constraints, and every target edge resolves to one EdgeDemand {"{ required, preferred }"}. A caller sets it per edge
        or for the whole solve with set_default_edge_policy. The default lives in the constraints, so solve_topology_mapping
        no longer takes a ConnectionValidationMode, and allow_unmatched_target_edges becomes one value of the default. The
        intra-mesh solve sets its mesh's policy as the default; the inter-mesh seat solve sets per-pair demands.
      </Callout>
      <H2>Worked example: one STRICT seam, one RELAXED seam</H2>
      <ThreeMeshExample />
      <H2>Two separate things are conflated</H2>
      <Table
        headers={["Axis", "Values", "Today"]}
        rows={[
          ["Granularity", "device-level (pins exit chips) or mesh-level (a channel total between two meshes)", "device-level is called strict and mesh-level relaxed, and they cannot coexist"],
          ["Policy", "STRICT requires the channels; RELAXED prefers them and tolerates fewer, including zero (#56762)", "one global bool"],
        ]}
      />
      <Text size="small" tone="tertiary">
        The two axes are independent: a mesh-level connection may be STRICT and a device-level one RELAXED. #49960 frames them
        as one.
      </Text>
    </Stack>
  );
}

function TodayTab() {
  return (
    <Table
      headers={["Step", "Where", "What it does with the policy"]}
      rows={[
        ["Parse", "populate_inter_mesh_policy (mesh_graph_descriptor.cpp:789)", "one bool: the first FABRIC connection, else top-level graph_topology"],
        ["Validate", "MGD 1.0 compatibility check (:1198)", "rejects mixed policies within a graph"],
        ["Merge", "merge (:384), validate_shared_inter_mesh_policy (topology_mapper_utils.cpp:378), generate_rank_bindings.cpp:177", "three fatals when parts disagree"],
        ["MeshGraph", "mesh_graph.cpp:345-390", "requested ports / connections keep no policy; a second mesh-level connection on a pair overwrites the first"],
        ["Mapper config", "topology_mapper.cpp:563 (TODO #49960), topology_mapper_utils.cpp:1624", "inter_mesh_validation_mode = the bool"],
        ["Seat solve", "physical_grouping_descriptor_matching.cpp:3131, 3276, kSolveLadder :2953", "soft-seam rungs set allow_unmatched_target_edges for every seam"],
        ["Solver", "topology_solver_sat.cpp:856, 887, 1121, 1564", "soft switches off degree filter and AC-3 for the whole problem; realized literal and channel thresholds on every edge"],
        ["Exit nodes", "add_exit_node_constraints (topology_mapper_utils.cpp:955)", "per destination mesh, but the math reads the global mode"],
        ["Control plane", "control_plane.cpp:3595, 3747, 4152, 4606", "'either RelaxedGraph or Graph connections, not both'"],
        ["Control plane", ":3613, validate_requested_intermesh_connections :3818, downed links :1213, health :1015", "one policy for every inter-mesh boundary"],
      ]}
    />
  );
}

function SemanticsTab() {
  return (
    <Stack gap={14}>
      <Table
        headers={["Connection", "STRICT", "RELAXED"]}
        rows={[
          ["mesh-level, count c", "at least c channels between the two meshes", "up to c channels, maximized; zero allowed, warns"],
          ["device-level, count c", "c channels between the two named chips", "chips stay pinned as exit nodes; up to c channels, zero allowed"],
          ["graph_topology with channels", "each generated edge STRICT with that count", "each generated edge RELAXED with that count"],
          ["policy omitted", "STRICT (unchanged default)", "-"],
        ]}
      />
      <Grid columns={fit(360)} gap={16} align="start">
        <CodeCard
          path="demand per mesh pair"
          language="text"
          src={`
demand(A, B) = { strict:  sum of counts of STRICT connections on A-B,
                 relaxed: sum of counts of RELAXED connections on A-B }

edge A-B must exist       iff strict > 0
hard channel minimum      =   strict
preferred channel target  =   strict + relaxed`}
        />
        <Stack gap={8}>
          <H3>Example: both granularities on one pair</H3>
          <Text tone="secondary">
            A device-level STRICT pin of 1 channel plus a mesh-level RELAXED count of 2 on M0-M1 gives demand {"{1, 2}"}: one
            pinned channel is required, two more are wanted, and the pair is never zero-link. Today this MGD is rejected twice:
            at parse (mixed policies) and in the control plane (both granularities).
          </Text>
        </Stack>
      </Grid>
    </Stack>
  );
}

function SolverTab() {
  return (
    <Stack gap={16}>
      <H3>One structure per edge: today's two arguments are three points on it</H3>
      <Table
        headers={["EdgePolicy", "required", "preferred", "Today"]}
        rows={[
          ["STRICT", "c", "0", "ConnectionValidationMode::STRICT"],
          ["RELAXED", "1", "c - 1", "ConnectionValidationMode::RELAXED: the edge must exist, its channel count is best effort"],
          ["BEST_EFFORT", "0", "c", "RELAXED + allow_unmatched_target_edges (#56762): zero links allowed"],
        ]}
      />
      <Text size="small" tone="tertiary">
        c is the edge's channel count in the target graph. The MGD word RELAXED means two things: an intra-mesh RELAXED mesh
        still needs every link (solver RELAXED), and an inter-mesh RELAXED connection tolerates zero (solver BEST_EFFORT).
      </Text>
      <Grid columns={fit(420)} gap={16} align="start">
        <CodeCard
          added
          path="topology_solver.hpp  (MappingConstraints)"
          src={`
struct EdgeDemand {
    std::size_t required = 0;   // hard minimum; > 0 means the edge must exist
    std::size_t preferred = 0;  // soft extra; required + preferred == channels
};
enum class EdgePolicy { STRICT, RELAXED, BEST_EFFORT };

// Whole solve: every target edge with no per-edge constraint.
// Unset means RELAXED, today's default argument.
void set_default_edge_policy(EdgePolicy policy);
std::optional<EdgePolicy> default_edge_policy() const;

// One edge, sized from its channel count in the target graph.
bool set_edge_policy(TargetNode a, TargetNode b, EdgePolicy policy);

// One edge, explicit numbers (an inter-mesh pair's {strict, relaxed}).
// Calls on the same edge add up.
bool add_required_edge_constraint(TargetNode a, TargetNode b, std::size_t min_channels = 1);
bool add_preferred_edge_constraint(TargetNode a, TargetNode b, std::size_t channels);

// For the backends and the validator.
std::map<std::pair<TargetNode, TargetNode>, EdgeDemand> resolve_edge_demands(
    const AdjacencyGraph<TargetNode>& target_graph) const;`}
        />
        <Stack gap={10}>
          <H3>Rules</H3>
          <Table
            headers={["Rule", ""]}
            rows={[
              ["Precedence", "explicit numbers, then set_edge_policy, then the default; numbers and a policy on one edge is rejected"],
              ["Counts", "explicit required + preferred must equal the target edge's channel count, or the solve fails naming the edge"],
              ["merge", "explicit numbers add up; two different defaults, or two policies on one edge, are rejected"],
              ["allow_unmatched", "set_allow_unmatched_target_edges(true) becomes a deprecated alias for set_default_edge_policy(BEST_EFFORT)"],
            ]}
          />
          <Callout tone="neutral" title="Faster for mixed MGDs">
            Today one RELAXED seam makes every seam soft and switches off the degree filter and AC-3 for the whole master solve.
            Per edge, those prunings stay on for required edges.
          </Callout>
        </Stack>
      </Grid>
      <H3>The solver call no longer takes a mode</H3>
      <Grid columns={fit(420)} gap={16} align="start">
        <CodeCard
          added
          path="topology_solver.hpp"
          src={`
template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    bool quiet_mode = false,
    TopologyMappingSolverEngine solver_engine = TopologyMappingSolverEngine::Auto);
// Same for solve_topology_mapping_n / _all and the enumeration session.`}
        />
        <Text tone="secondary">
          The old signatures stay, deprecated, with the mode made required. They copy the constraints and set the mode as the
          default edge policy when none is set; a different default already set fails the solve. A three-argument call
          resolves to the new overload, and enum class and bool don't convert, so nothing is ambiguous. About 190 uses of
          ConnectionValidationMode:: in 13 files (127 in test_topology_solver.cpp) can move later. Inside the backends the mode
          disappears: the SAT encoder, DFS and the validator read EdgeDemand per edge.
        </Text>
      </Grid>
      <H2>Both solves use it</H2>
      <Table
        headers={["Solve", "Where", "Today", "Per edge"]}
        rows={[
          ["Intra-mesh", "complete_intra_mesh_for_placement (topology_mapper_utils.cpp:1459-1468)", "per-mesh mode passed to solve_topology_mapping; TT_FATAL forbids allow_unmatched", "set_default_edge_policy(STRICT or RELAXED) from the mesh's policy; no edge may resolve to required == 0"],
          ["Intra-mesh candidate pools", "CandidatePool, create_sat_placement_pools (physical_grouping_descriptor_matching.cpp:2211, 2558)", "sat_intra_mesh_mode_by_mesh_ holds a mode; part of the pool hash", "holds an EdgePolicy, set as the pool's default; hash uses the policy"],
          ["PGD grouping matches", ":586, :1361, :1698, :1779", "mode argument", "default edge policy on their constraints"],
          ["Inter-mesh seat solve", "SatPlacementEnumerationSession", "mode argument plus soft-seam bool", "explicit numbers per mesh pair; default STRICT so a missing demand fails loudly"],
          ["Single-mesh fallback", "topology_mapper.cpp:1746", "RELAXED argument", "set_default_edge_policy(RELAXED)"],
        ]}
      />
      <Callout tone="info" title="Per-connection intra-mesh policy">
        The MGD can't say it today: a mesh has one channels {"{ count policy }"} and express links inherit it
        (mesh_graph_descriptor.proto:246). The solver would support it with set_edge_policy per edge; giving express links their
        own channels is a separate proto change (open question 5). Intra-mesh BEST_EFFORT stays forbidden: a missing link
        inside a mesh breaks routing.
      </Callout>
      <H2>SAT encoding (topology_solver_sat.cpp)</H2>
      <Text size="small" tone="tertiary">
        Required edge: required &gt; 0. Preferred-only edge: required == 0. A uniform default reproduces today's encoding for
        the matching mode.
      </Text>
      <Table
        headers={["Step", "Today", "Per edge"]}
        rows={[
          ["1. Domain degree filter (:856)", "skipped for every target when soft", "global degree >= number of required edges"],
          ["2. AC-3 (:887)", "skipped for the whole problem when soft", "runs over required arcs only"],
          ["3b. prune_active_lit (:1364)", "guards the hard domains of a soft encoding", "same, from required-only domains"],
          ["6. Adjacency support (:1121)", "every edge hard, or every edge guarded", "required: unguarded, channel-compatible with its minimum; preferred-only: guarded by its realized literal"],
          ["Channel thresholds (:1564)", "RELAXED: levels 1..count on every edge", "levels required+1 .. required+preferred, only where preferred > 0"],
          ["Realized-edge objective (:1867, :2033)", "every edge when soft", "preferred-only edges"],
          ["Seam toggle (:1926)", "assume every realized literal", "set_best_effort_edges_required(bool): BEST_EFFORT edges act as RELAXED; required edges hard in both modes"],
          ["Validator (topology_solver.tpp:3800)", "missing edge: error, or warning when soft", "required missing or short: error; preferred missing or short: warning naming the edge"],
          ["DFS backend (tpp:1354)", "soft edges need SAT", "same: a zero-link preferred edge needs SAT"],
        ]}
      />
      <Text size="small" tone="tertiary">
        Objective order is unchanged: host cap and host objective, then realized preferred edges, then channel thresholds, then
        preferred node hits.
      </Text>
      <H3>The seat solve's ladder (kSolveLadder, physical_grouping_descriptor_matching.cpp:2953)</H3>
      <Table
        headers={["Rung", "Today", "Per edge"]}
        rows={[
          ["hard seams, host cap / dropped", "every seam hard", "set_best_effort_edges_required(true); STRICT hard"],
          ["soft seams, host cap / dropped", "only if the global policy is RELAXED; every seam soft", "only if some pair has strict == 0 and relaxed > 0; only those seams soft"],
        ]}
      />
    </Stack>
  );
}

function ChangesTab() {
  return (
    <Stack gap={16}>
      <Grid columns={fit(380)} gap={16} align="start">
        <Stack gap={8}>
          <H3>MGD and MeshGraph</H3>
          <Table
            headers={["Change", "Where"]}
            rows={[
              ["per-pair demand table from every FABRIC connection and graph_topology", "replaces populate_inter_mesh_policy"],
              ["drop 'Cannot mix STRICT and RELAXED' (last PR)", "mesh_graph_descriptor.cpp:1198"],
              ["requested ports carry a policy; connections become a demand map that sums", "mesh_graph.cpp:363, 383"],
              ["drop the three merge-agreement fatals", "merge, validate_shared_inter_mesh_policy, generate_rank_bindings"],
              ["LogicalMultiMeshGraph carries demand per mesh edge and exit edge", "build_logical_multi_mesh_adjacency_graph_impl (TODO :198)"],
              ["proto and MGD_README comments only; no schema change", "mesh_graph_descriptor.proto"],
            ]}
          />
        </Stack>
        <CodeCard
          added
          path="mesh_graph_descriptor.hpp / mesh_graph.hpp"
          src={`
struct InterMeshChannelDemand {
    std::uint32_t strict_channels = 0;
    std::uint32_t relaxed_channels = 0;
};
// Summed over every connection between the two meshes, both granularities.
InterMeshChannelDemand get_inter_mesh_channel_demand(MeshId a, MeshId b) const;
bool has_relaxed_inter_mesh_connection() const;

// Deprecated: answer "any RELAXED" / "any policy given", not "the first connection".
bool is_inter_mesh_policy_relaxed() const;
bool is_inter_mesh_policy_specified() const;`}
        />
      </Grid>
      <Grid columns={fit(380)} gap={16} align="start">
        <Stack gap={8}>
          <H3>Mapper</H3>
          <Table
            headers={["Change", "Where"]}
            rows={[
              ["edge constraints from each pair's demand", "SatPlacementEnumerationSession"],
              ["STRICT math on strict channels, capped RELAXED math on relaxed; skip only when strict == 0", "add_exit_node_constraints"],
              ["take the demand table instead of one mode; set the mesh's intra policy as the default edge policy", "complete_intra_mesh_for_placement"],
              ["inter_mesh_validation_mode becomes a tools/tests override, unset by default", "TopologyMappingConfig"],
              ["unique_shapes off when policies mix", "map_multi_mesh_to_physical, --all-solutions"],
            ]}
          />
        </Stack>
        <Stack gap={8}>
          <H3>Control plane (#49960)</H3>
          <Table
            headers={["Change", "Where"]}
            rows={[
              ["remove the four 'not both' fatals", ":3595, :3747, :4152, :4606"],
              ["quota per boundary: per exit node if device-level, else channel total", "pair_logical_intermesh_ports, single-host path"],
              ["strict shortfall throws (FSD excepted), relaxed shortfall warns, zero only if strict == 0", ":3613, validate_requested_intermesh_connections"],
              ["count a downed inter-mesh link only on a boundary with strict > 0", "reject_used_downed_links_under_strict_policy"],
              ["RELAXED if any connection is RELAXED", "validate_system_health_against_mesh_graph_policy"],
            ]}
          />
        </Stack>
      </Grid>
    </Stack>
  );
}

function ImplicationsTab() {
  return (
    <Table
      headers={["Area", "Effect"]}
      rows={[
        ["Existing MGDs", "unchanged: every connection in a graph shares one policy today, so demands give the same constraints; goldens pin it"],
        ["Newly accepted", "mixed policies (rejected at parse today), merged parts that disagree (rejected at merge), device- plus mesh-level connections (rejected by the control plane)"],
        ["Hierarchical MGDs", "the compatibility check is per graph, so two graphs may differ today and the first connection silently wins; after, each is honoured. List affected MGDs before landing"],
        ["Solver API", "the mode argument and allow_unmatched_target_edges become one constraint, set_default_edge_policy; old signatures keep working, deprecated"],
        ["Intra-mesh", "same behaviour, expressed as the default edge policy; ready for per-edge intra-mesh policy"],
        ["Solve time", "mixed MGDs faster than all-RELAXED; uniform unchanged"],
        ["Symmetry", "swapping two same-shape meshes can move a STRICT seam onto a pair with fewer links, so unique_shapes must be off when policies mix"],
        ["Warnings", "zero-link and shortfall warnings name the connection and its policy"],
        ["Pipeline design", "a STRICT pipeline edge over a RELAXED connection may end zero-link: Tier B keeps declared channels as an upper bound, Tier C decides; merging parts with different policies stops failing"],
      ]}
      rowTone={["success", "success", "warning", undefined, "success", "success", "warning", undefined, undefined]}
    />
  );
}

function PlanTab() {
  return (
    <Stack gap={16}>
      <Text tone="secondary">
        Land the MGD side and the solver side as one PR stack. The parse-time "Cannot mix" check stays until the last PR, so no
        MGD is accepted before every consumer can handle it. A1, A2, A3 go in order; the A series and B are independent; C
        needs A1 and B; D and E need C.
      </Text>
      <Table
        headers={["PR", "Contents", "Behaviour change"]}
        rows={[
          ["A1. Solver edge constraints", "EdgePolicy, EdgeDemand, set_default_edge_policy, set_edge_policy, add_required / add_preferred, resolve_edge_demands, merge; SAT steps; validator and DFS per edge; the old mode argument becomes the default inside the solver", "none: old signatures resolve to the same demands"],
          ["A2. Solver call without a mode", "overloads without ConnectionValidationMode; old ones deprecated; allow_unmatched alias; session set_best_effort_edges_required", "none"],
          ["A3. Intra-mesh callers", "complete_intra_mesh_for_placement, candidate pools, PGD grouping matches, topology_mapper.cpp:1746 set the default edge policy", "none: same policy, now a constraint"],
          ["B. MGD demand tables", "InterMeshChannelDemand, get_inter_mesh_channel_demand; requested maps carry policy; sum instead of overwrite", "none for valid MGDs, except the overwrite fix"],
          ["C. Mapper", "seat solve and exit nodes read demands; ladder scoped; config override; unique_shapes gate; drop merge-agreement fatals", "merged parts with different policies accepted"],
          ["D. Control plane", "per-boundary quota, validation, downed links, health (#49960)", "device- and mesh-level connections may coexist"],
          ["E. Allow mixing", "drop the MGD 1.0 'Cannot mix' check; proto and README comments; deprecate is_inter_mesh_policy_relaxed", "MGDs may mix policies"],
        ]}
        rowTone={[undefined, undefined, undefined, undefined, "warning", "warning", "warning"]}
      />
      <H3>Tests</H3>
      <Table
        headers={["PR", "Test"]}
        rows={[
          ["A1, A2", "required edge with no global edge is UNSAT; preferred edge solves with a warning and is realized when possible; required+preferred meets its minimum and prefers the total; pruning still applies to required edges; a uniform STRICT / RELAXED / BEST_EFFORT default matches the mode argument (property test); precedence, count check, merge conflicts and a disagreeing mode argument fail"],
          ["A3", "intra-mesh goldens in TopologyMapperUtilsTest and the PGD suites unchanged; an intra-mesh edge resolving to required == 0 fails"],
          ["B", "device-level STRICT 1 + mesh-level RELAXED 2 gives {1, 2}; two mesh-level connections on a pair sum; every existing textproto gives the same demands as its global policy"],
          ["C", "M0-M1 STRICT, M1-M2 RELAXED: no M1-M2 link accepted with a warning, no M0-M1 link rejected; STRICT part merged with RELAXED part succeeds; unique_shapes off when mixed"],
          ["D", "#49960 acceptance criteria; STRICT shortfall throws, RELAXED warns; downed link on a RELAXED boundary not rejected; single- and multi-host goldens"],
          ["all", "TopologyMapperUtilsTest, MeshGraphDescriptorTests, control-plane and CPU-only fabric suites unchanged for uniform MGDs"],
        ]}
      />
      <H3>Open questions</H3>
      <Table
        headers={["#", "Question"]}
        rows={[
          ["1", "Device-level RELAXED: keep the named chips pinned (proposed), or let the link move to other chips?"],
          ["2", "graph_topology plus an explicit connection on the same pair: add up (proposed), or override?"],
          ["3", "The realized-edge objective already prefers hard seams: keep the hard-seam rungs for the host-cap interplay, or drop them?"],
          ["4", "How long does TopologyMappingConfig::inter_mesh_validation_mode stay as a forcing override?"],
          ["5", "Give express links (or topology dimensions) their own channels { count policy } for per-edge intra-mesh policy, or keep one policy per mesh?"],
          ["6", "An unset default edge policy means RELAXED, today's default argument. Flip it to STRICT, matching the MGD, once every caller sets it?"],
        ]}
      />
    </Stack>
  );
}

const SECTIONS = [
  { id: "overview", label: "Overview", body: OverviewTab },
  { id: "today", label: "How the policy flows today", body: TodayTab },
  { id: "semantics", label: "Semantics per connection", body: SemanticsTab },
  { id: "solver", label: "Solver: a constraint, not an argument", body: SolverTab },
  { id: "changes", label: "MGD, mapper and control plane", body: ChangesTab },
  { id: "implications", label: "Implications", body: ImplicationsTab },
  { id: "plan", label: "Implementation plan", body: PlanTab },
] as const;

export default function MgdIntermeshPolicyPlan() {
  const dispatch = useCanvasAction();
  const t = useHostTheme();
  return (
    <Stack gap={28} style={{ padding: 24, maxWidth: 1240 }}>
      <Stack gap={8}>
        <H1>Per-connection STRICT / RELAXED for inter-mesh connections</H1>
        <Text tone="secondary">
          Design for review; nothing is implemented. Related: tt-metal#49960 (mixed inter-mesh connections in the control
          plane), #56762 (RELAXED zero-link tolerance, landed), and the pipeline-graph design, whose merge of several parts
          needs this.
        </Text>
        <Row gap={8}>
          <Button
            variant="ghost"
            onClick={() => dispatch({ type: "openFile", path: "/data/rsong/tt-metal3/docs/scratch/mgd-intermesh-policy-plan.md" })}
          >
            Open the plan doc
          </Button>
        </Row>
        <Row gap={6} wrap>
          {SECTIONS.map((s, i) => (
            <span key={s.id}>
              <Pill onClick={() => scrollToId(`sec-${s.id}`)}>
                {i + 1}. {s.label}
              </Pill>
            </span>
          ))}
        </Row>
      </Stack>
      {SECTIONS.map((s, i) => {
        const Body = s.body;
        return (
          <div key={s.id} id={`sec-${s.id}`} style={{ scrollMarginTop: 12 }}>
            <Stack gap={16}>
              <div style={{ borderTop: `1px solid ${t.stroke.secondary}`, paddingTop: 16 }}>
                <Text size="small" tone="tertiary">
                  PART {i + 1} OF {SECTIONS.length}
                </Text>
                <H2 style={{ margin: 0 }}>{s.label}</H2>
              </div>
              <Body />
            </Stack>
          </div>
        );
      })}
    </Stack>
  );
}
