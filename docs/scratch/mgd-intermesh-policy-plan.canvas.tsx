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
        intra-mesh solve sets its mesh's policy as the default; the inter-mesh seat solve sets per-pair demands. The same
        structure carries the cable PortType: allowed types (hard) and preferred types (soft) per edge or per solve, plus an
        implicit tie-break preference from a built-in cable order when the MGD says nothing (part 7).
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
              ["drop 'Cannot mix STRICT and RELAXED'", "mesh_graph_descriptor.cpp:1198"],
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

type CableSeat = { id: string; label: string; cable: string; links: number };

const CABLE_SEATS: CableSeat[] = [
  { id: "a", label: "Seating A", cable: "WARP400", links: 2 },
  { id: "b", label: "Seating B", cable: "QSFP_DD", links: 2 },
  { id: "c", label: "Seating C", cable: "WARP100", links: 2 },
];

function CableExample() {
  const t = useHostTheme();
  const [useAllowed, setUseAllowed] = useCanvasState<boolean>("cable-allowed", true);
  const [usePreferred, setUsePreferred] = useCanvasState<boolean>("cable-preferred", false);
  const [useImplicit, setUseImplicit] = useCanvasState<boolean>("cable-implicit", true);
  const allowed = (cable: string) => !useAllowed || cable === "WARP400" || cable === "QSFP_DD";
  const builtIn = ["TRACE", "LINKING_BOARD_1", "LINKING_BOARD_2", "LINKING_BOARD_3", "WARP400", "WARP100", "QSFP_DD"];
  const source = usePreferred ? "explicit preferred_port_types (PRIMARY)" : useImplicit ? "built-in order (TIE_BREAK)" : "";
  const tier = (cable: string) => (usePreferred ? (cable === "WARP400" ? 0 : 1) : useImplicit ? builtIn.indexOf(cable) : 0);
  const valid = CABLE_SEATS.filter((s) => allowed(s.cable));
  const bestTier = Math.min(...valid.map((s) => tier(s.cable)));
  const chosen = valid.filter((s) => tier(s.cable) === bestTier).map((s) => s.id);
  const verdict = (s: CableSeat) => {
    if (!allowed(s.cable)) return "rejected: WARP100 not in allowed_port_types (STRICT)";
    if (chosen.includes(s.id)) return chosen.length > 1 ? "valid, tied: no cable preference, either may win" : `valid, chosen by ${source}`;
    return "valid, not chosen: worse cable tier";
  };
  const proto = [
    "channels {",
    "  count: 2",
    "  policy: STRICT",
    ...(useAllowed ? ["  allowed_port_types: [ WARP400, QSFP_DD ]"] : []),
    ...(usePreferred ? ["  preferred_port_types: [ WARP400 ]"] : []),
    "}",
  ].join("\n");
  const color = (cable: string) =>
    cable === "WARP400" ? t.accent.primary : cable === "QSFP_DD" ? t.category.purple : t.category.orange;
  return (
    <Grid columns={fit(360)} gap={20} align="start">
      <Stack gap={10}>
        <svg width="100%" viewBox="0 0 360 200" style={{ maxWidth: 360 }}>
          {CABLE_SEATS.map((s, i) => {
            const y = 14 + i * 62;
            const dim = !allowed(s.cable);
            return (
              <g key={s.id} opacity={dim ? 0.45 : 1}>
                <rect x={10} y={y} width={70} height={40} rx={5} fill={t.fill.secondary} stroke={t.stroke.primary} />
                <text x={45} y={y + 25} textAnchor="middle" fontSize={12} fontWeight={600} fill={t.text.primary}>M0</text>
                <rect x={200} y={y} width={70} height={40} rx={5} fill={t.fill.secondary} stroke={t.stroke.primary} />
                <text x={235} y={y + 25} textAnchor="middle" fontSize={12} fontWeight={600} fill={t.text.primary}>M1</text>
                {[0, 1].map((k) => (
                  <line key={k} x1={80} y1={y + 14 + k * 12} x2={200} y2={y + 14 + k * 12} stroke={color(s.cable)} strokeWidth={2} strokeDasharray={dim ? "4 3" : undefined} />
                ))}
                <text x={140} y={y + 10} textAnchor="middle" fontSize={10} fill={t.text.secondary}>{s.cable} x{s.links}</text>
                <text x={282} y={y + 25} fontSize={11} fontWeight={600} fill={chosen.includes(s.id) ? t.accent.primary : t.text.tertiary}>
                  {s.label}
                </text>
              </g>
            );
          })}
        </svg>
        <Row gap={6} wrap>
          <span>
            <Pill active={useAllowed} onClick={() => setUseAllowed(!useAllowed)}>allowed_port_types</Pill>
          </span>
          <span>
            <Pill active={usePreferred} onClick={() => setUsePreferred(!usePreferred)}>preferred_port_types</Pill>
          </span>
          <span>
            <Pill active={useImplicit} onClick={() => setUseImplicit(!useImplicit)}>implicit default</Pill>
          </span>
        </Row>
      </Stack>
      <Stack gap={10}>
        <Table
          headers={["Seating", "Verdict"]}
          rows={CABLE_SEATS.map((s) => [`${s.label} (${s.cable})`, verdict(s)])}
          rowTone={CABLE_SEATS.map((s) => (!allowed(s.cable) ? "danger" : chosen.includes(s.id) ? "success" : undefined))}
        />
        <CodeCard
          path="the MGD (hypothetical)"
          language="protobuf"
          src={useImplicit ? `# M0-M1 connection\n${proto}` : `port_type_defaults { disable: true }\n# M0-M1 connection\n${proto}`}
        />
      </Stack>
    </Grid>
  );
}

function CableTab() {
  return (
    <Stack gap={16}>
      <Text tone="secondary">
        Every Ethernet channel in the PSD has a PortType (EthConnection::port_type): TRACE, QSFP_DD, WARP100, WARP400,
        LINKING_BOARD_1..3 or UNKNOWN. The solver never sees it: build_adjacency_graph_physical (topology_solver.cpp:164) and
        the seat graph (physical_grouping_descriptor_matching.cpp:2667) turn each channel into one parallel edge. The per-edge
        structure is where it fits: each global edge gets a count per cable type, and each target edge, or the whole solve,
        says which types may count (hard) and which it prefers (soft).
      </Text>
      <H3>Example: prefer WARP400 over QSFP-DD, accept only those two</H3>
      <CableExample />
      <Text size="small" tone="tertiary">
        Under RELAXED instead of STRICT, seating C would be accepted as zero-link with a warning. The cable orderings here are
        illustrations, not recommendations.
      </Text>
      <Grid columns={fit(420)} gap={16} align="start">
        <CodeCard
          added
          path="topology_solver.hpp  (MappingConstraints)"
          src={`
using ChannelClass = std::uint32_t;   // the mapper uses static_cast<ChannelClass>(port_type)

// Global side: channels per class on a global edge; must add up to its multiplicity.
// Unannotated edges have all channels in class 0.
bool set_global_edge_channel_classes(
    GlobalNode a, GlobalNode b, const std::map<ChannelClass, std::size_t>& channels_per_class);

// Hard: only these classes count toward an edge's demand. Empty = all.
void set_default_allowed_channel_classes(std::set<ChannelClass> classes);
bool set_edge_allowed_channel_classes(TargetNode a, TargetNode b, std::set<ChannelClass> classes);

// Soft: preference tiers, best first. Unlisted classes are in the last tier.
// PRIMARY ranks above preferred node hits; TIE_BREAK below them.
enum class PreferenceStrength { PRIMARY, TIE_BREAK };
void set_default_channel_class_preference(
    std::vector<ChannelClass> best_first, PreferenceStrength strength = PreferenceStrength::PRIMARY);
bool set_edge_channel_class_preference(
    TargetNode a, TargetNode b, std::vector<ChannelClass> best_first,
    PreferenceStrength strength = PreferenceStrength::PRIMARY);

struct EdgeDemand {
    std::size_t required = 0;
    std::size_t preferred = 0;
    std::set<ChannelClass> allowed_classes;          // empty = all
    std::vector<ChannelClass> primary_preference;    // best first; empty = none
    std::vector<ChannelClass> tie_break_preference;  // best first; empty = none
};`}
        />
        <CodeCard
          added
          path="mesh_graph_descriptor.proto"
          language="protobuf"
          src={`
message Channels {
    int32 count = 1;
    optional Policy policy = 2;
    // Hard: only these cable types count toward \`count\`. Empty = any. UNKNOWN always counts.
    repeated PortType allowed_port_types = 3;
    // Soft: preferred cable types, best first. Unlisted types rank last.
    repeated PortType preferred_port_types = 4;
}

message PortTypeDefaults {
    // Replaces the built-in tie-break order for this MGD. Best first.
    repeated PortType preferred_port_types = 1;
    // No implicit preference at all; only explicit Channels fields apply.
    bool disable = 2;
}

message MeshGraphDescriptor {
    // ... fields 1-5 unchanged ...
    optional PortTypeDefaults port_type_defaults = 6;
}
// PortType imported from physical_system_descriptor.proto (open question 8).`}
        />
      </Grid>
      <H3>Semantics</H3>
      <Table
        headers={["Piece", "Meaning"]}
        rows={[
          ["Allowed classes (hard)", "usable channels of target edge e on global edge g = g's channels in e's allowed classes; every STRICT / RELAXED / BEST_EFFORT rule uses that count instead of the multiplicity"],
          ["Preference (soft)", "e's tier on g = worst tier among the best c_e allowed channels; minimize edges in the worst tier, then the next (lexicographic); no weight tuning"],
          ["Objective order", "host cap and host objective, realized preferred edges, channel thresholds, PRIMARY cable tiers, preferred node hits, TIE_BREAK cable tiers; capacity before cable type (open question 7)"],
          ["UNKNOWN", "always allowed, last tier, one warning per solve; all-UNKNOWN systems map exactly as today"],
        ]}
      />
      <H3>SAT encoding</H3>
      <Table
        headers={["Piece", "Encoding"]}
        rows={[
          ["Allowed classes", "usable count per (target edge, global pair) computed up front; support clauses and AC-3 filter neighbours per target edge; same clause shape"],
          ["Channel thresholds", "count usable channels, not the multiplicity"],
          ["Preference", "z[e,T] for tiers worse than the best; x[t1,g1] ∧ x[t2,g2] → z[e,T] for pairs in tier T; best-tier pairs add nothing; minimize with the existing binary search (:2033)"],
          ["Cost", "O(|E_target| × |E_global|) ternary clauses outside the best tier; nothing when no preference is set"],
          ["DFS", "allowed classes are a per-edge channel check DFS can do; preferences need SAT (DFS only orders candidates)"],
        ]}
      />
      <H3>Where the classes come from</H3>
      <Table
        headers={["Solve", "Global edge", "Counts per class built in"]}
        rows={[
          ["Intra-mesh", "ASIC pair", "build_adjacency_graph_physical already loops get_eth_connections; also emit counts per port_type"],
          ["Seat solve", "footprint pair", "AdjacencyMatrixCache / saturated_link_count (:2395) counts per port type; build_sat_placement_seat_graph annotates seat pairs"],
          ["Exit nodes", "chip", "prefer exit chips whose inter-mesh links are in a better tier (preferred node constraints)"],
          ["Control plane binding", "channel", "the solver picks chip pairs, not channels: pair_logical_intermesh_ports binds allowed channels in tier order; shortfall counts allowed channels"],
        ]}
      />
      <Grid columns={fit(420)} gap={16} align="start">
        <CodeCard
          path="intra-mesh: prefer on-board traces and linking boards (hypothetical)"
          language="protobuf"
          src={`
mesh_descriptors {
  name: "M"
  arch: BLACKHOLE
  device_topology { dims: [ 4, 4 ] dim_types: [ RING, RING ] }
  host_topology   { dims: [ 1, 1 ] }
  channels {
    count: 2
    policy: STRICT
    preferred_port_types: [ TRACE, LINKING_BOARD_1, LINKING_BOARD_2, LINKING_BOARD_3 ]
  }
}`}
        />
        <CodeCard
          added
          path="C++ against the solver"
          src={`
MappingConstraints<FabricNodeId, PhysicalNodeId> c;
c.set_default_edge_policy(EdgePolicy::STRICT);
for (const auto& [pair, per_type] : channels_per_port_type(psd, physical_graph)) {
    c.set_global_edge_channel_classes(pair.first, pair.second, per_type);
}
c.set_default_channel_class_preference({cls(PortType::TRACE), cls(PortType::LINKING_BOARD_1)});
// One edge must not use QSFP-DD at all:
c.set_edge_allowed_channel_classes(n0, n1, {cls(PortType::TRACE), cls(PortType::WARP400)});
auto result = solve_topology_mapping(logical_graph, physical_graph, c);`}
        />
      </Grid>
      <Text size="small" tone="tertiary">
        Intra-mesh fabric uses every channel between two adjacent chips, so an intra-mesh preference changes which chip pairs a
        smaller mesh uses, not which channels.
      </Text>
      <ImplicitCableSection />
    </Stack>
  );
}

function ImplicitCableSection() {
  return (
    <Stack gap={14}>
      <H2>Implicit preference: on by default</H2>
      <Text tone="secondary">
        Most MGDs will never mention cables, and the mapper should still prefer better cables when everything else is equal.
        Every edge gets a tie-break preference by default: an implicit preferred constraint, added the same way the mapper
        already adds PGD pinnings as preferred node constraints (add_pgd_pinning_preferred_constraints,
        topology_mapper_utils.cpp:889, called at :1450).
      </Text>
      <Table
        headers={["Source (highest first)", "Scope", "Strength"]}
        rows={[
          ["Channels.preferred_port_types on a connection", "that inter-mesh connection", "PRIMARY"],
          ["Channels.preferred_port_types on a mesh descriptor", "every intra-mesh edge of that mesh", "PRIMARY"],
          ["MeshGraphDescriptor.port_type_defaults", "every edge in the MGD", "TIE_BREAK"],
          ["TopologyMappingConfig::port_type_preference", "every edge in the solve", "TIE_BREAK"],
          ["built-in default_port_type_preference()", "every edge in the solve", "TIE_BREAK"],
        ]}
      />
      <Grid columns={fit(420)} gap={16} align="start">
        <Stack gap={8}>
          <Callout tone="info" title="Why tie-break, below preferred node hits">
            An implicit preference must never override anything the user or the system asked for. Below the PGD pinning
            preferences it only chooses among mappings equal on everything else: host use, realized edges, channel counts,
            pinning hits. The PGD-chosen layout stays stable and golden changes are limited to former ties.
            port_type_defaults {"{ disable: true }"} or disable_implicit_port_type_preference removes the three tie-break rows.
          </Callout>
          <Text size="small" tone="tertiary">
            Nothing in the codebase ranks port types today (link_health records medium but doesn't order it), so the order
            needs a sign-off from the hardware and scaleout owners. The proposal is fewest connectors and shortest reach first,
            to be confirmed.
          </Text>
        </Stack>
        <CodeCard
          added
          path="tt_metal/fabric/port_type_preference.hpp  (new)"
          src={`
inline std::vector<PortType> default_port_type_preference() {
    return {PortType::TRACE,
            PortType::LINKING_BOARD_1, PortType::LINKING_BOARD_2, PortType::LINKING_BOARD_3,
            PortType::WARP400, PortType::WARP100,
            PortType::QSFP_DD};
    // UNKNOWN is never listed: it always ranks last.
}`}
        />
      </Grid>
      <H3>Implementation: one more family of mapper-built constraints</H3>
      <Grid columns={fit(420)} gap={16} align="start">
        <CodeCard
          added
          path="topology_mapper_utils.cpp"
          src={`
// What one solve needs, resolved once from the MGD and the config.
struct PortTypeRules {
    std::set<PortType> default_allowed;       // mesh-level allowed, or empty
    std::vector<PortType> default_primary;    // mesh-level preferred, or empty
    std::vector<PortType> tie_break;          // after precedence; empty if disabled
    std::map<std::pair<MeshId, MeshId>, std::set<PortType>> pair_allowed;     // inter-mesh
    std::map<std::pair<MeshId, MeshId>, std::vector<PortType>> pair_primary;  // inter-mesh
};

// Annotates every global edge with its channels per PortType and adds the allowed
// and preferred port-type constraints for this solve, in precedence order.
template <typename TargetNode, typename GlobalNode>
void add_port_type_constraints(
    MappingConstraints<TargetNode, GlobalNode>& constraints,
    const PortTypeRules& rules,
    const std::map<std::pair<GlobalNode, GlobalNode>,
                   std::map<PortType, std::size_t>>& channels_per_port_type);`}
        />
        <Table
          headers={["Step", "Change"]}
          rows={[
            ["1. MGD", "parse per-mesh and per-connection lists and port_type_defaults; get_mesh_port_type_rules, get_inter_mesh_port_type_rules, get_port_type_defaults; conflicting lists on one pair rejected at parse"],
            ["2. Counts per type", "build_adjacency_graph_physical returns channels per PortType per ASIC pair; AdjacencyMatrixCache counts per type next to saturated_link_count; the seat graph returns a map per seat pair"],
            ["3. Intra-mesh", "complete_intra_mesh_for_placement calls add_port_type_constraints after the PGD preferred pinnings (:1450); pools and PGD grouping matches use the tie-break only"],
            ["4. Seat solve", "pair_allowed and pair_primary as per-edge constraints, tie-break as the default"],
            ["5. Exit nodes", "prefer exit chips whose inter-mesh channels are in the best tier, at tie-break strength"],
            ["6. Solver", "tie-break objective after preferred node hits; nothing added when one tier covers every global edge"],
            ["7. Results", "MappingResult reports edges per tier; one log line per solve, so golden diffs say why"],
            ["8. Control plane", "binding takes allowed channels explicit order first, then the tie-break table"],
            ["9. unique_shapes", "keep the best-tier representative: SAT finds it first under the objective"],
          ]}
        />
      </Grid>
      <Callout tone="warning" title="Goldens">
        The implicit default changes which mapping wins a tie, so goldens can move on systems whose descriptors carry cabling
        data. A report test lists every golden that changes, and the same PR updates them.
      </Callout>
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
        ["Cable type, explicit", "opt-in; an allowed list can make a STRICT connection UNSAT, and the error names the connection and the types found"],
        ["Cable type, implicit", "on by default; only breaks ties left after every other objective, so the constraints met never change, but which tied mapping wins can; UNKNOWN-only systems unchanged; disable: true restores today's choice"],
        ["Solve time", "mixed MGDs faster than all-RELAXED; uniform unchanged"],
        ["Symmetry", "swapping two same-shape meshes can move a STRICT seam onto a pair with fewer links, so unique_shapes must be off when policies mix"],
        ["Warnings", "zero-link and shortfall warnings name the connection and its policy"],
        ["Pipeline design", "a STRICT pipeline edge over a RELAXED connection may end zero-link: Tier B keeps declared channels as an upper bound, Tier C decides; merging parts with different policies stops failing"],
      ]}
      rowTone={["success", "success", "warning", undefined, "success", "warning", "warning", "success", "warning", undefined, undefined]}
    />
  );
}

function PlanTab() {
  return (
    <Stack gap={16}>
      <Text tone="secondary">
        One PR, with the MGD side, the solver side, the mapper, the control plane and cable types together. Every piece needs
        another to be useful (the demand table needs the solver encoding, the solver change needs a caller, cable preferences
        need the per-edge structure), so splitting it would only land code with no caller.
      </Text>
      <Table
        headers={["Area", "What the PR changes"]}
        rows={[
          ["Solver", "EdgePolicy, EdgeDemand, set_default_edge_policy, set_edge_policy, add_required / add_preferred, resolve_edge_demands, merge; SAT steps; validator and DFS per edge; overloads without ConnectionValidationMode, old ones deprecated; allow_unmatched alias; session set_best_effort_edges_required"],
          ["Solver, cable types", "ChannelClass, set_global_edge_channel_classes, allowed classes, preferences at both scopes and both strengths; SAT pieces; tiers in MappingResult"],
          ["MGD and MeshGraph", "demand table, get_inter_mesh_channel_demand; requested maps carry policy and sum; allowed_port_types / preferred_port_types and port_type_defaults; drop 'Cannot mix' and the merge-agreement fatals; proto and README comments; deprecate is_inter_mesh_policy_relaxed"],
          ["Mapper", "intra-mesh callers set the default edge policy; seat solve and exit nodes read demands; PortTypeRules, add_port_type_constraints, counts per PortType, default_port_type_preference(); config overrides; unique_shapes gate"],
          ["Control plane", "per-boundary quota, validation, downed links, health (#49960); binding in cable-tier order; shortfall counts allowed channels"],
        ]}
      />
      <Text size="small" tone="tertiary">
        The ~190 existing uses of ConnectionValidationMode:: keep compiling through the deprecated overloads, so the PR doesn't
        have to touch them.
      </Text>
      <H3>Behaviour changes in the PR</H3>
      <Table
        headers={["Change", "How it's reviewed"]}
        rows={[
          ["MGDs may mix policies; merged parts may disagree; device- and mesh-level connections may coexist", "nothing accepted today is rejected"],
          ["hierarchical MGDs with different policies per graph honour each one", "the PR description lists the affected textprotos"],
          ["the implicit cable preference can change which tied mapping wins", "a report test lists every golden that moves; the PR updates them; the built-in order is settled in review before merge"],
        ]}
        rowTone={["success", "warning", "warning"]}
      />
      <H3>Tests</H3>
      <Table
        headers={["Area", "Test"]}
        rows={[
          ["Solver", "required edge with no global edge is UNSAT; preferred edge solves with a warning and is realized when possible; required+preferred meets its minimum and prefers the total; pruning still applies to required edges; a uniform STRICT / RELAXED / BEST_EFFORT default matches the mode argument (property test); precedence, count check, merge conflicts and a disagreeing mode argument fail"],
          ["Intra-mesh", "intra-mesh goldens in TopologyMapperUtilsTest and the PGD suites unchanged; an intra-mesh edge resolving to required == 0 fails"],
          ["MGD", "device-level STRICT 1 + mesh-level RELAXED 2 gives {1, 2}; two mesh-level connections on a pair sum; every existing textproto gives the same demands as its global policy"],
          ["Mapper", "M0-M1 STRICT, M1-M2 RELAXED: no M1-M2 link accepted with a warning, no M0-M1 link rejected; STRICT part merged with RELAXED part succeeds; unique_shapes off when mixed"],
          ["Control plane", "#49960 acceptance criteria; STRICT shortfall throws, RELAXED warns; downed link on a RELAXED boundary not rejected; single- and multi-host goldens"],
          ["Cable type", "equal link counts, one over the preferred type: preferred wins; STRICT edge on only disallowed links is UNSAT, BEST_EFFORT is zero-link with a warning; UNKNOWN always counts; no cabling data maps as today; capacity beats tier (4 QSFP-DD over 2 WARP400 for a RELAXED edge); binding takes preferred channels first"],
          ["Implicit default", "no MGD fields: a TRACE / QSFP-DD tie goes to TRACE; a PGD preferred pinning beats the implicit order; explicit beats port_type_defaults beats config beats built-in; disable: true gives today's mapping; a report test lists every golden the default would change"],
          ["Regression", "TopologyMapperUtilsTest, MeshGraphDescriptorTests, control-plane and CPU-only fabric suites unchanged for uniform MGDs"],
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
          ["7", "Channel count ranks above cable tier. Should a preferred cable type win over more channels instead, or be configurable?"],
          ["8", "Import PortType from physical_system_descriptor.proto (proposed), or define an MGD-owned enum?"],
          ["9", "Lexicographic tiers (proposed), or integer costs per type, e.g. two QSFP-DD links as good as one WARP400?"],
          ["10", "Who signs off on default_port_type_preference(), and is fewest connectors / shortest reach the right rule? Per arch or cluster type?"],
          ["11", "Implicit default as tie-break below preferred node hits (proposed), or above them so cables can outrank a PGD-preferred layout?"],
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
  { id: "cable", label: "Cable type: constraints by port type", body: CableTab },
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
