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
        implicit tie-break preference from a built-in cable order when the MGD says nothing (part 6). After the solve, the
        mapper hands the control plane every allowed inter-mesh channel, and the control plane drops its gather but keeps
        channel selection and direction assignment (part 9). The SAT encoding and constraint API are in part 10. Moving channel selection and directions into the mapper is a stretch goal (part 11).
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
          ["Channel thresholds (:1564)", "RELAXED: levels 1..count on every edge, built but discarded (:1861), never optimized", "levels required+1 .. required+preferred, only where preferred > 0, maximized as their own stage (part 10)"],
          ["Realized-edge objective (:1867, :2033)", "every edge when soft", "preferred-only edges"],
          ["Seam toggle (:1926)", "assume every realized literal", "set_best_effort_edges_required(bool): BEST_EFFORT edges act as RELAXED; required edges hard in both modes"],
          ["Validator (topology_solver.tpp:3800)", "missing edge: error, or warning when soft", "required missing or short: error; preferred missing or short: warning naming the edge"],
          ["DFS backend (tpp:1354)", "soft edges need SAT", "same: a zero-link preferred edge needs SAT"],
        ]}
      />
      <Text size="small" tone="tertiary">
        Today's objectives keep their order: host cap and host packing, then preferred node hits, then realized preferred-only
        edges. Preferred channels follow as a new stage; today the channel-threshold literals are built but never optimized.
        Part 10 has the full schedule.
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
// PRIMARY is optimized before TIE_BREAK; both after capacity and node hits (part 10).
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
          ["Objective order", "host cap and host packing, preferred node hits, realized preferred-only edges, preferred channels, PRIMARY cable tiers, TIE_BREAK cable tiers, TIE_BREAK node preferences; the first three are today's order; capacity before cable type (open question 7)"],
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
            needs a sign-off from the hardware and scaleout owners. The proposal follows tt-metal#36425: TRACE (same host), then
            WARP400 (400G), then QSFP-DD (400G, below WARP400), then WARP100 (100G). Linking boards go after TRACE, to be confirmed.
          </Text>
        </Stack>
        <CodeCard
          added
          path="tt_metal/fabric/port_type_preference.hpp  (new)"
          src={`
inline std::vector<PortType> default_port_type_preference() {
    return {PortType::TRACE,
            PortType::LINKING_BOARD_1, PortType::LINKING_BOARD_2, PortType::LINKING_BOARD_3,
            PortType::WARP400, PortType::QSFP_DD,
            PortType::WARP100};
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
            ["6. Solver", "tie-break objective after the primary tiers (part 10); nothing added when one tier covers every global edge"],
            ["7. Results", "MappingResult reports edges per tier; one log line per solve, so golden diffs say why"],
            ["8. Control plane", "binding takes allowed channels explicit order first, then the tie-break table"],
            ["9. unique_shapes", "cable objectives skipped there, as node hits are today; the seat solve and final intra-mesh solve, which don't use it, make the cable choice"],
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

function CandidatesTab() {
  return (
    <Stack gap={16}>
      <Text tone="secondary">
        The mapper is the one place that knows which inter-mesh channels are allowed: the ones between the meshes it mapped, on
        cable types the rules allow, on links that are up. After the solve it hands that set to the control plane. The control
        plane keeps channel selection (budgets, trimming, round-robin) and direction assignment (Z claim, N/E/S/W, overflow
        onto Z) as they are. What goes away is the gather: ranks no longer read their own cables and send them to rank 0.
      </Text>
      <Flow
        title="Today"
        boxes={[
          { text: "mapper on rank 0", sub: "mapping + exit nodes, broadcast", kind: "plain" },
          { text: "gather per rank", sub: "PortDescriptorTable, link-up check", kind: "problem" },
          { text: "rank 0 pairs", sub: "pair_logical_intermesh_ports", kind: "plain" },
          { text: "broadcast + apply", sub: "every rank sets its ETH chans", kind: "plain" },
        ]}
      />
      <Flow
        title="Proposed"
        boxes={[
          { text: "ranks report down links", sub: "in the chip info already sent", kind: "new" },
          { text: "mapper lists candidates", sub: "collect_inter_mesh_candidates", kind: "new" },
          { text: "rank 0 pairs", sub: "unchanged, tier-ordered links", kind: "plain" },
          { text: "broadcast + apply", sub: "unchanged", kind: "plain" },
        ]}
      />
      <Callout tone="warning" title="Why">
        The gather re-derives from scratch what the mapper already knew, and it knows nothing about cable types. So the pairing
        can bind a channel on a cable type the mapper was told not to use. With the table, the control plane can only bind
        channels the mapper allows.
      </Callout>
      <H3>What the control plane does today (generate_intermesh_connectivity, control_plane.cpp:3580)</H3>
      <Table
        headers={["Step", "Where", "What"]}
        rows={[
          ["Gather", "generate_port_descriptor_table (:3737), gather_intermesh_cables_for_exit_nodes (:3649)", "each rank lists its exit-node cables from the PSD; filters by MPI ownership, requested exit nodes, is_ethernet_link_up (:3725); one-dst-chip fatal (:3696)"],
          ["Send to rank 0", "forward_descriptors_to_controller", "MPI gather of every PortDescriptorTable"],
          ["Pair", "pair_logical_intermesh_ports (:4144)", "join by connection_hash; budgets per boundary or exit chip; trim to budget (has_budget_after_trim, :4379); boundary-major or round-robin; Z claim, N/E/S/W, overflow to Z; one neighbor per (chip, direction)"],
          ["Broadcast and apply", "forward_intermesh_connections_from_controller, convert_port_descriptors_to_intermesh_connections (:4476)", "every rank maps (mesh, port_id, hash) back to its channel"],
          ["Check and load", ":3613, validate_requested_intermesh_connections (:3818), rebuild exit maps (:3549), load_intermesh_connections", "shortfall policy, exit maps, routing tables"],
          ["Single host", "generate_intermesh_connections_on_local_host (:4588)", "a different greedy algorithm for the same job"],
        ]}
      />
      <H3>What changes and what stays</H3>
      <Table
        headers={["Concern", "Today", "After"]}
        rows={[
          ["Which channels exist between two mapped meshes", "each rank reads its PSD; MPI gather to rank 0", "mapper on rank 0, after the solve: get_inter_mesh_candidates()"],
          ["Cable type", "ignored", "disallowed PortType left out; each channel carries its type and tier"],
          ["Liveness", "is_ethernet_link_up per rank during the gather", "ranks report down channels in the chip info they already send to rank 0 (topology_mapper.cpp:515)"],
          ["Requested exit nodes (device-level)", "filtered per rank", "filtered by the mapper, same rule"],
          ["How many channels", "every live cable on the requested boundaries", "the same: every allowed channel, not capped to the demand"],
          ["One dst chip per (src chip, neighbor mesh)", "fatal per rank", "same check and message, once on rank 0 over the table"],
          ["Channel selection", "pair_logical_intermesh_ports", "unchanged, except links are taken best tier first, so trimming keeps preferred cables"],
          ["Direction assignment", "pair_logical_intermesh_ports", "unchanged"],
          ["Broadcast, apply, routing tables, shortfall, validation", "control plane", "unchanged"],
        ]}
        rowTone={["success", "success", "success", undefined, undefined, undefined, undefined, undefined, undefined]}
      />
      <Grid columns={fit(440)} gap={16} align="start">
        <CodeCard
          added
          path="tt-metalium/experimental/fabric/inter_mesh_candidates.hpp"
          src={`
struct InterMeshCandidateChannel {
    FabricNodeId src;                 // exit node on the src mesh
    FabricNodeId dst;                 // peer exit node
    tt::tt_metal::AsicID src_asic;
    tt::tt_metal::AsicID dst_asic;
    std::uint8_t src_chan = 0;        // physical ETH channels
    std::uint8_t dst_chan = 0;
    tt::tt_metal::PortType port_type = tt::tt_metal::PortType::UNKNOWN;
    std::uint8_t tier = 0;            // 0 = most preferred (PortTypeRules)
    std::size_t connection_hash = 0;  // std::hash<ExitNodeConnection>
};

enum class InterMeshCandidateExclusion : std::uint8_t {
    LINK_DOWN, PORT_TYPE_DISALLOWED, NOT_REQUESTED_EXIT,
};

struct InterMeshCandidates {
    // (src mesh, dst mesh), both directions; sorted by
    // (src chip, tier, src chan, dst chan).
    std::map<std::pair<MeshId, MeshId>,
             std::vector<InterMeshCandidateChannel>> channels;
    std::vector<std::pair<InterMeshCandidateChannel,
                          InterMeshCandidateExclusion>> excluded;
    PortDescriptorTable to_port_descriptor_table() const;
};

struct InterMeshCandidateOptions {
    std::set<std::size_t> down_connection_hashes;
    PortTypeRules port_type_rules;
};

// Pure: no cluster, no MPI.
InterMeshCandidates collect_inter_mesh_candidates(
    const MeshGraph& mesh_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const std::map<FabricNodeId, tt::tt_metal::AsicID>& fabric_node_to_asic,
    const InterMeshCandidateOptions& options);

// Valid on the rank that solved (rank 0).
const InterMeshCandidates& TopologyMapper::get_inter_mesh_candidates() const;`}
        />
        <Stack gap={10}>
          <H3>How the table is built</H3>
          <Table
            headers={["Keep a channel when", "Detail"]}
            rows={[
              ["1. Both ends are mapped", "to fabric nodes on the two meshes of a requested boundary; nothing into unmapped ASICs or a third mesh"],
              ["2. The link is up", "connection_hash not in down_connection_hashes"],
              ["3. The cable type is allowed", "by PortTypeRules; UNKNOWN always counts; tier from preferred list, then port_type_defaults, then the built-in order"],
              ["4. Device-level only", "src chip is a requested exit node, as get_requested_exit_nodes does today"],
            ]}
          />
          <Callout tone="info" title="All of them, not just the demand">
            Everything that passes goes in, including channels beyond the demand. A RELAXED boundary with more cabling than
            requested gets all of it, exactly as the gather gave today. Choosing among them stays the pairing's job.
          </Callout>
          <Text size="small" tone="secondary">
            Down links: the chip info each rank already sends to rank 0 (broadcast_chip_info_to_hosts) gains the hashes of PSD
            cables from this rank's ASICs to another rank's whose src ETH link is down. build_mapping already has the Cluster.
            The set is computed before the mapping exists, so the rank doesn't need to know which cables will be inter-mesh.
          </Text>
          <Text size="small" tone="tertiary">
            Order: today's gather order (control_plane.cpp:3674) with tier added. With no cable data every tier is 0, so the
            order is unchanged.
          </Text>
        </Stack>
      </Grid>
      <H3>What the control plane changes</H3>
      <Grid columns={fit(440)} gap={16} align="start">
        <CodeCard
          added
          path="control_plane.cpp"
          src={`
if (!generate_mapping_locally_ && world_size > 1) {
    AnnotatedIntermeshConnections paired;
    if (my_rank == 0) {
        const auto& candidates = topology_mapper_->get_inter_mesh_candidates();
        check_one_dst_chip_per_src_chip(candidates);  // today's :3696 fatal
        auto port_descriptors = candidates.to_port_descriptor_table();
        paired = pair_logical_intermesh_ports(port_descriptors);  // + tier order
    }
    forward_intermesh_connections_from_controller(paired);       // unchanged
    intermesh_connections = apply_intermesh_connections_to_local_chips(paired);
} else {
    intermesh_connections = generate_intermesh_connections_on_local_host();  // filtered
}
// shortfall, validation, exit maps, routing tables: unchanged`}
        />
        <Table
          headers={["Function", "Change"]}
          rows={[
            ["generate_port_descriptor_table, gather_intermesh_cables_for_exit_nodes, get_requested_exit_nodes, forward_descriptors_to_controller", "removed"],
            ["convert_port_descriptors_to_intermesh_connections", "split: the apply half (cable_lookup from the local PSD, then exit_node_directions_ and intermesh_chan_to_peer_, :4489-4585) stays as apply_intermesh_connections_to_local_chips"],
            ["pair_logical_intermesh_ports", "PortDescriptor gains tier; the per-link sort (:4305) sorts by (tier, hash). has_budget_after_trim keeps a link's first channels, so trimming keeps preferred cables. All tiers 0: same as today"],
            ["generate_intermesh_connections_on_local_host", "same algorithm; skips cables not in the table, so single-host respects cable types and down links"],
            ["shortfall, validation, exit maps, routing tables", "unchanged; a STRICT shortfall from disallowed types or down links names them, from excluded"],
          ]}
        />
      </Grid>
      <Callout tone="neutral" title="Out of scope: a link that goes down after the solve">
        Today's gather checks is_ethernet_link_up once, and nothing handles a link that goes down after that. The down-link report
        in the chip info replaces that check at about the same point, so behaviour is unchanged. Later link-downs are left for a
        follow-up.
      </Callout>
      <H3>Implications</H3>
      <Table
        headers={["Area", "Effect"]}
        rows={[
          ["Consistency", "the control plane only binds channels the mapper allows: no disallowed cable type, no down link, no meshes the mapper didn't map together"],
          ["MPI", "the port-descriptor gather goes away; the down-link report rides the existing chip-info message; the paired-result broadcast stays"],
          ["Control plane code", "pairing and direction assignment unchanged apart from the tier sort; gather code removed"],
          ["RELAXED boundaries", "every allowed channel is a candidate, more than requested when the cabling has more"],
          ["Cable preference", "trimmed links keep preferred cables; which links the round-robin picks is unchanged (that is the stretch goal)"],
          ["Testing", "collect_inter_mesh_candidates is pure: CPU-only tests on PSD fixtures"],
          ["Debugging", "inter_mesh_candidates.yaml: every boundary, its channels with type and tier, every excluded channel with its reason"],
          ["Parity", "no cable data and no down links: to_port_descriptor_table() equals the gathered table, so pairings are identical"],
        ]}
        rowTone={["success", "success", "success", undefined, undefined, "success", undefined, "success"]}
      />
      <H3>Implementation: one PR, after the policy PR</H3>
      <Table
        headers={["Area", "Change"]}
        rows={[
          ["New", "inter_mesh_candidates.hpp/.cpp: types, collect_inter_mesh_candidates, to_port_descriptor_table"],
          ["Mapper", "down-link hashes in the chip info to rank 0; collect after the final mapping; get_inter_mesh_candidates(); yaml dump"],
          ["Control plane", "new multi-host branch; apply_intermesh_connections_to_local_chips; PortDescriptor::tier and the tier sort; single-host filter; removals"],
          ["Tests", "parity on every multi-host golden; each filter with its exclusion reason, UNKNOWN passes; RELAXED demand 2 over 4 live channels gets 4 candidates and binds 2; a trimmed mixed link keeps TRACE over QSFP-DD; single-host never binds a cable outside the table; same paired result on every rank"],
        ]}
      />
    </Stack>
  );
}

function SatTab() {
  return (
    <Stack gap={16}>
      <Text tone="secondary">
        The implementation reference for the solver side: how each constraint is set, how it reaches the SAT encoder, and the
        exact variables, clauses and objective stages. A constraint goes through the same three layers as today's node
        constraints: MappingConstraints stores it, ConstraintIndexData resolves it to indices once per solve, and
        TopologySatConstraintView hands it to the encoder.
      </Text>
      <Callout tone="warning" title="Two findings in today's encoder">
        The RELAXED channel-threshold literals are built (topology_sat_append_relaxed_channel_threshold_literals, :1564) but the
        result is discarded (:1861), so SAT never maximizes channels today. And preferred node hits outrank realized edges: the
        node-hit stage stays assumed while realized edges are maximized (:2005). The schedule below keeps today's order for
        today's objectives and adds the channel stage for real.
      </Callout>
      <H3>From API call to encoder</H3>
      <Table
        headers={["API call", "Stored in MappingConstraints", "Resolved to", "Read by"]}
        rows={[
          ["set_default_edge_policy(p)", "optional<EdgePolicy> default_edge_policy_", "edge_demands", "support clauses, realized literals, channel indicators"],
          ["set_edge_policy(a, b, p)", "edge_policies_, key (min, max)", "edge_demands", "same"],
          ["add_required_edge_constraint / add_preferred_edge_constraint", "explicit_edge_demands_; calls add up", "edge_demands", "same"],
          ["set_global_edge_channel_classes(a, b, counts)", "global_edge_classes_", "global_class_count", "usable-count table"],
          ["set_default_ / set_edge_allowed_channel_classes", "default_allowed_classes_, edge_allowed_classes_", "IndexedEdgeDemand::allowed", "usable-count table"],
          ["set_default_ / set_edge_channel_class_preference", "one default and one per-edge map per strength", "::primary, ::tie_break", "tier table, tier literals"],
          ["add_preferred_constraint(t, globals, strength)", "preferred_mappings_ (PRIMARY), tie_break_preferred_mappings_", "preferred_global_indices, tie_break_preferred_global_indices", "node-hit indicators"],
          ["set_allow_unmatched_target_edges(true)", "deprecated alias: default BEST_EFFORT", "", ""],
          ["session set_best_effort_edges_required(bool)", "live session flag", "", "assumptions in next()"],
        ]}
      />
      <Grid columns={fit(440)} gap={16} align="start">
        <CodeCard
          added
          path="topology_solver.hpp (ConstraintIndexData)"
          src={`
// One per undirected target edge (t1 < t2).
struct IndexedEdgeDemand {
    std::size_t t1 = 0;
    std::size_t t2 = 0;
    std::size_t required = 0;
    std::size_t preferred = 0;
    std::vector<ChannelClass> allowed;    // sorted; empty = every class
    std::vector<ChannelClass> primary;    // best first
    std::vector<ChannelClass> tie_break;  // best first
};
std::vector<IndexedEdgeDemand> edge_demands;
std::vector<std::map<std::size_t, std::size_t>> edge_demand_of;  // t1 -> t2 -> index

// Channels per class for each adjacent global pair; absent = all class 0.
std::vector<std::map<std::size_t,
    std::vector<std::pair<ChannelClass, std::size_t>>>> global_class_count;
std::vector<std::vector<std::size_t>> tie_break_preferred_global_indices;`}
        />
        <Stack gap={8}>
          <H3>Resolution, once per solve</H3>
          <Table
            headers={["Step", "Rule"]}
            rows={[
              ["1", "explicit numbers, else set_edge_policy, else the default, else RELAXED"],
              ["2", "policy to numbers from c = target_conn_count: STRICT {c, 0}, RELAXED {1, c-1}, BEST_EFFORT {0, c}"],
              ["3", "required + preferred == c; class counts add up to global_conn_count; else the solve fails through trivial_reason, naming the edge"],
              ["4", "allowed classes and preferences: per edge if set, else the default"],
            ]}
          />
          <Text size="small" tone="secondary">
            Precomputed per target edge e and candidate pair (ga, gb): the usable count u_e (channels in allowed classes plus
            every UNKNOWN; with no annotation, today's multiplicity) and the tier per strength (worst tier among the best c_e
            usable channels). Built in the loop the support clauses already run, so no extra cost order.
          </Text>
        </Stack>
      </Grid>
      <H3>Hard constraints</H3>
      <Table
        headers={["Piece", "Today", "Per edge"]}
        rows={[
          ["Assignment, exactly-one, injectivity, resources, groups, cardinality", "x[t,g] per domain global (:979, :1012, :1027)", "unchanged"],
          ["Domain degree filter (:842)", "skipped when soft", "g stays in dom(t) only with at least deg_req(t) neighbours, counting t's required edges"],
          ["AC-3 (:880)", "skipped when soft; STRICT compares multiplicity with c", "arcs only for required edges; ga supported if some gb has u_e(ga, gb) at least required_e"],
          ["Required edge", "STRICT: multiplicity at least c; RELAXED: adjacent", "¬x[t1,ga] ∨ ⋁ x[t2,gb] over gb with u_e(ga,gb) at least required_e, both directions"],
          ["Preferred-only edge (required 0)", "every edge guarded when soft", "realized literal r_e: ¬r_e ∨ ¬x[t1,ga] ∨ ⋁ x[t2,gb] over gb with u_e at least 1; only these edges get r_e"],
          ["prune_active_lit (:1364)", "guards hard domains of a soft encoding", "built from the domains required edges pruned"],
        ]}
      />
      <Text size="small" tone="tertiary">
        Uniform STRICT gives required = c on every edge, today's STRICT clause; uniform RELAXED gives required = 1, today's
        RELAXED clause; uniform BEST_EFFORT gives today's soft encoding.
      </Text>
      <H3>Indicators for the objectives</H3>
      <Grid columns={fit(440)} gap={16} align="start">
        <CodeCard
          added
          path="topology_solver_sat.cpp (new literals)"
          language="text"
          src={`
Realized edges     r_e, full-width counter R[k] (:512)

Preferred channels for k = required_e+1 .. required_e+preferred_e (max 24 levels)
  I[e,k]  <=>  OR { x[t1,ga] AND x[t2,gb] : u_e(ga,gb) >= k }   (:1491)
  I[e,k+1] -> I[e,k]
  counter over all I  ->  C[k]
  limits: 512 pairs per indicator, 2048 indicators; past them the stage is skipped

Cable tiers, per strength S, tiers 0..L-1 (0 best)
  z_S[e,j] = "e uses a pair of tier >= j",  j = 1..L-1
  each pair with tier t >= 1:  NOT x[t1,ga] OR NOT x[t2,gb] OR z_S[e,t]
  z_S[e,j] -> z_S[e,j-1]
  counter per (S, j)

Node hits
  PRIMARY:   p_t <=> OR x[t,g] over preferred globals (:1439), as today
  TIE_BREAK: q_t, same shape, full-width counter`}
        />
        <Stack gap={8}>
          <Text tone="secondary">
            Maximizing the I literals maximizes Σ min(c_e, u_e), the score DFS already orders candidates by. Pairs in the best
            tier add no clause, and an edge whose pairs share one tier gets no z literals. A BEST_EFFORT edge left with zero
            links selects no adjacent pair, so it isn't charged a tier.
          </Text>
          <Table
            headers={["Piece", "Variables", "Clauses"]}
            rows={[
              ["Support clauses", "none", "unchanged"],
              ["Realized literals", "one per preferred-only edge", "unchanged shape, guarded"],
              ["Preferred-channel indicators", "at most 24 per edge, 2048 per solve, plus one per pair", "3 per pair, 1 + pairs per indicator, one chain per level"],
              ["Cable tiers", "at most L - 1 per edge with a preference", "one ternary per pair above tier 0, plus the chain"],
              ["Counters", "m × k", "about 4 m k (:434)"],
            ]}
          />
        </Stack>
      </Grid>
      <H3>Objective schedule</H3>
      <Grid columns={fit(440)} gap={16} align="start">
        <CodeCard
          added
          path="topology_solver_sat.cpp (Impl)"
          src={`
struct SatObjective {
    enum class Kind : std::uint8_t { MAXIMIZE, MINIMIZE };
    Kind kind;
    std::vector<int> thresholds;  // thresholds[k-1] true => at least k literals true
    std::string name;             // for stats and logs
};
std::vector<SatObjective> objectives;  // after today's \`stages\`

// next(): today's stage fallback unchanged, then each objective in order.
// MAXIMIZE: binary search the largest k with thresholds[k-1] satisfiable.
// MINIMIZE: the smallest k with NOT thresholds[k] satisfiable.
// The optimum stays assumed for every later objective (lexicographic).
// Each probe tries the symmetry hint first; two undecided probes end the
// objective with the best model found, as today.`}
        />
        <Table
          headers={["#", "Objective", "Kind", "Status"]}
          rows={[
            ["1", "host packing (minimize_lit)", "assumption stage", "today"],
            ["2", "preferred node hits, at least the lower bound (preferred_lit)", "assumption stage", "today"],
            ["3", "realized preferred-only edges, R[k]", "MAXIMIZE", "today"],
            ["4", "preferred channels, C[k]", "MAXIMIZE", "new (built today, unused)"],
            ["5", "PRIMARY cable tiers, worst tier first", "MINIMIZE per tier", "new"],
            ["6", "TIE_BREAK cable tiers", "MINIMIZE per tier", "new"],
            ["7", "TIE_BREAK node hits, q_t", "MAXIMIZE", "new"],
          ]}
          rowTone={[undefined, undefined, undefined, "info", "info", "info", "info"]}
        />
      </Grid>
      <Grid columns={fit(440)} gap={16} align="start">
        <Callout tone="info" title="Cable tiers rank below node hits">
          Keeping today's results fixes rows 1 to 3, and node hits sit above realized edges there. So both cable strengths rank
          below node hits as well as below capacity. Ranking PRIMARY above node hits would mean moving node hits below realized
          edges, changing every soft solve with pinnings (open question 11).
        </Callout>
        <Callout tone="neutral" title="unique_shapes and cost when unused">
          Objectives 4 to 7 are skipped under unique_shapes, as node hits are today (:1813): that mode lists every footprint. The
          cable choice is made by the seat solve and the final intra-mesh solve, which don't use it. An objective with no
          literals isn't added; with no classes and no preferences only objective 4 is new, covered by the golden report test.
        </Callout>
      </Grid>
      <H3>How the mapper sets them</H3>
      <Grid columns={fit(440)} gap={16} align="start">
        <CodeCard
          added
          path="topology_mapper_utils.cpp, physical_grouping_descriptor_matching.cpp"
          src={`
// Intra-mesh (complete_intra_mesh_for_placement, :1459)
intra.set_default_edge_policy(mesh_is_relaxed ? EdgePolicy::RELAXED : EdgePolicy::STRICT);
add_port_type_constraints(intra, rules.for_mesh(mesh_id), channels_per_port_type);
auto result = solve_topology_mapping(logical_graph, physical_graph, intra);

// Seat solve (build_sat_placement_constraints, :2732)
constraints.set_default_edge_policy(EdgePolicy::STRICT);
for (const auto& [pair, demand] : inter_mesh_channel_demands) {
    if (demand.strict > 0) {
        constraints.add_required_edge_constraint(pair.first, pair.second, demand.strict);
    }
    if (demand.relaxed > 0) {
        constraints.add_preferred_edge_constraint(pair.first, pair.second, demand.relaxed);
    }
}
for (const auto& [seats, links_per_type] : seat_pair_links_per_port_type) {
    constraints.set_global_edge_channel_classes(seats.first, seats.second, to_classes(links_per_type));
}
add_port_type_constraints(constraints, rules.for_seat_solve(), seat_pair_links_per_port_type);

// Exit nodes (add_exit_node_constraints, :955): minimum from demand.strict only
intra.add_preferred_constraint(fabric_node, exit_asics_in_best_tier, PreferenceStrength::TIE_BREAK);`}
        />
        <Stack gap={8}>
          <Text tone="secondary">
            The mesh-level target graph carries strict + relaxed parallel edges per pair, so the count check holds; the seat graph
            already carries one parallel edge per link (:2723), and the per-type counts add up to it. A pair with strict 0 is
            preferred-only and gets an r_e. The ladder's hard-seam rungs call set_best_effort_edges_required(true), which assumes
            those r_e; STRICT pairs are hard in every rung.
          </Text>
          <Table
            headers={["DFS backend", "Behaviour"]}
            rows={[
              ["Required edge, allowed classes", "edge check compares u_e with required_e"],
              ["Preferred-only edge", "rejected with today's message (topology_solver.tpp:1354)"],
              ["Preferred channels", "candidate order by Σ min(c_e, u_e)"],
              ["Cable tiers, TIE_BREAK node hits", "candidate order only"],
            ]}
          />
          <Table
            headers={["Change in a live session", "Without re-encoding?"]}
            rows={[
              ["node forbidden / required", "yes: unit clauses via refresh_constraints (:1900)"],
              ["set_best_effort_edges_required", "yes: assumptions on r_e"],
              ["blocking clauses", "yes, as today"],
              ["edge demands, classes, preferences", "no: they shape clauses; new session"],
            ]}
          />
          <Text size="small" tone="tertiary">
            The stretch goal's seat-solve conditions need no new encoding (multiplicities, class counts, forbidden seats); only its
            pair-level block is a new clause (part 11).
          </Text>
        </Stack>
      </Grid>
      <H3>Tests for the encoding</H3>
      <Table
        headers={["Test", "What it checks"]}
        rows={[
          ["Brute-force cross-check", "random instances up to 5 targets and 7 globals with demands, classes, allowed sets and preferences: SAT returns a feasible mapping with the optimal objective vector, and UNSAT exactly when nothing is feasible"],
          ["Encoding parity", "uniform STRICT / RELAXED / BEST_EFFORT default gives the same variables and clauses as the mode argument, apart from objective 4 (CNF dump hook)"],
          ["Pieces", "STRICT edge on only disallowed classes is UNSAT; UNKNOWN always counts; z fires only for its tier, chain holds; one-tier edge adds nothing"],
          ["Order", "for each adjacent pair of objectives, a hand-built conflict where the higher one wins"],
          ["Limits", "past the indicator caps the stage is skipped, the mapping is valid, one debug line names the edge"],
        ]}
      />
    </Stack>
  );
}

function StretchTab() {
  return (
    <Stack gap={16}>
      <Callout tone="warning" title="Optional, experimental, not planned for implementation">
        Part 9 is the plan. This records the design for doing more in the mapper, so it can be picked up later.
      </Callout>
      <Text tone="secondary">
        With the global view, the mapper could also choose the channels. Choosing channels without directions is the wrong split:
        a chip's channels toward one neighbor mesh must sit in one direction, and each direction serves one neighbor
        (find_free_dir / dir_owner, control_plane.cpp:4183). A selection that ignores directions can be legal in count and still
        bad. So if the mapper chooses channels, it assigns directions in the same pass, and the control plane only applies the
        result.
      </Text>
      <Table
        headers={["Problem", "Example"]}
        rows={[
          ["Overflow", "4 channels picked from chip 3 toward M1 when its only free direction has 2 ports: 2 dropped, or a STRICT boundary fails that another selection meets"],
          ["Congestion", "a boundary's channels piled onto one chip and direction while its other exit chips have idle ports"],
          ["Contention", "two boundaries share an exit chip; the first takes the best direction and the second gets only Z, or nothing"],
        ]}
        rowTone={["warning", "warning", "warning"]}
      />
      <Flow
        title="Stretch"
        boxes={[
          { text: "mapper lists candidates", sub: "part 9", kind: "plain" },
          { text: "mapper binds + directs", sub: "resolve_inter_mesh_binding", kind: "new" },
          { text: "plan broadcast", sub: "with the existing chip info", kind: "new" },
          { text: "every rank applies", sub: "no pairing, no 2nd broadcast", kind: "new" },
        ]}
      />
      <Grid columns={fit(440)} gap={16} align="start">
        <CodeCard
          added
          path="tt-metalium/experimental/fabric/inter_mesh_binding.hpp"
          src={`
struct InterMeshChannelBinding {
    InterMeshCandidateChannel channel;  // part 9
    port_id_t src_port;                 // (direction, index)
    port_id_t dst_port;
    bool strict = true;                 // STRICT part of the demand, else RELAXED
};

struct InterMeshBindingPlan {
    std::vector<InterMeshBoundaryBinding> boundaries;  // sorted by (a, b)
    std::size_t strict_shortfall() const;
    AnnotatedIntermeshConnections to_annotated_connections() const;
};

struct InterMeshBindingFailure {
    MeshId a, b;
    enum class Kind : std::uint8_t {
        FOOTPRINT,  // too few allowed channels between the footprints
        PLACEMENT,  // depends on the chip mapping: pins, ports per direction
    } kind;
    std::string message;
};

struct InterMeshBindingResult {  // exactly one is set
    std::optional<InterMeshBindingPlan> plan;
    std::optional<InterMeshBindingFailure> failure;
};

InterMeshBindingResult resolve_inter_mesh_binding(
    const MeshGraph& mesh_graph,
    const InterMeshCandidates& candidates,
    const PortCapacityModel& port_capacity);`}
        />
        <Stack gap={10}>
          <H3>How channels and directions are chosen</H3>
          <Table
            headers={["Step", "Rule"]}
            rows={[
              ["1. Links", "group by (src chip, dst chip); per src chip and neighbor mesh keep one: most channels, best tier, lowest dst chip id (replaces the fatal)"],
              ["2. Demand", "device-level pins first, boundary-major; then mesh-level strict, then up to relaxed more, round-robin, as today"],
              ["3. Directions", "today's phases (Z claim, N/E/S/W, overflow onto Z); within a phase take the link whose directions have the fewest channels assigned, then best tier, then lowest chip id"],
              ["4. Result", "STRICT channel unplaced: failure, FOOTPRINT if allowed channels are below the STRICT demand, else PLACEMENT"],
            ]}
          />
          <Text size="small" tone="tertiary">
            With one candidate link per choice, the common case, step 3 picks the same channels as today.
          </Text>
        </Stack>
      </Grid>
      <H3>Keeping the seat solve and the binding in step</H3>
      <Text tone="secondary">
        Decision: no channel selection inside the seat solve. It works on footprints, not chips, and which channels can be bound
        depends on exit chips the intra-mesh solve picks later; encoding it means a joint seat, intra-mesh and channel solve, the
        largest problem in the stack, on every solve. Instead, three cheap necessary conditions, and the exact binding after each
        placement.
      </Text>
      <Table
        headers={["Condition", "Where", "Implementation"]}
        rows={[
          ["1. Placeable link count per seat pair", "AdjacencyMatrix::build (physical_grouping_descriptor_matching.cpp:2434)", "Candidate keeps each boundary link's src chip (boundary_link_src_dense_) and only allowed cable types; per (from seat, to seat) group links by (src chip, dst chip), keep the largest group per src chip, cap at max_ports_per_direction(mesh), sum. Orientation-free max keeps it necessary. Same clauses, smaller numbers"],
          ["2. Aggregate per seat", "build_sat_placement_constraints (:2732)", "for each seat of mesh m: placeable channels to all neighbors must cover the sum of m's STRICT demands. Unary, so failing seats become forbidden seats; no SAT constraint"],
          ["3. Exit-node capacity", "add_exit_node_constraints (topology_mapper_utils.cpp:955)", "per exit ASIC count min(links to its one dst chip, ports per direction) instead of max_physical_exit_edges_per_asic_toward_mesh (:1110); exact per fabric node, since the node is known here"],
        ]}
      />
      <H3>The backstop</H3>
      <Text tone="secondary">
        In MultiMeshSolutionEnumerator::next (topology_mapper_utils.cpp:1770), after complete_intra_mesh_for_placement succeeds,
        call resolve_inter_mesh_binding. Today a failed intra-mesh solve forbids that mesh's footprint everywhere
        (add_forbidden_constraint, :1830), which is too strong for either kind of binding failure.
      </Text>
      <Table
        headers={["Failure kind", "Action", "Why"]}
        rows={[
          ["FOOTPRINT", "add_forbidden_seat_pair(a, footprint_a, b, footprint_b)", "depends only on the two footprints; each stays usable next to other neighbors"],
          ["PLACEMENT", "exclude_mapping(seating), the existing whole-seating exclusion (:3452)", "another chip mapping of the same footprints might bind"],
        ]}
      />
      <Grid columns={fit(440)} gap={16} align="start">
        <Stack gap={8}>
          <H3>New primitive: partial-mapping blocking clause</H3>
          <Text tone="secondary">
            A pair block forbids two variables together without saying anything about the other meshes. Nothing does that
            today: the SAT blocking clause (topology_sat_add_blocking_clause_for_mapping_impl, topology_solver_sat.cpp:329) spans
            every target and returns false on an unmapped entry (:361), and DFS mapping_is_excluded (topology_solver.tpp:3416)
            compares whole mappings.
          </Text>
        </Stack>
        <Table
          headers={["Piece", "Change"]}
          rows={[
            ["SAT", "topology_sat_add_partial_blocking_clause: one negative literal per mapped entry, unmapped skipped, at least one required; ignores unique_shapes"],
            ["DFS", "blocked_partial_mappings_, subset match during the search, pruned once both meshes are assigned"],
            ["Session", "TopologyMappingEnumerationSession::exclude_partial_mapping, routed to both engines"],
            ["Placement session", "forbidden_seat_pairs_ stored as footprints like extra_forbidden_; replayed in MasterSolve::restart (:3258) next to excluded_seat_maps(); one clause per (seat_a, seat_b) from seats_matching"],
            ["Stats", "PlacementSolveStats gains binding_rejections_footprint, binding_rejections_placement, binding_elapsed"],
          ]}
        />
      </Grid>
      <Grid columns={fit(440)} gap={16} align="start">
        <CodeCard
          added
          path="control_plane.cpp"
          src={`
void ControlPlane::generate_intermesh_connectivity() {
    const auto& plan = topology_mapper_->get_inter_mesh_binding();
    enforce_inter_mesh_shortfall_policy(plan);  // STRICT throws (FSD: records)
    auto intermesh_connections = plan.to_annotated_connections();
    apply_intermesh_connections_to_local_chips(intermesh_connections);
    validate_requested_intermesh_connections(intermesh_connections);
    rebuild_intermesh_exit_maps_from_connections(intermesh_connections);
    routing_table_generator_->load_intermesh_connections(intermesh_connections);
}`}
        />
        <Stack gap={8}>
          <H3>When to pick it up</H3>
          <Table
            headers={["Any one of"]}
            rows={[
              ["part 9 is in and bindings show overflow, congestion or contention"],
              ["a solve objective needs directions, e.g. consistent directions around a ring"],
              ["a consumer needs channels and directions at mapping time, e.g. a pipeline builder"],
            ]}
          />
          <Text size="small" tone="tertiary">
            Cost to weigh: the mapper's output, broadcast and goldens then carry logical ports, so a port-layout change (torus
            reservations, Z ports, fabric config) shows up as a mapping change.
          </Text>
        </Stack>
      </Grid>
      <H3>Implementation sketch (not scheduled)</H3>
      <Table
        headers={["Area", "Change"]}
        rows={[
          ["New", "inter_mesh_binding.hpp/.cpp: types, PortCapacityModel, resolve_inter_mesh_binding (body moved from pair_logical_intermesh_ports)"],
          ["Solver", "partial blocking clause in SAT; subset-match blocks in DFS; exclude_partial_mapping"],
          ["Seat solve", "placeable counts in AdjacencyMatrix::build; per-seat filter; forbidden_seat_pairs_ and replay; add_forbidden_seat_pair"],
          ["Mapper", "capacity-aware add_exit_node_constraints; binding after each intra-mesh success; failure kind to pair block or seating exclusion; stats; plan in the chip-info broadcast; get_inter_mesh_binding(); yaml dump"],
          ["Control plane", "the function above; pairing, its broadcast and the single-host greedy removed"],
          ["Tests", "parity with part 9's pairing when there is one link per choice; congestion spreads over exit chips; shared exit chip serves both boundaries; seat-solve conditions; pair block leaves each seat usable elsewhere; SAT and DFS agree on partial blocks across a session rebuild"],
        ]}
      />
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
          ["10", "Who signs off on default_port_type_preference()? Is the #36425 order right, and where do the linking boards go? Per arch or cluster type?"],
          ["11", "Today node hits outrank realized edges, so keeping today's results puts both cable strengths below node hits. Keep that (proposed), or move node hits below realized edges so an explicit MGD cable preference can outrank a PGD-preferred layout, changing soft solves with pinnings?"],
          ["12", "The candidate table lives on rank 0, where the pairing runs. Also send it with the chip-info broadcast so every rank can log or dump it, or keep it rank-0 only (proposed)?"],
          ["13", "Stretch goal (part 11), if picked up: is the per-seat filter enough to keep binding rejections rare on SC16 / SC20 and the dual- and quad-galaxy systems, or does it need capacity per exit chip from the start?"],
        ]}
      />
    </Stack>
  );
}

const SECTIONS = [
  { id: "overview", label: "Overview", body: OverviewTab },
  { id: "today", label: "How the policy flows today", body: TodayTab },
  { id: "semantics", label: "Semantics per connection", body: SemanticsTab },
  { id: "solver", label: "Solver: STRICT / RELAXED as a per-edge channel requirement, not a solver-wide mode", body: SolverTab },
  { id: "changes", label: "MGD, mapper and control plane", body: ChangesTab },
  { id: "cable", label: "Cable type: constraints by port type", body: CableTab },
  { id: "implications", label: "Implications", body: ImplicationsTab },
  { id: "plan", label: "Implementation plan", body: PlanTab },
  { id: "candidates", label: "Inter-mesh candidate channels from the mapper", body: CandidatesTab },
  { id: "sat", label: "SAT encoding and constraint API in detail", body: SatTab },
  { id: "stretch", label: "Stretch goal (experimental): channels and directions in the mapper", body: StretchTab },
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
