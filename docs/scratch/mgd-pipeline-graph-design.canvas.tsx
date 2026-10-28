// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  DiffStats,
  DiffView,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Table,
  Text,
  computeDAGLayout,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";
import type { DiffLineData } from "cursor/canvas";

// ---------------------------------------------------------------------------
// Code helpers
// ---------------------------------------------------------------------------

/** Lines prefixed "+", "-" or " " become a unified diff. */
function diffLines(src: string): DiffLineData[] {
  const out: DiffLineData[] = [];
  let n = 1;
  const lines = src.replace(/^\n/, "").replace(/\n\s*$/, "").split("\n");
  for (const raw of lines) {
    const tag = raw.charAt(0);
    const body = raw.slice(1);
    if (tag === "+") out.push({ type: "added", content: body, lineNumber: n++ });
    else if (tag === "-") out.push({ type: "removed", content: body });
    else out.push({ type: "unchanged", content: body, lineNumber: n++ });
  }
  return out;
}

function plainLines(src: string): DiffLineData[] {
  return src
    .replace(/^\n/, "")
    .replace(/\n\s*$/, "")
    .split("\n")
    .map((content, i) => ({ type: "unchanged" as const, content, lineNumber: i + 1 }));
}

function newLines(src: string): DiffLineData[] {
  return plainLines(src).map((l) => ({ ...l, type: "added" as const }));
}

type CodeMode = "diff" | "new" | "plain";

function CodeCard(props: { path: string; src: string; mode: CodeMode; language?: string; note?: string }) {
  const lines =
    props.mode === "diff" ? diffLines(props.src) : props.mode === "new" ? newLines(props.src) : plainLines(props.src);
  const adds = lines.filter((l) => l.type === "added").length;
  const dels = lines.filter((l) => l.type === "removed").length;
  return (
    <Card>
      <CardHeader
        trailing={
          props.mode === "plain" ? (
            <Text size="small" tone="tertiary">
              reference
            </Text>
          ) : (
            <DiffStats additions={adds} deletions={dels} />
          )
        }
      >
        {props.path}
      </CardHeader>
      <CardBody style={{ padding: 0 }}>
        <DiffView
          lines={lines}
          path={props.path}
          language={props.language}
          showAccentStrip={props.mode !== "plain"}
          coloredLineNumbers={props.mode !== "plain"}
        />
        {props.note ? (
          <div style={{ padding: "8px 12px" }}>
            <Text size="small" tone="secondary">
              {props.note}
            </Text>
          </div>
        ) : null}
      </CardBody>
    </Card>
  );
}

// ---------------------------------------------------------------------------
/** Grid columns that wrap to fewer columns when the panel is narrow. */
function fit(minPx: number) {
  return `repeat(auto-fit, minmax(min(100%, ${minPx}px), 1fr))`;
}

function ScrollX({ children }: { children: Parameters<typeof Stack>[0]["children"] }) {
  return <div style={{ overflowX: "auto", maxWidth: "100%" }}>{children}</div>;
}

// ---------------------------------------------------------------------------
// SVG primitives
// ---------------------------------------------------------------------------

function ArrowDefs({ id, color }: { id: string; color: string }) {
  return (
    <defs>
      <marker id={id} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
      </marker>
    </defs>
  );
}

type BoxKind = "existing" | "new" | "reuse" | "fail" | "muted";

function useBoxStyle() {
  const t = useHostTheme();
  return (kind: BoxKind) => {
    switch (kind) {
      case "new":
        return { fill: t.fill.secondary, stroke: t.accent.primary, text: t.text.primary, dash: undefined as string | undefined };
      case "reuse":
        return { fill: t.fill.tertiary, stroke: t.category.green, text: t.text.primary, dash: undefined };
      case "fail":
        return { fill: t.fill.quaternary, stroke: t.category.red, text: t.text.primary, dash: "4 3" };
      case "muted":
        return { fill: "none", stroke: t.stroke.secondary, text: t.text.tertiary, dash: "4 3" };
      default:
        return { fill: t.fill.quaternary, stroke: t.stroke.primary, text: t.text.secondary, dash: undefined };
    }
  };
}

function SvgBox(props: {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  kind?: BoxKind;
}) {
  const style = useBoxStyle()(props.kind ?? "existing");
  const t = useHostTheme();
  const cx = props.x + props.w / 2;
  return (
    <g>
      <rect
        x={props.x}
        y={props.y}
        width={props.w}
        height={props.h}
        rx={6}
        fill={style.fill}
        stroke={style.stroke}
        strokeDasharray={style.dash}
        strokeWidth={1.2}
      />
      <text
        x={cx}
        y={props.sub ? props.y + props.h / 2 - 3 : props.y + props.h / 2 + 4}
        textAnchor="middle"
        fontSize={12}
        fontWeight={600}
        fill={style.text}
      >
        {props.title}
      </text>
      {props.sub ? (
        <text x={cx} y={props.y + props.h / 2 + 12} textAnchor="middle" fontSize={10.5} fill={t.text.tertiary}>
          {props.sub}
        </text>
      ) : null}
    </g>
  );
}

function Legend() {
  const t = useHostTheme();
  const box = useBoxStyle();
  const items: Array<[BoxKind, string]> = [
    ["existing", "exists today, unchanged"],
    ["reuse", "exists, reused or refactored"],
    ["new", "new in this design"],
    ["fail", "failure / feedback path"],
  ];
  return (
    <Row gap={16} wrap>
      {items.map(([k, label]) => {
        const s = box(k);
        return (
          <div key={k}>
            <Row gap={6} align="center">
              <svg width={22} height={14}>
                <rect x={1} y={1} width={20} height={12} rx={3} fill={s.fill} stroke={s.stroke} strokeDasharray={s.dash} />
              </svg>
              <Text size="small" tone="secondary">
                {label}
              </Text>
            </Row>
          </div>
        );
      })}
      <Text size="small" tone="tertiary" style={{ color: t.text.tertiary }}>
        Boxes carry the PR id that builds them.
      </Text>
    </Row>
  );
}

// ---------------------------------------------------------------------------
// Overview diagrams
// ---------------------------------------------------------------------------

function FlowCompare() {
  const t = useHostTheme();
  const W = 940;
  const bw = 150;
  const gap = 34;
  const xs = [0, 1, 2, 3, 4].map((i) => 10 + i * (bw + gap));
  const arrow = (x1: number, y: number, x2: number) => (
    <line x1={x1} y1={y} x2={x2} y2={y} stroke={t.stroke.primary} strokeWidth={1.3} markerEnd="url(#fc-arrow)" />
  );
  return (
    <ScrollX>
      <svg width="100%" viewBox={`0 0 ${W} 300`} style={{ maxWidth: W, minWidth: 0.8 * (W) }}>
        <ArrowDefs id="fc-arrow" color={t.stroke.primary} />
        <text x={10} y={18} fontSize={12} fontWeight={600} fill={t.text.secondary}>
          TODAY
        </text>
        <SvgBox x={xs[0]} y={28} w={bw} h={46} title="MGD" sub="meshes + connections" />
        <SvgBox x={xs[1]} y={28} w={bw} h={46} title="Topology mapper" sub="commits a seating blind" />
        <SvgBox x={xs[2]} y={28} w={bw} h={46} title="Control plane" sub="binds lowest-id cables" />
        <SvgBox x={xs[3]} y={28} w={bw} h={46} title="blaze carve + resolve" sub="first chance to check" />
        <SvgBox x={xs[4]} y={28} w={bw} h={46} title="no valid submesh" sub="hard failure" kind="fail" />
        {arrow(xs[0] + bw, 51, xs[1])}
        {arrow(xs[1] + bw, 51, xs[2])}
        {arrow(xs[2] + bw, 51, xs[3])}
        {arrow(xs[3] + bw, 51, xs[4])}
        <path
          d={`M ${xs[4] + bw / 2} 74 L ${xs[4] + bw / 2} 104 L ${xs[1] + bw / 2} 104 L ${xs[1] + bw / 2} 78`}
          fill="none"
          stroke={t.category.red}
          strokeDasharray="4 3"
          markerEnd="url(#fc-arrow)"
        />
        <text x={(xs[1] + xs[4] + bw) / 2} y={120} textAnchor="middle" fontSize={11} fill={t.text.tertiary}>
          recourse: generate_rank_bindings --all-solutions, then launch each mapping on hardware until one builds
        </text>

        <text x={10} y={160} fontSize={12} fontWeight={600} fill={t.accent.primary}>
          PROPOSED
        </text>
        <SvgBox x={xs[0]} y={170} w={bw} h={46} title="MGD + pipeline" sub="stages, shapes, edges" kind="new" />
        <SvgBox x={xs[1]} y={170} w={bw} h={46} title="Topology mapper" sub="footprint solve + resolver per seating" kind="new" />
        <SvgBox x={xs[2]} y={170} w={bw} h={46} title="Control plane" sub="binds the links it needs" kind="new" />
        <SvgBox x={xs[3]} y={170} w={bw} h={46} title="blaze carve + resolve" sub="unchanged" />
        <SvgBox x={xs[4]} y={170} w={bw} h={46} title="pipeline builds" sub="a placement is known to exist" kind="reuse" />
        {arrow(xs[0] + bw, 193, xs[1])}
        {arrow(xs[1] + bw, 193, xs[2])}
        {arrow(xs[2] + bw, 193, xs[3])}
        {arrow(xs[3] + bw, 193, xs[4])}
        <path
          d={`M ${xs[1] + 30} 216 C ${xs[1] + 30} 262, ${xs[1] + bw - 30} 262, ${xs[1] + bw - 30} 220`}
          fill="none"
          stroke={t.category.red}
          strokeDasharray="4 3"
          markerEnd="url(#fc-arrow)"
        />
        <text x={xs[1] + bw / 2} y={276} textAnchor="middle" fontSize={11} fill={t.text.tertiary}>
          reject seating, ask SAT for the next
        </text>
      </svg>
    </ScrollX>
  );
}

function TierTimeline() {
  const t = useHostTheme();
  const W = 940;
  const cols = [
    { title: "MGD ctor", sub: "each part, before merge", tier: "Tier A + B", what: "that part's meshes and ranks", kind: "new" as BoxKind },
    { title: "merge ctor", sub: "MeshGraphDescriptor::merge", tier: "Tier A + B", what: "merged meshes, all pipelines", kind: "new" as BoxKind },
    { title: "SAT seating loop", sub: "MultiMeshSolutionEnumerator", tier: "Tier C", what: "PSD + each candidate", kind: "new" as BoxKind },
    { title: "inter-mesh binding", sub: "ControlPlane init", tier: "4.2", what: "binds Tier C's links first", kind: "new" as BoxKind },
    { title: "pipeline build", sub: "blaze build_topology", tier: "Tier D", what: "exists, unchanged", kind: "existing" as BoxKind },
  ];
  const bw = 160;
  const gap = 24;
  return (
    <ScrollX>
      <svg width="100%" viewBox={`0 0 ${W} 150`} style={{ maxWidth: W, minWidth: 0.8 * (W) }}>
        <ArrowDefs id="tt-arrow" color={t.stroke.primary} />
        <line x1={10} y1={34} x2={W - 10} y2={34} stroke={t.stroke.tertiary} />
        <text x={10} y={20} fontSize={11} fill={t.text.tertiary}>
          time, left to right: everything before "inter-mesh binding" can still steer the mapping
        </text>
        {cols.map((c, i) => {
          const x = 10 + i * (bw + gap);
          return (
            <g key={c.title}>
              <SvgBox x={x} y={48} w={bw} h={46} title={c.title} sub={c.sub} kind={c.kind} />
              {i < cols.length - 1 ? (
                <line x1={x + bw} y1={71} x2={x + bw + gap} y2={71} stroke={t.stroke.primary} markerEnd="url(#tt-arrow)" />
              ) : null}
              <text x={x + bw / 2} y={114} textAnchor="middle" fontSize={12} fontWeight={600} fill={i === 2 ? t.accent.primary : t.text.primary}>
                {c.tier}
              </text>
              <text x={x + bw / 2} y={130} textAnchor="middle" fontSize={11} fill={t.text.tertiary}>
                {c.what}
              </text>
            </g>
          );
        })}
      </svg>
    </ScrollX>
  );
}

// ---------------------------------------------------------------------------
// Mesh drawings
// ---------------------------------------------------------------------------

function SingleMeshRing() {
  const t = useHostTheme();
  const cell = 26;
  const ox = 30;
  const oy = 20;
  const tiles = [
    { r: 0, c: 0, label: "s0 Embedding" },
    { r: 0, c: 1, label: "s1 Windowed" },
    { r: 1, c: 1, label: "s2 Global" },
    { r: 1, c: 0, label: "s3 LMHead" },
  ];
  const tw = 2 * cell;
  const th = 4 * cell;
  const center = (r: number, c: number) => ({ x: ox + c * tw + tw / 2, y: oy + r * th + th / 2 });
  const order = [0, 1, 2, 3, 0];
  return (
    <svg width={260} height={260}>
      <ArrowDefs id="smr-arrow" color={t.accent.primary} />
      {Array.from({ length: 8 }).map((_, r) =>
        Array.from({ length: 4 }).map((__, c) => (
          <rect
            key={`${r}-${c}`}
            x={ox + c * cell + 2}
            y={oy + r * cell + 2}
            width={cell - 4}
            height={cell - 4}
            rx={3}
            fill={t.fill.tertiary}
            stroke={t.stroke.tertiary}
          />
        )),
      )}
      {tiles.map((tl) => (
        <rect
          key={tl.label}
          x={ox + tl.c * tw}
          y={oy + tl.r * th}
          width={tw}
          height={th}
          fill="none"
          stroke={t.stroke.primary}
          strokeWidth={1.5}
        />
      ))}
      {order.slice(0, 4).map((from, i) => {
        const a = center(tiles[from].r, tiles[from].c);
        const b = center(tiles[order[i + 1]].r, tiles[order[i + 1]].c);
        const dx = (b.x - a.x) * 0.28;
        const dy = (b.y - a.y) * 0.28;
        return (
          <line
            key={i}
            x1={a.x + dx}
            y1={a.y + dy}
            x2={b.x - dx}
            y2={b.y - dy}
            stroke={t.accent.primary}
            strokeWidth={1.8}
            markerEnd="url(#smr-arrow)"
          />
        );
      })}
      {tiles.map((tl) => {
        const p = center(tl.r, tl.c);
        const [id, ...rest] = tl.label.split(" ");
        return (
          <g key={`l-${tl.label}`}>
            <rect x={p.x - 24} y={p.y - 30} width={48} height={18} rx={4} fill={t.bg.editor} />
            <text x={p.x} y={p.y - 17} textAnchor="middle" fontSize={12} fontWeight={600} fill={t.text.primary}>
              {id}
            </text>
            <text x={p.x} y={p.y + 30} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
              {rest.join(" ")}
            </text>
          </g>
        );
      })}
      <text x={ox + 2 * tw + 10} y={oy + 12} fontSize={10} fill={t.text.tertiary}>
        8x4 mesh
      </text>
      <text x={ox + 2 * tw + 10} y={oy + 26} fontSize={10} fill={t.text.tertiary}>
        host_topology 2x2
      </text>
      <text x={ox + 2 * tw + 10} y={oy + 40} fontSize={10} fill={t.text.tertiary}>
        = four 4x2 slices
      </text>
    </svg>
  );
}

type CableKind = "bound" | "candidate" | "required" | "unused";
type Cable = { side: "left" | "right"; f: number; kind: CableKind; label?: string };

/** M0 on top (slices a0 | a1), M1 below (b0 | b1), cables in between. */
function TwoMesh(props: {
  labels: { a0: string; a1: string; b0: string; b1: string };
  cables: Cable[];
  hostEdge?: { from: "a0" | "a1" | "b0" | "b1"; to: "a0" | "a1" | "b0" | "b1"; label: string };
  width?: number;
}) {
  const t = useHostTheme();
  const W = 440;
  const H = 260;
  const mx = 20;
  const mw = 400;
  const half = mw / 2;
  const y0 = 22;
  const y1 = 162;
  const mh = 76;
  const sliceCenter = (s: "a0" | "a1" | "b0" | "b1") => ({
    x: mx + (s.endsWith("0") ? half / 2 : half + half / 2),
    y: (s.startsWith("a") ? y0 : y1) + mh / 2,
  });
  const cableColor = (k: CableKind) =>
    k === "required" ? t.accent.primary : k === "bound" ? t.text.primary : k === "candidate" ? t.text.tertiary : t.stroke.secondary;
  return (
    <ScrollX>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: props.width ?? W, minWidth: 0.8 * (props.width ?? W) }}>
        <ArrowDefs id="tm-arrow" color={t.category.orange} />
        {(["a", "b"] as const).map((m) => {
          const y = m === "a" ? y0 : y1;
          return (
            <g key={m}>
              <rect x={mx} y={y} width={mw} height={mh} rx={6} fill={t.fill.quaternary} stroke={t.stroke.primary} />
              <line x1={mx + half} y1={y} x2={mx + half} y2={y + mh} stroke={t.stroke.secondary} strokeDasharray="3 3" />
              <text x={mx + 6} y={y - 6} fontSize={11} fontWeight={600} fill={t.text.secondary}>
                {m === "a" ? "M0" : "M1"}
              </text>
            </g>
          );
        })}
        {(["a0", "a1", "b0", "b1"] as const).map((s) => {
          const c = sliceCenter(s);
          return (
            <g key={s}>
              <text x={c.x} y={c.y - 6} textAnchor="middle" fontSize={10.5} fill={t.text.tertiary}>
                slice {s}
              </text>
              <text x={c.x} y={c.y + 12} textAnchor="middle" fontSize={13} fontWeight={600} fill={t.text.primary}>
                {props.labels[s]}
              </text>
            </g>
          );
        })}
        {props.cables.map((cb, i) => {
          const x = mx + (cb.side === "left" ? 0 : half) + half * cb.f;
          const color = cableColor(cb.kind);
          return (
            <g key={i}>
              <line
                x1={x}
                y1={y0 + mh}
                x2={x}
                y2={y1}
                stroke={color}
                strokeWidth={cb.kind === "required" || cb.kind === "bound" ? 2.4 : 1.4}
                strokeDasharray={cb.kind === "candidate" || cb.kind === "unused" ? "4 3" : undefined}
              />
              {cb.label ? (
                <text x={x + 5} y={(y0 + mh + y1) / 2 + 4} fontSize={10} fill={color}>
                  {cb.label}
                </text>
              ) : null}
            </g>
          );
        })}
        {props.hostEdge
          ? (() => {
              const a = sliceCenter(props.hostEdge.from);
              const b = sliceCenter(props.hostEdge.to);
              const outX = W - 6;
              return (
                <g>
                  <path
                    d={`M ${mx + mw} ${a.y} C ${outX} ${a.y}, ${outX} ${b.y}, ${mx + mw + 2} ${b.y}`}
                    fill="none"
                    stroke={t.category.orange}
                    strokeWidth={1.8}
                    strokeDasharray="5 3"
                    markerEnd="url(#tm-arrow)"
                  />
                  <text x={mx + mw - 4} y={(a.y + b.y) / 2 + 4} textAnchor="end" fontSize={10.5} fill={t.category.orange}>
                    {props.hostEdge.label}
                  </text>
                </g>
              );
            })()
          : null}
      </svg>
    </ScrollX>
  );
}

// ---------------------------------------------------------------------------
// Worked example: cases A/B/C with every edge STRICT, and with s3 > s0 RELAXED
// ---------------------------------------------------------------------------

type CaseId = "A" | "B" | "C";
type Mode = "all STRICT" | "s3 > s0 RELAXED (later)";

const CASES: Record<
  CaseId,
  {
    title: string;
    cables: Cable[];
    strict: { ok: boolean; why: string };
    relaxed: { ok: boolean; why: string; labels: { a0: string; a1: string; b0: string; b1: string }; host?: boolean };
  }
> = {
  A: {
    title: "2 links on 2 different slice pairs",
    cables: [
      { side: "left", f: 0.5, kind: "bound", label: "a0-b0" },
      { side: "right", f: 0.5, kind: "bound", label: "a1-b1" },
    ],
    strict: { ok: true, why: "Slices form the cycle a0 > a1 > b1 > b0 > a0. plan_ring_carve finds it and the resolver lays the ring on it." },
    relaxed: {
      ok: true,
      why: "Same as all STRICT: an all-fabric cycle exists, so the soft edge is matched and nothing goes to host.",
      labels: { a0: "s0", a1: "s1", b0: "s3", b1: "s2" },
    },
  },
  B: {
    title: "2 links, both on the same slice pair",
    cables: [
      { side: "left", f: 0.35, kind: "bound", label: "a0-b0" },
      { side: "left", f: 0.7, kind: "bound", label: "a0-b0" },
    ],
    strict: {
      ok: false,
      why: "Adjacency is a1-a0, a0-b0, b0-b1: a path, not a cycle. The footprint solve finds no 4-cycle carve (blaze's planner: no-cycle-of-length-4).",
    },
    relaxed: {
      ok: true,
      why: "With s3 > s0 a soft edge, the solve maps s0..s3 along the fabric path a1 > a0 > b0 > b1 and leaves only the loopback to host, the same cut blaze's plan_ring_placement makes. Allowed: it is RELAXED and not stage 0's outgoing edge.",
      labels: { a0: "s1", a1: "s0", b0: "s2", b1: "s3" },
      host: true,
    },
  },
  C: {
    title: "1 link",
    cables: [{ side: "left", f: 0.5, kind: "bound", label: "a0-b0" }],
    strict: { ok: false, why: "Same adjacency as Case B: no 4-slice cycle. Rejected as a ring; an open chain (no loopback) would be accepted." },
    relaxed: {
      ok: true,
      why: "Same plan as Case B: one fabric chain a1 > a0 > b0 > b1, loopback over host.",
      labels: { a0: "s1", a1: "s0", b0: "s2", b1: "s3" },
      host: true,
    },
  },
};

function CaseExplorer() {
  const t = useHostTheme();
  const [caseId, setCaseId] = useCanvasState<CaseId>("case", "B");
  const [mode, setMode] = useCanvasState<Mode>("mode", "all STRICT");
  const c = CASES[caseId];
  const verdict = mode === "all STRICT" ? c.strict : c.relaxed;
  const labels = mode === "all STRICT" ? { a0: "s0", a1: "s1", b0: "s3", b1: "s2" } : c.relaxed.labels;
  return (
    <Grid columns={fit(360)} gap={24} align="start">
      <Stack gap={8}>
        <TwoMesh
          labels={labels}
          cables={c.cables}
          hostEdge={mode === "s3 > s0 RELAXED (later)" && c.relaxed.host ? { from: "b1", to: "a1", label: "s3 > s0 over host" } : undefined}
        />
        <Text size="small" tone="tertiary">
          mesh4x4_stage4x2 from the blaze sweep: two 4x4 meshes, host_topology [1, 2], 4-stage ring of 4x2 stages.
        </Text>
      </Stack>
      <Stack gap={12}>
        <Row gap={8} wrap>
          {(["A", "B", "C"] as CaseId[]).map((id) => (
            <span key={id}>
              <Pill active={id === caseId} onClick={() => setCaseId(id)}>
                Case {id}
              </Pill>
            </span>
          ))}
          <div style={{ width: 12 }} />
          {(["all STRICT", "s3 > s0 RELAXED (later)"] as Mode[]).map((m) => (
            <span key={m}>
              <Pill active={m === mode} onClick={() => setMode(m)}>
                {m}
              </Pill>
            </span>
          ))}
        </Row>
        <H3>{c.title}</H3>
        <Row gap={8} align="center">
          <span
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: 5,
              background: verdict.ok ? t.category.green : t.category.red,
            }}
          />
          <Text weight="semibold">{verdict.ok ? "Tier C accepts this seating" : "Tier C rejects this seating"}</Text>
        </Row>
        <Text tone="secondary">{verdict.why}</Text>
        <Divider />
        <Text size="small" tone="tertiary">
          Tier B only sees the count of declared channels (2 in every case). Only Tier C sees where the bound links land,
          because that depends on the seats SAT picks. In the first version the exact check rejects Case B. Plan 1 (later, optional) could
          reject it before the solve, but only when hard pinnings fix the chip layout.
        </Text>
      </Stack>
    </Grid>
  );
}

// ---------------------------------------------------------------------------
// Tier C loop flowchart
// ---------------------------------------------------------------------------

function LoopFlow() {
  const t = useHostTheme();
  const W = 940;
  const bx = 170;
  const bw = 380;
  const bh = 44;
  const gap = 22;
  const steps: Array<{ title: string; sub: string; kind: BoxKind }> = [
    { title: "1. SAT seating", sub: "placement_session_->next()", kind: "existing" },
    { title: "2. intra-mesh completion", sub: "complete_intra_mesh_for_placement", kind: "existing" },
    { title: "C1. build the physical pipeline graph", sub: "one position graph per seating (overlapping positions)  [P5]", kind: "new" },
    { title: "C2. pipeline solve", sub: "one solve, one carve; resolver check; R = exact channels; R within demand  [P6]", kind: "reuse" },
    { title: "C3. on success: emit + publish R", sub: "mapping saved, R broadcast; control plane binds R first  [P8 P9 P12]", kind: "new" },
  ];
  const y = (i: number) => 14 + i * (bh + gap);
  const H = y(steps.length) + 4;
  const c4x = bx + bw + 70;
  const c4w = 200;
  const c4y = y(3) - 4;
  const c4h = bh + 8;
  return (
    <ScrollX>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W, minWidth: 0.8 * W }}>
        <ArrowDefs id="lf-arrow" color={t.stroke.primary} />
        <ArrowDefs id="lf-red" color={t.category.red} />
        {steps.map((s, i) => (
          <g key={s.title}>
            <SvgBox x={bx} y={y(i)} w={bw} h={bh} title={s.title} sub={s.sub} kind={s.kind} />
            {i < steps.length - 1 ? (
              <line x1={bx + bw / 2} y1={y(i) + bh} x2={bx + bw / 2} y2={y(i + 1)} stroke={t.stroke.primary} markerEnd="url(#lf-arrow)" />
            ) : null}
          </g>
        ))}
        {/* intra-mesh failure: existing */}
        <path
          d={`M ${bx} ${y(1) + bh / 2} L ${bx - 50} ${y(1) + bh / 2} L ${bx - 50} ${y(0) + bh / 2} L ${bx} ${y(0) + bh / 2}`}
          fill="none"
          stroke={t.stroke.primary}
          strokeDasharray="4 3"
          markerEnd="url(#lf-arrow)"
        />
        <text x={bx - 56} y={y(1) + bh / 2 + 16} textAnchor="end" fontSize={10.5} fill={t.text.tertiary}>
          fail: forbid that seat
        </text>
        <text x={bx - 56} y={y(1) + bh / 2 + 29} textAnchor="end" fontSize={10.5} fill={t.text.tertiary}>
          (today)
        </text>
        {/* any C2 failure: C4 */}
        <line x1={bx + bw} y1={y(3) + bh / 2} x2={c4x} y2={y(3) + bh / 2} stroke={t.category.red} strokeDasharray="4 3" markerEnd="url(#lf-red)" />
        <text x={bx + bw + 6} y={y(3) + bh / 2 - 6} fontSize={10.5} fill={t.category.red}>
          no carve, resolver or demand fails
        </text>
        <SvgBox x={c4x} y={c4y} w={c4w} h={c4h} title="C4. on failure" sub="constraints for the next solve  [P8 P15]" kind="fail" />
        {/* C4 move 1: exit-node retry, back to intra-mesh */}
        <path
          d={`M ${c4x + 50} ${c4y} L ${c4x + 50} ${y(1) + bh / 2} L ${bx + bw + 2} ${y(1) + bh / 2}`}
          fill="none"
          stroke={t.category.red}
          strokeDasharray="4 3"
          markerEnd="url(#lf-red)"
        />
        <text x={c4x + 56} y={y(2) + 6} fontSize={10.5} fill={t.category.red}>
          1. retry the same seating:
        </text>
        <text x={c4x + 56} y={y(2) + 20} fontSize={10.5} fill={t.text.tertiary}>
          block the exit-node slice
        </text>
        <text x={c4x + 56} y={y(2) + 34} fontSize={10.5} fill={t.text.tertiary}>
          signature, at most 4 times
        </text>
        {/* C4 move 2: exclude the seating, back to SAT */}
        <path
          d={`M ${c4x + 190} ${c4y} L ${c4x + 190} ${y(0) + bh / 2} L ${bx + bw + 2} ${y(0) + bh / 2}`}
          fill="none"
          stroke={t.category.red}
          strokeDasharray="4 3"
          markerEnd="url(#lf-red)"
        />
        <text x={c4x + 196} y={y(0) + bh / 2 + 20} fontSize={10.5} fill={t.category.red}>
          2. then
        </text>
        <text x={c4x + 196} y={y(0) + bh / 2 + 34} fontSize={10.5} fill={t.text.tertiary}>
          exclude_mapping
        </text>
        <text x={c4x + 196} y={y(0) + bh / 2 + 48} fontSize={10.5} fill={t.text.tertiary}>
          (the seat set)
        </text>
      </svg>
    </ScrollX>
  );
}

// ---------------------------------------------------------------------------
// Carve path drawings
// ---------------------------------------------------------------------------

function TileGrid(props: {
  rows: number;
  cols: number;
  tile: [number, number];
  label: (r: number, c: number) => string;
  title: string;
  cell?: number;
  highlight?: (r: number, c: number) => boolean;
}) {
  const t = useHostTheme();
  const cell = props.cell ?? 14;
  const tr = props.rows / props.tile[0];
  const tc = props.cols / props.tile[1];
  const W = props.cols * cell + 4;
  const H = props.rows * cell + 22;
  return (
    <svg width={W} height={H}>
      <text x={2} y={11} fontSize={10.5} fontWeight={600} fill={t.text.secondary}>
        {props.title}
      </text>
      <g transform="translate(2, 16)">
        {Array.from({ length: tr }).map((_, r) =>
          Array.from({ length: tc }).map((__, c) => {
            const on = props.highlight ? props.highlight(r, c) : true;
            return (
              <g key={`${r}-${c}`}>
                <rect
                  x={c * props.tile[1] * cell}
                  y={r * props.tile[0] * cell}
                  width={props.tile[1] * cell}
                  height={props.tile[0] * cell}
                  fill={on ? t.fill.secondary : t.fill.quaternary}
                  stroke={t.stroke.primary}
                />
                <text
                  x={c * props.tile[1] * cell + (props.tile[1] * cell) / 2}
                  y={r * props.tile[0] * cell + (props.tile[0] * cell) / 2 + 4}
                  textAnchor="middle"
                  fontSize={props.tile[0] * cell < 20 ? 8 : 10.5}
                  fill={on ? t.text.primary : t.text.tertiary}
                >
                  {props.label(r, c)}
                </text>
              </g>
            );
          }),
        )}
      </g>
    </svg>
  );
}

function CarvePaths() {
  const t = useHostTheme();
  return (
    <Grid columns={fit(280)} gap={16} align="start">
      <Card>
        <CardHeader trailing={<Text size="small" tone="tertiary">no submesh_shape</Text>}>Planned carve</CardHeader>
        <CardBody>
          <Stack gap={10}>
            <Row gap={12}>
              <TileGrid rows={4} cols={4} tile={[4, 2]} title="M0" cell={20} label={(_, c) => (c === 0 ? "s0" : "s1")} />
              <TileGrid rows={4} cols={4} tile={[4, 2]} title="M1" cell={20} label={(_, c) => (c === 0 ? "s3" : "s2")} />
            </Row>
            <Text size="small" tone="secondary">
              Search the slice graph for a cycle of exactly the ring's length, then cut it into stage rectangles. Handles joint
              stages (a 4x4 stage = 2 slices), also on one mesh. Needs uniform slices, stage 0 on one slice, and a loopback.
            </Text>
            <Text size="small" tone="tertiary">
              plan_ring_carve / plan_fork_carve
            </Text>
          </Stack>
        </CardBody>
      </Card>
      <Card>
        <CardHeader trailing={<Text size="small" tone="tertiary">submesh_shape given</Text>}>Per-rank tiling</CardHeader>
        <CardBody>
          <Stack gap={10}>
            <Row gap={12}>
              <TileGrid rows={8} cols={4} tile={[4, 2]} title="M0 (4x2 tiles)" cell={10} label={(r, c) => `r${r * 2 + c}`} />
              <TileGrid rows={8} cols={4} tile={[1, 2]} title="M1 (1x2 tiles)" cell={10} label={() => ""} />
            </Row>
            <Text size="small" tone="secondary">
              Stamp one shape row-major over each mesh. Each rank keeps the first tile whose (0,0) chip it owns; identical chip
              sets collapse into a joint stage. The resolver then picks which stage goes on which tile.
            </Text>
            <Text size="small" tone="tertiary">
              create_submeshes + _collect_submeshes. The only path for llama_8b_pod (4x2 and 1x2 slices).
            </Text>
          </Stack>
        </CardBody>
      </Card>
      <Card>
        <CardHeader trailing={<Text size="small" tone="tertiary">policy: RELAXED</Text>}>Host cuts on RELAXED edges (later)</CardHeader>
        <CardBody>
          <Stack gap={10}>
            <svg width="100%" viewBox="0 0 260 96" style={{ maxWidth: 260 }}>
              <ArrowDefs id="fx-arrow" color={t.text.secondary} />
              <ArrowDefs id="fx-host" color={t.category.orange} />
              {["s0", "s1", "s2", "s3"].map((s, i) => (
                <g key={s}>
                  <rect x={8 + i * 62} y={20} width={46} height={30} rx={5} fill={t.fill.secondary} stroke={t.stroke.primary} />
                  <text x={31 + i * 62} y={40} textAnchor="middle" fontSize={12} fontWeight={600} fill={t.text.primary}>
                    {s}
                  </text>
                  {i < 3 ? (
                    <line x1={54 + i * 62} y1={35} x2={70 + i * 62} y2={35} stroke={t.text.secondary} markerEnd="url(#fx-arrow)" />
                  ) : null}
                </g>
              ))}
              <path d="M 217 50 C 217 88, 31 88, 31 54" fill="none" stroke={t.category.orange} strokeDasharray="5 3" markerEnd="url(#fx-host)" />
              <text x={124} y={92} textAnchor="middle" fontSize={10} fill={t.category.orange}>
                host MPI hop
              </text>
              <text x={124} y={12} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
                one fabric chain
              </text>
            </svg>
            <Text size="small" tone="secondary">
              For a pipeline with RELAXED edges. First try an all-fabric cycle. Otherwise cover the ring with the fewest fabric
              chains and join them over host, cutting only RELAXED edges. Then carve from the plan, or tile by the uniform stage
              shape, and resolve each chain on its own.
            </Text>
            <Text size="small" tone="tertiary">
              host_edge_planner.plan_ring_placement (tt-blaze #4513)
            </Text>
          </Stack>
        </CardBody>
      </Card>
    </Grid>
  );
}

// ---------------------------------------------------------------------------
// unique_shapes illustration
// ---------------------------------------------------------------------------

function UniqueShapesDiagram() {
  const t = useHostTheme();
  const seat = (x: number, y: number, seatId: string, mesh: string, tried: boolean) => (
    <g>
      <rect x={x} y={y} width={70} height={40} rx={5} fill={tried ? t.fill.secondary : t.fill.quaternary} stroke={tried ? t.stroke.primary : t.stroke.secondary} strokeDasharray={tried ? undefined : "4 3"} />
      <text x={x + 35} y={y + 17} textAnchor="middle" fontSize={11} fill={t.text.tertiary}>
        seat {seatId}
      </text>
      <text x={x + 35} y={y + 32} textAnchor="middle" fontSize={12} fontWeight={600} fill={tried ? t.text.primary : t.text.tertiary}>
        {mesh}
      </text>
    </g>
  );
  return (
    <ScrollX>
      <svg width="100%" viewBox="0 0 560 130" style={{ maxWidth: 560, minWidth: 0.8 * (560) }}>
        {seat(10, 30, "a", "M0", true)}
        {seat(90, 30, "b", "M1", true)}
        <text x={90} y={20} textAnchor="middle" fontSize={11} fill={t.text.secondary}>
          seating 1: tried
        </text>
        {seat(240, 30, "a", "M1", false)}
        {seat(320, 30, "b", "M0", false)}
        <text x={320} y={20} textAnchor="middle" fontSize={11} fill={t.text.secondary}>
          seating 2: swapped
        </text>
        <text x={460} y={46} fontSize={11} fill={t.text.secondary}>
          key = sorted seats
        </text>
        <text x={460} y={62} fontSize={12} fontWeight={600} fill={t.text.primary}>
          {"{a, b}"} for both
        </text>
        <text x={10} y={100} fontSize={11} fill={t.text.tertiary}>
          With unique_shapes on, seating 2 is never offered. But M0 and M1 can complete to different chip layouts (pinnings,
        </text>
        <text x={10} y={116} fontSize={11} fill={t.text.tertiary}>
          rank bindings and exit nodes are per mesh), so seating 2 may build when seating 1 does not.
        </text>
      </svg>
    </ScrollX>
  );
}

// ---------------------------------------------------------------------------
// Exit-node binding: today vs proposed
// ---------------------------------------------------------------------------

function BindingCompare() {
  const today: Cable[] = [
    { side: "left", f: 0.25, kind: "bound", label: "chip 0" },
    { side: "left", f: 0.6, kind: "bound", label: "chip 1" },
    { side: "right", f: 0.25, kind: "unused", label: "chip 2" },
    { side: "right", f: 0.82, kind: "unused", label: "chip 3" },
  ];
  const proposed: Cable[] = [
    { side: "left", f: 0.25, kind: "unused", label: "chip 0" },
    { side: "left", f: 0.6, kind: "required", label: "chip 1  (s3>s0)" },
    { side: "right", f: 0.25, kind: "required", label: "chip 2  (s1>s2)" },
    { side: "right", f: 0.82, kind: "unused", label: "chip 3" },
  ];
  return (
    <Stack gap={10}>
      <Grid columns={fit(360)} gap={24} align="start">
        <Stack gap={8}>
          <Row gap={8} align="center">
            <Pill size="sm">Today</Pill>
            <Text weight="semibold">Lowest chip ids win</Text>
          </Row>
          <TwoMesh labels={{ a0: "s0", a1: "s1", b0: "s3", b1: "s2" }} cables={today} />
          <Text tone="secondary">
            RELAXED count 2, four candidate cables. pair_logical_intermesh_ports orders links by (src node, dst node) and stops
            at the requested count. Chips 0 and 1 both sit in slice a0, so both channels land on the a0-b0 pair. That is Case B:
            the ring cannot close over fabric, even though cables on a1-b1 exist.
          </Text>
        </Stack>
        <Stack gap={8}>
          <Row gap={8} align="center">
            <Pill size="sm" active>
              Proposed
            </Pill>
            <Text weight="semibold">Bind the layout's cables first</Text>
          </Row>
          <TwoMesh labels={{ a0: "s0", a1: "s1", b0: "s3", b1: "s2" }} cables={proposed} />
          <Text tone="secondary">
            R = the cable behind each fabric crossing edge in the resolver's result: chip 1 for s3 {">"} s0 and chip 2 for s1{" "}
            {">"} s2. Phase 0 binds R, then today's phases fill whatever budget is left. Same count, different cables, and the
            ring closes (Case A).
          </Text>
        </Stack>
      </Grid>
      <Text size="small" tone="tertiary">
        Solid accent = required by the pipeline (R). Solid = bound. Dashed = gathered but not bound.
      </Text>
    </Stack>
  );
}

function PhaseStrip() {
  const t = useHostTheme();
  const phases: Array<{ title: string; sub: string; kind: BoxKind }> = [
    { title: "Phase 0", sub: "bind R (pipeline links)", kind: "new" },
    { title: "Phase 1a", sub: "assign_z pairs on Z", kind: "existing" },
    { title: "Phase 1b", sub: "other pairs on NESW", kind: "existing" },
    { title: "Phase 2", sub: "NESW overflow on Z", kind: "existing" },
    { title: "Fill order", sub: "today's order", kind: "muted" },
  ];
  const bw = 150;
  const gap = 22;
  return (
    <ScrollX>
      <svg width="100%" viewBox={`0 0 ${phases.length * (bw + gap)} 66`} style={{ maxWidth: phases.length * (bw + gap), minWidth: 0.8 * (phases.length * (bw + gap)) }}>
        <ArrowDefs id="ph-arrow" color={t.stroke.primary} />
        {phases.map((p, i) => (
          <g key={p.title}>
            <SvgBox x={i * (bw + gap) + 2} y={8} w={bw} h={44} title={p.title} sub={p.sub} kind={p.kind} />
            {i < phases.length - 2 ? (
              <line x1={i * (bw + gap) + bw + 2} y1={30} x2={(i + 1) * (bw + gap) + 2} y2={30} stroke={t.stroke.primary} markerEnd="url(#ph-arrow)" />
            ) : null}
          </g>
        ))}
      </svg>
    </ScrollX>
  );
}

// ---------------------------------------------------------------------------
// Merge diagram
// ---------------------------------------------------------------------------

function MergeDiagram() {
  const t = useHostTheme();
  return (
    <ScrollX>
      <svg width="100%" viewBox="0 0 700 150" style={{ maxWidth: 700, minWidth: 0.8 * (700) }}>
        <ArrowDefs id="mg-arrow" color={t.stroke.primary} />
        <SvgBox x={10} y={14} w={180} h={50} title="part 0 MGD" sub="M0, M1 + pipeline_descriptors" kind="new" />
        <SvgBox x={10} y={84} w={180} h={50} title="part 1 MGD" sub="M0, M1 + pipeline_descriptors" kind="new" />
        <SvgBox x={260} y={44} w={180} h={60} title="MeshGraphDescriptor::merge" sub="prefix + remap mesh ids" kind="reuse" />
        <SvgBox x={510} y={44} w={180} h={60} title="merged MGD" sub="M0..M3 + both pipelines" kind="new" />
        <line x1={190} y1={39} x2={260} y2={64} stroke={t.stroke.primary} markerEnd="url(#mg-arrow)" />
        <line x1={190} y1={109} x2={260} y2={84} stroke={t.stroke.primary} markerEnd="url(#mg-arrow)" />
        <line x1={440} y1={74} x2={510} y2={74} stroke={t.stroke.primary} markerEnd="url(#mg-arrow)" />
        <text x={350} y={130} textAnchor="middle" fontSize={10.5} fill={t.text.tertiary}>
          today: pipeline_descriptors silently dropped when there are 2+ parts
        </text>
      </svg>
    </ScrollX>
  );
}

// ---------------------------------------------------------------------------
// Implementation plan data
// ---------------------------------------------------------------------------

type Pr = {
  id: string;
  title: string;
  milestone: "M1 Format" | "M2 Shared code" | "M3 Tier C (STRICT)" | "M4 Pruning" | "M5 Later";
  size: "S" | "M" | "L";
  deps: string[];
  goal: string;
  files: string[];
  steps: string[];
  code: Array<{ path: string; src: string; mode: CodeMode; language?: string; note?: string }>;
  tests: string[];
  behaviour: string;
};

const PRS: Pr[] = [
  {
    id: "P1",
    title: "Schema + Tier A",
    milestone: "M1 Format",
    size: "M",
    deps: [],
    goal: "Add pipeline_descriptors to the MGD proto and validate it at parse time.",
    files: [
      "tt_metal/fabric/protobuf/mesh_graph_descriptor.proto",
      "tt_metal/fabric/mesh_graph_descriptor.cpp",
      "tt_metal/api/tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp",
    ],
    steps: [
      "Add messages PipelineDescriptor, PipelineStage, PipelineEdge; field 6 on MeshGraphDescriptor. Every edge is STRICT: PipelineEdge has optional Policy policy = 3, omitted or STRICT is accepted, and RELAXED is rejected by validation until P14. PipelineEdge also has bool loopback = 4, set on the edge that closes a cycle.",
      "Add validate_pipeline_descriptors to static_validate, after validate_pinnings.",
      "Add accessors has_pipeline_descriptors() and pipeline_descriptors().",
      "Store each pipeline as PipelineDescriptorData (string stage ids, textproto order). Each stored edge keeps its loopback mark. The LogicalPipelineGraph is built from them in Tier C (P5).",
    ],
    code: [
      {
        path: "tt_metal/fabric/protobuf/mesh_graph_descriptor.proto",
        mode: "diff",
        language: "proto",
        src: `
     // Top-level pinnings section (separate from descriptors and instances).
     // Maps logical fabric nodes to physical ASIC positions for pinning constraints.
     repeated AsicPinning pinnings = 5;
+
+    // Pipeline stage graphs the topology mapper must be able to place, all at once.
+    repeated PipelineDescriptor pipeline_descriptors = 6;
 }
+
+message PipelineDescriptor {
+    string name = 1;                   // unique across the (merged) MGD
+    repeated PipelineStage stages = 2;
+    repeated PipelineEdge edges = 3;
+}
+
+message PipelineStage {
+    string id = 1;              // unique, referenced by edges
+    string label = 2;           // free text
+    TorusTopology shape = 3;    // rank 2, (rows, cols) in logical mesh orientation
+}
+
+message PipelineEdge {
+    string src = 1;
+    string dst = 2;
+    optional Policy policy = 3;  // omitted or STRICT; RELAXED is rejected until P14
+    bool loopback = 4;           // closes a cycle; the other edges must have none
+}`,
      },
      {
        path: "tt_metal/fabric/mesh_graph_descriptor.cpp",
        mode: "diff",
        language: "cpp",
        src: `
         validate_graph_topology_and_connections(proto, all_errors);
         validate_pinnings(proto, all_errors);
+        validate_pipeline_descriptors(proto, all_errors);`,
      },
      {
        path: "tt_metal/fabric/mesh_graph_descriptor.cpp  (new function)",
        mode: "new",
        language: "cpp",
        src: `
void validate_pipeline_descriptors(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors) {
    std::unordered_set<std::string> names;
    for (const auto& p : proto.pipeline_descriptors()) {
        if (p.name().empty() || !names.insert(p.name()).second) {
            errors.push_back(fmt::format("pipeline name '{}' is empty or repeated", p.name()));
        }
        std::unordered_set<std::string> ids;
        for (const auto& s : p.stages()) {
            if (s.id().empty() || !ids.insert(s.id()).second) {
                errors.push_back(fmt::format("pipeline '{}': stage id '{}' is empty or repeated", p.name(), s.id()));
            }
            const auto& d = s.shape().dims();
            if (d.size() != 2 || d[0] <= 0 || d[1] <= 0) {
                errors.push_back(fmt::format("pipeline '{}': stage '{}' shape must be rank 2, dims > 0", p.name(), s.id()));
            }
        }
        for (const auto& e : p.edges()) {
            if (!ids.contains(e.src()) || !ids.contains(e.dst())) {
                errors.push_back(fmt::format("pipeline '{}': edge {} -> {} names an unknown stage", p.name(), e.src(), e.dst()));
            }
            if (e.src() == e.dst()) {
                errors.push_back(fmt::format("pipeline '{}': self-edge on '{}'", p.name(), e.src()));
            }
        }
        // The edges not marked loopback must have no cycle (a depth-first walk names the first one):
        //   "pipeline '{}': cycle {} has no loopback edge; mark one of its edges loopback: true"
        // and each loopback edge must close one, dst reaching src over the other edges:
        //   "pipeline '{}': loopback edge {} -> {} closes no cycle; {} does not reach {} over the other edges"
        // With no cycle, exactly one stage may have no incoming edge besides loopbacks (it owns host I/O):
        //   "pipeline '{}': stages {} have no incoming edge besides loopbacks; exactly one may, the stage that owns host I/O"
        // NOTE: otherwise no stage-topology check. In-degree, out-degree, and whether a branch is a
        // chain are all accepted.
    }
}`,
      },
    ],
    tests: [
      "Golden textprotos: GPT-OSS ring, llama_8b_pod (40 stages), joint 4x4 stage, dflash fork with both returns on fabric, two pipelines in one MGD.",
      "One negative textproto per rule (duplicate id, unknown stage, self-edge, rank-3 shape, two pipelines with one name). Two chains, and a stage with no edges beside a chain, are rejected (one host I/O stage). A stage with three outgoing edges, a branch stage with out-degree 2, and a dead-end branch all parse. An edge with policy: STRICT parses; policy: RELAXED is rejected.",
      "Loopback: a ring with no loopback edge, a fork with one unmarked return, and a loopback edge that closes no cycle are rejected. A loopback on any edge of the cycle (repeated, too) parses, and the mark is stored. Every ring and fork above marks its closing edges.",
      "Every existing MGD in tt_metal/fabric/mesh_graph_descriptors still parses unchanged.",
    ],
    behaviour: "No behaviour change without the new section.",
  },
  {
    id: "P2",
    title: "merge keeps the pipeline",
    milestone: "M1 Format",
    size: "S",
    deps: ["P1"],
    goal: "MeshGraphDescriptor::merge copies every part's pipeline_descriptors and rejects a name declared twice.",
    files: ["tt_metal/fabric/mesh_graph_descriptor.cpp"],
    steps: [
      "Single part: already copied whole, nothing to do.",
      "Several parts: copy each part's pipelines in the per-part loop; TT_FATAL if a name is already taken, naming both parts.",
      "No remap: stages bind by shape, not by mesh id or name.",
    ],
    code: [
      {
        path: "tt_metal/fabric/mesh_graph_descriptor.cpp  (MeshGraphDescriptor::merge)",
        mode: "diff",
        language: "cpp",
        src: `
     auto merged = std::make_shared<proto::MeshGraphDescriptor>();
     auto* fabric = merged->add_graph_descriptors();
     fabric->set_name("G0");
     fabric->set_type("FABRIC");
     std::optional<proto::Policy> shared_connection_policy;
+    std::unordered_map<std::string, std::size_t> pipeline_part;   // name -> part that declared it

     for (std::size_t i = 0; i < descriptors.size(); ++i) {
         ...
         for (const auto& pin : part.pinnings()) {
             *merged->add_pinnings() = pin;
         }
+        for (const auto& pipeline : part.pipeline_descriptors()) {
+            const auto [it, inserted] = pipeline_part.emplace(pipeline.name(), i);
+            TT_FATAL(
+                inserted,
+                "MeshGraphDescriptor::merge: pipeline '{}' declared in descriptor {} and descriptor {}",
+                pipeline.name(),
+                it->second,
+                i);
+            *merged->add_pipeline_descriptors() = pipeline;
+        }`,
      },
    ],
    tests: [
      "Two parts, pipeline in one: present after merge, byte-identical to the source.",
      "Pipelines with different names in two parts: both present after merge.",
      "The same pipeline name in two parts: merge fails and names both part indexes.",
      "No pipeline anywhere: merged proto is byte-identical to today's.",
    ],
    behaviour: "None without a pipeline.",
  },
  {
    id: "P3",
    title: "Tier B feasibility",
    milestone: "M1 Format",
    size: "M",
    deps: ["P1", "P2"],
    goal: "Reject pipelines impossible on every mapping of the MGD, before SAT runs, from counts only: slices per stage and channels per mesh pair. Never pick a split or a placement; the Tier C solve is fast enough to do that.",
    files: [
      "tt_metal/fabric/mesh_graph_descriptor.cpp",
      "tt_metal/api/tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp",
    ],
    steps: [
      "Binding set per stage: meshes whose slice tiles the stage shape exactly and that fit it. The slice is device_topology / host_topology. A stage that fits no mesh fails here.",
      "Capacity in slices: all stages against all slices, then the stages that fit only one mesh against that mesh.",
      "Channel cap per mesh pair, an upper bound read from the declared connections at every graph level: a graph end expands to every mesh under it, and meshes on one switch get the min of their counts.",
      "Assignment: depth-first stage -> mesh over every pipeline at once, stopping at the first survivor. Most constrained stage first, then breadth-first along edges; meshes holding a neighbour first. Rules: slices fit, and every crossing rides a declared connection within its channel count. A step limit (10^6) ends the search and accepts, with no warning, leaving the answer to Tier C.",
      "Nothing is stored: a private method, MeshGraphDescriptor::check_pipeline_feasibility() const, called in the proto constructor after populate(). It returns at once without pipelines. merge() constructs a new descriptor, so the merged meshes and all pipelines are checked again.",
      "Backup if the Tier C solve is slow (not implemented): the per-mesh segment carve, interchangeable-mesh symmetry, look-ahead on free slices and channels, the \"find all\" run, and PipelineMappingConstraints (k_AB, boundary slices) exposed through pipeline_mapping_constraints() for Plan 1 and the exit-node constraints.",
    ],
    code: [
      {
        path: "tt_metal/fabric/mesh_graph_descriptor.cpp  (proto constructor)",
        mode: "diff",
        language: "cpp",
        src: `
     proto_ = std::move(proto);
     populate();
+    check_pipeline_feasibility();`,
      },
      {
        path: "tt_metal/api/tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp",
        mode: "diff",
        language: "cpp",
        src: `
 private:
     ...
+    // Tier B. Throws, naming the pipeline and the rule, when no mapping of the descriptor can hold it.
+    void check_pipeline_feasibility() const;
 };`,
      },
    ],
    tests: [
      "Stages that only fit one mesh: five 4x2 stages against M0's four slices are rejected, though the total fits.",
      "Two pods under a CLUSTER with one channel: a pipeline crossing once is accepted, one crossing twice is rejected (needs 2 channels and 1 are declared).",
      "A 7x7 grid of one-slice meshes and a ring of 49 stages: the search hits the step limit in about 250 ms and accepts.",
      "subtorus_4x4_pipeline_8stage: the 8-stage ring of 4x4 stages is accepted; without the 7-0 connection it is rejected.",
      "llama_8b_pod with nine 4x2 stages and thirty-one 1x2 stages: the 40 slices fit in total, but only M0 holds 4x2 stages and it has 8 slices.",
      "Not covered without the backup carve: two copies of ring4 on two line meshes pass Tier B and are rejected by Tier C.",
    ],
    behaviour: "An impossible pipeline fails when its MeshGraphDescriptor is constructed, before MeshGraph and before SAT.",
  },
  {
    id: "P5",
    title: "Pipeline graphs and the footprint solve (C1, C2's solve)",
    milestone: "M2 Shared code",
    size: "M",
    deps: [],
    goal: "The two graphs Tier C solves, built like the mesh graphs: LogicalPipelineGraph once per run from the merged MGD (topology_mapper_utils, beside build_logical_multi_mesh_adjacency_graph), and PhysicalPipelineGraph once per seating from the PSD and the intra-mesh layout (pipeline_builder.cpp, with the rest of the per-seating check). The inter-mesh cables come straight from the PSD with existing calls, as ExitNodeConnections, so R and the pairing share one hash. The footprint solve lands here too: solve_pipeline_footprint builds MappingConstraints over the two graphs and returns the topology solver's own MappingResult<PipelineStageNode, PipelinePositionNode>, whose target_to_global is the carve. No new carve type. Replaces the planner port (planner + resolver placed 0 of 132 carve-requiring cases; footprint + resolver decided 127). Library only.",
    files: [
      "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
      "tt_metal/fabric/topology_mapper_utils.cpp",
      "tt_metal/api/tt-metalium/experimental/fabric/pipeline_builder.hpp",
      "tt_metal/fabric/pipeline_builder.cpp",
      "tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper_utils.cpp",
      "tests/tt_metal/tt_fabric/fabric_router/test_pipeline_builder.cpp",
    ],
    steps: [
      "Types, all in topology_mapper_utils.hpp (section 4.0 lists them): PipelineStageNode (pipeline name, stage id), PipelineEdge, PipelineHostSlice, LogicalPipelineGraph, PipelinePositionNode, PipelinePositionLink, PhysicalPipelineGraph. pipeline_builder.hpp only forward-declares them, so the ttnn ops and nanobind module that include it are unchanged.",
      "Logical graph, build_logical_pipeline_adjacency_graph(mgd) in topology_mapper_utils.cpp: stage_graph_ with both directions of every edge (a repeated edge is a duplicate entry), stage shapes and labels, the directed edges with their loopback marks, each pipeline's host I/O stage, and each mesh's host slices (get_coord_range per host rank), because the enumerator does not keep the merged MGD.",
      "Cables, file-local inter_mesh_cables(psd, fabric_node_to_asic, mesh_a, mesh_b) in pipeline_builder.cpp: for each ASIC mapped onto mesh_a, psd.get_asic_neighbors + get_eth_connections to ASICs mapped onto mesh_b, one ExitNodeConnection per ETH channel, sorted by (src chip, src chan). Today's gather rules that need no cluster: both ends mapped onto the two meshes; for a device-level connection only requested exit nodes; one destination chip per (source chip, neighbour mesh), else no cable. Not filtered yet: cable type and link state (P16; open question 20).",
      "Physical graph, detail::build_physical_pipeline_adjacency_graph(pipelines, logical, full, psd) in pipeline_builder.cpp: for each distinct stage shape and each mesh, every rectangle of host slices the shape fills is a position. Positions overlap on purpose: the graph lists every place a stage could go, and the solve's resource constraint keeps the chosen ones apart, as SAT seating does for overlapping seat footprints. An edge joins two disjoint positions that border each other, by grid adjacency or by a cable, one duplicate entry per link. Keep, per edge, the links behind it and their ExitNodeConnections, and per position its chips and slice ids.",
      "Constraints for solve_pipeline_footprint: add_required_constraint(stage, positions it fits on its own): its shape, at least as many core slots on the position's chips as the stage has endpoints, and for the host I/O stage room for H2D and D2H. These are the only failures that depend on one stage and one position, so they are filtered here and never learned from the resolver. add_resource_constraint(position -> covered slices).",
      "Resource completeness with the existing API: when the stage areas add up to exactly the slice count, add_cardinality_constraint(pairs, 1) per slice over every (stage, position) pair covering it. With min_count 1 the encoder emits one plain clause (report 5.4: 66 s to 0.59 s, 170 s to 0.32 s).",
      "One solve, returned as is: a TopologyMappingEnumerationSession<PipelineStageNode, PipelinePositionNode> with STRICT validation, the SAT engine (Auto picks it) and conflict_cap, and one next(), whose MappingResult<PipelineStageNode, PipelinePositionNode> is the return value: success, error_message, stats and target_to_global (the carve). No PipelineCarve or PipelineFootprint type. A capped solve reads as no carve with today's API (open question 13). One carve, no loop: the resolver accepted the first carve on 101 of 101 benchmark cases; a second carve, if ever needed, is exclude_mapping(carve) and next() on the same session (open question 24).",
      "No control-plane change: the gather stays as it is.",
    ],
    code: [
      {
        path: "tt_metal/fabric/topology_mapper_utils.cpp  (beside build_logical_multi_mesh_adjacency_graph)",
        mode: "new",
        language: "cpp",
        src: `
LogicalPipelineGraph build_logical_pipeline_adjacency_graph(const MeshGraphDescriptor& mgd) {
    LogicalPipelineGraph graph;
    AdjacencyGraph<PipelineStageNode>::AdjacencyMap adj;
    for (const auto& pipeline : mgd.pipeline_descriptors()) {              // Tier A has checked them
        for (const auto& stage : pipeline.stages) {
            const PipelineStageNode node{pipeline.name, stage.id};
            adj[node];                                                     // isolated stages stay listed
            graph.stage_shapes_[node] = stage.shape;
            graph.stage_labels_[node] = stage.label;
        }
        graph.host_io_stage_[pipeline.name] = {pipeline.name, the_one_root(pipeline)};   // 2.1
        for (const auto& edge : pipeline.edges) {
            const PipelineStageNode src{pipeline.name, edge.src}, dst{pipeline.name, edge.dst};
            adj[src].push_back(dst);
            adj[dst].push_back(src);
            graph.edges_.push_back(PipelineEdge{.src = src, .dst = dst, .loopback = edge.loopback});
        }
    }
    graph.stage_graph_ = AdjacencyGraph<PipelineStageNode>(std::move(adj));
    graph.host_slices_ = host_slices(mgd);                                 // get_coord_range per host rank
    return graph;
}`,
      },
      {
        path: "tt_metal/fabric/pipeline_builder.cpp  (C1)",
        mode: "new",
        language: "cpp",
        src: `
// Pure: PSD + mapping only. Same struct as the PSD's exit-node table, so std::hash matches the pairing's.
std::vector<tt::tt_metal::ExitNodeConnection> inter_mesh_cables(
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const std::map<FabricNodeId, tt::tt_metal::AsicID>& fabric_node_to_asic,
    MeshId mesh_a, MeshId mesh_b);                                         // anonymous namespace

// C1: every position of every stage shape, per mesh. Positions overlap: the graph lists every place a
// stage could go. An edge joins two disjoint positions that border each other, one entry per link.
PhysicalPipelineGraph detail::build_physical_pipeline_adjacency_graph(
    const LogicalPipelineGraph& pipelines, const LogicalMultiMeshGraph& logical,
    const TopologyMappingResult& full, const tt::tt_metal::PhysicalSystemDescriptor& psd) {
    PhysicalPipelineGraph g;
    const SliceGrid slices = slice_grid(pipelines.host_slices_, full);          // slice ids, chips per slice
    for (const MeshShape& shape : distinct_stage_shapes(pipelines)) {
        for (const PipelinePositionNode& p : slices.positions(shape)) {        // per mesh, in slices
            g.position_slices_[p] = slices.covered(p);
            g.position_chips_[p] = slices.chips(p);
        }
    }
    const SliceLinks links = slice_links(slices, logical, full, psd);          // grid links + inter_mesh_cables
    AdjacencyGraph<PipelinePositionNode>::AdjacencyMap adj;
    for (const auto& [p, p_slices] : g.position_slices_) {
        for (const auto& [q, q_slices] : g.position_slices_) {
            if (!(p < q) || !disjoint(p_slices, q_slices)) {
                continue;                                                      // overlapping: never adjacent
            }
            std::vector<PipelinePositionLink> between = links.between(p_slices, q_slices);
            for (std::size_t i = 0; i < between.size(); ++i) {
                adj[p].push_back(q);
                adj[q].push_back(p);
            }
            if (!between.empty()) {
                g.links_[{p, q}] = std::move(between);
            }
        }
    }
    g.position_graph_ = AdjacencyGraph<PipelinePositionNode>(std::move(adj));
    g.slice_count_ = slices.count();
    return g;
}
`,
      },
      {
        path: "tests: test_topology_mapper_utils.cpp (logical) and test_pipeline_builder.cpp (physical)",
        mode: "new",
        language: "cpp",
        src: `
TEST(PipelineGraphTest, LogicalGraphFromTheGptOssRing) {
    const LogicalPipelineGraph g = build_logical_pipeline_adjacency_graph(MeshGraphDescriptor(kGptOssRingMgd));
    const PipelineStageNode s0{"gpt_oss_120b_single_galaxy", "s0"};
    EXPECT_EQ(g.stage_graph_.get_neighbors(s0).size(), 2u);              // s1 and s3, both directions kept
    EXPECT_EQ(g.host_io_stage_.at("gpt_oss_120b_single_galaxy"), s0);
    EXPECT_TRUE(g.edges_.back().loopback);
}

TEST(PipelineFootprintTest, OverlappingPositionsAreNeverAdjacent) {
    const auto c = two_by_two_slices_one_by_two_stages();                  // four positions, each slice in two
    const PhysicalPipelineGraph positions = detail::build_physical_pipeline_adjacency_graph(c.pipelines, c.logical, c.full, c.psd);
    EXPECT_EQ(positions.position_slices_.size(), 4u);
    for (const auto& [p, q] : overlapping_pairs(positions)) {
        EXPECT_FALSE(adjacent(positions, p, q));
    }
}`,
      },
      {
        path: "tt_metal/fabric/pipeline_builder.cpp  (the footprint solve)",
        mode: "new",
        language: "cpp",
        src: `
// Tight carve: every slice must be covered. Existing API: at least one (stage, position) covering it.
void add_resource_completeness(MappingConstraints<PipelineStageNode, PipelinePositionNode>& constraints,
                               const LogicalPipelineGraph& pipelines, const PhysicalPipelineGraph& g,
                               const std::map<PipelineStageNode, std::set<PipelinePositionNode>>& fits) {
    if (total_slices(pipelines, g) != g.slice_count_) {
        return;                                                                // spare slices: not tight
    }
    for (std::uint32_t s = 0; s < g.slice_count_; ++s) {
        CardinalityPairSet covering;
        for (const auto& [stage, positions] : fits) {
            for (const PipelinePositionNode& p : positions) {
                if (covers(g, p, s)) {
                    covering.insert({stage, p});
                }
            }
        }
        constraints.add_cardinality_constraint(covering, /*min_count=*/1);    // one clause
    }
}

// C2: one solve, one carve, returned as the solver's own MappingResult. The caller checks it; no second carve.
MappingResult<PipelineStageNode, PipelinePositionNode> detail::solve_pipeline_footprint(
    const LogicalPipelineGraph& pipelines, const PhysicalPipelineGraph& g, std::size_t conflict_cap) {
    MappingConstraints<PipelineStageNode, PipelinePositionNode> constraints;
    std::map<PipelineStageNode, std::set<PipelinePositionNode>> fits;
    for (const auto& [stage, shape] : pipelines.stage_shapes_) {
        fits[stage] = positions_that_fit(pipelines, g, stage);                 // shape, core slots, host I/O
        if (fits[stage].empty()) {
            MappingResult<PipelineStageNode, PipelinePositionNode> none;                // success = false
            none.error_message = fmt::format("stage '{}/{}' has no position on this seating", stage.pipeline, stage.stage);
            return none;
        }
        constraints.add_required_constraint(stage, fits[stage]);
    }
    constraints.add_resource_constraint<std::uint32_t>(g.position_slices_);   // chosen positions share no slice
    add_resource_completeness(constraints, pipelines, g, fits);
    TopologyMappingEnumerationSession<PipelineStageNode, PipelinePositionNode> session(
        pipelines.stage_graph_, g.position_graph_, constraints, ConnectionValidationMode::STRICT,
        /*quiet_mode=*/true, TopologyMappingSolverEngine::Auto, /*unique_shapes=*/false, conflict_cap);
    return session.next();   // target_to_global is the carve; a capped solve reads as success = false
}`,
      },
      {
        path: "tests/tt_metal/tt_fabric/fabric_router/test_pipeline_builder.cpp  (CPU; MGD textprotos and PSD fixtures)",
        mode: "new",
        language: "cpp",
        src: `
TEST_P(PipelineFootprintTest, MatchesBruteForceAndVerifies) {
    const FootprintCase c = GetParam();                                    // MGD + PSD + one seating
    const PhysicalPipelineGraph positions = detail::build_physical_pipeline_adjacency_graph(c.pipelines, c.logical, c.full, c.psd);
    const auto footprint = detail::solve_pipeline_footprint(c.pipelines, positions, kCap);
    EXPECT_EQ(footprint.success, brute_force_carve_exists(c));  // small cases only
    if (footprint.success) {
        EXPECT_TRUE(is_valid_carve(c, footprint.target_to_global));                  // own shape, disjoint, every edge touching
    }
}

TEST(PipelineFootprintTest, StageThatCannotFitIsFilteredBeforeTheSolve) {
    const auto c = reject_demo_case();       // the host I/O stage cannot fit on A0
    const PhysicalPipelineGraph positions = detail::build_physical_pipeline_adjacency_graph(c.pipelines, c.logical, c.full, c.psd);
    const auto footprint = detail::solve_pipeline_footprint(c.pipelines, positions, kCap);
    ASSERT_TRUE(footprint.success);
    EXPECT_NE(footprint.target_to_global.at(c.host_stage), c.a0);        // s0 is never put on A0
}`,
      },
    ],
    tests: [
      "Logical graph from the GPT-OSS ring, the dflash fork, two pipelines in one MGD, and a repeated edge (two entries).",
      "Overlapping positions: a 2x2 grid of slices gives four 1x2 positions; overlapping pairs are never adjacent, and a carve never picks two of them.",
      "Cables on SC16 revAB/revC and single-galaxy PSD fixtures: the hashes equal the connection_hashes today's gather produces for the same mapping, on every boundary.",
      "Device-level connection: only requested exit nodes; a source chip cabled to two chips of one neighbour mesh gives no cable.",
      "Brute-force agreement on every case with at most 8 stages; prototype: pipeline_placement_experiments/topo/carve_footprint_bench.cpp.",
      "Position filter: a stage never lands on a position it cannot fit; verdicts with and without the filter are equal on every case.",
      "Resource completeness: same verdicts with and without on every designed, repo, ring and mixed-cable case, and the tight-case speed-up of report 5.4.",
    ],
    behaviour: "None. Library only. Tier C may accept carves blaze's planner rejects (open question 8).",
  },
  {
    id: "P6",
    title: "The per-seating check (C2): pre-checks, resolver, R and demand in pipeline_builder",
    milestone: "M2 Shared code",
    size: "M",
    deps: ["P5"],
    goal: "One call per seating, detail::check_pipeline_seating in pipeline_builder.cpp: build the physical graph (P5), run the pre-checks, take one carve from P5's solve, run the resolver on that carve, record R as exact channels, check R against each connection's demand, and fill the mapping. The mapper (P8) only calls it and reads accept or reject. No topology solver changes.",
    files: [
      "tt_metal/api/tt-metalium/experimental/fabric/pipeline_builder.hpp",
      "tt_metal/fabric/pipeline_builder.cpp",
      "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
      "tests/tt_metal/tt_fabric/fabric_router/test_pipeline_builder.cpp",
    ],
    steps: [
      "Types: the mapping result (PipelinePlacement, PipelineMappingResult) in topology_mapper_utils.hpp; PipelineCheck in pipeline_builder.hpp. The carve is P5's MappingResult::target_to_global. No new placement types: PipelinePlacement stores the resolver's own submesh_chips (ChipTuple) and GraphLayoutResult, so topology_mapper_utils.hpp includes pipeline_builder.hpp, which stays light (fabric_types.hpp, mesh_graph.hpp).",
      "Pre-checks before the solve: area, lattice count, and the colour and segment test on tight 1-cable meshes; cable pairs the segment test rules out are dropped from the graph.",
      "Resolver arguments, one call per pipeline, all in the resolver's existing types: submesh_chips -> std::vector<std::vector<ChipTuple>> (mesh_id, chip_id, row, col) per chip of each stage's position, slot-local; direct_links -> DirectLinks, every link between two submeshes under {i, j} and mirrored under {j, i}, ordered by chip id; stage ids, EdgeInputTuple edges with the loopback marks, node_chip_counts and blaze's default core counts from LogicalPipelineGraph. No wrapper struct and no reverse map: a resolved chip is read back from submesh_chips.",
      "The resolver call, resolve_graph_layout_with_connections; a throw rejects the carve with its message (pipeline_layout).",
      "IMPORTANT: R from the layout's resolved_edges, one channel per crossing edge (loopback included, deduplicated): the PSD ExitNodeConnection between the exit and entry ASIC from inter_mesh_cables (lowest src_chan; best tier first after P16), stored as is; its std::hash is the pairing's connection_hash. Intra-mesh edges are not in it.",
      "Demand check right after R: channels in R per mesh pair <= strict + relaxed from get_inter_mesh_channel_demand; over demand rejects with pipeline_binding and the mesh pair.",
      "Fill PipelinePlacement per pipeline: submesh_chips, the GraphLayoutResult as returned, required_link per resolved edge (index into R), stage_host_ranks; then idle_host_ranks. The mapping is written only on accept.",
    ],
    code: [
      {
        path: "tt_metal/api/tt-metalium/experimental/fabric/pipeline_builder.hpp  (detail)",
        mode: "diff",
        language: "cpp",
        src: `
+struct LogicalPipelineGraph;       // topology_mapper_utils.hpp: forward declarations only
+struct PhysicalPipelineGraph;
+struct LogicalMultiMeshGraph;
+struct PipelineMappingResult;
+struct TopologyMappingResult;
+template <typename TargetNode, typename GlobalNode>
+struct MappingResult;              // topology_solver.hpp
+struct PipelineStageNode;
+struct PipelinePositionNode;
+
 namespace detail {
 ...
+PhysicalPipelineGraph build_physical_pipeline_adjacency_graph(          // P5
+    const LogicalPipelineGraph& pipelines, const LogicalMultiMeshGraph& logical,
+    const TopologyMappingResult& full, const tt::tt_metal::PhysicalSystemDescriptor& psd);
+
+// P5: the solver's own result; target_to_global is the carve.
+MappingResult<PipelineStageNode, PipelinePositionNode> solve_pipeline_footprint(
+    const LogicalPipelineGraph& pipelines, const PhysicalPipelineGraph& positions, std::size_t conflict_cap);
+
+struct PipelineCheck {
+    bool accepted = false;
+    std::string failure_stage;     // pipeline_carve | pipeline_layout | pipeline_binding
+    std::string reason;
+};
+
+// The one call the mapper makes per seating. Fills mapping only on accept; C4 stays with the caller.
+PipelineCheck check_pipeline_seating(
+    const LogicalPipelineGraph& pipelines,
+    const LogicalMultiMeshGraph& logical,          // declared channels per mesh pair, for the demand check
+    const TopologyMappingResult& full,
+    const tt::tt_metal::PhysicalSystemDescriptor& psd,
+    PipelineMappingResult& mapping);
 }  // namespace detail`,
      },
      {
        path: "tt_metal/fabric/pipeline_builder.cpp  (detail::check_pipeline_seating: one seating, end to end)",
        mode: "new",
        language: "cpp",
        src: `
PipelineCheck detail::check_pipeline_seating(
    const LogicalPipelineGraph& pipelines,
    const LogicalMultiMeshGraph& logical,
    const TopologyMappingResult& full,
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    PipelineMappingResult& mapping) {
    // C1: every position of every stage shape. Positions overlap; edges join disjoint, bordering ones.
    const PhysicalPipelineGraph positions = build_physical_pipeline_adjacency_graph(pipelines, logical, full, psd);
    if (auto pre = pipeline_pre_checks(pipelines, positions)) {               // area, lattice, colour and segment
        return PipelineCheck::reject("pipeline_layout", pre->reason);
    }
    // C2: one footprint solve. The resource constraint keeps the chosen positions disjoint.
    const auto footprint = solve_pipeline_footprint(pipelines, positions, kPipelineConflictCap);   // P5
    if (!footprint.success) {
        return PipelineCheck::reject("pipeline_carve", footprint.error_message);     // no position, no carve, or the cap
    }
    // The resolver on that carve, one call per pipeline, in the resolver's own types.
    PipelineMappingResult result;
    for (const std::string& name : pipeline_names(pipelines)) {
        PipelinePlacement placement{.pipeline = name};
        placement.submesh_chips = submesh_chips(pipelines, name, positions, footprint.target_to_global);   // slot-local (row, col)
        const DirectLinks links = direct_links(placement.submesh_chips, positions);   // every candidate link, both ways
        try {
            placement.layout = resolve_graph_layout_with_connections(
                stage_ids(pipelines, name), edge_tuples(pipelines, name), placement.submesh_chips,
                chip_counts(pipelines, name), core_counts(pipelines, name), std::nullopt, &links);
        } catch (const std::runtime_error& e) {
            return PipelineCheck::reject("pipeline_layout", fmt::format("pipeline '{}': {}", name, e.what()));
        }
        // IMPORTANT: R is the exact channel of every crossing, so phase 0 binds the cable the check used.
        // Chips come back through submesh_chips: node_to_submesh + (row, col) -> (mesh, chip).
        placement.required_link = record_required_links(placement, positions, result.required_links);
        placement.stage_host_ranks = stage_host_ranks(pipelines, name, footprint.target_to_global);
        result.pipelines.push_back(std::move(placement));
    }
    if (auto over = first_over_demand(logical, result.required_links)) {   // strict + relaxed per mesh pair
        return PipelineCheck::reject("pipeline_binding", over->reason);
    }
    result.idle_host_ranks = idle_host_ranks(pipelines, footprint.target_to_global);
    mapping = std::move(result);
    return PipelineCheck::accept();
}`,
      },
      {
        path: "tt_metal/fabric/pipeline_builder.cpp  (anonymous namespace: the resolver's arguments from a carve)",
        mode: "new",
        language: "cpp",
        src: `
// Each returns a type the resolver already takes; there is no wrapper struct.
std::vector<std::vector<ChipTuple>> submesh_chips(const LogicalPipelineGraph& pipelines, const std::string& pipeline,
                                                  const PhysicalPipelineGraph& positions,
                                                  const std::map<PipelineStageNode, PipelinePositionNode>& carve);
DirectLinks direct_links(const std::vector<std::vector<ChipTuple>>& submesh_chips, const PhysicalPipelineGraph& positions);
std::vector<std::string> stage_ids(const LogicalPipelineGraph& pipelines, const std::string& pipeline);
std::vector<EdgeInputTuple> edge_tuples(const LogicalPipelineGraph& pipelines, const std::string& pipeline);
std::map<std::string, uint32_t> chip_counts(const LogicalPipelineGraph& pipelines, const std::string& pipeline);
std::map<std::string, uint32_t> core_counts(const LogicalPipelineGraph& pipelines, const std::string& pipeline);

// Reads each crossing's chips back through submesh_chips, appends its exact channel to R, and
// returns, per layout.resolved_edges entry, its index into R (empty inside a mesh).
std::vector<std::optional<std::size_t>> record_required_links(
    const PipelinePlacement& placement, const PhysicalPipelineGraph& positions,
    std::vector<tt::tt_metal::ExitNodeConnection>& required_links);`,
      },
    ],
    tests: [
      "The rings blaze's planner rejects (4x4 with 13 slices, 4x6, 4x8): accepted, carve verified.",
      "mixed4 from the doc: (P,R) accept; (P,Q) and Case B reject. On (P,R): four stage slots with 8, 16, 16, 8 chips; direct_links has the A2-B2 and A0-B0 pairs both ways; R is those two channels.",
      "Counting and area failures (5x5 with five 2x2, 6x6 with area 37) rejected by the pre-check before the solve.",
      "A fork (one stage with two returning branches) and two meshes joined by cables.",
      "First carve passes the resolver on every benchmark case (the 101 of footprint_then_resolver).",
      "A slot spanning two host slices gets slot-local (row, col) starting at (0, 0); every resolved edge reads back, through submesh_chips, to a FabricNodeId pair with a cable from inter_mesh_cables.",
      "Each R channel's connection_hash equals std::hash<ExitNodeConnection> of the PSD entry.",
      "A ring across two meshes over a count-1 connection: R needs 2, rejected with pipeline_binding.",
      "No change to topology_solver_sat.cpp or topology_solver.hpp.",
    ],
    behaviour: "None. Library only.",
  },
  {
    id: "P8",
    title: "Tier C in the mapper: the hook, seating exclusion (C4) and the saved mapping",
    milestone: "M3 Tier C (STRICT)",
    size: "M",
    deps: ["P3", "P6"],
    goal: "Wire P6 into the mapper. The enumerator builds the logical pipeline graph once, calls detail::check_pipeline_seating after every intra-mesh success, excludes rejected seatings (C4) and returns the accepted pipeline mapping in its result; TopologyMapper saves it on rank 0 and broadcasts R. The mapper side holds no pipeline logic beyond this call.",
    files: [
      "tt_metal/fabric/topology_mapper_utils.cpp",
      "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
      "tt_metal/fabric/topology_mapper.cpp",
      "tools/scaleout/src/generate_rank_bindings.cpp",
      "tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper_utils.cpp",
    ],
    steps: [
      "TopologyMappingConfig::check_pipelines (default true); TopologyMapper sets it false under the kill switch (4.7).",
      "Enumerator constructor: std::optional<LogicalPipelineGraph> pipelines_ = build_logical_pipeline_adjacency_graph(merged) (P5) when the merged MGD has pipelines and check_pipelines is set. Nothing else goes through the config, so every overload and generate_rank_bindings get it the same way.",
      "next(): after intra-mesh success, call detail::check_pipeline_seating(*pipelines_, logical_, full, psd, mapping) (P6). Accept: full.pipeline = mapping. Reject (C4): failure_stage and reason into the result, last_failed_, exclude_mapping(seating) on the seat set (never add_forbidden_constraint), and the next seating.",
      "Cap rejections per run (kMaxPipelineRejections, proposed 256, overridable for sweeps) and report the last reason; record rejections and solve time in the placement stats, so the sweep and CI show whether P10 is needed.",
      "Turn unique_shapes off when a pipeline is present (runtime overloads and --all-solutions): swapping meshes between the same seats can change the pipeline answer.",
      "Result: TopologyMappingResult::pipeline; split_mapping_to_local_parts remaps it per part.",
      "TopologyMapper stores mapping_result.pipeline on rank 0 (pipeline_mapping_, get_pipeline_mapping()). R is copied to pipeline_required_links_ and broadcast with the chip info, because every rank checks it after the pairing (P9).",
      "Feedback is whole-seating exclusion only; the exit-node retry is P15 and the seat-pair clause P13.",
    ],
    code: [
      {
        path: "tt_metal/fabric/topology_mapper.cpp  (TopologyMapper::build_mapping, rank 0: the call, and where the mapping is saved)",
        mode: "diff",
        language: "cpp",
        src: `
+        config.check_pipelines = !rtoptions.get_ignore_pipeline_descriptors();   // kill switch (4.7)
         const auto mapping_result = pgd.has_value()
                                         ? map_multi_mesh_to_physical(physical_system_descriptor_, *pgd, mesh_graph_descriptor,
                                                                      config, pinnings_by_mesh, asic_ranks, fabric_ranks)
                                         : map_multi_mesh_to_physical(physical_system_descriptor_, mesh_graph_descriptor,
                                                                      config, pinnings_by_mesh, asic_ranks, fabric_ranks);
         TT_FATAL(mapping_result.success, "Graph specified in MGD could not fit in the discovered physical topology. {}. ...",
                  mapping_result.error_message);
         for (const auto& [fabric_node, asic] : mapping_result.fabric_node_to_physical) { ... }   // today
+        pipeline_mapping_ = mapping_result.pipeline;                         // saved on rank 0
+        if (pipeline_mapping_) pipeline_required_links_ = pipeline_mapping_->required_links;  // R: broadcast with the chip info
         if (!generate_mapping_locally_) {
             broadcast_chip_info_to_hosts(all_host_ranks, -1);
         }`,
      },
      {
        path: "topology_mapper_utils.hpp and topology_mapper.hpp  (where it is saved)",
        mode: "diff",
        language: "cpp",
        src: `
 struct TopologyMappingConfig {
     ...
+    bool check_pipelines = true;                           // false: no pipeline code runs (kill switch)
 };

 struct TopologyMappingResult {
     bool success = false;
     std::string error_message;
     TopologyMappingStats stats;
     std::map<FabricNodeId, PhysicalNodeId> fabric_node_to_physical;
     std::map<PhysicalNodeId, FabricNodeId> physical_to_fabric_node;
+    std::optional<PipelineMappingResult> pipeline;         // output: stage slots, edge layouts, R, idle ranks
 };

 class MultiMeshSolutionEnumerator {
     ...
     std::unique_ptr<SatPlacementEnumerationSession> placement_session_;
+    std::optional<LogicalPipelineGraph> pipelines_;        // empty without pipelines, like logical_
+    std::size_t pipeline_rejections_ = 0;
 };

 class TopologyMapper {
     ...
+    const std::optional<PipelineMappingResult>& get_pipeline_mapping() const { return pipeline_mapping_; }
+    const std::vector<tt::tt_metal::ExitNodeConnection>& get_pipeline_required_links() const { return pipeline_required_links_; }  // every rank
 private:
+    std::optional<PipelineMappingResult> pipeline_mapping_;            // rank 0, from mapping_result.pipeline
+    std::vector<tt::tt_metal::ExitNodeConnection> pipeline_required_links_;  // every rank: rank 0's R, broadcast
 };`,
      },
      {
        path: "tt_metal/fabric/topology_mapper_utils.cpp  (MultiMeshSolutionEnumerator: constructor and next)",
        mode: "diff",
        language: "cpp",
        src: `
 // constructor, after init_from_parts
+    if (config_.check_pipelines && merged.has_pipeline_descriptors()) {
+        pipelines_ = build_logical_pipeline_adjacency_graph(merged);   // once per run
+    }

 // next()
     while (true) {
         AssignedMeshes seating = placement_session_->next();
         if (seating.empty()) {
             return give_up();
         }
         TopologyMappingResult full = complete_intra_mesh_for_placement(identity, logical_, physical, ...);
         if (placement_stats_ != nullptr) {
             apply_placement_stats(full.stats, *placement_stats_);
         }
         if (full.success) {
+            if (pipelines_.has_value()) {
+                PipelineMappingResult mapping;
+                const PipelineCheck check =
+                    detail::check_pipeline_seating(*pipelines_, logical_, full, *physical_system_descriptor_, mapping);   // P6
+                if (!check.accepted) {                                    // C4: this set of seats fails
+                    full.success = false;
+                    full.stats.failure_stage = check.failure_stage;   // pipeline_carve | pipeline_layout | pipeline_binding
+                    full.error_message = check.reason;
+                    last_failed_ = full;
+                    if (++pipeline_rejections_ > kMaxPipelineRejections ||
+                        !placement_session_->exclude_mapping(seating)) {
+                        return give_up();
+                    }
+                    continue;
+                }
+                full.pipeline = std::move(mapping);                       // saved in the result
+            }
             ++emitted_;
             return split_mapping_to_local_parts(full, per_part_local_to_global_mesh_ids_);   // remaps pipeline too
         }
         // intra-mesh failure: add_forbidden_constraint(failed seat), as today
     }`,
      },
      {
        path: "tt_metal/fabric/topology_mapper_utils.cpp  (map_multi_mesh_to_physical, both overloads)",
        mode: "diff",
        language: "cpp",
        src: `
     MultiMeshSolutionEnumerator enumerator(
         physical_system_descriptor,
         mesh_graph_descriptor,
         config,
-        /*unique_shapes=*/true,  // Unique shapes commit fewer orientations than necessary, shrinking the SAT sweep
+        // Swapping meshes between the same seats can change the pipeline answer, so keep every permutation.
+        /*unique_shapes=*/!mesh_graph_descriptor.has_pipeline_descriptors(),
         pinnings,`,
      },
      {
        path: "tools/scaleout/src/generate_rank_bindings.cpp  (--all-solutions)",
        mode: "diff",
        language: "cpp",
        src: `
                 // Solver-level unique_shapes dedup is ALWAYS ON (collapses automorphic physical footprints);
                 // only the hidden --allow-shape-permutations turns it off.
-                const bool unique_shapes = !args.allow_shape_permutations;
+                const bool unique_shapes = !args.allow_shape_permutations && !any_has_pipeline_descriptors(mgds);`,
        note: "run_topology_mapping already passes false; seat-pool enumeration must stay false; the three enumerate_flat_grouping_embeddings calls keep their default.",
      },
    ],
    tests: [
      "mesh4x4_stage4x2 on a crafted PSD with Case A / B / C cabling: accept / reject / reject (STRICT).",
      "GPT-OSS single galaxy, llama_8b_pod, subtorus 8-stage: accepted, and blaze builds on the emitted mapping (hardware CI).",
      "A seating that fails only when swapped (unique_shapes regression test).",
      "Single galaxy, host_topology [1, 8], ring 4x1, 4x2, 4x2, 4x2, 4x1: takes the planned carve, not single-mesh tiling, and is accepted; the same with a LINE column axis is rejected.",
      "Two copies of ring4 on the mixed4 meshes: rejected with cables A0-B0, A2-B2, A3-B3 (only one 4-cycle), accepted once A1-B1 is added.",
      "No pipeline: every existing topology-mapper golden unchanged.",
    ],
    behaviour: "The mapper skips seatings the pipeline cannot be built on. --all-solutions only emits buildable ones.",
  },
  {
    id: "P9",
    title: "Bind the cables the layout uses (C3)",
    milestone: "M3 Tier C (STRICT)",
    size: "M",
    deps: ["P8"],
    goal: "Have the control plane bind exactly the channels in R (recorded and checked against demand in P6) before any other channel, and fail on every rank together if one is not bound. IMPORTANT: R is channels, not mesh pairs or chip pairs.",
    files: [
      "tt_metal/fabric/topology_mapper.cpp",
      "tt_metal/fabric/control_plane.cpp",
    ],
    steps: [
      "No PortDescriptor change: each rank sends rank 0 a proto with only hash, src chip and dst chip, and only rank 0 has R. Rank 0 passes R's hashes to pair_logical_intermesh_ports(port_descriptors, required_hashes).",
      "Broadcast R with the chip info (TopologyMapper), so validate_requested_intermesh_connections can check on every rank that every hash in R is in the broadcast AnnotatedIntermeshConnections.",
      "Phase 0 in pair_logical_intermesh_ports: for each link (chip pair) holding a required hash, move the required hashes to the front before has_budget_after_trim (today's trim keeps the lowest hash values and could drop the required channel), place it with its boundary's Z rule, count it against the budget. Never TT_FATAL on rank 0: the other ranks are already waiting in recv for the broadcast. Unplaced required channels fail the check after the broadcast, on every rank together (pipeline_binding).",
      "Fill after phase 0: today's order.",
      "Single host: phase 0 in generate_intermesh_connections_on_local_host, or a warning until it lands.",
      "No rerun over the bound links: once R is bound, the witness layout is valid over the bound set.",
    ],
    code: [
      {
        path: "tt_metal/fabric/control_plane.cpp  (pair_logical_intermesh_ports)",
        mode: "diff",
        language: "cpp",
        src: `
+    // Phase 0: links the pipeline layout depends on go first, so the fill cannot strand them.
+    // No throw here: rank 0 must still broadcast; the check after the broadcast fails on every rank.
+    for (auto& [boundary, link] : order) {
+        auto& h = link->connection_hashes;
+        const auto first_free = std::stable_partition(h.begin(), h.end(), [&](std::size_t x) { return required_hashes.contains(x); });
+        if (first_free == h.begin()) {
+            continue;                                              // no required channel on this chip pair
+        }
+        if (has_budget_after_trim(boundary, *link) &&               // trims from the back: required hashes stay
+            (is_marked_z(boundary) ? place_link(*link, /*want_z=*/true)
+                                   : (place_link(*link, /*want_z=*/false) || place_link(*link, /*want_z=*/true)))) {
+            account_placed(boundary, *link);
+        }
+    }
+
     // Phase 1a: assign_z (marked) boundaries claim the Z lane; they are excluded from the NESW phases below.
     for (auto& [boundary, link] : order) {
         if (!is_marked_z(boundary) || link->placed) {`,
      },
      {
        path: "tt_metal/fabric/control_plane.cpp  (convert_port_descriptors_to_intermesh_connections, multi-host)",
        mode: "diff",
        language: "cpp",
        src: `
     this->forward_descriptors_to_controller(port_descriptors, my_rank, my_host);   // every rank -> rank 0
     AnnotatedIntermeshConnections intermesh_connections;
     if (my_rank == 0) {
-        intermesh_connections = this->pair_logical_intermesh_ports(port_descriptors);
+        intermesh_connections = this->pair_logical_intermesh_ports(
+            port_descriptors, required_hashes(topology_mapper_->get_pipeline_required_links()));
     }
     this->forward_intermesh_connections_from_controller(intermesh_connections);   // rank 0 -> every rank
     // each rank: hash -> its own ETH channel and peer from its PSD copy (cable_lookup), as today

 // validate_requested_intermesh_connections, every rank, same broadcast list
+    for (const auto& r : topology_mapper_->get_pipeline_required_links()) {
+        TT_FATAL(contains_hash(intermesh_connections, std::hash<tt::tt_metal::ExitNodeConnection>{}(r)),
+                 "pipeline_binding: required channel {} chan {} -> {} chan {} was not bound",
+                 get_fabric_node_id_from_asic_id(r.src_exit_node), r.eth_conn.src_chan,
+                 get_fabric_node_id_from_asic_id(r.dst_exit_node), r.eth_conn.dst_chan);
+    }`,
      },
    ],
    tests: [
      "RELAXED count 2 with four candidates where the lowest chip ids share a slice pair: today Case B, after P9 Case A.",
      "R empty (no pipeline): AnnotatedIntermeshConnections byte-identical to today.",
      "Multi-host: R broadcast with the chip info is identical on every rank; validate_requested_intermesh_connections passes everywhere.",
      "Each R channel appears in the broadcast list, and every rank resolves it to the recorded src_chan / dst_chan.",
      "A chip pair with 2 channels and budget 1, required channel has the higher hash: phase 0 binds the required one (today's trim would keep the other).",
      "Phase 0 cannot place a link (exit chip already facing a third mesh): no hang; every rank fails the check together, naming the channel and its two chips.",
      "A required channel whose link is down: not gathered, so every rank fails the check together (open question 20).",
    ],
    behaviour: "With a pipeline, the bound cables can differ from today's lowest-id choice.",
  },
  {
    id: "P15",
    title: "Exit-node retry on the same seating (C4)",
    milestone: "M3 Tier C (STRICT)",
    size: "M",
    deps: ["P8"],
    goal: "Before excluding a seating, retry its intra-mesh solve with the failing exit-node slice signature blocked, so a seating is rejected only when its other layouts fail too.",
    files: [
      "tt_metal/fabric/topology_mapper_utils.cpp",
    ],
    steps: [
      "On a C2 failure, record sig(m) for each mesh on a crossed boundary: cabled exit ASIC -> host slice of the fabric node on it.",
      "Skip when hard rank bindings or MGD pins fix every ASIC in sig(m) under its slice.",
      "Block with the existing API: the retry builds a fresh intra-mesh solve, so add add_cardinality_constraint(pairs, 1) over every (fabric node f, exit ASIC a) with f outside the slice a sat under last time (at least one exit ASIC moves slice). No new solver API.",
      "Re-solve one mesh at a time with the others fixed; blocks live for that retry only.",
      "Back to C1 with the new layout; at most kMaxLayoutRetries (proposed 4) per seating, then exclude_mapping(seating).",
      "Count retries and retry successes in the placement stats.",
    ],
    code: [
      {
        path: "tt_metal/fabric/topology_mapper_utils.cpp  (MultiMeshSolutionEnumerator::next)",
        mode: "diff",
        language: "cpp",
        src: `
                 PipelineCheck check =
                     check_pipeline_seating(*pipelines_, logical_, full, *physical_system_descriptor_, mapping);
+                for (int retry = 0; !check.accepted && retry < kMaxLayoutRetries; ++retry) {
+                    auto blocks = slice_signature_blocks(check, full, mesh_graph_);   // empty when fixed by bindings
+                    if (blocks.empty()) {
+                        break;
+                    }
+                    auto again = complete_intra_mesh_for_placement(seating, /*extra=*/blocks, ...);
+                    if (!again.success) {
+                        break;
+                    }
+                    full = std::move(again);
+                    check = check_pipeline_seating(*pipelines_, logical_, full, *physical_system_descriptor_, mapping);
+                }
                 if (!check.accepted) {`,
      },
    ],
    tests: [
      "A seating whose first layout puts both cable ends under one slice and a second layout splits them: accepted after one retry (host-to-rank free).",
      "The same seating with every host rank bound: no retry, excluded as in P8.",
      "Retry cap: a seating with no working layout is excluded after kMaxLayoutRetries.",
      "The block admits exactly the layouts whose signature differs (brute force on small meshes).",
    ],
    behaviour: "Some seatings rejected by P8 are accepted with another layout; offline generate_rank_bindings finds more mappings.",
  },
  {
    id: "P10",
    title: "Pruning (later): exit-node and seat constraints before the first solve (C4)",
    milestone: "M4 Pruning",
    size: "M",
    deps: ["P3", "P8"],
    goal: "Later, optional speed-up, added only if P8's measured rejection counts call for it. Add a generic ExtraMappingConstraints to TopologyMappingConfig so a caller can pass extra hard constraints into the SAT placement and intra-mesh solves; the mapper applies them at its existing build points and knows nothing about pipelines. TopologyMapper fills them from Tier B's PipelineMappingConstraints (backup if slow: Tier B saves none in the first version). Start with the seat-pair (k_AB) drop.",
    files: [
      "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
      "tt_metal/fabric/physical_grouping_descriptor_matching.cpp",
      "tt_metal/fabric/topology_mapper_utils.cpp",
      "tt_metal/fabric/pipeline_builder.cpp",
      "tt_metal/fabric/topology_mapper.cpp",
    ],
    steps: [
      "API: ExtraMappingConstraints { allow_seat, allow_seat_pair, intra_mesh } on TopologyMappingConfig, in public types only (MeshId, PhysicalNodeId sets, MappingConstraints<FabricNodeId, PhysicalNodeId>). Empty by default: today's behaviour.",
      "SAT placement: MultiMeshSolutionEnumerator passes them to SatPlacementEnumerationSession. allow_seat runs in apply_extra_constraints (forbid the seats where it is false), so it is rebuilt with the session; allow_seat_pair runs in build_sat_placement_seat_graph, and a canonical seat pair is dropped only when it is false for every mesh pair that visits it.",
      "Intra-mesh: complete_intra_mesh_for_placement calls intra_mesh after add_exit_node_constraints, with the seated ASICs of every mesh; a returned message fails that mesh with failure_stage extra_constraints, like the exit-node failure today.",
      "Pipeline: detail::pipeline_extra_constraints(pmc, psd) in pipeline_builder.cpp builds the three callbacks from Tier B (seat: min_crossings_out; pair: k_AB candidate links; intra-mesh: ceil(k / links per exit ASIC) boundary chips on ASICs cabled toward n). TopologyMapper::build_mapping sets config.extra_constraints next to config.pipelines.",
      "stage_to_mesh_constraints(pmc) stays a report in pipeline_builder.cpp, not a solve (many stages share a mesh).",
      "Slice-level form (bipartite matching on distinct boundary slices) only where hard pinnings fix the layout.",
      "Single seat: remove only when it fails under every projection and layout.",
      "k_AB counts every crossing (P3), because every edge is STRICT. P14 changes it to skip RELAXED crossings."
    ],
    code: [
      {
        path: "tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
        mode: "new",
        language: "cpp",
        src: `
// topology_mapper_utils.hpp: generic, no pipeline types. Callers (TopologyMapper) fill it; empty = today.
struct ExtraMappingConstraints {
    using Asics = std::unordered_set<tt::tt_metal::experimental::PhysicalNodeId>;
    // SAT placement, unary: false forbids mesh on a seat with these ASICs.
    std::function<bool(MeshId mesh, const Asics& seat)> allow_seat;
    // SAT placement, pair: false drops the seat-graph edge between two seats of connected meshes.
    std::function<bool(MeshId a, const Asics& seat_a, MeshId b, const Asics& seat_b)> allow_seat_pair;
    // Intra-mesh: add hard constraints to mesh's solve once the seating is fixed, after rank, exit-node
    // and MGD-pin constraints. A returned message fails that mesh (failure_stage "extra_constraints").
    std::function<std::optional<std::string>(
        MeshId mesh,
        const std::map<MeshId, std::set<tt::tt_metal::experimental::PhysicalNodeId>>& seated_asics,
        MappingConstraints<FabricNodeId, tt::tt_metal::experimental::PhysicalNodeId>& constraints)> intra_mesh;
};

struct TopologyMappingConfig {
    // ...existing fields...
    ExtraMappingConstraints extra_constraints;
};`,
      },
      {
        path: "where the mapper applies them",
        mode: "new",
        language: "cpp",
        src: `
// physical_grouping_descriptor_matching.cpp: the session takes them from the enumerator's config.
//   allow_seat:      in apply_extra_constraints, add_forbidden_constraint(mesh, {seats where false}); rebuilt with the session.
//   allow_seat_pair: in build_sat_placement_seat_graph, skip a canonical seat pair only when it is false for
//                    every mesh pair that visits it (footprints are shared across meshes).
// topology_mapper_utils.cpp, complete_intra_mesh_for_placement, after add_exit_node_constraints:
if (config.extra_constraints.intra_mesh) {
    if (auto failure = config.extra_constraints.intra_mesh(logical_mesh_id, seated_asics, intra_mesh_constraints)) {
        record_failing_meshes(result, logical_mesh_id, physical_mesh_id);
        set_mapping_failure(result, "extra_constraints", *failure);
        return result;
    }
}`,
      },
      {
        path: "tt_metal/fabric/pipeline_builder.cpp, topology_mapper.cpp  (the pipeline fills them)",
        mode: "diff",
        language: "cpp",
        src: `
// pipeline_builder.cpp: P10 fills the generic hooks from Tier B; the mapper never sees pipeline types.
ExtraMappingConstraints detail::pipeline_extra_constraints(const PipelineMappingConstraints& pmc,
                                                           const PhysicalSystemDescriptor& psd);
//   allow_seat:      cables_leaving(seat, psd) >= pmc.min_crossings_out(mesh)
//   allow_seat_pair: k = pmc.min_crossings(a, b); k == 0 || candidate_links_between(seat_a, seat_b, psd) >= k
//   intra_mesh:      per neighbour n with k(m, n) > 0, add_cardinality_constraint(boundary chips of
//                    boundary_slices(m, n), ASICs cabled toward seated_asics[n], ceil(k / links per exit ASIC))

// topology_mapper.cpp, build_mapping, next to config.check_pipelines:
+   if (const auto* pmc = mesh_graph_descriptor.pipeline_mapping_constraints()) {
+       config.extra_constraints = detail::pipeline_extra_constraints(*pmc, physical_system_descriptor_);
+   }`,
      },
    ],
    tests: [
      "Empty ExtraMappingConstraints: every mapper golden unchanged.",
      "Generic: allow_seat false for one seat never yields it; allow_seat_pair false for one pair never yields both; intra_mesh returning a message fails that mesh with extra_constraints. Each survives a session rebuild (pool growth, add_forbidden_constraint).",
      "Pruning never removes a seating P8 would accept (property test over the sweep MGDs), for the seat, pair and intra-mesh hooks alike.",
      "pod8: the intra-mesh hook narrows M0's exit chips toward M1 to A0 and A3 (16 of 32), k = 2; with M0's layout pinned so that its cables land on A1 and A2, M0's intra-mesh solve is infeasible.",
      "A \"find all\" run over its budget (or skipped) gives k = 0: pipeline_extra_constraints returns empty hooks.",
      "Seatings tried before the first accept drop on mesh4x4_stage4x2 and superpod cases.",
    ],
    behaviour: "Faster; same accepted set.",
  },
  {
    id: "P11",
    title: "RELAXED edges: host cuts in Tier C",
    milestone: "M5 Later",
    size: "L",
    deps: ["P6", "P8", "P14"],
    goal: "After P14 lets the format say RELAXED. RELAXED pipeline edges become soft target edges in the same footprint solve (set_allow_unmatched_target_edges, #56762), so the solve picks the carve and the host cuts together; then one resolver call per fabric chain. No port of the host-cut planner. Until this lands, P14 has Tier C check RELAXED edges as STRICT (sound, but it misses seatings that need a host hop).",
    files: [
      "tt_metal/fabric/pipeline_builder.cpp",
      "tt-blaze: pipeline_builder/graph.py, host_edge_planner.py, blaze/models/cli_options.py",
    ],
    steps: [
      "In solve_pipeline_footprint (pipeline_builder.cpp), leave edges with needs_fabric = false out of the hard stage graph and pass them as soft target edges (allow unmatched). P14 already keeps the host I/O stage's outgoing edge as needs_fabric, as blaze never cuts it.",
      "The host cuts are the soft edges the solve's MappingResult leaves unmatched; check_pipeline_seating drops them from the resolver's edges. No new carve type.",
            "Resolve each fabric chain separately (blaze's _resolve_chains_separately), and record R only for matched crossings.",
      "Preference for fewest cuts and blaze's tie-breaks: open question 16.",
      "Drop P14's RELAXED-as-STRICT warning: Tier C now treats edges with needs_fabric = false as soft.",
      "tt-blaze, same change: Edge gets a policy (STRICT | RELAXED); _group_walk skips alignments that cut a STRICT edge instead of only '0 in cuts'; the fork planner cuts only RELAXED returns; --pipeline-builder flexible becomes a shortcut that marks every edge RELAXED.",
    ],
    code: [
      {
        path: "tt_metal/fabric/pipeline_builder.cpp  (solve_pipeline_footprint and check_pipeline_seating)",
        mode: "new",
        language: "cpp",
        src: `
// solve_pipeline_footprint: RELAXED edges may stay unmatched (carried over host).
for (const PipelineEdge& e : pipelines.edges_) {
    const bool hard = e.needs_fabric;   // P14: STRICT, or RELAXED out of the host I/O stage
    hard ? stage_adj.add_edge(e.src, e.dst) : soft_adj.add_edge(e.src, e.dst);
}
// ... the session gets soft_adj through set_allow_unmatched_target_edges (#56762)
const auto host_cuts = unmatched_soft_edges(footprint);   // read from the MappingResult

// check_pipeline_seating: host cuts are dropped; resolve each fabric chain on its own.
for (const auto& chain : fabric_chains(edges_minus(pipelines.edges_, host_cuts))) {
    auto layout = resolve_chain(chain, slots_of(m), links);
    if (!layout) { return reject(m, layout.error()); }
    record_required_links(*layout, R);        // matched crossings only
}`,
      },
    ],
    tests: [
      "Case B and Case C accepted with s3 -> s0 RELAXED, loopback on host; still rejected with every edge STRICT.",
      "mixed4 on (P, Q) with s3 -> s0 RELAXED: cover 1 skipped (it cuts STRICT s1 -> s2), cover 2 accepted.",
      "Agreement with host_edge_planner on rings and forks: same accept or reject, same number of host cuts.",
    ],
    behaviour: "Pipelines with RELAXED edges are checked and can be accepted with host hops.",
  },
  {
    id: "P12",
    title: "Write the witness plan",
    milestone: "M3 Tier C (STRICT)",
    size: "S",
    deps: ["P8"],
    goal: "Record each rank's stage in rank_bindings.yaml. Not consumed yet.",
    files: ["tools/scaleout/src/generate_rank_bindings.cpp"],
    steps: [
      "Read TopologyMappingResult::pipeline from the enumerator: each PipelinePlacement's stage_host_ranks give each rank its stage id; idle_host_ranks get idle.",
      "Co-owners of a joint stage get the same id.",
    ],
    code: [
      {
        path: "rank_bindings.yaml  (output)",
        mode: "plain",
        language: "yaml",
        src: `
rank_bindings:
  - rank: 3
    mesh_id: 1
    mesh_host_rank: 0
    env_overrides:
      TT_PIPELINE_STAGE: "s7"     # the mapper's witness; blaze may place differently`,
      },
    ],
    tests: ["Golden rank bindings for the sweep MGDs include TT_PIPELINE_STAGE; nothing reads it."],
    behaviour: "Extra field in the output file only.",
  },
  {
    id: "P13",
    title: "Later: seat-pair nogoods",
    milestone: "M4 Pruning",
    size: "M",
    deps: ["P3", "P8", "P9", "P15"],
    goal: "When a rejected seating is provably the fault of two meshes' seats, block that seat pair for every seating, not just the one seating P8 excludes. No solver change: the clause is one existing cardinality constraint.",
    files: [
      "tt_metal/api/tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp",
      "tt_metal/fabric/physical_grouping_descriptor_matching.cpp",
      "tt_metal/fabric/topology_mapper_utils.cpp",
      "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
      "tt_metal/fabric/pipeline_builder.cpp",
    ],
    steps: [
      "Session API: add_seat_pair_nogood(mesh_a, seat_a, mesh_b, seat_b) on SatPlacementEnumerationSession, built like add_forbidden_constraint: store in extra_nogoods_, trial-rebuild (pop and return false if no seating is left), then invalidate_pending_solve(). Seatings already yielded stay blocked through excluded_seat_maps().",
      "Encoding in apply_extra_constraints: 'mesh_a not on seat_a OR mesh_b not on seat_b' as add_cardinality_constraint(pairs, 1), where pairs is mesh_a's other seats plus mesh_b's other seats from get_valid_mappings. Exact because each mesh takes exactly one seat. Seats compare by ASIC set, because the domains hold canonical (footprint-shared) candidates.",
      "Twins: apply each nogood to every (x, y) in twin group(mesh_a) x twin group(mesh_b), x != y. The twin windowing in build_sat_placement_constraints keeps one representative per orbit; blocking one twin alone could cut the only representative of a working orbit. Factor the twin grouping into sat_placement_twin_groups so both use one definition.",
      "Why per mesh pair, not 'any mesh on seat_a and any mesh on seat_b': two other meshes on those seats carry different stages and cables, so the proof does not cover them.",
      "Why the ASIC set and not orientation: every producer below reads only orientation-free facts (footprints, cable counts, slice-level geometry under a hard-fixed signature). A future producer that reads chip positions must key on the canonical candidate instead.",
      "LogicalPipelineGraph gains tier_b_ (Tier B's min_crossings per mesh pair), filled by build_logical_pipeline_adjacency_graph from the descriptor's Tier B result, because check_pipeline_seating does not see the MGD.",
      "PipelineCheck gains seat_pair_proofs (mesh pairs). The P8 loop turns each into add_seat_pair_nogood before counting the rejection; false from the session ends the run with the existing could-not-fit error.",
      "Producer 1, before C1: cables between the two seats < Tier B's k_AB (a minimum over every projection). Redundant once P10's allow_seat_pair is in, and kept as the fallback when P10 is off.",
      "Producer 2, a C2 pre-check failure on mesh m, only when all three hold: the verdict holds for every projection; m's only MGD neighbour is n, so all of m's cables land on n's seat; and m's slice signature is fixed by hard constraints (P15's skip test). Resolver failures never produce a proof: one resolver run is one carve.",
      "Stats: pipeline_seat_pair_nogoods next to pipeline_rejections, so the sweep shows how much each producer prunes.",
    ],
    code: [
      {
        path: "tt_metal/api/tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp",
        mode: "diff",
        language: "cpp",
        src: `
 // physical_grouping_descriptor.hpp, SatPlacementEnumerationSession
     bool add_required_constraint(const PlacedMesh& placed);
     bool exclude_mapping(const AssignedMeshes& assigned);
+    // Block mesh_a on a seat with seat_a's ASICs together with mesh_b on a seat with seat_b's,
+    // and the same for every twin of either mesh (the meshes the symmetry break treats as
+    // interchangeable). The rest of a seating is free. Hard on every solve-ladder rung; kept
+    // across rebuilds like extra_forbidden_ / extra_required_. False: no seating is left.
+    bool add_seat_pair_nogood(
+        MeshId mesh_a, const std::unordered_set<PhysicalNodeId>& seat_a,
+        MeshId mesh_b, const std::unordered_set<PhysicalNodeId>& seat_b);
 private:
+    struct SeatPairNogood {
+        MeshId mesh_a;
+        std::unordered_set<PhysicalNodeId> seat_a;
+        MeshId mesh_b;
+        std::unordered_set<PhysicalNodeId> seat_b;
+    };
+    std::vector<SeatPairNogood> extra_nogoods_;
`,
      },
      {
        path: "tt_metal/fabric/physical_grouping_descriptor_matching.cpp",
        mode: "diff",
        language: "cpp",
        src: `
 // physical_grouping_descriptor_matching.cpp
+bool SatPlacementEnumerationSession::add_seat_pair_nogood(
+    MeshId mesh_a, const std::unordered_set<PhysicalNodeId>& seat_a,
+    MeshId mesh_b, const std::unordered_set<PhysicalNodeId>& seat_b) {
+    extra_nogoods_.push_back({mesh_a, seat_a, mesh_b, seat_b});
+    if (!ready_) {
+        return true;
+    }
+    MappingConstraints<GlobalMeshId, const Candidate*> trial;
+    if (build_sat_placement_constraints(*pools_, *physical_system_descriptor_, mesh_level_graph_, trial) &&
+        !apply_extra_constraints(trial)) {
+        extra_nogoods_.pop_back();
+        return false;
+    }
+    invalidate_pending_solve();   // re-encode with the clause; yielded seatings stay blocked (excluded_seat_maps)
+    return true;
+}

 bool SatPlacementEnumerationSession::apply_extra_constraints(
     MappingConstraints<GlobalMeshId, const Candidate*>& constraints) const {
     ...existing forbidden / required loops...
+    // "x not on seat_a OR y not on seat_b". The solver takes positive literals only, and each mesh
+    // takes exactly one seat, so "x on any seat but seat_a" is "x not on seat_a": one cardinality
+    // constraint with min_count 1 over x's other seats and y's other seats. No solver change.
+    for (const SeatPairNogood& ng : extra_nogoods_) {
+        for (const GlobalMeshId x : twins_of(ng.mesh_a)) {
+            for (const GlobalMeshId y : twins_of(ng.mesh_b)) {
+                if (x == y) {
+                    continue;
+                }
+                MappingConstraints<GlobalMeshId, const Candidate*>::CardinalityPairSet pairs;
+                bool x_can_take_a = false;
+                bool y_can_take_b = false;
+                for (const Candidate* seat : constraints.get_valid_mappings(x)) {   // canonical seats, set by build
+                    if (same_footprint(*seat, ng.seat_a)) { x_can_take_a = true; } else { pairs.insert({x, seat}); }
+                }
+                for (const Candidate* seat : constraints.get_valid_mappings(y)) {
+                    if (same_footprint(*seat, ng.seat_b)) { y_can_take_b = true; } else { pairs.insert({y, seat}); }
+                }
+                if (!x_can_take_a || !y_can_take_b) {
+                    continue;   // this combination cannot occur: nothing to block
+                }
+                if (!constraints.add_cardinality_constraint(pairs, 1)) {
+                    return false;
+                }
+            }
+        }
+    }
     return true;
 }

+// twins_of(m): m plus the meshes in its twin group (same pool class, same mesh-graph neighbours,
+// mutually non-adjacent). build_sat_placement_constraints windows each twin group's seats; a nogood
+// on one twin alone would break that symmetry and could cut the representative of a working orbit.
+// The grouping it does today moves into sat_placement_twin_groups(pools, mesh_level_graph), used by both.
`,
      },
      {
        path: "tt_metal/fabric/topology_mapper_utils.cpp",
        mode: "diff",
        language: "cpp",
        src: `
 // topology_mapper_utils.cpp, MultiMeshSolutionEnumerator::next, the pipeline branch (P8)
                if (!check.accepted) {                                         // C4
                    full.success = false;
                    full.stats.failure_stage = check.failure_stage;
                    full.error_message = check.reason;
                    last_failed_ = full;
+                   for (const auto& [mesh_a, mesh_b] : check.seat_pair_proofs) {
+                       if (!placement_session_->add_seat_pair_nogood(
+                               mesh_a, seat_asics(seating, mesh_a), mesh_b, seat_asics(seating, mesh_b))) {
+                           return give_up();
+                       }
+                       ++placement_stats_->pipeline_seat_pair_nogoods;
+                   }
                    if (++pipeline_rejections_ > kMaxPipelineRejections) {
                        return give_up();
                    }
                    continue;
                }

 // pipeline_builder.hpp
 struct PipelineCheck {
     bool accepted = false;
     std::string failure_stage;
     std::string reason;
+    // Mesh pairs whose two seats in this seating fail for every other seat of every other mesh.
+    std::vector<std::pair<MeshId, MeshId>> seat_pair_proofs;
 };
`,
      },
      {
        path: "tt_metal/fabric/pipeline_builder.cpp",
        mode: "diff",
        language: "cpp",
        src: `
 // pipeline_builder.cpp, detail::check_pipeline_seating
 // A producer may add a proof only when its verdict reads nothing but the two seats' footprints,
 // layouts fixed by hard constraints, and Tier B facts (true in every projection). Each one below
 // reads only orientation-free facts, so matching seats by ASIC set (as seats_matching does) is sound.

+    // 1. Before C1: fewer cables between the two seats than Tier B's minimum crossings (k_AB, a
+    //    minimum over every projection). Never fires once P10's allow_seat_pair is in.
++    //    k_AB is Tier B's, kept on the logical graph: pipelines.tier_b_ (P13 adds it, filled in
+    //    build_logical_pipeline_adjacency_graph from the descriptor's Tier B result).
+    for (const auto& [pair, k] : pipelines.tier_b_.min_crossings) {
+        const std::size_t cables =
+            cables_between(seat_asics(full, pair.first), seat_asics(full, pair.second), psd);
+        if (k > 0 && cables < k) {
+            PipelineCheck check = reject("pipeline_seat_pair",
+                fmt::format("meshes {} and {}: {} cable(s) between their seats, the pipeline crosses at least {}",
+                            *pair.first, *pair.second, cables, k));
+            check.seat_pair_proofs.push_back(pair);
+            return check;
+        }
+    }

     const PhysicalPipelineGraph positions = build_physical_pipeline_adjacency_graph(pipelines, logical, full, psd);   // C1
     if (auto pre = pipeline_pre_checks(pipelines, positions)) {
+        // 2. C2 pre-check: mesh m's segment test failed for every projection Tier B allows, m's only
+        //    MGD neighbour is n (so all of m's cables go to n's seat), and m's slice signature is fixed
+        //    by hard constraints (P15's skip test). Then no other seat of any other mesh changes it.
+        if (pre->mesh.has_value() && pre->every_projection) {
+            const MeshId m = *pre->mesh;
+            if (const auto n = only_mgd_neighbour(logical, m); n.has_value() && slice_signature_fixed(full, m)) {
+                check.seat_pair_proofs.push_back({m, *n});
+            }
+        }
         return reject(pre->stage, pre->reason, ...);
     }
`,
      },
    ],
    tests: [
      "Session: a nogood on (M0 -> P, M1 -> Q) blocks every seating with both, offers every other one (brute-force compare on a 3-mesh, 4-seat PSD), and survives a rebuild after a pool growth.",
      "Session: a nogood that leaves no seating returns false and is popped; the session still enumerates as before.",
      "Twins: ALL_TO_ALL of identical meshes with twin windowing on. A nogood on M0 also blocks its twins; the set of accepted seatings equals the brute-force set (no working orbit lost).",
      "Soft seams: the nogood still holds on the soft-seam rung of the solve ladder.",
      "Producer 1: seats with 1 cable between them and k_AB = 2 -> one nogood; with P10 off, no later seating puts that mesh pair on those seats.",
      "Producer 2: pod8 with every host bound: (P, U) fails on M1's corners -> nogood (M1 on U, M0 on P); unbound hosts -> no proof, P8 exclusion only.",
      "Soundness sweep: on every sweep MGD, the first accepted seating and its R are the same with and without P13; only the rejection count drops.",
    ],
    behaviour: "Fewer wasted seatings when a failure is provably about two seats. Only the SAT placement changes; the control plane and the binding are untouched. Without a pipeline nothing changes.",
  },
  {
    id: "P14",
    title: "RELAXED edges: schema + Tier B + Plan 1",
    milestone: "M5 Later",
    size: "M",
    deps: ["P1", "P3"],
    goal: "Let a pipeline edge say policy: RELAXED (may go over host MPI). Tier B, and the constraints P10 passes from the MGD into the first solve, stop counting edges that may go over host. Tier C keeps checking every edge as fabric until P11, which is sound: it never accepts a seating it cannot build.",
    files: [
      "tt_metal/fabric/protobuf/mesh_graph_descriptor.proto",
      "tt_metal/fabric/mesh_graph_descriptor.cpp",
      "tt_metal/api/tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp",
      "tt_metal/fabric/topology_mapper_utils.cpp",
      "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
      "tests/tt_metal/tt_fabric/fabric_router/test_mesh_graph_descriptor.cpp",
    ],
    steps: [
      "Proto: the field already exists (optional Policy policy = 3, the enum Channels.policy uses); only its comment changes. Omitted means STRICT, so every first-version file keeps its meaning.",
      "Tier A: drop the rejection of an explicit policy other than STRICT.",
      "Tier A: reject an explicit INVALID_POLICY. Nothing else: RELAXED is allowed on any edge.",
      "Descriptor data: PipelineDescriptorData::Edge gains bool relaxed (the header only forward-declares proto::Policy). populate_pipeline_descriptors sets it.",
      "Logical pipeline graph: PipelineEdge gains relaxed and needs_fabric. needs_fabric = STRICT, or RELAXED out of the host I/O stage, which blaze's host-cut planner never cuts. One rule, computed once in build_logical_pipeline_adjacency_graph, read by Tier C (P11); Tier B applies the same rule to the descriptor edges in mesh_graph_descriptor.cpp.",
      "Tier B: build nbrs and the per-mesh edge lists from needs_fabric edges only, so the crossing rule, the channel count, the look-ahead and the carve adjacency all skip host-capable edges.",
      "Plan 1 and P10: min_crossings (k_AB), min_crossings_out and boundary_slices come from the same filtered search, so P10's ExtraMappingConstraints hooks (the MGD's constraints passed into the first solve) need no code change; they just prune less. Every host-capable edge RELAXED gives k = 0 on those pairs.",
      "Tier C until P11: no change to the check; every edge stays a fabric edge. the enumerator constructor logs one warning per pipeline with RELAXED edges, and a failed run says RELAXED edges were checked as STRICT.",
      "Merge (1.5): nothing to do; the field rides in the proto that merge() copies.",
    ],
    code: [
      {
        path: "tt_metal/fabric/protobuf/mesh_graph_descriptor.proto",
        mode: "diff",
        language: "proto",
        src: `
 // Every edge rides a fabric link. policy may be omitted or STRICT. RELAXED is
-// rejected by MeshGraphDescriptor validation.
+// An edge may set policy: RELAXED to say it may go over host MPI when the
+// placement has no fabric link for it. Omitted means STRICT.
 ...
 message PipelineEdge {
     ...
     string dst = 2;

-    // Omitted means STRICT. RELAXED is rejected during descriptor validation.
+    // Optional; omitted is STRICT (same enum and default as Channels.policy).
+    // STRICT: the edge rides a fabric link. RELAXED: fabric where the placement
+    // allows, otherwise host MPI. INVALID_POLICY is rejected.
     optional Policy policy = 3;

     bool loopback = 4;
 }`,
      },
      {
        path: "tt_metal/fabric/mesh_graph_descriptor.cpp / .hpp",
        mode: "diff",
        language: "cpp",
        src: `
 // validate_pipeline_descriptors (Tier A), in the edge loop: RELAXED is allowed now.
-            // An omitted policy means STRICT. An explicit policy other than STRICT is rejected.
-            if (edge.has_policy() && edge.policy() != proto::Policy::STRICT) {
+            if (edge.has_policy() && edge.policy() == proto::Policy::INVALID_POLICY) {
                 ...
             }

 // populate_pipeline_descriptors:
-            record.edges.push_back(
-                PipelineDescriptorData::Edge{.src = edge.src(), .dst = edge.dst(), .loopback = edge.loopback()});
+            record.edges.push_back(PipelineDescriptorData::Edge{
+                .src = edge.src(),
+                .dst = edge.dst(),
+                .loopback = edge.loopback(),
+                .relaxed = edge.has_policy() && edge.policy() == proto::Policy::RELAXED});

 // mesh_graph_descriptor.hpp (proto::Policy is only forward-declared there, so a bool):
-// Edges stay directional pairs of stage ids, each with its loopback mark. Omitted and explicit
-// STRICT both survive validation, so a stored edge has no policy of its own.
+// Edges stay directional pairs of stage ids, each with its loopback mark. relaxed is
+// policy: RELAXED; omitted is STRICT.
 struct PipelineDescriptorData {
     struct Edge {
         std::string src;
         std::string dst;
         bool loopback = false;  // closes a cycle; the other edges have none
+        bool relaxed = false;
     };`,
      },
      {
        path: "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp / .cpp",
        mode: "diff",
        language: "cpp",
        src: `
 // topology_mapper_utils.hpp
 struct PipelineEdge {
     PipelineStageNode src, dst;
     bool loopback = false;       // loopback: true in the MGD
+    bool relaxed = false;        // policy: RELAXED in the MGD
+    // STRICT, or RELAXED out of stage 0: blaze's host-cut planner never cuts that
+    // (_group_walk skips cuts at stage 0). The one rule Tier B, Plan 1 and (P11) Tier C all read.
+    bool needs_fabric = true;
 };

 // topology_mapper_utils.cpp, build_logical_pipeline_adjacency_graph (host_io_stage_ is filled first)
-        graph.edges_.push_back(PipelineEdge{.src = src, .dst = dst, .loopback = edge.loopback});
+        graph.edges_.push_back(PipelineEdge{
+            .src = src, .dst = dst, .loopback = edge.loopback, .relaxed = edge.relaxed,
+            .needs_fabric = !edge.relaxed || src == graph.host_io_stage_.at(src.pipeline)});`,
      },
      {
        path: "tt_metal/fabric/pipeline_builder.cpp  (Tier B, P3)",
        mode: "diff",
        language: "cpp",
        src: `
 // mesh_graph_descriptor.cpp, check_pipeline_feasibility (Tier B): neighbour copies built once.
 // Every rule reads them, so one filter here covers the crossing rule, the channel count and, in
 // the backup plan, the look-ahead, the carve adjacency and the Plan 1 facts.
 for (const PipelineDescriptorData& pipeline : pipeline_descriptors_) {
     for (const PipelineDescriptorData::Edge& edge : pipeline.edges) {
+        if (edge.relaxed && edge.src != host_io_stage(pipeline)) {
+            continue;   // may go over host: no crossing, no channel, no adjacency, not in k_AB
+        }
         // one neighbour copy each way, as today
     }
 }

 // P10 (if landed) needs no change: pipeline_extra_constraints reads min_crossings, min_crossings_out and
 // boundary_slices, which are now computed over needs_fabric edges only, so the constraints passed from
 // the MGD into the first solve shrink with them. Every edge RELAXED (except stage 0's outgoing edge) gives k = 0.`,
      },
      {
        path: "tt_metal/fabric/pipeline_builder.cpp, topology_mapper_utils.cpp  (Tier C until P11)",
        mode: "diff",
        language: "cpp",
        src: `
 // topology_mapper_utils.cpp, MultiMeshSolutionEnumerator constructor (P8), once per run.
 // Until P11, Tier C checks every edge as a fabric edge. Sound: an accepted seating carries
 // every edge on fabric, so it is buildable; it can miss seatings that need a host hop.
+    std::map<std::string, std::size_t> relaxed;
+    for (const PipelineEdge& e : pipelines_->edges_) {
+        relaxed[e.src.pipeline] += e.relaxed ? 1 : 0;
+    }
+    for (const auto& [name, count] : relaxed) {
+        if (count > 0) {
+            log_warning(tt::LogFabric,
+                        "pipeline '{}': {} RELAXED edge(s) are checked as STRICT until host cuts are supported",
+                        name, count);
+            relaxed_checked_as_strict_ = true;
+        }
+    }

 // MultiMeshSolutionEnumerator::next, give_up(): say why a RELAXED pipeline found nothing.
+    if (pipelines_.has_value() && relaxed_checked_as_strict_) {
+        last_failed_->error_message += " (RELAXED pipeline edges are checked as STRICT until host cuts are supported)";
+    }`,
      },
    ],
    tests: [
      "EdgePolicyDoesNotParse becomes EdgePolicyParses: policy RELAXED stores relaxed = true; STRICT and omitted store false. policy: INVALID_POLICY is rejected by Tier A, naming the edge.",
      "UnknownFieldElsewhereStillParses and ShippedMeshGraphDescriptorsStillParse unchanged; every P1-P10 golden gives the same Tier B result.",
      "Pipeline graph: ring with s3 -> s0 RELAXED: needs_fabric false. s0 -> s1 RELAXED: needs_fabric true (stage 0).",
      "Tier B: mixed4 with no MGD connection: s1 -> s2 and s3 -> s0 RELAXED passes; only s3 -> s0 RELAXED still rejects.",
      "Plan 1: mesh4x4_stage4x2 with s3 -> s0 RELAXED gives k(M0, M1) = 1; every cuttable edge RELAXED gives k = 0, and P10's hooks add nothing for that pair.",
      "Tier C: a RELAXED pipeline accepts exactly the seatings the all-STRICT copy accepts, and logs the warning once.",
      "Merge: a part with a RELAXED edge keeps it after merge().",
    ],
    behaviour: "Files can say RELAXED. Tiers A and B, and P10's pruning, accept more; Tier C still requires fabric for every edge until P11.",
  },
  {
    id: "P16",
    title: "C1 on the policy plan's candidate table (part 9)",
    milestone: "M5 Later",
    size: "S",
    deps: ["P5", "P9"],
    goal: "Next step, after the inter-mesh policy plan's part 9 lands. Replace P5's PSD read (inter_mesh_cables) with collect_inter_mesh_candidates, so C1 and the control plane read one table: links that are up, on allowed cable types, ordered by tier.",
    files: ["tt_metal/fabric/pipeline_builder.cpp", "tt_metal/fabric/topology_mapper_utils.cpp"],
    steps: [
      "Depends on the policy plan's part 9 (candidate table, down-link hashes in the chip info, PortTypeRules).",
      "Call collect_inter_mesh_candidates(mesh_graph, psd, full.fabric_node_to_asic, options) per seating from MultiMeshSolutionEnumerator::next, with the same InterMeshCandidateOptions as the final call; delete inter_mesh_cables.",
      "Pick R's channel by best tier, then lowest src_chan; order DirectLinks by tier, then chip id.",
      "Phase 0 in the pairing takes the candidate table's PortDescriptorTable (rank 0, no gather); the check after the broadcast stays.",
      "Close open question 20: down links never reach C1.",
    ],
    code: [],
    tests: [
      "No cable data and no down links: same R and same AnnotatedIntermeshConnections as with inter_mesh_cables.",
      "A disallowed cable type or a down link between two stages is never in R; the seating uses another cable or is rejected.",
    ],
    behaviour: "C1 skips down links and disallowed cable types; R prefers better tiers.",
  },
];

const MILESTONES: Array<{ id: Pr["milestone"]; text: string }> = [
  { id: "M1 Format", text: "The MGD can carry a pipeline, survives merge, and impossible pipelines fail at load." },
  { id: "M2 Shared code", text: "Libraries Tier C needs, each landing with no behaviour change." },
  { id: "M3 Tier C (STRICT)", text: "The mapper only emits seatings the pipeline builds on, and binds the cables it relied on." },
  { id: "M4 Pruning", text: "Speed-ups. Not needed for correctness." },
  { id: "M5 Later", text: "Not in the first version. RELAXED: edge policy in the schema, then host cuts in Tier C. Candidate table: C1 switches from its own PSD read (P5) to the policy plan's collect_inter_mesh_candidates (part 9). M1-M4 are complete without it: every edge is STRICT, C1 reads the PSD." },
];

// ---------------------------------------------------------------------------
// PR DAG
// ---------------------------------------------------------------------------

function PrDag({ selected, onSelect }: { selected: string; onSelect: (id: string) => void }) {
  const t = useHostTheme();
  const nodeW = 150;
  const nodeH = 44;
  const layout = computeDAGLayout({
    nodes: PRS.map((p) => ({ id: p.id })),
    edges: PRS.flatMap((p) => p.deps.map((d) => ({ from: d, to: p.id }))),
    direction: "horizontal",
    nodeWidth: nodeW,
    nodeHeight: nodeH,
    rankGap: 46,
    nodeGap: 14,
    padding: 12,
  });
  const byId = new Map(PRS.map((p) => [p.id, p]));
  return (
    <ScrollX>
      <svg width="100%" viewBox={`0 0 ${layout.width} ${layout.height}`} style={{ maxWidth: layout.width, minWidth: 0.8 * (layout.width) }}>
        <ArrowDefs id="dag-arrow" color={t.stroke.primary} />
        {layout.edges.map((e) => {
          const mid = (e.sourceX + e.targetX) / 2;
          const hot = e.from === selected || e.to === selected;
          return (
            <path
              key={`${e.from}-${e.to}`}
              d={`M ${e.sourceX} ${e.sourceY} C ${mid} ${e.sourceY}, ${mid} ${e.targetY}, ${e.targetX - 2} ${e.targetY}`}
              fill="none"
              stroke={hot ? t.accent.primary : t.stroke.secondary}
              strokeWidth={hot ? 1.6 : 1}
              markerEnd="url(#dag-arrow)"
            />
          );
        })}
        {layout.nodes.map((n) => {
          const p = byId.get(n.id)!;
          const isSel = n.id === selected;
          return (
            <g key={n.id} onClick={() => onSelect(n.id)} style={{ cursor: "pointer" }}>
              <rect
                x={n.x}
                y={n.y}
                width={nodeW}
                height={nodeH}
                rx={6}
                fill={isSel ? t.fill.secondary : t.fill.quaternary}
                stroke={isSel ? t.accent.primary : t.stroke.primary}
                strokeWidth={isSel ? 1.8 : 1}
              />
              <text x={n.x + 8} y={n.y + 17} fontSize={11.5} fontWeight={600} fill={t.text.primary}>
                {p.id} · {p.size}
              </text>
              <text x={n.x + 8} y={n.y + 33} fontSize={10.5} fill={t.text.secondary}>
                {p.title.length > 24 ? p.title.slice(0, 23) + "…" : p.title}
              </text>
            </g>
          );
        })}
      </svg>
    </ScrollX>
  );
}

function PrDetail({ pr }: { pr: Pr }) {
  return (
    <Stack gap={14}>
      <Row gap={10} align="center" wrap>
        <H2 style={{ margin: 0 }}>
          {pr.id} · {pr.title}
        </H2>
        <Pill size="sm">{pr.milestone}</Pill>
        <Pill size="sm">size {pr.size}</Pill>
        {pr.deps.length ? <Pill size="sm">after {pr.deps.join(", ")}</Pill> : <Pill size="sm">no dependencies</Pill>}
      </Row>
      <Text>{pr.goal}</Text>
      <Grid columns={fit(320)} gap={20} align="start">
        <Stack gap={6}>
          <H3>Steps</H3>
          {pr.steps.map((s, i) => (
            <div key={i}>
              <Text tone="secondary">
                {i + 1}. {s}
              </Text>
            </div>
          ))}
        </Stack>
        <Stack gap={6}>
          <H3>Files</H3>
          {pr.files.map((f) => (
            <div key={f}>
              <Text size="small">
                <Code>{f}</Code>
              </Text>
            </div>
          ))}
          <H3 style={{ marginTop: 8 }}>Behaviour change</H3>
          <Text size="small" tone="secondary">
            {pr.behaviour}
          </Text>
        </Stack>
      </Grid>
      {pr.code.length ? (
        <Stack gap={10}>
          <H3>Code</H3>
          {pr.code.map((c) => (
            <div key={c.path}>
              <CodeCard path={c.path} src={c.src} mode={c.mode} language={c.language} note={c.note} />
            </div>
          ))}
        </Stack>
      ) : null}
      <Stack gap={6}>
        <H3>Tests / done when</H3>
        {pr.tests.map((s, i) => (
          <div key={i}>
            <Text tone="secondary">- {s}</Text>
          </div>
        ))}
      </Stack>
    </Stack>
  );
}

function scrollToId(id: string) {
  if (typeof document === "undefined") return;
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

const TIER_C_FILES: string[][] = [
  ["topology_mapper_utils.hpp", "every Tier C type: LogicalPipelineGraph, PhysicalPipelineGraph, PipelineMappingResult (the carve is the solver's own MappingResult); TopologyMappingConfig::check_pipelines; TopologyMappingResult::pipeline", "P5, P6, P8"],
  ["topology_mapper_utils.cpp", "build_logical_pipeline_adjacency_graph; the enumerator's pipelines_, the next() hook, exclude_mapping and the rejection cap", "P5, P8"],
  ["pipeline_builder.hpp (detail)", "forward declarations; build_physical_pipeline_adjacency_graph, solve_pipeline_footprint, PipelineCheck, check_pipeline_seating", "P5, P6"],
  ["pipeline_builder.cpp", "per seating: inter_mesh_cables, the physical graph, pre-checks, the footprint solve, resolver arguments and one call per pipeline, R, the demand check", "P5, P6"],
  ["topology_mapper.cpp", "kill switch; the mapping saved on rank 0; R broadcast with the chip info", "P8"],
  ["control_plane.cpp", "phase 0 binds R; every rank checks it was bound", "P9"],
  ["test_topology_mapper_utils.cpp / test_pipeline_builder.cpp", "logical graph and the hook / physical graph, solve and check on CPU PSD fixtures", "P5, P6, P8"],
];

const TIER_C_FLOW = `
once per run (topology_mapper_utils.cpp, enumerator constructor)
  pipelines_ = build_logical_pipeline_adjacency_graph(merged MGD)                   P5

per seating (MultiMeshSolutionEnumerator::next, after intra-mesh success)
  detail::check_pipeline_seating(*pipelines_, logical_, full, psd, mapping)       P6, pipeline_builder.cpp
    C1  build_physical_pipeline_adjacency_graph    positions overlap; edges join disjoint, bordering ones
    C2  pipeline_pre_checks                        area, lattice, colour and segment
        solve_pipeline_footprint                   P5: required + resource + completeness, conflict_cap;
                                                   returns the solver's MappingResult (target_to_global = the carve)
        resolver on the carve                      a throw rejects
        record_required_links -> R                 exact channels
        first_over_demand(logical, R)              strict + relaxed per mesh pair
        PipelinePlacement per pipeline             submesh_chips + GraphLayoutResult, as returned
  reject: exclude_mapping(seating), next seating                                  P8 (C4); P15 retries first
  accept: full.pipeline = mapping; TopologyMapper saves it, broadcasts R          P8
          control plane phase 0 binds R                                           P9 (C3)`;

function PlanTab() {
  const [selected, setSelected] = useCanvasState("pr", "P8");
  const t = useHostTheme();
  const select = (id: string) => {
    setSelected(id);
    scrollToId(`pr-${id}`);
  };
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H2>Fourteen PRs: STRICT in M1-M4; RELAXED and the policy plan's candidate table later in M5</H2>
        <Text tone="secondary">
          The first version is STRICT only: every pipeline edge rides fabric, C1 reads inter-mesh cables from the PSD itself, and M1-M4 need nothing from M5. Arrows are
          hard dependencies. M2 (P5 then P6) is independent of M1, so it can land in parallel with the format work. Every PR
          is written out below with its steps, code and tests; click a box to jump to it.
        </Text>
      </Stack>
      <Stack gap={8}>
        <H2>Tier C at a glance</H2>
        <Text tone="secondary">
          One rule: topology_mapper_utils owns the types and what runs once per run (the logical pipeline graph, the enumerator hook, C4);
          pipeline_builder owns everything that runs per seating, behind one call, detail::check_pipeline_seating. The mapper calls it after
          every intra-mesh success and reads back accept or reject. pipeline_builder.hpp only forward-declares the graph types, so the ttnn
          ops and nanobind module that include it are unchanged.
        </Text>
        <Table
          headers={["File", "Holds", "PR"]}
          rows={TIER_C_FILES}
        />
        <CodeCard path="one run, one seating" mode="plain" language="text" src={TIER_C_FLOW} />
      </Stack>
      <PrDag selected={selected} onSelect={select} />
      <Table
        headers={["Milestone", "PRs", "Outcome"]}
        rows={MILESTONES.map((m) => [
          m.id,
          PRS.filter((p) => p.milestone === m.id)
            .map((p) => `${p.id} ${p.title}`)
            .join(" · "),
          m.text,
        ])}
      />
      {MILESTONES.map((m) => (
        <div key={m.id}>
          <Stack gap={18}>
            <div style={{ borderTop: `2px solid ${t.stroke.primary}`, paddingTop: 12 }}>
              <Stack gap={4}>
                <H2 style={{ margin: 0 }}>{m.id}</H2>
                <Text tone="secondary">{m.text}</Text>
              </Stack>
            </div>
            {PRS.filter((p) => p.milestone === m.id).map((p) => (
              <div
                key={p.id}
                id={`pr-${p.id}`}
                style={{
                  borderLeft: `3px solid ${p.id === selected ? t.accent.primary : t.stroke.tertiary}`,
                  paddingLeft: 16,
                  scrollMarginTop: 12,
                }}
              >
                <PrDetail pr={p} />
              </div>
            ))}
          </Stack>
        </div>
      ))}
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Worked examples: what the code checks, constrains and places
// ---------------------------------------------------------------------------

type SliceStage = { mesh: 0 | 1; from: number; to: number; label: string; idle?: boolean };
type SliceCable = { a: number; b: number; kind?: CableKind; label?: string };

/** Two meshes drawn as rows of four 4x2 host slices; stages are boxes over whole slices. */
function SliceMap(props: {
  stages: SliceStage[];
  cables: SliceCable[];
  intraBreak?: boolean;
  host?: { from: [0 | 1, number]; to: [0 | 1, number]; label: string };
  verdict?: { ok: boolean; text: string };
}) {
  const t = useHostTheme();
  const sw = 92;
  const sh = 46;
  const x0 = 44;
  const y = [44, 150] as const;
  const W = x0 + 4 * sw + 20;
  const H = 228;
  const cx = (i: number) => x0 + i * sw + sw / 2;
  const cableColor = (k?: CableKind) =>
    k === "required" ? t.accent.primary : k === "unused" ? t.stroke.secondary : t.text.primary;
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W, minWidth: 0.8 * W }}>
      <ArrowDefs id="sm-host" color={t.category.orange} />
      {([0, 1] as const).map((m) => (
        <g key={m}>
          <text x={4} y={y[m] + sh / 2 + 4} fontSize={12} fontWeight={600} fill={t.text.secondary}>
            {m === 0 ? "M0" : "M1"}
          </text>
          {[0, 1, 2, 3].map((i) => (
            <g key={i}>
              <rect
                x={x0 + i * sw}
                y={y[m]}
                width={sw}
                height={sh}
                fill={t.fill.quaternary}
                stroke={t.stroke.secondary}
                strokeDasharray="3 3"
              />
              <text x={cx(i)} y={m === 0 ? y[m] - 6 : y[m] + sh + 14} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
                {(m === 0 ? "A" : "B") + i} · rank {m * 4 + i}
              </text>
            </g>
          ))}
        </g>
      ))}
      {props.stages.map((s) => (
        <g key={`${s.mesh}-${s.from}-${s.label}`}>
          <rect
            x={x0 + s.from * sw + 3}
            y={y[s.mesh] + 3}
            width={(s.to - s.from + 1) * sw - 6}
            height={sh - 6}
            rx={5}
            fill={s.idle ? "none" : t.fill.secondary}
            stroke={s.idle ? t.stroke.tertiary : t.accent.primary}
            strokeDasharray={s.idle ? "3 3" : undefined}
          />
          <text
            x={(cx(s.from) + cx(s.to)) / 2}
            y={y[s.mesh] + sh / 2 + 4}
            textAnchor="middle"
            fontSize={12}
            fontWeight={s.idle ? 400 : 600}
            fill={s.idle ? t.text.tertiary : t.text.primary}
          >
            {s.label}
          </text>
        </g>
      ))}
      {props.cables.map((c, i) => (
        <g key={i}>
          <line
            x1={cx(c.a) + (c.label === "2nd" ? 10 : 0)}
            y1={y[0] + sh}
            x2={cx(c.b) + (c.label === "2nd" ? 10 : 0)}
            y2={y[1]}
            stroke={cableColor(c.kind)}
            strokeWidth={c.kind === "unused" ? 1.2 : 2.2}
            strokeDasharray={c.kind === "unused" ? "4 3" : undefined}
          />
        </g>
      ))}
      {props.host
        ? (() => {
            const a = props.host.from;
            const b = props.host.to;
            const ax = cx(a[1]);
            const bx = cx(b[1]);
            const ay = a[0] === 0 ? y[0] : y[1] + sh;
            const by = b[0] === 0 ? y[0] : y[1] + sh;
            const lift = a[0] === 0 ? -1 : 1;
            return (
              <g>
                <path
                  d={`M ${ax} ${ay} C ${ax} ${ay + lift * 34}, ${bx} ${by + lift * 34}, ${bx} ${by + lift * 2}`}
                  fill="none"
                  stroke={t.category.orange}
                  strokeWidth={1.8}
                  strokeDasharray="5 3"
                  markerEnd="url(#sm-host)"
                />
                <text x={(ax + bx) / 2} y={a[0] === 0 ? 10 : H - 2} textAnchor="middle" fontSize={10.5} fill={t.category.orange}>
                  {props.host.label}
                </text>
              </g>
            );
          })()
        : null}
      {props.verdict ? (
        <text x={W - 6} y={y[1] - 30} textAnchor="end" fontSize={11} fontWeight={600} fill={props.verdict.ok ? t.category.green : t.category.red}>
          {props.verdict.text}
        </text>
      ) : null}
    </svg>
  );
}

/** One galaxy as eight 4x1 rank columns on a column ring; stages are boxes over whole columns. */
function ColumnRing(props: {
  stages: Array<{ from: number; to: number; label: string; bad?: boolean }>;
  caption: string;
  verdict: { ok: boolean; text: string };
}) {
  const t = useHostTheme();
  const cw = 56;
  const ch = 70;
  const x0 = 14;
  const y0 = 28;
  const W = x0 * 2 + 8 * cw;
  const H = 172;
  const cx = (i: number) => x0 + i * cw + cw / 2;
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W, minWidth: 0.8 * W }}>
      <text x={x0} y={14} fontSize={11} fill={t.text.secondary}>
        {props.caption}
      </text>
      {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
        <g key={i}>
          <rect x={x0 + i * cw} y={y0} width={cw} height={ch} fill={t.fill.quaternary} stroke={t.stroke.secondary} strokeDasharray="3 3" />
          <text x={cx(i)} y={y0 + ch + 14} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
            C{i} · r{i}
          </text>
        </g>
      ))}
      {props.stages.map((s) => (
        <g key={`${s.from}-${s.label}`}>
          <rect
            x={x0 + s.from * cw + 3}
            y={y0 + 3}
            width={(s.to - s.from + 1) * cw - 6}
            height={ch - 6}
            rx={5}
            fill={s.bad ? "none" : t.fill.secondary}
            stroke={s.bad ? t.category.red : t.accent.primary}
            strokeDasharray={s.bad ? "4 3" : undefined}
          />
          <text x={(cx(s.from) + cx(s.to)) / 2} y={y0 + ch / 2 + 4} textAnchor="middle" fontSize={12} fontWeight={600} fill={s.bad ? t.category.red : t.text.primary}>
            {s.label}
          </text>
        </g>
      ))}
      <path
        d={`M ${cx(7)} ${y0 + ch + 20} C ${cx(7)} ${H - 26}, ${cx(0)} ${H - 26}, ${cx(0)} ${y0 + ch + 20}`}
        fill="none"
        stroke={t.text.secondary}
        strokeWidth={1.4}
      />
      <text x={(cx(0) + cx(7)) / 2} y={H - 4} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
        column wrap C7 to C0 (RING)
      </text>
      <text x={W - x0} y={14} textAnchor="end" fontSize={11} fontWeight={600} fill={props.verdict.ok ? t.category.green : t.category.red}>
        {props.verdict.text}
      </text>
    </svg>
  );
}

/** Three pods as SAT seats, with the seat-graph link counts between them. */
function SeatGraph() {
  const t = useHostTheme();
  const W = 380;
  const H = 210;
  const pos: Record<string, [number, number]> = { P: [190, 40], Q: [70, 170], R: [310, 170] };
  const edges: Array<{ a: string; b: string; label: string; dropped?: boolean }> = [
    { a: "P", b: "Q", label: "2 links" },
    { a: "P", b: "R", label: "2 links" },
    { a: "Q", b: "R", label: "1 link: dropped once Plan 1 lands (k = 2)", dropped: true },
  ];
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W, minWidth: 0.8 * W }}>
      {edges.map((e) => {
        const [ax, ay] = pos[e.a];
        const [bx, by] = pos[e.b];
        const x1 = e.dropped ? ax + 46 : ax;
        const y1 = e.dropped ? ay : ay + 18;
        const x2 = e.dropped ? bx - 46 : bx;
        const y2 = e.dropped ? by : by - 18;
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        return (
          <g key={e.a + e.b}>
            <line
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={e.dropped ? t.category.red : t.accent.primary}
              strokeWidth={e.dropped ? 1.5 : 2.4}
              strokeDasharray={e.dropped ? "5 4" : undefined}
            />
            <text
              x={e.dropped ? mx : mx + (e.a === "P" && e.b === "Q" ? -34 : 34)}
              y={e.dropped ? my + 34 : my}
              textAnchor="middle"
              fontSize={11}
              fill={e.dropped ? t.category.red : t.text.secondary}
            >
              {e.label}
            </text>
          </g>
        );
      })}
      {Object.entries(pos).map(([name, [x, y]]) => (
        <g key={name}>
          <rect x={x - 46} y={y - 18} width={92} height={36} rx={6} fill={t.fill.secondary} stroke={t.stroke.primary} />
          <text x={x} y={y - 2} textAnchor="middle" fontSize={13} fontWeight={600} fill={t.text.primary}>
            pod {name}
          </text>
          <text x={x} y={y + 12} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
            4x8, hosts h0..h3
          </text>
        </g>
      ))}
    </svg>
  );
}

/** M0 as four 4x2 slices above M1, a 2x4 mesh with one 1x2 slice per rank. */
function SmallMeshMap(props: {
  m0: string[];
  m1: string[];
  cables: Array<{ a: number; row: number; col: number }>;
  verdict: { ok: boolean; text: string };
}) {
  const t = useHostTheme();
  const sw = 80;
  const x0 = 44;
  const W = x0 + 4 * sw + 120;
  const yA = 26;
  const cell = 46;
  const mx = x0 + 2 * sw - 2 * cell;
  const yB = 124;
  const H = yB + 2 * 30 + 24;
  const tiles = [
    { name: "T0", r: 0, c: 0, rank: 4 },
    { name: "T1", r: 0, c: 2, rank: 5 },
    { name: "T2", r: 1, c: 0, rank: 6 },
    { name: "T3", r: 1, c: 2, rank: 7 },
  ];
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W, minWidth: 0.8 * W }}>
      <text x={4} y={yA + 25} fontSize={12} fontWeight={600} fill={t.text.secondary}>
        M0
      </text>
      <text x={4} y={yB + 34} fontSize={12} fontWeight={600} fill={t.text.secondary}>
        M1
      </text>
      {props.m0.map((label, i) => (
        <g key={i}>
          <rect x={x0 + i * sw + 3} y={yA + 3} width={sw - 6} height={38} rx={5} fill={t.fill.secondary} stroke={t.accent.primary} />
          <text x={x0 + i * sw + sw / 2} y={yA + 26} textAnchor="middle" fontSize={12} fontWeight={600} fill={t.text.primary}>
            {label}
          </text>
          <text x={x0 + i * sw + sw / 2} y={yA - 6} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
            A{i} · rank {i}
          </text>
        </g>
      ))}
      {tiles.map((tile, i) => (
        <g key={tile.name}>
          <rect
            x={mx + tile.c * cell + 2}
            y={yB + tile.r * 30 + 2}
            width={2 * cell - 4}
            height={26}
            rx={4}
            fill={t.fill.secondary}
            stroke={t.accent.primary}
          />
          <text x={mx + tile.c * cell + cell} y={yB + tile.r * 30 + 19} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>
            {tile.name} {props.m1[i]}
          </text>
          <text
            x={tile.c === 0 ? mx - 6 : mx + 4 * cell + 6}
            y={yB + tile.r * 30 + 19}
            textAnchor={tile.c === 0 ? "end" : "start"}
            fontSize={10}
            fill={t.text.tertiary}
          >
            rank {tile.rank}
          </text>
        </g>
      ))}
      {[1, 2, 3].map((c) => (
        <line key={c} x1={mx + c * cell} y1={yB + 2} x2={mx + c * cell} y2={yB + 58} stroke={t.stroke.tertiary} strokeDasharray="2 3" />
      ))}
      {props.cables.map((c, i) => (
        <g key={i}>
          <line
            x1={x0 + c.a * sw + sw / 2}
            y1={yA + 41}
            x2={mx + c.col * cell + cell / 2}
            y2={yB + c.row * 30 + 15}
            stroke={props.verdict.ok ? t.accent.primary : t.category.red}
            strokeWidth={2}
          />
          <circle cx={mx + c.col * cell + cell / 2} cy={yB + c.row * 30 + 15} r={3.5} fill={props.verdict.ok ? t.accent.primary : t.category.red} />
        </g>
      ))}
      <text x={W - 4} y={yB + 34} textAnchor="end" fontSize={11} fontWeight={600} fill={props.verdict.ok ? t.category.green : t.category.red}>
        {props.verdict.text}
      </text>
    </svg>
  );
}

const RUN_MIXED4 = `
mesh_descriptors {
  name: "M"  arch: BLACKHOLE
  device_topology { dims: [ 4, 8 ] }
  host_topology   { dims: [ 1, 4 ] }   # 4 ranks per mesh, each a 4x2 slice
  channels { count: 2 policy: RELAXED }
}

graph_descriptors {
  name: "G"  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M" mesh_id: 0 } }   # M0: slices A0..A3, ranks 0-3
  instances { mesh { mesh_descriptor: "M" mesh_id: 1 } }   # M1: slices B0..B3, ranks 4-7
  graph_topology { layout_type: ALL_TO_ALL channels { count: 2 policy: RELAXED } }
}
top_level_instance { graph { graph_descriptor: "G" graph_id: 0 } }

# 2 meshes x 4 slices = 8 slices; the ring below needs 1 + 2 + 2 + 1 = 6
pipeline_descriptors {
  name: "mixed4"
  stages { id: "s0" shape { dims: [ 4, 2 ] } }   # 1 slice
  stages { id: "s1" shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: "s2" shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: "s3" shape { dims: [ 4, 2 ] } }   # 1 slice
  edges { src: "s0" dst: "s1" }  edges { src: "s1" dst: "s2" }
  edges { src: "s2" dst: "s3" }  edges { src: "s3" dst: "s0" }
}`;

const RUN_POD8 = `
mesh_descriptors {
  name: "Big"  arch: BLACKHOLE
  device_topology { dims: [ 4, 8 ] }
  host_topology   { dims: [ 1, 4 ] }   # 4 ranks, each a 4x2 slice
  channels { count: 2 policy: RELAXED }
}

mesh_descriptors {
  name: "Small"  arch: BLACKHOLE
  device_topology { dims: [ 2, 4 ] }
  host_topology   { dims: [ 2, 2 ] }   # 4 ranks, each a 1x2 slice
  channels { count: 2 policy: RELAXED }
}

graph_descriptors {
  name: "G"  type: "FABRIC"
  instances { mesh { mesh_descriptor: "Big"   mesh_id: 0 } }   # M0: slices A0..A3, ranks 0-3
  instances { mesh { mesh_descriptor: "Small" mesh_id: 1 } }   # M1: slices T0..T3, ranks 4-7
  graph_topology { layout_type: ALL_TO_ALL channels { count: 2 policy: RELAXED } }
}
top_level_instance { graph { graph_descriptor: "G" graph_id: 0 } }

# 2 meshes x 4 slices = 8 slices; the ring below needs 8 x 1 = 8
pipeline_descriptors {
  name: "pod8"
  stages { id: "s0" shape { dims: [ 4, 2 ] } }   # 1 slice of Big
  stages { id: "s1" shape { dims: [ 4, 2 ] } }
  stages { id: "s2" shape { dims: [ 4, 2 ] } }
  stages { id: "s3" shape { dims: [ 4, 2 ] } }
  stages { id: "s4" shape { dims: [ 1, 2 ] } }   # 1 slice of Small
  stages { id: "s5" shape { dims: [ 1, 2 ] } }
  stages { id: "s6" shape { dims: [ 1, 2 ] } }
  stages { id: "s7" shape { dims: [ 1, 2 ] } }
  edges { src: "s0" dst: "s1" }  edges { src: "s1" dst: "s2" }
  edges { src: "s2" dst: "s3" }  edges { src: "s3" dst: "s4" }
  edges { src: "s4" dst: "s5" }  edges { src: "s5" dst: "s6" }
  edges { src: "s6" dst: "s7" }  edges { src: "s7" dst: "s0" }
}`;

const RUN_MACHINE = `
pods P, Q, R: 4x8 grids, four hosts h0..h3 (4x2 each)
boards U, V:  2x4, four 2-chip hosts

P.h0 - Q.h0    P.h3 - Q.h3       P-Q: 2 links
P.h0 - R.h0    P.h2 - R.h2       P-R: 2 links
Q.h1 - R.h1                      Q-R: 1 link
P.h0, P.h3 - two chips of U      P-U: 2 links
P.h0, P.h3 - two chips of V      P-V: 2 links

assumed layout: mesh host rank i on host hi of its seat
  (M0 on P, M1 on Q): cables on A0-B0, A3-B3
  (M0 on P, M1 on R): cables on A0-B0, A2-B2`;

function RunningExamples() {
  return (
    <Stack gap={10}>
      <H2>Running examples</H2>
      <Text tone="secondary">
        Every part below ends with examples of what that step checks, including its edge cases. They are hypothetical, traced by
        hand through the code as it stands; nothing was run. Slices of a 4x8 mesh with host_topology [1, 4] are A0..A3 on M0
        (ranks 0-3) and B0..B3 on M1 (ranks 4-7); inside a mesh they form a line.
      </Text>
      <Grid columns={fit(360)} gap={16} align="start">
        <CodeCard path="mixed4: stages that span two ranks" mode="plain" language="protobuf" src={RUN_MIXED4} />
        <Stack gap={12}>
          <CodeCard path="pod8: different slice shapes (llama_8b_pod pattern)" mode="plain" language="protobuf" src={RUN_POD8} />
          <CodeCard path="the machine" mode="plain" language="text" src={RUN_MACHINE} />
        </Stack>
      </Grid>
    </Stack>
  );
}

// --- Descriptor tab: Tier A and Tier B -------------------------------------

/** mixed4 ring with each stage painted by the mesh a projection assigns it to. A red edge crosses M0/M1. */
function ProjectionRing(props: { title: string; mesh: [0 | 1, 0 | 1, 0 | 1, 0 | 1]; verdict: string; ok: boolean }) {
  const t = useHostTheme();
  const W = 210;
  const H = 148;
  const pos: Array<[number, number]> = [
    [48, 36],
    [162, 36],
    [162, 100],
    [48, 100],
  ];
  const names = ["s0", "s1", "s2", "s3"];
  const edges: Array<[number, number]> = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
  ];
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W }}>
      <text x={4} y={14} fontSize={11} fontWeight={600} fill={t.text.primary}>
        {props.title}
      </text>
      {edges.map(([a, b], i) => {
        const cross = props.mesh[a] !== props.mesh[b];
        const [x1, y1] = pos[a];
        const [x2, y2] = pos[b];
        const dx = x2 - x1;
        const dy = y2 - y1;
        const len = Math.hypot(dx, dy) || 1;
        const pad = 18;
        return (
          <line
            key={i}
            x1={x1 + (dx / len) * pad}
            y1={y1 + (dy / len) * pad}
            x2={x2 - (dx / len) * pad}
            y2={y2 - (dy / len) * pad}
            stroke={cross ? t.category.red : t.text.tertiary}
            strokeWidth={cross ? 2.4 : 1.4}
          />
        );
      })}
      {names.map((name, i) => {
        const [x, y] = pos[i];
        const onM0 = props.mesh[i] === 0;
        return (
          <g key={name}>
            <rect x={x - 16} y={y - 12} width={32} height={24} rx={5} fill={onM0 ? t.accent.primary : t.fill.secondary} stroke={t.stroke.primary} />
            <text x={x} y={y + 4} textAnchor="middle" fontSize={11} fontWeight={600} fill={onM0 ? t.text.onAccent : t.text.primary}>
              {name}
            </text>
          </g>
        );
      })}
      <text x={4} y={H - 6} fontSize={10} fontWeight={600} fill={props.ok ? t.category.green : t.category.red}>
        {props.verdict}
      </text>
    </svg>
  );
}

function DescriptorExamples() {
  return (
    <Stack gap={12}>
      <H2>Examples: Tier A</H2>
      <Table
        headers={["Descriptor", "Result", "Why"]}
        rows={[
          ["mixed4", "pass", "ids unique, edges name declared stages, out-degree 1 everywhere, one cycle (s3 > s0 is the loopback)"],
          ["pod8", "pass", "a ring of 8; loopback s7 > s0"],
          ["mixed4 with s3 > s0 not marked loopback", "reject", "the cycle s0 > s1 > s2 > s3 > s0 has no loopback edge"],
          ["mixed4 with s1 > s2 marked loopback instead", "pass", "any edge of the cycle may be the loopback"],
          ["a chain s0 > s1 marked loopback", "reject", "the loopback closes no cycle"],
          ["mixed4 + edge s1 > s1", "reject", "self-edge"],
          ["mixed4 with s0 > s1 repeated", "pass", "two channels on that hop; the builder reads one link per edge"],
          ["dflash: router > target0..37 > router, router > draft0..10 > router, both returns loopback", "pass", "ids unique, edges name declared stages, shapes positive, each cycle has its loopback"],
          ["s0 > s1, s0 > s2, s1 > s0; s2 has no out-edge", "pass", "a branch may dead-end; Tier A does not limit stage topology"],
          ["a stage with three outgoing branches", "pass", "Tier A does not limit stage topology"],
          ["a branch stage with out-degree 2", "pass", "Tier A does not limit stage topology"],
          ["two stages with in-degree 0", "reject", "exactly one stage may have no incoming edge besides loopbacks; it owns host I/O"],
          ["mixed4 with policy: RELAXED on s3 > s0", "reject", "RELAXED is rejected by Tier A in the first version. RELAXED (later): passes"],
          ["RELAXED (later): mixed4 with s0 > s1 RELAXED", "pass", "allowed; Tier C's planner still never cuts stage 0's outgoing edge"],
          ["RELAXED (later): dflash with router > draft0 RELAXED", "pass", "any edge may be RELAXED"],
          ["mixed4 plus a copy named mixed4_b", "pass here", "each is checked on its own; Tier B checks them together"],
          ["mixed4 plus a copy also named mixed4", "reject", "pipeline names must be unique"],
          ["a stage [2, 4] where every slice is 4x2", "pass here", "no transpose, so Tier B finds no mesh it binds to"],
        ]}
        rowTone={[
          "success", "success", "danger", "success", "success", "warning", "warning", "warning", "warning", "success", "warning",
          "warning", "warning", "danger", "warning",
        ]}
      />

      <H2>Examples: Tier B</H2>
      <Text tone="secondary">
        mixed4: slices are 4x2, four per mesh; 4x2 is one slice and 4x4 two, so every stage binds to either mesh; the ring needs
        6 of 8 slices. The table lists projections as the optional "find all" run sees them (each row also stands for its
        mirror, since M0 and M1 are interchangeable); the existence search stops at the first survivor and chooses nothing.
      </Text>
      <Table
        headers={["Projection", "Slices M0 / M1", "Crossings", "Result"]}
        rows={[
          ["all four stages on M0", "6 / 0", "0", "dropped: M0 holds 4 slices"],
          ["s0, s1 > M0; s2, s3 > M1", "3 / 3", "2", "survives"],
          ["s0, s1, s3 > M0; s2 > M1", "4 / 2", "2", "survives"],
          ["s0, s2 > M0; s1, s3 > M1", "3 / 3", "4", "dropped: 2 channels"],
        ]}
        rowTone={["danger", "success", "success", "danger"]}
      />
      <Text size="small" tone="tertiary">
        Filled stage is on M0, outlined stage is on M1. A red edge is a hop whose two ends sit on different meshes. graph_topology
        allows 2 such hops.
      </Text>
      <Grid columns={fit(200)} gap={12} align="start">
        <ProjectionRing title="all four on M0" mesh={[0, 0, 0, 0]} ok={false} verdict="0 crossings, 6 slices on M0" />
        <ProjectionRing title="s0 s1 | s2 s3" mesh={[0, 0, 1, 1]} ok={true} verdict="2 crossings, survives" />
        <ProjectionRing title="s0 s1 s3 | s2" mesh={[0, 0, 1, 0]} ok={true} verdict="2 crossings, survives" />
        <ProjectionRing title="s0 s2 | s1 s3" mesh={[0, 1, 0, 1]} ok={false} verdict="4 crossings, 2 channels" />
      </Grid>
      <Text tone="secondary">
        The existence search: s1 (2 slices) first, on M0 (the lowest empty mesh, so its mirror is never tried); s0 joins it
        (3 of 4); s2 needs 2 slices with 1 free, so M1 (channel 1); s3 ties between neighbours and takes M1, with more room
        (channel 2). The leaf carves both meshes and Tier B passes after a handful of nodes. In the "find all" run every
        survivor crosses, and a ring crosses a cut an even number of times, so Plan 1 gets k(M0, M1) = 2. pod8 has exactly
        one projection: 4x2 binds only to M0 (M1 has 2 rows) and 1x2 only to M1 (not whole 4x2 slices). M0's four stages fill a
        line, so its boundary slices are A0 and A3.
      </Text>
      <Table
        headers={["Change", "Result", "Rule"]}
        rows={[
          ["mixed4 + a fifth stage [4, 6]", "reject", "9 slices needed, 8 declared"],
          ["mixed4 + a fifth stage [4, 4]", "pass", "8 of 8: s0, s3 and one 4x4 on one mesh, two 4x4 on the other"],
          ["mixed4 with a stage [4, 10]", "reject", "wider than either mesh"],
          ["pod8 with M1 4x8, host [1, 4], still 1x2 stages", "reject", "1x2 is not a whole slice; stricter than blaze's per-rank tiling"],
          ["mixed4 with no MGD connection", "reject", "every projection crosses on some edge"],
          ["RELAXED (later): the same, s1 > s2 and s3 > s0 RELAXED", "pass", "s0, s1 > M0; s2, s3 > M1 crosses only on RELAXED edges, which need no connection"],
          ["RELAXED (later): the same, only s3 > s0 RELAXED", "reject", "every projection still crosses on a STRICT edge"],
          ["mixed4 plus a copy named mixed4_b", "reject", "6 + 6 = 12 slices needed, 8 declared, although each fits alone"],
          ["two copies of ring4 (four 4x2 stages each)", "reject", "8 of 8 slices fit, but A0..A3 is a line, so each ring splits and crosses twice: 4 crossings, 2 channels"],
          ["the same, channels { count: 4 }", "pass", "each ring splits 2 + 2; a projection places all eight stages at once"],
        ]}
        rowTone={["danger", "success", "danger", "warning", "danger", "success", "danger", "danger", "danger", "success"]}
      />
      <H3>How the connectivity check runs (plan)</H3>
      <Text tone="secondary">
        Tier B never looks at chips or cables. It searches two small graphs the MGD already holds: the mesh graph, with a
        channel count per mesh pair, and inside each mesh the slice grid (host_topology, with RING wraps). One search covers
        every pipeline at once, because pipelines share slices and channels.
      </Text>
      <Grid columns={fit(420)} gap={16} align="start">
        <CodeCard path="detail::check_pipeline_feasibility (pipeline_builder.cpp): the search" mode="plain" language="text" src={TIER_B_SEARCH} />
        <Stack gap={10}>
          <Table
            headers={["", "STRICT edge (first version)", "RELAXED edge (later)"]}
            rows={[
              ["crosses meshes A and B", "needs cap(A, B) >= 1, uses one channel", "no condition"],
              ["both ends on one mesh", "the two rectangles must touch", "no condition"],
              ["k_AB, boundary slices", "counted", "not counted"],
              ["every edge of a pipeline", "full search", "steps 1, 2 and no-overlap only"],
            ]}
          />
          <Text tone="secondary">
            cap(A, B) is the sum of count over FABRIC connections with nodes [A, B]. populate stores each connection in both
            directions, so only one is read. The connection's own policy does not change cap: STRICT binds exactly count and
            RELAXED binds up to count, so count is the most Tier B may assume. Binding fewer is Tier C's problem.
          </Text>
        </Stack>
      </Grid>
      <Table
        headers={["Change", "Step that decides", "Why"]}
        rows={[
          ["fifth stage [4, 6]", "2", "1 + 2 + 2 + 1 + 3 = 9 slices, 8 declared"],
          ["fifth stage [4, 4] between s3 and s0", "3, 4", "stops at s3, s4, s0 > M0 (path A0 / A1-A2 / A3) and s1, s2 > M1; 2 crossings, 2 channels; nothing stored. \"Find all\" adds the mirror and both carves on M0 (s3 or s0 on A0), so boundary slices {A0, A3}"],
          ["a stage [4, 10]", "1", "10 columns > 8: no mesh"],
          ["pod8, M1 as 4x8", "1", "[1, 2] / [4, 2] is not whole: no mesh"],
          ["no MGD connection", "3", "cap = 0, and no single mesh holds 6 slices, so some edge crosses"],
          ["RELAXED (later): s1 > s2, s3 > s0 RELAXED", "3, 4", "s0, s1 > M0; s2, s3 > M1 crosses only on RELAXED edges"],
          ["RELAXED (later): only s3 > s0 RELAXED", "3", "a ring crosses a cut at least twice; only one crossing can be RELAXED"],
          ["mixed4 + mixed4_b", "2", "6 + 6 = 12 slices, 8 declared"],
          ["two copies of ring4, 2 channels", "passes (backup step 4 would reject)", "no 4-cycle on a line of slices, so each ring crosses twice: 4 > 2; Tier C rejects it today"],
        ]}
      />
      <H3>Output: mapping constraints for Tier C (backup if the solve is slow, not implemented)</H3>
      <Text tone="secondary">
        Not in the first version: Tier B saves nothing. As a backup, it would save what it found as PipelineMappingConstraints on the descriptor: binding sets, k_AB and the minimum
        crossings per mesh, and the boundary slices per mesh pair. No projection list is stored. Every fact holds for every
        surviving projection and carve, so constraints built from it never drop a buildable seating. The MGD library stores plain MGD
        types; pipeline_builder turns them into the mapper's generic ExtraMappingConstraints hooks, which TopologyMapper passes in through TopologyMappingConfig (P10).
      </Text>
      <Table
        headers={["Helper", "Fills", "Adds"]}
        rows={[
          ["stage_to_mesh_constraints (report)", "MappingConstraints<PipelineStageNode, MeshId>", "required: stage to its binding set (not a solve)"],
          ["ExtraMappingConstraints::allow_seat", "SAT seating, in apply_extra_constraints", "forbidden: seats of m with fewer than min_crossings_out(m) cables leaving the footprint"],
          ["ExtraMappingConstraints::allow_seat_pair", "SAT seating, in build_sat_placement_seat_graph", "no edge between seats with fewer than k_AB links (Plan 1)"],
          ["ExtraMappingConstraints::intra_mesh", "intra-mesh, after add_exit_node_constraints", "cardinality: at least ceil(k(m, n) / links per ASIC) chips of boundary_slices(m, n) on ASICs cabled toward n"],
        ]}
      />
      <Text tone="secondary">
        Cost: of the repo's host_topology values, about 200 are one slice, 60 are 2 to 4, 15 are 8 to 16 and 8 are 32. A stage
        with one mesh in its binding set makes no branch, so llama_8b_pod is a single path. Interchangeable-mesh symmetry turns
        8 identical meshes on ALL_TO_ALL (8! = 40320 orderings) into one. The only slow case is a fully used 32-slice grid
        of one-slice stages that must close a ring (a Hamiltonian-cycle search); the exact segment test rejects the
        impossible ones before searching, so there is no budget. Why not SAT: a variable per (stage, mesh), pseudo-Boolean
        capacity sums, crossing counters per mesh pair and placement variables for the carve cost more to build than the
        search takes, an unsat answer gives no reason, and a plain encoding was no faster on the hard failures.
        The carve check depends only on the MGD, not on seats, so it stays here: as a SAT placement constraint it would
        prune nothing.
        Open: a connection between two sub-graphs does not name meshes, so the first version counts only mesh-to-mesh
        connections.
      </Text>
    </Stack>
  );
}

const TIER_B_SEARCH = `
built once per constructor
  slices[m]   = prod(host_dims), grid dims / host_dims
  adj(m)      = host_dims grid; wrap on axis i if ring_dims[i] and host_dims[i] > 2
  cap[m][n]   = sum of count, FABRIC connections [m, n]
  need[s][m]  = slices of shape(s) / slice(m), or none (no transpose)
  nbrs[s]     = (neighbour stage, edge copies), every pipeline
  classes     = interchangeable meshes: same topology, same cap row

1 binding    every stage has some m with need[s][m]       else reject
2 capacity   sum of min need over all pipelines
             <= sum of slices[m]                          else reject
3 search     DFS stage -> mesh, stop at the FIRST survivor
             order: most constrained stage, then BFS along edges
             values: meshes holding a neighbour, then most free
             allow s -> m when
               used[m] + need[s][m] <= slices[m]
               each edge copy s - t, t on n != m:
                 crossings[m][n] + 1 <= cap[m][n]
               an empty m is the lowest empty of its class
             look-ahead: every open stage still has a mesh;
               open needs fit in the free slices
             (RELAXED later: skip RELAXED edges in cap checks)
             leaf -> step 4 on each mesh; fail -> backtrack
4 carve      per mesh, DFS: uint64 masks over the slice grid
             largest stage first, then BFS along mesh edges
             a position: no overlap, touches every placed nbr
             (grown(a) & b != 0, wraps included), no wrap
             memo (mesh descriptor, sorted stage ids)
5 result     first full success -> pass, nothing stored
             none -> reject with the deepest failure
             no budget: tight meshes take the O(1) segment
             test first, so failures are proved fast

optional     "find all": same searches, keep going after
             each survivor and enumerate every carve;
             k_AB = min crossings, boundary = union of slices
             holding a crossing stage. budget 10^5, else
             defaults k = 0, every slice a boundary`;


// --- Mapper tab: Tier C, carve planner, resolver, carve paths, loop, Tier D ---

const CARVE_A = `
phase 0   slices all 4x2; stage_counts = [1, 2, 2, 1]; need = 6 <= 8; stage 0 is 1 slice
phase 2   A3 has one live neighbour (A2), B3 has one (B2): drop both
          alive = {A0 A1 A2 B0 B1 B2}; all have 2 neighbours -> no branch slices
phase 3   _cycles_no_branch walks from the lowest slice A0:
          A0 A1 A2 B2 B1 B0              length 6 == need
phase 4   canonical walk: anchor A0; A1 (0,0,2) < B0 (1,0,0), so step to A1
          forward, offset 0:
            s0 <- [A0]      M0 rows 0-3 cols 0-1   4x2   ok
            s1 <- [A1 A2]   M0 rows 0-3 cols 2-5   4x4   ok
            s2 <- [B2 B1]   M1 rows 0-3 cols 2-5   4x4   ok
            s3 <- [B0]      M1 rows 0-3 cols 0-1   4x2   ok
phase 5   slots M0 4x2@(0,0)  M0 4x4@(0,2)  M1 4x4@(0,2)  M1 4x2@(0,0)
          fillers M0 4x2@(0,6)  M1 4x2@(0,6)          (ranks 3 and 7 idle)`;

const CARVE_B = `
phase 2   every slice has 2 neighbours: nothing pruned, no branch slices
phase 3   one loop: A0 A1 A2 A3 B3 B2 B1 B0, length 8
error     no-cycle-of-length-6: this fabric admits no simple cycle of
          exactly 6 host slices, which is what a 4-stage loopback ring needs.
            achievable cycle lengths: [8]
            nearest: none below, 8 above`;

const CARVE_C = `
phase 2   nothing pruned. Branch slices: A2 (A1, A3, B2) and B2 (B1, B3, A2)
          _contract walks out of A2 in neighbour order, three chains A2 -> B2:
            c0  interior A1 A0 B0 B1   weight 4
            c1  interior A3 B3         weight 2
            c2  direct                 weight 0
phase 3   a cycle = two different chains between A2 and B2 = 2 + w + w slices
            c0 + c2 = 6     c1 + c2 = 4     c0 + c1 = 8
          need 6 -> c0 + c2 = A2 A1 A0 B0 B1 B2
          (found 4 times: 2 start slices x 2 directions; one ring)
          c0 + c1 passes 6 and is pruned before it closes
phase 4   same canonical walk and plan as example A
          a 4x2 ring of 4 stages (need 4) would take c1 + c2 = A2 A3 B3 B2

quirk     need 5 here fails with  achievable cycle lengths: [4]
          lengths above need are never recorded when branch slices exist`;

const CARVE_D = `
ring      s0 4x2, s1 4x2, s2 4x8     stage_counts [1, 1, 4], need 6
phase 3   cycle A0 A1 A2 B2 B1 B0, length 6
phase 4   s2 needs 4 slices in one row of one mesh; the cycle's same-mesh
          runs are [3, 3], so every rotation puts s2 across a crossing
error     no-alignment: found 1 cycle(s) of the right length (6 slices) but
          no rotation places every stage as its declared rectangle ...
            cycle's same-mesh run lengths: [3, 3]
            stage shapes: ['4x2', '4x2', '4x8']`;

const CARVE_F = `
galaxy    one 4x8 mesh, host_topology [1, 8], both axes RING
          ranks own 4x1 columns C0..C7; slice graph is one 8-cycle
ring      s0 4x1, s1 4x2, s2 4x2, s3 4x2, s4 4x1, loopback s4 -> s0
path      one mesh, but s1..s3 are joint: build_pipeline sets multi_mesh
          and passes no submesh_shape -> planned carve, not tiling
phase 0   slices all 4x1; stage_counts [1, 2, 2, 2, 1]; need 8 <= 8
phase 2   every column has 2 neighbours: nothing pruned, no branch slices
phase 3   _cycles_no_branch: C0 C1 ... C7, length 8 == need
phase 4   canonical walk: anchor C0; C1 (0,0,1) < C7 (0,0,7) -> toward C1
          forward, offset 0:
            s0 <- [C0]      col 0      4x1   ok
            s1 <- [C1 C2]   cols 1-2   4x2   ok
            s2 <- [C3 C4]   cols 3-4   4x2   ok
            s3 <- [C5 C6]   cols 5-6   4x2   ok
            s4 <- [C7]      col 7      4x1   ok
phase 5   4x1@(0,0) 4x2@(0,1) 4x2@(0,3) 4x2@(0,5) 4x1@(0,7); no fillers
collect   ranks 1+2, 3+4, 5+6 each collapse into one co-owned submesh`;

const COLLECT = `
M0 slots in plan order: 4x2@(0,0), 4x4@(0,2), 4x2@(0,6)
  rank 0 -> 4x2@(0,0)   rank 1 -> 4x4@(0,2)   rank 2 -> 4x4@(0,2)   rank 3 -> filler
  (a rank in zero or two slots is a planner bug and raises)
collapse identical chip sets, first-rank order:
  submesh 0 = A0 (8)   1 = A1+A2 (16, ranks 1, 2)   2 = A3 (8)
  submesh 3 = B0 (8)   4 = B1+B2 (16, ranks 5, 6)   5 = B3 (8)
6 distinct submeshes >= 4 stages`;

const RESOLVE_MIXED4 = `
topological order, loopback excluded: s0 s1 s2 s3
chip counts: s0 8, s1 16, s2 16, s3 8
submesh links: 0-1 1-2 3-4 4-5 (in mesh), 0-3 and 1-4 (cables)

s0: 2 links + 2 host endpoints = 4 <= capacity 1 x 8; needs 2 neighbours
    submesh 0 (nbrs 1, 3) candidate     submesh 2 (nbr 1)   skip
    submesh 3 (nbrs 0, 4) candidate     submesh 5 (nbr 4)   skip
    try submesh 0
s1: 16 chips, linked to 0 -> submesh 1
s2: 16 chips, linked to 1 -> submesh 4 (A2-B2 cable)
s3: 8 chips, from 4 and to 0 -> submesh 3 (A0-B0 cable closes the ring)
LinkSearch: one chip pair per edge, exclusive chips first;
            host endpoints on s0's unused chips
result: s0 -> 0  s1 -> 1  s2 -> 4  s3 -> 3
(if submesh 0 failed: submesh 3 next, the mirror s0 B0 ... s3 A0)`;

const RESOLVE_POD8 = `
planned carve   rejected up front: host slices are not uniform (4x2 vs 1x2)
per-rank tiling 4x2 on M0, 1x2 on M1; each tile is one rank's slice
submeshes       A0..A3 (8 chips), T0..T3 (2 chips); nothing collapses
chip counts     s0..s3 only on A, s4..s7 only on T
capacity        1 slot/chip on >= 8 chips, 2 on smaller: T has 4 slots
M0              s0..s3 run A0..A3 or A3..A0
M1              a path through all of a 4-cycle ends NEXT TO its start

seat U   A3 -> (0,1) in T0, A0 -> (1,3) in T3     opposite corners
         s4 at T0 -> s7 at T1 or T2, but the cable back is on T3
         s4 at T3 -> s7 at T1 or T2, but the cable back is on T0
         no valid submesh assignment found

seat V   A3 -> (0,3) in T1, A0 -> (0,0) in T0     neighbours
         s0 A0  s1 A1  s2 A2  s3 A3  s4 T1  s5 T3  s6 T2  s7 T0
         mirror: s0 A3 .. s3 A0, s4 T0  s5 T2  s6 T3  s7 T1`;

const HOSTCUT_TRACE = `
phase 1   no 6-cycle (only the 8-cycle)
          census: spare = 2 and 6 + 2 is achievable -> passthrough_ring = 2
phase 3   one fabric component of 8 >= 6, so k starts at 1
          _cover_with_chains(1): smallest slice first, smallest free neighbour next
            1. A0 A1 A2 A3 B3 B2     2. A0 B0 B1 B2 B3 A3     3. A1 A0 B0 B1 B2 B3 ...
phase 4   _group_walk per cover; cuts only between stages, only on RELAXED edges
          cover 1 best: s0 A3, s1 B3+B2, s2 A0+A1, s3 A2   cut s1->s2  skipped: STRICT
          cover 2 best: s0 A3, s1 B3+B2, s2 B1+B0, s3 A0   cut s3->s0  key (1,0,(3,))
          key = (cut page sizes, 0 if the loopback is cut else 1, cuts)
result    cover 2: its one cut is the RELAXED loopback; s3 -> s0 goes over host; carve
          M0 4x2@(0,6), M1 4x4@(0,4), M1 4x4@(0,0), M0 4x2@(0,0); A1, A2 idle
          _resolve_chains_separately: one fabric chain s0 -> s1 -> s2 -> s3`;

const TIER_D = `
mixed4 on (P, R), runtime
  control plane            binds the two cables Tier C relied on (4.2)
  build_topology           no submesh_shape -> plan_carve
  discover_host_slices     allgather: A0..A3, B0..B3, 4x2 each
  discover_slice_adjacency get_forwarding_direction-confirmed: lines + A0-B0, A2-B2
  plan_ring_carve          the same 6-cycle and plan as Tier C (example A)
  crc32 exchange           every rank derived the same plan
  create_submesh           one call per slot on each rank's own mesh
  collapse                 ranks 1+2 and 5+6 -> joint submeshes
  resolve_graph_layout     s0 A0, s1 A1+A2, s2 B1+B2, s3 B0

the same machine today, with a second A0-B0 cable on lower chip ids
  binding takes both A0-B0 cables; blaze sees only A0-B0
  _prune_leaves: A3, B3, then A2, B2, then A1, B1, then A0, B0
  CarvePlanError: no-cycle-of-length-6: only 0 slices can lie on any cycle
  the mapping is committed, so this is a hard failure`;

function MapperExamples() {
  return (
    <Stack gap={14}>
      <H2>Examples: what Tier C decides</H2>
      <Table
        headers={["Pipeline", "Seating (M0, M1)", "Cables land on", "Tier C", "Why"]}
        rows={[
          ["mixed4", "(P, Q)", "A0-B0, A3-B3", "reject", "one loop of 8 slices, the ring needs exactly 6: the footprint solve finds no carve"],
          ["mixed4", "(P, R)", "A0-B0, A2-B2", "accept", "a 6-slice loop exists and the stages align onto it"],
          ["RELAXED (later): mixed4 with s3 > s0 RELAXED", "(P, Q)", "A0-B0, A3-B3", "accept", "one fabric path covers 6 slices; the RELAXED loopback goes over host"],
          ["pod8", "(P, U)", "A3-T0, A0-T3", "reject", "opposite corners of M1: no path through M1's tiles joins them"],
          ["pod8", "(P, V)", "A3-T1, A0-T0", "accept", "neighbouring tiles"],
        ]}
        rowTone={["danger", "success", "success", "danger", "success"]}
      />

      <H2>Blaze&apos;s carve planner (runtime reference, not ported)</H2>
      <Callout tone="info" title="Tier C does not port this">
        The footprint solve decides the carve, stage order and crossings in one solve (C2). Planner + resolver placed 0 of
        132 carve-requiring benchmark cases; footprint + resolver decided 127. The planner is kept here because blaze still
        runs it at runtime, and Tier C should agree with it on whether a carve exists (open question 8).
      </Callout>
      <Text tone="secondary">
        plan_ring_carve works on whole host slices, because _collect_submeshes needs every rank's chips in exactly one slot.
        Every rank runs it alone and must reach the same plan, so slices are ordered by geometry, never by rank.
      </Text>
      <Table
        headers={["Phase", "Code", "What it does"]}
        rows={[
          ["1. discover (not ported)", "discover_host_slices, discover_slice_adjacency", "Each rank's chips become a HostSlice. A neighbour on another slice counts only if get_forwarding_direction confirms it; one link per slice pair is enough. Tier C uses get_coord_range and the candidate links."],
          ["0. input checks", "top of plan_ring_carve", "Uniform slices; each stage a whole number of slices; need <= slices; stage 0 one slice."],
          ["2. reduce", "_prune_leaves, _contract", "Drop slices with fewer than 2 live neighbours. Slices with more than 2 are branch slices; runs between them become weighted chains."],
          ["3. cycle search", "_cycles_no_branch, _cycles_of_length", "Simple cycles of exactly need slices; DFS over branch slices, one chain per step, pruned above need. 200,000 expansions, 64 cycles."],
          ["4. align", "_group_cycle, _canonical_walk, _as_rectangle", "Fewest crossings first; canonical walk, every rotation, reversed. Each stage's group must be one gap-free rectangle on one mesh with the declared shape."],
          ["5. emit", "validate_carve_plan", "One CarveSlot per stage in order, a filler per unused slice; every chip covered once."],
        ]}
      />

      <H3>A. One loop, no branch slices (mixed4 on (P, R))</H3>
      <Grid columns={fit(380)} gap={20} align="start">
        <SliceMap
          stages={[
            { mesh: 0, from: 0, to: 0, label: "s0" },
            { mesh: 0, from: 1, to: 2, label: "s1 (4x4)" },
            { mesh: 1, from: 1, to: 2, label: "s2 (4x4)" },
            { mesh: 1, from: 0, to: 0, label: "s3" },
            { mesh: 0, from: 3, to: 3, label: "pruned", idle: true },
            { mesh: 1, from: 3, to: 3, label: "pruned", idle: true },
          ]}
          cables={[
            { a: 0, b: 0, kind: "required" },
            { a: 2, b: 2, kind: "required" },
          ]}
          verdict={{ ok: true, text: "plan found" }}
        />
        <CodeCard path="plan_ring_carve, cables A0-B0 and A2-B2" mode="plain" language="text" src={CARVE_A} />
      </Grid>
      <Text tone="secondary">
        Twelve alignments exist (six rotations, two directions). Four work, each putting the 4x4 stages on a different pair of
        ranks; the rest fail in _as_rectangle because a 4x4 group would take one slice from each mesh.
      </Text>
      <Grid columns={fit(300)} gap={14} align="start">
        {[
          {
            name: "forward, offset 0 (returned)",
            stages: [
              { mesh: 0, from: 0, to: 0, label: "s0" },
              { mesh: 0, from: 1, to: 2, label: "s1" },
              { mesh: 1, from: 1, to: 2, label: "s2" },
              { mesh: 1, from: 0, to: 0, label: "s3" },
            ],
          },
          {
            name: "forward, offset 3",
            stages: [
              { mesh: 1, from: 2, to: 2, label: "s0" },
              { mesh: 1, from: 0, to: 1, label: "s1" },
              { mesh: 0, from: 0, to: 1, label: "s2" },
              { mesh: 0, from: 2, to: 2, label: "s3" },
            ],
          },
          {
            name: "reverse, offset 1 (the resolver's mirror)",
            stages: [
              { mesh: 1, from: 0, to: 0, label: "s0" },
              { mesh: 1, from: 1, to: 2, label: "s1" },
              { mesh: 0, from: 1, to: 2, label: "s2" },
              { mesh: 0, from: 0, to: 0, label: "s3" },
            ],
          },
          {
            name: "reverse, offset 4",
            stages: [
              { mesh: 0, from: 2, to: 2, label: "s0" },
              { mesh: 0, from: 0, to: 1, label: "s1" },
              { mesh: 1, from: 0, to: 1, label: "s2" },
              { mesh: 1, from: 2, to: 2, label: "s3" },
            ],
          },
        ].map((m) => (
          <div key={m.name}>
            <Stack gap={4}>
              <Text weight="semibold" size="small">
                {m.name}
              </Text>
              <SliceMap stages={m.stages as SliceStage[]} cables={[{ a: 0, b: 0 }, { a: 2, b: 2 }]} />
            </Stack>
          </div>
        ))}
      </Grid>

      <H3>B. Wrong loop length (mixed4 on (P, Q))</H3>
      <Grid columns={fit(380)} gap={20} align="start">
        <SliceMap stages={[]} cables={[{ a: 0, b: 0 }, { a: 3, b: 3 }]} verdict={{ ok: false, text: "STRICT: rejected" }} />
        <CodeCard path="plan_ring_carve, cables A0-B0 and A3-B3" mode="plain" language="text" src={CARVE_B} />
      </Grid>

      <H3>C. Branch slices and contraction (add cable A3-B3)</H3>
      <Grid columns={fit(380)} gap={20} align="start">
        <SliceMap
          stages={[
            { mesh: 0, from: 0, to: 0, label: "s0" },
            { mesh: 0, from: 1, to: 2, label: "s1 (4x4)" },
            { mesh: 1, from: 1, to: 2, label: "s2 (4x4)" },
            { mesh: 1, from: 0, to: 0, label: "s3" },
            { mesh: 0, from: 3, to: 3, label: "c1", idle: true },
            { mesh: 1, from: 3, to: 3, label: "c1", idle: true },
          ]}
          cables={[
            { a: 0, b: 0, kind: "required" },
            { a: 2, b: 2, kind: "required" },
            { a: 3, b: 3, kind: "unused" },
          ]}
        />
        <CodeCard path="plan_ring_carve, three cables" mode="plain" language="text" src={CARVE_C} />
      </Grid>

      <H3>D. Right length, no alignment</H3>
      <Grid columns={fit(380)} gap={20} align="start">
        <SliceMap
          stages={[
            { mesh: 0, from: 0, to: 2, label: "same-mesh run of 3", idle: true },
            { mesh: 1, from: 0, to: 2, label: "same-mesh run of 3", idle: true },
          ]}
          cables={[
            { a: 0, b: 0 },
            { a: 2, b: 2 },
          ]}
          verdict={{ ok: false, text: "no-alignment" }}
        />
        <CodeCard path="plan_ring_carve, a 4x8 stage" mode="plain" language="text" src={CARVE_D} />
      </Grid>

      <H3>F. Stages wider than a rank, on one mesh</H3>
      <Text tone="secondary">
        A separate setup, galaxy4x1, as in blaze's single_bh_galaxy_4x1_slices MGD. Each rank owns a full-height 4x1 column,
        and each 4x2 stage is two ranks. The stages start on odd columns because s0 takes column 0 by itself.
      </Text>
      <Grid columns={fit(420)} gap={20} align="start">
        <Stack gap={12}>
          <ColumnRing
            caption="forward, start C0 (returned)"
            verdict={{ ok: true, text: "ok" }}
            stages={[
              { from: 0, to: 0, label: "s0" },
              { from: 1, to: 2, label: "s1" },
              { from: 3, to: 4, label: "s2" },
              { from: 5, to: 6, label: "s3" },
              { from: 7, to: 7, label: "s4" },
            ]}
          />
          <ColumnRing
            caption="forward, start C1"
            verdict={{ ok: true, text: "ok" }}
            stages={[
              { from: 0, to: 0, label: "s4" },
              { from: 1, to: 1, label: "s0" },
              { from: 2, to: 3, label: "s1" },
              { from: 4, to: 5, label: "s2" },
              { from: 6, to: 7, label: "s3" },
            ]}
          />
          <ColumnRing
            caption="forward, start C2"
            verdict={{ ok: false, text: "s3 = {C7, C0}: not a rectangle" }}
            stages={[
              { from: 0, to: 0, label: "s3?", bad: true },
              { from: 2, to: 2, label: "s0" },
              { from: 3, to: 4, label: "s1" },
              { from: 5, to: 6, label: "s2" },
              { from: 7, to: 7, label: "s3?", bad: true },
            ]}
          />
        </Stack>
        <CodeCard path="plan_ring_carve, 4x2 stages on 4x1 ranks" mode="plain" language="text" src={CARVE_F} />
      </Grid>
      <Table
        headers={["Alignment", "s0", "s1", "s2", "s3", "s4", "Result"]}
        rows={[
          ["forward, start C0 (returned)", "C0", "C1+C2", "C3+C4", "C5+C6", "C7", "ok; the loopback rides the wrap"],
          ["forward, start C1", "C1", "C2+C3", "C4+C5", "C6+C7", "C0", "ok; s3 to s4 rides the wrap"],
          ["forward, start C2", "C2", "C3+C4", "C5+C6", "C7+C0", "", "fails: s3 is not a rectangle"],
          ["reverse, start C0", "C0", "C7+C6", "C5+C4", "C3+C2", "C1", "ok; s0 to s1 rides the wrap"],
          ["reverse, start C1", "C1", "C0+C7", "", "", "", "fails: s1 is not a rectangle"],
        ]}
        rowTone={["success", "success", "danger", "success", "danger"]}
      />
      <Text tone="secondary">
        Sixteen alignments, ten work. The six that fail all put a 4x2 stage on C7 and C0: fabric neighbours through the wrap,
        but not one rectangle. With one mesh there are no crossings, so Tier B's per-mesh carve check already runs the whole
        ring and gives the verdict; Tier C agrees.
      </Text>
      <Table
        headers={["Case on this galaxy", "Phase", "Result"]}
        rows={[
          ["ring of 4x1, 4x2, 4x2, 4x1 (need 6)", "3", "no-cycle-of-length-6, achievable [8]; blaze pads with 4x1 passthroughs"],
          ["s0 declared 4x2", "0", "input-infeasible: one rank must drive stage 0"],
          ["a stage declared 2x2", "0", "not a whole number of 4x1 host slices"],
          ["a 4x2 stage on C7 and C0", "4", "_as_rectangle rejects it"],
          ["column axis LINE, not RING", "2", "leaf pruning removes every column"],
          ["a ring needing 9 slices", "0", "input-infeasible: only 8 slices"],
        ]}
      />
      <Table
        headers={["Other planner edge case", "Phase", "Result"]}
        rows={[
          ["pod8 (4x2 and 1x2 slices)", "0", "host slices are not uniform; per-rank tiling is the path"],
          ["a 1x2 stage on 4x2 slices", "0", "not a whole number of 4x2 host slices"],
          ["mixed4 with s0 4x4", "0", "input-infeasible: one rank must drive stage 0"],
          ["only one cable, A0-B0", "2", "leaf pruning removes every slice"],
          ["an open chain, no loopback", "before 0", "plan_carve refuses; give it a submesh_shape"],
          ["a huge fabric", "3", "budget-exhausted (every 8x8 or larger grid in the benchmark); the footprint decides these in seconds"],
        ]}
      />

      <H2>Examples: blaze&apos;s runtime carve paths</H2>
      <Text tone="secondary">
        What blaze does at runtime on the running examples. Tier C does not try these paths; its one position graph covers
        all of them (except a stage smaller than a host slice, open question 15).
      </Text>
      <Grid columns={fit(420)} gap={20} align="start">
        <CodeCard path="slots to submeshes (planned carve, mixed4 on (P, R))" mode="plain" language="text" src={COLLECT} />
        <Stack gap={8}>
          <Text tone="secondary">Per-rank tiling on mixed4 (P, Q), after the planned carve failed:</Text>
          <Table
            headers={["M0 tiles", "M1 tiles", "Result"]}
            rows={[
              ["4x2", "4x2", "eight 8-chip submeshes: nothing with 16 chips for s1"],
              ["4x4", "4x4", "four 16-chip submeshes: nothing with 8 chips for s0"],
              ["4x2", "4x4", "s0, s3 need neighbouring A slices that both have cables; the cables are on A0 and A3"],
              ["4x4", "4x2", "the mirror, same failure"],
            ]}
            rowTone={["danger", "danger", "danger", "danger"]}
          />
        </Stack>
      </Grid>
      <H3>RELAXED (later): host cuts on RELAXED edges, mixed4 with s3 &gt; s0 RELAXED, on (P, Q)</H3>
      <Grid columns={fit(380)} gap={20} align="start">
        <SliceMap
          stages={[
            { mesh: 0, from: 3, to: 3, label: "s0" },
            { mesh: 1, from: 2, to: 3, label: "s1 (4x4)" },
            { mesh: 1, from: 0, to: 1, label: "s2 (4x4)" },
            { mesh: 0, from: 0, to: 0, label: "s3" },
            { mesh: 0, from: 1, to: 2, label: "idle", idle: true },
          ]}
          cables={[
            { a: 0, b: 0, kind: "required" },
            { a: 3, b: 3, kind: "required" },
          ]}
          host={{ from: [0, 0], to: [0, 3], label: "s3 > s0 over host" }}
          verdict={{ ok: true, text: "s3 > s0 RELAXED: accepted" }}
        />
        <CodeCard path="plan_ring_placement" mode="plain" language="text" src={HOSTCUT_TRACE} />
      </Grid>
      <Table
        headers={["Host-cut edge case", "Result"]}
        rows={[
          ["a cover whose only alignment cuts a STRICT edge", "skipped"],
          ["a cover whose only alignment cuts s0 > s1", "skipped, even if s0 > s1 is declared RELAXED: stage 0's outgoing edge stays on fabric"],
          ["a joint between chains inside a 4x4 stage", "skipped: a wide stage cannot be split by a host hop"],
          ["pod8 with a RELAXED edge", "HostEdgePlanError (slices not uniform), not caught: reject"],
          ["all fabric already, single-slice stages", "no plan; the resolver places it as for an all-STRICT pipeline"],
          ["every edge RELAXED", "cover 1's cut is allowed too; cover 2 still wins, because it cuts the loopback"],
        ]}
      />
      <H3>Several pipelines: two copies of ring4 on the mixed4 meshes</H3>
      <Text tone="secondary">
        Here the MGD declares channels count 4 between M0 and M1; with mixed4's 2, Tier B already rejects it. The pipelines form one graph whose components are the pipelines. Per seating, place each on the slices the earlier ones
        left free, trying every order of the pipelines; unused slices become filler once, at the end. Pipelines share no edge,
        so each is resolved as its own graph. Accept only if every pipeline is placed and resolves.
      </Text>
      <Table
        headers={["Cables", "Placement", "Tier C"]}
        rows={[
          [
            "A0-B0, A2-B2, A3-B3",
            "ring4_a takes the only 4-cycle, A2 A3 B3 B2; A0, A1, B0, B1 form a path with no cycle, so ring4_b fails; the other order fails the same way",
            "reject, although each fits alone",
          ],
          ["add A1-B1", "ring4_a takes A0 A1 B1 B0, ring4_b takes A2 A3 B3 B2", "accept, every slice used"],
        ]}
        rowTone={["danger", "success"]}
      />
      <H3>Tile smaller than a host slice</H3>
      <Text tone="secondary">
        pod8 with M1 as 4x8, host [1, 4] and 1x2 stages: create_submeshes(1x2) makes 16 tiles; each M1 rank keeps the one whose
        (0,0) chip is local, all in row 0. A cable landing in row 3 does not count, so the same seat passes or fails depending
        on the intra-mesh layout. Tier B as written rejects this MGD.
      </Text>

      <H2>Examples: the resolver</H2>
      <Table
        headers={["Check", "Rule"]}
        rows={[
          ["chip count", "the submesh has exactly node_chip_counts[stage] chips"],
          ["core capacity", "links (one per edge, +2 for stage 0's host-to-device and device-to-host) fit capacity x chips; 1 per chip on >= 8 chips, 2 below"],
          ["neighbour count", "at least as many neighbouring submeshes as the stage has logical neighbours"],
          ["connectivity", "linked to every placed neighbour's submesh"],
          ["lookahead", "remaining_stages_can_connect"],
        ]}
      />
      <Grid columns={fit(420)} gap={20} align="start">
        <CodeCard path="mixed4 on (P, R)" mode="plain" language="text" src={RESOLVE_MIXED4} />
        <CodeCard path="pod8 on seats U and V (per-rank tiling)" mode="plain" language="text" src={RESOLVE_POD8} />
      </Grid>
      <Grid columns={fit(380)} gap={20} align="start">
        <Stack gap={6}>
          <Text weight="semibold">pod8 on U: cables on opposite corners of M1</Text>
          <SmallMeshMap
            m0={["s0", "s1", "s2", "s3"]}
            m1={["", "", "", ""]}
            cables={[
              { a: 3, row: 0, col: 1 },
              { a: 0, row: 1, col: 3 },
            ]}
            verdict={{ ok: false, text: "rejected" }}
          />
        </Stack>
        <Stack gap={6}>
          <Text weight="semibold">pod8 on V: cables on neighbouring tiles</Text>
          <SmallMeshMap
            m0={["s0", "s1", "s2", "s3"]}
            m1={["s7", "s4", "s6", "s5"]}
            cables={[
              { a: 3, row: 0, col: 3 },
              { a: 0, row: 0, col: 0 },
            ]}
            verdict={{ ok: true, text: "accepted" }}
          />
        </Stack>
      </Grid>
      <Table
        headers={["Resolver edge case", "What happens"]}
        rows={[
          ["mixed4 tiled 4x2 on both meshes", "s1 finds nothing with 16 chips"],
          ["mixed4 tiled 4x4 on both meshes", "s0 finds nothing with 8 chips"],
          ["a ring stage on A3 or B3 in mixed4 on (P, R)", "one neighbouring submesh: skipped before any link search"],
          ["harness core-pool overrides", "not in the MGD; Tier C can accept what runtime rejects"],
        ]}
      />

      <H2>Example: mixed4 through the mapper loop</H2>
      <Table
        headers={["Iteration", "Seating", "Intra-mesh", "Tier C", "Next"]}
        rows={[
          ["1", "(P, Q)", "cables on A0-B0, A3-B3", "no-cycle-of-length-6 (B); every tiling fails", "exclude_mapping({M0>P, M1>Q})"],
          ["2", "(Q, P)", "same cables, meshes swapped", "same failures", "exclude_mapping({M0>Q, M1>P})"],
          ["3", "(P, R)", "cables on A0-B0, A2-B2", "plan (A), resolver, R fits demand 2", "emit"],
        ]}
        rowTone={["danger", "danger", "success"]}
      />
      <Table
        headers={["Intra-mesh constraint", "Function", "On (P, R)"]}
        rows={[
          ["rank binding (hard)", "add_rank_binding_constraints", "M0 host rank i's fabric nodes on P.hi's ASICs; M1's on R.hi's"],
          ["exit nodes (hard)", "add_exit_node_constraints", "mesh-level: at least 2 M0 nodes on ASICs cabled toward R (capped by the physical total under RELAXED); P has two, on h0 and h2"],
          ["MGD pins (hard)", "add_pinning_constraints", "none declared"],
          ["PGD layout (soft)", "add_pgd_pinning_preferred_constraints", "puts host rank i on host hi"],
        ]}
      />
      <Table
        headers={["Rank", "Slice", "TT_PIPELINE_STAGE (written, not consumed)"]}
        rows={[
          ["0", "A0", "s0"],
          ["1, 2", "A1, A2", "s1 (co-owned)"],
          ["3", "A3", "idle"],
          ["4", "B0", "s3"],
          ["5, 6", "B1, B2", "s2 (co-owned)"],
          ["7", "B3", "idle"],
        ]}
      />
      <Text size="small" tone="tertiary">
        If blaze's runtime carve picked another alignment of the same cycle (for example s0 on B0), every entry would differ
        while the mapping stays the same: the entry is a witness.
      </Text>

      <H2>Example: Tier D at runtime</H2>
      <CodeCard path="blaze build_topology (unchanged)" mode="plain" language="text" src={TIER_D} />
    </Stack>
  );
}

// --- SAT tab -----------------------------------------------------------------

function SatExamples() {
  return (
    <Stack gap={14}>
      <H2>Examples: the master solve on mixed4</H2>
      <Grid columns={fit(380)} gap={20} align="start">
        <SeatGraph />
        <Table
          headers={["Step", "Code", "Result"]}
          rows={[
            ["seat pools", "create_sat_placement_pools", "both meshes: {P, Q, R}; orientations merged by sat_footprint_canonical"],
            ["seat graph", "build_sat_placement_seat_graph", "saturated_link_count parallel edges: P-Q x2, P-R x2, Q-R x1"],
            ["constraints", "build_sat_placement_constraints", "M0, M1 share a pool but are adjacent: no symmetry window; footprints disjoint"],
            ["later: Plan 1 (P10)", "seat-graph edge drop", "k = 2: Q-R (1 link) dropped. First version: Q-R kept, and the exact check rejects (Q, R) and (R, Q) one seating each"],
            ["new: unique_shapes", "MultiMeshSolutionEnumerator callers", "false: (P, Q) and (Q, P) are separate"],
            ["master session", "MasterSolve::restart", "(P, Q), (Q, P), (P, R), (R, P)"],
          ]}
        />
      </Grid>
      <Table
        headers={["Plan 1 case (later, P10)", "Result", "Why"]}
        rows={[
          ["mixed4, seat pair Q-R", "dropped", "1 link < k = 2"],
          ["mixed4, seat pair P-Q", "kept, fails later", "2 links on distinct slice pairs; it fails on loop length, which no link count sees"],
          ["pod8, seat pairs P-U and P-V", "both kept", "2 links each; with hard pins both touch {A0, A3}; U fails on M1's corners, which is about the path, not the boundary"],
          ["using one projection's boundary slices", "not allowed", "mixed4's survivors disagree (s0, s1 vs s1, s3 on M0); only the union is sound"],
          ["RELAXED (later): mixed4 with s3 > s0 RELAXED, seat pair Q-R", "kept", "k = 1 (only s1 > s2 is a STRICT crossing); Q-R works with the loopback over host"],
          ["RELAXED (later): mixed4 with every edge RELAXED", "nothing pruned", "k = 0"],
        ]}
        rowTone={["success", "warning", "warning", "danger", "success", undefined]}
      />
      <H2>Examples: feedback</H2>
      <Table
        headers={["Feedback after mixed4 fails on (P, Q)", "Effect", "Verdict"]}
        rows={[
          ["exclude_mapping({M0>P, M1>Q})", "blocks this seat set", "correct: (P, R) is still offered and succeeds"],
          ["add_forbidden_constraint(M0 on P)", "M0 never on P", "wrong: removes (P, R), one of the two seatings that work"],
          ["add_forbidden_constraint(M1 on Q)", "M1 never on Q", "wrong in general; harmless here"],
          ["block 'P and Q both used' (new API)", "blocks (P, Q) and (Q, P)", "only when proven for every carve over the pair"],
        ]}
        rowTone={["success", "danger", "warning", undefined]}
      />
      <Text tone="secondary">
        unique_shapes: with it on, (P, Q) and (Q, P) share the key {"{P, Q}"} and only the first is checked. Harmless when both
        lay out the same. If M1 had an MGD pin putting its host rank 0 on h1, the swap would lay out differently and could
        succeed, but exclude_mapping on {"{P, Q}"} would also hide it.
      </Text>
    </Stack>
  );
}

// --- Binding tab -------------------------------------------------------------

function BindingExamples() {
  return (
    <Stack gap={12}>
      <H2>Example: mixed4 on (P, R) with an extra cable</H2>
      <Text tone="secondary">Two A0-B0 cables, both on lower chip ids than the A2-B2 cable; the MGD asks for 2 channels.</Text>
      <Grid columns={fit(380)} gap={20} align="start">
        <SliceMap
          stages={[
            { mesh: 0, from: 0, to: 0, label: "s0" },
            { mesh: 0, from: 1, to: 2, label: "s1" },
            { mesh: 1, from: 1, to: 2, label: "s2" },
            { mesh: 1, from: 0, to: 0, label: "s3" },
          ]}
          cables={[
            { a: 0, b: 0, kind: "required" },
            { a: 0, b: 0, kind: "unused", label: "2nd" },
            { a: 2, b: 2, kind: "required" },
          ]}
        />
        <Table
          headers={["Step", "Today", "With R first"]}
          rows={[
            ["candidates", "A0-B0, A0-B0, A2-B2", "same"],
            ["required links", "none", "A2-B2 (s1 > s2), A0-B0 (s3 > s0)"],
            ["bind", "two lowest ids: both A0-B0", "R, filling the count of 2"],
            ["budget", "", "2 links in R, demand 2: fits"],
            ["runtime", "no-cycle-of-length-6", "builds"],
          ]}
        />
      </Grid>
      <Table
        headers={["Binding edge case", "What happens"]}
        rows={[
          ["count 3", "R takes two, the fill takes one spare in today's order"],
          ["count 1", "R needs 2 links: over budget, the seating is rejected (Tier B already rejects this MGD)"],
          ["phase 0 cannot place a cable in R", "first version: the check after the broadcast fails on every rank, naming the channel; with part 11, the mapper rejects the seating"],
          ["STRICT device-level connections with pins on A0 and A3", "the candidates are the pinned cables; mixed4 rejects (example B); nothing to choose"],
        ]}
      />
    </Stack>
  );
}

function OverviewTab() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>The problem</H2>
        <Text tone="secondary">
          The topology mapper picks which physical chips each logical mesh lands on without knowing the pipeline that will run
          there. The pipeline builder is the first thing that checks, after the mapping is committed, and fails with "no valid
          submesh assignment found" or a CarvePlanError. The only recourse today is to enumerate every mapping offline and launch
          them one by one.
        </Text>
      </Stack>
      <FlowCompare />
      <Stack gap={8}>
        <H2>Where each check runs</H2>
        <TierTimeline />
      </Stack>
      <Callout tone="warning" title="First version: STRICT only">
        Every pipeline edge is STRICT and rides a fabric link. The schema has no edge policy field yet (field 3 is reserved).
        RELAXED edges, which may go over host MPI, are later work (M5: P14 then P11) and are tagged "RELAXED (later)" wherever
        they appear. Channel policy on meshes and connections (channels policy: RELAXED) is a different, existing feature and
        is unchanged.
      </Callout>
      <Grid columns={fit(260)} gap={16}>
        <Callout tone="info" title="Prove a placement exists">
          Tier C accepts a seating when it finds some carve and layout. That placement is a witness; blaze carves again at runtime
          and may land elsewhere. That is fine while nothing consumes the mapper's placement.
        </Callout>
        <Callout tone="warning" title="Never eliminate early">
          Tiers A and B and Plan 1 reject only what is impossible for every stage-to-mesh split and every chip layout. Tier C
          is the one exception, by choice: it checks one carve per seating, so a seating whose first carve fails the resolver
          is rejected even if another carve would pass. No benchmark case did (101 of 101 first carves passed; open question 24).
        </Callout>
        <Callout tone="neutral" title="Use the builder's code">
          Reuse the C++ resolver as-is for every layout rule. The carve comes from the footprint solve; blaze&apos;s planner stays
          the runtime reference. C1 reads inter-mesh cables from the PSD (P5); the policy plan's shared candidate table is a later step (P16).
        </Callout>
      </Grid>
      <RunningExamples />
    </Stack>
  );
}

function DescriptorTab() {
  return (
    <Stack gap={22}>
      <Grid columns={fit(300)} gap={20} align="start">
        <Stack gap={8}>
          <H2>The new MGD section</H2>
          <Text tone="secondary">
            Stages, their shapes, and how they connect. No stage names a mesh: the mapper decides where each stage lands. GPT-OSS
            single galaxy, as one MGD file: the mesh part is blaze's existing single_bh_galaxy_4x2 MGD, unchanged; the pipeline
            part is new.
          </Text>
          <CodeCard
            path="gpt_oss_120b_single_galaxy.textproto"
            mode="plain"
            language="protobuf"
            src={`
# --- Existing: the mesh ------------------------------------------
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 8, 4 ] }   # 8 rows x 4 cols = 32 chips
  host_topology   { dims: [ 2, 2 ] }   # 2x2 hosts = 4 ranks; each owns
                                       #   (8/2) x (4/2) = a 4x2 slice
  channels { count: 2 policy: RELAXED }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }

# --- New: the pipeline -------------------------------------------
pipeline_descriptors {
  name: "gpt_oss_120b_single_galaxy"
  stages { id: "s0" label: "Embedding" shape { dims: [ 4, 2 ] } }
  stages { id: "s1" label: "Windowed"  shape { dims: [ 4, 2 ] } }
  stages { id: "s2" label: "Global"    shape { dims: [ 4, 2 ] } }
  stages { id: "s3" label: "LMHead"    shape { dims: [ 4, 2 ] } }
  edges { src: "s0" dst: "s1" }
  edges { src: "s1" dst: "s2" }
  edges { src: "s2" dst: "s3" }
  edges { src: "s3" dst: "s0" }
}`}
          />
        </Stack>
        <Stack gap={4}>
          <SingleMeshRing />
          <Text size="small" tone="tertiary">
            One way to place it: single-mesh tiling gives four 4x2 tiles, the resolver lays the ring on them.
          </Text>
          <CodeCard
            path="where the 4x2 slice comes from"
            mode="plain"
            language="text"
            src={`
device_topology [8, 4] / host_topology [2, 2]

              cols 0-1    cols 2-3
  rows 0-3    rank 0      rank 1
  rows 4-7    rank 2      rank 3

= four 4x2 slices, one per rank. Each stage is
[4, 2] = one slice, so each runs on one rank.`}
          />
        </Stack>
      </Grid>

      <Grid columns={fit(260)} gap={16} align="start">
        <CodeCard
          path="joint stage"
          mode="plain"
          language="protobuf"
          src={`
# a 4x4 LM head over two 4x2 slices
# = one stage, co-owned by two ranks
stages { id: "s7" label: "LMHead"
         shape { dims: [ 4, 4 ] } }`}
        />
        <CodeCard
          path="fork, both returns on fabric"
          mode="plain"
          language="protobuf"
          src={`
edges { src: "router"   dst: "target0" }
edges { src: "target37" dst: "router" }
edges { src: "router"   dst: "draft0" }
edges { src: "draft10"  dst: "router" }
# RELAXED (later): policy: RELAXED here to return over host`}
        />
        <CodeCard
          path="heterogeneous (llama_8b_pod)"
          mode="plain"
          language="protobuf"
          src={`
# 8 x 4x2 then 32 x 1x2, one ring
stages { id: "s0" shape { dims: [ 4, 2 ] } }
# ... s1..s7 4x2
stages { id: "s8" shape { dims: [ 1, 2 ] } }
# ... s9..s39 1x2`}
        />
      </Grid>

      <Stack gap={8}>
        <H2>Which mesh a stage can bind to</H2>
        <Text tone="secondary">
          A stage must be a rectangle of whole host slices, in the mesh's logical orientation (no transpose). Example mesh: 8x4
          device, host_topology 2x2, so each slice is 4x2.
        </Text>
        <Table
          headers={["Stage shape", "Slices", "Ranks", "Binds?", "Why"]}
          rows={[
            ["4x2", "1", "1", "yes", "one slice"],
            ["4x4", "2", "2", "yes", "joint stage, co-owned"],
            ["8x2", "2", "2", "yes", "joint stage, vertical"],
            ["2x4", "-", "-", "no", "not whole slices in this orientation"],
            ["2x2", "1/2", "-", "no", "a rank's slice cannot be split"],
          ]}
          rowTone={["success", "success", "success", "danger", "danger"]}
        />
        <Text size="small" tone="tertiary">
          NOTE: no stage is required to be one slice. The descriptor does not say which stage owns host-to-device, so Tier B
          cannot know which stage, if any, must be a single rank. Blaze's carve planner still rejects a multi-rank stage 0 at
          build time.
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>Edge policy: STRICT or RELAXED, per edge - RELAXED (later)</H2>
        <Callout tone="warning" title="RELAXED (later)">
          Not in the first version: every edge is STRICT: the policy field exists, but Tier A rejects RELAXED. This is the design for M5 (P14, P11).
        </Callout>
        <Text tone="secondary">
          Each edge says whether it must ride the fabric; the words match channels policy. STRICT is the default (the field is optional, as on
          channels): an edge with no policy is STRICT, and only edges marked policy: RELAXED may go over host. Every edge STRICT is blaze's
          placement="strict" and every edge RELAXED is placement="flexible"; a mix, such as a STRICT ring whose only RELAXED edge
          is the loopback, is what neither mode can say. There is no HOST value: an edge that always goes over host needs no
          link, so to the mapper it is RELAXED. Blaze's Edge carries the same policy, so the MGD and the runtime graph agree.
        </Text>
        <Grid columns={fit(340)} gap={16} align="start">
          <CodeCard
            path="mixed4 with a RELAXED loopback"
            mode="plain"
            language="protobuf"
            src={`
// PipelineEdge: optional Policy policy = 3 (the enum channels uses); omitted = STRICT

edges { src: "s0" dst: "s1" }                    # STRICT
edges { src: "s1" dst: "s2" }                    # STRICT
edges { src: "s2" dst: "s3" }                    # STRICT
edges { src: "s3" dst: "s0" policy: RELAXED loopback: true }   # may go over host`}
          />
          <Table
            headers={["Step", "STRICT edge", "RELAXED edge"]}
            rows={[
              ["Tier B crossing rules", "rides a declared connection, within its channels", "left out"],
              ["Plan 1 k_AB", "counted", "not counted"],
              ["Tier C", "a fabric link in the layout", "the host-cut planner may cut it; each fabric chain resolved alone"],
            ]}
          />
        </Grid>
        <Callout tone="neutral" title="RELAXED is allowed on every edge">
          An MGD stage does not have to map onto a blaze stage yet, so the MGD does not restrict where RELAXED goes. Tier B and
          Plan 1 leave every RELAXED crossing out. Tier C mirrors blaze's host-cut planner, which never cuts stage 0's
          outgoing edge, so a RELAXED one there still needs a fabric link (_group_walk skips 0 in cuts; stage 0's rank drives
          inference and cannot pump).
        </Callout>
        <Callout tone="info" title="Note: rules that might be good to add">
          Once MGD stages map onto blaze stages, Tier A could reject RELAXED where blaze can never carry the edge over host:
          stage 0's outgoing edge, and any edge of an open chain (graph.py raises "host edges are
          only supported on a looped-back ring"). Tier B and Plan 1 would then count those edges as STRICT, which prunes more.
        </Callout>
      </Stack>

      <Stack gap={8}>
        <H2>How Tier A and Tier B differ</H2>
        <Text>
          Both run inside every MeshGraphDescriptor constructor. Tier A asks whether the pipeline text is a valid descriptor.
          Tier B asks whether those stages can fit on the meshes that same descriptor declares. Tier A never looks at a mesh.
          Tier B never looks at the physical machine, and it does not choose which mesh a stage lands on.
        </Text>
        <Table
          headers={["", "Tier A", "Tier B"]}
          rows={[
            ["When", "static_validate, on the proto, before populate", "after populate, on this descriptor's meshes"],
            ["Sees", "stage ids, edges, shapes, pipeline names", "host_topology rank grids, slices, declared connections"],
            ["Asks", "Is this text a pipeline?", "Can these stages fit on these meshes?"],
            ["Does not", "look at a mesh, or limit how stages connect", "see the physical machine, or pick an assignment"],
          ]}
        />
        <Text tone="secondary">
          On 4×2 slices, Tier A accepts [4, 2], [4, 4], and [2, 4], because each is two positive numbers. Tier B keeps [4, 2]
          (one slice) and [4, 4] (two slices). [2, 4] binds to no mesh: 2 is not a multiple of 4, and the shape is not turned
          sideways. mixed4's stages and edges pass Tier A. Tier B then drops "all four stages on M0" (6 slices, M0 has 4) and
          "s0, s2 on one mesh and s1, s3 on the other" (4 crossings, 2 channels). "s0, s1 on M0 and s2, s3 on M1" uses 3 and 3
          and crosses twice, so it survives. Several assignments survive, and none is chosen.
        </Text>
        <Text tone="secondary">
          Loading a part runs both tiers on that part. merge builds a new descriptor, so both run again. The second Tier B can
          reject two pipelines which each fit their own file but do not fit together.
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>Tier A rules (parse time)</H2>
        <Table
          headers={["Rule", "Rejected example"]}
          rows={[
            ["Stage ids unique and non-empty; edges name real stages", 'edges { src: "s0" dst: "s9" } with no s9'],
            ["No self-edges", 'edges { src: "s0" dst: "s0" }'],
            ["The edges not marked loopback have no cycle", 'edges s0 > s1 and s1 > s0, neither marked loopback'],
            ["Each loopback edge closes a cycle", 'only edges { src: "s0" dst: "s1" loopback: true }'],
            ["Exactly one stage has no incoming edge besides loopbacks; it owns host I/O", "two chains, s0 > s1 and s2 > s3"],
            ["Shapes rank 2, dims > 0", "shape { dims: [ 4, 2, 1 ] }"],
            ["Pipeline names unique and non-empty; each pipeline is checked on its own", "two pipelines both named mixed4"],
          ]}
        />
        <Text size="small" tone="tertiary">
          Repeating an edge declares a second channel on that hop. The edge that closes a ring is marked loopback: true, as in s3 to s0; a fork marks each return.
          Every edge is STRICT in the first version. RELAXED (later): any edge may be RELAXED; Tier A will not restrict where.
        </Text>
        <Callout tone="info" title="NOTE: beyond one host I/O stage, Tier A does not limit stage topology">
          In-degree, out-degree, and whether a branch is a chain are all accepted. A graph
          blaze cannot build still passes Tier A and fails at build time.
        </Callout>
        <Callout tone="warning" title="Dead-end branches are allowed">
          A branch may end without an edge back. Tier A accepts that. A graph blaze cannot build still passes Tier A and
          fails at build time.
        </Callout>
      </Stack>

      <Stack gap={8}>
        <H2>Merging MGDs</H2>
        <Text tone="secondary">
          The mapper always merges its MGD parts. With two or more parts, merge rebuilds the proto field by field and would drop
          the pipelines. P2 copies every part's pipelines and rejects a name declared twice. Each part is a MeshGraphDescriptor,
          so Tier B has already run on that part's meshes. merge() constructs a new descriptor, and Tier B runs again on the
          merged meshes and every pipeline together. Plan 1 reads the merged descriptor's facts. Tier C places the pipelines on
          disjoint slices and resolves each on its own.
        </Text>
        <MergeDiagram />
      </Stack>
      <DescriptorExamples />
    </Stack>
  );
}

function MapperTab() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>The per-seating check</H2>
        <Text tone="secondary">
          MultiMeshSolutionEnumerator::next() today stops after step 2. Tier C adds four steps after intra-mesh success, and
          they run only when the MGD has a pipeline. This tab covers C1 and C2; C3 and C4 have their own tabs.
        </Text>
        <Table
          headers={["Step", "Does", "Outputs", "First version", "Later (optional)"]}
          rows={[
            ["C1. Build the physical pipeline graph", "candidates, host slices, adjacency -> one position graph and its constraints", "one position graph per seating, with chip-level link tables", "every STRICT edge (P5)", "RELAXED edges as soft edges in the same solve (P11)"],
            ["C2. Pipeline solve", "footprint solve picks the carve, stage order and crossings; the resolver checks chip collisions; R recorded as exact channels", "carve, layout and R within each demand, or a reason", "one solve on the existing API (completeness as cardinality constraints, conflict_cap); pre-checks; the resolver on that one carve; any failure fails the seating; per-stage misfits filtered before the solve (P6)", "the demand as a crossing cap inside the encoding"],
            ["C3. On success", "emit the mapping and publish R", "mapping, plan, get_pipeline_required_links()", "phase 0 and fill in the control plane (P9)", "phase 0 in the mapper (policy plan part 11)"],
            ["C4. On failure", "constraints for the next solve", "exit-node retry, then exclude_mapping(seating)", "seating exclusion, unique_shapes off, cap (P8); exit-node retry (P15)", "seat-pair clauses (P13); constraints before the first solve (P10)"],
          ]}
        />
        <Callout tone="info" title="Rule: do not commit to a carve before it is needed">
          A different intra-mesh layout or a different set of bound cables can make a different carve work. So C1 builds
          its one position graph over every cable that could be bound, not a binding fixed in advance, and the binding follows the solve (R is
          bound first). A constraint C4 derives from a failure only steers that seating's retry. Anything added before the
          intra-mesh or master solve may only state what every carve agrees on (minimum crossings, union of boundary slices). A
          reject constrains the seat set, never one mesh on one seat.
        </Callout>
        <Text size="small" tone="tertiary">
          C2 sees one intra-mesh layout at a time, so a failure is exact for that layout, not the seating. C4&apos;s exit-node
          retry (P15) tries other layouts of the same seating before excluding it.
        </Text>
        <Legend />
        <LoopFlow />
      </Stack>
      <Grid columns={fit(380)} gap={20} align="start">
        <BindingPiece
          title="New: inter-mesh cables per seating"
          tag="C1: P5"
          what="C1 reads the cables between two mapped meshes straight from the PSD (inter_mesh_cables in pipeline_builder.cpp, P5), as ExitNodeConnections: both ends mapped onto the two meshes, requested exit nodes when the connection is device-level, one destination chip per source chip. Cable type and link state are not filtered yet; that comes with the policy plan's candidate table (P16, later)."
          todo={[
            "In build_physical_pipeline_adjacency_graph, call inter_mesh_cables(psd, full.fabric_node_to_asic, a, b) for each mesh pair the MGD connects.",
            "Per position-graph edge, keep the links behind it and each inter-mesh link's ExitNodeConnections; C2 builds DirectLinks from them for the chosen rectangles, ordered by chip id.",
            "Later (P16): swap in collect_inter_mesh_candidates, shared with the control plane, with tiers and down links.",
          ]}
        />
        <BindingPiece
          title="New: R must fit each connection's demand"
          tag="C2: P6"
          what="Phase 0 takes one channel per link in R, counted against the pair's demand. Tier B's channel rule does not guarantee this: it checks that some projection fits, and C1 may carve one with more crossings."
          todo={[
            "Budget: links in R on a pair <= strict + relaxed; R uses the STRICT part first, then RELAXED.",
            "RELAXED and BEST_EFFORT pairs with no link in R keep the policy plan's rules, zero-link included; pairs with links in R need them.",
            "Check in C2 as soon as R is recorded. Over budget fails the seating (pipeline_binding, the mesh pair in the reason), and C4 runs.",
            "If it fires in practice: a per-mesh-pair crossing cap inside the footprint encoding, so over-demand carves are never offered.",
          ]}
        />
      </Grid>
      <Stack gap={8}>
        <H2>C2: one solve, one check, one C1 graph</H2>
        <Text tone="secondary">
          There is one position graph and one carve per seating. C2 solves once: the carve is one rectangle per stage, the
          stage order and the crossings. The resolver checks that carve for chip collisions; R is recorded as exact channels
          and checked against each connection&apos;s demand. Any failure fails the seating and C4 runs; C2 never asks for a
          second carve.
        </Text>
        <Table
          headers={["Part", "What it does"]}
          rows={[
            ["Pre-checks", "area, lattice count, colour and segment test on tight 1-cable meshes; ruled-out cables never enter the position graph"],
            ["Solve", "target = pipeline, global = position graph; required: stage to positions of its shape; resource: position to covered slices; Auto = SAT"],
            ["Solver changes", "none: required, resource and cardinality constraints, one session next() under conflict_cap; topology_solver_sat.cpp gets no pipeline or session code"],
            ["Check", "resolver once, on one submesh per chosen rectangle; chip counts force its placement, so its link step is the chip-collision check"],
            ["R", "from the resolver's resolved_edges, mapped to the exact candidate channel (IMPORTANT, below)"],
            ["Cap hit", "reads as no carve with today's API; the seating fails and C4 runs (open question 13)"],
          ]}
        />
        <Callout tone="warning" title="IMPORTANT: record the exact channel each crossing uses, and bind that channel">
          The mapping only says two rectangles touch through some cable; with several cables between them it does not pick
          one. The resolver&apos;s link step does, and resolved_edges gives the exit and entry chip. C2 maps that chip pair to its
          PSD ExitNodeConnection (lowest tier, then lowest source ETH channel) and appends it to R as is; its
          std::hash&lt;ExitNodeConnection&gt; is the hash the pairing and its broadcast use. C3 publishes it through get_pipeline_required_links(),
          and phase 0 of pair_logical_intermesh_ports binds exactly those channels before any other. Binding another channel,
          even between the same two slices, can move an endpoint onto a chip the runtime&apos;s resolver rejects. A recorded
          channel that cannot be bound is a hard error naming the pipeline edge, never a silent substitute.
        </Callout>
        <H2>When C2 fails</H2>
        <Table
          headers={["Failure", "Found by", "Result"]}
          rows={[
            ["stage s cannot fit on position p on its own (endpoints exceed p's core slots; stage 0's host I/O does not fit)", "C1, before the solve", "p left out of s's required positions; never offered"],
            ["a pre-check fails", "before the solve", "seating fails: C4, with the mesh and reason"],
            ["no carve (or the conflict cap hit)", "the solve", "seating fails: C4"],
            ["the resolver throws on the carve", "the check", "seating fails: C4, the message is the reason"],
            ["R over demand on (A, B)", "demand check", "seating fails: C4, pipeline_binding and the pair"],
          ]}
        />
        <Text tone="secondary">
          Why one carve is enough: the resolver accepted the first carve on all 101 benchmark cases
          (results/rerun_2026-10-27/footprint_then_resolver, at most 4 ms each). The one case built to fail it
          (chip_level/reject_demo) has no passing carve at all. The cost: a seating can be lost when another carve on it would
          pass. If a real MGD hits that, the same session gives a second carve with exclude_mapping(carve) and next(), still the
          existing API (open question 24).
        </Text>
        <H2>Speed: blaze&apos;s carve planner + resolver against footprint + resolver</H2>
        <Table
          headers={["Group (60 s cap)", "Cases", "Planner + resolver: placed", "Footprint + resolver: decided", "Resolver after footprint, worst"]}
          rows={[
            ["original carve cases", "17", "0 (17 budget-exhausted)", "16", "0.029 s"],
            ["heterogeneous sweep, 60 to 95% fill", "30", "0 (30 budget-exhausted)", "30", "0.001 s"],
            ["single mesh, no 1x1 stages", "45", "0 (stage-0 rule; off: 2)", "43", "0.029 s"],
            ["heterogeneous multi-mesh", "12", "0 (stage-0 rule; off: 0)", "12", "0.001 s"],
            ["multi-mesh, no 1x1 stages", "28", "0 (stage-0 rule; off: 0)", "26", "0.036 s"],
            ["all", "132", "0", "127", "0.036 s"],
          ]}
        />
        <Table
          headers={["Case (seconds, carve + resolver)", "Planner + resolver", "Footprint + resolver"]}
          rows={[
            ["4x4: 1x1, 3x 2x2, 1x1", "0.000 + 0.000", "0.000 + 0.000"],
            ["4x8: 1x1, 2x4, 2x4, 2x2, 1x1", "0.009 + 0.000", "0.001 + 0.000"],
            ["4x8 ring of 32 x 1x1", "0.038 + 0.001", "0.004 + 0.001"],
            ["4x8 ring of 16 x 1x2 (planner stage-0 rule off)", "0.036 + 0.000", "0.002 + 0.000"],
            ["4x4: 1x1, 3x 2x2 (feasible)", "wrong reject: no-cycle-of-length-13", "0.000 + 0.000"],
            ["4x6 and 4x8: 1x1, 2x2 ring, 1x1 (feasible)", "wrong reject: no-alignment", "0.001 + 0.000"],
            ["8x8 fill: 1x1, 15x 2x2, 3x 1x1", "budget exhausted", "0.026 + 0.000"],
            ["16x16: 1x1, 62x 2x2, 1x1", "budget exhausted", "0.13 + 0.004"],
            ["random 8x8, 30 stages", "budget exhausted", "0.60 + 0.001"],
            ["random 8x16, 37 stages", "budget exhausted", "5.6 + 0.001"],
            ["4 meshes of 4x8, 1 cable per pair, ring of 112 x 1x1", "budget exhausted", "9.9 + 0.020"],
            ["7x7: 1x1, 10x 2x2 (infeasible)", "budget exhausted, no verdict", "no, 2.1"],
            ["5x5 ring of 25 x 1x1 (odd, infeasible)", "budget exhausted, no verdict", "no, 0.003"],
          ]}
        />
        <Callout tone="info" title="Use footprint + resolver">
          The planner only completes on small rings of columns, where both take milliseconds. On grid-shaped meshes it has no
          degree-2 chains to contract, so its cycle search runs out of budget on every 8x8 or larger grid and every multi-mesh
          case; its stage-0 rule rejects every case without a 1x1 stage, and its grouping wrongly rejects 16 cases the
          footprint carves in milliseconds. In the footprint path the carve is the whole cost: the resolver checks every
          carve in at most 36 ms.
        </Callout>
        <Text size="small" tone="tertiary">
          Planner: blaze&apos;s plan_ring_carve ported to C++ (topo/carve_planner_port.hpp, blaze&apos;s 200k-expansion and 64-cycle
          budgets). Footprint: with resource completeness. Source: REPORT.md section 5.3 and
          results/rerun_2026-10-27/carve_comparison.md, one machine.
        </Text>
        <H2>The check: resolver inputs</H2>
        <Table
          headers={["Resolver argument", "What C2 builds for the chosen rectangles", "Rule it enforces"]}
          rows={[
            ["nodes", "every stage name, isolated stages included", "every edge endpoint listed, no duplicates"],
            ["edges", "(src, dst, is_loopback); is_loopback is the MGD edge's loopback mark, checked by Tier A", "non-loopback edges form a DAG"],
            ["submesh_chips", "per rectangle: (mesh_id, chip_id, row, col)", "(row, col) slot-local, from the rectangle's own origin"],
            ["node_chip_counts", "rows x cols of each stage's shape", "forces each stage onto its own rectangle"],
            ["node_pipeline_core_counts", "blaze's _pipeline_core_defaults, or empty", "empty: 1 slot per chip on 8+ chips, 2 on smaller"],
            ["direct_links", "per ordered rectangle pair, every candidate LinkPair between them", "directed per key: insert both {i, j} and {j, i}"],
          ]}
        />
        <Text tone="secondary">
          Fail fast: every C2 run ends in a carve or a proof. The tight 1-cable cases a solver is worst at are decided by the
          segment test before the solve (Itai-Papadimitriou-Szwarcfiter: colours plus a few cases on grids 1 to 3 wide). Not
          covered: tight meshes with dead slices, and tight multi-mesh rings with several cables per pair; there the
          session&apos;s conflict_cap bounds the solve, and a capped solve reads as no mapping with today&apos;s API (open question 13).
        </Text>
      </Stack>

      <Stack gap={10}>
        <H2>Worked example: where the links land matters</H2>
        <Text tone="secondary">
          A 4-stage ring split two stages per mesh crosses M0 to M1 twice. Every case below declares two channels, so the
          descriptor-only checks cannot tell them apart. Pick a case, and whether the loopback s3 &gt; s0 is RELAXED (later; the first version is all STRICT).
        </Text>
        <CaseExplorer />
        <Table
          headers={["Case", "Bound links", "every edge STRICT", "s3 > s0 RELAXED (later)"]}
          rows={(["A", "B", "C"] as CaseId[]).map((id) => [
            `Case ${id}`,
            CASES[id].title,
            CASES[id].strict.ok ? "accept: all-fabric ring" : "reject: no 4-slice cycle",
            CASES[id].relaxed.host ? "accept: fabric chain + host loopback" : "accept: all-fabric ring",
          ])}
          rowTone={[undefined, "warning", "warning"]}
        />
      </Stack>

      <Stack gap={10}>
        <H2>Blaze&apos;s carve paths and the one position graph</H2>
        <Text tone="secondary">
          Blaze turns meshes into submeshes in one of these ways. Tier C does not try them one by one: C1 builds one position
          graph per seating (every rectangle each stage shape fills), which covers all of them, and C2 solves it once.
        </Text>
        <CarvePaths />
        <Table
          headers={["MGD", "Blaze at runtime", "Tier C, every edge STRICT", "Tier C, some edge RELAXED (later)"]}
          rows={[
            ["one mesh, every stage on one rank", "single-mesh tiling", "one footprint solve", "same, RELAXED edges soft"],
            ["one mesh with a joint stage, or several meshes", "planned carve (or per-rank tiling with submesh_shape)", "one footprint solve", "same, RELAXED edges soft"],
            ["a stage smaller than a host slice", "per-rank tiling, first tile", "not modelled; Tier B rejects", "same (open question 15)"],
          ]}
        />
        <Text size="small" tone="tertiary">
          Blaze's multi_mesh flag is set for more than one mesh or any stage spanning several ranks, so a galaxy with 4x1 rank
          columns and 4x2 stages takes blaze's planned carve (3.2, example F).
        </Text>
      </Stack>

      <Stack gap={10}>
        <H2>Reuse or rebuild</H2>
        <Table
          headers={["Piece", "Today", "Plan", "PR"]}
          rows={[
            ["Graph-layout resolver", "resolve_graph_layout_with_connections (C++)", "use as-is; a throw rejects the carve", "P8"],
            ["Submesh links", "discover_submesh_links (C++, asks the control plane)", "not needed: C2 supplies direct_links", "-"],
            ["Host-rank chip ranges", "MeshGraph::get_coord_range, coordinate_to_chip", "use as-is", "P8"],
            ["Carve planner", "carve_planner.py", "replaced by the footprint solve (not ported)", "P5"],
            ["Slots from the carve", "_collect_submeshes (Python)", "rebuild, small: one slot per chosen rectangle", "P6"],
            ["Host-cut planner (RELAXED, later)", "host_edge_planner.py", "replaced by soft edges in the same solve", "P11"],
            ["Per-chain resolve (RELAXED, later)", "_resolve_chains_separately (Python)", "loop over the resolver", "P11"],
            ["Inter-mesh binding", "pair_logical_intermesh_ports (control plane)", "pure helpers, shared", "P5"],
          ]}
        />
      </Stack>

      <Stack gap={8}>
        <H2>Where blaze&apos;s planner and the footprint differ</H2>
        <Table
          headers={["Behaviour in blaze's planner", "Footprint solve in Tier C"]}
          rows={[
            ["Returns the first aligned cycle; never retries if the resolver then fails", "the same: one carve, checked once (open question 24)"],
            ["Stage 0 exactly one slice", "stage 0 on any position with room for host I/O (open question 14)"],
            ["First-fit grouping: no-alignment on 16 feasible benchmark cases", "exact"],
            ["budget-exhausted after 200,000 expansions (every 8x8+ grid)", "SAT with resource completeness (cardinality constraints) and conflict_cap"],
            ["Adjacency: one confirmed link makes two slices adjacent", "same, from the candidate links; R records the exact channel"],
            ["Stage rectangle must equal the declared (rows, cols)", "same, no transpose"],
            ["Unused slices become idle filler slots", "same"],
            ["crc32 digest exchanged across ranks", "not needed: the mapper solves centrally"],
          ]}
        />
      </Stack>

      <Stack gap={10}>
        <H2>Without a pipeline descriptor: every hook off</H2>
        <Text tone="secondary">
          An MGD with no pipeline_descriptors after merge maps, binds and writes rank bindings exactly as today: no extra search,
          no extra SAT clauses, no changed setting. Generated MGDs (generate_mesh_graph_descriptor_of_shape) have no pipelines,
          so they are always off. One gate, not a proto check per hook: the merged descriptor's has_pipeline_descriptors()
          (and, in the backup plan, pipeline_mapping_constraints(), null when off), and the enumerator's pipelines_, built once from
          the merged descriptor. An empty pipelines_ means the hook adds nothing, never "run with defaults".
        </Text>
        <CodeCard path="the gate" mode="plain" language="cpp" src={PIPELINE_GATE} />
        <Table
          headers={["Hook", "Where", "When off"]}
          rows={[
            ["schema pipeline_descriptors (1.3)", "MGD proto", "absent field; existing textprotos parse unchanged"],
            ["merge (1.5)", "MeshGraphDescriptor::merge", "merged proto byte-identical to today's"],
            ["Tier A (2.1)", "MGD constructor", "loops over zero pipelines"],
            ["Tier B (2.2)", "MGD constructor", "check_pipeline_feasibility returns at once; nothing is stored"],
            ["stage-to-mesh, seat and exit helpers (2.2, 4.3)", "mapper helpers", "return without adding a constraint"],
            ["seat-graph edge drop for k_AB (4.3)", "build_sat_placement_seat_graph", "today's saturated_link_count edges"],
            ["unique_shapes (4.3)", "map_multi_mesh_to_physical, generate_rank_bindings --all-solutions", "true, as today"],
            ["exact check and SAT feedback (4, 4.3)", "MultiMeshSolutionEnumerator::next", "today's loop; no pipeline_* failure stage"],
            ["required links R and budget check (4.2)", "mapper", "R empty; no budget check"],
            ["phase 0 (4.2)", "pair_logical_intermesh_ports", "no required channel: today's phases"],
            ["single-host phase 0 (4.2)", "generate_intermesh_connections_on_local_host", "nothing required: today's greedy"],
            ["stage plan in rank bindings (4.4)", "generate_rank_bindings", "no stage fields written"],
          ]}
        />
        <Text tone="secondary">
          Nothing here changes the control plane&apos;s gather: P5's inter_mesh_cables only reads the PSD inside the per-seating check, and only for MGDs with
          a pipeline. Moving to the policy plan&apos;s candidate table (P16, later) is that plan&apos;s refactor, with its own parity
          tests.
        </Text>
        <Text tone="secondary">
          Kill switch (proposed): TT_METAL_FABRIC_IGNORE_PIPELINE_DESCRIPTORS=1 in rtoptions makes the mapper treat the descriptor
          as having no pipelines. Tier B and every mapper hook are off; Tier A still parses and validates the format. An escape
          hatch for a Tier B false reject or a mapper bug, and an A/B comparison on one MGD. Blaze (Tier D) is unaffected.
        </Text>
        <Table
          headers={["Test", "Expect"]}
          rows={[
            ["every MGD in the repo (none has pipelines today), single-host and multi-host", "mapping, rank bindings and AnnotatedIntermeshConnections match main exactly"],
            ["counter or test-only hook, no descriptor", "no check_pipeline_feasibility or check_pipeline_seating call, no extra SAT clauses"],
            ["pipeline MGD with the kill switch", "same mapping as that MGD with pipeline_descriptors removed"],
          ]}
        />
      </Stack>
      <MapperExamples />
    </Stack>
  );
}

const PIPELINE_GATE = `
// MeshGraphDescriptor (merged)
bool has_pipeline_descriptors() const;                                  // proto_->pipeline_descriptors_size() > 0
// Backup if the solve is slow (not implemented):
// const PipelineMappingConstraints* pipeline_mapping_constraints() const; // null when off

// MultiMeshSolutionEnumerator, built once in the constructor from the merged MGD
std::optional<LogicalPipelineGraph> pipelines_;                        // empty when off`;

function SatTab() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>C4: on failure, constraints for the next solve</H2>
        <Text tone="secondary">
          The SAT targets are logical meshes; the globals are seats (physical footprints). A seat fixes the footprint, but its
          chip layout is only a preference to the intra-mesh solve, so which ASICs sit under each host slice is unknown before
          solving (unless hard pinnings fix it).
        </Text>
        <Table
          headers={["Move", "Scope", "What it adds", "When"]}
          rows={[
            ["1. Exit-node retry", "the same seating", "intra-mesh block on the exit-node slice signature, then back to C1", "first version (P15), at most 4 per seating"],
            ["2. Exclude the seating", "the SAT placement", "exclude_mapping(seating), then the next seating", "first version (P8)"],
            ["later: seat-pair nogood", "the SAT placement", "add_seat_pair_nogood: not mesh_a on seat_a with mesh_b on seat_b (and twins), when proven for every carve", "P13"],
            ["later: before the first solve", "every seating", "exit-node and seat constraints every carve agrees on", "P10, only if measured"],
          ]}
        />
        <Grid columns={fit(380)} gap={20} align="start">
          <BindingPiece
            title="Move 1: exit-node retry on the same seating"
            tag="P15"
            what="A C2 failure is exact for the layout, not the seating: another layout puts the cabled ASICs under other host slices. Record each boundary mesh's slice signature (cabled exit ASIC -> slice) and re-solve that mesh's intra-mesh problem with the signature blocked, keeping the other meshes fixed."
            todo={[
              "Skip when hard rank bindings or MGD pins fix every cabled ASIC's slice (runtime with every host bound): the signature cannot change.",
              "Block with an existing cardinality constraint on the fresh intra-mesh solve: at least one exit ASIC moves to another slice. No new solver API.",
              "Blocks are per retry, not kept: once another mesh's layout changes, an old signature may work again.",
              "At most kMaxLayoutRetries (proposed 4) per seating, then move 2.",
            ]}
          />
          <BindingPiece
            title="Later, optional: exit-node constraints for every seating"
            tag="P10"
            what="Before C1, on every seating, steer the intra-mesh solve's exit nodes, only with what every carve agrees on (Tier B's minimum k and union of boundary slices); k = 0 adds nothing."
            todo={[
              "Through the generic ExtraMappingConstraints::intra_mesh hook: at least ceil(k / links per exit ASIC) nodes of boundary_slices(m, n) on ASICs cabled toward n.",
              "Only if P8's rejection counts call for it.",
            ]}
          />
        </Grid>
        <Callout tone="info" title="Decision">
          First version: Plan 2 (P8) and the exit-node retry (P15). After intra-mesh success, run C1 and C2 (one carve); when C2 fails and the exit-node retry (P15) cannot help, exclude_mapping(seating)
          blocks that seat set in the live master session and the loop draws the next seating. Plan 1 (P10) is a later,
          optional speed-up, added only if the measured rejection counts call for it. It must never remove a seating Plan 2
          would accept, so it changes speed, not answers.
        </Callout>
        <Table
          headers={["", "Option 1: pipeline in the SAT solve", "Option 2: check each seating, feed back"]}
          rows={[
            ["Solves", "one", "one master session; each rejected seating is blocked in place, no re-encode"],
            ["Exact?", "no: only necessary conditions fit at seat level", "yes: the footprint solve and the builder's resolver on the real layout"],
            ["New code", "seat-graph edge drop, seat and exit constraints (P10)", "the check plus one exclude_mapping call in the existing loop (P8)"],
            ["Cost", "a larger master encoding", "one intra-mesh solve and one check per rejected seating"],
          ]}
        />
        <Text tone="secondary">
          Option 1 cannot be the whole check. Feasibility depends on facts that exist only after the intra-mesh solve: which
          chips sit under each host slice, a stage whose entry and exit share a chip, per-chip core capacity, ring alignment,
          stage 0's host I/O. Encoding those exactly needs chip-level and stage-to-slice variables for every mesh in the master,
          close to one flat solve of the whole system. mixed4 on (P, Q) passes every seat-level rule and fails on loop length;
          pod8 on U fails because its cables land on opposite corners of M1. The check runs after intra-mesh completion and
          uses the builder's own planner, not a second SAT model of it.
        </Text>
      </Stack>
      <Grid columns={fit(340)} gap={20} align="start">
        <Stack gap={8}>
          <H3>Plan 1: prune before the solve (later, optional, P10)</H3>
          <Table
            headers={["Constraint", "Sound when", "Effect"]}
            rows={[
              ["Seat pair has fewer candidate links than k_AB", "always (k_AB is the min over projections)", "drop the seat-graph edge"],
              ["Seat pair cannot hit k_AB distinct boundary slices", "hard-pinned layouts only", "drop the edge (rejects Case B early)"],
              ["Seat has no link toward a mesh every projection crosses to", "always", "remove the seat"],
              ["RELAXED edges (later, P14)", "always", "left out of k_AB; every edge RELAXED gives k_AB = 0, nothing pruned"],
            ]}
          />
        </Stack>
        <Stack gap={8}>
          <H3>Plan 2: check and teach (first version, P8)</H3>
          <Table
            headers={["Failure attributed to", "Feedback"]}
            rows={[
              ["the seating as a whole (default)", "exclude_mapping(seating): block this seat set"],
              ["two meshes' seats, proven for every carve", "add_seat_pair_nogood (P13)"],
              ["a seat no carve can use in any seating", "forbid that seat for every mesh"],
              ["never", "add_forbidden_constraint(mesh on seat)"],
            ]}
            rowTone={[undefined, undefined, undefined, "danger"]}
          />
          <Text size="small" tone="tertiary">
            A pipeline failure says "these seats with these layouts", not which mesh was at fault. Forbidding one mesh on one seat
            would eliminate seatings another split could build on. The first version uses whole-seating exclusion only; the
            seat-pair clause waits for attributed failures (P13).
          </Text>
          <Text size="small" tone="tertiary">
            Rejection cap: give up after kMaxPipelineRejections (proposed 256, overridable for sweeps) with the last reason.
            Record the count in the placement stats; those numbers decide whether Plan 1 is needed.
          </Text>
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>Turn unique_shapes off when a pipeline is present</H2>
        <UniqueShapesDiagram />
        <Table
          headers={["Call site", "Today", "With a pipeline"]}
          rows={[
            ["map_multi_mesh_to_physical (both overloads, runtime path)", "true", "false"],
            ["generate_rank_bindings --all-solutions", "true unless --allow-shape-permutations", "false"],
            ["run_topology_mapping (single solution)", "false", "false (no change)"],
            ["seat-pool enumeration", "false (must stay: keeps every orientation)", "false"],
            ["enumerate_flat_grouping_embeddings (3 calls)", "default true", "unchanged (existence checks / distinct placements)"],
          ]}
        />
        <Callout tone="warning" title="Cost">
          Off, the master solve can yield every mesh permutation of a seat set. Cap pipeline rejections per run and measure on
          ring-stress and superpod MGDs. The code change is in P8.
        </Callout>
      </Stack>
      <SatExamples />
    </Stack>
  );
}

function BindingPiece(props: { title: string; tag: string; what: string; todo: string[]; code?: string }) {
  return (
    <Stack gap={8}>
      <Row gap={8} align="center">
        <H3>{props.title}</H3>
        <Pill size="sm">{props.tag}</Pill>
      </Row>
      <Text tone="secondary">{props.what}</Text>
      {props.code ? <CodeCard path="proposed" mode="plain" language="cpp" src={props.code} /> : null}
      <Table headers={["To do"]} rows={props.todo.map((s) => [s])} />
    </Stack>
  );
}

const REQUIRED_LINKS_SRC = `
// No new type: R is the PSD's own ExitNodeConnection entries, one per crossing.
// ExitNodeConnection = {src_exit_node, dst_exit_node (AsicID), eth_conn {src_chan, dst_chan, is_local, port_type}};
// std::hash<ExitNodeConnection> is the connection_hash the pairing and the broadcast use.
// FabricNodeIds come from the mapping; the pipeline edge is PipelinePlacement::required_link, per resolved edge.

// PipelineMappingResult
std::vector<tt::tt_metal::ExitNodeConnection> required_links;   // R

// TopologyMapper, every rank (broadcast with the chip info via exit_node_connection_to_proto)
const std::vector<tt::tt_metal::ExitNodeConnection>& get_pipeline_required_links() const;

// Rank 0 only: phase 0 binds these hashes first. No PortDescriptor change.
AnnotatedIntermeshConnections ControlPlane::pair_logical_intermesh_ports(
    const PortDescriptorTable& port_descriptors, const std::unordered_set<std::size_t>& required_hashes);`;

function BindingTab() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>C3: on success, emit the mapping and publish the cables</H2>
        <Text tone="secondary">
          When C2 accepts, C3 emits the mapping and plan, and makes sure the cables the layout crosses on are bound. The MGD does not change for this: it says how many channels each connection gets and, with the
          [inter-mesh policy plan](https://riddy21.github.io/tt-metal/mgd-intermesh-policy-plan.html), under which policy and on
          which cable types. Which physical cables the pipeline needs depends on the seating and the intra-mesh layout, so it is
          a mapper output.
        </Text>
        <Table
          headers={["Piece", "Where it runs", "When", "New or reused"]}
          rows={[
            ["Required links R", "mapper, from C2's layout", "when C2 accepts", "new"],
            ["Phase 0 in the pairing", "control plane, pair_logical_intermesh_ports on rank 0", "control-plane init", "new"],
            ["Fill order", "same", "after phase 0", "reused: today's order"],
            ["Single host", "generate_intermesh_connections_on_local_host", "control-plane init", "new"],
            ["Phase 0 inside the mapper", "mapper, resolve_inter_mesh_binding", "before emit", "later, with policy plan part 11"],
          ]}
        />
        <Grid columns={fit(380)} gap={16} align="start">
          <Callout tone="warning" title="Order: carve first, then bind">
            Binding follows C2, so the carve is chosen over every candidate cable. Binding first and carving over the bound set
            would throw away carves another binding allows (mixed4 below).
          </Callout>
          <Callout tone="info" title="No rerun over the bound cables">
            Once phase 0 binds all of R, a rerun of C2 cannot fail: the bound set contains R and the witness crosses only on
            R. The rerun is dropped. It would only matter if Tier C checked blaze&apos;s first cycle (question 8).
          </Callout>
        </Grid>
      </Stack>
      <Stack gap={8}>
        <H2>Bind the cables the pipeline needs</H2>
        <Text tone="secondary">
          After mapping, the control plane picks which physical cables carry each MGD connection. Today it ignores the pipeline,
          so it can throw away the cables Tier C relied on.
        </Text>
        <BindingCompare />
      </Stack>
      <Stack gap={8}>
        <H2>Taken from the inter-mesh policy plan (later, P16)</H2>
        <Table
          headers={["Piece", "Policy plan part", "Used here for"]}
          rows={[
            ["get_inter_mesh_channel_demand(a, b): {strict, relaxed}", "3, 5", "the budget R must fit; Tier B's cap(A, B)"],
            ["collect_inter_mesh_candidates(mesh_graph, psd, fabric_node_to_asic, options)", "9", "P16: replaces P5's inter_mesh_cables, once per seating"],
            ["cable-type filter and tier (PortTypeRules)", "6", "disallowed cables never reach C1; better tiers tried first"],
            ["PortDescriptor::tier and the tier sort in the pairing", "9", "the fill order after R"],
            ["resolve_inter_mesh_binding in the mapper (stretch)", "11", "later: phase 0 checked before the mapping is emitted"],
          ]}
        />
        <Text size="small" tone="tertiary">
          Order of work: this design does not wait for part 9. P5 reads the PSD itself and P9 binds R on today&apos;s gather;
          P16 switches C1 to the candidate table once part 9 lands (question 10).
        </Text>
      </Stack>
      <Divider />
      <BindingPiece
        title="New: the required links R, and where they live"
        tag="P9"
        what="R is, for each crossing edge in the layout's resolved_edges, the one ETH channel it uses: the PSD's ExitNodeConnection and its hash, the same hash the pairing and the broadcast use. It is not every bound link (intra-mesh edges are not in it; the pairing fills the rest of the count). Not in the MGD (it depends on the seating and layout), not in rank_bindings.yaml for the first version. It is a mapper output, broadcast with the chip info and passed to the pairing on rank 0 as a set of hashes."
        code={REQUIRED_LINKS_SRC}
        todo={[
          "Fill R at the end of C2 from the accepted layout; store it on the TopologyMapper with the mapping. Empty without a pipeline.",
          "Pass R's hashes to pair_logical_intermesh_ports on rank 0. No PortDescriptor change: other ranks' tables reach rank 0 as proto with only hash, src chip and dst chip.",
          "Multi-host: gather on every rank, pair on rank 0, broadcast (mesh, port_id) pairs with the hash, each rank maps the hash to its ETH channel from its PSD. Every rank then checks that each hash in R was bound.",
        ]}
      />
      <Stack gap={8}>
        <Row gap={8} align="center">
          <H3>New: phase 0 in the pairing, bind R first</H3>
          <Pill size="sm">P9</Pill>
        </Row>
        <PhaseStrip />
        <Text tone="secondary">
          Phase 0 takes every chip pair holding a required hash, moves those hashes to the front so the budget trim keeps
          them, places the link with its pair&apos;s Z rule and counts it against the budget. Today&apos;s work-list then fills
          the rest. With no required links the output is byte-identical to today. Phase 0 never throws on rank 0 (the other
          ranks are waiting for the broadcast); a required channel that was not placed, or not gathered because its link is
          down, fails the check after the broadcast on every rank together, naming the channel and its two chips.
        </Text>
      </Stack>
      <Grid columns={fit(340)} gap={20} align="start">
        <BindingPiece
          title="New: fill order"
          tag="P9"
          what="After phase 0, the remaining budget is filled in today's order, with or without a pipeline, so goldens do not move. Best cable tier first once the policy plan lands."
          todo={["Tier first once the policy plan lands."]}
        />
        <BindingPiece
          title="New: single host"
          tag="P9"
          what="generate_intermesh_connections_on_local_host is a separate greedy pass. It needs phase 0 (and, once the policy plan lands, its candidate-table filter)."
          todo={[
            "Place R first, then today's greedy; or route single host through the pairing (moves single-host goldens).",
            "Until then, warn that R is not guaranteed to be bound on single host.",
          ]}
        />
      </Grid>
      <Grid columns={fit(340)} gap={20} align="start">
        <BindingPiece
          title="Later: phase 0 inside the mapper"
          tag="policy plan part 11"
          what="The policy plan's stretch goal moves channel selection and directions into the mapper (resolve_inter_mesh_binding)."
          todo={[
            "Pass R as required links; call it after C2 accepts.",
            "A failure to place R is the PLACEMENT kind: C4 handles it like a C2 failure, and the init-time failure goes away.",
          ]}
        />
      </Grid>
      <Grid columns={fit(340)} gap={20} align="start">
        <Stack gap={8}>
          <H3>Who computes what</H3>
          <Table
            headers={["Step", "Where", "Ranks"]}
            rows={[
              ["Seating, carve, layout, R, budget check", "TopologyMapper: the enumerator calls detail::check_pipeline_seating (Tier C)", "rank 0"],
              ["Inter-mesh cables (C1)", "pipeline_builder.cpp, from the PSD (P5; part 9 table in P16)", "rank 0"],
              ["Phase 0, then fill", "ControlPlane, pairing", "rank 0"],
              ["Broadcast connections", "ControlPlane", "rank 0 to all"],
              ["validate_requested_intermesh_connections", "ControlPlane", "all"],
            ]}
          />
        </Stack>
        <Stack gap={8}>
          <H3>Gaps</H3>
          <Text tone="secondary">
            Until P16, C1 reads the PSD on rank 0, which has no link state, so R can name a down channel; the check after the
            broadcast then fails on every rank (question 20). With the policy plan&apos;s candidate table (P16) down links are left
            out at runtime. Offline runs (factory PSD) see every cable as up, so the offline answer can be optimistic. STRICT
            device-level connections pin exit chips, so there is nothing to choose; Tier C checks against the pins.
          </Text>
        </Stack>
      </Grid>
      <BindingExamples />
    </Stack>
  );
}

function QuestionsTab() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>Known gaps</H2>
        <Table
          headers={["Gap", "Effect", "Handling"]}
          rows={[
            ["Harness core-count overrides are not in MGD", "Tier C could accept a layout runtime rejects", "parked for now"],
            ["Blaze uses only its first carve plan", "Tier C may accept where blaze's first cycle fails", "binding R first does not close it (the fill can change blaze's first cycle); question 8"],
            ["Offline link state", "offline answer can be optimistic", "runtime mapper re-checks"],
            ["Tile smaller than a host slice", "binding rule is stricter than per-rank tiling", "relax the binding rule for the tiling path if runtime supports it"],
            ["Witness placement differs from blaze's", "TT_PIPELINE_STAGE is not what blaze builds", "not consumed; align before it is"],
          ]}
        />
      </Stack>
      <Stack gap={8}>
        <H2>Open questions</H2>
        <Table
          headers={["#", "Question"]}
          rows={[
            ["1", "Adjacency parity: do candidate inter-mesh links (then R) plus routed intra-mesh links match get_forwarding_direction-confirmed neighbours of the bound links?"],
            ["2", "How many projections survive Tier B on heterogeneous and fork pipelines? Many survivors weaken Plan 1. Ship the optional \"find all\" run in the first version, or save the defaults until a sweep shows Plan 1 would cut many seatings?"],
            ["3", "Shared seat pairs that need different k: handle generally or reject until needed?"],
            ["4", "Seat-pair nogood API: when is a failure provably about one pair of seats?"],
            ["5", "Cost of unique_shapes off on large symmetric machines."],
            ["6", "Key name for the rank-binding env override (TT_PIPELINE_STAGE?)."],
            ["7", "Per-stage core counts: no MGD field for now; C2 computes blaze's defaults from the graph shape. Add one only if a model overrides them."],
            ["8", "Agreement with blaze's planner: blaze calls the same C++ footprint solve, blaze's planner learns to retry, or Tier C restricts itself to what blaze builds (14, 15)?"],
            ["9", "Dropped: no spread fill. Spare links after phase 0 keep today's order; the pipeline only needs R."],
            ["10", "Decided: do not wait for part 9. P5 reads the PSD; P16 (later) switches to the candidate table. Does P16 need to land before the first pipeline model ships?"],
            ["11", "Phase 0 failure before part 11: fail at control-plane init on every rank via the check after the broadcast (proposed), or add a cheap mapper-side check first?"],
            ["12", "Exit-node retry: is kMaxLayoutRetries = 4 right, and is the slice signature the right granularity?"],
            ["13", "C2 bound: conflict_cap is deterministic, but a capped solve reads as no carve and C4 excludes the seating. Accept that, or add a one-field stopped flag to MappingResult (the only solver change left)?"],
            ["14", "Stage 0 on one slice: restrict stage 0's positions in C1 while blaze's planner keeps that rule?"],
            ["15", "Stage smaller than a host slice: model sub-slice positions (first tile only, like blaze) or keep Tier B's reject?"],
            ["16", "RELAXED as soft edges: does the solver's soft-edge mode give fewest cuts first and blaze's tie-breaks, or is an objective needed?"],
            ["17", "C1 position filter (shape, core slots, stage 0 host I/O) must match the resolver's per-stage rules; stricter removes carves that would pass, looser only adds rejects."],
            ["18", "Demand inside the encoding: a per-mesh-pair cap on crossing edges if the demand check fires often."],
            ["19", "Resource completeness off the tight case: are carves with a few spare slices slow, and does a relaxed form help?"],
            ["20", "A recorded channel that is down: the PSD on rank 0 has no link state, but each rank's gather drops down links, so R can name one the pairing never sees. Fail on every rank (proposed) or rerun Tier C without it?"],
            ["21", "Broadcasting: R goes out with the chip info (every rank checks it); the rest of the pipeline mapping stays on rank 0 until another rank consumes it. With generate_mapping_locally_ every rank maps on its own allowlist, so R can differ: take rank 0's, or disallow pipelines there?"],
            ["22", "Total time per mapping: conflict cap x exit-node retries x rejected seatings can reach hours. Add an overall cap on the pipeline loop, or rely on the rejection cap?"],
            ["23", "Pipelines across parts: name the far end of a cross-part edge as (part index, local FabricNodeId), or return the pipeline mapping once per run?"],
            ["24", "One carve per seating: a seating is lost if its first carve fails but another would pass (101 of 101 first carves passed). Keep one, or allow a few more (exclude_mapping(carve), next()) before failing the seating?"],
          ]}
        />
      </Stack>
    </Stack>
  );
}

const SECTIONS = [
  { id: "overview", label: "Overview", body: OverviewTab },
  { id: "descriptor", label: "The MGD pipeline section", body: DescriptorTab },
  { id: "mapper", label: "Tier C, C1 and C2: graph and solve", body: MapperTab },
  { id: "binding", label: "C3: on success, emit and bind", body: BindingTab },
  { id: "sat", label: "C4: on failure, next solve", body: SatTab },
  { id: "plan", label: "Implementation plan", body: PlanTab },
  { id: "questions", label: "Gaps and open questions", body: QuestionsTab },
] as const;

export default function MgdPipelineDesign() {
  const dispatch = useCanvasAction();
  const t = useHostTheme();
  return (
    <Stack gap={28} style={{ padding: 24, maxWidth: 1240 }}>
      <Stack gap={8}>
        <H1>Pipeline-aware topology mapping</H1>
        <Text tone="secondary">
          Encode pipeline stages in the MGD so the topology mapper only commits mappings the pipeline builder can build on.
          Design for [tt-metal#52016](https://github.com/tenstorrent/tt-metal/issues/52016). Nothing here is implemented yet;
          blaze's builder (Tier D) is unchanged.
        </Text>
        <Row gap={8}>
          <Button
            variant="ghost"
            onClick={() => dispatch({ type: "openFile", path: "/data/rsong/tt-metal3/docs/scratch/mgd-pipeline-graph-design.md" })}
          >
            Open the full design doc
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
                <H1 style={{ margin: 0 }}>{s.label}</H1>
              </div>
              <Body />
            </Stack>
          </div>
        );
      })}
    </Stack>
  );
}
