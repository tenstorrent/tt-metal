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
        <SvgBox x={xs[1]} y={170} w={bw} h={46} title="Topology mapper" sub="carve + resolve per seating" kind="new" />
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
      why: "Same as all STRICT: an all-fabric cycle exists, so the planner cuts nothing to host.",
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
      why: "Adjacency is a1-a0, a0-b0, b0-b1: a path, not a cycle. Planned carve fails with no-cycle-of-length-4; per-rank tiling fails in the resolver (s3 > s0 has no link).",
    },
    relaxed: {
      ok: true,
      why: "plan_ring_placement lays s0..s3 along the fabric path a1 > a0 > b0 > b1 and moves only the loopback s3 > s0 onto host. Allowed: it is RELAXED and not stage 0's outgoing edge.",
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
    { title: "C1. build the physical pipeline graph", sub: "candidates, host slices, adjacency, carve paths -> slots  [P5 P6 P7 P8]", kind: "new" },
    { title: "C2. pipeline solve", sub: "resolver: pipeline graph onto a C1 graph; R within demand  [P4 P8 P9]", kind: "reuse" },
    { title: "C3. on success: emit + publish R", sub: "get_pipeline_required_links(); control plane binds R first  [P9 P12]", kind: "new" },
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
        {/* C2 fails on one graph: next C1 graph */}
        <path
          d={`M ${bx} ${y(3) + bh / 2} L ${bx - 24} ${y(3) + bh / 2} L ${bx - 24} ${y(2) + bh / 2} L ${bx} ${y(2) + bh / 2}`}
          fill="none"
          stroke={t.stroke.primary}
          strokeDasharray="4 3"
          markerEnd="url(#lf-arrow)"
        />
        <text x={bx - 30} y={y(3) + bh / 2 + 16} textAnchor="end" fontSize={10.5} fill={t.text.tertiary}>
          next carve path or cycle
        </text>
        {/* every graph fails: C4 */}
        <line x1={bx + bw} y1={y(3) + bh / 2} x2={c4x} y2={y(3) + bh / 2} stroke={t.category.red} strokeDasharray="4 3" markerEnd="url(#lf-red)" />
        <text x={bx + bw + 6} y={y(3) + bh / 2 - 6} fontSize={10.5} fill={t.category.red}>
          all fail
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
    { title: "Fill order", sub: "tier, then slice-pair spread", kind: "muted" },
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
  milestone: "M1 Format" | "M2 Shared code" | "M3 Tier C (STRICT)" | "M4 Pruning" | "M5 RELAXED (later)";
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
      "Add messages PipelineDescriptor, PipelineStage, PipelineEdge; field 6 on MeshGraphDescriptor. Every edge is STRICT: PipelineEdge has no policy field, and field 3 is reserved for it (P14).",
      "Add validate_pipeline_descriptors to static_validate, after validate_pinnings.",
      "Add accessors has_pipeline_descriptors() and pipeline_descriptors().",
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
+    reserved 3;              // edge policy (P14, later); every edge rides a fabric link
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
        // NOTE: no stage-topology check. In-degree, out-degree, number of roots, and whether a
        // branch is a chain are all accepted. Once the MGD identifies the router, this can
        // constrain the graph to blaze's shape (one root, chains, at most two branches).
    }
}`,
      },
    ],
    tests: [
      "Golden textprotos: GPT-OSS ring, llama_8b_pod (40 stages), joint 4x4 stage, dflash fork with both returns on fabric, two pipelines in one MGD.",
      "One negative textproto per rule (duplicate id, unknown stage, self-edge, rank-3 shape, two pipelines with one name). Two in-degree-0 stages, a stage with three outgoing edges, a branch stage with out-degree 2, and a dead-end branch all parse. An edge with policy: RELAXED fails to parse (no such field yet).",
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
    goal: "Reject pipelines impossible on every mapping of the MGD, before SAT runs. Never pick a split.",
    files: [
      "tt_metal/fabric/pipeline_feasibility.cpp (new)",
      "tt_metal/fabric/pipeline_feasibility.hpp (new)",
      "tt_metal/fabric/mesh_graph_descriptor.cpp",
    ],
    steps: [
      "Binding set per stage: meshes whose slice tiles the stage shape exactly and that fit it. The slice is device_topology / host_topology; host_topology.dims is the rank grid already on the descriptor (DeclaredTopology::host_dims). No stage is required to be one slice: the descriptor does not say which stage owns host I/O.",
      "Capacity in slices, per mesh class.",
      "Projection exists: depth-first stage -> mesh over every pipeline at once, stopping at the first survivor. Most constrained stage first, then breadth-first along edges; meshes holding a neighbour first. Rules: slices fit, every crossing rides a declared connection within its channel count, look-ahead on free slices and channels, and an empty mesh is used only if it is the lowest empty one among interchangeable meshes.",
      "Segment carve at each leaf, per mesh: uint64 slice masks, rectangles placed largest first then along edges, no overlap, every edge inside the mesh joins touching rectangles. Memoised on (mesh descriptor, sorted stage ids), failures included.",
      "No node budget: both searches end in pass or proved impossible, never UNDECIDED. Fail fast instead: an odd grid holds no ring, and a fully used mesh entered and left through fixed slices gets the exact O(1) Hamiltonian-path test (Itai-Papadimitriou-Szwarcfiter) before any search.",
      "Save PipelineMappingConstraints (binding sets; k_AB, min crossings per mesh, boundary slices per mesh pair), exposed as MeshGraphDescriptor::pipeline_mapping_constraints(). No projection list: any carve Tier C finds already passes Tier B's rules, so a stored list could never reject anything. The Plan 1 facts need every projection and carve, so they come from an optional \"find all\" run of the same searches with a smaller budget (10^5); skipped or over budget, they are the defaults k = 0 and every slice a boundary slice. init_from_parts reads them; it does not run the check.",
      "Run at the end of every MeshGraphDescriptor constructor, after populate(). Loading a part checks that part (before merge). merge() constructs a new descriptor, so the merged meshes and all pipelines are checked again (after merge).",
    ],
    code: [
      {
        path: "tt_metal/fabric/mesh_graph_descriptor.cpp  (proto constructor)",
        mode: "diff",
        language: "cpp",
        src: `
     proto_ = std::move(proto);
     populate();
+    // Tier B. host_topology already gives each mesh its rank grid, so this does not
+    // wait for MeshGraph. Every construction runs it: each part before merge, and
+    // the descriptor merge() returns.
+    if (proto_->pipeline_descriptors_size() > 0) {
+        pipeline_feasibility_ = std::make_unique<PipelineFeasibility>(*this);
+    }`,
      },
      {
        path: "tt_metal/fabric/pipeline_feasibility.hpp  (new)",
        mode: "new",
        language: "cpp",
        src: `
// MGD types only: the MGD library does not include topology_solver.hpp.
struct PipelineStageKey { std::string pipeline; std::string stage; auto operator<=>(const PipelineStageKey&) const = default; };

// Facts true for every surviving projection and every carve, so constraints built from them are safe.
struct PipelineMappingConstraints {
    std::map<PipelineStageKey, std::set<MeshId>> binding_sets;
    bool enumerated = false;                                            // false: "find all" skipped or over budget,
                                                                        // so k = 0 and every slice a boundary
    std::map<std::pair<MeshId, MeshId>, std::uint32_t> min_crossings;   // k_AB, min over projections
    std::map<MeshId, std::uint32_t> min_crossings_out;                  // per mesh, min over projections
    std::map<std::pair<MeshId, MeshId>, std::set<MeshHostRankId>> boundary_slices;   // union
};

class PipelineFeasibility {
public:
    explicit PipelineFeasibility(const MeshGraphDescriptor& descriptor);
    const PipelineMappingConstraints& mapping_constraints() const;

    // Facts that hold for every surviving projection (Plan 1 reads these).
    std::size_t min_crossings(MeshId a, MeshId b) const;            // k_AB; every edge counts
    const std::set<SliceKey>& boundary_slices(MeshId mesh) const;  // union over projections

    // Tier C exact check, filled in by P8.
    PipelineCheck check(const AssignedMeshes& seating, const TopologyMappingResult& full,
                        const tt::tt_metal::PhysicalSystemDescriptor& psd) const;

private:
    std::map<std::string, std::vector<MeshId>> binding_sets_;
    std::map<std::pair<MeshId, MeshId>, std::size_t> min_crossings_;
    std::map<MeshId, std::set<SliceKey>> boundary_slices_;
};`,
      },
    ],
    tests: [
      "subtorus_4x4_pipeline_8stage: ring 0..7..0 accepted; drop the 7-0 connection and it is rejected.",
      "llama_8b_pod: 8 x 4x2 + 32 x 1x2 accepted; 9 x 4x2 rejected on M0 capacity.",
      "mesh4x4_stage4x2: accepted, k(M0, M1) = 2 (interleaved splits cross 4 times; min is 2).",
      "mixed4 + mixed4_b in one descriptor: rejected when that descriptor is constructed (12 > 8). Two copies of ring4: rejected with 2 channels (4 crossings), accepted with 4.",
      "Two parts that each fit: each constructs. merge() constructs the combined descriptor and rejects it when the pipelines together do not fit.",
    ],
    behaviour: "An impossible pipeline fails when its MeshGraphDescriptor is constructed, before MeshGraph and before SAT.",
  },
  {
    id: "P4",
    title: "Resolver: structured failure + pure link core",
    milestone: "M2 Shared code",
    size: "M",
    deps: [],
    goal: "Let Tier C call the existing resolver without a control plane and learn why it failed.",
    files: ["tt_metal/fabric/pipeline_builder.cpp", "tt_metal/api/tt-metalium/experimental/fabric/pipeline_builder.hpp"],
    steps: [
      "Add try_resolve_graph_layout_with_connections returning a result or a PipelineLayoutFailure; the throwing version wraps it, message unchanged.",
      "Split discover_submesh_links into a pure core that takes a link query callback. Tier D passes the control-plane query it uses today; Tier C passes candidate links.",
    ],
    code: [
      {
        path: "tt_metal/api/tt-metalium/experimental/fabric/pipeline_builder.hpp",
        mode: "diff",
        language: "cpp",
        src: `
 using DirectLinks = std::map<ConnectionKey, std::vector<LinkPair>>;

+struct PipelineLayoutFailure {
+    enum class Kind { NoSubmeshAssignment, CoreCapacity, NoLinkForEdge, InvalidGraph };
+    Kind kind;
+    std::optional<std::string> stage;                               // failing node, if known
+    std::optional<std::pair<std::string, std::string>> edge;        // failing edge, if known
+    std::string message;                                            // today's exception text
+};
+
+std::variant<GraphLayoutResult, PipelineLayoutFailure> try_resolve_graph_layout_with_connections(
+    const std::vector<std::string>& nodes,
+    const std::vector<EdgeInputTuple>& edges,
+    const std::vector<std::vector<ChipTuple>>& submesh_chips,
+    const std::map<std::string, uint32_t>& node_chip_counts,
+    const std::map<std::string, uint32_t>& node_pipeline_core_counts,
+    uint32_t pipeline_core_count,
+    const DirectLinks* direct_links);
+
+// Pure core of discover_submesh_links: is chip a linked to chip b, and in which direction.
+using ChipLinkQuery = std::function<std::optional<RoutingDirection>(FabricNodeId a, FabricNodeId b)>;
+DirectLinks discover_submesh_links(const std::vector<std::vector<ChipTuple>>& submesh_chips, const ChipLinkQuery& linked);
+
 // CPU tests can supply discovered links; nullptr uses the control plane.
 GraphLayoutResult resolve_graph_layout_with_connections(`,
      },
    ],
    tests: [
      "Existing resolver tests unchanged (the throwing API keeps its messages).",
      "New CPU tests: each failure kind on a crafted DirectLinks map.",
      "discover_submesh_links with a control-plane callback equals today's output on recorded topologies.",
    ],
    behaviour: "None for blaze.",
  },
  {
    id: "P5",
    title: "Carve planner C++ port",
    milestone: "M2 Shared code",
    size: "L",
    deps: [],
    goal: "Second copy of blaze's pure planning core in tt-metal; same verdicts on the same inputs.",
    files: [
      "tt_metal/fabric/carve_planner.cpp (new)",
      "tt_metal/api/tt-metalium/experimental/fabric/carve_planner.hpp (new)",
      "tests/tt_metal/tt_fabric/pipeline/carve_goldens/*.json (new)",
      "tt-blaze: scripts/dump_carve_goldens.py (new, test-only)",
    ],
    steps: [
      "Port HostSlice, CarveSlot, plan_ring_carve, plan_fork_carve, fork_structure, validate_carve_plan, keeping blaze's sort keys.",
      "One deliberate difference: fork_structure and plan_fork_carve accept a dead-end branch and plan it as an open path, like a host return. Blaze rejects it, so its goldens are hand-written, not dumped.",
      "Expose a cycle iterator: Tier C keeps trying cycles until one resolves; blaze stops at the first.",
      "budget-exhausted returns Unknown, which Tier C treats as accept.",
      "Golden parity: a blaze script dumps (slices, adjacency, shapes) -> verdict + phase; a gtest replays them against the port.",
    ],
    code: [
      {
        path: "tt_metal/api/tt-metalium/experimental/fabric/carve_planner.hpp  (new)",
        mode: "new",
        language: "cpp",
        src: `
namespace tt::tt_fabric::detail {

struct HostSlice {
    uint32_t mesh_id, row0, col0, nrows, ncols;
    auto key() const { return std::tie(mesh_id, row0, col0); }   // geometry key, as in blaze
};
struct CarveSlot { uint32_t mesh_id; std::array<uint32_t, 2> shape, offset; };
using SliceAdjacency = std::set<std::pair<HostSlice, HostSlice>>;  // unordered pairs

enum class CarvePhase { InputInfeasible, NoCycleOfLength, NoAlignment, BudgetExhausted };
struct CarveFailure { CarvePhase phase; std::string reason; };

// Calls on_plan for each aligned plan in blaze's order (fewest mesh crossings, then slice keys)
// until it returns true. Returns the failure when no plan was accepted.
std::optional<CarveFailure> for_each_ring_carve(
    const std::vector<HostSlice>& slices, const SliceAdjacency& adjacency,
    const std::vector<std::array<uint32_t, 2>>& stage_shapes,
    const std::map<uint32_t, std::array<uint32_t, 2>>& mesh_shapes,
    const std::function<bool(const std::vector<CarveSlot>&)>& on_plan);

std::optional<CarveFailure> for_each_fork_carve(/* same shape, router + branches */);
ForkStructure fork_structure(const proto::PipelineDescriptor& pipeline);

}  // namespace tt::tt_fabric::detail`,
      },
      {
        path: "tests/tt_metal/tt_fabric/pipeline/test_carve_planner_parity.cpp  (new)",
        mode: "new",
        language: "cpp",
        src: `
TEST_P(CarvePlannerParity, SameVerdictAsBlaze) {
    const CarveGolden g = load_golden(GetParam());            // dumped by tt-blaze dump_carve_goldens.py
    std::optional<std::vector<CarveSlot>> first;
    const auto failure = detail::for_each_ring_carve(
        g.slices, g.adjacency, g.stage_shapes, g.mesh_shapes, [&](const auto& plan) {
            first = plan;
            return true;                                      // blaze semantics: stop at the first plan
        });
    EXPECT_EQ(failure.has_value(), !g.accepted);
    if (failure) { EXPECT_EQ(failure->phase, g.phase); }
    if (first && g.plan) { EXPECT_EQ(*first, *g.plan); }      // optional: same first plan
}`,
      },
    ],
    tests: [
      "Goldens: ring, heterogeneous, joint stage, fork with fabric and host returns, every failure phase.",
      "Goldens on one galaxy with 4x1 rank columns (carver example F): 4x2 stages on odd columns, a 4x2 group on the C7-C0 wrap rejected, a 6-slice ring rejected with achievable [8].",
      "Determinism: same input twice gives the same plan sequence.",
    ],
    behaviour: "None. Library only.",
  },
  {
    id: "P6",
    title: "Slot builders",
    milestone: "M2 Shared code",
    size: "S",
    deps: [],
    goal: "Pure C++ for row-major tiling, a rank's slot pick, and joint-stage collapse, shared with MeshDevice.",
    files: [
      "tt_metal/fabric/carve_planner.cpp",
      "tt_metal/distributed/mesh_device.cpp",
    ],
    steps: [
      "tile_row_major(mesh_shape, tile_shape): offsets in create_submeshes order, same divisibility TT_FATAL.",
      "MeshDeviceImpl::create_submeshes calls it, so the order cannot drift.",
      "pick_rank_slot: first tile whose (0,0) chip is local, else the first tile with any local chip.",
      "collapse_joint_stages: identical chip sets merge, ordered by first rank.",
      "Tile-shape candidates per mesh: stage shapes that divide the mesh, plus the slice shape.",
    ],
    code: [
      {
        path: "tt_metal/distributed/mesh_device.cpp  (MeshDeviceImpl::create_submeshes)",
        mode: "diff",
        language: "cpp",
        src: `
 std::vector<std::shared_ptr<MeshDevice>> MeshDeviceImpl::create_submeshes(
     const std::shared_ptr<MeshDevice>& parent_mesh, const MeshShape& submesh_shape) {
-    // Calculate how many submeshes fit in each dimension.
-    ttsl::SmallVector<uint32_t> steps;
-    for (size_t dim = 0; dim < shape().dims(); dim++) {
-        TT_FATAL(shape()[dim] % submesh_shape[dim] == 0, ...);
-        steps.push_back(shape()[dim] / submesh_shape[dim]);
-    }
-    std::vector<std::shared_ptr<MeshDevice>> submeshes;
-    for (const auto& step_position : MeshCoordinateRange(MeshShape(steps))) { ... }
+    // Shared with the topology mapper's tiling path, so both stamp tiles in the same order.
+    std::vector<std::shared_ptr<MeshDevice>> submeshes;
+    for (const MeshCoordinate& offset : tt::tt_fabric::detail::tile_row_major(shape(), submesh_shape)) {
+        submeshes.push_back(create_submesh(parent_mesh, submesh_shape, offset));
+    }
     return submeshes;
 }`,
      },
    ],
    tests: [
      "tile_row_major matches the old create_submeshes order for every shape in the blaze sweep.",
      "pick_rank_slot on a 2x2 tile over a 4x2 slice keeps the first tile only (edge case from the doc).",
    ],
    behaviour: "None (refactor).",
  },
  {
    id: "P7",
    title: "Inter-mesh candidates from the policy plan (part 9)",
    milestone: "M2 Shared code",
    size: "S",
    deps: [],
    goal: "C1 and the control plane read one candidate table: the inter-mesh policy plan's collect_inter_mesh_candidates. This PR wires it into the mapper; if part 9 has not landed, it adds an interim pure helper with the same signature.",
    files: ["tt_metal/fabric/inter_mesh_candidates.cpp (policy plan, part 9)", "tt_metal/fabric/topology_mapper_utils.cpp"],
    steps: [
      "Depends on the inter-mesh policy plan's part 9 (candidate table) and its demand table (get_inter_mesh_channel_demand). No own gather in this design.",
      "Make collect_inter_mesh_candidates callable per seating from MultiMeshSolutionEnumerator::next, with the same InterMeshCandidateOptions as the final call.",
      "Interim only, if part 9 is not in yet: factor gather_intermesh_cables_for_exit_nodes into a pure helper with collect_inter_mesh_candidates's signature, so the later swap is mechanical.",
    ],
    code: [
      {
        path: "tt-metalium/experimental/fabric/inter_mesh_candidates.hpp  (policy plan, part 9)",
        mode: "plain",
        language: "cpp",
        src: `
// Pure: no cluster, no MPI. Links up, allowed cable types, mapped meshes, requested exit nodes.
InterMeshCandidates collect_inter_mesh_candidates(
    const MeshGraph& mesh_graph,
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const std::map<FabricNodeId, tt::tt_metal::AsicID>& fabric_node_to_asic,
    const InterMeshCandidateOptions& options);`,
      },
    ],
    tests: [
      "Per-seating call on a recorded PSD returns the same table as the final call for the emitted seating.",
      "Interim helper only: AnnotatedIntermeshConnections byte-identical before and after on SC16 revAB/revC and single galaxy.",
    ],
    behaviour: "None.",
  },
  {
    id: "P8",
    title: "Tier C exact check (STRICT): C1, C2 and seating exclusion (C4)",
    milestone: "M3 Tier C (STRICT)",
    size: "L",
    deps: ["P3", "P4", "P5", "P6", "P7"],
    goal: "Check every candidate seating with the builder's own code; reject only when every carve path fails.",
    files: [
      "tt_metal/fabric/pipeline_feasibility.cpp",
      "tt_metal/fabric/topology_mapper_utils.cpp",
      "tools/scaleout/src/generate_rank_bindings.cpp",
    ],
    steps: [
      "PipelineFeasibility::check: candidates -> slices -> carve paths -> resolver.",
      "allowed_paths follows blaze's multi_mesh rule: one mesh with every stage on one rank gets single-mesh tiling; one mesh with a joint stage (e.g. 4x2 stages on 4x1 rank columns), or several meshes, gets the planned carve, then per-rank tiling.",
      "Hook into MultiMeshSolutionEnumerator::next() after intra-mesh success; on reject call exclude_mapping(seating) (seat set), never add_forbidden_constraint.",
      "Turn unique_shapes off when a pipeline is present (runtime overloads and --all-solutions).",
      "failure_stage pipeline_carve / pipeline_layout; cap rejections per run (kMaxPipelineRejections, proposed 256, overridable for sweeps) and report the last reason.",
      "Feedback is whole-seating exclusion only; the seat-pair clause and per-seat forbid wait for attributed failures (P13).",
      "Record pipeline rejections per run in the placement stats (seatings tried before the first accept), so the sweep and CI show whether P10 is needed.",
      "Several pipelines: place each on the slices the earlier ones left free, trying every order of the pipelines; resolve each as its own graph; accept only if all fit.",
    ],
    code: [
      {
        path: "tt_metal/fabric/topology_mapper_utils.cpp  (MultiMeshSolutionEnumerator::next)",
        mode: "diff",
        language: "cpp",
        src: `
         if (placement_stats_ != nullptr) {
             apply_placement_stats(full.stats, *placement_stats_);
         }
         if (full.success) {
+            if (pipeline_ != nullptr) {
+                PipelineCheck check = pipeline_->check(seating, full, *physical_system_descriptor_);
+                if (!check.accepted) {
+                    full.success = false;
+                    full.stats.failure_stage = check.failure_stage;   // pipeline_carve | pipeline_layout | pipeline_binding
+                    full.error_message = check.reason;
+                    last_failed_ = full;
+                    // The failure is about this set of seats, not about one mesh on one seat.
+                    if (++pipeline_rejections_ > kMaxPipelineRejections ||
+                        !placement_session_->exclude_mapping(seating)) {
+                        return give_up();
+                    }
+                    continue;
+                }
+                full.pipeline_plan = std::move(check.plan);
+            }
             ++emitted_;
             return split_mapping_to_local_parts(full, per_part_local_to_global_mesh_ids_);
         }`,
      },
      {
        path: "tt_metal/fabric/pipeline_feasibility.cpp  (PipelineFeasibility::check)",
        mode: "new",
        language: "cpp",
        src: `
PipelineCheck PipelineFeasibility::check(
    const AssignedMeshes& seating, const TopologyMappingResult& full, const PhysicalSystemDescriptor& psd) const {
    const auto cands  = collect_inter_mesh_candidates(mesh_graph_, psd, full.fabric_node_to_asic, candidate_options_);  // P7
    const auto slices = host_slices(mesh_graph_);                       // get_coord_range per (mesh, host rank)
    const auto links  = routed_links(mesh_graph_, full, cands);         // intra-mesh topology + candidates

    PipelineCheck last = PipelineCheck::reject("pipeline_carve", "no carve path applies");
    // A joint stage takes the planned carve even on one mesh, as blaze's multi_mesh flag does.
    for (const CarvePath path : allowed_paths(pipeline_, mesh_graph_)) {   // planned | tiling | single mesh
        auto attempt = try_path(path, slices, links);                       // slots -> subs -> resolver (P4-P6)
        if (!attempt.accepted) { last = attempt; continue; }                // reject only if every path fails
        attempt.required_links = crossing_links(attempt.layout, links);     // R, used by P9
        return attempt;
    }
    return last;
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
    title: "Emit and bind the cables the layout uses (C2 demand check, C3)",
    milestone: "M3 Tier C (STRICT)",
    size: "M",
    deps: ["P7", "P8"],
    goal: "Record the links the accepted layout crosses on (R), check they fit each connection's demand, and have the control plane bind them first.",
    files: [
      "tt_metal/api/tt-metalium/experimental/fabric/pipeline_required_links.hpp (new)",
      "tt_metal/fabric/pipeline_feasibility.cpp",
      "tt_metal/fabric/topology_mapper.cpp",
      "tt_metal/fabric/control_plane.cpp",
    ],
    steps: [
      "R from the layout's resolved_edges: one (exit chip, entry chip) link per crossing edge, keyed by mesh pair; stored on the TopologyMapper as get_pipeline_required_links().",
      "Budget check in check(): links in R per pair <= strict + relaxed from get_inter_mesh_channel_demand; over budget fails the path with pipeline_binding.",
      "PortDescriptor gains required (next to the policy plan's tier); to_port_descriptor_table sets it.",
      "Phase 0 in pair_logical_intermesh_ports: place every required link (Z rule of its pair, best-tier channel) before phase 1a, counted against the budget; failure is a TT_FATAL naming the link and pipeline edge.",
      "Fill after phase 0: (tier, slice-pair round-robin, chip id) with a pipeline; the policy plan's (tier, chip id) without.",
      "Single host: phase 0 in generate_intermesh_connections_on_local_host, or a warning until it lands.",
      "No rerun over the bound links: once R is bound, the witness layout is valid over the bound set.",
    ],
    code: [
      {
        path: "tt_metal/fabric/pipeline_feasibility.cpp  (inside check, after a path succeeds)",
        mode: "diff",
        language: "cpp",
        src: `
         attempt.required_links = crossing_links(attempt.layout, links);     // R
+        if (auto over = first_pair_over_budget(attempt.required_links, mesh_graph_)) {   // strict + relaxed
+            last = PipelineCheck::reject("pipeline_binding", over->message());
+            continue;                                                        // try the next carve path
+        }
         return attempt;`,
      },
      {
        path: "tt_metal/fabric/control_plane.cpp  (pair_logical_intermesh_ports)",
        mode: "diff",
        language: "cpp",
        src: `
+    // Phase 0: links the pipeline layout depends on go first, so the fill cannot strand them.
+    for (auto& [boundary, link] : order) {
+        if (!link->has_required_channel()) {
+            continue;
+        }
+        const bool placed = has_budget_after_trim(boundary, *link) &&
+                            (is_marked_z(boundary) ? place_link(*link, /*want_z=*/true)
+                                                   : (place_link(*link, /*want_z=*/false) || place_link(*link, /*want_z=*/true)));
+        TT_FATAL(placed, "pipeline_binding: cannot bind required link {} -> {} ({})",
+                 link->src_node, link->dst_node, link->required_by);
+        account_placed(boundary, *link);
+    }
+
     // Phase 1a: assign_z (marked) boundaries claim the Z lane; they are excluded from the NESW phases below.
     for (auto& [boundary, link] : order) {
         if (!is_marked_z(boundary) || link->placed) {`,
      },
      {
        path: "tt_metal/fabric/control_plane.cpp  (generate_intermesh_connectivity, multi-host branch from the policy plan)",
        mode: "diff",
        language: "cpp",
        src: `
     const auto& candidates = topology_mapper_->get_inter_mesh_candidates();
     check_one_dst_chip_per_src_chip(candidates);
-    auto port_descriptors = candidates.to_port_descriptor_table();
+    auto port_descriptors = to_port_descriptor_table(candidates, topology_mapper_->get_pipeline_required_links());
     paired = pair_logical_intermesh_ports(port_descriptors);`,
      },
    ],
    tests: [
      "RELAXED count 2 with four candidates where the lowest chip ids share a slice pair: today Case B, after P9 Case A.",
      "count 1 with a ring across two meshes: R needs 2, rejected with pipeline_binding.",
      "R empty (no pipeline): AnnotatedIntermeshConnections byte-identical to the policy plan's output.",
      "Multi-host: R identical on every rank; validate_requested_intermesh_connections passes everywhere.",
      "Phase 0 cannot place a link (exit chip already facing a third mesh): init fails naming the link and stage edge.",
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
      "tt_metal/api/tt-metalium/experimental/fabric/topology_solver.hpp",
      "tt_metal/fabric/topology_solver_sat.cpp",
      "tt_metal/fabric/topology_mapper_utils.cpp",
    ],
    steps: [
      "On a C2 failure, record sig(m) for each mesh on a crossed boundary: cabled exit ASIC -> host slice of the fabric node on it.",
      "Skip when hard rank bindings or MGD pins fix every ASIC in sig(m) under its slice.",
      "MappingConstraints::add_forbidden_joint_placement(groups): indicator y(a, S) <-> OR x[f, a] over S's fabric nodes; one clause OR NOT y.",
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
                 PipelineCheck check = pipeline_->check(seating, full, *physical_system_descriptor_);
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
+                    check = pipeline_->check(seating, full, *physical_system_descriptor_);
+                }
                 if (!check.accepted) {`,
      },
    ],
    tests: [
      "A seating whose first layout puts both cable ends under one slice and a second layout splits them: accepted after one retry (host-to-rank free).",
      "The same seating with every host rank bound: no retry, excluded as in P8.",
      "Retry cap: a seating with no working layout is excluded after kMaxLayoutRetries.",
      "SAT and DFS backends agree on add_forbidden_joint_placement on random small instances.",
    ],
    behaviour: "Some seatings rejected by P8 are accepted with another layout; offline generate_rank_bindings finds more mappings.",
  },
  {
    id: "P10",
    title: "Pruning (later): exit-node and seat constraints before the first solve (C4)",
    milestone: "M4 Pruning",
    size: "M",
    deps: ["P3", "P8"],
    goal: "Later, optional speed-up, added only if P8's measured rejection counts call for it. Turn Tier B's PipelineMappingConstraints into the MappingConstraints the mapper already solves with, and drop seat pairs that cannot carry the pipeline under any split or layout. Start with the footprint-level k_AB seat-graph edge drop.",
    files: [
      "tt_metal/fabric/pipeline_mapping_constraints.cpp (new)",
      "tt_metal/fabric/pipeline_mapping_constraints.hpp (new)",
      "tt_metal/fabric/physical_grouping_descriptor_matching.cpp",
      "tt_metal/fabric/topology_mapper_utils.cpp",
    ],
    steps: [
      "stage_to_mesh_constraints(pmc) -> MappingConstraints<PipelineStageKey, MeshId>: each stage required onto its binding set. A report, not a solve (many stages share a mesh).",
      "apply_pipeline_seat_constraints(pmc, pools, psd, c) on MappingConstraints<MeshId, const Candidate*>, called in build_sat_placement_constraints next to apply_extra_constraints: forbid seats of mesh m with fewer than min_crossings_out(m) cables leaving the footprint.",
      "apply_pipeline_exit_constraints(pmc, mesh_graph, m, n, ...) on MappingConstraints<FabricNodeId, PhysicalNodeId>, called in add_exit_node_constraints after the existing one: at least ceil(k(m, n) / links per exit ASIC) of the chips in boundary_slices(m, n) on ASICs cabled toward n's seat.",
      "Seat graph: for mesh pairs with k_AB > 0, drop the edge between two seats with fewer than k_AB candidate links between their footprints.",
      "Slice-level form (bipartite matching on distinct boundary slices) only where hard pinnings fix the layout.",
      "Single seat: remove only when it fails under every projection and layout.",
      "Shared seat pairs with different k: fall back to the exact check for those pairs.",
      "k_AB counts every crossing (P3), because every edge is STRICT. P14 changes it to skip RELAXED crossings."
    ],
    code: [
      {
        path: "tt_metal/fabric/physical_grouping_descriptor_matching.cpp  (build_sat_placement_seat_graph)",
        mode: "new",
        language: "cpp",
        src: `
// pipeline_mapping_constraints.cpp: seat level, a unary forbid per mesh.
bool apply_pipeline_seat_constraints(const PipelineMappingConstraints& pmc, const CandidatePools& pools,
                                     const PhysicalSystemDescriptor& psd,
                                     MappingConstraints<MeshId, const Candidate*>& constraints) {
    for (const auto& [mesh, k] : pmc.min_crossings_out) {
        std::set<const Candidate*> too_few;
        for (const Candidate& seat : pools.at(mesh)->candidates()) {
            if (cables_leaving(seat, psd) < k) { too_few.insert(&seat); }
        }
        if (!too_few.empty() && !constraints.add_forbidden_constraint(mesh, too_few)) { return false; }
    }
    return true;
}

// pipeline_mapping_constraints.cpp: intra-mesh level, inside add_exit_node_constraints.
std::optional<std::string> apply_pipeline_exit_constraints(
    const PipelineMappingConstraints& pmc, const MeshGraph& mesh_graph, MeshId m, MeshId n,
    const std::set<PhysicalNodeId>& exit_asics_toward_n, std::uint32_t links_per_exit_asic,
    MappingConstraints<FabricNodeId, PhysicalNodeId>& constraints) {
    const auto k = pmc.min_crossings.contains({m, n}) ? pmc.min_crossings.at({m, n}) : 0u;
    if (k == 0) { return std::nullopt; }
    std::set<FabricNodeId> boundary_chips;
    for (MeshHostRankId rank : pmc.boundary_slices.at({m, n})) {
        for (const auto& coord : mesh_graph.get_coord_range(m, rank)) { boundary_chips.insert(chip_at(mesh_graph, m, coord)); }
    }
    const std::uint32_t min_count = (k + links_per_exit_asic - 1) / links_per_exit_asic;
    if (!constraints.add_cardinality_constraint(boundary_chips, exit_asics_toward_n, min_count)) {
        return fmt::format("pipeline needs {} crossing(s) from mesh {}'s boundary slices toward mesh {}", k, m.get(), n.get());
    }
    return std::nullopt;
}

// Plan 1, footprint level. Sound for every projection: k_AB is the minimum over all of them.
if (pipeline != nullptr) {
    const std::size_t k = pipeline->min_crossings(mesh_a, mesh_b);
    if (k > 0 && candidate_links_between(seat_a.asics, seat_b.asics, psd) < k) {
        continue;   // no seat-graph edge: this pair cannot carry the pipeline's crossings
    }
}`,
      },
    ],
    tests: [
      "Pruning never removes a seating P8 would accept (property test over the sweep MGDs), for the seat, exit and seat-graph constraints alike.",
      "pod8: M0's exit helper narrows exit chips toward M1 to A0 and A3 (16 of 32), k = 2; with M0's layout pinned so that its cables land on A1 and A2, M0's intra-mesh solve is infeasible.",
      "A \"find all\" run over its budget (or skipped) gives k = 0 and adds no constraint.",
      "Seatings tried before the first accept drop on mesh4x4_stage4x2 and superpod cases.",
    ],
    behaviour: "Faster; same accepted set.",
  },
  {
    id: "P11",
    title: "RELAXED edges: host cuts in Tier C",
    milestone: "M5 RELAXED (later)",
    size: "L",
    deps: ["P5", "P8", "P14"],
    goal: "After P14 lets the format say RELAXED. Plan host cuts on the RELAXED edges, then carve or uniform-shape tiling, then one resolve per fabric chain. Until this lands, P14 should keep Tier C rejecting a pipeline with a RELAXED edge rather than accepting it unchecked.",
    files: [
      "tt_metal/fabric/carve_planner.cpp",
      "tt_metal/fabric/pipeline_feasibility.cpp",
      "tt-blaze: pipeline_builder/graph.py, host_edge_planner.py, blaze/models/cli_options.py",
    ],
    steps: [
      "Port plan_ring_placement and plan_fork_placement (host_edge_planner.py), taking the set of edges they may cut = the RELAXED edges; ForkShapeUnsupported falls back to the all-STRICT paths.",
      "Carry the planned cuts over host; carve from the plan or tile by the uniform stage shape.",
      "Resolve each fabric chain separately, with blaze's pool rule and greedy order.",
      "Remove P14's 'reject RELAXED in Tier C' guard.",
      "tt-blaze, same change: Edge gets a policy (STRICT | RELAXED); _group_walk skips alignments that cut a STRICT edge instead of only '0 in cuts'; the fork planner cuts only RELAXED returns; --pipeline-builder flexible becomes a shortcut that marks every edge RELAXED.",
    ],
    code: [
      {
        path: "tt_metal/fabric/pipeline_feasibility.cpp  (try_path, host cuts)",
        mode: "new",
        language: "cpp",
        src: `
case CarvePath::HostCuts: {
    const auto cuttable = pipeline_->relaxed_edges();   // the planners still skip s0's outgoing and router -> branch edges
    auto plan = pipeline_->is_fork() ? detail::plan_fork_placement(slices, adjacency, cuttable, ...)
                                      : detail::plan_ring_placement(slices, adjacency, stage_shapes, cuttable);
    if (plan.fork_shape_unsupported) { return try_strict_paths(slices, links); }   // blaze resolves as declared
    if (!plan.ok) { return PipelineCheck::reject("pipeline_carve", plan.reason); }  // ring: blaze raises too
    auto subs = plan.carve ? submeshes_from_slots(*plan.carve)
                           : tile_uniform(stage_shape_if_uniform());                // blaze sets submesh_shape
    return resolve_each_chain(fabric_chains(edges_minus(plan.host_edges)), subs, links);
}`,
      },
    ],
    tests: [
      "Case B and Case C accepted with s3 -> s0 RELAXED, loopback on host; still rejected with every edge STRICT.",
      "mixed4 on (P, Q) with s3 -> s0 RELAXED: cover 1 skipped (it cuts STRICT s1 -> s2), cover 2 accepted.",
      "Golden parity against host_edge_planner for rings and forks.",
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
      "For each rank: its mesh and host rank give its slice, then its slot, then its stage id (or idle).",
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
    title: "Later: seat-pair nogoods + spread fill",
    milestone: "M4 Pruning",
    size: "M",
    deps: ["P8", "P9"],
    goal: "Stronger feedback to SAT, and spare bound links that add slice adjacency.",
    files: ["tt_metal/fabric/topology_solver_sat.cpp", "tt_metal/fabric/intermesh_binding.cpp"],
    steps: [
      "New clause 'not (any mesh on seat_a and any mesh on seat_b)', keyed on seat identity including orientation.",
      "Use it only when a failure is proven for every carve over that pair.",
      "Fill order round-robin over slice pairs, only with a pipeline descriptor (keeps goldens).",
    ],
    code: [],
    tests: ["Seat-pair clause blocks exactly the seatings that use both seats.", "Spread fill: bound links cover more distinct slice pairs."],
    behaviour: "Fewer wasted seatings; with a pipeline, spare links spread.",
  },
  {
    id: "P14",
    title: "RELAXED edges: schema + Tier B + Plan 1",
    milestone: "M5 RELAXED (later)",
    size: "M",
    deps: ["P1", "P3", "P10"],
    goal: "Let a pipeline edge say RELAXED (may go over host MPI). Descriptor-only tiers stop counting RELAXED edges; Tier C still rejects them until P11.",
    files: [
      "tt_metal/fabric/protobuf/mesh_graph_descriptor.proto",
      "tt_metal/fabric/pipeline_feasibility.cpp",
      "tt_metal/fabric/topology_mapper_utils.cpp",
    ],
    steps: [
      "Proto: add enum EdgePolicy { STRICT = 0; RELAXED = 1; } and un-reserve PipelineEdge field 3 as EdgePolicy policy = 3. STRICT is 0, so every first-version file keeps its meaning.",
      "Tier A: nothing new; RELAXED is allowed on any edge.",
      "Tier B: skip RELAXED edges in the crossing rule (no cap, no channel) and in the carve adjacency rule; a RELAXED return counts 1 toward a router's degree.",
      "Plan 1: k_AB and boundary slices count STRICT crossings only; every edge RELAXED gives k_AB = 0.",
      "Tier C: add has_relaxed_edge(); reject such a pipeline with failure_stage pipeline_relaxed_unsupported until P11.",
    ],
    code: [
      {
        path: "tt_metal/fabric/protobuf/mesh_graph_descriptor.proto",
        mode: "diff",
        language: "proto",
        src: `
+enum EdgePolicy {
+    STRICT = 0;     // must ride a fabric link
+    RELAXED = 1;    // fabric if the placement allows; otherwise host MPI (D2H -> MPI -> H2D)
+}
+
 message PipelineEdge {
     string src = 1;
     string dst = 2;
-    reserved 3;
+    EdgePolicy policy = 3;   // default STRICT; RELAXED allowed on any edge
 }`,
      },
    ],
    tests: [
      "Every P1-P10 golden parses unchanged and gives the same Tier B result.",
      "mixed4 with no MGD connection: s1 -> s2 and s3 -> s0 RELAXED passes Tier B; only s3 -> s0 RELAXED still rejects.",
      "mesh4x4_stage4x2 with s3 -> s0 RELAXED: k(M0, M1) = 1. Every edge RELAXED: k_AB = 0, no crossing-based rejects.",
      "A pipeline with a RELAXED edge is rejected by Tier C with pipeline_relaxed_unsupported (until P11).",
    ],
    behaviour: "Files can say RELAXED; Tiers A and B accept more; Tier C refuses RELAXED until P11.",
  },
];

const MILESTONES: Array<{ id: Pr["milestone"]; text: string }> = [
  { id: "M1 Format", text: "The MGD can carry a pipeline, survives merge, and impossible pipelines fail at load." },
  { id: "M2 Shared code", text: "Libraries Tier C needs, each landing with no behaviour change." },
  { id: "M3 Tier C (STRICT)", text: "The mapper only emits seatings the pipeline builds on, and binds the cables it relied on." },
  { id: "M4 Pruning", text: "Speed-ups. Not needed for correctness." },
  { id: "M5 RELAXED (later)", text: "Not in the first version. Adds edge policy to the schema, then host cuts in Tier C. M1-M4 are complete without it: every edge is STRICT." },
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
        <H2>Fourteen PRs: STRICT in M1-M4, RELAXED later in M5</H2>
        <Text tone="secondary">
          The first version is STRICT only: every pipeline edge rides fabric, and M1-M4 need nothing from M5. Arrows are
          hard dependencies. M2 PRs are independent of each other and of M1, so they can land in parallel. Every PR
          is written out below with its steps, code and tests; click a box to jump to it.
        </Text>
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
          ["mixed4 + edge s1 > s1", "reject", "self-edge"],
          ["mixed4 with s0 > s1 repeated", "pass", "two channels on that hop; the builder reads one link per edge"],
          ["dflash: router > target0..37 > router, router > draft0..10 > router", "pass", "one router, two chain branches, both return to it on fabric"],
          ["s0 > s1, s0 > s2, s1 > s0; s2 has no out-edge", "pass", "a fork whose branch s2 dead-ends, which is allowed; blaze rejects it today (TODO)"],
          ["a router with three branches", "pass", "NOTE: Tier A does not limit stage topology. Blaze's fork runtime still builds exactly two"],
          ["a branch stage with out-degree 2", "pass", "NOTE: Tier A does not limit stage topology. Once the MGD names the router, only that stage may split"],
          ["two stages with in-degree 0", "pass", "NOTE: Tier A does not limit stage topology. Once the MGD names the router, a fork has exactly one"],
          ["mixed4 with policy: RELAXED on s3 > s0", "reject", "no policy field in the first version: does not parse. RELAXED (later): passes"],
          ["RELAXED (later): mixed4 with s0 > s1 RELAXED", "pass", "allowed; Tier C's planner still never cuts stage 0's outgoing edge"],
          ["RELAXED (later): dflash with router > draft0 RELAXED", "pass", "allowed; Tier C still needs a fabric link from the router to a branch head"],
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
        <CodeCard path="PipelineFeasibility: the search" mode="plain" language="text" src={TIER_B_SEARCH} />
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
          ["two copies of ring4, 2 channels", "3, 4", "no 4-cycle on a line of slices, so each ring crosses twice: 4 > 2"],
        ]}
      />
      <H3>Output: mapping constraints for Tier C</H3>
      <Text tone="secondary">
        Tier B saves what it found as PipelineMappingConstraints on the descriptor: binding sets, k_AB and the minimum
        crossings per mesh, and the boundary slices per mesh pair. No projection list is stored. Every fact holds for every
        surviving projection and carve, so constraints built from it never drop a buildable seating. The MGD library stores plain MGD
        types; helpers beside the topology mapper turn them into the MappingConstraints objects it already solves with (P10).
      </Text>
      <Table
        headers={["Helper", "Fills", "Adds"]}
        rows={[
          ["stage_to_mesh_constraints", "MappingConstraints<PipelineStageKey, MeshId>", "required: stage to its binding set (report, not a solve)"],
          ["apply_pipeline_seat_constraints", "MappingConstraints<MeshId, const Candidate*> (SAT seating)", "forbidden: seats of m with fewer than min_crossings_out(m) cables leaving the footprint"],
          ["apply_pipeline_exit_constraints", "MappingConstraints<FabricNodeId, PhysicalNodeId> (intra-mesh)", "cardinality: at least k(m, n) cables, by links per ASIC, from chips in boundary_slices(m, n) toward n"],
          ["(seat graph, Plan 1)", "build_sat_placement_seat_graph", "pair condition, so not a MappingConstraints entry: no edge between seats with fewer than k_AB links"],
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

const CARVE_E = `
fork      router R 4x2; branch X = three 4x2 stages, returns over fabric
                        branch Y = one 4x2 stage, returns over host
need_edges  = 2 (X) + 1 (Y) = 3      need_slices = 1 + 3 + 1 = 5
candidates  slices with >= 3 neighbours: A2, B2      try A2
order       fabric branches first, shortest first; then host branches
X           _cycles_through(A2, length 4) on free slices: A2 A3 B3 B2
            walks [A3 B3 B2] and [B2 B3 A3]; take the first
            X0 A3, X1 B3, X2 B2, return B2 -> A2 over the cable
Y           A2's free neighbours: {A1}; _path_of_length(A1, 1) = [A1]
            Y0 A1, return over host
plan        R A2 | X A3 B3 B2 | Y A1 | fillers A0 B0 B1

if Y also returned over fabric: need_edges 4, best any slice has is 3
  -> no slice has the 4 fabric neighbours this fork needs`;

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
          ["mixed4", "(P, Q)", "A0-B0, A3-B3", "reject", "one loop of 8 slices, the ring needs exactly 6; every tiling fails"],
          ["mixed4", "(P, R)", "A0-B0, A2-B2", "accept", "a 6-slice loop exists and the stages align onto it"],
          ["RELAXED (later): mixed4 with s3 > s0 RELAXED", "(P, Q)", "A0-B0, A3-B3", "accept", "one fabric path covers 6 slices; the RELAXED loopback goes over host"],
          ["pod8", "(P, U)", "A3-T0, A0-T3", "reject", "opposite corners of M1: no path through M1's tiles joins them"],
          ["pod8", "(P, V)", "A3-T1, A0-T0", "accept", "neighbouring tiles"],
        ]}
        rowTone={["danger", "success", "success", "danger", "success"]}
      />

      <H2>How the carve planner works</H2>
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

      <H3>E. A fork (plan_fork_carve)</H3>
      <Grid columns={fit(380)} gap={20} align="start">
        <SliceMap
          stages={[
            { mesh: 0, from: 2, to: 2, label: "R" },
            { mesh: 0, from: 3, to: 3, label: "X0" },
            { mesh: 1, from: 3, to: 3, label: "X1" },
            { mesh: 1, from: 2, to: 2, label: "X2" },
            { mesh: 0, from: 1, to: 1, label: "Y0" },
            { mesh: 0, from: 0, to: 0, label: "idle", idle: true },
            { mesh: 1, from: 0, to: 1, label: "idle", idle: true },
          ]}
          cables={[
            { a: 0, b: 0, kind: "unused" },
            { a: 2, b: 2, kind: "required" },
            { a: 3, b: 3, kind: "required" },
          ]}
          host={{ from: [0, 1], to: [0, 2], label: "Y0 > R over host" }}
        />
        <CodeCard path="plan_fork_carve, router plus two branches" mode="plain" language="text" src={CARVE_E} />
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
          ["a huge fabric", "3", "budget-exhausted; Tier C treats it as unknown and accepts"],
        ]}
      />

      <H2>Examples: carve paths</H2>
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
          ["a fork with a two-slice router", "ForkShapeUnsupported: use the all-STRICT paths"],
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
          ["a 1x2 router as stage 0 with 3 edges", "3 + 2 = 5 links against 2 chips x 2 = 4 slots"],
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
          ["count 3, spare on A0-B0 only", "R takes two, the fill takes the spare; spread order changes nothing"],
          ["count 3, spare on A3-B3", "the spread fill takes A3-B3 first: a new slice pair, more cycles for blaze (example C)"],
          ["count 1", "R needs 2 links: over budget, the seating is rejected (Tier B already rejects this MGD)"],
          ["phase 0 cannot place a cable in R", "first version: control-plane init fails naming the link; with part 11, the mapper rejects the seating"],
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
          rejects only when every carve path fails.
        </Callout>
        <Callout tone="neutral" title="Use the builder's code">
          Reuse the C++ resolver as-is, port the carve planner, share the inter-mesh binding rule with the control plane. One
          rule per language, parity-tested.
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
          Not in the first version: every edge is STRICT and there is no policy field. This is the design for M5 (P14, P11).
        </Callout>
        <Text tone="secondary">
          Each edge says whether it must ride the fabric; the words match channels policy. STRICT is the default (enum value
          0): an edge with no policy is STRICT, and only edges marked policy: RELAXED may go over host. Every edge STRICT is blaze's
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
enum EdgePolicy {
  STRICT = 0;    // must ride a fabric link (default)
  RELAXED = 1;   // fabric if the placement allows, else host MPI
}

edges { src: "s0" dst: "s1" }                    # STRICT
edges { src: "s1" dst: "s2" }                    # STRICT
edges { src: "s2" dst: "s3" }                    # STRICT
edges { src: "s3" dst: "s0" policy: RELAXED }    # may go over host`}
          />
          <Table
            headers={["Step", "STRICT edge", "RELAXED edge"]}
            rows={[
              ["Tier B crossing rules", "rides a declared connection, within its channels", "left out"],
              ["Plan 1 k_AB", "counted", "not counted"],
              ["Tier C", "a fabric link in the layout", "the host-cut planner may cut it; each fabric chain resolved alone"],
              ["Fork router degree", "STRICT return: 2 router neighbours", "RELAXED return or dead end: 1"],
            ]}
          />
        </Grid>
        <Callout tone="neutral" title="RELAXED is allowed on every edge">
          An MGD stage does not have to map onto a blaze stage yet, so the MGD does not restrict where RELAXED goes. Tier B and
          Plan 1 leave every RELAXED crossing out. Tier C mirrors blaze's host-cut planner, which never cuts two edges, so a
          RELAXED one still needs a fabric link: stage 0's outgoing edge (_group_walk skips 0 in cuts; stage 0's rank drives
          inference and cannot pump), and the router's edge into each branch (_find_branches asserts a fabric neighbour; the
          router sends through its d2d_exchange). A branch's return to the router may go over host.
        </Callout>
        <Callout tone="info" title="Note: rules that might be good to add">
          Once MGD stages map onto blaze stages, Tier A could reject RELAXED where blaze can never carry the edge over host:
          stage 0's outgoing edge, a router's edge into a branch, and any edge of an open chain (graph.py raises "host edges are
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
            ["Shapes rank 2, dims > 0", "shape { dims: [ 4, 2, 1 ] }"],
            ["Pipeline names unique and non-empty; each pipeline is checked on its own", "two pipelines both named mixed4"],
          ]}
        />
        <Text size="small" tone="tertiary">
          Repeating an edge declares a second channel on that hop. The ring needs no loopback marker: s3 to s0 is the loopback.
          Every edge is STRICT in the first version. RELAXED (later): any edge may be RELAXED; Tier A will not restrict where.
        </Text>
        <Callout tone="info" title="NOTE: Tier A does not limit stage topology">
          In-degree, out-degree, how many stages have in-degree 0, and whether a branch is a chain are all accepted. The MGD
          does not identify which stage is the router yet. Once it does, the graph can be constrained to blaze's shape: one
          router, each branch a chain, and at most two branches (_find_branches builds exactly two; fork_structure accepts
          any number). Until then a graph blaze cannot build still passes Tier A and fails at build time.
        </Callout>
        <Callout tone="warning" title="Dead-end branches are allowed">
          A branch may end without returning to the router. Tier B and the carve planner place it like a host-return branch:
          an open path from a router neighbour, using 1 router neighbour instead of 2. TODO (blaze): fork_structure and
          _find_branches both reject a branch with no return today, so until they accept it such a descriptor passes Tiers A
          to C and fails at build time.
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
            ["C1. Build the physical pipeline graph", "candidates, host slices, adjacency, carve paths -> slots", "one chip-level graph per carve path and cycle", "every STRICT carve path (P5, P6, P7, P8)", "host-cut path for RELAXED edges (P11)"],
            ["C2. Pipeline solve", "the resolver solves the pipeline graph onto a C1 graph", "layout and R within each demand, or a reason", "resolver, demand check, tight-mesh segment test and colour pruning; no node budget (P4, P8, P9)", "per-pair link budget inside the resolver"],
            ["C3. On success", "emit the mapping and publish R", "mapping, plan, get_pipeline_required_links()", "phase 0 and fill in the control plane (P9)", "phase 0 in the mapper (policy plan part 11)"],
            ["C4. On failure", "constraints for the next solve", "exit-node retry, then exclude_mapping(seating)", "seating exclusion, unique_shapes off, cap (P8); exit-node retry (P15)", "seat-pair clauses (P13); constraints before the first solve (P10)"],
          ]}
        />
        <Callout tone="info" title="Rule: do not commit to a carve before it is needed">
          A different intra-mesh layout or a different set of bound cables can make a different carve work. So C1 builds
          its graphs over every cable that could be bound, not a binding fixed in advance, and the binding follows the solve (R is
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
          title="New: candidates per seating"
          tag="C1: P7, P8"
          what="C1 reads its inter-mesh cables from the policy plan's candidate table, so it sees exactly what the pairing may bind: links that are up, on allowed cable types, between the two mapped meshes, on requested exit nodes when the connection is device-level."
          todo={[
            "After intra-mesh success, call collect_inter_mesh_candidates(mesh_graph, psd, full.fabric_node_to_asic, options) with the same options as the final call (down-link hashes, PortTypeRules).",
            "Build the resolver's DirectLinks from it: one LinkPair per candidate link, ordered by best tier then chip id, so the first-found layout crosses on the best cables.",
            "Keep the table with the accepted seating; the control plane uses the same one (part 9).",
          ]}
        />
        <BindingPiece
          title="New: R must fit each connection's demand"
          tag="C2: P9"
          what="Phase 0 takes one channel per link in R, counted against the pair's demand. Tier B's channel rule does not guarantee this: it checks that some projection fits, and C1 may carve one with more crossings."
          todo={[
            "Budget: links in R on a pair <= strict + relaxed; R uses the STRICT part first, then RELAXED.",
            "RELAXED and BEST_EFFORT pairs with no link in R keep the policy plan's rules, zero-link included; pairs with links in R need them.",
            "Check in C2 after the resolver; over budget fails that graph with pipeline_binding and C2 moves to the next C1 graph.",
            "If it fires in practice: an optional per-mesh-pair link budget in LinkSearch (default none, so Tier D is unchanged).",
          ]}
        />
      </Grid>
      <Stack gap={8}>
        <H2>C2 cost: the resolver on layouts that do not exist</H2>
        <Table
          headers={["Case (rings of single-slot stages, 4x2-chip slots)", "Layout exists?", "Resolver", "Plain CaDiCaL", "Segment test"]}
          rows={[
            ["one 4x8 grid, 32 stages", "yes", "0.7 ms", "5 ms", "passes, resolver runs"],
            ["one 8x8 grid, 64 stages", "yes", "4.4 ms", "35 ms", "passes, resolver runs"],
            ["odd 5x5 grid, 25 stages", "no", "2.5 s", "2 ms", "rejects: odd grid, no ring"],
            ["two 4x4 meshes, 32 stages, cables on same-colour slices", "no", "3.4 s", "3.0 s", "rejects: same-colour ends"],
            ["two 4x8 meshes, 64 stages, same", "no", "over 120 s", "over 230 s", "rejects: same-colour ends"],
          ]}
        />
        <Text tone="secondary">
          Rule: fail fast, no node budget. Every C2 run ends in a layout or a proof, and a slower average is fine. Neither
          solver proves the failures fast on its own; the exact segment test does. When a mesh is fully used, the pipeline
          runs through it between two cable chips, and on a full rectangular grid that path has an O(1) existence test
          (Itai-Papadimitriou-Szwarcfiter: colours plus a few cases on grids 1 to 3 wide). The resolver applies the same
          test in remaining_stages_can_connect once a mesh's free slots exactly match its unplaced stages.
        </Text>
        <Text tone="secondary">
          Not covered: a fully used mesh with dead slots (Hamiltonian path is NP-complete with holes). None in the repo today.
          Proposed fallback is a hard fail after 10 s naming the mesh, never an undecided pass (open question 13).
        </Text>
        <Text size="small" tone="tertiary">Source: standalone benchmark of resolve_pipeline_placement copied verbatim; CaDiCaL encoding of stage-to-slot placement only.</Text>
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
        <H2>Carve paths (step 5)</H2>
        <Text tone="secondary">
          Blaze turns meshes into submeshes in one of these ways. Tier C tries every path blaze has for the MGD and accepts if
          any succeeds.
        </Text>
        <CarvePaths />
        <Table
          headers={["MGD", "every edge STRICT (first version)", "some edge RELAXED (later)"]}
          rows={[
            ["one mesh, every stage on one rank", "single-mesh tiling", "single-mesh tiling"],
            [
              "one mesh with a joint stage, or several meshes",
              "planned carve, then per-rank tiling",
              "host cuts on the RELAXED edges, then the plan's carve or tiling; a fork the planner cannot handle falls back to the all-STRICT paths",
            ],
          ]}
        />
        <Text size="small" tone="tertiary">
          Blaze's multi_mesh flag is set for more than one mesh or any stage spanning several ranks, so a galaxy with 4x1 rank
          columns and 4x2 stages takes the planned carve (carver example F).
        </Text>
      </Stack>

      <Stack gap={10}>
        <H2>Reuse or rebuild</H2>
        <Table
          headers={["Piece", "Today", "Plan", "PR"]}
          rows={[
            ["Graph-layout resolver", "resolve_graph_layout_with_connections (C++)", "use as-is, add structured failure", "P4"],
            ["Submesh links", "discover_submesh_links (C++, asks the control plane)", "pure core + link callback", "P4"],
            ["Host-rank chip ranges", "MeshGraph::get_coord_range, coordinate_to_chip", "use as-is", "P8"],
            ["Row-major tiling", "MeshDeviceImpl::create_submeshes (needs a device)", "shared pure helper", "P6"],
            ["Carve planner", "carve_planner.py", "C++ port, golden parity", "P5"],
            ["Slot pick + joint collapse", "_collect_submeshes (Python)", "rebuild, small", "P6"],
            ["Host-cut planner (RELAXED, later)", "host_edge_planner.py", "C++ port; cuts only RELAXED edges", "P11"],
            ["Per-chain resolve (RELAXED, later)", "_resolve_chains_separately (Python)", "loop over the resolver", "P11"],
            ["Inter-mesh binding", "pair_logical_intermesh_ports (control plane)", "pure helpers, shared", "P7"],
          ]}
        />
      </Stack>

      <Stack gap={8}>
        <H2>Carve planner behaviour the port must keep (or change on purpose)</H2>
        <Table
          headers={["Behaviour in blaze", "Tier C does"]}
          rows={[
            ["Returns the first aligned cycle; never retries if the resolver then fails", "tries further cycles until one resolves (existence). Gap noted, open question 8"],
            ["budget-exhausted after 200,000 expansions", "treats it as unknown: accept and let blaze decide"],
            ["Adjacency: one confirmed link makes two slices adjacent", "same; the resolver gives the exact answer"],
            ["Stage rectangle must equal the declared (rows, cols)", "same, no transpose"],
            ["Unused slices become idle filler slots; plan must cover every chip", "same"],
            ["crc32 digest exchanged across ranks", "not needed: Tier C plans centrally"],
          ]}
        />
      </Stack>

      <Stack gap={10}>
        <H2>Without a pipeline descriptor: every hook off</H2>
        <Text tone="secondary">
          An MGD with no pipeline_descriptors after merge maps, binds and writes rank bindings exactly as today: no extra search,
          no extra SAT clauses, no changed setting. Generated MGDs (generate_mesh_graph_descriptor_of_shape) have no pipelines,
          so they are always off. One gate, not a proto check per hook: the merged descriptor's has_pipeline_descriptors() and
          pipeline_mapping_constraints() (null when off), and TopologyMapper's pipeline_ pointer, set once from
          mesh_graph_.get_mesh_graph_descriptor(). A null pointer means the hook adds nothing, never "run with defaults".
        </Text>
        <CodeCard path="the gate" mode="plain" language="cpp" src={PIPELINE_GATE} />
        <Table
          headers={["Hook", "Where", "When off"]}
          rows={[
            ["schema pipeline_descriptors (1.3)", "MGD proto", "absent field; existing textprotos parse unchanged"],
            ["merge (1.5)", "MeshGraphDescriptor::merge", "merged proto byte-identical to today's"],
            ["Tier A (2.1)", "MGD constructor", "loops over zero pipelines"],
            ["Tier B (2.2)", "MGD constructor", "PipelineFeasibility not built; pipeline_mapping_constraints() is null"],
            ["stage-to-mesh, seat and exit helpers (2.2, 4.3)", "mapper helpers", "return without adding a constraint"],
            ["seat-graph edge drop for k_AB (4.3)", "build_sat_placement_seat_graph", "today's saturated_link_count edges"],
            ["unique_shapes (4.3)", "map_multi_mesh_to_physical, generate_rank_bindings --all-solutions", "true, as today"],
            ["exact check and SAT feedback (4, 4.3)", "MultiMeshSolutionEnumerator::next", "today's loop; no pipeline_* failure stage"],
            ["required links R and budget check (4.2)", "mapper", "R empty; no budget check"],
            ["phase 0 (4.2)", "pair_logical_intermesh_ports", "no required channel: today's phases"],
            ["spread fill (4.2)", "fill order", "the policy plan's (tier, chip id) order"],
            ["single-host phase 0 (4.2)", "generate_intermesh_connections_on_local_host", "nothing required: today's greedy, with the policy plan's table filter"],
            ["stage plan in rank bindings (4.4)", "generate_rank_bindings", "no stage fields written"],
          ]}
        />
        <Text tone="secondary">
          Not gated, so it must be a pure refactor: the candidate table (policy plan, part 9) runs for every MGD, and that plan's
          parity tests show identical AnnotatedIntermeshConnections. If P7 needs the interim gather helper, it lands on its own
          with the same goldens before any pipeline logic uses it.
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
            ["counter or test-only hook, no descriptor", "no PipelineFeasibility, no extra SAT clauses, no exact-check calls"],
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
const PipelineMappingConstraints* pipeline_mapping_constraints() const; // null when off

// TopologyMapper, set once in the constructor from mesh_graph_.get_mesh_graph_descriptor()
const PipelineFeasibility* pipeline_ = nullptr;                          // null when off`;

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
            ["later: seat-pair clause", "the SAT placement", "not seat_a and seat_b together, when proven for every carve", "P13"],
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
              "New MappingConstraints API add_forbidden_joint_placement: one indicator per (ASIC, slice), one clause that not all repeat.",
              "Blocks are per retry, not kept: once another mesh's layout changes, an old signature may work again.",
              "At most kMaxLayoutRetries (proposed 4) per seating, then move 2.",
            ]}
          />
          <BindingPiece
            title="Later, optional: exit-node constraints for every seating"
            tag="P10"
            what="Before C1, on every seating, steer the intra-mesh solve's exit nodes, only with what every carve agrees on (Tier B's minimum k and union of boundary slices); k = 0 adds nothing."
            todo={[
              "apply_pipeline_exit_constraints: at least ceil(k / links per exit ASIC) nodes of boundary_slices(m, n) on ASICs cabled toward n.",
              "Only if P8's rejection counts call for it.",
            ]}
          />
        </Grid>
        <Callout tone="info" title="Decision">
          First version: Plan 2 (P8) and the exit-node retry (P15). After intra-mesh success, run C1 and C2; when every graph fails and the exit-node retry (P15) cannot help, exclude_mapping(seating)
          blocks that seat set in the live master session and the loop draws the next seating. Plan 1 (P10) is a later,
          optional speed-up, added only if the measured rejection counts call for it. It must never remove a seating Plan 2
          would accept, so it changes speed, not answers.
        </Callout>
        <Table
          headers={["", "Option 1: pipeline in the SAT solve", "Option 2: check each seating, feed back"]}
          rows={[
            ["Solves", "one", "one master session; each rejected seating is blocked in place, no re-encode"],
            ["Exact?", "no: only necessary conditions fit at seat level", "yes: the builder's carve planner and resolver on the real layout"],
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
              ["one seat pair, proven for every carve", "seat-pair clause (new API, P13)"],
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
// tt-metalium/experimental/fabric/pipeline_required_links.hpp (new)
struct PipelineRequiredLink {
    FabricNodeId src;  // exit node on the source mesh
    FabricNodeId dst;  // entry node on the destination mesh
    std::string pipeline, src_stage, dst_stage;  // for logs and failure messages
};
// Keyed by (lower mesh id, higher mesh id); both directions of an edge fold into one pair.
using PipelineRequiredLinks = std::map<std::pair<MeshId, MeshId>, std::vector<PipelineRequiredLink>>;

const PipelineRequiredLinks& TopologyMapper::get_pipeline_required_links() const;

// Marks the channels of each required link, so the pairing binds them first.
PortDescriptorTable to_port_descriptor_table(const InterMeshCandidates&, const PipelineRequiredLinks&);`;

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
            ["Fill order", "same", "after phase 0", "reused tier order, plus a new spread order"],
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
        <H2>Taken from the inter-mesh policy plan</H2>
        <Table
          headers={["Piece", "Policy plan part", "Used here for"]}
          rows={[
            ["get_inter_mesh_channel_demand(a, b): {strict, relaxed}", "3, 5", "the budget R must fit; Tier B's cap(A, B)"],
            ["collect_inter_mesh_candidates(mesh_graph, psd, fabric_node_to_asic, options)", "9", "C1's candidate cables, once per seating"],
            ["cable-type filter and tier (PortTypeRules)", "6", "disallowed cables never reach C1; better tiers tried first"],
            ["PortDescriptor::tier and the tier sort in the pairing", "9", "the fill order after R"],
            ["resolve_inter_mesh_binding in the mapper (stretch)", "11", "later: phase 0 checked before the mapping is emitted"],
          ]}
        />
        <Text size="small" tone="tertiary">
          Order of work: the policy PR and its part 9 land first, then P9 here. If P9 must go first, P7 lands an interim pure
          gather helper with collect_inter_mesh_candidates&apos;s signature (question 10).
        </Text>
      </Stack>
      <Divider />
      <BindingPiece
        title="New: the required links R, and where they live"
        tag="P9"
        what="R is, for each crossing edge in the layout's resolved_edges, the link it uses: exit chip on the source mesh, entry chip on the destination mesh. Not in the MGD (it depends on the seating and layout; pinning it would fix a carve before the solve), not in rank_bindings.yaml for the first version (the runtime mapper recomputes it). It is a mapper output, and a required flag on its channels in the table handed to the pairing."
        code={REQUIRED_LINKS_SRC}
        todo={[
          "Fill R at the end of C2 from the accepted layout; store it on the TopologyMapper with the mapping. Empty without a pipeline.",
          "Add required to PortDescriptor, next to the policy plan's tier; set it in to_port_descriptor_table.",
          "Write required_by: <pipeline>/<src_stage> -> <dst_stage> in inter_mesh_candidates.yaml.",
          "Multi-host: rank 0 holds the table and runs the pairing; R is identical on every rank, only rank 0 consumes it.",
        ]}
      />
      <Stack gap={8}>
        <Row gap={8} align="center">
          <H3>New: phase 0 in the pairing, bind R first</H3>
          <Pill size="sm">P9</Pill>
        </Row>
        <PhaseStrip />
        <Text tone="secondary">
          Phase 0 takes every link with a required channel, places it with its pair&apos;s Z rule on its best-tier channel, and
          counts it against the budget. Today&apos;s work-list then fills the rest. With no required links the output is
          byte-identical to today. If phase 0 cannot place a link (direction owned toward another mesh, Z-ness, one peer per
          direction, no ports), the first version fails at control-plane init naming the link and pipeline edge; it is rare,
          since R is at most the requested count.
        </Text>
      </Stack>
      <Grid columns={fit(340)} gap={20} align="start">
        <BindingPiece
          title="New: fill order"
          tag="P9"
          what="After phase 0, the policy plan's order: best cable tier first. With a pipeline, ties within a tier go to spread: round-robin over host-slice pairs, then chip id. Without a pipeline: (tier, chip id), so goldens do not move."
          todo={["Sort key (tier, slice-pair round-robin, chip id) only with a pipeline descriptor (question 9)."]}
        />
        <BindingPiece
          title="New: single host"
          tag="P9"
          what="generate_intermesh_connections_on_local_host is a separate greedy pass. The policy plan already filters it by the candidate table; it also needs phase 0."
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
              ["Seating, carve, layout, R, budget check", "TopologyMapper (Tier C)", "all, identical"],
              ["Candidate table", "TopologyMapper (policy plan, part 9)", "rank 0"],
              ["Phase 0, then fill", "ControlPlane, pairing", "rank 0"],
              ["Broadcast connections", "ControlPlane", "rank 0 to all"],
              ["validate_requested_intermesh_connections", "ControlPlane", "all"],
            ]}
          />
        </Stack>
        <Stack gap={8}>
          <H3>Gaps</H3>
          <Text tone="secondary">
            At runtime the candidate table leaves out down links, so R never holds one. Offline runs (factory PSD) see every
            cable as up, so the offline answer can be optimistic; the runtime mapper moves to another seating. STRICT
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
            ["1", "Adjacency parity: do bound inter-mesh plus routed intra-mesh links match get_forwarding_direction-confirmed neighbours?"],
            ["2", "How many projections survive Tier B on heterogeneous and fork pipelines? Many survivors weaken Plan 1. Ship the optional \"find all\" run in the first version, or save the defaults until a sweep shows Plan 1 would cut many seatings?"],
            ["3", "Shared seat pairs that need different k: handle generally or reject until needed?"],
            ["4", "Seat-pair nogood API: when is a failure provably about one pair of seats?"],
            ["5", "Cost of unique_shapes off on large symmetric machines."],
            ["6", "Key name for the rank-binding env override (TT_PIPELINE_STAGE?)."],
            ["7", "Per-stage core counts in PipelineStage, or accept the gap."],
            ["8", "Teach blaze to retry the next cycle, or have Tier C check only blaze's first cycle once the plan is consumed?"],
            ["9", "Spread fill only with a pipeline, or everywhere after goldens are regenerated? Should spread ever outrank cable tier?"],
            ["10", "Wait for the policy plan's part 9 before P9, or land an interim pure gather helper first?"],
            ["11", "Phase 0 failure before part 11: fail at control-plane init (proposed), or add a cheap mapper-side check first?"],
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
