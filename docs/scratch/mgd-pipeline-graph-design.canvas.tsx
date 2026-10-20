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
    { title: "MGD parse", sub: "MeshGraphDescriptor ctor", tier: "Tier A", what: "descriptor only", kind: "new" as BoxKind },
    { title: "merge + MeshGraph", sub: "init_from_parts", tier: "Tier B", what: "declared fabric, no PSD", kind: "new" as BoxKind },
    { title: "SAT seating loop", sub: "MultiMeshSolutionEnumerator", tier: "Tier C", what: "PSD + each candidate", kind: "new" as BoxKind },
    { title: "inter-mesh binding", sub: "ControlPlane init", tier: "5.3", what: "binds Tier C's links first", kind: "new" as BoxKind },
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
// Worked example: cases A/B/C under STRICT and FLEXIBLE
// ---------------------------------------------------------------------------

type CaseId = "A" | "B" | "C";
type Mode = "STRICT" | "FLEXIBLE";

const CASES: Record<
  CaseId,
  {
    title: string;
    cables: Cable[];
    strict: { ok: boolean; why: string };
    flexible: { ok: boolean; why: string; labels: { a0: string; a1: string; b0: string; b1: string }; host?: boolean };
  }
> = {
  A: {
    title: "2 links on 2 different slice pairs",
    cables: [
      { side: "left", f: 0.5, kind: "bound", label: "a0-b0" },
      { side: "right", f: 0.5, kind: "bound", label: "a1-b1" },
    ],
    strict: { ok: true, why: "Slices form the cycle a0 > a1 > b1 > b0 > a0. plan_ring_carve finds it and the resolver lays the ring on it." },
    flexible: {
      ok: true,
      why: "Same as STRICT: an all-fabric cycle exists, so the planner cuts nothing to host.",
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
    flexible: {
      ok: true,
      why: "plan_ring_placement lays s0..s3 along the fabric path a1 > a0 > b0 > b1 and moves only the loopback s3 > s0 onto host. Allowed: it is not stage 0's outgoing edge.",
      labels: { a0: "s1", a1: "s0", b0: "s2", b1: "s3" },
      host: true,
    },
  },
  C: {
    title: "1 link",
    cables: [{ side: "left", f: 0.5, kind: "bound", label: "a0-b0" }],
    strict: { ok: false, why: "Same adjacency as Case B: no 4-slice cycle. Rejected as a ring; an open chain (no loopback) would be accepted." },
    flexible: {
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
  const [mode, setMode] = useCanvasState<Mode>("mode", "STRICT");
  const c = CASES[caseId];
  const verdict = mode === "STRICT" ? c.strict : c.flexible;
  const labels = mode === "STRICT" ? { a0: "s0", a1: "s1", b0: "s3", b1: "s2" } : c.flexible.labels;
  return (
    <Grid columns={fit(360)} gap={24} align="start">
      <Stack gap={8}>
        <TwoMesh
          labels={labels}
          cables={c.cables}
          hostEdge={mode === "FLEXIBLE" && c.flexible.host ? { from: "b1", to: "a1", label: "s3 > s0 over host" } : undefined}
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
          {(["STRICT", "FLEXIBLE"] as Mode[]).map((m) => (
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
          because that depends on the seats SAT picks. Under STRICT the slice-level form of Plan 1 can reject Case B before the
          solve, but only when hard pinnings fix the chip layout.
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
  const W = 820;
  const bx = 200;
  const bw = 380;
  const bh = 40;
  const gap = 16;
  const steps: Array<{ title: string; sub: string; kind: BoxKind }> = [
    { title: "1. SAT seating", sub: "placement_session_->next()", kind: "existing" },
    { title: "2. intra-mesh completion", sub: "complete_intra_mesh_for_placement", kind: "existing" },
    { title: "3. candidate links", sub: "gather_intermesh_candidates + routed intra-mesh links  [P7]", kind: "reuse" },
    { title: "4. host slices + slice adjacency", sub: "get_coord_range per rank  [P8]", kind: "new" },
    { title: "5. carve paths -> slots -> submeshes", sub: "planned carve | per-rank tiling | single mesh | flexible  [P5 P6 P11]", kind: "new" },
    { title: "6. resolve layout", sub: "resolve_graph_layout_with_connections (as-is)  [P4]", kind: "reuse" },
    { title: "7. bind the layout's crossing links first", sub: "bind_intermesh_connections(cands, mgd, R)  [P9]", kind: "new" },
    { title: "8. rerun 5-6 over the bound links", sub: "same links runtime will see  [P9]", kind: "new" },
    { title: "9. emit mapping + witness plan", sub: "TT_PIPELINE_STAGE in rank bindings  [P12]", kind: "new" },
  ];
  const y = (i: number) => 14 + i * (bh + gap);
  const H = y(steps.length) + 4;
  return (
    <ScrollX>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W, minWidth: 0.8 * (W) }}>
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
        <text x={bx - 54} y={y(1) + bh / 2 + 16} textAnchor="end" fontSize={10.5} fill={t.text.tertiary}>
          fail: forbid that seat
        </text>
        <text x={bx - 54} y={y(1) + bh / 2 + 29} textAnchor="end" fontSize={10.5} fill={t.text.tertiary}>
          (today)
        </text>
        {/* pipeline failure: new */}
        {[4, 5, 6, 7].map((i) => (
          <line key={i} x1={bx + bw} y1={y(i) + bh / 2} x2={bx + bw + 40} y2={y(i) + bh / 2} stroke={t.category.red} strokeDasharray="4 3" />
        ))}
        <path
          d={`M ${bx + bw + 40} ${y(7) + bh / 2} L ${bx + bw + 40} ${y(0) + bh / 2} L ${bx + bw + 2} ${y(0) + bh / 2}`}
          fill="none"
          stroke={t.category.red}
          strokeDasharray="4 3"
          markerEnd="url(#lf-red)"
        />
        <text x={bx + bw + 48} y={y(3) + 4} fontSize={10.5} fill={t.category.red}>
          all paths fail:
        </text>
        <text x={bx + bw + 48} y={y(3) + 18} fontSize={10.5} fill={t.text.tertiary}>
          exclude_mapping(seating)
        </text>
        <text x={bx + bw + 48} y={y(3) + 32} fontSize={10.5} fill={t.text.tertiary}>
          block the seat set, not a
        </text>
        <text x={bx + bw + 48} y={y(3) + 46} fontSize={10.5} fill={t.text.tertiary}>
          mesh on a seat
        </text>
        <text x={bx + bw + 48} y={y(5) + 14} fontSize={10.5} fill={t.text.tertiary}>
          one path fails:
        </text>
        <text x={bx + bw + 48} y={y(5) + 28} fontSize={10.5} fill={t.text.tertiary}>
          try the next path
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
              stages (a 4x4 stage = 2 slices). Needs uniform slices, stage 0 on one slice, and a loopback.
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
        <CardHeader trailing={<Text size="small" tone="tertiary">placement="flexible"</Text>}>Flexible placement</CardHeader>
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
              First try an all-fabric cycle. Otherwise cover the ring with the fewest fabric chains and join them over host. Then
              carve from the plan, or tile by the uniform stage shape, and resolve each chain on its own.
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
    { title: "Fill order", sub: "spread over slice pairs", kind: "muted" },
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
        <SvgBox x={10} y={84} w={180} h={50} title="part 1 MGD" sub="M0, M1" />
        <SvgBox x={260} y={44} w={180} h={60} title="MeshGraphDescriptor::merge" sub="prefix + remap mesh ids" kind="reuse" />
        <SvgBox x={510} y={44} w={180} h={60} title="merged MGD" sub="M0..M3 + one pipeline" kind="new" />
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
  milestone: "M1 Format" | "M2 Shared code" | "M3 Tier C (STRICT)" | "M4 Pruning + FLEXIBLE";
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
      "Add messages PipelineDescriptor, PipelineStage, PipelineEdge and enums EdgeTransport, PipelinePlacement; field 6 on MeshGraphDescriptor.",
      "Add validate_pipeline_descriptors to static_validate, after validate_pinnings.",
      "Add accessors has_pipeline_descriptor() and pipeline_descriptor().",
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
+    // Pipeline stage graph the topology mapper must be able to place. At most one per run.
+    repeated PipelineDescriptor pipeline_descriptors = 6;
 }
+
+message PipelineDescriptor {
+    string name = 1;
+    repeated PipelineStage stages = 2;
+    repeated PipelineEdge edges = 3;
+    PipelinePlacement placement = 4;   // mirrors blaze build_topology(placement=...)
+}
+
+enum PipelinePlacement {
+    STRICT = 0;     // every FABRIC edge must ride fabric
+    FLEXIBLE = 1;   // the builder may move edges onto host MPI where fabric ends
+}
+
+enum EdgeTransport {
+    FABRIC = 0;
+    HOST = 1;       // D2H -> MPI -> H2D, needs no fabric link
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
+    EdgeTransport transport = 3;
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
    if (proto.pipeline_descriptors_size() > 1) {
        errors.push_back("pipeline_descriptors: at most one pipeline per MGD");
    }
    for (const auto& p : proto.pipeline_descriptors()) {
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
        bool any_host = false;
        for (const auto& e : p.edges()) {
            if (!ids.contains(e.src()) || !ids.contains(e.dst())) {
                errors.push_back(fmt::format("pipeline '{}': edge {} -> {} names an unknown stage", p.name(), e.src(), e.dst()));
            }
            if (e.src() == e.dst()) {
                errors.push_back(fmt::format("pipeline '{}': self-edge on '{}'", p.name(), e.src()));
            }
            any_host |= e.transport() == proto::EdgeTransport::HOST;
        }
        if (p.placement() == proto::PipelinePlacement::FLEXIBLE && any_host) {
            errors.push_back(fmt::format("pipeline '{}': FLEXIBLE plans host edges itself; declare none", p.name()));
        }
        // Linear (out-degree <= 1) or one fork: shared with the carve-planner port (P5).
        if (auto err = detail::check_pipeline_structure(p); err.has_value()) {
            errors.push_back(fmt::format("pipeline '{}': {}", p.name(), *err));
        }
    }
}`,
      },
    ],
    tests: [
      "Golden textprotos: GPT-OSS ring, llama_8b_pod (40 stages), joint 4x4 stage, dflash fork with a HOST return.",
      "One negative textproto per rule (duplicate id, unknown stage, self-edge, rank-3 shape, two routers, FLEXIBLE + HOST).",
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
    goal: "MeshGraphDescriptor::merge copies pipeline_descriptors and rejects two parts that both declare one.",
    files: ["tt_metal/fabric/mesh_graph_descriptor.cpp"],
    steps: [
      "Single part: already copied whole, nothing to do.",
      "Several parts: copy the section in the per-part loop; TT_FATAL if a second part declares one.",
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
+    std::optional<std::size_t> pipeline_part;

     for (std::size_t i = 0; i < descriptors.size(); ++i) {
         ...
         for (const auto& pin : part.pinnings()) {
             *merged->add_pinnings() = pin;
         }
+        if (part.pipeline_descriptors_size() > 0) {
+            TT_FATAL(
+                !pipeline_part.has_value(),
+                "MeshGraphDescriptor::merge: pipeline_descriptors declared in descriptor {} and descriptor {}; "
+                "declare the pipeline in exactly one part.",
+                *pipeline_part,
+                i);
+            pipeline_part = i;
+            for (const auto& pipeline : part.pipeline_descriptors()) {
+                *merged->add_pipeline_descriptors() = pipeline;
+            }
+        }`,
      },
    ],
    tests: [
      "Two parts, pipeline in one: present after merge, byte-identical to the source.",
      "Pipelines in two parts: merge fails and names both part indexes.",
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
      "tt_metal/fabric/topology_mapper_utils.cpp",
    ],
    steps: [
      "Binding set per stage: meshes whose slice tiles the stage shape exactly and that fit it. Stage 0 and a fork router must be one slice.",
      "Capacity in slices, per mesh class.",
      "Projection exists: backtracking stage -> mesh with capacity; under STRICT every FABRIC crossing rides a declared connection within its channel count. FLEXIBLE skips the crossing rules.",
      "Keep facts for Plan 1: min crossings k_AB over all surviving projections, union of boundary slices.",
      "Run once, right after merge in init_from_parts, so runtime and generate_rank_bindings both get it.",
    ],
    code: [
      {
        path: "tt_metal/fabric/topology_mapper_utils.cpp  (init_from_parts)",
        mode: "diff",
        language: "cpp",
        src: `
     MeshGraphDescriptor merged =
         MeshGraphDescriptor::merge(mesh_graph_descriptors, &per_part_local_to_global_mesh_ids_);
     logical_ = build_logical_multi_mesh_adjacency_graph(merged);
+    if (const auto* pipeline = merged.pipeline_descriptor(); pipeline != nullptr) {
+        // Tier B: throws with every reason when no stage-to-mesh projection exists.
+        pipeline_ = std::make_unique<PipelineFeasibility>(merged, *pipeline);
+    }`,
      },
      {
        path: "tt_metal/fabric/pipeline_feasibility.hpp  (new)",
        mode: "new",
        language: "cpp",
        src: `
class PipelineFeasibility {
public:
    PipelineFeasibility(const MeshGraphDescriptor& merged, const proto::PipelineDescriptor& pipeline);

    // Tier B facts that hold for every surviving projection (Plan 1 reads only these).
    std::size_t min_crossings(MeshId a, MeshId b) const;            // k_AB, 0 if some projection avoids A-B
    const std::set<SliceKey>& boundary_slices(MeshId mesh) const;  // union over projections
    bool flexible() const;

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
      "Same MGDs with placement FLEXIBLE: no crossing-based rejects.",
    ],
    behaviour: "MGDs with an impossible pipeline now fail at load instead of at blaze build.",
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
    title: "Split inter-mesh gather from bind",
    milestone: "M2 Shared code",
    size: "M",
    deps: [],
    goal: "One binding rule, callable from the control plane and from the mapper. No behaviour change yet.",
    files: ["tt_metal/fabric/control_plane.cpp", "tt_metal/fabric/intermesh_binding.cpp (new)"],
    steps: [
      "gather_intermesh_candidates(psd, mapping, mgd): both-sides join, link up, one dst chip per src chip, STRICT pins. Pure over the PSD.",
      "bind_intermesh_connections(candidates, mgd, required_links): today's phases 1a / 1b / 2 lifted out of pair_logical_intermesh_ports.",
      "ControlPlane calls both; the single-host generate_intermesh_connections_on_local_host moves onto them too.",
    ],
    code: [
      {
        path: "tt_metal/fabric/intermesh_binding.hpp  (new)",
        mode: "new",
        language: "cpp",
        src: `
struct IntermeshLink { FabricNodeId src, dst; std::vector<std::size_t> connection_hashes; };
using IntermeshCandidates = std::map<std::pair<MeshId, MeshId>, std::vector<IntermeshLink>>;

// Every link the pairing could bind for this mapping. No MPI: the PSD has every host's cables.
IntermeshCandidates gather_intermesh_candidates(
    const tt::tt_metal::PhysicalSystemDescriptor& psd, const MeshGraph& mesh_graph,
    const LogicalToPhysical& mapping, const LinkUpFn& link_up);

// Today's pair_logical_intermesh_ports, with an optional phase 0 (P9).
std::expected<AnnotatedIntermeshConnections, BindFailure> bind_intermesh_connections(
    const IntermeshCandidates& candidates, const MeshGraph& mesh_graph,
    const std::set<std::pair<FabricNodeId, FabricNodeId>>& required_links = {});`,
      },
    ],
    tests: [
      "Recorded PSDs (SC16 revAB/revC, single galaxy): AnnotatedIntermeshConnections byte-identical before and after.",
      "Single-host multi-mesh MGDs: same connections as today's greedy path, or a reviewed diff if the allocator differs.",
    ],
    behaviour: "None intended; the single-host move is the one place to watch.",
  },
  {
    id: "P8",
    title: "Tier C exact check (STRICT)",
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
      "Hook into MultiMeshSolutionEnumerator::next() after intra-mesh success; on reject call exclude_mapping(seating) (seat set), never add_forbidden_constraint.",
      "Turn unique_shapes off when a pipeline is present (runtime overloads and --all-solutions).",
      "failure_stage pipeline_carve / pipeline_layout; cap rejections per run and report the last reason.",
      "FLEXIBLE descriptors: accept without checking until P11.",
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
+            if (pipeline_ != nullptr && !pipeline_->flexible()) {
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
    const auto cands  = gather_intermesh_candidates(psd, mesh_graph_, full.fabric_node_to_asic, link_up_);  // P7
    const auto slices = host_slices(mesh_graph_);                       // get_coord_range per (mesh, host rank)
    const auto links  = routed_links(mesh_graph_, full, cands);         // intra-mesh topology + candidates

    PipelineCheck last = PipelineCheck::reject("pipeline_carve", "no carve path applies");
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
+        /*unique_shapes=*/!mesh_graph_descriptor.has_pipeline_descriptor(),
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
+                const bool unique_shapes = !args.allow_shape_permutations && !any_has_pipeline_descriptor(mgds);`,
        note: "run_topology_mapping already passes false; seat-pool enumeration must stay false; the three enumerate_flat_grouping_embeddings calls keep their default.",
      },
    ],
    tests: [
      "mesh4x4_stage4x2 on a crafted PSD with Case A / B / C cabling: accept / reject / reject (STRICT).",
      "GPT-OSS single galaxy, llama_8b_pod, subtorus 8-stage: accepted, and blaze builds on the emitted mapping (hardware CI).",
      "A seating that fails only when swapped (unique_shapes regression test).",
      "No pipeline: every existing topology-mapper golden unchanged.",
    ],
    behaviour: "The mapper skips seatings the pipeline cannot be built on. --all-solutions only emits buildable ones.",
  },
  {
    id: "P9",
    title: "Pipeline-aware exit-node binding",
    milestone: "M3 Tier C (STRICT)",
    size: "M",
    deps: ["P7", "P8"],
    goal: "Bind the cables Tier C's layout used, then re-check over the bound set.",
    files: ["tt_metal/fabric/intermesh_binding.cpp", "tt_metal/fabric/control_plane.cpp", "tt_metal/fabric/topology_mapper.cpp"],
    steps: [
      "Phase 0 in bind_intermesh_connections: place each required link (Z rule of its mesh pair) before phase 1a, counted against the budget.",
      "In check(): bind with R, rerun the path over the bound links; reject with pipeline_binding if phase 0 or the rerun fails.",
      "TopologyMapper exposes pipeline_required_intermesh_links(); ControlPlane passes it to the bind on rank 0. Every rank still validates.",
    ],
    code: [
      {
        path: "tt_metal/fabric/intermesh_binding.cpp  (was pair_logical_intermesh_ports)",
        mode: "diff",
        language: "cpp",
        src: `
+    // Phase 0: links the pipeline layout depends on go first, so the fill cannot strand them.
+    for (auto& [boundary, link] : order) {
+        if (!required_links.contains({link->src_node, link->dst_node})) {
+            continue;
+        }
+        if (!has_budget_after_trim(boundary, *link)) {
+            return std::unexpected(BindFailure{link->src_node, link->dst_node, "no budget left"});
+        }
+        const bool placed = is_marked_z(boundary) ? place_link(*link, /*want_z=*/true)
+                                                  : (place_link(*link, /*want_z=*/false) || place_link(*link, /*want_z=*/true));
+        if (!placed) {
+            return std::unexpected(BindFailure{link->src_node, link->dst_node, "no free ports in one direction"});
+        }
+        account_placed(boundary, *link);
+    }
+
     // Phase 1a: assign_z (marked) boundaries claim the Z lane; they are excluded from the NESW phases below.
     for (auto& [boundary, link] : order) {
         if (!is_marked_z(boundary)) {`,
      },
      {
        path: "tt_metal/fabric/pipeline_feasibility.cpp  (inside check, after a path succeeds)",
        mode: "diff",
        language: "cpp",
        src: `
         attempt.required_links = crossing_links(attempt.layout, links);
-        return attempt;
+        auto bound = bind_intermesh_connections(cands, mesh_graph_, attempt.required_links);
+        if (!bound) { last = PipelineCheck::reject("pipeline_binding", bound.error().message()); continue; }
+        auto rerun = try_path(path, slices, routed_links(mesh_graph_, full, *bound));   // links runtime will see
+        if (!rerun.accepted) { last = rerun; continue; }
+        rerun.required_links = attempt.required_links;
+        return rerun;`,
      },
    ],
    tests: [
      "RELAXED count 2 with four candidates where the lowest chip ids share a slice pair: today Case B, after P9 Case A.",
      "R empty (no pipeline): AnnotatedIntermeshConnections byte-identical to today.",
      "Multi-host: R identical on every rank; validate_requested_intermesh_connections passes everywhere.",
    ],
    behaviour: "With a pipeline, the bound cables can differ from today's lowest-id choice.",
  },
  {
    id: "P10",
    title: "Plan 1 pruning",
    milestone: "M4 Pruning + FLEXIBLE",
    size: "M",
    deps: ["P3", "P8"],
    goal: "Drop seat pairs before the solve when they cannot carry the pipeline under any split or layout.",
    files: ["tt_metal/fabric/physical_grouping_descriptor_matching.cpp"],
    steps: [
      "Seat graph: for mesh pairs with k_AB > 0, drop the edge between two seats with fewer than k_AB candidate links between their footprints.",
      "Slice-level form (bipartite matching on distinct boundary slices) only where hard pinnings fix the layout.",
      "Single seat: remove only when it fails under every projection and layout.",
      "Shared seat pairs with different k: fall back to the exact check for those pairs.",
      "FLEXIBLE: k_AB = 0 everywhere, so nothing is pruned.",
    ],
    code: [
      {
        path: "tt_metal/fabric/physical_grouping_descriptor_matching.cpp  (build_sat_placement_seat_graph)",
        mode: "new",
        language: "cpp",
        src: `
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
      "Pruning never removes a seating P8 would accept (property test over the sweep MGDs).",
      "Seatings tried before the first accept drop on mesh4x4_stage4x2 and superpod cases.",
    ],
    behaviour: "Faster; same accepted set.",
  },
  {
    id: "P11",
    title: "FLEXIBLE placement",
    milestone: "M4 Pruning + FLEXIBLE",
    size: "L",
    deps: ["P5", "P8"],
    goal: "Mirror blaze placement=flexible: host cuts, then carve or uniform-shape tiling, then one resolve per fabric chain.",
    files: ["tt_metal/fabric/carve_planner.cpp", "tt_metal/fabric/pipeline_feasibility.cpp"],
    steps: [
      "Port plan_ring_placement and plan_fork_placement (host_edge_planner.py); ForkShapeUnsupported falls back to STRICT paths.",
      "Mark planned cuts HOST; carve from the plan or tile by the uniform stage shape.",
      "Resolve each fabric chain separately, with blaze's pool rule and greedy order.",
      "Remove the 'accept FLEXIBLE unchecked' stub from P8.",
    ],
    code: [
      {
        path: "tt_metal/fabric/pipeline_feasibility.cpp  (try_path, FLEXIBLE)",
        mode: "new",
        language: "cpp",
        src: `
case CarvePath::Flexible: {
    auto plan = pipeline_->is_fork() ? detail::plan_fork_placement(slices, adjacency, ...)
                                      : detail::plan_ring_placement(slices, adjacency, stage_shapes);
    if (plan.fork_shape_unsupported) { return try_strict_paths(slices, links); }   // blaze resolves as declared
    if (!plan.ok) { return PipelineCheck::reject("pipeline_carve", plan.reason); }  // ring: blaze raises too
    auto subs = plan.carve ? submeshes_from_slots(*plan.carve)
                           : tile_uniform(stage_shape_if_uniform());                // blaze sets submesh_shape
    return resolve_each_chain(fabric_chains(edges_minus(plan.host_edges)), subs, links);
}`,
      },
    ],
    tests: [
      "Case B and Case C accepted under FLEXIBLE with the loopback on host.",
      "Golden parity against host_edge_planner for rings and forks.",
    ],
    behaviour: "FLEXIBLE descriptors get checked.",
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
    milestone: "M4 Pruning + FLEXIBLE",
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
];

const MILESTONES: Array<{ id: Pr["milestone"]; text: string }> = [
  { id: "M1 Format", text: "The MGD can carry a pipeline, survives merge, and impossible pipelines fail at load." },
  { id: "M2 Shared code", text: "Libraries Tier C needs, each landing with no behaviour change." },
  { id: "M3 Tier C (STRICT)", text: "The mapper only emits seatings the pipeline builds on, and binds the cables it relied on." },
  { id: "M4 Pruning + FLEXIBLE", text: "Speed-ups and FLEXIBLE placement. Not needed for correctness." },
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
        <H2>Thirteen PRs in four milestones</H2>
        <Text tone="secondary">
          Arrows are hard dependencies. M2 PRs are independent of each other and of M1, so they can land in parallel. Every PR
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

/** Three pods as SAT seats, with the seat-graph link counts between them. */
function SeatGraph() {
  const t = useHostTheme();
  const W = 380;
  const H = 210;
  const pos: Record<string, [number, number]> = { P: [190, 40], Q: [70, 170], R: [310, 170] };
  const edges: Array<{ a: string; b: string; label: string; dropped?: boolean }> = [
    { a: "P", b: "Q", label: "2 links" },
    { a: "P", b: "R", label: "2 links" },
    { a: "Q", b: "R", label: "1 link: dropped by Plan 1 (k = 2)", dropped: true },
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
  host_topology   { dims: [ 1, 4 ] }   # 4 ranks, each a 4x2 slice
  channels { count: 2 policy: RELAXED }
}
# graph: M0 <-> M1, channels { count: 2 policy: RELAXED }

pipeline_descriptors {
  name: "mixed4"  placement: STRICT
  stages { id: "s0" shape { dims: [ 4, 2 ] } }   # 1 slice
  stages { id: "s1" shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: "s2" shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: "s3" shape { dims: [ 4, 2 ] } }   # 1 slice
  edges { src: "s0" dst: "s1" }  edges { src: "s1" dst: "s2" }
  edges { src: "s2" dst: "s3" }  edges { src: "s3" dst: "s0" }
}`;

const RUN_POD8 = `
M0  [4, 8], host_topology [1, 4]   4x2 slices A0..A3, ranks 0-3
M1  [2, 4], host_topology [2, 2]   1x2 slices T0..T3, ranks 4-7

        cols 0-1   cols 2-3        slice adjacency in M1
row 0   T0 (r4)    T1 (r5)          T0 -- T1
row 1   T2 (r6)    T3 (r7)          |      |
                                    T2 -- T3

ring of 8: s0..s3 [4, 2], s4..s7 [1, 2], loopback s7 -> s0, 2 channels`;

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
          <CodeCard path="pod8: different slice shapes (llama_8b_pod pattern)" mode="plain" language="text" src={RUN_POD8} />
          <CodeCard path="the machine" mode="plain" language="text" src={RUN_MACHINE} />
        </Stack>
      </Grid>
    </Stack>
  );
}

// --- Descriptor tab: Tier A and Tier B -------------------------------------

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
          ["s0 > s1, s0 > s2, s1 > s0; s2 has no out-edge", "reject", "out-degree 2 makes it a fork; the branch at s2 never returns to the router"],
          ["mixed4 FLEXIBLE with s3 > s0 as HOST", "reject", "FLEXIBLE plans its own host cuts, only when none are declared"],
          ["mixed4 STRICT with s3 > s0 as HOST", "pass", "a host hop anywhere in a ring is allowed"],
          ["a stage [2, 4] where every slice is 4x2", "pass here", "no transpose, so Tier B finds no mesh it binds to"],
        ]}
        rowTone={["success", "success", "danger", "success", "danger", "danger", "success", "warning"]}
      />

      <H2>Examples: Tier B</H2>
      <Text tone="secondary">
        mixed4: slices are 4x2, four per mesh; 4x2 is one slice and 4x4 two, so every stage binds to either mesh; the ring needs
        6 of 8 slices. Several projections survive and none is chosen.
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
      <Text tone="secondary">
        Every survivor crosses, and a ring crosses a cut an even number of times, so Plan 1 gets k(M0, M1) = 2. pod8 has exactly
        one projection: 4x2 binds only to M0 (M1 has 2 rows) and 1x2 only to M1 (not whole 4x2 slices). M0's four stages fill a
        line, so its boundary slices are A0 and A3.
      </Text>
      <Table
        headers={["Change", "Result", "Rule"]}
        rows={[
          ["mixed4 with s0 [4, 4]", "reject", "stage 0 binds to one slice"],
          ["mixed4 + a fifth stage [4, 6]", "reject", "9 slices needed, 8 declared"],
          ["mixed4 + a fifth stage [4, 4]", "pass", "8 of 8: s0, s3 and one 4x4 on one mesh, two 4x4 on the other"],
          ["mixed4 with a stage [4, 10]", "reject", "wider than either mesh"],
          ["pod8 with M1 4x8, host [1, 4], still 1x2 stages", "reject", "1x2 is not a whole slice; stricter than blaze's per-rank tiling"],
          ["mixed4 with no MGD connection", "reject", "every projection crosses"],
          ["the same, FLEXIBLE", "pass", "crossings may become host hops"],
        ]}
        rowTone={["danger", "danger", "success", "danger", "warning", "danger", "success"]}
      />
    </Stack>
  );
}

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

const FLEX_TRACE = `
phase 1   no 6-cycle (only the 8-cycle)
          census: spare = 2 and 6 + 2 is achievable -> passthrough_ring = 2
phase 3   one fabric component of 8 >= 6, so k starts at 1
          _cover_with_chains(1): smallest slice first, smallest free neighbour next
            1. A0 A1 A2 A3 B3 B2     2. A0 B0 B1 B2 B3 A3     3. A1 A0 B0 B1 B2 B3 ...
phase 4   _group_walk per cover; joints only between stages, never cut 0
          cover 1 best: s0 A3, s1 B3+B2, s2 A0+A1, s3 A2   cut s1->s2  key (1,1,(1,))
          cover 2 best: s0 A3, s1 B3+B2, s2 B1+B0, s3 A0   cut s3->s0  key (1,0,(3,))
          key = (cut page sizes, 0 if the loopback is cut else 1, cuts)
result    cover 2 wins: s3 -> s0 becomes HOST; carve
          M0 4x2@(0,6), M1 4x4@(0,4), M1 4x4@(0,0), M0 4x2@(0,0); A1, A2 idle
          _resolve_chains_separately: one fabric chain s0 -> s1 -> s2 -> s3`;

const TIER_D = `
mixed4 on (P, R), runtime
  control plane            binds the two cables Tier C relied on (5.3)
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
          ["mixed4, FLEXIBLE", "(P, Q)", "A0-B0, A3-B3", "accept", "one fabric path covers 6 slices; the loopback goes over host"],
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
      <H3>Flexible placement: mixed4 as FLEXIBLE on (P, Q)</H3>
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
          verdict={{ ok: true, text: "FLEXIBLE: accepted" }}
        />
        <CodeCard path="plan_ring_placement" mode="plain" language="text" src={FLEX_TRACE} />
      </Grid>
      <Table
        headers={["FLEXIBLE edge case", "Result"]}
        rows={[
          ["a cover whose only alignment cuts s0 > s1", "skipped: stage 0's outgoing edge stays on fabric"],
          ["a joint between chains inside a 4x4 stage", "skipped: a wide stage cannot be split by a host hop"],
          ["pod8 as FLEXIBLE", "HostEdgePlanError (slices not uniform), not caught: reject"],
          ["a fork with a two-slice router", "ForkShapeUnsupported: use the STRICT paths"],
          ["all fabric already, single-slice stages", "no plan; the resolver places it as in STRICT"],
        ]}
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
          ["3", "(P, R)", "cables on A0-B0, A2-B2", "plan (A), resolver, binding rerun pass", "emit"],
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
            ["new: Plan 1", "seat-graph edge drop", "k = 2: Q-R (1 link) dropped"],
            ["new: unique_shapes", "MultiMeshSolutionEnumerator callers", "false: (P, Q) and (Q, P) are separate"],
            ["master session", "MasterSolve::restart", "(P, Q), (Q, P), (P, R), (R, P)"],
          ]}
        />
      </Grid>
      <Table
        headers={["Plan 1 case", "Result", "Why"]}
        rows={[
          ["mixed4, seat pair Q-R", "dropped", "1 link < k = 2"],
          ["mixed4, seat pair P-Q", "kept, fails later", "2 links on distinct slice pairs; it fails on loop length, which no link count sees"],
          ["pod8, seat pairs P-U and P-V", "both kept", "2 links each; with hard pins both touch {A0, A3}; U fails on M1's corners, which is about the path, not the boundary"],
          ["using one projection's boundary slices", "not allowed", "mixed4's survivors disagree (s0, s1 vs s1, s3 on M0); only the union is sound"],
          ["mixed4 as FLEXIBLE", "nothing pruned", "k = 0; Q-R works with one crossing over host"],
        ]}
        rowTone={["success", "warning", "warning", "danger", undefined]}
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
            ["gather", "A0-B0, A0-B0, A2-B2", "same"],
            ["required links", "none", "A2-B2 (s1 > s2), A0-B0 (s3 > s0)"],
            ["bind", "two lowest ids: both A0-B0", "R, filling the count of 2"],
            ["rerun", "not done", "same plan and layout"],
            ["runtime", "no-cycle-of-length-6", "builds"],
          ]}
        />
      </Grid>
      <Table
        headers={["Binding edge case", "What happens"]}
        rows={[
          ["count 3, spare on A0-B0 only", "R takes two, the fill takes the spare; spread order changes nothing"],
          ["count 3, spare on A3-B3", "the spread fill takes A3-B3 first: a new slice pair, more cycles for blaze (example C)"],
          ["phase 0 cannot place a cable in R", "failure_stage = pipeline_binding; the seating is blocked"],
          ["STRICT Graph connections with pins on A0 and A3", "Tier C may use only the pinned cables; mixed4 rejects (example B); nothing to choose"],
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
            single galaxy:
          </Text>
          <CodeCard
            path="gpt_oss_120b_single_galaxy.textproto"
            mode="plain"
            language="protobuf"
            src={`
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
          path="fork with a host return"
          mode="plain"
          language="protobuf"
          src={`
edges { src: "router"   dst: "target0" }
edges { src: "target37" dst: "router" }
edges { src: "router"   dst: "draft0" }
edges { src: "draft10"  dst: "router"
        transport: HOST }`}
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
          Stage 0 and a fork router must each be exactly one slice: one rank owns host-to-device, the loopback entry and
          device-to-host.
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
            ["Linear (out-degree <= 1) or exactly one fork router", "two stages with out-degree 2"],
            ["Every fork branch returns to the router (FABRIC or HOST)", "a branch that dead-ends"],
            ["FLEXIBLE declares no HOST edges (blaze plans them)", "placement: FLEXIBLE + transport: HOST"],
            ["At most one pipeline per MGD", "two pipeline_descriptors entries"],
          ]}
        />
        <Text size="small" tone="tertiary">
          Repeating an edge declares a second channel on that hop. The ring needs no loopback marker: s3 to s0 is the loopback.
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>Merging MGDs</H2>
        <Text tone="secondary">
          The mapper always merges its MGD parts. With two or more parts, merge rebuilds the proto field by field and would drop
          the pipeline. P2 copies it, and rejects a second part that also declares one.
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
          MultiMeshSolutionEnumerator::next() today stops after step 2. Steps 3 to 9 are added, and run only when the MGD has a
          pipeline.
        </Text>
        <Legend />
        <LoopFlow />
      </Stack>

      <Stack gap={10}>
        <H2>Worked example: where the links land matters</H2>
        <Text tone="secondary">
          A 4-stage ring split two stages per mesh crosses M0 to M1 twice. Every case below declares two channels, so the
          descriptor-only checks cannot tell them apart. Pick a case and a placement mode.
        </Text>
        <CaseExplorer />
        <Table
          headers={["Case", "Bound links", "STRICT", "FLEXIBLE"]}
          rows={(["A", "B", "C"] as CaseId[]).map((id) => [
            `Case ${id}`,
            CASES[id].title,
            CASES[id].strict.ok ? "accept: all-fabric ring" : "reject: no 4-slice cycle",
            CASES[id].flexible.host ? "accept: fabric chain + host loopback" : "accept: all-fabric ring",
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
          headers={["MGD", "STRICT", "FLEXIBLE"]}
          rows={[
            ["one mesh", "single-mesh tiling", "single-mesh tiling"],
            ["several meshes", "planned carve, then per-rank tiling", "flexible placement; a fork it cannot handle falls back to STRICT"],
          ]}
        />
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
            ["Host-cut planner", "host_edge_planner.py", "C++ port, FLEXIBLE only", "P11"],
            ["Per-chain resolve", "_resolve_chains_separately (Python)", "loop over the resolver", "P11"],
            ["Inter-mesh binding", "pair_logical_intermesh_ports (control plane)", "pure helpers, shared", "P7"],
          ]}
        />
      </Stack>

      <Stack gap={8}>
        <H2>Carve planner behaviour the port must keep (or change on purpose)</H2>
        <Table
          headers={["Behaviour in blaze", "Tier C does"]}
          rows={[
            ["Returns the first aligned cycle; never retries if the resolver then fails", "tries further cycles until one resolves (existence). Gap noted, open question 9"],
            ["budget-exhausted after 200,000 expansions", "treats it as unknown: accept and let blaze decide"],
            ["Adjacency: one confirmed link makes two slices adjacent", "same; the resolver gives the exact answer"],
            ["Stage rectangle must equal the declared (rows, cols)", "same, no transpose"],
            ["Unused slices become idle filler slots; plan must cover every chip", "same"],
            ["crc32 digest exchanged across ranks", "not needed: Tier C plans centrally"],
          ]}
        />
      </Stack>
      <MapperExamples />
    </Stack>
  );
}

function SatTab() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>Two plans, one solve</H2>
        <Text tone="secondary">
          The SAT targets are logical meshes; the globals are seats (physical footprints). A seat fixes the footprint, but its
          chip layout is only a preference to the intra-mesh solve, so which ASICs sit under each host slice is unknown before
          solving (unless hard pinnings fix it).
        </Text>
      </Stack>
      <Grid columns={fit(340)} gap={20} align="start">
        <Stack gap={8}>
          <H3>Plan 1: prune before the solve (P10)</H3>
          <Table
            headers={["Constraint", "Sound when", "Effect"]}
            rows={[
              ["Seat pair has fewer candidate links than k_AB", "always (k_AB is the min over projections)", "drop the seat-graph edge"],
              ["Seat pair cannot hit k_AB distinct boundary slices", "hard-pinned layouts only", "drop the edge (rejects Case B early)"],
              ["Seat has no link toward a mesh every projection crosses to", "always", "remove the seat"],
              ["FLEXIBLE", "-", "k_AB = 0, nothing pruned"],
            ]}
          />
        </Stack>
        <Stack gap={8}>
          <H3>Plan 2: check and teach (P8)</H3>
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
            would eliminate seatings another split could build on.
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

function BindingTab() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>Bind the cables the pipeline needs</H2>
        <Text tone="secondary">
          After mapping, the control plane picks which physical cables carry each MGD connection. Today it ignores the pipeline,
          so it can throw away the cables Tier C relied on.
        </Text>
        <BindingCompare />
      </Stack>
      <Stack gap={8}>
        <H2>Allocator phases</H2>
        <PhaseStrip />
        <Text size="small" tone="tertiary">
          Phase 0 is new (P9). With no pipeline, R is empty and the output is byte-identical to today. The spread fill (dashed) is
          later work (P13) and only applies with a pipeline.
        </Text>
      </Stack>
      <Grid columns={fit(340)} gap={20} align="start">
        <Stack gap={8}>
          <H3>Who computes what</H3>
          <Table
            headers={["Step", "Where", "Ranks"]}
            rows={[
              ["Seating, carve, layout, R", "TopologyMapper (Tier C)", "all, identical"],
              ["Bind with R first", "ControlPlane, pairing", "rank 0"],
              ["Broadcast connections", "ControlPlane", "rank 0 to all"],
              ["validate_requested_intermesh_connections", "ControlPlane", "all"],
            ]}
          />
        </Stack>
        <Stack gap={8}>
          <H3>Gaps</H3>
          <Text tone="secondary">
            Offline runs (factory PSD) see every cable as up; runtime drops cables whose link is down. A required link that is
            down fails phase 0, and the runtime mapper (which runs Tier C itself) moves to another seating. Strict Graph
            connections pin exit chips, so there is nothing to choose; Tier C checks against the pins.
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
            ["Blaze uses only its first carve plan", "Tier C may accept where blaze's first cycle fails", "rerun over bound links narrows it; question 9"],
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
            ["2", "How many projections survive Tier B on heterogeneous and fork pipelines? Many survivors weaken Plan 1."],
            ["3", "Shared seat pairs that need different k: handle generally or reject until needed?"],
            ["4", "Seat-pair nogood API: when is a failure provably about one pair of seats?"],
            ["5", "Cost of unique_shapes off on large symmetric machines."],
            ["6", "Key name for the rank-binding env override (TT_PIPELINE_STAGE?)."],
            ["7", "Per-stage core counts in PipelineStage, or accept the gap."],
            ["8", "Does the MGD carry STRICT / FLEXIBLE, or does Tier C always check the most permissive mode used?"],
            ["9", "Teach blaze to retry the next cycle, or have Tier C check only blaze's first cycle once the plan is consumed?"],
            ["10", "Spread fill only with a pipeline, or everywhere after goldens are regenerated?"],
          ]}
        />
      </Stack>
    </Stack>
  );
}

const SECTIONS = [
  { id: "overview", label: "Overview", body: OverviewTab },
  { id: "descriptor", label: "The MGD pipeline section", body: DescriptorTab },
  { id: "mapper", label: "Mapper check (Tier C)", body: MapperTab },
  { id: "sat", label: "SAT placement", body: SatTab },
  { id: "binding", label: "Exit-node binding", body: BindingTab },
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
