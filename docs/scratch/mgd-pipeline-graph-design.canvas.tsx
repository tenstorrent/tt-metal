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

function Legend({ prNote = false }: { prNote?: boolean }) {
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
      {prNote ? (
        <Text size="small" tone="tertiary" style={{ color: t.text.tertiary }}>
          Boxes carry the PR id that builds them.
        </Text>
      ) : null}
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
    { title: "inter-mesh binding", sub: "ControlPlane init", tier: "C3", what: "binds Tier C's links first", kind: "new" as BoxKind },
    { title: "pipeline build", sub: "blaze build_topology", tier: "blaze", what: "exists, unchanged", kind: "existing" as BoxKind },
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
    { title: "C2. pipeline solve", sub: "one solve, one carve; resolver check; R = every channel that can carry each crossing; R within demand  [P6]", kind: "reuse" },
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
    { side: "left", f: 0.25, kind: "required", label: "chip 0  (s3>s0, other path)" },
    { side: "left", f: 0.6, kind: "required", label: "chip 1  (s3>s0)" },
    { side: "right", f: 0.25, kind: "required", label: "chip 2  (s1>s2)" },
    { side: "right", f: 0.82, kind: "required", label: "chip 3  (s1>s2, other path)" },
  ];
  const layoutOnly: Cable[] = [
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
              Option 1
            </Pill>
            <Text weight="semibold">Every chip pair that can carry each crossing</Text>
          </Row>
          <TwoMesh labels={{ a0: "s0", a1: "s1", b0: "s3", b1: "s2" }} cables={proposed} />
          <Text tone="secondary">
            R = one entry per crossing, holding every cable between its two slices: chips 1 and 0 for s3 {">"} s0, chips 2
            and 3 for s1 {">"} s2, the resolver&apos;s choice first. Both phases record all four. Phase 0 binds min_channels of each crossing (chip 1 and chip 2, or the other path when one cannot be placed), then today&apos;s phases fill whatever
            budget is left. Same count, different cables, and the ring closes (Case A).
          </Text>
        </Stack>
        <Stack gap={8}>
          <Row gap={8} align="center">
            <Pill size="sm">Option 2</Pill>
            <Text weight="semibold">Only the chip pair the layout determined</Text>
          </Row>
          <TwoMesh labels={{ a0: "s0", a1: "s1", b0: "s3", b1: "s2" }} cables={layoutOnly} />
          <Text tone="secondary">
            R = one entry per crossing with only the resolver&apos;s chip pair: chip 1 for s3 {">"} s0, chip 2 for s1{" "}
            {">"} s2, each with every channel on it (the layout fixes the chip pair, not the channel). Phase 0 binds those;
            chips 0 and 3 are left to today&apos;s fill. No fallback if chip 1 or chip 2 cannot be bound. Open question 25.
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

const PIPELINE_SESSION_DECL = `
// --- what the session's next() returns (P5; channels_by_resolved_edge and required_links P6) ---
// A chip's slot-local (row, col) inside one submesh: what every row and column in GraphLayoutResult means.
using SlotCoord = std::pair<std::uint32_t, std::uint32_t>;

// ETH channels, each one PSD ExitNodeConnection {src_exit_node, dst_exit_node (AsicID), eth_conn {src_chan,
// dst_chan, is_local, port_type}}; std::hash of each is the inter-mesh pairing's connection_hash.
using EthChannels = std::vector<tt::tt_metal::ExitNodeConnection>;

// One submesh the resolver was given: one stage's position on this carve. A position is a rectangle of host slices
// on one mesh, so it has one mesh and one or more of that mesh's host ranks.
struct PipelineSubmesh {
    MeshId mesh;
    std::vector<MeshHostRankId> host_ranks;      // the MGD host ranks whose slices the position covers, ascending
    std::map<SlotCoord, FabricNodeId> chips;     // every chip of the position, by slot-local (row, col)
};

// One pipeline on one carve: the resolver's own input and output (the same layout blaze gets at runtime), and the
// PSD channels behind each resolved edge. The resolver names a chip only by submesh index and slot-local (row, col),
// so the layout is always read together with the submeshes it was given:
//   - stage S sits in submesh i = layout.node_to_submesh.at(S); submeshes[i] gives its mesh, host ranks and chips;
//   - resolved edge k leaves the chip at (exit_row, exit_col) in submesh node_to_submesh.at(edge.src) and enters the
//     chip at (entry_row, entry_col) in submesh node_to_submesh.at(edge.dst);
//   - H2D enters at (h2d_entry_row, h2d_entry_col) and D2H leaves at (d2h_exit_row, d2h_exit_col), both in the
//     submesh of layout.stage_order[0];
//   - so any chip the layout names is submeshes[i].chips.at({row, col}), one lookup.
struct PipelinePlacement {
    std::vector<PipelineSubmesh> submeshes;
        // Indexed by submesh index (the values of layout.node_to_submesh); on this carve submesh i is the position
        // of stage_order[i]. The resolver was given the same chips as InternalChip lists ({fid, row, col}, from
        // position_chips_); here they are keyed by (row, col) so a reader never searches a list.
    GraphLayoutResult layout;
        // Resolver output, as the resolver returns it: stage_order; node_to_submesh (stage -> submesh index above);
        // resolved_edges, one per MGD edge in the same order (src and dst stage names, the loopback mark, the exit and
        // entry (row, col), the exit and entry core slots); and the H2D and D2H chips' (row, col) and core slots.
    std::vector<EthChannels> channels_by_resolved_edge;
        // P6, indexed like layout.resolved_edges: entry k belongs to resolved_edges[k]. Empty for an edge inside one
        // mesh (intra-mesh fabric; the inter-mesh pairing does not touch it). For an edge between two meshes: the
        // channels that can carry it, src on edge.src's side. The layout fixes the chip pair, not the channel, so
        // every channel on a chip pair is listed. Which chip pairs is open question 25: option 1 (the code below)
        // every link between the edge's two stage positions, the resolver's chip pair first; option 2 only the
        // resolver's chip pair.
};

// P6: one crossing: a pair of stage positions on two meshes that pipeline edges join. It lists every link (chip
// pair) between the two positions, each with every PSD channel between its two chips; binding min_channels of
// them, on any of the links, keeps the mapping working. The control plane reads it in P9.
struct RequiredCrossing {
    struct Link {
        FabricNodeId exit;                               // chip on meshes.first
        FabricNodeId entry;                              // chip on meshes.second
        EthChannels channels;                            // every PSD channel between the two chips, lowest
                                                         // source channel first
    };
    std::string pipeline;
    std::pair<std::string, std::string> stages;          // as in resolved_edges; stages.first sits on meshes.first
    std::pair<MeshId, MeshId> meshes;
    std::uint32_t min_channels = 1;                      // the minimum to bind, across all links: the pipeline edges
                                                         // between the two stages, either direction (a repeated
                                                         // edge or a loopback is another channel; Tier B's count)
    std::vector<Link> links;                             // links[0] is the resolver's chip pair (the one the layout
                                                         // determined). Open question 25: option 1 (the code below)
                                                         // adds every other link between the two positions, in
                                                         // chip-id order; option 2 keeps links[0] only
};

// One carve, resolved, its channels recorded and checked. TopologyMappingResult::pipeline keeps the accepted one.
struct PipelineLayoutResult {
    bool success = false;
    std::string failure_stage;                           // when !success: pipeline_carve | pipeline_layout | pipeline_binding
    std::string error_message;
    std::chrono::microseconds solve_time{};              // the solver's elapsed_time; feeds pipeline_solve_seconds
    std::map<std::string, PipelinePlacement> pipelines;  // by pipeline name, when success
    std::vector<RequiredCrossing> required_links;        // P6, R: one entry per crossing; phase 0 binds at least
                                                         // min_channels of each, more if the budget allows (P9)
};

// --- the pipeline placement session (P5): the pipeline twin of SatPlacementEnumerationSession ---
// One seating's pipeline layouts. The constructor builds every position (C1), runs the pre-checks, and sets up
// the constraints, one solver session and each pipeline's carve-independent resolver inputs. Each next() solves
// for one more carve, runs the resolver on it and records the channels it crosses on. It owns every position
// field, so PhysicalMultiMeshGraph is unchanged, and reads cables from the PSD: the seating's exit-node graphs
// list one entry per channel but no channel numbers. Built per seating after intra-mesh success and only when the
// stage graph has nodes, because which ASIC each logical chip sits on decides where the cables between positions
// land. Declared after TopologyMappingResult; PipelinePlacement and PipelineLayoutResult go before it, because
// TopologyMappingResult::pipeline holds one. Methods in pipeline_builder.cpp.
class PipelinePlacementEnumerationSession {
public:
    PipelinePlacementEnumerationSession(
        const LogicalMultiMeshGraph& logical,                // stage graph and info, host grids, intra-mesh links,
                                                             // requested exit nodes, channels per MGD connection
        const tt::tt_metal::PhysicalSystemDescriptor& psd,   // every cable, one ExitNodeConnection per ETH channel
        const TopologyMappingResult& full);                  // the seating's intra-mesh result, as
                                                             // complete_intra_mesh_for_placement returns it (success):
                                                             // its layout, both ways
    PipelinePlacementEnumerationSession(const PipelinePlacementEnumerationSession&) = delete;
    PipelinePlacementEnumerationSession& operator=(const PipelinePlacementEnumerationSession&) = delete;
    PipelinePlacementEnumerationSession(PipelinePlacementEnumerationSession&&) = delete;
    PipelinePlacementEnumerationSession& operator=(PipelinePlacementEnumerationSession&&) = delete;
    ~PipelinePlacementEnumerationSession();

    // Solves for the next carve (never one solved before), runs the resolver on it once per pipeline, records the
    // channels that can carry every crossing and how many each needs (R), and checks them against each connection's
    // demand. success = false with failure_stage pipeline_carve (no carve left, a pre-check, a stage with no position,
    // the conflict cap), pipeline_layout (the resolver threw on this carve) or pipeline_binding (a crossing with fewer
    // channels than it needs, or crossings that need more than a connection's demand).
    PipelineLayoutResult next();

private:
    // The resolver's arguments for one pipeline that do not depend on the carve, made once.
    struct ResolverInputs {
        std::vector<PipelineStageNode> stages;                    // in stage_order
        std::vector<std::string> stage_order;                     // topological_sort of the stage ids as text
        std::vector<::tt::tt_fabric::EdgeInputTuple> edges;       // (src, dst, dst in src's loopback_to), every
                                                                  // MGD edge, repeated ones included
        std::map<std::string, std::uint32_t> stage_chip_counts;   // shape.mesh_size()
        std::map<std::string, std::uint32_t> capacity_overrides;  // blaze's core defaults
    };

    // One pipeline's chips and links on a carve, keyed by submesh index (positions in stage order).
    std::vector<std::vector<::tt::tt_fabric::detail::InternalChip>> submesh_chips(
        const std::vector<std::size_t>& positions) const;
    ::tt::tt_fabric::detail::DirectLinks submesh_links(const std::vector<std::size_t>& positions) const;

    // The one PSD read, in the constructor: every ETH channel between two chips on two meshes the MGD connects,
    // under the gather's rules, keyed by the two chips (both orders), lowest source channel first.
    std::map<std::pair<FabricNodeId, FabricNodeId>, EthChannels> inter_mesh_cables(
        const LogicalMultiMeshGraph& logical, const tt::tt_metal::PhysicalSystemDescriptor& psd,
        const TopologyMappingResult& full) const;

    // P6: for one pipeline on the carve (positions in submesh order), fills placement.channels_by_resolved_edge and adds one R
    // entry per crossing, with its min_channels counted from in.edges.
    void record_channels(const std::string& pipeline,
                         const ResolverInputs& in,
                         PipelinePlacement& placement,
                         const std::vector<std::size_t>& positions,
                         std::vector<RequiredCrossing>& required_links) const;
    // P6: the first crossing with fewer channels than its min_channels, or mesh pair whose crossings need more than
    // its demand (strict + relaxed), as a reason.
    std::optional<std::string> first_over_demand(const std::vector<RequiredCrossing>& required_links) const;

    // Position i is a rectangle of host slices on one mesh that some stage's dims fill exactly.
    std::vector<std::vector<::tt::tt_fabric::detail::InternalChip>> position_chips_;   // FabricNodeId, slot-local row and col
    std::vector<PipelineSubmesh> position_submeshes_;           // the same chips by (row, col), with mesh and host ranks
    std::vector<std::vector<std::uint32_t>> position_slices_;   // slice ids, global across meshes
    ::tt::tt_fabric::detail::DirectLinks position_links_;       // keyed by (position, position), both ways
    AdjacencyGraph<std::size_t> position_graph_;                // disjoint, bordering; one entry per link
    AdjacencyGraph<PipelineStageNode> stage_graph_;             // logical.pipeline_stage_graph_, both ways
    MappingConstraints<PipelineStageNode, std::size_t> constraints_;   // required, resource, completeness
    std::map<std::string, ResolverInputs> resolver_inputs_;     // by pipeline name
    std::map<std::pair<FabricNodeId, FabricNodeId>, EthChannels> channels_;
                                                                // inter_mesh_cables: C1's links between meshes, and
                                                                // every channel a crossing can use (P6)
    std::map<std::pair<MeshId, MeshId>, std::uint32_t> channel_demand_;   // P6: strict + relaxed per MGD connection
    std::optional<std::string> infeasible_;                     // why no carve can exist; next() reports it
    std::unique_ptr<TopologyMappingEnumerationSession<PipelineStageNode, std::size_t>> solver_;   // null if infeasible_
};`;

const PIPELINE_SESSION_FLOW = `
MultiMeshSolutionEnumerator::place_pipelines(full)              per seating, after intra-mesh success (P8)
  PipelinePlacementEnumerationSession pipelines(logical_, psd, full)
    constructor
      C1  cables: inter_mesh_cables -> channels_, every PSD channel between two meshes' chips
          positions: every rectangle of host slices a stage's dims fill, per mesh (they overlap)
          links: per ordered pair of disjoint positions, one LinkPair per linked chip pair
            inside a mesh:    logical.mesh_adjacency_graphs_
            between meshes:   channels_ (the PSD, through the layout)
      C2  pre-checks: area, lattice count, colour and segment     fail: no solver; next() says why
          constraints: required (stage -> positions it fits), resource (position -> slices),
                       completeness when the carve is tight
          solver: TopologyMappingEnumerationSession(stage graph both ways, position graph,
                  constraints, STRICT, conflict_cap)
      inputs: per pipeline, the resolver's carve-independent arguments (stage order, edges, counts)
      demand: channels allowed per MGD connection                                             P6
  layouts = pipelines.next()
      solve:    one carve (stage -> position); each further next() is a new one
      resolve:  per pipeline, the carve's chips and links (private lookups) -> resolve_pipeline_placement
      channels: per resolved edge, the exact PSD channel it crosses on -> channels_by_resolved_edge, and R      P6
      demand:   R per mesh pair <= strict + relaxed, else pipeline_binding                         P6
      returns:  PipelineLayoutResult: each pipeline's chips, GraphLayoutResult and channels, and R
  accept: full.pipeline = layouts       reject: full.success = false, failure_stage, error_message`;

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
      "tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp",
    ],
    steps: [
      "Add messages PipelineDescriptor, PipelineStage, PipelineEdge; field 6 on MeshGraphDescriptor. Every edge is STRICT: PipelineEdge has optional Policy policy = 3, omitted or STRICT is accepted, and RELAXED is rejected by validation until P14. PipelineEdge also has bool loopback = 4, set on the edge that closes a cycle.",
      "Add validate_pipeline_descriptors to static_validate, after validate_pinnings.",
      "Add accessors has_pipeline_descriptors() and pipeline_descriptors().",
      "Store each pipeline as PipelineDescriptorData (textproto order). Stage ids are integers, PipelineStageId (uint32_t, in fabric_types.hpp beside MeshId), like mesh_id and graph_id; edges name stages by them and the label carries the human name. An omitted id is 0, so a second stage without one is a repeated id. Each stored edge keeps its loopback mark. Tier C copies these records into LogicalMultiMeshGraph and builds its stage graph from them (P5).",
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
+    uint32 id = 1;              // unique within the pipeline, referenced by edges
+    string label = 2;           // free text
+    TorusTopology shape = 3;    // rank 2, (rows, cols) in logical mesh orientation
+}
+
+message PipelineEdge {
+    uint32 src = 1;
+    uint32 dst = 2;
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
        std::unordered_set<PipelineStageId> ids;
        for (const auto& s : p.stages()) {
            if (!ids.insert(s.id()).second) {
                errors.push_back(fmt::format("pipeline '{}': stage id {} is repeated", p.name(), s.id()));
            }
            const auto& d = s.shape().dims();
            if (d.size() != 2 || d[0] <= 0 || d[1] <= 0) {
                errors.push_back(fmt::format("pipeline '{}': stage {} shape must be rank 2, dims > 0", p.name(), s.id()));
            }
        }
        for (const auto& e : p.edges()) {
            if (!ids.contains(e.src()) || !ids.contains(e.dst())) {
                errors.push_back(fmt::format("pipeline '{}': edge {} -> {} names an unknown stage", p.name(), e.src(), e.dst()));
            }
            if (e.src() == e.dst()) {
                errors.push_back(fmt::format("pipeline '{}': self-edge on stage {}", p.name(), e.src()));
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
      "One negative textproto per rule (duplicate id, an omitted id repeating 0, unknown stage, self-edge, rank-3 shape, two pipelines with one name). Two chains, and a stage with no edges beside a chain, are rejected (one host I/O stage). A stage with three outgoing edges, a branch stage with out-degree 2, and a dead-end branch all parse. An edge with policy: STRICT parses; policy: RELAXED is rejected.",
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
    goal: "The stage graph, and PipelinePlacementEnumerationSession: the position graph Tier C solves on and the footprint solve. The stage graph is part of LogicalMultiMeshGraph, beside today's mesh graph and exit-node graphs: build_logical_multi_mesh_adjacency_graph merges every pipeline of the merged MGD (Tier A has checked them) into that one graph, beside each stage's own facts (PipelineStageInfo: its MeshShape and its loopback marks; the graph is one way, so it keeps each edge's direction) and each mesh's host grid of slice ranges, so every Tier C function, the resolver call included, reads only the logical graph. The enumerator keeps nothing else from the MGD. Its node is PipelineStageNode (pipeline name, PipelineStageId). The positions and the solve belong to PipelinePlacementEnumerationSession, the pipeline twin of SatPlacementEnumerationSession: built per seating from the logical graph, the PSD and the seating's intra-mesh TopologyMappingResult, it stores each position's chips as the resolver's InternalChips and its links as the resolver's DirectLinks, sets up the constraints and one solver session, and each next() solves for one more carve and runs the resolver on it, returning each pipeline's GraphLayoutResult (with the chips it indexes) in a PipelineLayoutResult. PhysicalMultiMeshGraph is unchanged. Replaces the planner port (planner + resolver placed 0 of 132 carve-requiring cases; footprint + resolver decided 127). Library only.",
    files: [
      "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
      "tt_metal/fabric/topology_mapper_utils.cpp",
      "tt_metal/fabric/pipeline_builder.cpp",
      "tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper_utils.cpp",
      "tests/tt_metal/tt_fabric/fabric_router/test_pipeline_builder.cpp",
    ],
    steps: [
      "Stage node, in topology_mapper_utils.hpp beside LogicalExitNode: PipelineStageNode { std::string pipeline; PipelineStageId stage_id; } with a defaulted operator<=> and an fmt formatter printing <pipeline>/<stage id>. Stage ids are unique only inside a pipeline, so the pair is the identity; the pipeline name is unique across the merged MGD (Tier A).",
      "Three fields on LogicalMultiMeshGraph: pipeline_stage_graph_ (AdjacencyGraph<PipelineStageNode>), pipeline_stages_ (std::map<PipelineStageNode, PipelineStageInfo>: the shape from the stage's dims, and loopback_to, the dst of each loopback edge from that stage) and mesh_host_grids_ (std::map<LogicalMeshNode, MeshContainer<MeshCoordinateRange>>). build_logical_multi_mesh_adjacency_graph fills them only when the MGD has pipelines. The stage graph is all pipelines merged into one: every stage of every pipeline listed (isolated ones too), every edge one way (src -> dst), a repeated edge as a duplicate entry, each pipeline its own component with no edges between pipelines. With several MGD parts, MeshGraphDescriptor::merge has already carried every part's pipelines into the merged MGD and rejected a name declared twice, so they all land in this one graph. The host grid per mesh has get_host_ranks(mesh)'s layout, and each cell is that host's slice, get_coord_range(mesh, rank): its start and its shape. The MGD stores each rank's range explicitly and slices of one mesh need not match (pod8 mixes 4x2 and 1x2), so a cell keeps the range, not a shape alone. The rank is not stored: slices are numbered by (mesh, grid coordinate) and P12 reads the rank at a coordinate from get_host_ranks. A grid, not an adjacency graph: positions are rectangles of slices, and which slices touch is physical (C1, per seating). Without pipelines all three stay empty and the logical graph is today's.",
      "No Tier C function takes the MGD and the enumerator keeps nothing new: the merged MGD stays a local of each constructor, as today. LogicalMultiMeshGraph keeps no copy of the descriptors, only what the graph cannot hold (shapes, edge direction, loopback marks) in pipeline_stages_.",
      "The session, PipelinePlacementEnumerationSession: declared in topology_mapper_utils.hpp beside the mapping types, its methods in pipeline_builder.cpp, shaped like SatPlacementEnumerationSession (non-copyable, non-movable). Public: the constructor (logical, psd, full: the seating's TopologyMappingResult, as complete_intra_mesh_for_placement returns it) and next(), nothing else. Private: submesh_chips(positions) and submesh_links(positions), the lookups next() feeds the resolver with, inter_mesh_cables (the one PSD read, into channels_), ResolverInputs, and the fields, every one an existing type: position_chips_ (std::vector<std::vector<detail::InternalChip>>: FabricNodeId, slot-local row and col), position_slices_ (slice ids for add_resource_constraint), position_links_ (detail::DirectLinks keyed by position pairs), position_graph_ (AdjacencyGraph<std::size_t>), stage_graph_ (the both-ways copy), constraints_, resolver_inputs_ (each pipeline's carve-independent resolver arguments), infeasible_ (why no carve can exist) and the TopologyMappingEnumerationSession. PhysicalMultiMeshGraph is unchanged: it comes from the flat PSD graph and the seating, the positions from the stage shapes, the host grids and the layout. The session takes full itself, so the enumerator builds it right after complete_intra_mesh_for_placement: positions are logical chips (FabricNodeId) and the PSD is in ASIC ids, so full's layout (physical_to_fabric_node) is what says which cable lands on which logical chip. Reused, not defined: PipelineDescriptorData and its Edge for stages and edges, MeshShape for stage shapes, MeshContainer<MeshCoordinateRange> for host slices, std::size_t for positions, MappingResult for the carve inside the session, GraphLayoutResult for what next() returns. topology_mapper_utils.hpp already includes topology_solver.hpp, and includes pipeline_builder.hpp for InternalChip, DirectLinks, EdgeInputTuple and GraphLayoutResult; pipeline_builder.hpp does not change in P5.",
      "Cables, the private inter_mesh_cables(logical, psd, full), called once in the constructor into channels_: for each ASIC holding a logical chip (full.physical_to_fabric_node, through psd.get_asic_id), psd.get_asic_neighbors + get_eth_connections to ASICs holding chips of a mesh the MGD connects to its own, one ExitNodeConnection per ETH channel (the same fields as the PSD's exit-node table, so its std::hash is the pairing's connection_hash), keyed by the FabricNodeId pair, both orders, lowest source channel first. The PSD, not the seating's PhysicalMultiMeshGraph: its exit-node graphs list one entry per channel but no channel numbers, which R (P6) needs; with the layout the PSD gives both. Today's gather rules that need no cluster: both ends mapped onto the two meshes; for a device-level connection only requested exit nodes (logical.mesh_exit_node_graphs_); one destination chip per (source chip, neighbour mesh), else no cable. Not filtered yet: cable type and link state (P16; open question 20).",
      "C1, the first half of the constructor: the positions, into the session's position fields. Slices: logical.mesh_host_grids_, one coordinate range per host laid out as the host grid (the slice Tier B divides out of the declared topology), numbered by (mesh, grid coordinate), ids global across meshes; stage shapes from logical.pipeline_stages_; intra-mesh links from logical.mesh_adjacency_graphs_. For each distinct stage dims and each mesh, every rectangle of host slices the dims fill is a position, appended with its index. Positions overlap on purpose: the graph lists every place a stage could go, and the solve's resource constraint keeps the chosen ones apart, as SAT seating does for overlapping seat footprints. An edge joins two disjoint positions that border each other, by grid adjacency or by a cable, one duplicate entry per link. Each position keeps its chips as the resolver's InternalChips and each ordered pair of positions its links as LinkPairs, found chip pair by chip pair as discover_submesh_links does (that function asks the control plane, which does not exist at mapping time; C1 tests the MGD's intra-mesh links and channels_ instead), so next()'s submesh_chips and submesh_links only look them up. The position graph is built from position_links_ after the pre-checks, one entry per link. No slice or link types are added: while C1 builds a position it is a MeshCoordinateRange of the host grid.",
      "Pre-checks, in the constructor after C1 (file-local pipeline_pre_checks): area, lattice count, and the colour and segment test on tight 1-cable meshes; cable pairs the segment test rules out are dropped from position_links_. A failure sets infeasible_, no solver is built, and next() returns it.",
      "The target is a both-ways copy of logical.pipeline_stage_graph_, the session's stage_graph_, made in the constructor: the solver reads each target edge once, from its lower-indexed end (topology_solver_sat.cpp skips t2 <= t1), so a one-way edge stored at its higher-indexed end would be dropped. Until P11 every edge is a STRICT fabric edge. Error messages print the PipelineStageNode as <pipeline>/<stage id>.",
      "Constraints, in the constructor, as SatPlacementEnumerationSession builds its own: add_required_constraint(stage, positions it fits on its own): its dims, at least as many core slots on the position's chips as the stage has endpoints, and for the host I/O stage room for H2D and D2H. These are the only failures that depend on one stage and one position, so they are filtered here and never learned from the resolver. add_resource_constraint(position -> covered slices).",
      "The resolver in next(), on every carve it solves: per pipeline, the positions of its stages in stage order (target_to_global.at(stage)), submesh_chips and submesh_links of those positions, and the constructor's ResolverInputs (stage order from topological_sort of the stage ids as text, EdgeInputTuple edges with the loopback marks, stage chip counts, blaze's default core counts), then resolve_pipeline_placement and layout.stage_order = stage_order, as the wrapper does. The wrapper's other work (input checks Tier A has done, ChipTuple conversion, control-plane links) is not needed. A throw is failure_stage pipeline_layout, with the pipeline's name and the resolver's message; otherwise next() returns every pipeline's PipelinePlacement (submesh_chips and the GraphLayoutResult). submesh_chips rides along because the layout's node_to_submesh and (row, col) index into it; R reads chips back through it.",
      "Resource completeness with the existing API: when the stage areas add up to exactly the slice count, add_cardinality_constraint(pairs, 1) per slice over every (stage, position) pair covering it. With min_count 1 the encoder emits one plain clause (report 5.4: 66 s to 0.59 s, 170 s to 0.32 s).",
      "The solve: the constructor keeps a TopologyMappingEnumerationSession<PipelineStageNode, std::size_t> with STRICT validation, the SAT engine (Auto picks it) and conflict_cap, and next() calls its next() once: the carve (target_to_global: stage -> position index), never one solved before. No carve left, or the cap, is failure_stage pipeline_carve; so is a constructor that found no carve can exist (a pre-check, a stage with no position), which builds no solver. A capped solve reads as no carve with today's API (open question 13). P6 takes one carve: the resolver accepted the first carve on 101 of 101 benchmark cases; a second carve, if ever needed, is one more next() (open question 24).",
      "No control-plane change: the gather stays as it is.",
    ],
    code: [
      {
        path: "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp  (the stage graph)",
        mode: "diff",
        language: "cpp",
        src: `
+// A stage: its pipeline's name and its id in that pipeline. Stage ids are unique only inside a pipeline.
+struct PipelineStageNode {
+    std::string pipeline;
+    ::tt::tt_fabric::PipelineStageId stage_id = 0;
+
+    auto operator<=>(const PipelineStageNode&) const = default;
+};
+
+// A stage's own facts, kept beside the stage graph (which has no attributes).
+struct PipelineStageInfo {
+    tt::tt_metal::distributed::MeshShape shape;                        // from the stage's dims
+    std::set<::tt::tt_fabric::PipelineStageId> loopback_to;           // dst of each edge from this stage
+                                                                       // marked loopback (copies of one edge agree, Tier A)
+};

 struct LogicalMultiMeshGraph {
     std::map<LogicalMeshNode, AdjacencyGraph<FabricNodeId>> mesh_adjacency_graphs_;
     AdjacencyGraph<LogicalMeshNode> mesh_level_graph_;
     std::map<LogicalMeshNode, AdjacencyGraph<LogicalExitNode>> mesh_exit_node_graphs_;
+
+    // Empty by default. Filled only when the MGD has pipeline_descriptors, and read only when
+    // pipeline_stage_graph_ has nodes (exactly when the MGD has pipelines; Tier A rejects a pipeline
+    // with no stages).
+    // All pipelines merged into one graph: every stage of every pipeline, isolated ones included. Edges
+    // one way, src -> dst, one entry per declared edge (a repeated edge is another channel). Each pipeline
+    // is one component; no edges between pipelines. The solve gives the solver a both-ways copy.
+    AdjacencyGraph<PipelineStageNode> pipeline_stage_graph_;
+    // Every stage's shape and loopback marks: what the graph does not hold.
+    std::map<PipelineStageNode, PipelineStageInfo> pipeline_stages_;
+    // Per mesh: the host grid (the MGD's get_host_ranks layout); each cell is that host's slice of device
+    // coordinates, get_coord_range(mesh, rank).
+    std::map<LogicalMeshNode, tt::tt_metal::distributed::MeshContainer<MeshCoordinateRange>> mesh_host_grids_;
 };

+// fmt::formatter<PipelineStageNode>: "<pipeline>/<stage id>", beside the LogicalExitNode formatter.`,
      },
      {
        path: "tt_metal/fabric/topology_mapper_utils.cpp  (build_logical_multi_mesh_adjacency_graph)",
        mode: "diff",
        language: "cpp",
        src: `
 LogicalMultiMeshGraph build_logical_multi_mesh_adjacency_graph(
     const ::tt::tt_fabric::MeshGraphDescriptor& mesh_graph_descriptor) {
     auto mesh_adjacency_graphs = ::tt::tt_fabric::build_adjacency_graph_logical(mesh_graph_descriptor);
     auto [requested_intermesh_connections, requested_intermesh_ports] =
         get_requested_intermesh_from_mgd(mesh_graph_descriptor);
-    return build_logical_multi_mesh_adjacency_graph_impl(
+    LogicalMultiMeshGraph logical = build_logical_multi_mesh_adjacency_graph_impl(
         mesh_adjacency_graphs, requested_intermesh_connections, requested_intermesh_ports);
+    if (mesh_graph_descriptor.has_pipeline_descriptors()) {
+        AdjacencyGraph<PipelineStageNode>::AdjacencyMap stages;              // one map for every pipeline
+        for (const PipelineDescriptorData& pipeline : mesh_graph_descriptor.pipeline_descriptors()) {
+            for (const PipelineDescriptorData::Stage& stage : pipeline.stages) {
+                const PipelineStageNode node{pipeline.name, stage.id};
+                stages[node];                                                // isolated stages stay listed
+                logical.pipeline_stages_.emplace(node, PipelineStageInfo{MeshShape(stage.dims[0], stage.dims[1]), {}});   // rank 2 (Tier A)
+            }
+            for (const PipelineDescriptorData::Edge& edge : pipeline.edges) {   // P11 later skips needs_fabric == false
+                const PipelineStageNode src{pipeline.name, edge.src};
+                const PipelineStageNode dst{pipeline.name, edge.dst};
+                stages[src].push_back(dst);                                    // one way: the graph keeps direction
+                if (edge.loopback) {
+                    logical.pipeline_stages_.at(src).loopback_to.insert(edge.dst);
+                }
+            }
+        }
+        logical.pipeline_stage_graph_ = AdjacencyGraph<PipelineStageNode>(std::move(stages));
+        for (const auto& [mesh, adjacency] : logical.mesh_adjacency_graphs_) {
+            const MeshContainer<MeshHostRankId>& ranks = mesh_graph_descriptor.get_host_ranks(mesh);
+            std::vector<MeshCoordinateRange> slices;                         // row-major, as ranks.values()
+            for (const MeshHostRankId rank : ranks.values()) {
+                slices.push_back(mesh_graph_descriptor.get_coord_range(mesh, rank));
+            }
+            logical.mesh_host_grids_.emplace(mesh, MeshContainer<MeshCoordinateRange>(ranks.shape(), std::move(slices)));
+        }
+    }
+    return logical;
 }`,
      },
      {
        path: "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp  (the pipeline placement session)",
        mode: "new",
        language: "cpp",
        src: PIPELINE_SESSION_DECL,
      },
      {
        path: "tt_metal/fabric/pipeline_builder.cpp  (the session: C1, the pre-checks, the constraints, and next(): solve and resolve)",
        mode: "new",
        language: "cpp",
        src: `
// File-local: distinct_shapes, host_ranges_filling, chip_id (row-major over the mesh the grid tiles), disjoint,
// mgd_connects, drop_fanned_out_sources, pipeline_pre_checks, positions_that_fit, both_ways, by_index, add_resource_completeness and
// kPipelineConflictCap (the per-solve conflict budget); for the resolver, stages_by_pipeline, edge_tuples,
// node_names, in_stage_order, chip_counts, core_counts and today's topological_sort.
PipelinePlacementEnumerationSession::PipelinePlacementEnumerationSession(
    const LogicalMultiMeshGraph& logical, const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const TopologyMappingResult& full) {
    // C1: every position of every stage's dims, per mesh. Positions overlap: they list every place a stage could go.
    std::uint32_t first_slice = 0;                                             // slice ids are global across meshes
    for (const auto& [mesh, grid] : logical.mesh_host_grids_) {                // host coordinate -> its chip range
        for (const MeshShape& shape : distinct_shapes(logical.pipeline_stages_)) {
            for (const MeshCoordinateRange& hosts : host_ranges_filling(grid, shape)) {   // host-grid ranges of shape's chips
                const MeshCoordinate origin = grid.at(hosts.start_coord()).start_coord();   // the position's first chip
                std::vector<std::uint32_t> slices;
                std::vector<InternalChip> chips;
                PipelineSubmesh submesh{.mesh = mesh};
                for (const MeshCoordinate& host : hosts) {
                    // The MGD numbers a mesh's host ranks row-major over its host grid, as mesh_host_grids_ is laid out.
                    const auto rank = static_cast<std::uint32_t>(host.to_linear_index(grid.shape()));
                    slices.push_back(first_slice + rank);
                    submesh.host_ranks.push_back(MeshHostRankId{rank});
                    for (const MeshCoordinate& coord : grid.at(host)) {
                        const InternalChip chip{
                            FabricNodeId(mesh, chip_id(grid, coord)), coord[0] - origin[0], coord[1] - origin[1]};
                        chips.push_back(chip);
                        submesh.chips.emplace(SlotCoord{chip.row, chip.col}, chip.fid);
                    }
                }
                position_slices_.push_back(std::move(slices));
                position_chips_.push_back(std::move(chips));
                position_submeshes_.push_back(std::move(submesh));
            }
        }
        first_slice += grid.shape().mesh_size();
    }
    // discover_submesh_links' loop with this test in place of the control plane, which mapping runs before:
    // inside a mesh the MGD's links; between meshes a PSD channel between the two chips, read through the layout,
    // under the gather's rules (requested exit nodes, one destination chip per source chip).
    channels_ = inter_mesh_cables(logical, psd, full);                       // both orders
    const auto linked = [&](const FabricNodeId& a, const FabricNodeId& b) {
        return a.mesh_id == b.mesh_id
                   ? std::ranges::contains(logical.mesh_adjacency_graphs_.at(a.mesh_id).get_neighbors(a), b)
                   : channels_.contains({a, b});
    };
    const std::size_t count = position_chips_.size();
    for (std::size_t p = 0; p < count; ++p) {
        for (std::size_t q = 0; q < count; ++q) {
            if (p == q || !disjoint(position_slices_[p], position_slices_[q])) {
                continue;                                                      // overlapping: never adjacent
            }
            std::vector<LinkPair> p_to_q;                                      // exit on p, entry on q
            for (const InternalChip& a : position_chips_[p]) {
                for (const InternalChip& b : position_chips_[q]) {
                    if (linked(a.fid, b.fid)) {
                        p_to_q.push_back({a.row, a.col, b.row, b.col});
                    }
                }
            }
            if (!p_to_q.empty()) {
                position_links_[{p, q}] = std::move(p_to_q);                  // directed per key, as the resolver reads it
            }
        }
    }
    // C2's O(1) pre-checks: area, lattice count, colour and segment on tight meshes. The segment test drops the
    // cable pairs it rules out from position_links_.
    if ((infeasible_ = pipeline_pre_checks(logical, position_slices_, position_links_))) {
        return;
    }
    AdjacencyGraph<std::size_t>::AdjacencyMap adj;
    for (std::size_t p = 0; p < count; ++p) {
        adj[p];                                                                // every position listed
    }
    for (const auto& [pq, links] : position_links_) {
        adj[pq.first].insert(adj[pq.first].end(), links.size(), pq.second);   // one entry per link
    }
    position_graph_ = AdjacencyGraph<std::size_t>(std::move(adj));

    // The constraints, as SatPlacementEnumerationSession builds its own: each stage onto the positions it fits.
    std::map<PipelineStageNode, std::set<std::size_t>> fits;
    for (const auto& [node, stage] : logical.pipeline_stages_) {
        std::set<std::size_t>& fit = fits[node] = positions_that_fit(node, stage, position_chips_);   // shape, core slots, host I/O
        if (fit.empty()) {
            infeasible_ = fmt::format("stage {} has no position on this seating", node);
            return;
        }
        constraints_.add_required_constraint(node, fit);
    }
    constraints_.add_resource_constraint<std::uint32_t>(by_index(position_slices_));   // chosen positions share no slice
    add_resource_completeness(constraints_, logical, position_slices_, fits);         // only when tight
    // The solver reads each target edge once, from its lower-indexed end, so it needs both directions.
    stage_graph_ = both_ways(logical.pipeline_stage_graph_);
    solver_ = std::make_unique<TopologyMappingEnumerationSession<PipelineStageNode, std::size_t>>(
        stage_graph_, position_graph_, constraints_, ConnectionValidationMode::STRICT,
        /*quiet_mode=*/true, TopologyMappingSolverEngine::Auto, /*unique_shapes=*/false, kPipelineConflictCap);

    // The resolver's carve-independent arguments, once per pipeline. Node names are the stage ids as text.
    for (const auto& [pipeline, stages] : stages_by_pipeline(logical.pipeline_stages_)) {   // by name, stage id order
        ResolverInputs& in = resolver_inputs_[pipeline];
        in.edges = edge_tuples(pipeline, logical);                               // loopback marks from loopback_to
        in.stage_order = topological_sort(node_names(stages), in.edges);         // as the wrapper does
        in.stages = in_stage_order(pipeline, in.stage_order);
        in.stage_chip_counts = chip_counts(stages, logical);
        in.capacity_overrides = core_counts(stages, logical);
    }
}

// The one PSD read. For each ASIC that holds a logical chip, its PSD neighbours that hold a chip of another mesh
// the MGD connects to this one, one ExitNodeConnection per ETH channel, built from the same fields as the PSD's
// exit-node table so its std::hash is the pairing's connection_hash. Today's gather rules that need no cluster:
// both ends mapped; for a device-level connection only requested exit nodes (logical.mesh_exit_node_graphs_);
// one destination chip per (source chip, neighbour mesh), else no cable. Not filtered yet: cable type and link
// state (P16; open question 20).
std::map<std::pair<FabricNodeId, FabricNodeId>, EthChannels>
PipelinePlacementEnumerationSession::inter_mesh_cables(
    const LogicalMultiMeshGraph& logical, const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const TopologyMappingResult& full) const {
    const auto& chip_on = full.physical_to_fabric_node;                      // every mapped ASIC and its chip
    std::map<std::pair<FabricNodeId, FabricNodeId>, EthChannels> cables;
    for (const auto& [node, chip] : chip_on) {
        const tt::tt_metal::AsicID asic = psd.get_asic_id(node);
        for (const tt::tt_metal::AsicID neighbour : psd.get_asic_neighbors(asic)) {
            const auto other = chip_on.find(psd.get_physical_node_id(neighbour));
            if (other == chip_on.end() || other->second.mesh_id == chip.mesh_id ||
                !mgd_connects(logical, chip, other->second)) {             // includes the requested exit nodes
                continue;
            }
            for (const tt::tt_metal::EthConnection& eth : psd.get_eth_connections(asic, neighbour)) {
                cables[{chip, other->second}].push_back({asic, neighbour, eth});
            }
        }
    }
    drop_fanned_out_sources(cables);                                         // one destination chip per source chip
    for (auto& [chips, channels] : cables) {
        std::ranges::sort(channels, {}, [](const auto& c) { return c.eth_conn.src_chan; });
    }
    return cables;
}

PipelineLayoutResult PipelinePlacementEnumerationSession::next() {
    PipelineLayoutResult result;
    if (!solver_) {
        result.failure_stage = "pipeline_carve";
        result.error_message = *infeasible_;
        return result;
    }
    const MappingResult<PipelineStageNode, std::size_t> carve = solver_->next();   // never a carve solved before
    result.solve_time = carve.stats.elapsed_time;
    if (!carve.success) {                                                     // none left, or the conflict cap
        result.failure_stage = "pipeline_carve";
        result.error_message = carve.error_message;
        return result;
    }
    // The resolver on every carve, one call per pipeline: the runtime's own chip-level check (3.1).
    for (const auto& [pipeline, in] : resolver_inputs_) {
        std::vector<std::size_t> positions;                                    // submesh i is in.stages[i]'s position
        for (const PipelineStageNode& stage : in.stages) {
            positions.push_back(carve.target_to_global.at(stage));
        }
                PipelinePlacement placement;
        for (const std::size_t p : positions) {
            placement.submeshes.push_back(position_submeshes_[p]);              // submesh i: stage i
        }
        try {
            placement.layout = resolve_pipeline_placement(
                in.stage_order, in.edges, submesh_links(positions), in.stage_chip_counts, submesh_chips(positions),
                in.capacity_overrides);
            placement.layout.stage_order = in.stage_order;
        } catch (const std::runtime_error& e) {
            result.pipelines.clear();
            result.failure_stage = "pipeline_layout";
            result.error_message = fmt::format("pipeline '{}': {}", pipeline, e.what());
            return result;
        }
        result.pipelines.emplace(pipeline, std::move(placement));
    }
    result.success = true;
    return result;
}

std::vector<std::vector<InternalChip>> PipelinePlacementEnumerationSession::submesh_chips(
    const std::vector<std::size_t>& positions) const {
    std::vector<std::vector<InternalChip>> chips;
    for (const std::size_t p : positions) {
        chips.push_back(position_chips_.at(p));                                // already slot-local
    }
    return chips;
}

DirectLinks PipelinePlacementEnumerationSession::submesh_links(const std::vector<std::size_t>& positions) const {
    DirectLinks links;                                                         // re-keyed to submesh indices
    for (std::size_t i = 0; i < positions.size(); ++i) {
        for (std::size_t j = 0; j < positions.size(); ++j) {
            if (const auto it = position_links_.find({positions[i], positions[j]}); it != position_links_.end()) {
                links[{i, j}] = it->second;
            }
        }
    }
    return links;
}`,
      },
      {
        path: "tests/tt_metal/tt_fabric/fabric_router/test_pipeline_builder.cpp  (CPU; MGD textprotos and PSD fixtures)",
        mode: "new",
        language: "cpp",
        src: `
// c.logical = build_logical_multi_mesh_adjacency_graph(c.mgd): stage graph, stage shapes and host grids included.
// c.psd is a CPU PSD fixture; c.full is one seating's intra-mesh result on it.
// The session's only public call is next(), so every check reads the layouts it returns: each slot's chips are
// placement.submeshes, and layout.node_to_submesh says which stage sits in which slot.
TEST(PipelinePlacementSessionTest, OverlappingPositionsAreNeverBothChosen) {
    const auto c = two_by_two_slices_one_by_two_stages();                  // four positions, each slice in two
    PipelinePlacementEnumerationSession carves(c.logical, c.psd, c.full);
    for (auto layouts = carves.next(); layouts.success; layouts = carves.next()) {
        EXPECT_TRUE(slots_are_disjoint(layouts));                          // no chip in two slots
    }
}

TEST_P(PipelinePlacementSessionTest, MatchesBruteForceAndVerifies) {
    const FootprintCase c = GetParam();                                    // MGD + PSD + one seating
    PipelinePlacementEnumerationSession carves(c.logical, c.psd, c.full);
    const PipelineLayoutResult layouts = carves.next();
    EXPECT_EQ(layouts.success, brute_force_layout_exists(c));              // small cases only
    if (layouts.success) {
        EXPECT_TRUE(is_valid_layout(c, layouts));                          // own dims, disjoint, every edge on a link
    }
}

TEST(PipelinePlacementSessionTest, EveryNextIsANewCarve) {
    const auto c = two_by_two_slices_one_by_two_stages();                  // few carves, each passes the resolver
    PipelinePlacementEnumerationSession carves(c.logical, c.psd, c.full);
    std::set<std::map<std::string, std::vector<std::vector<FabricNodeId>>>> seen;   // each pipeline's slots
    for (auto layouts = carves.next(); layouts.success; layouts = carves.next()) {
        EXPECT_TRUE(seen.insert(slot_chips(layouts)).second);              // never the same carve twice
    }
    EXPECT_EQ(seen.size(), brute_force_carve_count(c));
}

TEST(PipelinePlacementSessionTest, ResolverRejectIsPipelineLayout) {
    const auto c = reject_demo_case();       // the solve finds carves; the resolver rejects each one
    PipelinePlacementEnumerationSession carves(c.logical, c.psd, c.full);
    const PipelineLayoutResult layouts = carves.next();
    EXPECT_FALSE(layouts.success);
    EXPECT_EQ(layouts.failure_stage, "pipeline_layout");
    EXPECT_THAT(layouts.error_message, ::testing::HasSubstr("pipeline 'reject_demo'"));
}

TEST(PipelinePlacementSessionTest, PreCheckFailureNeedsNoSolve) {
    const auto c = five_by_five_with_five_two_by_two();                   // the area fits; the lattice holds four
    PipelinePlacementEnumerationSession carves(c.logical, c.psd, c.full);
    const PipelineLayoutResult layouts = carves.next();
    EXPECT_FALSE(layouts.success);
    EXPECT_EQ(layouts.failure_stage, "pipeline_carve");
    EXPECT_THAT(layouts.error_message, ::testing::HasSubstr("lattice"));
    EXPECT_EQ(layouts.solve_time.count(), 0);                             // no solver was built
}`,
      },
      {
        path: "tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper_utils.cpp  (the logical graph's pipeline fields)",
        mode: "new",
        language: "cpp",
        src: `
TEST(LogicalMultiMeshGraphTest, PipelineStageGraphOfTwoPipelines) {
    const MeshGraphDescriptor mgd(two_pipelines_textproto());              // "ring" (0..3, loopback 3 -> 0) and "pair" (0, 1)
    const LogicalMultiMeshGraph logical = build_logical_multi_mesh_adjacency_graph(mgd);
    const auto& stages = logical.pipeline_stage_graph_;                    // one graph, both pipelines merged
    EXPECT_EQ(stages.get_nodes().size(), 6u);                             // both stage 0s are nodes
    EXPECT_THAT(stages.get_neighbors(PipelineStageNode{"ring", 0}),
                ::testing::ElementsAre(PipelineStageNode{"ring", 1}));      // one way: 3 -> 0 is stage 3's
    EXPECT_THAT(stages.get_neighbors(PipelineStageNode{"pair", 0}), ::testing::ElementsAre(PipelineStageNode{"pair", 1}));
    EXPECT_EQ(fmt::format("{}", PipelineStageNode{"ring", 3}), "ring/3");
    const PipelineStageInfo& head = logical.pipeline_stages_.at(PipelineStageNode{"ring", 0});
    EXPECT_EQ(head.shape, MeshShape(4, 2));                               // its dims
    EXPECT_TRUE(head.loopback_to.empty());                                // 0 -> 1 is not a loopback
    EXPECT_THAT(logical.pipeline_stages_.at(PipelineStageNode{"ring", 3}).loopback_to,
                ::testing::ElementsAre(0u));                              // 3 -> 0 is
}

TEST(LogicalMultiMeshGraphTest, PipelinesOfTwoMergedPartsShareOneStageGraph) {
    const MeshGraphDescriptor a(ring_textproto("ring_a")), b(ring_textproto("ring_b"));   // four stages each
    const MeshGraphDescriptor merged = MeshGraphDescriptor::merge({&a, &b});
    const LogicalMultiMeshGraph logical = build_logical_multi_mesh_adjacency_graph(merged);
    EXPECT_EQ(logical.pipeline_stage_graph_.get_nodes().size(), 8u);
    EXPECT_TRUE(logical.pipeline_stage_graph_.get_neighbors(PipelineStageNode{"ring_b", 0}).size() > 0);
}

TEST(LogicalMultiMeshGraphTest, HostGridsMatchTheDescriptor) {
    const MeshGraphDescriptor mgd(gpt_oss_galaxy_textproto());            // 8x4 devices over a 2x2 host grid
    const LogicalMultiMeshGraph logical = build_logical_multi_mesh_adjacency_graph(mgd);
    const auto& grid = logical.mesh_host_grids_.at(MeshId{0});
    const auto& ranks = mgd.get_host_ranks(MeshId{0});
    EXPECT_EQ(grid.shape(), ranks.shape());                               // a 2x2 host grid
    for (const auto& [coord, slice] : grid) {
        EXPECT_EQ(slice, mgd.get_coord_range(MeshId{0}, ranks.at(coord)));
        EXPECT_EQ(slice.shape(), MeshShape(4, 2));                        // four 4x2 slices
    }
}

TEST(LogicalMultiMeshGraphTest, NoPipelinesLeavesTheFieldsEmpty) {
    const LogicalMultiMeshGraph logical = build_logical_multi_mesh_adjacency_graph(MeshGraphDescriptor(galaxy_mgd_path()));
    EXPECT_TRUE(logical.pipeline_stage_graph_.get_nodes().empty());
    EXPECT_TRUE(logical.pipeline_stages_.empty());
    EXPECT_TRUE(logical.mesh_host_grids_.empty());
}`,
      },
    ],
    tests: [
      "Logical graph (test_topology_mapper_utils.cpp): the stage graph of the GPT-OSS ring, the dflash fork, two pipelines that both have a stage 0 (two nodes), a repeated edge (two entries) and an isolated stage (listed, no neighbours); two merged MGD parts whose pipelines share one stage graph; PipelineStageNode formats as ring/3; each stage's shape matches its dims; edges are stored one way and loopback_to marks 3 -> 0; the solve's both-ways copy lists 3 -> 0 under both stages; each mesh_host_grids_ cell equals get_coord_range for the rank at that coordinate; an MGD without pipelines leaves all three fields empty.",
      "Positions and slices from mesh_host_grids_: 4x2-chip and 2x2-chip slices on two meshes; an 8-chip stage is one slice on the first and a 2x2 block on the second.",
      "Overlapping positions: a 2x2 grid of slices gives four 1x2 positions, and no layout next() returns puts two stages on overlapping ones.",
      "A carve the resolver rejects (chip_level/reject_demo): next() returns pipeline_layout with the pipeline's name and the resolver's message.",
      "Public interface: only the constructor and next(); every test reads the layouts next() returns.",
      "Every next() is a new carve: on a small case, calling next() until it fails lists each carve once, as many as brute force finds.",
      "Counting and area failures (5x5 with five 2x2, 6x6 with area 37): next() fails with the pre-check's reason and no solver is built.",
      "Cables on SC16 revAB/revC and single-galaxy PSD fixtures: the channels inter_mesh_cables reads are the ones today's gather connects for the same mapping, on every boundary, with the same connection hashes.",
      "Device-level connection: only requested exit nodes; a source chip cabled to two chips of one neighbour mesh gives no cable.",
      "Brute-force agreement on every case with at most 8 stages; prototype: pipeline_placement_experiments/topo/carve_footprint_bench.cpp.",
      "Position filter: a stage never lands on a position it cannot fit; verdicts with and without the filter are equal on every case.",
      "Resource completeness: same verdicts with and without on every designed, repo, ring and mixed-cable case, and the tight-case speed-up of report 5.4.",
    ],
    behaviour: "None. Library only. Tier C may accept carves blaze's planner rejects (open question 8).",
  },
  {
    id: "P6",
    title: "Channels, R and the demand check, inside the session's next() (C2)",
    milestone: "M2 Shared code",
    size: "M",
    deps: ["P5"],
    goal: "Finish P5's next(): after the resolver, record every PSD channel that can carry each edge that crosses meshes (PipelinePlacement::channels_by_resolved_edge: every channel of every link between its two stage positions, the resolver's chip pair first) and one entry per crossing in PipelineLayoutResult::required_links (R), then check the crossings against each connection's demand. Both are private session functions, so there is no separate per-seating check: the enumerator (P8) builds the session and takes one next(). The PSD is already P5's input and channels_ already holds every channel per chip pair, so R needs no new input. No topology solver changes.",
    files: [
      "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
      "tt_metal/fabric/pipeline_builder.cpp",
      "tests/tt_metal/tt_fabric/fabric_router/test_pipeline_builder.cpp",
    ],
    steps: [
      "Why the layout is not enough: GraphLayoutResult gives each edge's exit and entry chip (slot-local row and col, read back through node_to_submesh and submeshes), not its ETH channel. Two chips can share several channels, and phase 0 binds by channel (connection_hash). Two positions can be joined by several chip pairs too, and the resolver chose among exactly these links (position_links_), so any of them can carry the crossing: the session records all of them, the resolver's chip pair first. The resolver checked only its own choice. Open question 25 has two options: option 1 (the code) carries every link, option 2 only the chip pair the layout determined (links[0]); the layout fixes the chip pair, not the channel, so both list every channel on a kept chip pair.",
      "Constructor: channel_demand_, strict + relaxed per MGD connection. The channels come from P5's channels_ (the PSD's, per chip pair, lowest source channel first), so there is no second PSD read.",
      "IMPORTANT: record_channels (private), per pipeline right after the resolver, with that pipeline's positions. Inside a mesh, channels_by_resolved_edge gets an empty entry (intra-mesh fabric; the pairing does not touch it). Across meshes, every channel of every link between the edge's two positions (position_links_ through channels_): the resolver's chip pair first, then the others in chip-id order (best tier first after P16), stored as is, so each std::hash is the pairing's connection_hash. R gets one RequiredCrossing per crossing (position pair, both directions): those links, each with its channels, the pipeline, stages and meshes, and min_channels, the edges between the two stages in in.edges, either direction (a repeated edge or a loopback is another channel).",
      "first_over_demand (private): every crossing has at least min_channels channels between its positions, and the min_channels per mesh pair add up to at most channel_demand_ (Tier B's sum, on the carve); otherwise it clears the result and returns failure_stage pipeline_binding with the crossing or the mesh pair.",
      "Both phases: generate_rank_bindings (Phase 1) and TopologyMapper at control-plane init (Phase 2) run the same session on their own PSD, so both record channels_by_resolved_edge and R in their TopologyMappingResult, with no phase-specific code. Phase 1 keeps them for logs and tests and uses only the mapping; it writes no channel into the rank bindings. Phase 2 recomputes R and binds it (P9). Hashes are compared only within Phase 2: is_local, part of the hash, comes from host ids and can differ between a single-process Phase 1 PSD and Phase 2's.",
      "Not stored, derived when needed: each stage's host ranks and the idle ranks (the ranks at the mesh_host_grids_ coordinates its position covers, read from get_host_ranks, and the rest, computed by P12); the host I/O stage (layout.stage_order.front()).",
    ],
    code: [
      {
        path: "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp  (the session's P6 lines)",
        mode: "plain",
        language: "cpp",
        src: `
// In P5's declaration (MapperTab), the lines marked P6: PipelinePlacement::channels_by_resolved_edge,
// PipelineLayoutResult::required_links and pipeline_binding, the private record_channels and
// first_over_demand, and channel_demand_.`,
      },
      {
        path: "tt_metal/fabric/pipeline_builder.cpp  (constructor, next(), and the two private functions)",
        mode: "diff",
        language: "cpp",
        src: `
 // pipeline_builder.cpp, the session's constructor, after C1's links
+    // P6: each MGD connection's demand. channels_ (P5) already holds every channel a crossing can take.
+    for (const auto& [mesh_a, mesh_b] : connected_mesh_pairs(logical)) {
+        channel_demand_[ordered(mesh_a, mesh_b)] = channel_demand(logical, mesh_a, mesh_b);   // strict + relaxed
+    }

 // pipeline_builder.cpp, PipelinePlacementEnumerationSession::next, the resolver loop and after it
+    // IMPORTANT: one R entry per crossing: every channel that can carry it and how many it needs.
     for (const auto& [pipeline, in] : resolver_inputs_) {
         ...                                                            // positions, submeshes, the resolver (P5)
         } catch (const std::runtime_error& e) {
             result.pipelines.clear();
+            result.required_links.clear();
             ...
         }
+        record_channels(pipeline, in, placement, positions, result.required_links);
         result.pipelines.emplace(pipeline, std::move(placement));
     }
+    if (auto over = first_over_demand(result.required_links)) {
+        result.pipelines.clear();
+        result.required_links.clear();
+        result.failure_stage = "pipeline_binding";
+        result.error_message = *over;
+        return result;
+    }
     result.success = true;
     return result;
 }

+// Each resolved edge's two submeshes come from node_to_submesh, and its chips from looking the edge's slot-local
+// (row, col) up in their chips; the carve gives the two positions. Between two meshes every link between
+// the two positions can carry the edge (the resolver chose among exactly these), so every channel of every one is
+// recorded: the resolver's chip pair first, then the others in chip-id order. C1 links only cabled chip pairs, so
+// each lookup finds at least one channel.
+void PipelinePlacementEnumerationSession::record_channels(
+    const std::string& pipeline,
+    const ResolverInputs& in,
+    PipelinePlacement& placement,
+    const std::vector<std::size_t>& positions,
+    std::vector<RequiredCrossing>& required_links) const {
+    const std::vector<PipelineSubmesh>& submeshes = placement.submeshes;   // submesh i: mesh, host ranks, chips

+    std::set<std::pair<std::size_t, std::size_t>> recorded;     // position pairs, lower index first
+    for (const ResolvedEdge& edge : placement.layout.resolved_edges) {
+        const std::size_t i = placement.layout.node_to_submesh.at(edge.src);
+        const std::size_t j = placement.layout.node_to_submesh.at(edge.dst);
+        const std::size_t p = positions.at(i), q = positions.at(j);          // the two positions, for position_links_
+        const FabricNodeId exit = submeshes[i].chips.at({edge.exit_row, edge.exit_col});   // the resolver's link
+        const FabricNodeId entry = submeshes[j].chips.at({edge.entry_row, edge.entry_col});
+        if (submeshes[i].mesh == submeshes[j].mesh) {
+            placement.channels_by_resolved_edge.emplace_back();                 // intra-mesh fabric; the pairing does not touch it
+            continue;
+        }
+        std::vector<RequiredCrossing::Link> links{{exit, entry, channels_.at({exit, entry})}};   // the resolver's
+        // Open question 25. Option 1 (shown): also every other link between the two positions, which the resolver
+        // could have chosen but did not check. Option 2: skip this loop, so R carries only the chip pair the layout
+        // determined, with every channel on it.
+        for (const LinkPair& link : position_links_.at({p, q})) {   // every other path between the two positions
+            const FabricNodeId a = submeshes[i].chips.at({link.exit_row, link.exit_col});
+            const FabricNodeId b = submeshes[j].chips.at({link.entry_row, link.entry_col});
+            if (a != exit || b != entry) {
+                links.push_back({a, b, channels_.at({a, b})});
+            }
+        }
+        auto& channels = placement.channels_by_resolved_edge.emplace_back();   // the per-edge view: the same channels, flat
+        for (const RequiredCrossing::Link& link : links) {
+            std::ranges::copy(link.channels, std::back_inserter(channels));
+        }
+        if (recorded.insert(ordered(p, q)).second) {               // an edge and its loopback share one crossing
+            required_links.push_back(RequiredCrossing{
+                .pipeline = pipeline,
+                .stages = {edge.src, edge.dst},
+                .meshes = {submeshes[i].mesh, submeshes[j].mesh},
+                .min_channels = edges_between(in.edges, edge.src, edge.dst),
+                .links = std::move(links)});
+        }
+    }
+}
+
+// Each crossing needs min_channels bound between its two positions, and they all come out of the MGD connection's
+// count: the same sum Tier B checks per projection, here on the carve.
+std::optional<std::string> PipelinePlacementEnumerationSession::first_over_demand(
+    const std::vector<RequiredCrossing>& required_links) const {
+    std::map<std::pair<MeshId, MeshId>, std::uint32_t> needed;
+    for (const RequiredCrossing& crossing : required_links) {
+        if (const std::size_t available = channel_count(crossing); available < crossing.min_channels) {
+            return fmt::format("pipeline '{}': stages {} and {} need {} channels and the carve joins their positions "
+                               "with {} on {} links", crossing.pipeline, crossing.stages.first, crossing.stages.second,
+                               crossing.min_channels, available, crossing.links.size());
+        }
+        needed[ordered(crossing.meshes.first, crossing.meshes.second)] += crossing.min_channels;
+    }
+    for (const auto& [meshes, count] : needed) {
+        if (const std::uint32_t allowed = channel_demand_.at(meshes); count > allowed) {
+            return fmt::format("meshes {} and {}: the layout's crossings need {} channels, the MGD connection has {} "
+                               "(strict + relaxed)", meshes.first, meshes.second, count, allowed);
+        }
+    }
+    return std::nullopt;
+}

 // pipeline_builder.cpp, anonymous namespace

+std::uint32_t edges_between(const std::vector<EdgeInputTuple>& edges, const std::string& a, const std::string& b);
+                                                                // either direction, repeats included (Tier B's count)
+std::size_t channel_count(const RequiredCrossing& crossing);      // channels over all its links`,
      },
    ],
    tests: [
      "The rings blaze's planner rejects (4x4 with 13 slices, 4x6, 4x8): accepted, carve verified.",
      "mixed4 from the doc: (P,R) accept; (P,Q) and Case B reject. On (P,R): four stage slots with 8, 16, 16, 8 chips; submesh_links has the A2-B2 and A0-B0 pairs both ways; channels_by_resolved_edge lists every channel between s1's and s2's positions (A2-B2 first) and between s3's and s0's (B0-A0 first), and nothing for the two edges inside a mesh; R is two crossings with those channels.",
      "Counting and area failures (5x5 with five 2x2, 6x6 with area 37) rejected with pipeline_carve before any solve.",
      "A fork (one stage with two returning branches) and two meshes joined by cables.",
      "Two pipelines in one MGD that both have an s0: two placements, keyed by pipeline name, and one R.",
      "First carve passes the resolver on every benchmark case (the 101 of footprint_then_resolver).",
      "A slot spanning two host slices gets slot-local (row, col) starting at (0, 0); every crossing's channels_by_resolved_edge entry starts with the channels between the two ASICs its resolved edge names.",
      "Each R channel's connection_hash equals std::hash<ExitNodeConnection> of the PSD entry, and the hashes equal today's gather on the SC16 and galaxy PSDs.",
            "An edge and its loopback over the same two positions list the same channels; R holds one entry with min_channels 2, and the demand check counts 2 (Tier B's count).",
      "Two copies of one edge across meshes: min_channels 2; a carve whose two positions share 1 channel is rejected with pipeline_binding, naming the pipeline and stages.",
      "Two positions joined on chip 0 and chip 1 (the lowest-chip-id example): the crossing's entry holds both chip pairs' channels, the resolver's first; the demand check counts 1.",
      "Phase 1 and Phase 2 on the same machine: the same channels_by_resolved_edge by (ASIC, channel); the rank bindings hold no channel.",
      "A ring across two meshes over a count-1 connection: the layout crosses twice; next() returns pipeline_binding with the pair in error_message, and no pipelines.",
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
    goal: "Wire the session into the mapper. The enumerator already holds logical_, with the stage graph, each stage's shape and edges, and the host grids (P5), so it copies nothing: after every intra-mesh success its private place_pipelines builds a PipelinePlacementEnumerationSession for the seating and takes one next() (P5, P6); it excludes rejected seatings (C4) and returns the accepted PipelineLayoutResult in its result; TopologyMapper saves it on rank 0 and broadcasts R. The mapper side holds no pipeline logic beyond that call.",
    files: [
      "tt_metal/fabric/topology_mapper_utils.cpp",
      "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
      "tt_metal/fabric/topology_mapper.cpp",
      "tools/scaleout/src/generate_rank_bindings.cpp",
      "tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper_utils.cpp",
    ],
    steps: [
      "TopologyMappingConfig::check_pipelines (default true); TopologyMapper sets it false under the kill switch (4.7).",
      "Enumerator constructor: unchanged. logical_ = build_logical_multi_mesh_adjacency_graph(merged) already fills pipeline_stage_graph_, pipeline_stages_ and mesh_host_grids_ (P5), so the enumerator needs no pipeline members of its own and the merged MGD stays a local. Nothing else goes through the config, so every overload and generate_rank_bindings get it the same way.",
      "next(): after intra-mesh success, when check_pipelines is set and logical_.pipeline_stage_graph_ has nodes (empty by default; nodes exactly when the MGD has pipelines), call the private place_pipelines(full): one session, one next(). Accept: full.pipeline = the PipelineLayoutResult. Reject (C4): its failure_stage and error_message go onto full with success = false; then last_failed_ and the next seating (the SAT session never yields a seating twice); never add_forbidden_constraint. TODO, not in the first implementation: exclude_mapping(seating) on the seat set.",
      "Record rejections and solve time in the placement stats, so the sweep and CI show whether P10 is needed. TODO, not in the first implementation: cap rejections per run (kMaxPipelineRejections, proposed 256, overridable for sweeps) and report the last reason.",
      "Turn unique_shapes off when a pipeline is present (runtime overloads and --all-solutions): swapping meshes between the same seats can change the pipeline answer.",
      "Result: TopologyMappingResult::pipeline; split_mapping_to_local_parts remaps it per part.",
      "TopologyMapper stores mapping_result.pipeline on rank 0 (pipeline_mapping_, get_pipeline_mapping()). R is copied to pipeline_required_links_ and broadcast with the chip info, because every rank checks it after the pairing (P9).",
      "No feedback to the SAT session in the first version (TODO: whole-seating exclusion); the exit-node retry is P15 and the seat-pair clause P13.",
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
+    std::optional<PipelineLayoutResult> pipeline;          // output: per pipeline chips, layout and channels, and R
 };

 class MultiMeshSolutionEnumerator {
     ...
     std::unique_ptr<SatPlacementEnumerationSession> placement_session_;
     LogicalMultiMeshGraph logical_;                        // today; with P5 it carries the stage graph, shapes and host grids too
+    bool place_pipelines(TopologyMappingResult& full);   // one session, one next()
 };

 class TopologyMapper {
     ...
+    const std::optional<PipelineLayoutResult>& get_pipeline_mapping() const { return pipeline_mapping_; }
+    // R: one RequiredCrossing per crossing (P6): every link that can carry it, each with its channels, and
+    // min_channels, the minimum to bind across them.
+    const std::vector<RequiredCrossing>& get_pipeline_required_links() const {
+        return pipeline_required_links_;                                // every rank
+    }
 private:
+    std::optional<PipelineLayoutResult> pipeline_mapping_;             // rank 0, from mapping_result.pipeline
+    std::vector<RequiredCrossing> pipeline_required_links_;  // every rank: rank 0's R, broadcast
 };`,
      },
      {
        path: "tt_metal/fabric/topology_mapper_utils.cpp  (MultiMeshSolutionEnumerator: constructor, next and place_pipelines)",
        mode: "diff",
        language: "cpp",
        src: `
 // init_from_parts: today's line, unchanged; with P5 it also fills the stage graph, each stage's
 // shape and edges, and the host grids
     logical_ = build_logical_multi_mesh_adjacency_graph(merged);

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
+            if (config_.check_pipelines && !logical_.pipeline_stage_graph_.get_nodes().empty() &&   // sets full.pipeline on accept
+                !place_pipelines(full)) {
+                last_failed_ = full;                                      // C4: success, failure_stage, error_message already set
+                // TODO (not in the first implementation): a rejection cap (kMaxPipelineRejections) and
+                // exclude_mapping(seating). next() already never yields a seating twice (remember_yielded).
+                continue;
+            }
             ++emitted_;
             return split_mapping_to_local_parts(full, per_part_local_to_global_mesh_ids_);   // remaps pipeline too
         }
         // intra-mesh failure: add_forbidden_constraint(failed seat), as today
     }

// topology_mapper_utils.cpp, MultiMeshSolutionEnumerator::place_pipelines (private)
+// One session, one next(): a carve, its layouts, their channels and R, or why not. Accept sets full.pipeline;
+// reject copies the reason onto full, as an intra-mesh failure does. C4 stays with the caller.
+bool MultiMeshSolutionEnumerator::place_pipelines(TopologyMappingResult& full) {
+    PipelinePlacementEnumerationSession pipelines(
+        logical_, *physical_system_descriptor_, full);
+    PipelineLayoutResult layouts = pipelines.next();   // no second carve (open question 24)
+    full.stats.pipeline_solve_seconds += std::chrono::duration<double>(layouts.solve_time).count();
+    if (!layouts.success) {
+        full.success = false;
+        full.stats.failure_stage = layouts.failure_stage;   // pipeline_carve | pipeline_layout | pipeline_binding
+        full.error_message = layouts.error_message;
+        return false;
+    }
+    full.pipeline = std::move(layouts);
+    return true;
+}`,
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
    goal: "Have the control plane bind, before any other link, at least min_channels of every crossing in R (the number the topology mapper says it needs: the pipeline edges between its two stages, recorded and checked against demand in P6), the resolver's chip pair first, and more when the budget allows; and fail on every rank together if a crossing gets fewer. IMPORTANT: R is one entry per crossing, each with every channel of every link between its two positions and its min_channels.",
    files: [
      "tt_metal/fabric/topology_mapper.cpp",
      "tt_metal/fabric/control_plane.cpp",
    ],
    steps: [
      "No PortDescriptor change: each rank sends rank 0 a proto with only hash, src chip and dst chip, and only rank 0 has R. Rank 0 passes R's hashes to pair_logical_intermesh_ports(port_descriptors, required_crossings).",
      "Broadcast R with the chip info (TopologyMapper): each crossing's channels, min_channels and names. validate_requested_intermesh_connections then checks on every rank that every crossing in R has at least min_channels distinct hashes of its channels in the broadcast AnnotatedIntermeshConnections (each bound channel appears once per direction).",
      "Phase 0 in pair_logical_intermesh_ports: sum min_channels per boundary, then for each crossing place its links in R's order (the chip pairs holding one of its hashes, the resolver's first) with its boundary's Z rule until min_channels are bound. Each link is trimmed to the budget left after the other crossings' minimums, so it keeps more than this crossing's minimum when the budget allows. Every channel of those links is in the crossing's R entry, so whichever the trim keeps (today it keeps the lowest hash values) serves it. Channels an earlier crossing bound count. Never TT_FATAL on rank 0: the other ranks are already waiting in recv for the broadcast. A crossing left short fails the check after the broadcast, on every rank together (pipeline_binding).",
      "Fill after phase 0: today's order, up to the MGD's count, so a crossing can end up with more than its min_channels when there are channels left; that is fine.",
      "Single host: phase 0 in generate_intermesh_connections_on_local_host (min_channels of each crossing in R first, matched by src_chan and dst_chan, then today's greedy for the rest), or a warning until it lands; the same validate rule runs after it.",
      "No rerun over the bound links: once every crossing in R has its min_channels bound, the witness carve is valid over the bound set.",
    ],
    code: [
      {
        path: "what the control plane reads: TopologyMapper::get_pipeline_required_links() (R, P6 and P8)",
        mode: "plain",
        language: "cpp",
        src: `
// P6: one crossing: a pair of stage positions on two meshes that pipeline edges join. It lists every link (chip
// pair) between the two positions, each with every PSD channel between its two chips; binding min_channels of
// them, on any of the links, keeps the mapping working. The control plane reads it in P9.
struct RequiredCrossing {
    struct Link {
        FabricNodeId exit;                               // chip on meshes.first
        FabricNodeId entry;                              // chip on meshes.second
        EthChannels channels;                            // every PSD channel between the two chips, lowest
                                                         // source channel first
    };
    std::string pipeline;
    std::pair<std::string, std::string> stages;          // as in resolved_edges; stages.first sits on meshes.first
    std::pair<MeshId, MeshId> meshes;
    std::uint32_t min_channels = 1;                      // the minimum to bind, across all links: the pipeline edges
                                                         // between the two stages, either direction (a repeated
                                                         // edge or a loopback is another channel; Tier B's count)
    std::vector<Link> links;                             // links[0] is the resolver's chip pair (the one the layout
                                                         // determined). Open question 25: option 1 (the code below)
                                                         // adds every other link between the two positions, in
                                                         // chip-id order; option 2 keeps links[0] only
};

// TopologyMapper, every rank (rank 0's R, broadcast with the chip info)
const std::vector<RequiredCrossing>& get_pipeline_required_links() const;

// Example, the ring of section 6 (s0 -> s1 -> s2 -> s3, loopback s3 -> s0; s0, s1 on M0 and s2, s3 on M1):
//   { stages {s1, s2}, meshes {M0, M1}, min_channels 1,
//     links { {A2, B2, [every A2-B2 channel]}, then every other cabled chip pair between s1's and s2's slices } }
//   { stages {s3, s0}, meshes {M1, M0}, min_channels 1,
//     links { {B0, A0, [every B0-A0 channel]}, then every other cabled chip pair between s3's and s0's slices } }`,
      },
      {
        path: "tt_metal/fabric/control_plane.cpp  (pair_logical_intermesh_ports)",
        mode: "diff",
        language: "cpp",
        src: `
+    // Phase 0: every crossing in R gets at least min_channels bound first, on crossing.links in order (the resolver's
+    // chip pair first), so the fill cannot strand it. A placed link keeps every channel the budget allows beyond the
+    // minimum, as long as the other crossings' minimums still fit; today's phases then fill the rest.
+    // No throw here: rank 0 must still broadcast; the check after the broadcast fails on every rank.
+    auto still_needed = min_channels_per_boundary(order, required_crossings);   // per boundary: sum of min_channels
+    for (const RequiredCrossing& crossing : required_crossings) {
+        std::uint32_t bound = count_bound(order, crossing);           // bound for a crossing before this one
+        for (const RequiredCrossing::Link& candidate : crossing.links) {   // the resolver's chip pair first
+            if (bound >= crossing.min_channels) {
+                break;
+            }
+            // The pairing's link for this chip pair: its hashes are the gathered ones among candidate.channels.
+            auto* link = find_link(order, candidate.exit, candidate.entry);
+            if (link == nullptr || link->placed) {                    // not gathered (down), or placed already
+                continue;
+            }
+            const auto boundary = boundary_of(*link);
+            const std::uint32_t short_by = crossing.min_channels - bound;
+            // Trim to the budget minus the other crossings' minimums; fail if that leaves fewer than this link can give.
+            if (has_budget_after_trim(boundary, *link, /*reserve=*/still_needed[boundary] - short_by,
+                                      /*at_least=*/std::min<std::uint32_t>(short_by, link->connection_hashes.size())) &&
+                (is_marked_z(boundary) ? place_link(*link, /*want_z=*/true)
+                                       : (place_link(*link, /*want_z=*/false) || place_link(*link, /*want_z=*/true)))) {
+                const auto kept = static_cast<std::uint32_t>(link->connection_hashes.size());
+                account_placed(boundary, *link);
+                still_needed[boundary] -= std::min(kept, short_by);
+                bound += kept;
+            }
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
+            port_descriptors, topology_mapper_->get_pipeline_required_links());
     }
     this->forward_intermesh_connections_from_controller(intermesh_connections);   // rank 0 -> every rank
     // each rank: hash -> its own ETH channel and peer from its PSD copy (cable_lookup), as today

 // validate_requested_intermesh_connections, every rank, same broadcast list
+    // Every crossing in R must have at least min_channels of its channels bound; more is fine. Each bound channel
+    // appears once per direction, so count distinct hashes.
+    for (const RequiredCrossing& crossing : topology_mapper_->get_pipeline_required_links()) {
+        const std::uint32_t bound = count_bound_hashes(intermesh_connections, crossing);
+        TT_FATAL(bound >= crossing.min_channels,
+                 "pipeline_binding: pipeline '{}', stages {} and {} (meshes {} and {}) need {} channels between "
+                 "their positions and {} are bound",
+                 crossing.pipeline, crossing.stages.first, crossing.stages.second, crossing.meshes.first,
+                 crossing.meshes.second, crossing.min_channels, bound);
+    }`,
      },
    ],
    tests: [
      "RELAXED count 2 with four candidates where the lowest chip ids share a slice pair: today Case B, after P9 Case A.",
      "R empty (no pipeline): AnnotatedIntermeshConnections byte-identical to today.",
      "Multi-host: R broadcast with the chip info is identical on every rank; validate_requested_intermesh_connections passes everywhere.",
            "Each crossing in R has at least min_channels distinct channels in the broadcast list, and every rank resolves them to channels recorded in R.",
      "A crossing whose positions are joined on chip 0 and chip 1, budget 1: phase 0 binds the resolver's chip pair; if it cannot be placed, the other one; either passes the check.",
      "Two crossings on one boundary with budget 3, the first needing 2 (an edge and its loopback) with 3 channels, the second needing 1: the first keeps 2, the second 1; the first does not take the second's budget.",
      "One crossing needing 1 with 4 channels, budget 4: phase 0 binds 1 or more, the fill binds the rest; the check passes with 4 bound.",
      "A crossing needing 2 whose resolver chip pair has 1 channel and another link has 3: phase 0 binds the resolver's channel, then one on the other link.",
      "A crossing needing 2 with only 1 channel bound (the other link's exit chip already faces a third mesh): every rank fails the check together, naming the pipeline, the stages, the meshes, 2 needed and 1 bound.",
      "Phase 0 cannot place any link of a crossing (every exit chip already facing a third mesh): no hang; every rank fails the check together, naming the crossing and its edge.",
      "A crossing with one channel down: phase 0 binds another gathered channel of its links; with all of them down, every rank fails the check together (open question 20).",
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
-            if (config_.check_pipelines && !logical_.pipeline_stage_graph_.get_nodes().empty() &&
-                !place_pipelines(full)) {
+            bool accepted = !config_.check_pipelines || logical_.pipeline_stage_graph_.get_nodes().empty() ||
+                place_pipelines(full);
+            for (int retry = 0; !accepted && retry < kMaxLayoutRetries; ++retry) {
+                auto blocks = slice_signature_blocks(full, mesh_graph_);   // full's layout; empty when fixed by bindings
+                if (blocks.empty()) {
+                    break;
+                }
+                auto again = complete_intra_mesh_for_placement(seating, /*extra=*/blocks, ...);
+                if (!again.success) {
+                    break;
+                }
+                full = std::move(again);
+                accepted = place_pipelines(full);
+            }
+            if (!accepted) {`,
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
      "tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp",
      "tt_metal/fabric/topology_mapper_utils.cpp",
      "tt_metal/fabric/pipeline_builder.cpp",
      "tt-blaze: pipeline_builder/graph.py, host_edge_planner.py, blaze/models/cli_options.py",
    ],
    steps: [
      "In build_logical_multi_mesh_adjacency_graph, leave edges with needs_fabric = false out of LogicalMultiMeshGraph::pipeline_stage_graph_ and put them in a new pipeline_soft_stage_graph_ beside it (AdjacencyGraph<PipelineStageNode>). PipelinePlacementEnumerationSession passes the second to its solver as soft target edges (allow unmatched). P14 already keeps the host I/O stage's outgoing edge as needs_fabric, as blaze never cuts it.",
      "The host cuts are the soft edges the solve's MappingResult leaves unmatched; next() drops them from the resolver's edges. No new carve type.",
            "Resolve each fabric chain separately (blaze's _resolve_chains_separately), and record R only for matched crossings.",
      "Preference for fewest cuts and blaze's tie-breaks: open question 16.",
      "Drop P14's RELAXED-as-STRICT warning: Tier C now treats edges with needs_fabric = false as soft.",
      "tt-blaze, same change: Edge gets a policy (STRICT | RELAXED); _group_walk skips alignments that cut a STRICT edge instead of only '0 in cuts'; the fork planner cuts only RELAXED returns; --pipeline-builder flexible becomes a shortcut that marks every edge RELAXED.",
    ],
    code: [
      {
        path: "tt_metal/fabric/pipeline_builder.cpp  (the session's next())",
        mode: "new",
        language: "cpp",
        src: `
// build_logical_multi_mesh_adjacency_graph: pipeline_stage_graph_ keeps only fabric edges; the rest go to a
// second graph, pipeline_soft_stage_graph_, beside it. RELAXED edges may stay unmatched (carried over host).
for (const PipelineDescriptorData& pipeline : mesh_graph_descriptor.pipeline_descriptors()) {
    for (const PipelineDescriptorData::Edge& e : pipeline.edges) {
        const PipelineStageNode src{pipeline.name, e.src}, dst{pipeline.name, e.dst};
        (e.needs_fabric ? stages : soft)[src].push_back(dst);   // one way, as P5; P14: STRICT, or out of the host I/O stage
    }
}
// ... the session's solver gets soft_adj through set_allow_unmatched_target_edges (#56762)
const auto host_cuts = unmatched_soft_edges(carve);   // inside next(), from the solver's MappingResult

// next(): host cuts are dropped; resolve each fabric chain on its own.
for (const auto& chain : fabric_chains(edges_minus(pipeline.edges, host_cuts))) {
    auto layout = resolve_chain(chain, slots_of(m), links);
    if (!layout) { return failed("pipeline_layout", layout.error()); }
    record_channels(pipeline, in, *layout, positions, R);   // matched crossings only
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
      "Read TopologyMappingResult::pipeline from the enumerator and read each stage's ranks from it: for each PipelinePlacement, submeshes[node_to_submesh.at(stage)] gives the stage's mesh and the MGD host ranks that own it (PipelineSubmesh::host_ranks); ranks owning no stage are idle.",
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
      "The enumerator's place_pipelines gains an optional out-parameter, std::vector<std::pair<MeshId, MeshId>>* seat_pair_proofs (default nullptr), so no check-result type comes back. The P8 loop turns each proof into add_seat_pair_nogood before counting the rejection; false from the session ends the run with the existing could-not-fit error.",
      "No k_AB producer. Tier B saves nothing in the first version, so cables-between-seats < k_AB would need P10's stored PipelineMappingConstraints::min_crossings, and with P10 in, allow_seat_pair already keeps those seat pairs out of the first solve.",
      "The producer, a C2 pre-check failure on mesh m, only when all three hold: the verdict holds for every projection; m's only MGD neighbour is n, so all of m's cables land on n's seat; and m's slice signature is fixed by hard constraints (P15's skip test). Resolver failures never produce a proof: one resolver run is one carve.",
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
+           std::vector<std::pair<MeshId, MeshId>> proofs;
            if (config_.check_pipelines && !logical_.pipeline_stage_graph_.get_nodes().empty() &&
-               !place_pipelines(full)) {
+               !place_pipelines(full, &proofs)) {   // C4
                last_failed_ = full;
+               for (const auto& [mesh_a, mesh_b] : proofs) {
+                   if (!placement_session_->add_seat_pair_nogood(
+                           mesh_a, seat_asics(seating, mesh_a), mesh_b, seat_asics(seating, mesh_b))) {
+                       return give_up();
+                   }
+                   ++placement_stats_->pipeline_seat_pair_nogoods;
+               }
                continue;                         // P8's TODO: the cap and exclude_mapping(seating)
            }

 // topology_mapper_utils.hpp, MultiMeshSolutionEnumerator (private)
-    bool place_pipelines(TopologyMappingResult& full);
+    bool place_pipelines(TopologyMappingResult& full,
+                         // Mesh pairs whose two seats in this seating fail for every other seat of every other mesh.
+                         std::vector<std::pair<MeshId, MeshId>>* seat_pair_proofs = nullptr);
`,
      },
      {
        path: "tt_metal/fabric/topology_mapper_utils.cpp",
        mode: "diff",
        language: "cpp",
        src: `
 // topology_mapper_utils.cpp, MultiMeshSolutionEnumerator::place_pipelines
 // A producer may add a proof only when its verdict reads nothing but the two seats' footprints,
 // layouts fixed by hard constraints, and facts true in every projection. The one below reads only
 // orientation-free facts, so matching seats by ASIC set (as seats_matching does) is sound.

     PipelinePlacementEnumerationSession pipelines(
         logical_, *physical_system_descriptor_, full);   // C1, pre-checks
     PipelineLayoutResult layouts = pipelines.next();
     if (!layouts.success) {
+        // The session keeps the pre-check's mesh and scope (P13 adds pre_check_failure(), public because the caller reads it).
+        const auto& pre = pipelines.pre_check_failure();
+        // C2 pre-check: mesh m's segment test failed for every projection, m's only MGD neighbour is
+        // n (so all of m's cables go to n's seat), and m's slice signature is fixed by hard
+        // constraints (P15's skip test). Then no other seat of any other mesh changes it.
+        if (seat_pair_proofs != nullptr && pre.has_value() && pre->mesh.has_value() && pre->every_projection) {
+            const MeshId m = *pre->mesh;
+            if (const auto n = only_mgd_neighbour(logical_, m); n.has_value() && slice_signature_fixed(full, m)) {
+                seat_pair_proofs->push_back({m, *n});
+            }
+        }
         full.success = false;                               // failure_stage and error_message too, as in P8
         return false;
     }
`,
      },
    ],
    tests: [
      "Session: a nogood on (M0 -> P, M1 -> Q) blocks every seating with both, offers every other one (brute-force compare on a 3-mesh, 4-seat PSD), and survives a rebuild after a pool growth.",
      "Session: a nogood that leaves no seating returns false and is popped; the session still enumerates as before.",
      "Twins: ALL_TO_ALL of identical meshes with twin windowing on. A nogood on M0 also blocks its twins; the set of accepted seatings equals the brute-force set (no working orbit lost).",
      "Soft seams: the nogood still holds on the soft-seam rung of the solve ladder.",
      "Producer: pod8 with every host bound: (P, U) fails on M1's corners -> nogood (M1 on U, M0 on P); unbound hosts -> no proof, P8 exclusion only.",
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
      "Descriptor data: PipelineDescriptorData::Edge gains bool relaxed and bool needs_fabric (the header only forward-declares proto::Policy). needs_fabric = STRICT, or RELAXED out of the host I/O stage, which blaze's host-cut planner never cuts. One rule, set once in populate_pipeline_descriptors (Tier A has found the root), read by Tier B, Plan 1 and Tier C (P11) from the same record. There is no Tier C edge type to copy it into.",
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
     uint32 dst = 2;

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
+            const bool relaxed = edge.has_policy() && edge.policy() == proto::Policy::RELAXED;
+            record.edges.push_back(PipelineDescriptorData::Edge{
+                .src = edge.src(),
+                .dst = edge.dst(),
+                .loopback = edge.loopback(),
+                .relaxed = relaxed,
+                .needs_fabric = !relaxed || edge.src() == host_io_stage});   // the one root Tier A found

 // mesh_graph_descriptor.hpp (proto::Policy is only forward-declared there, so a bool):
-// Edges stay directional pairs of stage ids, each with its loopback mark. Omitted and explicit
-// STRICT both survive validation, so a stored edge has no policy of its own.
+// Edges stay directional pairs of stage ids, each with its loopback mark. relaxed is
+// policy: RELAXED; omitted is STRICT.
 struct PipelineDescriptorData {
     struct Edge {
         PipelineStageId src = 0;
         PipelineStageId dst = 0;
         bool loopback = false;  // closes a cycle; the other edges have none
+        bool relaxed = false;
+        // STRICT, or RELAXED out of stage 0: blaze's host-cut planner never cuts that
+        // (_group_walk skips cuts at stage 0). The one rule Tier B, Plan 1 and (P11) Tier C all read.
+        bool needs_fabric = true;
     };`,
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
+        if (!edge.needs_fabric) {
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
+    for (const PipelineDescriptorData& pipeline : merged.pipeline_descriptors()) {
+        const auto count = std::ranges::count_if(pipeline.edges, &PipelineDescriptorData::Edge::relaxed);
+        if (count > 0) {
+            log_warning(tt::LogFabric,
+                        "pipeline '{}': {} RELAXED edge(s) are checked as STRICT until host cuts are supported",
+                        pipeline.name, count);
+            relaxed_checked_as_strict_ = true;
+        }
+    }

 // MultiMeshSolutionEnumerator::next, give_up(): say why a RELAXED pipeline found nothing.
+    if (relaxed_checked_as_strict_) {
+        last_failed_->error_message += " (RELAXED pipeline edges are checked as STRICT until host cuts are supported)";
+    }`,
      },
    ],
    tests: [
      "EdgePolicyDoesNotParse becomes EdgePolicyParses: policy RELAXED stores relaxed = true; STRICT and omitted store false. policy: INVALID_POLICY is rejected by Tier A, naming the edge.",
      "UnknownFieldElsewhereStillParses and ShippedMeshGraphDescriptorsStillParse unchanged; every P1-P10 golden gives the same Tier B result.",
      "Descriptor data: ring with s3 -> s0 RELAXED: needs_fabric false. s0 -> s1 RELAXED: needs_fabric true (stage 0).",
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
      "Call collect_inter_mesh_candidates(mesh_graph, psd, full.fabric_node_to_physical, options) per seating from MultiMeshSolutionEnumerator::next, with the same InterMeshCandidateOptions as the final call; delete inter_mesh_cables.",
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
  { id: "M5 Later", text: "Not in the first version. RELAXED: edge policy in the schema, then host cuts in Tier C. Candidate table: C1 and R switch from their own PSD read (inter_mesh_cables) to the policy plan's collect_inter_mesh_candidates (part 9). M1-M4 are complete without it: every edge is STRICT, and C1 and R read the PSD." },
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
  ["topology_mapper_utils.hpp", "PipelineStageNode; three LogicalMultiMeshGraph fields (pipeline_stage_graph_, all pipelines merged; pipeline_stages_, shape and loopback marks per stage; mesh_host_grids_), PipelineStageInfo, PipelinePlacementEnumerationSession (the positions, the footprint solve and the resolver call), PipelinePlacement and PipelineLayoutResult (P5); their channel fields, channels_by_resolved_edge and required_links (R), and the session's PSD input (P6); TopologyMappingConfig::check_pipelines; TopologyMappingResult::pipeline; the enumerator's private place_pipelines (P8)", "P5, P6, P8"],
  ["topology_mapper_utils.cpp", "build_logical_multi_mesh_adjacency_graph fills the three logical fields (P5); the next() hook and place_pipelines (one session, one next()), exclude_mapping and the rejection cap (P8)", "P5, P8"],
  ["pipeline_builder.cpp", "per seating: the session's methods (positions, pre-checks, constraints, the footprint solve, the resolver on every carve; P6: inter_mesh_cables, each edge's channel and R, the demand check)", "P5, P6"],
  ["topology_mapper.cpp", "kill switch; the mapping saved on rank 0; R broadcast with the chip info", "P8"],
  ["control_plane.cpp", "phase 0 binds at least min_channels per crossing in R; every rank checks each got them", "P9"],
  ["test_pipeline_builder.cpp / test_topology_mapper_utils.cpp", "the session (positions, carves, channels and R) on CPU PSD fixtures / the logical graph's pipeline fields and the hook", "P5, P6, P8"],
];

const TIER_C_REUSED: string[][] = [
  ["stages, labels, edges, loopback", "PipelineDescriptorData (MGD, Tier A), read once by the builder"],
  ["a stage's shape and loopback marks", "PipelineStageInfo: MeshShape from the dims, loopback_to (dst ids); the edges are the one-way stage graph"],
  ["a stage's id", "PipelineStageId (fabric_types.hpp), an integer"],
  ["where the stage graph, stage shapes and host grids live", "LogicalMultiMeshGraph, beside today's mesh graphs; one stage graph for all pipelines"],
  ["where the positions and the solve live", "PipelinePlacementEnumerationSession, one per seating, as SatPlacementEnumerationSession holds the seat graph; PhysicalMultiMeshGraph is unchanged"],
  ["host slices per mesh", "MeshContainer<MeshCoordinateRange>, shaped as the host grid; each cell the MGD's get_coord_range for that host"],
  ["position nodes", "std::size_t indices"],
  ["a position", "a MeshCoordinateRange of the host grid while C1 runs; then its index"],
  ["a position's chips", "std::vector<detail::InternalChip> (resolver input: FabricNodeId, slot-local row and col)"],
  ["a position's mesh and host ranks", "MeshId, MeshHostRankId (in PipelineSubmesh, with the chips keyed by SlotCoord)"],
  ["links between positions", "detail::DirectLinks (resolver input)"],
  ["the carve, inside the session", "MappingResult<PipelineStageNode, std::size_t> (solver); next() returns the layouts, not the carve"],
  ["a pipeline's layout", "GraphLayoutResult (resolver output)"],
  ["a reject", "TopologyMappingResult: success, error_message, stats.failure_stage"],
  ["a channel in R", "ExitNodeConnection (PSD)"],
];

const TIER_C_FLOW = `
once per run (topology_mapper_utils.cpp, enumerator constructor)
  logical_ = build_logical_multi_mesh_adjacency_graph(merged)   today; P5: + pipeline_stage_graph_ (all pipelines),
                                                                pipeline_stages_, mesh_host_grids_

per seating (MultiMeshSolutionEnumerator::next; physical = build_hierarchical_from_flat_graph(flat, seating), today)
  after intra-mesh success:
  place_pipelines(full)                                           P8, topology_mapper_utils.cpp
    PipelinePlacementEnumerationSession pipelines(logical_, psd, full)   P5
    C1    constructor                              positions (they overlap); edges join disjoint,
                                                   bordering ones; cables from the PSD (inter_mesh_cables)
    C2    constructor                              pre-checks (area, lattice, colour and segment);
                                                   logical_.pipeline_stage_graph_ (both ways) onto the
                                                   position graph; required + resource + completeness,
                                                   conflict_cap
          pipelines.next()                         one carve, then the resolver on it per pipeline
                                                   (a throw: pipeline_layout)
            record_channels -> channels_by_resolved_edge, R    the exact PSD channel of every crossing edge (P6)
            first_over_demand                      min_channels per mesh pair <= strict + relaxed (P6)
          PipelineLayoutResult                     per pipeline: submeshes, GraphLayoutResult,
                                                   channels_by_resolved_edge; and R
  reject: full.success = false, failure_stage, error_message;
          next seating (TODO: exclude_mapping and the cap)                        P8 (C4); P15 retries first
  accept: full.pipeline set; TopologyMapper saves it, broadcasts R                P8
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
          The first version is STRICT only: every pipeline edge rides fabric, C1 and R read inter-mesh cables and their channels from the PSD (one read per seating), and M1-M4 need nothing from M5. Arrows are
          hard dependencies. M2 (P5 then P6) is independent of M1, so it can land in parallel with the format work. Every PR
          is written out below with its steps, code and tests; click a box to jump to it.
        </Text>
      </Stack>
      <Stack gap={8}>
        <H2>Tier C at a glance</H2>
        <Text tone="secondary">
          One rule: topology_mapper_utils owns the types and what runs once per run (the stage graph, each stage's shape and edges, and the host grids in the logical graph, the enumerator hook, C4);
          pipeline_builder owns the methods of the one per-seating class, PipelinePlacementEnumerationSession. After every intra-mesh
          success the enumerator builds one and takes one next(): a carve, its layouts, their channels and R, or why not. pipeline_builder.hpp
          gets no Tier C change, so the ttnn ops and nanobind module that include it are unchanged.
        </Text>
        <Table
          headers={["File", "Holds", "PR"]}
          rows={TIER_C_FILES}
        />
        <Text tone="secondary">
          One node type, PipelineStageNode; new fields on LogicalMultiMeshGraph: the merged stage graph, the stage shapes and the
          host grids; and one class, PipelinePlacementEnumerationSession, which owns the positions and the footprint solve (P5).
          PhysicalMultiMeshGraph is unchanged. The stage graph
          is one way and each stage keeps its loopback marks, so nothing per seating reads the MGD. Small new structs for what
          next() returns: PipelineSubmesh (one submesh's mesh, host ranks and chips by (row, col)), PipelinePlacement and
          PipelineLayoutResult (P5) and RequiredCrossing (P6), each a bundle of existing types, plus two names: SlotCoord,
          a slot-local (row, col), and EthChannels, a list of PSD ExitNodeConnections. Everything else already exists:
        </Text>
        <Table
          headers={["Need", "Existing type"]}
          rows={TIER_C_REUSED}
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
// ===========================================================================
// PART I: THE DESIGN
// ===========================================================================

function Bullets({ items }: { items: string[] }) {
  return (
    <Stack gap={4}>
      {items.map((s, i) => (
        <Row key={i} gap={8} align="start">
          <Text size="small" tone="tertiary" style={{ minWidth: 10 }}>
            -
          </Text>
          <Text size="small" tone="secondary">
            {s}
          </Text>
        </Row>
      ))}
    </Stack>
  );
}

function StageBox(props: { x: number; y: number; w?: number; h?: number; label: string; accent?: boolean; muted?: boolean }) {
  const t = useHostTheme();
  const w = props.w ?? 52;
  const h = props.h ?? 30;
  return (
    <g>
      <rect
        x={props.x}
        y={props.y}
        width={w}
        height={h}
        rx={5}
        fill={props.accent ? t.fill.secondary : props.muted ? "none" : t.fill.tertiary}
        stroke={props.accent ? t.accent.primary : props.muted ? t.stroke.tertiary : t.stroke.primary}
        strokeDasharray={props.muted ? "3 3" : undefined}
      />
      <text
        x={props.x + w / 2}
        y={props.y + h / 2 + 4}
        textAnchor="middle"
        fontSize={12}
        fontWeight={props.muted ? 400 : 600}
        fill={props.muted ? t.text.tertiary : t.text.primary}
      >
        {props.label}
      </text>
    </g>
  );
}

// --- Overview --------------------------------------------------------------

function OverviewTab() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>The problem</H2>
        <Text tone="secondary">
          The topology mapper picks which chips each mesh lands on without knowing the pipeline. Blaze is the first to
          check, after the mapping is committed, and fails with "no valid submesh assignment found".
        </Text>
      </Stack>
      <FlowCompare />
      <Stack gap={8}>
        <H2>Where each check runs</H2>
        <TierTimeline />
      </Stack>
      <Grid columns={fit(240)} gap={16}>
        <Callout tone="info" title="Prove a placement exists">
          Tier C accepts a seating once it finds one carve and layout. Blaze still places the pipeline itself.
        </Callout>
        <Callout tone="warning" title="Reject only the impossible">
          Tiers A and B reject only what no mapping could build. Tier C checks one carve per seating.
        </Callout>
        <Callout tone="neutral" title="Use the builder's resolver">
          Every layout rule comes from the same C++ resolver blaze calls at runtime.
        </Callout>
      </Grid>
      <Callout tone="warning" title="First version: STRICT only">
        Every pipeline edge rides a fabric link. RELAXED edges (may go over host MPI) are later work.
      </Callout>
    </Stack>
  );
}

// --- The MGD pipeline section ---------------------------------------------

const FORMAT_FIELDS: string[][] = [
  ["PipelineDescriptor", "name", "Non-empty, unique in the MGD"],
  ["", "stages", "At least one, in declaration order"],
  ["", "edges", "Directed. Repeating (src, dst) asks for another channel on that hop"],
  ["PipelineStage", "id", "Integer, unique in the pipeline; edges refer to it"],
  ["", "label", "Free text, may be empty or repeat"],
  ["", "shape { dims: [ rows, cols ] }", "Chips, in the mesh's orientation. Names no mesh: the mapper picks it"],
  ["PipelineEdge", "src, dst", "Stage ids; src != dst"],
  ["", "loopback", "true on the edge that closes a cycle"],
  ["", "policy", "Omitted or STRICT. RELAXED is rejected for now"],
];

function LoopbackRuleDiagram({ part = "both" }: { part?: "both" | "fork" }) {
  const t = useHostTheme();
  const accent = t.accent.primary;
  const line = { stroke: t.stroke.primary, strokeWidth: 1.2, fill: "none" } as const;
  const back = { stroke: accent, strokeWidth: 1.6, fill: "none", strokeDasharray: "5 3" } as const;
  const both = part === "both";
  return (
    <ScrollX>
      <svg
        width="100%"
        viewBox={both ? "0 0 760 240" : "370 0 390 240"}
        style={{ maxWidth: both ? 760 : 390, minWidth: both ? 600 : 0 }}
      >
        <ArrowDefs id="lb-arrow" color={t.stroke.primary} />
        <ArrowDefs id="lb-back" color={accent} />
        <text x={20} y={16} fontSize={11} fontWeight={600} fill={t.text.secondary}>
          RING (mixed4, gpt_oss)
        </text>
        <text x={46} y={68} textAnchor="middle" fontSize={10.5} fill={t.category.green}>
          host I/O
        </text>
        {["0", "1", "2", "3"].map((id, i) => (
          <g key={id}>
            <StageBox x={20 + i * 90} y={76} label={id} accent={i === 0} />
            {i < 3 ? <line {...line} x1={72 + i * 90} y1={91} x2={110 + i * 90} y2={91} markerEnd="url(#lb-arrow)" /> : null}
          </g>
        ))}
        <path {...back} d="M 316 106 C 316 156, 46 156, 46 110" markerEnd="url(#lb-back)" />
        <text x={181} y={160} textAnchor="middle" fontSize={10.5} fill={accent}>
          3 -&gt; 0  loopback: true
        </text>

        <text x={420} y={16} fontSize={11} fontWeight={600} fill={t.text.secondary}>
          FORK (DFlash router)
        </text>
        <text x={414} y={125} textAnchor="end" fontSize={10.5} fill={t.category.green}>
          host I/O
        </text>
        <StageBox x={420} y={106} label="0" accent />
        {[
          ["1", "2..37", "38"],
          ["39", "40..48", "49"],
        ].map((chain, r) => {
          const y = r === 0 ? 60 : 152;
          return (
            <g key={r}>
              <line {...line} x1={472} y1={121} x2={510} y2={y + 15} markerEnd="url(#lb-arrow)" />
              {chain.map((id, i) => (
                <g key={id}>
                  <StageBox x={512 + i * 84} y={y} w={56} label={id} />
                  {i < 2 ? <line {...line} x1={568 + i * 84} y1={y + 15} x2={594 + i * 84} y2={y + 15} markerEnd="url(#lb-arrow)" /> : null}
                </g>
              ))}
            </g>
          );
        })}
        <path {...back} d="M 708 60 C 708 26, 446 26, 446 104" markerEnd="url(#lb-back)" />
        <text x={600} y={30} textAnchor="middle" fontSize={10.5} fill={accent}>
          38 -&gt; 0  loopback: true
        </text>
        <path {...back} d="M 708 182 C 708 218, 446 218, 446 138" markerEnd="url(#lb-back)" />
        <text x={600} y={232} textAnchor="middle" fontSize={10.5} fill={accent}>
          49 -&gt; 0  loopback: true
        </text>
      </svg>
    </ScrollX>
  );
}

function JointStageDiagram() {
  const t = useHostTheme();
  const slice = (x: number, y: number, label: string) => (
    <g key={label}>
      <rect x={x} y={y} width={100} height={84} fill={t.fill.quaternary} stroke={t.stroke.secondary} strokeDasharray="3 3" />
      <text x={x + 50} y={y + 16} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
        {label}
      </text>
    </g>
  );
  return (
    <svg width="100%" viewBox="0 0 250 210" style={{ maxWidth: 250 }}>
      <text x={20} y={14} fontSize={10.5} fill={t.text.tertiary}>
        8x4 mesh, host_topology 2x2
      </text>
      {slice(20, 24, "rank 0 · 4x2")}
      {slice(120, 24, "rank 1 · 4x2")}
      {slice(20, 108, "rank 2 · 4x2")}
      {slice(120, 108, "rank 3 · 4x2")}
      <rect x={24} y={128} width={192} height={60} rx={6} fill={t.fill.secondary} stroke={t.accent.primary} strokeWidth={1.5} />
      <text x={120} y={152} textAnchor="middle" fontSize={12} fontWeight={600} fill={t.text.primary}>
        stage 7 LMHead [4, 4]
      </text>
      <text x={120} y={170} textAnchor="middle" fontSize={10.5} fill={t.text.secondary}>
        one stage, ranks 2 + 3
      </text>
    </svg>
  );
}

function HeteroDiagram() {
  const t = useHostTheme();
  const accent = t.accent.primary;
  return (
    <ScrollX>
      <svg width="100%" viewBox="0 0 700 150" style={{ maxWidth: 700, minWidth: 560 }}>
        <ArrowDefs id="ht-arrow" color={t.stroke.primary} />
        <ArrowDefs id="ht-back" color={accent} />
        <text x={20} y={30} fontSize={11} fontWeight={600} fill={t.text.secondary}>
          mesh with 4x2 slices
        </text>
        <text x={360} y={30} fontSize={11} fontWeight={600} fill={t.text.secondary}>
          meshes with 1x2 slices
        </text>
        {Array.from({ length: 8 }).map((_, i) => (
          <g key={i}>
            <rect x={20 + i * 40} y={40} width={38} height={40} rx={3} fill={t.fill.secondary} stroke={t.stroke.primary} />
            <text x={39 + i * 40} y={64} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={t.text.primary}>
              {i}
            </text>
          </g>
        ))}
        {Array.from({ length: 32 }).map((_, i) => (
          <rect key={i} x={360 + i * 10} y={52} width={9} height={16} rx={1} fill={t.fill.secondary} stroke={t.stroke.primary} />
        ))}
        <text x={520} y={96} textAnchor="middle" fontSize={10.5} fill={t.text.secondary}>
          stages 8..39, one [1, 2] slice each
        </text>
        <line x1={340} y1={60} x2={358} y2={60} stroke={t.stroke.primary} markerEnd="url(#ht-arrow)" />
        <path d="M 676 68 C 676 136, 39 136, 39 84" fill="none" stroke={accent} strokeWidth={1.6} strokeDasharray="5 3" markerEnd="url(#ht-back)" />
        <text x={360} y={130} textAnchor="middle" fontSize={10.5} fill={accent}>
          39 -&gt; 0  loopback: true
        </text>
      </svg>
    </ScrollX>
  );
}

type MeshRowsProps = {
  meshes: Array<{ name: string; slices: string[] }>;
  stages: Array<{ mesh: 0 | 1; from: number; to: number; label: string; idle?: boolean }>;
  cables: Array<{ a: number; b: number }>;
  note?: string;
};

/** Two meshes as rows of host slices; stages are boxes over whole slices, cables join the rows. */
function MeshRows(props: MeshRowsProps) {
  const t = useHostTheme();
  const sw = 92;
  const sh = 44;
  const x0 = 46;
  const ys = [30, 132] as const;
  const W = x0 + 4 * sw + 16;
  const cx = (i: number) => x0 + i * sw + sw / 2;
  return (
    <svg width="100%" viewBox={`0 0 ${W} 210`} style={{ maxWidth: W }}>
      {props.meshes.map((m, mi) => (
        <g key={m.name}>
          <text x={4} y={ys[mi] + sh / 2 + 4} fontSize={12} fontWeight={600} fill={t.text.secondary}>
            {m.name}
          </text>
          {m.slices.map((s, i) => (
            <g key={s}>
              <rect x={x0 + i * sw} y={ys[mi]} width={sw} height={sh} fill={t.fill.quaternary} stroke={t.stroke.secondary} strokeDasharray="3 3" />
              <text x={cx(i)} y={mi === 0 ? ys[mi] - 6 : ys[mi] + sh + 14} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
                {s}
              </text>
            </g>
          ))}
        </g>
      ))}
      {props.cables.map((c, i) => (
        <line key={i} x1={cx(c.a)} y1={ys[0] + sh} x2={cx(c.b)} y2={ys[1]} stroke={t.text.primary} strokeWidth={2} />
      ))}
      {props.stages.map((s) => (
        <g key={`${s.mesh}-${s.from}`}>
          <rect
            x={x0 + s.from * sw + 3}
            y={ys[s.mesh] + 3}
            width={(s.to - s.from + 1) * sw - 6}
            height={sh - 6}
            rx={5}
            fill={s.idle ? "none" : t.fill.secondary}
            stroke={s.idle ? t.stroke.tertiary : t.accent.primary}
            strokeDasharray={s.idle ? "3 3" : undefined}
          />
          <text
            x={(cx(s.from) + cx(s.to)) / 2}
            y={ys[s.mesh] + sh / 2 + 4}
            textAnchor="middle"
            fontSize={12}
            fontWeight={s.idle ? 400 : 600}
            fill={s.idle ? t.text.tertiary : t.text.primary}
          >
            {s.label}
          </text>
        </g>
      ))}
      {props.note ? (
        <text x={x0} y={204} fontSize={10.5} fill={t.text.tertiary}>
          {props.note}
        </text>
      ) : null}
    </svg>
  );
}

function MachineDiagram() {
  const t = useHostTheme();
  const node = (x: number, y: number, title: string, sub: string) => <SvgBox x={x} y={y} w={130} h={42} title={title} sub={sub} />;
  const edge = (x1: number, y1: number, x2: number, y2: number, label: string, lx: number, ly: number) => (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={t.text.primary} strokeWidth={1.6} />
      <text x={lx} y={ly} textAnchor="middle" fontSize={10.5} fill={t.text.secondary}>
        {label}
      </text>
    </g>
  );
  return (
    <ScrollX>
      <svg width="100%" viewBox="0 0 600 230" style={{ maxWidth: 600, minWidth: 480 }}>
        {edge(150, 46, 235, 101, "P-Q: h0, h3", 170, 86)}
        {edge(150, 186, 235, 129, "P-R: h0, h2", 170, 150)}
        {edge(85, 67, 85, 165, "Q-R: h1", 112, 120)}
        {edge(365, 101, 450, 46, "P-U: 2", 430, 86)}
        {edge(365, 129, 450, 186, "P-V: 2", 430, 150)}
        {node(20, 25, "pod Q", "4x8, hosts h0..h3")}
        {node(20, 165, "pod R", "4x8, hosts h0..h3")}
        {node(235, 94, "pod P", "4x8, hosts h0..h3")}
        {node(450, 25, "board U", "2x4, four 2-chip hosts")}
        {node(450, 165, "board V", "2x4, four 2-chip hosts")}
        <text x={300} y={224} textAnchor="middle" fontSize={10.5} fill={t.text.tertiary}>
          cables run between the named hosts; P-U and P-V land on P.h0 and P.h3
        </text>
      </svg>
    </ScrollX>
  );
}

const RUN_MIXED4_NEW = `
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

# 8 slices declared; the ring needs 1 + 2 + 2 + 1 = 6
pipeline_descriptors {
  name: "mixed4"
  stages { id: 0 shape { dims: [ 4, 2 ] } }   # 1 slice; owns host I/O
  stages { id: 1 shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: 2 shape { dims: [ 4, 4 ] } }   # 2 slices, two ranks
  stages { id: 3 shape { dims: [ 4, 2 ] } }   # 1 slice
  edges { src: 0 dst: 1 }
  edges { src: 1 dst: 2 }
  edges { src: 2 dst: 3 }
  edges { src: 3 dst: 0 loopback: true }       # closes the ring
}`;

const RUN_POD8_NEW = `
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
  instances { mesh { mesh_descriptor: "Big"   mesh_id: 0 } }   # M0: slices A0..A3
  instances { mesh { mesh_descriptor: "Small" mesh_id: 1 } }   # M1: slices T0..T3
  graph_topology { layout_type: ALL_TO_ALL channels { count: 2 policy: RELAXED } }
}
top_level_instance { graph { graph_descriptor: "G" graph_id: 0 } }

# 8 slices declared; the ring needs 8
pipeline_descriptors {
  name: "pod8"
  stages { id: 0 shape { dims: [ 4, 2 ] } }   # binds only to Big; owns host I/O
  stages { id: 1 shape { dims: [ 4, 2 ] } }
  stages { id: 2 shape { dims: [ 4, 2 ] } }
  stages { id: 3 shape { dims: [ 4, 2 ] } }
  stages { id: 4 shape { dims: [ 1, 2 ] } }   # binds only to Small
  stages { id: 5 shape { dims: [ 1, 2 ] } }
  stages { id: 6 shape { dims: [ 1, 2 ] } }
  stages { id: 7 shape { dims: [ 1, 2 ] } }
  edges { src: 0 dst: 1 }
  edges { src: 1 dst: 2 }
  edges { src: 2 dst: 3 }
  edges { src: 3 dst: 4 }                      # crosses Big -> Small
  edges { src: 4 dst: 5 }
  edges { src: 5 dst: 6 }
  edges { src: 6 dst: 7 }
  edges { src: 7 dst: 0 loopback: true }       # crosses back, closes the ring
}`;

function PatternCard(props: { title: string; diagram: ReturnType<typeof Text>; code: string; bullets: string[] }) {
  return (
    <Card>
      <CardHeader>{props.title}</CardHeader>
      <CardBody>
        <Grid columns={fit(320)} gap={16} align="start">
          <Stack gap={10}>
            {props.diagram}
            <Bullets items={props.bullets} />
          </Stack>
          <CodeCard path={props.title} mode="plain" language="protobuf" src={props.code} />
        </Grid>
      </CardBody>
    </Card>
  );
}

function DescriptorTab() {
  return (
    <Stack gap={24}>
      <Grid columns={fit(340)} gap={20} align="start">
        <Stack gap={8}>
          <H2>The format</H2>
          <Text tone="secondary">
            A new pipeline_descriptors section next to the meshes. It lists stages, their shapes and how they connect.
          </Text>
          <Table headers={["Message", "Field", "Meaning"]} rows={FORMAT_FIELDS} />
        </Stack>
        <Stack gap={8}>
          <CodeCard
            path="gpt_oss_120b_single_galaxy.textproto"
            mode="plain"
            language="protobuf"
            src={`
# --- Existing: the mesh -------------------------------------
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 8, 4 ] }   # 32 chips
  host_topology   { dims: [ 2, 2 ] }   # 4 ranks, each a 4x2 slice
  channels { count: 2 policy: RELAXED }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }

# --- New: the pipeline --------------------------------------
pipeline_descriptors {
  name: "gpt_oss_120b_single_galaxy"
  stages { id: 0 label: "Embedding" shape { dims: [ 4, 2 ] } }
  stages { id: 1 label: "Windowed"  shape { dims: [ 4, 2 ] } }
  stages { id: 2 label: "Global"    shape { dims: [ 4, 2 ] } }
  stages { id: 3 label: "LMHead"    shape { dims: [ 4, 2 ] } }
  edges { src: 0 dst: 1 }
  edges { src: 1 dst: 2 }
  edges { src: 2 dst: 3 }
  edges { src: 3 dst: 0 loopback: true }
}`}
          />
          <Row gap={12} align="center">
            <SingleMeshRing />
          </Row>
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>Loopback and host I/O</H2>
        <LoopbackRuleDiagram />
        <Bullets
          items={[
            "Without the loopback edges, the edges have no cycle.",
            "Each loopback edge closes a cycle: its dst reaches its src over the other edges.",
            "Exactly one stage has no incoming edge except loopbacks. It owns host I/O (green).",
          ]}
        />
      </Stack>

      <Stack gap={8}>
        <H2>Which mesh a stage can bind to</H2>
        <Text tone="secondary">A stage is a rectangle of whole host slices, in the mesh&apos;s orientation. On 4x2 slices:</Text>
        <Table
          headers={["Stage shape", "Slices", "Binds?"]}
          rows={[
            ["4x2", "1 (one rank)", "yes"],
            ["4x4 or 8x2", "2 (joint stage, two ranks)", "yes"],
            ["2x4", "not whole slices (no transpose)", "no"],
            ["2x2", "half a slice", "no"],
          ]}
          rowTone={["success", "success", "danger", "danger"]}
        />
      </Stack>

      <Stack gap={12}>
        <H2>Three patterns</H2>
        <PatternCard
          title="Joint stage: one stage over several ranks"
          diagram={<JointStageDiagram />}
          code={`
# a 4x4 LM head over two 4x2 slices
stages { id: 7 label: "LMHead" shape { dims: [ 4, 4 ] } }`}
          bullets={[
            "The shape is bigger than one slice, so the stage takes two whole slices.",
            "Both ranks own it together. Nothing else in the MGD changes.",
          ]}
        />
        <PatternCard
          title="Fork: a router with two branches"
          diagram={<LoopbackRuleDiagram part="fork" />}
          code={`
# stage 0 is the router, 1..38 the target chain, 39..49 the draft chain
edges { src: 0  dst: 1 }
# ... 1 -> 2, ..., 37 -> 38
edges { src: 38 dst: 0  loopback: true }   # target returns to the router
edges { src: 0  dst: 39 }
# ... 39 -> 40, ..., 48 -> 49
edges { src: 49 dst: 0  loopback: true }   # draft returns to the router`}
          bullets={[
            "The router has two outgoing edges; nothing limits a stage's degree.",
            "Each branch's return is its own loopback, and both ride fabric in the first version.",
            "Later: policy: RELAXED on a return lets it go over host MPI.",
          ]}
        />
        <PatternCard
          title="Heterogeneous: stages of different shapes (llama_8b_pod)"
          diagram={<HeteroDiagram />}
          code={`
# 8 stages of 4x2, then 32 of 1x2, one ring
stages { id: 0 shape { dims: [ 4, 2 ] } }
# ... ids 1..7: [ 4, 2 ]
stages { id: 8 shape { dims: [ 1, 2 ] } }
# ... ids 9..39: [ 1, 2 ]
edges { src: 0 dst: 1 }
# ... i -> i + 1, up to 38 -> 39
edges { src: 39 dst: 0 loopback: true }`}
          bullets={[
            "A [4, 2] stage binds only to meshes with 4x2 slices; [1, 2] only to meshes with 1x2 slices.",
            "So the ring must cross from one kind of mesh to the other and back.",
          ]}
        />
      </Stack>

      <Stack gap={12}>
        <H2>Reference MGDs</H2>
        <Text tone="secondary">The rules below use these two descriptors and this machine.</Text>
        <Grid columns={fit(380)} gap={16} align="start">
          <Stack gap={8}>
            <CodeCard path="mixed4: stages that span two ranks" mode="plain" language="protobuf" src={RUN_MIXED4_NEW} />
            <MeshRows
              meshes={[
                { name: "M0", slices: ["A0 · r0", "A1 · r1", "A2 · r2", "A3 · r3"] },
                { name: "M1", slices: ["B0 · r4", "B1 · r5", "B2 · r6", "B3 · r7"] },
              ]}
              stages={[
                { mesh: 0, from: 0, to: 0, label: "0" },
                { mesh: 0, from: 1, to: 2, label: "1" },
                { mesh: 0, from: 3, to: 3, label: "idle", idle: true },
                { mesh: 1, from: 0, to: 0, label: "3" },
                { mesh: 1, from: 1, to: 2, label: "2" },
                { mesh: 1, from: 3, to: 3, label: "idle", idle: true },
              ]}
              cables={[
                { a: 0, b: 0 },
                { a: 2, b: 2 },
              ]}
              note="one placement on pods (P, R): cables A0-B0 and A2-B2"
            />
          </Stack>
          <Stack gap={8}>
            <CodeCard path="pod8: two slice shapes" mode="plain" language="protobuf" src={RUN_POD8_NEW} />
            <MeshRows
              meshes={[
                { name: "M0", slices: ["A0", "A1", "A2", "A3"] },
                { name: "M1", slices: ["T0", "T1", "T2", "T3"] },
              ]}
              stages={[
                { mesh: 0, from: 0, to: 0, label: "0" },
                { mesh: 0, from: 1, to: 1, label: "1" },
                { mesh: 0, from: 2, to: 2, label: "2" },
                { mesh: 0, from: 3, to: 3, label: "3" },
                { mesh: 1, from: 0, to: 0, label: "7" },
                { mesh: 1, from: 1, to: 1, label: "4" },
                { mesh: 1, from: 2, to: 2, label: "6" },
                { mesh: 1, from: 3, to: 3, label: "5" },
              ]}
              cables={[
                { a: 0, b: 0 },
                { a: 3, b: 1 },
              ]}
              note="M0 on pod P, M1 on board U; T0 T1 / T2 T3 form a 2x2 grid"
            />
          </Stack>
        </Grid>
        <MachineDiagram />
      </Stack>
    </Stack>
  );
}

// --- Tier A and Tier B -----------------------------------------------------

const TIER_A_RULES: string[][] = [
  ["Pipeline name non-empty and unique", "two pipelines both named mixed4"],
  ["At least one stage; stage ids unique", "two stages with id 3"],
  ["Shape is two dims, both > 0", "shape { dims: [ 4, 2, 1 ] }"],
  ["Edges name declared stages, and src != dst", "edges { src: 0 dst: 9 } with no stage 9; edges { src: 0 dst: 0 }"],
  ["Without the loopback edges, no cycle", "mixed4 with 3 -> 0 not marked loopback"],
  ["Each loopback edge closes a cycle", "a chain 0 -> 1 whose only edge is marked loopback"],
  ["Exactly one stage has no incoming edge except loopbacks", "two chains, 0 -> 1 and 2 -> 3"],
  ["Policy omitted or STRICT", "mixed4 with policy: RELAXED on 3 -> 0 (allowed later)"],
];

const TIER_B_RULES: string[][] = [
  ["Binding", "Every stage is whole slices of some mesh, in that mesh's orientation", "a [2, 4] stage when every slice is 4x2"],
  ["Capacity", "Slices needed by every pipeline <= slices declared; stages that fit only one kind of mesh fit those meshes", "mixed4 plus a [4, 6] stage (9 > 8); mixed4 twice (12 > 8)"],
  ["A split exists", "Some choice of one mesh per stage fits each mesh, and every edge between two meshes rides a declared connection within its channel count", "mixed4 with no connection between M0 and M1"],
];

function TiersTab() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>Tier A: is it a valid pipeline? (parse time, no mesh)</H2>
        <Table headers={["Rule", "Rejected example"]} rows={TIER_A_RULES} />
        <Text size="small" tone="tertiary">
          Allowed: any number of edges into or out of a stage, branches that end, a repeated edge (another channel), any label.
        </Text>
      </Stack>
      <Stack gap={8}>
        <H2>Tier B: can it fit the declared meshes? (no hardware)</H2>
        <Table headers={["Rule", "Checks", "Rejected example"]} rows={TIER_B_RULES} />
        <Text size="small" tone="tertiary">
          Counts and channels only: where a stage sits inside its mesh is left to Tier C. The split search stops at its first
          success, and accepts if it runs out of steps.
        </Text>
      </Stack>
      <Stack gap={8}>
        <H3>Both tiers run on each MGD part, and again on the merged MGD</H3>
        <MergeDiagram />
      </Stack>
    </Stack>
  );
}

// --- Tier C: the mapping ---------------------------------------------------

function TierCFlow() {
  const t = useHostTheme();
  const bx = 40;
  const bw = 380;
  const bh = 44;
  const y = (i: number) => 10 + i * 64;
  const steps: Array<{ title: string; sub: string; kind: BoxKind }> = [
    { title: "SAT seating", sub: "each mesh on a set of physical chips (a seat)", kind: "existing" },
    { title: "intra-mesh layout", sub: "which ASIC each logical chip is", kind: "existing" },
    { title: "C1. position graph", sub: "every place each stage shape fits, and which places touch", kind: "new" },
    { title: "C2. solve + resolver", sub: "one rectangle per stage; a chip link for every edge; R", kind: "new" },
    { title: "C3. on success: emit", sub: "the mapping, plus R: the crossings and their cables", kind: "reuse" },
    { title: "control plane", sub: "binds R's cables first, then today's fill", kind: "reuse" },
    { title: "blaze builds", sub: "unchanged; a placement is known to exist", kind: "existing" },
  ];
  const red = t.category.red;
  return (
    <ScrollX>
      <svg width="100%" viewBox="0 0 860 460" style={{ maxWidth: 860, minWidth: 640 }}>
        <ArrowDefs id="tc-arrow" color={t.stroke.primary} />
        <ArrowDefs id="tc-red" color={red} />
        {steps.map((s, i) => (
          <g key={s.title}>
            <SvgBox x={bx} y={y(i)} w={bw} h={bh} title={s.title} sub={s.sub} kind={s.kind} />
            {i < steps.length - 1 ? (
              <line x1={bx + bw / 2} y1={y(i) + bh} x2={bx + bw / 2} y2={y(i + 1)} stroke={t.stroke.primary} markerEnd="url(#tc-arrow)" />
            ) : null}
          </g>
        ))}
        <text x={bx + bw / 2 + 8} y={y(4) - 6} fontSize={10.5} fill={t.category.green}>
          accept
        </text>
        <SvgBox x={540} y={y(3)} w={280} h={bh} title="C4. on failure" sub="change the layout, or the seating" kind="fail" />
        <line x1={bx + bw} y1={y(3) + bh / 2} x2={540} y2={y(3) + bh / 2} stroke={red} strokeDasharray="4 3" markerEnd="url(#tc-red)" />
        <text x={480} y={y(3) + bh / 2 + 16} textAnchor="middle" fontSize={10.5} fill={red}>
          fails
        </text>
        <path d={`M 600 ${y(3)} L 600 ${y(1) + bh / 2} L ${bx + bw + 2} ${y(1) + bh / 2}`} fill="none" stroke={red} strokeDasharray="4 3" markerEnd="url(#tc-red)" />
        <text x={436} y={y(1) + bh / 2 - 6} fontSize={10.5} fill={red}>
          1. new layout for the boundary mesh (at most 4 times)
        </text>
        <path d={`M 780 ${y(3)} L 780 ${y(0) + bh / 2} L ${bx + bw + 2} ${y(0) + bh / 2}`} fill="none" stroke={red} strokeDasharray="4 3" markerEnd="url(#tc-red)" />
        <text x={436} y={y(0) + bh / 2 - 6} fontSize={10.5} fill={red}>
          2. then exclude this seat set; the next seating
        </text>
      </svg>
    </ScrollX>
  );
}

function PositionDiagram() {
  const t = useHostTheme();
  const sw = 70;
  const x0 = 120;
  const pos = (x: number, y: number, w: number, label: string) => (
    <g key={`${x}-${y}`}>
      <rect x={x + 2} y={y} width={w - 4} height={22} rx={4} fill={t.fill.secondary} stroke={t.accent.primary} />
      <text x={x + w / 2} y={y + 15} textAnchor="middle" fontSize={10.5} fill={t.text.primary}>
        {label}
      </text>
    </g>
  );
  return (
    <ScrollX>
      <svg width="100%" viewBox="0 0 420 220" style={{ maxWidth: 420, minWidth: 360 }}>
        <text x={4} y={44} fontSize={11} fontWeight={600} fill={t.text.secondary}>
          slices
        </text>
        {["A0", "A1", "A2", "A3"].map((s, i) => (
          <g key={s}>
            <rect x={x0 + i * sw} y={26} width={sw} height={30} fill={t.fill.quaternary} stroke={t.stroke.secondary} strokeDasharray="3 3" />
            <text x={x0 + i * sw + sw / 2} y={46} textAnchor="middle" fontSize={11} fill={t.text.tertiary}>
              {s}
            </text>
          </g>
        ))}
        <text x={4} y={88} fontSize={11} fontWeight={600} fill={t.text.secondary}>
          [4, 2] stage
        </text>
        <text x={4} y={102} fontSize={10.5} fill={t.text.tertiary}>
          4 positions
        </text>
        {[0, 1, 2, 3].map((i) => pos(x0 + i * sw, 76, sw, `A${i}`))}
        <text x={4} y={140} fontSize={11} fontWeight={600} fill={t.text.secondary}>
          [4, 4] stage
        </text>
        <text x={4} y={154} fontSize={10.5} fill={t.text.tertiary}>
          3 positions
        </text>
        {pos(x0, 124, 2 * sw, "A0-A1")}
        {pos(x0 + sw, 150, 2 * sw, "A1-A2")}
        {pos(x0 + 2 * sw, 176, 2 * sw, "A2-A3")}
        <text x={4} y={214} fontSize={10.5} fill={t.text.tertiary}>
          positions of one shape overlap; the solve uses each slice at most once
        </text>
      </svg>
    </ScrollX>
  );
}

function C2Strip() {
  const t = useHostTheme();
  const boxes: Array<{ title: string; sub: string; fail?: string }> = [
    { title: "footprint solve", sub: "one rectangle per stage, no overlap", fail: "no carve" },
    { title: "resolver", sub: "a chip link for every edge", fail: "no layout" },
    { title: "record R", sub: "cables each crossing can use" },
    { title: "demand check", sub: "R fits each mesh pair's channels", fail: "over demand" },
  ];
  const bw = 190;
  const gap = 34;
  return (
    <ScrollX>
      <svg width="100%" viewBox="0 0 900 110" style={{ maxWidth: 900, minWidth: 700 }}>
        <ArrowDefs id="c2-arrow" color={t.stroke.primary} />
        {boxes.map((b, i) => {
          const x = 4 + i * (bw + gap);
          return (
            <g key={b.title}>
              <SvgBox x={x} y={10} w={bw} h={46} title={b.title} sub={b.sub} kind="new" />
              {i < boxes.length - 1 ? <line x1={x + bw} y1={33} x2={x + bw + gap} y2={33} stroke={t.stroke.primary} markerEnd="url(#c2-arrow)" /> : null}
              {b.fail ? (
                <text x={x + bw / 2} y={76} textAnchor="middle" fontSize={10.5} fill={t.category.red}>
                  {b.fail}: C4
                </text>
              ) : null}
            </g>
          );
        })}
        <text x={4 + 3 * (bw + gap) + bw / 2} y={94} textAnchor="middle" fontSize={10.5} fill={t.category.green}>
          pass: C3
        </text>
      </svg>
    </ScrollX>
  );
}

function SuccessDiagram() {
  const t = useHostTheme();
  const boxes: Array<{ title: string; sub: string; kind: BoxKind }> = [
    { title: "C2 accepted", sub: "carve + layout + R", kind: "new" },
    { title: "emit the mapping", sub: "mesh -> chips, as today", kind: "existing" },
    { title: "publish R", sub: "every rank gets it", kind: "new" },
    { title: "phase 0: bind R", sub: "before any other cable", kind: "new" },
    { title: "today's fill", sub: "the rest of the budget", kind: "existing" },
  ];
  const bw = 160;
  const gap = 24;
  return (
    <ScrollX>
      <svg width="100%" viewBox="0 0 920 110" style={{ maxWidth: 920, minWidth: 720 }}>
        <ArrowDefs id="sc-arrow" color={t.stroke.primary} />
        {boxes.map((b, i) => {
          const x = 4 + i * (bw + gap);
          return (
            <g key={b.title}>
              <SvgBox x={x} y={10} w={bw} h={46} title={b.title} sub={b.sub} kind={b.kind} />
              {i < boxes.length - 1 ? <line x1={x + bw} y1={33} x2={x + bw + gap} y2={33} stroke={t.stroke.primary} markerEnd="url(#sc-arrow)" /> : null}
            </g>
          );
        })}
        <text x={4 + 3 * (bw + gap) + bw / 2} y={78} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={t.accent.primary}>
          constrained here:
        </text>
        <text x={4 + 3 * (bw + gap) + bw / 2} y={93} textAnchor="middle" fontSize={10.5} fill={t.text.secondary}>
          which cables carry each crossing
        </text>
      </svg>
    </ScrollX>
  );
}

function FailureDiagram() {
  const t = useHostTheme();
  const red = t.category.red;
  const box = useBoxStyle();
  const scope = (x: number, y: number, w: number, h: number, label: string, kind: BoxKind) => {
    const s = box(kind);
    return (
      <g>
        <rect x={x} y={y} width={w} height={h} rx={6} fill={s.fill} stroke={s.stroke} strokeDasharray={s.dash} />
        <text x={x + 10} y={y + 16} fontSize={11} fontWeight={600} fill={t.text.secondary}>
          {label}
        </text>
      </g>
    );
  };
  const note = (y: number, x1: number, lines: string[], color: string) => (
    <g>
      <line x1={x1} y1={y} x2={490} y2={y} stroke={color} strokeDasharray="3 3" />
      {lines.map((l, i) => (
        <text key={i} x={498} y={y + 4 + i * 14} fontSize={10.5} fill={i === 0 ? color : t.text.secondary} fontWeight={i === 0 ? 600 : 400}>
          {l}
        </text>
      ))}
    </g>
  );
  return (
    <ScrollX>
      <svg width="100%" viewBox="0 0 780 250" style={{ maxWidth: 780, minWidth: 640 }}>
        {scope(10, 10, 460, 230, "SAT master solve: every seat set", "existing")}
        {scope(30, 44, 420, 186, "one seating: a seat per mesh", "existing")}
        {scope(50, 80, 380, 140, "intra-mesh layout of one mesh", "existing")}
        {scope(70, 140, 210, 66, "C1 + C2 on this layout", "fail")}
        {note(174, 280, ["C2 fails", "exact for this layout only"], red)}
        {note(110, 430, ["1. retry the layout", "block where the cabled chips sit,", "solve the mesh again (at most 4)"], red)}
        {note(58, 450, ["2. exclude the seat set", "the master gives the next seating"], red)}
        {note(24, 470, ["never: forbid one mesh on one seat", "another split could still use it"], t.text.tertiary)}
      </svg>
    </ScrollX>
  );
}

const TIER_C_TYPES: string[][] = [
  ["PipelineStageNode", "A stage in the mapper's logical graph; all pipelines in one graph"],
  ["PipelineStageInfo", "A stage's shape and its loopback edges"],
  ["position", "A rectangle of whole host slices that one stage shape fills"],
  ["PipelinePlacementEnumerationSession", "C1 and C2 for one seating: builds the position graph, solves, runs the resolver"],
  ["GraphLayoutResult", "The resolver's answer: each stage's submesh and the chip pair under each edge"],
  ["PipelineLayoutResult", "Per pipeline: its submeshes and GraphLayoutResult; plus R"],
  ["RequiredCrossing (R)", "One per edge between meshes: the cables that can carry it, and how many channels it needs"],
  ["ExitNodeConnection", "One ETH channel between two ASICs, from the physical system descriptor"],
];

function TierCTab() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H2>The loop</H2>
        <Text tone="secondary">Tier C runs once per seating, after the intra-mesh layout, and only when the MGD has a pipeline.</Text>
        <Legend />
        <TierCFlow />
      </Stack>

      <Grid columns={fit(380)} gap={24} align="start">
        <Stack gap={8}>
          <H2>C1: the position graph</H2>
          <PositionDiagram />
          <Bullets
            items={[
              "A node is a position: a rectangle of slices one stage shape fills.",
              "Two positions are joined when they do not overlap and touch, or a cable links them across meshes.",
              "A position a stage cannot use (too few core slots, no room for stage 0's host I/O) is left out for that stage.",
            ]}
          />
        </Stack>
        <Stack gap={8}>
          <H2>C2: one solve, one check</H2>
          <C2Strip />
          <Bullets
            items={[
              "The solve maps the stage graph onto the position graph: one position per stage, each slice used once.",
              "The resolver, the same code blaze runs, lays every edge on a chip link.",
              "Each edge between meshes adds a crossing to R. R must fit the channels the MGD gives that mesh pair.",
              "Any failure fails this layout: C4.",
            ]}
          />
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H2>C3: on success</H2>
        <SuccessDiagram />
        <Text tone="secondary">
          The count of channels per mesh pair stays what the MGD says. What changes is which cables are bound: the ones the
          layout crosses on.
        </Text>
        <BindingCompare />
      </Stack>

      <Stack gap={8}>
        <H2>C4: on failure</H2>
        <FailureDiagram />
        <Text size="small" tone="tertiary">
          With a pipeline, the master also offers seatings that only swap meshes (unique_shapes off): two meshes swapped can lay
          out differently.
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>Types worth knowing</H2>
        <Table headers={["Type", "What it is"]} rows={TIER_C_TYPES} />
      </Stack>
    </Stack>
  );
}

// --- Worked examples -------------------------------------------------------

type Lane = "MGD" | "Tier A" | "Tier B" | "Tier C" | "control plane" | "blaze";
type FlowStep = { lane: Lane; title: string; detail: string };

function FlowSteps({ steps }: { steps: FlowStep[] }) {
  const t = useHostTheme();
  const colors: Record<Lane, string> = {
    MGD: t.text.tertiary,
    "Tier A": t.category.blue,
    "Tier B": t.category.blue,
    "Tier C": t.accent.primary,
    "control plane": t.category.orange,
    blaze: t.category.green,
  };
  return (
    <div>
      {steps.map((s, i) => (
        <div key={i}>
          <div style={{ display: "flex", gap: 12, alignItems: "stretch" }}>
            <div style={{ width: 104, flexShrink: 0, paddingTop: 7, textAlign: "right" }}>
              <Text size="small" weight="semibold" style={{ color: colors[s.lane] }}>
                {s.lane}
              </Text>
            </div>
            <div
              style={{
                flex: 1,
                borderLeft: `3px solid ${colors[s.lane]}`,
                background: t.fill.quaternary,
                borderRadius: 4,
                padding: "6px 10px",
              }}
            >
              <Stack gap={2}>
                <Text size="small" weight="semibold">
                  {s.title}
                </Text>
                <Text size="small" tone="secondary">
                  {s.detail}
                </Text>
              </Stack>
            </div>
          </div>
          {i < steps.length - 1 ? <div style={{ marginLeft: 117, height: 10, borderLeft: `1px solid ${t.stroke.secondary}` }} /> : null}
        </div>
      ))}
    </div>
  );
}

/** One mesh as a row of host slices with the stage on each, and the loopback over the wrap. */
function StripRing(props: { cells: string[]; subs?: string[]; cellW: number; wrap: string; title: string }) {
  const t = useHostTheme();
  const n = props.cells.length;
  const x0 = 10;
  const W = x0 * 2 + n * props.cellW;
  const accent = t.accent.primary;
  return (
    <ScrollX>
      <svg width="100%" viewBox={`0 0 ${W} 132`} style={{ maxWidth: W, minWidth: Math.min(W, 640) }}>
        <ArrowDefs id={`sr-${n}`} color={accent} />
        <text x={x0} y={14} fontSize={11} fontWeight={600} fill={t.text.secondary}>
          {props.title}
        </text>
        {props.cells.map((c, i) => (
          <g key={i}>
            <rect x={x0 + i * props.cellW + 1} y={24} width={props.cellW - 2} height={44} rx={4} fill={t.fill.secondary} stroke={t.stroke.primary} />
            <text x={x0 + i * props.cellW + props.cellW / 2} y={props.subs ? 42 : 50} textAnchor="middle" fontSize={11} fontWeight={600} fill={t.text.primary}>
              {c}
            </text>
            {props.subs ? (
              <text x={x0 + i * props.cellW + props.cellW / 2} y={58} textAnchor="middle" fontSize={10} fill={t.text.tertiary}>
                {props.subs[i]}
              </text>
            ) : null}
          </g>
        ))}
        <path
          d={`M ${x0 + (n - 0.5) * props.cellW} 68 C ${x0 + (n - 0.5) * props.cellW} 112, ${x0 + props.cellW / 2} 112, ${x0 + props.cellW / 2} 72`}
          fill="none"
          stroke={accent}
          strokeWidth={1.6}
          strokeDasharray="5 3"
          markerEnd={`url(#sr-${n})`}
        />
        <text x={W / 2} y={122} textAnchor="middle" fontSize={10.5} fill={accent}>
          {props.wrap}
        </text>
      </svg>
    </ScrollX>
  );
}

function DFlashPlacementDiagram() {
  const t = useHostTheme();
  const accent = t.accent.primary;
  const back = { fill: "none", stroke: accent, strokeWidth: 1.6, strokeDasharray: "5 3" } as const;
  return (
    <ScrollX>
      <svg width="100%" viewBox="0 0 760 260" style={{ maxWidth: 760, minWidth: 620 }}>
        <ArrowDefs id="df-arrow" color={t.stroke.primary} />
        <ArrowDefs id="df-back" color={accent} />
        <SvgBox x={10} y={34} w={80} h={40} title="0 Router" kind="new" />
        <SvgBox x={130} y={14} w={610} h={34} title="1..63 target: Embedding, Dense0, MoE1..60, LMHead" />
        <SvgBox x={130} y={100} w={300} h={34} title="64..72 draft: Pre, 6 decoders, Post, Relay" />
        <line x1={90} y1={46} x2={128} y2={31} stroke={t.stroke.primary} markerEnd="url(#df-arrow)" />
        <line x1={90} y1={62} x2={128} y2={117} stroke={t.stroke.primary} markerEnd="url(#df-arrow)" />
        <path {...back} d="M 735 48 L 735 64 L 92 64" markerEnd="url(#df-back)" />
        <text x={420} y={80} textAnchor="middle" fontSize={10.5} fill={accent}>
          63 -&gt; 0  loopback
        </text>
        <path {...back} d="M 425 134 L 425 150 L 50 150 L 50 76" markerEnd="url(#df-back)" />
        <text x={240} y={166} textAnchor="middle" fontSize={10.5} fill={accent}>
          72 -&gt; 0  loopback (blaze sends it over host MPI)
        </text>
        {[0, 1, 2, 3, 4].map((m) => (
          <SvgBox key={m} x={10 + m * 148} y={190} w={138} h={40} title={`M${m}`} sub="4x32, 16 slices of 4x2" />
        ))}
        <text x={380} y={252} textAnchor="middle" fontSize={10.5} fill={t.text.tertiary}>
          80 slices for 73 stages: 7 stay idle. Every mesh pair has a connection with 16 channels.
        </text>
      </svg>
    </ScrollX>
  );
}

const KIMI16_MGD = `
# --- the mesh: quad_bh_galaxy_4x2 (exists) ----------------------------
mesh_descriptors {
  name: "M0"  arch: BLACKHOLE
  device_topology { dims: [ 4, 32 ] dim_types: [ RING, RING ] }
  host_topology   { dims: [ 1, 16 ] }   # 16 ranks; slice j = cols 2j, 2j+1
  channels { count: 2 policy: RELAXED }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }

# --- new: the ring stage.build_pipeline builds for 16 processes -------
pipeline_descriptors {
  name: "kimi_k2_6_16"
  stages { id: 0  label: "Embedding"   shape { dims: [ 4, 2 ] } }
  stages { id: 1  label: "Dense0"      shape { dims: [ 4, 2 ] } }
  stages { id: 2  label: "MoE1"        shape { dims: [ 4, 2 ] } }
  # ... ids 3..13: MoE2 .. MoE12, each [ 4, 2 ]
  stages { id: 14 label: "LMHead"      shape { dims: [ 4, 2 ] } }
  stages { id: 15 label: "Passthrough" shape { dims: [ 4, 2 ] } }
  edges { src: 0 dst: 1 }
  # ... i -> i + 1, up to 14 -> 15
  edges { src: 15 dst: 0 loopback: true }   # over the col 31 -> col 0 wrap
}`;

const KIMI16_STEPS: FlowStep[] = [
  { lane: "MGD", title: "One quad galaxy and a 16-stage ring", detail: "A 4x32 mesh, both axes RING, 16 hosts along the columns. The pipeline is the ring blaze builds for 16 processes." },
  { lane: "Tier A", title: "A valid pipeline", detail: "Ids 0..15 are unique. Without 15 -> 0 the edges are a chain; 15 -> 0 is marked loopback and closes it. Stage 0 is the only stage with no other incoming edge, so it owns host I/O." },
  { lane: "Tier B", title: "It fits the mesh", detail: "[4, 2] is one slice of M0, so every stage binds to M0. 16 slices needed, 16 declared. One mesh: no crossings to count." },
  { lane: "Tier C", title: "Seating and intra-mesh layout", detail: "One mesh, one seat: the quad galaxy. The intra-mesh solve fixes which ASIC each logical chip is, as today." },
  { lane: "Tier C", title: "C1: 16 positions", detail: "Every stage fits any of the 16 slices. Neighbouring slices touch, and so do slice 15 and slice 0 over the column wrap." },
  { lane: "Tier C", title: "C2: solve and resolver", detail: "Stage j lands on slice j (or a rotation or mirror). The resolver finds a direct chip link for every edge, 15 -> 0 included. Nothing crosses meshes, so R is empty." },
  { lane: "control plane", title: "C3: nothing extra to bind", detail: "The mapping is emitted as today. R is empty, so the links are bound exactly as today." },
  { lane: "blaze", title: "entry.build_pipeline, generic path", detail: "Neither reload nor DFlash: stage.build_pipeline. The world size is 16, so get_pipeline_harnesses returns Embedding, Dense0, MoE1..12, LMHead, Passthrough." },
  { lane: "blaze", title: "Harnesses become the same graph", detail: "Harness i becomes node s{i}, with edges s0 -> ... -> s15 and the loopback s15 -> s0: the MGD's pipeline. One mesh, so build_topology tiles it into 4x2 submeshes and the resolver assigns stages to them." },
  { lane: "blaze", title: "Each rank builds its stage", detail: "GenericStage takes harnesses[my_stage_idx]: that layer's weights, socket sizes and kernels." },
];

const RELOAD_MGD = `
# --- the mesh: single_bh_galaxy_4x8_torus_xy (exists) -----------------
mesh_descriptors {
  name: "M0"  arch: BLACKHOLE
  device_topology { dims: [ 4, 8 ] dim_types: [ RING, RING ] }
  host_topology   { dims: [ 1, 4 ] }   # 4 ranks; stage j = cols 2j, 2j+1
  channels { count: 2 policy: RELAXED }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }

# --- new: the 4-stage ring ReloadPipeline.open builds ------------------
pipeline_descriptors {
  name: "kimi_k2_6_reload"
  stages { id: 0 label: "m0" shape { dims: [ 4, 2 ] } }   # 17 images
  stages { id: 1 label: "m1" shape { dims: [ 4, 2 ] } }   # 16 images
  stages { id: 2 label: "m2" shape { dims: [ 4, 2 ] } }   # 16 images
  stages { id: 3 label: "m3" shape { dims: [ 4, 2 ] } }   # 16 images
  edges { src: 0 dst: 1 }
  edges { src: 1 dst: 2 }
  edges { src: 2 dst: 3 }
  edges { src: 3 dst: 0 loopback: true }   # over the col 7 -> col 0 wrap
}`;

const RELOAD_STEPS: FlowStep[] = [
  { lane: "MGD", title: "One galaxy torus and a 4-stage ring", detail: "A 4x8 mesh, both axes RING, 4 hosts, 2 channels on every link. The MGD has the 4 ring stages, not the 65 model steps." },
  { lane: "Tier A", title: "A valid pipeline", detail: "A chain 0 -> 1 -> 2 -> 3, closed by the loopback 3 -> 0. Stage 0 owns host I/O." },
  { lane: "Tier B", title: "It fits the mesh", detail: "4 slices needed, 4 declared, one mesh, no crossings." },
  { lane: "Tier C", title: "C1: 4 positions", detail: "One per slice. Slice 3 touches slice 0 only because the column axis is RING." },
  { lane: "Tier C", title: "C2: the ring needs the wrap", detail: "The only ring is m0..m3 around the 4 slices. The resolver needs a direct link for 3 -> 0, which only the col 7 -> col 0 wrap gives. With a LINE column axis this seating fails. R is empty." },
  { lane: "control plane", title: "C3: nothing extra to bind", detail: "One mesh; every link has the mesh's 2 channels, as today." },
  { lane: "blaze", title: "entry.build_pipeline(reload=True)", detail: "build_reload_pipeline checks for 4 processes, slow dispatch and the blaze BRISC firmware." },
  { lane: "blaze", title: "Deal the model onto the ring", detail: "FULL_SEQUENCE is 65 steps: embedding, dense 0, MoE 1..60, 2 bypasses, head. Step i goes to stage i % 4: m0 gets 17 harnesses, m1..m3 get 16." },
  { lane: "blaze", title: "ReloadPipeline.open builds the 4-node graph", detail: "s0..s3 with the loopback s3 -> s0: the MGD's pipeline. On one mesh, build_topology cuts four 4x2 submeshes, the resolver assigns stages, and each rank gets my_stage_idx." },
  { lane: "blaze", title: "Capture, install, launch once", detail: "Each rank compiles its stage's 16 or 17 images and captures each into the reload table. After one launch, the firmware swaps images as packets pass." },
];

const DFLASH_MGD = `
# --- the meshes: SC20_4x2 (exists) ------------------------------------
mesh_descriptors {
  name: "M0"  arch: BLACKHOLE
  device_topology { dims: [ 4, 32 ] dim_types: [ RING, RING ] }
  host_topology   { dims: [ 1, 16 ] }   # 16 ranks per mesh
  channels { count: 2 policy: RELAXED }
}
graph_descriptors {
  name: "G0"  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  # ... mesh_id 1 .. 4
  connections {                       # one per mesh pair, 10 in all
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
    channels { count: 16 policy: RELAXED }
  }
  # ... the other 9 pairs, 16 channels each
}
top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }

# --- new: the router fork ---------------------------------------------
pipeline_descriptors {
  name: "kimi_k2_6_dflash"
  stages { id: 0  label: "Router"     shape { dims: [ 4, 2 ] } }
  stages { id: 1  label: "Embedding"  shape { dims: [ 4, 2 ] } }   # target 1..63
  stages { id: 2  label: "Dense0"     shape { dims: [ 4, 2 ] } }
  # ... ids 3..62: MoE1 .. MoE60
  stages { id: 63 label: "LMHead"     shape { dims: [ 4, 2 ] } }
  stages { id: 64 label: "PreDecode"  shape { dims: [ 4, 2 ] } }   # draft 64..72
  # ... ids 65..70: DraftDecoder0 .. DraftDecoder5
  stages { id: 71 label: "PostDecode" shape { dims: [ 4, 2 ] } }
  stages { id: 72 label: "Relay"      shape { dims: [ 4, 2 ] } }
  edges { src: 0  dst: 1 }
  # ... 1 -> 2, ..., 62 -> 63
  edges { src: 63 dst: 0  loopback: true }   # verifier returns
  edges { src: 0  dst: 64 }
  # ... 64 -> 65, ..., 71 -> 72
  edges { src: 72 dst: 0  loopback: true }   # drafter returns (STRICT: fabric)
}`;

const DFLASH_STEPS: FlowStep[] = [
  { lane: "MGD", title: "Five galaxies and a router fork", detail: "Five 4x32 meshes of 16 slices (80 in all), every pair joined with 16 channels. 73 stages: the router, the target chain 1..63 and the draft chain 64..72." },
  { lane: "Tier A", title: "A valid fork", detail: "The router has two outgoing edges, which is allowed. Each return is marked loopback and closes a cycle through stage 0. Stage 0 is the only stage with no other incoming edge." },
  { lane: "Tier B", title: "It fits the meshes", detail: "Every [4, 2] stage binds to any mesh. 73 slices needed, 80 declared. A mesh holds 16 stages, so the target loop must cross meshes; some split keeps every pair within its 16 channels." },
  { lane: "Tier C", title: "Seating and intra-mesh layouts", detail: "SAT seats the 5 meshes on the 5 galaxies, then lays out each mesh's chips." },
  { lane: "Tier C", title: "C1: 80 positions", detail: "16 per mesh. Positions on two meshes join where a cable connects them." },
  { lane: "Tier C", title: "C2: place 73 stages", detail: "The solve puts each stage on its own position; 7 stay idle. The resolver lays every edge on a chip link, both loopbacks included. Each edge between meshes becomes a crossing in R, and R must fit 16 channels per pair." },
  { lane: "control plane", title: "C3: bind R first", detail: "Phase 0 binds a cable for every crossing in R before today's fill uses the rest of the 16." },
  { lane: "blaze", title: "entry.build_pipeline(dflash_router=True)", detail: "build_kimi_dflash_router_pipeline_graph writes the 73-node graph by hand: the same stages and edges, with the draft return marked transport=\"host\"." },
  { lane: "blaze", title: "build_model_pipeline_from_graph", detail: "multi_mesh, submesh 4x2: the fork path resolves the graph into a ForkLayout. 80 ranks, 73 stages, 7 idle ranks." },
  { lane: "blaze", title: "Each rank builds its stage", detail: "Router: BlockDiffusionRouterStageKind. Targets: GenericStage over the 63 Kimi harnesses, picked by the branch-local index. Drafts: their own DFlash stage kinds." },
];

function ExampleSection(props: {
  title: string;
  intro: string;
  mgdPath: string;
  mgd: string;
  picture: ReturnType<typeof Text>;
  steps: FlowStep[];
  after?: ReturnType<typeof Text>;
}) {
  return (
    <Stack gap={12}>
      <H2>{props.title}</H2>
      <Text tone="secondary">{props.intro}</Text>
      <Grid columns={fit(420)} gap={20} align="start">
        <CodeCard path={props.mgdPath} mode="plain" language="protobuf" src={props.mgd} />
        <Stack gap={12}>
          {props.picture}
          <FlowSteps steps={props.steps} />
        </Stack>
      </Grid>
      {props.after}
    </Stack>
  );
}

function ExamplesTab() {
  return (
    <Stack gap={28}>
      <Text tone="secondary">
        Each example starts from a real MGD in tt-blaze, adds the pipeline section, and follows it through the tiers, the
        control plane and blaze. Colours mark who does each step.
      </Text>
      <ExampleSection
        title="1. A regular ring: Kimi K2.6 on 16 processes"
        intro="The generic path: entry.py builds no graph; blaze turns one harness per process into a ring."
        mgdPath="quad_bh_galaxy_4x2 + kimi_k2_6_16"
        mgd={KIMI16_MGD}
        picture={
          <StripRing
            title="M0, 4x32: 16 slices of 4x2"
            cells={["E", "D0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "H", "P"]}
            cellW={44}
            wrap="15 -> 0 loopback over the col 31 -> col 0 wrap"
          />
        }
        steps={KIMI16_STEPS}
      />
      <ExampleSection
        title="2. The reload ring: Kimi K2.6 on one galaxy"
        intro="Four processes run all 65 model steps by swapping kernels. The mapper only needs the 4-stage ring."
        mgdPath="single_bh_galaxy_4x8_torus_xy + kimi_k2_6_reload"
        mgd={RELOAD_MGD}
        picture={
          <StripRing
            title="M0, 4x8: 4 slices of 4x2"
            cells={["m0", "m1", "m2", "m3"]}
            subs={["embed, head, …", "dense 0, MoE 4, …", "MoE 1, 5, …", "MoE 2, 6, …"]}
            cellW={150}
            wrap="3 -> 0 loopback over the col 7 -> col 0 wrap"
          />
        }
        steps={RELOAD_STEPS}
        after={
          <Callout tone="info" title="Not in the MGD">
            The images per stage, each rank&apos;s L1 size and the firmware. They stay in blaze; the mapper only needs the ring.
          </Callout>
        }
      />
      <ExampleSection
        title="3. The DFlash router fork: Kimi K2.6 on SC20"
        intro="entry.py writes the fork graph itself. A router feeds a 63-stage verifier and a 9-stage drafter, and both return to it."
        mgdPath="SC20_4x2 + kimi_k2_6_dflash"
        mgd={DFLASH_MGD}
        picture={<DFlashPlacementDiagram />}
        steps={DFLASH_STEPS}
        after={
          <Callout tone="warning" title="First version: the draft return is STRICT">
            The MGD asks for a fabric link on 72 -&gt; 0. Blaze carries it over host MPI, because a second fabric loopback makes
            its resolver too slow on the 63-stage branch. So Tier C is stricter than blaze here and can reject a machine blaze
            could build on, and its resolver call faces the same two-loopback cost. RELAXED (later) on that edge makes the two
            agree.
          </Callout>
        }
      />
      <Stack gap={8}>
        <H2>Side by side</H2>
        <Table
          headers={["", "Regular ring", "Reload ring", "DFlash fork"]}
          rows={[
            ["MGD stages", "16", "4 (65 steps inside)", "73"],
            ["Meshes", "1", "1", "5"],
            ["R (crossings)", "none", "none", "every edge between meshes"],
            ["Who writes blaze's graph", "stage.build_pipeline", "ReloadPipeline.open", "entry.py, by hand"],
            ["What blaze adds", "harness per stage", "16-17 images per stage, firmware", "stage kinds, host draft return"],
          ]}
        />
      </Stack>
    </Stack>
  );
}

// ===========================================================================
// PART II: IMPLEMENTATION
// ===========================================================================

const PROTO_CHANGE = `
+message PipelineStage {
+  uint32 id = 1;
+  string label = 2;
+  TorusTopology shape = 3;        // dims only; dim_types ignored
+}
+message PipelineEdge {
+  uint32 src = 1;
+  uint32 dst = 2;
+  optional Policy policy = 3;      // omitted = STRICT; RELAXED rejected for now
+  bool loopback = 4;
+}
+message PipelineDescriptor {
+  string name = 1;
+  repeated PipelineStage stages = 2;
+  repeated PipelineEdge edges = 3;
+}
 message MeshGraphDescriptor {
   ...
+  repeated PipelineDescriptor pipeline_descriptors = 6;
 }`;

function ImplFormatTab() {
  return (
    <Stack gap={22}>
      <Grid columns={fit(380)} gap={20} align="start">
        <Stack gap={8}>
          <H2>The proto change</H2>
          <CodeCard path="tt_metal/fabric/protobuf/mesh_graph_descriptor.proto" mode="diff" language="protobuf" src={PROTO_CHANGE} />
          <Table
            headers={["Check", "Where"]}
            rows={[
              ["Tier A", "static_validate, on the proto, before populate"],
              ["Tier B", "after populate, on the descriptor's meshes"],
              ["Merge", "merge() copies every part's pipelines, rejects a repeated name; the merged ctor runs A and B again"],
            ]}
          />
        </Stack>
        <Stack gap={8}>
          <H2>Tier B: the search</H2>
          <CodeCard path="Tier B, pseudo-code" mode="plain" src={TIER_B_SEARCH} />
          <Text size="small" tone="tertiary">
            The first version runs steps 1 to 3. Step 4 (the carve check), "find all" and the mesh-class symmetry are a backup,
            not implemented.
          </Text>
        </Stack>
      </Grid>
      <Stack gap={8}>
        <H2>RELAXED edges (later)</H2>
        <Text tone="secondary">
          STRICT is the default; only edges marked policy: RELAXED may go over host MPI. Blaze&apos;s Edge carries the same policy.
        </Text>
        <Grid columns={fit(340)} gap={16} align="start">
          <CodeCard
            path="mixed4 with a RELAXED loopback"
            mode="plain"
            language="protobuf"
            src={`
edges { src: 0 dst: 1 }
edges { src: 1 dst: 2 }
edges { src: 2 dst: 3 }
edges { src: 3 dst: 0 policy: RELAXED loopback: true }   # may go over host`}
          />
          <Table
            headers={["Step", "STRICT edge", "RELAXED edge"]}
            rows={[
              ["Tier B crossings", "rides a declared connection, within its channels", "left out"],
              ["Plan 1 k_AB", "counted", "not counted"],
              ["Tier C", "a fabric link in the layout", "a soft edge in the same solve; each fabric chain resolved alone"],
            ]}
          />
        </Grid>
        <Text size="small" tone="tertiary">
          Tier C mirrors blaze&apos;s host-cut planner, which never cuts stage 0&apos;s outgoing edge, so a RELAXED edge there still
          needs a fabric link.
        </Text>
      </Stack>
    </Stack>
  );
}


function ImplMapperTab() {
  return (
    <Stack gap={22}>
      <Stack gap={8}>
        <H2>The per-seating check</H2>
        <Text tone="secondary">
          MultiMeshSolutionEnumerator::next() today stops after step 2. Tier C adds four steps after intra-mesh success, and
          they run only when the MGD has a pipeline. This part covers C1 and C2; C3 and C4 follow.
        </Text>
        <Table
          headers={["Step", "Does", "Outputs", "First version", "Later (optional)"]}
          rows={[
            ["C1. Build the physical pipeline graph", "candidates, host slices, adjacency -> one position graph and its constraints", "one position graph per seating, with chip-level link tables", "every STRICT edge (P5)", "RELAXED edges as soft edges in the same solve (P11)"],
            ["C2. Pipeline solve", "footprint solve picks the carve, stage order and crossings; the resolver checks chip collisions; R recorded as every channel that can carry each crossing", "carve, layout and R within each demand, or a reason", "one solve on the existing API (completeness as cardinality constraints, conflict_cap); pre-checks; the resolver on that one carve; any failure fails the seating; per-stage misfits filtered before the solve (P6)", "the demand as a crossing cap inside the encoding"],
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
        <Legend prNote />
        <LoopFlow />
      </Stack>
      <Stack gap={8}>
        <H2>PipelinePlacementEnumerationSession: C1 and C2&apos;s solve in one class (P5)</H2>
        <Text tone="secondary">
          The pipeline twin of SatPlacementEnumerationSession. Its constructor builds the position graph, runs the pre-checks and
          sets up the constraints and one solver session; each next() solves for one more carve, never one solved before, runs the resolver on it,
          records every channel that can carry each crossing (R) and checks the crossings against each connection's demand (P6). It owns every position field, so
          PhysicalMultiMeshGraph is unchanged. The enumerator builds one per seating (place_pipelines, P8) and asks for one. Its only public
          calls are the constructor and next().
        </Text>
        <Table
          headers={["", "SatPlacementEnumerationSession (today)", "PipelinePlacementEnumerationSession (P5)"]}
          rows={[
            ["Built", "once per run, in the enumerator's constructor", "once per seating, in the enumerator's place_pipelines, after intra-mesh success"],
            ["Target", "the logical mesh graph", "logical.pipeline_stage_graph_, as a both-ways copy"],
            ["Global graph it owns", "the seat graph (every seat of every shape; seats overlap)", "the position graph (every position of every stage shape; positions overlap)"],
            ["Constraints it sets up", "pinnings, shapes, resource (seat -> ASICs)", "required (stage -> positions it fits), resource (position -> slices), completeness when tight"],
            ["next()", "one more seating", "one more carve with the resolver run on it: each pipeline's chips, GraphLayoutResult and edge channels, and R (PipelineLayoutResult)"],
          ]}
        />
        <Table
          headers={["Input", "What the session reads from it", "Why it is needed"]}
          rows={[
            ["logical (LogicalMultiMeshGraph)", "stage graph, stage shapes and loopback marks, host grids, intra-mesh links, requested exit nodes", "the stages to place and the slices they are placed on"],
            ["full (the seating's TopologyMappingResult)", "fabric_node_to_physical and physical_to_fabric_node: the intra-mesh layout, both ways", "positions are logical chips and the PSD is in ASIC ids; the layout says which cable lands on which chip"],
            ["psd (PhysicalSystemDescriptor)", "get_asic_id, get_asic_neighbors and get_eth_connections: every cable, one ExitNodeConnection per ETH channel", "C1's links between meshes, and R's channels and connection_hashes (P6). Not the seating's PhysicalMultiMeshGraph: its exit-node entries are one per channel but carry no channel numbers"],
          ]}
        />
        <CodeCard path="tt_metal/api/tt-metalium/experimental/fabric/topology_mapper_utils.hpp  (the session)" mode="new" language="cpp" src={PIPELINE_SESSION_DECL} />
        <CodeCard path="one seating, through the session" mode="plain" language="text" src={PIPELINE_SESSION_FLOW} />
      </Stack>
      <Grid columns={fit(380)} gap={20} align="start">
        <BindingPiece
          title="New: inter-mesh cables per seating"
          tag="C1: P5"
          what="One PSD read per seating, the session's private inter_mesh_cables (P5): every ETH channel between two mapped meshes' chips, as ExitNodeConnections. C1 links positions through it; R (P6) takes each crossing's channel from it. It keeps: both ends mapped onto the two meshes, requested exit nodes when the connection is device-level, one destination chip per source chip. Cable type and link state are not filtered yet; that comes with the policy plan's candidate table (P16, later)."
          todo={[
            "In the session's constructor, channels_ = inter_mesh_cables(logical, psd, full): per ASIC in full.physical_to_fabric_node, get_asic_neighbors + get_eth_connections, keyed by the two chips.",
            "Per position-graph edge, keep the links behind it and each inter-mesh link's ExitNodeConnections; C2 builds DirectLinks from them for the chosen rectangles, ordered by chip id.",
            "Later (P16): swap in collect_inter_mesh_candidates, shared with the control plane, with tiers and down links.",
          ]}
        />
        <BindingPiece
          title="New: R must fit each connection's demand"
          tag="C2: P6"
          what="Phase 0 needs min_channels per crossing in R (the pipeline edges between its two stages), counted against the pair's demand. Tier B's channel rule does not guarantee this: it checks that some projection fits, and C1 may carve one with more crossings."
          todo={[
            "Budget: crossings in R on a mesh pair <= strict + relaxed; R uses the STRICT part first, then RELAXED.",
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
          stage order and the crossings. The resolver checks that carve for chip collisions; R is recorded as every channel that can carry each crossing

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
        <Callout tone="warning" title="IMPORTANT: record every channel that can carry each crossing, and bind at least as many as it needs">
          The mapping only says two rectangles touch through some cable; with several cables between them it does not pick
          one. The resolver&apos;s link step does, and resolved_edges gives the exit and entry chip. Any link between the two
          positions can carry the crossing (the resolver chose among exactly these), so C2 records every PSD
          ExitNodeConnection of every one in channels_by_resolved_edge, the resolver&apos;s chip pair first, and adds one entry per crossing
          to R; each std::hash&lt;ExitNodeConnection&gt; is the hash the pairing and its broadcast use. If two stage positions are joined on chip 0 and on chip 1, option 1 records both and option 2 only the resolver's (open question 25). C3 publishes R through get_pipeline_required_links(), and
          phase 0 of pair_logical_intermesh_ports binds at least min_channels of every crossing (the pipeline edges between its two stages) before any other link,
          preferring the resolver&apos;s chip pair. A crossing with fewer bound is a hard error naming the pipeline and stages.
          Both phases record channels_by_resolved_edge and R in their result; only Phase 2 binds R, and no channel is written into the
          rank bindings.
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
        <H2>The check: resolver inputs</H2>
        <Table
          headers={["Resolver argument", "What C2 builds for the chosen rectangles", "Rule it enforces"]}
          rows={[
            ["stage_order", "topological_sort of every stage name, isolated stages included", "non-loopback edges form a DAG (Tier A)"],
            ["edges", "(src, dst, is_loopback); is_loopback is the MGD edge's loopback mark, checked by Tier A", "each edge needs a link between its two slots"],
            ["submesh_links", "per ordered rectangle pair, every candidate LinkPair between them", "directed per key: insert both {i, j} and {j, i}"],
            ["stage_chip_counts", "rows x cols of each stage's shape", "forces each stage onto its own rectangle"],
            ["chips", "per rectangle: InternalChip{fid, row, col} per chip", "(row, col) slot-local, from the rectangle's own origin"],
            ["capacity_overrides", "blaze's _pipeline_core_defaults, or empty", "empty: 1 slot per chip on 8+ chips, 2 on smaller"],
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
        <H2>Reuse or rebuild</H2>
        <Table
          headers={["Piece", "Today", "Plan", "PR"]}
          rows={[
            ["Graph-layout resolver", "resolve_pipeline_placement (C++), after topological_sort", "use as-is; a throw rejects the carve", "P8"],
            ["Submesh links", "discover_submesh_links (C++, asks the control plane)", "its loop, in C1 with the MGD's links and the seating's cables in place of the control plane", "P5"],
            ["Host slices", "the MGD's get_host_ranks (a grid of host ranks) and get_coord_range, the slices Tier B divides out", "each host's range, laid out as the host grid, in LogicalMultiMeshGraph::mesh_host_grids_", "P5"],
            ["Stages and edges", "PipelineDescriptorData (MGD)", "the stage graph beside today's mesh graphs is built from them; each stage's shape and edges kept in pipeline_stages_", "P5"],
            ["Positions", "SatPlacementEnumerationSession (seat graph and constraints, today)", "PipelinePlacementEnumerationSession, built per seating after intra-mesh success", "P5"],
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
            ["Adjacency: one confirmed link makes two slices adjacent", "same, from the candidate links; R records every channel that can carry each crossing"],
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
          so they are always off. The pipeline fields of both graphs are empty by default and are read only when
          logical_.pipeline_stage_graph_ has nodes, which it does exactly when the merged MGD has pipelines. Before the
          logical graph exists, one gate, not a proto check per hook: the merged descriptor's has_pipeline_descriptors()
          (and, in the backup plan, pipeline_mapping_constraints(), null when off), read from the merged descriptor
          the enumerator keeps. No pipelines means the hook adds nothing, never "run with defaults".
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
          Nothing here changes the control plane&apos;s gather: P5's inter_mesh_cables only reads the PSD inside the per-seating session, and only for MGDs with
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
            ["counter or test-only hook, no descriptor", "no check_pipeline_feasibility call, no PipelinePlacementEnumerationSession built, no extra SAT clauses"],
            ["pipeline MGD with the kill switch", "same mapping as that MGD with pipeline_descriptors removed"],
          ]}
        />
      </Stack>
    </Stack>
  );
}

const PIPELINE_GATE = `
// MeshGraphDescriptor (merged)
bool has_pipeline_descriptors() const;                                  // proto_->pipeline_descriptors_size() > 0
// Backup if the solve is slow (not implemented):
// const PipelineMappingConstraints* pipeline_mapping_constraints() const; // null when off

// LogicalMultiMeshGraph, filled once per run by build_logical_multi_mesh_adjacency_graph (P5)
AdjacencyGraph<PipelineStageNode> pipeline_stage_graph_;               // empty by default; the gate: nodes iff pipelines
std::map<PipelineStageNode, PipelineStageInfo> pipeline_stages_;     // empty likewise
std::map<LogicalMeshNode, tt::tt_metal::distributed::MeshContainer<MeshCoordinateRange>> mesh_host_grids_;   // empty likewise

// PhysicalMultiMeshGraph: unchanged. PipelinePlacementEnumerationSession is built only inside
// the enumerator's place_pipelines, which runs only for a stage graph with nodes`;

function ImplSatTab() {
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
            TODO, not in the first implementation: a rejection cap (kMaxPipelineRejections, proposed 256) and exclude_mapping(seating).
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
// R: one RequiredCrossing per crossing (P6): every link (chip pair) between its two positions, each with the PSD's
// own ExitNodeConnection entries between its two chips, and min_channels, the pipeline edges between its two stages in either direction (a repeated edge or a loopback is
// another channel; Tier B's count). ExitNodeConnection = {src_exit_node, dst_exit_node (AsicID),
// eth_conn {src_chan, dst_chan, is_local, port_type}}; std::hash of it is the connection_hash the pairing uses.

// PipelineLayoutResult
std::vector<RequiredCrossing> required_links;   // R, one entry per crossing

// TopologyMapper, every rank (broadcast with the chip info: channels via exit_node_connection_to_proto,
// min_channels, and the names for the error)
const std::vector<RequiredCrossing>& get_pipeline_required_links() const;

// Rank 0 only: phase 0 binds at least min_channels of each crossing first, more if the budget allows.
// No PortDescriptor change.
AnnotatedIntermeshConnections ControlPlane::pair_logical_intermesh_ports(
    const PortDescriptorTable& port_descriptors, const std::vector<RequiredCrossing>& required_crossings);`;

function ImplBindingTab() {
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
            Once phase 0 binds a channel for every crossing in R, a rerun of C2 cannot fail: the witness carve crosses only
            between those positions. The rerun is dropped. It would only matter if Tier C checked blaze&apos;s first cycle (question 8).
          </Callout>
        </Grid>
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
          Phase 0 takes each crossing in R, reserves its min_channels of the budget, and places its links until that many are bound
          (the resolver&apos;s chip pair first, then the other links between the two positions), with the pair&apos;s Z rule,
          each trimmed to the budget left after the other crossings' minimums, so it keeps more when it can. Today&apos;s work-list then fills the rest. With no
          required links the output is byte-identical to today. Phase 0 never throws on rank 0 (the other ranks are
          waiting for the broadcast); a crossing with no channel placed, or none gathered because its links are down,
          fails the check after the broadcast on every rank together, naming the crossing and its edge.
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
              ["Seating, carve, layout, R, budget check", "TopologyMapper: the enumerator's PipelinePlacementEnumerationSession::next() (Tier C)", "rank 0"],
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
            Until P16, C1 and R read rank 0's PSD (C1 through the physical graph built from it), which has no link state, so R can name a down channel; the check after the
            broadcast then fails on every rank (question 20). With the policy plan&apos;s candidate table (P16) down links are left
            out at runtime. Offline runs (factory PSD) see every cable as up, so the offline answer can be optimistic. STRICT
            device-level connections pin exit chips, so there is nothing to choose; Tier C checks against the pins.
          </Text>
        </Stack>
      </Grid>
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
            ["22", "Total time per mapping: conflict cap x exit-node retries x rejected seatings can reach hours (the rejection cap is a TODO, not in the first implementation). Add the rejection cap, an overall cap on the pipeline loop, or both?"],
            ["23", "Pipelines across parts: name the far end of a cross-part edge as (part index, local FabricNodeId), or return the pipeline mapping once per run?"],
            ["24", "One carve per seating: a seating is lost if its first carve fails but another would pass (101 of 101 first carves passed). Keep one, or allow a few more (more next() calls on the seating's PipelinePlacementEnumerationSession) before failing the seating?"],
            ["25", "What the pipeline mapping carries per crossing. GraphLayoutResult fixes one chip pair per edge, not a channel, so either way every channel on a chip pair is listed. Option 1: every chip pair and channel that can make the connection (every link between the two positions, the resolver's first): phase 0 can fall back, but the other links were not checked by the resolver. Option 2: only the chip pair the layout determined (links[0]), with its channels: exactly what was verified, no fallback. The code shows option 1; option 2 drops the loop over position_links_ in record_channels. Which one?"],
          ]}
        />
      </Stack>
    </Stack>
  );
}

const SECTIONS = [
  { id: "overview", part: 1, label: "Overview", body: OverviewTab },
  { id: "descriptor", part: 1, label: "The MGD pipeline section", body: DescriptorTab },
  { id: "tiers", part: 1, label: "Tier A and Tier B rules", body: TiersTab },
  { id: "tierc", part: 1, label: "Tier C: the mapping", body: TierCTab },
  { id: "examples", part: 1, label: "Worked examples, MGD to blaze", body: ExamplesTab },
  { id: "plan", part: 2, label: "PR plan", body: PlanTab },
  { id: "impl-format", part: 2, label: "Format, Tier A and Tier B", body: ImplFormatTab },
  { id: "impl-mapper", part: 2, label: "Tier C: C1 and C2", body: ImplMapperTab },
  { id: "impl-binding", part: 2, label: "Tier C: C3, emit and bind", body: ImplBindingTab },
  { id: "impl-sat", part: 2, label: "Tier C: C4, the SAT loop", body: ImplSatTab },
  { id: "questions", part: 2, label: "Gaps and open questions", body: QuestionsTab },
] as const;

const PARTS = [
  { part: 1, title: "Part I: the design", sub: "What the MGD says, what each tier checks, and how a pipeline is mapped." },
  { part: 2, title: "Part II: implementation", sub: "PRs, files and code. Read Part I first." },
] as const;

export default function MgdPipelineDesign() {
  const dispatch = useCanvasAction();
  const t = useHostTheme();
  return (
    <Stack gap={28} style={{ padding: 24, maxWidth: 1240 }}>
      <Stack gap={10}>
        <H1>Pipeline-aware topology mapping</H1>
        <Text tone="secondary">
          Describe pipeline stages in the MGD, so the topology mapper only commits mappings the pipeline builder can build on.
          Design for [tt-metal#52016](https://github.com/tenstorrent/tt-metal/issues/52016). Blaze&apos;s builder is unchanged.
        </Text>
        <Row gap={8}>
          <Button
            variant="ghost"
            onClick={() => dispatch({ type: "openFile", path: "/data/rsong/tt-blaze/tt-metal/docs/scratch/mgd-pipeline-graph-design.md" })}
          >
            Open the full design doc
          </Button>
        </Row>
        {PARTS.map((p) => (
          <Row key={p.part} gap={6} wrap align="center">
            <Text size="small" weight="semibold" style={{ minWidth: 150 }}>
              {p.title}
            </Text>
            {SECTIONS.filter((s) => s.part === p.part).map((s) => (
              <span key={s.id}>
                <Pill size="sm" onClick={() => scrollToId(`sec-${s.id}`)}>
                  {s.label}
                </Pill>
              </span>
            ))}
          </Row>
        ))}
      </Stack>
      {PARTS.map((p) => (
        <div key={p.part} id={`part-${p.part}`}>
          <Stack gap={28}>
            <div
              style={{
                background: p.part === 1 ? t.fill.secondary : t.fill.tertiary,
                borderLeft: `4px solid ${p.part === 1 ? t.accent.primary : t.category.orange}`,
                borderRadius: 6,
                padding: "14px 18px",
              }}
            >
              <H1 style={{ margin: 0 }}>{p.title}</H1>
              <Text tone="secondary">{p.sub}</Text>
            </div>
            {SECTIONS.filter((s) => s.part === p.part).map((s, i) => {
              const Body = s.body;
              return (
                <div key={s.id} id={`sec-${s.id}`} style={{ scrollMarginTop: 12 }}>
                  <Stack gap={16}>
                    <div style={{ borderTop: `1px solid ${t.stroke.secondary}`, paddingTop: 16 }}>
                      <Text size="small" tone="tertiary">
                        {p.part === 1 ? "I" : "II"}.{i + 1}
                      </Text>
                      <H2 style={{ margin: 0 }}>{s.label}</H2>
                    </div>
                    <Body />
                  </Stack>
                </div>
              );
            })}
          </Stack>
        </div>
      ))}
    </Stack>
  );
}
