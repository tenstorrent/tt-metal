// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

import {
  BarChart,
  Callout,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const SOURCE =
  "Source: docs/scratch/pipeline_placement_experiments, REPORT.md section 5.3 and results/rerun_2026-10-27/carve_comparison.md · one 64-core machine, 2026-10-27 · 60 s cap per case";

const GROUPS: { name: string; cases: number; planner: string; footprint: number; resolver: string }[] = [
  { name: "Original carve cases", cases: 17, planner: "0 (17 budget exhausted)", footprint: 16, resolver: "0.029 s" },
  { name: "Heterogeneous sweep, 60 to 95% fill", cases: 30, planner: "0 (30 budget exhausted)", footprint: 30, resolver: "0.001 s" },
  { name: "Single mesh, no 1x1 stages", cases: 45, planner: "0 (stage-0 rule; 2 with it off)", footprint: 43, resolver: "0.029 s" },
  { name: "Heterogeneous multi-mesh", cases: 12, planner: "0 (stage-0 rule; 0 with it off)", footprint: 12, resolver: "0.001 s" },
  { name: "Multi-mesh, no 1x1 stages", cases: 28, planner: "0 (stage-0 rule; 0 with it off)", footprint: 26, resolver: "0.036 s" },
];

const BOTH_FINISH: string[][] = [
  ["4x8: 1x1, 2x4, 2x4, 2x2, 1x1", "32", "0.009 + 0.000", "0.001 + 0.000", "~9x"],
  ["4x8 ring of 32 x 1x1", "32", "0.038 + 0.001", "0.004 + 0.001", "~8x"],
  ["4x8 ring of 16 x 1x2 (planner's stage-0 rule off)", "32", "0.036 + 0.000", "0.002 + 0.000", "~18x"],
  ["4x4: 1x1, 3x 2x2, 1x1", "16", "0.000 + 0.000", "0.000 + 0.000", "both instant"],
];

const PLANNER_FAILS: { row: string[]; tone: "danger" | "warning" }[] = [
  { row: ["4x4: 1x1, 3x 2x2 (feasible)", "1 mesh", "wrong reject: no-cycle-of-length-13", "0.000 + 0.000"], tone: "danger" },
  { row: ["4x6 and 4x8: 1x1, ring of 2x2, 1x1 (feasible)", "1 mesh", "wrong reject: no-alignment", "0.001 + 0.000"], tone: "danger" },
  { row: ["8x8 fill: 1x1, 15x 2x2, 3x 1x1", "1 mesh", "budget exhausted", "0.026 + 0.000"], tone: "warning" },
  { row: ["16x16: 1x1, 62x 2x2, 1x1", "1 mesh", "budget exhausted", "0.13 + 0.004"], tone: "warning" },
  { row: ["Random 8x8, 30 stages", "1 mesh", "budget exhausted", "0.60 + 0.001"], tone: "warning" },
  { row: ["Random 8x16, 37 stages", "1 mesh", "budget exhausted", "5.6 + 0.001"], tone: "warning" },
  { row: ["Ring of 112 x 1x1, 1 cable per pair", "4 galaxies", "budget exhausted", "9.9 + 0.020"], tone: "warning" },
  { row: ["Domino ring, 128 x 1x2, 4 cables per pair (2 seeds)", "8 galaxies", "budget exhausted", "2.6 / 21 + 0.036"], tone: "warning" },
  { row: ["Mixed shapes, 56 / 62 stages, 4 cables per pair", "8 galaxies", "budget exhausted", "36 / 24 + 0.004"], tone: "warning" },
  { row: ["7x7: 1x1, 10x 2x2 (infeasible)", "1 mesh", "budget exhausted, no verdict", "proved no, 2.1"], tone: "warning" },
  { row: ["5x5 ring of 25 x 1x1 (infeasible)", "1 mesh", "budget exhausted, no verdict", "proved no, 0.003"], tone: "warning" },
];

type Solve = { s: number; note?: string } | { failed: string };

const SOLVE_TIMES: { name: string; fabric: string; planner: Solve; footprint: Solve }[] = [
  { name: "4x4: 1x1, 3x 2x2, 1x1", fabric: "16 slices", planner: { s: 0 }, footprint: { s: 0 } },
  { name: "4x8: 1x1, 2x4, 2x4, 2x2, 1x1", fabric: "32 slices", planner: { s: 0.009 }, footprint: { s: 0.001 } },
  { name: "4x8 ring of 32 x 1x1", fabric: "32 slices", planner: { s: 0.039 }, footprint: { s: 0.005 } },
  {
    name: "4x8 ring of 16 x 1x2 (stage-0 rule off)",
    fabric: "32 slices",
    planner: { s: 0.036 },
    footprint: { s: 0.002 },
  },
  {
    name: "4x4: 1x1, 3x 2x2 (feasible)",
    fabric: "16 slices",
    planner: { failed: "wrong reject: no-cycle-of-length-13" },
    footprint: { s: 0 },
  },
  {
    name: "4x6 / 4x8: 1x1, ring of 2x2, 1x1 (feasible)",
    fabric: "24 / 32 slices",
    planner: { failed: "wrong reject: no-alignment" },
    footprint: { s: 0.001 },
  },
  { name: "5x5 ring of 25 x 1x1 (infeasible)", fabric: "25 slices", planner: { failed: "budget exhausted, no verdict" }, footprint: { s: 0.003, note: "proved no" } },
  { name: "8x8 fill: 1x1, 15x 2x2, 3x 1x1", fabric: "64 slices", planner: { failed: "budget exhausted" }, footprint: { s: 0.026 } },
  { name: "16x16: 1x1, 62x 2x2, 1x1", fabric: "256 slices", planner: { failed: "budget exhausted" }, footprint: { s: 0.134 } },
  { name: "Random 8x8, 30 stages", fabric: "64 slices", planner: { failed: "budget exhausted" }, footprint: { s: 0.601 } },
  { name: "7x7: 1x1, 10x 2x2 (infeasible)", fabric: "49 slices", planner: { failed: "budget exhausted, no verdict" }, footprint: { s: 2.1, note: "proved no" } },
  { name: "Random 8x16, 37 stages", fabric: "128 slices", planner: { failed: "budget exhausted" }, footprint: { s: 5.6 } },
  { name: "Ring of 112 x 1x1, 1 cable per pair", fabric: "4 galaxies", planner: { failed: "budget exhausted" }, footprint: { s: 9.9 } },
  { name: "Domino ring, 128 x 1x2, seed 0", fabric: "8 galaxies", planner: { failed: "budget exhausted" }, footprint: { s: 2.6 } },
  { name: "Domino ring, 128 x 1x2, seed 1", fabric: "8 galaxies", planner: { failed: "budget exhausted" }, footprint: { s: 21.1 } },
  { name: "Mixed shapes, 62 stages", fabric: "8 galaxies", planner: { failed: "budget exhausted" }, footprint: { s: 24.4 } },
  { name: "Mixed shapes, 56 stages", fabric: "8 galaxies", planner: { failed: "budget exhausted" }, footprint: { s: 35.7 } },
];

const LOG_MIN = Math.log10(0.001);
const LOG_MAX = Math.log10(60);
const TICKS: { v: number; label: string }[] = [
  { v: 0.001, label: "1 ms" },
  { v: 0.01, label: "10 ms" },
  { v: 0.1, label: "100 ms" },
  { v: 1, label: "1 s" },
  { v: 10, label: "10 s" },
  { v: 60, label: "60 s cap" },
];

function pos(v: number): number {
  const x = (Math.log10(Math.max(v, 0.001)) - LOG_MIN) / (LOG_MAX - LOG_MIN);
  return Math.min(1, Math.max(0, x)) * 100;
}

function fmt(v: number): string {
  if (v < 0.001) return "<1 ms";
  if (v < 1) return `${Math.round(v * 1000)} ms`;
  return `${v} s`;
}

function SolveTimeChart() {
  const t = useHostTheme();
  const plannerColor = t.category.orange;
  const footprintColor = t.accent.primary;
  const labelCol = 300;
  const barH = 12;

  const bar = (solve: Solve, color: string, who: string) => {
    if ("failed" in solve) {
      return (
        <div
          title={`${who}: ${solve.failed}`}
          style={{
            height: barH,
            width: "100%",
            border: `1px dashed ${t.stroke.secondary}`,
            borderRadius: 2,
            display: "flex",
            alignItems: "center",
            paddingLeft: 6,
            boxSizing: "border-box",
            fontSize: 10,
            color: t.text.tertiary,
            whiteSpace: "nowrap",
            overflow: "hidden",
          }}
        >
          did not finish: {solve.failed}
        </div>
      );
    }
    const w = pos(solve.s);
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 6, height: barH }}>
        <div
          title={`${who}: ${fmt(solve.s)}`}
          style={{ height: barH, width: `max(2px, ${w}%)`, background: color, borderRadius: 2, flexShrink: 0 }}
        />
        <span style={{ fontSize: 10, color: t.text.secondary, whiteSpace: "nowrap" }}>
          {fmt(solve.s)}
          {solve.note ? `, ${solve.note}` : ""}
        </span>
      </div>
    );
  };

  return (
    <Stack gap={8}>
      <Row gap={16}>
        <Row gap={6} align="center">
          <div style={{ width: 12, height: 12, background: plannerColor, borderRadius: 2 }} />
          <Text size="small">carve.py planner + resolver</Text>
        </Row>
        <Row gap={6} align="center">
          <div style={{ width: 12, height: 12, background: footprintColor, borderRadius: 2 }} />
          <Text size="small">Footprint SAT solver + resolver</Text>
        </Row>
        <Row gap={6} align="center">
          <div style={{ width: 12, height: 10, border: `1px dashed ${t.stroke.secondary}`, borderRadius: 2 }} />
          <Text size="small">carve.py planner did not finish</Text>
        </Row>
      </Row>
      <div style={{ display: "grid", gridTemplateColumns: `${labelCol}px 1fr`, rowGap: 10, columnGap: 12 }}>
        {SOLVE_TIMES.map((c) => (
          <div key={c.name} style={{ display: "contents" }}>
            <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
              <span style={{ fontSize: 12, color: t.text.primary }}>{c.name}</span>
              <span style={{ fontSize: 10, color: t.text.tertiary }}>{c.fabric}</span>
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 3,
                justifyContent: "center",
                borderLeft: `1px solid ${t.stroke.tertiary}`,
                paddingLeft: 0,
              }}
            >
              {bar(c.planner, plannerColor, "carve.py planner")}
              {bar(c.footprint, footprintColor, "footprint")}
            </div>
          </div>
        ))}
        <div />
        <div style={{ position: "relative", height: 16, borderTop: `1px solid ${t.stroke.secondary}` }}>
          {TICKS.map((k) => (
            <span
              key={k.label}
              style={{
                position: "absolute",
                left: `${pos(k.v)}%`,
                transform: k.v === 60 ? "translateX(-100%)" : k.v === 0.001 ? "none" : "translateX(-50%)",
                top: 2,
                fontSize: 10,
                color: t.text.tertiary,
                whiteSpace: "nowrap",
              }}
            >
              {k.label}
            </span>
          ))}
        </div>
      </div>
    </Stack>
  );
}

const SCALING: string[][] = [
  ["2", "12 to 32", "18 of 18", "0.023 s", "1.1 s", "0.10 to 8.1 s (3 cases)"],
  ["4", "26 to 64", "16 of 18 (2 open at the cap)", "0.78 s", "11 s", "51 s (1 case)"],
  ["8", "56 to 128", "4 of 4", "23 s", "36 s", "none in this set"],
];

export default function CarvePlannerVsFootprint() {
  const totalCases = GROUPS.reduce((s, g) => s + g.cases, 0);
  const totalFootprint = GROUPS.reduce((s, g) => s + g.footprint, 0);

  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>Blaze carve planner vs the footprint SAT solver</H1>
        <Text tone="secondary">
          Blaze's ring carve planner (<Code>plan_ring_carve</Code> in <Code>pipeline_builder/carve_planner.py</Code>)
          against the carve as a footprint mapping through tt-metal's topology solver (SAT, with resource
          completeness), each followed by the resolver. 132 carve-requiring cases.
        </Text>
      </Stack>

      <Callout tone="info" title="The finding">
        Once a solve involves more than 32 host slices (any 8x8 or larger mesh, or more than one 4x8 galaxy), the
        carve.py planner runs out of its 200k-expansion search budget on every case and places none of the 132
        carve-requiring cases. The SAT-based footprint solver decided 127 of the 132 within 60 s, most in under a
        second, the largest (8 galaxies, 128 stages over 256 slices) in 2.6 to 21 s. Where both finish, on small
        rings of 32 slices or fewer, the SAT solver is about 9x to 18x faster. With more galaxies the footprint solver
        itself slows down (seconds to tens of seconds on 8 galaxies), and forks were tested on one small case only.
      </Callout>

      <Grid columns={4} gap={16}>
        <Stat value={`0 / ${totalCases}`} label="placed by the carve.py planner" tone="danger" />
        <Stat value={`${totalFootprint} / ${totalCases}`} label="decided by the footprint solver within 60 s" tone="success" />
        <Stat value="9x to 18x" label="footprint speedup where both finish (32 slices or fewer)" />
        <Stat value="36 ms" label="worst resolver time after a footprint carve" />
      </Grid>

      <Divider />

      <Stack gap={12}>
        <H2>Solve time per case: carve + resolver</H2>
        <SolveTimeChart />
        <Text size="small" tone="tertiary">
          Wall-clock seconds for the carve plus the resolver (x axis, log scale, 1 ms to the 60 s cap) per case (y
          axis). A dashed bar is a case the carve.py planner did not finish: it ran out of its 200k-expansion budget
          (in under 0.06 s) or rejected a feasible case. {SOURCE}
        </Text>
      </Stack>

      <Stack gap={12}>
        <H2>Cases placed or decided, by group</H2>
        <Table
          headers={["Group", "Cases", "carve.py planner + resolver: placed", "Footprint + resolver: decided", "Resolver after footprint, worst"]}
          columnAlign={["left", "right", "left", "right", "right"]}
          rows={[
            ...GROUPS.map((g) => [g.name, String(g.cases), g.planner, String(g.footprint), g.resolver]),
            ["All", String(totalCases), "0", String(totalFootprint), "0.036 s"],
          ]}
          rowTone={[...GROUPS.map(() => undefined), "info"]}
        />
      </Stack>

      <Grid columns="1fr 1fr" gap={24} align="start">
        <Stack gap={12}>
          <H3>Where both finish: the only finite speedups</H3>
          <Table
            headers={["Case", "Slices", "carve.py planner (s)", "Footprint (s)", "Speedup"]}
            columnAlign={["left", "right", "right", "right", "right"]}
            rows={BOTH_FINISH}
            rowTone={["success", "success", "success", undefined]}
          />
          <Text size="small" tone="tertiary">
            Seconds as carve + resolver. Every case the planner completed has 32 host slices or fewer.
          </Text>
        </Stack>
        <Stack gap={12}>
          <H3>Why there is no speedup on the large cases</H3>
          <Text>
            The planner does not run slowly on them; it gives up. It stops at its 200k-expansion budget in under
            0.06 s and reports <Code>budget-exhausted</Code>. Grid-shaped meshes have no degree-2 chains for it to
            contract, so its cycle search becomes a raw Hamiltonian-cycle search the budget was never sized for. Its
            budget comment cites about 900 expansions for SC16, a ring of columns.
          </Text>
          <Text>
            Two rules reject cases before the search: stage 0 must be one slice (85 of the 132 cases have no 1x1
            stage), and first-fit grouping reports <Code>no-alignment</Code> on 16 feasible cases.
          </Text>
        </Stack>
      </Grid>

      <Stack gap={12}>
        <H2>Per case: where the planner fails</H2>
        <Table
          headers={["Case", "Fabric", "carve.py planner", "Footprint + resolver (s)"]}
          columnAlign={["left", "left", "left", "right"]}
          rows={PLANNER_FAILS.map((r) => r.row)}
          rowTone={PLANNER_FAILS.map((r) => r.tone)}
        />
        <Text size="small" tone="tertiary">
          Red: the planner wrongly rejects a feasible case. Amber: it runs out of budget with no verdict. Galaxies
          are 4x8 meshes of host slices. {SOURCE}
        </Text>
      </Stack>

      <Divider />

      <Stack gap={12}>
        <H2>The worry: the footprint solver as galaxies are added</H2>
        <Row gap={24} align="start" wrap>
          <Stack gap={8} style={{ flex: "1 1 360px" }}>
            <BarChart
              categories={["2 galaxies", "4 galaxies", "8 galaxies"]}
              series={[
                { name: "Median, feasible cases", data: [0.023, 0.78, 23] },
                { name: "Slowest feasible case", data: [1.1, 11, 36], tone: "warning" },
              ]}
              valueSuffix=" s"
              showValues
              height={240}
              referenceLines={[{ value: 60, label: "60 s cap", tone: "danger" }]}
            />
            <Text size="small" tone="tertiary">
              Footprint + completeness solve time in seconds (y axis) by number of 4x8 galaxies (x axis); the 40
              carve-requiring multi-mesh rings (heterogeneous and no-1x1 groups). {SOURCE}
            </Text>
          </Stack>
          <Stack gap={8} style={{ flex: "1 1 420px" }}>
            <Table
              headers={["Galaxies", "Stages", "Decided", "Median feasible", "Slowest feasible", "Infeasibility proofs"]}
              columnAlign={["right", "right", "left", "right", "right", "right"]}
              rows={SCALING}
              rowTone={[undefined, "warning", "warning"]}
            />
          </Stack>
        </Row>
        <Callout tone="warning" title="What the data does not cover yet">
          On 8 galaxies with 1 to 2 cables per pair, most single-slice rings run to the time cap for every engine.
          Forks were tested on one designed 6x6 case only (footprint 0.008 s; the planner port handles rings only).
          There is no fork or multi-galaxy fork comparison at scale.
        </Callout>
      </Stack>

      <Stack gap={8}>
        <H3>Method</H3>
        <Text size="small" tone="secondary">
          The planner column is <Code>topo/carve_planner_port.hpp</Code>, a C++ port of blaze's{" "}
          <Code>plan_ring_carve</Code> with blaze's 200k-expansion and 64-cycle budgets. It reproduces blaze's Python
          verdicts on the designed cases (<Code>no-cycle-13</Code>, both <Code>no-alignment</Code> rings, the mixed4
          cablings); blaze's Python code itself was run only on those designed rings. The footprint column is{" "}
          <Code>solve_topology_mapping</Code> under Auto (SAT on every case) with the uncommitted resource
          completeness patch, conflict budget removed so a "no" is a proof. Every returned solution was re-checked
          independently. Cap 60 s per case, one 64-core machine.
        </Text>
      </Stack>
    </Stack>
  );
}
