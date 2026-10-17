import {
  BarChart,
  Callout,
  CollapsibleSection,
  Code,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
  useState,
} from "cursor/canvas";

type Evidence = { signal: string; severity: string; when: string; runs: number; chips: string; detail: string; where: string };
type Example = { key: string; summary: string; host: string; score: number; evidence: Evidence[] };
type CategoryRow = {
  category: string;
  label: string;
  jql: string;
  tickets: number;
  jqlTickets: number;
  titleTickets: number;
  pairs: number;
  corroborated: number;
  weak: number;
  none: number;
  examples: Example[];
};
type ValidationRow = { category: string; label: string; cases: number; evidence: number; top1: number; family: number; none: number };
type FleetRow = { host: string; status: string; likely: string; confidence: string; where: string; scores: string; lastRun: string; open: string[]; auto: string[] };
type PostResetRow = { host: string; device: string; chip: string; runs: number; first: string; last: string; tickets: string[] };
type Data = {
  generated: string;
  windowDays: number;
  index: { machineRuns: number; rebuiltRuns: number; clusterRecords: number; logs: number; events: number; tickets: number };
  fleetCounts: Record<string, number>;
  untracked: FleetRow[];
  fleet: FleetRow[];
  categories: CategoryRow[];
  validation: ValidationRow[];
  validationTotal: { cases: number; evidence: number; top1: number; family: number };
  postReset: PostResetRow[];
};

const DATA: Data = __DATA__;

const STATUSES = ["FAILING", "FLAKY", "DEGRADED", "HEALTHY", "UNKNOWN"] as const;

function statusTone(status: string): "danger" | "warning" | "info" | "success" | "neutral" {
  if (status === "FAILING") return "danger";
  if (status === "FLAKY") return "warning";
  if (status === "DEGRADED") return "info";
  if (status === "HEALTHY") return "success";
  return "neutral";
}

function EvidenceLine({ ev }: { ev: Evidence }) {
  const theme = useHostTheme();
  return (
    <Stack gap={2} style={{ paddingLeft: 12, borderLeft: `1px solid ${theme.stroke.tertiary}` }}>
      <Text size="small">
        <Code>{ev.signal}</Code> {ev.severity} · {ev.when} · {ev.runs} run{ev.runs === 1 ? "" : "s"}
        {ev.chips ? ` · chips ${ev.chips}` : ""}
      </Text>
      <Text size="small" tone="secondary">{ev.detail}</Text>
      <Text size="small" tone="tertiary" truncate="start">{ev.where}</Text>
    </Stack>
  );
}

function FleetTable() {
  const [filter, setFilter] = useState<string>("FAILING");
  const rows = DATA.fleet.filter((r) => r.status === filter);
  return (
    <Stack gap={10}>
      <Row gap={8}>
        {STATUSES.filter((s) => DATA.fleetCounts[s]).map((s) => (
          <div key={s}>
            <Pill active={s === filter} onClick={() => setFilter(s)}>
              {`${s} ${DATA.fleetCounts[s]}`}
            </Pill>
          </div>
        ))}
      </Row>
      <Table
        headers={["Host", "Likely issue", "Where", "Category scores", "Last machine HC", "Open tickets", "HC bot auto-ticket"]}
        rows={rows.map((r) => [r.host, r.likely === "none" ? "none" : `${r.likely} (${r.confidence})`, r.where || "-", r.scores || "-", r.lastRun, r.open.join(", ") || "-", r.auto.join(", ") || "-"])}
        rowTone={rows.map((r) => statusTone(r.status))}
        striped
      />
    </Stack>
  );
}

export default function DeviceTriageOverview() {
  const v = DATA.validationTotal;
  return (
    <Stack gap={28} style={{ padding: 24, maxWidth: 1180 }}>
      <Stack gap={6}>
        <H1>Health-check logs vs DIIM categories</H1>
        <Text tone="secondary">
          Machine HC (/data/ci-provisioner) and Cluster HC (/data/dcamp) matched to {DATA.index.tickets} DIIM tickets.
          Index: {DATA.index.machineRuns.toLocaleString()} machine HC runs ({DATA.index.rebuiltRuns.toLocaleString()} failing
          runs from April on rebuilt from DCAMP bot-ticket attachments), {DATA.index.clusterRecords.toLocaleString()} cluster
          records, {DATA.index.logs.toLocaleString()} recover/validation logs, {DATA.index.events.toLocaleString()} events.
          Generated {DATA.generated} from tt-metal/device_triage.
        </Text>
      </Stack>

      <Row gap={32}>
        <Stat value={DATA.fleet.length} label={`hosts with machine HC in ${DATA.windowDays}d`} />
        <Stat value={DATA.fleetCounts.FAILING ?? 0} label="failing now" tone="danger" />
        <Stat value={DATA.fleetCounts.FLAKY ?? 0} label="flaky (passed last, failed ≥2× in 7d)" tone="warning" />
        <Stat value={DATA.untracked.length} label="failing/flaky with no open ticket for it" tone="danger" />
      </Row>

      <Stack gap={10}>
        <H2>Which DIIM categories the logs can see</H2>
        <Text tone="secondary">
          Each DIIM ticket in a category (JQL over title and description) that names an exabox host, scored against
          that host's logs in the {DATA.windowDays} days before it was filed.
        </Text>
        <BarChart
          categories={DATA.categories.map((c) => c.label)}
          series={[
            { name: "Corroborated by logs (score ≥ 1)", data: DATA.categories.map((c) => c.corroborated), tone: "success" },
            { name: "Weak (0.3–1)", data: DATA.categories.map((c) => c.weak), tone: "warning" },
            { name: "No log evidence", data: DATA.categories.map((c) => c.none), tone: "neutral" },
          ]}
          stacked
          horizontal
          height={240}
        />
        <Text size="small" tone="tertiary">
          Ticket × host pairs per category · x axis: number of ticket × host pairs · Source: device_triage index + Jira
          DIIM · triage replayed at ticket creation + 12h
        </Text>
      </Stack>

      <Grid columns={2} gap={24}>
        <Stack gap={10}>
          <H2>Prediction vs ticket title</H2>
          <Text tone="secondary">
            Tickets whose title names a device category. {v.evidence} of {v.cases} have log evidence for it; for {v.top1} it
            is the tool's top prediction ({v.family} counting PCIe and reset as one family: a chip that drops off the bus and
            the reset that can't bring it back).
          </Text>
          <Table
            headers={["Category", "Cases", "Log evidence", "Top prediction", "Top, PCIe≈reset", "No signal"]}
            rows={DATA.validation.map((r) => [r.label, r.cases, r.evidence, r.top1, r.family, r.none])}
            columnAlign={["left", "right", "right", "right", "right", "right"]}
          />
        </Stack>
        <Stack gap={10}>
          <H2>Why GDDR, Tensix and BMC are invisible</H2>
          <Callout tone="neutral" title="The evidence lives elsewhere">
            <Stack gap={6}>
              <Text size="small">GDDR EDC tickets come from tt-triage in CI hangs (DIIM-597); DRAM aliasing from a tt-blaze stress test (DIIM-550).</Text>
              <Text size="small">Tensix ND and hangs come from model workloads (DIIM-575 → SYS-5041); HC tiers barely run compute. The
                workload logs attached to DIIM tickets do show it: AICLK settle timeouts in 12 tickets, physical-core timeouts in 7.</Text>
              <Text size="small">BMC failures: recover logs only show ipmitool resets that worked; a host with a dead BMC or OS sends no HC runs.</Text>
            </Stack>
          </Callout>
          <Callout tone="info" title="What HC sees well">
            <Text size="small">
              Trays and chips missing from PCIe (exact UBB / BDF), resets that do not recover, POST_RESET per device, ETH ports
              down per chip, firmware mismatches.
            </Text>
          </Callout>
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>Failing or flaky with no open ticket for the likely issue</H2>
        <Text tone="secondary">
          Open tickets: human DIIM/DCAMP tickets about the host (PCIe and reset count as one issue). HC bot auto-tickets
          are shown but don't count: nobody has triaged them.
        </Text>
        <Table
          headers={["Host", "Status", "Likely issue", "Where", "Category scores", "Open tickets", "HC bot auto-ticket"]}
          rows={DATA.untracked.map((r) => [r.host, r.status, `${r.likely} (${r.confidence})`, r.where || "-", r.scores, r.open.join(", ") || "-", r.auto.join(", ") || "-"])}
          rowTone={DATA.untracked.map((r) => statusTone(r.status))}
          striped
        />
      </Stack>

      <Stack gap={10}>
        <H2>Chips that keep failing POST_RESET</H2>
        <Text tone="secondary">
          Cluster HC recover logs, "POST_RESET failed for device N"; chip resolved from the host's triage tables.
        </Text>
        <Table
          headers={["Host", "Device", "Chip", "Recover runs", "First → last", "Tickets naming the host"]}
          rows={DATA.postReset.map((r) => [r.host, r.device, r.chip, r.runs, `${r.first} → ${r.last}`, r.tickets.join(", ") || "none"])}
          columnAlign={["left", "left", "left", "right", "left", "left"]}
          rowTone={DATA.postReset.map((r) => (r.tickets.length ? undefined : "danger"))}
        />
      </Stack>

      <Stack gap={10}>
        <H2>Ticket ↔ log matches</H2>
        <Text tone="secondary">Corroborated tickets per category, with the strongest log lines and where to find them.</Text>
        {DATA.categories
          .filter((c) => c.examples.length)
          .map((c) => (
            <div key={c.category}>
              <CollapsibleSection title={c.label} count={c.examples.length} defaultOpen={c.category === "pcie"}>
                <Stack gap={14}>
                  {c.examples.map((ex) => (
                    <div key={`${ex.key}-${ex.host}`}>
                      <Stack gap={6}>
                        <Text weight="semibold">
                          {ex.key} · {ex.host} · score {ex.score.toFixed(1)}
                        </Text>
                        <Text size="small" tone="secondary">{ex.summary}</Text>
                        {ex.evidence.map((ev, i) => (
                          <div key={i}>
                            <EvidenceLine ev={ev} />
                          </div>
                        ))}
                      </Stack>
                    </div>
                  ))}
                </Stack>
              </CollapsibleSection>
            </div>
          ))}
      </Stack>

      <Stack gap={10}>
        <H2>Fleet</H2>
        <FleetTable />
      </Stack>

      <Text size="small" tone="tertiary">
        Regenerate: cd tt-metal/device_triage && python3 -m device_triage index && python3 -m device_triage canvas · per-host
        detail: python3 -m device_triage triage HOST
      </Text>
    </Stack>
  );
}
