# 10 · Progress Report — Tracking an Optimization Loop with an HTML Dashboard

`07_METHODOLOGY` covers *how* to gate each change. `09_PROFILING_AND_OP_ANALYSIS` covers
*what* the per-op data says. This file covers *how to keep score across iterations* — a
single-file HTML dashboard that holds the baseline, target, every experiment, and the
current top buckets. It's how you avoid forgetting which configs you've already tried at
hour 5 of a long loop.

---

## 1. Why a progress report at all

The persistence contract (AGENT_INDEX) says you keep iterating until the metric is met or
the floor is reached. By the 5th-10th experiment you will:

- forget whether a config was already tried,
- conflate noise wins with real wins,
- lose track of which bucket each change targeted,
- mis-remember the baseline number.

A small static HTML page solves all four. It is the cheapest possible "session memory"
that survives context resets and lets a reviewer audit the loop later.

---

## 2. What the report must contain

Minimum fields the dashboard tracks (each tied to a section of these guidelines):

| Field | Why | Source |
|---|---|---|
| Baseline FPS / wall time | the number we are improving over | first reproducible measurement before changes (07 §1) |
| Goal FPS / wall time | the stop condition | the task statement / AGENT_INDEX persistence contract |
| Current best | what we have right now | best `keep`'d experiment |
| Speedup × baseline | progress toward goal | `current_best / baseline` |
| Per-experiment row: status, FPS, inference_s, commit, description | the audit trail | one per `run_experiment` |
| Top op buckets from latest profile | what to attack next | bucket script output (09 §3) |

Status taxonomy (matches `log_experiment`):
- `baseline` — the reference run; never changed.
- `keep` — primary metric improved by more than noise floor.
- `discard` — primary metric did not improve, **or** improved within noise.
- `crash` — run failed (OOM, CB clash, PCC fail, harness bug).

---

## 3. The single-file dashboard

The dashboard is one self-contained HTML file. **Chart.js via CDN is the only external
dependency.** No backend, no build step.

Suggested location in the repo:

```
perf_progress/
  progress.html      # the dashboard (this file)
  GOAL.md            # the target, baseline source, stop conditions
  serve.sh           # one-liner: python3 -m http.server 8899
```

### Skeleton — the data block

The dashboard renders everything from a single `DATA = { ... }` object near the bottom of
the HTML. Append a row after every experiment, save, refresh browser. Example shape:

```js
const DATA = {
  goal: { fps: 1.000, inference_s: 1.000, label: "≥ 1 FPS" },
  baseline: {
    fps: 0.0547,
    inference_s: 18.30,
    when: "2026-06-04 (unprofiled, reproduced)",
    source: "test_e2e_perf_2cq.py MODE=wall, median-of-2 18.2993 / 18.2975"
  },
  experiments: [
    { when: "2026-06-04 02:53", status: "baseline",
      fps: 0.0547, inference_s: 18.30, commit: "8614186",
      description: "Reproduced production wall." },
    { when: "2026-06-04 02:19", status: "keep",
      fps: 0.0717, inference_s: 13.94, commit: "abc1234",
      description: "UPLOAD_CHUNK_QUERIES 2048→2304. Within noise; kept tentatively." },
    { when: "2026-06-04 02:38", status: "crash",
      fps: null, inference_s: null, commit: "local",
      description: "bf8 value: PCC 0.9992 but trace capture asserts on event sync." },
  ],
  buckets_when: "2026-06-03 23:00",
  buckets_source: "ops_perf_results_2026_06_03_23_05.csv",
  buckets: [
    { op: "GridSampleOperation", ms: 4184, pct: 45.5, count: 1350, us_per_call: 3099,
      notes: "MSDeformAttn 6 layers × 44 chunks × 5 levels" },
    { op: "SliceDeviceOperation", ms: 1313, pct: 14.3, count: 14384, us_per_call: 91,
      notes: "Per-chunk slices; 06 §4 fusion target" },
  ],
};
```

### What the page renders

- **4 KPI cards** at the top: baseline FPS, current best FPS, goal FPS, progress %.
- **Line chart** of FPS over experiments, with the goal and baseline as dashed reference
  lines. Hover shows commit + description.
- **Experiment table** color-coded by status (`keep` green, `discard`/`crash` muted).
  Includes Δ FPS vs baseline per row.
- **Top buckets table** (from `09 §3`). Renders `notes` so you can tag which lever a row
  needs (e.g. "06 §4 fusion target", "irreducible compute").
- **Footer** linking back to the 4 methodology gates.

The whole page is ~300 lines and renders in milliseconds. No state outside `DATA`.

---

## 4. Accessing the dashboard

Three options, in order of practicality on a shared dev box:

### A. Public HTTP on the remote — anyone on the network
```bash
cd perf_progress
python3 -m http.server 8899 --bind 0.0.0.0
# anyone reaches it at http://<remote-ip>:8899/progress.html
```

### B. SSH tunnel — private and secure
```bash
# on the remote:
python3 -m http.server 8899 --bind 127.0.0.1
# on the laptop:
ssh -N -L 8899:localhost:8899 <user>@<remote-host>
# then browse http://localhost:8899/progress.html
```

### C. Copy and open locally
```bash
scp <user>@<remote-host>:.../perf_progress/progress.html .
open progress.html
```

A `serve.sh` one-liner in `perf_progress/` is recommended so reviewers don't need to
remember the command.

---

## 5. Update flow — what to do after every experiment

A consistent update pattern keeps the dashboard honest:

```
1. Run the experiment (MODE=2cq / MODE=wall / etc, EXP_TAG=<short-name>).
2. Record the result locally first (one bullet in the agent's notes).
3. Edit DATA.experiments in progress.html — append at the bottom:
     { when: "<UTC iso>", status: "keep|discard|crash|baseline",
       fps: <number-or-null>, inference_s: <number-or-null>,
       commit: "<short-sha-or-local>",
       description: "<one sentence, including which guideline section the lever came from>" }
4. If a new profile was captured, update DATA.buckets from the bucket script (09 §3) so
   the next iteration knows the new top bucket.
5. Save the file. Reviewer refreshes the browser.
```

The `description` field is the most important. Make it specific enough that 24 hours later
you still know which change it refers to. Bad: "tried bf8". Good: "bf8 value output: PCC
0.9992 in encoder, but trace capture asserts on event sync — 04 §5 style trap".

### What NOT to put in DATA.experiments

- Runs that crashed before reaching the device (env-var typos, missing checkpoint, tmux
  paste error). Note those in the agent's working memory but they're not real
  experiments — they pollute the chart.
- Half-finished sweep runs. Wait until you have a measured `inference_s`.

---

## 6. Hygiene rules

- **One row per experiment.** Re-running the same config to confirm noise is still one
  row — note "reproduction" in description.
- **Never delete rows.** Discards are valuable. They document what you tried.
- **Mark the noise floor explicitly.** When a `keep` row's delta vs baseline is within run
  noise (see 07 §3: ~50 µs sweeps / ~50–200 ms wall), say so in the description
  ("within noise — kept tentatively"). This prevents you (or a reviewer) from celebrating
  a 0.1% improvement as a win.
- **Update baseline only when forced.** If you discover the baseline measurement was
  itself wrong (e.g. a profiled wall vs unprofiled wall), update the baseline number and
  add a `baseline` row dated when the correction was made — do not silently rewrite
  history.
- **Top buckets get refreshed after every kept change.** Per 09 §8, the next bucket *is*
  the new bottleneck.

---

## 7. The 1-line authoring rule

If updating the dashboard takes more than 30 seconds, the dashboard is wrong. The whole
point is to lower the bar to recording results — anything more and you'll skip it under
deadline pressure, which is precisely when you need it.

The recommended cadence: **edit `DATA`, save, refresh, move on.** No PRs, no migrations,
no schema changes. If the schema needs to grow, add an optional field that defaults to
sensible behavior.

---

## 8. Quick reference

| Goal | How |
|---|---|
| Where to put it | `perf_progress/progress.html` + `GOAL.md` |
| External deps | Chart.js via CDN — that's it |
| Update step | edit the `DATA = {...}` block, save, refresh |
| Serve publicly | `python3 -m http.server 8899 --bind 0.0.0.0` |
| Serve privately | `--bind 127.0.0.1` + SSH `-L` tunnel |
| Open locally | scp the single file, open in browser |
| Status values | `baseline`, `keep`, `discard`, `crash` |
| Required per row | when, status, fps, inference_s, commit, description |
| Mandatory after a kept change | refresh `DATA.buckets` from the latest CSV (09 §3) |
| Noise floor reminder | 07 §3 — call it out in the description if the win is within it |
