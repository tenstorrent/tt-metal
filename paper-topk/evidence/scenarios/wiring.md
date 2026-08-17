# Model-scenarios region: sweep + ledger wiring design

Implementation-ready design for measuring **named LLM model scenarios** (the shapes real
models pass to `ttnn.topk` / `ttnn.experimental.topk_large_indices` / MoE gates / sampling)
through the canonical pipeline, and rendering them as a new `MODEL_SCENARIOS` ledger region.
All file references are on branch `nkapre/sorting` as of 2026-08-16 (HEAD 22563f240c2).

Files:
- Sweep: `tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py` (2002 lines)
- Renderer: `tests/ttnn/unit_tests/operations/reduction/_topk_ledger_render.py` (192 lines)
- Ledger: `TOPK_LEDGER.html` (273 lines)

---

## 1. Ground truth: what the existing machinery already gives us

### 1.1 Competition machinery (reused wholesale)

- **Per-cell subprocess + watchdog**: `run_cell()` (:1043–1143) launches
  `python -m tracy -r -v <this file>` per cell with the spec riding
  `CANONICAL_SWEEP_CHILD_SPEC` (:137), a `--timeout` watchdog (default 900 s, :1870),
  DPRINT/Watcher env scrubbing (:1062–1064), and provenance stamping
  (`provenance_stamp()`, :1019–1040: HEAD sha, `git diff --stat` md5, `_ttnn.so`
  mtime+md5). Result JSON lands at `results/{cell_id}.{arm}.t{trial}.json`
  (`result_path`, :1008) with `ns_median`, `cores`, `max_abs_err`, `status`, `error`,
  `notes` (schema built at :1066–1076).
- **Correctness gating**: `run_child()` (:488–558) — first call is correctness-checked
  *before* any timing; `wrong` ⇒ status `WRONG`, timing never recorded (:526–529).
  Exception in `setup`/`first_call` ⇒ `UNSUPPORTED`; later ⇒ `FAILED` (:551).
- **The three engines we need already exist as child ops**:
  - `topk_routed` / `topk_stock` (:584–618): both are `ttnn.topk` on `(1,1,batch,n)`
    TILE input; `largest` is the router (`largest = op == "topk_routed"`, :592).
    Correctness is exact (value multiset vs `torch.topk` + index self-consistency,
    :602–617) and fully **batched** — `gather(-1, idx)` and per-dim sorts work for
    any `batch`.
  - `topk_large_indices` (:638–679): builds `(batch, n)` ROW_MAJOR bf16, supports
    `valid_length` and `num_slices` (:649–654), order-insensitive gathered-vs-torch
    correctness, `strict` flag gates timing (:673–677). Also batched.
  - `moe_gate` (:681–743): fixed 256-expert DeepSeek geometry, `index_match_frac`
    correctness (tie-tolerant, **not** strict-gated).
- **Composite attribution**: `parse_tracy_composite()` (:932–1005) sums ALL device ops
  per iteration anchored on the top-k op's row count — this is what makes
  `topk_routed` (untilize+TopkLargeIndices+gather+tilize+…) and `topk_stock`
  (FillPad+TopK) honest one-number layers. Selected by `cell["composite"]` in
  `run_cell` (:1124–1127). Works unchanged for multi-row cells.
- **Layer table**: `COMPETITION_LAYERS` (:286–292) —
  `(layer, child_op, composite, arm, seed_index)` with **pinned** seed indices
  (op=0, routed=1, stocknow=2, prebranch=3, opstock=4); seeds via
  `competition_seed(k, w, seed_index)` (:341–344).
- **Cell building / ids**: `build_competition_cells()` (:1546–1625); competition cell
  id format `comp_{layer}_k{k}_w{w}[_p{num_slices}]` (:1596). `--layers-competition`
  (:1885–1889) subsets layers; `--ks/--ns` (:1850–1851) override the grid;
  `--op-num-slices` (:1862–1869) applies to the `op` layer only (:1595).
- **Run loop**: `run_competition()` (:1628–1699) — fixed layer-major order, header-arm
  layers last, `_done()` resume (:1646–1654: `MEASURED/UNSUPPORTED/WRONG` skip,
  `FAILED` retried), table rewritten after **every** cell (:1692), header restored in
  `finally` (:1693–1696). Competition runs **one trial per cell** (`run_cell(cell,
  cell["layer"], 0, args)`, :1685) — the number is the median over in-cell iters.
- **CSV**: `write_competition_reports()` (:1775–1842) → `competition_table.csv`
  (:1794) with per-layer `{layer}_us` / `{layer}_cores` columns.

### 1.2 Can cells carry rows>1? — YES

- The classic grid's `add(op, batch, …)` (:400) carries `batch`, embedded in cell ids
  as `{op}_b{batch}xN{n}_…` (:405). `LARGE_INDICES_ANCHORS` (:228–232) already runs
  rows=640 (`prod_prefill`) and rows=2 (`prod_bounded_cache`) through the pipeline.
- Competition mode fixes `batch=1` except `opstock`, which uses `batch=2` as the
  as-shipped proxy (:1609).
- Child harnesses are batched throughout: `topk`/`topk_routed`/`topk_stock` build
  `(1,1,batch,n)` (:594), `topk_large_indices` builds `(batch,n)` (:640), and every
  correctness function uses batched torch ops. **A scenario with rows>1 needs zero
  child changes.** rows>1 through the `op` engine exercises the row-parallel
  multi-row path (with the new chunk-skip); rows>1 through `routed` exercises the
  composite whose inner TopkLargeIndices is likewise row-parallel.

### 1.3 How the competition bounds ms-class stock cells (the convention to reuse)

- `COMPETITION_SLOW_WK = 1<<24`, `COMPETITION_DEFAULT_ITERS = 5`,
  `COMPETITION_SLOW_ITERS = 3` (:265–267), applied to stock layers only at
  :1587–1589 (`if layer in ("stocknow","prebranch") and w*k >= COMPETITION_SLOW_WK:
  iters = min(base_iters, 3)`).
- One trial per cell (no `--trials` in competition mode), a 900 s watchdog, and
  warmup fixed at `--warmup` (default 3). The worst shipped stock cell
  (k=2048, W=262144, ≈631 ms/iter per the ledger) costs
  1 first_call + 3 warmup + 3 iters ≈ 4.4 s device time — the "631 ms × trials"
  blow-up never happens because **trials is already 1 and iters is already capped**.
  Every measurement is real; nothing is extrapolated; the iters count rides in the
  cell record.

### 1.4 Renderer + ledger contract

- `load_competition()` reads `competition_table.csv` (:57–59); `load_psweep()` globs
  per-cell result JSONs by filename regex `comp_op_k(\d+)_w(\d+)_p(\d+)` (:62–73) —
  precedent for both CSV-fed and glob-fed regions.
- `us_fmt()` (:49–54): µs with `— ` em-dash for missing (never invents numbers),
  auto-switches to ms above 10 000 µs.
- `splice()` (:163–169) replaces `<!-- MARKER --> … <!-- /MARKER -->`; **`sys.exit`s
  if the marker is missing**, so a new region must only be spliced when its data dir
  is supplied.
- Ledger regions: `EXEC_NUMBERS` (78–83), `COMPETITION_TABLE` (105–135),
  `PSWEEP_TABLE` (232–235); each lives inside a `<details><summary>…</summary><div>`
  block, PSWEEP's closes at line 236 (`</div></details>`).

---

## 2. Design (a): `--model-scenarios` sweep mode

### 2.1 Scenario spec: data, not code

Scenarios live in a JSON grid file (default built-ins in the script, overridable via
`--scenarios-file`). Each scenario:

```json
{
  "version": 1,
  "scenarios": [
    {
      "name": "sampling_full_vocab",
      "model": "Llama3 70B decode sampling (single device, pad-to-pow2)",
      "callsite": "models/common/sampling/tt_sampling.py:847",
      "rows": 32, "n": 131072, "k": 32, "dtype": "bf16",
      "engines": ["stocknow", "routed", "op"],
      "today_engine": "stocknow",
      "valid_length": null, "num_slices": null,
      "calls_per_token": "1x/token/device",
      "notes": "vocab 128256 padded to 131072 (pow2); k=max_top_k=32"
    }
  ]
}
```

Field semantics:
- `name`: `re.fullmatch(r"[a-z0-9_]+", name)` enforced (it becomes a filename and
  glob key); must not begin with reserved prefixes `comp_`/`topk_`/`sort_`/`moe_`.
- `engines`: subset of the **engine registry** (below). Do NOT pre-filter engines
  by predicted supportability — the pipeline's philosophy (:31–34) is attempt and
  record the real error; an `UNSUPPORTED` cell IS the datum the ledger renders as —.
- `today_engine`: which engine represents what the model gets **pre-branch**
  (drives the ledger's "today" column). For `ttnn.topk` callers that is `stocknow`
  (largest=False = stock factory, same convention as the competition, :86–88).
  For models already calling `topk_large_indices` (minimax) it is `op`.
- `calls_per_token`, `model`, `callsite`, `notes`: pass-through metadata columns.

### 2.2 Engine registry (new constant, reuses existing child ops verbatim)

Insert after the BLAZE constants block (after line 318):

```python
# ---------------------------------------------------------------------------
# Model-scenario mode constants (--model-scenarios).
# Engines reuse the competition child ops verbatim; seed_index values are the
# SAME pinned indices as COMPETITION_LAYERS so a scenario that coincides with
# a competition cell reproduces the identical input tensor.
# engine -> (child op, composite, strict, seed_index)
# ---------------------------------------------------------------------------
SCENARIO_ENGINES = {
    "op":       ("topk_large_indices", False, True, 0),   # our op, direct
    "routed":   ("topk_routed",        True,  True, 1),   # ttnn.topk largest=True (this branch)
    "stocknow": ("topk_stock",         True,  True, 2),   # ttnn.topk largest=False = stock factory
}
SCENARIO_ENGINE_ORDER = ["op", "routed", "stocknow"]      # fixed run order, cheap engines first
SCENARIO_STOCK_NS_PER_ELEM = 137          # ledger-measured linear single-core rate; SIZING ONLY
SCENARIO_DEFAULT_ITERS = COMPETITION_DEFAULT_ITERS        # 5
SCENARIO_SLOW_ITERS = COMPETITION_SLOW_ITERS              # 3
MODEL_SCENARIOS = [ ...built-in defaults, §5... ]
```

Semantics note to encode in the constant's comment: the `routed` column is honestly
"**`ttnn.topk` largest=True on this branch**" — for shapes where routing does not
engage (e.g. pow2 W in [8192,65535) with k≤64, which stays on the stock multi-core
bitonic) it measures that bitonic, which is exactly what a model calling `ttnn.topk`
gets post-branch with zero code change. The per-cell `attrs`/`cores` in the result
JSON (notes field, :1139) disambiguate which factory actually ran.

**No `prebranch`/`opstock` engines in scenario mode** ⇒ no header edits, no
`--allow-header-edit`, no `finally` arm-restore needed — every engine runs on the
committed header. (`stocknow` on the committed header is the stock factory with
replay ON; if a pre-replay "today" is ever wanted, run the scenario dir a second
time with the existing competition `prebranch` machinery — out of scope here.)

### 2.3 Cell construction: `build_scenario_cells()` (new function)

Insert between `write_competition_reports()` (ends :1842) and `main()` (:1845):

```python
def load_scenario_specs(args):
    if args.scenarios_file:
        with open(args.scenarios_file) as f:
            specs = json.load(f)["scenarios"]
    else:
        specs = [dict(s) for s in MODEL_SCENARIOS]
    if args.scenarios:                       # comma-list subset, like --layers-competition
        wanted = set(args.scenarios.split(","))
        missing = wanted - {s["name"] for s in specs}
        if missing:
            sys.exit(f"unknown scenario(s) {sorted(missing)}")
        specs = [s for s in specs if s["name"] in wanted]
    for s in specs:
        if not re.fullmatch(r"[a-z0-9_]+", s["name"]):
            sys.exit(f"scenario name {s['name']!r} must be [a-z0-9_]+")
    return specs

def _scenario_iters(engine, rows, n, k, warmup, timeout):
    """Bounding tiers (extends the competition's stock-cell rule to rows>1).
    Tier A: default 5 iters / given warmup.
    Tier B: stock engine with rows*n*k >= COMPETITION_SLOW_WK -> 3 iters, warmup 1
            (the competition rule at :1587-1589, with rows folded into the work term).
    Tier C: predicted stock iter time (rows*n*SCENARIO_STOCK_NS_PER_ELEM) puts
            first_call+warmup+iters over timeout/2 -> iters=1, warmup=0
            (single-sample, still a REAL measurement; iters rides in the record).
    Tier D: even ONE predicted iter > timeout -> ("SKIPPED_SLOW", est_ms) -- the
            orchestrator writes the result JSON itself with the linear-model
            estimate in notes; the estimate is rendered as an estimate, never as
            a measurement."""
    if engine != "stocknow":
        return SCENARIO_DEFAULT_ITERS, warmup, None
    iters, wu = SCENARIO_DEFAULT_ITERS, warmup
    if rows * n * k >= COMPETITION_SLOW_WK:
        iters, wu = SCENARIO_SLOW_ITERS, 1
    est_iter_s = rows * n * SCENARIO_STOCK_NS_PER_ELEM * 1e-9
    if est_iter_s > timeout:
        return 0, 0, est_iter_s * 1e3          # tier D: skip, carry estimate (ms)
    if est_iter_s * (1 + wu + iters) > timeout / 2:
        iters, wu = 1, 0                        # tier C
    return iters, wu, None

def build_scenario_cells(specs, args):
    cells = []
    for s in specs:
        rows, n, k, dt = s.get("rows", 1), s["n"], s["k"], s.get("dtype", "bf16")
        for engine in SCENARIO_ENGINE_ORDER:
            if engine not in s["engines"]:
                continue
            child_op, composite, strict, seed_index = SCENARIO_ENGINES[engine]
            iters, wu, est_ms = _scenario_iters(engine, rows, n, k, args.warmup, args.timeout)
            num_slices = s.get("num_slices") if (engine == "op" and rows == 1) else None
            cid = f"scen_{s['name']}_{engine}"
            cells.append({
                "id": cid, "scenario": s["name"], "layer": engine,
                "op": child_op, "num_slices": num_slices,
                "batch": rows, "n": n, "k": k, "dtype": dt, "dim": -1,
                "anchor": "model_scenario",
                "valid_length": s.get("valid_length"),
                "apriori": "", "expected_factory": "",
                "composite": composite, "strict": strict,
                "seed": competition_seed(k, n, seed_index),
                "iters": iters, "warmup": wu, "est_ms": est_ms, "arm": None,
            })
    return cells
```

Notes:
- **Cell id** `scen_{name}_{engine}` → result file
  `results/scen_{name}_{engine}.{engine}.t0.json` (via `run_cell(cell,
  cell["layer"], 0, args)` and `result_path`, :1008). The `scen_` prefix keeps the
  namespace disjoint from `comp_*` (competition + psweep) and from classic-grid ids
  (`topk_b1xN…`), so `load_psweep`'s glob (`comp_op_k*_p*.json`, renderer :65) can
  never pick one up and vice versa.
- **Seeds**: `competition_seed(k, n, seed_index)` with the pinned per-engine indices —
  bit-reproducible reruns, and a scenario that happens to equal a competition cell
  sees the identical tensor.
- `dim`/`anchor`/`apriori`/`expected_factory` are populated so `run_cell`'s result
  dict and the child's `entry = dict(cell)` (:504) stay schema-compatible.

### 2.4 One-line child change: per-cell warmup override

`run_child` currently reads a single spec-level warmup (:496) and uses it at
:531–533. Add, next to the existing per-cell iters override at :510
(`cell_iters = cell.get("iters", iters)`):

```python
cell_warmup = cell.get("warmup", warmup)      # new, additive; absent key == old behavior
```

and change :532 `for _ in range(warmup):` → `for _ in range(cell_warmup):`.
Cells without a `"warmup"` key (ALL classic + competition cells) behave
byte-identically. This is the **only** touch inside existing measurement code.

### 2.5 Orchestrator: `run_scenarios()` (new function, mirrors `run_competition`)

```python
def run_scenarios(args):
    specs = load_scenario_specs(args)
    cells = build_scenario_cells(specs, args)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "scenario_grid.json"), "w") as f:
        json.dump({"specs": specs, "cells": cells}, f, indent=1)
    if args.report:
        write_scenario_reports(build_scenario_table(specs, cells, args.out), args.out)
        return 0

    def _done(cell):   # same resume semantics as competition (:1646-1654) + SKIPPED_SLOW
        path = result_path(args.out, cell["id"], cell["layer"], 0)
        if not (args.resume and os.path.exists(path)):
            return False
        with open(path) as f:
            return json.load(f)["status"] in ("MEASURED", "UNSUPPORTED", "WRONG", "SKIPPED_SLOW")

    for engine in SCENARIO_ENGINE_ORDER:            # engine-major, like competition
        for cell in [c for c in cells if c["layer"] == engine]:
            if _done(cell):
                print(f"SKIP (resume) {cell['id']}", flush=True); continue
            if cell["iters"] == 0:                  # tier D: never dispatch
                r = {"cell": cell, "arm": engine, "trial": 0, "status": "SKIPPED_SLOW",
                     "error": "", "ns_median": None, "cores": None, "max_abs_err": None,
                     "notes": f"est~{cell['est_ms']:.0f}ms/iter (linear {SCENARIO_STOCK_NS_PER_ELEM}ns/elem model, NOT measured) exceeds --timeout"}
                r.update(provenance_stamp())
                _write_result(args.out, f"{cell['id']}.{engine}.t0", r)
            else:
                r = run_cell(cell, cell["layer"], 0, args)
            print(f"{r['status']:<12} {cell['id']} iters={cell['iters']} "
                  f"{r['ns_median'] or ''} {r['error'][:80]}", flush=True)
            write_scenario_reports(build_scenario_table(specs, cells, args.out), args.out)
    write_scenario_reports(build_scenario_table(specs, cells, args.out), args.out)
    return 0
```

No `try/finally` header restore — no engine edits the header. Table rewritten after
every cell (early-stop keeps data, same as :1692).

### 2.6 `scenarios_table.csv` (new writer, mirrors `write_competition_reports`)

`build_scenario_table(specs, cells, outdir)` reads the per-cell result JSONs (same
`result_path` join as :1707–1711, keyed on `(scenario, engine)`), collects
provenance-drift exactly like :1713/:1735/:1762–1771, and emits one row per
scenario. `write_scenario_reports` writes `scenarios_table.csv` + `scenarios_table.md`
into `--out`. Fixed column order (engines always all three, blank when not in the
scenario's list — same convention as competition rows without a blaze cell):

```
scenario, model, callsite, rows, n, k, dtype, today_engine, calls_per_token,
op_us, op_cores, op_iters, op_status,
routed_us, routed_cores, routed_iters, routed_status,
stocknow_us, stocknow_cores, stocknow_iters, stocknow_status,
today_us, speedup_today_over_routed, speedup_today_over_op, notes, provenance
```

- `{e}_us` filled ONLY from `status == MEASURED` (`ns_median/1000`, the competition
  rule at :1736–1738); anything else leaves it blank and puts the verbatim
  status+error in `{e}_status` — WRONG/FAILED/UNSUPPORTED timings never enter
  numeric columns.
- `{e}_status` for a SKIPPED_SLOW stock cell carries `SKIPPED_SLOW(est~Xms)` — the
  renderer shows `≈ X ms †` with a "linear-model estimate, not measured" footnote,
  visibly distinct from measured cells.
- `{e}_iters` makes single-sample (tier C) cells self-describing.
- `today_us` = the `today_engine` column's value (copied so the renderer never
  recomputes); speedups are ratio-of-medians like the competition (:1751–1758),
  blank unless both sides are MEASURED.

### 2.7 New CLI flags (in `main()`, after `--with-blaze` ends at :1896)

```python
p.add_argument("--model-scenarios", action="store_true",
               help="named LLM model-scenario sweep (engines: op/routed/stocknow on the "
                    "committed header; no header edits) -> scenarios_table.csv")
p.add_argument("--scenarios-file", default=None,
               help="JSON scenario grid overriding the built-in MODEL_SCENARIOS")
p.add_argument("--scenarios", default=None,
               help="comma-list of scenario names to (re)run (subset of the grid)")
```

Dispatch in `main()` immediately after the competition dispatch (:1909–1910):

```python
if args.competition:
    return run_competition(args)
if args.model_scenarios:          # NEW
    return run_scenarios(args)
```

`--ks/--ns/--dtypes/--ops/--layers-competition/--op-num-slices` are **ignored** in
scenario mode (shapes and P come from the scenario file); guard with a loud
`sys.exit` if `--model-scenarios` is combined with `--competition`. `--iters` maps
to overriding `SCENARIO_DEFAULT_ITERS` before tiering (same spirit as :1549).
`--resume`, `--report`, `--timeout`, `--warmup`, `--out` work unchanged.

### 2.8 Runbook

```bash
# full scenario run (fresh out dir; ~15 cells x ~40-90 s subprocess each)
python tests/.../reduction/_canonical_topk_sweep.py --model-scenarios \
    --out generated/canonical_sweep/scen1

# one scenario, one engine class rerun
python ... --model-scenarios --scenarios sampling_full_vocab --resume --out .../scen1

# render (competition + psweep dirs unchanged from the current ledger build)
python tests/.../reduction/_topk_ledger_render.py \
    --competition-dir generated/canonical_sweep/comp3 \
    --psweep-dir generated/canonical_sweep/psweep4 \
    --scenarios-dir generated/canonical_sweep/scen1
```

---

## 3. Design (b): `MODEL_SCENARIOS` ledger region + renderer

### 3.1 One-time ledger edit

Insert after line 236 of `TOPK_LEDGER.html` (the `</div></details>` closing the
PSWEEP details block), matching the existing details-block pattern:

```html
  <details open><summary>Model scenarios — real LLM shapes through the canonical pipeline</summary><div>
  <p class="sub">Named callsite shapes measured end-to-end (Tracy device-kernel, composite-summed,
  correctness-gated). today = the engine the model hits pre-branch; routed = <code>ttnn.topk</code>
  on this branch, zero model-code change; op = <code>topk_large_indices</code> called directly.
  † = linear-model estimate (cell too slow to measure inside the watchdog), not a measurement.</p>
<!-- MODEL_SCENARIOS -->
<!-- /MODEL_SCENARIOS -->
  </div></details>
```

(Prose outside the markers is hand-written per the ledger's convention, :4–8 of the
renderer docstring; the markers bound only the table.)

### 3.2 Renderer changes (`_topk_ledger_render.py`)

1. New loader (next to `load_competition`, after :59):

```python
def load_scenarios(scen_dir):
    path = os.path.join(scen_dir, "scenarios_table.csv")
    return list(csv.DictReader(open(path)))
```

2. New render function (after `render_psweep_table`, :122). Columns per the task:
   scenario | model + callsite | shape | today (engine+µs) | routed | op | speedup |
   calls/token. Reuses `us_fmt` (em-dash rule) and `fnum`:

```python
def render_model_scenarios(rows):
    body = []
    for r in rows:
        shape = f"{int(r['rows'])}×{int(r['n']):,}, k={int(r['k'])}"
        today_e = r.get("today_engine", "")
        today = fnum(r.get("today_us"))
        ro, op = fnum(r.get("routed_us")), fnum(r.get("op_us"))
        # SKIPPED_SLOW estimate: render as "≈ X ms †", styled flat, never as a win
        est = re.search(r"SKIPPED_SLOW\(est~(\d+)ms\)", r.get(f"{today_e}_status", "") or "")
        today_td = (us_fmt(today) if today is not None
                    else (f'<td class="n flat">≈ {int(est.group(1)):,} ms †</td>' if est
                          else '<td class="n flat">—</td>'))
        best = min((v for v in (ro, op) if v is not None), default=None)
        sp = (f'<td class="n win">{today / best:,.0f}×</td>' if today and best
              else '<td class="n flat">—</td>')
        callsite = r.get("callsite", "")
        body.append(
            f'      <tr><td>{r["scenario"]}</td>'
            f'<td>{r.get("model","")}<br><code class="sub">{callsite}</code></td>'
            f'<td class="n">{shape}</td>'
            f'<td class="n">{today_e}</td>{today_td}'
            f'{us_fmt(ro, "win")}{us_fmt(op, "win")}{sp}'
            f'<td>{r.get("calls_per_token","")}</td></tr>')
    head = ('<thead><tr><th>scenario</th><th>model + callsite</th><th class="n">shape (rows×N, k)</th>'
            '<th>today: engine</th><th class="n">today µs</th>'
            '<th class="n ours">ttnn.topk (this branch)</th><th class="n ours">op direct</th>'
            '<th class="n">speedup</th><th>calls/token</th></tr></thead>')
    return (f'  <div class="tablewrap"><table>\n    {head}\n    <tbody>\n'
            + "\n".join(body) + "\n    </tbody>\n  </table></div>")
```

3. `main()` (:172–187): add an **optional** flag and a conditional splice:

```python
ap.add_argument("--scenarios-dir", default=None)          # after :176
...
if args.scenarios_dir:                                    # after the PSWEEP splice, :185
    t = splice(t, "MODEL_SCENARIOS", render_model_scenarios(load_scenarios(args.scenarios_dir)))
```

Because the splice is conditional on the new optional flag, a render invoked the
old way (`--competition-dir comp3 --psweep-dir psweep4`) produces **byte-identical
output for the three existing regions** and leaves the new region untouched (the
marker block is inert HTML comments when empty). Conversely `splice`'s marker check
means running with `--scenarios-dir` before the one-time HTML edit fails loudly —
correct order: HTML edit first, then render.

---

## 4. Default scenario table (built-in `MODEL_SCENARIOS`)

Shapes below are from the named callsites; rows=32 is the representative decode
batch — the point of the JSON grid is that these are **data**, editable without
touching code. Engine lists are deliberately permissive (unsupported ⇒ recorded
error ⇒ em-dash), except where the constraint is structural (k<16 can never reach
the op).

| name | callsite (evidence) | rows | n | k | engines | today | note |
|---|---|---|---|---|---|---|---|
| `sampling_full_vocab` | `models/common/sampling/tt_sampling.py:847` (pad-to-pow2 path :839–846) | 32 | 131072 | 32 | op,routed,stocknow | stocknow | vocab 128256→131072; routed small-k arm engages (W≥65535) |
| `sampling_split_half` | `tt_sampling.py:807` (multi_step_reduction split, :801–806) | 32 | 64128 | 32 | op,routed,stocknow | stocknow | 128256/2, non-pow2 → stock falls to linear single-core; routed small-k arm engages (non-pow2, ≥4096); 2 calls/token |
| `sampling_shard_pow2` | `models/common/modules/sampling/sampling_1d.py:568` (shard pad, :556–566) | 32 | 16384 | 32 | op,routed,stocknow | stocknow | 128256/8=16032→16384; stock IS the multi-core bitonic here (pow2, 8192≤W<65535, k≤64) — the honest "already fast" row |
| `minimax_msa_blocks` | `models/demos/minimax_m3/tt/attention/msa.py:147` (`topk_large_indices(block_scores, k=topk_blocks)`) | 8 | 1024 | 16 | op | op | model already ships our op; k=16 from `tests/unit/test_msa_sp_chunked_vs_ref.py:77`; rows/blocks TBD from prod config — edit in file |
| `gptoss_router` | `models/demos/gpt_oss/tt/topk.py:26` (k=experts_per_token) | 32 | 128 | 4 | stocknow,routed | stocknow | k=4: op (k<16) and small-k routing (W<4096) structurally out; row shows stock is the only engine |
| `qwen3_moe_fallback` | `models/common/modules/moe/tt_moe_gate.py:639` (topk over all experts) | 32 | 128 | 8 | stocknow,routed | stocknow | ungrouped fallback gate |
| `grok_moe_gate` | `models/experimental/grok/tt/grok_moe.py:119` (`ttnn.topk(gate_probs, 32)`) | 32 | 64 | 32 | stocknow,routed | stocknow | tiny-N sanity row |

The DeepSeek `generalized_moe_gate` scenario is **v2**: its child op exists
(:681–743) but its correctness is tie-tolerant `index_match_frac`, not the strict
gate every other scenario cell has — including it would need either a strict-gating
extension of that correctness function or an explicit "not strict-gated" caveat
column. Recommend deferring rather than diluting the gate. The two
`LARGE_INDICES_ANCHORS` (:228–232, rows=640 prefill / rows=2 bounded-cache) can be
added as op-only scenarios verbatim if wanted — `valid_length` already plumbs
through (§2.3).

---

## 5. What NOT to touch (byte-identical guarantees)

- `COMPETITION_LAYERS` (:286–292), `build_competition_cells` (:1546),
  `run_competition` (:1628), `build_competition_table` (:1702),
  `write_competition_reports` (:1775), the competition CSV schema, and cell-id
  formats — **unchanged**. Scenario mode is a parallel path dispatched before the
  classic grid, exactly like `--competition`.
- `run_child` / `_build_cell_callable`: only the additive
  `cell.get("warmup", warmup)` default (§2.4). No new child op branches, no
  dispatch changes; the HARNESS_BUG fall-through (:756–759) stays the backstop.
- `run_cell`, `parse_tracy_for_cell`, `parse_tracy_composite`, `provenance_stamp`,
  `result_path`, `_write_result`: reused, untouched.
- Renderer: `load_competition`/`load_psweep`/`render_*`/`render_exec_numbers`/
  `splice` untouched; `--competition-dir`/`--psweep-dir` stay required; the new
  `--scenarios-dir` is optional and its splice conditional ⇒ existing
  comp3/psweep4 renders are byte-identical.
- `TOPK_LEDGER.html`: one insertion after line 236; the three existing marker
  regions and everything inside them untouched.
- Header-arm machinery (`checkout_arm`, `ARM_DEFINES`, markers): never invoked in
  scenario mode — scenario runs are safe on a tree where `--allow-header-edit`
  would be refused.

## 6. Verification checklist for the implementer

1. `python _canonical_topk_sweep.py --competition --report --out <existing comp3>`
   before and after the patch: `competition_table.csv` byte-identical.
2. `python _topk_ledger_render.py --competition-dir comp3 --psweep-dir psweep4`
   before/after: `TOPK_LEDGER.html` diff is exactly the new (empty) details block.
3. `--model-scenarios --scenarios gptoss_router` on device: expect
   `stocknow=MEASURED`, `routed=MEASURED` (same stock factory, ratio ≈1), op absent.
4. `--model-scenarios --scenarios sampling_full_vocab`: routed engages the small-k
   arm (check `cores` > 1 in the result JSON notes), stocknow tier-B iters=3
   (32·131072·32 = 2^27 ≥ 2^24).
5. Resume: kill mid-run, rerun with `--resume`, confirm MEASURED/SKIPPED_SLOW cells
   skip and FAILED retries.
