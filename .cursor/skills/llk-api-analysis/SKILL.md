---
name: llk-api-analysis
description: >-
  Decompose the LLK (low-level kernel) APIs a tt-metal / TTNN model actually
  uses, with their template + runtime configuration, data formats, tile dims,
  math fidelity and dst-accum, then aggregate per templated API and per base
  API. Optionally identify which base APIs are missing on Quasar (the QSR arch
  gap). Use when asked to run the LLK API analysis / decomposition for a model,
  profile which compute LLK APIs a model compiles, or produce an
  llk_api_analyzer report for a given model run command.
---

# LLK API Analysis

Runs any tt-metal / TTNN model under `tt_metal.tools.llk_api_analyzer`, collapses
the result into per-API and per-base-API tables, summarizes it, and (optionally)
flags the Quasar gap. Works for any model — you supply the run command.

**Prerequisite — the analyzer tool.** `tt_metal/tools/llk_api_analyzer/` comes from PR
[#48671](https://github.com/tenstorrent/tt-metal/pull/48671) (branch `halgh/llk-api-analyzer`)
and is not yet on `main`. If `python -c "import tt_metal.tools.llk_api_analyzer"` fails,
merge/copy it in first — see [reference.md](reference.md).

## Inputs to collect

1. **Model run command** — the pytest/python command that executes the model
   (e.g. `pytest models/demos/yolov8s/tests/pcc/test_yolov8s.py::test_yolov8s_640 -q`).
2. **Output prefix** — where to write CSVs (default: `<model>_llk_report`).
3. **tt-metal root** — the checkout to run in. If the model lives only on a
   feature branch with C++ changes, set up an isolated worktree + build instead
   of switching the user's branch → see [reference.md](reference.md).

If any is unclear, ask before running. For everything else, pick sensible
defaults and proceed.

### Deriving the run command / device / batch from a PR

When the ask is "analyze model X from PR #N (find the run command / smallest batch
for device D)":

1. Read the PR (`gh pr view N`, the diff, and the CI manifest it edits — e.g.
   `tests/pipeline_reorg/models_e2e_tests.yaml`). The real command usually points at a
   demo/test file with a `-k` filter and required env (`MESH_DEVICE`, `HF_MODEL`,
   `TT_CACHE_PATH`, `SAMPLING_MODE`).
2. Open that demo/test file and read its parametrization: the `test_config`/batch
   `pytest.param` ids (the batch sizes) and the `MESH_DEVICE → mesh_shape` map.
   `MESH_DEVICE` is a **logical mesh label, not the physical board** — e.g.
   `N150=(1,1)`, `N300=(1,2)`, `T3K=(1,8)`. A single-chip board (N150 *or* a
   single Blackhole p150) can only host the `(1,1)` mesh; `(1,2)`/`(1,8)` need
   2 / 8 devices.
3. **Smallest batch that runs on device D** = the smallest batch `pytest.param`
   whose mesh fits D's device count, *and* which is not blocked by an arch guard
   (next section). Note the model may name-map its mesh (`get_device_name`) purely
   by device count, independent of the real silicon.

### Before spending device time, verify the model runs on this host

- **Arch-support guard.** Many suites hard-skip on an unsupported arch. TTTv2's shared
  `models/common/tests/conftest.py` does `if ttnn.device.is_blackhole(): pytest.skip(...)`,
  so *every* case (batch-1 included) skips on a Blackhole box → no kernels compile →
  empty CSV. Grep the model's `conftest.py`/test for `is_blackhole`/`is_wormhole`/
  `skip` before running; if the host arch is gated off, run on a supported machine
  (or temporarily bypass the guard only if the intent is to see what *would* compile).
- **Weights availability.** See the gated-model note in [reference.md](reference.md):
  a gated HF id (401 `GatedRepoError`) with no token/cache can be swapped for an
  ungated mirror with an identical config — LLK output depends on shapes/dtypes, not
  weight values.

## Ask about the Quasar gap

Unless the user already said whether they want it, **ask before the run** (so the
device time is spent once):

> Should I also identify the Quasar (QSR) gap — i.e. which of the base LLK APIs
> this model uses are missing on the Quasar architecture?

Use the AskQuestion tool with options: "Yes — include Quasar gap analysis" /
"No — LLK decomposition + aggregation only". Only run step 5 if yes.

## Workflow

```
- [ ] 0. Environment ready (+ worktree/build if the model is branch-only)
- [ ] 1. Run model under llk_api_analyzer -> per-call CSV
- [ ] 2. Sanity check the CSV
- [ ] 3. Aggregate -> by_api -> by_base_api
- [ ] 4. Summarize (format / fidelity / tile distributions)
- [ ] 5. (optional) Quasar gap
- [ ] 6. Write the report
```

### 0. Environment

```bash
cd <tt_metal_root>
source python_env/bin/activate
export PYTHONPATH=<tt_metal_root> OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
python -c "import elftools, tabulate" 2>/dev/null || uv pip install pyelftools tabulate
```

Branch-only model, device/grid choice, and known gotchas (CPU-reference timeout,
PCC-assert-is-fine, weights): see [reference.md](reference.md).

### 1. Run under the analyzer

```bash
python -m tt_metal.tools.llk_api_analyzer \
  --run '<MODEL_RUN_COMMAND>' \
  -f csv -o <OUT>_llk_report.csv
```

Redirect the run's stdout to a log (`... > <OUT>_analyzer.log 2>&1`) so you can
recover the JIT stats line later. Expect a few minutes; the run recompiles all
kernels with debug info in an isolated cache.

### 2. Sanity check

```bash
wc -l <OUT>_llk_report.csv      # only the header line == kernels never compiled
```

If empty, the model likely never reached the TTNN forward (see the CPU-reference
timeout gotcha in [reference.md](reference.md)). A `FAILED` PCC assert is fine —
kernels compile before the assert.

Grab the JIT stats (kernel lookups / build-once dedup) from the log:

```bash
grep "JIT cache stats" <OUT>_analyzer.log | tail -1
```

### 3. Aggregate

```bash
python3 <SKILL_DIR>/scripts/aggregate_llk.py      <OUT>_llk_report.csv        <OUT>_llk_report_by_api.csv
python3 <SKILL_DIR>/scripts/aggregate_llk_base.py <OUT>_llk_report_by_api.csv <OUT>_llk_report_by_base_api.csv
```

- `aggregate_llk.py` collapses per-call rows → one row per full **templated** API,
  and prints `configs` and `distinct LLK APIs`.
- `aggregate_llk_base.py` collapses further → one row per **base** API (template
  params moved into `Op Args`), and prints `distinct base APIs`.

### 4. Summarize

```bash
python3 <SKILL_DIR>/scripts/summarize_llk.py <OUT>_llk_report.csv
```

Prints per-call config count and the data-format / tile-dim / math-fidelity /
fp32-dest-accum distributions.

### 5. Quasar gap (only if requested)

```bash
python3 <SKILL_DIR>/scripts/quasar_gap.py \
  <OUT>_llk_report_by_base_api.csv <tt_metal_root> <OUT>_llk_report_quasar_gap.csv
```

Flags base APIs present on blackhole/wormhole_b0 but absent on quasar as candidate
GAPs. **This is a name-substring first pass — confirm each flagged GAP by reading
the actual Quasar headers** under `tt_metal/hw/ckernels/quasar/` and
`tt_metal/tt-llk/tt_llk_quasar/`, because an op may exist under a different name
or be routed through a datacopy/unpack-pack gasket. See [reference.md](reference.md)
for the method and caveats.

### 6. Report

Write `<OUT>_HANDOFF.md` next to the CSVs using the template below.

## Report template

```markdown
# LLK API Analysis — <MODEL> (<date>)

- Run command: `<MODEL_RUN_COMMAND>`
- tt-metal root: `<path>` (branch `<branch>`)
- Compiled: <N> kernel lookups, <M> build-once dedup.
- <configs> per-call configs -> <api> templated APIs -> <base> base APIs.

## Data formats (in+out CB refs)
| Format | Count |
|--------|-------|
| ... | ... |

## Notable
- <dominant families, math fidelity mix, SFPU ops, matmul/reduce/tilize usage>

## Quasar gap (if run)
- <K> candidate gaps of <base> base APIs. Confirmed missing on QSR:
  - <family / API> — <one-line confirmation from the headers>

## Output files
| File | Level |
|------|-------|
| `<OUT>_llk_report.csv` | per LLK call |
| `<OUT>_llk_report_by_api.csv` | per templated API |
| `<OUT>_llk_report_by_base_api.csv` | per base API |
| `<OUT>_llk_report_quasar_gap.csv` | per base API, arch presence (if run) |
```

## Scripts

All in `<SKILL_DIR>/scripts/` (execute them, don't inline the code):

| Script | Purpose |
|--------|---------|
| `aggregate_llk.py` | per-call CSV → per templated-API CSV |
| `aggregate_llk_base.py` | per-API CSV → per base-API CSV |
| `summarize_llk.py` | format / tile / fidelity / dst-accum distributions |
| `quasar_gap.py` | candidate Quasar gaps vs blackhole/wormhole_b0 |
