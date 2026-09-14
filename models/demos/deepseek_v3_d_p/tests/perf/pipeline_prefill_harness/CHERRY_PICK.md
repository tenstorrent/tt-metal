# PP=4 prefill perf campaign — the whole kit, in one commit

This directory is **working material, not proposed for merge** (the original wording, `d0972011a6f`). It lives on its own branch so it can be cherry-picked onto whatever branch you need to measure, without putting ~1,000 lines of site-specific shell into a product PR.

```bash
git cherry-pick <this commit>     # everything below lands at once; keep it off your PR branch
models/demos/deepseek_v3_d_p/tests/perf/pipeline_prefill_harness/run_campaign.sh --check
```

`test_mistral4_profile_single_layer.py` (one level up, in `tests/perf/`) is included in the same commit, because the `_deep` Tracy captures cannot run without it and it is absent from most branches. If the branch you cherry-pick onto already has it, the content is identical to `18cb77fdc1b` and git will not conflict.

## What is here

| | |
|---|---|
| `run_campaign.sh` | The whole campaign: two Tracy captures, the 16-cell matrix, a warm-latency re-run, the summary. `--check` preflights without touching chips; `--summary` re-prints from an existing run. |
| `run_matrix.sh`, `run_pp4_model.sh`, `run_single_layer_profile.sh` | The drivers it calls. |
| `env.sh` | Every path (checkpoint, TTNN caches, golden traces, outputs) — all overridable, none in the repo. **Check this first on a new box.** |
| `preflight.sh`, `check_board.sh` | Environment and fabric-health gates. |
| `wrappers/` | Narrower drivers from the 2026-09-14 campaign — see below. |
| `bindings_reference/` | Generated rank bindings for `bh-glx-120-b03u02`. **Reference only.** Deliberately NOT in `topology_configuration/`, so `pick_binding()` cannot pick them up on a galaxy they do not describe. |

### `wrappers/`

* **`make_tables.sh`** — re-prints every table from existing logs. Touches **no devices**; safe to run while someone else has the galaxy. Start here.
* `single_layer_rerun.sh` — the two `_deep` captures at `DEEP_CHUNKS=8`, gated on `check_board.sh`, into a separate output dir so earlier captures survive for comparison.
* `warm_ttft.sh` — the warm-latency phase only, pair-major by ISL so a truncated run still yields complete speedup pairs.
* `retest.sh` — runs a cell twice (re-warm, then measure) after a board reset.

The analyzers (`analyze_prefill_*.py`, `summarize_prefill_campaign.py`, `gen_pipeline_binding.py`, `probe_mesh_columns.py`) are **not** here — they are ordinary committed code one level up in `tests/perf/`, and duplicating them would only invite drift.

## Four traps that produce wrong numbers without erroring

1. **The `[8,1]` column→device map is per-galaxy**, and a wrong one does not fail — it builds stages that are not columns and reports plausible, wrong numbers. Regenerate on every new machine with `gen_pipeline_binding.py`, galaxy idle.
2. **`gen_pipeline_binding.py --profile` needs an explicit `--template`.** Without it, `--profile` writes to the plain binding's filename, and every later "e2e" cell runs instrumented and reports inflated times with no indication why.
3. **`run_pp4_model.sh` exits with the PRODUCER's rc, not the runner's.** A runner-side crash reports `rc=0`, the matrix records the cell complete, and a re-run skips it because `runner.log` exists — a failed cell looks passing, twice. Always `grep -cE "AssertionError|Traceback" */runner.log`.
4. **After any hard kill, `tt-smi -glx_reset` before measuring again.** A degraded galaxy is silently 2–5x slow rather than erroring; the first hard failure comes much later. Budget two resets — one can report success and still leave the fabric unable to map an 8x4 mesh. Measured 2026-09-14: 1rank@25,600 read 7.970 s degraded vs 0.820 s after reset.

Also: never launch a device run in a 2-minute-capped foreground, and phase 3 of the campaign is not optional — `run_matrix.sh` runs each latency cell once, cold, and cold is not slow-but-valid, it is wrong.

## Results this produced

`~/debug-docs/mistral4_prefill_planning-noissue/perf/RESULTS_PP4_RERUN_2026-09-14.md`, with the method and the full trap list in `HANDOFF_REGENERATE_PP4_TABLES.md` alongside it. Upstream copy of this harness: `tt-metal-3` branch `kmabee/mistral4-routing-capture`.
