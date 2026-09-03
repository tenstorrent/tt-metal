# GLM-4.7-Flash datatype-sweep work log

## DS-001: starting point and plan

Starting from the optimized-full-model stage (commit `b98e7ff6a1b`, plus the
checkpoint-SHA-recording commit `0a89462de76`). Read `$datatype-sweep`,
`$tt-device-usage`, `$stage-review`, `$autofix` SKILL.md files per the goal
contract. Read `doc/optimized_decoder/README.md` and `work_log.md` in full:
the per-tensor-group dtype/fidelity policy was already exhaustively swept at
the decoder-layer level with real-weight 202k-long-context PCC evidence, a
materially stronger evidence base than typical decoder-layer PCC (which
`$datatype-sweep` treats as ordering-only, not a pass/fail source of truth).
`doc/optimized_full_model/README.md` explicitly deferred the bf4-LM-head
candidate ("`head_probe.json`'s bf4-LM-head candidate at 624 us vs bf8's 878
us is a real, larger, correctly-deferred candidate that belongs to that stage
[this one], not this one") — that's this stage's headline candidate.

Dispatched a research subagent (Explore-equivalent) to map every dtype/
fidelity override hook in `tt/model.py`/`tt/optimized_decoder.py`/
`tt/functional_decoder.py`/`tt/fused_decoder.py`/`tt/generator.py` before
writing any sweep code. Findings: every tensor group except LM-head fidelity
and router/gate decode fidelity already had a real override hook (constructor
kwarg or `OptimizedDecoder`/`SharedRopeDecoder` class attribute, the same
mechanism `tests/dev_optimize.py` already uses at the decoder-layer level).
No existing full-model-level sweep driver or precision-config JSON loader
existed anywhere in the codebase.

## DS-002: plumbing

Added the two missing override hooks, both defaulting to the exact
pre-existing hardcoded value (no behavior change):

- `tt/model.py`: `GLM47FlashModel.from_pretrained(..., lm_head_fidelity: str
  = "hifi2", ...)`; `self.ck_lm_head = _ck(dev, *FIDELITY[lm_head_fidelity])`
  replaces the hardcoded `_ck(dev, ttnn.MathFidelity.HiFi2, False)`. Also
  stored `expert_dtype`/`weight_dtype`/`embed_dtype`/`lm_head_dtype`/
  `lm_head_fidelity`/`decoder_cls_name` on the model instance for the
  propagation check.
- `tt/optimized_decoder.py`: `OptimizedDecoder.router_fidelity = "hifi4_fp32"`
  class attribute; `from_state_dict` builds `self.ck_router = _ck(dev,
  *FIDELITY[cls.router_fidelity])`; `_router_scores_decode` now reads
  `self.ck_router` instead of `self.ck_hifi4`. Verified by grep that
  `self.ck_hifi4` has exactly one call site inside `optimized_decoder.py`
  (the one just changed) — every other `ck_hifi4` use lives in
  `fused_decoder.py`'s own prefill-attention/shared-MLP methods, which
  `OptimizedDecoder` overrides with its own `ck_prefill_proj`/`ck_mlp`-based
  implementations and therefore doesn't execute on the measured path, so
  leaving `ck_hifi4` itself unchanged (still real HiFi4+fp32acc) is safe.

Verified immediately, before writing any sweep code:

```
python models/autoports/zai_org_glm_4_7_flash/tests/dev_full_model.py smoke   # SMOKE_OK
python -m pytest models/autoports/zai_org_glm_4_7_flash/tests/test_full_model.py -q -s -p no:randomly
  # 47 passed, 250.58s -- includes test_deployment_dtype_policy_preserved
python -m pytest models/autoports/zai_org_glm_4_7_flash/tests/test_optimized_decoder.py -q -s -p no:randomly
  # 26 passed, 57.71s -- the decoder-level suite for the file router_fidelity was added to
```

## DS-003: sweep driver

Checked `run_prefill_check.py`/`run_teacher_forcing.py` and found both
already expose a programmatic entry point
(`run_prefill_check(model_dir, reference_path, mesh_device, build_kwargs)`)
that forwards `build_kwargs` straight into `build_generator(...)` ->
`GLM47FlashModel.from_pretrained(...)`. Wrote
`tests/dev_datatype_sweep.py`: builds the model once per candidate through
this exact path (real kwargs + a dynamic `SharedRopeDecoder` subclass for the
class-attribute groups), introspects a `policy_snapshot` from the constructed
model's real tensors/compute-kernel-configs, then reuses
`run_prefill_check._run_one_entry_prefill` and
`run_teacher_forcing._run_one_entry` directly (not reimplemented) against the
already-open device for both checks, saving a rebuild between them and
letting the driver capture the snapshot from a live object instead of a torn-
down one.

First sanity run (`C00_baseline`, zero overrides): prefill 0.880/1.000/1.000,
teacher-forcing 0.850/1.000/1.000, TTFT 590.28 ms, decode 44.02 t/s/u — matches
`doc/optimized_full_model/logs/run_{prefill_check,teacher_forcing}.log`
within noise. `policy_snapshot` shows every expected dtype/fidelity value.
Build time 172.4 s (JIT cache 100% hits).

## DS-004: the sweep itself

Ran candidates `C01`–`C09` in sequence (one hardware job at a time, per
`$tt-device-usage`), each `tests/dev_datatype_sweep.py` build_kwargs override
combination against the identical AIME24 reference. Full numbers and per-
candidate reasoning are in `README.md`'s "What was tested" section and
`sweep_results.json`; summary of the decision for each:

| id | result |
|---|---|
| C01/C02 (LM head bf4, LoFi/HiFi2) | identical (top1 0.790, top5 0.990, decode 44.62) — rejected, precedent-matched to FM-021 |
| C03 (LM head bf8+LoFi) | top1 0.830, decode 44.08 (noise) — rejected, no benefit |
| C04 (KV cache bf16) | top1 0.850 (tied), TTFT 620.48, decode 43.91 — rejected, slower |
| C05 (router HiFi2) | top1 0.820, decode 44.09 (noise) — rejected, no benefit |
| C06 (attn HiFi2) | top1 0.850 (tied), decode 41.87 — rejected, much slower |
| C07 (expert HiFi2) | top1 0.850 (tied), decode 43.82 — rejected, slower |
| C08 (mlp HiFi2) | top1 0.820, decode 43.48 — rejected, slower + accuracy |
| C09 (dense MLP bf4) | top1 0.870 (noisy), decode 44.03 (tied) — rejected, no speed benefit |
| C10 (all-bf8) | not run — hard DRAM limit (`doc/probe/README.md`) |

No candidate beat `C00_baseline` on the "fastest config that passes the bar"
criterion; several pass the stated top-5/top-100 gate while regressing top-1
for a smaller speed gain than this exact team already rejected once
(FM-021), or regress speed with no accuracy benefit. Kept baseline.

## DS-005: post-selection artifacts

Since the winning config is bit-identical to the already-shipped defaults,
closing the remaining goal requirements meant re-running the existing
harnesses through the *unmodified* default construction path rather than
building anything new:

- `tests/test_full_model_perf.py` (normal, zero overrides) — refreshed
  `perf.json`/`capacity.json`, copied to
  `postselection_perf.json`/`postselection_capacity.json`. Confirms the
  no-per-token-readback "warmed token-out" figure
  (`decode_ms_per_token.traced_model_plus_sampling` = 22.879 ms/tok) this
  stage records as the post-selection number.
- `tests/run_qualitative_suite.py --max-new-tokens 128 --skip-hf` — fresh
  run, copied to `qualitative/`. Greedy prefix agreement (8/128, 16/128,
  45/128, 14/128, 32/128, 15/128) is bit-identical to
  `doc/optimized_full_model/qualitative/`'s numbers — the strongest available
  evidence that DS-002's plumbing changed nothing about generation.
- `tests/test_prefill_padding.py` (13 passed) — fresh non-aligned-prompt
  evidence post-plumbing-edit, even though the selected config triggers none
  of `$datatype-sweep`'s "must rerun" conditions (no KV-cache/layout/chunking
  change).

Every one of `test_full_model_perf.py`'s and `run_qualitative_suite.py`'s
default output paths is hard-coded under `doc/full_model/` (the same
clobber-avoidance issue `doc/optimized_full_model/README.md`'s "Known
limitations" section documents for the previous stage). Copied each fresh
result into `doc/datatype_sweep/` immediately, then `git checkout`ed
`doc/full_model/` before the next step; verified `git status --porcelain --
doc/full_model` was empty before moving on and again before this stage's
commit.

## DS-006: reporting

`selected_precision_config.json`: every weight group, the two new plumbing
fields, layer exceptions (the model's only one, architectural), KV-cache/
logits/CCL fields, and a `construction` block recording the exact
`from_pretrained` kwargs / decoder class attrs that reproduce this policy as
the model default. `tests/check_selected_precision_config.py` proves (a) the
live source defaults match this file and (b) the introspected
`C00_baseline.json` policy snapshot matches this file — passes:

```
$ python -m models.autoports.zai_org_glm_4_7_flash.tests.check_selected_precision_config
PRECISION_CONFIG_PROPAGATION_CHECK: OK
```

`tests/build_datatype_sweep_report.py` assembles `sweep_results.{json,csv}`
and the two Pareto PNGs from `runs/*.json`. First render had two layout
defects caught by looking at the rendered PNG (not just generating it):
identical-coordinate labels (`C01`/`C02`) overlapped illegibly, and the
legend collided with the not-run-candidate caption. Fixed with exact-
duplicate-coordinate label grouping and a pixel-space greedy label-stacking
pass (minimum on-screen gap enforced via `ax.transData`, not a data-unit
heuristic, so it holds regardless of each chart's y-range) plus moving the
legend and caption to non-overlapping positions. Re-rendered and re-inspected
before treating the charts as done, per the dataviz skill's "render it and
look at it" step.

`doc/context_contract.json` gained a `datatype_sweep` section: no capability
reduction, KV-cache dtype unchanged (bf8 selected; bf16 tested and would also
still fit, ~28.5 GiB projected resident vs 31.5 GiB allocatable, rejected on
speed not capacity), non-aligned-prompt check rerun fresh post-edit.
`check_context_contract.py --stage full-model --require-contract` still
passes (`supported=202752`).

## Commands

```bash
# plumbing sanity
python models/autoports/zai_org_glm_4_7_flash/tests/dev_full_model.py smoke
python -m pytest models/autoports/zai_org_glm_4_7_flash/tests/test_full_model.py -q -s -p no:randomly

# per-candidate sweep (repeated for C00-C09 with the flags in sweep_results.csv's `command` column)
python -m models.autoports.zai_org_glm_4_7_flash.tests.dev_datatype_sweep \
    --config-id <id> [--lm-head-dtype ...] [--lm-head-fidelity ...] [--cache-dtype ...] \
    [--router-fidelity ...] [--attn-fidelity ...] [--expert-fidelity ...] [--mlp-fidelity ...] \
    [--dense-mlp-dtype ...] \
    --out doc/datatype_sweep/runs/<id>.json

# report + plots
python -m models.autoports.zai_org_glm_4_7_flash.tests.build_datatype_sweep_report

# propagation check (no device)
python -m models.autoports.zai_org_glm_4_7_flash.tests.check_selected_precision_config

# post-selection artifacts (normal, zero-override construction path)
python -m pytest models/autoports/zai_org_glm_4_7_flash/tests/test_full_model_perf.py -q -s
python models/autoports/zai_org_glm_4_7_flash/tests/run_qualitative_suite.py --max-new-tokens 128 --skip-hf
python -m pytest models/autoports/zai_org_glm_4_7_flash/tests/test_prefill_padding.py -q -s -p no:randomly

# context contract
python .agents/scripts/check_context_contract.py --model-dir models/autoports/zai_org_glm_4_7_flash \
    --hf-model zai-org/GLM-4.7-Flash --stage full-model --require-contract
```

## Commit SHAs

Recorded after commit (this file is amended in a follow-up commit per this
autoport's convention of separating source/evidence commits from the SHA
record, e.g. `doc/full_model/work_log.md`'s pattern).
