# GLM-4.7-Flash datatype sweep — stage report

Target: `zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`, 30.6B total / ~3.6B
active params, 47 decoder layers = 1 dense + 46 MoE, top-4-of-64 routing +
1 shared expert, vocab 154880, advertised context 202752), **one Blackhole
p150-class chip**, device 0, 1x1 mesh (`N150`), 11x10 compute grid, 8 DRAM
banks. Branch `ttmodelmanager/glm47-flash-probe`, starting from the
optimized-full-model stage. Deployment target is explicitly single-chip;
multichip parallelism is out of scope (goal), so there is no CCL dtype axis
to sweep.

## Headline

**Selected config: `C00_baseline` — the config already shipped by the
optimized-decoder/optimized-full-model stages, unchanged.** Every one of the
10 real candidates this stage measured, plus one candidate rejected without a
hardware run (hard DRAM limit), is slower, less accurate, or both.

| | value | measurement regime / workload |
|---|---|---|
| prefill top-1 / top-5 / top-100 | 0.880 / 1.000 / 1.000 | `run_prefill_check`, AIME24 chat-template reference, 100 generated-token positions |
| teacher-forced top-1 / top-5 / top-100 | 0.850 / 1.000 / 1.000 | `run_teacher_forcing`, same reference, **trace-verified** (`enable_trace=True` required and asserted) |
| TTFT | 590.28 ms | teacher-forcing harness, prompt 154 tokens (chat-template-rendered), request-boundary inclusive |
| trace-verified teacher-forcing decode | 44.02 t/s/u (22.72 ms/tok) | same run, batch 1, 99 decode steps, includes on-device sampling + one-token host readback per step (the teacher-forcing feedback loop) |
| post-selection token-out, no per-token readback | 22.879 ms/tok (43.71 t/s/u) | `tests/test_full_model_perf.py`, **normal default construction path, zero overrides** — `decode_ms_per_token.traced_model_plus_sampling`; warmed, batch 1, prompt 128 / generate 128 |
| post-selection TTFT | 334.1 ms | same run, prompt 128, warmed prefill + first token |

The teacher-forcing number (44.02 t/s/u) is this stage's ranking metric per
`$datatype-sweep` ("Always use trace-verified teacher-forcing decode t/s/u to
rank datatype candidates"). The post-selection row is the separate,
already-shipped no-per-token-readback benchmark ("warmed token-out"),
re-measured through the selected config's normal (unmodified default)
construction path, per the skill's Final Selection step. They differ because
teacher-forcing pays a host round-trip on every one of its 99 steps (needed
so the harness can force-feed the reference continuation) that the
steady-state no-readback benchmark does not.

## Accuracy bar

This model's readiness bar, stated identically in `doc/full_model/README.md`
and `doc/optimized_full_model/README.md` across many independently reviewed
stages, is **top-5 >= 0.98, top-100 = 1.00**. Top-1 is reported every time but
was never part of the stated bar — the shipped baseline itself sits at
teacher-forced top-1 = 0.850, below `$datatype-sweep`'s generic default
(top-1 >= 0.90) if that default were applied naively. The skill explicitly
allows this: *"Keep top-100 at the existing readiness expectation unless the
user or model-specific evidence explicitly changes it"* — the model-specific
evidence here (dozens of accepted review rounds citing exactly "top-5 >= 0.98,
top-100 = 1.00" as the bar, with top-1 never included) is the readiness
expectation this stage keeps. Top-1 is tracked and reported for every
candidate below because it is the metric most sensitive to the precision
changes this stage tests, and a real top-1 regression is treated as a real
finding even when it does not fail the stated bar (see `C01`/`C02`/`C05`/`C08`
below).

`top1_perf_pareto.png` therefore draws the skill's *generic* 0.90 default as
an explicitly labeled "informational only" reference line rather than
presenting it as this model's actual gate. `top5_perf_pareto.png` draws the
real 0.98 gate.

## Baseline refresh

Re-ran the main readiness reference (AIME24, chat template, 100 generated
tokens — `readiness_aime24_chat.refpt`, already at 100 tokens, so no
regeneration was needed) through this stage's own driver
(`tests/dev_datatype_sweep.py`), which builds through the exact same
`build_generator` factory and reuses the shared harness's own per-entry
evaluation functions (`run_prefill_check._run_one_entry_prefill`,
`run_teacher_forcing._run_one_entry`) verbatim. Result: prefill 0.880/1.000/
1.000, teacher-forcing 0.850/1.000/1.000, TTFT 590.28 ms, decode 44.02 t/s/u —
matching `doc/optimized_full_model/logs/run_{prefill_check,teacher_forcing}.log`
(0.880/1.000/1.000, 0.850/1.000/1.000, TTFT 590.6ms, decode 43.98-44.02 t/s/u)
within this harness's own run-to-run noise. See `runs/C00_baseline.json`.

## Plumbing added this stage

Two of the ten tensor-group policy fields already named in
`doc/optimized_decoder/README.md`'s precision table had no override hook at
all before this stage (compute fidelity was hardcoded, not a constructor
kwarg or class attribute) — `$datatype-sweep` requires adding that plumbing
before ranking candidates on those axes:

- **LM head fidelity** (`tt/model.py`): `GLM47FlashModel.from_pretrained`
  gained `lm_head_fidelity: str = "hifi2"` (previously
  `self.ck_lm_head = _ck(dev, ttnn.MathFidelity.HiFi2, False)` was hardcoded
  inline). The model instance now also exposes `expert_dtype`, `weight_dtype`,
  `embed_dtype`, `lm_head_dtype`, `lm_head_fidelity`, `decoder_cls_name` for
  the propagation check below.
- **Router/gate decode fidelity** (`tt/optimized_decoder.py`):
  `OptimizedDecoder` gained `router_fidelity = "hifi4_fp32"` (previously
  `_router_scores_decode` used `self.ck_hifi4`, inherited unconditionally from
  the functional-decoder stage). `from_state_dict` now builds
  `self.ck_router = _ck(dev, *FIDELITY[cls.router_fidelity])` and
  `_router_scores_decode` reads `self.ck_router`. `self.ck_hifi4` itself is
  left untouched (still bound to true HiFi4+fp32acc) since it is the only
  call site inside `OptimizedDecoder` that used it (verified by grep) and
  other inherited functional-stage codepaths (prefill routing) still depend
  on it unconditionally being real HiFi4.

Both defaults reproduce the exact pre-existing hardcoded values, so this is
additive plumbing, not a behavior change — confirmed bit-identical:
`test_full_model.py`'s 47 tests (including
`test_deployment_dtype_policy_preserved`, which hard-asserts the per-tensor
dtype/fidelity policy) pass unchanged, and the fresh qualitative suite run
this stage (`qualitative/`) reproduces the exact same greedy prefix agreement
counts as `doc/optimized_full_model/qualitative/` (8/128, 16/128, 45/128,
14/128, 32/128, 15/128) — bit-identical generation, not just a passing test.

Every other tensor group's dtype/fidelity was already a constructor kwarg
(`expert_dtype`, `weight_dtype`, `cache_dtype`, `lm_head_dtype`) or an
`OptimizedDecoder`/`SharedRopeDecoder` class attribute overridable before
`from_state_dict` (`attn_fidelity`, `mlp_fidelity`, `expert_fidelity`,
`attn_weight_dtype`, `mlp_gateup_dtype`, `mlp_down_dtype`, `dense_mlp_dtype`,
`prefill_proj_fidelity`, `prefill_expert_fidelity`) — this stage's driver
(`tests/dev_datatype_sweep.py`) builds a dynamic `SharedRopeDecoder` subclass
per candidate to set these, exactly the pattern `tests/dev_optimize.py`
already used at the single-decoder-layer level.

## Sweep methodology

`tests/dev_datatype_sweep.py` builds the real 47-layer model once per
candidate through `build_generator` with real constructor kwargs / a dynamic
decoder subclass, immediately introspects the constructed model's actual
`ttnn` tensors and `ttnn.init_device_compute_kernel_config` objects into a
`policy_snapshot` (proof the requested policy reached the measured runtime
path, not just this script's request), then runs the AIME24 prefill-check and
teacher-forcing checks by calling the shared harness's own per-entry
functions directly — so every number below is produced by the same code path
`doc/optimized_full_model/logs/run_{prefill_check,teacher_forcing}.log` used,
not a reimplementation. `tests/build_datatype_sweep_report.py` assembles
`sweep_results.{json,csv}` and the two Pareto PNGs from the per-candidate JSON
files under `runs/`.

```bash
# one candidate
python -m models.autoports.zai_org_glm_4_7_flash.tests.dev_datatype_sweep \
    --config-id C01_lmhead_bf4_lofi --lm-head-dtype bf4 --lm-head-fidelity lofi \
    --out doc/datatype_sweep/runs/C01_lmhead_bf4_lofi.json

# assemble the report + plots after all candidates are run
python -m models.autoports.zai_org_glm_4_7_flash.tests.build_datatype_sweep_report
```

### Default-search coverage

Mapping this stage's candidates onto `$datatype-sweep`'s coarse default
search:

1. **Canonical/comparability policy** — `C10_all_bf8_canonical` (no
   `bfloat4_b` anywhere). Not run: `doc/probe/README.md` already measured
   `bfloat8_b` routed experts alone at ~32 GB, which does not fit this box's
   31.5 GiB allocatable DRAM once the remaining layers/cache/scratch are
   added — a hard physical limit, not a speed/accuracy tradeoff to measure.
2. **BFP8 KV cache yes/no** — `C04_kvcache_bf16` (the "no" arm; `bfloat8_b` is
   already the shipped "yes" arm). Rejected: bf16 is equal on accuracy but
   slower (TTFT +5.1%, decode -0.25%).
3. **BFP8 CCL or residual-transfer activations yes/no** — N/A for CCL (single
   chip, no collective ops in the measured path, per the goal's explicit
   single-chip deployment target). Residual/activation dtype (bf16) is
   intentionally **not** swept this stage — see "Scoped out" below.
4. **BFP4 for eligible MLP/expert groups** — already the shipped policy for
   attention/shared-expert/routed-experts (established at the
   optimized-decoder stage with real-weight 202k-context PCC evidence, a
   harder test than this stage's 100-token AIME reference); `C09` retests
   dense-MLP bf4 (the one group still at bf8) and confirms the prior
   rejection at the full-model level.
5. **Restore order on failure** — not needed: no candidate failed the
   top-5/top-100 gate, so nothing needed restoring.
6. **Extend surviving bf4 choices to first/last layer** — N/A: the model's
   only first/last-layer-style split is architectural (layer 0 is the single
   dense layer; there is no separate "last MoE layer" precision policy to
   extend), and `doc/probe/README.md`'s per-expert PCC is already uniform
   across all 64 experts of the probed layer (0.980913–0.981341, no
   layer-position outlier), so a per-layer exception was not expected to help
   and was not tested.

### Compute-fidelity coverage (mandatory BFP4+LoFi / BFP4+HiFi2 pairs)

Every material `bfloat4_b` matmul group in the selected policy already has a
LoFi candidate (it *is* the shipped fidelity) and this stage adds the
required HiFi2 comparison for each:

| bf4 group | LoFi (shipped) | HiFi2 comparison | result |
|---|---|---|---|
| attention decode (wqkv_a, wq_b, w_uk, w_uv, wo) | `C00` | `C06_attn_hifi2` | ties on accuracy, **-4.9% decode** — LoFi wins |
| routed experts (gate_up + down) | `C00` | `C07_expert_hifi2` | ties on accuracy, -0.45% decode — LoFi wins |
| shared expert (gate_up + down) | `C00` | `C08_mlp_hifi2`* | top-1 -0.03, -1.2% decode — LoFi wins |

\* `mlp_fidelity` is one class attribute shared by shared-expert (bf4) and
dense-MLP (bf8), so `C08` is also this stage's BFP8+LoFi-vs-BFP8+HiFi2
comparison for dense MLP — see the table below.

The two remaining `bfloat8_b` dominant decode-projection groups also get the
skill's mandated LoFi/HiFi2 comparison even though they're not `bfloat4_b`:

| bf8 group | HiFi2 (shipped) | LoFi comparison | result |
|---|---|---|---|
| LM head | `C00` | `C03_lmhead_bf8_lofi` | top-1 -0.02, decode change +0.14% (noise) — HiFi2 wins (no benefit to LoFi) |
| dense MLP | `C00` (mlp_fidelity=lofi already) | — | dense MLP's decode fidelity is already LoFi in the shipped policy (only its *dtype* is bf8, not bf4); `C08` above is the fidelity comparison, at HiFi2, for both dense and shared together |

## Candidate matrix

Full machine-readable version: `sweep_results.json`, `sweep_results.csv`.
Hardware: 1x Blackhole p150-class chip. Mesh: `N150` (1x1). Reference:
`readiness_aime24_chat.refpt` (AIME24, chat template, 100 generated tokens).

| id | delta from baseline | tf top1 | tf top5 | tf top100 | TTFT ms | decode t/s/u | decision |
|---|---|---|---|---|---|---|---|
| C00_baseline | — (shipped policy) | 0.850 | 1.000 | 1.000 | 590.28 | **44.02** | **SELECTED** |
| C01_lmhead_bf4_lofi | LM head bf8→bf4, HiFi2→LoFi | 0.790 | 0.990 | 1.000 | 590.30 | 44.62 | REJECTED (accuracy) |
| C02_lmhead_bf4_hifi2 | LM head bf8→bf4, fidelity HiFi2 | 0.790 | 0.990 | 1.000 | 590.20 | 44.62 | REJECTED (accuracy) |
| C03_lmhead_bf8_lofi | LM head fidelity HiFi2→LoFi | 0.830 | 1.000 | 1.000 | 590.41 | 44.08 | REJECTED (accuracy, no speed gain) |
| C04_kvcache_bf16 | KV cache bf8→bf16 | 0.850 | 1.000 | 1.000 | 620.48 | 43.91 | REJECTED (slower) |
| C05_router_hifi2 | router fidelity HiFi4→HiFi2 | 0.820 | 1.000 | 1.000 | 590.72 | 44.09 | REJECTED (accuracy, no speed gain) |
| C06_attn_hifi2 | attention bf4 fidelity LoFi→HiFi2 | 0.850 | 1.000 | 1.000 | 590.33 | 41.87 | REJECTED (slower) |
| C07_expert_hifi2 | routed-expert fidelity LoFi→HiFi2 | 0.850 | 1.000 | 1.000 | 590.64 | 43.82 | REJECTED (slower) |
| C08_mlp_hifi2 | shared+dense MLP fidelity LoFi→HiFi2 | 0.820 | 1.000 | 1.000 | 590.41 | 43.48 | REJECTED (accuracy, slower) |
| C09_dense_mlp_bf4_lofi | dense MLP (1/47 layers) bf8→bf4 | 0.870 | 1.000 | 1.000 | 590.82 | 44.03 | REJECTED (no speed gain) |
| C10_all_bf8_canonical | no bf4 anywhere | — | — | — | — | — | REJECTED, not run (hard DRAM limit) |

Every rejected candidate's exact reasoning (numbers, precedent, capacity
math) is in `sweep_results.json`/`.csv`'s `reason` field and in "What was
tested" below.

## What was tested, in detail

### LM head: bf4 (C01/C02) and bf8+LoFi (C03) — the headline candidate, rejected

`doc/full_model/head_probe.json` measured the LM head's isolated matmul at
bf4 (624–658 us depending on `in0_block_w`) against bf8 (866–894 us), a ~30%
op-level win on the single largest op in the model (51.1% of model-only
device time per `doc/optimized_full_model/README.md`), and explicitly
deferred the accuracy question to this stage. Tested both fidelities
(`C01`=LoFi, `C02`=HiFi2) since bf4 is a material dtype group:

- Both give **identical** results (top-1 0.790, top-5 0.990, decode 44.62
  t/s/u): the accuracy cost is from the bf4 *dtype* quantization, not the
  fidelity choice. If bf4 LM head were ever adopted, LoFi is the right
  fidelity (no reason to pay HiFi2).
- top-5 (0.990) still clears the 0.98 bar and top-100 stays 1.00, so this
  candidate technically *passes* the stated gate. Top-1 drops 0.850→0.790
  (-0.060 absolute, -7.1% relative) — a real, repeatable regression, not
  noise (both fidelity arms hit exactly the same number).
- The full-model decode gain is **+1.36%** (44.02→44.62 t/s/u), much smaller
  than the op-level ~30% figure: the LM head is only ~4% of the 47-layer
  model-only step, so its isolated speedup doesn't scale to the full model.
- **Rejected on precedent already established by this exact autoport**:
  `doc/full_model/work_log.md` FM-021 measured a *smaller* magnitude change
  (an `in0_block_w` K-blocking change, not even a dtype change) that produced
  the identical top-1 drop (0.850→0.790) for a **0.04%** e2e gain, and this
  stage's own team reverted it, writing "five points of top-1 and a top-5
  that leaves 1.000 are not worth it... an LM-head program-config change is
  an accuracy change." This stage's bf4 candidate offers a bigger gain
  (1.36% vs 0.04%) but the same accuracy cost; applying the same standard
  that rejected the smaller-gain change rejects this one too.
- The LM head is not capacity-constrained: bf8 costs 0.314 GiB
  (`doc/full_model/README.md`'s dram budget), bf4 would save at most ~0.15
  GiB, irrelevant against the model's 8.15 GiB of headroom. Unlike routed
  experts, there is no capacity argument for bf4 here.
- `C03` (fidelity-only, dtype stays bf8) shows the same shape at smaller
  scale: top-1 0.850→0.830 for a decode change (+0.14%) inside this harness's
  own noise band. No benefit, so no reason to pay any accuracy risk.

**Decision: keep bf8+HiFi2.** Recorded as a real, quantified, measured, and
rejected candidate — not a dismissal.

### KV cache: bf16 (C04) — rejected, slower with no accuracy benefit

Identical teacher-forced accuracy to bf8 (0.850/1.000/1.000 both ways) but
TTFT regresses 590.28→620.48 ms (+5.1%, doubled cache read/write DRAM
traffic during the 154-token prefill) and decode is marginally slower
(44.02→43.91 t/s/u, -0.25%). Matches the already-committed 202k real-weight
long-context evidence in `doc/context_contract.json`'s `optimized_decoder`
section (bf8 == bf16 within noise on accuracy at full context). bf8 wins on
speed with zero accuracy cost either way; see the "Context contract" section
below for the capacity side of this comparison.

### Router fidelity: HiFi2 (C05) — rejected, real accuracy cost for no speed gain

Prefill routing is unaffected by this stage's plumbing (unchanged, still
HiFi4 via the inherited `ck_hifi4`), so prefill numbers are identical to
baseline. Decode router fidelity at HiFi2: top-1 0.850→0.820 (-0.030
absolute) from routing-decision flips under lower fidelity, for a decode
change of +0.16% — inside noise, and matching the isolated-op finding in
`doc/optimized_full_model/README.md` item 2 (~0.19% of the model-only step).
Router numerics feed top-4-of-64 expert selection, a tensor this codebase
already treats as correctness-sensitive (`dev_optimize.py --check-ties`).
**Decision: keep HiFi4+fp32acc.**

### Attention (C06), routed experts (C07), shared+dense MLP (C08) fidelity: HiFi2 — all rejected, LoFi wins outright

All three ties on accuracy (identical top-1/top-5/top-100 to baseline for
C06/C07; C08 regresses top-1 -0.03 because it also covers dense MLP's
fidelity) while being measurably slower: -4.9% (attention, the largest
non-LM-head delta in this sweep), -0.45% (experts), -1.2% (combined
shared+dense MLP). LoFi is unambiguously the right fidelity for every bf4
weight group already in the shipped policy. **Decision: keep LoFi
everywhere.**

### Dense MLP dtype: bf4 (C09) — rejected, no speed benefit regardless of accuracy

Mixed/noisy accuracy signal at this sample size: prefill top-1 0.880→0.850
but teacher-forced top-1 0.850→0.870 (higher). Decode is unchanged within
noise (44.02→44.03 t/s/u, +0.02%) because dense is only 1 of 47 layers — no
full-model speed benefit exists to adopt this candidate regardless of how
the accuracy question resolves. This matches (and does not need to relitigate)
the decoder-level rejection already on record in
`doc/optimized_decoder/README.md`: real-weight 202k dense-control regression
(decode@202751 0.99865 vs 0.99993) for the same 1-of-47-layers reason.
**Decision: keep bf8.**

### Canonical all-bf8 arm (C10) — rejected without a hardware run

`doc/probe/README.md` already measured `bfloat8_b` routed experts at ~32 GB
of expert weights alone — this does not fit the single p150's 31.5 GiB
allocatable DRAM (`doc/context_contract.json`'s
`full_model.measured_allocatable_dram_gib`) once the remaining decoder-layer
weights, KV cache and sampler scratch are added. This is a hard physical
DRAM limit, not a speed/accuracy tradeoff — running it would OOM at model
construction, before either check in this sweep's gate could execute, so it
is recorded as rejected-without-run rather than a wasted hardware trial.

## Pareto interpretation

`top1_perf_pareto.png` and `top5_perf_pareto.png` plot all 10 executed
candidates (accuracy on the x-axis, trace-verified teacher-forcing decode
t/s/u on the y-axis), the non-dominated Pareto frontier among them, the
selected point in red, and a vertical dotted reference line (the real 0.98
gate on the top-5 chart; the skill's generic informational-only 0.90 default
on the top-1 chart, since top-1 has no model-specific gate here — see
"Accuracy bar"). `C10` (not run) is noted in a caption, not plotted, since it
has no measured coordinate.

Reading the frontier: `C01`/`C02` (bf4 LM head) sit at the fast-but-
less-accurate end (44.62 t/s/u at top-1=0.79); `C09` (dense bf4) sits at the
accurate-and-tied-on-speed point (top-1=0.87, 44.03 t/s/u — its top-1 number
is a noisy *improvement*, not a controlled win, per the "What was tested"
discussion above); `C00` (selected) sits in the middle of the frontier at the
highest speed among the candidates that keep baseline-or-better accuracy
with no noisy signal. Every point below the frontier (`C03`–`C08`) is a
fidelity change that cost speed, accuracy, or both, for no compensating
benefit. On the top-5 chart, all candidates except `C01`/`C02` sit at
top-5=1.000, so the frontier is effectively vertical there — the real
tradeoff this sweep found is almost entirely on top-1 and decode speed, not
top-5.

## Context contract

KV-cache dtype is unchanged (bf8 selected, matching every prior stage).
`doc/context_contract.json` gains a `datatype_sweep` section recording: no
capability reduction, the existing bf8 budget figures still apply unmodified,
and the tested bf16 alternative (`C04`) would also still fit (projected ~28.5
GiB resident vs 31.5 GiB allocatable, ~3.0 GiB headroom) — it was rejected on
speed, not forced out by capacity. `supported_context` stays 202752 (full
HF-advertised context, no reduction) either way.

## Non-aligned prompt check

The selected config changes no KV-cache dtype, cache layout, trace buffer, or
prefill-chunking behavior from the optimized-decoder/full-model stages, so
`$datatype-sweep`'s "rerun if changed" trigger does not fire. Reran anyway,
fresh, after this stage's plumbing edits: `tests/test_prefill_padding.py` (13
passed) and the full `tests/test_full_model.py` suite (47 passed, including
`test_host_logits_paths_compile_nothing_at_an_unaligned_length` and
`test_single_chunk_prompt_shape_does_not_recapture`).

## Propagation check

`tests/check_selected_precision_config.py` proves, without opening a device:

1. `GLM47FlashModel.from_pretrained`'s live keyword defaults and
   `SharedRopeDecoder`'s live class attributes — what `build_generator()`
   actually uses when nothing overrides them — match
   `selected_precision_config.json`'s `construction` block field-for-field.
2. Those same defaults match `runs/C00_baseline.json`'s `policy_snapshot`,
   which was introspected from a real, once-built 47-layer model's actual
   `ttnn` tensors and compute-kernel-config objects (not from requested
   kwargs) — the "a JSON field ignored by hard-coded model code does not
   satisfy this requirement" check.

```
$ python -m models.autoports.zai_org_glm_4_7_flash.tests.check_selected_precision_config
PRECISION_CONFIG_PROPAGATION_CHECK: OK
```

No vLLM adapter exists yet for this autoport (vLLM integration is explicitly
out of scope for this goal). `selected_precision_config.json`'s
`construction.vllm_adapter_note` records the requirement for whoever builds
that adapter: call `build_generator()`/`from_pretrained()` with no overrides,
or with overrides reproducing this file's kwargs/class-attrs, to inherit this
policy.

## Scoped out of this sweep

- **Activation/residual dtype (bf16 throughout).** This is an
  architecture-wide convention set at the functional-decoder stage (every
  op's input/output contract assumes it), not a per-tensor-group dtype knob
  like the weight groups above. `$datatype-sweep`'s "try BFP8 residual-
  transfer activations as a yes/no switch" is a coarse suggestion for models
  where this is already a policy field; here it would mean redesigning the
  residual stream's contract across every op in the model, which is
  architecture work outside a single sweep stage's safe scope and outside
  this stage's review budget. Recorded as unswept, not as tested-and-passed.
- **Logits/sampling dtype.** No lower-precision sampling/logits dtype
  candidate exists anywhere in this codebase to sweep; the current policy
  (bf16 L1 decode logits, device-side split top-k sampling) is recorded in
  `selected_precision_config.json` for completeness, unchanged.
- **CCL dtype.** N/A — single chip, no collective ops in the measured path.

## Artifacts

- `sweep_results.json`, `sweep_results.csv` — the full candidate matrix.
- `selected_precision_config.json` — the winning policy, proof-of-consumption
  fields, and the `vllm_adapter_note` for the future vLLM stage.
- `runs/*.json` — one file per candidate: requested kwargs, the introspected
  `policy_snapshot`, per-entry and aggregate prefill-check/teacher-forcing
  results.
- `top1_perf_pareto.png`, `top5_perf_pareto.png` — the two required Pareto
  charts.
- `postselection_perf.json`, `postselection_capacity.json` — the
  post-selection no-per-token-readback benchmark (`test_full_model_perf.py`,
  normal default construction path), copied out of `doc/full_model/`'s
  hard-coded output path immediately after the run, with `doc/full_model/`
  restored (`git checkout`) before this stage's commit — the same
  clobber-avoidance pattern `doc/optimized_full_model/README.md`'s "Known
  limitations" section documents for the prior stage.
- `qualitative/` — fresh this stage (`--skip-hf`), reproduces
  `doc/optimized_full_model/qualitative/`'s greedy prefix agreement counts
  bit-for-bit.
- `tests/dev_datatype_sweep.py` — the per-candidate driver.
- `tests/build_datatype_sweep_report.py` — assembles the JSON/CSV/PNGs from
  `runs/`.
- `tests/check_selected_precision_config.py` — the propagation check.
- `work_log.md` — chronological narrative, exact commands, and commit SHAs.

## Known limitations (freshness / provenance / style; no open correctness bug)

- The Pareto charts' top-1 threshold line is the skill's *generic* default
  (0.90), not a bar this model is actually gated on — see "Accuracy bar".
  Labeled as informational-only directly on the chart to avoid misreading.
- `C09`'s teacher-forced top-1 *improvement* (0.850→0.870) at 100 samples is
  not large enough to treat as a real per-token win; it is reported as noise
  because the candidate has no speed benefit to justify chasing a tighter
  accuracy read regardless of which way that number moves.
- `probe/README.md`'s per-expert PCC evidence (cited above for the
  first/last-layer-exception discussion) is from one probed MoE layer with
  512 synthetic random-normal tokens, real checkpoint weights — a
  weight-quantization-sensitivity measurement, not an activation-distribution
  one; this stage treats it as sufficient grounds to not re-test a per-layer
  bf8 exception (consistent with `$datatype-sweep`'s allowance to use
  non-full-model evidence for ordering/debugging), not as full-model
  evidence in its own right.
- This stage's `tests/dev_datatype_sweep.py` opens and closes the mesh device
  once per candidate (a fresh subprocess-equivalent construction each time,
  matching how the CLI readiness tools are normally invoked) rather than
  reusing one open device across candidates; this was a deliberate choice for
  DRAM-cleanliness between candidates (47 layers at ~17-18 GiB resident
  leaves no room for two models at once), not a discovered constraint.
