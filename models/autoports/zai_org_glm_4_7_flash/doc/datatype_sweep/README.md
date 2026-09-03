# GLM-4.7-Flash datatype sweep -- stage report

Target: `zai-org/GLM-4.7-Flash` (`Glm4MoeLiteForCausalLM`, 30.6B total / ~3.6B
active params, 47 decoder layers = 1 dense + 46 MoE, top-4-of-64 routing +
1 shared expert, vocab 154880, advertised context 202752), **one Blackhole
p150-class chip**, device 0, 1x1 mesh (`N150`), 11x10 compute grid, 8 DRAM
banks. Branch `ttmodelmanager/glm47-flash-probe`, starting from the
optimized-full-model stage. Deployment target is explicitly single-chip;
multichip parallelism is out of scope (goal), so there is no CCL dtype axis
to sweep.

## Headline

**Selected config: `C00_baseline` -- the config already shipped by the
optimized-decoder/optimized-full-model stages, unchanged.** The real gate
this stage selects against is top5>=0.98, top100=1.00, and no teacher-forced
top-1 regression from this config's own 0.850 baseline value (see "Accuracy
bar" below). Five of the ten evaluated candidates fail that gate on top-1
alone; the other four that pass it are all measurably slower, or (`C09`)
faster only by a margin smaller than this harness's own measurement noise
while carrying an already-documented long-context accuracy risk this stage's
short reference cannot see -- see "What was tested" for the full, honest
picture (an earlier draft of this table said every alternative was strictly
worse, which is not accurate; corrected after independent review, work log
DS-007).

| | value | measurement regime / workload |
|---|---|---|
| prefill top-1 / top-5 / top-100 | 0.880 / 1.000 / 1.000 | `run_prefill_check`, AIME24 chat-template reference, 100 generated-token positions |
| teacher-forced top-1 / top-5 / top-100 | 0.850 / 1.000 / 1.000 | `run_teacher_forcing`, same reference, **trace-verified** (`enable_trace=True` required and asserted) |
| TTFT | 590.53 ms | teacher-forcing harness, prompt 154 tokens (chat-template-rendered), request-boundary inclusive |
| trace-verified teacher-forcing decode | 44.00 t/s/u (22.73 ms/tok) | same run, batch 1, 99 decode steps, includes on-device sampling + one-token host readback per step (the teacher-forcing feedback loop) |
| post-selection token-out, no per-token readback | 22.879 ms/tok (43.71 t/s/u) | `tests/test_full_model_perf.py`, **normal default construction path, zero overrides** -- `decode_ms_per_token.traced_model_plus_sampling`; warmed, batch 1, prompt 128 / generate 128 |
| post-selection TTFT | 334.1 ms | same run, prompt 128, warmed prefill + first token |

The teacher-forcing number (44.00 t/s/u) is this stage's ranking metric per
`$datatype-sweep` ("Always use trace-verified teacher-forcing decode t/s/u to
rank datatype candidates"). The post-selection row is the separate,
already-shipped no-per-token-readback benchmark ("warmed token-out"),
re-measured through the selected config's normal (unmodified default)
construction path, per the skill's Final Selection step. They differ because
teacher-forcing pays a host round-trip on every one of its 99 steps (needed
so the harness can force-feed the reference continuation) that the
steady-state no-readback benchmark does not.

## Accuracy bar

This model's top5/top100 readiness bar, stated identically in
`doc/full_model/README.md` and `doc/optimized_full_model/README.md` across
many independently reviewed stages, is **top-5 >= 0.98, top-100 = 1.00**.
Top-1 was reported every time in those stages but was never assigned a fixed
threshold there -- the shipped baseline itself sits at teacher-forced
top-1 = 0.850, below `$datatype-sweep`'s generic default (top-1 >= 0.90) if
that default were applied naively (which would fail the already-accepted
baseline this stage starts from). The goal contract for *this* stage names a
concrete "top-1/top-5 gate", and this exact autoport's FM-021 precedent
(`doc/full_model/work_log.md`) treats any material top-1 regression from a
precision change as a real, disqualifying finding regardless of whether it
happens to clear top-5/top-100. This stage's gate operationalizes both of
those into one concrete, checkable rule:

> **top5 >= 0.98 AND top100 = 1.00 AND top1 >= 0.850** (this config's own
> baseline value, not a fixed universal threshold -- because the model's own
> accepted history, not a generic default, is what "regression" means here).

Every candidate is checked against exactly that rule (`pass_fail` in
`sweep_results.json`/`.csv`); `C01`, `C02`, `C03`, `C05`, `C08` fail it on
top-1 alone regardless of speed. Both Pareto charts draw this real gate: the
0.850 line on `top1_perf_pareto.png`, the 0.98 line on `top5_perf_pareto.png`.

## Baseline refresh

Re-ran the main readiness reference (AIME24, chat template, 100 generated
tokens -- `readiness_aime24_chat.refpt`, already at 100 tokens, so no
regeneration was needed) through this stage's own driver
(`tests/dev_datatype_sweep.py`), which builds through the exact same
`build_generator` factory and reuses the shared harness's own per-entry
evaluation functions (`run_prefill_check._run_one_entry_prefill`,
`run_teacher_forcing._run_one_entry`) verbatim. Result: prefill 0.880/1.000/
1.000, teacher-forcing 0.850/1.000/1.000, TTFT 590.53 ms, decode 44.00 t/s/u --
matching `doc/optimized_full_model/logs/run_{prefill_check,teacher_forcing}.log`
(0.880/1.000/1.000, 0.850/1.000/1.000, TTFT 590.6ms, decode 43.98-44.02 t/s/u)
within this harness's own run-to-run noise. See `runs/C00_baseline.json`.

## Plumbing added this stage

Two of the ten tensor-group policy fields already named in
`doc/optimized_decoder/README.md`'s precision table had no override hook at
all before this stage (compute fidelity was hardcoded, not a constructor
kwarg or class attribute) -- `$datatype-sweep` requires adding that plumbing
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
additive plumbing, not a behavior change -- confirmed bit-identical:
`test_full_model.py`'s 47 tests (including
`test_deployment_dtype_policy_preserved`, which hard-asserts the per-tensor
dtype/fidelity policy) pass unchanged, and the fresh qualitative suite run
this stage (`qualitative/`) reproduces the exact same greedy prefix agreement
counts as `doc/optimized_full_model/qualitative/` (8/128, 16/128, 45/128,
14/128, 32/128, 15/128) -- bit-identical generation, not just a passing test.

Every other tensor group's dtype/fidelity was already a constructor kwarg
(`expert_dtype`, `weight_dtype`, `cache_dtype`, `lm_head_dtype`) or an
`OptimizedDecoder`/`SharedRopeDecoder` class attribute overridable before
`from_state_dict` (`attn_fidelity`, `mlp_fidelity`, `expert_fidelity`,
`attn_weight_dtype`, `mlp_gateup_dtype`, `mlp_down_dtype`, `dense_mlp_dtype`,
`prefill_proj_fidelity`, `prefill_expert_fidelity`) -- this stage's driver
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
functions directly -- so every number below is produced by the same code path
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

1. **Canonical/comparability policy** -- `C10_all_bf8_canonical` (no
   `bfloat4_b` anywhere). Not run: `doc/probe/README.md` already measured
   `bfloat8_b` routed experts alone at ~32 GB, which does not fit this box's
   31.5 GiB allocatable DRAM once the remaining layers/cache/scratch are
   added -- a hard physical limit, not a speed/accuracy tradeoff to measure.
2. **BFP8 KV cache yes/no** -- `C04_kvcache_bf16` (the "no" arm; `bfloat8_b` is
   already the shipped "yes" arm). Passes the gate but not selected: bf16 is
   equal on accuracy but slower (TTFT +5.1%, decode -0.25%).
3. **BFP8 CCL or residual-transfer activations yes/no** -- N/A for CCL (single
   chip, no collective ops in the measured path, per the goal's explicit
   single-chip deployment target). Residual/activation dtype (bf16) is
   intentionally **not** swept this stage -- see "Scoped out" below.
4. **BFP4 for eligible MLP/expert groups** -- already the shipped policy for
   attention/shared-expert/routed-experts (established at the
   optimized-decoder stage with real-weight 202k-context PCC evidence, a
   harder test than this stage's 100-token AIME reference); `C09` retests
   dense-MLP bf4 (the one group still at bf8). It passes this stage's gate
   with no full-model speed benefit (+0.02%, within noise) and does not
   overturn the decoder-level long-context rejection already on record --
   see "What was tested" below for why it is not selected despite passing.
5. **Restore order on failure** -- not needed: no candidate failed the
   top-5/top-100 gate, so nothing needed restoring.
6. **Extend surviving bf4 choices to first/last layer** -- N/A: the model's
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
| attention decode (wqkv_a, wq_b, w_uk, w_uv, wo) | `C00` | `C06_attn_hifi2` | ties on accuracy, **-4.9% decode** -- LoFi wins |
| routed experts (gate_up + down) | `C00` | `C07_expert_hifi2` | ties on accuracy, -0.45% decode -- LoFi wins |
| shared expert (gate_up + down) | `C00` | `C08_mlp_hifi2`* | top-1 -0.03, -1.2% decode -- LoFi wins |

\* `mlp_fidelity` is one class attribute shared by shared-expert (bf4) and
dense-MLP (bf8), so `C08` is also this stage's BFP8+LoFi-vs-BFP8+HiFi2
comparison for dense MLP -- see the table below.

The two remaining `bfloat8_b` dominant decode-projection groups also get the
skill's mandated LoFi/HiFi2 comparison even though they're not `bfloat4_b`:

| bf8 group | HiFi2 (shipped) | LoFi comparison | result |
|---|---|---|---|
| LM head | `C00` | `C03_lmhead_bf8_lofi` | top-1 -0.02, decode change +0.14% (noise) -- HiFi2 wins (no benefit to LoFi) |
| dense MLP | `C00` (mlp_fidelity=lofi already) | -- | dense MLP's decode fidelity is already LoFi in the shipped policy (only its *dtype* is bf8, not bf4); `C08` above is the fidelity comparison, at HiFi2, for both dense and shared together |

## Candidate matrix

Full machine-readable version: `sweep_results.json`, `sweep_results.csv`.
Hardware: 1x Blackhole p150-class chip. Mesh: `N150` (1x1). Reference:
`readiness_aime24_chat.refpt` (AIME24, chat template, 100 generated tokens).

| id | delta from baseline | tf top1 | tf top5 | tf top100 | TTFT ms | decode t/s/u | pass_fail | decision |
|---|---|---|---|---|---|---|---|---|
| C00_baseline | -- (shipped policy) | 0.850 | 1.000 | 1.000 | 590.53 | 44.00 | PASS | **SELECTED** |
| C01_lmhead_bf4_lofi | LM head bf8→bf4, HiFi2→LoFi | 0.790 | 0.990 | 1.000 | 590.59 | 44.49 | FAIL | FAIL_TOP1 |
| C02_lmhead_bf4_hifi2 | LM head bf8→bf4, fidelity HiFi2 | 0.790 | 0.990 | 1.000 | 590.23 | 44.48 | FAIL | FAIL_TOP1 |
| C03_lmhead_bf8_lofi | LM head fidelity HiFi2→LoFi | 0.830 | 1.000 | 1.000 | 590.65 | 44.05 | FAIL | FAIL_TOP1 |
| C04_kvcache_bf16 | KV cache bf8→bf16 | 0.850 | 1.000 | 1.000 | 619.79 | 43.91 | PASS | PASS_NOT_SELECTED (slower) |
| C05_router_hifi2 | router fidelity HiFi4→HiFi2 | 0.820 | 1.000 | 1.000 | 590.60 | 44.01 | FAIL | FAIL_TOP1 |
| C06_attn_hifi2 | attention bf4 fidelity LoFi→HiFi2 | 0.850 | 1.000 | 1.000 | 590.44 | 41.87 | PASS | PASS_NOT_SELECTED (much slower) |
| C07_expert_hifi2 | routed-expert fidelity LoFi→HiFi2 | 0.850 | 1.000 | 1.000 | 590.41 | 43.82 | PASS | PASS_NOT_SELECTED (slower) |
| C08_mlp_hifi2 | shared+dense MLP fidelity LoFi→HiFi2 | 0.820 | 1.000 | 1.000 | 590.82 | 43.47 | FAIL | FAIL_TOP1 |
| C09_dense_mlp_bf4_lofi | dense MLP (1/47 layers) bf8→bf4 | 0.870 | 1.000 | 1.000 | 590.69 | **44.02** | PASS | PASS_NOT_SELECTED (see below) |
| C10_all_bf8_canonical | no bf4 anywhere | -- | -- | -- | -- | -- | NOT_RUN | NOT_RUN (hard DRAM limit) |

`pass_fail` is the concrete gate from "Accuracy bar" above (top5>=0.98 AND
top100=1.00 AND top1>=0.850). `decision` is this stage's actual selection,
which is a separate, stricter judgment on top of `pass_fail`: `C09` passes
the gate and is nominally the fastest passing candidate (44.02 vs `C00`'s
44.00 t/s/u), but that delta is smaller than this harness's own run-to-run
noise and `C09` carries a documented long-context regression this stage's
short reference can't see -- see "What was tested" below for the full
reasoning, and `sweep_results.json`/`.csv`'s `reason` field for every
candidate's exact numbers and precedent citations.

## What was tested, in detail

### LM head: bf4 (C01/C02) and bf8+LoFi (C03) -- the headline candidate, fails the top-1 gate

`doc/full_model/head_probe.json` measured the LM head's isolated matmul at
bf4 (624–658 us depending on `in0_block_w`) against bf8 (866–894 us), a ~30%
op-level win on the single largest op in a **reduced 2-layer profile**
(51.1% of model-only device time in that reduced capture, per
`doc/optimized_full_model/README.md`'s signposted-window measurement -- the
real 47-layer model-only step is a different, much larger denominator, see
below), and explicitly deferred the accuracy question to this stage. Tested
both fidelities (`C01`=LoFi, `C02`=HiFi2) since bf4 is a material dtype
group:

- Both give **identical** results (top-1 0.790, top-5 0.990, decode ~44.49
  t/s/u): the accuracy cost is from the bf4 *dtype* quantization, not the
  fidelity choice. If bf4 LM head were ever adopted, LoFi is the right
  fidelity (no reason to pay HiFi2).
- top-5 (0.990) still clears the 0.98 bar and top-100 stays 1.00. Top-1
  drops 0.850→0.790 (-0.060 absolute, -7.1% relative, below this stage's
  0.850 no-regression gate) -- a real, repeatable regression, not noise
  (both fidelity arms hit exactly the same number). **This fails the real
  gate** (top1>=0.850) even though it clears top5/top100 alone.
- The full-model decode gain is **+1.1%** (44.00→44.49 t/s/u), much smaller
  than the op-level ~30% figure: the LM head is only ~4% of the 47-layer
  model-only step, so its isolated speedup doesn't scale to the full model.
- **Rejected on precedent already established by this exact autoport**:
  `doc/full_model/work_log.md` FM-021 measured a *smaller* magnitude change
  (an `in0_block_w` K-blocking change, not even a dtype change) that produced
  the identical top-1 drop (0.850→0.790) for a **0.04%** e2e gain, and this
  stage's own team reverted it, writing "five points of top-1 and a top-5
  that leaves 1.000 are not worth it... an LM-head program-config change is
  an accuracy change." This stage's bf4 candidate offers a bigger gain
  (1.1% vs 0.04%) but the same accuracy cost; applying the same standard
  that rejected the smaller-gain change rejects this one too.
- The LM head is not capacity-constrained: bf8 costs 0.314 GiB
  (`doc/full_model/README.md`'s dram budget), bf4 would save at most ~0.15
  GiB, irrelevant against the model's 8.15 GiB of headroom. Unlike routed
  experts, there is no capacity argument for bf4 here.
- `C03` (fidelity-only, dtype stays bf8) shows the same shape at smaller
  scale: top-1 0.850→0.830 for a decode change (+0.14%) inside this harness's
  own noise band. No benefit, so no reason to pay any accuracy risk. It also
  fails the real gate (top1 0.830 < 0.850).

**Decision: keep bf8+HiFi2.** All three candidates (C01, C02, C03) fail this
stage's real top-1 gate outright; recorded as real, quantified, measured
failures, not dismissals.

### KV cache: bf16 (C04) -- passes the gate, not selected: slower with no accuracy benefit

Identical teacher-forced accuracy to bf8 (0.850/1.000/1.000 both ways, so
this candidate passes the gate) but TTFT regresses 590.53→619.79 ms (+5.0%,
doubled cache read/write DRAM traffic during the 154-token prefill) and
decode is marginally slower (44.00→43.91 t/s/u, -0.2%). Matches the
already-committed 202k real-weight long-context evidence in
`doc/context_contract.json`'s `optimized_decoder` section (bf8 == bf16
within noise on accuracy at full context). bf8 wins on speed with zero
accuracy cost either way; see the "Context contract" section below for the
capacity side of this comparison.

### Router fidelity: HiFi2 (C05) -- fails the gate, real accuracy cost for no speed gain

Prefill routing is unaffected by this stage's plumbing (unchanged, still
HiFi4 via the inherited `ck_hifi4`), so prefill numbers are identical to
baseline. Decode router fidelity at HiFi2: top-1 0.850→0.820 (-0.030
absolute, below the 0.850 gate), consistent with (though not directly
measured as) routing-decision sensitivity under lower router fidelity --
this codebase's `tests/dev_optimize.py --check-ties` tooling exists for
verifying expert-selection ties/flips directly, which this stage did not
run. Decode change is +0.03% (44.00→44.01 t/s/u) -- inside noise, and
matching the isolated-op finding in `doc/optimized_full_model/README.md`
item 2 (~0.19% of the model-only step). Router numerics feed top-4-of-64
expert selection, a tensor this codebase already treats as
correctness-sensitive. **Decision: keep HiFi4+fp32acc.**

### Attention (C06), routed experts (C07), shared+dense MLP (C08) fidelity: HiFi2 -- LoFi wins outright

C06/C07 pass the gate (identical top-1/top-5/top-100 to baseline) but are
measurably slower: -4.9% (attention, the largest non-LM-head speed delta in
this sweep) and -0.45% (experts). C08 fails the gate outright (top-1
0.850→0.820) *and* is slower (-1.2%) because it also covers dense MLP's
fidelity. LoFi is unambiguously the right fidelity for every bf4 weight
group already in the shipped policy. **Decision: keep LoFi everywhere.**

### Dense MLP dtype: bf4 (C09) -- passes the gate, the fastest passing candidate, still not selected

Mixed signal at this sample size: prefill top-1 0.880→0.850 but
teacher-forced top-1 0.850→0.870 (*higher* than baseline -- this candidate
passes the gate with room to spare on this stage's own numbers). Decode is
44.00→44.02 t/s/u (+0.02%), making `C09` this stage's single fastest
gate-passing candidate. That +0.02% is, however, far inside this harness's
own measurement noise (`C00` itself reproduces at 43.98-44.02 t/s/u across
separate runs in this stage -- see "Baseline refresh"), and only 1 of 47
layers is dense, so there is no real full-model speed benefit either way.
Per `$datatype-sweep`'s explicit tie-break rule ("if two configs are within
measurement noise, prefer the simpler and safer one"), a noise-level tie is
not enough on its own to prefer `C09` -- and there is a specific reason to
prefer `C00` here rather than just defaulting to it: this matches (and does
not need to relitigate) the decoder-level rejection already on record in
`doc/optimized_decoder/README.md`: real-weight 202k dense-control regression
(decode@202751 0.99865 vs 0.99993, end window 28/32 vs 29/32 rows) for the
same 1-of-47-layers reason -- evidence from a much harder, already-committed
long-context test that this stage's own short 154-token/100-position sample
cannot see or overturn. **Decision: keep bf8 (`C09` is `PASS_NOT_SELECTED`,
not rejected outright -- it genuinely clears this stage's gate, it is just
not preferred over the simpler baseline).**

### Canonical all-bf8 arm (C10) -- rejected without a hardware run

`doc/probe/README.md` already measured `bfloat8_b` routed experts at ~32 GB
of expert weights alone -- this does not fit the single p150's 31.5 GiB
allocatable DRAM (`doc/context_contract.json`'s
`full_model.measured_allocatable_dram_gib`) once the remaining decoder-layer
weights, KV cache and sampler scratch are added. This is a hard physical
DRAM limit, not a speed/accuracy tradeoff -- running it would OOM at model
construction, before either check in this sweep's gate could execute, so it
is recorded as rejected-without-run rather than a wasted hardware trial.

## Pareto interpretation

`top1_perf_pareto.png` and `top5_perf_pareto.png` plot all 10 executed
candidates (accuracy on the x-axis, trace-verified teacher-forcing decode
t/s/u on the y-axis), the non-dominated Pareto frontier among them, the
selected point in red, and a vertical dotted reference line at the real gate
for that axis (top1>=0.850 on the top-1 chart, top5>=0.98 on the top-5
chart -- see "Accuracy bar"). `C10` (not run) is noted in a caption, not
plotted, since it has no measured coordinate.

**Reading the frontier honestly: `C00` (selected, red) is *not* on the
non-dominated frontier on either chart -- it sits just inside it.** On
`top1_perf_pareto.png` the frontier is `{C02, C03, C05, C09}`: `C09`
(top-1=0.87, decode=44.02) dominates `C00` (top-1=0.85, decode=44.00) by a
hair on both axes, and the frontier line visibly passes through `C09` above
`C00`'s red marker rather than through it. On `top5_perf_pareto.png` the
frontier is `{C02, C05}`; `C00` sits inside it there too (`C05` has the same
top-5=1.000 with marginally higher decode). This is expected and fine, not a
defect: the skill's own tie-break rule is exactly for this situation
(dominance within measurement noise), and "What was tested" above explains
in full why `C09` -- the only candidate that actually dominates `C00` -- is
not preferred despite that.

Reading the rest of the frontier: `C01`/`C02` (bf4 LM head) sit at the
fast-but-less-accurate end (~44.5 t/s/u at top-1=0.79) -- fast, but they fail
the real gate on top-1 alone. `C03` and `C05` sit on the frontier at
intermediate accuracy/speed points but also fail the top-1 gate. Every point
strictly below the frontier (`C04`, `C06`, `C07`, `C08`) is a fidelity or
dtype change that cost speed, accuracy, or both, for no compensating benefit.
On the top-5 chart, all candidates except `C01`/`C02` sit at top-5=1.000, so
the frontier is effectively vertical there -- the real tradeoff this sweep
found is almost entirely on top-1 and decode speed, not top-5.

## Context contract

KV-cache dtype is unchanged (bf8 selected, matching every prior stage).
`doc/context_contract.json` gains a `datatype_sweep` section recording: no
capability reduction, the existing bf8 budget figures still apply unmodified,
and the tested bf16 alternative (`C04`) would also still fit -- 28.49 GiB
resident vs 31.5 GiB allocatable, 2.68 GiB headroom net of the same 0.326
GiB trace-region reservation the bf8 baseline's 8.152 GiB headroom already
nets out, so the two numbers are apples-to-apples. This is measured, not
modeled: `tests/dev_datatype_sweep.py` never overrides `max_seq_len`, so
`C04` actually built and ran with `cache_dtype=bfloat16` at the default
`max_seq_len` (202752, the full advertised context) -- not selected on
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
   `SharedRopeDecoder`'s live class attributes -- what `build_generator()`
   actually uses when nothing overrides them -- match
   `selected_precision_config.json`'s `construction` block field-for-field.
2. Those same defaults match `runs/C00_baseline.json`'s `policy_snapshot`,
   which was introspected from a real, once-built 47-layer model's actual
   `ttnn` tensors and compute-kernel-config objects (not from requested
   kwargs) -- the "a JSON field ignored by hard-coded model code does not
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
- **CCL dtype.** N/A -- single chip, no collective ops in the measured path.

## Artifacts

- `sweep_results.json`, `sweep_results.csv` -- the full candidate matrix.
- `selected_precision_config.json` -- the winning policy, proof-of-consumption
  fields, and the `vllm_adapter_note` for the future vLLM stage.
- `runs/*.json` -- one file per candidate: requested kwargs, the introspected
  `policy_snapshot`, per-entry and aggregate prefill-check/teacher-forcing
  results.
- `top1_perf_pareto.png`, `top5_perf_pareto.png` -- the two required Pareto
  charts.
- `postselection_perf.json`, `postselection_capacity.json` -- the
  post-selection no-per-token-readback benchmark (`test_full_model_perf.py`,
  normal default construction path), copied out of `doc/full_model/`'s
  hard-coded output path immediately after the run, with `doc/full_model/`
  restored (`git checkout`) before this stage's commit -- the same
  clobber-avoidance pattern `doc/optimized_full_model/README.md`'s "Known
  limitations" section documents for the prior stage.
- `qualitative/` -- the TT-side generations are fresh this stage and
  reproduce `doc/optimized_full_model/qualitative/`'s greedy prefix
  agreement counts bit-for-bit; the HF-side control (`hf_control.json`) is
  reused verbatim via `--skip-hf` (it is a property of the checkpoint, not
  of this port, and costs ~25 minutes to regenerate) -- byte-identical to
  the prior stage's HF control file, as intended.
- `logs/` -- terminal logs from this stage's hardware runs and checks
  (per-candidate sweep logs, the post-selection/qualitative/non-aligned-
  prompt/decoder-suite runs, the propagation and context-contract checks).
- `tests/dev_datatype_sweep.py` -- the per-candidate driver.
- `tests/build_datatype_sweep_report.py` -- assembles the JSON/CSV/PNGs from
  `runs/`.
- `tests/check_selected_precision_config.py` -- the propagation check.
- `work_log.md` -- chronological narrative, exact commands, and commit SHAs.

## Known limitations (freshness / provenance / style; no open correctness bug)

This section, and the corrections threaded through "What was tested" and
"Pareto interpretation" above, reflect the response to one independent
`$stage-review` round (work log DS-007). That round found three required
(P2) issues -- a selection rule contradicted by this stage's own candidate
table, a Pareto-frontier-membership claim contradicted by the charts it
described, and a circular KV-cache "proof of consumption" -- all fixed
directly (see DS-007 for the full list, including smaller claims-that-were-
false fixes). The items below are the review's remaining findings that this
stage's review budget calls for recording rather than chasing with more
hardware:

- **No replicate runs, no variance estimate.** Every accuracy number in
  this stage comes from one 154-token chat-template prompt at 100 generated
  positions -- the model's established reference, not this stage's
  invention, but still the resolution ceiling on every accuracy conclusion
  drawn here. The 0.850 no-regression gate (see "Accuracy bar") removes the
  need for most per-candidate noise judgment calls, but does not add
  statistical power the underlying reference doesn't have.
- `C09`'s teacher-forced top-1 *improvement* (0.850→0.870) at this sample
  size is not strong evidence either way; "What was tested" explains why it
  doesn't change the selection regardless of which way that specific number
  moves (no real speed benefit either way, and a stronger, already-committed
  long-context signal points the other way).
- `probe/README.md`'s per-expert PCC evidence (cited above for the
  first/last-layer-exception discussion) is from one probed MoE layer with
  512 synthetic random-normal tokens, real checkpoint weights -- a
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
- `router_fidelity` (added this stage) only reaches the decode router
  matmul; prefill routing is architecturally separate (inherited,
  unmodified `_router_scores` via `_moe_prefill`) and stays real
  HiFi4+fp32acc regardless of this knob. Disclosed everywhere this file and
  `selected_precision_config.json` describe the field, but worth restating
  here: a future caller (e.g. a vLLM adapter) that sets `router_fidelity`
  expecting a model-wide effect will silently leave prefill routing
  unaffected.
- One artifact from an earlier point in this stage (before the DS-007 fix
  pass) stamped a `tt/model.py` source hash matching an intermediate,
  never-committed state of the file (a pre-commit `black` reformat that ran
  between generating that artifact and this stage's commit). The review
  round traced it to a benign, purely-cosmetic import reflow with no
  functional difference. All artifacts regenerated during DS-007 carry the
  post-format hash; this note exists so a future provenance audit doesn't
  have to re-discover the same non-issue.
- The ~12x gap between the prior stage's reduced-2-layer-profile device-time
  capture (1708.8 us/step) and the real 47-layer wall-clock decode time
  (21.758 ms/step) is inherited, unexplained in either stage's report, and
  not this stage's artifact to fix -- it means op-share percentages from
  that reduced profile (like the LM head's "51.1%" figure discussed above)
  cannot be used to predict full-model-level gains, which is exactly why
  this stage measured the full-model number directly instead of trusting
  the op-level percentage.
