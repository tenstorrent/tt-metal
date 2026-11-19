# Coarse matrix coverage and final-report audit

CPU-only, 2026-11-16. Read the installed datatype-sweep skill and precision-reasoning reference, current 24 validated policy JSON files, candidate/queue/report harnesses and capability decisions. No TTNN import, device execution or harness/runtime changes. Policy existence is planned coverage, not an evaluated result. The running initial P1 job remains provisional.

## The planned 18-policy coarse sweep

Excluding the inherited baseline and five capacity/reference controls (`e8_two_hifi2`, `e8_two_hifi4`, `e8_three_hifi2`, `e8_three_hifi4`, `e8_four_hifi2`), the current files define these 18 policies:

| Policies | Material experiment | Coverage assessment |
| --- | --- | --- |
| `e4_hifi2` | All required-to-fit BFP4 experts, HiFi2 | Pairs with measured inherited BFP4/LoFi. |
| `e8_edges_lofi`, `e8_edges_hifi2` | BFP8 experts at layers 0/47 with LoFi versus HiFi2 there; other experts retain BFP4/LoFi | Direct BFP8 same-dtype fidelity comparison and canonical edge protection. Distinct from `e8_two_hifi2`, which applies HiFi2 to **all** experts. |
| `proj8_lofi`, `proj8_hifi2` | Existing BFP8 shared gate/up, shared-down, QSA QKV and QSA output | Direct BFP8 fidelity pair for four dominant projection roles, compared with inherited HiFi4. |
| `shared4_lofi`, `shared4_hifi2` | Shared gate/up and shared-down BFP4 on layers 1–46; layers 0/47 retain BFP8/HiFi4 | Both material BFP4 groups have LoFi and HiFi2 coverage. |
| `qsa4_lofi`, `qsa4_hifi2` | QSA QKV/output BFP4 on QSA layers 3–43; layer 47 remains BFP8/HiFi4 | Both material BFP4 groups have LoFi and HiFi2 coverage. Layer 0 has no QSA projection. |
| `gdn_zab8_lofi`, `gdn_zab8_hifi2` | GDN Z/A/B BF16→BFP8, paired fidelity | Covers a likely projection bandwidth win without changing recurrent arithmetic. |
| `gr8_lofi`, `gr8_hifi2` | GR BF16→BFP8, paired fidelity | Covers GR bandwidth/fidelity; fixed exact hyperconnection paths remain separate. |
| `head8_lofi`, `head8_hifi2` | LM head BF16→BFP8, paired fidelity | Covers the full-vocabulary projection win. |
| `activation8`, `ccl8`, `kv8` | Independent BFP8 switches | Activation means DRAM projection inputs; residual storage remains BF16. KV8 already has an independent full-model B32 PCC rejection, which must survive any main-reference pass. |

All 24 policy files passed the current pure-CPU validator. Resolving every layer confirmed explicit LoFi and HiFi2 for every BFP4 role seriously considered: fused expert gate/up/down, shared gate/up, shared-down, QSA QKV and QSA output. There is **no missing mandatory BFP4+LoFi policy in this planned matrix**. Actual full-model execution or exact runtime blocker plus AutoFix evidence is still required for each planned material LoFi trial; file existence alone does not satisfy the gate.

The fused routed expert operation requires equal gate/up/down weight dtypes and shares one fidelity. It is correct to treat those three roles as one indivisible experimental group here. Shared and QSA roles can be restored independently. Router/top-k/weighting precision is intentionally excluded, as the skill recommends. There is no need to invent BFP4 GR/head/GDN candidates merely to increase the matrix; those groups have not been seriously proposed at BFP4 and their BFP8 pairs already cover plausible wins.

The closest available model policies are inherited optimized performance precision and the edge-protected, higher-fidelity precision controls. Do not label either an upstream vendor canonical policy without an actual source. The usual all-BFP8 canonical starting point is physically ruled out: experts plus PLE alone exceed device DRAM before KV. Retain the exact capacity rejection and highest-fitting mixed reference as the documented model-specific substitute; do not claim the sweep started from an actually executed all-BFP8 model.

## Conditional additions and exact policy changes

No unconditional extra coarse candidate is needed before these 18 run. Important follow-ups depend on outcomes:

1. **If inherited expert LoFi fails P1**, the 18 experiments based on it are not evidence that the same projection change is unsafe on a passing expert policy. Rebase each promising nonexpert change onto the fastest expert policy that passes P1. For example, copy `e8_two_hifi4`, set `config_id="e8_two_hifi4_head8_lofi"`, then only `weight_groups.lm_head="bfloat8_b"` and `compute_fidelities.lm_head="LoFi"`. Retain expert layer exceptions and global HiFi4. Do not merge configs by overwriting the whole `layer_exceptions` map; merge per-layer weight/fidelity entries.
2. **If a combined shared BFP4 policy fails accuracy/P1**, restore down first: copy `shared4_lofi`, change `config_id="shared_gateup4_lofi_down8_hifi4"`, set global `weight_groups.shared_down="bfloat8_b"` and `compute_fidelities.shared_down="HiFi4"`; retain edge exceptions. If this becomes a selectable BFP4 policy, compare a companion with `compute_fidelities.shared="HiFi2"` on the same weights. Existing pair supplies the mandatory LoFi trial, but a winning restoration deserves its own direct comparison.
3. **If combined QSA BFP4 fails**, restore QKV first: copy `qsa4_lofi`, set `config_id="qsa_out4_lofi_qkv8_hifi4"`, `weight_groups.qsa_qkv="bfloat8_b"`, `compute_fidelities.qsa_qkv="HiFi4"`, keeping the layer-47 exceptions. Add the otherwise-identical `qsa_out=HiFi2` companion if kept. Restoring output instead is the next isolation experiment if evidence points there.
4. **If `proj8_*` fails**, isolate shared versus QSA instead of declaring all four roles rejected. Copy inherited (or the passing expert base), change only `compute_fidelities.shared/shared_down` for one pair or only `qsa_qkv/qsa_out` for another pair, with LoFi and HiFi2 siblings. Keep weight dtypes unchanged.
5. **If multiple independent switches win**, measure a combined policy and its P1/capabilities before concluding the fastest evaluated choice. Example: copy the passing expert base, apply winning head dtype/fidelity, then winning activation or CCL field. Interactions cannot be inferred from individual wins. KV8 is excluded while its independent PCC gate fails.

`e8_two_hifi2` is worth retaining in the evaluated control set: against `e8_two_hifi4` it tests fidelity without a weight-memory change; against `e8_edges_hifi2` it isolates inner BFP4 expert fidelity. The existing files provide this, so it is not a missing policy.

## P1 admission and selection

Offer only full-advertised-context fitting policies to P1. `e8_three_hifi2` is numerically scored but fails retained-trace capacity; four-upgrade variants fail earlier. Their precision does not qualify them as the production full-context reference. `e8_two_hifi4` is the highest-precision **tested fitting** control currently available, not proof of a global optimum over every possible layer assignment.

A selected BFP4 policy must pass P1 relative to the fitting reference, or carry the runner-recognized required-to-fit limitation backed by actual physical evidence and measured loss. A required-to-fit waiver is not a blanket permission to ignore newly introduced shared/QSA BFP4 loss when their BFP8 versions fit. Do not turn a P1 loss into a pass by calling every BFP4 role necessary for fit.

Selection of the reference itself additionally needs its HF or higher-precision comparison; self-comparison is insufficient. Preserve the >=1000 paired-position gate and default noise bounds. Current two-text generation has 3435 answer positions; the eventual scored/gate result, not that text count alone, establishes P1 acceptance. Run the installed runner with the final shipped `--selected` ID as the last P1 invocation and retain previous results separately.

P1 scheduling can safely avoid strictly slower policies after a faster candidate passes all gates. Score candidates in descending valid traced teacher-forcing speed, allowing for the repeated-baseline noise band. Every unresolved candidate that could beat the chosen one must receive P1 or an independently evidenced execution/accuracy/capability rejection. A slower unscored candidate is **not** a P1 failure. Inside the noise band, it may still be the simpler/safer tie winner, so do not prune it solely on a tiny speed difference.

## Report schema gap: unscored P1 is currently conflated with failure

`tests/report_datatype_sweep.py` accepts per-config arbitrary P1 dictionaries, but missing entries default to `{"passed": false, "reason": "not scored"}` and `acceptance_status` treats false/missing/null as a failed P1 gate. Therefore **the current final CLI cannot honestly represent an unscored P1 candidate without labeling it a failure**. Encoding `passed=true` to bypass this would fabricate a pass. Setting `passed=null` preserves raw uncertainty but still produces `failed_gates=["p1"]` and `status="fail"`; documentation alone does not fix that contradiction.

Proposed minimal report-only change, not applied in this audit:

```json
{
  "slower_candidate": {
    "passed": null,
    "status": "not_scored_dominated",
    "reason": "Valid traced speed is below the accepted faster candidate by more than baseline spread; not offered to P1",
    "dominated_by": "actual_selected_id",
    "evidence": ["slower_candidate_v1.json", "selected_candidate_v1.json", "actual_p1_result.json"]
  }
}
```

Treat P1 as tri-state: true passes, false means a measured/runner gate failure, null/missing means unscored. Add `pending_gates` (including `p1` for unknown), keep `failed_gates` for actual false outcomes, and use `status="not_selected_p1_unscored"` when execution, accuracy, trace and capability pass but P1 was deliberately not scored. `eligible` must be true only when every required gate is explicitly true or an explicitly documented accepted waiver applies. Selected must be eligible. Require a cited faster eligible comparator and a speed gap greater than baseline spread before accepting the dominated-pruning reason; otherwise use `pending_p1` and leave selection incomplete. Plot unscored points with a distinct label/legend rather than describing them as measured failures.

For a runner-accepted required-to-fit limitation, retain the actual numerical verdict and waiver separately, for example `numerical_passed=false`, `gate_passed=true`, `status="pass_with_required_fit_limitation"`, plus runner result and limitations artifact. Report selection should consume the actual effective runner gate, not overwrite numerical failure. If the legacy `passed` field is retained as effective acceptance, explicitly document that meaning and keep both subordinate fields.

## Capability decisions and exact final CLI

`--capability-decisions` is required. Each evaluated config currently needs boolean `passed`, nonempty `reason`, nonempty `evidence`, and optionally `checks` with artifact paths and dotted **boolean** fields. Paths resolve relative to the decisions file. A missing entry becomes `status="missing"`, `passed=false`; the report rejects eligibility and labels it a capability failure even when the real state is not evaluated. Populate real decisions rather than letting absence masquerade as a measured rejection. A parallel unknown/pending status would make this distinction cleaner if a candidate's capability was intentionally not investigated after another decisive failure.

For ordinary BF16-KV candidates, a capability pass may cite their own completed full262144 construction, six retained prefill buckets, 31/32/33 prompts and sampling transitions from `run_datatype_candidate`, plus unchanged context/layout assumptions. Do not use accuracy `raw.passed` as the capability check: numerical failure and capability completion are distinct. `completed=true` is usable here because this harness sets it only after the capability sequence, but also cite the trace inventory and retained bucket evidence in the reason. Weight upgrades need their own fit evidence. Every KV policy must cite its recomputed context contract and diagnostic gates; `kv8` remains explicitly false even if full reference/P1 scores pass. Add selected-only long-context/resume/qualitative evidence during finalization. Existing decisions cover inherited, KV8, two-HiFi4, three-HiFi2 and four-HiFi2; the other coarse/control configs still need entries as results arrive.

The report currently requires these exact arguments (run from checkout root):

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.report_datatype_sweep \
  --selected ACTUAL_SELECTED_CONFIG_ID \
  --p1-decisions models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/p1_decisions.json \
  --capability-decisions models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/capability_decisions.json \
  --inputs models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/inherited_e4_lofi_v3.json \
    models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/e8_two_hifi4_v1.json \
    OTHER_EXPLICIT_CANDIDATE_RESULT_PATHS
```

Use an explicit result list; broad JSON globs include policies, provenance and diagnostic records with different schemas. Use the repeated warmed baseline `inherited_e4_lofi_v3`, not the cold-capture v1 or failed v2. Prefer one authoritative result per config; current reporter keys decisions by config ID and does not consolidate duplicate attempts. Preserve failed retries as referenced rejection evidence (as already done for four-upgrade fixed-region attempts). Three-upgrade valid traced points remain plotted with trace-capacity rejection. Rows without valid traced speed remain in JSON/CSV with null performance, not invented zero-speed chart points. Final selected JSON, plots, summaries and P1 IDs must agree.
