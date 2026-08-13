# IFEval reference-variant analysis (operator-supplied evidence)

Added by the run operator between the blocked turn and the resume turn, from the
release run's own saved `results_*.json`. It records measurements and the two
readings they permit; it does not re-grade the row.

## What the release report graded

`release_report_corrected_fail.md` scores `ifeval` at **18.67** against
**34.3**, ratio **0.5443**, FAIL at tolerance 0.05.

The 34.3 is the Falcon3-7B-Base model card
(`https://huggingface.co/tiiuae/Falcon3-7B-Base#benchmarks`). In
`reference_config/evals/eval_config.py` this run set
`gpu_reference_score=None` / `gpu_reference_score_ref=None`, so the acceptance
code fell back to the published score, per its own note: "If a value GPU
reference is not available, the accuracy check is based on the direct ratio to
the published score."

## What was actually measured

Full unrestricted run: 541/541 samples, `num_fewshot=0`, `limit=None`,
`apply_chat_template=False`. All four IFEval metrics were recorded:

| metric | score | ratio vs 34.3 |
|---|---:|---:|
| `prompt_level_strict_acc` (graded) | 18.67 | 0.5443 |
| `prompt_level_loose_acc` | 19.96 | 0.5819 |
| `inst_level_strict_acc` | 30.94 | 0.9020 |
| `inst_level_loose_acc` | 32.49 | **0.9472** |

Stderr is 1.68 / 1.72 on the two prompt-level metrics.

## The ambiguity

IFEval has four metrics and the model card publishes one number, **34.3, with
no variant stated** (verified by reading the card: the row is
`IFEval | 12.0 | 30.6 | 33.9 | - | 34.3`, and the card says only that
lm-evaluation-harness raw scores are reported). The graded metric is the
harshest of the four; 34.3 is closest to `inst_level_loose_acc` (32.49). The
mean of all four is 25.5, which does not match 34.3 either.

Two readings are consistent with this evidence:

1. **Variant mismatch.** The card's 34.3 is an instruction-level figure, so the
   like-for-like comparison is 32.49 vs 34.3 = 0.947 — near parity, and the
   0.544 is an artifact of grading a different metric than the reference
   represents. Under this reading the port has no IFEval regression.
2. **Genuine shortfall.** The card's 34.3 is prompt-level, and the port really
   is at 0.544 of it.

**Nothing in the current evidence distinguishes these**, because the reference
is an unsourced third-party number rather than a same-command control. That is
the same defect the acceptance note warns about and is why the previous turn
refused to waive the row.

## What would settle it, and what must not be done

**Do not simply repoint `score_func_kwargs` to whichever metric passes.**
Re-grading against the same unsourced number to convert FAIL into PASS is the
same failure mode as the EXPERIMENTAL implicit-eval-waiver bug this stage found
and fixed in `report_module/acceptance_criteria.py`. Note also that even the
most favourable variant is **0.947 against a 0.95 threshold**, so it does not
pass on the merits.

The sound fix is to **source the reference**: run the identical `lm_eval ifeval`
command (0-shot, `apply_chat_template=False`, same task version) against
HuggingFace `tiiuae/Falcon3-7B-Base` and populate `gpu_reference_score` /
`gpu_reference_score_ref` from that run. No GPU is required — CPU is acceptable
for a reference, and under `--limit-samples-mode ci-nightly` the sample count is
tractable; label any subset result as such. If that control reproduces ~18-19
prompt-level-strict, reading 1 is confirmed and the row should be graded against
a properly sourced reference. If it reproduces ~34, reading 2 holds and this is
a real port regression to fix.

## Related trap on the other blocked row

The GPQA row grades `gpqa_diamond_generative_n_shot` against the card's
**GPQA (0-shot) = 35.5**. That is a different task configuration from the graded
one, so once the dataset is reachable this row can fail for the same
reference-mismatch reason rather than for model quality. Check the shot count
and task variant before treating a GPQA ratio as a quality signal.

## Dataset access status

The gated dataset blocker is cleared: `Idavidrein/gpqa` now downloads with the
active token (`gpqa_diamond.csv` 1,373,492 bytes) and
`load_dataset("Idavidrein/gpqa", "gpqa_diamond")` yields 198 rows. The previous
turn's `GatedRepoError 403` / "not in the authorized list" no longer reproduces.
