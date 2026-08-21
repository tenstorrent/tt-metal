# TTI Release Waivers

## r1_gpqa_diamond
- Classification: `issue-waived`.
- Affected row: `r1_gpqa_diamond` in the final7 release report.
- Evidence: `logs/tti_release_ci_nightly_final7.log` lines 123-171.
- Failure mode: lm-eval fails before inference because the Hugging Face dataset `Idavidrein/gpqa` is gated in this environment.
- Local error text: `Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub. Visit the dataset page at https://huggingface.co/datasets/Idavidrein/gpqa to ask for access.`
- Current upstream task evidence:
  - EleutherAI lm-evaluation-harness GPQA README: `https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/README.md` says the dataset is gated and the task requires accepting terms and logging in before running.
  - Hugging Face dataset page: `https://huggingface.co/datasets/Idavidrein/gpqa` says users must agree to share contact information before accessing files/content.
  - Hugging Face gated datasets docs: `https://huggingface.co/docs/hub/en/datasets-gated` describe gated datasets as requiring user access requests and authenticated file downloads.
- Why this is not an autoport failure: the error occurs during dataset loading, before any OpenAI-compatible request is sent to the autoport server for GPQA.
- Required follow-up: rerun `r1_gpqa_diamond` after dataset access is granted.
- Scope: this waiver does not waive `meta_gpqa_cot`. No `meta_gpqa_cot` row was present in the final7 release report.

## Non-Waived Rows
- `leaderboard_ifeval`: no waiver used in final7. The row passed the `ci-nightly` subset with score 89.285714%, ratio 0.9591, tolerance 0.05.
- Benchmarks: no waivers used.
- Spec tests/API conformance: no waivers used.
- `meta_ifeval`: no row present in final7 report and no waiver used.
