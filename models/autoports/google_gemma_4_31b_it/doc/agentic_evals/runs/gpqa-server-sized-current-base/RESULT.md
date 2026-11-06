# Current-base GPQA10/C8 candidate

Prepared gpqa10-c8.patch against2c4a2628b974acc7ebc7a4fa867db68f0b108019 in detached worktree /home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa-c8-current. Supersedes the earlier8a3 candidate for application to this publication state. GPQA CI_NIGHTLY0→10, max_concurrent2→8; Terminal2→0; SWE remains0. Exact frozen IDs remain [152,189,46,87,52,181,31,185,50,187]. Two model-specific expectation test files updated.

Actual resolved fresh-autoport stock command confirms GPQA10/C8, serverB8/maxseq8 and disabled peers. Only model eval configuration and focused tests differ; generic Docker lifecycle repair, native parsers, cache/HF/Harbor pins, sampling and release limits remain base-identical.

32 focused tests and all full hooks passed (Ruff, format, full pytest, copyright), exit0. Log bringup/artifacts/stage12-gpqa-c8-current-hooks.log. git diff --check passed. Predicted/actual device hours0; validity unblocks GPQA; no speed saving claimed.

No active publication edits, commits, push or dispatch. Actual nightly validity and measured capacity/admission remain pending coordinator decisions; host confirmation does not authorize advancement. After eligibility, apply to the exact base or reconfirm after rebase. Stock dispatch workflow=evals must pin immutable model/TTI SHAs and vllm-git-ref=ca7a18246e4111629ed6cb4a7d1192002749fccf.
