# GPQA server-sized candidate confirmation

Prepared gpqa10-c8.patch against TTI8a3c949317fdf43ba11086ad90d11a6555046d54 in isolated detached worktree /home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa-server-sized. It changes only this model's GPQA max_concurrent2→8 and the model-specific command test's client expectations. No active publication edit, commit, push or dispatch.

The list stays exactly [152,189,46,87,52,181,31,185,50,187] (10), retaining the complete nightly cohort despite the eight-client server size. Peer suites remain disabled. Actual resolved model + stock build_eval_command confirmed these10 IDs and num_concurrent8; existing fresh-autoport registration provides batch/maxseq8. Native parsers, cache paths, Harbor e76c762f pin, HF revision, context262144, output126976, request14400s and sampling settings are unchanged.

32 focused model/registration/reference tests passed. All full hooks passed (Ruff, format, pytest, copyright), log bringup/artifacts/stage12-gpqa-server-sized-hooks.log. git diff --check passed. Host-only predicted/actual device hours0; release savings not claimed (validity prerequisite).

Eligibility is not established by this preparation: coordinator must first accept corrected nightly10 and demonstrate capacity/admission from actual histories. Do not dispatch merely because host tests pass. No larger/full cohort chosen.

After approval, apply gpqa10-c8.patch to the exact base or reconfirm after rebase; run existing model tests/full hooks before publication. Dispatch stock workflow=evals with immutable model/TTI SHAs and explicit vllm-git-ref=ca7a18246e4111629ed6cb4a7d1192002749fccf. Queue, timing projection and planned_clients8 must be recorded by the coordinator before launch.
