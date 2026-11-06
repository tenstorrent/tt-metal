# GPQA full198 candidate

Prepared gpqa198-c2.patch on e7643650635b2ab4e5f9662de26b1ae04433f031 in isolated worktree /home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa-full-preparation. Model config uses stock doc_ids=list(range(198)) and CI_NIGHTLY198; no selector/harness changes. All dataset indices0..197 are included exactly once, independently of scores. This contains the frozen nightly10 as a set; original cohorts.json and nightly list provenance are untouched.

Actual resolved fresh-autoport stock command confirms198uniqueIDs, C2/B8, peer limits0, pinned HF revision842da3794eaa0b77d5f08bae87a17459d91ff475, request14400s/output126976. Native parsers, cache, Harbor pin, sampling and generic Docker fix remain unchanged. The source patch changes only model config and two model test expectations.

32 focused tests and all full hooks passed (Ruff, format, full pytest, copyright), exit0. Log bringup/artifacts/stage12-gpqa-full-hooks.log; git diff --check passed. Device-hour box/actual cost0. No speed savings measured or claimed.

No active publication edits, commits, push or dispatch. Full-rung eligibility requires valid server-sized10 and coordinator's measured projection/admission; preparation alone does not authorize it. Use stock workflow=evals with immutable source SHAs and explicit vllm-git-ref=ca7a18246e4111629ed6cb4a7d1192002749fccf after approval. Candidate smoke-test mode follows stock first3 semantics (indices0,1,2); earlier immutable smoke publication preserves original frozen smoke2 and is not redefined by this candidate.
