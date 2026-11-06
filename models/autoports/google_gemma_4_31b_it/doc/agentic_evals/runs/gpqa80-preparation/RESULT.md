# GPQA80 preparation: held

Local source-only candidate `6ca5a1b89824bfcf1a197a1fcdc86f3e1d44fd8b` on `prep/gpqa80-retained`, based on `6efe6739e6513d0edbf49a8ef39527b80a23784d`. The frozen SHA256 seed rule reproduces80 unique indices with exactly the prior40 as its prefix; `cohort.json` records IDs, pinned harness source and the semantic-index versus logged-label mapping before scoring.

Production AST differs only in the model-specific GPQA document list40→80 and nightly limit40→80. Two tests track those changes. C4/B8, release limits, context, sampling, seed, pins, retention and disabled peer suites are unchanged. No runtime source changes.

Validation:34 focused host tests passed (3.42s); full ruff, formatting, pytest and copyright hooks passed after formatting. Exact AST audit and diff whitespace check passed. Device forecast/actual0h, host box0.2h. No scores or timing measurements were produced.

Candidate remains held pending valid40 and measured schedule admission; projection is unknown until that evidence exists. No push, dispatch or central-document edits. This is preparation only, not a rung result.
