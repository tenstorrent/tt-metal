# Harness registration effort

Goal: register the new source-only autoport on stock-main TTI with the stage-11 serving recipe and benchmark matrix. No device jobs; device-hour box 0.1.

1. Inspect generic model catalog, serving configuration and benchmark resolution paths.
2. Add the model-specific entry and host assertions for exact adapter overrides, revision, context, batches and sampling.
3. Run focused host checks and identify the CI dependency-stack gap without claiming live validation.

Confirmation: resolved P300X2 spec points at this autoport, revision 842da3794eaa0b77d5f08bae87a17459d91ff475, 262144 context, batches 1/8/16/32, top_k=32 and 128/128/C1 eight-prompt benchmark. Preserve stock plugin runtime. Suite and cohort selection belong to coordinator.

Coordinator smoke sizing: planned_clients=2 per suite selects batch8, the smallest validated batch at or above2. Initial catalog max_concurrency/max_num_seqs are8; this is not a measured fastest claim. The benchmark matrix remains128/128/C1 prompts8, with batch1 chosen separately for the eventual release benchmark. Repeated focused host checks:110 passed.
