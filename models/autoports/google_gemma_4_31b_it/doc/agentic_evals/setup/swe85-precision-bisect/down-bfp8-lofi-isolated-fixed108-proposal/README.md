# Isolated down-BFP8 LoFi fixed108 candidate

This is an unadmitted candidate for the 108-row full-mode diagnostic panel. Its precision delta remains only `down`: BFP4 LoFi to BFP8 LoFi. The served `lm_head` remains BFP8 LoFi, all other policy fields stay pinned, and the fixed108 manifest is unchanged.

The launch gate requires terminal P2 success, then validates three completed SK evidence arms: corrected selected-BFP8 served suffix512, the original full-BF16 suffix512 arm, and matched selected-BFP8 served/full suffix1024. It checks each admission, launch, completion, command, manifest, all three result hashes, root output review, independent triage, and exact complete `runtime_precision` object. Their hashes are also pinned in `source-hashes.json`.

Root must then provide a `sklearn-serial-review.json` bound to the current SK evidence map, plus a separate `hf-followup-decision.json`. A `completed` decision must reference a hash-verified HF package with admitted candidate, successful completions, complete outputs, and accepted root output review. A `not_required` decision must give a reason and bind that decision to the exact SK evidence map. A bare status enum is insufficient.

The H9 serialized HF follow-up is an independent global serialization gate. It must reach `all_groups_completed`; a root terminal review must bind the actual H9 admission, progress, every one of the eight group launches and successful completions, every result output, and the current SK evidence map. The runtime gate also refuses an active HF process and checks device-owner/readiness preflight.

No candidate admission or launch is present. `STOP_BEFORE_LAUNCH` remains. `--validate-only` and `--self-test` are host-only checks; do not use `--execute` before root has supplied the exact admission and terminal evidence records.
