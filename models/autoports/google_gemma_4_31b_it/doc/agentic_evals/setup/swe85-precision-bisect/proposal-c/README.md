# Candidate C: head BFP8 HiFi2 (proposal only)

Status: not admitted and not run. This folder proposes the first `stage12_swe85_head_bfp8_hifi2` precision arm. It changes only `lm_head` compute fidelity from LoFi to HiFi2 while keeping its weights at BFP8 and preserves max sequence length 262144.

`candidate-cases.json` contains all 36 initial onset fixtures at seeds 42, 43, and 44 (108 exact case/seed trials). It uses one free-running trial per row because the existing runner assigns seeds as `metadata.seed + trial` and has one cap for every trial in a row. The admitted 512-token extensions cover 36 of those case/seed pairs after deduplicating served/full fixture duplicates; all other pairs retain 256 tokens. Extension fixture prompts and chosen continuations were verified identical to the corresponding initial onset fixture. No favorable subset was selected.

Run the `command` in `command.json` only after the candidate is admitted and the pending-controls gate is cleared. Run from the repository root. `branches: []` avoids repeating teacher-forced branches; the panel measures exact seeded free-running continuations. Candidate output is written under this proposal folder.

The estimate is 32.5 minutes at observed served per-case rates, including the 140.7-second load, with a 0.7x–1.5x generation-time planning range (0.39–0.79 h). The input data had 108 baseline continuations; measured decode speed ranged 27.7–34.6 tokens/s/user (median 31.7). The cap-extension increments use each matched served sample's measured decode wall and preserve its prefill/host overhead. This is a diagnostic runtime estimate only.

Before interpreting outputs, require exact runtime precision and context validation plus per-step trace counter checks in `runtime-validation.json`. Expected decode deltas: model replays, sample replays, and readbacks each equal decoded tokens; captures delta is zero. No additional device synchronization or profiler is used.

Runner limitation: `precision_repro_batch` cannot specify arbitrary per-trial seeds or caps in one job. The minimal workaround is this seed-specific job grouping, with the seed stored in the input fixture and exactly one trial per row; no runner change is needed.
