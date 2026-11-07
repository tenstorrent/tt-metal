Host-only final normal-rung audit plan for C4/80 run42323448662.

Goal: establish the final frozen80 score and trajectory validity, then own the once-only same80 comparison against completed C8 run42332785769. No device hours, source changes, polling, dispatch or central-file edits. Start analysis only after coordinator confirms terminal success/failure and final collection. Do not modify the completed C8 RESULT.

Inputs: this arm's manifest and startup confirmation; final collected official results/samples and model-owned phase archive; terminal workflow metadata/complete private console; completed runs/gpqa80-c8-saturation-comparison/RESULT.json and its private chain-analysis/prompt-metadata/native metadata; accepted earlier C4/40 and C8/40 audits for nested provenance.

Prepared private analyzer: bringup/artifacts/gpqa80-c4-final-audit/analyze.py, adapted from the completed C8 analyzer, syntax checked only. It reconstructs all physical chains using positions/remaps, tracks continuity ambiguity and retention loss, preserves capped IDs privately, and computes pinned-tokenizer prompt lengths/hashes. It is not run on partial artifacts. Retained decode excludes the first prefill sample, host-sampled tokens and client request identity.

Final checks:

- Verify actual immutable TT/TTI/plugin/image, full60 model, pinned HF/tokenizer, B8/C4/default pool, caps126976/14400, sampling and stock route against manifest. Distinguish configured metadata from observed runtime proof.
- Verify exact frozen80 membership and official sample counts. Bind semantic dataset indices through ascending membership and document/prompt hashes, never the unsorted logged labels or physical row order. Validate nested-known prompt identity before output comparisons.
- Require contiguous unique submit/complete steps, no archive loss/exhaustion/pending errors, 80 coherent chains. Bind collisions using unique prompt lengths, matching accepted chain hashes or complete native usage+visible hashes. Use complete-set elimination only when unique and documented; leave ambiguous cases unknown.
- Compute official score and Wilson95 with all failures. Inspect every cap-consistent empty output's retained suffix privately for exact loops or degenerate behavior; do not infer cap from empty content alone, discard failures, or fabricate missing first token/native usage. Newly productive/unclassified tails remain a validity question.
- Compare every semantic prompt, visible answer, score, retained chain hash/length and aggregate active work to C8/80. Count changed work explicitly. Report elapsed differences descriptively with source/settings and scheduler scope; equal work/output is stronger evidence but does not establish all causality.
- Record evaluator/device/workflow/queue times separately, summed overlapping agent hours, longest full-cap chain, terminal tail and occupancy. Completion-boundary intervals are host elapsed, never profiled device time. Native outer CI timestamps may be replay; use envelope timestamps if available.

Output once at completion: audit.json and AUDIT.md in this arm, including valid/blocked disposition and exact comparison limitations. Raw gated prompts, solutions, reasoning, IDs and logs remain private. No final evidence written during preparation.
