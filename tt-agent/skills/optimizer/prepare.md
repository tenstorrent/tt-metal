# Prepare phase

Runs once per session, before the first baseline profile. Produces a short
note in `~/.tt-agent/notes/` that the Baseline phase reads.

## Steps

1. **Research the target op.** Invoke `tt:learn("<target> — config knobs,
   valid dim constraints, kernel variants")`. Record the note path in the
   trend file. The first iteration's hypothesis queue depends on this.

2. **Sibling-implementation scan.** If the target is "reused from X with
   changes" (very common in tt-metal — most multimodal and model variants
   extend a Llama reference), read X and diff against the target. The
   original's config patterns may not apply to the new model's dimensions.
   Note any divergent dims, config lambdas, or cached `args.*` fields.

   **Check the sibling's mode branch** before borrowing structural choices.
   tt-metal models often branch on `mode=="prefill" | "decode"` with very
   different memory layouts. `grep` the sibling for the mode keyword and
   trace which progcfg the target's shape would select. See `playbook.md`
   § "Verify a sibling's mode before borrowing structural choices".

3. **Actual-vs-derived dim check.** For every `args.<dim>` used in a
   progcfg, compare to `self.<weight>.shape[-1]` of the loaded tensor.
   `ModelArgs` fields are computed at config time and can truncate
   (int-div) or drift from actual weight shape. Prefer the weight-shape
   value in progcfgs. If a mismatch exists, flag it in the trend file —
   a standalone fix PR may be warranted.

4. **Authoritative docs sweep.** Ask the developer whether internal guides
   exist (Confluence pages, `tech_reports/`, PDF exports in the workspace).
   For matmul the PSE Matmul Configuration Guide governs variant selection,
   L1 budgeting, and subblock rules — reading it before iterating reshapes
   the hypothesis queue substantially.

5. **Playbook skim.** Skim `playbook.md`. Entries there cost prior
   sessions full iterations to discover.

## Output

A short Prepare note in `~/.tt-agent/notes/`:

- Research pointer (path to `tt:learn` note)
- Sibling diff highlights (if reused)
- Dim-validation findings (any actual-vs-derived mismatch)
- Doc pointers (internal guides, PDFs)
- Mode-branch confirmation (if sibling has PREFILL/DECODE split)

The Baseline phase consumes this.
