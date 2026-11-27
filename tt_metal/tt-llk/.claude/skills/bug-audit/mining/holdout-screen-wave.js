export const meta = {
    name: 'bug-mining-holdout-screen',
    description: 'Screen candidate benchmark cases: was the pre-fix code actually defective in a way an auditor should report?',
    whenToUse: 'After select.py holdout (oversampled). args = {repo, git, cases_file, ids: [case ids]}. Then select.py screened.',
    phases: [{title: 'Screen', detail: 'one cheap agent per candidate holdout case'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'valid', 'kind', 'reason'],
  properties: {
    id: { type: 'string' },
    valid: { type: 'boolean', description: 'true only if an auditor reading the PRE-fix code would have been right to report a defect' },
    kind: { type: 'string', enum: ['real-defect', 'cleanup-or-refactor', 'lint-or-analyzer', 'feature-or-enablement',
                                   'workaround-removal', 'timing-tweak', 'include-or-build-hygiene', 'debuggability', 'other-not-defect'] },
    reason: { type: 'string', description: '1-3 sentences citing the pre-fix line that is (or is not) wrong' },
  },
}

const prompt = (id) => `You are screening one candidate for a bug-audit recall benchmark in ${A.repo}. Read-only.

Find the case whose "id" is "${id}" in ${A.cases_file} (JSON lines). Read its fix with \`git -C ${A.git} show <fix_commit>\`
and read the changed code as it was BEFORE the fix (\`git -C ${A.git} show <fix_commit>^:<path>\`).

Question: would an auditor reading the PRE-fix code have been right to report a defect: a wrong result, crash,
hang, race, leak, broken contract, or compile failure of valid code in a supported configuration?
- valid=false for: cleanups/refactors, lint or static-analyzer appeasement, enabling a deliberately gated feature,
  removing a workaround whose real fix is elsewhere, timing tweaks that make a flaky test pass without naming a
  defect, redundant/include-only changes, debuggability-only changes.
- valid=true when you can point at the pre-fix line that is wrong and say what goes wrong.
Do not use the knowledge of what the fix did to invent a defect the pre-fix code did not have.`

const results = await pipeline(A.ids, (id) =>
  agent(prompt(id), { label: `screen:${id}`, phase: 'Screen', schema: SCHEMA, effort: 'low' }))
const verdicts = results.filter(Boolean)
log(`screened ${verdicts.length}/${A.ids.length}: ${verdicts.filter((v) => v.valid).length} valid`)
return { verdicts, missing: A.ids.filter((id, i) => !results[i]) }
