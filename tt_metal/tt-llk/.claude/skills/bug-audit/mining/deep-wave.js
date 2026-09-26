export const meta = {
    name: 'bug-mining-deep',
    description:
        'Deep-read selected historical bugs: root cause, whether the fix was complete (history + current tree), unfixed siblings, and the audit check that would have caught it',
    whenToUse: 'After select_deep.py. args = {repo, git, xgit?, current, classes, experience?, batches: [abs paths]}. Persist with persist_mining.py deep.',
    phases: [{title: 'Deep read', detail: 'one agent per batch of ~4 cases'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args
if (!A.batches) A.batches = Array.from({ length: A.n_batches }, (_, i) => `${A.batch_dir}/deep-${String(i).padStart(4, '0')}.json`)

const SIB = {
  type: 'object',
  additionalProperties: false,
  required: ['location', 'status', 'why'],
  properties: {
    location: { type: 'string', description: 'file:line in the CURRENT tree' },
    status: { type: 'string', enum: ['unfixed', 'fixed', 'not-applicable', 'unsure'] },
    why: { type: 'string' },
  },
}
const CASE = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'is_real_bug', 'primary_class', 'other_classes', 'component', 'root_cause', 'trigger', 'symptom',
             'detected_by', 'fix_summary', 'fix_verdict', 'fix_verdict_evidence', 'siblings', 'audit_check', 'seed', 'public_safe'],
  properties: {
    id: { type: 'string' },
    is_real_bug: { type: 'string', enum: ['yes', 'no', 'unclear'] },
    primary_class: { type: 'string' },
    other_classes: { type: 'array', items: { type: 'string' } },
    component: { type: 'string' },
    root_cause: { type: 'string', description: '1-3 sentences, general terms: what was wrong and why' },
    trigger: { type: 'string', description: 'the concrete configuration/shape/dtype/arch/flag that exposed it' },
    symptom: { type: 'string' },
    detected_by: { type: 'string', enum: ['ci-test', 'nightly-or-model-test', 'user-report', 'code-review', 'audit-or-tool', 'unknown'] },
    fix_summary: { type: 'string' },
    fix_verdict: { type: 'string', enum: ['complete', 'partial', 'reverted', 'reverted-then-relanded', 'superseded', 'wrong-root-cause', 'workaround', 'unknown'] },
    fix_verdict_evidence: { type: 'string', description: 'the later commits, current code, or reasoning that support the verdict' },
    siblings: { type: 'array', items: SIB, description: 'places in the CURRENT tree with the same pattern (other archs, dtypes, overloads, copies)' },
    audit_check: { type: 'string', description: 'the general check a code auditor should run to catch this class of bug, phrased so it applies beyond this case' },
    seed: { type: 'string', description: 'a grep-able pattern or precise description of the code shape to hunt for elsewhere' },
    public_safe: { type: 'boolean', description: 'false if your text relies on internal-only material (RTL signal names, .sv paths, private docs)' },
  },
}
const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['batch', 'cases'],
  properties: { batch: { type: 'string' }, cases: { type: 'array', items: CASE } },
}

const prompt = (batch) => `You are studying past bugs in ${A.repo} so a future code audit knows what to hunt for. Read-only:
do NOT edit any file, do NOT push, comment or file anything.

Read ${batch}: a few fix cases. Each has the issue/PR text, the fix commits, and later history (reverts, later commits
citing it, later fix-like commits to the same files), plus a first-pass triage.

For EACH case:
1. Understand the bug. Read the fix: \`git -C ${A.git} show <oid>\`; \`gh issue view\` / \`gh pr view --comments\` (repo
   ${A.repo}) as needed. Find the root cause, not the symptom.
2. Judge the FIX, don't trust it. A merged fix may be incomplete. Read the later history listed in the case (and
   \`git -C ${A.git} log --follow -p\` on the fixed lines if needed): was it reverted, re-landed, re-fixed, or
   superseded?${A.xgit ? ` The code may have moved: later history can also be in ${A.xgit} (search its log for the
   issue/PR number, the fix subject, and the fixed file names).` : ''} Then read the SAME code in the current tree at ${A.current} (read-only) and decide whether the fixed
   invariant still holds there.
3. Look for SIBLINGS in the current tree: the same pattern in the other arch/dtype/layout variants, overloads, copies,
   and callers. Record each with its status. A real unfixed sibling is the most valuable thing you can find.
4. Write the audit_check: how an auditor who has never seen this bug would catch this class of defect in other code.
${A.experience ? `5. The auditor's own prior findings are indexed at ${A.experience}. If a case touches an area listed there, read
   that note: it may already say whether a fix was complete. Keep internal-only details out of your text (set
   public_safe=false if you cannot).` : ''}

Classes: use the ids from ${A.classes.join(', ')} EXACTLY as written (hyphens included); invent a kebab-case id only
when none fits. Keep every text field in general engineering terms. Do not cite issue or PR numbers in audit_check or seed.
Return one entry per case, same ids.`

const results = await pipeline(A.batches, (b) =>
  agent(prompt(b), { label: `deep:${b.split('/').pop()}`, phase: 'Deep read', schema: SCHEMA })
    .then((r) => (r ? { ...r, batch: b } : null)))
const good = results.filter(Boolean)
log(`deep-read ${good.reduce((s, r) => s + r.cases.length, 0)} cases in ${good.length}/${A.batches.length} batches`)
return { results: good, missing: A.batches.filter((b) => !good.find((r) => r.batch === b)) }
