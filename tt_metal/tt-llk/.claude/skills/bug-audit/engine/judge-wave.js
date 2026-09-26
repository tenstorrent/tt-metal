export const meta = {
    name: 'bug-audit-judge',
    description:
        'Semantic benchmark scoring: for each held-out bug, judge which blinded findings describe the defect its fix removed (2 judges, 3rd on disagreement)',
    whenToUse: 'After bench.py judge-inputs. args = {inputs: [abs paths]}. Then bench.py judged --dir <dir> --raw <this output>.',
    phases: [{title: 'Judge', detail: 'two independent judges per case, a tie-breaker per disputed finding'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['case_id', 'defect', 'judgments'],
  properties: {
    case_id: { type: 'string' },
    defect: { type: 'string', description: 'one sentence: the defect the fix removed, in your own words' },
    judgments: {
      type: 'array',
      description: 'one entry per finding in the input, same ids',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'match', 'why'],
        properties: {
          id: { type: 'string' },
          match: { type: 'boolean' },
          why: { type: 'string', description: 'one short sentence' },
        },
      },
    },
  },
}

const rules = `A finding MATCHES only if it identifies the same defect the fix removed: the same mechanism in the same code,
such that fixing what the finding describes would fix this bug. Judge the substance, not the line number: a finding
reported at a call site, a definition or a neighbouring line still matches if it names this defect. A finding does NOT
match merely because it is near the changed lines, in the same function, or in the same general area (for example
another format bug in the same factory). A finding that bundles this defect with other claims matches if this defect
is clearly one of its claims. You may read the pre-fix code in git if the diff alone is ambiguous.`

const judgePrompt = (path, k) => `You are grading a bug-audit benchmark (judge ${k}). Read-only; do not edit anything.

Read ${path}. It holds one real historical bug (its title, a summary, and the fix diff) and a list of findings that
auditors reported for the pre-fix code. Each finding has an opaque id; you do not know who produced it.

First state the defect the fix removed. Then judge EVERY finding.
${rules}`

const tiePrompt = (path, f, a, b) => `You are the tie-breaking judge in a bug-audit benchmark. Read-only.

Read ${path} (a real bug's fix diff and the findings auditors reported). Two judges disagreed about finding ${f}:
- judge 1 said ${a.match ? 'MATCH' : 'NO MATCH'}: ${a.why}
- judge 2 said ${b.match ? 'MATCH' : 'NO MATCH'}: ${b.why}
Decide for finding ${f} only.
${rules}`

const TIE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['match', 'why'],
  properties: { match: { type: 'boolean' }, why: { type: 'string' } },
}

const results = await pipeline(A.inputs, async (path) => {
  const [j1, j2] = await parallel([1, 2].map((k) => () =>
    agent(judgePrompt(path, k), { label: `judge${k}:${path.split('/').pop()}`, phase: 'Judge', schema: SCHEMA })))
  if (!j1 || !j2) return { path, failed: true }
  const m2 = Object.fromEntries(j2.judgments.map((x) => [x.id, x]))
  const matching = []
  let disagreements = 0
  const ties = []
  for (const x of j1.judgments) {
    const y = m2[x.id]
    if (!y) continue
    if (x.match === y.match) { if (x.match) matching.push(x.id) } else { disagreements++; ties.push([x, y]) }
  }
  const tb = await parallel(ties.map(([x, y]) => () =>
    agent(tiePrompt(path, x.id, x, y), { label: `tie:${x.id}`, phase: 'Judge', schema: TIE_SCHEMA }).then((v) => ({ id: x.id, v }))))
  for (const t of tb.filter(Boolean)) if (t.v && t.v.match) matching.push(t.id)
  return { path, case_id: j1.case_id, defect: j1.defect, matching, disagreements,
           n_findings: j1.judgments.length }
})

const good = results.filter((r) => r && !r.failed)
log(`judged ${good.length}/${A.inputs.length} cases; ${good.reduce((s, r) => s + r.disagreements, 0)} disagreements tie-broken`)
return { cases: good, failed: results.filter((r) => !r || r.failed).map((r) => (r ? r.path : null)) }
