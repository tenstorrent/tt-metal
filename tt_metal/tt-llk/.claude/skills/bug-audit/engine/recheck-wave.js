export const meta = {
    name: 'bug-audit-recheck',
    description: 'Second look at unsettled and sampled-refuted bug-audit candidates: 3 fresh verifiers each, prior reasoning shown',
    whenToUse: 'After recheck.py queue. args = {run, root, items}. Persist with recheck.py persist.',
    phases: [{title: 'Recheck', detail: 'three independent verifiers per candidate, each shown the earlier verdicts'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args
const ROOT = A.root

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'reason'],
  properties: {
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'uncertain'] },
    reason: { type: 'string' },
  },
}

const LENSES = [
  'reachability — trace a concrete caller/config from a public entry point to the site, or prove none exists',
  'semantics — what the code actually computes vs what its callers, siblings and declared contract expect',
  'history — git log/blame the site: was it written this way on purpose, changed by a fix, or copied from a sibling that differs?',
]

const prompt = (it, lens) => `Settle whether this claimed bug in the tree at ${ROOT} is REAL. It was previously judged
"${it.why}", so the earlier verifiers either disagreed, died, or it is a random re-examination of a refutation.

File: ${it.finding.file}:${it.finding.line}   Category: ${it.finding.category}   Severity claimed: ${it.finding.severity}
Claim: ${it.finding.summary}
Failure scenario: ${it.finding.failure_scenario}
Evidence offered: ${it.finding.evidence}

Earlier verdicts (do not defer to them; check each decisive claim they make against the code):
${(it.prior_reasons || []).map((r) => `- ${r.slice(0, 700)}`).join('\n') || '- none recorded'}

Reachability of library code: a public API, LLK or header-library entry point is reachable by default, even when this
tree has no caller, because its callers may live in another repository. Refute on reachability only if every legal
call is guarded or rejected.

Lens: ${lens}. Read the code yourself. Answer "confirmed" only if the defect is real and reachable, "refuted" only
with the concrete line that makes it safe or unreachable, "uncertain" if the code cannot settle it — and then say
what experiment or owner question would.`

const results = await pipeline(A.items, (it) =>
  parallel(LENSES.map((lens) => () =>
    agent(prompt(it, lens), { label: `recheck:${it.finding.file.split('/').pop()}:${it.finding.line}`, phase: 'Recheck', schema: VERDICT_SCHEMA })))
    .then((vs) => {
      const got = vs.filter(Boolean)
      const n = (v) => got.filter((x) => x.verdict === v).length
      const c = n('confirmed'), r = n('refuted')
      const outcome = c >= 2 ? 'confirmed' : r >= 2 ? 'refuted' : 'uncertain'
      return { finding: it.finding, why: it.why, outcome,
               votes: { confirmed: c, refuted: r, uncertain: n('uncertain'), died: LENSES.length - got.length },
               reasons: got.map((v) => `[${v.verdict}] ${v.reason}`) }
    }))

const items = results.filter(Boolean)
const flips = items.filter((i) => i.why === 'refuted-sample' && i.outcome === 'confirmed').length
log(`rechecked ${items.length}: confirmed ${items.filter((i) => i.outcome === 'confirmed').length}, refuted ${items.filter((i) => i.outcome === 'refuted').length}, uncertain ${items.filter((i) => i.outcome === 'uncertain').length}; refuted-sample reversals ${flips}`)
return { items }
