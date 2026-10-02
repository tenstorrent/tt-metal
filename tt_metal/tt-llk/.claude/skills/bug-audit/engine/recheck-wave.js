export const meta = {
    name: 'bug-audit-recheck',
    description: 'Second look at unsettled and sampled-refuted bug-audit candidates: 3 fresh verifiers each, prior reasoning shown',
    whenToUse: 'After recheck.py queue. args = {run, root, items}. Persist with recheck.py persist.',
    phases: [{title: 'Recheck', detail: 'three independent verifiers per candidate, each shown the earlier verdicts'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args
const ROOT = A.root
// items inline, or (for large waves) one file per item: {items_dir, n} as written by recheck.py queue --to-dir
const ITEMS = A.items || Array.from({ length: A.n }, (_, i) => ({ path: `${A.items_dir}/c${String(i).padStart(4, '0')}.json` }))
if (!ITEMS.length) throw new Error('no items: pass items, or items_dir + n')
if (ITEMS.length * 3 > 990) throw new Error(`${ITEMS.length} items x 3 verifiers exceeds the 1000-agent cap; split with --max 330`)

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

const itemText = (it) => it.path
  ? `Read the item file ${it.path}: its "finding" (file, line, category, severity, summary, failure_scenario, evidence),
"why" it is being re-examined, and "prior_reasons" (earlier verdicts or analysis: do not defer to them; check each decisive
claim against the code).`
  : `It was previously judged "${it.why}".

File: ${it.finding.file}:${it.finding.line}   Category: ${it.finding.category}   Severity claimed: ${it.finding.severity}
Claim: ${it.finding.summary}
Failure scenario: ${it.finding.failure_scenario}
Evidence offered: ${it.finding.evidence}

Earlier verdicts (do not defer to them; check each decisive claim they make against the code):
${(it.prior_reasons || []).map((r) => `- ${r.slice(0, 700)}`).join('\n') || '- none recorded'}`

const prompt = (it, lens) => `Settle whether this claimed bug in the tree at ${ROOT} is REAL. The earlier verifiers either
disagreed or died, it is a random re-examination of a refutation, or it is an unverified lead.
${itemText(it)}

Reachability of library code: a public API, LLK or header-library entry point is reachable by default, even when this
tree has no caller, because its callers may live in another repository. Refute on reachability only if every legal
call is guarded or rejected.

Lens: ${lens}. Read the code yourself. Answer "confirmed" only if the defect is real and reachable, "refuted" only
with the concrete line that makes it safe or unreachable, "uncertain" if the code cannot settle it — and then say
what experiment or owner question would.`

const results = await pipeline(ITEMS, (it) =>
  parallel(LENSES.map((lens) => () =>
    agent(prompt(it, lens), { label: it.path ? `recheck:${it.path.split('/').pop()}` : `recheck:${it.finding.file.split('/').pop()}:${it.finding.line}`, phase: 'Recheck', schema: VERDICT_SCHEMA })))
    .then((vs) => {
      const got = vs.filter(Boolean)
      const n = (v) => got.filter((x) => x.verdict === v).length
      const c = n('confirmed'), r = n('refuted')
      const died = LENSES.length - got.length
      // as in a wave, a dead verifier is unknown, never a vote: short of a confirmation, the item stays queued for the
      // next recheck wave instead of being settled on the votes that happened to be cast
      let outcome = c >= 2 ? 'confirmed' : r >= 2 ? 'refuted' : 'uncertain'
      if (died > 0 && outcome !== 'confirmed') outcome = 'queued'
      return { finding: it.finding, why: it.why, path: it.path, outcome,
               votes: { confirmed: c, refuted: r, uncertain: n('uncertain'), died },
               reasons: got.map((v) => `[${v.verdict}] ${v.reason}`) }
    }))

const items = results.filter(Boolean)
const flips = items.filter((i) => i.why === 'refuted-sample' && i.outcome === 'confirmed').length  // file-mode items report why=undefined
log(`rechecked ${items.length}: confirmed ${items.filter((i) => i.outcome === 'confirmed').length}, refuted ${items.filter((i) => i.outcome === 'refuted').length}, uncertain ${items.filter((i) => i.outcome === 'uncertain').length}, still queued (a verifier died) ${items.filter((i) => i.outcome === 'queued').length}; refuted-sample reversals ${flips}`)
return { items }
