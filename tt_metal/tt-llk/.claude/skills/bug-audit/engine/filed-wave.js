export const meta = {
    name: 'bug-audit-filed-check',
    description: 'For each open finding, judge whether any candidate GitHub issue or PR already reports the same defect',
    whenToUse: 'After filed_check.py candidates. args = {inputs: [abs paths]}. Then filed_check.py persist <output>.',
    phases: [{title: 'Filed?', detail: 'one judge per finding with GitHub candidates'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args
if (!Array.isArray(A.inputs)) throw new Error('args.inputs must be an array of paths from filed_check.py candidates')

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['key', 'matches'],
  properties: {
    key: { type: 'string', description: 'the finding key from the input, copied exactly' },
    matches: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['number', 'kind', 'state', 'url', 'same_bug', 'why'],
        properties: {
          number: { type: 'integer' },
          kind: { type: 'string', enum: ['issue', 'pr'] },
          state: { type: 'string', description: 'OPEN, CLOSED or MERGED, as given' },
          url: { type: 'string' },
          same_bug: { type: 'boolean' },
          why: { type: 'string' },
        },
      },
      description: 'only the candidates that are plausibly related; same_bug=true only for the same defect',
    },
  },
}

const prompt = (p) => `Decide whether a bug is already reported on GitHub. Read-only: do not comment, file or edit anything.

Read ${p}: one audit finding (file, line, defect, failure scenario, evidence; also_at lists its arch copies or other
sites) and candidate GitHub issues/PRs found by searching its file name and identifiers.

A candidate is the SAME bug only if it reports this defect: the same mechanism at the same code, even if described
differently or reported for one arch when the finding covers several. It is NOT the same bug merely because it
mentions the file, the op or the symptom. A PR is a match if it fixes (or tries to fix) this defect. You may use
\`gh issue view <n>\` / \`gh pr view <n> --comments\` or \`gh pr diff <n>\` for a candidate whose text is not decisive.
List only plausibly related candidates; mark same_bug=true only when you are confident.`

const results = await pipeline(A.inputs, (p) =>
  agent(prompt(p), { label: `filed:${p.split('/').pop()}`, phase: 'Filed?', schema: SCHEMA, effort: 'low' }))
const good = results.filter(Boolean)
log(`checked ${good.length}/${A.inputs.length}: ${good.filter((r) => r.matches.some((m) => m.same_bug)).length} already reported`)
return { results: good }
