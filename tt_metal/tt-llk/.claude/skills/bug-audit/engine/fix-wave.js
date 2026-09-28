export const meta = {
    name: 'bug-audit-fixes',
    description: 'Write a site-grounded suggested fix for each open confirmed finding, in batches',
    whenToUse:
        'After consolidation, before filing: every open finding (and each merged site) gets a fix a developer can act on without the history. args = {input_dir, n, root} from fixes.py prepare; apply with fixes.py persist.',
    phases: [{title: 'Fix', detail: 'one agent per batch of about 20 findings'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)
const A = typeof args === 'string' ? JSON.parse(args) : args
if (!A.input_dir || !Number.isInteger(A.n) || A.n < 1 || !A.root) throw new Error('args must be {input_dir, n, root}')
const inputs = Array.from({ length: A.n }, (_, i) => `${A.input_dir}/f${String(i).padStart(4, '0')}.json`)

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['fixes'],
  properties: {
    fixes: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['key', 'fix'],
        properties: {
          key: { type: 'string', description: 'copy the item key exactly' },
          fix: { type: 'string', description: 'the suggested fix, ending with a line that starts "Test:"' },
        },
      },
    },
  },
}

const prompt = (p) => `Write concrete SUGGESTED FIXES for confirmed bugs. You do NOT edit any source code.
Read ${p}: about 20 findings, each with a key, the site (file:line), the claim, the failure scenario, the evidence and
the verifiers' write-ups, which trace the defect in the CURRENT code and are your main source. When the finding came
from history ("history-sibling"), its scenario and evidence describe an OLD bug elsewhere that this site resembles.
The code tree, which the line numbers refer to, is ${A.root}. Read-only.

For EACH finding:
1. Read the write-ups, then open the code at the site and the files they cite. Check their line numbers.
2. Write a fix a developer can act on WITHOUT knowing any past bug: WHAT to change, WHERE (file:line or function),
   and HOW (the guard, the corrected expression, the missing barrier, the argument to pass; a short snippet if it
   helps). If the write-ups propose fixes, pick the minimal correct one. If there are real options (fix the shared
   default or each caller), give the recommended one first.
3. End with one line starting "Test:" naming how to prove fail-without / pass-with: the existing test to extend if
   you can find one, or the deciding experiment the write-ups name.
4. Never write "apply the past fix". Never invent: if the code does not support a confident fix, say what must be
   decided first ("needs owner decision: X or Y") and still give the best candidate.
Two to six sentences plus the Test line. Return one fix per item, same keys.`

const results = await pipeline(inputs, (p) =>
  agent(prompt(p), { label: `fix:${p.split('/').pop()}`, phase: 'Fix', schema: SCHEMA }))
const all = results.filter(Boolean).flatMap((r) => r.fixes)
log(`wrote ${all.length} fixes; ${all.filter((f) => !/\nTest:|^Test:/m.test(f.fix)).length} lack a Test line`)
return { fixes: all, missing: inputs.filter((p, i) => !results[i]) }
