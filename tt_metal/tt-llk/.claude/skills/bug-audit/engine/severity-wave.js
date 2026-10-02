export const meta = {
    name: 'bug-audit-severity',
    description: 'Re-rate the severity of confirmed findings (high / medium / low) with a fixed rubric, in batches',
    whenToUse:
        'Every audit, before filing, so all confirmed findings are rated against one rubric. args = {input_dir, n} from severity.py prepare (batch files b0000.json ... each {items: [...]}). Then severity.py persist <output>.',
    phases: [{title: 'Rate', detail: 'one agent per batch of about 10 findings'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)
const A = typeof args === 'string' ? JSON.parse(args) : args
if (!A.input_dir || !Number.isInteger(A.n) || A.n < 1) throw new Error('args must be {input_dir, n}')
const inputs = Array.from({ length: A.n }, (_, i) => `${A.input_dir}/b${String(i).padStart(4, '0')}.json`)

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ratings'],
  properties: {
    ratings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['key', 'severity', 'why'],
        properties: {
          key: { type: 'string', description: 'copy the item key exactly' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          why: { type: 'string', description: 'one sentence: who hits it, and what happens' },
        },
      },
    },
  },
}

const prompt = (p) => `Rate the severity of confirmed bugs. Read-only. Read ${p}: about 10 findings, each with a key, the
location, the claim, the failure scenario, the verifiers' reasons (why it was confirmed, with the triggering path),
any other_sites the entry covers, and hunters_worst.

Rubric, judged by who hits it and what happens:
- high: wrong results, a crash, a hang, or memory corruption on a normal, supported configuration or common
  op/model path, in shipped code.
- medium: the same kinds of failure, but only on a narrower configuration (a specific dtype, shape, arch, flag or
  sharding), or a significant contract violation a caller can reasonably hit.
- low: diagnostics, error messages, logging, tests, benchmarks, tooling, debug-only paths, or behaviour that needs an
  unusual input to reach.
Rate each on its own merits, from the code and the verifier reasons, not from the claimed severity.
An item's other_sites are the same defect in sibling copies (another arch, dtype or variant) or another confirmed
defect on the same line. One rating covers the whole entry, so rate its WORST site: read each site's claim and failure
scenario too. hunters_worst is the highest rating any hunter gave across these sites; rate below it only when no
site's own claim supports it, and say why. Return one rating per item, same keys.`

const results = await pipeline(inputs, (p) =>
  agent(prompt(p), { label: `rate:${p.split('/').pop()}`, phase: 'Rate', schema: SCHEMA, effort: 'low' }))
const good = results.filter(Boolean)
const all = good.flatMap((r) => r.ratings)
log(`rated ${all.length}: high ${all.filter((r) => r.severity === 'high').length}, medium ${all.filter((r) => r.severity === 'medium').length}, low ${all.filter((r) => r.severity === 'low').length}`)
return { ratings: all, missing: inputs.filter((p, i) => !results[i]) }
