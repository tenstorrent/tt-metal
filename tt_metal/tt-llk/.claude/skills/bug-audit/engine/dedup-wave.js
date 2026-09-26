export const meta = {
    name: 'bug-audit-dedup',
    description: 'Judge which confirmed findings in each directory group are the same defect (or arch copies of it)',
    whenToUse: 'After dedup.py inputs. args = {inputs: [abs paths]}. Then dedup.py persist <output> and consolidate.py.',
    phases: [{title: 'Dedup', detail: 'one judge per directory group'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args
if (!Array.isArray(A.inputs)) throw new Error('args.inputs must be an array of paths from dedup.py inputs')

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['group', 'clusters'],
  properties: {
    group: { type: 'string' },
    clusters: {
      type: 'array',
      description: 'only groups of 2+ findings that are the same defect; omit singletons',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['canonical', 'duplicates', 'relation', 'why'],
        properties: {
          canonical: { type: 'string', description: 'key of the finding that best states the defect' },
          duplicates: { type: 'array', items: { type: 'string' }, description: 'keys of the other findings of the SAME defect' },
          relation: { type: 'string', enum: ['same-defect', 'arch-copy'], description: 'arch-copy: the same defect in per-arch copies of the code (separate fix sites)' },
          why: { type: 'string' },
        },
      },
    },
  },
}

const prompt = (p) => `Deduplicate bug-audit findings. Read-only. Read ${p}: confirmed findings from one directory group
(per-arch copies are grouped together), each with a key.

Two findings are the SAME defect only when fixing one would fix the other: the same mechanism, e.g. reported once at a
call site and once at the definition, or twice at neighbouring lines. Findings that merely share a class, a file or
a theme are NOT duplicates. The same defect in per-arch copies of a file (wormhole_b0 / blackhole / quasar ...) is an
"arch-copy" cluster: one issue with several fix sites. Check that each copy really has the defect: an identical line
can be correct on one arch and wrong on another. You may open the code to decide.
Return only real clusters. Keep the canonical finding the one that states the defect most precisely.`

const results = await pipeline(A.inputs, (p) =>
  agent(prompt(p), { label: `dedup:${p.split('/').pop()}`, phase: 'Dedup', schema: SCHEMA, effort: 'low' }))
const good = results.filter(Boolean)
log(`dedup: ${good.reduce((s, r) => s + r.clusters.length, 0)} clusters over ${good.length}/${A.inputs.length} groups`)
return { results: good }
