export const meta = {
    name: 'bug-mining-reviews',
    description: 'Mine PR review threads for defects reviewers caught before merge, and the checks reviewers keep applying',
    whenToUse:
        'After fetch_reviews.py + chunking. args = {repo, git, chunks: [abs paths] | chunk_dir + n_chunks, classes: [abs paths]}. Persist with persist_mining.py.',
    phases: [{title: 'Reviews', detail: 'one agent per chunk of ~10 PRs worth of review threads'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args
if (!A.chunks) A.chunks = Array.from({ length: A.n_chunks }, (_, i) => `${A.chunk_dir}/chunk-${String(i).padStart(4, '0')}.json`)

const ITEM = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'pr', 'path', 'class', 'defect', 'addressed', 'reviewer_check', 'confidence'],
  properties: {
    id: { type: 'string', description: '<pr number>:<thread idx field>' },
    pr: { type: 'integer' },
    path: { type: 'string' },
    class: { type: 'string', description: 'class id from the class files, or a new kebab-case id' },
    defect: { type: 'string', description: 'the defect the reviewer pointed at, in general terms' },
    addressed: { type: 'string', enum: ['code-changed', 'author-disputed', 'not-addressed', 'unknown'] },
    reviewer_check: { type: 'string', description: 'the general check the reviewer applied, phrased so an auditor can reuse it' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'], description: 'that this was a real defect, not a preference' },
  },
}
const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['chunk', 'items', 'rejections'],
  properties: {
    chunk: { type: 'string' },
    items: { type: 'array', items: ITEM },
    rejections: {
      type: 'array',
      description: 'for PRs closed WITHOUT merging: why the approach was rejected, when that reveals an invariant',
      items: {
        type: 'object', additionalProperties: false, required: ['pr', 'lesson'],
        properties: { pr: { type: 'integer' }, lesson: { type: 'string' } },
      },
    },
  },
}

const prompt = (chunk) => `You are mining code-review history from ${A.repo} to learn what defects reviewers catch before
they ship. Read-only; do not comment or edit anything.

1. Skim the class lists: ${A.classes.join(', ')}.
2. Read ${chunk}: several PRs, each with the inline review threads that a keyword filter kept (file, line,
   resolved/outdated flags, comments, and the diff hunk the comment was on). Most will still be non-defects.
3. Emit one item per thread where a reviewer pointed at a real DEFECT: something that would produce a wrong result,
   crash, hang, race, leak or broken contract. SKIP style, naming, docs, test-coverage requests, perf-only remarks,
   questions answered with "it is fine because ...", and bot comments.
4. "addressed": code-changed if the thread is outdated/resolved after the author agreed or the next commit changed
   that code (\`gh pr view <n> --repo ${A.repo} --json commits\` or \`git -C ${A.git} log\` if unsure); author-disputed if
   the author argued it was fine; else not-addressed/unknown. A disputed item is still useful when the reviewer was
   right, but lower its confidence.
5. For PRs in state CLOSED (not merged), add a rejection lesson when the discussion shows WHY the approach was wrong.
Use the class ids EXACTLY as written in the class files (hyphens included). Phrase defect and reviewer_check in general
engineering terms, with no issue/PR numbers in reviewer_check.`

const results = await pipeline(A.chunks, (c) =>
  agent(prompt(c), { label: `reviews:${c.split('/').pop()}`, phase: 'Reviews', schema: SCHEMA, effort: 'low' })
    .then((r) => (r ? { ...r, chunk: c } : null)))
const good = results.filter(Boolean)
log(`review items ${good.reduce((s, r) => s + r.items.length, 0)}, rejection lessons ${good.reduce((s, r) => s + r.rejections.length, 0)}`)
return { results: good, missing: A.chunks.filter((c) => !good.find((r) => r.chunk === c)) }
