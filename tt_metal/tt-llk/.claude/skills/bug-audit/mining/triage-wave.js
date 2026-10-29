export const meta = {
    name: 'bug-mining-triage',
    description: 'Triage every mined fix case: real code bug or not, bug class, component, symptom, and whether it deserves a deep read',
    whenToUse: 'After make_chunks.py. args = {repo, chunks: [abs paths] | chunk_dir + n_chunks, classes: [abs paths]}. Persist with persist_mining.py triage.',
    phases: [{title: 'Triage', detail: 'one cheap agent per chunk of ~40 case digests'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args
// chunks may be given as a list, or as {chunk_dir, n_chunks} (make_chunks.py names them chunk-0000.json ...)
if (!A.chunks) A.chunks = Array.from({ length: A.n_chunks }, (_, i) => `${A.chunk_dir}/chunk-${String(i).padStart(4, '0')}.json`)

const CASE = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'verdict', 'not_bug_kind', 'classes', 'component', 'arch', 'symptom', 'triggers', 'mechanism', 'deep_priority'],
  properties: {
    id: { type: 'string' },
    verdict: { type: 'string', enum: ['code-bug', 'test-bug', 'not-a-code-bug', 'unclear'],
               description: 'code-bug = a defect in shipped (non-test) code; test-bug = a defect in tests/harness/CI scripts' },
    not_bug_kind: { type: 'string', enum: ['n/a', 'infra-ci', 'feature', 'perf', 'docs', 'build-deps', 'model-tuning', 'hardware', 'flaky-unknown', 'user-error', 'other'] },
    classes: { type: 'array', items: { type: 'string' }, description: 'class ids from the class files, best first; a new kebab-case id only when none fits' },
    component: { type: 'string', description: 'the subsystem, as a short path-like name (e.g. ttnn/matmul, llk/unpack, fabric, dispatch)' },
    arch: { type: 'array', items: { type: 'string', enum: ['wormhole', 'blackhole', 'quasar', 'grayskull', 'all', 'unknown'] } },
    symptom: { type: 'string', enum: ['hang', 'crash', 'wrong-output', 'accuracy-pcc', 'compile-error', 'perf', 'flaky', 'resource-leak', 'other'] },
    triggers: { type: 'array', items: { type: 'string' }, description: 'short tags for what triggers it: dtype, shape class, layout, sharding, multi-device, tile size, flag...' },
    mechanism: { type: 'string', description: 'one sentence: the defect mechanism in general terms, or why it is not a bug' },
    deep_priority: { type: 'integer', minimum: 0, maximum: 3,
                     description: '3 = a mechanism an auditor must learn (subtle, recurring, or the fix looks incomplete/reverted); 2 = solid ordinary bug; 1 = trivial; 0 = not a bug' },
  },
}
const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['chunk', 'cases'],
  properties: { chunk: { type: 'string' }, cases: { type: 'array', items: CASE } },
}

const prompt = (chunk) => `You are triaging historical bug reports and fixes from the ${A.repo} repository, to learn
which kinds of bugs this codebase actually has. Read-only.

1. Read the bug-class lists: ${A.classes.join(', ')}. Put their ids in "classes" EXACTLY as written there,
   hyphens included (e.g. 'index-math', 'tt-tile-shape'); invent a new kebab-case id only when none fits.
2. Read ${chunk}. It holds ~40 case digests: an issue or PR title, labels, a trimmed body, the fix commit subjects
   and files, and later history (reverted? later commits citing it? later fix-like commits to the same files?).
3. Return ONE entry per case, in the same order, with the same id. Judge from the digest; you may run
   \`gh issue view\`/\`gh pr view\` (repo ${A.repo}) for a case whose digest is too thin to classify, but keep it cheap.

Be strict about "code-bug": a defect in shipped code (wrong result, crash, hang, race, leak, compile failure of valid
user code). CI flakes with no identified code cause, infra, feature requests, perf work, model accuracy tuning and
docs are NOT code bugs. Neither is a change whose pre-fix code was already correct: a cleanup or refactor, a
lint or static-analyzer appeasement, "no functional change", enabling a feature that was deliberately gated,
removing a workaround whose real fix landed elsewhere, a timing tweak that makes a flaky test pass without naming a
defect, a redundant or include-only change, or a debuggability improvement. Ask: would an auditor reading the
PRE-fix code have been right to report a defect? If not, it is not-a-code-bug. Prefer "unclear" to guessing. For deep_priority, favour cases where the fix touches code
(not only tests/CI), the mechanism is non-obvious, or the history shows a revert or several fix attempts.`

const results = await pipeline(A.chunks, (chunk) =>
  agent(prompt(chunk), { label: `triage:${chunk.split('/').pop()}`, phase: 'Triage', schema: SCHEMA, effort: 'low' })
    .then((r) => (r ? { ...r, chunk } : null)))

const good = results.filter(Boolean)
const n = good.reduce((s, r) => s + r.cases.length, 0)
log(`triaged ${n} cases in ${good.length}/${A.chunks.length} chunks`)
const missing = A.chunks.filter((c) => !good.find((r) => r.chunk === c))
if (missing.length) log(`MISSING chunks (agent died): ${missing.length} — rerun them`)
return { results: good, missing }
