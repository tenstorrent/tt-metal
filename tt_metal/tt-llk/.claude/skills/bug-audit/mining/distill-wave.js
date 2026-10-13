export const meta = {
    name: 'bug-mining-distill',
    description: 'Distill each bug class of a repo into a pack section: what goes wrong, checks, seeds, hot areas, incomplete-fix lessons, candidate sites',
    whenToUse: 'After distill.py inputs. args = {repo, files: [abs paths], current}. Assemble with distill.py assemble.',
    phases: [{title: 'Distill', detail: 'one agent per class input file'}],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['class', 'title', 'what_goes_wrong', 'checks', 'seeds', 'hot_areas', 'incomplete_fix_lessons', 'candidate_sites'],
  properties: {
    class: { type: 'string', description: 'the class id exactly as given in the input file' },
    title: { type: 'string', description: 'a 3-8 word title for how this class shows up in this repo' },
    what_goes_wrong: { type: 'string', description: '2-5 sentences: the recurring mechanisms, in general terms' },
    checks: { type: 'array', maxItems: 10, items: { type: 'string' }, description: 'concrete, reusable audit checks, most productive first' },
    seeds: { type: 'array', maxItems: 10, items: { type: 'string' }, description: 'code shapes / grep patterns worth hunting, each with where it applies' },
    hot_areas: { type: 'array', maxItems: 8, items: { type: 'string' }, description: 'directories/subsystems where this class clusters' },
    incomplete_fix_lessons: { type: 'array', maxItems: 6, items: { type: 'string' }, description: 'how past fixes of this class ended up partial/reverted/wrong, as lessons' },
    candidate_sites: { type: 'array', maxItems: 12, items: { type: 'string' }, description: '"path:line — one-line why" for the strongest unfixed-sibling leads you re-checked in the current tree' },
  },
}

const prompt = (f) => `You are writing one section of a bug-audit knowledge pack for ${A.repo}. Future code auditors (LLM
agents) will read it before hunting bugs, so it must be dense, concrete and reusable. Read-only; edit nothing.

Read ${f}. It holds every analysed historical bug of ONE class in this repo: root cause, trigger, how the fix went
(complete / partial / reverted ...), the audit check and seed each analyst proposed, and unfixed-sibling leads. It may
also hold reviewer-caught defects for the class.

Write the section:
- what_goes_wrong: the RECURRING mechanisms. Merge duplicates; do not list cases one by one.
- checks: the most productive checks, each phrased so it applies to code the auditor has never seen. Prefer checks
  that several bugs support. Merge overlapping analyst checks into one sharper check.
- seeds: code shapes or grep patterns with where they apply (directory or API family).
- hot_areas: where the class clusters.
- incomplete_fix_lessons: patterns in how fixes went partial, got reverted, or fixed the wrong root cause.
- candidate_sites: the strongest unfixed-sibling leads. Re-check each one in the current tree at ${A.current}
  (read-only), keep only those that still look live, and write "path:line — why". Paths relative to the repo root.
  These are leads, not findings; at most 12.

Rules: general engineering terms; NO issue or PR numbers anywhere; drop any material from inputs marked
public_safe=false; no hardware-internal signal names or RTL file paths. Class id: copy it exactly from the input file.`

const results = await pipeline(A.files, (f) =>
  agent(prompt(f), { label: `distill:${f.split('/').pop()}`, phase: 'Distill', schema: SCHEMA }))
const sections = results.filter(Boolean)
log(`distilled ${sections.length}/${A.files.length} class sections`)
return { sections, missing: A.files.filter((f, i) => !results[i]) }
