export const meta = {
    name: 'bug-audit-wave',
    description: 'Hunt one wave of file batches for real bugs, then verify every candidate adversarially (staged, 3-valued)',
    whenToUse:
        'One wave of a bug-audit run. args = {run, root, batches: [ids], knowledge: [abs paths], roots?: {batch: tree}, roots_dir?, exec_signals_dir?, known_dir?}. Persist the result with persist_wave.py immediately after it finishes.',
    phases: [
        {title: 'Hunt', detail: 'one agent per batch: read every assigned file fully, hunt grounded bugs'},
        {title: 'Trace audit', detail: 're-trace a sample of each hunter\u2019s boundary ledger; misses become new candidates'},
        {title: 'Screen', detail: 'one refuter per candidate; only a clear refutation kills it'},
        {title: 'Verify', detail: 'survivors get 2 more verifiers on different lenses'},
    ],
}

// clang-format off: repo-wide clang-format mangles JS (splits `return {...}`, so ASI returns undefined)

const A = typeof args === 'string' ? JSON.parse(args) : args
const RUN = A.run
const ROOT = A.root
const BATCHES = A.batches
const KNOWLEDGE = A.knowledge || []
// per-batch trees (the recall benchmark audits each case at the commit before its fix); default is the run's tree
const ROOTS = A.roots || {}
// bench runs may pass roots_dir instead: batch BENCH-<id> lives at <roots_dir>/<id>
if (A.roots_dir) for (const b of BATCHES) ROOTS[b] = `${A.roots_dir}/${b.replace(/^BENCH-/, '')}`
const rootOf = (batch) => ROOTS[batch] || ROOT

const FIND_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['batch', 'files_read', 'files_skipped', 'boundaries', 'classes_checked', 'boundaries_skipped', 'findings'],
  properties: {
    batch: { type: 'string' },
    files_read: {
      type: 'array',
      description: 'one entry per assigned file you read to the end',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['path', 'lines', 'last_line'],
        properties: {
          path: { type: 'string' },
          lines: { type: 'integer', description: 'total line count of the file as you read it' },
          last_line: { type: 'string', description: 'the exact text of the last non-blank line of the file' },
        },
      },
    },
    files_skipped: { type: 'array', items: { type: 'string' } },
    boundaries: {
      type: 'array',
      description: 'the contract-trace ledger: every boundary your files cross that you checked, one entry each',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['site', 'kind', 'other_side', 'verdict', 'note'],
        properties: {
          site: { type: 'string', description: 'repo-relative path:line in YOUR batch where the boundary is crossed' },
          kind: { type: 'string', enum: ['call-callee', 'host-kernel', 'producer-consumer', 'validate-impl', 'decl-callers', 'symbol-resolution', 'sibling'] },
          other_side: { type: 'string', description: 'REPO-RELATIVE path:line (or path:a-b) of the other side you actually opened and read' },
          verdict: { type: 'string', enum: ['consistent', 'mismatch', 'unclear'] },
          note: { type: 'string', description: 'what you compared (types, counts, order, units, conditions), one line' },
        },
      },
    },
    classes_checked: {
      type: 'array', items: { type: 'string' },
      description: 'class ids you deliberately swept these files for (the class x file coverage ledger); be honest',
    },
    boundaries_skipped: {
      type: 'array', items: { type: 'string' },
      description: 'boundaries you saw but did not trace, as "file:line: why" (budget, out of scope...)',
    },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['file', 'line', 'category', 'severity', 'summary', 'failure_scenario', 'evidence', 'suggested_fix', 'source'],
        properties: {
          file: { type: 'string' },
          line: { type: 'integer' },
          category: { type: 'string', description: 'class id from the knowledge files when one fits, else a short kebab-case name' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          summary: { type: 'string' },
          failure_scenario: { type: 'string', description: 'concrete inputs/config/state -> wrong output, crash, hang, leak' },
          evidence: { type: 'string', description: 'quoted code + why it is wrong, incl. the sibling/caller/contract that proves intent' },
          suggested_fix: { type: 'string' },
          source: { type: 'string', enum: ['own-hunt', 'knowledge-class', 'knowledge-seed', 'execution-signal', 'trace-audit'], description: 'what led you to it' },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'reason'],
  properties: {
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'uncertain'] },
    reason: { type: 'string', description: 'what you checked and the decisive line(s)' },
  },
}

const knowledgeBlock = KNOWLEDGE.length
  ? `KNOWLEDGE — read these files BEFORE the code, in order:
${KNOWLEDGE.map((k) => `  - ${k}`).join('\n')}
They hold a bug-class list weighted by this codebase's real bug history, plus per-area hot spots and "fix seeds"
(past defects whose pattern may recur in sibling code). Use them to PRIORITISE: check the high-weight classes
first, and grep the pack AND its "-detail-*.md" companions (same directory) for your batch's directory and file names,
to pick up the seeds, incomplete-fix lessons and candidate sites that apply here. A candidate site is a lead to
verify, never a finding by itself.
They are a FLOOR, never a ceiling — hunt every defect you can ground, whether or not a listed class names it.`
  : ''

const execBlock = (batch) => A.exec_signals_dir
  ? `EXECUTION SIGNALS: if ${A.exec_signals_dir}/${batch}.json exists, read it first. It lists diagnostics that a
build, a static analyzer, a sanitizer or an existing test produced on YOUR files (tool, file:line, severity, message).
Treat each one as a lead, not a verdict. Decide whether it reveals a real, reachable defect: most warnings are noise,
while a link error, a sanitizer report or a failing test almost always points at something real. Report the real ones
with source "execution-signal", quoting the signal in the evidence.
`
  : ''

const knownBlock = (batch) => A.known_dir
  ? `PRIOR RUNS: if ${A.known_dir}/${batch}.json exists, read it. It lists what earlier audits already CONFIRMED or
REFUTED in your files, with the refutation reasons. Do not re-report a confirmed one. Re-raise a refuted one only
if you have evidence its refutation missed, and say so explicitly in the evidence.
`
  : ''

const huntPrompt = (batch) => `You are hunting REAL BUGS in a source tree. Read-only audit; do NOT edit any file under the tree.

Tree to audit: ${rootOf(batch)}   (pinned checkout — audit THIS tree, not any other checkout on the box)
${ROOTS[batch] ? 'This is a benchmark tree: do NOT run git log/show/blame or read any other checkout, issue or PR; judge the code as it stands.\n' : ''}
${knowledgeBlock}

${execBlock(batch)}${knownBlock(batch)}
Do NOT build, run tests, run the code, or touch any device or hardware: this hunt is read-only analysis. Execution,
when the user enabled it, happens in a separate tier whose results you see as EXECUTION SIGNALS.

ASSIGNMENT: batch \`${batch}\`.
1. Read ${RUN}/batches/manifest.json and find the object whose "batch" field == "${batch}". Its "files" array lists
   paths relative to the tree root.
2. Read EVERY one of those files COMPLETELY — all lines, no head/tail/limit/sampling; read large files in successive
   chunks until you reach the last line. For each file, record its line count and the exact text of its last
   non-blank line: that is checked mechanically against the tree, and a mismatch sends the file back for re-audit.

WHAT COUNTS AS A FINDING: a concrete defect in the code as written, that a real caller or configuration reaches:
wrong result, crash, hang/deadlock, data race, leak, silent corruption, or a guard that does not guard.
NOT findings: style, naming, comments, missing docs, pure performance, speculative refactors, or anything you cannot
ground in code you actually read.

METHOD (be a hunter, not a checklist-filler):
- FOLLOW every suspicious site: callers, sibling overloads, the header, the enum, whoever supplies the arguments and
  whoever consumes the result. The strongest evidence is a sibling that does the same thing correctly, or a consumer
  whose stated expectation the code violates.
- Vary the parameters: for each function ask which shapes, sizes, counts, dtypes, flags, per-platform variants and
  edge values (0, 1, odd, max, empty) its callers can pass, and whether every one of them is handled.
- CONTRACT TRACE (mandatory, one hop, RECORDED): most missed bugs sit on a boundary, with the evidence in the file
  next door. For every boundary your files cross, open the other side and check that the two agree, and record each
  one in "boundaries" (site, kind, the other side's file:line you read, verdict, what you compared). A boundary you
  saw but did not trace goes in "boundaries_skipped" with the reason. The ledger is checked: a sample of your
  "consistent" entries is re-traced by an independent auditor, so record only what you actually compared.
  * call -> callee: each argument against the parameter type, including silent narrowing (a wider value into a
    narrower parameter, or into a bit field whose *_MASK/*_SHIFT constant bounds it), argument order and defaults;
  * host -> kernel: compile-time/runtime arg order and count, CB ids and page sizes, defines the kernel reads;
  * producer -> consumer: counts that both sides derive (credits, dvalids, pages, tiles), on every branch;
  * validate() -> implementation: every input combination the validator admits has a factory or kernel path that
    handles it, and nothing the implementation needs goes unchecked;
  * declaration -> definition -> every caller: after a signature, keyword or return-type change, each call site;
  * symbol -> where it resolves: extern and linker symbols, macros (a define set on the compile line can collide
    with an identifier in a header), namespace lookup, visibility across shared-library boundaries.
- Compare SIBLINGS: the same API called elsewhere with different arguments, the same quantity derived by two
  formulas, the same register or constant defined twice. A call site or constant that differs from all its siblings
  is a lead.
- Mechanical sweeps worth doing on every file: a value computed and never used (often a dropped parameter);
  .begin()/front()/[0] on a container that can be empty; an error message whose printed bound disagrees with its
  guard; #if/#elif on a macro that some build flavour leaves undefined.
- Use Grep/Glob/Bash freely across the whole tree. You are RESPONSIBLE only for your files, not limited to them.
- Before reporting, argue the other side: an invariant, assert, caller-side guarantee or specialisation that makes it
  safe? Is the path reachable? If it is safe or dead, drop it.
- Look HARD at every site you read, not just once over. Before moving on from a function, ask what input, config,
  build flavour or arch would make it wrong. Skimming is the most common way a real defect in front of you gets missed.
- One defect per finding. Never bundle a second claim into a finding: a wrong extra claim gets the whole finding refuted.
- Zero findings is a normal, honest result. Do not invent anything to look productive.

OUTPUT: return the JSON object (the harness persists it). "files_skipped" should be empty; if you genuinely could not
read a file, list it there and leave it out of files_read.`

const verifyPrompt = (f, lens, root) => `Adversarially examine this claimed bug in the tree at ${root}. Your job is to find out whether it is REAL.

File: ${f.file}:${f.line}
Category: ${f.category}   Severity claimed: ${f.severity}
Claim: ${f.summary}
Failure scenario claimed: ${f.failure_scenario}
Evidence offered: ${f.evidence}

${root !== ROOT ? 'This is a benchmark tree: do NOT run git log/show/blame or read any other checkout, issue or PR.\n' : ''}Examine it through the ${lens} lens. Read the file and its surroundings yourself — do NOT trust the quoted evidence.
Check callers, declarations, enum values, template instantiations, asserts, and every invariant that could make the
code correct as written. Consider unreachable paths, upstream guards, intended behaviour, and misread overloads,
operator precedence or implicit conversions.

Grounding rules:
- A missing or silent document is never evidence. "The docs do not say X is ordered" proves nothing either way. Hold
  hardware-behaviour claims to the code that emits the instruction and the authoritative docs; if neither settles
  it, the verdict is "uncertain".
- "No caller" is not "unreachable". Test kernels and JIT-compiled kernels get template arguments and constants from
  build defines, so trace the defines before calling a value impossible.

Reachability of library code: a public API, LLK or header-library entry point is reachable by default, even when
this tree has no caller. Its callers may live in another repository (tt-llk functions are called from tt-metal, for
example). Only refute on reachability if you can show that every legal call is guarded or rejected, not merely that
no caller is visible here.

Verdict:
- "refuted": you found the concrete reason it is NOT a bug (cite the line that makes it safe or unreachable).
- "confirmed": you independently reproduced the reasoning and the defect is real AND reachable (cite the lines).
- "uncertain": you could neither confirm nor refute it from the code. Say what evidence would settle it.
Do not pick "refuted" merely because you are unsure — that is what "uncertain" is for.`

const FINDING = FIND_SCHEMA.properties.findings.items
const TRACE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['rechecked', 'findings'],
  properties: {
    rechecked: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false, required: ['site', 'agrees', 'note'],
        properties: { site: { type: 'string' }, agrees: { type: 'boolean', description: 'true if the hunter verdict holds' }, note: { type: 'string' } },
      },
    },
    findings: { type: 'array', items: FINDING, description: 'defects you found at boundaries the hunter called consistent or left unreported' },
  },
}

// deterministic sample: first, middle and last "consistent" entries, plus every "mismatch" with no finding at its site
function traceSample(res) {
  const b = res.boundaries || []
  const ok = b.filter((x) => x.verdict === 'consistent')
  const pick = ok.length <= 3 ? ok : [ok[0], ok[Math.floor(ok.length / 2)], ok[ok.length - 1]]
  const reported = new Set((res.findings || []).map((f) => `${f.file}:${f.line}`))
  const unreported = b.filter((x) => x.verdict !== 'consistent' && !reported.has(x.site))
  return [...pick, ...unreported]
}

const tracePrompt = (batch, sample, root) => `You are auditing another auditor's contract-trace ledger in the tree at ${root}.
Read-only. ${ROOTS[batch] ? 'This is a benchmark tree: do NOT run git log/show/blame or read any other checkout, issue or PR.' : ''}

For each boundary below, independently open BOTH sides and decide whether they really agree: argument types and
widths (including narrowing into bit fields), argument order and count, units, counts both sides derive, the
conditions each side assumes, and what each accepted input combination does. The auditor's verdict and note are
shown; do not trust them.
${sample.map((x, i) => `${i + 1}. [${x.kind}] ${x.site}  <->  ${x.other_side}   auditor said: ${x.verdict} (${x.note})`).join('\n')}

For each one, say whether the auditor's verdict holds. Report every real, reachable defect you find at these
boundaries as a finding (one defect per finding; set source to "own-hunt"). Zero findings is a fine answer.`

const SCREEN_LENS = 'correctness / control-flow — does the defect actually exist in the code as written?'
const DEEP_LENSES = [
  'reachability — can a real caller/config actually hit this, or is the path dead or guarded upstream?',
  'type & API semantics — overload resolution, default arguments, integer conversions, units, and the declared contract',
]

function tally(f, batch, votes, planned) {
  const got = votes.filter(Boolean)
  const n = (v) => got.filter((x) => x.verdict === v).length
  const c = n('confirmed'), r = n('refuted'), u = n('uncertain')
  const died = planned - got.length
  // majority of the votes that were actually cast; a dead voter is unknown, never a refutation
  let status = 'uncertain'
  if (c >= 2 && c > r) status = 'confirmed'
  else if (r >= 2 && r > c) status = 'refuted'
  if (died > 0 && status !== 'confirmed') status = 'needs_recheck'
  return { ...f, batch, status, votes: { confirmed: c, refuted: r, uncertain: u, died },
           reasons: got.map((v) => `[${v.verdict}] ${v.reason}`) }
}

// A workflow may spawn at most 1000 agents. Measured on the final version: about 17.7 agents per batch (hunt, trace
// audit, a screen per candidate including trace-audit ones, two deep verifiers per survivor). Past the cap, agent()
// throws and the tail of the wave is lost, so keep waves at 50 batches (about 885 agents) or fewer.
const MAX_BATCHES = A.max_batches || 50
if (BATCHES.length > MAX_BATCHES) {
  throw new Error(`wave of ${BATCHES.length} batches exceeds ${MAX_BATCHES} (the 1000-agent cap at ~18 agents/batch); ` +
                  'split it with next_wave.py N <= 50, or pass max_batches if your measured agents/batch is lower')
}
log(`wave: ${BATCHES.length} batches, knowledge files: ${KNOWLEDGE.length}`)

const results = await pipeline(
  BATCHES,
  (batch) => agent(huntPrompt(batch), { label: `hunt:${batch}`, phase: 'Hunt', schema: FIND_SCHEMA }),
  async (res, batch) => {
    if (!res) return { batch, ok: false, hunt: null, judged: [] }
    const sample = traceSample(res)
    let traceAudit = null
    if (sample.length) {
      traceAudit = await agent(tracePrompt(batch, sample, rootOf(batch)), { label: `trace:${batch}`, phase: 'Trace audit', schema: TRACE_SCHEMA })
    }
    const extra = ((traceAudit && traceAudit.findings) || []).map((f) => ({ ...f, source: 'trace-audit' }))
    res.trace_audit = traceAudit ? { sampled: sample.length, rechecked: traceAudit.rechecked } : { sampled: 0, rechecked: [] }
    const cands = [...(res.findings || []), ...extra]
    if (!cands.length) return { batch, ok: true, hunt: res, judged: [] }

    const screened = await parallel(cands.map((f) => () =>
      agent(verifyPrompt(f, SCREEN_LENS, rootOf(batch)), { label: `screen:${f.file.split('/').pop()}:${f.line}`, phase: 'Screen', schema: VERDICT_SCHEMA })
        .then((v) => ({ f, v }))))

    const judged = []
    const alive = []
    for (const s of screened.filter(Boolean)) {
      if (!s.v) judged.push(tally(s.f, batch, [], 1))
      else if (s.v.verdict === 'refuted') judged.push({ ...tally(s.f, batch, [s.v], 1), status: 'refuted' })
      else alive.push(s)
    }
    const deep = await parallel(alive.map((s) => () =>
      parallel(DEEP_LENSES.map((lens) => () =>
        agent(verifyPrompt(s.f, lens, rootOf(batch)), { label: `verify:${s.f.file.split('/').pop()}:${s.f.line}`, phase: 'Verify', schema: VERDICT_SCHEMA })))
        .then((vs) => tally(s.f, batch, [s.v, ...vs], 1 + DEEP_LENSES.length))))
    judged.push(...deep.filter(Boolean))
    return { batch, ok: true, hunt: res, judged }
  },
)

const good = results.filter(Boolean)
const all = good.flatMap((r) => r.judged)
const count = (s) => all.filter((f) => f.status === s).length
log(`confirmed ${count('confirmed')}, uncertain ${count('uncertain')}, refuted ${count('refuted')}, needs_recheck ${count('needs_recheck')}; batches ok ${good.filter((r) => r.ok).length}/${BATCHES.length}`)
return { batches: BATCHES, results: good }
