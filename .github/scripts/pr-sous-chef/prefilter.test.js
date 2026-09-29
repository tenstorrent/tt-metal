'use strict';

// Unit tests for prefilter.js (the decision logic behind pr-sous-chef.md). Zero
// dependencies -- run with:
//
//   node --test .github/scripts/pr-sous-chef/prefilter.test.js
//
// The last group downloads the gh-aw output sanitizer at the EXACT commit the compiled
// pr-sous-chef.lock.yml pins (gh-aw-actions/setup@<sha>) and runs a nudge body through it,
// so a change in either the pin or the sanitizer that breaks nudge recognition fails here
// instead of silently in production. It needs network access; it is skipped unless
// PR_SOUS_CHEF_SANITIZER_TEST=1 is set.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const https = require('node:https');

const {
  run, FLOW_LABEL, HANDOFF_LABEL, WORKFLOW_ID, HANDOFF_MARKER, MAX_NUDGES_PER_PR, COOLDOWN_MS, SESSION_STALE_MS, PRE_ACTIVATION_LOGIN,
  matchesWorkflowId, isCopilotCodingAgent, isResolvableReviewerBot, isConflicting,
  makeIdentity, verifyFix, buildHandoffComment
} = require('./prefilter.js');

// What gh-aw's add-comment handler appends AFTER sanitization (generate_footer.cjs
// generateXMLMarker + the custom footer from pr-sous-chef.md's `messages.footer`).
const HANDLER_FOOTER = '\n\n> 🍳 *[PR Sous Chef (tt-metal)](https://github.com/tenstorrent/tt-metal/actions/runs/1) — automated follow-up*'
  + `\n\n<!-- gh-aw-agentic-workflow: PR Sous Chef (tt-metal), engine: copilot, id: 1, workflow_id: ${WORKFLOW_ID}, run: https://github.com/tenstorrent/tt-metal/actions/runs/1 -->\n`;
const comment = (login, type, body) => ({ user: { login, type }, body, created_at: '2026-09-28T00:00:00Z' });

// ------------------------------ identity ------------------------------

test('matchesWorkflowId accepts the handler marker and rejects others', () => {
  assert.equal(matchesWorkflowId(`hi${HANDLER_FOOTER}`), true);
  assert.equal(matchesWorkflowId(`<!-- gh-aw-workflow-id: ${WORKFLOW_ID} -->`), true);
  assert.equal(matchesWorkflowId('<!-- gh-aw-agentic-workflow: Other, workflow_id: repo-assist, run: x -->'), false);
  assert.equal(matchesWorkflowId('<!-- gh-aw-comment-type: reaction --> <!-- gh-aw-workflow-id: pr-sous-chef -->'), false);
  assert.equal(matchesWorkflowId(''), false);
  assert.equal(matchesWorkflowId(null), false);
});

test('nudge = own login + handler marker + @copilot; slash reply and forgeries are not', () => {
  const id = makeIdentity('github-actions[bot]');
  assert.equal(id.isNudge(comment('github-actions[bot]', 'Bot', `@copilot fix CI${HANDLER_FOOTER}`)), true);
  // informational /souschef reply: marker, no mention
  assert.equal(id.isNudge(comment('github-actions[bot]', 'Bot', `Nothing to do: checks still running${HANDLER_FOOTER}`)), false);
  assert.equal(id.isOwnComment(comment('github-actions[bot]', 'Bot', `Nothing to do${HANDLER_FOOTER}`)), true);
  // same login, no marker (e.g. copilot-flow-ready.yaml or another workflow) -> not ours
  assert.equal(id.isNudge(comment('github-actions[bot]', 'Bot', '@copilot please')), false);
  // marker + mention from a different account -> not ours
  assert.equal(id.isNudge(comment('someone', 'User', `@copilot fix${HANDLER_FOOTER}`)), false);
});

test('identity follows the resolved login (PAT user) case-insensitively', () => {
  const id = makeIdentity('Some-Maintainer');
  assert.equal(id.login, 'some-maintainer');
  assert.equal(id.isNudge(comment('some-maintainer', 'User', `@copilot go${HANDLER_FOOTER}`)), true);
  assert.equal(id.isNudge(comment('github-actions[bot]', 'Bot', `@copilot go${HANDLER_FOOTER}`)), false);
  // a maintainer's manual @copilot comment has no handler marker -> not counted as a nudge
  assert.equal(id.isNudge(comment('some-maintainer', 'User', '@copilot please fix')), false);
});

test('hand-off is recognised only from github-actions[bot] (pre-activation identity), whatever the nudge identity', () => {
  const id = makeIdentity('some-maintainer');
  assert.equal(id.isHandoff(comment('github-actions[bot]', 'Bot', `${HANDOFF_MARKER}\nhanding off`)), true);
  assert.equal(id.isHandoff(comment('some-maintainer', 'User', `${HANDOFF_MARKER}\nhanding off`)), false);
  assert.equal(id.isHandoff(comment('github-actions[bot]', 'User', `${HANDOFF_MARKER}`)), false);
});

test('hand-off comment carries the marker, no at-mention, and is not a nudge', () => {
  const body = buildHandoffComment({
    pr: { createdAt: '2026-09-27T00:00:00Z' }, nudgeCount: MAX_NUDGES_PER_PR, conflicting: true, zeroDiffStalled: false,
    failedForCopilot: [{ name: 'pr-gate', conclusion: 'FAILURE', url: 'https://x/y' }],
    unansweredAll: [{ reviewer: 'alice', url: 'https://x/t' }], needsMaintainerApproval: [],
    runUrl: 'https://x/run', now: Date.parse('2026-09-28T00:00:00Z')
  });
  assert.ok(body.startsWith(HANDOFF_MARKER));
  assert.equal(/(^|[^\w`])@\w/.test(body), false, 'hand-off must not at-mention anyone');
  const id = makeIdentity('github-actions[bot]');
  assert.equal(id.isHandoff(comment('github-actions[bot]', 'Bot', body)), true);
  assert.equal(id.isNudge(comment('github-actions[bot]', 'Bot', body)), false);
  assert.match(body, /Merge conflict/);
  assert.match(body, /\[pr-gate\]\(https:\/\/x\/y\)/);
  assert.match(body, /alice/);
});

// ------------------------------ actors ------------------------------

test('Copilot coding agent logins are matched exactly, not by prefix', () => {
  assert.equal(isCopilotCodingAgent('copilot-swe-agent'), true);
  assert.equal(isCopilotCodingAgent('Copilot'), true);
  assert.equal(isCopilotCodingAgent('copilot-pull-request-reviewer'), false);
  assert.equal(isCopilotCodingAgent('copilotfan'), false);
  assert.equal(isCopilotCodingAgent(undefined), false);
});

test('resolvable reviewer bots: allowlisted Bot logins only', () => {
  assert.equal(isResolvableReviewerBot({ __typename: 'Bot', login: 'copilot-pull-request-reviewer' }), true);
  assert.equal(isResolvableReviewerBot({ __typename: 'Bot', login: 'github-actions[bot]' }), true);
  assert.equal(isResolvableReviewerBot({ __typename: 'Bot', login: 'cycode-security' }), false);
  assert.equal(isResolvableReviewerBot({ __typename: 'User', login: 'github-actions' }), false);
});

// ------------------------------ merge state ------------------------------

test('merge conflict is DIRTY / mergeable CONFLICTING; CONFLICTING is not a MergeStateStatus', () => {
  assert.equal(isConflicting({ mergeStateStatus: 'DIRTY', mergeable: 'CONFLICTING' }), true);
  assert.equal(isConflicting({ mergeStateStatus: 'DIRTY', mergeable: 'UNKNOWN' }), true);
  assert.equal(isConflicting({ mergeStateStatus: 'UNKNOWN', mergeable: 'CONFLICTING' }), true);
  assert.equal(isConflicting({ mergeStateStatus: 'BLOCKED', mergeable: 'MERGEABLE' }), false);
  assert.equal(isConflicting({ mergeStateStatus: 'UNKNOWN', mergeable: 'UNKNOWN' }), false);
  // the old predicate compared against a value the enum does not have
  assert.equal(isConflicting({ mergeStateStatus: 'CONFLICTING' }), false);
});

// ------------------------------ fix verification ------------------------------

test('a bot thread is resolvable only when the reply cites a PR commit made after the thread opened', () => {
  const opened = Date.parse('2026-09-28T10:00:00Z');
  const commits = [
    { sha: '96fa8f3aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', at: Date.parse('2026-09-28T11:00:00Z') },
    { sha: 'deadbeef00000000000000000000000000000000', at: Date.parse('2026-09-28T09:00:00Z') }, // before the thread
  ];
  assert.equal(verifyFix('Fixed in commit 96fa8f3. Now counting unique NUMA nodes.', commits, opened), true);
  assert.equal(verifyFix('Updated the help text (96FA8F3)', commits, opened), true);
  assert.equal(verifyFix('Done.', commits, opened), false, 'reply without a commit is not evidence');
  assert.equal(verifyFix('Fixed in deadbeef', commits, opened), false, 'commit predates the thread');
  assert.equal(verifyFix('Fixed in 1234567', commits, opened), false, 'not a commit of this PR');
  assert.equal(verifyFix('I could not fix this in 96fa8f3, needs a maintainer', commits, opened), false, 'veto phrase');
  assert.equal(verifyFix("Can't reproduce; see 96fa8f3", commits, opened), false);
  assert.equal(verifyFix('', commits, opened), false);
});

// ------------------------------ run() control flow (integration, mocked GitHub) ------------------------------
//
// The helper tests above cannot catch a bug in the ORDER of run()'s filters — which is
// exactly the class of bug that shipped three times (an unanswered 6th nudge stuck at
// Filter 4 before the hand-off check; a reset that never changed the nudge count; a
// hand-off write that trusted a stale label snapshot). These fixtures drive run()
// end-to-end against an in-memory GitHub: one search hit per scenario, canned GraphQL /
// REST responses, and a recorder for every write.

const MIN = 60 * 1000;
const COOLDOWN_MIN = COOLDOWN_MS / MIN;
const ago = (minutes) => new Date(Date.now() - minutes * MIN).toISOString();
const bot = { login: PRE_ACTIVATION_LOGIN, type: 'Bot' };
// One actionable nudge as the add-comment handler would have left it on the PR.
const nudgeAt = (minutesAgo) => ({ user: bot, body: `@copilot pr-gate failed, please fix.${HANDLER_FOOTER}`, created_at: ago(minutesAgo) });
const handoffAt = (minutesAgo) => ({ user: bot, body: `${HANDOFF_MARKER}\nhanding off`, created_at: ago(minutesAgo) });
const humanAt = (minutesAgo) => ({ user: { login: 'alice', type: 'User' }, body: 'looking', created_at: ago(minutesAgo) });
const labelEvent = (event, name, minutesAgo) => ({ event, label: { name }, actor: { login: 'maintainer' }, created_at: ago(minutesAgo) });
const workEvent = (kind, minutesAgo) => ({ event: `copilot_work_${kind}`, created_at: ago(minutesAgo) });
const SESSION_STALE_MIN = SESSION_STALE_MS / MIN;
const failingCheck = { __typename: 'CheckRun', name: 'pr-gate', status: 'COMPLETED', conclusion: 'FAILURE', startedAt: ago(200), detailsUrl: 'https://x/pr-gate' };
const greenCheck = { __typename: 'CheckRun', name: 'pr-gate', status: 'COMPLETED', conclusion: 'SUCCESS', startedAt: ago(200), detailsUrl: 'https://x/pr-gate' };

// A PR as the mocked GitHub reports it. `labels` is what the GraphQL read (top of the loop)
// returns; `freshLabels` / `freshState` / `freshDraft` is what the pre-write `pulls.get`
// returns, so a stale snapshot can be simulated by making them disagree.
const scenario = (o = {}) => ({
  number: 1, labels: [FLOW_LABEL], freshLabels: undefined, freshState: 'open', freshDraft: false,
  comments: [], timeline: [], checks: [failingCheck], headCommittedAt: ago(24 * 60), assignees: ['Copilot'],
  mergeStateStatus: 'BLOCKED', mergeable: 'MERGEABLE', threads: [], changedFiles: 3, createdAt: ago(48 * 60), ...o
});

async function runPrefilter(scenarios) {
  const byNumber = new Map(scenarios.map(s => [s.number, s]));
  const writes = []; // ordered log of every REST call that matters to the findings
  const github = {
    rest: {
      search: { issuesAndPullRequests: async () => ({ data: { items: scenarios.map(s => ({ number: s.number })) } }) },
      issues: {
        listEventsForTimeline: async ({ issue_number }) => { writes.push(`timeline#${issue_number}`); return byNumber.get(issue_number).timeline; },
        listComments: async ({ issue_number }) => byNumber.get(issue_number).comments,
        createComment: async (p) => { writes.push(`createComment#${p.issue_number}`); return { data: { id: 1, body: p.body } }; },
        addLabels: async (p) => { writes.push(`addLabels#${p.issue_number}:${p.labels.join(',')}`); return { data: [] }; }
      },
      pulls: {
        get: async ({ pull_number }) => {
          writes.push(`pulls.get#${pull_number}`);
          const s = byNumber.get(pull_number);
          return { data: { number: pull_number, state: s.freshState, draft: s.freshDraft, labels: (s.freshLabels ?? s.labels).map(name => ({ name })) } };
        },
        listCommits: async ({ pull_number }) => byNumber.get(pull_number).prCommits || []
      }
    },
    paginate: (fn, params) => fn(params),
    graphql: async (query, vars) => {
      const s = byNumber.get(vars.number);
      if (query.includes('reviewThreads')) {
        return { repository: { pullRequest: { reviewThreads: { pageInfo: { hasNextPage: false, endCursor: null }, nodes: s.threads } } } };
      }
      return { repository: { pullRequest: {
        number: s.number, title: `PR ${s.number}`, url: `https://github.com/tenstorrent/tt-metal/pull/${s.number}`,
        isDraft: false, state: 'OPEN', createdAt: s.createdAt, updatedAt: ago(30), changedFiles: s.changedFiles,
        mergeStateStatus: s.mergeStateStatus, mergeable: s.mergeable, reviewDecision: null,
        headRefOid: 'a'.repeat(40), headRefName: `copilot/fix-${s.number}`, author: { login: 'copilot-swe-agent' },
        assignees: { nodes: s.assignees.map(login => ({ login })) },
        labels: { nodes: s.labels.map(name => ({ name })) },
        commits: { nodes: [{ commit: { committedDate: s.headCommittedAt,
          statusCheckRollup: { contexts: { pageInfo: { hasNextPage: false, endCursor: null }, nodes: s.checks } } } }] }
      } } };
    }
  };
  const outputs = {}; const logs = { info: [], warning: [] };
  const core = {
    info: (m) => logs.info.push(m), warning: (m) => logs.warning.push(m), setOutput: (k, v) => { outputs[k] = v; },
    summary: { addHeading() { return this; }, addRaw() { return this; }, async write() {} }
  };
  const context = { repo: { owner: 'tenstorrent', repo: 'tt-metal' }, serverUrl: 'https://github.com', runId: 1, eventName: 'schedule', payload: {} };
  delete process.env.SOUS_CHEF_LOGIN; // nudges authored as github-actions[bot], the fallback identity
  const result = await run({ github, context, core });
  const created = writes.filter(w => w.startsWith('createComment')).length;
  const labeled = writes.filter(w => w.startsWith('addLabels')).length;
  return { ...result, outputs, logs, writes, created, labeled };
}

const sixNudgesEndingAt = (lastMinutesAgo) => Array.from({ length: MAX_NUDGES_PER_PR }, (_, i) => nudgeAt(lastMinutesAgo + (MAX_NUDGES_PER_PR - 1 - i) * 90));

test('run(): a failing, never-nudged flow PR is eligible and nothing is written', async () => {
  const r = await runPrefilter([scenario()]);
  assert.equal(r.counters.eligible, 1);
  assert.deepEqual(r.selected.map(p => [p.number, p.needs_nudge, p.nudge_count]), [[1, true, 0]]);
  assert.equal(r.created + r.labeled, 0);
  assert.equal(r.outputs.eligible_numbers, '1');
});

test('run(): a recent copilot_work_started with no finished event is treated as an active session', async () => {
  const r = await runPrefilter([scenario({ timeline: [workEvent('started', SESSION_STALE_MIN - 1)] })]);
  assert.equal(r.counters.filtered_copilot_session_active, 1);
  assert.equal(r.counters.eligible, 0);
});

test('run(): a copilot_work_started older than SESSION_STALE_MS with no finished event is treated as stale, not active (PR #58341)', async () => {
  const r = await runPrefilter([scenario({ timeline: [workEvent('started', SESSION_STALE_MIN + 1)] })]);
  assert.equal(r.counters.filtered_copilot_session_active, 0, 'a stale started event must not block the PR forever');
  assert.equal(r.counters.session_stale_override, 1);
  assert.equal(r.counters.eligible, 1);
  assert.ok(r.logs.info.some(m => m.includes('treating the session as stale')));
});

test('run(): copilot_work_finished after a started event is not affected by the staleness override', async () => {
  const r = await runPrefilter([scenario({ timeline: [workEvent('started', SESSION_STALE_MIN + 100), workEvent('finished', 5)] })]);
  assert.equal(r.counters.filtered_copilot_session_active, 0);
  assert.equal(r.counters.session_stale_override, undefined, 'the override only applies when the LATEST event is a started with no finished after it');
  assert.equal(r.counters.eligible, 1);
});

test('run() finding 1: an UNANSWERED 6th nudge (no push since) reaches the hand-off instead of dying at Filter 4', async () => {
  // Exactly the stuck state: comments[0] is the bot's own unanswered nudge, head unchanged,
  // not conflicting — Filter 4's condition holds — and the cap is reached.
  const r = await runPrefilter([scenario({ comments: sixNudgesEndingAt(COOLDOWN_MIN + 1) })]);
  assert.equal(r.counters.filtered_last_comment_from_sous_chef, 0, 'Filter 4 must not short-circuit a capped PR');
  assert.equal(r.counters.handed_off_now, 1);
  assert.equal(r.counters.eligible, 0, 'a handed-off PR is not also nudged');
  assert.equal(r.created, 1);
  assert.deepEqual(r.writes.filter(w => !w.startsWith('timeline')), ['pulls.get#1', 'createComment#1', `addLabels#1:${HANDOFF_LABEL}`]);
  assert.match(r.reasons[1], /handed off to maintainers after 6 nudges/);
});

test('run() finding 1: below the cap, Filter 4 still suppresses a duplicate nudge', async () => {
  const r = await runPrefilter([scenario({ comments: sixNudgesEndingAt(COOLDOWN_MIN + 1).slice(1) })]); // 5 nudges
  assert.equal(r.counters.filtered_last_comment_from_sous_chef, 1);
  assert.equal(r.counters.handed_off_now + r.counters.eligible + r.created + r.labeled, 0);
});

test('run() finding 1: at the cap but inside the cooldown, the hand-off is deferred, not lost', async () => {
  const r = await runPrefilter([scenario({ comments: sixNudgesEndingAt(10) })]);
  assert.equal(r.counters.filtered_cooldown, 1);
  assert.match(r.reasons[1], /hand-off waits for the 60 min cooldown/);
  assert.equal(r.created + r.labeled, 0);
  // ... and once the cooldown has elapsed the same PR hands off (previous test).
});

test('run() finding 1: a capped PR that has become green is "nothing actionable", not a hand-off', async () => {
  const r = await runPrefilter([scenario({ comments: sixNudgesEndingAt(COOLDOWN_MIN + 1), checks: [greenCheck] })]);
  assert.equal(r.counters.filtered_nothing_actionable, 1);
  assert.equal(r.counters.handed_off_now + r.created + r.labeled, 0);
});

test('run() finding 1: a capped PR whose 6th nudge WAS answered by a push but still fails hands off too', async () => {
  const r = await runPrefilter([scenario({ comments: sixNudgesEndingAt(COOLDOWN_MIN + 30), headCommittedAt: ago(COOLDOWN_MIN + 5) })]);
  assert.equal(r.counters.handed_off_now, 1);
});

test('run() finding 2: label removed + hand-off comment deleted, 6 old nudges still present -> NOT re-handed off; nudge count restarts at 0', async () => {
  const r = await runPrefilter([scenario({
    comments: sixNudgesEndingAt(400),
    timeline: [labelEvent('labeled', HANDOFF_LABEL, 390), labelEvent('unlabeled', HANDOFF_LABEL, 120)]
  })]);
  assert.equal(r.counters.filtered_handed_off, 0);
  assert.equal(r.counters.handed_off_now, 0, 'the documented reset must not re-trigger the hand-off');
  assert.equal(r.created + r.labeled, 0);
  assert.equal(r.counters.eligible, 1);
  assert.equal(r.selected[0].nudge_count, 0);
  assert.ok(r.logs.info.some(m => m.includes(`${HANDOFF_LABEL} was removed at`)), 'baseline is logged');
});

test('run() finding 2: removing the label alone is the reset; the old hand-off comment may stay', async () => {
  const r = await runPrefilter([scenario({
    comments: [handoffAt(380), ...sixNudgesEndingAt(400)],
    timeline: [labelEvent('labeled', HANDOFF_LABEL, 390), labelEvent('unlabeled', HANDOFF_LABEL, 120)]
  })]);
  assert.equal(r.counters.filtered_handed_off, 0, 'a hand-off comment older than the label removal is history');
  assert.equal(r.counters.eligible, 1);
  assert.equal(r.created + r.labeled, 0);
});

test('run() finding 2: after a reset only post-reset nudges count, and a fresh 6 hands off again with the right count', async () => {
  const reset = [labelEvent('labeled', HANDOFF_LABEL, 900), labelEvent('unlabeled', HANDOFF_LABEL, 800)];
  // 6 stale nudges + 2 since the reset; Copilot pushed after the last one, so Filter 4 does not apply.
  const partial = await runPrefilter([scenario({
    comments: [...sixNudgesEndingAt(1000), nudgeAt(300), nudgeAt(COOLDOWN_MIN + 20)],
    headCommittedAt: ago(COOLDOWN_MIN + 10), timeline: reset
  })]);
  assert.equal(partial.counters.eligible, 1);
  assert.equal(partial.selected[0].nudge_count, 2);
  // 6 stale + 6 new, the newest unanswered and past the cooldown -> second hand-off, counting 6, not 12.
  const again = await runPrefilter([scenario({
    comments: [handoffAt(880), ...sixNudgesEndingAt(1000), ...sixNudgesEndingAt(COOLDOWN_MIN + 1)], timeline: reset
  })]);
  assert.equal(again.counters.handed_off_now, 1);
  assert.match(again.reasons[1], /after 6 nudges/);
});

test('run() finding 2: a hand-off comment NEWER than the label removal still counts as handed off', async () => {
  const r = await runPrefilter([scenario({
    comments: [handoffAt(60), ...sixNudgesEndingAt(400)],
    timeline: [labelEvent('unlabeled', HANDOFF_LABEL, 120)]
  })]);
  assert.equal(r.counters.filtered_handed_off, 1);
  assert.equal(r.created + r.labeled, 0);
});

test('run() finding 2: without any label removal the whole history counts (no accidental reset)', async () => {
  const r = await runPrefilter([scenario({ comments: [humanAt(5), ...sixNudgesEndingAt(400)], timeline: [labelEvent('labeled', 'unrelated', 100), labelEvent('unlabeled', FLOW_LABEL, 100)] })]);
  assert.equal(r.counters.handed_off_now, 1);
});

test('run() finding 3: the hand-off write re-reads the PR and skips when copilot-flow is gone', async () => {
  const r = await runPrefilter([scenario({ comments: sixNudgesEndingAt(COOLDOWN_MIN + 1), freshLabels: [] })]);
  assert.equal(r.counters.handoff_skipped_stale_state, 1);
  assert.equal(r.counters.handed_off_now, 0);
  assert.equal(r.created + r.labeled, 0, 'neither the comment nor the label may be written');
  assert.match(r.reasons[1], /kill switch honoured/);
  assert.deepEqual(r.writes.filter(w => !w.startsWith('timeline')), ['pulls.get#1']);
});

test('run() finding 3: the fresh read also catches a hand-off that another run already performed, and a closed/draft PR', async () => {
  const already = await runPrefilter([scenario({ comments: sixNudgesEndingAt(COOLDOWN_MIN + 1), freshLabels: [FLOW_LABEL, HANDOFF_LABEL] })]);
  assert.equal(already.counters.handoff_skipped_stale_state, 1);
  assert.equal(already.created + already.labeled, 0);
  const drafted = await runPrefilter([scenario({ comments: sixNudgesEndingAt(COOLDOWN_MIN + 1), freshDraft: true })]);
  assert.equal(drafted.counters.handoff_skipped_stale_state, 1);
  assert.equal(drafted.created + drafted.labeled, 0);
});

test('run() finding 3: a search hit whose authoritative labels lack copilot-flow is dropped before any evaluation', async () => {
  const r = await runPrefilter([scenario({ labels: [] })]);
  assert.equal(r.counters.filtered_flow_label_missing, 1);
  assert.equal(r.writes.length, 0, 'no timeline/comments/threads calls, no writes');
  assert.equal(r.counters.eligible, 0);
});

test('run(): hand-off and nudge decisions are independent across PRs in one run', async () => {
  const r = await runPrefilter([
    scenario({ number: 11 }),                                                                 // eligible
    scenario({ number: 12, comments: sixNudgesEndingAt(COOLDOWN_MIN + 1) }),                 // hands off
    scenario({ number: 13, comments: sixNudgesEndingAt(COOLDOWN_MIN + 1), freshLabels: [] }) // skipped at write time
  ]);
  assert.equal(r.outputs.eligible_numbers, '11');
  assert.equal(r.counters.handed_off_now, 1);
  assert.equal(r.counters.handoff_skipped_stale_state, 1);
  assert.deepEqual(r.writes.filter(w => w.startsWith('createComment')), ['createComment#12']);
});

// ------------------------------ real sanitizer (opt-in, network) ------------------------------

const lock = fs.readFileSync(path.join(__dirname, '..', '..', 'workflows', 'pr-sous-chef.lock.yml'), 'utf8');
const pin = (lock.match(/gh-aw-actions\/setup@([0-9a-f]{40})/) || [])[1];

test('pr-sous-chef.lock.yml pins gh-aw-actions to a full commit SHA', () => {
  assert.ok(pin, 'expected uses: github/gh-aw-actions/setup@<40-hex sha> in the lock');
});

const fetchText = (url) => new Promise((resolve, reject) => {
  https.get(url, { headers: { 'user-agent': 'tt-metal-pr-sous-chef-test' } }, (res) => {
    if (res.statusCode !== 200) { res.resume(); return reject(new Error(`${url}: HTTP ${res.statusCode}`)); }
    let data = ''; res.setEncoding('utf8'); res.on('data', d => data += d); res.on('end', () => resolve(data));
  }).on('error', reject);
});

test('nudge survives the pinned gh-aw sanitizer; agent-written markers do not', { skip: process.env.PR_SOUS_CHEF_SANITIZER_TEST !== '1' ? 'set PR_SOUS_CHEF_SANITIZER_TEST=1 (needs network)' : false }, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gh-aw-sanitizer-'));
  const files = ['sanitize_content.cjs', 'sanitize_content_core.cjs', 'markdown_code_region_balancer.cjs',
    'repo_helpers.cjs', 'slash_command_matcher.cjs', 'glob_pattern_helpers.cjs', 'error_codes.cjs'];
  for (const f of files) {
    fs.writeFileSync(path.join(dir, f), await fetchText(`https://raw.githubusercontent.com/github/gh-aw-actions/${pin}/setup/js/${f}`));
  }
  global.core = { info() {}, warning() {}, debug() {} };
  const { sanitizeContent } = require(path.join(dir, 'sanitize_content.cjs'));
  const opts = { allowedAliases: ['copilot-swe-agent'] }; // what `mentions.allowed: ["@copilot"]` resolves to
  const agentBody = '<!-- gh-aw-pr-sous-chef-nudge -->\n@copilot the PR gate failed.\n\n- **pr-gate** → run pre-commit\n\nPush and reply here.';
  const out = sanitizeContent(agentBody, opts);
  assert.equal(out.includes('<!-- gh-aw-pr-sous-chef-nudge -->'), false, 'the old agent-emitted marker is stripped (finding #1)');
  assert.equal(out.includes('@copilot'), true, 'the allowed mention survives');
  // an agent cannot forge the handler marker either
  assert.equal(matchesWorkflowId(sanitizeContent(`<!-- gh-aw-workflow-id: ${WORKFLOW_ID} -->\nhi`, opts)), false);
  assert.equal(matchesWorkflowId(sanitizeContent(`<!-- gh-aw-agentic-workflow: x, workflow_id: ${WORKFLOW_ID}, run: y -->\nhi`, opts)), false);
  // ... but the handler's own post-sanitization footer is recognised
  const id = makeIdentity('github-actions[bot]');
  assert.equal(id.isNudge(comment('github-actions[bot]', 'Bot', out + HANDLER_FOOTER)), true);
  assert.equal(id.isNudge(comment('github-actions[bot]', 'Bot', sanitizeContent('Nothing to do right now.', opts) + HANDLER_FOOTER)), false);
});
