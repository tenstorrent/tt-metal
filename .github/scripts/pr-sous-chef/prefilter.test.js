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
  WORKFLOW_ID, HANDOFF_MARKER, MAX_NUDGES_PER_PR,
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
