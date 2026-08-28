'use strict';

// Unit tests for culprits.js (the decision logic behind
// sanity-tests-slack-notify.yaml). Zero dependencies -- run with:
//
//   node --test .github/scripts/sanity-notify/
//
// Timers are injected (now/sleep), so multi-hour poll scenarios run
// instantly against the exact production code path.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const {
  isFinal,
  findBaseline,
  buildCulpritEntries,
  escapeSlack,
  DEFAULT_POLL_INTERVAL_MS,
  DEFAULT_MAX_WAIT_MS,
} = require('./culprits.js');

// ----------------------------- helpers -----------------------------

const run = (n, conclusion, attempt, status = 'completed') => ({
  run_number: n,
  status,
  conclusion,
  run_attempt: attempt,
  html_url: `https://example.test/runs/${n}`,
});

// Builds a harness around findBaseline: fake clock, scripted sequence of
// listWorkflowRuns responses (last one repeats), captured logs/outputs.
// parentMessage/parentPulls feed the [skip ci] short-circuit check.
function harness({
  parents = [{ sha: 'PARENT' }],
  parentMessage = 'ordinary commit subject\n\nbody',
  parentPulls = [],
  responses,
  ...overrides
} = {}) {
  let fakeNow = 0;
  let lookups = 0;
  const logs = [];
  const github = {
    rest: {
      git: {
        getCommit: async ({ commit_sha }) => commit_sha === 'CURRENT'
          ? { data: { parents, message: 'current commit subject' } }
          : { data: { parents: [{ sha: 'GRANDPARENT' }], message: parentMessage } },
      },
      repos: {
        listPullRequestsAssociatedWithCommit: async ({ commit_sha }) => {
          assert.equal(commit_sha, 'PARENT'); // skip-ci check inspects the parent only
          return { data: parentPulls };
        },
      },
      actions: {
        listWorkflowRuns: async (params) => {
          assert.equal(params.head_sha, 'PARENT'); // always the exact parent sha, never a recency scan
          assert.equal(params.event, 'push');
          const r = responses[Math.min(lookups, responses.length - 1)];
          lookups += 1;
          return { data: { workflow_runs: r } };
        },
      },
    },
  };
  const call = () => findBaseline({
    github,
    owner: 'o',
    repo: 'r',
    headSha: 'CURRENT',
    now: () => fakeNow,
    sleep: async (ms) => { fakeNow += ms; },
    log: (m) => logs.push(m),
    warn: (m) => logs.push('WARN: ' + m),
    ...overrides,
  });
  return { call, logs, elapsed: () => fakeNow, lookups: () => lookups };
}

// ----------------------------- isFinal -----------------------------

test('isFinal: success is final at any attempt', () => {
  assert.equal(isFinal(run(1, 'success', 1)), true);
});

test('isFinal: failure is final only once the retry budget is exhausted', () => {
  assert.equal(isFinal(run(1, 'failure', 1)), false);
  assert.equal(isFinal(run(1, 'failure', 2)), false);
  assert.equal(isFinal(run(1, 'failure', 3)), true);
  assert.equal(isFinal(run(1, 'failure', 4)), true); // manual re-run past the budget
});

test('isFinal: non-failure terminal conclusions are final immediately', () => {
  for (const c of ['cancelled', 'skipped', 'timed_out', 'startup_failure']) {
    assert.equal(isFinal(run(1, c, 1)), true, c);
  }
});

test('isFinal: a run that is not completed is never final', () => {
  assert.equal(isFinal(run(1, null, 1, 'in_progress')), false);
  assert.equal(isFinal(run(1, null, 2, 'queued')), false);
});

// --------------------------- findBaseline ---------------------------

test('settled green parent: notifies immediately with zero waiting', async () => {
  const h = harness({ responses: [[run(100, 'success', 1)]] });
  const result = await h.call();
  assert.equal(result.decision, 'notify');
  assert.equal(result.baselineRun.run_number, 100);
  assert.equal(result.parentSha, 'PARENT');
  assert.equal(h.lookups(), 1);
  assert.equal(h.elapsed(), 0);
});

test('a confirmed success decides it even with a pending sibling run', async () => {
  const h = harness({ responses: [[run(100, 'success', 1), run(101, null, 1, 'in_progress')]] });
  const result = await h.call();
  assert.equal(result.decision, 'notify');
  assert.equal(h.elapsed(), 0);
});

test('any success among many duplicate run entries wins', async () => {
  const h = harness({
    responses: [[run(100, 'failure', 3), run(101, 'cancelled', 1), run(102, 'success', 1)]],
  });
  const result = await h.call();
  assert.equal(result.decision, 'notify');
  assert.equal(result.baselineRun.run_number, 102);
});

test('settled red parent (failure at max attempts): abstains immediately', async () => {
  const h = harness({ responses: [[run(100, 'failure', 3)]] });
  const result = await h.call();
  assert.equal(result.decision, 'abstain');
  assert.equal(result.reason, 'conclusively-red');
  assert.equal(h.elapsed(), 0);
});

test('settled cancelled/skipped parent: abstains immediately (observed, never green)', async () => {
  for (const c of ['cancelled', 'skipped']) {
    const h = harness({ responses: [[run(100, c, 1)]] });
    const result = await h.call();
    assert.equal(result.decision, 'abstain', c);
    assert.equal(result.reason, 'conclusively-red', c);
  }
});

test('parent mid-retry (failure attempt 1) polls instead of abstaining, then abstains once conclusively red', async () => {
  const h = harness({
    responses: [
      [run(100, 'failure', 1)],            // completed but NOT final: retry may be coming
      [run(100, null, 2, 'in_progress')],  // retry running
      [run(100, 'failure', 3)],            // budget exhausted: final red
    ],
  });
  const result = await h.call();
  assert.equal(result.decision, 'abstain');
  assert.equal(result.reason, 'conclusively-red');
  assert.equal(h.lookups(), 3);
  assert.equal(h.elapsed(), 2 * DEFAULT_POLL_INTERVAL_MS);
});

test('parent mid-retry that eventually succeeds: notifies', async () => {
  const h = harness({
    responses: [
      [run(100, 'failure', 1)],
      [run(100, 'success', 2)],
    ],
  });
  const result = await h.call();
  assert.equal(result.decision, 'notify');
  assert.equal(h.lookups(), 2);
});

test('no run ever appears: abstains on timeout after the full wait budget', async () => {
  const h = harness({ responses: [[]] });
  const result = await h.call();
  assert.equal(result.decision, 'abstain');
  assert.equal(result.reason, 'timeout');
  // Sleeps only while elapsed + interval <= budget.
  assert.equal(h.elapsed(), Math.floor(DEFAULT_MAX_WAIT_MS / DEFAULT_POLL_INTERVAL_MS) * DEFAULT_POLL_INTERVAL_MS);
  assert.ok(h.logs.at(-1).startsWith('WARN: Timed out'));
});

test('maxWaitMs of zero never sleeps: undecided history resolves to timeout instantly (backtest mode)', async () => {
  const h = harness({
    responses: [[run(100, 'failure', 1)]],
    maxWaitMs: 0,
    sleep: async () => { throw new Error('sleep must not be called'); },
  });
  const result = await h.call();
  assert.equal(result.decision, 'abstain');
  assert.equal(result.reason, 'timeout');
  assert.equal(h.lookups(), 1);
});

test('root commit (no parent): abstains without any run lookup', async () => {
  const h = harness({ parents: [], responses: [[run(100, 'success', 1)]] });
  const result = await h.call();
  assert.equal(result.decision, 'abstain');
  assert.equal(result.reason, 'no-parent');
  assert.equal(h.lookups(), 0);
});

// ----------------- [skip ci] parent short-circuit (accepted gap) -----------------

test('[skip ci] in the parent commit message: abstains immediately, no run lookup, no sleep', async () => {
  for (const msg of ['[skip ci] docs tweak', 'prefix [SKIP CI] suffix', '[ci skip] chore']) {
    const h = harness({
      parentMessage: msg,
      responses: [[run(100, 'success', 1)]],
      sleep: async () => { throw new Error('sleep must not be called'); },
    });
    const result = await h.call();
    assert.equal(result.decision, 'abstain', msg);
    assert.equal(result.reason, 'parent-skip-ci', msg);
    assert.equal(h.lookups(), 0, msg); // never even queried the parent's runs
    assert.match(h.logs.at(-1), /by policy it carries no baseline signal .*Abstaining immediately without polling/);
  }
});

test('[skip ci] only in an associated PR title of the parent: same immediate abstain', async () => {
  const h = harness({
    parentMessage: 'ordinary subject with no marker',
    parentPulls: [
      { number: 1, title: 'normal PR' },
      { number: 2, title: '[skip ci] Update README' },
    ],
    responses: [[run(100, 'success', 1)]],
    sleep: async () => { throw new Error('sleep must not be called'); },
  });
  const result = await h.call();
  assert.equal(result.decision, 'abstain');
  assert.equal(result.reason, 'parent-skip-ci');
  assert.equal(h.lookups(), 0);
});

test('no run found and parent is NOT [skip ci]: still falls through to poll-then-timeout', async () => {
  const h = harness({
    parentMessage: 'a real change whose retry workflow silently died',
    parentPulls: [{ number: 3, title: 'a real PR title' }],
    responses: [[]],
  });
  const result = await h.call();
  assert.equal(result.decision, 'abstain');
  assert.equal(result.reason, 'timeout'); // NOT parent-skip-ci; the stuck-retry category is unaffected
  assert.equal(h.elapsed(), Math.floor(DEFAULT_MAX_WAIT_MS / DEFAULT_POLL_INTERVAL_MS) * DEFAULT_POLL_INTERVAL_MS);
});

// ------------------------- buildCulpritEntries -------------------------

function compareGithub({ commits, pullsBySha }) {
  return {
    rest: {
      repos: {
        compareCommits: async () => ({ data: { commits } }),
        listPullRequestsAssociatedWithCommit: async ({ commit_sha }) =>
          ({ data: pullsBySha[commit_sha] ?? [] }),
      },
    },
  };
}

const commitObj = (sha, subject, login) => ({
  sha,
  html_url: `https://example.test/commit/${sha}`,
  commit: { message: `${subject}\n\nbody` },
  author: login ? { login } : null,
});

test('culprits: PR-associated commit yields one deduped PR entry per PR', async () => {
  const github = compareGithub({
    commits: [commitObj('C1', 'subj', 'alice')],
    pullsBySha: {
      C1: [
        { number: 7, html_url: 'u7', title: 'Fix <thing> & stuff', user: { login: 'alice' } },
        { number: 7, html_url: 'u7', title: 'dup delivery', user: { login: 'alice' } },
        { number: 9, html_url: 'u9', title: 'other PR containing commit', user: { login: 'bob' } },
      ],
    },
  });
  const { entries, logins } = await buildCulpritEntries({ github, owner: 'o', repo: 'r', baseSha: 'B', headSha: 'H' });
  assert.equal(entries.length, 2);
  assert.deepEqual(entries.map((e) => e.number), [7, 9]);
  assert.equal(entries[0].title, 'Fix &lt;thing&gt; &amp; stuff'); // Slack-escaped
  assert.deepEqual(logins, ['alice', 'bob']);
});

test('culprits: commit with zero associated PRs falls back to the commit author', async () => {
  const github = compareGithub({
    commits: [commitObj('C1', 'direct push <oops>', 'carol')],
    pullsBySha: {},
  });
  const { entries, logins } = await buildCulpritEntries({ github, owner: 'o', repo: 'r', baseSha: 'B', headSha: 'H' });
  assert.equal(entries.length, 1);
  assert.equal(entries[0].type, 'commit');
  assert.equal(entries[0].title, 'direct push &lt;oops&gt;');
  assert.deepEqual(logins, ['carol']);
});

test('culprits: null authors are kept as entries but excluded from mention logins', async () => {
  const github = compareGithub({
    commits: [commitObj('C1', 'subj', null)],
    pullsBySha: {},
  });
  const { entries, logins } = await buildCulpritEntries({ github, owner: 'o', repo: 'r', baseSha: 'B', headSha: 'H' });
  assert.equal(entries.length, 1);
  assert.equal(entries[0].authorLogin, null);
  assert.deepEqual(logins, []);
});

test('escapeSlack neutralizes mrkdwn metacharacters', () => {
  assert.equal(escapeSlack('<!channel> & <http://x|y>'), '&lt;!channel&gt; &amp; &lt;http://x|y&gt;');
});
