'use strict';

// Backtests the sanity-tests-slack-notify decision logic (culprits.js)
// against real, already-settled "Sanity tests" push-to-main history.
//
//   GITHUB_TOKEN=$(gh auth token) node .github/scripts/sanity-notify/backtest.js [--days 30]
//
// For every historical run that satisfied the workflow's finality gate
// (a "final red state"), the REAL findBaseline() is invoked against the
// live API -- with maxWaitMs: 0 and a sleep() that throws, since closed
// history is already final and the poll path must never be needed. Any
// event that would still want to poll therefore resolves to a 'timeout'
// abstain and is flagged for human eyes.
//
// Each decision is also compared against a naive time-ordered expectation
// (previous distinct-sha run in the window was green -> expect notify),
// which approximates what a human reading the run list would conclude.
// Mismatches are listed individually; they are exactly the interesting
// cases (merge-queue reordering, duplicate runs, out-of-window parents).

const { findBaseline } = require('./culprits.js');

const token = process.env.GITHUB_TOKEN;
if (!token) {
  console.error('GITHUB_TOKEN is required (e.g. GITHUB_TOKEN=$(gh auth token) node backtest.js)');
  process.exit(1);
}

const args = process.argv.slice(2);
const argVal = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : dflt;
};
const DAYS = Number(argVal('--days', '30'));
const [OWNER, REPO] = argVal('--repo', 'tenstorrent/tt-metal').split('/');
const WORKFLOW_ID = 'sanity-tests.yaml';

let apiCalls = 0;
async function api(path, params = {}) {
  const qs = new URLSearchParams(params).toString();
  const url = `https://api.github.com${path}${qs ? '?' + qs : ''}`;
  apiCalls += 1;
  const res = await fetch(url, {
    headers: {
      authorization: `Bearer ${token}`,
      accept: 'application/vnd.github+json',
      'user-agent': 'sanity-notify-backtest',
      'x-github-api-version': '2022-11-28',
    },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}: ${await res.text()}`);
  return res.json();
}

// Minimal octokit-compatible client covering exactly what culprits.js uses.
const github = {
  rest: {
    git: {
      getCommit: async ({ owner, repo, commit_sha }) =>
        ({ data: await api(`/repos/${owner}/${repo}/git/commits/${commit_sha}`) }),
    },
    repos: {
      listPullRequestsAssociatedWithCommit: async ({ owner, repo, commit_sha }) =>
        ({ data: await api(`/repos/${owner}/${repo}/commits/${commit_sha}/pulls`) }),
    },
    actions: {
      listWorkflowRuns: async ({ owner, repo, workflow_id, ...params }) =>
        ({ data: await api(`/repos/${owner}/${repo}/actions/workflows/${workflow_id}/runs`, params) }),
    },
  },
};

// The workflow's top-level finality gate, applied to a historical run.
const isGatePassingRed = (r) =>
  r.conclusion !== null &&
  r.conclusion !== 'success' &&
  (r.conclusion !== 'failure' || r.run_attempt >= 3);

async function main() {
  const until = new Date();
  const since = new Date(until.getTime() - DAYS * 24 * 60 * 60 * 1000);
  const created = `${since.toISOString().slice(0, 10)}..${until.toISOString().slice(0, 10)}`;

  // 1. Collect the window's push-to-main run history (same filters as prod).
  // The list endpoint silently caps any filtered query at 1000 results, so
  // fetch in small date chunks and dedupe on run id (chunk edges overlap by
  // a day since the `created` filter has date granularity).
  const CHUNK_DAYS = 4;
  const byId = new Map();
  for (let t = since.getTime(); t < until.getTime(); t += CHUNK_DAYS * 24 * 60 * 60 * 1000) {
    const chunkEnd = Math.min(t + CHUNK_DAYS * 24 * 60 * 60 * 1000, until.getTime());
    const chunkCreated = `${new Date(t).toISOString().slice(0, 10)}..${new Date(chunkEnd).toISOString().slice(0, 10)}`;
    let fetched = 0;
    for (let page = 1; ; page++) {
      const data = await api(`/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW_ID}/runs`, {
        branch: 'main', event: 'push', per_page: 100, page, created: chunkCreated,
      });
      for (const r of data.workflow_runs) byId.set(r.id, r);
      fetched += data.workflow_runs.length;
      if (data.workflow_runs.length < 100) break;
    }
    if (fetched >= 1000) console.error(`WARNING: chunk ${chunkCreated} hit the 1000-result cap; shrink CHUNK_DAYS.`);
  }
  const allRuns = [...byId.values()].sort((a, b) => new Date(a.created_at) - new Date(b.created_at));

  // 2. Gate-passing red events, deduped per head_sha (the idempotency marker
  //    allows at most one notification per broken sha in production).
  const redBySha = new Map();
  for (const r of allRuns.filter(isGatePassingRed)) {
    if (!redBySha.has(r.head_sha)) redBySha.set(r.head_sha, r);
  }

  console.log(`window: ${created} (${DAYS} days)`);
  console.log(`push-to-main runs fetched: ${allRuns.length}`);
  console.log(`gate-passing final-red events (unique head_sha): ${redBySha.size}`);
  console.log('');

  // Naive time-ordered expectation: was the previous distinct-sha run green?
  const naiveExpectation = (redRun) => {
    const idx = allRuns.indexOf(redRun);
    for (let i = idx - 1; i >= 0; i--) {
      if (allRuns[i].head_sha === redRun.head_sha) continue;
      if (['cancelled', 'skipped'].includes(allRuns[i].conclusion)) continue;
      return allRuns[i].conclusion === 'success' ? 'notify' : 'abstain';
    }
    return 'unknown (window edge)';
  };

  // 3. Run every red event through the REAL decision logic.
  const tally = {};
  const rows = [];
  const flagged = [];
  for (const [sha, redRun] of redBySha) {
    const result = await findBaseline({
      github,
      owner: OWNER,
      repo: REPO,
      headSha: sha,
      maxWaitMs: 0, // closed history must decide instantly; anything undecided -> 'timeout'
      sleep: async () => { throw new Error(`sleep() reached for settled history (sha ${sha})`); },
    });

    let key = result.decision === 'notify' ? 'notify' : `abstain:${result.reason}`;
    // 'parent-skip-ci' is now detected upfront by findBaseline itself (the
    // accepted-gap short-circuit; in production it abstains instantly with
    // no polling). With maxWaitMs: 0, any remaining 'timeout' collapses two
    // distinct situations; split them, since for settled history they mean
    // very different things:
    //  - parent has NO push run and is NOT [skip ci] (in production this
    //    would poll the full budget and abstain) -- unexpected, flag it.
    //  - parent has a run that never settled (e.g. a failure@1 whose
    //    auto-retry never fired) -- in production this also polls the full
    //    budget; evidence the retry workflow itself can fail.
    if (result.reason === 'timeout') {
      key = result.parentRuns.length === 0
        ? 'abstain:parent-has-no-push-run'
        : 'abstain:parent-never-settled';
    }
    tally[key] = (tally[key] ?? 0) + 1;

    const expected = naiveExpectation(redRun);
    const agree = expected.startsWith('unknown') ? 'n/a' : (expected === result.decision ? 'yes' : 'NO');
    const row = {
      run: `#${redRun.run_number}`,
      created: redRun.created_at,
      conclusion: `${redRun.conclusion}@${redRun.run_attempt}`,
      sha: sha.slice(0, 9),
      decision: key,
      baseline: result.baselineRun ? `#${result.baselineRun.run_number}` : '-',
      timeOrderExpected: expected,
      agree,
    };
    rows.push(row);
    // parent-skip-ci is a known, accepted category (see BACKTEST.md) -- it
    // is tallied but not flagged.
    if (agree === 'NO' || key === 'abstain:parent-has-no-push-run' ||
        key === 'abstain:parent-never-settled' || key === 'abstain:no-parent') flagged.push(row);
    console.log(`${row.run} ${row.created} ${row.conclusion} sha=${row.sha} -> ${row.decision}` +
      `${result.baselineRun ? ` (baseline ${row.baseline})` : ''}  [time-order expects: ${expected}${agree === 'NO' ? ' <-- MISMATCH' : ''}]`);
  }

  console.log('\n=== summary ===');
  for (const [k, v] of Object.entries(tally).sort()) console.log(`${k}: ${v}`);
  console.log(`api calls: ${apiCalls}`);
  if (flagged.length) {
    console.log(`\n=== flagged for human review (${flagged.length}) ===`);
    for (const f of flagged) console.log(JSON.stringify(f));
  } else {
    console.log('\nno mismatches or undecided-history cases flagged');
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
