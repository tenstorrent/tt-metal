'use strict';

// Decision logic for .github/workflows/sanity-tests-slack-notify.yaml.
//
// ==================== ONE-COMMIT-BACK EDGE DETECTION ====================
// Only the immediate git parent of the failing commit is ever consulted
// (main is linear via merge queue). If any of the parent's push runs
// finally succeeded, main was green immediately before this commit -- so
// this commit broke it. If the parent conclusively never succeeded, or
// never settles within the wait budget, abstain -- never guess further
// back.
//
// The parent's runs are resolved by exact head_sha -- a point lookup keyed
// by the sha we ask for -- never by position in a recency-sorted
// listWorkflowRuns page, which is not guaranteed to be a complete or
// current window. A lagging index can make the point lookup MISS
// (-> keep polling, eventually abstain); it can never return the wrong
// commit's runs. The diff is always exactly one commit (parent..current),
// which structurally bounds who one message can blame.
//
// All functions take an octokit-compatible client (`github`) plus plain
// params, and findBaseline takes injectable `now`/`sleep`/`log`/`warn`,
// so the exact same code runs in the workflow (via actions/github-script),
// in unit tests (fake timers), and in backtests (no waiting).

// Must stay in sync with MAX_ATTEMPTS in _auto-retry-post-commit.yaml.
const MAX_ATTEMPTS = 3;

const DEFAULT_WORKFLOW_ID = 'sanity-tests.yaml';
const DEFAULT_POLL_INTERVAL_MS = 5 * 60 * 1000;
const DEFAULT_MAX_WAIT_MS = 270 * 60 * 1000; // caller's job timeout minus headroom for later steps

// Mirrors the notify workflow's own top-level finality gate, applied to a
// PARENT run: success is always final; 'failure' is only final once the
// auto-retry budget is exhausted -- before that, a retry may still be
// queued even though the attempt already shows status 'completed'. Every
// other conclusion is final immediately, since only 'failure' triggers a
// retry at all.
function isFinal(run) {
  return run.status === 'completed' &&
    (run.conclusion !== 'failure' || run.run_attempt >= MAX_ATTEMPTS);
}

function describeRuns(runs) {
  return runs.length === 0
    ? 'no runs found'
    : runs.map((r) => `#${r.run_number}=${r.conclusion ?? r.status}(attempt ${r.run_attempt})`).join(', ');
}

// Resolves whether the commit at `headSha` freshly broke main.
//
// Decision per poll of the parent's push runs (a single head_sha can have
// several push runs -- duplicate deliveries have produced two success runs
// for one sha -- so ALL of them are fetched):
//  - any FINAL success               -> { decision: 'notify' } immediately.
//  - all found runs final,
//    none succeeded                  -> { decision: 'abstain', reason: 'conclusively-red' } immediately.
//  - anything else (no runs indexed
//    yet, or some run not yet final) -> sleep and re-poll until maxWaitMs
//                                       -> { decision: 'abstain', reason: 'timeout' }.
// A commit with no parent aborts with { decision: 'abstain', reason: 'no-parent' }.
// A [skip ci] parent aborts upfront with { decision: 'abstain', reason: 'parent-skip-ci' }
// (see the comment at that check).
//
// Known, ACCEPTED gap (maintainer decision, 2026-08-21): headSha's OWN runs
// are NOT settled here -- the triggering run's final-red conclusion (already
// vetted by the caller's finality gate) is taken as-is. Duplicate push
// deliveries can give one sha several runs, so in theory a sibling run of
// headSha could have succeeded (or still be running), making a notification
// on this commit a false blame. Deliberately not handled: the 30-day
// backtest observed exactly one duplicate-run pair and its outcomes matched
// (success/success), never a split outcome. Revisit if a bad notification is
// ever traced to a green sibling of the blamed sha.
async function findBaseline({
  github,
  owner,
  repo,
  headSha,
  workflowId = DEFAULT_WORKFLOW_ID,
  branch = 'main',
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  maxWaitMs = DEFAULT_MAX_WAIT_MS,
  now = Date.now,
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  log = () => {},
  warn = () => {},
}) {
  const { data: commit } = await github.rest.git.getCommit({ owner, repo, commit_sha: headSha });
  const parentSha = commit.parents[0]?.sha;
  if (!parentSha) {
    warn(`${headSha} has no parent commit; skipping.`);
    return { decision: 'abstain', reason: 'no-parent', parentSha: null, parentRuns: [] };
  }

  // Known, ACCEPTED gap (maintainer decision, 2026-08-21): a [skip ci]
  // parent usually gets no push run, so its greenness is unknowable under
  // the one-commit-back rule, and the break on top of it deliberately goes
  // un-notified. By policy the marker alone decides -- even in the odd case
  // where a marked commit does have a run (it happens; see BACKTEST.md) --
  // so abstain immediately instead of spending the poll budget (~8% of red
  // events have a [skip ci] parent per the backtest).
  const SKIP_CI_RE = /\[(skip ci|ci skip)\]/i;
  const { data: parentCommit } = await github.rest.git.getCommit({ owner, repo, commit_sha: parentSha });
  let skipCiSource = SKIP_CI_RE.test(parentCommit.message) ? 'commit message' : null;
  if (!skipCiSource) {
    const { data: parentPulls } = await github.rest.repos.listPullRequestsAssociatedWithCommit({
      owner,
      repo,
      commit_sha: parentSha,
    });
    if (parentPulls.some((pr) => SKIP_CI_RE.test(pr.title))) skipCiSource = 'associated PR title';
  }
  if (skipCiSource) {
    log(`Parent ${parentSha} is marked [skip ci] (${skipCiSource}); by policy it carries no baseline signal ` +
      '(accepted gap, see BACKTEST.md). Abstaining immediately without polling.');
    return { decision: 'abstain', reason: 'parent-skip-ci', parentSha, parentRuns: [] };
  }

  // head_sha is an exact-match filter (requires the full 40-char sha); the
  // event/branch filters exclude the merge_group run of the same sha.
  const fetchParentRuns = async () => {
    const { data: resp } = await github.rest.actions.listWorkflowRuns({
      owner,
      repo,
      workflow_id: workflowId,
      head_sha: parentSha,
      branch,
      event: 'push',
      per_page: 10,
    });
    return resp.workflow_runs;
  };

  const startedAtMs = now();
  for (;;) {
    const parentRuns = await fetchParentRuns();

    const baselineRun = parentRuns.find((r) => isFinal(r) && r.conclusion === 'success');
    if (baselineRun) { // a confirmed success decides it; never wait on sibling runs
      log(`Parent ${parentSha} was green (run #${baselineRun.run_number}); ${headSha} broke main.`);
      return { decision: 'notify', reason: 'parent-green', parentSha, baselineRun, parentRuns };
    }

    if (parentRuns.length > 0 && parentRuns.every(isFinal)) {
      log(`Parent ${parentSha} conclusively never succeeded (${describeRuns(parentRuns)}); main may already be red. Abstaining.`);
      return { decision: 'abstain', reason: 'conclusively-red', parentSha, parentRuns };
    }

    const elapsedMs = now() - startedAtMs;
    const elapsedMin = Math.round(elapsedMs / 60000);
    if (elapsedMs + pollIntervalMs > maxWaitMs) {
      warn(`Timed out after ${elapsedMin} min waiting for parent ${parentSha} to settle (${describeRuns(parentRuns)}); abstaining.`);
      return { decision: 'abstain', reason: 'timeout', parentSha, parentRuns };
    }
    log(`[${elapsedMin} min elapsed] Parent ${parentSha} not settled yet (${describeRuns(parentRuns)}); polling again in ${pollIntervalMs / 60000} min.`);
    await sleep(pollIntervalMs);
  }
}

// PR titles and commit subjects are attacker-controlled (anyone who can
// open a PR) and end up inside Slack `<url|label>` mrkdwn links -- they
// must be entity-escaped the same way notify-slack-on-mention.yaml escapes
// user-supplied text, or a crafted title could break out of the link
// syntax or smuggle in a broadcast mention like <!subteam^...>.
function escapeSlack(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Builds the culprit entry list for the baseSha..headSha range (exactly one
// commit under the one-commit-back rule, but written against the compare
// API so it stays correct regardless). PRs are deduped by number; a commit
// with no associated PR (direct push, or API lag right after merge) falls
// back to the commit's own author so a culprit is never silently dropped.
async function buildCulpritEntries({ github, owner, repo, baseSha, headSha }) {
  const { data: cmp } = await github.rest.repos.compareCommits({
    owner,
    repo,
    base: baseSha,
    head: headSha,
  });

  const entries = [];
  const seenPrNumbers = new Set();

  for (const commit of cmp.commits) {
    const { data: pulls } = await github.rest.repos.listPullRequestsAssociatedWithCommit({
      owner,
      repo,
      commit_sha: commit.sha,
    });

    if (pulls.length > 0) {
      for (const pr of pulls) {
        if (seenPrNumbers.has(pr.number)) continue;
        seenPrNumbers.add(pr.number);
        entries.push({
          type: 'pr',
          number: pr.number,
          url: pr.html_url,
          title: escapeSlack(pr.title),
          authorLogin: pr.user ? pr.user.login : null,
        });
      }
    } else {
      entries.push({
        type: 'commit',
        sha: commit.sha,
        url: commit.html_url,
        title: escapeSlack(commit.commit.message.split('\n')[0]),
        authorLogin: commit.author ? commit.author.login : null,
      });
    }
  }

  const logins = [...new Set(entries.map((e) => e.authorLogin).filter((l) => l !== null))].sort();
  return { entries, logins };
}

module.exports = {
  MAX_ATTEMPTS,
  DEFAULT_WORKFLOW_ID,
  DEFAULT_POLL_INTERVAL_MS,
  DEFAULT_MAX_WAIT_MS,
  isFinal,
  describeRuns,
  escapeSlack,
  findBaseline,
  buildCulpritEntries,
};
