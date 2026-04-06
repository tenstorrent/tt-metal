// Analysis Module
// Handles statistics, workflow analysis, and commit analysis

const github = require('@actions/github');
const { SUCCESS_RATE_DECIMAL_PLACES, EMPTY_VALUE, getCommitDescription, SHA_SHORT_LENGTH } = require('./data-loading');

/**
 * Calculates statistics for a set of workflow runs.
 *
 * @param {Array<object>} runs - Array of workflow run objects
 * @returns {object} Statistics object containing:
 *   - eventTypes: Comma-separated string of unique event types
 *   - successRate: Percentage of runs that succeeded (based on last attempt)
 *   - uniqueSuccessRate: Percentage of runs that succeeded on first attempt
 *   - retryRate: Percentage of successful runs that required retries
 *   - uniqueRuns: Number of unique runs (excluding attempts)
 *   - totalRuns: Total number of runs including attempts
 *   - successfulRuns: Number of runs that succeeded (based on last attempt)
 */
function getWorkflowStats(runs) {
  // Group runs by their original run ID to handle retries and reruns
  const uniqueRuns = new Map();
  let totalRunsIncludingRetries = 0;
  let totalSuccessfulUniqueRuns = 0;
  let successfulUniqueRunsOnFirstTry = 0;
  let successfulUniqueRunsWithRetries = 0;

  // First pass: identify all unique runs and their attempts
  for (const run of runs) {
    totalRunsIncludingRetries++;

    // GitHub keeps the same run ID for re-runs, only incrementing run_attempt
    // So we group by run.id directly, not by calculating a different ID
    const runId = run.id;
    const currentAttempt = run.run_attempt || 1;

    if (!uniqueRuns.has(runId)) {
      // First time seeing this run ID
      uniqueRuns.set(runId, {
        run,
        attempts: 1, // Initialize to 1 since this is the first attempt
        isSuccessful: false,
        requiredRetry: false,
        succeededOnFirstTry: false,
        lastAttempt: run
      });
    } else {
      // This is a re-run (same run ID, different attempt)
      const existingRun = uniqueRuns.get(runId);
      existingRun.attempts++; // increment the attempts counter

      // Update last attempt if this is a newer attempt (higher run_attempt number)
      const existingAttempt = existingRun.lastAttempt.run_attempt || 1;
      if (currentAttempt > existingAttempt) {
        existingRun.lastAttempt = run; // update to the latest attempt
      }
    }
  }

  // Second pass: determine final status based on last attempt
  for (const runInfo of uniqueRuns.values()) { // iterate through the unique runs
    const lastAttempt = runInfo.lastAttempt;
    if (lastAttempt.conclusion === 'success') { // if the last attempt is successful
      runInfo.isSuccessful = true;
      totalSuccessfulUniqueRuns++; // increment the total successful unique runs

      if (lastAttempt.run_attempt === 1) { // if the last attempt is the first attempt
        runInfo.succeededOnFirstTry = true; // set the succeeded on first try to true
        successfulUniqueRunsOnFirstTry++; // increment the total successful unique runs on first try
      } else {
        runInfo.requiredRetry = true; // set the required retry to true
        successfulUniqueRunsWithRetries++; // increment the total successful unique runs with retries
      }
    }
  }

  const uniqueRunsArray = Array.from(uniqueRuns.values()).map(r => r.run); // creates an array of the run objects
  const eventTypes = [...new Set(uniqueRunsArray.map(r => r.event))].join(', '); // create a string of all the event types that were used to trigger runs

  // Calculate rates
  const successRate = uniqueRunsArray.length === 0 ? "N/A" : (totalSuccessfulUniqueRuns / uniqueRunsArray.length * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";
  const uniqueSuccessRate = uniqueRunsArray.length === 0 ? "N/A" : (successfulUniqueRunsOnFirstTry / uniqueRunsArray.length * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";
  const retryRate = totalSuccessfulUniqueRuns === 0 ? "N/A" : (successfulUniqueRunsWithRetries / totalSuccessfulUniqueRuns * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";

  return { // return the eventypes, success rates, retry rates, and total run info
    eventTypes,
    successRate,
    uniqueSuccessRate,
    retryRate,
    uniqueRuns: uniqueRunsArray.length,
    totalRuns: totalRunsIncludingRetries,
    successfulRuns: totalSuccessfulUniqueRuns
  };
}

/**
 * Finds, within the provided window (newest→oldest, main branch only), either:
 * - the first failing run since the most recent success (oldest in the current failing streak), or
 * - if no success exists in-window, the oldest failing run in the window.
 * Returns null if there are no failing runs in the window.
 *
 * @param {Array<object>} mainBranchRunsWindow - Runs on main, sorted by created_at desc (newest first)
 * @returns {{run: object, noSuccessInWindow: boolean}|null}
 */
function findFirstFailInWindow(mainBranchRunsWindow) {
  let seenAnyFailure = false;
  let firstFailInStreak = null; // oldest failure observed before crossing a success boundary

  for (const run of mainBranchRunsWindow) {
    if (run.conclusion === 'success') {
      if (firstFailInStreak) {
        // We found a success after observing failures: return the oldest failure in the streak
        return { run: firstFailInStreak, boundarySuccessRun: run, noSuccessInWindow: false };
      }
      // Success encountered before any failure in the current scan; keep scanning older entries
    } else if (run.conclusion && run.conclusion !== 'cancelled' && run.conclusion !== 'skipped') {
      // Treat anything non-success, non-cancelled/skipped as failure for this purpose
      seenAnyFailure = true;
      firstFailInStreak = run; // update to become oldest failure within the current failing streak (and window)
    }
  }

  if (seenAnyFailure) {
    // No success found in-window; report oldest failure in the window
    return { run: firstFailInStreak, boundarySuccessRun: undefined, noSuccessInWindow: true };
  }
  return null;
}

function renderCommitsTable(commits) {
  if (!Array.isArray(commits) || commits.length === 0) {
    return '<p><em>None</em></p>';
  }
  const escapeHtml = (str) => String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  const rows = commits.map(c => {
    const short = c.short || (c.sha ? c.sha.substring(0, SHA_SHORT_LENGTH) : '');
    const url = c.url || (c.sha ? `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${c.sha}` : undefined);
    const who = c.author_login ? `@${c.author_login}` : (c.author_name || 'unknown');
    const whoHtml = (c.author_login && c.author_url)
      ? `<a href="${escapeHtml(c.author_url)}">@${escapeHtml(c.author_login)}</a>`
      : escapeHtml(who);
    const shaHtml = url ? `<a href="${escapeHtml(url)}"><code>${escapeHtml(short)}</code></a>` : `<code>${escapeHtml(short)}</code>`;
    const descHtml = escapeHtml(getCommitDescription(c));
    return `<tr><td>${shaHtml}</td><td>${whoHtml}</td><td>${descHtml}</td></tr>`;
  }).join('\n');
  return [
    '<table>',
    '<thead>',
    '<tr><th>SHA</th><th>Author</th><th>Description</th></tr>',
    '</thead>',
    '<tbody>',
    rows,
    '</tbody>',
    '</table>',
    ''
  ].join('\n');
}

/**
 * Fetches PR information associated with a commit.
 *
 * @param {object} context - GitHub Actions context
 * @param {string} commitSha - Full SHA of the commit to look up
 * @returns {Promise<object>} Object containing:
 *   - prNumber: Markdown link to the PR (e.g., [#123](url))
 *   - prTitle: Title of the PR or EMPTY_VALUE if not found
 *   - prAuthor: GitHub username of the PR author or 'unknown'
 */
// Disabled: PR fetching via GitHub API removed for offline analysis
async function fetchPRInfo(_github, _context, _commitSha) {
  return { prNumber: EMPTY_VALUE, prTitle: EMPTY_VALUE, prAuthor: EMPTY_VALUE };
}

/**
 * Fetch commit author info for a commit SHA.
 * Returns GitHub login (if associated), author display name, and profile URL (if available).
 */
// Disabled: commit author fetch via GitHub API; will be inferred from commits index if present
async function fetchCommitAuthor(_commitSha) {
  return { login: undefined, name: undefined, htmlUrl: undefined };
}

module.exports = {
  getWorkflowStats,
  findFirstFailInWindow,
  renderCommitsTable,
  fetchPRInfo,
  fetchCommitAuthor,
};
