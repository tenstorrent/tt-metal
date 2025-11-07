// Reporting Module
// Handles report generation, workflow filtering, alerting, status changes, and enrichment

const core = require('@actions/core');
const { DEFAULT_LOOKBACK_DAYS, EMPTY_VALUE, SUCCESS_EMOJI, FAILURE_EMOJI, SHA_SHORT_LENGTH } = require('./data-loading');
const { getWorkflowStats, fetchPRInfo } = require('./analysis');

/**
 * Generates a GitHub Actions workflow URL.
 *
 * @param {object} context - GitHub Actions context
 * @param {string} workflowFile - Path to the workflow file relative to .github/workflows/
 * @returns {string} Full URL to the workflow in GitHub Actions
 */
function getWorkflowLink(context, workflowFile) {
  return workflowFile // return the workflow link querying on main branch if the workflow file exists
    ? `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/workflows/${workflowFile}?query=branch%3Amain`
    : `https://github.com/${context.repo.owner}/${context.repo.repo}/actions`; // return the github actions link if the workflow file does not exist
}

/**
 * Analyzes scheduled runs to find the last good and earliest bad commits.
 *
 * @param {Array<object>} scheduledMainRuns - Array of scheduled runs on main branch, sorted by date (newest first)
 * @param {object} context - GitHub Actions context
 * @returns {object} Object containing:
 *   - newestGoodSha: Short SHA of the most recent successful run (e.g., `a1b2c3d`)
 *   - newestBadSha: Short SHA of the most recent failing run (e.g., `e4f5g6h`)
 */
function findGoodBadCommits(scheduledMainRuns, context) {
  let newestGoodSha = EMPTY_VALUE;
  let newestBadSha = EMPTY_VALUE;
  let foundGood = false;
  let foundBad = false;

  for (const run of scheduledMainRuns) {
    if (!foundGood && run.conclusion === 'success') { // find the most recent successful run
      const shortSha = run.head_sha.substring(0, SHA_SHORT_LENGTH);
      newestGoodSha = `[\`${shortSha}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${run.head_sha})`; // set the newest good sha to the short sha of the most recent successful run
      foundGood = true;
    }
    if (!foundBad && run.conclusion !== 'success') { // find the most recent failing run
      const shortSha = run.head_sha.substring(0, SHA_SHORT_LENGTH);
      newestBadSha = `[\`${shortSha}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${run.head_sha})`; // set the newest bad sha to the short sha of the most recent failing run
      foundBad = true;
    }
    if (foundGood && foundBad) break; // break the loop if both good and bad runs are found
  }

  return { newestGoodSha, newestBadSha };
}

/**
 * Gets information about the last run on main branch.
 *
 * @param {Array<object>} mainBranchRuns - Array of runs on main branch, sorted by date (newest first)
 * @param {object} context - GitHub Actions context
 * @returns {Promise<object>} Object containing run information
 */
async function getLastRunInfo(mainBranchRuns, context) {
  const lastMainRun = mainBranchRuns[0];
  if (!lastMainRun) { // if there is no last main run, return the empty values
    return {
      status: EMPTY_VALUE,
      sha: EMPTY_VALUE,
      run: EMPTY_VALUE,
      pr: EMPTY_VALUE,
      title: EMPTY_VALUE,
      newestGoodSha: EMPTY_VALUE,
      newestBadSha: EMPTY_VALUE
    };
  }

  const prInfo = await fetchPRInfo(null, context, lastMainRun.head_sha); // fetch the PR info for the last main run
  // Current approach: filter by event type
  const mainRuns = mainBranchRuns.filter(r => r.event === lastMainRun.event || r.event === 'workflow_dispatch'); // get the relevant main runs for finding good and bad commits
  // Alternative approach: include all runs on main branch
  // const mainRuns = mainBranchRuns;
  const { newestGoodSha, newestBadSha } = findGoodBadCommits(mainRuns, context); // find the newest good and bad commits

  return {
    status: lastMainRun.conclusion === 'success' ? SUCCESS_EMOJI : FAILURE_EMOJI,
    sha: `[\`${lastMainRun.head_sha.substring(0, SHA_SHORT_LENGTH)}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${lastMainRun.head_sha})`, // get the short sha hyperlink for the latest main run
    run: `[Run](${lastMainRun.html_url})${lastMainRun.run_attempt > 1 ? ` (#${lastMainRun.run_attempt})` : ''}`, // get the run hyperlink for the latest main run
    pr: prInfo.prNumber, // get the PR number for the latest main run
    title: prInfo.prTitle, // get the PR title for the latest main run
    newestGoodSha, // get the newest good sha for the latest main run
    newestBadSha: lastMainRun.conclusion !== 'success' ? newestBadSha : EMPTY_VALUE // if the last main run is successful, don't bother showing the most recent failure
  };
}

/**
 * Generates summary tables for push and scheduled workflows.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} context - GitHub Actions context
 * @returns {Promise<string>} Markdown table for all workflows
 */
async function generateSummaryBox(grouped, context) {
  const rows = [];
  const escapeHtml = (str) => String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');

  // Helper to convert markdown links to HTML
  const mdToHtml = (md) => {
    if (!md || md === EMPTY_VALUE) return escapeHtml(md);
    // Match markdown link format: [text](url) or [`text`](url)
    return md.replace(/\[`([^`]+)`\]\(([^)]+)\)/g, (_, text, url) => `<a href="${escapeHtml(url)}"><code>${escapeHtml(text)}</code></a>`)
             .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) => `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`);
  };

  for (const [name, runs] of grouped.entries()) {
    // First deduplicate by run ID, keeping highest attempt
    const runsByID = new Map();
    for (const run of runs.filter(r => r.head_branch === 'main')) {
      const runId = run.id;
      const currentAttempt = run.run_attempt || 1;
      const existingRun = runsByID.get(runId);
      const existingAttempt = existingRun ? (existingRun.run_attempt || 1) : 0;
      if (!existingRun || currentAttempt > existingAttempt) {
        runsByID.set(runId, run);
      }
    }
    const mainBranchRuns = Array.from(runsByID.values())
      .sort((a, b) => {
        // Sort by date (newest first), then by run_attempt (highest first) as tiebreaker
        const dateDiff = new Date(b.created_at) - new Date(a.created_at);
        if (dateDiff !== 0) {
          return dateDiff;
        }
        const attemptA = a.run_attempt || 1;
        const attemptB = b.run_attempt || 1;
        return attemptB - attemptA; // Prefer higher attempt number
      });

    const stats = getWorkflowStats(runs); // get the workflow stats for the workflow

    // Skip workflows that only have workflow_dispatch as their event type. Presumably because if they have to be triggered to run they aren't super important
    if (stats.eventTypes === 'workflow_dispatch') {
      continue; // if the workflow only has workflow_dispatch as its event type, skip it
    }

    const workflowLink = getWorkflowLink(context, runs[0]?.path); // get the link to the workflow page that lists all the runs
    const runInfo = await getLastRunInfo(mainBranchRuns, context); // get the last run info for the workflow

    // basically, create the statistics table for the workflow. these are the columns that are displayed in the summary box.
    const row = `<tr>
<td><a href="${escapeHtml(workflowLink)}">${escapeHtml(name)}</a></td>
<td>${escapeHtml(stats.eventTypes || 'unknown')}</td>
<td>${stats.totalRuns}</td>
<td>${stats.successfulRuns}</td>
<td>${escapeHtml(stats.successRate)}</td>
<td>${escapeHtml(stats.uniqueSuccessRate)}</td>
<td>${escapeHtml(stats.retryRate)}</td>
<td>${escapeHtml(runInfo.status)}</td>
<td>${mdToHtml(runInfo.sha)}</td>
<td>${mdToHtml(runInfo.run)}</td>
<td>${mdToHtml(runInfo.pr)}</td>
<td>${escapeHtml(runInfo.title)}</td>
<td>${mdToHtml(runInfo.newestBadSha)}</td>
<td>${mdToHtml(runInfo.newestGoodSha)}</td>
</tr>`;
    rows.push(row);
  }

  return [
    '## Workflow Summary',
    '<table>',
    '<thead>',
    '<tr><th>Workflow</th><th>Event Type(s)</th><th>Total Runs</th><th>Successful Runs</th><th>Success Rate</th><th>Unique Success Rate</th><th>Retry Rate</th><th>Last Run on <code>main</code></th><th>Last SHA</th><th>Last Run</th><th>Last PR</th><th>PR Title</th><th>Newest Bad SHA</th><th>Newest Good SHA</th></tr>',
    '</thead>',
    '<tbody>',
    ...rows,
    '</tbody>',
    '</table>',
    ''
  ].join('\n');
}

/**
 * Builds the complete markdown report.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} context - GitHub Actions context
 * @returns {Promise<string>} Complete markdown report
 */
async function buildReport(grouped, context) {
  const days = parseInt(core.getInput('days') || DEFAULT_LOOKBACK_DAYS, 10); // get the number of days to look back for workflow data
  const timestamp = new Date().toISOString(); // get the timestamp for the report
  return [
    `# Workflow Summary (Last ${days} Days) - Generated at ${timestamp}\n`,
    await generateSummaryBox(grouped, context),
    '\n## Column Descriptions\n',
    '<p>A unique run represents a single workflow execution, which may have multiple retry attempts. For example, if a workflow fails and is retried twice, this counts as one unique run with three attempts (initial run + two retries).</p>\n',
    '\n### Success Rate Calculations\n',
    '<p>The success rates are calculated based on unique runs (not including retries in the denominator):</p>\n',
    '<ul>',
    '<li><strong>Success Rate</strong>: (Number of unique runs that eventually succeeded / Total number of unique runs) × 100%',
    '  <ul><li>Example: 3 successful unique runs out of 5 total unique runs = 60% success rate</li></ul>',
    '</li>',
    '<li><strong>Unique Success Rate</strong>: (Number of unique runs that succeeded on first try / Total number of unique runs) × 100%',
    '  <ul><li>Example: 1 unique run succeeded on first try out of 5 total unique runs = 20% unique success rate</li></ul>',
    '</li>',
    '<li><strong>Retry Rate</strong>: (Number of successful unique runs that needed retries / Total number of successful unique runs) × 100%',
    '  <ul><li>Example: 2 successful unique runs needed retries out of 3 total successful unique runs = 66.67% retry rate</li></ul>',
    '</li>',
    '</ul>\n',
    '<p><strong>Note:</strong> Unique Success Rate + Retry Rate does not equal 100% because they measure different things:</p>',
    '<ul>',
    '<li>Unique Success Rate is based on all unique runs</li>',
    '<li>Retry Rate is based only on successful unique runs</li>',
    '</ul>\n',
    '<table>',
    '<thead>',
    '<tr><th>Column</th><th>Description</th></tr>',
    '</thead>',
    '<tbody>',
    '<tr><td>Workflow</td><td>Name of the workflow with link to its GitHub Actions page</td></tr>',
    '<tr><td>Event Type(s)</td><td>Types of events that trigger this workflow (e.g., push, pull_request, schedule)</td></tr>',
    '<tr><td>Total Runs</td><td>Total number of workflow runs including all retry attempts (e.g., 1 unique run with 2 retries = 3 total runs)</td></tr>',
    '<tr><td>Successful Runs</td><td>Number of unique workflow runs that eventually succeeded, regardless of whether they needed retries</td></tr>',
    '<tr><td>Success Rate</td><td>Percentage of unique workflow runs that eventually succeeded (e.g., 3/5 unique runs succeeded = 60%)</td></tr>',
    '<tr><td>Unique Success Rate</td><td>Percentage of unique workflow runs that succeeded on their first attempt without needing retries (e.g., 1/5 unique runs succeeded on first try = 20%)</td></tr>',
    '<tr><td>Retry Rate</td><td>Percentage of successful unique runs that needed retries to succeed (e.g., of 3 successful unique runs, 2 needed retries = 66.67%)</td></tr>',
    '<tr><td>Last Run on <code>main</code></td><td>Status of the most recent run on the main branch (✅ for success, ❌ for failure)</td></tr>',
    '<tr><td>Last SHA</td><td>Short SHA of the most recent run on main</td></tr>',
    '<tr><td>Last Run</td><td>Link to the most recent run on main, with attempt number if applicable</td></tr>',
    '<tr><td>Last PR</td><td>Link to the PR associated with the most recent run, if any</td></tr>',
    '<tr><td>PR Title</td><td>Title of the PR associated with the most recent run, if any</td></tr>',
    '<tr><td>Newest Bad SHA</td><td>Short SHA of the most recent failing run on main (only shown if last run failed)</td></tr>',
    '<tr><td>Newest Good SHA</td><td>Short SHA of the most recent successful run on main</td></tr>',
    '</tbody>',
    '</table>'
  ].join('\n');
}

/**
 * Filters workflow runs by date range
 * @param {Array<Object>} runs - Array of workflow runs
 * @param {number} days - Number of days to look back
 * @returns {Array<Object>} Filtered runs within the date range
 */
function filterRunsByDate(runs, days) {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - days); // set the cutoff date to the number of days ago

  return runs.filter(run => {
    const runDate = new Date(run.created_at);
    return runDate >= cutoffDate; // return the runs that are within the date range
  });
}

module.exports = {
  getWorkflowLink,
  findGoodBadCommits,
  getLastRunInfo,
  generateSummaryBox,
  buildReport,
  filterRunsByDate,
};
