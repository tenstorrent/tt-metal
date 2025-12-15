// Reporting Module
// Handles report generation, workflow filtering, alerting, status changes, and enrichment

const core = require('@actions/core');
const github = require('@actions/github');
const { DEFAULT_LOOKBACK_DAYS, EMPTY_VALUE, SUCCESS_EMOJI, FAILURE_EMOJI, SHA_SHORT_LENGTH, DEFAULT_INFRA_OWNER, getTimeSinceLastSuccess } = require('./data-loading');
const { getWorkflowStats, fetchPRInfo, findFirstFailInWindow, fetchCommitAuthor, renderCommitsTable } = require('./analysis');
const { fetchErrorSnippetsForRun, inferJobAndTestFromSnippet, resolveOwnersForSnippet, findOwnerForLabel, renderErrorsTable } = require('./error-processing');
const { getAnnotationsDirForRunId, listCommitsBetweenOffline } = require('./data-loading');

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
 * Counts total failing jobs from regressions and stayed failing sections
 * @param {Array} regressedDetails - Array of regression detail objects
 * @param {Array} stayedFailingDetails - Array of stayed failing detail objects
 * @returns {number} Total number of failing jobs
 */
function countTotalFailingJobs(regressedDetails, stayedFailingDetails) {
  let total = 0;

  // Count from regressions
  for (const item of regressedDetails) {
    if (Array.isArray(item.failing_jobs)) {
      total += item.failing_jobs.length;
    }
  }

  // Count from stayed failing
  for (const item of stayedFailingDetails) {
    if (Array.isArray(item.failing_jobs)) {
      total += item.failing_jobs.length;
    }
  }

  return total;
}

/**
 * Builds the complete markdown report.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} context - GitHub Actions context
 * @param {number} totalFailingJobs - Total number of failing jobs (optional)
 * @returns {Promise<string>} Complete markdown report
 */
async function buildReport(grouped, context, totalFailingJobs = 0) {
  const days = parseInt(core.getInput('days') || DEFAULT_LOOKBACK_DAYS, 10); // get the number of days to look back for workflow data
  const timestamp = new Date().toISOString(); // get the timestamp for the report

  const failingJobsHeader = totalFailingJobs > 0
    ? `\n## Total Failing Jobs\n\n**Total failing jobs across all workflows: ${totalFailingJobs}**\n\n`
    : '';

  return [
    `# Workflow Summary (Last ${days} Days) - Generated at ${timestamp}\n`,
    failingJobsHeader,
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

/**
 * Filters workflows by date and tracks failed workflows
 * Processes all workflows since we only fetch runs for workflows we care about
 * @param {Map} grouped - Map of workflow names to their runs
 * @param {Map} previousGrouped - Map of previous workflow names to their runs (can be null)
 * @param {number} days - Number of days to look back
 * @returns {Object} Object containing filteredGrouped, filteredPreviousGrouped, and failedWorkflows
 */
function filterWorkflowsByConfig(grouped, previousGrouped, days) {
  const filteredGrouped = new Map();
  const filteredPreviousGrouped = new Map();
  const failedWorkflows = [];
  const hasPrevious = previousGrouped && Array.isArray(previousGrouped);

  // Process all workflows (we only fetched runs for workflows we care about)
  for (const [name, runs] of grouped) {
    // Filter runs by date range
    const filteredRuns = filterRunsByDate(runs, days);
    if (filteredRuns.length > 0) {
      filteredGrouped.set(name, filteredRuns);

      // Check if latest run on main is failing
      // First deduplicate by run ID, keeping highest attempt
      const runsByID = new Map();
      for (const run of filteredRuns.filter(r => r.head_branch === 'main')) {
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
      if (mainBranchRuns[0]?.conclusion !== 'success') {
        failedWorkflows.push(name);
      }
    }
  }

  if (hasPrevious) {
    for (const [name, runs] of previousGrouped) {
      const filteredRuns = filterRunsByDate(runs, days);
      if (filteredRuns.length > 0) {
        filteredPreviousGrouped.set(name, filteredRuns);
      }
    }
  }

  return { filteredGrouped, filteredPreviousGrouped, failedWorkflows };
}

/**
 * Builds a Slack-ready alert message for failing workflows
 * @param {Map} filteredGrouped - Map of workflow names to their runs
 * @param {Array} failedWorkflows - Array of workflow names that are failing
 * @param {boolean} alertAll - Whether to ping owners in the message
 * @param {Map} errorSnippetsCache - Cache for error snippets (will be populated)
 * @returns {Promise<string>} Alert message string (empty if no failures)
 */
async function buildAlertMessage(filteredGrouped, failedWorkflows, alertAll, errorSnippetsCache) {
  if (failedWorkflows.length === 0) {
    return '';
  }

  const mention = (owners) => {
    const arr = Array.isArray(owners) ? owners : (owners ? [owners] : []);
    const parts = arr.map(o => {
      if (!o || !o.id) return '';
      const id = String(o.id);
      if (id.startsWith('S')) {
        const fallback = o.name ? `@${o.name}` : '@team';
        return `<!subteam^${id}|${fallback}>`;
      }
      return `<@${id}>`;
    }).filter(Boolean);
    return parts.length ? parts.join(' ') : '';
  };

  const failingItems = [];
  for (const [name, runs] of filteredGrouped.entries()) {
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
    const mainRuns = Array.from(runsByID.values())
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
    if (!mainRuns[0] || mainRuns[0].conclusion === 'success') continue;

    // Use the latest failing run for snippet-based owner detection
    const latestFail = mainRuns.find(r => r.conclusion !== 'success');
    let owners = undefined;
    let combinedOwnerNames = [];
    let failingJobNames = undefined;
    try {
      const errs = await fetchErrorSnippetsForRun(
        latestFail.id,
        20,
        undefined,
        getAnnotationsDirForRunId(latestFail.id)
      );
      errorSnippetsCache.set(latestFail.id, errs);
      // Infer job/test and resolve owners per snippet, then aggregate
      const ownerSet = new Map();
      const genericExitOrigOwners = new Map();
      const isGenericExit = (s) => typeof s === 'string' && /^Process completed with exit code 1\.?$/i.test(String(s).trim());
      for (const sn of (errs || [])) {
        const inferred = inferJobAndTestFromSnippet(sn);
        if (inferred) { sn.job = inferred.job; sn.test = inferred.test; }
        resolveOwnersForSnippet(sn, name);
        if (Array.isArray(sn.owner)) {
          for (const o of sn.owner) {
            if (!o) continue;
            const k = `${o.id || ''}|${o.name || ''}`;
            ownerSet.set(k, o);
          }
        }
        // Always include original pipeline owners if they exist (even when infra is assigned)
        if (Array.isArray(sn.original_owners)) {
          for (const oo of sn.original_owners) {
            const nm = (oo && (oo.name || oo.id)) || '';
            if (nm) genericExitOrigOwners.set(nm, true);
            if (oo) {
              const k2 = `${oo.id || ''}|${oo.name || ''}`;
              ownerSet.set(k2, { id: oo.id, name: oo.name });
            }
          }
        }
      }
      owners = Array.from(ownerSet.values());
      const origNames = Array.from(genericExitOrigOwners.keys());
      combinedOwnerNames = (() => {
        const seen = new Map();
        for (const o of (owners || [])) {
          const nm = (o && (o.name || o.id)) || '';
          if (nm) seen.set(nm, true);
        }
        for (const nm of origNames) { if (nm) seen.set(nm, true); }
        return Array.from(seen.keys());
      })();
      // Extract failing job names from error snippets
      failingJobNames = [];
      const jobs = new Set();
      for (const sn of (errs || [])) {
        const jobName = (sn && sn.job) ? String(sn.job) : '';
        if (jobName) jobs.add(jobName);
      }
      failingJobNames = Array.from(jobs);
    } catch (_) { /* ignore */ }
    if (!failingJobNames) failingJobNames = [];
    // Fallback: try to resolve owners from the workflow name
    if (!owners || owners.length === 0) {
      owners = findOwnerForLabel(name) || [DEFAULT_INFRA_OWNER];
    }
    const ownerNamesText = (() => {
      const names = Array.isArray(combinedOwnerNames) ? combinedOwnerNames : [];
      return (names.length ? names.join(', ') : DEFAULT_INFRA_OWNER.name);
    })();
    const fallbackMention = `<!subteam^${DEFAULT_INFRA_OWNER.id}|${DEFAULT_INFRA_OWNER.name}>`;
    const ownerMentions = alertAll ? (mention(owners) || fallbackMention) : ownerNamesText;
    const jobsNote = failingJobNames.length > 0 ? ` (failed ${failingJobNames.join(', ')})` : '';
    const wfUrl = getWorkflowLink(github.context, runs[0]?.path);
    failingItems.push(`• ${name} ${wfUrl ? `<${wfUrl}|open>` : ''} ${ownerMentions}${jobsNote}`.trim());
  }

  if (failingItems.length) {
    return [
      '*Alerts: failing workflows on main*',
      ...failingItems
    ].join('\n');
  }

  return '';
}

/**
 * Helper function to deduplicate runs by run ID, keeping highest attempt, and sort by date
 * @param {Array} runs - Array of workflow runs
 * @returns {Array} Deduplicated and sorted runs
 */
function deduplicateAndSortRuns(runs) {
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
  return Array.from(runsByID.values())
    .sort((a, b) => {
      const dateDiff = new Date(b.created_at) - new Date(a.created_at);
      if (dateDiff !== 0) {
        return dateDiff;
      }
      const attemptA = a.run_attempt || 1;
      const attemptB = b.run_attempt || 1;
      return attemptB - attemptA;
    });
}

/**
 * Computes the latest conclusion for a set of runs
 * @param {Array} runs - Array of workflow runs
 * @returns {string|null} 'success', 'failure', or null
 */
function computeLatestConclusion(runs) {
  const mainBranchRuns = deduplicateAndSortRuns(runs);
  const latest = mainBranchRuns[0];
  if (!latest) return null;
  return latest.conclusion === 'success' ? 'success' : 'failure';
}

/**
 * Gets the latest run info for a set of runs
 * @param {Array} runs - Array of workflow runs
 * @returns {Object|null} Run info object or null
 */
function computeLatestRunInfo(runs) {
  const mainBranchRuns = deduplicateAndSortRuns(runs);
  const latest = mainBranchRuns[0];
  if (!latest) return null;
  return { id: latest.id, url: latest.html_url, created_at: latest.created_at, head_sha: latest.head_sha, path: latest.path };
}

/**
 * Gets main branch runs within the current window, deduplicated and sorted
 * @param {Array} runs - Array of workflow runs
 * @returns {Array} Deduplicated and sorted main branch runs
 */
function getMainWindowRuns(runs) {
  return deduplicateAndSortRuns(runs);
}

/**
 * Computes status changes between current and previous workflow runs
 * @param {Map} filteredGrouped - Map of current workflow names to their runs
 * @param {Map} filteredPreviousGrouped - Map of previous workflow names to their runs
 * @param {object} context - GitHub Actions context
 * @returns {Object} Object containing changes, regressedDetails, and stayedFailingDetails arrays
 */
function computeStatusChanges(filteredGrouped, filteredPreviousGrouped, context) {
  const allNames = new Set([
    ...Array.from(filteredGrouped.keys()),
    ...Array.from(filteredPreviousGrouped.keys())
  ]);

  const changes = [];
  const regressedDetails = [];
  const stayedFailingDetails = [];

  for (const name of allNames) {
    const currentRuns = filteredGrouped.get(name);
    const previousRuns = filteredPreviousGrouped.get(name);
    if (!currentRuns || !previousRuns) continue;

    const current = computeLatestConclusion(currentRuns);
    const previous = computeLatestConclusion(previousRuns);
    if (!current || !previous) continue;

    let change;
    if (previous === 'success' && current === 'success') change = 'stayed_succeeding';
    else if (previous !== 'success' && current !== 'success') change = 'stayed_failing';
    else if (previous !== 'success' && current === 'success') change = 'fail_to_success';
    else if (previous === 'success' && current !== 'success') change = 'success_to_fail';

    if (change) {
      const info = computeLatestRunInfo(currentRuns);
      const workflowUrl = info?.path ? getWorkflowLink(context, info.path) : undefined;
      const aggregateRunUrl = `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;
      const commitUrl = info?.head_sha ? `https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${info.head_sha}` : undefined;
      const commitShort = info?.head_sha ? info.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;

      changes.push({
        name,
        previous,
        current,
        change,
        run_id: info?.id,
        run_url: info?.url,
        created_at: info?.created_at,
        workflow_url: workflowUrl,
        workflow_path: info?.path,
        aggregate_run_url: aggregateRunUrl,
        commit_sha: info?.head_sha,
        commit_short: commitShort,
        commit_url: commitUrl
      });
      if (change === 'success_to_fail' && info) {
        regressedDetails.push({
          name,
          run_id: info.id,
          run_url: info.url,
          created_at: info.created_at,
          workflow_url: workflowUrl,
          workflow_path: info.path,
          aggregate_run_url: aggregateRunUrl,
          commit_sha: info.head_sha,
          commit_short: commitShort,
          commit_url: commitUrl,
          owners: []
        });
      } else if (change === 'stayed_failing' && info) {
        stayedFailingDetails.push({
          name,
          run_id: info.id,
          run_url: info.url,
          created_at: info.created_at,
          workflow_url: workflowUrl,
          workflow_path: info.path,
          aggregate_run_url: aggregateRunUrl,
          commit_sha: info.head_sha,
          commit_short: commitShort,
          commit_url: commitUrl
        });
      }
    }
  }

  return { changes, regressedDetails, stayedFailingDetails };
}

/**
 * Enriches regression details with first failing run, commits, authors, and error snippets
 * @param {Array} regressedDetails - Array of regression detail objects (modified in place)
 * @param {Map} filteredGrouped - Map of workflow names to their runs
 * @param {Map} errorSnippetsCache - Cache for error snippets (will be populated)
 * @param {Array} changes - Array of change objects (will be updated with enrichment data)
 * @param {object} context - GitHub Actions context
 */
async function enrichRegressions(regressedDetails, filteredGrouped, errorSnippetsCache, changes, context) {
  for (const item of regressedDetails) {
    try {
      const windowRuns = getMainWindowRuns(filteredGrouped.get(item.name) || []);
      const res = findFirstFailInWindow(windowRuns);
      if (res && res.run) {
        item.first_failed_run_id = res.run.id;
        item.first_failed_run_url = res.run.html_url;
        item.first_failed_created_at = res.run.created_at;
        item.first_failed_head_sha = res.run.head_sha;
        item.first_failed_head_short = res.run.head_sha ? res.run.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;
        item.no_success_in_window = !!res.noSuccessInWindow;
        if (!res.noSuccessInWindow && res.boundarySuccessRun && res.boundarySuccessRun.head_sha) {
          item.commits_between = listCommitsBetweenOffline(context, res.boundarySuccessRun.head_sha, item.first_failed_head_sha);
        }
        if (item.first_failed_head_sha) {
          const author = await fetchCommitAuthor(item.first_failed_head_sha);
          item.first_failed_author_login = author.login;
          item.first_failed_author_name = author.name;
          item.first_failed_author_url = author.htmlUrl;
        }
        if (item.run_id) {
          item.error_snippets = errorSnippetsCache.get(item.run_id) || await fetchErrorSnippetsForRun(
            item.run_id,
            Number.POSITIVE_INFINITY,
            undefined,
            getAnnotationsDirForRunId(item.run_id)
          );
          if (!errorSnippetsCache.has(item.run_id)) {
            errorSnippetsCache.set(item.run_id, item.error_snippets);
          }
        } else {
          item.error_snippets = [];
        }
        try {
          for (const sn of (item.error_snippets || [])) {
            const inferred = inferJobAndTestFromSnippet(sn);
            if (inferred) { sn.job = inferred.job; sn.test = inferred.test; }
            resolveOwnersForSnippet(sn, item.name);
          }
          const ownerSet = new Map();
          const genericExitOrigOwners = new Map();
          const isGenericExit = (s) => typeof s === 'string' && /^Process completed with exit code 1\.?$/i.test(String(s).trim());
          for (const sn of (item.error_snippets || [])) {
            if (Array.isArray(sn.owner)) {
              for (const o of sn.owner) {
                if (!o) continue;
                const k = `${o.id || ''}|${o.name || ''}`;
                ownerSet.set(k, o);
              }
            }
            // Always include original pipeline owners if they exist (even when infra is assigned)
            if (Array.isArray(sn.original_owners)) {
              for (const oo of sn.original_owners) {
                const nm = (oo && (oo.name || oo.id)) || '';
                if (nm) genericExitOrigOwners.set(nm, true);
                if (oo) {
                  const k2 = `${oo.id || ''}|${oo.name || ''}`;
                  ownerSet.set(k2, { id: oo.id, name: oo.name });
                }
              }
            }
          }
          let owners = Array.from(ownerSet.values());
          if (!owners.length) {
            owners = findOwnerForLabel(item.name) || [DEFAULT_INFRA_OWNER];
          }
          item.owners = owners;
          item.original_owner_names_for_generic_exit = Array.from(genericExitOrigOwners.keys());
          const failingJobNames = (() => {
            const jobs = new Set();
            for (const sn of (item.error_snippets || [])) {
              const jobName = (sn && sn.job) ? String(sn.job) : '';
              if (jobName) jobs.add(jobName);
            }
            return Array.from(jobs);
          })();
          item.failing_jobs = failingJobNames;
        } catch (_) { /* ignore */ }

        item.repeated_errors = [];
        const changeRef = changes.find(c => c.name === item.name && c.change === 'success_to_fail');
        if (changeRef) {
          Object.assign(changeRef, {
            first_failed_run_id: item.first_failed_run_id,
            first_failed_run_url: item.first_failed_run_url,
            first_failed_created_at: item.first_failed_created_at,
            first_failed_head_sha: item.first_failed_head_sha,
            first_failed_head_short: item.first_failed_head_short,
            no_success_in_window: item.no_success_in_window,
            first_failed_author_login: item.first_failed_author_login,
            first_failed_author_name: item.first_failed_author_name,
            first_failed_author_url: item.first_failed_author_url,
            commits_between: item.commits_between || [],
            error_snippets: item.error_snippets || [],
            repeated_errors: item.repeated_errors || [],
            failing_jobs: item.failing_jobs || [],
            owners: item.owners || [],
            original_owner_names_for_generic_exit: item.original_owner_names_for_generic_exit || [],
          });
        }
      }
    } catch (e) {
      core.warning(`Failed to find first failing run for ${item.name}: ${e.message}`);
    }
  }
}

/**
 * Enriches stayed failing details with first failing run, commits, authors, and error snippets
 * @param {Array} stayedFailingDetails - Array of stayed failing detail objects (modified in place)
 * @param {Map} filteredGrouped - Map of workflow names to their runs
 * @param {Map} errorSnippetsCache - Cache for error snippets (will be populated)
 * @param {Array} changes - Array of change objects (will be updated with enrichment data)
 * @param {object} context - GitHub Actions context
 */
async function enrichStayedFailing(stayedFailingDetails, filteredGrouped, errorSnippetsCache, changes, context) {
  for (const item of stayedFailingDetails) {
    try {
      const windowRuns = getMainWindowRuns(filteredGrouped.get(item.name) || []);
      const res = findFirstFailInWindow(windowRuns);
      if (res && res.run) {
        item.first_failed_run_id = res.run.id;
        item.first_failed_run_url = res.run.html_url;
        item.first_failed_created_at = res.run.created_at;
        item.first_failed_head_sha = res.run.head_sha;
        item.first_failed_head_short = res.run.head_sha ? res.run.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;
        item.no_success_in_window = !!res.noSuccessInWindow;
        if (!item.no_success_in_window && res.boundarySuccessRun && res.boundarySuccessRun.head_sha) {
          item.commits_between = listCommitsBetweenOffline(context, res.boundarySuccessRun.head_sha, item.first_failed_head_sha);
        }
        if (item.first_failed_head_sha) {
          const author = await fetchCommitAuthor(item.first_failed_head_sha);
          item.first_failed_author_login = author.login;
          item.first_failed_author_name = author.name;
          item.first_failed_author_url = author.htmlUrl;
        }
        if (item.run_id) {
          item.error_snippets = errorSnippetsCache.get(item.run_id) || await fetchErrorSnippetsForRun(
            item.run_id,
            Number.POSITIVE_INFINITY,
            undefined,
            getAnnotationsDirForRunId(item.run_id)
          );
          if (!errorSnippetsCache.has(item.run_id)) {
            errorSnippetsCache.set(item.run_id, item.error_snippets);
          }
        } else {
          item.error_snippets = [];
        }
        try {
          for (const sn of (item.error_snippets || [])) {
            const inferred = inferJobAndTestFromSnippet(sn);
            if (inferred) { sn.job = inferred.job; sn.test = inferred.test; }
            resolveOwnersForSnippet(sn, item.name);
          }
          // Extract failing job names from error snippets
          const failingJobNames = (() => {
            const jobs = new Set();
            for (const sn of (item.error_snippets || [])) {
              const jobName = (sn && sn.job) ? String(sn.job) : '';
              if (jobName) jobs.add(jobName);
            }
            return Array.from(jobs);
          })();
          item.failing_jobs = failingJobNames;
        } catch (_) { /* ignore */ }
        if (!item.failing_jobs) item.failing_jobs = [];
        item.repeated_errors = [];
      }
      const changeRef = changes.find(c => c.name === item.name && c.change === 'stayed_failing');
      if (changeRef) {
        Object.assign(changeRef, {
          first_failed_run_id: item.first_failed_run_id,
          first_failed_run_url: item.first_failed_run_url,
          first_failed_created_at: item.first_failed_created_at,
          first_failed_head_sha: item.first_failed_head_sha,
          first_failed_head_short: item.first_failed_head_short,
          no_success_in_window: item.no_success_in_window,
          first_failed_author_login: item.first_failed_author_login,
          first_failed_author_name: item.first_failed_author_name,
          first_failed_author_url: item.first_failed_author_url,
          commits_between: item.commits_between || [],
          error_snippets: item.error_snippets || [],
          repeated_errors: item.repeated_errors || [],
          failing_jobs: item.failing_jobs || [],
        });
      }
    } catch (e) {
      core.warning(`Failed to find first failing run for ${item.name}: ${e.message}`);
    }
  }
}

function buildWorkflowBadge(workflowPath, timeSinceSuccess) {
  const workflowFileName = workflowPath
    ? workflowPath
      .replace(/^\.github\/workflows\//, '')
      .replace(/\.ya?ml$/i, '')
    : '';
  const badgeParts = [];
  if (timeSinceSuccess !== EMPTY_VALUE) {
    badgeParts.push(`Last success: ${timeSinceSuccess}`);
  }
  if (workflowFileName) {
    badgeParts.push(`workflow file: ${workflowFileName}`);
  }
  return badgeParts.length ? ` <em>(${badgeParts.join(', ')})</em>` : '';
}

/**
 * Builds the regressions section of the report
 * @param {Array} regressedDetails - Array of enriched regression detail objects
 * @param {object} context - GitHub Actions context
 * @returns {string} Markdown section for regressions
 */
function buildRegressionsSection(regressedDetails, context) {
  if (regressedDetails.length === 0) {
    return ['', '## Regressions (Pass → Fail)', '- None', ''].join('\n');
  }

  const lines = regressedDetails.map(it => {
    const workflowName = it.workflow_url ? `<a href="${it.workflow_url}">${it.name}</a>` : it.name;
    const timeSinceSuccess = getTimeSinceLastSuccess(it.name);
    const timeBadge = buildWorkflowBadge(it.workflow_path, timeSinceSuccess);

    if (it.first_failed_run_url) {
      const sha = it.first_failed_head_short || (it.first_failed_head_sha ? it.first_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
      const shaLink = sha ? `[\`${sha}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${it.first_failed_head_sha})` : '';
      const when = it.first_failed_created_at ? new Date(it.first_failed_created_at).toISOString() : '';
      const author = it.first_failed_author_login
        ? `by [@${it.first_failed_author_login}](${it.first_failed_author_url})`
        : (it.first_failed_author_name ? `by ${it.first_failed_author_name}` : '');

      let errorsList = '';
      const errorsHtml = renderErrorsTable(it.error_snippets || []);
      errorsList = [errorsHtml, ''].join('\n');

      if (it.no_success_in_window) {
        const latestWhenIso = it.created_at ? new Date(it.created_at).toISOString() : '';
        const latestShaShort = it.commit_short || (it.commit_sha ? it.commit_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
        const latestShaLink = (latestShaShort && it.commit_sha)
          ? ` [\`${latestShaShort}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${it.commit_sha})`
          : '';
        const latestLine = it.run_url
          ? ` | Latest failing run: [Run](${it.run_url}) ${latestWhenIso}${latestShaLink}`
          : '';
        const content = `  - Failed to find any successful run in the last two weeks. Oldest failing run is: [Run](${it.first_failed_run_url}) ${when} ${shaLink}${latestLine}`;
        return ['<details>', `<summary>${workflowName}${timeBadge}</summary>`, '', content, errorsList, '</details>', ''].join('\n');
      }

      let commitsList = '';
      const commitsMd = renderCommitsTable(it.commits_between || []);
      commitsList = [commitsMd, ''].join('\n');

      const latestWhenIso = it.created_at ? new Date(it.created_at).toISOString() : '';
      const latestShaShort = it.commit_short || (it.commit_sha ? it.commit_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
      const latestShaLink = (latestShaShort && it.commit_sha)
        ? ` [\`${latestShaShort}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${it.commit_sha})`
        : '';
      const latestLine = it.run_url
        ? `\n  - Latest failing run: [Run](${it.run_url}) ${latestWhenIso}${latestShaLink}`
        : '';
      const content = `  - First failing run on main: [Run](${it.first_failed_run_url}) ${when} ${shaLink} ${author}${latestLine}`;
      return ['<details>', `<summary>${workflowName}${timeBadge}</summary>`, '', content, errorsList, commitsList, '</details>', ''].join('\n');
    }
    return ['<details>', `<summary>${workflowName}${timeBadge}</summary>`, '', '  - No failure details available', '</details>', ''].join('\n');
  });

  return ['', '## Regressions (Pass → Fail)', ...lines, ''].join('\n');
}

/**
 * Builds the stayed failing section of the report
 * @param {Array} stayedFailingDetails - Array of enriched stayed failing detail objects
 * @param {object} context - GitHub Actions context
 * @returns {string} Markdown section for stayed failing workflows
 */
function buildStayedFailingSection(stayedFailingDetails, context) {
  if (stayedFailingDetails.length === 0) {
    return ['', '## Still Failing (No Recovery)', '- None', ''].join('\n');
  }

  const lines = stayedFailingDetails.map(it => {
    const workflowName = it.workflow_url ? `<a href="${it.workflow_url}">${it.name}</a>` : it.name;
    const timeSinceSuccess = getTimeSinceLastSuccess(it.name);
    const timeBadge = buildWorkflowBadge(it.workflow_path, timeSinceSuccess);

    if (it.first_failed_run_url) {
      const sha = it.first_failed_head_short || (it.first_failed_head_sha ? it.first_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
      const shaLink = sha ? `[\`${sha}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${it.first_failed_head_sha})` : '';
      const when = it.first_failed_created_at ? new Date(it.first_failed_created_at).toISOString() : '';

      let errorsList = '';
      const errorsHtml2 = renderErrorsTable(it.error_snippets || []);
      errorsList = [errorsHtml2, ''].join('\n');

      if (it.no_success_in_window) {
        const latestWhenIso = it.created_at ? new Date(it.created_at).toISOString() : '';
        const latestShaShort = it.commit_short || (it.commit_sha ? it.commit_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
        const latestShaLink = (latestShaShort && it.commit_sha)
          ? ` [\`${latestShaShort}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${it.commit_sha})`
          : '';
        const latestLine = it.run_url
          ? ` | Latest failing run: [Run](${it.run_url}) ${latestWhenIso}${latestShaLink}`
          : '';
        const content = `  - Failed to find any successful run in the last two weeks. Oldest failing run is: [Run](${it.first_failed_run_url}) ${when} ${shaLink}${latestLine}`;
        return ['<details>', `<summary>${workflowName}${timeBadge}</summary>`, '', content, errorsList, '</details>', ''].join('\n');
      }

      let commitsList = '';
      const commitsMd2 = renderCommitsTable(it.commits_between || []);
      commitsList = [commitsMd2, ''].join('\n');

      const latestWhenIso = it.created_at ? new Date(it.created_at).toISOString() : '';
      const latestShaShort = it.commit_short || (it.commit_sha ? it.commit_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
      const latestShaLink = (latestShaShort && it.commit_sha)
        ? ` [\`${latestShaShort}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${it.commit_sha})`
        : '';
      const latestLine = it.run_url
        ? `\n  - Latest failing run: [Run](${it.run_url}) ${latestWhenIso}${latestShaLink}`
        : '';
      const content = `  - First failing run on main: [Run](${it.first_failed_run_url}) ${when} ${shaLink}${latestLine}`;
      return ['<details>', `<summary>${workflowName}${timeBadge}</summary>`, '', content, errorsList, commitsList, '</details>', ''].join('\n');
    }
    return ['<details>', `<summary>${workflowName}${timeBadge}</summary>`, '', '  - No failure details available', '</details>', ''].join('\n');
  });

  return ['', '## Still Failing (No Recovery)', ...lines, ''].join('\n');
}

module.exports = {
  getWorkflowLink,
  findGoodBadCommits,
  getLastRunInfo,
  generateSummaryBox,
  buildReport,
  filterRunsByDate,
  filterWorkflowsByConfig,
  buildAlertMessage,
  computeLatestConclusion,
  computeLatestRunInfo,
  getMainWindowRuns,
  computeStatusChanges,
  enrichRegressions,
  enrichStayedFailing,
  buildRegressionsSection,
  buildStayedFailingSection,
  countTotalFailingJobs,
};
