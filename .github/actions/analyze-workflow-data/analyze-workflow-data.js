// Analyze Workflow Data GitHub Action
// This action analyzes cached workflow run data and generates a report
//
// See: https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28

const core = require('@actions/core');
const github = require('@actions/github');
const fs = require('fs');

// Constants for reporting
const RECENT_RUNS_COUNT = 5; // Number of recent runs to show for push events

/**
 * Fetch PR information for a commit, including PR number, title, and author.
 * @param {object} github - Octokit client
 * @param {object} context - GitHub Actions context
 * @param {string} commitSha - Commit SHA
 * @returns {Promise<object>} PR info or placeholders
 */
async function fetchPRInfo(github, context, commitSha) {
  try {
    const { data: prs } = await github.rest.repos.listPullRequestsAssociatedWithCommit({
      owner: context.repo.owner,
      repo: context.repo.repo,
      commit_sha: commitSha,
    });
    if (prs.length > 0) {
      const pr = prs[0];
      return {
        prNumber: `[#${pr.number}](https://github.com/${context.repo.owner}/${context.repo.repo}/pull/${pr.number})`,
        prTitle: pr.title || '—',
        prAuthor: pr.user?.login || 'unknown'
      };
    }
  } catch (e) {
    core.warning(`Could not fetch PR for commit ${commitSha}: ${e.message}`);
  }
  return { prNumber: '—', prTitle: '—', prAuthor: '—' };
}

/**
 * Generate a single line of run report with commit, PR, and run details.
 */
function generateRunReportLine(run, prInfo) {
  const shortSha = run.head_sha.substring(0, 7);
  const failureReason = run.failure_message || run.conclusion || '—';
  const retryAttempt = run.run_attempt ? run.run_attempt - 1 : 0;
  const runLink = `[Run](${run.html_url})`;
  return `${shortSha} | ${prInfo.prNumber} | ${run.conclusion} | ${retryAttempt} | ${failureReason} | ${runLink} | ${prInfo.prAuthor} | ${prInfo.prTitle}\n`;
}

/**
 * Generate workflow summary section with statistics and status.
 */
function generateWorkflowSummary(name, runs, successes, context, lastMainPassing) {
  const eventTypes = [...new Set(runs.map(r => r.event))].join(', ');
  const workflowFile = runs[0]?.path;
  const workflowLink = workflowFile
    ? `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/workflows/${workflowFile}?query=branch%3Amain`
    : `https://github.com/${context.repo.owner}/${context.repo.repo}/actions`;
  return [
    `### ${name}`,
    `- **Workflow Link:** [View ${name} runs](${workflowLink})`,
    `- **Event Type(s):** ${eventTypes || 'unknown'}`,
    `- **Generated At:** ${(new Date()).toISOString()}`,
    `- **Total Completed Runs**: ${runs.length}`,
    `- **Successful Runs**: ${successes.length}`,
    `- **Success Rate**: ${(runs.length === 0 ? "N/A" : (successes.length / runs.length * 100).toFixed(2) + "%")}`,
    `- **Last Run on \`main\` Passed**: ${lastMainPassing}`,
    ''  // Empty line for better readability
  ].join('\n');
}

/**
 * Handle scheduled runs: show latest, earliest failed, and last known good run.
 * Only outputs if the latest scheduled run is failing.
 * Order: latest failed, earliest failed in sequence, last known good (if available).
 */
async function handleScheduledRuns(scheduledMainRuns, github, context) {
  if (!scheduledMainRuns.length) return '';
  // Get latest scheduled run (should be failing)
  const latestRun = scheduledMainRuns[0];
  if (latestRun.conclusion === 'success') return '';
  // Find the block of consecutive failures at the top (newest)
  let latestFailed = null;
  let earliestFailed = null;
  let lastGood = null;
  for (let i = 0; i < scheduledMainRuns.length; i++) {
    const run = scheduledMainRuns[i];
    if (run.conclusion !== 'success') {
      if (latestFailed === null) {
        latestFailed = run;
      }
      earliestFailed = run;
    } else {
      // First success after failures
      lastGood = run;
      break;
    }
  }
  // Prepare the runs to show
  // We conditionally add earliestFailed and lastGood to avoid duplicate entries (if latestFailed === earliestFailed)
  // and to avoid adding undefined if lastGood does not exist.
  const runsToShow = [latestFailed];
  if (earliestFailed !== latestFailed) runsToShow.push(earliestFailed);
  if (lastGood) runsToShow.push(lastGood);
  const prInfos = await Promise.all(
    runsToShow.map(run => fetchPRInfo(github, context, run.head_sha))
  );
  // Add a plain text explanation for the order/meaning of runsToShow
  const explanation =
    'Order of runs: Latest failed run, earliest failed run in a sequence, last known good run (if available).\n';
  return explanation + runsToShow.map((run, i) => generateRunReportLine(run, prInfos[i])).join('');
}

/**
 * Handle push runs: show recent failed runs (up to RECENT_RUNS_COUNT).
 */
async function handlePushRuns(mainBranchRuns, github, context) {
  const recent = mainBranchRuns.slice(0, RECENT_RUNS_COUNT);
  const prInfos = await Promise.all(
    recent.map(run => fetchPRInfo(github, context, run.head_sha))
  );
  return recent.map((run, i) =>
    generateRunReportLine(run, prInfos[i])
  ).join('');
}

/**
 * Generate a summary box showing all workflows and their latest status
 */
async function generateSummaryBox(grouped, github, context) {
  const pushRows = [];
  const scheduleRows = [];

  for (const [name, runs] of grouped.entries()) {
    const successes = runs.filter(r => r.conclusion === 'success');
    const mainBranchRuns = runs
      .filter(r => r.head_branch === 'main')
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    const lastMainRun = mainBranchRuns[0];
    const lastMainPassing = lastMainRun?.conclusion === 'success' ? '✅' : '❌';
    const eventTypes = [...new Set(runs.map(r => r.event))].join(', ');

    // Skip workflows that only have workflow_dispatch as their event type
    if (eventTypes === 'workflow_dispatch') {
      continue;
    }

    const workflowFile = runs[0]?.path;
    const workflowLink = workflowFile
      ? `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/workflows/${workflowFile}?query=branch%3Amain`
      : `https://github.com/${context.repo.owner}/${context.repo.repo}/actions`;
    const successRate = runs.length === 0 ? "N/A" : (successes.length / runs.length * 100).toFixed(2) + "%";

    let failureSha = '—';
    let failureRun = '—';
    let failurePr = '—';
    let failureTitle = '—';
    let lastGoodSha = '—';
    let earliestBadSha = '—';

    if (lastMainRun && lastMainRun.conclusion !== 'success') {
      const prInfo = await fetchPRInfo(github, context, lastMainRun.head_sha);
      failureSha = `\`${lastMainRun.head_sha.substring(0, 7)}\``;
      failureRun = `[Run](${lastMainRun.html_url})`;
      failurePr = prInfo.prNumber;
      failureTitle = prInfo.prTitle;

      // For scheduled runs, find the last good and earliest bad commits
      if (lastMainRun.event === 'schedule') {
        // Include both scheduled and manually triggered runs on main branch
        const scheduledMainRuns = mainBranchRuns.filter(r => r.event === 'schedule' || r.event === 'workflow_dispatch');
        let foundGood = false;
        let foundBad = false;

        for (const run of scheduledMainRuns) {
          if (!foundGood && run.conclusion === 'success') {
            lastGoodSha = `\`${run.head_sha.substring(0, 7)}\``;
            foundGood = true;
          }
          if (!foundBad && run.conclusion !== 'success') {
            earliestBadSha = `\`${run.head_sha.substring(0, 7)}\``;
            foundBad = true;
          }
          if (foundGood && foundBad) break;
        }
      }
    }

    const row = `| [${name}](${workflowLink}) | ${eventTypes || 'unknown'} | ${runs.length} | ${successes.length} | ${successRate} | ${lastMainPassing} | ${failureSha} | ${failureRun} | ${failurePr} | ${failureTitle} |`;

    if (eventTypes.includes('schedule')) {
      scheduleRows.push(row + ` ${earliestBadSha} | ${lastGoodSha} |`);
    } else {
      pushRows.push(row);
    }
  }

  const pushTable = [
    '## Push Event Workflows',
    '| Workflow | Event Type(s) | Total Runs | Successful Runs | Success Rate | Last Run on `main` | Failed SHA | Failed Run | Failed PR | PR Title |',
    '|----------|---------------|------------|-----------------|--------------|-------------------|------------|------------|-----------|-----------|',
    ...pushRows,
    ''  // Empty line for better readability
  ].join('\n');

  const scheduleTable = [
    '## Scheduled Workflows',
    '| Workflow | Event Type(s) | Total Runs | Successful Runs | Success Rate | Last Run on `main` | Failed SHA | Failed Run | Failed PR | PR Title | Earliest Bad SHA | Last Good SHA |',
    '|----------|---------------|------------|-----------------|--------------|-------------------|------------|------------|-----------|-----------|------------------|---------------|',
    ...scheduleRows,
    ''  // Empty line for better readability
  ].join('\n');

  return [pushTable, scheduleTable].join('\n');
}

/**
 * Build the markdown report for all grouped runs.
 */
async function buildReport(grouped, github, context) {
  const days = core.getInput('days', { required: false }) || '15';
  return [
    `# Workflow Summary (Last ${days} Days)\n`,
    await generateSummaryBox(grouped, github, context)
  ].join('\n');
}

/**
 * Main entrypoint for the action.
 * Loads cached workflow data, filters by workflow configurations, and generates a summary report.
 */
async function run() {
  try {
    // Get cache path from input
    const cachePath = core.getInput('cache-path', { required: true });
    if (!fs.existsSync(cachePath)) {
      throw new Error(`Cache file not found at ${cachePath}`);
    }
    // Load cached data
    const grouped = JSON.parse(fs.readFileSync(cachePath, 'utf8'));
    // Get workflow configurations
    const workflowConfigs = JSON.parse(core.getInput('workflow_configs', { required: true }));

    // Track failed workflows
    const failedWorkflows = [];

    // Filter and process each workflow configuration
    const filteredGrouped = new Map();
    for (const config of workflowConfigs) {
      for (const [name, runs] of grouped) {
        if (config.wkflw_name && name === config.wkflw_name) {
          filteredGrouped.set(name, runs);
          // Check if latest run on main is failing
          const mainBranchRuns = runs
            .filter(r => r.head_branch === 'main')
            .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
          if (mainBranchRuns[0]?.conclusion !== 'success') {
            failedWorkflows.push(name);
          }
        } else if (config.wkflw_prefix && name.startsWith(config.wkflw_prefix)) {
          filteredGrouped.set(name, runs);
          // Check if latest run on main is failing
          const mainBranchRuns = runs
            .filter(r => r.head_branch === 'main')
            .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
          if (mainBranchRuns[0]?.conclusion !== 'success') {
            failedWorkflows.push(name);
          }
        }
      }
    }

    // Create authenticated Octokit client for PR info
    const octokit = github.getOctokit(core.getInput('GITHUB_TOKEN', { required: true }));
    // Generate report
    const report = await buildReport(filteredGrouped, octokit, github.context);

    // Set outputs
    core.setOutput('failed_workflows', JSON.stringify(failedWorkflows));
    core.setOutput('report', report);

    await core.summary.addRaw(report).write();
  } catch (error) {
    core.setFailed(error.message);
  }
}

// Run the action if this file is executed directly
if (require.main === module) {
  run();
}
