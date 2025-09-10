// Analyze Workflow Data GitHub Action
// This action analyzes cached workflow run data and generates a summary report of workflow statuses.
// It provides two tables: one for push-triggered workflows and another for scheduled workflows.
// For scheduled workflows, it also tracks the last known good commit and earliest bad commit.
//
// See: https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28

const core = require('@actions/core');
const fs = require('fs');
const { summary } = require('@actions/core');

// Constants
const DEFAULT_LOOKBACK_DAYS = 15;
const SHA_SHORT_LENGTH = 7;
const SUCCESS_RATE_DECIMAL_PLACES = 2;
const SUCCESS_EMOJI = '✅';
const FAILURE_EMOJI = '❌';
const EMPTY_VALUE = '—';

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

    // Calculate the original run ID by subtracting (run_attempt - 1) from the current run ID
    const originalRunId = run.run_attempt > 1 ? run.id - (run.run_attempt - 1) : run.id;

    if (!uniqueRuns.has(originalRunId)) {
      uniqueRuns.set(originalRunId, {
        run,
        attempts: 0,
        isSuccessful: false,
        requiredRetry: false,
        succeededOnFirstTry: false,
        lastAttempt: run
      });
    } else {
      // This is an attempt
      const existingRun = uniqueRuns.get(originalRunId);
      existingRun.attempts++;

      // Update last attempt if this is a newer attempt
      if (run.run_attempt > existingRun.lastAttempt.run_attempt) {
        existingRun.lastAttempt = run;
      }
    }
  }

  // Second pass: determine final status based on last attempt
  for (const runInfo of uniqueRuns.values()) {
    const lastAttempt = runInfo.lastAttempt;
    if (lastAttempt.conclusion === 'success') {
      runInfo.isSuccessful = true;
      totalSuccessfulUniqueRuns++;

      if (lastAttempt.run_attempt === 1) {
        runInfo.succeededOnFirstTry = true;
        successfulUniqueRunsOnFirstTry++;
      } else {
        runInfo.requiredRetry = true;
        successfulUniqueRunsWithRetries++;
      }
    }
  }

  const uniqueRunsArray = Array.from(uniqueRuns.values()).map(r => r.run);
  const eventTypes = [...new Set(uniqueRunsArray.map(r => r.event))].join(', ');

  // Calculate rates
  const successRate = uniqueRunsArray.length === 0 ? "N/A" : (totalSuccessfulUniqueRuns / uniqueRunsArray.length * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";
  const uniqueSuccessRate = uniqueRunsArray.length === 0 ? "N/A" : (successfulUniqueRunsOnFirstTry / uniqueRunsArray.length * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";
  const retryRate = totalSuccessfulUniqueRuns === 0 ? "N/A" : (successfulUniqueRunsWithRetries / totalSuccessfulUniqueRuns * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";

  return {
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
 * Analyzes scheduled runs to find the last good and earliest bad commits.
 *
 * @param {Array<object>} scheduledMainRuns - Array of scheduled runs on main branch, sorted by date (newest first)
 * @returns {object} Object containing:
 *   - lastGoodSha: Short SHA of the last successful run (e.g., `a1b2c3d`)
 *   - earliestBadSha: Short SHA of the earliest failing run (e.g., `e4f5g6h`)
 */
function findGoodBadCommits(scheduledMainRuns) {
  let lastGoodSha = EMPTY_VALUE;
  let earliestBadSha = EMPTY_VALUE;
  let foundGood = false;
  let foundBad = false;

  for (const run of scheduledMainRuns) {
    if (!foundGood && run.conclusion === 'success') {
      const shortSha = run.head_sha.substring(0, SHA_SHORT_LENGTH);
      lastGoodSha = run.run_url ?
        `[\`${shortSha}\`](${run.run_url})` :
        shortSha;
      foundGood = true;
    }
    if (!foundBad && run.conclusion !== 'success') {
      const shortSha = run.head_sha.substring(0, SHA_SHORT_LENGTH);
      earliestBadSha = run.run_url ?
        `[\`${shortSha}\`](${run.run_url})` :
        shortSha;
      foundBad = true;
    }
    if (foundGood && foundBad) break;
  }

  return { lastGoodSha, earliestBadSha };
}

/**
 * Gets information about the last run on main branch.
 *
 * @param {Array<object>} mainBranchRuns - Array of runs on main branch, sorted by date (newest first)
 * @returns {object} Object containing run information
 */
function getLastRunInfo(mainBranchRuns) {
  const lastMainRun = mainBranchRuns[0];
  if (!lastMainRun) {
    return {
      status: EMPTY_VALUE,
      sha: EMPTY_VALUE,
      run: EMPTY_VALUE,
      pr: EMPTY_VALUE,
      title: EMPTY_VALUE,
      lastGoodSha: EMPTY_VALUE,
      earliestBadSha: EMPTY_VALUE,
      workflow: EMPTY_VALUE
    };
  }

  const mainRuns = mainBranchRuns.filter(r => r.event === lastMainRun.event || r.event === 'workflow_dispatch');
  const { lastGoodSha, earliestBadSha } = findGoodBadCommits(mainRuns);

  const prTitleWithAuthor = lastMainRun.pr_title ?
    `${lastMainRun.pr_title} - @${lastMainRun.pr_author}` :
    EMPTY_VALUE;

  // Create GitHub Actions run link using repository info from enriched data
  const runLink = lastMainRun.run_url ?
    `[\`${lastMainRun.head_sha.substring(0, SHA_SHORT_LENGTH)}\`](${lastMainRun.run_url})` :
    lastMainRun.head_sha.substring(0, SHA_SHORT_LENGTH);

  // Create workflow link
  const workflowLink = lastMainRun.workflow_url ?
    `[${lastMainRun.name}](${lastMainRun.workflow_url})` :
    lastMainRun.name;

  return {
    status: lastMainRun.conclusion === 'success' ? SUCCESS_EMOJI : FAILURE_EMOJI,
    sha: runLink,
    run: lastMainRun.run_attempt > 1 ? `#${lastMainRun.run_attempt}` : '',
    pr: lastMainRun.pr_number || EMPTY_VALUE,
    title: prTitleWithAuthor,
    lastGoodSha,
    earliestBadSha: lastMainRun.conclusion !== 'success' ? earliestBadSha : EMPTY_VALUE,
    workflow: workflowLink
  };
}

/**
 * Generates summary tables for push and scheduled workflows.
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {Array<object>} workflowConfigs - Array of workflow config objects
 * @returns {string} Markdown table for all workflows
 */
function generateSummaryBox(grouped, workflowConfigs) {
  const rows = [];

  // Process workflows in the order they appear in the config
  for (const config of workflowConfigs) {
    // Find matching workflows
    const matchingWorkflows = Array.from(grouped.entries()).filter(([name]) => {
      if (config.wkflw_name) {
        return name === config.wkflw_name;
      }
      if (config.wkflw_prefix) {
        return name.startsWith(config.wkflw_prefix);
      }
      return false;
    });

    // Process each matching workflow
    for (const [name, runs] of matchingWorkflows) {
      const stats = getWorkflowStats(runs);
      const runInfo = getLastRunInfo(runs);

      const row = `| ${runInfo.workflow} | ${stats.eventTypes || 'unknown'} | ${stats.totalRuns} | ${stats.successfulRuns} | ${stats.successRate} | ${stats.uniqueSuccessRate} | ${runInfo.status} | ${runInfo.sha} | ${runInfo.run} | ${runInfo.pr} | ${runInfo.title} | ${runInfo.earliestBadSha} | ${runInfo.lastGoodSha} |`;
      rows.push(row);
    }
  }

  // Generate the table header
  const header = `| Workflow | Event Types | Total Runs | Successful Runs | Success Rate | First Try Success | Status | SHA | Run | PR | Title | Earliest Bad SHA | Last Good SHA |`;
  const separator = `| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |`;

  return `\n${header}\n${separator}\n${rows.join('\n')}\n`;
}

/**
 * Builds the complete markdown report.
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {Array<object>} workflowConfigs - Array of workflow config objects
 * @returns {string} Complete markdown report
 */
function buildReport(grouped, workflowConfigs) {
  const days = parseFloat(core.getInput('days') || DEFAULT_LOOKBACK_DAYS);
  const timestamp = new Date().toISOString();

  // Format the time range display
  const hours = Math.floor(days * 24);
  const minutes = Math.floor((days * 24 * 60) % 60);
  const timeRange = hours > 0
    ? `${hours} hours${minutes > 0 ? ` and ${minutes} minutes` : ''}`
    : `${minutes} minutes`;

  return [
    `# Workflow Summary (Last ${timeRange}) - Generated at ${timestamp}\n`,
    generateSummaryBox(grouped, workflowConfigs),
    '\n## Column Descriptions\n',
    'A unique run represents a single workflow execution, which may have multiple retry attempts. For example, if a workflow fails and is retried twice, this counts as one unique run with three attempts (initial run + two retries).\n',
    '\n### Success Rate Calculations\n',
    'The success rates are calculated based on unique runs (not including retries in the denominator):\n',
    '- **Success Rate**: (Number of unique runs that eventually succeeded / Total number of unique runs) × 100%\n',
    '  - Example: 3 successful unique runs out of 5 total unique runs = 60% success rate\n',
    '- **First Try Success Rate**: (Number of unique runs that succeeded on first try / Total number of unique runs) × 100%\n',
    '  - Example: 1 unique run succeeded on first try out of 5 total unique runs = 20% first try success rate\n',
    '\n| Column | Description |',
    '|--------|-------------|',
    '| Workflow | Name of the workflow with link to its GitHub Actions page |',
    '| Event Types | Types of events that trigger this workflow |',
    '| Total Runs | Total number of workflow runs, including all retry attempts |',
    '| Successful Runs | Number of unique workflow runs that eventually succeeded |',
    '| Success Rate | Percentage of unique workflow runs that eventually succeeded |',
    '| First Try Success | Percentage of unique workflow runs that succeeded on their first attempt |',
    '| Status | Status of the most recent run (✅ for success, ❌ for failure) |',
    '| SHA | Short SHA of the most recent run with link to the run |',
    '| Run | Run attempt number if applicable |',
    '| PR | PR number associated with the most recent run, if any |',
    '| Title | Title and author of the PR (e.g., "Fix bug - @username") |',
    '| Earliest Bad SHA | Short SHA of the earliest failing run (only shown if last run failed) |',
    '| Last Good SHA | Short SHA of the last successful run |'
  ].join('\n');
}

/**
 * Main function to run the action
 */
async function run() {
  try {
    // Get inputs
    const cachePath = core.getInput('cache-path', { required: true });
    const daysInput = core.getInput('days') || DEFAULT_LOOKBACK_DAYS;
    const days = parseFloat(daysInput);
    if (isNaN(days)) {
      throw new Error(`Invalid days value: ${daysInput}. Must be a number.`);
    }
    if (days <= 0) {
      throw new Error(`Days must be positive, got: ${days}`);
    }

    const workflowConfigs = JSON.parse(core.getInput('workflow_configs', { required: true }));

    // Validate inputs
    if (!fs.existsSync(cachePath)) {
      throw new Error(`Cache file not found at ${cachePath}`);
    }
    if (!Array.isArray(workflowConfigs)) {
      throw new Error('Workflow configs must be a JSON array');
    }

    // Load cached data
    const grouped = new Map(JSON.parse(fs.readFileSync(cachePath, 'utf8')));

    // Track failed workflows
    const failedWorkflows = [];

    // Process each workflow
    for (const [name, runs] of grouped) {
      // Check if latest run is failing
      const sortedRuns = runs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      if (sortedRuns[0]?.conclusion !== 'success') {
        const lastRun = sortedRuns[0];
        const { lastGoodSha, earliestBadSha } = findGoodBadCommits(sortedRuns);

        failedWorkflows.push({
          name,
          pr: lastRun.pr_number || EMPTY_VALUE,
          author: lastRun.pr_author || EMPTY_VALUE,
          badSha: earliestBadSha,
          goodSha: lastGoodSha
        });
      }
    }

    // Format failed workflows as a simple string
    const failedWorkflowsStr = failedWorkflows.map(wf =>
      `${wf.name} (PR: ${wf.pr}, Author: ${wf.author}, Bad: ${wf.badSha}, Good: ${wf.goodSha})`
    ).join('\n');

    // Generate report
    const report = buildReport(grouped, workflowConfigs);

    // Set outputs
    core.setOutput('failed_workflows', failedWorkflowsStr);
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
