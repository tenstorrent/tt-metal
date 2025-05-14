// Analyze Workflow Data GitHub Action
// This action analyzes cached workflow run data and generates a summary report of workflow statuses.
// It provides two tables: one for push-triggered workflows and another for scheduled workflows.
// For scheduled workflows, it also tracks the last known good commit and earliest bad commit.
//
// See: https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28

const core = require('@actions/core');
const github = require('@actions/github');
const fs = require('fs');

// Constants
const DEFAULT_LOOKBACK_DAYS = 15;
const SHA_SHORT_LENGTH = 7;
const SUCCESS_RATE_DECIMAL_PLACES = 2;
const SUCCESS_EMOJI = '✅';
const FAILURE_EMOJI = '❌';
const EMPTY_VALUE = '—';

/**
 * Fetches PR information associated with a commit.
 *
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @param {string} commitSha - Full SHA of the commit to look up
 * @returns {Promise<object>} Object containing:
 *   - prNumber: Markdown link to the PR (e.g., [#123](url))
 *   - prTitle: Title of the PR or EMPTY_VALUE if not found
 *   - prAuthor: GitHub username of the PR author or 'unknown'
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
        prTitle: pr.title || EMPTY_VALUE,
        prAuthor: pr.user?.login || 'unknown'
      };
    }
  } catch (e) {
    core.warning(`Could not fetch PR for commit ${commitSha}: ${e.message}`);
  }
  return { prNumber: EMPTY_VALUE, prTitle: EMPTY_VALUE, prAuthor: EMPTY_VALUE };
}

/**
 * Calculates statistics for a set of workflow runs.
 *
 * @param {Array<object>} runs - Array of workflow run objects
 * @returns {object} Statistics object containing:
 *   - eventTypes: Comma-separated string of unique event types
 *   - successRate: Percentage of total runs that succeeded (e.g., "95.00%")
 *   - retryRate: Percentage of unique runs that had retries (e.g., "20.00%")
 *   - uniqueRuns: Number of unique runs (excluding retries and reruns)
 *   - totalRuns: Total number of runs including retries and reruns
 *   - successfulRuns: Number of successful runs (including retries)
 */
function getWorkflowStats(runs) {
  // Group runs by their original run ID to handle retries and reruns
  const uniqueRuns = new Map();
  let totalRuns = 0;
  let totalSuccesses = 0;
  let runsWithRetries = 0;

  for (const run of runs) {
    totalRuns++;
    if (run.conclusion === 'success') {
      totalSuccesses++;
    }
    // Use the original run ID as the key, considering both retries and reruns
    const originalRunId = run.run_attempt > 1 ? run.id - (run.run_attempt - 1) : run.id;
    if (!uniqueRuns.has(originalRunId)) {
      uniqueRuns.set(originalRunId, run);
    } else if (run.run_attempt > 1) {
      // If this is a retry, increment the counter for this unique run
      runsWithRetries++;
    }
  }

  const uniqueRunsArray = Array.from(uniqueRuns.values());
  const eventTypes = [...new Set(uniqueRunsArray.map(r => r.event))].join(', ');
  const successRate = totalRuns === 0 ? "N/A" : (totalSuccesses / totalRuns * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";
  const retryRate = uniqueRunsArray.length === 0 ? "N/A" : (runsWithRetries / uniqueRunsArray.length * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";

  return {
    eventTypes,
    successRate,
    retryRate,
    uniqueRuns: uniqueRunsArray.length,
    totalRuns,
    successfulRuns: totalSuccesses
  };
}

/**
 * Generates a GitHub Actions workflow URL.
 *
 * @param {object} context - GitHub Actions context
 * @param {string} workflowFile - Path to the workflow file relative to .github/workflows/
 * @returns {string} Full URL to the workflow in GitHub Actions
 */
function getWorkflowLink(context, workflowFile) {
  return workflowFile
    ? `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/workflows/${workflowFile}?query=branch%3Amain`
    : `https://github.com/${context.repo.owner}/${context.repo.repo}/actions`;
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
      lastGoodSha = `\`${run.head_sha.substring(0, SHA_SHORT_LENGTH)}\``;
      foundGood = true;
    }
    if (!foundBad && run.conclusion !== 'success') {
      earliestBadSha = `\`${run.head_sha.substring(0, SHA_SHORT_LENGTH)}\``;
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
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @returns {Promise<object>} Object containing run information
 */
async function getLastRunInfo(mainBranchRuns, github, context) {
  const lastMainRun = mainBranchRuns[0];
  if (!lastMainRun) {
    return {
      status: EMPTY_VALUE,
      sha: EMPTY_VALUE,
      run: EMPTY_VALUE,
      pr: EMPTY_VALUE,
      title: EMPTY_VALUE,
      lastGoodSha: EMPTY_VALUE,
      earliestBadSha: EMPTY_VALUE
    };
  }

  const prInfo = await fetchPRInfo(github, context, lastMainRun.head_sha);
  const mainRuns = mainBranchRuns.filter(r => r.event === lastMainRun.event || r.event === 'workflow_dispatch');
  const { lastGoodSha, earliestBadSha } = findGoodBadCommits(mainRuns);

  return {
    status: lastMainRun.conclusion === 'success' ? SUCCESS_EMOJI : FAILURE_EMOJI,
    sha: `\`${lastMainRun.head_sha.substring(0, SHA_SHORT_LENGTH)}\``,
    run: `[Run](${lastMainRun.html_url})${lastMainRun.run_attempt > 1 ? ` (#${lastMainRun.run_attempt})` : ''}`,
    pr: prInfo.prNumber,
    title: prInfo.prTitle,
    lastGoodSha,
    earliestBadSha: lastMainRun.conclusion !== 'success' ? earliestBadSha : EMPTY_VALUE
  };
}

/**
 * Generates summary tables for push and scheduled workflows.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @returns {Promise<string>} Markdown table for all workflows
 */
async function generateSummaryBox(grouped, github, context) {
  const rows = [];

  for (const [name, runs] of grouped.entries()) {
    const mainBranchRuns = runs
      .filter(r => r.head_branch === 'main')
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    const stats = getWorkflowStats(runs);

    // Skip workflows that only have workflow_dispatch as their event type
    if (stats.eventTypes === 'workflow_dispatch') {
      continue;
    }

    const workflowLink = getWorkflowLink(context, runs[0]?.path);
    const runInfo = await getLastRunInfo(mainBranchRuns, github, context);

    const row = `| [${name}](${workflowLink}) | ${stats.eventTypes || 'unknown'} | ${stats.totalRuns} | ${stats.uniqueRuns} | ${stats.successfulRuns} | ${stats.successRate} | ${stats.retryRate} | ${runInfo.status} | ${runInfo.sha} | ${runInfo.run} | ${runInfo.pr} | ${runInfo.title} | ${runInfo.earliestBadSha} | ${runInfo.lastGoodSha} |`;
    rows.push(row);
  }

  return [
    '## Workflow Summary',
    '| Workflow | Event Type(s) | Total Runs | Unique Runs | Successful Runs | Success Rate | Retry Rate | Last Run on `main` | Last SHA | Last Run | Last PR | PR Title | Earliest Bad SHA | Last Good SHA |',
    '|----------|---------------|------------|-------------|-----------------|--------------|------------|-------------------|----------|----------|---------|-----------|------------------|---------------|',
    ...rows,
    ''  // Empty line for better readability
  ].join('\n');
}

/**
 * Builds the complete markdown report.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @returns {Promise<string>} Complete markdown report
 */
async function buildReport(grouped, github, context) {
  const days = core.getInput('days', { required: false }) || DEFAULT_LOOKBACK_DAYS;
  return [
    `# Workflow Summary (Last ${days} Days)\n`,
    await generateSummaryBox(grouped, github, context)
  ].join('\n');
}

/**
 * Main entrypoint for the action.
 * Loads cached workflow data, filters by workflow configurations, and generates a summary report.
 *
 * The action:
 * 1. Loads workflow data from cache
 * 2. Filters workflows based on provided configurations
 * 3. Generates a summary report with push and scheduled workflow tables
 * 4. Sets outputs for failed workflows and the report
 *
 * @throws {Error} If cache file is not found or required inputs are missing
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
    const workflowConfigs = JSON.parse(core.getInput('workflow_configs', { required: true }));

    // Track failed workflows
    const failedWorkflows = [];

    // Filter and process each workflow configuration
    const filteredGrouped = new Map();
    for (const config of workflowConfigs) {
      for (const [name, runs] of grouped) {
        if ((config.wkflw_name && name === config.wkflw_name) ||
            (config.wkflw_prefix && name.startsWith(config.wkflw_prefix))) {
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
