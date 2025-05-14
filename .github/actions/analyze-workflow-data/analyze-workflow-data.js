// Analyze Workflow Data GitHub Action
// This action analyzes cached workflow run data and generates a report
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
 * Get workflow statistics including success rate and event types
 * @param {Array} runs - Array of workflow runs
 * @returns {object} Statistics about the workflow runs
 */
function getWorkflowStats(runs) {
  const successes = runs.filter(r => r.conclusion === 'success');
  const eventTypes = [...new Set(runs.map(r => r.event))].join(', ');
  const successRate = runs.length === 0 ? "N/A" : (successes.length / runs.length * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";

  return {
    successes,
    eventTypes,
    successRate,
    totalRuns: runs.length,
    successfulRuns: successes.length
  };
}

/**
 * Get the workflow link for a given workflow file
 * @param {object} context - GitHub Actions context
 * @param {string} workflowFile - Path to the workflow file
 * @returns {string} URL to the workflow
 */
function getWorkflowLink(context, workflowFile) {
  return workflowFile
    ? `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/workflows/${workflowFile}?query=branch%3Amain`
    : `https://github.com/${context.repo.owner}/${context.repo.repo}/actions`;
}

/**
 * Find the last good and earliest bad commits for scheduled runs
 * @param {Array} scheduledMainRuns - Array of scheduled runs on main branch
 * @returns {object} Last good and earliest bad commit SHAs
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
 * Generate a row for the workflow summary table
 * @param {object} params - Parameters for generating the row
 * @returns {string} Markdown table row
 */
function generateWorkflowRow({ name, workflowLink, stats, lastMainPassing, failureInfo }) {
  const baseRow = `| [${name}](${workflowLink}) | ${stats.eventTypes || 'unknown'} | ${stats.totalRuns} | ${stats.successfulRuns} | ${stats.successRate} | ${lastMainPassing} | ${failureInfo.sha} | ${failureInfo.run} | ${failureInfo.pr} | ${failureInfo.title} |`;

  return failureInfo.isScheduled
    ? baseRow + ` ${failureInfo.earliestBadSha} | ${failureInfo.lastGoodSha} |`
    : baseRow;
}

/**
 * Generate a summary box showing all workflows and their latest status
 */
async function generateSummaryBox(grouped, github, context) {
  const pushRows = [];
  const scheduleRows = [];

  for (const [name, runs] of grouped.entries()) {
    const mainBranchRuns = runs
      .filter(r => r.head_branch === 'main')
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    const stats = getWorkflowStats(runs);

    // Skip workflows that only have workflow_dispatch as their event type
    if (stats.eventTypes === 'workflow_dispatch') {
      continue;
    }

    const lastMainRun = mainBranchRuns[0];
    const lastMainPassing = lastMainRun?.conclusion === 'success' ? SUCCESS_EMOJI : FAILURE_EMOJI;
    const workflowLink = getWorkflowLink(context, runs[0]?.path);

    const failureInfo = {
      sha: EMPTY_VALUE,
      run: EMPTY_VALUE,
      pr: EMPTY_VALUE,
      title: EMPTY_VALUE,
      lastGoodSha: EMPTY_VALUE,
      earliestBadSha: EMPTY_VALUE,
      isScheduled: stats.eventTypes.includes('schedule')
    };

    if (lastMainRun && lastMainRun.conclusion !== 'success') {
      const prInfo = await fetchPRInfo(github, context, lastMainRun.head_sha);
      failureInfo.sha = `\`${lastMainRun.head_sha.substring(0, SHA_SHORT_LENGTH)}\``;
      failureInfo.run = `[Run](${lastMainRun.html_url})`;
      failureInfo.pr = prInfo.prNumber;
      failureInfo.title = prInfo.prTitle;

      if (lastMainRun.event === 'schedule') {
        const scheduledMainRuns = mainBranchRuns.filter(r => r.event === 'schedule' || r.event === 'workflow_dispatch');
        const { lastGoodSha, earliestBadSha } = findGoodBadCommits(scheduledMainRuns);
        failureInfo.lastGoodSha = lastGoodSha;
        failureInfo.earliestBadSha = earliestBadSha;
      }
    }

    const row = generateWorkflowRow({
      name,
      workflowLink,
      stats,
      lastMainPassing,
      failureInfo
    });

    if (failureInfo.isScheduled) {
      scheduleRows.push(row);
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
  const days = core.getInput('days', { required: false }) || DEFAULT_LOOKBACK_DAYS;
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
