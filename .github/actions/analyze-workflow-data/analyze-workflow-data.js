// Analyze Workflow Data GitHub Action
// This action analyzes cached workflow run data and generates a summary report of workflow statuses.
// It provides two tables: one for push-triggered workflows and another for scheduled workflows.
// For scheduled workflows, it also tracks the last known good commit and earliest bad commit.
//
// See: https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28

// These are all node.js modules needed for the action to work
const core = require('@actions/core'); // Core utilities for I/O
const github = require('@actions/github'); // GitHub API client
const fs = require('fs'); // File system operations
const path = require('path'); // File path utilities

// Import modules
const dataLoading = require('./lib/data-loading');
const errorProcessing = require('./lib/error-processing');
const analysis = require('./lib/analysis');
const reporting = require('./lib/reporting');

// Import constants used in this file
const {
  DEFAULT_LOOKBACK_DAYS,
  EMPTY_VALUE,
  DEFAULT_INFRA_OWNER,
} = dataLoading;
// Import functions from modules
const {
  loadAnnotationsIndexFromFile,
  getAnnotationsDirForRunId,
  loadLogsIndexFromFile,
  getGtestLogsDirForRunId,
  getOtherLogsDirForRunId,
  loadLastSuccessTimestamps,
  getTimeSinceLastSuccess,
  getCommitDescription,
  loadCommitsIndex,
  listCommitsBetweenOffline,
  setAnnotationsIndexMap,
  setGtestLogsIndexMap,
  setOtherLogsIndexMap,
  setLastSuccessTimestamps,
  setCommitsIndex,
} = dataLoading;

const {
  loadOwnersMapping,
  normalizeOwners,
  extractSignificantTokens,
  getJobNameComponentTail,
  findOwnerForLabel,
  inferJobAndTestFromSnippet,
  resolveOwnersForSnippet,
  fetchErrorSnippetsForRun,
  renderErrorsTable,
} = errorProcessing;

const {
  getWorkflowStats,
  findFirstFailInWindow,
  renderCommitsTable,
  fetchPRInfo,
  fetchCommitAuthor,
} = analysis;

const {
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
  enrichFailingDetails,
  detectJobLevelRegressions,
  buildRegressionsSection,
  buildStayedFailingSection,
} = reporting;

// Import the update owners script
const { updateOwnersJson } = require('./update-owners-from-pipeline');


/**
 * Main function to run the action
 */
async function run() {
  try {
    // Update owners.json from pipeline_reorg files at the beginning
    try {
      const ownersPath = path.join(__dirname, 'owners.json');
      const pipelineReorgDir = path.join(__dirname, '../../..', 'tests/pipeline_reorg');
      updateOwnersJson(ownersPath, pipelineReorgDir);
      core.info('Updated owners.json from pipeline_reorg files');
    } catch (error) {
      core.warning(`Failed to update owners.json: ${error.message}`);
      // Continue execution even if update fails
    }

    // Get inputs
    const cachePath = core.getInput('cache-path', { required: true }); // get the workflow data cache made by the fetch-workflow-data action
    const previousCachePath = core.getInput('previous-cache-path', { required: false }); // get the previous workflow data cache made by the fetch-workflow-data action from the most recent previous run on main branch
    const days = parseInt(core.getInput('days') || DEFAULT_LOOKBACK_DAYS, 10); // get the number of days to look back for workflow data
    const alertAll = String(core.getInput('alert-all') || 'false').toLowerCase() === 'true'; // get the alert-all input from the action inputs
    const annotationsIndexPath = core.getInput('annotations-index-path', { required: false }); // optional: path to JSON mapping runId -> annotations dir
    const commitsPath = core.getInput('commits-path', { required: false }); // optional: path to commits index JSON
    const gtestLogsIndexPath = core.getInput('gtest-logs-index-path', { required: false }); // optional: path to gtest logs index JSON
    const otherLogsIndexPath = core.getInput('other-logs-index-path', { required: false }); // optional: path to other logs index JSON
    const lastSuccessTimestampsPath = core.getInput('last-success-timestamps-path', { required: false }); // optional: path to last success timestamps JSON

    // Validate inputs
    if (!fs.existsSync(cachePath)) {
      throw new Error(`Cache file not found at ${cachePath}`); // throw an error if the cache file does not exist (no data was fetched)
    }
    if (isNaN(days) || days <= 0) {
      throw new Error('Days must be a positive number'); // throw an error if the days is not a positive number
    }

    // Load cached data
    const grouped = JSON.parse(fs.readFileSync(cachePath, 'utf8')); // parse the cached data into a json object
    const hasPrevious = previousCachePath && fs.existsSync(previousCachePath); // check if the previous cache file exists
    const previousGrouped = hasPrevious ? JSON.parse(fs.readFileSync(previousCachePath, 'utf8')) : null; // parse the previous cached data into a json object


    if (annotationsIndexPath) {
      const annotationsIndexMap = loadAnnotationsIndexFromFile(annotationsIndexPath);
      setAnnotationsIndexMap(annotationsIndexMap);
      if (annotationsIndexMap && annotationsIndexMap.size) {
        core.info(`Loaded annotations index with ${annotationsIndexMap.size} entries from ${annotationsIndexPath}`);
      } else if (annotationsIndexPath) {
        core.info(`No valid entries found in annotations index file at ${annotationsIndexPath}`);
      }
    }

    if (gtestLogsIndexPath) {
      const gtestLogsIndexMap = loadLogsIndexFromFile(gtestLogsIndexPath);
      setGtestLogsIndexMap(gtestLogsIndexMap);
      if (gtestLogsIndexMap && gtestLogsIndexMap.size) {
        core.info(`Loaded gtest logs index with ${gtestLogsIndexMap.size} entries from ${gtestLogsIndexPath}`);
      } else if (gtestLogsIndexPath) {
        core.info(`No valid entries found in gtest logs index file at ${gtestLogsIndexPath}`);
      }
    }

    if (otherLogsIndexPath) {
      const otherLogsIndexMap = loadLogsIndexFromFile(otherLogsIndexPath);
      setOtherLogsIndexMap(otherLogsIndexMap);
      if (otherLogsIndexMap && otherLogsIndexMap.size) {
        core.info(`Loaded other logs index with ${otherLogsIndexMap.size} entries from ${otherLogsIndexPath}`);
      } else if (otherLogsIndexPath) {
        core.info(`No valid entries found in other logs index file at ${otherLogsIndexPath}`);
      }
    }

    if (lastSuccessTimestampsPath) {
      const lastSuccessTimestamps = loadLastSuccessTimestamps(lastSuccessTimestampsPath);
      setLastSuccessTimestamps(lastSuccessTimestamps);
      if (lastSuccessTimestamps && lastSuccessTimestamps.size) {
        core.info(`Loaded last success timestamps with ${lastSuccessTimestamps.size} entries from ${lastSuccessTimestampsPath}`);
      } else {
        core.info(`No valid entries found in last success timestamps file at ${lastSuccessTimestampsPath}`);
      }
    }

    // Load commits index (optional)
    const commitsIndex = loadCommitsIndex(commitsPath) || [];
    setCommitsIndex(commitsIndex);
    core.info(`Loaded commits index entries: ${Array.isArray(commitsIndex) ? commitsIndex.length : 0}`);

    // Filter and process workflows (process all workflows since we only fetched runs for workflows we care about)
    const { filteredGrouped, filteredPreviousGrouped, failedWorkflows } = filterWorkflowsByConfig(
      grouped,
      previousGrouped,
      days
    );

    // Generate primary report
    const mainReport = await buildReport(filteredGrouped, github.context);

    // END OF BASIC ANALYZE STEP

    // Cache for error snippets to avoid fetching the same run's errors multiple times
    const errorSnippetsCache = new Map();

    // Optional: Build Slack-ready alert message for all failing workflows with owner mentions
    const alertAllMessage = await buildAlertMessage(filteredGrouped, failedWorkflows, alertAll, errorSnippetsCache);

    // Compute status changes vs previous and write JSON
    const { changes, regressedDetails, stayedFailingDetails } = computeStatusChanges(
      filteredGrouped,
      filteredPreviousGrouped,
      github.context
    );

    // Enrich regressions with first failing run within the window
    await enrichFailingDetails(regressedDetails, filteredGrouped, errorSnippetsCache, changes, github.context, 'success_to_fail');

    // Enrich stayed failing with first failing run within the window
    await enrichFailingDetails(stayedFailingDetails, filteredGrouped, errorSnippetsCache, changes, github.context, 'stayed_failing');

    // Detect job-level regressions in stayed_failing workflows
    // This identifies NEW failing jobs in pipelines that were already failing
    await detectJobLevelRegressions(stayedFailingDetails, regressedDetails, errorSnippetsCache, github.context);

    // upload the changes json to the artifact space
    const outputDir = process.env.GITHUB_WORKSPACE || process.cwd();
    const statusChangesPath = path.join(outputDir, 'workflow-status-changes.json');
    fs.writeFileSync(statusChangesPath, JSON.stringify(changes));
    core.setOutput('status_changes_path', statusChangesPath);

    // Build report sections
    let regressionsSection = '';
    let stayedFailingSection = '';
    try {
      regressionsSection = buildRegressionsSection(regressedDetails, github.context);
      stayedFailingSection = buildStayedFailingSection(stayedFailingDetails, github.context);
    } catch (_) {
      regressionsSection = ['', '## Regressions (Pass → Fail)', '- None', ''].join('\n');
      stayedFailingSection = ['', '## Still Failing (No Recovery)', '- None', ''].join('\n');
    }

    // Do not include alerts section inside the report; Slack message will carry it
    const finalReport = [mainReport, regressionsSection, stayedFailingSection]
      .filter(Boolean)
      .join('\n');

    // Set outputs
    core.setOutput('failed_workflows', JSON.stringify(failedWorkflows));
    core.setOutput('report', finalReport);
    core.setOutput('alert_all_message', alertAllMessage || '');
    core.setOutput('regressed_workflows', JSON.stringify(regressedDetails));

    await core.summary.addRaw(finalReport).write();

  } catch (error) {
    core.setFailed(error.message);
  }
}

// Run the action if this file is executed directly
if (require.main === module) {
  run();
}
