// Fetch Workflow Data GitHub Action
// This action fetches workflow run data and caches it for analysis
//
// See: https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28

const core = require('@actions/core');
const github = require('@actions/github');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

// Import modules
const fetching = require('./lib/fetching');
const cache = require('./lib/cache');
const logs = require('./lib/logs');

// Import constants and functions used in this file
const {
  DEFAULT_DAYS,
  getCutoffDate,
  fetchAllWorkflowRuns,
  groupRunsByName,
  fetchNewWorkflowRuns,
  mergeAndDeduplicateRuns,
  recheckForNewerAttempts,
} = fetching;

const {
  findPreviousAggregateRun,
  pruneOldRuns,
  pruneAnnotationsIndex,
  pruneLogsIndex,
  pruneCommits,
  downloadAndExtractArtifact,
  findFileInDirectory,
  removeUnkeptDirectories,
  restoreArtifactsFromPreviousRun,
} = cache;

const {
  processWorkflowLogs,
} = logs;

const dataBuilding = require('./lib/data-building');
const {
  updateLastSuccessTimestamps,
  buildCommitsIndex,
} = dataBuilding;


/**
 * Main entrypoint for the action.
 * Loads previous cache, fetches new runs, merges/deduplicates, and saves updated cache.
 */
async function run() {
  try {
    // Get inputs
    const branch = core.getInput('branch') || 'main';
    const days = parseInt(core.getInput('days') || DEFAULT_DAYS);
    const forceFresh = String(core.getInput('force-fresh') || 'false').toLowerCase() === 'true';
    const rawCachePath = core.getInput('cache-path', { required: false });
    const defaultOutputPath = path.join(process.env.GITHUB_WORKSPACE || process.cwd(), 'workflow-data.json');
    const outputPath = rawCachePath && rawCachePath.trim() ? rawCachePath : defaultOutputPath;
    const workflowIdsInput = core.getInput('workflow_ids', { required: false });
    let workflowIds = null;
    if (workflowIdsInput) {
      try {
        workflowIds = JSON.parse(workflowIdsInput);
        if (!Array.isArray(workflowIds)) {
          core.warning('[CONFIG] workflow_ids is not an array, ignoring');
          workflowIds = null;
        } else {
          core.info(`[CONFIG] Loaded ${workflowIds.length} workflow IDs to fetch`);
        }
      } catch (e) {
        core.warning(`[CONFIG] Failed to parse workflow_ids: ${e.message}`);
        workflowIds = null;
      }
    }
    // Create authenticated Octokit client
    const octokit = github.getOctokit(core.getInput('GITHUB_TOKEN', { required: true }));
    const owner = github.context.repo.owner;
    const repo = github.context.repo.repo;
    const workspace = process.env.GITHUB_WORKSPACE || process.cwd();
    const cutoffDate = getCutoffDate(days);

    // Log initial GitHub API rate limit
    const initialRateLimit = await octokit.rest.rateLimit.get();
    const initialRemaining = initialRateLimit.data.resources.core.remaining;
    const limit = initialRateLimit.data.resources.core.limit;
    core.info(`[RATE_LIMIT] Initial GitHub API rate limit: ${initialRemaining} / ${limit} (${((initialRemaining/limit)*100).toFixed(1)}%)`);

    // Load cached data from previous aggregate run
    let previousRuns = [];
    let cachedGrouped = new Map();
    let cachedAnnotationsIndex = {};
    let cachedGtestLogsIndex = {};
    let cachedOtherLogsIndex = {};
    let cachedCommits = [];
    let cachedLastSuccessTimestamps = {};
    let annotationsIndexPath;
    let gtestLogsIndexPath;
    let otherLogsIndexPath;
    let commitsPath;

    // Find and restore artifacts from previous successful run (unless force-fresh is enabled)
    if (forceFresh) {
      core.info('[CACHE] force-fresh enabled, skipping artifact restoration and starting fresh');
    } else {
      // const previousRunId = await findPreviousAggregateRun(octokit, github.context, branch);
      const previousRunId = 20300472217;
      if (previousRunId) {
        const restored = await restoreArtifactsFromPreviousRun(octokit, github.context, previousRunId, workspace, cutoffDate, days);
        cachedGrouped = restored.cachedGrouped;
        cachedAnnotationsIndex = restored.cachedAnnotationsIndex;
        cachedGtestLogsIndex = restored.cachedGtestLogsIndex;
        cachedOtherLogsIndex = restored.cachedOtherLogsIndex;
        cachedCommits = restored.cachedCommits;
        cachedLastSuccessTimestamps = restored.cachedLastSuccessTimestamps;
      } else {
        core.info('[CACHE] No previous aggregate run found, starting fresh');
      }
    }


    // Convert cached grouped map to array of runs for merging
    for (const runs of cachedGrouped.values()) {
      previousRuns.push(...runs);
    }

    // Build set of cached (run ID, attempt) tuples to avoid re-downloading logs/annotations
    // This ensures we download logs for new attempts even if the run ID was seen before
    const cachedRunIds = new Set(); // Keep for backward compatibility with fetchAllWorkflowRuns
    const cachedRunAttempts = new Set(); // Set of `${runId}:${attempt}` tuples
    for (const runs of cachedGrouped.values()) {
      for (const run of runs) {
        const runIdStr = String(run.id);
        const attempt = run.run_attempt || 1; // Default to 1 if not present (older data)
        cachedRunIds.add(runIdStr);
        cachedRunAttempts.add(`${runIdStr}:${attempt}`);
      }
    }

    core.info(`[CACHE] Summary: ${previousRuns.length} runs restored, ${cachedGrouped.size} workflows`);
    core.info(`[CACHE] Cached run IDs: ${cachedRunIds.size} unique run IDs`);
    core.info(`[CACHE] Cached run attempts: ${cachedRunAttempts.size} unique (run ID, attempt) combinations`);

    // Fetch new runs from GitHub (skipping runs that are already cached)
    const newRuns = await fetchNewWorkflowRuns(octokit, github.context, days, cachedRunIds, workflowIds, branch);

    // Merge and deduplicate runs
    const mergedRuns = mergeAndDeduplicateRuns(previousRuns, newRuns, days);

    // Group runs by workflow name
    const grouped = groupRunsByName(mergedRuns);
    core.info(`[MERGE] Grouped into ${grouped.size} workflows`);

    // Check for newer attempts on the latest run ID for each workflow we care about
    // Pre-compute Set of existing (run ID, attempt) combinations for O(1) lookup
    const existingAttemptsSet = new Set(mergedRuns.map(r => `${r.id}:${r.run_attempt || 1}`));
    const newRunsToAdd = await recheckForNewerAttempts(octokit, github.context, grouped, branch, days, existingAttemptsSet);

    // Add the new runs to mergedRuns (both old and new attempts will be present)
    if (newRunsToAdd.length > 0) {
      mergedRuns.push(...newRunsToAdd);
      core.info(`[RECHECK] Added ${newRunsToAdd.length} runs with newer attempts to mergedRuns`);
      // Re-group after adding new runs
      const updatedGrouped = groupRunsByName(mergedRuns);
      grouped.clear();
      for (const [name, runs] of updatedGrouped.entries()) {
        grouped.set(name, runs);
      }
      core.info(`[RECHECK] Re-grouped after adding new attempts: ${grouped.size} workflows`);
    }

    // Ensure output directory exists
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }


    // Save grouped runs to artifact file
    fs.writeFileSync(outputPath, JSON.stringify(Array.from(grouped.entries())));

    // Update last success timestamps
    await updateLastSuccessTimestamps(
      grouped,
      branch,
      octokit,
      github.context,
      cachedLastSuccessTimestamps,
      workspace,
      outputDir
    );

    // Process workflow logs: download annotations and logs for failing runs
    const { annotationsIndexPath: annIndexPath, gtestLogsIndexPath: gtestIndexPath, otherLogsIndexPath: otherIndexPath } = await processWorkflowLogs(
      grouped,
      branch,
      workspace,
      cachedAnnotationsIndex,
      cachedGtestLogsIndex,
      cachedOtherLogsIndex,
      cachedRunAttempts,
      octokit,
      github.context
    );
    annotationsIndexPath = annIndexPath;
    gtestLogsIndexPath = gtestIndexPath;
    otherLogsIndexPath = otherIndexPath;

    // Build commits index
    commitsPath = await buildCommitsIndex(
      cachedCommits,
      days,
      branch,
      octokit,
      github.context,
      workspace
    );

    // Set output
    core.info(`[OUTPUT] Setting action outputs...`);
    const gtestLogsRoot = path.join(workspace, 'logs', 'gtest');
    const otherLogsRoot = path.join(workspace, 'logs', 'other');
    const annotationsRoot = path.join(workspace, 'annotations');
    core.setOutput('total-runs', mergedRuns.length);
    core.setOutput('workflow-count', grouped.size);
    core.setOutput('cache-path', outputPath);
    core.setOutput('gtest-logs-root', gtestLogsRoot);
    core.setOutput('gtest-logs-index-path', gtestLogsIndexPath);
    core.setOutput('other-logs-root', otherLogsRoot);
    core.setOutput('other-logs-index-path', otherLogsIndexPath);
    core.setOutput('annotations-root', annotationsRoot);
    core.setOutput('annotations-index-path', annotationsIndexPath);
    core.setOutput('commits-path', commitsPath);
    core.info(`[OUTPUT] total-runs: ${mergedRuns.length}, workflow-count: ${grouped.size}`);
    // Read indices to get entry counts for logging
    try {
      const annotationsIndex = fs.existsSync(annotationsIndexPath) ? JSON.parse(fs.readFileSync(annotationsIndexPath, 'utf8')) : {};
      const gtestLogsIndex = fs.existsSync(gtestLogsIndexPath) ? JSON.parse(fs.readFileSync(gtestLogsIndexPath, 'utf8')) : {};
      const otherLogsIndex = fs.existsSync(otherLogsIndexPath) ? JSON.parse(fs.readFileSync(otherLogsIndexPath, 'utf8')) : {};
      core.info(`[OUTPUT] annotations: ${Object.keys(annotationsIndex).length} entries`);
      core.info(`[OUTPUT] gtest-logs: ${Object.keys(gtestLogsIndex).length} entries`);
      core.info(`[OUTPUT] other-logs: ${Object.keys(otherLogsIndex).length} entries`);
    } catch (e) {
      core.warning(`[OUTPUT] Failed to read indices for logging: ${e.message}`);
    }

    // Log final GitHub API rate limit and calculate usage
    const finalRateLimit = await octokit.rest.rateLimit.get();
    const finalRemaining = finalRateLimit.data.resources.core.remaining;
    const finalLimit = finalRateLimit.data.resources.core.limit;
    const apiCallsUsed = initialRemaining - finalRemaining;
    core.info(`[RATE_LIMIT] Final GitHub API rate limit: ${finalRemaining} / ${finalLimit} (${((finalRemaining/finalLimit)*100).toFixed(1)}%)`);
    core.info(`[RATE_LIMIT] API calls used during this run: ${apiCallsUsed} (${initialRemaining} → ${finalRemaining})`);
  } catch (error) {
    core.setFailed(error.message);
  }
}

// Run the action if this file is executed directly
if (require.main === module) {
  run();
}
