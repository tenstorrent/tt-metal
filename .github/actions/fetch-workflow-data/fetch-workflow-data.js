// Fetch Workflow Data GitHub Action
// This action fetches workflow run data and caches it for analysis
//
// See: https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28

const core = require('@actions/core');
const github = require('@actions/github');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

// Constants for pagination and filtering
const MAX_PAGES = 100; // Maximum number of pages to fetch from GitHub API (tune for rate limits/performance)
const RUNS_PER_PAGE = 100; // GitHub API max per page
const DEFAULT_DAYS = 15; // Default rolling window in days
/**
 * Get the cutoff date for filtering runs.
 * @param {number} days - Number of days to look back
 * @returns {Date} The cutoff date
 */
function getCutoffDate(days) {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d;
}

/**
 * Fetch workflow runs for specific workflows, paginated, stopping early when cached runs are encountered.
 * @param {object} github - Octokit client
 * @param {object} context - GitHub Actions context
 * @param {number} days - Number of days to look back
 * @param {Set<string>} cachedRunIds - Set of run IDs that are already cached (skip these)
 * @param {Array<string>} workflowIds - Array of workflow file paths to fetch runs for
 * @param {string} eventType - Optional event type filter
 * @returns {Promise<Array>} Array of workflow run objects (only new, non-cached runs)
 */
async function fetchAllWorkflowRuns(github, context, days, cachedRunIds = null, workflowIds = null, eventType='') {
  const allRuns = [];
  const cutoffDate = getCutoffDate(days);
  const twoWeeksAgo = getCutoffDate(14); // For early exit check
  const cachedIds = cachedRunIds || new Set();

  core.info(`[FETCH] cutoffDate: ${cutoffDate.toISOString()}`);
  core.info(`[FETCH] days: ${days}, cachedRunIds: ${cachedIds.size}, eventType: ${eventType || 'all'}`);
  core.info(`[FETCH] workflowIds: ${workflowIds ? workflowIds.length + ' workflows' : 'all workflows (backward compatibility)'}`);

  const MAX_CONSECUTIVE_CACHED = 50; // Stop after 50 consecutive cached runs
  let totalSkippedOldRuns = 0;
  let totalSkippedCachedRuns = 0;
  let totalAddedNewRuns = 0;

  // If workflow IDs are provided, fetch runs for each workflow specifically
  if (workflowIds && workflowIds.length > 0) {
    core.info(`[FETCH] Fetching runs for ${workflowIds.length} specific workflows`);
    const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

    for (const workflowId of workflowIds) {
      let consecutiveCachedRuns = 0;
      let skippedOldRuns = 0;
      let skippedCachedRuns = 0;
      let addedNewRuns = 0;
      let shouldStopWorkflow = false;

      core.info(`[FETCH] Fetching runs for workflow: ${workflowId}`);

      for (let page = 1; page <= MAX_PAGES; page++) {
        core.info(`[FETCH] ${workflowId} - Fetching page ${page}...`);
        const params = {
          owner: context.repo.owner,
          repo: context.repo.repo,
          workflow_id: workflowId,
          per_page: RUNS_PER_PAGE,
          page,
        };
        if (eventType) {
          params.event = eventType;
        }

        try {
          const { data: runs } = await github.rest.actions.listWorkflowRuns(params);
          if (!runs.workflow_runs || runs.workflow_runs.length === 0) {
            core.info(`[FETCH] ${workflowId} - No runs on page ${page}, stopping`);
            break;
          }

          core.info(`[FETCH] ${workflowId} - Page ${page}: processing ${runs.workflow_runs.length} runs`);

          for (const run of runs.workflow_runs) {
            const runDate = new Date(run.created_at);
            const runIdStr = String(run.id);

            // Early exit if runs are older than 2 weeks
            if (runDate < twoWeeksAgo) {
              core.info(`[FETCH] ${workflowId} - Early exit: runs older than 2 weeks (${runDate.toISOString()} < ${twoWeeksAgo.toISOString()})`);
              shouldStopWorkflow = true;
              break;
            }

            // Skip runs older than cutoff date
            if (runDate < cutoffDate) {
              skippedOldRuns++;
              totalSkippedOldRuns++;
              continue;
            }

            // If we have cached run IDs, check if this run is already cached
            if (cachedIds.size > 0 && cachedIds.has(runIdStr)) {
              consecutiveCachedRuns++;
              skippedCachedRuns++;
              totalSkippedCachedRuns++;
              // If we've seen many consecutive cached runs, we've likely reached the boundary
              if (consecutiveCachedRuns >= MAX_CONSECUTIVE_CACHED) {
                core.info(`[FETCH] ${workflowId} - Early exit: found ${consecutiveCachedRuns} consecutive cached runs`);
                shouldStopWorkflow = true;
                break;
              }
              continue;
            }

            // Reset consecutive cached counter when we find a new run
            if (consecutiveCachedRuns > 0) {
              core.info(`[FETCH] ${workflowId} - Found new run after ${consecutiveCachedRuns} cached runs, resetting counter`);
            }
            consecutiveCachedRuns = 0;

            // This is a new run, add it
            addedNewRuns++;
            totalAddedNewRuns++;
            allRuns.push(run);
            if (addedNewRuns <= 5) {
              core.info(`[FETCH] ${workflowId} - Added new run ${runIdStr} (${run.name}, ${runDate.toISOString()})`);
            }
          }

          if (shouldStopWorkflow) {
            break;
          }

          // If we got fewer runs than requested, we've reached the end
          if (runs.workflow_runs.length < RUNS_PER_PAGE) {
            break;
          }

          // Small delay between pages to avoid rate limiting
          await delay(100);
        } catch (e) {
          core.warning(`[FETCH] Failed to fetch runs for workflow ${workflowId} page ${page}: ${e.message}`);
          break;
        }
      }

      core.info(`[FETCH] ${workflowId} - Summary: added ${addedNewRuns} new runs, skipped ${skippedCachedRuns} cached, ${skippedOldRuns} old`);

      // Small delay between workflows to avoid rate limiting
      await delay(200);
    }

    core.info(`[FETCH] Completed all workflows. Total: added ${totalAddedNewRuns} new runs, skipped ${totalSkippedCachedRuns} cached, ${totalSkippedOldRuns} old`);
    return allRuns;
  }

  // Backward compatibility: if no workflow IDs provided, use old behavior
  core.info(`[FETCH] No workflow IDs provided, using backward compatibility mode (fetching all workflows)`);
  let consecutiveCachedRuns = 0;
  let skippedOldRuns = 0;
  let skippedCachedRuns = 0;
  let addedNewRuns = 0;

  for (let page = 1; page <= MAX_PAGES; page++) {
    core.info(`[FETCH] Fetching page ${page}...`);
    const params = {
      owner: context.repo.owner,
      repo: context.repo.repo,
      per_page: RUNS_PER_PAGE,
      page,
    };
    if (eventType) {
      params.event = eventType;
    }
    const { data: runs } = await github.rest.actions.listWorkflowRunsForRepo(params);
    if (!runs.workflow_runs.length) {
      core.info(`[FETCH] No runs on page ${page}, stopping`);
      break;
    }

    core.info(`[FETCH] Page ${page}: processing ${runs.workflow_runs.length} runs`);

    for (const run of runs.workflow_runs) {
      const runDate = new Date(run.created_at);
      const runIdStr = String(run.id);

      // Early exit if runs are older than 2 weeks
      if (runDate < twoWeeksAgo) {
        core.info(`[FETCH] Early exit: runs older than 2 weeks (${runDate.toISOString()} < ${twoWeeksAgo.toISOString()})`);
        core.info(`[FETCH] Summary: added ${addedNewRuns} new runs, skipped ${skippedCachedRuns} cached, ${skippedOldRuns} old`);
        return allRuns;
      }

      // Skip runs older than cutoff date
      if (runDate < cutoffDate) {
        skippedOldRuns++;
        if (skippedOldRuns <= MAX_CONSECUTIVE_CACHED) {
          core.info(`[FETCH] Skipping run ${runIdStr} (older than cutoff: ${runDate.toISOString()} < ${cutoffDate.toISOString()})`);
        }
        continue;
      }

      // If we have cached run IDs, check if this run is already cached
      if (cachedIds.size > 0 && cachedIds.has(runIdStr)) {
        consecutiveCachedRuns++;
        skippedCachedRuns++;
        if (consecutiveCachedRuns <= MAX_CONSECUTIVE_CACHED) {
          core.info(`[FETCH] Skipping cached run ${runIdStr} (consecutive cached: ${consecutiveCachedRuns})`);
        }
        // If we've seen many consecutive cached runs, we've likely reached the boundary
        if (consecutiveCachedRuns >= MAX_CONSECUTIVE_CACHED) {
          core.info(`[FETCH] Early exit: found ${consecutiveCachedRuns} consecutive cached runs, stopping fetch`);
          core.info(`[FETCH] Summary: added ${addedNewRuns} new runs, skipped ${skippedCachedRuns} cached, ${skippedOldRuns} old`);
          return allRuns;
        }
        continue;
      }

      // Reset consecutive cached counter when we find a new run
      if (consecutiveCachedRuns > 0) {
        core.info(`[FETCH] Found new run after ${consecutiveCachedRuns} cached runs, resetting counter`);
      }
      consecutiveCachedRuns = 0;

      // This is a new run, add it
      addedNewRuns++;
      allRuns.push(run);
      if (addedNewRuns <= 10) {
        core.info(`[FETCH] Added new run ${runIdStr} (${run.name}, ${runDate.toISOString()})`);
      }
    }

    // If the api call returned no runs, assume we've reached the end and break
    if (!runs.workflow_runs.length) break;
  }

  core.info(`[FETCH] Completed all pages. Summary: added ${addedNewRuns} new runs, skipped ${skippedCachedRuns} cached, ${skippedOldRuns} old`);
  return allRuns;
}

/**
 * Group runs by workflow name.
 * @param {Array} runs - Array of workflow run objects
 * @returns {Map} Map of workflow name to array of runs
 */
function groupRunsByName(runs) {
  const grouped = new Map();
  for (const run of runs) {
    if (!grouped.has(run.name)) {
      grouped.set(run.name, []);
    }
    grouped.get(run.name).push(run);
  }
  return grouped;
}

/**
 * Find the most recent successful run of aggregate-workflow-data workflow on main branch.
 * @param {object} octokit - Octokit client
 * @param {object} context - GitHub Actions context
 * @returns {Promise<number|null>} Run ID or null if not found
 */
async function findPreviousAggregateRun(octokit, context, branch = '31803-increase-caching-threshold-for-triage-pipeline') {
  try {
    core.info('[CACHE] Searching for previous successful aggregate-workflow-data run...');
    const workflowId = '.github/workflows/aggregate-workflow-data.yaml';
    const resp = await octokit.rest.actions.listWorkflowRuns({
      owner: context.repo.owner,
      repo: context.repo.repo,
      workflow_id: workflowId,
      branch: branch,
      status: 'completed',
      per_page: 10
    });
    const allRuns = resp.data.workflow_runs || [];
    core.info(`[CACHE] Found ${allRuns.length} completed runs for aggregate-workflow-data`);
    const runs = allRuns.filter(r => r.conclusion === 'success');
    core.info(`[CACHE] Found ${runs.length} successful runs`);
    if (runs.length > 0) {
      core.info(`[CACHE] Selected previous run ID: ${runs[0].id} (created: ${runs[0].created_at})`);
      return runs[0].id;
    }
    core.info('[CACHE] No previous successful run found');
    return null;
  } catch (e) {
    core.warning(`[CACHE] Failed to find previous aggregate run: ${e.message}`);
    return null;
  }
}

/**
 * Prune data older than cutoff date from grouped runs.
 * @param {Map} grouped - Map of workflow name to array of runs
 * @param {Date} cutoffDate - Cutoff date
 * @returns {Map} Pruned grouped runs
 */
function pruneOldRuns(grouped, cutoffDate) {
  const pruned = new Map();
  let totalRemoved = 0;
  for (const [name, runs] of grouped.entries()) {
    const beforeCount = runs.length;
    const filtered = runs.filter(run => {
      const runDate = new Date(run.created_at);
      return runDate >= cutoffDate;
    });
    const removed = beforeCount - filtered.length;
    if (removed > 0) {
      totalRemoved += removed;
      const newestRemoved = runs.find(r => new Date(r.created_at) < cutoffDate);
      if (newestRemoved) {
        core.info(`[PRUNE] Workflow '${name}': removed ${removed} runs (newest removed: ${newestRemoved.created_at})`);
      }
    }
    if (filtered.length > 0) {
      pruned.set(name, filtered);
    } else {
      core.info(`[PRUNE] Workflow '${name}': removed all ${beforeCount} runs (workflow has no runs >= ${cutoffDate.toISOString()})`);
    }
  }
  if (totalRemoved > 0) {
    core.info(`[PRUNE] Total runs removed across all workflows: ${totalRemoved}`);
  } else {
    core.info(`[PRUNE] No runs removed (all runs are >= ${cutoffDate.toISOString()})`);
  }
  return pruned;
}

/**
 * Prune annotations index to remove entries for runs older than cutoff date.
 * @param {object} annotationsIndex - Index mapping runId -> directory
 * @param {Map} grouped - Map of workflow name to array of runs (for date lookup)
 * @param {Date} cutoffDate - Cutoff date
 * @returns {object} Pruned annotations index
 */
function pruneAnnotationsIndex(annotationsIndex, grouped, cutoffDate) {
  const pruned = {};
  const runDates = new Map();
  // Build map of run IDs to dates
  for (const runs of grouped.values()) {
    for (const run of runs) {
      runDates.set(String(run.id), new Date(run.created_at));
    }
  }
  for (const [runId, dir] of Object.entries(annotationsIndex || {})) {
    const runDate = runDates.get(runId);
    if (runDate && runDate >= cutoffDate) {
      pruned[runId] = dir;
    }
  }
  return pruned;
}

/**
 * Prune logs index to remove entries for runs older than cutoff date.
 * @param {object} logsIndex - Index mapping runId -> directory
 * @param {Map} grouped - Map of workflow name to array of runs (for date lookup)
 * @param {Date} cutoffDate - Cutoff date
 * @returns {object} Pruned logs index
 */
function pruneLogsIndex(logsIndex, grouped, cutoffDate) {
  const pruned = {};
  const runDates = new Map();
  // Build map of run IDs to dates
  for (const runs of grouped.values()) {
    for (const run of runs) {
      runDates.set(String(run.id), new Date(run.created_at));
    }
  }
  for (const [runId, dir] of Object.entries(logsIndex || {})) {
    const runDate = runDates.get(runId);
    if (runDate && runDate >= cutoffDate) {
      pruned[runId] = dir;
    }
  }
  return pruned;
}

/**
 * Prune commits older than cutoff date.
 * @param {Array} commits - Array of commit objects
 * @param {Date} cutoffDate - Cutoff date
 * @returns {Array} Pruned commits
 */
function pruneCommits(commits, cutoffDate) {
  return (commits || []).filter(c => {
    const commitDate = c.date ? new Date(c.date) : null;
    return commitDate && commitDate >= cutoffDate;
  });
}

/**
 * Check if a workflow name matches any configuration in workflow_configs.
 * @param {string} workflowName - Name of the workflow to check
 * @param {Array} workflowConfigs - Array of config objects with wkflw_name or wkflw_prefix
 * @returns {boolean} True if workflow matches any config
 */
function workflowMatchesConfig(workflowName, workflowConfigs) {
  if (!Array.isArray(workflowConfigs) || workflowConfigs.length === 0) {
    return true; // If no configs provided, match all workflows (backward compatibility)
  }
  for (const config of workflowConfigs) {
    if (config.wkflw_name && workflowName === config.wkflw_name) {
      return true;
    }
    if (config.wkflw_prefix && workflowName.startsWith(config.wkflw_prefix)) {
      return true;
    }
  }
  return false;
}

/**
 * Download and extract an artifact to a temporary directory.
 * @param {object} octokit - Octokit client
 * @param {string} owner - Repository owner
 * @param {string} repo - Repository name
 * @param {object} artifact - Artifact object
 * @param {string} tmpDir - Temporary directory path
 * @returns {Promise<string>} Path to extracted directory
 */
async function downloadAndExtractArtifact(octokit, owner, repo, artifact, tmpDir) {
  const resp = await octokit.rest.actions.downloadArtifact({ owner, repo, artifact_id: artifact.id, archive_format: 'zip' });
  const zipPath = path.join(tmpDir, `${artifact.name}.zip`);
  fs.writeFileSync(zipPath, Buffer.from(resp.data));
  const extractDir = path.join(tmpDir, `${artifact.name}-extract`);
  if (!fs.existsSync(extractDir)) fs.mkdirSync(extractDir, { recursive: true });
  execFileSync('unzip', ['-o', zipPath, '-d', extractDir], { stdio: 'ignore' });
  return extractDir;
}

/**
 * Recursively find a file by name in a directory.
 * @param {string} rootDir - Root directory to search
 * @param {string|Set<string>} targetFileName - File name(s) to find
 * @returns {string|null} Path to found file or null
 */
function findFileInDirectory(rootDir, targetFileName) {
  const targetNames = typeof targetFileName === 'string' ? new Set([targetFileName]) : targetFileName;
  const stack = [rootDir];
  while (stack.length) {
    const dir = stack.pop();
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const ent of entries) {
      const p = path.join(dir, ent.name);
      if (ent.isDirectory()) {
        stack.push(p);
      } else if (ent.isFile() && targetNames.has(ent.name)) {
        return p;
      }
    }
  }
  return null;
}

/**
 * Remove directories in a root directory that are not in the kept set.
 * @param {string} rootDir - Root directory to clean
 * @param {Set<string>} keptIds - Set of directory names to keep
 * @param {string} indexFileName - Name of index file to exclude from removal
 * @returns {number} Number of directories removed
 */
function removeUnkeptDirectories(rootDir, keptIds, indexFileName) {
  if (!fs.existsSync(rootDir)) {
    return 0;
  }
  let removedDirs = 0;
  const entries = fs.readdirSync(rootDir, { withFileTypes: true });
  for (const ent of entries) {
    if (ent.isDirectory() && ent.name !== indexFileName) {
      if (!keptIds.has(ent.name)) {
        fs.rmSync(path.join(rootDir, ent.name), { recursive: true, force: true });
        removedDirs++;
      }
    }
  }
  return removedDirs;
}

/**
 * Main entrypoint for the action.
 * Loads previous cache, fetches new runs, merges/deduplicates, and saves updated cache.
 */
async function run() {
  try {
    // Get inputs
    const branch = core.getInput('branch') || 'main';
    const days = parseInt(core.getInput('days') || DEFAULT_DAYS);
    const rawCachePath = core.getInput('cache-path', { required: false });
    const defaultOutputPath = path.join(process.env.GITHUB_WORKSPACE || process.cwd(), 'workflow-data.json');
    const outputPath = rawCachePath && rawCachePath.trim() ? rawCachePath : defaultOutputPath;
    const workflowConfigsInput = core.getInput('workflow_configs', { required: false });
    let workflowConfigs = [];
    if (workflowConfigsInput) {
      try {
        workflowConfigs = JSON.parse(workflowConfigsInput);
        if (!Array.isArray(workflowConfigs)) {
          core.warning('[CONFIG] workflow_configs is not an array, ignoring');
          workflowConfigs = [];
        } else {
          core.info(`[CONFIG] Loaded ${workflowConfigs.length} workflow configurations`);
        }
      } catch (e) {
        core.warning(`[CONFIG] Failed to parse workflow_configs: ${e.message}`);
        workflowConfigs = [];
      }
    }
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

    // Find and restore artifacts from previous successful run
    const previousRunId = await findPreviousAggregateRun(octokit, github.context, branch);
    if (previousRunId) {
      core.info(`[CACHE] Starting artifact restoration from run ${previousRunId}`);
      try {
        const { data: artifactsResp } = await octokit.rest.actions.listWorkflowRunArtifacts({ owner, repo, run_id: previousRunId, per_page: 100 });
        const artifacts = artifactsResp.artifacts || [];
        core.info(`[CACHE] Found ${artifacts.length} artifacts in previous run`);
        for (const art of artifacts) {
          core.info(`[CACHE]   - ${art.name} (${art.size_in_bytes} bytes, expired: ${art.expired})`);
        }
        const tmpDir = fs.mkdtempSync(path.join(require('os').tmpdir(), 'wfzip-'));
        core.info(`[CACHE] Created temp directory: ${tmpDir}`);

        // Restore workflow-data.json
        const workflowDataArtifact = artifacts.find(a => a && a.name === 'workflow-data');
        if (workflowDataArtifact) {
          try {
            const extractDir = await downloadAndExtractArtifact(octokit, owner, repo, workflowDataArtifact, tmpDir);
            const foundPath = findFileInDirectory(extractDir, new Set(['workflow-data.json', 'workflow.json']));
            if (foundPath) {
              core.info(`[CACHE] Found workflow-data.json at ${foundPath}`);
              const groupedArray = JSON.parse(fs.readFileSync(foundPath, 'utf8'));
              cachedGrouped = new Map(Array.isArray(groupedArray) ? groupedArray : []);
              core.info(`[CACHE] Loaded ${cachedGrouped.size} workflows from cache (before pruning)`);
              core.info(`[CACHE] Cutoff date for pruning: ${cutoffDate.toISOString()} (${days} days ago)`);

              // Count runs before pruning
              let totalRunsBeforePrune = 0;
              let oldestRunDate = null;
              for (const runs of cachedGrouped.values()) {
                totalRunsBeforePrune += runs.length;
                for (const run of runs) {
                  const runDate = new Date(run.created_at);
                  if (!oldestRunDate || runDate < oldestRunDate) {
                    oldestRunDate = runDate;
                  }
                }
              }
              if (oldestRunDate) {
                core.info(`[CACHE] Oldest run in cache: ${oldestRunDate.toISOString()}`);
              }

              // Prune old runs
              cachedGrouped = pruneOldRuns(cachedGrouped, cutoffDate);

              // Count runs after pruning
              let totalRunsAfterPrune = 0;
              for (const runs of cachedGrouped.values()) {
                totalRunsAfterPrune += runs.length;
              }

              core.info(`[CACHE] Summary: ${totalRunsBeforePrune} runs before pruning, ${totalRunsAfterPrune} runs after pruning (removed ${totalRunsBeforePrune - totalRunsAfterPrune})`);
              core.info(`[CACHE] Retained ${totalRunsAfterPrune} runs across ${cachedGrouped.size} workflows`);
            } else {
              core.warning(`[CACHE] workflow-data.json not found in artifacts`);
            }
          } catch (e) {
            core.warning(`Failed to restore workflow-data: ${e.message}`);
          }
        }

        // Restore annotations
        const annotationsArtifact = artifacts.find(a => a && a.name === 'workflow-annotations');
        const annotationsRoot = path.join(workspace, 'annotations');
        if (!fs.existsSync(annotationsRoot)) fs.mkdirSync(annotationsRoot, { recursive: true });
        let annotationsIndexPath = path.join(annotationsRoot, 'annotations-index.json');
        if (annotationsArtifact) {
          try {
            const extractAnnDir = await downloadAndExtractArtifact(octokit, owner, repo, annotationsArtifact, tmpDir);
            const foundIndex = findFileInDirectory(extractAnnDir, 'annotations-index.json');
            if (foundIndex) {
              core.info(`[CACHE] Found annotations-index.json at ${foundIndex}`);
              const artifactAnnRoot = path.dirname(foundIndex);
              fs.cpSync(artifactAnnRoot, annotationsRoot, { recursive: true });
              annotationsIndexPath = path.join(annotationsRoot, 'annotations-index.json');
              const beforePrune = JSON.parse(fs.readFileSync(annotationsIndexPath, 'utf8'));
              const beforePruneCount = Object.keys(beforePrune).length;
              cachedAnnotationsIndex = beforePrune;
              core.info(`[CACHE] Loaded ${beforePruneCount} annotation entries (before pruning)`);

              // Prune old annotations (removes entries from index)
              cachedAnnotationsIndex = pruneAnnotationsIndex(cachedAnnotationsIndex, cachedGrouped, cutoffDate);
              const afterPruneCount = Object.keys(cachedAnnotationsIndex).length;
              core.info(`[CACHE] Pruned ${beforePruneCount - afterPruneCount} annotation entries`);

              // Also remove annotation directories for runs not in pruned index
              const keptRunIds = new Set(Object.keys(cachedAnnotationsIndex));
              const removedDirs = removeUnkeptDirectories(annotationsRoot, keptRunIds, 'annotations-index.json');
              if (removedDirs > 0) {
                core.info(`[CACHE] Removed ${removedDirs} annotation directories for pruned runs`);
              }
              fs.writeFileSync(annotationsIndexPath, JSON.stringify(cachedAnnotationsIndex));
              core.info(`[CACHE] Restored annotations index with ${afterPruneCount} entries`);
            } else {
              core.info(`[CACHE] annotations-index.json not found in artifacts, starting fresh`);
            }
          } catch (e) {
            core.warning(`Failed to restore annotations: ${e.message}`);
          }
        }

        // Restore gtest logs
        const gtestLogsRoot = path.join(workspace, 'logs', 'gtest');
        if (!fs.existsSync(gtestLogsRoot)) fs.mkdirSync(gtestLogsRoot, { recursive: true });
        let gtestLogsIndexPath = path.join(gtestLogsRoot, 'gtest-logs-index.json');
        const gtestLogsArtifact = artifacts.find(a => a && a.name === 'workflow-gtest-logs');
        if (gtestLogsArtifact) {
          try {
            const extractLogsDir = await downloadAndExtractArtifact(octokit, owner, repo, gtestLogsArtifact, tmpDir);
            fs.cpSync(extractLogsDir, gtestLogsRoot, { recursive: true });
            const candidateIdx = path.join(gtestLogsRoot, 'gtest-logs-index.json');
            if (fs.existsSync(candidateIdx)) {
              gtestLogsIndexPath = candidateIdx;
              cachedGtestLogsIndex = JSON.parse(fs.readFileSync(gtestLogsIndexPath, 'utf8'));
              const beforePruneCount = Object.keys(cachedGtestLogsIndex).length;
              core.info(`[CACHE] Loaded ${beforePruneCount} gtest log entries (before pruning)`);

              // Prune old logs (removes entries from index)
              cachedGtestLogsIndex = pruneLogsIndex(cachedGtestLogsIndex, cachedGrouped, cutoffDate);
              const afterPruneCount = Object.keys(cachedGtestLogsIndex).length;
              core.info(`[CACHE] Pruned ${beforePruneCount - afterPruneCount} gtest log entries`);

              // Also remove log directories for runs not in pruned index
              const keptGtestRunIds = new Set(Object.keys(cachedGtestLogsIndex));
              const removedDirs = removeUnkeptDirectories(gtestLogsRoot, keptGtestRunIds, 'gtest-logs-index.json');
              if (removedDirs > 0) {
                core.info(`[CACHE] Removed ${removedDirs} gtest log directories for pruned runs`);
              }
              fs.writeFileSync(gtestLogsIndexPath, JSON.stringify(cachedGtestLogsIndex));
              core.info(`[CACHE] Restored gtest logs index with ${afterPruneCount} entries`);
            }
          } catch (e) {
            core.warning(`Failed to restore gtest logs: ${e.message}`);
          }
        }

        // Restore other logs
        const otherLogsRoot = path.join(workspace, 'logs', 'other');
        if (!fs.existsSync(otherLogsRoot)) fs.mkdirSync(otherLogsRoot, { recursive: true });
        let otherLogsIndexPath = path.join(otherLogsRoot, 'other-logs-index.json');
        const otherLogsArtifact = artifacts.find(a => a && a.name === 'workflow-other-logs');
        if (otherLogsArtifact) {
          try {
            const extractLogsDir = await downloadAndExtractArtifact(octokit, owner, repo, otherLogsArtifact, tmpDir);
            fs.cpSync(extractLogsDir, otherLogsRoot, { recursive: true });
            const candidateIdx = path.join(otherLogsRoot, 'other-logs-index.json');
            if (fs.existsSync(candidateIdx)) {
              otherLogsIndexPath = candidateIdx;
              cachedOtherLogsIndex = JSON.parse(fs.readFileSync(otherLogsIndexPath, 'utf8'));
              const beforePruneCount = Object.keys(cachedOtherLogsIndex).length;
              core.info(`[CACHE] Loaded ${beforePruneCount} other log entries (before pruning)`);

              // Prune old logs (removes entries from index)
              cachedOtherLogsIndex = pruneLogsIndex(cachedOtherLogsIndex, cachedGrouped, cutoffDate);
              const afterPruneCount = Object.keys(cachedOtherLogsIndex).length;
              core.info(`[CACHE] Pruned ${beforePruneCount - afterPruneCount} other log entries`);

              // Also remove log directories for runs not in pruned index
              const keptOtherRunIds = new Set(Object.keys(cachedOtherLogsIndex));
              const removedDirs = removeUnkeptDirectories(otherLogsRoot, keptOtherRunIds, 'other-logs-index.json');
              if (removedDirs > 0) {
                core.info(`[CACHE] Removed ${removedDirs} other log directories for pruned runs`);
              }
              fs.writeFileSync(otherLogsIndexPath, JSON.stringify(cachedOtherLogsIndex));
              core.info(`[CACHE] Restored other logs index with ${afterPruneCount} entries`);
            }
          } catch (e) {
            core.warning(`Failed to restore other logs: ${e.message}`);
          }
        }

        // Restore commits
        const commitsArtifact = artifacts.find(a => a && a.name === 'commits-main');
        let commitsPath = path.join(workspace, 'commits-main.json');
        if (commitsArtifact) {
          try {
            const extractCommitsDir = await downloadAndExtractArtifact(octokit, owner, repo, commitsArtifact, tmpDir);
            const foundCommitsPath = findFileInDirectory(extractCommitsDir, 'commits-main.json');
            if (foundCommitsPath) {
              core.info(`[CACHE] Found commits-main.json at ${foundCommitsPath}`);
              const beforePrune = JSON.parse(fs.readFileSync(foundCommitsPath, 'utf8'));
              const beforePruneCount = beforePrune.length;
              cachedCommits = beforePrune;
              core.info(`[CACHE] Loaded ${beforePruneCount} commits (before pruning)`);

              cachedCommits = pruneCommits(cachedCommits, cutoffDate);
              const afterPruneCount = cachedCommits.length;
              core.info(`[CACHE] Pruned ${beforePruneCount - afterPruneCount} commits older than ${cutoffDate.toISOString()}`);

              fs.writeFileSync(commitsPath, JSON.stringify(cachedCommits));
              core.info(`[CACHE] Restored ${afterPruneCount} commits from cache`);
            } else {
              core.info(`[CACHE] commits-main.json not found in artifacts, starting fresh`);
            }
          } catch (e) {
            core.warning(`Failed to restore commits: ${e.message}`);
          }
        }

        // Restore last success timestamps
        const lastSuccessArtifact = artifacts.find(a => a && a.name === 'last-success-timestamps');
        let lastSuccessPath = path.join(workspace, 'last-success-timestamps.json');
        if (lastSuccessArtifact) {
          try {
            const extractTimestampsDir = await downloadAndExtractArtifact(octokit, owner, repo, lastSuccessArtifact, tmpDir);
            const foundTimestampsPath = findFileInDirectory(extractTimestampsDir, 'last-success-timestamps.json');
            if (foundTimestampsPath) {
              core.info(`[CACHE] Found last-success-timestamps.json at ${foundTimestampsPath}`);
              cachedLastSuccessTimestamps = JSON.parse(fs.readFileSync(foundTimestampsPath, 'utf8'));
              fs.writeFileSync(lastSuccessPath, JSON.stringify(cachedLastSuccessTimestamps));
              core.info(`[CACHE] Restored last success timestamps for ${Object.keys(cachedLastSuccessTimestamps).length} workflows`);
            } else {
              core.info(`[CACHE] last-success-timestamps.json not found in artifacts, starting fresh`);
            }
          } catch (e) {
            core.warning(`Failed to restore last success timestamps: ${e.message}`);
          }
        }

        // Clean up temp directory
        try {
          fs.rmSync(tmpDir, { recursive: true, force: true });
          core.info(`[CACHE] Cleaned up temp directory`);
        } catch (_) { /* ignore */ }

        core.info(`[CACHE] Artifact restoration completed successfully`);
      } catch (e) {
        core.warning(`[CACHE] Failed to restore artifacts from previous run: ${e.message}`);
        core.warning(`[CACHE] Stack trace: ${e.stack}`);
      }
    } else {
      core.info('[CACHE] No previous aggregate run found, starting fresh');
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

    // 1. Fetch runs for each event type separately

    const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

    // Fetch runs from specified workflows (or all workflows if workflowIds not provided)
    core.info('Fetching workflow runs...');
    const allRuns = await fetchAllWorkflowRuns(octokit, github.context, days, cachedRunIds, workflowIds);
    core.info(`[FETCH] Fetched ${allRuns.length} new runs (skipped cached runs during fetch)`);

    // If workflow IDs are provided, we've already fetched all runs for those workflows
    // Otherwise, for backward compatibility, also fetch scheduled runs separately
    let newRuns = allRuns;
    if (!workflowIds || workflowIds.length === 0) {
      // Wait for 1 second to avoid rate limiting
      await delay(1000);
      core.info('[FETCH] Fetching scheduled runs...');
      const scheduledRuns = await fetchAllWorkflowRuns(octokit, github.context, days, cachedRunIds, null, 'schedule');
      core.info(`[FETCH] Fetched ${scheduledRuns.length} new scheduled runs (skipped cached runs during fetch)`);
      // Combine all the results into a single array (already filtered for new runs only)
      newRuns = [...scheduledRuns, ...allRuns];
      core.info(`[FETCH] Total new runs fetched: ${newRuns.length} (${allRuns.length} all events + ${scheduledRuns.length} scheduled)`);
    } else {
      core.info(`[FETCH] Total new runs fetched: ${newRuns.length} (from ${workflowIds.length} workflows)`);
    }

    // Merge and deduplicate by (run id, attempt) tuple to avoid duplicates
    // Note: Same run ID with different attempt numbers are different runs
    core.info(`[MERGE] Merging ${previousRuns.length} cached runs + ${newRuns.length} new runs`);
    const seen = new Map(); // key: `${run.id}:${run_attempt}`, value: run
    // Process newRuns first so they naturally take precedence over previousRuns
    [...newRuns, ...previousRuns].forEach(run => {
      const runId = run.id;
      const attempt = run.run_attempt || 1;
      const key = `${runId}:${attempt}`;
      // If we already have this exact (run id, attempt) combination, keep the first one encountered.
      // Since newRuns are processed first, they will naturally take precedence over previousRuns.
      const existingRun = seen.get(key);
      if (!existingRun) {
        seen.set(key, run);
      } else {
        // An entry with this key already exists. Since newRuns are processed first,
        // the existing entry is either a newRun (which we want to keep) or a previousRun
        // that has no corresponding newRun (which we also want to keep).
        // No action needed, the preferred run is already in 'seen'.
      }
    });
    let mergedRuns = Array.from(seen.values());
    core.info(`[MERGE] After deduplication: ${mergedRuns.length} unique runs (by run id + attempt)`);

    // Only keep runs on main branch, completed, and within the last N days
    const cutoff = getCutoffDate(days);
    const beforeFilter = mergedRuns.length;
    mergedRuns = mergedRuns.filter(run =>
      run.head_branch === branch &&
      run.status === 'completed' &&
      new Date(run.created_at) >= cutoff
    );
    core.info(`[MERGE] After filtering (branch=${branch}, status=completed, date>=${cutoff.toISOString()}): ${mergedRuns.length} runs (removed ${beforeFilter - mergedRuns.length})`);

    // Group runs by workflow name
    const grouped = groupRunsByName(mergedRuns);
    core.info(`[MERGE] Grouped into ${grouped.size} workflows`);

    // Check for newer attempts on the latest run ID for each workflow we care about
    // Filter workflows to only those matching the configuration (if provided)
    const workflowsToRecheck = [];
    if (workflowConfigs.length > 0) {
      for (const [name, runs] of grouped.entries()) {
        if (workflowMatchesConfig(name, workflowConfigs)) {
          workflowsToRecheck.push([name, runs]);
        }
      }
      core.info(`[RECHECK] Filtered to ${workflowsToRecheck.length} workflows matching config (out of ${grouped.size} total)`);
    } else {
      // If no configs provided, check all workflows (backward compatibility)
      workflowsToRecheck.push(...grouped.entries());
      core.info(`[RECHECK] No workflow_configs provided, checking all ${workflowsToRecheck.length} workflows`);
    }

    // For each workflow, find the latest run ID (by date, then by attempt) and check for newer attempts
    const delayBetweenChecks = 100; // Small delay to avoid rate limits
    let updatedAttempts = 0;
    const newRunsToAdd = [];

    // Pre-compute Set of existing (run ID, attempt) combinations for O(1) lookup
    const existingAttemptsSet = new Set(mergedRuns.map(r => `${r.id}:${r.run_attempt || 1}`));

    for (const [workflowName, runs] of workflowsToRecheck) {
      // Find the latest run for this workflow (on target branch, completed)
      const latestRuns = runs
        .filter(r => r.head_branch === branch && r.status === 'completed')
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

      if (latestRuns.length === 0) continue;

      const latestRun = latestRuns[0];
      const latestRunId = latestRun.id;
      const currentAttempt = latestRun.run_attempt || 1;

      // Find the highest attempt we have for this run ID (in case there are multiple attempts in the array)
      let highestAttempt = currentAttempt;
      for (const run of runs) {
        if (run.id === latestRunId) {
          const attempt = run.run_attempt || 1;
          if (attempt > highestAttempt) {
            highestAttempt = attempt;
          }
        }
      }

      try {
        // Fetch the run details directly - GitHub API returns the latest attempt
        const { data: latestRunData } = await octokit.rest.actions.getWorkflowRun({
          owner: github.context.repo.owner,
          repo: github.context.repo.repo,
          run_id: latestRunId
        });

        const apiAttempt = latestRunData.run_attempt || 1;

        // If API has a higher attempt, add it to mergedRuns (don't replace the old one)
        if (apiAttempt > highestAttempt) {
          core.info(`[RECHECK] Found newer attempt for workflow '${workflowName}' run ${latestRunId}: current=${highestAttempt}, API=${apiAttempt}, adding`);
          // Check if the run is still within our date window and meets other criteria
          const runDate = new Date(latestRunData.created_at);
          const cutoff = getCutoffDate(days);
          if (runDate >= cutoff && latestRunData.head_branch === branch && latestRunData.status === 'completed') {
            // Check if we already have this attempt (shouldn't happen, but be safe)
            const alreadyHaveAttempt = existingAttemptsSet.has(`${latestRunId}:${apiAttempt}`);
            if (!alreadyHaveAttempt) {
              // Add the new run to the list (don't replace the old one)
              newRunsToAdd.push(latestRunData);
              updatedAttempts++;
            } else {
              core.info(`[RECHECK] Run ${latestRunId} attempt ${apiAttempt} already in mergedRuns, skipping`);
            }
          } else {
            core.info(`[RECHECK] Run ${latestRunId} attempt ${apiAttempt} is outside date window or doesn't meet criteria, skipping`);
          }
        }

        await delay(delayBetweenChecks); // Small delay to avoid rate limits
      } catch (e) {
        // Run might not exist anymore, or API error - log and continue
        if (e.status !== 404) {
          core.warning(`[RECHECK] Failed to check workflow '${workflowName}' run ${latestRunId} for newer attempts: ${e.message}`);
        }
      }
    }

    // Add the new runs to mergedRuns (both old and new attempts will be present)
    if (newRunsToAdd.length > 0) {
      mergedRuns.push(...newRunsToAdd);
      core.info(`[RECHECK] Added ${updatedAttempts} runs with newer attempts to mergedRuns`);
      // Re-group after adding new runs
      const updatedGrouped = groupRunsByName(mergedRuns);
      grouped.clear();
      for (const [name, runs] of updatedGrouped.entries()) {
        grouped.set(name, runs);
      }
      core.info(`[RECHECK] Re-grouped after adding new attempts: ${grouped.size} workflows`);
    } else {
      core.info(`[RECHECK] No newer attempts found`);
    }

    // Ensure output directory exists
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    // Save grouped runs to artifact file
    fs.writeFileSync(outputPath, JSON.stringify(Array.from(grouped.entries())));

    // Merge cached last success timestamps with new ones (cached takes precedence for existing workflows)
    const lastSuccessTimestamps = new Map(Object.entries(cachedLastSuccessTimestamps || {}));
    core.info(`[LAST_SUCCESS] Starting last success search (cached: ${lastSuccessTimestamps.size} workflows)`);

    // Filter workflows to only those matching the configuration (if provided)
    const workflowsToCheck = [];
    if (workflowConfigs.length > 0) {
      for (const [name, runs] of grouped.entries()) {
        if (workflowMatchesConfig(name, workflowConfigs)) {
          workflowsToCheck.push([name, runs]);
        } else {
          core.info(`[LAST_SUCCESS] Skipping workflow '${name}' (not in workflow_configs)`);
        }
      }
      core.info(`[LAST_SUCCESS] Filtered to ${workflowsToCheck.length} workflows matching config (out of ${grouped.size} total)`);
    } else {
      // If no configs provided, check all workflows (backward compatibility)
      workflowsToCheck.push(...grouped.entries());
      core.info(`[LAST_SUCCESS] No workflow_configs provided, checking all ${workflowsToCheck.length} workflows`);
    }

    for (const [name, runs] of workflowsToCheck) {
      try {
        // Check if latest run on main is failing
        const mainRuns = runs
          .filter(r => r.head_branch === branch)
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

        const latestRun = mainRuns[0];
        if (!latestRun || latestRun.conclusion === 'success') {
          // Workflow is passing, store the success timestamp
          if (latestRun && latestRun.conclusion === 'success') {
            lastSuccessTimestamps.set(name, {
              timestamp: latestRun.created_at,
              sha: latestRun.head_sha,
              run_id: latestRun.id,
              in_window: true
            });
            core.info(`[LAST_SUCCESS] Workflow '${name}' is passing (run ${latestRun.id})`);
          }
          continue;
        }

        // Workflow is currently failing - check if we already have a success in our window
        const successInWindow = mainRuns.find(r => r.conclusion === 'success');
        if (successInWindow) {
          // We already have the info, no API call needed
          lastSuccessTimestamps.set(name, {
            timestamp: successInWindow.created_at,
            sha: successInWindow.head_sha,
            run_id: successInWindow.id,
            in_window: true
          });
          core.info(`[LAST_SUCCESS] Workflow '${name}' failing, found success in window (run ${successInWindow.id})`);
          continue;
        }

        // Workflow is failing and no success in window
        // Check if we have a cached timestamp - if so, reuse it instead of making API calls
        const cachedTimestamp = lastSuccessTimestamps.get(name);
        if (cachedTimestamp && !cachedTimestamp.never_succeeded) {
          // We have a cached timestamp and no new success was found, so reuse the cached one
          core.info(`[LAST_SUCCESS] Workflow '${name}' failing, no new success found, reusing cached timestamp (run ${cachedTimestamp.run_id || 'unknown'}, timestamp: ${cachedTimestamp.timestamp})`);
          // Keep the cached timestamp (it's already in the map)
          continue;
        }

        // No cached timestamp or workflow never succeeded - make targeted API call to search all history
        const workflowPath = runs[0]?.path;
        if (!workflowPath) {
          core.info(`[LAST_SUCCESS] Workflow '${name}' has no path, skipping`);
          continue;
        }

        core.info(`[LAST_SUCCESS] Searching for last success in full history for failing workflow: ${name}${cachedTimestamp ? ' (no cached timestamp found)' : ''}`);

        // Search through workflow history to find the last successful run
        let foundSuccess = false;
        const maxPagesToSearch = 10; // Limit search to up to 1000 most recent runs to avoid excessive API calls

        for (let page = 1; page <= maxPagesToSearch; page++) {
          const { data } = await octokit.rest.actions.listWorkflowRuns({
            owner,
            repo,
            workflow_id: workflowPath,
            branch,
            status: 'completed',
            per_page: 100,
            page
          });

          if (!data.workflow_runs || data.workflow_runs.length === 0) {
            break; // No more runs to check
          }

          // Find first successful run on this page
          const successRun = data.workflow_runs.find(r => r.conclusion === 'success');
          if (successRun) {
            lastSuccessTimestamps.set(name, {
              timestamp: successRun.created_at,
              sha: successRun.head_sha,
              run_id: successRun.id,
              in_window: false
            });
            core.info(`[LAST_SUCCESS] Found last success for '${name}': run ${successRun.id} at ${successRun.created_at}`);
            foundSuccess = true;
            break;
          }

          // If we got fewer runs than requested, we've reached the end
          if (data.workflow_runs.length < 100) {
            break;
          }
        }

        if (!foundSuccess) {
          // Never succeeded in searchable history
          core.info(`[LAST_SUCCESS] No successful run found in history for '${name}' (never succeeded or very old)`);
          lastSuccessTimestamps.set(name, { never_succeeded: true });
        }

        // Small delay to avoid rate limiting
        await delay(500);
      } catch (e) {
        core.warning(`Failed to fetch last success timestamp for ${name}: ${e.message}`);
      }
    }

    // Save the last success timestamps index (merge cached + new)
    const lastSuccessPath = path.join(outputDir, 'last-success-timestamps.json');
    // Update cached timestamps with any new ones found
    const finalLastSuccessTimestamps = Object.fromEntries(lastSuccessTimestamps);
    fs.writeFileSync(lastSuccessPath, JSON.stringify(finalLastSuccessTimestamps));
    core.setOutput('last-success-timestamps-path', lastSuccessPath);
    core.info(`[LAST_SUCCESS] Saved last success timestamps for ${lastSuccessTimestamps.size} workflows to ${lastSuccessPath}`);

    // Download logs for the latest failing run per workflow and build an index
    // Constraint: Only fetch logs when the latest run for that workflow (on target branch)
    // is failing (i.e., conclusion neither success nor skipped/cancelled).
    // The logs will be extracted under a dedicated directory so downstream steps
    // can parse them without performing network calls again.
    // Separate gtest logs from other logs (non-gtest failures with no annotations)
    const annotationsRoot = path.join(workspace, 'annotations');
    const gtestLogsRoot = path.join(workspace, 'logs', 'gtest');
    const otherLogsRoot = path.join(workspace, 'logs', 'other');
    if (!fs.existsSync(annotationsRoot)) {
      fs.mkdirSync(annotationsRoot, { recursive: true });
    }
    if (!fs.existsSync(gtestLogsRoot)) {
      fs.mkdirSync(gtestLogsRoot, { recursive: true });
    }
    if (!fs.existsSync(otherLogsRoot)) {
      fs.mkdirSync(otherLogsRoot, { recursive: true });
    }
    // Initialize indices from cache
    const annotationsIndex = { ...cachedAnnotationsIndex };
    const gtestLogsIndex = { ...cachedGtestLogsIndex };
    const otherLogsIndex = { ...cachedOtherLogsIndex };

    // Build a map of run ID -> workflow name for quick lookup
    const runIdToWorkflowName = new Map();
    for (const [name, runs] of grouped.entries()) {
      for (const run of runs) {
        runIdToWorkflowName.set(String(run.id), name);
      }
    }

    for (const [name, runs] of grouped.entries()) {
      try {
        // First deduplicate by run ID, keeping highest attempt (for logs/annotations, we only want latest attempt)
        const runsByRunId = new Map();
        for (const run of (runs || []).filter(r => r && r.head_branch === branch)) {
          const runId = run.id;
          const attempt = run.run_attempt || 1;
          const existingRun = runsByRunId.get(runId);
          const existingAttempt = existingRun ? (existingRun.run_attempt || 1) : 0;
          if (!existingRun || attempt > existingAttempt) {
            runsByRunId.set(runId, run);
          }
        }

        // Consider only deduplicated runs (latest attempt per run ID) and sort newest first
        const branchRuns = Array.from(runsByRunId.values())
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

        // Scan newest→older, skipping skipped/cancelled runs until either:
        // A) we find a success (stop; no logs), or
        // B) we find a failing run (download logs), or
        // C) we run out of runs (stop; no logs).
        let targetRun = undefined;
        for (const run of branchRuns) {
          const conc = run && run.conclusion;
          if (conc === 'skipped' || conc === 'cancelled') {
            continue; // ignore skipped/cancelled and check next older
          }
          if (conc === 'success') {
            targetRun = undefined; // found a success → do not fetch logs for this workflow
            break;
          }
          // Any other conclusion at this point is considered failing for log fetching purposes
          targetRun = run;
          break;
        }

        // Remove old annotations/logs for this workflow (only keep the latest failing run)
        const runsToKeep = targetRun ? new Set([String(targetRun.id)]) : new Set();
        let removedAnnotations = 0;
        let removedGtestLogs = 0;
        let removedOtherLogs = 0;

        // Remove old annotations for this workflow
        for (const [runId, dir] of Object.entries(annotationsIndex)) {
          if (runIdToWorkflowName.get(runId) === name && !runsToKeep.has(runId)) {
            delete annotationsIndex[runId];
            removedAnnotations++;
            // Remove directory from disk (dir is relative to workspace)
            const fullPath = path.join(workspace, dir);
            try {
              if (fs.existsSync(fullPath)) {
                fs.rmSync(fullPath, { recursive: true, force: true });
                core.info(`[LOGS] Removed old annotations directory for run ${runId} (workflow: ${name})`);
              }
            } catch (e) {
              core.warning(`[LOGS] Failed to remove old annotations directory ${fullPath}: ${e.message}`);
            }
          }
        }

        // Remove old gtest logs for this workflow
        for (const [runId, dir] of Object.entries(gtestLogsIndex)) {
          if (runIdToWorkflowName.get(runId) === name && !runsToKeep.has(runId)) {
            delete gtestLogsIndex[runId];
            removedGtestLogs++;
            // Remove directory from disk (dir is relative to workspace)
            const fullPath = path.join(workspace, dir);
            try {
              if (fs.existsSync(fullPath)) {
                fs.rmSync(fullPath, { recursive: true, force: true });
                core.info(`[LOGS] Removed old gtest logs directory for run ${runId} (workflow: ${name})`);
              }
            } catch (e) {
              core.warning(`[LOGS] Failed to remove old gtest logs directory ${fullPath}: ${e.message}`);
            }
          }
        }

        // Remove old other logs for this workflow
        for (const [runId, dir] of Object.entries(otherLogsIndex)) {
          if (runIdToWorkflowName.get(runId) === name && !runsToKeep.has(runId)) {
            delete otherLogsIndex[runId];
            removedOtherLogs++;
            // Remove directory from disk (dir is relative to workspace)
            const fullPath = path.join(workspace, dir);
            try {
              if (fs.existsSync(fullPath)) {
                fs.rmSync(fullPath, { recursive: true, force: true });
                core.info(`[LOGS] Removed old other logs directory for run ${runId} (workflow: ${name})`);
              }
            } catch (e) {
              core.warning(`[LOGS] Failed to remove old other logs directory ${fullPath}: ${e.message}`);
            }
          }
        }

        if (removedAnnotations > 0 || removedGtestLogs > 0 || removedOtherLogs > 0) {
          core.info(`[LOGS] Workflow '${name}': removed ${removedAnnotations} old annotation entries, ${removedGtestLogs} old gtest log entries, ${removedOtherLogs} old other log entries`);
        }

        // If workflow is passing (no targetRun), we're done (already cleaned up old logs)
        if (!targetRun) {
          if (removedAnnotations > 0 || removedGtestLogs > 0 || removedOtherLogs > 0) {
            core.info(`[LOGS] Workflow '${name}' is now passing, cleaned up old logs`);
          }
          continue;
        }

        // Skip if this exact (run ID, attempt) combination is already cached
        // Note: Different attempts of the same run ID need different logs/annotations
        const targetRunIdStr = String(targetRun.id);
        const targetRunAttempt = targetRun.run_attempt || 1;
        const targetRunKey = `${targetRunIdStr}:${targetRunAttempt}`;
        if (cachedRunAttempts.has(targetRunKey)) {
          core.info(`[LOGS] Skipping download for run ${targetRunIdStr} attempt ${targetRunAttempt} (workflow: ${name}) - already in cache`);
          continue;
        }

        core.info(`[LOGS] Processing failing run ${targetRun.id} for workflow '${name}' (conclusion: ${targetRun.conclusion})`);

        // Fetch check-run annotations for this failing run
        let sawGtestFailure = false;
        let sawAnyFailureAnnotations = false;
        let annotationsFetchFailed = false;
        const gtestJobNames = new Set();
        try {
          // List jobs and extract check_run_ids
          const jobsResp = await octokit.rest.actions.listJobsForWorkflowRun({ owner, repo, run_id: targetRun.id, per_page: 100 });
          const jobs = Array.isArray(jobsResp.data.jobs) ? jobsResp.data.jobs : [];
          const checkRunIds = [];
          for (const j of jobs) {
            const cru = j && j.check_run_url;
            if (typeof cru === 'string' && cru.includes('/check-runs/')) {
              const idStr = cru.split('/check-runs/')[1];
              const id = idStr ? parseInt(idStr, 10) : NaN;
              if (!Number.isNaN(id)) checkRunIds.push({ id, job_name: j.name });
            }
          }
          const annRoot = path.join(workspace, 'annotations', String(targetRun.id));
          if (!fs.existsSync(annRoot)) fs.mkdirSync(annRoot, { recursive: true });
          const allAnnotations = [];
          for (const { id: checkRunId, job_name } of checkRunIds) {
            let page = 1;
            const budget = 500; // safety cap per run
            while (allAnnotations.length < budget) {
              const perPage = Math.min(100, budget - allAnnotations.length);
              const { data } = await octokit.rest.checks.listAnnotations({ owner, repo, check_run_id: checkRunId, per_page: perPage, page });
              const arr = Array.isArray(data) ? data : [];
              if (!arr.length) break;
              for (const a of arr) {
                allAnnotations.push({
                  job_name,
                  path: a.path,
                  start_line: a.start_line,
                  end_line: a.end_line,
                  annotation_level: a.annotation_level,
                  message: a.message,
                  title: a.title,
                  raw_details: a.raw_details,
                });
                // If any failing/error annotation belongs to a job that looks like gtest, note it
                try {
                  const levelLc = String(a.annotation_level || '').toLowerCase();
                  const msgTrim = String(a.message || '').trim();
                  const msgLc = msgTrim.toLowerCase();
                  const isUnknownFileLead = msgLc.startsWith('unknown file');
                  // Primary heuristic: failing/error annotation whose message starts with 'unknown file' is a gtest indicator
                  if (levelLc === 'failure' || levelLc === 'error') {
                    sawAnyFailureAnnotations = true;
                    if (isUnknownFileLead) {
                      sawGtestFailure = true;
                      if (job_name) gtestJobNames.add(String(job_name));
                    }
                    // Keep legacy job-name heuristic as a secondary signal
                    else if ((/gtest/i.test(String(job_name || '')) || /gtests/i.test(String(job_name || '')))) {
                      sawGtestFailure = true;
                      if (job_name) gtestJobNames.add(String(job_name));
                    }
                  }
                } catch (_) { /* ignore */ }
              }
              if (arr.length < perPage) break;
              page += 1;
            }
          }
          const annPath = path.join(annRoot, 'annotations.json');
          fs.writeFileSync(annPath, JSON.stringify(allAnnotations));
          const relativeAnn = path.relative(workspace, annRoot) || annRoot;
          annotationsIndex[String(targetRun.id)] = relativeAnn;
          core.info(`[LOGS] Fetched annotations for run ${targetRun.id} → ${allAnnotations.length} items`);
        } catch (e) {
          core.warning(`Failed to fetch annotations for run ${targetRun.id}: ${e.message}`);
          annotationsFetchFailed = true;
        }

        // Download logs if: gtest failure detected OR no error/failure annotations found OR annotations fetch failed
        // Separate gtest logs from other logs into different directories
        try {
          if (sawGtestFailure) {
            // Download and index gtest logs
            const runLogsZip = await octokit.rest.actions.downloadWorkflowRunLogs({ owner, repo, run_id: targetRun.id });
            const runDir = path.join(gtestLogsRoot, String(targetRun.id));
            if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });
            const zipPath = path.join(runDir, `logs-${targetRun.id}.zip`);
            fs.writeFileSync(zipPath, Buffer.from(runLogsZip.data));
            const extractDir = path.join(runDir, 'extract');
            if (!fs.existsSync(extractDir)) fs.mkdirSync(extractDir, { recursive: true });
            // Extract quietly to avoid ENOBUFS
            execFileSync('unzip', ['-o', zipPath, '-d', extractDir], { stdio: 'ignore' });

            // Helper to sanitize strings for fuzzy file matching
            const sanitize = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
            const wanted = new Map();
            // Add any job names discovered via 'unknown file' annotations (gtest jobs)
            for (const jn of Array.from(gtestJobNames.values())) {
              const key = sanitize(jn);
              if (!wanted.has(key)) wanted.set(key, { name: jn, files: [] });
            }
            if (wanted.size === 0) {
              // If we didn't identify explicit gtest jobs from jobs API, fall back to any file containing 'gtest'
              wanted.set('gtest', { name: 'gtest', files: [] });
            }
            // Walk extracted tree and record .txt files that match wanted keys
            const stack = [extractDir];
            while (stack.length) {
              const dir = stack.pop();
              const entries = fs.readdirSync(dir, { withFileTypes: true });
              for (const ent of entries) {
                const p = path.join(dir, ent.name);
                if (ent.isDirectory()) {
                  stack.push(p);
                } else if (ent.isFile() && /\.txt$/i.test(ent.name)) {
                  const fileKey = sanitize(p);
                  for (const [k, rec] of wanted.entries()) {
                    if (fileKey.includes(k)) {
                      const rel = path.relative(runDir, p);
                      if (!rec.files.includes(rel)) rec.files.push(rel);
                    }
                  }
                }
              }
            }
            const jobsIndex = { jobs: Array.from(wanted.values()).filter(j => (j.files || []).length > 0) };
            const jobsIndexPath = path.join(runDir, 'jobs.json');
            fs.writeFileSync(jobsIndexPath, JSON.stringify(jobsIndex));
            const relativeRunDir = path.relative(workspace, runDir) || runDir;
            gtestLogsIndex[String(targetRun.id)] = relativeRunDir;
            core.info(`[LOGS] Downloaded and indexed gtest logs for run ${targetRun.id} → ${jobsIndex.jobs.length} job(s), ${jobsIndex.jobs.reduce((sum, j) => sum + (j.files || []).length, 0)} files`);
          } else if (!sawAnyFailureAnnotations || annotationsFetchFailed) {
            // Download other logs (non-gtest failures with no annotations, or if annotation fetch failed entirely)
            // Just download and list files, no detailed indexing
            const runLogsZip = await octokit.rest.actions.downloadWorkflowRunLogs({ owner, repo, run_id: targetRun.id });
            const runDir = path.join(otherLogsRoot, String(targetRun.id));
            if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });
            const zipPath = path.join(runDir, `logs-${targetRun.id}.zip`);
            fs.writeFileSync(zipPath, Buffer.from(runLogsZip.data));
            const extractDir = path.join(runDir, 'extract');
            if (!fs.existsSync(extractDir)) fs.mkdirSync(extractDir, { recursive: true });
            // Extract quietly to avoid ENOBUFS
            execFileSync('unzip', ['-o', zipPath, '-d', extractDir], { stdio: 'ignore' });

            // Build a simple index of all .txt log files (no parsing, just list files)
            const logFiles = [];
            const stack = [extractDir];
            while (stack.length) {
              const dir = stack.pop();
              const entries = fs.readdirSync(dir, { withFileTypes: true });
              for (const ent of entries) {
                const p = path.join(dir, ent.name);
                if (ent.isDirectory()) {
                  stack.push(p);
                } else if (ent.isFile() && /\.txt$/i.test(ent.name)) {
                  const rel = path.relative(runDir, p);
                  logFiles.push(rel);
                }
              }
            }
            const logsListPath = path.join(runDir, 'logs-list.json');
            fs.writeFileSync(logsListPath, JSON.stringify({ files: logFiles }));
            const relativeRunDir = path.relative(workspace, runDir) || runDir;
            otherLogsIndex[String(targetRun.id)] = relativeRunDir;
            core.info(`[LOGS] Downloaded other logs for run ${targetRun.id} → ${logFiles.length} file(s)`);
          }
        } catch (e) {
          core.warning(`Failed to download/index logs for run ${targetRun.id}: ${e.message}`);
        }
      } catch (e) {
        core.warning(`Failed to fetch logs for latest failing run in workflow '${name}': ${e.message}`);
      }
    }
    // Persist annotations index alongside annotations directory
    const annotationsIndexPath = path.join(annotationsRoot, 'annotations-index.json');
    fs.writeFileSync(annotationsIndexPath, JSON.stringify(annotationsIndex));

    // Persist gtest logs index
    const gtestLogsIndexPath = path.join(gtestLogsRoot, 'gtest-logs-index.json');
    try {
      fs.writeFileSync(gtestLogsIndexPath, JSON.stringify(gtestLogsIndex));
    } catch (e) {
      core.warning(`Failed to write gtest logs index: ${e.message}`);
    }

    // Persist other logs index
    const otherLogsIndexPath = path.join(otherLogsRoot, 'other-logs-index.json');
    try {
      fs.writeFileSync(otherLogsIndexPath, JSON.stringify(otherLogsIndex));
    } catch (e) {
      core.warning(`Failed to write other logs index: ${e.message}`);
    }

    // Build a commits index for the main branch within the last N days
    // The index is an array of commits sorted by commit author/commit date ascending.
    // Each entry: { sha, short, url, author_login, author_name, author_url, date }
    // Start with cached commits and merge with new ones
    core.info(`[COMMITS] Starting commits fetch (cached: ${cachedCommits.length})`);
    const cachedCommitsSet = new Map((cachedCommits || []).map(c => [c.sha, c]));
    // Initialize commits as empty array to avoid keeping cached commits in memory twice
    // We'll add cached commits back at the end if needed, or push them as we process
    const commits = [];
    let newCommitsCount = 0;
    let skippedCachedCommits = 0;
    let consecutiveCachedCommits = 0;
    const MAX_CONSECUTIVE_CACHED = 50; // Stop after 50 consecutive cached commits
    try {
      const sinceIso = getCutoffDate(days).toISOString();
      core.info(`[COMMITS] Fetching commits since ${sinceIso}`);
      const perPage = 100;
      let page = 1;
      while (true) {
        const resp = await octokit.rest.repos.listCommits({
          owner,
          repo,
          sha: branch,
          since: sinceIso,
          per_page: perPage,
          page,
        });
        const arr = resp.data || [];
        if (!arr.length) {
          core.info(`[COMMITS] No more commits (page ${page})`);
          break;
        }
        core.info(`[COMMITS] Fetched page ${page}: ${arr.length} commits`);
        let shouldBreak = false;
        for (const c of arr) {
          const sha = c.sha;
          // Skip if already in cache
          if (cachedCommitsSet.has(sha)) {
            consecutiveCachedCommits++;
            skippedCachedCommits++;
            if (skippedCachedCommits <= 5) {
              core.info(`[COMMITS] Skipping cached commit ${sha.substring(0, 7)}`);
            }
            // If we've seen many consecutive cached commits, we've likely reached the boundary
            // Commits are returned newest-first, so all subsequent commits will also be cached
            if (consecutiveCachedCommits >= MAX_CONSECUTIVE_CACHED) {
              core.info(`[COMMITS] Early exit: found ${consecutiveCachedCommits} consecutive cached commits, stopping fetch`);
              core.info(`[COMMITS] Summary: added ${newCommitsCount} new commits, skipped ${skippedCachedCommits} cached`);
              shouldBreak = true;
              break;
            }
            continue;
          }

          // Reset consecutive cached counter when we find a new commit
          if (consecutiveCachedCommits > 0) {
            core.info(`[COMMITS] Found new commit after ${consecutiveCachedCommits} cached commits, resetting counter`);
          }
          consecutiveCachedCommits = 0;

          newCommitsCount++;
          const short = sha ? sha.substring(0, 7) : '';
          const url = `https://github.com/${owner}/${repo}/commit/${sha}`;
          const author_login = c.author?.login;
          const author_name = c.commit?.author?.name;
          const author_url = c.author?.html_url;
          const date = c.commit?.author?.date || c.commit?.committer?.date || null;
          const message = c.commit?.message || '';
          const description = typeof message === 'string' ? (message.split(/\r?\n/)[0] || '') : '';
          const commitObj = { sha, short, url, author_login, author_name, author_url, date, message, description };
          commits.push(commitObj);
          cachedCommitsSet.set(sha, commitObj);
          if (newCommitsCount <= 10) {
            core.info(`[COMMITS] Added new commit ${sha.substring(0, 7)} (${date || 'no date'})`);
          }
        }

        // Break if we hit the early exit condition
        if (shouldBreak) {
          break;
        }

        if (arr.length < perPage) break;
        page++;
      }

      if (skippedCachedCommits > 5) {
        core.info(`[COMMITS] Skipped ${skippedCachedCommits} cached commits (showing first 5)`);
      }
      // Add all cached commits to the array (they've already been pruned by date)
      commits.push(...cachedCommits);

      // Sort oldest -> newest by date for deterministic slicing
      commits.sort((a, b) => new Date(a.date || 0) - new Date(b.date || 0));
      core.info(`[COMMITS] Total commits: ${commits.length}; cached: ${cachedCommits.length}; new: ${newCommitsCount}`);
    } catch (e) {
      core.warning(`[COMMITS] Failed to build commits index: ${e.message}`);
    }
    const commitsPath = path.join(workspace, 'commits-main.json');
    fs.writeFileSync(commitsPath, JSON.stringify(commits));
    core.info(`[COMMITS] Saved commits index to ${commitsPath}`);

    // Set output
    core.info(`[OUTPUT] Setting action outputs...`);
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
    core.info(`[OUTPUT] annotations: ${Object.keys(annotationsIndex).length} entries`);
    core.info(`[OUTPUT] gtest-logs: ${Object.keys(gtestLogsIndex).length} entries`);
    core.info(`[OUTPUT] other-logs: ${Object.keys(otherLogsIndex).length} entries`);

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
