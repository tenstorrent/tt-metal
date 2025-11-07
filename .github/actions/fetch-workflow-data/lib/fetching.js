// Fetching Module
// Handles constants, date utils, workflow fetching, grouping, merging, filtering, attempt recheck

const core = require('@actions/core');

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
 * @param {string} branch - Optional branch filter (e.g., 'main')
 * @param {string} status - Optional status filter (e.g., 'completed')
 * @returns {Promise<Array>} Array of workflow run objects (only new, non-cached runs)
 */
async function fetchAllWorkflowRuns(github, context, days, cachedRunIds = null, workflowIds = null, eventType='', branch = null, status = null) {
  const allRuns = [];
  const cutoffDate = getCutoffDate(days);
  const twoWeeksAgo = getCutoffDate(14); // For early exit check
  const cachedIds = cachedRunIds || new Set();

  core.info(`[FETCH] cutoffDate: ${cutoffDate.toISOString()}`);
  core.info(`[FETCH] days: ${days}, cachedRunIds: ${cachedIds.size}, eventType: ${eventType || 'all'}, branch: ${branch || 'all'}, status: ${status || 'all'}`);
  core.info(`[FETCH] workflowIds: ${workflowIds ? workflowIds.length + ' workflows' : 'all workflows (backward compatibility)'}`);

  const MAX_CONSECUTIVE_CACHED = 100; // Stop after 50 consecutive cached runs
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
        if (branch) {
          params.branch = branch;
        }
        if (status) {
          params.status = status;
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
    if (branch) {
      params.branch = branch;
    }
    if (status) {
      params.status = status;
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

module.exports = {
  // Constants
  MAX_PAGES,
  RUNS_PER_PAGE,
  DEFAULT_DAYS,
  // Functions
  getCutoffDate,
  fetchAllWorkflowRuns,
  groupRunsByName,
};
