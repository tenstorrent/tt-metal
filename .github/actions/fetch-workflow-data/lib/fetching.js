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
  const twoWeeksAgo = getCutoffDate(15); // For early exit check
  const cachedIds = cachedRunIds || new Set();

  core.info(`[FETCH] cutoffDate: ${cutoffDate.toISOString()}`);
  core.info(`[FETCH] days: ${days}, cachedRunIds: ${cachedIds.size}, eventType: ${eventType || 'all'}, branch: ${branch || 'all'}, status: ${status || 'all'}`);
  core.info(`[FETCH] workflowIds: ${workflowIds ? workflowIds.length + ' workflows' : 'all workflows (backward compatibility)'}`);

  const MAX_CONSECUTIVE_CACHED = 100; // Stop after 100 consecutive cached runs
  let totalSkippedOldRuns = 0;
  let totalSkippedCachedRuns = 0;
  let totalAddedNewRuns = 0;

  // Helper function to delay execution
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

  // If workflow IDs are provided, fetch runs for each workflow specifically
  if (workflowIds && workflowIds.length > 0) {
    core.info(`[FETCH] Fetching runs for ${workflowIds.length} specific workflows`);

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

/**
 * Helper function to delay execution
 * @param {number} ms - Milliseconds to delay
 * @returns {Promise} Promise that resolves after delay
 */
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Remove manually triggered workflow runs.
 * @param {Array} runs - Runs to filter
 * @param {string} source - Label used in logs
 * @returns {Array} Filtered runs
 */
function filterManualRuns(runs, source) {
  if (!Array.isArray(runs) || runs.length === 0) {
    return runs;
  }
  const filteredRuns = runs.filter(run => run.event !== 'workflow_dispatch');
  const removed = runs.length - filteredRuns.length;
  if (removed > 0) {
    core.info(`[${source}] Filtered out ${removed} manually triggered runs (workflow_dispatch)`);
  }
  return filteredRuns;
}

/**
 * Fetches new workflow runs from GitHub API
 * @param {object} octokit - Octokit client
 * @param {object} context - GitHub Actions context
 * @param {number} days - Number of days to look back
 * @param {Set} cachedRunIds - Set of cached run IDs
 * @param {Array|null} workflowIds - Array of workflow IDs to fetch, or null for all
 * @param {string} branch - Branch to filter runs by
 * @returns {Promise<Array>} Array of new workflow runs
 */
async function fetchNewWorkflowRuns(octokit, context, days, cachedRunIds, workflowIds, branch) {
  // Fetch runs from specified workflows (or all workflows if workflowIds not provided)
  // Filter at API level: only fetch completed runs on the target branch
  core.info('Fetching workflow runs...');
  const allRuns = await fetchAllWorkflowRuns(octokit, context, days, cachedRunIds, workflowIds, '', branch, 'completed');
  core.info(`[FETCH] Fetched ${allRuns.length} new runs (skipped cached runs during fetch)`);

  // If workflow IDs are provided, we've already fetched all runs for those workflows
  // Otherwise, for backward compatibility, also fetch scheduled runs separately
  let newRuns = allRuns;
  if (!workflowIds || workflowIds.length === 0) {
    // Wait for 1 second to avoid rate limiting
    await delay(1000);
    core.info('[FETCH] Fetching scheduled runs...');
    const scheduledRuns = await fetchAllWorkflowRuns(octokit, context, days, cachedRunIds, null, 'schedule', branch, 'completed');
    core.info(`[FETCH] Fetched ${scheduledRuns.length} new scheduled runs (skipped cached runs during fetch)`);
    // Combine all the results into a single array (already filtered for new runs only)
    newRuns = [...scheduledRuns, ...allRuns];
    core.info(`[FETCH] Total new runs fetched: ${newRuns.length} (${allRuns.length} all events + ${scheduledRuns.length} scheduled)`);
  } else {
    core.info(`[FETCH] Total new runs fetched: ${newRuns.length} (from ${workflowIds.length} workflows)`);
  }

  return filterManualRuns(newRuns, 'FETCH');
}

/**
 * Merges and deduplicates runs by (run id, attempt) tuple
 * @param {Array} previousRuns - Array of previous/cached runs
 * @param {Array} newRuns - Array of newly fetched runs
 * @param {number} days - Number of days to look back for date filtering
 * @returns {Array} Array of merged and deduplicated runs
 */
function mergeAndDeduplicateRuns(previousRuns, newRuns, days) {
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
    }
  });
  let mergedRuns = Array.from(seen.values());
  core.info(`[MERGE] After deduplication: ${mergedRuns.length} unique runs (by run id + attempt)`);

  // Filter by date (branch and status are already filtered at API level)
  const cutoff = getCutoffDate(days);
  const beforeDateFilter = mergedRuns.length;
  mergedRuns = mergedRuns.filter(run =>
    new Date(run.created_at) >= cutoff
  );
  core.info(`[MERGE] After filtering (date>=${cutoff.toISOString()}): ${mergedRuns.length} runs (removed ${beforeDateFilter - mergedRuns.length})`);

  mergedRuns = filterManualRuns(mergedRuns, 'MERGE');
  core.info(`[MERGE] Final count after all filtering: ${mergedRuns.length} runs`);
  core.info(`[MERGE] Note: branch and status=completed were already filtered at API level`);

  return mergedRuns;
}

/**
 * Rechecks workflows for newer run attempts and adds them to the grouped runs
 * @param {object} octokit - Octokit client
 * @param {object} context - GitHub Actions context
 * @param {Map} grouped - Map of workflow names to their runs
 * @param {string} branch - Branch to filter runs by
 * @param {number} days - Number of days to look back
 * @param {Set} existingAttemptsSet - Set of existing (run ID, attempt) combinations
 * @returns {Promise<Array>} Array of new runs with newer attempts to add
 */
async function recheckForNewerAttempts(octokit, context, grouped, branch, days, existingAttemptsSet) {
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

  // Process all workflows (we only fetched runs for workflows we care about)
  const workflowsToRecheck = Array.from(grouped.entries());
  core.info(`[RECHECK] Checking all ${workflowsToRecheck.length} workflows for newer attempts`);

  // For each workflow, find the latest run ID (by date, then by attempt) and check for newer attempts
  const delayBetweenChecks = 100; // Small delay to avoid rate limits
  let updatedAttempts = 0;
  const newRunsToAdd = [];

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
        owner: context.repo.owner,
        repo: context.repo.repo,
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

  if (newRunsToAdd.length > 0) {
    core.info(`[RECHECK] Found ${updatedAttempts} runs with newer attempts`);
  } else {
    core.info(`[RECHECK] No newer attempts found`);
  }

  return newRunsToAdd;
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
  fetchNewWorkflowRuns,
  mergeAndDeduplicateRuns,
  recheckForNewerAttempts,
};
