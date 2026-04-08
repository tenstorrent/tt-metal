// Data Building Module
// Handles commits index building and last success timestamp tracking

const core = require('@actions/core');
const fs = require('fs');
const path = require('path');

const fetching = require('./fetching');
const { getCutoffDate } = fetching;

/**
 * Updates last success timestamps for workflows
 * @param {Map} grouped - Map of workflow names to their runs
 * @param {string} branch - Branch to filter runs by
 * @param {object} octokit - Octokit client
 * @param {object} context - GitHub Actions context
 * @param {object} cachedLastSuccessTimestamps - Cached last success timestamps
 * @param {string} workspace - Workspace directory path
 * @param {string} outputDir - Output directory path
 * @returns {Promise<string>} Path to the saved last success timestamps file
 */
async function updateLastSuccessTimestamps(grouped, branch, octokit, context, cachedLastSuccessTimestamps, workspace, outputDir) {
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  // Merge cached last success timestamps with new ones (cached takes precedence for existing workflows)
  const lastSuccessTimestamps = new Map(Object.entries(cachedLastSuccessTimestamps || {}));
  core.info(`[LAST_SUCCESS] Starting last success search (cached: ${lastSuccessTimestamps.size} workflows)`);

  // Process all workflows (we only fetched runs for workflows we care about)
  const workflowsToCheck = Array.from(grouped.entries());
  core.info(`[LAST_SUCCESS] Checking all ${workflowsToCheck.length} workflows`);

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

  return lastSuccessPath;
}

/**
 * Builds a commits index for the main branch within the last N days
 * @param {Array} cachedCommits - Cached commits array
 * @param {number} days - Number of days to look back
 * @param {string} branch - Branch to fetch commits from
 * @param {object} octokit - Octokit client
 * @param {object} context - GitHub Actions context
 * @param {string} workspace - Workspace directory path
 * @returns {Promise<string>} Path to the saved commits index file
 */
async function buildCommitsIndex(cachedCommits, days, branch, octokit, context, workspace) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;

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
  const MAX_CONSECUTIVE_CACHED = 100; // Stop after 100 consecutive cached commits (unified with runs threshold in fetching.js)
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

  return commitsPath;
}

module.exports = {
  updateLastSuccessTimestamps,
  buildCommitsIndex,
};
