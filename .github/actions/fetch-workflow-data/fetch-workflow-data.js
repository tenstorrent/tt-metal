// Fetch Workflow Data GitHub Action
// This action fetches workflow run data and caches it for analysis
//
// See: https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28

const core = require('@actions/core');
const github = require('@actions/github');
const fs = require('fs');

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
 * Get the latest created_at date from cached runs.
 * @param {Array} runs - Array of workflow run objects
 * @returns {Date} The latest created_at date, or epoch if none
 */
function getLatestCachedDate(runs) {
  if (!runs.length) return new Date(0); // Safe default: epoch
  // Map to timestamps, filter out invalid dates
  const times = runs
    .map(run => new Date(run.created_at).getTime())
    .filter(t => !isNaN(t) && t > 0);
  if (!times.length) return new Date(0);
  return new Date(Math.max(...times));
}

/**
 * Fetch all workflow runs for the repository, paginated, stopping at sinceDate if provided.
 * @param {object} github - Octokit client
 * @param {object} context - GitHub Actions context
 * @param {number} days - Number of days to look back
 * @param {Date} sinceDate - Only fetch runs after this date
 * @returns {Promise<Array>} Array of workflow run objects
 */
async function fetchAllWorkflowRuns(github, context, days, sinceDate) {
  const allRuns = [];
  const cutoffDate = getCutoffDate(days);
  core.info(`days ${days}, sinceDate: ${sinceDate}`);
  for (let page = 1; page <= MAX_PAGES; page++) {
    const { data: runs } = await github.rest.actions.listWorkflowRunsForRepo({
      owner: context.repo.owner,
      repo: context.repo.repo,
      per_page: RUNS_PER_PAGE,
      page
    });
    if (!runs.workflow_runs.length) {
      break;
    }
    for (const run of runs.workflow_runs) {
      const runDate = new Date(run.created_at);
      if (sinceDate && runDate <= sinceDate) {
        core.info(`Early exit: found run at ${runDate} <= latest cached date ${sinceDate}`);
        return allRuns;
      }
      if (runDate >= cutoffDate) {
        allRuns.push(run);
      }
    }
    // If we got fewer runs than requested, we've reached the end
    if (runs.workflow_runs.length < RUNS_PER_PAGE) break;
  }
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
 * Main entrypoint for the action.
 * Loads previous cache, fetches new runs, merges/deduplicates, and saves updated cache.
 */
async function run() {
  try {
    // Get inputs
    const branch = core.getInput('branch') || 'main';
    const days = parseInt(core.getInput('days') || DEFAULT_DAYS);
    const cachePath = core.getInput('cache-path', { required: true });
    // Create authenticated Octokit client
    const octokit = github.getOctokit(core.getInput('GITHUB_TOKEN', { required: true }));
    // Load previous cache if it exists
    let previousRuns = [];
    let latestCachedDate = null;
    if (fs.existsSync(cachePath)) {
      try {
        const rawCache = fs.readFileSync(cachePath, 'utf8');
        const prev = JSON.parse(rawCache);
        if (Array.isArray(prev)) {
          if (prev.length && Array.isArray(prev[0])) {
            // Array of [name, runs[]] pairs
            previousRuns = prev.flatMap(([_, runs]) => runs);
          } else if (prev.length && prev[0] && prev[0].id) {
            // Array of runs
            previousRuns = prev;
          } else {
            previousRuns = [];
          }
        } else if (typeof prev === 'object' && prev !== null) {
          previousRuns = Object.values(prev).flat();
        } else {
          previousRuns = [];
        }
        latestCachedDate = getLatestCachedDate(previousRuns);
      } catch (e) {
        core.warning('Could not parse previous cache, ignoring.');
      }
    }
    core.info(`Restored previousRuns count: ${previousRuns.length}`);
    core.info(`Latest cached run date: ${latestCachedDate}`);
    // Fetch new runs from GitHub (for the last N days, only after latest cached run)
    const newRuns = await fetchAllWorkflowRuns(octokit, github.context, days, latestCachedDate);
    core.info(`Fetched newRuns count: ${newRuns.length}`);
    // Merge and deduplicate by run id
    // This ensures we keep the most recent data for each run and avoid duplicates
    const seen = new Map();
    [...previousRuns, ...newRuns].forEach(run => seen.set(run.id, run));
    let mergedRuns = Array.from(seen.values());
    // Only keep runs on main branch, completed, and within the last N days
    const cutoff = getCutoffDate(days);
    mergedRuns = mergedRuns.filter(run =>
      run.head_branch === branch &&
      run.status === 'completed' &&
      new Date(run.created_at) >= cutoff
    );
    // Group runs by workflow name
    const grouped = groupRunsByName(mergedRuns);
    // Ensure cache directory exists
    const cacheDir = require('path').dirname(cachePath);
    if (!fs.existsSync(cacheDir)) {
      fs.mkdirSync(cacheDir, { recursive: true });
    }
    // Save grouped runs to cache file
    fs.writeFileSync(cachePath, JSON.stringify(Array.from(grouped.entries())));
    // Set output
    core.setOutput('total-runs', mergedRuns.length);
    core.setOutput('workflow-count', grouped.size);
    // Log remaining GitHub API rate limit
    const rateLimit = await octokit.rest.rateLimit.get();
    const remaining = rateLimit.data.resources.core.remaining;
    const limit = rateLimit.data.resources.core.limit;
    core.info(`GitHub API rate limit remaining: ${remaining} / ${limit}`);
  } catch (error) {
    core.setFailed(error.message);
  }
}

// Run the action if this file is executed directly
if (require.main === module) {
  run();
}
