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
 * Get the oldest created_at date from cached runs.
 * @param {Array} runs - Array of workflow run objects
 * @returns {Date} The oldest created_at date, or current date if none
 */
function getOldestCachedDate(runs) {
  if (!runs.length) return new Date(); // Safe default: current date
  // Map to timestamps, filter out invalid dates
  const times = runs
    .map(run => new Date(run.created_at).getTime())
    .filter(t => !isNaN(t) && t > 0);
  if (!times.length) return new Date();
  return new Date(Math.min(...times));
}

/**
 * Fetch all workflow runs for the repository, paginated, stopping at sinceDate if provided.
 * @param {object} github - Octokit client
 * @param {object} context - GitHub Actions context
 * @param {number} days - Number of days to look back
 * @param {Date} sinceDate - Only fetch runs after this date
 * @param {Date} oldestCachedDate - Oldest date in our cache
 * @returns {Promise<Array>} Array of workflow run objects
 */
async function fetchAllWorkflowRuns(github, context, days, sinceDate, oldestCachedDate) {
  const allRuns = [];
  const cutoffDate = getCutoffDate(days);
  core.info(`days ${days}, sinceDate: ${sinceDate}, cutoffDate: ${cutoffDate}, oldestCachedDate: ${oldestCachedDate}`);

  // If our cutoff date is newer than oldest cached date, we only need to fetch new data
  const needHistoricalData = cutoffDate < oldestCachedDate;
  core.info(`Need historical data: ${needHistoricalData} (cutoffDate ${cutoffDate} < oldestCachedDate ${oldestCachedDate})`);

  // If we don't need historical data and we have a valid sinceDate, we can optimize our fetch
  if (!needHistoricalData && sinceDate && !isNaN(sinceDate.getTime())) {
    core.info(`Optimizing fetch: only getting runs after ${sinceDate.toISOString()}`);
    try {
      for (let page = 1; page <= MAX_PAGES; page++) {
        const { data: runs } = await github.rest.actions.listWorkflowRunsForRepo({
          owner: context.repo.owner,
          repo: context.repo.repo,
          per_page: RUNS_PER_PAGE,
          page,
          created: `>=${sinceDate.toISOString()}`
        });

        if (!runs.workflow_runs.length) {
          core.info('No more runs found, stopping fetch');
          break;
        }

        // Add all runs since they're all newer than our sinceDate
        allRuns.push(...runs.workflow_runs);
        core.info(`Fetched ${runs.workflow_runs.length} runs on page ${page}`);

        // If we got fewer runs than requested, we've reached the end
        if (runs.workflow_runs.length < RUNS_PER_PAGE) {
          core.info('Received fewer runs than requested, reached end of data');
          break;
        }
      }
      return allRuns;
    } catch (error) {
      core.warning(`Error during optimized fetch: ${error.message}. Falling back to full fetch.`);
      // Fall back to full fetch if optimized fetch fails
    }
  }

  // If we need historical data, don't have a sinceDate, or optimized fetch failed, fetch everything
  core.info('Performing full fetch of workflow runs');
  for (let page = 1; page <= MAX_PAGES; page++) {
    try {
      const { data: runs } = await github.rest.actions.listWorkflowRunsForRepo({
        owner: context.repo.owner,
        repo: context.repo.repo,
        per_page: RUNS_PER_PAGE,
        page
      });

      if (!runs.workflow_runs.length) {
        core.info('No more runs found, stopping fetch');
        break;
      }

      for (const run of runs.workflow_runs) {
        const runDate = new Date(run.created_at);

        // If we don't need historical data and we hit a run older than our oldest cached date, we can stop
        if (!needHistoricalData && runDate <= oldestCachedDate) {
          core.info(`Early exit: found run at ${runDate} <= oldest cached date ${oldestCachedDate}`);
          return allRuns;
        }

        // Only add runs that are within our cutoff date window
        if (runDate >= cutoffDate) {
          allRuns.push(run);
        } else {
          // If we hit a run older than our cutoff date, we can stop
          core.info(`Early exit: found run at ${runDate} <= cutoff date ${cutoffDate}`);
          return allRuns;
        }
      }

      // If we got fewer runs than requested, we've reached the end
      if (runs.workflow_runs.length < RUNS_PER_PAGE) {
        core.info('Received fewer runs than requested, reached end of data');
        break;
      }
    } catch (error) {
      core.warning(`Error fetching page ${page}: ${error.message}`);
      break;
    }
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
  // Group all runs by name
  for (const run of runs) {
    if (!run.name) continue; // Skip runs without names
    if (!grouped.has(run.name)) {
      grouped.set(run.name, []);
    }
    grouped.get(run.name).push(run);
  }
  return grouped;
}

/**
 * Filter runs based on workflow configs.
 * @param {Array} runs - Array of workflow run objects
 * @param {Array} workflowConfigs - Array of workflow config objects
 * @returns {Array} Filtered array of workflow run objects
 */
function filterRunsByConfig(runs, workflowConfigs) {
  // Create sets of workflow names and prefixes from configs
  const configWorkflows = new Set();
  const configPrefixes = new Set();
  workflowConfigs.forEach(config => {
    if (config.wkflw_name) {
      configWorkflows.add(config.wkflw_name);
    }
    if (config.wkflw_prefix) {
      configPrefixes.add(config.wkflw_prefix);
    }
  });

  // Filter runs based on config
  return runs.filter(run => {
    if (!run.name) return false;

    // Check for exact name match
    if (configWorkflows.has(run.name)) {
      return true;
    }

    // Check for prefix match
    for (const prefix of configPrefixes) {
      if (run.name.startsWith(prefix)) {
        return true;
      }
    }

    return false;
  });
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
    const workflowConfigs = JSON.parse(core.getInput('workflow_configs', { required: true }));

    // Validate inputs
    if (!Array.isArray(workflowConfigs)) {
      throw new Error('Workflow configs must be a JSON array');
    }

    // Create authenticated Octokit client
    const octokit = github.getOctokit(core.getInput('GITHUB_TOKEN', { required: true }));

    // Load previous cache if it exists
    let previousRuns = [];
    let latestCachedDate = null;
    let oldestCachedDate = null;
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
        oldestCachedDate = getOldestCachedDate(previousRuns);
      } catch (e) {
        core.warning('Could not parse previous cache, ignoring.');
      }
    }
    core.info(`Restored previousRuns count: ${previousRuns.length}`);
    core.info(`Latest cached run date: ${latestCachedDate}`);
    core.info(`Oldest cached run date: ${oldestCachedDate}`);

    // Fetch new runs from GitHub
    const newRuns = await fetchAllWorkflowRuns(octokit, github.context, days, latestCachedDate, oldestCachedDate);
    core.info(`Fetched newRuns count: ${newRuns.length}`);

    // Merge and deduplicate by run id
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

    // Filter runs based on workflow configs
    mergedRuns = filterRunsByConfig(mergedRuns, workflowConfigs);
    core.info(`Filtered runs count: ${mergedRuns.length}`);

    // Group runs by workflow name
    const grouped = groupRunsByName(mergedRuns);

    // Ensure cache directory exists
    const cacheDir = require('path').dirname(cachePath);
    if (!fs.existsSync(cacheDir)) {
      fs.mkdirSync(cacheDir, { recursive: true });
    }

    // Save grouped runs to cache file
    const sortedEntries = Array.from(grouped.entries());
    fs.writeFileSync(cachePath, JSON.stringify(sortedEntries, null, 2));

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
