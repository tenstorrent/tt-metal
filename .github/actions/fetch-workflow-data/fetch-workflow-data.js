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
const MAIN_BRANCH = 'main'; // Only collect data from main branch

/**
 * Core business logic, independent of GitHub Actions
 */
class WorkflowDataFetcher {
  constructor(options = {}) {
    this.maxPages = options.maxPages || MAX_PAGES;
    this.runsPerPage = options.runsPerPage || RUNS_PER_PAGE;
    this.defaultDays = options.defaultDays || DEFAULT_DAYS;
  }

  /**
   * Date utility functions
   */
  getCutoffDate(days) {
    // Validate days parameter
    if (typeof days !== 'number' || isNaN(days)) {
      throw new Error('Days parameter must be a number');
    }
    if (days <= 0) {
      throw new Error('Days parameter must be positive');
    }

    const d = new Date();
    // Convert days to milliseconds (including fractional days for hours)
    const hoursInMs = days * 24 * 60 * 60 * 1000;
    d.setTime(d.getTime() - hoursInMs);

    // Log the time range for debugging
    const hours = Math.floor(days * 24);
    const minutes = Math.floor((days * 24 * 60) % 60);
    core.info(`[Fetch] Time range: ${hours} hours and ${minutes} minutes (${days} days)`);
    core.info(`[Fetch] Cutoff time: ${d.toISOString()}`);

    return d;
  }

  getMostRecentDateInRuns(runs) {
    if (!runs.length) return new Date(0); // Safe default: epoch
    const times = runs
      .map(run => new Date(run.created_at).getTime())
      .filter(t => !isNaN(t) && t > 0);
    if (!times.length) return new Date(0);
    return new Date(Math.max(...times));
  }

  getEarliestDateInRuns(runs) {
    if (!runs.length) return new Date(); // Safe default: current date
    const times = runs
      .map(run => new Date(run.created_at).getTime())
      .filter(t => !isNaN(t) && t > 0);
    if (!times.length) return new Date();
    return new Date(Math.min(...times));
  }

  /**
   * Debug function to fetch earliest and latest data from runs
   * @param {Array} runs - Array of workflow runs
   * @returns {Object} Object containing earliest and latest run data
   */
  getEarliestAndLatestData(runs) {
    if (!runs.length) {
      return {
        earliest: null,
        latest: null,
        message: 'No runs available'
      };
    }

    // Sort runs by creation date
    const sortedRuns = [...runs].sort((a, b) => {
      const dateA = new Date(a.created_at).getTime();
      const dateB = new Date(b.created_at).getTime();
      return dateA - dateB;
    });

    const earliest = sortedRuns[0];
    const latest = sortedRuns[sortedRuns.length - 1];

    return {
      earliest: {
        id: earliest.id,
        name: earliest.name,
        created_at: earliest.created_at,
        conclusion: earliest.conclusion,
        status: earliest.status,
        head_branch: earliest.head_branch,
        event: earliest.event
      },
      latest: {
        id: latest.id,
        name: latest.name,
        created_at: latest.created_at,
        conclusion: latest.conclusion,
        status: latest.status,
        head_branch: latest.head_branch,
        event: latest.event
      },
      total_runs: runs.length,
      date_range: {
        start: earliest.created_at,
        end: latest.created_at,
        days: ((new Date(latest.created_at) - new Date(earliest.created_at)) / (1000 * 60 * 60 * 24)).toFixed(1)
      }
    };
  }

  /**
   * Core data processing functions
   */
  groupRunsByName(runs) {
    const grouped = new Map();
    for (const run of runs) {
      if (!run.name) continue; // Skip runs without names
      if (!grouped.has(run.name)) {
        grouped.set(run.name, []);
      }
      grouped.get(run.name).push(run);
    }
    return grouped;
  }

  filterRuns(runs, workflowConfigs) {
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

    return runs.filter(run => {
      // Skip runs without names
      if (!run.name) return false;

      // Skip skipped runs
      if (run.conclusion === 'skipped' || run.status === 'skipped') return false;

      // Skip non-scheduled/push runs
      if (run.event !== 'schedule' && run.event !== 'push') return false;

      // Skip non-main branch runs
      if (run.head_branch !== MAIN_BRANCH) return false;

      // Skip incomplete runs
      if (run.status !== 'completed') return false;

      // Check for exact name match
      if (configWorkflows.has(run.name)) return true;

      // Check for prefix match
      for (const prefix of configPrefixes) {
        if (run.name.startsWith(prefix)) return true;
      }

      return false;
    });
  }

  mergeRuns(previousRuns, newRuns) {
    const allRuns = [...previousRuns];
    const seenIds = new Set(previousRuns.map(run => run.id));
    for (const run of newRuns) {
      if (!seenIds.has(run.id)) {
        allRuns.push(run);
        seenIds.add(run.id);
      }
    }
    return allRuns;
  }
}

/**
 * GitHub Actions specific implementation
 */
class GitHubWorkflowFetcher extends WorkflowDataFetcher {
  constructor(githubClient, context, options = {}) {
    super(options);
    this.github = githubClient;
    this.context = context;
  }

  /**
   * Fetches PR information for a commit
   * @param {string} commitSha - The commit SHA to look up
   * @returns {Promise<object>} PR information or null if not found
   */
  async fetchPRInfo(commitSha) {
    try {
      const { data: prs } = await this.github.rest.repos.listPullRequestsAssociatedWithCommit({
        owner: this.context.repo.owner,
        repo: this.context.repo.repo,
        commit_sha: commitSha,
      });

      if (prs.length > 0) {
        const pr = prs[0];
        return {
          pr_number: `[#${pr.number}](https://github.com/${this.context.repo.owner}/${this.context.repo.repo}/pull/${pr.number})`,
          pr_title: pr.title || null,
          pr_author: pr.user?.login || null
        };
      }
    } catch (error) {
      core.warning(`Could not fetch PR for commit ${commitSha}: ${error.message}`);
    }
    return null;
  }

  /**
   * Enriches a run with PR information
   * @param {object} run - The workflow run to enrich
   * @returns {Promise<object>} The enriched run
   */
  async enrichRunWithPRInfo(run) {
    const prInfo = await this.fetchPRInfo(run.head_sha);
    const baseUrl = `https://github.com/${this.context.repo.owner}/${this.context.repo.repo}`;
    const workflowUrl = `${baseUrl}/actions/workflows/${encodeURIComponent(run.path)}`;
    const runUrl = `${baseUrl}/actions/runs/${run.id}`;

    if (prInfo) {
      return {
        ...run,
        pr_number: prInfo.pr_number,
        pr_title: prInfo.pr_title,
        pr_author: prInfo.pr_author,
        repository: {
          owner: this.context.repo.owner,
          name: this.context.repo.repo,
          full_name: `${this.context.repo.owner}/${this.context.repo.repo}`
        },
        workflow_url: workflowUrl,
        run_url: runUrl
      };
    }
    return {
      ...run,
      repository: {
        owner: this.context.repo.owner,
        name: this.context.repo.repo,
        full_name: `${this.context.repo.owner}/${this.context.repo.repo}`
      },
      workflow_url: workflowUrl,
      run_url: runUrl
    };
  }

  async fetchAllWorkflowRuns(days, sinceDate, oldestCachedDate) {
    const allRuns = [];
    const cutoffDate = this.getCutoffDate(days);
    const startTime = Date.now();
    let apiCallCount = 0;
    let dateRange = { earliest: null, latest: null };

    core.info(`[Fetch] Starting data collection for last ${days} days (cutoff: ${cutoffDate.toISOString()})`);

    // Validate inputs
    if (!oldestCachedDate || isNaN(oldestCachedDate.getTime())) {
      core.warning('[Fetch] Invalid oldestCachedDate, will perform full fetch');
      oldestCachedDate = new Date(0);
    }

    // If our cutoff date is newer than oldest cached date, we only need to fetch new data
    const needHistoricalData = cutoffDate < oldestCachedDate;
    core.info(`[Fetch] Cache status: ${needHistoricalData ? 'Need historical data' : 'Using cached data'} (oldest cache: ${oldestCachedDate.toISOString()})`);

    // If we don't need historical data and we have a valid sinceDate, we can optimize our fetch
    if (!needHistoricalData && sinceDate && !isNaN(sinceDate.getTime())) {
      core.info(`[Fetch] Optimized fetch: collecting runs after ${sinceDate.toISOString()}`);
      try {
        for (let page = 1; page <= this.maxPages; page++) {
          apiCallCount++;
          const { data: runs } = await this.github.rest.actions.listWorkflowRunsForRepo({
            owner: this.context.repo.owner,
            repo: this.context.repo.repo,
            per_page: this.runsPerPage,
            page,
            created: `>=${sinceDate.toISOString()}`,
            branch: MAIN_BRANCH
          });

          if (!runs.workflow_runs.length) {
            core.info('[Fetch] No more runs found in optimized fetch');
            break;
          }

          allRuns.push(...runs.workflow_runs);
          core.info(`[Fetch] Page ${page}: Added ${runs.workflow_runs.length} runs (total: ${allRuns.length})`);

          if (runs.workflow_runs.length < this.runsPerPage) {
            core.info('[Fetch] Reached end of data in optimized fetch');
            break;
          }

          // Check for rate limits
          const rateLimit = await this.getRateLimitInfo();
          if (rateLimit.remaining < 100) {
            core.warning(`[Fetch] Low rate limit remaining: ${rateLimit.remaining}, stopping optimized fetch`);
            break;
          }
        }
      } catch (error) {
        core.warning(`[Fetch] Error during optimized fetch: ${error.message}. Falling back to full fetch.`);
      }
    }

    // Full fetch - only needed if we don't have enough historical data
    if (needHistoricalData) {
      core.info('[Fetch] Starting full fetch to collect historical data');
      let hasCompleteData = false;

      for (let page = 1; page <= this.maxPages; page++) {
        try {
          apiCallCount++;
          const { data: runs } = await this.github.rest.actions.listWorkflowRunsForRepo({
            owner: this.context.repo.owner,
            repo: this.context.repo.repo,
            per_page: this.runsPerPage,
            page,
            branch: MAIN_BRANCH
          });

          if (!runs.workflow_runs.length) {
            core.info('[Fetch] No more runs found in full fetch');
            break;
          }

          for (const run of runs.workflow_runs) {
            const runDate = new Date(run.created_at);
            if (isNaN(runDate.getTime())) {
              core.warning(`[Fetch] Invalid run date for run ${run.id}, skipping`);
              continue;
            }

            // Update date range
            if (!dateRange.earliest || runDate < dateRange.earliest) {
              dateRange.earliest = runDate;
              if (dateRange.earliest) {
                core.debug(`[Fetch] New earliest run found: ${dateRange.earliest.toISOString()}`);
              }
            }
            if (!dateRange.latest || runDate > dateRange.latest) {
              dateRange.latest = runDate;
            }

            // If we've found a run older than our cutoff date, we have complete data
            if (runDate < cutoffDate) {
              hasCompleteData = true;
              core.info(`[Fetch] Found run older than cutoff date: ${runDate.toISOString()} < ${cutoffDate.toISOString()}`);
            }

            // Always add to allRuns for caching, but we'll filter later for the requested period
            allRuns.push(run);
          }

          // If we have complete data coverage, we can stop
          if (hasCompleteData) {
            core.info('[Fetch] Complete data coverage achieved, stopping full fetch');
            break;
          }

          if (runs.workflow_runs.length < this.runsPerPage) {
            core.info('[Fetch] Reached end of data in full fetch');
            break;
          }

          // Check for rate limits
          const rateLimit = await this.getRateLimitInfo();
          if (rateLimit.remaining < 100) {
            core.warning(`[Fetch] Low rate limit remaining: ${rateLimit.remaining}, stopping full fetch`);
            break;
          }
        } catch (error) {
          if (error.status === 403 && error.message.includes('rate limit')) {
            core.warning('[Fetch] Rate limit exceeded, stopping full fetch');
            break;
          }
          core.warning(`[Fetch] Error fetching page ${page}: ${error.message}`);
          break;
        }
      }

      // If we don't have complete data coverage, log a warning
      if (!hasCompleteData && dateRange.earliest && dateRange.latest) {
        core.warning(`[Fetch] Incomplete data coverage: Earliest run (${dateRange.earliest.toISOString()}) is newer than cutoff date (${cutoffDate.toISOString()})`);
      }

      // Log the date range of fetched runs
      if (dateRange.earliest && dateRange.latest) {
        core.info(`[Fetch] Fetched runs date range: ${dateRange.earliest.toISOString()} to ${dateRange.latest.toISOString()}`);
      }
    }

    // Enrich runs with PR information
    core.info('[Fetch] Enriching runs with PR information...');
    const enrichedRuns = [];
    for (const run of allRuns) {
      const enrichedRun = await this.enrichRunWithPRInfo(run);
      enrichedRuns.push(enrichedRun);
    }
    core.info(`[Fetch] Enriched ${enrichedRuns.length} runs with PR information`);

    const duration = Date.now() - startTime;
    core.info(`[Fetch] Completed: Collected ${enrichedRuns.length} runs in ${duration}ms (${apiCallCount} API calls)`);

    // Return the complete dataset for filtering in the main function
    return {
      completeRuns: enrichedRuns,
      dateRange: dateRange
    };
  }

  async getRateLimitInfo() {
    const rateLimit = await this.github.rest.rateLimit.get();
    return {
      remaining: rateLimit.data.resources.core.remaining,
      limit: rateLimit.data.resources.core.limit
    };
  }

  /**
   * Combines cached and newly fetched data, then filters for the requested time period
   */
  async combineAndFilterData(previousRuns, newRuns, days, workflowConfigs) {
    // Combine all runs
    const allRuns = this.mergeRuns(previousRuns, newRuns);
    core.info(`[Fetch] Combined ${previousRuns.length} cached runs with ${newRuns.length} new runs into ${allRuns.length} total runs`);

    // Filter by workflow configs
    const filteredRuns = this.filterRuns(allRuns, workflowConfigs);
    core.info(`[Fetch] Filtered to ${filteredRuns.length} runs based on workflow configs, triggers, and branch`);

    // Get cutoff date for requested period
    const cutoffDate = this.getCutoffDate(days);

    // Filter for requested time period
    const requestedPeriodRuns = filteredRuns.filter(run => {
      const runDate = new Date(run.created_at);
      return runDate >= cutoffDate;
    });
    core.info(`[Fetch] Filtered to ${requestedPeriodRuns.length} runs within requested ${days} day period`);

    // Calculate date ranges for both datasets
    const completeDateRange = this.calculateDateRange(filteredRuns);
    const requestedDateRange = this.calculateDateRange(requestedPeriodRuns);

    // Log date ranges
    if (completeDateRange.earliest && completeDateRange.latest) {
      core.info(`[Fetch] Complete dataset date range: ${completeDateRange.earliest.toISOString()} to ${completeDateRange.latest.toISOString()}`);
    }
    if (requestedDateRange.earliest && requestedDateRange.latest) {
      const daysCovered = (requestedDateRange.latest - requestedDateRange.earliest) / (1000 * 60 * 60 * 24);
      core.info(`[Fetch] Requested period coverage: ${daysCovered.toFixed(1)} days (${requestedDateRange.earliest.toISOString()} to ${requestedDateRange.latest.toISOString()})`);

      if (daysCovered < days) {
        core.warning(`[Fetch] Warning: Requested period coverage (${daysCovered.toFixed(1)} days) is less than requested days (${days} days)`);
      }
    }

    return {
      completeRuns: filteredRuns,
      requestedPeriodRuns: requestedPeriodRuns,
      completeDateRange,
      requestedDateRange
    };
  }

  /**
   * Calculate date range for a set of runs
   */
  calculateDateRange(runs) {
    return runs.reduce((range, run) => {
      const runDate = new Date(run.created_at);
      if (isNaN(runDate.getTime())) {
        core.warning(`[Fetch] Invalid run date found, skipping`);
        return range;
      }
      return {
        earliest: !range.earliest || runDate < range.earliest ? runDate : range.earliest,
        latest: !range.latest || runDate > range.latest ? runDate : range.latest
      };
    }, { earliest: null, latest: null });
  }
}

/**
 * Cache management
 */
class CacheManager {
  constructor(fetcher) {
    this.fetcher = fetcher;
  }

  loadPreviousCache(cachePath) {
    let previousRuns = [];
    let mostRecentCachedDate = null;
    let earliestCachedDate = null;

    if (fs.existsSync(cachePath)) {
      try {
        const rawCache = fs.readFileSync(cachePath, 'utf8');
        const prev = JSON.parse(rawCache);

        // Handle different cache formats
        if (Array.isArray(prev)) {
          if (prev.length && Array.isArray(prev[0])) {
            // Format: [[workflowName, runs], ...]
            previousRuns = prev.flatMap(([_, runs]) => runs);
          } else if (prev.length && prev[0] && prev[0].id) {
            // Format: [run1, run2, ...]
            previousRuns = prev;
          }
        }

        // Calculate date boundaries
        mostRecentCachedDate = this.fetcher.getMostRecentDateInRuns(previousRuns);
        earliestCachedDate = this.fetcher.getEarliestDateInRuns(previousRuns);

        core.info(`Loaded ${previousRuns.length} runs from cache`);
        core.info(`Cache date range: ${earliestCachedDate.toISOString()} to ${mostRecentCachedDate.toISOString()}`);
      } catch (error) {
        core.warning(`Error loading cache: ${error.message}`);
      }
    }

    return { previousRuns, mostRecentCachedDate, earliestCachedDate };
  }

  saveCache(cachePath, groupedRuns) {
    const cacheData = Array.from(groupedRuns.entries());
    fs.writeFileSync(cachePath, JSON.stringify(cacheData, null, 2));
    core.info(`Saved ${groupedRuns.size} workflows to cache`);
  }
}

/**
 * Main entrypoint for the action.
 */
async function run() {
  try {
    // Get inputs
    const daysInput = core.getInput('days') || DEFAULT_DAYS;
    const days = parseFloat(daysInput);
    if (isNaN(days)) {
      throw new Error(`Invalid days value: ${daysInput}. Must be a number.`);
    }
    if (days <= 0) {
      throw new Error(`Days must be positive, got: ${days}`);
    }

    const cachePath = core.getInput('cache-path', { required: true });
    const workflowConfigs = JSON.parse(core.getInput('workflow_configs', { required: true }));

    // Validate inputs
    if (!Array.isArray(workflowConfigs)) {
      throw new Error('Workflow configs must be a JSON array');
    }

    // Initialize components
    const octokit = github.getOctokit(core.getInput('GITHUB_TOKEN', { required: true }));
    const fetcher = new GitHubWorkflowFetcher(octokit, github.context);
    const cacheManager = new CacheManager(fetcher);

    // Load previous cache
    const { previousRuns, mostRecentCachedDate, earliestCachedDate } = cacheManager.loadPreviousCache(cachePath);

    // Fetch new runs
    const { completeRuns, dateRange } = await fetcher.fetchAllWorkflowRuns(days, mostRecentCachedDate, earliestCachedDate);
    if (dateRange && dateRange.earliest && dateRange.latest) {
      core.info(`[Fetch] Fetched ${completeRuns.length} new runs with date range: ${dateRange.earliest.toISOString()} to ${dateRange.latest.toISOString()}`);
    } else {
      core.info(`[Fetch] Fetched ${completeRuns.length} new runs (no valid date range available)`);
    }

    // Merge new runs with cached runs
    const allRuns = fetcher.mergeRuns(previousRuns, completeRuns);
    core.info(`[Fetch] Combined ${previousRuns.length} cached runs with ${completeRuns.length} new runs into ${allRuns.length} total runs`);

    // Filter by workflow configs
    const filteredRuns = fetcher.filterRuns(allRuns, workflowConfigs);
    core.info(`[Fetch] Filtered to ${filteredRuns.length} runs based on workflow configs, triggers, and branch`);

    // Filter for requested time period
    const cutoffDate = fetcher.getCutoffDate(days);
    const requestedPeriodRuns = filteredRuns.filter(run => {
      const runDate = new Date(run.created_at);
      return !isNaN(runDate.getTime()) && runDate >= cutoffDate;
    });
    core.info(`[Fetch] Filtered to ${requestedPeriodRuns.length} runs within requested ${days} day period`);

    // Group runs by name for caching
    const groupedRuns = fetcher.groupRunsByName(filteredRuns);
    const groupedRequestedRuns = fetcher.groupRunsByName(requestedPeriodRuns);

    // Save complete dataset to cache
    cacheManager.saveCache(cachePath, groupedRuns);
    core.info(`[Fetch] Saved complete dataset to cache: ${groupedRuns.size} workflows, ${filteredRuns.length} total runs`);

    // Save requested period data to a separate file for upload
    const uploadPath = cachePath.replace('.json', '-upload.json');
    fs.writeFileSync(uploadPath, JSON.stringify(Array.from(groupedRequestedRuns.entries()), null, 2));
    core.info(`[Fetch] Saved filtered dataset to upload file: ${groupedRequestedRuns.size} workflows, ${requestedPeriodRuns.length} total runs`);

    // // Debug: Get earliest and latest data
    // const debugData = fetcher.getEarliestAndLatestData(requestedPeriodRuns);
    // core.info('[Debug] Data range information:');
    // core.info(JSON.stringify(debugData, null, 2));

    // // Debug: Get earliest and latest data for all-post-commit
    // const postCommitRuns = requestedPeriodRuns.filter(run => run.name === 'All post-commit');
    // const postCommitDebugData = fetcher.getEarliestAndLatestData(postCommitRuns);
    // core.info('[Debug] all-post-commit data range information:');
    // core.info(JSON.stringify(postCommitDebugData, null, 2));

    // Set outputs
    core.setOutput('total-runs', requestedPeriodRuns.length);
    core.setOutput('workflow-count', groupedRequestedRuns.size);
    core.setOutput('cache-path', uploadPath);

    // Log rate limit info
    const { remaining, limit } = await fetcher.getRateLimitInfo();
    core.info(`[Fetch] GitHub API rate limit remaining: ${remaining} / ${limit}`);
  } catch (error) {
    core.setFailed(error.message);
  }
}

// Run the action if this file is executed directly
if (require.main === module) {
  run();
}

// Export for testing
module.exports = {
  WorkflowDataFetcher,
  GitHubWorkflowFetcher,
  CacheManager
};
