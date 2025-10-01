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
 * Fetch all workflow runs for the repository, paginated, stopping at sinceDate if provided.
 * @param {object} github - Octokit client
 * @param {object} context - GitHub Actions context
 * @param {number} days - Number of days to look back
 * @param {Date} sinceDate - Only fetch runs after this date
 * @returns {Promise<Array>} Array of workflow run objects
 */
async function fetchAllWorkflowRuns(github, context, days, sinceDate, eventType='') {
  const allRuns = [];
  const cutoffDate = getCutoffDate(days);
  const createdDateFilter = `>=${cutoffDate.toISOString()}`;

  core.info(`createdDateFilter: ${createdDateFilter}`);
  core.info(`days ${days}, sinceDate: ${sinceDate}`);
  for (let page = 1; page <= MAX_PAGES; page++) { // download pages of runs from the GitHub API
    const params = {
      owner: context.repo.owner,
      repo: context.repo.repo,
      per_page: RUNS_PER_PAGE,
      page,
    }
    if (eventType) {
      params.event = eventType;
    }
    const { data: runs } = await github.rest.actions.listWorkflowRunsForRepo(params); // listWorkflowRunsForRepo is a GitHub API call to list the workflow runs for the repository
    if (!runs.workflow_runs.length) {
      break;
    }

    for (const run of runs.workflow_runs) {
      const runDate = new Date(run.created_at);
      if (sinceDate && runDate <= sinceDate) {
        core.info(`Early exit: found run at ${runDate} <= latest cached date ${sinceDate}`);
        return allRuns;
      } // if a run is found that's too old, we can early exit cuz all future runs will be even older
      if (runDate >= cutoffDate) {
        allRuns.push(run);
      }
    }
    // If the api call returned no runs, assume we've reached the end and break
    if (!runs.workflow_runs.length) break;
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
    const rawCachePath = core.getInput('cache-path', { required: false });
    const defaultOutputPath = path.join(process.env.GITHUB_WORKSPACE || process.cwd(), 'workflow-data.json');
    const outputPath = rawCachePath && rawCachePath.trim() ? rawCachePath : defaultOutputPath;
    const testRunIdInput = core.getInput('test_run_id') || process.env.TEST_RUN_ID || '';
    // Create authenticated Octokit client
    const octokit = github.getOctokit(core.getInput('GITHUB_TOKEN', { required: true }));
    // Load previous cache if it exists
    let previousRuns = [];
    let latestCachedDate = null;

    core.info(`Restored previousRuns count: ${previousRuns.length}`);
    core.info(`Latest cached run date: ${latestCachedDate}`);
    // Test mode: download workflow-data.json artifact from a specific run and exit early
    // TODO: Remove this once we're done testing
    if (testRunIdInput) {
      const runId = parseInt(testRunIdInput, 10);
      if (Number.isNaN(runId)) {
        throw new Error(`Invalid test_run_id: ${testRunIdInput}`);
      }
      core.info(`[TEST MODE] Using workflow-data.json from run_id=${runId}`);
      const owner = github.context.repo.owner;
      const repo = github.context.repo.repo;
      const { data: artifactsResp } = await octokit.rest.actions.listWorkflowRunArtifacts({ owner, repo, run_id: runId, per_page: 100 });
      const artifacts = artifactsResp.artifacts || [];
      if (artifacts.length === 0) {
        throw new Error(`No artifacts found for run_id=${runId}`);
      }
      // Find an artifact zip that contains workflow-data.json
      let found = false;
      const tmpDir = fs.mkdtempSync(path.join(require('os').tmpdir(), 'wfzip-'));
      for (const art of artifacts) {
        try {
          const resp = await octokit.rest.actions.downloadArtifact({ owner, repo, artifact_id: art.id, archive_format: 'zip' });
          const zipPath = path.join(tmpDir, `${art.name}.zip`);
          fs.writeFileSync(zipPath, Buffer.from(resp.data));
          // Extract artifact to a temp dir and search for workflow JSON file
          const extractDir = path.join(tmpDir, art.name);
          if (!fs.existsSync(extractDir)) fs.mkdirSync(extractDir, { recursive: true });
          // Avoid buffering large stdout/stderr to prevent ENOBUFS
          execFileSync('unzip', ['-o', zipPath, '-d', extractDir], { stdio: 'ignore' });
          // Search recursively for workflow JSON
          const targetNames = new Set(['workflow-data.json', 'workflow.json']);
          const stack = [extractDir];
          let foundPath;
          while (stack.length && !foundPath) {
            const dir = stack.pop();
            const entries = fs.readdirSync(dir, { withFileTypes: true });
            for (const ent of entries) {
              const p = path.join(dir, ent.name);
              if (ent.isDirectory()) {
                stack.push(p);
              } else if (ent.isFile() && targetNames.has(ent.name)) {
                foundPath = p;
                break;
              }
            }
          }
          if (foundPath) {
            const jsonBuf = fs.readFileSync(foundPath);
            // Ensure output directory exists
            const outputDir = path.dirname(outputPath);
            if (!fs.existsSync(outputDir)) {
              fs.mkdirSync(outputDir, { recursive: true });
            }
            fs.writeFileSync(outputPath, jsonBuf);
            core.info(`[TEST MODE] Wrote ${outputPath} from ${path.basename(foundPath)} in artifact ${art.name}`);
            // Compute outputs from content
            let grouped;
            try {
              grouped = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
            } catch (e) {
              throw new Error(`Parsed JSON invalid at ${outputPath}: ${e.message}`);
            }
            const workflowCount = Array.isArray(grouped) ? grouped.length : 0;
            let totalRuns = 0;
            if (Array.isArray(grouped)) {
              for (const entry of grouped) {
                if (Array.isArray(entry) && Array.isArray(entry[1])) {
                  totalRuns += entry[1].length;
                }
              }
            }
            core.setOutput('total-runs', totalRuns);
            core.setOutput('workflow-count', workflowCount);
            core.setOutput('cache-path', outputPath);
            found = true;
            break;
          }
        } catch (e) {
          core.warning(`[TEST MODE] Failed processing artifact ${art.name}: ${e.message}`);
        }
      }
      if (!found) {
        throw new Error(`[TEST MODE] Could not find workflow-data.json in any artifacts for run_id=${runId}`);
      }
      // Exit early
      return;
    }
    // TODO: Remove this once we're done testing
    // Fetch new runs from GitHub (for the last N days, only after latest cached run)

    // 1. Fetch runs for each event type separately

    const delay = ms => new Promise(resolve => setTimeout(resolve, ms));


    core.info('Fetching all runs...');
    const allRuns = await fetchAllWorkflowRuns(octokit, github.context, days, latestCachedDate);
    core.info(`Fetched allRuns count: ${allRuns.length}`);

    // Wait for 1 second to avoid rate limiting
    await delay(1000);

    core.info('Fetching scheduled runs...');
    const scheduledRuns = await fetchAllWorkflowRuns(octokit, github.context, days, latestCachedDate, 'schedule');
    core.info(`Fetched scheduledRuns count: ${scheduledRuns.length}`);

    // 2. Combine all the results into a single array
    const newRuns = [...scheduledRuns, ...allRuns];

    core.info(`Fetched a total of ${newRuns.length} new runs across all event types.`);

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
    // Ensure output directory exists
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    // Save grouped runs to artifact file
    fs.writeFileSync(outputPath, JSON.stringify(Array.from(grouped.entries())));
    // Set output
    core.setOutput('total-runs', mergedRuns.length);
    core.setOutput('workflow-count', grouped.size);
    core.setOutput('cache-path', outputPath);
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
