// Analyze Workflow Data GitHub Action
// This action analyzes cached workflow run data and generates a summary report of workflow statuses.
// It provides two tables: one for push-triggered workflows and another for scheduled workflows.
// For scheduled workflows, it also tracks the last known good commit and earliest bad commit.
//
// See: https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28

const core = require('@actions/core');
const github = require('@actions/github');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execFileSync } = require('child_process');

// Constants
const DEFAULT_LOOKBACK_DAYS = 15;
const SHA_SHORT_LENGTH = 7;
const SUCCESS_RATE_DECIMAL_PLACES = 2;
const SUCCESS_EMOJI = '✅';
const FAILURE_EMOJI = '❌';
const EMPTY_VALUE = '—';

// Owners mapping cache
let __ownersMapping = undefined;
function loadOwnersMapping() {
  if (__ownersMapping !== undefined) return __ownersMapping;
  try {
    const ownersPath = path.join(__dirname, 'owners.json');
    if (fs.existsSync(ownersPath)) {
      const raw = fs.readFileSync(ownersPath, 'utf8');
      __ownersMapping = JSON.parse(raw);
    } else {
      __ownersMapping = null;
    }
  } catch (_) {
    __ownersMapping = null;
  }
  return __ownersMapping;
}

function normalizeOwners(value) {
  if (!value) return undefined;
  // Single string -> id-only
  if (typeof value === 'string') return [{ id: value }];
  // Object with id/name
  if (typeof value === 'object' && !Array.isArray(value)) {
    const id = value.id || undefined;
    const name = value.name || undefined;
    if (!id && !name) return undefined;
    return [{ id, name }];
  }
  // Array -> map each entry
  if (Array.isArray(value)) {
    const arr = [];
    for (const entry of value) {
      if (typeof entry === 'string') arr.push({ id: entry });
      else if (entry && typeof entry === 'object') arr.push({ id: entry.id, name: entry.name });
    }
    return arr.length ? arr : undefined;
  }
  return undefined;
}

function findOwnerForLabel(label) {
  try {
    const mapping = loadOwnersMapping();
    if (!mapping) return undefined;
    const lbl = typeof label === 'string' ? label : '';
    // Prefer exact keys (label may contain the key as a substring)
    if (mapping.exact && typeof mapping.exact === 'object') {
      for (const key of Object.keys(mapping.exact)) {
        if (lbl.includes(key)) return normalizeOwners(mapping.exact[key]);
      }
    }
    // Fallback to contains list
    if (Array.isArray(mapping.contains)) {
      for (const entry of mapping.contains) {
        if (entry && typeof entry.needle === 'string' && lbl.includes(entry.needle)) {
          return normalizeOwners(entry.owner);
        }
      }
    }
  } catch (_) {
    // ignore
  }
  return undefined;
}

// Simple HTML escaping for rendering snippets safely in summary HTML
function escapeHtml(text) {
  if (typeof text !== 'string') return text;
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function renderErrorsTable(errorSnippets) {
  if (!Array.isArray(errorSnippets) || errorSnippets.length === 0) {
    return '<em>No error info found</em>';
  }
  const rows = errorSnippets.map(obj => {
    const label = escapeHtml(obj.label || '');
    const snippet = escapeHtml(obj.snippet || '');
    // Render owner display name(s) if present; fallback to id(s); else 'no owner found'
    let ownerDisplay = 'no owner found';
    if (obj.owner && Array.isArray(obj.owner) && obj.owner.length) {
      const names = obj.owner.map(o => (o && (o.name || o.id)) || '').filter(Boolean);
      if (names.length) ownerDisplay = names.join(', ');
    }
    const owner = escapeHtml(ownerDisplay);
    return `<tr><td style="vertical-align:top;"><pre style="white-space:pre-wrap;word-break:break-word;margin:0;">${label}</pre></td><td>${owner}</td><td><pre style="white-space:pre-wrap;margin:0;">${snippet}</pre></td></tr>`;
  }).join('\n');
  return `<table><thead><tr><th style="text-align:left;">Test</th><th style="text-align:left;">Owner</th><th style="text-align:left;">Error</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderRepeatedErrorsTable(repeatedErrors) {
  if (!Array.isArray(repeatedErrors) || repeatedErrors.length === 0) {
    return '<em>None</em>';
  }
  const rows = repeatedErrors.map(e => {
    const snippet = escapeHtml(e.snippet || '');
    const count = typeof e.count === 'number' ? String(e.count) : '';
    return `<tr><td>${count}</td><td><pre style="white-space:pre-wrap;margin:0;">${snippet}</pre></td></tr>`;
  }).join('\n');
  return `<table><thead><tr><th>Count</th><th>Snippet</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderCommitsTable(commits) {
  if (!Array.isArray(commits) || commits.length === 0) {
    return '<em>None</em>';
  }
  const rows = commits.map(c => {
    const short = escapeHtml(c.short || (c.sha ? c.sha.substring(0, 7) : ''));
    const url = c.url || (c.sha ? `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${c.sha}` : undefined);
    const who = c.author_login ? `@${escapeHtml(c.author_login)}` : escapeHtml(c.author_name || 'unknown');
    const whoHtml = c.author_login && c.author_url ? `<a href="${c.author_url}">${who}</a>` : who;
    const shaHtml = url ? `<a href="${url}"><code>${short}</code></a>` : `<code>${short}</code>`;
    return `<tr><td>${shaHtml}</td><td>${whoHtml}</td></tr>`;
  }).join('\n');
  return `<table><thead><tr><th>SHA</th><th>Author</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/**
 * Fetches PR information associated with a commit.
 *
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @param {string} commitSha - Full SHA of the commit to look up
 * @returns {Promise<object>} Object containing:
 *   - prNumber: Markdown link to the PR (e.g., [#123](url))
 *   - prTitle: Title of the PR or EMPTY_VALUE if not found
 *   - prAuthor: GitHub username of the PR author or 'unknown'
 */
async function fetchPRInfo(github, context, commitSha) {
  try {
    const { data: prs } = await github.rest.repos.listPullRequestsAssociatedWithCommit({
      owner: context.repo.owner,
      repo: context.repo.repo,
      commit_sha: commitSha,
    });
    if (prs.length > 0) {
      const pr = prs[0];
      return {
        prNumber: `[#${pr.number}](https://github.com/${context.repo.owner}/${context.repo.repo}/pull/${pr.number})`,
        prTitle: pr.title || EMPTY_VALUE,
        prAuthor: pr.user?.login || 'unknown'
      };
    }
  } catch (e) {
    core.warning(`Could not fetch PR for commit ${commitSha}: ${e.message}`);
  }
  return { prNumber: EMPTY_VALUE, prTitle: EMPTY_VALUE, prAuthor: EMPTY_VALUE };
}

/**
 * Fetch commit author info for a commit SHA.
 * Returns GitHub login (if associated), author display name, and profile URL (if available).
 */
async function fetchCommitAuthor(octokit, context, commitSha) {
  try {
    const { data } = await octokit.rest.repos.getCommit({
      owner: context.repo.owner,
      repo: context.repo.repo,
      ref: commitSha,
    });
    const login = data.author?.login;
    const htmlUrl = data.author?.html_url;
    const name = data.commit?.author?.name;
    return { login, name, htmlUrl };
  } catch (e) {
    core.warning(`Could not fetch commit author for ${commitSha}: ${e.message}`);
    return { login: undefined, name: undefined, htmlUrl: undefined };
  }
}

/**
 * Calculates statistics for a set of workflow runs.
 *
 * @param {Array<object>} runs - Array of workflow run objects
 * @returns {object} Statistics object containing:
 *   - eventTypes: Comma-separated string of unique event types
 *   - successRate: Percentage of runs that succeeded (based on last attempt)
 *   - uniqueSuccessRate: Percentage of runs that succeeded on first attempt
 *   - retryRate: Percentage of successful runs that required retries
 *   - uniqueRuns: Number of unique runs (excluding attempts)
 *   - totalRuns: Total number of runs including attempts
 *   - successfulRuns: Number of runs that succeeded (based on last attempt)
 */
function getWorkflowStats(runs) {
  // Group runs by their original run ID to handle retries and reruns
  const uniqueRuns = new Map();
  let totalRunsIncludingRetries = 0;
  let totalSuccessfulUniqueRuns = 0;
  let successfulUniqueRunsOnFirstTry = 0;
  let successfulUniqueRunsWithRetries = 0;

  // First pass: identify all unique runs and their attempts
  for (const run of runs) {
    totalRunsIncludingRetries++;

    // Calculate the original run ID by subtracting (run_attempt - 1) from the current run ID
    const originalRunId = run.run_attempt > 1 ? run.id - (run.run_attempt - 1) : run.id;

    if (!uniqueRuns.has(originalRunId)) {
      uniqueRuns.set(originalRunId, {
        run,
        attempts: 0,
        isSuccessful: false,
        requiredRetry: false,
        succeededOnFirstTry: false,
        lastAttempt: run
      });
    } else {
      // This is an attempt
      const existingRun = uniqueRuns.get(originalRunId);
      existingRun.attempts++;

      // Update last attempt if this is a newer attempt
      if (run.run_attempt > existingRun.lastAttempt.run_attempt) {
        existingRun.lastAttempt = run;
      }
    }
  }

  // Second pass: determine final status based on last attempt
  for (const runInfo of uniqueRuns.values()) {
    const lastAttempt = runInfo.lastAttempt;
    if (lastAttempt.conclusion === 'success') {
      runInfo.isSuccessful = true;
      totalSuccessfulUniqueRuns++;

      if (lastAttempt.run_attempt === 1) {
        runInfo.succeededOnFirstTry = true;
        successfulUniqueRunsOnFirstTry++;
      } else {
        runInfo.requiredRetry = true;
        successfulUniqueRunsWithRetries++;
      }
    }
  }

  const uniqueRunsArray = Array.from(uniqueRuns.values()).map(r => r.run);
  const eventTypes = [...new Set(uniqueRunsArray.map(r => r.event))].join(', ');

  // Calculate rates
  const successRate = uniqueRunsArray.length === 0 ? "N/A" : (totalSuccessfulUniqueRuns / uniqueRunsArray.length * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";
  const uniqueSuccessRate = uniqueRunsArray.length === 0 ? "N/A" : (successfulUniqueRunsOnFirstTry / uniqueRunsArray.length * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";
  const retryRate = totalSuccessfulUniqueRuns === 0 ? "N/A" : (successfulUniqueRunsWithRetries / totalSuccessfulUniqueRuns * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";

  return {
    eventTypes,
    successRate,
    uniqueSuccessRate,
    retryRate,
    uniqueRuns: uniqueRunsArray.length,
    totalRuns: totalRunsIncludingRetries,
    successfulRuns: totalSuccessfulUniqueRuns
  };
}

/**
 * Generates a GitHub Actions workflow URL.
 *
 * @param {object} context - GitHub Actions context
 * @param {string} workflowFile - Path to the workflow file relative to .github/workflows/
 * @returns {string} Full URL to the workflow in GitHub Actions
 */
function getWorkflowLink(context, workflowFile) {
  return workflowFile
    ? `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/workflows/${workflowFile}?query=branch%3Amain`
    : `https://github.com/${context.repo.owner}/${context.repo.repo}/actions`;
}

/**
 * Finds, within the provided window (newest→oldest, main branch only), either:
 * - the first failing run since the most recent success (oldest in the current failing streak), or
 * - if no success exists in-window, the oldest failing run in the window.
 * Returns null if there are no failing runs in the window.
 *
 * @param {Array<object>} mainBranchRunsWindow - Runs on main, sorted by created_at desc (newest first)
 * @returns {{run: object, noSuccessInWindow: boolean}|null}
 */
function findFirstFailInWindow(mainBranchRunsWindow) {
  let seenAnyFailure = false;
  let oldestFailure = null;
  let firstFailInStreak = null; // oldest failure observed before crossing a success boundary

  for (const run of mainBranchRunsWindow) {
    if (run.conclusion === 'success') {
      if (firstFailInStreak) {
        return { run: firstFailInStreak, noSuccessInWindow: false };
      }
      // Success encountered before any failure in the current scan; keep scanning older entries
    } else if (run.conclusion && run.conclusion !== 'cancelled' && run.conclusion !== 'skipped') {
      // Treat anything non-success, non-cancelled/skipped as failure for this purpose
      seenAnyFailure = true;
      firstFailInStreak = run; // update to become oldest failure within the current failing streak
      oldestFailure = run; // this will end up as the oldest failure in the entire window
    }
  }

  if (seenAnyFailure) {
    // No success found in-window; report oldest failure in the window
    return { run: oldestFailure, noSuccessInWindow: true };
  }
  return null;
}

/**
 * Finds the first failing run on main since the last success (i.e., the start of the current failing streak).
 * Scans runs in reverse chronological order and returns the oldest failure before the first encountered success.
 * Falls back to the oldest failure in history if no success is found.
 *
 * @param {object} octokit - Authenticated Octokit client
 * @param {object} context - GitHub Actions context
 * @param {string} workflowPath - Path to the workflow file (e.g., .github/workflows/ci.yaml)
 * @returns {Promise<object|null>} The workflow run object or null if none found
 */
async function findFirstFailOnMainSinceLastSuccess(octokit, context, workflowPath) {
  if (!workflowPath) return null;
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const workflowId = path.basename(workflowPath); // API accepts file name as workflow_id

  let firstFail = null;
  let foundSuccessBoundary = false;

  await octokit.paginate(
    octokit.rest.actions.listWorkflowRuns,
    { owner, repo, workflow_id: workflowId, branch: 'main', status: 'completed', per_page: 100 },
    (res, done) => {
      const runs = res.data.workflow_runs || [];
      for (const run of runs) {
        // Newest -> oldest
        if (run.conclusion === 'success') {
          // We hit the boundary; earliest failure of the current streak is in firstFail
          if (firstFail) {
            foundSuccessBoundary = true;
            done();
            return [];
          }
          // No failures seen yet; continue scanning older pages in case the failure streak started after this success
        } else if (run.conclusion && run.conclusion !== 'cancelled' && run.conclusion !== 'skipped') {
          // Record as we go; due to newest->oldest ordering, the last assigned before a success will be the oldest failure in the streak
          firstFail = run;
        }
      }
      // continue pagination
      return [];
    }
  );

  return firstFail;
}

/**
 * Analyzes scheduled runs to find the last good and earliest bad commits.
 *
 * @param {Array<object>} scheduledMainRuns - Array of scheduled runs on main branch, sorted by date (newest first)
 * @param {object} context - GitHub Actions context
 * @returns {object} Object containing:
 *   - lastGoodSha: Short SHA of the last successful run (e.g., `a1b2c3d`)
 *   - earliestBadSha: Short SHA of the earliest failing run (e.g., `e4f5g6h`)
 */
function findGoodBadCommits(scheduledMainRuns, context) {
  let lastGoodSha = EMPTY_VALUE;
  let earliestBadSha = EMPTY_VALUE;
  let foundGood = false;
  let foundBad = false;

  for (const run of scheduledMainRuns) {
    if (!foundGood && run.conclusion === 'success') {
      const shortSha = run.head_sha.substring(0, SHA_SHORT_LENGTH);
      lastGoodSha = `[\`${shortSha}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${run.head_sha})`;
      foundGood = true;
    }
    if (!foundBad && run.conclusion !== 'success') {
      const shortSha = run.head_sha.substring(0, SHA_SHORT_LENGTH);
      earliestBadSha = `[\`${shortSha}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${run.head_sha})`;
      foundBad = true;
    }
    if (foundGood && foundBad) break;
  }

  return { lastGoodSha, earliestBadSha };
}

/**
 * Gets information about the last run on main branch.
 *
 * @param {Array<object>} mainBranchRuns - Array of runs on main branch, sorted by date (newest first)
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @returns {Promise<object>} Object containing run information
 */
async function getLastRunInfo(mainBranchRuns, github, context) {
  const lastMainRun = mainBranchRuns[0];
  if (!lastMainRun) {
    return {
      status: EMPTY_VALUE,
      sha: EMPTY_VALUE,
      run: EMPTY_VALUE,
      pr: EMPTY_VALUE,
      title: EMPTY_VALUE,
      lastGoodSha: EMPTY_VALUE,
      earliestBadSha: EMPTY_VALUE
    };
  }

  const prInfo = await fetchPRInfo(github, context, lastMainRun.head_sha);
  // Current approach: filter by event type
  const mainRuns = mainBranchRuns.filter(r => r.event === lastMainRun.event || r.event === 'workflow_dispatch');
  // Alternative approach: include all runs on main branch
  // const mainRuns = mainBranchRuns;
  const { lastGoodSha, earliestBadSha } = findGoodBadCommits(mainRuns, context);

  return {
    status: lastMainRun.conclusion === 'success' ? SUCCESS_EMOJI : FAILURE_EMOJI,
    sha: `[\`${lastMainRun.head_sha.substring(0, SHA_SHORT_LENGTH)}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${lastMainRun.head_sha})`,
    run: `[Run](${lastMainRun.html_url})${lastMainRun.run_attempt > 1 ? ` (#${lastMainRun.run_attempt})` : ''}`,
    pr: prInfo.prNumber,
    title: prInfo.prTitle,
    lastGoodSha,
    earliestBadSha: lastMainRun.conclusion !== 'success' ? earliestBadSha : EMPTY_VALUE
  };
}

/**
 * Generates summary tables for push and scheduled workflows.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @returns {Promise<string>} Markdown table for all workflows
 */
async function generateSummaryBox(grouped, github, context) {
  const rows = [];

  for (const [name, runs] of grouped.entries()) {
    const mainBranchRuns = runs
      .filter(r => r.head_branch === 'main')
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    const stats = getWorkflowStats(runs);

    // Skip workflows that only have workflow_dispatch as their event type
    if (stats.eventTypes === 'workflow_dispatch') {
      continue;
    }

    const workflowLink = getWorkflowLink(context, runs[0]?.path);
    const runInfo = await getLastRunInfo(mainBranchRuns, github, context);

    const row = `| [${name}](${workflowLink}) | ${stats.eventTypes || 'unknown'} | ${stats.totalRuns} | ${stats.successfulRuns} | ${stats.successRate} | ${stats.uniqueSuccessRate} | ${stats.retryRate} | ${runInfo.status} | ${runInfo.sha} | ${runInfo.run} | ${runInfo.pr} | ${runInfo.title} | ${runInfo.earliestBadSha} | ${runInfo.lastGoodSha} |`;
    rows.push(row);
  }

  return [
    '## Workflow Summary',
    '| Workflow | Event Type(s) | Total Runs | Successful Runs | Success Rate | Unique Success Rate | Retry Rate | Last Run on `main` | Last SHA | Last Run | Last PR | PR Title | Earliest Bad SHA | Last Good SHA |',
    '|----------|---------------|------------|-----------------|--------------|-------------------|------------|-------------------|----------|----------|---------|-----------|------------------|---------------|',
    ...rows,
    ''  // Empty line for better readability
  ].join('\n');
}

/**
 * Builds the complete markdown report.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @returns {Promise<string>} Complete markdown report
 */
async function buildReport(grouped, github, context) {
  const days = parseInt(core.getInput('days') || DEFAULT_LOOKBACK_DAYS, 10);
  const timestamp = new Date().toISOString();
  return [
    `# Workflow Summary (Last ${days} Days) - Generated at ${timestamp}\n`,
    await generateSummaryBox(grouped, github, context),
    '\n## Column Descriptions\n',
    'A unique run represents a single workflow execution, which may have multiple retry attempts. For example, if a workflow fails and is retried twice, this counts as one unique run with three attempts (initial run + two retries).\n',
    '\n### Success Rate Calculations\n',
    'The success rates are calculated based on unique runs (not including retries in the denominator):\n',
    '- **Success Rate**: (Number of unique runs that eventually succeeded / Total number of unique runs) × 100%\n',
    '  - Example: 3 successful unique runs out of 5 total unique runs = 60% success rate\n',
    '- **Unique Success Rate**: (Number of unique runs that succeeded on first try / Total number of unique runs) × 100%\n',
    '  - Example: 1 unique run succeeded on first try out of 5 total unique runs = 20% unique success rate\n',
    '- **Retry Rate**: (Number of successful unique runs that needed retries / Total number of successful unique runs) × 100%\n',
    '  - Example: 2 successful unique runs needed retries out of 3 total successful unique runs = 66.67% retry rate\n',
    '\nNote: Unique Success Rate + Retry Rate does not equal 100% because they measure different things:\n',
    '- Unique Success Rate is based on all unique runs\n',
    '- Retry Rate is based only on successful unique runs\n',
    '\n| Column | Description |',
    '|--------|-------------|',
    '| Workflow | Name of the workflow with link to its GitHub Actions page |',
    '| Event Type(s) | Types of events that trigger this workflow (e.g., push, pull_request, schedule) |',
    '| Total Runs | Total number of workflow runs including all retry attempts (e.g., 1 unique run with 2 retries = 3 total runs) |',
    '| Successful Runs | Number of unique workflow runs that eventually succeeded, regardless of whether they needed retries |',
    '| Success Rate | Percentage of unique workflow runs that eventually succeeded (e.g., 3/5 unique runs succeeded = 60%) |',
    '| Unique Success Rate | Percentage of unique workflow runs that succeeded on their first attempt without needing retries (e.g., 1/5 unique runs succeeded on first try = 20%) |',
    '| Retry Rate | Percentage of successful unique runs that needed retries to succeed (e.g., of 3 successful unique runs, 2 needed retries = 66.67%) |',
    '| Last Run on `main` | Status of the most recent run on the main branch (✅ for success, ❌ for failure) |',
    '| Last SHA | Short SHA of the most recent run on main |',
    '| Last Run | Link to the most recent run on main, with attempt number if applicable |',
    '| Last PR | Link to the PR associated with the most recent run, if any |',
    '| PR Title | Title of the PR associated with the most recent run, if any |',
    '| Earliest Bad SHA | Short SHA of the earliest failing run on main (only shown if last run failed) |',
    '| Last Good SHA | Short SHA of the last successful run on main |'
  ].join('\n');
}

/**
 * Filters workflow runs by date range
 * @param {Array<Object>} runs - Array of workflow runs
 * @param {number} days - Number of days to look back
 * @returns {Array<Object>} Filtered runs within the date range
 */
function filterRunsByDate(runs, days) {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - days);

  return runs.filter(run => {
    const runDate = new Date(run.created_at);
    return runDate >= cutoffDate;
  });
}

/**
 * Collect commits between two SHAs on the default branch (main), inclusive of endSha.
 * Returns an array of { sha, short, url, author_login, author_name, author_url }.
 * Note: Uses compareCommits, which is base..head; base is typically the success commit, head is the failed run commit.
 */
async function listCommitsBetween(octokit, context, startShaExclusive, endShaInclusive) {
  try {
    const { data } = await octokit.rest.repos.compareCommits({
      owner: context.repo.owner,
      repo: context.repo.repo,
      base: startShaExclusive,
      head: endShaInclusive,
    });
    // compareCommits includes both endpoints; to make start exclusive, filter it out explicitly
    const commits = data.commits || [];
    return commits
      .filter(c => c.sha !== startShaExclusive)
      .concat(data.merge_base_commit && data.merge_base_commit.sha === endShaInclusive ? [] : [])
      .map(c => ({
        sha: c.sha,
        short: c.sha.substring(0, SHA_SHORT_LENGTH),
        url: `https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${c.sha}`,
        author_login: c.author?.login,
        author_name: c.commit?.author?.name,
        author_url: c.author?.html_url,
      }));
  } catch (e) {
    core.warning(`Failed to list commits between ${startShaExclusive}..${endShaInclusive}: ${e.message}`);
    return [];
  }
}

/**
 * Main function to run the action
 */
async function run() {
  try {
    // Get inputs
    const cachePath = core.getInput('cache-path', { required: true });
    const previousCachePath = core.getInput('previous-cache-path', { required: false });
    const workflowConfigs = JSON.parse(core.getInput('workflow_configs', { required: true }));
    const days = parseInt(core.getInput('days') || DEFAULT_LOOKBACK_DAYS, 10);
    const alertAll = String(core.getInput('alert-all') || 'false').toLowerCase() === 'true';

    // Validate inputs
    if (!fs.existsSync(cachePath)) {
      throw new Error(`Cache file not found at ${cachePath}`);
    }
    if (!Array.isArray(workflowConfigs)) {
      throw new Error('Workflow configs must be a JSON array');
    }
    if (isNaN(days) || days <= 0) {
      throw new Error('Days must be a positive number');
    }

    // Load cached data
    const grouped = JSON.parse(fs.readFileSync(cachePath, 'utf8'));
    const hasPrevious = previousCachePath && fs.existsSync(previousCachePath);
    const previousGrouped = hasPrevious ? JSON.parse(fs.readFileSync(previousCachePath, 'utf8')) : null;

    // Track failed workflows
    const failedWorkflows = [];

    // Filter and process each workflow configuration
    const filteredGrouped = new Map();
    const filteredPreviousGrouped = new Map();
    for (const config of workflowConfigs) {
      core.info(`Processing config: ${JSON.stringify(config)}`);
      for (const [name, runs] of grouped) {
        if ((config.wkflw_name && name === config.wkflw_name) ||
            (config.wkflw_prefix && name.startsWith(config.wkflw_prefix))) {
          core.info(`Matched workflow: ${name} with config: ${JSON.stringify(config)}`);
          // Filter runs by date range
          const filteredRuns = filterRunsByDate(runs, days);
          if (filteredRuns.length > 0) {
            filteredGrouped.set(name, filteredRuns);

            // Check if latest run on main is failing
            const mainBranchRuns = filteredRuns
              .filter(r => r.head_branch === 'main')
              .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
            if (mainBranchRuns[0]?.conclusion !== 'success') {
              failedWorkflows.push(name);
            }
          }
        }
      }

      if (hasPrevious && Array.isArray(previousGrouped)) {
        for (const [name, runs] of previousGrouped) {
          if ((config.wkflw_name && name === config.wkflw_name) ||
              (config.wkflw_prefix && name.startsWith(config.wkflw_prefix))) {
            const filteredRuns = filterRunsByDate(runs, days);
            if (filteredRuns.length > 0) {
              filteredPreviousGrouped.set(name, filteredRuns);
            }
          }
        }
      }
    }

    // Create authenticated Octokit client for PR info
    const octokit = github.getOctokit(core.getInput('GITHUB_TOKEN', { required: true }));

    // Generate primary report
    const mainReport = await buildReport(filteredGrouped, octokit, github.context);

    // Optional: Build Slack-ready alert message for all failing workflows with owner mentions
    let alertAllMessage = '';
    if (alertAll && failedWorkflows.length > 0) {
      const mention = (owners) => {
        const arr = Array.isArray(owners) ? owners : (owners ? [owners] : []);
        const ids = arr.map(o => (o && o.id) ? `<@${o.id}>` : '').filter(Boolean);
        return ids.length ? ids.join(' ') : '';
      };

      const failingItems = [];
      for (const [name, runs] of filteredGrouped.entries()) {
        const mainRuns = runs
          .filter(r => r.head_branch === 'main')
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        if (!mainRuns[0] || mainRuns[0].conclusion === 'success') continue;
        // Try to attach owners from the first failing run's label via snippets; fallback to job name
        // Use the latest failing run for snippet-based owner detection
        const latestFail = mainRuns.find(r => r.conclusion !== 'success');
        let owners = undefined;
        try {
          const errs = await fetchErrorSnippetsForRun(octokit, github.context, latestFail.id, 10);
          // Aggregate owners from snippets
          const ownerSet = new Map();
          for (const e of (errs || [])) {
            if (Array.isArray(e.owner)) {
              for (const o of e.owner) {
                if (!o) continue;
                const key = `${o.id || ''}|${o.name || ''}`;
                ownerSet.set(key, o);
              }
            }
          }
          owners = Array.from(ownerSet.values());
        } catch (_) { /* ignore */ }
        // Fallback: try to resolve owners from the workflow name
        if (!owners || owners.length === 0) {
          owners = findOwnerForLabel(name);
        }
        const ownerMentions = mention(owners) || '(no owner found)';
        const wfUrl = getWorkflowLink(github.context, runs[0]?.path);
        failingItems.push(`• ${name} ${wfUrl ? `<${wfUrl}|open>` : ''} ${ownerMentions}`.trim());
      }
      if (failingItems.length) {
        alertAllMessage = [
          '*Alerts: failing workflows on main*',
          ...failingItems
        ].join('\n');
      }
    }

    // Compute status changes vs previous and write JSON
    const computeLatestConclusion = (runs) => {
      const mainBranchRuns = runs
        .filter(r => r.head_branch === 'main')
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      const latest = mainBranchRuns[0];
      if (!latest) return null;
      return latest.conclusion === 'success' ? 'success' : 'failure';
    };
    const computeLatestRunInfo = (runs) => {
      const mainBranchRuns = runs
        .filter(r => r.head_branch === 'main')
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      const latest = mainBranchRuns[0];
      if (!latest) return null;
      return { id: latest.id, url: latest.html_url, created_at: latest.created_at, head_sha: latest.head_sha, path: latest.path };
    };

    const allNames = new Set([
      ...Array.from(filteredGrouped.keys()),
      ...Array.from(filteredPreviousGrouped.keys())
    ]);

    const changes = [];
    const regressedDetails = [];
    const stayedFailingDetails = [];
    for (const name of allNames) {
      const currentRuns = filteredGrouped.get(name);
      const previousRuns = filteredPreviousGrouped.get(name);
      if (!currentRuns || !previousRuns) continue; // require data on both sides
      const current = computeLatestConclusion(currentRuns);
      const previous = computeLatestConclusion(previousRuns);
      if (!current || !previous) continue;

      let change;
      if (previous === 'success' && current === 'success') change = 'stayed_succeeding';
      else if (previous !== 'success' && current !== 'success') change = 'stayed_failing';
      else if (previous !== 'success' && current === 'success') change = 'fail_to_success';
      else if (previous === 'success' && current !== 'success') change = 'success_to_fail';

      if (change) {
        const info = computeLatestRunInfo(currentRuns);
        const workflowUrl = info?.path ? getWorkflowLink(github.context, info.path) : undefined;
        const aggregateRunUrl = `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/actions/runs/${github.context.runId}`;
        const commitUrl = info?.head_sha ? `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${info.head_sha}` : undefined;
        const commitShort = info?.head_sha ? info.head_sha.substring(0, 7) : undefined;
        changes.push({ name, previous, current, change, run_id: info?.id, run_url: info?.url, created_at: info?.created_at, workflow_url: workflowUrl, workflow_path: info?.path, aggregate_run_url: aggregateRunUrl, commit_sha: info?.head_sha, commit_short: commitShort, commit_url: commitUrl });
        if (change === 'success_to_fail' && info) {
          regressedDetails.push({ name, run_id: info.id, run_url: info.url, created_at: info.created_at, workflow_url: workflowUrl, workflow_path: info.path, aggregate_run_url: aggregateRunUrl, commit_sha: info.head_sha, commit_short: commitShort, commit_url: commitUrl });
        }
        else if (change === 'stayed_failing' && info) {
          stayedFailingDetails.push({ name, run_id: info.id, run_url: info.url, created_at: info.created_at, workflow_url: workflowUrl, workflow_path: info.path, aggregate_run_url: aggregateRunUrl, commit_sha: info.head_sha, commit_short: commitShort, commit_url: commitUrl });
        }
      }
    }

    // Helper to get main runs within the current window from a grouped collection
    const getMainWindowRuns = (runs) => runs
      .filter(r => r.head_branch === 'main')
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    // Enrich regressions with first failing run within the window
    for (const item of regressedDetails) {
      try {
        const windowRuns = getMainWindowRuns(filteredGrouped.get(item.name) || []);
        const res = findFirstFailInWindow(windowRuns);
        if (res && res.run) {
          item.first_failed_run_id = res.run.id;
          item.first_failed_run_url = res.run.html_url;
          item.first_failed_created_at = res.run.created_at;
          item.first_failed_head_sha = res.run.head_sha;
          item.first_failed_head_short = res.run.head_sha ? res.run.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;
          item.no_success_in_window = !!res.noSuccessInWindow;
          // Commit author enrichment
          const author = await fetchCommitAuthor(octokit, github.context, item.first_failed_head_sha);
          item.first_failed_author_login = author.login;
          item.first_failed_author_name = author.name;
          item.first_failed_author_url = author.htmlUrl;
          // Mirror into the corresponding change entry
          const changeRef = changes.find(c => c.name === item.name && c.change === 'success_to_fail');
          if (changeRef) {
            changeRef.first_failed_run_id = item.first_failed_run_id;
            changeRef.first_failed_run_url = item.first_failed_run_url;
            changeRef.first_failed_created_at = item.first_failed_created_at;
            changeRef.first_failed_head_sha = item.first_failed_head_sha;
            changeRef.first_failed_head_short = item.first_failed_head_short;
            changeRef.no_success_in_window = item.no_success_in_window;
            changeRef.first_failed_author_login = item.first_failed_author_login;
            changeRef.first_failed_author_name = item.first_failed_author_name;
            changeRef.first_failed_author_url = item.first_failed_author_url;
          }
        }
      } catch (e) {
        core.warning(`Failed to find first failing run for ${item.name}: ${e.message}`);
      }
    }

    // Enrich stayed failing with first failing run within the window
    for (const item of stayedFailingDetails) {
      try {
        const windowRuns = getMainWindowRuns(filteredGrouped.get(item.name) || []);
        const res = findFirstFailInWindow(windowRuns);
        if (res && res.run) {
          item.first_failed_run_id = res.run.id;
          item.first_failed_run_url = res.run.html_url;
          item.first_failed_created_at = res.run.created_at;
          item.first_failed_head_sha = res.run.head_sha;
          item.first_failed_head_short = res.run.head_sha ? res.run.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;
          item.no_success_in_window = !!res.noSuccessInWindow;
          const author = await fetchCommitAuthor(octokit, github.context, item.first_failed_head_sha);
          item.first_failed_author_login = author.login;
          item.first_failed_author_name = author.name;
          item.first_failed_author_url = author.htmlUrl;
        }
        // Mirror into the corresponding change entry
        const changeRef = changes.find(c => c.name === item.name && c.change === 'stayed_failing');
        if (changeRef) {
          changeRef.first_failed_run_id = item.first_failed_run_id;
          changeRef.first_failed_run_url = item.first_failed_run_url;
          changeRef.first_failed_created_at = item.first_failed_created_at;
          changeRef.first_failed_head_sha = item.first_failed_head_sha;
          changeRef.first_failed_head_short = item.first_failed_head_short;
          changeRef.no_success_in_window = item.no_success_in_window;
          changeRef.first_failed_author_login = item.first_failed_author_login;
          changeRef.first_failed_author_name = item.first_failed_author_name;
          changeRef.first_failed_author_url = item.first_failed_author_url;
        }
      }
      catch (e) {
        core.warning(`Failed to find first failing run for ${item.name}: ${e.message}`);
      }
    }

    const outputDir = process.env.GITHUB_WORKSPACE || process.cwd();
    const statusChangesPath = path.join(outputDir, 'workflow-status-changes.json');
    fs.writeFileSync(statusChangesPath, JSON.stringify(changes));
    core.setOutput('status_changes_path', statusChangesPath);

    // Build a minimal regressions section (success -> fail only)
    let regressionsSection = '';
    let stayedFailingSection = '';
    try {
      const parsed = Array.isArray(changes) ? changes : [];
      const regressionsItems = parsed.filter(item => item.change === 'success_to_fail');
      const stayedFailingItems = parsed.filter(item => item.change === 'stayed_failing');
      if (regressionsItems.length > 0) {
        const lines = regressionsItems.map(it => {
          const base = it.workflow_url ? `- [${it.name}](${it.workflow_url})` : `- ${it.name}`;
          if (it.first_failed_run_url) {
            const sha = it.first_failed_head_short || (it.first_failed_head_sha ? it.first_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
            const shaLink = sha ? `[\`${sha}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.first_failed_head_sha})` : '';
            const when = it.first_failed_created_at ? new Date(it.first_failed_created_at).toISOString() : '';
            const author = it.first_failed_author_login
              ? `by [@${it.first_failed_author_login}](${it.first_failed_author_url})`
              : (it.first_failed_author_name ? `by ${it.first_failed_author_name}` : '');
            if (it.no_success_in_window) {
              return `${base}\n  - Failed to find any successful run in the last two weeks. Oldest failing run is: [Run](${it.first_failed_run_url}) ${when} ${shaLink} ${author}`;
            }
            return `${base}\n  - First failing run on main: [Run](${it.first_failed_run_url}) ${when} ${shaLink} ${author}`;
          }
          return base;
        });
        regressionsSection = ['', '## Regressions (Pass → Fail)', ...lines, ''].join('\n');
      } else {
        regressionsSection = ['','## Regressions (Pass → Fail)','- None',''].join('\n');
      }
      if (stayedFailingItems.length > 0) {
        const lines = stayedFailingItems.map(it => {
          const base = it.workflow_url ? `- [${it.name}](${it.workflow_url})` : `- ${it.name}`;
          if (it.first_failed_run_url) {
            const sha = it.first_failed_head_short || (it.first_failed_head_sha ? it.first_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
            const shaLink = sha ? `[\`${sha}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.first_failed_head_sha})` : '';
            const when = it.first_failed_created_at ? new Date(it.first_failed_created_at).toISOString() : '';
            const author = it.first_failed_author_login
              ? `by [@${it.first_failed_author_login}](${it.first_failed_author_url})`
              : (it.first_failed_author_name ? `by ${it.first_failed_author_name}` : '');
            if (it.no_success_in_window) {
              return `${base}\n  - Failed to find any successful run in the last two weeks. Oldest failing run is: [Run](${it.first_failed_run_url}) ${when} ${shaLink} ${author}`;
            }
            return `${base}\n  - First failing run on main: [Run](${it.first_failed_run_url}) ${when} ${shaLink} ${author}`;
          }
          return base;
        });
        stayedFailingSection = ['', '## Still Failing (No Recovery)', ...lines, ''].join('\n');
      } else {
        stayedFailingSection = ['','## Still Failing (No Recovery)','- None',''].join('\n');
      }
    } catch (_) {
      // Fallback: always show headers even if nothing parsed
      regressionsSection = ['','## Regressions (Pass → Fail)','- None',''].join('\n');
      stayedFailingSection = ['','## Still Failing (No Recovery)','- None',''].join('\n');
    }

    // Do not include alerts section inside the report; Slack message will carry it
    const finalReport = [mainReport, regressionsSection, stayedFailingSection]
      .filter(Boolean)
      .join('\n');

    // Set outputs
    core.setOutput('failed_workflows', JSON.stringify(failedWorkflows));
    core.setOutput('report', finalReport);
    if (alertAll) core.setOutput('alert_all_message', alertAllMessage || '');
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
