// Cache Management Module
// Handles cache management, pruning, and artifact utilities

const core = require('@actions/core');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

/**
 * Find the most recent successful run of aggregate-workflow-data workflow on main branch.
 * @param {object} octokit - Octokit client
 * @param {object} context - GitHub Actions context
 * @returns {Promise<number|null>} Run ID or null if not found
 */
async function findPreviousAggregateRun(octokit, context, branch = 'main') {
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

module.exports = {
  findPreviousAggregateRun,
  pruneOldRuns,
  pruneAnnotationsIndex,
  pruneLogsIndex,
  pruneCommits,
  downloadAndExtractArtifact,
  findFileInDirectory,
  removeUnkeptDirectories,
};
