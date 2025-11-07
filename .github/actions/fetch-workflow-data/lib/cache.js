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

/**
 * Restores artifacts from a previous aggregate run
 * @param {object} octokit - Octokit client
 * @param {object} context - GitHub Actions context
 * @param {number} previousRunId - Run ID of the previous aggregate run
 * @param {string} workspace - Workspace directory path
 * @param {Date} cutoffDate - Cutoff date for pruning
 * @param {number} days - Number of days to look back
 * @returns {Promise<Object>} Object containing restored cache data and paths
 */
async function restoreArtifactsFromPreviousRun(octokit, context, previousRunId, workspace, cutoffDate, days) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  let cachedGrouped = new Map();
  let cachedAnnotationsIndex = {};
  let cachedGtestLogsIndex = {};
  let cachedOtherLogsIndex = {};
  let cachedCommits = [];
  let cachedLastSuccessTimestamps = {};

  let annotationsIndexPath = path.join(workspace, 'annotations', 'annotations-index.json');
  let gtestLogsIndexPath = path.join(workspace, 'logs', 'gtest', 'gtest-logs-index.json');
  let otherLogsIndexPath = path.join(workspace, 'logs', 'other', 'other-logs-index.json');
  let commitsPath = path.join(workspace, 'commits-main.json');
  let lastSuccessPath = path.join(workspace, 'last-success-timestamps.json');

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

          cachedGrouped = pruneOldRuns(cachedGrouped, cutoffDate);

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

          cachedAnnotationsIndex = pruneAnnotationsIndex(cachedAnnotationsIndex, cachedGrouped, cutoffDate);
          const afterPruneCount = Object.keys(cachedAnnotationsIndex).length;
          core.info(`[CACHE] Pruned ${beforePruneCount - afterPruneCount} annotation entries`);

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

          cachedGtestLogsIndex = pruneLogsIndex(cachedGtestLogsIndex, cachedGrouped, cutoffDate);
          const afterPruneCount = Object.keys(cachedGtestLogsIndex).length;
          core.info(`[CACHE] Pruned ${beforePruneCount - afterPruneCount} gtest log entries`);

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

          cachedOtherLogsIndex = pruneLogsIndex(cachedOtherLogsIndex, cachedGrouped, cutoffDate);
          const afterPruneCount = Object.keys(cachedOtherLogsIndex).length;
          core.info(`[CACHE] Pruned ${beforePruneCount - afterPruneCount} other log entries`);

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

  return {
    cachedGrouped,
    cachedAnnotationsIndex,
    cachedGtestLogsIndex,
    cachedOtherLogsIndex,
    cachedCommits,
    cachedLastSuccessTimestamps,
    annotationsIndexPath,
    gtestLogsIndexPath,
    otherLogsIndexPath,
    commitsPath,
    lastSuccessPath,
  };
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
  restoreArtifactsFromPreviousRun,
};
