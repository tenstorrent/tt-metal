// Logs Module
// Handles log processing and workflow matching

const core = require('@actions/core');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

/**
 * Process workflow logs: download annotations and logs for failing runs
 * @param {Map} grouped - Map of workflow names to their runs
 * @param {string} branch - Branch to filter runs by
 * @param {string} workspace - Workspace directory path
 * @param {object} cachedAnnotationsIndex - Cached annotations index
 * @param {object} cachedGtestLogsIndex - Cached gtest logs index
 * @param {object} cachedOtherLogsIndex - Cached other logs index
 * @param {Set} cachedRunAttempts - Set of cached (run ID, attempt) tuples
 * @param {object} octokit - Octokit client
 * @param {object} context - GitHub Actions context
 * @returns {Promise<{annotationsIndexPath: string, gtestLogsIndexPath: string, otherLogsIndexPath: string}>}
 */
async function processWorkflowLogs(grouped, branch, workspace, cachedAnnotationsIndex, cachedGtestLogsIndex, cachedOtherLogsIndex, cachedRunAttempts, octokit, context) {
  const owner = context.repo.owner;
  const repo = context.repo.repo;

  // Download logs for the latest failing run per workflow and build an index
  // Constraint: Only fetch logs when the latest run for that workflow (on target branch)
  // is failing (i.e., conclusion neither success nor skipped/cancelled).
  // The logs will be extracted under a dedicated directory so downstream steps
  // can parse them without performing network calls again.
  // Separate gtest logs from other logs (non-gtest failures with no annotations)
  const annotationsRoot = path.join(workspace, 'annotations');
  const gtestLogsRoot = path.join(workspace, 'logs', 'gtest');
  const otherLogsRoot = path.join(workspace, 'logs', 'other');
  if (!fs.existsSync(annotationsRoot)) {
    fs.mkdirSync(annotationsRoot, { recursive: true });
  }
  if (!fs.existsSync(gtestLogsRoot)) {
    fs.mkdirSync(gtestLogsRoot, { recursive: true });
  }
  if (!fs.existsSync(otherLogsRoot)) {
    fs.mkdirSync(otherLogsRoot, { recursive: true });
  }
  // Initialize indices from cache
  const annotationsIndex = { ...cachedAnnotationsIndex };
  const gtestLogsIndex = { ...cachedGtestLogsIndex };
  const otherLogsIndex = { ...cachedOtherLogsIndex };

  // Build a map of run ID -> workflow name for quick lookup
  const runIdToWorkflowName = new Map();
  for (const [name, runs] of grouped.entries()) {
    for (const run of runs) {
      runIdToWorkflowName.set(String(run.id), name);
    }
  }

  for (const [name, runs] of grouped.entries()) {
    try {
      // First deduplicate by run ID, keeping highest attempt (for logs/annotations, we only want latest attempt)
      const runsByRunId = new Map();
      for (const run of (runs || []).filter(r => r && r.head_branch === branch)) {
        const runId = run.id;
        const attempt = run.run_attempt || 1;
        const existingRun = runsByRunId.get(runId);
        const existingAttempt = existingRun ? (existingRun.run_attempt || 1) : 0;
        if (!existingRun || attempt > existingAttempt) {
          runsByRunId.set(runId, run);
        }
      }

      // Consider only deduplicated runs (latest attempt per run ID) and sort newest first
      const branchRuns = Array.from(runsByRunId.values())
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

      // Scan newest→older, skipping skipped/cancelled runs until either:
      // A) we find a success (stop; no logs), or
      // B) we find a failing run (download logs), or
      // C) we run out of runs (stop; no logs).
      let targetRun = undefined;
      for (const run of branchRuns) {
        const conc = run && run.conclusion;
        if (conc === 'skipped' || conc === 'cancelled') {
          continue; // ignore skipped/cancelled and check next older
        }
        if (conc === 'success') {
          targetRun = undefined; // found a success → do not fetch logs for this workflow
          break;
        }
        // Any other conclusion at this point is considered failing for log fetching purposes
        targetRun = run;
        break;
      }

      // Remove old annotations/logs for this workflow (only keep the latest failing run)
      const runsToKeep = targetRun ? new Set([String(targetRun.id)]) : new Set();
      let removedAnnotations = 0;
      let removedGtestLogs = 0;
      let removedOtherLogs = 0;

      // Remove old annotations for this workflow
      for (const [runId, dir] of Object.entries(annotationsIndex)) {
        if (runIdToWorkflowName.get(runId) === name && !runsToKeep.has(runId)) {
          delete annotationsIndex[runId];
          removedAnnotations++;
          // Remove directory from disk (dir is relative to workspace)
          const fullPath = path.join(workspace, dir);
          try {
            if (fs.existsSync(fullPath)) {
              fs.rmSync(fullPath, { recursive: true, force: true });
              core.info(`[LOGS] Removed old annotations directory for run ${runId} (workflow: ${name})`);
            }
          } catch (e) {
            core.warning(`[LOGS] Failed to remove old annotations directory ${fullPath}: ${e.message}`);
          }
        }
      }

      // Remove old gtest logs for this workflow
      for (const [runId, dir] of Object.entries(gtestLogsIndex)) {
        if (runIdToWorkflowName.get(runId) === name && !runsToKeep.has(runId)) {
          delete gtestLogsIndex[runId];
          removedGtestLogs++;
          // Remove directory from disk (dir is relative to workspace)
          const fullPath = path.join(workspace, dir);
          try {
            if (fs.existsSync(fullPath)) {
              fs.rmSync(fullPath, { recursive: true, force: true });
              core.info(`[LOGS] Removed old gtest logs directory for run ${runId} (workflow: ${name})`);
            }
          } catch (e) {
            core.warning(`[LOGS] Failed to remove old gtest logs directory ${fullPath}: ${e.message}`);
          }
        }
      }

      // Remove old other logs for this workflow
      for (const [runId, dir] of Object.entries(otherLogsIndex)) {
        if (runIdToWorkflowName.get(runId) === name && !runsToKeep.has(runId)) {
          delete otherLogsIndex[runId];
          removedOtherLogs++;
          // Remove directory from disk (dir is relative to workspace)
          const fullPath = path.join(workspace, dir);
          try {
            if (fs.existsSync(fullPath)) {
              fs.rmSync(fullPath, { recursive: true, force: true });
              core.info(`[LOGS] Removed old other logs directory for run ${runId} (workflow: ${name})`);
            }
          } catch (e) {
            core.warning(`[LOGS] Failed to remove old other logs directory ${fullPath}: ${e.message}`);
          }
        }
      }

      if (removedAnnotations > 0 || removedGtestLogs > 0 || removedOtherLogs > 0) {
        core.info(`[LOGS] Workflow '${name}': removed ${removedAnnotations} old annotation entries, ${removedGtestLogs} old gtest log entries, ${removedOtherLogs} old other log entries`);
      }

      // If workflow is passing (no targetRun), we're done (already cleaned up old logs)
      if (!targetRun) {
        if (removedAnnotations > 0 || removedGtestLogs > 0 || removedOtherLogs > 0) {
          core.info(`[LOGS] Workflow '${name}' is now passing, cleaned up old logs`);
        }
        continue;
      }

      // Skip if this exact (run ID, attempt) combination is already cached
      // Note: Different attempts of the same run ID need different logs/annotations
      const targetRunIdStr = String(targetRun.id);
      const targetRunAttempt = targetRun.run_attempt || 1;
      const targetRunKey = `${targetRunIdStr}:${targetRunAttempt}`;
      if (cachedRunAttempts.has(targetRunKey)) {
        core.info(`[LOGS] Skipping download for run ${targetRunIdStr} attempt ${targetRunAttempt} (workflow: ${name}) - already in cache`);
        continue;
      }

      core.info(`[LOGS] Processing failing run ${targetRun.id} for workflow '${name}' (conclusion: ${targetRun.conclusion})`);

      // Fetch check-run annotations for this failing run
      let sawGtestFailure = false;
      let sawAnyFailureAnnotations = false;
      let annotationsFetchFailed = false;
      const gtestJobNames = new Set();
      try {
        // List jobs and extract check_run_ids
        const jobsResp = await octokit.rest.actions.listJobsForWorkflowRun({ owner, repo, run_id: targetRun.id, per_page: 100 });
        const jobs = Array.isArray(jobsResp.data.jobs) ? jobsResp.data.jobs : [];
        const checkRunIds = [];
        for (const j of jobs) {
          const cru = j && j.check_run_url;
          if (typeof cru === 'string' && cru.includes('/check-runs/')) {
            const idStr = cru.split('/check-runs/')[1];
            const id = idStr ? parseInt(idStr, 10) : NaN;
            if (!Number.isNaN(id)) checkRunIds.push({ id, job_name: j.name });
          }
        }
        const annRoot = path.join(workspace, 'annotations', String(targetRun.id));
        if (!fs.existsSync(annRoot)) fs.mkdirSync(annRoot, { recursive: true });
        const allAnnotations = [];
        for (const { id: checkRunId, job_name } of checkRunIds) {
          let page = 1;
          const budget = 500; // safety cap per run
          while (allAnnotations.length < budget) {
            const perPage = Math.min(100, budget - allAnnotations.length);
            const { data } = await octokit.rest.checks.listAnnotations({ owner, repo, check_run_id: checkRunId, per_page: perPage, page });
            const arr = Array.isArray(data) ? data : [];
            if (!arr.length) break;
            for (const a of arr) {
              allAnnotations.push({
                job_name,
                path: a.path,
                start_line: a.start_line,
                end_line: a.end_line,
                annotation_level: a.annotation_level,
                message: a.message,
                title: a.title,
                raw_details: a.raw_details,
              });
              // If any failing/error annotation belongs to a job that looks like gtest, note it
              try {
                const levelLc = String(a.annotation_level || '').toLowerCase();
                const msgTrim = String(a.message || '').trim();
                const msgLc = msgTrim.toLowerCase();
                const isUnknownFileLead = msgLc.startsWith('unknown file');
                // Primary heuristic: failing/error annotation whose message starts with 'unknown file' is a gtest indicator
                if (levelLc === 'failure' || levelLc === 'error') {
                  sawAnyFailureAnnotations = true;
                  if (isUnknownFileLead) {
                    sawGtestFailure = true;
                    if (job_name) gtestJobNames.add(String(job_name));
                  }
                  // Keep legacy job-name heuristic as a secondary signal
                  else if ((/gtest/i.test(String(job_name || '')) || /gtests/i.test(String(job_name || '')))) {
                    sawGtestFailure = true;
                    if (job_name) gtestJobNames.add(String(job_name));
                  }
                }
              } catch (_) { /* ignore */ }
            }
            if (arr.length < perPage) break;
            page += 1;
          }
        }
        const annPath = path.join(annRoot, 'annotations.json');
        fs.writeFileSync(annPath, JSON.stringify(allAnnotations));
        const relativeAnn = path.relative(workspace, annRoot) || annRoot;
        annotationsIndex[String(targetRun.id)] = relativeAnn;
        core.info(`[LOGS] Fetched annotations for run ${targetRun.id} → ${allAnnotations.length} items`);
      } catch (e) {
        core.warning(`Failed to fetch annotations for run ${targetRun.id}: ${e.message}`);
        annotationsFetchFailed = true;
      }

      // Download logs if: gtest failure detected OR no error/failure annotations found OR annotations fetch failed
      // Separate gtest logs from other logs into different directories
      try {
        if (sawGtestFailure) {
          // Download and index gtest logs
          const runLogsZip = await octokit.rest.actions.downloadWorkflowRunLogs({ owner, repo, run_id: targetRun.id });
          const runDir = path.join(gtestLogsRoot, String(targetRun.id));
          if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });
          const zipPath = path.join(runDir, `logs-${targetRun.id}.zip`);
          fs.writeFileSync(zipPath, Buffer.from(runLogsZip.data));
          const extractDir = path.join(runDir, 'extract');
          if (!fs.existsSync(extractDir)) fs.mkdirSync(extractDir, { recursive: true });
          // Extract quietly to avoid ENOBUFS
          execFileSync('unzip', ['-o', zipPath, '-d', extractDir], { stdio: 'ignore' });

          // Helper to sanitize strings for fuzzy file matching
          const sanitize = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
          const wanted = new Map();
          // Add any job names discovered via 'unknown file' annotations (gtest jobs)
          for (const jn of Array.from(gtestJobNames.values())) {
            const key = sanitize(jn);
            if (!wanted.has(key)) wanted.set(key, { name: jn, files: [] });
          }
          if (wanted.size === 0) {
            // If we didn't identify explicit gtest jobs from jobs API, fall back to any file containing 'gtest'
            wanted.set('gtest', { name: 'gtest', files: [] });
          }
          // Walk extracted tree and record .txt files that match wanted keys
          const stack = [extractDir];
          while (stack.length) {
            const dir = stack.pop();
            const entries = fs.readdirSync(dir, { withFileTypes: true });
            for (const ent of entries) {
              const p = path.join(dir, ent.name);
              if (ent.isDirectory()) {
                stack.push(p);
              } else if (ent.isFile() && /\.txt$/i.test(ent.name)) {
                const fileKey = sanitize(p);
                for (const [k, rec] of wanted.entries()) {
                  if (fileKey.includes(k)) {
                    const rel = path.relative(runDir, p);
                    if (!rec.files.includes(rel)) rec.files.push(rel);
                  }
                }
              }
            }
          }
          const jobsIndex = { jobs: Array.from(wanted.values()).filter(j => (j.files || []).length > 0) };
          const jobsIndexPath = path.join(runDir, 'jobs.json');
          fs.writeFileSync(jobsIndexPath, JSON.stringify(jobsIndex));
          const relativeRunDir = path.relative(workspace, runDir) || runDir;
          gtestLogsIndex[String(targetRun.id)] = relativeRunDir;
          core.info(`[LOGS] Downloaded and indexed gtest logs for run ${targetRun.id} → ${jobsIndex.jobs.length} job(s), ${jobsIndex.jobs.reduce((sum, j) => sum + (j.files || []).length, 0)} files`);
        } else if (!sawAnyFailureAnnotations || annotationsFetchFailed) {
          // Download other logs (non-gtest failures with no annotations, or if annotation fetch failed entirely)
          // Store actual job names from GitHub API instead of extracting from file paths
          const runLogsZip = await octokit.rest.actions.downloadWorkflowRunLogs({ owner, repo, run_id: targetRun.id });
          const runDir = path.join(otherLogsRoot, String(targetRun.id));
          if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });
          const zipPath = path.join(runDir, `logs-${targetRun.id}.zip`);
          fs.writeFileSync(zipPath, Buffer.from(runLogsZip.data));
          const extractDir = path.join(runDir, 'extract');
          if (!fs.existsSync(extractDir)) fs.mkdirSync(extractDir, { recursive: true });
          // Extract quietly to avoid ENOBUFS
          execFileSync('unzip', ['-o', zipPath, '-d', extractDir], { stdio: 'ignore' });

          // Fetch all failing job names from GitHub API with pagination (separate fetch to ensure all jobs are captured)
          const failingJobNames = new Set();
          try {
            // Get jobs for this run - filter for failed jobs, handle pagination
            let page = 1;
            let hasMore = true;
            while (hasMore) {
              const jobsResp = await octokit.rest.actions.listJobsForWorkflowRun({ owner, repo, run_id: targetRun.id, per_page: 100, page });
              const jobs = Array.isArray(jobsResp.data.jobs) ? jobsResp.data.jobs : [];
              for (const job of jobs) {
                // Include jobs that failed (conclusion is 'failure' or 'cancelled')
                const conclusion = (job.conclusion || '').toLowerCase();
                if (conclusion === 'failure' || conclusion === 'cancelled') {
                  if (job.name) {
                    failingJobNames.add(String(job.name));
                  }
                }
              }
              // Continue if we got a full page (might have more), stop if we got fewer
              hasMore = jobs.length >= 100;
              page++;
            }
          } catch (e) {
            core.warning(`Failed to fetch job names from API for run ${targetRun.id}: ${e.message}`);
          }

          // Build a simple index of all .txt log files (no parsing, just list files)
          const logFiles = [];
          const stack = [extractDir];
          while (stack.length) {
            const dir = stack.pop();
            const entries = fs.readdirSync(dir, { withFileTypes: true });
            for (const ent of entries) {
              const p = path.join(dir, ent.name);
              if (ent.isDirectory()) {
                stack.push(p);
              } else if (ent.isFile() && /\.txt$/i.test(ent.name)) {
                const rel = path.relative(runDir, p);
                logFiles.push(rel);
              }
            }
          }
          // Store both file list and actual job names from GitHub API
          const logsListPath = path.join(runDir, 'logs-list.json');
          fs.writeFileSync(logsListPath, JSON.stringify({
            files: logFiles,
            job_names: Array.from(failingJobNames)
          }));
          const relativeRunDir = path.relative(workspace, runDir) || runDir;
          otherLogsIndex[String(targetRun.id)] = relativeRunDir;
          core.info(`[LOGS] Downloaded other logs for run ${targetRun.id} → ${logFiles.length} file(s), ${failingJobNames.size} failing job(s)`);
        }
      } catch (e) {
        core.warning(`Failed to download/index logs for run ${targetRun.id}: ${e.message}`);
      }
    } catch (e) {
      core.warning(`Failed to fetch logs for latest failing run in workflow '${name}': ${e.message}`);
    }
  }
  // Persist annotations index alongside annotations directory
  const annotationsIndexPath = path.join(annotationsRoot, 'annotations-index.json');
  fs.writeFileSync(annotationsIndexPath, JSON.stringify(annotationsIndex));

  // Persist gtest logs index
  const gtestLogsIndexPath = path.join(gtestLogsRoot, 'gtest-logs-index.json');
  try {
    fs.writeFileSync(gtestLogsIndexPath, JSON.stringify(gtestLogsIndex));
  } catch (e) {
    core.warning(`Failed to write gtest logs index: ${e.message}`);
  }

  // Persist other logs index
  const otherLogsIndexPath = path.join(otherLogsRoot, 'other-logs-index.json');
  try {
    fs.writeFileSync(otherLogsIndexPath, JSON.stringify(otherLogsIndex));
  } catch (e) {
    core.warning(`Failed to write other logs index: ${e.message}`);
  }

  return {
    annotationsIndexPath,
    gtestLogsIndexPath,
    otherLogsIndexPath,
  };
}

module.exports = {
  processWorkflowLogs,
};
