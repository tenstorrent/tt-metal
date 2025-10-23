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
      // Additionally fetch the annotations artifact from the same run, if present
      try {
        const owner = github.context.repo.owner;
        const repo = github.context.repo.repo;
        const annotationsArtifact = artifacts.find(a => a && a.name === 'workflow-annotations');
        const workspace = process.env.GITHUB_WORKSPACE || process.cwd();
        const annotationsRoot = path.join(workspace, 'annotations');
        if (!fs.existsSync(annotationsRoot)) fs.mkdirSync(annotationsRoot, { recursive: true });
        let annotationsIndexPath = path.join(annotationsRoot, 'annotations-index.json');
        if (annotationsArtifact) {
          const annZipPath = path.join(tmpDir, `${annotationsArtifact.name}.zip`);
          const respAnn = await octokit.rest.actions.downloadArtifact({ owner, repo, artifact_id: annotationsArtifact.id, archive_format: 'zip' });
          fs.writeFileSync(annZipPath, Buffer.from(respAnn.data));
          const extractAnnDir = path.join(tmpDir, `${annotationsArtifact.name}-extract`);
          if (!fs.existsSync(extractAnnDir)) fs.mkdirSync(extractAnnDir, { recursive: true });
          execFileSync('unzip', ['-o', annZipPath, '-d', extractAnnDir], { stdio: 'ignore' });
          // Find the annotations-index.json inside the extracted tree
          const stack = [extractAnnDir];
          let foundIndex;
          while (stack.length && !foundIndex) {
            const dir = stack.pop();
            const entries = fs.readdirSync(dir, { withFileTypes: true });
            for (const ent of entries) {
              const p = path.join(dir, ent.name);
              if (ent.isDirectory()) stack.push(p);
              else if (ent.isFile() && ent.name === 'annotations-index.json') { foundIndex = p; break; }
            }
          }
          if (foundIndex) {
            const artifactAnnRoot = path.dirname(foundIndex);
            fs.cpSync(artifactAnnRoot, annotationsRoot, { recursive: true });
            annotationsIndexPath = path.join(annotationsRoot, 'annotations-index.json');
            core.info(`[TEST MODE] Restored annotations to ${annotationsRoot}`);
          } else {
            core.info('[TEST MODE] No annotations-index.json found; creating empty index');
            if (!fs.existsSync(annotationsIndexPath)) fs.writeFileSync(annotationsIndexPath, JSON.stringify({}));
          }
        } else {
          core.info('[TEST MODE] No workflow-annotations artifact found in selected run; creating empty index');
          if (!fs.existsSync(annotationsIndexPath)) fs.writeFileSync(annotationsIndexPath, JSON.stringify({}));
        }
        core.setOutput('annotations-root', annotationsRoot);
        core.setOutput('annotations-index-path', annotationsIndexPath);
      } catch (e) {
        core.warning(`[TEST MODE] Failed to restore annotations artifact: ${e.message}`);
      }

      // Additionally fetch the commits artifact from the same run, if present
      try {
        const owner = github.context.repo.owner;
        const repo = github.context.repo.repo;
        const commitsArtifact = artifacts.find(a => a && a.name === 'commits-main');
        const workspace = process.env.GITHUB_WORKSPACE || process.cwd();
        let commitsPath = path.join(workspace, 'commits-main.json');
        if (commitsArtifact) {
          const commitsZipPath = path.join(tmpDir, `${commitsArtifact.name}.zip`);
          const respCommits = await octokit.rest.actions.downloadArtifact({ owner, repo, artifact_id: commitsArtifact.id, archive_format: 'zip' });
          fs.writeFileSync(commitsZipPath, Buffer.from(respCommits.data));
          const extractCommitsDir = path.join(tmpDir, `${commitsArtifact.name}-extract`);
          if (!fs.existsSync(extractCommitsDir)) fs.mkdirSync(extractCommitsDir, { recursive: true });
          execFileSync('unzip', ['-o', commitsZipPath, '-d', extractCommitsDir], { stdio: 'ignore' });
          // Find commits-main.json
          const stack2 = [extractCommitsDir];
          let foundCommitsPath;
          while (stack2.length && !foundCommitsPath) {
            const dir = stack2.pop();
            const entries = fs.readdirSync(dir, { withFileTypes: true });
            for (const ent of entries) {
              const p = path.join(dir, ent.name);
              if (ent.isDirectory()) stack2.push(p);
              else if (ent.isFile() && ent.name === 'commits-main.json') { foundCommitsPath = p; break; }
            }
          }
          if (foundCommitsPath) {
            fs.cpSync(foundCommitsPath, commitsPath, { recursive: false });
            core.info(`[TEST MODE] Restored commits index to ${commitsPath}`);
          } else {
            core.info('[TEST MODE] No commits-main.json found; creating empty list');
            if (!fs.existsSync(commitsPath)) fs.writeFileSync(commitsPath, JSON.stringify([]));
          }
        } else {
          core.info('[TEST MODE] No commits-main artifact found in selected run; creating empty list');
          if (!fs.existsSync(commitsPath)) fs.writeFileSync(commitsPath, JSON.stringify([]));
        }
        core.setOutput('commits-path', commitsPath);
      } catch (e) {
        core.warning(`[TEST MODE] Failed to restore commits artifact: ${e.message}`);
      }

      // Additionally download and index logs for failing workflow runs referenced by this prior aggregate run (test mode)
      try {
        const owner = github.context.repo.owner;
        const repo = github.context.repo.repo;
        const workspace = process.env.GITHUB_WORKSPACE || process.cwd();

        // Restore gtest logs
        const gtestLogsRoot = path.join(workspace, 'logs', 'gtest');
        if (!fs.existsSync(gtestLogsRoot)) fs.mkdirSync(gtestLogsRoot, { recursive: true });
        let gtestLogsIndexPath = path.join(gtestLogsRoot, 'gtest-logs-index.json');
        const gtestLogsArtifact = artifacts.find(a => a && a.name === 'workflow-gtest-logs');
        if (gtestLogsArtifact) {
          const logsZipPath = path.join(tmpDir, `${gtestLogsArtifact.name}.zip`);
          const respLogs = await octokit.rest.actions.downloadArtifact({ owner, repo, artifact_id: gtestLogsArtifact.id, archive_format: 'zip' });
          fs.writeFileSync(logsZipPath, Buffer.from(respLogs.data));
          const extractLogsDir = path.join(tmpDir, `${gtestLogsArtifact.name}-extract`);
          if (!fs.existsSync(extractLogsDir)) fs.mkdirSync(extractLogsDir, { recursive: true });
          execFileSync('unzip', ['-o', logsZipPath, '-d', extractLogsDir], { stdio: 'ignore' });
          // Copy the extracted logs tree into workspace logs/gtest/
          fs.cpSync(extractLogsDir, gtestLogsRoot, { recursive: true });
          const candidateIdx = path.join(gtestLogsRoot, 'gtest-logs-index.json');
          if (fs.existsSync(candidateIdx)) {
            gtestLogsIndexPath = candidateIdx;
          } else if (!fs.existsSync(gtestLogsIndexPath)) {
            fs.writeFileSync(gtestLogsIndexPath, JSON.stringify({}));
          }
          core.info(`[TEST MODE] Restored gtest logs to ${gtestLogsRoot}`);
        } else {
          core.info('[TEST MODE] No workflow-gtest-logs artifact found in selected run; creating empty index');
          if (!fs.existsSync(gtestLogsIndexPath)) fs.writeFileSync(gtestLogsIndexPath, JSON.stringify({}));
        }

        // Restore other logs
        const otherLogsRoot = path.join(workspace, 'logs', 'other');
        if (!fs.existsSync(otherLogsRoot)) fs.mkdirSync(otherLogsRoot, { recursive: true });
        let otherLogsIndexPath = path.join(otherLogsRoot, 'other-logs-index.json');
        const otherLogsArtifact = artifacts.find(a => a && a.name === 'workflow-other-logs');
        if (otherLogsArtifact) {
          const logsZipPath = path.join(tmpDir, `${otherLogsArtifact.name}.zip`);
          const respLogs = await octokit.rest.actions.downloadArtifact({ owner, repo, artifact_id: otherLogsArtifact.id, archive_format: 'zip' });
          fs.writeFileSync(logsZipPath, Buffer.from(respLogs.data));
          const extractLogsDir = path.join(tmpDir, `${otherLogsArtifact.name}-extract`);
          if (!fs.existsSync(extractLogsDir)) fs.mkdirSync(extractLogsDir, { recursive: true });
          execFileSync('unzip', ['-o', logsZipPath, '-d', extractLogsDir], { stdio: 'ignore' });
          // Copy the extracted logs tree into workspace logs/other/
          fs.cpSync(extractLogsDir, otherLogsRoot, { recursive: true });
          const candidateIdx = path.join(otherLogsRoot, 'other-logs-index.json');
          if (fs.existsSync(candidateIdx)) {
            otherLogsIndexPath = candidateIdx;
          } else if (!fs.existsSync(otherLogsIndexPath)) {
            fs.writeFileSync(otherLogsIndexPath, JSON.stringify({}));
          }
          core.info(`[TEST MODE] Restored other logs to ${otherLogsRoot}`);
        } else {
          core.info('[TEST MODE] No workflow-other-logs artifact found in selected run; creating empty index');
          if (!fs.existsSync(otherLogsIndexPath)) fs.writeFileSync(otherLogsIndexPath, JSON.stringify({}));
        }

        core.setOutput('gtest-logs-root', gtestLogsRoot);
        core.setOutput('gtest-logs-index-path', gtestLogsIndexPath);
        core.setOutput('other-logs-root', otherLogsRoot);
        core.setOutput('other-logs-index-path', otherLogsIndexPath);
      } catch (e) {
        core.warning(`[TEST MODE] Failed to restore logs artifact: ${e.message}`);
      }

      // Exit early
      return;
    }
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

    // Download logs for the latest failing run per workflow and build an index
    // Constraint: Only fetch logs when the latest run for that workflow (on target branch)
    // is failing (i.e., conclusion neither success nor skipped/cancelled).
    // The logs will be extracted under a dedicated directory so downstream steps
    // can parse them without performing network calls again.
    // Separate gtest logs from other logs (non-gtest failures with no annotations)
    const owner = github.context.repo.owner;
    const repo = github.context.repo.repo;
    const workspace = process.env.GITHUB_WORKSPACE || process.cwd();
    const gtestLogsRoot = path.join(workspace, 'logs', 'gtest');
    const otherLogsRoot = path.join(workspace, 'logs', 'other');
    if (!fs.existsSync(gtestLogsRoot)) {
      fs.mkdirSync(gtestLogsRoot, { recursive: true });
    }
    if (!fs.existsSync(otherLogsRoot)) {
      fs.mkdirSync(otherLogsRoot, { recursive: true });
    }
    const annotationsIndex = {};
    const gtestLogsIndex = {};
    const otherLogsIndex = {};
    for (const [name, runs] of grouped.entries()) {
      try {
        // Consider only the target branch and sort newest first
        const branchRuns = (runs || [])
          .filter(r => r && r.head_branch === branch)
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
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
        if (!targetRun) continue;

        // Fetch check-run annotations for this failing run
        try {
          // List jobs and extract check_run_ids
          const jobsResp = await octokit.rest.actions.listJobsForWorkflowRun({ owner, repo, run_id: targetRun.id, per_page: 100 });
          const jobs = Array.isArray(jobsResp.data.jobs) ? jobsResp.data.jobs : [];
          const checkRunIds = [];
          const gtestJobNames = new Set();
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
          let sawGtestFailure = false;
          let sawAnyFailureAnnotations = false;
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
          core.info(`Fetched annotations for failing run ${targetRun.id} → ${allAnnotations.length} items`);

          // Download logs if: gtest failure detected OR no error/failure annotations found
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
              core.info(`Downloaded and indexed gtest logs for failing run ${targetRun.id} → ${jobsIndex.jobs.length} job(s)`);
            } else if (!sawAnyFailureAnnotations) {
              // Download other logs (non-gtest failures with no annotations)
              // Just download and list files, no detailed indexing
              const runLogsZip = await octokit.rest.actions.downloadWorkflowRunLogs({ owner, repo, run_id: targetRun.id });
              const runDir = path.join(otherLogsRoot, String(targetRun.id));
              if (!fs.existsSync(runDir)) fs.mkdirSync(runDir, { recursive: true });
              const zipPath = path.join(runDir, `logs-${targetRun.id}.zip`);
              fs.writeFileSync(zipPath, Buffer.from(runLogsZip.data));
              const extractDir = path.join(runDir, 'extract');
              if (!fs.existsSync(extractDir)) fs.mkdirSync(extractDir, { recursive: true });
              // Extract quietly to avoid ENOBUFS
              execFileSync('unzip', ['-o', zipPath, '-d', extractDir], { stdio: 'ignore' });

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
              const logsListPath = path.join(runDir, 'logs-list.json');
              fs.writeFileSync(logsListPath, JSON.stringify({ files: logFiles }));
              const relativeRunDir = path.relative(workspace, runDir) || runDir;
              otherLogsIndex[String(targetRun.id)] = relativeRunDir;
              core.info(`Downloaded other logs for failing run ${targetRun.id} → ${logFiles.length} file(s)`);
            }
          } catch (e) {
            core.warning(`Failed to download/index logs for run ${targetRun.id}: ${e.message}`);
          }
        } catch (e) {
          core.warning(`Failed to fetch annotations for run ${targetRun.id}: ${e.message}`);
        }
      } catch (e) {
        core.warning(`Failed to fetch logs for latest failing run in workflow '${name}': ${e.message}`);
      }
    }
    // Persist annotations index alongside annotations directory
    const annotationsRoot = path.join(workspace, 'annotations');
    if (!fs.existsSync(annotationsRoot)) fs.mkdirSync(annotationsRoot, { recursive: true });
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

    // Build a commits index for the main branch within the last N days
    // The index is an array of commits sorted by commit author/commit date ascending.
    // Each entry: { sha, short, url, author_login, author_name, author_url, date }
    const commits = [];
    try {
      const sinceIso = getCutoffDate(days).toISOString();
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
        if (!arr.length) break;
        for (const c of arr) {
          const sha = c.sha;
          const short = sha ? sha.substring(0, 7) : '';
          const url = `https://github.com/${owner}/${repo}/commit/${sha}`;
          const author_login = c.author?.login;
          const author_name = c.commit?.author?.name;
          const author_url = c.author?.html_url;
          const date = c.commit?.author?.date || c.commit?.committer?.date || null;
          const message = c.commit?.message || '';
          const description = typeof message === 'string' ? (message.split(/\r?\n/)[0] || '') : '';
          commits.push({ sha, short, url, author_login, author_name, author_url, date, message, description });
        }
        if (arr.length < perPage) break;
        page++;
      }
      // Sort oldest -> newest by date for deterministic slicing
      commits.sort((a, b) => new Date(a.date || 0) - new Date(b.date || 0));
    } catch (e) {
      core.warning(`Failed to build commits index: ${e.message}`);
    }
    const commitsPath = path.join(workspace, 'commits-main.json');
    fs.writeFileSync(commitsPath, JSON.stringify(commits));
    // Set output
    core.setOutput('total-runs', mergedRuns.length);
    core.setOutput('workflow-count', grouped.size);
    core.setOutput('cache-path', outputPath);
    core.setOutput('gtest-logs-root', gtestLogsRoot);
    core.setOutput('gtest-logs-index-path', gtestLogsIndexPath);
    core.setOutput('other-logs-root', otherLogsRoot);
    core.setOutput('other-logs-index-path', otherLogsIndexPath);
    core.setOutput('annotations-root', annotationsRoot);
    core.setOutput('annotations-index-path', annotationsIndexPath);
    core.setOutput('commits-path', commitsPath);
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
