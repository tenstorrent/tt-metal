// Analyze Workflow Data GitHub Action
// This action analyzes cached workflow run data and generates a summary report of workflow statuses.
// It provides two tables: one for push-triggered workflows and another for scheduled workflows.
// For scheduled workflows, it also tracks the last known good commit and earliest bad commit.
//
// See: https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28

// These are all node.js modules needed for the action to work
const core = require('@actions/core'); // Core utilities for I/O
const github = require('@actions/github'); // GitHub API client
const fs = require('fs'); // File system operations
const path = require('path'); // File path utilities

// Import modules
const dataLoading = require('./lib/data-loading');
const errorProcessing = require('./lib/error-processing');
const analysis = require('./lib/analysis');
const reporting = require('./lib/reporting');

// Import constants used in this file
const {
  DEFAULT_LOOKBACK_DAYS,
  SHA_SHORT_LENGTH,
  EMPTY_VALUE,
  DEFAULT_INFRA_OWNER,
} = dataLoading;
// Import functions from modules
const {
  loadAnnotationsIndexFromFile,
  getAnnotationsDirForRunId,
  loadLogsIndexFromFile,
  getGtestLogsDirForRunId,
  getOtherLogsDirForRunId,
  loadLastSuccessTimestamps,
  getTimeSinceLastSuccess,
  getCommitDescription,
  loadCommitsIndex,
  listCommitsBetweenOffline,
  setAnnotationsIndexMap,
  setGtestLogsIndexMap,
  setOtherLogsIndexMap,
  setLastSuccessTimestamps,
  setCommitsIndex,
} = dataLoading;

const {
  loadOwnersMapping,
  normalizeOwners,
  extractSignificantTokens,
  getJobNameComponentTail,
  findOwnerForLabel,
  inferJobAndTestFromSnippet,
  resolveOwnersForSnippet,
  fetchErrorSnippetsForRun,
  renderErrorsTable,
} = errorProcessing;

const {
  getWorkflowStats,
  findFirstFailInWindow,
  renderCommitsTable,
  fetchPRInfo,
  fetchCommitAuthor,
} = analysis;

const {
  getWorkflowLink,
  findGoodBadCommits,
  getLastRunInfo,
  generateSummaryBox,
  buildReport,
  filterRunsByDate,
} = reporting;


/**
 * Main function to run the action
 */
async function run() {
  try {
    // Get inputs
    const cachePath = core.getInput('cache-path', { required: true }); // get the workflow data cache made by the fetch-workflow-data action
    const previousCachePath = core.getInput('previous-cache-path', { required: false }); // get the previous workflow data cache made by the fetch-workflow-data action from the most recent previous run on main branch
    const workflowConfigs = JSON.parse(core.getInput('workflow_configs', { required: true })); // get the json of pipelines that we want to analyze
    const days = parseInt(core.getInput('days') || DEFAULT_LOOKBACK_DAYS, 10); // get the number of days to look back for workflow data
    const alertAll = String(core.getInput('alert-all') || 'false').toLowerCase() === 'true'; // get the alert-all input from the action inputs
    const annotationsIndexPath = core.getInput('annotations-index-path', { required: false }); // optional: path to JSON mapping runId -> annotations dir
    const commitsPath = core.getInput('commits-path', { required: false }); // optional: path to commits index JSON
    const gtestLogsIndexPath = core.getInput('gtest-logs-index-path', { required: false }); // optional: path to gtest logs index JSON
    const otherLogsIndexPath = core.getInput('other-logs-index-path', { required: false }); // optional: path to other logs index JSON
    const lastSuccessTimestampsPath = core.getInput('last-success-timestamps-path', { required: false }); // optional: path to last success timestamps JSON

    // Validate inputs
    if (!fs.existsSync(cachePath)) {
      throw new Error(`Cache file not found at ${cachePath}`); // throw an error if the cache file does not exist (no data was fetched)
    }
    if (!Array.isArray(workflowConfigs)) {
      throw new Error('Workflow configs must be a JSON array'); // throw an error if the workflow configs are not a JSON array
    }
    if (isNaN(days) || days <= 0) {
      throw new Error('Days must be a positive number'); // throw an error if the days is not a positive number
    }

    // Load cached data
    const grouped = JSON.parse(fs.readFileSync(cachePath, 'utf8')); // parse the cached data into a json object
    const hasPrevious = previousCachePath && fs.existsSync(previousCachePath); // check if the previous cache file exists
    const previousGrouped = hasPrevious ? JSON.parse(fs.readFileSync(previousCachePath, 'utf8')) : null; // parse the previous cached data into a json object


    if (annotationsIndexPath) {
      const annotationsIndexMap = loadAnnotationsIndexFromFile(annotationsIndexPath);
      setAnnotationsIndexMap(annotationsIndexMap);
      if (annotationsIndexMap && annotationsIndexMap.size) {
        core.info(`Loaded annotations index with ${annotationsIndexMap.size} entries from ${annotationsIndexPath}`);
      } else if (annotationsIndexPath) {
        core.info(`No valid entries found in annotations index file at ${annotationsIndexPath}`);
      }
    }

    if (gtestLogsIndexPath) {
      const gtestLogsIndexMap = loadLogsIndexFromFile(gtestLogsIndexPath);
      setGtestLogsIndexMap(gtestLogsIndexMap);
      if (gtestLogsIndexMap && gtestLogsIndexMap.size) {
        core.info(`Loaded gtest logs index with ${gtestLogsIndexMap.size} entries from ${gtestLogsIndexPath}`);
      } else if (gtestLogsIndexPath) {
        core.info(`No valid entries found in gtest logs index file at ${gtestLogsIndexPath}`);
      }
    }

    if (otherLogsIndexPath) {
      const otherLogsIndexMap = loadLogsIndexFromFile(otherLogsIndexPath);
      setOtherLogsIndexMap(otherLogsIndexMap);
      if (otherLogsIndexMap && otherLogsIndexMap.size) {
        core.info(`Loaded other logs index with ${otherLogsIndexMap.size} entries from ${otherLogsIndexPath}`);
      } else if (otherLogsIndexPath) {
        core.info(`No valid entries found in other logs index file at ${otherLogsIndexPath}`);
      }
    }

    if (lastSuccessTimestampsPath) {
      const lastSuccessTimestamps = loadLastSuccessTimestamps(lastSuccessTimestampsPath);
      setLastSuccessTimestamps(lastSuccessTimestamps);
      if (lastSuccessTimestamps && lastSuccessTimestamps.size) {
        core.info(`Loaded last success timestamps with ${lastSuccessTimestamps.size} entries from ${lastSuccessTimestampsPath}`);
      } else {
        core.info(`No valid entries found in last success timestamps file at ${lastSuccessTimestampsPath}`);
      }
    }

    // Load commits index (optional)
    const commitsIndex = loadCommitsIndex(commitsPath) || [];
    setCommitsIndex(commitsIndex);
    core.info(`Loaded commits index entries: ${Array.isArray(commitsIndex) ? commitsIndex.length : 0}`);

    // Track failed workflows
    const failedWorkflows = [];

    // Filter and process each workflow configuration
    const filteredGrouped = new Map();
    const filteredPreviousGrouped = new Map();
    for (const config of workflowConfigs) { // for each pipeline name that we want to analyze
      core.info(`Processing config: ${JSON.stringify(config)}`); // log the config that we are processing
      for (const [name, runs] of grouped) { // for each set of pipeline runs in the cached data
        if ((config.wkflw_name && name === config.wkflw_name) ||
            (config.wkflw_prefix && name.startsWith(config.wkflw_prefix))) { // if the pipeline run name matches the pipeline name or pipeline prefix
          core.info(`Matched workflow: ${name} with config: ${JSON.stringify(config)}`);
          // Filter runs by date range
          const filteredRuns = filterRunsByDate(runs, days); // filter the runs by the number of days to look back for workflow data
          if (filteredRuns.length > 0) {
            filteredGrouped.set(name, filteredRuns); // set the filtered runs in the filtered grouped map

            // Check if latest run on main is failing
            // First deduplicate by run ID, keeping highest attempt
            const runsByID = new Map();
            for (const run of filteredRuns.filter(r => r.head_branch === 'main')) {
              const runId = run.id;
              const currentAttempt = run.run_attempt || 1;
              const existingRun = runsByID.get(runId);
              const existingAttempt = existingRun ? (existingRun.run_attempt || 1) : 0;
              if (!existingRun || currentAttempt > existingAttempt) {
                runsByID.set(runId, run);
              }
            }
            const mainBranchRuns = Array.from(runsByID.values())
              .sort((a, b) => {
                // Sort by date (newest first), then by run_attempt (highest first) as tiebreaker
                const dateDiff = new Date(b.created_at) - new Date(a.created_at);
                if (dateDiff !== 0){
                  return dateDiff;
                }
                const attemptA = a.run_attempt || 1;
                const attemptB = b.run_attempt || 1;
                return attemptB - attemptA; // Prefer higher attempt number
              });
            if (mainBranchRuns[0]?.conclusion !== 'success') {
              failedWorkflows.push(name); // if the latest run on main is failing, add the pipeline name to the failed workflows array
            }
          }
        }
      }

      if (hasPrevious && Array.isArray(previousGrouped)) {
        for (const [name, runs] of previousGrouped) { // for each set of pipeline runs in the previous cached data
          if ((config.wkflw_name && name === config.wkflw_name) ||
              (config.wkflw_prefix && name.startsWith(config.wkflw_prefix))) { // if the pipeline run name matches the pipeline name or pipeline prefix
            const filteredRuns = filterRunsByDate(runs, days); // filter the runs by the number of days to look back for workflow data
            if (filteredRuns.length > 0) {
              filteredPreviousGrouped.set(name, filteredRuns); // set the filtered runs in the filtered previous grouped map
              // this should make it so that if a specific pipeline run from the old aggregated data is no longer within the time window, it stops being used
            }
          }
        }
      }
    }

    // Generate primary report
    const mainReport = await buildReport(filteredGrouped, github.context);

    // END OF BASIC ANALYZE STEP

    // Cache for error snippets to avoid fetching the same run's errors multiple times
    const errorSnippetsCache = new Map();

    // Optional: Build Slack-ready alert message for all failing workflows with owner mentions
    let alertAllMessage = '';
    if (failedWorkflows.length > 0) {
      const mention = (owners) => {
        const arr = Array.isArray(owners) ? owners : (owners ? [owners] : []);
        const parts = arr.map(o => {
          if (!o || !o.id) return '';
          const id = String(o.id);
          if (id.startsWith('S')) {
            const fallback = o.name ? `@${o.name}` : '@team';
            return `<!subteam^${id}|${fallback}>`;
          }
          return `<@${id}>`;
        }).filter(Boolean);
        return parts.length ? parts.join(' ') : '';
      }; // format owners into Slack mentions; supports user groups (S-ids)

      // create a list of all the failing workflows with their owner information for slack messaging
      const failingItems = [];
      for (const [name, runs] of filteredGrouped.entries()) { // for each pipeline run in the filtered grouped map
        // First deduplicate by run ID, keeping highest attempt
        const runsByID = new Map();
        for (const run of runs.filter(r => r.head_branch === 'main')) {
          const runId = run.id;
          const currentAttempt = run.run_attempt || 1;
          const existingRun = runsByID.get(runId);
          const existingAttempt = existingRun ? (existingRun.run_attempt || 1) : 0;
          if (!existingRun || currentAttempt > existingAttempt) {
            runsByID.set(runId, run);
          }
        }
        const mainRuns = Array.from(runsByID.values())
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
        if (!mainRuns[0] || mainRuns[0].conclusion === 'success') continue; // if the latest run on main is not failing, continue
        // Try to attach owners from the first failing run's label via snippets; fallback to job name
        // Use the latest failing run for snippet-based owner detection
        const latestFail = mainRuns.find(r => r.conclusion !== 'success');
        let owners = undefined;
        let combinedOwnerNames = [];
        let failingJobNames = undefined;
        try {
          const errs = await fetchErrorSnippetsForRun(
            latestFail.id,
            20,
            undefined,
            getAnnotationsDirForRunId(latestFail.id)
          ); // get the error snippets for the latest failing run using annotations if available
          errorSnippetsCache.set(latestFail.id, errs); // cache the error snippets for reuse
          // Infer job/test and resolve owners per snippet, then aggregate
          const ownerSet = new Map();
          const genericExitOrigOwners = new Map(); // key by name to dedupe
          const isGenericExit = (s) => typeof s === 'string' && /^Process completed with exit code 1\.?$/i.test(String(s).trim());
          for (const sn of (errs || [])) {
            const inferred = inferJobAndTestFromSnippet(sn);
            if (inferred) { sn.job = inferred.job; sn.test = inferred.test; }
            resolveOwnersForSnippet(sn, name);
            if (Array.isArray(sn.owner)) {
              for (const o of sn.owner) {
                if (!o) continue;
                const k = `${o.id || ''}|${o.name || ''}`;
                ownerSet.set(k, o);
              }
            }
            // Capture original pipeline owners (names + ids if available) when infra override occurred
            if (sn.owner_source && String(sn.owner_source).startsWith('infra_due_to_missing_test') && Array.isArray(sn.original_owners)) {
              for (const oo of sn.original_owners) {
                const nm = (oo && (oo.name || oo.id)) || '';
                if (nm) genericExitOrigOwners.set(nm, true);
                // Also include in owners for downstream mention support if ids exist
                if (oo) {
                  const k2 = `${oo.id || ''}|${oo.name || ''}`;
                  ownerSet.set(k2, { id: oo.id, name: oo.name });
                }
              }
            }
          }
          owners = Array.from(ownerSet.values());
          // Build combined owner names list (infra + inferred + pipeline owners) without extra labeling text
          const origNames = Array.from(genericExitOrigOwners.keys());
          combinedOwnerNames = (() => {
            const seen = new Map();
            for (const o of (owners || [])) {
              const nm = (o && (o.name || o.id)) || '';
              if (nm) seen.set(nm, true);
            }
            for (const nm of origNames) { if (nm) seen.set(nm, true); }
            return Array.from(seen.keys());
          })();
          // Extract failing job names from error snippets
          failingJobNames = [];
          const jobs = new Set();
          for (const sn of (errs || [])) {
            const jobName = (sn && sn.job) ? String(sn.job) : '';
            if (jobName) jobs.add(jobName);
          }
          failingJobNames = Array.from(jobs);
        } catch (_) { /* ignore */ }
        // Ensure failingJobNames exists even if try block fails
        if (!failingJobNames) failingJobNames = [];
        // Fallback: try to resolve owners from the workflow name
        if (!owners || owners.length === 0) {
          owners = findOwnerForLabel(name) || [DEFAULT_INFRA_OWNER];
        }
        // When alertAll is false, list owner names (no pings); include pipeline owners if known
        const ownerNamesText = (() => {
          const names = Array.isArray(combinedOwnerNames) ? combinedOwnerNames : [];
          return (names.length ? names.join(', ') : DEFAULT_INFRA_OWNER.name);
        })();
        const fallbackMention = `<!subteam^${DEFAULT_INFRA_OWNER.id}|${DEFAULT_INFRA_OWNER.name}>`;
        const ownerMentions = alertAll ? (mention(owners) || fallbackMention) : ownerNamesText; // do not ping in non-regression summary
        const jobsNote = failingJobNames.length > 0 ? ` (failed ${failingJobNames.join(', ')})` : '';
        const wfUrl = getWorkflowLink(github.context, runs[0]?.path); // get the workflow url link for the pipeline run (can use any run to get the workflow link)
        failingItems.push(`• ${name} ${wfUrl ? `<${wfUrl}|open>` : ''} ${ownerMentions}${jobsNote}`.trim()); // the run is failing because if it wasn't the for loop would have continued earlier
      }
      if (failingItems.length) {
        alertAllMessage = [
          '*Alerts: failing workflows on main*',
          ...failingItems
        ].join('\n'); // building part of the slack message
      }
    }

    // Compute status changes vs previous and write JSON
    const computeLatestConclusion = (runs) => {
      // First deduplicate by run ID, keeping highest attempt
      const runsByID = new Map();
      for (const run of runs.filter(r => r.head_branch === 'main')) {
        const runId = run.id;
        const currentAttempt = run.run_attempt || 1;
        const existingRun = runsByID.get(runId);
        const existingAttempt = existingRun ? (existingRun.run_attempt || 1) : 0;
        if (!existingRun || currentAttempt > existingAttempt) {
          runsByID.set(runId, run);
        }
      }
      const mainBranchRuns = Array.from(runsByID.values())
        .sort((a, b) => {
          // Sort by date (newest first), then by run_attempt (highest first) as tiebreaker
          const dateDiff = new Date(b.created_at) - new Date(a.created_at);
          if (dateDiff !== 0){
            return dateDiff;
          }
          const attemptA = a.run_attempt || 1;
          const attemptB = b.run_attempt || 1;
          return attemptB - attemptA; // Prefer higher attempt number
        });
      const latest = mainBranchRuns[0];
      if (!latest) return null;
      return latest.conclusion === 'success' ? 'success' : 'failure';
    }; // compute the latest conclusion of the pipeline run
    const computeLatestRunInfo = (runs) => {
      // First deduplicate by run ID, keeping highest attempt
      const runsByID = new Map();
      for (const run of runs.filter(r => r.head_branch === 'main')) {
        const runId = run.id;
        const currentAttempt = run.run_attempt || 1;
        const existingRun = runsByID.get(runId);
        const existingAttempt = existingRun ? (existingRun.run_attempt || 1) : 0;
        if (!existingRun || currentAttempt > existingAttempt) {
          runsByID.set(runId, run);
        }
      }
      const mainBranchRuns = Array.from(runsByID.values())
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
      const latest = mainBranchRuns[0];
      if (!latest) return null;
      return { id: latest.id, url: latest.html_url, created_at: latest.created_at, head_sha: latest.head_sha, path: latest.path };
    }; // get the latest run info for the pipeline run

    const allNames = new Set([
      ...Array.from(filteredGrouped.keys()),
      ...Array.from(filteredPreviousGrouped.keys())
    ]); // create a set of all the pipeline names

    const changes = [];
    const regressedDetails = [];
    const stayedFailingDetails = [];
    for (const name of allNames) { // for each pipeline name in the set of all the pipeline names
      const currentRuns = filteredGrouped.get(name); // get the current runs for the pipeline name
      const previousRuns = filteredPreviousGrouped.get(name); // get the previous runs for the pipeline name
      if (!currentRuns || !previousRuns) continue; // require data on both sides (could be problematic if a pipeline is added to the config after the first run)
      const current = computeLatestConclusion(currentRuns); // compute the latest conclusion of the current runs
      const previous = computeLatestConclusion(previousRuns); // compute the latest conclusion of the previous runs
      if (!current || !previous) continue; // if the current or previous runs are not found, continue

      let change;
      // determine the change in the pipeline run
      if (previous === 'success' && current === 'success') change = 'stayed_succeeding';
      else if (previous !== 'success' && current !== 'success') change = 'stayed_failing';
      else if (previous !== 'success' && current === 'success') change = 'fail_to_success';
      else if (previous === 'success' && current !== 'success') change = 'success_to_fail';

      if (change) { // if the change is found, compute the latest run info for the current runs
        const info = computeLatestRunInfo(currentRuns); // get the run info for the latest run on the current runs
        const workflowUrl = info?.path ? getWorkflowLink(github.context, info.path) : undefined; // get the workflow url link for the pipeline run (can use any run to get the workflow link)
        const aggregateRunUrl = `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/actions/runs/${github.context.runId}`; // get the specific link to the aggregate workflow data run that's executing this script right now
        const commitUrl = info?.head_sha ? `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${info.head_sha}` : undefined; // get the commit url link for the latest pipeline run
        const commitShort = info?.head_sha ? info.head_sha.substring(0, 7) : undefined; // get the short sha of the latest pipeline run
        changes.push({ name, previous, current, change, run_id: info?.id, run_url: info?.url, created_at: info?.created_at, workflow_url: workflowUrl, workflow_path: info?.path, aggregate_run_url: aggregateRunUrl, commit_sha: info?.head_sha, commit_short: commitShort, commit_url: commitUrl }); // push the change into the changes array
        if (change === 'success_to_fail' && info) { // if the change is a success to fail, push the change into the regressed details array. Each item represents a pipeline with its latest failing run.
          regressedDetails.push({ name, run_id: info.id, run_url: info.url, created_at: info.created_at, workflow_url: workflowUrl, workflow_path: info.path, aggregate_run_url: aggregateRunUrl, commit_sha: info.head_sha, commit_short: commitShort, commit_url: commitUrl, owners: [] });
        }
        else if (change === 'stayed_failing' && info) { // if the change is a stayed failing, push the change into the stayed failing details array. this is for the latest run
          stayedFailingDetails.push({ name, run_id: info.id, run_url: info.url, created_at: info.created_at, workflow_url: workflowUrl, workflow_path: info.path, aggregate_run_url: aggregateRunUrl, commit_sha: info.head_sha, commit_short: commitShort, commit_url: commitUrl });
        }
      }
    }

    // Helper to get main runs within the current window from a grouped collection
    // Deduplicates by run ID (keeping highest attempt) before sorting
    const getMainWindowRuns = (runs) => {
      // First deduplicate by run ID, keeping highest attempt
      const runsByID = new Map();
      for (const run of runs.filter(r => r.head_branch === 'main')) {
        const runId = run.id;
        const currentAttempt = run.run_attempt || 1;
        const existingRun = runsByID.get(runId);
        const existingAttempt = existingRun ? (existingRun.run_attempt || 1) : 0;
        if (!existingRun || currentAttempt > existingAttempt) {
          runsByID.set(runId, run);
        }
      }
      return Array.from(runsByID.values())
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
    };

    // Enrich regressions with first failing run within the window
    for (const item of regressedDetails) {
      try {
        const windowRuns = getMainWindowRuns(filteredGrouped.get(item.name) || []); // get the main window runs for the pipeline name
        const res = findFirstFailInWindow(windowRuns); // find the first failing run in the window (oldest failing run)
        if (res && res.run) {
          // set the first failed run id, url, created at, head sha, and head short for the pipeline run
          item.first_failed_run_id = res.run.id;
          item.first_failed_run_url = res.run.html_url;
          item.first_failed_created_at = res.run.created_at;
          item.first_failed_head_sha = res.run.head_sha;
          item.first_failed_head_short = res.run.head_sha ? res.run.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;
          item.no_success_in_window = !!res.noSuccessInWindow; // explicitely cast to a boolean
          if (!res.noSuccessInWindow && res.boundarySuccessRun && res.boundarySuccessRun.head_sha) {
            // Get commits between boundary success and first failing run (inclusive of failing run)
            item.commits_between = listCommitsBetweenOffline(github.context, res.boundarySuccessRun.head_sha, item.first_failed_head_sha);
          }
          // Commit author enrichment is now superseded by commits_between list; keep top-level for convenience if present
          if (item.first_failed_head_sha) {
            const author = await fetchCommitAuthor(item.first_failed_head_sha);
            item.first_failed_author_login = author.login;
            item.first_failed_author_name = author.name;
            item.first_failed_author_url = author.htmlUrl;
          }
          // Error snippets from the latest failing run (item.run_id already contains the latest failing run)
          // Reuse cached error snippets if available, otherwise fetch
          if (item.run_id) {
            item.error_snippets = errorSnippetsCache.get(item.run_id) || await fetchErrorSnippetsForRun(
              item.run_id,
              20,
              undefined,
              getAnnotationsDirForRunId(item.run_id)
            );
            if (!errorSnippetsCache.has(item.run_id)) {
              errorSnippetsCache.set(item.run_id, item.error_snippets);
            }
          } else {
            item.error_snippets = [];
          }
          // Infer job/test and resolve owners for each snippet
          try {
            for (const sn of (item.error_snippets || [])) {
              const inferred = inferJobAndTestFromSnippet(sn);
              if (inferred) { sn.job = inferred.job; sn.test = inferred.test; }
              resolveOwnersForSnippet(sn, item.name);
            }
            // Aggregate owners across snippets for this regression item
            const ownerSet = new Map();
            const genericExitOrigOwners = new Map();
            const isGenericExit = (s) => typeof s === 'string' && /^Process completed with exit code 1\.?$/i.test(String(s).trim());
            for (const sn of (item.error_snippets || [])) {
              if (Array.isArray(sn.owner)) {
                for (const o of sn.owner) {
                  if (!o) continue;
                  const k = `${o.id || ''}|${o.name || ''}`;
                  ownerSet.set(k, o);
                }
              }
              if (sn.owner_source && String(sn.owner_source).startsWith('infra_due_to_missing_test') && isGenericExit(sn.snippet) && Array.isArray(sn.original_owners)) {
                for (const oo of sn.original_owners) {
                  const nm = (oo && (oo.name || oo.id)) || '';
                  if (nm) genericExitOrigOwners.set(nm, true);
                }
              }
            }
            let owners = Array.from(ownerSet.values());
            if (!owners.length) {
              owners = findOwnerForLabel(item.name) || [DEFAULT_INFRA_OWNER];
            }
            item.owners = owners;
            item.original_owner_names_for_generic_exit = Array.from(genericExitOrigOwners.keys());
            // Extract failing job names for display
            const failingJobNames = (() => {
              const jobs = new Set();
              for (const sn of (item.error_snippets || [])) {
                const jobName = (sn && sn.job) ? String(sn.job) : '';
                if (jobName) jobs.add(jobName);
              }
              return Array.from(jobs);
            })();
            item.failing_jobs = failingJobNames;
          } catch (_) { /* ignore */ }
          // Omit repeated errors logic (simplified)
          item.repeated_errors = [];
          // Mirror into the corresponding change entry
          const changeRef = changes.find(c => c.name === item.name && c.change === 'success_to_fail');
          if (changeRef) { // update the change entry with the regression information (this is what gets uploaded as an artifact)
            Object.assign(changeRef, {
              first_failed_run_id: item.first_failed_run_id,
              first_failed_run_url: item.first_failed_run_url,
              first_failed_created_at: item.first_failed_created_at,
              first_failed_head_sha: item.first_failed_head_sha,
              first_failed_head_short: item.first_failed_head_short,
              no_success_in_window: item.no_success_in_window,
              first_failed_author_login: item.first_failed_author_login,
              first_failed_author_name: item.first_failed_author_name,
              first_failed_author_url: item.first_failed_author_url,
              commits_between: item.commits_between || [],
              error_snippets: item.error_snippets || [],
              repeated_errors: item.repeated_errors || [],
              failing_jobs: item.failing_jobs || [],
              owners: item.owners || [],
              original_owner_names_for_generic_exit: item.original_owner_names_for_generic_exit || [],
            });
          }
        }
      } catch (e) {
        core.warning(`Failed to find first failing run for ${item.name}: ${e.message}`);
      }
    }

    // Enrich stayed failing with first failing run within the window
    for (const item of stayedFailingDetails) { // basically the same as the regressed details, but for stayed failing
      try {
        const windowRuns = getMainWindowRuns(filteredGrouped.get(item.name) || []);
        const res = findFirstFailInWindow(windowRuns); // get the oldest failing run in the window, that hopefully came after some successes
        if (res && res.run) {
          item.first_failed_run_id = res.run.id;
          item.first_failed_run_url = res.run.html_url;
          item.first_failed_created_at = res.run.created_at;
          item.first_failed_head_sha = res.run.head_sha;
          item.first_failed_head_short = res.run.head_sha ? res.run.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;
          item.no_success_in_window = !!res.noSuccessInWindow;
          // Do not fetch commits/authors for stayed_failing if no success in-window
          if (!item.no_success_in_window && res.boundarySuccessRun && res.boundarySuccessRun.head_sha) {
            item.commits_between = listCommitsBetweenOffline(github.context, res.boundarySuccessRun.head_sha, item.first_failed_head_sha);
          }
          // Commit author of the first failed in-window (optional)
          if (item.first_failed_head_sha) {
            const author = await fetchCommitAuthor(item.first_failed_head_sha);
            item.first_failed_author_login = author.login;
            item.first_failed_author_name = author.name;
            item.first_failed_author_url = author.htmlUrl;
          }
          // Error snippets from the latest failing run (item.run_id already contains the latest failing run)
          // Reuse cached error snippets if available, otherwise fetch
          if (item.run_id) {
            item.error_snippets = errorSnippetsCache.get(item.run_id) || await fetchErrorSnippetsForRun(
              item.run_id,
              20,
              undefined,
              getAnnotationsDirForRunId(item.run_id)
            );
            if (!errorSnippetsCache.has(item.run_id)) {
              errorSnippetsCache.set(item.run_id, item.error_snippets);
            }
          } else {
            item.error_snippets = [];
          }
          // Infer job/test and resolve owners for each snippet
          try {
            for (const sn of (item.error_snippets || [])) {
              const inferred = inferJobAndTestFromSnippet(sn);
              if (inferred) { sn.job = inferred.job; sn.test = inferred.test; }
              resolveOwnersForSnippet(sn, item.name);
            }
          } catch (_) { /* ignore */ }
          // Omit repeated errors (simplified)
          item.repeated_errors = [];
        }
        // Mirror into the corresponding change entry
        const changeRef = changes.find(c => c.name === item.name && c.change === 'stayed_failing');
        if (changeRef) {
          Object.assign(changeRef, {
            first_failed_run_id: item.first_failed_run_id,
            first_failed_run_url: item.first_failed_run_url,
            first_failed_created_at: item.first_failed_created_at,
            first_failed_head_sha: item.first_failed_head_sha,
            first_failed_head_short: item.first_failed_head_short,
            no_success_in_window: item.no_success_in_window,
            first_failed_author_login: item.first_failed_author_login,
            first_failed_author_name: item.first_failed_author_name,
            first_failed_author_url: item.first_failed_author_url,
            commits_between: item.commits_between || [],
            error_snippets: item.error_snippets || [],
            repeated_errors: item.repeated_errors || [],
          });
        }
      }
      catch (e) {
        core.warning(`Failed to find first failing run for ${item.name}: ${e.message}`);
      }
    }

    // upload the changes json to the artifact space
    const outputDir = process.env.GITHUB_WORKSPACE || process.cwd();
    const statusChangesPath = path.join(outputDir, 'workflow-status-changes.json');
    fs.writeFileSync(statusChangesPath, JSON.stringify(changes));
    core.setOutput('status_changes_path', statusChangesPath);

    // Build a minimal regressions section (success -> fail only)
    let regressionsSection = '';
    let stayedFailingSection = '';
    try {
      // Use the already-enriched regressedDetails and stayedFailingDetails arrays directly
      if (regressedDetails.length > 0) { // check if there are any regressed pipelines
        const lines = regressedDetails.map(it => { // for each regressed pipeline, build a markdown line with details
          // Build the workflow name with optional link for the summary (use HTML anchor tag, not markdown)
          const workflowName = it.workflow_url ? `<a href="${it.workflow_url}">${it.name}</a>` : it.name;
          const timeSinceSuccess = getTimeSinceLastSuccess(it.name);
          const timeBadge = timeSinceSuccess !== EMPTY_VALUE ? ` <em>(Last success: ${timeSinceSuccess})</em>` : '';

          if (it.first_failed_run_url) { // if we found the first failing run in the window
            // Extract the short SHA for the first failing commit
            const sha = it.first_failed_head_short || (it.first_failed_head_sha ? it.first_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
            // Create a clickable link to the first failing commit
            const shaLink = sha ? `[\`${sha}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.first_failed_head_sha})` : '';
            // Format the timestamp of the first failure
            const when = it.first_failed_created_at ? new Date(it.first_failed_created_at).toISOString() : '';
            // Build the author attribution (GitHub username with link, or just name)
            const author = it.first_failed_author_login
              ? `by [@${it.first_failed_author_login}](${it.first_failed_author_url})`
              : (it.first_failed_author_name ? `by ${it.first_failed_author_name}` : '');

            // Render error snippets from the latest failing run as a Markdown table
            let errorsList = '';
            const errorsHtml = renderErrorsTable(it.error_snippets || []);
            errorsList = [errorsHtml, ''].join('\n');

            if (it.no_success_in_window) { // if no successful run was found in the 2-week window
              // Build a link to the latest (most recent) failing run with timestamp and commit
              const latestWhenIso = it.created_at ? new Date(it.created_at).toISOString() : '';
              const latestShaShort = it.commit_short || (it.commit_sha ? it.commit_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
              const latestShaLink = (latestShaShort && it.commit_sha)
                ? ` [\`${latestShaShort}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.commit_sha})`
                : '';
              const latestLine = it.run_url
                ? ` | Latest failing run: [Run](${it.run_url}) ${latestWhenIso}${latestShaLink}`
                : '';
              // Return a collapsible workflow with details
              const content = `  - Failed to find any successful run in the last two weeks. Oldest failing run is: [Run](${it.first_failed_run_url}) ${when} ${shaLink}${latestLine}`;
              return ['<details>',`<summary>${workflowName}${timeBadge}</summary>`,'',content, errorsList,'</details>',''].join('\n');
            }

            // If we found a success in the window, show commits between success and first failure
            let commitsList = '';
            const commitsMd = renderCommitsTable(it.commits_between || []);
            commitsList = [commitsMd, ''].join('\n');

            // Build information about the latest failing run (for normal regressions)
            const latestWhenIso = it.created_at ? new Date(it.created_at).toISOString() : ''; // timestamp of latest failure
            const latestShaShort = it.commit_short || (it.commit_sha ? it.commit_sha.substring(0, SHA_SHORT_LENGTH) : undefined); // short SHA of latest failure
            const latestShaLink = (latestShaShort && it.commit_sha)
              ? ` [\`${latestShaShort}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.commit_sha})` // clickable link to latest failing commit
              : '';
            const latestLine = it.run_url
              ? `\n  - Latest failing run: [Run](${it.run_url}) ${latestWhenIso}${latestShaLink}` // link to latest failing run with details
              : '';
            // Return the full collapsible workflow with all details
            const content = `  - First failing run on main: [Run](${it.first_failed_run_url}) ${when} ${shaLink} ${author}${latestLine}`;
            return ['<details>',`<summary>${workflowName}${timeBadge}</summary>`,'',content, errorsList, commitsList,'</details>',''].join('\n');
          }
          // If no first_failed_run_url, just return a collapsed workflow name
          return ['<details>',`<summary>${workflowName}${timeBadge}</summary>`,'','  - No failure details available','</details>',''].join('\n');
        });
        // Build the regressions section with header and all lines
        regressionsSection = ['', '## Regressions (Pass → Fail)', ...lines, ''].join('\n');
      } else {
        // If no regressions, show a "None" message
        regressionsSection = ['','## Regressions (Pass → Fail)','- None',''].join('\n');
      }
      if (stayedFailingDetails.length > 0) { // check if there are any pipelines that stayed failing
        const lines = stayedFailingDetails.map(it => { // for each stayed-failing pipeline, build a markdown line
          // Build the workflow name with optional link for the summary (use HTML anchor tag, not markdown)
          const workflowName = it.workflow_url ? `<a href="${it.workflow_url}">${it.name}</a>` : it.name;
          const timeSinceSuccess = getTimeSinceLastSuccess(it.name);
          const timeBadge = timeSinceSuccess !== EMPTY_VALUE ? ` <em>(Last success: ${timeSinceSuccess})</em>` : '';

          if (it.first_failed_run_url) { // if we found the first failing run in the window
            // Extract the short SHA for the first failing commit
            const sha = it.first_failed_head_short || (it.first_failed_head_sha ? it.first_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
            // Create a clickable link to the first failing commit
            const shaLink = sha ? `[\`${sha}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.first_failed_head_sha})` : '';
            // Format the timestamp of the first failure
            const when = it.first_failed_created_at ? new Date(it.first_failed_created_at).toISOString() : '';

            // Render error snippets from the latest failing run as a Markdown table
            let errorsList = '';
            const errorsHtml2 = renderErrorsTable(it.error_snippets || []);
            errorsList = [errorsHtml2, ''].join('\n');

            if (it.no_success_in_window) { // if no successful run was found in the 2-week window
              // Build information about the latest failing run
              const latestWhenIso = it.created_at ? new Date(it.created_at).toISOString() : ''; // timestamp of latest failure
              const latestShaShort = it.commit_short || (it.commit_sha ? it.commit_sha.substring(0, SHA_SHORT_LENGTH) : undefined); // short SHA of latest failure
              const latestShaLink = (latestShaShort && it.commit_sha)
                ? ` [\`${latestShaShort}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.commit_sha})` // clickable link to latest failing commit
                : '';
              const latestLine = it.run_url
                ? ` | Latest failing run: [Run](${it.run_url}) ${latestWhenIso}${latestShaLink}` // link to latest failing run with details
                : '';
              // Return a collapsible workflow with details
              const content = `  - Failed to find any successful run in the last two weeks. Oldest failing run is: [Run](${it.first_failed_run_url}) ${when} ${shaLink}${latestLine}`;
              return ['<details>',`<summary>${workflowName}${timeBadge}</summary>`,'',content, errorsList,'</details>',''].join('\n');
            }

            // If there is a success boundary in-window, show commits between; otherwise, just show first failure
            let commitsList = '';
            const commitsMd2 = renderCommitsTable(it.commits_between || []);
            commitsList = [commitsMd2, ''].join('\n');

            // Build information about the latest failing run
            const latestWhenIso = it.created_at ? new Date(it.created_at).toISOString() : ''; // timestamp of latest failure
            const latestShaShort = it.commit_short || (it.commit_sha ? it.commit_sha.substring(0, SHA_SHORT_LENGTH) : undefined); // short SHA of latest failure
            const latestShaLink = (latestShaShort && it.commit_sha)
              ? ` [\`${latestShaShort}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.commit_sha})` // clickable link to latest failing commit
              : '';
            const latestLine = it.run_url
              ? `\n  - Latest failing run: [Run](${it.run_url}) ${latestWhenIso}${latestShaLink}` // link to latest failing run with details
              : '';
            // Return the full collapsible workflow with all details
            const content = `  - First failing run on main: [Run](${it.first_failed_run_url}) ${when} ${shaLink}${latestLine}`;
            return ['<details>',`<summary>${workflowName}${timeBadge}</summary>`,'',content, errorsList, commitsList,'</details>',''].join('\n');
          }
          // If no first_failed_run_url, just return a collapsed workflow name
          return ['<details>',`<summary>${workflowName}${timeBadge}</summary>`,'','  - No failure details available','</details>',''].join('\n');
        });
        // Build the stayed-failing section with header and all lines
        stayedFailingSection = ['', '## Still Failing (No Recovery)', ...lines, ''].join('\n');
      } else {
        // If no stayed-failing pipelines, show a "None" message
        stayedFailingSection = ['','## Still Failing (No Recovery)','- None',''].join('\n');
      }
    } catch (_) {
      // Fallback: if any error occurs during rendering, show empty sections with headers
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
    core.setOutput('alert_all_message', alertAllMessage || '');
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
