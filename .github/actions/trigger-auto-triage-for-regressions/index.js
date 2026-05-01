const core = require('@actions/core');
const github = require('@actions/github');
const https = require('https');
const fs = require('fs');

const MAX_JOBS_PER_WORKFLOW = 5;

/**
 * Send a Slack message to a thread
 * @param {string} channelId - Slack channel ID
 * @param {string} botToken - Slack bot token
 * @param {string} message - Message text to send
 * @param {string} threadTs - Thread timestamp to reply to
 * @returns {Promise<void>}
 */
async function sendSlackMessage(channelId, botToken, message, threadTs) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      channel: channelId,
      text: message,
      thread_ts: threadTs
    });

    const options = {
      hostname: 'slack.com',
      port: 443,
      path: '/api/chat.postMessage',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${botToken}`,
        'Content-Length': Buffer.byteLength(payload)
      }
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        try {
          const response = JSON.parse(data);
          if (res.statusCode >= 200 && res.statusCode < 300 && response.ok === true) {
            core.info(`✓ Successfully sent Slack message`);
            resolve();
          } else {
            const errorMsg = response.error || data;
            const error = new Error(`Slack API error: ${res.statusCode} - ${errorMsg}`);
            core.warning(`Failed to send Slack message: ${error.message}`);
            reject(error);
          }
        } catch (parseError) {
          core.warning(`Failed to parse Slack response: ${parseError.message}`);
          reject(parseError);
        }
      });
    });

    req.on('error', (error) => {
      core.warning(`Failed to send Slack message: ${error.message}`);
      reject(error);
    });

    req.write(payload);
    req.end();
  });
}

async function run() {
  try {
    const regressedWorkflowsPath = core.getInput('regressed_workflows_path', { required: true });
    const githubToken = core.getInput('github_token', { required: true });
    const slackTs = core.getInput('slack_ts') || '';
    const slackChannelId = core.getInput('slack_channel_id') || '';
    const slackBotToken = core.getInput('slack_bot_token') || '';
    // Whether downstream auto-triage workflows should send their own Slack message.
    // Note: core.getInput() always returns a string, and GitHub API's createWorkflowDispatch
    // requires all inputs to be strings (even if the target workflow defines them as booleans).
    // The action.yml default is true, so we default to 'true' string here.
    const sendSlackMessageFlag = core.getInput('send-slack-message') || 'true';

    // Per-job (and per-workflow fallback) Slack thread timestamps.  Keys are
    // "<workflow.name> / <job.name>" for per-job top-level messages, or just
    // "<workflow.name>" for regressions that had no failing jobs.  This matches
    // the boundary keys emitted by slack-report-analyze-workflow-data.
    // When no entry is found, the lookup falls back to the workflow-name key,
    // then to the global slackTs.
    const slackTsMapRaw = core.getInput('slack_ts_map') || '{}';
    let slackTsMap = {};
    try {
      const parsedSlackTsMap = JSON.parse(slackTsMapRaw);
      const isPlainObject =
        typeof parsedSlackTsMap === 'object' &&
        parsedSlackTsMap !== null &&
        !Array.isArray(parsedSlackTsMap);

      if (isPlainObject) {
        slackTsMap = parsedSlackTsMap;
        core.info(`Parsed slack_ts_map with ${Object.keys(slackTsMap).length} entries`);
      } else {
        slackTsMap = {};
        core.warning('Invalid slack_ts_map value; expected a non-null plain object. Falling back to global slack_ts.');
      }
    } catch (e) {
      slackTsMap = {};
      core.warning(`Failed to parse slack_ts_map, falling back to global slack_ts: ${e.message}`);
    }

    // Optional global cap on auto-triage dispatches per invocation.
    // Empty/unset = unlimited. Tracks successful createWorkflowDispatch calls
    // (a failed dispatch doesn't count so transient GHA errors don't consume
    // the budget).  Primarily for safe testing via workflow_dispatch.
    const maxDispatchesRaw = (core.getInput('max_dispatches') || '').trim();
    let maxDispatches = null;
    if (maxDispatchesRaw !== '') {
      const parsed = Number.parseInt(maxDispatchesRaw, 10);
      if (!Number.isFinite(parsed) || parsed < 0 || String(parsed) !== maxDispatchesRaw) {
        throw new Error(`Invalid max_dispatches value: "${maxDispatchesRaw}" (expected non-negative integer or empty).`);
      }
      maxDispatches = parsed;
      core.info(`max_dispatches cap: ${maxDispatches}`);
    }
    let dispatchCount = 0;

    // Use the same ref as the workflow that is invoking this action so that
    // auto-triage.yml runs on the same branch (and picks up any in-branch changes),
    // while still defaulting to main when invoked from main.
    const dispatchRef = github.context.ref || 'refs/heads/main';

    if (!fs.existsSync(regressedWorkflowsPath)) {
      throw new Error(`Regressed workflows file not found: ${regressedWorkflowsPath}`);
    }

    const regressedWorkflowsJson = fs.readFileSync(regressedWorkflowsPath, 'utf8');
    const regressedWorkflows = JSON.parse(regressedWorkflowsJson);
    const octokit = github.getOctokit(githubToken);

    core.info(`Found ${regressedWorkflows.length} regressed workflow(s)`);
    if (slackTs) {
      core.info(`Global Slack timestamp (fallback): ${slackTs}`);
    }

    for (const workflow of regressedWorkflows) {
      const workflowPath = workflow.workflow_path || workflow.name;

      // Extract workflow file name (remove .github/workflows/ prefix and .yaml extension)
      const workflowFileName = workflowPath
        .replace(/^\.github\/workflows\//, '')
        .replace(/\.ya?ml$/i, '');

      const allFailingJobs = workflow.failing_jobs || [];

      core.info(`Processing workflow: ${workflowFileName} with ${allFailingJobs.length} failing job(s)`);

      if (allFailingJobs.length === 0) {
        core.warning(`No failing jobs found for workflow: ${workflowFileName}`);

        // Resolve the Slack thread ts — fall back to workflow-level then global.
        const pipelineTs = slackTsMap[workflow.name] || slackTsMap[workflowFileName] || slackTs;

        // Send Slack notification if credentials are available
        if (pipelineTs && slackChannelId && slackBotToken) {
          try {
            const message = `⚠️ Failed to find failing jobs for workflow: \`${workflowFileName}\``;
            await sendSlackMessage(slackChannelId, slackBotToken, message, pipelineTs);
          } catch (error) {
            // Log but don't fail the workflow if Slack message fails
            core.warning(`Failed to send Slack notification: ${error.message}`);
          }
        }
        continue;
      }

      const failingJobs = allFailingJobs.slice(0, MAX_JOBS_PER_WORKFLOW);
      if (allFailingJobs.length > MAX_JOBS_PER_WORKFLOW) {
        core.warning(`Workflow ${workflowFileName} has ${allFailingJobs.length} failing jobs, capping auto-triage to first ${MAX_JOBS_PER_WORKFLOW} to control cost`);
        // Resolve a sensible ts for the cap notification: prefer the first
        // failing job's per-job ts, then the workflow-level keys, then global.
        const firstJobName = (typeof allFailingJobs[0] === 'object' && allFailingJobs[0]?.name) ? allFailingJobs[0].name : String(allFailingJobs[0]);
        const capTs = slackTsMap[`${workflow.name} / ${firstJobName}`] || slackTsMap[workflow.name] || slackTsMap[workflowFileName] || slackTs;
        if (capTs && slackChannelId && slackBotToken) {
          try {
            const skipped = allFailingJobs.length - MAX_JOBS_PER_WORKFLOW;
            const message = `⚠️ \`${workflowFileName}\` has ${allFailingJobs.length} failing jobs. Auto-triage capped to ${MAX_JOBS_PER_WORKFLOW} jobs (${skipped} skipped). This usually indicates a systemic issue or shared root cause.`;
            await sendSlackMessage(slackChannelId, slackBotToken, message, capTs);
          } catch (error) {
            core.warning(`Failed to send Slack cap notification: ${error.message}`);
          }
        }
      }

      if (maxDispatches !== null && dispatchCount >= maxDispatches) {
        core.warning(`Reached max_dispatches=${maxDispatches}; skipping remaining auto-triage dispatches for this run.`);
        break;
      }

      for (const job of failingJobs) {
        if (maxDispatches !== null && dispatchCount >= maxDispatches) {
          core.warning(`Reached max_dispatches=${maxDispatches}; skipping remaining auto-triage dispatches for this run.`);
          break;
        }

        // Handle both old format (string) and new format ({name, url} object)
        const jobName = (typeof job === 'object' && job !== null && job.name) ? job.name : String(job);

        // Resolve the Slack thread ts for this specific job.  Primary key is
        // "<workflow.name> / <jobName>" (the boundary key slack-report emits
        // for per-job top-level messages).  Workflow-level keys are fallbacks
        // for the zero-failing-jobs edge case; global slackTs is last resort.
        const jobTs = slackTsMap[`${workflow.name} / ${jobName}`] || slackTsMap[workflow.name] || slackTsMap[workflowFileName] || slackTs;
        if (jobTs && jobTs !== slackTs) {
          core.info(`Using per-job ts=${jobTs} for ${workflow.name} / ${jobName}`);
        }

        core.info(`Triggering auto-triage for workflow: ${workflowFileName}, job: ${jobName}`);

        try {
          await octokit.rest.actions.createWorkflowDispatch({
            owner: github.context.repo.owner,
            repo: github.context.repo.repo,
            workflow_id: 'auto-triage.yml',
            ref: dispatchRef,
            inputs: {
              workflow_name: workflowFileName,
              job_name: jobName,
              slack_ts: jobTs,
              'send-slack-message': sendSlackMessageFlag,
              slack_channel_id: slackChannelId
            }
          });
          dispatchCount += 1;
          core.info(`✓ Successfully triggered auto-triage for ${workflowFileName} / ${jobName} (dispatch ${dispatchCount}${maxDispatches !== null ? `/${maxDispatches}` : ''})`);

          // Add a small delay to avoid rate limiting
          await new Promise(resolve => setTimeout(resolve, 1000));

        } catch (error) {
          core.error(`Failed to trigger auto-triage for ${workflowFileName} / ${jobName}: ${error.message}`);
          // Continue with other jobs even if one fails
        }
      }
    }

    if (maxDispatches !== null) {
      core.info(`Total auto-triage dispatches this run: ${dispatchCount}/${maxDispatches}`);
    } else {
      core.info(`Total auto-triage dispatches this run: ${dispatchCount}`);
    }

  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
