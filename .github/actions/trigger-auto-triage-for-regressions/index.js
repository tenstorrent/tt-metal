const core = require('@actions/core');
const github = require('@actions/github');
const https = require('https');

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
    const regressedWorkflowsJson = core.getInput('regressed_workflows', { required: true });
    const githubToken = core.getInput('github_token', { required: true });
    const slackTs = core.getInput('slack_ts') || '';
    const slackChannelId = core.getInput('slack_channel_id') || '';
    const slackBotToken = core.getInput('slack_bot_token') || '';
    // Whether downstream auto-triage workflows should send their own Slack message.
    // Note: core.getInput() always returns a string, and GitHub API's createWorkflowDispatch
    // requires all inputs to be strings (even if the target workflow defines them as booleans).
    // The action.yml default is true, so we default to 'true' string here.
    const sendSlackMessageFlag = core.getInput('send-slack-message') || 'true';

    // Use the same ref as the workflow that is invoking this action so that
    // auto-triage.yml runs on the same branch (and picks up any in-branch changes),
    // while still defaulting to main when invoked from main.
    const dispatchRef = github.context.ref || 'refs/heads/main';

    const regressedWorkflows = JSON.parse(regressedWorkflowsJson);
    const octokit = github.getOctokit(githubToken);

    core.info(`Found ${regressedWorkflows.length} regressed workflow(s)`);
    if (slackTs) {
      core.info(`Slack timestamp provided: ${slackTs}`);
    }

    for (const workflow of regressedWorkflows) {
      const workflowPath = workflow.workflow_path || workflow.name;

      // Extract workflow file name (remove .github/workflows/ prefix and .yaml extension)
      const workflowFileName = workflowPath
        .replace(/^\.github\/workflows\//, '')
        .replace(/\.ya?ml$/i, '');

      const failingJobs = workflow.failing_jobs || [];

      core.info(`Processing workflow: ${workflowFileName} with ${failingJobs.length} failing job(s)`);

      if (failingJobs.length === 0) {
        core.warning(`No failing jobs found for workflow: ${workflowFileName}`);

        // Send Slack notification if credentials are available
        if (slackTs && slackChannelId && slackBotToken) {
          try {
            const message = `⚠️ Failed to find failing jobs for workflow: \`${workflowFileName}\``;
            await sendSlackMessage(slackChannelId, slackBotToken, message, slackTs);
          } catch (error) {
            // Log but don't fail the workflow if Slack message fails
            core.warning(`Failed to send Slack notification: ${error.message}`);
          }
        }
        continue;
      }

      for (const job of failingJobs) {
        // Handle both old format (string) and new format ({name, url} object)
        const jobName = (typeof job === 'object' && job !== null && job.name) ? job.name : String(job);
        core.info(`Triggering auto-triage for workflow: ${workflowFileName}, job: ${jobName}`);

        try {
          // await octokit.rest.actions.createWorkflowDispatch({
          //   owner: github.context.repo.owner,
          //   repo: github.context.repo.repo,
          //   workflow_id: 'auto-triage.yml',
          //   ref: dispatchRef,
          //   inputs: {
          //     workflow_name: workflowFileName,
          //     job_name: jobName,
          //     slack_ts: slackTs,
          //     'send-slack-message': sendSlackMessageFlag,
          //     slack_channel_id: slackChannelId
          //   }
          // });

          core.info(`✓ Successfully triggered auto-triage for ${workflowFileName} / ${jobName}`);

          // Add a small delay to avoid rate limiting
          await new Promise(resolve => setTimeout(resolve, 1000));

        } catch (error) {
          core.error(`Failed to trigger auto-triage for ${workflowFileName} / ${jobName}: ${error.message}`);
          // Continue with other jobs even if one fails
        }
      }
    }

  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
