const core = require('@actions/core');
const axios = require('axios');

/**
 * Sends a message to Slack
 */
async function sendSlackMessage() {
  try {
    // Get inputs
    const webhookUrl = core.getInput('slack-webhook-url', { required: true });
    const messageTitle = core.getInput('message-title', { required: true });
    const messageData = JSON.parse(core.getInput('message-data', { required: true }));

    // Create Slack message
    const message = {
      blocks: [
        {
          type: "header",
          text: {
            type: "plain_text",
            text: messageTitle,
            emoji: true
          }
        }
      ]
    };

    // Handle failed workflows section
    if (messageData.failed_workflows) {
      let failedWorkflowsText;
      try {
        // Try to parse if it's a JSON string
        const parsed = JSON.parse(messageData.failed_workflows);
        failedWorkflowsText = Array.isArray(parsed) ? parsed.join('\n') : parsed;
      } catch (e) {
        // If not JSON, use as is
        failedWorkflowsText = messageData.failed_workflows;
      }

      message.blocks.push({
        type: "section",
        text: {
          type: "mrkdwn",
          text: "*Failed Workflows:*\n" + failedWorkflowsText
        }
      });
    }

    // Add metadata section
    const metadataFields = [];
    if (messageData.repository) {
      metadataFields.push({
        type: "mrkdwn",
        text: `*Repository:*\n${messageData.repository}`
      });
    }
    if (messageData.branch) {
      metadataFields.push({
        type: "mrkdwn",
        text: `*Branch:*\n${messageData.branch}`
      });
    }
    if (messageData.workflow_run) {
      metadataFields.push({
        type: "mrkdwn",
        text: `*Workflow Run:*\n<${messageData.workflow_run}|View Run>`
      });
    }

    if (metadataFields.length > 0) {
      message.blocks.push({
        type: "section",
        fields: metadataFields
      });
    }

    // Send to Slack
    await axios.post(webhookUrl, message);
    core.info('Successfully sent message to Slack');

  } catch (error) {
    core.setFailed(`Failed to send Slack message: ${error.message}`);
  }
}

// Run the action
if (require.main === module) {
  sendSlackMessage();
}

module.exports = { sendSlackMessage };
