const core = require('@actions/core');
const github = require('@actions/github');

async function run() {
  try {
    const regressedWorkflowsJson = core.getInput('regressed_workflows', { required: true });
    const githubToken = core.getInput('github_token', { required: true });
    const slackTs = core.getInput('slack_ts') || '';

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
        continue;
      }

      for (const jobName of failingJobs) {
        core.info(`Triggering auto-triage for workflow: ${workflowFileName}, job: ${jobName}`);

        try {
          await octokit.rest.actions.createWorkflowDispatch({
            owner: github.context.repo.owner,
            repo: github.context.repo.repo,
            workflow_id: 'auto-triage.yml',
            ref: 'main',
            inputs: {
              workflow_name: workflowFileName,
              job_name: jobName,
              slack_ts: slackTs
            }
          });

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
