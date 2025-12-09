module.exports = async ({ github, context }) => {
  const targetWorkflow = 'All post-commit tests';
  const overrideCmd    = '/override';
  const runWorkflowCmd = '/run-it';

  console.log(`🤖 Post-commit nag bot triggered by: ${context.eventName}`);

  /* -------- helpers -------- */

  // locate (or create) the single bot comment for this PR
  async function upsertComment(prNumber, body) {
    console.log(`📝 Updating comment for PR #${prNumber}`);
    const comments = await github.rest.issues.listComments({
      owner: context.repo.owner,
      repo:  context.repo.repo,
      issue_number: prNumber
    });

    const existing = comments.data.find(
      c => c.user.login === 'github-actions[bot]' &&
           (c.body.startsWith('**⛔️') || c.body.startsWith('**✅'))
    );

    if (existing) {
      console.log(`🗑️ Deleting old comment (ID: ${existing.id}) to move to bottom`);
      await github.rest.issues.deleteComment({
        owner: context.repo.owner,
        repo:  context.repo.repo,
        comment_id: existing.id
      });

      console.log(`📌 Creating new comment at bottom`);
      await github.rest.issues.createComment({
        owner: context.repo.owner,
        repo:  context.repo.repo,
        issue_number: prNumber,
        body
      });
    } else {
      console.log(`📌 Creating new comment`);
      await github.rest.issues.createComment({
        owner: context.repo.owner,
        repo:  context.repo.repo,
        issue_number: prNumber,
        body
      });
    }
  }

  // has the target workflow run successfully on the HEAD commit?
  async function workflowGood(pr) {
    const runs = await github.rest.actions.listWorkflowRunsForRepo({
      owner: context.repo.owner,
      repo:  context.repo.repo,
      branch: pr.head.ref,
      status: 'success'
    });
    return runs.data.workflow_runs.some(r =>
      r.name === targetWorkflow && r.head_sha === pr.head.sha);
  }

  /* -------- router -------- */

  let prNumber, forceGreen = false;

  if (context.eventName === 'pull_request') {
    prNumber = context.payload.pull_request.number;
    console.log(`🔍 Processing PR #${prNumber} event: ${context.payload.action}`);

  } else if (context.eventName === 'workflow_run') {
    const run = context.payload.workflow_run;
    console.log(`🏃 Processing workflow run: ${run.name} (${run.conclusion})`);
    if (run.name !== targetWorkflow || run.conclusion !== 'success') {
      console.log(`⏭️ Skipping - not a successful ${targetWorkflow} run`);
      return;
    }

    const prs = await github.rest.pulls.list({
      owner: context.repo.owner,
      repo:  context.repo.repo,
      head:  `${context.repo.owner}:${run.head_branch}`
    });
    if (!prs.data.length) {
      console.log(`⏭️ No open PRs found for branch: ${run.head_branch}`);
      return;
    }
    prNumber = prs.data[0].number;
    console.log(`✅ Found PR #${prNumber} for workflow run`);

  } else if (context.eventName === 'issue_comment') {
    if (!context.payload.issue.pull_request) {
      console.log(`⏭️ Comment is not on a PR, ignoring`);
      return;
    }

    const commentBody = context.payload.comment.body;
    console.log(`💬 Processing comment by ${context.payload.comment.user.login}`);

    if (commentBody.includes(overrideCmd)) {
      console.log(`🟢 Override command detected!`);
      prNumber   = context.payload.issue.number;
      forceGreen = true;
    } else if (commentBody.includes(runWorkflowCmd)) {
      console.log(`🚀 Run workflow command detected!`);
      prNumber = context.payload.issue.number;

      const { data: pr } = await github.rest.pulls.get({
        owner: context.repo.owner,
        repo:  context.repo.repo,
        pull_number: prNumber
      });

      try {
        console.log(`🎯 Dispatching workflow for PR #${prNumber} on ref: ${pr.head.ref}`);

        // Get the workflow ID first
        const workflowInfo = await github.rest.actions.getWorkflow({
          owner: context.repo.owner,
          repo: context.repo.repo,
          workflow_id: 'all-post-commit-workflows.yaml'
        });
        const workflowId = workflowInfo.data.id;
        console.log(`📋 Workflow ID: ${workflowId}`);

        // Dispatch the workflow
        await github.rest.actions.createWorkflowDispatch({
          owner: context.repo.owner,
          repo: context.repo.repo,
          workflow_id: 'all-post-commit-workflows.yaml',
          ref: pr.head.ref
        });

        console.log(`✅ Workflow dispatched successfully, polling for run URL...`);

        // Poll for the new run (for up to 15 seconds)
        let workflowRunUrl = null;
        const maxAttempts = 15;

        for (let i = 0; i < maxAttempts; i++) {
          // Wait a bit before checking (first attempt is immediate)
          if (i > 0) {
            await new Promise(resolve => setTimeout(resolve, 1000));
          }

          const runs = await github.rest.actions.listWorkflowRunsForRepo({
            owner: context.repo.owner,
            repo: context.repo.repo,
            branch: pr.head.ref,
            per_page: 10
          });

          // Find a run that was created in the last 30 seconds
          const now = new Date();
          for (const run of runs.data.workflow_runs) {
            if (run.workflow_id === workflowId) {
              const createdAt = new Date(run.created_at);
              const secondsAgo = (now - createdAt) / 1000;

              if (secondsAgo < 30) {
                workflowRunUrl = run.html_url;
                console.log(`🎯 Found workflow run: ${workflowRunUrl}`);
                break;
              }
            }
          }

          if (workflowRunUrl) break;
        }

        // Create comment with direct link or fallback to filtered page
        if (workflowRunUrl) {
          await github.rest.issues.createComment({
            owner: context.repo.owner,
            repo: context.repo.repo,
            issue_number: prNumber,
            body: `🔄 Running \`${targetWorkflow}\` on the latest commit (${pr.head.sha.substring(0, 7)}). [View run](${workflowRunUrl})`
          });
        } else {
          // Fallback to filtered page if we couldn't find the run
          const workflowRunLink = `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/workflows/all-post-commit-workflows.yaml?query=branch%3A${encodeURIComponent(pr.head.ref)}`;
          await github.rest.issues.createComment({
            owner: context.repo.owner,
            repo: context.repo.repo,
            issue_number: prNumber,
            body: `🔄 Running \`${targetWorkflow}\` on the latest commit (${pr.head.sha.substring(0, 7)}). [View progress](${workflowRunLink})`
          });
        }
      } catch (dispatchError) {
        console.error("❌ Error dispatching workflow:", dispatchError);
        await github.rest.issues.createComment({
            owner: context.repo.owner,
            repo: context.repo.repo,
            issue_number: prNumber,
            body: `⚠️ Failed to trigger workflow \`${targetWorkflow}\`. Ensure the workflow file exists and the GITHUB_TOKEN has \`actions:write\` permission. Error: ${dispatchError.message}`
          });
      }
      return; // Exit after triggering the workflow
    } else {
      console.log(`⏭️ Ignoring comment - no recognized commands`);
      return; // Ignore other comments
    }

  } else {
    console.log(`⏭️ Ignoring event: ${context.eventName}`);
    return;   // ignore other events
  }

  /* -------- main logic -------- */

  console.log(`🔎 Checking PR #${prNumber} status...`);
  const { data: pr } = await github.rest.pulls.get({
    owner: context.repo.owner,
    repo:  context.repo.repo,
    pull_number: prNumber
  });

  const happy = `**✅🎉 Thanks for running \`${targetWorkflow}\`! You're clear to merge. 🎉✅**`;
  const angry = `**⛔️🚨 \`${targetWorkflow}\` has NOT run on the latest commit. Run it before merging! 🚨⛔️**\n - /run-it to run the workflow on the latest commit.\n - /override to ignore the workflow check.`;

  if (forceGreen) {
    console.log(`✅ Force green via override command`);
    await upsertComment(prNumber, happy);
  } else {
    const isGood = await workflowGood(pr);
    console.log(`🎯 Workflow status check: ${isGood ? 'PASSED' : 'FAILED'} for commit ${pr.head.sha.substring(0, 7)}`);
    const body = isGood ? happy : angry;
    await upsertComment(prNumber, body);
  }
};
