const core = require('@actions/core');
const github = require('@actions/github');

const DEFAULT_OWNER = 'tenstorrent';
const DEFAULT_REPO = 'tt-metal';
const DEFAULT_RUN_ID = 18101502836;
const DEFAULT_CHECK_NAMES = ['Galaxy Fabric tests'];

async function run() {
  try {
    const token = core.getInput('github-token') || core.getInput('GITHUB_TOKEN');
    if (!token) {
      throw new Error('Missing github-token/GITHUB_TOKEN input');
    }
    const ownerInput = core.getInput('owner');
    const repoInput = core.getInput('repo');
    const runIdInput = core.getInput('run-id');
    const checkName = core.getInput('check-name');
    const maxAnnotations = parseInt(core.getInput('max-annotations') || '50', 10);

    const owner = ownerInput || github.context.repo?.owner || DEFAULT_OWNER;
    const repo = repoInput || github.context.repo?.repo || DEFAULT_REPO;
    const runId = runIdInput ? Number(runIdInput) : (github.context.runId || DEFAULT_RUN_ID);

    if (!runId) {
      throw new Error('No run-id available. Provide the run-id input or run within a workflow context.');
    }

    const octokit = github.getOctokit(token);

    core.startGroup(`Fetching workflow run ${owner}/${repo}#${runId}`);
    const { data: workflowRun } = await octokit.rest.actions.getWorkflowRun({ owner, repo, run_id: runId });
    const headSha = workflowRun?.head_sha;
    core.info(`Workflow name: ${workflowRun?.name || 'unknown'}`);
    core.info(`Status: ${workflowRun?.status} | Conclusion: ${workflowRun?.conclusion}`);
    core.info(`Head SHA: ${headSha || 'n/a'}`);
    core.endGroup();

    if (!headSha) {
      throw new Error('Workflow run does not expose head_sha; cannot list check runs.');
    }

    const { data: checksResponse } = await octokit.rest.checks.listForRef({ owner, repo, ref: headSha, per_page: 100 });
    const checkRuns = Array.isArray(checksResponse.check_runs) ? checksResponse.check_runs : [];

    if (checkRuns.length === 0) {
      core.info('No check runs found for this commit.');
      core.setOutput('annotations', '[]');
      return;
    }

    const defaultChecks = checkName ? [checkName] : DEFAULT_CHECK_NAMES;
    const filteredRuns = checkRuns.filter(run => defaultChecks.some(name => (run.name || '').toLowerCase().includes(name.toLowerCase())));

    if (filteredRuns.length === 0) {
      core.info(`No check runs matched filter(s) "${defaultChecks.join(', ')}". Available: ${checkRuns.map(r => r.name).join(', ')}`);
      core.setOutput('annotations', '[]');
      return;
    }

    const collected = [];

    for (const checkRun of filteredRuns) {
      core.startGroup(`Annotations for ${checkRun.name} (id: ${checkRun.id})`);
      const annotations = await fetchAnnotations(octokit, owner, repo, checkRun.id, maxAnnotations);
      if (annotations.length === 0) {
        core.info('No annotations returned.');
      } else {
        annotations.forEach((ann, idx) => {
          const location = ann.path
            ? `${ann.path}${ann.start_line ? `:${ann.start_line}${ann.end_line && ann.end_line !== ann.start_line ? `-${ann.end_line}` : ''}` : ''}`
            : 'no-path';
          core.info(`${idx + 1}. [${ann.annotation_level}] ${location} :: ${ann.title || 'no title'}`);
          if (ann.message) core.info(`   ${ann.message}`);
        });
      }
      core.endGroup();

      collected.push({
        checkRun: checkRun.name,
        annotations,
      });

      if (collected.flatMap(entry => entry.annotations).length >= maxAnnotations) {
        core.info(`Reached max-annotations limit (${maxAnnotations}).`);
        break;
      }
    }

    core.setOutput('annotations', JSON.stringify(collected));
  } catch (error) {
    core.setFailed(error.message);
  }
}

async function fetchAnnotations(octokit, owner, repo, checkRunId, budget) {
  const collected = [];
  let page = 1;
  const max = Math.max(budget || 1, 1);

  while (collected.length < max) {
    const perPage = Math.min(100, max - collected.length);
    const { data } = await octokit.rest.checks.listAnnotations({
      owner,
      repo,
      check_run_id: checkRunId,
      per_page: perPage,
      page,
    });

    if (!Array.isArray(data) || data.length === 0) break;

    collected.push(...data);
    if (data.length < perPage) break;
    page += 1;
  }

  return collected.slice(0, max);
}

if (require.main === module) {
  run();
}

module.exports = { run };
