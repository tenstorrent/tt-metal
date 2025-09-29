const core = require('@actions/core');
const github = require('@actions/github');

const DEFAULT_OWNER = 'tenstorrent';
const DEFAULT_REPO = 'tt-metal';
const DEFAULT_RUN_ID = 18082047199;

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
    const runId = runIdInput ? Number(runIdInput) : DEFAULT_RUN_ID;

    if (!runId) {
      throw new Error('No run-id available. Provide the run-id input.');
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

    const { data: jobsResponse } = await octokit.rest.actions.listJobsForWorkflowRun({ owner, repo, run_id: runId, per_page: 100 });
    const failingJobs = (jobsResponse.jobs || []).filter(job => isJobFailure(job.conclusion));
    const failingJobIds = new Set(failingJobs.map(job => String(job.id)));
    const failingJobNames = new Set(failingJobs.map(job => (job.name || '').toLowerCase()));
    core.info(`Failing job IDs: ${failingJobIds.size > 0 ? Array.from(failingJobIds).join(', ') : 'none'}`);
    core.info(`Failing job names: ${failingJobNames.size > 0 ? Array.from(failingJobNames).join(' | ') : 'none'}`);

    const { data: checksResponse } = await octokit.rest.checks.listForRef({ owner, repo, ref: headSha, per_page: 100 });
    const checkRuns = Array.isArray(checksResponse.check_runs) ? checksResponse.check_runs : [];

    if (checkRuns.length === 0) {
      core.info('No check runs found for this commit.');
      core.setOutput('annotations', '[]');
      return;
    }

    const checkFilters = checkName
      ? checkName.split(',').map(s => s.trim()).filter(Boolean)
      : [];

    let filteredRuns = checkRuns;
    if (failingJobNames.size > 0) {
      const before = filteredRuns.length;
      filteredRuns = filteredRuns.filter(run => failingJobNames.has((run.name || '').toLowerCase()));
      core.info(`Matched ${filteredRuns.length} check run(s) out of ${before} using failing job names.`);
    }
    if (filteredRuns.length === 0 && failingJobIds.size > 0) {
      // Some old runs may only expose job IDs in URLs; attempt a secondary match.
      filteredRuns = checkRuns.filter(run => {
        const jobId = extractJobId(run.details_url || run.html_url);
        return jobId && failingJobIds.has(jobId);
      });
      core.info(`Matched ${filteredRuns.length} check run(s) after secondary job-id filter.`);
    }
    if (checkFilters.length > 0) {
      filteredRuns = filteredRuns.filter(run => checkFilters.some(name => (run.name || '').toLowerCase().includes(name.toLowerCase())));
    }

    if (filteredRuns.length === 0) {
      core.info('No matching check runs found after applying failure/job filters. Falling back to all check runs for this run.');
      filteredRuns = checkRuns;
    }

    const collected = [];

    for (const checkRun of filteredRuns) {
      core.startGroup(`Annotations for ${checkRun.name} (id: ${checkRun.id})`);
      const annotationsRaw = await fetchAnnotations(octokit, owner, repo, checkRun.id, maxAnnotations);
      const annotations = annotationsRaw
        .filter(ann => (ann.annotation_level || '').toLowerCase() === 'failure')
        .map(sanitizeAnnotation)
        .filter(ann => ann !== null);
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
      core.info(`Collected ${annotations.length} annotation(s) for ${checkRun.name}.`);

      collected.push({
        checkRun: checkRun.name,
        htmlUrl: checkRun.html_url,
        annotations,
      });

      if (collected.flatMap(entry => entry.annotations).length >= maxAnnotations) {
        core.info(`Reached max-annotations limit (${maxAnnotations}).`);
        break;
      }
    }

    const totalAnnotations = collected.flatMap(entry => entry.annotations).length;
    core.setOutput('annotations', JSON.stringify(collected));
    core.info(`Total annotations collected: ${totalAnnotations}`);

    if (totalAnnotations > 0) {
      const markdown = buildAnnotationsMarkdown(collected);
      core.setOutput('annotations_markdown', markdown);
      await core.summary.addRaw(markdown, true).write();
    }
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

async function resolveRunId(token, owner, repo, workflowName, fallbackRunId) {
  return fallbackRunId;
}

function buildAnnotationsMarkdown(collected) {
  const sections = ['# Galaxy Fabric Test Annotations'];
  for (const entry of collected) {
    sections.push('');
    sections.push(`## ${entry.checkRun}`);
    if (entry.htmlUrl) {
      sections.push(`Run: [View check run](${entry.htmlUrl})`);
    }
    if (!entry.annotations || entry.annotations.length === 0) {
      sections.push('- No annotations found.');
      continue;
    }
    for (const ann of entry.annotations) {
      const location = ann.path
        ? `${ann.path}${ann.start_line ? `:${ann.start_line}${ann.end_line && ann.end_line !== ann.start_line ? `-${ann.end_line}` : ''}` : ''}`
        : 'no-path';
      const level = (ann.annotation_level || 'info').toUpperCase();
      const title = ann.title || '(no title)';
      sections.push(`- **[${level}]** ${title}`);
      sections.push(`  - Location: ${location}`);
      if (ann.message) {
        sections.push(`  - Message:\n${formatMultiline(ann.message, '    ')}`);
      }
      if (ann.raw_details) {
        sections.push(`  - Details:\n${formatMultiline(ann.raw_details, '    ')}`);
      }
    }
  }
  return sections.join('\n');
}

function formatMultiline(text, indent = '') {
  if (!text) return '';
  return text
    .split(/\r?\n/)
    .map(line => `${indent}${line}`)
    .join('\n');
}

function sanitizeAnnotation(ann) {
  if (!ann) return null;
  const message = stripBacktrace(ann.message);
  const details = stripBacktrace(ann.raw_details);
  if (!message && !details) {
    // If both message and details are empty after sanitizing, drop the annotation.
    return {
      ...ann,
      message: undefined,
      raw_details: undefined,
    };
  }
  return {
    ...ann,
    message: message || undefined,
    raw_details: details || undefined,
  };
}

function stripBacktrace(text) {
  if (!text) return '';
  const lower = text.toLowerCase();
  const idx = lower.indexOf('backtrace:');
  const truncated = idx !== -1 ? text.slice(0, idx) : text;
  return truncated.trim();
}

function extractJobId(url) {
  if (!url) return null;
  const match = url.match(/\/jobs\/(\d+)/);
  return match ? match[1] : null;
}

function isJobFailure(conclusion) {
  if (!conclusion) return false;
  const lowered = conclusion.toLowerCase();
  return ['failure', 'cancelled', 'timed_out', 'action_required'].includes(lowered);
}

if (require.main === module) {
  run();
}

module.exports = { run };
