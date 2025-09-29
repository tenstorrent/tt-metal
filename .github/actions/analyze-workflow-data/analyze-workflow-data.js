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
const os = require('os'); // Operating system utilities
const { execFileSync } = require('child_process'); // Used to run external commands

// Constants
const DEFAULT_LOOKBACK_DAYS = 15;
const SHA_SHORT_LENGTH = 7;
const SUCCESS_RATE_DECIMAL_PLACES = 2;
const SUCCESS_EMOJI = '✅';
const FAILURE_EMOJI = '❌';
const EMPTY_VALUE = '—';

// CODE FOR OWNERSHIP MAPPING

// Owners mapping cache
let __ownersMapping = undefined; // Stores the owners data
function loadOwnersMapping() {
  if (__ownersMapping !== undefined) return __ownersMapping; // If the owners mapping is already loaded, return it
  try {
    const ownersPath = path.join(__dirname, 'owners.json'); // Path to the owners.json file
    if (fs.existsSync(ownersPath)) {
      const raw = fs.readFileSync(ownersPath, 'utf8'); // Read the owners.json file in some raw format
      __ownersMapping = JSON.parse(raw); // Parse the raw data into a JSON object
    } else {
      __ownersMapping = null; // If the owners.json file does not exist, set the owners mapping to null
    }
  } catch (_) {
    __ownersMapping = null; // If there is an error, set the owners mapping to null
  }
  return __ownersMapping;
}

// Normalizes the owner information to a consistent format for easier processing later
// Example: ["charlie", { id: "david", name: "David" }] → [{ id: "charlie" }, { id: "david", name: "David" }]
function normalizeOwners(value) {
  if (!value) return undefined;
  // Single string -> id-only
  if (typeof value === 'string') return [{ id: value }];
  // Object with id/name
  if (typeof value === 'object' && !Array.isArray(value)) {
    const id = value.id || undefined;
    const name = value.name || undefined;
    if (!id && !name) return undefined;
    return [{ id, name }];
  }
  // Array -> map each entry
  if (Array.isArray(value)) {
    const arr = [];
    for (const entry of value) {
      if (typeof entry === 'string') arr.push({ id: entry });
      else if (entry && typeof entry === 'object') arr.push({ id: entry.id, name: entry.name });
    }
    return arr.length ? arr : undefined;
  }
  return undefined;
}

function extractSignificantTokens(str) {
  if (typeof str !== 'string') return [];
  return str
    .split(/\s+[^A-Za-z0-9\s]+\s+/) // Splits the string where there's whitespace plus punctuation plus whitespace
                                    // For example: "test::function_name" → ["test", "function_name"]

    .flatMap(segment => segment.split(/[\r\n]+/)) // Splits the string where there's a newline
                                                // For example: "test\nfunction_name" → ["test", "function_name"]

    .map(segment => segment.trim()) // Trims the string
                                    // For example: "   test" → "test"

    .filter(Boolean) // Removes empty values

    .map(segment => segment.replace(/^[^A-Za-z0-9_\-\s]+|[^A-Za-z0-9_\-\s]+$/g, '').trim()) // Removes non-alphanumeric characters from the beginning and end of each segment
                                                                                            // Keeps letters, numbers, underscores, hyphens, and spaces

    .filter(segment => segment.replace(/\s+/g, '').length > 1); // Removes segments that are 1 character or less without spaces
}

function getNeedleTail(needle) {
  const tokens = extractSignificantTokens(needle);
  return tokens.length ? tokens[tokens.length - 1] : undefined; // Returns the last token in the array
}

function findOwnerForLabel(label) {
  try {
    const mapping = loadOwnersMapping(); // Load mapping
    if (!mapping) return undefined; // If the mapping is not loaded, return undefined
    const lbl = typeof label === 'string' ? label : ''; // If the label is not a string, return an empty string
    const labelTokens = extractSignificantTokens(lbl); // Extract significant tokens from the label

    if (Array.isArray(mapping.contains)) { // If the mapping's contains parameter is an array
      for (const entry of mapping.contains) {
        if (!entry || typeof entry.needle !== 'string') continue; // If the entry is not an object or the needle is not a string, continue
        const needle = entry.needle;
        if (lbl.includes(needle)) { // If the label includes the needle
          return normalizeOwners(entry.owner); // Return the normalized owners
        }
        // Fuzzy match: try last token from needle
        const tail = getNeedleTail(needle); // Get the last token from the needle
        if (tail && labelTokens.includes(tail)) { // If the tail is not empty and the label tokens include the tail
          return normalizeOwners(entry.owner);
        }
        // Additional heuristic: if label tokens end with the last two tokens of the needle
        const needleTokens = extractSignificantTokens(needle);
        if (needleTokens.length >= 2 && labelTokens.length >= 2) { // If the needle tokens are at least 2 and the label tokens are at least 2
          const needleTailPair = needleTokens.slice(-2).join(' '); // Get the last two tokens from the needle
          const labelTailPair = labelTokens.slice(-2).join(' '); // Get the last two tokens from the label
          if (needleTailPair === labelTailPair) { // If the needle tail pair is the same as the label tail pair
            return normalizeOwners(entry.owner); // Basically, if the last two pieces of the label and the needle are the same, define a match. This heuristic may be flawed
          }
        }
      }
    }
  } catch (_) {
    // ignore
  }
  return undefined;
}

// END OF CODE FOR OWNERSHIP MAPPING

// Simple HTML escaping for rendering snippets safely in summary HTML
// This is used to prevent XSS attacks by converting special characters to basic text that can't be rendered as HTML
function escapeHtml(text) {
  if (typeof text !== 'string') return text;
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function renderErrorsTable(errorSnippets) {
  if (!Array.isArray(errorSnippets) || errorSnippets.length === 0) { // If the error snippets are not an array or the length is 0
    return '<em>No error info found</em>';
  }
  const rows = errorSnippets.map(obj => { // Map each object in the error snippets array to something else
    const label = escapeHtml(obj.label || ''); // Escape the label so no security issues
    const snippet = escapeHtml(obj.snippet || ''); // Escape the snippet so no security issues
    // Render owner display name(s) if present; fallback to id(s); else 'no owner found'
    let ownerDisplay = 'no owner found';
    if (obj.owner && Array.isArray(obj.owner) && obj.owner.length) {
      const names = obj.owner.map(o => (o && (o.name || o.id)) || '').filter(Boolean); // if the owner exists and has a name or and id that isn't falsy then make that the new owner value in the mapping
      if (names.length) ownerDisplay = names.join(', '); // If the names array is not empty, join the names with a comma
    }
    const owner = escapeHtml(ownerDisplay);
    return `<tr><td style="vertical-align:top;"><pre style="white-space:pre-wrap;word-break:break-word;margin:0;">${label}</pre></td><td>${owner}</td><td><pre style="white-space:pre-wrap;margin:0;">${snippet}</pre></td></tr>`; // Return the table row. this is formatted so that word wrapping occurs
  }).join('\n');
  return `<table><thead><tr><th style="text-align:left;">Test</th><th style="text-align:left;">Owner</th><th style="text-align:left;">Error</th></tr></thead><tbody>${rows}</tbody></table>`; // Return the table
}

function renderCommitsTable(commits) {
  if (!Array.isArray(commits) || commits.length === 0) {
    return '<em>None</em>';
  }
  const rows = commits.map(c => {
    const short = escapeHtml(c.short || (c.sha ? c.sha.substring(0, 7) : '')); // Return the short SHA of the commit using c.short if available. otherwise use the first 7 characters of the SHA
    const url = c.url || (c.sha ? `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${c.sha}` : undefined); // Return the URL of the commit or build it from the sha if the url is not available
    const who = c.author_login ? `@${escapeHtml(c.author_login)}` : escapeHtml(c.author_name || 'unknown'); // return the author login name if available, otherwise use their display name, but default to unknown if nothing is available
    const whoHtml = c.author_login && c.author_url ? `<a href="${c.author_url}">${who}</a>` : who; // if author login and url are available, make the author login name a clickable link
    const shaHtml = url ? `<a href="${url}"><code>${short}</code></a>` : `<code>${short}</code>`; // if the sha url is available, make the sha a clickable link
    return `<tr><td>${shaHtml}</td><td>${whoHtml}</td></tr>`;
  }).join('\n');
  return `<table><thead><tr><th>SHA</th><th>Author</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/**
 * Fetches PR information associated with a commit.
 *
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @param {string} commitSha - Full SHA of the commit to look up
 * @returns {Promise<object>} Object containing:
 *   - prNumber: Markdown link to the PR (e.g., [#123](url))
 *   - prTitle: Title of the PR or EMPTY_VALUE if not found
 *   - prAuthor: GitHub username of the PR author or 'unknown'
 */
async function fetchPRInfo(github, context, commitSha) {
  try {
    const { data: prs } = await github.rest.repos.listPullRequestsAssociatedWithCommit({
      owner: context.repo.owner,
      repo: context.repo.repo,
      commit_sha: commitSha,
    }); // listPullRequestsAssociatedWithCommit is a GitHub API call to get the PRs associated with a commit
    if (prs.length > 0) {
      const pr = prs[0]; // get the first PR (usually the only one)
      return { // return the PR number, title, and author
        prNumber: `[#${pr.number}](https://github.com/${context.repo.owner}/${context.repo.repo}/pull/${pr.number})`, // make a link to the PR attached to the commit number
        prTitle: pr.title || EMPTY_VALUE, // return the PR title or EMPTY_VALUE if the PR title is not available
        prAuthor: pr.user?.login || 'unknown' // return the PR author or 'unknown' if the PR author is not available
      };
    }
  } catch (e) {
    core.warning(`Could not fetch PR for commit ${commitSha}: ${e.message}`); // if there is an error, log it
  }
  return { prNumber: EMPTY_VALUE, prTitle: EMPTY_VALUE, prAuthor: EMPTY_VALUE }; //return nothing if there is no PR
}

/**
 * Fetch commit author info for a commit SHA.
 * Returns GitHub login (if associated), author display name, and profile URL (if available).
 */
async function fetchCommitAuthor(octokit, context, commitSha) {
  try {
    const { data } = await octokit.rest.repos.getCommit({ // getCommit is a GitHub API call to get the commit associated with a commit SHA
      owner: context.repo.owner,
      repo: context.repo.repo,
      ref: commitSha,
    });
    const login = data.author?.login; // return the author login name if available
    const htmlUrl = data.author?.html_url; // return the author profile URL if available
    const name = data.commit?.author?.name; // return the author display name if available
    return { login, name, htmlUrl };
  } catch (e) {
    core.warning(`Could not fetch commit author for ${commitSha}: ${e.message}`); // if there is an error, log it
    return { login: undefined, name: undefined, htmlUrl: undefined }; // return nothing if there is no author
  }
}

/**
 * Download workflow run logs and extract up to N error snippets.
 * Returns an array of strings (snippets).
 */
async function fetchErrorSnippetsForRun(octokit, context, runId, maxSnippets = 50) {
  const owner = context.repo.owner; // return the owner of the repository
  const repo = context.repo.repo; // return the repository name
  try {
    await core.startGroup(`Extracting error snippets for run ${runId}`); // start a group to log the error snippets (we log this to the console)
    const { data } = await octokit.rest.actions.downloadWorkflowRunLogs({ owner, repo, run_id: runId }); // downloadWorkflowRunLogs is a GitHub API call to download the logs for a run
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), `runlogs-${runId}-`)); // create a temporary directory to store the logs
    const zipPath = path.join(tmpDir, 'run_logs.zip'); // create a path to the zip file
    fs.writeFileSync(zipPath, Buffer.from(data)); // write the logs to the zip file in the temporary directory
    const extractDir = path.join(tmpDir, 'extract'); // create a path to the extract directory
    fs.mkdirSync(extractDir, { recursive: true }); // create the extract directory
    execFileSync('unzip', ['-o', zipPath, '-d', extractDir], { stdio: 'ignore' }); // unzip the zip file to the extract directory. ignore the child process' IO
    let snippets = findErrorSnippetsInDir(extractDir, maxSnippets); // find the error snippets in the extract directory
    core.info(`Total snippets collected: ${snippets.length}`); // log the total number of snippets collected

    // Query job/step status once to validate findings and/or provide fallback
    let hasFailingJob = false;
    let failingLabel = 'no failing job detected';
    let apiCheckSucceeded = false;
    try {
      const { data } = await octokit.rest.actions.listJobsForWorkflowRun({ owner, repo, run_id: runId }); // listJobsForWorkflowRun is a GitHub API call to list the jobs for a workflow run
      const jobs = Array.isArray(data.jobs) ? data.jobs : []; // return the jobs array if it is an array, otherwise return an empty array
      let failingJob = jobs.find(j => j.conclusion && j.conclusion !== 'success' && j.conclusion !== 'skipped' && j.conclusion !== 'cancelled'); // find the failing job. the assumption is a job is failing if the conclusion isn't success, skipped, or cancelled
      let failingStep = undefined; // initialize the failing step to undefined
      if (!failingJob) { // if there is no failing job, find the failing step (sometimes a job might pass while a step fails I guess)
        for (const job of jobs) {
          const step = (job.steps || []).find(s => s.conclusion === 'failure');
          if (step) { failingJob = job; failingStep = step; break; } // if the step is a failure, set the failing job and step
        }
      }
      apiCheckSucceeded = true;
      if (failingJob) {
        hasFailingJob = true;
        failingLabel = `${failingJob.name}${failingStep ? ' / ' + failingStep.name : ''}`; // set the failing label to the job name and step name if the step is a failure
      }
    } catch (e) {
      core.info(`Job status lookup failed for run ${runId}: ${e.message}`);
    }

    // If we found FAILED lines but the run has no failing job, suppress false positives
    // Only do this if the API check succeeded; on API failure, keep the snippets.
    // if ((snippets && snippets.length > 0) && apiCheckSucceeded && !hasFailingJob) {
    //   snippets = [];
    // }

    // If we did not find FAILED lines but the run has a failing job, emit synthetic entry
    if ((!snippets || snippets.length === 0) && hasFailingJob) {
      const owner = findOwnerForLabel(failingLabel) || 'no owner found';
      snippets = [{ label: failingLabel, owner, snippet: 'could not find failure in logs' }];
    }

    return snippets;
  } catch (e) {
    core.info(`Failed to obtain run logs for ${runId}: ${e.message}`);
    return [];
  } finally {
    core.endGroup();
  }
}

/**
 * Recursively find up to maxCount error snippets in a directory of text logs.
 */
function findErrorSnippetsInDir(rootDir, maxCount) {
  // Relaxed: match anywhere on the line (after prefix stripping)
  const infoRegex = /info:/;
  const backtraceRegex = /backtrace:/;
  // Failure markers used both in primary and fallback passes
  const failureMarkers = [
    /\bFAILED\b/,          // pytest summary and generic FAILED
    /\[\s*FAILED\s*\]/,  // gtest [  FAILED  ]
    /\bERROR\b/i,           // generic ERROR
    /Traceback\b/i,         // python tracebacks
    /AssertionError\b/i,
    /Segmentation fault/i
  ];

  const collected = [];
  const stack = [rootDir]; // trace of the stack to use when recursing through the directories

  while (stack.length && collected.length < maxCount) { // while the stack is not empty and the collected snippets are less than the max count
    const dir = stack.pop(); //remove the last directory from the stack
    const entries = fs.readdirSync(dir, { withFileTypes: true }); // list all the files and directories in the directory
    for (const ent of entries) {
      const p = path.join(dir, ent.name); // join the directory and the entry name to get the path to the entry
      if (ent.isDirectory()) {
        stack.push(p); // add the path to the stack
      } else if (ent.isFile() && (p.endsWith('.txt') || p.endsWith('.log') || !path.basename(p).includes('.'))) { // if the entry is a file and the file name ends with .txt, .log, or doesn't include a period
        // Only consider logs whose filenames indicate test content
        const fileBaseName = path.basename(p); // get the base name of the file
        if (!/tests?/i.test(fileBaseName)) { // if the file name doesn't include tests
          core.info(`Skipping non-test log file: ${fileBaseName}`); // log the file name
          continue; // continue to the next entry
        }
        try {
          const text = fs.readFileSync(p, 'utf8'); // read the file as utf8
          const rawLines = text.split(/\r?\n/); // split the text into lines
          const lines = rawLines.map(l => l // map each line to a new line
            .replace(/^\s*\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+/, '') // remove the timestamp
            .replace(/^\s*\[[0-9]+,[0-9]+\]<[^>]+>:\s*/, '') // remove the line number
          );
          core.info(`Scanning ${lines.length} lines in ${p}`); // log the number of lines in the file
          let foundInFile = 0; // initialize the found in file count to 0

          // A) info: ... until backtrace:
          for (let i = 0; i < lines.length && collected.length < maxCount; i++) {
            if (infoRegex.test(lines[i])) { // test if the line contains info:
              // Guardrail: only consider this an error snippet if nearby lines include
              // a failure marker or a backtrace. Scan a small window ahead.
              const upper = Math.min(lines.length, i + 200); // set the upper limit to the length of the lines array or the index plus 200
              let windowHasFailure = false;
              for (let k = i; k < upper; k++) { // iterate through the lines in the window
                const ln = lines[k]; // get the line
                if (backtraceRegex.test(ln)) { // test if the line contains backtrace
                  windowHasFailure = true; // set the window has failure to true
                  break;
                }
              }
              if (!windowHasFailure) continue; // if the window has no failure, continue to the next line

              const block = [lines[i].trim()]; // add the line to the block
              let j = i + 1; // set the index to the next line
              while (j < lines.length && !backtraceRegex.test(lines[j]) && collected.length < maxCount) {
                if (lines[j].trim() !== '') block.push(lines[j].trim()); // if the line is not empty, add it to the block
                j++;
              }
              const testLabel = extractTestLabelBackward(lines, i); // extract the test label from the lines
              const textBlock = block.join('\n'); // join the block into a single string
              const fileBase = path.basename(p, path.extname(p)); // get the base name of the log file
              const finalLabel = testLabel ? `${fileBase}:\n${testLabel}` : `${fileBase}:\nno label found`; // set the final label to the file base and test label if the test label is not empty, otherwise set it to the file base and no label found
              collected.push({ snippet: textBlock.length > 600 ? textBlock.slice(0, 600) + '…' : textBlock, label: finalLabel }); // add the snippet to the collected snippets. if the text block is longer than 600 characters, truncate it to 600 characters
              foundInFile++;
              i = j;
            }
          }

          // No other passes by design (keep it simple): only info..backtrace blocks

          core.info(`Parsed log file: ${p} → found ${foundInFile} snippet(s)`); // log the number of snippets found in the file
        } catch (_) { /* ignore */ }
      }
      if (collected.length >= maxCount) break; // if the collected snippets are greater than or equal to the max count, break the loop
    }
  }

  // Fallback: if no snippets collected via primary heuristics, try to at least
  // return names (and a small context) of log files that clearly indicate failure.
  if (collected.length === 0) {

    const stack2 = [rootDir]; // set up new stack
    while (stack2.length && collected.length < maxCount) { // while the stack is not empty and the collected snippets are less than the max count
      const dir = stack2.pop(); // remove the last directory from the stack
      const entries = fs.readdirSync(dir, { withFileTypes: true }); // list all the files and directories in the directory
      for (const ent of entries) {
        const p = path.join(dir, ent.name); // join the directory and the entry name to get the path to the entry
        if (ent.isDirectory()) { // if the entry is a directory
          stack2.push(p); // add the path to the stack
        } else if (ent.isFile() && (p.endsWith('.txt') || p.endsWith('.log') || !path.basename(p).includes('.'))) {
          try { // try to read the file
            // Simple fallback: Only consider lines where 'FAILED' is the first word after
            // stripping leading timestamps and whitespace.
            const text = fs.readFileSync(p, 'utf8'); // read the file as utf8
            const rawLines = text.split(/\r?\n/); // split the text into lines
            const timestampPrefix = /^\s*\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+/; // regex to match the timestamp prefix
            const FAILED_AT_START = /^FAILED\b/; // case-sensitive, first word is FAILED
            for (let i = 0; i < rawLines.length && collected.length < maxCount; i++) { // iterate through the lines
              const original = rawLines[i];
              const modified = rawLines.slice(i, i + 2).join('\n'); // join two consecutive lines into a single string (safe because slice doesn't throw an error if the index is out of bounds)
              const stripped = original.replace(timestampPrefix, '').replace(/^\s+/, ''); // remove the timestamp prefix and leading whitespace
              if (FAILED_AT_START.test(stripped)) { // if the line starts with FAILED (this logic is flawed)
                const fileBase = path.basename(p).split('.')[0]; // get the base name of the log file
                collected.push({ // add the snippet to the collected snippets
                  label: `${fileBase}:\nFAILED line`,
                  // Return the original line (full context), truncated
                  snippet: original.length > 600 ? original.slice(0, 600) + '…' : modified,
                });
                break; // one per file is enough
              }
            }
          } catch (_) { /* ignore */ }
        }
        if (collected.length >= maxCount) break; // if the collected snippets are greater than or equal to the max count, break the loop
      }
    }
  }
  // TODO: the failure markers are incorrectly flagging the line below them when the failing test
  // is actually far above them. the logic above needs to be changed

  // Attach owners based on mapping
  try {
    for (const it of collected) { // iterate through the collected snippets
      if (it && !it.owner) { // if the item is not empty and the owner is not set
        const owner = findOwnerForLabel(it.label || '');
        if (owner) it.owner = owner; // if the owner is found, set the owner to the owner
      }
    }
  } catch (_) { /* ignore */ }

  return collected; // return the collected snippets
}

/**
 * Try to extract a test identifier from the nearby log lines or path.
 */
function extractTestLabelBackward(lines, errIdx) {
  const runExact = /^\s*\[\s*RUN\s*\]\s*(.+)$/; // capture gtest name after [ RUN ]
  const failedTests = /^\s*FAILED\s+tests\//;      // lines starting with FAILED tests/
  const failedOnly = /^\s*FAILED\s*$/;      // exactly "FAILED"
  const testsPathStart = /^\s*tests\//;       // must start with tests/

  for (let i = errIdx; i >= 0; i--) {
    // Normalize the line the same way as in the scanner (strip prefixes)
    const raw = lines[i] || '';
    const line = raw
      .replace(/^\s*\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+/, '')
      .replace(/^\s*\[[0-9]+,[0-9]+\]<[^>]+>:\s*/, ''); //this basically removes timestamps and process/thread IDs
    // gtest: return the part after [ RUN ]
    const mRun = line.match(runExact); // check if the cleaned up line starts with [ RUN ]
    if (mRun) return mRun[1].trim(); // if so, return the part after [ RUN ]
    // pytest FAILED header: extract tests/...py::test_name (without params)
    if (failedTests.test(line)) { // this is just a boolean check
      const mF = line.match(/^\s*FAILED\s+((tests\/[^\s]+\.py))::([^\s\[]+)/); // this is a regex to match the test name
      if (mF) return `${mF[1]}::${mF[3]}`; // return the test file path appended with the test name
      // fallback to any tests path at start
      const mFs = line.match(/^\s*(tests\/[^\s]+\.py(?:::[^\s\[]+)*)/); // this is a regex to match the test file path. stops capturing after whitespace of [
      if (mFs) return mFs[1]; // if so, return the test file path
      return line.trim();
    }
    if (failedOnly.test(line)) {
      // return first non-empty line below
      let j = i + 1; // set the index to the next line
      while (j < lines.length && lines[j].trim() === '') j++; // while the index is less than the length of the lines array and the line is empty, increment the index
      if (j < lines.length) {
        const candRaw = lines[j]; // get the line
        const cand = candRaw // clean up the line
          .replace(/^\s*\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+/, '')
          .replace(/^\s*\[[0-9]+,[0-9]+\]<[^>]+>:\s*/, ''); // this basically removes timestamps and process/thread IDs
        const mPy = cand.match(/^\s*((tests\/[^\s]+\.py))::([^\s\[]+)/); // this is a regex to match a test file path and test name
        if (mPy) return `${mPy[1]}::${mPy[3]}`; // if so, return the test file path appended with the test name
        const mStart = cand.match(/^\s*(tests\/[^\s]+\.py(?:::[^\s\[]+)*)/); // simpler regex to match a test file path
        if (mStart) return mStart[1]; // if so, return the test file path
      }
    }
    if (testsPathStart.test(line)) { // this is just a boolean check to see if the line starts with tests/
      const m = line.match(/^\s*((tests\/[^\s]+\.py))::([^\s\[]+)/); // this is a regex to match a test file path and test name
      if (m) return `${m[1]}::${m[3]}`; // if so, return the test file path appended with the test name
      const m2 = line.match(/^\s*(tests\/[^\s]+\.py(?:::[^\s\[]+)*)/); // simpler regex to match a test file path
      if (m2) return m2[1]; // if so, return the test file path
      return line.trim(); // return the trimmed line
    }
  }
  return undefined; // return undefined if no test label is found
}

/**
 * Calculates statistics for a set of workflow runs.
 *
 * @param {Array<object>} runs - Array of workflow run objects
 * @returns {object} Statistics object containing:
 *   - eventTypes: Comma-separated string of unique event types
 *   - successRate: Percentage of runs that succeeded (based on last attempt)
 *   - uniqueSuccessRate: Percentage of runs that succeeded on first attempt
 *   - retryRate: Percentage of successful runs that required retries
 *   - uniqueRuns: Number of unique runs (excluding attempts)
 *   - totalRuns: Total number of runs including attempts
 *   - successfulRuns: Number of runs that succeeded (based on last attempt)
 */
function getWorkflowStats(runs) {
  // Group runs by their original run ID to handle retries and reruns
  const uniqueRuns = new Map();
  let totalRunsIncludingRetries = 0;
  let totalSuccessfulUniqueRuns = 0;
  let successfulUniqueRunsOnFirstTry = 0;
  let successfulUniqueRunsWithRetries = 0;

  // First pass: identify all unique runs and their attempts
  for (const run of runs) {
    totalRunsIncludingRetries++;

    // Calculate the original run ID by subtracting (run_attempt - 1) from the current run ID
    const originalRunId = run.run_attempt > 1 ? run.id - (run.run_attempt - 1) : run.id;

    if (!uniqueRuns.has(originalRunId)) { // returns true if the run id is not in the map
      uniqueRuns.set(originalRunId, { // set default values for the run
        run,
        attempts: 0,
        isSuccessful: false,
        requiredRetry: false,
        succeededOnFirstTry: false,
        lastAttempt: run
      });
    } else {
      // This is an attempt
      const existingRun = uniqueRuns.get(originalRunId);
      existingRun.attempts++; // this is just a re-run so increment the attempts

      // Update last attempt if this is a newer attempt
      if (run.run_attempt > existingRun.lastAttempt.run_attempt) {
        existingRun.lastAttempt = run; // update the last attempt to the current run
      }
    }
  }

  // Second pass: determine final status based on last attempt
  for (const runInfo of uniqueRuns.values()) { // iterate through the unique runs
    const lastAttempt = runInfo.lastAttempt;
    if (lastAttempt.conclusion === 'success') { // if the last attempt is successful
      runInfo.isSuccessful = true;
      totalSuccessfulUniqueRuns++; // increment the total successful unique runs

      if (lastAttempt.run_attempt === 1) { // if the last attempt is the first attempt
        runInfo.succeededOnFirstTry = true; // set the succeeded on first try to true
        successfulUniqueRunsOnFirstTry++; // increment the total successful unique runs on first try
      } else {
        runInfo.requiredRetry = true; // set the required retry to true
        successfulUniqueRunsWithRetries++; // increment the total successful unique runs with retries
      }
    }
  }

  const uniqueRunsArray = Array.from(uniqueRuns.values()).map(r => r.run); // creates an array of the run objects
  const eventTypes = [...new Set(uniqueRunsArray.map(r => r.event))].join(', '); // create a string of all the even types that were used to trigger runs

  // Calculate rates
  const successRate = uniqueRunsArray.length === 0 ? "N/A" : (totalSuccessfulUniqueRuns / uniqueRunsArray.length * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";
  const uniqueSuccessRate = uniqueRunsArray.length === 0 ? "N/A" : (successfulUniqueRunsOnFirstTry / uniqueRunsArray.length * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";
  const retryRate = totalSuccessfulUniqueRuns === 0 ? "N/A" : (successfulUniqueRunsWithRetries / totalSuccessfulUniqueRuns * 100).toFixed(SUCCESS_RATE_DECIMAL_PLACES) + "%";

  return { // return the eventypes, success rates, retry rates, and total run info
    eventTypes,
    successRate,
    uniqueSuccessRate,
    retryRate,
    uniqueRuns: uniqueRunsArray.length,
    totalRuns: totalRunsIncludingRetries,
    successfulRuns: totalSuccessfulUniqueRuns
  };
}

/**
 * Generates a GitHub Actions workflow URL.
 *
 * @param {object} context - GitHub Actions context
 * @param {string} workflowFile - Path to the workflow file relative to .github/workflows/
 * @returns {string} Full URL to the workflow in GitHub Actions
 */
function getWorkflowLink(context, workflowFile) {
  return workflowFile // return the workflow link querying on main branch if the workflow file exists
    ? `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/workflows/${workflowFile}?query=branch%3Amain`
    : `https://github.com/${context.repo.owner}/${context.repo.repo}/actions`; // return the github actions link if the workflow file does not exist
}

/**
 * Finds, within the provided window (newest→oldest, main branch only), either:
 * - the first failing run since the most recent success (oldest in the current failing streak), or
 * - if no success exists in-window, the oldest failing run in the window.
 * Returns null if there are no failing runs in the window.
 *
 * @param {Array<object>} mainBranchRunsWindow - Runs on main, sorted by created_at desc (newest first)
 * @returns {{run: object, noSuccessInWindow: boolean}|null}
 */
function findFirstFailInWindow(mainBranchRunsWindow) {
  let seenAnyFailure = false;
  let firstFailInStreak = null; // oldest failure observed before crossing a success boundary

  for (const run of mainBranchRunsWindow) {
    if (run.conclusion === 'success') {
      if (firstFailInStreak) {
        // We found a success after observing failures: return the oldest failure in the streak
        return { run: firstFailInStreak, boundarySuccessRun: run, noSuccessInWindow: false };
      }
      // Success encountered before any failure in the current scan; keep scanning older entries
    } else if (run.conclusion && run.conclusion !== 'cancelled' && run.conclusion !== 'skipped') {
      // Treat anything non-success, non-cancelled/skipped as failure for this purpose
      seenAnyFailure = true;
      firstFailInStreak = run; // update to become oldest failure within the current failing streak (and window)
    }
  }

  if (seenAnyFailure) {
    // No success found in-window; report oldest failure in the window
    return { run: firstFailInStreak, boundarySuccessRun: undefined, noSuccessInWindow: true };
  }
  return null;
}

/**
 * Finds the first failing run on main since the last success (i.e., the start of the current failing streak).
 * Scans runs in reverse chronological order and returns the oldest failure before the first encountered success.
 * Falls back to the oldest failure in history if no success is found.
 *
 * @param {object} octokit - Authenticated Octokit client
 * @param {object} context - GitHub Actions context
 * @param {string} workflowPath - Path to the workflow file (e.g., .github/workflows/ci.yaml)
 * @returns {Promise<object|null>} The workflow run object or null if none found
 */

/**
 * Analyzes scheduled runs to find the last good and earliest bad commits.
 *
 * @param {Array<object>} scheduledMainRuns - Array of scheduled runs on main branch, sorted by date (newest first)
 * @param {object} context - GitHub Actions context
 * @returns {object} Object containing:
 *   - newestGoodSha: Short SHA of the last successful run (e.g., `a1b2c3d`)
 *   - newestBadSha: Short SHA of the earliest failing run (e.g., `e4f5g6h`)
 */
function findGoodBadCommits(scheduledMainRuns, context) {
  let newestGoodSha = EMPTY_VALUE;
  let newestBadSha = EMPTY_VALUE;
  let foundGood = false;
  let foundBad = false;

  for (const run of scheduledMainRuns) {
    if (!foundGood && run.conclusion === 'success') { // find the most recent successful run
      const shortSha = run.head_sha.substring(0, SHA_SHORT_LENGTH);
      newestGoodSha = `[\`${shortSha}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${run.head_sha})`; // set the last good sha to the short sha of the most recent successful run
      foundGood = true;
    }
    if (!foundBad && run.conclusion !== 'success') { // find the earliest failing run
      const shortSha = run.head_sha.substring(0, SHA_SHORT_LENGTH);
      newestBadSha = `[\`${shortSha}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${run.head_sha})`; // set the earliest bad sha to the short sha of the earliest failing run
      foundBad = true;
    }
    if (foundGood && foundBad) break; // break the loop if both good and bad runs are found
  }

  return { newestGoodSha, newestBadSha };
}

/**
 * Gets information about the last run on main branch.
 *
 * @param {Array<object>} mainBranchRuns - Array of runs on main branch, sorted by date (newest first)
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @returns {Promise<object>} Object containing run information
 */
async function getLastRunInfo(mainBranchRuns, github, context) {
  const lastMainRun = mainBranchRuns[0];
  if (!lastMainRun) { // if there is no last main run, return the empty values
    return {
      status: EMPTY_VALUE,
      sha: EMPTY_VALUE,
      run: EMPTY_VALUE,
      pr: EMPTY_VALUE,
      title: EMPTY_VALUE,
      lastGoodSha: EMPTY_VALUE,
      earliestBadSha: EMPTY_VALUE
    };
  }

  const prInfo = await fetchPRInfo(github, context, lastMainRun.head_sha); // fetch the PR info for the last main run
  // Current approach: filter by event type
  const mainRuns = mainBranchRuns.filter(r => r.event === lastMainRun.event || r.event === 'workflow_dispatch'); // get the relevant main runs for finding good and bad commits
  // Alternative approach: include all runs on main branch
  // const mainRuns = mainBranchRuns;
  const { lastGoodSha, earliestBadSha } = findGoodBadCommits(mainRuns, context); // find the newest good and bad commits

  return {
    status: lastMainRun.conclusion === 'success' ? SUCCESS_EMOJI : FAILURE_EMOJI,
    sha: `[\`${lastMainRun.head_sha.substring(0, SHA_SHORT_LENGTH)}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${lastMainRun.head_sha})`, // get the short sha hyperlink for the latest main run
    run: `[Run](${lastMainRun.html_url})${lastMainRun.run_attempt > 1 ? ` (#${lastMainRun.run_attempt})` : ''}`, // get the run hyperlink for the latest main run
    pr: prInfo.prNumber, // get the PR number for the latest main run
    title: prInfo.prTitle, // get the PR title for the latest main run
    lastGoodSha, // get the last good sha for the latest main run
    earliestBadSha: lastMainRun.conclusion !== 'success' ? earliestBadSha : EMPTY_VALUE // if the last main run is successful, don't bother showing the most recent failure
  };
}

/**
 * Generates summary tables for push and scheduled workflows.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @returns {Promise<string>} Markdown table for all workflows
 */
async function generateSummaryBox(grouped, github, context) {
  const rows = [];

  for (const [name, runs] of grouped.entries()) {
    const mainBranchRuns = runs
      .filter(r => r.head_branch === 'main')
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at)); // for each workflow, sort the main runs by date within the window

    const stats = getWorkflowStats(runs); // get the workflow stats for the workflow

    // Skip workflows that only have workflow_dispatch as their event type. Presumably because if they have to be triggered to run they aren't super important
    if (stats.eventTypes === 'workflow_dispatch') {
      continue; // if the workflow only has workflow_dispatch as its event type, skip it
    }

    const workflowLink = getWorkflowLink(context, runs[0]?.path); // get the link to the workflow page that lists all the runs
    const runInfo = await getLastRunInfo(mainBranchRuns, github, context); // get the last run info for the workflow

    const row = `| [${name}](${workflowLink}) | ${stats.eventTypes || 'unknown'} | ${stats.totalRuns} | ${stats.successfulRuns} | ${stats.successRate} | ${stats.uniqueSuccessRate} | ${stats.retryRate} | ${runInfo.status} | ${runInfo.sha} | ${runInfo.run} | ${runInfo.pr} | ${runInfo.title} | ${runInfo.earliestBadSha} | ${runInfo.lastGoodSha} |`; // build the row of the summary table
    rows.push(row);
  }

  return [
    '## Workflow Summary',
    '| Workflow | Event Type(s) | Total Runs | Successful Runs | Success Rate | Unique Success Rate | Retry Rate | Last Run on `main` | Last SHA | Last Run | Last PR | PR Title | Earliest Bad SHA | Last Good SHA |',
    '|----------|---------------|------------|-----------------|--------------|-------------------|------------|-------------------|----------|----------|---------|-----------|------------------|---------------|',
    ...rows,
    ''  // Empty line for better readability
  ].join('\n'); // turn the array of rows into a single string so it can be rendered as a markdown table
}

/**
 * Builds the complete markdown report.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} github - Octokit client instance
 * @param {object} context - GitHub Actions context
 * @returns {Promise<string>} Complete markdown report
 */
async function buildReport(grouped, github, context) {
  const days = parseInt(core.getInput('days') || DEFAULT_LOOKBACK_DAYS, 10);
  const timestamp = new Date().toISOString();
  return [
    `# Workflow Summary (Last ${days} Days) - Generated at ${timestamp}\n`,
    await generateSummaryBox(grouped, github, context),
    '\n## Column Descriptions\n',
    'A unique run represents a single workflow execution, which may have multiple retry attempts. For example, if a workflow fails and is retried twice, this counts as one unique run with three attempts (initial run + two retries).\n',
    '\n### Success Rate Calculations\n',
    'The success rates are calculated based on unique runs (not including retries in the denominator):\n',
    '- **Success Rate**: (Number of unique runs that eventually succeeded / Total number of unique runs) × 100%\n',
    '  - Example: 3 successful unique runs out of 5 total unique runs = 60% success rate\n',
    '- **Unique Success Rate**: (Number of unique runs that succeeded on first try / Total number of unique runs) × 100%\n',
    '  - Example: 1 unique run succeeded on first try out of 5 total unique runs = 20% unique success rate\n',
    '- **Retry Rate**: (Number of successful unique runs that needed retries / Total number of successful unique runs) × 100%\n',
    '  - Example: 2 successful unique runs needed retries out of 3 total successful unique runs = 66.67% retry rate\n',
    '\nNote: Unique Success Rate + Retry Rate does not equal 100% because they measure different things:\n',
    '- Unique Success Rate is based on all unique runs\n',
    '- Retry Rate is based only on successful unique runs\n',
    '\n| Column | Description |',
    '|--------|-------------|',
    '| Workflow | Name of the workflow with link to its GitHub Actions page |',
    '| Event Type(s) | Types of events that trigger this workflow (e.g., push, pull_request, schedule) |',
    '| Total Runs | Total number of workflow runs including all retry attempts (e.g., 1 unique run with 2 retries = 3 total runs) |',
    '| Successful Runs | Number of unique workflow runs that eventually succeeded, regardless of whether they needed retries |',
    '| Success Rate | Percentage of unique workflow runs that eventually succeeded (e.g., 3/5 unique runs succeeded = 60%) |',
    '| Unique Success Rate | Percentage of unique workflow runs that succeeded on their first attempt without needing retries (e.g., 1/5 unique runs succeeded on first try = 20%) |',
    '| Retry Rate | Percentage of successful unique runs that needed retries to succeed (e.g., of 3 successful unique runs, 2 needed retries = 66.67%) |',
    '| Last Run on `main` | Status of the most recent run on the main branch (✅ for success, ❌ for failure) |',
    '| Last SHA | Short SHA of the most recent run on main |',
    '| Last Run | Link to the most recent run on main, with attempt number if applicable |',
    '| Last PR | Link to the PR associated with the most recent run, if any |',
    '| PR Title | Title of the PR associated with the most recent run, if any |',
    '| Earliest Bad SHA | Short SHA of the earliest failing run on main (only shown if last run failed) |',
    '| Last Good SHA | Short SHA of the last successful run on main |'
  ].join('\n');
}

/**
 * Filters workflow runs by date range
 * @param {Array<Object>} runs - Array of workflow runs
 * @param {number} days - Number of days to look back
 * @returns {Array<Object>} Filtered runs within the date range
 */
function filterRunsByDate(runs, days) {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - days);

  return runs.filter(run => {
    const runDate = new Date(run.created_at);
    return runDate >= cutoffDate;
  });
}

/**
 * Collect commits between two SHAs on the default branch (main), inclusive of endSha.
 * Returns an array of { sha, short, url, author_login, author_name, author_url }.
 * Note: Uses compareCommits, which is base..head; base is typically the success commit, head is the failed run commit.
 */
async function listCommitsBetween(octokit, context, startShaExclusive, endShaInclusive) {
  try {
    const { data } = await octokit.rest.repos.compareCommits({
      owner: context.repo.owner,
      repo: context.repo.repo,
      base: startShaExclusive,
      head: endShaInclusive,
    });
    // compareCommits includes both endpoints; to make start exclusive, filter it out explicitly
    const commits = data.commits || [];
    return commits
      .filter(c => c.sha !== startShaExclusive)
      .concat(data.merge_base_commit && data.merge_base_commit.sha === endShaInclusive ? [] : [])
      .map(c => ({
        sha: c.sha,
        short: c.sha.substring(0, SHA_SHORT_LENGTH),
        url: `https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${c.sha}`,
        author_login: c.author?.login,
        author_name: c.commit?.author?.name,
        author_url: c.author?.html_url,
      }));
  } catch (e) {
    core.warning(`Failed to list commits between ${startShaExclusive}..${endShaInclusive}: ${e.message}`);
    return [];
  }
}

/**
 * Main function to run the action
 */
async function run() {
  try {
    // Get inputs
    const cachePath = core.getInput('cache-path', { required: true });
    const previousCachePath = core.getInput('previous-cache-path', { required: false });
    const workflowConfigs = JSON.parse(core.getInput('workflow_configs', { required: true }));
    const days = parseInt(core.getInput('days') || DEFAULT_LOOKBACK_DAYS, 10);
    const alertAll = String(core.getInput('alert-all') || 'false').toLowerCase() === 'true';

    // Validate inputs
    if (!fs.existsSync(cachePath)) {
      throw new Error(`Cache file not found at ${cachePath}`);
    }
    if (!Array.isArray(workflowConfigs)) {
      throw new Error('Workflow configs must be a JSON array');
    }
    if (isNaN(days) || days <= 0) {
      throw new Error('Days must be a positive number');
    }

    // Load cached data
    const grouped = JSON.parse(fs.readFileSync(cachePath, 'utf8'));
    const hasPrevious = previousCachePath && fs.existsSync(previousCachePath);
    const previousGrouped = hasPrevious ? JSON.parse(fs.readFileSync(previousCachePath, 'utf8')) : null;

    // Track failed workflows
    const failedWorkflows = [];

    // Filter and process each workflow configuration
    const filteredGrouped = new Map();
    const filteredPreviousGrouped = new Map();
    for (const config of workflowConfigs) {
      core.info(`Processing config: ${JSON.stringify(config)}`);
      for (const [name, runs] of grouped) {
        if ((config.wkflw_name && name === config.wkflw_name) ||
            (config.wkflw_prefix && name.startsWith(config.wkflw_prefix))) {
          core.info(`Matched workflow: ${name} with config: ${JSON.stringify(config)}`);
          // Filter runs by date range
          const filteredRuns = filterRunsByDate(runs, days);
          if (filteredRuns.length > 0) {
            filteredGrouped.set(name, filteredRuns);

            // Check if latest run on main is failing
            const mainBranchRuns = filteredRuns
              .filter(r => r.head_branch === 'main')
              .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
            if (mainBranchRuns[0]?.conclusion !== 'success') {
              failedWorkflows.push(name);
            }
          }
        }
      }

      if (hasPrevious && Array.isArray(previousGrouped)) {
        for (const [name, runs] of previousGrouped) {
          if ((config.wkflw_name && name === config.wkflw_name) ||
              (config.wkflw_prefix && name.startsWith(config.wkflw_prefix))) {
            const filteredRuns = filterRunsByDate(runs, days);
            if (filteredRuns.length > 0) {
              filteredPreviousGrouped.set(name, filteredRuns);
            }
          }
        }
      }
    }

    // Create authenticated Octokit client for PR info
    const octokit = github.getOctokit(core.getInput('GITHUB_TOKEN', { required: true }));

    // Generate primary report
    const mainReport = await buildReport(filteredGrouped, octokit, github.context);

    // Optional: Build Slack-ready alert message for all failing workflows with owner mentions
    let alertAllMessage = '';
    if (alertAll && failedWorkflows.length > 0) {
      const mention = (owners) => {
        const arr = Array.isArray(owners) ? owners : (owners ? [owners] : []);
        const ids = arr.map(o => (o && o.id) ? `<@${o.id}>` : '').filter(Boolean);
        return ids.length ? ids.join(' ') : '';
      };

      const failingItems = [];
      for (const [name, runs] of filteredGrouped.entries()) {
        const mainRuns = runs
          .filter(r => r.head_branch === 'main')
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        if (!mainRuns[0] || mainRuns[0].conclusion === 'success') continue;
        // Try to attach owners from the first failing run's label via snippets; fallback to job name
        // Use the latest failing run for snippet-based owner detection
        const latestFail = mainRuns.find(r => r.conclusion !== 'success');
        let owners = undefined;
        try {
          const errs = await fetchErrorSnippetsForRun(octokit, github.context, latestFail.id, 10);
          // Aggregate owners from snippets
          const ownerSet = new Map();
          for (const e of (errs || [])) {
            if (Array.isArray(e.owner)) {
              for (const o of e.owner) {
                if (!o) continue;
                const key = `${o.id || ''}|${o.name || ''}`;
                ownerSet.set(key, o);
              }
            }
          }
          owners = Array.from(ownerSet.values());
        } catch (_) { /* ignore */ }
        // Fallback: try to resolve owners from the workflow name
        if (!owners || owners.length === 0) {
          owners = findOwnerForLabel(name);
        }
        const ownerMentions = mention(owners) || '(no owner found)';
        const wfUrl = getWorkflowLink(github.context, runs[0]?.path);
        failingItems.push(`• ${name} ${wfUrl ? `<${wfUrl}|open>` : ''} ${ownerMentions}`.trim());
      }
      if (failingItems.length) {
        alertAllMessage = [
          '*Alerts: failing workflows on main*',
          ...failingItems
        ].join('\n');
      }
    }

    // Compute status changes vs previous and write JSON
    const computeLatestConclusion = (runs) => {
      const mainBranchRuns = runs
        .filter(r => r.head_branch === 'main')
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      const latest = mainBranchRuns[0];
      if (!latest) return null;
      return latest.conclusion === 'success' ? 'success' : 'failure';
    };
    const computeLatestRunInfo = (runs) => {
      const mainBranchRuns = runs
        .filter(r => r.head_branch === 'main')
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      const latest = mainBranchRuns[0];
      if (!latest) return null;
      return { id: latest.id, url: latest.html_url, created_at: latest.created_at, head_sha: latest.head_sha, path: latest.path };
    };

    const allNames = new Set([
      ...Array.from(filteredGrouped.keys()),
      ...Array.from(filteredPreviousGrouped.keys())
    ]);

    const changes = [];
    const regressedDetails = [];
    const stayedFailingDetails = [];
    for (const name of allNames) {
      const currentRuns = filteredGrouped.get(name);
      const previousRuns = filteredPreviousGrouped.get(name);
      if (!currentRuns || !previousRuns) continue; // require data on both sides
      const current = computeLatestConclusion(currentRuns);
      const previous = computeLatestConclusion(previousRuns);
      if (!current || !previous) continue;

      let change;
      if (previous === 'success' && current === 'success') change = 'stayed_succeeding';
      else if (previous !== 'success' && current !== 'success') change = 'stayed_failing';
      else if (previous !== 'success' && current === 'success') change = 'fail_to_success';
      else if (previous === 'success' && current !== 'success') change = 'success_to_fail';

      if (change) {
        const info = computeLatestRunInfo(currentRuns);
        const workflowUrl = info?.path ? getWorkflowLink(github.context, info.path) : undefined;
        const aggregateRunUrl = `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/actions/runs/${github.context.runId}`;
        const commitUrl = info?.head_sha ? `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${info.head_sha}` : undefined;
        const commitShort = info?.head_sha ? info.head_sha.substring(0, 7) : undefined;
        changes.push({ name, previous, current, change, run_id: info?.id, run_url: info?.url, created_at: info?.created_at, workflow_url: workflowUrl, workflow_path: info?.path, aggregate_run_url: aggregateRunUrl, commit_sha: info?.head_sha, commit_short: commitShort, commit_url: commitUrl });
        if (change === 'success_to_fail' && info) {
          regressedDetails.push({ name, run_id: info.id, run_url: info.url, created_at: info.created_at, workflow_url: workflowUrl, workflow_path: info.path, aggregate_run_url: aggregateRunUrl, commit_sha: info.head_sha, commit_short: commitShort, commit_url: commitUrl });
        }
        else if (change === 'stayed_failing' && info) {
          stayedFailingDetails.push({ name, run_id: info.id, run_url: info.url, created_at: info.created_at, workflow_url: workflowUrl, workflow_path: info.path, aggregate_run_url: aggregateRunUrl, commit_sha: info.head_sha, commit_short: commitShort, commit_url: commitUrl });
        }
      }
    }

    // Helper to get main runs within the current window from a grouped collection
    const getMainWindowRuns = (runs) => runs
      .filter(r => r.head_branch === 'main')
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    // Helper to get up to N most-recent failing runs (excluding successes) from a run list
  const getRecentFailingRuns = (runs, limit = 1) => {
      return runs
        .filter(r => r.head_branch === 'main' && r.conclusion !== 'success')
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
        .slice(0, limit);
    };

    // Enrich regressions with first failing run within the window
    for (const item of regressedDetails) {
      try {
        const windowRuns = getMainWindowRuns(filteredGrouped.get(item.name) || []);
        const res = findFirstFailInWindow(windowRuns);
        if (res && res.run) {
          item.first_failed_run_id = res.run.id;
          item.first_failed_run_url = res.run.html_url;
          item.first_failed_created_at = res.run.created_at;
          item.first_failed_head_sha = res.run.head_sha;
          item.first_failed_head_short = res.run.head_sha ? res.run.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;
          item.no_success_in_window = !!res.noSuccessInWindow;
          if (!res.noSuccessInWindow && res.boundarySuccessRun && res.boundarySuccessRun.head_sha) {
            // Get commits between boundary success and first failing run (inclusive of failing run)
            item.commits_between = await listCommitsBetween(octokit, github.context, res.boundarySuccessRun.head_sha, item.first_failed_head_sha);
          }
          // Also capture the latest failing run in the window
          const latestFailRun = (getRecentFailingRuns(filteredGrouped.get(item.name) || [], 1)[0]);
          if (latestFailRun) {
            item.latest_failed_run_id = latestFailRun.id;
            item.latest_failed_run_url = latestFailRun.html_url;
            item.latest_failed_created_at = latestFailRun.created_at;
            item.latest_failed_head_sha = latestFailRun.head_sha;
            item.latest_failed_head_short = latestFailRun.head_sha ? latestFailRun.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;
          }
          // Commit author enrichment is now superseded by commits_between list; keep top-level for convenience if present
          if (item.first_failed_head_sha) {
            const author = await fetchCommitAuthor(octokit, github.context, item.first_failed_head_sha);
            item.first_failed_author_login = author.login;
            item.first_failed_author_name = author.name;
            item.first_failed_author_url = author.htmlUrl;
          }
          // Error snippets for the first failing run (best-effort)
          // Use the most recent failing run instead of the first in-window failure
          const latestFail = (getRecentFailingRuns(filteredGrouped.get(item.name) || [], 1)[0]) || { id: item.first_failed_run_id };
          item.error_snippets = latestFail?.id ? await fetchErrorSnippetsForRun(octokit, github.context, latestFail.id, 20) : [];
          // Omit repeated errors logic (simplified)
          item.repeated_errors = [];
          // Mirror into the corresponding change entry
          const changeRef = changes.find(c => c.name === item.name && c.change === 'success_to_fail');
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
              latest_failed_run_id: item.latest_failed_run_id,
              latest_failed_run_url: item.latest_failed_run_url,
              latest_failed_created_at: item.latest_failed_created_at,
              latest_failed_head_sha: item.latest_failed_head_sha,
              latest_failed_head_short: item.latest_failed_head_short,
            });
          }
        }
      } catch (e) {
        core.warning(`Failed to find first failing run for ${item.name}: ${e.message}`);
      }
    }

    // Enrich stayed failing with first failing run within the window
    for (const item of stayedFailingDetails) {
      try {
        const windowRuns = getMainWindowRuns(filteredGrouped.get(item.name) || []);
        const res = findFirstFailInWindow(windowRuns);
        if (res && res.run) {
          item.first_failed_run_id = res.run.id;
          item.first_failed_run_url = res.run.html_url;
          item.first_failed_created_at = res.run.created_at;
          item.first_failed_head_sha = res.run.head_sha;
          item.first_failed_head_short = res.run.head_sha ? res.run.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;
          item.no_success_in_window = !!res.noSuccessInWindow;
          // Do not fetch commits/authors for stayed_failing if no success in-window
          if (!item.no_success_in_window && res.boundarySuccessRun && res.boundarySuccessRun.head_sha) {
            item.commits_between = await listCommitsBetween(octokit, github.context, res.boundarySuccessRun.head_sha, item.first_failed_head_sha);
          }
          // Also capture latest failing run in window
          const latestFailRun2 = (getRecentFailingRuns(filteredGrouped.get(item.name) || [], 1)[0]);
          if (latestFailRun2) {
            item.latest_failed_run_id = latestFailRun2.id;
            item.latest_failed_run_url = latestFailRun2.html_url;
            item.latest_failed_created_at = latestFailRun2.created_at;
            item.latest_failed_head_sha = latestFailRun2.head_sha;
            item.latest_failed_head_short = latestFailRun2.head_sha ? latestFailRun2.head_sha.substring(0, SHA_SHORT_LENGTH) : undefined;
          }
          // Commit author of the first failed in-window (optional)
          if (item.first_failed_head_sha) {
            const author = await fetchCommitAuthor(octokit, github.context, item.first_failed_head_sha);
            item.first_failed_author_login = author.login;
            item.first_failed_author_name = author.name;
            item.first_failed_author_url = author.htmlUrl;
          }
          // Use the most recent failing run instead of the first in-window failure
          const latestFail2 = (getRecentFailingRuns(filteredGrouped.get(item.name) || [], 1)[0]) || { id: item.first_failed_run_id };
          item.error_snippets = latestFail2?.id ? await fetchErrorSnippetsForRun(octokit, github.context, latestFail2.id, 20) : [];
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
            latest_failed_run_id: item.latest_failed_run_id,
            latest_failed_run_url: item.latest_failed_run_url,
            latest_failed_created_at: item.latest_failed_created_at,
            latest_failed_head_sha: item.latest_failed_head_sha,
            latest_failed_head_short: item.latest_failed_head_short,
          });
        }
      }
      catch (e) {
        core.warning(`Failed to find first failing run for ${item.name}: ${e.message}`);
      }
    }

    const outputDir = process.env.GITHUB_WORKSPACE || process.cwd();
    const statusChangesPath = path.join(outputDir, 'workflow-status-changes.json');
    fs.writeFileSync(statusChangesPath, JSON.stringify(changes));
    core.setOutput('status_changes_path', statusChangesPath);

    // Build a minimal regressions section (success -> fail only)
    let regressionsSection = '';
    let stayedFailingSection = '';
    try {
      const parsed = Array.isArray(changes) ? changes : [];
      const regressionsItems = parsed.filter(item => item.change === 'success_to_fail');
      const stayedFailingItems = parsed.filter(item => item.change === 'stayed_failing');
      if (regressionsItems.length > 0) {
        const lines = regressionsItems.map(it => {
          const base = it.workflow_url ? `- [${it.name}](${it.workflow_url})` : `- ${it.name}`;
          if (it.first_failed_run_url) {
            const sha = it.first_failed_head_short || (it.first_failed_head_sha ? it.first_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
            const shaLink = sha ? `[\`${sha}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.first_failed_head_sha})` : '';
            const when = it.first_failed_created_at ? new Date(it.first_failed_created_at).toISOString() : '';
            const author = it.first_failed_author_login
              ? `by [@${it.first_failed_author_login}](${it.first_failed_author_url})`
              : (it.first_failed_author_name ? `by ${it.first_failed_author_name}` : '');
            // Error snippets first
            let errorsList = '';
            const errorsHtml = renderErrorsTable(it.error_snippets || []);
            errorsList = ['','  - Errors (table below):','', errorsHtml, ''].join('\n');
            if (it.no_success_in_window) {
              const latestLink = it.latest_failed_run_url ? ` | Latest failing run: [Run](${it.latest_failed_run_url}) ${it.latest_failed_created_at ? new Date(it.latest_failed_created_at).toISOString() : ''} ${it.latest_failed_head_short ? `[\\\`${it.latest_failed_head_short}\\"](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.latest_failed_head_sha})` : ''}` : '';
              return [`${base}\n  - Failed to find any successful run in the last two weeks. Oldest failing run is: [Run](${it.first_failed_run_url}) ${when} ${shaLink}${latestLink}`, errorsList].filter(Boolean).join('\n');
            }
            // Include commits between success and failure
            let commitsList = '';
            const commitsHtml = renderCommitsTable(it.commits_between || []);
            commitsList = ['','  - Commits between last success and first failure (table below):','', commitsHtml, ''].join('\n');
            const latestWhenIso = it.latest_failed_created_at ? new Date(it.latest_failed_created_at).toISOString() : '';
            const latestShaShort = it.latest_failed_head_short || (it.latest_failed_head_sha ? it.latest_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
            const latestShaLink = (latestShaShort && it.latest_failed_head_sha)
              ? ` [\`${latestShaShort}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.latest_failed_head_sha})`
              : '';
            const latestLine = it.latest_failed_run_url
              ? `\n  - Latest failing run: [Run](${it.latest_failed_run_url}) ${latestWhenIso}${latestShaLink}`
              : '';
            return [`${base}\n  - First failing run on main: [Run](${it.first_failed_run_url}) ${when} ${shaLink} ${author}${latestLine}`, errorsList, commitsList].filter(Boolean).join('\n');
          }
          return base;
        });
        regressionsSection = ['', '## Regressions (Pass → Fail)', ...lines, ''].join('\n');
      } else {
        regressionsSection = ['','## Regressions (Pass → Fail)','- None',''].join('\n');
      }
      if (stayedFailingItems.length > 0) {
        const lines = stayedFailingItems.map(it => {
          const base = it.workflow_url ? `- [${it.name}](${it.workflow_url})` : `- ${it.name}`;
          if (it.first_failed_run_url) {
            const sha = it.first_failed_head_short || (it.first_failed_head_sha ? it.first_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
            const shaLink = sha ? `[\`${sha}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.first_failed_head_sha})` : '';
            const when = it.first_failed_created_at ? new Date(it.first_failed_created_at).toISOString() : '';
            // Error snippets first
            let errorsList = '';
            const errorsHtml2 = renderErrorsTable(it.error_snippets || []);
            errorsList = ['','  - Errors (table below):','', errorsHtml2, ''].join('\n');
            if (it.no_success_in_window) {
              const latestWhenIso = it.latest_failed_created_at ? new Date(it.latest_failed_created_at).toISOString() : '';
              const latestShaShort = it.latest_failed_head_short || (it.latest_failed_head_sha ? it.latest_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
              const latestShaLink = (latestShaShort && it.latest_failed_head_sha)
                ? ` [\`${latestShaShort}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.latest_failed_head_sha})`
                : '';
              const latestLine = it.latest_failed_run_url
                ? ` | Latest failing run: [Run](${it.latest_failed_run_url}) ${latestWhenIso}${latestShaLink}`
                : '';
              return [`${base}\n  - Failed to find any successful run in the last two weeks. Oldest failing run is: [Run](${it.first_failed_run_url}) ${when} ${shaLink}${latestLine}`, errorsList].filter(Boolean).join('\n');
            }
            // If there is a success boundary in-window, show commits between; otherwise, just show first failure
            let commitsList = '';
            const commitsHtml2 = renderCommitsTable(it.commits_between || []);
            commitsList = ['','  - Commits between last success and first failure (table below):','', commitsHtml2, ''].join('\n');
            const latestWhenIso = it.latest_failed_created_at ? new Date(it.latest_failed_created_at).toISOString() : '';
            const latestShaShort = it.latest_failed_head_short || (it.latest_failed_head_sha ? it.latest_failed_head_sha.substring(0, SHA_SHORT_LENGTH) : undefined);
            const latestShaLink = (latestShaShort && it.latest_failed_head_sha)
              ? ` [\`${latestShaShort}\`](https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${it.latest_failed_head_sha})`
              : '';
            const latestLine = it.latest_failed_run_url
              ? `\n  - Latest failing run: [Run](${it.latest_failed_run_url}) ${latestWhenIso}${latestShaLink}`
              : '';
            return [`${base}\n  - First failing run on main: [Run](${it.first_failed_run_url}) ${when} ${shaLink}${latestLine}`, errorsList, commitsList].filter(Boolean).join('\n');
          }
          return base;
        });
        stayedFailingSection = ['', '## Still Failing (No Recovery)', ...lines, ''].join('\n');
      } else {
        stayedFailingSection = ['','## Still Failing (No Recovery)','- None',''].join('\n');
      }
    } catch (_) {
      // Fallback: always show headers even if nothing parsed
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
    if (alertAll) core.setOutput('alert_all_message', alertAllMessage || '');
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
