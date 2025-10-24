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

// Constants
const DEFAULT_LOOKBACK_DAYS = 15;
const SHA_SHORT_LENGTH = 7;
const SUCCESS_RATE_DECIMAL_PLACES = 2;
const SUCCESS_EMOJI = '✅';
const FAILURE_EMOJI = '❌';
const EMPTY_VALUE = '—';
// Default owner override when a test name cannot be extracted
const DEFAULT_INFRA_OWNER = { id: 'S0985AN7TC5', name: 'metal infra team' };

// Optional annotations index mapping (runId -> directory)
// This is populated when action inputs provide their paths.
let __annotationsIndexMap = undefined;

// Optional gtest logs index mapping (runId -> directory)
let __gtestLogsIndexMap = undefined;

// Optional other logs index mapping (runId -> directory)
let __otherLogsIndexMap = undefined;



/**
 * Resolve the extracted logs directory for a given run ID using the loaded index.
 */
function loadAnnotationsIndexFromFile(filePath) {
  try {
    if (!filePath || !fs.existsSync(filePath)) return undefined;
    const raw = fs.readFileSync(filePath, 'utf8');
    const json = JSON.parse(raw);
    const map = new Map();
    if (json && typeof json === 'object' && !Array.isArray(json)) {
      for (const [k, v] of Object.entries(json)) {
        if (typeof v === 'string' && v) map.set(String(k), v);
      }
      return map;
    }
  } catch (_) { /* ignore */ }
  return undefined;
}

function getAnnotationsDirForRunId(runId) {
  try {
    if (!__annotationsIndexMap) return undefined;
    const key = String(runId);
    return __annotationsIndexMap.get(key);
  } catch (_) {
    return undefined;
  }
}

function loadLogsIndexFromFile(filePath) {
  try {
    if (!filePath || !fs.existsSync(filePath)) return undefined;
    const raw = fs.readFileSync(filePath, 'utf8');
    const json = JSON.parse(raw);
    const map = new Map();
    if (json && typeof json === 'object' && !Array.isArray(json)) {
      for (const [k, v] of Object.entries(json)) {
        if (typeof v === 'string' && v) map.set(String(k), v);
      }
      return map;
    }
  } catch (_) { /* ignore */ }
  return undefined;
}

function getGtestLogsDirForRunId(runId) {
  try {
    if (!__gtestLogsIndexMap) return undefined;
    const key = String(runId);
    return __gtestLogsIndexMap.get(key);
  } catch (_) {
    return undefined;
  }
}

function getOtherLogsDirForRunId(runId) {
  try {
    if (!__otherLogsIndexMap) return undefined;
    const key = String(runId);
    return __otherLogsIndexMap.get(key);
  } catch (_) {
    return undefined;
  }
}

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

function getJobNameComponentTail(component) {
  const tokens = extractSignificantTokens(component);
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
        // Backward-compat: accept both "job-name-component" and legacy "needle" keys
        const component = (entry && typeof entry["job-name-component"] === 'string' && entry["job-name-component"]) || (entry && typeof entry.needle === 'string' && entry.needle) || undefined;
        if (!component) continue; // skip if missing
        if (lbl.includes(component)) { // direct substring match
          return normalizeOwners(entry.owner);
        }
        // Fuzzy match: try last token from the component
        const tail = getJobNameComponentTail(component);
        if (tail && labelTokens.includes(tail)) {
          return normalizeOwners(entry.owner);
        }
        // Additional heuristic: if label tokens end with the last two tokens of the component
        const componentTokens = extractSignificantTokens(component);
        if (componentTokens.length >= 2 && labelTokens.length >= 2) {
          const componentTailPair = componentTokens.slice(-2).join(' ');
          const labelTailPair = labelTokens.slice(-2).join(' ');
          if (componentTailPair === labelTailPair) {
            return normalizeOwners(entry.owner);
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

// START OF CODE FOR ERROR HANDLING

// Helper: infer job/test from a snippet's existing fields, label, and message
function inferJobAndTestFromSnippet(snippet) {
  try {
    const rawLabel = (snippet && snippet.label) ? String(snippet.label) : '';
    const rawSnippet = (snippet && snippet.snippet) ? String(snippet.snippet) : '';
    let jobName = (snippet && typeof snippet.job === 'string') ? snippet.job : '';
    let testName = (snippet && typeof snippet.test === 'string') ? snippet.test : '';

    // 1) Parse from label when needed
    if (!jobName || !testName) {
      if (rawLabel && rawLabel.includes(':')) {
        const idx = rawLabel.indexOf(':');
        const left = rawLabel.slice(0, idx).trim();
        const right = rawLabel.slice(idx + 1).trim();
        if (!jobName) jobName = left;
        if (!testName && right) testName = right.replace(/\s*\[[^\]]+\]\s*$/, '').trim();
      }
    }

    // 2) Fallback heuristics based on snippet body
    if (!testName) {
      const trimmed = rawSnippet.replace(/^\s+/, '');
      const generic = /(lost\s+connection|timeout|timed\s*out|connection\s+reset|network\s+is\s+unreachable|no\s+space\s+left|killed\s+by|out\s+of\s+memory)/i;
      const ib = trimmed.indexOf('[');
      const ispace = trimmed.indexOf(' ');
      const endWord = (ispace === -1 && ib === -1) ? trimmed.length : Math.min(...[ispace, ib].filter(v => v !== -1));
      const firstToken = trimmed.slice(0, Math.max(0, endWord)).trim();
      if (generic.test(trimmed)) testName = 'NA';
      else if (firstToken && /test/i.test(firstToken)) testName = firstToken;
      else if (ib !== -1) testName = trimmed.slice(0, ib).trim() || 'NA';
      else testName = 'NA';
    }

    // 3) Final fallback for job from label
    if (!jobName) jobName = rawLabel || '';
    return { job: jobName.replace(/\s*\[[^\]]+\]\s*$/, ''), test: testName };
  } catch (_) {
    return { job: (snippet && snippet.job) || '', test: (snippet && snippet.test) || 'NA' };
  }
}

// Helper: resolve owners for a snippet after test has been inferred
function resolveOwnersForSnippet(snippet, workflowName) {
  try {
    // Do not set owners here if already present; we recompute from mapping
    const label = (snippet && snippet.label) ? String(snippet.label) : '';
    const stripBracketSuffix = (s) => (typeof s === 'string' ? s.replace(/\s*\[[^\]]+\]\s*$/, '').trim() : s);
    const cleaned = stripBracketSuffix(label || '');
    let ownersArr = findOwnerForLabel(cleaned) || findOwnerForLabel(label || '') || [];
    // Normalize and dedupe
    const normalized = [];
    const seenOwners = new Set();
    for (const o of (ownersArr || [])) {
      if (!o) continue;
      const id = o.id || undefined;
      const name = o.name || undefined;
      const k = `${id || ''}|${name || ''}`;
      if (seenOwners.has(k)) continue;
      seenOwners.add(k);
      normalized.push({ id, name });
    }
    // Apply infra policy: if no owners or test missing/NA, infra is sole owner
    const testName = (snippet && snippet.test) ? snippet.test : 'NA';
    const hadOriginalOwners = normalized.length > 0;
    if (!testName || testName === 'NA') {
      // Missing/NA test: infra is the only owner; preserve original owners for report context
      snippet.owner = [DEFAULT_INFRA_OWNER];
      snippet.owner_source = hadOriginalOwners ? 'infra_due_to_missing_test' : 'infra_due_to_missing_test_no_original';
      if (hadOriginalOwners) snippet.original_owners = normalized;
    } else if (!hadOriginalOwners) {
      // No mapping owners found: default to infra only
      snippet.owner = [DEFAULT_INFRA_OWNER];
      snippet.owner_source = 'infra_due_to_no_owner';
    } else {
      // Normal case: keep resolved owners
      snippet.owner = normalized;
      snippet.owner_source = 'resolved_mapping';
    }
  } catch (_) {
    snippet.owner = [DEFAULT_INFRA_OWNER];
    snippet.owner_source = 'infra_due_to_error';
  }
}

function renderErrorsTable(errorSnippets) {
  if (!Array.isArray(errorSnippets) || errorSnippets.length === 0) {
    return '<p><em>No error info found</em></p>';
  }
  let __maxJobLen = 0;
  const rows = errorSnippets.map(obj => {
    const rawLabel = (obj && obj.label) ? String(obj.label) : '';
    const rawSnippet = (obj && obj.snippet) ? String(obj.snippet) : '';
    // Rendering-only: rely on precomputed job/test/owner; do minimal fallbacks for display
    let jobName = (obj && typeof obj.job === 'string') ? obj.job : '';
    let testName = (obj && typeof obj.test === 'string') ? obj.test : '';
    if (!jobName) jobName = rawLabel || '';
    if (!testName) testName = 'NA';
    const errorForDisplay = rawSnippet;

    // Aesthetics: drop trailing bracketed status like [failure] or [error]
    jobName = jobName.replace(/\s*\[[^\]]+\]\s*$/, '');
    // Prevent excessive wrapping: use non-breaking hyphens and keep slash groups together
    const jobDisplay = String(jobName)
      .replace(/-/g, '\u2011')
      .replace(/\s\/\s/g, '\u00A0/\u00A0');
    // HTML escape the content
    const escapeHtml = (str) => String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    const jobEsc = escapeHtml(jobDisplay).replace(/\r?\n/g, ' ⇥ ');


    // THE FOLLOWING SECTION IS PURELY FOR DISPLAY AESTHETICS. NO IMPORTANT LOGIC IS HERE
    // ========================================================
    // Build a forced two-line HTML for the job label by inserting a <br/> near the midpoint.
    // Preference: split at '/' boundary; fallback to nearest space; otherwise hard split at midpoint.
    let jobHtml = jobEsc;
    try {
      const plain = String(jobDisplay).replace(/\r?\n/g, ' ').trim();
      const mid = Math.floor(plain.length / 2);
      let breakIdx = -1;
      // Prefer splitting at slash
      const slashIdx = plain.indexOf('/');
      if (slashIdx !== -1) {
        breakIdx = slashIdx + 1; // keep slash at end of first line
      } else {
        // Look for a breakable space near the midpoint (normal or NBSP)
        const window = 25;
        const right = plain.slice(mid, Math.min(plain.length, mid + window)).search(/[\u00A0\s]/);
        const left = plain.slice(Math.max(0, mid - window), mid).lastIndexOf(' ');
        const leftNbsp = plain.slice(Math.max(0, mid - window), mid).lastIndexOf('\u00A0');
        const leftBest = Math.max(left, leftNbsp);
        if (right !== -1) breakIdx = mid + right + 1;
        else if (leftBest !== -1) breakIdx = Math.max(1, Math.max(0, mid - window) + leftBest + 1);
      }
      if (breakIdx <= 0 || breakIdx >= plain.length) breakIdx = Math.max(1, mid);
      const part1 = plain.slice(0, breakIdx).trimEnd();
      const part2 = plain.slice(breakIdx).trimStart();
      jobHtml = `${escapeHtml(part1).replace(/\s/g, '\u00A0')}<br/>${escapeHtml(part2).replace(/\s/g, '\u00A0')}`;
    } catch (_) { /* keep fallback jobEsc */ }
    // Track longest job (by characters shown) to size the column later (2-line target ≈ len/2ch)
    try { __maxJobLen = Math.max(__maxJobLen, jobDisplay.length); } catch (_) { /* ignore */ }
    const testEsc = escapeHtml(testName).replace(/\r?\n/g, ' ⇥ ');
    const snippetOneLine = escapeHtml(errorForDisplay || '').replace(/\r?\n/g, ' ⇥ ');
    let ownerDisplay = 'no owner found';
    if (obj && Array.isArray(obj.owner) && obj.owner.length) {
      const names = obj.owner.map(o => (o && (o.name || o.id)) || '').filter(Boolean);
      if (names.length) ownerDisplay = names.join(', ');
      // If owner is infra-only due to missing test, surface original pipeline owner for human context
      if (obj.owner_source && String(obj.owner_source).startsWith('infra_due_to_missing_test') && Array.isArray(obj.original_owners) && obj.original_owners.length) {
        const origNames = obj.original_owners.map(o => (o && (o.name || o.id)) || '').filter(Boolean);
        if (origNames.length) ownerDisplay = `${DEFAULT_INFRA_OWNER.name} (pipeline owner: ${origNames.join(', ')})`;
      }
    }
    const ownerEsc = escapeHtml(ownerDisplay);
    // Force exactly two lines: compute a break point (prefer comma, else space near middle), NBSP around words
    let ownerHtml = ownerEsc;
    try {
      const plainO = String(ownerDisplay).replace(/\r?\n/g, ' ').trim();
      const lenO = plainO.length;
      const midO = Math.floor(lenO / 2);
      let breakO = -1;
      const commaIdx = plainO.indexOf(',');
      if (commaIdx !== -1 && commaIdx < lenO - 1) breakO = commaIdx + 1;
      if (breakO === -1) {
        const window = 24;
        const after = plainO.slice(midO, Math.min(lenO, midO + window)).search(/[\u00A0\s]/);
        const before = plainO.slice(Math.max(0, midO - window), midO).lastIndexOf(' ');
        const beforeNb = plainO.slice(Math.max(0, midO - window), midO).lastIndexOf('\u00A0');
        const bestBefore = Math.max(before, beforeNb);
        if (after !== -1) breakO = midO + after + 1; else if (bestBefore !== -1) breakO = Math.max(1, Math.max(0, midO - window) + bestBefore + 1);
      }
      if (breakO <= 0 || breakO >= lenO) breakO = Math.max(1, midO);
      const p1 = plainO.slice(0, breakO).trimEnd();
      const p2 = plainO.slice(breakO).trimStart();
      ownerHtml = `${escapeHtml(p1).replace(/\s/g, '\u00A0')}<br/>${escapeHtml(p2).replace(/\s/g, '\u00A0')}`;
    } catch (_) { /* keep fallback ownerEsc */ }
    return `<tr><td style="white-space: normal; word-break: normal; overflow-wrap: break-word; hyphens: none;"><div style="line-height:1.35; min-height: calc(2 * 1.35em); max-height: calc(2 * 1.35em); overflow: auto; white-space: nowrap;">${jobHtml}</div></td><td style="white-space: normal; word-break: normal; overflow-wrap: break-word; hyphens: none;">${testEsc}</td><td style="white-space: nowrap; word-break: keep-all; overflow: hidden;"><div style="line-height:1.35; min-height: calc(2 * 1.35em); max-height: calc(2 * 1.35em); overflow: auto; white-space: nowrap;">${ownerHtml}</div></td><td style="white-space: normal; word-break: normal; overflow-wrap: break-word; hyphens: none;">${snippetOneLine}</td></tr>`;
  }).join('\n');
  // Compute dynamic width for Job column based on longest job label; cap to reasonable bounds
  const jobWidthCh = (() => {
    const est = Math.ceil((__maxJobLen || 0) / 2) + 2; // two-line target + padding
    const minCh = 28; // lower bound so it doesn't get too narrow
    const maxCh = 88; // upper bound to avoid crowding other columns
    return Math.max(minCh, Math.min(maxCh, est));
  })();
  // ========================================================

  return [
    '<table style="table-layout: auto; width: 100%;">',
    '<thead>',
    '<tr>',
    `<th style="width: ${jobWidthCh}ch;">Job</th>`,
    '<th>Test</th>',
    '<th>Owner</th>',
    '<th>Error</th>',
    '</tr>',
    '</thead>',
    '<tbody>',
    rows,
    '</tbody>',
    '</table>',
    ''
  ].join('\n');
}

// Helper: derive a one-line description for a commit
function getCommitDescription(commit) {
  try {
    if (!commit) return '';
    const fromField = (typeof commit.description === 'string' && commit.description) || '';
    if (fromField) return fromField;
    const msg = typeof commit.message === 'string' ? commit.message : '';
    if (!msg) return '';
    const firstLine = String(msg).split(/\r?\n/)[0] || '';
    return firstLine;
  } catch (_) {
    return '';
  }
}

function renderCommitsTable(commits) {
  if (!Array.isArray(commits) || commits.length === 0) {
    return '<p><em>None</em></p>';
  }
  const escapeHtml = (str) => String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  const rows = commits.map(c => {
    const short = c.short || (c.sha ? c.sha.substring(0, 7) : '');
    const url = c.url || (c.sha ? `https://github.com/${github.context.repo.owner}/${github.context.repo.repo}/commit/${c.sha}` : undefined);
    const who = c.author_login ? `@${c.author_login}` : (c.author_name || 'unknown');
    const whoHtml = (c.author_login && c.author_url)
      ? `<a href="${escapeHtml(c.author_url)}">@${escapeHtml(c.author_login)}</a>`
      : escapeHtml(who);
    const shaHtml = url ? `<a href="${escapeHtml(url)}"><code>${escapeHtml(short)}</code></a>` : `<code>${escapeHtml(short)}</code>`;
    const descHtml = escapeHtml(getCommitDescription(c));
    return `<tr><td>${shaHtml}</td><td>${whoHtml}</td><td>${descHtml}</td></tr>`;
  }).join('\n');
  return [
    '<table>',
    '<thead>',
    '<tr><th>SHA</th><th>Author</th><th>Description</th></tr>',
    '</thead>',
    '<tbody>',
    rows,
    '</tbody>',
    '</table>',
    ''
  ].join('\n');
}

/**
 * Fetches PR information associated with a commit.
 *
 * @param {object} context - GitHub Actions context
 * @param {string} commitSha - Full SHA of the commit to look up
 * @returns {Promise<object>} Object containing:
 *   - prNumber: Markdown link to the PR (e.g., [#123](url))
 *   - prTitle: Title of the PR or EMPTY_VALUE if not found
 *   - prAuthor: GitHub username of the PR author or 'unknown'
 */
// Disabled: PR fetching via GitHub API removed for offline analysis
async function fetchPRInfo(_github, _context, _commitSha) {
  return { prNumber: EMPTY_VALUE, prTitle: EMPTY_VALUE, prAuthor: EMPTY_VALUE };
}

/**
 * Fetch commit author info for a commit SHA.
 * Returns GitHub login (if associated), author display name, and profile URL (if available).
 */
// Disabled: commit author fetch via GitHub API; will be inferred from commits index if present
async function fetchCommitAuthor(_commitSha) {
  return { login: undefined, name: undefined, htmlUrl: undefined };
}

/**
 * Download workflow run logs and extract up to N error snippets.
 * Returns an array of strings (snippets).
 */
async function fetchErrorSnippetsForRun(runId, maxSnippets = 50, logsDirPath = undefined, annotationsDirPath = undefined) {
  try {
    await core.startGroup(`Extracting error snippets for run ${runId}`); // start a group to log the error snippets (we log this to the console)
    let snippets = [];

    // Helper: filter out generic exit-code snippets when a job has more specific errors
    const isGenericExit = (s) => typeof s === 'string' && /^Process completed with exit code 1\.?$/i.test(s.trim()); // this usually has no meaningful information because a previous error that was already registered just caused this one to appear
    const filterGenericExitSnippets = (arr) => {
      try {
        const byJob = new Map();
        for (const sn of (arr || [])) {
          const job = (sn && sn.job) ? String(sn.job) : '';
          if (!byJob.has(job)) byJob.set(job, []);
          byJob.get(job).push(sn);
        }
        const out = [];
        for (const [job, list] of byJob.entries()) {
          const hasNonGeneric = list.some(sn => !isGenericExit(sn && sn.snippet)); // test to see if there is a non-generic exit snippet
          if (hasNonGeneric) {
            for (const sn of list) { if (!isGenericExit(sn && sn.snippet)) out.push(sn); } // if there is a non-generic exit snippet, add it to the output
          } else {
            out.push(...list); // if there is no non-generic exit snippet, add all the snippets to the output (no point in filtering)
          }
        }
        return out;
      } catch (_) { return arr || []; }
    };

    // If gtest logs are available for this run, parse them and return. This leaves pytest/others unchanged.
    try {
      const runLogsDir = logsDirPath || getGtestLogsDirForRunId(runId);
      if (runLogsDir && fs.existsSync(runLogsDir)) {
        const idxPath = path.join(runLogsDir, 'jobs.json');
        const extractDir = path.join(runLogsDir, 'extract');
        if (fs.existsSync(idxPath) && fs.existsSync(extractDir)) {
          core.info(`[GTEST] Using logs for run ${runId}: runLogsDir=${runLogsDir}`);
          core.info(`[GTEST] Index path: ${idxPath}`);
          let idx;
          try {
            idx = JSON.parse(fs.readFileSync(idxPath, 'utf8'));
          } catch (parseErr) {
            core.warning(`[GTEST] Failed to parse jobs.json for run ${runId}: ${parseErr.message}`);
            // Continue to other log sources
          }
          if (idx) {
            const jobs = Array.isArray(idx.jobs) ? idx.jobs : [];
            core.info(`[GTEST] Jobs detected: ${jobs.length}`);
            const out = [];
            // Helper to strip ANSI color codes and other escape sequences
            const stripAnsi = (s) => typeof s === 'string' ? s.replace(/\x1b\[[0-9;]*m/g, '') : s; // this is probably unnecessary but no harm in doing it

            for (const job of jobs) {
              const jobName = (job && job.name) ? String(job.name) : 'gtest';
              const files = Array.isArray(job.files) ? job.files : [];
              core.info(`[GTEST] Processing job: name='${jobName}', files=${files.length}`);
              let lastSeenTestName = undefined; // persist across files in this job
              for (const rel of files) {
                if (out.length >= maxSnippets) break;
                try {
                  const abs = path.join(runLogsDir, rel);
                  if (!fs.existsSync(abs)) { core.info(`[GTEST]   Skip missing file: ${abs}`); continue; }
                  const text = fs.readFileSync(abs, 'utf8');
                  const rawLines = text.split(/\r?\n/);
                  const lines = rawLines.map(l =>
                    stripAnsi(l)
                      .replace(/^\s*\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+/, '')
                      .replace(/^\s*\[[0-9]+,[0-9]+\]<[^>]+>:\s*/, '')
                  );
                  core.info(`[GTEST]   Parsing file: ${abs} (lines=${lines.length})`);
                  let currentTest = lastSeenTestName;
                  let capturing = false;
                  let buf = [];
                  let snippetsAdded = 0;

                  const flush = (lineNo, iIndex) => {
                    if (!capturing) return;
                    const msg = buf.join('\n').trim();
                    if (msg) {
                      // If name unknown, try to infer by scanning forward a small window
                      let labelName = (currentTest && String(currentTest).trim()) || '';
                      if (!labelName) {
                        for (let j = iIndex + 1; j < Math.min(lines.length, iIndex + 200); j++) {
                          const l2 = lines[j];
                          const mFail = l2 && l2.match(/\[\s*FAILED\s*\]\s+(.+?)\s*$/);
                          if (mFail) { labelName = mFail[1]; break; }
                          const mRun2 = l2 && l2.match(/\[\s*RUN\s*\]\s+(.+?)\s*$/);
                          if (mRun2) { labelName = mRun2[1]; break; }
                        }
                      }
                      if (!labelName) labelName = lastSeenTestName || 'unknown gtest';
                      // Do not assign owners here; only return raw snippet with inferred job/test via label for hint
                      out.push({ label: `${jobName}: ${labelName}`, job: jobName, test: labelName, snippet: msg });
                      snippetsAdded++;
                      core.info(`[GTEST]     Added snippet for '${labelName}' (len=${msg.length}) @ line ${lineNo}`);
                    }
                    capturing = false; buf = [];
                  };

                  for (let i = 0; i < lines.length && out.length < maxSnippets; i++) {
                    const rawLine = lines[i];
                    const line = stripAnsi(rawLine);
                    // Detect new test start
                    // Match RUN anywhere in the line (timestamps/prefixes may precede it)
                    const runMatch = line && line.match(/\[\s*RUN\s*\]\s+(.+?)\s*$/);
                    if (runMatch) {
                      // Starting a new test block; stop any capture in progress (without emitting)
                      capturing = false; buf = [];
                      currentTest = runMatch[1];
                      lastSeenTestName = currentTest;
                      core.info(`[GTEST]     RUN -> '${currentTest}' @ line ${i+1}`);
                      continue;
                    }

                    // Look for info/backtrace block boundaries
                    const infoMatch = line && line.match(/^\s*info:\s*(.*)$/i);
                    if (!capturing && infoMatch) {
                      const lower = line.toLowerCase();
                      const btIdx = lower.indexOf('backtrace:');
                      if (btIdx !== -1) {
                        // Same-line info..backtrace
                        const infoIdx = lower.indexOf('info:');
                        const between = line.substring(infoIdx + 5, btIdx).replace(/^\s*|\s*$/g, '');
                        if (between) buf.push(between);
                        flush(i+1, i);
                        continue;
                      }
                      // Start multi-line capture
                      capturing = true;
                      buf = [];
                      if (infoMatch[1]) buf.push(infoMatch[1]);
                      core.info(`[GTEST]       info: begin capture @ line ${i+1}`);
                      continue;
                    }

                    if (capturing) {
                      const lower = line.toLowerCase();
                      const btIdx2 = lower.indexOf('backtrace:');
                      if (btIdx2 !== -1) {
                        const head = line.substring(0, btIdx2).replace(/^\s*|\s*$/g, '');
                        if (head) buf.push(head);
                        flush(i+1, i);
                        continue;
                      }
                      buf.push(line);
                    }
                  }
                  core.info(`[GTEST]   File complete: snippets_added=${snippetsAdded}`);
                } catch (errFile) {
                  core.info(`[GTEST]   Failed parsing file '${rel}': ${errFile && errFile.message || String(errFile)}`);
                }
              }
            }
            if (out.length) {
              // Dedupe by normalized label+snippet to avoid repeated rows
              const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
              const seen = new Set();
              const unique = [];
              for (const e of out) {
                const key = `${norm(e && e.label)}|${norm(e && e.snippet)}`;
                if (!seen.has(key)) { seen.add(key); unique.push(e); }
              }
              core.info(`[GTEST] Total snippets before cap: ${out.length}, after dedupe: ${unique.length}`);
              snippets = unique.slice(0, Math.max(1, Math.min(maxSnippets, unique.length)));
              // Apply generic-exit filter per job
              snippets = filterGenericExitSnippets(snippets);
              core.info(`[GTEST] Collected ${snippets.length} gtest snippet(s) from logs for run ${runId} (after filtering)`);
              // Do not attach owners here; ownership resolution is performed later
              return snippets;
            } else {
              core.info(`[GTEST] No snippets extracted from logs for run ${runId}`);
            }
          }
        }
      }
    } catch (e) {
      core.warning(`Failed gtest log parsing for run ${runId}: ${e.message}`);
    }

    // If other logs (non-gtest) are available for this run, extract job names from file names
    // Don't parse the logs, just use the file names to determine job names
    try {
      const otherLogsDir = getOtherLogsDirForRunId(runId);
      if (otherLogsDir && fs.existsSync(otherLogsDir)) {
        const logsListPath = path.join(otherLogsDir, 'logs-list.json');
        if (fs.existsSync(logsListPath)) {
          core.info(`[OTHER LOGS] Using logs list for run ${runId}: otherLogsDir=${otherLogsDir}`);
          let logsListData;
          try {
            logsListData = JSON.parse(fs.readFileSync(logsListPath, 'utf8'));
          } catch (parseErr) {
            core.warning(`[OTHER LOGS] Failed to parse logs-list.json for run ${runId}: ${parseErr.message}`);
            // Continue to annotations
          }
          if (logsListData) {
            const files = Array.isArray(logsListData.files) ? logsListData.files : [];
            core.info(`[OTHER LOGS] Files detected: ${files.length}`);

            // Extract job names from file paths (e.g., "extract/1_job-name/step.txt" -> "job-name")
            const jobNamesSet = new Set();
            for (const filePath of files) {
              try {
                // Parse the path to extract job name
                // Expected format: extract/<step>_<job-name>/<file>.txt
                const parts = filePath.split(path.sep);
                if (parts.length >= 2) {
                  const folderName = parts[1]; // e.g., "1_job-name"
                  // Remove leading step number and underscore
                  const jobName = folderName.replace(/^\d+_/, '').trim();
                  if (jobName) {
                    jobNamesSet.add(jobName);
                  }
                }
              } catch (_) { /* ignore */ }
            }

            // Create snippets with job names, but blank test and informative error message
            const out = [];
            for (const jobName of jobNamesSet) {
              out.push({
                label: jobName,
                job: jobName,
                test: '',
                snippet: 'currently aggregate-workflow-data is not able to parse these kinds of errors'
              });
              if (out.length >= maxSnippets) break;
            }

            if (out.length > 0) {
              core.info(`[OTHER LOGS] Collected ${out.length} job name(s) from logs for run ${runId}`);
              snippets = out;
              // Apply generic-exit filter per job
              snippets = filterGenericExitSnippets(snippets);
              return snippets;
            } else {
              core.info(`[OTHER LOGS] No job names extracted from logs for run ${runId}`);
            }
          }
        }
      }
    } catch (e) {
      core.warning(`Failed other log processing for run ${runId}: ${e.message}`);
    }

    // Primary (default): use annotations if available (non-gtest)
    // If we already collected snippets from logs, skip annotations entirely
    if (Array.isArray(snippets) && snippets.length > 0) {
      return snippets;
    }
    const annDir = annotationsDirPath || getAnnotationsDirForRunId(runId);
    if (annDir && fs.existsSync(annDir)) {
      try {
        const filePath = path.join(annDir, 'annotations.json');
        if (fs.existsSync(filePath)) {
          const raw = fs.readFileSync(filePath, 'utf8');
          const arr = JSON.parse(raw);
          if (Array.isArray(arr)) {
            const seen = new Set();
            for (const a of arr) {
              const job = a.job_name || '';
              const title = a.title || '';
              const level = a.annotation_level || '';
              const message = a.message || '';
              const levelLc = String(level).toLowerCase();
              if (levelLc !== 'failure' && levelLc !== 'error') continue; // ignore warnings
              const msgTrim = String(message).trim();
              const label = `${job}${title ? `: ${title}` : ''} [${level}]`;
              const dedupeKey = `${job}|${title}|${levelLc}|${msgTrim}`;
              if (seen.has(dedupeKey)) continue;
              seen.add(dedupeKey);
              snippets.push({ label, job, snippet: message });
              if (snippets.length >= maxSnippets) break;
            }
          }
        }
      } catch (e) {
        core.warning(`Failed reading annotations for run ${runId}: ${e.message}`);
      }
    }
    // Fallback: none (we no longer parse logs). If needed, could fallback to logs here.
    // Do not attach owners here; ownership resolution is performed later

    core.info(`Total snippets collected from annotations: ${snippets.length}`);
    // Apply generic-exit filter per job before returning
    return filterGenericExitSnippets(snippets || []);
  } catch (e) {
    core.info(`Failed to obtain run logs for ${runId}: ${e.message}`);
    return [];
  } finally {
    core.endGroup();
  }
}

// END OF CODE FOR ERROR HANDLING

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
 *   - newestGoodSha: Short SHA of the most recent successful run (e.g., `a1b2c3d`)
 *   - newestBadSha: Short SHA of the most recent failing run (e.g., `e4f5g6h`)
 */
function findGoodBadCommits(scheduledMainRuns, context) {
  let newestGoodSha = EMPTY_VALUE;
  let newestBadSha = EMPTY_VALUE;
  let foundGood = false;
  let foundBad = false;

  for (const run of scheduledMainRuns) {
    if (!foundGood && run.conclusion === 'success') { // find the most recent successful run
      const shortSha = run.head_sha.substring(0, SHA_SHORT_LENGTH);
      newestGoodSha = `[\`${shortSha}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${run.head_sha})`; // set the newest good sha to the short sha of the most recent successful run
      foundGood = true;
    }
    if (!foundBad && run.conclusion !== 'success') { // find the most recent failing run
      const shortSha = run.head_sha.substring(0, SHA_SHORT_LENGTH);
      newestBadSha = `[\`${shortSha}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${run.head_sha})`; // set the newest bad sha to the short sha of the most recent failing run
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
 * @param {object} context - GitHub Actions context
 * @returns {Promise<object>} Object containing run information
 */
async function getLastRunInfo(mainBranchRuns, context) {
  const lastMainRun = mainBranchRuns[0];
  if (!lastMainRun) { // if there is no last main run, return the empty values
    return {
      status: EMPTY_VALUE,
      sha: EMPTY_VALUE,
      run: EMPTY_VALUE,
      pr: EMPTY_VALUE,
      title: EMPTY_VALUE,
      newestGoodSha: EMPTY_VALUE,
      newestBadSha: EMPTY_VALUE
    };
  }

  const prInfo = await fetchPRInfo(null, context, lastMainRun.head_sha); // fetch the PR info for the last main run
  // Current approach: filter by event type
  const mainRuns = mainBranchRuns.filter(r => r.event === lastMainRun.event || r.event === 'workflow_dispatch'); // get the relevant main runs for finding good and bad commits
  // Alternative approach: include all runs on main branch
  // const mainRuns = mainBranchRuns;
  const { newestGoodSha, newestBadSha } = findGoodBadCommits(mainRuns, context); // find the newest good and bad commits

  return {
    status: lastMainRun.conclusion === 'success' ? SUCCESS_EMOJI : FAILURE_EMOJI,
    sha: `[\`${lastMainRun.head_sha.substring(0, SHA_SHORT_LENGTH)}\`](https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${lastMainRun.head_sha})`, // get the short sha hyperlink for the latest main run
    run: `[Run](${lastMainRun.html_url})${lastMainRun.run_attempt > 1 ? ` (#${lastMainRun.run_attempt})` : ''}`, // get the run hyperlink for the latest main run
    pr: prInfo.prNumber, // get the PR number for the latest main run
    title: prInfo.prTitle, // get the PR title for the latest main run
    newestGoodSha, // get the newest good sha for the latest main run
    newestBadSha: lastMainRun.conclusion !== 'success' ? newestBadSha : EMPTY_VALUE // if the last main run is successful, don't bother showing the most recent failure
  };
}

/**
 * Generates summary tables for push and scheduled workflows.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} context - GitHub Actions context
 * @returns {Promise<string>} Markdown table for all workflows
 */
async function generateSummaryBox(grouped, context) {
  const rows = [];
  const escapeHtml = (str) => String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');

  // Helper to convert markdown links to HTML
  const mdToHtml = (md) => {
    if (!md || md === EMPTY_VALUE) return escapeHtml(md);
    // Match markdown link format: [text](url) or [`text`](url)
    return md.replace(/\[`([^`]+)`\]\(([^)]+)\)/g, (_, text, url) => `<a href="${escapeHtml(url)}"><code>${escapeHtml(text)}</code></a>`)
             .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, url) => `<a href="${escapeHtml(url)}">${escapeHtml(text)}</a>`);
  };

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
    const runInfo = await getLastRunInfo(mainBranchRuns, context); // get the last run info for the workflow

    // basically, create the statistics table for the workflow. these are the columns that are displayed in the summary box.
    const row = `<tr>
<td><a href="${escapeHtml(workflowLink)}">${escapeHtml(name)}</a></td>
<td>${escapeHtml(stats.eventTypes || 'unknown')}</td>
<td>${stats.totalRuns}</td>
<td>${stats.successfulRuns}</td>
<td>${escapeHtml(stats.successRate)}</td>
<td>${escapeHtml(stats.uniqueSuccessRate)}</td>
<td>${escapeHtml(stats.retryRate)}</td>
<td>${escapeHtml(runInfo.status)}</td>
<td>${mdToHtml(runInfo.sha)}</td>
<td>${mdToHtml(runInfo.run)}</td>
<td>${mdToHtml(runInfo.pr)}</td>
<td>${escapeHtml(runInfo.title)}</td>
<td>${mdToHtml(runInfo.newestBadSha)}</td>
<td>${mdToHtml(runInfo.newestGoodSha)}</td>
</tr>`;
    rows.push(row);
  }

  return [
    '## Workflow Summary',
    '<table>',
    '<thead>',
    '<tr><th>Workflow</th><th>Event Type(s)</th><th>Total Runs</th><th>Successful Runs</th><th>Success Rate</th><th>Unique Success Rate</th><th>Retry Rate</th><th>Last Run on <code>main</code></th><th>Last SHA</th><th>Last Run</th><th>Last PR</th><th>PR Title</th><th>Newest Bad SHA</th><th>Newest Good SHA</th></tr>',
    '</thead>',
    '<tbody>',
    ...rows,
    '</tbody>',
    '</table>',
    ''
  ].join('\n');
}

/**
 * Builds the complete markdown report.
 *
 * @param {Map<string, Array<object>>} grouped - Map of workflow names to their runs
 * @param {object} context - GitHub Actions context
 * @returns {Promise<string>} Complete markdown report
 */
async function buildReport(grouped, context) {
  const days = parseInt(core.getInput('days') || DEFAULT_LOOKBACK_DAYS, 10); // get the number of days to look back for workflow data
  const timestamp = new Date().toISOString(); // get the timestamp for the report
  return [
    `# Workflow Summary (Last ${days} Days) - Generated at ${timestamp}\n`,
    await generateSummaryBox(grouped, context),
    '\n## Column Descriptions\n',
    '<p>A unique run represents a single workflow execution, which may have multiple retry attempts. For example, if a workflow fails and is retried twice, this counts as one unique run with three attempts (initial run + two retries).</p>\n',
    '\n### Success Rate Calculations\n',
    '<p>The success rates are calculated based on unique runs (not including retries in the denominator):</p>\n',
    '<ul>',
    '<li><strong>Success Rate</strong>: (Number of unique runs that eventually succeeded / Total number of unique runs) × 100%',
    '  <ul><li>Example: 3 successful unique runs out of 5 total unique runs = 60% success rate</li></ul>',
    '</li>',
    '<li><strong>Unique Success Rate</strong>: (Number of unique runs that succeeded on first try / Total number of unique runs) × 100%',
    '  <ul><li>Example: 1 unique run succeeded on first try out of 5 total unique runs = 20% unique success rate</li></ul>',
    '</li>',
    '<li><strong>Retry Rate</strong>: (Number of successful unique runs that needed retries / Total number of successful unique runs) × 100%',
    '  <ul><li>Example: 2 successful unique runs needed retries out of 3 total successful unique runs = 66.67% retry rate</li></ul>',
    '</li>',
    '</ul>\n',
    '<p><strong>Note:</strong> Unique Success Rate + Retry Rate does not equal 100% because they measure different things:</p>',
    '<ul>',
    '<li>Unique Success Rate is based on all unique runs</li>',
    '<li>Retry Rate is based only on successful unique runs</li>',
    '</ul>\n',
    '<table>',
    '<thead>',
    '<tr><th>Column</th><th>Description</th></tr>',
    '</thead>',
    '<tbody>',
    '<tr><td>Workflow</td><td>Name of the workflow with link to its GitHub Actions page</td></tr>',
    '<tr><td>Event Type(s)</td><td>Types of events that trigger this workflow (e.g., push, pull_request, schedule)</td></tr>',
    '<tr><td>Total Runs</td><td>Total number of workflow runs including all retry attempts (e.g., 1 unique run with 2 retries = 3 total runs)</td></tr>',
    '<tr><td>Successful Runs</td><td>Number of unique workflow runs that eventually succeeded, regardless of whether they needed retries</td></tr>',
    '<tr><td>Success Rate</td><td>Percentage of unique workflow runs that eventually succeeded (e.g., 3/5 unique runs succeeded = 60%)</td></tr>',
    '<tr><td>Unique Success Rate</td><td>Percentage of unique workflow runs that succeeded on their first attempt without needing retries (e.g., 1/5 unique runs succeeded on first try = 20%)</td></tr>',
    '<tr><td>Retry Rate</td><td>Percentage of successful unique runs that needed retries to succeed (e.g., of 3 successful unique runs, 2 needed retries = 66.67%)</td></tr>',
    '<tr><td>Last Run on <code>main</code></td><td>Status of the most recent run on the main branch (✅ for success, ❌ for failure)</td></tr>',
    '<tr><td>Last SHA</td><td>Short SHA of the most recent run on main</td></tr>',
    '<tr><td>Last Run</td><td>Link to the most recent run on main, with attempt number if applicable</td></tr>',
    '<tr><td>Last PR</td><td>Link to the PR associated with the most recent run, if any</td></tr>',
    '<tr><td>PR Title</td><td>Title of the PR associated with the most recent run, if any</td></tr>',
    '<tr><td>Newest Bad SHA</td><td>Short SHA of the most recent failing run on main (only shown if last run failed)</td></tr>',
    '<tr><td>Newest Good SHA</td><td>Short SHA of the most recent successful run on main</td></tr>',
    '</tbody>',
    '</table>'
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
  cutoffDate.setDate(cutoffDate.getDate() - days); // set the cutoff date to the number of days ago

  return runs.filter(run => {
    const runDate = new Date(run.created_at);
    return runDate >= cutoffDate; // return the runs that are within the date range
  });
}

// START MORE ERROR HANDLING CODE

/**
 * Collect commits between two SHAs on the default branch (main), inclusive of endSha.
 * Returns an array of { sha, short, url, author_login, author_name, author_url }.
 * Note: Uses compareCommits, which is base..head; base is typically the success commit, head is the failed run commit.
 */
// Offline commits lookup: read from commits index built by fetch step
let __commitsIndex = undefined;
function loadCommitsIndex(commitsPath) {
  if (!commitsPath) return undefined;
  try {
    if (!fs.existsSync(commitsPath)) return undefined;
    const raw = fs.readFileSync(commitsPath, 'utf8');
    const arr = JSON.parse(raw);
    if (Array.isArray(arr)) return arr;
  } catch (_) { /* ignore */ }
  return undefined;
}

function listCommitsBetweenOffline(context, startShaExclusive, endShaInclusive) { // get all the commits between two SHAs on the default branch (main), inclusive of endSha
  if (!Array.isArray(__commitsIndex) || __commitsIndex.length === 0) return [];
  const commits = __commitsIndex;
  const startIdx = commits.findIndex(c => c.sha === startShaExclusive);
  const endIdx = commits.findIndex(c => c.sha === endShaInclusive);
  if (endIdx === -1) return [];
  // We want commits strictly after start and up to and including end
  const from = startIdx === -1 ? 0 : (startIdx + 1); // basically, if the start index is -1, then we start from the beginning of the commits array
  const to = endIdx;
  const slice = commits.slice(from, to + 1);
  return slice.map(c => ({
    sha: c.sha,
    short: (c.short || (c.sha ? c.sha.substring(0, SHA_SHORT_LENGTH) : '')),
    url: c.url || (c.sha ? `https://github.com/${context.repo.owner}/${context.repo.repo}/commit/${c.sha}` : ''),
    author_login: c.author_login,
    author_name: c.author_name,
    author_url: c.author_url,
    message: c.message,
    description: getCommitDescription(c),
  }));
}

// END MORE ERROR HANDLING CODE

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
      __annotationsIndexMap = loadAnnotationsIndexFromFile(annotationsIndexPath);
      if (__annotationsIndexMap && __annotationsIndexMap.size) {
        core.info(`Loaded annotations index with ${__annotationsIndexMap.size} entries from ${annotationsIndexPath}`);
      } else if (annotationsIndexPath) {
        core.info(`No valid entries found in annotations index file at ${annotationsIndexPath}`);
      }
    }

    if (gtestLogsIndexPath) {
      __gtestLogsIndexMap = loadLogsIndexFromFile(gtestLogsIndexPath);
      if (__gtestLogsIndexMap && __gtestLogsIndexMap.size) {
        core.info(`Loaded gtest logs index with ${__gtestLogsIndexMap.size} entries from ${gtestLogsIndexPath}`);
      } else if (gtestLogsIndexPath) {
        core.info(`No valid entries found in gtest logs index file at ${gtestLogsIndexPath}`);
      }
    }

    if (otherLogsIndexPath) {
      __otherLogsIndexMap = loadLogsIndexFromFile(otherLogsIndexPath);
      if (__otherLogsIndexMap && __otherLogsIndexMap.size) {
        core.info(`Loaded other logs index with ${__otherLogsIndexMap.size} entries from ${otherLogsIndexPath}`);
      } else if (otherLogsIndexPath) {
        core.info(`No valid entries found in other logs index file at ${otherLogsIndexPath}`);
      }
    }

    // Load commits index (optional)
    __commitsIndex = loadCommitsIndex(commitsPath) || [];
    core.info(`Loaded commits index entries: ${Array.isArray(__commitsIndex) ? __commitsIndex.length : 0}`);

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
            const mainBranchRuns = filteredRuns
              .filter(r => r.head_branch === 'main')
              .sort((a, b) => new Date(b.created_at) - new Date(a.created_at)); // sort the runs by date within the window
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
        const mainRuns = runs
          .filter(r => r.head_branch === 'main') // filter the runs by the main branch
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at)); //iterate through the runs and sort them by date within the window
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
      const mainBranchRuns = runs
        .filter(r => r.head_branch === 'main')
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      const latest = mainBranchRuns[0];
      if (!latest) return null;
      return latest.conclusion === 'success' ? 'success' : 'failure';
    }; // compute the latest conclusion of the pipeline run
    const computeLatestRunInfo = (runs) => {
      const mainBranchRuns = runs
        .filter(r => r.head_branch === 'main')
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
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
    const getMainWindowRuns = (runs) => runs
      .filter(r => r.head_branch === 'main')
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

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
              return ['<details>',`<summary>${workflowName}</summary>`,'',content, errorsList,'</details>',''].join('\n');
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
            return ['<details>',`<summary>${workflowName}</summary>`,'',content, errorsList, commitsList,'</details>',''].join('\n');
          }
          // If no first_failed_run_url, just return a collapsed workflow name
          return ['<details>',`<summary>${workflowName}</summary>`,'','  - No failure details available','</details>',''].join('\n');
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
              return ['<details>',`<summary>${workflowName}</summary>`,'',content, errorsList,'</details>',''].join('\n');
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
            return ['<details>',`<summary>${workflowName}</summary>`,'',content, errorsList, commitsList,'</details>',''].join('\n');
          }
          // If no first_failed_run_url, just return a collapsed workflow name
          return ['<details>',`<summary>${workflowName}</summary>`,'','  - No failure details available','</details>',''].join('\n');
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
