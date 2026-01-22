// Error Processing Module
// Handles ownership mapping, error extraction, and error rendering

const core = require('@actions/core');
const fs = require('fs');
const path = require('path');
const { DEFAULT_INFRA_OWNER, EMPTY_VALUE, getGtestLogsDirForRunId, getOtherLogsDirForRunId, getAnnotationsDirForRunId } = require('./data-loading');

// Owners mapping cache
let __ownersMapping = undefined;

function loadOwnersMapping() {
  if (__ownersMapping !== undefined) return __ownersMapping;
  try {
    const ownersPath = path.join(__dirname, '..', 'owners.json');
    if (fs.existsSync(ownersPath)) {
      const raw = fs.readFileSync(ownersPath, 'utf8');
      __ownersMapping = JSON.parse(raw);
    } else {
      __ownersMapping = null;
    }
  } catch (_) {
    __ownersMapping = null;
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
    // Apply infra policy: if no owners or test missing/NA, add infra as owner
    const testName = (snippet && snippet.test) ? snippet.test : 'NA';
    const hadOriginalOwners = normalized.length > 0;
    if (!testName || testName === 'NA') {
      // Missing/NA test: add infra as owner (alongside original owners if any)
      const infraKey = `${DEFAULT_INFRA_OWNER.id || ''}|${DEFAULT_INFRA_OWNER.name || ''}`;
      const hasInfra = normalized.some(o => `${o.id || ''}|${o.name || ''}` === infraKey);
      if (hadOriginalOwners) {
        snippet.original_owners = normalized;
        // Add infra if not already present
        snippet.owner = hasInfra ? normalized : [...normalized, DEFAULT_INFRA_OWNER];
      } else {
        snippet.owner = [DEFAULT_INFRA_OWNER];
      }
      snippet.owner_source = hadOriginalOwners ? 'infra_due_to_missing_test' : 'infra_due_to_missing_test_no_original';
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

/**
 * Download workflow run logs and extract up to N error snippets.
 * Returns an array of strings (snippets).
 */
async function fetchErrorSnippetsForRun(runId, maxSnippets = 50, logsDirPath = undefined, annotationsDirPath = undefined) {
  try {
    await core.startGroup(`Extracting error snippets for run ${runId}`);
    let snippets = [];

    // Helper: filter out generic exit-code snippets when a job has more specific errors
    const isGenericExit = (s) => typeof s === 'string' && /^Process completed with exit code 1\.?$/i.test(s.trim());
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
          const hasNonGeneric = list.some(sn => !isGenericExit(sn && sn.snippet));
          if (hasNonGeneric) {
            for (const sn of list) { if (!isGenericExit(sn && sn.snippet)) out.push(sn); }
          } else {
            out.push(...list);
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
            const stripAnsi = (s) => typeof s === 'string' ? s.replace(/\x1b\[[0-9;]*m/g, '') : s;

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

    // If other logs (non-gtest) are available for this run, use job names from GitHub API
    // Don't parse the logs, just use the stored job names from the API
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
            // Use job names from GitHub API if available (preferred method)
            let jobNamesSet = new Set();
            if (Array.isArray(logsListData.job_names) && logsListData.job_names.length > 0) {
              // Use actual job names from GitHub API
              for (const jobName of logsListData.job_names) {
                if (jobName && typeof jobName === 'string') {
                  // Remove .txt suffix if present (shouldn't happen with API names, but be safe)
                  let cleanedName = jobName.trim().replace(/\.txt$/i, '').trim();
                  // Replace " _ " (space-underscore-space) with " / " to match GitHub Actions job name format
                  cleanedName = cleanedName.replace(/\s+_\s+/g, ' / ');
                  if (cleanedName) {
                    jobNamesSet.add(cleanedName);
                  }
                }
              }
              core.info(`[OTHER LOGS] Using ${jobNamesSet.size} job name(s) from GitHub API for run ${runId}`);
            } else {
              // Fallback: extract job names from file paths (legacy method, less reliable)
              const files = Array.isArray(logsListData.files) ? logsListData.files : [];
              core.info(`[OTHER LOGS] No API job names found, falling back to file path extraction: ${files.length} files`);
              for (const filePath of files) {
                try {
                  // Parse the path to extract job name
                  // Expected format: extract/<step>_<job-name>/<file>.txt
                  // Note: GitHub Actions folder names sometimes include .txt suffix, sometimes not
                  const parts = filePath.split(path.sep);
                  if (parts.length >= 2) {
                    const folderName = parts[1]; // e.g., "1_job-name" or "1_job-name.txt"
                    // Remove leading step number and underscore, then remove .txt suffix if present
                    let jobName = folderName.replace(/^\d+_/, '');
                    // Remove .txt extension if present (GitHub Actions sometimes includes it in folder names)
                    jobName = jobName.replace(/\.txt$/i, '');
                    // Replace " _ " (space-underscore-space) with " / " to match GitHub Actions job name format
                    jobName = jobName.replace(/\s+_\s+/g, ' / ').trim();
                    if (jobName) {
                      jobNamesSet.add(jobName);
                    }
                  }
                } catch (_) { /* ignore */ }
              }
              core.info(`[OTHER LOGS] Extracted ${jobNamesSet.size} job name(s) from file paths for run ${runId}`);
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
              core.info(`[OTHER LOGS] No job names found for run ${runId}`);
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

module.exports = {
  // Ownership functions
  loadOwnersMapping,
  normalizeOwners,
  extractSignificantTokens,
  getJobNameComponentTail,
  findOwnerForLabel,
  // Error handling functions
  inferJobAndTestFromSnippet,
  resolveOwnersForSnippet,
  fetchErrorSnippetsForRun,
  // Error rendering functions
  renderErrorsTable,
};
