// Data Loading Module
// Handles constants, index loading, and commit loading

const fs = require('fs');

// Constants
const DEFAULT_LOOKBACK_DAYS = 15;
const SHA_SHORT_LENGTH = 7;
const SUCCESS_RATE_DECIMAL_PLACES = 2;
const SUCCESS_EMOJI = '✅';
const FAILURE_EMOJI = '❌';
const EMPTY_VALUE = '—';
// Default owner override when a test name cannot be extracted
const DEFAULT_INFRA_OWNER = { id: 'S0985AN7TC5', name: 'metal infra team' };

// Module-level index state
let __annotationsIndexMap = undefined;
let __gtestLogsIndexMap = undefined;
let __otherLogsIndexMap = undefined;
let __lastSuccessTimestamps = undefined;
let __commitsIndex = undefined;

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

function loadLastSuccessTimestamps(filePath) {
  try {
    if (!filePath || !fs.existsSync(filePath)) return undefined;
    const raw = fs.readFileSync(filePath, 'utf8');
    const json = JSON.parse(raw);
    const map = new Map();
    if (json && typeof json === 'object' && !Array.isArray(json)) {
      for (const [k, v] of Object.entries(json)) {
        if (v && typeof v === 'object') map.set(String(k), v);
      }
      return map;
    }
  } catch (_) { /* ignore */ }
  return undefined;
}

function getTimeSinceLastSuccess(workflowName) {
  try {
    if (!__lastSuccessTimestamps) return EMPTY_VALUE;
    const info = __lastSuccessTimestamps.get(workflowName);
    if (!info) return EMPTY_VALUE;

    if (info.never_succeeded) {
      return 'Never';
    }

    if (!info.timestamp) return EMPTY_VALUE;

    const lastSuccessDate = new Date(info.timestamp);
    const now = new Date();
    const daysSince = Math.floor((now - lastSuccessDate) / (1000 * 60 * 60 * 24));

    if (daysSince === 0) return 'Today';
    if (daysSince === 1) return '1 day ago';
    return `${daysSince} days ago`;
  } catch (_) {
    return EMPTY_VALUE;
  }
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

function listCommitsBetweenOffline(context, startShaExclusive, endShaInclusive) {
  if (!Array.isArray(__commitsIndex) || __commitsIndex.length === 0) return [];
  const commits = __commitsIndex;
  const startIdx = commits.findIndex(c => c.sha === startShaExclusive);
  const endIdx = commits.findIndex(c => c.sha === endShaInclusive);
  if (endIdx === -1) return [];
  // We want commits strictly after start and up to and including end
  const from = startIdx === -1 ? 0 : (startIdx + 1);
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

// Setters for module-level state
function setAnnotationsIndexMap(map) {
  __annotationsIndexMap = map;
}

function setGtestLogsIndexMap(map) {
  __gtestLogsIndexMap = map;
}

function setOtherLogsIndexMap(map) {
  __otherLogsIndexMap = map;
}

function setLastSuccessTimestamps(map) {
  __lastSuccessTimestamps = map;
}

function setCommitsIndex(index) {
  __commitsIndex = index;
}

module.exports = {
  // Constants
  DEFAULT_LOOKBACK_DAYS,
  SHA_SHORT_LENGTH,
  SUCCESS_RATE_DECIMAL_PLACES,
  SUCCESS_EMOJI,
  FAILURE_EMOJI,
  EMPTY_VALUE,
  DEFAULT_INFRA_OWNER,
  // Index loading functions
  loadAnnotationsIndexFromFile,
  getAnnotationsDirForRunId,
  loadLogsIndexFromFile,
  getGtestLogsDirForRunId,
  getOtherLogsDirForRunId,
  loadLastSuccessTimestamps,
  getTimeSinceLastSuccess,
  // Commit functions
  getCommitDescription,
  loadCommitsIndex,
  listCommitsBetweenOffline,
  // State setters
  setAnnotationsIndexMap,
  setGtestLogsIndexMap,
  setOtherLogsIndexMap,
  setLastSuccessTimestamps,
  setCommitsIndex,
};
