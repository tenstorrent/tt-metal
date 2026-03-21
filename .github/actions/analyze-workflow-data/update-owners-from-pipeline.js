// Script to scan pipeline_reorg files and update owners.json with missing entries
const fs = require('fs');
const path = require('path');

/**
 * Convert YAML filename to workflow prefix format used in owners.json
 * Examples:
 *   t3k_unit_tests.yaml -> t3000-unit-tests
 *   t3k_demo_tests.yaml -> t3000-demo-tests
 */
function getWorkflowPrefix(filePath) {
  const relativePath = path.relative(path.join(__dirname, '../../..', 'tests/pipeline_reorg'), filePath);
  const dirParts = path.dirname(relativePath).split(path.sep).filter(p => p !== '.');
  const fileName = path.basename(relativePath, '.yaml');

  // Convert t3k_* to t3000-* then normalize underscores to hyphens
  let prefix = fileName.replace(/^t3k_/, 't3000-').replace(/_/g, '-');

  // If there's a directory, prepend it with hyphen separator
  if (dirParts.length > 0) {
    prefix = dirParts.join('-') + '-' + prefix;
  }

  return prefix;
}

/**
 * Convert job name from pipeline format to owners.json format
 * Examples:
 *   t3k_ttmetal_tests -> t3k ttmetal tests
 *   t3k_LLM_falcon7b_model_perf_tests -> t3k LLM falcon7b model perf tests
 */
function normalizeJobName(jobName) {
  return jobName.replace(/_/g, ' ');
}

/**
 * Extract owner name from comment
 * Examples:
 *   "ULMEPM2MA # Sean Nijjar" -> "Sean Nijjar"
 *   "U045U3DEKM4 # Mohamed Bahnas (Aniruddha Tupe)" -> "Mohamed Bahnas"
 */
function extractOwnerName(comment) {
  if (!comment) return null;
  const match = comment.match(/#\s*(.+)/);
  if (match) {
    // Remove parenthetical notes like "(Aniruddha Tupe)"
    return match[1].replace(/\s*\([^)]*\)\s*$/, '').trim();
  }
  return null;
}

/**
 * Extract Slack user ID from an owner_id field value (before any # comment)
 * Examples:
 *   "ULMEPM2MA # Sean Nijjar" -> "ULMEPM2MA"
 *   "U045U3DEKM4" -> "U045U3DEKM4"
 */
function extractOwnerId(ownerIdField) {
  if (!ownerIdField) return null;
  // The owner_id may contain a comment after '#'; take only the part before it
  const parts = ownerIdField.split('#');
  return parts[0].trim() || null;
}

/**
 * Simple YAML parser for our specific format (array of objects with name and owner_id)
 * Handles the format:
 *   - name: t3k_ttmetal_tests
 *     owner_id: ULMEPM2MA # Sean Nijjar
 *
 * WHY custom parser: the pipeline YAML files use only a small subset of YAML
 * (flat arrays of {name, owner_id} objects). Avoiding a heavy yaml-parse
 * dependency keeps the action self-contained and fast.
 */
function parseYamlFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const entries = [];
  const lines = content.split('\n');

  let currentEntry = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Skip pure comment lines and empty lines
    if (trimmed.startsWith('#') || trimmed === '') {
      continue;
    }

    // Check if we're starting an array entry (starts with '-')
    if (trimmed.startsWith('-')) {
      // Save previous entry if it has both required fields
      if (currentEntry && currentEntry.name && currentEntry.owner_id) {
        entries.push(currentEntry);
      }
      currentEntry = {};

      // Check if there's inline content after the dash (e.g., "- name: foo")
      const afterDash = trimmed.substring(1).trim();
      if (afterDash && afterDash.includes(':')) {
        const colonIndex = afterDash.indexOf(':');
        const key = afterDash.substring(0, colonIndex).trim();
        // Preserve inline comments for owner_id parsing — do NOT strip '#' here
        let value = afterDash.substring(colonIndex + 1).trim();
        // Remove surrounding quotes only (not inline comments)
        if ((value.startsWith('"') && value.endsWith('"')) ||
            (value.startsWith("'") && value.endsWith("'"))) {
          value = value.slice(1, -1);
        }
        currentEntry[key] = value;
      }
      continue;
    }

    // Parse indented key-value pairs under the current entry
    if (line.startsWith('  ') && trimmed.includes(':')) {
      const colonIndex = trimmed.indexOf(':');
      const key = trimmed.substring(0, colonIndex).trim();
      // Preserve full value including inline comments (needed for owner_id + name extraction)
      let value = trimmed.substring(colonIndex + 1).trim();

      // Remove surrounding quotes if present, but only if they wrap the ENTIRE value
      // (i.e., no inline comment outside the quotes)
      if ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }

      if (currentEntry) {
        currentEntry[key] = value;
      }
    }
  }

  // Flush the last entry
  if (currentEntry && currentEntry.name && currentEntry.owner_id) {
    entries.push(currentEntry);
  }

  return entries;
}

/**
 * Scan a YAML file and extract job entries with owners.
 * Returns an array of { jobName, ownerId, ownerName } objects.
 *
 * WHY separate ownerId/ownerName extraction here: callers need the Slack ID
 * (for owners.json keys) separately from the human-readable name. Doing both
 * in one place avoids duplicating the '#' comment parsing logic.
 */
function scanPipelineFile(filePath) {
  const jobs = [];
  try {
    const entries = parseYamlFile(filePath);

    for (const entry of entries) {
      if (entry.name && entry.owner_id) {
        const ownerId = extractOwnerId(entry.owner_id);
        const ownerName = extractOwnerName(entry.owner_id);
        if (ownerId) {
          jobs.push({
            jobName: normalizeJobName(entry.name),
            ownerId,
            ownerName
          });
        }
      }
    }
  } catch (err) {
    // Log but don't throw — a single bad file should not abort the whole scan
    console.error(`Warning: failed to parse ${filePath}: ${err.message}`);
  }
  return jobs;
}

module.exports = {
  getWorkflowPrefix,
  normalizeJobName,
  extractOwnerName,
  extractOwnerId,
  parseYamlFile,
  scanPipelineFile,
};
