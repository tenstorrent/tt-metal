// Script to scan pipeline_reorg files and update owners.json with missing entries
const fs = require('fs');
const path = require('path');

/**
 * Convert YAML filename to workflow prefix format used in owners.json
 * Examples:
 *   t3k_unit_tests.yaml -> t3000-unit-tests
 *   t3k_demo_tests.yaml -> t3000-demo-tests
 *   ops/sanity/tests.yaml -> ops-sanity
 */
function getWorkflowPrefix(filePath) {
  const relativePath = path.relative(path.join(__dirname, '../../..', 'tests/pipeline_reorg'), filePath);
  const dirParts = path.dirname(relativePath).split(path.sep).filter(p => p !== '.');
  const fileName = path.basename(relativePath, '.yaml');

  // Convert t3k_* to t3000-*
  let prefix = fileName.replace(/^t3k_/, 't3000-').replace(/_/g, '-');

  // If there's a directory, prepend it
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
 * Simple YAML parser for our specific format (array of objects with name and owner_id)
 * Handles the format:
 *   - name: t3k_ttmetal_tests
 *     owner_id: ULMEPM2MA # Sean Nijjar
 */
function parseYamlFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const entries = [];
  const lines = content.split('\n');

  let currentEntry = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Skip comments and empty lines
    if (trimmed.startsWith('#') || trimmed === '') {
      continue;
    }

    // Check if we're starting an array entry (starts with -)
    if (trimmed.startsWith('-')) {
      // Save previous entry if exists
      if (currentEntry && currentEntry.name && currentEntry.owner_id) {
        entries.push(currentEntry);
      }
      currentEntry = {};

      // Check if there's inline content after the dash
      const afterDash = trimmed.substring(1).trim();
      if (afterDash && afterDash.includes(':')) {
        const colonIndex = afterDash.indexOf(':');
        const key = afterDash.substring(0, colonIndex).trim();
        let value = afterDash.substring(colonIndex + 1).trim();
        // Remove quotes
        if ((value.startsWith('"') && value.endsWith('"')) ||
            (value.startsWith("'") && value.endsWith("'"))) {
          value = value.slice(1, -1);
        }
        currentEntry[key] = value;
      }
      continue;
    }

    // Parse key-value pairs (indented with spaces)
    if (line.startsWith('  ') && trimmed.includes(':')) {
      const colonIndex = trimmed.indexOf(':');
      const key = trimmed.substring(0, colonIndex).trim();
      let value = trimmed.substring(colonIndex + 1).trim();

      // Remove quotes if present
      if ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }

      if (currentEntry) {
        currentEntry[key] = value;
      }
    }
  }

  // Don't forget the last entry
  if (currentEntry && currentEntry.name && currentEntry.owner_id) {
    entries.push(currentEntry);
  }

  return entries;
}

/**
 * Scan a YAML file and extract job entries with owners
 */
function scanPipelineFile(filePath) {
  const jobs = [];
  try {
    const entries = parseYamlFile(filePath);

    for (const entry of entries) {
      if (entry.name && entry.owner_id) {
        const ownerName = extractOwnerName(entry.owner_id);
        if (ownerName) {
          // Extract just the ID part (before #)
          const ownerId = entry.owner_id.split('#')[0].trim();
          jobs.push({
            name: entry.name,
            ownerId: ownerId,
            ownerName: ownerName
          });
        }
      }
    }
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error.message);
  }

  return jobs;
}

/**
 * Recursively scan all YAML files in pipeline_reorg directory
 */
function scanAllPipelineFiles(pipelineReorgDir) {
  const jobs = [];

  function scanDir(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        scanDir(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.yaml')) {
        const fileJobs = scanPipelineFile(fullPath);
        for (const job of fileJobs) {
          jobs.push({
            ...job,
            workflowPrefix: getWorkflowPrefix(fullPath)
          });
        }
      }
    }
  }

  if (fs.existsSync(pipelineReorgDir)) {
    scanDir(pipelineReorgDir);
  }

  return jobs;
}

/**
 * Check if a job name already exists in owners.json
 * We check both the normalized name and the workflow-prefixed version
 * Also check the original name with underscores for backward compatibility
 */
function jobExistsInOwners(jobName, normalizedJobName, workflowPrefix, ownersData) {
  if (!ownersData || !ownersData.contains) {
    return false;
  }

  const prefixedName = `${workflowPrefix} / ${normalizedJobName}`;
  const prefixedNameWithUnderscores = `${workflowPrefix} / ${jobName}`;

  for (const entry of ownersData.contains) {
    const component = entry['job-name-component'];
    if (!component) continue;

    // Check various formats:
    // 1. Exact match with normalized name (spaces)
    // 2. Exact match with workflow prefix and normalized name
    // 3. Exact match with original name (underscores)
    // 4. Ends with normalized name (for workflow-prefixed entries)
    // 5. Contains the job name (for fuzzy matching)
    if (component === normalizedJobName ||
        component === jobName ||
        component === prefixedName ||
        component === prefixedNameWithUnderscores ||
        component.endsWith(` / ${normalizedJobName}`) ||
        component.endsWith(` / ${jobName}`) ||
        component.includes(normalizedJobName) ||
        component.includes(jobName)) {
      return true;
    }
  }

  return false;
}

/**
 * Update owners.json with new entries
 */
function updateOwnersJson(ownersPath, pipelineReorgDir) {
  // Load existing owners.json
  let ownersData = { contains: [] };
  if (fs.existsSync(ownersPath)) {
    try {
      const content = fs.readFileSync(ownersPath, 'utf8');
      ownersData = JSON.parse(content);
      if (!ownersData.contains) {
        ownersData.contains = [];
      }
    } catch (error) {
      console.error(`Error reading ${ownersPath}:`, error.message);
      return;
    }
  }

  // Scan pipeline files
  const pipelineJobs = scanAllPipelineFiles(pipelineReorgDir);
  console.log(`Found ${pipelineJobs.length} jobs in pipeline files`);

  // Find jobs that don't exist in owners.json
  const newEntries = [];
  const seenJobs = new Set(); // Track jobs we've already processed

  // Track which job-name-components we've added during this run
  const addedWithPrefix = new Set();
  const addedWithoutPrefix = new Set();

  for (const job of pipelineJobs) {
    const normalizedName = normalizeJobName(job.name);
    const workflowPrefix = job.workflowPrefix;
    const jobKey = `${workflowPrefix}:${normalizedName}`;

    // Skip if we've already processed this job (for this prefix/name pair)
    if (seenJobs.has(jobKey)) {
      continue;
    }

    const prefixedComponent = `${workflowPrefix} / ${normalizedName}`;
    const unprefixedComponent = normalizedName;

    // Check if either version exists in owners.json
    const existsWithPrefix = jobExistsInOwners(job.name, normalizedName, workflowPrefix, ownersData);
    const existsWithoutPrefix = jobExistsInOwners(job.name, normalizedName, '', ownersData);

    const alreadyAddedWithPrefix = addedWithPrefix.has(prefixedComponent);
    const alreadyAddedWithoutPrefix = addedWithoutPrefix.has(unprefixedComponent);

    const shouldAddWithPrefix = !existsWithPrefix && !alreadyAddedWithPrefix;
    const shouldAddWithoutPrefix = !existsWithoutPrefix && !alreadyAddedWithoutPrefix;

    if (!shouldAddWithPrefix && !shouldAddWithoutPrefix) {
      continue;
    }

    seenJobs.add(jobKey);

    if (shouldAddWithPrefix) {
      // Add entry with workflow prefix
      newEntries.push({
        'job-name-component': prefixedComponent,
        owner: {
          id: job.ownerId,
          name: job.ownerName
        }
      });
      addedWithPrefix.add(prefixedComponent);
    }

    if (shouldAddWithoutPrefix) {
      // Add entry without workflow prefix
      newEntries.push({
        'job-name-component': unprefixedComponent,
        owner: {
          id: job.ownerId,
          name: job.ownerName
        }
      });
      addedWithoutPrefix.add(unprefixedComponent);
    }
  }

  if (newEntries.length === 0) {
    console.log('No new entries to add');
    return;
  }

  console.log(`Adding ${newEntries.length} new entries to owners.json`);

  // Read the original file to preserve exact formatting
  const originalContent = fs.readFileSync(ownersPath, 'utf8');

  // Find the insertion point - we need to insert before the closing ] of the "contains" array
  // The last entry is:    { "job-name-component": "...", "owner": {...} }
  // Followed by:  ]

  const lines = originalContent.split('\n');

  // Find the line with the closing bracket of the contains array
  let closingBracketLineIndex = -1;
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i].trim() === ']') {
      closingBracketLineIndex = i;
      break;
    }
  }

  if (closingBracketLineIndex === -1) {
    console.error('Could not find closing bracket in owners.json');
    return;
  }

  // Find the last entry line (the one before the closing bracket)
  // It should be the last line with a closing brace
  let lastEntryEndLineIndex = closingBracketLineIndex - 1;
  for (let i = closingBracketLineIndex - 1; i >= 0; i--) {
    const trimmed = lines[i].trim();
    if (trimmed === '}' || trimmed === '},') {
      lastEntryEndLineIndex = i;
      break;
    }
  }

  // Build new entries with exact formatting matching the existing style
  // Format:    { "job-name-component": "...", "owner": { "id": "...", "name": "..." } },
  const newEntriesLines = [];
  for (let i = 0; i < newEntries.length; i++) {
    const entry = newEntries[i];
    const entryLine = `    { "job-name-component": ${JSON.stringify(entry['job-name-component'])}, "owner": { "id": ${JSON.stringify(entry.owner.id)}, "name": ${JSON.stringify(entry.owner.name)} } }`;
    // Add comma to all entries except the very last one (which will be before the closing bracket)
    // But since we're adding multiple entries, only the last of our new entries shouldn't have a comma
    if (i < newEntries.length - 1) {
      newEntriesLines.push(entryLine + ',');
    } else {
      // Check if there are more entries after (in afterInsert) - if the closing bracket is next, no comma
      newEntriesLines.push(entryLine);
    }
  }

  // Insert new entries after the last entry
  // The last entry is at lastEntryEndLineIndex, and it doesn't have a comma
  // We need to add a comma to it, then add our new entries

  const beforeInsertLines = lines.slice(0, lastEntryEndLineIndex);
  const lastEntryLine = lines[lastEntryEndLineIndex];
  const afterInsert = lines.slice(lastEntryEndLineIndex + 1).join('\n');

  // Add comma to the last existing entry
  const lastEntryWithComma = lastEntryLine + ',';

  // Build the new content
  const newContent = beforeInsertLines.join('\n') + '\n' +
                     lastEntryWithComma + '\n' +
                     newEntriesLines.join('\n') + '\n' +
                     afterInsert;

  fs.writeFileSync(ownersPath, newContent, 'utf8');

  // Write back to file
  console.log(`Successfully updated ${ownersPath} with ${newEntries.length} new entries`);
}

// Main execution
if (require.main === module) {
  const ownersPath = path.join(__dirname, 'owners.json');
  const pipelineReorgDir = path.join(__dirname, '../../..', 'tests/pipeline_reorg');

  updateOwnersJson(ownersPath, pipelineReorgDir);
}

module.exports = { updateOwnersJson, scanAllPipelineFiles };
