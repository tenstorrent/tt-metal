// Tests for update-owners-from-pipeline.js
// Focused on the three areas most likely to regress:
//   1. extractOwnerId / extractOwnerName comment parsing
//   2. parseYamlFile entry detection (including flush of last entry)
//   3. scanPipelineFile end-to-end with a temp file

const fs = require('fs');
const os = require('os');
const path = require('path');
const {
  extractOwnerName,
  extractOwnerId,
  parseYamlFile,
  scanPipelineFile,
  normalizeJobName,
  getWorkflowPrefix,
} = require('./update-owners-from-pipeline');

// ---------------------------------------------------------------------------
// Helper: write a temp YAML file and return its path
// ---------------------------------------------------------------------------
function writeTempYaml(content) {
  const filePath = path.join(os.tmpdir(), `pipeline_test_${Date.now()}_${Math.random().toString(36).slice(2)}.yaml`);
  fs.writeFileSync(filePath, content, 'utf8');
  return filePath;
}

// ---------------------------------------------------------------------------
// 1. extractOwnerId and extractOwnerName
// ---------------------------------------------------------------------------
describe('extractOwnerId', () => {
  test('extracts Slack ID before # comment', () => {
    expect(extractOwnerId('ULMEPM2MA # Sean Nijjar')).toBe('ULMEPM2MA');
  });

  test('extracts Slack ID when no comment present', () => {
    expect(extractOwnerId('U045U3DEKM4')).toBe('U045U3DEKM4');
  });

  test('returns null for empty / null input', () => {
    expect(extractOwnerId('')).toBeNull();
    expect(extractOwnerId(null)).toBeNull();
    expect(extractOwnerId(undefined)).toBeNull();
  });

  test('trims whitespace around the ID', () => {
    expect(extractOwnerId('  ULMEPM2MA  # Sean Nijjar')).toBe('ULMEPM2MA');
  });
});

describe('extractOwnerName', () => {
  test('extracts plain name after #', () => {
    expect(extractOwnerName('ULMEPM2MA # Sean Nijjar')).toBe('Sean Nijjar');
  });

  test('strips parenthetical note from name', () => {
    expect(extractOwnerName('U045U3DEKM4 # Mohamed Bahnas (Aniruddha Tupe)')).toBe('Mohamed Bahnas');
  });

  test('returns null when no # comment present', () => {
    expect(extractOwnerName('ULMEPM2MA')).toBeNull();
  });

  test('returns null for falsy input', () => {
    expect(extractOwnerName(null)).toBeNull();
    expect(extractOwnerName('')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// 2. parseYamlFile
// ---------------------------------------------------------------------------
describe('parseYamlFile', () => {
  test('parses a well-formed two-entry YAML file', () => {
    const yaml = [
      '- name: t3k_ttmetal_tests',
      '  owner_id: ULMEPM2MA # Sean Nijjar',
      '- name: t3k_demo_tests',
      '  owner_id: U045U3DEKM4 # Mohamed Bahnas (Aniruddha Tupe)',
    ].join('\n');

    const filePath = writeTempYaml(yaml);
    const entries = parseYamlFile(filePath);
    fs.unlinkSync(filePath);

    expect(entries).toHaveLength(2);
    expect(entries[0]).toEqual({ name: 't3k_ttmetal_tests', owner_id: 'ULMEPM2MA # Sean Nijjar' });
    expect(entries[1].name).toBe('t3k_demo_tests');
    // owner_id must preserve the inline comment so extractOwnerName can work
    expect(entries[1].owner_id).toContain('Mohamed Bahnas');
  });

  test('flushes the last entry correctly (no trailing newline)', () => {
    // Regression: last entry was dropped if file did not end with a blank line
    const yaml = '- name: single_job\n  owner_id: UABC123 # Alice';
    const filePath = writeTempYaml(yaml);
    const entries = parseYamlFile(filePath);
    fs.unlinkSync(filePath);

    expect(entries).toHaveLength(1);
    expect(entries[0].name).toBe('single_job');
  });

  test('skips entries missing name or owner_id', () => {
    const yaml = [
      '- name: has_name_only',
      '- owner_id: UABC123 # Bob',
      '- name: complete_entry',
      '  owner_id: UDEF456 # Carol',
    ].join('\n');

    const filePath = writeTempYaml(yaml);
    const entries = parseYamlFile(filePath);
    fs.unlinkSync(filePath);

    // Only the entry with both name and owner_id should appear
    expect(entries).toHaveLength(1);
    expect(entries[0].name).toBe('complete_entry');
  });
});

// ---------------------------------------------------------------------------
// 3. scanPipelineFile end-to-end
// ---------------------------------------------------------------------------
describe('scanPipelineFile', () => {
  test('returns normalised job records from a valid file', () => {
    const yaml = [
      '- name: t3k_ttmetal_tests',
      '  owner_id: ULMEPM2MA # Sean Nijjar',
      '- name: t3k_LLM_falcon7b_model_perf_tests',
      '  owner_id: U999XYZ # Jane Doe',
    ].join('\n');

    const filePath = writeTempYaml(yaml);
    const jobs = scanPipelineFile(filePath);
    fs.unlinkSync(filePath);

    expect(jobs).toHaveLength(2);

    // Job names must be normalised (underscores -> spaces)
    expect(jobs[0].jobName).toBe('t3k ttmetal tests');
    expect(jobs[1].jobName).toBe('t3k LLM falcon7b model perf tests');

    // Owner IDs must be extracted (no trailing comment text)
    expect(jobs[0].ownerId).toBe('ULMEPM2MA');
    expect(jobs[1].ownerId).toBe('U999XYZ');

    // Owner names must be extracted from the comment
    expect(jobs[0].ownerName).toBe('Sean Nijjar');
    expect(jobs[1].ownerName).toBe('Jane Doe');
  });

  test('returns empty array and does not throw for a non-existent file', () => {
    // WHY: a missing file should degrade gracefully, not crash the whole action
    const jobs = scanPipelineFile('/tmp/__nonexistent_pipeline_file__.yaml');
    expect(jobs).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// 4. Utility helpers
// ---------------------------------------------------------------------------
describe('normalizeJobName', () => {
  test('replaces all underscores with spaces', () => {
    expect(normalizeJobName('t3k_LLM_falcon7b')).toBe('t3k LLM falcon7b');
  });

  test('leaves already-normalised names unchanged', () => {
    expect(normalizeJobName('already normal')).toBe('already normal');
  });
});
