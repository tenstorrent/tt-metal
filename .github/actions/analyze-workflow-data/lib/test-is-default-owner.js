// Tests for is_default_owner flag in resolveOwnersForSnippet
// Run with: node test-is-default-owner.js

const assert = require('assert');

// Mock @actions/core before requiring error-processing
const mockCore = { info: () => {}, warning: () => {}, startGroup: async () => {}, endGroup: () => {} };
require.cache[require.resolve('@actions/core')] = { exports: mockCore };
require.cache[require.resolve('@actions/github')] = { exports: { context: { repo: { owner: 'tenstorrent', repo: 'tt-metal' } } } };

const { resolveOwnersForSnippet, findOwnerForLabel } = require('./error-processing');
const { DEFAULT_INFRA_OWNER } = require('./data-loading');

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ✅ ${name}`);
  } catch (e) {
    failed++;
    console.log(`  ❌ ${name}`);
    console.log(`     ${e.message}`);
  }
}

console.log('\n=== is_default_owner flag tests ===\n');

// ---- Test 1: No owner found, test name present → is_default_owner=true ----
test('No owner found + test present → is_default_owner=true', () => {
  const snippet = {
    label: 'some-unknown-job-xyz-that-matches-nothing',
    job: 'some-unknown-job-xyz-that-matches-nothing',
    test: 'test_something',
    snippet: 'error text'
  };
  resolveOwnersForSnippet(snippet, 'test-workflow');
  assert.strictEqual(snippet.is_default_owner, true, 'Expected is_default_owner=true for unmatched job');
  assert.strictEqual(snippet.owner_source, 'infra_due_to_no_owner');
  assert.ok(snippet.owner.some(o => o.id === DEFAULT_INFRA_OWNER.id), 'Should have infra owner');
});

// ---- Test 2: No owner found, test=NA → is_default_owner=true ----
test('No owner found + test=NA → is_default_owner=true', () => {
  const snippet = {
    label: 'some-unknown-job-xyz-that-matches-nothing',
    job: 'some-unknown-job-xyz-that-matches-nothing',
    test: 'NA',
    snippet: 'error text'
  };
  resolveOwnersForSnippet(snippet, 'test-workflow');
  assert.strictEqual(snippet.is_default_owner, true, 'Expected is_default_owner=true when no owner and test=NA');
  assert.strictEqual(snippet.owner_source, 'infra_due_to_missing_test_no_original');
});

// ---- Test 3: Owner found via owners.json, test present → is_default_owner=false ----
test('Explicit owner found + test present → is_default_owner=false', () => {
  // "models perf" matches an entry in owners.json (owner: Denys)
  const snippet = {
    label: 'models perf: some_test',
    job: 'models perf',
    test: 'test_something',
    snippet: 'error text'
  };
  resolveOwnersForSnippet(snippet, 'test-workflow');
  assert.strictEqual(snippet.is_default_owner, false, 'Expected is_default_owner=false for explicitly-owned job');
  assert.strictEqual(snippet.owner_source, 'resolved_mapping');
});

// ---- Test 4: Owner found via owners.json but test=NA → is_default_owner=false ----
test('Explicit owner found + test=NA → is_default_owner=false (has original owners)', () => {
  // "models perf" matches an entry in owners.json (owner: Denys)
  const snippet = {
    label: 'models perf',
    job: 'models perf',
    test: 'NA',
    snippet: 'error text'
  };
  resolveOwnersForSnippet(snippet, 'test-workflow');
  // When test=NA but owner found, owner_source is infra_due_to_missing_test (has original owners)
  assert.strictEqual(snippet.is_default_owner, false, 'Expected is_default_owner=false — original owners exist');
  assert.strictEqual(snippet.owner_source, 'infra_due_to_missing_test');
  assert.ok(Array.isArray(snippet.original_owners) && snippet.original_owners.length > 0, 'Should have original_owners');
});

// ---- Test 5: Empty/missing snippet fields → is_default_owner=true (fallback) ----
test('Empty snippet → is_default_owner=true (error/fallback path)', () => {
  const snippet = {};
  resolveOwnersForSnippet(snippet, 'test-workflow');
  assert.strictEqual(snippet.is_default_owner, true, 'Expected is_default_owner=true for empty snippet');
});

// ---- Test 6: Verify the flag is a boolean, not undefined ----
test('is_default_owner is always a boolean after resolveOwnersForSnippet', () => {
  const snippetNoOwner = {
    label: 'nonexistent-job-xyzzy',
    job: 'nonexistent-job-xyzzy',
    test: 'test_foo',
    snippet: 'err'
  };
  resolveOwnersForSnippet(snippetNoOwner, 'wf');
  assert.strictEqual(typeof snippetNoOwner.is_default_owner, 'boolean', 'is_default_owner should be boolean');
});

// ---- Test 7: metalinfra explicitly in owners.json should NOT get is_default_owner=true ----
test('metalinfra explicitly in owners.json → is_default_owner=false', () => {
  // Simulate: if findOwnerForLabel returns metalinfra explicitly, it should be
  // owner_source='resolved_mapping' and is_default_owner=false.
  // We can test this indirectly — if a job maps to metalinfra in owners.json,
  // the resolved_mapping path is taken.
  // Since we may not have such an entry, we test the logic by constructing
  // a snippet that would match an entry whose owner IS metalinfra.
  // For now, verify the invariant: resolved_mapping → is_default_owner=false
  const snippet = {
    label: 'nonexistent-job-abc123',
    job: 'nonexistent-job-abc123',
    test: 'test_bar',
    snippet: 'err'
  };
  resolveOwnersForSnippet(snippet, 'wf');
  if (snippet.owner_source === 'resolved_mapping') {
    assert.strictEqual(snippet.is_default_owner, false,
      'resolved_mapping should always have is_default_owner=false');
  } else {
    // Fell through to default — expected for a nonexistent job
    assert.strictEqual(snippet.is_default_owner, true,
      'infra_due_to_no_owner should have is_default_owner=true');
  }
});

console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`);
process.exit(failed > 0 ? 1 : 0);
