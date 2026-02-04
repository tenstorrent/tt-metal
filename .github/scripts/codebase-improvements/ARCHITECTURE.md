# Architecture Documentation

This document explains the design and implementation of the automated fix implementation system.

## Overview

The system provides an end-to-end workflow for:
1. Reproducing CI failures locally
2. Implementing fixes automatically
3. Creating PRs with the changes
4. Documenting the entire process

**Goal**: Enable anyone to trigger automated fix attempts for CI failures via a simple JSON configuration.

## Directory Structure

```
codebase-improvements/
├── info.json                              # ← USER INPUT: Configuration
├── run.sh                                 # ← ENTRY POINT: Main orchestrator
├── outputs/                               # ← OUTPUT: Execution reports
│   ├── .gitkeep
│   └── TEMPLATE_report.md                # Report format template
│
├── README.md                              # High-level documentation
├── QUICK_START.md                         # Getting started guide
├── EXAMPLE_usage.md                       # Practical examples
├── ARCHITECTURE.md                        # This file
│
├── reproduce-deterministic-failures/      # For 100% reproducible failures
│   ├── AI_PROMPT.md                      # Detailed AI instructions
│   ├── README.md                          # User documentation
│   └── timeout-in-datamovement/          # Example: existing test
│       ├── README.md
│       ├── logs/
│       └── tests/
│           └── test_gather_timeout_stress.py
│
├── reproduce-ND-failures/                 # For intermittent failures
│   ├── AI_PROMPT.md                      # Detailed AI instructions
│   ├── README.md                          # User documentation
│   └── performance-in-models/            # Example: perf regression
│       ├── logs/
│       └── tests/
│           └── test_mamba_perf_stress.py
│
├── implementing-features/                 # Fix implementation phase
│   ├── AI_PROMPT.md                      # Detailed AI instructions
│   └── README.md                          # User documentation
│
└── analyze-nd-failures/                   # Pattern analysis (existing)
    ├── README.md
    ├── analysis_prompt.md
    └── *.sh scripts
```

## System Components

### 1. User Input Layer

**File**: `info.json`

**Purpose**: Single configuration point for all inputs

**Schema**:
```json
{
  "deterministic": boolean,         // true = always fails, false = intermittent
  "url": string,                   // GitHub Actions run URL
  "prompt": string,                // User description of issue/desired fix
  "raw-logs": string,              // Alternative to url
  "existing-test-path": string     // Path to existing test (skips reproduction)
}
```

**Validation**:
- `prompt` is required
- `deterministic` is required
- Either `existing-test-path` OR (`url` OR `raw-logs`) must be provided
- If `existing-test-path` is provided, logs are not required
- If `existing-test-path` is not provided, exactly one of `url` or `raw-logs` must be provided

### 2. Orchestration Layer

**File**: `run.sh`

**Purpose**: Coordinates the entire workflow

**Responsibilities**:
1. Parse and validate `info.json`
2. Fetch logs from URL or use raw-logs
3. Route to appropriate reproduction workflow
4. Invoke AI for test creation
5. Create fix branch and invoke AI for implementation
6. Generate final report

**Design Decisions**:
- Bash script for simplicity and CI compatibility
- Manual Claude invocation points (not fully automated yet)
- Error handling with trap
- Color-coded logging
- Modular phases

**Execution Flow**:
```
parse_info_json()
    ↓
    ├─ existing-test-path provided? ─ YES → Set TEST_FILE, skip to implement_fix()
    │
    └─ NO → Continue to reproduction
        ↓
    fetch_logs_from_url() OR use_raw_logs()
        ↓
    create_reproduction_test()
        → Creates failure folder
        → Copies logs
        → Prompts user to invoke Claude
        → Verifies test created
        → Commits test to current branch
        ↓
implement_fix()
    → Creates fix branch from main
    → Cherry-picks test (or uses existing-test-path)
    → Prompts user to invoke Claude
    → Checks for PR creation
    → Returns to original branch
    ↓
finalize_run()
    → Finds or creates report
    → Cleans up temp files
    → Displays summary
```

### 3. Reproduction Layer

**Purpose**: Create tests that reproduce failures

**Components**:
- `reproduce-deterministic-failures/`: For consistent failures
- `reproduce-ND-failures/`: For race conditions and intermittent issues

**Workflow**:
1. AI reads logs in `<failure-name>/logs/`
2. AI finds original test code
3. AI creates minimal/stress test in `<failure-name>/tests/`
4. AI verifies test reproduces failure
5. AI documents approach in `<failure-name>/README.md`

**Key Differences**:

| Aspect | Deterministic | Non-Deterministic |
|--------|---------------|-------------------|
| Test type | Minimal reproduction | Stress test |
| Iterations | 1 | 50-100+ |
| Parallelism | Sequential | `pytest -n auto` |
| Time limit | Seconds | Up to 5 minutes |
| Goal | Easy debugging | Amplify failure |

### 4. Implementation Layer

**Purpose**: Analyze root cause and implement fixes

**Components**:
- `implementing-features/AI_PROMPT.md`: Detailed fix instructions
- `implementing-features/README.md`: User documentation

**Workflow**:
1. Run reproduction test to confirm failure
2. Analyze root cause
3. Create fix branch from main
4. Copy test to fix branch
5. Implement fix iteratively
6. Verify test passes (5x)
7. Remove test from branch
8. Create draft PR
9. Write execution report

**Time Budget**: 15 minutes
- Root cause analysis: 2 min
- Branch setup: 1 min
- Fix implementation: 8 min
- PR preparation: 2 min
- PR creation: 2 min

### 5. Reporting Layer

**Purpose**: Document execution results

**Output Location**: `outputs/YYYY-MM-DD_HH-MM-SS_<description>.md`

**Report Sections**:
1. **Summary**: Status and outcome
2. **Input Configuration**: Original info.json
3. **Phase 1**: Reproduction test details
4. **Phase 2**: Root cause analysis
5. **Phase 3**: Fix implementation
6. **Phase 4**: PR details
7. **Next Steps**: Actions needed
8. **Relevant Developers**: Who to contact

**Status Levels**:
- ✅ Success: Full automation completed
- ⚠️ Partial: Some progress made
- ❌ Failed: Could not complete

## AI Prompt Design

### General Principles

1. **Comprehensive**: Cover all scenarios and edge cases
2. **Structured**: Step-by-step with clear checkpoints
3. **Time-bounded**: Explicit time limits for each phase
4. **Failure-aware**: Detailed guidance on giving up gracefully
5. **Examples**: Show good and bad approaches

### Reproduction Prompts

**Deterministic (`reproduce-deterministic-failures/AI_PROMPT.md`)**:
- Focus on minimalism (under 50 lines)
- Fast execution (seconds)
- Clear expected vs actual behavior
- Easy debugging

**Non-Deterministic (`reproduce-ND-failures/AI_PROMPT.md`)**:
- Focus on amplification strategies
- Parallel execution
- High iteration counts
- 5-minute time limit
- Background execution with monitoring

### Implementation Prompt

**File**: `implementing-features/AI_PROMPT.md`

**Structure**:
1. **Checklist**: Required information before starting
2. **Step-by-Step**: Detailed process for each phase
3. **Failure Modes**: How to handle each failure type
4. **Time Management**: Strict timeline enforcement
5. **Give-up Protocol**: When and how to stop

**Key Features**:
- Iterative approach (test after each change)
- Stability verification (5x test runs)
- Clean commit preparation
- PR description template
- Developer contact recommendations

## Data Flow

```
┌─────────────────┐
│   info.json     │ ← User Configuration
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│    run.sh       │ ← Orchestrator
└────────┬────────┘
         │
         ├─→ fetch_logs() → .temp_logs_<RUN_ID>/
         │
         ├─→ create_reproduction_test()
         │   ├─→ Create failure folder
         │   ├─→ Copy logs
         │   ├─→ [PAUSE] Invoke Claude
         │   ├─→ Verify test
         │   └─→ Git commit
         │
         ├─→ implement_fix()
         │   ├─→ Create fix branch
         │   ├─→ Cherry-pick test
         │   ├─→ [PAUSE] Invoke Claude
         │   ├─→ Check for PR
         │   └─→ Return to original branch
         │
         └─→ finalize_run()
             ├─→ Find/create report
             ├─→ Clean temp files
             └─→ Display summary

Output:
┌─────────────────────────────────────┐
│ outputs/<timestamp>_<desc>.md       │ ← Execution Report
│ reproduce-*/failure-name/           │ ← Reproduction Test
│ GitHub PR (draft)                   │ ← Fix Implementation
└─────────────────────────────────────┘
```

## Branching Strategy

```
main ─────────────────────────┬──── (latest production code)
                               │
                               │
dev-branch ────┬───────────────┘
               │
               │ (reproduction test created here)
               │
               ├─ test committed
               │
               └─ stays on dev-branch
                  (test not included in PR)


fix/<name> ────────────────────
               │
               ├─ created from main
               │
               ├─ test cherry-picked
               │
               ├─ fix implemented
               │
               ├─ test removed before PR
               │
               └─ PR created → main
```

**Key Points**:
- Reproduction test stays on development branch
- Fix branch created from latest main
- Test is copied to fix branch for development
- Test is removed before creating PR
- PR only contains the fix, not the test

## Extension Points

### 1. Automatic Log Fetching

Currently, URL fetching is a placeholder. To implement:

```bash
fetch_logs_from_url() {
    RUN_ID=$(echo "$URL" | grep -oP 'runs/\K\d+')
    gh run view "$RUN_ID" --log > "$LOGS_DIR/ci_log.txt"
    gh run view "$RUN_ID" --json jobs > "$LOGS_DIR/jobs.json"
}
```

### 2. Full Automation

Remove manual Claude invocation points by:
- Using Claude API directly
- Implementing retry logic
- Adding progress monitoring

```bash
invoke_claude_for_reproduction() {
    # Use Claude API to read AI_PROMPT.md and complete task
    # Poll for completion
    # Verify test created
}
```

### 3. GitHub Actions Integration

Create workflow that triggers on:
- Manual dispatch with inputs
- Repeated test failures
- Pull request comments (e.g., `/auto-fix`)

```yaml
on:
  workflow_dispatch:
    inputs:
      prompt: ...
      deterministic: ...
      ci_run_url: ...
```

### 4. Success Tracking

Add metrics collection:

```bash
# Track success rates
grep "Status:" outputs/*.md | \
  awk '{print $NF}' | \
  sort | uniq -c
```

Create dashboard showing:
- Success rate over time
- Most common failure types
- Average fix time
- Most helpful developers

### 5. Learning System

Analyze successful fixes to improve prompts:
- Extract common patterns
- Identify effective strategies
- Update AI prompts with learnings

## Security Considerations

### Code Execution
- AI-generated code should be reviewed before merging
- Draft PRs prevent accidental merges
- Tests run in isolated environments

### Credential Handling
- GitHub token needed for `gh` CLI
- Should use GitHub Actions secrets in CI
- Never commit credentials

### Branch Protection
- Fix branches don't bypass branch protection
- PRs still require reviews
- CI must pass before merge

## Testing Strategy

### Unit Testing

Test individual components:

```bash
# Test info.json parsing
test_parse_info_json() {
    echo '{"deterministic": true, ...}' > test_info.json
    # Run parser
    # Assert correct values extracted
}

# Test log fetching
test_fetch_logs() {
    # Mock GitHub API
    # Verify logs downloaded
}
```

### Integration Testing

Test end-to-end with known failures:

```bash
# Use existing timeout-in-datamovement test
./run.sh
# Verify:
# - Test is found
# - Fix branch created
# - PR created
# - Report generated
```

### Smoke Testing

Quick validation before deployment:

```bash
# Test basic functionality
./run.sh --validate
# Checks:
# - info.json exists
# - Required tools installed (gh, python, git)
# - Directory structure correct
```

## Performance Optimization

### Parallel Phases

Some phases could run in parallel:
- Log analysis + Test code search
- Multiple fix approaches simultaneously
- Parallel test verification

### Caching

Cache frequently accessed data:
- Test file locations
- Common operation implementations
- Developer contact information

### Time Management

Enforce strict timeouts:
- Reproduction: 5 minutes max
- Implementation: 15 minutes max
- Total: 23 minutes max

If approaching limit:
- Finalize current work
- Write report with progress
- Recommend continuation

## Future Enhancements

### Short Term (1-3 months)
- [ ] Implement GitHub Actions log fetching
- [ ] Add success rate tracking
- [ ] Create GitHub Actions workflow
- [ ] Add more example tests

### Medium Term (3-6 months)
- [ ] Full automation (remove manual steps)
- [ ] Web UI for configuration
- [ ] Integration with error tracking
- [ ] Automatic expert recommendation based on files changed

### Long Term (6+ months)
- [ ] Learning system that improves over time
- [ ] Multi-strategy fix attempts in parallel
- [ ] Automatic regression testing
- [ ] Integration with code review tools

## Troubleshooting

### Common Issues

**Issue**: Script fails to parse info.json
- **Cause**: Invalid JSON syntax
- **Fix**: Validate JSON with `python3 -m json.tool info.json`

**Issue**: Claude cannot reproduce failure
- **Cause**: Insufficient logs or environment mismatch
- **Fix**: Provide more detailed logs, check environment variables

**Issue**: Fix doesn't work
- **Cause**: Root cause misidentified or approach incorrect
- **Fix**: Review AI's analysis, try alternative approach

**Issue**: PR creation fails
- **Cause**: GitHub CLI not authenticated or permissions issue
- **Fix**: Run `gh auth login`, check repository permissions

### Debug Mode

Enable verbose logging:

```bash
DEBUG=1 ./run.sh
```

Generates additional output:
- Full command execution
- Variable values at each step
- Detailed error messages

## Maintenance

### Regular Tasks

**Weekly**:
- Review generated reports
- Analyze success patterns
- Update developer contacts

**Monthly**:
- Update AI prompts based on learnings
- Review and archive old reports
- Check for new tool versions

**Quarterly**:
- Comprehensive system review
- Performance optimization
- Documentation updates

### Version Control

**Prompt Versioning**:
- Track changes to AI_PROMPT.md files
- Document why changes were made
- A/B test prompt improvements

**Report Schema**:
- Version report format
- Ensure backward compatibility
- Provide migration scripts if needed

## Contributing

### Adding New Reproduction Strategies

1. Create new subfolder in `reproduce-*/`
2. Add AI_PROMPT.md with strategy
3. Add README.md with examples
4. Update main README.md
5. Test with real failures

### Improving AI Prompts

1. Identify failure patterns
2. Propose prompt improvements
3. Test with historical failures
4. Measure success rate change
5. Update if improvement > 10%

### Adding New Examples

1. Reproduce a failure manually
2. Document the process
3. Create example in EXAMPLE_usage.md
4. Add test to appropriate reproduce-*/ folder

---

**System Version**: 1.0
**Last Updated**: 2026-02-04
**Maintainer**: DevOps Team
