#!/bin/bash

# Main orchestration script for automated fix implementation
# This script coordinates the entire process from failure reproduction to PR creation

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_ID="$TIMESTAMP"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Function to parse info.json
parse_info_json() {
    log_info "Parsing info.json..."

    if [ ! -f "info.json" ]; then
        log_error "info.json not found!"
        exit 1
    fi

    # Extract fields using python
    DETERMINISTIC=$(python3 -c "import json; print(json.load(open('info.json'))['deterministic'])")
    URL=$(python3 -c "import json; print(json.load(open('info.json')).get('url', ''))")
    PROMPT=$(python3 -c "import json; print(json.load(open('info.json')).get('prompt', ''))")
    RAW_LOGS=$(python3 -c "import json; print(json.load(open('info.json')).get('raw-logs', ''))")
    EXISTING_TEST_PATH=$(python3 -c "import json; print(json.load(open('info.json')).get('existing-test-path', ''))")

    log_info "Deterministic: $DETERMINISTIC"
    log_info "URL: ${URL:-<empty>}"
    log_info "Prompt: $PROMPT"
    log_info "Raw logs: ${RAW_LOGS:+<provided>}${RAW_LOGS:-<empty>}"
    log_info "Existing test: ${EXISTING_TEST_PATH:-<empty>}"

    # Validate input
    if [ -z "$PROMPT" ]; then
        log_error "Prompt is empty in info.json"
        exit 1
    fi

    # Validate we have error context (logs or URL)
    # ALWAYS required - even with existing test, Claude needs to know the error
    if [ -z "$URL" ] && [ -z "$RAW_LOGS" ]; then
        log_error "Both URL and raw-logs are empty."
        log_error "Claude needs error context to implement fixes."
        log_error "Provide either 'url' or 'raw-logs' with the error message."
        exit 1
    fi

    # If existing test path is provided, just validate it exists
    if [ -n "$EXISTING_TEST_PATH" ]; then
        log_info "Existing test path provided, will skip reproduction phase"

        # Validate test file exists
        if [ ! -f "$EXISTING_TEST_PATH" ]; then
            log_error "Existing test file not found: $EXISTING_TEST_PATH"
            exit 1
        fi

        log_info "Error context available in raw-logs/url for implementation phase"
    fi

    if [ -n "$URL" ] && [ -n "$RAW_LOGS" ]; then
        log_warning "Both URL and raw-logs provided. Using URL and ignoring raw-logs."
        RAW_LOGS=""
    fi
}

# Function to fetch logs from URL
fetch_logs_from_url() {
    log_info "Fetching logs from URL: $URL"

    # Create temp directory for logs
    LOGS_DIR="$SCRIPT_DIR/.temp_logs_$RUN_ID"
    mkdir -p "$LOGS_DIR"

    # TODO: Implement GitHub Actions log fetching
    # For now, this is a placeholder
    log_warning "URL log fetching not yet implemented"
    log_info "Please use raw-logs in info.json or implement URL fetching"

    # Placeholder: Copy logs using gh CLI
    # gh run view <run-id> --log > "$LOGS_DIR/ci_log.txt"

    echo "$URL" > "$LOGS_DIR/url.txt"
    log_info "Logs would be saved to: $LOGS_DIR"
}

# Function to use raw logs
use_raw_logs() {
    log_info "Using raw logs from info.json"

    LOGS_DIR="$SCRIPT_DIR/.temp_logs_$RUN_ID"
    mkdir -p "$LOGS_DIR"

    # Write raw logs to file
    echo "$RAW_LOGS" > "$LOGS_DIR/raw_log.txt"

    log_info "Raw logs saved to: $LOGS_DIR/raw_log.txt"
}

# Function to invoke Claude for reproduction test creation
create_reproduction_test() {
    log_section "PHASE 1: Creating Reproduction Test (5 min timeout)"

    # Determine which reproduction workflow to use
    if [ "$DETERMINISTIC" = "true" ]; then
        REPRO_DIR="reproduce-deterministic-failures"
        log_info "Using deterministic failure workflow"
    else
        REPRO_DIR="reproduce-ND-failures"
        log_info "Using non-deterministic failure workflow"
    fi

    # Create a unique failure folder name from prompt
    FAILURE_NAME=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | cut -c1-50)
    FAILURE_NAME="${FAILURE_NAME}-${TIMESTAMP}"

    FAILURE_DIR="$SCRIPT_DIR/$REPRO_DIR/$FAILURE_NAME"
    mkdir -p "$FAILURE_DIR/logs"
    mkdir -p "$FAILURE_DIR/tests"

    log_info "Created failure directory: $FAILURE_DIR"

    # Copy logs to failure directory
    cp -r "$LOGS_DIR"/* "$FAILURE_DIR/logs/"

    # Create a prompt file for Claude
    cat > "$FAILURE_DIR/.claude_task.md" <<EOF
# Task: Create Reproduction Test

Read the AI_PROMPT.md in $REPRO_DIR/ and create a reproduction test for this failure.

## User Prompt
$PROMPT

## Logs Location
$FAILURE_DIR/logs/

## Expected Output
- Create test in $FAILURE_DIR/tests/
- Create README in $FAILURE_DIR/
- Verify test reproduces the failure

## Time Limit
5 minutes

## Success Criteria
- Test runs and fails with the expected error
- Test is minimal and focused
- README documents the failure clearly

## If You Cannot Reproduce
Document why in the README and recommend next steps.
EOF

    log_info "Created task file: $FAILURE_DIR/.claude_task.md"
    log_info ""
    log_info "📋 MANUAL STEP REQUIRED:"
    log_info "   Invoke Claude with: Read $REPRO_DIR/AI_PROMPT.md and complete the task in $FAILURE_DIR/.claude_task.md"
    log_info ""
    log_warning "Press ENTER after Claude has created the reproduction test..."
    read

    # Check if test was created
    TEST_FILE=$(find "$FAILURE_DIR/tests" -name "test_*" -type f | head -1)

    if [ -z "$TEST_FILE" ]; then
        log_error "No test file found in $FAILURE_DIR/tests/"
        log_error "Cannot proceed without reproduction test"
        write_failure_report "reproduction_failed" "No test file created"
        exit 1
    fi

    log_success "Found test file: $TEST_FILE"

    # Verify test fails
    log_info "Verifying test reproduces the failure..."
    log_info "📋 MANUAL STEP REQUIRED:"
    log_info "   Run the test and verify it fails with the expected error"
    log_info "   If test does not reproduce, fix it before continuing"
    log_info ""
    log_warning "Press ENTER after verifying the test reproduces the failure..."
    read

    # Commit the test to current branch
    log_info "Committing reproduction test to current branch..."
    CURRENT_BRANCH=$(git branch --show-current)

    git add "$FAILURE_DIR"
    git commit -m "Add reproduction test for: $PROMPT

Test location: $TEST_FILE
Failure type: $([ "$DETERMINISTIC" = "true" ] && echo "Deterministic" || echo "Non-deterministic")

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

    log_success "Test committed to branch: $CURRENT_BRANCH"

    # Export variables for next phase
    export TEST_FILE
    export FAILURE_DIR
    export CURRENT_BRANCH
}

# Function to implement the fix
implement_fix() {
    log_section "PHASE 2: Implementing Fix (15 min timeout)"

    # Create fix branch name
    FIX_BRANCH_NAME=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | cut -c1-40)
    FIX_BRANCH="fix/${FIX_BRANCH_NAME}"

    log_info "Creating fix branch: $FIX_BRANCH"

    # Save current branch
    OLD_BRANCH=$(git branch --show-current)

    # Stash any uncommitted changes (including info.json)
    log_info "Stashing uncommitted changes..."
    STASH_NAME="auto-fix-stash-$(date +%s)"
    git stash push -u -m "$STASH_NAME" || {
        log_warning "No changes to stash or stash failed"
    }

    # Create branch off main
    log_info "Checking out main branch..."
    git checkout main

    log_info "Pulling latest changes from origin/main..."
    git pull origin main

    log_info "Creating fix branch: $FIX_BRANCH"
    git checkout -b "$FIX_BRANCH"

    log_success "Created branch $FIX_BRANCH from main"

    # Copy the test to the fix branch
    if [ -n "$EXISTING_TEST_PATH" ]; then
        # Using existing test - copy the entire test directory
        log_info "Copying existing test directory to fix branch..."

        # Get the test directory structure
        # e.g., test path: reproduce-deterministic-failures/timeout-in-datamovement/tests/test.py
        # TEST_DIR: reproduce-deterministic-failures/timeout-in-datamovement/tests
        # PARENT_DIR: reproduce-deterministic-failures/timeout-in-datamovement
        TEST_DIR=$(dirname "$EXISTING_TEST_PATH")
        PARENT_DIR=$(dirname "$TEST_DIR")

        # Copy the entire parent directory (includes run_test.sh, tests/, logs/, README.md)
        if [ -d "$SCRIPT_DIR/$PARENT_DIR" ]; then
            log_info "Copying $PARENT_DIR/ (includes run_test.sh and test files)"
            mkdir -p "$PARENT_DIR"
            cp -r "$SCRIPT_DIR/$PARENT_DIR"/* "$PARENT_DIR/" 2>/dev/null || true

            # Verify run_test.sh was copied
            if [ -f "$PARENT_DIR/run_test.sh" ]; then
                log_success "✓ run_test.sh copied"
                chmod +x "$PARENT_DIR/run_test.sh"
            else
                log_warning "run_test.sh not found in $PARENT_DIR/"
            fi

            git add "$PARENT_DIR"
            git commit -m "Add reproduction test for testing

Test location: $EXISTING_TEST_PATH

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>" || {
                log_warning "No changes to commit (test may already exist)"
            }
            log_success "Test copied to fix branch"
        else
            log_error "Test directory not found: $PARENT_DIR"
            exit 1
        fi
    else
        # Cherry-pick the test from the reproduction branch
        log_info "Cherry-picking reproduction test..."
        TEST_COMMIT=$(git log "$OLD_BRANCH" --oneline --all-match --grep "Add reproduction test" | head -1 | awk '{print $1}')

        if [ -z "$TEST_COMMIT" ]; then
            log_error "Could not find test commit on $OLD_BRANCH"
            exit 1
        fi

        git cherry-pick "$TEST_COMMIT" || {
            log_error "Failed to cherry-pick test commit"
            git cherry-pick --abort
            exit 1
        }

        log_success "Test copied to fix branch"
    fi

    # Create task file for implementation
    cat > "$SCRIPT_DIR/.impl_task.md" <<EOF
# Task: Implement Fix

Read the AI_PROMPT.md at:
$SCRIPT_DIR/implementing-features/AI_PROMPT.md

And implement a fix for this failure.

## User Prompt
$PROMPT

## Reproduction Test
- **Test file**: $TEST_FILE
- **Test directory**: $(dirname "$TEST_FILE")
- **Parent directory**: $(dirname "$(dirname "$TEST_FILE")")
- **Bash runner**: $(dirname "$(dirname "$TEST_FILE")")/run_test.sh

## Error Information (from raw-logs)
\`\`\`
$RAW_LOGS
\`\`\`

This is the error you need to fix. Analyze it to understand the root cause.

## Current Branch
$FIX_BRANCH (created from main)

## Original Branch
$OLD_BRANCH

## CRITICAL INSTRUCTIONS

1. **Build Metal first**: Run /tt-metal/build_metal.sh before testing
2. **Use bash script**: Run tests via ./run_test.sh
   - Navigate to: $(dirname "$(dirname "$TEST_FILE")")
   - Run: ./run_test.sh
3. **Rebuild after changes**: Run /tt-metal/build_metal.sh after EVERY code change
4. **Test multiple times**: Verify stability with $([ "$DETERMINISTIC" = "true" ] && echo "2" || echo "5") consecutive successful runs (deterministic=$DETERMINISTIC)
5. **DO NOT create PR**: Push branch and write PR description, but don't run gh pr create
6. **Write report**: Document everything in $SCRIPT_DIR/outputs/

The bash script (run_test.sh) was copied to the fix branch along with the test.

## Expected Output
1. Analyze root cause from error information above
2. Build Metal: ./build_metal.sh
3. Implement fix iteratively (rebuild after each change)
4. Verify test passes 5/5 times using ./run_test.sh
5. Remove test from branch
6. Push fix branch
7. Write PR description to /tmp/pr_description.md
8. Write execution report to outputs/

## Time Limit
30 minutes (includes build time)

## Success Criteria
- Test passes $([ "$DETERMINISTIC" = "true" ] && echo "2/2" || echo "5/5") times using ./run_test.sh
- Metal rebuilt after all changes
- Branch pushed (but PR NOT created)
- Report written to outputs/

## If You Cannot Fix
Document attempts, explain blockers, recommend experts.
Write report with Status: Failed.
EOF

    log_info "Created implementation task: $SCRIPT_DIR/.impl_task.md"
    log_info ""
    log_info "📋 MANUAL STEP REQUIRED:"
    log_info "   Invoke Claude with:"
    log_info "   "
    log_info "   Read $SCRIPT_DIR/implementing-features/AI_PROMPT.md and complete $SCRIPT_DIR/.impl_task.md"
    log_info "   "
    log_info ""
    log_warning "Press ENTER after Claude has completed the implementation..."
    read

    # Check if branch was pushed (not looking for PR - Claude shouldn't create it)
    log_info "Checking if fix branch was pushed..."

    if git ls-remote --heads origin "$FIX_BRANCH" 2>/dev/null | grep -q "$FIX_BRANCH"; then
        log_success "Fix branch pushed to origin: $FIX_BRANCH"

        # Check if PR description was written
        if [ -f "/tmp/pr_description.md" ]; then
            log_success "PR description found at /tmp/pr_description.md"
            log_info ""
            log_info "📋 To create PR, run:"
            log_info "   gh pr create --draft --base main --head $FIX_BRANCH \\"
            log_info "     --title \"<title>\" --body-file /tmp/pr_description.md"
            log_info ""
        else
            log_warning "No PR description found at /tmp/pr_description.md"
        fi

        export PR_URL=""
        export FIX_BRANCH_PUSHED="yes"
    else
        log_warning "Fix branch was not pushed to origin"
        log_info "Claude may have encountered errors during implementation"

        # Check if there are any commits on the branch
        COMMIT_COUNT=$(git rev-list --count main.."$FIX_BRANCH" 2>/dev/null || echo "0")

        if [ "$COMMIT_COUNT" -eq "0" ] || [ "$COMMIT_COUNT" -eq "1" ]; then
            # No meaningful work done, clean up the branch
            log_info "No meaningful changes on fix branch, cleaning up..."

            # Return to original branch first
            log_info "Returning to original branch: $OLD_BRANCH"
            git checkout "$OLD_BRANCH"

            # Delete the abandoned fix branch
            git branch -D "$FIX_BRANCH" 2>/dev/null && {
                log_success "Deleted abandoned branch: $FIX_BRANCH"
            }
        else
            log_warning "Fix branch has $COMMIT_COUNT commits but wasn't pushed"
            log_info "Keeping branch for manual review: $FIX_BRANCH"
        fi

        export FIX_BRANCH_PUSHED="no"
    fi

    # Return to original branch (if not already there)
    CURRENT=$(git branch --show-current)
    if [ "$CURRENT" != "$OLD_BRANCH" ]; then
        log_info "Returning to original branch: $OLD_BRANCH"
        git checkout "$OLD_BRANCH"
    fi

    # Restore stashed changes if they exist
    log_info "Restoring stashed changes..."
    if git stash list | grep -q "$STASH_NAME"; then
        git stash pop || {
            log_warning "Could not restore stash automatically. Use 'git stash list' to find your changes."
        }
    else
        log_info "No stash to restore"
    fi

    export FIX_BRANCH
}

# Function to write failure report
write_failure_report() {
    FAILURE_REASON="$1"
    DETAILS="$2"

    REPORT_FILE="outputs/${TIMESTAMP}_failed.md"

    cat > "$REPORT_FILE" <<EOF
# ❌ Failed Implementation Report

**Generated**: $(date)
**Status**: Failed
**Duration**: N/A

---

## Summary

Automated fix implementation failed during: $FAILURE_REASON

## Details

$DETAILS

## Input Configuration

\`\`\`json
$(cat info.json)
\`\`\`

## Failure Stage

$FAILURE_REASON

## Next Steps

1. Review the logs and error messages
2. Manually investigate the issue
3. Update info.json if needed
4. Re-run the automation

## Recommended Developers

Contact the relevant team based on the failure type.

---

**Report Generated by run.sh**
EOF

    log_error "Failure report written to: $REPORT_FILE"
}

# Function to finalize and create summary
finalize_run() {
    log_section "FINALIZATION"

    # Find the execution report from Claude
    REPORT_FILE=$(find outputs/ -name "${TIMESTAMP}*.md" -type f | head -1)

    if [ -z "$REPORT_FILE" ]; then
        log_warning "No execution report found in outputs/"
        log_info "Creating summary report..."

        REPORT_FILE="outputs/${TIMESTAMP}_summary.md"

        cat > "$REPORT_FILE" <<EOF
# Implementation Summary

**Generated**: $(date)
**Run ID**: $RUN_ID

---

## Configuration

\`\`\`json
$(cat info.json)
\`\`\`

## Phases Completed

- ✅ Phase 1: Reproduction test creation
- ✅ Phase 2: Fix implementation
- ⚠️  Phase 3: Report generation (manual)

## Reproduction Test

- **Location**: $TEST_FILE
- **Branch**: $CURRENT_BRANCH

## Fix Branch

- **Branch**: $FIX_BRANCH
- **PR**: ${PR_URL:-Not created}

## Next Steps

1. Review the changes in branch $FIX_BRANCH
2. Check if PR was created: $PR_URL
3. Run recommended CI workflows
4. Request review from relevant developers

---

**Report Generated by run.sh**
EOF
    fi

    log_success "Execution complete!"
    log_info "Report: $REPORT_FILE"

    if [ -n "$PR_URL" ]; then
        log_success "PR: $PR_URL"
    fi

    # Cleanup temp logs
    if [ -d "$LOGS_DIR" ]; then
        rm -rf "$LOGS_DIR"
        log_info "Cleaned up temporary logs"
    fi

    log_info ""
    log_success "🎉 Run completed! Check $REPORT_FILE for details."
}

# Main execution
main() {
    log_section "Automated Fix Implementation"
    log_info "Starting run at $(date)"
    log_info "Run ID: $RUN_ID"

    # Phase 0: Parse configuration
    parse_info_json

    # Check if we're on main branch (common in CI)
    CURRENT_BRANCH=$(git branch --show-current)
    if [ "$CURRENT_BRANCH" = "main" ]; then
        log_warning "Running on main branch (common in CI)"
        log_info "Will create a temporary branch for reproduction test"

        # Create a temporary branch for the reproduction phase
        TEMP_BRANCH="auto-fix-temp-$(date +%s)"
        git checkout -b "$TEMP_BRANCH"
        export CURRENT_BRANCH="$TEMP_BRANCH"
        log_info "Created temporary branch: $TEMP_BRANCH"
    fi

    # Check if using existing test (skip reproduction)
    if [ -n "$EXISTING_TEST_PATH" ]; then
        log_section "Using Existing Test"
        log_info "Skipping reproduction phase"
        log_info "Test file: $EXISTING_TEST_PATH"

        # Set variables for later phases
        export TEST_FILE="$EXISTING_TEST_PATH"
        export CURRENT_BRANCH=$(git branch --show-current)
        export FAILURE_DIR=$(dirname "$(dirname "$EXISTING_TEST_PATH")")

        log_success "Using existing test, proceeding to fix implementation"
    else
        # Phase 0.5: Get logs
        if [ -n "$URL" ]; then
            fetch_logs_from_url
        else
            use_raw_logs
        fi

        # Phase 1: Create reproduction test
        create_reproduction_test
    fi

    # Phase 2: Implement fix
    implement_fix

    # Phase 3: Finalize
    finalize_run
}

# Trap errors and create failure report
trap 'write_failure_report "unexpected_error" "Script encountered an error. Check logs above."' ERR

# Run main function
main "$@"
