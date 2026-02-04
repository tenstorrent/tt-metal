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

    # If existing test path is provided, skip log requirements
    if [ -n "$EXISTING_TEST_PATH" ]; then
        log_info "Existing test path provided, will skip reproduction phase"

        # Validate test file exists
        if [ ! -f "$EXISTING_TEST_PATH" ]; then
            log_error "Existing test file not found: $EXISTING_TEST_PATH"
            exit 1
        fi

        # No need for logs if using existing test
        return
    fi

    # Otherwise, require logs
    if [ -z "$URL" ] && [ -z "$RAW_LOGS" ]; then
        log_error "No existing-test-path provided and both URL and raw-logs are empty."
        log_error "Provide either: existing-test-path OR (url OR raw-logs)"
        exit 1
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
        # Using existing test - just copy the files
        log_info "Copying existing test to fix branch..."

        # Get the test directory structure
        TEST_DIR=$(dirname "$EXISTING_TEST_PATH")
        PARENT_DIR=$(dirname "$TEST_DIR")

        # Copy the entire test directory structure
        if [ -d "$PARENT_DIR" ]; then
            mkdir -p "$PARENT_DIR"
            cp -r "$SCRIPT_DIR/$PARENT_DIR"/* "$PARENT_DIR/" 2>/dev/null || true

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

Read the AI_PROMPT.md in implementing-features/ and implement a fix for this failure.

## User Prompt
$PROMPT

## Reproduction Test
$TEST_FILE

## Current Branch
$FIX_BRANCH (created from main)

## Original Branch
$OLD_BRANCH (has the test)

## Expected Output
1. Analyze root cause
2. Implement fix iteratively
3. Verify test passes reliably
4. Create draft PR (excluding test)
5. Write execution report

## Time Limit
15 minutes

## Success Criteria
- Test passes 5/5 times
- Changes are well-documented
- Draft PR created
- Report written to outputs/

## If You Cannot Fix
Document attempts, explain blockers, recommend experts.
Write report with Status: Failed.
EOF

    log_info "Created implementation task: $SCRIPT_DIR/.impl_task.md"
    log_info ""
    log_info "📋 MANUAL STEP REQUIRED:"
    log_info "   Invoke Claude with: Read implementing-features/AI_PROMPT.md and complete .impl_task.md"
    log_info ""
    log_warning "Press ENTER after Claude has completed the implementation..."
    read

    # Check for PR creation
    PR_URL=$(gh pr list --head "$FIX_BRANCH" --json url --jq '.[0].url' 2>/dev/null || echo "")

    if [ -n "$PR_URL" ]; then
        log_success "Draft PR created: $PR_URL"
        export PR_URL
    else
        log_warning "No PR found for branch $FIX_BRANCH"
        log_info "Claude may have documented why PR was not created"

        # If no PR was created, offer to clean up the branch
        log_warning "Fix branch was created but no PR resulted"
        log_info "Cleaning up abandoned fix branch..."

        # Return to original branch first
        log_info "Returning to original branch: $OLD_BRANCH"
        git checkout "$OLD_BRANCH"

        # Delete the abandoned fix branch
        git branch -D "$FIX_BRANCH" 2>/dev/null && {
            log_success "Deleted abandoned branch: $FIX_BRANCH"
        } || {
            log_warning "Could not delete branch $FIX_BRANCH (may not exist locally)"
        }

        export PR_URL=""
        # Continue to restore stash
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
