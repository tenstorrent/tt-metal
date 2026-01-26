#!/bin/bash

# Script to create a PR from an ND failure analysis
# Usage: ./create_pr_from_analysis.sh [OPTIONS] <analysis_result.md>
#
# This script:
# 1. Reads an analysis result markdown file
# 2. Uses Claude CLI to implement the recommended fixes
# 3. Creates a branch, commits changes, and opens a PR

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve to absolute path
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Non-root user for running Claude CLI
CLAUDE_USER="claude-runner"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1" >&2
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_tools=()

    # Check gh CLI
    if ! command -v gh &> /dev/null; then
        missing_tools+=("gh (GitHub CLI)")
    fi

    # Check Claude CLI
    if ! command -v claude &> /dev/null; then
        missing_tools+=("claude (Claude CLI)")
    fi

    # Check git
    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools:"
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        exit 1
    fi

    # Check GitHub authentication
    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI is not authenticated. Please run 'gh auth login' first."
        exit 1
    fi

    # Check git repo
    if ! git -C "$REPO_ROOT" rev-parse --git-dir &> /dev/null; then
        log_error "Not in a git repository: $REPO_ROOT"
        exit 1
    fi

    log_info "All prerequisites met"
}

# Function to ensure non-root user exists for Claude CLI
ensure_claude_user() {
    # Skip if not running as root
    if [[ $(id -u) -ne 0 ]]; then
        CLAUDE_USER=$(whoami)
        log_debug "Not running as root, using current user: $CLAUDE_USER"
        return 0
    fi

    # Check if user already exists
    if id "$CLAUDE_USER" &>/dev/null; then
        log_debug "User $CLAUDE_USER already exists"
    else
        log_info "Creating non-root user '$CLAUDE_USER' for Claude CLI..."
        useradd -m -s /bin/bash "$CLAUDE_USER" 2>/dev/null || {
            log_error "Failed to create user $CLAUDE_USER"
            exit 1
        }
        log_info "User '$CLAUDE_USER' created successfully"
    fi

    # Get the claude user's home directory
    local claude_home
    claude_home=$(eval echo "~$CLAUDE_USER")

    # Copy Claude credentials to the non-root user if they exist in root's home
    if [[ -f /root/.claude.json ]]; then
        cp /root/.claude.json "$claude_home/.claude.json" 2>/dev/null || true
        chown "$CLAUDE_USER:$CLAUDE_USER" "$claude_home/.claude.json" 2>/dev/null || true
        chmod 600 "$claude_home/.claude.json" 2>/dev/null || true
        log_debug "Copied Claude credentials to $CLAUDE_USER"
    fi

    # Copy Claude config directory if it exists
    if [[ -d /root/.claude ]]; then
        cp -r /root/.claude "$claude_home/.claude" 2>/dev/null || true
        chown -R "$CLAUDE_USER:$CLAUDE_USER" "$claude_home/.claude" 2>/dev/null || true
        log_debug "Copied Claude config directory to $CLAUDE_USER"
    fi

    # Ensure the user has read/write access to the repository
    if [[ -d "$REPO_ROOT" ]]; then
        chmod -R a+rwX "$REPO_ROOT" 2>/dev/null || true
    fi
}

# Function to generate a branch name from analysis
generate_branch_name() {
    local analysis_file=$1

    # Extract key info from analysis to create branch name
    # Look for the test name or error type
    local test_name=""
    local error_type=""

    # Try to extract test name
    test_name=$(grep -m1 "^\*\*Test\*\*:" "$analysis_file" 2>/dev/null | sed 's/.*: *//' | tr -cd '[:alnum:]_' | head -c 30)

    # Try to extract a short error description
    error_type=$(grep -m1 "^\*\*Error\*\*:" "$analysis_file" 2>/dev/null | sed 's/.*: *//' | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_' | head -c 20)

    # Fallback to generic name with timestamp
    if [[ -z "$test_name" && -z "$error_type" ]]; then
        echo "fix/nd-failure-$(date +%Y%m%d-%H%M%S)"
    elif [[ -n "$test_name" ]]; then
        echo "fix/nd-${test_name}"
    else
        echo "fix/nd-${error_type}"
    fi
}

# Function to extract PR title from analysis
extract_pr_title() {
    local analysis_file=$1

    # Try to get root cause for title
    local root_cause=""
    root_cause=$(grep -A1 "^\*\*Root cause\*\*:" "$analysis_file" 2>/dev/null | head -1 | sed 's/.*: *//' | head -c 70)

    if [[ -n "$root_cause" ]]; then
        echo "Fix ND failure: $root_cause"
    else
        echo "Fix non-deterministic test failure"
    fi
}

# Function to create implementation prompt for Claude
create_implementation_prompt() {
    local analysis_file=$1
    local prompt_file=$2

    cat > "$prompt_file" << 'PROMPT_HEADER'
# Task: Implement Code Changes from Analysis

You are implementing fixes for a non-deterministic test failure based on an analysis document.

## Instructions

1. Read the analysis document below carefully
2. Implement ONLY the "Priority 1" or "Fix 1" changes (the most impactful fix)
3. Make the minimal changes necessary - don't refactor unrelated code
4. Follow the existing code style in each file
5. If the analysis suggests multiple fixes, implement only the first/primary one

## Important Rules

- Only edit files that are explicitly mentioned in the analysis
- Use the exact code changes suggested when possible
- If the suggested code doesn't compile or has issues, make minimal adjustments
- Do NOT create new files unless absolutely necessary
- Do NOT modify test files unless the analysis specifically recommends it
- After making changes, verify they compile (if C++) by checking syntax

## Output

After implementing the changes, output a brief summary of what you changed:
- Which file(s) you modified
- What the change does
- Any deviations from the suggested fix and why

---

## Analysis Document

PROMPT_HEADER

    # Append the analysis content
    cat "$analysis_file" >> "$prompt_file"

    cat >> "$prompt_file" << 'PROMPT_FOOTER'

---

Now implement the primary fix from this analysis. Make the code changes directly to the repository files.
PROMPT_FOOTER
}

# Function to run Claude to implement changes
run_claude_implementation() {
    local prompt_file=$1
    local output_file=$2
    local model=$3

    log_info "Running Claude to implement fixes (model: $model)..."

    local exit_code=0
    local temp_output
    temp_output=$(mktemp)
    chmod 666 "$temp_output"

    # Make prompt file readable by claude user
    chmod 644 "$prompt_file" 2>/dev/null || true

    # Run Claude with permission to edit files
    if [[ $(id -u) -eq 0 ]]; then
        if command -v timeout &> /dev/null; then
            log_info "Using timeout (15 minute limit for implementation)"
            if ! timeout 900 runuser -u "$CLAUDE_USER" -- bash -c "cd '$REPO_ROOT' && cat '$prompt_file' | claude --model '$model' -p --dangerously-skip-permissions" 2>&1 | tee "$temp_output"; then
                exit_code=$?
            fi
        else
            if ! runuser -u "$CLAUDE_USER" -- bash -c "cd '$REPO_ROOT' && cat '$prompt_file' | claude --model '$model' -p --dangerously-skip-permissions" 2>&1 | tee "$temp_output"; then
                exit_code=$?
            fi
        fi
    else
        if command -v timeout &> /dev/null; then
            log_info "Using timeout (15 minute limit for implementation)"
            if ! timeout 900 bash -c "cd '$REPO_ROOT' && cat '$prompt_file' | claude --model '$model' -p --dangerously-skip-permissions" 2>&1 | tee "$temp_output"; then
                exit_code=$?
            fi
        else
            if ! bash -c "cd '$REPO_ROOT' && cat '$prompt_file' | claude --model '$model' -p --dangerously-skip-permissions" 2>&1 | tee "$temp_output"; then
                exit_code=$?
            fi
        fi
    fi

    cp "$temp_output" "$output_file" 2>/dev/null || true
    rm -f "$temp_output"

    if [[ $exit_code -eq 124 ]]; then
        log_error "Claude implementation timed out after 15 minutes"
        return 1
    fi

    return $exit_code
}

# Function to create the PR
create_pull_request() {
    local analysis_file=$1
    local branch_name=$2
    local base_branch=$3
    local implementation_log=$4

    log_info "Creating pull request..."

    # Check if there are any changes to commit
    cd "$REPO_ROOT"

    if git diff --quiet && git diff --cached --quiet; then
        log_error "No changes were made by Claude. Cannot create PR."
        log_info "This might mean:"
        log_info "  - The suggested fixes couldn't be applied"
        log_info "  - The files mentioned in the analysis don't exist"
        log_info "  - Claude chose not to make changes"
        return 1
    fi

    # Generate PR title and body
    local pr_title
    pr_title=$(extract_pr_title "$analysis_file")

    # Create PR body from analysis
    local pr_body
    pr_body=$(cat << EOF
## Summary

This PR implements fixes for a non-deterministic test failure, based on automated analysis.

## Analysis

<details>
<summary>Click to expand full analysis</summary>

$(cat "$analysis_file")

</details>

## Implementation Notes

$(cat "$implementation_log" 2>/dev/null || echo "See commits for details.")

## Testing

- [ ] Verify the changes compile
- [ ] Run the affected tests locally
- [ ] Monitor CI for the specific test that was failing

---

*This PR was automatically generated by the ND failure analysis system.*
EOF
)

    # Stage all changes
    git add -A

    # Create commit - pre-commit hooks may modify files
    local commit_msg="Fix non-deterministic test failure

Based on automated analysis of CI failures.

Co-authored-by: Claude <claude@anthropic.com>"

    log_info "Committing changes (pre-commit hooks will run)..."
    if ! git commit -m "$commit_msg"; then
        log_warn "Initial commit failed (likely pre-commit hooks modified files)"
    fi

    # Check if pre-commit hooks modified files - if so, add and commit again
    local max_attempts=3
    local attempt=1
    while ! git diff --quiet || ! git diff --cached --quiet; do
        if [[ $attempt -ge $max_attempts ]]; then
            log_error "Pre-commit hooks keep modifying files after $max_attempts attempts. Giving up."
            return 1
        fi
        log_info "Pre-commit hooks modified files, adding changes (attempt $attempt/$max_attempts)..."
        git add -A
        if ! git commit --amend --no-edit; then
            log_warn "Amend failed, trying new commit..."
            git commit -m "Apply formatting fixes" || true
        fi
        attempt=$((attempt + 1))
    done

    log_info "Changes committed to branch: $branch_name"

    # Verify we have commits ahead of base
    local commits_ahead
    commits_ahead=$(git rev-list --count "origin/$base_branch..HEAD" 2>/dev/null || echo "0")
    if [[ "$commits_ahead" == "0" ]]; then
        log_error "No commits to push. Something went wrong with the commit."
        return 1
    fi
    log_info "Branch has $commits_ahead commit(s) ahead of origin/$base_branch"

    # Push branch
    log_info "Pushing branch to origin..."
    if ! git push -u origin "$branch_name"; then
        log_error "Failed to push branch to origin"
        return 1
    fi

    # Create PR
    log_info "Creating PR..."
    local pr_url
    local pr_exit_code
    pr_url=$(gh pr create \
        --title "$pr_title" \
        --body "$pr_body" \
        --base "$base_branch" \
        --head "$branch_name" 2>&1) || pr_exit_code=$?

    # Check if PR creation succeeded
    if [[ -n "$pr_exit_code" ]] || [[ -z "$pr_url" ]] || [[ "$pr_url" == *"failed"* ]] || [[ "$pr_url" != "https://"* ]]; then
        log_error "Failed to create PR: $pr_url"
        log_info "Cleaning up: deleting remote branch..."
        git push origin --delete "$branch_name" 2>/dev/null || true
        return 1
    fi

    echo ""
    log_info "========================================="
    log_info "PR created successfully!"
    log_info "URL: $pr_url"
    log_info "========================================="

    echo "$pr_url"
}

# Main function
main() {
    local analysis_file=""
    local base_branch="main"
    local branch_name=""
    local claude_model="sonnet"
    local dry_run=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --base|-b)
                base_branch=$2
                shift 2
                ;;
            --branch)
                branch_name=$2
                shift 2
                ;;
            --model|-m)
                claude_model=$2
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS] <analysis_result.md>"
                echo ""
                echo "Create a PR from an ND failure analysis document."
                echo ""
                echo "Options:"
                echo "  --base, -b <branch>   Base branch for PR (default: main)"
                echo "  --branch <name>       Branch name for the fix (auto-generated if not provided)"
                echo "  --model, -m <model>   Claude model: haiku, sonnet, opus (default: sonnet)"
                echo "  --dry-run             Make changes but don't push or create PR"
                echo "  --help, -h            Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 analysis_result.md"
                echo "  $0 --base develop --model opus analysis_result.md"
                echo "  $0 --dry-run analysis_result.md"
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                exit 1
                ;;
            *)
                if [[ -z "$analysis_file" ]]; then
                    analysis_file=$1
                else
                    log_error "Unexpected argument: $1"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # Validate analysis file
    if [[ -z "$analysis_file" ]]; then
        log_error "No analysis file provided. Use --help for usage."
        exit 1
    fi

    if [[ ! -f "$analysis_file" ]]; then
        log_error "Analysis file not found: $analysis_file"
        exit 1
    fi

    # Convert to absolute path
    analysis_file=$(cd "$(dirname "$analysis_file")" && pwd)/$(basename "$analysis_file")

    log_info "Starting PR creation from analysis..."
    log_info "Analysis file: $analysis_file"
    log_info "Base branch: $base_branch"
    log_info "Model: $claude_model"

    # Check prerequisites
    check_prerequisites

    # Ensure claude user exists (for root environments)
    ensure_claude_user

    # Generate branch name if not provided
    if [[ -z "$branch_name" ]]; then
        branch_name=$(generate_branch_name "$analysis_file")
    fi
    log_info "Branch name: $branch_name"

    # Change to repo root
    cd "$REPO_ROOT"

    # Save current branch to return to later (on failure)
    local original_branch
    original_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    log_debug "Original branch: $original_branch"

    # Check for uncommitted changes and stash them
    local stashed=false
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log_warn "Working directory has uncommitted changes."
        log_info "Stashing changes before proceeding..."
        git stash push -m "Auto-stash before ND failure PR creation $(date +%Y%m%d-%H%M%S)"
        stashed=true
    fi

    # Track if we created a branch (for cleanup) and if PR was successful
    local branch_created=false
    local pr_success=false

    # Function to restore original state (called on exit unless PR succeeded)
    cleanup() {
        # Always cd to repo root for git operations
        cd "$REPO_ROOT" 2>/dev/null || true

        # If PR was successful, don't clean up
        if [[ "$pr_success" == "true" ]]; then
            return 0
        fi

        log_warn "Cleaning up..."

        # Delete the branch we created if it exists
        if [[ "$branch_created" == "true" ]] && [[ -n "$branch_name" ]]; then
            # First checkout a different branch
            git checkout "$original_branch" 2>/dev/null || git checkout "$base_branch" 2>/dev/null || git checkout main 2>/dev/null || true

            # Always try to delete remote branch (won't error if it doesn't exist)
            log_info "Deleting remote branch 'origin/$branch_name' (if exists)..."
            git push origin --delete "$branch_name" 2>/dev/null || true

            # Delete local branch
            log_info "Deleting local branch '$branch_name'..."
            git branch -D "$branch_name" 2>/dev/null || true
        elif [[ -n "$original_branch" ]] && [[ "$original_branch" != "HEAD" ]]; then
            git checkout "$original_branch" 2>/dev/null || true
        fi

        if [[ "$stashed" == "true" ]]; then
            log_info "Restoring stashed changes..."
            git stash pop 2>/dev/null || true
        fi
    }

    # Set up trap to always clean up on exit
    trap cleanup EXIT

    # Fetch latest from origin to ensure we have up-to-date refs
    log_info "Fetching latest from origin..."
    git fetch origin --prune

    # Verify base branch exists on remote
    if ! git show-ref --verify --quiet "refs/remotes/origin/$base_branch"; then
        log_error "Base branch 'origin/$base_branch' not found."
        log_error "Available remote branches:"
        git branch -r | head -10
        exit 1
    fi

    # Delete local branch if it already exists (from previous failed run)
    if git show-ref --verify --quiet "refs/heads/$branch_name"; then
        log_warn "Branch '$branch_name' already exists locally. Deleting it..."
        git branch -D "$branch_name" 2>/dev/null || true
    fi

    # Check if branch already exists on remote - append timestamp to make unique
    if git show-ref --verify --quiet "refs/remotes/origin/$branch_name"; then
        log_warn "Branch '$branch_name' already exists on remote."
        log_warn "Will create a unique local branch name."
        branch_name="${branch_name}-$(date +%H%M%S)"
        log_info "Using branch name: $branch_name"
    fi

    # Create and checkout new branch from latest origin/base_branch
    log_info "Creating branch '$branch_name' from 'origin/$base_branch'..."
    if ! git checkout -b "$branch_name" "origin/$base_branch"; then
        log_error "Failed to create branch from origin/$base_branch"
        exit 1
    fi
    branch_created=true
    log_info "Now on branch: $branch_name (based on latest origin/$base_branch)"

    # Create implementation prompt
    local impl_prompt
    impl_prompt=$(mktemp)
    create_implementation_prompt "$analysis_file" "$impl_prompt"

    # Run Claude to implement changes
    local impl_log
    impl_log=$(mktemp)

    log_info "Asking Claude to implement the fixes..."
    if ! run_claude_implementation "$impl_prompt" "$impl_log" "$claude_model"; then
        log_error "Claude implementation failed"
        rm -f "$impl_prompt" "$impl_log"
        exit 1
    fi

    rm -f "$impl_prompt"

    # Show what changed
    echo ""
    log_info "Changes made by Claude:"
    git diff --stat
    echo ""

    if [[ "$dry_run" == "true" ]]; then
        log_info "Dry run mode - not creating PR"
        log_info "Changes are on branch: $branch_name"
        log_info "To create the PR manually:"
        log_info "  git add -A && git commit -m 'Fix ND failure'"
        log_info "  git push -u origin $branch_name"
        log_info "  gh pr create --base $base_branch"
        rm -f "$impl_log"
        # Cleanup will run via trap - branch deleted, original branch restored, stash popped
        exit 0
    fi

    # Create the PR
    local pr_url
    if pr_url=$(create_pull_request "$analysis_file" "$branch_name" "$base_branch" "$impl_log"); then
        # PR created successfully - don't clean up
        pr_success=true
        log_info "Success! PR URL: $pr_url"
        if [[ "$stashed" == "true" ]]; then
            log_info ""
            log_info "Note: Your original changes are stashed. To restore:"
            log_info "  git checkout $original_branch && git stash pop"
        fi
    else
        log_error "Failed to create PR"
        rm -f "$impl_log"
        exit 1
    fi

    rm -f "$impl_log"
}

main "$@"
