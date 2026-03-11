#!/usr/bin/env bash
# GitHub Actions Security Linting Script
# Detects common shell injection and security anti-patterns in workflows/actions
#
# Usage: ./check-actions-security.sh [--strict]
#   --strict: Exit with error code if any issues found

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
GITHUB_DIR="$REPO_ROOT/.github"

STRICT_MODE=false
ISSUES_FOUND=0

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

if [[ "${1:-}" == "--strict" ]]; then
    STRICT_MODE=true
fi

log_issue() {
    local severity="$1"
    local file="$2"
    local message="$3"
    local color="$YELLOW"

    if [[ "$severity" == "CRITICAL" ]] || [[ "$severity" == "HIGH" ]]; then
        color="$RED"
    fi

    echo -e "${color}[$severity]${NC} $file: $message"
    ((ISSUES_FOUND++)) || true
}

# Prints a one-time fix example for a check class.
# Uses "shown_example_<class>" variables to deduplicate.
log_fix_example() {
    local class="$1"
    local example="$2"
    local flag_var="shown_example_${class}"

    if [[ "${!flag_var:-}" == "true" ]]; then
        return
    fi
    eval "$flag_var=true"

    echo -e "${CYAN}  Fix example:${NC}"
    while IFS= read -r line; do
        echo -e "    ${DIM}${line}${NC}"
    done <<< "$example"
    echo ""
}

echo "=========================================="
echo "GitHub Actions Security Lint Check"
echo "=========================================="
echo ""

# Helper: AWK snippet for detecting run block boundaries with proper indentation tracking.
# This avoids false positives from step names like "- name: ${{ matrix.foo }}" being
# flagged as run block content. Exits run block when:
#   1. A new YAML list item is encountered (line starting with "- ")
#   2. A YAML key at the same or lesser indentation as run: is encountered
AWK_RUN_BLOCK_DETECT='
    BEGIN { in_run_block = 0; run_indent = 0 }
    /^[[:space:]]*run:[[:space:]]*\|/ {
        match($0, /^[[:space:]]*/); run_indent = RLENGTH; in_run_block = 1; next
    }
    in_run_block {
        if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_run_block = 0; next }
        match($0, /^[[:space:]]*/); curr_indent = RLENGTH
        if (curr_indent <= run_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_run_block = 0; next }
    }
'

# Check 1: Direct input interpolation in run blocks (high risk patterns)
# Only flags UNSAFE usage: direct interpolation in run: blocks
# Safe usage (passing via env var) is NOT flagged
echo "Checking for dangerous direct interpolation patterns..."
dangerous_patterns='github\.event\.(comment\.body|issue\.body|pull_request\.body|pull_request\.title)'
while IFS= read -r -d '' file; do
    if grep -qE "$dangerous_patterns" "$file" 2>/dev/null; then
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*run:/ && /\$\{\{.*github\.event\.(comment\.body|issue\.body|pull_request\.body|pull_request\.title)/ { print; next }
            in_run_block && /\$\{\{.*github\.event\.(comment\.body|issue\.body|pull_request\.body|pull_request\.title)/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "Contains dangerous event data interpolation directly in run: block (comment/issue/PR body/title)"
            log_fix_example "event_data" \
'# BEFORE (unsafe - attacker controls PR title):
  run: echo "${{ github.event.pull_request.title }}"
# AFTER (safe - shell variable cannot inject commands):
  env:
    PR_TITLE: ${{ github.event.pull_request.title }}
  run: echo "$PR_TITLE"'
        fi
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 2: eval usage
echo "Checking for eval usage..."
while IFS= read -r file; do
    if [[ -n "$file" ]]; then
        log_issue "HIGH" "$file" "Contains 'eval' statement - consider safer alternatives"
        log_fix_example "eval" \
'# BEFORE (unsafe - eval interprets arbitrary code):
  run: eval "$USER_INPUT"
# AFTER (safe - use direct execution or a case/switch):
  run: |
    case "$VALIDATED_CMD" in
      build)  make build ;;
      test)   make test ;;
      *)      echo "Unknown command"; exit 1 ;;
    esac'
    fi
done < <(grep -rl 'eval\s' "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null || true)

# Check 3: bash -c with direct ${{ }} interpolation
echo "Checking for bash -c with direct interpolation..."
while IFS= read -r file; do
    if [[ -n "$file" ]]; then
        log_issue "MEDIUM" "$file" "Contains 'bash -c' with direct \${{ }} - ensure input is via env var"
        log_fix_example "bash_c" \
'# BEFORE (unsafe - interpolation happens before shell parsing):
  run: bash -c "echo ${{ inputs.command }}"
# AFTER (safe - pass via env var):
  env:
    CMD: ${{ inputs.command }}
  run: bash -c "echo \"$CMD\""'
    fi
done < <(grep -rl 'bash.*-c.*\${{' "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null || true)

# Check 4: pull_request_target with potentially unsafe patterns
echo "Checking for pull_request_target with checkout..."
while IFS= read -r file; do
    if [[ -n "$file" ]] && grep -q 'actions/checkout' "$file" 2>/dev/null; then
        log_issue "HIGH" "$file" "Uses pull_request_target with checkout - verify PR code is not executed unsafely"
        log_fix_example "prt_checkout" \
'# pull_request_target runs with BASE repo privileges but can check out HEAD code.
# BEFORE (unsafe - checks out and runs untrusted fork code with write token):
  on: pull_request_target
  steps:
    - uses: actions/checkout@v4
      with: { ref: ${{ github.event.pull_request.head.sha }} }
    - run: make test  # <-- runs fork code with base repo secrets!
# AFTER (safe - only check out base, or use a separate unprivileged job):
  on: pull_request_target
  steps:
    - uses: actions/checkout@v4  # checks out base branch by default
    - run: make test'
    fi
done < <(grep -rl 'pull_request_target' "$GITHUB_DIR/workflows" 2>/dev/null || true)

# Check 5: Count unpinned external action references
echo "Checking for unpinned external action references..."
external_unpinned=$(grep -rh 'uses:' "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null | \
    grep -v 'tenstorrent/' | \
    grep -v './' | \
    grep -v '#' | \
    grep -E '@(v[0-9]|main|master)' | \
    wc -l | tr -d '[:space:]' || echo 0)
external_unpinned="${external_unpinned:-0}"
if [[ "$external_unpinned" =~ ^[0-9]+$ ]] && [[ "$external_unpinned" -gt 0 ]]; then
    log_issue "LOW" "Multiple files" "Found $external_unpinned external actions using mutable refs (@v*, @main) - consider pinning to SHA"
    log_fix_example "unpin_action" \
'# BEFORE (mutable tag - can be moved to point to different code):
  uses: actions/checkout@v4
# AFTER (pinned to immutable commit SHA):
  uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2'
fi

# Check 6: Count secrets: inherit usage
echo "Checking for secrets: inherit usage..."
secrets_inherit_count=$(grep -rl 'secrets:\s*inherit' "$GITHUB_DIR/workflows" 2>/dev/null | wc -l | tr -d ' ' || echo 0)
if [[ "$secrets_inherit_count" -gt 50 ]]; then
    log_issue "LOW" "Multiple files" "Found $secrets_inherit_count workflows using 'secrets: inherit' - consider explicit secret passing"
    log_fix_example "secrets_inherit" \
'# BEFORE (passes ALL repo secrets to called workflow):
  uses: ./.github/workflows/build.yaml
  secrets: inherit
# AFTER (passes only the secrets the called workflow needs):
  uses: ./.github/workflows/build.yaml
  secrets:
    DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}'
fi

# Check 7: GITHUB_TOKEN or ACTIONS_RUNTIME_TOKEN exposure in logs
echo "Checking for potential token exposure in logs..."
while IFS= read -r file; do
    if [[ -n "$file" ]]; then
        log_issue "HIGH" "$file" "Potential token exposure in echo/print statement"
        log_fix_example "token_log" \
'# BEFORE (token value visible in workflow logs):
  run: echo "Token: $GITHUB_TOKEN"
# AFTER (use token directly without logging, or mask it):
  run: |
    echo "::add-mask::$GITHUB_TOKEN"
    curl -H "Authorization: Bearer $GITHUB_TOKEN" ...'
    fi
done < <(grep -rlE 'echo.*\$\{?\{?(GITHUB_TOKEN|ACTIONS_RUNTIME_TOKEN|ACTIONS_ID_TOKEN_REQUEST_TOKEN)' \
    "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null || true)

# Check 8: Overly broad permissions
echo "Checking for overly broad permissions..."
while IFS= read -r file; do
    if [[ -n "$file" ]]; then
        log_issue "MEDIUM" "$file" "Uses 'write-all' permissions - apply principle of least privilege"
        log_fix_example "broad_perms" \
'# BEFORE (grants write access to everything):
  permissions: write-all
# AFTER (grant only what is needed):
  permissions:
    contents: read
    pull-requests: write'
    fi
done < <(grep -rl 'permissions:.*write-all\|write-all' "$GITHUB_DIR/workflows" 2>/dev/null || true)

# Check 9: Ref-based injection vectors (head.ref, base.ref in ${{ }})
# Only flags UNSAFE usage: direct interpolation in run: blocks
# Safe usage (passing via env var) is NOT flagged
echo "Checking for ref-based injection patterns..."
while IFS= read -r -d '' file; do
    if grep -qE '\$\{\{.*\.(head|base)\.ref' "$file" 2>/dev/null; then
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*run:/ && /\$\{\{.*\.(head|base)\.ref/ { print; next }
            in_run_block && /\$\{\{.*\.(head|base)\.ref/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "MEDIUM" "$file" "Contains head.ref/base.ref interpolation directly in run: block - use env var instead"
            log_fix_example "ref_inject" \
'# BEFORE (unsafe - branch name can contain shell metacharacters):
  run: echo "Branch: ${{ github.head_ref }}"
# AFTER (safe - shell variable is not re-parsed):
  env:
    HEAD_REF: ${{ github.head_ref }}
  run: echo "Branch: $HEAD_REF"'
        fi
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 10: curl | bash patterns
echo "Checking for curl pipe to shell patterns..."
while IFS= read -r file; do
    if [[ -n "$file" ]]; then
        log_issue "HIGH" "$file" "Contains 'curl | bash' pattern - download and verify scripts before execution"
        log_fix_example "curl_bash" \
'# BEFORE (unsafe - executes remote code without verification):
  run: curl -sSL https://example.com/install.sh | bash
# AFTER (safe - download, inspect, then execute):
  run: |
    curl -sSL -o install.sh https://example.com/install.sh
    sha256sum -c <<< "expected_hash  install.sh"
    bash install.sh'
    fi
done < <(grep -rlE 'curl.*\|\s*(ba)?sh|wget.*\|\s*(ba)?sh' "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null || true)

# Check 11: inputs.*, steps.*.outputs.*, and needs.*.outputs.* direct interpolation in run blocks
# Only flags UNSAFE usage: direct interpolation in run: blocks
# Safe usage (passing via env var) is NOT flagged
echo "Checking for inputs/outputs interpolation in run blocks..."
while IFS= read -r -d '' file; do
    if grep -qE '\$\{\{.*(inputs\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.)' "$file" 2>/dev/null; then
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*run:/ && /\$\{\{.*(inputs\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.)/ { print; next }
            in_run_block && /\$\{\{.*(inputs\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.)/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "Contains \${{ inputs.* }}, \${{ steps.*.outputs.* }}, or \${{ needs.*.outputs.* }} directly in run: block - use env var instead"
            log_fix_example "inputs_outputs" \
'# BEFORE (unsafe - input/output is macro-expanded into shell code):
  run: echo "${{ needs.build.outputs.pr-number }}"
# AFTER (safe - assign to env var, reference as shell variable):
  env:
    PR_NUMBER: ${{ needs.build.outputs.pr-number }}
  run: echo "$PR_NUMBER"'
        fi
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 12: Deprecated ::set-output and ::save-state workflow commands
# These were deprecated in October 2022 and are vulnerable to log injection.
# Use $GITHUB_OUTPUT and $GITHUB_STATE environment files instead.
echo "Checking for deprecated workflow commands..."
while IFS= read -r -d '' file; do
    if grep -qE '::set-output |::save-state ' "$file" 2>/dev/null; then
        log_issue "MEDIUM" "$file" "Uses deprecated ::set-output or ::save-state command - use \$GITHUB_OUTPUT/\$GITHUB_STATE files instead"
        log_fix_example "deprecated_cmd" \
'# BEFORE (deprecated - vulnerable to log injection):
  run: echo "::set-output name=result::$VALUE"
# AFTER (safe - writes to environment file):
  run: echo "result=$VALUE" >> "$GITHUB_OUTPUT"'
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 13: ACTIONS_ALLOW_UNSECURE_COMMANDS re-enabled
# This env var re-enables the dangerous set-env and add-path workflow commands
# that were disabled for CVE-2020-15228.
echo "Checking for ACTIONS_ALLOW_UNSECURE_COMMANDS..."
while IFS= read -r file; do
    if [[ -n "$file" ]]; then
        log_issue "CRITICAL" "$file" "Enables ACTIONS_ALLOW_UNSECURE_COMMANDS - this re-enables dangerous set-env/add-path commands (CVE-2020-15228)"
        log_fix_example "unsecure_cmds" \
'# BEFORE (critical - re-enables commands disabled for CVE-2020-15228):
  env:
    ACTIONS_ALLOW_UNSECURE_COMMANDS: true
  run: echo "::set-env name=PATH::$NEW_PATH"
# AFTER (safe - use GITHUB_ENV and GITHUB_PATH files):
  run: |
    echo "PATH=$NEW_PATH" >> "$GITHUB_ENV"
    echo "/new/bin" >> "$GITHUB_PATH"'
    fi
done < <(grep -rlE 'ACTIONS_ALLOW_UNSECURE_COMMANDS.*true' "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null || true)

# Check 14: toJSON() in run blocks
# toJSON() output in shell context is dangerous because JSON can contain
# quotes, newlines, and backticks that break out of shell string contexts.
echo "Checking for toJSON() in run blocks..."
while IFS= read -r -d '' file; do
    if grep -qE 'toJSON\(' "$file" 2>/dev/null; then
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*run:/ && /toJSON\(/ { print; next }
            in_run_block && /toJSON\(/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "MEDIUM" "$file" "Uses toJSON() directly in run: block - JSON can break shell quoting; assign to env var first"
            log_fix_example "tojson" \
'# BEFORE (unsafe - JSON quotes/newlines can break shell context):
  run: |
    data='"'"'${{ toJSON(matrix.config) }}'"'"'
    echo "$data" | jq .
# AFTER (safe - env var preserves JSON without shell re-parsing):
  env:
    CONFIG_JSON: ${{ toJSON(matrix.config) }}
  run: |
    echo "$CONFIG_JSON" | jq .'
        fi
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 15: Cross-repo reusable workflow calls with mutable refs
# Calling org/repo/.github/workflows/foo.yaml@branch_name allows anyone
# with push access to that branch to modify the workflow and access all
# secrets passed to it. Pin to a full commit SHA instead.
echo "Checking for cross-repo reusable workflow calls with mutable refs..."
while IFS= read -r -d '' file; do
    unsafe_calls=$(grep -E 'uses:\s+[^./][^[:space:]]*/.github/workflows/[^[:space:]]+@' "$file" 2>/dev/null | \
        grep -vE '@[a-f0-9]{40}' || true)
    if [[ -n "$unsafe_calls" ]]; then
        log_issue "HIGH" "$file" "Cross-repo reusable workflow call uses mutable ref instead of commit SHA - pin to full SHA"
        log_fix_example "crossrepo_ref" \
'# BEFORE (unsafe - branch can be modified by anyone with push access):
  uses: org/repo/.github/workflows/build.yaml@main
# AFTER (safe - pinned to immutable commit):
  uses: org/repo/.github/workflows/build.yaml@abc123def456... # main
# To get the SHA: git ls-remote https://github.com/org/repo refs/heads/main'
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 16: Additional attacker-controlled event context in run blocks
# Beyond comment/issue/PR body/title (Check 1), many other github.event
# fields are attacker-controlled and dangerous in shell context.
# Only flags UNSAFE usage: direct interpolation in run: blocks
echo "Checking for additional attacker-controlled event fields in run blocks..."
additional_event_fields='github\.event\.(discussion\.(title|body)|review\.body|review_comment\.body|head_commit\.(message|author\.(name|email))|commits\[\*\]\.(message|author)|pages\[\*\]\.page_name|workflow_run\.head_branch|workflow_run\.head_sha|label\.name|milestone\.title)'
while IFS= read -r -d '' file; do
    if grep -qE "$additional_event_fields" "$file" 2>/dev/null; then
        unsafe_usage=$(awk -v pattern="$additional_event_fields" "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*run:/ && $0 ~ pattern { print; next }
            in_run_block && $0 ~ pattern { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "Contains attacker-controlled github.event field directly in run: block - use env var instead"
            log_fix_example "event_fields" \
'# BEFORE (unsafe - attacker controls branch name in workflow_run triggers):
  run: echo "Branch: ${{ github.event.workflow_run.head_branch }}"
# AFTER (safe - assign to env var first):
  env:
    HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
  run: echo "Branch: $HEAD_BRANCH"'
        fi
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 17: actions/github-script with direct expression interpolation
# When ${{ }} expressions containing attacker-controlled data are used in
# the script: input of actions/github-script, the expanded value is injected
# directly into JavaScript code, enabling arbitrary code execution.
echo "Checking for github-script with expression interpolation..."
while IFS= read -r -d '' file; do
    if grep -qE 'actions/github-script' "$file" 2>/dev/null; then
        unsafe_usage=$(awk '
            BEGIN { in_action = 0; in_script = 0; script_indent = 0 }
            /uses:.*actions\/github-script/ { in_action = 1; next }
            in_action && /^[[:space:]]*script:[[:space:]]*\|/ {
                match($0, /^[[:space:]]*/); script_indent = RLENGTH; in_script = 1; next
            }
            in_action && /^[[:space:]]*script:/ && /\$\{\{.*github\.event\./ { print; next }
            in_script {
                if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_script = 0; in_action = 0; next }
                match($0, /^[[:space:]]*/); curr_indent = RLENGTH
                if (curr_indent <= script_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_script = 0; in_action = 0; next }
            }
            in_script && /\$\{\{.*github\.event\./ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "actions/github-script contains \${{ github.event.* }} in script body - JavaScript injection risk"
            log_fix_example "github_script" \
'# BEFORE (unsafe - JS injection via issue title):
  uses: actions/github-script@v7
  with:
    script: |
      const title = "${{ github.event.issue.title }}";
      console.log(title);
# AFTER (safe - read from env var via process.env):
  uses: actions/github-script@v7
  env:
    ISSUE_TITLE: ${{ github.event.issue.title }}
  with:
    script: |
      const title = process.env.ISSUE_TITLE;
      console.log(title);'
        fi
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 18: secrets.* directly interpolated in run blocks
# Secrets should always be passed via env: to avoid leaking in error traces
# or debug output if the shell command fails unexpectedly.
echo "Checking for secrets interpolation in run blocks..."
while IFS= read -r -d '' file; do
    if grep -qE '\$\{\{.*secrets\.' "$file" 2>/dev/null; then
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*run:/ && /\$\{\{.*secrets\./ { print; next }
            in_run_block && /\$\{\{.*secrets\./ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "Contains \${{ secrets.* }} directly in run: block - pass via env var to avoid exposure in error traces"
            log_fix_example "secrets_run" \
'# BEFORE (risky - secret expanded into shell source, may leak in traces):
  run: git remote set-url origin https://${{ secrets.MY_TOKEN }}@github.com/org/repo.git
# AFTER (safe - secret stays in env var, not embedded in shell source):
  env:
    MY_TOKEN: ${{ secrets.MY_TOKEN }}
  run: git remote set-url origin "https://${MY_TOKEN}@github.com/org/repo.git"'
        fi
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 19: matrix.* directly interpolated in run blocks
# Matrix values are typically hardcoded, but should still use env: for
# consistency and defense-in-depth against future changes.
echo "Checking for matrix.* interpolation in run blocks..."
while IFS= read -r -d '' file; do
    if grep -qE '\$\{\{.*matrix\.' "$file" 2>/dev/null; then
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*run:/ && /\$\{\{.*matrix\./ { print; next }
            in_run_block && /\$\{\{.*matrix\./ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "MEDIUM" "$file" "Contains \${{ matrix.* }} directly in run: block - use env var instead"
            log_fix_example "matrix_run" \
'# BEFORE (matrix value is macro-expanded into shell code):
  run: echo "Testing on ${{ matrix.os }}"
# AFTER (safe - assign to env var, reference as shell variable):
  env:
    MATRIX_OS: ${{ matrix.os }}
  run: echo "Testing on $MATRIX_OS"'
        fi
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 20: github.repository and github.event_name directly in run blocks
# These have restricted character sets and low injection risk, but using env:
# is still preferred for defense-in-depth and consistency.
echo "Checking for github.repository/github.event_name interpolation in run blocks..."
while IFS= read -r -d '' file; do
    if grep -qE '\$\{\{.*(github\.repository[^_]|github\.event_name)' "$file" 2>/dev/null; then
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*run:/ && /\$\{\{.*(github\.repository[^_]|github\.event_name)/ { print; next }
            in_run_block && /\$\{\{.*(github\.repository[^_]|github\.event_name)/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "LOW" "$file" "Contains \${{ github.repository }} or \${{ github.event_name }} directly in run: block - prefer env var for defense-in-depth"
            log_fix_example "github_context_run" \
'# BEFORE (low risk but inconsistent with defense-in-depth):
  run: gh pr view 123 --repo "${{ github.repository }}"
# AFTER (consistent - use workflow-level env or step env):
  env:
    REPO: ${{ github.repository }}
  run: gh pr view 123 --repo "$REPO"'
        fi
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

echo ""
echo "=========================================="
echo "Scan Complete"
echo "=========================================="
echo ""

if [[ $ISSUES_FOUND -eq 0 ]]; then
    echo -e "${GREEN}No security issues found!${NC}"
    exit 0
else
    echo -e "${YELLOW}Found $ISSUES_FOUND potential security issues${NC}"
    echo ""
    echo "Remediation guidance:"
    echo "  - CRITICAL: Fix immediately - allows arbitrary code execution or secret exfiltration"
    echo "  - HIGH: Fix immediately - direct shell injection risk"
    echo "  - MEDIUM: Fix soon - potential injection or supply chain risk"
    echo "  - LOW: Defense in depth - address in regular maintenance"
    echo ""
    echo "For more details, see the security remediation plan."

    if [[ "$STRICT_MODE" == "true" ]]; then
        exit 1
    fi
    exit 0
fi
