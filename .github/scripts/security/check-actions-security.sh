#!/usr/bin/env bash
# GitHub Actions Security Linting Script
# Detects common shell injection and security anti-patterns in workflows/actions
#
# Usage: ./check-actions-security.sh [OPTIONS] [FILE...]
#   -h, --help    Show help message with check descriptions
#   --strict      Exit with error code if any issues found
#
# If no FILEs are provided, scans all .yml/.yaml files in .github/workflows and .github/actions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
GITHUB_DIR="$REPO_ROOT/.github"

STRICT_MODE=false
MACHINE_OUTPUT=false
SKIP_AGGREGATE=false
FORMAT_RESULTS_FILE=""
ISSUES_FOUND=0
CHECKS_TO_RUN=()
CURRENT_CHECK=""
MAX_CHECK_NUM=49

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

# =============================================================================
# Usage / Help
# =============================================================================

usage() {
    cat <<'EOF'
Usage: ./check-actions-security.sh [OPTIONS] [FILE...]

Options:
  -h, --help              Show this help message and exit
  -c, --checks LIST       Run only specified checks (comma-separated, supports ranges)
                          Example: -c 1-3,19 runs checks 1, 2, 3, and 19
  --strict                Exit with error code if any issues found
  --machine-output        Output tab-delimited format for parallel processing
  --skip-aggregate        Skip aggregate checks (5, 6) for parallel workers
  --format-results FILE   Read results file and display with deduplicated examples

If no FILEs are provided, scans all .yml/.yaml files in .github/workflows and .github/actions.

Checks:
  1  Direct event data interpolation in run blocks (HIGH)
  2  eval usage (HIGH)
  3  bash -c with direct interpolation (MEDIUM)
  4  pull_request_target with checkout (HIGH)
  5  Unpinned external action references (LOW)
  6  secrets: inherit usage (LOW)
  7  Token exposure in logs (HIGH)
  8  Overly broad permissions (MEDIUM)
  9  Ref-based injection patterns (MEDIUM)
 10  curl | bash patterns (HIGH)
 11  inputs/outputs interpolation in run blocks (HIGH)
 12  Deprecated ::set-output/::save-state commands (MEDIUM)
 13  ACTIONS_ALLOW_UNSECURE_COMMANDS enabled (CRITICAL)
 14  toJSON() in run blocks (MEDIUM)
 15  Cross-repo reusable workflow mutable refs (HIGH)
 16  Additional attacker-controlled event fields (HIGH)
 17  github-script with expression interpolation (HIGH)
 18  secrets.* interpolation in run blocks (HIGH)
 19  matrix.* interpolation in run blocks (MEDIUM)
 20  github.repository/event_name in run blocks (LOW)
 21  pull_request_target privilege boundary issues (HIGH/MEDIUM)
 22  workflow_run privileged follow-up patterns (MEDIUM)
 23  self-hosted runners on untrusted triggers (HIGH)
 24  Write-capable permissions on untrusted triggers (MEDIUM)
 25  Mutable container and docker:// image references (MEDIUM)
 26  fromJSON() with attacker-controlled input (MEDIUM)
 27  Dynamic expression interpolation in 'uses:' (HIGH)
 28  Cache key injection patterns (MEDIUM)
 29  Artifact path expressions (MEDIUM)
 30  URL construction in HTTP clients (MEDIUM)
 31  Container volume mount expressions (MEDIUM)
 32  Expression interpolation in GITHUB_ENV/PATH/OUTPUT writes (HIGH)
 33  Missing explicit permissions key (LOW)
 34  Concurrency group with attacker-controlled data (MEDIUM)
 35  issue_comment trigger without authorization gate (MEDIUM)
 36  working-directory with attacker-controlled expressions (MEDIUM)
 37  github-script with inputs/outputs/secrets interpolation (HIGH)
 38  github-script dynamic code execution primitives (HIGH)
 39  Inline interpreter eval/exec patterns in run blocks (HIGH)
 40  github-script command execution APIs (HIGH)
 41  Inline interpreter command execution APIs (HIGH)
 42  Shell command execution from variables (HIGH)
 43  Remote script execution variants beyond pipes (HIGH)
 44  Attacker-controlled env vars written to GITHUB_ENV/PATH/OUTPUT (HIGH)
 45  Expression injection in 'if:' conditions (HIGH)
 46  Bracket notation bypass in expressions (HIGH)
 47  Additional attacker-controlled event contexts (HIGH)
 48  Export statements with direct interpolation (HIGH)
 49  Dynamic 'shell:' property expressions (MEDIUM)
EOF
    exit 0
}

# =============================================================================
# Parse Check List (comma-separated with range support)
# =============================================================================

parse_checks() {
    local input="$1"
    local -a result=()

    IFS=',' read -ra parts <<< "$input"
    for part in "${parts[@]}"; do
        part="${part// /}"  # Remove whitespace
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start="${BASH_REMATCH[1]}"
            local end="${BASH_REMATCH[2]}"
            if [[ $start -gt $end ]]; then
                echo "Error: Invalid range $part (start > end)" >&2
                exit 1
            fi
            for ((i=start; i<=end; i++)); do
                if [[ $i -ge 1 && $i -le $MAX_CHECK_NUM ]]; then
                    result+=("$i")
                else
                    echo "Error: Check $i is out of range (valid: 1-$MAX_CHECK_NUM)" >&2
                    exit 1
                fi
            done
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            if [[ $part -ge 1 && $part -le $MAX_CHECK_NUM ]]; then
                result+=("$part")
            else
                echo "Error: Check $part is out of range (valid: 1-$MAX_CHECK_NUM)" >&2
                exit 1
            fi
        else
            echo "Error: Invalid check specification: $part" >&2
            exit 1
        fi
    done

    CHECKS_TO_RUN=("${result[@]}")
}

# =============================================================================
# Argument Parsing
# =============================================================================

FILES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        -c|--checks)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --checks requires an argument" >&2
                exit 1
            fi
            parse_checks "$2"
            shift 2
            ;;
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --machine-output)
            MACHINE_OUTPUT=true
            shift
            ;;
        --skip-aggregate)
            SKIP_AGGREGATE=true
            shift
            ;;
        --format-results)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --format-results requires a file argument" >&2
                exit 1
            fi
            FORMAT_RESULTS_FILE="$2"
            shift 2
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Default to all checks if none specified
if [[ ${#CHECKS_TO_RUN[@]} -eq 0 ]]; then
    CHECKS_TO_RUN=($(seq 1 "$MAX_CHECK_NUM"))
fi

# =============================================================================
# Helper Functions
# =============================================================================

log_issue() {
    local severity="$1"
    local file="$2"
    local message="$3"

    if [[ "$MACHINE_OUTPUT" == "true" ]]; then
        # Tab-delimited: CHECK<tab>SEVERITY<tab>FILE<tab>MESSAGE
        printf '%s\t%s\t%s\t%s\n' "$CURRENT_CHECK" "$severity" "$file" "$message"
    else
        local color="$YELLOW"
        if [[ "$severity" == "CRITICAL" ]] || [[ "$severity" == "HIGH" ]]; then
            color="$RED"
        fi
        echo -e "${color}[$severity]${NC} $file: $message"
    fi
    ((ISSUES_FOUND++)) || true
}

# Prints a fix example for a check class (only once per class).
declare -A shown_examples
log_fix_example() {
    local class="$1"
    local example="$2"

    if [[ "${shown_examples[$class]:-}" == "true" ]]; then
        return
    fi
    shown_examples[$class]=true

    echo -e "${CYAN}  Fix example:${NC}"
    while IFS= read -r line; do
        echo -e "    ${DIM}${line}${NC}"
    done <<< "$example"
    echo ""
}

has_untrusted_trigger() {
    local file="$1"
    grep -qE '(^|[[:space:][:punct:]])(pull_request_target|pull_request|issue_comment|discussion|discussion_comment)([[:space:][:punct:]]|$)' "$file" 2>/dev/null
}

# =============================================================================
# AWK Helper for Run Block Detection
# =============================================================================

AWK_RUN_BLOCK_DETECT='
    BEGIN { in_run_block = 0; run_indent = 0 }
    /^[[:space:]]*(-[[:space:]]+)?run:[[:space:]]*\|/ {
        match($0, /^[[:space:]]*/); run_indent = RLENGTH; in_run_block = 1; next
    }
    in_run_block {
        if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_run_block = 0; next }
        match($0, /^[[:space:]]*/); curr_indent = RLENGTH
        if (curr_indent <= run_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_run_block = 0; next }
    }
'

# =============================================================================
# File List Population
# =============================================================================

if [[ ${#FILES[@]} -eq 0 ]]; then
    while IFS= read -r -d '' f; do
        FILES+=("$f")
    done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" \( -name "*.yml" -o -name "*.yaml" \) -print0 2>/dev/null)
fi

# =============================================================================
# Check 1: Direct input interpolation in run blocks (high risk patterns)
# =============================================================================
check_1_description="dangerous direct interpolation patterns"
check_1_severity="HIGH"

example_check_1() {
    cat <<'EOF'
# BEFORE (unsafe - attacker controls PR title):
  run: echo "${{ github.event.pull_request.title }}"
# AFTER (safe - shell variable cannot inject commands):
  env:
    PR_TITLE: ${{ github.event.pull_request.title }}
  run: echo "$PR_TITLE"
EOF
}

check_1() {
    local file="$1"
    local dangerous_patterns='github\.event\.(comment\.body|issue\.body|pull_request\.body|pull_request\.title)'
    if grep -qE "$dangerous_patterns" "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*github\.event\.(comment\.body|issue\.body|pull_request\.body|pull_request\.title)/ { print; next }
            in_run_block && /\$\{\{.*github\.event\.(comment\.body|issue\.body|pull_request\.body|pull_request\.title)/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "Contains dangerous event data interpolation directly in run: block (comment/issue/PR body/title)"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 2: eval usage
# =============================================================================
check_2_description="eval usage"
check_2_severity="HIGH"

example_check_2() {
    cat <<'EOF'
# BEFORE (unsafe - eval interprets arbitrary code):
  run: eval "$USER_INPUT"
# AFTER (safe - use direct execution or a case/switch):
  run: |
    case "$VALIDATED_CMD" in
      build)  make build ;;
      test)   make test ;;
      *)      echo "Unknown command"; exit 1 ;;
    esac
EOF
}

check_2() {
    local file="$1"
    if grep -qE 'eval\s' "$file" 2>/dev/null; then
        log_issue "HIGH" "$file" "Contains 'eval' statement - consider safer alternatives"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 3: bash -c with direct ${{ }} interpolation
# =============================================================================
check_3_description="bash -c with direct interpolation"
check_3_severity="MEDIUM"

example_check_3() {
    cat <<'EOF'
# BEFORE (unsafe - interpolation happens before shell parsing):
  run: bash -c "echo ${{ inputs.command }}"
# AFTER (safe - pass via env var):
  env:
    CMD: ${{ inputs.command }}
  run: bash -c "echo \"$CMD\""
EOF
}

check_3() {
    local file="$1"
    if grep -qE 'bash.*-c.*\$\{\{' "$file" 2>/dev/null; then
        log_issue "MEDIUM" "$file" "Contains 'bash -c' with direct \${{ }} - ensure input is via env var"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 4: pull_request_target with potentially unsafe patterns
# =============================================================================
check_4_description="pull_request_target with checkout"
check_4_severity="HIGH"

example_check_4() {
    cat <<'EOF'
# pull_request_target runs with BASE repo privileges but can check out HEAD code.
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
    - run: make test
EOF
}

check_4() {
    local file="$1"
    if grep -qE 'pull_request_target' "$file" 2>/dev/null && grep -q 'actions/checkout' "$file" 2>/dev/null; then
        log_issue "HIGH" "$file" "Uses pull_request_target with checkout - verify PR code is not executed unsafely"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 5: Count unpinned external action references (aggregate check)
# =============================================================================
check_5_description="unpinned external action references"
check_5_severity="LOW"
check_5_aggregate=true

example_check_5() {
    cat <<'EOF'
# BEFORE (mutable tag - can be moved to point to different code):
  uses: actions/checkout@v4
# AFTER (pinned to immutable commit SHA):
  uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
EOF
}

check_5() {
    local external_unpinned=0
    for file in "$@"; do
        local count
        count=$(awk '
            /^[[:space:]]*-[[:space:]]*uses:[[:space:]]*[^.#[:space:]][^[:space:]#]*@|^[[:space:]]*uses:[[:space:]]*[^.#[:space:]][^[:space:]#]*@/ {
                ref = $0
                sub(/^[[:space:]]*-[[:space:]]*uses:[[:space:]]*/, "", ref)
                sub(/^[[:space:]]*uses:[[:space:]]*/, "", ref)
                sub(/[[:space:]]+#.*$/, "", ref)
                gsub(/^[\"\047]|[\"\047]$/, "", ref)

                if (ref ~ /^docker:\/\//) next
                if (ref ~ /^\.\//) next
                if (ref ~ /^tenstorrent\//) next
                if (ref ~ /\/\.github\/workflows\//) next

                n = split(ref, parts, "@")
                refspec = parts[n]
                if (refspec !~ /^[a-f0-9]{40}$/) {
                    count++
                }
            }
            END { print count + 0 }
        ' "$file" 2>/dev/null)
        count="${count:-0}"
        if [[ "$count" =~ ^[0-9]+$ ]]; then
            ((external_unpinned += count)) || true
        fi
    done
    external_unpinned="${external_unpinned:-0}"
    if [[ "$external_unpinned" =~ ^[0-9]+$ ]] && [[ "$external_unpinned" -gt 0 ]]; then
        log_issue "LOW" "Multiple files" "Found $external_unpinned external actions not pinned to a full commit SHA - consider pinning to immutable refs"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 6: Count secrets: inherit usage (aggregate check)
# =============================================================================
check_6_description="secrets: inherit usage"
check_6_severity="LOW"
check_6_aggregate=true

example_check_6() {
    cat <<'EOF'
# BEFORE (passes ALL repo secrets to called workflow):
  uses: ./.github/workflows/build.yaml
  secrets: inherit
# AFTER (passes only the secrets the called workflow needs):
  uses: ./.github/workflows/build.yaml
  secrets:
    DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
EOF
}

check_6() {
    local secrets_inherit_count=0
    for file in "$@"; do
        if grep -qE 'secrets:\s*inherit' "$file" 2>/dev/null; then
            ((secrets_inherit_count++)) || true
        fi
    done
    if [[ "$secrets_inherit_count" -gt 50 ]]; then
        log_issue "LOW" "Multiple files" "Found $secrets_inherit_count workflows using 'secrets: inherit' - consider explicit secret passing"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 7: GITHUB_TOKEN or ACTIONS_RUNTIME_TOKEN exposure in logs
# =============================================================================
check_7_description="potential token exposure in logs"
check_7_severity="HIGH"

example_check_7() {
    cat <<'EOF'
# BEFORE (token value visible in workflow logs):
  run: echo "Token: $GITHUB_TOKEN"
# AFTER (use token directly without logging, or mask it):
  run: |
    echo "::add-mask::$GITHUB_TOKEN"
    curl -H "Authorization: Bearer $GITHUB_TOKEN" ...
EOF
}

check_7() {
    local file="$1"
    if grep -qE 'echo.*\$\{?\{?(GITHUB_TOKEN|ACTIONS_RUNTIME_TOKEN|ACTIONS_ID_TOKEN_REQUEST_TOKEN)' "$file" 2>/dev/null; then
        log_issue "HIGH" "$file" "Potential token exposure in echo/print statement"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 8: Overly broad permissions
# =============================================================================
check_8_description="overly broad permissions"
check_8_severity="MEDIUM"

example_check_8() {
    cat <<'EOF'
# BEFORE (grants write access to everything):
  permissions: write-all
# AFTER (grant only what is needed):
  permissions:
    contents: read
    pull-requests: write
EOF
}

check_8() {
    local file="$1"
    if grep -qE 'permissions:.*write-all|write-all' "$file" 2>/dev/null; then
        log_issue "MEDIUM" "$file" "Uses 'write-all' permissions - apply principle of least privilege"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 9: Ref-based injection vectors (head.ref, base.ref in ${{ }})
# =============================================================================
check_9_description="ref-based injection patterns"
check_9_severity="MEDIUM"

example_check_9() {
    cat <<'EOF'
# BEFORE (unsafe - branch name can contain shell metacharacters):
  run: echo "Branch: ${{ github.head_ref }}"
# AFTER (safe - shell variable is not re-parsed):
  env:
    HEAD_REF: ${{ github.head_ref }}
  run: echo "Branch: $HEAD_REF"
EOF
}

check_9() {
    local file="$1"
    if grep -qE '\$\{\{.*(\.head[_.]ref|\.base[_.]ref|github\.head_ref|github\.base_ref)' "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*(\.head[_.]ref|\.base[_.]ref|github\.head_ref|github\.base_ref)/ { print; next }
            in_run_block && /\$\{\{.*(\.head[_.]ref|\.base[_.]ref|github\.head_ref|github\.base_ref)/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "MEDIUM" "$file" "Contains head.ref/base.ref interpolation directly in run: block - use env var instead"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 10: curl | bash patterns
# =============================================================================
check_10_description="curl pipe to shell patterns"
check_10_severity="HIGH"

example_check_10() {
    cat <<'EOF'
# BEFORE (unsafe - executes remote code without verification):
  run: curl -sSL https://example.com/install.sh | bash
# AFTER (safe - download, inspect, then execute):
  run: |
    curl -sSL -o install.sh https://example.com/install.sh
    sha256sum -c <<< "expected_hash  install.sh"
    bash install.sh
EOF
}

check_10() {
    local file="$1"
    if grep -qE 'curl.*\|\s*(ba)?sh|wget.*\|\s*(ba)?sh' "$file" 2>/dev/null; then
        log_issue "HIGH" "$file" "Contains 'curl | bash' pattern - download and verify scripts before execution"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 11: inputs.*, steps.*.outputs.*, and needs.*.outputs.* direct interpolation
# =============================================================================
check_11_description="inputs/outputs interpolation in run blocks"
check_11_severity="HIGH"

example_check_11() {
    cat <<'EOF'
# BEFORE (unsafe - input/output is macro-expanded into shell code):
  run: echo "${{ needs.build.outputs.pr-number }}"
# AFTER (safe - assign to env var, reference as shell variable):
  env:
    PR_NUMBER: ${{ needs.build.outputs.pr-number }}
  run: echo "$PR_NUMBER"
EOF
}

check_11() {
    local file="$1"
    if grep -qE '\$\{\{.*(inputs\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.)' "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*(inputs\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.)/ { print; next }
            in_run_block && /\$\{\{.*(inputs\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.)/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "Contains \${{ inputs.* }}, \${{ steps.*.outputs.* }}, or \${{ needs.*.outputs.* }} directly in run: block - use env var instead"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 12: Deprecated ::set-output and ::save-state workflow commands
# =============================================================================
check_12_description="deprecated workflow commands"
check_12_severity="MEDIUM"

example_check_12() {
    cat <<'EOF'
# BEFORE (deprecated - vulnerable to log injection):
  run: echo "::set-output name=result::$VALUE"
# AFTER (safe - writes to environment file):
  run: echo "result=$VALUE" >> "$GITHUB_OUTPUT"
EOF
}

check_12() {
    local file="$1"
    if grep -qE '::set-output |::save-state ' "$file" 2>/dev/null; then
        log_issue "MEDIUM" "$file" "Uses deprecated ::set-output or ::save-state command - use \$GITHUB_OUTPUT/\$GITHUB_STATE files instead"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 13: ACTIONS_ALLOW_UNSECURE_COMMANDS re-enabled
# =============================================================================
check_13_description="ACTIONS_ALLOW_UNSECURE_COMMANDS"
check_13_severity="CRITICAL"

example_check_13() {
    cat <<'EOF'
# BEFORE (critical - re-enables commands disabled for CVE-2020-15228):
  env:
    ACTIONS_ALLOW_UNSECURE_COMMANDS: true
  run: echo "::set-env name=PATH::$NEW_PATH"
# AFTER (safe - use GITHUB_ENV and GITHUB_PATH files):
  run: |
    echo "PATH=$NEW_PATH" >> "$GITHUB_ENV"
    echo "/new/bin" >> "$GITHUB_PATH"
EOF
}

check_13() {
    local file="$1"
    if grep -qE 'ACTIONS_ALLOW_UNSECURE_COMMANDS.*true' "$file" 2>/dev/null; then
        log_issue "CRITICAL" "$file" "Enables ACTIONS_ALLOW_UNSECURE_COMMANDS - this re-enables dangerous set-env/add-path commands (CVE-2020-15228)"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 14: toJSON() in run blocks
# =============================================================================
check_14_description="toJSON() in run blocks"
check_14_severity="MEDIUM"

example_check_14() {
    cat <<'EOF'
# BEFORE (unsafe - JSON quotes/newlines can break shell context):
  run: |
    data='${{ toJSON(matrix.config) }}'
    echo "$data" | jq .
# AFTER (safe - env var preserves JSON without shell re-parsing):
  env:
    CONFIG_JSON: ${{ toJSON(matrix.config) }}
  run: |
    echo "$CONFIG_JSON" | jq .
EOF
}

check_14() {
    local file="$1"
    if grep -qE 'toJSON\(' "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /toJSON\(/ { print; next }
            in_run_block && /toJSON\(/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "MEDIUM" "$file" "Uses toJSON() directly in run: block - JSON can break shell quoting; assign to env var first"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 15: Cross-repo reusable workflow calls with mutable refs
# =============================================================================
check_15_description="cross-repo reusable workflow calls with mutable refs"
check_15_severity="HIGH"

example_check_15() {
    cat <<'EOF'
# BEFORE (unsafe - branch can be modified by anyone with push access):
  uses: org/repo/.github/workflows/build.yaml@main
# AFTER (safe - pinned to immutable commit):
  uses: org/repo/.github/workflows/build.yaml@abc123def456... # main
# To get the SHA: git ls-remote https://github.com/org/repo refs/heads/main
EOF
}

check_15() {
    local file="$1"
    local unsafe_calls
    unsafe_calls=$(awk '
        /^[[:space:]]*uses:[[:space:]]*[^.#[:space:]][^[:space:]#]*\/\.github\/workflows\/[^[:space:]#]+@/ {
            ref = $0
            sub(/^[[:space:]]*uses:[[:space:]]*/, "", ref)
            sub(/[[:space:]]+#.*$/, "", ref)
            gsub(/^[\"\047]|[\"\047]$/, "", ref)
            n = split(ref, parts, "@")
            refspec = parts[n]
            if (refspec !~ /^[a-f0-9]{40}$/) {
                print ref
            }
        }
    ' "$file" 2>/dev/null)
    if [[ -n "$unsafe_calls" ]]; then
        log_issue "HIGH" "$file" "Cross-repo reusable workflow call uses mutable ref instead of commit SHA - pin to full SHA"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 16: Additional attacker-controlled event context in run blocks
# =============================================================================
check_16_description="additional attacker-controlled event fields in run blocks"
check_16_severity="HIGH"

example_check_16() {
    cat <<'EOF'
# BEFORE (unsafe - attacker controls branch name in workflow_run triggers):
  run: echo "Branch: ${{ github.event.workflow_run.head_branch }}"
# AFTER (safe - assign to env var first):
  env:
    HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
  run: echo "Branch: $HEAD_BRANCH"
EOF
}

check_16() {
    local file="$1"
    local additional_event_fields='github\.event\.(discussion\.(title|body)|review\.body|review_comment\.body|head_commit\.(message|author\.(name|email))|commits\[\*\]\.(message|author)|pages\[\*\]\.page_name|workflow_run\.head_branch|workflow_run\.head_sha|label\.name|milestone\.title)'
    if grep -qE "$additional_event_fields" "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk -v pattern="$additional_event_fields" "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && $0 ~ pattern { print; next }
            in_run_block && $0 ~ pattern { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "Contains attacker-controlled github.event field directly in run: block - use env var instead"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 17: actions/github-script with direct expression interpolation
# =============================================================================
check_17_description="github-script with expression interpolation"
check_17_severity="HIGH"

example_check_17() {
    cat <<'EOF'
# BEFORE (unsafe - JS injection via issue title):
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
      console.log(title);
EOF
}

check_17() {
    local file="$1"
    if grep -qE 'actions/github-script' "$file" 2>/dev/null; then
        local unsafe_usage
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
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 18: secrets.* directly interpolated in run blocks
# =============================================================================
check_18_description="secrets interpolation in run blocks"
check_18_severity="HIGH"

example_check_18() {
    cat <<'EOF'
# BEFORE (risky - secret expanded into shell source, may leak in traces):
  run: git remote set-url origin https://${{ secrets.MY_TOKEN }}@github.com/org/repo.git
# AFTER (safe - secret stays in env var, not embedded in shell source):
  env:
    MY_TOKEN: ${{ secrets.MY_TOKEN }}
  run: git remote set-url origin "https://${MY_TOKEN}@github.com/org/repo.git"
EOF
}

check_18() {
    local file="$1"
    if grep -qE '\$\{\{.*secrets\.' "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*secrets\./ { print; next }
            in_run_block && /\$\{\{.*secrets\./ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "Contains \${{ secrets.* }} directly in run: block - pass via env var to avoid exposure in error traces"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 19: matrix.* directly interpolated in run blocks
# =============================================================================
check_19_description="matrix.* interpolation in run blocks"
check_19_severity="MEDIUM"

example_check_19() {
    cat <<'EOF'
# BEFORE (matrix value is macro-expanded into shell code):
  run: echo "Testing on ${{ matrix.os }}"
# AFTER (safe - assign to env var, reference as shell variable):
  env:
    MATRIX_OS: ${{ matrix.os }}
  run: echo "Testing on $MATRIX_OS"
EOF
}

check_19() {
    local file="$1"
    if grep -qE '\$\{\{.*matrix\.' "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*matrix\./ { print; next }
            in_run_block && /\$\{\{.*matrix\./ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "MEDIUM" "$file" "Contains \${{ matrix.* }} directly in run: block - use env var instead"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 20: github.repository and github.event_name directly in run blocks
# =============================================================================
check_20_description="github.repository/github.event_name interpolation in run blocks"
check_20_severity="LOW"

example_check_20() {
    cat <<'EOF'
# BEFORE (low risk but inconsistent with defense-in-depth):
  run: gh pr view 123 --repo "${{ github.repository }}"
# AFTER (consistent - use workflow-level env or step env):
  env:
    REPO: ${{ github.repository }}
  run: gh pr view 123 --repo "$REPO"
EOF
}

check_20() {
    local file="$1"
    if grep -qE '\$\{\{.*(github\.repository[^_]|github\.event_name)' "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*(github\.repository[^_]|github\.event_name)/ { print; next }
            in_run_block && /\$\{\{.*(github\.repository[^_]|github\.event_name)/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "LOW" "$file" "Contains \${{ github.repository }} or \${{ github.event_name }} directly in run: block - prefer env var for defense-in-depth"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 21: pull_request_target privilege boundary issues
# =============================================================================
check_21_description="pull_request_target privilege boundary issues"
check_21_severity="HIGH"

example_check_21() {
    cat <<'EOF'
# BEFORE (unsafe - PR head checkout plus persisted credentials):
  on: pull_request_target
  steps:
    - uses: actions/checkout@v4
      with:
        ref: ${{ github.event.pull_request.head.sha }}
    - uses: ./.github/actions/build
# AFTER (safer - avoid PR head checkout in privileged jobs and do not persist credentials):
  on: pull_request_target
  steps:
    - uses: actions/checkout@v4
      with:
        persist-credentials: false
    - run: ./scripts/label-pr.sh
EOF
}

check_21() {
    local file="$1"
    local found=0

    if ! grep -qE 'pull_request_target' "$file" 2>/dev/null; then
        return 0
    fi

    if grep -q 'actions/checkout' "$file" 2>/dev/null && \
       grep -qE 'github\.event\.pull_request\.head\.(sha|ref|repo\.full_name)|github\.head_ref' "$file" 2>/dev/null; then
        log_issue "HIGH" "$file" "pull_request_target checks out the PR head ref/repository - this can execute untrusted code with base-repository privileges"
        found=1
    fi

    if grep -q 'actions/checkout' "$file" 2>/dev/null && \
       grep -qE 'github\.event\.pull_request\.head\.(sha|ref|repo\.full_name)|github\.head_ref' "$file" 2>/dev/null && \
       grep -qE '^[[:space:]]*uses:[[:space:]]*\./' "$file" 2>/dev/null; then
        log_issue "HIGH" "$file" "pull_request_target invokes a local action after untrusted checkout - local action code comes from the checked-out repository"
        found=1
    fi

    if grep -q 'actions/checkout' "$file" 2>/dev/null && \
       ! grep -qE 'persist-credentials:[[:space:]]*false' "$file" 2>/dev/null; then
        log_issue "MEDIUM" "$file" "pull_request_target uses actions/checkout without persist-credentials: false - the workflow token may remain in git config"
        found=1
    fi

    [[ $found -eq 0 ]]
}

# =============================================================================
# Check 22: workflow_run privileged follow-up patterns
# =============================================================================
check_22_description="workflow_run privileged follow-up patterns"
check_22_severity="MEDIUM"

example_check_22() {
    cat <<'EOF'
# BEFORE (risky - privileged workflow_run job consumes upstream artifacts):
  on: workflow_run
  permissions:
    contents: write
  steps:
    - run: gh run download "${{ github.event.workflow_run.id }}"
# AFTER (safer - keep follow-up job read-only or validate artifacts before privileged use):
  on: workflow_run
  permissions:
    contents: read
  steps:
    - run: ./scripts/verify-artifacts.sh
EOF
}

check_22() {
    local file="$1"
    local found=0

    if ! grep -qE 'workflow_run' "$file" 2>/dev/null; then
        return 0
    fi

    if grep -qE 'github\.event\.workflow_run\.id|gh[[:space:]]+run[[:space:]]+download|dawidd6/action-download-artifact|/actions/artifacts|run_id:[[:space:]]*\$\{\{[[:space:]]*github\.event\.workflow_run\.id' "$file" 2>/dev/null && \
       grep -qE 'secrets:[[:space:]]*inherit|\$\{\{[[:space:]]*secrets\.|write-all|contents:[[:space:]]*write|actions:[[:space:]]*write|packages:[[:space:]]*write|pull-requests:[[:space:]]*write|id-token:[[:space:]]*write|uses:[[:space:]]*actions/checkout' "$file" 2>/dev/null; then
        log_issue "MEDIUM" "$file" "workflow_run downloads or references upstream run artifacts while also having elevated capabilities - validate artifacts before privileged use"
        found=1
    fi

    [[ $found -eq 0 ]]
}

# =============================================================================
# Check 23: self-hosted runners on untrusted triggers
# =============================================================================
check_23_description="self-hosted runners on untrusted triggers"
check_23_severity="HIGH"

example_check_23() {
    cat <<'EOF'
# BEFORE (risky - public PRs can reach self-hosted infrastructure):
  on: pull_request
  jobs:
    test:
      runs-on: [self-hosted, linux]
# AFTER (safer - use GitHub-hosted runners for untrusted events):
  on: pull_request
  jobs:
    test:
      runs-on: ubuntu-latest
EOF
}

check_23() {
    local file="$1"
    if has_untrusted_trigger "$file" && grep -qE 'self-hosted' "$file" 2>/dev/null; then
        log_issue "HIGH" "$file" "Untrusted trigger targets a self-hosted runner - isolate self-hosted runners from public-input workflows"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 24: write-capable permissions on untrusted triggers
# =============================================================================
check_24_description="write-capable permissions on untrusted triggers"
check_24_severity="MEDIUM"

example_check_24() {
    cat <<'EOF'
# BEFORE (risky - untrusted trigger gets write-capable token scopes):
  on: pull_request_target
  permissions:
    contents: write
    pull-requests: write
# AFTER (safer - keep token read-only unless a separate trusted job needs elevation):
  on: pull_request_target
  permissions:
    contents: read
EOF
}

check_24() {
    local file="$1"
    local scoped_write

    if ! has_untrusted_trigger "$file"; then
        return 0
    fi

    scoped_write=$(awk '
        /^[[:space:]]*permissions:[[:space:]]*.*write/ { print; next }
        /^[[:space:]]*(actions|attestations|checks|contents|deployments|discussions|id-token|issues|packages|pages|pull-requests|repository-projects|security-events|statuses):[[:space:]]*write([[:space:]]|$)/ { print }
    ' "$file" 2>/dev/null)

    if [[ -n "$scoped_write" ]] && ! grep -qE 'permissions:[[:space:]]*write-all|write-all' "$file" 2>/dev/null; then
        log_issue "MEDIUM" "$file" "Untrusted trigger requests write-capable token scopes - keep permissions minimal and isolate privileged jobs"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 25: Mutable container and docker:// image references
# =============================================================================
check_25_description="mutable container and docker image references"
check_25_severity="MEDIUM"

example_check_25() {
    cat <<'EOF'
# BEFORE (mutable image tags can be retargeted):
  container:
    image: ghcr.io/example/build-env:latest
  steps:
    - uses: docker://alpine:3.20
# AFTER (pin to immutable digests):
  container:
    image: ghcr.io/example/build-env@sha256:0123456789abcdef...
  steps:
    - uses: docker://alpine@sha256:0123456789abcdef...
EOF
}

check_25() {
    local file="$1"
    local found=0
    local mutable_docker_uses
    local mutable_images

    mutable_docker_uses=$(awk '
        /^[[:space:]]*-[[:space:]]*uses:[[:space:]]*docker:\/\// || /^[[:space:]]*uses:[[:space:]]*docker:\/\// {
            ref = $0
            sub(/^[[:space:]]*-[[:space:]]*uses:[[:space:]]*docker:\/\//, "", ref)
            sub(/^[[:space:]]*uses:[[:space:]]*docker:\/\//, "", ref)
            sub(/[[:space:]]+#.*$/, "", ref)
            gsub(/^[\"\047]|[\"\047]$/, "", ref)
            if (ref !~ /@sha256:[a-f0-9]{64}$/) {
                print ref
            }
        }
    ' "$file" 2>/dev/null)

    if [[ -n "$mutable_docker_uses" ]]; then
        log_issue "MEDIUM" "$file" "Uses docker:// image reference without an immutable @sha256 digest"
        found=1
    fi

    mutable_images=$(awk '
        /^[[:space:]]*image:[[:space:]]*/ {
            image = $0
            sub(/^[[:space:]]*image:[[:space:]]*/, "", image)
            sub(/[[:space:]]+#.*$/, "", image)
            gsub(/^[\"\047]|[\"\047]$/, "", image)

            if (image == "Dockerfile" || image ~ /^\.\//) next
            if (image ~ /^docker-image:\/\//) next

            if (image !~ /@sha256:[a-f0-9]{64}$/ && image ~ /[\/:]/) {
                print image
            }
        }
    ' "$file" 2>/dev/null)

    if [[ -n "$mutable_images" ]]; then
        log_issue "MEDIUM" "$file" "Uses container or service images without immutable @sha256 digests"
        found=1
    fi

    [[ $found -eq 0 ]]
}

# =============================================================================
# Check 26: fromJSON() with attacker-controlled input
# =============================================================================
check_26_description="fromJSON() with attacker-controlled input"
check_26_severity="MEDIUM"

example_check_26() {
    cat <<'EOF'
# BEFORE (risky - fromJSON parses attacker-controlled JSON):
  runs-on: ${{ fromJSON(inputs.runner-label) }}
# AFTER (safe - validate inputs before parsing):
  env:
    LABEL_JSON: ${{ inputs.runner-label }}
  run: |
    # Validate JSON structure before use
    echo "$LABEL_JSON" | jq -e 'type == "array"' || exit 1
EOF
}

check_26() {
    local file="$1"
    local risky_fromjson

    # Look for fromJSON() containing potentially attacker-controlled inputs
    risky_fromjson=$(grep -nE '\$\{\{.*fromJSON\([^)]*(inputs\.|github\.event\.|matrix\.)' "$file" 2>/dev/null || true)

    if [[ -n "$risky_fromjson" ]]; then
        log_issue "MEDIUM" "$file" "fromJSON() used with potentially attacker-controlled input (inputs.*, github.event.*, or matrix.*) - validate JSON before parsing"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 27: Dynamic expression interpolation in 'uses:'
# =============================================================================
check_27_description="dynamic expression interpolation in 'uses:'"
check_27_severity="HIGH"

example_check_27() {
    cat <<'EOF'
# BEFORE (unsafe - path traversal via expression):
  uses: ./.github/workflows/${{ inputs.workflow }}.yaml
# AFTER (safe - hardcoded workflow reference with input validation):
  uses: ./.github/workflows/build.yaml
# Or use a switch/if pattern with validated, hardcoded paths
EOF
}

check_27() {
    local file="$1"
    local dynamic_uses

    # Look for ${{ }} expressions in uses: statements
    dynamic_uses=$(awk '
        /^[[:space:]]*-[[:space:]]*uses:[[:space:]]*\$\{\{/ ||
        /^[[:space:]]*uses:[[:space:]]*\$\{\{/ {
            print
            next
        }
        /^[[:space:]]*-[[:space:]]*uses:.*\$\{\{/ ||
        /^[[:space:]]*uses:.*\$\{\{/ {
            print
        }
    ' "$file" 2>/dev/null || true)

    if [[ -n "$dynamic_uses" ]]; then
        log_issue "HIGH" "$file" "Uses statement contains expression interpolation - can lead to path traversal or execution of unintended workflows"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 28: Cache key injection patterns
# =============================================================================
check_28_description="cache key injection patterns"
check_28_severity="MEDIUM"

example_check_28() {
    cat <<'EOF'
# BEFORE (risky - attacker controls cache key):
  - uses: actions/cache@v4
    with:
      key: ${{ inputs.cache-key }}-deps
# AFTER (safe - use commit SHA or validated identifier):
  - uses: actions/cache@v4
    with:
      key: ${{ github.sha }}-${{ hashFiles('**/lockfiles') }}-deps
EOF
}

check_28() {
    local file="$1"
    local risky_cache

    # Look for actions/cache or cache-related actions with inputs/github.event in keys
    risky_cache=$(awk '
        /^[[:space:]]*-[[:space:]]*uses:.*actions\/cache/ ||
        /^[[:space:]]*uses:.*actions\/cache/ {
            in_cache = 1
            next
        }
        in_cache && /^[[:space:]]*-( |$)/ { in_cache = 0 }
        in_cache && /key:/ && /\$\{\{.*(inputs\.|github\.event\.)/ {
            print
            in_cache = 0
        }
    ' "$file" 2>/dev/null || true)

    if [[ -n "$risky_cache" ]]; then
        log_issue "MEDIUM" "$file" "Cache key contains potentially attacker-controlled input - can enable cache poisoning attacks"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 29: Artifact path expressions
# =============================================================================
check_29_description="artifact path expressions"
check_29_severity="MEDIUM"

example_check_29() {
    cat <<'EOF'
# BEFORE (risky - path traversal via expression):
  - uses: actions/upload-artifact@v4
    with:
      path: ${{ inputs.artifact-path }}
# AFTER (safe - use validated, hardcoded paths with optional filters):
  - uses: actions/upload-artifact@v4
    with:
      path: ./build/
EOF
}

check_29() {
    local file="$1"
    local risky_artifact

    # Look for upload-artifact or download-artifact with expressions in path
    risky_artifact=$(awk '
        /^[[:space:]]*-[[:space:]]*uses:.*actions\/(upload|download)-artifact/ ||
        /^[[:space:]]*uses:.*actions\/(upload|download)-artifact/ {
            in_artifact = 1
            next
        }
        in_artifact && /^[[:space:]]*-( |$)/ { in_artifact = 0 }
        in_artifact && /path:/ && /\$\{\{.*(inputs\.|github\.event\.)/ {
            print
            in_artifact = 0
        }
    ' "$file" 2>/dev/null || true)

    if [[ -n "$risky_artifact" ]]; then
        log_issue "MEDIUM" "$file" "Artifact path contains expression interpolation - can lead to path traversal or unintended file access"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 30: URL construction in HTTP clients
# =============================================================================
check_30_description="URL construction in HTTP clients"
check_30_severity="MEDIUM"

example_check_30() {
    cat <<'EOF'
# BEFORE (risky - SSRF via expression in URL):
  run: curl -sSL "${{ inputs.download-url }}" -o file.tar.gz
# AFTER (safe - validate URL before use):
  env:
    DOWNLOAD_URL: ${{ inputs.download-url }}
  run: |
    # Validate URL pattern before download
    if [[ ! "$DOWNLOAD_URL" =~ ^https://trusted\.example\.com/ ]]; then
      echo "Invalid URL"
      exit 1
    fi
    curl -sSL "$DOWNLOAD_URL" -o file.tar.gz
EOF
}

check_30() {
    local file="$1"
    local risky_curl

    # Look for curl/wget with expression interpolation in URL construction
    # Check both single-line and multi-line run blocks
    # Note: run: can appear as "run:" or "- run:" (in steps list)
    risky_curl=$(awk '
        BEGIN { in_run_block = 0; run_indent = 0 }
        /^[[:space:]]*-?[[:space:]]*run:[[:space:]]*\|/ {
            match($0, /^[[:space:]]*/); run_indent = RLENGTH; in_run_block = 1; next
        }
        /^[[:space:]]*-?[[:space:]]*run:/ {
            # Single-line run statement
            if (/curl.*\$\{\{/ || /wget.*\$\{\{/) {
                print
            }
            next
        }
        in_run_block {
            if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_run_block = 0; next }
            match($0, /^[[:space:]]*/); curr_indent = RLENGTH
            if (curr_indent <= run_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_run_block = 0; next }
            if (/curl.*\$\{\{/ || /wget.*\$\{\{/) {
                print
            }
        }
    ' "$file" 2>/dev/null || true)

    if [[ -n "$risky_curl" ]]; then
        log_issue "MEDIUM" "$file" "curl/wget command contains expression interpolation in URL - can lead to SSRF or downloading from attacker-controlled servers"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 31: Container volume mount expressions
# =============================================================================
check_31_description="container volume mount expressions"
check_31_severity="MEDIUM"

example_check_31() {
    cat <<'EOF'
# BEFORE (risky - container escape via expression):
  container:
    image: ubuntu:latest
    volumes:
      - ${{ inputs.mount-path }}:/workspace
# AFTER (safe - hardcoded volume mounts):
  container:
    image: ubuntu:latest
    volumes:
      - /workspace:/workspace
EOF
}

check_31() {
    local file="$1"
    local risky_volume

    # Look for volumes: sections with expression interpolation
    risky_volume=$(awk '
        /^[[:space:]]*volumes:/ {
            in_volumes = 1
            next
        }
        in_volumes && /^[[:space:]]*-[[:space:]]*\$\{\{/ {
            print
        }
        in_volumes && /^[[:space:]]*[a-z_-]+:/ && !/volumes:/ { in_volumes = 0 }
    ' "$file" 2>/dev/null || true)

    if [[ -n "$risky_volume" ]]; then
        log_issue "MEDIUM" "$file" "Container volume mount contains expression interpolation - can lead to container escape or unauthorized host filesystem access"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 32: Expression interpolation written to GITHUB_ENV/PATH/OUTPUT
# =============================================================================
check_32_description="expression interpolation in GITHUB_ENV/PATH/OUTPUT writes"
check_32_severity="HIGH"

example_check_32() {
    cat <<'EOF'
# BEFORE (unsafe - expression injected into env file enables LD_PRELOAD/BASH_ENV injection):
  run: echo "BRANCH=${{ github.head_ref }}" >> "$GITHUB_ENV"
# AFTER (safe - pass via env var, then write shell variable):
  env:
    BRANCH: ${{ github.head_ref }}
  run: echo "BRANCH=$BRANCH" >> "$GITHUB_ENV"
EOF
}

check_32() {
    local file="$1"
    if ! grep -qE 'GITHUB_(ENV|PATH|OUTPUT)' "$file" 2>/dev/null; then
        return 0
    fi
    if ! grep -qE '\$\{\{' "$file" 2>/dev/null; then
        return 0
    fi

    local unsafe_usage
    unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
        /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{/ && /GITHUB_(ENV|PATH|OUTPUT)/ { print; next }
        in_run_block && /\$\{\{/ && /GITHUB_(ENV|PATH|OUTPUT)/ { print }
    ' "$file" 2>/dev/null)
    if [[ -n "$unsafe_usage" ]]; then
        log_issue "HIGH" "$file" "Contains \${{ }} expression on line that writes to GITHUB_ENV/PATH/OUTPUT - enables environment variable injection (LD_PRELOAD, BASH_ENV)"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 33: Missing explicit permissions key (aggregate check)
# =============================================================================
check_33_description="missing explicit permissions key"
check_33_severity="LOW"
check_33_aggregate=true

example_check_33() {
    cat <<'EOF'
# BEFORE (inherits default, potentially broad permissions):
  on: push
  jobs:
    build: ...
# AFTER (explicit least-privilege):
  on: push
  permissions:
    contents: read
  jobs:
    build: ...
EOF
}

check_33() {
    local missing_count=0
    for file in "$@"; do
        # Only check workflow files (must have top-level 'on:' trigger), skip actions
        if ! grep -qE '^on:|^"on":|^on$' "$file" 2>/dev/null; then
            continue
        fi
        # Skip reusable workflow implementations (they inherit caller permissions)
        if grep -qE '^\s*workflow_call:' "$file" 2>/dev/null; then
            continue
        fi
        # Check for top-level permissions key
        if ! grep -qE '^permissions:' "$file" 2>/dev/null; then
            ((missing_count++)) || true
        fi
    done
    if [[ "$missing_count" -gt 0 ]]; then
        log_issue "LOW" "Multiple files" "Found $missing_count workflows without an explicit top-level 'permissions:' key - add permissions with least-privilege scopes"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 34: Concurrency group with attacker-controlled data
# =============================================================================
check_34_description="concurrency group with attacker-controlled data"
check_34_severity="MEDIUM"

example_check_34() {
    cat <<'EOF'
# BEFORE (risky - branch name controls cancellation group, enabling DoS):
  concurrency:
    group: deploy-${{ github.head_ref }}
# AFTER (safe - use PR number or SHA):
  concurrency:
    group: deploy-${{ github.event.pull_request.number || github.sha }}
EOF
}

check_34() {
    local file="$1"
    if ! grep -qE 'concurrency:' "$file" 2>/dev/null; then
        return 0
    fi

    local risky_concurrency
    risky_concurrency=$(awk '
        /^concurrency:/ || /^[[:space:]]+concurrency:/ { in_concurrency = 1 }
        in_concurrency && /^[a-z]/ && !/^concurrency:/ { in_concurrency = 0 }
        in_concurrency && /\$\{\{/ && /(github\.head_ref|github\.base_ref|github\.event\.[^}]*(title|body|head_branch)|inputs\.)/ {
            print
        }
        /^concurrency:[[:space:]]*[^{[:space:]]/ { next }
        /^concurrency:.*\$\{\{/ && /(github\.head_ref|github\.base_ref|github\.event\.[^}]*(title|body|head_branch)|inputs\.)/ {
            print
        }
    ' "$file" 2>/dev/null || true)

    if [[ -n "$risky_concurrency" ]]; then
        log_issue "MEDIUM" "$file" "Concurrency group contains attacker-controlled expression - can enable DoS by cancelling legitimate runs"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 35: issue_comment trigger without authorization gate
# =============================================================================
check_35_description="issue_comment trigger without authorization gate"
check_35_severity="MEDIUM"

example_check_35() {
    cat <<'EOF'
# BEFORE (any commenter triggers workflow):
  on: issue_comment
  jobs:
    deploy:
      runs-on: ubuntu-latest
      steps: ...
# AFTER (only repo members can trigger):
  on: issue_comment
  jobs:
    deploy:
      if: >-
        github.event.comment.author_association == 'MEMBER' ||
        github.event.comment.author_association == 'OWNER'
      runs-on: ubuntu-latest
      steps: ...
EOF
}

check_35() {
    local file="$1"
    if ! grep -qE 'issue_comment' "$file" 2>/dev/null; then
        return 0
    fi

    # Check for common authorization patterns
    if grep -qE 'author_association' "$file" 2>/dev/null; then
        return 0
    fi
    if grep -qE '/orgs/.*members|/teams/' "$file" 2>/dev/null; then
        return 0
    fi

    log_issue "MEDIUM" "$file" "issue_comment trigger has no author_association or org membership check - any commenter can trigger this workflow"
    return 1
}

# =============================================================================
# Check 36: working-directory with attacker-controlled expressions
# =============================================================================
check_36_description="working-directory with attacker-controlled expressions"
check_36_severity="MEDIUM"

example_check_36() {
    cat <<'EOF'
# BEFORE (risky - attacker can control working directory):
  - run: make build
    working-directory: ${{ inputs.build-dir }}
# AFTER (safe - validate path before use):
  env:
    BUILD_DIR: ${{ inputs.build-dir }}
  run: |
    [[ "$BUILD_DIR" =~ ^[a-zA-Z0-9._/-]+$ ]] || exit 1
    cd "$BUILD_DIR" && make build
EOF
}

check_36() {
    local file="$1"
    local risky_workdir

    risky_workdir=$(grep -nE 'working-directory:.*\$\{\{.*(inputs\.|github\.event\.|github\.head_ref|github\.base_ref)' "$file" 2>/dev/null || true)

    if [[ -n "$risky_workdir" ]]; then
        log_issue "MEDIUM" "$file" "working-directory contains attacker-controlled expression interpolation - can influence which directory code runs in"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 37: actions/github-script with non-event attacker-controlled interpolation
# =============================================================================
check_37_description="github-script with inputs/outputs/secrets interpolation"
check_37_severity="HIGH"

example_check_37() {
    cat <<'EOF'
# BEFORE (unsafe - attacker-controlled input becomes JavaScript source):
  uses: actions/github-script@v7
  with:
    script: |
      const body = "${{ inputs.user-script }}";
      console.log(body);
# AFTER (safe - pass through env and read from process.env):
  uses: actions/github-script@v7
  env:
    USER_SCRIPT: ${{ inputs.user-script }}
  with:
    script: |
      const body = process.env.USER_SCRIPT;
      console.log(body);
EOF
}

check_37() {
    local file="$1"
    local attacker_controlled_js='(inputs\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.|matrix\.|secrets\.|github\.head_ref|github\.base_ref)'
    if grep -qE 'actions/github-script' "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk -v pattern="$attacker_controlled_js" '
            BEGIN { in_action = 0; in_script = 0; script_indent = 0 }
            /uses:.*actions\/github-script/ { in_action = 1; next }
            in_action && /^[[:space:]]*script:[[:space:]]*\|/ {
                match($0, /^[[:space:]]*/); script_indent = RLENGTH; in_script = 1; next
            }
            in_action && /^[[:space:]]*script:/ && $0 ~ pattern { print; next }
            in_script {
                if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_script = 0; in_action = 0; next }
                match($0, /^[[:space:]]*/); curr_indent = RLENGTH
                if (curr_indent <= script_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_script = 0; in_action = 0; next }
            }
            in_script && $0 ~ pattern { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "actions/github-script contains attacker-controlled \${{ inputs.* }}, outputs, matrix, secrets, or ref data in script body - JavaScript injection risk"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 38: actions/github-script dynamic code execution primitives
# =============================================================================
check_38_description="github-script dynamic code execution primitives"
check_38_severity="HIGH"

example_check_38() {
    cat <<'EOF'
# BEFORE (unsafe - runtime input is executed as JavaScript):
  uses: actions/github-script@v7
  with:
    script: |
      const dynamic = core.getInput('payload');
      return new Function(dynamic)();
# AFTER (safe - treat input as data, not code):
  uses: actions/github-script@v7
  with:
    script: |
      const payload = core.getInput('payload');
      console.log(payload);
EOF
}

check_38() {
    local file="$1"
    if grep -qE 'actions/github-script' "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk '
            BEGIN { in_action = 0; in_script = 0; script_indent = 0 }
            /uses:.*actions\/github-script/ { in_action = 1; next }
            in_action && /^[[:space:]]*script:[[:space:]]*\|/ {
                match($0, /^[[:space:]]*/); script_indent = RLENGTH; in_script = 1; next
            }
            function is_dynamic_js() {
                return $0 ~ /eval[[:space:]]*\(/ ||
                       $0 ~ /new[[:space:]]+Function[[:space:]]*\(/ ||
                       $0 ~ /vm\.(runInNewContext|runInThisContext|Script)[[:space:]]*\(/
            }
            in_action && /^[[:space:]]*script:/ && is_dynamic_js() { print; next }
            in_script {
                if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_script = 0; in_action = 0; next }
                match($0, /^[[:space:]]*/); curr_indent = RLENGTH
                if (curr_indent <= script_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_script = 0; in_action = 0; next }
            }
            in_script && is_dynamic_js() { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "actions/github-script uses eval/new Function/vm dynamic code execution in script body - treat inputs and env data as data, not code"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 39: Inline interpreter eval/exec patterns in run blocks
# =============================================================================
check_39_description="inline interpreter eval/exec patterns in run blocks"
check_39_severity="HIGH"

example_check_39() {
    cat <<'EOF'
# BEFORE (unsafe - env/input is re-executed by Node.js):
  env:
    USER_SCRIPT: ${{ inputs.payload }}
  run: node -e "eval(process.env.USER_SCRIPT)"
# AFTER (safe - consume as data without dynamic evaluation):
  env:
    USER_SCRIPT: ${{ inputs.payload }}
  run: node -e "console.log(process.env.USER_SCRIPT)"
EOF
}

check_39() {
    local file="$1"
    local unsafe_usage
    unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
        function check_line() {
            if ($0 ~ /node([^[:alnum:]_]|$)/ && $0 ~ /-e/ && $0 ~ /(eval[[:space:]]*\(|new[[:space:]]+Function[[:space:]]*\(|vm\.(runInNewContext|runInThisContext|Script)[[:space:]]*\()/) { print; return }
            if ($0 ~ /python3?([^[:alnum:]_]|$)/ && $0 ~ /-c/ && $0 ~ /(^|[^[:alnum:]_])(eval|exec)[[:space:]]*\(/) { print; return }
            if ($0 ~ /perl([^[:alnum:]_]|$)/ && $0 ~ /-e/ && $0 ~ /(^|[^[:alnum:]_])eval([^[:alnum:]_]|$)/) { print; return }
            if ($0 ~ /ruby([^[:alnum:]_]|$)/ && $0 ~ /-e/ && $0 ~ /(^|[^[:alnum:]_])eval[[:space:]]*\(/) { print; return }
            if ($0 ~ /php([^[:alnum:]_]|$)/ && $0 ~ /-r/ && $0 ~ /(^|[^[:alnum:]_])eval[[:space:]]*\(/) { print; return }
            if ($0 ~ /(pwsh|powershell)([^[:alnum:]_]|$)/ && $0 ~ /(-Command|-c)/ && $0 ~ /Invoke-Expression/) { print; return }
        }
        /^[[:space:]]*(-[[:space:]]+)?run:/ { check_line(); next }
        in_run_block { check_line() }
    ' "$file" 2>/dev/null)
    if [[ -n "$unsafe_usage" ]]; then
        log_issue "HIGH" "$file" "Inline interpreter command uses eval/exec-style dynamic code execution in run: block - avoid executing constructed code"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 40: actions/github-script command execution APIs
# =============================================================================
check_40_description="github-script command execution APIs"
check_40_severity="HIGH"

example_check_40() {
    cat <<'EOF'
# BEFORE (unsafe - runtime input is executed as a shell command):
  uses: actions/github-script@v7
  with:
    script: |
      const { execSync } = require('child_process');
      const cmd = core.getInput('payload');
      execSync(cmd, { shell: true });
# AFTER (safe - avoid runtime shell execution from inputs):
  uses: actions/github-script@v7
  with:
    script: |
      const payload = core.getInput('payload');
      console.log(payload);
EOF
}

check_40() {
    local file="$1"
    if grep -qE 'actions/github-script' "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk '
            BEGIN { in_action = 0; in_script = 0; script_indent = 0; has_runtime_source = 0; has_command_exec = 0 }
            function flush_script() {
                if (in_script && has_runtime_source && has_command_exec) {
                    print "unsafe"
                }
                has_runtime_source = 0
                has_command_exec = 0
            }
            /uses:.*actions\/github-script/ {
                flush_script()
                in_action = 1
                in_script = 0
                next
            }
            in_action && /^[[:space:]]*script:[[:space:]]*\|/ {
                match($0, /^[[:space:]]*/); script_indent = RLENGTH; in_script = 1; next
            }
            in_action && /^[[:space:]]*script:/ {
                if ($0 ~ /core\.getInput[[:space:]]*\(|process\.env[[:space:]]*[\[.]/) has_runtime_source = 1
                if ($0 ~ /(execSync|exec|spawnSync|spawn|execFileSync|execFile)[[:space:]]*\(/ || $0 ~ /shell:[[:space:]]*true/) has_command_exec = 1
                if (has_runtime_source && has_command_exec) print "unsafe"
                next
            }
            in_script {
                if ($0 ~ /^[[:space:]]*-[[:space:]]/) {
                    flush_script()
                    in_script = 0
                    in_action = 0
                    next
                }
                match($0, /^[[:space:]]*/); curr_indent = RLENGTH
                if (curr_indent <= script_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) {
                    flush_script()
                    in_script = 0
                    in_action = 0
                    next
                }
                if ($0 ~ /core\.getInput[[:space:]]*\(|process\.env[[:space:]]*[\[.]/) has_runtime_source = 1
                if ($0 ~ /(execSync|exec|spawnSync|spawn|execFileSync|execFile)[[:space:]]*\(/ || $0 ~ /shell:[[:space:]]*true/) has_command_exec = 1
            }
            END { flush_script() }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "actions/github-script combines runtime input/env data with child_process command execution APIs - command injection risk"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 41: Inline interpreter command execution APIs
# =============================================================================
check_41_description="inline interpreter command execution APIs"
check_41_severity="HIGH"

example_check_41() {
    cat <<'EOF'
# BEFORE (unsafe - env/input is executed through a shell):
  env:
    USER_CMD: ${{ inputs.payload }}
  run: python -c "import os, subprocess; subprocess.run(os.environ['USER_CMD'], shell=True, check=True)"
# AFTER (safe - avoid shell execution of runtime input):
  env:
    USER_CMD: ${{ inputs.payload }}
  run: python -c "import os; print(os.environ['USER_CMD'])"
EOF
}

check_41() {
    local file="$1"
    local unsafe_usage
    unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
        /^[[:space:]]*(-[[:space:]]+)?run:/ { print; next }
        in_run_block { print }
    ' "$file" 2>/dev/null | grep -E \
        'node([^[:alnum:]_]|$).*-e.*(execSync|exec|spawnSync|spawn|execFileSync|execFile)[[:space:]]*\(|python3?([^[:alnum:]_]|$).*-c.*(subprocess\..*shell=True|os\.system\()|ruby([^[:alnum:]_]|$).*-e.*(^|[^[:alnum:]_])(system|exec|spawn|Open3\.capture2e|Open3\.capture3)[[:space:]]*\(|perl([^[:alnum:]_]|$).*-e.*(^|[^[:alnum:]_])(system|exec)([^[:alnum:]_]|$)|php([^[:alnum:]_]|$).*-r.*(shell_exec|exec|system|passthru|proc_open)[[:space:]]*\(' \
        || true)
    if [[ -n "$unsafe_usage" ]]; then
        log_issue "HIGH" "$file" "Inline interpreter command uses shell/process execution APIs in run: block - avoid executing runtime data as commands"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 42: Shell command execution from variables
# =============================================================================
check_42_description="shell command execution from variables"
check_42_severity="HIGH"

example_check_42() {
    cat <<'EOF'
# BEFORE (unsafe - variable contents are executed as shell code):
  env:
    USER_CMD: ${{ inputs.payload }}
  run: bash -c "$USER_CMD"
# AFTER (safe - do not execute runtime data as code):
  env:
    USER_CMD: ${{ inputs.payload }}
  run: printf '%s\n' "$USER_CMD"
EOF
}

check_42() {
    local file="$1"
    local unsafe_usage
    unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
        /^[[:space:]]*(-[[:space:]]+)?run:/ { print; next }
        in_run_block { print }
    ' "$file" 2>/dev/null | grep -E \
        '(^|[[:space:]])(ba)?sh[[:space:]]+-c[[:space:]]+["'\'']?\$[A-Za-z_][A-Za-z0-9_]*|(^|[[:space:]])(zsh|dash|ksh)[[:space:]]+-c[[:space:]]+["'\'']?\$[A-Za-z_][A-Za-z0-9_]*|(^|[[:space:]])(pwsh|powershell)[[:space:]]+(-Command|-c)[[:space:]]+["'\'']?\$env:[A-Za-z_][A-Za-z0-9_]*' \
        || true)
    if [[ -n "$unsafe_usage" ]]; then
        log_issue "HIGH" "$file" "Shell command is executed from a variable via *sh -c or PowerShell -Command - this treats runtime data as code"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 43: Remote script execution variants beyond curl | bash
# =============================================================================
check_43_description="remote script execution variants beyond pipes"
check_43_severity="HIGH"

example_check_43() {
    cat <<'EOF'
# BEFORE (unsafe - remote script executed via process substitution / IEX):
  run: bash <(curl -sSL https://example.com/install.sh)
  # or
  run: pwsh -Command "iex (irm https://example.com/install.ps1)"
# AFTER (safer - download, verify, and inspect before executing):
  run: |
    curl -sSL -o install.sh https://example.com/install.sh
    sha256sum -c <<< "expected_hash  install.sh"
    bash install.sh
EOF
}

check_43() {
    local file="$1"
    local unsafe_usage
    unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
        /^[[:space:]]*(-[[:space:]]+)?run:/ { print; next }
        in_run_block { print }
    ' "$file" 2>/dev/null | grep -E \
        '(bash|sh|zsh|ksh)[[:space:]]+<\([[:space:]]*(curl|wget)[^)]*\)|(^|[[:space:]])(\.|source)[[:space:]]+<\([[:space:]]*(curl|wget)[^)]*\)|(pwsh|powershell).*((-Command|-c).*)?(iex|Invoke-Expression)[[:space:]]*\((irm|iwr|Invoke-RestMethod|Invoke-WebRequest)[[:space:]]+' \
        || true)
    if [[ -n "$unsafe_usage" ]]; then
        log_issue "HIGH" "$file" "Remote content is executed directly via process substitution or PowerShell IEX/IRM - download and verify first"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 44: Attacker-controlled env vars written to GITHUB_ENV/PATH/OUTPUT
# =============================================================================
check_44_description="attacker-controlled env vars written to GITHUB_ENV/PATH/OUTPUT"
check_44_severity="HIGH"

example_check_44() {
    cat <<'EOF'
# BEFORE (unsafe - attacker-controlled value can inject extra env/path/output entries):
  env:
    EXTRA_PATH: ${{ inputs.extra-path }}
  run: echo "$EXTRA_PATH" >> "$GITHUB_PATH"
# AFTER (safer - validate before writing, or use a trusted constant):
  env:
    EXTRA_PATH: ${{ inputs.extra-path }}
  run: |
    [[ "$EXTRA_PATH" == /opt/tools/* ]] || exit 1
    printf '%s\n' "$EXTRA_PATH" >> "$GITHUB_PATH"
EOF
}

check_44() {
    local file="$1"
    local unsafe_usage
    unsafe_usage=$(awk '
        BEGIN {
            in_step = 0
            step_indent = -1
            in_env = 0
            env_indent = -1
            in_run_block = 0
            run_indent = -1
        }
        function clear_env_vars(    k) {
            for (k in dangerous_env) {
                delete dangerous_env[k]
            }
        }
        function reset_step() {
            in_step = 0
            step_indent = -1
            in_env = 0
            env_indent = -1
            in_run_block = 0
            run_indent = -1
            clear_env_vars()
        }
        function line_uses_dangerous_var(line,    k, pat1, pat2) {
            for (k in dangerous_env) {
                pat1 = "\\$" k "([^A-Za-z0-9_]|$)"
                pat2 = "\\$\\{" k "\\}"
                if (line ~ pat1 || line ~ pat2) {
                    return 1
                }
            }
            return 0
        }
        {
            match($0, /^[[:space:]]*/)
            curr_indent = RLENGTH
        }
        /^[[:space:]]*-[[:space:]]/ {
            reset_step()
            in_step = 1
            step_indent = curr_indent
        }
        in_step && curr_indent <= step_indent && $0 !~ /^[[:space:]]*-[[:space:]]/ && $0 ~ /^[[:space:]]*[A-Za-z_-]+:/ {
            reset_step()
        }
        in_step && /^[[:space:]]*(-[[:space:]]+)?env:[[:space:]]*$/ {
            in_env = 1
            env_indent = curr_indent
            next
        }
        in_env {
            if (curr_indent <= env_indent && $0 ~ /^[[:space:]]*[A-Za-z_-]+:/) {
                in_env = 0
            } else if ($0 ~ /^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*\$\{\{.*(inputs\.|github\.event\.|github\.head_ref|github\.base_ref|matrix\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.|secrets\.)/) {
                line = $0
                sub(/^[[:space:]]*/, "", line)
                split(line, parts, ":")
                dangerous_env[parts[1]] = 1
            }
        }
        in_step && /^[[:space:]]*run:[[:space:]]*\|/ {
            in_run_block = 1
            run_indent = curr_indent
            next
        }
        in_step && /^[[:space:]]*run:/ {
            if ($0 ~ /GITHUB_(ENV|PATH|OUTPUT)/ && line_uses_dangerous_var($0)) {
                print
            }
            next
        }
        in_run_block {
            if ($0 ~ /^[[:space:]]*-[[:space:]]/) {
                in_run_block = 0
                next
            }
            if (curr_indent <= run_indent && $0 ~ /^[[:space:]]*[A-Za-z_-]+:/) {
                in_run_block = 0
                next
            }
            if ($0 ~ /GITHUB_(ENV|PATH|OUTPUT)/ && line_uses_dangerous_var($0)) {
                print
            }
        }
    ' "$file" 2>/dev/null)
    if [[ -n "$unsafe_usage" ]]; then
        log_issue "HIGH" "$file" "Attacker-controlled env var is written to GITHUB_ENV/PATH/OUTPUT - newline injection can create unintended entries"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 45: Expression injection in 'if:' conditions
# =============================================================================
check_45_description="expression injection in 'if:' conditions"
check_45_severity="HIGH"

example_check_45() {
    cat <<'EOF'
# BEFORE (unsafe - attacker-controlled data in conditional logic):
  if: contains('${{ github.event.pull_request.title }}', 'build')
  run: echo "This step runs conditionally"
# AFTER (safe - avoid direct interpolation in if conditions):
  env:
    PR_TITLE: ${{ github.event.pull_request.title }}
  if: contains(env.PR_TITLE, 'build')
  run: echo "This step runs conditionally"
EOF
}

check_45() {
    local file="$1"
    local found=0

    # Check for if: conditions with attacker-controlled expressions
    # Look for if: containing ${{ with github.event or inputs contexts
    local dangerous_if
    dangerous_if=$(grep -nE '^[[:space:]]*if:.*\$\{\{.*(github\.event\.(pull_request|issue|comment|discussion|review|release)|inputs\.|client_payload|github\.head_ref|github\.base_ref)' "$file" 2>/dev/null || true)

    if [[ -n "$dangerous_if" ]]; then
        log_issue "HIGH" "$file" "'if:' condition contains attacker-controlled expression interpolation - can manipulate workflow logic"
        found=1
    fi

    [[ $found -eq 0 ]]
}

# =============================================================================
# Check 46: Bracket notation bypass in expressions
# =============================================================================
check_46_description="bracket notation bypass in expressions"
check_46_severity="HIGH"

example_check_46() {
    cat <<'EOF'
# BEFORE (unsafe - bracket notation bypasses dot-notation detection):
  run: echo "${{ inputs['user-command'] }}"
  run: echo "${{ github.event['pull_request']['title'] }}"
# AFTER (safe - use env var for all interpolation):
  env:
    USER_CMD: ${{ inputs['user-command'] }}
    PR_TITLE: ${{ github.event['pull_request']['title'] }}
  run: |
    echo "$USER_CMD"
    echo "$PR_TITLE"
EOF
}

check_46() {
    local file="$1"
    # Detect bracket notation with ${{} in run blocks
    # First extract run block content, then check for bracket notation

    local unsafe_usage
    unsafe_usage=$(awk '
        BEGIN { in_run_block = 0; run_indent = 0; content = "" }
        /^[[:space:]]*(-[[:space:]]+)?run:[[:space:]]*\|/ {
            match($0, /^[[:space:]]*/); run_indent = RLENGTH; in_run_block = 1; next
        }
        /^[[:space:]]*(-[[:space:]]+)?run:/ {
            if (/\$\{\{/) {
                print
            }
            next
        }
        in_run_block {
            if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_run_block = 0; next }
            match($0, /^[[:space:]]*/); curr_indent = RLENGTH
            if (curr_indent <= run_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_run_block = 0; next }
            print
        }
    ' "$file" 2>/dev/null | grep -E '\$\{\{[^}]*\[' || true)

    if [[ -n "$unsafe_usage" ]]; then
        log_issue "HIGH" "$file" "Uses bracket notation for expression in run block - can bypass dot-notation detection"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 47: Additional attacker-controlled event contexts
# =============================================================================
check_47_description="additional attacker-controlled event contexts"
check_47_severity="HIGH"

example_check_47() {
    cat <<'EOF'
# BEFORE (unsafe - additional event fields not covered by check 16):
  run: echo "${{ github.event.release.body }}"
  run: echo "${{ github.event.inputs.command }}"
  run: echo "${{ github.event.client_payload.data }}"
# AFTER (safe - use env var for all event data):
  env:
    RELEASE_BODY: ${{ github.event.release.body }}
    INPUT_CMD: ${{ github.event.inputs.command }}
    PAYLOAD: ${{ github.event.client_payload.data }}
  run: |
    echo "$RELEASE_BODY"
    echo "$INPUT_CMD"
    echo "$PAYLOAD"
EOF
}

check_47() {
    local file="$1"
    # Additional event fields not covered by check 16
    local additional_fields='github\.event\.(release\.(name|body|tag_name|target_commitish|draft|prerelease|author\.(login|name))|inputs\.|client_payload\.|schedule|ref_type|pusher\.(name|email)|forced|created|deleted|pages\[|forkee\.|action|number)'

    if grep -qE "$additional_fields" "$file" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /github\.event\.(release\.(name|body|tag_name|target_commitish|draft|prerelease|author\.(login|name))|inputs\.|client_payload\.|schedule|ref_type|pusher\.(name|email)|forced|created|deleted|pages\[|forkee\.|action|number)/ { print; next }
            in_run_block && /github\.event\.(release\.(name|body|tag_name|target_commitish|draft|prerelease|author\.(login|name))|inputs\.|client_payload\.|schedule|ref_type|pusher\.(name|email)|forced|created|deleted|pages\[|forkee\.|action|number)/ { print }
        ' "$file" 2>/dev/null)
        if [[ -n "$unsafe_usage" ]]; then
            log_issue "HIGH" "$file" "Contains additional attacker-controlled github.event field directly in run: block - use env var instead"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 48: Export statements with direct interpolation
# =============================================================================
check_48_description="export statements with direct interpolation"
check_48_severity="HIGH"

example_check_48() {
    cat <<'EOF'
# BEFORE (unsafe - export creates env var from attacker input that propagates):
  run: |
    export BUILD_DIR="${{ inputs.build-dir }}"
    cd "$BUILD_DIR" && make
# AFTER (safe - validate inputs before export, or use env: block):
  env:
    BUILD_DIR: ${{ inputs.build-dir }}
  run: |
    [[ "$BUILD_DIR" =~ ^[a-zA-Z0-9_/-]+$ ]] || exit 1
    cd "$BUILD_DIR" && make
EOF
}

check_48() {
    local file="$1"
    # Pattern to detect export statements with interpolation
    local export_pattern='export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*=.*\$\{\{'

    local unsafe_usage
    unsafe_usage=$(awk "$AWK_RUN_BLOCK_DETECT"'
        /^[[:space:]]*(-[[:space:]]+)?run:/ && /export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*=.*\$\{\{/ { print; next }
        in_run_block && /export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*=.*\$\{\{/ { print }
    ' "$file" 2>/dev/null)

    if [[ -n "$unsafe_usage" ]]; then
        log_issue "HIGH" "$file" "Export statement contains direct \${{ }} interpolation - exported variable propagates to child processes"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 49: Dynamic 'shell:' property expressions
# =============================================================================
check_49_description="dynamic 'shell:' property expressions"
check_49_severity="MEDIUM"

example_check_49() {
    cat <<'EOF'
# BEFORE (unsafe - shell property controlled by attacker input):
  run: echo "hello"
  shell: ${{ inputs.custom_shell }}
# AFTER (safe - hardcode shell or validate against allowlist):
  run: echo "hello"
  shell: bash
EOF
}

check_49() {
    local file="$1"
    # Pattern to detect shell: with expression interpolation
    local shell_pattern='^[[:space:]]*shell:.*\$\{\{'

    local unsafe_usage
    unsafe_usage=$(grep -nE "$shell_pattern" "$file" 2>/dev/null || true)

    if [[ -n "$unsafe_usage" ]]; then
        log_issue "MEDIUM" "$file" "'shell:' property contains expression interpolation - can execute arbitrary shell commands"
        return 1
    fi
    return 0
}

# =============================================================================
# Check Runner
# =============================================================================

run_check() {
    local check_num="$1"
    local desc_var="check_${check_num}_description"
    local agg_var="check_${check_num}_aggregate"
    local example_fn="example_check_${check_num}"
    local check_fn="check_${check_num}"
    local found=0

    # Set global for log_issue to use
    CURRENT_CHECK="$check_num"

    # Skip aggregate checks if requested
    if [[ "$SKIP_AGGREGATE" == "true" && "${!agg_var:-}" == "true" ]]; then
        return
    fi

    # Only print status in normal mode
    if [[ "$MACHINE_OUTPUT" != "true" ]]; then
        echo "Checking for ${!desc_var}..."
    fi

    if [[ "${!agg_var:-}" == "true" ]]; then
        if ! "$check_fn" "${FILES[@]}"; then
            found=1
        fi
    else
        for file in "${FILES[@]}"; do
            if ! "$check_fn" "$file"; then
                found=1
            fi
        done
    fi

    # Only print examples in normal mode
    if [[ $found -eq 1 && "$MACHINE_OUTPUT" != "true" ]]; then
        log_fix_example "check_${check_num}" "$($example_fn)"
    fi
}

# =============================================================================
# Format Results (for parallel post-processing)
# =============================================================================

format_results() {
    local results_file="$1"
    local -A checks_with_issues=()
    local total_issues=0

    if [[ ! -f "$results_file" ]]; then
        echo "Error: Results file not found: $results_file" >&2
        exit 1
    fi

    # First pass: print all issues, track which checks had failures
    while IFS=$'\t' read -r check severity file message; do
        [[ -z "$check" ]] && continue

        local color="$YELLOW"
        if [[ "$severity" == "CRITICAL" ]] || [[ "$severity" == "HIGH" ]]; then
            color="$RED"
        fi
        echo -e "${color}[$severity]${NC} $file: $message"

        checks_with_issues[$check]=1
        ((total_issues++)) || true
    done < "$results_file"

    # Second pass: print one example per unique failed check
    for check_num in "${!checks_with_issues[@]}"; do
        local example_fn="example_check_${check_num}"
        if declare -f "$example_fn" > /dev/null 2>&1; then
            log_fix_example "check_${check_num}" "$($example_fn)"
        fi
    done

    echo ""
    echo "=========================================="
    echo "Scan Complete"
    echo "=========================================="
    echo ""

    if [[ $total_issues -eq 0 ]]; then
        echo -e "${GREEN}No security issues found!${NC}"
        echo ""
    else
        echo -e "${YELLOW}Found $total_issues potential security issues${NC}"
        echo ""
        echo "Remediation guidance:"
        echo "  - CRITICAL: Fix immediately - allows arbitrary code execution or secret exfiltration"
        echo "  - HIGH: Fix immediately - direct shell injection risk"
        echo "  - MEDIUM: Fix soon - potential injection or supply chain risk"
        echo "  - LOW: Defense in depth - address in regular maintenance"
        echo ""
    fi

    # Set exit code based on issues found
    if [[ "$STRICT_MODE" == "true" && $total_issues -gt 0 ]]; then
        exit 1
    fi
    exit 0
}

# =============================================================================
# Main Execution
# =============================================================================

# Handle --format-results mode: read and display results from parallel run
if [[ -n "$FORMAT_RESULTS_FILE" ]]; then
    echo "=========================================="
    echo "GitHub Actions Security Lint Check"
    echo "=========================================="
    echo ""
    format_results "$FORMAT_RESULTS_FILE"
    # format_results exits, so this line is never reached
fi

# Print banner only in normal mode
if [[ "$MACHINE_OUTPUT" != "true" ]]; then
    echo "=========================================="
    echo "GitHub Actions Security Lint Check"
    echo "=========================================="
    echo ""
fi

for i in "${CHECKS_TO_RUN[@]}"; do
    run_check "$i"
done

# Print summary only in normal mode
if [[ "$MACHINE_OUTPUT" != "true" ]]; then
    echo ""
    echo "=========================================="
    echo "Scan Complete"
    echo "=========================================="
    echo ""

    if [[ $ISSUES_FOUND -eq 0 ]]; then
        echo -e "${GREEN}No security issues found!${NC}"
        echo ""
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
        echo ""

        if [[ "$STRICT_MODE" == "true" ]]; then
            exit 1
        fi
        exit 0
    fi
fi

# In machine output mode, exit with appropriate code
if [[ "$STRICT_MODE" == "true" && $ISSUES_FOUND -gt 0 ]]; then
    exit 1
fi
exit 0
