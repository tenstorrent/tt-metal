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

echo "=========================================="
echo "GitHub Actions Security Lint Check"
echo "=========================================="
echo ""

# Check 1: Direct input interpolation in run blocks (high risk patterns)
echo "Checking for dangerous direct interpolation patterns..."
dangerous_patterns='github\.event\.(comment\.body|issue\.body|pull_request\.body|pull_request\.title)'
while IFS= read -r -d '' file; do
    if grep -qE "$dangerous_patterns" "$file" 2>/dev/null; then
        log_issue "HIGH" "$file" "Contains dangerous event data interpolation (comment/issue/PR body/title)"
    fi
done < <(find "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" -name "*.yml" -o -name "*.yaml" 2>/dev/null | tr '\n' '\0')

# Check 2: eval usage
echo "Checking for eval usage..."
while IFS= read -r file; do
    [[ -n "$file" ]] && log_issue "HIGH" "$file" "Contains 'eval' statement - consider safer alternatives"
done < <(grep -rl 'eval\s' "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null || true)

# Check 3: bash -c with direct ${{ }} interpolation
echo "Checking for bash -c with direct interpolation..."
while IFS= read -r file; do
    [[ -n "$file" ]] && log_issue "MEDIUM" "$file" "Contains 'bash -c' with direct \${{ }} - ensure input is via env var"
done < <(grep -rl 'bash.*-c.*\${{' "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null || true)

# Check 4: pull_request_target with potentially unsafe patterns
echo "Checking for pull_request_target with checkout..."
while IFS= read -r file; do
    if [[ -n "$file" ]] && grep -q 'actions/checkout' "$file" 2>/dev/null; then
        log_issue "HIGH" "$file" "Uses pull_request_target with checkout - verify PR code is not executed unsafely"
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
fi

# Check 6: Count secrets: inherit usage
echo "Checking for secrets: inherit usage..."
secrets_inherit_count=$(grep -rl 'secrets:\s*inherit' "$GITHUB_DIR/workflows" 2>/dev/null | wc -l | tr -d ' ' || echo 0)
if [[ "$secrets_inherit_count" -gt 50 ]]; then
    log_issue "LOW" "Multiple files" "Found $secrets_inherit_count workflows using 'secrets: inherit' - consider explicit secret passing"
fi

# Check 7: GITHUB_TOKEN or ACTIONS_RUNTIME_TOKEN exposure in logs
echo "Checking for potential token exposure in logs..."
while IFS= read -r file; do
    [[ -n "$file" ]] && log_issue "HIGH" "$file" "Potential token exposure in echo/print statement"
done < <(grep -rlE 'echo.*\$\{?\{?(GITHUB_TOKEN|ACTIONS_RUNTIME_TOKEN|ACTIONS_ID_TOKEN_REQUEST_TOKEN)' \
    "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null || true)

# Check 8: Overly broad permissions
echo "Checking for overly broad permissions..."
while IFS= read -r file; do
    [[ -n "$file" ]] && log_issue "MEDIUM" "$file" "Uses 'write-all' permissions - apply principle of least privilege"
done < <(grep -rl 'permissions:.*write-all\|write-all' "$GITHUB_DIR/workflows" 2>/dev/null || true)

# Check 9: Ref-based injection vectors (head.ref, base.ref in ${{ }})
echo "Checking for ref-based injection patterns..."
while IFS= read -r file; do
    [[ -n "$file" ]] && log_issue "MEDIUM" "$file" "Contains head.ref/base.ref interpolation - ensure value is passed via env var"
done < <(grep -rlE '\$\{\{.*\.(head|base)\.ref' "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null || true)

# Check 10: curl | bash patterns
echo "Checking for curl pipe to shell patterns..."
while IFS= read -r file; do
    [[ -n "$file" ]] && log_issue "HIGH" "$file" "Contains 'curl | bash' pattern - download and verify scripts before execution"
done < <(grep -rlE 'curl.*\|\s*(ba)?sh|wget.*\|\s*(ba)?sh' "$GITHUB_DIR/workflows" "$GITHUB_DIR/actions" 2>/dev/null || true)

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
    echo "  - HIGH/CRITICAL: Fix immediately - direct shell injection risk"
    echo "  - MEDIUM: Fix soon - potential injection or supply chain risk"
    echo "  - LOW: Defense in depth - address in regular maintenance"
    echo ""
    echo "For more details, see the security remediation plan."

    if [[ "$STRICT_MODE" == "true" ]]; then
        exit 1
    fi
    exit 0
fi
