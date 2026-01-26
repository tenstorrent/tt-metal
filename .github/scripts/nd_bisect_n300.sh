#!/usr/bin/env bash
set -euo pipefail

# Generic ND bisect script for N300 tests
# Usage:
#   ./nd_bisect_n300.sh <test_function> [options]
#
# Options:
#   --bad-commit <sha>         Known-bad commit to start search from (default: HEAD)
#   --retries <num>            Number of retries per commit (auto-determined on CI if not provided)
#   --timeout <minutes>        Timeout per test run (default: 60)
#   --runner-label <label>     Runner label (default: N300)
#   --tracy                    Enable Tracy profiling (default: true)
#   --no-search                Disable search mode (requires --commit-range)
#   --commit-range <good,bad>  Commit range for bisect (required if --no-search)
#
# Examples:
#   # Start search from specific known-bad commit (auto-determines retry count on CI)
#   ./nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f28
#
#   # Same but with pre-determined retry count (faster)
#   ./nd_bisect_n300.sh run_bert_func --bad-commit 51fc518f28 --retries 30
#
#   # Use commit range instead of search mode
#   ./nd_bisect_n300.sh run_bert_func --no-search --commit-range abc123,def456 --retries 30

die() { echo "ERROR: $*" >&2; exit 1; }
usage() {
  cat <<EOF
Usage: $0 <test_function> [options]

Required:
  test_function              Name of test function from run_single_card_demo_tests.sh
                            (e.g., run_bert_func, run_resnet_func)

Options:
  --bad-commit <sha>        Known-bad commit to start search from (default: HEAD)
  --retries <num>           Number of retries per commit (auto-determined on CI if not provided)
  --timeout <minutes>       Timeout per test run (default: 60)
  --runner-label <label>    Runner label (default: N300)
  --tracy                   Enable Tracy profiling (default: true)
  --no-tracy                Disable Tracy profiling
  --no-search               Disable search mode (requires --commit-range)
  --commit-range <good,bad> Commit range for bisect (required if --no-search)

Examples:
  $0 run_bert_func --bad-commit 51fc518f28
  $0 run_bert_func --bad-commit 51fc518f28 --retries 30
  $0 run_bert_func --no-search --commit-range abc123,def456 --retries 30
EOF
  exit 1
}

# Parse arguments
TEST_FUNCTION=""
BAD_COMMIT=""
RETRIES=""
TIMEOUT=60
RUNNER_LABEL="N300"
TRACY=true
SEARCH_MODE=true
COMMIT_RANGE=""

if [ $# -eq 0 ]; then
  usage
fi

TEST_FUNCTION="$1"
shift

while [[ $# -gt 0 ]]; do
  case $1 in
    --bad-commit)
      BAD_COMMIT="$2"
      shift 2
      ;;
    --retries)
      RETRIES="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --runner-label)
      RUNNER_LABEL="$2"
      shift 2
      ;;
    --tracy)
      TRACY=true
      shift
      ;;
    --no-tracy)
      TRACY=false
      shift
      ;;
    --no-search)
      SEARCH_MODE=false
      shift
      ;;
    --commit-range)
      COMMIT_RANGE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

# Validate test function name
if [[ ! "$TEST_FUNCTION" =~ ^run_.+_(func|perf|demo)$ ]]; then
  die "Invalid test function name: $TEST_FUNCTION (should match pattern: run_*_func, run_*_perf, or run_*_demo)"
fi

# Default BAD_COMMIT to HEAD if not provided
if [ -z "$BAD_COMMIT" ]; then
  BAD_COMMIT="HEAD"
fi

# Validate search mode requirements
if [ "$SEARCH_MODE" = false ] && [ -z "$COMMIT_RANGE" ]; then
  die "When --no-search is used, --commit-range must be provided"
fi

# Determine repo root for git commands
if [ -d "/work" ]; then
  REPO_ROOT="/work"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT" || die "Could not cd to repo root: $REPO_ROOT"

# Determine retry count - if not provided, use "auto" to let CI determine
if [ -z "$RETRIES" ]; then
  RETRIES="auto"
  echo "Retry count will be auto-determined on CI by running until failure"
else
  echo "Using provided retry count: $RETRIES"
fi

# Resolve BAD_COMMIT to full SHA for workflow
BAD_COMMIT_SHA=$(git rev-parse "$BAD_COMMIT")

# Dispatch the workflow
echo "════════════════════════════════════════════════════════════════"
echo "Dispatching ND bisect workflow"
echo "Test function: $TEST_FUNCTION"
echo "Runner label: $RUNNER_LABEL"
echo "Retries per commit: $RETRIES"
echo "Timeout: $TIMEOUT minutes"
echo "Search mode: $SEARCH_MODE"
echo "Start commit (bad): $BAD_COMMIT_SHA"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Construct workflow dispatch command
WORKFLOW_ARGS=(
  "bisect-dispatch.yaml"
  --ref "$(git rev-parse --abbrev-ref HEAD)"
  -f "nd-mode=true"
  -f "runner-label=$RUNNER_LABEL"
  -f "timeout=$TIMEOUT"
  -f "attempts=$RETRIES"
  -f "download-artifacts=true"
)

# Add Tracy setting
if [ "$TRACY" = true ]; then
  WORKFLOW_ARGS+=(-f "tracy=true")
else
  WORKFLOW_ARGS+=(-f "tracy=false")
fi

# Add search mode and commit range
if [ "$SEARCH_MODE" = true ]; then
  WORKFLOW_ARGS+=(-f "search-mode=true")
  WORKFLOW_ARGS+=(-f "start-commit=$BAD_COMMIT_SHA")
else
  WORKFLOW_ARGS+=(-f "search-mode=false")
  WORKFLOW_ARGS+=(-f "commit-range=$COMMIT_RANGE")
fi

# Add test script (use CI path format for workflow)
WORKFLOW_ARGS+=(-f "test-script=source /work/tests/scripts/single_card/run_single_card_demo_tests.sh && $TEST_FUNCTION")

# Dispatch workflow
gh workflow run "${WORKFLOW_ARGS[@]}"

echo ""
echo "Bisect workflow dispatched successfully!"
echo "View the workflow run with: gh run list --workflow=bisect-dispatch.yaml"
