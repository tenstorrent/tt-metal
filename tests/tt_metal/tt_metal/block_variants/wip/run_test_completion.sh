#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

# ===========================================================================
# AI Agent Test TODO Completer - Orchestration Script
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/complete_test_todos.py"
REPO_ROOT="${SCRIPT_DIR}/../../../../.."
TEST_DIR="${SCRIPT_DIR}/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ===========================================================================
# Functions
# ===========================================================================

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  AI Agent Test TODO Completer                         ║"
    echo "║  tt-metal Compute API - Issue #35739                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Use AI agents to automatically complete TODO sections in generated test files.
Each agent works on one operation independently.

OPTIONS:
    --operation OP      Complete TODOs for specific operation
                        (eltwise_binary, broadcast, transpose, reduce, pack)
    --all               Complete TODOs for all operations (sequential)
    --parallel          Complete TODOs for all operations (parallel, 5 agents)
    --dry-run           Preview what would be done without making changes
    --generate-first    Generate test skeletons before completing
    --help              Show this help message

EXAMPLES:
    # Complete specific operation
    $0 --operation eltwise_binary

    # Complete all operations sequentially (~10 minutes total)
    $0 --all

    # Complete all operations in parallel (~2 minutes total)
    $0 --parallel

    # Generate skeletons first, then complete
    $0 --generate-first --all

    # Dry run to preview
    $0 --operation reduce --dry-run

WORKFLOW:
    1. First generate test skeletons:
       ./run_test_generation.sh --all

    2. Then complete TODOs with agents:
       $0 --parallel

    Or combine both steps:
       $0 --generate-first --parallel

EOF
}

check_prerequisites() {
    echo -e "${BLUE}[INFO]${NC} Checking prerequisites..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[✗]${NC} Python 3 not found"
        exit 1
    fi
    echo -e "${GREEN}[✓]${NC} Python $(python3 --version | cut -d' ' -f2) found"

    # Check anthropic package
    if ! python3 -c "import anthropic" 2>/dev/null; then
        echo -e "${YELLOW}[!]${NC} anthropic package not installed"
        echo -e "${BLUE}[INFO]${NC} Installing anthropic package..."
        pip install anthropic
    else
        echo -e "${GREEN}[✓]${NC} anthropic package installed"
    fi

    # Check API key
    source ~/.bashrc 2>/dev/null || true
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "${RED}[✗]${NC} ANTHROPIC_API_KEY not set in ~/.bashrc"
        exit 1
    fi
    echo -e "${GREEN}[✓]${NC} Anthropic API key found"

    # Check repository
    if [ ! -d "${REPO_ROOT}" ]; then
        echo -e "${RED}[✗]${NC} Repository not found: ${REPO_ROOT}"
        exit 1
    fi
    echo -e "${GREEN}[✓]${NC} Repository found"

    # Check test directory
    if [ ! -d "${TEST_DIR}" ]; then
        echo -e "${YELLOW}[!]${NC} Test directory not found: ${TEST_DIR}"
        echo -e "${BLUE}[INFO]${NC} You need to generate test skeletons first"
        echo -e "${BLUE}[INFO]${NC} Run: ./run_test_generation.sh --all"
        if [ "$GENERATE_FIRST" = true ]; then
            echo -e "${BLUE}[INFO]${NC} --generate-first specified, will generate now..."
            return 0
        fi
        exit 1
    fi
    echo -e "${GREEN}[✓]${NC} Test directory found"
}

generate_skeletons() {
    echo -e "${CYAN}[GENERATE]${NC} Generating test skeletons..."
    if ./run_test_generation.sh --all; then
        echo -e "${GREEN}[✓]${NC} Test skeletons generated"
    else
        echo -e "${RED}[✗]${NC} Failed to generate test skeletons"
        exit 1
    fi
}

complete_operation() {
    local operation=$1
    local dry_run=$2

    echo -e "${CYAN}[AGENT]${NC} Starting agent for: ${operation}"

    # Ensure user site-packages are in PYTHONPATH
    export PYTHONPATH="${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH}"

    local cmd="python3 ${PYTHON_SCRIPT} --operation ${operation}"
    if [ "$dry_run" = true ]; then
        cmd="${cmd} --dry-run"
    fi

    if eval "${cmd}"; then
        echo -e "${GREEN}[✓]${NC} Agent completed: ${operation}"
        return 0
    else
        echo -e "${RED}[✗]${NC} Agent failed: ${operation}"
        return 1
    fi
}

complete_all_sequential() {
    local dry_run=$1

    echo -e "${BLUE}[INFO]${NC} Completing all operations sequentially..."
    echo -e "${BLUE}[INFO]${NC} This may take 5-10 minutes..."
    echo ""

    local operations=("eltwise_binary" "reduce" "broadcast" "transpose" "pack")
    local success=0
    local failed=0

    for op in "${operations[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if complete_operation "$op" "$dry_run"; then
            ((success++))
        else
            ((failed++))
        fi
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}✅ Success: ${success}${NC} | ${RED}Failed: ${failed}${NC}"

    return $failed
}

complete_all_parallel() {
    local dry_run=$1

    echo -e "${BLUE}[INFO]${NC} Completing all operations in parallel (5 agents)..."
    echo -e "${BLUE}[INFO]${NC} This may take 2-3 minutes..."
    echo ""

    local operations=("eltwise_binary" "reduce" "broadcast" "transpose" "pack")
    local pids=()
    local results=()

    # Start all agents in parallel
    for op in "${operations[@]}"; do
        echo -e "${CYAN}[START]${NC} Agent ${op}..."
        (
            complete_operation "$op" "$dry_run" > "/tmp/agent_${op}.log" 2>&1
            echo $? > "/tmp/agent_${op}.exit"
        ) &
        pids+=($!)
    done

    echo ""
    echo -e "${BLUE}[INFO]${NC} All 5 agents running in parallel..."
    echo -e "${BLUE}[INFO]${NC} PID's: ${pids[@]}"
    echo ""

    # Wait for all to complete
    for i in "${!pids[@]}"; do
        local op="${operations[$i]}"
        local pid="${pids[$i]}"

        echo -e "${BLUE}[WAIT]${NC} Waiting for ${op} (PID: ${pid})..."
        wait "$pid"
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check results
    local success=0
    local failed=0

    for op in "${operations[@]}"; do
        local exit_code=$(cat "/tmp/agent_${op}.exit" 2>/dev/null || echo "1")
        if [ "$exit_code" = "0" ]; then
            echo -e "${GREEN}[✓]${NC} ${op}"
            ((success++))
        else
            echo -e "${RED}[✗]${NC} ${op}"
            echo -e "${YELLOW}[LOG]${NC} See /tmp/agent_${op}.log for details"
            ((failed++))
        fi

        # Cleanup
        rm -f "/tmp/agent_${op}.exit" "/tmp/agent_${op}.log"
    done

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}✅ Success: ${success}${NC} | ${RED}Failed: ${failed}${NC}"

    return $failed
}

# ===========================================================================
# Main
# ===========================================================================

print_header

# Parse arguments
OPERATION=""
ALL_OPS=false
PARALLEL=false
DRY_RUN=false
GENERATE_FIRST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --operation)
            OPERATION="$2"
            shift 2
            ;;
        --all)
            ALL_OPS=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            ALL_OPS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --generate-first)
            GENERATE_FIRST=true
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate
if [ "$ALL_OPS" = false ] && [ -z "$OPERATION" ]; then
    echo -e "${RED}[ERROR]${NC} Must specify --operation or --all"
    print_usage
    exit 1
fi

# Check prerequisites
check_prerequisites

# Generate skeletons if needed
if [ "$GENERATE_FIRST" = true ]; then
    generate_skeletons
fi

# Execute
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[DRY RUN]${NC} Preview mode - no files will be modified"
    echo ""
fi

if [ "$ALL_OPS" = true ]; then
    if [ "$PARALLEL" = true ]; then
        complete_all_parallel "$DRY_RUN"
    else
        complete_all_sequential "$DRY_RUN"
    fi
    exit_code=$?
else
    complete_operation "$OPERATION" "$DRY_RUN"
    exit_code=$?
fi

# Summary
if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All operations completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}[INFO]${NC} Next steps:"
    echo "   1. Review completed tests in: ${TEST_DIR}"
    echo "   2. Build: cd ${REPO_ROOT} && ./build_metal.sh --build-tests"
    echo "   3. Run: ./build/test/tt_metal/test_*_block"
    echo ""
else
    echo ""
    echo -e "${YELLOW}⚠️  Some operations failed${NC}"
    echo -e "${BLUE}[INFO]${NC} Check the logs above for details"
    echo ""
fi

exit $exit_code
