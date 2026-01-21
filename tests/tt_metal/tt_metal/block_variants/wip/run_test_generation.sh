#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

# ===========================================================================
# Block Variants Test Generator - User-Friendly Wrapper
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_block_tests.py"
REPO_ROOT="${SCRIPT_DIR}/tt-metal"
OUTPUT_DIR="${REPO_ROOT}/tests/tt_metal/tt_metal/block_variants"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ===========================================================================
# Helper Functions
# ===========================================================================

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Block Variants Test Generator                        ║"
    echo "║  tt-metal Compute API - Issue #35739                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate comprehensive tests for block variant Compute API functions.

OPTIONS:
    --operation OP      Generate tests for specific operation
                        (eltwise_binary, broadcast, transpose, reduce, pack)
    --all               Generate tests for all operations
    --list              List available operations
    --output DIR        Custom output directory (default: auto-detected)
    --dry-run           Show what would be generated without creating files
    --help              Show this help message

EXAMPLES:
    # Generate tests for element-wise binary operations
    $0 --operation eltwise_binary

    # Generate all tests
    $0 --all

    # List available operations
    $0 --list

    # Custom output directory
    $0 --operation reduce --output /path/to/output

    # Dry run to preview
    $0 --all --dry-run

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

    # Check repository
    if [ ! -d "${REPO_ROOT}" ]; then
        echo -e "${RED}[✗]${NC} tt-metal repository not found at ${REPO_ROOT}"
        exit 1
    fi
    echo -e "${GREEN}[✓]${NC} Repository found: ${REPO_ROOT}"

    # Check script
    if [ ! -f "${PYTHON_SCRIPT}" ]; then
        echo -e "${RED}[✗]${NC} Python script not found: ${PYTHON_SCRIPT}"
        exit 1
    fi
    echo -e "${GREEN}[✓]${NC} Generator script ready"
}

# ===========================================================================
# Main
# ===========================================================================

print_header

# Parse arguments
OPERATION=""
ALL_OPS=false
LIST_OPS=false
DRY_RUN=false
CUSTOM_OUTPUT=""

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
        --list)
            LIST_OPS=true
            shift
            ;;
        --output)
            CUSTOM_OUTPUT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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

# List operations
if [ "$LIST_OPS" = true ]; then
    python3 "${PYTHON_SCRIPT}" --list
    exit 0
fi

# Check prerequisites
check_prerequisites

# Build Python command
PY_CMD="python3 ${PYTHON_SCRIPT}"

if [ -n "$CUSTOM_OUTPUT" ]; then
    PY_CMD="${PY_CMD} --output ${CUSTOM_OUTPUT}"
else
    PY_CMD="${PY_CMD} --output ${OUTPUT_DIR}"
fi

if [ "$ALL_OPS" = true ]; then
    PY_CMD="${PY_CMD} --all"
elif [ -n "$OPERATION" ]; then
    PY_CMD="${PY_CMD} --operation ${OPERATION}"
else
    echo -e "${RED}[ERROR]${NC} Must specify --operation or --all"
    print_usage
    exit 1
fi

# Dry run
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[DRY RUN]${NC} Would execute:"
    echo "  ${PY_CMD}"
    echo ""
    echo -e "${YELLOW}[INFO]${NC} Dry run mode - no files will be created"
    exit 0
fi

# Run generator
echo -e "${BLUE}[INFO]${NC} Starting test generation..."
echo ""

if eval "${PY_CMD}"; then
    echo ""
    echo -e "${GREEN}[✓]${NC} Test generation completed successfully!"
    echo ""
    echo -e "${BLUE}[INFO]${NC} Next steps:"
    echo "  1. Review generated files in: ${OUTPUT_DIR}"
    echo "  2. Complete TODO sections in test harnesses"
    echo "  3. Build tests: cd ${REPO_ROOT} && ./build_metal.sh --build-tests"
    echo "  4. Run tests: ./build/test/tt_metal/test_*_block"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}[✗]${NC} Test generation failed"
    exit 1
fi
