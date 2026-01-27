#!/bin/bash
################################################################################
# Automated Block Variants Implementation Runner
################################################################################
#
# This script sets up the environment and runs the AI agent implementation
# for adding block variants to the tt-metal Compute API.
#
# Usage:
#   ./run_agent_implementation.sh [OPTIONS]
#
# Options:
#   --phase N       Run only phase N (1-7)
#   --dry-run       Show what would be done without making changes
#   --verbose       Show detailed output
#   --skip-api      Skip API calls, use cached results
#   --help          Show this help message
#
# Examples:
#   ./run_agent_implementation.sh                    # Run all phases
#   ./run_agent_implementation.sh --phase 1          # Run only inventory phase
#   ./run_agent_implementation.sh --dry-run          # Dry run (no changes)
#   ./run_agent_implementation.sh --phase 2 --verbose # Run phase 2 with verbose output
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/add_block_variants.py"

# Functions
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${GREEN}Block Variants Automated Implementation${NC}              ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  tt-metal Compute API - Issue #35739                   ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

show_help() {
    head -n 30 "$0" | tail -n +3 | sed 's/^# //'
    exit 0
}

# Check for help
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
fi

# Print header
print_header

# Check prerequisites
print_info "Checking prerequisites..."

# Check if bashrc exists and source it
if [ -f "$HOME/.bashrc" ]; then
    print_info "Sourcing ~/.bashrc for API configuration..."
    source "$HOME/.bashrc"
else
    print_warning "~/.bashrc not found, using environment variables as-is"
fi

# Verify API key
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    print_error "No API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY"
    exit 1
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    print_success "Anthropic API key found"
    if [ -n "$ANTHROPIC_BASE_URL" ]; then
        print_info "Using custom base URL: $ANTHROPIC_BASE_URL"
    fi
    if [ -n "$ANTHROPIC_MODEL" ]; then
        print_info "Using model: $ANTHROPIC_MODEL"
    fi
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Check for required Python packages
print_info "Checking Python dependencies..."
python3 -c "import anthropic" 2>/dev/null && print_success "anthropic package installed" || print_warning "anthropic package not installed (optional)"
python3 -c "import openai" 2>/dev/null && print_success "openai package installed" || print_warning "openai package not installed (optional)"

# Check repository
REPO_PATH="/localdev/ncvetkovic/reconfig/tt-metal"
if [ ! -d "$REPO_PATH" ]; then
    print_error "Repository not found: $REPO_PATH"
    exit 1
fi

print_success "Repository found: $REPO_PATH"

# Check git branch
cd "$REPO_PATH"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
print_info "Current branch: $CURRENT_BRANCH"
if [ "$CURRENT_BRANCH" != "ncvetkovic/35739_add_missing_functions" ]; then
    print_warning "Not on expected branch (ncvetkovic/35739_add_missing_functions)"
    print_warning "Current branch: $CURRENT_BRANCH"
fi

# Create cache directory
CACHE_DIR="${SCRIPT_DIR}/.cache"
mkdir -p "$CACHE_DIR"
print_success "Cache directory ready: $CACHE_DIR"

echo ""
print_info "Starting automated implementation..."
echo ""

# Run Python script with all arguments
python3 "$PYTHON_SCRIPT" "$@"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    print_success "Implementation completed successfully!"
    echo ""
    print_info "Next steps:"
    echo "  1. Review generated code in: ${REPO_PATH}/tt_metal/include/compute_kernel_api/"
    echo "  2. Run build: cd ${REPO_PATH} && ./build_metal.sh"
    echo "  3. Run tests (if implemented)"
    echo "  4. Commit changes: git commit -m '#35739: Add block variants'"
    echo ""
else
    print_error "Implementation failed with exit code: $EXIT_CODE"
    echo ""
    print_info "Check logs and cache files in: $CACHE_DIR"
    echo ""
    exit $EXIT_CODE
fi
