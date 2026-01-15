#!/bin/bash
set -eo pipefail

# Function to display help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

This script creates a Python virtual environment for tt-metal development using uv.
It installs the specified Python version, sets up the virtual environment, and
installs all required dependencies including dev dependencies and tt-metal itself.

OPTIONS:
    --python VERSION    Specify the Python version to use (e.g., 3.10, 3.11)
                        This can also be set via the VENV_PYTHON_VERSION environment variable.
                        Default: 3.10

    --help             Show this help message and exit

ENVIRONMENT VARIABLES:
    VENV_PYTHON_VERSION    Python version to use for the virtual environment.
                           Default: 3.10
                           Note: Command-line argument --python takes precedence over this variable.

    PYTHON_CMD             Python command to use for initial uv installation.
                           Default: python3

    PYTHON_ENV_DIR         Directory where the virtual environment will be created.
                           Default: ./python_env

NOTE:
    If you encounter venv issues, running "uv pip install -e ." with the venv active
    may fix them without having to rebuild the entire virtual environment.

EOF
}

# Parse --python argument if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            if [ -z "$2" ]; then
                echo "Error: --python requires a version argument (e.g., --python 3.10)"
                echo "Run '$0 --help' for usage information"
                exit 1
            fi
            VENV_PYTHON_VERSION="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Set default Python version if not specified via environment variable or argument
if [ -z "$VENV_PYTHON_VERSION" ]; then
    VENV_PYTHON_VERSION="3.10"
fi

# Allow overriding Python command via environment variable
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
else
    echo "Using user-specified Python: $PYTHON_CMD"
fi

# Verify Python command exists
if ! command -v "$PYTHON_CMD" &>/dev/null; then
    echo "Python command not found: $PYTHON_CMD"
    exit 1
fi

# Install uv if not already available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/scripts/install-uv.sh" ]; then
        bash "$SCRIPT_DIR/scripts/install-uv.sh"
        # Ensure uv is available in the current shell even if installed with --user
        if ! command -v uv &>/dev/null; then
            USER_BIN="${HOME}/.local/bin"
            if [[ -d "$USER_BIN" ]] && [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
                export PATH="$USER_BIN:$PATH"
            fi
        fi
    else
        echo "Warning: install-uv.sh not found, falling back to pip installation"
        UV_VERSION="0.7.12"  # Fallback version - keep in sync with scripts/install-uv.sh
        if ! ${PYTHON_CMD} -m pip install --no-cache-dir "uv==${UV_VERSION}"; then
            echo "Initial 'pip install uv' failed. This can happen in PEP 668 externally-managed environments."
            echo "Retrying uv installation with --break-system-packages..."
            if ! ${PYTHON_CMD} -m pip install --no-cache-dir --break-system-packages "uv==${UV_VERSION}"; then
                echo "Retry with --break-system-packages failed. Retrying uv installation with --user..."
                if ! ${PYTHON_CMD} -m pip install --no-cache-dir --user "uv==${UV_VERSION}"; then
                    echo "Error: Failed to install uv via pip after trying:" >&2
                    echo "  1. Standard installation" >&2
                    echo "  2. --break-system-packages flag" >&2
                    echo "  3. --user installation" >&2
                    echo "" >&2
                    echo "Please ensure Python and pip are properly configured, or install uv manually." >&2
                    echo "You can also try running '$SCRIPT_DIR/scripts/install-uv.sh' if available." >&2
                    exit 1
                fi
            fi
        fi
        # Add pip's bin directory to PATH for the current session
        USER_BIN="${HOME}/.local/bin"
        if [[ -d "$USER_BIN" ]] && [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
            export PATH="$USER_BIN:$PATH"
        fi
    fi
fi

# Verify uv is available
if ! command -v uv &>/dev/null; then
    echo "Error: uv not found in PATH after installation"
    echo "Please ensure ~/.local/bin is in your PATH and try again"
    exit 1
fi

# Set Python environment directory
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi
echo "Creating virtual env in: $PYTHON_ENV_DIR"

# Install Python via uv and create virtual environment
echo "Installing Python ${VENV_PYTHON_VERSION} via uv..."
uv python install "${VENV_PYTHON_VERSION}"
uv venv "$PYTHON_ENV_DIR" --python "${VENV_PYTHON_VERSION}"
source "$PYTHON_ENV_DIR/bin/activate"

# Import functions for detecting OS
. ./install_dependencies.sh --source-only
detect_os

# PyTorch CPU index URL for all uv pip commands
PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"

if [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "22.04" ]; then
    echo "Ubuntu 22.04 detected: force pip/setuptools/wheel versions"
    uv pip install --extra-index-url "$PYTORCH_INDEX" setuptools wheel==0.45.1
else
    echo "$OS_ID $OS_VERSION detected: updating wheel and setuptools to latest"
    uv pip install --upgrade wheel setuptools
fi

echo "Installing dev dependencies"
# Use --extra-index-url for PyTorch CPU wheels and index-strategy for transitive deps
uv pip install --extra-index-url "$PYTORCH_INDEX" --index-strategy unsafe-best-match -r "$(pwd)/tt_metal/python_env/requirements-dev.txt"

echo "Installing tt-metal"
uv pip install -e .

# Do not install hooks when this is a worktree
if [ "$(git rev-parse --git-dir)" = "$(git rev-parse --git-common-dir)" ]; then
    echo "Generating git hooks"
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "In worktree: not generating git hooks"
fi

echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
