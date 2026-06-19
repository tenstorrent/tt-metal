#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

# Function to display help message
show_help() {
    cat << 'EOF'
Usage: create_venv.sh [OPTIONS]

Create a Python virtual environment for tt-metal development using uv.
Installs the specified Python version, sets up the virtual environment, and
installs all required dependencies including dev dependencies and tt-metal itself.

OPTIONS:
    --python-version VER  Python version for the virtual environment (e.g., 3.10, 3.11)
                          Default: 3.10
    --env-dir DIR         Directory where the virtual environment will be created.
                          Parent directories are created automatically if needed.
                          Default: ./python_env
    --force               Overwrite existing virtual environment without prompting.
                          By default, warns and prompts for confirmation if the
                          target directory exists and is not empty.
    --bundle-python       Deep-copy the Python interpreter into the venv instead of
                          using symlinks. This makes the venv fully self-contained
                          and portable, at the cost of increased disk space.
    --skip-compat-check   Skip the package compatibility check (uv pip check).
    --help, -h            Show this help message and exit

ENVIRONMENT VARIABLES:
    Environment variables provide defaults that can be overridden by command-line
    arguments. Arguments always take precedence over environment variables.

    VENV_PYTHON_VERSION   Python version (overridden by --python-version)
    PYTHON_ENV_DIR        Virtual environment directory (overridden by --env-dir)

EXAMPLES:
    # Use defaults (Python 3.10, ./python_env)
    ./create_venv.sh

    # Specify Python version
    ./create_venv.sh --python-version 3.12

    # Custom environment directory
    ./create_venv.sh --env-dir /opt/venv

    # Nested directory (parent directories created automatically)
    ./create_venv.sh --env-dir /opt/myproject/envs/dev

    # Using environment variables
    PYTHON_ENV_DIR=/opt/venv VENV_PYTHON_VERSION=3.12 ./create_venv.sh

    # Arguments override environment variables
    VENV_PYTHON_VERSION=3.10 ./create_venv.sh --python-version 3.12  # Uses 3.12

NOTE:
    If you encounter venv issues, running "uv pip install -e ." with the venv active
    may fix them without having to rebuild the entire virtual environment.
EOF
}

# Variables to track argument-provided values (take precedence over env vars)
ARG_PYTHON_VERSION=""
ARG_ENV_DIR=""
FORCE_OVERWRITE="false"
BUNDLE_PYTHON="false"
SKIP_COMPAT_CHECK="false"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python-version)
            # Check for missing or flag-like argument (--* matches flags, not negative numbers)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --python-version requires a version argument (e.g., --python-version 3.10)" >&2
                echo "Run '$0 --help' for usage information" >&2
                exit 1
            fi
            ARG_PYTHON_VERSION="$2"
            shift 2
            ;;
        --env-dir)
            # Check for missing or flag-like argument (--* matches flags, not paths like ./dir)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --env-dir requires a directory argument (e.g., --env-dir /opt/venv)" >&2
                echo "Run '$0 --help' for usage information" >&2
                exit 1
            fi
            ARG_ENV_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE_OVERWRITE="true"
            shift
            ;;
        --bundle-python)
            BUNDLE_PYTHON="true"
            shift
            ;;
        --skip-compat-check)
            SKIP_COMPAT_CHECK="true"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            echo "Run '$0 --help' for usage information" >&2
            exit 1
            ;;
    esac
done

# ============================================================================
# Validation Functions
# ============================================================================

# Validate and prepare environment directory path
# Creates parent directories if they don't exist (mkdir -p semantics)
validate_env_dir() {
    local dir="$1"
    local parent_dir
    parent_dir="$(dirname "$dir")"

    # Check if path is empty
    if [ -z "$dir" ]; then
        echo "Error: Environment directory path cannot be empty" >&2
        exit 1
    fi

    # Check if directory already exists
    if [ -d "$dir" ]; then
        # Check if existing directory is writable (needed to overwrite)
        if [ ! -w "$dir" ]; then
            echo "Error: Environment directory exists but is not writable: $dir" >&2
            echo "Please check permissions or specify a different path." >&2
            exit 1
        fi

        # Warn if not empty (will be overwritten)
        if [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
            if [[ "$FORCE_OVERWRITE" == "true" ]]; then
                echo "Warning: Overwriting existing directory (--force specified): $dir" >&2
            else
                echo "" >&2
                echo "WARNING: Environment directory already exists and is not empty:" >&2
                echo "  $dir" >&2
                echo "" >&2
                echo "The existing virtual environment will be OVERWRITTEN." >&2
                echo "" >&2
                # Check if we're in an interactive terminal
                if [ -t 0 ]; then
                    read -r -p "Continue? [y/N] " response
                    case "$response" in
                        [yY][eE][sS]|[yY])
                            echo "Proceeding with overwrite..."
                            ;;
                        *)
                            echo "Aborted. Use --force to skip this prompt." >&2
                            exit 1
                            ;;
                    esac
                else
                    echo "Non-interactive mode: use --force to overwrite without prompting." >&2
                    exit 1
                fi
            fi
        fi
    else
        # Directory doesn't exist - create parent directories if needed
        if [ ! -d "$parent_dir" ]; then
            echo "Creating parent directories: $parent_dir"
            if ! mkdir -p "$parent_dir"; then
                echo "Error: Failed to create parent directory: $parent_dir" >&2
                echo "Please check permissions or specify a different path." >&2
                exit 1
            fi
        fi

        # Check if parent directory is writable (needed to create env dir)
        if [ ! -w "$parent_dir" ]; then
            echo "Error: Parent directory is not writable: $parent_dir" >&2
            echo "Please check permissions or specify a different path." >&2
            exit 1
        fi
    fi
}

# ============================================================================
# Apply Configuration
# ============================================================================

# Determine repository root from this script's location.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Apply configuration with precedence: arguments > OS detection > default
# Python version
if [ -n "$ARG_PYTHON_VERSION" ]; then
    VENV_PYTHON_VERSION="$ARG_PYTHON_VERSION"
fi

# Environment directory
if [ -n "$ARG_ENV_DIR" ]; then
    PYTHON_ENV_DIR="$ARG_ENV_DIR"
elif [ -z "${PYTHON_ENV_DIR:-}" ]; then
    PYTHON_ENV_DIR="$ROOT_DIR/python_env"
fi

# ============================================================================
# Validate Configuration
# ============================================================================

validate_env_dir "$PYTHON_ENV_DIR"

# Create virtual environment
echo "Creating virtual env in: $PYTHON_ENV_DIR"
echo "  Python version: ${VENV_PYTHON_VERSION}"


UV_VENV_ARGS=(--link-mode copy --relocatable --no-managed-python --python "${VENV_PYTHON_VERSION}")
if [[ "$FORCE_OVERWRITE" == "true" ]]; then
    UV_VENV_ARGS+=(--clear)
fi
uv venv "${UV_VENV_ARGS[@]}" "$PYTHON_ENV_DIR"

# Patch activate for POSIX sh (Docker/CI use /bin/sh; relocatable activate uses $BASH_SOURCE)
# Use 'if' to prevent set -e from exiting on patch failure
if PATCH_OUTPUT=$("${ROOT_DIR}/scripts/patch_activate_posix.sh" "$PYTHON_ENV_DIR" 2>&1); then
  if echo "$PATCH_OUTPUT" | grep -q "Skip"; then
    echo "INFO: patch_activate_posix.sh skipped (venv activate not relocatable or already patched)"
  else
    echo "INFO: $PATCH_OUTPUT"
  fi
else
  echo "WARNING: patch_activate_posix.sh failed (rc=$?): $PATCH_OUTPUT" >&2
fi

source "$PYTHON_ENV_DIR/bin/activate"

# PyTorch CPU index URL for all uv pip commands
PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"


echo "$OS_ID $OS_VERSION detected: updating wheel and setuptools to latest"
uv pip install --upgrade wheel setuptools==80


echo "Installing dev dependencies"
# Use --extra-index-url for PyTorch CPU wheels and index-strategy for transitive deps
# no-build-isolation as a workaround for setuptools/mmcv
uv pip install --extra-index-url "$PYTORCH_INDEX" \
    --index-strategy unsafe-best-match \
    --no-build-isolation \
    -r "$ROOT_DIR/tt_metal/python_env/requirements-dev.txt"

echo "Installing tt-triage dependencies"
uv pip install --index-strategy unsafe-best-match -r "$ROOT_DIR/tools/triage/requirements.txt"

echo "Installing tt-metal"
uv pip install -e "$ROOT_DIR"

if [[ "$SKIP_COMPAT_CHECK" == "true" ]]; then
    echo "Skipping package compatibility check (--skip-compat-check)"
else
    echo "Checking packages are compatible"
    if ! uv pip check; then
        echo -e "\033[1;31mERROR: Package compatibility check failed. See above for details.\033[0m" >&2
        exit 1
    fi
fi

# Create .pth files for ttml
# This allows using pre-built ttml from build_metal.sh --build-tt-train
SITE_PACKAGES="$PYTHON_ENV_DIR/lib/python${VENV_PYTHON_VERSION}/site-packages"

TTML_SRC_DIR="$ROOT_DIR/tt-train/sources/ttml"
TTML_BUILD_DIR="$ROOT_DIR/build/tt-train/sources/ttml"

# Add ttml Python source code, if available
if [ -d "$TTML_SRC_DIR" ]; then
    echo "$TTML_SRC_DIR" > "$SITE_PACKAGES/ttml.pth"
    echo "  Created: $SITE_PACKAGES/ttml.pth"
else
    echo "  Skipping ttml.pth creation (directory not found: $TTML_SRC_DIR)"
fi

# Add the built _ttml C++ extension (.so file), if available
# Uses the 'build' symlink which points to the active build directory (e.g., build_Release)
if [ -d "$TTML_BUILD_DIR" ]; then
    echo "$TTML_BUILD_DIR" > "$SITE_PACKAGES/_ttml.pth"
    echo "  Created: $SITE_PACKAGES/_ttml.pth"
else
    echo "  Skipping _ttml.pth creation (directory not found: $TTML_BUILD_DIR)"
fi
# Do not install hooks when this is a worktree
if [ "$(git -C "$ROOT_DIR" rev-parse --git-dir)" = "$(git -C "$ROOT_DIR" rev-parse --git-common-dir)" ]; then
    echo "Generating git hooks"
    (
        cd "$ROOT_DIR"
        pre-commit install
    )
    # Note: do NOT add `pre-commit install --hook-type commit-msg` here unless
    # .pre-commit-config.yaml contains hooks with `stages: [commit-msg]`.
    # Without matching hooks, the commit-msg invocation runs the default-stage
    # hooks a second time, causing every `git commit` to run pre-commit twice.
else
    echo "In worktree: not generating git hooks"
fi

# Bundle Python interpreter into the venv if requested
if [[ "$BUNDLE_PYTHON" == "true" ]]; then
    "${ROOT_DIR}/scripts/bundle_python_into_venv.sh" "$PYTHON_ENV_DIR" --force
fi

# Compile bytecode at the end to take advantage of parallelism
echo "Compiling bytecode (for improved startup performance)..."
python -m compileall -j 0 -q "$PYTHON_ENV_DIR/lib" 2>/dev/null || true
echo "Bytecode compilation completed"

echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
