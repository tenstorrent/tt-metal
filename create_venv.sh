#!/bin/bash
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

# Validate Python version format (e.g., 3.10, 3.11, 3.12)
validate_python_version() {
    local version="$1"
    if ! [[ "$version" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: Invalid Python version format: '$version'" >&2
        echo "Expected format: X.Y or X.Y.Z (e.g., 3.10, 3.11, 3.12.1)" >&2
        exit 1
    fi

    # Extract major and minor versions
    local major="${version%%.*}"
    local minor_with_patch="${version#*.}"
    local minor="${minor_with_patch%%.*}"

    # Force base-10 interpretation to prevent octal parsing issues.
    # Without 10#, bash treats "08" or "09" as invalid octal (octal digits are 0-7),
    # causing errors like "value too great for base". The 10# prefix ensures
    # the string is always parsed as decimal regardless of leading zeros.
    major=$((10#$major))
    minor=$((10#$minor))

    # Require Python 3.10+
    if [[ "$major" -lt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -lt 10 ]]; }; then
        echo "Error: Python version must be 3.10 or higher (got: $version)" >&2
        echo "Supported versions: 3.10, 3.11, etc." >&2
        exit 1
    fi
}

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

# Determine script directory (used for locating sibling scripts and OS detection)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Import functions for detecting OS
. "$SCRIPT_DIR/install_dependencies.sh" --source-only
detect_os

# Apply configuration with precedence: arguments > OS detection > default
# Python version
if [ -n "$ARG_PYTHON_VERSION" ]; then
    VENV_PYTHON_VERSION="$ARG_PYTHON_VERSION"
elif [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "24.04" ]; then
    VENV_PYTHON_VERSION="3.12"
elif [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "22.04" ]; then
    VENV_PYTHON_VERSION="3.10"
else
    VENV_PYTHON_VERSION="3.10"
fi

# Environment directory
if [ -n "$ARG_ENV_DIR" ]; then
    PYTHON_ENV_DIR="$ARG_ENV_DIR"
elif [ -z "${PYTHON_ENV_DIR:-}" ]; then
    PYTHON_ENV_DIR="$(pwd)/python_env"
fi

# ============================================================================
# Validate Configuration
# ============================================================================

validate_python_version "$VENV_PYTHON_VERSION"
validate_env_dir "$PYTHON_ENV_DIR"

# Determine script directory (used for locating sibling scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Extract UV_VERSION from install-uv.sh for consistency (if available)
# This ensures we're aware of the version being used, even though install-uv.sh handles installation
if [ -f "$SCRIPT_DIR/scripts/install-uv.sh" ]; then
    UV_VERSION=$(grep -m1 '^UV_VERSION=' "$SCRIPT_DIR/scripts/install-uv.sh" | cut -d'"' -f2)
    if [ -n "$UV_VERSION" ]; then
        echo "Using uv version from install-uv.sh: $UV_VERSION"
        # Optional: Verify installed version matches pinned version (informational only)
        if command -v uv &>/dev/null; then
            installed_version=$(uv --version 2>/dev/null | cut -d' ' -f2 || echo "")
            if [ -n "$installed_version" ] && [ "$installed_version" != "$UV_VERSION" ]; then
                echo "Warning: Installed uv version ($installed_version) differs from pinned version ($UV_VERSION)" >&2
                echo "Consider running: bash $SCRIPT_DIR/scripts/install-uv.sh --force" >&2
            fi
        fi
    fi
fi

# Install uv if not already available
if ! command -v uv &>/dev/null; then
    if [ -f "$SCRIPT_DIR/scripts/install-uv.sh" ]; then
        # install-uv.sh handles: version pinning, pip/standalone fallback, PATH updates
        bash "$SCRIPT_DIR/scripts/install-uv.sh"
    else
        echo "Error: scripts/install-uv.sh not found" >&2
        echo "Please ensure you are running this script from the repository root." >&2
        exit 1
    fi

    # Ensure uv is in PATH (install-uv.sh may have installed to ~/.local/bin or ~/.cargo/bin)
    for bin_dir in "${HOME}/.local/bin" "${HOME}/.cargo/bin"; do
        if [[ -d "$bin_dir" ]] && [[ ":$PATH:" != *":$bin_dir:"* ]]; then
            export PATH="$bin_dir:$PATH"
        fi
    done

    # Verify uv is available
    if ! command -v uv &>/dev/null; then
        echo "Error: uv not found in PATH after installation" >&2
        echo "Please ensure ~/.local/bin or ~/.cargo/bin is in your PATH and try again." >&2
        exit 1
    fi
fi

# Create virtual environment
echo "Creating virtual env in: $PYTHON_ENV_DIR"
echo "  Python version: ${VENV_PYTHON_VERSION}"

# Install Python via uv and create virtual environment
echo "Installing Python ${VENV_PYTHON_VERSION} via uv..."
uv python install "${VENV_PYTHON_VERSION}"
uv venv --link-mode copy --relocatable --managed-python --python "${VENV_PYTHON_VERSION}" "$PYTHON_ENV_DIR"
source "$PYTHON_ENV_DIR/bin/activate"

# Install uv into the venv at the same version as the invoking uv
UV_CURRENT_VERSION=$(uv --version | cut -d' ' -f2)
echo "Installing uv ${UV_CURRENT_VERSION} into venv..."
uv pip install "uv==${UV_CURRENT_VERSION}"

# Import functions for detecting OS (use absolute path from SCRIPT_DIR)
. "$SCRIPT_DIR/install_dependencies.sh" --source-only
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

# Create .pth files for ttml
# This allows using pre-built ttml from build_metal.sh --build-tt-train
SITE_PACKAGES="$PYTHON_ENV_DIR/lib/python${VENV_PYTHON_VERSION}/site-packages"

TTML_SRC_DIR="$SCRIPT_DIR/tt-train/sources/ttml"
TTML_BUILD_DIR="$SCRIPT_DIR/build/tt-train/sources/ttml"

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
if [ "$(git rev-parse --git-dir)" = "$(git rev-parse --git-common-dir)" ]; then
    echo "Generating git hooks"
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "In worktree: not generating git hooks"
fi

# Bundle Python interpreter into the venv if requested
if [[ "$BUNDLE_PYTHON" == "true" ]]; then
    echo "Bundling Python interpreter into venv..."

    # Get the real path to the Python interpreter
    REAL_PYTHON_PATH=$(readlink -f "$PYTHON_ENV_DIR/bin/python")
    echo "  Python interpreter: $REAL_PYTHON_PATH"

    # Extract the cpython installation directory (parent of bin/python)
    # Path is in form: <prefix>/python/cpython<version>/bin/python
    CPYTHON_DIR=$(dirname "$(dirname "$REAL_PYTHON_PATH")")
    echo "  CPython directory: $CPYTHON_DIR"

    # Remove python symlinks in venv (they may not match the interpreter's structure)
    echo "  Removing venv python symlinks..."
    rm -f "$PYTHON_ENV_DIR/bin/python"*

    # Copy the cpython directory contents into the venv
    echo "  Copying Python interpreter files into venv..."
    cp -r "$CPYTHON_DIR"/* "$PYTHON_ENV_DIR/"

    # Patch activation scripts to set PYTHONHOME for NFS/distributed contexts
    # PYTHONHOME should point to the venv root where lib/python<version> exists
    echo "  Patching activation scripts to set PYTHONHOME..."

    # Patch bin/activate (bash/zsh) using awk for portability
    # Key insight: The activate script has a section that unsets PYTHONHOME midway through,
    # so we must set PYTHONHOME after that, right before the final hash -r call.
    # We also need to modify deactivate() to unset PYTHONHOME if no old value was saved.
    if [ -f "$PYTHON_ENV_DIR/bin/activate" ]; then
        awk '
        # Modify deactivate to properly unset PYTHONHOME when no old value exists
        # Match: unset _OLD_VIRTUAL_PYTHONHOME followed by fi
        /^[[:space:]]*unset _OLD_VIRTUAL_PYTHONHOME$/ {
            print
            if (getline nextline > 0) {
                if (nextline ~ /^[[:space:]]*fi$/) {
                    print "    else"
                    print "        unset PYTHONHOME"
                }
                print nextline
            }
            next
        }
        # Insert PYTHONHOME export right before the final hash -r call
        /^hash -r 2>\/dev\/null \|\| true$/ {
            print "# Set PYTHONHOME for bundled Python (required for NFS/distributed contexts)"
            print "PYTHONHOME=\"$VIRTUAL_ENV\""
            print "export PYTHONHOME"
            print ""
            print $0
            next
        }
        { print }
        ' "$PYTHON_ENV_DIR/bin/activate" > "$PYTHON_ENV_DIR/bin/activate.tmp"
        mv "$PYTHON_ENV_DIR/bin/activate.tmp" "$PYTHON_ENV_DIR/bin/activate"
        chmod +x "$PYTHON_ENV_DIR/bin/activate"
    fi

    # Patch bin/activate.csh (C shell)
    if [ -f "$PYTHON_ENV_DIR/bin/activate.csh" ]; then
        awk '
        # Insert PYTHONHOME export right before the rehash call
        /^rehash$/ {
            print "# Set PYTHONHOME for bundled Python (required for NFS/distributed contexts)"
            print "setenv PYTHONHOME \"$VIRTUAL_ENV\""
            print ""
            print $0
            next
        }
        { print }
        ' "$PYTHON_ENV_DIR/bin/activate.csh" > "$PYTHON_ENV_DIR/bin/activate.csh.tmp"
        mv "$PYTHON_ENV_DIR/bin/activate.csh.tmp" "$PYTHON_ENV_DIR/bin/activate.csh"
    fi

    # Patch bin/activate.fish (Fish shell)
    if [ -f "$PYTHON_ENV_DIR/bin/activate.fish" ]; then
        awk '
        { print }
        END {
            print ""
            print "# Set PYTHONHOME for bundled Python (required for NFS/distributed contexts)"
            print "set -gx PYTHONHOME \"$VIRTUAL_ENV\""
        }
        ' "$PYTHON_ENV_DIR/bin/activate.fish" > "$PYTHON_ENV_DIR/bin/activate.fish.tmp"
        mv "$PYTHON_ENV_DIR/bin/activate.fish.tmp" "$PYTHON_ENV_DIR/bin/activate.fish"
    fi

    # Patch Activate.ps1 (PowerShell)
    if [ -f "$PYTHON_ENV_DIR/bin/Activate.ps1" ]; then
        awk '
        { print }
        END {
            print ""
            print "# Set PYTHONHOME for bundled Python (required for NFS/distributed contexts)"
            print "$env:PYTHONHOME = $env:VIRTUAL_ENV"
        }
        ' "$PYTHON_ENV_DIR/bin/Activate.ps1" > "$PYTHON_ENV_DIR/bin/Activate.ps1.tmp"
        mv "$PYTHON_ENV_DIR/bin/Activate.ps1.tmp" "$PYTHON_ENV_DIR/bin/Activate.ps1"
    fi

    echo "  Python interpreter bundled successfully"
fi

# Compile bytecode at the end to take advantage of parallelism
echo "Compiling bytecode (for improved startup performance)..."
python -m compileall -j 0 -q "$PYTHON_ENV_DIR/lib" 2>/dev/null || true
echo "Bytecode compilation completed"

echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
