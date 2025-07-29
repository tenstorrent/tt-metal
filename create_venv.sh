#!/bin/bash
set -eo pipefail

# Colors
GREEN="\e[32m"
YELLOW="\e[33m"
PURPLE="\e[35m"
CYAN="\e[36m"
RED="\e[31m"
BLUE="\e[34m"
BOLD="\e[1m"
RESET="\e[0m"

# Logging functions
log_info()            { echo -e "${CYAN}[INFO] $*${RESET}"; }
log_warn()            { echo -e "${YELLOW}[WARN] $*${RESET}"; }
log_error()           { echo -e "${RED}[ERROR] $*${RESET}"; }
log_success()         { echo -e "${GREEN}[OK] $*${RESET}"; }
log_step()            { echo -e "${BLUE}${BOLD}[*] $*${RESET}"; }
log_note()            { echo -e "${PURPLE}[NOTE] $*${RESET}"; }
log_option_selected() { echo -e "${GREEN}[SELECTED] ${BOLD}$*${RESET}"; }
log_done()            { echo -e "${GREEN}${BOLD}[DONE] $*${RESET}"; }

# Check if the script is sourced
is_sourced=false
if [ "$0" != "$BASH_SOURCE" ]; then
    is_sourced=true
fi

# Allow overriding Python command via environment variable
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
else
    log_info "Using user-specified Python: $PYTHON_CMD"
fi

# Verify Python command exists
if ! command -v $PYTHON_CMD &>/dev/null; then
    log_error "Python command not found: $PYTHON_CMD"
    exit 1
fi

# Set Python environment directory
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi
log_step "Creating virtual environment in: $PYTHON_ENV_DIR"

# Create virtual environment
$PYTHON_CMD -m venv $PYTHON_ENV_DIR

# Activate if sourced
if [ "$is_sourced" = true ]; then
    source $PYTHON_ENV_DIR/bin/activate
    log_success "Activated source as python_env."
else
    log_note "[Note] Virtual environment created but not activated in current shell."
    log_info "To activate, run: source $PYTHON_ENV_DIR/bin/activate"
fi

# Ensure Rust from rustup is used inside the venv
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
    log_info "Using Rust version: $(rustc --version)"

    # Verify minimum Rust version
    MIN_RUST="1.66.0"
    if [ "$(printf '%s\n' $MIN_RUST $(rustc --version | awk '{print $2}') | sort -V | head -n1)" != "$MIN_RUST" ]; then
        log_warn "Rust is older than $MIN_RUST. Updating..."
        rustup install stable
        rustup default stable
    fi
else
    log_warn "rustup environment not found. Rust may be outdated."
fi

log_info "Forcefully using a version of pip that will work with our view of editable installs"
$PYTHON_ENV_DIR/bin/pip install --force-reinstall pip==21.2.4

log_info "Setting up virtual environment"
$PYTHON_ENV_DIR/bin/python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
$PYTHON_ENV_DIR/bin/python3 -m pip install setuptools wheel==0.45.1

log_info "Installing dev dependencies"
$PYTHON_ENV_DIR/bin/python3 -m pip install -r $(pwd)/tt_metal/python_env/requirements-dev.txt

log_info "Installing tt-metal"
$PYTHON_ENV_DIR/bin/pip install -e .

# Prompt for card model
log_info "Select your Tenstorrent card model:"
PS3="Enter the number for your card model: "
options=("Grayskull" "Wormhole" "Blackhole")
select opt in "${options[@]}"; do
    case $opt in
        "Grayskull")
            ARCH_NAME="grayskull"
            break
            ;;
        "Wormhole")
            ARCH_NAME="wormhole_b0"
            break
            ;;
        "Blackhole")
            ARCH_NAME="blackhole"
            break
            ;;
        *)
            log_warn "Invalid option, please try again."
            ;;
    esac
done
log_option_selected "Card model: $ARCH_NAME"

# Add environment variables to venv activation script
log_step "Configuring TT-Metal environment variables in venv activation..."
cat <<EOL >> $PYTHON_ENV_DIR/bin/activate

# TT-Metal environment variables
export ARCH_NAME=$ARCH_NAME
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# Reminder: If you switch card types (e.g., wormhole_b0 → blackhole), update ARCH_NAME in this file.
EOL

log_note "If you switch card types (e.g., wormhole_b0 → blackhole), update ARCH_NAME in python_env/bin/activate"

# Do not install hooks when this is a worktree
if [ "$(git rev-parse --git-dir)" = "$(git rev-parse --git-common-dir)" ]; then
    log_step "Generating git hooks"

    # Ensure pre-commit is installed
    if ! $PYTHON_ENV_DIR/bin/pre-commit --version &> /dev/null; then
        log_warn "pre-commit not found in venv. Installing..."
        $PYTHON_ENV_DIR/bin/pip install pre-commit
    fi

    $PYTHON_ENV_DIR/bin/pre-commit install
    $PYTHON_ENV_DIR/bin/pre-commit install --hook-type commit-msg
else
    log_warn "In worktree: not generating git hooks"
fi

log_success "Successfully created venv as python_env."

if [ "$is_sourced" = false ]; then
    log_info "To activate, run: source $PYTHON_ENV_DIR/bin/activate"
fi

log_done "Happy coding!"
