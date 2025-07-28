#!/bin/bash
set -eo pipefail

# Colors
GREEN="\e[32m"
YELLOW="\e[33m"
PURPLE="\e[35m"
CYAN="\e[36m"
RESET="\e[0m"

# Check if the script is sourced
is_sourced=false
if [ "$0" != "$BASH_SOURCE" ]; then
    is_sourced=true
fi

# Allow overriding Python command via environment variable
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
else
    echo -e "${CYAN}Using user-specified Python: $PYTHON_CMD${RESET}"
fi

# Verify Python command exists
if ! command -v $PYTHON_CMD &>/dev/null; then
    echo -e "${YELLOW}Python command not found: $PYTHON_CMD${RESET}"
    exit 1
fi

# Set Python environment directory
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi
echo -e "${CYAN}Creating virtual environment in: $PYTHON_ENV_DIR${RESET}"

# Create virtual environment
$PYTHON_CMD -m venv $PYTHON_ENV_DIR

# Activate if sourced
if [ "$is_sourced" = true ]; then
    source $PYTHON_ENV_DIR/bin/activate
    echo -e "${GREEN}Activated source as python_env.${RESET}"
else
    echo -e "${YELLOW}[Note] Virtual environment created but not activated in current shell.${RESET}"
    echo -e "${CYAN}To activate, run:${RESET} source $PYTHON_ENV_DIR/bin/activate"
fi

# Ensure Rust from rustup is used inside the venv
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
    echo -e "${CYAN}Using Rust version: $(rustc --version)${RESET}"

    # Verify minimum Rust version
    MIN_RUST="1.66.0"
    if [ "$(printf '%s\n' $MIN_RUST $(rustc --version | awk '{print $2}') | sort -V | head -n1)" != "$MIN_RUST" ]; then
        echo -e "${YELLOW}Rust is older than $MIN_RUST. Updating...${RESET}"
        rustup install stable
        rustup default stable
    fi
else
    echo -e "${RED}Warning: rustup environment not found. Rust may be outdated.${RESET}"
fi

echo -e "${CYAN}Forcefully using a version of pip that will work with our view of editable installs${RESET}"
$PYTHON_ENV_DIR/bin/pip install --force-reinstall pip==21.2.4

echo -e "${CYAN}Setting up virtual environment${RESET}"
$PYTHON_ENV_DIR/bin/python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
$PYTHON_ENV_DIR/bin/python3 -m pip install setuptools wheel==0.45.1

echo -e "${CYAN}Installing dev dependencies${RESET}"
$PYTHON_ENV_DIR/bin/python3 -m pip install -r $(pwd)/tt_metal/python_env/requirements-dev.txt

echo -e "${CYAN}Installing tt-metal${RESET}"
$PYTHON_ENV_DIR/bin/pip install -e .

echo -e "${CYAN}Installing tt-metal${RESET}"
$PYTHON_ENV_DIR/bin/pip install -e .

# Prompt for card model
echo -e "${CYAN}Select your Tenstorrent card model:${RESET}"
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
            echo -e "${YELLOW}Invalid option, please try again.${RESET}"
            ;;
    esac
done
echo -e "${GREEN}Selected card model: ${BOLD}$ARCH_NAME${RESET}"

# Add environment variables to venv activation script
echo -e "${GREEN}Configuring TT-Metal environment variables in venv activation...${RESET}"
cat <<EOL >> $PYTHON_ENV_DIR/bin/activate

# TT-Metal environment variables
export ARCH_NAME=$ARCH_NAME
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

echo -e "${PURPLE}[Reminder:${RESET} ${BLUE}If you switch card types (e.g., wormhole_b0 → blackhole), update ARCH_NAME in python_env/bin/activate${RESET}]"
EOL

# Do not install hooks when this is a worktree
if [ $(git rev-parse --git-dir) = $(git rev-parse --git-common-dir) ]; then
    echo -e "${CYAN}Generating git hooks${RESET}"

    # Ensure pre-commit is installed
    if ! command -v pre-commit &> /dev/null; then
        echo -e "${YELLOW}pre-commit not found. Installing...${RESET}"
        pip install pre-commit
    fi

    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo -e "${YELLOW}In worktree: not generating git hooks${RESET}"
fi

echo -e "${GREEN}Successfully created venv as python_env.${RESET}"

if [ "$is_sourced" = false ]; then
    echo -e "${CYAN}To activate, run:${RESET} source $PYTHON_ENV_DIR/bin/activate"
fi

echo -e "${CYAN}Happy coding!"
