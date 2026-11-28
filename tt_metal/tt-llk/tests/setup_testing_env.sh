#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# --- Globals ---
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Function to setup pre-commit hooks
setup_precommit() {
    echo "Setting up pre-commit hooks..."

    # Install from the git root so the hook uses tt-metal's .pre-commit-config.yaml (which
    # already covers tt-llk), not tt_metal/tt-llk/.pre-commit-config.yaml.
    local repo_root
    if ! repo_root="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
        echo "WARNING: Not in a git repository, skipping pre-commit setup"
        return
    fi

    # Install pre-commit if not available
    if ! command -v pre-commit >/dev/null 2>&1; then
        echo "Installing pre-commit..."
        pip install pre-commit
    fi

    # Install hooks
    echo "Installing pre-commit hooks..."
    if (cd "$repo_root" && pre-commit install); then
        echo "Pre-commit hooks installed successfully!"
    else
        echo "WARNING: Failed to install pre-commit hooks"
    fi
}

# --- Main Script ---

main() {
    # Setup pre-commit hooks
    setup_precommit

    # The toolchain has its own script: a suite that only needs SFPI, or needs it somewhere else
    # (install_sfpi.sh --dest DIR), runs that one.
    "$SCRIPT_DIR/install_sfpi.sh" "$@" || exit 1

    echo "Setup complete. You can now run your tests."
    echo "Pre-commit hooks have been configured for code quality checks."
}

# Execute the main function
main "$@"
