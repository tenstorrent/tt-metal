#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Install LLK CodeGen agents to Claude agents directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
AGENTS_SRC="$SCRIPT_DIR/../agents"
CLAUDE_AGENTS_DIR="${CLAUDE_AGENTS_DIR:-$REPO_ROOT/.claude/agents}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  LLK CodeGen Agents Installer"
echo "=========================================="
echo ""

# Check if source agents directory exists
if [ ! -d "$AGENTS_SRC" ]; then
    echo -e "${RED}Error: Agents source directory not found: $AGENTS_SRC${NC}"
    exit 1
fi

# Create Claude agents directory if it doesn't exist
if [ ! -d "$CLAUDE_AGENTS_DIR" ]; then
    echo -e "${YELLOW}Creating Claude agents directory: $CLAUDE_AGENTS_DIR${NC}"
    mkdir -p "$CLAUDE_AGENTS_DIR"
fi

echo "Source: $AGENTS_SRC"
echo "Destination: $CLAUDE_AGENTS_DIR"
echo ""

# List of agents to install
AGENTS=(
    "llk-analyzer.md"
    "llk-arch-lookup.md"
    "llk-debugger.md"
    "llk-kernel-writer.md"
    "llk-planner.md"
    "llk-prettifier.md"
    "llk-phase-tester.md"
    "llk-regression-tester.md"
)

# Install each agent
installed=0
skipped=0
for agent in "${AGENTS[@]}"; do
    src="$AGENTS_SRC/$agent"
    dest="$CLAUDE_AGENTS_DIR/$agent"

    if [ ! -f "$src" ]; then
        echo -e "${YELLOW}Warning: Agent not found: $agent${NC}"
        skipped=$((skipped + 1))
        continue
    fi

    # Check if destination exists and is different
    if [ -f "$dest" ]; then
        if cmp -s "$src" "$dest"; then
            echo -e "${YELLOW}[SKIP]${NC} $agent (already up to date)"
            skipped=$((skipped + 1))
            continue
        else
            echo -e "${GREEN}[UPDATE]${NC} $agent"
        fi
    else
        echo -e "${GREEN}[INSTALL]${NC} $agent"
    fi

    cp "$src" "$dest"
    installed=$((installed + 1))
done

echo ""
echo "=========================================="
echo "  Installation Complete"
echo "=========================================="
echo -e "  Installed: ${GREEN}$installed${NC}"
echo -e "  Skipped:   ${YELLOW}$skipped${NC}"
echo ""
echo "Agents are now available in Claude:"
echo "  - llk-analyzer"
echo "  - llk-arch-lookup"
echo "  - llk-debugger"
echo "  - llk-kernel-writer"
echo "  - llk-planner"
echo "  - llk-prettifier"
echo "  - llk-phase-tester"
echo "  - llk-regression-tester"
echo ""
echo "To use, start Claude from the codegen directory:"
echo "  cd codegen"
echo "  claude"
echo "  > Generate gelu for Quasar"
echo ""
