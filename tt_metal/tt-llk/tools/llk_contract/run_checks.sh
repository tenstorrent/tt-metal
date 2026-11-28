#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Run every check of the LLK contract against a tt-metal checkout, and regenerate docs/.
#   ./run_checks.sh                 the tt-metal checkout this folder is in (tt_metal/tt-llk/tools/llk_contract)
#   ./run_checks.sh ~/tt-metal      another checkout
# Needs: pip install pyyaml jsonschema z3-solver
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd "${1:-../../../..}" && pwd)"
[ -d "$REPO/tt_metal/tt-llk/tt_llk_blackhole" ] || { echo "not a tt-metal checkout: $REPO" >&2; exit 2; }
C=contract/llk_contract.yaml
API_MAP="$(mktemp -d)/api_map.json"

echo "== validate";       python3 tools/contract_tool.py validate   "$C"
echo "== check-code";     python3 tools/contract_tool.py check-code "$C" "$REPO"
echo "== check-api (reads the whole compute API: about a minute)"
                          python3 tools/compute_api_map.py "$REPO" --all --json "$API_MAP" > /dev/null
                          python3 tools/contract_tool.py check-api  "$C" "$API_MAP"
echo "== trace";          python3 tools/contract_tool.py trace      "$C" "$REPO" --md docs/requirements_trace.md | tail -1
                          python3 tools/contract_tool.py trace      "$C" "$REPO" --op tilize,reconfig_srca --md docs/tilize_contract.md > /dev/null
echo "== prove";          python3 tools/contract_z3.py prove        "$C" | tail -1
echo "== sequences"
for f in examples/sequences/*.yaml; do
  printf '  %-55s %s | %s\n' "$(basename "$f")" \
    "$(python3 tools/contract_tool.py check-sequence "$C" "$f" | tail -1 | sed 's/^check-sequence: //')" \
    "$(python3 tools/contract_z3.py prove-sequence "$C" "$f" | tail -1 | sed 's/^prove-sequence: //')"
done
