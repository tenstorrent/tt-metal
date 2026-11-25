#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Offline self-test of hal-llk-gen (no card, no agent, no network): shell syntax, Python byte-compilation,
# unit tests, profiles and pinned prompt checksums.
set -uo pipefail
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TOOL_DIR" || exit 2
fail=0
ok() { echo "  ok    $*"; }
bad() { echo "  FAIL  $*"; fail=1; }

echo "== shell syntax"
for f in hal-llk-gen harness/run_hal_gen.sh stages/*.sh tests/selftest.sh; do
  if bash -n "$f"; then ok "$f"; else bad "$f"; fi
done

echo "== python byte-compilation"
for f in harness/*.py python/*.py python/dyn/*.py tests/*.py; do
  if python3 -c 'import ast, sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$f"; then ok "$f"; else bad "$f"; fi
done

echo "== profiles"
for p in profiles/*.env; do
  missing=$(bash -c 'set -a; source "$1"; for v in PROFILE_DESCRIPTION BASE_COMMIT HAL_COMMIT DOCS_COMMIT PROMPT TESTS MODEL; do [[ -n "${!v:-}" ]] || echo "$v"; done' _ "$p")
  if [[ -n $missing ]]; then bad "$p: missing $(echo $missing)"; continue; fi
  prompt=$(bash -c 'source "$1"; echo "$PROMPT"' _ "$p"); want=$(bash -c 'source "$1"; echo "${PROMPT_MD5:-}"' _ "$p")
  if [[ ! -f harness/prompts/$prompt ]]; then bad "$p: prompt $prompt missing"
  elif [[ -n $want && $(md5sum < "harness/prompts/$prompt" | cut -d' ' -f1) != "$want" ]]; then bad "$p: md5 of $prompt is not $want"
  else ok "$p ($prompt)"; fi
done

echo "== unit tests"
if python3 -m unittest discover -s tests -p 'test_*.py' 2>&1 | tail -n 3 | sed 's/^/  /'; then :; fi
python3 -m unittest discover -s tests -p 'test_*.py' > /dev/null 2>&1 || bad "unit tests"

echo "== CLI"
./hal-llk-gen help > /dev/null && ok "help" || bad "help"
./hal-llk-gen profiles | grep -q run8 && ok "profiles" || bad "profiles"

if (( fail )); then echo "SELFTEST FAILED"; exit 1; fi
echo "SELFTEST OK"
