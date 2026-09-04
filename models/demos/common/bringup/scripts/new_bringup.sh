#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Scaffold a bring-up package: directory tree, the nine log files, and the kit
# helpers copied in. Run from the repo root.
#
#   ./models/demos/common/bringup/scripts/new_bringup.sh models/demos/my_model
set -euo pipefail

PKG="${1:?usage: new_bringup.sh <package path relative to repo root>}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -e "$PKG" ] && { echo "refusing: $PKG already exists"; exit 1; }

SPDX_PY=$'# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.\n#\n# SPDX-License-Identifier: Apache-2.0\n'

mkdir -p "$PKG"/{tt,tests/unit,scripts,configs}
mkdir -p "$PKG"/bringup_log/raw
for d in "" /tt /tests /tests/unit /scripts; do printf '%s' "$SPDX_PY" > "$PKG$d/__init__.py"; done

# The nine log files (recipe section 1.1). Templates exist for five; the rest get a heading.
for t in "$KIT"/templates/*.md; do cp "$t" "$PKG/bringup_log/$(basename "$t")"; done
for f in 01_REFERENCE 02_SURVEY 03_OUTLINE 04_CCL_PLAN; do
  [ -f "$PKG/bringup_log/$f.md" ] || echo "# ${f#*_}

_(pending — filled in its phase)_" > "$PKG/bringup_log/$f.md"
done
mv "$PKG/bringup_log/08_INTEGRATION.md" "$PKG/bringup_log/08_PREFILL_INTEGRATION.md" 2>/dev/null || true

# raw/ logs are evidence: re-include them past a repo-wide *.log ignore
cat > "$PKG/bringup_log/raw/.gitignore" <<'GI'
# The repo root .gitignore has a blanket *.log, which would exclude every raw gate
# log here. These are the EVIDENCE for ../06_GATES.md -- a gate with no raw log did
# not happen -- so they are deliberately tracked.
!*.log
GI

cp "$KIT"/examples/verify_citations.py "$PKG"/scripts/
sed -i "s|models/demos/<your_package>|$PKG|" "$PKG"/scripts/verify_citations.py
echo "note: copy $KIT/examples/noise_floor.py into your test helpers in P1/P5 (keep exactly one definition)"

printf '%s\n' "scaffolded $PKG:"
find "$PKG" -type f | sort | sed 's/^/  /'
printf '\nNext: read %s/BRINGUP_RECIPE.md end to end, then execute P0.\n' "$KIT"
