#!/usr/bin/env bash
# nuke_op.sh - Remove a TTNN operation from the codebase for agent evaluation.
#
# Usage:
#   ./scripts/nuke_op.sh <category> <operation> [--dry-run]
#
# The <operation> argument is used as a substring match for test file discovery.
# For ops with naming variants (group_norm vs groupnorm), run the script
# multiple times with each variant. The script is idempotent for the op
# directory deletion (skips if already gone) and additive for test deletion.
#
# Example:
#   ./scripts/nuke_op.sh normalization softmax
#   ./scripts/nuke_op.sh normalization groupnorm
#   ./scripts/nuke_op.sh normalization group_norm   # catch the underscore variant
#
# Restore:
#   git checkout -- .

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN=false

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <category> <operation> [--dry-run]"
    echo "  category: normalization, eltwise/unary, reduction, data_movement, etc."
    echo "  operation: softmax, groupnorm, group_norm, concat, etc."
    exit 1
fi

CATEGORY="$1"
OPERATION="$2"
[[ "${3:-}" == "--dry-run" ]] && DRY_RUN=true

# Paths
CPP_OPS_DIR="${REPO_ROOT}/ttnn/cpp/ttnn/operations/${CATEGORY}"
OP_DIR="${CPP_OPS_DIR}/${OPERATION}"
CMAKE_FILE="${CPP_OPS_DIR}/CMakeLists.txt"
CATEGORY_LEAF="$(basename "${CATEGORY}")"
NANOBIND_FILE="${CPP_OPS_DIR}/${CATEGORY_LEAF}_nanobind.cpp"
PYTHON_FILE="${REPO_ROOT}/ttnn/ttnn/operations/${CATEGORY_LEAF}.py"
BACKUP_DIR="/tmp/nuked_ops/${CATEGORY}/${OPERATION}"

# ---- Test discovery ----
# Find all .py test files whose filename contains the operation name (case-insensitive).
# Searches tests/ and models/ directories. Skips YAML configs and sweep framework.
find_test_files() {
    local op="$1"
    local found=()
    while IFS= read -r f; do
        found+=("$f")
    done < <(find "${REPO_ROOT}/tests" "${REPO_ROOT}/models" \
        -type f -iname "*${op}*" \
        -name "*.py" \
        ! -path "*/sweep_framework/*" \
        ! -path "*__pycache__*" \
        2>/dev/null || true)
    printf '%s\n' "${found[@]}"
}

# ---- Dry run ----
if $DRY_RUN; then
    echo "NUKE_DRY_RUN: Would nuke ${CATEGORY}/${OPERATION}"
    if [[ -d "${OP_DIR}" ]]; then
        echo "NUKE_DELETE: ${OP_DIR}/"
    else
        echo "NUKE_SKIP: ${OP_DIR}/ (already deleted)"
    fi
    [[ -f "${CMAKE_FILE}" ]] && echo "NUKE_MODIFY: ${CMAKE_FILE}"
    [[ -f "${NANOBIND_FILE}" ]] && echo "NUKE_MODIFY: ${NANOBIND_FILE}"
    [[ -f "${PYTHON_FILE}" ]] && echo "NUKE_MODIFY: ${PYTHON_FILE}"
    echo "NUKE_BACKUP: ${BACKUP_DIR}/"
    if [[ -f "${CMAKE_FILE}" ]]; then
        count=$(grep -c "${OPERATION}/" "${CMAKE_FILE}" || true)
        echo "NUKE_CMAKE_LINES: ${count}"
    fi
    echo "NUKE_TEST_FILES:"
    find_test_files "${OPERATION}" | while read -r f; do
        echo "  ${f}"
    done
    exit 0
fi

echo "NUKE_START: ${CATEGORY}/${OPERATION}"

# --- Step 1: Backup ---
echo "NUKE_STEP: backup"
mkdir -p "${BACKUP_DIR}"
if [[ -d "${OP_DIR}" ]]; then
    cp -r "${OP_DIR}" "${BACKUP_DIR}/"
fi
[[ -f "${CMAKE_FILE}" ]] && cp "${CMAKE_FILE}" "${BACKUP_DIR}/CMakeLists.txt.bak"
[[ -f "${NANOBIND_FILE}" ]] && cp "${NANOBIND_FILE}" "${BACKUP_DIR}/$(basename "${NANOBIND_FILE}").bak"
[[ -f "${PYTHON_FILE}" ]] && cp "${PYTHON_FILE}" "${BACKUP_DIR}/$(basename "${PYTHON_FILE}").bak"
echo "NUKE_BACKUP: ${BACKUP_DIR}/"

# --- Step 2: Delete operation directory ---
echo "NUKE_STEP: delete_op"
if [[ -d "${OP_DIR}" ]]; then
    file_count=$(find "${OP_DIR}" -type f | wc -l)
    rm -rf "${OP_DIR}"
    echo "NUKE_DELETED_OP: ${OP_DIR}/ (${file_count} files)"
else
    echo "NUKE_SKIP: ${OP_DIR}/ (already deleted, running for name variant?)"
fi

# --- Step 3a: Clean CMakeLists.txt ---
echo "NUKE_STEP: cmake"
if [[ -f "${CMAKE_FILE}" ]]; then
    before=$(wc -l < "${CMAKE_FILE}")
    sed -i "/${OPERATION}\//d" "${CMAKE_FILE}"
    after=$(wc -l < "${CMAKE_FILE}")
    removed=$((before - after))
    echo "NUKE_CMAKE: removed ${removed} lines from CMakeLists.txt"
else
    echo "NUKE_SKIP: CMakeLists.txt not found"
fi

# --- Step 3b: Clean nanobind file ---
echo "NUKE_STEP: nanobind"
if [[ -f "${NANOBIND_FILE}" ]]; then
    before=$(wc -l < "${NANOBIND_FILE}")
    sed -i "/#include \"${OPERATION}\//d" "${NANOBIND_FILE}"
    sed -i "/detail::bind.*${OPERATION}/Id" "${NANOBIND_FILE}"
    sed -i "/bind_.*${OPERATION}/Id" "${NANOBIND_FILE}"
    after=$(wc -l < "${NANOBIND_FILE}")
    removed=$((before - after))
    echo "NUKE_NANOBIND: removed ${removed} lines from $(basename "${NANOBIND_FILE}")"
else
    echo "NUKE_SKIP: nanobind file not found"
fi

# --- Step 3c: Clean Python file ---
echo "NUKE_STEP: python"
if [[ -f "${PYTHON_FILE}" ]]; then
    before=$(wc -l < "${PYTHON_FILE}")
    python3 - "${PYTHON_FILE}" "${OPERATION}" <<'PYEOF'
import sys

filepath = sys.argv[1]
operation = sys.argv[2]

with open(filepath, 'r') as f:
    lines = f.read().split('\n')

result = []
i = 0
while i < len(lines):
    line = lines[i]
    op_match = operation.lower().replace('_', '')

    # Skip attach_golden_function blocks referencing the operation
    if 'attach_golden_function' in line and op_match in line.lower().replace('_', ''):
        while i < len(lines) and lines[i].strip() != ')':
            i += 1
        i += 1  # skip closing paren
        while i < len(lines) and lines[i].strip() == '':
            i += 1
        continue

    # Skip golden/postprocess function defs that feed into an attach for this op
    if line.strip().startswith('def _golden_function') or line.strip().startswith('def _postprocess_golden'):
        lookahead_end = min(i + 30, len(lines))
        found_attach = False
        for j in range(i + 1, lookahead_end):
            if 'attach_golden_function' in lines[j]:
                if op_match in lines[j].lower().replace('_', ''):
                    found_attach = True
                break
        if found_attach:
            i += 1
            while i < len(lines) and (lines[i].startswith('    ') or lines[i].strip() == ''):
                if lines[i].strip() == '' and i + 1 < len(lines) and not lines[i+1].startswith('    '):
                    break
                i += 1
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            continue

    # Skip config aliases referencing the operation
    if '= ttnn._ttnn.operations.' in line and op_match in line.lower().replace('_', ''):
        i += 1
        continue

    # Skip imports referencing the operation
    if ('from ttnn._ttnn.operations' in line or 'import' in line) and op_match in line.lower().replace('_', ''):
        i += 1
        continue

    result.append(line)
    i += 1

# Collapse excessive blank lines
cleaned = []
blank_count = 0
for line in result:
    if line.strip() == '':
        blank_count += 1
        if blank_count <= 2:
            cleaned.append(line)
    else:
        blank_count = 0
        cleaned.append(line)

with open(filepath, 'w') as f:
    f.write('\n'.join(cleaned))
PYEOF
    after=$(wc -l < "${PYTHON_FILE}")
    removed=$((before - after))
    echo "NUKE_PYTHON: removed ${removed} lines from $(basename "${PYTHON_FILE}")"
else
    echo "NUKE_SKIP: Python file not found"
fi

# --- Step 4: Delete test files ---
echo "NUKE_STEP: tests"
test_count=0
while IFS= read -r test_file; do
    [[ -z "${test_file}" ]] && continue
    # Backup test file preserving relative path
    rel_path="${test_file#${REPO_ROOT}/}"
    backup_test_dir="${BACKUP_DIR}/tests_backup/$(dirname "${rel_path}")"
    mkdir -p "${backup_test_dir}"
    cp "${test_file}" "${backup_test_dir}/"
    rm "${test_file}"
    echo "NUKE_DELETED_TEST: ${rel_path}"
    test_count=$((test_count + 1))
done < <(find_test_files "${OPERATION}")
echo "NUKE_TESTS_TOTAL: ${test_count} test files deleted"

# --- Summary ---
echo "NUKE_COMPLETE: ${CATEGORY}/${OPERATION}"
echo "NUKE_RESTORE: git checkout -- ."
