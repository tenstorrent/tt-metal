#!/usr/bin/env bash
# nuke_op.sh - Remove a TTNN operation (and all related variants) from the codebase.
#
# Usage:
#   ./scripts/nuke_op.sh <category> <operation> [--dry-run]
#
# Auto-discovers and nukes ALL related operation directories across the entire
# operations tree. For example, "normalization softmax" will also find and nuke:
#   - moreh/moreh_softmax
#   - moreh/moreh_softmax_backward
#   - transformer/attention_softmax
#
# NOTE: Discovery uses substring matching (e.g., "sum" also matches "cumsum").
# Always run with --dry-run first to review discovered targets.
#
# The <operation> argument is also used as a substring match for test file discovery.
# For ops with naming variants (group_norm vs groupnorm), run the script
# multiple times with each variant. The script is idempotent for directory
# deletion (skips if already gone) and additive for test deletion.
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

# Primary paths (the explicitly requested category/operation)
CPP_OPS_BASE="${REPO_ROOT}/ttnn/cpp/ttnn/operations"
PRIMARY_CPP_OPS_DIR="${CPP_OPS_BASE}/${CATEGORY}"
PRIMARY_OP_DIR="${PRIMARY_CPP_OPS_DIR}/${OPERATION}"
BACKUP_BASE="/tmp/nuked_ops/${CATEGORY}/${OPERATION}"

# ---- Discover all related operation directories ----
# Searches the entire operations tree for directories AND CMake references
# whose name contains the operation name (case-insensitive, underscore-insensitive).
# This catches moreh variants, backward ops, training variants, attention variants, etc.
discover_related_ops() {
    local op="$1"
    # Normalize: remove underscores for matching
    local op_no_underscore
    op_no_underscore=$(echo "$op" | tr -d '_' | tr '[:upper:]' '[:lower:]')

    # Use associative array to deduplicate
    declare -A found_map

    # Method 1: Find all depth-2 directories under the operations tree
    while IFS= read -r dir; do
        local dirname
        dirname=$(basename "$dir")
        local dirname_normalized
        dirname_normalized=$(echo "$dirname" | tr -d '_' | tr '[:upper:]' '[:lower:]')

        if [[ "$dirname_normalized" == *"$op_no_underscore"* ]]; then
            local rel_path="${dir#${CPP_OPS_BASE}/}"
            found_map["$rel_path"]=1
        fi
    done < <(find "${CPP_OPS_BASE}" -mindepth 2 -maxdepth 2 -type d 2>/dev/null)

    # Method 2: Scan all CMakeLists.txt files for references to directories
    # containing the operation name. This catches ops whose directories may
    # have already been partially deleted but still have CMake references.
    while IFS= read -r cmake_file; do
        local cat_dir
        cat_dir=$(dirname "$cmake_file")
        local category_rel="${cat_dir#${CPP_OPS_BASE}/}"

        # Extract directory references from CMake (lines like "op_name/file.cpp")
        while IFS= read -r ref_dir; do
            [[ -z "$ref_dir" ]] && continue
            local ref_normalized
            ref_normalized=$(echo "$ref_dir" | tr -d '_' | tr '[:upper:]' '[:lower:]')
            if [[ "$ref_normalized" == *"$op_no_underscore"* ]]; then
                found_map["${category_rel}/${ref_dir}"]=1
            fi
        done < <(grep -oP '^\s*\K[a-zA-Z0-9_]+(?=/)' "$cmake_file" 2>/dev/null | sort -u)
    done < <(find "${CPP_OPS_BASE}" -name "CMakeLists.txt" -maxdepth 2 2>/dev/null)

    # Method 3: Scan nanobind files for #include references
    while IFS= read -r nanobind_file; do
        local cat_dir
        cat_dir=$(dirname "$nanobind_file")
        local category_rel="${cat_dir#${CPP_OPS_BASE}/}"

        while IFS= read -r ref_dir; do
            [[ -z "$ref_dir" ]] && continue
            local ref_normalized
            ref_normalized=$(echo "$ref_dir" | tr -d '_' | tr '[:upper:]' '[:lower:]')
            if [[ "$ref_normalized" == *"$op_no_underscore"* ]]; then
                found_map["${category_rel}/${ref_dir}"]=1
            fi
        done < <(grep -oP '#include\s+"\K[a-zA-Z0-9_]+(?=/)' "$nanobind_file" 2>/dev/null | sort -u)
    done < <(find "${CPP_OPS_BASE}" -name "*_nanobind.cpp" -maxdepth 2 2>/dev/null)

    printf '%s\n' "${!found_map[@]}" | sort
}

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

# ---- Nuke a single category/operation pair ----
# Deletes the op directory, cleans CMake, nanobind, and Python files for that category.
nuke_single_op() {
    local cat_op="$1"  # e.g. "moreh/moreh_softmax"
    local category="${cat_op%/*}"      # e.g. "moreh"
    local op_name="${cat_op##*/}"      # e.g. "moreh_softmax"

    local cpp_ops_dir="${CPP_OPS_BASE}/${category}"
    local op_dir="${CPP_OPS_BASE}/${cat_op}"
    local category_leaf
    category_leaf="$(basename "${category}")"
    local cmake_file="${cpp_ops_dir}/CMakeLists.txt"
    local nanobind_file="${cpp_ops_dir}/${category_leaf}_nanobind.cpp"
    local python_file="${REPO_ROOT}/ttnn/ttnn/operations/${category_leaf}.py"
    local backup_dir="${BACKUP_BASE}/related/${cat_op}"

    echo "NUKE_RELATED: ${cat_op}"

    # Backup
    mkdir -p "${backup_dir}"
    if [[ -d "${op_dir}" ]]; then
        cp -r "${op_dir}" "${backup_dir}/"
    fi
    [[ -f "${cmake_file}" ]] && cp "${cmake_file}" "${backup_dir}/CMakeLists.txt.bak"
    [[ -f "${nanobind_file}" ]] && cp "${nanobind_file}" "${backup_dir}/$(basename "${nanobind_file}").bak"
    [[ -f "${python_file}" ]] && cp "${python_file}" "${backup_dir}/$(basename "${python_file}").bak"

    # Delete operation directory
    if [[ -d "${op_dir}" ]]; then
        local file_count
        file_count=$(find "${op_dir}" -type f | wc -l)
        rm -rf "${op_dir}"
        echo "NUKE_DELETED_OP: ${op_dir}/ (${file_count} files)"
    else
        echo "NUKE_SKIP: ${op_dir}/ (already deleted)"
    fi

    # Clean CMakeLists.txt
    if [[ -f "${cmake_file}" ]]; then
        local before after removed
        before=$(wc -l < "${cmake_file}")
        sed -i "/${op_name}\//d" "${cmake_file}"
        after=$(wc -l < "${cmake_file}")
        removed=$((before - after))
        echo "NUKE_CMAKE: ${cat_op}: removed ${removed} lines from CMakeLists.txt"
    fi

    # Clean nanobind file
    if [[ -f "${nanobind_file}" ]]; then
        local before after removed
        before=$(wc -l < "${nanobind_file}")
        sed -i "/#include \"${op_name}\//d" "${nanobind_file}"
        sed -i "/#include.*${op_name}/Id" "${nanobind_file}"
        sed -i "/detail::bind.*${op_name}/Id" "${nanobind_file}"
        sed -i "/bind_.*${op_name}/Id" "${nanobind_file}"
        # Also clean namespace references like "moreh_softmax::"
        sed -i "/${op_name}::/d" "${nanobind_file}"
        after=$(wc -l < "${nanobind_file}")
        removed=$((before - after))
        echo "NUKE_NANOBIND: ${cat_op}: removed ${removed} lines from $(basename "${nanobind_file}")"
    fi

    # Clean Python file
    if [[ -f "${python_file}" ]]; then
        local before after removed
        before=$(wc -l < "${python_file}")
        python3 - "${python_file}" "${op_name}" <<'PYEOF'
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
        after=$(wc -l < "${python_file}")
        removed=$((before - after))
        echo "NUKE_PYTHON: ${cat_op}: removed ${removed} lines from $(basename "${python_file}")"
    fi
}

# ---- Collect all targets ----
# Start with the primary target
ALL_TARGETS=()
PRIMARY_REL="${CATEGORY}/${OPERATION}"

# Discover related operations
echo "NUKE_DISCOVERY: Searching for all operations related to '${OPERATION}'..."
while IFS= read -r related; do
    [[ -z "$related" ]] && continue
    ALL_TARGETS+=("$related")
done < <(discover_related_ops "${OPERATION}")

# Ensure the primary target is in the list (it should be, but just in case)
primary_found=false
for t in "${ALL_TARGETS[@]}"; do
    if [[ "$t" == "$PRIMARY_REL" ]]; then
        primary_found=true
        break
    fi
done
if ! $primary_found; then
    ALL_TARGETS=("$PRIMARY_REL" "${ALL_TARGETS[@]}")
fi

echo "NUKE_TARGETS: ${#ALL_TARGETS[@]} operation directories found:"
for t in "${ALL_TARGETS[@]}"; do
    echo "  - ${t}"
done

# ---- Dry run ----
if $DRY_RUN; then
    echo ""
    echo "NUKE_DRY_RUN: Would nuke the following:"
    for target in "${ALL_TARGETS[@]}"; do
        target_dir="${CPP_OPS_BASE}/${target}"
        if [[ -d "${target_dir}" ]]; then
            echo "NUKE_DELETE: ${target_dir}/"
        else
            echo "NUKE_SKIP: ${target_dir}/ (already deleted)"
        fi

        _category="${target%/*}"
        _op_name="${target##*/}"
        _category_leaf="$(basename "${_category}")"
        _cmake="${CPP_OPS_BASE}/${_category}/CMakeLists.txt"
        _nanobind="${CPP_OPS_BASE}/${_category}/${_category_leaf}_nanobind.cpp"
        _pyfile="${REPO_ROOT}/ttnn/ttnn/operations/${_category_leaf}.py"

        [[ -f "${_cmake}" ]] && echo "  NUKE_MODIFY: ${_cmake} (lines matching ${_op_name}/)"
        [[ -f "${_nanobind}" ]] && echo "  NUKE_MODIFY: ${_nanobind}"
        [[ -f "${_pyfile}" ]] && echo "  NUKE_MODIFY: ${_pyfile}"
    done

    echo ""
    echo "NUKE_BACKUP: ${BACKUP_BASE}/"

    echo ""
    echo "NUKE_TEST_FILES:"
    find_test_files "${OPERATION}" | while read -r f; do
        echo "  ${f}"
    done
    exit 0
fi

# ---- Real execution ----
echo ""
echo "NUKE_START: ${CATEGORY}/${OPERATION} (+ ${#ALL_TARGETS[@]} total targets)"

# Track which categories we've already processed for dedup
declare -A processed_categories

# --- Step 1: Nuke each target ---
for target in "${ALL_TARGETS[@]}"; do
    echo ""
    nuke_single_op "$target"
done

# --- Step 2: Delete test files (once, using the operation name as substring) ---
echo ""
echo "NUKE_STEP: tests"
test_count=0
while IFS= read -r test_file; do
    [[ -z "${test_file}" ]] && continue
    # Backup test file preserving relative path
    rel_path="${test_file#${REPO_ROOT}/}"
    backup_test_dir="${BACKUP_BASE}/tests_backup/$(dirname "${rel_path}")"
    mkdir -p "${backup_test_dir}"
    cp "${test_file}" "${backup_test_dir}/"
    rm "${test_file}"
    echo "NUKE_DELETED_TEST: ${rel_path}"
    test_count=$((test_count + 1))
done < <(find_test_files "${OPERATION}")
echo "NUKE_TESTS_TOTAL: ${test_count} test files deleted"

# --- Summary ---
echo ""
echo "NUKE_COMPLETE: ${CATEGORY}/${OPERATION}"
echo "NUKE_TARGETS_NUKED: ${#ALL_TARGETS[@]}"
for t in "${ALL_TARGETS[@]}"; do
    echo "  - ${t}"
done
echo "NUKE_RESTORE: git checkout -- ."
