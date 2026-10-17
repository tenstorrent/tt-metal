#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run-pytest-group.sh --spec CMD [--spec CMD ...] [-- EXTRA_FLAGS...]

Run several pytest invocations as one CI command so that flags a pipeline
appends to the cmd string reach EVERY invocation, not only the last one in
an &&-chain.

Each CMD is a full shell command; it may set environment variables inline,
e.g. 'TT_METAL_ALLOCATOR_MODE_HYBRID=1 pytest --timeout 300 tests/... -xv'.
Invocations run in order and stop at the first failure, matching && semantics.

Flags after "--" are appended to every invocation. Pipelines inject flags
such as --deselect, -p no:timeout, or -n 4 by appending them to the cmd
string, which places them after this script's "--".

Examples:
  run-pytest-group.sh \
    --spec 'pytest --timeout 300 tests/ttnn/unit_tests/base_functionality -xv -m "not disable_fast_runtime_mode"' \
    --spec 'TT_METAL_ALLOCATOR_MODE_HYBRID=1 pytest --timeout 300 tests/ttnn/unit_tests/per_core_allocation -xv' \
    -- --deselect tests/ttnn/unit_tests/tensor/test_x.py
EOF
}

main() {
    local specs=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            --spec)
                if [[ $# -lt 2 ]]; then
                    echo "run-pytest-group.sh: --spec requires an argument" >&2
                    exit 2
                fi
                specs+=("$2")
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "run-pytest-group.sh: unexpected argument '$1' (use --spec CMD [--spec CMD ...] [-- EXTRA_FLAGS])" >&2
                exit 2
                ;;
        esac
    done
    if [[ ${#specs[@]} -eq 0 ]]; then
        echo "run-pytest-group.sh: at least one --spec is required" >&2
        exit 2
    fi
    # Flags appended by the pipeline to the cmd string land here and are
    # forwarded to every pytest invocation.
    local extra="$*"
    local spec
    for spec in "${specs[@]}"; do
        bash --noprofile --norc -euo pipefail -c "$spec $extra"
    done
}

main "$@"
