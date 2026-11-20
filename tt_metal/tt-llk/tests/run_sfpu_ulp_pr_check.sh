#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# The SFPU ULP PR check, one card leg per architecture (tests/pipeline_reorg/llk_pr_gate_tests.yaml).
# Fails a change that makes a cell of the unary ULP sweep (test_unary_sfpu_ulp.py) less accurate on
# this card, or that loosens a budget in python_tests/helpers/sfpu_accuracy_budget_*.yaml.
#   Budgets  a loosened budget fails unless the PR changes only the tables (and the acknowledgement
#            lists in test_sfpu_accuracy_budget.py) and has the `ulp-budget-raise-approved` label
#            (helpers/ulp_budget_diff.py guard).
#   Select   builds every sweep cell at HEAD^1 and at HEAD and picks those whose loaded ELF bytes or
#            table row differ (helpers/ulp_affected.py). A change to the check picks every cell.
#   Sweep    runs the picked cells on the card; one over its row's figure fails.
# Does nothing unless the change touches this architecture's SFPU kernels, a table or the check: a
# change to the harness, the golden or SFPI alone is the nightly sweep's (llk_e2e_tests.yaml).
#
# Environment, all set by llk-smoke-impl.yaml:
#   CHIP_ARCH          wormhole or blackhole: the kernels to build and the table to judge by.
#   GITHUB_EVENT_NAME  only pull_request and merge_group runs are checked; HEAD is then the merge
#                      the PR proposes (or the queue's commit) and HEAD^1 what it lands on.
#   GITHUB_REF, GITHUB_REPOSITORY, GITHUB_API_URL, GH_TOKEN  to read the PR's labels.
#   GITHUB_STEP_SUMMARY  where the reports go.
# Arguments are passed to the sweep's pytest (--junitxml). Locally, with no GITHUB_ACTIONS, it
# checks HEAD against HEAD^1 on the attached card.

set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$TESTS_DIR/../../.." && pwd)"
TESTS=tt_metal/tt-llk/tests/python_tests
TABLE_GLOB="$TESTS/helpers/sfpu_accuracy_budget_*.yaml"
# The check itself, with the code that picks the cells and the sweep that records them.
CHECK_PATHS=(
    tt_metal/tt-llk/tests/run_sfpu_ulp_pr_check.sh
    "$TESTS/helpers/ulp_affected.py"
    "$TESTS/test_unary_sfpu_ulp.py"
)
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

# Builds every sweep cell of $CHIP_ARCH at the revision checked out, into $1/tt-llk-build.
build() {
    local out="$1"
    shift
    (cd "$TESTS_DIR" && ./setup_testing_env.sh) \
        && (cd "$TESTS_DIR/python_tests" && RUNNER_TEMP="$out" pytest -q -n auto -p no:cacheprovider \
            --compile-producer --no-debug-symbols "$@" test_unary_sfpu_ulp.py)
}

main() {
    if [[ -n "${GITHUB_ACTIONS:-}" && "${GITHUB_EVENT_NAME:-}" != pull_request && "${GITHUB_EVENT_NAME:-}" != merge_group ]]; then
        echo "Not a pull request or a merge queue run: nothing to check."
        exit 0
    fi
    local short
    case "${CHIP_ARCH:-}" in
        wormhole) short=wh arch_dir=wormhole_b0 ;;
        blackhole) short=bh arch_dir=blackhole ;;
        *)
            echo "::error::No SFPU accuracy table for CHIP_ARCH='${CHIP_ARCH:-}'."
            exit 1
            ;;
    esac
    # The SFPU kernels: this architecture's files named *sfpu*, or under a directory so named, in
    # tt-llk and in the metal LLK API.
    local kernel_paths=(
        ":(glob)tt_metal/tt-llk/tt_llk_$arch_dir/**/*sfpu*" ":(glob)tt_metal/tt-llk/tt_llk_$arch_dir/**/*sfpu*/**"
        ":(glob)tt_metal/hw/ckernels/$arch_dir/**/*sfpu*" ":(glob)tt_metal/hw/ckernels/$arch_dir/**/*sfpu*/**"
    )
    # The pytest steps of the pipeline's other legs; not this script's.
    unset PYTEST_ADDOPTS
    # The container's uid does not own the checkout; git refuses it without this.
    local n="${GIT_CONFIG_COUNT:-0}"
    export GIT_CONFIG_COUNT=$((n + 1)) "GIT_CONFIG_KEY_$n=safe.directory" "GIT_CONFIG_VALUE_$n=$REPO"
    cd "$REPO"
    local head_sha
    head_sha="$(git rev-parse HEAD)"
    # The pipeline checks out one commit.
    git rev-parse -q --verify 'HEAD^1^{commit}' > /dev/null || git fetch -q --depth=2 origin "$head_sha"

    local check=false tables=false kernels=false
    local table="sfpu_accuracy_budget_$short.yaml" own_table=false
    git diff --quiet HEAD^1 HEAD -- "${CHECK_PATHS[@]}" || check=true
    git diff --quiet HEAD^1 HEAD -- "$TABLE_GLOB" || tables=true
    git diff --quiet HEAD^1 HEAD -- "$TESTS/helpers/$table" || own_table=true
    git diff --quiet HEAD^1 HEAD -- "${kernel_paths[@]}" || kernels=true
    if [[ $tables == true ]]; then
        (cd "$TESTS_DIR/python_tests" && python3 -m helpers.ulp_budget_diff guard --base-rev 'HEAD^1' --out "$SUMMARY")
    fi
    if [[ $check == false && $kernels == false && $own_table == false ]]; then
        echo "No $CHIP_ARCH SFPU kernel, $CHIP_ARCH budget table or check change: nothing to sweep."
        exit 0
    fi

    local ulp="${RUNNER_TEMP:-/tmp}/sfpu-ulp-pr-check"
    rm -rf "$ulp"
    mkdir -p "$ulp"
    git show "HEAD^1:$TESTS/helpers/$table" > "$ulp/$table" 2> /dev/null || : > "$ulp/$table"

    # Both revisions build in this checkout. A cell the parent does not build reads as new and
    # runs, so a failure there only costs card time.
    trap 'git -C "$REPO" checkout -q --detach "$head_sha"' EXIT
    git checkout -q --detach HEAD^1
    build "$ulp/base" || echo "::warning::HEAD^1 did not build every $CHIP_ARCH sweep cell; those run."
    git checkout -q --detach "$head_sha"
    trap - EXIT
    build "$ulp/head" --ulp-variants="$ulp/records.jsonl"

    cd "$TESTS_DIR/python_tests"
    local all=()
    [[ $check == true ]] && all=(--all)
    python3 -m helpers.ulp_affected --records "$ulp/records.jsonl" \
        --base "$ulp/base/tt-llk-build" --head "$ulp/head/tt-llk-build" \
        --base-table "$ulp/$table" --nodes "$ulp/nodes.txt" \
        "${all[@]}" | tee -a "$SUMMARY"
    if [[ ! -s "$ulp/nodes.txt" ]]; then
        exit 0
    fi

    local status=0
    RUNNER_TEMP="$ulp/head" pytest -q -n auto --dist=worksteal --timeout=120 \
        --ulp-measure=ulp_measurements.jsonl --compile-consumer "$@" "@$ulp/nodes.txt" || status=$?
    python3 -m helpers.ulp_budget_diff headroom --measured ulp_measurements.jsonl --out "$SUMMARY" || status=$?
    exit "$status"
}

# Bash reads a script as it runs it, and the base build checks out another revision of this file:
# everything above is parsed before main starts.
main "$@"
exit
