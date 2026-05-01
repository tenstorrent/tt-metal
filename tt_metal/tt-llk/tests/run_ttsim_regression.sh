#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the LLK pytest suite against the ttsim functional simulator, excluding
# tests marked `quasar`, `nightly`, or `perf`. Each test runs in a forked
# subprocess so that ttsim's `_Exit(1)` on UnimplementedFunctionality (and
# similar) only kills that one test and the suite continues.
#
# Generates:
#   - JUnit XML at python_tests/ttsim_results/ttsim_<timestamp>.xml
#   - Self-contained HTML at python_tests/ttsim_results/ttsim_<timestamp>.html
#   - Symlinks `latest.xml` / `latest.html` pointing at the most recent run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="${SCRIPT_DIR}/python_tests"
RESULTS_DIR="${TESTS_DIR}/ttsim_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
XML_PATH="${RESULTS_DIR}/ttsim_${TIMESTAMP}.xml"
HTML_PATH="${RESULTS_DIR}/ttsim_${TIMESTAMP}.html"
LATEST_XML="${RESULTS_DIR}/latest.xml"
LATEST_HTML="${RESULTS_DIR}/latest.html"

WORKERS="${WORKERS:-10}"
TIMEOUT="${TIMEOUT:-300}"
TEST_PATHS=()
PYTEST_ARGS=()

# ──────────────────────────────────────────────────────────────
# Usage
# ──────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [TEST_PATH...] [-- PYTEST_EXTRA_ARGS...]

Runs the LLK pytest suite on ttsim, excluding tests marked
'quasar', 'nightly', or 'perf'. Per-test process isolation via
pytest-forked converts ttsim _Exit(1) crashes into normal pytest
failures; junit XML + HTML report are produced in:

  ${RESULTS_DIR}

Options:
  -n, --workers N       Number of xdist workers (default: 10; env: WORKERS).
                        Use 0 to disable xdist (serial, --forked only).
  -t, --timeout SEC     Per-test timeout in seconds (default: 300; env: TIMEOUT).
  -h, --help            Show this help message.

Required environment:
  TT_METAL_SIMULATOR    Path to libttsim_<arch>.so

Examples:
  $(basename "$0")
  $(basename "$0") -n 16 test_eltwise_unary_datacopy.py
  $(basename "$0") -- -k Float16_b
  WORKERS=8 $(basename "$0")
EOF
    exit 0
}

# ──────────────────────────────────────────────────────────────
# Parse CLI
# ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--workers) WORKERS="$2"; shift 2 ;;
        -t|--timeout) TIMEOUT="$2"; shift 2 ;;
        -h|--help)    usage ;;
        --)           shift; PYTEST_ARGS+=("$@"); break ;;
        -*)           PYTEST_ARGS+=("$1"); shift ;;
        *)            TEST_PATHS+=("$1"); shift ;;
    esac
done

# ──────────────────────────────────────────────────────────────
# Validate
# ──────────────────────────────────────────────────────────────
if [[ -z "${TT_METAL_SIMULATOR:-}" ]]; then
    echo "ERROR: TT_METAL_SIMULATOR must be set"               >&2
    echo "       e.g. export TT_METAL_SIMULATOR=~/sim/libttsim_bh.so" >&2
    exit 1
fi

if ! command -v pytest &>/dev/null; then
    echo "ERROR: pytest not found in PATH (activate the test venv first)" >&2
    exit 1
fi

if [[ ! -d "$TESTS_DIR" ]]; then
    echo "ERROR: tests directory not found: $TESTS_DIR" >&2
    exit 1
fi

# ttsim does not implement SFPLOADMACRO; default to disabling unless caller set it.
export DISABLE_SFPLOADMACRO="${DISABLE_SFPLOADMACRO:-1}"

mkdir -p "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────
# Build pytest argv
# ──────────────────────────────────────────────────────────────
PYTEST_BASE_ARGS=(
    # Display config: this is the verbose, capture-untouched setup that we
    # confirmed actually preserves the child's captured stdout in the junit
    # XML. -v gives one line per test result, sugar stays off (it confused
    # the failure tally with pytest-forked anyway), and we deliberately do
    # NOT override `-s` from python_tests/pytest.ini or `log_cli=true` —
    # both of those overrides empirically caused pytest-forked's child
    # output to disappear from <system-out>, so the ttsim ERROR lines were
    # missing from the HTML report.
    -v
    -p no:sugar
    --run-simulator
    --timeout="$TIMEOUT"
    --forked
    --show-progress
    -m "not quasar and not nightly and not perf"
    --junit-xml="$XML_PATH"
    # ttsim writes via printf (stdout), so caplog and stderr are always
    # empty for these tests. Capture only stdout to keep the XML and the
    # rendered HTML focused on the actual ttsim ERROR line.
    -o junit_logging=system-out
    -o junit_log_passing_tests=False
    # python_tests/pytest.ini sets log_cli=true, which streams every
    # logging.* record live to the tty. This is independent of stdout
    # capture (ttsim uses printf, not Python logging), so disabling the
    # live stream doesn't affect what lands in <system-out>; it just
    # quiets the terminal during the run.
    -o log_cli=false
)

# `--dist=loadfile` keeps every test in a given file on the same xdist worker,
# which avoids races on per-file ELF compilation. `--max-worker-restart` lets
# xdist re-spawn a worker if one ever dies (in addition to per-test forking).
if [[ "$WORKERS" -gt 0 ]]; then
    PYTEST_BASE_ARGS+=(
        -n "$WORKERS"
        --dist=loadfile
        --max-worker-restart=10000
    )
fi

# ──────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────
echo "============================================================"
echo " ttsim LLK regression (excludes: quasar, nightly, perf)"
echo "============================================================"
echo " Simulator     : ${TT_METAL_SIMULATOR}"
echo " SFPLOADMACRO  : disabled=${DISABLE_SFPLOADMACRO}"
echo " Workers (-n)  : ${WORKERS}"
echo " Per-test fork : on"
echo " Timeout       : ${TIMEOUT}s"
echo " JUnit XML     : ${XML_PATH}"
echo " HTML report   : ${HTML_PATH}"
echo " Test paths    : ${TEST_PATHS[*]:-<all>}"
echo " Extra args    : ${PYTEST_ARGS[*]:-<none>}"
echo "============================================================"
echo ""

# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────
# Build the full pytest argv as one array. Conditional appends keep us safe
# under `set -u` even when TEST_PATHS / PYTEST_ARGS are empty (the older
# `${arr[@]+"${arr[@]}"}` form is fragile across shell versions).
PYTEST_CMD=("pytest" "${PYTEST_BASE_ARGS[@]}")
if [[ ${#TEST_PATHS[@]} -gt 0 ]]; then
    PYTEST_CMD+=("${TEST_PATHS[@]}")
fi
if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    PYTEST_CMD+=("${PYTEST_ARGS[@]}")
fi

pytest_exit=0
(
    cd "$TESTS_DIR"
    "${PYTEST_CMD[@]}"
) || pytest_exit=$?

# ──────────────────────────────────────────────────────────────
# Render HTML (always, so partial/failed runs still produce a report)
# ──────────────────────────────────────────────────────────────
# Prefer the in-tree renderer (render_ttsim_report.py) which produces a
# polished single-page report with pass/fail/skip percentages, a donut
# chart, and a ttsim-error-category breakdown. Fall back to junit2html
# only if the local renderer or its single dep (junitparser) is missing.
RENDERER="${SCRIPT_DIR}/render_ttsim_report.py"
if [[ -f "$XML_PATH" ]]; then
    rendered=0
    if [[ -x "$RENDERER" || -f "$RENDERER" ]] \
       && python3 -c "import junitparser" &>/dev/null; then
        if python3 "$RENDERER" "$XML_PATH" "$HTML_PATH"; then
            rendered=1
        fi
    fi
    if [[ $rendered -eq 0 ]] && command -v junit2html &>/dev/null; then
        echo "Falling back to junit2html (install junitparser for the nicer report)" >&2
        junit2html "$XML_PATH" "$HTML_PATH"
        rendered=1
    fi
    if [[ $rendered -eq 1 ]]; then
        ln -sfn "$(basename "$XML_PATH")"  "$LATEST_XML"
        ln -sfn "$(basename "$HTML_PATH")" "$LATEST_HTML"
    else
        echo ""
        echo "WARNING: no HTML renderer available; only XML produced." >&2
        echo "         Install with:  pip install junitparser"          >&2
    fi
fi

echo ""
echo "============================================================"
if [[ $pytest_exit -eq 0 ]]; then
    echo " ALL TESTS PASSED"
else
    echo " TESTS FAILED (pytest exit code: $pytest_exit)"
fi
echo "------------------------------------------------------------"
echo " XML  : ${XML_PATH}"
[[ -f "$HTML_PATH" ]] && echo " HTML : ${HTML_PATH}"
[[ -L "$LATEST_HTML" ]] && echo " latest: ${LATEST_HTML}"
echo "============================================================"

exit "$pytest_exit"
