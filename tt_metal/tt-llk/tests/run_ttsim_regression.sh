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
ARCHITECTURE="${TTSIM_ARCHITECTURE:-blackhole}"
TEST_PATHS=()
PYTEST_ARGS=()

# Cache layout for auto-provisioned ttsim artifacts. Version and hashes come
# from the in-tree `ttsim-version` file (same pattern as `sfpi-version`), so
# CI can pin the simulator by bumping one file.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TTSIM_CACHE_ROOT="${TTSIM_CACHE_DIR:-${HOME}/.cache/ttsim}"

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
  -a, --architecture A  ttsim architecture to auto-provision: 'blackhole'
                        or 'wormhole' (default: blackhole;
                        env: TTSIM_ARCHITECTURE). Ignored when
                        TT_METAL_SIMULATOR is already set.
  -h, --help            Show this help message.

Environment:
  TT_METAL_SIMULATOR    Optional. Path to libttsim_<arch>.so. If unset, the
                        script downloads the version pinned in ./ttsim-version
                        into \${TTSIM_CACHE_DIR:-\$HOME/.cache/ttsim} and
                        exports this automatically.
  TTSIM_CACHE_DIR       Override the download cache root (default: ~/.cache/ttsim).

Examples:
  $(basename "$0")                              # blackhole, auto-downloads simulator
  $(basename "$0") -a wormhole                  # wormhole variant
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
        -n|--workers)      WORKERS="$2"; shift 2 ;;
        -t|--timeout)      TIMEOUT="$2"; shift 2 ;;
        -a|--architecture) ARCHITECTURE="$2"; shift 2 ;;
        -h|--help)         usage ;;
        --)           shift; PYTEST_ARGS+=("$@"); break ;;
        -*)           PYTEST_ARGS+=("$1"); shift ;;
        *)            TEST_PATHS+=("$1"); shift ;;
    esac
done

# ──────────────────────────────────────────────────────────────
# Auto-provision ttsim (libttsim_<arch>.so + soc_descriptor.yaml)
# ──────────────────────────────────────────────────────────────
# If the caller pre-exports TT_METAL_SIMULATOR we trust it and do nothing;
# otherwise we materialize a cache dir matching ttsim's layout requirement
# (soc_descriptor.yaml must sit next to the .so — ttsim derives its path
# from the .so path). This lets CI call the script as a one-liner.
provision_ttsim() {
    local architecture="$1"
    local so_name soc_src hash_var
    # Upstream ttsim releases ship the .so with short suffixes (libttsim_bh.so
    # / libttsim_wh.so) so we map full architecture names → upstream suffix.
    case "$architecture" in
        blackhole|bh)
            architecture=blackhole
            so_name=libttsim_bh.so
            soc_src="${REPO_ROOT}/tt_metal/soc_descriptors/blackhole_140_arch.yaml"
            hash_var=ttsim_bh_so_hash
            ;;
        wormhole|wormhole_b0|wh)
            architecture=wormhole
            so_name=libttsim_wh.so
            soc_src="${REPO_ROOT}/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"
            hash_var=ttsim_wh_so_hash
            ;;
        *)
            echo "ERROR: unknown --architecture '$architecture' (expected 'blackhole' or 'wormhole')" >&2
            exit 1
            ;;
    esac

    local version_file="${SCRIPT_DIR}/ttsim-version"
    if [[ ! -f "$version_file" ]]; then
        echo "ERROR: missing pin file: $version_file" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$version_file"

    local cache_dir="${TTSIM_CACHE_ROOT}/${ttsim_version}/${architecture}"
    local so_path="${cache_dir}/${so_name}"
    local soc_path="${cache_dir}/soc_descriptor.yaml"
    local url="${ttsim_repo}/releases/download/${ttsim_tag}/${so_name}"
    local expected_hash="${!hash_var}"

    mkdir -p "$cache_dir"

    local need_download=1
    if [[ -f "$so_path" ]]; then
        local got
        got=$(${ttsim_hashtype}sum "$so_path" | awk '{print $1}')
        if [[ "$got" == "$expected_hash" ]]; then
            need_download=0
        else
            echo "Cached ${so_name} ${ttsim_hashtype} mismatch (got=$got expected=$expected_hash); re-downloading" >&2
        fi
    fi
    if [[ $need_download -eq 1 ]]; then
        echo "Downloading ${url}"
        local tmp="${so_path}.tmp.$$"
        if ! curl -fSL --retry 5 --retry-delay 2 -o "$tmp" "$url"; then
            rm -f "$tmp"
            echo "ERROR: failed to download $url" >&2
            exit 1
        fi
        local got
        got=$(${ttsim_hashtype}sum "$tmp" | awk '{print $1}')
        if [[ "$got" != "$expected_hash" ]]; then
            rm -f "$tmp"
            echo "ERROR: ${ttsim_hashtype} mismatch for ${so_name} (got=$got expected=$expected_hash)" >&2
            exit 1
        fi
        mv "$tmp" "$so_path"
    fi

    if [[ ! -f "$soc_src" ]]; then
        echo "ERROR: soc descriptor source not found: $soc_src" >&2
        exit 1
    fi
    cp -f "$soc_src" "$soc_path"

    export TT_METAL_SIMULATOR="$so_path"
}

if [[ -z "${TT_METAL_SIMULATOR:-}" ]]; then
    provision_ttsim "$ARCHITECTURE"
fi

# ──────────────────────────────────────────────────────────────
# Validate
# ──────────────────────────────────────────────────────────────
if [[ ! -f "$TT_METAL_SIMULATOR" ]]; then
    echo "ERROR: TT_METAL_SIMULATOR points to a missing file: $TT_METAL_SIMULATOR" >&2
    exit 1
fi
if [[ ! -f "$(dirname "$TT_METAL_SIMULATOR")/soc_descriptor.yaml" ]]; then
    echo "ERROR: soc_descriptor.yaml is missing next to $TT_METAL_SIMULATOR" >&2
    echo "       (ttsim derives the SoC descriptor path from the .so path)" >&2
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
echo " Architecture   : ${ARCHITECTURE}"
echo " Simulator      : ${TT_METAL_SIMULATOR}"
echo " SoC descriptor : $(dirname "$TT_METAL_SIMULATOR")/soc_descriptor.yaml"
echo " SFPLOADMACRO   : disabled=${DISABLE_SFPLOADMACRO}"
echo " Workers (-n)   : ${WORKERS}"
echo " Per-test fork  : on"
echo " Timeout        : ${TIMEOUT}s"
echo " JUnit XML      : ${XML_PATH}"
echo " HTML report    : ${HTML_PATH}"
echo " Test paths     : ${TEST_PATHS[*]:-<all>}"
echo " Extra args     : ${PYTEST_ARGS[*]:-<none>}"
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
