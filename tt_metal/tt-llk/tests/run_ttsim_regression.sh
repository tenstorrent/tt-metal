#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the LLK pytest suite against the ttsim functional simulator. For
# Wormhole/Blackhole this excludes Quasar-only tests; for Quasar it selects
# Quasar-only tests. Each test runs in a forked subprocess so that ttsim's
# `_Exit(1)` on UnimplementedFunctionality (and similar) only kills that one
# test and the suite continues. LLK node ids in
# tests/pipeline_reorg/ttsim-skip-list.yaml are deselected for the architecture
# under test, on both the collection pass and the real run. The weekly workflow
# passes --no-ignore; that list is for callers that must stay green.
#
# Generates:
#   - JUnit XML at python_tests/ttsim_results/ttsim_<timestamp>.xml
#   - Self-contained HTML at python_tests/ttsim_results/ttsim_<timestamp>.html,
#     unless --no-html is set (the Quasar PR-gate shards use that)
#   - Symlinks `latest.xml` / `latest.html` pointing at the most recent run.
#     latest.html is omitted when HTML is disabled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="${SCRIPT_DIR}/python_tests"
RESULTS_DIR="${TESTS_DIR}/ttsim_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
XML_PATH="${RESULTS_DIR}/ttsim_${TIMESTAMP}.xml"
HTML_PATH="${RESULTS_DIR}/ttsim_${TIMESTAMP}.html"
COLLECTED_PATH="${RESULTS_DIR}/ttsim_${TIMESTAMP}.collected.txt"
LATEST_XML="${RESULTS_DIR}/latest.xml"
LATEST_HTML="${RESULTS_DIR}/latest.html"
LATEST_COLLECTED="${RESULTS_DIR}/latest.collected.txt"

WORKERS="${WORKERS:-10}"
TIMEOUT="${TIMEOUT:-300}"
ARCHITECTURE="${TTSIM_ARCHITECTURE:-blackhole}"
USE_IGNORE=1
LIST_IGNORE=0
WRITE_HTML=1
TEST_PATHS=()
PYTEST_ARGS=()
DESELECT_ARGS=()
IGNORE_PREFIXES=()
# pytest's own status when the selection matches nothing.
PYTEST_NO_TESTS_COLLECTED=5

# Cache layout for auto-provisioned ttsim artifacts. The release tag comes from
# tt_metal/ttsim-version, the same pin every tt-metal sim pipeline uses, so
# bumping that one file moves this script too.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# Same file the ttnn sim legs deselect from. LLK entries live in its per-arch
# sections; this script keeps the ones under python_tests/.
IGNORE_FILE="${TTSIM_IGNORE_FILE:-${REPO_ROOT}/tests/pipeline_reorg/ttsim-skip-list.yaml}"
TTSIM_VERSION_FILE="${REPO_ROOT}/tt_metal/ttsim-version"
TTSIM_REPO="tenstorrent/ttsim"
TTSIM_CACHE_ROOT="${TTSIM_CACHE_DIR:-${HOME}/.cache/ttsim}"

# ──────────────────────────────────────────────────────────────
# Usage
# ──────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [TEST_PATH...] [-- PYTEST_EXTRA_ARGS...]

Runs the LLK pytest suite on ttsim. Wormhole/Blackhole runs exclude
Quasar-only tests; Quasar runs select Quasar-only tests. Per-test
process isolation via pytest-forked converts ttsim _Exit(1) crashes into normal pytest
failures; junit XML is produced in the directory below. HTML is produced too
unless --no-html is set:

  ${RESULTS_DIR}

Options:
  -n, --workers N       Number of xdist workers (default: 10; env: WORKERS).
                        Use 0 to disable xdist (serial, --forked only).
  -t, --timeout SEC     Per-test timeout in seconds (default: 300; env: TIMEOUT).
  -a, --architecture A  ttsim architecture: 'blackhole', 'wormhole', or
                        'quasar' (default: blackhole; env: TTSIM_ARCHITECTURE).
                        Controls test selection and collection.
  --ignore-file PATH    ttsim skip list (default: tests/pipeline_reorg/ttsim-skip-list.yaml;
                        env: TTSIM_IGNORE_FILE). LLK entries are repo-root paths
                        under tt_metal/tt-llk/tests/python_tests/. The matching
                        architecture section is wormhole_b0, blackhole, or quasar.
                        file.py::test_name also skips test_name_extra; use
                        file.py::test_name[ to skip one parametrized test.
  --no-ignore           Run every collected test; do not read the skip list.
  --no-html             Do not render the HTML report. JUnit XML is still written.
  --list-ignore         Print the prefixes selected for --architecture and exit.
                        Does not download ttsim or run pytest.
  -h, --help            Show this help message.

Environment:
  TT_METAL_SIMULATOR    Optional. Path to libttsim_<arch>.so. If unset, the
                        script downloads the release pinned in
                        tt_metal/ttsim-version into
                        \${TTSIM_CACHE_DIR:-\$HOME/.cache/ttsim} and exports
                        this automatically.
  TTSIM_CACHE_DIR       Override the download cache root (default: ~/.cache/ttsim).
  TTSIM_IGNORE_FILE     Override the default ttsim skip-list path.
  GH_TOKEN              Optional. Sent with the GitHub API request for the
                        release digest, to avoid the unauthenticated rate limit.

Examples:
  $(basename "$0")                              # blackhole, auto-downloads simulator
  $(basename "$0") -a wormhole                  # wormhole variant
  TT_METAL_SIMULATOR=~/sim/libttsim.so $(basename "$0") -a quasar
  $(basename "$0") -n 16 test_eltwise_unary_datacopy.py
  $(basename "$0") -- -k Float16_b
  $(basename "$0") -a quasar --no-ignore
  $(basename "$0") -a quasar --list-ignore
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
        --ignore-file)
            if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == -* ]]; then
                echo "ERROR: --ignore-file requires a path" >&2
                exit 1
            fi
            IGNORE_FILE="$2"
            shift 2
            ;;
        --no-ignore)       USE_IGNORE=0; shift ;;
        --no-html)         WRITE_HTML=0; shift ;;
        --list-ignore)     LIST_IGNORE=1; shift ;;
        -h|--help)         usage ;;
        --)           shift; PYTEST_ARGS+=("$@"); break ;;
        -*)           PYTEST_ARGS+=("$1"); shift ;;
        *)            TEST_PATHS+=("$1"); shift ;;
    esac
done


# Normalize architecture aliases once so provisioning, collection, marker
# selection, and reporting all agree.
case "$ARCHITECTURE" in
    blackhole|bh)                 ARCHITECTURE=blackhole ;;
    wormhole|wormhole_b0|wh)      ARCHITECTURE=wormhole ;;
    quasar|qsr)                   ARCHITECTURE=quasar ;;
    *)
        echo "ERROR: unknown --architecture '$ARCHITECTURE' (expected 'blackhole', 'wormhole', or 'quasar')" >&2
        exit 1
        ;;
esac

# Marker expression shared by the collection pass and the real run so both
# apply the same architecture, nightly, performance, and accuracy filtering.
case "$ARCHITECTURE" in
    quasar) MARKER_EXPR="quasar and not nightly and not perf and not accuracy" ;;
    *)      MARKER_EXPR="not quasar and not nightly and not perf and not accuracy" ;;
esac

# Quasar tests live under python_tests/quasar. Restrict the default path so
# xdist workers do not collect the full WH/BH LLK suite just to marker-deselect
# it. Explicit test paths from the caller are still honored as-is.
if [[ "$ARCHITECTURE" == "quasar" && ${#TEST_PATHS[@]} -eq 0 ]]; then
    TEST_PATHS=(quasar)
fi

# ──────────────────────────────────────────────────────────────
# Skip list
# ──────────────────────────────────────────────────────────────
# Entries in the shared skip list are repo-root pytest node ids. This suite
# collects with python_tests/ as rootdir, so only entries under that tree are
# kept and the directory prefix is removed. `--deselect` uses
# nodeid.startswith(), so a file path skips that file, `file.py::test_name`
# also skips `test_name_extra`, and `file.py::test_name[` skips only the
# parametrizations of that test. The same prefixes go to the collection pass
# and the real run; otherwise a deselected test would still be in the
# collected denominator and the HTML report would count it as crashed.
load_ttsim_skip_list() {
    local file="$1"
    local arch="$2"
    local yaml_arch prefix parsed node_id

    case "$arch" in
        wormhole)  yaml_arch=wormhole_b0 ;;
        blackhole|quasar) yaml_arch="$arch" ;;
        *)
            echo "ERROR: no ttsim skip-list section for architecture '$arch'" >&2
            exit 1
            ;;
    esac

    if [[ ! -f "$file" ]]; then
        echo "ERROR: ttsim skip list not found: $file" >&2
        exit 1
    fi

    prefix="tt_metal/tt-llk/tests/python_tests/"
    if ! parsed="$(
        python3 - "$file" "$yaml_arch" "$prefix" <<'PY'
import sys

path, arch, prefix = sys.argv[1:]
current = None
with open(path, encoding="utf-8") as handle:
    for raw in handle:
        line = raw.split("#", 1)[0].rstrip("\r\n")
        if not line.strip():
            continue
        stripped = line.strip()
        if not line[:1].isspace() and stripped.endswith(":"):
            current = stripped[:-1].strip().strip("\"'")
            continue
        if not stripped.startswith("- "):
            continue
        item = stripped[2:].strip().strip("\"'")
        if current is None:
            sys.stderr.write(f"ERROR: skip entry before any architecture key in {path}: {item}\n")
            sys.exit(1)
        if current == arch and item.startswith(prefix):
            sys.stdout.write(item[len(prefix) :] + "\n")
PY
    )"; then
        echo "ERROR: failed to read ttsim skip list: $file" >&2
        exit 1
    fi

    while IFS= read -r node_id; do
        [[ -z "$node_id" ]] && continue
        IGNORE_PREFIXES+=("$node_id")
        DESELECT_ARGS+=(--deselect "$node_id")
    done <<<"$parsed"
}

if [[ "$USE_IGNORE" -eq 1 ]]; then
    load_ttsim_skip_list "$IGNORE_FILE" "$ARCHITECTURE"
fi

if [[ "$LIST_IGNORE" -eq 1 ]]; then
    if [[ ${#IGNORE_PREFIXES[@]} -gt 0 ]]; then
        printf '%s\n' "${IGNORE_PREFIXES[@]}"
    fi
    exit 0
fi

# ──────────────────────────────────────────────────────────────
# Auto-provision ttsim (libttsim_<arch>.so + soc_descriptor.yaml)
# ──────────────────────────────────────────────────────────────
# If the caller pre-exports TT_METAL_SIMULATOR we trust it and do nothing;
# otherwise we materialize a cache dir matching ttsim's layout requirement
# (soc_descriptor.yaml must sit next to the .so — ttsim derives its path
# from the .so path). This lets CI call the script as a one-liner.
provision_ttsim() {
    local architecture="$1"
    local so_name soc_src
    # Upstream ttsim releases ship the .so with short suffixes (libttsim_bh.so
    # / libttsim_wh.so) so we map full architecture names → upstream suffix.
    case "$architecture" in
        blackhole|bh)
            architecture=blackhole
            so_name=libttsim_bh.so
            soc_src="${REPO_ROOT}/tt_metal/soc_descriptors/blackhole_140_arch.yaml"
            ;;
        wormhole|wormhole_b0|wh)
            architecture=wormhole
            so_name=libttsim_wh.so
            soc_src="${REPO_ROOT}/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"
            ;;
        quasar)
            architecture=quasar
            so_name=libttsim_qsr.so
            soc_src="${REPO_ROOT}/tt_metal/soc_descriptors/quasar_32_arch_ttsim.yaml"
            ;;
        *)
            echo "ERROR: unknown --architecture '$architecture' (expected 'blackhole', 'wormhole', or 'quasar')" >&2
            exit 1
            ;;
    esac

    if [[ ! -f "$TTSIM_VERSION_FILE" ]]; then
        echo "ERROR: missing pin file: $TTSIM_VERSION_FILE" >&2
        exit 1
    fi
    local tag
    tag="$(tr -d '[:space:]' < "$TTSIM_VERSION_FILE")"
    if [[ -z "$tag" ]]; then
        echo "ERROR: $TTSIM_VERSION_FILE is empty; expected a ${TTSIM_REPO} release tag" >&2
        exit 1
    fi

    local cache_dir="${TTSIM_CACHE_ROOT}/${tag}/${architecture}"
    local so_path="${cache_dir}/${so_name}"
    local soc_path="${cache_dir}/soc_descriptor.yaml"
    local url="https://github.com/${TTSIM_REPO}/releases/download/${tag}/${so_name}"

    mkdir -p "$cache_dir"

    # A cached .so is only ever moved into place after it passed the digest
    # check below, so its presence under the tag's directory is enough.
    if [[ ! -f "$so_path" ]]; then
        # GitHub records a sha256 digest for every release asset; checking
        # against it replaces the per-arch hashes the pin file used to carry.
        local expected_hash=""
        local api_url="https://api.github.com/repos/${TTSIM_REPO}/releases/tags/${tag}"
        local auth=()
        [[ -n "${GH_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer ${GH_TOKEN}")
        local release_json
        if release_json=$(curl -fsSL --retry 5 --retry-delay 2 "${auth[@]}" "$api_url"); then
            expected_hash=$(SO_NAME="$so_name" python3 -c '
import json, os, sys
for asset in json.load(sys.stdin).get("assets", []):
    if asset.get("name") == os.environ["SO_NAME"]:
        print((asset.get("digest") or "").removeprefix("sha256:"))
' <<<"$release_json")
        fi
        if [[ -z "$expected_hash" ]]; then
            echo "ERROR: no sha256 digest for ${so_name} from $api_url; refusing an unverified download" >&2
            exit 1
        fi

        echo "Downloading ${url}"
        local tmp="${so_path}.tmp.$$"
        if ! curl -fSL --retry 5 --retry-delay 2 -o "$tmp" "$url"; then
            rm -f "$tmp"
            echo "ERROR: failed to download $url" >&2
            exit 1
        fi
        if [[ -n "$expected_hash" ]]; then
            local got
            got=$(sha256sum "$tmp" | awk '{print $1}')
            if [[ "$got" != "$expected_hash" ]]; then
                rm -f "$tmp"
                echo "ERROR: sha256 mismatch for ${so_name} (got=$got expected=$expected_hash)" >&2
                exit 1
            fi
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
export TT_METAL_DISABLE_SFPLOADMACRO="${TT_METAL_DISABLE_SFPLOADMACRO:-1}"

# ttsim does not implement NOC API v2; default to using v1 unless caller set it.
export TT_METAL_QUASAR_NOC_API_VERSION="${TT_METAL_QUASAR_NOC_API_VERSION:-1}"

mkdir -p "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────
# Build pytest argv
# ──────────────────────────────────────────────────────────────
PYTEST_BASE_ARGS=(
    # Display config: capture-untouched setup that we confirmed actually
    # preserves the child's captured stdout in the junit XML. -q keeps the
    # terminal to one dot per test, `console_output_style=classic` drops the
    # trailing percentage column so it's pure dots, sugar stays off (it
    # confused the failure tally with pytest-forked anyway), and we
    # deliberately do NOT override `-s` from python_tests/pytest.ini or
    # `log_cli=true` — both of those overrides empirically caused
    # pytest-forked's child output to disappear from <system-out>, so the
    # ttsim ERROR lines were missing from the HTML report.
    -q
    -o console_output_style=classic
    -p no:sugar
    --run-simulator
    --timeout="$TIMEOUT"
    --forked
    -m "$MARKER_EXPR"
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
        --dist=worksteal
        --max-worker-restart=10000
    )
fi

if [[ ${#DESELECT_ARGS[@]} -gt 0 ]]; then
    PYTEST_BASE_ARGS+=("${DESELECT_ARGS[@]}")
fi

# ──────────────────────────────────────────────────────────────
# Enumerate the expected test set (stable denominator)
# ──────────────────────────────────────────────────────────────
# `--collect-only` lists the tests pytest *would* run for this selection. This
# set is pure static parametrization (there is no pytest_generate_tests, and
# pytest_collection_modifyitems no-ops without --test-order-file) and therefore
# identical across ttsim versions, so it gives us a version-independent
# denominator. The ignore list's --deselect prefixes are applied here too, so
# an ignored test is absent from both lists. The renderer diffs this list
# against the testcases actually recorded in the JUnit XML; anything collected
# but never recorded is counted as "crashed"
# (a hard crash that took down the worker before its result could be written,
# so --forked never turned it into a normal failure).
#
# We avoid --run-simulator here (it would load+init the ttsim .so just to
# enumerate). Instead:
#   * CHIP_ARCH=<arch> lets conftest import without probing for a real device
#     (helpers.device runs get_all_cores() at import → get_chip_architecture(),
#     which short-circuits on CHIP_ARCH instead of calling check_context()).
#   * --compile-producer puts conftest in BuildMode.PRODUCE, which is the only
#     mode that skips *both* check_context() calls in pytest_configure
#     (override_gprs_used_by_tensix_dump and the device/simulator init block).
# --collect-only never executes a test, so producer mode compiles nothing. Note
# that producer mode collapses runtime-only axes, so this is a stable compile
# denominator, not necessarily the full runtime test count.
COLLECT_CHIP_ARCH="$ARCHITECTURE"
COLLECT_CMD=(
    "pytest"
    --collect-only
    -q
    -p no:sugar
    -o log_cli=false
    --compile-producer
    -m "$MARKER_EXPR"
)
if [[ ${#DESELECT_ARGS[@]} -gt 0 ]]; then
    COLLECT_CMD+=("${DESELECT_ARGS[@]}")
fi
if [[ ${#TEST_PATHS[@]} -gt 0 ]]; then
    COLLECT_CMD+=("${TEST_PATHS[@]}")
fi
if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    COLLECT_CMD+=("${PYTEST_ARGS[@]}")
fi

collected_count=0
collect_exit=0
(
    cd "$TESTS_DIR"
    CHIP_ARCH="$COLLECT_CHIP_ARCH" "${COLLECT_CMD[@]}"
) 2>/dev/null | grep -E '\.py::' >"$COLLECTED_PATH" || collect_exit=$?
# `grep` returns 1 when it matches nothing; only treat a missing/empty file as
# a real failure so the run still proceeds (gap analysis is best-effort).
if [[ -s "$COLLECTED_PATH" ]]; then
    collected_count="$(wc -l <"$COLLECTED_PATH" | tr -d ' ')"
else
    echo "WARNING: test collection produced no node IDs (exit=$collect_exit);" >&2
    echo "         crashed-test gap analysis will be skipped." >&2
    rm -f "$COLLECTED_PATH"
fi

# ──────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────
echo "============================================================"
echo " ttsim LLK regression"
echo "============================================================"
echo " Architecture   : ${ARCHITECTURE}"
echo " Marker expr    : ${MARKER_EXPR}"
echo " Simulator      : ${TT_METAL_SIMULATOR}"
echo " SoC descriptor : $(dirname "$TT_METAL_SIMULATOR")/soc_descriptor.yaml"
echo " SFPLOADMACRO   : disabled=${TT_METAL_DISABLE_SFPLOADMACRO}"
echo " NOC API version: ${TT_METAL_QUASAR_NOC_API_VERSION}"
echo " Workers (-n)   : ${WORKERS}"
echo " Per-test fork  : on"
echo " Timeout        : ${TIMEOUT}s"
echo " Tests collected: ${collected_count}"
if [[ "$USE_IGNORE" -eq 1 ]]; then
    echo " Skip list      : ${IGNORE_FILE} (${#IGNORE_PREFIXES[@]} for ${ARCHITECTURE})"
else
    echo " Skip list      : disabled"
fi
echo " JUnit XML      : ${XML_PATH}"
if [[ "$WRITE_HTML" -eq 1 ]]; then
    echo " HTML report    : ${HTML_PATH}"
else
    echo " HTML report    : disabled"
fi
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
# Render HTML unless --no-html. Partial and failed runs still get a report
# when rendering is enabled.
# ──────────────────────────────────────────────────────────────
# Prefer the in-tree renderer (render_ttsim_report.py) which produces a
# polished single-page report with pass/fail/skip percentages, a donut
# chart, and a ttsim-error-category breakdown. Fall back to junit2html
# only if the local renderer or its single dep (junitparser) is missing.
RENDERER="${SCRIPT_DIR}/render_ttsim_report.py"
if [[ -f "$XML_PATH" ]]; then
    rendered=0
    if [[ "$WRITE_HTML" -eq 0 ]]; then
        ln -sfn "$(basename "$XML_PATH")" "$LATEST_XML"
        [[ -s "$COLLECTED_PATH" ]] && ln -sfn "$(basename "$COLLECTED_PATH")" "$LATEST_COLLECTED"
    elif [[ -x "$RENDERER" || -f "$RENDERER" ]] \
       && python3 -c "import junitparser" &>/dev/null; then
        RENDER_CMD=(python3 "$RENDERER" "$XML_PATH" "$HTML_PATH")
        # Pass the expected-test list so the renderer can flag the
        # collected-but-unrecorded gap as crashed tests.
        if [[ -s "$COLLECTED_PATH" ]]; then
            RENDER_CMD+=(--collected "$COLLECTED_PATH")
        fi
        if "${RENDER_CMD[@]}"; then
            rendered=1
        fi
    fi
    if [[ "$WRITE_HTML" -eq 1 && $rendered -eq 0 ]] && command -v junit2html &>/dev/null; then
        echo "Falling back to junit2html (install junitparser for the nicer report)" >&2
        junit2html "$XML_PATH" "$HTML_PATH"
        rendered=1
    fi
    if [[ "$WRITE_HTML" -eq 1 && $rendered -eq 1 ]]; then
        ln -sfn "$(basename "$XML_PATH")"  "$LATEST_XML"
        ln -sfn "$(basename "$HTML_PATH")" "$LATEST_HTML"
        [[ -s "$COLLECTED_PATH" ]] && ln -sfn "$(basename "$COLLECTED_PATH")" "$LATEST_COLLECTED"
    elif [[ "$WRITE_HTML" -eq 1 ]]; then
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
    if [[ $pytest_exit -eq $PYTEST_NO_TESTS_COLLECTED && ${#IGNORE_PREFIXES[@]} -gt 0 ]]; then
        echo " No tests were collected. The skip list may have deselected every requested test."
        echo " Re-run with --no-ignore to include those tests."
    fi
fi
echo "------------------------------------------------------------"
echo " XML  : ${XML_PATH}"
[[ -f "$HTML_PATH" ]] && echo " HTML : ${HTML_PATH}"
[[ -L "$LATEST_HTML" ]] && echo " latest: ${LATEST_HTML}"
echo "============================================================"

exit "$pytest_exit"
