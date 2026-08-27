#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_LOG="/tmp/smollm2_135m_p150_smoke_$(date +%Y%m%d_%H%M%S).log"
FULL_LOG="/tmp/smollm2_135m_p150_full_$(date +%Y%m%d_%H%M%S).log"

cd "$SCRIPT_DIR"
source python_env/bin/activate

# =============================================================================
# ABI preflight
# =============================================================================
# Guards the failure recorded as Bug 1 in SMOLLM2_P150_HANDOFF.md: a `_ttnn.so`
# compiled against one CPython minor version and imported under another
# segfaults inside CPython's own type machinery during nanobind type
# registration, before any device or model code runs, with no usable traceback.
# Diagnosing that cost a full rebuild. This check reads the build's
# `CMakeCache.txt`, works out which CPython minor version `_ttnn.so` was compiled
# against, compares it with the interpreter that is about to run pytest, and stops
# here rather than letting pytest segfault.
#
# The build-side version comes from what CMake recorded at configure time
# (`_Python3_INTERPRETER_PROPERTIES`, else the `Python3_INCLUDE_DIR` suffix), and
# only falls back to probing `Python3_EXECUTABLE` live when the cache recorded no
# version. Probing the path is not equivalent: an in-place interpreter upgrade
# leaves the path valid while the built `.so` goes stale, and a live probe would
# report the new version and wave the run through into the segfault.
#
# Degradation policy: only a confirmed minor-version mismatch, read from a cache
# whose interpreter still exists and whose version is known, is fatal. A missing
# build directory, a missing or unreadable cache, a missing entry, a stale entry
# pointing at an interpreter that no longer exists, and a cache that records no
# version whose interpreter reports none either all produce a warning and let the
# run continue. The check never invents a failure it cannot
# substantiate, and never treats an unreadable version as a version.
#
# Overrides, used to exercise the check without a rebuild:
#   ABI_PREFLIGHT_BUILD_DIR       build directory to read CMakeCache.txt from
#   ABI_PREFLIGHT_RUNTIME_PYTHON  interpreter to treat as the one running pytest
#   ABI_PREFLIGHT_VERBOSE=1       print the passing verdict instead of staying silent
# Run the check on its own with:  ./smollm2_135m_p150_demo.sh --preflight-only

abi_preflight_warn() {
  echo "ABI preflight: WARNING: $1" >&2
  echo "ABI preflight: continuing without the Python ABI check." >&2
}

abi_preflight() {
  local build_dir cache entry build_py runtime_py build_ver runtime_ver
  local pytest_bin shebang cand recorded bin_dir live_ver build_src upgraded_note

  # --- which build directory holds the _ttnn.so that will be imported ---
  if [ -n "${ABI_PREFLIGHT_BUILD_DIR:-}" ]; then
    build_dir="$ABI_PREFLIGHT_BUILD_DIR"
  elif [ -n "${TT_METAL_BUILD_DIR:-}" ]; then
    build_dir="$TT_METAL_BUILD_DIR"
  else
    # $SCRIPT_DIR/build is normally a symlink to the active build_Release.
    build_dir="$SCRIPT_DIR/build"
    if [ ! -d "$build_dir" ] && [ -d "$SCRIPT_DIR/build_Release" ]; then
      build_dir="$SCRIPT_DIR/build_Release"
    fi
  fi
  if [ ! -d "$build_dir" ]; then
    abi_preflight_warn "no build directory found at '$build_dir'."
    return 0
  fi

  cache="$build_dir/CMakeCache.txt"
  if [ ! -r "$cache" ]; then
    abi_preflight_warn "no readable CMakeCache.txt at '$cache'."
    return 0
  fi

  # --- interpreter the extension was compiled against ---
  entry="$(grep -m1 -E '^Python3_EXECUTABLE(:[^=]*)?=' "$cache" || true)"
  build_py="${entry#*=}"
  build_py="${build_py%$'\r'}"          # CRLF cache written on another host
  case "$build_py" in                   # a quoted cache value
    \"*\") build_py="${build_py#\"}"; build_py="${build_py%\"}" ;;
    \'*\') build_py="${build_py#\'}"; build_py="${build_py%\'}" ;;
  esac
  if [ -z "$build_py" ]; then
    abi_preflight_warn "CMakeCache.txt at '$cache' has no Python3_EXECUTABLE entry."
    return 0
  fi

  # --- interpreter that will actually run pytest ---
  runtime_py="${ABI_PREFLIGHT_RUNTIME_PYTHON:-}"
  if [ -z "$runtime_py" ]; then
    # Resolve the interpreter that the pytest below will actually re-exec into.
    pytest_bin="$(command -v pytest || true)"
    if [ -n "$pytest_bin" ] && [ -r "$pytest_bin" ]; then
      # Console scripts live in the same bin/ as their interpreter. This is the
      # reliable route, because the venv's pytest is a relocatable /bin/sh shim
      # that re-execs "$(dirname $(realpath $0))/python3", so its shebang names
      # /bin/sh rather than a Python.
      bin_dir="$(dirname -- "$(realpath -- "$pytest_bin" 2>/dev/null || echo "$pytest_bin")")"
      for cand in "$bin_dir/python3" "$bin_dir/python"; do
        if [ -x "$cand" ]; then runtime_py="$cand"; break; fi
      done
      # Fall back to a genuine python shebang if the bin/ layout is unusual.
      if [ -z "$runtime_py" ]; then
        shebang="$(head -n 1 "$pytest_bin" 2>/dev/null || true)"
        case "$shebang" in
          '#!'*)
            cand="${shebang#\#!}"
            cand="${cand%% *}"
            case "$(basename -- "$cand")" in
              python*) if [ -x "$cand" ]; then runtime_py="$cand"; fi ;;
            esac
            ;;
        esac
      fi
    fi
  fi
  if [ -z "$runtime_py" ]; then
    runtime_py="$(command -v python3 || command -v python || true)"
  fi
  if [ -z "$runtime_py" ] || [ ! -x "$runtime_py" ]; then
    abi_preflight_warn "could not identify the interpreter that will run pytest."
    return 0
  fi

  # Both the exit status and the output are checked. An executable that exits 0
  # and prints nothing must not be read as "version empty"; that would produce a
  # garbled hard failure against a real version, or, if both sides are empty, a
  # silently passing no-op check, which is the worst possible failure for a guard.
  runtime_ver="$("$runtime_py" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
  case "$runtime_ver" in
    [0-9]*.[0-9]*) ;;
    *)
      abi_preflight_warn "could not read a version from the runtime interpreter '$runtime_py' (got '$runtime_ver')."
      return 0
      ;;
  esac

  # The version CMake recorded at configure time, which is what _ttnn.so was
  # actually compiled against. Preferred over asking $build_py what it is today:
  # an in-place venv upgrade (uv venv --python 3.13 python_env, a distro bump of
  # /usr/bin/python3) leaves the path valid while the built .so goes stale, and a
  # live probe would report the new version and pass the run straight into the
  # segfault this guard exists to stop.
  # tr -d '\r' on both: a CRLF cache would otherwise leave the include-dir regex
  # unable to anchor on end-of-line, silently dropping this whole signal and
  # falling back to the live probe, which is the hole this block exists to close.
  recorded="$(grep -m1 -E '^_Python3_INTERPRETER_PROPERTIES:' "$cache" 2>/dev/null | tr -d '\r' | cut -d= -f2- | cut -d';' -f2,3 | tr ';' '.' || true)"
  case "$recorded" in
    [0-9]*.[0-9]*) ;;
    *)
      # Second recorded source: the include dir ends in .../python3.NN.
      recorded="$(grep -m1 -E '^Python3_INCLUDE_DIR(:[^=]*)?=' "$cache" 2>/dev/null | tr -d '\r' | sed -n 's|.*/python\([0-9][0-9]*\.[0-9][0-9]*\)$|\1|p' || true)"
      case "$recorded" in
        [0-9]*.[0-9]*) ;;
        *) recorded="" ;;
      esac
      ;;
  esac

  # A stale cache naming an interpreter that is gone cannot prove anything, so
  # report what the cache recorded and downgrade to a warning.
  if [ ! -x "$build_py" ]; then
    if [ -n "$recorded" ]; then
      abi_preflight_warn "stale cache: Python3_EXECUTABLE='$build_py' no longer exists (cache recorded Python $recorded; this run uses Python $runtime_ver)."
    else
      abi_preflight_warn "stale cache: Python3_EXECUTABLE='$build_py' no longer exists, and no recorded version to fall back on."
    fi
    return 0
  fi

  live_ver="$("$build_py" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
  case "$live_ver" in
    [0-9]*.[0-9]*) ;;
    *) live_ver="" ;;
  esac

  if [ -n "$recorded" ]; then
    build_ver="$recorded"
    build_src="recorded in $cache at configure time"
  elif [ -n "$live_ver" ]; then
    build_ver="$live_ver"
    build_src="probed from $build_py just now; the cache recorded no version"
  else
    abi_preflight_warn "could not read a version from the build interpreter '$build_py' (got '$live_ver'), and the cache recorded none."
    return 0
  fi

  # Worth saying out loud: the interpreter at that path is not the one the build
  # used, so a rebuild is needed even if the recorded version happens to match.
  upgraded_note=""
  if [ -n "$live_ver" ] && [ -n "$recorded" ] && [ "$live_ver" != "$recorded" ]; then
    upgraded_note="  NOTE: '$build_py' is Python $live_ver today but the build recorded Python $recorded.
        That path was upgraded in place after the build was configured, so the
        built _ttnn.so is stale regardless of which interpreter you run.

"
  fi

  if [ "$build_ver" != "$runtime_ver" ]; then
    cat >&2 <<EOF

=== ABI preflight FAILED: Python ABI mismatch, refusing to run pytest ===

$upgraded_note  _ttnn.so in this build was compiled against Python $build_ver, but pytest here
  would run under Python $runtime_ver. Importing ttnn across a CPython minor-version
  boundary segfaults inside CPython's type machinery while nanobind registers
  ttnn types, before any device or model code runs, and produces no usable
  traceback. This is Bug 1 in SMOLLM2_P150_HANDOFF.md.

  build directory        : $build_dir
  Python3_EXECUTABLE     : $build_py
  built against          : Python $build_ver  ($build_src)
  interpreter to be used : $runtime_py  (Python $runtime_ver)

  Fix: rebuild against the interpreter you intend to run.

    cd $SCRIPT_DIR
    source python_env/bin/activate     # makes python3 resolve to python_env
    rm -rf build_Release
    ./build_metal.sh

  build_metal.sh derives Python3_EXECUTABLE from 'command -v python3' at
  configure time, so python_env must be active and unrelated venvs (for example
  /home/stisi/tt-smi/venv) must be off PATH when you configure.

  If you believe the build is fine and this check is wrong, verify by hand:
    grep Python3_EXECUTABLE $cache
    $runtime_py -c 'import ttnn'

EOF
    return 1
  fi

  if [ "${ABI_PREFLIGHT_VERBOSE:-0}" = "1" ]; then
    echo "ABI preflight: OK (built against Python $build_ver, $build_src; running Python $runtime_ver via $runtime_py)"
  fi
  return 0
}

PREFLIGHT_ONLY=0
if [ "$#" -gt 0 ]; then
  for arg in "$@"; do
    case "$arg" in
      --preflight-only|--check-abi)
        PREFLIGHT_ONLY=1
        ;;
      -h|--help)
        echo "Usage: $0 [--preflight-only]"
        echo "  --preflight-only   run the ABI preflight and exit, without touching a device"
        exit 0
        ;;
      *)
        echo "Unknown argument: $arg (try --help)" >&2
        exit 2
        ;;
    esac
  done
fi

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  if abi_preflight; then
    echo "ABI preflight: passed."
    exit 0
  fi
  exit 1
fi

# Nothing below this line may run before the preflight has passed.
abi_preflight

# =============================================================================
# Demo run
# =============================================================================

# SmolLM2-135M is public/ungated (Apache 2.0), so HF_TOKEN is not required to
# download it. Set one anyway if you have HF rate-limit issues.
export TT_METAL_HOME="$SCRIPT_DIR"
export ARCH_NAME=blackhole
export MESH_DEVICE=P150
export HF_MODEL=HuggingFaceTB/SmolLM2-135M
export TT_CACHE_PATH="$SCRIPT_DIR/tt_cache/smollm2-135m"
# PAD_MLP_CORES is deliberately NOT set. It pads ModelArgs.hidden_dim, which
# comes from intermediate_size (1536), not from hidden_size (576), so it is a
# no-op for this model: nearest_multiple(1536, 16*32) == 1536. Measured in
# models/autoports/huggingfacetb_smollm2_135m/doc/p150_run/. Export it yourself
# if you want to experiment.
# Deliberately NOT setting CI=true: simple_text_demo.py skips any non-ci_only
# test id (including "performance and batch-1") when CI=true.

echo "Model:       $HF_MODEL"
echo "Mesh device: $MESH_DEVICE"
echo "PAD_MLP_CORES: ${PAD_MLP_CORES:-<unset>}"
echo ""

echo "=== Stage 1/2: smoke test (1 layer, 5 tokens) ==="
echo "Log: $SMOKE_LOG"
if ! pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" \
    --num_layers 1 --max_generated_tokens 5 --timeout=600 -s 2>&1 | tee "$SMOKE_LOG"; then
  echo ""
  echo "=== SMOKE TEST FAILED ==="
  echo "Full log: $SMOKE_LOG"
  echo "If this looks like a layout/shape error, try setting PAD_MLP_CORES"
  echo "(multiples of 8 between 8 and 64) and re-run:"
  echo "  PAD_MLP_CORES=32 $0"
  exit 1
fi

echo ""
echo "=== Stage 1/2 passed. Stage 2/2: full demo (30 layers, 200 tokens) ==="
echo "Log: $FULL_LOG"
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" \
    --timeout=600 -s 2>&1 | tee "$FULL_LOG"

echo ""
echo "=== SUMMARY ==="
echo "-- smoke test --"
grep -E "PASSED|FAILED|SKIPPED|ERROR" "$SMOKE_LOG" || true
echo "-- full demo --"
grep -E "PASSED|FAILED|SKIPPED|ERROR" "$FULL_LOG" || true
echo ""
echo "Smoke log: $SMOKE_LOG"
echo "Full log:  $FULL_LOG"
