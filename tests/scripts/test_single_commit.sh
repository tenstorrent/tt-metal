#!/usr/bin/env bash
set -euo pipefail

# Test a single commit to determine if it passes or fails
# Usage: test_single_commit.sh <commit_sha> <script_path> <timeout> <attempts> [--tracy] [--nd-mode] [--artifact-mode]
# Returns: 0 if good, 1 if bad, 2 if skipped

die() { echo "ERROR: $*" >&2; exit 1; }

commit_sha="$1"
script_path="$2"
timeout_minutes="$3"
attempts="$4"
shift 4

tracy_enabled=0
nd_mode=false
artifact_mode=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --tracy)
      tracy_enabled=1
      shift
      ;;
    --nd-mode)
      nd_mode=true
      shift
      ;;
    --artifact-mode)
      artifact_mode=true
      shift
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[ -n "$commit_sha" ] || die "Commit SHA required"
[ -n "$script_path" ] || die "Script path required"
[ -f "$script_path" ] || die "Script not found: $script_path"

# Create the virtual environment and install dependencies (if not already created)
if [ ! -d "./python_env" ]; then
  echo "Creating virtual environment and installing dependencies..."
  ./create_venv.sh --force
fi

# Checkout the commit
git checkout "$commit_sha" >/dev/null 2>&1 || die "Failed to checkout commit: $commit_sha"

# Keep CPM cache across iterations for speed, but ensure it's in the workspace
export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-$TT_METAL_HOME/.cpmcache}"
mkdir -p "$CPM_SOURCE_CACHE"
export CMAKE_ARGS="-DCPM_SOURCE_CACHE=$CPM_SOURCE_CACHE -DCPM_DOWNLOAD_ALL=ON -DCPM_USE_LOCAL_PACKAGES=OFF"

# Helper to ensure repo matches rev and all build products are fresh
fresh_clean() {
  git submodule sync --recursive
  git submodule update --init --recursive --force
  git reset --hard
  rm -rf build build_Release build_Debug || true
}

# After building, verify we import from the workspace
verify_import_path() {
  PY_BIN="python"
  if [ -x "./python_env/bin/python" ]; then
    PY_BIN="./python_env/bin/python"
  fi

  "$PY_BIN" - <<'PY'
import sys
try:
    import ttnn
    if hasattr(ttnn, 'get_arch_name'):
        ttnn.get_arch_name()
except Exception as e:
    print(f"ERROR during ttnn import/test: {e}")
    sys.exit(1)
PY
}

# Try to download build artifacts from GitHub Actions
try_download_artifacts() {
  local commit_sha="$1"
  local download_script="./build_bisect/download_artifacts.sh"

  if [ ! -f "$download_script" ]; then
    echo "Download script not found: $download_script"
    return 1
  fi

  echo "Attempting to download artifacts for $commit_sha..."
  if [ "$tracy_enabled" -eq 1 ]; then
    "$download_script" "$commit_sha" --tracy
  else
    "$download_script" "$commit_sha"
  fi
}

# Check if commit message contains [skip-ci]
commit_msg="$(git log -1 --pretty=%s HEAD)"
case "$commit_msg" in
  *"[skip ci]"* | *"[skip CI]"* | *"[skip-ci]"* | *"[skip-CI]"* )
    echo "Commit contains [skip ci], skipping"
    exit 2
    ;;
esac

echo "::group::Building $commit_sha"
fresh_clean

build_rc=0
# Always try to download artifacts first (unless explicitly disabled)
# This avoids rebuilding metal each time if artifacts are available
if try_download_artifacts "$commit_sha"; then
  echo "Using downloaded artifacts for $commit_sha"
else
  echo "Artifact download failed or not available, falling back to local build"
  build_args="--build-all --enable-ccache"
  [ "$tracy_enabled" -eq 0 ] && build_args="$build_args --disable-profiler"
  ./build_metal.sh $build_args || build_rc=$?
fi

echo "::endgroup::"

if [ $build_rc -ne 0 ]; then
  echo "Build failed (rc=$build_rc); skipping this commit"
  exit 2
fi

echo "::group::Import sanity ($commit_sha)"
if ! verify_import_path 2>&1; then
  echo "Import path check failed; skipping this commit"
  echo "::endgroup::"
  exit 2
fi
echo "::endgroup::"

detect_galaxy() {
    local smi_output=$(tt-smi -ls 2>/dev/null)
    if echo "$smi_output" | grep -q "Wormhole"; then
        local device_count=$(echo "$smi_output" | grep -c "tt-galaxy-wh")
        if [[ "$device_count" -ge 32 ]]; then
            echo "topology-6u"
            return 0
        fi
    fi
    echo ""
}

is_galaxy=$(detect_galaxy)

echo "::group::Testing $commit_sha"
output_file="bisect_test_output.log"

if [ "$nd_mode" = true ]; then
  success_count=0
  failure_count=0
  skip_count=0
  run_idx=1

  while [ $run_idx -le $attempts ]; do
    echo "Attempt $run_idx/$attempts on $(git rev-parse HEAD)"
    echo "Resetting devices..."
    if [ "$is_galaxy" == "topology-6u" ]; then
      tt-smi -glx_reset_auto 2>&1 || true
    else
      tt-smi -r >/dev/null 2>&1 || true
    fi

    if timeout -k 10s "${timeout_minutes}m" bash -lc "$script_path" 2>&1 | tee "$output_file"; then
      if grep -qiE "(^|[^a-zA-Z])(SKIP|SKIPPED)([^a-zA-Z]|$)" "$output_file"; then
        echo "Attempt $run_idx: detected skip"
        skip_count=$((skip_count+1))
      else
        echo "Attempt $run_idx: success"
        success_count=$((success_count+1))
      fi
    else
      rc=$?
      if [ $rc -eq 124 ] || [ $rc -eq 137 ] || [ $rc -eq 143 ]; then
        echo "Attempt $run_idx: timeout/kill (rc=$rc) -> counting as skipped"
        skip_count=$((skip_count+1))
      else
        echo "Attempt $run_idx: failure (rc=$rc)"
        failure_count=$((failure_count+1))
      fi
    fi
    run_idx=$((run_idx+1))
  done

  evaluated=$((success_count + failure_count))
  echo "ND summary: successes=$success_count failures=$failure_count skips=$skip_count evaluated=$evaluated"
  echo "::endgroup::"

  if [ $failure_count -ge 1 ]; then
    exit 1  # Bad
  elif [ $evaluated -eq 0 ]; then
    exit 2  # Skipped
  else
    exit 0  # Good
  fi
else
  run_idx=1
  timeout_rc=1

  while [ $run_idx -le $attempts ]; do
    echo "Attempt $run_idx/$attempts on $(git rev-parse HEAD)"
    echo "Resetting devices..."
    tt-smi -r >/dev/null 2>&1 || true

    if timeout -k 10s "${timeout_minutes}m" bash -lc "$script_path" 2>&1 | tee "$output_file"; then
      timeout_rc=0
      break
    else
      timeout_rc=$?
      echo "Test failed (code $timeout_rc), retrying…"
      run_idx=$((run_idx+1))
    fi
  done

  echo "Final exit code: $timeout_rc"
  echo "::endgroup::"

  if [ $timeout_rc -eq 0 ]; then
    exit 0  # Good
  elif [ $timeout_rc -eq 124 ] || [ $timeout_rc -eq 137 ] || [ $timeout_rc -eq 143 ]; then
    exit 2  # Skipped (timeout)
  else
    exit 1  # Bad
  fi
fi
