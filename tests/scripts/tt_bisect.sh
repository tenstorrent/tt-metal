#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

: << 'END'
Usage:
  -f TEST        : test command to run (quote if it has spaces)
  -g GOOD_SHA    : known good commit
  -b BAD_SHA     : known bad commit
  -t TIMEOUT     : per-iteration timeout (default 30m)
  -p             : enable Tracy profiling
  -r RETRIES     : number of retries (default 3)
  -n             : enable non-deterministic detection mode (ND mode)
  -a             : enable artifact download optimization (requires gh CLI)
END

timeout_duration_iteration="30m"
test=""
good_commit=""
bad_commit=""
tracy_enabled=0
retries=3
nd_mode=false
artifact_mode=false
run_idx=0
timeout_rc=1

while getopts ":f:g:b:t:pr:na" opt; do
  case "$opt" in
    f) test="$OPTARG" ;;
    g) good_commit="$OPTARG" ;;
    b) bad_commit="$OPTARG" ;;
    t) timeout_duration_iteration="$OPTARG" ;;
    p) tracy_enabled=1 ;;
    r) retries="$OPTARG" ;;
    n) nd_mode=true ;;
    a) artifact_mode=true ;;
    \?) die "Invalid option: -$OPTARG" ;;
    :)  die "Option -$OPTARG requires an argument." ;;
  esac
done

[ -n "$test" ] || die "Please specify -f TEST."
[ -n "$good_commit" ] || die "Please specify -g GOOD_SHA."
[ -n "$bad_commit" ] || die "Please specify -b BAD_SHA."

echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "ARCH_NAME: ${ARCH_NAME:-}"
echo "pwd: $(pwd)"
if [ "$tracy_enabled" -eq 1 ]; then
  echo "Tracy profiling enabled for builds."
fi

if [ "$artifact_mode" = true ]; then
  echo "Artifact download optimization enabled."
fi

# Set up environment (skip if already in CI container with pre-configured venv)
if [ ! -d "$PYTHON_ENV_DIR" ]; then
  echo "Creating virtual environment and installing dependencies..."
  CXX=clang++-17 CC=clang-17 ./create_venv.sh
  pip install -r models/tt_transformers/requirements.txt
else
  echo "Using existing virtual environment: $PYTHON_ENV_DIR"
fi

git cat-file -e "$good_commit^{commit}" 2>/dev/null || die "Invalid good commit: $good_commit"
git cat-file -e "$bad_commit^{commit}" 2>/dev/null  || die "Invalid bad commit: $bad_commit"

echo "Good: $good_commit"
echo "Bad : $bad_commit"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"
echo "Status:"
git status --porcelain=v1

# Keep CPM cache across iterations for speed, but ensure it's in the workspace
export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-$TT_METAL_HOME/.cpmcache}"
mkdir -p "$CPM_SOURCE_CACHE"
export CMAKE_ARGS="-DCPM_SOURCE_CACHE=$CPM_SOURCE_CACHE -DCPM_DOWNLOAD_ALL=ON -DCPM_USE_LOCAL_PACKAGES=OFF"

# Helper to ensure repo matches rev and all build products are fresh
fresh_clean() {
  # Match submodules exactly for current rev
  git submodule sync --recursive
  git submodule update --init --recursive --force

  # Nuke build outputs but keep venv/cache
  git reset --hard
  rm -rf build build_Release build_Debug || true
}

# After building, verify we import from the workspace
verify_import_path() {
  python - <<'PY'
import ttnn, sys
print(ttnn.get_arch_name())
print("ttnn imported from:", ttnn.__file__)
PY
}

# Try to download build artifacts from GitHub Actions using external script
try_download_artifacts() {
  local commit_sha="$1"
  local script_dir="$(dirname "${BASH_SOURCE[0]}")"
  local download_script="$script_dir/download_artifacts.sh"

  if [ ! -f "$download_script" ]; then
    echo "ERROR: Download artifacts script not found: $download_script"
    return 1
  fi

  # Call the external script with appropriate flags
  if [ "$tracy_enabled" -eq 1 ]; then
    "$download_script" "$commit_sha" --tracy
  else
    "$download_script" "$commit_sha"
  fi
}

echo "Starting git bisect…"
git bisect start "$bad_commit" "$good_commit"

found=false
while [[ "$found" == "false" ]]; do
  rev="$(git rev-parse --short=12 HEAD)"
  full_sha="$(git rev-parse HEAD)"
  echo "::group::Building $rev"

  fresh_clean

  build_rc=0

  # Try to download artifacts first if artifact mode is enabled
  if [ "$artifact_mode" = true ]; then
    if try_download_artifacts "$full_sha"; then
      echo "Using downloaded artifacts for $rev"
      build_rc=0
    else
      echo "WARNING: Artifact download failed, falling back to local build"
      if [ "$tracy_enabled" -eq 1 ]; then
        ./build_metal.sh \
          --build-all \
          --enable-ccache \
          --enable-profiler || build_rc=$?
      else
        ./build_metal.sh \
          --build-all \
          --enable-ccache || build_rc=$?
      fi
    fi
  else
    # Standard local build
    if [ "$tracy_enabled" -eq 1 ]; then
      ./build_metal.sh \
        --build-all \
        --enable-ccache \
        --enable-profiler || build_rc=$?
    else
      ./build_metal.sh \
        --build-all \
        --enable-ccache || build_rc=$?
    fi
  fi

  echo "::endgroup::"

  if [ $build_rc -ne 0 ]; then
    echo "Build failed (rc=$build_rc); skipping this commit"
    git bisect skip
    continue
  fi

  echo "::group::Import sanity ($rev)"
  if ! verify_import_path; then
    echo "Import path check failed; skipping this commit"
    git bisect skip
    echo "::endgroup::"
    continue
  fi
  echo "::endgroup::"

  echo "::group::Testing $rev"
  output_file="bisect_test_output.log"
  if [ "$nd_mode" = true ]; then
    success_count=0
    failure_count=0
    skip_count=0
    run_idx=1
    while [ $run_idx -le $retries ]; do
      echo "Attempt $run_idx/$retries on $(git rev-parse HEAD)"
      echo "Resetting devices..."
      tt-smi -r >/dev/null 2>&1 || true
      echo "Devices reset"

      echo "Run: $test"
      if timeout -k 10s "$timeout_duration_iteration" bash -lc "$test" 2>&1 | tee "$output_file"; then
        if grep -qiE "(^|[^a-zA-Z])(SKIP|SKIPPED)([^a-zA-Z]|$)" "$output_file"; then
          echo "Attempt $run_idx: detected skip (exit 0 with 'SKIP' in output)"
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

      echo "--- Logs (attempt $run_idx) ---"
      sed -n '1,200p' "$output_file" || true
      echo "------------------------------"
      run_idx=$((run_idx+1))
    done

    evaluated=$((success_count + failure_count))
    if [ $evaluated -gt 0 ]; then
      passrate=$(awk -v s=$success_count -v e=$evaluated 'BEGIN { printf "%.2f", (s*100.0)/e }')
    else
      passrate="NA"
    fi

    nd_log_file="bisect_nd_results.csv"
    if [ ! -f "$nd_log_file" ]; then
      echo "timestamp,commit_sha,rev_short,successes,failures,skips,evaluated,passrate_percent" > "$nd_log_file"
    fi
    echo "$(date -Iseconds),$(git rev-parse HEAD),$rev,$success_count,$failure_count,$skip_count,$evaluated,$passrate" >> "$nd_log_file"

    echo "ND summary for $rev: successes=$success_count failures=$failure_count skips=$skip_count evaluated=$evaluated passrate=$passrate%"
    echo "::endgroup::"

    if [ $failure_count -ge 1 ]; then
      out="$(git bisect bad || true)"
    elif [ $evaluated -eq 0 ]; then
      echo "All attempts were skipped; skipping this commit"
      git bisect skip
      continue
    else
      out="$(git bisect good || true)"
    fi

    first_line="$(printf '%s\n' "$out" | head -n1)"
    case "$first_line" in
      *"is the first bad commit"*)
        bad_sha="$(git rev-parse HEAD)"
        echo "FOUND IT: $first_line"
        echo "Commit: $bad_sha"
        echo "Title : $(git log -1 --pretty=%s "$bad_sha")"
        found=true
        ;;
      *"There are only 'skip'ped commits left to test."*)
        echo "Bisect inconclusive: only skipped commits left."
        echo "Last bisect output: $first_line"
        break
        ;;
      "")
        echo "git bisect produced no output; stopping to avoid an infinite loop."
        echo "Last bisect output: $first_line"
        break
        ;;
    esac
  else
    run_idx=1
    timeout_rc=1
    while [ $run_idx -le $retries ]; do
      echo "Attempt $run_idx/$retries on $(git rev-parse HEAD)"
      echo "Run: $test"
      if timeout -k 10s "$timeout_duration_iteration" bash -lc "$test" 2>&1 | tee "$output_file"; then
        timeout_rc=0
        echo "--- Logs (attempt $run_idx) ---"
        sed -n '1,200p' "$output_file" || true
        echo "------------------------------"
        break
      else
        timeout_rc=$?
        echo "Test failed (code $timeout_rc), retrying…"
        echo "--- Logs (attempt $run_idx) ---"
        sed -n '1,200p' "$output_file" || true
        echo "------------------------------"
        run_idx=$((run_idx+1))
      fi
    done
    echo "Final exit code: $timeout_rc"
    echo "::endgroup::"

    if [ $timeout_rc -eq 0 ]; then
      out="$(git bisect good || true)"
    elif [ $timeout_rc -eq 124 ] || [ $timeout_rc -eq 137 ] || [ $timeout_rc -eq 143 ]; then
      echo "Timeout/kill detected; skipping this commit"
      git bisect skip
      continue
    else
      out="$(git bisect bad || true)"
    fi

    first_line="$(printf '%s\n' "$out" | head -n1)"
    case "$first_line" in
      *"is the first bad commit"*)
        echo "FOUND IT: $first_line"
        found=true
        ;;
      *"There are only 'skip'ped commits left to test."*)
        echo "Bisect inconclusive: only skipped commits left."
        echo "Last bisect output: $first_line"
        break
        ;;
      "")
        echo "git bisect produced no output; stopping to avoid an infinite loop."
        echo "Last bisect output: $first_line"
        break
        ;;
    esac
  fi

done

git bisect reset || true
