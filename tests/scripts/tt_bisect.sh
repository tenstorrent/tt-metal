#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

: << 'END'
Usage:
  -f TEST        : test command to run (quote if it has spaces)
  -s SCRIPT      : path to script to run instead of test command (optional)
  -g GOOD_SHA    : known good commit
  -b BAD_SHA     : known bad commit
  -t TIMEOUT     : per-iteration timeout in minutes (default 30)
  -p             : enable Tracy profiling (disabled by default)
  -r RETRIES     : number of retries (default 3)
  -n             : enable non-deterministic detection mode (ND mode)
  -a             : enable artifact download optimization (requires gh CLI)
  -c SKIP_LIST   : comma-separated list of commits to automatically skip
Note: Either -f or -s must be specified, but not both.
END

timeout_duration_iteration="30"
test=""
script_path=""
good_commit=""
bad_commit=""
tracy_enabled=0
retries=3
nd_mode=false
artifact_mode=false
skip_commits=""
run_idx=0
timeout_rc=1

while getopts ":f:s:g:b:t:pr:nac:" opt; do
  case "$opt" in
    f) test="$OPTARG" ;;
    s) script_path="$OPTARG" ;;
    g) good_commit="$OPTARG" ;;
    b) bad_commit="$OPTARG" ;;
    t) timeout_duration_iteration="$OPTARG" ;;
    p) tracy_enabled=1 ;;
    r) retries="$OPTARG" ;;
    n) nd_mode=true ;;
    a) artifact_mode=true ;;
    c) skip_commits="$OPTARG" ;;
    \?) die "Invalid option: -$OPTARG" ;;
    :)  die "Option -$OPTARG requires an argument." ;;
  esac
done

export CXX=clang++-17
export CC=clang-17

# Either test or script_path must be specified, but not both
if [ -n "$script_path" ] && [ -n "$test" ]; then
  die "Cannot specify both -f TEST and -s SCRIPT. Use one or the other."
elif [ -z "$script_path" ] && [ -z "$test" ]; then
  die "Please specify either -f TEST or -s SCRIPT."
fi

[ -n "$good_commit" ] || die "Please specify -g GOOD_SHA."
[ -n "$bad_commit" ] || die "Please specify -b BAD_SHA."

# Validate script exists if specified
if [ -n "$script_path" ] && [ ! -f "$script_path" ]; then
  die "Script not found: $script_path"
fi

echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "ARCH_NAME: ${ARCH_NAME:-}"
echo "pwd: $(pwd)"
if [ "$tracy_enabled" -eq 1 ]; then
  echo "Tracy profiling enabled for builds."
else
  echo "Tracy profiling disabled for builds."
fi

if [ "$artifact_mode" = true ]; then
  echo "Artifact download optimization enabled."
fi

if [ -n "$skip_commits" ]; then
  echo "Auto-skip commits list: $skip_commits"
fi

# Create the virtual environment and install dependencies (including wheel). Allowing non-interactive overwrite.
echo "Creating virtual environment and installing dependencies..."
./create_venv.sh --force


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
  # Prefer the venv python if it exists so imports reflect the built workspace
  PY_BIN="python"
  if [ -x "./python_env/bin/python" ]; then
    PY_BIN="./python_env/bin/python"
  fi

  echo "DEBUG: Checking ttnn installation..."
  echo "DEBUG: PYTHONPATH=$PYTHONPATH"
  echo "DEBUG: Looking for ttnn at /work/ttnn/ttnn/__init__.py"
  ls -la /work/ttnn/ttnn/__init__.py || echo "WARNING: ttnn __init__.py not found!"
  echo "DEBUG: Looking for _ttnn.so"
  find /work/ttnn/ttnn -name "_ttnn*.so" -ls || echo "WARNING: _ttnn.so not found!"
  echo "DEBUG: Looking for built libraries in build_Release"
  ls -la /work/build_Release/lib/_ttnn*.so || ls -la /work/build/lib/_ttnn*.so || echo "WARNING: _ttnn.so not in build libs!"

  "$PY_BIN" - <<'PY'
import sys
print("DEBUG: Python sys.path:", sys.path)
try:
    import ttnn
    print("✓ ttnn imported from:", ttnn.__file__)
    print("DEBUG: ttnn module attributes:", dir(ttnn))
    if hasattr(ttnn, 'get_arch_name'):
        print("✓ Arch:", ttnn.get_arch_name())
    else:
        print("ERROR: get_arch_name not found in ttnn")
        print("DEBUG: Available functions starting with 'get':", [x for x in dir(ttnn) if x.startswith('get')])
        sys.exit(1)
except Exception as e:
    print(f"ERROR during ttnn import/test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PY
}

# Check if current commit should be auto-skipped
should_skip_commit() {
  local current_commit="$1"
  local skip_list="$2"

  if [ -z "$skip_list" ]; then
    return 1  # Don't skip if no list provided
  fi

  # Split comma-separated list and check each commit
  IFS=',' read -ra SKIP_ARRAY <<< "$skip_list"
  for skip_sha in "${SKIP_ARRAY[@]}"; do
    # Trim whitespace
    skip_sha="$(echo "$skip_sha" | xargs)"

    # Check if current commit starts with the skip SHA (supports short SHAs)
    if [[ "$current_commit" == "$skip_sha"* ]]; then
      return 0  # Should skip
    fi
  done

  return 1  # Don't skip
}

# Try to download build artifacts from GitHub Actions using external script
try_download_artifacts() {
  local commit_sha="$1"
  local download_script="./build_bisect/download_artifacts.sh"

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

# START GIT BISECT
echo "Starting git bisect…"
git bisect start "$bad_commit" "$good_commit"

found=false
while [[ "$found" == "false" ]]; do
  rev="$(git rev-parse --short=12 HEAD)"
  full_sha="$(git rev-parse HEAD)"

  commit_msg="$(git log -1 --pretty=%s HEAD)"

  echo "Commit message: $commit_msg"

  # Check if commit message contains [skip-ci] or [skip ci] (case-insensitive) and skip if so
  case "$commit_msg" in
    *"[skip ci]"* | *"[skip CI]"* | *"[skip-ci]"* | *"[skip-CI]"* )
      echo "Commit $rev contains [skip ci] or [skip-ci], skipping: $commit_msg"
      git bisect skip
      continue
      ;;
  esac


  # Check if this commit should be auto-skipped
  if should_skip_commit "$full_sha" "$skip_commits"; then
    echo "Auto-skipping commit $rev (matches skip list)"
    git bisect skip
    continue
  fi


  echo "::group::Building $rev"

  fresh_clean

  build_rc=0

  # Try to download artifacts first if artifact mode is enabled
  if [ "$artifact_mode" = true ] && try_download_artifacts "$full_sha"; then
    echo "Using downloaded artifacts for $rev"
  else
    [ "$artifact_mode" = true ] && echo "WARNING: Artifact download failed, falling back to local build"
    build_args="--build-all --enable-ccache"
    [ "$tracy_enabled" -eq 0 ] && build_args="$build_args --disable-profiler"
    ./build_metal.sh $build_args || build_rc=$?
  fi

  echo "::endgroup::"

  if [ $build_rc -ne 0 ]; then
    echo "Build failed (rc=$build_rc); skipping this commit"
    git bisect skip
    continue
  fi

  echo "::group::Import sanity ($rev)"
  if ! verify_import_path 2>&1; then
    echo "Import path check failed; skipping this commit"
    git bisect skip
    echo "::endgroup::"
    continue
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

      # Default for non-galaxy systems
      echo ""
  }

  is_galaxy=$(detect_galaxy)
  echo "Is Galaxy Detected: $is_galaxy"

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
      if [ "$is_galaxy" == "topology-6u" ]; then
        echo "Using galaxy reset mode"
        tt-smi -glx_reset_auto 2>&1 || true  # use galaxy reset mode
      else
        tt-smi -r >/dev/null 2>&1 || true  # use regular reset mode
      fi

      if [ -n "$script_path" ]; then
        echo "Run: $script_path"
        run_cmd="$script_path"
      else
        echo "Run: $test"
        run_cmd="$test"
      fi

      if timeout -k 10s "${timeout_duration_iteration}m" bash -lc "$run_cmd" 2>&1 | tee "$output_file"; then
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
      echo "Resetting devices..."
      tt-smi -r >/dev/null 2>&1 || true
      echo "Devices reset"

      if [ -n "$script_path" ]; then
        echo "Run: $script_path"
        run_cmd="$script_path"
      else
        echo "Run: $test"
        run_cmd="$test"
      fi

      if timeout -k 10s "${timeout_duration_iteration}m" bash -lc "$run_cmd" 2>&1 | tee "$output_file"; then
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
