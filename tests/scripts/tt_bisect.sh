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

# Check required tools for artifact mode
if [ "$artifact_mode" = true ]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "GitHub CLI (gh) not found, artifact mode disabled"
    artifact_mode=false
  elif ! command -v jq >/dev/null 2>&1; then
    echo "jq not found, artifact mode disabled"
    artifact_mode=false
  elif ! gh auth status >/dev/null 2>&1; then
    echo "GitHub CLI not authenticated, artifact mode disabled"
    artifact_mode=false
  else
    echo "Artifact download mode enabled"
  fi
fi


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

# Creating virtual environment where we can install ttnn
# ./create_venv.sh
# pip install -r models/tt_transformers/requirements.txt

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

# Try to download build artifacts from GitHub Actions
try_download_artifacts() {
  local commit_sha="$1"
  local tracy_suffix=""

  if [ "$tracy_enabled" -eq 1 ]; then
    tracy_suffix="-tracy"
  fi

  # Look for workflow runs for this commit
  echo "🔎 Searching for workflow runs for commit $commit_sha..."
  local runs
  echo "🔧 Debug: Running command - gh run list --repo tenstorrent/tt-metal --commit $commit_sha --json conclusion,databaseId,workflowName"
  gh run list --repo tenstorrent/tt-metal --commit "$commit_sha" --json conclusion,databaseId,workflowName
  runs="$(gh run list --repo tenstorrent/tt-metal --commit "$commit_sha" --json conclusion,databaseId,workflowName)"
  echo "🔧 Debug: Command output:"
  echo "$runs"
  # Check if gh command failed
  if [ $? -ne 0 ]; then
    echo "❌ gh run list command failed:"
    echo "$runs"
    return 1
  fi

  if [ -z "$runs"  ]; then
    echo "No workflow runs found for commit $commit_sha"
    return 1
  fi

  # Look for successful build workflows (prioritize "All post-commit tests")
  local build_run_id
  # First try to find "All post-commit tests" workflow specifically
  build_run_id="$(echo "$runs" | jq -r '.[] | select(.workflowName == "All post-commit tests") | .databaseId' | head -1)"

  # If not found, fall back to other build workflow patterns
  if [ -z "$build_run_id" ]; then
    build_run_id="$(echo "$runs" | jq -r '.[] | select(.workflowName | test("build|Build|CI|build-wheels")) | select(.conclusion == "success") | .databaseId' | head -1)"
  fi

  if [ -z "$build_run_id" ]; then
    echo "No successful build workflow found for commit $commit_sha"
    return 1
  fi

  echo "Found successful build run: $build_run_id"

  # Download build artifacts - need to find the actual artifact name pattern
  echo "🔍 Looking for build artifacts..."

  # Get list of all artifacts for this run using GitHub API
  local artifacts
  artifacts="$(gh api repos/tenstorrent/tt-metal/actions/runs/"$build_run_id"/artifacts --jq '.artifacts[].name' 2>/dev/null || echo "")"

  if [ -z "$artifacts" ]; then
    echo "❌ Could not list artifacts for run $build_run_id"
    return 1
  fi

  echo "📋 Available artifacts:"
  echo "$artifacts"

  # Search for build artifacts - collect both TTMetal and eager-dist patterns
  local ttmetal_artifact=""
  local eagerdist_artifact=""

  if [ "$tracy_enabled" -eq 1 ]; then
    # Look for artifacts with profiler in the name
    ttmetal_artifact="$(echo "$artifacts" | grep "TTMetal_build_any.*_profiler_" | head -1)"
    eagerdist_artifact="$(echo "$artifacts" | grep "eager-dist.*profiler" | head -1)"
  else
    # Look for artifacts without profiler in the name
    ttmetal_artifact="$(echo "$artifacts" | grep "TTMetal_build_any" | grep -v "_profiler_" | head -1)"
    eagerdist_artifact="$(echo "$artifacts" | grep "eager-dist" | grep -v "profiler" | head -1)"
  fi

  # Download available artifacts
  local downloaded_any=false

  if [ -n "$ttmetal_artifact" ]; then
    echo "✅ Found TTMetal artifact: $ttmetal_artifact"
    if gh run download "$build_run_id" --repo tenstorrent/tt-metal --name "$ttmetal_artifact" --dir . 2>&1; then
      echo "✅ TTMetal artifact downloaded successfully"
      downloaded_any=true
    else
      echo "❌ Failed to download TTMetal artifact"
    fi
  fi

  if [ -n "$eagerdist_artifact" ]; then
    echo "✅ Found eager-dist artifact: $eagerdist_artifact"
    if gh run download "$build_run_id" --repo tenstorrent/tt-metal --name "$eagerdist_artifact" --dir . 2>&1; then
      echo "✅ Eager-dist artifact downloaded successfully"
      downloaded_any=true
    else
      echo "❌ Failed to download eager-dist artifact"
    fi
  fi

  if [ "$downloaded_any" = false ]; then
    echo "❌ No matching build artifacts found or downloaded"
    echo "   Tracy enabled: $tracy_enabled"
    echo "   Available artifacts:"
    echo "$artifacts"
    return 1
  fi

  # Debug: List what was actually downloaded
  echo "📂 Files in current directory after download:"
  ls -la . | head -10

  # Process downloaded artifacts
  local ttmetal_extracted=false
  local wheel_installed=false

  # Process TTMetal artifact (tar.zst extraction)
  if [ -n "$ttmetal_artifact" ] && [ -f "$ttmetal_artifact" ]; then
    echo "✅ Processing TTMetal artifact: $ttmetal_artifact..."
    if unzip -q "$ttmetal_artifact"; then
      echo "✅ $ttmetal_artifact unzipped successfully"
      rm -f "$ttmetal_artifact"

      # Extract the tar.zst archive
      if [ -f "ttm_any.tar.zst" ]; then
        echo "✅ Found ttm_any.tar.zst, extracting..."
        if tar --zstd -xf ttm_any.tar.zst; then
          echo "✅ TTMetal build artifact extracted successfully"
          rm -f ttm_any.tar.zst
          ttmetal_extracted=true
        else
          echo "❌ Failed to extract ttm_any.tar.zst"
        fi
      fi
    else
      echo "❌ Failed to unzip $ttmetal_artifact"
    fi
  fi

  # Process eager-dist artifact (Python wheel installation)
  if [ -n "$eagerdist_artifact" ] && [ -f "$eagerdist_artifact" ]; then
    echo "✅ Processing eager-dist wheel: $eagerdist_artifact..."
    if unzip -q "$eagerdist_artifact"; then
      echo "✅ $eagerdist_artifact unzipped successfully"
      rm -f "$eagerdist_artifact"

      # Find and install the wheel file
      local wheel_file
      wheel_file="$(find . -name "*.whl" -type f | head -1)"
      if [ -n "$wheel_file" ]; then
        echo "✅ Found wheel file: $wheel_file"
        echo "� Installing wheel with pip..."
        if pip install --force-reinstall "$wheel_file"; then
          echo "✅ Python wheel installed successfully"
          wheel_installed=true
          rm -f "$wheel_file"
        else
          echo "❌ Failed to install Python wheel"
        fi
      else
        echo "❌ No wheel file found in eager-dist artifact"
      fi
    else
      echo "❌ Failed to unzip $eagerdist_artifact"
    fi
  fi

  # Check if we got at least one artifact processed successfully
  if [ "$ttmetal_extracted" = true ] || [ "$wheel_installed" = true ]; then
    echo "✅ Artifact processing completed successfully"
    if [ "$ttmetal_extracted" = true ]; then
      echo "   - TTMetal build artifacts extracted"
    fi
    if [ "$wheel_installed" = true ]; then
      echo "   - Python wheel installed"
    fi
    return 0
  else
    echo "❌ No artifacts were processed successfully"
    return 1
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
      echo "⚠️  Artifact download failed, falling back to local build"
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
