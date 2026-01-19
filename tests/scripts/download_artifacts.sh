#!/usr/bin/env bash
set -euo pipefail

# Download build artifacts from GitHub Actions
# Usage: download_artifacts.sh <commit_sha> [--tracy]

die() { echo "ERROR: $*" >&2; exit 1; }

# Parse arguments
commit_sha=""
tracy_enabled=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --tracy)
      tracy_enabled=1
      shift
      ;;
    -*)
      die "Unknown option $1"
      ;;
    *)
      if [ -z "$commit_sha" ]; then
        commit_sha="$1"
      else
        die "Too many arguments"
      fi
      shift
      ;;
  esac
done

[ -n "$commit_sha" ] || die "Please specify commit SHA as first argument"

# Check required tools
if ! command -v gh >/dev/null 2>&1; then
  die "GitHub CLI (gh) not found"
elif ! command -v jq >/dev/null 2>&1; then
  die "jq not found"
elif ! gh auth status >/dev/null 2>&1; then
  die "GitHub CLI not authenticated"
fi

# Configuration
WORKFLOW_RUNS_LIMIT="${WORKFLOW_RUNS_LIMIT:-1000}"

# Look for workflow runs for this commit
echo "Searching for workflow runs for commit $commit_sha..."
runs="$(gh run list --repo tenstorrent/tt-metal --commit "$commit_sha" --json conclusion,databaseId,workflowName --limit $WORKFLOW_RUNS_LIMIT)"

# Check if gh command failed
if [ $? -ne 0 ]; then
  echo "ERROR: gh run list command failed:"
  echo "$runs"
  exit 1
fi

if [ -z "$runs" ]; then
  echo "No workflow runs found for commit $commit_sha"
  exit 1
fi

# First try to find "All post-commit tests" workflow specifically
build_run_id="$(echo "$runs" | jq -r '.[] | select(.workflowName == "All post-commit tests") | .databaseId' | head -1)"

# If not found, fall back to other build workflow patterns
if [ -z "$build_run_id" ]; then
  build_run_id="$(echo "$runs" | jq -r '.[] | select(.workflowName | test("build|Build|CI|build-wheels")) | select(.conclusion == "success") | .databaseId' | head -1)"
fi

if [ -z "$build_run_id" ]; then
  echo "No successful build workflow found for commit $commit_sha"
  exit 1
fi

echo "Found successful build run: $build_run_id"

# Download build artifacts - need to find the actual artifact name pattern
echo "Looking for build artifacts..."

# Get list of all artifacts for this run using GitHub API
artifacts="$(gh api repos/tenstorrent/tt-metal/actions/runs/"$build_run_id"/artifacts --jq '.artifacts[].name' 2>/dev/null || echo "")"

if [ -z "$artifacts" ]; then
  echo "ERROR: Could not list artifacts for run $build_run_id"
  exit 1
fi

echo "Available artifacts:"
echo "$artifacts"

# Search for build artifacts - collect both TTMetal and eager-dist patterns
ttmetal_artifact=""
eagerdist_artifact=""

if [ "$tracy_enabled" -eq 1 ]; then
  echo "DEBUG: Tracy enabled, looking for profiler builds..."
  # Look for artifacts with profiler in the name (TTMetal only)
  ttmetal_artifact="$(echo "$artifacts" | grep "TTMetal_build_any.*_profiler_" | head -1)"
  # Wheel artifacts (ttnn-dist) don't have profiler in the name
  eagerdist_artifact="$(echo "$artifacts" | grep -E "(eager-dist|ttnn-dist)" | head -1)"
else
  echo "DEBUG: Tracy disabled, looking for non-profiler builds..."
  # Look for artifacts without profiler in the name
  ttmetal_artifact="$(echo "$artifacts" | grep "TTMetal_build_any" | grep -v "_profiler_" | head -1)"
  eagerdist_artifact="$(echo "$artifacts" | grep -E "(eager-dist|ttnn-dist)" | head -1)"
fi

# Download available artifacts
downloaded_any=false

if [ -n "$ttmetal_artifact" ]; then
  echo "Found TTMetal artifact: $ttmetal_artifact"
  if gh run download "$build_run_id" --repo tenstorrent/tt-metal --name "$ttmetal_artifact" --dir . 2>&1; then
    echo "TTMetal artifact downloaded successfully $ttmetal_artifact"
    downloaded_any=true
  else
    echo "ERROR: Failed to download TTMetal artifact"
  fi
fi

if [ -n "$eagerdist_artifact" ]; then
  echo "Found eager-dist artifact: $eagerdist_artifact"
  if gh run download "$build_run_id" --repo tenstorrent/tt-metal --name "$eagerdist_artifact" --dir . 2>&1; then
    echo "Eager-dist artifact downloaded successfully $eagerdist_artifact"
    downloaded_any=true
  else
    echo "ERROR: Failed to download eager-dist artifact"
  fi
fi

if [ "$downloaded_any" = false ]; then
  echo "ERROR: No matching build artifacts found or downloaded"
  echo "   Tracy enabled: $tracy_enabled"
  echo "   Available artifacts:"
  echo "$artifacts"
  exit 1
fi

# Debug: List what was actually downloaded
echo "Files in current directory after download:"
pwd
ls -la .

# Cleanup function to remove temporary files
cleanup_artifacts() {
  echo "Cleaning up artifact files..."
  rm -rf "$ttmetal_artifact" "$eagerdist_artifact" 2>/dev/null || true
  find . -maxdepth 1 -name "*.whl" -type f -delete 2>/dev/null || true
}

# Process downloaded artifacts
ttmetal_extracted=false
wheel_installed=false

# Process TTMetal artifact (tar.zst extraction)
if [ -n "$ttmetal_artifact" ]; then
  echo "Processing TTMetal artifact: $ttmetal_artifact..."
  # gh run download automatically extracts the zip, look for tar file
  if [ -f "ttm_any.tar.zst" ]; then
    echo "Found ttm_any.tar.zst, extracting..."
    if tar --zstd -xf ttm_any.tar.zst; then
      echo "TTMetal build artifact extracted successfully"
      rm -f ttm_any.tar.zst
      ttmetal_extracted=true
    else
      echo "ERROR: Failed to extract ttm_any.tar.zst"
    fi
  elif [ -f "ttm_any.tar" ]; then
    echo "Found ttm_any.tar, extracting..."
    if tar -xf ttm_any.tar; then
      echo "TTMetal build artifact extracted successfully"
      rm -f ttm_any.tar
      ttmetal_extracted=true
    else
      echo "ERROR: Failed to extract ttm_any.tar"
    fi
  else
    echo "ERROR: Neither ttm_any.tar.zst nor ttm_any.tar found after download"
  fi
fi

# Process eager-dist artifact (Python wheel installation)
if [ -n "$eagerdist_artifact" ]; then
  echo "Processing eager-dist wheel: $eagerdist_artifact..."
  # gh run download automatically extracts the zip, look for wheel file
  wheel_file="$(find . -name "*.whl" -type f | head -1)"
  if [ -n "$wheel_file" ]; then
    echo "Found wheel file: $wheel_file"
    echo "Installing wheel with uv pip..."
    if uv pip install --force-reinstall "$wheel_file"; then
      echo "Python wheel installed successfully"
      wheel_installed=true
      rm -f "$wheel_file"
    else
      echo "ERROR: Failed to install Python wheel"
    fi
  else
    echo "ERROR: No wheel file found in eager-dist artifact"
  fi
fi

# Check if we got at least one artifact processed successfully
if [ "$ttmetal_extracted" = true ] && [ "$wheel_installed" = true ]; then
  echo "Artifact processing completed successfully"
  if [ "$ttmetal_extracted" = true ]; then
    echo "   - TTMetal build artifacts extracted"
  fi
  if [ "$wheel_installed" = true ]; then
    echo "   - Python wheel installed"
  fi

  cleanup_artifacts
  exit 0
else
  cleanup_artifacts
  echo "ERROR: No artifacts were processed successfully"
  exit 1
fi
