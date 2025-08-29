#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

# Detect container (best-effort)
if [ -f /.dockerenv ] || grep -Eq '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null; then
  echo "Running inside a container"
  cid="$(awk -F/ '/(docker|containerd)/{id=$NF} END{print id}' /proc/self/cgroup 2>/dev/null || true)"
  [ -n "${cid:-}" ] && echo "Container ID: $cid"
  [ -f /etc/os-release ] && grep '^PRETTY_NAME=' /etc/os-release || true
else
  echo "Not running inside Docker"
fi




# Make in-tree modules win and avoid user site interference
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export PYTHONMALLOC="debug"
echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "ARCH_NAME: ${ARCH_NAME:-}"
echo "pwd: $(pwd)"
./create_venv.sh
./build_metal.sh \
  --build-all \
  --build-type RelWithDebInfo \
  --enable-ccache || build_rc=$?

echo "Checking environment variables:"
echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "ARCH_NAME: ${ARCH_NAME:-}"
echo "pwd: $(pwd)"
echo "ls: $(ls -la)"

python - <<'PY'
import ttnn, sys
print(ttnn.get_arch_name())
print("ttnn imported from:", ttnn.__file__)
PY

echo "::group::Testing $rev"
  timeout_rc=1
  max_retries=3
  attempt=1
  output_file="bisect_test_output.log"
  while [ $attempt -le $max_retries ]; do
    echo "Attempt $attempt on $(git rev-parse HEAD)"
    echo "Run: $test"
    if timeout -k 10s "$timeout_duration_iteration" bash -lc "$test" >"$output_file" 2>&1; then
      timeout_rc=0
      break
    else
      timeout_rc=$?
      echo "Test failed (code $timeout_rc), retrying…"
      echo "--- Logs (attempt $attempt) ---"
      sed -n '1,200p' "$output_file" || true
      echo "------------------------------"
      attempt=$((attempt+1))
    fi
  done
  echo "Final exit code: $timeout_rc"
  echo "::endgroup::"

: << 'END'
Usage:
  -f TEST        : test command to run (quote if it has spaces)
  -g GOOD_SHA    : known good commit
  -b BAD_SHA     : known bad commit
  -t TIMEOUT     : per-iteration timeout (default 30m)
END

timeout_duration_iteration="30m"
test=""
good_commit=""
bad_commit=""

while getopts ":f:g:b:t:" opt; do
  case "$opt" in
    f) test="$OPTARG" ;;
    g) good_commit="$OPTARG" ;;
    b) bad_commit="$OPTARG" ;;
    t) timeout_duration_iteration="$OPTARG" ;;
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

echo "Starting git bisect…"
git bisect start "$bad_commit" "$good_commit"

found=false
while [[ "$found" == "false" ]]; do
  rev="$(git rev-parse --short=12 HEAD)"
  echo "::group::Building $rev"

  fresh_clean

  build_rc=0
  ./build_metal.sh \
    --build-all \
    --enable-ccache || build_rc=$?

  echo "Checking environment variables:"
  echo "TT_METAL_HOME: $TT_METAL_HOME"
  echo "PYTHONPATH: $PYTHONPATH"
  echo "ARCH_NAME: ${ARCH_NAME:-}"
  echo "pwd: $(pwd)"
  echo "ls: $(ls -la)"

  # Find all ttnn*.so files
  echo "Searching for ttnn*.so files:"
  find . -type f -name '*ttnn*.so'

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
  timeout_rc=1
  max_retries=3
  attempt=1
  output_file="bisect_test_output.log"
  while [ $attempt -le $max_retries ]; do
    echo "Attempt $attempt on $(git rev-parse HEAD)"
    echo "Run: $test"
    if timeout -k 10s "$timeout_duration_iteration" bash -lc "$test" >"$output_file" 2>&1; then
      timeout_rc=0
      break
    else
      timeout_rc=$?
      echo "Test failed (code $timeout_rc), retrying…"
      echo "--- Logs (attempt $attempt) ---"
      sed -n '1,200p' "$output_file" || true
      echo "------------------------------"
      attempt=$((attempt+1))
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
      break
      ;;
    "")
      echo "git bisect produced no output; stopping to avoid an infinite loop."
      break
      ;;
  esac
done

git bisect reset || true
