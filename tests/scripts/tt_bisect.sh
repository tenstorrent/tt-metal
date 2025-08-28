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
echo "ARCH_NAME: $ARCH_NAME"
echo "pwd: $(pwd)"

git cat-file -e "$good_commit^{commit}" 2>/dev/null || die "Invalid good commit: $good_commit"
git cat-file -e "$bad_commit^{commit}" 2>/dev/null  || die "Invalid bad commit: $bad_commit"

echo "Good: $good_commit"
echo "Bad : $bad_commit"
echo "PWD: $(pwd)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"
echo "Status:"
git status --porcelain=v1

echo "Starting git bisect…"
git bisect start "$bad_commit" "$good_commit"

echo "Environment (filtered):"
env | grep -E '^(TT_|PYTHON|CC|CXX|PATH)=' || true

found=false
while [[ "$found" == "false" ]]; do
  rev="$(git rev-parse --short=12 HEAD)"
  echo "::group::Building $rev"

  git submodule update --init --recursive --force

  # Use and clean the CPM cache that CMake will use
  export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-/work/.cpmcache}"
  rm -rf "$CPM_SOURCE_CACHE" build build_Release build_Debug
  mkdir -p "$CPM_SOURCE_CACHE"

  export CMAKE_ARGS="-DCPM_SOURCE_CACHE=$CPM_SOURCE_CACHE -DCPM_DOWNLOAD_ALL=ON -DCPM_USE_LOCAL_PACKAGES=OFF"

  build_rc=0
  ./build_metal.sh \
    --build-dir build \
    --build-type Release \
    --toolchain-path cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake \
    --build-all \
    --enable-ccache \
    --configure-only || build_rc=$?

  if [ $build_rc -eq 0 ]; then
    cmake --build build --target install|| build_rc=$?
  fi
  echo "::endgroup::"

  if [ $build_rc -ne 0 ]; then
    echo "Build failed; skipping this commit"
    git bisect skip
    continue
  fi

  echo "::group::Testing $rev"
  timeout_rc=1
  max_retries=3
  attempt=1
  while [ $attempt -le $max_retries ]; do
    echo "Attempt $attempt on $(git rev-parse HEAD)"
    echo "Run: $test"
    output_file="bisect_test_output.log"
    if timeout -k 10s "$timeout_duration_iteration" $test >"$output_file" 2>&1; then
      timeout_rc=0
      break
    else
      echo "Test logs:"
      cat "$output_file"
      timeout_rc=$?
      echo "Test failed (code $timeout_rc), retrying…"
      attempt=$((attempt+1))
    fi
  done
  echo "Exit code: $timeout_rc"
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
