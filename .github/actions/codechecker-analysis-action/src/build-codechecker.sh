#!/bin/bash
set -eo pipefail
if [[ ! -z "$CODECHECKER_ACTION_DEBUG" ]]; then
  set -x
fi

echo "::group::Installing CodeChecker dependencies"
sudo apt-get -y update
sudo apt-get -y --no-install-recommends install \
  build-essential \
  curl \
  gcc-multilib \
  python3-dev \
  python3-venv
echo "::endgroup::"

echo "::group::Build CodeChecker locally"
if [[ "$CODECHECKER_WILL_USE_WEB_API" == "false" ]]; then
  # If the job is only running analysis, do not spend time with building the API stuff!
  echo "Building only 'analyzer' module..."
  pushd CodeChecker/analyzer
else
  echo "Building full CodeChecker package..."
  pushd CodeChecker
fi

make venv
source venv/bin/activate
BUILD_UI_DIST=NO make standalone_package
deactivate
echo "::endgroup::"

./build/CodeChecker/bin/CodeChecker analyzer-version
if [[ "$CODECHECKER_WILL_USE_WEB_API" == "true" ]]; then
  ./build/CodeChecker/bin/CodeChecker web-version
else
  echo "CodeChecker 'web' package not built."
fi

echo "PATH=$(readlink -f ./build/CodeChecker/bin)" >> "$GITHUB_OUTPUT"
echo "VERSION=$(./build/CodeChecker/bin/CodeChecker analyzer-version | grep 'Base package' | cut -d'|' -f 2 | tr -d ' ')" >> "$GITHUB_OUTPUT"
echo "GITSEVEN=$(./build/CodeChecker/bin/CodeChecker analyzer-version | grep 'Git commit' | cut -d'|' -f 2 | cut -c 2-8)" >> "$GITHUB_OUTPUT"

popd
