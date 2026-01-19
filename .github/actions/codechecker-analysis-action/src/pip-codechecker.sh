#!/bin/bash
set -o pipefail
if [[ ! -z "$CODECHECKER_ACTION_DEBUG" ]]; then
  set -x
fi

# Check if CodeChecker is already installed and skip installation if requested
if [[ "$IN_VERSION" == "ignore" ]]; then
  echo "::group::Using pre-installed CodeChecker (ignore requested)"
  if ! command -v CodeChecker &> /dev/null; then
    echo "::error::CodeChecker not found in PATH but version='ignore' was specified"
    echo "Please ensure CodeChecker is installed before running this action with version='ignore'"
    exit 1
  fi
  echo "CodeChecker found at: $(which CodeChecker)"
  echo "::endgroup::"
else
  echo "::group::Installing CodeChecker $IN_VERSION from PyPI"
  if [[ "$IN_VERSION" == "master" ]]; then
    # The default branch name "master" is offered as a convenient shortcut for
    # fetching the latest release. Unfortunately, this might just be a release
    # candidate, which we do not wish to supply to automated production users
    # this eagerly...

    # Hack to get pip list us which versions are available...
    # (thanks, http://stackoverflow.com/a/26664162)
    pip3 install codechecker=="You_cant_be_serious_mate" 2>&1  \
      | grep "ERROR: Could not find a version"                 \
      | sed 's/^.*(from versions: \(.*\))/\1/'                 \
      | sed 's/, /\n/g'                                        \
      | grep -v 'rc\|a'                                        \
      | sort -V                                                \
      | tail -n 1                                              \
      >> "codechecker_latest_release.txt"

    IN_VERSION=$(cat "codechecker_latest_release.txt")
    echo "Selected CodeChecker version $IN_VERSION automatically."
    rm "codechecker_latest_release.txt"
  fi

  set -e

  pip3 install codechecker=="$IN_VERSION"

  pip3 show codechecker
  echo "::endgroup::"
fi

which CodeChecker
CodeChecker analyzer-version
CodeChecker web-version

echo "PATH=$(dirname $(which CodeChecker))" >> "$GITHUB_OUTPUT"
echo "VERSION=$(CodeChecker analyzer-version | grep 'Base package' | cut -d'|' -f 2 | tr -d ' ')" >> "$GITHUB_OUTPUT"
echo "GITSEVEN=$(CodeChecker analyzer-version | grep 'Git commit' | cut -d'|' -f 2 | cut -c 2-8)" >> "$GITHUB_OUTPUT"
