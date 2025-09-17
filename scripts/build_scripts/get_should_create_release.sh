#!/bin/bash

set -eo pipefail

release_candidate=false

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --release-candidate)
      release_candidate=true
      ;;
    *)
      echo "Error: Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

# Function to check if a tag has "-rc" in its name
is_rc_tag() {
  tag_name="$1"
  if [[ "$tag_name" == *"-rc"* ]]; then
    # "True" value in bash, because it's a return code thing
    return 0
  else
    return 1
  fi
}

# Check if the script is invoked inside a git repository
if [ ! -d .git ]; then
  echo "Error: Not a git repository."
  exit 1
fi

# Check if there are any tags at the HEAD of the current branch
tags_at_head=$(git tag --points-at HEAD)

# Determine if any tags at the HEAD of the current branch have "-rc" in the name
all_rc_tags=true

for tag in $tags_at_head; do
  if ! is_rc_tag "$tag"; then
    all_rc_tags=false
  fi
done

if [ -z "$tags_at_head" ]; then
  all_rc_tags=false
fi

# We considered issues that could arise by replaying a release pipeline, but
# probably ok because we just redo jobs that failed only per tag...

# If there are any tags in release_candidate mode, then we should not generate
# any candidates because there are no releases created based off it
if $release_candidate; then
  if [ -z "$tags_at_head" ]; then
    echo "true"
  else
    echo "false"
  fi
else
  if [ -z "$tags_at_head" ]; then
    echo "Error: No tags at HEAD of repo"
    exit 1
  elif $all_rc_tags; then
    echo "true"
  elif ! $all_rc_tags; then
    echo "false"
  else
    echo "Error: We should never be coming into this branch, but made it for pattern matching practices"
    exit 1
  fi
fi
