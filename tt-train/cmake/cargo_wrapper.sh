#!/usr/bin/env bash
set -euo pipefail

# Ensure we are invoked in the crate directory (tokenizers-cpp/rust)
# Remove lockfile so older Cargo (<=1.75) can regenerate a v3 lock.
if [[ -f Cargo.lock ]]; then
  rm -f Cargo.lock
fi

exec cargo "$@"
