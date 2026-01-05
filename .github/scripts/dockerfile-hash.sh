#!/bin/bash
# Compute a hash for a Dockerfile and its COPY source files
# This is used for Docker image cache invalidation
#
# Usage: dockerfile-hash.sh <dockerfile> [extra-files...]
# Computes hash of: dockerfile + files from COPY statements + extra files
#
# The hash automatically includes:
#   - The Dockerfile itself
#   - All local files referenced in COPY statements (excluding --from= stages)
#   - Any extra files passed as arguments
#
# Example:
#   dockerfile-hash.sh dockerfile/Dockerfile .github/workflows/build-docker-artifact.yaml

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dockerfile> [extra-files...]" >&2
    exit 1
fi

DOCKERFILE="$1"; shift
declare -a FILES=("$DOCKERFILE")

# Add extra files passed as arguments
for f in "$@"; do
    [[ -f "$f" ]] && FILES+=("$f")
done

# Extract COPY source paths (skip --from=, skip ${VAR} refs, skip directories)
while IFS= read -r line; do
    # Skip COPY --from= which copies from other build stages
    [[ "$line" =~ --from= ]] && continue

    if [[ "$line" =~ ^[[:space:]]*COPY[[:space:]]+ ]]; then
        src="${line#*COPY}"

        # Remove flags like --chmod=xxx, --chown=xxx
        while [[ "$src" =~ ^[[:space:]]*--[a-z]+= ]]; do
            src="${src#*--*=* }"
        done

        # Trim leading whitespace
        src="${src#"${src%%[![:space:]]*}"}"

        # Handle JSON array format: ["source", "dest"]
        if [[ "$src" =~ ^\[ ]]; then
            src=$(echo "$src" | sed -E 's/^\[\"([^"]+)\".*/\1/')
        else
            # Standard format: take first word (source path)
            src="${src%% *}"
        fi

        # Skip variable references like ${VAR}
        [[ "$src" =~ \$\{ ]] && continue

        # Remove leading / if present (paths are relative to context)
        src="${src#/}"

        # Only add if it's a file (not a directory)
        [[ -f "$src" ]] && FILES+=("$src")
    fi
done < "$DOCKERFILE"

# Sort, dedupe, and hash all files
printf '%s\n' "${FILES[@]}" | sort -u | xargs cat | sha1sum | cut -d' ' -f1
