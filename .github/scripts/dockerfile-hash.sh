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

# Validate that Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    echo "Error: Dockerfile not found: $DOCKERFILE" >&2
    exit 1
fi

declare -a FILES=("$DOCKERFILE")

# Add extra files passed as arguments
for f in "$@"; do
    if [[ -f "$f" ]]; then
        FILES+=("$f")
    else
        echo "Warning: Extra file not found, skipping: $f" >&2
    fi
done

# Extract COPY source paths (skip --from=, skip ${VAR} refs, skip directories)
# Handle multi-line statements with backslash continuations
accumulated_line=""
while IFS= read -r line || [[ -n "$line" ]]; do
    # Handle backslash line continuations
    # Remove trailing whitespace for continuation check
    trimmed="${line%"${line##*[![:space:]]}"}"
    if [[ "$trimmed" == *'\' ]]; then
        # Line continues - remove trailing backslash and accumulate
        accumulated_line+="${trimmed%\\} "
        continue
    else
        # Complete line (either no continuation or final line of continuation)
        full_line="${accumulated_line}${line}"
        accumulated_line=""
    fi

    # Skip COPY --from= which copies from other build stages
    [[ "$full_line" =~ --from= ]] && continue

    if [[ "$full_line" =~ ^[[:space:]]*COPY[[:space:]]+ ]]; then
        src="${full_line#*COPY}"

        # Remove flags like --chmod=xxx, --chown=xxx using sed for robustness
        # This handles multiple flags and various whitespace patterns
        src=$(echo "$src" | sed -E 's/^[[:space:]]*(--[a-z]+(=[^[:space:]]*)?[[:space:]]+)*//')

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
