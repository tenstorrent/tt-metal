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
#   - For extra files that are Dockerfiles (basename starts with "Dockerfile"),
#     also includes their COPY sources (transitive dependencies)
#
# Example:
#   dockerfile-hash.sh dockerfile/Dockerfile dockerfile/Dockerfile.tools
#   When Dockerfile uses FROM tool images built by Dockerfile.tools, passing
#   Dockerfile.tools as extra ensures tool install scripts are included in the hash.
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

declare -a FILES=()

# Add a Dockerfile and all files it COPYs (excluding --from= stages)
add_dockerfile_and_copy_sources() {
    local df="$1"
    if [[ ! -f "$df" ]]; then
        echo "Warning: Dockerfile not found, skipping: $df" >&2
        return
    fi
    FILES+=("$df")

    # Extract COPY source paths (skip --from=, skip ${VAR} refs, skip directories)
    # Handle multi-line statements with backslash continuations
    local accumulated_line=""
    while IFS= read -r line || [[ -n "$line" ]]; do
        local trimmed="${line%"${line##*[![:space:]]}"}"
        if [[ "$trimmed" == *'\' ]]; then
            accumulated_line+="${trimmed%\\} "
            continue
        else
            local full_line="${accumulated_line}${line}"
            accumulated_line=""
        fi

        [[ "$full_line" =~ --from= ]] && continue

        if [[ "$full_line" =~ ^[[:space:]]*COPY[[:space:]]+ ]]; then
            local src="${full_line#*COPY}"
            src=$(echo "$src" | sed -E 's/^[[:space:]]*(--[a-z]+(=[^[:space:]]*)?[[:space:]]+)*//')

            if [[ "$src" =~ ^\[ ]]; then
                src=$(echo "$src" | sed -E 's/^\[\"([^"]+)\".*/\1/')
            else
                src="${src%% *}"
            fi

            [[ "$src" =~ \$\{ ]] && continue
            src="${src#/}"

            [[ -f "$src" ]] && FILES+=("$src")
        fi
    done < "$df"
}

# Process primary Dockerfile (content + COPY sources)
add_dockerfile_and_copy_sources "$DOCKERFILE"

# Process extra files
for f in "$@"; do
    if [[ ! -f "$f" ]]; then
        echo "Warning: Extra file not found, skipping: $f" >&2
        continue
    fi
    if [[ "$(basename "$f")" == Dockerfile* ]]; then
        # Extra Dockerfile: add it and its COPY sources (transitive deps)
        add_dockerfile_and_copy_sources "$f"
    else
        FILES+=("$f")
    fi
done

# Sort, dedupe, and hash all files
# Use while-read instead of xargs to avoid ARG_MAX/sysconf issues in some environments
printf '%s\n' "${FILES[@]}" | sort -u | while IFS= read -r f; do cat "$f"; done | sha1sum | cut -d' ' -f1
