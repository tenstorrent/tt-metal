#!/usr/bin/env bash
# calculate_version.sh - Calculate the next semantic version from git tags and
# commit messages.
#
# Outputs the final version string to stdout.
#
# Ported from: .github/actions/calculate-version/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BUMP=""
TAG_TYPE=""
IGNORE_COMMITS=""

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") [OPTIONS]

Calculate the next semantic version based on git tags, commit messages, and
an optional forced bump type.

Options:
  --bump major|minor|patch|none   Force a specific version bump instead of
                                  auto-detecting from commit messages
  --tag-type rc|dev               Append a release-candidate or dev suffix
  --ignore-commits SHA[,SHA,...]  Comma-separated commit SHAs to exclude from
                                  bump analysis
  -h, --help                      Show this help

Output:
  Prints the calculated version string to stdout (e.g. v1.2.3, v1.3.0-rc1,
  v1.3.0-dev20260305).
EOF
    exit 1
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bump)           BUMP="$2";           shift 2 ;;
        --tag-type)       TAG_TYPE="$2";       shift 2 ;;
        --ignore-commits) IGNORE_COMMITS="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *) log_fatal "Unknown argument: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Find the latest semver tag
# ---------------------------------------------------------------------------

tags=$(git tag -l | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | grep -v '-' || true)
rc_tags=$(git tag -l | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+-rc[0-9]+$' || true)

if [[ -z "$tags" ]]; then
    latest_semver="v0.0.0"
else
    latest_semver=$(echo "$tags" | sort -V | tail -1)
fi

if [[ -z "$rc_tags" ]]; then
    latest_rc="v0.0.0-rc0"
else
    latest_rc=$(echo "$rc_tags" | sort -V | tail -1)
fi

log_info "Latest semver tag: $latest_semver" >&2
log_info "Latest rc tag: $latest_rc" >&2

# ---------------------------------------------------------------------------
# Step 2: Determine bump type
# ---------------------------------------------------------------------------

if [[ -n "$BUMP" ]]; then
    bump_type="$BUMP"
    log_info "Using forced bump type: $bump_type" >&2
else
    log_info "Analyzing commits since $latest_semver for version bump indicators" >&2

    if [[ "$latest_semver" == "v0.0.0" ]]; then
        all_commits=$(git log --oneline --pretty=format:"%H|%s")
    else
        all_commits=$(git log "${latest_semver}..HEAD" --oneline --pretty=format:"%H|%s")
    fi

    # Filter out ignored commits
    filtered_commits="$all_commits"
    if [[ -n "$IGNORE_COMMITS" ]]; then
        IFS=',' read -ra IGNORE_ARRAY <<< "$IGNORE_COMMITS"
        for ignore_sha in "${IGNORE_ARRAY[@]}"; do
            ignore_sha=$(echo "$ignore_sha" | xargs)
            [[ -z "$ignore_sha" ]] && continue
            filtered_commits=$(echo "$filtered_commits" | grep -v "^$ignore_sha" || true)
        done
    fi

    commits=$(echo "$filtered_commits" | cut -d'|' -f2)

    bump_type="none"
    if echo "$commits" | grep -q "(MAJOR)"; then
        bump_type="major"
    elif echo "$commits" | grep -q "(MINOR)"; then
        bump_type="minor"
    elif echo "$commits" | grep -q "(PATCH)"; then
        bump_type="patch"
    fi

    log_info "Determined bump type: $bump_type" >&2
fi

# ---------------------------------------------------------------------------
# Step 3: Calculate new version
# ---------------------------------------------------------------------------

if [[ "$TAG_TYPE" == "dev" ]]; then
    # For dev tags, pick the higher of (semver, rc core) then bump minor
    semver_no_v="${latest_semver#v}"
    IFS='.' read -r s_major s_minor s_patch <<< "$semver_no_v"

    rc_no_v="${latest_rc#v}"
    rc_core="${rc_no_v%%-*}"
    IFS='.' read -r r_major r_minor r_patch <<< "$rc_core"

    use_rc=0
    if   (( r_major > s_major )); then use_rc=1
    elif (( r_major == s_major && r_minor > s_minor )); then use_rc=1
    elif (( r_major == s_major && r_minor == s_minor && r_patch > s_patch )); then use_rc=1
    fi

    if (( use_rc )); then
        major=$r_major; minor=$r_minor; patch=$r_patch
    else
        major=$s_major; minor=$s_minor; patch=$s_patch
    fi

    minor=$((minor + 1))
    patch=0
    new_version="v${major}.${minor}.${patch}"
elif [[ "$bump_type" == "none" ]]; then
    new_version="$latest_semver"
else
    version_no_v="${latest_semver#v}"
    IFS='.' read -r major minor patch <<< "$version_no_v"

    case "$bump_type" in
        major) major=$((major + 1)); minor=0; patch=0 ;;
        minor) minor=$((minor + 1)); patch=0 ;;
        patch) patch=$((patch + 1)) ;;
    esac

    new_version="v${major}.${minor}.${patch}"
fi

# ---------------------------------------------------------------------------
# Step 4: Format final version with tag type suffix
# ---------------------------------------------------------------------------

if [[ "$TAG_TYPE" == "rc" ]]; then
    rc_count=$(git tag -l "${new_version}-rc*" | wc -l | tr -d ' ')
    new_number=$((rc_count + 1))
    final_version="${new_version}-rc${new_number}"
elif [[ "$TAG_TYPE" == "dev" ]]; then
    date_int=$(date +%Y%m%d)
    final_version="${new_version}-dev${date_int}"
else
    final_version="$new_version"
fi

log_info "Final version: $final_version" >&2
echo "$final_version"
