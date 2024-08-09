#!/usr/bin/env bash
# Run clang-format over the entire repository and fix issues if specified by the user

set -eu

SUCCESS=0
FAIL=1
CLANG_FORMAT_BIN=${CLANG_FORMAT_BIN:-clang-format}

usage() {
    echo "-m | Select 'all' or 'diff': run formatter over all files or just modified ones"
    echo "-n | If present, do a dry-run (don't actually run clang-format)"
    echo "-f | If present, we should try to fix all issues automatically"
    echo "Usage: $0 [-m <diff|all>] [-n] [-f]" 1>&2
    exit $FAIL
}

log() {
    BOLD="\033[1m"
    DEFAULT="\e[0m"
    echo -e "${BOLD}[$(basename $BASH_SOURCE)] $1$DEFAULT"
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..
log "Running from $PWD"

MODE="diff"
DRY_RUN=false
SHOULD_FIX=false
while getopts "fnm:" OPT; do
    case "${OPT}" in
        f)
            SHOULD_FIX=true
            log "Will attempt to fix changes"
            ;;
        n)
            DRY_RUN=true
            ;;
        m)
            MODE=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

log "Using '$MODE' mode"

PATHSPEC=(
        ":/*.cpp" \
        ":/*.h" \
        ":/*.hpp"
    )

# Either use all the files that match certain patterns or use a diff from master
FILES=()
if [ $MODE == "all" ]; then
    mapfile -t FILES < <(git ls-files -- ${PATHSPEC[@]})
elif [ $MODE == "diff" ]; then
    # Use the base of our current branch if the user hasn't supplied one. If we are
    # on CI, we have to pass this in.
    BASE_COMMIT_SHA=${BASE_COMMIT_SHA:-$(git merge-base origin/main HEAD)}

    # Get all files that are different from main
    mapfile -t ALL_FILES < <(git diff --name-only $BASE_COMMIT_SHA -- ${PATHSPEC[@]})

    # Filter out files that don't exist since we've deleted them
    for FILE in ${ALL_FILES[@]}; do
        if [[ -e "$FILE" ]]; then
            FILES+=($FILE)
        else
            log "Could not find $FILE, it must have been deleted."
        fi
    done
else
    echo "Unexpected MODE encountered: $MODE"
    exit $FAIL
fi

if $DRY_RUN; then
    log "Dry run found ${#FILES[@]} file(s)..."
    for FILE in ${FILES[@]}; do
        echo "- $FILE"
    done
    exit $SUCCESS
fi

# Just succeed if there's no files to modify
if (( ${#FILES[@]} )); then
    FIX="-i"
    CHECK="--dry-run -Werror"
    log "Running formatter on ${#FILES[@]} file(s)..."
    if $SHOULD_FIX; then
        exit $($CLANG_FORMAT_BIN $FIX "${FILES[@]}")
    else
        exit $($CLANG_FORMAT_BIN $CHECK "${FILES[@]}")
    fi
else
    log "There was nothing to do. Exiting."
    exit $SUCCESS
fi
