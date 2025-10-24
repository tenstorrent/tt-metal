#!/bin/bash

# Iterative cherry-pick script for tt-metal project
# Usage: ./iterative-cherry-pick.sh

set -e

# Configuration
BASE_REF="v0.64.0-rc7"
TARGET_REF="main"
BRANCH_PREFIX="mbezulj/cherry-pick"
STATE_FILE=".cherry-pick-state"

# Path filters for tt-metal project
PATH_FILTERS=(
    # "ttnn/cpp/ttnn/operations/conv/"
    # "ttnn/cpp/ttnn/operations/pool/"
    # "ttnn/cpp/ttnn/operations/matmul/"
    # "ttnn/cpp/ttnn/operations/data_movement/"
    # "ttnn/cpp/ttnn-pybind/"
    "models/tt_cnn/"
    "models/experimental/panoptic_deeplab/"
    "models/experimental/oft/"
)

# Function to display usage
usage() {
    echo "Usage: $0 [--resume] [--reset]"
    echo ""
    echo "Options:"
    echo "  --resume   Resume from where we left off (default behavior)"
    echo "  --reset    Start fresh, ignoring any previous state"
    echo ""
    echo "This script will:"
    echo "  1. Find commits between $BASE_REF and $TARGET_REF affecting specified paths"
    echo "  2. Cherry-pick them one by one, creating branch per commit"
    echo "  3. Stop on conflicts for manual testing"
    echo "  4. Can be resumed to continue with next commit"
    exit 1
}

# Parse command line arguments
RESET_MODE=false
if [[ "$1" == "--reset" ]]; then
    RESET_MODE=true
elif [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
fi

echo "=== Iterative Cherry-pick Helper for tt-metal ==="
echo "Base ref: $BASE_REF"
echo "Target ref: $TARGET_REF"
echo "Branch prefix: $BRANCH_PREFIX"
echo ""

# Verify git references exist
if ! git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
    echo "Error: Git reference '$BASE_REF' does not exist"
    exit 1
fi

if ! git rev-parse --verify "$TARGET_REF" >/dev/null 2>&1; then
    echo "Error: Git reference '$TARGET_REF' does not exist"
    exit 1
fi

# Build the path filter arguments for git log
PATH_ARGS=""
for path in "${PATH_FILTERS[@]}"; do
    PATH_ARGS="$PATH_ARGS $path"
done

# Get all commits to cherry-pick (in chronological order)
echo "Finding commits that changed specified paths between $BASE_REF and $TARGET_REF..."
ALL_COMMITS=$(git log --reverse --pretty=format:"%H" "$BASE_REF..$TARGET_REF" -- $PATH_ARGS)

if [ -z "$ALL_COMMITS" ]; then
    echo "No commits found that modify the specified paths between $BASE_REF and $TARGET_REF"
    exit 0
fi

# Convert to array
readarray -t COMMIT_ARRAY <<< "$ALL_COMMITS"
TOTAL_COMMITS=${#COMMIT_ARRAY[@]}

echo "Found $TOTAL_COMMITS commits to cherry-pick"
echo ""

# Load or initialize state
CURRENT_INDEX=0
if [[ -f "$STATE_FILE" && "$RESET_MODE" == "false" ]]; then
    CURRENT_INDEX=$(cat "$STATE_FILE")
    echo "Resuming from commit index: $CURRENT_INDEX"
else
    echo "Starting fresh"
    echo "0" > "$STATE_FILE"
fi

# Show remaining commits
if [ $CURRENT_INDEX -lt $TOTAL_COMMITS ]; then
    echo "Remaining commits to cherry-pick:"
    for ((i=CURRENT_INDEX; i<TOTAL_COMMITS; i++)); do
        commit=${COMMIT_ARRAY[$i]}
        subject=$(git log --format="%s" -n 1 "$commit")
        echo "  $((i+1))/$TOTAL_COMMITS: $commit - $subject"
    done
    echo ""
fi

# Function to save state
save_state() {
    echo "$1" > "$STATE_FILE"
}

# Function to create branch name
create_branch_name() {
    local commit_hash="$1"
    local short_hash=$(git rev-parse --short "$commit_hash")
    echo "$BRANCH_PREFIX-$short_hash"
}

# Function to check and handle ongoing cherry-pick
handle_ongoing_cherry_pick() {
    if [ -d ".git/sequencer" ]; then
        echo "🔍 Detected ongoing cherry-pick operation..."

        # Check if it's an empty cherry-pick
        if git status --porcelain | grep -q "^."; then
            echo "  → There are staged/unstaged changes"
            echo "  → Manual intervention required to complete the cherry-pick"
            echo ""
            echo "Please resolve manually:"
            echo "1. Review the changes with: git status"
            echo "2. Add files if needed: git add <files>"
            echo "3. Continue: git cherry-pick --continue"
            echo "4. Or skip: git cherry-pick --skip"
            echo "5. Then run this script again"
            return 1
        else
            echo "  → Cherry-pick resulted in no changes (empty commit)"
            if git cherry-pick --skip; then
                echo "  ✓ Automatically skipped empty cherry-pick"
                return 0
            else
                echo "  ✗ Failed to skip empty cherry-pick"
                return 1
            fi
        fi
    fi
    return 0
}

# Function to cherry-pick single commit
cherry_pick_commit() {
    local commit="$1"
    local index="$2"

    echo "=== Processing commit $((index+1))/$TOTAL_COMMITS ==="

    # Get commit info
    local short_hash=$(git rev-parse --short "$commit")
    local subject=$(git log --format="%s" -n 1 "$commit")
    local branch_name=$(create_branch_name "$commit")

    echo "Commit: $commit ($short_hash)"
    echo "Subject: $subject"
    echo "Branch: $branch_name"
    echo ""

    # Check if branch already exists
    if git show-ref --verify --quiet "refs/heads/$branch_name"; then
        echo "Branch '$branch_name' already exists. Checking it out..."
        git checkout "$branch_name"

        # Check if we're in the middle of cherry-picking this commit
        if [ -d ".git/sequencer" ]; then
            echo "Found ongoing cherry-pick operation for this commit..."
            # Let the handle_ongoing_cherry_pick function deal with it
            return 3  # Special return code to indicate we should handle this differently
        fi

        # Check if this commit is already applied
        if git log -1 --pretty=format:"%s" | grep -q "$(git log --format="%s" -n 1 "$commit")"; then
            echo "✓ Commit already applied to this branch, skipping..."
            save_state $((index+1))
            return 0
        fi
    else
        # Create branch from HEAD (current position)
        echo "Creating branch '$branch_name' from HEAD..."
        git checkout -b "$branch_name"
    fi

    echo "Cherry-picking commit $short_hash..."

    if git cherry-pick -x "$commit"; then
        echo "✓ Cherry-pick successful!"
        save_state $((index+1))
        return 0
    else
        echo "⚠️  Cherry-pick conflict detected!"
        echo "📝 Commit: $short_hash - $subject"
        echo ""

        # Auto-resolve by taking incoming changes
        echo "Auto-resolving conflicts by taking incoming changes..."

        # Handle conflicts
        git status --porcelain | while read -r status_line; do
            if [ -n "$status_line" ]; then
                status_code="${status_line:0:2}"
                file_path="${status_line:3}"

                case "$status_code" in
                    "UU"|"AA"|"DD")
                        echo "  → Resolving $file_path (taking theirs)"
                        git checkout --theirs "$file_path"
                        git add "$file_path"
                        ;;
                    "DU")
                        echo "  → Accepting deletion of $file_path"
                        git rm "$file_path"
                        ;;
                    "AU"|"UD")
                        echo "  → Accepting addition/modification of $file_path"
                        git add "$file_path"
                        ;;
                    *)
                        echo "  → Auto-adding $file_path (status: $status_code)"
                        git add "$file_path"
                        ;;
                esac
            fi
        done

        # Check if cherry-pick is in progress and if it's empty
        if git status --porcelain | grep -q "^."; then
            # There are changes, try to continue
            if git cherry-pick --continue; then
                echo "✓ Conflict resolved and cherry-pick completed!"
                save_state $((index+1))

                echo ""
                echo "🛑 STOPPING for manual testing..."
                echo ""
                echo "Branch '$branch_name' has been created with the cherry-picked commit."
                echo "Please test your changes. If they work:"
                echo "  Run: $0"
                echo "If they don't work:"
                echo "  Fix the issues manually, then run: $0"
                echo ""
                echo "To start over: $0 --reset"
                return 1
            else
                echo "✗ Failed to continue cherry-pick"
                return 2
            fi
        else
            # No changes (empty cherry-pick), skip it
            echo "  → Cherry-pick resulted in no changes (empty commit)"
            if git cherry-pick --skip; then
                echo "✓ Skipped empty cherry-pick and continuing"
                save_state $((index+1))
                return 0
            else
                echo "✗ Failed to skip empty cherry-pick"
                return 2
            fi
        fi
    fi
}

# Check if there's an ongoing cherry-pick operation before starting
if ! handle_ongoing_cherry_pick; then
    echo "Please resolve the ongoing cherry-pick operation first."
    exit 1
fi

# Main cherry-pick loop
for ((i=CURRENT_INDEX; i<TOTAL_COMMITS; i++)); do
    commit=${COMMIT_ARRAY[$i]}

    result=$(cherry_pick_commit "$commit" "$i")
    exit_code=$?

    if [ $exit_code -eq 3 ]; then
        # Special case: ongoing cherry-pick detected, handle it
        echo "Handling ongoing cherry-pick operation..."
        if ! handle_ongoing_cherry_pick; then
            echo "Please resolve the ongoing cherry-pick operation first."
            exit 1
        fi
        # After resolving, save state and continue
        save_state $((i+1))
        echo ""
        continue
    elif [ $exit_code -ne 0 ]; then
        if [ $exit_code -eq 1 ]; then
            # Stopped for testing
            exit 0
        else
            # Failed with conflicts needing manual resolution
            exit 1
        fi
    fi

    echo ""
done

# All commits processed successfully
echo "🎉 All commits cherry-picked successfully!"
echo ""
echo "Branches created:"
for ((i=0; i<TOTAL_COMMITS; i++)); do
    commit=${COMMIT_ARRAY[$i]}
    branch_name=$(create_branch_name "$commit")
    short_hash=$(git rev-parse --short "$commit")
    subject=$(git log --format="%s" -n 1 "$commit")
    echo "  $branch_name: $short_hash - $subject"
done

# Clean up state file
rm -f "$STATE_FILE"

echo ""
echo "Process completed! You can now test each branch individually."
