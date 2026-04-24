# Cleaning a Merge-Polluted Branch

When a feature branch has accumulated merge commits from `origin/main` (or sub-branch merges), and especially when other people's PRs were cherry-picked or committed directly on the branch and later squash-merged to main independently, the PR diff on GitHub becomes polluted with hundreds of unrelated file changes.

A standard `git rebase -i` often fails in this situation because:
- Merge-conflict resolutions are baked into merge commits that rebase discards
- Squash-merged PRs have different SHAs than their branch counterparts, so git can't auto-deduplicate them
- The sheer number of conflicts (one per commit that depended on a merge resolution) makes interactive rebase impractical

This document describes a reliable alternative: **diff-apply with author-scoped file filtering**.

## Diagnosis

Symptoms of a merge-polluted branch:

```bash
# Shows merge commits in the branch history
git log --oneline --merges origin/main..HEAD

# Commit count is inflated (includes other people's merged PRs)
git rev-list --count origin/main..HEAD

# Files changed is far larger than expected
git diff --stat origin/main...HEAD | tail -1
```

To identify the noise sources:

```bash
# List unique authors (your commits vs. others that came through merges)
git log --format="%an" --reverse origin/main..HEAD --no-merges | sort -u

# Count commits by each author
git log --format="%an" origin/main..HEAD --no-merges | sort | uniq -c | sort -rn

# Check if "other people's" commits already landed in main (different SHAs)
git log --oneline origin/main | grep "<PR number>"
```

## Procedure

### 1. Create a backup branch

```bash
git branch backup-$(git rev-parse --abbrev-ref HEAD | tr '/' '-')-$(date +%Y%m%d)
```

### 2. Fetch latest main

```bash
git fetch origin main
```

### 3. Identify files YOUR commits actually touched

Filter commits by your author name(s) and extract the union of all files modified:

```bash
# Get full SHAs of your commits (adjust --author to match your git identity)
git log --format="%H" --reverse origin/main..HEAD \
    --no-merges --not origin/main \
    --author="yourGitUsername" > /tmp/my_commits.txt

# Extract the unique set of files those commits touched
cat /tmp/my_commits.txt \
    | xargs -I{} git diff-tree --no-commit-id --name-only -r {} \
    | sort -u > /tmp/my_files.txt

wc -l /tmp/my_files.txt  # Should be a reasonable number
```

**Important:** If you used multiple git author identities (e.g., a Copilot auto-commit used a different name), pass multiple `--author=` flags.

### 4. Create a clean branch from origin/main

```bash
git checkout -b <branch-name>-clean origin/main
```

### 5. Check out YOUR files from the backup

Instead of applying a diff (which struggles with new files), directly check out the file contents from the backup branch:

```bash
BACKUP=backup-<your-branch>-<date>

while read f; do
    git checkout "$BACKUP" -- "$f" 2>/dev/null || echo "SKIP: $f"
done < /tmp/my_files.txt
```

This stages all your files with their exact content from the branch tip.

### 6. Check for missing build-system files

The author-based file detection misses files that were only modified inside merge-conflict resolutions (not in any standalone commit). Common culprits:

- `sources.cmake` files (adding new `.cpp` files to build targets)
- `CMakeLists.txt` files (adding new subdirectories or targets)
- Header include changes needed by your new code

Find them by diffing the backup against your clean branch for build files:

```bash
git diff "$BACKUP" HEAD --name-only -- '*.cmake' '*/CMakeLists.txt'
```

For each file listed, compare the backup version against main to see if the change is yours:

```bash
# See what the backup has that main doesn't
git diff origin/main "$BACKUP" -- path/to/sources.cmake
```

If the diff shows additions related to your new files (e.g., adding `filesystem_utils.cpp` to a source list), apply that change manually.

### 7. Commit

```bash
git commit -m "Your commit message"
```

If pre-commit hooks modify files (e.g., clang-format), add the modified files and amend:

```bash
git add <modified-files>
git commit --amend --no-edit
```

### 8. Move the original branch pointer and force-push

```bash
git branch -f <original-branch-name> HEAD
git checkout <original-branch-name>
git branch -D <branch-name>-clean  # delete temp branch
git push --force-with-lease origin <original-branch-name>
```

### 9. Verify

```bash
# Should show only your files
git diff --stat origin/main...HEAD | tail -1

# Should show no merge commits
git log --oneline --merges origin/main..HEAD

# Should be 1 commit (or however many you want)
git rev-list --count origin/main..HEAD
```

## Gotchas

**Merge-resolution-only changes are invisible to `git diff-tree`.** If a file was only modified as part of resolving a merge conflict (never in a standalone commit), it won't appear in the author-filtered file list. The build will fail with linker errors or missing includes. Always check `*.cmake` and `CMakeLists.txt` files after the initial commit.

**Squash-merged PRs have different SHAs.** Git cannot detect that commit `abc123` on your branch is the same change as commit `def456` in main (the squash-merged version). The two-dot diff (`origin/main..backup`) correctly shows zero diff for these files only if the content is byte-identical, which it usually is for the squash-merged version. But if the branch had the pre-squash multi-commit version, the intermediate states may have left artifacts.

**The two-dot diff is the right one when branching from current main.** Since the clean branch starts at `origin/main`, `git diff origin/main..backup` gives exactly the delta needed. The three-dot diff (`...`) would include changes relative to the merge-base, which may be stale.

**Untracked files get staged by `git add -A`.** If you have local untracked files (`.cursor/`, editor backups, etc.), make sure to unstage them before committing: `git reset HEAD -- .cursor/ PLANDIR/` etc.

## When to use this vs. interactive rebase

| Situation | Approach |
|-----------|----------|
| Few merge commits, no squash-merged PRs | `git rebase -i origin/main` works fine |
| Many merge commits but no other authors' PRs | `git rebase -i` with `--no-reapply-cherry-picks` |
| Other authors' PRs on branch + squash-merged to main | **This procedure** (diff-apply with file filtering) |
| Branch is a total mess and you just want the final state | This procedure, squashed to 1 commit |
