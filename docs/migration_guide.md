# Migrating tt-llk branches to tt-metal

This guide applies once the `tt-llk` submodule is absorbed into `tt-metal` as a regular subdirectory at `tt_metal/tt-llk/`. If you have an unmerged tt-llk branch, follow this guide to migrate your changes into a tt-metal PR.

## Prerequisites

- A local clone of `tt-metal` with the latest `main`
- A local clone of `tt-llk` with your feature branch checked out and `origin` up to date (`git fetch origin`)

## Steps

### 1. Rebase onto the absorbed commit

Rebase your tt-llk branch onto `origin/main`, which is the last commit absorbed into tt-metal:

```bash
cd path/to/tt-llk
git fetch origin
git rebase origin/main
```

Resolve any conflicts here, in the familiar tt-llk repo. This ensures the patch will apply cleanly to tt-metal in the next step.

### 2. Generate a patch

```bash
git diff origin/main...HEAD > /tmp/my-feature.patch
```

### 3. Apply the patch in tt-metal

```bash
cd path/to/tt-metal
git checkout main && git pull
git checkout -b my-feature

git apply --directory=tt_metal/tt-llk /tmp/my-feature.patch
```

The `--directory` flag prepends `tt_metal/tt-llk/` to every path in the patch automatically, and no manual path editing is required. Since you rebased in step 1, this should apply without conflicts.

### 4. Commit and push

```bash
git add -A
git commit -m "Migrate my-feature from tt-llk"
git push -u origin my-feature
```

Then open a PR targeting `tt-metal` `main`.

## Multiple concurrent PRs

Once migrated, these are normal tt-metal PRs. If another migrated PR merges before yours, rebase your tt-metal branch against `main` and resolve conflicts there, the same as any concurrent PR workflow:

```bash
git fetch origin
git rebase origin/main
# resolve conflicts if any
git push --force-with-lease
```

There is no need to go back to the tt-llk repo or regenerate the patch. After the initial migration, your branch lives entirely in tt-metal.

## Troubleshooting

### Dry run

To verify the patch applies without modifying any files:

```bash
git apply --directory=tt_metal/tt-llk --check /tmp/my-feature.patch
```

### Patch doesn't apply cleanly

If you skipped the rebase step or the absorbed commit has diverged, add `--3way` to fall back to a three-way merge:

```bash
git apply --3way --directory=tt_metal/tt-llk /tmp/my-feature.patch
```

Resolve any conflicts, then `git add` the resolved files and commit.
