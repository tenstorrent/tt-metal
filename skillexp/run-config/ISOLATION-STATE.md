# advchal-v3 — cell isolation state of ~/tt-metal

Applied 2026-08-10, before the v3 run. Reason: **step 1's value is that phi `fuse-noadvise` reaches
−10.43 % by measuring it.** v2 published the answer for that exact cell, and it was readable from this
clone in one command — so a cell could have passed by looking it up.

## What was done

1. **150 refs whose trees hold `doc/advisor_challenger/` output were unnamed** (73 branches, 22 tags,
   55 remote-tracking). Recorded in `parked-refs.txt` as `<sha> <refname>`.
2. **`gc.pruneExpire=never` and `gc.auto=0`** on this clone, so unnaming a ref cannot lose its objects.
3. **Negative fetch refspecs** for `skillexp/{cell,run,wip,parked}/*`, `skillexp-cell/*`, `tmp-cellchk`
   and the v2 `done` tag namespaces, plus **`remote.origin.tagOpt=--no-tags`** so tag auto-following
   cannot re-import them. Verified: `git fetch --dry-run origin` restores none of them.
4. The prediction documents were moved off the skill branch to
   `mvasiljevic/qb2/skillexp/advchal-v3-docs`, so a cell's working tree no longer contains the number it
   is supposed to measure. That was the larger of the two channels.

## Reversing it

```
bash /home/mvasiljevic/skillexp-logs/advchal-v3/restore-parked-refs.sh
```

Offline, no network, restores all 150 exactly. **Tested both directions**: restored, confirmed v2's phi
`fuse-noadvise` `final.json` readable again, re-parked, confirmed unreadable.

## Residual — two refs left named, both pinned by a live worktree

| ref | worktree | holds |
|---|---|---|
| `mvasiljevic/qb2/skillexp/run/advchal-v2/fuse-noadvise/google_gemma_4_26b_a4b_it` | `~/tt-metal` itself | gemma-4-26B `fuse-noadvise` |
| `publish/phi-exp17` | `~/skillexp-book` | phi `exp17` |

Deleting either would orphan a worktree, and the first is the active checkout carrying uncommitted work,
so both were left alone. **Neither is step 1's cell** — phi `fuse-noadvise` is unreachable by name. They
are step-9 cells; if that matters when the run reaches them, detach the worktree HEAD and park the name.

And the standing residual by design: unnaming a ref does not delete objects, so anything is still
reachable by raw SHA. With the prediction documents off the skill branch there is no in-tree source of a
SHA, which is what makes that acceptable. Closing it entirely needs the per-cell run roots on
`mvasiljevic/qb2/skillexp/pipeline-isolation`, which are blocked on moving `~/tt-metal` behind a symlink.

## Also done

`chown 1000:1000` over 546 root-owned entries in `.git` — created by `docker exec` running git as root,
since these containers default to uid 0. Use `docker exec -u 1000:1000` for anything that writes git
objects, or it recurs.
