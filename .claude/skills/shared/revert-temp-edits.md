# Reverting temporary source edits (shared procedure)

A reference for any skill that makes **temporary** source edits (kernel or host C++) to take a
measurement, then must remove them cleanly. Used by `dram-bytes-per-tile` and `core-occupancy`
(its optional `DeviceZoneScopedN` work-unit marker).

## The one rule

**Never revert with `git checkout <file>` / `git stash`.** The file may already contain the
user's *uncommitted* edits; reverting to HEAD would destroy them. Revert only **your own** edits,
back to whatever the file looked like **when your skill started** (committed or not).

## Mechanism: snapshot-then-restore

Before your **first** edit to any file, copy it verbatim to a backup and record it in a manifest.
The backup captures the current on-disk content — the user's uncommitted work included — so
restoring it returns the file to exactly its pre-skill state. Git is never consulted.

### Where things live

Use your **session scratchpad** directory (session-isolated, disappears with the session):

```
<scratchpad>/temp-edits/manifest.json      # the journal
<scratchpad>/temp-edits/<basename>.orig    # one verbatim backup per edited file
```

### Manifest schema

```json
[
  { "file": "<repo-relative path>",
    "backup": "<scratchpad>/temp-edits/<basename>.orig",
    "sha256_original": "<hash before any edit>",
    "sha256_after_edit": "<hash of what you wrote>",
    "needs_rebuild": false }
]
```

- **One entry per file**, created on first touch; later edits to the same file do NOT re-snapshot.
- **`needs_rebuild`**: `true` for host/C++ (`.cpp`/`.hpp` under `tt_metal/`, `ttnn/cpp/`, program
  factories) — reverting these requires `./build_metal.sh`. `false` for device kernels
  (`.../kernels/...`), which are JIT-compiled — no build needed on revert.

## Procedure

**On first edit to a file** (do this before the edit):
```bash
mkdir -p "<scratchpad>/temp-edits"
cp "<file>" "<scratchpad>/temp-edits/<basename>.orig"
sha256sum "<file>"   # record as sha256_original in the manifest
```
Then make the edit. **Tag every inserted or modified line with a marker comment** so it is
human-visible and greppable, e.g. `// [SKILLNAME temp]`. After editing, record `sha256_after_edit`.

**On revert** (reverse order; resumable — a fresh session can read the manifest and finish):
1. **Safety check**: `sha256sum <file>` now. If it ≠ `sha256_after_edit`, someone edited the file
   *after* your hack — **stop and warn**, show a diff, let the user decide. Do NOT clobber.
2. If it matches, restore: `cp "<backup>" "<file>"`.
3. Delete the backup.
4. If any restored file had `needs_rebuild: true`, run `./build_metal.sh` once at the end.
5. Delete the manifest when all entries are done.
6. **Verify**: `grep -rn "\[SKILLNAME temp\]" <touched files>` must return nothing.

## Guardrails

- **Refuse to start** if a backup can't be written (no silent partial state).
- The **marker-comment grep** is an independent "did I fully revert?" check — trust it in addition
  to the manifest, since it catches edits the manifest somehow missed.
- Tell the user **not to edit the touched files while the skill runs** (the snapshot is taken at
  skill start; the step-1 hash check is the backstop if they do).
