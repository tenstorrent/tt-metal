---
name: tt-bringup-log
description: Read and write the per-model append-only bring-up log (bringup/<model>/LOG.md) with commit-id entries and the staleness rule. Use at session start, before quoting any measured number, and after every measurement, decision, retraction or failure.
---

# Bring-up log: how to read and write it

Last validated against commit: <none yet>

Why: in the DSV4 bring-up numbers outlived the commits and hosts that produced them and sent agents down wrong paths
(see `bringup/GUIDE_dsv4_lessons.md`, retractions). The log is the only place measured facts live; skills hold procedures.

File: `bringup/<model>/LOG.md` in the bring-up branch; append-only (never edit or delete an entry; supersede it with a new one).
The tool creates the file on the first entry. Commit the log with your work.

## Entry format (kit tool writes it; never hand-edit)
```
## E<yymmdd>-<hhmm>-<3hex>
- date / commit / branch / host / track / status (verified|inferred) / claim / evidence / paths / supersedes / data (JSON)
```
Ids are time-based so parallel branches do not collide. `claim` carries the number and its shape (layers, batch, ISL, dtype, flags/env).
`paths` = repo paths the claim depends on (the staleness trigger). `data` = machine-readable numbers (the scorecard reads it).
`inferred` = not measured at that commit (modelled, extrapolated, read from code, copied): never quotable as a fact.
A retraction is a new entry that `supersedes` the old one and names the contradicting evidence in `claim`.

## Staleness rule (the tool applies it; apply the same by hand if the tool is missing)
An entry is FRESH only if all hold: (1) its `commit` is an ancestor of HEAD; (2) no commit since touched its `paths`
(`git log --oneline <sha>..HEAD -- <paths>` empty); (3) it was not superseded; (4) status is verified, and the host was not in `bad_hosts.txt`.
Otherwise it is STALE: a hypothesis only. Re-measure before building on it. Never copy a stale or inferred number into a report,
a skill or a commit message without the words "stale, from <sha>".

## Reading at session start (run from the repo root)
```
K=models/common/bringup_kit
python $K/log_tool.py show  --log bringup/<model>/LOG.md --fresh-only     # what you may rely on
python $K/log_tool.py check --log bringup/<model>/LOG.md                  # every entry with its state: FRESH / STALE / SUPERSEDED
python $K/scorecard.py run --card bringup/<model>/scorecard.json --log bringup/<model>/LOG.md   # gate status; exit 0 only if all required gates PASS
```
By hand if the tools are missing: `grep -n '^## E' LOG.md` and the rules above. Read the tail plus every superseding entry and every `decision`-track entry.

## Writing
* Write an entry: after every measurement you may be asked about later, every decision made with the human, every failure
  (hang, OOM, wrong output) with its triage path, every retraction, and at every handoff (what is done, what is not, how to resume).
* Command (needs a CLEAN tree and HEAD == the branch tip, else it refuses; commit your work first, then log):
  `python $K/log_tool.py add --log bringup/<model>/LOG.md --track prefill --claim "..." --evidence <log path> --paths <dirs the claim depends on> --data '{"ttft_ms": 9358}' [--host 45] [--status inferred]`
  `--commit` only if the measured commit is not HEAD. `--allow-dirty` forces `inferred`.
* Commit the log entry in your own branch with your work; the coordinator merges logs (append-only conflicts: keep both sides, renumber).
* Numbers need: shape (layers, batch, ISL), flags, host, whether warm (first call in a session is slow: discard it), paired or not.
  A single unpaired run on one host is `INFERRED`-grade for any claim of "X% faster" (host-to-host differs up to 10%).
* Mark negative results too ("tried X, no gain, commit, log"): they stop the next agent repeating it.

## Do not
* Quote a number from memory, from a note file, or from another model's bring-up as if measured on this model.
* Write a verified entry for an uncommitted tree or without an evidence path (the tool refuses).
* Delete or rewrite earlier entries, even wrong ones: retract them.

Escalate to the human: when two FRESH entries contradict each other, or a gate result conflicts with a human-approved decision.
