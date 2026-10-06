---
name: tt-bringup-log
description: Read and write the per-model append-only bring-up log (bringup/<model>/LOG.md) with commit-id entries and the staleness rule. Use at session start, before quoting any measured number, and after every measurement, decision, retraction or failure.
---

# Bring-up log: how to read and write it

Last validated against commit: <none yet>

Why: in the DSV4 bring-up numbers outlived the commits and hosts that produced them and sent agents down wrong paths
(see `bringup/GUIDE_dsv4_lessons.md`, retractions). The log is the only place measured facts live; skills hold procedures.

File: `bringup/<model>/LOG.md` in the bring-up branch; append-only (never edit or delete an entry; supersede it with a new one).
Create the directory and file on the first entry if missing.

## Entry format (one block per entry, newest at the bottom)
```
### E-<NNN> <UTC date> <track> <agent-name>
kind: measurement | decision | failure | retraction | hypothesis | handoff
commit: <12-char sha of the tree that produced the evidence>   (uncommitted tree = not allowed; commit first, even a WIP commit)
host: <last two octets, e.g. .45>   scope: <paths/functions the claim depends on, e.g. tt/moe_block.py tt/router.py>
claim: <one sentence, with the number and its shape: layers, batch, ISL, dtype, flags/env>
evidence: <log path under /mnt/tt-data/ssinghal/<model>-logs/..., test command, dump dir>
supersedes: E-<NNN> (optional)    status: FRESH | INFERRED
```
Rules: `INFERRED` = not measured (modelled, extrapolated, read from code). Env vars and flags that change behaviour go into the claim.
A retraction names the entry or document it retracts and the evidence that contradicts it.

## Staleness rule (apply to every entry you consider using)
An entry is FRESH only if all hold:
1. its `commit` is an ancestor of your HEAD: `git merge-base --is-ancestor <sha> HEAD`;
2. no commit since touched its `scope`: `git log --oneline <sha>..HEAD -- <scope paths>` is empty;
3. it was not superseded or retracted by a later entry;
4. status is FRESH (measured), and the host was not on the bad-host list (`tt-lab-ops`).
Otherwise it is STALE: use it only as a hypothesis and re-measure before building on it or quoting it. Never copy a stale number
into a report, a skill or a commit message without the words "stale, from <sha>".

## Reading at session start
```
python models/common/bringup_kit/log_tool.py fresh bringup/<model>/LOG.md --head HEAD     # prints FRESH entries only (intended CLI)
python models/common/bringup_kit/log_tool.py check bringup/<model>/LOG.md                 # lists entries by state: FRESH/STALE/RETRACTED
python models/common/bringup_kit/scorecard.py bringup/<model>/LOG.md                      # per-track gate status
```
If the kit tools are not in your branch yet, do it by hand: `grep -n '^### E-' LOG.md`, apply the 4 rules above with `git`.
Read the tail (last ~40 entries) plus every `retraction` and `decision` entry; those override older claims.

## Writing
* Write an entry: after every measurement you may be asked about later, every decision made with the human, every failure
  (hang, OOM, wrong output) with its triage path, every retraction, and at every handoff (what is done, what is not, how to resume).
* Command: `python models/common/bringup_kit/log_tool.py add --kind measurement --commit $(git rev-parse --short=12 HEAD) --host .45 --scope "tt/x.py" --claim "..." --evidence "..."` (intended CLI; manual append is fine).
* Commit the log entry in your own branch with your work; the coordinator merges logs (append-only conflicts: keep both sides, renumber).
* Numbers need: shape (layers, batch, ISL), flags, host, whether warm (first call in a session is slow: discard it), paired or not.
  A single unpaired run on one host is `INFERRED`-grade for any claim of "X% faster" (host-to-host differs up to 10%).
* Mark negative results too ("tried X, no gain, commit, log"): they stop the next agent repeating it.

## Do not
* Quote a number from memory, from a note file, or from another model's bring-up as if measured on this model.
* Write an entry for an uncommitted tree or without an evidence path.
* Delete or rewrite earlier entries, even wrong ones: retract them.

Escalate to the human: when two FRESH entries contradict each other, or a gate result conflicts with a human-approved decision.
