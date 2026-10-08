# SDPA autofix: working rules

- `LESSONS.md` is the bots' memory. Every agent prompt reads its tagged lines (`fixer.sh` `lessons_for <role>`; `~/.sdpa-watch/watch.sh` and `dryrun.sh` for the watcher).
- When a human corrects a bot: add one rule line to `LESSONS.md` (`- [roles] rule (example)`), fix the source hint / prompt / code that taught the wrong thing, re-check on the real case, then copy to the repo snapshot (`.sdpa-fix/`, `.sdpa-watch/` on `skrstic/sdpa-pipeline-watcher`), commit, push.
- Read `LESSONS.md` before changing prompts, the judge, triage, or the PR loop; don't reintroduce a mistake it records.
- The live copies (`~/.sdpa-fix`, `~/.sdpa-watch`) are what cron runs; the repo holds snapshots. Edit live, copy, commit.
