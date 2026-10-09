---
name: dream
description: Launch, follow and manage Dream-RSI optimization campaigns on tt-metal ops (agent_orch). Use when the user wants agents to optimize or speed up a tt-metal op automatically, wants to start/stop/resume a campaign, or asks for a campaign's status, report, best result or policies.
---

# /dream: Dream-RSI campaigns on tt-metal ops

A campaign runs Claude Code worker agents that propose, implement and measure
optimizations of one tt-metal op, while an exploration policy (which improves
itself between rounds) decides where attempts start. Everything is driven by
one CLI, `agent_orch/bin/dream`, in a tt-metal checkout that contains
`agent_orch/`. Read `agent_orch/README.md` there for the full reference.

Set `D=<tt-metal checkout>/agent_orch/bin/dream`. Find the checkout from the
current directory, the user's words, or `~/.config/dream/campaigns.json`
(`local_repo` of known campaigns). If none has `agent_orch/bin/dream`, ask.

Campaigns spend real money (model sessions) and hold a device for hours.
Never run `start`, `resume`, `delete` or a commit without the user's go-ahead
in this conversation. `check` is free of model cost (it builds and runs the
test on the machine) and fine to run once the spec is agreed.

## Start a new campaign

1. **Agree on the target** (ask only for what you can't find yourself):
   - the op, and from its code the `editable` globs (usually the op's directory);
   - the test that measures it. If none exists, write one: a pytest test that
     calls the op `warmup + measured` times per parametrized case and logs its
     accuracy (e.g. `PCC: <x>`), following `campaigns/rmsnorm-prefill/` and
     README §1. It is measured with `agent_orch/adapters/ttmetal_op_perf.py`;
     you need the op's `OP CODE` (the device operation class name in the ops
     perf CSV);
   - the machine: `local`, or `host:port` (an IRD reservation's ssh port).
     Check `ssh -o BatchMode=yes -p PORT HOST true` works; if not, help set up
     key auth first;
   - the budget (hard limits: `max_attempts`, `max_hours`, `max_usd`) and W/R/rounds.
     A worker attempt typically costs a few dollars and 20-45 minutes;
   - the starting policy: run `$D policies --json` and offer the choices with
     AskUserQuestion (name, one-line description, origin). `fresh` explores
     exhaustively (best first tree for dreaming); a learned policy prunes harder.
2. `$D init <name> --machine <m>`, then fill `agent_orch/campaigns/<name>/dream.yaml`
   and `brief.md`. Write the brief from the op's code: what it computes, which
   files hold the program factory and kernels, what the test measures, rules.
3. The campaign starts from the checkout's `HEAD` plus the campaign directory.
   The test must be committed. Show the user what would be committed and ask
   before committing.
4. Run `$D check <name>` in the background (the first build can take 30-60
   minutes). It prints the baseline per case and the noise band. If a case
   fails, fix the test or the eval command and run it again.
5. Summarize the baseline and the budget, and ask whether to start. Then
   `$D start <name>`.
6. Publish the report and follow it (next section).

## Follow a campaign (live report)

The campaign runs on its machine whether or not this session is open. The
report page is only republished while you follow it here.

1. `$D report <name> --out <scratchpad>/dream-<name>.html`, then publish that
   file with the Artifact tool (icon `chart`, a one-sentence description).
   Give the user the link. It is private until they share it from the page's
   Share menu; tell them so if the team should see it.
2. Start the Monitor tool on
   `$D watch <name> --out <scratchpad>/dream-<name>.html --interval 60`.
   Each `UPDATE ...` line means the file changed: republish the same file path
   (same URL, no icon) and, at most once per round, tell the user the
   headline (best score, attempts and money used). Don't narrate every line.
3. On the `FINAL ...` line: republish once more. If the state is `finished`,
   `watch` has already run `dream fetch`; tell the user the best node, its
   per-case result, and the branch `dream/<name>/best` with
   `git log -1 dream/<name>/best` / `git diff <base>..dream/<name>/best`.
   Offer to export the final policy to the library
   (`$D export-policy <name> --version vN --as <name>-vN --description "..."`).
   For `blocked` or `error`, show the message and the last log lines from
   `$D status <name>`, and propose the fix.

If the Monitor tool isn't available, refresh on request: `$D report` + republish.

## Manage

| User wants | Command |
|---|---|
| Status | `$D status <name>` (`--json` for details) |
| Pause / continue | `$D stop <name>` / `$D resume <name>` (resume continues the interrupted step) |
| The best change now | `$D finalize <name>` (writes and fetches `dream/<name>/best`) |
| Refs and attempts locally | `$D fetch <name>`; nodes are `refs/dream/<name>/n/<node>` |
| Read an attempt | `git show refs/dream/<name>/n/<node>:agent_orch/campaigns/<name>/attempts/<node>/reflection.md` |
| Share a learned policy | `$D export-policy ...`, then commit `agent_orch/policies/library/<new>/` |
| Remove a campaign | `$D delete <name> --yes` (only when asked; it deletes the refs and machine data) |

The budget is a hard stop: when a limit is reached, no new attempt starts and
the campaign finishes and writes its best branch. The spec is part of the
campaign's root commit, so limits can't be raised mid-campaign. To keep going,
start a follow-up campaign from `dream/<name>/best` (check it out, `init` a new
name, and pick the old campaign's last policy if it was exported).
