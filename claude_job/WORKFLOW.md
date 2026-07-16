# Working with Claude — How to Keep Track

This folder (`claude_job/`) is the **durable memory** for tasks you do with Claude.
Chat history is disposable; a folder of markdown files is permanent. Anything important
should live in files here, not only in the conversation.

## Folder layout

```
claude_job/
  WORKFLOW.md            ← this file (how the workflow works)
  <task-name>/
    prompt.md            ← the task definition: the "what & why" (rarely changes)
    progress.md          ← running journal: what we tried, worked, failed, next step
    findings.md          ← keep-forever facts (working configs, commands, gotchas)
```

Example: `claude_job/glm5_t3k/` holds the GLM-5.1-on-T3K task.

- **prompt.md** — the brief. Write it once with Claude, edit rarely.
- **progress.md** — the most useful file for re-opening context later. Append a dated
  entry each time you stop. This is your trail of breadcrumbs.
- **findings.md** — stable conclusions you don't want to rediscover (the exact `tt-run`
  command, a working rankfile, a gotcha that cost you an hour).

## The loop you run each session

1. Point Claude at the task folder:
   > "Read `claude_job/glm5_t3k/prompt.md` and `progress.md`, then continue from the next step."
2. Do a chunk of work together.
3. Before stopping, have Claude append a dated entry to `progress.md`
   (what was done, what's next).
4. Next session: repeat from step 1.

This works in **any** new chat — even after history is gone — because the state lives in
files, not in the conversation.

## Re-opening context outside this chat

Run these from the repo directory (`/home/namvu/npu-k8s/third_party/tt-metal`):

- `claude --continue` — reopen your **most recent** session with full history.
- `claude --resume` — show a **list** of past sessions to pick from.

Don't rely only on session history (it can be summarized or lost). The robust path is
always: start fresh and tell Claude to read the task folder (step 1 above).

## Extras worth knowing

- **CLAUDE.md** (repo root) holds persistent project rules Claude reads every session.
  That's why it already knows device/lock conventions. You can add task rules there too.
- **Memory** — Claude can save cross-chat facts (e.g. "GLM5 T3K work lives in
  `claude_job/glm5_t3k`"). Useful, but the folder pattern is more visible and under your
  control.
- **Keep entries short.** A good `progress.md` entry is 3–6 lines: date, what you did,
  result, next step. You're writing notes to your future self, not a report.
