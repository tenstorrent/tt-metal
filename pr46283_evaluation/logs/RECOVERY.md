# Recovery — PR #46283 evaluation

Written 2026-08-19. Read this first if a session ends unexpectedly.

## The short version

**Nothing important lives only in a chat session.** Every conclusion is in a committed file,
pushed to `origin/lserbedzija/pr46283-findings` on `github.com:tenstorrent/tt-metal`. The
transcripts below are for conversational context only.

## 1. Resume the session natively (try this first)

    claude --continue          # resume the most recent session in this project
    claude --resume            # pick from a list of past sessions

Run from `/localdev/lserbedzija` (the project dir the sessions were started in), otherwise the
picker will not show them.

## 2. If that fails, the transcripts are here

    pr46283_evidence/session_transcripts/
      2026-08-19_current_f0cd94f8.jsonl   6031 lines  (this session)
      2026-08-15_prior_80488796.jsonl     5485 lines  (the pipeline run)
      2026-08-11_earlier_5377abb4.jsonl   5077 lines  (earlier work)

Originals remain in `~/.claude/projects/-localdev-lserbedzija/`. These are copies, so Claude Code
cannot resume *from* them — they are for reading. One JSON object per line; to skim just the turns:

    python3 -c "
    import json,sys
    for l in open(sys.argv[1]):
        d=json.loads(l); m=d.get('message') or {}
        if d.get('type') in ('user','assistant') and isinstance(m.get('content'),str):
            print(f\"--- {d['type']} ---\n{m['content'][:2000]}\n\")
    " session_transcripts/2026-08-19_current_f0cd94f8.jsonl | less

## 3. What to re-read to rebuild context, in order

| file | what it is |
|---|---|
| `tt-metal-pr46283/PRSuggestions.md` | the deliverable: 8 points, 12 optimizations. Send this. |
| `tt-metal-pr46283/PRSuggestions.txt` | same words, plain text. **Source of truth** — the .md is generated from it. |
| `tt-metal-pr46283/TOOL_FINDINGS.md` | ~5600 lines, all 48 findings with evidence. The receipts. |
| `pr46283_evidence/RUN_PLAN.md` | stage-by-stage state of the pipeline run, plus the resolver test |
| `tt-metal/models/experimental/voxtral_tts/STATUS.md` | our own port's history and measurements |
| `tt-metal/models/experimental/voxtral_tts/tt/NOTES.md` | our port's per-module notes, where the 12 levers came from |

## 4. State as of writing

- **PR repo** `/localdev/lserbedzija/repos/tt-metal-pr46283`, branch `lserbedzija/pr46283-findings`
  — fully committed and pushed, 0 unpushed. Untracked: `tt_metal/impl/profiler/profiler.cpp.perfauto_bak`,
  a tool-generated backup, deliberately not committed.
- **Hand-port repo** `/localdev/lserbedzija/repos/tt-metal`, branch `lserbedzija/voxtral-pr46283-xfer`
  — 0 unpushed, but **4 uncommitted items** are the XFER-4 cross-comparison harness
  (`VOXTRAL_TOOL_BLOCK1=1` runs the PR's generated Block 1 in front of our Block 2/3). Still to commit.

## 5. Deliberate omissions, so nobody "fixes" them back in

- No listening-pass recommendation — an automated pipeline cannot do one.
- The measurement traps (uncalibrated absolute scores, short prompts being seed noise, the 30 s
  recogniser window, MCD failing its own self-test) are held back for a follow-up, not lost. They are
  in `TOOL_FINDINGS.md` and `STATUS.md` §6.59.
- No closing/contact line in the document, by choice.
