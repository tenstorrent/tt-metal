# Decision log (append-only)

One block per judgement call, numbered monotonically. A superseded entry is never edited — a new
entry says `Supersedes DEC-NNN`.

A DEC is **mandatory** when you: pick a number not read verbatim from config; choose between repo
patterns; deviate from the recipe; lower a threshold or skip a case; add an env var; leave something
stubbed; or find that the reference and the repo disagree.

---

### DEC-001 — <one-line question>
- **Phase / module:**
- **Date (UTC):**
- **Trigger:** what forced the decision now
- **Question:**
- **Options considered:** 1. … (cost/benefit)  2. …
- **Choice:**
- **Why:**
- **Evidence:** `path:line`, a config key, or a measured number
- **Confidence:** high | medium | low
- **Falsifier:** what observation would prove this wrong
- **Revisit if:** the trigger that should reopen it
- **Blast radius:** files and gates affected if reversed
