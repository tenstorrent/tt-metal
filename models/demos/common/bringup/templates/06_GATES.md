# Gate ledger (append-only)

| Gate | Phase | What it proves | Threshold | Measured | Verdict | Date (UTC) | Raw log |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Verdicts: `PASS` · `FAIL` · `PASS-WITH-DEVIATION` (needs a DEC) · `BLOCKED` (needs a risk entry
naming the blocker) · `NOT-RUN` (needs the reason).

---

### G-<NAME> — <what it proves>
- **Command:**
- **Mesh / device:**
- **Input distribution:** *(mandatory — never chosen to make a gate pass)*
- **Reference dtype policy:** *(mandatory — a reference that shares the device's rounding flatters the number)*
- **Threshold:** and where it came from
- **Noise floor (computed):** and the error ratio `(1-measured)/(1-floor)`
- **Measured:**
- **Negative control:** what you broke, and what it scored
- **Verdict:**
- **Raw log:** `raw/<GATE>_<UTC>.log`   *(gzip anything over the repo's file-size hook limit rather than trimming it)*
- **What this does NOT prove:**
