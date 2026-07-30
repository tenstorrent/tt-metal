# AutoDebug / AutoTriage / AutoFix reports

Where the `autodebug`, `autotriage` and `autofix` skills write their reports.

These used to be written to `./AUTODEBUG.md`, `./AUTOFIX.md` and `./AUTOTRIAGE.md` — the **repo
root** — because the skills said `./` and the working directory during DiffusionGemma work is the
checkout root. All three ended up committed as tracked root files, which is exactly what the
no-out-of-folder rule forbids. They were ported and deleted on 2026-07-30 and the skills now name
this directory explicitly.

Reports here are working artifacts, not documentation:

- A report is a snapshot of one investigation. It is **not** a source of truth, and it goes stale
  the moment the code moves. `autofix` is required to check whether a report describes the current
  symptom before starting from it.
- When a report produces a durable conclusion — a refuted hypothesis, a hardware rule, a measured
  number — that conclusion belongs in [`../REFUTED.md`](../REFUTED.md) or the relevant
  `../optimize_perf/` doc, **not** left in the report. Two findings had to be rescued out of the
  deleted root reports this way (the Blackhole DRAM NoC low-six-bits alignment rule, and the
  refuted GPT-OSS precision transfer).
- Feel free to delete a report once its conclusions have been promoted.
