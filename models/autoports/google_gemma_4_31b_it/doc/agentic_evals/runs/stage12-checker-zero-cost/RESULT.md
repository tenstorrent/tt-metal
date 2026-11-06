# Zero-cost validity ranking candidate

Confirmed checker defect: strict positive device hours rejects real host-only validity repairs even though validity rows do not enter the savings/device-hour division. Minimal patch permits zero only for rows with unblocks; no other gate change.

Seven regression tests pass; original checker fails the legitimate zero-device validity case. Installed package unchanged. Review checker-zero-cost.patch; candidate source/tests and AUTODEBUG/AUTOFIX are included locally. No devices or stage evidence rewrite.

Null savings and verbose basis are genuine documentation contract gaps: fill numeric estimates honestly (mark guesses) and use basis exactly measured/calculated/guess, retaining explanatory text separately. Missing measurements may not be relabeled measured. The candidate still rejects those cases.

Run: `python -m unittest discover -s models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/stage12-checker-zero-cost/candidate/tests -v`. Apply only through an approved plugin update, never to the installed read-only package during this stage.
