# Recurrent compile-only repair

The first actual227 device attempt failed NCRISC compilation at the phase2 beta-fill loop: `-Werror=misleading-indentation` rejected the following `betas.push_back(1)` on the same line. The device closed cleanly; no numerical execution result exists for that attempt. Its JSON retains the earlier CPU-setup status and is not a device pass.

The new compilefix component inserts one newline before the push. Every C++ non-whitespace token is identical; all other Python/component AST is identical. The new probe is byte-identical to the cached probe after restoring the component filename and SHA. The original eb7f/00bec files and failed log remain unchanged. Exact hashes and restoration assertions are recorded in `v6_recurrent_six_stage_compilefix_transfer.json`.

Pinned autoflake/isort/Black made no changes. The CPU mapping/setup gate passed under the new source pins. Device compilation and all exactness/timing gates remain pending; this repair does not weaken any gate or change arithmetic, allocations, queue order, or fixtures.
