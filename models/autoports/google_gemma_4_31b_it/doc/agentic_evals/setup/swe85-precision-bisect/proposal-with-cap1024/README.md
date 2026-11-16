# Candidate C panel copy with common 1024-token cap extensions

Status: proposal only; not admitted and not run. This directory preserves the fixed 108 onset/seed panel from `proposal-c`, with no subset selection. It applies the cap1024 extension to all 29 onset+seed pairs finalized as inconclusive at cap512; remaining rows retain their frozen caps (72 at 256 and 7 at 512).

Duplicate served/full cap1024 fixtures were deduplicated only after exact equality checks on prompt token IDs, chosen continuations, sampling metadata, and seed. The full-mode fixture is preferred when identical; provenance records both source modes and fixture hashes. The `candidate-cases.json` retains `branches: []`, one free-running trial per seed, and the same onset/seed identities.

The planning forecast uses exact saved per-case diagnostic wall for cap256 and cap512. At cap1024 it preserves measured cap512 prefill/host overhead and projects decode by 1023/511. It adds the observed served cap-extension load and a fixed 180-second first-preparation allowance; uncertainty scales case generation only. This is a planning estimate, not a performance or result claim.

Run command is recorded in `command.json` and is intentionally not executed. CPU-only schema validation passed (`cpu-validation.json`). No device was accessed.
