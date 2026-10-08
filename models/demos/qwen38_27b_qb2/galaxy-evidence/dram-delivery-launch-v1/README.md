# Bank-local read plus remote delivery: launch evidence

**Hardware results pending.** CPU: 355 tests and 40 subtests passed. Installed
semaphore descriptor construction also passed without opening hardware.

Unit `qwen38-dram-delivery-v1-20261008.service` started Oct 8 at 16:19:51 UTC,
PID 3360483 observed active. It waits for the existing GPQA unit to terminate
before acquiring `/tmp/tt-device.lock`. This is persistent across disconnects,
not an automatic reboot-resume workflow. The source snapshot and results are
under `/home/ttuser/qwen38-artifacts-20261007/dram-delivery-source-v1` and
`dram-delivery-v1`; the log is `dram-delivery-v1.log`.

Eight bank-adjacent workers read bulk packets, then deliver to eight separate
consumer workers per chip. Two cumulative semaphore counters enforce payload
visibility and prevent remote slot overwrite. The raw-reader control is the
same eight-page/four-slot path; before/after controls report timing drift.

- Three consumer placements, ring depths 1/2/4, packet sizes 8/15 pages.
- Full-byte equality at 8 and 2056 pages, independent live allocations and salts,
  no-delay and delayed consumers, three trace replays, all four physical ranks.
- Twenty-four timed cases at 272/544/1088 MiB per chip, with packet markers.
- Maximum 12-hour service, 10-hour dependency wait, 1-hour hardware/lock timeout,
  64-GiB host memory and 8 CPU equivalents; immutable source hashes.

This isolates redistribution and synchronization. It has no attention math,
production KV page table, or model integration. No new bandwidth or model gain
has been measured. The eight-consumer geometry is a diagnostic, not a claim
that it matches the attention compute grid.

Files: exact launch command/source hashes, CPU JUnit, queue snapshot, and live
service observation. `collection.json` records the observation time. The queued
controller records terminal process/JUnit/cleanup evidence before any pass.
