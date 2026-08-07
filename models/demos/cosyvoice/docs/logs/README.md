# Raw validation logs (R4)

Unedited pytest output, kept raw rather than summarised so every result can be checked
at source.

| file | tier | result |
|---|---|---|
| `device_blackhole_p150a_2026-08-05.log` | device — Blackhole `p150a` | 40 passed |
| `host_2026-08-05.log` | host, no hardware | 85 passed |

The device log is filtered for **kernel-compilation noise only** — `riscv-tt-elf-g++` command
lines and their `-Wdeprecated-declarations` output, which run to tens of thousands of lines and
say nothing about the model. Every test result, PCC value and timing is verbatim.

Reproduce with the commands in [`../../PERF.md`](../../PERF.md).
