# Fused-prefill watcher repair and selection

Independent `AUTOTRIAGE.md` proves a dedicated-NoC accounting violation in the
resident fused operation: NCRISC issues NoC1 W reads while BRISC checks NoC1 using
its private issued-read baseline. Completed reads from the other RISC explain the
assertion; the capture did not retain the exact outstanding/read counter pair.
The report was read and its factory, firmware exit predicate and allocator query
verified against current source. No watcher suppression or shared-kernel edit was
used.

A pure host C++ planner probe (`tests/multichip_prefill_plan.cpp`, linked to the
current `_ttnncpp.so`) enumerated 640 plans: 32 padded M sizes, K1024/K3072 and ten
L1 budgets. `planner_sweep.csv/log` show all 64- and 96-KiB plans use NoC0-only
reads and fit. At 128 KiB and above some plans return to split-NoC residency.
The 64-KiB bound additionally excludes residency algebraically for BF16 local
K>=1024, as explained by AutoTriage. It is exposed through setup-time
`get_memory_view(...).total_bytes_per_bank` and `l1_headroom`; no runtime host query.

Focused 96-KiB watcher run passed S1024 linear prefill, eight changing traced steps,
all-rank output/contribution PCC and recurrent/conv state checks:
`../watcher_budget_0.json/log`, `../watcher_budget_probe/generated/watcher/`.
Both 64- and 96-KiB families passed real-weight numerical and warmed-trace timing
probes on both layer kinds. Measured S1024 prefill milliseconds:

| Path | Linear | Full |
|---|---:|---:|
| constrained fused 64 KiB | 6.7282 | 6.6478 |
| constrained fused 96 KiB | 4.5252 | 4.6077 |
| separate minimal MM + RS + AG | 2.7402 | 2.7577 |
| separate path, L1 prefill inputs | 2.6894 | 2.7355 |

Choose the separate path with L1 inputs as the production default. Keep the
optional fused path constrained to the source-proven 64-KiB family, never the
resident split-NoC plan. Decode retains direct all-reduce and local sharded L1
residual/norm/MLP execution; measured ~0.324/0.305 ms for linear/full attention.
The resident fused operation itself is not claimed repaired. The model has a
scope-compatible correct alternative and an explicit safe fused configuration.
The final full contract, max-context, stress, stack, watcher and profiler battery
is running on the integrated selected source; results belong in the stage audit.

The original watcher aborted with exit134; all raw logs and inspector metadata
were copied to `original_run/` and `original_console.log`. The initial tt-triage
attempt used the default inspector path and retained only failure diagnostics;
the explicit saved-path attempt retained health evidence but no usable dispatch
state after abort. One bounded authorized reset completed0, all four chips were
listed and the all-chip trivial smoke passed (`reset.log`, `post_reset_list.log`,
`recovered_smoke.log`). No manual process kill or broad exception was used.

## Final acceptance

The final two-link implementation passed both complete watcher suites:
`../watcher_0.json/log` and `../watcher_3.json/log`, all11 logical lengths and
fresh-request trace replay. `../final_audit.json` and `../final_queue.log` record
all acceptance gates passing for source
`bc8386dc4e5c8f6fb18f24addca003e9cd37efe2b57341e6be9b06948aef88fc`.
Original failure evidence stays in `original_run/` and `original_console.log`;
`../pre_native/` and `../diagnostic_native_profiles/` retain superseded source and
measurements. Paths mentioned by the independent AutoTriage report describe its
inspection time; final root-level gate artifacts now refer to the repaired path.
