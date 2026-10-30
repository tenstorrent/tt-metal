# Prompt for the next overseer session (after the 2026-10-30 reboot of bh-glx-110-c10u02)

```
/bringup report on deepseek_v41_flash_d_p galaxy_8x4: continue as overseer of the seeded run (spec
models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml, node bh-glx-110-c10u02, branch
dnijemcevic/model_bringup). The node was just rebooted to free ~304 GiB of host RAM held outside any process.
Read supervision.md from "2026-10-30 10:30 HAND-OFF" to the end. First: report host RAM (free / available /
unaccounted, as in the hand-off) and run the quick box check. P.16's uncommitted change is still in the working tree
(also pushed as dnijemcevic/model_bringup_p16wip); rebuild ttnn if the build is stale. Before re-running P.16, fix F228
in the framework (an A/B side whose run did not finish must never be reused by decide --rerun; selftest + BREADCRUMBS),
then `decide --task P.16 --rerun` and `resume --until P.16`; decide by my rule (accept wins in speed or memory when
accuracy meets the bar). Then, through the framework: the e2e checks (runner smoke = Paris through tt-d-gen, contract,
KV read-back PCC, acks, layout) and the 50k->55k A/B profile. No fork regression suites; on any kill, read the logs
and rerun through the framework, never improvise. Set up the watch, heartbeat and stop guard.
```
