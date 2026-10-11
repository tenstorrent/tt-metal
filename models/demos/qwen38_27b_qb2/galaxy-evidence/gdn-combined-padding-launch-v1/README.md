# Combined resident GDN, compact gates and epilogue padding skip

Snapshot: October 11, 2026, 02:32:07 UTC. The combined full-model comparison
is running; no new full-model throughput or GPQA result is claimed here.

The opt-in policy `single_step_compact_gdn_resident_gates_padding` combines
the measured resident-state/compact-gate implementation with the separately
validated epilogue padding skip. Qualified defaults and numerical precisions
are unchanged. Small-batch fallback behavior is retained.

The frozen source passed 746 CPU tests, one skipped and 104 subtests. Hardware
test collection passed. The actual model integration then passed 4096
changing-input updates at B16 and B32 against qualified compact GDN on all
ranks, checking recurrent/conv state and projected outputs. Cleanup completed
before the full-model comparison began.

The persistent comparison measures B16/32K and B16/16K with before/candidate/
after controls, three repetitions plus warmup. It runs in user systemd with
the shared device lock, a 30-minute component timeout, 90-minute bound per
full-model arm, and a 24-hour supervisor limit. It survives client disconnect,
but is not configured to resume across a host reboot.

- Comparison: `qwen38-gdn-combined-padding-v1-20261011.service`, invocation
  `28043374fa9447778241fb4f4828de90`, PID 759228.
- Conditional follower: `qwen38-gdn-combined-padding-followup-v1-20261011.service`,
  invocation `3b91a375acef4a09b5cc4a9835bcc9ff`, PID 759231; waiting at capture.
- The follower recomputes matched measurements and requires at least 1 TSU
  absolute B16/32K gain before G0/API/full GPQA. It skips another full profile.
- The pre-run 20.8-20.9-TSU estimate is a projection, not a result. The best
  prior combined full-model measurement was 20.319 TSU.

`capture.json` records the timestamp, live unit properties, artifact hashes
and sizes. Frozen manifests and exact launch commands are retained. Logs and
JUnit are compressed without changing their content. Queue receipts are named
`queue-at-capture.json` to distinguish this snapshot from final status.

The missing CPU fixture from the prior follower is addressed by including
the tracked baseline full-model fixture in this frozen source. No active source,
native installation, NFS or firmware was modified; no AgentX run was launched.
See the [profiling audit](../../experiments/PROFILING-AUDIT.md) for what the
existing profiles do and do not establish.
