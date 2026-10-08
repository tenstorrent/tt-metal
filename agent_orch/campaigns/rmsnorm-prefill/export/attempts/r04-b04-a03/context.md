## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job, allowed paths, metric.
- $DREAM_HOME/rmsnorm-prefill/history.md — index of all 43 nodes.
- kernels/dataflow/dit_rmsnorm_fused_reader.cpp — trid-pipelined resident input read: `kInputLookahead = 4` blocks of
  `block_size` tiles in flight, trids 1..14, input CB reserves the whole row (no CB back-pressure on the read).
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — stick push (flush-then-inc), streamed gamma chunks with
  poll_stick inside the gamma loop, posted dual-NoC drain; uses `my_x[0] < 9` as the left-of-DRAM-column test.
- ccl/dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp — F_COLLECT waits for every group worker's
  arrival inc, so the slowest stick push gates the AG start; go = serial unicast incs in slot order.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — reader/writer in DM_DYNAMIC_NOC, forwarder kernel path.
- Parent profiler log $DREAM_HOME/rmsnorm-prefill/reports/r04-b04-a02/.logs/profile_log_device.csv, analysed with:
  - analysis/tl.py — chip-level phase timeline (tl_parent_out.txt). h7168: read end max 6.56, push end 7.81,
    F_FABRIC end 10.41, go 10.58-10.93, POST start 12.18, drain end 17.72.
  - analysis/percore.py — per device x core table (percore_parent_out.txt): starts, read end, push, go, drain.
  - analysis/pushgate.py — read end vs W_GAMMA vs W_PUSH per core (pushgate_parent_out.txt): pushes run inside the
    gamma loop on every core; the gamma loop ends ~7.8 µs (off the path).
  - analysis/lr.py — left (x<9) vs right (x>9) group max of read end / push end vs F_COLLECT (lr_parent_out.txt,
    lr_r04-b01-a03_out.txt). The left group gates F_COLLECT on every shape and chip.
- Read rate check: per-core R_INPUT duration grows 2.6 µs from 28 to 56 tiles (~440-480 GB/s marginal per chip) with
  ~0.6 µs fixed start; aggregate over the whole read ~340-360 GB/s.
- r02-b02-a01 analysis/nocmodel.py + fluid.py (run on a copy in /tmp): a link-only model predicts NoC0 reads are
  uniform across cores, so the left/right asymmetry is not a simple link-load effect, and dual-NoC reads score worse
  in that model (not tried here).

## Nodes consulted
- All 43 committed nodes' reflections (r01..r04), plus the in-flight r04-b01-a03 (HiFi2 + writer stack, 1.4475),
  r04-b02-a03 (proposal) and r04-b03-a03 (bank de-phasing, neutral) from the sibling worktrees.
- r04-b04-a02 (parent) — the three writer wins; next-step list (launch skew, h4096 AG, dev3 straggler, HiFi2).
- r04-b01-a02 / r04-b01-a03 — HiFi2 PRE; the PRE tail at HiFi2 is a flat 0.5 µs, so read end then sets push time.
- r04-b03-a03 — per-core bank rotation for read/drain was neutral: lockstep bank phase is not the read limit.
- r03-b04-a02 — lookahead 8 lifted per-core read rate 54% in a 10-core wave (read depth matters per core).
- r03-b04-a03 / r04-b01-a01 — cross-call straggler loops (late finisher restarts late); seen again here as the left
  cores' ~1 µs late launch on h4096/h6144.
- r02-b02-a01, r02-b01-a01, r02-b03-a01, r01-b0x-a04 — dual-NoC/path experiments; reason for not touching NoC
  choice for the read here.

## Docs / external references
- none beyond the code.
