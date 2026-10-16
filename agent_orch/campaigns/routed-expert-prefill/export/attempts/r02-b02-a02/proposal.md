# r02-b02-a02: per-sender NoC request VCs for the weight DRAM reads ((column + group) & 3) to break the group-tail lottery

## Motivation
The parent (r02-b02-a01) is DRAM-bound on the big shapes (dsv3, kimi-k2.7, kimi-k3, glm, dsv4-pro at ~410-434 GB/s).
The few-expert shapes are not: mistral, minimax-m3, dsv4-flash and gpt-oss run one or two experts per group. On them,
row groups doing near-identical work finish far apart **within one call**, and the slowest group sets the op time.
Per-row TRISC1 kernel end of the parent (reports/r02-b02-a01, /tmp/routed_r02b02a02/grp.py), 10 measured calls:

- minimax-m3 (4 groups, one 5-6-tile expert each, pcM 3): per call the 4 group ends spread e.g.
  373/379/337/**434**, 398/362/322/332, 325/**388**/374/311. Mean group ~360 µs, op ~393 µs.
- mistral (same structure): e.g. 165/158/174/**202**, mean ~177, op ~191.
- dsv4-flash and gpt-oss show the same per-call scatter (gpt-oss 221-281 for one 8-9-tile expert per group).

The slow group changes from call to call, but within a call it stays slow. r01-b01-a03 (pacing) and r01-b04-a03
(deeper rings) showed this is not short-term jitter, and it is not compute (identical work, deterministic FPU). It
fits **unfair arbitration of the weight-stream DRAM requests**. Every weight read in this op (the reader's gate and
down reads on NoC 0, the writer's up and down-split reads on NoC 1, 5 sites through `WeightRuns::read`) uses
the default request VC (`NOC_UNICAST_WRITE_VC` = 1). tech_reports/Saturating_DRAM_bandwidth explains that requests
inside one VC are served first-come first-served, so one reader can get all its requests through while another
in the same row waits. The DRAM-sharded matmul fixes this by giving neighbouring readers different request VCs
(`vc = bank_id & 3`, bumped so that readers in the same row differ;
matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:728). Here, 11 column senders of a group sit in one
physical row and stream concurrently, and up to 5 groups' sender rows share the vertical links. Because a group's
11 columns are coupled in lock-step through the row x multicast, the most-starved column paces its whole group.

## Mechanism
- `kernels/weight_runs.hpp`: `WeightRuns::read` takes a request `vc` and issues every run with
  `async_read<NocOptions::CUSTOM_VC>(..., NocOptVals{.vc = vc})`.
- reader and writer compute one `weight_vc = (my_nt_gu + my_group) & 3` after the plan is built, and pass it to all
  5 weight read sites. Neighbouring columns of a sender row get different VCs, and so do the same column's senders in
  consecutive groups (their requests share the column's vertical links). VCs 0-3 are the unicast request VCs the
  DRAM-sharded matmul uses. Writes, multicasts and the x / bias / counts reads are unchanged.
- Kernel-only change (JIT, no host rebuild). The data is identical, so the numerics stay bit-identical.

## Why this is not a repeat
- r01-b01-a03 (pacing via a progress board) tried to equalise groups by throttling the fast ones. It added a
  per-block multicast on the critical sender and lost 30%. This change adds no instruction on the critical path. It
  only changes the arbitration class of requests already issued.
- r01-b04-a03 (3-slot rings) and the read-ahead nodes (r01-b03-a01, r01-b04-a01) changed buffering and read timing,
  not arbitration fairness.
- r01-b02-a04's reflection proposed exactly this probe (`read_req_vc` per row group). Nobody has run it.

## Expected effect and risk
- If the scatter is request-arbitration unfairness: the per-call max group end moves toward the mean on mm3 (up to
  ~-8%), mistral (~-6%), dsv4-flash and gpt-oss (~-3..-5%), and the 2-expert-group tails of dsv3/glm/kimi tighten a
  little. The DRAM-bound big shapes should be neutral, perhaps +-1-2% from changed DRAM efficiency.
- If arbitration is not the cause, everything is neutral within noise. I will judge by the per-group spread in
  profile_log_device.csv (max/mean of group ends per call), not only by µs.
- Risk: none to accuracy (same bytes, same order of consumption). Hang risk is very low: VCs 0-3 are the standard
  unicast request classes, already used for reads by the DRAM-sharded matmul. A slowdown is possible if VC 1 is
  somehow better served by the DRAM NIU than 0/2/3. That would show up as a uniform loss on the DRAM-bound shapes.
