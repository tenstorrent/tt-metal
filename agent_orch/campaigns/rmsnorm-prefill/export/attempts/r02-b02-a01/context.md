## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml: job, metric, allowed paths.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: stick push, AG wait, gathered-stick DRAM reads, W_DRAIN
  loop (one 2 KB tile write per tile on the writer's NoC, flush + pop per block). This is the file I changed.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp: PRE (DST-accumulated x^2, reduce, transpose), x*gamma
  pre-pass, POST single pass (fp32 intermediate * bcast 1/rms, HiFi4).
- device/kernels/dataflow/dit_rmsnorm_fused_writer.cpp (drain-only TP=1 writer) and
  ccl/dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp (outside allowed_paths): the AG protocol
  (collect sticks -> fabric line mcast into DRAM scratch -> go-sem per worker).
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: kernel creation (ReaderDataMovementConfig /
  WriterDataMovementConfig), worker grid, compute config, CT arg layout.
- tt_metal/api/tt-metalium/kernel_types.hpp: on BH the DRAM-read NoC is NOC_0 (reader) and the DRAM-write NoC is
  NOC_1 (writer). Earlier reflections had the labels swapped.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml: DRAM channel -> 3 endpoints, dram_views worker_endpoint
  [noc0, noc1], channels 0-3 at x=0, 4-7 at x=9.
- tt_metal/impl/allocator/allocator.cpp: DRAM bank id == channel id.
- tt_metal/hw/inc/api/dataflow/noc.h, noc_traits.h, internal/tt-1xx/blackhole/noc_nonblocking_api.h: `Noc(uint8_t)`
  selects the NoC per object, and TensorAccessor addresses use that NoC's endpoint table. DM_DYNAMIC_NOC gives each
  RISC its own cmd buffers on both NoCs, with shared L1 counters.
- tt_metal/impl/program/program.cpp: all DM kernels on a core must share a noc mode, so the reader is dynamic too.
- tt_metal/impl/profiler/profiler.cpp: profiler core_x/core_y are physical NoC0 coords (used in the model).
- tt_metal/hw/inc/internal/tt-1xx/risc_common.h: my_x[] comes from NOC_ID_LOGICAL (translated coords).

## Nodes consulted
- All 16 r01 nodes (reflections). The most relevant:
  - r01-b04-a04 (round root): timeline, drain tail, "PRE tail is fixed".
  - r01-b04-a03: BRISC gamma read.
  - r01-b02-a04 (50/50 dual-NoC drain on this 20-core layout: 0.985; its "NoC1 half" was really NoC0 = long eastward
    wrap for left cores).
  - r01-b03-a04 (50/50 on the 80-core split: +3%, gradient flip).
  - r01-b03-a03: bank de-phasing; the drain is not bank-bound.
  - r01-b02-a02: flush batching is neutral.

## Analysis (scripts in analysis/)
- tails.py: per-shape POST and drain tail relative to AG end. The drain runs 1.8-3.6 µs past compute; the effective
  write rate is ~200-245 GB/s.
- gap.py / startspread.py: the cores that drained last in call N start call N+1 up to 1.4 µs late at h7168. The
  BRISC-FW restart takes ~1.1 µs after the kernel ends, plus ~0.7 µs to kernel start.
- nocmodel.py / rules.py / fluid.py / opt.py: link-load model of reads and writes on both NoCs. The chosen rule cuts
  the hottest write link from 4.25 B to 3.0 B.

## Docs / external references
- none beyond the in-tree sources above.
