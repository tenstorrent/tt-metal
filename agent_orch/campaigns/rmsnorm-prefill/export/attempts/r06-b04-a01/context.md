## Files read
- agent_orch/WORKER.md, campaign.yaml: the job, the metric, allowed paths and accuracy gate.
- `$HISTORY` (history.md): every node's mechanism, score and next-step line, across rounds r01-r05.
- `device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`: the stick push, the streamed gamma, the go wait,
  the pair gather, and the W_DRAIN loop.
  - The drain issues tiles in column order with the static path-aware NoC rule (NoC0 only for short-eastward banks,
    every other visit).
  - It does a posted flush + pop per block.
- `device/kernels/dataflow/dit_rmsnorm_wave_forwarder.cpp`: the two-wave forwarder (F_SEND / F_GO). Serial go incs.
- `device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp`: the trid-pipelined resident input read on NoC0 and the
  wave-B start semaphore (lead 2 blocks).
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp`:
  - kernels in DM_DYNAMIC_NOC;
  - output_cb = 2 full padded rows;
  - block_size = dst_reg_count;
  - col_split forces one row per worker;
  - workers row-major, then the forwarder.
- tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:
  - dynamic-NoC posted-write counters: per-RISC L1 counters, summed by `ncrisc_dynamic_noc_posted_writes_sent`;
  - `NIU_MST_POSTED_WR_REQ_SENT`;
  - `noc_cmd_buf_ready`.
- tt_metal/hw/inc/internal/dataflow/dataflow_cmd_bufs.h: `write_cmd_buf` for BRISC in dynamic mode.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml: the DRAM endpoints per bank and NoC
  (worker_endpoint [noc0_sub, noc1_sub]).
- tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp: fabric NoC multicast write is unsupported
  (ASSERT), so the gathered sticks can't be multicast into worker L1 by the EDM. That idea was dropped.
- tests/.../test_fused_rms_norm_prefill.py: FABRIC_1D, Linear topology, one link.

## Nodes consulted
- r05-b01-a01 (parent/best): its reflection and analysis. My analyses run on its report:
  - `analysis/chain.py`: per-device wave chain;
  - `push.py`: last push → F_SEND lag, 0.5-1.0 µs at h7168 for wave A;
  - `cores.py`: per-core zones for one call;
  - `readpos.py`: per-core read and drain medians on one chip.

  Findings:
  - POST is a constant 1.97 µs per core at h7168, and the drain ends 0.7-2.3 µs after it.
  - The straggler drains depend on position (row y=4, x=12-14; x=6-7).
  - The reads are uniform.
- r05-b01-a03, r05-b03-a01/a03, r05-b04-a01, r05-b02-a01: wave and AG behaviour.
  - Each wave's AG is ~4 µs under load.
  - The go releases are serial.
  - Changing the PRE formulation is neutral.
- r02-b02-a01: the path-aware dual-NoC drain rule that this node keeps for eligibility.
- r02-b03-a01: drain stragglers are set by per-router arbitration into a DRAM column, not by hop count.
- r01-b02-a04: long NoC0 wraps are bad, so they stay ineligible.
- r03-b02-a03: two command buffers on one NoC hurt, so this node keeps one per NoC.
- r02-b02-a04: a multicast go doesn't help, so the release fan-out isn't worth chasing.
- r04-b03-a03: bank de-phase is neutral.
- Round-6 siblings r06-b02-a01 / r06-b03-a01 (proposals only): an uneven wave split, which is orthogonal.

## Docs / external references
- None beyond the in-tree headers above.
