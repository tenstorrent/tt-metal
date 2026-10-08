## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — rules, shapes, gate
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — drain loop: per-tile TensorAccessor write on NoC1 (+ NoC0 alt), flush per block, single static VC 1
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp — trid-pipelined input read (lookahead 4 blocks = 16 tiles over all 8 banks); reads already bank-spread
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE (DST-accumulated x*x + transpose_dest + sfpu column sum), x*gamma pre-pass, POST
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — block_size = DST count, output_cb = 2 rows, dual_noc_drain/DM_DYNAMIC_NOC
- tt_metal/hw/inc/api/dataflow/noc.h — async_write CUSTOM_VC path (vc reaches ncrisc_noc_fast_write's NOC_CMD_STATIC_VC)
- tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h — ncrisc_noc_fast_write (always VC_STATIC), cmd-buf ready poll, dynamic-noc cmd buf map (BRISC owns bufs 0/1)
- tt_metal/hw/inc/internal/dataflow/dataflow_api_common.h — NOC_UNICAST_WRITE_VC 1, MULTICAST 4, DISPATCH MCAST 5
- ttnn/.../ccl/all_gather/device/kernels/unicast_writer.cpp — precedent: unicast writes on VC NOC_UNICAST_WRITE_VC+1
- JIT-built brisc.elf of the writer (objdump) — drain loop ~50 instr/tile + spin on NOC_CMD_CTRL; observed ~147 cycles/tile
- parent profile_log_device.csv (reports/r03-b03-a02) via /tmp drain.py — per-shape drain ns/tile vs pack end
## Nodes consulted
- r03-b04-a02 — drain is per-core-bound (same per-core rate with 10 writers); suggested posted/deeper writes
- r01-b03-a03 — bank de-phasing helped reads, not the drain ("drain is NOT bank-bound")
- r02-b02-a01, r01-b02-a04, r01-b03-a04, r02-b01-a01, r02-b03-a01 — NoC/path choice for drain; all single static VC
- r03-b03-a01/a02, r03-b02-a01/a02, r03-b01-a02, r02-b02-a04 — current post-AG timeline (combine gap 0.38 µs, stick read 0.6 µs, drain tail 1-1.9 µs)
- r01-b02-a02, r01-b04-a01 — earlier drain throughput numbers (~200-260 GB/s then, ~380 GB/s now)
## Docs / external references
- tenstorrent/tt-isa-documentation (DeepWiki): NOC_CMD_CTRL returns ready once a VC is assigned; static VC field = buddy bit (13) + class bits (14-15); classes 0b00/0b01 unicast request, 0b10 broadcast, 0b11 response; different VCs may progress independently
