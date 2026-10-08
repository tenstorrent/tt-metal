## Files read
- `agent_orch/WORKER.md`, `agent_orch/campaigns/rmsnorm-prefill/campaign.yaml` — process, metric, allowed paths.
- `tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py` — seq 640 -> 20 tile-rows, local
  widths 28/32/48/56 tiles, bf16 broadcast gamma, 1 link, Linear 1x4.
- `device/dit_fused_distributed_rmsnorm_program_factory.cpp` — `compute_sizing` (stats buffer geometry: pages =
  ring * forwarders * max_rounds), `create_at` (one worker per row, forwarder group of 20, CB sizing, CT/RT args,
  forwarder present_count per round). Added `pick_col_splits` used by both.
- `dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp` (read-only, outside allowed paths) — per round:
  wait cumulative arrivals += pc, fused fabric write of packet_buf[r%2] to page (dev, f, r), wait out_ready, go incs.
  Arrival count is cumulative, so a worker's round r+1 stick must only arrive after round r's packet left; the
  writer pushes round r+1 only after go(r).
- `device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp` — gamma streaming + poll_stick, push handshake,
  gathered-stick read, posted dual-NoC drain.
- `device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp` — trid-pipelined resident input pass.
- `device/kernels/compute/dit_rmsnorm_fused_compute.cpp` — mm_row_stat PRE, prescale x*gamma, packed combine
  (add_rsqrt on row 0 + transpose_dest), single-pass POST.
- `device/dit_fused_distributed_rmsnorm_device_operation.cpp`, `dit_fused_distributed_rmsnorm.cpp` — validate and
  create_stats_buffer both go through `compute_sizing`, so the split must be decided there from shape only.
- `tt_metal/hw/inc/api/semaphore.h` (`value()`), `api/dataflow/circular_buffer.h` (`pages_available_at_front`).

## Nodes consulted
- Every node's reflection (all 47, via `git show dream/.../n/<id>:.../reflection.md`), round 4 in full.
- r04-b04-a02 (parent) — timeline and remaining costs; `analysis/tl.py` here re-derives per-core zone medians from
  its profile (dev3 h7168: read end 5.84, push end 6.92, go 9.65, POST 10.89-14.63, drain end 16.13).
- r03-b04-a02 — two waves of 10 *cores* lost 5%: per-core PRE/drain rates stretched each wave. Its #4 ("waves need
  more cores than rows") motivated splitting the work instead of the cores.
- r01-b02-a01, r01-b03-a01/a02 — column split onto more cores: 34-stick packet cap / leader-combine hop cost.
  This node keeps 20 cores and 20 sticks per round.
- r04-b04-a03, r04-b02-a03 — cross-call launch skew and the post-AG drain tail (~40% of the kernel).
- r04-b01-a02/a03, r04-b02-a03 — HiFi2 PRE nodes, now forbidden (campaign rule: everything HiFi4).

## Docs / external references
- None beyond the code.
