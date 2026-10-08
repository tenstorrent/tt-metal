## Files read
- dit_fused_distributed_rmsnorm_program_factory.cpp: compute_sizing (wave_slots, wave_span, col_split_capable), the
  worker -> wave/slot/row mapping, the reader/writer RT args, and the forwarder CT args and slot table. All the
  wave-size assumptions are on the host side, apart from the forwarder's static wave_slots.
- kernels/dataflow/dit_rmsnorm_wave_forwarder.cpp: per-wave arrival threshold and go-release loop (wave_slots, now
  per wave).
- kernels/dataflow/dit_rmsnorm_fused_reader.cpp: the wave-B start-sem handshake (the wave_role, partner coords and
  wave_signal_block).
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: stick_off / pair_off / arrival_inc come from RT args only.
  No change needed.
- kernels/compute/dit_rmsnorm_fused_compute.cpp: gathered-tile indexing depends only on the pair layout, not on
  the wave size. No change needed.
- tt_metal/fabric/erisc_datamover_builder.hpp: fabric max payload is 4352 B, enough for the 22-slot span (3456 B).
## Nodes consulted
- r05-b01-a01 (parent, best 1.5696): per-wave timeline (waves_out.txt, fwd_out.txt). Its drains overlap and are
  write-aggregate-bound, and B's drain start is pinned by the full read end + its chain. That is the basis of the
  bandwidth model in proposal.md.
- r05-b03-a01: same 2-wave design. Its #3 suggested an uneven split (a smaller B). I argue for the opposite skew.
- r05-b01-a02/a03, r05-b03-a02/a03: 4 waves lose (serial go releases, AG under load ~4 µs). So I keep 2 waves and
  move only the boundary.
- r03-b04-a02: 10-core waves were per-core read-capped. That is the risk for an 18-core wave A.
- All round 1-5 reflections via history.md (fidelity / approx are forbidden; the HiFi2 nodes are forbidden_edit).
## Docs / external references
- none
