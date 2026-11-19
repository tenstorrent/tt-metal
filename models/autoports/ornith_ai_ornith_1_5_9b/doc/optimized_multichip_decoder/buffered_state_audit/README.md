# Buffered FP32 state source audit

Inspected `tests/multichip_buffered_state_kernel.py` with unchanged `kernels/fast_zero`. Source only; no edits, device calls, or hardware acceptance claims.

**No concrete ring-wrap or producer-ownership hazard found for CB5/CB7 depths16 or32.** CB5 remains NCRISC-producer/TRISC0-consumer. Each head reserves, reads, publishes, waits, and pops16 FP32 tiles. At capacity32, successive heads occupy tiles0–15 then16–31, with no transaction straddling the ring end. Reader destinations use the current `get_write_ptr(5)` and 4096-byte tile strides; compute addresses the current front. A depth32 permits prefetching the next disjoint head while the current one is consumed, not overwriting it.

CB7 remains TRISC2-producer/BRISC-consumer. Scan reserves/pushes16 and writer waits/writes/pops16. The optional depth32 follows the same two-slot phases. `api/compute/pack.h:38–87` specifies that sequential pack starts at the reserved write region and resets its internal cursor on push; explicit tile-index arguments are ignored in sequential mode. Therefore the packer's t0+j argument does not force later heads back into physical slot0. Writer uses current `get_read_ptr(7)`, so reads the matching slot. Reader/compute/writer use identical first/stride partitions and ordered per-head loops. No buffer changes producer or consumer.

The reader can prefetch the next state before blocking on prior gamma/Q CB reuse. Its input data are a different persistent head slab. Writer drains output then state, and compute's two buffers fit complete state publications, so increasing capacity introduces no dependency cycle. Each launch resets counters once; per-head loops do not restart hardware. Native B32/wave104 gives both 2- and3-head cores; wave7 is a stronger multi-wrap stress. Request boundaries do not affect head-slab independence.

Memory cost is +64 KiB per active core for CB5=32 versus16, plus another64 KiB if CB7=32. Allocation success, L1 contention with other live buffers, and any performance gain require measurement. Only capacities divisible by16 and at least16 satisfy this analysis; smaller or nonmultiple overrides would invalidate the contiguous transaction assumption. The current defaults32/16 and proposed32/32 satisfy it.

Required evidence: bitwise state/unchanged output direct comparison across multiple wraps (wave7,B32), alternating DRAM/L1 sources, watcher clean runs, and final decoder long gates. If selected, include allocation/trace coverage because per-core L1 use changed. Source hashes are recorded in `source_manifest.json`.
