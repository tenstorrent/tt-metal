# Tenstorrent domain classes (tt-metal, tt-llk, and code built on them)

Add these to the universal floor (`classes-universal.md`) when the scope includes Tensix kernels, LLK, dataflow
kernels, or the host code that configures them. They come from real defects found and confirmed in these trees.
The repo packs (`packs/`) weight them by observed frequency. For synchronization hazards, the specialised audits
(`race-audit-all` and its nine sub-audits) go deeper. This list is the breadth pass, so hand a suspicious site to
the matching sub-audit rather than stopping at a shallow verdict.

HW-semantics claims (ordering, latency, what an instruction latches) are grounded in the ISA docs and the code that
emits the instruction, per the source ladder in `race-audit-all`. A missing doc page is never evidence of absence.

## Dataflow and NoC (`tt_metal/hw/inc/api/dataflow`, kernels under `ttnn/` and `models/`)
- `tt-cb-balance`: reserve/push or wait/pop counts that differ on some path (loop remainder, early exit, a branch
  taken by only some cores) → one side waits forever.
- `tt-cb-capacity`: a reserve or wait larger than the CB's configured page count, or a page size in the kernel
  different from the host's CB config.
- `tt-cb-access`: reading a CB before `cb_wait_front`, writing past the reserved pages, or using a read/write pointer
  after the push/pop that invalidated it.
- `tt-noc-data-before-credit`: a credit or semaphore sent to a remote core before the data it announces is committed.
  For an atomic credit, `noc_async_writes_flushed` means departure only. Committing it needs the write barrier
  (ack), except where both writes target the same memory kind on the same NoC/VC and the ordering doc guarantees
  it.
- `tt-noc-read-barrier`: consuming data from an async read before `noc_async_read_barrier`.
- `tt-noc-fanout`: a multicast destination count, or a semaphore wait target, that does not match the real number
  of receivers (harvested/odd grids, the sender included or excluded, remainder cores).
- `tt-noc-exit`: a kernel that exits with a non-posted write or atomic still in flight.
- `tt-noc-index`: a transaction issued on one NoC and flushed or barriered on another, or an object-API write
  drained with the global barrier instead of the object's own.
- `tt-l1-poll`: a hand-rolled poll of an L1 word updated by another core, without `volatile` and (on Blackhole) without
  `invalidate_l1_cache`. This is latent while the RISC L1 data cache is disabled by default, and a real hang once
  it is enabled.
- `tt-shared-scratch`: L1 scratch shared by the two data-movement RISCs, or by successive ops, without a handshake.

## Compute and LLK (`tt_metal/tt-llk`, `tt_metal/hw/ckernels`, compute kernels)
- `tt-tile-shape`: code that hard-codes 32x32 tiles, 4 faces or 16-row faces, and breaks on tiny or partial tiles
  (num_faces 1/2, face_r_dim < 16). Check MOP/replay loop counts, face loops and address increments. Trace
  `num_faces`/`face_r_dim` through EVERY layer: host args, compute API, LLK, and test harness. A layer that drops
  one silently runs the default.
- `tt-dest-capacity`: Dest tile counts that ignore the sync mode (half vs full) or fp32 accumulation (which halves
  capacity), or a loop whose parity or odd count leaves the Dest sections out of step.
- `tt-init-uninit`: an init without its matching uninit (or a reconfig without restoring), leaving state (address
  modes, counters, formats, packer or unpacker config) that the next op silently inherits. Check the order across
  fused ops.
- `tt-reconfig-drain`: a config or format rewrite with no drain of the unit still consuming the old config. Hand it
  to `reconfig-stall-audit` / `mmio-race-audit`.
- `tt-format-resolution`: a data-format decision taken from the wrong source. For example, fp32 Dest accumulation
  overriding the source format in a load mode; Int32 routed through a narrower source register; TF32/bfp truncation
  assumed exact; a dtype pair the unpacker, packer or math path does not support together.
- `tt-sfpu-const-clobber`: successive SFPU inits writing the same programmable constant registers, so the op that
  runs uses the last init's constants rather than its own.
- `tt-approx-guard`: a wrong range guard, approximation mode or special-value path in an SFPU function
  (exp/log/recip/sqrt/trig): NaN/Inf/negative/zero/large-magnitude inputs, per arch.
- `tt-arch-divergence`: WH, BH and Quasar copies of a file that differ. A fix applied to one arch and not its
  siblings is a candidate. But an identical line is NOT automatically the same bug: a handshake count or stall is
  only wrong relative to its counterpart on the SAME arch, so check both sides before porting a fix (a BH fix can
  regress WH).
- `tt-throttle-combination`: performance knobs (matmul throttle, fidelity, packer modes, approximation) whose legal
  combinations with tile shapes or formats are narrower than the API accepts.
- `tt-dvalid-balance`: the unpack MOP and the math MOP disagree on how many SrcA/SrcB dvalids are set and
  cleared for some (broadcast type x format x acc_to_dest x reuse) combination, so one side waits forever or reads a
  stale bank. Count both sides per branch. Hand it to `srcreg-bank-sync-audit` for the bank-flip verdict.
- `tt-stall-operand`: a STALLWAIT/SEMWAIT/semaphore-wait operand taken from the wrong constant group. The "what to
  stall" resource mask (`p_stall::STALL_*`) and the "what to wait on" condition (`UNPACK`, `PACK`, `MATH`,
  `SRCA_VLD`, ...) are different namespaces, and a value from one in the other's slot compiles fine.
- `tt-register-literal`: a hand-written register bit or mask literal that disagrees with its field macro, its
  comment, or the sibling definition of the same register (shifted bit, wrong width, wrong arch copy).
- `tt-dead-llk-assert`: an `LLK_ASSERT` or debug-only check that is the only thing guarding an invalid combination,
  and is compiled out in production builds.

## Host program construction (`ttnn/cpp/ttnn/operations`, `tt_metal/impl`)
- `tt-program-cache-key`: an op attribute, tensor property (dtype, layout, memory config, shard spec) or compile
  arg that changes the generated program but is missing from the program-cache hash.
- `tt-runtime-args-override`: a cache-hit override path that fails to update a buffer address or other runtime arg
  that differs between calls. This one is invisible on the first run and wrong on the second.
- `tt-work-split`: core/work splitting that underflows or misassigns remainder work: zero-work cores, the second
  core group, `num_cores > num_units`, a start id that is not zero.
- `tt-host-kernel-args`: compile-time or runtime args whose order, count or meaning differs between the program
  factory and the kernel that reads them, often after one side is edited. Also: a CB passed to a one-time
  `*_init_common`/`hw_configure` that the host does not create under the same compile-time condition.
- `tt-dispatch-field-width`: a host value packed into a dispatch/prefetch command field narrower than its range
  (check against the field's `*_MASK`), or a device reader that masks it back and silently wraps.
- `tt-shard-layout`: logic valid for interleaved tensors reused for sharded ones (height/width/block), or the reverse;
  padding and alignment on the last shard.
- `tt-hardcoded-arch-constant`: a literal grid bound, alignment, L1/DRAM size or core count in host or kernel code
  that is correct for one arch only. Compare it against every supported arch's soc descriptor and the HAL query
  (for example `hal.get_alignment`, the compute grid, the DRAM alignment).
- `tt-core-flavour-map`: code that handles several core kinds (Tensix, active ERISC, idle ERISC, DRAM, Quasar NEO)
  but takes a bound, address or constant from the wrong core's memory map, or shares mutable firmware state across
  harts that should be `thread_local`.
- `tt-jit-define`: a define the JIT build puts on a kernel's compile line that collides with an identifier in a
  kernel or LLK header, or a hand-rolled define list that diverges from the canonical JIT/HAL list for that core.
- `tt-mesh-coords`: device/mesh coordinate or fabric-routing math that assumes a single device, a specific topology,
  or a rectangular mesh.

## Tests and harnesses
- `tt-test-param-drop`: a test layer (Python harness → C++ test → kernel defines) that drops or hard-codes a
  parameter. The test goes green without exercising the case it names. A green namesake test does not prove the
  changed line ran: check the flag that gates it.
- `tt-test-arch-split`: per-arch helpers or signatures that differ, so one arch's test binds arguments wrongly or does
  not compile, and the case is skipped rather than run.
