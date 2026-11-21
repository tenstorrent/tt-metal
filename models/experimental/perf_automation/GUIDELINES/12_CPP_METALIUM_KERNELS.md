# 12 · C++ Metalium Kernels — the raw-metal escape hatch {#cpp-metalium-index}

The terminal kernel rung, reached only after the tt-lang rung (GUIDELINES/11) is tried and the op
still has a material roofline gap. Both tt-lang and this path run through the SAME target —
`ttnn.generic_op` fed a `ProgramDescriptor` — so C++ is not more powerful in principle; it exposes the
full descriptor surface directly (explicit NOC/processor selection, circular-buffer choreography,
semaphores, multi-core orchestration) that the tt-lang DSL wraps a higher-level subset of. Reach here
only for what tt-lang cannot express or cannot make fast enough.

**SAFETY — this rung is dangerous, not just deep.** A bad Metalium kernel can WEDGE a device core
(deadlock the NoC); tt-lang/ttnn fail gracefully, raw C++ can hang. Device runs are subprocess-isolated
and timeout-bounded, so a hang is recoverable, but a hung run costs a full timeout. Prefer tt-lang; use
C++ only when it is provably necessary. A plain hand-written matmul usually just MATCHES stock
(NO-GAIN) — the win comes from a fusion or dispatch-collapse the op library can't express, not from
hand-writing being inherently faster (a prior hand-C++ matmul measured 3.3x SLOWER than stock).

## Author a C++ Metalium kernel via ttnn.generic_op {#cpp-metalium-kernel}
<!-- route
op_class: matmul,attention,eltwise,reduction,conv_pool
rank: time
lever_type: structural
-->

**Fires when** the op's bottleneck is kernel-level (roofline `regime_verdict` = kernel), the tt-lang
rung has a recorded attempt, and a material gap remains — OR tt-lang provably cannot express the op
(e.g. a mixed-dtype matmul tt-lang's MLIR won't lower, an op tt-lang can't legalize). The op must be
kernel-able (not a sealed native op).

**The API (verified, ttnn 0.65.1):** everything runs through
`ttnn.generic_op(io_tensors, program_descriptor)`:
- `io_tensors`: list of inputs then the PRE-ALLOCATED output tensor(s) last.
- `program_descriptor = ttnn.ProgramDescriptor(cbs=..., kernels=..., semaphores=...)`.
  - `cbs`: list of `ttnn.CBDescriptor(core_ranges=..., format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=, data_format=, page_size=, tile=)])` — one circular buffer per in/out/intermediate tile stream.
  - `kernels`: list of `ttnn.KernelDescriptor(...)` — typically three: a reader (data movement, NoC in), a compute, a writer (data movement, NoC out). Each carries its `.cpp` source, `config` (a `DataMovementConfigDescriptor` with explicit `processor`/`noc`, or a `ComputeConfigDescriptor`), `compile_time_args`, and `common_runtime_args`.
  - `semaphores`: `ttnn.SemaphoreDescriptor(core_ranges=, initial_value=)` for multi-core coordination.
  - `ttnn.ComputeConfigDescriptor` fields: `math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`, `bfp8_pack_precise`, `dst_full_sync_en`, `unpack_to_dest_mode` — the same knobs as a compute-kernel-config.

**The kernel bodies (reader / compute / writer `.cpp`) are the raw-metal part** — DO NOT invent them
from memory. TT-Metalium is Tenstorrent's low-level SDK: a kernel runs on the Tensix core's baby
RISC-V processors (data-movement cores drive the NoC in/out; the compute core drives the Matrix/Vector
engines), streaming tiles through circular buffers. ADAPT a canonical working example rather than
writing from scratch. Authoritative sources in this repo:
- `tt_metal/programming_examples/` — the official templates. Closest matches by op:
  - matmul → `matmul/matmul_single_core`, `matmul/matmul_multi_core`, `matmul/matmul_multicore_reuse_mcast`
  - eltwise → `eltwise_binary`, `eltwise_sfpu`, `sfpu_eltwise_chain`
  - reader/writer + sharding → `shard_data_rm/kernels/reader_sharded_rm.cpp`, `vecadd_sharding`, `vecadd_multi_core`
  - minimal reader/compute/writer triple → `add_2_integers_in_compute`, `hello_world_compute_kernel`, `hello_world_datamovement_kernel`
- `METALIUM_GUIDE.md` (repo root) — §"Register control and Data Flow within the Compute Kernels", §"Running code on device", §"Native tile based computing".
- `tech_reports/` — `op_kernel_dev`, `Programming_Mesh_of_Devices` (multi-device), plus the existing
  `*_device_operation.cpp` kernels under `ttnn/cpp/ttnn/operations/` as production templates to adapt.

Recall the layer split (from the TT-NN vs TT-Metalium distinction): TT-NN is the op library you've
already exhausted at the knob rungs; this rung drops to TT-Metalium — the raw kernel model — for what
the op library and tt-lang can't express.

**Multi-device:** for a mesh, use `ttnn.MeshProgramDescriptor` and follow
`tech_reports/Programming_Mesh_of_Devices`; the collective (all_gather / reduce_scatter) is a separate
ttnn CCL op, not part of the per-chip compute kernel.

**HARD CONSTRAINTS (same as tt-lang):** preserve the op's I/O dtype / layout / memory_config so the
emit-e2e stitching survives; **e2e PCC is the ONLY correctness authority** (per-component PCC is
unsound — it breaks stitching). Tile-align shapes; a compute grid with more cores than N-tiles can
deadlock idle cores.

**FORCE-TRY + SELF-VERIFY:** build + run the kernel through `ttnn.generic_op`, `check_pcc` (must be ok
AND verdict valid), `measure_candidate` (only a real gain that beats stock is a win — a matching or
slower kernel is NO-GAIN, revert it), then `record_kernel_attempt(op, 'cpp', measured_ms, beat_baseline)`
EVEN on a no-gain result (that records the rung as tried). A REJECTED/crashed measurement is never a win.
