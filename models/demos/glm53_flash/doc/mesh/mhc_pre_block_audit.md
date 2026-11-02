# mhc_pre_block NOC / dataflow audit (ledger A1) -- 2026-10-22, code only (no device opened)

Scope: `models/demos/glm53_flash/custom_ops/mhc_pre_block` (op.py, kernels/reader, writer, compute) in
`/home/ttuser/ashai/glm53/wt_mesh`, its invocation in `tt/mhc.py` (`_fused_pre_block`, `_pre`, `forward`), the
single-chip test `tests/custom_ops/test_mhc_pre_block.py`, the sibling `mhc_post_mix` (v2) and `kda_decode`
(semaphore patterns), the tt-metal mesh / dispatch / program-cache code the op's mesh safety rests on. Checklists:
tt-review-skills `ttnn-op-kernel-review` (10 categories) and `llk-race-audit-review` (9 classes + cross-invocation).

## 1. Verdict

| question | answer |
|---|---|
| Any NoC target or address only valid on one chip? | **No.** Every remote target is `(noc_x[x], noc_y[y])` from `worker_core_from_logical_core` tables built at op construction (op.py:163-164 before / `__init__` now): Blackhole *translated* coordinates, which depend on the logical grid only (umd `blackhole_coordinate_manager.cpp:103-147` maps logical column x to the x-th physical column's translated coordinate regardless of harvesting), and `MeshDevice::worker_core_from_logical_core` / `compute_with_storage_grid_size` throw unless every chip of the mesh agrees (`mesh_device.cpp:111-136, 912-917, 1464-1467`). Every remote address is a CB base (`get_write_ptr` of a CB nothing was reserved on, writer:66-69, reader:124) or a semaphore address (`get_semaphore(id)`): one address on every core of the program's single core range. Every op-held tensor (`fn`, `w`, `mask`, `consts`) is a `from_torch(device=mesh)` replica = one MeshBuffer address on every chip; `x`/outputs are mesh tensors whose `buffer_address()` is the lock-step MeshBuffer address. |
| Buffer addresses baked into compile-time args? | **No.** All eight addresses are runtime args (reader 0-3, 9; writer 0-2). Compile-time args carry CB ids, grid, semaphore ids, the NoC tables, the TensorAccessor configs. The generic-op program hash excludes runtime-arg values (`generic_op_device_operation.cpp:50-66`: `runtime_args.size()` only) and the adapter re-derives every runtime arg on each cache hit for a descriptor without buffer bindings (`mesh_device_operation_adapter.hpp:758-762`: slow-path `create_descriptor` + `apply_descriptor_runtime_args`), so the 90 `MhcPreBlock` instances sharing one cached program are safe (the same mechanism carries `mhc_post_mix` and `router_topk`, both validated e2e). |
| Gather tree / semaphores | **One real hazard (H1, below): an idle core's round-2 send is unordered.** Everything else is sound: GY/GX counts match the senders (every core of the range sends once per round, idle cores the zero tile), each waiter resets its own semaphore after the wait, round-2 increments are ordered after the round-1 reset through the pre-tile dependency (writers send round 2 only after `mix_h` consumed `CB_PRE`, which the leader forwards after its own reset: reader:141->152-157), all four semaphores are zero when a launch ends, and the dispatcher writes the initial values again at every launch (`dispatch.cpp:1405-1418`). A discrete-event model of the three kernels (test_mhc_pre_block_host.py) reproduces H1 on 13x10 and passes on 12x10 / 13x10 / 1x1 / 3x2 / 2x3 with the fix, including a second launch on un-reset state. |
| NOC barriers | **Complete.** Every remote write is followed by `noc_async_write_barrier()` before its semaphore increment (writer:87-88, 99-100, 109-112; reader:130-134); every CB push follows the read/zero barrier (reader:101-107, 111, 118); both data-movement kernels end with `noc_async_full_barrier()` (atomics acked before the next program's NOC init). The `Noc` helper and the raw `noc_async_*` calls share the per-NOC counters (`noc.h` is a thin wrapper over `noc_id_`), so mixing them is safe. |
| CB sizes vs tile counts | **Consistent** (table in section 3): receive buffers hold exactly one round (`GATHER1 == gy`, `GATHER2 == gx`, `PRE == RS2 == 1`, the invariant the base-address writes need), `PART == 2` covers a leader's own partial + column sum, `COEF_OUT == n+2`, X/FN/W/H/PARTS sized for the two-column cores. ~350 KB per core (382 KB exact variant) on every core. |
| Single-chip test vs mesh configuration | Shapes, dtypes, layouts, memory configs, constants and the norm weight are identical (section 4). The only configuration-dependent behaviour is the **grid**: the unit-test chip (device 1) and the rows 4-5 block both report **12x10 = 120 cores** (block tracy op list `_logs/mesh/report_prof2_l4_b4_stream/dev0_r0c0_decode_ops.csv`: the post-mix GenericOp on 120 of 120 available cores), so neither has idle cores and H1 cannot fire on the block. H1 fires on any 13x10 (or larger) chip or mesh. |
| Trace safety | Outputs preallocated, no host state in `__call__`, semaphores self-resetting; the lazy build in `_fused_pre_block` (device uploads + one device->host read) must stay in the eager warm-up (it does: `functional_decoder.capture_decode_trace` warms up before capture). |
| **What explains PCC 0.70 on the block?** | **Not a NOC / addressing / race hazard.** The fused run was deterministic and chip-identical (per-user chain PCC min = median = max = 0.701425 for all 8 users across 2 rows x 4 columns), which a NOC race or a per-chip addressing error would not give; H1 needs idle cores the block does not have. Two explanations remain, which the device test discriminates in one run: (a) the fused projection's rounding differs from the composite's (fused mixes 5.9e-5 vs composite 4.3e-5 from float64 on real rows; comb lane tiles 3.7e-4 from the Sinkhorn kernel on exact mixes) and the 45-layer bfp4 chain on the dump's *random* prompt is depth-chaotic (memory `glm53-pcc-metric`: chain PCC is not the gate; on the 8x4 the same prompt gave per-user chain PCC 0.75..0.83 across users under ulp-level differences) -- the step time (64.0 vs 79.3 ms = -15.3 ms) matches 90 wraps x (102 us device + ~5 us dispatch per removed op x 15 ops), i.e. the fused op ran as designed; (b) a deterministic model-level defect the unit test cannot see (it never compares the fused op with the composite path and never runs the model's integration). `test_mhc_pre_block_mesh.py` compares both paths per chip on the model's shapes through the model's own `_pre`; the logits test on natural text (`test_mesh_model_logits.py`, top-1/top-5) is the accuracy gate, not the random-prompt chain PCC. |
| **What explains the ARC hang?** | **Not attributable to the kernels.** No kernel path can address a non-worker core or an address outside L1 (coordinates table-driven and mesh-validated; addresses are CB / semaphore L1 addresses below 1.5 MB; no multicast rectangles); the run completed (test PASSED, 16 traced prompt steps + 20 perf replays) and the freeze surfaced at device close in `set_clock_state` (`preblock_ab2_pb1.log:485-515`), after which the next open timed out on the ARC queue; the identical board-wide signature occurred on 2026-10-20 21:41 with no custom op involved (doc/mesh/README.md:368-373). H1's lost-increment deadlock would have shown as a timeout *during* the run and needs idle cores. Open: treat as the board-level management-firmware freeze; the cheapest kernel-side exclusion is the watcher NOC sanitizer on one chip (step 0 below). |

## 2. Hazard table

Severity per `tt-review-core`: MUST-FIX = demonstrated at a site; SHOULD-FIX = class applies, invariant unstated;
CONSIDER = latent for the current configuration.

| # | sev | file:line (before the patch) | mechanism | consequence | fix (this patch) |
|---|---|---|---|---|---|
| H1 | **MUST-FIX** (latent on 12x10; fires on 13x10 and any grid with `cores > C/32`) | writer_mhc_pre_block.cpp:117 (`send_to_leader(cb_part)`) then :133 (`send_to_leader(cb_hsq)`) with `worker == false` (:22, :78-85) | An idle core sends the zero tile for round 1 and, with no data dependency in between, immediately for round 2: both land in slot `y` of the leader's GATHER1 and both increment ARRIVE1. The leader's round-1 `noc_semaphore_wait_min(GY)` (reader:140) is then satisfied with up to `#idle` real partials still in flight: it sums stale slots (garbage on the first launch, the previous launch's h^2 tiles afterwards). The late increments race the leader's `noc_semaphore_set(0)` (reader:141): one lost increment leaves round 2 one short. | wrong mixes -> wrong pre/post/comb/hn for every row, deterministically (idle cores always win: they do nothing else); occasionally a round-2 deadlock (hang). The single-chip test on a 12x10 chip has no idle core, so it cannot see it; it would fail loudly (1e-2 bars) on a 13x10 chip. | writer: an idle core waits for the pre tile (`CircularBuffer(cb_pre).wait_front(1)`, the reader's push after SEM_PRE) before its round-2 send -- the same dependency the workers have through `mix_h`. Modelled and verified host-side (13x10 now passes, including a second un-reset launch). |
| H2 | SHOULD-FIX (cross-invocation state) | compute_mhc_pre_block.cpp:96-102 | `pack_to` cached the last packed CB in a function-local `static`; it survives a launch whose binary stays resident in L1 and could skip `pack_reconfig_data_format` after another program changed the packer's format. Today harmless by luck (the last pack of every launch is CB_OUT, which `compute_kernel_hw_startup(CB_X, CB_OUT)` re-establishes). | silent wrong-format output on a cache-warm launch if the invariant ever breaks | per-launch cache `g_pack_last`, reset at `kernel_main` entry. |
| H3 | SHOULD-FIX (mesh test helper) | op.py:181 `golden()`: `ttnn.to_torch(self.w)` | A plain `to_torch` of a mesh-replicated tensor throws ("Supply a mesh composer") -- the same class as the norm-weight read-back fixed in d83e20b2094; the unit test's golden cannot run on a multi-chip mesh. | test-only failure on a mesh | the host copy `w_host` is kept and used by `golden` (mesh-safe). |
| H4 | SHOULD-FIX (binding of device-specific tables) | op.py:150-165 `__init__` | grid, work plan and NoC tables are taken from the device object at construction and never re-checked; an op built for one device (or sub-mesh) and called with tensors of another silently targets the wrong cores. Not the case in the model (one `MeshDevice`). | wrong cores / hang on misuse | `__call__` asserts `x.device().id() == self.device_id`. |
| H5 | CONSIDER (trace) | tt/mhc.py:825-846 `_fused_pre_block` | lazy build on the first fused call: three `from_torch` uploads and one device->host read (`get_device_tensors` + `to_torch`); inside a trace capture that is invalid. The decoder warms up eagerly before capture (`functional_decoder.py:552-556`), so it is a documented trap, not a defect. | corrupt capture on a future call pattern | docstring in `_fused_pre_block` and `MhcPreBlock.__init__`. |
| H6 | INFO | op.py:83, :154 ("120-core grid"), README.md:22 ("13x10") | stale, mutually contradictory grid comments; the work plan handles any grid with `>= C/64` cores (two column tiles per core), idle cores only beyond `C/32`. | none | comments / README rewritten; `plan_work` is a tested static method. |
| H7 | CONSIDER (L1 footprint) | op.py CB plan (all CBs on `self.cores` = the whole grid) | every core allocates the coordinator's and leaders' buffers too (CONSTS 32 KB, GATHER2 48 KB, Sinkhorn scratch 60 KB, COEF_OUT 24 KB): ~350 KB per core. A clash with the L1 allocator (persistent L1 tensors, CCL semaphores) fails loudly at enqueue (`validate_circular_buffer_region`), never silently. | launch failure in a deeper config | none now (a per-role `core_ranges` split would save ~170 KB per worker core; not minimal). |

Verified-safe items (so the table is a superset, per the race-audit contract): semaphore handshake counts and ordering
(section 1), per-launch semaphore init, NOC barriers before every push / increment, CB ownership (each CB has one
producer RISC and one consumer RISC: X/FN/W/MASK/CONSTS/GATHER*/PRE/RS2/ZERO reader->compute, PART/HSQ/COEF_OUT/
RS2OUT/OUT compute->writer, the rest compute-internal; the writer's `wait_front` on CB_PRE for idle cores reads the
shared counters only), control-flow balance (every `if (worker/leader/coordinator)` branch is mirrored on all three
RISCs with the same CB counts), tile_regs acquire/commit/wait/release balanced in every phase, init/reconfig before
every FPU/SFPU use (`reconfig_data_format*` + `*_init`), dest slots 0-7 within the 8-tile fp32 full-sync dest,
row-locality of every operation (pad rows cannot contaminate valid rows: FPU matmuls are row-local, every SFPU
routine works inside a row's 8-lane subvectors), the work split (every column tile exactly once, two-column cores first,
idle cores last), accessor / runtime-arg ABI between op.py and the kernels (host test), output specs (`hn` last in
`io_tensors`), `ReaderConfigDescriptor` (NCRISC, NOC1) / `WriterConfigDescriptor` (BRISC, NOC0) with `get_noc_addr`
doing the per-NOC coordinate encoding.

## 3. CB plan vs in-flight tiles (per core; the host test asserts the invariants)

| CB | tiles | bytes | producer -> consumer | max in flight | why the size |
|---|---|---|---|---|---|
| X | 8 fp32 | 32 K | reader -> compute (popped in `finish`) | 4 x count | two column tiles per core |
| FN | 8 bf16 (24 exact) | 16 K (48 K) | reader -> compute | 4 x count x frep | |
| HI/LO/LO2 | 4 bf16 each | 24 K | compute internal, per column | 4 | popped per column |
| ZERO | 1 | 4 K | reader -> compute, writer (never popped) | 1 | |
| PART | 2 | 8 K | compute -> writer | 2 (leader: own partial + column sum) | |
| GATHER1 | gy | 40 K | remote writers -> reader push -> compute | gy per round | slot y of the BASE; one round per push/pop |
| GATHER2 | gx | 48 K | leaders -> coordinator | gx per round | slot x of the BASE |
| PRE / RS2 | 1 | 4 K each | coordinator / leader -> reader push -> compute | 1 | the BASE is the only page |
| W | 2 bf16 | 4 K | reader -> compute | count | |
| CONSTS | 8 | 32 K | reader -> compute (coordinator) | 8 | |
| MIXES, SPRE, SPOST, SCOMB | 1 each | 16 K | compute internal | 1 | |
| MA, MB, RS, RECIP, TMP | 2 each | 40 K | compute internal | 1 | |
| COEF_OUT | n+2 = 6 | 24 K | compute -> writer | 6 | one push of 6 |
| H | 2 | 8 K | compute -> compute (`finish`) | count | |
| HSQ | 2 | 8 K | compute -> writer | 1 | |
| RS2OUT | 1 | 4 K | compute -> writer | 1 | |
| OUT | 2 bf16 | 4 K | compute -> writer | <= 2 | |
| PARTS | 2 | 8 K | compute internal | count | |
| DBG | 3 | 12 K | debug only | 0 | |
| MASK | 1 | 4 K | reader -> compute (exact only) | 1 | |

## 4. Single-chip test configuration vs the mesh model (what is and is not the same)

| item | unit test (`test_mhc_pre_block.py`, device 1) | model on the block (`test_mesh_model -k chain`, PB=1) | same? |
|---|---|---|---|
| x | `[1,1,32,n*C]` fp32 TILE DRAM interleaved, B=4/32 real rows + random pad rows | `_view_rows(x, 32)` of the residual streams: `[1,1,32,n*C]` fp32 TILE interleaved (the op asserts dtype / layout / not sharded; pad rows = whatever the previous op left, finite) | yes (memory config may be L1 vs DRAM: handled by the TensorAccessor and hashed into a separate program) |
| outputs | `allocate_outputs(x)` in `x.memory_config()` | same call | yes |
| fn / consts / w / mask | real layer-4 `hc_ffn_fn`, `build_consts`, random bf16-exact norm weight | the wrap's own fn / consts, the layer's bf16 norm weight read from shard 0 | yes (per-wrap values) |
| eps / iters | `kcfg.eps`, `kcfg.norm_eps`, `cfg.rms_norm_eps`, `kcfg.sinkhorn_iters` | `self.eps`, `self.norm_eps`, `float(eps)` of the norm tuple, `self.iters` | yes |
| grid | 12x10 (log `optmhc_preblock_unit1.log`: "grid 12x10") | 12x10 (block tracy: 120 available cores) | yes: 8 two-column cores, no idle cores |
| instances / cache | 1 instance, 1 call (trace test: 1 eager + 1 captured call on the SAME buffers) | 90 instances, 89 cache hits per step with different buffers | no: covered by the adapter's slow-path rebuild and by `test_two_instances_share_one_program` |
| consumers | none (host read-back) | `_post` -> post-mix v2 with lane-layout post/comb (`expand=False` by shape: `[32,32]`/`[32,128]`), collapse on the last layer | lane layout + collapse covered by `test_post_mix_v2_layouts[x8=True]` |
| reference | float64 math (bars 1e-2 / 5e-3) | the composite path (HF-identical digits at 16 K / 64 K) | **never compared directly before** -> `test_mhc_pre_block_mesh.py` |

## 5. Device test plan (exact commands; `cd /home/ttuser/ashai/glm53/wt_mesh && source ../env.sh` first)

Common: `export W=/home/ttuser/ashai/glm53/tt-metal WT=/home/ttuser/ashai/glm53/wt_mesh PYTHONPATH=$WT:$W/ttnn:$W TT_METAL_HOME=$W ARCH_NAME=blackhole`;
block env: `BLOCK="TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4"`.
Gate every step on a confirmed zero-holder check of its chips (`lsof` is blind under `sg docker`: confirm by process name too); append as gated stages of the scheduler, never beside a gated chain.

0. **Kernel-side exclusion of the hang, one chip, no fabric** (watcher NOC sanitizer flags any NOC target outside the worker grid / L1 at the first transaction; the watcher is incompatible with fabric, so single chip only):
   `TT_METAL_WATCHER=5 TT_VISIBLE_DEVICES=27 pytest models/demos/glm53_flash/tests/custom_ops/test_mhc_pre_block.py -s`
   then `TT_METAL_WATCHER=5 GLM53_MHC_PRE_BLOCK=1 TT_VISIBLE_DEVICES=27 pytest models/demos/glm53_flash/tests/test_decoder_layer.py -s -k decode` (the fused path inside one decoder layer; pick the decode case of that file). Expected: no watcher NOC error; the existing unit bars hold.
1. **One chip of the block, the exact mesh shapes / dtypes, fused vs composite + model integration + two instances + trace** (default `GLM53_MESH_SHAPE=1,1`):
   `TT_VISIBLE_DEVICES=27 pytest models/demos/glm53_flash/tests/custom_ops/test_mhc_pre_block_mesh.py -s`
   Expected: per chip `hn` PCC >= 0.9999 / rel rms < 5e-3 vs the composite, post/comb lane tiles PCC >= 0.9999 / rel < 2e-3, the model's `_pre` bit-identical to the direct call, trace replay bit-identical to eager. Also rerun the existing unit test on this chip (it now runs the patched kernels): `TT_VISIBLE_DEVICES=27 pytest models/demos/glm53_flash/tests/custom_ops/test_mhc_pre_block.py -s`.
2. **The (2,4) block, all 8 chips, bit-level**:
   `env $BLOCK pytest models/demos/glm53_flash/tests/custom_ops/test_mhc_pre_block_mesh.py -s`
   Expected as in 1 on every chip, plus bit-identical fused outputs on all 8 chips for the replicated input (the composite too). A chip-dependent difference here is a mesh hazard; a uniform fused-vs-composite difference above the bars is a numerical/integration defect (explanation (b)); within the bars on every chip = explanation (a).
3. **22-layer chain A/B on the block** (the EP32 reference config of the ledger; chain PCC must match the composite's 0.942429 to the known run-to-run band, greedy ids identical):
   `env $BLOCK GLM53_MHC_PRE_BLOCK=1 GLM53_MODEL_LAYERS=0-21 GLM53_MODEL_B=4 GLM53_MODEL_S=16 GLM53_MODEL_PCC_BAR=0.5 GLM53_MODEL_TRACE_MB=512 GLM53_MOE_LAYOUT=ep pytest models/demos/glm53_flash/tests/test_mesh_model.py -s -k chain`
   and the same with `GLM53_MHC_PRE_BLOCK=0`; then the accuracy gate on natural text per the PCC-metric rule: `env $BLOCK GLM53_MHC_PRE_BLOCK=1 pytest models/demos/glm53_flash/tests/test_mesh_model_logits.py -s` vs `=0` (logits PCC, top-1/top-5 agreement). Report TTFT and decode tok/s from the same run.
4. Only then the 45-layer bfp4 block chain (`sched_preblock_ab2.sh` config) and the 8x4.

## 6. Files changed (all under `models/demos/glm53_flash`; nothing committed, no git state touched)

| file | change |
|---|---|
| `custom_ops/mhc_pre_block/kernels/writer_mhc_pre_block.cpp` | H1: idle cores wait for the pre tile before the round-2 send |
| `custom_ops/mhc_pre_block/kernels/compute_mhc_pre_block.cpp` | H2: per-launch packer-format cache (`g_pack_last`, reset in `kernel_main`) |
| `custom_ops/mhc_pre_block/op.py` | H3 mesh-safe golden (`w_host`); H4 device-id guard; the arg packing factored into device-free `plan_work` / `cb_plan` / `compile_time_args` / `runtime_args_for_core` (same vectors, same CB order -> same program hash); ABI constants; comments |
| `custom_ops/mhc_pre_block/README.md` | grid text, mesh notes, trace note |
| `tt/mhc.py` | H5 docstring only |
| `tests/custom_ops/test_mhc_pre_block_host.py` (new) | host-only: kernel ABI vs op.py, work plan, CB plan, the gather-tree model (reproduces H1 without the fix) |
| `tests/custom_ops/test_mhc_pre_block_mesh.py` (new) | device: fused vs composite per chip at the model's shapes through the model's `_pre`, two instances / cache hits, trace replay (not run) |

Interface unchanged (`MhcPreBlock(device, fn, consts, norm_weight, *, ...)`, `allocate_outputs`, `__call__`, `golden`);
`GLM53_MHC_PRE_BLOCK` default stays `0`.

## 7. What was run here (host only)

* `pytest --noconftest models/demos/glm53_flash/tests/custom_ops/test_mhc_pre_block_host.py`: 33 passed (before the
  writer fix the 13x10 model cases failed with "gather1 slot 9 of (12,0) overwritten before it was consumed" + deadlock,
  i.e. the model found H1 on its own).
* Syntax-only compile of the patched kernels with the JIT's toolchain (`runtime/sfpi/compiler/bin/riscv-tt-elf-g++
  -fsyntax-only`, the JIT's flags, include dirs and defines, the 12x10 compile-time arg vectors from
  `MhcPreBlock.compile_time_args`): reader (NCRISC, NOC1), writer (BRISC, NOC0), compute on all three TRISCs: rc 0
  (script `/tmp/claude-1000/-home-ttuser-ashai/84631012-4793-4df4-ba96-fdeb1d31445d/scratchpad/kcompile/compile.sh`).
* `pre-commit run --files <the 7 files>` green.
* NOT run: anything that opens a device (the chips are in use), any pytest under the model dir without `--noconftest`.

## 8. Open assumptions

1. The block / 8x4 chips' grid: 12x10 is inferred from the block tracy op list (120 available cores) and the single-chip
   logs (chips 1-4); the mesh open validates equality, it does not print the size. A 13x10 chip anywhere makes H1 the
   explanation of wrong values on that device; the fix covers both.
2. The explanation of PCC 0.70 is decided by step 2 of the plan, not by this audit: the audit excludes NOC / addressing /
   race mechanisms (deterministic, chip-identical result; no device-specific addressing in the program), it cannot
   exclude a deterministic numerical / integration difference without running the A/B.
3. `CircularBuffer(cb_pre).wait_front(1)` on BRISC for a CB produced by NCRISC and (on worker cores) consumed by TRISC:
   used only on idle cores, where nothing pops the CB; it reads the shared tiles-received counter (standard CB sync).
4. Program-cache hits re-derive runtime args through the adapter's slow-path rebuild (code read, not instrumented); the
   two-instance device test is the check.
5. The ARC freeze at close is treated as the board-level event seen on 2026-10-20 21:41; the watcher run (step 0) is the
   kernel-side exclusion.
