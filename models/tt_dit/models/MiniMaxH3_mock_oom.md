# Reproducing MiniMax-H3 out-of-memory failures without a galaxy

MiniMax-H3 on the Wormhole 4x8 galaxy runs within a few hundred MB of the 12 GB per chip, and
its memory failures show up minutes into a warmup that needs all 32 chips. Every one of those
failures is raised on the host: the allocator (`tt_metal/impl/allocator/bank_manager.cpp:495`)
and the program's static circular-buffer check (`tt_metal/impl/program/program.cpp:2470`) run
before anything is dispatched. tt-metal's mock device keeps exactly those paths and turns every
device read and write into a no-op, so the whole H3 allocation sequence can be replayed on a box
with no Tenstorrent hardware, with silicon's error text; with a warm kernel cache it reaches the
first DRAM-critical point in 23 s and binds the whole ladder in about 3 minutes.

This page records how to do that, what it was measured to buy over running on the galaxy, and
walks through the two memory bugs it was used on in October 2026: an L1 circular-buffer overflow
in the audio vocoder's conv3d (`d42e9f3880d`) and a DRAM leak of the audio decoder's CCL
ping-pong buffers (`3a33765dd5d`).

## What the mock device keeps and what it drops

`TT_METAL_MOCK_CLUSTER_DESC_PATH=<yaml>` switches the runtime target to `TargetDevice::Mock`
(`tt_metal/llrt/rtoptions.cpp:530`). UMD then builds a `MockChip` per chip instead of opening
PCIe (`tt_metal/llrt/tt_cluster.cpp:463`), and the rest of the stack short-circuits wherever it
would have touched hardware.

| Kept, identical to silicon | Dropped |
|---|---|
| DRAM and L1 allocator, bank sizes, OOM message (`bank_manager.cpp:495`; the mock allocator is a thin `L1BankingAllocator` subclass, `tt_metal/impl/device/mock_allocator.cpp:17`) | Device reads: `ttnn.to_torch` returns zeros, so every PCC or value assert fails |
| Static circular-buffer validation at program compile (`program.cpp:3274`, `:3296`, `:3306`) | Kernel-binary DRAM buffers: `populate_dispatch_data` returns early (`program.cpp:2744`), so the allocator holds 17 to 57 MB less than silicon at the same point (measured below) |
| Kernel JIT with the real RISC-V toolchain, so binary sizes and compile errors are real | Trace capture: `create_trace_node` dereferences that missing binary buffer (`tt_metal/impl/program/dispatch.cpp:2788`) and segfaults. The WH 4x8 preset does not trace; the 4x32 quad does |
| Compute grid, L1 base, `l1_small`, fabric and mesh-graph selection (the control plane runs on the yaml) | Timing, hangs, data-dependent behaviour |
| `ttnn.get_memory_view`, so `MINIMAX_H3_DRAM_PROBE=1` works and attributes the same tensors | |

Measured on `UF-EV-B12-GWH02` on 2026-10-05: the mock 4x8 and the real galaxy both report an 8x9
compute grid, 72 L1 banks with 1327904 B free per bank at `l1_small_size=65536`, and 12 DRAM
banks of 1070773184 B.

## Setup

**1. Get a cluster descriptor of the target.** Two Wormhole 6U descriptors ship in the tree and
they are not interchangeable. `tt_metal/third_party/umd/tests/cluster_descriptor_examples/6u_cluster_desc.yaml`
(the one `configure_mock_mode(WORMHOLE_B0, 32)` picks) lacks the fields fabric auto-discovery
needs, so the control plane derives a 32x1 system mesh and `open_mesh_device((4, 8))` fails with
`Requested mesh is too big and is not rotatable: MeshShape([4, 8]) and SystemMesh MeshShape([32, 1])`.
`tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_cluster_desc/6u_cluster_desc.yaml` was
captured from a real galaxy, opens the 4x8 mesh and reports the same geometry as the hardware
(8x9 compute grid, 12 x 1070773184 B DRAM); use that one, it is what the CPU-only fabric tests
use too. Alternatively dump your own galaxy's descriptor once:

```bash
python -c "import ttnn; print(ttnn.cluster.serialize_cluster_descriptor())"
# -> /tmp/umd_XXXX/cluster_descriptor.yaml ; copy it somewhere permanent
```

With the real descriptor the mesh matches `4x8_Mesh_flat_torus_xy` and opens in about 15 s.
A copy of this galaxy's descriptor lives at
`~/mock_cluster_descs/wh_galaxy_UF-EV-B12-GWH02_cluster_desc.yaml` on `UF-EV-B12-GWH02`.

**2. Run the test node unchanged, from the repo root.** The pytest fixtures need nothing special:
`silicon_arch_name` comes from `ttnn.get_arch_name()`, which reads the arch out of the yaml
(`tt_metal/llrt/get_platform_architecture.hpp:80`), and `get_num_devices()` reports 32.

```bash
export TT_METAL_MOCK_CLUSTER_DESC_PATH=/path/to/wh_galaxy_cluster_desc.yaml
export TT_DIT_CACHE_DIR=/path/to/tt_dit_cache      # pre-converted weights; otherwise safetensors load
export TT_METAL_CACHE=/path/to/kernel/cache        # see note below
export MINIMAX_H3_DRAM_PROBE=1                     # per-owner DRAM accounting at each checkpoint

pytest models/tt_dit/tests/models/minimax_h3/test_pipeline_ref2va_minimax_h3.py \
    -k "one_image and 4x8_WH" -s -x --timeout 14400
```

Use `-x` and read the log up to the first failure. A run that gets through warmup, the forced
ladder requests and the served request has exercised every allocation; it then fails on the
first assert that looks at output values, which is expected and is the "pass" condition here.

Notes:

* **Run from the root of the checkout that owns the build.** The runtime resolves an op's kernel
  source against the current directory while the include path follows `TT_METAL_RUNTIME_ROOT`
  or the library's build root (`tt_metal/llrt/rtoptions.cpp:343` onward). Starting the same test
  from a second worktree pulled `ring_joint_reader.cpp` from one tree and its headers from the
  other, and every SDPA kernel failed to JIT with `redefinition of 'struct KVPadRotationContext'`.
* The JIT cache is keyed by build key, not by header contents, so a mock run after a kernel
  header change (such as `ae9ac7626c4`) needs a fresh `TT_METAL_CACHE` or it reuses stale
  binaries. The first run in a fresh cache compiles every H3 kernel; later runs are much faster
  (see the timings).
* Weight loading is host work and costs the same as on silicon. `fp32 -> bf16` tilize in
  `from_torch` is the slow part; the `TT_DIT_CACHE_DIR` cache avoids it.
* The mock allocator also has snapshot and restore hooks
  (`extract_mock_allocator_state` / `override_mock_allocator_state`, exercised by
  `tests/tt_metal/tt_metal/device/test_mock_allocator.cpp`) for unit tests that want to start
  from a known occupancy.

## What it buys: measured timings

Same test (`test_ref2va_end_to_end[4x8_WH-one_image]`), same box, same weight cache. "Cold"
means an empty `TT_METAL_CACHE` for the mock; the cold silicon run used the default cache, which
held 16 kernel directories for this build key and compiled 107 more during the run, so it is cold
in all but name. "Warm" is a second run against the cache the cold run filled. The cold mock run
also shared the host with two other mock runs, so its JIT numbers are pessimistic; the warm runs
had the host to themselves. "arch cache" is the same warm run after the `get_platform_architecture`
fix described in the next section. Times are from pytest start.

| Checkpoint | Mock, cold JIT | Mock, warm JIT | Mock, warm JIT + arch cache | Silicon, cold JIT | Silicon, warm JIT |
|---|---|---|---|---|---|
| Old tip `1798af10c31`: conv3d CB overflow raised | 9.5 s | | | 7.2 s | |
| Fixed tip `280c014c0ac`: pipeline init begins | 0:31 | 0:04 | 0:02 | 0:26 | 0:11 |
| transformer loaded (cache hit) | 3:02 | 0:31 | 0:19 | 4:40 | 0:58 |
| **top rung 326656 bound, step 1 done** (where the Oct 3 DRAM OOM fired) | **4:08** | **0:37** | **0:23** | **7:22** | **2:17** |
| all 17 rungs bound | 19:57 | 6:28 | 3:06 | 29:09 | 12:22 |
| audio decode warm (12 lengths) done | 32:49 | 10:26 | 6:47 | 40:33 | 16:42 |
| prompt-encoder warm (46 shapes) done | 52:12 | 15:26 | 10:08 | 56:47 | 21:05 |
| served request done, first assert | 54:43 | 17:55 | 10:38 | 60:23 | 24:40 |

Two things follow. First, the point that decides whether the preset fits in DRAM is the top
rung's forced request, and the mock reaches it in a minute with a warm cache, against seven on
the galaxy and, before `280c014c0ac` moved the ladder walk first, about thirty. Second, the
compile-only warms (audio lengths and prompt-encoder envelope, roughly 3000 programs) cost the
same on both, because on the mock they are pure JIT. They never allocate anything the ladder
did not, so for OOM work stop the run after the ladder, or set
`MINIMAX_H3_WARMUP_SKIP_DECODE_WARM=1`.

The mock's DRAM under-count, from the probe at identical checkpoints (device 0, per device):

| Rung | Mock allocated | Silicon allocated | Silicon minus mock | Largest free block, mock / silicon (MB per bank) |
|---|---|---|---|---|
| 326656 | 9.811 GB | 9.828 GB | 17 MB | 106.8 / 105.4 |
| 240640 | 9.071 GB | 9.105 GB | 34 MB | 208.7 / 205.9 |
| 147712 | 8.265 GB | 8.311 GB | 46 MB | 324.4 / 320.6 |
| 65280 | 7.550 GB | 7.601 GB | 51 MB | 419.8 / 415.5 |
| 24064 | 7.193 GB | 7.250 GB | 57 MB | 460.3 / 455.6 |

The gap grows with the number of compiled programs, as the kernel-binary explanation predicts.
Treat a mock headroom figure as optimistic by about 60 MB per device for a fully warmed H3.

## Where the warm mock run's time goes

With every kernel cached the mock still takes 6:28 for the 17-rung ladder walk, about 21 s per
rung, and 17:55 for the whole test. A 20 Hz Python-level sample of that run
(`py-spy record --nonblocking`, so nothing was paused; 21537 samples) splits as follows.

| Share of samples | What was running |
|---|---|
| 48% | inside a ttnn op call (the C++ host path: validation, program-cache hash, program build, mock dispatch). Two thirds of it is the CCL manager: `all_gather_async` 9.2%, `neighbor_pad_async` 7.6%, ping-pong buffer fills 4.1%, `reduce_scatter_async` 4.0%; then `conv3d` 7.5% |
| 31% | `ttnn.get_arch_name()`, reached through `is_blackhole()` from `get_matmul_core_grid` (`models/tt_dit/utils/matmul.py:452`, 22.8%) and `Linear.forward` (`models/tt_dit/layers/linear.py:467`, 5.7%) |
| 4% | host tensor conversion (`from_torch`, `to_torch`, tile/untile) |
| 3% | weight-cache loads for the per-rung stage reloads |
| 14% | everything else: torch glue in the VAE stitch and blend, PIL, the probe, pytest |

By phase: ladder walk 34%, audio warm 21%, prompt-encoder warm 28%, the served request 14%.

The 31% is mock-specific and avoidable. `ttnn.get_arch_name()` is bound to
`tt::tt_metal::detail::get_platform_architecture_name`
(`ttnn/cpp/ttnn-nanobind/device.cpp:673`, `tt_metal/impl/host_api/tt_metal.cpp:329`), which
builds a fresh `RunTimeOptions` and calls `get_platform_architecture`; in mock mode that parses
the cluster-descriptor YAML on every call (`tt_metal/llrt/get_platform_architecture.hpp:82`).
Measured: 5.55 ms per call in mock against 0.01 ms on silicon, and the H3 matmul path calls it
per linear layer, roughly 60000 times in this run. The mock branch of `get_platform_architecture`
now caches the parsed arch per descriptor path (`tt_metal/llrt/get_platform_architecture.hpp`),
which brought the call to 0.01 ms and the warm run from 17:55 to 10:38, the ladder walk from 6:28
to 3:06 and the top rung from 0:37 to 0:23 (the "arch cache" column above). The model-side
alternative, resolving `is_blackhole()` once per module instead of per op in `get_matmul_core_grid`
and `Linear.forward`, would remove the remaining 0.01 ms calls on both targets.

The 48% is the real per-op host floor and is paid on silicon as well; there it overlaps with
device execution, which is why warm silicon's ladder (12:22) is only twice the mock's. It is
dominated by the CCL collectives, whose programs are built per call for each of the 32 chips.
For a memory check that floor is already acceptable: the DRAM-critical top rung is reached at
0:23 and the whole ladder at 3:06 with the arch cache in place.

## Which MiniMax-H3 tests run under the mock

Every test in `models/tt_dit/tests/models/minimax_h3/` was tried on the mock 4x8 (one process
per test, fixed tip). The outcomes fall into four groups and none of them is a mock-specific
breakage:

* **Host-only tests pass**: `test_vae_minimax_h3::test_tiling_geometry_matches_reference`,
  `test_references_minimax_h3`, `test_packing_minimax_h3`, `test_scheduler_minimax_h3`,
  `test_conditioning_minimax_h3`.
* **Device tests run all their allocation and dispatch, then fail at the value check** with
  `PCC = nan` or an element-wise assert on the zero readback: `test_audio_minimax_h3`
  (`test_decode`, the depthwise conv1d), `test_text_encoder_minimax_h3`,
  `test_vision_conditioner_minimax_h3`, `test_stitch_device_minimax_h3` (stitch, blend,
  unpatchify), `test_unpatchify_gather_minimax_h3`, `test_vae_parallel_minimax_h3` (the
  sharded-encoder cases die in the PCC helper with a divide by zero instead), and the Wormhole
  rows of `test_transformer_minimax_h3` (`4x8sp1tp0nl4_ring_is_fsdp0/1`: block, attention and
  the two-layer transformer, 19 to 72 s each). An OOM or CB overflow anywhere in those paths
  would have fired first.
* **Blackhole-only rows skip** exactly as on a Wormhole galaxy, since `is_blackhole()` reads the
  mock's arch.
* **The end-to-end pipeline tests run through warmup and the served request** and stop at the
  first host-side assert. For `test_ref2va_end_to_end` that used to be a pinned padded length
  (`_EXPECTED_PADDED_LEN`, probed in #57097 before 4x8 meshes had a bucket ladder; since #59060
  every 4x8 preset buckets through `MINIMAX_H3_REF2VA_BUCKET_LADDER_4X8`, and the video cases'
  packing had moved too), so all three cases failed on any 4x8 mesh, silicon included. The test
  now pins the packed (logical) length per case and checks the padded length against the
  pipeline's own resolver, `select_bucket` on its ladder. The mock probed the three logical
  lengths (39655, 75474, 77404) in one 16-minute pass each, without a galaxy.

## Case 1: L1 circular-buffer overflow in the vocoder conv3d

**Symptom.** On `1798af10c31`, the ref2va `one_image` case on `4x8_WH` dies 10 s after the
mesh opens, in `MiniMaxH3Pipeline.__init__` -> `_warm_audio_decode`, at the first of 12 audio
lengths:

```
TT_THROW @ tt_metal/impl/program/program.cpp:2470
Statically allocated circular buffers on core range [0-0 - 7-8] grow to 1510624 B
which is beyond max L1 size of 1499136 B
```

The mock log and the silicon log are byte-identical for that message, and both put it in
`ttnn.experimental.conv3d` called from `models/tt_dit/layers/audio_ops.py:135`
(`conv3d_maybe_split`) inside `models/tt_dit/models/audio_vae/vocoder_ltx.py:516`.

**Why mock sees it.** Circular-buffer sizes come from the conv3d program factory
(`ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_program_factory.cpp`), which only
reads shapes, the blocking table in `models/tt_dit/utils/conv3d.py`, the arch and the device's
L1 base. None of that depends on device state, so the program the mock compiles is the one
silicon compiles, and the static check at `program.cpp:3296` rejects it the same way.

**Fix** (`d42e9f3880d`). Two budgeting bugs in the factory: the prefetch budget was measured
against the whole unreserved L1 minus a fixed 200 KB stand-in for the kernel-config ring buffer,
which over-allocates on Wormhole where ring buffer plus `l1_small` exceed 200 KB; and Wormhole's
`out_subblock_h` scaling multiplied the `vol2col_tiled` and operand-split CBs, overrunning L1 by
11 KB at `sub_h = 2` on a blocking swept on Blackhole. The factory now budgets from
`l1_size_per_core() - allocator L1 base`, counts the pad-offset CB, and shrinks `sub_h` toward 1
until the static CBs fit.

**After.** On `280c014c0ac` the same mock command passes all 12 audio lengths
(`audio decode warmed: +1563 programs`) and goes on to the served request; silicon does the same.

## Case 2: DRAM leak of the audio decoder's CCL ping-pong pairs

**Symptom.** On the WH 4x8, the top ref2va rung (326656 rows) ran out of DRAM in ff2's
reduce-scatter pair with 105 MB per bank free and a largest block of 30 MB, and the seven
smallest rungs then failed in `prepare_static_sources` on the refiner's gather pairs. Nothing in
the DiT had grown; the headroom had gone to 842 small ping-pong tensors (1.14 GB per device,
about 95 MB per bank) that the 12-length audio decode warm left behind.

**How the probe names it.** The batch-sharded audio decoder gathers through
`audio_ccl_manager`, whose ping-pong cache is keyed by shape and never evicts.
`MINIMAX_H3_DRAM_PROBE=1` (`models/tt_dit/utils/dram_probe.py`) walks the pipeline's object
graph at each checkpoint, groups live device tensors by owner (module weights, module-held
tensors, pipeline state, CCL ping-pong pairs) and reconciles against the allocator's view of
device 0, printing the unattributed remainder down to individual blocks. The leak showed up as
`ccl_ping_pong  842 tensors` before every rung, against 0 on 2026-10-03 before the warm order
changed. The probe reads only host bookkeeping (`buffer_address()`, page counts,
`get_memory_view`), so it reports the same owners and the same bytes on the mock, minus the
kernel-binary gap above.

**Fix** (`3a33765dd5d`). Each audio decode (warm and served) and the reference encode run inside
the manager's transient scope when the preset neither persists CCL buffers nor traces audio, so
the pairs are released per decode. `280c014c0ac` additionally moves the ladder walk to the front
of warmup, so a top-rung OOM now surfaces minutes after launch instead of after half an hour of
compile-only warms.

**After.** On `280c014c0ac` the mock binds all 17 rungs; at the top rung the probe reports
9.811 GB allocated with a 106.8 MB per-bank largest free block, and silicon agrees to within the
17 MB of kernel binaries.

## A mock gate for the ladder

`models/tt_dit/tests/models/minimax_h3/test_mock_oom_minimax_h3.py` turns the recipe into a test.
It skips unless `TT_METAL_MOCK_CLUSTER_DESC_PATH` is set, so it never takes a galaxy; opens the
Wormhole 4x8 preset with the ref2va gates' `l1_small_size=16384`; builds the ref2va pipeline with
`warmup=False`; then calls the constructor's own `_warmup_on_init` with the compile-only warms
disabled (`MINIMAX_H3_WARMUP_SKIP_DECODE_WARM=1`, prompt-encoder warm stubbed) and
`MINIMAX_H3_WARMUP_SKIP_OOM_RUNGS=1`, so one walk binds every rung with serving's own warmup
requests and reports the rungs that do not fit instead of raising on the first. The assert is
`pipeline.unfittable_rungs == []`. Measured here: 5:11 with a warm kernel cache, 17 rungs bound,
6.65 GB per device still allocated at the end of the walk.

The test matrix entry is `tests/pipeline_reorg/models_mock_device_tests.yaml`
(`minimax-h3-ref2va-ladder-mock-wh-4x8`, `cpu_medium`, 100 min for a cold kernel cache). It points
the descriptor at the checked-in `tt-cluster-descriptors` 6U yaml and sets `MINIMAX_H3_DRAM_PROBE=1`
so a failure comes with the per-owner attribution. It runs through the existing CPU-only lane,
`.github/workflows/fabric-cpu-only-tests-impl.yaml`, by passing `tests-yaml-path` to it the way
`merge-gate.yaml` does for the fabric matrix; which trigger to hang it on (nightly or merge gate)
is a scheduling decision this page does not make. Two requirements on the runner that the fabric
lane does not have: the H3 weights (`MINIMAX_H3_MODEL_PATH` or the HuggingFace cache; the test
skips without them) and, for a run measured in minutes rather than an hour, a populated
`TT_DIT_CACHE_DIR` and kernel cache. A random-weights mode for the pipeline would lift the
weights requirement and is the natural next step for a truly CPU-only lane.

The first run of this test was itself a demonstration. It was written with the t2va meshes'
`l1_small_size=65536`, and 23 s after the mesh opened the vision tower's windowed SDPA failed with
`Statically allocated circular buffers in program 148 clash with L1 buffers ... L1 buffer allocated
at 1427840 and static circular buffer region ends at 1441024`, identically with both descriptors.
The ref2va gates run a 16 KB pool for exactly that reason (`test_performance_minimax_h3.py:154`);
with it the test passes. An L1 layout mistake that would have cost a galaxy reservation to find
cost 23 s on the mock.

## A workflow for the next OOM

1. Reproduce under mock with `MINIMAX_H3_DRAM_PROBE=1` and `-x`, from the repo root, with a
   warm `TT_METAL_CACHE`. Allocation order is driven by the host code, so the failure lands on
   the same op as on silicon, and the top rung is bound within about a minute.
2. If it is a circular-buffer or L1 overflow, the fix is in the op's program factory or its
   blocking table; the mock compile loop is the whole iteration, no device needed.
3. If it is a DRAM OOM, read the probe report at the failing checkpoint: a large `ccl_ping_pong`
   or `pipeline.*` owner that should not be resident names the leak; a large unattributed total
   points at transients, and the block table says how fragmented they left the heap.
   `MINIMAX_H3_WARMUP_SKIP_OOM_RUNGS=1` makes the ladder walk report and skip unfittable rungs
   instead of raising, so one run names the largest rung the mesh binds.
4. Iterate on mock until the run reaches the value asserts, then confirm once on silicon. Keep
   the two caveats above in mind: mock under-counts DRAM by the live kernel binaries (up to about
   60 MB per device here), and it cannot see hangs or data-dependent paths.
