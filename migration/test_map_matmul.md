# Matmul kernel → test map (Phase 2, STATIC analysis only)

Repo root: `/localdev/sjovic/tt-metal`. No device runs were performed; all conclusions are from grep/read.

## Top-level factory dispatch (matmul_device_operation.cpp:483 `select_program_factory`)
Dispatch is by `MatmulProgramConfig` variant type (std::visit), not by tensor layout directly:
- `MatmulMultiCoreReuseMultiCastProgramConfig`   → `MatmulMultiCoreReuseMcast2DProgramFactory`  (2D mcast factory)
- `MatmulMultiCoreReuseMultiCast1DProgramConfig` → `MatmulMultiCoreReuseMcast1DProgramFactory`  (1D mcast factory; `gather_in0=true` instead routes to the MeshWorkload 1D factory → ring kernels, deferred)
- `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` → `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` (DRAM-sharded factory)

Within 1D/2D factories the in0 sender/receiver kernel pick is gated on `in0_is_sharded` / `in0_block_sharded`:
- in0 INTERLEAVED → `in0_sender_padding.cpp` (+ `in0_receiver.cpp` for the mcast-receiver cores)
- in0 BLOCK-SHARDED → `in0_sender_receiver_padding_block_sharded.cpp` (R6, DEFERRED)
Both factories emit via two code paths: a descriptor path (`create_*_descriptor`) and a legacy `CreateKernel` path (`create_program_mcast_in0_in1`). The 2D factory uses the descriptor path; both reference the same kernel source strings.

NOTE on grids: `in0_receiver` only exists when the mcast topology has receiver cores (>1 core along the in0-mcast axis). `in1_receiver_writer_padding` (2D) only exists when there are in1 receiver cores (>1 core along the in1-mcast axis). So a multi-row, multi-col grid (e.g. 8x4) instantiates all four interleaved-in0 kernels at once.

---

### ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp
- role / tag: in0 mcast SENDER reader; clean
- factory:
  - `create_program_mcast_in0_in1_descriptor` @ matmul_multicore_reuse_mcast_2d_program_factory.cpp:766 (2D path; selected when in0 NOT block-sharded — `else` of the `in0_block_sharded` branch)
  - `create_program_mcast_in0_in1` (legacy) @ matmul_multicore_reuse_mcast_2d_program_factory.cpp:2251
  - 1D factory `create_program` @ matmul_multicore_reuse_mcast_1d_program_factory.cpp:599 and legacy @ 1555 / descriptor @ 3456 / 4416 (selected when `!in0_is_sharded`)
  - sparse 1D @ sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:451
  - Selected when: matmul/linear with an mcast (1D or 2D) program_config AND in0 is INTERLEAVED (in0 NOT block/width-sharded), num_cores>1.
- op: ttnn.matmul / ttnn.linear
- candidate_validation_set:
  - `test_matmul.py::test_matmul_1d_multiple_output_blocks_per_core` with `in_sharded` False, `mcast_in0` True, smallest sub-case (m=256,k=1024,n=2048,grid 8x2) — 1D interleaved sender.
  - `test_matmul.py::test_matmul_2d_multiple_output_blocks_per_core` with `in0_sharded` False (grid 8x4) — 2D interleaved sender (also exercises the receiver/in1 kernels).
- candidate_regression_set: `test_matmul_1d_multiple_output_blocks_per_core` (full mcast_in0×in_sharded×blocks matrix), `test_matmul_2d_multiple_output_blocks_per_core` (full), `test_matmul_1d_tiny_tile`, `test_matmul_2d_tiny_tile`, `test_sharded_matmul` (ids mcast_in0 / mcast_in1), sweeps `tests/sweep_framework/sweeps/matmul/short/matmul_user_program_config_mcast_1d.py` + `..._mcast_2d.py`.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k "matmul_1d_multiple_output_blocks_per_core and True-False"`  (mcast_in0=True, in_sharded=False; filter on value substrings only, no name=value)
  Confirm build: `find generated -path '*reader_bmm_tile_layout_in0_sender_padding*' -name '*.o' -o -path '*reader_bmm_tile_layout_in0_sender_padding*'` — the kernel basename must appear under `generated/jit_build/.../reader_bmm_tile_layout_in0_sender_padding/`.
- coverage_confidence: high
- gaps: none significant; sanity-tier coverage exists (in-repo, single-device).

### ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp
- role / tag: in0 mcast RECEIVER reader; clean
- factory:
  - 2D descriptor @ matmul_multicore_reuse_mcast_2d_program_factory.cpp:818 (`!in0_block_sharded && in0_receiver_interleaved.num_cores()>0`); also "other cores" variant @ 851; legacy @ 2307 / 2341
  - 1D @ matmul_multicore_reuse_mcast_1d_program_factory.cpp:661 (`!in0_is_sharded && in0_mcast_receivers.num_cores()>0`); descriptor @ 3526
  - sparse 1D @ sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:469
  - Selected when: mcast matmul, in0 INTERLEAVED, AND there are receiver cores on the in0-mcast axis (i.e. >1 core along that axis).
- op: ttnn.matmul / ttnn.linear
- candidate_validation_set:
  - `test_matmul.py::test_matmul_1d_multiple_output_blocks_per_core` `mcast_in0` True + `in_sharded` False, grid 8x2 (15 receivers, 1 sender).
  - `test_matmul.py::test_matmul_2d_multiple_output_blocks_per_core` `in0_sharded` False, grid 8x4 (multi-row → in0 receivers exist).
- candidate_regression_set: same suites as in0_sender_padding (receiver is co-instantiated with the sender whenever the grid has >1 core on the in0 axis).
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k "matmul_1d_multiple_output_blocks_per_core and True-False"`
  Confirm build: kernel basename `reader_bmm_tile_layout_in0_receiver` present under `generated/jit_build/.../reader_bmm_tile_layout_in0_receiver/` (NOTE: prefix-match carefully so it doesn't also match `_in0_receiver_in1_*` example kernels — they live under a different build path / not in this op).
- coverage_confidence: high
- gaps: none significant; always co-built with the sender on multi-core grids.

### ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp
- role / tag: in1 SENDER + output WRITER (hybrid reader/writer on RISCV_0); clean
- factory:
  - 2D descriptor @ matmul_multicore_reuse_mcast_2d_program_factory.cpp:783; legacy @ 2268
  - 1D @ matmul_multicore_reuse_mcast_1d_program_factory.cpp:674 (created on `all_cores_with_work`, unconditional); legacy @ 1571; descriptor @ 3538 / 4431
  - sparse 1D @ sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:483
  - Selected when: ANY mcast matmul/linear (1D or 2D). Always present — in1 is the sender/writer on every worker core. Independent of in0 sharding.
- op: ttnn.matmul / ttnn.linear
- candidate_validation_set:
  - `test_matmul.py::test_matmul_1d_multiple_output_blocks_per_core` `mcast_in0` True, `in_sharded` False (smallest) — present on all worker cores.
  - `test_matmul.py::test_matmul_2d_multiple_output_blocks_per_core` `in0_sharded` False.
- candidate_regression_set: every mcast matmul test (1d/2d tiny_tile, multiple_output_blocks, sharded_matmul, padded_1d/2d, linear fused-bias 1d/2d), sweeps mcast_1d + mcast_2d.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k "matmul_1d_multiple_output_blocks_per_core and True-False"`
  Confirm build: basename `reader_bmm_tile_layout_in1_sender_writer_padding` under `generated/jit_build/.../`.
- coverage_confidence: high
- gaps: none; broadest coverage of the set (instantiated in essentially every mcast test).

### ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp
- role / tag: in1 RECEIVER + output WRITER; clean
- factory:
  - 2D ONLY. descriptor @ matmul_multicore_reuse_mcast_2d_program_factory.cpp:802 (`in1_receiver.num_cores()>0`) and the "other cores" variant @ 834 (`in0_receiver_in1_receiver_interleaved_other_cores.has_value()`); legacy @ 2288 / 2326.
  - Also referenced in the 1D factory @ matmul_multicore_reuse_mcast_1d_program_factory.cpp:1591 / 4450 — these are in the legacy/descriptor in0-block-sharded branch (R6 in0 path); reached only with block-sharded in0 on 1D.
  - Selected when: 2D mcast with >1 core along the in1-mcast (column) axis, so there are in1 receiver cores. NOT used for single-column 2D or for the common interleaved-in0 1D path.
- op: ttnn.matmul / ttnn.linear
- candidate_validation_set:
  - `test_matmul.py::test_matmul_2d_multiple_output_blocks_per_core` `in0_sharded` False, grid 8x4 (multi-column → in1 receivers exist). This is the cleanest hit.
  - `test_matmul.py::test_sharded_matmul -k mcast_2d` (BLOCK-sharded 2D, 5x7 grid) also instantiates it (with the R6 in0 sender, but the in1_receiver_writer kernel is the same file).
- candidate_regression_set: `test_matmul_2d_multiple_output_blocks_per_core` (full), `test_matmul_2d_tiny_tile`, `test_sharded_matmul` (mcast_2d / mcast_2d_transposed / mcast_2d_shard_width_gt_1), `test_linear_fused_non_broadcast_bias_2d_*`, sweeps `matmul_user_program_config_mcast_2d.py`.
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k "matmul_2d_multiple_output_blocks_per_core and False-False"`  (in0_sharded=False, out_sharded=False; needs an 8x8-capable device else it self-skips). This is a mesh_device test (mesh (1,NUM_DEVICES)).
  Confirm build: basename `reader_bmm_tile_layout_in1_receiver_writer_padding` under `generated/jit_build/.../`.
- coverage_confidence: high
- gaps: requires a multi-column grid (≥2 cores in the in1 axis) and an 8x8 device for the cleanest test; on a single-column program_config the kernel is NOT built. Validation test is mesh_device + has a built-in skip on small grids — confirm grid before relying on it.

### ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp
- role / tag: in0 sender, DRAM-sharded; refactor; HYBRID — 3 runtime worker types selected by runtime arg0 `worker_core_type` (0=idle/no-work early return @ kernel:41, 1=mcast sender + no compute @ :73, 2=mcast sender + compute @ :140).
- factory: `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory::create_descriptor` → kernel desc @ matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:388 (single CreateKernel over `all_cores_in_rect_grid`; the 3 worker types are differentiated at RUNTIME via runtime args, not by separate CreateKernel calls).
  - Selected when: program_config is `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` (width-sharded in1 living in DRAM; in0 width-sharded in L1). (The Batched-HS DRAM-sharded factory uses a different kernel `..._dram_sharded_height.cpp`, out of scope.)
- op: ttnn.matmul / ttnn.linear
- candidate_validation_set:
  - `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py::test_matmul_in1_dram_sharded_with_program_cache` smallest case M=32,K=8192,N=1280,grid (8,1) (single-device). Hits worker_core_type 2 across the row; the (8,2) cases exercise more of the type-1/idle split.
  - `tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_in1_dram_sharded_tiny_tile` (mesh_device, SKIPPED on Blackhole — issue #31385).
- candidate_regression_set: `test_matmul_dram_sharded.py` (all: with_program_cache full param list + activations relu/gelu/silu/sigmoid, with_mm_chain, 2d_in1_dram_sharded), `test_matmul_activations.py` (dram-sharded rows), models that use dram-sharded matmul (falcon/llama MLP).
- verification_command:
  `scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py -k "with_program_cache and 1280"`  (selects the N=1280 grid-8x1 sub-case by value substring).
  Confirm build: basename `reader_bmm_tile_layout_in0_sender_dram_sharded` under `generated/jit_build/.../`.
- coverage_confidence: med
- gaps: the 3 worker types are runtime-selected, so a single build/test compiles ALL THREE branches but a given grid may only exercise some at runtime — type-0 (idle) and type-1 (sender-no-compute) need a grid where some cores lack output work (the (8,2) cases). The in-`test_matmul.py` tiny-tile entry is mesh_device and Blackhole-skipped; primary coverage is nightly-only (not in the fast sanity suite).

---

## DEFERRED kernels (1-line notes only, not mapped deeply)
- `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` — R6 role-flip (single kernel acts as both sender and receiver for block-sharded in0); selected in 1D factory @ :599/620/642 and 2D factory when `in0_block_sharded`. Hit by `test_matmul_2d_multiple_output_blocks_per_core` / `test_matmul_1d_*` with in0_sharded=True and `test_sharded_matmul -k mcast_2d`.
- `reader_bmm_tile_layout_in1_ring_all_gather.cpp` + `reader_bmm_tile_layout_in0_ring_all_gather.cpp` — ring/collective (gather_in0 / all-gather-fused); routed via the MeshWorkload 1D factory (`gather_in0=true`). Hit by `test_matmul_1d_gather_in0.py` / `test_rs_matmul_1d_gather_in0.py` / `test_ring_matmul.py` (t3000 multi-device).

## Programming-example didactic readers (NOT pytest — standalone C++ example)
All three are CreateKernel'd by ONE host file:
`tt_metal/programming_examples/matmul/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp`
(refs: in0_sender_in1_sender @ :269, in0_sender_in1_receiver @ :278, in0_receiver_in1_sender @ :287; runtime-arg wiring @ :423/427/431). Built as CMake target `metal_example_matmul_multicore_reuse_mcast` (compiled with `TT_METAL_CI_MODE`). It is NOT run by pytest or by `tests/scripts/run_cpp_unit_tests.sh` (that script only runs the `vecadd` contributed example). It is run by directly executing the built binary `./build/programming_examples/metal_example_matmul_multicore_reuse_mcast`.

### tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp
- role / tag: in0 sender + in1 sender (top-left corner core of the mcast grid); clean, didactic
- factory: `matmul_multicore_reuse_mcast.cpp:267` (CreateKernel over CoreRange `in0_sender_in1_sender` @ :191). Selected when: it is the example's fixed core-role partition (one corner core both-senders) — no runtime dispatch condition, the example hard-partitions the grid.
- op: none (TT-Metalium programming example, not a TTNN op)
- candidate_validation_set: run the binary `./build/programming_examples/metal_example_matmul_multicore_reuse_mcast` (no pytest param).
- candidate_regression_set: none (not in any pytest/CI sweep found).
- verification_command: build via `./build_metal.sh --build-programming-examples` (or full build) then run `./build/programming_examples/metal_example_matmul_multicore_reuse_mcast`. Confirm build: basename `reader_bmm_tile_layout_in0_sender_in1_sender` under `generated/jit_build/.../` after running the binary once.
- coverage_confidence: low
- gaps: NO automated test coverage — example binary only, not wired into pytest or run_cpp_unit_tests.sh. Manual run required.

### tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp
- role / tag: in0 receiver + in1 sender (left-edge cores, not corner); clean, didactic
- factory: `matmul_multicore_reuse_mcast.cpp:285` (CoreRange `in0_receiver_in1_sender` @ :198, runtime RISCV_1). Selected when: example's fixed grid partition for the in1-mcast sender column that receives in0.
- op: none (programming example)
- candidate_validation_set / regression_set / verification_command: same example binary as above (`metal_example_matmul_multicore_reuse_mcast`).
- coverage_confidence: low
- gaps: NO automated coverage — manual example run only.

### tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp
- role / tag: in0 sender + in1 receiver (top-edge cores, not corner); clean, didactic
- factory: `matmul_multicore_reuse_mcast.cpp:276` (CoreRange `in0_sender_in1_receiver` @ :194). Selected when: example's fixed grid partition for the in0-mcast sender row that receives in1.
- op: none (programming example)
- candidate_validation_set / regression_set / verification_command: same example binary (`metal_example_matmul_multicore_reuse_mcast`).
- coverage_confidence: low
- gaps: NO automated coverage — manual example run only. (Note: there is also a 4th example kernel `reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp` for interior cores, instantiated by the same host file, not in scope.)
