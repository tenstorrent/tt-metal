# ttsim verification of the compute-kernel API ordering checker

Date: 2026-11-05

## Environment

- `ttsim` v1.16.0, Wormhole B0 (`libttsim_wh.so`), CPU only.
- tt-metal built locally from `main` with the host GCC and the pinned SFPI
  toolchain; no Tenstorrent card and no GPU work was used.
- SOC descriptor: `tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml`.
- Slow dispatch (`TT_METAL_SLOW_DISPATCH_MODE=1`) and forced JIT compilation
  (`TT_METAL_FORCE_JIT_COMPILE=1`) so every kernel compile command is captured
  (`TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`).

## Clean workloads

| Example | ttsim result | Captured compute TUs | Checker findings |
| --- | --- | ---: | ---: |
| `metal_example_add_2_integers_in_riscv` | `Success: Result is 21` | 0 | 0 |
| `metal_example_matmul_single_core` | `Metalium vs Golden -- PCC = 0.98135424`, `Test Passed` | 3 (UNPACK/MATH/PACK) | 0 |
| `metal_example_eltwise_binary` | `Test Passed` | 3 (UNPACK/MATH/PACK) | 0 |

The matmul and eltwise logs were replayed through the checker's real-compile
path (`--log`), not the synthesized `--kernel` flags. Both runs parsed the
actual generated kernel translation units with zero holes and zero findings.

## Injected bug

`tt_metal/programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp`
was temporarily changed to comment out `add_init(cb_in0, cb_in1)`, leaving
`add_tiles` without its init. The source was restored immediately afterwards.

- **ttsim**: exits 1 with
  `[4436] ERROR: UnimplementedFunctionality: mop_ingest: outer_loop_len=0`.
- **Checker** (on the captured build log):
  `missing-init` for `llk_math_eltwise_binary` on MATH and `llk_unpack_AB` on
  UNPACK, both at the `add_tiles` call in `tiles_add.cpp:43`.

This is the same failure the runtime LLK sanitizer would report as an execute
for an operation the engine is not initialized for; the static checker finds it
without running the kernel.
