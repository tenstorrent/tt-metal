# Native C++ port series: rules for every NC.* task (OWNER 2026-11-27)

Written by the overseer from the owner's decisions of 2026-11-27 (supervision.md). Every NC.* brief points here.

- Port to native C++ TTNN device ops (no prepare_program / run_prepared), as NC.1 did for grouped_matmul (commit b10e845af78; results/NC.1_layer.md).
- Tool: tt_ops_code_gen native_migrate (~/repos/tt_ops_code_gen/native_migrate/README.md), run in the SEPARATE checkout /scratch/dnijemcevic/native/tt-metal (it cuts its own branch and builds there; first fast-forward that checkout's dnijemcevic/native_gmm branch to this branch's HEAD; never run it in ~/repos/tt-metal).
- Lessons from NC.1 (supervision.md 17:40-18:20): do NOT pass --unit-test-dir (the flow keeps acceptance files only under the repo's tests/); export BRINGUP_SPEC for tests that use the bring-up harness; the port venv already has clang-tidy 20.1.8 and py-spy; a fork's unit test that opens its own mesh must run in its own process.
- For code that is not yet an op package (a class inside the model), first move its program builder into a ttnn.bringup fork package unchanged (skill bringup-fork-op; INDEX.md line), then port; or hand-port it following ttnn/ttnn/bringup/moe_block_ttnn/flat_routed_expert (C++ device op, sources.cmake, nanobind) if the tool cannot.
- Bring the result into this tree as is (resolve only real conflicts), rebuild (cmake --build build --target install).
- The model keeps its call sites (or the smallest change).
- OWNER rules for this series (binding, 2026-11-27): op-level checks only; no model load, no ladder, no 40-layer run (the series ends with ONE ladder, task NC.8).
- Write the op-level test named in the gate (BATCH_SHARED = True and BATCH_LAYERS = "none": the resident runner's mesh, no model; F309): for every program ported, at the model's real shapes and arguments (single 8192 and packed 4x2k where the path differs; 5120 where it differs): native vs the Python prepared route (keep that route reachable for this test only, as grouped_matmul_ttnn/generic) max abs diff == 0 on every output (fabric programs: every chip), repeats on fresh buffers == 0, program-cache growth after warm-up on fresh buffers / new runtime values == 0, host us a warm call native vs Python, device us both (real-time profiler).
- Record native_mismatch_<case>, native_repeat_<case>, native_cache_growth_<case>, native_host_us_<route>_<case>, native_device_us_<route>_<case>.
- Report in results/<task>_layer.md (one row per program: host us before / after, calls a chunk, host ms a chunk saved).
- New kernels: bounded waits, every awaited semaphore signalled, NOC writes inside buffers, fabric payloads <= 6208 B.
- Run only the tests you add plus the gate; no fork regression suites on the 8x4.
