# Honor accurate exponential mode in sparse SDPA probabilities

Evidence class: **hardware-reproduced numerical error on pinned upstream source**.

The sparse SDPA main probability calculation hard-codes the approximate exponential even though the online correction observes the requested mode. Pass the mode into the main helper, initialize accurate-mode constants, and preserve FP32 scaling when a scale is not BF16-exact. The helper default remains approximate for existing callers.

Historical Blackhole runtime base `70082ba2f706972c9e3812f56635ac5118dff8a0`: the two-key example changed from `0.7187500` to `0.7265625` against `sigmoid(1) = 0.7310586`. Fixed absolute error was `0.0044961`, below the unchanged `0.005` gate. FP32-destination normalized L2 improved from `0.025599/0.025718` to `0.008853/0.008302` for the exact/inexact scales. Approximate-mode outputs in four checked configurations were bitwise unchanged. `sparse_sdpa_msa` is outside this fix.

## Hardware and scope

Target: one Blackhole P150a, or one selected chip of a Blackhole P300. The fresh pair was measured on one selected p300c chip; P150a and cross-card generality are separate coverage. No four-card setup is needed. For the standalone examples device 0 is used; pytest accepts `--device-id`.

The branch is one proposed commit above upstream main `4ff4adaa690abe880c08496e4a3f85d7dcaf314c`. Fresh result: one p300c chip; stock suite 2/4 pass and fixed 4/4 pass; standalone example fails stock and passes fixed. Historical results above belong to the older source stated explicitly.

Use a standard tt-metal development host with the hardware prerequisites from [INSTALLING.md](../../INSTALLING.md). From this branch's repository root, build the matching runtime and Python package:

```sh
git submodule update --init --recursive
./build_metal.sh --release
./create_venv.sh
source python_env/bin/activate
```

To see the bug, run the commands below with the two production files restored from the parent commit (the test and example stay):

```sh
git checkout HEAD~1 -- ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sparse_sdpa_compute.cpp
```

Restore the fix with `git checkout HEAD --` on the same two files. Use a new JIT cache directory for each arm (`TT_METAL_CACHE`), as in our validation, so each arm starts cold and keeps its own compiled kernels for comparison.

## Run and expected outcome

```sh
timeout --signal=TERM --kill-after=10s 900s python -m pytest -q tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa_exp_mode.py --device-id 0
timeout --signal=TERM --kill-after=10s 180s python ttnn/ttnn/examples/usage/sparse_sdpa_exp_mode.py
```

A successful command exits 0; an assertion or failed numerical oracle exits nonzero. GNU timeout exits 124 when its deadline expires (or 137 if killing is required). An unexpectedly stalled card may need `tt-smi -r` after the process is stopped; do not reset a card used by another job.

The four regression cases cross BF16-exact/inexact scale with BF16/FP32 destination accumulation. Each compares represented BF16 inputs against the CPU reference and checks the requested mode. The standalone two-key example checks every element and exits 1 above absolute error 0.005.

Runtime: Suite wall times including cold JIT: 16.5 s stock and 15.7 s fixed. Example: 8.1 s stock and 7.4 s fixed. The 900-second suite and 180-second example deadlines are bounds, not measured durations. Kernel JIT compilation can dominate a cold first run.

## Additional coverage

The paired suite establishes the accurate-mode result for these four scale/destination configurations. Cross-arm approximate-mode output equivalence is a separate measurement, summarized in the PR description. P150a, sparse_sdpa_msa, other architectures and model quality are outside this pair.

For one visible chip of a P300 board, the upstream runtime requires a one-chip mesh descriptor: set `TT_MESH_GRAPH_DESC_PATH` to `tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto`. Select the physical chip with `TT_VISIBLE_DEVICES`; it appears as device 0 inside the process. Keep the stock and fixed JIT caches separate.
