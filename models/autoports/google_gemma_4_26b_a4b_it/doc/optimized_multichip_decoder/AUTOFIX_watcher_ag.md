# AutoFix: linear all-gather endpoint connection

## Starting evidence

`AUTOTRIAGE_watcher_ag.md` identifies the source contract explaining the
original final-default watcher failure. `watcher_failure/run.log` records the
one-worker minimal all-gather BRISC assertion on line 119 of
`FabricConnectionManager::get_forward_connection()`. The unguarded accessor
requires a forward connection even for a linear endpoint without one. The
reader's CRBW wait follows the writer's halt.

The original command was the layer-0 4096-token prefill / 128-token traced
decode check with cache validation and eight duplicate replays, under
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1`. Its exact arguments and
source hashes are in `final_watcher_sliding.command.json`; the failure log is
preserved separately so the final rerun can replace the command result.

## Recovery

The process had already aborted before Inspector capture could attach. Root
saved the watcher/run logs and captured all-device Ethernet and ARC status
with explicit device selection. `watcher_failure/tt-triage-devices.txt` shows
healthy active Ethernet links and ARC heartbeats. Root then serialized a
reset, device listing and 1x4 mesh smoke before resuming device experiments.
`watcher_failure/reset.log`, `list-before.log`, `list-after.log` and
`mesh-smoke.log` preserve recovery evidence; the smoke reports `MESH_SMOKE_OK`.

## Hypothesis experiments

**Hypothesis:** the non-mux writer looks up an absent directional connection
before applying its existing destination guards. Prediction: retaining the
same one-worker, linear, persistent-L1 path and guarding only that lookup
eliminates the assertion without changing collective output.

**Fix:** root applied `watcher_ag_endpoint.patch`. In
`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp`,
the connection pointer starts null and the existing checked accessor runs only
when `detail::valid_targets(direction)` holds. The actual fabric sends already
use this predicate or a forwarding-loop bound that is zero for absent routes.
Assertions remain enabled for connections that must exist. No precision,
worker-count, buffer, residual-layout or topology policy was changed.

**Focused verification:** root ran the two probe cases separately with
serialized hardware ownership. Equivalent reproduction commands for the
recorded arguments are:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/probe_watcher_all_gather.py --workers 1 --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/watcher_ag_attention.json
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/probe_watcher_all_gather.py --workers 1 --dtype bfloat8_b --channels 2 --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/watcher_ag_moe.json
```

| Payload | Per-rank logical input | Output | Eager mismatches | Eight traced replays |
| --- | --- | --- | --- | --- |
| BF16 attention | `[1,1,1,704]` | `[1,1,1,2816]` | 0 on every rank | 0 on every rank and replay |
| BF8 paired MoE | `[1,2,1,704]` | `[1,2,1,2816]` | 0 on every rank | 0 on every rank and replay |

Both use the real 1x4 mesh, persistent L1 output, full-grid semaphores and one
worker per direction. Both JSON files record `passed: true`, enabled watcher
environment and kernel SHA256
`14ca782e9b60777f75e3d5a8e3ed8700bc2508cc6af0b0efaac86941d35b490a`.
Their `.log` files show `AG_PASS` and orderly device teardown. The attention
probe's JIT report is 55/63 cache hits, documenting new kernel compilation.
There is no watcher assertion in either successful probe log.

**Verdict:** verified source-contract violation; focused native fix passes both
model-applicable payload cases under watcher. Original full-layer and stack
watcher gates are separate required checks and were still running when this
report was written.

## Durable regression

Added
`tests/ttnn/unit_tests/operations/ccl/test_all_gather_async_linear_endpoint.py::test_all_gather_async_linear_one_worker_endpoint`.
It uses the existing 1x4 mesh fixture and native full-grid semaphores, with no
model imports. Two parameters cover the BF16 single-channel and BF8 paired
shapes above. It warms both ping-pong signatures, checks exact output on all
four ranks, captures the one-worker persistent-L1 operation, and checks eight
trace replays. The test also runs with watcher disabled for copy correctness;
watcher must be enabled to catch the original device assertion.

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 python_env/bin/python -m pytest -q tests/ttnn/unit_tests/operations/ccl/test_all_gather_async_linear_endpoint.py
```

The investigation agent ran only Black (`--target-version py310 --line-length
120`), `py_compile`, and whitespace checks for the new test; all passed. Root
owns hardware execution of this pytest command and the remaining decoder gates.

## Build and final status

The patched device writer compiled through native JIT during the successful
watcher probes. Root also retried `.github/scripts/copilot-build.sh`; the
wrapper cannot run because Docker is unavailable, as recorded in
`watcher_failure/kernel_wrapper_build.log`. This is not a host-library build
claim. Root reports that formatting of the changed C++ lines passed.

The endpoint bug is fixed with focused watcher evidence. Full decoder watcher
reruns, the durable pytest run and final stage acceptance remain root-owned;
this report does not claim that those pending gates have passed.

## Final verification

The durable native regression ran under `TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 python_env/bin/python -m pytest -q tests/ttnn/unit_tests/operations/ccl/test_all_gather_async_linear_endpoint.py`: **2 passed in1.75s** (`watcher_ag_regression.log`). All four original/default model Watcher gates now exit0: `final_watcher_sliding`, `final_watcher_full`, `final_watcher_stack_samekind`, `final_watcher_stack_mixed`. See `final_watcher_summary.json` for commands, source/kernel hashes, PCC and log hashes. Watcher reports no disabled features. The endpoint assertion is fixed; no assertion or safety feature was disabled. The kernel source is unchanged after these checks.
