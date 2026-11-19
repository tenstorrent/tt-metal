# Source-only diagnosis: native event type ABI and stopped flush observer

The native extension and the loaded TTNN extension use different nanobind type registries. This is a supported integration defect sufficient to prevent conversion of TTNN's `MeshEvent` into the native `EventQuery` argument. The precise exception from the live observer has not been exported; its likely `TypeError` text is inferred, not observed.

## Evidence captured before repair

CPU-only `strings` inspection of the native extension and checkout TTNN binary found these tags:

| Binary | nanobind ABI tag |
|---|---|
| `tt/_benchmark_event_query.abi3.so` | `v20_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1_stable` |
| `build/lib/_ttnn.so` (also copied to `ttnn/ttnn/_ttnn.so`) | `v20_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1` |

`tt/benchmark_native/CMakeLists.txt:17` used `nanobind_add_module(_benchmark_event_query STABLE_ABI event_query.cpp)`. Its generated `build/CMakeFiles/_benchmark_event_query.dir/flags.make` includes `-DPy_LIMITED_API=0x030C0000`. The nanobind source used by both builds, `.cpmcache/nanobind/7965351fcedc97e6ade403d51a150a266d188dc6/src/nb_abi.h:79–85`, explicitly states that limited and non-limited API modules can coexist but cannot communicate because their nanobind data structures differ, and adds `_stable` under `Py_LIMITED_API`.

`ttnn/CMakeLists.txt:372` names its low-level target `nanobind-static-abi3`, which is misleading if used as evidence for the actual module ABI. The emitted ELF tag above establishes the checkout module's non-limited ABI.

`ttnn/cpp/ttnn-nanobind/events.cpp:27` registers the concrete `MeshEvent` type. The `record_event` binding at line 38 uses `record_mesh_event`; `ttnn/core/events.cpp:17–24` returns a real `MeshEvent` after `enqueue_record_event_to_host`. Thus the return is neither a wrapper tuple nor a non-host-notifying marker. The extension's `tt/benchmark_native/event_query.cpp` accepts `const tt::tt_metal::distributed::MeshEvent&`. Its GIL release call guard is normal for such a bound native function and does not reconcile separate type registries.

## Failure propagation

`tt/benchmark_phases.py:132–148` uses one observer thread for query polling and flush-file control. `_poll_once()` calls the native conversion/query before examining the flush file. Any conversion exception is caught, appended to the in-memory `_errors`, and followed by `_stop.set()`. The observer exits without flushing and never services later `.flush` requests. This explains an unchanged initial snapshot and no serving-process traceback while ordinary generation continues.

The last initial snapshot, outstanding `.flush` marker, server log, corrected pilot, and old native binary should be retained before restarting. No benchmark phase completion measurements can be recovered by inventing timestamps after the fact.

## Minimal repair and verification

1. Remove `STABLE_ABI` from this model-local native target, cleanly rebuild using the checkout's same nanobind and Python environment, and verify its ELF ABI tag exactly matches `_ttnn.so`. Preserve the old binary outside import search; ordinary CPython extension suffix and stale `.abi3.so` must not make provenance ambiguous.
2. Keep control-file servicing available after a query failure. Record the first error and disable further native queries, but allow a post-measurement flush to export errors and pending records. An immediate failure snapshot may also preserve diagnostics, explicitly marking the attempt invalid. Do not silently acknowledge a flush unless export succeeded.
3. Before accuracy, execute a small owned-server smoke containing both prefill and asynchronous decode and demand a fresh acknowledged snapshot with both phases, no query errors, and no pending tickets. Import success alone cannot test native argument conversion.

The source-only audit did not import target modules, open devices, invoke RPC, attach a debugger, signal a process, or modify source. It did inspect `/proc/3013006/status` to confirm the named engine was live and untraced.

## Live snapshot recovery limitation

The live server exposes `/collective_rpc`, but its JSON router only accepts named worker methods and serialized string arguments. `vllm/v1/serial_utils.py:485` resolves string methods with a single `getattr`, without dotted traversal. No existing worker method was found that exports the model-local recorder, and `PhaseRecorder.close()` is not registered with `atexit` or called by the adapter. Therefore the existing flush marker cannot safely revive the dead observer, and merely editing the file will not replace the loaded class. Debugger injection could execute Python in the live process, but has thread/GIL and execution risks; it is not recommended for this idle owned server when a bounded restart is available. Preserve the external evidence and state explicitly that the original in-memory exception was not exported.
