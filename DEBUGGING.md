<!--
SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
SPDX-License-Identifier: Apache-2.0
-->

# TT‑Metal Debugging Guide

This guide consolidates the debugging and profiling tools available in this repository, with quick commands and links to their docs/code. It also includes a symptom → tool cheat sheet.

> Build tip: profiling tools (Tracy + device profiler) require a profiler build. Use:
>
> ```bash
> ./build_metal.sh --enable-profiler
> ```

## Cheat Sheet: Symptoms → Tools

- Program hangs or makes no progress
  - Enable Watcher to catch NoC/asserts and capture waypoints: `TT_METAL_WATCHER=10 python your_script.py` → see `generated/watcher/watcher.log`.
  - If hang wasn’t run with Watcher, dump after the fact: `./build/tools/watcher_dump -d=0 -w`.
  - Get active callstacks from RISCs: `python scripts/debugging_scripts/dump_callstacks.py --active_cores`.
  - Add kernel prints to narrow down: `TT_METAL_DPRINT_CORES=0,0 python your_script.py` and use `DPRINT` in kernels.
  - One‑shot triage run: `python scripts/debugging_scripts/tt-triage.py`.

- NoC errors, bad coordinates, mismatched counters
  - Watcher will report NoC violations with last waypoints.
  - Verify NoC node mapping: `python scripts/debugging_scripts/check_noc_locations.py`.
  - Check counters vs registers: `python scripts/debugging_scripts/check_noc_status.py`.

- Performance regression or throughput drop
  - Generate OPs report + Tracy capture: `python -m tracy -r -p -m pytest tests/ttnn/…::test_case`.
  - Include device timing logs: `TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -m pytest …`.
  - Capture NoC events: `python -m tracy -r --collect-noc-traces -m pytest …`.
  - Post‑process device logs: `python tt_metal/tools/profiler/process_device_log.py`.

- Kernel assert tripped
  - Watcher will halt and print assert details and last waypoint. Inspect `generated/watcher/watcher.log` and the stderr message.
  - Add DPRINTs around the assert location to increase context.

- Need to reproduce/share a failing run
  - Capture & replay LightMetal binary. Capture with LightMetal APIs in your test; replay with: `./build/tools/lightmetal_runner /path/to/capture.lmb`.

- Verify binaries on device match uploaded images
  - `python scripts/debugging_scripts/check_binary_integrity.py`.

- General device health (ARC heartbeat, uptime)
  - `python scripts/debugging_scripts/check_arc.py`.

- Inspect host runtime (program/kernel states, compile/cache info)
  - Enable Inspector during run: `TT_METAL_INSPECTOR=1 python your_script.py`.
  - Parse logs: `python scripts/debugging_scripts/parse_inspector_logs.py generated/inspector`.

- Device needs reset or system snapshot
  - Use TT‑SMI: `tt-smi -tr 0` (device 0 soft reset) and see the TT‑SMI README for snapshots.
  - Note (Grayskull): soft resets skew timers for profiling; do a full host reboot before profiling.

---

## Tools Catalog (purpose • example • links)

### Watcher (device monitor + on‑device debug)
- Purpose: Periodically inspects device state, validates NoC transactions, supports device asserts/waypoints/ring buffer, and logs to aid hang triage.
- Example: `TT_METAL_WATCHER=10 python your_script.py`
- Docs: [docs/source/tt-metalium/tools/watcher.rst](docs/source/tt-metalium/tools/watcher.rst)

### Kernel Debug Print (device → host prints)
- Purpose: Print variables, circular buffer slices, and formatted text from kernels to host logs/files for precise tracing.
- Example: `TT_METAL_DPRINT_CORES=0,0 python your_script.py`
- Docs: [docs/source/tt-metalium/tools/kernel_print.rst](docs/source/tt-metalium/tools/kernel_print.rst)

### Inspector (host runtime logging)
- Purpose: Lightweight, on‑by‑default host‑side logging of program/kernel lifecycle and state; produces queryable YAML logs.
- Example: `TT_METAL_INSPECTOR=1 python your_script.py`
- Docs: [docs/source/tt-metalium/tools/inspector.rst](docs/source/tt-metalium/tools/inspector.rst)

### Watcher Dump (offline hang dump)
- Purpose: Dumps watcher state (and optional NoC transfer data) from devices after a hang, even if original run lacked Watcher.
- Example: `./build/tools/watcher_dump -d=0 -w`
- Code: [tt_metal/tools/watcher_dump/watcher_dump.cpp](tt_metal/tools/watcher_dump/watcher_dump.cpp)

### Tracy Profiler (host+device profiling; OP report)
- Purpose: End‑to‑end profiling of Python/C++ and device RISCs; emits an OPs CSV and supports live viewing in Tracy GUI.
- Example: `python -m tracy -r -p -m pytest tests/ttnn/sweep_tests/test_all_sweep_tests.py::test_matmul[matmul.py-0]`
- Docs: [docs/source/tt-metalium/tools/tracy_profiler.rst](docs/source/tt-metalium/tools/tracy_profiler.rst)
- Module: [ttnn/tracy/__main__.py](ttnn/tracy/__main__.py)

### Device Program Profiler (scoped device timing)
- Purpose: Adds scoped timing markers to device programs; emits per‑RISC/per‑core timings and integrates with Tracy.
- Example: `TT_METAL_DEVICE_PROFILER=1 ./build/programming_examples/profiler/test_full_buffer`
- Docs: [docs/source/tt-metalium/tools/device_program_profiler.rst](docs/source/tt-metalium/tools/device_program_profiler.rst)
- Post‑proc scripts: [tt_metal/tools/profiler](tt_metal/tools/profiler)

### NoC Event Traces (profiler add‑on)
- Purpose: Capture NoC events alongside profiler logs for post‑run analysis.
- Example: `python -m tracy -r --collect-noc-traces -m pytest tests/ttnn/…::test_case`
- Code: [tt_metal/tools/profiler/noc_event_profiler.hpp](tt_metal/tools/profiler/noc_event_profiler.hpp)

### LightMetal Capture/Replay (binary replay)
- Purpose: Capture host API + device CQ workloads into a portable LightMetal binary and replay for repro/triage (experimental).
- Example (replay): `./build/tools/lightmetal_runner /path/to/capture.lmb`
- Runner: [tt_metal/tools/lightmetal_runner/lightmetal_runner.cpp](tt_metal/tools/lightmetal_runner/lightmetal_runner.cpp)
- APIs: [tt_metal/api/tt-metalium/host_api.hpp](tt_metal/api/tt-metalium/host_api.hpp)

### tt‑triage (aggregated diagnostics)
- Purpose: Run a suite of checks (ARC, NoC, callstacks, integrity) and visualize results; integrates with tt‑exalens.
- Example: `python scripts/debugging_scripts/tt-triage.py --run check_noc_status`
- Docs: [scripts/debugging_scripts/tt-triage.md](scripts/debugging_scripts/tt-triage.md)
- Entry: [scripts/debugging_scripts/tt-triage.py](scripts/debugging_scripts/tt-triage.py)

### Parse Inspector Logs (post‑run analysis)
- Purpose: Convert Inspector YAML logs into structured data for scripts or manual inspection.
- Example: `python scripts/debugging_scripts/parse_inspector_logs.py generated/inspector`
- Code: [scripts/debugging_scripts/parse_inspector_logs.py](scripts/debugging_scripts/parse_inspector_logs.py)

### Dump Callstacks (RISCs)
- Purpose: Dump callstacks for RISCs on active cores (optionally via gdb), plus dispatcher metadata.
- Example: `python scripts/debugging_scripts/dump_callstacks.py --active_cores`
- Code: [scripts/debugging_scripts/dump_callstacks.py](scripts/debugging_scripts/dump_callstacks.py)

### Check NoC Status (vars vs registers)
- Purpose: Validate firmware NoC counters against hardware registers to spot inconsistencies.
- Example: `python scripts/debugging_scripts/check_noc_status.py`
- Code: [scripts/debugging_scripts/check_noc_status.py](scripts/debugging_scripts/check_noc_status.py)

### Check NoC Locations (node IDs)
- Purpose: Ensure NoC node IDs match expected logical coordinates for worker/ETH blocks.
- Example: `python scripts/debugging_scripts/check_noc_locations.py`
- Code: [scripts/debugging_scripts/check_noc_locations.py](scripts/debugging_scripts/check_noc_locations.py)

### Check Binary Integrity (firmware/kernels)
- Purpose: Verify device‑resident binaries match what was uploaded (integrity/relocation sanity).
- Example: `python scripts/debugging_scripts/check_binary_integrity.py`
- Code: [scripts/debugging_scripts/check_binary_integrity.py](scripts/debugging_scripts/check_binary_integrity.py)

### ARC Health Check
- Purpose: Confirm ARC heartbeat is running and at sane rate; report uptime and clock.
- Example: `python scripts/debugging_scripts/check_arc.py`
- Code: [scripts/debugging_scripts/check_arc.py](scripts/debugging_scripts/check_arc.py)

### Profiler Post‑Processing (artifacts → reports)
- Process Device Logs: Parse device profiler CSV into stats/JSON/TXT.
  - Example: `python tt_metal/tools/profiler/process_device_log.py`
  - Code: [tt_metal/tools/profiler/process_device_log.py](tt_metal/tools/profiler/process_device_log.py)
- Process OPs Logs: Consolidated OP‑level performance CSV from Tracy logs.
  - Example: `python tt_metal/tools/profiler/process_ops_logs.py`
  - Code: [tt_metal/tools/profiler/process_ops_logs.py](tt_metal/tools/profiler/process_ops_logs.py)

### TT‑SMI (external)
- Purpose: Reset devices, dump system information, and take snapshots for hardware triage.
- Example: `tt-smi -tr 0`
- Repo: https://github.com/tenstorrent/tt-smi

---

## Notes & Requirements

- Profiler build: Tracy + device profiling require `./build_metal.sh --enable-profiler`.
- NoC transfer logging: to dump post‑run with watcher_dump, enable recording during the run via `TT_METAL_RECORD_NOC_TRANSFER_DATA=1`.
- tt‑triage dependencies: install tt‑exalens if missing via `./scripts/install_debugger.sh`.
- Grayskull profiling caveat: after `tt-smi`/`tensix_reset`, timers skew; do a full host reboot before profiling. Wormhole doesn’t have this issue.

