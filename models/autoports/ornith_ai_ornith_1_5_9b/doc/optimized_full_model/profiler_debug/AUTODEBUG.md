# AutoDebug: stage7 profiler source-ID collision

## Finding

**The current capture contains the same genuine 16-bit collision documented during full-model profiling: `rd_kick` and `pp_kdec` both hash to 1831. The stage7 invocation omitted the existing source-path overlay. Reuse that overlay from the model directory, with absolute output paths.** No implementation change or cache purge is needed to address this identified collision.

This is a host profiler catalogue failure. The source map is built before `DeviceProfiler::readResults` reads the device profiler buffers (`tt_metal/impl/profiler/profiler.cpp:2571–2584`). It does not establish a model numerical failure, device failure, or successful measured replay.

Investigation was inspection-only: source/log/cache reads, SHA256 comparisons, local text/hash calculations, shell syntax validation, and a host-preprocessor check of symlink source spelling. No devices, target execution, reset, implementation changes, or AutoDebug launcher were used. Repository HEAD was `ebd521f744bb61f787a164f6b207c6defb8dd306`; the workspace is dirty and model sources have changed since the failing run. The only repository file written by this investigation is this report.

Paths below are repository-relative unless an absolute path is given. `MODEL` denotes `models/autoports/ornith_ai_ornith_1_5_9b`. Captured evidence remains under `/home/ttuser/ornith-9b/artifacts`.

## Direct observations

### Exact failure and source identities

`MODEL/doc/optimized_full_model/commands.jsonl:19` records the failed command at 12:05:55–12:06:07 UTC on 2026-11-13:

```text
timeout 1800 python -m tracy -r -p -v --no-web-server
  -o /home/ttuser/ornith-9b/artifacts/optimized_full_model_masked_before
  -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_perf_probe
  --profile-reduced --profile-batch 32 --context 512
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/profile_masked_before.json
```

Its recorded environment has no `TT_METAL_KERNEL_PATH`. This is meaningful: `tests/stage7_run.py:27` records every `TT_` variable, and its subprocess inherits the environment (`:35–36`). The wrapper does **not** record cwd or `PYTHONPATH`; neither should be invented from this manifest. Repository cwd is consistent with the relative result path landing at the expected location, but it is an inference.

The failure log shows the first prefill warmup at `logs/profile_masked_before.log:248`, then the collision at `:249–250`. The stack reaches `full_model_perf_probe.py:80`, `ttnn.ReadDeviceProfiler(mesh)` (`log:252–278`). The JSON reports `phase: load_prepare_warm`, layers `[0,3]`, and zero completed drains. The increment occurs after the read (`full_model_perf_probe.py:80–81`), so zero drains does not mean the drain safeguard was absent. The signposted measurement at `full_model_perf_probe.py:135–141` was never reached.

Let `K` be the absolute prefix:

```text
/home/ttuser/ornith-9b/tt-metal/ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels
```

The hashed strings, with `K` expanded before hashing, are:

```text
rd_kick,K/dataflow/reader_chunk_gdn_prep.cpp,281,KERNEL_PROFILER
pp_kdec,K/compute/chunk_gdn_math.hpp,1004,KERNEL_PROFILER
```

| Marker | Actual source | FNV-1a 32-bit result | Folded 16-bit result |
|---|---|---|---|
| `rd_kick` | `reader_chunk_gdn_prep.cpp:281` | `0x4e8249a5` | `0x0727` = **1831** |
| `pp_kdec` | `chunk_gdn_math.hpp:1004` | `0x7e267901` | `0x0727` = **1831** |

Both source lines still contain those markers. In the failed capture's `.logs`, `zone_src_locations.log:17` contains the original-path `rd_kick`; `new_zone_src_locations.log:936` contains `pp_kdec`, the first conflicting entry when loading the saved catalogue followed by new entries.

Static audit of those exact files found:

| Capture | Saved catalogue | New catalogue | Combined unique strings | Distinct hashes | Colliding buckets |
|---|---:|---:|---:|---:|---:|
| `optimized_full_model_masked_before` | 27 rows / 27 unique | 1202 rows / 39 unique | 39 | 38 | 1: hash 1831 |
| Prior `full_model_tracy_final` | 63 rows / 63 unique | 4366 rows / 63 unique | 63 | 63 | 0 |

There were no malformed marker entries in the failed catalogue. The repeated rows are expected across kernels/RISCs and are not the collision.

### Source and cache agree

`kernel_profiler.hpp:38–43` constructs the identity as `name,__FILE__,__LINE__,KERNEL_PROFILER` and hashes it on the device compilation path. Its `Hash32_CT`/`Hash16_CT` (`:128–135`) matches the host implementation in `profiler.cpp:291–297`: FNV-1a with 32-bit wraparound, then XOR of the upper and lower 16-bit halves.

`populateZoneSrcLocations` extracts the complete pragma string (`profiler.cpp:308–317`). Exact duplicates are admitted; a newly distinct string with an occupied 16-bit ID throws (`:319–321`). This is the correct rejection of an ambiguous ID. Removing this check would permit incorrect marker attribution.

The JIT cache contains matching compiler diagnostics, not merely similar present-day source:

```text
/home/ttuser/ornith-9b/tt-metal-cache/tt-metal-cache9957800640679114378/kernels/
  reader_chunk_gdn_prep/16605874887825011506/ncrisc/ncrisck.o.log:16
    -> original-path rd_kick, line 281
  reader_chunk_gdn_prep/7197736802637641925/ncrisc/ncrisck.o.log:16
    -> overlay-path rd_kick, line 281
  compute/9744737704604605877/trisc0/trisck.o.log:128
    -> original-path pp_kdec, line 1004
```

Each reader cache entry's `kernel_includes.hpp:2` names the corresponding original or overlay path, and its `ncrisc/ncrisck.o.dephash:96` records the same source content hash, `3920730861473388043`. These November 13 cache artifacts independently corroborate the path-only intervention. The failed log's compile commands use this same cache-root suffix (`9957800640679114378`). The captured zone files are the authoritative inputs for the observed throw; these cache examples do not prove that these specific objects were individually identified in an execution trace.

`JitBuildState::extract_zone_src_locations` copies `KERNEL_PROFILER` diagnostic lines from cached `*.o.log` files into the current catalogue (`tt_metal/jit_build/build.cpp:1015–1034`). Extraction runs on ordinary cache reuse as well as builds (`:1162–1164`), and on the warmed-ELF early-return path (`:1072`). Consequently a run can reproduce this collision with almost no recompilation; the failed log reports 462/469 JIT hits (`log:361`).

### The existing overlay is valid, and cwd determines whether it wins

All five entries beneath `MODEL/doc/optimized_decoder/profiler_kernel_overlay` are currently symlinks to their corresponding repository sources. `samefile` and byte/SHA256 comparisons pass for all five. They comprise the reader `.cpp` and four sibling `.hpp` files. The reader/source SHA256 is:

```text
cf494cf6efe2da1886e5e901c5958013d84a496a832a1ca33fff03cf55ca736d
```

The workaround changes compiler source identity while retaining source bytes. It is not a change to the kernel algorithm; instrumentation values and compiled artifacts can change.

`tt_metal/impl/kernels/kernel.cpp:53–87` resolves a relative source in this order: cwd, `TT_METAL_KERNEL_PATH`, system kernel directory, runtime root. Absolute source paths pass through immediately. The chunk-GDN fused/phased program factories request this reader through a relative repository path (`chunk_gdn_fused_program_factory.cpp:338`, `chunk_gdn_phased_program_factory.cpp:296`). That path exists beneath repository cwd, and does not exist beneath model cwd. With the runtime root pointing at this checkout and no competing system reader, as in the inspected setup:

| cwd / environment | Selected reader source |
|---|---|
| Repository cwd, with or without overlay variable | Original repository path |
| Model cwd, overlay variable omitted | Original source through fallback |
| Model cwd, existing overlay variable set | Overlay path |

The resolver preserves the symlink spelling; it does not canonicalize to the target (`kernel.cpp:69–74`). The generated include uses the resolved path (`tt_metal/jit_build/genfiles.cpp:60–70`). Therefore `__FILE__` changes.

For the overlay path used in the prior successful manifest, `rd_kick` hashes to **48724** (`0xbe54`, FNV32 `0x8ea230f6`), while `pp_kdec` remains **1831**. The prior final capture actually contains these exact two identities (`full_model_tracy_final/.logs/zone_src_locations.log:17,28`). Static substitution of the two affected strings in the failing 39-string catalogue changes `rd_kick: 1831→48724` and `rd_item: 51010→51792`, leaving **39 distinct hashes and no collision**. This predicts removal of the identified failure; it does not prove that later B32 warmup or replay will succeed.

The resolved source directory also enters `Kernel::compute_hash` through `process_include_paths` (`kernel.cpp:435–440,889–897`), and the compiled kernel hash selects the cache directory (`tt_metal/impl/program/program.cpp:3417–3421`). There is no evidence requiring a cache purge or forced recompilation. The resolved directory participates in the cache key, and the inspected cache contains separate original/overlay reader entries.

## Controlled contrast and causal interpretation

`MODEL/doc/full_model/work_log.md:249–270` is supported by the artifacts:

1. The earlier overlay attempt still collided. Its cwd was not recorded; repository cwd is inferred from its relative output and the work-log explanation. Independently, resolver precedence proves that setting the variable alone cannot make the overlay win from repository cwd.
2. `reduced_profile_accepted_command.json` records model cwd plus overlay. Its log passes source-map construction, then reports dropped profiler markers.
3. `reduced_profile_final_command.json:20–29` records model cwd, overlay, and runtime-plus-repository `PYTHONPATH`; the runner drained eager preparation profiles before measurement. The manifest exits 0, and the final result reports `status: pass`, 21 warmup drains, exactly one model/sampling replay, and no refresh/readback/recapture inside the measured window. The archived final log has no collision or dropped-marker warning.
4. Stage7 retains that drain mechanism but omitted the source overlay. Its first drain encounters the original-path pair and throws before buffer ingestion. A second collision during cleanup is consistent with rebuilding the same map; it does not establish a second collision pair.

The previous successful profile used default capacity 1 and context 2048; this failed command requests capacity 32 and context 512. Current model and runner revisions also differ from the prior successful run. The source/capture/hash evidence establishes the profiler identity problem independently of these model differences. Prior success is not proof of current capacity-32 readiness.

| Candidate | Evidence and intervention | Assessment |
|---|---|---|
| Genuine original-path 16-bit collision, existing overlay omitted | Exact two strings in captured inputs, matching source/cache diagnostics and hash algorithm; restore overlay plus effective cwd | **Confirmed immediate cause; leading diagnosis** |
| Historical stale saved catalogue | Saved/new files alone could create this symptom generally, but both colliding identities occur in this run's new catalogue; Tracy clears selected `.logs` at startup | Unnecessary to explain this run |
| Duplicate rows or malformed/interleaved cache output | Duplicates are explicitly allowed; all current entries parse and exactly one distinct pair collides | Refuted for captured inputs |
| Warmup buffer overflow | Established in an earlier run after resolving source IDs; current throw precedes profiler-buffer reads and existing drain completion | Separate follow-up risk, not this throw |
| Model masking/precision/device execution bug | No such precondition is needed for the deterministic host catalogue rejection | Not established by these artifacts |

## Exact reusable invocation proposed for the calling session

**Not executed by AutoDebug.** The essential change is model-directory cwd plus the existing absolute `TT_METAL_KERNEL_PATH`. Retain the current `--profile-batch 32 --context 512`, drain hook, and absolute result path. This example also uses a unique attempt name and presets the profiler output environment before Tracy import to avoid the separate stale default-copy path described below.

```bash
set -euo pipefail
repo=/home/ttuser/ornith-9b/tt-metal
model_dir="$repo/models/autoports/ornith_ai_ornith_1_5_9b"
runtime_dir=/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/runtime
run_name="profile_masked_before_overlay_$(date -u +%Y%m%dT%H%M%SZ)_$$"
capture_dir="/home/ttuser/ornith-9b/artifacts/$run_name"
result_json="$model_dir/doc/optimized_full_model/$run_name.json"
test ! -e "$capture_dir"
test ! -e "$result_json"

cd "$model_dir"
export PYTHONPATH="$runtime_dir:$repo"
export TT_METAL_HOME="$repo" TT_METAL_RUNTIME_ROOT="$repo"
export TT_METAL_KERNEL_PATH="$model_dir/doc/optimized_decoder/profiler_kernel_overlay"
export TT_METAL_PROFILER_DIR="$capture_dir"
export TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export TRACY_NO_WEB_SERVER=1 OMP_NUM_THREADS=8

"$repo/python_env/bin/python" -m models.autoports.ornith_ai_ornith_1_5_9b.tests.stage7_run \
  "$run_name" timeout 1800 "$repo/python_env/bin/python" -m tracy \
  -r -p -v --no-web-server -o "$capture_dir" \
  -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_perf_probe \
  --profile-reduced --profile-batch 32 --context 512 --output "$result_json"
```

This preserves the installed environment's existing JIT-cache settings. `TT_METAL_RUNTIME_ROOT` supplies the C++ runtime root; `TT_METAL_HOME` supplies Tracy/tool paths. `PYTHONPATH` matches the prior successful invocation. Keep the exact overlay directory spelling: relocating or canonicalizing it changes source hashes and requires a fresh collision audit.

Do not blindly reuse the repository-relative `--output` from the failed command after changing cwd: it would write beneath a second `models/...` subtree inside the model directory.

## Evidence safeguards and acceptance conditions

- **Preserve existing captures before any retry.** Tracy `run_report_setup` deletes the selected `<output>/.logs` recursively before capture (`tools/tracy/__init__.py:96–105`). A new attempt name protects the failed and prior successful catalogues, CSVs, host traces, and command/log files. Do not seed the new catalogue from the failed capture: retaining the original saved `rd_kick` would preserve its collision with `pp_kdec` even if new reader entries use the overlay. Do not edit marker rows to suppress the collision: then device IDs and host metadata could disagree.
- Record the actual child cwd, exact argv, interpreter, effective `PYTHONPATH`, relevant `TT_`/`TRACY_` flags, overlay symlink spelling/targets, source hashes, and start/end times. `stage7_run.py` already records many fields but omits cwd, interpreter metadata, `PYTHONPATH`, kernel sources, and overlay identity. Supplement its manifest without dumping unrelated environment values.
- Snapshot relevant source/cache evidence and hashes before and after a retry. `commands.jsonl:19` records failing-run generator SHA256 `8fd30f9c758cc5d3fae9a3c38028195c27127c57648af3d5cac83b330ea8b61c` and model SHA256 `ab04b74a86d0714bd64acca2315f91f8582840644fe3d5e7b53b83ebebbf76ec`; current disk hashes differ. The current perf runner still matches that run (`03e1f1921099abc511e5f6831e7cc299b7960f3eab92cd053a733a1bc3fc5132`). A retry of the current tree must be labeled as such, rather than represented as an exact historical rerun.
- Retain eager warmup draining and restoration before measurement (`full_model_perf_probe.py:76–84,112–113,335–337`). After collection, audit the actual new/saved catalogue union, rather than relying solely on the static substituted subset. Capacity 32 can introduce additional marker identities later in warmup.
- Accept only a passing workload JSON and completed collection/report with no collision, dropped-marker warnings, or missing-device-data error. Verify the intended signposts and existing one-model/one-sampler counter assertions, plus zero refresh/readback/recapture in the window. The JSON's top-level `batch: 1` is hard-coded (`full_model_perf_probe.py:60`); record requested capacity 32 and actual active rows separately, rather than labeling this as 32 active requests.

Inspection-time SHA256 anchors:

| File | SHA256 |
|---|---|
| Failed `profile_masked_before.json` | `33f7089912d623d85dc047a698c1a5bd22d7cd5e4f036d7dc3f0ddc982fb75c9` |
| Failed `logs/profile_masked_before.log` | `edd2e46f9a77e2cf7325ce34c40df157bc95941d61650e20f123fabd5f6da338` |
| Failed capture `.logs/zone_src_locations.log` | `ad24ec01d5c4f69f2a555b4e2b1419abcc41ae6b442178683976eb25964ba816` |
| Failed capture `.logs/new_zone_src_locations.log` | `89b1948f1872385307abae3e7c6bd1aa7dc4311e168395abd4f2a4e0b04f06a5` |
| `tt_metal/impl/profiler/profiler.cpp` | `b8b1ea765325b104b5f5732fc088e51d008aeac2e1ccfc5104b605954dd1e7b8` |

## Other potential issues and limits

**Separate demonstrated Tracy path bug:** `tools/tracy/common.py:21–23,61` calculates `PROFILER_LOGS_DIR` at import. `__main__.py:219–225` later applies `-o`, but its copy at `:472` still uses the imported constant. The failed log consequently warns about copying from `generated/profiler/.logs` (`log:377–378`), even though the real capture exists under the explicit output directory. Presetting `TT_METAL_PROFILER_DIR` to the same absolute `-o` path before starting Python avoids the discrepancy. `--no-web-server` does work; the failed log confirms the UI was disabled (`:380`). This copy issue occurs after the collision and cannot cause it.

**Downstream report error:** Tracy's workload return-code check is opt-in (`tools/tracy/__main__.py:164–168,428–430`). This command lacks `--check-exit-code`, so report generation continues after the workload fails and ends with missing device rows (`log:386–404`). The captured device CSVs contain headers only. This is consistent with source-map failure before data ingestion; it is not a second model diagnosis. `--check-exit-code` can expose the workload failure earlier, but its failure branch exits before normal capture-process waiting, so it is not included as an unqualified cleanup improvement in the recipe.

**Profiler design limitation:** the catalogue uses a 16-bit namespace with no collision resolution. Names, absolute source paths, and line changes can create another genuine collision. A durable shared fix would require coordinated device/host identity handling or a collision-safe ID scheme; merely widening the host map, ignoring the exception, stripping filenames, or rewriting log hashes would not fix device-emitted IDs. Improving the exception to print the ID and both full strings would make future diagnoses easier. None of these implementation changes was made.

Runtime proof remaining: the proposed invocation must be collected by the calling session on its authorized hardware to establish that all subsequent capacity-32/context-512 preparation, trace replay, buffer capacity, and CSV/report generation succeed. The exact current failure and the existing overlay's removal of its collision are proven from captured inputs and source without that execution.

Report review: after drafting, rechecked the exact strings/counts/hashes, paired cache includes/dependency content hashes, resolver order, current drain/error sequence, historical passing contrast, invocation paths, and shell syntax. Independent investigators reviewed the draft; the overlay FNV32 transcription was corrected. The separate Tracy path/report issues remain secondary because they occur after the headline failure.
