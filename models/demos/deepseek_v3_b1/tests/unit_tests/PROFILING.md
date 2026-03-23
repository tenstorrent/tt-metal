# Tracy Profiling for DeepSeek V3 B1 Unit Tests

This document describes known issues and fixes when running unit tests (e.g.
`test_attention_block`) with Tracy profiling enabled.

## Running with Tracy

```bash
python3 -m tracy -r --device -o generated/profiler/reports \
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_attention_block.py \
    -k "test_attention_block[...-8191-...]"
```

---

## Bug 1: `ValueError: stoull` crash during teardown

### Symptom

The test passes but pytest reports an `ERROR` during teardown:

```
ttnn/ttnn/distributed/distributed.py:689: ValueError
E       ValueError: stoull
```

### Root cause (two interacting issues)

**Race condition in `new_zone_src_locations.log`**

When kernels are compiled in parallel (`jit_build_subset` → `detail::async`), each
build thread calls `extract_zone_src_locations`, which runs:

```
grep KERNEL_PROFILER /path/to/kernel/*.o.log >> new_zone_src_locations.log 2>&1
```

Multiple `system()` calls execute concurrently with no synchronization. Since `>>`
append-mode does not guarantee atomic multi-line writes across processes, outputs
from different grep processes interleave at the byte level, producing corrupted
lines such as:

```
# Two lines written simultaneously — bytes overwrite each other mid-line:
...note: '#pragma message: NCRISC-KERNEL,/data/bliupragma message: CCL_SENDER_SEND,...'
```

**Unguarded `stoull` call**

`populateZoneSrcLocations` in `profiler.cpp` reads the log and parses each entry
as `zone_name,source_file,line_number,KERNEL_PROFILER`. A corrupted line causes
`line_num_str` to contain a file path instead of a number, so `std::stoull` throws
`std::invalid_argument`. This propagates through `close_mesh_device` to Python as
`ValueError: stoull`, aborting teardown before the profiler report is generated.

### Fixes

**`tt_metal/jit_build/build.cpp`** — serialize log writes with a mutex:

```cpp
static std::mutex zone_log_mutex;
// ...
std::lock_guard<std::mutex> lk(zone_log_mutex);
tt::jit_build::utils::run_command(cmd, NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG, ...);
```

**`tt_metal/impl/profiler/profiler.cpp`** — validate before calling `stoull`:

```cpp
if (line_num_str.empty()) {
    log_warning(tt::LogMetal, "Skipping malformed zone source location entry: {}", zone_src_location);
    continue;
}
try {
    line_num = std::stoull(line_num_str);
} catch (const std::exception& e) {
    log_warning(tt::LogMetal, "Skipping zone source location with invalid line number '{}': {}", ...);
    continue;
}
```

The mutex is the root-cause fix; the `try/catch` is a defensive fallback.

---

## Bug 2: `JSONDecodeError: Invalid control character` in report generation

### Symptom

After the test completes, `process_ops_logs.py` crashes:

```
json.decoder.JSONDecodeError: Invalid control character at: line 1 column 65460 (char 65459)
```

No `ops_perf_results_*.csv` is generated, or it is missing `COMPUTE KERNEL SOURCE`,
`DATA MOVEMENT KERNEL SOURCE`, `PROGRAM HASH`, and kernel size columns.

### Root cause (two interacting issues)

**Per-core kernel explosion in `kernel_info`**

The attention block op uses per-core compile-time args, which produces one `Kernel`
object per core. With 381 cores all sharing the same source file, `get_kernels_json`
emitted 381 identical-source entries — one per core — making `kernel_info` alone
~62 KB.

**Tracy message truncation breaks CSV quoting**

The Tracy op profiler wraps each message in backticks:

```
`TT_DNN_DEVICE_OP: "OpName", hash, device_id, false, op_id ->\n{json}`
```

The backtick is the CSV quotechar used by `tracy-csvexport`. Tracy's hard limit is
65,534 bytes. With `kernel_info` at ~62 KB, the total message was 65,631 bytes.

The old fallback from `j.dump(4)` → `j.dump(-1)` reduced whitespace but not
`kernel_info` content, so the message was still 65,631 bytes. `tracy_message()`
truncated it to 65,534, cutting off the closing backtick. The CSV reader then saw
an unclosed quoted field and absorbed the next row's content into `MessageName`,
including a literal `\n` that `json.loads` rejected.

### Fixes

**`ttnn/api/tools/profiler/op_profiler.hpp` — deduplicate `kernel_info` by source file**

```cpp
std::unordered_set<std::string_view> seenComputeSources;
std::unordered_set<std::string_view> seenDMSources;

for (const auto& kernel : detail::collect_kernel_meta(program, device)) {
    auto& seenSources = (processor_class == HalProcessorClassType::COMPUTE)
        ? seenComputeSources : seenDMSources;

    if (seenSources.insert(kernel.source).second) {
        // Emit one entry per unique source file, not one per core.
        ...
    }
    // kernel_sizes accumulation still processes every kernel for correct max.
}
```

This reduces 381 entries to the number of unique source files (typically 1–3),
shrinking `kernel_info` from ~62 KB to ~200 bytes.

**`ttnn/api/tools/profiler/op_profiler.hpp` — add `kernel_info` drop as last resort**

```cpp
constexpr size_t tracy_limit = std::numeric_limits<uint16_t>::max() - 1;
auto msg = fmt::format("{}{} ->\n{}`", short_str, operation_id, j.dump(4));
if (msg.size() > tracy_limit) {
    msg = fmt::format("{}{} ->\n{}`", short_str, operation_id, j.dump(-1));
}
if (msg.size() > tracy_limit) {
    j.erase("kernel_info");  // process_ops_logs.py guards this with "if in"
    msg = fmt::format("{}{} ->\n{}`", short_str, operation_id, j.dump(-1));
    log_warning(tt::LogMetal, "Tracy op profiler message for op '{}' (call {}, device {}) "
        "exceeded the {} byte limit even after compacting JSON. "
        "kernel_info was dropped from the perf report for this op.", ...);
}
```

The deduplication fix makes this fallback unnecessary in practice, but it guards
against any future case where `kernel_info` is still too large.

Also fixes an off-by-one: the old check was `>= uint16_t::max` (65535), which
could pass for a 65,535-byte message that would still be truncated. The new check
uses `> 65534` consistently.

---

## Effect on report artifacts

| Artifact | Before fixes | After fixes |
|---|---|---|
| `new_zone_src_locations.log` | Contains ~433 corrupted lines from concurrent writes | Clean — all lines well-formed |
| `tracy_profile_log_host.tracy` | May be truncated if crash aborts Tracy flush | Fully written |
| `tracy_ops_data.csv` | Attention block op row absorbs next row's content; `MessageName` contains embedded `\n` | Each row self-contained; all 8 device rows parse cleanly |
| `ops_perf_results_*.csv` | Crash before generation, or missing `COMPUTE KERNEL SOURCE` / `DATA MOVEMENT KERNEL SOURCE` / `PROGRAM HASH` / kernel size columns | All columns populated |
| `profile_log_device.csv` | Unaffected | Unaffected |

---

## Effect on `ops_perf_results_*.csv` columns

The fixes do not change any timing measurements. All timing data was always
recorded correctly; the bugs only affected whether metadata columns could be
populated.

**Columns that go from empty → populated after the fixes:**

| Column | Why it was empty | Why it is now populated |
|---|---|---|
| `COMPUTE KERNEL SOURCE` | `kernel_info` missing from parsed op data (truncated message or crash) | `kernel_info` now fits within Tracy's limit; deduplication reduces it from ~62 KB to ~200 bytes |
| `DATA MOVEMENT KERNEL SOURCE` | Same | Same |
| `PROGRAM HASH` | Same | Same |
| `COMPUTE KERNEL SIZE` | Same | Same |
| `DATA MOVEMENT KERNEL SIZE` | Same | Same |

**Columns that are identical before and after:**

- All timing columns: `HOST DURATION`, `DEVICE KERNEL DURATION`, `OP TO OP LATENCY`, etc.
- `OP CODE`, `CORE COUNT`, `INPUT`/`OUTPUT` tensor shapes, `MATH FIDELITY`
- All 8 per-device rows for the attention block op are present and complete

**Columns that remain empty (unrelated to these fixes):**

| Column | Reason |
|---|---|
| `DEVICE KERNEL DURATION PER CORE MIN/MAX/AVG` | Requires `TT_METAL_DEVICE_PROFILER=1` |
| `PARALLELIZATION STRATEGY` | Not set by this op |
| `METAL TRACE ID` | Metal tracing not used in this test |
| `ERISC KERNEL DURATION` | No Ethernet kernels in this op |

**`.tracy` file**: The Tracy binary file is not meaningfully affected. All the
same ops and timings are recorded. The only visible difference in the Tracy UI is
that the message text for the attention block op is shorter — `kernel_info` lists
one entry per unique source file instead of 381 per-core entries.
