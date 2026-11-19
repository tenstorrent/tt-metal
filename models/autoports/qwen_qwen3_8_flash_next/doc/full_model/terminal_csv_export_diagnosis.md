# Terminal profiler CSV export repair

2026-11-14. CPU-only source investigation, exporter build and retained-capture re-export. No device access or model/runtime changes by this author. The original capture and malformed CSV remain unchanged.

`terminal_ops_v1.log` has two separate failures. The experiment first fails during force-argmax setup because its `Collectives` instance lacks `get_and_cycle_barrier_semaphore_handle`. Subsequently, report export fails while parsing `tracy_ops_times.csv`. Repairing export does not repair the sampler setup or turn this capture into performance acceptance.

## Root cause

`tt_metal/impl/dispatch/realtime_profiler_tracy_handler.cpp::HandleRecord` joins `record.kernel_sources` with literal comma/newline and stores the result in the Tracy marker's file field. Generic-op source entries may contain complete inline kernel text. The exporter includes GPU/device zones in `-u`, copies this field into `src_file`, then previously joined all fields without CSV escaping.

The original first malformed row begins at physical line305815: `Program_11980` has three source paths spread over three lines. PyArrow reports expected9 columns/got3. At line319011, an inline kernel's16-element C++ constant array produces the C parser's expected9/got16 error. These are literal data inside one source field, not missing event records or profiler buffer truncation. Increasing buffers, using another delimiter, or silently skipping malformed rows is not a valid repair.

`--disable-device-data-push-to-tracy` guards `DeviceProfiler::pushTracyDeviceResults`; it does not guard the independently registered realtime profiler handler. It is not a reliable workaround for these program zones. `-f TT_` would omit other zones and child timing; it was not used to conceal this failure. No collection settings were changed to reduce evidence.

## Minimal repair

The Tracy submodule now has `csvexport/src/CsvWriter.hpp`, used for zone/plot CSV rows and headers. It quotes fields containing the delimiter, a quote, CR or LF and doubles embedded quotes. The former manual quoting of CPU `zone_text` is removed so decoded text is not double-quoted as data. Message export retains its existing backtick-quoted producer protocol unchanged.

The same loop also had a concrete adjacent bounds defect: the nine-column unwrapped schema wrote a special parent's text to `values[9]`; the actual final column is `values[8]`. This index is corrected. The retained capture contains no special-child function rows, so it does not provide dynamic coverage of that correction.

CPU regression `tests/ttnn/tracy/test_csvexport_quoting.py` compiles the actual writer header and round-trips seven nine-field rows through comma, semicolon and tab separators using strict Python CSV and pandas C readers. Cases include multiple source paths, inline code, quotes, CRLF, trailing newline, empty fields and TT host `zone_text`. Run directly to avoid TTNN pytest fixture imports:

```bash
python_env/bin/python tests/ttnn/tracy/test_csvexport_quoting.py
cmake --build build_Release --target tracy-csvexport --parallel 2
```

Both pass. CMake regenerated after existing glob changes; the build then compiled only the exporter object and linked only the exporter executable. Source hashes, binary hash and exact export argv are recorded in [validation JSON](terminal_csv_reexport_validation.json). Exporter changes are in the Tracy submodule and need its local commit plus the parent gitlink when the parent records stage-owned changes; no commit/push was performed here.

## Lossless retained-capture validation

New files under `terminal_tracy/.logs/` are `tracy_ops_times.quoted.csv` and `tracy_ops_data.quoted.csv`. The original files remain present. Commands use the same capture and export selection as the installed report path:

```bash
build/tools/profiler/bin/tracy-csvexport -u -t TT_ \
  -x CompileProgram,HWCommandQueue_write_buffer \
  models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_tracy/.logs/tracy_profile_log_host.tracy \
  > models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_tracy/.logs/tracy_ops_times.quoted.csv
build/tools/profiler/bin/tracy-csvexport -m -s ';' \
  models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_tracy/.logs/tracy_profile_log_host.tracy \
  > models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_tracy/.logs/tracy_ops_data.quoted.csv
```

| Validation | Result |
|---|---:|
| Strict CSV records, each exactly9 fields | 436,897 |
| CPU zone rows | 305,813 |
| Program rows | 131,064 |
| Rows retaining multiline source | 130,648 |
| Rows retaining source quotes | 768 |
| TT_DNN_DEVICE_OP rows | 131,108 |
| Message rows | 131,145 |
| Target or other signposts | **0** |

Re-rendering every decoded field using the old malformed serialization reproduces the original timing CSV byte-for-byte: SHA256 `7bbdcdcb13c8ef09ab7538f20cba72170092064b92eec53fba943b95228bac23`. This verifies no row or field was skipped, joined incorrectly, or silently changed. Message output is itself byte-identical to the original.

The installed importer `import_tracy_op_logs` was exercised with temporary links to the new files: all131,108 operations correlate with host timing, zero skipped-op warnings, zero signposts, zero trace-device entries. PyArrow still rejects multiline fields under its default `newlines_in_values=False`; the existing caught `ParserError` fallback to pandas C parses all436,897 rows. No parser relaxation or `on_bad_lines` setting is used. A pandas mixed-type warning for nullable text columns is informational and does not drop records.

## Next collection

The rebuilt exporter is available through the normal `build/tools/profiler/bin/tracy-csvexport` path; no new model configuration is needed. After the independently owned sampler setup repair passes, use the same reduced-layer profiler command from `tests/profile_full_model_terminal.py` with a fresh output folder. `--no-web-server` may suppress the unrelated web-UI launch/copy warning; it is not needed for CSV validity. Retain device profiling and the intended signposts. Watcher and profiler must remain separate.

Before running `tt-perf-report`, verify the successful experiment reached the expected `PERF_MODEL`, `PERF_SPLIT_GREEDY`, and `PERF_FORCE_ARGMAX` start/end markers, and require valid device durations and complete op correlation in the generated `ops_perf_results` CSV. Use the pinned optimize recipe with advice enabled. The retained failed run has no such markers and cannot supply those measurements, even though its export is now lossless.
