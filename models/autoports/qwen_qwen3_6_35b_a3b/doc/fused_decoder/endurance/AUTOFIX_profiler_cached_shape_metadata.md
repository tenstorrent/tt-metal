# Cached profiler tensor metadata in the frozen profile

The decode shape discrepancy is inherited from Tracy's cached operation metadata. It is already present in the original `profile_moe_padding_cached/tracy/layer_0/ops.csv`; `summarize_fused_topology.py` selects the first measured replay correctly and copies those columns directly. It is not evidence that decode executed 256 tokens.

The first measured replay is session 5. Step 2 is `BinaryNgDeviceOperation`, global call 8055808, program hash 9075831506467028033, with measured duration 3161 ns. That program hash first appears at global call 7168, cache miss, with prefill shape `[1,1,256,2048]`. All 190 raw records for this hash retain that shape, including later scalar/broadcast products and all measured decode sessions. Adjacent decode typecast and reshape records have logical singleton inputs. Production `FusedDecoder.norm` explicitly squares the singleton `wide` tensor before reshaping to `[4,16,1,32]`.

The metadata path explains this exactly:

- `ttnn/api/tools/profiler/op_profiler_serialize.hpp:339` enables metadata caching unless `TT_METAL_PROFILER_NO_CACHE_OP_INFO` is set. Full tensor metadata is emitted only for a previously unseen device/program hash; the cached path at line 388 emits operation identity, hash, cache-hit flag and call ID only.
- `ttnn/cpp/tools/profiler/op_profiler_json.cpp:159` stores program hash as `op_hash`; lines 175–184 cache the abbreviated prefix by device/program hash.
- `tools/tracy/process_ops_logs.py:465` saves the first full record by hash. Lines 490–496 copy its tensor metadata on subsequent abbreviated messages, replacing call/cache/trace identifiers but not shapes.
- `binary_ng_device_operation.cpp:291` intentionally omits interleaved logical shape from the program hash. Operand dtypes, memory configuration, tile/alignment and sharded geometry remain hash inputs.
- `binary_ng_program_factory.cpp:414` documents the shape-independent cache contract. `override_runtime_arguments` at line 1560 rebuilds current tensor shape/work-split arguments, including inactive cores, from the same builder used on cache misses. Shape-independent compilation does not imply shape-independent runtime arguments.

The raw CSV and current source therefore support a metadata-cache diagnosis, not a new execution failure. Fresh repeated correctness and watcher evidence provide separate execution checks. This audit did not inspect captured low-level runtime arguments or execute hardware, so it does not independently prove every runtime address or work split.

## Reporting limits and proposed stage-local correction

Keep the raw CSV unchanged. Device durations, call IDs, trace sessions and measured operation counts are per-call evidence; cached shape labels do not alter the 195-operation decode sequence or its timing sum. Do not use the cached dimensions to infer runtime bytes, FLOPs, useful core occupancy, or a roofline utilization claim. Shape-derived tt-perf-report columns need the same caveat. Even a true program cache hit is not a reliable indication that the listed shape is current.

The stage-local topology summarizer should preserve raw operand strings and add `global_call_count`, `program_hash`, `trace_replay_session`, and `program_cache_hit` columns. Its introductory text should label shapes as profiler-reported metadata that may reflect the first call of a shape-independent program. It should not silently replace cached shapes with inferred singleton dimensions. A separate source-derived shape annotation can be added only when clearly labeled as such.

For authoritative shape collection through the Tracy CLI, pass `--no-op-info-cache` and verify `TT_METAL_PROFILER_NO_CACHE_OP_INFO=1` in the actual profiler subprocess provenance. Exporting that variable alone is insufficient because the CLI resets it, as documented below. This existing native switch disables metadata compression, not the device program cache. Fresh per-call metadata should then be checked against the source and replay completeness. This is a proposed collection correction, not a claim that such a rerun already occurred. No shared-native or production changes are required.

## Preservation and collection follow-up

The stage owner preserved the complete cached-metadata collection, including raw evidence, under `doc/fused_decoder/profile_moe_padding_cached`. `collect_fused_perf.sh` now exports the supported no-cache-info switch, and provenance records it. Accurate-metadata replacement remains pending until the active exact optimization candidates are benchmarked. No prior timing data was edited. The topology exporter now carries call ID, program hash, replay session and program-cache-hit fields and documents the shape limitation; it does not infer or overwrite actual tensor dimensions.

## V2 recollection exposed a CLI override

The first complete v2 profile was collected after the shell exported `TT_METAL_PROFILER_NO_CACHE_OP_INFO=1`, but its subprocess provenance records `None` and decode norm metadata still reports the cached 256-row prefill shape. The stage owner preserved that collection, including all valid timing data, under `profile_v2_cached`. It is not an accurate-shape replacement.

`tools/tracy/__main__.py:252–257` explains the override: when the default `options.opInfoCache` is true, Tracy deletes the inherited environment variable. Only the `--no-op-info-cache` CLI option selects the branch that sets it to `"1"`. The supported remedy therefore belongs on the Tracy invocation, and the collector now asserts the flag in the resulting provenance. No shared-native change was made.

A new serialized collection using that option is running. Neither this memo nor the README claims it has completed. Acceptance of replacement metadata requires the recorded flag, singleton decode operand shapes, complete replay sequences and finished tables/CSV; prior cached collection timings remain preserved rather than edited. The completed v2 correctness/aggregate-checker PASS is independent of this metadata repair and does not by itself validate shape-derived profile estimates.

## Accurate-metadata replacement completed

The corrected `--no-op-info-cache` collection is complete for selected v2 SHA256 `0b96b6204c5c1b276b14c5f29d8dd4789569608e092cec90b7a3898ee1cfa26c`. `profile_metadata_audit.json` records flag `"1"` for both layer kinds. The first linear decode square now reports physical H32/logical H1, replacing the stale H256 label. `profiler_completeness.json` verifies 184 linear and 78 full-attention operations per replay, 16 measured identical sequences and no dropped markers. Current `perf_summary.json`, `op_topology.md`, per-phase CSVs and human tables belong to this accurate-metadata run; the aggregate checker passes again.

Summed decode kernels are 1.2655336875 ms linear and 0.560454625 ms full attention. These remain separate from uninstrumented warmed trace latency, and profiler drain gaps must not be reported as normal dispatch cost. Shape correction does not convert tensor footprint estimates into measured DRAM traffic. Historical cached profiles remain preserved under `profile_moe_padding_cached` and `profile_v2_cached`. This closes the metadata anomaly; optimization stopping, remaining candidates, review and commits are separate open requirements.
