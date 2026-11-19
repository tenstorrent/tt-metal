# Rejected second S64 profiler capture

The raw CSV is preserved in place; `rejected_capture.json` records its exact SHA256 and failed completeness checks. No duplicate large CSV or performance table was generated. This is not a clean latency measurement.

The parent reran with `TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1`, but the log again records the32K source-location instrumentation failure. The collected CSV lacks complete measured phase boundaries and the required20 total/16 measured trace sessions.

That flag guards only `DeviceProfiler::pushTracyDeviceResults` (`tt_metal/impl/profiler/profiler.cpp:2663-2671`). An independent `RealtimeProfilerTracyHandler` is constructed in `tt_metal/distributed/realtime_profiler_manager.cpp:909`. Its `make_marker` (`tt_metal/impl/dispatch/realtime_profiler_tracy_handler.cpp:39-53`) creates a distinct `Program_<runtime_id>` marker name; `HandleRecord` at212-254 pushes that marker directly without consulting the legacy disable flag. The supported eligibility checks do not expose an equivalent ordinary profiling-only disable switch. The source is sufficient to explain why the legacy flag cannot bound all dynamic locations; an internal location-table census was not performed.

The existing `-p` invocation already skips global Python function tracing; `-l` was absent. No shared/native change, unsafe alternate device mode, or further hardware retry was made by this agent. The parent directed further prefill capture to wait for a materially lower operation count. The new single-dispatch corrected matmul probe is an independent model-local experiment; its arithmetic/performance still require device validation.

See `../v6_profile_corrected_fma_s64/csvonly_followup.md` for source details and the distinction between correctness success and complete profiler evidence.
