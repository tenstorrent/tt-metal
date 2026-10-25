# Serving runtime audit

Host-only source audit; no devices opened or serving processes changed. This is
not the final process-cleanup certificate or a replacement for full-server gates.

## Sampling and runtime fallback

`tt/generator_vllm.py` enables `ttnn.CONFIG.throw_exception_on_fallback=True` at
initialization. Its production device path delegates model execution and split
sampling to `Generator.decode_forward`/`sample_prefill`. It reads only one replica
of sampled IDs. Resident steps do not refresh host tokens/current positions;
changed page tables are explicitly reloaded without replacing resident tokens.
There is no adapter argmax, generic greedy fallback, or device-path full-logit
readback. Final prefill uses the canonical sampler with subset RNG preservation;
intermediate chunks update state without sampling/readback.

The explicit compatibility mode `QWEN38_HOST_SAMPLING=1` permits full-logit host
sampling for shared tests (unsupported penalties/logprob modes can require it).
It must be absent for headline performance and all-device isolation claims.
Without that opt-in, an unsupported host route fails explicitly. Batch-wide
host/device transitions can change seeded streams in mixed compatibility tests;
those results must not be described as an all-device request-isolation proof.
`host_logits_reads` and the shutdown audit distinguish these paths. Presence of
the optional code does not establish that the measured performance path used it.

Synchronous output finalization now reads raw device IDs through the same
single-replica helper; already-read async host outputs incur no additional read.
Three host cases pass in `output_finalization_host_tests.log`. The parent reports
full 64-layer B8 synchronous page growth passed in
`page_growth_sync_control.json` (three concurrent requests, 4160 generated tokens,
4096/8192 boundaries). The field `host_logits_reads=0` means no full-logit read;
normal sampled-token reads remain required and must not be called zero readbacks.

## Per-token work

- Fixed: padding a shared cache-group table separately for 64 layers. Padding is
  now once per distinct used group; output aliasing matches the existing
  already-shaped-table contract. Allocation count 64→4; focused 42 tests and lint
  pass. `page_table_padding_microbenchmark.json` records synthetic host helper
  timing, not serving/device performance.
- Performance mode must omit `QWEN38_SERVING_EVENTS`. The adapter guards decode
  event construction, scheduler-table serialization and JSONL writes behind it.
  Diagnostic runs intentionally pay this instrumentation cost.
- Steady decode does not reapply sampling parameters/reset RNG, expand/copy the
  scheduler table, remap state or refresh token/position inputs. Those operations
  occur on explicit transitions or page changes. Recurrent snapshots are saved
  only at 4096-token boundaries. Small host slot counters and active-row checks
  are necessary to know when to snapshot without reading device positions.
- The plugin compares current group tables with cloned last-submitted tables
  each step. This ensures scheduler page growth is not mistaken for stable input;
  clones are taken only after an accepted reload. Removing this check without a
  scheduler mutation/version contract would weaken correctness.
- Generic plugin marshalling still converts nine sampling fields to lists each
  decode, even when reload_sampling_params is false. A host-only CPU benchmark
  measured 3.349 µs (B1), 4.246 µs (B8), 5.658 µs (B32), seven repeats of 10000 calls after 100
  warmups (`sampling_params_host_microbenchmark.json`). This causes no device
  synchronization/readback. No material serving impact is established, so the
  audit does not add a stateful cache to the upstream interface. A future cache
  must handle prefill, layout/mode/parameter changes, deferred grammar sampling,
  speculative submissions and legacy adapters; blindly reusing prior objects
  would introduce ownership risks.

## Cleanup

The plugin runner drains pending async reads, calls adapter capture release while
its mesh is open, then drops model/cache references. The worker closes the mesh
and clears device/config references. Failed async buffers remain retained until
process exit intentionally, avoiding a DMA use-after-free. Cleanup exceptions are
logged and mesh close is still attempted. Adapter capture release synchronizes,
releases each distinct trace and clears capture handles; it emits sampling/read
counters for audit. Forced process termination cannot guarantee orderly hooks.

The installed shared runner's `_shutdown` sends TERM to its direct Popen child,
waits 15 seconds and sends KILL if necessary. It neither kills/reaps a process group
nor proves all EngineCore descendants exited if the parent already died. Do not
modify installed skill runtime for this audit. After stopping each final run,
record surviving launcher/vLLM/EngineCore PIDs and device holders, then perform
scope-specific cleanup of descendants if needed. During this audit the parent
owns a live server; this document intentionally makes no empty-device claim.

Host lifecycle/reload/accelerator-cleanup suite: 51 passed, evidence
`readiness_vllm/runtime_audit_host_tests.log`:

```sh
docker exec -w /workspace/bringup/dependencies/vllm-tt-plugin -e USER=mvasiljev -e PYTHONPATH=tests/host-stubs qwen38-functional-build /opt/venv/bin/python -m pytest --confcutdir=. --rootdir=. tests/test_worker_lifecycle.py tests/test_accelerator_cleanup.py tests/test_decode_reload_contract.py -q
```

All bare evidence filenames are under `readiness_vllm/`. Source audit covers the
current working tree; final commits and startup source paths must still be
recorded by the stage owner. No device profiling or code changes were made in
this final audit.
