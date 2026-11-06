# AutoDebug: omitted writable JIT cache environment

Diagnosis: this attempt failed in host filesystem setup while opening the mesh, before the model diagnostic ran. `bringup/artifacts/astropy-capped-prefix/tt-b8.log:52-59` locates the exception at `diagnose.py:50`, `ttnn.open_mesh_device`, with `fabric_erisc_router` binary generation unable to create its directory under `/home/bringup/.cache/tt-metal-cache`. This is a launch-environment error, not evidence of a model, precision or device execution failure.

Source chain in this checkout:

- `tt_metal/jit_build/build.cpp:192-203`: default root is `$HOME/.cache/tt-metal-cache/` when HOME exists, otherwise `/tmp/tt-metal-cache/`; explicit RunTimeOptions cache takes precedence. XDG_CACHE_HOME is not consulted here.
- `build.cpp:252-260`: JitBuildEnv stores that resolved root as `out_root_`.
- `tt_metal/llrt/rtoptions.cpp:2052-2061`: runtime options read process environment; supply the cache variable before Python imports/initialization, rather than changing it after the runtime snapshot.
- `rtoptions.cpp:497-499` and `:289-295`: TT_METAL_CACHE marks the cache specified and normalizes the value **with a `tt-metal-cache` child appended**. Preserve the already validated launch value and check its effective child location; do not mistake the variable for a literal final leaf path.
- `tt_metal/jit_build/genfiles.cpp:1211` creates the build directory; `tt_metal/impl/program/program.cpp:414-419` catches the host runtime error and wraps it as failed binary generation.

Read-only filesystem inspection: `/home/bringup/.cache` is root-owned0755 and the failing child does not exist; the execution checkout's `bringup/artifacts/tt-metal-cache` is bringup-owned0775. `bringup/artifacts/cache-env-validation/tt-metal-jit` is a symlink to that approved cache. No permission mutation or write test was performed here. The coordinator already has independent writable-cache validation.

Recovery assessment: the log records user-mode driver close at15:26:04.027, cluster close completed at15:26:04.228 and destructor completion. The coordinator reports no current device owners. This is not a hang, forced termination during dispatch or ARC/ERISC transport error. These facts support owner-check followed by a bounded, serialized mesh open/close or trivial-op health check using the corrected cache environment; a reset is not justified solely by this permission exception. Clean teardown is not proof of healthy device operation, so do not skip that probe or claim recovered hardware before it passes. If health fails, preserve its exact evidence and use the device recovery policy rather than repeatedly launching the model.

AutoFix scope: coordinator restores the previously validated TT_METAL_CACHE environment to the diagnostic launch, keeps model/runtime/precision/prompt unchanged, accounts failed queue time and health/retry against the existing effort box, and preserves this failure log separately. Do not repurpose HOME, alter global permissions, reset preemptively or modify runtime source. This isolated AutoDebug turn performed only source/log/filesystem inspection, no target/device execution and no runtime edits.
