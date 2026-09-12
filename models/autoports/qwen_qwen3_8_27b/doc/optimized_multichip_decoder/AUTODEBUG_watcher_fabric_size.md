# AutoDebug: full Ethernet watcher fabric program size

## Verdict and recommended experiment

The recorded failure is a real software memory-partition limit during fabric
configuration, before the model runs. It is not evidence that Blackhole lacks
physical memory for full Ethernet watcher instrumentation.

The smallest supported adaptation is an existing environment option:
`TT_METAL_FABRIC_OPT_LEVEL=Os`, keeping `TT_METAL_WATCHER=10` and every watcher
feature enabled. Run it with a fresh `TT_METAL_CACHE` directory so an old O3
binary cannot invalidate the experiment. `Oz` and
`TT_METAL_WATCHER_NOINLINE=1` are further existing size-reduction options that
preserve checks. Their effectiveness on this exact target remains a device
experiment for the parent; no build, import, linked probe, or device command
was run for this investigation.

No source change is proposed for integration at this point. Increasing the
Blackhole Ethernet config partition is a source-supported fallback, with
channel-buffer and build-consistency consequences described below.

## Evidence and watcher contract

The preserved stage-4 log is
`../multichip_decoder/watcher_fabric_size_failure.log`. Its failure is:

```text
Program size (29072) too large for kernel config buffer (26624) on ACTIVE_ETH
```

The stack enters `ProgramImpl::finalize_program_offsets` from
`Device::configure_fabric`, then mesh initialization. The log shows watcher
initialized with disabled features `None`. Stage-4 `commands.log:135-136`
records the linear-tail runner with layer 0, length 2049, and 30 repeats.
The stage-4 work log records its ETH-disabled fallback separately.

Current skill instructions are consistent with seeking a complete run:

- `.agents/skills/optimize/SKILL.md:175` requires watcher 10 without skipping
  assertions, in a separate run from profiling.
- `.agents/skills/multichip/SKILL.md:50` requires a watcher-clean run.
- Optimize line 480 permits an ETH-disabled retry for this precise size
  failure, with an explicit scoped limitation. That explains the previous
  artifact; it is not proof of complete Ethernet coverage.
- `.agents/skills/tt-device-usage/SKILL.md` requires hardware serialization
  and closed devices between runs. The parent owns that execution.

## Why size optimization is a supported adaptation

The complete source chain is present:

1. `tt_metal/llrt/rtoptions.cpp:1712-1727` parses
   `TT_METAL_FABRIC_OPT_LEVEL`, including `Os` and `Oz`, and logs
   `Fabric kernel optimization level override: -Os` when selected.
2. `tt_metal/fabric/erisc_datamover_builder.cpp:1781-1790` gives this override
   priority. Automatic selection otherwise uses Os with VC1 and O3 without
   VC1. The original log does not establish its actual compiler flag;
   automatic O3 is a source-based expectation for the usual VC0-only ring.
3. `tt_metal/fabric/compute_mesh_router_builder.cpp:980-994` passes that
   level into the `EthernetConfig` for each fabric router RISC.
4. `tt_metal/impl/kernels/kernel.cpp:298-302` returns the selected level to
   both compiler and linker. JIT recipe creation preserves it
   (`jit_build/build.cpp:985-986`), compiler invocation uses it at 670-680,
   and linker invocation uses it at 751.
5. Watcher defines are controlled separately at `jit_build/build.cpp:256-268`.
   Changing the optimization level does not disable assertions, sanitizers,
   stack monitoring, or Ethernet instrumentation.

Existing tests in
`tests/tt_metal/tt_fabric/fabric_router/test_fabric_opt_level.cpp` explicitly
cover `ParseOs`, `ParseOz`, override state, and restoration. These were read,
not executed.

The program needs to shed at least 2,448 bytes, approximately 8.42% of the
recorded aggregate. Source inspection establishes that the option is wired
through; it cannot establish the emitted byte reduction or resulting runtime
behavior without compilation and execution.

`TT_METAL_WATCHER_NOINLINE=1` is also directly documented for oversized
watcher binaries (`docs/source/tt-metalium/tools/watcher.rst:55-57`) and
parsed at `rtoptions.cpp:1240-1244`. JIT adds `WATCHER_NOINLINE`; with watcher
enabled, `hw/inc/internal/risc_attribs.h:43-49` changes `FORCE_INLINE` from
`inline __attribute__((always_inline))` to empty. This removes forced
inlining rather than adding a literal `noinline` attribute, and allows
the optimizer to retain function calls. Ethernet/dataflow helpers use that
macro extensively. Checks remain present. Extra calls can affect stack
usage and timing, so retain the full watcher checks and validate completion.

## Avoid an invalid cached-binary experiment

Current source has an additional cache-key limitation relevant to this A/B:

- `KernelCompileHash` in `impl/program/kernel_compile_utils.cpp:23-33`
  combines the device build key, HLK descriptor, and `Kernel::compute_hash`.
- `Kernel::compute_hash` and `EthernetKernel::config_hash` include defines,
  arguments, processor, NoC, and Ethernet mode, but not `config_.opt_level`
  (`impl/kernels/kernel.cpp:516-523,551-676`). The HLK descriptor has no
  optimization-level member either.
- The environment key hashes general flags/defines
  (`jit_build/build.cpp:375-402`); `get_compile_hash_string`
  (`llrt/rtoptions.hpp:648-663`) omits the fabric optimization override.
- `JitBuildState` hashes its invariant default optimization levels
  (`jit_build/build.cpp:525-546`), not the per-kernel configured level.
- Normal local reuse tests dependency freshness, and warmed-ELF reuse tests
  the same build state (`jit_build/build.cpp:576-590,700-702,851-875`).

Consequently an already-built watcher O3 fabric kernel can be reused for a
nominal Os run. The reverse can also contaminate a later normal O3 timing
run if a force-build overwrites the same cache path. This report does not
expand scope into repairing the general cache key.

Use a separate fresh `TT_METAL_CACHE` for each optimization-level experiment.
That option is supported at `rtoptions.cpp:439-446` and is used by JIT at
`jit_build/build.cpp:131-132`. Enable
`TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1` to capture actual fabric `-Os`
compile commands, rather than treating the runtime override message alone
as proof. `TT_METAL_FORCE_JIT_COMPILE=1` also bypasses local/warmed reuse, but
an isolated directory better preserves ordinary benchmark artifacts.

## Can the Ethernet config partition grow?

Yes, the source has address space to move its boundary. It is not safe to
increase only the final size check or HAL's reported capacity.

`ProgramImpl::finalize_program_offsets` lays out RTAs, semaphores, CB/DFB
metadata, and kernel binaries, then checks the complete size against HAL's
config buffer (`impl/program/program.cpp:2908-3001`). For non-Tensix cores,
`get_ringbuffer_size` takes HAL's fixed size directly (108-116); changing the
mesh's worker-L1 reservation does not enlarge this Ethernet partition.
`program/dispatch.cpp:480-541` packs and aligns all binaries in each kernel
group, taking the maximum group size. The 29,072-byte number is therefore
an aggregate per-core-group config requirement, not a single ELF's text size
or a sum over all four chips.

The Blackhole source memory map gives:

| Quantity | Bytes |
| --- | ---: |
| Total Ethernet L1 | 524,288 |
| Top syseng reservation | 65,536 |
| Active firmware/mailbox map end, config start | 62,256 |
| Application upper bound, before routing reservations | 455,264 |
| Current config size | 26,624 |
| Current aligned unreserved start | 88,896 |
| Unreserved region before fabric metadata/channels | 366,368 |

These are arithmetic from `hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`,
not live device readings: lines 37, 51-52, 216-265, and 376-393. The map end
is 256 + 12,768 + 64 + 16 + 2 * 24,576. The upper bound is
524,288 - 65,536 - 64 - 48 - 288 - 3,088. HAL aligns the unreserved start
to max(DRAM,L1) = 64 bytes
(`llrt/hal/tt-1xx/blackhole/bh_hal_active_eth.cpp:83-102` and
`hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:374-394`).

The central `MEM_ERISC_KERNEL_CONFIG_SIZE` macro is 26 KiB
(`dev_mem_map.h:232-233`), with a comment explaining its sizing for an earlier
25,680-byte torus program. Increasing it to 29 KiB would cover the recorded
29,072-byte program with only 624 bytes of headroom; 32 KiB would provide
3,696 bytes. The latter shifts the aligned active unreserved start to 95,040
and leaves 360,224 bytes before fabric metadata, a 6,144-byte reduction.

Both config size and the following unreserved start derive from this macro
in the active HAL. The compatibility address helper also derives from it
(`blackhole/eth_l1_address_map.h:35-36`), and idle Ethernet shares it. The
fabric builder starts its metadata at HAL's unreserved base, then allocates
channels below the unchanged application upper bound
(`fabric/erisc_datamover_builder.cpp:233-242,370-376`). Its channel allocator
selects a slot configuration that fits available space and retains fatal
checks on both total bytes and final addresses
(`fabric/builder/fabric_static_sized_channels_allocator.cpp:130-135,266-275,505-590`).

Thus a coherent central partition change has a source-supported way to move
subsequent allocations without overlap. It can reduce channel depth, affect
idle-ETH layouts, and change fabric performance or startup behavior. It
requires a rebuilt/installed host runtime and consistent device headers/JIT
artifacts on every participant, plus exact Ring2 validation. A host-only
max-size override would admit writes into the current metadata/channel area
and is unsafe. A watcher-only preprocessor change to the device macro would
also risk disagreement with the host HAL. Do not disable the size assertion.

The independent per-ELF ERISC text bound remains 24 KiB
(`dev_mem_map.h:50`, `hw/toolchain/main.ld:82-101,270-280`); the ELF loader
enforces it in `llrt/tt_elffile.cpp:390-402`. Growing aggregate config storage
does not remove that bound or enlarge private stack/data RAM. The observed
run got as far as config finalization, so this independent ELF limit was not
its reported failure.

## Parent-owned discriminating checks

Run the current selected model policy with watcher 10, profiling absent,
and no inherited `TT_METAL_WATCHER_DISABLE_*` variables. An example matching
the historical layer/length is:

```bash
TT_METAL_CACHE=/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/watcher_os_stage5 \
TT_METAL_WATCHER=10 TT_METAL_FABRIC_OPT_LEVEL=Os \
TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
models/autoports/qwen_qwen3_8_27b/tests/run_optimized_multichip_experiment.sh \
  watcher_full_eth_os --layer 0 --length 2049 --repeats 30
```

The parent should supply the accepted policy arguments as needed. The wrapper
and command were inspected, not executed here.

1. Confirm actual fabric compile commands contain `-Os`, watcher reports
   disabled features `None`, mesh initialization finishes, and the model
   passes strict numerical/replay checks with clean watcher output.
2. If it still exceeds config space with confirmed newly emitted Os binaries,
   repeat with a separate fresh Oz cache. Alternatively try watcher noinline
   as a separate controlled option. Do not stack unproven changes together.
3. Once an option passes the opening failure, rerun the selected linear,
   full-attention, and stack watcher controls under the same environment.
   Watcher timing is diagnostic; ordinary performance measurements retain
   their own environment/cache.
4. Only if the existing compiler options fail should a coherent partition
   growth patch be tested. Keep both the config-size and channel-allocation
   assertions, build/install all affected runtime artifacts, and verify both
   active and idle Ethernet layout consumers.

Status: source diagnosis complete; supported options identified; full-ETH
success awaits the parent's serialized run. No physical-limit conclusion is
justified by the old 26 KiB failure alone.
