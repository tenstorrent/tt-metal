# Post-hoc clang-tidy on JIT-compiled kernel code

Device kernels are compiled at runtime by `tt_metal/jit_build/` with the SFPI
cross-compiler, so they are invisible to the host build's static analysis.
This flow runs clang-tidy on them **after the fact**: run any workload
normally with the build system's own compile-command logging enabled
(`TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`), parse the logged real JIT compiler
invocations, translate them for clang, and lint.
([`bear`](https://github.com/rizsotto/Bear) capture is also supported as an
alternative input — see the gotchas below for why it is not the default.)

No synthetic build system, no stub headers, no enumeration of kernel configs:
the runtime already produced everything needed (real compile-time args, real
defines, real generated headers) as a side effect of just running the test.
Coverage is exactly the set of kernels that run JIT-compiled — no more.

Prior art / related:

* Jira MINFRA-1199 ("Explore clang-tidy static analysis for device/kernel
  code") — this flow is a concrete answer to it.
* PR [#37252](https://github.com/tenstorrent/tt-metal/pull/37252) — earlier
  clang-tidy attempt using host clang + hand-written stubs for SFPI internals
  and generated files. This flow supersedes the stubs by capturing real
  compiles instead (see `tt_metal/jit_build/kernel_clang_tidy/README.md` for
  why no mock SFPI headers are needed anymore).
* [`kernel-code-indexing.md`](./kernel-code-indexing.md) — the adjacent IDE
  indexing flow. Note its script (`build_kernel_compile_commands_json.py`)
  rewrites TUs to the bare kernel source and dedups TRISC roles, which is fine
  for clangd but wrong for linting; this flow keeps the real wrapper TUs so
  UNPACK/MATH/PACK stay distinct and role defines stay correct.

## Running it locally

Prerequisites: any clang-tidy ≥ 17.

```bash
cd <tt-metal root>

# 1. Cache hits skip the compile, so force real compiles.
export TT_METAL_FORCE_JIT_COMPILE=1
# 2. If kernel ccache is enabled (TT_METAL_CCACHE_KERNEL_SUPPORT), a ccache hit
#    also skips the real compile. Take ccache out of the path entirely, and
#    disable it as a fallback in case something else invokes it.
unset TT_METAL_CCACHE_KERNEL_SUPPORT
export CCACHE_DISABLE=1
# 3. Have the JIT build log every kernel compile command. The lines are logged
#    at info level, so the logger must be at info too.
export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
export TT_LOGGER_LEVEL=info

# 4. Run any test/workload, teeing its output to a file.
pytest tests/ttnn/...::test_case 2>&1 | tee /tmp/kernel_run.log

# 5. Parse the logged compile commands, translate GCC->clang, run clang-tidy.
python3 scripts/build_kernel_clang_tidy_commands.py \
    --input-log /tmp/kernel_run.log \
    --output-dir /tmp/kernel_tidy \
    --run \
    --config-file "$PWD/tt_metal/jit_build/kernel_clang_tidy/.clang-tidy"

less /tmp/kernel_tidy/findings.txt
```

Alternative capture: wrap the run in `bear --output raw.json --` and pass
`--input raw.json` instead of `--input-log`. Equivalent output; see the bear
gotcha below before relying on it in a container.

Do **not** clear the tt-metal cache (`~/.cache/tt-metal-cache` or
`$TT_METAL_CACHE`) between the run and the lint: the captured commands
reference the generated headers there (`chlkc_*.cpp`, `chlkc_descriptors.h`,
`kernel_includes.hpp`, `defines_generated.h`), and clang-tidy re-parses from
those sources. Object files do not need to survive (they don't; the JIT build
uses temp names) — only sources/headers matter, and those are durable.

### Gotcha: bear + `http_proxy` (why bear failed in CI)

The first CI iteration of this flow used `bear` and failed instantly with
`wrapper: failed with: gRPC call failed: failed to connect to all addresses`,
killing the wrapped command **before pytest even started** (bear's nonzero
exit then failed the leg). The verified mechanism, for anyone using bear
locally:

* bear ≥3.x runs an intercept supervisor as a **gRPC server on
  `127.0.0.1:<random port>`**; the `wrapper`/`libexec.so` clients in every
  intercepted process (including the initial one that launches the wrapped
  command) connect back over that channel.
* gRPC's C core routes **all** channels — including loopback ones — through
  `http_proxy`/`https_proxy` unless the target host appears in `no_proxy`.
* The CI runners inject
  `http_proxy=http://proxy.restricted-proxy.svc.cluster.local:3128` into the
  job container, with a `no_proxy` list that does **not** contain
  `localhost`/`127.0.0.1` (visible verbatim in the job's `docker create`
  command). So bear's client tried to reach its own loopback supervisor via
  the cluster proxy, which cannot connect back into the container — hence
  "failed to connect to all addresses" within milliseconds.
* This is a documented upstream failure mode: Bear issues
  [#296](https://github.com/rizsotto/Bear/issues/296) ("the solution was to
  remove the HTTP proxy environment variables"),
  [#635](https://github.com/rizsotto/Bear/issues/635), and PR
  [#631](https://github.com/rizsotto/Bear/pull/631) (merged 2025), which makes
  bear strip the proxy variables from its gRPC channel **by default** — a fix
  newer than 3.0.18, the only version packaged for Ubuntu 22.04.

It is *not* a container/namespace limitation per se: the supervisor and the
intercepted compilers all run in one process tree inside one job container.
Local workaround if you want bear: `no_proxy="$no_proxy,localhost,127.0.0.1"
NO_PROXY="$NO_PROXY,localhost,127.0.0.1" bear -- <cmd>` (or unset the proxy
vars, or use a bear release containing the #631 fix).

CI still uses the log-based capture regardless: no wrapper process means
capture is structurally incapable of failing the test run, and there is no
third-party interception dependency at all.

### Gotcha: every kernel-compile cache silently defeats the capture

Both capture modes only see compiles that actually happen. A cache hit at
**any** layer serves the result without running the real compile, and the
capture silently comes up empty (the filter script warns on zero entries, but
the failure mode to understand is "cache hit", not "capture broke"). There are
three layers:

1. **The tt-metal JIT cache** (`~/.cache/tt-metal-cache`): a hit means
   `JitBuildState::need_compile` returns false and no process is spawned at
   all. Defeat: `TT_METAL_FORCE_JIT_COMPILE=1`.
2. **Kernel ccache** (`TT_METAL_CCACHE_KERNEL_SUPPORT`): when set,
   `JitBuildEnv::init` prepends `ccache` to the SFPI command; a ccache hit
   skips the compiler exec. Defeat: `unset TT_METAL_CCACHE_KERNEL_SUPPORT`
   (removes ccache from the process tree), plus `CCACHE_DISABLE=1` as a
   fallback (a disabled ccache is a pure pass-through, so the compile is still
   observable even if ccache does get invoked).
3. **The CI Redis-backed kernel ccache**: in CI, `.github/actions/setup-job`
   (`enable-kernel-ccache: true`) configures kernel ccache with
   `CCACHE_REMOTE_ONLY=true` and `CCACHE_REMOTE_STORAGE=redis://...` — i.e.
   *all* kernel-compile cache lookups go to a shared Redis instance, and a
   remote hit serves the object without ever running `riscv-tt-elf-g++`. This
   is the same ccache from layer 2, so the same defeats apply; the CI wiring
   additionally clears `CCACHE_REMOTE_STORAGE`/`CCACHE_REMOTE_ONLY` for the
   capture leg to make the intent explicit and to avoid touching the shared
   Redis cache from an instrumented run.

## What the translation does

See the docstring of `scripts/build_kernel_clang_tidy_commands.py` for the
full list. Summary: keep only SFPI `-c` compile entries; swap the compiler for
`clang++ --target=riscv32-unknown-elf` (mapped from `-mcpu=tt-wh`/`tt-bh`);
drop SFPI-GCC-only flags (`-ftt-*`, `-flto=auto`, `--param=min-pagesize=0`,
dep-file flags); upgrade `-std=c++17` to `-std=c++20` (the `-ftt-*` flags
backport C++20 features that tt-llk headers use); wire in the SFPI toolchain's
own newlib/libstdc++ headers; apply the riscv32 `int32_t`-is-`long`
type-model overrides that `sfpi.h` static_asserts. Everything else — CTAs,
defines, include paths, generated files — passes through untouched.

By default entries are deduplicated to one per (kernel source, RISC target):
the same kernel recompiled under many compile-time-arg configurations is
linted once, with the first-captured config. Pass `--dedupe none` to lint
every configuration.

## CI wiring (prototype)

`.github/workflows/ttnn-sanity-tests-impl.yaml` has an opt-in experiment:
callers pass `enable-kernel-clang-tidy: true` and **every** hardware leg
captures and lints its own kernels, the same way `collect-coverage` applies to
every leg. Each leg's pytest run gets
`TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1` + `TT_LOGGER_LEVEL=info` (the logged
compile commands land in the leg's run-with-log file), with all three
kernel-compile cache layers defeated for that leg only (see the gotcha above:
`TT_METAL_FORCE_JIT_COMPILE=1`, `TT_METAL_CCACHE_KERNEL_SUPPORT` unset,
`CCACHE_DISABLE=1`, and the Redis `CCACHE_REMOTE_STORAGE` cleared). After the
tests, a non-blocking (`continue-on-error`) step translates the captured
commands (this script, no `--run`) and analyzes them with CodeChecker —
`CodeChecker analyze compile_commands.json --config
tt_metal/jit_build/kernel_clang_tidy/codechecker.json` — the same tooling
`clang-static-analyzer.yaml` uses. See
[Checker selection and scoping](#checker-selection-and-scoping) for how that
config decides what runs and what is reported. The leg does not render HTML;
it uploads one `kernel-clang-tidy-<group>` artifact (plists,
compile_commands.json, `reports.json` counts, summary) and the consolidate job
does all rendering. The suggested-fix YAMLs `CodeChecker` writes to
`reports/fixit` are deleted before upload: measured at 304 MB of an 880 MB
artifact, and nothing downstream reads them.
Gotcha found while wiring this: CodeChecker's compilation-db parser consults
`ClangSA.analyzer_binary()` unconditionally, so a `clang` binary must be
resolvable even though only clang-tidy runs. Handled the way
`tt-umd`'s `code-analysis.yaml` does it — `update-alternatives` symlinks for
`clang`, `clang++` and `clang-tidy` so CodeChecker resolves them natively,
rather than detecting versioned binaries and passing `CC_ANALYZER_BIN`. The
`apt-get` fallback stays because these legs run in the ci-test image, not the
dev image tt-umd uses, so the toolchain is not guaranteed present.

The dedicated caller is `.github/workflows/kernel-clang-tidy.yaml` (structured
after `code-coverage.yaml`): build → run the ttnn sanity suite on hardware via
`ttnn-sanity-tests-impl.yaml` with the experiment enabled on every leg → a
`consolidate-report` job that merges every leg's plists into one report and
pushes it to `tenstorrent/tt-metal-kernel-clang-tidy-results` gh-pages (on
main, or when dispatched with `publish-html: true`). Launch with:

```sh
gh workflow run kernel-clang-tidy.yaml --ref <branch> \
  -f enabled-skus=wh_n300_civ2 -f publish-html=true
```

### Consolidating the legs into one report

`consolidate-report` merges every leg's plists and runs `CodeChecker parse
--export html` once, so the published site is one genuine CodeChecker report: a
sortable table with Severity, Checker name, File and Message columns, plus its
own checker- and severity-statistics pages. No hand-written HTML and no per-leg
navigation — leg provenance is deliberately dropped, since the same kernel code
is analyzed on many legs and the reader only cares about the finding.

#### Why the merge deduplicates first

The first consolidated run (33984801327) exhausted a 90-minute job timeout
mid-render. Profiling the cause rather than trimming checkers:

* Each TU is a firmware wrapper with the kernel `#include`d into it, so every TU
  re-reports the defects of every header it pulls in, and every leg repeats that
  again. On two legs, **370,060 raw diagnostics represent 19,863 distinct
  findings — 18.7x redundancy.** Plists run 11 MB mean, 588 MB for two legs.
* `CodeChecker parse` is single-threaded — there is no `--jobs` — and its cost
  scales with the *raw* count, so ~95% of its runtime rediscovers findings it has
  already seen. Of the 115s it took on those two legs, 33s was `plistlib` XML
  parsing and 82s was CodeChecker's own per-diagnostic processing.

`.github/scripts/utils/merge_kernel_tidy_plists.py` therefore drops duplicates
before CodeChecker sees them, keying on file, line, column, checker and message.
It shrinks 588 MB to 31 MB, after which `parse` drops from **115s to 6.7s, a 17x
speedup, with zero findings lost or gained — the unique finding set is
identical.** The saving grows with leg count, since cross-leg repetition is most
of the redundancy.

Note where that 17x comes from: feeding CodeChecker 19,863 diagnostics instead
of 370,060. It does not depend on the core count. Only the merge itself is
parallel (the XML parse is embarrassingly so), and it is the cheap half. On the
two-leg sample:

| Workers | Merge wall time |
| --- | --- |
| 64 (capped to 52, one per plist) | 3.1s |
| 16 (`tt-ubuntu-2204-large-stable`) | 5.3s |
| 4 (a GitHub-hosted runner) | 16s |

Against a budget that was overrunning 90 minutes, all three are noise; at 15
legs the 16-core case projects to well under a minute. Worker count comes from
`os.sched_getaffinity` and the cgroup v2 CPU quota rather than `cpu_count()`,
which inside a CPU-limited container reports the host's cores and would
oversubscribe while multiplying peak memory to match. Peak RSS is ~70 MB for the
largest single plist and ~135 MB for the run.

The merge makes two read passes over the inputs instead of one pass plus a
staged rewrite. Duplicates are almost entirely *cross*-plist — within a single
plist the raw and distinct counts are the same 370,060 — so a per-plist
intermediate would be a byte-for-byte copy of the input, about 4 GB of writes at
15 legs, to remove nothing. Note that `/tmp` in this container is a tmpfs, so
that scratch would have come out of RAM.

This is why no checker is disabled for volume: report size no longer drives CI
cost, and the point of the report is a complete database of problems to fix.

#### Rendering needs the sources

Rendering needs the analyzed sources at the same absolute paths the legs used,
which is why this job runs in the ci-test container with `setup-job` installing
the wheel, rather than on a bare `ubuntu-latest`. Two roots cover everything:
`/opt/venv/...` (16,829 of the 19,835 unique findings, wheel-installed headers)
and `/work/...` (3,006, the checkout). Skip the sources and `--export html`
silently produces `index.html` and `statistics.html` with **zero** finding pages
— measured, not assumed, which is what an earlier attempt to merge on
`ubuntu-latest` got wrong.

The exception is the generated JIT glue, which is analyzed (see below) but lives
in the kernel cache on the test runner, and only plists are uploaded. Those
findings therefore appear in the tables and statistics with no browsable source.

The JSON export runs before the render and uploads unconditionally: it is the
machine-readable form of the same data, and it is what to point an agent at.

### Why not the simulator legs?

Capturing on a `sim_*` leg instead of hardware looks appealing — only the
compile commands matter, not the device result — but it was tried and
abandoned; the tidy steps skip `sim_*` SKUs. Three independent blockers:

* **pytest-xdist.** Sim legs run `-n 4`, and xdist workers use their stdout as
  the execnet RPC channel, so raw C++ stdout (tt-logger's default sink, and
  with it the logged compile commands) never reaches the run log. Serializing
  the leg does fix that — the repo-wide `-s` in `pytest.ini` keeps stdout
  uncaptured in-process, which is why the HW capture works at all — but it
  forfeits the parallelism sim needs most. `TT_LOGGER_FILE` is not a way
  around it: the file sink opens with truncate, so each xdist worker and each
  pytest invocation of a multi-command test group would wipe the previous
  capture.
* **Slow dispatch.** ttsim runs slow dispatch, so tests needing trace or fast
  dispatch skip themselves ("not working for slow dispatch"). Verified in run
  33931876161: a serialized `trace allocation tracker [sim_wh_n150]` leg
  passed in ~2 minutes having skipped every device test — **0 captured
  commands**. It also means the dispatch kernels that dominate a hardware
  capture are never built under sim.
* **Cost.** Sim is 10-50x slower per op on cloud runners (sim group timeouts
  are 40-60 min against 10 for the same group on HW), and the capture leg is
  already slow by design (`TT_METAL_FORCE_JIT_COMPILE=1`, every kernel cache
  defeated).

### Checker selection and scoping

Both are driven by CodeChecker, in
`tt_metal/jit_build/kernel_clang_tidy/codechecker.json`, not by a check list in
`.clang-tidy`.

**Selection** is `--enable-all` (CodeChecker's ~477 known clang-tidy checkers)
minus a short `--disable` list. Twelve of the entries are whole families that
cannot apply to bare-metal RISC-V device code — `abseil`, `altera`, `android`,
`boost`, `darwin`, `fuchsia`, `linuxkernel`, `llvmlibc`, `mpi`, `objc`,
`openmp`, `zircon` — 72 checkers between them. The other three are specific to
this domain:

| Disabled | Reason |
| --- | --- |
| `hicpp-no-assembler` | Inline asm is pervasive in `tt_metal/hw/inc` and tt-llk; recorded at 242K hits under the previous flow. |
| `portability-simd-intrinsics` | SFPI *is* a SIMD intrinsics layer, by design. |
| `clang-diagnostic-c++98-compat` | Device code is C++17/20. |

Two more are muted on volume. `modernize-use-trailing-return-type` (1,182
findings) is pure style and tt-umd mutes it too.

`modernize-avoid-c-arrays` (21,785 findings, 35% of the whole report) is muted
for a measured reason rather than a stylistic one. It is not that the codebase
merely prefers C arrays: 98.3% of its findings are in generated JIT code, almost
all in `chlkc_descriptors.h`, which emits ~23 constexpr tables per kernel variant
across 451 kernel names. Only 364 findings are in hand-written code, and the
checker collapses to 377 distinct locations.

The reason not to act on it is compile time, which matters here because kernels
are compiled at runtime. `std::array` is available and already used in device
code (15 files in `tt_metal/hw/inc`, 14 in tt-llk, 43 of 1,475 ttnn kernel
files), so this is not a capability limit. But measured on 23 constexpr integer
tables, the cost is almost entirely the header, not the type:

| Translation unit | g++ | clang |
| --- | --- | --- |
| C arrays, no `<array>` | 14.4 ms | 30.6 ms |
| `#include <array>` alone | 56.4 ms | 84.4 ms |
| C arrays with `<array>` already included | 56.4 ms | — |
| `std::array` tables | 56.9 ms | 86.7 ms |

Pulling in `<array>` costs ~42 ms per TU on g++; converting the tables once it is
present costs ~0.5 ms. A cache-missing JIT build of twenty-odd TUs would pay
close to a second of runtime compile if this newly introduced the header, for
constexpr tables that only generated code indexes. Whether it *would* introduce
it is unresolved: every firmware wrapper transitively reaches 5 or 6 headers that
include `<array>`, but several are conditional (`device_print.h` is DPRINT-gated,
`compile_time_args.h` is prolog-emitted only for named-arg kernels,
`dataflow_buffer.h` guards its include with `__has_include`). Settling it needs
`g++ -H` on a captured compile command in the CI container.

Nothing else is muted for volume — see
[Report consolidation](#report-consolidation) for why finding count no longer
drives CI cost.

Everything else is on. This replaced an inherited 189-entry opt-out list (13
families plus 176 individual checks) carried over from the `kernel_clang_tidy`
CMake target in PR #37252, which had been assembled by muting whatever fired
under that static flow and never re-triaged. It had `clang-analyzer-*` off
wholesale along with `bugprone-narrowing-conversions`,
`bugprone-integer-division`, `bugprone-too-small-loop-variable`,
`bugprone-sizeof-expression`, `misc-const-correctness` and
`performance-no-int-to-ptr` — close to the bug classes device code most wants.
Note that `take-config-from-directory` must stay off for any of this to take
effect: with it set, CodeChecker returns an empty checker list
(`clangtidy/analyzer.py:471`) and every `--enable`/`--disable` is inert.

**Scoping** is `--skip <skiplist> --drop-reports-from-skipped-files`, which
replaced `HeaderFilterRegex`. Two reasons the skiplist is the better mechanism
here: it also drops `clang-diagnostic-*` findings, which a header filter
structurally cannot (hence the SFPI and glibc parse noise in earlier reports),
and it keeps scope in one reviewable file. The list is exclusion-only — the
same shape as tt-umd's `.codechecker.skiplist` — and now covers only upstream
SFPI and host libc, neither of which is fixable in this repo.

**SFPI headers are demoted to `-isystem`**, which is the mechanism that actually
saves analysis time; the skiplist only discards findings after the analyzer has
already produced them. The device build passes `-I /opt/tenstorrent/sfpi/include`
while every other SFPI path already arrives as `-isystem`, so clang-tidy treated
`sfpi.h` as first-party and reported on it — the same distinction CMake's
`SYSTEM` keyword draws for host dependencies like nlohmann/json.
`build_kernel_clang_tidy_commands.py` rewrites it during capture (21 of 44
entries on a sample leg, the compute TUs; project `-I` flags are untouched), so
this is analysis-only and cannot affect the device build.

This matters more than it looks. CodeChecker forces `HeaderFilterRegex=".*"`
whenever no `--analyzer-config` is given (`clangtidy/analyzer.py:665`), so every
non-system header is fair game; `-isystem` is what takes SFPI back out, since
clang-tidy's `SystemHeaders` defaults to false and is not overridden by the
header filter. Measured on a synthetic 4,000-function vendor header, moving it
from `-I` to `-isystem` took clang-tidy from **1.93s and 23,834 warnings to
0.10s and 2**. That is a pathological density and an upper bound, not a
prediction for SFPI, but the mechanism is the point: the parse still happens
because the TU needs the AST, while diagnostic matching, fix-it construction and
message rendering do not.

The skiplist keeps its SFPI entry as a backstop, since `clang-diagnostic-error`
and friends can still surface from system headers.

The machine-generated JIT glue in the kernel cache used to be excluded too, on
the grounds that findings there belong to the generator rather than the output.
That was the wrong call: `genfiles.cpp` emits the `chlkc_*.cpp` prologs, the
`kernel_main()` shim for `TT_KERNEL`-tagged entries and the
`chlkc_descriptors.h` format tables, and a generator emitting bad code is a bug
worth filing. It is analyzed now. Two consequences are accepted deliberately:
cache paths embed the kernel name and two content hashes, so one generator
defect lands once per kernel variant instead of collapsing under deduplication;
and those findings have no browsable source in the report, per the note above.
If the volume proves unmanageable, the fix is to canonicalise cache paths during
the merge and ship one representative copy of the generated tree — not to stop
looking. Note this does not affect coverage of the kernels themselves, which
resolve to wheel or checkout paths and were always analyzed.

Worth knowing why an
allow-list is not an option here: the translation units are the firmware
wrappers under `tt_metal/hw/firmware/src/`, with kernels `#include`d into them,
so no TU path contains `kernels/` and an allow-list keyed on it would analyze
nothing at all. Exclusion-only also fails open as new in-repo device
directories appear.

**Per-path exceptions** are the fourth mechanism, and the right one when a
checker is correct in general but wrong about a specific construct.
`tt_metal/jit_build/kernel_clang_tidy/review_status.yaml` assigns a review status
to findings matched by `filepath` (an fnmatch glob), `checker_name` (exact) or
`report_hash` (prefix). `parse` defaults `--review-status` to
`confirmed,unreviewed`, so anything marked `intentional` or `false_positive`
drops out of the HTML report, `findings.json` and the statistics alike: the
filter runs before the exporter picks a format
(`codechecker_analyzer/cli/parse.py:646`, CodeChecker 6.27.1). The schema key is
`$version`, not `version`, which `--help` does not mention.

It is applied by the consolidate job, which copies the file into the merged
report directory, rather than by the legs via `analyze --review-status-config`.
`parse` reads `review_status.yaml` out of whatever directory it is handed, so an
exception can be added or withdrawn by re-running one job instead of sixteen
hardware legs. The cost is that suppressed findings are still analyzed, which for
the one rule below is immaterial.

That rule is `bugprone-suspicious-include` (1,488 findings), which objects to
`#include`ing a `.cpp` — exactly how the JIT builds a kernel. `kernel_includes.hpp`
(687) and the `chlkc_{math,pack,unpack}.cpp` prologs (798) pull the kernel body in
as a source file, and `chlkc_list.h` (3) does the same in-repo. Two globs,
`*/kernel_includes.hpp` and `*/chlkc_*`, cover all 1,488 exactly while leaving the
21,550 findings other checkers report in those same files visible, which a
`--disable` would not.

The bar for a rule is that the finding is wrong about this code — not that it is
unwelcome or numerous — and that it is scopable by path. Most volume candidates
fail one of those. `readability-magic-numbers` (2,567) was proposed for an
exception on the grounds that it fires on register offsets and bit positions, and
the data did not support it: only 5% is generated, and the concentrations are SFPU
polynomial coefficients (`ckernel_sfpu_trigonometry.h`, 139), tile geometry in
dataflow helpers (`moreh_common.hpp`, 159 — `1024`, `16`, `>> 16`) and Tensix
instruction encodings (`tensix_functions.h`, 102 — `(addr_mode << 15) |
(zero_write << 12)`). Bit positions are precisely what should be a named constant,
so the checker is right. The one arguable case is the NOC coordinate table in
`eth_chan_noc_mapping.h` (65), which is tabular data whose shifts are already
named, and 65 findings do not justify a rule. `performance-no-int-to-ptr` (694),
`clang-diagnostic-old-style-cast` (766) and `readability-uppercase-literal-suffix`
(3,762) fail the scoping test instead — their top five files hold 13–29% of their
findings, so any path glob would be arbitrary. The last is also trivially
auto-fixable, which makes it a target for an agent rather than for suppression.

**Check options are the quietest suppressor here**, so they are worth reading as
carefully as the `--disable` list. Nothing readability- or complexity-related is
disabled, but five options retune three checks, and they were mirrored verbatim
from the repo's root host `.clang-tidy` ("mirror host .clang-tidy strategy for
kernel checks") rather than chosen for device code:

| Option | clang-tidy default | Ours |
| --- | --- | --- |
| `readability-function-cognitive-complexity.Threshold` | 25 | 25 (was 312) |
| `readability-function-cognitive-complexity.IgnoreMacros` | false | false (was true) |
| `readability-simplify-boolean-expr.SimplifyDeMorgan` | true | false |
| `readability-else-after-return.WarnOnUnfixable` | true | false |
| `readability-else-after-return.WarnOnConditionVariables` | true | false |

The first two are now back at clang-tidy's defaults, deliberately diverging from
the host config. At `Threshold=312` with `IgnoreMacros=true`, cognitive
complexity produced **5 findings out of 62,710** — a 12.5x-loosened threshold
that also ignored precisely the macro-driven complexity that dominates LLK and
the SFPU headers. That pairing suits a blocking host gate; it defeats a
non-blocking report whose purpose is to enumerate problems. Both files state the
values explicitly so a future "sync with host" edit does not silently undo it.

**Check options** go through `--checker-config
clang-tidy:<checker>:<option>=<value>`, not through a config file. Forwarding
`--config-file` via `cc-verbatim-args-file` was the first attempt and it fails:
once `take-config-from-directory` is off, CodeChecker builds its own `-config`
for clang-tidy (`clangtidy/analyzer.py:509`) and appends our verbatim args
too, so clang-tidy aborts every TU with "--config-file and --config are
mutually exclusive". CodeChecker does merge a `-config=<JSON>` supplied in the
verbatim args, but only matches args starting with `-config`, so
`--config-file` slips past the merge. `--checker-config` avoids the whole
problem by landing the options inside that single `-config`, and CodeChecker
then also defaults `HeaderFilterRegex` to `.*`, which is what we want with the
skiplist doing the scoping. Verified by reading the built command out of a
failed-analysis zip: one `-config`, no `--config-file`, all five options
present. `tt_metal/jit_build/kernel_clang_tidy/.clang-tidy` survives only for
the local `--run` path and mirrors those options.

`clangsa` is deliberately not enabled yet. Nothing path-sensitive runs today,
and on riscv32 cross-compiled code leaning on SFPI's analysis-fallback
builtins it may report false positives from the intrinsic stubs; it is a
follow-up once the `--enable-all` volume is understood.

## Coverage and known gaps

* **What limits the finding count.** The first all-hardware run should be read
  with three suppressors in mind, because they explain a surprisingly small
  number far better than "the kernels are clean". First, coverage is per-leg
  JIT compiles: a single small group (`trace allocation tracker`) compiled
  only **8** TUs, against ~2450 kernel sources in the tree — hence wiring
  every leg. Second, `--dedupe kernel-role` (the default) lints one config per
  (kernel, RISC target), so the same kernel under many compile-time-arg
  configurations is analyzed once. Third and largest, the two gates described
  in [Checker selection and scoping](#checker-selection-and-scoping) were both
  set far tighter than anyone had reviewed: 189 muted checks including all of
  `clang-analyzer-*`, and a `HeaderFilterRegex` covering only the `kernels/`
  directories, which dropped every check finding from some 2000 in-repo device
  headers (`tt_metal/hw/ckernels` 447, `tt_metal/hw/inc` 378, `tt_metal/tt-llk`
  1138, `tt_metal/fabric/hw` 55) — where much of the device logic actually
  lives. Both have since been replaced. Any comparison against the early runs
  should account for that rather than reading it as a regression.
* **Colourized log suffix (fixed).** tt-logger appends a `(build.cpp:NNN)`
  source location, and with colour on it arrives wrapped in SGR escapes:
  `...idle_erisck.cc \x1b[90m(build.cpp:686)\x1b[0m`. The escapes pushed the
  suffix off the `$` anchor in `LOG_CMD_RE`, so the strip missed and the whole
  escape run was spliced into the argv as an extra input file — every captured
  entry carried it, and `clang++` reported "no such file or directory" once
  per TU. The parser now strips SGR sequences before matching, with a
  `--self-test` case covering the colourized form.
* **Coverage = what the run compiled.** One test lints one test's kernels; a
  suite lints what the suite exercises. Kernels (or TRISC roles, or `#ifdef`
  branches) the run never compiled are not analyzed.
* **Data-movement / ethernet / dispatch / fabric kernels: parse cleanly** (no
  SFPI intrinsics involved). Verified on wormhole BRISC reader kernels.
* **Compute (TRISC) kernels: MATH and UNPACK parse cleanly** against the real
  `sfpi.h` header stack, thanks to SFPI's shipped analysis fallback
  (`tensix_builtins.h` + machine-generated `tensix_builtins.def`; present in
  the pinned SFPI ≥ 7.73.0). **Wormhole PACK TUs currently fail to parse**:
  `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h:393` puts
  `[[maybe_unused]]` on a template parameter, a GCC extension clang rejects.
  `ckernel_sfpu_recip.h` has the same pattern. This surfaces in the report as
  a `clang-diagnostic-error` and is left there deliberately, for the LLK
  owners to judge: `tt_metal/tt-llk` is part of this repo, so it is fixable
  here, but the call belongs to them rather than to this tooling change.
  Until it is resolved, wormhole PACK TUs contribute parse errors instead of
  check findings.
* Blackhole is expected to behave like wormhole (same mechanisms; multilib and
  `-mcpu` mappings are in place) but has not been exercised yet. Quasar is
  untested and its headers use the same template-parameter-attribute extension
  in several places.
* Findings quality: the parse differs from the device build in controlled ways
  (clang vs GCC, generic `rv32im` instead of the TT cpu model, address-space
  attributes `rvtt_l1_ptr`/`rvtt_reg_ptr` ignored). Fine for tidy checks;
  don't expect codegen-dependent diagnostics to be meaningful.
* The `--disable` list is deliberately short and is not a triage backlog: the
  report is meant to be a complete database of problems, and deduplication —
  not muting checkers — is what keeps it affordable. Note that finding *counts*
  cannot be reproduced locally: the captured translation units reference
  wheel-installed sources that exist only in the CI container, so a local
  analyze fails with `no-sources`. The numbers here come from CI plists parsed
  locally.
