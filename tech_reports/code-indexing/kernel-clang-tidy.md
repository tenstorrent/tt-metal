# Post-hoc clang-tidy on JIT-compiled kernel code

Device kernels are compiled at runtime by `tt_metal/jit_build/` with the SFPI
cross-compiler, so the host build's static analysis never sees them. This flow
lints them after the fact: run any workload with the JIT build's compile-command
logging enabled, parse the logged compiler invocations, translate them for
clang, and run clang-tidy.

Nothing is synthesized — no stub headers, no enumerated kernel configs. The
runtime already produced the real compile-time args, defines and generated
headers as a side effect of running the test. Coverage is exactly the set of
kernels that ran JIT-compiled.

Related reading:

* [`kernel-code-indexing.md`](./kernel-code-indexing.md) — the adjacent IDE
  indexing flow. Its script rewrites TUs to the bare kernel source and dedups
  TRISC roles, which suits clangd but is wrong for linting; this flow keeps the
  real wrapper TUs so UNPACK/MATH/PACK stay distinct and role defines stay
  correct.
* `tt_metal/jit_build/kernel_clang_tidy/README.md` — why no mock SFPI headers
  are needed.

## Running it locally

Prerequisites: any clang-tidy ≥ 17.

```bash
cd <tt-metal root>

# 1. Cache hits skip the compile, so force real compiles.
export TT_METAL_FORCE_JIT_COMPILE=1
# 2. Kernel ccache also skips the real compile on a hit. Take it out of the
#    path, and disable it as a fallback in case something else invokes it.
unset TT_METAL_CCACHE_KERNEL_SUPPORT
export CCACHE_DISABLE=1
# 3. Log every kernel compile command. The lines are logged at info level, so
#    the logger must be at info too.
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

Both capture modes see only compiles that actually happen, and every cache layer
defeats them silently — the script warns on zero entries, but the failure to
suspect is "cache hit", not "capture broke". Three layers exist: the tt-metal
JIT cache (`~/.cache/tt-metal-cache`, defeated by `TT_METAL_FORCE_JIT_COMPILE`),
kernel ccache (`TT_METAL_CCACHE_KERNEL_SUPPORT`, which prepends `ccache` to the
SFPI command), and in CI a Redis-backed remote ccache configured by
`.github/actions/setup-job`, which the capture leg clears.

Do **not** clear the tt-metal cache (`~/.cache/tt-metal-cache` or
`$TT_METAL_CACHE`) between the run and the lint: the captured commands reference
the generated headers there (`chlkc_*.cpp`, `chlkc_descriptors.h`,
`kernel_includes.hpp`, `defines_generated.h`), and clang-tidy re-parses from
those sources. Object files do not need to survive; only sources and headers do.

A [`bear`](https://github.com/rizsotto/Bear)-captured `compile_commands.json`
works as an alternative input (`--input` instead of `--input-log`). Note that
bear 3.0.x's intercept channel is a gRPC server on loopback, and gRPC routes
loopback through `http_proxy` unless `no_proxy` covers it — which is why it
fails outright in containers that set a proxy, CI's included.

## What the translation does

See the docstring of `scripts/build_kernel_clang_tidy_commands.py` for the full
list. Summary: keep only SFPI `-c` compile entries; swap the compiler for
`clang++ --target=riscv32-unknown-elf` (mapped from `-mcpu=tt-wh`/`tt-bh`); drop
SFPI-GCC-only flags (`-ftt-*`, `-flto=auto`, `--param=min-pagesize=0`, dep-file
flags); upgrade `-std=c++17` to `-std=c++20` (the `-ftt-*` flags backport C++20
features that tt-llk headers use); wire in the SFPI toolchain's own
newlib/libstdc++ headers; apply the riscv32 `int32_t`-is-`long` type-model
overrides that `sfpi.h` static_asserts. Everything else — CTAs, defines, include
paths, generated files — passes through untouched.

By default entries are deduplicated to one per (kernel source, RISC target): the
same kernel recompiled under many compile-time-arg configurations is linted once,
with the first-captured config. Pass `--dedupe none` to lint every configuration.

## CI wiring

`.github/workflows/kernel-clang-tidy.yaml` is the entry point: build → run the
ttnn sanity suite on hardware via `ttnn-sanity-tests-impl.yaml` with
`enable-kernel-clang-tidy: true` → a `consolidate-report` job that merges every
leg's findings into one report and publishes it to
`tenstorrent/tt-metal-kernel-clang-tidy-results` gh-pages. It runs weekly on
Saturdays at noon PST, and on dispatch. The weekly run publishes because it is
on main; a dispatch publishes only with `publish-html: true`.

```sh
gh workflow run kernel-clang-tidy.yaml --ref <branch> \
  -f enabled-skus=wh_n300_civ2 -f publish-html=true
```

**Every** hardware leg captures and lints its own kernels, the way
`collect-coverage` applies to every leg. The leg's pytest run gets
`TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1` and `TT_LOGGER_LEVEL=info` with all
three cache layers defeated for that leg only, after which a non-blocking
(`continue-on-error`) step translates the captured commands and runs
`CodeChecker analyze --config
tt_metal/jit_build/kernel_clang_tidy/codechecker.json` — the same tooling
`clang-static-analyzer.yaml` uses. The leg renders no HTML. It uploads one
`kernel-clang-tidy-<group>` artifact holding the plists, the compile database,
finding counts, and the generated JIT sources those plists reference.
CodeChecker's `reports/fixit` suggestions are deleted first: 304 MB of an 880 MB
artifact, and nothing downstream reads them.

CodeChecker's compilation-database parser consults `ClangSA.analyzer_binary()`
unconditionally, so a `clang` binary must be resolvable even though only
clang-tidy runs. Both jobs install `update-alternatives` symlinks for `clang`,
`clang++` and `clang-tidy`, as `tt-umd`'s `code-analysis.yaml` does.

### The consolidated report

`consolidate-report` merges every leg's plists and runs `CodeChecker parse
--export html` once, so the published site is a genuine CodeChecker report: a
sortable Severity / Checker / File / Message table plus its own checker- and
severity-statistics pages. Leg provenance is dropped deliberately — the same
kernel code is analyzed on many legs and only the finding matters.
`.github/scripts/utils/brand_kernel_tidy_site.py` then sets a page title and
favicon, neither of which `--export html` makes configurable.

Three constraints shape that job.

**The merge deduplicates before CodeChecker sees anything.** Each TU is a
firmware wrapper with the kernel `#include`d into it, so every TU re-reports the
defects of every header it pulls in, and every leg repeats that again: two legs
produced 370,060 raw diagnostics for 19,863 distinct findings — 18.7x
redundancy, 588 MB of plists. `CodeChecker parse` is single-threaded, has no
`--jobs`, and its cost scales with the raw count, so most of its runtime
rediscovers findings it has already seen.
`.github/scripts/utils/merge_kernel_tidy_plists.py` drops the duplicates in
parallel first, keying on file, line, column, checker and message; `parse` then
takes 6.7s instead of 115s for an identical finding set.

**Rendering needs the analyzed sources at the same absolute paths the legs
used**, which is why the job runs in the ci-test container with `setup-job`
installing the wheel rather than on a bare `ubuntu-latest`. Two roots cover
almost everything: `/opt/venv/...` (wheel-installed headers) and `/work/...`
(the checkout). Without them, `--export html` silently writes `index.html` and
`statistics.html` with **zero** finding pages. The generated JIT glue is the
exception — it lives in the test runner's kernel cache, so each leg ships the
referenced sources in its artifact and the consolidate job restores them.

**The JSON export runs before the render and uploads unconditionally.** It is
the machine-readable form of the same data and the thing to point an agent at.
`findings.json` is a slimmed projection of it; CodeChecker's own export carries
bug paths and macro expansions and runs to ~124 MB, over GitHub's 100 MB
per-file limit, so it stays in the artifact and is excluded from the published
site.

### Why not the simulator legs?

Only the compile commands matter, not the device result, so a `sim_*` leg looks
appealing. It does not work, and the tidy steps skip those SKUs. Sim runs
pytest-xdist `-n 4`, and xdist workers use their stdout as the execnet RPC
channel, so the logged compile commands never reach the run log. Serializing the
leg fixes that but forfeits the parallelism sim most needs — and ttsim runs slow
dispatch, so tests needing trace or fast dispatch skip themselves: a serialized
sim leg captured zero commands. Sim is also 10-50x slower per op on cloud
runners, against a capture leg that is already slow by design.

## Checker selection and scoping

Both are driven by `tt_metal/jit_build/kernel_clang_tidy/codechecker.json`, not
by a check list in `.clang-tidy`.

**Selection** is `--enable-all` minus a short `--disable` list. `--enable-all`
is not literally all: of the 1,510 checkers CodeChecker knows, 1,288 run, and 50
of the rest are excluded by `--enable-all` itself because they carry
`profile:extreme`. That exclusion is silent and worth remembering when a checker
seems absent — `google-readability-casting` is one such (see below).

Twelve entries are whole families that cannot apply to bare-metal RISC-V device
code (`abseil`, `altera`, `android`, `boost`, `darwin`, `fuchsia`,
`linuxkernel`, `llvmlibc`, `mpi`, `objc`, `openmp`, `zircon` — 72 checkers
between them). The rest are specific to this domain:

| Disabled | Reason |
| --- | --- |
| `hicpp-no-assembler` | Inline asm is pervasive in `tt_metal/hw/inc` and tt-llk. |
| `portability-simd-intrinsics` | SFPI *is* a SIMD intrinsics layer, by design. |
| `clang-diagnostic-c++98-compat` | Device code is C++17/20. |
| `prefix:cert` | All 41 `cert-*` checks are aliases, so the prefix removes no capability; 461 of the 463 findings it drops are reported at an identical position under the aliased original. |
| `clang-diagnostic-reserved-identifier` | A third name for what `bugprone-reserved-identifier` already covers. Its 3 unique findings are the linker-mandated `_start`. |
| `clang-diagnostic-unused-parameter` | 846 of its 907 findings share an exact position with `misc-unused-parameters`, which is kept because it also carries a fix-it. |
| `modernize-use-trailing-return-type` | Pure style, 1,182 findings; tt-umd mutes it too. |
| `bugprone-easily-swappable-parameters` | Every device API parameter is `uint32_t`, so it fires on nearly every function. 1,410 findings whose only fix is strong-typedef wrappers across the whole ABI. |
| `clang-diagnostic-unsafe-buffer-usage` | Wants `std::span` for all pointer arithmetic, but kernels address L1 at fixed hardware addresses through raw pointers. 1,315 findings carrying two distinct messages, neither specific. |
| `clang-diagnostic-documentation` | 561 of its 642 findings object to LLK's prescribed `@param name:` style — clang reads the colon as part of the parameter name — in comments no docs build parses (`docs/Doxyfile` names no tt-llk path). |
| `clang-diagnostic-extra-semi-stmt` | 96% of its 347 findings are function-like macros invoked in statement position (`v_endif;` and friends), where the semicolon belongs to the call site and some macros expand to nothing by design. |
| `performance-enum-size` | Measured zero saving. Of 86 enums flagged, most are never the type of a stored field; the ~5 that are either pad back to the same size or are ABI contracts whose offsets are fixed by `static_assert`. |
| `modernize-avoid-c-arrays` | 21,785 findings, 98.3% of them generated `chlkc_descriptors.h` tables. `std::array` is available and used in device code, but measured on those tables the cost is the header, not the type: `<array>` is ~42 ms per TU to parse against ~0.5 ms to convert, and kernels compile at runtime. |

Nothing else is muted, and volume alone is not a reason to mute: deduplication,
not disabling checkers, is what keeps the report affordable, and the point of it
is a complete database of problems to fix.

`take-config-from-directory` must stay off, or CodeChecker returns an empty
checker list (`clangtidy/analyzer.py:471`) and every `--enable`/`--disable` is
inert.

`clangsa` is deliberately not enabled yet. Nothing path-sensitive runs today,
and on riscv32 code leaning on SFPI's analysis-fallback builtins it may report
false positives from the intrinsic stubs; it is a follow-up once the
`--enable-all` volume is understood.

**Scoping** is `--skip <skiplist> --drop-reports-from-skipped-files`. The
skiplist is preferred to `HeaderFilterRegex` because it also drops
`clang-diagnostic-*` findings, which a header filter structurally cannot, and it
keeps scope in one reviewable file. It is exclusion-only, and an allow-list is
not an option: the TUs are the firmware wrappers under
`tt_metal/hw/firmware/src/` with kernels `#include`d into them, so no TU path
contains `kernels/` and an allow-list keyed on it would analyze nothing.
Exclusion-only also fails open as new in-repo device directories appear. The
list now covers only upstream SFPI and host libc.

**SFPI headers are demoted to `-isystem` during capture**, which is what
actually saves analysis time; the skiplist only discards findings the analyzer
has already produced. The device build passes `-I /opt/tenstorrent/sfpi/include`
while every other SFPI path already arrives as `-isystem`, so clang-tidy treats
`sfpi.h` as first-party — the same distinction CMake's `SYSTEM` keyword draws for
host dependencies. `-isystem` is the only lever available, because CodeChecker
forces `HeaderFilterRegex=".*"` whenever no `--analyzer-config` is given
(`clangtidy/analyzer.py:665`), and clang-tidy's `SystemHeaders` defaults to false
independently of it. The parse still happens — the TU needs the AST — but
diagnostic matching and fix-it construction do not. The rewrite is in the capture
script, so it cannot affect the device build, and the skiplist keeps its SFPI
entry as a backstop since `clang-diagnostic-error` can still surface from system
headers.

**The generated JIT glue in the kernel cache is analyzed on purpose.**
`genfiles.cpp` emits the `chlkc_*` prologs, the `kernel_main()` shim and the
`chlkc_descriptors.h` tables, and a generator emitting bad code is a bug worth
filing. One consequence is accepted: cache paths embed the kernel name and two
content hashes, so a single generator defect lands once per kernel variant
instead of collapsing under deduplication. If that becomes unmanageable, the fix
is to canonicalise cache paths during the merge, not to stop looking. Kernel
coverage is unaffected either way — kernels resolve to wheel or checkout paths.

**Per-path exceptions** are the right mechanism when a checker is correct in
general but wrong about a specific construct.
`tt_metal/jit_build/kernel_clang_tidy/review_status.yaml` assigns a review
status to findings matched by `filepath` (an fnmatch glob), `checker_name`
(exact) or `report_hash` (prefix). `parse` defaults `--review-status` to
`confirmed,unreviewed`, so anything marked `intentional` or `false_positive`
drops out of the HTML report, `findings.json` and the statistics alike. The
schema key is `$version`, not `version`, which `--help` does not mention.

The consolidate job applies it by copying the file into the merged report
directory, rather than the legs applying it at analyze time, so an exception
costs one job re-run instead of sixteen hardware legs. Suppressed findings are
still analyzed. Two rules qualify today:

* `bugprone-suspicious-include` on `*/kernel_includes.hpp` and `*/chlkc_*`
  (1,488 findings). Including a `.cpp` is exactly how the JIT builds a kernel.
  The globs cover all 1,488 while leaving the 21,550 findings other checkers
  report in those same files visible, which a `--disable` would not.
* `modernize-macro-to-enum` on `*/hw/inc/internal/*` (3,968 findings, every one
  of them under that path, 3,181 in `cfg_defines.h` alone). Those are Tensix
  register definitions that track the hardware and are tested with `#if`, where
  an enum is invisible to the preprocessor.

The bar is that the finding is *wrong about this code* — not that it is
unwelcome or numerous — and that it is scopable by path.
`readability-magic-numbers` (2,567) fails the first test: it concentrates in
SFPU polynomial coefficients, tile geometry and Tensix instruction encodings,
and bit positions are precisely what should be a named constant.
`performance-no-int-to-ptr` (694) and `clang-diagnostic-old-style-cast` (766)
fail the second, their top five files holding only 13-29% of their findings.
`misc-non-private-member-variables-in-classes` (416) fails both: roughly 115
findings are genuine hardware layouts with `sizeof`/`offsetof` asserts, but the
rest are ordinary classes carrying dozens of member functions, and the two kinds
are mixed within single files.

**Check options** retune six checks. Three values are inherited from the repo's
root host `.clang-tidy`; the rest are chosen for device code.

| Option | clang-tidy default | Ours |
| --- | --- | --- |
| `readability-function-cognitive-complexity.Threshold` | 25 | 25 (host config: 312) |
| `readability-function-cognitive-complexity.IgnoreMacros` | false | false (host config: true) |
| `readability-simplify-boolean-expr.SimplifyDeMorgan` | true | false |
| `readability-else-after-return.WarnOnUnfixable` | true | false |
| `readability-else-after-return.WarnOnConditionVariables` | true | false |
| `modernize-use-auto.MinTypeNameLength` | 5 | 9 |
| `readability-identifier-length.IgnoredLoopCounterNames` | `^[ijk_]$` | `^[ijk_nchwdbtr]$` |
| `readability-identifier-length.IgnoredVariableNames` | (none) | `^([NCHWD]\|[xyz]\|[ijk]\|[WHCND]t\|cb\|id)$` |
| `readability-identifier-length.IgnoredParameterNames` | `^[n]$` | `^(n\|[xyzab]\|[ijk]\|id\|cb\|vc\|[WHCND]t)$` |
| `readability-uppercase-literal-suffix.NewSuffixes` | (all) | `L;UL;LL;ULL` |

Cognitive complexity is the one place that deliberately does *not* follow the
host config, whose `Threshold=312` with `IgnoreMacros=true` produced 5 findings
out of 62,710 while ignoring exactly the macro-driven complexity that dominates
LLK and the SFPU headers. That pairing suits a blocking gate, not a report meant
to enumerate problems. Both config files restate clang-tidy's defaults
explicitly so a future "sync with host" edit does not silently undo it.

`modernize-use-auto.MinTypeNameLength` measures the *base* type name, ignoring
`const`, `volatile` and `*`, and fires when it is **at least** the threshold. At
the default of 5 the check reported 2,137 times, 89% of them on the kernel
runtime-arg preamble `uint32_t x = get_arg_val<uint32_t>(i)`, where the
duplication it objects to is eight characters on the same line. 9 is the value
that excludes `uint32_t` (8) while keeping the ~131 sites with a genuinely long
name to duplicate: `sfpi::vFloat`, `RealtimeProfilerState`, `DataFormat`.
Compile time played no part — measured front-end only, `auto` costs about a
microsecond per declaration and the sign flips between compilers.

`readability-identifier-length` exempts this domain's vocabulary: NCDHW layout
letters as loop counters, `Wt`/`Ht` tile dimensions, `cb` for circular buffers,
`x`/`y`/`z` NOC coordinates, `vc` for virtual channels, `a`/`b` as SFPU binary
operands. The three lists are separate options because the check scores
variables, parameters and loop counters separately, and they are deliberately
asymmetric: `h` is fine as a counter bounded three lines away but not as a
function-scope variable. Together they take the check from 3,243 findings to
1,385, leaving the terse-rather-than-conventional names (`s`, `r`, `v`, `p`)
reported. The blunter `MinimumVariableNameLength=2` was rejected because it
accepts any two-character name rather than naming what is exempt and why.

`readability-uppercase-literal-suffix.NewSuffixes` keeps only the l-family. The
check exists because a lowercase `l` is confusable with `1`, so `1l` reads as
`11`; no such ambiguity exists for `u` or `f`, which is all this codebase writes
(2,713 and 1,052, against a single `ul` and no bare `l`). Two non-obvious
details: each listed suffix is also the *suggested replacement*, so entries must
be spelled fully uppercase or the fix-it produces `3uL`; and matching is against
the suffix as written, so an all-uppercase list still catches lowercase literals.

Options go through `--checker-config clang-tidy:<checker>:<option>=<value>`,
never a config file. Once `take-config-from-directory` is off, CodeChecker
builds its own `-config` for clang-tidy (`clangtidy/analyzer.py:509`), and
passing `--config-file` alongside it makes clang-tidy abort every TU with
"--config-file and --config are mutually exclusive".
`tt_metal/jit_build/kernel_clang_tidy/.clang-tidy` exists only for the local
`--run` path and mirrors the same options; keep the two in sync.

## Coverage and known gaps

* **Coverage is what the run compiled.** One test lints one test's kernels; a
  suite lints what the suite exercises. Kernels, TRISC roles or `#ifdef`
  branches the run never compiled are not analyzed. The scale of that matters:
  one small test group compiled 8 TUs against ~2,450 kernel sources in the tree,
  which is why every leg captures. `--dedupe kernel-role` then lints one config
  per (kernel, RISC target).
* **Wormhole PACK TUs currently fail to parse.**
  `tt_llk_wormhole_b0/llk_lib/llk_pack.h` and `ckernel_sfpu_recip.h` put
  `[[maybe_unused]]` on a template parameter, a GCC extension clang rejects, so
  those TUs contribute a `clang-diagnostic-error` instead of check findings.
  [#55669](https://github.com/tenstorrent/tt-metal/pull/55669) removes it.
  Data-movement, ethernet, dispatch and fabric kernels parse cleanly, as do the
  MATH and UNPACK roles.
* Blackhole is expected to behave like wormhole (same mechanisms; multilib and
  `-mcpu` mappings are in place) but has not been exercised yet. Quasar is
  untested and uses the same template-parameter-attribute extension in several
  places.
* **Findings quality.** The parse differs from the device build in controlled
  ways: clang rather than GCC, generic `rv32im` rather than the TT cpu model,
  address-space attributes `rvtt_l1_ptr`/`rvtt_reg_ptr` ignored. Fine for tidy
  checks; codegen-dependent diagnostics are not meaningful.
* **Finding counts cannot be reproduced locally.** The captured TUs reference
  wheel-installed sources that exist only in the CI container, so a local
  analyze fails with `no-sources`. The counts quoted here come from CI plists
  parsed locally.
* Fixes filed from this report:
  [#55658](https://github.com/tenstorrent/tt-metal/pull/55658) (stray
  semicolons), [#55659](https://github.com/tenstorrent/tt-metal/pull/55659)
  (declare `kernel_main()`, which is 461 of the 542
  `clang-diagnostic-missing-prototypes` findings — kernels define it at global
  scope and nothing declares it first),
  [#55669](https://github.com/tenstorrent/tt-metal/pull/55669)
  (template-parameter attributes).
* If `clang-diagnostic-old-style-cast` (766) is ever taken on, switch checker
  first: the compiler diagnostic emits one message and no fix, whereas
  `google-readability-casting` names the replacement and emits fix-its for the
  mechanical cases while declining to auto-fix integer-to-pointer casts. It is
  off only because it carries `profile:extreme`; one `--enable` turns it on.
