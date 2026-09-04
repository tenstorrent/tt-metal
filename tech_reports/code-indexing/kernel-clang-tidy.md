# Post-hoc clang-tidy on JIT-compiled kernel code

Device kernels are compiled at runtime by `tt_metal/jit_build/` with the SFPI
cross-compiler, so they are invisible to the host build's static analysis.
This flow runs clang-tidy on them **after the fact**: run any workload
normally, capture the real JIT compiler invocations with
[`bear`](https://github.com/rizsotto/Bear), translate them for clang, and lint.

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

Prerequisites: `bear` (`apt-get install bear`), any clang-tidy ≥ 17.

```bash
cd <tt-metal root>

# 1. Cache hits spawn no compiler process, so force real compiles.
export TT_METAL_FORCE_JIT_COMPILE=1
# 2. If kernel ccache is enabled (TT_METAL_CCACHE_KERNEL_SUPPORT), a ccache hit
#    also skips the compiler exec; make ccache a pass-through.
export CCACHE_DISABLE=1

# 3. Run any test/workload under bear. Use absolute paths.
bear --output /tmp/kernel_raw_cc.json -- pytest /abs/path/to/tests/ttnn/...::test_case

# 4. Translate GCC->clang and run clang-tidy over every captured kernel compile.
python3 scripts/build_kernel_clang_tidy_commands.py \
    --input /tmp/kernel_raw_cc.json \
    --output-dir /tmp/kernel_tidy \
    --run \
    --config-file "$PWD/tt_metal/jit_build/kernel_clang_tidy/.clang-tidy"

less /tmp/kernel_tidy/findings.txt
```

Do **not** clear the tt-metal cache (`~/.cache/tt-metal-cache` or
`$TT_METAL_CACHE`) between the run and the lint: the captured commands
reference the generated headers there (`chlkc_*.cpp`, `chlkc_descriptors.h`,
`kernel_includes.hpp`, `defines_generated.h`), and clang-tidy re-parses from
those sources. Object files do not need to survive (they don't; the JIT build
uses temp names) — only sources/headers matter, and those are durable.

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
callers pass `enable-kernel-clang-tidy: true` plus `clang-tidy-target-group:
"<exact matrix group name>"`. That leg's pytest run is wrapped in `bear` (with
`TT_METAL_FORCE_JIT_COMPILE=1` and `CCACHE_DISABLE=1`), and after the tests a
non-blocking (`continue-on-error`) step runs the filter + clang-tidy and
uploads `kernel-clang-tidy-<group>` as an artifact (raw + translated
compile_commands.json, findings.txt, summary.txt). It is a prototype riding on
partial, run-dependent coverage — explicitly not a merge gate.

## Coverage and known gaps

* **Coverage = what the run compiled.** One test lints one test's kernels; a
  suite lints what the suite exercises. Kernels (or TRISC roles, or `#ifdef`
  branches) the run never compiled are not analyzed.
* **Data-movement / ethernet / dispatch / fabric kernels: parse cleanly** (no
  SFPI intrinsics involved). Verified on wormhole BRISC reader kernels.
* **Compute (TRISC) kernels: MATH and UNPACK parse cleanly** against the real
  `sfpi.h` header stack, thanks to SFPI's shipped analysis fallback
  (`tensix_builtins.h` + machine-generated `tensix_builtins.def`; present in
  the pinned SFPI ≥ 7.73.0). **Wormhole PACK TUs currently fail to parse**:
  `tt_llk_wormhole_b0/llk_lib/llk_pack.h:393` puts `[[maybe_unused]]` on a
  template parameter, a GCC extension clang rejects. One-line upstream tt-llk
  fix unblocks it.
* Blackhole is expected to behave like wormhole (same mechanisms; multilib and
  `-mcpu` mappings are in place) but has not been exercised yet. Quasar is
  untested and its headers use the same template-parameter-attribute extension
  in several places.
* Findings quality: the parse differs from the device build in controlled ways
  (clang vs GCC, generic `rv32im` instead of the TT cpu model, address-space
  attributes `rvtt_l1_ptr`/`rvtt_reg_ptr` ignored). Fine for tidy checks;
  don't expect codegen-dependent diagnostics to be meaningful.
* The `.clang-tidy` check list is inherited from PR #37252 and not yet
  re-triaged for this flow.
