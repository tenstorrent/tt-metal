# Compute-API ordering check

`check_compute_api_order.py` is a static checker for the stateful part of the compute kernel API.
It answers one question for every compute kernel: *is the hardware configured the way this
operation assumes?*

Today those rules live in prose (`docs/source/tt-metalium/tt_metal/apis/kernel_apis/compute/init_functions.rst`),
in "must be called before" comments, and in the runtime LLK Sanitizer. Breaking them still compiles
and the failure shows up later as a hang or silently wrong results (#61146, #59276). This checker
moves that knowledge into a rule table and applies it to the real per-thread translation unit that
the build compiles.

## What it reports

| kind | meaning |
| --- | --- |
| `missing-init` | an operation runs on a path where its init did not run, or an init for a different operation ran in between |
| `init-conflict` | an init for another operation clobbered the configuration the operation relies on |
| `missing-uninit` | a region that requires an uninit (tilize, untilize, broadcast, ...) was not closed |
| `deprecated-init` | an init listed in `.github/deprecations.json` |
| `arg-mismatch` | a compile-time init argument differs from the operation's argument it must match |
| `loop-invariant-init` | warning: an init inside a loop does not depend on the loop and can be hoisted |

Arguments that are not knowable at compile time are reported in the summary as
`not comparable at compile time` and are never guessed. Two different runtime locals are *not* a
mismatch; two different `constexpr` identities are.

## Running it

The checker needs a compile database that contains the generated per-thread translation units
(`chlkc_unpack.cpp`, `chlkc_math.cpp`, `chlkc_pack.cpp`). The weekly kernel clang-tidy workflow
publishes one as the `kernel-clang-tidy-*` artifact; on a machine with a card, set
`TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1` and run a test.

```bash
python tt_metal/tt-llk/infra/check_compute_api_order.py \
    --compile-db /path/to/compile_commands.json \
    --repo-root /path/to/tt-metal \
    --sfpi-root /opt/tenstorrent/sfpi \
    --strict
```

Useful flags:

* `--thread UNPACK|MATH|PACK` — analyse one thread.
* `--kernel REGEX` — only kernels whose path matches.
* `--json` — one object per finding (`kind`, `rule`, `location`, `rule_source`, ...).
* `--list-skipped` — say why a translation unit was skipped (parse error, no compute thread, ...).
* `--path-map OLD=NEW` — rewrite a prefix from the captured build environment, for example
  `/github/home/=/tmp/artifacts/github/home/`.

The checker translates the SFPI/GCC command line captured by the build into a clang command line
with `scripts/build_kernel_clang_tidy_commands.py::transform`, so a database produced by the normal
build works unchanged. Databases that already contain clang commands are used as they are.

## The rule table

`compute_api_rules.yaml` is the specification; the checker contains no API knowledge. Each rule is
one init/operation pair for one thread:

```yaml
- id: eltwise-binary-add
  thread: MATH
  init: add_init
  op: add_tiles
  arg_match: [[0, 0], [1, 1]]   # init arg 0 must match op arg 0, and so on
  invalidated_by: [sub_init, mul_init, ...]
  source: tt_metal/hw/inc/api/compute/eltwise_binary.h:87
```

Optional keys:

* `requires_uninit` / `uninit` — only for the curated set of regions that really must be closed.
  This is deliberately not "every init needs an uninit"; most inits do not.
* `arches` — restrict a rule to `wormhole_b0` or `blackhole`.
* `configured_at_entry` — the thread starts with this configuration already applied
  (`compute_kernel_hw_startup` configures the packer, for example).

`sfpi_analysis_shim.h` supplies one definition that SFPI 7.90.0 lacks on its non-Tensix fallback
path (`__xtt_vector8`). It is only needed to make the parse succeed; it changes no semantics.

### Adding a rule

1. Find the init call and the operation in the LLK API (`tt_metal/hw/ckernels/<arch>/metal/llk_api/`)
   or in the compute API (`tt_metal/hw/inc/api/compute/`).
2. Add the rule with a `source:` that points at the doc comment, LLK code, or bug that justifies it.
3. Run the validator tests. They check the schema, that every named symbol exists in the headers,
   that argument indices fit the signatures, that citations point at real lines, and that the
   deprecated-init list stays in sync with `.github/deprecations.json`.
4. Add one positive and one negative kernel to `tests/test_check_compute_api_order.py`.

## What it does not do

* It does not execute the kernel and has no model of runtime values. It only proves what the
  source proves.
* It does not analyse dataflow (NoC) kernels.
* It does not cover Quasar, or kernels written against the folded LLK 2.0 API that manage
  init/uninit internally.
* It follows helper functions and templates that it can see in the translation unit; a call whose
  body lives outside the TU is treated as a leaf.
* A translation unit that does not parse cleanly is reported as skipped rather than analysed
  partially, so a clean run is never a false clean.

## Tests

```bash
python -m pytest tt_metal/tt-llk/infra/tests/test_check_compute_api_order.py \
                 tt_metal/tt-llk/infra/tests/test_compute_api_rules.py
```

The end-to-end tests use a hermetic stub of the compute API and a one-entry compile database, so
they run without the SFPI toolchain. Each test exercises one diagnostic; the assertion message
states the behaviour a reviewer should be able to read off the kernel.
