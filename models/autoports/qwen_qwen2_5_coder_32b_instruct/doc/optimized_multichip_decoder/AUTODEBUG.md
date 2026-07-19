# AutoDebug Report: shard-advisor import failure

## Status

The import failure is explained. It is a native TTNN ABI mismatch caused by
the inherited current-checkout `LD_LIBRARY_PATH` overriding the advisor
runtime's pinned `RUNPATH`. The advisor import barrier has a no-rebuild shell
workaround, verified with import-only controls. No TT device was opened and no
implementation or build file was changed.

The prescribed fresh `autodebug.sh` runner was attempted first with Codex
`gpt-5.5`/`xhigh`, but its nested sandbox could not start because `bwrap` was
unavailable. The investigation below was therefore completed directly in the
already-fresh AutoFix subagent.

## Direct observations

- Current `tt-metal`: `860d7d688063c51fcc41a49c9d7611a1f761d535`.
- Advisor `tt-mlir`: `3f8b9c0a258772648a754e0c39504ef3c19add09`
  on `ttnn-jit-shard-advisor`, with local tracer/advisor edits.
- Advisor's vendored `tt-metal` is a real directory, not the same-checkout
  symlink described by the shard-advisor setup guide. It is at
  `13adda80c119631d18b0bc06163416ba148c25ab`.
- The installed advisor uses Python 3.12, and the failing runtime currently is
  `/opt/ttmlir-toolchain/venv/lib/python3.12/site-packages/ttnn_jit/libTTMLIRRuntime.so`.
- That installed runtime has SHA-256
  `1ac229c6af84c5551a007020b4f2512d710f45e3b2efcd49ca436b7c462c1d91`,
  exactly matching `/home/mvasiljevic/tt-mlir/build/runtime/lib/libTTMLIRRuntime.so`.
- Its ELF `RUNPATH` includes
  `/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/lib`,
  but the active shell had
  `LD_LIBRARY_PATH=/home/mvasiljevic/tt-metal/build_Release/lib:`.
  `LD_LIBRARY_PATH` takes precedence over `DT_RUNPATH`.

## Headline finding: mixed TTNN native ABIs

`nm -D` shows that `libTTMLIRRuntime.so` requires the older symbols built at
the advisor's vendored `tt-metal` commit. For example, its unresolved
`moe_compute` symbol ends in:

```text
...MoEActivationFunctionEEbS9_S9_
```

The vendored `_ttnncpp.so` defines that exact mangled symbol. The current
checkout's `_ttnncpp.so` instead defines:

```text
...MoEActivationFunctionEEbS9_
```

The extra `S9_` is the old public `bh_ring_size` optional parameter. The same
ABI drift exists for the two weight preparation helpers: the runtime expects
five trailing integer arguments, while current TTNN exports four. Source diff
between `13adda80...` and `860d7d688...` identifies commit `107623bb9dd`
(`Bug Fix: MoE: Fix Blackhole problems`) as the change that removed the public
`bh_ring_size` arguments and auto-detects the ring from the device architecture.

The loader controls complete the causal chain:

| Control | `_ttnncpp.so` selected by `ldd` | `ldd -r` MoE result |
| --- | --- | --- |
| `LD_LIBRARY_PATH=/home/mvasiljevic/tt-metal/build_Release/lib` | current `860d7d...` | old runtime symbols undefined |
| `LD_LIBRARY_PATH=/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/lib` | vendored `13adda...` | no MoE undefined symbols |
| empty `LD_LIBRARY_PATH` | vendored `13adda...` via runtime `RUNPATH` | no MoE undefined symbols |

An import-only reproduction with vendored TTNN Python sources but the current
checkout library path fails exactly at `ttnn/ttnn/_ttnn.so` with the reported
old `moe_compute` symbol. With vendored Python sources and vendored libraries,
both `import ttnn` and `import ttnn_jit` pass, and `ttnn-advise --help` exits 0.
This also refutes the theory that the runtime was built without any provider:
the provider exists, but the dynamic loader was directed to an incompatible
provider first.

Qwen2.5-Coder-32B is dense and does not execute `moe_compute`. That does not
avoid this failure: `libTTMLIRRuntime.so` directly references the symbol, so
the dynamic loader must resolve it when the library is imported.

## Secondary finding: bootstrap is working-directory sensitive

`/home/mvasiljevic/tt-mlir/env/activate` derives all roots from `pwd`, while
the repo-local bootstrap sources it without changing directory. Sourcing the
bootstrap from `/home/mvasiljevic/tt-metal` sets `TT_MLIR_HOME` to the
`tt-metal` checkout and `TT_METAL_HOME` to the nonexistent
`/home/mvasiljevic/tt-metal/third_party/tt-metal/src/tt-metal`. This explains
the earlier missing-`ttnn` behavior. Sourcing it while the working directory
is `/home/mvasiljevic/tt-mlir` produces the intended Python roots, but still
does not change the inherited current-checkout `LD_LIBRARY_PATH`, leaving the
ABI failure above.

## Smallest safe repair for the existing advisor build

Use the advisor's Python and native TTNN artifacts as one matched set. This is
an environment-only repair and was verified through imports and CLI startup:

```bash
cd /home/mvasiljevic/tt-mlir
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh

advisor_metal=/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal
export LD_LIBRARY_PATH="$advisor_metal/build_Release/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$advisor_metal:$advisor_metal/ttnn${PYTHONPATH:+:$PYTHONPATH}"

python -c 'import ttnn, ttnn_jit; print(ttnn.__file__, ttnn_jit.__file__)'
ttnn-advise --help
```

Expected `ttnn.__file__` begins with
`/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/ttnn/`.
Do not solve this by copying only `_ttnncpp.so`, `_ttnn.so`, or
`libTTMLIRRuntime.so`; those libraries must remain an ABI-matched set.

The actual `capture` command was deliberately not run during AutoDebug because
it opens a TT device. The main optimization agent can now rerun it under the
repaired environment.

## Exact-current-checkout repair

The environment-only repair uses the advisor's vendored `tt-metal` commit, not
the current `tt-metal` checkout. The shard-advisor setup contract says those
should be the same source. If recommendations must be generated against the
exact current TTNN ABI, the correct one-time operator repair is to align the
advisor's TTNN source to `860d7d688...` and rebuild/reinstall the advisor's
TTNN, `libTTMLIRRuntime.so`, and `ttnn_jit` together. That is a large setup
operation and was outside this inspection-only task. Pointing current Python
sources or current native libraries at the old installed runtime is not a safe
substitute.

## Hypothesis ledger

| Hypothesis | Result | Evidence |
| --- | --- | --- |
| Missing `ttnn` remains the current failure | Refuted | Vendored `ttnn` is found; the exact failure is then a native symbol lookup. |
| The current and advisor TTNN builds have incompatible ABIs | Verified | Source diff and distinct mangled symbols for `moe_compute` and both preparation helpers. |
| The advisor runtime has no matching provider anywhere | Refuted | Vendored `_ttnncpp.so` defines every exact required MoE symbol. |
| Inherited `LD_LIBRARY_PATH` selects the wrong provider | Verified | `ldd`, `ldd -r`, exact reproduction, and passing pinned-pair control. |
| Qwen's capture script itself invokes MoE | Refuted | The target contains only RMSNorm, dense matmuls, add, slice, and mul; failure occurs before target loading. |

## Residual uncertainty

- Import and CLI startup are proven; hardware capture was not run in this
  inspection-only pass.
- The vendored-pair workaround may next expose a tracer handler or model API
  difference. Such a later error would be separate from the resolved ELF
  failure.
- For strict compliance with the same-checkout shard-advisor setup contract,
  the aligned rebuild remains required even if the vendored pair can capture
  this stable dense subgraph.
