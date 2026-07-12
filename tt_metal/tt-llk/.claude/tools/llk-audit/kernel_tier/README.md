<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# llk-audit kernel tier (opt-in, on-request)

The `cb-sync`, `noc-sync` and `mailbox-sync` checkers live in the committed tool
and are unit-tested — but most of the code they audit (circular-buffer / NoC /
mailbox handshakes) lives in **JIT-compiled kernels outside tt-llk** (ttnn ops, the
compute API, models), not in the tt-llk headers. Over the tt-llk tree `cb-sync` and
`noc-sync` correctly find nothing (zero in-tree surface); `mailbox-sync` yields its
small in-tree surface on a plain run but its large kernel surface only here. This
module builds the **kernel fact base** those already-committed checkers then run over.

**The split that matters:** the *checkers* are durable and in `main`. Only the
*capture* — turning a live JIT build into a fact base — is fragile (it tracks
tt-metal's `jit_build` and the sfpi GCC→clang gap), so it is opt-in and re-run
fresh each sweep, never wired into `main`'s build.

## Run it

From the tool dir, once this module is present (`MANIFEST` exists):

```bash
./run.sh wormhole --full-jit          # in-tree audit + this kernel tier
```

`--full-jit` runs the normal in-tree audit, then calls `bootstrap.sh <arch>
<out_dir>`. Two ways to feed it kernels:

```bash
# A) audit a pre-captured build log (recommended: capture once on HW, audit offline)
TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 <your workload> > build.log 2>&1
LLK_KT_LOG=build.log ./run.sh wormhole --full-jit

# B) let bootstrap run a workload for you (needs a WH/BH device or sim)
LLK_KT_CLEAR_CACHE=1 LLK_KT_WORKLOAD='<cmd that compiles the kernels>' \
  ./run.sh wormhole --full-jit
```

`LLK_KT_CLEAR_CACHE=1` forces the op kernels to recompile — already-cached
kernels emit no compile command, so without it the ledger may show 0 op kernels.

## How it works (`capture.py`)

1. Scrape `g++ compile cmd: <cmd>` lines from the build log (tt-metal emits one
   per kernel TU under `TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`; `build.cpp`).
2. Translate each RISC-V-GCC command to clang: drop the GCC-only flags
   (`-flto`, `-ftt-*`, `-mcpu=tt-wh`, `-fdump*`, `-Werror`, `-c -o`, `-MF`), add
   clang's `--target=riscv32-unknown-elf` + the sfpi `-isystem` paths + an SFPU
   shim, keep the kernel's own `-I`/`-D`.
3. Run `llk_extract` per kernel **from its cache build dir** (so relative includes
   and the generated `kernel_includes.hpp` resolve), then merge into one fact base
   scoped to the KERNEL surface. This scope must NOT be the repo root: sfpi (STL)
   lives under `runtime/sfpi` and the dataflow/LLK primitive **definitions** live
   under `hw/inc/api`, `hw/ckernels`, `tt_llk_*` — all in-repo — so a repo-root
   filter floods the base with library internals and the checkers flag the primitive
   *definitions* (e.g. `Semaphore::inc_multicast`'s own body) as kernel races. The
   scope is applied in **two stages** because no single `.contains()` substring can
   both catch `models/.../unified_kernels/` **and** exclude the LLK `.../ckernels/`
   defs (any substring loose enough for the former also matches the latter):
   - a COARSE `--path-filter` (default **`kernels/`**) is passed to the extractor as
     a cheap pre-scope (already excludes `hw/inc/api` + sfpi; it *does* admit
     `hw/ckernels`);
   - the PRECISE keep/drop (`capture.in_kernel_surface`: keep iff the path has
     `/kernels/` **or** `_kernels/` and not `/ckernels/`) is applied in Python on the
     emitted facts. `_kernels/` matches `unified_kernels/` but not `ckernels/` (no
     underscore) nor `/kernels/` (no underscore-before), so the two keep-substrings
     together cover the JIT/op kernel dirs + `<prefix>_kernels/` model trees while the
     `ckernels/` defs are trimmed. Dropped-fact counts are reported per-TU in the
     coverage ledger (`:drop=N`), never silently swallowed.
4. `bootstrap.sh` runs `cb-sync,noc-sync,mailbox-sync` over the merged facts.

## Coverage is honest, not silent

`capture.py` writes `kernel_coverage.<arch>.txt`: every kernel TU with parsed /
skipped / parse-error status and its fact count. **Recall is complete only over
the kernel variants the workload actually exercised** — a different op, dtype, or
tile shape compiles a different kernel. Kernels that fail to translate are listed,
never silently dropped. If nothing parses, `bootstrap.sh` exits non-zero rather
than emit a false "0 findings" all-clear.

## Requirements

- The `extractor/llk_extract` binary built (`extractor/build.sh`).
- sfpi matching the repo pin (`tt_metal/sfpi-version`) at `runtime/sfpi`
  (the GCC→clang `-isystem` paths and `sfpi::clamp/min/max` come from it).
- A runtime (WH/BH device or sim) only when bootstrap *runs* a workload (path B);
  path A needs only the log.
