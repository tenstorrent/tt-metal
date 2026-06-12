# Preprocess-and-ship: a tree-agnostic JIT compile server

## Problem
The remote JIT compile server (upstream infra) ships **paths**, not code: a `CompileRequest`
references source/header paths + include dirs + defines, and the server compiles them off its
**own filesystem**. So the server needs a byte-identical checkout of the client's tree, and a
divergence fails silently (wrong/stale kernels, no cross-machine content check). That coupling
blocks using a generic compile farm.

## Idea
Make the compilation unit **self-contained on the client** and ship the *content*, so the server
needs only the toolchain — no source tree, no headers, no defines. Preprocess each kernel with
`g++ -E -P` (the include universe is only ~21MB and pulls in zero ttnn), producing a `.ii` with all
`#include`s expanded and `#define`s applied. Ship the `.ii` as content; the server compiles it
directly. This also removes the source/header fingerprint problem entirely — only toolchain
coupling survives.

## Implementation (env-gated: `TT_METAL_JIT_PREPROCESS=1`)
- **`program.cpp` (`build_kernel_descriptor`)** — client side. For each target source: run
  `<gpp> -<opt> <cflags> <includes> <defines> -E -P -o <name>.ii <src>` in the target cwd (so
  `-I.`/`-I..` resolve identically to the real compile env), read the `.ii`, push it as a
  `GeneratedFile` (the existing content channel — written into the per-kernel cache dir on the
  server, no RPC/server change needed), and repoint `target.srcs[i]` to `../<name>.ii`. Then
  **clear `target.includes` and `target.defines`** (baked into the `.ii`) and add `-Wno-error`
  (`-P` drops the system-header markers that mask libstdc++ warnings; codegen is unaffected, and
  the original source compiled clean).
- **`jit_compile_server.cpp` (`build_target`)** — server side. A preprocessed `.ii` has no
  `#include`s, so `-MMD` yields an empty `.d` and **no `.dephash` is written**; tolerate its
  absence (a missing dephash conservatively forces a recompile next time — correct, and the common
  fresh-cache farm case).

## Why `-P`
`-P` inhibits line markers so the unit is fully self-contained: the server's `-MMD` scan then
references only the shipped `.ii`, never the (absent) original source/header paths. `__FILE__`/
`__LINE__` are already expanded by `-E`, so `-P` costs only debug line-table attribution, not
codegen.

## Status / next
Implemented + gated; default-off (no behavior change). Validate end-to-end against the remote
server (fresh-cache run, confirm kernels compile tree-free + a real run hits them). The
`up_front_compile` collector (in `wt_origin_main`) is the natural producer of the program set to
preprocess-and-ship to a farm. See memory: standalone-jit-server, jit-server-state-coupling.
