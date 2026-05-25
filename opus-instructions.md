# tt-metal Static Polyglot Dependency Graph — Working Document

> **This file is the canonical living spec.** It must be read in full at the start of every session and after every context compression, before taking any action on this project. The memory entries at `/home/ubuntu/.claude/projects/-home-ubuntu-tt-metal/memory/` exist to enforce this — `MEMORY.md` is auto-loaded into context and points here.
>
> When the user approves a deviation, decision, or new scope, append it to this file *in the same turn*. Do not rely on conversation memory.

---

## 0. Operating Rules

### 0.1 Re-read on resume
Before any action on this project (session start, post-compaction, /resume), read this file in full. If you find yourself reasoning from memory of "what the spec says" without having read the file in the current context window, stop and read it.

### 0.2 Build & execute inside the container; write files from the host
- All compilation, CMake configure, libclang parsing, Python AST analysis, and any other heavy work runs **inside** the docker container `tt-metal-basic-dev-container`.
- Invoke commands inside the container via `sudo docker exec tt-metal-basic-dev-container bash -c "..."`. Docker on this host requires sudo; `Bash(sudo docker *)` is allowlisted in `.claude/settings.local.json`.
- `/workspace` inside the container is a bind-mount of `/home/ubuntu/tt-metal` on the host.
- **All file writes for scripts, source, configs, and graph outputs must be performed from the HOST** using the Write/Edit tools. Writing files from inside the container creates root-owned files in the bind mount and corrupts the git tree's ownership (user `ubuntu` / uid 1000).
- If a script run inside the container must emit files, redirect its output into a directory the script creates with the correct uid (or chown afterward).

### 0.3 Python virtual environment
- Every build, compilation, or analysis command run inside the container **must** execute inside the `/opt/venv` virtual environment.
- That venv is auto-activated by `/etc/bash.bashrc` for interactive shells, but `docker exec ... bash -c "..."` does not run interactively by default. Either:
  - prefix commands with `source /opt/venv/bin/activate && ...`, or
  - invoke the venv interpreter directly via `/opt/venv/bin/python` / `/opt/venv/bin/pip`.
- **Any custom library installs go into `/opt/venv`** (`uv pip install --python /opt/venv/bin/python ...` or activate the venv first and `pip install ...`). Never install into the system Python.

### 0.4 Output location
- All project artifacts (scripts, intermediate caches, final graph) live under `/home/ubuntu/tt-metal/dep-graph/`.
- Subdirs: `dep-graph/scripts/`, `dep-graph/out/`, `dep-graph/cache/`.
- `dep-graph/out/` and `dep-graph/cache/` must be `.gitignore`-d before any large files land there. **TODO**: confirm with user before editing `.gitignore`.

### 0.5 Documenting deviations
Every deviation from the **original spec** (Section 5, preserved verbatim) is logged in **Section 2** with date and rationale. Do not silently diverge — record it.

---

## 1. Current Scope

Updated 2026-05-25 after Phase 1 (slice) completion.

**In scope:**
- Host C++ under `tt_metal/` and `ttnn/cpp/`.
- `tt_metal/impl/` — host-side hardware orchestration. **In scope per user direction 2026-05-25**, even where it touches hardware.
- `tt_metal/jit_build/` — host code that drives kernel compilation. **In scope per user direction 2026-05-25**.
- Python under `ttnn/`, `tt_metal/` (host-side helpers), and any other in-tree Python that participates in the public ttnn surface.
- The nanobind / pybind bridge that connects the two.

**Out of scope (v1):**
- `tests/`, `**/test_*.py` — user wants them eventually; defer.
- `.github/` — not source code.
- `tt_metal/hw/` — kernel hardware path / firmware source.
- `tt_metal/third_party/tt_llk_*/` and any other LLK kernel source.
- `runtime/sfpi/` — SFPI compiler.
- `.cpmcache/` and stdlib — treat as opaque leaves at most; preferably filtered out.
- Build directories (`build_Release/`, `.build/`, `build/`).

---

## 2. Approved Deviations from Original Spec

| #  | Original | Deviation | Rationale | Date |
|----|----------|-----------|-----------|------|
| D1 | "complete ... massive codebase (~100k+ lines)" | tt-metal is millions of LOC; v1 restricts scope per Section 1 | tractability + user direction | 2026-05-25 |
| D2 | All Python + C++ | v1 excludes `tests/`, `.github/`, `tt_metal/hw/`, LLK (`tt_metal/third_party/tt_llk_*/`), SFPI (`runtime/sfpi/`), `.cpmcache/`, stdlib, build dirs. **Includes** `tt_metal/impl/` and `tt_metal/jit_build/` per refined user direction. See §1. | per user | 2026-05-25 |
| D3 | Node attrs `{id, language, file, line}` | Extended schema in Section 3 | required for useful RAG | 2026-05-25 |
| D4 | "Phase 1: configure CMake to emit compile_commands.json" | Configure **and** run codegen targets so generated headers (FlatBuffers, version, jit) exist before libclang parses | otherwise libclang produces malformed ASTs and silently drops edges | 2026-05-25 |
| D5 | "*_nanobind.hpp" binding files only | Also scan `*_pybind.cpp`, raw `m.def(...)`, `nb::class_<...>::def(...)`, any pybind11 holdouts | actual surface area is wider than the spec implies | 2026-05-25 |
| D6 | "@Operation" decorator only | Enumerate the real decorator surface (`@ttnn.register_python_operation`, etc.) by grepping the Python tree before encoding the bypass list | actual surface area | 2026-05-25 |
| D7 | Single big run | Start with a **vertical slice** through ONE ttnn op, end-to-end, then scale | validate schema + stitching cheaply | 2026-05-25 |
| D8 | Implicit: run anywhere | Run inside docker container; write files from host (see §0.2) | per user; protects git tree ownership | 2026-05-25 |
| D9 | Implicit: any python | Always use `/opt/venv` for runs and installs (see §0.3) | per user | 2026-05-25 |

---

## 3. Extended Schema (draft — may evolve during vertical slice)

### Node
```jsonc
{
  "id": "<fully-qualified symbol, language-prefixed>",
  "language": "python" | "cpp",
  "kind": "function" | "method" | "free_function" | "template" | "class" | "module" | "binding",
  "file": "/workspace/...",
  "line_start": int,
  "line_end": int,
  "signature": "<canonical signature string>",
  "qualifiers": ["static" | "virtual" | "constexpr" | ...],
  "template_params": [...],            // C++ only, optional
  "decorators": [...],                 // Python only, optional
  "is_binding_target": bool,           // C++ symbol exposed via nanobind/pybind
  "is_binding_caller": bool            // Python symbol that crosses into C++
}
```

### Edge
```jsonc
{
  "from": "<node id>",
  "to":   "<node id>",
  "kind": "calls" | "binds" | "instantiates" | "inherits" | "imports",
  "site_file": "/workspace/...",
  "site_line": int,
  "crosses_language": bool,
  "via_decorator": "@..." | null
}
```

---

## 4. Vertical Slice Plan (current focus)

Goal: produce a small but real `dependency_graph.json` covering ONE ttnn op end-to-end before scaling. Validate schema and the cross-language stitcher on something a human can eyeball.

Candidate slice: **`ttnn.add`** or another simple eltwise op (final choice in Step 1).

### Steps
1. **Identify the op.** Locate its Python entry point, the registration call site, the bound C++ symbol, and the TUs containing both implementation and binding.
2. **CMake configure inside the container** to emit `/workspace/build/compile_commands.json` with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.
3. **Run codegen targets** so generated headers exist (exact target list TBD — likely FlatBuffers, version header, jit configs).
4. **Write `dep-graph/scripts/cpp_index.py`** (host-written, container-run): libclang-driven; given a TU list, emits nodes + intra-C++ call edges + a nanobind/pybind binding-map (Python string → C++ symbol).
5. **Write `dep-graph/scripts/py_index.py`** (host-written, container-run): walks the Python entry-point and its imports; emits nodes + call edges; unwraps the decorator list from §2 D6.
6. **Write `dep-graph/scripts/stitch.py`** (host-written, container-run): merges both halves and rewires Python→binding leaves to their C++ targets.
7. **Hand-validate the JSON** for the slice. Fix schema / visitor logic until it looks right.
8. Only then expand: ttnn → tt_metal host → full repo (within Section 1 scope).

### Open questions (resolve as we go)
- Q1: Virtual-dispatch and template-instantiation edges — v1 or v2?
- Q2: Resolve overloads to specific instantiations, or keep them as overload sets?
- Q3: `.cpmcache/` symbols — opaque leaves or omit entirely?
- Q4: How aggressive is the partial build needed for D4? (Codegen-only target list TBD.)

---

## 5. Original Spec (preserved verbatim)

> The text below is the original `opus-instructions.md` content as delivered. Any divergence from it is recorded in §2. The trailing venv note added by the user is preserved verbatim at the end.

### 5.1 Project Context & High-Level Objective
We need to construct a complete, function-level Static Dependency Call Graph across a massive codebase (~100k+ lines) containing both Python and C++ source code.
* **The Project:** tt-metal (Tenstorrent's user-friendly ttnn operator layer and low-level tt-metalium kernel infrastructure).
* **The Bridge:** nanobind is utilized to expose C++ types and methods to the Python ecosystem.
* **Primary Use Case:** Enabling "Graph RAG" and strict structural impact lookups for LLM developer agents operating on this repository so they understand upstream/downstream blast radiuses across the language boundary without running code.

### 5.2 Core Architectural Roadblocks & Requirements

#### Constraint A: Generating the C++ Semantic Blueprint
tt-metal relies on heavy C++ template metaprogramming, macros, and modern structural concepts (DeviceOperationConcept). You cannot parse the C++ side using lightweight syntax parsers like standard Tree-sitter or Doxygen; they cannot resolve template instantiations or macro expansion.
* **Requirement:** Configure the repository's CMake build system to export a full compilation database (compile_commands.json) using `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. Use Clang's official frontend via libclang Python bindings to walk the true C++ Abstract Syntax Tree (AST) using semantic cursors.

#### Constraint B: Decoding the nanobind Bridge
The ttnn subsystem registers operations using custom registration headers (typically named `*_nanobind.hpp` within `ttnn/cpp/ttnn/operations/`).
* **The Pattern:** Operations are registered via patterns like `ttnn::register_operation` and bound to Python using calls like `ttnn::bind_registered_operation` or standard nanobind module syntax (`.def("python_method_name", &CppNamespace::FunctionName)`).
* **Requirement:** The C++ indexer must explicitly scan the AST for these binding entry points, parse the string literal representing the exposed Python name, and cross-reference it with the underlying fully qualified C++ function pointer or symbol address.

#### Constraint C: Python Decorator Unwrapping
tt-metal wraps core functions using performance-tracking and graph-capturing Python decorators (such as `@Operation`). Standard call-graph extractors will mistake the decorator logic for the leaf target, bottlenecking the entire call tree.
* **Requirement:** When parsing the Python AST, inspect the `decorator_list` of any `FunctionDef`. If a target hardware decorator is present, bypass it and map calls directly to the inner function body.

### 5.3 Step-by-Step Implementation Roadmap

#### Phase 1: Environment & Artifact Generation
1. Initialize the build environment and generate `compile_commands.json` via CMake. Ensure the library compiles far enough to output this artifact.
2. Run `python -m nanobind.stubgen` against the compiled tt-metal extension to generate standard Python type stubs (`.pyi` files). Use these to map the structural signatures of the external binary module.

#### Phase 2: C++ AST Indexing (libclang)
Write a Python script using libclang bindings to:
1. Traverse the translation units defined in `compile_commands.json`.
2. Map internal C++ function-to-function calls (`CursorKind.CALL_EXPR` linked to `CursorKind.FUNCTION_DECL`).
3. Locate the nanobind binding files and extract a strict translation dictionary matching the Python method string to the structural C++ symbol name.

#### Phase 3: Python AST Indexing
Write a companion Python script to:
1. Walk the Python source code directories.
2. Parse files into an AST, resolving function definitions and method invocations.
3. Account for decorator wrapping, exposing the underlying function targets.

#### Phase 4: Graph Stitching & Serialization
1. Read the Python call-tree, the C++ call-tree, and the nanobind translation dictionary.
2. Wherever a Python call targets a nanobind module function, swap that leaf node with an edge targeting the root of the mapped C++ symbol.
3. Export the final global dependency graph into a cleanly structured schema (such as a hierarchical NetworkX JSON file or an edge-list CSV).

### 5.4 Expected Final Outputs (per original spec — superseded by §3 schema)
* **extract_graph.py**: Unified, production-grade automation script containing both the Python AST parser and the libclang AST parser.
* **dependency_graph.json**: Final comprehensive multi-language call graph. Every node should track: `{"id": "symbol_name", "language": "python|cpp", "file": "path/to/file", "line": 42}`.

### 5.5 Trailing user note (verbatim)
> note: you must always be in the python virtual environment in /opt/venv in the docker container to run any build or compilation commands, and any custom libraries you install must go into that virtual environment.

---

## 6. Vertical Slice Findings — `ttnn.add`

### 6.1 Cross-language surface

**Python entry point:** `ttnn.add` (no Python-side `def`; the symbol is bound directly from the C++ extension `ttnn._ttnn`). The only Python source that references it explicitly is `ttnn/ttnn/operations/binary.py:41`:
```python
ttnn.attach_golden_function(ttnn.add, golden_function=_golden_function)
```
**Caveat for the slice:** `add` is *not* wrapped by `@ttnn.register_python_operation`, so this slice does **not** exercise the decorator-unwrap path. Pick a second op (e.g., `ttnn.from_torch` at `ttnn/ttnn/operations/core.py:241`) when validating Task #3.

**C++ binding call site:** `ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp:1680-1687`
```cpp
detail::bind_binary_operation<"add">(
    m,
    "Adds ...",                                   // description
    "...",                                        // math
    static_cast<detail::BinaryOpTensorScalarFn>(&ttnn::add),
    static_cast<detail::BinaryOpTensorTensorFn>(&ttnn::add));
```

**Binding helper:** `detail::bind_binary_operation<unique_string Name, …>` at `binary_nanobind.cpp:264-349` forwards to `ttnn::bind_function<Name>(mod, doc, overload_t(...), overload_t(...))`.

**Generic binding template:** `ttnn::bind_function<unique_string FuncName, unique_string Namespace = "ttnn.", …>` at `ttnn/cpp/ttnn-nanobind/bind_function.hpp:115-…`. It:
- Builds the Python fully-qualified name as `Namespace + FuncName` (e.g., `"ttnn." + "add"`).
- Defines a unique wrapper class `<FuncName>_t` and binds `__call__` for each overload (line 66: `cls.def("__call__", method, args..., nb::call_guard<nb::gil_scoped_release>())`).
- This is the actual `.def(...)` site that nanobind sees.

**C++ implementation:** `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:963-964` (in namespace `ttnn`):
```cpp
TTNN_BINARY_OP_TENSOR_TENSOR_IMPL(add, ADD)   // Tensor add(Tensor, Tensor, …)
TTNN_BINARY_OP_TENSOR_FLOAT_IMPL(add, ADD)    // Tensor add(Tensor, float, …)
```
Both expansions ultimately call `ttnn::detail::invoke_binary_ng(lhs, rhs, BinaryOpType::ADD, …)`. The declarations are produced by the matching `TTNN_BINARY_OP_*` macros at `binary.hpp:16-41` and `:197-198`.

### 6.2 What this slice exercises

| Pipeline feature | Exercised by `ttnn.add`? |
|---|---|
| Template non-type string parameter → Python name | yes (`bind_binary_operation<"add">`) |
| Macro-expanded C++ function definitions | yes (`TTNN_BINARY_OP_*_IMPL`) |
| Overload sets bound under one Python name | yes (Tensor-Tensor + Tensor-float) |
| Multi-level binding (op-specific helper → generic `bind_function` → `cls.def`) | yes |
| Python `@register_python_operation` decorator unwrap | **no** — needs a second op |
| `attach_golden_function`-style cross-language reference | yes |

### 6.3 TUs the C++ indexer must process for the slice

- `ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp` (binding)
- `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp` (impl)
- Transitively pulled headers including `binary.hpp`, `bind_function.hpp`, `binary_op_types.hpp`, etc.

Compile commands for these TUs come from `compile_commands.json` (Task #2).

---

## 7. Build / Parse Notes (added during Task #2)

- **Build dir:** `/workspace/build_Release/` (host: `/home/ubuntu/tt-metal/build_Release/`). `build/` is a symlink to it. Both are gitignored.
- **Cache state:** existing CMakeCache had `CMAKE_EXPORT_COMPILE_COMMANDS=OFF`. Reconfigured with `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .` inside the container (took ~1.5s for the generate stage thanks to the cache).
- **compile_commands.json:** 2420 entries after reconfigure. Both slice TUs (`binary.cpp`, `binary_nanobind.cpp`) are present.
- **Codegen targets:** *not* needed for the slice. The configure step alone produces all generated headers reachable from the slice TUs. (Deviation **D4** turned out to be over-cautious for the slice — we did not need to build any codegen targets. Keep D4 as a guarantee for future scope expansion: if a new TU reports `'foo/bar.h' file not found`, we add a codegen target build before re-running.)
- **libclang:** Python bindings installed via `uv pip install libclang` into `/opt/venv` (libclang 18.1.1, bundled .so at `/opt/venv/lib/python3.10/site-packages/clang/native/libclang.so`). System libclang-20.so.1 also available at `/usr/lib/x86_64-linux-gnu/libclang-20.so.1`; the indexer pins this when present (closer match to the compiler).
- **PCH conflict:** the build uses precompiled headers. Compile commands include `-Xclang -include-pch -Xclang <…>.pch` plus a re-include of the PCH header. libclang 18 cannot consume a clang-20 PCH, so these args must be stripped. See `dep-graph/scripts/probe_parse.py` for the stripping logic — it also drops `-c`, `-o <file>`, `-Winvalid-pch`. After stripping, both slice TUs parse with 0 diagnostics.
- **-Werror noise:** unrelated `-Werror`-escalated warnings (unused fields/functions) surfaced as errors initially. Stripping `-Werror` and `-pedantic-errors` from the argv is the simplest fix; we still get full ASTs.

## 8. Plan Adjustments After Task #2

- **Defer Task #3 (decorator enum)**: `ttnn.add` does not go through any Python decorator, so the slice can complete end-to-end without decorator unwrap logic. Decorator handling will be added on a second slice (`ttnn.from_torch` at `ttnn/ttnn/operations/core.py:241`) once the core machinery is validated.
- The probe script `dep-graph/scripts/probe_parse.py` is the foundation for the real `cpp_index.py` — it already handles PCH/Werror stripping and TU-arg extraction.

## 9. Vertical-Slice Result

Slice produced `dep-graph/out/dependency_graph.json` (~2.7 MB).

Totals (slice scope: `binary.cpp` + `binary_nanobind.cpp` + `binary.py`):
- 2578 nodes
- 959 edges (894 intra-C++ calls, 0 intra-Python, **65 cross-language**)
- 78 nanobind binding records
- 0 parse diagnostics
- 216 unresolved Python refs (correctly unresolved — `attach_golden_function`, `torch.is_tensor`, etc., are not in the binding table)

**Hand-validated trace for `ttnn.add`:**
```
py:module:ttnn.operations.binary
    └─ binds → cpp:ttnn::add(Tensor, Tensor, …)    [via bind_binary_operation, binary.py:41]
       └─ calls → cpp:ttnn::detail::invoke_binary_ng [binary.cpp:963]
    └─ binds → cpp:ttnn::add(Tensor, float, …)     [via bind_binary_operation, binary.py:41]
       └─ calls → cpp:ttnn::detail::invoke_binary_ng [binary.cpp:964]
```
Two overload-distinguished `ttnn::add` nodes, each pointing at the right downstream symbol. Cross-language link correctly resolves the templated `bind_binary_operation<"add">` call.

`invoke_binary_ng`'s further callees fan out into TUs we did not index (`binary_ng/…`), so its outgoing edges currently point to nodes not present in the slice. This is the expected behavior at slice scope — adding more TUs will resolve those leaves.

## 10. Tools Delivered

| Path | Purpose |
|---|---|
| `dep-graph/scripts/probe_parse.py` | Single-TU libclang parse probe (debug / argv prune validation) |
| `dep-graph/scripts/cpp_index.py` | libclang C++ indexer — emits nodes, intra-C++ call edges, and nanobind/pybind binding map |
| `dep-graph/scripts/py_index.py` | Python AST indexer — emits nodes, decorators, and cross-language candidate refs |
| `dep-graph/scripts/stitch.py` | Merges C++ + Python indexes; resolves Python `ttnn.<name>` refs against the binding map |
| `dep-graph/cache/cpp_index.json` | C++ slice index (regenerable) |
| `dep-graph/cache/py_index.json` | Python slice index (regenerable) |
| `dep-graph/out/dependency_graph.json` | Final stitched slice graph |

End-to-end runner (executed inside the container as uid 1000):
```sh
sudo docker exec --user 1000:1000 tt-metal-basic-dev-container bash -c "
  source /opt/venv/bin/activate &&
  python /workspace/dep-graph/scripts/cpp_index.py \
      --db /workspace/build_Release/compile_commands.json \
      --tu /workspace/ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp \
      --tu /workspace/ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp \
      --out /workspace/dep-graph/cache/cpp_index.json &&
  python /workspace/dep-graph/scripts/py_index.py \
      --file /workspace/ttnn/ttnn/operations/binary.py \
      --module-root /workspace/ttnn \
      --out /workspace/dep-graph/cache/py_index.json &&
  python /workspace/dep-graph/scripts/stitch.py \
      --cpp /workspace/dep-graph/cache/cpp_index.json \
      --py /workspace/dep-graph/cache/py_index.json \
      --out /workspace/dep-graph/out/dependency_graph.json
"
```

## 11. Phase 2 Plan — Repo-Wide Queryable Graph

**Goal:** ship a host-code dependency graph for the full §1 in-scope set, queryable interactively from a developer/agent workflow.

**Driving user decisions (2026-05-25):**
- Consumption pattern is **interactive query** → SQLite output is required, not optional.
- `tt_metal/jit_build/` and `tt_metal/impl/` are **in scope** (see §1).
- LLK, firmware, SFPI, `.github/`, tests stay out of scope (see §1).

### Milestone P2-A: Parallelize cpp_index

Serial libclang over ~2,400 in-scope TUs is hours of wall time. This is the lever everything else depends on.

Approach:
- Split `cpp_index.py` into:
  - `cpp_index_worker.py` — parses **one** TU, writes a JSONL shard per record-type (`nodes.jsonl`, `edges.jsonl`, `bindings.jsonl`, `diagnostics.jsonl`) under `dep-graph/cache/tu_shards/<tu-hash>/`.
  - `cpp_index_driver.py` — discovers in-scope TUs from `compile_commands.json`, applies §1 scope filter, fans workers out (multiprocessing.Pool, `--max-workers N` flag with sensible default).
  - `cpp_index_merger.py` — folds shards into the master index, USR-keyed dedup, emits a single `cpp_index.jsonl` per record-type.
- Content-hash cache: `hash = sha256(tu_path + canonical_argv + tu_mtime + included-header mtimes)`. If a shard for that hash exists, skip re-parse. Maintain `dep-graph/cache/tu_shards/manifest.json` mapping TU → hash → shard dir.
- TU errors (parse fail, missing header) get logged to a global diagnostics file, do not abort the run.

Acceptance:
- Full repo run under 30 min on a 16-core workstation.
- Incremental (1 source change) under 30s.
- No worker holds more than ~2GB resident.

Risks: libclang `Index` is not picklable → each worker creates its own; peak memory with many workers can be uncomfortable; lock contention on the shared manifest if not written carefully.

### Milestone P2-B: SQLite output for interactive queries

JSON is unworkable for "find all callers of X" at full-repo scale. SQLite with the right indexes is.

Schema (draft):
```sql
CREATE TABLE nodes (
    id          TEXT PRIMARY KEY,
    language    TEXT NOT NULL,           -- 'cpp' | 'python'
    kind        TEXT NOT NULL,
    name        TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    file        TEXT NOT NULL,
    line_start  INTEGER NOT NULL,
    line_end    INTEGER NOT NULL,
    signature   TEXT,
    is_definition INTEGER NOT NULL,
    is_binding_target INTEGER NOT NULL DEFAULT 0,
    is_binding_caller INTEGER NOT NULL DEFAULT 0,
    attrs_json  TEXT
);
CREATE INDEX idx_nodes_name  ON nodes(name);
CREATE INDEX idx_nodes_qname ON nodes(qualified_name);
CREATE INDEX idx_nodes_file  ON nodes(file);

CREATE TABLE edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    src         TEXT NOT NULL,
    dst         TEXT NOT NULL,
    kind        TEXT NOT NULL,            -- 'calls' | 'binds' | 'instantiates' | 'inherits' | 'imports'
    site_file   TEXT NOT NULL,
    site_line   INTEGER NOT NULL,
    crosses_language INTEGER NOT NULL DEFAULT 0,
    via_decorator TEXT,
    via_helper    TEXT
);
CREATE INDEX idx_edges_src  ON edges(src);
CREATE INDEX idx_edges_dst  ON edges(dst);   -- "callers of X"
CREATE INDEX idx_edges_kind ON edges(kind);

CREATE TABLE bindings (
    python_name TEXT NOT NULL,
    cpp_node_id TEXT NOT NULL,
    cpp_qualified_name TEXT NOT NULL,
    site_file   TEXT NOT NULL,
    site_line   INTEGER NOT NULL,
    helper      TEXT
);
CREATE INDEX idx_bindings_pyname ON bindings(python_name);
CREATE INDEX idx_bindings_cppid  ON bindings(cpp_node_id);
```
- `stitch.py` either gains a SQLite writer or is replaced by `stitch_sqlite.py`. Output: `dep-graph/out/dep-graph.sqlite`.
- Add `dep-graph/scripts/query.py` — small CLI with subcommands:
  - `query.py callers <symbol>` — all incoming edges
  - `query.py callees <symbol>` — all outgoing edges
  - `query.py blast-radius <symbol> --depth N` — BFS in both directions
  - `query.py by-file <path>` — every node defined in a file
  - `query.py crosses-lang <symbol>` — every cross-language edge touching a symbol
- Keep JSONL output as the canonical *cache* form. SQLite is a derived index.

Acceptance:
- `query.py callers ttnn::add` returns in <100ms on the full graph.
- `query.py blast-radius ttnn::detail::invoke_binary_ng --depth 2` returns within seconds.

### Milestone P2-C: Decorator unwrap (deferred Task #3 picked up)

`@ttnn.register_python_operation` is the dominant decorator on the ttnn surface. The current py_index records it as metadata but doesn't rewire calls.

Approach:
- Grep `ttnn/ttnn/` to enumerate the actual decorator surface. Known candidates so far: `@ttnn.register_python_operation`, `ttnn.attach_golden_function(...)` (module-level call, not a decorator), `ttnn.register_python_operation(name=...)(impl_function)` (call-form registration). Record the canonical list here in §2 D6 once enumerated.
- In `py_index.py`:
  - When a `FunctionDef` carries a recognized wrap decorator, the inner-function node still gets emitted but its **incoming edges from callers** route through a `via_decorator=<decorator_label>` attribute. Callees of the inner function should be visible.
  - For module-level registration calls (`ttnn.register_python_operation(name="ttnn.X")(impl)`), emit a `binds`-kind edge from `impl` → the C++ symbol for `"ttnn.X"` if the stitcher's binding lookup finds one. Mark the edge with `via_decorator`.
- Validate on `ttnn.from_torch` (`ttnn/ttnn/operations/core.py:241`) and `ttnn.to_torch` (`:379`).

Acceptance:
- `query.py callers ttnn.from_torch` returns callers of the *inner* function, not the decorator wrapper.
- An edge exists from the impl of `ttnn.from_torch` → the C++ symbol it ultimately binds (when applicable), with `via_decorator` set.

### Milestone P2-D: Validation harness

Without spot checks, regressions ship silently and are noticed only when a query returns garbage.

Approach:
- Pick ~10 ops spanning the surface (eltwise binary, eltwise unary, reduction, matmul, ccl, normalization, conv2d, data_movement, embedding, transformer/sdpa).
- Author `dep-graph/tests/expected_chains.yaml`, e.g.:
  ```yaml
  - name: ttnn.add
    chains:
      - ["py:module:ttnn.operations.binary", "binds", "ttnn::add"]
      - ["ttnn::add", "calls", "ttnn::detail::invoke_binary_ng"]
  ```
- `dep-graph/scripts/validate.py` loads the SQLite DB, runs each chain as a SQL query, fails non-zero on any miss. Output a concise pass/fail report.
- Run validate.py automatically at end of every full re-index.

Acceptance:
- All 10 chains pass on a clean run.
- A deliberate breakage (e.g., commenting out a `bind_*` call) makes the right chain assertion fail loudly.

### Milestone P2-E (stretch): Schema enrichment

Add the missing edge kinds. Defer if A–D run long.

- `instantiates`: record template specialization → primary template edges via libclang's `Cursor.get_specialization`.
- `inherits`: walk `CXX_BASE_SPECIFIER` when visiting class declarations.
- `imports`: in py_index, walk `Import` / `ImportFrom` and emit module-level edges.

Acceptance: §3 schema doc updated; per-kind counts visible in the manifest.

### Sequencing

| Order | Milestone | Reason for position |
|---|---|---|
| 1 | P2-A Parallelize | Nothing downstream is testable at repo scale without it |
| 2 | P2-B SQLite | Required for interactive query (user direction); design at the same time as A so shard format and DB format are consistent |
| 3 | P2-C Decorators | Independent of A/B mechanically, but pointless to ship without D's validation |
| 4 | P2-D Validation | Once A/B/C are in place, lock down correctness |
| 5 | P2-E Enrichment | Stretch only |

I'd treat **A+B together as one work block** since they share shard/DB format decisions, then C, then D, then E if there's time.

## 12. Progress Log

| Date | Note |
|------|------|
| 2026-05-25 | Container verified; toolchain confirmed. Memory + canonical-spec persistence set up. |
| 2026-05-25 | Task #1 complete: located the `ttnn.add` surface end-to-end (Python entry, nanobind binding via templated unique_string, C++ impl via macro). See §6. |
| 2026-05-25 | Task #2 complete: reconfigured CMake with compile-commands export (2420 TUs). libclang installed and verified parsing both slice TUs after PCH-strip. See §7. |
| 2026-05-25 | Tasks #4–#7 complete: wrote cpp_index.py, py_index.py, stitch.py; produced a hand-validated end-to-end slice graph for `ttnn.add`. See §9–§10. Task #3 (decorator enum) deferred — picked up in P2-C. |
| 2026-05-25 | Scope refined: `tt_metal/jit_build/` and `tt_metal/impl/` now in scope; LLK / firmware / `.github/` confirmed out. Phase 2 plan recorded in §11 — interactive SQLite query is now a hard requirement. |
| 2026-05-25 | P2-A draft (driver + worker + merger) written. First 605-TU partial run uncovered two bugs: (a) over-eager binding extraction picking up `nb::arg`/`nb::none`/lambda-body refs; (b) libclang 18.1.1 raises `ValueError` on clang-20 CursorKind 155, aborting whole TUs. Fixes landed: (a) namespace + lambda-subtree filter in `_find_referenced_functions`; (b) `_safe_kind`/`_iter_children` guards in `_walk`. 250-TU curated validation (incl. all 8 previously-failed TUs + every `*_nanobind.cpp`): **250/250 succeeded, 0 failures**; binding noise cut from 781 → 489; `__init__` factory bindings now clean. |
| 2026-05-25 | P2-B (SQLite stitcher + query.py) wired up. Schema lives in §11. Cross-language resolution now goes through a `py_registrations` table first, then falls back to C++ bindings. Output DB at `dep-graph/out/dep-graph.sqlite` (~120 MB on the validation subset). |
| 2026-05-25 | P2-C (decorator unwrap) landed: `py_index.py` detects `@ttnn.register_python_operation(name="ttnn.X")` decorators AND call-form `ttnn.register_python_operation(name="ttnn.X")(impl)` at module scope. Emits new `registrations` record. 17 registrations detected across `ttnn/ttnn/`: 7 decorator-form (Python impls including `from_torch`, `to_torch`, `as_tensor`, `dump_tensor`, `load_tensor`, `pearson_correlation_coefficient`, `Tensor.__getitem__`) and 10 call-form (C++ pass-throughs like `unsqueeze_to_4D`, `deallocate`, etc.). After stitch: `ttnn.from_torch` correctly resolves to its Python impl (51 callers), `ttnn.to_torch` (43 callers), all `via_decorator=@ttnn.register_python_operation`. |
| 2026-05-25 | P2-D precursor: wrote `dep-graph/tests/expected_chains.yaml` with 11 chains and `dep-graph/scripts/validate.py`. Covers positive (cross-language slice, decorator unwrap, call-form pass-through, overload sets) and negative (Bug A regressions: no `nanobind::*` / `std::*` / `__builtin_*` in bindings). **11/11 chains pass.** Validation harness is now the gate before launching the full P2-A run. |
