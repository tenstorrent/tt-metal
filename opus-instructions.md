# SYSTEM SPECIFICATION: Static Polyglot Dependency Graph Generator for tt-metal

## 1. Project Context & High-Level Objective
We need to construct a complete, function-level Static Dependency Call Graph across a massive codebase (~100k+ lines) containing both Python and C++ source code. 
* **The Project:** tt-metal (Tenstorrent’s user-friendly ttnn operator layer and low-level tt-metalium kernel infrastructure).
* **The Bridge:** nanobind is utilized to expose C++ types and methods to the Python ecosystem.
* **Primary Use Case:** Enabling "Graph RAG" and strict structural impact lookups for LLM developer agents operating on this repository so they understand upstream/downstream blast radiuses across the language boundary without running code.

---

## 2. Core Architectural Roadblocks & Requirements

To succeed, you must bypass lightweight text-matching (Regex) and run a compilation-assisted static extraction pipeline. You must account for these three non-trivial implementation constraints:

### Constraint A: Generating the C++ Semantic Blueprint
tt-metal relies on heavy C++ template metaprogramming, macros, and modern structural concepts (DeviceOperationConcept). You cannot parse the C++ side using lightweight syntax parsers like standard Tree-sitter or Doxygen; they cannot resolve template instantiations or macro expansion.
* **Requirement:** You must configure the repository's CMake build system to export a full compilation database (compile_commands.json) using -DCMAKE_EXPORT_COMPILE_COMMANDS=ON. You will then use Clang’s official frontend via libclang Python bindings to walk the true C++ Abstract Syntax Tree (AST) using semantic cursors.

### Constraint B: Decoding the nanobind Bridge
The ttnn subsystem registers operations using custom registration headers (typically named *_nanobind.hpp within the ttnn/cpp/ttnn/operations/ folders). 
* **The Pattern:** Operations are registered via patterns like ttnn::register_operation and bound to Python using calls like ttnn::bind_registered_operation or standard nanobind module syntax (.def("python_method_name", &CppNamespace::FunctionName)).
* **Requirement:** Your C++ indexer script must explicitly scan the AST for these specific binding entry points. It must parse the string literal representing the exposed Python name and cross-reference it with the underlying fully qualified C++ function pointer or symbol address.

### Constraint C: Python Decorator Unwrapping
tt-metal heavily wraps core functions using performance-tracking and graph-capturing Python decorators (such as @Operation). Standard call graph extractors will mistake the decorator logic for the leaf target, bottlenecking the entire call tree.
* **Requirement:** When parsing the Python AST using the native ast module or libcst, your script must inspect the decorator_list of any FunctionDef. If a target hardware decorator is present, bypass it and map calls directly to the inner function body.

---

## 3. Step-by-Step Implementation Roadmap

Execute this project sequentially. Verify the output of each phase before proceeding.

### Phase 1: Environment & Artifact Generation
1. Initialize the build environment and generate compile_commands.json via CMake. Ensure the library compiles far enough to output this artifact.
2. Run "python -m nanobind.stubgen" against the compiled tt-metal extension to generate standard Python type stubs (.pyi files). Use these files to map the structural signatures of the external binary module.

### Phase 2: C++ AST Indexing (libclang)
Write a robust Python script using libclang bindings to:
1. Traverse the translation units defined in compile_commands.json.
2. Map internal C++ function-to-function calls (CursorKind.CALL_EXPR linked to CursorKind.FUNCTION_DECL).
3. Locate the nanobind binding files and extract a strict translation dictionary matching the Python method string to the structural C++ symbol name.

### Phase 3: Python AST Indexing
Write a companion python script to:
1. Walk the Python source code directories.
2. Parse files into an AST, resolving function definitions and method invocations.
3. Account for decorator wrapping, exposing the underlying function targets.

### Phase 4: Graph Stitching & Serialization
1. Read the Python call-tree, the C++ call-tree, and the nanobind translation dictionary.
2. Wherever a Python call targets a nanobind module function, swap that leaf node with an edge targeting the root of the mapped C++ symbol.
3. Export the final global dependency graph into a cleanly structured schema (such as a hierarchical NetworkX JSON file or an edge-list CSV).

---

## 4. Expected Final Outputs
* **extract_graph.py**: A unified, production-grade automation script containing both the Python AST parser and the libclang AST parser.
* **dependency_graph.json**: The final comprehensive multi-language call graph. Every node should clearly track attributes: {"id": "symbol_name", "language": "python|cpp", "file": "path/to/file", "line": 42}.

Let’s begin with Phase 1. Inspect the repository root, determine how the CMake configuration is structured, and execute the command to generate compile_commands.json. Show me your analysis of the build architecture before running the compilation.

note: you must always be in the python virtual environment in /opt/venv in the docker container to run any build or compiliation commands, and any custom libraries you install must go into that virtual environment.