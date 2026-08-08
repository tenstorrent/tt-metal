# LLK API Analyzer

Analyze the compute kernels produced by a TTNN model / tt-metal run and report
**which LLK APIs were actually used, with what template arguments and runtime
arguments**.

For each LLK API it recovers the configuration it was instantiated with — data
formats, tile sizes, sync scheme (`DstSync`), dest / L1 accumulation
(`is_fp32_dest_acc_en`), math fidelity, broadcast type, eltwise op type, etc.

## How it works

A TTNN/metal run JIT-compiles every compute kernel into three Tensix RISC-V
ELFs (`trisc0/1/2.elf` = unpack / math / pack) under the kernel cache
(`$TT_METAL_CACHE` or `~/.cache/tt-metal-cache/`). These ELFs are built with
`-O3 -flto` and (when debug info is enabled) carry DWARF.

The analyzer reads the DWARF with [`pyelftools`] and walks the
`DW_TAG_inlined_subroutine` tree:

1. **Inlined tree ≈ runtime call graph.** All compute APIs are `ALWI`
   (always-inline), so every LLK call that survives optimization appears as an
   inlined subroutine, and dead code is eliminated. What remains is what the
   kernel runs. (Some compiled-but-not-executed code may remain — acceptable per
   the requirements.)
2. **Template args = configuration.** Each call's abstract definition carries
   `DW_TAG_template_value_param` children whose names, values and types are read
   directly (e.g. `eltwise_binary_type=ELWSUB`, `is_fp32_dest_acc_en=false`,
   `math_fidelity=LoFi`). Enum values are resolved to names via the DWARF
   enumeration tables.
3. **Runtime args.** Each call's `DW_TAG_formal_parameter` children give the
   argument names; constants the optimizer propagated are reported as static
   values, the rest are flagged dynamic (live in registers at runtime and not
   statically recoverable). At `-O3` these constants are rarely emitted as
   `DW_AT_const_value`; most are recovered by evaluating the parameter's
   constant location expression (`DW_OP_litN` / `DW_OP_constNu` …
   `DW_OP_stack_value`).
4. **Data formats / tile sizes** are additionally parsed from the generated
   `chlkc_descriptors.h` next to the ELFs, since the LLKs read those from
   per-circular-buffer arrays at runtime rather than as template args.

Classification is purely by naming convention (`_llk_*` core lib, `llk_*` API
wrapper) and DWARF source paths, so **new LLKs, ops and models are picked up
automatically** with no per-op code.

### Report format

Per kernel, calls are sorted by **API layer**, then **thread**
(unpack → math → pack), then **function name**, and grouped by API
configuration and call site. Each group shows the call site
(`file:line:column`, from `DW_AT_call_file/line/column`), how many times it is
inlined, and the distinct constant argument combinations (with how many
occurrences each), followed by any arguments that stay dynamic at runtime:

```
  -- pack / llk_api --
   `llk_pack<is_fp32_dest_acc_en=false, out_of_order_output=false, pack_mode=Default>`  (op: `pack_tile`)
      @ pack.h:90:5    (1 call, 1 distinct arg-combo)
        (x1)  tile_index={0..7}, output=4, output_tile_index=0
```

## Requirements

- The kernels must have been built with DWARF debug info:
  `export TT_METAL_RISCV_DEBUG_INFO=1` before the run.
- Python environment with [`pyelftools`] and [`tabulate`]. Should also have ttnn
  and dependencies if you want to run a model.

```bash
./create_venv.sh
source python_env/bin/activate
python -m ensurepip
python -m pip install pyelftools tabulate
```

## Usage

There are two modes: **run a model and analyze it**, or **analyze an existing
cache**. Run from the repository root.

### Mode 1 — run a model, then analyze it (`--run`)

Pass the command that runs your model/test. The tool runs it with an isolated
`TT_METAL_CACHE` and `TT_METAL_RISCV_DEBUG_INFO=1`, then analyzes the kernels it
produced.

```bash
# Run a pytest and analyze the LLK APIs it used
python_env/bin/python -m tt_metal.tools.llk_api_analyzer \
    --run "pytest tests/ttnn/.../test_add.py -k bf16"

# Run an arbitrary python model script, write JSON
python_env/bin/python -m tt_metal.tools.llk_api_analyzer \
    --run "python models/demos/.../demo.py" -f json -o report.json

# Reuse/inspect a persistent cache instead of a temp dir
python_env/bin/python -m tt_metal.tools.llk_api_analyzer \
    --run "pytest ..." --cache-dir ./my_run_cache --keep-cache
```

By default the run uses a fresh temporary cache that is deleted after analysis
(so every run starts clean and recompiles with debug info). Use `--cache-dir`
for a persistent location and `--keep-cache` to retain a temp cache.

### Mode 2 — analyze an existing cache

```bash
# Analyze a single kernel build directory
python_env/bin/python -m tt_metal.tools.llk_api_analyzer \
    ~/.cache/tt-metal-cache/<build_key>/kernels/<kernel>/<hash>

# Analyze every compute kernel from a whole run (the cache root or a build key)
python_env/bin/python -m tt_metal.tools.llk_api_analyzer ~/.cache/tt-metal-cache/<build_key>

# JSON output, written to a file
python_env/bin/python -m tt_metal.tools.llk_api_analyzer <path> -f json -o report.json

# Collapse the whole run into one flat table (Markdown or CSV)
python_env/bin/python -m tt_metal.tools.llk_api_analyzer <path> -f table
python_env/bin/python -m tt_metal.tools.llk_api_analyzer <path> -f csv -o llk_calls.csv

# Include the user-facing compute-API layer too
python_env/bin/python -m tt_metal.tools.llk_api_analyzer <path> -l llk_core,llk_api,compute_api
```

The `path` may be the cache root, a `<build_key>`, a `kernels/` directory, a
single kernel directory, or one compile-hash directory. (For an existing cache,
the kernels must have been built with `TT_METAL_RISCV_DEBUG_INFO=1`.)

### Options

| Flag | Description |
|------|-------------|
| `-r, --run COMMAND` | Shell command that runs the model; its kernels are then analyzed. |
| `--cache-dir DIR` | `TT_METAL_CACHE` to use for `--run` (default: a fresh temp dir). |
| `--run-cwd DIR` | Working directory for the `--run` command. |
| `--keep-cache` | Keep the temporary `--run` cache after analysis. |
| `-f, --format {text,json,table,csv}` | Output format (default `text`). `table`/`csv` collapse the run into one row per LLK call. |
| `-o, --output FILE` | Write to a file instead of stdout. |
| `-l, --layers L1,L2,...` | Layers to collect from `llk_core,llk_api,compute_api,other` (default `llk_api`). Use `llk_core,llk_api` to include internal `_llk_*` calls. |

### Collapsed table (`-f table` / `-f csv`)

Flattens the whole run into a single table with one row per distinct LLK call
(rows identical except for the TTNN op are merged, listing all contributing
ops). Columns:

| Column | Source |
|--------|--------|
| LLK API | Call name + template args (`name<param=value, ...>`). |
| TTNN Ops | Enclosing compute-API op(s) that use this call. |
| Op Args | The call's runtime args (constants shown as values, dynamic as `name=?`). |
| Input Data Formats | Formats of the circular buffers this call reads (`operand*` args). |
| Output Data Formats | Formats of the circular buffers this call writes (`output` / `pack_output`). |
| Tile Dims | Tile `RxC` for the circular buffers this call references (`operand*` / `output` args). |
| Math Fidelity | `MATH_FIDELITY` from `chlkc_descriptors.h`. |
| Math Approx | `APPROX`. |
| FP32 Dest Accum | `DST_ACCUM_MODE`. |
| Dst Sync Mode | `DST_SYNC_MODE`. |

Input/output CBs are taken from **this call's** static runtime arguments when
present (e.g. `operandA=5, operandB=3` → `cb3` and `cb5` only). A direction
with no static CB index on the call is shown as `-`. When the call names no CBs
at all (fully dynamic operands), the columns fall back to the kernel-wide union,
then to all configured CBs.

## Programmatic API

```python
from tt_metal.tools.llk_api_analyzer import LlkAnalyzer, ModelRunner

# Analyze an existing cache
analysis = LlkAnalyzer().analyze_run("~/.cache/tt-metal-cache/<build_key>")
for api in analysis.aggregate():
    print(api.name, [(t.name, t.display_value) for t in api.template_args])

# Run a model and analyze it
result = ModelRunner().run("pytest tests/.../test_add.py")
analysis = LlkAnalyzer().analyze_run(result.cache_dir)
ModelRunner.cleanup(result)
```

## Module layout

| Module | Responsibility |
|--------|----------------|
| `dwarf_helpers.py` | Generic DIE helpers: names, enum tables, types, source paths. |
| `extractor.py` | Walk the inlined-subroutine tree → `ApiCall` records. |
| `descriptors.py` | Parse `chlkc_descriptors.h` (formats, tile sizes, sync, accum). |
| `discovery.py` | Find compute-kernel ELF triples under a run directory. |
| `runner.py` | Run a model command with an isolated, debug-info-enabled cache. |
| `analyzer.py` | Orchestrate discovery + extraction → `RunAnalysis`. |
| `model.py` | Result dataclasses (+ JSON serialization). |
| `report.py` | Text / JSON / Markdown / CSV rendering. |
| `cli.py` | Command-line entry point. |

[`pyelftools`]: https://github.com/eliben/pyelftools
[`tabulate`]: https://github.com/astanin/python-tabulate
