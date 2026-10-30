# HAL explorer

Compiles HAL code and lists, in issue order, the Tensix instructions and MMIO accesses it
produces. Use it to size `hal::replay` buffers from the compiler instead of counting by
hand, and to check that a HAL rewrite of an existing LLK kernel issues exactly what the
original does. It comes as a command-line tool and as a local page; both give the same
results.

The tool needs only the Python standard library and the SFPI toolchain in `tests/sfpi`.
It compiles against the headers of the checkout it lives in, from any working directory.
The HAL headers it includes (and the namespace aliases of body mode) are the ones present
in `tt_llk_blackhole/llk_lib/hal/`, so it also works on branches lower in the HAL stack.

## Command line

```
python3 tools/python/hal_explorer/cli.py FILE|- [--mode code|body] [--var "TYPE NAME"]...
                                         [--function NAME] [--counts] [--gimple] [--json]
```

| Option | Meaning |
|---|---|
| `FILE` or `-` | Source to compile; `-` reads stdin. |
| `--mode code` | Default. The input is top-level C++, compiled as written with the HAL headers included. Every function defined in it is analysed. |
| `--mode body` | The input is the body of a `replay::record` lambda. The namespaces `atomic`, `cfg`, `adc`, `rwc`, `counters`, `misc`, `mop`, `replay`, `scalar` and `sync` are aliased and `ckernel` is visible. |
| `--var "TYPE NAME"` | Body mode only: a runtime value the body reads, e.g. `--var "bool partial_face"`. Repeat for each one. In code mode, use function parameters. |
| `--function NAME` | Print only this function (name or signature) and its comparisons. |
| `--counts` | One tab-separated row per path: `signature`, `count`, `conditions`. Comparisons go to stderr. |
| `--gimple` | Also print the compiler output the result was read from. |
| `--json` | Print the full result as JSON (schema below), the same data the page receives. |

| Exit status | Meaning |
|---|---|
| 0 | Complete result; every comparison identical. |
| 1 | Compile error (diagnostics on stderr, structured in `errors` with `--json`). |
| 2 | A comparison differs, or a `reference_<name>` has no `<name>` with the same parameter types. |
| 3 | Invalid arguments or unreadable input. |
| 4 | The result is incomplete: a warning says what is not counted (an un-inlined call, SFPI code, an unrecognised statement, the path limit) or a comparison is inconclusive. |

### Output

```
== unpack_matmul_advance_a(const bool): 2 path(s), 2 distinct sequence(s)
  -- count 14  when partial_face  fingerprint 9b6538cbd6ec
       0  REPLAY       0x040000c1       [store] start_idx=0 len=12 execute_while_loading=0 load_mode=1  <- record (tests/sfpi/include/lltt.h:28)
       1  SETADCZW     0x54200005       [imm] CntSetMask=1 ... BitMask=5  <- emit (hal/counters/adc.h:503)
      ...
== compare pack_rows_init(const uint32_t) vs reference_pack_rows_init(const uint32_t): DIFFERS
  -- counterexample: num_rows=0
     MMIO write 32b TENSIX_MOP_CFG_BASE + 0xc = 0x2000000
    -MMIO write 32b TENSIX_MOP_CFG_BASE + 0x10 = 0x2000000
```

- Each function has one block per distinct issued sequence. `count` is the number of
  Tensix instructions; `when` lists every branch condition under which that sequence is
  issued, separated by `|`.
- An instruction line shows its position, mnemonic, 32-bit word, how it reached the
  Tensix (`imm`: `.ttinsn` immediate; `store`: word built at runtime and stored to
  `__instrn_buffer`; `builtin`), its fields, and the HAL function and line that issued it
  (`input:N` for a line of the input). A field marked `rt` depends on a runtime value,
  given as `rt=(expression)`.
- `MMIO` lines are volatile loads and stores to memory-mapped registers (configuration
  space, MOP config, `pc_buf_base`, ...) and blocking-store asm. They issue no Tensix
  instruction and are not counted.
- A loop whose trip count is only known at runtime prints as `loop, repeats while COND`.
  Its count is fitted from concrete runs, e.g. `1 + n  (fitted on 21 samples; ...)`, and
  the samples are listed. A loop that issues the same words on every run is shown
  unrolled with those exact words.
- `!!` lines are warnings: something that may issue instructions was not counted.

### Comparing a rewrite with the original

Define the original as `reference_<name>` and the rewrite as `<name>` with the same
parameter types:

```cpp
#include "llk_pack_rows.h"

void reference_pack_rows_init(const std::uint32_t num_rows) { _llk_pack_rows_init_(num_rows); }

void pack_rows_init(const std::uint32_t num_rows)
{
    // HAL rewrite
}
```

Both functions are run on the same concrete inputs: combinations of boundary values of
the parameters (matched by position, so names may differ) and of the globals they read
(`cfg_state_id` is 0 or 1), up to 256 input vectors. Each vector runs twice: once with
every MMIO read returning zero, once with the k-th read returning a fixed derived value.
Values loaded through pointer parameters are derived from the parameter position and
offset, so two functions that read the same memory see the same values. The first run on
which they issue anything different (instruction words, MMIO targets, widths or values)
is reported with a diff. How a word reached the Tensix (`TTI_` immediate or `TT_` store) is not a difference.
A `reference_` without a partner is reported as unpaired. Names in namespaces are paired
within the namespace.

### Typical uses

Count what a replay body issues, to choose its `BufferRange`:

```
printf '%s\n' \
  'if (partial_face) { counters::set<adc::AddrCounter {.channel0 = {.z = 0}}.for_clients<adc::Client::Unpacker0>()>(); }' \
  'scalar::add(hal::gpr<p_gpr_unpack::TILE_OFFSET>(), hal::gpr<p_gpr_unpack::TILE_SIZE>(), hal::gpr<p_gpr_unpack::TILE_OFFSET>());' |
  python3 tools/python/hal_explorer/cli.py - --mode body --var "bool partial_face" --counts
```

Check a rewrite: `python3 tools/python/hal_explorer/cli.py rewrite.cpp`; exit status 0
means identical on every input, 2 means a counterexample was printed.

### JSON

```
Result      {ok, mode, diagnostics, errors: [{file, line, column, message}],
             functions: [Function], comparisons: [Comparison]}
Function    {name, signature, mangled, path_count, truncated, groups: [Group],
             samples: [Sample], warnings: [{message, origin}], gimple: [{text, mark}]}
Group       {fingerprint, count (null with a runtime loop), count_text, when: [[condition]],
             sequence: [Item]}
Item        {kind: "insn", mnemonic, word, word_text, runtime_mask, runtime_expr,
             issue: "immediate" | "store" | "builtin", fields: [{name, value, runtime}],
             origin, replay: {load, start, count, count_text} | null}
          | {kind: "mmio", operation: "write" | "read" | "asm", width, address, target,
             value, value_text, origin}
          | {kind: "loop", repeat_while, body: [Item], variants: [{when, body}]}
Sample      {inputs, count, fingerprint, complete, note}
Comparison  {name, reference, verdict: "identical" | "differs" | "inconclusive" | "unpaired",
             samples, reasons: [string], counterexample: {inputs, diff} | null}
origin      {file, line, text, function, in_input} | null
```

In body mode the single function is named `replay body`.

## Local page

```
python3 tools/python/hal_explorer/server.py [--port 8765] [--host 127.0.0.1] [--jobs 4]
```

Open `http://localhost:8765/` (forward the port when working on a remote machine). The
page has the two modes of the CLI: **Replay body**, with runtime variables and the replay
slot of every instruction, and **HAL code**, with function tabs, comparison verdicts and
a **Load example** picker for the kernels in `examples/`. A collapsed section shows the
compiler output with every counted statement marked.

The server compiles whatever it receives. It binds to localhost, accepts only JSON posted
with a local `Host` header, and limits request size and compile time, but an input can
still `#include` and so reveal any readable file in its diagnostics: do not expose it.

## How it works

1. **Compile** (`toolchain/build.py`). The input is compiled with the flags of the Blackhole HAL lit
   tests (`tools/tests/unit/lit/hal/blackhole/lit.local.cfg`; a test keeps them in sync)
   plus `-fdump-tree-optimized-lineno-blocks`: every function after all inlining and
   constant folding, with a source location on every statement and the explicit
   successors of every basic block.
2. **Parse** (`gimple/reader.py`). The dump becomes a control-flow graph. A Tensix instruction is
   a `.ttinsn` asm (compile-time word), a volatile store to `__instrn_buffer` (word built
   at runtime), or a `__builtin_rvtt_tt*` builtin such as REPLAY. Any other volatile
   access is MMIO. Calls left un-inlined, SFPI builtins and any unrecognised statement
   that could issue something become warnings. Instruction words are volatile, so later
   compiler passes can neither add, remove nor merge them on any path.
3. **Walk symbolically** (`analysis/paths.py`, `gimple/loops.py`, `gimple/expr.py`). Every path from entry to
   exit is followed with its branch conditions. Natural loops are folded into one loop
   item; values that change between iterations (loop-carried PHIs, MMIO reads) stay
   symbolic. Runtime words show their constant part and which fields depend on runtime
   values.
4. **Run concretely** (`analysis/interp.py`, `analysis/samples.py`). The same GIMPLE is executed on concrete
   inputs with C semantics for every operation. This gives exact words inside loops,
   per-path count formulas fitted from the runs, and the comparison of a rewrite with its
   reference. Branches on MMIO reads fork the run; poll loops are assumed to end.
5. **Check itself** (`analysis/function.py`). Every concrete run is matched to the symbolic path it
   took; a path whose words are all constant must issue exactly what the run issued.
6. **Decode** (`isa/instructions.py`, `isa/origin.py`). Mnemonics and fields come from the
   `TT_OP_*` macros in `ckernel_ops.h`; MMIO addresses are named from `tensix.h` and
   `cfg_defines.h`; the issuing function is the definition whose braces enclose the
   statement's source line.

```
hal_explorer/
  cli.py, server.py      entry points (also: python3 -m hal_explorer, from tools/python)
  pipeline.py            run(mode, source, tables, variables): compile, analyse, compare
  model.py               result types and their JSON
  toolchain/build.py     input wrapping, compile flags, running the compiler
  gimple/reader.py       dump parsing and statement classification
  gimple/types.py        C types on RISC-V and symbolic pointers
  gimple/expr.py         expression trees: evaluation, source text, runtime bits
  gimple/loops.py        natural loops
  analysis/paths.py      symbolic path walk
  analysis/interp.py     concrete execution
  analysis/samples.py    input vectors and count fitting
  analysis/function.py   per-function result and self-check
  analysis/compare.py    reference_<name> comparisons
  isa/instructions.py    opcode layouts and MMIO address names
  isa/events.py          what one statement issues
  isa/origin.py          source line and issuing function
  ui/render.py           command-line text
  ui/page.html           the local page
  examples/              HAL rewrites of existing kernels
  tests/                 pytest suite and the object-file oracle
```

## Limits

- Code reached through a call GCC does not inline is not visible; the call is reported.
- SFPI vector code is not counted: its builtins are not volatile and later passes fuse
  them. It is reported.
- A MOP or REPLAY run counts as the one instruction that starts it, not as the
  instructions it expands into.
- The origin is the innermost source location GCC keeps: the HAL line that issued the
  word, not the call in the input that led to it.
- Comparisons and samples use boundary values, not every possible input. Runs stop after
  4000 basic blocks; a comparison says how many inputs were cut short.
- At most 1024 paths are listed per function; comparisons are not affected.
- Blackhole only.

## Tests

```
tests/.venv/bin/python -m pytest -q tools/python/hal_explorer/tests            # about 1.5 min
tests/.venv/bin/python -m pytest -q tools/python/hal_explorer/tests -m slow    # the corpus
```

`test_parse.py` needs no compiler; the other tests use the SFPI toolchain. The `slow`
tests (`test_corpus.py`) run every example, every lit HAL codegen source and every file in
`tests/cases/` through the tool and through `tests/oracle.py`, which executes the compiled
RISC-V object and records the Tensix words it issues, and require both to agree. They
also require every `reference_` pair in the lit sources to compare identical.
