# Blackhole state-hazard audit: schema and plugin contract

Package `tools.llk_state_audit.bh` (`--arch blackhole`). It is static, source-only and
deterministic. It needs no git, no compiler and no libclang. Run it from `tt_metal/tt-llk`:

```sh
python3 -m tools.llk_state_audit bh-effects --repo-root <tree> --out effects.json
python3 -m tools.llk_state_audit bh-hazards --repo-root <tree> --out findings.json
#   optional: --families sfpu,pack   (load only these bh/families modules)
```

`<tree>` is a tt-metal checkout root (`tt_metal/`, `ttnn/`, `models/`, `tests/tt_metal/`).

| Module | Owner | Role |
|---|---|---|
| `scan.py` | foundation | C++ function/macro/call extractor (`FunctionDef`, `CallSite`, `ScanIndex`) |
| `model.py` | foundation | `Resource`, `Effect`, `Pattern`, `PatternMatch`, `CallEdge`, `FunctionModel`, `CallGraph` |
| `resources.py` | foundation | resource taxonomy, `ResourceTables` (normalizers, cfg/GPR tables, overlaps, groups) |
| `effects_db.py` | foundation | core patterns, family loading, `build()` -> `EffectDB`, JSON, findings normalization |
| `families/<f>.py` | one FAMILY agent each | domain patterns/hooks (`sfpu`, `unpack`, `pack`, `math`, `compute_api`) |
| `kernels.py`, `hazards.py` | CHECKER | kernel scanning and `run(db, repo_root)` -> findings |

Pipeline: `scan.scan_tree` -> direct effects (core patterns + every family) -> roles and
threads -> statically resolved call edges -> `CallGraph` (transitive, context-sensitive) ->
`EffectDB`.

---

## 1. What is scanned (`scan.py`)

| Layer | Roots (repo-relative) |
|---|---|
| `compute_api` | `tt_metal/hw/inc/api/compute/**` |
| `llk_api` | `tt_metal/hw/ckernels/blackhole/metal/llk_api/**`, `.../metal/llk_io/**` |
| `llk_lib` | `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/**` (incl. `experimental/`, `debug/`, `hal/`) |
| `common` | `tt_metal/tt-llk/tt_llk_blackhole/common/inc/**` |
| `sfpu` | any `.../common/inc/sfpu/**` file and any `ckernel_sfpu*.h` (e.g. `llk_api/llk_sfpu/ckernel_sfpu_exp.h`) |

The scanner reads files with suffix `.h .hpp .hh .inc`. Kernel `.cpp` files are not
scanned here; see `EffectDB.add_sources`.

Each file goes through these steps, and every step keeps byte offsets:

1. Comments and string literals are blanked.
2. The preprocessor is evaluated with three-valued logic. `ARCH_BLACKHOLE` is 1;
   `ARCH_WORMHOLE`, `ARCH_WORMHOLE_B0`, `ARCH_GRAYSKULL` and `ARCH_QUASAR` are undefined.
   `__has_builtin(__builtin_rvtt_*)` is true, so the folded tree's intrinsic branches are
   the live ones. Header guards are true.
   - A branch that is decidably false is blanked.
   - An undecidable branch is kept, and its text becomes `pp_condition` or part of a call's
     `condition` (e.g. `defined(PERF_DUMP)`).
   - `#ifdef TRISC_{UNPACK,MATH,PACK}` and `UCK_CHLKC_*` regions do not become conditions.
     They are recorded as the thread of the lines they cover.
   - Directive lines are blanked. `#define` lines become `MacroDef`.
   - The argument lists of `LLK_ASSERT`, `LLK_ASSERT_BLOCK`, `LLK_PANIC`, `ASSERT`, `SAN_HOOK` and
     `static_assert` are blanked (`scan.UNEVALUATED_MACROS`): in a production build they are not
     evaluated, so calls and effects written inside them are not part of the analysed code.
3. A brace-scope walk finds every definition at namespace or class scope. This covers free
   functions, `ALWI`, `inline`, `sfpi_inline`, `__attribute__((..))` and attribute-macro
   prefixes (e.g. `RECONFIG_DF_DEPRECATED("..") ALWI void f(..)`). It also covers templates,
   explicit specializations (`specialization_args`), overloads, in-class methods,
   out-of-class methods (`ckernel_template::program`), constructors, partial-specialization
   class bodies and `void MAIN`.
4. Call sites are extracted from each body. Function-like macros defined in the scanned
   tree are fully expanded (`SFPU_UNARY_CALL`, `SFPU_UNARY_INIT_FN`, `_SFPU_EXPAND`, ...).
   The expanded calls carry `via_macro` and the line of the invocation. Instruction
   macros from `ckernel_ops.h` (`TTI_*`, `TT_*`, `TT_OP_*`) are never expanded: they stay as
   call sites with `is_macro=True` and are matched by name. `UNPACK((..))`, `MATH((..))`
   and `PACK((..))` are not call sites. They set `CallSite.thread` for everything inside.

`FunctionDef` fields:

| Field | Meaning |
|---|---|
| `key` | Unique overload key: `qualified<template param types>[<spec args>](param types)`, plus `#2`, `#3` for `#if`/`#else` twins. Kernel functions added later get `@<path>`. |
| `name`, `qualified_name`, `namespace`, `owner`, `layer` | Identity |
| `path`, `line`, `start_line`, `end_line`, `body_line` | Location. `line` is the name token, `body_line` the `{`. |
| `params`, `template_params` | `Param(name, type, default, kind, position, variadic)` |
| `body`, `masked_body` | `masked_body` has comments, strings, inactive `#if` code and directives blanked. Offsets are identical to `body`. |
| `calls`, `expansions` | `CallSite` and `MacroExpansion` records |
| `thread_hint` | Thread from the name prefix (`llk_unpack*`, `_llk_math*`, ...), else the `#ifdef TRISC_*` region, else the file name (`cunpack_common.h`, ...). `None` for `compute_api` and `sfpu`. |
| `pp_condition`, `pp_thread`, `using`, `locals`, `attributes`, `fingerprint` | Other recorded properties |
| `line_at(off)`, `evidence_at(off)`, `condition_at(off)` | Map a `masked_body` offset to a file line, a `path:line`, or the enclosing `if`/`else`/undecidable `#if` text |

`CallSite` fields: `name` (final identifier), `callee` (as written), `qualifier`,
`template_args`, `args` (compacted texts), `line`, `offset`, `thread`, `condition`,
`via_macro`, `is_member`, `receiver`, `receiver_type`, `is_ctor` (`Type var(args)`), and
`is_macro`.

The `receiver_type` of a member call is the class of a typed local (`ckernel_template tmp(..); tmp.program()`)
or of a temporary (`addr_mod_t{..}.set(ADDR_MOD_0)`).

Helpers: `scan.evaluate(expr)` and `scan.truth(expr)` do three-valued constant folding. Qualified
names are symbolic: `a::X == a::X` is true and `a::X == a::Y` is false; two spellings of one
enumerator with more or less qualification compare equal (`ckernel::BinaryOp::ADD == BinaryOp::ADD`),
and a bare identifier equals a qualified enumerator with that last component (`StartZero ==
DstStart::StartZero`, from `using namespace`). A template call `f<..>(..)` is an opaque operand.
`truth` is memoized; when the whole text does not evaluate, the top-level `||` / `&&` operands are
decided one by one (one false conjunct makes the conjunction false). `scan.simplify_condition(expr)`
returns `(verdict, text)` with the always-true conjuncts dropped. Conditions are joined with
`scan.join_conditions`, which parenthesizes parts that hold a top-level `||` or `?:`. `scan.split_args`,
`match_close`, `match_angle` and `expand_all(text, macros)` are also exported.

---

## 2. Family plugin interface (`bh/families/<family>.py`)

`effects_db.load_families()` imports every `families/*.py` in sorted order. Files whose
name starts with `_` are templates and are not loaded. Copy `families/_example.py` to
start a family. A module exports:

| Name | Required | Contract |
|---|---|---|
| `FAMILY: str` | yes | Unique family name. Every pattern id and normalizer name must start with `"<FAMILY>."`. |
| `ENABLED: bool` | no | `False` keeps the module loaded but contributes no patterns. |
| `PATTERNS: list[model.Pattern]` | no | Declarative recognizers (below). |
| `DISABLE_CORE: tuple[str]` | no | Core pattern ids this family replaces (see section 3). |
| `GROUPS: list[(group, regex)]` | no | Extra semantic groups over canonical names (`tables.groups_of`). |
| `NORMALIZERS: dict[str, fn]` | no | `fn(expr, thread=None) -> Normalized \| list[Normalized] \| str \| None`, registered under the given `"<FAMILY>.x"` name. |
| `extract(fn, ctx) -> Iterable[Effect]` | no | Imperative hook called once per `scan.FunctionDef`. `ctx.effects(fn, pattern_id=..., resource_expr=..., normalizer=..., access=..., value=..., offset=... or line=..., condition=..., thread=..., detail=...)` builds effects exactly like a pattern would. |
| `role(fn, default) -> str \| None` | no | Role override (section 6). The first family in name order that returns non-`None` wins. |
| `thread(fn, default) -> str \| None` | no | Function thread override (`unpack`, `math`, `pack` or `any`). The first non-`None` wins. |
| `SUPERSEDES: dict[str, tuple[str]]` | no | `{"<family pattern>": ("<coarse pattern>", ..)}`: an effect of a coarse pattern is dropped when an effect of the refining pattern exists in the same function at the same evidence with the same access (e.g. `unpack.cfg_ternary` refines `core.cfg_assign`). |

`ctx` (`effects_db.FamilyContext`) carries `tables`, `family` and `index` (the tree's
`scan.ScanIndex`, for callee lookup); `ctx.effects(.., evidence="path:line")` overrides the sink.
A family effect whose content (resource, access, value, condition, evidence, thread, granularity,
detail) equals a core effect of the same function is dropped in favour of the core one: once the
core models what a family worked around, the companion is redundant.

Keep hooks scoped to your own files or names. Another family may otherwise override the same function.

### `model.Pattern`

```python
Pattern(id, kind, match, resource, normalizer="literal", access="write", value=None,
        detail=None, thread="inherit", granularity=None, receiver_type=None,
        qualifier=None, when=None, stop=True, confidence="high", note="")
```

| `kind` | Matches | `match` is |
|---|---|---|
| `call` | Every non-member call site and macro invocation, including those produced by macro expansion | A regex full-matched against `CallSite.name`, e.g. `r"TT(?:I\|_OP)?_SETC16"`. `qualifier` (a regex) must also full-match the written qualifier, e.g. `r"(?:::)?lltt"`. |
| `member` | Member calls only | The method name. `receiver_type` (a regex) must full-match the inferred class. |
| `ctor` | `Type var(args)` declarations | The class name |
| `regex` | Searched in `masked_body` and in every `MacroExpansion.text` | Any regex. Named groups become `group:NAME`. |
| `assign` | `NAME[idx] op= rhs`, `NAME op= rhs`, `++NAME`, `NAME++` (an optional `ns::` qualifier is allowed) | A regex for `NAME`, e.g. `r"vConst(?:Float\|Int)Prgm[0-3]"` |

A spec is the value of `resource`, `value` or `detail`. It is one of:

- a string:
  - `"arg:N"` or `"targ:N"` (runtime or template argument N);
  - `"args:A,B"` (several arguments joined);
  - `"targs"` or `"allargs"`;
  - `"group:G"`;
  - `"index"`, `"rhs"`, `"op"`, `"name"`, `"receiver"`;
  - `"literal:TEXT"`;
  - `"fmt:...{arg0}..{targ1}..{group:G}..{index}..{rhs}..{name}.."`;
- or a callable `(PatternMatch) -> str | None`.

When `resource` evaluates to `None` or empty, the pattern does not fire. `when(m)` filters
matches. `PatternMatch` exposes `name, callee, template_args, args, groups, index, rhs, op,
receiver, receiver_type, text, line, offset, thread, condition, via_macro`.

`normalizer` is a registered name (section 5) or a callable `(expr, thread=None)`. A
normalizer may return several names, e.g. `UNP_AB` gives `ADC.UNP_A.*` and `ADC.UNP_B.*`.
Each name becomes its own `Effect`.

`thread="inherit"` (the default) takes, in order:

1. the call site's `UNPACK`/`MATH`/`PACK` wrapper;
2. the `#ifdef TRISC_*` region;
3. the function's thread;
4. `any`.

Use an explicit `unpack`, `math` or `pack` only for state that belongs to one thread
regardless of the caller.

`stop=True` (the default) applies to call, member and ctor patterns. The pattern fully
describes the call, so transitive resolution does not descend into a scanned callee of the
same name. Use it for primitives such as `cfg_reg_rmw_tensix` or `addr_mod_t::set`. Set
`stop=False` to add a summary effect and still keep the callee's own effects.

### How to declare the common Tensix idioms

The core (section 3) already covers each of these. Copy a row to refine it, and list the
core id in `DISABLE_CORE` when you replace it.

```python
# cfg_reg_rmw_tensix<X_RMW>(v) / <X_ADDR32, SHAMT, MASK>(v): field-granular CFG write
Pattern("pack.relu_rmw", "call", r"cfg_reg_rmw_tensix", "targs", "cfg_rmw", value="arg:0",
        when=lambda m: "STACC_RELU" in " ".join(m.template_args))
# SETDMAREG -> GPR write, WRCFG -> GPR read + CFG word write (GPR flow: value names the GPR)
Pattern("unpack.setdmareg", "call", r"TT(?:I|_OP)?_SETDMAREG", "arg:3", "gpr", value="arg:1")
Pattern("unpack.wrcfg", "call", r"TT(?:I|_OP)?_WRCFG", "arg:2", "cfg_addr32", value="fmt:GPR({arg0})")
Pattern("unpack.wrcfg_src", "call", r"TT(?:I|_OP)?_WRCFG", "arg:0", "gpr", access="read")
# SFPCONFIG(imm16, dest, mod1): dest 0-3 LOADMACRO.INSTRn, 4-7 SEQn, 8 CTRL, 11-14 LREGn, 15 SFPU.CTRL
Pattern("sfpu.cfg", "call", r"TT(?:I|_OP)?_SFPCONFIG", "arg:1", "sfpconfig_dest", value="fmt:imm={arg0};mod={arg2}")
# LOADMACRO executes the configured macros (read)
Pattern("sfpu.lm", "call", r"TT(?:I|_OP)?_SFPLOADMACRO", "literal:8", "sfpconfig_dest", access="read")
# addr_mod_t{..}.set(ADDR_MOD_n)  /  addr_mod_pack_t{..}.set(n)
Pattern("math.am", "member", r"set", "arg:0", "addr_mod", receiver_type=r"addr_mod_t", value="receiver")
# ckernel_template / ckernel_unpack_template ... .program()
Pattern("math.mop", "member", r"program|program_and_run", "literal:MOP", "mop",
        receiver_type=r"ckernel_template|ckernel_unpack_template")
# sfpi programmable constants (assignment = write, any other use = read)
Pattern("sfpu.c", "assign", r"vConst(?:Float|Int)Prgm[0-3]", "name", "sfpi_const", value="rhs")
```

`TT_OP_*` forms build instruction words that a MOP or a replay program executes later. The
core puts `detail="deferred"` on them, and families should do the same by passing
`detail=effects_db.deferred`. Instructions that are only recorded into the replay buffer are deferred
too: the callable of `load_replay_buf<NoExec>(..)` and the next `len` instruction macros after
`lltt::record<NoExec>(start, len)` or `TTI_REPLAY(start, len, 0, 1)` (`effects_db.replay_recorded`).
For declarative patterns the core appends `deferred` to the effect detail (`LO_16;deferred`); a
normalizer's own detail and the pattern's detail are merged (`model.merge_detail`).

Suggested ownership, to avoid duplicate effects:

| Family | Owns |
|---|---|
| `sfpu` | LREG0-10 writes, `l_reg[]`, SFPU control bits, macro semantics, implicit ADDR_MOD of sfpi `dst_reg` access |
| `unpack` | THCON SEC0/1 unpacker fields, tilize/untilize state, unpack MOP and ADC, `unp_cfg_context` |
| `pack` | relu, edge masks, l1_acc, strides, formats, dest offsets, packer ADC |
| `math` | ALU format/accumulate fields, ADDR_MOD semantics, RWC, dest section, math MOP |
| `compute_api` | Roles of compute-API functions, fold-specific effects such as `param:<api>.<param>` |

Identical effects from two sources are deduplicated (`Effect.identity()`).

---

## 3. Core patterns (`effects_db.CORE_PATTERNS`, family `core`)

| id | Recognizes | Resource |
|---|---|---|
| `core.cfg_reg_rmw_tensix`, `core.cfg_rmw`, `core.cfg_rmw_gpr.*` | Field RMW | `CFG.<FIELD>` (field) |
| `core.cfg_write`, `core.cfg_store`, `core.cfg_assign` | `cfg_write`, `TT_CFG_STORE`, `cfg[X_ADDR32+k] = v` | `CFG.<FIELD>` (word) |
| `core.wrcfg.*`, `core.rdcfg.*` | WRCFG/RDCFG | CFG word and `GPR.<t>.<N>` |
| `core.rmwcib`, `core.cfgshiftmask` | RMWCIB0-3, CFGSHIFTMASK | `CFG.<FIELD>` (byte) |
| `core.setc16` | SETC16 | `THDCFG.<FIELD>`, or `ADDR_MOD.<n>` for `addr_mod_*_reg_addr[n]` |
| `core.setdmareg`, `core.dmareg_alu.*`, `core.regfile_assign` | GPR writes and reads (DMA-register ALU: result, OpA and, unless `OpBisConst`, OpB; 5-operand forms `BITWOP`/`SHIFT`/`CMP` take `OpSel` first) | `GPR.<t>.<NAME>` |
| `core.addr_mod_set`, `core.addr_mod_pack_set` | `addr_mod_t` / `addr_mod_pack_t` `.set(n)` | `ADDR_MOD.n`, `ADDR_MOD_PACK.n` |
| `core.addr_mod_use`, `core.addr_mod_pack_use` | ADDR_MOD operand of ELW*, MVMUL, MOV*, POOL, SFPLOAD/STORE/LOADMACRO, ZEROACC (read); PACR (read) | `ADDR_MOD.n` / `ADDR_MOD_PACK.n` |
| `core.mop_program`, `core.mop_cfg_assign`, `core.mop_run.*` | MOP program (write), run or TTI_MOP (read) | `MOP` |
| `core.replay_record`, `core.load_replay_buf`, `core.replay_insn.*`, `core.replay_run` | `lltt::record`, `load_replay_buf`, REPLAY, `lltt::replay` | `REPLAY` (`detail` holds `start=..;len=..`) |
| `core.setadc*`, `core.setrwc` | ADC and RWC | `ADC.<UNP_A\|UNP_B\|PAC>.<X\|Y\|Z\|W\|XY\|ZW\|XX>`, `RWC.<SRCA\|SRCB\|DEST\|FIDELITY\|CR>` |
| `core.sfpconfig`, `core.sfploadmacro` | SFPCONFIG write, LOADMACRO read | `SFPU.*` |
| `core.sfpi_constant`, `core.sfpi_constant_read` | `vConst*Prgm0-3` assigned or used | `SFPU.LREG11-14` |
| `core.seminit`, `core.sempost_get`, `core.semwait`, `core.t6_semaphore_*`, `core.semaphore_read` | Semaphores (`semaphore_read(semaphore::X)` is a read) | `SEM.<NAME>` |
| `core.mailbox_*` | `mailbox_write`, `mailbox_read` | `MAILBOX.<thread>` |
| `core.sw_global` | Assignment to an LLK global (`unp_cfg_context`, `dest_offset_id`, `src_zero_flag_*`, ...) | `SW.<name>` |

---

## 4. `FunctionModel` / `Effect` / `EffectDB` API (what the CHECKER consumes)

```python
from tools.llk_state_audit.bh import effects_db
db = effects_db.build(repo_root)          # EffectDB
```

| Member | Meaning |
|---|---|
| `db.functions: dict[key, FunctionModel]`, `db.keys()`, `db.function(key)`, `db.by_name(name)` | The functions |
| `db.effects(key) -> tuple[Effect]` | Transitive effects in the function's own scope (cached) |
| `db.direct_effects(key)` | Effects in the function's own body only |
| `db.call_effects(name, qualifier=, template_args=, args=, thread=, condition=, evidence="path:line", caller_path=, is_member=, receiver_type=)` | Effects of a call written somewhere else (e.g. a kernel), with that call's arguments substituted. Pattern-only primitives work too, e.g. `call_effects("TTI_SETC16", args=(..))`. `evidence` is prepended to `via`. |
| `db.add_sources([(path, text), ...], layer="kernel") -> [FunctionModel]` | Scan kernels and kernel-local headers into the same graph, keyed `<key>@<path>`. `MAIN` and `kernel_main` get role `execute`. Then `db.effects(model.key)` works. They are excluded from bh-effects JSON and counts. |
| `db.resolve(name, ...) -> (candidates, resolution)` | Overload resolution |
| `db.reachable(key) -> [(callee_key, depth, call_evidence)]` | Static call closure; stop edges are skipped unless `include_stopped=True` |
| `db.tables` (`ResourceTables`) | `overlaps(a, b, gran_a, gran_b)`, `groups_of(name)`, `storage(name)`, `normalize(kind, expr, thread)`, `int_value(expr)` |
| `resources.same_thread(t1, t2)`, `resources.is_per_thread(name)`, `resources.scope_of(name)` | Thread and scope helpers |
| `db.scan` (`ScanIndex`), `db.graph` (`CallGraph`), `db.tree_fingerprint`, `db.repo_root`, `db.layer_counts()`, `db.role_counts()` | Raw scanner output, the call graph and tree metadata |

`FunctionModel` fields:

- identity: `key, name, qualified_name, owner, path, line, end_line, layer, role, thread`;
- signature: `params, template_params, signature, attributes`;
- `pp_condition`, `fingerprint`, `local_constants`;
- `effects` (direct), `calls` (`CallEdge`s), `definition` (the `scan.FunctionDef`), and
  `param_names`.

`CallEdge` fields:

- `name, callee, qualifier, callee_keys`;
- `resolution`: `exact`, `overload`, `ambiguous`, `unresolved`, `param_callable` or `stopped`;
- `line, evidence, thread, condition, template_args, args, via_macro, is_member,
  receiver_type, is_ctor`;
- `stop`, and `patterns` (ids of the patterns that matched this call).

Calls with no resolution and no matching pattern are dropped from `calls`. The raw site
is still in `definition.calls`.

`Effect` fields:

| Field | Meaning |
|---|---|
| `resource` | Canonical name (section 5). `resolved=False` when it still depends on an unbound parameter (contains `?`). |
| `access` | `read` or `write` |
| `value_kind`, `value` | `const`, `param` (value is exactly a parameter of the reporting function), `expr` or `unknown`, plus the value text |
| `condition` | `if`/`else` and undecidable `#if` text joined with ` && `; `""` when unconditional |
| `evidence` | `path:line` of the sink: the instruction or primitive, not the call site |
| `thread` | `unpack`, `math`, `pack` or `any` |
| `via` | `((callee_key, call_evidence), ...)` from the reporting function down to the function holding the sink. Empty when direct. |
| `params` | Parameters of the reporting function that flow into resource, value or condition, including through locals |
| `family`, `pattern`, `granularity` | Provenance and granularity: `field`, `word`, `byte`, `word_range`, `slot`, `range` or `whole` |
| `detail` | Extra information: `deferred` for `TT_OP_` words, `start=..;len=..` for REPLAY, `LO_16`/`HI_16` for GPR halves |
| `confidence` | `high` direct, `medium` transitive, `low` through an ambiguous overload or an unresolved resource |
| `resource_expr`, `normalizer`, `norm_index` | Used for renormalization after substitution |

Transitive semantics (`CallGraph`) follow these rules:

- **Substitution.** An effect of callee `g` reached from `f` through `g<T..>(a..)` is
  re-expressed in `f`'s scope. Missing arguments take their declared defaults; arguments
  that are still missing become `?param`. The callee's `constexpr`/`const` locals, and its
  locals assigned exactly once, are substituted too (e.g. `UNP_SEL`, `addr_mod`). Pack
  expansions (`args...`) expand against the bound pack.
- **Folding.** A condition that folds to `false` drops the effect, and one that folds to
  `true` is removed; conjuncts that are always true are dropped from an undecided condition.
  `if constexpr (sfpu_op == SfpuType::abs)` therefore folds under `SfpuType::exponential`.
  Conditions are kept in full while they propagate (the 300-character cap applies to the JSON
  only), so a long `if constexpr` chain still folds several levels up.
- **Renormalization.** A resource that depended on a parameter is normalized again
  (`addr_mod_t::set(mod_index)` -> `ADDR_MOD.3`). `_RMW` template arguments are expanded
  to `ADDR32, SHAMT, MASK`.
- **Callable parameters.** `init_func(..)` or `sfpu_func(..)` resolves through the bound
  argument, e.g. `sfpu::exp_init<APPROX>` or `::ckernel::sfpu::calculate_exponential<..>`.
- **Overloads.** Lookup applies unqualified-lookup visibility (enclosing namespaces plus
  `using namespace`), arity, explicit specializations, template-argument kinds,
  deducibility, argument types, same file and unconditional definitions. When more than
  one candidate remains, the edge is `ambiguous` and effects of all candidates are unioned
  with `confidence=low`. An ambiguous edge whose template arguments become concrete under
  substitution is re-resolved.
- **Threads.** An effect whose thread is `any` takes the call site's wrapper thread, else
  the caller's thread.
- **Limits.** The analysis stops at depth 24 and on cycles. Results are memoized per
  `(function, relevant bindings)`. Substituted texts are kept up to 4000 characters; value and
  detail are capped at 300, and the condition is capped at 300 in the JSON only.
- **Params.** Literal words (`true`, `false`, `nullptr`) of a bound argument are not params.
- **Value kinds.** Direct effects are classified too (`classify_value`); a `GPR(..)` value (the
  content of a GPR) is `expr`, never `const`.

The bh-effects JSON uses sorted keys and puts one function per line under `functions`.
Each function holds:

- `key, name, path, line, layer, role, thread, params, ...`;
- `effects` (direct, full);
- `calls`;
- `transitive_summary`: one row per (resource, access, thread) with `count`, the first 3
  `evidence`, `values` and `conditions`, the `params`, `families`, and the shortest `via`
  as callee keys.

`counts` holds functions per layer, roles, and the direct and transitive totals. The JSON
also has `patterns`, `families`, `resource_classes` and `tree_fingerprint`. Use the Python
API for full transitive detail.

---

## 5. Resource naming (`resources.py`)

| Class | Scope | Name | Normalizer (`ResourceTables.normalizers`) |
|---|---|---|---|
| `CFG` | shared | `CFG.<FIELD>`: the `cfg_defines.h` field without `_ADDR32`/`_SHAMT`/`_MASK`/`_RMW`. A word write is named after the field whose `ADDR32` the source used, or the first field of the word for `X_ADDR32 + k`. `cfg_rmw<X_ADDR32, SHAMT, MASK>` names every field of X's word that MASK covers when MASK evaluates (cfg_defines `_ADDR32`/`_SHAMT`/`_MASK` symbols and the function's constexpr locals are known to `int_value`); a SHAMT of a field in another word does not rename the field. | `cfg_rmw` (field), `cfg_addr32` (word), `cfg_byte` (byte) |
| `THDCFG` | per_thread | `THDCFG.<FIELD>` for THREAD-section registers (SETC16) | `thread_cfg` |
| `ADDR_MOD`, `ADDR_MOD_PACK` | per_thread | `ADDR_MOD.<0-7>`, `ADDR_MOD_PACK.<0-3>` | `addr_mod`, `addr_mod_pack` |
| `GPR` | thread_named | `GPR.<unpack\|math\|pack\|any>.<NAME>`, where `NAME` comes from `ckernel_gpr_map.h`. `p_gpr::*` maps to `any`. An unnamed index is `R<n>`. `LO_16`/`HI_16` go in `detail`. | `gpr` |
| `MOP` | per_thread | `MOP` | `mop` |
| `REPLAY` | per_thread | `REPLAY` (`detail` holds `start=..;len=..`) | `replay` |
| `ADC` | per_thread | `ADC.<UNP_A\|UNP_B\|PAC>.<dims>` | `adc` (expr `"<mask>\|<dims>"`) |
| `RWC` | per_thread | `RWC.<SRCA\|SRCB\|DEST\|FIDELITY\|CR>` | `rwc` |
| `SFPU` | shared | `SFPU.LREG<n>` (11-14 are the programmable constants; `vConst*Prgm0/1/2/3` map to LREG12/13/14/11), `SFPU.LOADMACRO.INSTR<0-3>`, `SFPU.LOADMACRO.SEQ<0-3>`, `SFPU.LOADMACRO.CTRL`, `SFPU.CTRL` | `sfpconfig_dest`, `lreg`, `sfpi_const` |
| `SEM` | shared | `SEM.<FPU_SFPU\|MATH_PACK\|UNPACK_TO_DEST\|UNPACK_OPERAND_SYNC\|PACK_DONE\|UNPACK_SYNC\|UNPACK_MATH_DONE\|MATH_DONE>` | `semaphore` |
| `MAILBOX` | shared | `MAILBOX.<thread>` | `mailbox` |
| `SW` | per_thread | `SW.<global>` | `sw` |
| `DEST`, `PACK`, `UNPACK` | shared | Names defined by families, for state that is not a single CFG field (e.g. `DEST.SECTION`, `PACK.RELU`) | Family `NORMALIZERS` |
| `param` | n/a | `param:<api>.<param>`: a compute-API parameter treated as state (fold findings) | n/a |

Rules:

- An unresolved name contains `?` (e.g. `ADDR_MOD.?addr_mod`). Its effect has
  `resolved=False`, and `overlaps()` treats it as overlapping its whole class.
- For `per_thread` classes, two effects touch the same storage only when
  `resources.same_thread(e1.thread, e2.thread)`.
- `tables.overlaps(a, b, granularity_a, granularity_b)` compares cfg storage. A word, byte
  or word_range write overlaps every field of its word. ADC names overlap when the unit is the
  same and they share a dimension (`ADC.UNP_A.XY` overlaps `.X` and `.Y`).
- `tables.groups_of(name)` returns semantic groups such as `pack.relu`, `pack.edge_mask`,
  `pack.l1_acc`, `pack.strides`, `pack.format`, `unpack.format`, `alu.format`,
  `thread.dest_offset`, `sfpu.constants`, `sfpu.loadmacro` and `addr_mod`. Families add
  more with `GROUPS`.

---

## 6. Roles (`effects_db.default_role`, then family `role()` hooks)

The default role comes from the name, after stripping leading and trailing `_` and
lowercasing. The first matching rule wins:

1. A constructor is `helper`.
2. `hw_configure`, `hw_startup`, `*_init_common` or `init_common` is `hw_startup`.
3. A name containing `uninit` is `uninit`.
4. A name containing `reconfig` or `reconfigure` is `reconfig`.
5. `init` as a word, or a name ending in `init`, is `init`. Names starting with `is_` or
   `get_` are excluded.
6. `config`, `configure`, `cfg`, `setup` or `program` as a word, or a `set_`, `clear_`,
   `reset_`, `enable_` or `disable_` prefix, is `config`.
7. Otherwise, a function in the `compute_api`, `llk_api` or `llk_lib` layer is `execute`.
8. Otherwise, an `sfpu` function named `calculate_*` or `calc_*` is `execute`.
9. Everything else is `helper`.

After the FOLD, execute APIs keep role `execute`. Their folded init and reconfig calls show
up as `CallEdge`s to callees with role `init` or `reconfig`, which `db.reachable` exposes,
and as transitive effects whose `via[0]` is that callee.

---

## 7. Findings (`bh-hazards`)

`bh-hazards` builds the DB. When `bh/hazards.py` exists, it calls `hazards.run(db, repo_root)`,
which returns either a list of finding dicts or `{"findings": [...], "tree_fingerprint": "..."}`
(to fold kernel sources into the fingerprint). Without `hazards.py` it writes an empty list.
`effects_db.normalize_findings` validates and canonicalizes the findings:

```json
{"tree_fingerprint": "<sha256>", "findings": [{
  "id": "H0001", "rank": 1,
  "kind": "resource_conflict | fold_drop | ordering",
  "confidence": "high | medium | low",
  "resource": "<canonical name, e.g. SFPU.LREG12, GPR.unpack.TMP0, CFG.<FIELD>, ADDR_MOD.3, param:<api>.<param>>",
  "ops": ["<compute-API / LLK names, most specific first>"],
  "summary": "<one sentence>",
  "kernels": ["<repo-relative kernel paths, sorted, max 20>"], "kernel_count": 0,
  "evidence": ["<repo-relative path:line>"]}]}
```

Normalization does the following:

- `kind`, `confidence`, `rank` (an int) and `resource` (non-empty) are required; invalid
  values raise `ValueError`;
- `ops` are deduplicated in order;
- `kernels` are sorted, deduplicated and truncated to 20;
- `kernel_count` defaults to the number of distinct kernels before truncation;
- `evidence` is sorted by path, then line, and deduplicated;
- `summary` whitespace is compacted;
- unknown keys are dropped;
- findings are sorted by `(rank, kind, resource, ops)`, with `summary`, `kernels` and
  `evidence` as tie-breakers, and then numbered `H0001`...

`hazards.run` itself merges findings with the same `(kind, resource, ops)` (overloads of one API)
and drops findings whose resource is unresolved (contains `?`).

`tree_fingerprint` is the sha256 over the (path, bytes) of every analysed header in path
order, plus `cfg_defines.h`.

## 8. Determinism rules (all modules)

- No timestamps, absolute paths, environment-dependent values, set-iteration order or
  hash randomization in any output.
- Sort before emitting, and use repo-relative paths.
- `effects_db.dumps()` (`indent=1`, `sort_keys`) is used for findings and
  `dumps_effects()` for bh-effects.
- Run every command twice and `cmp` the outputs.
