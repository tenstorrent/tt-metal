# Doxygen Docstring Style

How we document the LLK codebase. The goal is **high-signal, low-noise** docs: enough
to explain intent, parameters, and the init/execute/uninit contract — without the
ceremony of heavily-regulated industries. When in doubt, write less. A docstring that
restates the signature adds noise; one that explains *why*, names a non-obvious
contract, or points at the function you actually need next earns its place.

## Where these rules apply

| Layer | Path | Doxygen tags? |
|-------|------|---------------|
| LLK lib (`_llk_*`) | `tt_llk_{arch}/llk_lib/`, `.../common/inc/` | Yes — full tag style below. Internal; not published. |
| LLK API (`llk_*`) | `tt_metal/hw/ckernels/{arch}/metal/llk_api/` | Yes — full tag style below. Internal; not published. |
| Compute API | `tt_metal/hw/inc/api/compute/` | **Special case — see below.** |

The library is **header-only**: a function's declaration *is* its definition, so the
docstring sits directly above the single definition. There is no separate header to
keep in sync.

## Format

Use a Javadoc-style block comment immediately above the definition:

```cpp
/**
 * @brief One-line summary of what the function does.
 *
 * Optional free-text paragraph for context the tags below can't carry.
 *
 * @tparam ...
 * @param ...
 * @note ...
 */
```

- One blank ` *` line between `@brief` and the body, and between the body and the tags.
- Keep `@param`/`@tparam` order matching the declaration order.
- The codebase uses `@param name: description` (colon after the name). Match it.

## Tags we use

| Tag | When | Notes |
|-----|------|-------|
| `@brief` | Always, one line | Summarize what the function does. |
| `@param` | Every runtime parameter | Describe meaning/units, not the type (the signature has the type). |
| `@tparam` | Every template parameter | List valid values for enums, e.g. `values = <NONE/COL/ROW/SCALAR>`. |
| `@ref` | Cross-references | Link to the related function/type the reader needs next — see *Cross-thread referencing*. |
| `@note` | The init/execute/uninit contract, cross-thread pairings, and surprising side effects | State what the caller must run first (typically the matching `_init_`) and after (typically the matching `_uninit_`), plus any genuinely surprising side effect (e.g. "writes LREG7") or non-obvious constraint. Prefer this over `@remark`. See *The init / execute / uninit contract*. |

## Tags to avoid

These tend to bloat without informing:

| Tag | Why avoid | Use instead |
|-----|-----------|-------------|
| `@details` | The plain paragraph after `@brief` already serves this. | Free-text body. |
| `@return` | Most LLK functions are `void`. Document returns only on **test functions and helpers** that actually return a value. | (omit for void) |
| `@author` / `@date` / `@version` | Git already tracks this, and it goes stale. | `git blame` / history. |
| `@todo` | Tends to rot in-tree. | A GitHub issue. |
| `@remark` | Ambiguous vs `@note`. | `@note`. |
| `@pre` / `@post` | In standard Doxygen semantics these designate conditions *guaranteed* to hold before/after the call. Our init/uninit contract is an **imperative** the caller must satisfy ("you must call `_init_` first / `_uninit_` after"), not a guarantee — so using `@pre`/`@post` for it overloads their meaning and risks being misread (by humans and AI alike). We don't support real pre/post-conditions, so we don't use these tags at all. | `@note`, phrased imperatively. |

## The init / execute / uninit contract

This is where docstrings earn their keep. LLK functions come in `_init_` / execute /
`_uninit_` families, and every register touched in `_init_` must be restored in
`_uninit_`. Encode that contract in a `@note` (with `@ref` links) so a caller can't get
the ordering wrong. State both halves — the `_init_` that must run before and the
`_uninit_` that must run after — as an **imperative**, in a single note:

```cpp
/**
 * @brief Perform an elementwise binary op: Output = SrcA [+, -, *] SrcB.
 *
 * Dispatches to the standard or dest-reuse implementation based on binary_reuse_dest.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam binary_reuse_dest: Reuse dest as source, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @param tensor_shape: Tile dimensions of the operands.
 * @param dst_index: Tile index into the destination register.
 * @note Call @ref _llk_math_eltwise_binary_init_ with matching template args before this
 *       function, and @ref _llk_math_eltwise_binary_uninit_ after it to restore modified state.
 */
```

Why `@note` and not `@pre`/`@post`: in standard Doxygen, `@pre`/`@post` are conditions
*guaranteed* to hold around the call. Ours is the opposite — an imperative the *caller*
must satisfy. Phrasing it as a `@note` ("call X before this and Y after") says exactly
that without overloading tag semantics. Write it loosely and plainly; the goal is that a
reader (or AI) can't misread it as an automatic guarantee.

## Cross-thread referencing

A Tensix compute op runs on **three TRISC threads in lockstep** — T0 (unpack), T1
(math), T2 (pack) — each executing a *different* LLK that drives a different execution
unit. The three halves of one logical operation live in three different files, and how
they pair up is often non-obvious. When a cross-reference makes that pairing clear,
add it.

For example, a reader looking at `_llk_math_reduce_init_` (T1) can't tell from its
signature which unpack init feeds it. Spell it out:

```cpp
/**
 * @brief Configure the math (FPU) thread for a reduce operation.
 *
 * @tparam type: Reduction op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @note On the unpack thread, pair with @ref _llk_unpack_reduce_init_ (single operand)
 *       or @ref _llk_unpack_AB_reduce_init_ (with scaler operand).
 * @ref _llk_math_reduce_  is the matching execute call on this thread.
 */
```

Rule of thumb: **cross-reference same-family functions across threads when it increases
readability.** Don't cross-reference for its own sake — only where the pairing isn't
obvious from names alone.

## Compute API exception (published docs)

The **Compute API** (`tt_metal/hw/inc/api/compute/`) is the only LLK-adjacent layer
fed into the public Sphinx documentation: `docs/Doxyfile` lists these headers as
`INPUT`, and `docs/source/.../kernel_apis/compute/*.rst` pulls each function in with
`.. doxygenfunction::`. Those docstrings therefore use an **established published
format**, not `@param`/`@tparam` tags:

```cpp
// clang-format off
/**
 * Prose description of the function and its contract.
 *
 * NOTE: caveats go in NOTE: lines.
 *
 * Return value: None
 *
 * | Param Type | Name | Description | Type | Valid Range | Required |
 * |------------|------|-------------|------|-------------|----------|
 * | Template   | ...  | ...         | ...  | ...         | ...      |
 * | Function   | ...  | ...         | ...  | ...         | ...      |
 */
// clang-format on
```

**Do not convert Compute API docstrings to `@param`/`@tparam`** — the markdown table is
what renders cleanly in the public docs, and consistency across that surface matters.
What *does* carry over is the *thinking*: state the init/execute/uninit contract and
the cross-thread/cross-call pairing in the prose (e.g. "must be followed by a call to
`reduce_tile`"), just expressed in sentences rather than `@note`/`@ref` tags.

### Standardizing the table

The surface is already largely consistent; these rules lock that in and say where *not*
to spend effort:

- **Always wrap the doc block in `// clang-format off` / `// clang-format on`.** Without
  the guard the formatter will eventually reflow and mangle the table.
- **Fixed columns, fixed order:** `Param Type | Name | Description | Type | Valid Range | Required`.
  `Param Type` is `Template` or `Function`.
- **`Return value: None`** for void functions; a single descriptive line otherwise.
- **Do not hand-align column widths.** Markdown renders identically regardless of
  padding, so reflowing columns to line up is pure churn for zero rendered benefit —
  leave existing spacing alone.
- **Content discipline** (so the table earns its keep rather than bloating):
  - `Description` explains meaning/constraints; it does not restate the C++ type, which
    `Type` already carries.
  - Fill `Valid Range` only when there's a genuine constraint (e.g. CB IDs `0–31`,
    `{true, false}`); leave it minimal otherwise.
  - Omit the table entirely for zero-parameter functions — just prose + the contract.

## Rule of thumb

- `@brief` on everything public.
- `@param`/`@tparam` for everything the caller passes.
- `@note` (with `@ref`) to encode the init/execute/uninit ordering contract as an
  imperative, and to pair up the per-thread halves of an op when the pairing isn't obvious.
- Never use `@pre`/`@post` — phrase the contract as an imperative `@note` instead.
- Reach for the body paragraph or extra `@note` lines only when the *why* isn't obvious.
- Compute API: keep the published prose+table format; carry the contract in prose.
- Everything else: leave it out.
