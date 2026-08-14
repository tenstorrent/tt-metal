# LLK Formal Contract — glossary & semantics

Status: **draft, converging with the Sanitizer.** This is the math statement of the LLK
contract the **Sanitizer** enforces. It tracks the code on branch `llk/san/state-refactor`
(`tools/include/sanitizer/{types,operation,api,impl}.h`).

Neither side is frozen. Where the math and the code disagree today, §7 lists it as a
convergence item. The target: every symbol here maps to one place in the code, and the
checker tool reads its rules from §5–§6.

Notation: `≜` is "defined as", `⊕` joins named fields into a record, `⊥` is unset,
`∏` is a product over fields.

---

## 1. Execution units and threads

```
I ≜ { U, F, S, P }                        // Exu = { Unpack, Fpu, Sfpu, Pack }   [types.h]
```

Native thread per EXU (`detail::is_exu_native`, types.h):

| EXU | native thread |
|-----|---------------|
| U (Unpack) | TRISC0 |
| F (Fpu)    | TRISC1 |
| S (Sfpu)   | TRISC1 (the math thread drives both F and S) |
| P (Pack)   | TRISC2 |

TRISC3 drives no EXU. Hooks from it are rejected (`OperandDefect::Unsupported`).

---

## 2. State hierarchy

### 2.1 Complete sanitizer state `S`

```
S ≜ ⟨ s_U, s_F, s_S, s_P ; Γ ⟩           // struct State [operation.h]
```

`Γ` is the global `UnwindContext unwind`. It holds source locations, not contract state.
Your `S = Σᵢ sᵢ` is this, plus `Γ`.

### 2.2 Per-EXU state `sᵢ`  (`ExuState<i>`, types.h)

```
sᵢ ≜ ⟨ ρᵢ , ωᵢ , yᵢ ⟩
```

| symbol | code | meaning |
|--------|------|---------|
| `ρᵢ` | `ApiClass previous` | last API class called on this EXU |
| `ωᵢ` | `OperationUnion<i>::Struct operation` | the live operation record (a tagged union) |
| `yᵢ` | `Operand<i>::Struct operand` | operand state of the EXU |

Live operation record:

```
ωᵢ  ∈  { ∅ } ∪ { x̂ⱼ : j ∈ Ωᵢ }          // std::variant<monostate, OperationExtended<Ops>...>
```

Your `sᵢ = Σⱼ (x̂ⱼ + yᵢ)` maps to this if `Σ` is a tagged union, not a sum, and `yᵢ` is
pulled out of it:

```
sᵢ  =  ρᵢ  ⊕  yᵢ  ⊕  ⨁̇_{j∈Ωᵢ} x̂ⱼ
```

`⨁̇` (your `Σ`) is a tagged union: at runtime one term is live. That live term is `ωᵢ`, or
`∅` before the first `init`. `yᵢ` is shared by all operations of the EXU, so it sits once in
`sᵢ` and once as the snapshot `y'ⱼ` inside each `x̂ⱼ`. This is the `std::variant` (one live
alternative) plus the single `Operand<i>::Struct` in `ExuState`.

### 2.3 Declared operation set `Ωᵢ`

```
Ωᵢ ≜ ExuOperations<i>::type = OperationList<…>     [operation.h]
```

Registered today (operation.h):

| EXU | `Ωᵢ` | Hoistable |
|-----|------|-----------|
| U | `OperationUnpackUnary`, `OperationUnpackBinary`, `OperationUnpackMatmul` | Yes |
| U | `OperationUnpackTilize` | **No** |
| F | *(empty. TODOs: matmul, datacopy, binary add/sub/mul, dest-reuse)* | — |
| S | *(empty)* | — |
| P | *(empty. TODOs: pack, pack-untilize)* | — |

### 2.4 Operand state `yᵢ` and its fields `φ`

```
yᵢ ≜ ( φ_{i,0}, …, φ_{i,Qᵢ-1} ),   each φ ∈ dom(φ) ∪ {⊥}
```

`⊥` means the field's `known` bit is clear (`StateStruct::known`, types.h). Field sets
(operation.h):

- `y_U` (Q=9): `InputFormatA, OutputFormatA, FaceHeightA, NumFacesA, InputFormatB,
  OutputFormatB, FaceHeightB, NumFacesB, DestWidth32`
- `y_F` (Q=1): `Format`
- `y_S` (Q=1): `Format`
- `y_P` (Q=8): `InputFormat, OutputFormat, FaceHeight, TileWidth, NumFaces, PartialFace,
  NarrowTile, DestWidth32`

### 2.5 Extended operation state `x̂ⱼ`  (`OperationExtended<Op>`, types.h)

```
x̂ⱼ ≜ ⟨ σⱼ , xⱼ , y'ⱼ ⟩
```

| symbol | code | meaning |
|--------|------|---------|
| `σⱼ` | `OperationStatus status` ∈ {Init, Exec, Uninit} | lifecycle status of this operation |
| `xⱼ` | `G::Struct state` | the operation's own fields (§2.6) |
| `y'ⱼ` | `Operand<i>::Struct snap` | operand snapshot = `maskⱼ(yᵢ)` |

Your `x̂ⱼ = xⱼ + y'ⱼ` leaves out `σⱼ`. Use `x̂ⱼ = σⱼ ⊕ xⱼ ⊕ y'ⱼ`.

**Mask.** `maskⱼ` is the `known` bits of `snap`. `init` snapshots only the operand fields the
hook restated; the rest stay `⊥`. So `y'ⱼ` is a partial view of `yᵢ`: the fields operation `j`
depends on.

Why the snapshot exists (confirmed): an operation depends on a subset of the operand fields.
`y'ⱼ` lets `execute` and `uninit` catch that subset changing between the two halves of the
operation. Comparing against the `configure` record would miss it, because that record has not
changed. Example: tilize restates `OutputFormat`, `FaceHeight`, `NumFaces` at `execute`; a
mismatch there is a real bug.

### 2.6 Own-field sets `xⱼ` (operation.h)

| operation `j` | own fields `xⱼ` |
|---------------|-----------------|
| UnpackUnary   | BroadcastType, AccumulateToDest, BinaryReuseDest, UnpackToDest |
| UnpackBinary  | BroadcastType, FaceWidth, NumFacesRow, NumFacesCol, Transpose |
| UnpackMatmul  | KernelBroadcastA, KernelBroadcastB, Transpose, CtDim, RtDim, KtDim, PartialFaceA, PartialFaceB |
| UnpackTilize  | BlockCtDim, NarrowTile |

---

## 3. Program points and the parameter-dependence law

`k` indexes hook calls on one EXU's thread, in program order. `yᵢ[k]` is the operand state
before the k-th state-changing call; `yᵢ[k+1]` is after. Same for `x̂ⱼ[k]`.

Every state-changing hook takes parameters `p = (p₀, …, p_{M-1})`. The law:

```
(L1)  ∃ f.  yᵢ[k+1] = f( p₀[k], …, p_{M-1}[k] )
(L2)  ∀ q.  ∃ g_q.  φ_q[k+1] = g_q(p)          // each field is its own pure function of p
```

In code, each hook argument is a `StateVal` over one field. The stored field is `g_q` of the
argument, not the argument itself. `g_q` encodes the parameter into the field's form: a shift
(`v<<16`), a boolean-to-int (`v?1:0`), a format-enum lookup, and so on. It is the identity only
in the simple case. Whether the checker should see the raw LLK params or only the encoded value
at the hook is **Q4**.

---

## 4. The five entry points as state transformers

From `api.h` (`configure/reconfigure/init/execute/uninit`) into `impl.h`
(`state_operand_impl`, `state_operation_impl`, `operand_write`, `operation_write`).

`v` ranges over the `StateVal`s passed in. `StateDiscard` writes nothing. For a `StateVal` `v`
targeting field `φ`, the stored value is `g_φ(v)`: the field's encoding of the parameter (§3),
identity only in the simple case. `g_φ` generalizes §3's `g_q` from operand fields to any field.

```
configure(p)   :  ∀ v∈p targeting φ_q of i.   yᵢ.φ_q ← g_q(v) ; known(φ_q) ← 1
                  ρᵢ ← Configure
reconfigure(p) :  same write;  ρᵢ ← Reconfigure

init⟨Op⟩(p)    :  ωᵢ ← x̂_Op  (create if ωᵢ ≠ x̂_Op)
                  ∀ v∈p:  v targets Op-own φ  → x_Op.φ  ← g_φ(v)
                          v targets operand φ → y'_Op.φ ← g_φ(v) ; known ← 1   (snapshot)
                  σ_Op ← Init         [C3: not set yet]
                  ρᵢ ← Initialize
execute⟨Op⟩(p) :  ∀ v∈p:  v targets Op-own φ  → assert x_Op.φ  = g_φ(v)        [C2: today writes]
                          v targets operand φ → assert y'_Op.φ = g_φ(v)        (drift check) [C4: result dropped]
                  σ_Op ← Exec         [C3]
                  ρᵢ ← Execute
uninit⟨Op⟩()   :  σ_Op ← Uninit       [C3];  ρᵢ ← Uninitialize
```

Guards (compile-time, on now. `operand_defect`/`operation_defect`, types.h): every argument
must be a `StateVal` or `StateDiscard` (`Params`), of the right family (`Kind`: Operand for
(re)configure; own-or-operand for init/exec/uninit), and of an EXU this thread drives
(`Native`). Operations must also name a listed, native `Op` (`NotAnOperation`, `NotNative`,
`NotListed`). This is the type-level half of the contract. It runs today.

---

## 5. The four contract functions (your definitions, reconciled)

**F1 — reconfig.** `reconfig_i( yᵢ[k]; p ) = yᵢ[k+1]`, with `yᵢ[k] ≠ yᵢ[k+1]` for a useful
reconfigure. From L1/L2: if every parameter equals the one from the call that produced `yᵢ[k]`,
the state does not change.

```
(F1-idem)  (∀m. p_m[k+1] = p_m[k])  ⇒  reconfig_i(yᵢ[k]; p[k+1]) = yᵢ[k]
```

A reconfigure whose result equals the prior state is redundant. That is a performance defect,
not a correctness one. (Legacy FSM: `CONFIGURED→RECONFIGURED` is WARN;
`INITIALIZED[Op]→RECONFIGURED` with no required uninit is a deprecated WARN.)

**F2 — init.** `initⱼ( x̂ⱼ[k] ) = x̂ⱼ[k+1]` with `x̂ⱼ[k] ≠ x̂ⱼ[k+1]`. It sets `σⱼ ← Init`,
writes `xⱼ`, and snapshots `y'ⱼ = maskⱼ(yᵢ)`. `init` may read `yᵢ`, which is why `y'ⱼ` lives
inside `x̂ⱼ`. (Your tilize example: `init` sets own fields that a prior `reconfig` left at
operand defaults.) A no-op init is redundant (FSM WARN).

**F3 — uninit = init⁻¹.** `uninit` returns the extended operation state to its value before
`init`: `σⱼ ← Uninit`, and (intended) `y'ⱼ` released. It needs a matching prior `init`, so it
is legal only from `INITIALIZED[Op]` or `EXECUTED[Op]`. It reads `y'ⱼ` and is the left inverse
of the matching `init` on the `⟨σ, y'⟩` part.

**F4 — execute (visibility condition).** `execute` need not avoid writing state. It may write
and revert. What it must guarantee: the state any later operation can observe is the same
before and after. Anything else is a state leak.

Split the machine state into `Tracked ⊎ Scratch`:

- `Scratch` — HW config that every `execute` sets fresh before use, so a leftover value is
  never observed: MOP, `SETADC`/`SETADCXX`, address modes (`ADDRMOD`). An `execute` may
  clobber these freely.
- `Tracked` — everything the Sanitizer models: `xⱼ`, `y'ⱼ`, `yᵢ`. These must read back
  unchanged.

```
(F4)  execute⟨Op⟩ :   ∀ c ∈ Tracked.  observe(c)[k+1] = observe(c)[k]      // net identity
                      ∀ c ∈ Scratch.  (no constraint)
                      σ ← Exec
      drift alarm :   ∃ restated operand field φ.  yᵢ.φ ≠ y'_Op.φ   ⇒  VIOLATION
```

`observe` is the net effect. A write-then-revert on a `Tracked` field is fine; a bare write is
not.

**Own fields are init-fixed.** Every field in `xⱼ` is set at `init` and must be the same at
`execute`. To use different values, the kernel calls `init` again. Matmul `CtDim`, `RtDim`,
`KtDim` follow this: one set of dims per init/execute pair. New dims need a new `init`, because
the dims program the MOP and the REPLAY buffer.

```
F4 on own fields :  ∀ c ∈ xⱼ.  observe(c)[k+1] = observe(c)[k]
```

No execute-varying own field is known. If one appears, mark it and exempt it here.

> **C2 — execute writes own fields instead of comparing them.** `operation_write` sends every
> own field to `record.state.update()` for all ApiClasses (`impl.h`). Own fields are
> init-fixed, so execute should compare, not write. The write is a stub: reporting (`output.h`)
> is off, so nothing reads own-field equality yet. Fix: at execute, assert each own field
> equals the value from `init`.

---

## 6. Lifecycle ordering (the sequencing contract)

Per EXU, the API calls drive a state machine — the `fsm_check` in the Sanitizer (impl.h,
currently `#if 0`, C5). This section writes it out in full: it is the authoritative,
complete transition set. **Any transition not listed below as `ok` or `WARN` is an `ERROR`.**

**States** and the call that enters each:

| state | entered by | meaning |
|-------|-----------|---------|
| `INITIAL` | — | before any call |
| `CONFIGURED` | `configure` (hw config) | operand state set |
| `INITIALIZED[Op]` | `init⟨Op⟩` | operation `Op` set up |
| `EXECUTED[Op]` | `execute⟨Op⟩` | `Op` has run at least once |
| `UNINITIALIZED[Op]` | `uninit⟨Op⟩` | `Op` torn down |
| `RECONFIGURED` | `reconfigure` | operand state changed |

Conventions: `[Op]` = the *same* operation as the current state; `[Any]` = any operation (the
op may change on this edge). `expect_uninit(Op)` (defined below) = whether `Op` requires an
uninit. A `—` in the `expect_uninit` column means the rule holds either way.

**Complete transition table** (unlisted `to` from a given state ⇒ `ERROR`):

| from | to | expect_uninit | verdict |
|------|----|:-------------:|---------|
| `INITIAL` | `CONFIGURED` | — | ok |
| `INITIAL` | anything else | — | **ERROR** — first call must be `configure` |
| `CONFIGURED` | `INITIALIZED[Any]` | — | ok |
| `CONFIGURED` | `RECONFIGURED` | — | WARN — reconfigure right after configure is wasted |
| `INITIALIZED[Op]` | `EXECUTED[Op]` | — | ok — the intended path |
| `INITIALIZED[Op]` | `INITIALIZED[Any]` | — | WARN — redundant re-init |
| `INITIALIZED[Op]` | `UNINITIALIZED[Op]` | — | WARN — init then teardown, no execute |
| `INITIALIZED[Op]` | `RECONFIGURED` | No | WARN — deprecated |
| `INITIALIZED[Op]` | `RECONFIGURED` | Yes | **ERROR** — `Op` needs its uninit first |
| `EXECUTED[Op]` | `EXECUTED[Op]` | — | ok — run again |
| `EXECUTED[Op]` | `UNINITIALIZED[Op]` | Yes | ok |
| `EXECUTED[Op]` | `INITIALIZED[Any]` | No | ok |
| `EXECUTED[Op]` | `RECONFIGURED` | No | ok |
| `EXECUTED[Op]` | `UNINITIALIZED[Op]` | No | **ERROR** — `Op` has no uninit |
| `EXECUTED[Op]` | `INITIALIZED[Any]` \| `RECONFIGURED` | Yes | **ERROR** — must uninit `Op` first |
| `UNINITIALIZED[Op]` | `INITIALIZED[Any]` | — | ok |
| `UNINITIALIZED[Op]` | `RECONFIGURED` | — | ok |
| `RECONFIGURED` | `INITIALIZED[Any]` | — | ok |
| `RECONFIGURED` | `RECONFIGURED` | — | ok |

**Invariants that fall out of the table:**

- `EXECUTED` is reachable **only** from `INITIALIZED` or `EXECUTED`. So you must `init` before
  the first `execute`, and after a `reconfigure` or an `uninit` you must re-`init` before the
  next `execute` — `RECONFIGURED` and `UNINITIALIZED` never step straight to `EXECUTED`.
- `[Op]` matching: `execute` and `uninit` must name the operation currently live; only `init`
  may switch operation (`[Any]`). (The FSM matches op *type* only — operand identity is not
  tracked; see §14.)
- Accepting states (a kernel may legally end here): `EXECUTED[Op]`, and `UNINITIALIZED[Op]` for
  ops with `expect_uninit`. Ending in `INITIALIZED`/`RECONFIGURED` (setup with no following
  execute) is wasted work — a WARN, not an ERROR.

(§14's fuzzer generator uses the permissive union of this table — accept if valid under either
`expect_uninit` regime — since it decorates skeletons before the op's `expect_uninit` is known.)

`expect_uninit(Op)` is a per-operation, per-architecture attribute. It is not derivable from
`Hoistable`. Operations that need uninit include at least tilize, untilize, reduce, and maybe
transpose (unconfirmed). This set does not match `Hoistable=No`. Quasar is messier, so
`expect_uninit` is set per arch and carried as its own tag on each operation, not inferred.

```
expect_uninit : Ωᵢ × Arch → { Yes, No }        // explicit table, filled per op per arch
```

**Missing uninit.** A skipped mandatory uninit surfaces as `EXECUTED[Op] → INITIALIZED[*]`, so
marking that transition invalid catches it — not a blind spot. Strict form (require uninit for
every op; the no-op uninits exist for this) catches all missing uninits without trusting the
`expect_uninit` table, at the cost of placing uninit hooks kernel-wide (code changes).

**Kernel compliance.** A kernel's per-EXU hook sequence `c₁;c₂;…` obeys the contract when
every transition `ρ/σ(c_n) → ρ/σ(c_{n+1})` is `ok` in the table, every `execute` passes F4,
and every `configure`/`reconfigure` passes the guards in §4. The first non-`ok` transition is
the violation. `WARN` transitions obey the contract but waste work (perf defects).

---

## 7. Convergence items (contract ↔ current implementation)

| id | gap | where |
|----|-----|-------|
| ~~C1~~ | **Resolved.** `sᵢ` is a tagged union with shared `yᵢ` (§2.2); `variant` (≤1 live) is right. | — |
| C2 | `execute` writes own fields instead of comparing. Fix: assert-equal (own fields are init-fixed). | `operation_write`, A==Execute, impl.h |
| C3 | `σ` (`OperationStatus`) never set; `ρᵢ` set is a TODO. | `// sstanisic todo` in `state_operation_impl`/`state_operand_impl`, impl.h |
| C4 | operand drift is computed, then dropped. No report. | `operation_write` else-branch, impl.h |
| C5 | full FSM (§6) is `#if 0`; ordering not enforced. Only the type-level guards (§4) run. | legacy hooks, impl.h/api.h |
| C6 | `DestWidth32` assumed to be configure-only shared state. | `#47440`, legacy `unpack_operand_configure_impl` |

**Note on C2 — why `execute` writes own fields today.** Checked on the branch (one squashed
`temp` commit, no commit message to go on). It is a stub, not a chosen semantics:

- The documented policy is the comment above `operation_write`: own field → written; operand
  field → snapshot at init, compare after. The real drift-check was put on the operand snapshot
  `y'ⱼ`, not on `xⱼ`.
- The checking half is unbuilt: `output.h` reporting `#if 0`, FSM `#if 0`, `status` unset, and
  the operand compare's result is dropped (C4). Nothing detects anything yet.
- `StateStruct::update` is defined to overwrite (`test_state_functional.cpp` asserts
  "Overwriting a known field tracks the new value"). The `execute<Tilize>(BlockCtDim, …)` in
  `valid_operation.cpp` is a compile-time guard positive (execute accepts own fields), not a
  runtime test.

Writing is a safe stub while checking is off: a wrong write only misses a detection. Own fields
are init-fixed (Q7), so the target is to compare at execute. Waits on the reporting path (C4).

---

## 8. Open questions

- **Q2** Cross-EXU sync (producer→consumer, U→F→P) is not in `State` at all. In scope for this
  contract, or a separate one?
- **Q3** Fill the `expect_uninit : Ωᵢ × Arch → {Yes,No}` table. Needs uninit: tilize, untilize,
  reduce, transpose(?). Characterize Quasar separately.
- **Q4** Should the checker see the raw LLK params (to verify `g_q`), or only the hook boundary
  where `g_q` is the identity?
- **Q5** Are `WARN` (perf) transitions counted as violations in the obeys-vs-violates metric,
  or only `ERROR`?
- **Q6** List `Scratch` exactly, per arch (MOP, SETADC/SETADCXX, ADDRMOD, …), so F4's exemption
  set is closed and auditable.
- **Q7 (resolved)** All own fields are init-fixed; execute compares. Matmul `CtDim`/`RtDim`/
  `KtDim` are one set per init/execute pair. New dims need a new `init`, because the dims
  program the MOP and the REPLAY buffer. No execute-varying own field is known.

---

## 9. How to extend

To add an operation: add its `xⱼ` field table (§2.6), register it in `Ωᵢ` (§2.3), and set its
`Hoistable` and `expect_uninit`. To add an operand field: add it to §2.4. §5–§6 then follow,
and the checker reads its rules from those two sections.

---

## 10. Coverage gaps (what the Sanitizer cannot verify by design)

The Sanitizer flags wrong usage and stays silent on good usage. "Silent" hides two cases:
the state determines correctness and it's fine (**true OK**), or correctness depends on
information the tool does not hold, so it is silent either way (**blind**). A third verdict,
`UNVERIFIABLE`, names the second case. Usability is then a number:

```
certainty = 1 − (blind sites) / (state-changing calls)
```

### 10.1 What counts as a blind spot

Test: a real blind spot is one a *correctly implemented and maintained* Sanitizer still
cannot catch. Anything a correct mask / config / FSM would catch is a process flaw or a plain
bug, not a design gap.

| gap | class | verdict |
|-----|-------|---------|
| **G1** | **no history** — no pre-init baseline, so `uninit`-restore cannot be verified | **real, in scope** |
| G2 | masking — `y'ⱼ` keeps only `maskⱼ(yᵢ)` | not a gap. A correct mask covers what init/execute read; a stale mask is a maintenance miss |
| G3 | unknown (`⊥`) — field read before any `configure` set it | not a gap. The known-bit is clear, so it is catchable as use-before-set |
| G4 | out-of-model — `Scratch` (MOP/SETADC/ADDRMOD); cross-EXU sync | `Scratch` is safe by design (every `execute` re-sets it). Cross-EXU sync is a real gap, but out of the model today (Q2) |

So the only in-scope, quantifiable design gap is **G1**. Cross-EXU sync (G4) is real but out
of scope until `State` models inter-thread order.

### 10.2 G1 measured on `origin/main`

Commit `e71a69bbc5a`, 435 compute kernels. Static scan of the per-kernel call sequence, §6 FSM
overlaid. A `uninit` is G1 when the FSM stays silent (renders no verdict) **and** downstream
code relies on the operand state across the `uninit` with no `configure`/`reconfig` to
re-establish it (the `Uninit; Init/Execute` with nothing in between).

State-changing calls (denominator): **3014** — init 1870, reconfig 671, configure 368,
uninit 105. (Execute op calls, for context: 4844.)

Of 105 `uninit` sites:

| | count |
|---|---|
| FSM-catchable (tool objects — not blind) | 15 |
| FSM-silent, restore not relied on (verifiable) | 45 |
| **G1 (FSM-silent, restore relied on — blind)** | **45** |

```
certainty ≥ 98.5%   (2969 / 3014 state-changing calls)
G1 blind  ≤ 1.5%    (45 uninit sites; the only in-scope gap)
```

**45 is an upper bound.** Whether a *following init* truly re-establishes the operand, or only
re-snapshots a possibly-unrestored value, depends on per-op field writes not visible in kernel
source. True G1 ≤ 45; certainty ≥ 98.5%.

Closing G1 (colleague's call, a design change): push `(op, full operand snapshot)` at `init`,
pop + diff at `uninit`. Converts every G1 site from `UNVERIFIABLE` to `OK`/`ERROR`.

### 10.3 Method and limits

- Regex over comment-stripped source; classify calls by name suffix; template-arg lists
  handled; §6 FSM at op-type granularity.
- **Limit 1**: whole-file linear scan conflates helper-definition order with call order
  (minor — init/uninit stay adjacent within a helper).
- **Limit 2**: cross-TU init/uninit is invisible to a per-file scan.
- **Limit 3**: operand/CB-identity (the literal cb1/cb2 case) is not counted — comparing
  positional args needs per-function signatures (`init(in,out)` vs `uninit(out)` differ
  trivially). 10.2 uses the signature-free reliance test instead.
- Script: `tools/analysis/blindspot.py` (to be committed).

Not measured: G4 cross-EXU sync (out of model). G2/G3 excluded by 10.1.

---

## 11. Contract-conformance gaps (F3: `uninit` = `init⁻¹`)

Distinct from §10: not "can the tool see it" but "does the LLK code satisfy the contract." F3
requires `uninit` to return the EXU to its pre-`init` state — but only for **non-hoistable**
ops. For a hoistable op the next `init` fully re-establishes the state, so a no-op `uninit` is
fine.

Measured by two per-arch sage swarms driven off the state-audit map
(`tools/llk_state_audit/`), each agent fed the audit's persistent-write list for the `init`
and judging whether `uninit` reverts each entry. Every WH+BH `init`/`uninit` pair was covered.

```
44 WH+BH pairs verified →  6 gaps  |  34 benign (hoistable)  |  4 correct
```

Quasar has **no `uninit` functions** — it does not use the teardown pattern, so F3-inverse
does not apply; QSR restores via `canonical_reset` config helpers instead. So ~34/44 uninits
don't fully restore, but that's benign (hoistable — the next `init` re-establishes the state);
**6/44 are real bugs.** "Not a true inverse" is common; "must be and isn't" is rare.

### The 6 confirmed gaps

| arch | function | leaked state | why it bites |
|------|----------|--------------|--------------|
| BH | `_llk_unpack_untilize_uninit_` | SrcA Y-stride set to `FACE_C_DIM·face_r_dim·x` = 16× canonical (`llk_unpack_untilize.h:152,166`) | `uninit` actively writes a wrong value; WH save/restores correctly (WH/BH divergence). Deprecated op (tt-metal#22904) |
| WH | `_llk_unpack_tilizeA_B_uninit_` | `Tile_x_dim_cntx0` always `FACE_DIM_16x16` (`:592`) | wrong for `face_r_dim<16` (tiny tiles); the sibling `_llk_unpack_tilize_uninit_` was fixed (`:576` `canonical_unpA_tile_x_dim_cntx`), this one wasn't |
| BH | `_llk_math_fast_tilize_uninit_` | `DEST_ACCESS_CFG` `remap_addrs` + `swizzle_32b` never cleared | see systemic note below |
| BH | `_llk_math_fast_untilize_uninit_` | same `DEST_ACCESS_CFG` remap/swizzle (no-op uninit) | see systemic note below |
| BH | `_llk_pack_fast_untilize_uninit_` | `DEST_TARGET_REG_CFG` offset + `DEST_OFFSET_LO/HI` GPRs left at bottom-strip phase | next pack inherits a stale DEST offset |
| BH | `_llk_unpack_AB_reduce_block_max_row_uninit_` | `ALU_ACC_CTRL_Zero_Flag_disabled_src` left at 1 (fp32 path) | not cleared/restored; WH sibling is benign (WH/BH divergence) |

**Systemic — BH DEST remap ownership gap (2 of the 6).** `fast_tilize`/`fast_untilize` init
call `_llk_math_reconfig_remap_(true)` (`llk_math_fast_tilize.h:35`, `llk_math_fast_untilize.h:67`)
but **no `_(false)` call exists anywhere in the BH tree**. The comments contradict each other —
`llk_math_fast_tilize.h:99` "DEST remap is cleared by pack uninit" vs `llk_pack_fast_tilize.h:282`
"DEST remap is NOT cleared here — owned by the math thread" — so each thread assumes the other
clears it and neither does. A later op reading DEST with linear addressing gets remapped data.

Notes: 3 of 6 gaps are in `experimental/` fast-(un)tilize paths; 3 are WH/BH divergences the
per-arch swarm split surfaced (untilize WH-ok/BH-gap; tilizeA_B WH-gap/BH-ok;
reduce_block_max_row WH-benign/BH-gap).

---

## 12. Reconfig-escape gaps (sticky cfg mode-bits)

A third class, from the audit's largest persistent bucket (1479 `cfg_register` effects
"retained until reconfigured"). A gap here is a behavioral cfg bit — accumulate, relu,
format-override, DEST-remap — that an op sets to a non-default and that nothing resets except a
full `hw_configure`, so a later op inherits the wrong mode. Distinct from §11: it applies to
ops with no `uninit`, and to Quasar (which has no uninit pattern).

Method: from the 1479, drop the self-refreshing address/stride/counter regs (recomputed by
every op that uses them) and the bits the baseline `hw_configure` re-establishes; keep
behavioral bits with no in-body clear-to-default. That leaves 13 one-way candidates (WH 5,
BH 2, QSR 6). A per-arch sage swarm (incl. `sage-quasar`) then checked each: fixed vs
parameterized setter, and whether any reset/disable path exists.

```
13 candidates →  3 confirmed gaps  |  8 benign  |  2 swarm false-positives (overturned)
```

Confirmed:

- BH `DEST_ACCESS_CFG_remap_addrs` + `swizzle_32b` — the **same two bits as §11's fast-(un)tilize
  gaps**, re-derived here from the no-uninit lens (`_llk_math_reconfig_remap_(true)` set, no
  `(false)` caller anywhere). Method-validation, not a new gap.
- **QSR `THCON_PACKER0_REG3_PACK_STRIDE_NO_WRITE` (new)** — set to fixed `1` in the small-tile
  branch of `_llk_pack_untilize_strided_init_` (`llk_pack_untilize.h:271`); no write of `0`
  exists anywhere in `tt_llk_quasar`, and `_llk_pack_hw_configure_` touches only REG0. A
  following PACKER0 op inherits row-write suppression.

Benign (8): WH `Pack_L1_Acc` ×4 and `ALU SrcA_val` (reset via `hw_configure` / dedicated clear);
QSR PACKER0 `L1_ACC` / `RELU_MODE` / `EDGE_MASK_MODE` (re-established each `_llk_pack_init_`,
which hardcodes the PACK0 branch, `llk_pack.h:73`).

**Methodology note (swarm reliability).** The swarm flagged QSR PACKER1 `L1_ACC` and `RELU_MODE`
as gaps while calling their PACKER0 twins benign — but the code is structurally identical
(`PACK_SEL`-branched, lines 212/216 and 255/260). Source validation overturned both:
`_llk_pack_init_` only ever instantiates the PACK0 branch, so PACKER1's RELU is never written in
the standard path, and L1_ACC is caller-managed symmetrically for both packers. On Quasar
(Sonnet, no DeepWiki) the swarm needs a same-code consistency cross-check; contradictory
verdicts on twin resources are the tell.

**Running total across classes: §11 F3 (6) + §12 reconfig-escape (1 new) = 7 distinct confirmed
gaps** (the 2 BH DEST bits are shared between the classes, counted once).

---

## 13. Effort to map `x̂_j` and `y'_j`

Building the Sanitizer's per-operation state map means, for each op `j`: its own fields `x_j`
and the operand fields it depends on, `y'_j` (§2.5). Estimated against the state-audit map.

**Key asymmetry.** The audit records state *writes* (effects), but `y'_j` is a *read*
dependency — "op `j` reads `φ`, so snapshot it." So the audit gives `x_j` (own writes) and
`init`'s operand-*writes* for free, but the `y'_j` read-deps are the manual residual. That is
why it is hard: when neither `init` nor `execute` receives `φ` as a parameter, the dependency is
only visible by reading the body and its helpers.

`x_j` — essentially auto-mapped: it is the audit's effects table (5463 rows).

`y'_j` — splits by whether `init` even takes the operand field as a parameter:

| arch | init fns | `init` takes an operand param (explicit `y'_j`) | `init` does not (implicit → read code) |
|------|----------|------------------------------------------------|----------------------------------------|
| WH   | 49 | 18 (37%) | 31 |
| BH   | 52 | 20 (38%) | 32 |
| QSR  | 25 | **0 (0%)** | **25** |

So **~88 operation definitions** need a manual code read to recover operand dependencies. Two
things sharpen it:

- **Quasar is fully implicit** — 0/25 QSR inits take an operand-format/face param; it threads
  `TensorShape`/`buf_desc_id` semantically, so every QSR op's `y'_j` is hidden in the body.
- **62% of `cfg_register` effects are `parameter.kind = fixed`** (917/1479, over 146 functions,
  median 4/fn) — a state write the parser could not attribute to any argument. That is the
  concrete scale of "not readable from the signature."

The `init-does-not` bucket (88) further splits into "execute receives it" (medium — pair
init/execute) and "neither" (hard — trace body + helpers); QSR's 25 are all the hard kind.

**Reframe.** ~88 (WH+BH implicit) + all QSR defs × one focused code read each — which is exactly
the shape of the per-op, per-arch sage swarms used in §11–§12. So the mapping is **~1–2 swarm
passes**, not a hand audit; the 37% where `init` takes the operand params is free from the
signature.

---

## 14. Fuzzer audit (independent, all FSM-valid sequences)

A generator (`tools/analysis/fuzz_sequences.py`) enumerates every FSM-valid call sequence over
`{CFG, RCFG, INIT, EXE, UNI}` × operand `{A, B}`, length ≤ 6 — 60 skeletons → 584 distinct
scenarios (incl. the `cb1/cb2` example). An independent AI auditor, given only §1–§9 plus a
blind-spot rubric (§10–§13 withheld), labelled each `covered` / `blind` / `invalid`.

```
584 →  342 covered | 37 invalid | 205 blind   (no_history 107, ground_truth 98, cross_exu 0)
```

Do not read 205 as "35% of usage is blind" — the space weights all orderings equally, including
sequences no sane kernel writes. The value is that the blind set collapses to **two structural
gaps**:

- **Operand identity (`no_history`, 107).** The operation record is keyed by op *type*, not
  operand/CB identity, so `UNI_B` on an `INIT_A` record — a wrong-operand teardown — is accepted.
  This is G1 generalized to the `cb1/cb2` case; the tool keeps no operand-ownership baseline.
- **Operand-vs-tracked-`y` consistency (`ground_truth`, 98; 71 use operand B with no `CFG_B`).**
  `init` snapshots the operand requirement and `execute` checks against that snapshot
  (self-consistent), but nothing compares the requirement against tracked `y_i` with its
  known-bits — so using an operand that was never configured, or misconfigured, passes silently.

Consequences:

1. Both gaps close with mechanisms already identified: (a) add operand identity to the operation
   record; (b) add the `init`/`execute` operand-args-vs-tracked-`y_i` check that §4 currently
   omits and legacy `unpack_operand_check` had (see §11 C2 / the missing-reconfig analysis). With
   both, the 205 collapse toward covered.
2. This **conditions §10's "certainty ≥ 98.5%":** that figure assumed the operand-consistency
   check works and ignored operand identity. The fuzzer, reasoning from §4 *as written*, shows
   those two are load-bearing — absent them the abstract blind fraction is 35%, and real usage is
   more exposed than 98.5% implied.

`cross_exu` = 0 only because the single-EXU alphabet cannot express cross-thread order; probing
G4b needs a multi-EXU fuzzer. Caveat: the auditor is Sonnet; verdicts are contract-reasoning
estimates, corroborated by the two classes matching independently-derived findings, not
individually source-verified.
