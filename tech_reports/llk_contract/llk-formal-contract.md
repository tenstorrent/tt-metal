# LLK Formal Contract: glossary & semantics

This is the math behind the LLK contract that the **Sanitizer** checks. It fixes the words we
use, splits the state into parts, and says what the five entry points, the four contract
functions, and the per-EXU lifecycle machine do. Every symbol maps to the Sanitizer code in
`tools/include/sanitizer/{types,operation,api,impl}.h` (branch `llk/san/state-refactor`), and
each one names its file where it first shows up.

The code does not match this yet. For the bugs, the rules not switched on, the open questions,
and the audited LLK gaps, see the companion doc [`san_contract_gaps.md`](./san_contract_gaps.md).
The goal is plain: one symbol, one place in the code, and a checker that reads its rules straight
from §5 and §6.

Notation: `≜` is "defined as", `∈` is "is an element of", `⨁` joins named fields into a record, `⊥` is unset (as in not
touched by the Sanitizer, not a HW reset value).

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

TRISC3 drives no EXU. Hooks from it are turned away (`OperandDefect::Unsupported`).

---

## 2. State hierarchy

### 2.1 Complete sanitizer state `S`

```
S ≜ ⟨ s_U, s_F, s_S, s_P ; Γ ⟩           // struct State [operation.h]
```

`Γ` is the global `UnwindContext unwind`. It holds source locations, not contract state.
We can restate this as `S = Σᵢ sᵢ + Γ`, where `i ∈ I`.

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
Therefore:

```
sᵢ  =  ρᵢ  +  yᵢ  +  Σ_{j∈Ωᵢ} x̂ⱼ
```

`ρᵢ` is only for the FSM check; the rest is for the operand check. If you care only about the
operand check, drop `ρᵢ`: `sᵢ = Σⱼ x̂ⱼ + yᵢ`, where `Σ` is a tagged union, not a sum.

`⨁` is that tagged union: at runtime one term is live. The live term is `ωᵢ`, or `∅` before the
first `init`. `yᵢ` is shared by every operation of the EXU, so it sits once in `sᵢ` and again as
the snapshot `y'ⱼ` inside each `x̂ⱼ`. In code that is the `std::variant` (one live alternative)
plus the single `Operand<i>::Struct` in `ExuState`.

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

Watch the short form: `x̂ⱼ = xⱼ + y'ⱼ` drops `σⱼ`. The full one is `x̂ⱼ = σⱼ ⊕ xⱼ ⊕ y'ⱼ`.

**Mask.** `maskⱼ` is the `known` bits of `snap`. `init` snapshots only the operand fields the
hook restated; the rest stay `⊥`. So `y'ⱼ` is a partial view of `yᵢ`: the fields operation `j`
leans on.

The snapshot earns its place. An operation depends on some of the operand fields, and `y'ⱼ` lets
`execute` and `uninit` catch that subset changing between the two halves of the op. Checking
against the `configure` record would miss it, because that record has not moved. Take tilize: it
restates `OutputFormat`, `FaceHeight`, `NumFaces` at `execute`, and a mismatch there is a real
bug.

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

In code, each hook argument is a `StateVal` over one field. What gets stored is `g_q` of the
argument, not the argument itself. `g_q` puts the parameter into the field's form: a shift
(`v<<16`), a bool to int (`v?1:0`), a format-enum lookup, and so on. It is the identity only in
the easy case. Whether the checker should see the raw LLK params or only the encoded value at the
hook is an open question (gaps §8, Q4).

---

## 4. The five entry points as state transformers

From `api.h` (`configure/reconfigure/init/execute/uninit`) into `impl.h`
(`state_operand_impl`, `state_operation_impl`, `operand_write`, `operation_write`).

`v` ranges over the `StateVal`s passed in. `StateDiscard` writes nothing. For a `StateVal` `v`
aimed at field `φ`, the stored value is `g_φ(v)`: the field's encoding of the parameter (§3),
identity only in the easy case. `g_φ` is §3's `g_q` widened from operand fields to any field.

```
configure(p)   :  ∀ v∈p targeting φ_q of i.   yᵢ.φ_q ← g_q(v) ; known(φ_q) ← 1
                  ρᵢ ← Configure
reconfigure(p) :  same write;  ρᵢ ← Reconfigure

init⟨Op⟩(p)    :  ωᵢ ← x̂_Op  (create if ωᵢ ≠ x̂_Op)
                  ∀ v∈p:  v targets Op-own φ  → x_Op.φ  ← g_φ(v)
                          v targets operand φ → y'_Op.φ ← g_φ(v) ; known ← 1   (snapshot)
                  σ_Op ← Init
                  ρᵢ ← Initialize
execute⟨Op⟩(p) :  ∀ v∈p:  v targets Op-own φ  → assert x_Op.φ  = g_φ(v)
                          v targets operand φ → assert y'_Op.φ = g_φ(v)        (drift check)
                  σ_Op ← Exec
                  ρᵢ ← Execute
uninit⟨Op⟩()   :  σ_Op ← Uninit;  ρᵢ ← Uninitialize
```

Guards (compile-time, live today. `operand_defect`/`operation_defect`, types.h): every argument
must be a `StateVal` or `StateDiscard` (`Params`), of the right family (`Kind`: Operand for
(re)configure; own-or-operand for init/exec/uninit), and of an EXU this thread drives
(`Native`). Operations must also name a listed, native `Op` (`NotAnOperation`, `NotNative`,
`NotListed`). This is the type half of the contract, and it already runs.

---

## 5. The four contract functions

**F1, reconfig.** `reconfig_i( yᵢ[k]; p ) = yᵢ[k+1]`, with `yᵢ[k] ≠ yᵢ[k+1]` for a reconfigure
that does anything. From L1/L2: if every parameter matches the one from the call that produced
`yᵢ[k]`, the state does not budge.

```
(F1-idem)  (∀m. p_m[k+1] = p_m[k])  ⇒  reconfig_i(yᵢ[k]; p[k+1]) = yᵢ[k]
```

A reconfigure that lands on the state you already had is dead weight: a perf defect, not a
correctness one. (Legacy FSM: `CONFIGURED→RECONFIGURED` is WARN; `INITIALIZED[Op]→RECONFIGURED`
with no required uninit is a deprecated WARN.)

**F2, init.** `initⱼ( x̂ⱼ[k] ) = x̂ⱼ[k+1]` with `x̂ⱼ[k] ≠ x̂ⱼ[k+1]`. It sets `σⱼ ← Init`, writes
`xⱼ`, and snapshots `y'ⱼ = maskⱼ(yᵢ)`. `init` may read `yᵢ`, which is why `y'ⱼ` lives inside
`x̂ⱼ`. (Tilize is the example: `init` sets own fields that a prior `reconfig` left at operand
defaults.) A no-op init is dead weight too (FSM WARN).

**F3, uninit = init⁻¹.** `uninit` puts the extended operation state back where it was before
`init`: `σⱼ ← Uninit`, and (the plan) `y'ⱼ` released. It needs a matching prior `init`, so it is
legal only from `INITIALIZED[Op]` or `EXECUTED[Op]`. It reads `y'ⱼ` and is the left inverse of
the matching `init` on the `⟨σ, y'⟩` part.

**F4, execute (what stays visible).** `execute` may write state, so long as it puts it back. The
one rule: what any later operation can see is the same before and after. Anything else is a leak.

Split the machine state into `Tracked ⊎ Scratch`:

- `Scratch`: HW config every `execute` sets fresh before use, so a leftover value is never seen.
  MOP, `SETADC`/`SETADCXX`, address modes (`ADDRMOD`). An `execute` may stomp on these freely.
- `Tracked`: everything the Sanitizer models, namely `xⱼ`, `y'ⱼ`, `yᵢ`. These must read back the same.

```
(F4)  execute⟨Op⟩ :   ∀ c ∈ Tracked.  observe(c)[k+1] = observe(c)[k]      // net identity
                      ∀ c ∈ Scratch.  (no constraint)
                      σ ← Exec
      drift alarm :   ∃ restated operand field φ.  yᵢ.φ ≠ y'_Op.φ   ⇒  VIOLATION
```

`observe` is the net effect. Write-then-revert on a `Tracked` field is fine; a bare write is not.

**Own fields are set once, at init.** Every field in `xⱼ` is set at `init` and must be the same
at `execute`. To use other values, the kernel calls `init` again. Matmul `CtDim`, `RtDim`,
`KtDim` work this way: one set of dims per init/execute pair. New dims need a new `init`, because
the dims program the MOP and the REPLAY buffer.

```
F4 on own fields :  ∀ c ∈ xⱼ.  observe(c)[k+1] = observe(c)[k]
```

No own field is known to change at execute. If one turns up, mark it and exempt it here.

---

## 6. Lifecycle ordering (the sequencing contract)

Per EXU, the API calls drive a state machine: the lifecycle FSM (`fsm_check`, impl.h). Here it is
in full, and this is the whole of it. **Any transition not listed below as `ok` or `WARN` is an
`ERROR`.**

**States** and the call that enters each:

| state | entered by | meaning |
|-------|-----------|---------|
| `INITIAL` | — | before any call |
| `CONFIGURED` | `configure` (hw config) | operand state set |
| `INITIALIZED[Op]` | `init⟨Op⟩` | operation `Op` set up |
| `EXECUTED[Op]` | `execute⟨Op⟩` | `Op` has run at least once |
| `UNINITIALIZED[Op]` | `uninit⟨Op⟩` | `Op` torn down |
| `RECONFIGURED` | `reconfigure` | operand state changed |

Conventions: `[Op]` = the *same* operation as the current state; `[Any]` = any operation (the op
may change on this edge). `expect_uninit(Op)` (below) = whether `Op` requires an uninit. A `—` in
the `expect_uninit` column means the rule holds either way.

**Complete transition table** (an unlisted `to` from a given state is an `ERROR`):

| from | to | expect_uninit | verdict |
|------|----|:-------------:|---------|
| `INITIAL` | `CONFIGURED` | — | ok |
| `INITIAL` | anything else | — | **ERROR**: first call must be `configure` |
| `CONFIGURED` | `INITIALIZED[Any]` | — | ok |
| `CONFIGURED` | `RECONFIGURED` | — | WARN: reconfigure right after configure is wasted |
| `INITIALIZED[Op]` | `EXECUTED[Op]` | — | ok: the intended path |
| `INITIALIZED[Op]` | `INITIALIZED[Any]` | — | WARN: redundant re-init |
| `INITIALIZED[Op]` | `UNINITIALIZED[Op]` | — | WARN: init then teardown, no execute |
| `INITIALIZED[Op]` | `RECONFIGURED` | No | WARN: deprecated |
| `INITIALIZED[Op]` | `RECONFIGURED` | Yes | **ERROR**: `Op` needs its uninit first |
| `EXECUTED[Op]` | `EXECUTED[Op]` | — | ok: run again |
| `EXECUTED[Op]` | `UNINITIALIZED[Op]` | Yes | ok |
| `EXECUTED[Op]` | `INITIALIZED[Any]` | No | ok |
| `EXECUTED[Op]` | `RECONFIGURED` | No | ok |
| `EXECUTED[Op]` | `UNINITIALIZED[Op]` | No | **ERROR**: `Op` has no uninit |
| `EXECUTED[Op]` | `INITIALIZED[Any]` \| `RECONFIGURED` | Yes | **ERROR**: must uninit `Op` first |
| `UNINITIALIZED[Op]` | `INITIALIZED[Any]` | — | ok |
| `UNINITIALIZED[Op]` | `RECONFIGURED` | — | ok |
| `RECONFIGURED` | `INITIALIZED[Any]` | — | ok |
| `RECONFIGURED` | `RECONFIGURED` | — | ok |

**What the table forces:**

- `EXECUTED` is reachable **only** from `INITIALIZED` or `EXECUTED`. So you must `init` before the
  first `execute`, and after a `reconfigure` or an `uninit` you must `init` again before the next
  `execute`. `RECONFIGURED` and `UNINITIALIZED` never step straight to `EXECUTED`.
- `[Op]` matching: `execute` and `uninit` must name the operation that is live; only `init` may
  switch operation (`[Any]`). The FSM matches op *type* only, not operand identity (see gaps §14).
- Where a kernel may legally stop: `EXECUTED[Op]`, and `UNINITIALIZED[Op]` for ops with
  `expect_uninit`. Stopping in `INITIALIZED`/`RECONFIGURED` (setup with no execute after it) is
  wasted work: a WARN, not an ERROR.

(The fuzzer in gaps §14 uses the loose union of this table, accepting an edge if it is valid under
either `expect_uninit` setting, since it builds skeletons before the op's `expect_uninit` is
known.)

`expect_uninit(Op)` is set per operation, per architecture. You cannot read it off `Hoistable`.
The ops that need an uninit include at least tilize, untilize, reduce, and maybe transpose (not
confirmed), which is not the same set as `Hoistable=No`. Quasar is messier still, so we tag
`expect_uninit` on each op per arch rather than guess it.

```
expect_uninit : Ωᵢ × Arch → { Yes, No }        // explicit table, filled per op per arch
```

**Missing uninit.** Skip a required uninit and it shows up as `EXECUTED[Op] → INITIALIZED[*]`, so
marking that edge invalid catches it. Not a blind spot. The strict form (require an uninit for
every op, which is what the no-op uninits are for) catches every missing uninit without trusting
the `expect_uninit` table, at the cost of putting uninit hooks kernel-wide (code changes).

**Kernel compliance.** A kernel's per-EXU hook sequence `c₁;c₂;…` obeys the contract when every
transition `ρ/σ(c_n) → ρ/σ(c_{n+1})` is `ok` in the table, every `execute` passes F4, and every
`configure`/`reconfigure` passes the guards in §4. The first non-`ok` transition is the
violation. `WARN` edges obey the contract but waste work (perf defects).

---

## 9. How to extend

To add an operation: add its `xⱼ` field table (§2.6), register it in `Ωᵢ` (§2.3), and set its
`Hoistable` and `expect_uninit`. To add an operand field: add it to §2.4. §5 and §6 then follow,
and the checker reads its rules from those two sections.
