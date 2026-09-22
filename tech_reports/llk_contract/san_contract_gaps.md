# LLK Contract: Sanitizer & code gaps

Companion to [`llk-formal-contract.md`](./llk-formal-contract.md). The formal contract says how
things should behave. This doc tracks the places the code does not, yet:

- **Sanitizer bugs and rules not switched on:** §7 convergence items (C1–C6), §8 open
  questions (Q2–Q7).
- **What the Sanitizer cannot check by design:** §10 coverage gaps.
- **LLK code that breaks the contract:** §11 F3 inverse (`uninit` = `init⁻¹`) gaps, §12
  reconfig-escape (sticky cfg mode-bit) gaps.
- **Effort and next steps:** §13 mapping effort, §14 fuzzer audit.

Nothing here is frozen. The aim is one symbol, one place in the code, with the checker reading its
rules from formal §5 and §6. The list below is how far we still are from that.

Section numbers carry over from the original combined contract so cross-references keep working.
§1–§6 and §9 (words, state, semantics, the FSM, how to extend) live in the formal contract; the
sections below (§7–§8, §10–§14) live here. A `§n` reference points to whichever of the two docs
holds that number.

---

## 7. Convergence items (contract ↔ current code)

| id | gap | where |
|----|-----|-------|
| ~~C1~~ | **Resolved.** `sᵢ` is a tagged union with shared `yᵢ` (§2.2); `variant` (≤1 live) is right. | — |
| C2 | `execute` writes own fields instead of comparing them. Fix: assert-equal (own fields are set once, at init). | `operation_write`, A==Execute, impl.h |
| C3 | `σ` (`OperationStatus`) never set; `ρᵢ` set is a TODO. | `// sstanisic todo` in `state_operation_impl`/`state_operand_impl`, impl.h |
| C4 | operand drift is computed, then thrown away. No report. | `operation_write` else-branch, impl.h |
| C5 | the full FSM (§6) is `#if 0`; ordering is not enforced. Only the type-level guards (§4) run. | legacy hooks, impl.h/api.h |
| C6 | `DestWidth32` is assumed to be configure-only shared state. | `#47440`, legacy `unpack_operand_configure_impl` |

**On C2, why `execute` writes own fields today.** Checked on the branch (one squashed `temp`
commit, no message to go on). It is a stub, not a decision:

- The stated policy is the comment above `operation_write`: own field gets written; operand field
  gets snapshotted at init and compared after. The real drift check went onto the operand snapshot
  `y'ⱼ`, not onto `xⱼ`.
- The checking half is not built: `output.h` reporting is `#if 0`, the FSM is `#if 0`, `status` is
  unset, and the operand compare's result is dropped (C4). Nothing catches anything yet.
- `StateStruct::update` is meant to overwrite (`test_state_functional.cpp` asserts "Overwriting a
  known field tracks the new value"). The `execute<Tilize>(BlockCtDim, …)` in `valid_operation.cpp`
  is a compile-time guard positive (execute accepts own fields), not a runtime test.

Writing is a safe stub while checking is off: a wrong write only misses a catch. Own fields are set
once (Q7), so the plan is to compare at execute. It waits on the reporting path (C4).

---

## 8. Open questions

- **Q2** Cross-EXU sync (producer→consumer, U→F→P) is not in `State` at all. In scope for this
  contract, or a separate one?
- **Q3** Fill the `expect_uninit : Ωᵢ × Arch → {Yes,No}` table. Needs uninit: tilize, untilize,
  reduce, transpose(?). Handle Quasar on its own.
- **Q4** Should the checker see the raw LLK params (to verify `g_q`), or only the hook boundary
  where `g_q` is the identity?
- **Q5** Do `WARN` (perf) edges count as violations in the obeys-vs-violates number, or only
  `ERROR`?
- **Q6** List `Scratch` exactly, per arch (MOP, SETADC/SETADCXX, ADDRMOD, …), so F4's exemption set
  is closed and auditable.
- **Q7 (resolved)** Every own field is set once, at init; execute compares. Matmul
  `CtDim`/`RtDim`/`KtDim` are one set per init/execute pair. New dims need a new `init`, because the
  dims program the MOP and the REPLAY buffer. No own field is known to change at execute.

---

## 10. Coverage gaps (what the Sanitizer cannot check by design)

The Sanitizer flags bad usage and stays quiet on good usage. "Quiet" hides two cases: the state
decides correctness and it is fine (**true OK**), or correctness rides on something the tool does
not hold, so it is quiet either way (**blind**). A third verdict, `UNVERIFIABLE`, names the second.
Usability is then a number:

```
certainty = 1 − (blind sites) / (state-changing calls)
```

### 10.1 What counts as a blind spot

Test: a real blind spot is one that a *correctly built and kept-up* Sanitizer still cannot catch.
Anything a correct mask, config, or FSM would catch is a process slip or a plain bug, not a design
gap.

| gap | class | verdict |
|-----|-------|---------|
| **G1** | **no history:** no pre-init baseline, so `uninit`-restore cannot be checked | **real, in scope** |
| G2 | masking: `y'ⱼ` keeps only `maskⱼ(yᵢ)` | not a gap. A correct mask covers what init/execute read; a stale mask is a maintenance miss |
| G3 | unknown (`⊥`): field read before any `configure` set it | not a gap. The known-bit is clear, so it is catchable as use-before-set |
| G4 | out-of-model: `Scratch` (MOP/SETADC/ADDRMOD); cross-EXU sync | `Scratch` is safe by design (every `execute` re-sets it). Cross-EXU sync is a real gap, but out of the model today (Q2) |

So the one in-scope, countable design gap is **G1**. Cross-EXU sync (G4) is real but out of scope
until `State` models inter-thread order.

### 10.2 G1 measured on `origin/main`

Commit `e71a69bbc5a`, 435 compute kernels. Static scan of each kernel's call sequence, §6 FSM laid
over it. A `uninit` is G1 when the FSM stays quiet (no verdict) **and** downstream code leans on the
operand state across the `uninit` with no `configure`/`reconfig` to set it up again (the
`Uninit; Init/Execute` with nothing in between).

State-changing calls (the denominator): **3014**, split as init 1870, reconfig 671, configure 368,
uninit 105. (Execute op calls, for context: 4844.)

Of 105 `uninit` sites:

| | count |
|---|---|
| FSM-catchable (tool objects, not blind) | 15 |
| FSM-quiet, restore not leaned on (verifiable) | 45 |
| **G1 (FSM-quiet, restore leaned on, blind)** | **45** |

```
certainty ≥ 98.5%   (2969 / 3014 state-changing calls)
G1 blind  ≤ 1.5%    (45 uninit sites; the only in-scope gap)
```

**45 is an upper bound.** Whether a *following init* really sets the operand up again, or only
re-snapshots a value that was never restored, rides on per-op field writes you cannot see in kernel
source. True G1 ≤ 45; certainty ≥ 98.5%.

Closing G1 (a colleague's call, a design change): push `(op, full operand snapshot)` at `init`, pop
and diff at `uninit`. That turns every G1 site from `UNVERIFIABLE` into `OK`/`ERROR`.

### 10.3 Method and limits

- Regex over comment-stripped source; sort calls by name suffix; template-arg lists handled; §6 FSM
  at op-type grain.
- **Limit 1**: a whole-file linear scan mixes helper-definition order with call order (minor:
  init/uninit stay next to each other inside a helper).
- **Limit 2**: cross-TU init/uninit is invisible to a per-file scan.
- **Limit 3**: operand/CB identity (the literal cb1/cb2 case) is not counted; comparing positional
  args needs per-function signatures (`init(in,out)` vs `uninit(out)` differ for a dull reason).
  10.2 uses the signature-free reliance test instead.
- Script: `tools/analysis/blindspot.py` (to be committed).

Not measured: G4 cross-EXU sync (out of model). G2/G3 ruled out by 10.1.

---

## 11. Contract-conformance gaps (F3: `uninit` = `init⁻¹`)

Not the same question as §10. Here it is not "can the tool see it" but "does the LLK code keep the
contract." F3 asks `uninit` to put the EXU back where it was before `init`, but only for
**non-hoistable** ops. For a hoistable op the next `init` sets everything up again, so a no-op
`uninit` is fine.

Measured by two per-arch sage swarms driven off the state-audit map (`tools/llk_state_audit/`), each
agent handed the audit's persistent-write list for the `init` and asked whether `uninit` reverts
each entry. Every WH+BH `init`/`uninit` pair was covered.

```
44 WH+BH pairs verified →  6 gaps  |  34 benign (hoistable)  |  4 correct
```

Quasar has **no `uninit` functions** at all. It does not use the teardown pattern, so F3-inverse
does not apply; QSR resets through `canonical_reset` config helpers instead. So ~34/44 uninits do
not fully restore, but that is benign (hoistable: the next `init` sets it up again); **6/44 are real
bugs.** "Not a true inverse" is common; "must be and isn't" is rare.

### The 6 confirmed gaps

| arch | function | leaked state | why it bites |
|------|----------|--------------|--------------|
| BH | `_llk_unpack_untilize_uninit_` | SrcA Y-stride set to `FACE_C_DIM·face_r_dim·x` = 16× canonical (`llk_unpack_untilize.h:152,166`) | `uninit` writes a wrong value on purpose; WH save/restores correctly (WH/BH divergence). Deprecated op (tt-metal#22904) |
| WH | `_llk_unpack_tilizeA_B_uninit_` | `Tile_x_dim_cntx0` always `FACE_DIM_16x16` (`:592`) | wrong for `face_r_dim<16` (tiny tiles); the sibling `_llk_unpack_tilize_uninit_` was fixed (`:576` `canonical_unpA_tile_x_dim_cntx`), this one was not |
| BH | `_llk_math_fast_tilize_uninit_` | `DEST_ACCESS_CFG` `remap_addrs` + `swizzle_32b` never cleared | see the systemic note below |
| BH | `_llk_math_fast_untilize_uninit_` | same `DEST_ACCESS_CFG` remap/swizzle (no-op uninit) | see the systemic note below |
| BH | `_llk_pack_fast_untilize_uninit_` | `DEST_TARGET_REG_CFG` offset + `DEST_OFFSET_LO/HI` GPRs left at bottom-strip phase | next pack inherits a stale DEST offset |
| BH | `_llk_unpack_AB_reduce_block_max_row_uninit_` | `ALU_ACC_CTRL_Zero_Flag_disabled_src` left at 1 (fp32 path) | not cleared or restored; WH sibling is benign (WH/BH divergence) |

**Systemic: the BH DEST remap ownership gap (2 of the 6).** `fast_tilize`/`fast_untilize` init
call `_llk_math_reconfig_remap_(true)` (`llk_math_fast_tilize.h:35`, `llk_math_fast_untilize.h:67`),
but **no `_(false)` call exists anywhere in the BH tree**. The comments even disagree:
`llk_math_fast_tilize.h:99` says "DEST remap is cleared by pack uninit", `llk_pack_fast_tilize.h:282`
says "DEST remap is NOT cleared here — owned by the math thread". Each thread trusts the other to
clear it, and neither does. A later op reading DEST with linear addressing gets remapped data.

Notes: 3 of the 6 gaps live in `experimental/` fast-(un)tilize paths; 3 are WH/BH splits that the
per-arch swarm surfaced (untilize WH-ok/BH-gap; tilizeA_B WH-gap/BH-ok; reduce_block_max_row
WH-benign/BH-gap).

---

## 12. Reconfig-escape gaps (sticky cfg mode-bits)

A third class, out of the audit's biggest persistent bucket (1479 `cfg_register` effects "retained
until reconfigured"). A gap here is a behavioral cfg bit (accumulate, relu, format-override,
DEST-remap) that an op sets away from default and that nothing resets short of a full
`hw_configure`, so a later op inherits the wrong mode. Not the same as §11: this covers ops with no
`uninit`, and Quasar (which has no uninit pattern).

Method: from the 1479, drop the self-refreshing address/stride/counter regs (recomputed by every op
that uses them) and the bits the baseline `hw_configure` sets up again; keep behavioral bits with no
in-body reset to default. That leaves 13 one-way candidates (WH 5, BH 2, QSR 6). A per-arch sage
swarm (incl. `sage-quasar`) then checked each: fixed vs parameterized setter, and whether any
reset/disable path exists.

```
13 candidates →  3 confirmed gaps  |  8 benign  |  2 swarm false-positives (overturned)
```

Confirmed:

- BH `DEST_ACCESS_CFG_remap_addrs` + `swizzle_32b`: the **same two bits as §11's fast-(un)tilize
  gaps**, re-found here from the no-uninit angle (`_llk_math_reconfig_remap_(true)` set, no
  `(false)` caller anywhere). Method check, not a new gap.
- **QSR `THCON_PACKER0_REG3_PACK_STRIDE_NO_WRITE` (new)**: set to a fixed `1` in the small-tile
  branch of `_llk_pack_untilize_strided_init_` (`llk_pack_untilize.h:271`); no write of `0` exists
  anywhere in `tt_llk_quasar`, and `_llk_pack_hw_configure_` touches only REG0. A following PACKER0
  op inherits row-write suppression.

Benign (8): WH `Pack_L1_Acc` ×4 and `ALU SrcA_val` (reset via `hw_configure` or a dedicated clear);
QSR PACKER0 `L1_ACC` / `RELU_MODE` / `EDGE_MASK_MODE` (set up again each `_llk_pack_init_`, which
hardcodes the PACK0 branch, `llk_pack.h:73`).

**Note on swarm reliability.** The swarm flagged QSR PACKER1 `L1_ACC` and `RELU_MODE` as gaps while
calling their PACKER0 twins benign, but the code is structurally the same (`PACK_SEL`-branched,
lines 212/216 and 255/260). Reading the source overturned both: `_llk_pack_init_` only ever builds
the PACK0 branch, so PACKER1's RELU is never written on the standard path, and L1_ACC is
caller-managed the same way for both packers. On Quasar (Sonnet, no DeepWiki) the swarm needs a
same-code consistency cross-check; two opposite verdicts on twin resources are the tell.

**Running total across classes: §11 F3 (6) + §12 reconfig-escape (1 new) = 7 distinct confirmed
gaps** (the 2 BH DEST bits are shared between the classes, counted once).

---

## 13. Effort to map `x̂_j` and `y'_j`

Building the Sanitizer's per-operation state map means, for each op `j`: its own fields `x_j` and
the operand fields it leans on, `y'_j` (§2.5). Sized against the state-audit map.

**The catch.** The audit records state *writes* (effects), but `y'_j` is a *read* dependency: "op
`j` reads `φ`, so snapshot it." So the audit hands you `x_j` (own writes) and `init`'s operand
*writes* for free, but the `y'_j` read-deps are the manual residual. That is the hard part: when
neither `init` nor `execute` takes `φ` as a parameter, the dependency only shows up if you read the
body and its helpers.

`x_j`: basically auto-mapped, since it is the audit's effects table (5463 rows).

`y'_j`: splits by whether `init` even takes the operand field as a parameter:

| arch | init fns | `init` takes an operand param (explicit `y'_j`) | `init` does not (implicit → read code) |
|------|----------|------------------------------------------------|----------------------------------------|
| WH   | 49 | 18 (37%) | 31 |
| BH   | 52 | 20 (38%) | 32 |
| QSR  | 25 | **0 (0%)** | **25** |

So **~88 operation definitions** need a manual code read to recover operand dependencies. Two things
sharpen the picture:

- **Quasar is fully implicit:** 0/25 QSR inits take an operand-format or face param; it threads
  `TensorShape`/`buf_desc_id` by meaning, so every QSR op's `y'_j` is buried in the body.
- **62% of `cfg_register` effects are `parameter.kind = fixed`** (917/1479, over 146 functions,
  median 4/fn). That is a state write the parser could not pin to any argument, the real size of
  "not readable from the signature."

The `init-does-not` bucket (88) splits again into "execute takes it" (medium: pair init/execute) and
"neither" (hard: trace body plus helpers); QSR's 25 are all the hard kind.

**Reframe.** ~88 (WH+BH implicit) plus all QSR defs, one focused code read each, which is exactly
the shape of the per-op, per-arch sage swarms from §11–§12. So the mapping is **~1–2 swarm passes**,
not a hand audit; the 37% where `init` takes the operand params come free from the signature.

---

## 14. Fuzzer audit (independent, all FSM-valid sequences)

A generator (`tools/analysis/fuzz_sequences.py`) walks every FSM-valid call sequence over
`{CFG, RCFG, INIT, EXE, UNI}` × operand `{A, B}`, length ≤ 6: 60 skeletons → 584 distinct scenarios
(incl. the `cb1/cb2` example). An independent AI auditor, given only §1–§9 plus a blind-spot rubric
(§10–§13 held back), labelled each `covered` / `blind` / `invalid`.

```
584 →  342 covered | 37 invalid | 205 blind   (no_history 107, ground_truth 98, cross_exu 0)
```

Do not read 205 as "35% of usage is blind." The space weights every ordering the same, including
sequences no sane kernel would write. The point is that the blind set boils down to **two structural
gaps**:

- **Operand identity (`no_history`, 107).** The operation record is keyed by op *type*, not
  operand/CB identity, so `UNI_B` on an `INIT_A` record (teardown of the wrong operand) is
  accepted. This is G1 blown up to the `cb1/cb2` case; the tool keeps no operand-ownership baseline.
- **Operand-vs-tracked-`y` consistency (`ground_truth`, 98; 71 use operand B with no `CFG_B`).**
  `init` snapshots the operand requirement and `execute` checks against that snapshot
  (self-consistent), but nothing checks the requirement against tracked `y_i` and its known-bits. So
  using an operand that was never configured, or wrongly configured, passes in silence.

What this means:

1. Both gaps close with mechanisms already named: (a) put operand identity into the operation
   record; (b) add the `init`/`execute` operand-args-vs-tracked-`y_i` check that §4 skips today and
   legacy `unpack_operand_check` had (see §11 C2 and the missing-reconfig analysis). With both, the
   205 slide toward covered.
2. It also **puts a condition on §10's "certainty ≥ 98.5%":** that number assumed the
   operand-consistency check works and set operand identity aside. The fuzzer, reasoning from §4 *as
   written*, shows those two are load-bearing. Without them the abstract blind fraction is 35%, and
   real usage is more exposed than 98.5% let on.

`cross_exu` = 0 only because a single-EXU alphabet cannot say anything about cross-thread order;
probing G4b needs a multi-EXU fuzzer. Caveat: the auditor is Sonnet, so the verdicts are
contract-reasoning estimates, backed by the two classes matching findings we derived
independently, not checked line by line against source.
