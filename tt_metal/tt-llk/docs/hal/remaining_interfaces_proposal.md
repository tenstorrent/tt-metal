# HAL Interface Proposal — Remaining Instruction Groups

**Status:** proposal only — interface shapes, naming, and placement. No implementation.
**Scope:** the Blackhole Tensix instructions not yet covered by `tt_llk_blackhole/llk_lib/hal/` — RWCs, Src/Dest register features, data movement, scalar/ThCon operations, data-valid handoff, pack, NOPs, and miscellaneous instructions.
**Sources:** tt-isa-documentation (DeepWiki), `instructions/assembly.yaml`, `common/inc/ckernel_instr_params.h`, current LLK call sites.

---

## 1. Design language — what the new interfaces reuse

The existing HAL has three established patterns. Every proposal below picks one of them; no new pattern is introduced.

| Pattern | Used by | When it fits |
|---|---|---|
| **Descriptor struct** — aggregate with defaulted fields, `is_valid()`, compile-time and runtime `get_operation()`, compile-time and runtime `run()` | `hal::fpu`, `hal::unpack` | One instruction (or a small canonical family), several orthogonal fields |
| **Fluent compile-time builder** — selector chain folded into template parameters, emitter picks the minimal instruction sequence | `hal::ADC` / `hal::RuntimeADC` (`address_counters`) | Several counters/fields that hardware lets one instruction set together, where the emitter should choose the cheapest encoding |
| **Free functions over operand identity types** — `hal::Gpr<Index>`, `hal::gpr_ops::Immediate`, compile-time and runtime overloads | `hal::sync`, `hal::gpr_ops` | Verb-like operations on named resources (post, acquire, add) |

Cross-cutting rules kept from the existing headers:

- Every operation offers `get_operation()` so it can be embedded in a MOP template or replay buffer — this is non-negotiable for anything the math/unpack/pack inner loops touch.
- Compile-time path validates with `static_assert`; runtime path with `LLK_ASSERT` / `__builtin_trap()`.
- Fields with no characterized meaning are encoded at their only characterized value and **not** exposed (precedent: DOTPV reserved bits, GAPOOL bit 19). The raw `TT_OP_*` / `TTI_*` macros remain the compatibility path for anything the HAL deliberately does not model.
- Selections with dangerous implicit defaults get an `Unset` sentinel that forces an explicit choice (precedent: `fpu::SourceRelease`).

---

## 2. Shared vocabulary additions

Three small additions, needed by several interfaces below. Proposed home: `hal/utils/` next to `gpr.h`.

```cpp
// utils/source.h — shared SrcA/SrcB selection vocabulary.
namespace hal
{

/** @brief Select one source register file. */
enum class Source : std::uint8_t
{
    SrcA,
    SrcB
};

/** @brief Select one or both source register files. */
enum class SourceMask : std::uint8_t
{
    None = 0x0,
    SrcA = 0x1,
    SrcB = 0x2,
    Both = 0x3
};

} // namespace hal
```

- `fpu::SourceRelease` moves here (or is aliased here) so `hal::rwc`, `hal::src`, and `hal::fpu` share one release vocabulary; `fpu.h` keeps a using-declaration for compatibility.

```cpp
// utils/gpr.h addition — 16-bit half of a GPR, used as the byte-offset operand of
// indirect loads/stores (LOADIND/STOREIND OffsetIndex = gpr_index * 2 + half).
namespace hal
{

enum class GprHalf : std::uint8_t
{
    Low,
    High
};

template <std::uint32_t Index>
struct GprHalfRef; // Gpr identity + half selector; compile-time and runtime forms like Gpr

} // namespace hal
```

```cpp
// utils/mmio.h — identity of one register in the 0xFFB0_0000 local-register window
// (TDMA/THCON registers, PC-buf, mailboxes). NOT the CFG space — that is hal::cfg.
namespace hal
{

struct MmioRegister
{
    std::uint32_t byte_address; // encoders validate 0xFFB00000 base, 4B alignment, 18-bit index range
};

} // namespace hal
```

---

## 3. Interfaces

### 3.1 `math_counters.h` — `hal::math_counters` / `hal::rwc` (SETRWC / INCRWC / SETIBRWC)

**Implemented model.** A compile-time fluent builder groups one set or increment value with a `Counters` mask. One entry can assign the same value to several counters, and repeated disjoint entries can assign different values. SrcA/SrcB/Dest state folds into one SETRWC or INCRWC. Bias uses SETIBRWC, so `apply()` may issue both instruction families while `get_operation()` statically requires exactly one.

```cpp
using hal::rwc::Counters;

// One SETRWC: SrcA = 4 and SrcB = 4.
hal::math_counters
    .set<Counters::SrcA | Counters::SrcB, 4>()
    .apply();

// SETRWC followed by SETIBRWC; get_operation() is intentionally unavailable.
hal::math_counters
    .set<Counters::SrcA | Counters::SrcB, 4>()
    .set<Counters::Bias, 0x120>()
    .apply();
```

**Decisions & constraints**

- `Counters` normalizes Bias as bit 3 and is reused for carry-shadow selections. `All` remains SrcA/SrcB/Dest (`0x7`) so it stays a one-instruction selection; `AllIncludingBias` is `0xf`.
- SrcA/SrcB/Dest values are 4-bit immediates (`< 16`); Bias is a 12-bit immediate (`< 4096`).
- Set and increment entries cannot be mixed, and a counter cannot be assigned twice in one chain.
- `release<>()`, `clear_fidelity()`, and `advance_dest_and_save_to_carry<>()` select SETRWC-only behavior. `advance_carry_and_reload<>()` requires a matching assigned counter.
- SETRWC BitMask bit 4 ("SFPU SP", yaml-only, absent from ISA docs, zero call sites) is **not exposed**; the bit is encoded 0.
- Bias is included for encoding completeness. Blackhole's raw headers expose SETIBRWC, but public functional documentation is absent and the current simulator marks it unsupported. Coverage therefore checks encoding and optimized instruction emission rather than simulated Bias behavior.
- The builder is compile-time-only. Unlike address counters, this instruction family does not require a parallel runtime builder for instruction selection; existing raw runtime paths remain available.

---

### 3.2 `src.h` — `hal::src` (source-register features, clear, data-valid handoff)

**Model.** Everything that manipulates the SrcA/SrcB register files *as a resource* lives together: zeroing, transpose, row shift, clock-gating reset, and the data-valid bank handoff. This keeps the whole bank-ownership story (who owns a bank, how it is returned, how it is published) in one header, abstracting the scheme the repo already uses (unpacker publishes via UNPACR handoff → math consumes → math releases via SETRWC/CLEARDVALID). Pattern: **descriptor structs** (fpu.h style).

```cpp
namespace hal::src
{

// ---------- Bank handoff (DValid) ----------

/** @brief Select whether the math thread also flips to the paired bank when releasing. */
enum class BankAdvance : std::uint8_t
{
    Flip,           // normal: hand bank to unpacker, math flips to the paired bank
    KeepReadingSame // CLEARDVALID KeepReadingSameSrc: give the bank away but keep reading it
};

/** @brief Release the current SrcA/SrcB bank(s) back to the unpackers (CLEARDVALID). */
struct Release
{
    SourceMask sources; // must be nonzero
    BankAdvance advance = BankAdvance::Flip;
};

/**
 * @brief Publish the unpacker-side write bank(s) to the math thread (SETDVALID).
 *
 * @note Constrained support on Blackhole: the ISA marks SETDVALID unsupported because the
 *       published bank's implied Src format becomes unpredictable. Every live call site is
 *       SrcB-only with math-supplied data (MOVD2B) and implied-format inference disabled.
 *       The validator allows both sources; the docstring carries the constraint, and the
 *       sanctioned general-purpose path is the unpacker-side publish (see unpack additions, 3.7).
 */
struct Publish
{
    SourceMask sources; // must be nonzero
};

// ---------- Clear ----------

enum class BankScope : std::uint8_t
{
    Current,
    All
};

/** @brief Select which bank "Current" means: the unpacker write bank or the math read bank. */
enum class CurrentBankView : std::uint8_t
{
    UnpackerWrite,
    MathRead
};

enum class Fill : std::uint8_t
{
    Zero,
    AllOnes // reads as -inf where relevant; GMPOOL identity
};

/** @brief Fill SrcA/SrcB bank(s) from the math thread (ZEROSRC). */
struct Zero
{
    SourceMask sources; // must be nonzero
    BankScope scope           = BankScope::Current;
    CurrentBankView view      = CurrentBankView::MathRead;
    Fill fill                 = Fill::Zero;
};

// ---------- Register features ----------

/**
 * @brief In-place 16x16 transpose of SrcB rows 16-31 of the current math bank (TRNSPSRCB).
 *
 * @note Waits for SrcB data-valid; does not use or advance the SrcB RWC. Stage data into the
 *       upper half first (move::DestToSrcB with row offset 16).
 */
struct TransposeSrcB
{
};

enum class ShiftFill : std::uint8_t
{
    Rotate, // datum 0 wraps into datum 15
    Zero    // zero injected into datum 15
};

/** @brief Shift one SrcB row left by one datum (SHIFTXB). */
struct ShiftRowSrcB
{
    std::uint8_t row;
    ShiftFill fill                = ShiftFill::Zero;
    std::uint8_t address_modifier = 0;
};

/** @brief Reset SrcA/SrcB pipeline clock gating to "don't gate" (GATESRCRST). */
struct GatingReset
{
    SourceMask sources; // must be nonzero
};

// Each descriptor: is_valid() + get_operation() (CT/RT) + run() (CT/RT).

} // namespace hal::src
```

**Decisions & constraints**

- **CLEARDVALID's Reset bit is not exposed.** The ISA marks it `UnsupportedFunctionality` on Blackhole (risk of nondeterministic hangs); it is encoded 0. Debug full-reset flows keep the raw macro.
- CLEARDVALID bits 17:15 look like an addr_mod to the predecoder but are unused — encoded 0, not exposed.
- `SourceRelease` inside `fpu` descriptors and SETRWC remains the *preferred* release path in hot loops (zero extra instructions); `src::Release` is the standalone form. The docstrings cross-reference this.
- `hal::fpu` and `hal::rwc` docs point at `src::Release`/`src::Publish` as the single explanation of the bank-handoff protocol.

---

### 3.3 `dest.h` — `hal::dest` (Dest zeroing / zero-flags — ZEROACC)

**Model.** Descriptor struct. ZEROACC does not write datums — it marks rows undefined via zero-flags (packer reads 0, FPU reads identity) or, with the flag action inverted, re-marks rows initialized. The interface names that honestly instead of pretending it is a memset.

```cpp
namespace hal::dest
{

/** @brief Select how many Dest rows one ZEROACC touches. */
enum class ZeroScope : std::uint8_t
{
    SingleRow, // current Dst counter + index (+ configured base offsets)
    Face,      // 16 rows: index selects the face (32 rows in 32-bit mode)
    Half,      // 512 rows: index bit 0 selects low/high half
    All        // all 1024 rows; index ignored
};

/** @brief Select whether rows are marked undefined or re-marked initialized. */
enum class ZeroFlagAction : std::uint8_t
{
    MarkUndefined, // normal zeroing
    ClearFlags     // clear_zero_flags = 1; re-arm rows (budabackend#2730 mitigation path)
};

/** @brief Select 16-bit or 32-bit Dest row addressing. */
enum class DestWidth : std::uint8_t
{
    Bits16,
    Bits32
};

struct Zero
{
    ZeroScope scope;
    std::uint16_t index           = 0; // row / face / half selector per scope
    ZeroFlagAction flags          = ZeroFlagAction::MarkUndefined;
    DestWidth width               = DestWidth::Bits16;
    std::uint8_t address_modifier = 0; // applied by hardware only for SingleRow/Face
};

} // namespace hal::dest
```

**Decisions & constraints**

- Bits 23:22 of ZEROACC clear SrcA/SrcB data-valid (yaml warning) — **encoded 0, never exposed**; releases go through `src`/`rwc`/`fpu`.
- The WH-heritage 3-bit `p_zeroacc::CLR_*_32B` encodings are not carried over; 32-bit-ness is the explicit `DestWidth` field (bit 18), which is what BH LLK already does.
- The "apply addr_mod, zero nothing" trick (`Face` scope with an out-of-range index) is documented on the descriptor rather than given its own canonical type — revisit if call sites want it.
- Alternative placement considered: `hal::fpu` (ZEROACC executes on the math unit). Rejected: `dest.h` keeps Dest-state management discoverable next to `src.h`, and the FPU header stays purely arithmetic. See open question 8.4.

---

### 3.4 `move.h` — `hal::move` (the movement matrix)

**Model.** One namespace expressing the user-visible concept: *move data from endpoint X to endpoint Y*, where endpoints are Dest, SrcA, SrcB, L1, GPRs, MMIO, and the mover's L0. One descriptor per route (fields differ too much for a single generic descriptor), all following the fpu.h trio. The route table is the interface's homepage diagram:

| Route | Instruction | Executes on | Waits for source validity? |
|---|---|---|---|
| Dest → SrcA | MOVD2A | Matrix Unit | no — caller stalls (`sync::wait`) |
| Dest → SrcB | MOVD2B | Matrix Unit | no — caller stalls |
| SrcB → SrcA | MOVB2A | Matrix Unit | no — caller stalls |
| SrcA → Dest | MOVA2D / MOVDBGA2D | Matrix Unit | yes / **ignored** (validity override) |
| SrcB → Dest | MOVB2D / MOVDBGB2D | Matrix Unit | yes / **ignored** (validity override) |
| L0 ↔ L1 | XMOV | Mover (TDMA) | n.a. — posted |
| L1 → GPR | LOADIND | Scalar unit (ThCon) | n.a. — posted, async write-back |
| GPR → L1 | STOREIND (L1) | Scalar unit | n.a. — posted |
| GPR → SrcA/SrcB | STOREIND (reg space) | Scalar unit | n.a. — posted |
| GPR → MMIO | STOREREG | Scalar unit | n.a. — posted |
| MMIO → GPR | LOADREG | Scalar unit | n.a. — posted, async write-back |

#### Matrix-unit routes

```cpp
namespace hal::move
{

/** @brief Rows moved per issue; legal counts differ per route and are validated per descriptor. */
enum class RowCount : std::uint8_t
{
    One,
    Four,
    Eight
};

/** @brief Select the low or high 16-bit half in 32-bit Dest mode. */
enum class DestHalf : std::uint8_t
{
    Full, // 16-bit Dest
    Low32 // dest_32b_lo
};

/**
 * @brief Select whether a Src-to-Dest move honors the bank data-valid handshake.
 *
 * Ignore selects the MOVDBG* encoding — same datapath with the AllowedClient wait removed.
 * Required when reading a bank no unpacker will ever publish (debug dumps, self-staged data);
 * pair with disabled implied-format inference.
 */
enum class SourceValidity : std::uint8_t
{
    Wait,
    Ignore
};

struct DestToSrcA
{
    std::uint16_t dest_row;       // Dest row read (RWC-relative)
    std::uint8_t srca_row;        // SrcA row written; 4-row moves align down
    RowCount rows                 = RowCount::One; // One | Four
    std::uint8_t address_modifier = 0;
    DestHalf dest_half            = DestHalf::Full;
};

struct DestToSrcB
{
    // same shape; srcb_row. Row 16 offset is the TransposeSrcB staging idiom.
};

struct SrcBToSrcA
{
    std::uint16_t srcb_row;
    std::uint8_t srca_row;
    RowCount rows                 = RowCount::One; // One | Four
    std::uint8_t address_modifier = 0;
};

struct SrcAToDest
{
    std::uint8_t srca_row;
    std::uint16_t dest_row;
    RowCount rows                 = RowCount::One; // One | Eight
    std::uint8_t address_modifier = 0;
    DestHalf dest_half            = DestHalf::Full;
    SourceValidity validity       = SourceValidity::Wait; // Ignore => MOVDBGA2D
};

/** @brief SrcB-to-Dest transfer shape (p_movb2d modes: row counts x datum-0 broadcast). */
enum class BroadcastShape : std::uint8_t
{
    OneRow,
    OneRowDatum0,
    EightRowsBroadcast,
    EightRowsBroadcastDatum0,
    FourRows,
    FourRowsDatum0
};

struct SrcBToDest
{
    std::uint8_t srcb_row;
    std::uint16_t dest_row;
    BroadcastShape shape          = BroadcastShape::OneRow;
    std::uint8_t address_modifier = 0;
    DestHalf dest_half            = DestHalf::Full;
    SourceValidity validity       = SourceValidity::Wait; // Ignore => MOVDBGB2D
};

} // namespace hal::move
```

Docstring obligations (this is where the HAL earns its keep — every one of these is tribal knowledge today):

- Writes **into** Src never auto-wait: `@note` prescribing the `sync::wait::stall<..., SrcAValid/SrcBValid>` guard (the MOVB2A/MOVD2B comment in `llk_math_transpose_dest.h` becomes interface documentation).
- Reads from Src auto-wait unless `SourceValidity::Ignore`.
- None of these set or clear data-valid; releases are `src::Release` / addr-mod clear / `SourceRelease`.
- Implied-format caveat when touching a non-valid bank (`DISABLE_IMPLIED_SRCA/B_FMT`).
- MOVA2D/MOVB2D 4-cycle result latency note.
- Field-name trap resolved by construction: descriptors name rows by register file and direction (`dest_row` read vs `srca_row` written), unlike the raw macros' `src`/`dst` args.

#### Mover route (XMOV)

```cpp
namespace hal::move
{

enum class MoverDirection : std::uint8_t
{
    L0ToL1, // memset-style fill toward L1
    L1ToL0,
    L0ToL0,
    L1ToL1  // memcpy within L1
};

/**
 * @brief Trigger one Mover transfer (XMOV). Source, destination, and size are pre-programmed
 *        THCON configuration (16-byte units); the instruction only launches and finalizes.
 */
struct MemoryCopy
{
    MoverDirection direction;
    bool last = true; // flush mover accumulation buffers on the final XMOV of a task
};

} // namespace hal::move
```

Scope note: this covers the Tensix-issued XMOV only. The RISC-MMIO mover path (`ckernel_xmov.h`) is a different access mechanism; whether HAL absorbs it is open question 8.6.

#### Scalar-unit routes

```cpp
namespace hal::move
{

enum class TransferSize : std::uint8_t
{
    Bytes16,
    Bits32,
    Bits16,
    Bits8
};

enum class OffsetIncrement : std::uint8_t
{
    None,
    Bytes2,
    Bytes4,
    Bytes16
};

/** @brief Indirect L1 load into GPR(s) (LOADIND). Address = base*16 + offset half-GPR. */
template </* Gpr/GprHalf identity indices */>
struct L1ToGpr
{
    // hal::Gpr base (L1 address in 16-byte units), hal::GprHalfRef offset (byte offset),
    // hal::Gpr data (destination; 16-byte loads fill an aligned quad),
    // TransferSize size, OffsetIncrement post_increment
};

struct GprToL1 { /* mirror of L1ToGpr (STOREIND, L1 space) */ };

/** @brief Scalar-unit store into a source register file (STOREIND register space). */
struct GprToSrc
{
    // Source target (SrcA/SrcB); Bytes16 / Bits32 only
};

struct GprToMmio { /* MmioRegister + hal::Gpr data (STOREREG) */ };
struct MmioToGpr { /* MmioRegister + hal::Gpr data (LOADREG) */ };

} // namespace hal::move
```

Docstring obligations: all scalar routes are **posted** — the instruction retires when the request is issued; GPR write-backs (L1ToGpr, MmioToGpr) land asynchronously. Each descriptor's `@note` names the drain: `sync::wait::stall<..., StallCondition::ScalarIdle>` before consuming results. STOREIND's L1 size encoding differs from LOADIND's (yaml is authoritative); the shared `TransferSize` enum hides that asymmetry in the encoders.

---

### 3.5 `atomic.h` — `hal::atomic` (L1 atomics: descriptors implemented, object layer proposed)

**Model — two layers.** The descriptor layer (`FetchIncrement`, `FifoAcquire`, `CompareSet`, `MaskedWrite`, sketched below) is implemented in `atomic.h` and stays as the encoding layer. On top of it, the proposed *public* surface is **objects + verbs** (pattern 3): these four instructions operate three recurring concepts — a counter cell, a hardware-managed FIFO, a lock cell — plus one masked line store, and `hal::sync` already models the on-core versions of the same concepts as identity + verb (`mutex::acquire`/`release`, `semaphore::post`/`get`) rather than SEMINIT/ATGETM-shaped wrappers. The identities are stateless typed address bundles; every verb delegates to exactly one descriptor, so nothing implemented is discarded and `get_operation()` embedding still holds. ATSWAP sits here rather than in `move` because it shares the atomic line-RMW model and mask semantics (open question 8.3 records the alternative).

```cpp
namespace hal::atomic
{

// ---------- Object identities (stateless; the cell's layout lives here, once) ----------

/** @brief A WidthBits-wide wrapping counter in one 32-bit word of a 16-byte L1 line. */
template <
    std::uint8_t WidthBits, // [1, 32]; the encoder owns WrapVal = WidthBits - 1
    std::uint32_t AddressIndex>
struct Counter
{
    hal::Gpr<AddressIndex> address; // 16-byte-line address
    std::uint8_t word = 0;          // 32-bit word within the line, [0, 3]
};

/** @brief Select the FIFO condition that blocks a readiness wait. */
enum class FifoState : std::uint8_t
{
    Empty,
    Full
};

/** @brief A hardware-managed FIFO: 16-byte-aligned FIFOControl with read/write counters. */
template <
    std::uint8_t CapacityLog2, // capacity 2^N; the encoder owns the (N+1) & 0xF width encoding
    std::uint32_t AddressIndex>
struct Fifo
{
    hal::Gpr<AddressIndex> control_address;
};

/** @brief A 4-bit lock/flag cell in one 32-bit word, operated by ATCAS. */
template <std::uint32_t AddressIndex>
struct Lock
{
    hal::Gpr<AddressIndex> address;
    std::uint8_t word = 0;
};

// ---------- Verbs (each delegates to one implemented descriptor; CT/RT + get_operation) ----------

// ATINCGET — posted; the previous value lands in `data` asynchronously, drain before reading.
fetch_add(counter, data /* hal::Gpr — in: addend, out: previous value */);

// ATINCGETPTR — BLOCKING (hardware retry loop); returns the selected pointer.
wait_while<FifoState>(fifo, result /* hal::Gpr; pointer is not advanced */);
pop_slots(fifo, result /* old rd */, increment_log2 = 0);
push_slots(fifo, result /* old wr */, increment_log2 = 0);

// ATCAS — BLOCKING until the compare succeeds; mirrors the sync::mutex on-core vocabulary.
acquire(lock);                   // {compare 0, set 1}
release(lock);                   // {compare 1, set 0}
compare_set(lock, compare, set); // generic form, 4-bit immediates

// ATSWAP — posted masked 16-byte store; a plain verb, no object (it is a store, not a protocol).
store_masked(address /* hal::Gpr */, data /* 4-aligned quad base */, granule_mask /* 8 x 16-bit */);

} // namespace hal::atomic
```

**Why the object layer earns its place over the bare descriptors**

- **The cell's layout is single-sourced.** With bare descriptors, every call site restates `word_select` and `counter_width` (or `capacity_log2`) — two sites can silently disagree about the same L1 cell, a protocol bug no per-instruction validator can see. An object declares the layout once (typically `constexpr`) and every verb on it inherits it.
- **Human quantities replace encoding quantities.** The implementation already contains the tells: `counter_width - 1u` and `(capacity_log2 + 1u) & 0xfu` — two mutually inconsistent wrap encodings (ATINCGET masks at `(2 << WrapVal) - 1`, ATINCGETPTR at `(1 << WrapVal) - 1`, plus the 2^15 zero-width special case). `Counter<WidthBits>` and `Fifo<CapacityLog2>` keep that arithmetic in one place.
- **Completion contracts attach to verbs**, where they belong: `fetch_add`/`store_masked` are posted (drain per the existing `@note`s); `wait_while`/`pop_slots`/`push_slots`/`acquire` block in a hardware retry loop.
- **One lock vocabulary across scopes:** `acquire`/`release` on `atomic::Lock` (cross-core, L1) reads identically to `sync::mutex::acquire`/`release` (on-core), while `compare_set` stays available for non-lock protocols.

**The implemented descriptor layer (encoding surface):**

```cpp
namespace hal::atomic
{

/**
 * @brief Atomic fetch-and-add on one 32-bit word (ATINCGET). Posted; the pre-increment value
 *        lands in the data GPR asynchronously — drain before reading.
 */
template <
    std::uint32_t AddressIndex,
    std::uint32_t DataIndex>
struct FetchIncrement
{
    hal::Gpr<AddressIndex> address; // 16-byte units
    hal::Gpr<DataIndex> data;       // increment in, original value out
    std::uint8_t word_select   = 0; // which 32-bit word of the 16-byte line, [0, 3]
    std::uint8_t counter_width = 32; // wrap width in bits, [1, 32] (encodes WrapVal = width - 1)
};

/** @brief Select which FIFO pointer an ATINCGETPTR manipulates. */
enum class FifoPointer : std::uint8_t
{
    Read, // blocks while empty
    Write // blocks while full
};

/**
 * @brief Atomic FIFO-pointer acquire (ATINCGETPTR). BLOCKING: hardware retries until the FIFO
 *        condition clears. The pre-increment pointer value lands in the data GPR.
 */
template <
    std::uint32_t AddressIndex,
    std::uint32_t DataIndex>
struct FifoAcquire
{
    hal::Gpr<AddressIndex> address; // FIFOControl structure, 16-byte units
    hal::Gpr<DataIndex> data;       // old pointer out
    FifoPointer pointer;
    std::uint8_t capacity_log2;      // [0, 15]; FIFO capacity 2^N; counters are (N+1)-bit
    std::uint8_t increment_log2 = 0; // post-increment by 1 << value
    bool probe_only             = false; // NoIncr: still blocks, but leaves the pointer unchanged
};

/**
 * @brief Atomic compare-and-set of one 32-bit word against 4-bit immediates (ATCAS).
 *        BLOCKING: hardware retries until the compare succeeds. Mutex idiom:
 *        acquire = {compare 0, set 1}; release = {compare 1, set 0}.
 */
template <std::uint32_t AddressIndex>
struct CompareSet
{
    hal::Gpr<AddressIndex> address;
    std::uint8_t word_select = 0;
    std::uint8_t compare;    // [0, 15]
    std::uint8_t set;        // [0, 15]
};

/**
 * @brief Masked 16-byte write from a 4-aligned GPR quad (ATSWAP — nothing is returned;
 *        "swap" names the granule replacement, not an exchange). Posted.
 */
template <
    std::uint32_t AddressIndex,
    std::uint32_t DataIndex>
struct MaskedWrite
{
    hal::Gpr<AddressIndex> address;
    hal::Gpr<DataIndex> data; // first of four consecutive, index % 4 == 0
    std::uint8_t granule_mask; // 8 x 16-bit write-enable granules
};

// Compile-time path
template <auto Operation>
constexpr std::uint32_t get_operation();

template <auto Operation>
void run();

// Runtime path: typed get_operation(operation) and run(operation) overloads for every descriptor.

} // namespace hal::atomic
```

None of these have BH LLK call sites today — the intended consumers are metal-side FIFO/semaphore code, which is exactly the audience that benefits from the conceptual surface. The implemented layer is L1-only and fixes bit 23 to zero: the authoritative ISA reserves that bit, despite local `assembly.yaml` describing a conflicting `MemHierSel` field. If the verb layer lands, the descriptors can retire into `detail` in a later cleanup (open question 8.9); until then the verbs are sugar-free one-instruction delegations.

---

### 3.6 `pack.h` — `hal::pack` (PACR family)

**Model.** Mirror of `hal::unpack`: one `DataTransfer` descriptor for the parameterized instruction plus canonical descriptors for the degenerate flavors. Blackhole has one packer with four Dest-read interfaces; that vocabulary (not WH's four packers) is the interface's language.

```cpp
namespace hal::pack
{

/** @brief Select how a PACR resolves its configuration and address-counter contexts. */
struct ContextSelection
{
    static constexpr ContextSelection thread_default();
    static constexpr ContextSelection explicit_context(
        std::uint8_t configuration_context, // [0, 3]
        std::uint8_t address_counter_context); // [0, 2], with thread-ID override
    static constexpr ContextSelection hardware_counter(); // CtxtCtrl RTL-flops auto-increment modes
};

/** @brief Select which of the four Dest-read interfaces participate. */
struct ReadInterfaces
{
    static constexpr ReadInterfaces all();               // mask 0
    static constexpr ReadInterfaces mask(std::uint8_t);  // explicit 4-bit mask
};

/** @brief Select which transfers get zero-padded rows, and the padding granularity. */
enum class RowPadding : std::uint8_t
{
    None,
    AllTransfers,
    NonConcatenated,
    FinalOnly
};

enum class PaddingAlignment : std::uint8_t
{
    PerRow,
    To16Datums
};

enum class DestAccess : std::uint8_t
{
    RowMajor, // rows 0,1,2,3
    Strided   // rows 0,8,16,24 (untilize)
};

/** @brief Select whether transferred datums retain their values or become zero (ZeroWrite). */
enum class DatumOverride : bool
{
    None = false,
    Zero = true
}; // same vocabulary as hal::unpack::DatumOverride

enum class Concatenation : bool
{
    NewRow = false,
    Append = true
};

enum class TileBoundary : bool
{
    NotLast = false,
    Last    = true // flush write-aligners, zero-pad to 16B, close the tile
};

struct DataTransfer
{
    std::uint8_t address_modifier = 0; // 2-bit ADDR_MOD_PACK index
    ContextSelection context      = ContextSelection::thread_default();
    ReadInterfaces interfaces     = ReadInterfaces::all();
    DatumOverride datum_override  = DatumOverride::None;
    DestAccess dest_access        = DestAccess::RowMajor;
    RowPadding padding            = RowPadding::None;
    PaddingAlignment alignment    = PaddingAlignment::PerRow;
    Concatenation concatenation   = Concatenation::NewRow;
    TileBoundary boundary         = TileBoundary::NotLast;
};

/** @brief Canonical flush of non-empty write-aligners with no new data (PACR Flush=1). */
struct WriteAlignerFlush
{
};

/**
 * @brief Pipeline-ordered MMIO register write through the packer (PACR_SETREG).
 *
 * Two-phase: LoadValue pre-loads one 16-bit half of the write-data flops (ModeSel form);
 * Write assembles {0xFFB, addr-slot, stream id} and issues the ordered register write.
 * No LLK call site exists today; the shape follows the ISA and stays minimal.
 */
struct RegisterWriteValue { /* half selector + 16-bit data */ };
struct RegisterWrite      { /* stream_id, preprogrammed-address slot [0,3] */ };

/** @brief Packer edge window: datums outside [x,y] ranges are zeroed (SETPKEDGOF). */
struct EdgeWindow
{
    std::uint8_t x_start;
    std::uint8_t x_end;
    std::uint8_t y_start;
    std::uint8_t y_end;
};

/** @brief Clear the packer exponent histogram (CLREXPHIST). Free-function trio, no descriptor. */
// clear_exponent_histogram() + get_operation form.

} // namespace hal::pack
```

**Decisions & constraints**

- `ReadInterfaces` validation encodes the Special/Normal-mode rules (ZeroWrite + all-interfaces implies the x_start=0 / multiple-of-16 constraints) as documented notes; the hard span rules live in packer CFG, outside this instruction.
- `EdgeWindow` is included for cross-architecture parity (Quasar firmware uses it) but documented as "BH LLK uses the `PCK_EDGE_OFFSET`/`TILE_ROW_SET_MAPPING` CFG path instead" — it may be dropped from the BH cut without loss (open exclusions, §5).
- `TileBoundary::Last` and `WriteAlignerFlush` are deliberately separate: one is a mode of a data transfer, the other is a degenerate no-data instruction — same split as unpack's canonical descriptors.

---

### 3.7 `nop.h` — `hal::nop` (pure pipeline bubbles) + unpack additions

**Model.** A NOP is only meaningful relative to the queue it occupies. One tiny namespace of free functions, one per pipeline, each with the `get_operation()` form for replay/MOP embedding:

```cpp
namespace hal::nop
{

void thread();  // NOP      — frontend issue slot, exactly one cycle
void scalar();  // DMANOP   — ThCon/TDMA queue slot (order a delay behind THCON traffic)
void vector();  // SFPNOP   — SFPU issue bubble (BH auto-stall errata, SFPLOADMACRO sequences)
void unpacker(unpack::Engine engine); // UNPACR_NOP pure-delay flavor

// each: *_operation() constexpr encoder for embedding.

} // namespace hal::nop
```

**UNPACR_NOP's side-effectful flavors are not NOPs** and go to `hal::unpack` as canonical descriptors, completing the bank-handoff story from the producer side:

```cpp
namespace hal::unpack
{

/**
 * @brief Fill and/or publish a source bank from the unpacker without unpacking (UNPACR_NOP).
 *
 * The sanctioned Blackhole way to hand the math thread a valid bank with no data movement —
 * covers the "other operand" pattern in unary ops and replaces raw SETDVALID in new code.
 */
struct SourceBankInitialize
{
    Engine engine; // Unpacker0 => SrcA path, Unpacker1 => SrcB path
    // Fill: Zero / NegativeInfinity / One / ConfiguredValue (UNP_NOP_REG_CLR_VAL)
    // BankScope: Current / All
    // Publish: whether Set_Dvalid hands the bank to the math thread
};

/** @brief Overlay-stream buffer pop through the unpacker sideband (UNPACR_NOP pop flavors). */
struct StreamBufferPop
{
    // stream id, message count, in-instruction vs sideband selection
};

} // namespace hal::unpack
```

---

### 3.8 `sync.h` addition — AutoSync resource declaration (RESOURCEDECL)

RESOURCEDECL configures the RISC-vs-Tensix auto-interlock; it is synchronization state, so it extends `hal::sync` rather than creating a header:

```cpp
namespace hal::sync::autosync
{

/** @brief The sixteen AutoSync instruction classes whose resource usage can be declared. */
enum class InstructionClass : std::uint8_t
{
    GprAlu = 0,
    Atomics,
    RegisterToFlop,
    IndirectAccess,
    StreamConfig,
    StoreRegister,
    LoadRegister,
    FlushDma,
    ConfigWrite,
    ConfigRead,
    Mover,
    Unpack,
    Pack,
    Mop,
    Replay,
    Others
};

/** @brief Declared resources: CFG read/write + zone, GPR read/write, TDMA-control read/write. */
enum class Resource : std::uint16_t { /* bit flags with operator| */ };

template <InstructionClass Class, Resource Resources, std::uint8_t Linger = 0b0001>
void declare();
// + runtime overload and *_operation() encoders.

} // namespace hal::sync::autosync
```

Documented caveat: MOP/replay-expanded instructions inherit the parent's declared class (13/14) — the interface docstring carries that rule since it is the main foot-gun. No LLK call site exists today (hardware defaults are used); low delivery priority.

---

## 4. How this fits the user-facing HAL story

The mental model the interfaces add up to, in the order a kernel author meets them:

1. **Ownership** — banks move between unpackers and math: `unpack::SourceBankInitialize` / UNPACR handoff publish → math consumes → `src::Release`, `SourceRelease` in `fpu`/`rwc` ops. One documented protocol, three entry points, all cross-referenced.
2. **Addressing** — `address_counters` selects among the ADC encodings; `math_counters` groups immediate RWC assignments and splits Bias into SETIBRWC only when required.
3. **Movement** — `hal::move` is a single routed matrix over {Dest, SrcA, SrcB, L1, L0, GPR, MMIO}; the engine that executes each route is a documented property of the route, not something the caller selects.
4. **State clears** — `src::Zero` (banks) and `dest::Zero` (zero-flags) live with the resource they clear.
5. **Pack** — `hal::pack` mirrors `hal::unpack`, closing the T0/T2 symmetry.
6. **Waiting and interlocks** — already in `hal::sync`; the new scalar/mover interfaces reference its stall conditions instead of inventing completion APIs.

---

## 5. Explicit exclusions (raw `TT_OP_*` macros remain the path)

| Instruction / bit | Reason |
|---|---|
| TRNSPSRCA | Absent from ISA docs; zero call sites; SrcA transpose is the unpacker's job |
| SHIFTXA | BH hardware bug (cannot select the 16-row block — reuses last SrcA address); ISA discourages use; zero call sites |
| SETASHRMH0/H1/V/H | Grayskull-era conv-halo legacy; absent from ISA docs; zero call sites |
| RAREB | Legacy rarefication; absent from ISA docs; zero call sites |
| RSTDMA | yaml carries `fixme; remove instr?`; no ISA documentation; zero call sites |
| FLUSHDMA | Blocks *all* threads' ThCon issue; ISA recommends STALLWAIT, which `hal::sync::wait` already covers |
| TBUFCMD | Reserved for future use; no documentation |
| SETRWC "SFPU SP" bit 4 | yaml-only claim, absent from ISA docs, unused — encoded 0 |
| CLEARDVALID Reset bit | `UnsupportedFunctionality` on BH (nondeterministic hang risk) — encoded 0 |
| ZEROACC bits 23:22 | Undocumented data-valid clears — encoded 0; releases have dedicated interfaces |

Each exclusion is a doc line in the interface guide, so the decision is visible rather than a gap.

---

## 6. Coverage map

| Group (from the grouping exercise) | Instruction | Interface |
|---|---|---|
| RWCs | SETRWC | `hal::math_counters.set<>()` (+ `release<>()`, `clear_fidelity()`, `advance_dest_and_save_to_carry<>()`) |
| | INCRWC | `hal::math_counters.increment<>()` |
| | SETIBRWC | `hal::math_counters.set<rwc::Counters::Bias>()` / `increment<>()` |
| Transpose | TRNSPSRCB | `hal::src::TransposeSrcB` |
| | TRNSPSRCA | excluded |
| Shift | SHIFTXB | `hal::src::ShiftRowSrcB` |
| | SHIFTXA | excluded |
| SrcA shift masks | SETASHRM* | excluded |
| Movement (Tensix) | MOVD2A / MOVD2B / MOVB2A | `hal::move::DestToSrcA / DestToSrcB / SrcBToSrcA` |
| | MOVA2D / MOVDBGA2D | `hal::move::SrcAToDest` (+ `SourceValidity::Ignore`) |
| | MOVB2D / MOVDBGB2D | `hal::move::SrcBToDest` (+ `SourceValidity::Ignore`) |
| L1/L0 movement | XMOV | `hal::move::MemoryCopy` |
| Scalar / pure L1 | ATINCGET | `hal::atomic::fetch_add` on `Counter` (descriptor `FetchIncrement` implemented) |
| | ATINCGETPTR | `hal::atomic::wait_while` / `pop_slots` / `push_slots` on `Fifo` (descriptor `FifoAcquire` implemented) |
| | ATCAS | `hal::atomic::acquire` / `release` / `compare_set` on `Lock` (descriptor `CompareSet` implemented) |
| Scalar / movement | ATSWAP | `hal::atomic::store_masked` (descriptor `MaskedWrite` implemented; see 8.3) |
| | LOADIND | `hal::move::L1ToGpr` |
| | STOREIND | `hal::move::GprToL1` / `GprToSrc` / (MMIO variant, see 8.7) |
| | STOREREG | `hal::move::GprToMmio` |
| | LOADREG | `hal::move::MmioToGpr` |
| Clear | ZEROACC | `hal::dest::Zero` |
| | ZEROSRC | `hal::src::Zero` |
| DValid | CLEARDVALID | `hal::src::Release` |
| | SETDVALID | `hal::src::Publish` (constrained) + `hal::unpack::SourceBankInitialize` (sanctioned) |
| Pack | PACR | `hal::pack::DataTransfer` / `WriteAlignerFlush` |
| | PACR_SETREG | `hal::pack::RegisterWrite(+Value)` |
| NOP | NOP / DMANOP / SFPNOP / UNPACR_NOP | `hal::nop::thread / scalar / vector / unpacker` |
| Misc | GATESRCRST | `hal::src::GatingReset` |
| | CLREXPHIST | `hal::pack::clear_exponent_histogram()` |
| | RESOURCEDECL | `hal::sync::autosync::declare` |
| | SETPKEDGOF | `hal::pack::EdgeWindow` (parity candidate) |
| | RAREB / RSTDMA / FLUSHDMA / TBUFCMD | excluded |

---

## 7. Suggested delivery order

Ordered by call-site pressure (what unblocks migrating LLK off raw macros fastest):

1. **`math_counters.h`** — implemented additive builder; migration remains for ~124 SETRWC + ~40 INCRWC sites.
2. **`move.h` matrix-unit routes + `src.h`** — the MOV* family (~250 sites), transpose/dvalid/zero idioms; unlocks `llk_math_transpose_dest`, reduce, datacopy migrations. These two ship together (transpose staging and validity notes cross-reference).
3. **`dest.h`** — ZEROACC sites in pack_common/datacopy/reduce.
4. **`pack.h`** — completes the T0/T2 symmetry; `llk_pack`/untilize migration.
5. **`nop.h` + unpack additions** — small, finishes the bank-handoff story.
6. **`move.h` scalar routes + `sync::autosync`** — no current call sites; API surface for metal-side consumers, lowest urgency. **`atomic.h` is implemented.**

Each step lands with its `docs/hal/` interface guide and an index/roadmap update, per the HAL documentation-sync rule.

---

## 8. Open questions

1. **SETDVALID policy.** ISA marks it unsupported on BH; three live SrcB-only call sites exist (datacopy self-staged SrcB, debug). Proposal: expose `src::Publish` with the constraint documented, migrate the three sites to it, and point new code at `unpack::SourceBankInitialize`. Alternative: don't expose it at all and migrate datacopy to the unpacker-side publish — needs a perf check (extra unpacker-queue traffic vs math-thread instruction).
2. **Scalar routes' home.** Proposed inside `hal::move` to honor the single movement matrix. Alternative: a `hal::scalar` namespace holding moves + atomics together (everything ThCon). The matrix wins on discoverability; the alternative wins on "one namespace per execution unit" purity.
3. **ATSWAP placement (resolved).** Implemented as `atomic::MaskedWrite` because it shares line-RMW semantics and the posted-completion story. Recorded alternative: a `move::GprToL1` masked variant.
4. **ZEROACC home.** Proposed `dest.h`. Alternative: `fpu.h` (it executes on the math unit and takes an addr_mod). `dest.h` preferred to keep fpu.h purely arithmetic.
5. **SETIBRWC semantics (implementation resolved, validation open).** The encoding is included through `Counters::Bias`; functional simulator coverage remains unavailable because public BH semantics are absent and the current simulator marks it unsupported.
6. **XMOV scope.** Instruction-only (proposed), or also absorb the RISC-MMIO mover path from `ckernel_xmov.h` as a runtime-only overload set?
7. **STOREIND MMIO variant.** The ISA documents a GPR→MMIO STOREIND form (bit23=0, bit22=1) overlapping STOREREG's job with indirect addressing. Cover it as a `GprToMmio` variant, or leave STOREREG as the only MMIO write path until a consumer needs indirection?
8. **PACR_SETREG shape.** No call sites exist to validate the two-phase (LoadValue/Write) split; confirm against the packer-pipeline use case (NOC stream reprogramming between packs) with the HW owner before freezing names.
9. **Atomic descriptor layer visibility.** Once the object + verb layer lands, do the implemented descriptors stay public (two supported surfaces), or retire into `detail` so the verbs become the only public path? Retiring keeps one way to say each thing; staying public keeps a raw escape hatch above the `TT_OP_*` macros.
