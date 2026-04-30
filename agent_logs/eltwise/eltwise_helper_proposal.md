# Eltwise Helper Proposal — Phase 3

**Date**: 2026-04-30
**Inputs**:
- `agent_logs/eltwise/eltwise_catalog.md` (146 ops, 11 groups)
- `agent_logs/eltwise/{binary, unary_sfpu_omnibus, ternary_special_misc}_investigation.md`
- `agent_logs/eltwise/eltwise_verification.md` (12/12 CONFIRMED)
- `ttnn/cpp/ttnn/kernel_lib/agents/eltwise_helper_lessons.md` (design knowledge)

**Constraint**: NEW eltwise helpers must NOT use existing `sfpu_helpers.{hpp,inl}` or `binary_op_helpers.{hpp,inl}` under the hood. Those are the prior iteration; treat as faulty reference.

---

## 1. File Layout (per lessons §5.1)

```
ttnn/cpp/ttnn/kernel_lib/
  eltwise_chain.hpp          # core: Dst, policies, CRTP bases, CopyTile, EltwiseChain, eltwise_pipeline
  eltwise_chain.inl
  eltwise_activations.hpp    # relu, gelu, silu, sigmoid, softplus, threshold, leaky_relu, hardmish, ...
  eltwise_activations.inl
  eltwise_binary.hpp         # FPU + SFPU binary; binary_op<>, DestReuseOp<>; BroadcastDim
  eltwise_binary.inl
  eltwise_math.hpp           # exp, log, sqrt, rsqrt, recip, abs, sign, erf/erfc/erfinv, i0/i1, lgamma
  eltwise_math.inl
  eltwise_misc.hpp           # fill, dropout, rand, mask, identity, heaviside
  eltwise_misc.inl
  eltwise_predicates.hpp     # eqz, nez, ltz, gtz, lez, gez, isnan, isinf, isfinite, logical_not, comp
  eltwise_predicates.inl
  eltwise_rounding.hpp       # floor, ceil, round, trunc
  eltwise_rounding.inl
  eltwise_scalar.hpp         # binop_with_scalar (unary_add/sub/mul/div), unary_max/min, rpow, rsub, power
  eltwise_scalar.inl
  eltwise_special.hpp        # clamp, fmod, remainder, xlogy
  eltwise_special.inl
  eltwise_ternary.hpp        # addcmul, addcdiv, lerp, where (TernaryOp CRTP)
  eltwise_ternary.inl
  eltwise_trig.hpp           # sin, cos, tan, asin/acos/atan, sinh/cosh/tanh, asinh/acosh/atanh
  eltwise_trig.inl
  eltwise_bitwise.hpp        # bitwise_not, bitwise_and/or/xor (binary), left/right_shift
  eltwise_bitwise.inl
  eltwise_helpers.hpp        # aggregator: includes all of the above for back-compat
```

**Migration path**: legacy `sfpu_helpers.{hpp,inl}` and `binary_op_helpers.{hpp,inl}` kept until call sites migrate. Eventually delete.

---

## 2. Core Types (`eltwise_chain.hpp`)

### 2.1 Dst slot enum (lessons §1.3, §4.3, verified §11)

```cpp
namespace compute_kernel_lib {

enum class Dst : uint32_t {
    D0 = 0, D1 = 1, D2 = 2, D3 = 3,
    D4 = 4, D5 = 5, D6 = 6, D7 = 7,
};

// All Dst-using structs static_assert: Slot < DEST_AUTO_LIMIT
// DEST_AUTO_LIMIT lives in dest_helpers.hpp (verified in Phase 2)
} // namespace compute_kernel_lib
```

### 2.2 Self-documenting flag enums

```cpp
enum class Approx : bool { Exact = false, Fast = true };
enum class Legacy : bool { Off = false, On = true };
enum class FP32DestAcc : bool { Off = false, On = true };
enum class MathFidelity : uint32_t { LoFi = 0, HiFi2 = 2, HiFi3 = 3, HiFi4 = 4 };
enum class OutputActivation : uint32_t { None = 0, Relu = 1, Relu6 = 2 };
enum class VectorMode : int { R = 1, C = 2, RC = 4 };
enum class RoundingMode : uint32_t { None = 0, Up = 1, Down = 2, Nearest = 3, Trunc = 4 };
```

### 2.3 CB Lifecycle Policy (lessons §2.1)

```cpp
// Wait/Pop combinations the kernels actually need
enum class CopyTilePolicy : uint32_t {
    WaitAndPop                  = 0, // per-tile wait + per-tile pop (streaming, default)
    WaitNoPop                   = 1, // per-tile wait + no pop (fan-out first / persistent)
    NoWaitPop                   = 2, // no wait + per-tile pop (fan-out last / pre-waited)
    NoWaitNoPop                 = 3, // no wait + no pop (caller owns CB lifecycle)
    WaitUpfrontPopAtEnd         = 4, // upfront wait + upfront pop (block access, indexed CopyTile)
    CumulativeWaitUpfrontEndPop = 5, // wait(base + i*step) per outer-loop iter, single pop at end
};
```

`CumulativeWaitUpfrontEndPop` exists because 6 production kernels use the pattern (see `agent_logs/eltwise/cumulative_wait_search.md`):
- `normalization/{rmsnorm,layernorm}_distributed/.../*_pre_allgather{,_2d}.cpp` × 4
- `experimental/transformer/dit_layernorm_post_all_gather/.../layernorm_post_allgather_welford.cpp`
- `experimental/transformer/fused_distributed_rmsnorm/.../rmsnorm_post_allgather.cpp`

Pattern: `for (var = 0; var < MAX; var += STEP) { cb_wait_front(cb, var + STEP); ...block... }`. Wait count grows; tiles stay fronted across outer-loop iters; pop once at end.

Compatible index modes: `BlockIter` (caller reads tile i within accumulated window) and `Pinned` / `Absolute` (caller index ≤ current accumulated count).

### 2.4 CB Index Mode (lessons §2.7)

```cpp
enum class CbIndexMode : uint32_t {
    FirstTile = 0, // always tile 0 (streaming consume after wait/pop)
    BlockIter = 1, // tile i where i is tile loop index (block-at-a-time)
    Pinned    = 2, // fixed compile/runtime k (scalar, broadcast-once, mask)
    Absolute  = 3, // caller-supplied runtime index (gather-style)
};
```

Compile-time validation (lessons §2.7 matrix):
- WaitAndPop / WaitNoPop / NoWaitPop + (BlockIter | Absolute) → static_assert fail
- WaitUpfrontPopAtEnd / NoWaitNoPop + any index mode → OK

### 2.5 BroadcastDim (verified §5)

```cpp
enum class BroadcastDim : uint32_t {
    NONE   = 0, // [Ht, Wt] tiles
    ROW    = 1, // [1, Wt] — replicate down (broadcast height direction)
    COL    = 2, // [Ht, 1] — replicate right (broadcast width direction)
    SCALAR = 3, // [1, 1]  — replicate everywhere
};

enum class BroadcastSide : uint32_t { LHS = 0, RHS = 1 };
```

### 2.6 DestReuse (lessons §1.7, verified §10)

```cpp
enum class DestReuseType : uint32_t {
    DEST_TO_SRCA = 0, // CB → srcb, DEST → srca
    DEST_TO_SRCB = 1, // CB → srca, DEST → srcb (verified: distinct unpack MOPs)
};

enum class DestReuseReconfig : uint32_t {
    None  = 0,
    Input = 1, // srca-or-srcb depending on ReuseType
};
```

### 2.7 CRTP bases (lessons §1.1)

```cpp
template <typename Derived, Dst Slot>
struct UnaryOp {
    static_assert(static_cast<uint32_t>(Slot) < DEST_AUTO_LIMIT,
                  "DEST slot exceeds compile-time DEST capacity");
    static constexpr Dst dst_idx = Slot;
    static constexpr bool is_copy_tile_op = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI void apply() const {
        static_cast<const Derived*>(this)->init();
        static_cast<const Derived*>(this)->exec();
    }
    ALWI void exec() const { static_cast<const Derived*>(this)->call(); }
};

template <typename Derived, Dst In0, Dst In1, Dst Out>
struct BinaryOp {
    static_assert(static_cast<uint32_t>(In0) < DEST_AUTO_LIMIT, "");
    static_assert(static_cast<uint32_t>(In1) < DEST_AUTO_LIMIT, "");
    static_assert(static_cast<uint32_t>(Out) < DEST_AUTO_LIMIT, "");
    static constexpr Dst in0 = In0, in1 = In1, out = Out;
    static constexpr bool is_copy_tile_op = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI void apply() const {
        static_cast<const Derived*>(this)->init();
        static_cast<const Derived*>(this)->exec();
    }
    ALWI void exec() const { static_cast<const Derived*>(this)->call(); }
};

template <typename Derived, Dst In0, Dst In1, Dst In2, Dst Out>
struct TernaryOp {
    // Strict slot order verified in Phase 2 §6: (in0, in1, in2, out)
    static_assert(static_cast<uint32_t>(In0) < DEST_AUTO_LIMIT, "");
    static_assert(static_cast<uint32_t>(In1) < DEST_AUTO_LIMIT, "");
    static_assert(static_cast<uint32_t>(In2) < DEST_AUTO_LIMIT, "");
    static_assert(static_cast<uint32_t>(Out) < DEST_AUTO_LIMIT, "");
    static constexpr Dst in0 = In0, in1 = In1, in2 = In2, out = Out;
    static constexpr bool is_copy_tile_op = false;
    static constexpr bool clashes_with_fpu = false;

    ALWI void apply() const {
        static_cast<const Derived*>(this)->init();
        static_cast<const Derived*>(this)->exec();
    }
    ALWI void exec() const { static_cast<const Derived*>(this)->call(); }
};
```

### 2.8 CopyTile element (lessons §3.5)

```cpp
template <
    uint32_t CB,
    Dst Slot                     = Dst::D0,
    CopyTilePolicy Policy        = CopyTilePolicy::WaitAndPop,
    CbIndexMode IndexMode        = CbIndexMode::FirstTile,
    CopyTileReconfig Reconfig    = CopyTileReconfig::None
>
struct CopyTile {
    // Tag for chain dispatch (lessons §1.5)
    using is_copy_tile_tag = std::true_type;
    static constexpr bool is_copy_tile_op = true;
    static constexpr bool clashes_with_fpu = true;
    static constexpr bool clobbers_sfpu_lut = false;
    static constexpr bool is_upfront = (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr uint32_t cb = CB;
    static constexpr Dst dst_idx = Slot;

    // CbIndexMode validation (lessons §2.7)
    static_assert(
        IndexMode == CbIndexMode::FirstTile
            || (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd
                || Policy == CopyTilePolicy::NoWaitNoPop),
        "BlockIter/Absolute index requires multi-tile window policy");

    // Pinned/Absolute carry runtime index (only when needed)
    uint32_t cb_tile_idx_runtime = 0;

    // Pipeline-private state — reset by the pipeline at end-of-block
private:
    mutable uint32_t cb_tile_idx_ = 0;
    template <typename...> friend struct EltwiseChain;
public:
    ALWI void wait_upfront(uint32_t n) const;  // implementation in .inl
    ALWI void exec_one_tile() const;
    ALWI void pop_at_end(uint32_t n) const;
};

enum class CopyTileReconfig : uint32_t {
    None  = 0,
    Input = 1, // srca reconfig (dtype-aware)
};
```

### 2.8a Hoisting hardware-resource taxonomy (validated)

Hoisting `init()` outside the loop only works if no per-tile call inside the loop re-touches the same hardware resource. Per-init resource map (from `agent_logs/eltwise/init_hoist_survey.md`):

| Init call | HW resource it touches | Per-tile re-touch by exec? | Hoist-safe? |
|---|---|---|---|
| `unary_op_init_common(in, out)` | unpack MOP + ADDR_MOD + pack | No | ✓ Always |
| `binary_op_init_common(a, b, out)` | unpack AB + math MOP + pack | No | ✓ Always |
| `copy_tile_to_dst_init_short()` | ADDR_MOD (unpack-A → DEST) | No (unless intervening dt-reconfig) | ✓ Conditional — invariant input dtype |
| `*_derivative_tile_init()` (tanh/gelu) | SFPU type register | No | ✓ Always |
| `mul_binary_tile_init()` / `add_binary_tile_init()` | math FPU MOP | No | ✓ Always |
| `square_tile_init()` | math FPU MOP | No | ✓ Always |
| `rsqrt_tile_init()` | SFPU LUT (rsqrt-only polynomial; idempotent) | No | ✓ Always |
| `add_tiles_init()` / `mul_tiles_init()` / `binary_tiles_init<Op>()` | math MOP per ELWADD/ELWMUL | No within same Op | ✓ Within single Op type |
| `exp_tile_init` / `log_tile_init` / `tanh_tile_init` / `sigmoid_tile_init` / `silu_tile_init` / `gelu_tile_init` | shared SFPU LUT (polynomial) | LUT clobbered by another LUT-op init | ✗ Hoist OK ONLY if no other LUT op in chain |
| `dropout_init(seed)` / `rand_init(seed)` | global PRNG | RNG advances per-tile (state mutates) | ✗ Per-tile reset breaks determinism — leave per-tile |
| `mask_init()` | SFPU type register (SfpuType::mask) | No | ✓ Always |

**Chain trait derivation** (compile time):

```cpp
// Per element static constexpr bool clobbers_sfpu_lut
template <typename Chain>
inline constexpr bool chain_has_lut_clobber_collision_v =
    /* count of elements with clobbers_sfpu_lut == true is > 1 */;

template <typename Chain>
inline constexpr bool chain_is_hoist_safe_v =
    Chain::has_any_copy_tile
    && !Chain::has_non_copy_tile_fpu_clash
    && !chain_has_lut_clobber_collision_v<Chain>;
```

V1 conservatively starts per-tile init by default; helper exposes `EnableHoist` template flag (default `false`). Set `true` only for chains the trait gates allow. As the per-op `clobbers_sfpu_lut` classification tightens, more chains become eligible automatically.

### 2.9 Chain combinator (lessons §3.3)

```cpp
template <typename... Elements>
struct EltwiseChain {
    std::tuple<Elements...> elements;

    ALWI explicit EltwiseChain(Elements... e) : elements(e...) {}

    // Compile-time traits
    static constexpr bool has_any_copy_tile = (Elements::is_copy_tile_op || ...);
    static constexpr bool has_non_copy_tile_fpu_clash =
        ((Elements::clashes_with_fpu && !Elements::is_copy_tile_op) || ...);
    static constexpr bool any_clobbers_lut = (Elements::clobbers_sfpu_lut || ...);

    // Static_assert: no two upfront CB-input elements share a CB
    // (lessons §1.7) — implemented via constexpr fold over all pairs

    // Pipeline drives elements in order
    ALWI void run_per_tile() const;  // each call: init+exec each element
    ALWI void run_hoist_safe() const; // init once, exec per tile (only if traits permit)
};

template <typename... Elements>
ALWI EltwiseChain<Elements...> eltwise_chain(Elements... e) {
    return EltwiseChain<Elements...>(e...);
}

template <typename Chain>
inline constexpr bool chain_is_hoist_safe_v =
    Chain::has_any_copy_tile
    && !Chain::has_non_copy_tile_fpu_clash
    && !Chain::any_clobbers_lut
    && std::tuple_size_v<decltype(std::declval<Chain>().elements)> <= 2;
```

### 2.10 `eltwise_pipeline` free function

```cpp
// The single entry point that wraps an EltwiseChain in a tile_regs lifecycle.
// Caller writes:
//
//   eltwise_pipeline<cb_in, cb_out>(num_tiles, eltwise_chain(
//       CopyTile<cb_in>{},
//       Exp<>{}));
//
// Helper handles: tile_regs_acquire/commit/wait/release, output CB
// reserve/push, DEST batching, cumulative or per-tile init.

template <
    uint32_t OutCB,
    typename Chain
>
ALWI void eltwise_pipeline(uint32_t num_tiles, Chain chain);
```

---

## 3. Op Structs (representative)

### 3.1 Plain unary SFPU (lessons §1.1)

```cpp
// eltwise_math.hpp
template <Approx ApproxMode = Approx::Exact, Dst Slot = Dst::D0>
struct Abs : UnaryOp<Abs<ApproxMode, Slot>, Slot> {
    ALWI static void init() {
        llk_math_eltwise_unary_sfpu_abs_init();
    }
    ALWI static void call() {
        llk_math_eltwise_unary_sfpu_abs<static_cast<bool>(ApproxMode)>(
            static_cast<uint32_t>(Slot));
    }
};

// Exp / Sqrt / Sign / Cos / Sin / Tan etc. follow same shape.
```

### 3.2 LUT-dependent unary SFPU (lessons §3.4)

```cpp
template <
    Approx ApproxMode = Approx::Exact,
    Approx FastApprox = Approx::Fast,
    FP32DestAcc Fp32 = FP32DestAcc::Off,
    Dst Slot = Dst::D0
>
struct Exp : UnaryOp<Exp<ApproxMode, FastApprox, Fp32, Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = true; // disqualifies hoisting
    ALWI static void init() {
        llk_math_eltwise_unary_sfpu_exp_init<
            static_cast<bool>(ApproxMode),
            static_cast<bool>(FastApprox),
            static_cast<bool>(Fp32)>();
    }
    ALWI static void call() {
        llk_math_eltwise_unary_sfpu_exp<
            static_cast<bool>(ApproxMode),
            static_cast<bool>(FastApprox),
            static_cast<bool>(Fp32)>(static_cast<uint32_t>(Slot));
    }
};
```

### 3.3 rsqrt — full param surface exported (verified §1)

LLK init takes 2 templates, exec takes 4. Helper struct exposes ALL 4 at the same template-param surface so any existing call site already passing the full set migrates 1:1 — no surface contraction, no information loss. Init internally routes only the 2 LLK-init params; exec routes all 4.

```cpp
template <
    Approx ApproxMode      = Approx::Exact,    // routed to init AND exec
    FP32DestAcc Fp32       = FP32DestAcc::Off, // exec only (LLK init ignores)
    Approx FastApprox      = Approx::Fast,     // exec only (LLK init ignores)
    Legacy LegacyMode      = Legacy::Off,      // routed to init AND exec
    Dst Slot               = Dst::D0
>
struct Rsqrt : UnaryOp<Rsqrt<ApproxMode, Fp32, FastApprox, LegacyMode, Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false; // rsqrt LUT is idempotent — hoist-safe
    ALWI static void init() {
        llk_math_eltwise_unary_sfpu_rsqrt_init<
            static_cast<bool>(ApproxMode),
            static_cast<bool>(LegacyMode)>();
    }
    ALWI static void call() {
        llk_math_eltwise_unary_sfpu_rsqrt<
            static_cast<bool>(ApproxMode),
            static_cast<bool>(Fp32),
            static_cast<bool>(FastApprox),
            static_cast<bool>(LegacyMode)>(static_cast<uint32_t>(Slot));
    }
};

// Caller migration is mechanical:
//   raw: llk_math_eltwise_unary_sfpu_rsqrt<true, true, false, true>(0);
//   new: Rsqrt<Approx::Fast, FP32DestAcc::On, Approx::Exact, Legacy::On>{}
//        invoked through eltwise_pipeline<cb_out>(n, eltwise_chain(CopyTile<cb_in>{}, Rsqrt<...>{}));
```

### 3.4 Scalar SFPU (lessons §1.2 — runtime fields)

```cpp
template <Approx ApproxMode = Approx::Exact, Dst Slot = Dst::D0>
struct LeakyRelu : UnaryOp<LeakyRelu<ApproxMode, Slot>, Slot> {
    float alpha;
    ALWI static void init() { llk_math_eltwise_unary_sfpu_leaky_relu_init(); }
    ALWI void call() const {
        llk_math_eltwise_unary_sfpu_leaky_relu<static_cast<bool>(ApproxMode)>(
            static_cast<uint32_t>(Slot), std::bit_cast<uint32_t>(alpha));
    }
};

template <Approx ApproxMode = Approx::Exact, FP32DestAcc Fp32 = FP32DestAcc::Off, Dst Slot = Dst::D0>
struct Power : UnaryOp<Power<ApproxMode, Fp32, Slot>, Slot> {
    float exponent;
    ALWI static void init() { llk_math_eltwise_unary_sfpu_power_init(); }
    ALWI void call() const {
        llk_math_eltwise_unary_sfpu_power<
            static_cast<bool>(ApproxMode), static_cast<bool>(Fp32)>(
                static_cast<uint32_t>(Slot), std::bit_cast<uint32_t>(exponent));
    }
};

template <Approx ApproxMode = Approx::Exact, Dst Slot = Dst::D0>
struct Clamp : UnaryOp<Clamp<ApproxMode, Slot>, Slot> {
    float min_val;
    float max_val;
    ALWI static void init() { llk_math_eltwise_unary_sfpu_clamp_init(); }
    ALWI void call() const {
        llk_math_eltwise_unary_sfpu_clamp<static_cast<bool>(ApproxMode)>(
            static_cast<uint32_t>(Slot),
            std::bit_cast<uint32_t>(min_val),
            std::bit_cast<uint32_t>(max_val));
    }
};
```

### 3.5 Mask (hardcoded slot+1 contract, verified §3)

```cpp
// Mask reads from DataSlot+1 unconditionally. Encoded in template instantiation
// via BinaryOp base; runtime DataFormat picks the kernel routine.
template <DataFormat DF = DataFormat::Float16_b, Dst DataSlot = Dst::D0>
struct Mask : BinaryOp<
    Mask<DF, DataSlot>,
    DataSlot,
    static_cast<Dst>(static_cast<uint32_t>(DataSlot) + 1),
    DataSlot>
{
    static_assert(static_cast<uint32_t>(DataSlot) + 1 < DEST_AUTO_LIMIT,
                  "Mask requires DataSlot + 1 < DEST capacity");
    ALWI static void init() { llk_math_eltwise_unary_sfpu_mask_init(); }
    ALWI static void call() {
        llk_math_eltwise_unary_sfpu_mask<false /*APPROXIMATE*/>(
            static_cast<uint32_t>(DataSlot), DF);
    }
};
```

### 3.6 Predicates (tag pattern, lessons §1.5)

```cpp
struct PredicateTag {};

template <Approx ApproxMode = Approx::Exact, Dst Slot = Dst::D0>
struct Eqz : UnaryOp<Eqz<ApproxMode, Slot>, Slot>, PredicateTag {
    ALWI static void init() { llk_math_eltwise_unary_sfpu_eqz_init(); }
    ALWI static void call() {
        llk_math_eltwise_unary_sfpu_eqz<static_cast<bool>(ApproxMode)>(
            static_cast<uint32_t>(Slot));
    }
};
// Nez, Ltz, Gtz, Lez, Gez, Isnan, Isinf, Isfinite, LogicalNot follow same shape.
```

### 3.7 Ternary (verified slot order, §6)

```cpp
template <Approx ApproxMode = Approx::Exact, FP32DestAcc Fp32 = FP32DestAcc::Off,
          Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst In2 = Dst::D2, Dst Out = Dst::D0>
struct AddCMul : TernaryOp<AddCMul<ApproxMode, Fp32, In0, In1, In2, Out>, In0, In1, In2, Out> {
    static constexpr bool clobbers_sfpu_lut = true;
    ALWI static void init() {
        llk_math_eltwise_ternary_sfpu_addcmul_init<
            static_cast<bool>(ApproxMode), static_cast<bool>(Fp32)>();
    }
    ALWI static void call() {
        llk_math_eltwise_ternary_sfpu_addcmul<
            static_cast<bool>(ApproxMode), static_cast<bool>(Fp32)>(
                static_cast<uint32_t>(In0),
                static_cast<uint32_t>(In1),
                static_cast<uint32_t>(In2),
                static_cast<uint32_t>(Out));
    }
};
// AddCDiv, Lerp, Where follow the same template/parameter shape.
```

### 3.8 FPU binary (`binary_op<>`)

```cpp
// eltwise_binary.hpp
enum class BinaryOpType : uint32_t { Add = 0, Sub = 1, Mul = 2 };

template <
    uint32_t CB_A, uint32_t CB_B, uint32_t CB_OUT,
    BinaryOpType Op,
    BroadcastDim BCast = BroadcastDim::NONE,
    BroadcastSide BCastSide = BroadcastSide::RHS,
    MathFidelity Fidelity = MathFidelity::LoFi,
    FP32DestAcc Fp32 = FP32DestAcc::Off,
    OutputActivation PostOp = OutputActivation::None,
    CopyTilePolicy InputAPolicy = CopyTilePolicy::WaitAndPop,
    CopyTilePolicy InputBPolicy = InputAPolicy,
    CbIndexMode AIndex = CbIndexMode::FirstTile,
    CbIndexMode BIndex = CbIndexMode::FirstTile,
    typename PostOpChain = NoOp
>
ALWI void binary_op(uint32_t num_tiles, PostOpChain post_op = NoOp{});

// Common-case wrappers:
template <uint32_t A, uint32_t B, uint32_t Out, /*defaults*/...>
ALWI void add_op(uint32_t num_tiles) {
    binary_op<A, B, Out, BinaryOpType::Add, /*defaults*/>(num_tiles);
}
// sub_op, mul_op similarly.
```

### 3.9 DestReuseOp (lessons §1.7, verified §10)

```cpp
template <
    uint32_t CB,
    BinaryOpType Op,
    DestReuseType ReuseType = DestReuseType::DEST_TO_SRCA,
    Dst DestSlot = Dst::D0,
    DestReuseReconfig Reconfig = DestReuseReconfig::None,
    CopyTilePolicy CbPolicy = CopyTilePolicy::WaitAndPop,
    CbIndexMode CbIndex = CbIndexMode::FirstTile
>
struct DestReuseOp {
    static constexpr bool is_copy_tile_op = false;
    static constexpr bool clashes_with_fpu = true; // FPU binary, lessons §3.2
    static constexpr bool is_upfront = (CbPolicy == CopyTilePolicy::WaitUpfrontPopAtEnd);
    static constexpr uint32_t cb = CB;

    uint32_t cb_tile_idx_runtime = 0; // for Pinned / Absolute

    ALWI static void init() {
        binary_dest_reuse_tiles_init<to_eltwise_binary_type<Op>(), to_reuse_dest_type<ReuseType>()>(CB);
    }
    ALWI void call() const {
        binary_dest_reuse_tiles<to_eltwise_binary_type<Op>(), to_reuse_dest_type<ReuseType>()>(
            CB, cb_tile_idx_runtime, static_cast<uint32_t>(DestSlot));
    }
};

// Convenience alias for V1
template <uint32_t CB, Dst Slot = Dst::D0>
using DestReuseMul = DestReuseOp<CB, BinaryOpType::Mul, DestReuseType::DEST_TO_SRCA, Slot>;
```

---

## 4. LLK Sequence Validation

### Per-helper sequence table

| Helper | Internal LLK Sequence | Codebase Precedent | Status |
|---|---|---|---|
| `eltwise_pipeline` (per-tile init) | acquire → init_each → exec_each → commit → wait → pack → release | every SFPU kernel (e.g. `eltwise_sfpu.cpp`) | VALIDATED |
| `eltwise_pipeline` (hoist-safe: CopyTile + 1 plain SFPU + 1 binary mul) | unary_op_init_common → derivative_init + mul_binary_tile_init (once) → loop: copy×2 → derivative → mul → pack | `eltwise_bw_tanh_deriv.cpp:32-46`, `eltwise_bw_gelu_poly.cpp:29-46` | VALIDATED |
| `eltwise_pipeline` (hoist: rsqrt LUT-once) | rsqrt_tile_init once → loop: copy → rsqrt → pack | 48 occurrences across batch_norm / layernorm / groupnorm | VALIDATED |
| `eltwise_pipeline` (cumulative wait) | cumulative cb_wait_front grows per outer iter → block process → pop at end | rmsnorm_pre_allgather.cpp:52, layernorm_pre_allgather.cpp:47, layernorm_pre_allgather_2d.cpp:49, rmsnorm_pre_allgather_2d.cpp:58 | VALIDATED |
| `binary_op` (no broadcast) | binary_op_init_common → binary_tiles_init<true, ELWADD>(A, B) → loop: wait A; wait B; acquire; add_tiles; pack; release; pop A; pop B | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/.../eltwise_binary_no_bcast.cpp` | VALIDATED |
| `binary_op` (broadcast) | `binary_op_init_common` → `mul_tiles_bcast_init<BCAST>` → loop with mul_bcast_tiles | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/.../eltwise_binary_*_bcast.cpp` | VALIDATED |
| `binary_op` SFPU path | `binary_op_init_common` → loop: copy_tile A→D0; copy_tile B→D1; sfpu init; sfpu_op(D0,D1,D0); pack | `binary_ng/.../eltwise_binary_sfpu_no_bcast.cpp` | VALIDATED |
| DestReuseOp | `binary_dest_reuse_tiles_init<Op,ReuseType>(CB)` → loop: copy_tile; binary_dest_reuse_tiles | batch_norm / running_stats kernels (per lessons §1.7) | VALIDATED |
| Mask | `mask_init` → exec reads slot+1 hardcoded | LLK ckernel_sfpu_mask.h (verified §3) | VALIDATED |
| Ternary (addcmul, addcdiv, lerp, where) | `addcmul_init` → loop: copy 3 inputs to D0/D1/D2 → exec(0,1,2,out) | LLK + ternary kernels (verified §6) | VALIDATED |

### Init mutual exclusion check

| Init A | Init B | Helper combines them in same chain? | Status |
|---|---|---|---|
| `exp_init` | `log_init` | NO — chain trait `clobbers_sfpu_lut` prevents combining LUT ops | OK |
| `dropout_init` | `rand_init` | NO — caller must invoke separately | OK (verified §7) |
| `add_tiles_init` | `mul_tiles_init` | NO — separate `binary_op<Add>` and `binary_op<Mul>` invocations | OK |
| `binary_max_init` | `binary_min_init` | NO — distinct helper instantiations; caller dedups | OK (verified §4) |
| `copy_tile_init_short` | any SFPU init | YES — chain combinator emits both, paired with their exec | OK (lessons §3.4) |

No violations.

---

## 5. Caller Examples (Before / After)

### Example 1 — exp(x)

```cpp
// BEFORE (raw LLK):
unary_op_init_common(cb_in, cb_out);
exp_tile_init<false /*FAST_APPROX*/>();
copy_tile_to_dst_init_short();
for (uint32_t i = 0; i < num_tiles; ++i) {
    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_pop_front(cb_in, 1);
    cb_push_back(cb_out, 1);
}

// AFTER:
using namespace compute_kernel_lib;
eltwise_pipeline<cb_out>(num_tiles, eltwise_chain(
    CopyTile<cb_in>{},
    Exp<Approx::Fast>{}));
```

### Example 2 — `mul(x, scalar)`

```cpp
// BEFORE: ~25 lines of init/wait/copy/scalar bitcast/pop boilerplate

// AFTER:
eltwise_pipeline<cb_out>(num_tiles, eltwise_chain(
    CopyTile<cb_in>{},
    Power<>{.exponent = 2.0f}));
// Or for scalar binop:
eltwise_pipeline<cb_out>(num_tiles, eltwise_chain(
    CopyTile<cb_in>{},
    UnaryMul<>{.scalar = 1.5f}));
```

### Example 3 — `add(a, b)` with Relu fusion

```cpp
// BEFORE: macro-injection BINARY_OP=add_tiles, ELTWISE_OP=add, PACK_RELU=1
// 30+ lines of #ifdef-laden boilerplate

// AFTER:
using namespace compute_kernel_lib;
binary_op<cb_a, cb_b, cb_out, BinaryOpType::Add,
          BroadcastDim::NONE, BroadcastSide::RHS,
          MathFidelity::LoFi, FP32DestAcc::Off,
          OutputActivation::Relu>(num_tiles);
```

### Example 4 — `square(x)` via same-CB dedup (lessons §3.6)

```cpp
// AFTER:
binary_op<cb_x, cb_x, cb_out, BinaryOpType::Mul>(num_tiles);
// helper detects icb_a == icb_b and dedups wait/pop
```

### Example 5 — fan-out (lessons §3.5)

```cpp
// x * exp(x) — the same CB tile read into D0 (multiplicand) AND fed through exp into D1
eltwise_pipeline<cb_out>(num_tiles, eltwise_chain(
    CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitNoPop>{},  // wait once, no pop
    CopyTile<cb_in, Dst::D1, CopyTilePolicy::NoWaitPop>{},  // skip wait, pop after
    Exp<>{},                                                  // operates on D1
    // multiply: D0 * D1 -> D0 via SFPU binary
    MulBinary<Dst::D0, Dst::D1, Dst::D0>{}));
```

### Example 6 — DEST reuse (mul-accumulate)

```cpp
// running_sum = a + b * c (per-tile accumulation)
eltwise_pipeline<cb_out>(num_tiles, eltwise_chain(
    CopyTile<cb_a, Dst::D0>{},
    DestReuseOp<cb_b, BinaryOpType::Mul, DestReuseType::DEST_TO_SRCA, Dst::D0>{}));
```

### Example 7 — tanh backward derivative (matches `eltwise_bw_tanh_deriv.cpp:32-46`)

```cpp
// BEFORE: 18 lines manual init/copy/exec/commit/pack/release per tile
unary_op_init_common(cb_grad_out, cb_grad_in);
tanh_derivative_tile_init<false>();
mul_binary_tile_init();
for (uint32_t b = 0; b < blocks; ++b) {
    cb_wait_front(cb_grad_out, block_size);
    cb_wait_front(cb_input, block_size);
    cb_reserve_back(cb_grad_in, block_size);
    for (uint32_t i = 0; i < block_size; ++i) {
        tile_regs_acquire();
        copy_tile(cb_grad_out, i, 0);
        copy_tile(cb_input, i, 1);
        tanh_derivative_tile<false>(1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_grad_in);
        tile_regs_release();
    }
    cb_pop_front(cb_grad_out, block_size);
    cb_pop_front(cb_input, block_size);
    cb_push_back(cb_grad_in, block_size);
}

// AFTER (with hoist enabled, chain validated by exemplar):
eltwise_pipeline<cb_grad_in, /*EnableHoist=*/true>(blocks * block_size, eltwise_chain(
    CopyTile<cb_grad_out, Dst::D0>{},
    CopyTile<cb_input,    Dst::D1>{},
    TanhDerivative<Dst::D1>{},
    MulBinary<Dst::D0, Dst::D1, Dst::D0>{}));
```

### Example 8 — logsigmoid (matches `logsigmoid_kernel.cpp` Tier 2 pattern)

```cpp
// AFTER:
eltwise_pipeline<cb_out>(num_tiles, eltwise_chain(
    CopyTile<cb_in>{},
    Negative<>{},
    Exp<Approx::Fast>{},
    LogSigmoid<>{}));
```

Note: Exp clobbers SFPU LUT and LogSigmoid loads its own activation LUT — chain trait `chain_has_lut_clobber_collision_v == true` → hoist disabled, init runs per-tile. Default-safe.

### Example 9 — ternary addcmul / addcdiv

```cpp
// BEFORE: addcmul(a, b, c, scalar) = a + scalar*b*c via FPU mul + scalar mul + add chain
// AFTER (TernaryOp single struct, no chain needed):
eltwise_pipeline<cb_out>(num_tiles, eltwise_chain(
    CopyTile<cb_a, Dst::D0>{},
    CopyTile<cb_b, Dst::D1>{},
    CopyTile<cb_c, Dst::D2>{},
    AddCMul<>{ /* scalar passed via runtime field if templated; else compile-time */ }));
```

### Example 10 — cumulative wait (RMSNorm pre-allgather staging)

```cpp
// BEFORE (rmsnorm_pre_allgather.cpp:52):
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    cb_wait_front(cb_inp, wt + blk);  // cumulative
    cb_reserve_back(cb_x2, blk);
    // ...mul x*x in DEST, pack to cb_x2
    cb_push_back(cb_x2, blk);
}
cb_pop_front(cb_inp, Wt);  // single pop at end

// AFTER:
binary_op<cb_inp, cb_inp, cb_x2,
    BinaryOpType::Mul,
    BroadcastDim::NONE, BroadcastSide::RHS,
    MathFidelity::LoFi, FP32DestAcc::Off,
    OutputActivation::None,
    /*InputAPolicy=*/CopyTilePolicy::CumulativeWaitUpfrontEndPop,
    /*InputBPolicy=*/CopyTilePolicy::CumulativeWaitUpfrontEndPop,
    /*AIndex=*/CbIndexMode::BlockIter,
    /*BIndex=*/CbIndexMode::BlockIter>(Wt);
// helper detects icb_a == icb_b and dedups (lessons §3.6)
```

---

## 6. Op Catalog (which structs land in which file)

### eltwise_chain.hpp
- Dst, all flag enums
- CRTP bases: UnaryOp, BinaryOp, TernaryOp
- Policies: CopyTilePolicy, CbIndexMode, BroadcastDim, BroadcastSide, DestReuseType, DestReuseReconfig, CopyTileReconfig
- CopyTile element
- EltwiseChain combinator + traits
- eltwise_pipeline free function
- NoOp sentinel

### eltwise_activations.hpp (~24 ops)
Relu, Relu6, RelyMax, ReluMin, Sigmoid, SigmoidApprox, Silu, Gelu, GeluApprox, Elu, Selu, Celu, Mish, MishApprox, Hardmish, Hardsigmoid, Hardtanh, Softplus, Softshrink, Softsign, Threshold, Typecast, Xielu, PRelu, LeakyRelu

### eltwise_binary.hpp (~34 ops)
- BinaryOpType enum + binary_op<> entry point (FPU binary)
- Convenience: AddOp, SubOp, MulOp wrappers
- SFPU binary tile-shape ops: AddBinary, SubBinary, MulBinary, DivBinary, RsubBinary, PowerBinary, BinaryMax/Min (+ int32/uint32 variants), BinaryPow, BinaryRemainder, BinaryFmod, BinaryShift, Atan2BinaryTile
- DestReuseOp + DestReuseMul alias

### eltwise_math.hpp (~25 ops)
Abs, Sign, Sqrt, Rsqrt, Recip, Cbrt, Exp, Exp2, Expm1, Log, Log1p, Log2, Log10, LogWithBase, Erf, Erfc, Erfinv, I0, I1, Lgamma, Digamma, Polygamma, Cosh, Sinh, AsinH, AcosH, AtanH

### eltwise_misc.hpp (~6 ops in scope)
Fill, Identity, Heaviside, Mask, Dropout, Rand
**Out of scope** (flagged as gaps): Reduce (other category), SfpuIntSum (cross-iteration state), TiledProd (cross-iteration state), Reshuffle (external index), AddTopRow, CopyDestValues, MaxPoolIndices.

### eltwise_predicates.hpp (~11 ops)
Eqz, Nez, Ltz, Gtz, Lez, Gez, Isnan, Isinf, Isfinite, LogicalNot, Comp + int32 variants

### eltwise_rounding.hpp (~4 ops)
Floor, Ceil, Round (RoundingMode template), Trunc

### eltwise_scalar.hpp (~8 ops)
UnaryAdd, UnarySub, UnaryMul, UnaryDiv (binop_with_scalar template), UnaryMin, UnaryMax, Rpow, Rsub, Power, PowerIterative

### eltwise_special.hpp (~4 ops in scope)
Clamp, Fmod, Remainder, Xlogy

### eltwise_ternary.hpp (~4 ops in scope)
AddCMul, AddCDiv, Lerp, Where
**Mac**: missing primitive (verified §2) — gap.

### eltwise_trig.hpp (~9 ops)
Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, AltComplexRotate90

### eltwise_bitwise.hpp (~5 ops)
BitwiseNot (unary), BitwiseAnd, BitwiseOr, BitwiseXor (binary), LeftShift, RightShift

### eltwise_helpers.hpp
Aggregator. Includes everything above. Provides backward-compat alias if needed.

---

## 7. Design Decisions

| Decision | Rationale |
|---|---|
| Free functions, not classes | Bare-metal RISC-V firmware (per HQ doc) |
| File split per op-family | Build time + locality (lessons §5.1) |
| CRTP bases UnaryOp/BinaryOp/TernaryOp | 4-line derived structs (lessons §1.1) |
| Self-documenting enums (Approx/Legacy/Dst) | Boolean template params unreadable at call site (lessons §1.3) |
| `DEST_AUTO_LIMIT` constexpr | Verified §11 — never literal 8 |
| Mask hardcodes Slot+1 | Verified §3 — LLK ignores second slot arg |
| `mac` excluded | Verified §2 — missing primitive |
| Cumulative wait omitted | Verified §12 — no production kernel needs it |
| DEST_TO_SRCA and DEST_TO_SRCB as separate template params | Verified §10 — distinct unpack MOPs |
| BroadcastDim explicit, never inferred | Lessons §10 — hidden inference forbidden |
| Per-tile init default; hoisting opt-in | Lessons §3.4, §10 — wrong output is not a perf win |
| Same-CB dedup in helper | Lessons §3.6 — user-side dedup fragile |
| `clobbers_sfpu_lut` trait | Phase 1 finding — LUT clash detection (exp/log/tanh/sigmoid) |
| Independent A/B policies in binary_op | Lessons §2.3 — A streams, B persists patterns |
| Default OutputActivation::None, default Approx::Exact | Lessons §2.6 — safe defaults |
| Do not include `using namespace compute_kernel_lib;` examples | HQ doc Step 3 — full qualification mandatory |
| Do not call into existing `sfpu_helpers` / `binary_op_helpers` | User constraint — fresh implementation |

---

## 8. Migration Tier Table

Survey results: 49 eltwise kernels + 4 normalization adjacent. Aggressive reclassification — macro-injection and "weird" kernels promoted into tiers with concrete attack paths. Don't run from hard kernels; the helper's job is to handle them.

### Tier 1 — Direct swap (7 kernels)

| File | Pattern | Helper Call |
|---|---|---|
| `eltwise/unary/.../eltwise_identity_kernel.cpp` | copy_tile only | `eltwise_pipeline<cb_out>(n, eltwise_chain(CopyTile<cb_in>{}))` |
| `eltwise/unary_backward/tanh_bw/.../eltwise_bw_tanh_deriv.cpp` | copy×2 + tanh_deriv + mul_binary | Example 7 |
| `eltwise/unary_backward/gelu_bw/.../eltwise_bw_gelu_poly.cpp` | copy×2 + gelu_deriv + mul_binary | Example 7 (swap derivative) |
| `normalization/rmsnorm_distributed/.../rmsnorm_pre_allgather.cpp` | cumulative wait + x*x | Example 10 |
| `normalization/rmsnorm_distributed/.../rmsnorm_pre_allgather_2d.cpp` | same, 2D variant | Example 10 |
| `normalization/layernorm_distributed/.../layernorm_pre_allgather.cpp` | cumulative wait + x*x | Example 10 |
| `normalization/layernorm_distributed/.../layernorm_pre_allgather_2d.cpp` | same, 2D | Example 10 |

### Tier 2 — Moderate restructuring (35 kernels)

#### 2a. Eltwise direct chain (5)

| File | Pattern | Helper Call |
|---|---|---|
| `eltwise/unary/.../eltwise_sfpu.cpp` | copy + SFPU_OP_CHAIN_0 (config-driven) | `eltwise_pipeline<cb_out>(n, eltwise_chain(CopyTile<cb_in>{}, <op>{}))` |
| `eltwise/unary/.../logsigmoid_kernel.cpp` | copy + Negative + Exp + LogSigmoid | Example 8 |
| `eltwise/ternary/.../ternary_addc_ops_fpu.cpp` | mul + scalar mul + add via DestReuse | `binary_op<Mul>` + `DestReuseOp<Add>` chain |
| `eltwise/ternary/.../ternary_addcmul_int_sfpu.cpp` | int copy×3 + fill + mul_int×2 + add_int | TernaryOp `AddCMul<int32>{}` once primitive lands |
| `eltwise/ternary/.../ternary_sfpu_no_bcast_ttt.cpp` | copy×3 + ternary SFPU op | Example 9 (TernaryOp single struct) |

#### 2b. Macro-injection BINARY (replace dispatch surface) — 20 kernels

The macro-injection pattern (`-DBINARY_OP=add_tiles`) is the SOURCE of the dispatch the helper replaces. Migration unit is per-program-factory, not per-kernel-source: the program factory chooses the BinaryOpType template arg and the kernel is `binary_op<>` once. Lessons §11.1 calls macro-injection out-of-scope ONLY because per-kernel migration churns; here we replace the whole dispatch surface in one shot.

Each `eltwise_binary_<bcast>_<op>_<sfpu>.cpp` collapses to ONE compute kernel that takes `BinaryOpType` + `BroadcastDim` + `BroadcastSide` as template params from the program factory:

```cpp
// New compute kernel content (replaces 20 macro-injected variants):
namespace NAMESPACE { void MAIN {
    compute_kernel_lib::binary_op<
        get_compile_time_arg_val(0)  /*cb_a*/,
        get_compile_time_arg_val(1)  /*cb_b*/,
        get_compile_time_arg_val(2)  /*cb_out*/,
        static_cast<compute_kernel_lib::BinaryOpType>(get_compile_time_arg_val(3)),
        static_cast<compute_kernel_lib::BroadcastDim>(get_compile_time_arg_val(4)),
        static_cast<compute_kernel_lib::BroadcastSide>(get_compile_time_arg_val(5)),
        compute_kernel_lib::MathFidelity::LoFi,
        compute_kernel_lib::FP32DestAcc::Off>(get_arg_val<uint32_t>(0));
}}
```

Program factory side: pass enum integer values via `compute_compile_time_args`. No `-DBINARY_OP=...` flags.

Affected kernels (all collapse to one):
- `binary_ng/.../eltwise_binary_no_bcast.cpp`, `eltwise_binary_sfpu_no_bcast.cpp`
- `eltwise_binary_{col,row,scalar}_bcast.cpp`, `eltwise_binary_sfpu_{col,row,scalar}_bcast.cpp`
- `eltwise_binary_sub_bcast_{col,row}.cpp`
- `binary/.../eltwise_binary.cpp` and broadcast variants
- (~20 source files; ALL collapse to 1)

#### 2c. Macro-injection TERNARY — 8 kernels

Same pattern: `-DTERNARY_SFPU_OP=where` etc. → `eltwise_pipeline + TernaryOp` template arg from program factory.

```cpp
// New ternary compute kernel:
eltwise_pipeline<cb_out>(num_tiles, eltwise_chain(
    CopyTile<cb_in0, Dst::D0>{},
    CopyTile<cb_in1, Dst::D1>{},
    CopyTile<cb_in2, Dst::D2>{},
    TernaryOpDispatch<get_compile_time_arg_val(N)>{}));
```

Where `TernaryOpDispatch<E>` is a CRTP wrapper that selects between AddCMul / AddCDiv / Lerp / Where at compile time via `if constexpr`. 8 source kernels collapse to 1.

#### 2d. Mid-loop dtype swap — 3 kernels (hardswish, mish, tanhshrink)

These flip between BF16 and FP32 paths inside the tile loop via `#ifdef INP_FLOAT32 / INP_FLOAT`. Strategy: helper exposes `MidChainReinit` policy. Two paths:

**Option A (preferred — split chain)**: emit two `eltwise_pipeline` invocations with intermediate CB:
```cpp
// hardswish(x) = x * relu6(x + 3) / 6
// Phase A (BF16 → BF16):
eltwise_pipeline<cb_intermediate>(n, eltwise_chain(
    CopyTile<cb_in>{}, UnaryAdd<>{ .scalar = 3.0f }, Relu6<>{}));
// Phase B (BF16 → BF16):
eltwise_pipeline<cb_out>(n, eltwise_chain(
    CopyTile<cb_in,        Dst::D0>{},
    CopyTile<cb_intermediate, Dst::D1>{},
    MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
    UnaryDiv<>{ .scalar = 6.0f }));
```

**Option B (single-chain mid-reinit policy)**: helper extends with `MidChainReinitOnDtype` template flag that emits an explicit `unary_op_init_common(cb_a, cb_b)` reconfig point inside the chain. Higher complexity, deferred to V2.

Tier 2 with Option A. mish and tanhshrink follow same split.

#### 2e. Intermediate-CB round-trip — 1 kernel (logit)

`logit(x) = log(x / (1 - x))` requires two `log()` evaluations or one `log(x/(1-x))` post-clamp. Strategy: two pipelined helper calls.

```cpp
// Stage 1: clamp_eps(x)
eltwise_pipeline<cb_clamped>(n, eltwise_chain(
    CopyTile<cb_in>{}, Clamp<>{ .min_val = eps, .max_val = 1.0f - eps }));
// Stage 2: log(x_clamped / (1 - x_clamped)) — no intermediate CB needed
eltwise_pipeline<cb_out>(n, eltwise_chain(
    CopyTile<cb_clamped, Dst::D0, CopyTilePolicy::WaitNoPop>{},  // hold for 1-x
    CopyTile<cb_clamped, Dst::D1, CopyTilePolicy::NoWaitPop>{},  // already waited
    Rsub<Dst::D1>{ .scalar = 1.0f },                              // D1 = 1 - x
    DivBinary<Dst::D0, Dst::D1, Dst::D0>{},                       // D0 = x / (1-x)
    Log<>{}));
```

#### 2f. Multi-DST orchestration — 2 kernels (lgamma, lgamma_fast)

lgamma uses `log + sin + floor + where` simultaneously across multiple DEST slots. Express as chain with explicit slot pinning:

```cpp
// Approximate sketch (full chain needs full polynomial coefficients):
eltwise_pipeline<cb_out>(n, eltwise_chain(
    CopyTile<cb_in,  Dst::D0>{},                  // x
    CopyTile<cb_in,  Dst::D1, CopyTilePolicy::NoWaitNoPop>{},  // x copy for floor
    Floor<Dst::D1>{},                              // D1 = floor(x)
    CopyTile<cb_in,  Dst::D2, CopyTilePolicy::NoWaitNoPop>{},  // x copy for sin
    Sin<Dst::D2>{},                                // D2 = sin(πx)
    Log<Dst::D0>{},                                // D0 = lgamma_polynomial
    // Where(D2 > 0, D0, -D0) handled by ternary Where
    Where<Dst::D2, Dst::D0, /*neg*/Dst::D0, Dst::D0>{}));
```

`static_assert(max_slot < DEST_AUTO_LIMIT)` validates slot 0-2 fit.

### Tier 3 — Permanently blocked (7 kernels)

Only kernels with cross-iteration DEST hold (lessons §3.7) or out-of-category remain blocked. Helper as designed cannot model these without fundamental redesign:

| File | Reason |
|---|---|
| `sfpu_int_sum` | held DEST accumulator across acquire/release boundaries |
| `tiled_prod` | held DEST product accumulator |
| `reduce` | other helper family (`reduce_helpers_*`) |
| `reshuffle` | external L1 index dependency, not a CB-driven op |
| `add_top_row` | DEST-internal helper; not a CB op |
| `copy_dest_values` | DEST-to-DEST, no CB involvement |
| `max_pool_indices` | pooling-specific multi-tile orchestration |

**Reduction in V1 scope blockers**: from "41 deferred" → "7 truly blocked, 35 promoted". The Tier 2 expansion (macro-injection, mid-loop dtype, multi-DST, CB round-trip) is the meat of the migration.

### Adjacent — already promoted to Tier 1

The 4 normalization pre-allgather kernels (cumulative-wait users) moved into Tier 1 above. Other adjacent candidates:
- `transformer/*` post-attention residual adds → straight `binary_op<Add>`. Sweep separately after Phase 5.
- `embedding/*` lookup compute → `eltwise_pipeline + Mask{}`. Sweep separately.
- `experimental/transformer/{dit_layernorm,fused_distributed_rmsnorm}/*` → Example 10 pattern with cumulative wait. Tier 1 once Phase 5 lands.

---

## 9. Test Plan

```bash
# Build
./build_metal.sh

# Helper unit tests (one kernel + golden per case)
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/test_eltwise_chain.py
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/test_eltwise_binary.py
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/test_eltwise_ternary.py
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/test_dest_reuse.py
```

Test cases per helper:
- `num_tiles ∈ {1, 8, 64}`
- `fp32_dest_acc_en ∈ {False, True}`
- For each binary op: srcA-only reconfig, srcB-only reconfig, both reconfigured
- For DestReuseOp: both DEST_TO_SRCA and DEST_TO_SRCB
- For each BroadcastDim ∈ {NONE, ROW, COL, SCALAR}
- For each policy: WaitAndPop, WaitUpfrontPopAtEnd, NoWaitNoPop (the three lifecycles real kernels need)
- Same-CB dedup test (`square = mul(x, x)`)
- Fan-out test (`x * exp(x)` via WaitNoPop + NoWaitPop)
- Mask test (DataSlot+1 hardcoded contract)
- Ternary slot order test
- Predicate output (0/1 fp)
- Tolerance: `comp_pcc(...)` with rtol≈5e-2, atol≈1e-1 for compounded bf16 ULPs (lessons §8)

---

## 10. Open Questions / Status

| # | Item | Status |
|---|---|---|
| 1 | Hoist-safe `CopyTile + plain SFPU + binary mul` sequence | **CLOSED — VALIDATED**. `eltwise_bw_tanh_deriv.cpp:32-46` and `eltwise_bw_gelu_poly.cpp:29-46` are exemplars. HW-resource taxonomy in §2.8a above. Helper exposes `EnableHoist` flag default `false`; chain trait `chain_is_hoist_safe_v` gates compile-time. |
| 2 | rsqrt template mismatch (init 2 / exec 4) | **POLICY DECIDED**. Helper struct exposes ALL 4 params (`Approx, Fp32DestAcc, FastApprox, Legacy`) at the same template-param surface so any caller already passing the full set migrates 1:1 — no surface contraction. Init internally drops `Fp32DestAcc` + `FastApprox` (LLK init only takes 2). Phase 4c runtime test still required: `Legacy × Fp32DestAcc × FastApprox` matrix vs torch. |
| 3 | `clobbers_sfpu_lut` per-op classification | OPEN — V1 over-conservative; only mark `true` for ops verified to write SFPU LUT (exp, log, log1p, log2, log10, tanh, sigmoid, silu, gelu, hardmish, mish, sigmoid_approx, gelu_approx, erf, erfc, erfinv, i0, i1, lgamma, digamma, polygamma, rsqrt-LUT-but-idempotent → mark `false`). May still relax for ops that touch disjoint LUT regions. Resolved during Phase 5 implementation per-op. |
| 4 | `mul_tiles_bcast` mis-listed as separate function | **CLOSED**. Verified §5: template mode only. Catalog footnote will note this; no separate struct in helper. |
| 5 | Macro-injection kernels | OUT OF SCOPE V1 per lessons §11.1 + agreed. 26 kernels deferred to separate workstream (rewrites the dispatch surface, not per-kernel migration). |
| 6 | Reduce category overlap | **CLOSED**. Reduce has its own helper (`reduce_helpers_*`); eltwise will not touch it. |
| 7 | `CumulativeWaitUpfrontEndPop` policy | **CLOSED — ADDED**. 6 production kernels need it (rmsnorm/layernorm pre-allgather × 4 + DIT + fused-rmsnorm). Policy lives in §2.3. |

---

## 11. PART 2 — Op Struct Quick Reference Table

| Op | File | Pattern | Template Params | Runtime Fields | Derived Params |
|---|---|---|---|---|---|
| Abs | math | ZERO_PARAM | Approx, Slot | – | – |
| Exp | math | ZERO_PARAM | Approx, FastApprox, Fp32, Slot | – | – |
| Log | math | ZERO_PARAM | Approx, FastApprox, Fp32, Slot | – | – |
| LogWithBase | math | NAMED_FIELDS | Approx, FastApprox, Fp32, Slot | float base_scale | – |
| Sqrt | math | ZERO_PARAM | Approx, Slot | – | – |
| Rsqrt | math | ZERO_PARAM | Approx, Fp32, FastApprox, Legacy, Slot | – | – |
| Recip | math | ZERO_PARAM | Approx, Fp32, Slot | – | – |
| Power | scalar | NAMED_FIELDS | Approx, Fp32, Slot | float exponent | – |
| LeakyRelu | activations | NAMED_FIELDS | Approx, Slot | float alpha | – |
| PRelu | activations | NAMED_FIELDS | Approx, Slot | float alpha | – |
| Elu | activations | NAMED_FIELDS | Approx, Slot | float alpha | – |
| Selu | activations | NAMED_FIELDS | Approx, Slot | float alpha, float gamma | – |
| Celu | activations | DERIVED_PARAM | Approx, Fp32, ITERATIONS, Slot | float alpha | float alpha_recip = 1.0f/alpha |
| Softplus | activations | DERIVED_PARAM | Approx, Slot | float beta, float threshold | float beta_recip = 1.0f/beta |
| Hardtanh | activations | NAMED_FIELDS | Approx, Slot | float min_val, float max_val | – |
| Threshold | activations | NAMED_FIELDS | Approx, Slot | float threshold, float value | – |
| Clamp | special | NAMED_FIELDS | Approx, Slot | float min_val, float max_val | – |
| Fmod | special | DERIVED_PARAM | Approx, Slot | float divisor | float divisor_recip |
| Remainder | special | DERIVED_PARAM | Approx, Slot | float divisor | float divisor_recip |
| UnaryAdd | scalar | NAMED_FIELDS | Approx, Slot | float scalar | – |
| UnarySub | scalar | NAMED_FIELDS | Approx, Slot | float scalar | – |
| UnaryMul | scalar | NAMED_FIELDS | Approx, Slot | float scalar | – |
| UnaryDiv | scalar | DERIVED_PARAM | Approx, Slot | float scalar | float scalar_recip |
| Mask | misc | ZERO_PARAM | DF (DataFormat), DataSlot | – | – |
| Fill | misc | NAMED_FIELDS | Slot | float value | – |
| Dropout | misc | NAMED_FIELDS | Slot | uint32_t seed, int integer_prob, int scale_factor | – |
| Rand | misc | NAMED_FIELDS | Slot | uint32_t seed | – |
| Heaviside | misc | NAMED_FIELDS | Slot | float cutoff | – |
| Round | rounding | ZERO_PARAM | RoundingMode, Slot | – | – |
| Floor / Ceil / Trunc | rounding | ZERO_PARAM | Slot | – | – |
| Eqz / Nez / Ltz / Gtz / Lez / Gez | predicates | ZERO_PARAM | Approx, Slot | – | – |
| Isnan / Isinf / Isfinite | predicates | ZERO_PARAM | Approx, Slot | – | – |
| LogicalNot | predicates | ZERO_PARAM | Approx, DF, Slot | – | – |
| AddCMul / AddCDiv / Lerp / Where | ternary | ZERO_PARAM (slot-templated) | Approx, Fp32, In0, In1, In2, Out | – | – |
| BitwiseNot | bitwise | ZERO_PARAM | Slot | – | – |
| BitwiseAnd / Or / Xor (binary) | bitwise | NAMED_FIELDS | Slot | uint32_t mask | – |
| LeftShift / RightShift | bitwise | NAMED_FIELDS | Slot | uint32_t shift | – |
| Sin / Cos / Tan / Asin / Acos / Atan / Sinh / Cosh / Tanh / AsinH / AcosH / AtanH | trig | ZERO_PARAM | Approx, Slot | – | – |
| Relu / Relu6 / Sigmoid / Silu / Gelu / Mish / Hardmish / Hardsigmoid / Softsign / Softshrink | activations | ZERO_PARAM | Approx [+ Fp32 for some], Slot | – | – |

Approximate counts: ~50 ZERO_PARAM, ~25 NAMED_FIELDS, ~5 DERIVED_PARAM, plus binary_op/DestReuseOp + ternary set.

---

Generated: 2026-04-30
Helpers proposed: 11 files (eltwise_chain core + 10 op-family files + aggregator)
Ops covered: ~95 in scope (out of 146 catalogued)
Ops excluded: reduce (other category), sfpu_int_sum/tiled_prod/reshuffle (cross-iteration / external-index), macro-injection kernels, mac (missing primitive), add_top_row, copy_dest_values, max_pool_indices
Open questions: 6 (hoist-safe sequence, rsqrt mismatch, LUT clobber per-op, catalog cleanup, macro-injection workstream, reduce category overlap)
