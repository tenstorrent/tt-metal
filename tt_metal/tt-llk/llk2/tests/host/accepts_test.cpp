// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Which named part of an op's accepts (resolver/resolver.h, Accepts) refuses a call. The Compute API prints one
// sentence per part, so the part is what decides the message a kernel author reads. Every check is a static_assert:
// compiling this file is the test (tests/hardware_free/kernels/test_llk2_accepts.py, device compiler, no card).
// Built with a 16-bit Dest and LoFi (LLK2_DEST_FP32=0, LLK2_MATH_FIDELITY=0).

#include <cstdint>

#include "resolver/copy.h"
#include "resolver/eltwise_binary.h"
#include "resolver/matmul.h"
#include "resolver/pack.h"
#include "resolver/read_value.h"
#include "resolver/reduce.h"
#include "resolver/vec.h"

using namespace llk2;

namespace
{

#define L1(name, height, width, format) inline constexpr const auto& name = l1_spec<height, width, DataFormat::format>
L1(a24, 64, 128, Float16_b);   // [2, 4] tiles
L1(b43, 128, 96, Float16_b);   // [4, 3]
L1(a24_f16, 64, 128, Float16); // 5-bit exponent
L1(a24_i32, 64, 128, Int32);
L1(s44_i32, 128, 128, Int32); // [4, 4]
L1(t11, 32, 32, Float16_b);   // one tile
L1(r14, 32, 128, Float16_b);  // one tile row
L1(wide, 96, 512, Float16_b); // [3, 16]
#undef L1
inline constexpr const auto& window   = narrowed_spec<wide, Cols, 4>; // [3, 4] of wide: not contiguous
inline constexpr const auto& rows_a24 = runtime_rows<a24>;
inline constexpr const auto& cols_r14 = runtime_cols<r14>;
inline constexpr const auto& cols_b43 = runtime_cols<b43>;
inline constexpr const auto& d24      = dst_spec<64, 128, kernel_dest_format>;
inline constexpr const auto& cols_d24 = runtime_cols<d24>;

/// Whether `part` is the one part of a that is false.
constexpr bool only(Accepts a, bool Accepts::* part)
{
    const bool refused = !(a.*part);
    a.*part            = true;
    return refused && a.all();
}

#define ACCEPTED(op, in0, in1, ...) static_assert(accepted<op, in0, in1, __VA_ARGS__>, "accepted: " #op "(" #in0 ", " #in1 ")")
#define REFUSED(part, op, in0, in1, ...)                                                         \
    static_assert(!accepted<op, in0, in1, __VA_ARGS__>, "refused: " #op "(" #in0 ", " #in1 ")"); \
    static_assert(only(accepted_parts<op, in0, in1, __VA_ARGS__>, &Accepts::part), "refused by " #part " alone: " #op "(" #in0 ", " #in1 ")")

using E = eltwise_binary_modifiers;

// matmul
ACCEPTED(matmul_op, a24, b43, matmul_modifiers {});
REFUSED(modifiers, matmul_op, a24, b43, matmul_modifiers {.accumulate = false}); // a matmul that overwrites acc: not ported
REFUSED(storage, matmul_op, d24, b43, matmul_modifiers {});
#if defined(EXPECT_FAIL) && EXPECT_FAIL == 1
REFUSED(layout, matmul_op, a24, t11, matmul_modifiers {}); // the wrong part, on purpose: this must not compile
#else
REFUSED(shapes, matmul_op, a24, t11, matmul_modifiers {});
#endif
// Integers in a 16-bit Dest: two parts say so (no Src format for them there, and they accumulate in a 32-bit Dest).
static_assert(
    !accepted_parts<matmul_op, s44_i32, s44_i32, matmul_modifiers {}>.dest_mode && !accepted_parts<matmul_op, s44_i32, s44_i32, matmul_modifiers {}>.formats);
static_assert(accepted_parts<matmul_op, s44_i32, s44_i32, matmul_modifiers {}>.shapes);
REFUSED(runtime, matmul_op, a24, cols_b43, matmul_modifiers {}); // a run-time N needs in0 to be one tile row

// add, sub, mul
ACCEPTED(add_op, a24, a24, E {});
ACCEPTED(mul_op, a24, a24, E {.accumulate = true});
REFUSED(storage, add_op, d24, a24, E {});
REFUSED(shapes, add_op, a24, t11, E {});
REFUSED(formats, add_op, a24_f16, a24, E {});
REFUSED(formats, add_op, a24_i32, a24_i32, E {});
REFUSED(modifiers, add_op, a24, a24, E {.accumulate = true});
REFUSED(runtime, sub_op, a24, rows_a24, E {});

// reduce
ACCEPTED(reduce_op, a24, t11, reduce_modifiers {});
REFUSED(shapes, reduce_op, a24, r14, reduce_modifiers {});
REFUSED(formats, reduce_op, a24_f16, t11, reduce_modifiers {});
REFUSED(runtime, reduce_op, rows_a24, t11, reduce_modifiers {});

// copy
ACCEPTED(copy_op, a24, a24, copy_modifiers {});
REFUSED(layout, copy_op, window, window, copy_modifiers {});
REFUSED(formats, copy_op, a24_i32, a24_i32, copy_modifiers {});
REFUSED(runtime, copy_op, cols_r14, cols_r14, copy_modifiers {});

// the SFPU functions
ACCEPTED(vec_op, d24, d24, vec_call {VecFunction::Silu, vec_modifiers {}});
REFUSED(storage, vec_op, a24, a24, vec_call {VecFunction::Silu, vec_modifiers {}});
REFUSED(modifiers, vec_op, d24, d24, vec_call {VecFunction::None, vec_modifiers {}});
ACCEPTED(vec_op, d24, d24, vec_call {VecFunction::Tanh, vec_modifiers {.form = VecForm::Fast}});
REFUSED(modifiers, vec_op, d24, d24, vec_call {VecFunction::Silu, vec_modifiers {.form = VecForm::Fast}}); // silu has one form
// A region: the left faces for any function whose body is one face; their even columns for exp's half-face bodies.
ACCEPTED(vec_op, d24, d24, vec_call {VecFunction::Silu, vec_modifiers {.region = VecRegion::LeftFaces}});
ACCEPTED(vec_op, d24, d24, vec_call {VecFunction::Reciprocal, vec_modifiers {.form = VecForm::Fast, .region = VecRegion::LeftFaces}});
ACCEPTED(vec_op, d24, d24, vec_call {VecFunction::Exp, vec_modifiers {.region = VecRegion::LeftFacesEvenColumns}});
ACCEPTED(vec_op, d24, d24, vec_call {VecFunction::Exp, vec_modifiers {.form = VecForm::Polynomial, .region = VecRegion::LeftFacesEvenColumns}});
REFUSED(modifiers, vec_op, d24, d24, vec_call {VecFunction::Exp, vec_modifiers {.form = VecForm::Polynomial}}); // no whole-face polynomial
REFUSED(modifiers, vec_op, d24, d24, vec_call {VecFunction::Tanh, vec_modifiers {.form = VecForm::Polynomial, .region = VecRegion::LeftFacesEvenColumns}});
REFUSED(modifiers, vec_op, d24, d24, vec_call {VecFunction::Reciprocal, vec_modifiers {.region = VecRegion::LeftFacesEvenColumns}});
// The fast exp: the whole tile, and only without input clamping; no other form is unclamped.
ACCEPTED(vec_op, d24, d24, vec_call {VecFunction::Exp, vec_modifiers {.form = VecForm::Fast, .clamp_input = false}});
REFUSED(modifiers, vec_op, d24, d24, vec_call {VecFunction::Exp, vec_modifiers {.form = VecForm::Fast}});
REFUSED(modifiers, vec_op, d24, d24, vec_call {VecFunction::Exp, vec_modifiers {.form = VecForm::Fast, .region = VecRegion::LeftFaces, .clamp_input = false}});
REFUSED(modifiers, vec_op, d24, d24, vec_call {VecFunction::Exp, vec_modifiers {.clamp_input = false}});
// An input scale is exp's.
ACCEPTED(vec_op, d24, d24, vec_call {VecFunction::Exp, vec_modifiers {.input_scale = 0x40000000u}});
REFUSED(modifiers, vec_op, d24, d24, vec_call {VecFunction::Silu, vec_modifiers {.input_scale = 0x40000000u}});
REFUSED(runtime, vec_op, cols_d24, cols_d24, vec_call {VecFunction::Silu, vec_modifiers {}}); // two tile rows

// pack
ACCEPTED(pack_op, d24, a24, pack_modifiers {});
REFUSED(storage, pack_op, a24, a24, pack_modifiers {});
REFUSED(shapes, pack_op, d24, r14, pack_modifiers {});
REFUSED(modifiers, pack_op, d24, a24, pack_modifiers {.untilize = true});

// read_value: its rule is an accepts without an OpApi (no unit runs, so no OpConfig, Work or Plan).
inline constexpr const auto& front = owned_by<t11, Thread::Unpack, Exu::Unpacker>; // a tensor at a buffer's front
static_assert(read_value_accepts<std::uint32_t>(front).all());
static_assert(read_value_accepts<std::uint16_t>(front).all());
static_assert(only(read_value_accepts<std::uint32_t>(t11), &Accepts::storage)); // nobody's: not from a buffer
static_assert(only(read_value_accepts<std::uint32_t>(d24), &Accepts::storage)); // Dest
static_assert(only(read_value_accepts<std::int32_t>(front), &Accepts::formats));
static_assert(only(read_value_accepts<std::uint64_t>(front), &Accepts::formats));

} // namespace
