// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define LLK_SAN_ENABLE

#include <cstdint>

#include "params.h"
#include "sanitizer/impl.h"

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    llk::san::UnpackOperandState unpack_state;

    const llk::san::UnpackSrcState& src_a = unpack_state.src_a;
    const llk::san::UnpackSrcState& src_b = unpack_state.src_b;

    // state gets default initialized to UNKNOWN

    LLK_ASSERT(src_a.input_format.is_unknown(), "src_a.input_format must be in unknown state");
    LLK_ASSERT(src_a.output_format.is_unknown(), "src_a.output_format must be in unknown state");
    LLK_ASSERT(src_a.face_height.is_unknown(), "src_a.face_height must be in unknown state");
    LLK_ASSERT(src_a.num_faces.is_unknown(), "src_a.num_faces must be in unknown state");
    LLK_ASSERT(src_b.input_format.is_unknown(), "src_b.input_format must be in unknown state");
    LLK_ASSERT(src_b.output_format.is_unknown(), "src_b.output_format must be in unknown state");
    LLK_ASSERT(src_b.face_height.is_unknown(), "src_b.face_height must be in unknown state");
    LLK_ASSERT(src_b.num_faces.is_unknown(), "src_b.num_faces must be in unknown state");
    LLK_ASSERT(unpack_state.dest_width_32.is_unknown(), "unpack_state.dest_width_32 must be in unknown state");
    LLK_ASSERT(!unpack_state.is_configured, "unpack_state must be not configured");

    // configure state

    llk::san::unpack_operand_configure_impl<false>(
        unpack_state,
        true /* dst_acc_en */,
        1 /* src_fmt_A */,
        2 /* src_fmt_B */,
        3 /* dst_fmt_A */,
        4 /* dst_fmt_B */,
        5 /* face_height_A */,
        6 /* face_height_B */,
        7 /* num_faces_A */,
        8 /* num_faces_B */
    );

    LLK_ASSERT(src_a.input_format.is_known(), "src_a.input_format must be in known state");
    LLK_ASSERT(src_a.input_format.get_underlying() == 1, "src_a.input_format must be set to 1");
    LLK_ASSERT(src_a.output_format.is_known(), "src_a.output_format must be in known state");
    LLK_ASSERT(src_a.output_format.get_underlying() == 3, "src_a.output_format must be set to 3");
    LLK_ASSERT(src_a.face_height.is_known(), "src_a.face_height must be in known state");
    LLK_ASSERT(src_a.face_height.get_underlying() == 5, "src_a.face_height must be set to 5");
    LLK_ASSERT(src_a.num_faces.is_known(), "src_a.num_faces must be in known state");
    LLK_ASSERT(src_a.num_faces.get_underlying() == 7, "src_a.num_faces must be set to 7");
    LLK_ASSERT(src_b.input_format.is_known(), "src_b.input_format must be in known state");
    LLK_ASSERT(src_b.input_format.get_underlying() == 2, "src_b.input_format must be set to 2");
    LLK_ASSERT(src_b.output_format.is_known(), "src_b.output_format must be in known state");
    LLK_ASSERT(src_b.output_format.get_underlying() == 4, "src_b.output_format must be set to 4");
    LLK_ASSERT(src_b.face_height.is_known(), "src_b.face_height must be in known state");
    LLK_ASSERT(src_b.face_height.get_underlying() == 6, "src_b.face_height must be set to 6");
    LLK_ASSERT(src_b.num_faces.is_known(), "src_b.num_faces must be in known state");
    LLK_ASSERT(src_b.num_faces.get_underlying() == 8, "src_b.num_faces must be set to 8");
    LLK_ASSERT(unpack_state.dest_width_32.is_known(), "unpack_state.dest_width_32 must be in known state");
    LLK_ASSERT(unpack_state.dest_width_32.get_underlying() == true, "unpack_state.dest_width_32 must be set to true");
    LLK_ASSERT(unpack_state.is_configured, "unpack_state must be configured");

    // reconfigure state

    llk::san::unpack_operand_configure_impl<true>(
        unpack_state,
        false /* dst_acc_en */,
        llk::san::IGNORE /* src_fmt_A */,
        9 /* src_fmt_B */,
        llk::san::IGNORE /* dst_fmt_A */,
        10 /* dst_fmt_B */,
        llk::san::IGNORE /* face_height_A */,
        11 /* face_height_B */,
        llk::san::IGNORE /* num_faces_A */,
        12 /* num_faces_B */
    );

    LLK_ASSERT(src_a.input_format.is_known(), "src_a.input_format must be in known state");
    LLK_ASSERT(src_a.input_format.get_underlying() == 1, "src_a.input_format must be set to 1");
    LLK_ASSERT(src_a.output_format.is_known(), "src_a.output_format must be in known state");
    LLK_ASSERT(src_a.output_format.get_underlying() == 3, "src_a.output_format must be set to 3");
    LLK_ASSERT(src_a.face_height.is_known(), "src_a.face_height must be in known state");
    LLK_ASSERT(src_a.face_height.get_underlying() == 5, "src_a.face_height must be set to 5");
    LLK_ASSERT(src_a.num_faces.is_known(), "src_a.num_faces must be in known state");
    LLK_ASSERT(src_a.num_faces.get_underlying() == 7, "src_a.num_faces must be set to 7");
    LLK_ASSERT(src_b.input_format.is_known(), "src_b.input_format must be in known state");
    LLK_ASSERT(src_b.input_format.get_underlying() == 9, "src_b.input_format must be set to 9");
    LLK_ASSERT(src_b.output_format.is_known(), "src_b.output_format must be in known state");
    LLK_ASSERT(src_b.output_format.get_underlying() == 10, "src_b.output_format must be set to 10");
    LLK_ASSERT(src_b.face_height.is_known(), "src_b.face_height must be in known state");
    LLK_ASSERT(src_b.face_height.get_underlying() == 11, "src_b.face_height must be set to 11");
    LLK_ASSERT(src_b.num_faces.is_known(), "src_b.num_faces must be in known state");
    LLK_ASSERT(src_b.num_faces.get_underlying() == 12, "src_b.num_faces must be set to 12");
    LLK_ASSERT(unpack_state.dest_width_32.is_known(), "unpack_state.dest_width_32 must be in known state");
    LLK_ASSERT(unpack_state.dest_width_32.get_underlying() == false, "unpack_state.dest_width_32 must be set to false");
    LLK_ASSERT(unpack_state.is_configured, "unpack_state must be configured");

    // scramble parts of the state
    llk::san::unpack_operand_configure_impl<true>(
        unpack_state,
        llk::san::UNKNOWN /* dst_acc_en */,
        llk::san::IGNORE /* src_fmt_A */,
        llk::san::UNKNOWN /* src_fmt_B */,
        llk::san::UNKNOWN /* dst_fmt_A */,
        llk::san::UNKNOWN /* dst_fmt_B */,
        llk::san::IGNORE /* face_height_A */,
        llk::san::UNKNOWN /* face_height_B */,
        llk::san::UNKNOWN /* num_faces_A */,
        llk::san::UNKNOWN /* num_faces_B */
    );

    LLK_ASSERT(src_a.input_format.is_known(), "src_a.input_format must be in known state");
    LLK_ASSERT(src_a.input_format.get_underlying() == 1, "src_a.input_format must be set to 1");
    LLK_ASSERT(src_a.output_format.is_unknown(), "src_a.output_format must be in unknown state");
    LLK_ASSERT(src_a.face_height.is_known(), "src_a.face_height must be in known state");
    LLK_ASSERT(src_a.face_height.get_underlying() == 5, "src_a.face_height must be set to 5");
    LLK_ASSERT(src_a.num_faces.is_unknown(), "src_a.num_faces must be in unknown state");
    LLK_ASSERT(src_b.input_format.is_unknown(), "src_b.input_format must be in unknown state");
    LLK_ASSERT(src_b.output_format.is_unknown(), "src_b.output_format must be in unknown state");
    LLK_ASSERT(src_b.face_height.is_unknown(), "src_b.face_height must be in unknown state");
    LLK_ASSERT(src_b.num_faces.is_unknown(), "src_b.num_faces must be in unknown state");
    LLK_ASSERT(unpack_state.dest_width_32.is_unknown(), "unpack_state.dest_width_32 must be in unknown state");
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS params)
{
    llk::san::MathOperandState math_state;

    const llk::san::MathSrcState& src_a = math_state.src_a;
    const llk::san::MathSrcState& src_b = math_state.src_b;

    // state gets default initialized to UNKNOWN

    LLK_ASSERT(src_a.input_format.is_unknown(), "src_a.input_format must be in unknown state");
    LLK_ASSERT(src_b.input_format.is_unknown(), "src_b.input_format must be in unknown state");
    LLK_ASSERT(!math_state.is_configured, "math_state must be not configured");

    // configure state

    llk::san::math_operand_configure_impl<false>(
        math_state, 1 /* math_fmt_A */, 2 /* math_fmt_B */
    );

    LLK_ASSERT(src_a.input_format.is_known(), "src_a.input_format must be in known state");
    LLK_ASSERT(src_a.input_format.get_underlying() == 1, "src_a.input_format must be set to 1");
    LLK_ASSERT(src_b.input_format.is_known(), "src_b.input_format must be in known state");
    LLK_ASSERT(src_b.input_format.get_underlying() == 2, "src_b.input_format must be set to 2");
    LLK_ASSERT(math_state.is_configured, "math_state must be configured");

    // reconfigure state

    llk::san::math_operand_configure_impl<true>(
        math_state, llk::san::IGNORE /* math_fmt_A */, 9 /* math_fmt_B */
    );

    LLK_ASSERT(src_a.input_format.is_known(), "src_a.input_format must be in known state");
    LLK_ASSERT(src_a.input_format.get_underlying() == 1, "src_a.input_format must be set to 1");
    LLK_ASSERT(src_b.input_format.is_known(), "src_b.input_format must be in known state");
    LLK_ASSERT(src_b.input_format.get_underlying() == 9, "src_b.input_format must be set to 9");
    LLK_ASSERT(math_state.is_configured, "math_state must be configured");

    // scramble parts of the state
    llk::san::math_operand_configure_impl<true>(
        math_state, llk::san::UNKNOWN /* math_fmt_A */, llk::san::IGNORE /* math_fmt_B */
    );

    LLK_ASSERT(src_a.input_format.is_unknown(), "src_a.input_format must be in unknown state");
    LLK_ASSERT(src_b.input_format.is_known(), "src_b.input_format must be in known state");
    LLK_ASSERT(src_b.input_format.get_underlying() == 9, "src_b.input_format must be set to 9");
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    llk::san::PackOperandState pack_state;

    // state gets default initialized to UNKNOWN

    LLK_ASSERT(pack_state.input_format.is_unknown(), "pack_state.input_format must be in unknown state");
    LLK_ASSERT(pack_state.output_format.is_unknown(), "pack_state.output_format must be in unknown state");
    LLK_ASSERT(pack_state.face_height.is_unknown(), "pack_state.face_height must be in unknown state");
    LLK_ASSERT(pack_state.tile_width.is_unknown(), "pack_state.tile_width must be in unknown state");
    LLK_ASSERT(pack_state.num_faces.is_unknown(), "pack_state.num_faces must be in unknown state");
    LLK_ASSERT(pack_state.partial_face.is_unknown(), "pack_state.partial_face must be in unknown state");
    LLK_ASSERT(pack_state.narrow_tile.is_unknown(), "pack_state.narrow_tile must be in unknown state");
    LLK_ASSERT(pack_state.dest_width_32.is_unknown(), "pack_state.dest_width_32 must be in unknown state");
    LLK_ASSERT(!pack_state.is_configured, "pack_state must be not configured");

    // configure state

    llk::san::pack_operand_configure_impl<false>(
        pack_state,
        true /* dest_acc_en */,
        1 /* src_fmt */,
        2 /* dst_fmt */,
        3 /* face_height */,
        4 /* tile_width */,
        5 /* num_faces */,
        true /* partial_face */,
        false /* narrow_tile */
    );

    LLK_ASSERT(pack_state.input_format.is_known(), "pack_state.input_format must be in known state");
    LLK_ASSERT(pack_state.input_format.get_underlying() == 1, "pack_state.input_format must be set to 1");
    LLK_ASSERT(pack_state.output_format.is_known(), "pack_state.output_format must be in known state");
    LLK_ASSERT(pack_state.output_format.get_underlying() == 2, "pack_state.output_format must be set to 2");
    LLK_ASSERT(pack_state.face_height.is_known(), "pack_state.face_height must be in known state");
    LLK_ASSERT(pack_state.face_height.get_underlying() == 3, "pack_state.face_height must be set to 3");
    LLK_ASSERT(pack_state.tile_width.is_known(), "pack_state.tile_width must be in known state");
    LLK_ASSERT(pack_state.tile_width.get_underlying() == 4, "pack_state.tile_width must be set to 4");
    LLK_ASSERT(pack_state.num_faces.is_known(), "pack_state.num_faces must be in known state");
    LLK_ASSERT(pack_state.num_faces.get_underlying() == 5, "pack_state.num_faces must be set to 5");
    LLK_ASSERT(pack_state.partial_face.is_known(), "pack_state.partial_face must be in known state");
    LLK_ASSERT(pack_state.partial_face.get_underlying() == true, "pack_state.partial_face must be set to true");
    LLK_ASSERT(pack_state.narrow_tile.is_known(), "pack_state.narrow_tile must be in known state");
    LLK_ASSERT(pack_state.narrow_tile.get_underlying() == false, "pack_state.narrow_tile must be set to false");
    LLK_ASSERT(pack_state.dest_width_32.is_known(), "pack_state.dest_width_32 must be in known state");
    LLK_ASSERT(pack_state.dest_width_32.get_underlying() == true, "pack_state.dest_width_32 must be set to true");
    LLK_ASSERT(pack_state.is_configured, "pack_state must be configured");

    // reconfigure state

    llk::san::pack_operand_configure_impl<true>(
        pack_state,
        false /* dest_acc_en */,
        llk::san::IGNORE /* src_fmt */,
        9 /* dst_fmt */,
        llk::san::IGNORE /* face_height */,
        10 /* tile_width */,
        llk::san::IGNORE /* num_faces */,
        false /* partial_face */,
        true /* narrow_tile */
    );

    LLK_ASSERT(pack_state.input_format.is_known(), "pack_state.input_format must be in known state");
    LLK_ASSERT(pack_state.input_format.get_underlying() == 1, "pack_state.input_format must be set to 1");
    LLK_ASSERT(pack_state.output_format.is_known(), "pack_state.output_format must be in known state");
    LLK_ASSERT(pack_state.output_format.get_underlying() == 9, "pack_state.output_format must be set to 9");
    LLK_ASSERT(pack_state.face_height.is_known(), "pack_state.face_height must be in known state");
    LLK_ASSERT(pack_state.face_height.get_underlying() == 3, "pack_state.face_height must be set to 3");
    LLK_ASSERT(pack_state.tile_width.is_known(), "pack_state.tile_width must be in known state");
    LLK_ASSERT(pack_state.tile_width.get_underlying() == 10, "pack_state.tile_width must be set to 10");
    LLK_ASSERT(pack_state.num_faces.is_known(), "pack_state.num_faces must be in known state");
    LLK_ASSERT(pack_state.num_faces.get_underlying() == 5, "pack_state.num_faces must be set to 5");
    LLK_ASSERT(pack_state.partial_face.is_known(), "pack_state.partial_face must be in known state");
    LLK_ASSERT(pack_state.partial_face.get_underlying() == false, "pack_state.partial_face must be set to false");
    LLK_ASSERT(pack_state.narrow_tile.is_known(), "pack_state.narrow_tile must be in known state");
    LLK_ASSERT(pack_state.narrow_tile.get_underlying() == true, "pack_state.narrow_tile must be set to true");
    LLK_ASSERT(pack_state.dest_width_32.is_known(), "pack_state.dest_width_32 must be in known state");
    LLK_ASSERT(pack_state.dest_width_32.get_underlying() == false, "pack_state.dest_width_32 must be set to false");
    LLK_ASSERT(pack_state.is_configured, "pack_state must be configured");

    // scramble parts of the state
    llk::san::pack_operand_configure_impl<true>(
        pack_state,
        llk::san::UNKNOWN /* dest_acc_en */,
        llk::san::IGNORE /* src_fmt */,
        llk::san::UNKNOWN /* dst_fmt */,
        llk::san::UNKNOWN /* face_height */,
        llk::san::IGNORE /* tile_width */,
        llk::san::UNKNOWN /* num_faces */,
        llk::san::IGNORE /* partial_face */,
        llk::san::UNKNOWN /* narrow_tile */
    );

    LLK_ASSERT(pack_state.input_format.is_known(), "pack_state.input_format must be in known state");
    LLK_ASSERT(pack_state.input_format.get_underlying() == 1, "pack_state.input_format must be set to 1");
    LLK_ASSERT(pack_state.output_format.is_unknown(), "pack_state.output_format must be in unknown state");
    LLK_ASSERT(pack_state.face_height.is_unknown(), "pack_state.face_height must be in unknown state");
    LLK_ASSERT(pack_state.tile_width.is_known(), "pack_state.tile_width must be in known state");
    LLK_ASSERT(pack_state.tile_width.get_underlying() == 10, "pack_state.tile_width must be set to 10");
    LLK_ASSERT(pack_state.num_faces.is_unknown(), "pack_state.num_faces must be in unknown state");
    LLK_ASSERT(pack_state.partial_face.is_known(), "pack_state.partial_face must be in known state");
    LLK_ASSERT(pack_state.partial_face.get_underlying() == false, "pack_state.partial_face must be set to false");
    LLK_ASSERT(pack_state.narrow_tile.is_unknown(), "pack_state.narrow_tile must be in unknown state");
    LLK_ASSERT(pack_state.dest_width_32.is_unknown(), "pack_state.dest_width_32 must be in unknown state");
}

#endif
