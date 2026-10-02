// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_diagnose} %{blackhole_math_thread} %s

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using SrcAToDst = move::Transfer<move::SrcA, move::Dst>;
using SrcBToDst = move::Transfer<move::SrcB, move::Dst>;
using DstToSrcB = move::Transfer<move::Dst, move::SrcB>;
using L1ToGpr   = move::Transfer<move::L1, move::Gpr>;
using GprToL1   = move::Transfer<move::Gpr, move::L1>;
using GprToSrcA = move::Transfer<move::Gpr, move::SrcA>;
using GprToSrcB = move::Transfer<move::Gpr, move::SrcB>;
using GprToMmio = move::Transfer<move::Gpr, move::Mmio>;
using MmioToGpr = move::Transfer<move::Mmio, move::Gpr>;
using L1ToL1    = move::Transfer<move::L1, move::L1>;

// Shared shapes.
static_assert(move::rows::One.value == 1 && move::rows::Four.value == 4 && move::rows::Eight.value == 8 && move::rows::Face.value == 16);
static_assert(move::rows::faces<1>().value == 16 && move::rows::faces<4095>().value == 65520);
static_assert(move::is_canonical(move::rows::One) && move::is_canonical(move::rows::Four) && move::is_canonical(move::rows::Eight));
static_assert(move::is_canonical(move::rows::faces<3>()));
static_assert(!move::is_canonical({0}) && !move::is_canonical({2}) && !move::is_canonical({12}) && !move::is_canonical({24}));
static_assert(move::dst_layout::RowsPerFace == 16 && move::dst_layout::FacesPerTile == 4 && move::dst_layout::RowsPerTile == 64);
static_assert(move::dst_layout::tile_row(0) == 0 && move::dst_layout::tile_row(15, 3, 15) == 1023);
static_assert(move::relative_row<-512>() == 512 && move::relative_row<511>() == 511 && move::relative_row<0>() == 0);
static_assert(move::relative_row(-1) == 1023);
static_assert(move::MaxMatrixTransferInstructions == 8);

// Single-instruction encodings match the raw instruction macros.
static_assert(
    SrcAToDst {.source_row = 3, .destination_row = 17, .address_mode = 7, .number_of_rows = move::rows::One}.get_operation() == TT_OP_MOVA2D(0, 3, 7, 0, 17));
static_assert(
    SrcAToDst {
        .source_row            = 56,
        .destination_row       = 1016,
        .address_mode          = 2,
        .number_of_rows        = move::rows::Eight,
        .destination_32bit_low = true,
        .source_validity       = move::SourceValidity::Ignore}
        .get_operation() == TT_OP_MOVDBGA2D(1, 56, 2, 2, 1016));
static_assert(
    SrcBToDst {.source_row = 5, .destination_row = 27, .address_mode = 6, .number_of_rows = move::rows::One, .broadcast = move::Broadcast::Column0}
        .get_operation() == TT_OP_MOVB2D(0, 5, 6, 1, 27));
static_assert(
    SrcBToDst {.source_row = 4, .destination_row = 32, .address_mode = 1, .number_of_rows = move::rows::Four}.get_operation() == TT_OP_MOVB2D(0, 4, 1, 4, 32));
static_assert(
    SrcBToDst {.source_row = 63, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Eight, .broadcast = move::Broadcast::Row}
        .get_operation() == TT_OP_MOVB2D(0, 63, 0, 2, 0));
static_assert(
    SrcBToDst {
        .source_row            = 63,
        .destination_row       = 40,
        .address_mode          = 4,
        .number_of_rows        = move::rows::Eight,
        .broadcast             = move::Broadcast::Scalar,
        .destination_32bit_low = true,
        .source_validity       = move::SourceValidity::Ignore}
        .get_operation() == TT_OP_MOVDBGB2D(1, 63, 4, 3, 40));
static_assert(
    DstToSrcB {.source_row = 19, .destination_row = 16, .address_mode = 3, .number_of_rows = move::rows::One, .source_32bit_low = true}.get_operation() ==
    TT_OP_MOVD2B(1, 16, 3, 0, 19));
static_assert(
    DstToSrcB {.source_row = 1020, .destination_row = 60, .address_mode = 0, .number_of_rows = move::rows::Four}.get_operation() ==
    TT_OP_MOVD2B(0, 60, 0, 2, 1020));
static_assert(
    L1ToGpr {
        .source = {.base = hal::gpr<1>(), .offset = {.gpr = hal::gpr<2>(), .half = move::GprHalf::High}, .offset_increment = move::OffsetIncrement::Bytes4},
        .destination = hal::gpr<3>(),
        .size        = move::ScalarSize::Bytes4}
        .get_operation() == TT_OP_LOADIND(1, 5, 2, 3, 1));
static_assert(
    GprToL1 {
        .source = hal::gpr<60>(),
        .destination =
            {.base = hal::gpr<5>(), .offset = {.gpr = hal::gpr<63>(), .half = move::GprHalf::High}, .offset_increment = move::OffsetIncrement::Bytes16}}
        .get_operation() == TT_OP_STOREIND(1, 0, 0, 127, 3, 60, 5));
static_assert(
    GprToSrcA {
        .source = hal::gpr<8>(),
        .destination =
            {.base = hal::gpr<9>(), .offset = {.gpr = hal::gpr<10>(), .half = move::GprHalf::High}, .offset_increment = move::OffsetIncrement::Bytes2}}
        .get_operation() == TT_OP_STOREIND(0, 0, 0, 21, 1, 8, 9));
static_assert(
    GprToSrcB {.source = hal::gpr<12>(), .destination = {.base = hal::gpr<13>(), .offset = {.gpr = hal::gpr<14>()}}}.get_operation() ==
    TT_OP_STOREIND(0, 0, 1, 28, 0, 12, 13));
static_assert(
    GprToMmio {.source = hal::gpr<15>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb11000u}}.get_operation() ==
    TT_OP_STOREREG(15, 0x4400));
static_assert(
    GprToMmio {
        .source      = hal::gpr<16>(),
        .destination = {.addressing = move::MmioAddressing::Indirect, .base = hal::gpr<17>(), .offset = {.gpr = hal::gpr<18>(), .half = move::GprHalf::High}}}
        .get_operation() == TT_OP_STOREIND(0, 1, 0, 37, 0, 16, 17));
static_assert(
    MmioToGpr {.source = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffbffffcu}, .destination = hal::gpr<19>()}.get_operation() ==
    TT_OP_LOADREG(19, 0x3ffff));

// Matrix descriptor boundaries.
static_assert(!move::is_valid(SrcAToDst {}));
static_assert(move::is_valid(SrcAToDst {.source_row = 56, .destination_row = 1016, .address_mode = 7, .number_of_rows = move::rows::Eight}));
static_assert(!move::is_valid(SrcAToDst {.source_row = 57, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Eight}));
static_assert(!move::is_valid(SrcAToDst {.source_row = 0, .destination_row = 1017, .address_mode = 0, .number_of_rows = move::rows::Eight}));
static_assert(move::is_valid(SrcAToDst {
    .source_row             = 0,
    .destination_row        = 1023,
    .destination_addressing = move::RowAddressing::CounterRelative,
    .address_mode           = 0,
    .number_of_rows         = move::rows::Eight}));
static_assert(!move::is_valid(SrcAToDst {
    .source_row             = 0,
    .destination_row        = 1024,
    .destination_addressing = move::RowAddressing::CounterRelative,
    .address_mode           = 0,
    .number_of_rows         = move::rows::One}));
static_assert(move::is_valid(SrcAToDst {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::faces<4>()}));
static_assert(!move::is_valid(SrcAToDst {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::faces<5>()}));
static_assert(!move::is_valid(SrcAToDst {
    .source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::One, .final_address_mode = 1}));
static_assert(move::is_valid(SrcAToDst {
    .source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Four, .final_address_mode = 7}));
static_assert(!move::is_valid(SrcAToDst {
    .source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Four, .final_address_mode = 8}));

static_assert(move::is_valid(SrcBToDst {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::faces<2>()}));
static_assert(!move::is_valid(SrcBToDst {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::faces<3>()}));
static_assert(move::is_valid(SrcBToDst {
    .source_row = 63, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::faces<4>(), .broadcast = move::Broadcast::Row}));
static_assert(!move::is_valid(SrcBToDst {
    .source_row = 63, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::faces<4>(), .broadcast = move::Broadcast::None}));
static_assert(!move::is_valid(SrcBToDst {
    .source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::One, .broadcast = move::Broadcast::Row}));
static_assert(move::is_valid(SrcBToDst {
    .source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::One, .broadcast = move::Broadcast::Scalar}));

static_assert(!move::is_valid(DstToSrcB {}));
static_assert(move::is_valid(DstToSrcB {.source_row = 1008, .destination_row = 48, .address_mode = 0, .number_of_rows = move::rows::Face}));
static_assert(!move::is_valid(DstToSrcB {.source_row = 1009, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Face}));
static_assert(!move::is_valid(DstToSrcB {.source_row = 0, .destination_row = 49, .address_mode = 0, .number_of_rows = move::rows::Face}));
static_assert(move::is_valid(DstToSrcB {
    .source_row        = 1020,
    .source_addressing = move::RowAddressing::CounterRelative,
    .destination_row   = 0,
    .address_mode      = 0,
    .number_of_rows    = move::rows::Face}));
static_assert(!move::is_valid(DstToSrcB {
    .source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::One, .final_address_mode = 3}));

// Scalar and XMOV descriptor boundaries.
static_assert(move::is_valid(L1ToGpr {.source = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}, .destination = hal::gpr<60>()}));
static_assert(!move::is_valid(L1ToGpr {.source = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}, .destination = hal::gpr<62>()}));
static_assert(move::is_valid(L1ToGpr {
    .source = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}, .destination = hal::gpr<63>(), .size = move::ScalarSize::Bytes1}));
static_assert(!move::is_valid(L1ToGpr {.source = {.base = hal::gpr<0>()}, .destination = hal::gpr<4>()}));
static_assert(!move::is_valid(GprToL1 {.source = hal::gpr<2>(), .destination = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}}));
static_assert(move::is_valid(GprToL1 {
    .source = hal::gpr<2>(), .destination = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}, .size = move::ScalarSize::Bytes2}));
static_assert(!move::is_valid(GprToSrcA {.source = hal::gpr<1>(), .destination = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}}));
static_assert(!move::is_valid(GprToSrcB {.source = hal::gpr<64>(), .destination = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}}));
static_assert(!move::is_valid(GprToMmio {
    .source = hal::gpr<0>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb11002u}}));
static_assert(!move::is_valid(GprToMmio {
    .source = hal::gpr<0>(), .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb10ffcu}}));
static_assert(!move::is_valid(GprToMmio {.source = hal::gpr<0>()}));
static_assert(!move::is_valid(MmioToGpr {
    .source = {.addressing = move::MmioAddressing::Indirect, .base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}}, .destination = hal::gpr<2>()}));
static_assert(move::is_valid(L1ToL1 {.source = {.block = 0}, .destination = {.block = 0xc000}, .size = {.value = 0xc000}}));
static_assert(!move::is_valid(L1ToL1 {.source = {.block = 0}, .destination = {.block = 0x10000}, .size = {.value = 0x10000}}));
static_assert(!move::is_valid(L1ToL1 {.source = {.block = 0}, .destination = {.block = 1}, .size = {.value = 0}}));
static_assert(!move::is_valid(L1ToL1 {.source = {.block = 0x100}, .destination = {.block = 0x10f}, .size = {.value = 16}}));
static_assert(move::is_valid(L1ToL1 {.source = {.block = MEM_L1_SIZE / 16u - 1u}, .destination = {.block = 0}, .size = {.value = 1}}));
static_assert(!move::is_valid(L1ToL1 {.source = {.block = MEM_L1_SIZE / 16u - 1u}, .destination = {.block = 0}, .size = {.value = 2}}));
