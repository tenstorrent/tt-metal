#ifndef MUSSY_HOOK_H
#define MUSSY_HOOK_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "types.h"


namespace llk::san {

// UNPACK EXU OPERAND STATE

struct OperandUnpack {
    template <typename T>
    using Field = StateField<StateType::OperandUnpack, OperandUnpack, T>;

    struct InputFormatA  : Field<std::uint32_t> {};
    struct OutputFormatA : Field<std::uint32_t> {};
    struct FaceHeightA   : Field<std::uint32_t> {};
    struct NumFacesA     : Field<std::uint32_t> {};
    struct InputFormatB  : Field<std::uint32_t> {};
    struct OutputFormatB : Field<std::uint32_t> {};
    struct FaceHeightB   : Field<std::uint32_t> {};
    struct NumFacesB     : Field<std::uint32_t> {};
    struct DestWidth32   : Field<bool>          {};

    using State = StateStruct<
        InputFormatA,
        OutputFormatA,
        FaceHeightA,
        NumFacesA,
        InputFormatB,
        OutputFormatB,
        FaceHeightB,
        NumFacesB,
        DestWidth32
    >;
};

// FPU EXU OPERAND STATE

struct OperandFpu {
    template <typename T>
    using Field = StateField<StateType::OperandFpu, OperandFpu, T>;

    struct Format : Field<std::uint32_t> {};

    using State = StateStruct<Format>;
};

// SFPU EXU OPERAND STATE

struct OperandSfpu {
    template <typename T>
    using Field = StateField<StateType::OperandSfpu, OperandSfpu, T>;

    struct Format : Field<std::uint32_t> {};

    using State = StateStruct<Format>;
};

// PACK EXU OPERAND STATE

struct OperandPack {
    template <typename T>
    using Field = StateField<StateType::OperandPack, OperandPack, T>;

    struct InputFormat  : Field<std::uint32_t> {};
    struct OutputFormat : Field<std::uint32_t> {};
    struct FaceHeight   : Field<std::uint32_t> {};
    struct TileWidth    : Field<std::uint32_t> {};
    struct NumFaces     : Field<std::uint32_t> {};
    struct PartialFace  : Field<std::uint32_t> {};
    struct NarrowTile   : Field<std::uint32_t> {};
    struct DestWidth32  : Field<std::uint32_t> {};

    using State = StateStruct<
        InputFormat,
        OutputFormat,
        FaceHeight,
        TileWidth,
        NumFaces,
        PartialFace,
        NarrowTile,
        DestWidth32
    >;
};

// ---------------------------------------
// OPERATION - UNPACK UNARY (AKA UNPACK_A)
// ---------------------------------------

struct OperationUnpackUnary {
    template <typename T>
    using Field = StateField<StateType::Operation, OperationUnpackUnary, T>;

    struct BroadcastType    : Field<std::uint32_t> {};
    struct AccumulateToDest : Field<std::uint32_t> {};
    struct BinaryReuseDest  : Field<std::uint32_t> {};
    struct UnpackToDest     : Field<std::uint32_t> {};

    using State = StateStruct<
        BroadcastType,
        AccumulateToDest,
        BinaryReuseDest,
        UnpackToDest
    >;
};

// -----------------------------------------
// OPERATION - UNPACK BINARY (AKA UNPACK_AB)
// -----------------------------------------

// sstanisic todo: add UNPACK BINARY operation state

// ------------------------------------------------
// OPERATION - UNPACK MATMUL (AKA UNPACK_AB_MATMUL)
// ------------------------------------------------

// sstanisic todo: add UNPACK MATMUL operation state

// ---------------------------------------------
// OPERATION - UNPACK TILIZE (AKA UNPACK_TILIZE)
// ---------------------------------------------

// sstanisic todo: add UNPACK TILIZE operation state


template <typename... params>
void configure(StateVal<params>... args) {

}

template <typename... params>
void reconfigure(StateVal<params>... args) {

}

template <typename... params>
void init(StateVal<params>... args) {
    // write operation and operation state
    // locks for dependencies.

}

template <typename... params>
void execute(StateVal<params>... args) {
    // v
}

template <typename... params>
void uninit(StateVal<params>... args) {

}


} // namespace llk::san

#endif  // !MUSSY_HOOK_H
