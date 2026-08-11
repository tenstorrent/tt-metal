#pragma once
#include <cstdint>

#include "sanitizer/types.h"

namespace llk::san
{

// --------------------
// OPERAND - UNPACK EXU
// --------------------

template <>
struct Operand<Exu::Unpack>
{
    template <typename T>
    using Field = StateField<Operand<Exu::Unpack>, T>;

    struct InputFormatA : Field<std::uint32_t>
    {
    };

    struct OutputFormatA : Field<std::uint32_t>
    {
    };

    struct FaceHeightA : Field<std::uint32_t>
    {
    };

    struct NumFacesA : Field<std::uint32_t>
    {
    };

    struct InputFormatB : Field<std::uint32_t>
    {
    };

    struct OutputFormatB : Field<std::uint32_t>
    {
    };

    struct FaceHeightB : Field<std::uint32_t>
    {
    };

    struct NumFacesB : Field<std::uint32_t>
    {
    };

    struct DestWidth32 : Field<bool>
    {
    };

    using Struct = StateStruct<
        Operand<Exu::Unpack>,
        /* Fields */
        InputFormatA,
        OutputFormatA,
        FaceHeightA,
        NumFacesA,
        InputFormatB,
        OutputFormatB,
        FaceHeightB,
        NumFacesB,
        DestWidth32>;
};

// -----------------
// OPERAND - FPU EXU
// -----------------

template <>
struct Operand<Exu::Fpu>
{
    template <typename T>
    using Field = StateField<Operand<Exu::Fpu>, T>;

    struct Format : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<
        Operand<Exu::Fpu>,
        /* Fields */
        Format>;
};

// ------------------
// OPERAND - SFPU EXU
// ------------------

template <>
struct Operand<Exu::Sfpu>
{
    template <typename T>
    using Field = StateField<Operand<Exu::Sfpu>, T>;

    struct Format : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<
        Operand<Exu::Sfpu>,
        /* Fields */
        Format>;
};

// ------------------
// OPERAND - PACK EXU
// ------------------

template <>
struct Operand<Exu::Pack>
{
    template <typename T>
    using Field = StateField<Operand<Exu::Pack>, T>;

    struct InputFormat : Field<std::uint32_t>
    {
    };

    struct OutputFormat : Field<std::uint32_t>
    {
    };

    struct FaceHeight : Field<std::uint32_t>
    {
    };

    struct TileWidth : Field<std::uint32_t>
    {
    };

    struct NumFaces : Field<std::uint32_t>
    {
    };

    struct PartialFace : Field<std::uint32_t>
    {
    };

    struct NarrowTile : Field<std::uint32_t>
    {
    };

    struct DestWidth32 : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<
        Operand<Exu::Pack>,
        /* Fields */
        InputFormat,
        OutputFormat,
        FaceHeight,
        TileWidth,
        NumFaces,
        PartialFace,
        NarrowTile,
        DestWidth32>;
};

// ---------------------------------------
// OPERATION - UNPACK UNARY (AKA UNPACK_A)
// ---------------------------------------

struct OperationUnpackUnary : Operation<Exu::Unpack, Hoistable::Yes>
{
    template <typename T>
    using Field = StateField<OperationUnpackUnary, T>;

    struct BroadcastType : Field<std::uint32_t>
    {
    };

    struct AccumulateToDest : Field<std::uint32_t>
    {
    };

    struct BinaryReuseDest : Field<std::uint32_t>
    {
    };

    struct UnpackToDest : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<
        OperationUnpackUnary,
        /* Fields */
        BroadcastType,
        AccumulateToDest,
        BinaryReuseDest,
        UnpackToDest>;
};

// -----------------------------------------
// OPERATION - UNPACK BINARY (AKA UNPACK_AB)
// -----------------------------------------

struct OperationUnpackBinary : Operation<Exu::Unpack, Hoistable::Yes>
{
    template <typename T>
    using Field = StateField<OperationUnpackBinary, T>;

    struct BroadcastType : Field<std::uint32_t>
    {
    };

    struct FaceWidth : Field<std::uint32_t>
    {
    };

    struct NumFacesRow : Field<std::uint32_t>
    {
    };

    struct NumFacesCol : Field<std::uint32_t>
    {
    };

    struct Transpose : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<
        OperationUnpackBinary,
        /* Fields */
        BroadcastType,
        FaceWidth,
        NumFacesRow,
        NumFacesCol,
        Transpose>;
};

// ------------------------------------------------
// OPERATION - UNPACK MATMUL (AKA UNPACK_AB_MATMUL)
// ------------------------------------------------

struct OperationUnpackMatmul : Operation<Exu::Unpack, Hoistable::Yes>
{
    template <typename T>
    using Field = StateField<OperationUnpackMatmul, T>;

    struct KernelBroadcastA : Field<std::uint32_t>
    {
    };

    struct KernelBroadcastB : Field<std::uint32_t>
    {
    };

    struct Transpose : Field<std::uint32_t>
    {
    };

    struct CtDim : Field<std::uint32_t>
    {
    };

    struct RtDim : Field<std::uint32_t>
    {
    };

    struct KtDim : Field<std::uint32_t>
    {
    };

    struct PartialFaceA : Field<bool>
    {
    };

    struct PartialFaceB : Field<bool>
    {
    };

    using Struct = StateStruct<
        OperationUnpackMatmul,
        /* Fields */
        KernelBroadcastA,
        KernelBroadcastB,
        Transpose,
        CtDim,
        RtDim,
        KtDim,
        PartialFaceA,
        PartialFaceB>;
};

// ---------------------------------------------
// OPERATION - UNPACK TILIZE (AKA UNPACK_TILIZE)
// ---------------------------------------------

struct OperationUnpackTilize : Operation<Exu::Unpack, Hoistable::No>
{
    template <typename T>
    using Field = StateField<OperationUnpackTilize, T>;

    struct BlockCtDim : Field<std::uint32_t>
    {
    };

    struct NarrowTile : Field<bool>
    {
    };

    using Struct = StateStruct<
        OperationUnpackTilize,
        /* Fields */
        BlockCtDim,
        NarrowTile>;
};

// -------------------------
// OPERATION - FPU MATMUL
// -------------------------

struct OperationFpuMatmul : Operation<Exu::Fpu, Hoistable::Yes>
{
    template <typename T>
    using Field = StateField<OperationFpuMatmul, T>;

    struct MathFidelity : Field<std::uint32_t>
    {
    };

    struct ThrottleLevel : Field<std::uint32_t>
    {
    };

    struct CtDim : Field<std::uint32_t>
    {
    };

    struct RtDim : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<
        OperationFpuMatmul,
        /* Fields */
        MathFidelity,
        ThrottleLevel,
        CtDim,
        RtDim>;
};

// ----------------------------------------
// OPERATION - FPU ELTWISE UNARY DATACOPY
// ----------------------------------------

struct OperationFpuEltwiseUnaryDatacopy : Operation<Exu::Fpu, Hoistable::Yes>
{
    template <typename T>
    using Field = StateField<OperationFpuEltwiseUnaryDatacopy, T>;

    struct DataCopyType : Field<std::uint32_t>
    {
    };

    struct BroadcastType : Field<std::uint32_t>
    {
    };

    // Note: Blackhole specific
    struct NumFaces : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<
        OperationFpuEltwiseUnaryDatacopy,
        /* Fields */
        DataCopyType,
        BroadcastType,
        NumFaces>;
};

// ------------------
// OPERATION - PACK
// ------------------

// sstanisic fixme: rework Pack Init/Uninit interaction
struct OperationPack : Operation<Exu::Pack, Hoistable::Yes>
{
    template <typename T>
    using Field = StateField<OperationPack, T>;

    using Struct = StateStruct<OperationPack>;
};

// ---------------------------
// OPERATION - PACK UNTILIZE
// ---------------------------

struct OperationPackUntilize : Operation<Exu::Pack, Hoistable::No>
{
    template <typename T>
    using Field = StateField<OperationPackUntilize, T>;

    struct BlockCtDim : Field<std::uint32_t>
    {
    };

    struct FullCtDim : Field<std::uint32_t>
    {
    };

    struct NarrowRow : Field<bool>
    {
    };

    struct RowNumDatums : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<
        OperationPackUntilize,
        /* Fields */
        BlockCtDim,
        FullCtDim,
        NarrowRow,
        RowNumDatums>;
};

using UnpackOperations = OperationList<OperationUnpackUnary, OperationUnpackBinary, OperationUnpackMatmul, OperationUnpackTilize>;

using FpuOperations = OperationList<
    OperationFpuMatmul,
    OperationFpuEltwiseUnaryDatacopy
    // sstanisic todo: add FPU ELTWISE BINARY ADD operation state
    // sstanisic todo: add FPU ELTWISE BINARY SUB operation state
    // sstanisic todo: add FPU ELTWISE BINARY MUL operation state
    // sstanisic todo: add FPU ELTWISE BINARY ADD DEST REUSE operation state
    // sstanisic todo: add FPU ELTWISE BINARY SUB DEST REUSE operation state
    // sstanisic todo: add FPU ELTWISE BINARY MUL DEST REUSE operation state
    >;

using SfpuOperations = OperationList<>;

using PackOperations = OperationList<OperationPack, OperationPackUntilize>;

template <>
struct ExuOperations<Exu::Unpack>
{
    using type = UnpackOperations;
};

template <>
struct ExuOperations<Exu::Fpu>
{
    using type = FpuOperations;
};

template <>
struct ExuOperations<Exu::Sfpu>
{
    using type = SfpuOperations;
};

template <>
struct ExuOperations<Exu::Pack>
{
    using type = PackOperations;
};

struct State
{
    ExuState<Exu::Unpack> unpack;
    ExuState<Exu::Fpu> fpu;
    ExuState<Exu::Sfpu> sfpu;
    ExuState<Exu::Pack> pack;
};

} // namespace llk::san
