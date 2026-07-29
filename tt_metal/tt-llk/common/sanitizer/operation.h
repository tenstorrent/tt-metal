#pragma once

#include "sanitizer/types.h"

// UNPACK EXU (TRISC0)
constexpr Operation Operation::None                         = Operation::make(Exu::Unpack, Thread::Trisc0, 0);
constexpr Operation Operation::UnpackA                      = Operation::make(Exu::Unpack, Thread::Trisc0, 1);
constexpr Operation Operation::UnpackABMatmul               = Operation::make(Exu::Unpack, Thread::Trisc0, 2);
constexpr Operation Operation::UnpackUntilize               = Operation::make(Exu::Unpack, Thread::Trisc0, 3);

// FPU EXU (TRISC1)
constexpr Operation Operation::FpuMatmul                    = Operation::make(Exu::Fpu,    Thread::Trisc1, 1);
constexpr Operation Operation::FpuEltwiseUnaryDatacopy      = Operation::make(Exu::Fpu,    Thread::Trisc1, 2);
constexpr Operation Operation::FpuEltwiseBinaryAdd          = Operation::make(Exu::Fpu,    Thread::Trisc1, 3);
constexpr Operation Operation::FpuEltwiseBinarySub          = Operation::make(Exu::Fpu,    Thread::Trisc1, 4);
constexpr Operation Operation::FpuEltwiseBinaryMul          = Operation::make(Exu::Fpu,    Thread::Trisc1, 5);
constexpr Operation Operation::FpuEltwiseBinaryAddDestReuse = Operation::make(Exu::Fpu,    Thread::Trisc1, 6);
constexpr Operation Operation::FpuEltwiseBinarySubDestReuse = Operation::make(Exu::Fpu,    Thread::Trisc1, 7);
constexpr Operation Operation::FpuEltwiseBinaryMulDestReuse = Operation::make(Exu::Fpu,    Thread::Trisc1, 8);

// PACK EXU (TRISC2)
constexpr Operation Operation::Pack                         = Operation::make(Exu::Pack,   Thread::Trisc2, 1);
constexpr Operation Operation::PackUntilize                 = Operation::make(Exu::Pack,   Thread::Trisc2, 2);