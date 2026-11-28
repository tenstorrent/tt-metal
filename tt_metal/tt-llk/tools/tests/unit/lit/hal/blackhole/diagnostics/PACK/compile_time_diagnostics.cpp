// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/valid-options.cpp
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/invalid-address-modifier.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ADDRESS_MODIFIER
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/unsupported-interfaces.cpp 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED_INTERFACES
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/invalid-encode.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ENCODE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/runtime-address-modifier.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_ADDRESS_MODIFIER
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/runtime-interfaces.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_INTERFACES
// clang-format on

//--- valid-options.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

static_assert(pack::is_valid(pack::Transfer {}));
static_assert(pack::is_valid(pack::Transfer {
    .address_modifier = 3,
    .interfaces       = pack::ReadInterfaces::Row1And3,
    .datum_override   = pack::DatumOverride::Zero,
    .dest_access      = pack::DestAccess::Strided,
    .concatenation    = pack::Concatenation::Append,
    .boundary         = pack::TileBoundary::Last,
}));
static_assert(!pack::is_valid(pack::Transfer {.address_modifier = 4}));
// Only the read-interface masks the Blackhole ISA supports are valid; 0b1111 is spelled ReadInterfaces::All.
static_assert(!pack::is_valid(pack::Transfer {.interfaces = static_cast<pack::ReadInterfaces>(0b0110)}));
static_assert(!pack::is_valid(pack::Transfer {.interfaces = static_cast<pack::ReadInterfaces>(0b1111)}));

// The runtime encoder is also usable in constant expressions.
static_assert(
    pack::Packer0.operation().transfer({.address_modifier = 2, .interfaces = pack::ReadInterfaces::Row0, .boundary = pack::TileBoundary::Last}) ==
    pack::Packer0.operation()
        .transfer<pack::Transfer {.address_modifier = 2, .interfaces = pack::ReadInterfaces::Row0, .boundary = pack::TileBoundary::Last}>());
static_assert(pack::Packer0.operation().transfer<pack::Transfer {.boundary = pack::TileBoundary::Last}>() == TT_OP_PACR(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1));

//--- invalid-address-modifier.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

void issue()
{
    pack::Packer0.transfer<pack::Transfer {.address_modifier = 4}>();
}

// INVALID_ADDRESS_MODIFIER: error: static assertion failed: invalid PACR data-transfer options

//--- unsupported-interfaces.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

void issue()
{
    pack::Packer0.transfer<pack::Transfer {.interfaces = static_cast<pack::ReadInterfaces>(0b0110)}>();
}

// UNSUPPORTED_INTERFACES: error: static assertion failed: invalid PACR data-transfer options

//--- invalid-encode.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

constexpr auto operation = pack::Packer0.operation().transfer<pack::Transfer {.address_modifier = 4}>();
// INVALID_ENCODE: error: static assertion failed: invalid PACR data-transfer options

//--- runtime-address-modifier.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

// Constant evaluation of the runtime encoder rejects invalid options without a static_assert.
constexpr auto operation = pack::Packer0.operation().transfer({.address_modifier = 4});
// RUNTIME_ADDRESS_MODIFIER: in 'constexpr' expansion of 'hal::detail::require({{.*}}"PACR address modifier must be in [0, 3]"))'
// RUNTIME_ADDRESS_MODIFIER: error: '__builtin_trap()' is not a constant expression

//--- runtime-interfaces.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

constexpr auto operation = pack::Packer0.operation().transfer({.interfaces = static_cast<pack::ReadInterfaces>(0b0110)});
// RUNTIME_INTERFACES: in 'constexpr' expansion of 'hal::detail::require({{.*}}"PACR read interfaces must be a ReadInterfaces value"))'
// RUNTIME_INTERFACES: error: '__builtin_trap()' is not a constant expression
