// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/valid-descriptors.cpp
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/invalid-address-modifier.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ADDRESS_MODIFIER
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/invalid-configuration-context.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_CONFIGURATION_CONTEXT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/invalid-counter-context.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_COUNTER_CONTEXT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/invalid-interfaces.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_INTERFACES
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/alignment-without-padding.cpp 2>&1 | FileCheck %s --check-prefix=ALIGNMENT_WITHOUT_PADDING
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/invalid-address-slot.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_ADDRESS_SLOT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/invalid-stream-id.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_STREAM_ID
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/invalid-edge-offset.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_EDGE_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/invalid-constant-encoding.cpp 2>&1 | FileCheck %s --check-prefix=INVALID_CONSTANT_ENCODING
// clang-format on

//--- valid-descriptors.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

static_assert(pack::is_valid(pack::DataTransfer {}));
static_assert(pack::is_valid(pack::DataTransfer {
    .address_modifier      = 3,
    .configuration_context = 3,
    .counter_context       = 2,
    .interfaces            = 15,
    .padding               = pack::RowPadding::FinalOnly,
    .alignment             = pack::PaddingAlignment::To16Datums,
}));
static_assert(!pack::is_valid(pack::DataTransfer {.alignment = pack::PaddingAlignment::To16Datums}));
static_assert(!pack::is_valid(pack::DataTransfer {.context = static_cast<pack::ContextControl>(4)}));
static_assert(!pack::is_valid(pack::DataTransfer {.padding = static_cast<pack::RowPadding>(4)}));
static_assert(pack::is_valid(pack::RegisterWrite {.address_slot = 3, .stream_id = 63}));
static_assert(!pack::is_valid(pack::RegisterWrite {.stream_id = 64}));
static_assert(pack::is_valid(pack::EdgeWindow {15, 15, 15, 15}));
static_assert(!pack::is_valid(pack::EdgeWindow {.y_end = 16}));

// Descriptor encoders are usable in constant expressions.
static_assert(pack::DataTransfer {.boundary = pack::TileBoundary::Last}.get_operation() == TT_OP_PACR(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1));
static_assert(pack::EdgeWindow {1, 2, 3, 4}.get_operation() == TT_OP_SETPKEDGOF(4, 3, 2, 1));
static_assert(pack::flush_write_aligners_operation() == TT_OP_PACR(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0));
static_assert(pack::clear_exponent_histogram_operation() == TT_OP_CLREXPHIST);

//--- invalid-address-modifier.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

void issue()
{
    pack::run<pack::DataTransfer {.address_modifier = 4}>();
}

// INVALID_ADDRESS_MODIFIER: error: static assertion failed: invalid packer descriptor

//--- invalid-configuration-context.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

void issue()
{
    pack::run<pack::DataTransfer {.configuration_context = 4}>();
}

// INVALID_CONFIGURATION_CONTEXT: error: static assertion failed: invalid packer descriptor

//--- invalid-counter-context.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

void issue()
{
    pack::run<pack::DataTransfer {.counter_context = 3}>();
}

// INVALID_COUNTER_CONTEXT: error: static assertion failed: invalid packer descriptor

//--- invalid-interfaces.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

void issue()
{
    pack::run<pack::DataTransfer {.interfaces = 16}>();
}

// INVALID_INTERFACES: error: static assertion failed: invalid packer descriptor

//--- alignment-without-padding.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

void issue()
{
    pack::run<pack::DataTransfer {.alignment = pack::PaddingAlignment::To16Datums}>();
}

// ALIGNMENT_WITHOUT_PADDING: error: static assertion failed: invalid packer descriptor

//--- invalid-address-slot.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

void issue()
{
    pack::run<pack::RegisterWrite {.address_slot = 4}>();
}

// INVALID_ADDRESS_SLOT: error: static assertion failed: invalid packer descriptor

//--- invalid-stream-id.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

void issue()
{
    pack::run<pack::RegisterWrite {.stream_id = 64}>();
}

// INVALID_STREAM_ID: error: static assertion failed: invalid packer descriptor

//--- invalid-edge-offset.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

void issue()
{
    pack::run<pack::EdgeWindow {.x_start = 16}>();
}

// INVALID_EDGE_OFFSET: error: static assertion failed: invalid packer descriptor

//--- invalid-constant-encoding.cpp
#include "hal/pack.h"

namespace pack = hal::pack;

// Constant evaluation of an invalid descriptor is rejected without static_assert.
constexpr auto operation = pack::EdgeWindow {.x_end = 16}.get_operation();
// INVALID_CONSTANT_ENCODING: error: '__builtin_trap()' is not a constant expression
