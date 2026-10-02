// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/no-modifiers.cpp 2>&1 | FileCheck %s --check-prefix=NO_MODIFIERS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/row-step-range.cpp 2>&1 | FileCheck %s --check-prefix=ROW_STEP_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/face-step-range.cpp 2>&1 | FileCheck %s --check-prefix=FACE_STEP_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/selection-range.cpp 2>&1 | FileCheck %s --check-prefix=SELECTION_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_pack_thread} %t/duplicate-selection.cpp 2>&1 | FileCheck %s --check-prefix=DUPLICATE_SELECTION
// clang-format on

//--- no-modifiers.cpp
#include "hal/utils/address_modifier.h"

using hal::cfg::Sec;
using hal::pack::AddressModifier;
using hal::pack::configure_address_modifiers;

void no_modifiers()
{
    configure_address_modifiers<>();
}

// NO_MODIFIERS: error: static assertion failed: at least one packer address modifier is required

//--- row-step-range.cpp
#include "hal/utils/address_modifier.h"

using hal::cfg::Sec;
using hal::pack::AddressModifier;
using hal::pack::configure_address_modifiers;

void row_step_range()
{
    configure_address_modifiers<AddressModifier {.selection = Sec::S0, .destination_row = {.step = 16}}>();
}

// ROW_STEP_RANGE: error: static assertion failed: packer address modifier selection or counter step is out of range

//--- face-step-range.cpp
#include "hal/utils/address_modifier.h"

using hal::cfg::Sec;
using hal::pack::AddressModifier;
using hal::pack::configure_address_modifiers;

void face_step_range()
{
    configure_address_modifiers<AddressModifier {.selection = Sec::S1, .source_face = {.step = 2}}>();
}

// FACE_STEP_RANGE: error: static assertion failed: packer address modifier selection or counter step is out of range

//--- selection-range.cpp
#include "hal/utils/address_modifier.h"

using hal::cfg::Sec;
using hal::pack::AddressModifier;
using hal::pack::configure_address_modifiers;

void selection_range()
{
    configure_address_modifiers<AddressModifier {.selection = Sec::S4}>();
}

// SELECTION_RANGE: error: static assertion failed: packer address modifier selection or counter step is out of range

//--- duplicate-selection.cpp
#include "hal/utils/address_modifier.h"

using hal::cfg::Sec;
using hal::pack::AddressModifier;
using hal::pack::configure_address_modifiers;

void duplicate_selection()
{
    configure_address_modifiers<AddressModifier {.selection = Sec::S2}, AddressModifier {.selection = Sec::S2, .source_row = {.step = 1}}>();
}

// DUPLICATE_SELECTION: error: static assertion failed: a packer address-modifier selection can be configured only once per grouped write
