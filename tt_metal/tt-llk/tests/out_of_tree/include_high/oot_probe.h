// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// High-priority copy. Registered LAST, and add_include_dirs() prepends, so
// this is the one the driver must see. Same relative spelling as
// include_low/oot_probe.h — that is the point: it models a proprietary header
// shadowing an in-tree copy of the same path.

#pragma once

#define OOT_PROBE_ID 2
