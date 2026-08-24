// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NullState tags whether a Metal 2.0 binding token is attached to a real
// program-scope resource (NonNull) or is a compile-time null binding (Null).
//
// Unscoped enumerators so device code can write DFBBindingToken<NonNull> and
// `if constexpr (token.is_null)` without a nested-name qualifier.
enum NullState { Null, NonNull };
