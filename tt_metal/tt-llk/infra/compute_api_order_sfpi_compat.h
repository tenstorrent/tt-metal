// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SFPI's analysis-tool fallback (tensix_builtins.h) defines __xtt_vector,
// __xtt_vector2 and __xtt_vector4, while tensix_builtins.def also declares
// __xtt_vector8.  Provide the missing type before any SFPI header is parsed so a
// clang-based analysis TU does not stop at the first declaration.
#pragma once
using __xtt_vector8 = unsigned __attribute__((vector_size(sizeof(unsigned) * 32 * 8)));
