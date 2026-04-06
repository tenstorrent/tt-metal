// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

enum class SfpuType {
    unused = 0,
    cosh,
    cbrt,
    hardtanh,
    lgamma,
    hardsigmoid,
    rpow,
    softsign,
    selu,
    hardswish,
    softshrink,
};
