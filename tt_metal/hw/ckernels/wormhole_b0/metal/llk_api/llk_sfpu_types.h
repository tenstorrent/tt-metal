// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

enum class SfpuType {
    unused = 0,
    cosh,
    cbrt,
    hardsigmoid,
    selu,
    hardtanh,
    softsign,
    lgamma,
    rpow,
    hardswish,
    softshrink,
<<<<<<< HEAD
<<<<<<< HEAD
    frac,
=======
    swish,
>>>>>>> gen-swish-v2
=======
    atanh,
>>>>>>> gen-atanh-v2
};
