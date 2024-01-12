# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np


def triu(x):
    # create an upper triangular matrix
    Sa, Sb, Sy, Sz = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    y = np.triu(np.ones((Sa, Sb, Sy, Sz)))
    return y * x


def tril(x):
    # create an upper triangular matrix
    Sa, Sb, Sy, Sz = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    y = np.tril(np.ones((Sa, Sb, Sy, Sz)))
    return y * x
