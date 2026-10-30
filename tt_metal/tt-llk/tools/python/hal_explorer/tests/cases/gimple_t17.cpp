// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void nn(unsigned n, unsigned m) {
    for (unsigned i = 0; i < n; i++)
        for (unsigned j = 0; j < m; j++)
            TTI_NOP;
}
void nn3(unsigned n, unsigned m, unsigned k) {
    for (unsigned i = 0; i < n; i++)
        for (unsigned j = 0; j < m; j++)
            for (unsigned l = 0; l < k; l++)
                TTI_NOP;
}
void nn_const() {
    for (unsigned i = 0; i < 6; i++)
        for (unsigned j = 0; j < 7; j++)
            TTI_NOP;
}
