// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This file is a copy of TTNN's ttnn/core/tensor/layout/alignment.cpp
// at commit 9f3856801448f589170defe41b23c8b9b43e33a2, with modifications to
// use experimental tensor types.

#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>

#include <ostream>

namespace tt::tt_metal {

bool Alignment::operator==(const Alignment& other) const = default;

bool Alignment::operator==(const ttsl::SmallVector<uint32_t>& other) const { return this->value_ == other; }

std::ostream& operator<<(std::ostream& os, const Alignment& alignment) {
    os << "Alignment([";
    for (size_t i = 0; i < alignment.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << alignment[i];
    }
    os << "])";
    return os;
}

}  // namespace tt::tt_metal
