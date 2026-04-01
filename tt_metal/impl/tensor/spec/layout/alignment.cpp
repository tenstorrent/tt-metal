// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>

#include <ostream>

namespace tt::tt_metal {

bool Alignment::operator==(const Alignment& other) const = default;

bool Alignment::operator==(const ttsl::SmallVector<uint32_t>& other) const { return this->value_ == other; }

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Alignment& alignment) {
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

std::string ttsl::fmt_detail::to_string(const tt::tt_metal::Alignment& alignment) {
    std::string result = "Alignment([";
    for (size_t i = 0; i < alignment.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += std::to_string(alignment[i]);
    }
    result += "])";
    return result;
}
