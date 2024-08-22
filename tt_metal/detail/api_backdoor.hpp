// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/program/program.hpp"

namespace tt::tt_metal {

class MetalProgram;
struct Program;
typedef struct Program Program;
namespace detail {

MetalProgram *GetMetalProgram(std::shared_ptr<Program> program);

}  // namespace detail

}  // namespace tt::tt_metal
