// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/targets/execution_target.hpp -- which core family a sweep point runs on.

#ifndef X280_HARNESS_TARGETS_EXECUTION_TARGET_HPP
#define X280_HARNESS_TARGETS_EXECUTION_TARGET_HPP

namespace x280::harness {

enum class ExecutionTarget { X280, Drisc, IdleEthernet };

}  // namespace x280::harness

#endif  // X280_HARNESS_TARGETS_EXECUTION_TARGET_HPP
