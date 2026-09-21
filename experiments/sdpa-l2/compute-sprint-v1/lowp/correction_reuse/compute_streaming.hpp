// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#define SDPA_SPRINT_REUSE_SUM 1
#define SDPA_SPRINT_REUSE_NUM 1
#include "compensated_reuse.hpp"
#define calculate_sdpa_compensated_state calculate_sprint_compensated_reuse
#include "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#undef calculate_sdpa_compensated_state
