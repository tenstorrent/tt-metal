// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

void populate_fd_kernels(const std::set<chip_id_t> &device_ids, uint32_t num_hw_cqs);

std::unique_ptr<Program> create_mmio_cq_program(Device *device);

void configure_dispatch_cores(Device *device);
