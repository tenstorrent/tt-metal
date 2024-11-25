// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

std::unique_ptr<Program> create_mmio_cq_program(Device *device);

void configure_dispatch_cores(Device *device);
