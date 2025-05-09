// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed_context.hpp>
#include <gtest/gtest.h>
#include "common/multihost_asserts.hpp"
#include <fmt/format.h>
int main(int argc, char** argv) { return multihost::common::multihost_main(argc, argv); }
