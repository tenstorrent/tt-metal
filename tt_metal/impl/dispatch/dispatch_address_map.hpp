// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

/*
* This file contains addresses used for fast dispatch that are visible to both host and device compiled code
*/

// Command queue pointers
constexpr static uint32_t CQ_PREFETCH_Q_RD_PTR = L1_UNRESERVED_BASE;
constexpr static uint32_t CQ_COMPLETION_WRITE_PTR = CQ_PREFETCH_Q_RD_PTR + L1_ALIGNMENT;
constexpr static uint32_t CQ_COMPLETION_READ_PTR = CQ_COMPLETION_WRITE_PTR + L1_ALIGNMENT;
constexpr static uint32_t CQ_COMPLETION_LAST_EVENT = CQ_COMPLETION_READ_PTR + L1_ALIGNMENT;
constexpr static uint32_t CQ_COMPLETION_16B_SCRATCH = CQ_COMPLETION_LAST_EVENT + L1_ALIGNMENT;

// Dispatch message address
constexpr static std::uint32_t DISPATCH_MESSAGE_ADDR = CQ_COMPLETION_16B_SCRATCH + L1_ALIGNMENT;
constexpr static std::uint32_t DISPATCH_MESSAGE_ADDR_RESERVATION = 32;
constexpr static std::uint64_t DISPATCH_MESSAGE_REMOTE_SENDER_ADDR = DISPATCH_MESSAGE_ADDR + DISPATCH_MESSAGE_ADDR_RESERVATION;

constexpr static std::uint32_t DISPATCH_L1_UNRESERVED_BASE = (((DISPATCH_MESSAGE_REMOTE_SENDER_ADDR + DISPATCH_MESSAGE_ADDR_RESERVATION) - 1) | + (32 - 1)) + 1;

// Host addresses in hugepage for dispatch
static constexpr uint32_t HOST_CQ_ISSUE_READ_PTR = 0; // this seems to be unused
static constexpr uint32_t HOST_CQ_COMPLETION_WRITE_PTR = 32;
static constexpr uint32_t CQ_START = 64;
