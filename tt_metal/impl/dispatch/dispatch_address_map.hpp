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
// Used to notify host of how far device has gotten, doesn't need L1 alignment because it's only written locally by
// prefetch kernel.
constexpr static uint32_t CQ_PREFETCH_Q_PCIE_RD_PTR = CQ_PREFETCH_Q_RD_PTR + sizeof(uint32_t);
constexpr static uint32_t CQ_COMPLETION_WRITE_PTR = CQ_PREFETCH_Q_RD_PTR + L1_ALIGNMENT;
constexpr static uint32_t CQ_COMPLETION_READ_PTR = CQ_COMPLETION_WRITE_PTR + L1_ALIGNMENT;
// Max of 2 CQs. CQ0_COMPLETION_LAST_EVENT and CQ1_COMPLETION_LAST_EVENT track the last completed event in the respective CQs
constexpr static uint32_t CQ0_COMPLETION_LAST_EVENT = CQ_COMPLETION_READ_PTR + L1_ALIGNMENT;
constexpr static uint32_t CQ1_COMPLETION_LAST_EVENT = CQ0_COMPLETION_LAST_EVENT + L1_ALIGNMENT;

// Dispatch message address
constexpr static std::uint32_t DISPATCH_MESSAGE_ADDR = CQ1_COMPLETION_LAST_EVENT + L1_ALIGNMENT;
constexpr static std::uint32_t DISPATCH_MESSAGE_ADDR_RESERVATION = 32;
constexpr static std::uint64_t DISPATCH_MESSAGE_REMOTE_SENDER_ADDR = DISPATCH_MESSAGE_ADDR + DISPATCH_MESSAGE_ADDR_RESERVATION;

constexpr static std::uint32_t DISPATCH_L1_UNRESERVED_BASE = (((DISPATCH_MESSAGE_REMOTE_SENDER_ADDR + DISPATCH_MESSAGE_ADDR_RESERVATION) - 1) | (PCIE_ALIGNMENT - 1)) + 1;

// Host addresses in hugepage for dispatch
static constexpr uint32_t HOST_CQ_ISSUE_READ_PTR = 0; // Used by host
static constexpr uint32_t HOST_CQ_ISSUE_WRITE_PTR = HOST_CQ_ISSUE_READ_PTR + PCIE_ALIGNMENT;
static constexpr uint32_t HOST_CQ_COMPLETION_WRITE_PTR = HOST_CQ_ISSUE_WRITE_PTR + PCIE_ALIGNMENT;
static constexpr uint32_t HOST_CQ_COMPLETION_READ_PTR = HOST_CQ_COMPLETION_WRITE_PTR + PCIE_ALIGNMENT;
static constexpr uint32_t CQ_START = HOST_CQ_COMPLETION_READ_PTR + PCIE_ALIGNMENT;
