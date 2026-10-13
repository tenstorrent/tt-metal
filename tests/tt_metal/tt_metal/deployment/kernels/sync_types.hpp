// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef SYNC_TYPES_H
#define SYNC_TYPES_H

using spinlock = std::atomic_flag;

struct barrier {
    uint32_t total_threads;
    uint32_t waiting_threads;
    uint32_t flag;
    spinlock lock;
};

#endif /* SYNC_TYPES_H */
