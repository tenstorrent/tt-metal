// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _SYNC_TYPES_H
#define _SYNC_TYPES_H

typedef std::atomic_flag spinlock;

struct barrier {
    uint32_t total_threads;
    uint32_t waiting_threads;
    uint32_t flag;
    spinlock lock;
};

#endif /* _SYNC_TYPES_H */
