// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

namespace dev {

class dfb_register_t {
public:
    dfb_register_t() : capacity(0), num_pages_posted(0), num_pages_acked(0) {}

    void set_capacity(uint16_t capacity) { this->capacity = capacity; }

    void inc_pages_posted(uint16_t num_pages) { this->num_pages_posted += num_pages; }

    void inc_pages_acked(uint16_t num_pages) { this->num_pages_acked += num_pages; }

    uint16_t get_space_avail() const { return this->capacity - (this->num_pages_posted - this->num_pages_acked); }

    uint16_t get_pages_avail() const { return this->num_pages_posted - this->num_pages_acked; }

    uint16_t get_capacity() const { return this->capacity; }

private:
    volatile uint16_t capacity;
    volatile uint16_t num_pages_posted;
    volatile uint16_t num_pages_acked;
};

extern dfb_register_t intra_cluster_instances[16];  // these should only be used in compute kernels
// overlay sees all 64 but each tensix only sees 16 instances, how is this exposed?
extern dfb_register_t overlay_cluster_instances[64];

}  // namespace dev

}  // namespace tt::tt_metal