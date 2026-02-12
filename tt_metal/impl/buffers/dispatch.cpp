// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/span.hpp>
#include <device.hpp>
#include <tt-metalium/allocator.hpp>
#include <algorithm>
#include <optional>
#include <stack>
#include <type_traits>
#include <utility>

#include <tt_stl/assert.hpp>
#include "buffer_types.hpp"
#include "dispatch.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/kernels/cq_commands.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "hal_types.hpp"
#include <tt-logger/tt-logger.hpp>
#include "math.hpp"
#include <tt_stl/strong_type.hpp>
#include "sub_device_types.hpp"
#include "tt_align.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/event/dispatch.hpp"
#include "tt_metal/impl/device/dispatch.hpp"
#include <tt-metalium/graph_tracking.hpp>
#include <tracy/Tracy.hpp>
#include <tt_stl/overloaded.hpp>
#include "tt_metal/api/tt-metalium/experimental/pinned_memory.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

namespace tt::tt_metal::buffer_dispatch {

// ====== Utility Functions for Writes ======

// Dispatch constants required for writing buffer data
struct BufferDispatchConstants {
    uint32_t issue_queue_cmd_limit = 0;
    uint32_t max_prefetch_cmd_size = 0;
};

// Dispatch parameters computed during runtime. These are used
// to assemble dispatch commands and compute src + dst offsets
// required to write buffer data.
struct BufferWriteDispatchParams {
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    uint32_t address = 0;
    uint32_t page_size_to_write = 0;
    uint32_t data_size_to_copy = 0;
    uint32_t total_pages_to_write = 0;
    uint32_t total_pages_written = 0;
    uint32_t pages_per_txn = 0;
    bool issue_wait = false;
    IDevice* device = nullptr;
    uint32_t cq_id = 0;
    uint32_t pinned_src_noc_xy = 0;
    uint64_t pinned_src_addr = 0;
    bool use_pinned_transfer = false;
    bool remote_chip = false;

    BufferWriteDispatchParams() = default;
    BufferWriteDispatchParams(
        uint32_t src_noc_xy, uint64_t src_addr, bool src_pinned = false, bool remote_chip = false) :
        pinned_src_noc_xy{src_noc_xy},
        pinned_src_addr{src_addr},
        use_pinned_transfer{src_pinned},
        remote_chip{remote_chip} {}

    void calculate_issue_wait() {
        this->issue_wait = this->total_pages_written == 0;  // only stall for the first write of the buffer
    }
};

// Parameters specific to interleaved buffers
class InterleavedBufferWriteDispatchParams : public BufferWriteDispatchParams {
public:
    uint32_t dst_page_index = 0;
    // Number of bytes to copy on the CPU at the start of the write to align the write to the host memory alignment.
    size_t alignment_prefix_bytes = 0;

    InterleavedBufferWriteDispatchParams(
        const Buffer& buffer,
        uint32_t dst_page_index,
        uint32_t total_pages_to_write,
        uint32_t cq_id,
        tt::stl::Span<const uint32_t> expected_num_workers_completed,
        uint32_t src_noc_xy,
        uint64_t src_addr,
        bool src_pinned,
        bool remote_chip) :
        BufferWriteDispatchParams(src_noc_xy, src_addr, src_pinned, remote_chip), dst_page_index(dst_page_index) {
        this->num_banks = buffer.device()->allocator()->get_num_banks(buffer.buffer_type());
        this->address = buffer.address();

        this->page_size_to_write = buffer.aligned_page_size();
        this->data_size_to_copy = buffer.page_size();
        this->total_pages_to_write = total_pages_to_write;
        this->device = buffer.device();
        this->cq_id = cq_id;
        this->expected_num_workers_completed = expected_num_workers_completed;
        if (src_pinned) {
            const uint64_t relay_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
            const uint64_t alignment_offset = src_addr % relay_alignment;
            if (alignment_offset != 0) {
                this->alignment_prefix_bytes = relay_alignment - alignment_offset;
            }
        }
    }

    virtual ~InterleavedBufferWriteDispatchParams() = default;

    InterleavedBufferWriteDispatchParams(const InterleavedBufferWriteDispatchParams& other) = default;
    InterleavedBufferWriteDispatchParams& operator=(const InterleavedBufferWriteDispatchParams& other) = default;
    InterleavedBufferWriteDispatchParams(InterleavedBufferWriteDispatchParams&& other) = default;
    InterleavedBufferWriteDispatchParams& operator=(InterleavedBufferWriteDispatchParams&& other) = default;

    virtual void calculate_num_pages_for_write_transaction(uint32_t num_pages_available_in_cq) {
        this->pages_per_txn = std::min(this->total_pages_to_write, num_pages_available_in_cq);
    }

    virtual bool is_page_offset_out_of_bounds() const {
        return this->dst_page_index > CQ_DISPATCH_CMD_PAGED_WRITE_MAX_PAGE_INDEX;
    }

    // Page offset in CQ_DISPATCH_CMD_WRITE_PAGED is uint16_t
    // To handle larger page offsets move bank base address up and update page offset to be relative to the new
    // bank address
    virtual void update_params_to_be_within_bounds() {
        const uint32_t num_pages_written_per_bank = this->dst_page_index / this->num_banks;
        this->address += num_pages_written_per_bank * this->page_size_to_write;
        this->dst_page_index %= this->num_banks;
    }

    virtual void update_params_after_write_transaction() {
        this->total_pages_to_write -= this->pages_per_txn;
        this->total_pages_written += this->pages_per_txn;
        this->dst_page_index += this->pages_per_txn;
    }

    virtual bool write_large_pages() const { return false; }

    virtual uint32_t num_full_pages_written() const { return this->total_pages_written; }

    virtual uint32_t num_partial_pages_written_for_current_transaction_full_pages() const { return 1; }

    virtual uint32_t partial_page_size() const { return this->page_size_to_write; }

protected:
    uint32_t num_banks = 0;
};

class InterleavedBufferWriteLargePageDispatchParams : public InterleavedBufferWriteDispatchParams {
public:
    InterleavedBufferWriteLargePageDispatchParams(
        const Buffer& buffer,
        uint32_t dst_page_index,
        const PartialPageSpec& partial_page_spec,
        uint32_t total_pages_to_write,  // number of partial pages
        uint32_t num_full_pages,
        uint32_t cq_id,
        tt::stl::Span<const uint32_t> expected_num_workers_completed,
        uint32_t src_noc_xy,
        uint64_t src_addr,
        bool src_pinned,
        bool remote_chip) :
        InterleavedBufferWriteDispatchParams(
            buffer,
            dst_page_index,
            total_pages_to_write,
            cq_id,
            expected_num_workers_completed,
            src_noc_xy,
            src_addr,
            src_pinned,
            remote_chip),
        buffer(buffer),
        curr_full_pages_start_address(buffer.address()),
        size_of_partial_page(partial_page_spec.partial_page_size),
        num_partial_pages_in_single_full_page(partial_page_spec.num_partial_pages_per_full_page),
        full_pages_to_write(num_full_pages) {
        if (not use_pinned_transfer) {
            this->page_size_to_write = partial_page_spec.partial_page_size;
            this->data_size_to_copy = partial_page_spec.partial_page_size;

            this->end_bank_indices.push(this->num_banks);
            for (uint32_t i = 0; i < this->num_banks; i++) {
                this->curr_full_pages_curr_addresses.push_back(this->curr_full_pages_start_address);
            }
        }
    }

    void calculate_num_pages_for_write_transaction(uint32_t num_pages_available_in_cq) override {
        TT_ASSERT(this->end_bank_indices.top() > this->dst_page_index);
        this->pages_per_txn = std::min(
            {this->full_pages_to_write,
             this->end_bank_indices.top() - this->dst_page_index,
             num_pages_available_in_cq});
        if (this->dst_page_index + num_pages_available_in_cq < this->end_bank_indices.top()) {
            this->end_bank_indices.push(this->dst_page_index + num_pages_available_in_cq);
        }
    }

    bool is_page_offset_out_of_bounds() const override { return this->dst_page_index >= this->num_banks; }

    void update_params_to_be_within_bounds() override {
        const uint32_t num_pages_written_per_bank = this->dst_page_index / this->num_banks;
        this->address += num_pages_written_per_bank * this->buffer.aligned_page_size();
        this->curr_full_pages_start_address = this->address;
        this->dst_page_index %= this->num_banks;
    }

    uint32_t num_partial_pages_written_for_current_transaction_full_pages() const override {
        if (this->address - this->curr_full_pages_start_address == this->buffer.aligned_page_size()) {
            return this->num_partial_pages_in_single_full_page;
        }
        return (this->address - this->curr_full_pages_start_address) / this->size_of_partial_page;
    }

    void update_params_after_write_transaction() override {
        this->total_pages_to_write -= this->pages_per_txn;
        this->total_pages_written += this->pages_per_txn;
        this->address += this->page_size_to_write;
        for (uint32_t i = this->dst_page_index; i < this->dst_page_index + this->pages_per_txn; i++) {
            this->curr_full_pages_curr_addresses[i] = this->address;
        }
        if (this->were_full_pages_written_in_last_write_transaction()) {
            this->full_pages_to_write -= this->pages_per_txn;
            this->full_pages_written += this->pages_per_txn;
            if (!this->will_next_full_page_be_round_robined()) {
                TT_ASSERT(this->dst_page_index + this->pages_per_txn < this->num_banks);
                this->address = this->curr_full_pages_curr_addresses[this->dst_page_index + this->pages_per_txn];
            } else {
                this->curr_full_pages_start_address = this->address;
                for (uint32_t i = 0; i < this->num_banks; i++) {
                    this->curr_full_pages_curr_addresses[i] = this->curr_full_pages_start_address;
                }
            }

            this->dst_page_index += this->pages_per_txn;
            this->dst_page_index %= this->num_banks;
            this->page_size_to_write = this->size_of_partial_page;
            this->data_size_to_copy = this->size_of_partial_page;
            TT_ASSERT(!this->end_bank_indices.empty());
            if (this->end_bank_indices.top() != this->num_banks) {
                this->end_bank_indices.pop();
            }
        }
        if (this->will_full_pages_be_written_in_next_write_transaction()) {
            this->page_size_to_write =
                this->buffer.aligned_page_size() - (this->address - this->curr_full_pages_start_address);
            this->data_size_to_copy = this->buffer.page_size() - (this->address - this->curr_full_pages_start_address);
        }
    }

    bool write_large_pages() const override { return true; }

    uint32_t num_full_pages_written() const override { return this->full_pages_written; }

    uint32_t partial_page_size() const override { return this->size_of_partial_page; }

private:
    const Buffer& buffer;
    uint32_t curr_full_pages_start_address = 0;
    uint32_t size_of_partial_page = 0;
    uint32_t num_partial_pages_in_single_full_page = 0;
    uint32_t full_pages_written = 0;
    uint32_t full_pages_to_write = 0;
    std::stack<uint32_t> end_bank_indices;
    std::vector<uint32_t> curr_full_pages_curr_addresses;

    bool were_full_pages_written_in_last_write_transaction() const {
        const int32_t page_size = this->address - this->curr_full_pages_start_address;
        return page_size == this->buffer.aligned_page_size();
    }

    bool will_full_pages_be_written_in_next_write_transaction() const {
        const int32_t page_size = this->address + this->page_size_to_write - this->curr_full_pages_start_address;
        return page_size == (this->num_partial_pages_in_single_full_page * this->page_size_to_write);
    }

    bool will_next_full_page_be_round_robined() const {
        const uint32_t dst_page_index_next_txn = this->dst_page_index + this->pages_per_txn;
        return dst_page_index_next_txn != (dst_page_index_next_txn % this->num_banks);
    }
};

// Parameters specific to sharded buffers
class ShardedBufferWriteDispatchParams : public BufferWriteDispatchParams {
public:
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping = nullptr;
    BufferCorePageMapping::Iterator core_page_mapping_it;
    CoreCoord core;
    uint32_t core_num_pages_remaining_to_write = 0;

    ShardedBufferWriteDispatchParams(
        Buffer* buffer,
        uint32_t total_pages_to_write,
        uint32_t cq_id,
        tt::stl::Span<const uint32_t> expected_num_workers_completed,
        tt::stl::Span<const SubDeviceId> sub_device_ids,
        uint32_t pinned_noc_xy,
        uint64_t pinned_addr,
        bool is_pinned,
        bool remote_chip) :
        BufferWriteDispatchParams(pinned_noc_xy, pinned_addr, is_pinned, remote_chip),
        buffer_page_mapping(buffer->get_buffer_page_mapping()),
        buffer(buffer),
        are_pages_large(
            this->use_pinned_transfer ? false
                                      : are_pages_larger_than_max_prefetch_cmd_size(*buffer, sub_device_ids.size())) {
        this->cq_id = cq_id;
        this->device = buffer->device();
        this->expected_num_workers_completed = expected_num_workers_completed;

        this->total_pages_written = 0;

        if (this->are_pages_large) {
            const PartialPageSpec partial_page_spec = calculate_partial_page_spec(*buffer);
            this->size_of_partial_page = partial_page_spec.partial_page_size;
            this->page_size_to_write = partial_page_spec.partial_page_size;
            this->data_size_to_copy = partial_page_spec.partial_page_size;
            this->total_pages_to_write = total_pages_to_write * partial_page_spec.num_partial_pages_per_full_page;
            this->num_partial_pages_in_single_full_page = partial_page_spec.num_partial_pages_per_full_page;
        } else {
            this->total_pages_to_write = total_pages_to_write;
            this->page_size_to_write = buffer->aligned_page_size();
            this->data_size_to_copy = buffer->page_size();
            this->num_partial_pages_in_single_full_page = 1;
            this->size_of_partial_page = buffer->aligned_page_size();
        }
    }

    ~ShardedBufferWriteDispatchParams() = default;

    void reset_params_for_core(const CoreCoord& core, const BufferCorePageMapping& core_page_mapping) {
        this->core = core;
 //       fmt::println(stderr, "Host mapping count: {}", core_page_mapping.host_ranges.size());
        this->core_page_mapping_it = core_page_mapping.begin();
        this->address =
            this->buffer->address() + core_page_mapping.device_start_page * this->buffer->aligned_page_size();
        if (this->buffer->is_dram()) {
            this->address += this->buffer->device()->allocator()->get_bank_offset(
                BufferType::DRAM, this->buffer->device()->dram_channel_from_logical_core(core));
        }
        if (this->are_pages_large) {
            this->core_num_pages_remaining_to_write =
                core_page_mapping.num_pages * this->num_partial_pages_in_single_full_page;
        } else {
            this->core_num_pages_remaining_to_write = core_page_mapping.num_pages;
        }
    }

    bool write_large_pages() const { return this->are_pages_large; }

    uint32_t partial_page_size() const { return this->size_of_partial_page; }

    uint32_t num_partial_pages_written_for_current_transaction_full_page() const {
        return this->num_partial_pages_written_for_curr_full_page;
    }

    void calculate_params_for_write_transaction(uint32_t num_pages_available_in_cq) {
        if (this->are_pages_large) {
            const int32_t num_partial_pages_remaining_in_curr_full_page =
                this->num_partial_pages_in_single_full_page - this->num_partial_pages_written_for_curr_full_page;
            TT_ASSERT(num_partial_pages_remaining_in_curr_full_page > 0);
            const uint32_t max_num_partial_pages_to_write_in_curr_txn =
                (num_partial_pages_remaining_in_curr_full_page == 1)
                    ? 1
                    : num_partial_pages_remaining_in_curr_full_page - 1;
            this->pages_per_txn = std::min(
                {this->core_num_pages_remaining_to_write,
                 max_num_partial_pages_to_write_in_curr_txn,
                 num_pages_available_in_cq});

            if (num_partial_pages_remaining_in_curr_full_page == 1) {
                this->page_size_to_write =
                    this->buffer->aligned_page_size() -
                    (this->num_partial_pages_written_for_curr_full_page * this->size_of_partial_page);
                this->data_size_to_copy =
                    this->buffer->page_size() -
                    (this->num_partial_pages_written_for_curr_full_page * this->size_of_partial_page);
            }
        } else {
            this->pages_per_txn = std::min(this->core_num_pages_remaining_to_write, num_pages_available_in_cq);
        }
    }

    void update_params_after_write_transaction() {
        this->total_pages_to_write -= this->pages_per_txn;
        this->total_pages_written += this->pages_per_txn;
        this->address += this->pages_per_txn * this->page_size_to_write;
        this->core_num_pages_remaining_to_write -= this->pages_per_txn;
        if (this->are_pages_large) {
            this->num_partial_pages_written_for_curr_full_page += this->pages_per_txn;
            if (this->num_partial_pages_written_for_curr_full_page == this->num_partial_pages_in_single_full_page) {
                this->page_size_to_write = this->size_of_partial_page;
                this->data_size_to_copy = this->size_of_partial_page;

                this->num_partial_pages_written_for_curr_full_page = 0;
                ++this->core_page_mapping_it;
            }
        } else {
            this->num_partial_pages_written_for_curr_full_page = 1;
        }
    }

private:
    const Buffer* buffer = nullptr;
    bool are_pages_large = false;
    uint32_t size_of_partial_page = 0;
    uint32_t num_partial_pages_written_for_curr_full_page = 0;
    uint32_t num_partial_pages_in_single_full_page = 0;
};

int32_t calculate_num_pages_available_in_cq(
    const BufferWriteDispatchParams& dispatch_params,
    const BufferDispatchConstants& dispatch_constants,
    uint32_t byte_offset_in_cq) {
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t space_availableB = std::min(
        dispatch_constants.issue_queue_cmd_limit - sysmem_manager.get_issue_queue_write_ptr(dispatch_params.cq_id),
        dispatch_constants.max_prefetch_cmd_size);
    int32_t num_pages_available =
        (int32_t(space_availableB) - int32_t(byte_offset_in_cq)) / int32_t(dispatch_params.page_size_to_write);
    return num_pages_available;
}

bool are_pages_larger_than_max_prefetch_cmd_size(const Buffer& buffer, uint32_t num_subdevices) {
    const CoreType dispatch_core_type = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
    const uint32_t max_data_size = calculate_max_prefetch_data_size_bytes(dispatch_core_type, num_subdevices);
    return buffer.aligned_page_size() > max_data_size;
}

uint32_t calculate_partial_page_size(const Buffer& buffer) {
    const HalMemType buffer_mem_type = buffer.memory_type();
    const uint32_t partial_page_size = tt::align(
        DispatchSettings::BASE_PARTIAL_PAGE_SIZE_DISPATCH,
        MetalContext::instance().hal().get_common_alignment_with_pcie(buffer_mem_type));
    return partial_page_size;
}

PartialPageSpec calculate_partial_page_spec(const Buffer& buffer) {
    PartialPageSpec partial_page_spec;
    partial_page_spec.partial_page_size = calculate_partial_page_size(buffer);
    partial_page_spec.num_partial_pages_per_full_page =
        tt::div_up(buffer.aligned_page_size(), partial_page_spec.partial_page_size);
    return partial_page_spec;
}

// Generate dispatch constants
BufferDispatchConstants generate_buffer_dispatch_constants(
    const SystemMemoryManager& sysmem_manager, CoreType /*dispatch_core_type*/, uint32_t cq_id) {
    BufferDispatchConstants buf_dispatch_constants;

    buf_dispatch_constants.issue_queue_cmd_limit = sysmem_manager.get_issue_queue_limit(cq_id);
    buf_dispatch_constants.max_prefetch_cmd_size =
        MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();

    return buf_dispatch_constants;
}

void update_offset_on_issue_wait_cmd(uint32_t& byte_offset, bool issue_wait, uint32_t num_sub_devices) {
    if (issue_wait) {
        // commands prefixed with CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WAIT
        byte_offset += (MetalContext::instance().hal().get_alignment(HalMemType::HOST) * num_sub_devices);
    }
}

using InterleavedBufferWriteDispatchParamsVariant =
    std::variant<std::monostate, InterleavedBufferWriteDispatchParams, InterleavedBufferWriteLargePageDispatchParams>;

InterleavedBufferWriteDispatchParamsVariant initialize_interleaved_buf_dispatch_params(
    const Buffer& buffer,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    const BufferRegion& region,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    uint32_t pinned_src_noc_xy,
    uint64_t pinned_src_addr,
    bool use_pinned_transfer,
    bool remote_chip) {
    InterleavedBufferWriteDispatchParamsVariant dispatch_params;

    uint32_t total_pages_to_write = region.size / buffer.page_size();
    const uint32_t dst_page_index = region.offset / buffer.page_size();

    if (!use_pinned_transfer && are_pages_larger_than_max_prefetch_cmd_size(buffer, sub_device_ids.size())) {
        const PartialPageSpec partial_page_spec = calculate_partial_page_spec(buffer);
        const uint32_t num_full_pages = total_pages_to_write;
        total_pages_to_write = num_full_pages * partial_page_spec.num_partial_pages_per_full_page;
        dispatch_params.emplace<InterleavedBufferWriteLargePageDispatchParams>(
            buffer,
            dst_page_index,
            partial_page_spec,
            total_pages_to_write,
            num_full_pages,
            cq_id,
            expected_num_workers_completed,
            pinned_src_noc_xy,
            pinned_src_addr,
            use_pinned_transfer,
            remote_chip);
    } else {
        dispatch_params.emplace<InterleavedBufferWriteDispatchParams>(
            buffer,
            dst_page_index,
            total_pages_to_write,
            cq_id,
            expected_num_workers_completed,
            pinned_src_noc_xy,
            pinned_src_addr,
            use_pinned_transfer,
            remote_chip);
    }
    return dispatch_params;
}

// Populate/Assemble dispatch commands for writing buffer data
void populate_interleaved_buffer_write_dispatch_cmds(
    const void* src,
    HugepageDeviceCommand& command_sequence,
    Buffer& buffer,
    InterleavedBufferWriteDispatchParams& dispatch_params) {
    const uint8_t is_dram = uint8_t(buffer.is_dram());
    TT_ASSERT(
        dispatch_params.dst_page_index <= CQ_DISPATCH_CMD_PAGED_WRITE_MAX_PAGE_INDEX,
        "Page offset needs to fit within range of uint16_t, bank_base_address was computed incorrectly!");
    const uint16_t start_page = uint16_t(dispatch_params.dst_page_index & CQ_DISPATCH_CMD_PAGED_WRITE_MAX_PAGE_INDEX);

    bool use_pinned_transfer = dispatch_params.use_pinned_transfer;

    // If we're not using pinned transfer the data will be inline with the dispatch write in a single prefetch command
    // so we need to flush the prefetch. With pinned memory the data will come in a separate command and we shouldn't
    // flush between them.
    const bool flush_prefetch = !use_pinned_transfer;

    if (dispatch_params.alignment_prefix_bytes > 0) {
        // Pass the unaligned prefix bytes inline to reach alignment
        TT_ASSERT(dispatch_params.alignment_prefix_bytes < buffer.page_size(), "Alignment prefix exceeds page size");

        command_sequence.add_dispatch_write_paged_with_custom_inline_size(
            flush_prefetch,
            is_dram,
            start_page,
            dispatch_params.address,
            dispatch_params.page_size_to_write,
            dispatch_params.total_pages_to_write,
            dispatch_params.alignment_prefix_bytes,
            static_cast<const char*>(src));
    } else {
        command_sequence.add_dispatch_write_paged(
            flush_prefetch,
            is_dram,
            start_page,
            dispatch_params.address,
            dispatch_params.page_size_to_write,
            use_pinned_transfer ? dispatch_params.total_pages_to_write : dispatch_params.pages_per_txn);
    }

    if (not use_pinned_transfer) {
        const uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;
        // TODO: Consolidate
        if (dispatch_params.write_large_pages()) {
            const uint32_t num_full_pages_written = dispatch_params.num_full_pages_written();
            const uint32_t num_partial_pages_written_per_curr_full_pages =
                dispatch_params.num_partial_pages_written_for_current_transaction_full_pages();
            uint32_t num_partial_pages_written_curr_txn = 0;
            for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
                 sysmem_address_offset += dispatch_params.page_size_to_write) {
                const uint64_t src_address_offset =
                    ((uint64_t)num_full_pages_written * buffer.page_size()) +
                    (num_partial_pages_written_per_curr_full_pages * dispatch_params.partial_page_size()) +
                    (num_partial_pages_written_curr_txn * buffer.page_size());
                command_sequence.add_data(
                    static_cast<const char*>(src) + src_address_offset,
                    dispatch_params.data_size_to_copy,
                    dispatch_params.page_size_to_write);
                num_partial_pages_written_curr_txn += 1;
            }
        } else {
            DeviceAddr src_address_offset = DeviceAddr(dispatch_params.total_pages_written) * buffer.page_size();
            if (buffer.page_size() % buffer.alignment() != 0 and buffer.page_size() != buffer.size()) {
                // If page size is not aligned, we cannot do a contiguous write
                for (uint32_t sysmem_address_offset = 0; sysmem_address_offset < data_size_bytes;
                     sysmem_address_offset += dispatch_params.page_size_to_write) {
                    command_sequence.add_data(
                        static_cast<const char*>(src) + src_address_offset,
                        dispatch_params.data_size_to_copy,
                        dispatch_params.page_size_to_write);
                    src_address_offset += dispatch_params.data_size_to_copy;
                }
            } else {
                command_sequence.add_data(
                    static_cast<const char*>(src) + src_address_offset,
                    dispatch_params.data_size_to_copy * dispatch_params.pages_per_txn,
                    data_size_bytes);
            }
        }
        command_sequence.align_write_offset();
    }
}

bool logged_writes = false;
void populate_sharded_buffer_write_dispatch_cmds(
    const void* src,
    HugepageDeviceCommand& command_sequence,
    Buffer& buffer,
    ShardedBufferWriteDispatchParams& dispatch_params) {
    const uint32_t data_size_bytes = dispatch_params.pages_per_txn * dispatch_params.page_size_to_write;
    const CoreCoord virtual_core =
        buffer.device()->virtual_core_from_logical_core(dispatch_params.core, buffer.core_type());
    command_sequence.add_dispatch_write_linear(
        0,
        buffer.device()->get_noc_unicast_encoding(k_dispatch_downstream_noc, virtual_core),
        dispatch_params.address,
        data_size_bytes);

        #if 0
    if (!logged_writes) {
        fmt::println(stderr, "Host mappign count: {}", dispatch_params.core_page_mapping_it.
    }
    #endif

    uint8_t* dst = command_sequence.reserve_space<uint8_t*, true>(data_size_bytes);
    // TODO: Expose getter for cmd_write_offsetB?
    ptrdiff_t dst_offset = reinterpret_cast<ptrdiff_t>(dst - (uint8_t*)command_sequence.data());
    TT_ASSERT(dst_offset >= 0, "Offset into command sequence is negative");
    if (dispatch_params.write_large_pages()) {
        const auto cur_host_page = *dispatch_params.core_page_mapping_it;
        if (!cur_host_page) {
            command_sequence.align_write_offset();
            return;
        }
        for (uint32_t i = 0; i < dispatch_params.pages_per_txn; ++i) {
            const uint64_t src_offset =
                (*cur_host_page * (uint64_t)buffer.page_size()) +
                ((dispatch_params.num_partial_pages_written_for_current_transaction_full_page() + i) *
                 dispatch_params.partial_page_size());
            command_sequence.update_cmd_sequence(
                dst_offset, static_cast<const char*>(src) + src_offset, dispatch_params.data_size_to_copy);
            dst_offset += dispatch_params.page_size_to_write;
        }
    } else if (buffer.page_size() == dispatch_params.page_size_to_write) {
        uint32_t start_device_page_offset = dispatch_params.core_page_mapping_it.device_page_offset();
        uint32_t end_device_page_offset = start_device_page_offset + dispatch_params.pages_per_txn;
        while (true) {
            auto range = dispatch_params.core_page_mapping_it.next_range(end_device_page_offset);
            if (range.num_pages == 0) {
                command_sequence.align_write_offset();
                return;
            }
            uint64_t src_offset = (uint64_t)(range.host_page_start) * dispatch_params.page_size_to_write;
            auto cmd_region_offset =
                dispatch_params.page_size_to_write * (range.device_page_offset - start_device_page_offset);
                #if 0
            if (!logged_writes) {
                //fmt::println(stderr, "Writing pages to offset {} from offset {} size {} \n", dst_offset + cmd_region_offset, src_offset, range.num_pages * dispatch_params.page_size_to_write);
            }
            #endif
            command_sequence.update_cmd_sequence(
                dst_offset + cmd_region_offset,
                static_cast<const char*>(src) + src_offset,
                range.num_pages * dispatch_params.page_size_to_write);
        }
    } else {
        for (size_t i = 0; i < dispatch_params.pages_per_txn; i++) {
            auto cur_host_page = *dispatch_params.core_page_mapping_it;
            ++dispatch_params.core_page_mapping_it;
            if (!cur_host_page) {
                dst_offset += dispatch_params.page_size_to_write;
                continue;
            }
            const uint64_t src_offset = *cur_host_page * (uint64_t)buffer.page_size();
            command_sequence.update_cmd_sequence(
                dst_offset, static_cast<const char*>(src) + src_offset, buffer.page_size());
            dst_offset += dispatch_params.page_size_to_write;
        }
    }
    command_sequence.align_write_offset();
    logged_writes = true;
}

// Issue dispatch commands for writing sharded buffer data from pinned memory
void issue_sharded_buffer_pinned_dispatch_command_sequence(
    const void* src,
    Buffer& buffer,
    ShardedBufferWriteDispatchParams& dispatch_params,
    const BufferCorePageMapping& core_page_mapping,
    const CoreCoord& core,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t num_worker_counters = sub_device_ids.size();

    const uint8_t* src_ptr = static_cast<const uint8_t*>(src);
    const uint64_t pinned_src_addr_base = dispatch_params.pinned_src_addr;
    const uint32_t pinned_src_noc_xy = dispatch_params.pinned_src_noc_xy;

    // Build sub-commands on the fly with coalescing
    std::vector<CQDispatchWritePackedLargeUnicastSubCmd> write_sub_cmds;
    std::vector<CQPrefetchRelayLinearPackedSubCmd> relay_sub_cmds;

    const CoreCoord virtual_core = buffer.device()->virtual_core_from_logical_core(core, buffer.core_type());
    const uint32_t noc_xy_addr = buffer.device()->get_noc_unicast_encoding(k_dispatch_downstream_noc, virtual_core);

    // Calculate base destination address for this core
    uint32_t dst_base_address = buffer.address() + core_page_mapping.device_start_page * buffer.aligned_page_size();
    if (buffer.is_dram()) {
        dst_base_address += buffer.device()->allocator()->get_bank_offset(
            BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(core));
    }

    // Issue wait commands once at the beginning if needed
    if (dispatch_params.issue_wait && num_worker_counters > 0) {
        DeviceCommandCalculator calculator;
        for (int i = 0; i < num_worker_counters; ++i) {
            calculator.add_dispatch_wait();
        }

        const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();
        void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);
        HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

        for (const auto& sub_device_id : sub_device_ids) {
            auto offset_index = *sub_device_id;
            command_sequence.add_dispatch_wait(
                CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM,
                0,
                MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(offset_index),
                dispatch_params.expected_num_workers_completed[offset_index]);
        }

        TT_ASSERT(
            command_sequence.write_offset_bytes() == cmd_sequence_sizeB,
            "Command sequence size mismatch, calculator: {}, command sequence: {}",
            cmd_sequence_sizeB,
            command_sequence.write_offset_bytes());

        sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
        sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
        sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
    }

    // Helper lambda to emit a command pair
    auto emit_command_pair = [&]() {
        if (write_sub_cmds.empty() && relay_sub_cmds.empty()) {
            return;
        }
        // Calculate total relay length for the command
        uint32_t total_relay_length = 0;
        for (const auto& relay_sub_cmd : relay_sub_cmds) {
            total_relay_length += relay_sub_cmd.length;
        }

        // Use calculator to compute command sequence size
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_packed_large_unicast(write_sub_cmds.size());
        void* cmd_region = sysmem_manager.issue_queue_reserve(calculator.write_offset_bytes(), dispatch_params.cq_id);
        HugepageDeviceCommand command_sequence(cmd_region, calculator.write_offset_bytes());

        // Add write packed large unicast command
        command_sequence.add_dispatch_write_packed_large_unicast(
            CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_UNKNOWN, l1_alignment, write_sub_cmds.size(), write_sub_cmds);

        TT_ASSERT(
            command_sequence.write_offset_bytes() == calculator.write_offset_bytes(),
            "Command sequence size mismatch, calculator: {}, command sequence: {}",
            calculator.write_offset_bytes(),
            command_sequence.write_offset_bytes());

        sysmem_manager.issue_queue_push_back(calculator.write_offset_bytes(), dispatch_params.cq_id);
        sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
        sysmem_manager.fetch_queue_write(calculator.write_offset_bytes(), dispatch_params.cq_id);

        calculator.clear();

        // Put the CQ_PREFETCH_CMD_RELAY_LINEAR_PACKED_H into its own fetch queue entry so prefetch_h knows to process
        // it.
        if (dispatch_params.remote_chip) {
            calculator.add_prefetch_relay_linear_packed_h(relay_sub_cmds.size());
        } else {
            calculator.add_prefetch_relay_linear_packed(relay_sub_cmds.size());
        }

        cmd_region = sysmem_manager.issue_queue_reserve(calculator.write_offset_bytes(), dispatch_params.cq_id);
        HugepageDeviceCommand prefetch_command_sequence(cmd_region, calculator.write_offset_bytes());

        // Add relay linear packed command
        if (dispatch_params.remote_chip) {
            prefetch_command_sequence.add_prefetch_relay_linear_packed_h(
                pinned_src_noc_xy, total_relay_length, relay_sub_cmds, relay_sub_cmds.size(), 0);
        } else {
            prefetch_command_sequence.add_prefetch_relay_linear_packed(
                pinned_src_noc_xy, total_relay_length, relay_sub_cmds, relay_sub_cmds.size(), 0);
        }

        TT_ASSERT(
            prefetch_command_sequence.write_offset_bytes() == calculator.write_offset_bytes(),
            "Command sequence size mismatch, calculator: {}, command sequence: {}",
            calculator.write_offset_bytes(),
            prefetch_command_sequence.write_offset_bytes());

        sysmem_manager.issue_queue_push_back(calculator.write_offset_bytes(), dispatch_params.cq_id);
        sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
        sysmem_manager.fetch_queue_write(calculator.write_offset_bytes(), dispatch_params.cq_id);

        // Clear for next batch
        write_sub_cmds.clear();
        relay_sub_cmds.clear();
    };

    // Iterate through host ranges and build sub-commands
    for (const auto& host_range : core_page_mapping.host_ranges) {
        uint64_t src_offset = static_cast<uint64_t>(host_range.host_page_start) * buffer.page_size();
        const uint8_t* src_region_start = src_ptr + src_offset;
        uint64_t src_pinned_addr = pinned_src_addr_base + src_offset;

        uint32_t dst_addr = dst_base_address + (host_range.device_page_offset - core_page_mapping.device_start_page) *
                                                   buffer.aligned_page_size();
        uint32_t data_length = host_range.num_pages * buffer.page_size();

        // Assert alignments (should have been checked in write_to_device_buffer)
        TT_ASSERT(
            reinterpret_cast<uintptr_t>(src_region_start) % l1_alignment == 0,
            "Source address {:#x} must be L1-aligned to {} bytes",
            reinterpret_cast<uintptr_t>(src_region_start),
            l1_alignment);
        TT_ASSERT(
            dst_addr % l1_alignment == 0,
            "Destination address {:#x} must be L1-aligned to {} bytes",
            dst_addr,
            l1_alignment);

        // Align source address to PCIe alignment if needed
        uint64_t aligned_src_addr = src_pinned_addr;
        uint32_t padding_bytes = 0;
        if (src_pinned_addr % pcie_alignment != 0) {
            padding_bytes = src_pinned_addr % pcie_alignment;
            aligned_src_addr = src_pinned_addr - padding_bytes;
        }

        uint32_t total_read_length = padding_bytes + data_length;

        // Determine if relay or write can be coalesced
        bool can_coalesce_relay = false;
        if (!relay_sub_cmds.empty()) {
            auto& last_relay = relay_sub_cmds.back();
            if (last_relay.addr + last_relay.length == aligned_src_addr) {
                can_coalesce_relay = true;
            }
        }

        bool can_coalesce_write = false;
        if (padding_bytes == 0 && !write_sub_cmds.empty()) {
            auto& last_write = write_sub_cmds.back();
            if (last_write.noc_xy_addr == noc_xy_addr &&
                last_write.addr != CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_UNICAST_ADDR_DISCARD &&
                last_write.addr + last_write.length == dst_addr) {
                can_coalesce_write = true;
            }
        }

        // Calculate new counts after adding this range
        uint32_t new_relay_count = relay_sub_cmds.size() + (can_coalesce_relay ? 0 : 1);
        uint32_t new_write_count = write_sub_cmds.size() + (padding_bytes > 0 ? 1 : 0) +  // discard command if padding
                                   (can_coalesce_write ? 0 : 1);                          // data command

        // Check if either would exceed limits
        if (new_relay_count > CQ_PREFETCH_CMD_RELAY_LINEAR_PACKED_MAX_SUB_CMDS ||
            new_write_count > CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_UNICAST_MAX_SUB_CMDS) {
            // Emit command pair before adding new commands
            emit_command_pair();
            // After emitting, we can't coalesce with previous commands (vectors are now empty)
            can_coalesce_relay = false;
            can_coalesce_write = false;
        }

        // Add or coalesce relay sub-command
        if (can_coalesce_relay) {
            auto& last_relay = relay_sub_cmds.back();
            last_relay.length += total_read_length;
        } else {
            CQPrefetchRelayLinearPackedSubCmd relay_sub_cmd;
            relay_sub_cmd.addr = aligned_src_addr;
            relay_sub_cmd.length = total_read_length;
            relay_sub_cmds.push_back(relay_sub_cmd);
        }

        // Add discard sub-command if padding was needed
        if (padding_bytes > 0) {
            CQDispatchWritePackedLargeUnicastSubCmd discard_sub_cmd;
            discard_sub_cmd.noc_xy_addr = noc_xy_addr;
            discard_sub_cmd.addr = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_UNICAST_ADDR_DISCARD;
            discard_sub_cmd.length = padding_bytes;
            write_sub_cmds.push_back(discard_sub_cmd);
        }

        // Add or coalesce write sub-command
        if (can_coalesce_write) {
            auto& last_write = write_sub_cmds.back();
            last_write.length += data_length;
        } else {
            CQDispatchWritePackedLargeUnicastSubCmd write_sub_cmd;
            write_sub_cmd.noc_xy_addr = noc_xy_addr;
            write_sub_cmd.addr = dst_addr;
            write_sub_cmd.length = data_length;
            write_sub_cmds.push_back(write_sub_cmd);
        }
    }

    // Emit final command pair with remaining sub-commands
    emit_command_pair();
}

// Issue dispatch commands for writing buffer data
template <typename T>
void issue_buffer_dispatch_command_sequence(
    const void* src,
    Buffer& buffer,
    T& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType /*dispatch_core_type*/) {
    uint32_t num_worker_counters = sub_device_ids.size();
    bool use_pinned_memory = dispatch_params.use_pinned_transfer;
    uint32_t num_pages_to_write =
        use_pinned_memory ? dispatch_params.total_pages_to_write : dispatch_params.pages_per_txn;
    uint64_t data_size_bytes = uint64_t(num_pages_to_write) * dispatch_params.page_size_to_write;

    tt::tt_metal::DeviceCommandCalculator calculator;
    if (dispatch_params.issue_wait) {
        for (int i = 0; i < num_worker_counters; ++i) {
            calculator.add_dispatch_wait();
        }
    }
    if constexpr (std::is_same_v<T, ShardedBufferWriteDispatchParams>) {
        calculator.add_dispatch_write_linear<true, false>(data_size_bytes);
    } else {
        if (dispatch_params.alignment_prefix_bytes > 0) {
            // Use custom inline size variant to account for alignment prefix bytes
            calculator.add_dispatch_write_paged_with_custom_inline_size(0, 0, dispatch_params.alignment_prefix_bytes);
        } else {
            calculator.add_dispatch_write_paged<false>(0, 0);  // arguments are don't care for <false>
        }
    }
    if (use_pinned_memory) {
        // What follows is a command (which must be aligned), not data.
        calculator.add_alignment();
    } else {
        calculator.add_data<false>(data_size_bytes);
    }

    const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();
    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);

    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    if (dispatch_params.issue_wait) {
        for (const auto& sub_device_id : sub_device_ids) {
            auto offset_index = *sub_device_id;
            command_sequence.add_dispatch_wait(
                CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM,
                0,
                MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(offset_index),
                dispatch_params.expected_num_workers_completed[offset_index]);
        }
    }
    if constexpr (std::is_same_v<T, ShardedBufferWriteDispatchParams>) {
        populate_sharded_buffer_write_dispatch_cmds(src, command_sequence, buffer, dispatch_params);
    } else {
        populate_interleaved_buffer_write_dispatch_cmds(src, command_sequence, buffer, dispatch_params);
    }
    TT_ASSERT(
        command_sequence.write_offset_bytes() == cmd_sequence_sizeB,
        "Command sequence size mismatch, calculator: {}, command sequence: {}",
        cmd_sequence_sizeB,
        command_sequence.write_offset_bytes());

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);

    if (use_pinned_memory) {
        // Send CQ_PREFETCH_CMD_RELAY_LINEAR command in a separate fetch Q entry to ensure it will be processed in
        // prefetch_h for remote device. If we don't do this, prefetch_h will "fetch" it along with the
        // CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH command and send it to prefetch_d

        // Adjust address and length if we sent alignment prefix bytes inline
        uint64_t relay_src_addr = dispatch_params.pinned_src_addr;
        uint64_t relay_data_size = (uint64_t)dispatch_params.total_pages_to_write * dispatch_params.page_size_to_write;
        if constexpr (std::is_same_v<T, InterleavedBufferWriteDispatchParams>) {
            TT_ASSERT(
                dispatch_params.alignment_prefix_bytes % MetalContext::instance().hal().get_alignment(HalMemType::L1) ==
                    0,
                "Alignment prefix is not aligned to L1");
            relay_src_addr += dispatch_params.alignment_prefix_bytes;
            relay_data_size -= dispatch_params.alignment_prefix_bytes;
        }

        calculator.clear();
        if (dispatch_params.remote_chip) {
            calculator.add_prefetch_relay_linear_h();
        } else {
            calculator.add_prefetch_relay_linear();
        }
        const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();
        void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);
        HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

        if (dispatch_params.remote_chip) {
            command_sequence.add_prefetch_relay_linear_h(
                dispatch_params.pinned_src_noc_xy, relay_data_size, relay_src_addr);
        } else {
            command_sequence.add_prefetch_relay_linear(
                dispatch_params.pinned_src_noc_xy, relay_data_size, relay_src_addr);
        }
        sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
        sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
        sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
    }
}

// Top level helper functions to write buffer data
void write_interleaved_buffer_to_device(
    const void* src,
    InterleavedBufferWriteDispatchParams& dispatch_params,
    Buffer& buffer,
    const BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type) {
    bool use_pinned_memory = dispatch_params.use_pinned_transfer;

    // data appended after CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_WRITE_PAGED
    uint32_t byte_offset_in_cq = MetalContext::instance().hal().get_alignment(HalMemType::HOST);

    dispatch_params.calculate_issue_wait();
    update_offset_on_issue_wait_cmd(byte_offset_in_cq, dispatch_params.issue_wait, sub_device_ids.size());

    if (use_pinned_memory) {
        if (dispatch_params.is_page_offset_out_of_bounds()) {
            dispatch_params.update_params_to_be_within_bounds();
        }
        issue_buffer_dispatch_command_sequence(src, buffer, dispatch_params, sub_device_ids, dispatch_core_type);
    } else {
        // Prefetcher will read from hugepage in one or more iterations depending on transfer size
        while (dispatch_params.total_pages_to_write > 0) {
            // Ensure page offset can fit in uint16_t
            if (dispatch_params.is_page_offset_out_of_bounds()) {
                dispatch_params.update_params_to_be_within_bounds();
            }

            const int32_t num_pages_available_in_cq =
                calculate_num_pages_available_in_cq(dispatch_params, buf_dispatch_constants, byte_offset_in_cq);
            if (num_pages_available_in_cq <= 0) {
                SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
                sysmem_manager.wrap_issue_queue_wr_ptr(dispatch_params.cq_id);
                continue;
            }

            log_debug(
                tt::LogDispatch, "write_interleaved_buffer_to_device for command queue {}", dispatch_params.cq_id);

            dispatch_params.calculate_num_pages_for_write_transaction(num_pages_available_in_cq);
            issue_buffer_dispatch_command_sequence(src, buffer, dispatch_params, sub_device_ids, dispatch_core_type);
            dispatch_params.update_params_after_write_transaction();
        }
    }
}

void write_sharded_buffer_to_core(
    const void* src,
    uint32_t /*core_id*/,
    const BufferCorePageMapping& core_page_mapping,
    Buffer& buffer,
    ShardedBufferWriteDispatchParams& dispatch_params,
    const BufferDispatchConstants& buf_dispatch_constants,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const CoreCoord core,
    CoreType dispatch_core_type) {
    // Skip writing the padded pages along the bottom
    // Currently since writing sharded tensors uses write_linear, we write the padded pages on width
    // Alternative write each page row into separate commands, or have a strided linear write

    if (tt::tt_metal::GraphTracker::instance().hook_write_to_device(&buffer)) {
        return;
    }

    bool use_pinned_memory = dispatch_params.use_pinned_transfer;

    dispatch_params.reset_params_for_core(core, core_page_mapping);

    dispatch_params.calculate_issue_wait();

    if (use_pinned_memory) {
        issue_sharded_buffer_pinned_dispatch_command_sequence(
            src, buffer, dispatch_params, core_page_mapping, core, sub_device_ids);
    } else {
        DeviceCommandCalculator calculator;
        calculator.add_dispatch_write_linear<true, false>(0);
        uint32_t data_offset_bytes = calculator.write_offset_bytes();
        update_offset_on_issue_wait_cmd(data_offset_bytes, dispatch_params.issue_wait, sub_device_ids.size());

        while (dispatch_params.core_num_pages_remaining_to_write != 0) {
            const int32_t num_pages_available_in_cq =
                calculate_num_pages_available_in_cq(dispatch_params, buf_dispatch_constants, data_offset_bytes);
            if (num_pages_available_in_cq <= 0) {
                SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
                sysmem_manager.wrap_issue_queue_wr_ptr(dispatch_params.cq_id);
                continue;
            }

            log_debug(tt::LogDispatch, "write_sharded_buffer_to_core for command queue {}", dispatch_params.cq_id);

            dispatch_params.calculate_params_for_write_transaction(num_pages_available_in_cq);
            issue_buffer_dispatch_command_sequence(src, buffer, dispatch_params, sub_device_ids, dispatch_core_type);
            dispatch_params.update_params_after_write_transaction();
        }
    }
}

// Main API to write buffer data
bool write_to_device_buffer(
    const void* src,
    Buffer& buffer,
    uint32_t cq_id,
    tt::stl::Span<const uint32_t> expected_num_workers_completed,
    CoreType dispatch_core_type,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const std::shared_ptr<experimental::PinnedMemory>& pinned_memory) {
    SystemMemoryManager& sysmem_manager = buffer.device()->sysmem_manager();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    if (tt::tt_metal::GraphTracker::instance().hook_write_to_device(&buffer)) {
        return false;
    }

    const BufferDispatchConstants buf_dispatch_constants =
        generate_buffer_dispatch_constants(sysmem_manager, dispatch_core_type, cq_id);

    // TODO: When writing to L1, modify this function to use enqueue_write_to_core
    // Determine whether pinned direct read is feasible, and derive src noc params
    const bool is_unpadded = (buffer.page_size() == buffer.aligned_page_size());
    const bool has_pinned_inputs = (src != nullptr && pinned_memory != nullptr);
    uint32_t pinned_src_noc_xy = 0;
    uint64_t pinned_src_addr = 0;
    bool use_pinned_transfer = false;
    bool remote_chip = false;
    if (has_pinned_inputs && is_unpadded && !is_sharded(buffer.buffer_layout())) {
        auto device_id = buffer.device()->id();
        auto noc_addr_pair_opt = pinned_memory->get_noc_addr(device_id);
        if (noc_addr_pair_opt.has_value()) {
            remote_chip = noc_addr_pair_opt->device_id != device_id;
            const uint64_t pinned_noc_base = noc_addr_pair_opt->addr;
            const uint8_t* pinned_host_base = static_cast<const uint8_t*>(pinned_memory->get_host_ptr());
            const uint8_t* src_ptr = static_cast<const uint8_t*>(src);
            const uint64_t pinned_size = pinned_memory->get_buffer_size();
            auto region = buffer.root_buffer_region();
            const uint8_t* src_region_start = src_ptr + region.offset;
            const uint8_t* src_region_end = src_region_start + region.size;
            // Check against L1 alignment because we need the copy from the prefetcher to the dispatcher to be aligned.
            if (reinterpret_cast<uintptr_t>(src_region_start) % hal.get_read_alignment(HalMemType::L1) != 0) {
                log_info(
                    tt::LogMetal,
                    "Pinned source memory start address {:#x} must be aligned {} B",
                    reinterpret_cast<uintptr_t>(src_region_start),
                    hal.get_read_alignment(HalMemType::HOST));
            } else if ((src_region_start < pinned_host_base) or (pinned_host_base + pinned_size < src_region_end)) {
                log_info(
                    tt::LogMetal,
                    "Pinned memory region must contain source buffer region: pinned region start:{:#X} end:{:#X} src "
                    "start:{:#X} end:{:#X}",
                    reinterpret_cast<uintptr_t>(pinned_host_base),
                    reinterpret_cast<uintptr_t>(pinned_host_base + pinned_size),
                    reinterpret_cast<uintptr_t>(src_region_start),
                    reinterpret_cast<uintptr_t>(src_region_end));
            } else {
                const uint64_t src_offset_base = static_cast<uintptr_t>(src_region_start - pinned_host_base);
                pinned_src_addr = pinned_noc_base + src_offset_base;
                pinned_src_noc_xy = noc_addr_pair_opt->pcie_xy_enc;
                use_pinned_transfer = true;
            }
        }
    }
    if (is_sharded(buffer.buffer_layout())) {
        // Check alignment for sharded buffer pinned transfer
        if (has_pinned_inputs && is_unpadded) {
            auto device_id = buffer.device()->id();
            auto noc_addr_pair_opt = pinned_memory->get_noc_addr(device_id);
            if (noc_addr_pair_opt.has_value()) {
                remote_chip = noc_addr_pair_opt->device_id != device_id;
                const uint64_t pinned_noc_base = noc_addr_pair_opt->addr;
                const uint8_t* pinned_host_base = static_cast<const uint8_t*>(pinned_memory->get_host_ptr());
                const uint8_t* src_ptr = static_cast<const uint8_t*>(src);
                const uint64_t pinned_size = pinned_memory->get_buffer_size();

                // Get L1 alignment requirement
                const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

                // Check all source and destination addresses for L1 alignment
                bool all_aligned = true;
                auto buffer_page_mapping = buffer.get_buffer_page_mapping();
                const std::vector<CoreCoord>& cores = buffer_page_mapping->all_cores;

                for (uint32_t core_id = 0; core_id < buffer.num_cores() && all_aligned; ++core_id) {
                    for (const BufferCorePageMapping& core_page_mapping :
                         buffer_page_mapping->core_page_mappings[core_id]) {
                        // Check destination L1 address alignment
                        uint32_t dst_address =
                            buffer.address() + core_page_mapping.device_start_page * buffer.aligned_page_size();
                        if (buffer.is_dram()) {
                            dst_address += buffer.device()->allocator()->get_bank_offset(
                                BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(cores[core_id]));
                        }
                        if (dst_address % l1_alignment != 0) {
                            all_aligned = false;
                            break;
                        }

                        // Check source address alignment for each host range
                        for (const auto& host_range : core_page_mapping.host_ranges) {
                            uint64_t src_offset =
                                static_cast<uint64_t>(host_range.host_page_start) * buffer.page_size();
                            const uint8_t* src_region_start = src_ptr + src_offset;
                            uint32_t data_length = host_range.num_pages * buffer.page_size();
                            const uint8_t* src_region_end = src_region_start + data_length;

                            // Check if within pinned region
                            if (src_region_start < pinned_host_base ||
                                src_region_end > pinned_host_base + pinned_size) {
                                all_aligned = false;
                                break;
                            }

                            // Check L1 alignment of source
                            if (reinterpret_cast<uintptr_t>(src_region_start) % l1_alignment != 0) {
                                all_aligned = false;
                                break;
                            }
                        }

                        if (!all_aligned) {
                            break;
                        }
                    }
                }

                if (all_aligned) {
                    pinned_src_addr = pinned_noc_base;
                    pinned_src_noc_xy = noc_addr_pair_opt->pcie_xy_enc;
                    use_pinned_transfer = true;
                }
            }
        }
        if (has_pinned_inputs) {
          //  log_info(tt::LogMetal, "Sharded using pinned transfer: {}", use_pinned_transfer);
        }

        ShardedBufferWriteDispatchParams dispatch_params(
            &buffer,
            buffer.size() / buffer.page_size(),
            cq_id,
            expected_num_workers_completed,
            sub_device_ids,
            pinned_src_noc_xy,
            pinned_src_addr,
            use_pinned_transfer,
            remote_chip);
        const std::vector<CoreCoord>& cores = dispatch_params.buffer_page_mapping->all_cores;
        // Since we read core by core we are reading the device pages sequentially
        for (uint32_t core_id = 0; core_id < buffer.num_cores(); ++core_id) {
            for (const BufferCorePageMapping& core_page_mapping :
                 dispatch_params.buffer_page_mapping->core_page_mappings[core_id]) {
                write_sharded_buffer_to_core(
                    src,
                    core_id,
                    core_page_mapping,
                    buffer,
                    dispatch_params,
                    buf_dispatch_constants,
                    sub_device_ids,
                    cores[core_id],
                    dispatch_core_type);
            }
        }
    } else {
        auto root_buffer = buffer.root_buffer();
        auto region = buffer.root_buffer_region();
        InterleavedBufferWriteDispatchParamsVariant dispatch_params_variant =
            initialize_interleaved_buf_dispatch_params(
                *root_buffer,
                cq_id,
                expected_num_workers_completed,
                region,
                sub_device_ids,
                pinned_src_noc_xy,
                pinned_src_addr,
                use_pinned_transfer,
                remote_chip);

        InterleavedBufferWriteDispatchParams* dispatch_params = std::visit(
            ttsl::overloaded{
                [](std::derived_from<InterleavedBufferWriteDispatchParams> auto& val)
                    -> InterleavedBufferWriteDispatchParams* { return &val; },
                [](std::monostate) -> InterleavedBufferWriteDispatchParams* { return nullptr; },
            },
            dispatch_params_variant);
        TT_ASSERT(dispatch_params != nullptr);

        write_interleaved_buffer_to_device(
            src, *dispatch_params, *root_buffer, buf_dispatch_constants, sub_device_ids, dispatch_core_type);
        return use_pinned_transfer;
    }
    return use_pinned_transfer;
}

// ====== Utility Functions for Reads ======

// Initialize Dispatch Parameters - reused across write txns
ShardedBufferReadDispatchParams initialize_sharded_buf_read_dispatch_params(
    Buffer& buffer, uint32_t cq_id, tt::stl::Span<const uint32_t> expected_num_workers_completed) {
    // Note that the src_page_index is the device page idx, not the host page idx
    // Since we read core by core we are reading the device pages sequentially
    ShardedBufferReadDispatchParams dispatch_params;

    dispatch_params.cq_id = cq_id;
    dispatch_params.device = buffer.device();
    dispatch_params.padded_page_size = buffer.aligned_page_size();
    dispatch_params.src_page_index = 0;
    dispatch_params.unpadded_dst_offset = 0;
    dispatch_params.buffer_page_mapping = buffer.get_buffer_page_mapping();
    dispatch_params.total_pages_to_read = buffer.size() / buffer.page_size();
    dispatch_params.total_pages_read = 0;
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;
    dispatch_params.pages_per_txn = 0;
    return dispatch_params;
}

BufferReadDispatchParams initialize_interleaved_buf_read_dispatch_params(
    Buffer& buffer, uint32_t cq_id, tt::stl::Span<const uint32_t> expected_num_workers_completed) {
    auto root_buffer = buffer.root_buffer();
    const BufferRegion region = buffer.root_buffer_region();
    IDevice* device = root_buffer->device();

    BufferReadDispatchParams dispatch_params;
    dispatch_params.total_pages_to_read = region.size / root_buffer->page_size();
    dispatch_params.src_page_index = region.offset / root_buffer->page_size();
    dispatch_params.cq_id = cq_id;
    dispatch_params.device = device;
    dispatch_params.address = root_buffer->address();
    dispatch_params.unpadded_dst_offset = 0;
    dispatch_params.expected_num_workers_completed = expected_num_workers_completed;
    dispatch_params.num_banks = device->allocator()->get_num_banks(root_buffer->buffer_type());
    dispatch_params.padded_page_size = root_buffer->aligned_page_size();
    dispatch_params.pages_per_txn = 0;

    return dispatch_params;
}

// Issue dispatch commands for forwarding device buffer data to the Completion Queue
template <typename T>
void issue_read_buffer_dispatch_command_sequence(
    Buffer& buffer,
    T& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType /*dispatch_core_type*/) {
    if (tt::tt_metal::GraphTracker::instance().hook_read_from_device(&buffer)) {
        return;
    }

    // Mock devices don't have real hardware to read from, skip actual dispatch
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock) {
        return;
    }

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    SystemMemoryManager& sysmem_manager = dispatch_params.device->sysmem_manager();
    uint32_t num_worker_counters = sub_device_ids.size();

    // Precompute whether pinned direct write is feasible, and derive dst noc params
    const bool is_unpadded = (buffer.page_size() == dispatch_params.padded_page_size);
    const bool has_pinned_inputs = (dispatch_params.dst != nullptr && dispatch_params.pinned_memory != nullptr);
    const uint64_t xfer_bytes = static_cast<uint64_t>(dispatch_params.pages_per_txn) * dispatch_params.padded_page_size;
    bool use_pinned_transfer = false;
    uint32_t pinned_dst_noc_xy = 0;
    uint64_t pinned_dst_addr = 0;

    if (has_pinned_inputs && is_unpadded) {
        auto noc_addr_pair_opt = dispatch_params.pinned_memory->get_noc_addr(dispatch_params.device->id());
        if (noc_addr_pair_opt.has_value()) {
            const uint64_t pinned_noc_base = noc_addr_pair_opt->addr;
            const uint8_t* pinned_host_base =
                static_cast<const uint8_t*>(dispatch_params.pinned_memory->get_host_ptr());
            const uint8_t* dst_ptr = static_cast<const uint8_t*>(dispatch_params.dst);
            const uint8_t* pinned_region_end = pinned_host_base + dispatch_params.pinned_memory->get_buffer_size();
            const uint8_t* dst_region_start = dst_ptr + dispatch_params.unpadded_dst_offset;
            const uint8_t* dst_region_end = dst_region_start + xfer_bytes;
            if ((reinterpret_cast<uintptr_t>(dst_region_start) % hal.get_write_alignment(HalMemType::HOST) == 0) &&
                (dst_region_start >= pinned_host_base) && (dst_region_end <= pinned_region_end)) {
                const uint64_t dst_offset_base = static_cast<uint64_t>(dst_region_start - pinned_host_base);
                pinned_dst_addr = dst_offset_base + pinned_noc_base;
                pinned_dst_noc_xy = noc_addr_pair_opt->pcie_xy_enc;
                use_pinned_transfer = true;
            }
        }
    }

    // Build calculator with the chosen path
    tt::tt_metal::DeviceCommandCalculator calculator;
    for (uint32_t i = 0; i < num_worker_counters; ++i) {
        calculator.add_dispatch_wait();
    }
    calculator.add_prefetch_stall();
    if (use_pinned_transfer) {
        // When flush_prefetch=false and inline_data=false, size is ignored.
        calculator.add_dispatch_write_linear_h<false, false>(0);
    } else {
        calculator.add_dispatch_write_linear_host();
    }
    // Prefetch relay cmd has fixed header size in calculator regardless of type
    calculator.add_prefetch_relay_paged();
    const uint32_t cmd_sequence_sizeB = calculator.write_offset_bytes();

    void* cmd_region = sysmem_manager.issue_queue_reserve(cmd_sequence_sizeB, dispatch_params.cq_id);
    HugepageDeviceCommand command_sequence(cmd_region, cmd_sequence_sizeB);

    uint32_t last_index = num_worker_counters - 1;
    // We only need the write barrier + prefetch stall for the last wait cmd
    for (uint32_t i = 0; i < last_index; ++i) {
        auto offset_index = *sub_device_ids[i];
        command_sequence.add_dispatch_wait(
            CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM,
            0,
            MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(offset_index),
            dispatch_params.expected_num_workers_completed[offset_index]);
    }
    auto offset_index = *sub_device_ids[last_index];
    command_sequence.add_dispatch_wait_with_prefetch_stall(
        CQ_DISPATCH_CMD_WAIT_FLAG_WAIT_STREAM | CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER,
        0,
        MetalContext::instance().dispatch_mem_map().get_dispatch_stream_index(offset_index),
        dispatch_params.expected_num_workers_completed[offset_index]);

    // Select write op once, then unify relay
    if (use_pinned_transfer) {
        command_sequence.add_dispatch_write_linear_h<false, false>(0, pinned_dst_noc_xy, pinned_dst_addr, xfer_bytes);
    } else {
        bool flush_prefetch = false;
        command_sequence.add_dispatch_write_host(flush_prefetch, xfer_bytes, false, 0);
    }

    // Buffer layout specific logic
    if constexpr (std::is_same_v<T, ShardedBufferReadDispatchParams>) {
        const CoreCoord virtual_core =
            buffer.device()->virtual_core_from_logical_core(dispatch_params.core, buffer.core_type());
        command_sequence.add_prefetch_relay_linear(
            dispatch_params.device->get_noc_unicast_encoding(k_dispatch_downstream_noc, virtual_core),
            xfer_bytes,
            dispatch_params.address);
    } else {
        command_sequence.add_prefetch_relay_paged(
            buffer.is_dram(),
            dispatch_params.src_page_index,
            dispatch_params.address,
            dispatch_params.padded_page_size,
            dispatch_params.pages_per_txn);
    }

    // Mark whether completion read is needed
    dispatch_params.requires_completion_read = !use_pinned_transfer;

    sysmem_manager.issue_queue_push_back(cmd_sequence_sizeB, dispatch_params.cq_id);
    sysmem_manager.fetch_queue_reserve_back(dispatch_params.cq_id);
    sysmem_manager.fetch_queue_write(cmd_sequence_sizeB, dispatch_params.cq_id);
}

// Top level functions to copy device buffers into the completion queue
void copy_sharded_buffer_from_core_to_completion_queue(
    uint32_t /*core_id*/,
    const BufferCorePageMapping& core_page_mapping,
    Buffer& buffer,
    ShardedBufferReadDispatchParams& dispatch_params,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    const CoreCoord core,
    CoreType dispatch_core_type) {
    auto address = buffer.address();

    if (buffer.is_dram()) {
        address += buffer.device()->allocator()->get_bank_offset(
            BufferType::DRAM, buffer.device()->dram_channel_from_logical_core(core));
    }
    address += core_page_mapping.device_start_page * buffer.aligned_page_size();

    dispatch_params.pages_per_txn = core_page_mapping.num_pages;
    dispatch_params.total_pages_to_read -= dispatch_params.pages_per_txn;
    dispatch_params.total_pages_read += dispatch_params.pages_per_txn;
    dispatch_params.core_page_mapping = &core_page_mapping;

    if (dispatch_params.pages_per_txn > 0) {
        dispatch_params.address = address;
        dispatch_params.core = core;
        issue_read_buffer_dispatch_command_sequence(buffer, dispatch_params, sub_device_ids, dispatch_core_type);
    }
}

void copy_interleaved_buffer_to_completion_queue(
    BufferReadDispatchParams& dispatch_params,
    Buffer& buffer,
    tt::stl::Span<const SubDeviceId> sub_device_ids,
    CoreType dispatch_core_type,
    void* dst,
    const std::shared_ptr<experimental::PinnedMemory>& pinned_memory) {
    if (dispatch_params.total_pages_to_read > 0) {
        // Only 8 bits are assigned for the page offset in CQPrefetchRelayPagedCmd
        // To handle larger page offsets move bank base address up and update page offset to be relative to the new
        // bank address
        if (dispatch_params.src_page_index > CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK) {
            dispatch_params.update_params_to_be_within_bounds(buffer);
        }

        dispatch_params.dst = dst;
        dispatch_params.pinned_memory = pinned_memory;
        dispatch_params.calculate_num_pages_for_read_transaction();
        issue_read_buffer_dispatch_command_sequence(buffer, dispatch_params, sub_device_ids, dispatch_core_type);

        dispatch_params.update_params_after_read_transaction();
    }
}

// Functions used to copy buffer data from completion queue into user space
std::shared_ptr<tt::tt_metal::CompletionReaderVariant> generate_sharded_buffer_read_descriptor(
    void* dst, ShardedBufferReadDispatchParams& dispatch_params, Buffer& buffer) {
    // Increment the src_page_index after the Read Buffer Descriptor has been populated
    // for the current core/txn
    dispatch_params.src_page_index += dispatch_params.pages_per_txn;
    return std::make_shared<tt::tt_metal::CompletionReaderVariant>(
        std::in_place_type<tt::tt_metal::ReadBufferDescriptor>,
        buffer.page_size(),
        dispatch_params.padded_page_size,
        dst,
        dispatch_params.unpadded_dst_offset,
        dispatch_params.pages_per_txn,
        dispatch_params.buffer_page_mapping,
        dispatch_params.core_page_mapping);
}

std::shared_ptr<tt::tt_metal::CompletionReaderVariant> generate_interleaved_buffer_read_descriptor(
    void* dst, const BufferReadDispatchParams& dispatch_params, Buffer& buffer) {
    return std::make_shared<tt::tt_metal::CompletionReaderVariant>(
        std::in_place_type<tt::tt_metal::ReadBufferDescriptor>,
        buffer.page_size(),
        dispatch_params.padded_page_size,
        dst,
        dispatch_params.unpadded_dst_offset,
        dispatch_params.total_pages_read);
}

void copy_completion_queue_data_into_user_space(
    const ReadBufferDescriptor& read_buffer_descriptor,
    ChipId mmio_device_id,
    uint16_t channel,
    uint32_t cq_id,
    SystemMemoryManager& sysmem_manager,
    std::atomic<bool>& exit_condition) {
    const auto& [page_size, padded_page_size, buffer_page_mapping, core_page_mapping, dst, dst_offset, num_pages_read] =
        read_buffer_descriptor;
    const DeviceAddr padded_num_bytes = ((DeviceAddr)num_pages_read * padded_page_size) + sizeof(CQDispatchCmd);
    uint64_t contig_dst_offset = dst_offset;
    DeviceAddr remaining_bytes_to_read = padded_num_bytes;

    // track the amount of bytes read in the last non-aligned page
    uint32_t remaining_bytes_of_nonaligned_page = 0;
    std::optional<uint32_t> host_page_id = std::nullopt;
    uint32_t offset_in_completion_q_data = sizeof(CQDispatchCmd);

    uint32_t pad_size_bytes = padded_page_size - page_size;

    BufferCorePageMapping::Iterator core_page_mapping_it;
    if (core_page_mapping) {
        core_page_mapping_it = core_page_mapping->begin();
    }

    while (remaining_bytes_to_read != 0) {
        uint32_t completion_queue_write_ptr_and_toggle =
            sysmem_manager.completion_queue_wait_front(cq_id, exit_condition);

        if (exit_condition) {
            break;
        }

        uint32_t completion_q_write_ptr = (completion_queue_write_ptr_and_toggle & 0x7fffffff) << 4;
        uint32_t completion_q_write_toggle = completion_queue_write_ptr_and_toggle >> (31);
        uint32_t completion_q_read_ptr = sysmem_manager.get_completion_queue_read_ptr(cq_id);
        uint32_t completion_q_read_toggle = sysmem_manager.get_completion_queue_read_toggle(cq_id);

        uint32_t bytes_avail_in_completion_queue;
        if (completion_q_write_ptr > completion_q_read_ptr and completion_q_write_toggle == completion_q_read_toggle) {
            bytes_avail_in_completion_queue = completion_q_write_ptr - completion_q_read_ptr;
        } else {
            // Completion queue write pointer on device wrapped but read pointer is lagging behind.
            //  In this case read up until the end of the completion queue first
            bytes_avail_in_completion_queue = sysmem_manager.get_completion_queue_limit(cq_id) - completion_q_read_ptr;
        }

        // completion queue write ptr on device could have wrapped but our read ptr is lagging behind
        uint32_t bytes_xfered =
            static_cast<uint32_t>(std::min(remaining_bytes_to_read, (DeviceAddr)bytes_avail_in_completion_queue));
        uint32_t num_pages_xfered = div_up(bytes_xfered, DispatchSettings::TRANSFER_PAGE_SIZE);

        remaining_bytes_to_read -= bytes_xfered;

        if (buffer_page_mapping == nullptr) {
            void* contiguous_dst = (void*)(uint64_t(dst) + contig_dst_offset);
            if (page_size == padded_page_size) {
                uint32_t data_bytes_xfered = bytes_xfered - offset_in_completion_q_data;
                tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
                    contiguous_dst,
                    data_bytes_xfered,
                    completion_q_read_ptr + offset_in_completion_q_data,
                    mmio_device_id,
                    channel);
                contig_dst_offset += data_bytes_xfered;
                offset_in_completion_q_data = 0;
            } else {
                uint32_t src_offset_bytes = offset_in_completion_q_data;
                offset_in_completion_q_data = 0;
                uint64_t dst_offset_bytes = 0;

                while (src_offset_bytes < bytes_xfered) {
                    uint32_t src_offset_increment = padded_page_size;
                    uint32_t num_bytes_to_copy = 0;

                    if (remaining_bytes_of_nonaligned_page > 0) {
                        // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                        remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                        src_offset_increment = num_bytes_to_copy;
                        // We finished copying the page
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                            // There is more data after padding
                            if (rem_bytes_in_cq >= pad_size_bytes) {
                                src_offset_increment += pad_size_bytes;
                                // Only pad data left in queue
                            } else {
                                src_offset_increment += rem_bytes_in_cq;
                                offset_in_completion_q_data = pad_size_bytes - rem_bytes_in_cq;
                            }
                        }
                    } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                        // Case 2: Last page of data that was popped off the completion queue
                        // Don't need to compute src_offset_increment since this is end of loop
                        uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                        num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                        remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                        // We've copied needed data, start of next read is offset due to remaining pad bytes
                        if (remaining_bytes_of_nonaligned_page == 0) {
                            offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        }
                    } else {
                        num_bytes_to_copy = page_size;
                    }

                    tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
                        (char*)(uint64_t(contiguous_dst) + dst_offset_bytes),
                        num_bytes_to_copy,
                        completion_q_read_ptr + src_offset_bytes,
                        mmio_device_id,
                        channel);

                    src_offset_bytes += src_offset_increment;
                    dst_offset_bytes += num_bytes_to_copy;
                    contig_dst_offset += num_bytes_to_copy;
                }
            }
        } else {
            uint32_t src_offset_bytes = offset_in_completion_q_data;
            offset_in_completion_q_data = 0;
            uint64_t dst_offset_bytes = contig_dst_offset;
            uint32_t num_bytes_to_copy = 0;

            while (src_offset_bytes < bytes_xfered) {
                uint32_t src_offset_increment = padded_page_size;
                if (remaining_bytes_of_nonaligned_page > 0) {
                    // Case 1: Portion of the page was copied into user buffer on the previous completion queue pop.
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(remaining_bytes_of_nonaligned_page, num_bytes_remaining);
                    remaining_bytes_of_nonaligned_page -= num_bytes_to_copy;
                    src_offset_increment = num_bytes_to_copy;
                    // We finished copying the page
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        ++core_page_mapping_it;
                        uint32_t rem_bytes_in_cq = num_bytes_remaining - num_bytes_to_copy;
                        // There is more data after padding
                        if (rem_bytes_in_cq >= pad_size_bytes) {
                            src_offset_increment += pad_size_bytes;
                            offset_in_completion_q_data = 0;
                            // Only pad data left in queue
                        } else {
                            offset_in_completion_q_data = (pad_size_bytes - rem_bytes_in_cq);
                        }
                    }
                    if (!host_page_id.has_value()) {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else if (src_offset_bytes + padded_page_size >= bytes_xfered) {
                    // Case 2: Last page of data that was popped off the completion queue
                    // Don't need to compute src_offset_increment since this is end of loop
                    host_page_id = *core_page_mapping_it;
                    uint32_t num_bytes_remaining = bytes_xfered - src_offset_bytes;
                    num_bytes_to_copy = std::min(num_bytes_remaining, page_size);
                    remaining_bytes_of_nonaligned_page = page_size - num_bytes_to_copy;
                    // We've copied needed data, start of next read is offset due to remaining pad bytes
                    if (remaining_bytes_of_nonaligned_page == 0) {
                        offset_in_completion_q_data = padded_page_size - num_bytes_remaining;
                        ++core_page_mapping_it;
                    }
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = *host_page_id * uint64_t(page_size);
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                } else {
                    num_bytes_to_copy = page_size;
                    host_page_id = *core_page_mapping_it;
                    ++core_page_mapping_it;
                    if (host_page_id.has_value()) {
                        dst_offset_bytes = *host_page_id * uint64_t(page_size);
                    } else {
                        src_offset_bytes += src_offset_increment;
                        continue;
                    }
                }

                tt::tt_metal::MetalContext::instance().get_cluster().read_sysmem(
                    (char*)(uint64_t(dst) + dst_offset_bytes),
                    num_bytes_to_copy,
                    completion_q_read_ptr + src_offset_bytes,
                    mmio_device_id,
                    channel);

                src_offset_bytes += src_offset_increment;
            }
            dst_offset_bytes += num_bytes_to_copy;
            contig_dst_offset = dst_offset_bytes;
        }
        sysmem_manager.completion_queue_pop_front(num_pages_xfered, cq_id);
    }
}

tt::stl::Span<const SubDeviceId> select_sub_device_ids(
    IDevice* device, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    if (sub_device_ids.empty()) {
        return device->get_sub_device_stall_group();
    }
    for (const auto& sub_device_id : sub_device_ids) {
        TT_FATAL(*sub_device_id < device->num_sub_devices(), "Invalid sub-device id specified {}", *sub_device_id);
    }
    return sub_device_ids;
}

template void issue_buffer_dispatch_command_sequence<InterleavedBufferWriteDispatchParams>(
    const void*, Buffer&, InterleavedBufferWriteDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);
template void issue_buffer_dispatch_command_sequence<ShardedBufferWriteDispatchParams>(
    const void*, Buffer&, ShardedBufferWriteDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);

template void issue_read_buffer_dispatch_command_sequence<BufferReadDispatchParams>(
    Buffer&, BufferReadDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);
template void issue_read_buffer_dispatch_command_sequence<ShardedBufferReadDispatchParams>(
    Buffer&, ShardedBufferReadDispatchParams&, tt::stl::Span<const SubDeviceId>, CoreType);

}  // namespace tt::tt_metal::buffer_dispatch
