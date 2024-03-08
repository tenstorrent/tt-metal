// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/buffer.hpp"

#include "common/assert.hpp"
#include "llrt/llrt.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/tt_stl/stacktrace.hpp"

namespace tt {

namespace tt_metal {

bool is_sharded(const TensorMemoryLayout & layout){
    return (
        layout == TensorMemoryLayout::HEIGHT_SHARDED ||
        layout == TensorMemoryLayout::WIDTH_SHARDED ||
        layout == TensorMemoryLayout::BLOCK_SHARDED );
}


void validate_buffer_size_and_page_size(uint64_t size, uint64_t page_size, const BufferType &buffer_type, const TensorMemoryLayout &buffer_layout, std::optional<ShardSpecBuffer> shard_parameters) {
    TT_FATAL(size != 0 and page_size != 0, "Buffer size and page size should be larger than 0 bytes!");
    bool valid_page_size = (size % page_size == 0);
    TT_FATAL(valid_page_size, "For valid non-interleaved buffers page size {} must equal buffer size {}. For interleaved-buffers page size should be divisible by buffer size", page_size, size);
    TT_FATAL(page_size % sizeof(uint32_t) == 0, "Page size must be divisible by sizeof(uint32_t) because buffers hold uint32_t values");
    if(buffer_layout == TensorMemoryLayout::SINGLE_BANK){
        TT_ASSERT(page_size == size , "Continguous buffer must be one contiguous page");
    }
    else if(is_sharded(buffer_layout)){
        TT_ASSERT(shard_parameters != std::nullopt , "Sharded buffers must have a core grid assigned");
    }
}


inline std::vector< std::vector<uint32_t> > core_to_host_pages(
                                                    const uint32_t & total_pages,
                                                    const uint32_t & pages_per_shard,
                                                    const uint32_t & num_shards,
                                                    const TensorMemoryLayout & layout,
                                                    const std::array<uint32_t, 2> & page_shape,
                                                    const std::array<uint32_t, 2> &shard_shape,
                                                    const std::array<uint32_t, 2> & tensor2d_size)
{


    std::vector < std::vector<uint32_t> > ret_vec(num_shards);

    std::array<uint32_t, 2> shard_in_pages = {shard_shape[0]/page_shape[0], shard_shape[1]/page_shape[1]};

    if( layout == TensorMemoryLayout:: HEIGHT_SHARDED) {
        uint32_t num_pages_per_shard = pages_per_shard;
        uint32_t num_pages_per_shard_last = num_pages_per_shard;

        if (total_pages != num_shards * pages_per_shard) {
            num_pages_per_shard_last = total_pages % pages_per_shard;
        }
        uint32_t page_id = 0;
        for(uint32_t i = 0 ; i<num_shards; i++){
            uint32_t num_cols = (i == num_shards - 1) ? num_pages_per_shard_last: num_pages_per_shard;
            ret_vec[i] = std::vector<uint32_t>(num_cols);
            for(uint32_t j = 0; j<num_cols; j++){
                ret_vec[i][j]= page_id++;
            }
        }
    }
    else if(layout == TensorMemoryLayout::WIDTH_SHARDED or layout == TensorMemoryLayout::BLOCK_SHARDED) {
        uint32_t i_offset = 0;
        uint32_t j_offset = 0;
        uint32_t num_shard_columns = div_up(tensor2d_size[1], shard_in_pages[1]);
        uint32_t shard_in_row = 0;

        for(uint32_t shard_idx=0; shard_idx<num_shards; shard_idx++) {
            ret_vec[shard_idx].reserve(pages_per_shard);

            uint32_t host_idx = 0;
            for(uint32_t i=i_offset; i<(shard_in_pages[0] + i_offset); i++) {
                if (i >= tensor2d_size[0]) {
                    break;
                }
                for(uint32_t j=j_offset; j<(shard_in_pages[1] + j_offset) and (j < (tensor2d_size[1])); j++) {
                    uint32_t host_page = i*tensor2d_size[1] + j;
                    ret_vec[shard_idx].push_back(host_page);
                    host_idx++;
                }
            }
            if(((shard_in_row + 1) == (num_shard_columns))) {
                shard_in_row = 0;
                j_offset = 0;
                i_offset += shard_in_pages[0];
            }
            else{
                shard_in_row++;
                j_offset += shard_in_pages[1];
            }
        }


    }
    return ret_vec;
}


Buffer::Buffer(Device *device, uint64_t size, uint64_t page_size, const BufferType buffer_type,
                const TensorMemoryLayout buffer_layout,
                std::optional< ShardSpecBuffer> shard_parameters
                )
    : device_(device), size_(size), page_size_(page_size), buffer_type_(buffer_type), buffer_layout_(buffer_layout), shard_parameters_(shard_parameters) {
    TT_FATAL(this->device_ != nullptr and this->device_->allocator_ != nullptr);
    validate_buffer_size_and_page_size(size, page_size, buffer_type, buffer_layout, shard_parameters);
    if(is_sharded(buffer_layout)){
        auto row_major = shard_parameters.value().orientation() == ShardOrientation::ROW_MAJOR;
        all_cores_ = corerange_to_cores(shard_parameters.value().grid(), this->num_cores(), row_major);
        TT_ASSERT(this->num_cores() == all_cores_.size());
        uint32_t core_id = 0;
        for(auto core: all_cores_){
            this->core_to_core_id_.insert({core, core_id });
            core_id++;
        }
        this->core_host_page_indices_ = core_to_host_pages(this->num_pages(), shard_spec().size(), this->num_cores(), buffer_layout, shard_spec().page_shape, shard_spec().shape(), shard_spec().tensor2d_shape);

        auto total_dev_pages = this->num_pages();
        this->dev_page_to_host_page_mapping_ = std::vector<uint32_t>(total_dev_pages);
        this->dev_page_to_core_mapping_ = std::vector<uint32_t>(total_dev_pages);
        this->host_page_to_local_shard_page_mapping_ = std::vector<uint32_t>(total_dev_pages);
        this->host_page_to_dev_page_mapping_= std::vector<uint32_t>(total_dev_pages);

        int dev_page_index = 0;
        for(uint32_t core_index = 0; core_index < this->core_host_page_indices_.size() ; core_index++){
            for(uint32_t shard_page_id = 0; shard_page_id < this->core_host_page_indices_[core_index].size() ; shard_page_id++){
                auto host_page = this->core_host_page_indices_[core_index][shard_page_id];
                this->dev_page_to_core_mapping_[dev_page_index] = core_index;
                this->dev_page_to_host_page_mapping_[dev_page_index] = host_page;
                TT_ASSERT(host_page < this->host_page_to_local_shard_page_mapping_.size());
                TT_ASSERT(host_page < this->host_page_to_dev_page_mapping_.size());
                this->host_page_to_local_shard_page_mapping_[host_page] = shard_page_id;
                this->host_page_to_dev_page_mapping_[host_page] = dev_page_index;
                dev_page_index++;
            }
        }
    }

    this->allocate();
}

Buffer::Buffer(const Buffer &other) :
    device_(other.device_),
    size_(other.size_),
    page_size_(other.page_size_),
    buffer_type_(other.buffer_type_),
    buffer_layout_(other.buffer_layout_),
    shard_parameters_(other.shard_parameters_) {
    this->allocate();
}

Buffer &Buffer::operator=(const Buffer &other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->size_ = other.size_;
        this->page_size_ = other.page_size_;
        this->buffer_type_ = other.buffer_type_;
        this->buffer_layout_ = other.buffer_layout_;
        this->shard_parameters_ = other.shard_parameters_;
        this->allocate();
    }
    return *this;
}

Buffer::Buffer(Buffer &&other) : device_(other.device_), size_(other.size_), address_(other.address_), page_size_(other.page_size_), buffer_type_(other.buffer_type_) ,
                                    buffer_layout_(other.buffer_layout_), shard_parameters_(other.shard_parameters_) {
    // Set `other.device_` to be nullptr so destroying other does not deallocate reserved address space that is transferred to `this`
    other.device_ = nullptr;
}

Buffer &Buffer::operator=(Buffer &&other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->size_ = other.size_;
        this->address_ = other.address_;
        this->page_size_ = other.page_size_;
        this->buffer_type_ = other.buffer_type_;
        this->buffer_layout_ = other.buffer_layout_;
        this->shard_parameters_ = other.shard_parameters_;
        // Set `other.device_` to be nullptr so destroying other does not deallocate reserved address space that is transferred to `this`
        other.device_ = nullptr;
    }
    return *this;
}

void Buffer::allocate() {
    TT_ASSERT(this->device_ != nullptr);
    // L1 buffers are allocated top down!
    bool bottom_up = this->buffer_type_ == BufferType::DRAM;
    detail::AllocateBuffer(this, bottom_up);
    detail::BUFFER_MAP[{this->device_->id(), this->address_}] = this;
}

uint32_t Buffer::dram_channel_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->buffer_type_ == BufferType::DRAM, "Expected DRAM buffer!");
    return this->device_->dram_channel_from_bank_id(bank_id);
}

CoreCoord Buffer::logical_core_from_bank_id(uint32_t bank_id) const {
    TT_ASSERT(this->buffer_type_ == BufferType::L1, "Expected L1 buffer!");
    return this->device_->logical_core_from_bank_id(bank_id);
}

CoreCoord Buffer::noc_coordinates(uint32_t bank_id) const {
    switch (this->buffer_type_) {
        case BufferType::DRAM: {
            auto dram_channel = this->dram_channel_from_bank_id(bank_id);
            return llrt::get_core_for_dram_channel(dram_channel, this->device_->id());
        }
        case BufferType::L1: {
            auto logical_core = this->logical_core_from_bank_id(bank_id);
            return this->device_->worker_core_from_logical_core(logical_core);
        }
        break;
        case BufferType::SYSTEM_MEMORY: {
            TT_THROW("Host buffer is located in system memory! Cannot retrieve NoC coordinates for it");
        }
        break;
        default:
            TT_ASSERT(false && "Unsupported buffer type!");
    }
    return CoreCoord{0, 0};
}

CoreCoord Buffer::noc_coordinates() const {
    return this->noc_coordinates(0);
}

uint64_t Buffer::page_address(uint32_t bank_id, uint32_t page_index) const {
    auto num_banks = this->device_->num_banks(this->buffer_type_);
    TT_ASSERT(bank_id < num_banks, "Invalid Bank ID: {} exceeds total numbers of banks ({})!", bank_id, num_banks);

    // DRAM readers and writers in Cluster add DRAM bank offset before doing a read but L1 readers and writers do not
    uint64_t base_page_address = this->buffer_type_ == BufferType::DRAM ?
        this->address_ :
        this->address_ + this->device_->l1_bank_offset_from_bank_id(bank_id);

    int pages_offset_within_bank = (int)page_index / num_banks;
    auto offset = (round_up(this->page_size_, ADDRESS_ALIGNMENT) * pages_offset_within_bank);
    return base_page_address + offset;
}

uint64_t Buffer::sharded_page_address(uint32_t bank_id, uint32_t page_index) const {
    TT_ASSERT(is_sharded(this->buffer_layout()));

    // DRAM readers and writers in Cluster add DRAM bank offset before doing a read but L1 readers and writers do not
    uint64_t base_page_address = this->buffer_type_ == BufferType::DRAM ?
        this->address_ :
        this->address_ + this->device_->l1_bank_offset_from_bank_id(bank_id);

    int pages_offset_within_bank = page_index % shard_spec().size();
    auto offset = (round_up(this->page_size_, ADDRESS_ALIGNMENT) * pages_offset_within_bank);
    return base_page_address + offset;
}

void Buffer::deallocate() {
    if (this->device_ == nullptr or not this->device_->initialized_ or this->size_ == 0) {
        return;
    }
    // Mark as deallocated
    this->size_ = 0;
    TT_ASSERT(this->device_->allocator_ != nullptr, "Expected allocator to be initialized!");
    // Asynchronously deallocate
    detail::BUFFER_MAP.erase({this->device_->id(), this->address_});
    detail::DeallocateBuffer(this);
}

Buffer::~Buffer() {
    this->deallocate();
}

tt::stl::reflection::Attributes ShardSpec::attributes() const {
    return {
        {"shard_grid", this->grid.str()},
        {"shard_shape", this->shape},
        {"shard_orientation", this->orientation},
        {"halo", this->halo},
    };
}

bool operator==(const ShardSpec& spec_a, const ShardSpec& spec_b) {
    if (spec_a.grid != spec_b.grid) {
        return false;
    }
    if (spec_a.shape != spec_b.shape) {
        return false;
    }
    if (spec_a.orientation != spec_b.orientation) {
        return false;
    }
    if (spec_a.halo != spec_b.halo) {
        return false;
    }
    return true;
}

bool operator!=(const ShardSpec& spec_a, const ShardSpec& spec_b) {
    return not (spec_a == spec_b);
}

}  // namespace tt_metal
}  // namespace tt
