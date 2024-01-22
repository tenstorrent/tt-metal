// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/common/math.hpp"
#include "common/assert.hpp"
#include "tt_metal/impl/device/device.hpp"

#include "llrt/llrt.hpp"

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
                                                    const int & pages_per_shard,
                                                    const int & num_shards,
                                                    const TensorMemoryLayout & layout,
                                                    const std::array<uint32_t, 2> & page_shape,
                                                    const std::array<uint32_t, 2> &shard_shape,
                                                    const std::array<uint32_t, 2> & tensor2d_size)
{


    auto total_pages = pages_per_shard * num_shards;
    std::vector < std::vector<uint32_t> > ret_vec(num_shards, std::vector<uint32_t>(pages_per_shard));


    std::array<uint32_t, 2> shard_in_pages = {shard_shape[0]/page_shape[0], shard_shape[1]/page_shape[1]};
    if( layout == TensorMemoryLayout::WIDTH_SHARDED){
        int page_id  = 0;
        for(int i =0 ; i<num_shards; i++){
            for(int j=0; j<pages_per_shard; j++){
                ret_vec[i][j]= i*pages_per_shard + j;
            }
        }
    }
    else if( layout == TensorMemoryLayout:: HEIGHT_SHARDED){
        int page_id = 0;
        for(int i =0 ; i<num_shards; i++){
            for(int j=0; j<pages_per_shard; j++){
                ret_vec[i][j]= page_id++;
            }
        }
    }
    else if(layout == TensorMemoryLayout::BLOCK_SHARDED){
        int i_offset = 0;
        int j_offset = 0;

        for(int shard_idx=0; shard_idx<num_shards; shard_idx++){
            int host_idx = 0;
            for(int i=i_offset; i<(shard_in_pages[1] + i_offset); i++){
                //Shard reached end of row
                for(int j=j_offset; j<(shard_in_pages[0] + j_offset); j++){
                    auto host_page = i*tensor2d_size[0] + j;
                    ret_vec[shard_idx][host_idx] = host_page;
                    host_idx++;
                }
            }
            if(((shard_idx + 1) % (tensor2d_size[0]/shard_in_pages[0])) == 0){
                j_offset = 0;
                i_offset += shard_in_pages[1];
            }
            else{
                j_offset += shard_in_pages[0];
            }
        }


    }
    return ret_vec;
}


//#define DEBUG_SHARD_PRINT
std::string Buffer::get_shard_info() const {
    std::string ret_str = "Shard info for buffer \n";

    auto sspec = shard_spec();
    ret_str += "Page Size " + std::to_string(page_size_) + "\n";

    auto t2d_size = sspec.tensor2d_shape;
    ret_str += "Tensor2D Size: ("  + std::to_string(t2d_size[0]) + ", " + std::to_string(t2d_size[1]) + ")\n";
    auto s_shape = sspec.shape();
    ret_str += "Shard Shape: ("  + std::to_string(s_shape[0]) +  ", " + std::to_string(s_shape[1]) + ")\n"; ;

    auto p_shape = sspec.page_shape;
    ret_str += "Page Shape: ("  + std::to_string(p_shape[0]) +  ", " + std::to_string(p_shape[1]) + ")\n"; ;


    uint32_t core_index = 0;
    ret_str += "Core info:\n";
    for(auto core: all_cores_){
        ret_str += "Core " + core.str()  + "\n";
        ret_str += "Host pages on core: ";
        for(auto host_page_id: core_host_page_indices_[core_index]){
            ret_str+= std::to_string(host_page_id) + " ";
        }
        ret_str += "\n";
        ret_str += "Bank id for core: " + std::to_string(core_bank_indices_[core_index]) +  "\n";
        core_index++;
    }
    ret_str += "Dev page mappings:\n";;
    uint32_t num_pages = all_cores_.size() * sspec.size();
    for(uint32_t dev_page_id = 0; dev_page_id < num_pages; dev_page_id ++){
        ret_str += "Dev page: " + std::to_string(dev_page_id) +
            " mapped to core " + all_cores_[dev_page_to_core_mapping_[dev_page_id]].str() +
            " and host page " + std::to_string(dev_page_to_host_page_mapping_[dev_page_id]) + "\n";
    }
    return ret_str;

}

void Buffer::print_shard_info() const {
    std::cout << get_shard_info() << std::endl;
}

void Buffer::log_shard_info() const {

    auto shard_str = get_shard_info();
    log_info(LogTest, shard_str.c_str());

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
        core_host_page_indices_ = core_to_host_pages(shard_spec().size(), this->num_cores(), buffer_layout, shard_spec().page_shape, shard_spec().shape(), shard_spec().tensor2d_shape);
        core_bank_indices_.reserve(this->num_cores());

        auto total_dev_pages = this->num_cores() * shard_spec().size();
        dev_page_to_host_page_mapping_ = std::vector<uint32_t>(total_dev_pages);
        dev_page_to_core_mapping_ = std::vector<uint32_t>(total_dev_pages);
        for(auto core : all_cores_){
            core_bank_indices_.push_back(device->bank_ids_from_logical_core(core)[0]);
        }
        int dev_page_index = 0;
        for(int core_index = 0; core_index < core_host_page_indices_.size() ; core_index++){
            for(int host_page_index = 0; host_page_index < core_host_page_indices_[core_index].size() ; host_page_index++){
                dev_page_to_core_mapping_[dev_page_index] = core_index;
                dev_page_to_host_page_mapping_[dev_page_index] = core_host_page_indices_[core_index][host_page_index];
                dev_page_index++;
            }
        }
    }

    #ifdef DEBUG_SHARD_PRINT
        if(shard_parameters.has_value()){
            TT_ASSERT(is_sharded(buffer_layout));
            this->print_shard_info();
            //this->log_shard_info();
        }
    #endif
    this->allocate();
}

Buffer::Buffer(const Buffer &other)
    : device_(other.device_), size_(other.size_), page_size_(other.page_size_),
        buffer_type_(other.buffer_type_) , buffer_layout_(other.buffer_layout_), shard_parameters_(other.shard_parameters_){
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
    if(is_sharded(this->buffer_layout_)){
        this->address_ = allocator::allocate_buffer(*this->device_->allocator_, this->size_,
                                                this->page_size_, this->buffer_type_, bottom_up,
                                                this->num_cores());
    }
    else{
        this->address_ = allocator::allocate_buffer(*this->device_->allocator_, this->size_, this->page_size_, this->buffer_type_, bottom_up, std::nullopt);
    }

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
    return CoreCoord{.x=0, .y=0};
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

uint64_t Buffer::page_address(uint32_t page_index) const {
    TT_ASSERT(is_sharded(this->buffer_layout()));

    auto bank_id = this->get_bank_id_from_page_id(page_index);

    // DRAM readers and writers in Cluster add DRAM bank offset before doing a read but L1 readers and writers do not
    uint64_t base_page_address = this->buffer_type_ == BufferType::DRAM ?
        this->address_ :
        this->address_ + this->device_->l1_bank_offset_from_bank_id(bank_id);

    int pages_offset_within_bank = page_index % shard_spec().size();
    auto offset = (round_up(this->page_size_, ADDRESS_ALIGNMENT) * pages_offset_within_bank);
    return base_page_address + offset;
}

uint64_t Buffer::core_address(uint32_t core_id) const {
    TT_ASSERT(is_sharded(this->buffer_layout()));
    auto bank_id = this->core_bank_indices_[core_id];
    auto first_page = this->core_host_page_indices_[core_id][0];
    auto page_id = this->dev_page_to_host_page_mapping_[first_page];
    return this->page_address(bank_id, page_id);
}

void Buffer::deallocate() {
    if (this->device_ == nullptr or not this->device_->initialized_ or this->size_ == 0) {
        return;
    }
    this->size_ = 0;
    TT_ASSERT(this->device_->allocator_ != nullptr, "Expected allocator to be initialized!");
    allocator::deallocate_buffer(*this->device_->allocator_, this->address_, this->buffer_type_);
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
    if (spec_a.shape != spec_b.shape) {
        return false;
    }
    if (spec_a.grid != spec_b.grid) {
        return false;
    }
    if (spec_a.orientation != spec_b.orientation) {
        return false;
    }
    return true;
}

}  // namespace tt_metal

}  // namespace tt
