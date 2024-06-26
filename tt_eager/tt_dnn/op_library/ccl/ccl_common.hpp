// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <numeric>

#include "common/constants.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/ccl_host_datastructures.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program/program.hpp"

namespace tt {
namespace tt_metal {
namespace ccl {

// Eventual home: ccl_topology_descriptors
struct RingTopology {
    RingTopology(
        Device const* device,
        Topology topology,
        std::optional<uint32_t> sender_device_id,
        std::optional<uint32_t> receiver_device_id,
        uint32_t num_links,
        uint32_t ring_size,
        uint32_t ring_index) :
        num_links(num_links), ring_size(ring_size), ring_index(ring_index), is_linear(topology == Topology::Linear) {
        eth_sender_cores.reserve(num_links);
        eth_receiver_cores.reserve(num_links);

        uint32_t sender_socket_idx = 0;
        uint32_t receiver_socket_idx = 0;
        if (receiver_device_id == sender_device_id) {
            if (ring_index == 0) {
                receiver_socket_idx = 1;
            } else {
                sender_socket_idx = 1;
            }
        }

        for (uint32_t l = 0; l < num_links; ++l) {
            // Get the cores for the sender and receiver worker cores
            if (!is_linear || ring_index != ring_size - 1) {
                uint32_t receiver_device = receiver_device_id.value();
                auto const& sockets = device->get_ethernet_sockets(receiver_device);
                auto eth_sender_core = sockets.at(sender_socket_idx);
                eth_sender_cores.push_back(eth_sender_core);
                log_trace(
                    tt::LogOp, "\teth_sender_core on link {}: (x={},y={})", l, eth_sender_core.x, eth_sender_core.y);
            }
            if (!is_linear || ring_index != 0) {
                uint32_t sender_device = sender_device_id.value();
                auto const& sockets = device->get_ethernet_sockets(sender_device);
                auto eth_receiver_core = sockets.at(receiver_socket_idx);
                eth_receiver_cores.push_back(eth_receiver_core);
                log_trace(
                    tt::LogOp,
                    "\teth_receiver_core on link {}: (x={},y={})",
                    l,
                    eth_receiver_core.x,
                    eth_receiver_core.y);
            }

            if (receiver_device_id == sender_device_id) {
                receiver_socket_idx += 2;
                sender_socket_idx += 2;
            } else {
                receiver_socket_idx += 1;
                sender_socket_idx += 1;
            }
        }
    }

    std::vector<CoreCoord> eth_sender_cores;
    std::vector<CoreCoord> eth_receiver_cores;

    uint32_t num_links;
    uint32_t ring_size;
    uint32_t ring_index;
    bool is_linear;
};

class CclOpTensorConfig {
   public:
    static std::unique_ptr<CclOpTensorConfig> build_all_gather_tensor_config(Tensor const& tensor);

    CclOpTensorConfig(Tensor const& tensor) :
        buffer_start_address(tensor.buffer()->address()),
        df(tt_metal::datatype_to_dataformat_converter(tensor.get_dtype())) {}

    virtual uint32_t get_page_size() const = 0;
    virtual uint32_t get_unit_size() const = 0;

    uint32_t get_buffer_start_address() const { return this->buffer_start_address; }

    virtual ~CclOpTensorConfig() {};

   protected:
    uint32_t buffer_start_address;
    DataFormat df;
};

class CclOpInterleavedTensorConfig final : public virtual CclOpTensorConfig {
   public:
    CclOpInterleavedTensorConfig(Tensor const& input_tensor) : CclOpTensorConfig(input_tensor) {
        if (input_tensor.get_layout() == Layout::TILE) {
            this->page_size = tt_metal::detail::TileSize(this->df);
        } else {
            this->page_size = input_tensor.buffer()->page_size();
        }
    }
    virtual uint32_t get_page_size() const override { return this->page_size; }
    virtual uint32_t get_unit_size() const override { return this->page_size; }

   private:
    uint32_t page_size;
};

class CclOpShardedTensorConfig final : public virtual CclOpTensorConfig {
   public:
    CclOpShardedTensorConfig(Tensor const& tensor) :
        CclOpTensorConfig(tensor), shard_spec(tensor.shard_spec().value()) {
        if (tensor.get_layout() == Layout::TILE) {
            this->page_size = tt_metal::detail::TileSize(this->df);
            TT_ASSERT(
                this->shard_spec.shape.at(0) * this->shard_spec.shape.at(1) %
                    (constants::TILE_HEIGHT * constants::TILE_WIDTH) ==
                0);
            this->unit_size = (this->shard_spec.shape.at(0) * this->shard_spec.shape.at(1) /
                               (constants::TILE_HEIGHT * constants::TILE_WIDTH)) *
                              this->page_size;
        } else {
            this->page_size = tensor.get_legacy_shape()[-1] * tensor.element_size();
            this->unit_size = (this->page_size * this->shard_spec.shape.at(0) * this->shard_spec.shape.at(1)) /
                              tensor.shard_spec()->num_cores();
        }
    }

    virtual uint32_t get_page_size() const override { return this->page_size; }
    virtual uint32_t get_unit_size() const override { return this->unit_size; }

    uint32_t get_shard_size_in_bytes() const { return this->get_unit_size(); }

    ShardSpec const& get_shard_spec() const { return this->shard_spec; }

   private:
    uint32_t page_size;
    uint32_t unit_size;
    ShardSpec const shard_spec;
};

struct CclTensorSlicer {
    CclTensorSlicer(
        Shape tensor_shape,
        Shape dim_slice_factors,
        // Shape page_shape,
        std::size_t num_pages,
        std::size_t elem_size,
        std::size_t page_size_in_bytes) :
        tensor_shape(tensor_shape),
        dim_slice_factors_per_rank(dim_slice_factors),
        // page_shape(page_shape),
        num_pages(num_pages),
        page_size_in_bytes(page_size_in_bytes),
        elem_size(elem_size) {
        TT_ASSERT(
            tensor_shape.rank() == dim_slice_factors.rank(),
            "Tensor shape and dim slice factors must have the same size");
        TT_ASSERT(
            std::all_of(dim_slice_factors.begin(), dim_slice_factors.end(), [](uint32_t factor) { return factor > 0; }),
            "All factors must be greater than 0");
    }

    std::size_t get_num_pages_per_slice() const {
        std::size_t n = std::accumulate(
            dim_slice_factors_per_rank.begin(), dim_slice_factors_per_rank.end(), 1, std::multiplies<uint32_t>());
        for (uint32_t i = 0; i < (tensor_shape.rank() - dim_slice_factors_per_rank.rank()); ++i) {
            n *= tensor_shape[i];
        }
        return n;
    }

    Shape const tensor_shape;
    Shape const dim_slice_factors_per_rank;
    // Shape const page_shape;
    std::size_t const num_pages;

    // Shape rank_slice_shape;

    std::size_t const page_size_in_bytes;
    std::size_t const elem_size;
};

// To be replaced by the CclTensorSlicer class, which should be reusable between sharded and interleaved
// specs and also provides a simpler interface to reason about
struct LegacyCclTensorSlicer {
    LegacyCclTensorSlicer() :
        input_page_size(0),
        num_rows(0),
        num_cols(0),
        row_offset(0),
        col_offset(0),
        num_tiles(0),
        input_start_page_idx(0),
        output_addr_offset(0),
        col_idx(0),
        row_idx(0),
        output_page_offset(0),
        output_start_page_idx(0),
        output_start_addr_offset(0),
        row_major(false),
        slice_dim_is_width(false),
        is_sharded(false) {}

    LegacyCclTensorSlicer(
        uint32_t input_page_size,
        uint32_t num_rows,
        uint32_t num_cols,
        uint32_t row_offset,
        uint32_t col_offset,
        uint32_t num_tiles,
        uint32_t input_start_page_idx,
        uint32_t output_addr_offset,
        uint32_t col_idx,
        uint32_t row_idx,
        uint32_t output_page_offset,
        uint32_t output_start_page_idx,
        uint32_t output_start_addr_offset,
        bool row_major,
        bool slice_dim_is_width,
        bool is_sharded) :
        input_page_size(input_page_size),
        num_rows(num_rows),
        num_cols(num_cols),
        row_offset(row_offset),
        col_offset(col_offset),
        num_tiles(num_tiles),
        input_start_page_idx(input_start_page_idx),
        output_addr_offset(output_addr_offset),
        col_idx(col_idx),
        row_idx(row_idx),
        output_page_offset(output_page_offset),
        output_start_page_idx(output_start_page_idx),
        output_start_addr_offset(output_start_addr_offset),
        row_major(row_major),
        slice_dim_is_width(slice_dim_is_width),
        is_sharded(is_sharded) {}

    virtual void increment(uint32_t num_pages) = 0;

    uint32_t input_page_size;
    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t row_offset;
    uint32_t col_offset;
    uint32_t num_tiles;
    uint32_t input_start_page_idx;
    uint32_t output_addr_offset;
    uint32_t col_idx;
    uint32_t row_idx;
    uint32_t output_page_offset;
    uint32_t output_start_page_idx;
    uint32_t output_start_addr_offset;
    bool row_major;
    bool slice_dim_is_width;
    bool is_sharded;
};

// Uniform Tensor Worker Slice
struct InterleavedTensorWorkerSlice {
    InterleavedTensorWorkerSlice(
        tt_xy_pair const& tensor_shape,  // Don't _really_ need this
        tt_xy_pair const& tensor_slice_shape,
        tt_xy_pair const& worker_slice_shape,
        tt_xy_pair const& worker_slice_offset) :
        tensor_shape(tensor_shape),
        tensor_slice_shape(tensor_slice_shape),
        worker_slice_shape(worker_slice_shape),
        worker_slice_offset(worker_slice_offset) {}

    // Could probably be solved in some closed form
    std::size_t compute_num_worker_slice_iterations(std::size_t num_workers) const {
        auto slice_offset = coord_t(worker_slice_offset.x, worker_slice_offset.y);
        auto const& slice_shape = coord_t(worker_slice_shape.x, worker_slice_shape.y);
        auto const& outer_slice_shape = coord_t(tensor_slice_shape.x, tensor_slice_shape.y);
        uint32_t num_iterations = 0;
        while (slice_offset.y < tensor_slice_shape.y && slice_offset.x < tensor_slice_shape.x) {
            slice_offset =
                tt::tt_metal::ccl::advance_slice_row_major(slice_offset, slice_shape, outer_slice_shape, num_workers);
            num_iterations++;
        }

        return num_iterations;
    }

    std::size_t get_worker_slice_num_pages() const {
        return worker_slice_shape.x * worker_slice_shape.y;
    }

    tt_xy_pair tensor_shape;
    tt_xy_pair tensor_slice_shape;
    tt_xy_pair worker_slice_shape;
    tt_xy_pair worker_slice_offset;
};

class RingReduceScatterTensorSlicer : public LegacyCclTensorSlicer {
   public:
    RingReduceScatterTensorSlicer(
        Tensor const& input_tensor,
        Tensor const& output_tensor,
        int slice_dim,
        uint32_t ring_index,
        uint32_t ring_size,
        uint32_t total_num_workers,
        uint32_t max_slice_size_in_bytes,
        uint32_t half_cb_n_pages);

    ccl::InterleavedTensorWorkerSlice get_worker_slice(std::size_t global_worker_index) {
        return ccl::InterleavedTensorWorkerSlice(
            this->flattened_tensor_shape,
            this->tensor_slice_shape,
            this->worker_slice_shapes.at(global_worker_index),
            this->worker_slice_offsets.at(global_worker_index));
    }

    [[deprecated("deprecated code path for reduce scatter. Use nerw get_worker_slice API instead")]]
    virtual void increment(uint32_t num_pages) override {
        TT_FATAL(false, "deprecated code path for ");
    }

   public:
    std::vector<tt_xy_pair> get_worker_slice_shapes() const { return this->worker_slice_shapes; }
    static std::vector<tt_xy_pair> compute_worker_slice_offsets(
        std::vector<tt_xy_pair> const& worker_slice_shapes, tt_xy_pair const& tensor_slice_shape);

    static std::vector<tt_xy_pair> create_worker_slice_shapes_for_row_major_layout(
        tt_xy_pair const& tensor_slice_shape_in_elems, uint32_t num_workers, uint32_t max_slice_size_in_elements);

    std::vector<tt_xy_pair> create_worker_slice_shapes_for_tile_layout(
        Shape const& tensor_shape,
        tt_xy_pair const& tensor_slice_shape_in_tiles,
        uint32_t num_workers,
        uint32_t max_slice_size_in_pages,
        uint32_t half_cb_n_pages);

    void create_worker_slice_shape_for_row_major_layout(tt_xy_pair const& tensor_slice_shape, uint32_t num_workers) {
        TT_FATAL("Row major interleaved not supported by Reduce Scatter");
    }

   protected:
    tt_xy_pair flattened_tensor_shape;
    tt_xy_pair tensor_slice_shape;
    std::vector<tt_xy_pair> worker_slice_shapes;
    // For RowMajor - offset is in elements
    // For Tile - offset is in tiles
    std::vector<tt_xy_pair> worker_slice_offsets;
};

class InterleavedRingAllGatherTensorSlicer : public LegacyCclTensorSlicer {
   public:
    InterleavedRingAllGatherTensorSlicer(
        Tensor const& input_tensor, Tensor const& output_tensor, int slice_dim, uint32_t slice_idx) :
        LegacyCclTensorSlicer() {
        this->row_major = input_tensor.get_layout() == Layout::ROW_MAJOR;
        this->slice_dim_is_width = input_tensor.get_legacy_shape().rank() - 1 == slice_dim;
        this->is_sharded = input_tensor.is_sharded();

        int32_t shard_size_in_bytes =
            is_sharded ? (input_tensor.buffer()->page_size() * input_tensor.buffer()->shard_spec().tensor2d_shape[0] *
                          input_tensor.buffer()->shard_spec().tensor2d_shape[1]) /
                             input_tensor.shard_spec()->num_cores()
                       : -1;
        this->input_page_size = is_sharded ? shard_size_in_bytes : input_tensor.buffer()->page_size();
        ;
        if (row_major) {
            this->num_cols = input_tensor.get_legacy_shape()[-1];
            auto input_shape = input_tensor.get_legacy_shape();
            auto output_shape = output_tensor.get_legacy_shape();
            this->num_rows =
                std::accumulate(input_shape.begin() + slice_dim, input_shape.end() - 1, 1, std::multiplies<uint32_t>());
            this->row_offset =
                std::accumulate(
                    output_shape.begin() + slice_dim, output_shape.end() - 1, 1, std::multiplies<uint32_t>()) -
                num_rows;
        } else {
            this->num_cols = input_tensor.get_legacy_shape()[-1] / tt::constants::TILE_WIDTH;
            auto input_shape = input_tensor.get_legacy_shape();
            auto output_shape = output_tensor.get_legacy_shape();
            uint32_t num_output_cols = output_tensor.get_legacy_shape()[-1] / tt::constants::TILE_WIDTH;
            this->num_rows =
                std::accumulate(
                    input_shape.begin() + slice_dim, input_shape.end() - 1, 1, std::multiplies<uint32_t>()) /
                tt::constants::TILE_HEIGHT;
            this->row_offset =
                (std::accumulate(
                     output_shape.begin() + slice_dim, output_shape.end() - 1, 1, std::multiplies<uint32_t>()) /
                     tt::constants::TILE_HEIGHT -
                 num_rows) *
                num_output_cols;
            this->col_offset = num_output_cols - num_cols;
            this->num_tiles = num_rows * num_cols;
        }

        if (row_major) {
            if (slice_dim_is_width) {
                this->output_addr_offset = input_page_size;
            } else {
                this->output_page_offset = num_rows;
            }
        } else {
            if (slice_dim_is_width) {
                this->output_page_offset = num_cols;
            } else {
                this->output_page_offset = num_tiles;
            }
        }
        this->output_start_page_idx = slice_idx /*ring_index*/ * output_page_offset;
        this->output_start_addr_offset = slice_idx /*ring_index*/ * output_addr_offset;
    }

    virtual void increment(uint32_t num_pages) override {
        if (is_sharded) {
            // nothing to do here - is handled by
        } else {
            // Only for interleaved
            if (num_pages /*pages_per_worker*/ > 0) {
                if (row_major) {
                    uint32_t num_rows_shifted = row_idx + num_pages /*pages_per_worker*/;
                    uint32_t num_blocks_shifted = slice_dim_is_width ? 0 : num_rows_shifted / num_rows;
                    this->output_start_page_idx += num_pages /*pages_per_worker*/ + num_blocks_shifted * row_offset;
                    this->row_idx = slice_dim_is_width ? 0 : num_rows_shifted % num_rows;
                } else {
                    uint32_t num_cols_shifted = col_idx + num_pages /*pages_per_worker*/;
                    uint32_t num_rows_shifted = num_cols_shifted / num_cols;
                    uint32_t num_blocks_shifted = slice_dim_is_width ? 0 : num_rows_shifted / num_rows;
                    this->output_start_page_idx += num_pages /*pages_per_worker*/ + num_rows_shifted * col_offset +
                                                   num_blocks_shifted * row_offset;
                    this->col_idx = num_cols_shifted % num_cols;
                    this->row_idx = slice_dim_is_width ? 0 : num_rows_shifted % num_rows;
                }
            }
            this->input_start_page_idx += num_pages /*pages_per_worker*/;
        }
    }
};

KernelHandle generate_edm_kernel(
    tt_metal::Program& program,
    Device const* device,
    ccl::EriscDatamoverBuilder const& edm_builder,
    CoreCoord const& eth_core,
    NOC noc_id);

void generate_edm_kernels_for_ring_or_linear_topology(
    tt_metal::Program& program,
    Device const* device,
    RingTopology const& topology_config,
    std::vector<ccl::EriscDatamoverBuilder> const& clockwise_edm_builders,
    std::vector<ccl::EriscDatamoverBuilder> const& counter_clockwise_edm_builders,
    std::optional<uint32_t> receiver_device_id,
    std::optional<uint32_t> sender_device_id);

ccl::EriscDatamoverBuilder create_erisc_datamover_builder(
    std::size_t num_channels,
    uint32_t page_size,
    ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode,
    EriscDataMoverTerminationMode termination_mode);

}  // namespace ccl
}  // namespace tt_metal
}  // namespace tt
