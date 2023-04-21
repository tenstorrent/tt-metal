#include <errno.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>

#include "common/bfloat16.hpp"
#include "common/logger.hpp"
#include "device/grayskull/device_data.hpp"
#include "dispatch_data_structures.hpp"
#include "llrt.hpp"
#include "tensix.h"
// #include "test_libs/tiles.hpp"
#include "tests/tt_metal/llrt/test_libs/debug_mailbox.hpp"
#include "tt_cluster.hpp"
#include "utils.hpp"

// Just as defined in noc_parameters.h for grayskull
#define NOC_X(x) (x)
#define NOC_Y(y) (y)

#define NOC_ADDR_LOCAL_BITS 32
#define NOC_ADDR_NODE_ID_BITS 6

#define NOC_XY_ADDR(x, y, addr)                                                                                      \
    ((((uint64_t)(y)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | (((uint64_t)(x)) << NOC_ADDR_LOCAL_BITS) | \
     ((uint64_t)(addr)))

#define NOC_MULTICAST_ADDR(x_start, y_start, x_end, y_end, addr)                \
  ((((uint64_t)(x_start)) << (NOC_ADDR_LOCAL_BITS+2*NOC_ADDR_NODE_ID_BITS)) |   \
   (((uint64_t)(y_start)) << (NOC_ADDR_LOCAL_BITS+3*NOC_ADDR_NODE_ID_BITS)) |   \
   (((uint64_t)(x_end))   << NOC_ADDR_LOCAL_BITS) |                             \
   (((uint64_t)(y_end))   << (NOC_ADDR_LOCAL_BITS+NOC_ADDR_NODE_ID_BITS)) |     \
   ((uint64_t)(addr)))

uint32_t ACTIVATIONS_DRAM_SRC = 500 * 1024;
uint32_t NUM_TILES = 4;
uint32_t NUM_BYTES_PER_TILE = 2048;
uint32_t NUM_CB_TILES = 4;

uint32_t nearest_multiple_of_32(
    uint32_t addr);  // If addr is divisible by 32, returns addr, else it returns the next multiple of 32 following addr

void assert_32B_alignment(uint32_t addr);  // Asserts addr is divisible by 32

void assert_valid_dram_data_for_datacopy(
    const DramConfig &data);  // Asserts that all of the DRAM addresses are 32B-aligned and that the kernel/rt args/cb
                              // config data will fit in L1

void write_to_dram(
    tt_cluster *cluster, int chip_id, const DramConfig &data);  // Writes the kernels/rt args/cb configs to DRAM

DramConfig construct_dram_config(string op);  // Creates the DRAM copy descriptor

tuple<uint32_t, uint64_t, uint32_t> create_kernel_transfer_info(
    uint32_t kernel_id, DramConfig &dram_config, uint32_t &l1_src);

tuple<uint32_t, uint64_t, uint32_t> create_rt_args_transfer_info(
    uint32_t kernel_id, DramConfig &dram_config, uint32_t &l1_src);

tuple<uint32_t, uint64_t, uint32_t> create_cb_transfer_info(uint32_t cb_id, uint32_t &l1_src);

CopyDescriptor construct_copy_descriptor_from_dram_config(
    DramConfig &config, uint32_t copy_desc_l1_start, uint32_t read_to_addr);

vector<uint32_t> convert_copy_desc_to_flat_vec(const CopyDescriptor &copy_desc);

void write_copy_desc_to_l1(tt_cluster *cluster, int chip_id, tt_xy_pair dispatch_core, const CopyDescriptor &copy_desc);
