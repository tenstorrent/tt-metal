
#ifdef DEBUG
#define DEBUG_LOG(str) do { std::cout << str << std::endl; } while( false )
#else
#define DEBUG_LOG(str) do { } while ( false )
#endif

#include "tt_device.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "yaml-cpp/yaml.h"
#include "common/assert.hpp"

//
// Helper functions
//
std::string format_node(CoreCoord xy) { return std::to_string(xy.x) + "-" + std::to_string(xy.y); }

CoreCoord format_node(std::string str) {
  int x_coord;
  int y_coord;
  std::regex expr("([0-9]+)[-,xX]([0-9]+)");
  std::smatch x_y_pair;

  if (std::regex_search(str, x_y_pair, expr)) {
    x_coord = std::stoi(x_y_pair[1]);
    y_coord = std::stoi(x_y_pair[2]);
  } else {
    throw std::runtime_error("Could not parse the core id: " + str);
  }

  CoreCoord xy(x_coord, y_coord);

  return xy;
}

////////
// Device base
////////

tt_device::tt_device(std::unordered_map<chip_id_t, tt_SocDescriptor> soc_descriptor_per_chip_) : soc_descriptor(soc_descriptor_per_chip_.begin()->second), soc_descriptor_per_chip(soc_descriptor_per_chip_) {
}

tt_device::tt_device(const tt_SocDescriptor &soc_descriptor_) : soc_descriptor(soc_descriptor_), soc_descriptor_per_chip({}) {
}

tt_device::~tt_device() {
}

void tt_device::start(std::vector<std::string> plusargs,std::vector<std::string> dump_cores, bool no_checkers, bool init_device, bool skip_driver_allocs)
{
    // To be redefined with each derived device type
}

void tt_device::deassert_risc_reset(bool start_stagger) {
    // To be redefined with each derived device type
}
void tt_device::assert_risc_reset() {
    // To be redefined with each derived device type
}
bool tt_device::stop(){
    // To be redefined with each derived device type
    return true;
}

void tt_device::write_vector(std::vector<std::uint32_t> &mem_vector, tt_cxy_pair target, std::uint32_t address, bool host_resident, bool small_access, chip_id_t src_device_id)
{
     // To be redefined with each derived device type
     assert(false && "Should not be here");
}

void tt_device::write_vector(const std::uint32_t *mem_ptr, uint32_t len, tt_cxy_pair target, std::uint32_t address, bool host_resident, bool small_access, chip_id_t src_device_id)
{
    // To be redefined with each derived device type
    assert(false && "Should not be here");
}

void tt_device::read_vector(std::vector<std::uint32_t> &mem_vector, tt_cxy_pair target, std::uint32_t address, std::uint32_t size_in_bytes, bool host_resident, bool small_access, chip_id_t src_device_id)
{
    // To be redefined with each derived device type
    assert(false && "Should not be here");
}

void tt_device::read_vector(std::uint32_t *mem_ptr, tt_cxy_pair target, std::uint32_t address, std::uint32_t size_in_bytes, bool host_resident, bool small_access, chip_id_t src_device_id)
{
    // To be redefined with each derived device type
    assert(false && "Should not be here");
}

uint32_t tt_device::dma_allocation_size(chip_id_t src_device_id)
{
  return 0;
}

uint32_t * tt_device::get_usr_ptr(uint32_t d_addr, chip_id_t src_device_id) {
  return nullptr;
}

bool tt_device::wait_for_completion() {
    // To be redefined with each derived device type
    return true;
}

const tt_SocDescriptor *tt_device::get_soc_descriptor() const { return &soc_descriptor; }

void tt_device::dump_debug_mailbox(std::string output_path, int device_id) {
  // To be redefined with each derived device type
}

void tt_device::dump_wall_clock_mailbox(std::string output_path, int device_id) {
  // To be redefined with each derived device type
}

bool tt_device::test_write_read(tt_cxy_pair target)
{
    std::vector<uint32_t> test_vector1(30, 0xDEADBEEF);
    std::vector<uint32_t> test_vector2(30, 0xDEAD0000);
    this->write_vector(test_vector1, target, 512 * 1024);
    this->write_vector(test_vector2, target, 512 * 1024 + 120);
    std::vector<uint32_t> read_data;
    read_data.resize(30);
    this->read_vector(read_data, target, 512 * 1024, 240);
    bool result = true;
    for (int i = 0; i < test_vector1.size(); i++) {
        if (test_vector1[i] != read_data[i]) {
            result = false;
            std::cout << "Error: Mismatch in " << i << "th index -- expected: 0x" << std::hex << test_vector1[i]
                        << " got: 0x" << read_data[i] << std::dec << std::endl;
        } else {
            std::cout << "Data OK: " << i << "th index -- expected: 0x" << std::hex << test_vector1[i]
                        << " got: 0x" << read_data[i] << std::dec << std::endl;
        }
    }
    for (int i = 0; i < test_vector2.size(); i++) {
        if (test_vector2[i] != read_data[i + test_vector1.size()]) {
            result = false;
            std::cout << "Error: Mismatch in " << i << "th index -- expected: 0x" << std::hex << test_vector2[i]
                        << " got: 0x" << read_data[i + test_vector1.size()] << std::dec << std::endl;
        } else {
            std::cout << "Data OK: " << i << "th index -- expected: 0x" << std::hex << test_vector2[i]
                        << " got: 0x" << read_data[i + test_vector1.size()] << std::dec << std::endl;
        }
    }
    return result;
}

bool tt_device::get_dma_buffer(void **mapping, std::uint64_t *physical, std::size_t *size, chip_id_t src_device_id) const
{
    return false;
}

void *tt_device::channel_0_address(std::uint32_t offset, std::uint32_t device_id) const
{
    return nullptr;
}

void *tt_device::host_dma_address(std::uint64_t offset, chip_id_t src_device_id) const
{
    return nullptr;
}
