#include "test_buffer_utils.hpp"

namespace tt::test::buffer::detail {
void writeL1Backdoor(Device* device, CoreCoord coord, uint32_t address, std::vector<uint32_t>& data) {
    tt::log_info("{} -- coord={} address={}", __FUNCTION__, coord.str(), address);
    tt_metal::WriteToDeviceL1(device, coord, address, data);
}
void readL1Backdoor(
    Device* device, CoreCoord coord, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data) {
    tt::log_info("{} -- coord={} address={} byte_size={}", __FUNCTION__, coord.str(), address, byte_size);
    tt_metal::ReadFromDeviceL1(device, coord, address, byte_size, data);
}
void writeDramBackdoor(Device* device, uint32_t channel, uint32_t address, std::vector<uint32_t>& data) {
    tt::log_info("{} -- channel={} address={}", __FUNCTION__, channel, address);
    tt_metal::WriteToDeviceDRAMChannel(device, channel, address, data);
}
void readDramBackdoor(
    Device* device, uint32_t channel, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& data) {
    tt::log_info("{} -- channel={} address={} byte_size={}", __FUNCTION__, channel, address, byte_size);
    tt_metal::ReadFromDeviceDRAMChannel(device, channel, address, byte_size, data);
}
}  // namespace tt::test::buffer::detail
