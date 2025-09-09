
#include <cstdint>

namespace tt::tt_metal {

// Multiple things in dispatch assume that go_msg_t is 4B and has the same layout for all core types.
// This assumption is currently valid, and probably will always hold, but it is not guaranteed by HAL interface.
// We consolidate all go_msg_t creations in dispatch here.
uint32_t go_msg_u32_value(uint8_t signal, uint8_t master_x, uint8_t master_y, uint8_t dispatch_message_offset);

}  // namespace tt::tt_metal
