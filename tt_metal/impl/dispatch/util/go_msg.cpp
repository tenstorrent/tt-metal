#include <cstdint>
#include <mutex>

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

uint32_t go_msg_u32_value(uint8_t signal, uint8_t master_x, uint8_t master_y, uint8_t dispatch_message_offset) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    // Because many things in dispatch assume that go_msg_t is 4B and has the same layout for all core types,
    // we do a basic check that go_msg_t is indeed 4B for all core types.
    std::once_flag once_flag;
    std::call_once(once_flag, [&]() {
        for (uint32_t programmable_core_type_index = 0;
             programmable_core_type_index < tt::tt_metal::NumHalProgrammableCoreTypes;
             ++programmable_core_type_index) {
            auto factory = hal.get_dev_msgs_factory(hal.get_programmable_core_type(programmable_core_type_index));
            TT_FATAL(
                factory.size_of<dev_msgs::go_msg_t>() == sizeof(uint32_t),
                "go_msg_t size must be 4 bytes for all programmable core types.");
        }
    });
    // Note: because of the assumption, we can use the dev_msgs factory for whatever core type.
    uint32_t go_msg_u32_val = 0;
    auto dev_msgs_factory =
        tt::tt_metal::MetalContext::instance().hal().get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    auto go_msg = dev_msgs_factory.create_view<dev_msgs::go_msg_t>(reinterpret_cast<std::byte*>(&go_msg_u32_val));
    go_msg.signal() = signal;
    go_msg.master_x() = master_x;
    go_msg.master_y() = master_y;
    go_msg.dispatch_message_offset() = dispatch_message_offset;
    return go_msg_u32_val;
}

}  // namespace tt::tt_metal
