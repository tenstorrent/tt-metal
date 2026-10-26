#!/usr/bin/env python3
"""mk_cpp.py ROUTER_CPP: the grouped free-slots read through NOC_STREAM_READ_REG (the stream register API) instead of a
raw pointer to the same register."""
import sys
p = sys.argv[1]
s = open(p).read()
a = """template <size_t CH>
constexpr uint32_t free_slots_reg_addr() {
    return STREAM_REG_ADDR(
        sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(CH)],
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

template <size_t CH>
FORCE_INLINE uint32_t read_free_slots_raw() {
    return *reinterpret_cast<volatile uint32_t*>(free_slots_reg_addr<CH>());
}
"""
b = """// Channel CH's free-slots register, unmasked (the caller masks; get_ptr_val would mask at once).
template <size_t CH>
FORCE_INLINE uint32_t read_free_slots_raw() {
    return NOC_STREAM_READ_REG(
        sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(CH)],
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}
"""
assert s.count(a) == 1, s.count(a)
s = s.replace(a, b)
open(p, "w").write(s)
print("ok")
