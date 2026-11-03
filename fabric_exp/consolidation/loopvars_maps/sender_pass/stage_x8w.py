"""stage_x8w.py: v4txo8w = v4txo8 + the sender's worker interfaces built only where the ERISC serves sender channels
(start_worker_interfaces; coordinator's cleanup item 2026-11-03: a receiver-only E1 carried the 336 B storage and its
construction). Expected: every E0 image identical; receiver-only E1 images lose the storage."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo8", f"{V}/v4txo8w"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp"
s = open(p).read()
old = """// Builds the interfaces at their location (once, at init); the kernel then sets each one up.
FORCE_INLINE WorkerInterfaces& start_worker_interfaces() {
    return *new (worker_interface_storage.bytes) WorkerInterfaces{};
}"""
new = """// Builds the interfaces at their location (once, at init) where this ERISC serves sender channels; the kernel then sets
// each one up.
FORCE_INLINE WorkerInterfaces& start_worker_interfaces() {
    if constexpr (any_sender_channel_serviced) {
        new (worker_interface_storage.bytes) WorkerInterfaces{};
    }
    return *std::launder(reinterpret_cast<WorkerInterfaces*>(worker_interface_storage.bytes));
}"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("made", dst)
