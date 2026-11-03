"""stage_c0.py: v4txc0 = v4txo8 (the sender ownership pass's final build) merged with v4rxnsf2 (the receiver's N/S fix 1,
ForwardCommands; V4_NS_PRESETS.md §9.3). V4_SENDER_CONSOLIDATION.md §3, stage 0.

Every file is v4txo8's except:
  - router_downstream.hpp: v4rxnsf2's, without its INLINE_WRITE_ALL_BYTES definition (v4txo8 moved it to
    router_common.hpp; v4txo8's own router_downstream.hpp differs from v4rxg3 only by that removal);
  - the additive NoC API (dataflow_api.h, blackhole/ and wormhole/noc_nonblocking_api.h): v4rxnsf2's (v4txo8's equal
    v4rxg3's);
  - fabric_erisc_router_v4.cpp: v4txo8's, with v4rxnsf2's one line (ForwardCommands<0>::preset for
    preset_downstream_cmd_bufs<0>).
The script checks those premises (every other file of v4rxnsf2 equals v4rxg3's; the two v4.cpp files differ from
v4rxg3's as stated) before writing."""
import filecmp, os, shutil, sys

V = "/home/aagarwal/tt-metal/fabric_exp/variants"
TX, RX, BASE, DST = f"{V}/v4txo8", f"{V}/v4rxnsf2", f"{V}/v4rxg3", f"{V}/v4txc0"
if os.path.exists(DST):
    sys.exit(f"{DST} exists")
R = "tt_metal/fabric/hw/inc/edm_fabric/router/"
NOC_API = ["tt_metal/hw/inc/api/dataflow/dataflow_api.h",
           "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h",
           "tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h"]
K = "fabric_erisc_router_v4.cpp"


def files(d):
    return sorted(os.path.relpath(os.path.join(p, f), d) for p, _, fs in os.walk(d) for f in fs)


# Premises: v4rxnsf2 changes only router_downstream.hpp, the NoC API and v4.cpp against v4rxg3; v4txo8 leaves the NoC
# API as v4rxg3's.
assert files(RX) == files(BASE) == files(TX)
changed_rx = [f for f in files(RX) if not filecmp.cmp(f"{RX}/{f}", f"{BASE}/{f}", shallow=False)]
assert sorted(changed_rx) == sorted([R + "router_downstream.hpp", K] + NOC_API), changed_rx
for f in NOC_API:
    assert filecmp.cmp(f"{TX}/{f}", f"{BASE}/{f}", shallow=False), f

shutil.copytree(TX, DST, symlinks=True)
for f in NOC_API:
    shutil.copy(f"{RX}/{f}", f"{DST}/{f}")

# router_downstream.hpp: v4rxnsf2's without the INLINE_WRITE_ALL_BYTES definition (router_common.hpp has it).
s = open(f"{RX}/{R}router_downstream.hpp").read()
old = """// Byte enables of a whole-word inline write (the credit register).
constexpr uint32_t INLINE_WRITE_ALL_BYTES = 0xF;

"""
assert s.count(old) == 1
open(f"{DST}/{R}router_downstream.hpp", "w").write(s.replace(old, ""))
# The same removal is v4txo8's only change to the file.
assert open(f"{TX}/{R}router_downstream.hpp").read() == open(f"{BASE}/{R}router_downstream.hpp").read().replace(old, "")

# fabric_erisc_router_v4.cpp: v4txo8's with v4rxnsf2's one line.
s = open(f"{TX}/{K}").read()
old = "        preset_downstream_cmd_bufs<0>(args.vc0());\n"
new = "        ForwardCommands<0>::preset(args.vc0());\n"
assert s.count(old) == 1
assert open(f"{RX}/{K}").read() == open(f"{BASE}/{K}").read().replace(old, new)
open(f"{DST}/{K}", "w").write(s.replace(old, new))
print("made", DST)
