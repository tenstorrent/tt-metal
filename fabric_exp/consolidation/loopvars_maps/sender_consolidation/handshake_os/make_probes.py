"""make_probes.py: the run_handshakes optimize("Os") measurement (main session, 2026-11-03; study only, not stages).
  v4txc4p:  v4txc4 with VC1 + VC2 admitted by the bring-up gate (vc2_config_supported without !has_vc1, as v4rxc7v's
            size probe), so the VC1+VC2 configs build;
  v4txc4pn: v4txc4p without run_handshakes' __attribute__((optimize("Os"))).
Both are built at -O3 on the quick set and compared (handshake_os.sh)."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
for d in ("v4txc4p", "v4txc4pn"):
    if os.path.exists(f"{V}/{d}"):
        sys.exit(f"{d} exists")
shutil.copytree(f"{V}/v4txc4", f"{V}/v4txc4p", symlinks=True)
k = f"{V}/v4txc4p/fabric_erisc_router_v4.cpp"
s = open(k).read()
o = "constexpr bool vc2_config_supported = is_2d && !has_vc1 && ACTUAL_VC2_SENDER_CHANNELS == 1 &&"
assert s.count(o) == 1
open(k, "w").write(s.replace(o, "constexpr bool vc2_config_supported = is_2d && ACTUAL_VC2_SENDER_CHANNELS == 1 &&"))
shutil.copytree(f"{V}/v4txc4p", f"{V}/v4txc4pn", symlinks=True)
p = f"{V}/v4txc4pn/tt_metal/fabric/hw/inc/edm_fabric/router/router_init.hpp"
s = open(p).read()
o = '__attribute__((noinline, optimize("Os"))) void run_handshakes() {'
assert s.count(o) == 1
open(p, "w").write(s.replace(o, "__attribute__((noinline)) void run_handshakes() {"))
print("made v4txc4p, v4txc4pn")
