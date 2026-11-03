"""stage_a11.py: v4txa11 = v4txa10 with this_chip() reverted (user decision 2026-11-03, after A9's census: VC1
Mesh2HopMerge +3.4 / +2.4, MeshMulticast +1.8 cycles/packet, mapped to this_chip()'s probe, v4txa4p1): receiver::run
and single::run build my_chip in place again, from l1_routing_table(); the factory goes. Expected: receiver::run and
single::run loops as C4's (v4txc4) but for the round split's frame offsets (RoundCounters 16 -> 8 B), sender::run as
C4's but for the round split's operand swaps."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txa10", "v4txa11", "a11.patch")
