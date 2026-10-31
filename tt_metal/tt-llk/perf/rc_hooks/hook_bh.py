# LLK_RC_HOOK for Blackhole (bh-agent). RC_ORDER = comma list of TRISCs (e.g. "trisc2" or "trisc1,trisc2"); each one,
# in that order, does ONE L1 load through its own L1 port, so the last one in the list is the last grant on L1 port 3.
# BH: halting a running TRISC and resuming it hangs the next BRISC command, so first put all three TRISCs into soft
# reset (the harness's next RESET_TRISCS does the same), then use the in-reset path of exalens (JAL x0,0 at the code
# start, release, halt, load, back into reset). Addresses are 16-B aligned (BH TRISC2 debug access bug when
# addr % 16 > 4).
import os

from ttexalens.tt_exalens_lib import convert_coordinate

ADDR = {
    "trisc0": 0x1FF90,
    "trisc1": 0x1FFA0,
    "trisc2": 0x1FFB0,
}  # one address per TRISC: l1_addr_core_p3 shows who was last
ORDER = [
    r.strip() for r in os.environ.get("RC_ORDER", "trisc2").split(",") if r.strip()
]


def hook(location):
    blk = convert_coordinate(location, 0, None).noc_block
    for r in ("trisc0", "trisc1", "trisc2"):
        rd = blk.get_risc_debug(r)
        if not rd.is_in_reset():
            rd.set_reset_signal(True)
    for r in ORDER:
        rd = blk.get_risc_debug(r)
        with rd.ensure_private_memory_access():
            rd.debug_hardware.read_memory(ADDR[r])
