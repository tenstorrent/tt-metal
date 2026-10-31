# LLK_RC_HOOK (RC experiment): make the PACK TRISC (trisc2) the last requester on its shared L1 port before the
# measured kernel: debug-halt trisc2 and have it LOAD one L1 word through its own port, then resume / put back in
# reset. Works on Wormhole and Blackhole. Clears the TRISC branch-predictor disable bits if the architecture has them
# (the Wormhole halt sets them). LLK_RC_HOOK_ADDR overrides the address (default 0x1FFB8).
import os

from helpers.device import get_register_store
from ttexalens.tt_exalens_lib import convert_coordinate

ADDR = int(os.environ.get("LLK_RC_HOOK_ADDR", "0x1FFB8"), 0)


def hook(location):
    rd = convert_coordinate(location, 0, None).noc_block.get_risc_debug("trisc2")
    with rd.ensure_private_memory_access():
        rd.debug_hardware.read_memory(ADDR)
    try:
        get_register_store(location, 0).write_register(
            "DISABLE_RISC_BP_Disable_trisc", 0
        )
    except Exception:
        pass
