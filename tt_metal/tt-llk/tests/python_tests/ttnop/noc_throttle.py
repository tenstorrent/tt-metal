# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# LOCAL ONLY. This file is untracked and should not be pushed.
# See NOC-THROTTLE.md for the integration steps.

"""Quasar NIU NoC throttler for ttnop.

Holds this tile's NoC injection at one handshake per `window` cycles during a
sweep. Fillers change the timing of one thread. This changes the timing of every
NoC transfer on the tile.
"""

# Quasar NoC throttler, as NOC_CFG(cnt) = TT_NOC_REG_MAP_BASE_ADDR + 0x100 + cnt * 4
#   tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:146
#   tt_metal/hw/inc/internal/tt-2xx/quasar/noc/tt_tensix_noc_overlay_reg.h:77
# One shared window feeds every per-port handshake limit. NIU is this tile's own
# NoC injection, so a window of N limits the core to one handshake every N cycles.
# Wormhole and Blackhole have no throttler registers at all.
NOC_THROTTLER_CYCLES_PER_WINDOW = 0x0200012C  # cfg id 0xb
NOC_THROTTLER_HANDSHAKES_PER_WINDOW_NIU = 0x02000130  # cfg id 0xc


class NocThrottle:
    """Hold the NoC at one handshake per `window` cycles during a sweep.

    The throttle is applied once around the whole sweep. Reloading an ELF does
    not change NoC config. The clean control and variants must use the same
    setting so the report only measures the filler delay.

    The reset values are read back and restored instead of being assumed.
    Nothing in the register block is an enable, so zero handshakes per window
    may mean either a stopped or unthrottled NoC.

    `window == 0` does not write any registers. `log` is optional. When set,
    it is called with one status line after the throttle is applied.
    """

    def __init__(self, window: int, read_word, write_word, log=None):
        self.window = window
        self._read, self._write = read_word, write_word
        self._log = log
        self._saved = None

    def __enter__(self):
        if self.window:
            self._saved = [
                self._read(register)
                for register in (
                    NOC_THROTTLER_CYCLES_PER_WINDOW,
                    NOC_THROTTLER_HANDSHAKES_PER_WINDOW_NIU,
                )
            ]
            self._write(NOC_THROTTLER_CYCLES_PER_WINDOW, self.window)
            self._write(NOC_THROTTLER_HANDSHAKES_PER_WINDOW_NIU, 1)
            if self._log:
                self._log(f">> NoC throttled to 1/{self.window} (was {self._saved})")
        return self

    def __exit__(self, *exc):
        if self._saved is None:
            return False
        try:
            self._write(NOC_THROTTLER_HANDSHAKES_PER_WINDOW_NIU, self._saved[1])
            self._write(NOC_THROTTLER_CYCLES_PER_WINDOW, self._saved[0])
        except Exception:
            # Keep the original device error instead of raising a restore error.
            pass
        self._saved = None
        return False
