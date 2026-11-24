# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash on Tenstorrent hardware.

- `tt/` — the device implementation (see `README.md` for the module map)
- `reference/` — HF-based CPU reference, the truth the device tests grade against
- `tests/` — unit tests, one file per module
- `TESTS_AND_PORT.md` — the Blackhole -> Wormhole port write-up
"""

__all__: list[str] = []
