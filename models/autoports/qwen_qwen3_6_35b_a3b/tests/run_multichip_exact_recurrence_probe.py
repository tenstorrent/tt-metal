# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the ordinary real-weight multichip gates with the exact recurrence probe.

All current production defaults remain enabled. Pass the ordinary runner args;
this wrapper changes only construction-time recurrence installation. A failed
component, layer 34, changing-state, trace or watcher gate rejects adoption.
"""

import runpy

from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_exact_recurrence_probe import install
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder


def main():
    descriptor = MultichipDecoder.__dict__["from_state_dict"]
    original = descriptor.__func__

    def construct(cls, state_dict, **kwargs):
        return install(original(cls, state_dict, **kwargs))

    MultichipDecoder.from_state_dict = classmethod(construct)
    try:
        runpy.run_module("models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder", run_name="__main__")
    finally:
        MultichipDecoder.from_state_dict = descriptor


if __name__ == "__main__":
    main()
