# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic wrapper; phase progress and periodic Python stacks, no runtime changes."""

import faulthandler

from models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder import main
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import MultichipDecoder

if __name__ == "__main__":
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    original_forward = MultichipDecoder._forward
    counter = 0

    def tracked_forward(self, *args, **kwargs):
        global counter
        counter += 1
        selected = counter % 16 == 0 or counter == 1
        if selected:
            print("FORWARD_ENTER", counter, kwargs.get("chunk_start_idx"), flush=True)
        output = original_forward(self, *args, **kwargs)
        if selected:
            print("FORWARD_EXIT", counter, flush=True)
        return output

    MultichipDecoder._forward = tracked_forward
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
