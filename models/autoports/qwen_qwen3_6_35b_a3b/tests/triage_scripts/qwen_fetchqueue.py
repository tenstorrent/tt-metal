# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only capture for the preserved CQ0 hang; addresses are specific to its ELF.

Usage:
    qwen_fetchqueue

Owner:
    qwen_qwen3_6_35b_a3b functional-decoder bringup
"""

import json
import time
from pathlib import Path

from triage import ScriptConfig
from ttexalens.tt_exalens_lib import read_word_from_device, read_words_from_device

script_config = ScriptConfig()


def run(args, context):
    # AUTOTRIAGE.md verifies these against the existing cq_prefetch ELF.
    location, credit_address, base, entries = "16-2", 0x1B7C0, 0x1BD40, 1536
    before = read_word_from_device(location, credit_address, device_id=0, context=context)
    words = read_words_from_device(location, base, device_id=0, word_count=entries, context=context)
    after = read_word_from_device(location, credit_address, device_id=0, context=context)
    data = dict(
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        chip=0,
        noc0_core=location,
        credit_address=hex(credit_address),
        credit_before=hex(before),
        credit_after=hex(after),
        fetchq_base=hex(base),
        entry_bytes=4,
        words=words,
        nonzero={hex(base + 4 * i): hex(w) for i, w in enumerate(words) if w},
    )
    output = Path(__file__).parents[2] / "doc/functional_decoder/triage/fetchqueue_words.json"
    output.write_text(json.dumps(data, indent=2) + "\n")
    print("FETCHQUEUE_CAPTURE", output, "credits", hex(before), hex(after), "nonzero", len(data["nonzero"]))
    return None
