# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA target patterns for ttml.modules.LoraConfig (regex, matched on module paths)."""

from __future__ import annotations

ATTN_TARGETS = [
    r"blocks\.\d+\.attn[12]\.to_q",
    r"blocks\.\d+\.attn[12]\.to_k",
    r"blocks\.\d+\.attn[12]\.to_v",
    r"blocks\.\d+\.attn[12]\.to_out",
]

FFN_TARGETS = [
    r"blocks\.\d+\.ffn\.ff[12]",
]

TARGET_SETS = {
    "attn": ATTN_TARGETS,
    "attn+ffn": ATTN_TARGETS + FFN_TARGETS,
}


def resolve(target_set: str) -> list[str]:
    try:
        return list(TARGET_SETS[target_set])
    except KeyError:
        raise ValueError(
            f"unknown LORA_TARGET_SET {target_set!r}; expected one of: {', '.join(sorted(TARGET_SETS))}"
        ) from None
