# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Parsing of the raw Qwen3-ASR decode. Pure python, no device or model dependency."""
import re


def parse_asr(text):
    """Raw decode -> (language, transcription). Format: 'language <Lang><asr_text><text>'.

    Decode with skip_special_tokens=False so the '<asr_text>' marker is still present to split on.
    Same rule as the server's _parse.
    """
    m = re.search(r"language\s*(.*?)<asr_text>(.*)", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()
