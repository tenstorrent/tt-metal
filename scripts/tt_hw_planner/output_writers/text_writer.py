# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Text output writer — emits write_transcript() helper."""

from __future__ import annotations


def emit_helper_snippet() -> str:
    """Return Python source for a write_transcript helper.

    Used by S2TT / T2T / ASR / VLM templates inside their demo files.
    Not a separate file — task templates inline this where they need it.
    """
    return '''
def write_transcript(transcripts, output_path):
    """Write transcript entries as one segment per line.

    transcripts: list of (start_sec, end_sec, text) OR list of (text,)
                  OR single text string.
    """
    from pathlib import Path
    lines = []
    if isinstance(transcripts, str):
        lines.append(transcripts)
    else:
        for entry in transcripts:
            if isinstance(entry, tuple) and len(entry) == 3:
                start_sec, end_sec, text = entry
                lines.append(f"[{start_sec:6.2f} -> {end_sec:6.2f}] {text}")
            elif isinstance(entry, tuple) and len(entry) == 1:
                lines.append(str(entry[0]))
            else:
                lines.append(str(entry))
    Path(output_path).write_text("\\n".join(lines) + "\\n", encoding="utf-8")
'''
