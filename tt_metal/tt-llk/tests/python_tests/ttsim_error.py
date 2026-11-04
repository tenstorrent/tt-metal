# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parse ttsim's ``[clk] ERROR:`` lines.

Shared by ``helpers.llk_pytest_plugin`` (rewrite a forked ``_Exit`` into a
failure) and ``render_ttsim_report.py`` (HTML category breakdown). One
pattern so the two cannot drift.

ttsim prints either of:

    [<clk>] ERROR: <Category>: <function>
    [<clk>] ERROR: <Category>: <function>: <details>

Details may also follow the function as whitespace
(``tensix_elw_op src_fmt=10``). A bare function, including trailing
whitespace, has no detail.
"""

from __future__ import annotations

import re

_TTSIM_ERR_RE = re.compile(
    r"^\[\d+\] ERROR: (\w+): (\w+)(?:[: \t]+(.*?))?[ \t]*$",
    re.MULTILINE,
)


def parse_ttsim_error(text: str) -> tuple[str, str, str | None] | None:
    """Return ``(category, function, detail)`` for the first ttsim error line.

    ``detail`` is ``None`` when the line names only the function. Returns
    ``None`` when ``text`` has no matching line.
    """
    match = _TTSIM_ERR_RE.search(text or "")
    if not match:
        return None
    detail = (match.group(3) or "").strip() or None
    return match.group(1), match.group(2), detail
