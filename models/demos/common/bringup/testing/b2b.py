# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Back-to-back prefill workloads (DeepSeek-V4.1 P.48 / FIX.23; ported by P.50 with only what the determinism test
needs: the workloads' token ids). Workloads (``ids``): novel = the spec's prompt file (the ladder's ids); code = a
text file (WL_SOURCE, default CODE_TEXT: tt-metal C++ sources) as a model turn after CODE_REQUEST, through the spec's
prompt builder."""

from __future__ import annotations

import copy
import os

CODE_TEXT = "/scratch/dnijemcevic/moeplace/code.txt"
CODE_REQUEST = "Print the C++ source of tt-metal TTNN operations, file by file."


def ids(s, wl: str, seq: int) -> list[int]:
    """The workload's first ``seq`` token ids (novel: the spec's prompt file; code: WL_SOURCE / CODE_TEXT)."""
    from models.demos.common.bringup.reference import prompt as P

    if wl == "novel":
        rec = P.load(s)
        assert rec is not None, f"no prompt file {P.prompt_path(s)}"
        out = rec["token_ids"]
    else:
        from models.demos.common.bringup.reference.golden import tokenizer

        sc = copy.copy(s)
        sc.data = copy.deepcopy(s.data)
        src = os.environ.get("WL_SOURCE") or CODE_TEXT
        sc.data["text"].update(source=src, request=CODE_REQUEST, wrap="model_turn")
        out, _ = P.build_ids(sc, tokenizer(s), seq)
    assert len(out) >= seq, (wl, len(out), seq)
    return list(out[:seq])
