# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the qwen38_27b serving-contract tests (SC.1, see $BRINGUP_DIR/serving_contract.md).

The tests reach the model only through the hooks' device model (assemble) or the registered prefill adapter (adapter),
and fail with "not built: ..." while the part does not exist yet.
"""

from __future__ import annotations

import pytest
import torch
from bringup_fw.core import metrics
from bringup_fw.testing import server_rules as R
from bringup_fw.testing import serving

PAD_ID = R.PAD_ID
KV_TTNN = {"bfp8": "bfloat8_b", "bfloat8_b": "bfloat8_b", "bf16": "bfloat16", "bfloat16": "bfloat16"}
STATE_TTNN = {"fp32": "float32", "float32": "float32", "bf16": "bfloat16", "bfloat16": "bfloat16"}


def not_built(what: str):
    pytest.fail(f"not built: {what} (serving_contract.md)", pytrace=False)


def answer(s, key: str, default):
    v = serving.answer(s, key)
    return default if v in (None, "") else v


def kv_threshold(s) -> float:
    return max(float(s.get("serving.min_kv_pcc")), s.threshold("state", 0.97))


def attn_layers(s) -> list[int]:
    return [i for i in s.layers() if "key" in s.state_names(i)]


def linear_layers(s) -> list[int]:
    return [i for i in s.layers() if "recurrent" in s.state_names(i)]


def text_config(s) -> dict:
    import json
    from pathlib import Path

    c = json.loads((Path(s.hf_dir) / "config.json").read_text())
    return c.get("text_config", c)


def device_model(s, mesh, layers):
    try:
        model = s.hooks().device_model(mesh, s, layers, lm_head=False)
    except NotImplementedError as e:
        not_built(f"hooks.device_model ({e})")
    return model


def require(obj, *names: str, what: str = "device model"):
    miss = [n for n in names if not callable(getattr(obj, n, None))]
    if miss:
        not_built(f"{what} lacks {', '.join(miss)}")


def run_chunk(model, layers, state, tokens: torch.Tensor, start: int, end: int, chunk: int) -> None:
    """One server chunk: tokens[start:end] padded with PAD_ID to ``chunk``, every served layer."""
    t = torch.full((chunk,), PAD_ID, dtype=torch.int64)
    t[: end - start] = tokens[start:end]
    h = model.embed(t, start=start)
    for i in layers:
        h2 = model.layer(i, h, start, state, end=end)
        model.free(h)
        h = h2
    model.free(h)
    model.sync()


def pcc_check(name: str, got: torch.Tensor, want: torch.Tensor, thr: float, fails: list) -> float:
    got, want = got.float().reshape(want.shape), want.float()
    p = metrics.pcc(got, want)
    metrics.record(name, p)
    ok = p >= thr
    print(f"{'ok  ' if ok else 'FAIL'} {name}: pcc {p:.6f} (>= {thr})", flush=True)
    if not ok:
        fails.append(f"{name}: pcc {p:.5f} < {thr}")
    return p


def report(test: str, fails: list) -> None:
    metrics.record(f"contract_failed_{test}", len(fails))
    for f in fails:
        print(f"FAIL {f}")
    assert not fails, f"{len(fails)} contract checks failed: " + "; ".join(fails[:4])
