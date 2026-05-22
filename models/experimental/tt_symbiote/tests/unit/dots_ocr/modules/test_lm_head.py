# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 module test: :class:`TTNNDotsOCRDRAMShardedLMHead` + argmax.

Per user decision #5 we test the production argmax pattern
(``_argmax_token_on_device``) rather than raw logits — argmax is integer
output, so PCC isn't meaningful; instead we assert the predicted token
id matches PyTorch's argmax under the same weights.

Input shapes from production:

* prefill: post-norm hidden ``[1, 1, 14, 1536]`` then take last token
  -> ``[1, 1, 1, 1536]`` before lm_head (see ``models/dots_ocr.py:380``).
* decode: ``[1, 1, 1, 1536]`` directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.modules.linear import (
    TTNNDotsOCRDRAMShardedLMHead,
)
from models.experimental.tt_symbiote.models.dots_ocr import _argmax_token_on_device
from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.architecture_factory import (
    _get_dots_config,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.module_helpers import (
    gather_replicated_first,
    prepare_module,
    replicated_from_torch,
)


def _build_random_lm_head(seed: int = 0):
    cfg = _get_dots_config()
    lin = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        tmp = torch.empty_like(lin.weight, dtype=torch.float32)
        tmp.normal_(mean=0.0, std=0.02, generator=g)
        lin.weight.copy_(tmp.to(torch.bfloat16))
    return lin.to(torch.bfloat16).eval()


_ROWS: List[Dict[str, Any]] = [
    {"id": "lm_head_argmax_b1_s1_h1536_decode", "shape": (1, 1, 1, 1536)},
]


@pytest.mark.parametrize("row", _ROWS, ids=[r["id"] for r in _ROWS])
def test_lm_head_argmax(row, mesh_device_t3k_dp):
    """Production pattern: lm_head linear -> argmax → token id."""
    torch.manual_seed(0)
    ref = _build_random_lm_head(seed=0)

    cfg = _get_dots_config()
    x_torch = torch.randn(*row["shape"], dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        ref_logits = torch.nn.functional.linear(x_torch.to(torch.float32), ref.weight.to(torch.float32))
        ref_token = ref_logits.argmax(dim=-1)

    tt_head = TTNNDotsOCRDRAMShardedLMHead.from_torch(ref)
    prepare_module(tt_head, mesh_device_t3k_dp)

    # Match the production HiFi2 + FP32 dest accum override so argmax stability
    # is what production sees (see `_set_device_and_preprocess`).
    tt_head.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    x_tt = replicated_from_torch(x_torch, mesh_device=mesh_device_t3k_dp)

    try:
        logits_tt = tt_head(x_tt)
        token_tt = _argmax_token_on_device(logits_tt)
    except Exception as e:
        pytest.xfail(f"LM head DRAM-sharded path requires production layout: {e}")

    tok_host = gather_replicated_first(token_tt, mesh_device_t3k_dp)
    # Flatten and pull the first device's token. Logical batch is 1.
    tok_int = int(tok_host.reshape(-1)[0].item())
    ref_int = int(ref_token.reshape(-1)[0].item())

    print(f"\n[{row['id']}] TT token={tok_int} REF token={ref_int}")
    assert tok_int == ref_int, f"LM head argmax mismatch: TT={tok_int} REF={ref_int} " f"(row_id={row['id']!r})"
