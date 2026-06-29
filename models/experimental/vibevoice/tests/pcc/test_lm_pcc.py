# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 — Language Model PCC test.

Loads real LM weights, runs reference Qwen2 forward and TT forward for a
short sequence (S=32), asserts PCC >= 0.99 on last_hidden_state.
"""

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.tests.pcc.lm_pcc_common import (
    PCC_THRESHOLD,
    SEQ_LEN,
    build_tt_lm,
    reference_lm_forward,
    tt_decode_hidden,
    tt_prefill_hidden,
)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_hidden_state_pcc(mesh_device, vv_config, lm_state):
    torch.manual_seed(0)
    cfg = vv_config.decoder

    # Sequence of SEQ_LEN+1: first SEQ_LEN drive prefill, the last drives one decode step.
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN + 1), dtype=torch.long)

    # 1) Reference over the full SEQ_LEN+1 (incremental decode == prefix of this).
    ref_hidden_full = reference_lm_forward(lm_state, input_ids, vv_config)  # [1, S+1, hidden]
    ref_prefill = ref_hidden_full[:, :SEQ_LEN]
    ref_decode = ref_hidden_full[:, SEQ_LEN:]  # [1, 1, hidden]

    # 2) TT prefill on the fixed KV cache + one decode step.
    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    kv_cache = lm_tt.alloc_kv_cache(SEQ_LEN + 8)
    tt_prefill = tt_prefill_hidden(lm_tt, input_ids[:, :SEQ_LEN], kv_cache)
    tt_decode = tt_decode_hidden(lm_tt, input_ids[:, SEQ_LEN : SEQ_LEN + 1], SEQ_LEN, kv_cache)

    passed_p, pcc_p = comp_pcc(ref_prefill.to(torch.float32), tt_prefill, pcc=PCC_THRESHOLD)
    passed_d, pcc_d = comp_pcc(ref_decode.to(torch.float32), tt_decode, pcc=PCC_THRESHOLD)
    per_pos = [
        comp_pcc(ref_prefill[:, p].to(torch.float32), tt_prefill[:, p], pcc=PCC_THRESHOLD)[1] for p in range(SEQ_LEN)
    ]
    lows = sorted(range(SEQ_LEN), key=lambda i: per_pos[i])[:6]
    print(f"[test_lm_pcc] prefill PCC={pcc_p:.6f}  decode PCC={pcc_d:.6f}")
    print(
        f"[test_lm_pcc] per-pos prefill PCC: last={per_pos[-1]:.5f} min={min(per_pos):.5f} "
        f"median={sorted(per_pos)[SEQ_LEN // 2]:.5f}  lows=" + ",".join(f"p{i}={per_pos[i]:.4f}" for i in lows)
    )
    assert passed_p, f"LM prefill last_hidden PCC {pcc_p:.6f} < {PCC_THRESHOLD}"
    assert passed_d, f"LM decode last_hidden PCC {pcc_d:.6f} < {PCC_THRESHOLD} (fixed-cache + sdpa_decode)"
