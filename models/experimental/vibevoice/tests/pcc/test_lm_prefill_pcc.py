# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Language Model prefill-mode PCC vs HuggingFace Qwen2 (last_hidden_state only)."""

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.tests.pcc.lm_pcc_common import (
    PCC_THRESHOLD,
    PREFILL_ISL_SWEEP_LENGTHS,
    SEQ_LEN,
    build_tt_lm,
    compare_prefill_hidden_pcc,
    print_prefill_pcc_isl_debug,
    reference_lm_forward,
    tt_prefill_hidden,
)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_prefill_hidden_state_pcc(mesh_device, vv_config, lm_state):
    torch.manual_seed(0)
    cfg = vv_config.decoder

    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), dtype=torch.long)

    ref_prefill = reference_lm_forward(lm_state, input_ids, vv_config)

    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    kv_cache = lm_tt.alloc_kv_cache(SEQ_LEN + 8)
    tt_prefill = tt_prefill_hidden(lm_tt, input_ids, kv_cache)

    passed_p, pcc_p = comp_pcc(ref_prefill.to(torch.float32), tt_prefill, pcc=PCC_THRESHOLD)
    per_pos = [
        comp_pcc(ref_prefill[:, p].to(torch.float32), tt_prefill[:, p], pcc=PCC_THRESHOLD)[1] for p in range(SEQ_LEN)
    ]
    lows = sorted(range(SEQ_LEN), key=lambda i: per_pos[i])[:6]
    print(f"[test_lm_prefill_pcc] prefill PCC={pcc_p:.6f}")
    print(
        f"[test_lm_prefill_pcc] per-pos prefill PCC: last={per_pos[-1]:.5f} min={min(per_pos):.5f} "
        f"median={sorted(per_pos)[SEQ_LEN // 2]:.5f}  lows=" + ",".join(f"p{i}={per_pos[i]:.4f}" for i in lows)
    )
    assert passed_p, f"LM prefill last_hidden PCC {pcc_p:.6f} < {PCC_THRESHOLD}"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_prefill_hidden_state_pcc_isl_sweep(mesh_device, vv_config, lm_state):
    """Prefill PCC vs HuggingFace Qwen2 across multiple input sequence lengths."""
    cfg = vv_config.decoder
    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    failures = []
    print("[test_lm_prefill_pcc]")

    for seq_len in PREFILL_ISL_SWEEP_LENGTHS:
        torch.manual_seed(0)
        input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), dtype=torch.long)

        ref_prefill = reference_lm_forward(lm_state, input_ids, vv_config)
        kv_cache = lm_tt.alloc_kv_cache(seq_len + 8)
        tt_prefill = tt_prefill_hidden(lm_tt, input_ids, kv_cache)

        print_prefill_pcc_isl_debug(ref_prefill, tt_prefill, seq_len)
        passed_p, pcc_p, per_pos = compare_prefill_hidden_pcc(ref_prefill, tt_prefill, seq_len)

        print("--------------------------------------------------")
        print(f"Sequence Length : {seq_len}")
        print(f"Overall PCC     : {pcc_p:.5f}")
        print(f"Last Token PCC  : {per_pos[-1]:.5f}")
        print(f"Minimum PCC     : {min(per_pos):.5f}")
        print(f"Median PCC      : {sorted(per_pos)[seq_len // 2]:.5f}")
        print("--------------------------------------------------")

        if not passed_p:
            failures.append(f"sequence_length={seq_len} measured_pcc={pcc_p:.6f} threshold={PCC_THRESHOLD}")

    if failures:
        assert False, "LM prefill ISL sweep PCC below threshold:\n" + "\n".join(failures)

    print("PASS")
