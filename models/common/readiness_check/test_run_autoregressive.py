# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import inspect

from models.common.readiness_check.run_autoregressive import _hf_generate_greedy


def test_hf_greedy_supplies_attention_mask_for_eos_as_pad_tokenizers():
    source = inspect.getsource(_hf_generate_greedy)
    assert "attention_mask = torch.ones_like(input_ids)" in source
    assert "attention_mask=attention_mask" in source
