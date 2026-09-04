# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Single-shot forward pass, used as the profiler target."""

import pytest
import torch

import ttnn
from models.experimental.modernbert.common import build_inputs, load_config, load_torch_model
from models.experimental.modernbert.reference.modernbert import ModernBertModel
from models.experimental.modernbert.tt.modernbert_model import TtnnModernBertModel
from models.experimental.modernbert.tt.weights import prepare_weights


# Three batches at seq 256, because batch changes both the code path and the core
# counts. b1 runs the GeGLU block interleaved and b4 block-sharded, so profiling
# only the first would omit the sharded path entirely. b8 is here to settle where
# core count comes from: LayerNorm and the two head ops were measured on 8 cores
# at b1 and 32 at b4, which matches one core per tile-row (256 rows is 8 tile-rows,
# 1024 is 32). If that is the rule, b8's 2048 rows should put them on all 64.
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("seq_len, batch_size", [(256, 1), (256, 4), (256, 8)])
def test_modernbert_single_forward(device, seq_len, batch_size):
    config = load_config()
    hf = load_torch_model()
    ref = ModernBertModel(config)
    ref.load_state_dict(hf.state_dict(), strict=True)
    ref.eval()

    ids, attention_mask = build_inputs(seq_len=seq_len, batch_size=batch_size)
    params = prepare_weights(ref, device)
    model = TtnnModernBertModel(params, config, device, seq_len, attention_mask=attention_mask)
    tt_ids = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    out = model(tt_ids)
    ttnn.synchronize_device(device)
    assert tuple(ttnn.to_torch(out).shape)[-1] == config.hidden_size
