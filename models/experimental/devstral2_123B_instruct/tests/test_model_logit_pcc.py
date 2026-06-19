# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-model logit PCC vs HuggingFace ``AutoModelForCausalLM``.

After chunked prefill (golden corpus tokens), compares last-prefill logits and
``DECODE_GENERATION_LENGTH`` (10) decode logits with PCC ≥ 0.90. Decode follows
tt-transformers ``test_model.py``: **HF greedy** (temperature=0 argmax) picks the
next input token for both HF and TT after each logits comparison.

Prefill lengths follow the decoder-layer PCC sweep (powers of two 32 … 262144).
The TT model is built once per mesh; tiled weights load from the shared on-disk cache
at ``seq_{DEVSTRAL2_WEIGHT_CACHE_SEQ_LEN}`` (default **262144**) when present, otherwise
HF shards are downloaded and the cache is written. Runtime ``model_max_seq_len`` is fixed
for the whole sweep (covers worst-case prefill + eval). HF and TT prefill run in **128-token
chunks** with incremental ``DynamicCache`` on HF (O(chunk) host memory).
HF reference inference uses **CPU + disk offload** when no CUDA GPU is present.
Logit PCC compares **raw logits** (full vocab); token-matching tests use golden corpus refs.

Prefill lengths follow the decoder-layer PCC sweep (powers of two 32 … 262144).
The TT model is built once per mesh; tiled weights load from the shared on-disk cache
at ``seq_{DEVSTRAL2_WEIGHT_CACHE_SEQ_LEN}`` (default **262144**) when present, otherwise
HF shards are downloaded and the cache is written. Runtime ``model_max_seq_len`` is fixed
for the whole sweep (covers worst-case prefill + eval). HF and TT prefill run in **128-token
chunks** with incremental ``DynamicCache`` on HF (O(chunk) host memory).
HF reference inference uses **CPU + disk offload** when no CUDA GPU is present.

Run sanity (CI gate)::

    pytest models/experimental/devstral2_123B_instruct/tests/test_model_logit_pcc.py -k sanity -v

Run full sweep (default pytest timeout **12 h**, includes HF CPU + TT; override with
``DEVSTRAL2_LOGIT_PCC_SWEEP_TIMEOUT_SEC``)::

    pytest models/experimental/devstral2_123B_instruct/tests/test_model_logit_pcc.py -k sweep -v
"""

from __future__ import annotations

import pytest
import ttnn

from models.experimental.devstral2_123B_instruct.tests.logit_pcc_common import (
    PREFILL_SANITY_SEQ_LENGTHS,
    PREFILL_SWEEP_SEQ_LENGTHS,
    mesh_device_param,
    run_logit_pcc_sweep,
    sweep_timeout_seconds,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import DEVSTRAL2_LARGE_L1_SMALL_SIZE

_DEVICE_PARAMS = [
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 30_000_000,
        "num_command_queues": 1,
        "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    }
]


@pytest.mark.slow
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(sweep_timeout_seconds(PREFILL_SWEEP_SEQ_LENGTHS))
@pytest.mark.parametrize("mesh_device", [mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
def test_model_logit_pcc_sweep(mesh_device):
    """HF-greedy logit PCC sweep: prefill 32 … 262144 (powers of two)."""
    run_logit_pcc_sweep(mesh_device, PREFILL_SWEEP_SEQ_LENGTHS)


@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(sweep_timeout_seconds(PREFILL_SANITY_SEQ_LENGTHS))
@pytest.mark.parametrize("mesh_device", [mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
def test_model_logit_pcc_sanity(mesh_device):
    """Short prefill lengths (32, 128) before the full logit PCC sweep."""
    run_logit_pcc_sweep(mesh_device, PREFILL_SANITY_SEQ_LENGTHS)
