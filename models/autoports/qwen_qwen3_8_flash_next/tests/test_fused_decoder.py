# SPDX-License-Identifier: Apache-2.0
"""Run the complete established contract through the fused implementation."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from models.autoports.qwen_qwen3_8_flash_next.tests import test_functional_decoder as contract
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import (
    FusedDecoder,
    FusedGatedDeltaNet,
    FusedHyperConnection,
    FusedMoE,
    FusedPLE,
    FusedQSA,
)

EVIDENCE = Path(__file__).resolve().parents[1] / "doc/fused_decoder"


@pytest.mark.parametrize("layer_idx", [0, 3, 1], ids=["layer0", "layer3", "layer1"])
@pytest.mark.parametrize(
    "batch", [int(x) for x in os.environ.get("QWEN38_DECODER_BATCHES", "1,2").split(",")], ids=lambda n: f"B{n}"
)
def test_fused_decoder_contract(layer_idx, batch):
    source = Path(__file__).resolve().parents[1] / "tt/fused_decoder.py"
    print(
        json.dumps(
            dict(fused_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), layer=layer_idx, batch=batch)
        ),
        flush=True,
    )
    original_table = contract.LazyPLETable

    def reference_table(state):
        if not isinstance(state, contract.CheckpointState):
            return original_table(state)
        # Same HF lookup bytes, mapped once instead of reopening a safetensor
        # per token/index. This changes only CPU oracle I/O, never TT tables.
        import torch

        from models.autoports.qwen_qwen3_8_flash_next.tests.ple_long_context import MappedCheckpointTable

        mapped = MappedCheckpointTable(contract.CheckpointState(state.path))
        indices = torch.tensor([shard * 2500012 + offset for shard in range(128) for offset in (0, 2500011)])
        assert torch.equal(mapped(indices), original_table(state)(indices))
        print("HF_PLE_MAPPED_LOOKUP_BYTES_EXACT all128shards first_last_rows", flush=True)
        return mapped

    original_state = contract.SyntheticState

    def synthetic_state(path, **kwargs):
        return original_state(contract.ROOT / "doc/functional_decoder/weight_stats" / Path(path).name, **kwargs)

    original_factory = FusedDecoder.from_state_dict

    class CheckedFusedDecoder:
        @staticmethod
        def from_state_dict(*args, **kwargs):
            model = original_factory(*args, **kwargs)
            assert type(model) is FusedDecoder
            assert type(model.attention) is (FusedQSA if layer_idx == 3 else FusedGatedDeltaNet)
            if layer_idx == 1:
                assert type(model.ple) is FusedPLE
            assert type(model.mlp) is FusedMoE
            assert type(model.attn_hyper) is FusedHyperConnection
            assert type(model.mlp_hyper) is FusedHyperConnection
            return model

    with (
        patch.object(contract, "FunctionalDecoder", CheckedFusedDecoder),
        patch.object(contract, "EVIDENCE", EVIDENCE),
        patch.object(contract, "SyntheticState", synthetic_state),
        patch.object(contract, "LazyPLETable", reference_table),
    ):
        contract.test_functional_decoder_contract(layer_idx, batch)
    print(f"FUSED_DECODER_PASS layer={layer_idx} B={batch}", flush=True)
