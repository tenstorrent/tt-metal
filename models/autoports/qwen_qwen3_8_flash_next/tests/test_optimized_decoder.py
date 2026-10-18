# SPDX-License-Identifier: Apache-2.0
"""Run the optimized path against the independent HF oracle, using real weights by default."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from models.autoports.qwen_qwen3_8_flash_next.tests import test_functional_decoder as contract
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedGatedDeltaNet,
    OptimizedHyperConnection,
    OptimizedMoE,
    OptimizedPLE,
    OptimizedQSA,
)

EVIDENCE = Path(__file__).resolve().parents[1] / "doc/optimized_decoder"


@pytest.mark.parametrize("layer_idx", [0, 3, 1], ids=["layer0", "layer3", "layer1"])
@pytest.mark.parametrize(
    "batch", [int(x) for x in os.environ.get("QWEN38_DECODER_BATCHES", "1,2").split(",")], ids=lambda n: f"B{n}"
)
def test_optimized_decoder_contract(layer_idx, batch):
    source = Path(__file__).resolve().parents[1] / "tt/optimized_decoder.py"
    print(
        json.dumps(
            dict(optimized_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), layer=layer_idx, batch=batch)
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

    original_factory = OptimizedDecoder.from_state_dict

    class CheckedOptimizedDecoder:
        @staticmethod
        def from_state_dict(*args, **kwargs):
            model = original_factory(*args, **kwargs)
            assert type(model) is OptimizedDecoder
            assert type(model.attention) is (OptimizedQSA if layer_idx == 3 else OptimizedGatedDeltaNet)
            if layer_idx == 1:
                assert type(model.ple) is OptimizedPLE
            assert type(model.mlp) is OptimizedMoE
            assert type(model.attn_hyper) is OptimizedHyperConnection
            assert type(model.mlp_hyper) is OptimizedHyperConnection
            return model

    with (
        patch.dict(os.environ, {"QWEN38_WEIGHT_SOURCE": os.environ.get("QWEN38_WEIGHT_SOURCE", "real")}),
        patch.object(contract, "FunctionalDecoder", CheckedOptimizedDecoder),
        patch.object(contract, "EVIDENCE", EVIDENCE),
        patch.object(contract, "SyntheticState", synthetic_state),
        patch.object(contract, "LazyPLETable", reference_table),
    ):
        contract.test_functional_decoder_contract(layer_idx, batch)
    print(f"OPTIMIZED_DECODER_PASS layer={layer_idx} B={batch}", flush=True)
