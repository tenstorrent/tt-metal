# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_embedding")


@pytest.fixture(autouse=True)
def cleanup_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="linear"),
            id="linear-2x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_embedding_weights_cold_warm_cache(mesh_device, device_params):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    vocab_size = 129280
    emb_dim = 7168
    seq_len = 256

    # Create random weight
    torch_weight = torch.randn(vocab_size, emb_dim, dtype=torch.float32)

    # Create input (token IDs)
    token_ids = torch.randint(0, vocab_size, (1, 1, seq_len), dtype=torch.int32)
    token_ids_tt = ttnn.from_torch(
        token_ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Helper to convert TP-sharded output to torch
    def to_torch_concat(tt_tensor):
        """Convert TP-sharded tensor to torch with mesh composer."""
        # Embedding output is 3D: [1, seq_len, emb_dim/tp]
        # dims=(1, 2) means: replicate dim1 (seq), concat dim2 (emb_dim)
        return ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=(1, 2), mesh_shape=mesh_device.shape  # For 3D tensor: concat along last dim for TP
            ),
        )

    # === Path 1: From Weights ===
    emb_from_weights = TtParallelEmbedding(
        mesh_device,
        vocab_size,
        emb_dim,
        torch_weight=torch_weight,
        weight_cache_path=None,
    )
    output1_tt = emb_from_weights(token_ids_tt)
    output1 = to_torch_concat(output1_tt)

    # === Path 2: Cold Cache ===
    assert not TtParallelEmbedding.check_cache_complete(CACHE_DIR), "Cache should be empty before build"

    profiler.clear()
    profiler.start("build_cache")
    TtParallelEmbedding.build_ttnn_cache(
        torch_weight,
        vocab_size,
        emb_dim,
        mesh_device,
        CACHE_DIR,
    )
    profiler.end("build_cache")

    assert TtParallelEmbedding.check_cache_complete(CACHE_DIR), "Cache should be complete after build"

    profiler.start("cold_load")
    emb_cold = TtParallelEmbedding(
        mesh_device,
        vocab_size,
        emb_dim,
        torch_weight=None,
        weight_cache_path=CACHE_DIR,
    )
    profiler.end("cold_load")
    output2_tt = emb_cold(token_ids_tt)
    output2 = to_torch_concat(output2_tt)

    # === Path 3: Warm Cache ===
    profiler.start("warm_load")
    emb_warm = TtParallelEmbedding(
        mesh_device,
        vocab_size,
        emb_dim,
        torch_weight=None,
        weight_cache_path=CACHE_DIR,
    )
    profiler.end("warm_load")
    output3_tt = emb_warm(token_ids_tt)
    output3 = to_torch_concat(output3_tt)

    # === Validation ===
    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"Embedding Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")

    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
