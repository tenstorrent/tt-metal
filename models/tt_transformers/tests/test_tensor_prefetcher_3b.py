# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tensor Prefetcher variant of ``test_mlp.py`` for Llama-3.2-3B on Blackhole.

This is a near-copy of ``test_mlp_inference`` from ``test_mlp.py`` with one change:

  ``Prefetcher`` is replaced with ``make_prefetcher``, which automatically selects
  ``TensorPrefetcher`` when the model, mesh, and firmware support programmable DRAM cores.

``num_receiver_cores`` is left to ``TensorPrefetcher``'s auto-pick: the divisibility +
L1 budget check in ``is_tensor_prefetcher_config_supported`` rules out ring sizes that don't
fit. For 3B on Blackhole the only supported rings are ring=32 (1-card) and ring=16 (2-card).

Requires HF weights for ``meta-llama/Llama-3.2-3B`` (or a compatible variant). Skips if
``HF_MODEL`` doesn't point to a 3B model. Runs both decode (weights via the GCB) and prefill
(weights read directly from DRAM by the matmul) so the recv-contig in1 read is gated on PCC.

Run via ``scripts/run_safe_pytest.sh``.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_blackhole
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.prefetcher import make_prefetcher

pytestmark = [
    run_for_blackhole("Tensor Prefetcher requires Blackhole"),
    pytest.mark.skipif(
        "Llama-3.2-3B" not in os.environ.get("HF_MODEL", ""),
        reason="HF_MODEL must point to Llama-3.2-3B for this test",
    ),
]


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(mesh_device):
    if not ttnn.experimental.is_tensor_prefetcher_supported(mesh_device):
        pytest.skip("Tensor prefetcher requires Blackhole firmware >= 19.12.0.0")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, seq_len",
    # Decode consumes the recv-contig weights via the GCB; prefill reads the same ND_SHARDED
    # weight directly from DRAM through the matmul's TensorAccessor. Cover both so the
    # prefill direct-read path (no GCB) is gated on PCC, not just the decode GCB path.
    [(Mode.DECODE, 32), (Mode.PREFILL, 128)],
    ids=["decode", "prefill"],
)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_inference_tensor_prefetcher(batch_size, mode, seq_len, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b

    # FF1/FF3/FF2 are the prefetched MLP weights. Let TensorPrefetcher auto-pick
    # the ring size from is_tensor_prefetcher_config_supported's divisibility + L1 budget.
    num_tensors = 3
    prefetcher = make_prefetcher(mesh_device, num_tensors=num_tensors, num_layers=1)
    assert (
        prefetcher.__class__.__name__ == "TensorPrefetcher"
    ), f"Expected TensorPrefetcher but got {prefetcher.__class__.__name__}; check model/device support."
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=128,
        cache_hf=True,
        prefetcher=prefetcher,
    )
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }
    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)

    tt_ccl = TT_CCL(mesh_device)
    tt_model = MLP(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
        prefetcher=prefetcher,
    )
    prefetcher.init(mode)

    # In decode, init() started the long-running DRISC daemon; guarantee teardown even if model
    # execution or result conversion below raises, so a failure can't poison later hardware tests
    # in this pytest worker with an already-active prefetcher.
    try:
        torch_input = torch.randn(
            1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
        )
        reference_output = reference_model(torch_input)

        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 3) if model_args.is_galaxy else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
            dtype=ttnn.bfloat8_b,
            memory_config=model_args.get_mlp_input_mem_config(mode, prefetcher),
            layout=ttnn.TILE_LAYOUT,
        )
        logger.info(f"Run MLP through Tensor Prefetcher: ring={prefetcher.ring_size}")
        tt_output = tt_model(tt_input, mode)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_output_torch = tt_output_torch[:, :1, :, :]

        pcc_required = 0.99
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
    finally:
        prefetcher.teardown()
    assert passing, f"MLP PCC failed (ring={prefetcher.ring_size}): {pcc_message}"
