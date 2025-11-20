# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.prefetcher import Prefetcher, PrefetcherCoreConfig

PREFETCHER_NOC1_GRID = [
    (6, 6),
    (6, 7),
    (6, 9),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 4),
    (6, 5),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 9),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 4),
    (1, 4),
    (1, 5),
    (1, 9),
    (1, 0),
    (2, 0),
    (2, 4),
    (2, 5),
    (2, 9),
]


def get_core_ranges(num_reader_cores, num_global_cb_receivers, is_functional_test):
    """
    Helper function to get all the relevant core ranges for dram prefetcher + matmul configuration
    """

    all_dram_cores = [ttnn.CoreCoord(idx, 0) for idx in range(12)]  # 12 DRAM banks

    all_sender_cores = [
        ttnn.CoreCoord(0, 9),
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(0, 4),
        ttnn.CoreCoord(0, 5),
        ttnn.CoreCoord(4, 0),
        ttnn.CoreCoord(4, 9),
        ttnn.CoreCoord(4, 1),
        ttnn.CoreCoord(4, 7),
        ttnn.CoreCoord(4, 6),
        ttnn.CoreCoord(4, 2),
        ttnn.CoreCoord(4, 4),
        ttnn.CoreCoord(4, 5),
    ]
    dummy_sender_cores = [
        ttnn.CoreCoord(0, 1),
        ttnn.CoreCoord(0, 2),
        ttnn.CoreCoord(0, 3),
        ttnn.CoreCoord(0, 6),
        ttnn.CoreCoord(0, 7),
        ttnn.CoreCoord(0, 8),
        ttnn.CoreCoord(4, 3),
        ttnn.CoreCoord(4, 8),
    ]

    all_receiver_cores_list = [
        (1, 9),
        (2, 9),
        (1, 0),
        (2, 0),
        (1, 4),
        (2, 4),
        (1, 5),
        (2, 5),
        (5, 0),
        (6, 0),
        (5, 9),
        (6, 9),
        (5, 1),
        (6, 1),
        (5, 7),
        (6, 7),
        (5, 6),
        (6, 6),
        (5, 2),
        (6, 2),
        (5, 4),
        (6, 4),
        (5, 5),
        (6, 5),
    ]

    all_receiver_cores = [
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(*all_receiver_cores_list[idx]),
                    ttnn.CoreCoord(*all_receiver_cores_list[idx + 1 if num_global_cb_receivers == 2 else idx]),
                ),
            ]
        )
        for idx in range(0, len(all_receiver_cores_list), 2)
    ]

    dummy_receiver_cores = [
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(3, 0),
                    ttnn.CoreCoord(3, 0),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 1),
                    ttnn.CoreCoord(3, 1),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 2),
                    ttnn.CoreCoord(3, 2),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 3),
                    ttnn.CoreCoord(3, 3),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(3, 4),
                    ttnn.CoreCoord(3, 4),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(3, 5),
                    ttnn.CoreCoord(3, 5),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 6),
                    ttnn.CoreCoord(3, 6),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 7),
                    ttnn.CoreCoord(3, 7),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 8),
                    ttnn.CoreCoord(3, 8),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(3, 9),
                    ttnn.CoreCoord(3, 9),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 3),
                    ttnn.CoreCoord(6, 3),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 8),
                    ttnn.CoreCoord(6, 8),
                ),
            ]
        ),
    ]

    dram_cores = all_dram_cores[:num_reader_cores]
    hop_grid = []
    mm_optimised_ring_cores = []
    if not is_functional_test:
        sender_cores = all_sender_cores
        active_sender_cores = all_sender_cores[:num_reader_cores]
        sender_cores.extend(dummy_sender_cores)
        active_receiver_cores_list = all_receiver_cores_list[: num_reader_cores * num_global_cb_receivers]
        receiver_cores = all_receiver_cores
        receiver_cores.extend(dummy_receiver_cores)

        worker_cores_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            ]
        )

        mm_optimised_ring_cores = PREFETCHER_NOC1_GRID
        hop_grid = [
            (3, 6),
        ]
    else:
        sender_cores = [ttnn.CoreCoord(0, y) for y in range(num_reader_cores)]
        active_sender_cores = sender_cores

        active_receiver_cores_list = [
            (x, y) for y in range(num_reader_cores) for x in range(1, num_global_cb_receivers + 1)
        ]

        receiver_cores = [
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(*active_receiver_cores_list[idx * num_global_cb_receivers]),
                        ttnn.CoreCoord(*active_receiver_cores_list[(idx + 1) * num_global_cb_receivers - 1]),
                    )
                ]
            )
            for idx in range(num_reader_cores)
        ]

        worker_cores_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1)),
            ]
        )

    return (
        active_sender_cores,
        dram_cores,
        sender_cores,
        active_receiver_cores_list,
        receiver_cores,
        worker_cores_range_set,
        mm_optimised_ring_cores,
        hop_grid,
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "use_prefetcher",
    (True, False),
)
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
    "seq_len",
    (64 * 1024, 32 * 1024, 512, 32),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mlp_inference(seq_len, batch_size, mesh_device, reset_seeds, ensure_gc, use_prefetcher):
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    num_receiver_cores = 4

    prefetcher_core_config = PrefetcherCoreConfig(num_receiver_cores=num_receiver_cores, mesh_device=mesh_device)

    (
        active_sender_cores,
        dram_cores,
        sender_cores,
        active_receiver_cores_list,
        receiver_cores,
        worker_cores_range_set,
        mm_optimised_ring_cores,
        hop_grid,
    ) = get_core_ranges(num_reader_cores=12, num_global_cb_receivers=2, is_functional_test=False)

    prefetcher = (
        Prefetcher(mesh_device, num_tensors=3, num_receiver_cores=num_receiver_cores, num_layers=1, mode=mode)
        if use_prefetcher
        else None
    )

    model_args = ModelArgs(
        mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True, prefetcher=prefetcher
    )
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)
    if model_args.is_90b:
        # float32 ~3x faster than bfloat16. Also LLAMA_DIR uses float32
        # bfloat16 fails on CI (32k and 64k seq_len) with "This test seems to have hung... Timing out test case"
        reference_model.to(torch.float32)

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

    # Run prefetcher if it is used
    if prefetcher is not None:
        prefetcher.run()

    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
    )
    breakpoint()
    reference_output = reference_model(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3) if model_args.is_galaxy else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat8_b,
        memory_config=(
            (
                tt_model.model_config["MLP_ACT_MEMCFG"]
                if model_args.is_galaxy
                else (
                    model_args.model_config["SHARDED_MLP_INPUT_RING_MEMCFG"]
                    if prefetcher is not None
                    else model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
                )
            )
            if mode == "decode"
            else ttnn.DRAM_MEMORY_CONFIG
        ),
        layout=ttnn.TILE_LAYOUT,
    )
    breakpoint()
    logger.info("Run MLP")
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
    if passing:
        logger.info("MLP Passed!")
    else:
        logger.warning("MLP Failed!")

    assert passing, f"MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
