# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os

import pytest
import torch
from loguru import logger

# Import from local reference files instead of HuggingFace
from torch.nn import Embedding as EmbeddingReference

import ttnn
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.tt.embedding.embedding1d import Embedding1D
from models.demos.deepseek_v3.tt.embedding.embedding2d import Embedding2D
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    load_reference_io_tensors_for_module,
    run_module_forward,
)

TEST_CHECK_ITERS = 100
CI_ACTIVE = os.getenv("CI") == "true"
_CI_SKIP_MARK = pytest.mark.skipif(
    CI_ACTIVE,
    reason="CI runs traced coverage only.",
)


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 10485760},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "EmbeddingClass,mode,batch_size_or_seq_len",
    [
        pytest.param(Embedding1D, "decode", 32, marks=pytest.mark.requires_device(["TG"])),
        pytest.param(Embedding2D, "decode", 128, marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"])),
    ]
    + [
        pytest.param(Embedding1D, "prefill", seq_len, marks=pytest.mark.requires_device(["TG"]))
        if seq_len == 128
        else pytest.param(
            Embedding1D,
            "prefill",
            seq_len,
            marks=[
                pytest.mark.requires_device(["TG"]),
                pytest.mark.skipif(
                    CI_ACTIVE,
                    reason=(
                        f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
                    ),
                ),
            ],
        )
        for seq_len in PREFILL_SEQ_LENS
    ]
    + [
        pytest.param(Embedding2D, "prefill", seq_len, marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"]))
        if seq_len == 128
        else pytest.param(
            Embedding2D,
            "prefill",
            seq_len,
            marks=[
                pytest.mark.requires_device(["TG", "DUAL", "QUAD"]),
                pytest.mark.skipif(
                    CI_ACTIVE,
                    reason=(
                        f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
                    ),
                ),
            ],
        )
        for seq_len in PREFILL_SEQ_LENS
    ],
)
@pytest.mark.parametrize(
    "trace_mode",
    [
        pytest.param(False, marks=_CI_SKIP_MARK, id="eager"),
        pytest.param(True, id="tracing"),
    ],
)
@pytest.mark.parametrize(
    "use_real_weights",
    [
        pytest.param(True, id="real_weights"),
        pytest.param(False, marks=_CI_SKIP_MARK, id="random_weights"),
    ],
)
@pytest.mark.parametrize(
    "generate_reference_io",
    [True, False],
)
def test_embedding_forward_pass(
    EmbeddingClass,
    hf_config,
    mode,
    batch_size_or_seq_len,
    trace_mode,
    use_real_weights,
    generate_reference_io,
    mesh_device,
    ccl,
    model_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict,
):
    if trace_mode and mode != "decode":
        pytest.skip("Tracing is only supported for decode mode.")

    logger.info("Setting up reference IO")
    module_path = "model.embed_tokens"

    if not use_real_weights:
        generate_reference_io = True

    if generate_reference_io:
        reference_model = EmbeddingReference(
            hf_config.vocab_size,
            hf_config.hidden_size,
            hf_config.pad_token_id,
        ).eval()
        if use_real_weights:
            state_dict = reference_model.state_dict()
        else:
            random_state_dict = {}
            for name, tensor in reference_model.state_dict().items():
                if torch.is_floating_point(tensor):
                    random_state_dict[name] = torch.randn_like(tensor) * 0.02
                else:
                    random_state_dict[name] = torch.zeros_like(tensor)
            reference_model.load_state_dict(random_state_dict)
            state_dict = random_state_dict

        torch_input = torch.randint(0, hf_config.vocab_size, (1, 1, batch_size_or_seq_len))
        reference_output = reference_model(torch_input)

    else:
        state_dict = sub_state_dict(state_dict, module_path + ".")
        try:
            torch_input, reference_output = load_reference_io_tensors_for_module(
                mode, module_path, batch_size_or_seq_len, 1
            )
        except FileNotFoundError as exc:
            pytest.skip(str(exc))

    # Generate module configs and state
    logger.info("Setting up TTNN configs")
    weight_cache_root = cache_path if use_real_weights else cache_path / "random_weights"
    weight_config = get_test_weight_config(
        EmbeddingClass,
        hf_config,
        (state_dict,),
        weight_cache_root,
        mesh_device,
        force_recalculate_weight_config or not use_real_weights,
    )
    model_config = get_model_config(EmbeddingClass, mode, hf_config, mesh_device)
    model_state = EmbeddingClass.create_state(hf_config, mesh_device, ccl)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    logger.info("Preparing TTNN inputs")
    tt_input_ids = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def to_torch_output(tt_output: ttnn.Tensor) -> torch.Tensor:
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device,
                dims=(0, -1),
                mesh_shape=tuple(mesh_device.shape),
            ),
        )
        if EmbeddingClass is Embedding1D:
            tt_output_torch = tt_output_torch[:1]
        else:
            tt_output_torch = tt_output_torch.reshape(1, batch_size_or_seq_len, hf_config.hidden_size)
        return tt_output_torch

    # TTNN forward pass
    logger.info("Running TTNN forward pass")
    if trace_mode:
        tt_output = run_module_forward(EmbeddingClass, mode, tt_input_ids, run_config)
        ttnn.synchronize_device(mesh_device)
        tt_output_torch = to_torch_output(tt_output)
        assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)
        ttnn.deallocate(tt_output)

        # Reset CCL semaphore counters before trace capture
        ccl.reset_sem_counters()

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = run_module_forward(EmbeddingClass, mode, tt_input_ids, run_config)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for _ in range(TEST_CHECK_ITERS - 1):
            ttnn.execute_trace(mesh_device, trace_id, blocking=True)
        ttnn.synchronize_device(mesh_device)

        tt_output_torch = to_torch_output(trace_output)
        assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(trace_output)
    else:
        for iter_idx in range(TEST_CHECK_ITERS):
            tt_output = run_module_forward(EmbeddingClass, mode, tt_input_ids, run_config)
            ttnn.synchronize_device(mesh_device)
            if iter_idx in (0, TEST_CHECK_ITERS - 1):
                tt_output_torch = to_torch_output(tt_output)
                assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)
            ttnn.deallocate(tt_output)

    # Cleanup
    ttnn.deallocate(tt_input_ids)


if __name__ == "__main__":
    pytest.main([__file__])
