# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from models.experimental.bevformer.tt.tt_ms_deformable_attention import TTMSDeformableAttention
from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention

from models.experimental.bevformer.config import DeformableAttentionConfig

from models.experimental.bevformer.tests.test_utils import (
    print_detailed_comparison,
    check_with_tolerances,
    check_with_pcc,  # Legacy compatibility
    save_comparison_report,
)

from models.experimental.bevformer.tt.model_preprocessing import create_ms_deformable_attention_parameters

from loguru import logger


@pytest.mark.parametrize(
    "batch_size, num_query, embed_dims, num_cams, num_levels, num_points",
    [
        (1, 200 * 200, 256, 6, 4, 4),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10 * 1024}], indirect=True)
@pytest.mark.parametrize("seed", [0])
def test_ms_deformable_attention_forward(
    device,
    batch_size,
    num_query,
    embed_dims,
    num_cams,
    num_levels,
    num_points,
    seed,
):
    torch.manual_seed(seed)

    spatial_shapes = torch.tensor([[200, 113], [100, 57], [50, 29], [25, 15]], dtype=torch.long)
    total_key_length = [h * w for h, w in spatial_shapes.tolist()]

    query = torch.randn(batch_size, num_query, embed_dims, dtype=torch.float32)
    key = torch.randn(batch_size, sum(total_key_length), embed_dims, dtype=torch.float32)
    value = torch.randn(batch_size, sum(total_key_length), embed_dims, dtype=torch.float32)

    reference_points = torch.rand(batch_size, num_query, num_levels, 2, dtype=torch.float32)

    indices = spatial_shapes.prod(1).cumsum(0)
    level_start_index = torch.cat([torch.tensor([0], dtype=torch.long), indices[:-1]], 0)

    config = DeformableAttentionConfig(
        embed_dims=embed_dims,
        num_heads=4,
        num_levels=num_levels,
        num_points=num_points,
    )

    # Create PyTorch reference model
    ref_model = MSDeformableAttention(config)
    ref_model.eval()

    # Create preprocessed parameters from PyTorch model
    tt_parameters = create_ms_deformable_attention_parameters(
        torch_model=ref_model,
        device=device,
        config=config,
        dtype=ttnn.float32,
    )

    # Create ttnn model with preprocessed parameters
    tt_model = TTMSDeformableAttention(
        config=config,
        device=device,
        params=tt_parameters,
    )

    ref_model_output = ref_model(
        query,
        key,
        value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
    )

    tt_model_output = tt_model(
        query=query,
        key=key,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
    )

    # Comprehensive comparison using enhanced test utilities
    logger.info(f"Reference model output type: {type(ref_model_output)}, shape: {ref_model_output.shape}")
    logger.info(f"TT model output type: {type(tt_model_output)}")

    # Print detailed statistical comparison
    print_detailed_comparison(
        ref_model_output,
        tt_model_output,
        tensor_name="ms_deformable_attention_output",
        show_histograms=False,  # Set to True for even more detailed analysis
    )

    # Comprehensive tolerance checking with multiple criteria
    passed, results = check_with_tolerances(
        ref_model_output,
        tt_model_output,
        pcc_threshold=0.99,
        abs_error_threshold=1e-1,
        rel_error_threshold=0.05,
        max_error_ratio=0.01,
        tensor_name="ms_deformable_attention_output",
    )

    # Assert that the comprehensive check passes
    assert passed, f"Comprehensive tolerance check failed. Results: {results['individual_checks']}"

    logger.info("✅ All tolerance checks passed successfully!")

    # Optional: Save detailed report if needed (uncomment for debugging)
    # from pathlib import Path
    # report_path = Path("ms_deformable_attention_comparison_report.txt")
    # save_comparison_report(ref_model_output, tt_model_output, "ms_deformable_attention_output", str(report_path))
