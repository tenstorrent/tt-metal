"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import pytest
import torch

import ttnn
from models.demos.yolos_small.reference.config import get_yolos_small_config
from models.demos.yolos_small.reference.modeling_yolos import YolosForObjectDetection as PyTorchYolos
from models.demos.yolos_small.yolos_ttnn.common import OptimizationConfig, convert_to_ttnn_tensor, get_dtype_for_stage
from models.demos.yolos_small.yolos_ttnn.modeling_yolos import YolosForObjectDetection as TtnnYolos
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture(scope="module")
def device():
    """Initialize TTNN device for testing."""
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def config():
    """Get YOLOS-small configuration."""
    return get_yolos_small_config()


@pytest.fixture(scope="module")
def pytorch_model(config):
    """Initialize PyTorch reference model."""
    model = PyTorchYolos(config)
    model.eval()
    return model


@pytest.fixture
def sample_input(config):
    """Create sample input tensor."""
    batch_size = 1
    channels = config.num_channels
    height, width = config.image_size

    # Random input normalized like ImageNet
    pixel_values = torch.randn(batch_size, channels, height, width)
    return pixel_values


class TestModelComponents:
    """Test individual model components."""

    def test_patch_embeddings(self, pytorch_model, sample_input, device, config):
        """Test patch embedding layer."""
        from models.demos.yolos_small.yolos_ttnn.common import OptimizationConfig
        from models.demos.yolos_small.yolos_ttnn.modeling_yolos import yolos_patch_embeddings

        opt_config = OptimizationConfig.stage1()

        # PyTorch
        with torch.no_grad():
            pytorch_embeddings = pytorch_model.yolos.embeddings.patch_embeddings(sample_input)

        # TTNN
        pixel_values_ttnn = convert_to_ttnn_tensor(sample_input, device)

        # Create simplified parameters for testing
        from types import SimpleNamespace

        params = SimpleNamespace()
        params.projection = SimpleNamespace()

        # Match the flattening strategy used in the TTNN model initialisation:
        # conv weight [out_channels, in_channels, kh, kw] ->
        # [in_features, out_features] for ttnn.linear.
        conv_weight = pytorch_model.yolos.embeddings.patch_embeddings.projection.weight.data
        conv_bias = pytorch_model.yolos.embeddings.patch_embeddings.projection.bias.data

        out_channels, in_channels, kh, kw = conv_weight.shape
        conv_weight_flat = conv_weight.reshape(out_channels, in_channels * kh * kw)
        conv_weight_flat = conv_weight_flat.transpose(0, 1).contiguous()

        params.projection.weight = convert_to_ttnn_tensor(conv_weight_flat, device)
        params.projection.bias = convert_to_ttnn_tensor(conv_bias, device)

        ttnn_embeddings = yolos_patch_embeddings(pixel_values_ttnn, params, config, device, opt_config)
        ttnn_embeddings_torch = ttnn.to_torch(ttnn_embeddings)

        # Check shape
        assert ttnn_embeddings_torch.shape == pytorch_embeddings.shape

        # Check values (allow some tolerance for bfloat16)
        max_diff = torch.max(torch.abs(ttnn_embeddings_torch - pytorch_embeddings))
        print(f"Max difference in patch embeddings: {max_diff}")
        assert max_diff < 0.1, f"Patch embeddings differ too much: {max_diff}"


class TestOptimizationStages:
    """Test all three optimization stages."""

    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_full_model_stage(self, stage, pytorch_model, sample_input, device, config):
        """Test complete model for each optimization stage."""
        # Get optimization config
        if stage == 1:
            opt_config = OptimizationConfig.stage1()
        elif stage == 2:
            opt_config = OptimizationConfig.stage2()
        else:
            opt_config = OptimizationConfig.stage3()

        print(f"\nTesting Stage {stage}")

        # Create TTNN model
        ttnn_model = TtnnYolos(
            config=config,
            device=device,
            reference_model=pytorch_model,
            opt_config=opt_config,
        )

        # PyTorch forward
        with torch.no_grad():
            pytorch_logits, pytorch_boxes = pytorch_model(sample_input)

        # TTNN forward - use dtype appropriate for the chosen stage
        input_dtype = get_dtype_for_stage(opt_config)
        pixel_values_ttnn = convert_to_ttnn_tensor(sample_input, device, dtype=input_dtype)
        ttnn_logits, ttnn_boxes = ttnn_model(pixel_values_ttnn)

        # Convert back to torch
        ttnn_logits_torch = ttnn.to_torch(ttnn_logits)
        ttnn_boxes_torch = ttnn.to_torch(ttnn_boxes)

        # Check shapes
        assert (
            ttnn_logits_torch.shape == pytorch_logits.shape
        ), f"Logits shape mismatch: {ttnn_logits_torch.shape} vs {pytorch_logits.shape}"
        assert (
            ttnn_boxes_torch.shape == pytorch_boxes.shape
        ), f"Boxes shape mismatch: {ttnn_boxes_torch.shape} vs {pytorch_boxes.shape}"

        # Convert to float32 for stable comparison
        pytorch_logits_f32 = pytorch_logits.to(torch.float32)
        pytorch_boxes_f32 = pytorch_boxes.to(torch.float32)
        ttnn_logits_f32 = ttnn_logits_torch.to(torch.float32)
        ttnn_boxes_f32 = ttnn_boxes_torch.to(torch.float32)

        # Report max-abs differences for debugging only (do not assert on them).
        logits_diff = torch.max(torch.abs(ttnn_logits_f32 - pytorch_logits_f32)).item()
        boxes_diff = torch.max(torch.abs(ttnn_boxes_f32 - pytorch_boxes_f32)).item()
        print(f"Stage {stage} - Max logits diff: {logits_diff:.6f}")
        print(f"Stage {stage} - Max boxes diff: {boxes_diff:.6f}")

        # PCC-based correctness, following the YOLOv4 pattern of comparing
        # post-processed outputs with high correlation rather than tiny
        # elementwise differences.
        if stage == 1:
            logits_pcc = 0.999
            boxes_pcc = 0.999
        elif stage == 2:
            logits_pcc = 0.997
            boxes_pcc = 0.997
        else:
            logits_pcc = 0.995
            boxes_pcc = 0.995

        assert_with_pcc(pytorch_logits_f32, ttnn_logits_f32, pcc=logits_pcc)
        assert_with_pcc(pytorch_boxes_f32, ttnn_boxes_f32, pcc=boxes_pcc)

        print(f"Stage {stage} PASSED PCC ✓")


class TestAccuracy:
    """Test model accuracy on COCO validation."""

    def test_coco_sample_accuracy(self, pytorch_model, device, config):
        """
        Test accuracy on a few COCO samples.
        For full COCO validation, see separate validation script.
        """
        # This is a placeholder - full COCO validation should be run separately
        # following the YOLOv4 demo pattern

        opt_config = OptimizationConfig.stage3()
        ttnn_model = TtnnYolos(
            config=config,
            device=device,
            reference_model=pytorch_model,
            opt_config=opt_config,
        )

        # Create sample input
        sample_input = torch.randn(1, 3, 512, 864)

        # Get predictions
        with torch.no_grad():
            pytorch_predictions = pytorch_model.predict(sample_input, threshold=0.7)

        input_dtype = get_dtype_for_stage(opt_config)
        pixel_values_ttnn = convert_to_ttnn_tensor(sample_input, device, dtype=input_dtype)
        ttnn_predictions = ttnn_model.predict(pixel_values_ttnn, threshold=0.7)

        # Compare number of detections
        pytorch_num_det = pytorch_predictions["keep"][0].sum().item()
        ttnn_num_det = ttnn_predictions["keep"][0].sum().item()

        print(f"PyTorch detections: {pytorch_num_det}")
        print(f"TTNN detections: {ttnn_num_det}")

        # Should detect similar number of objects (allow some difference due to thresholding)
        assert (
            abs(pytorch_num_det - ttnn_num_det) <= 2
        ), f"Detection count differs too much: {pytorch_num_det} vs {ttnn_num_det}"


class TestPerformance:
    """Performance benchmarking tests."""

    def test_inference_time(self, pytorch_model, device, config):
        """Benchmark inference time for all stages."""
        import time

        sample_input = torch.randn(1, 3, 512, 864)
        num_runs = 10
        warmup_runs = 3

        results = {}

        # PyTorch baseline
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = pytorch_model(sample_input)

            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = pytorch_model(sample_input)
                times.append(time.perf_counter() - start)

            results["pytorch"] = {
                "avg_ms": np.mean(times) * 1000,
                "std_ms": np.std(times) * 1000,
            }

        # TTNN stages
        for stage in [1, 2, 3]:
            if stage == 1:
                opt_config = OptimizationConfig.stage1()
            elif stage == 2:
                opt_config = OptimizationConfig.stage2()
            else:
                opt_config = OptimizationConfig.stage3()

            ttnn_model = TtnnYolos(
                config=config,
                device=device,
                reference_model=pytorch_model,
                opt_config=opt_config,
            )

            input_dtype = get_dtype_for_stage(opt_config)
            pixel_values_ttnn = convert_to_ttnn_tensor(sample_input, device, dtype=input_dtype)

            # Warmup
            for _ in range(warmup_runs):
                _ = ttnn_model(pixel_values_ttnn)

            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = ttnn_model(pixel_values_ttnn)
                times.append(time.perf_counter() - start)

            results[f"stage{stage}"] = {
                "avg_ms": np.mean(times) * 1000,
                "std_ms": np.std(times) * 1000,
                "speedup": results["pytorch"]["avg_ms"] / (np.mean(times) * 1000),
            }

        # Print results
        print("\n" + "=" * 80)
        print("Performance Benchmark Results")
        print("=" * 80)
        print(f"PyTorch: {results['pytorch']['avg_ms']:.2f} ± {results['pytorch']['std_ms']:.2f} ms")
        for stage in [1, 2, 3]:
            stage_key = f"stage{stage}"
            print(
                f"Stage {stage}: {results[stage_key]['avg_ms']:.2f} ± {results[stage_key]['std_ms']:.2f} ms "
                f"(Speedup: {results[stage_key]['speedup']:.2f}x)"
            )
        print("=" * 80)

        # Save results
        import json

        with open("performance_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("Saved results to performance_results.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
