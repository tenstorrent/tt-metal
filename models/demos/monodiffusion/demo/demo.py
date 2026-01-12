# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MonoDiffusion Demo Script
Run monocular depth estimation on sample images
"""

import pytest
import torch
import ttnn
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from models.demos.monodiffusion.tt import (
    create_monodiffusion_from_parameters,
    create_monodiffusion_preprocessor,
    load_reference_model,
)


def load_sample_image(image_path: str, target_size: tuple = (192, 640)) -> torch.Tensor:
    """
    Load and preprocess sample image
    Following vanilla_unet demo pattern

    Args:
        image_path: Path to input image
        target_size: (height, width) for resizing

    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """
    img = Image.open(image_path).convert('RGB')

    # Resize
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor


def visualize_results(
    input_image: torch.Tensor,
    depth_map: torch.Tensor,
    uncertainty_map: torch.Tensor,
    save_path: str
):
    """
    Visualize depth and uncertainty predictions
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input image
    img = input_image[0].permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Depth map
    depth = depth_map[0, 0].cpu().numpy()
    im1 = axes[1].imshow(depth, cmap='plasma')
    axes[1].set_title('Predicted Depth')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Uncertainty map
    if uncertainty_map is not None:
        uncertainty = uncertainty_map[0, 0].cpu().numpy()
        im2 = axes[2].imshow(uncertainty, cmap='hot')
        axes[2].set_title('Uncertainty')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


@pytest.mark.parametrize(
    "device_id",
    [0],
)
def test_monodiffusion_demo_single_image(device_id):
    """
    Test MonoDiffusion on a single image
    """
    # Setup paths
    demo_dir = Path(__file__).parent
    image_dir = demo_dir / "images"
    output_dir = demo_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Create sample image if not exists
    sample_image_path = image_dir / "sample.png"
    if not sample_image_path.exists():
        image_dir.mkdir(exist_ok=True)
        # Create a dummy image for testing
        dummy_img = np.random.randint(0, 255, (192, 640, 3), dtype=np.uint8)
        Image.fromarray(dummy_img).save(sample_image_path)
        print(f"Created dummy sample image at {sample_image_path}")

    # Initialize device
    device = ttnn.open_device(device_id=device_id)

    try:
        # Load reference model and create preprocessor
        print("Loading reference model...")
        reference_model = load_reference_model()
        preprocessor = create_monodiffusion_preprocessor(device)

        # Preprocess weights
        print("Preprocessing weights...")
        parameters = preprocessor(reference_model, "monodiffusion", {})

        # Create TT model
        print("Creating MonoDiffusion model...")
        model = create_monodiffusion_from_parameters(
            parameters=parameters,
            device=device,
            batch_size=1,
            input_height=192,
            input_width=640,
        )

        # Load sample image
        print(f"Loading image from {sample_image_path}...")
        input_image = load_sample_image(str(sample_image_path), target_size=(192, 640))

        # Convert to TTNN tensor
        input_tensor = ttnn.from_torch(
            input_image,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Run inference
        print("Running inference...")
        depth_ttnn, uncertainty_ttnn = model(input_tensor, return_uncertainty=True)

        # Convert to torch
        depth_map = ttnn.to_torch(depth_ttnn)
        uncertainty_map = ttnn.to_torch(uncertainty_ttnn) if uncertainty_ttnn is not None else None

        # Apply sigmoid and scale depth
        depth_map = torch.sigmoid(depth_map)
        depth_map = 0.1 + (100.0 - 0.1) * depth_map

        # Visualize
        output_path = output_dir / "result_ttnn_1.png"
        visualize_results(input_image, depth_map, uncertainty_map, str(output_path))

        print(f"✓ Demo completed successfully!")
        print(f"  - Depth map shape: {depth_map.shape}")
        print(f"  - Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
        if uncertainty_map is not None:
            print(f"  - Uncertainty range: [{uncertainty_map.min():.4f}, {uncertainty_map.max():.4f}]")

    finally:
        ttnn.close_device(device)


@pytest.mark.parametrize(
    "device_id",
    [0],
)
def test_monodiffusion_demo_batch(device_id):
    """
    Test MonoDiffusion on multiple images (batch processing)
    """
    demo_dir = Path(__file__).parent
    image_dir = demo_dir / "images"
    output_dir = demo_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    device = ttnn.open_device(device_id=device_id)

    try:
        model = create_monodiffusion_model(device, config_type="kitti")

        # Process all images in directory
        image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

        for i, image_path in enumerate(image_files[:5]):  # Process up to 5 images
            print(f"\nProcessing {image_path.name}...")

            input_image = load_sample_image(str(image_path), target_size=(192, 640))
            input_tensor = preprocess_input_image(input_image, device, 192, 640)

            depth_ttnn, uncertainty_ttnn = model(input_tensor, return_uncertainty=True)

            depth_map = postprocess_depth_map(depth_ttnn)
            uncertainty_map = ttnn.to_torch(uncertainty_ttnn) if uncertainty_ttnn is not None else None

            output_path = output_dir / f"result_{i+1}.png"
            visualize_results(input_image, depth_map, uncertainty_map, str(output_path))

        print(f"\n✓ Batch processing completed!")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    # Run demo directly
    test_monodiffusion_demo_single_image(device_id=0)
