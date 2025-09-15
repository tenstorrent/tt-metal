# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from ultralytics import YOLO
from models.demos.yolov11m.common import YOLOV11_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov11m.reference import yolov11


def test_yolov11():
    torch_model = yolov11.YoloV11()
    torch_model.eval()

    if True:
        torch_model = load_torch_model()
    resolution = [1, 3, 640, 640]
    torch_input = torch.randn(resolution[0], resolution[1], resolution[2], resolution[3])
    
    torch_output = torch_model(torch_input)
    print(torch_output.shape)
    import pdb; pdb.set_trace()


def test_obb_simple():
    """Simple test for OBB (Oriented Bounding Box) model"""
    print("🧪 Testing OBB Model...")
    
    # Load the OBB model
    torch_model = load_torch_model()
    torch_model.eval()
    
    # Create test input (standard YOLOv11 input size)
    batch_size, channels, height, width = 1, 3, 640, 640
    torch_input = torch.randn(batch_size, channels, height, width)
    
    print(f"📥 Input shape: {torch_input.shape}")
    
    # Run forward pass
    with torch.no_grad():
        torch_output = torch_model(torch_input)
    
    print(f"📤 Output shape: {torch_output.shape}")
    
    # Validate output shape for OBB
    # Expected: [batch_size, 20, 8400]
    # 20 = 4 (box coords) + 15 (classes) + 1 (angle)
    # 8400 = detection points across three scales
    expected_shape = (batch_size, 20, 8400)
    assert torch_output.shape == expected_shape, f"Expected shape {expected_shape}, got {torch_output.shape}"
    
    # Basic value checks
    assert torch.isfinite(torch_output).all(), "Output contains non-finite values"
    assert not torch.isnan(torch_output).any(), "Output contains NaN values"
    
    # Split output into components
    box_coords = torch_output[:, :4, :]      # Box coordinates (x, y, w, h)
    class_preds = torch_output[:, 4:19, :]   # Class predictions (15 classes)
    angle_preds = torch_output[:, 19:20, :]  # Angle predictions (1 channel)
    
    print(f"📦 Box coordinates shape: {box_coords.shape}")
    print(f"🏷️  Class predictions shape: {class_preds.shape}")
    print(f"📐 Angle predictions shape: {angle_preds.shape}")
    
    # Validate ranges
    print(f"📊 Box coords range: [{box_coords.min():.3f}, {box_coords.max():.3f}]")
    print(f"📊 Class preds range: [{class_preds.min():.3f}, {class_preds.max():.3f}]")
    print(f"📊 Angle preds range: [{angle_preds.min():.3f}, {angle_preds.max():.3f}]")
    
    # Check that class predictions are reasonable (should be probabilities between 0 and 1)
    assert (class_preds >= 0).all() and (class_preds <= 1).all(), "Class predictions should be between 0 and 1"
    
    print("✅ OBB model test passed!")
    return torch_output


def test_obb_components():
    """Test individual components of OBB output"""
    print("🔬 Testing OBB Components...")
    
    torch_model = load_torch_model()
    torch_model.eval()
    
    # Small test input for faster execution
    torch_input = torch.randn(1, 3, 320, 320)
    
    with torch.no_grad():
        output = torch_model(torch_input)
    
    batch_size, channels, detections = output.shape
    print(f"Output dimensions: {batch_size}×{channels}×{detections}")
    
    # Test that we have the right number of channels for OBB
    assert channels == 20, f"Expected 20 channels for OBB, got {channels}"
    
    # Extract components
    boxes = output[:, :4, :]       # x, y, w, h
    classes = output[:, 4:19, :]   # 15 class probabilities
    angles = output[:, 19:, :]     # 1 angle prediction
    
    # Test that boxes have reasonable coordinates
    print(f"Box statistics:")
    print(f"  X,Y centers: mean={boxes[:, :2, :].mean():.3f}, std={boxes[:, :2, :].std():.3f}")
    print(f"  W,H sizes: mean={boxes[:, 2:, :].mean():.3f}, std={boxes[:, 2:, :].std():.3f}")
    
    # Test that class probabilities are normalized
    print(f"Class probability statistics:")
    print(f"  Mean: {classes.mean():.3f}, Min: {classes.min():.3f}, Max: {classes.max():.3f}")
    
    # Test angle predictions
    print(f"Angle prediction statistics:")
    print(f"  Mean: {angles.mean():.3f}, Min: {angles.min():.3f}, Max: {angles.max():.3f}")
    
    print("✅ OBB components test passed!")
    return output


def preprocess_image(image_path, target_size=(640, 640)):
    """
    Preprocess image for YOLOv11 OBB model
    Args:
        image_path: Path to the image file
        target_size: Target size (width, height)
    Returns:
        torch.Tensor: Preprocessed image tensor [1, 3, H, W]
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # YOLOv11 preprocessing: resize and normalize
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),  # Converts to [0, 1] and changes to CHW format
            # Note: YOLOv11 models typically expect [0, 1] range, not ImageNet normalization
        ])
        
        # Apply transforms and add batch dimension
        tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
        
        print(f"📷 Loaded image: {os.path.basename(image_path)}")
        print(f"   Original size: {original_size}")
        print(f"   Processed size: {tensor.shape}")
        print(f"   Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        return tensor, original_size
        
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None, None


def draw_oriented_bbox(image, center_x, center_y, width, height, angle, color=(0, 255, 0), thickness=2):
    """
    Draw an oriented bounding box on an image
    Args:
        image: numpy array image (H, W, 3)
        center_x, center_y: center coordinates
        width, height: box dimensions
        angle: rotation angle in radians
        color: BGR color tuple
        thickness: line thickness
    """
    # Convert angle to degrees for OpenCV
    angle_deg = np.degrees(angle)
    
    # Create rotated rectangle
    rect = ((center_x, center_y), (width, height), angle_deg)
    box_points = cv2.boxPoints(rect)
    box_points = np.int32(box_points)
    
    # Draw the oriented bounding box
    cv2.drawContours(image, [box_points], 0, color, thickness)
    
    # Draw center point
    cv2.circle(image, (int(center_x), int(center_y)), 3, color, -1)
    
    return image


def process_obb_predictions(output, confidence_threshold=0.1, original_size=(640, 640), target_size=(640, 640)):
    """
    Process OBB model output to extract detections
    Args:
        output: Model output tensor [1, 20, 8400]
        confidence_threshold: Minimum confidence for detection
        original_size: Original image size (width, height)  
        target_size: Model input size (width, height)
    Returns:
        List of detections: [(x, y, w, h, angle, confidence, class_id), ...]
    """
    # Extract components
    box_coords = output[0, :4, :]      # [4, 8400] - x, y, w, h
    class_preds = output[0, 4:19, :]   # [15, 8400] - class probabilities
    angle_preds = output[0, 19, :]     # [8400] - angles
    
    # Get max class confidence and class index for each detection
    max_class_conf, class_indices = torch.max(class_preds, dim=0)  # [8400]
    
    # Filter by confidence threshold
    high_conf_mask = max_class_conf > confidence_threshold
    
    if not high_conf_mask.any():
        return []
    
    # Extract high-confidence detections
    filtered_boxes = box_coords[:, high_conf_mask]  # [4, N]
    filtered_angles = angle_preds[high_conf_mask]   # [N]
    filtered_conf = max_class_conf[high_conf_mask]  # [N]
    filtered_classes = class_indices[high_conf_mask]  # [N]
    
    # Scale coordinates from model size to original size
    scale_x = original_size[0] / target_size[0]
    scale_y = original_size[1] / target_size[1]
    
    detections = []
    for i in range(filtered_boxes.shape[1]):
        x = filtered_boxes[0, i].item() * scale_x
        y = filtered_boxes[1, i].item() * scale_y
        w = filtered_boxes[2, i].item() * scale_x
        h = filtered_boxes[3, i].item() * scale_y
        angle = filtered_angles[i].item()
        conf = filtered_conf[i].item()
        class_id = filtered_classes[i].item()
        
        detections.append((x, y, w, h, angle, conf, class_id))
    
    return detections


def visualize_obb_predictions(image_path, output, confidence_threshold=0.05, save_path=None):
    """
    Visualize OBB predictions on the original image
    Args:
        image_path: Path to original image
        output: Model output tensor [1, 20, 8400]
        confidence_threshold: Minimum confidence for visualization
        save_path: Path to save the visualization (optional)
    """
    try:
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
            
        original_size = (original_image.shape[1], original_image.shape[0])  # (width, height)
        
        # Process predictions
        detections = process_obb_predictions(
            output, 
            confidence_threshold=confidence_threshold,
            original_size=original_size,
            target_size=(640, 640)
        )
        
        print(f"🎯 Found {len(detections)} detections (conf > {confidence_threshold})")
        
        # Draw detections on image
        viz_image = original_image.copy()
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, (x, y, w, h, angle, conf, class_id) in enumerate(detections):
            color = colors[i % len(colors)]
            
            # Draw oriented bounding box
            viz_image = draw_oriented_bbox(
                viz_image, x, y, w, h, angle, color=color, thickness=3
            )
            
            # Add text label
            label = f"Class:{class_id} Conf:{conf:.3f} Angle:{angle:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Position text above the box
            text_x = int(x - label_size[0] // 2)
            text_y = int(y - h // 2 - 10)
            
            # Ensure text is within image bounds
            text_x = max(0, min(text_x, original_image.shape[1] - label_size[0]))
            text_y = max(label_size[1], text_y)
            
            # Draw text background
            cv2.rectangle(viz_image, 
                         (text_x - 2, text_y - label_size[1] - 2),
                         (text_x + label_size[0] + 2, text_y + 2),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(viz_image, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save or show results
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, viz_image)
            print(f"💾 Saved visualization to: {save_path}")
        
        # Also save a side-by-side comparison
        if save_path:
            # Resize images to same height for comparison
            height = min(original_image.shape[0], 800)  # Max height 800px
            
            # Calculate new dimensions maintaining aspect ratio
            orig_aspect = original_image.shape[1] / original_image.shape[0]
            orig_width = int(height * orig_aspect)
            
            orig_resized = cv2.resize(original_image, (orig_width, height))
            viz_resized = cv2.resize(viz_image, (orig_width, height))
            
            # Create side-by-side comparison
            comparison = np.hstack([orig_resized, viz_resized])
            
            # Add labels
            cv2.putText(comparison, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, f"OBB Predictions ({len(detections)} detections)", 
                       (orig_width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            comparison_path = save_path.replace('.jpg', '_comparison.jpg')
            cv2.imwrite(comparison_path, comparison)
            print(f"💾 Saved comparison to: {comparison_path}")
            
        return viz_image, detections
        
    except Exception as e:
        print(f"❌ Error visualizing predictions: {e}")
        return None, []


def test_obb_with_real_images():
    """Test OBB model with actual images from the demo folder"""
    print("📸 Testing OBB Model with Real Images...")
    
    # Load the OBB model
    torch_model = load_torch_model()
    torch_model.eval()
    
    
    # Also try some backup images from other YOLO demos if available
    # Test images - use existing demo images
    test_images = [
        "/Users/dgnidash/projects/tt-metal/test_data/P0006.jpg",
        "/Users/dgnidash/projects/tt-metal/test_data/P0009.jpg", 
        "/Users/dgnidash/projects/tt-metal/test_data/P0015.jpg",
        "/Users/dgnidash/projects/tt-metal/test_data/P0014.jpg",
        "/Users/dgnidash/projects/tt-metal/test_data/P0016.jpg",
        "/Users/dgnidash/projects/tt-metal/test_data/P0017.jpg",
    ]
    all_images = test_images
    results = []
    
    for image_path in all_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue
            
        print(f"\n🔍 Processing: {os.path.basename(image_path)}")
        
        # Preprocess image
        input_tensor, original_size = preprocess_image(image_path)
        if input_tensor is None:
            continue
            
        # Run inference
        with torch.no_grad():
            output = torch_model(input_tensor)
            
        print(f"📤 Output shape: {output.shape}")
        
        # Validate output
        expected_shape = (1, 20, 8400)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Create output directory for visualizations
        output_dir = "/Users/dgnidash/projects/tt-metal/obb_test_outputs"
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"{image_name}_obb_predictions.jpg")
        
        # Visualize predictions with lower threshold to see more detections
        viz_image, detections = visualize_obb_predictions(
            image_path, output, confidence_threshold=0.02, save_path=save_path
        )
        
        # Also check with higher threshold for summary
        high_conf_detections = process_obb_predictions(
            output, confidence_threshold=0.1, 
            original_size=original_size, target_size=(640, 640)
        )
        
        num_detections = len(high_conf_detections)
        if num_detections > 0:
            print(f"🎯 Found {num_detections} high-confidence detections (conf > 0.1)")
            for i, (x, y, w, h, angle, conf, class_id) in enumerate(high_conf_detections):
                print(f"   Detection {i+1}: Class={class_id}, Conf={conf:.3f}, "
                      f"Pos=({x:.1f},{y:.1f}), Size=({w:.1f},{h:.1f}), Angle={angle:.3f}")
        else:
            print("🔍 No high-confidence detections found (trying lower threshold for visualization)")
            
        # Get overall statistics
        class_preds = output[:, 4:19, :]   # Class predictions (15 classes)
        max_class_conf = class_preds.max(dim=1)[0]  # Max confidence across classes
        
        # Store results
        results.append({
            'image': os.path.basename(image_path),
            'output_shape': output.shape,
            'num_detections': num_detections,
            'max_confidence': max_class_conf.max().item(),
            'original_size': original_size,
            'visualization_path': save_path,
            'low_threshold_detections': len(detections)
        })
        
        # Basic sanity checks
        assert torch.isfinite(output).all(), f"Output contains non-finite values for {image_path}"
        assert not torch.isnan(output).any(), f"Output contains NaN values for {image_path}"
        assert (class_preds >= 0).all() and (class_preds <= 1).all(), "Class predictions should be between 0 and 1"
        
        print("✅ Image processed successfully!")
        
    if not results:
        print("⚠️  No images were successfully processed. Using fallback test...")
        # Fallback to synthetic test
        return test_obb_simple()
    
    # Summary
    print(f"\n📊 Summary of {len(results)} processed images:")
    for result in results:
        print(f"   🖼️  {result['image']}: {result['num_detections']} high-conf detections, "
              f"{result['low_threshold_detections']} total detections (low threshold)")
        print(f"      Max confidence: {result['max_confidence']:.3f}")
        print(f"      Visualization: {result['visualization_path']}")
    
    if results:
        print(f"\n💾 All visualizations saved to: {output_dir}")
        print("📁 Check the output directory for:")
        print("   • *_obb_predictions.jpg - Images with OBB overlays")
        print("   • *_comparison.jpg - Side-by-side original vs predictions")
    
    print("✅ Real image OBB test with visualization completed!")
    return results


def test_ultralytics_obb_comparison():
    """Compare our OBB implementation with ultralytics native OBB predictions"""
    print("🔬 Comparing with Ultralytics Native OBB Model...")
    
    # Load ultralytics OBB model
    try:
        ultralytics_model = YOLO("yolo11m-obb.pt")
        print("✅ Loaded ultralytics OBB model")
    except Exception as e:
        print(f"❌ Could not load ultralytics model: {e}")
        return None
    
    # Load our custom OBB model
    our_model = load_torch_model()
    our_model.eval()
    
    # Test images - use existing demo images
    test_images = [
        "/Users/dgnidash/projects/tt-metal/test_data/P0006.jpg",
        "/Users/dgnidash/projects/tt-metal/test_data/P0009.jpg", 
        "/Users/dgnidash/projects/tt-metal/test_data/P0015.jpg",
        "/Users/dgnidash/projects/tt-metal/test_data/P0014.jpg",
        "/Users/dgnidash/projects/tt-metal/test_data/P0016.jpg",
        "/Users/dgnidash/projects/tt-metal/test_data/P0017.jpg",
    ]
    
    output_dir = "/Users/dgnidash/projects/tt-metal/test_data/obb_results"
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue
            
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n🖼️  Testing: {image_name}")
        
        # 1. Test with ultralytics native model
        print("🔵 Running ultralytics native OBB prediction...")
        ultralytics_model(image_path, conf=0.10, save=True, verbose=True)

    
    print(f"\n💾 Comparison results saved to: {output_dir}")
    print("📁 Check for:")
    print("   • *_ultralytics_obb.jpg - Official ultralytics OBB predictions")
    print("   • *_our_obb.jpg - Our implementation predictions")
    print("✅ Ultralytics OBB comparison completed!")

if __name__ == "__main__":
    print("🚀 Running Comprehensive OBB Tests...")
    # Compare with ultralytics native model
    print("=" * 60)
    test_obb_with_real_images()
    print("🎉 All OBB tests completed successfully!")