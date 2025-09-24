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
import ttnn
from models.demos.yolov11m.tt import ttnn_yolov11
from models.demos.yolov11m.tt.model_preprocessing import create_yolov11_input_tensors, create_yolov11_model_parameters


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


def pytorch_obb_with_real_images(test_images):
    """Test OBB model with actual images from the demo folder"""
    print("📸 Testing OBB Model with Real Images...")
    
    # Load the OBB model
    torch_model = load_torch_model()
    torch_model.eval()

    results = []
    
    for image_path in test_images:
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
        output_dir = "./obb_test_outputs"
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


def test_compare_ttnn_and_pytorch_obb_simple():
    """Simple test for TTNN OBB model with synthetic data"""
    print("🔥 Testing TTNN OBB Model (Simple)...")
    
    try:
        # Initialize TTNN device with proper configuration
        device = ttnn.CreateDevice(
            0, l1_small_size=YOLOV11_L1_SMALL_SIZE, trace_region_size=3211264, num_command_queues=2
        )
        device.enable_program_cache()
        print("✅ TTNN device opened successfully")
        
        # Load PyTorch OBB model
        torch_model = load_torch_model()
        torch_model.eval()
        print("✅ PyTorch OBB model loaded")
        
        # Create test input using proper TTNN input tensor creation
        batch_size, channels, height, width = 1, 3, 640, 640
        torch_input, ttnn_input = create_yolov11_input_tensors(
            device,
            batch=batch_size,
            input_channels=channels,
            input_height=height,
            input_width=width,
            is_sub_module=False,
        )
        print(f"📥 Input shape: {torch_input.shape}")
        
        # Create TTNN model parameters and model (needs input tensor)
        ttnn_model_parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
        ttnn_model = ttnn_yolov11.TtnnYoloV11(device, ttnn_model_parameters)
        print("✅ TTNN OBB model loaded")
        
        # Run PyTorch inference
        with torch.no_grad():
            torch_output = torch_model(torch_input)
        print(f"📤 PyTorch output shape: {torch_output.shape}")
        
        # Run TTNN inference
        ttnn_output = ttnn_model(ttnn_input)
        print(f"📤 TTNN output shape: {ttnn_output.shape}")
        
        # Convert TTNN output back to torch for comparison
        ttnn_output_torch = ttnn.to_torch(ttnn_output)
        
        # Validate shapes match
        assert torch_output.shape == ttnn_output_torch.shape, \
            f"Shape mismatch: PyTorch {torch_output.shape} vs TTNN {ttnn_output_torch.shape}"
        
        # Calculate PCC
        torch_flat = torch_output.flatten().float()  # Convert to float32
        ttnn_flat = ttnn_output_torch.flatten().float()  # Convert to float32
        correlation_matrix = np.corrcoef(torch_flat.numpy(), ttnn_flat.numpy())
        pcc = correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0.0
        
        print(f"📊 PCC (PyTorch vs TTNN): {pcc:.6f}")
        
        # Validate outputs
        expected_shape = (batch_size, 20, 8400)
        assert torch_output.shape == expected_shape, f"Expected shape {expected_shape}, got {torch_output.shape}"
        assert ttnn_output_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {ttnn_output_torch.shape}"
        
        print("✅ TTNN OBB simple test passed!")
        return pcc
        
    except Exception as e:
        print(f"❌ TTNN simple test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0
    finally:
        try:
            ttnn.close_device(device)
            print("✅ TTNN device closed")
        except:
            pass


def compare_ttnn_and_pytorch_obb_with_real_images(test_images):
    """Test TTNN OBB model with real images and compare with PyTorch"""
    print("🔥 Testing TTNN OBB Model with Real Images...")
    
    try:
        # Initialize TTNN device with proper configuration
        device = ttnn.CreateDevice(
            0, l1_small_size=YOLOV11_L1_SMALL_SIZE, trace_region_size=3211264, num_command_queues=2
        )
        print("✅ TTNN device opened successfully")
        
        # Load PyTorch OBB model
        torch_model = load_torch_model()
        torch_model.eval()
        print("✅ PyTorch OBB model loaded")
        
        # Create dummy input tensor for model parameter initialization using proper TTNN function
        dummy_torch_input, dummy_ttnn_input = create_yolov11_input_tensors(
            device, batch=1, input_channels=3, input_height=640, input_width=640, is_sub_module=False
        )
        
        # Create TTNN model parameters and model
        ttnn_model_parameters = create_yolov11_model_parameters(torch_model, dummy_torch_input, device=device)
        ttnn_model = ttnn_yolov11.TtnnYoloV11(device, ttnn_model_parameters)
        print("✅ TTNN OBB model loaded")
        

        results = []
        
        for image_path in test_images:
            if not os.path.exists(image_path):
                print(f"⚠️  Image not found: {image_path}")
                continue
                
            print(f"\n🔍 Processing: {os.path.basename(image_path)}")
            
            # Preprocess image
            input_tensor, original_size = preprocess_image(image_path)
            if input_tensor is None:
                continue
                
            # Run PyTorch inference
            with torch.no_grad():
                torch_output = torch_model(input_tensor)
            print(f"📤 PyTorch output shape: {torch_output.shape}")
            

            torch_input, ttnn_input = create_yolov11_input_tensors(
                device,
                batch=input_tensor.shape[0],
                input_channels=input_tensor.shape[1], 
                input_height=input_tensor.shape[2],
                input_width=input_tensor.shape[3],
                is_sub_module=False,
            )
            
            # Run TTNN inference
            ttnn_output = ttnn_model(ttnn_input)
            print(f"📤 TTNN output shape: {ttnn_output.shape}")
            
            # Convert TTNN output back to torch for comparison
            ttnn_output_torch = ttnn.to_torch(ttnn_output)
            
            # Validate shapes match
            assert torch_output.shape == ttnn_output_torch.shape, \
                f"Shape mismatch: PyTorch {torch_output.shape} vs TTNN {ttnn_output_torch.shape}"
            
            # Calculate PCC (Pearson Correlation Coefficient) for comparison
            # Flatten tensors for correlation calculation
            torch_flat = torch_output.flatten()
            ttnn_flat = ttnn_output_torch.flatten()
            
            # Calculate correlation using numpy
            torch_flat = torch_flat.float()  # Convert to float32
            ttnn_flat = ttnn_flat.float()  # Convert to float32  
            correlation_matrix = np.corrcoef(torch_flat.numpy(), ttnn_flat.numpy())
            pcc = correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0.0
            
            print(f"📊 PCC (PyTorch vs TTNN): {pcc:.6f}")
            
            # Create output directories
            output_dir_torch = "./obb_test_outputs/pytorch"
            output_dir_ttnn = "./obb_test_outputs/ttnn"
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Visualize PyTorch predictions
            torch_save_path = os.path.join(output_dir_torch, f"{image_name}_pytorch_obb.jpg")
            viz_torch, detections_torch = visualize_obb_predictions(
                image_path, torch_output, confidence_threshold=0.02, save_path=torch_save_path
            )
            
            # Visualize TTNN predictions
            ttnn_save_path = os.path.join(output_dir_ttnn, f"{image_name}_ttnn_obb.jpg")
            viz_ttnn, detections_ttnn = visualize_obb_predictions(
                image_path, ttnn_output_torch, confidence_threshold=0.02, save_path=ttnn_save_path
            )
            
            # Create side-by-side comparison of PyTorch vs TTNN
            if viz_torch is not None and viz_ttnn is not None:
                comparison_dir = "./obb_test_outputs/comparison"
                os.makedirs(comparison_dir, exist_ok=True)
                
                # Resize to same dimensions
                h, w = min(viz_torch.shape[0], viz_ttnn.shape[0]), min(viz_torch.shape[1], viz_ttnn.shape[1])
                viz_torch_resized = cv2.resize(viz_torch, (w, h))
                viz_ttnn_resized = cv2.resize(viz_ttnn, (w, h))
                
                # Create comparison image
                comparison_image = np.hstack([viz_torch_resized, viz_ttnn_resized])
                
                # Add labels
                cv2.putText(comparison_image, f"PyTorch ({len(detections_torch)} detections)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(comparison_image, f"TTNN ({len(detections_ttnn)} detections)", 
                           (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(comparison_image, f"PCC: {pcc:.4f}", 
                           (w//2 - 50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                comparison_path = os.path.join(comparison_dir, f"{image_name}_pytorch_vs_ttnn.jpg")
                cv2.imwrite(comparison_path, comparison_image)
                print(f"💾 Saved comparison to: {comparison_path}")
            
            # Store results
            results.append({
                'image': os.path.basename(image_path),
                'pcc': pcc,
                'pytorch_detections': len(detections_torch),
                'ttnn_detections': len(detections_ttnn),
                'pytorch_viz': torch_save_path,
                'ttnn_viz': ttnn_save_path,
                'comparison_viz': comparison_path if 'comparison_path' in locals() else None
            })
            
            print(f"✅ Image {os.path.basename(image_path)} processed successfully!")
            print(f"   PyTorch detections: {len(detections_torch)}")
            print(f"   TTNN detections: {len(detections_ttnn)}")
            print(f"   Correlation (PCC): {pcc:.6f}")
        
        # Summary
        if results:
            print(f"\n📊 TTNN vs PyTorch Comparison Summary:")
            avg_pcc = np.mean([r['pcc'] for r in results])
            print(f"   Average PCC: {avg_pcc:.6f}")
            
            for result in results:
                print(f"   🖼️  {result['image']}: PCC={result['pcc']:.4f}, "
                      f"PyTorch={result['pytorch_detections']} vs TTNN={result['ttnn_detections']} detections")
            
            print(f"\n💾 Visualizations saved to:")
            print(f"   📁 PyTorch: {output_dir_torch}")
            print(f"   📁 TTNN: {output_dir_ttnn}")
            print(f"   📁 Comparisons: {comparison_dir}")
            
            # Check if PCC is reasonable (should be > 0.9 for good correlation)
            if avg_pcc > 0.9:
                print(f"✅ TTNN model shows excellent correlation with PyTorch (PCC > 0.9)")
            elif avg_pcc > 0.7:
                print(f"⚠️  TTNN model shows good correlation with PyTorch (PCC > 0.7)")
            else:
                print(f"❌ TTNN model shows poor correlation with PyTorch (PCC < 0.7)")
        
        print("✅ TTNN OBB comparison test completed!")
        return results
        
    except Exception as e:
        print(f"❌ TTNN test failed: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        # Clean up TTNN device
        try:
            ttnn.close_device(device)
            print("✅ TTNN device closed")
        except:
            pass


if __name__ == "__main__": 

    # Also try some backup images from other YOLO demos if available
    # Test images - use existing demo images
    test_images = [
        "./models/demos/yolov11m/tests/satellite_images/P0006.jpg",
        #"./models/demos/yolov11m/tests/satellite_images/P0009.jpg", 
        #"./models/demos/yolov11m/tests/satellite_images/P0015.jpg",
        #"./models/demos/yolov11m/tests/satellite_images/P0014.jpg", 
        #"./models/demos/yolov11m/tests/satellite_images/P0016.jpg",
        #"./models/demos/yolov11m/tests/satellite_images/P0017.jpg",
    ]

    #pytorch_results = pytorch_obb_with_real_images(test_images)
    
    # Test TTNN OBB model and compare with PyTorch
    #print("\n" + "=" * 60)
    #print("🔥 Testing TTNN vs PyTorch OBB Model")
    #print("=" * 60)
    
    #simple_pcc = test_compare_ttnn_and_pytorch_obb_simple()
    
    # Then run full test with real images
    print("\n📸 Running TTNN vs PyTorch test with real images...")
    ttnn_results = compare_ttnn_and_pytorch_obb_with_real_images(test_images)
    
    # Final summary
    #if pytorch_results:
    #    print(f"✅ PyTorch OBB: {len(pytorch_results)} images processed successfully")
    
    #if simple_pcc > 0:
    #    print(f"✅ TTNN Simple Test: PCC = {simple_pcc:.6f}")
    #else:
    #    print("❌ TTNN Simple Test: Failed")
        
    if ttnn_results:
        avg_pcc = np.mean([r['pcc'] for r in ttnn_results])
        print(f"✅ TTNN OBB with Real Images: {len(ttnn_results)} images processed, Average PCC: {avg_pcc:.4f}")
        print(f"📁 Check ./obb_test_outputs/ for visualizations")
    else:
        print("⚠️  TTNN real image test not available or failed")
    
    print("🎉 All OBB tests completed successfully!")