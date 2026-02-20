# EfficientDet D0 Model

EfficientDet D0 is an object detection model optimized for Tenstorrent hardware using TTNN (Tenstorrent Neural Network). This implementation provides both PyTorch reference and TTNN-accelerated versions of the EfficientDet D0 architecture.

## Overview

EfficientDet is a family of efficient object detection models that achieve state-of-the-art accuracy with significantly fewer parameters and FLOPs compared to previous detection models. The D0 variant is the smallest and fastest model in the EfficientDet family, making it ideal for edge deployment and real-time applications.

### Key Features

- **Efficient Architecture**: Combines EfficientNet-B0 backbone with BiFPN (Bidirectional Feature Pyramid Network)
- **Multi-scale Detection**: Detects objects at 5 different scales (P3-P7)
- **COCO Dataset**: Pre-trained on COCO dataset with 90 object classes
- **TTNN Acceleration**: Optimized for Tenstorrent hardware with bfloat16 precision
- **High Accuracy**: Achieves competitive mAP on COCO validation set

## Model Architecture

The EfficientDet D0 model consists of three main components:

1. **Backbone Network (EfficientNet-B0)**: Extracts multi-scale feature maps from input images
   - Input: RGB images (3 channels)
   - Output: Feature maps at scales P3, P4, P5

2. **BiFPN (Bidirectional Feature Pyramid Network)**: Fuses multi-scale features
   - Processes features at 5 scales (P3, P4, P5, P6, P7)
   - Uses weighted feature fusion for better information flow
   - Repeats 3 times for D0 variant

3. **Detection Head**: Consists of two branches
   - **Regressor**: Predicts bounding box coordinates (4 values per anchor)
   - **Classifier**: Predicts object class probabilities (90 classes per anchor)

### Model Specifications

- **Input Size**: 512×512 pixels
- **Backbone**: EfficientNet-B0
- **BiFPN Channels**: 64
- **BiFPN Layers**: 3
- **Number of Classes**: 90 (COCO dataset)
- **Anchor Scales**: 3 scales per pyramid level
- **Aspect Ratios**: 3 ratios (1.0, 1.4, 0.7)
- **Pyramid Levels**: 5 (P3, P4, P5, P6, P7)

## References

- **Original Paper**: [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
- **Reference Implementation**: [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-Efficient-Pytorch)
- **EfficientNet Paper**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

## Performance(on N150)

### Inference Performance

##### FPS (Frames Per Second)

- **Batch Size 1, 512×512**: 36

##### Device Time

- **Total Inference Time**: 27160 ms

**Measurement Configuration:**
- Input size: 512×512
- Batch size: 1
- Device: Tenstorrent (N150)
- Precision: bfloat16


## Directory Structure

```
models/experimental/efficientdetd0/
├── README.md                          # This file
├── common.py                          # Common utilities (weight loading, key mapping)
│
├── demo/                              # Demo application
│   ├── demo.py                       # Main demo script
│   └── output/                        # Output directory for visualization images
│
├── reference/                         # PyTorch reference implementation
│   ├── efficientdet.py               # EfficientDet backbone reference
│   ├── efficientnetb0.py             # EfficientNet-B0 reference
│   ├── modules.py                    # BiFPN, Regressor, Classifier modules
│   └── utils.py                       # Utility functions
│
├── resources/                         # Resources and weights
│   ├── efficientdetd0_weights_download.sh  # Script to download weights
|   ├── efficientdet-d0.pth               # Pre-trained model weights
│   └── *.jpg                         # Sample test images
│
├── tests/                             # Test suite
│   ├── test_efficient_det.py         # Main integration test
│   ├── test_bifpn.py                 # BiFPN component test
│   ├── test_classifier.py            # Classifier component test
│   ├── test_regressor.py             # Regressor component test
│   └── evaluate_coco.py              # COCO evaluation script
│
└── tt/                                # TTNN implementation
    ├── efficient_det.py              # Main TTNN EfficientDet model
    ├── efficient_netb0.py            # TTNN EfficientNet-B0
    ├── bifpn.py                      # TTNN BiFPN implementation
    ├── classifier.py                 # TTNN Classifier implementation
    ├── regressor.py                  # TTNN Regressor implementation
    ├── custom_preprocessor.py        # Custom weight preprocessing
    └── utils.py                      # TTNN utility functions
```

## Setup

### Download Model Weights

The model weights are required for inference. You can download them automatically or manually:

**Automatic Download:**
```bash
cd models/experimental/efficientdetd0
bash resources/efficientdetd0_weights_download.sh
```

**Manual Download:**
The weights will be downloaded from:
```
https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth
```

The weights will be saved to either:
- `models/experimental/efficientdetd0/resources/efficientdet-d0.pth`

## Usage

### Running the Demo

The demo script demonstrates object detection on images:

```bash
# Basic usage with default settings (512x512 input)
python models/experimental/efficientdetd0/demo/demo.py

# Custom input size
python models/experimental/efficientdetd0/demo/demo.py --height 512 --width 512

# Specify input image
python models/experimental/efficientdetd0/demo/demo.py --image-path path/to/image.jpg

# Custom detection thresholds
python models/experimental/efficientdetd0/demo/demo.py --threshold 0.5 --iou-threshold 0.5

# Specify device and weights path
python models/experimental/efficientdetd0/demo/demo.py \
    --device-id 0 \
    --weights-path path/to/efficientdet-d0.pth \
    --output-dir path/to/output\

# Example
python models/experimental/efficientdetd0/demo/demo.py \
    --image-path models/experimental/efficientdetd0/resources/vid_4_10500.jpg \
    --weights-path models/experimental/efficientdetd0/resources/efficientdet-d0.pth \
    --output-dir models/experimental/efficientdetd0/demo/output \
    --threshold 0.2 \
    --iou-threshold 0.2
```

**Demo Arguments:**
- `--batch-size`: Batch size for inference (default: 1)
- `--height`: Input image height (default: 512)
- `--width`: Input image width (default: 512)
- `--num-classes`: Number of object classes (default: 90)
- `--device-id`: Device ID to use (default: 0)
- `--l1-small-size`: L1 small size for device (default: 24576)
- `--image-path`: Path to input image
- `--threshold`: Score threshold for detections (default: 0.5)
- `--iou-threshold`: IoU threshold for NMS (default: 0.5)
- `--weights-path`: Path to model weights file
- `--output-dir`: Directory to save output images with bounding boxes

The demo will:
1. Load and preprocess the input image
2. Run inference on both PyTorch reference and TTNN models
3. Post-process outputs to get bounding boxes
4. Display PCC (Pearson Correlation Coefficient) comparisons (threshold: 0.92)
5. Show detection summary (count of detections for PyTorch and TTNN)
6. Visualize detections and save output images (if input image is provided)

### Running Tests

Run the test suite to verify model correctness:

```bash
# Run all tests
pytest models/experimental/efficientdetd0/tests/ -v

# Run specific test
pytest models/experimental/efficientdetd0/tests/test_efficient_det.py -v

# Run with specific device parameters
pytest models/experimental/efficientdetd0/tests/test_efficient_det.py \
    --device-params '{"l1_small_size": 16384}' -v
```

**Test Coverage:**
- `test_efficient_det.py`: Full model integration test
- `test_bifpn.py`: BiFPN component test
- `test_classifier.py`: Classifier component test
- `test_regressor.py`: Regressor component test

### COCO Evaluation

To evaluate the model on COCO validation set and get standard COCO metrics (mAP, AP@0.5, AP@0.75, AR, etc.):

**Prerequisites:**
```bash
pip install pycocotools
```

**Download COCO Dataset:**
1. Download COCO 2017 validation images and annotations from [COCO website](https://cocodataset.org/#download)
2. Extract the dataset to a directory (e.g., `/path/to/coco/`)

**Run Evaluation:**
```bash
# Evaluate TTNN implementation
python models/experimental/efficientdetd0/tests/evaluate_coco.py \
    --coco-annotations /path/to/coco/annotations/instances_val2017.json \
    --coco-images /path/to/coco/val2017 \
    --weights-path models/experimental/efficientdetd0/resources/efficientdet-d0.pth \
    --device-id 0 \
    --num-samples 5000

# Evaluate PyTorch reference (for comparison)
python models/experimental/efficientdetd0/tests/evaluate_coco.py \
    --coco-annotations /path/to/coco/annotations/instances_val2017.json \
    --coco-images /path/to/coco/val2017 \
    --weights-path models/experimental/efficientdetd0/resources/efficientdet-d0.pth \
    --pytorch-only \
    --num-samples 5000
```

**Evaluation Arguments:**
- `--coco-annotations`: Path to COCO annotations JSON file (required)
- `--coco-images`: Path to COCO validation images directory (required)
- `--weights-path`: Path to model weights file
- `--device-id`: Device ID to use (default: 0)
- `--num-samples`: Number of samples to evaluate (default: all)
- `--batch-size`: Batch size (default: 1)
- `--height`: Input image height (default: 512)
- `--width`: Input image width (default: 512)
- `--threshold`: Score threshold for detections (default: 0.05)
- `--iou-threshold`: IoU threshold for NMS (default: 0.5)
- `--pytorch-only`: Use PyTorch reference instead of TTNN

**Expected Output Format:**
The evaluation script outputs metrics in the same format as the [reference benchmark](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/benchmark/coco_eval_result)

**Reference Benchmark Results:**
According to the [official benchmark](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/benchmark/coco_eval_result)

## Model Components

### EfficientNet-B0 Backbone

The backbone network extracts multi-scale features from input images:
- **Input**: `(batch, 3, height, width)` RGB images
- **Output**: Feature maps at three scales:
  - P3: `(batch, 40, H/4, W/4)` - 1/4 scale
  - P4: `(batch, 112, H/8, W/8)` - 1/8 scale
  - P5: `(batch, 320, H/16, W/16)` - 1/16 scale

### BiFPN (Bidirectional Feature Pyramid Network)

The BiFPN fuses multi-scale features:
- **Input**: P3, P4, P5 from backbone
- **Output**: Enhanced features at P3, P4, P5, P6, P7
- **Features**:
  - Bidirectional information flow
  - Weighted feature fusion
  - 3 repeated layers for D0 variant

### Regressor

Predicts bounding box coordinates:
- **Input**: BiFPN features at all pyramid levels
- **Output**: `(batch, num_anchors, 4)` - bounding box deltas (dy, dx, dh, dw)
- **Architecture**: 3-layer convolutional network

### Classifier

Predicts object class probabilities:
- **Input**: BiFPN features at all pyramid levels
- **Output**: `(batch, num_anchors, num_classes)` - class logits
- **Architecture**: 3-layer convolutional network

## Input/Output Format

### Input

- **Format**: RGB images
- **Preprocessing**:
  - Resize to 512×512 (maintaining aspect ratio with padding)
  - Normalize with ImageNet mean/std: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`
  - Convert to tensor format: `(batch, 3, 512, 512)`

### Output

The model returns three outputs:

1. **Feature Maps** (tuple of 5 tensors):
   - P3: `(batch, 64, H/4, W/4)`
   - P4: `(batch, 64, H/8, W/8)`
   - P5: `(batch, 64, H/16, W/16)`
   - P6: `(batch, 64, H/32, W/32)`
   - P7: `(batch, 64, H/64, W/64)`

2. **Regression Output**: `(batch, num_anchors, 4)`
   - Bounding box deltas: `[dy, dx, dh, dw]`
   - Used to transform anchor boxes to predicted boxes

3. **Classification Output**: `(batch, num_anchors, num_classes)`
   - Class logits for 90 COCO classes
   - Apply softmax to get probabilities

### Post-processing

To get final detections:
1. Generate anchor boxes for all pyramid levels
2. Transform anchors using regression outputs
3. Apply softmax to classification outputs
4. Filter detections by score threshold
5. Apply Non-Maximum Suppression (NMS)
6. Transform boxes back to original image coordinates



### Model Statistics

- **Total Parameters**: ~3.9M (EfficientDet D0)
- **FLOPs**: ~2.5B (for 512×512 input)
- **Inference Speed**: Optimized for Tenstorrent hardware acceleration


## Troubleshooting

### Common Issues

1. **Weights not found**
   - Run the download script: `bash resources/efficientdetd0_weights_download.sh`
   - Or manually download and place in the correct directory

2. **Device initialization errors**
   - Check device availability: `ttnn.list_devices()`
   - Verify device parameters (l1_small_size, etc.)

3. **Memory errors**
   - Reduce batch size
   - Adjust L1 small size parameter
   - Check available device memory

4. **PCC below threshold (0.92)**
   - Verify model weights are loaded correctly
   - Check input preprocessing matches reference
   - Ensure device is properly configured
   - Note: PCC threshold is set to 0.92 for the demo



## License

This implementation is licensed under Apache-2.0. See the SPDX license headers in source files for details.
