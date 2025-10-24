# DETR3D: 3D Object Detection with Transformers

A PyTorch implementation of DETR3D (Detection Transformer for 3D Object Detection) with TTNN (Tensor Train Neural Network) acceleration support.

## Overview

DETR3D is a transformer-based approach for 3D object detection that extends the DETR (Detection Transformer) framework to 3D point clouds. This implementation includes both PyTorch reference models and TTNN-accelerated versions for efficient inference.

## Features

- **3D Object Detection**: Support for SUNRGBD dataset
- **Transformer Architecture**: End-to-end trainable transformer-based detection
- **TTNN Acceleration**: Optimized inference using Tensor Train Neural Networks
- **Dataset Support**: SUNRGBD support with configurable preprocessing
- **Data Augmentation**: Random cuboid cropping, rotation, scaling, and color augmentation
- **Evaluation Metrics**: Average Precision (AP) calculation with configurable IoU thresholds

## Project Structure

```
models/experimental/detr3d/
├── demo/                          # Demo and inference scripts
│   └── detr3d_demo.py            # Main demo script with AP calculation
├── reference/                     # PyTorch reference implementation
│   ├── model_3detr.py            # Main DETR3D model implementation
│   ├── model_config.py            # Model configuration classes
│   ├── model_utils.py             # Utility functions
│   └── utils/                     # Reference utilities
│       ├── dataset/              # Dataset implementations
│       │   ├── __init__.py
│       │   └── sunrgbd.py         # SUNRGBD dataset (minimal, self-contained)
│       ├── ap_calculator.py      # Average Precision calculation
│       ├── box_util.py           # 3D bounding box utilities
│       ├── eval_det.py           # Detection evaluation
│       ├── misc.py                # Miscellaneous utilities
│       └── nms.py                 # Non-Maximum Suppression
├── source/                        # Original source implementation
│   └── detr3d/                    # Source DETR3D implementation
├── ttnn/                          # TTNN-accelerated implementation
│   └── model_3detr.py            # TTNN DETR3D model
└── tests/                         # Test files
    └── pcc/                       # Performance comparison tests
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy
- TTNN (for accelerated inference)

### Dependencies

```bash
pip install torch torchvision
pip install numpy scipy
```

## Usage

### Basic Inference

```python
from models.experimental.detr3d.reference.model_3detr import build_3detr
from models.experimental.detr3d.reference.model_config import Detr3dArgs
from models.experimental.detr3d.reference.utils.dataset import build_dataset

# Setup configuration
args = Detr3dArgs()
args.dataset_name = "sunrgbd"
args.dataset_root_dir = "/path/to/dataset"
args.use_color = False

# Build dataset and model
datasets, dataset_config = build_dataset(args)
model, _ = build_3detr(args, dataset_config)

# Run inference
test_dataset = datasets["test"]
# ... inference code
```

### Demo Script

Run the demo script for inference with AP calculation:

```bash
python models/experimental/detr3d/demo/detr3d_demo.py \
    --dataset-root-dir /path/to/sunrgbd/dataset \
    --test-ckpt /path/to/checkpoint.pth \
    --seed 0
```

**Example with SUNRGBD dataset:**

```bash
python models/experimental/detr3d/demo/detr3d_demo.py \
    --dataset-root-dir sunrgbd \
    --test-ckpt sunrgbd_masked_ep720.pth
```

### Command Line Arguments

- `--dataset-root-dir`: Path to dataset directory (required)
- `--test-ckpt`: Path to model checkpoint (.pth file, optional)
- `--seed`: Random seed for reproducibility (default: 0)

## Dataset Support

### SUNRGBD Dataset

- **Classes**: 10 semantic classes (bed, table, sofa, chair, toilet, desk, dresser, night_stand, bookshelf, bathtub)
- **Format**: Point clouds in `.npz` format, bounding boxes in `.npy` format
- **Coordinate System**: Upright depth coordinate (X right, Y forward, Z upward)
- **Data Augmentation**: Random cuboid cropping, rotation, scaling, color augmentation

#### Input Structure

The SUNRGBD dataset should be organized as follows:

```
sunrgbd/
├── sunrgbd_val/                    # Validation split
│   ├── 000001_bbox.npy            # Bounding box annotations (K×8)
│   ├── 000001_pc.npz              # Point cloud data (N×6: x,y,z,r,g,b)
│   ├── 000001_votes.npz           # Vote supervision data (optional)
│   ├── 000002_bbox.npy
│   ├── 000002_pc.npz
│   ├── 000002_votes.npz
│   └── ...
└── sunrgbd_train/                  # Training split (same structure)
    ├── 000001_bbox.npy
    ├── 000001_pc.npz
    ├── 000001_votes.npz
    └── ...
```

**File Formats:**
- **Point Clouds** (`.npz`): Contains `pc` key with shape (N×6) - [x, y, z, r, g, b]
- **Bounding Boxes** (`.npy`): Shape (K×8) - [cx, cy, cz, l, w, h, angle, class_id]
- **Votes** (`.npz`): Optional vote supervision data for training

## Model Architecture

### DETR3D Components

1. **Point Cloud Encoder**: Processes input point clouds
2. **Transformer Decoder**: Multi-head attention for object detection
3. **3D Bounding Box Head**: Predicts 3D bounding boxes and class labels
4. **Loss Functions**: Classification, regression, and IoU losses

### Key Features

- **End-to-End Training**: No post-processing required
- **Set Prediction**: Handles variable number of objects
- **3D Bounding Boxes**: Oriented bounding boxes with heading angles
- **Multi-Scale Features**: Hierarchical feature extraction

## Configuration

### Model Configuration (`Detr3dArgs`)

```python
class Detr3dArgs:
    def __init__(self):
        self.dataset_name = "sunrgbd"
        self.dataset_root_dir = None
        self.use_color = False
        self.batchsize_per_gpu = 1
        self.dataset_num_workers = 0
        # ... other parameters
```


### Dataset Configuration

- **SUNRGBD**: 10 classes, 12 angle bins, max 64 objects

## Evaluation

### Metrics

- **Average Precision (AP)**: Primary evaluation metric
- **IoU Thresholds**: Configurable (default: 0.25, 0.5)
- **Per-Class AP**: Individual class performance
- **Overall mAP**: Mean Average Precision

### Evaluation Script

```python
from models.experimental.detr3d.reference.utils.ap_calculator import APCalculator

ap_calculator = APCalculator(
    dataset_config=dataset_config,
    ap_iou_thresh=[0.25, 0.5],
    class2type_map=dataset_config.class2type,
    exact_eval=True,
)
```

## TTNN Acceleration

The project includes TTNN-accelerated models for efficient inference:

- **Tensor Train Decomposition**: Reduces model parameters
- **Hardware Acceleration**: Optimized for specific hardware
- **Performance Comparison**: PCC (Pearson Correlation Coefficient) validation

## Data Preprocessing

### Point Cloud Processing

1. **Sampling**: Random sampling to fixed number of points
2. **Normalization**: Coordinate normalization
3. **Augmentation**: Random transformations and cropping
4. **Color Processing**: RGB normalization (if enabled)

### Bounding Box Processing

1. **Angle Discretization**: Continuous angles to discrete classes
2. **Size Normalization**: Object size normalization
3. **Center Normalization**: Bounding box center normalization

## File Formats

### Input Data

- **Point Clouds**: `.npz` files with `pc` key (N×6 array: x,y,z,r,g,b)
- **Bounding Boxes**: `.npy` files (K×8 array: cx,cy,cz,l,w,h,angle,class)

### Output Data

- **Detections**: Dictionary with bounding boxes, classes, and scores
- **Metrics**: AP scores and per-class performance

## Troubleshooting

### Common Issues

1. **Dataset Path**: Ensure correct dataset directory structure
2. **Memory Issues**: Reduce batch size or number of points
3. **CUDA Issues**: Check GPU memory and CUDA compatibility
4. **Dependency Issues**: Install all required packages

### Performance Tips

- Use appropriate batch size for your GPU memory
- Enable data loading workers for faster training
- Use mixed precision training for memory efficiency
- Consider TTNN acceleration for inference

## Citation

If you use this code, please cite the original DETR3D paper:

```bibtex
@article{wang2022detr3d,
  title={DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries},
  author={Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and Solomon, Justin},
  journal={arXiv preprint arXiv:2110.06922},
  year={2022}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original DETR3D implementation from Facebook Research
- TTNN acceleration framework
- SUNRGBD and ScanNet dataset providers
