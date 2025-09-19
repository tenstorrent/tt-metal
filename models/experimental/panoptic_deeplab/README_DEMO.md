# Panoptic DeepLab Demo

This demo showcases the TTNN implementation of Panoptic DeepLab for panoptic segmentation on street scene images.

## Overview

Panoptic DeepLab is a state-of-the-art model that performs panoptic segmentation - combining both semantic segmentation (pixel-wise classification) and instance segmentation (object detection and segmentation) in a unified framework.

The model outputs:
- **Semantic segmentation**: Per-pixel class predictions for all 19 Cityscapes classes
- **Instance segmentation**: Individual object instances with center points and offsets
- **Panoptic segmentation**: Combined semantic and instance predictions

## Prerequisites

1. **Model Weights**: Download the Panoptic DeepLab weights trained on Cityscapes dataset:
   ```bash
   # Create weights directory
   mkdir -p models/experimental/panoptic_deeplab/weights

   # Download weights (example - replace with actual download link)
   wget -O models/experimental/panoptic_deeplab/weights/model_final_bd324a.pkl \
        "https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/model_final_bd324a.pkl"
   ```

2. **Sample Images**: Place test images in the resources directory:
   ```bash
   # Example street scene images work best (similar to Cityscapes dataset)
   cp your_street_scene_image.jpg models/experimental/panoptic_deeplab/resources/
   ```

## Usage

### Single Image Processing

```bash
# Run demo on a single image
python models/experimental/panoptic_deeplab/tt/demo.py \
    models/experimental/panoptic_deeplab/resources/sample_image.jpg \
    models/experimental/panoptic_deeplab/weights/model_final_bd324a.pkl \
    output_dir
```

### Batch Processing

```bash
# Process all images in a directory
python models/experimental/panoptic_deeplab/tt/demo.py \
    models/experimental/panoptic_deeplab/resources \
    models/experimental/panoptic_deeplab/weights/model_final_bd324a.pkl \
    output_dir \
    --batch
```

### Using pytest

```bash
# Run single image test
pytest models/experimental/panoptic_deeplab/tt/demo.py::test_panoptic_deeplab_demo -v

# Run batch processing test
pytest models/experimental/panoptic_deeplab/tt/demo.py::test_panoptic_deeplab_batch_demo -v
```

## Output Files

The demo generates several output files for each processed image:

1. **`{image_name}_original.jpg`**: Original input image
2. **`{image_name}_semantic.jpg`**: Semantic segmentation visualization
3. **`{image_name}_panoptic.jpg`**: Panoptic segmentation visualization (blended with original)
4. **`{image_name}_info.json`**: Detailed segmentation information including:
   - Number of detected segments
   - Per-segment information (category, area, etc.)

## Cityscapes Classes

The model predicts 19 semantic classes from the Cityscapes dataset:

**Stuff classes (background):**
- road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky

**Thing classes (objects):**
- person, rider, car, truck, bus, train, motorcycle, bicycle

## Model Configuration

- **Input size**: 512×1024 (height×width) - optimized for Cityscapes format
- **Architecture**: ResNet-50 backbone with DeepLab heads
- **Precision**: bfloat16 for efficient inference on TT hardware
- **Memory optimization**: Uses channel slicing for memory-efficient processing

## Performance Notes

- The demo uses Conv+BatchNorm fusion for improved performance
- Memory usage is optimized through channel slicing in convolution operations
- Input images are automatically resized to the target resolution
- Processing time depends on image complexity and number of objects

## Troubleshooting

1. **Missing weights file**: Ensure the weights file is downloaded and placed in the correct location
2. **Memory issues**: Try reducing batch size or image resolution for devices with limited memory
3. **Image format**: Supported formats include JPG, PNG, BMP, TIFF
4. **Poor results**: Model works best on street scene images similar to Cityscapes training data

## Example Results

The demo will show:
- Semantic segmentation with different colors for each class
- Instance boundaries for cars, people, etc.
- Combined panoptic view showing both semantic regions and individual objects
- JSON metadata with detailed segment information
