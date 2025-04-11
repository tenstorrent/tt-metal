# Ultra-Fast-Lane-Detection-v2

### Platforms:
    WH N300

### Introduction

The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

### Model Details

- The entry point to the UFLD_v2 is located at:`models/experimental/ufld_v2/ttnn/ttnn_UFLD_v2.py`
- The model picks up trained weights from the **tusimple_res34.pth** file located at:`models/experimental/ufld_v2/reference/tusimple_res34.pth`
- Batch Size :1

### Demo

Use the following command to run the demo :

`pytest models/experimental/ufld_v2/demo/demo.py`

To run the demo on your data:

- Add your images to the 'images' directory and list their filenames in 'input_images.txt' under demo folder
- Annotate the corresponding ground truth labels in 'ground_truth_labels.json' using the required format.


### Owner: [Venkatesh](https://github.com/vguduruTT)
