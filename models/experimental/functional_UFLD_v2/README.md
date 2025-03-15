# Ultra-Fast-Lane-Detection-v2

### Platforms:
    WH N300

### Introduction

The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

Dataset Link - [source](https://github.com/TuSimple/tusimple-benchmark/issues/3), [kaggle](https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download)


The repository includes models trained on various datasets such as [CULane](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/model/model_culane.py), [CurveLanes](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/model/model_curvelanes.py), and [Tusimple](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/model/model_tusimple.py)



### Model Details

- The entry point to the UFLD_v2 is located at:`models/experimental/functional_UFLD_v2/ttnn/ttnn_UFLD_v2.py`
- The model picks up trained weights from the **tusimple_res34.pth** file located at:`models/experimental/functional_UFLD_v2/reference/tusimple_res34.pth`
- Batch Size :2


### Owner: [Venkatesh](https://github.com/vguduruTT)
