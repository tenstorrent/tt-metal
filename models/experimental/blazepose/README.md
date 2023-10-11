# Blazepose model. Source description.
The model provided here was taken from the implementation found on [Blazepose-Pytorch-GitHub source](https://github.com/zmurez/MediaPipePyTorch/tree/master). It was as well referenced by [Ailia-Github](https://github.com/axinc-ai/ailia-models), in section for Pose Estimation.
The model [Blazepose-Pytorch-GitHub source](https://github.com/zmurez/MediaPipePyTorch/tree/master) holds ported Mediapipe models (tflite to Pytorch), but the repo was not updated for two years.
It is worth mentioning that this was the only PyTorch implementaion we found on blazepose, where the original one from [Mediapipe](https://github.com/google/mediapipe), even though it has python api, doesn't have PyTorch implementation of the models, because under the hood it uses graph definition language (protobuf) for defining the model graph, where the nodes (calculators) are written using C++.

# Model description.
Blazepose model provided here is upper-body pose estimation model and runs completely on CPU. The model therefore outputs 25 keypoints, whereas the full body pose estimation should output 33 of them.
The model consist of two networks, detection network and landmark network, both provided here.
The example demo-run of the model can be found in [file](blazepose_demo.py).
