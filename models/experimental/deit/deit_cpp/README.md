# DeiT Model Using TTNN

## Platforms:
- WH N300

## Details
- The C++ implementation provides end-to-end executables under build/test_deit and is expected to be comparable to the Python reference tests in models/experimental/deit/tests for functionality and throughput. The primary generated binary for end-to-end verification is test_deit_model.
- The program uses libtorch APIs to load TorchScript-format DeiT models. This project converts Hugging Face DeiT models to TorchScript using models/experimental/deit/deit_cpp/deit_model/bin2pt.py, and all subsequent tests operate on the converted .pt files.
- Image classification tests use OpenCV-based preprocessing (image_utils.*) and rely on BUILD_DEIT_OPENCV to fetch libtorch and OpenCV via CPM.

**Note:** You can specify the model file location by setting the DEIT_MODEL_FILE_PATH environment variable and passing it to the executables, or the examples will use the default TorchScript files under ${TT_METAL_HOME}/models/experimental/deit/deit_cpp/deit_model/ (e.g., deit_classifier_model.pt).

## Build
- Use the top-level build script and enable OpenCV and LibTorch support:
  - ./build_metal.sh --enable-deit-opencv

## Quick Start
- Step 1: Build with DeiT OpenCV support
  - ./build_metal.sh --enable-deit-opencv
- Step 2: Download and convert models to TorchScript
  - cd models/experimental/deit/deit_cpp/deit_model
  - python3 bin2pt.py
- Step 3: Execute binaries against the exported .pt models
  - ./build/test_deit/test_deit_model models/experimental/deit/deit_cpp/deit_model/deit_classifier_model.pt
  - ./build/test_deit/test_deit_for_image_classification models/experimental/deit/deit_cpp/deit_model/deit_classifier_model.pt
  - ./build/test_deit/test_deit_for_image_classification_with_teacher models/experimental/deit/deit_cpp/deit_model/deit_teacher_model.pt


## Binaries
- build/test_deit/test_deit_model
- build/test_deit/test_deit_for_image_classification
- build/test_deit/test_deit_for_image_classification_with_teacher

## Models and Assets
- TorchScript models:
  - [deit_encoder_model.pt](deit_model/deit_encoder_model.pt) — traced encoder returning last_hidden_state
  - [deit_classifier_model.pt](deit_model/deit_classifier_model.pt) — traced classifier returning logits
  - [deit_teacher_model.pt](deit_model/deit_teacher_model.pt) — traced distilled variant returning logits, cls_logits, distillation_logits
- Sample image:
  - [input_image.jpg](deit_model/input_image.jpg) — used by image classification tests with OpenCV preprocessing
- Export script:
  - [bin2pt.py](deit_model/bin2pt.py) — downloads Hugging Face models and exports TorchScript files used by the C++ tests

## Run
- Image classification:
  - ./build/test_deit/test_deit_for_image_classification models/experimental/deit/deit_cpp/deit_model/deit_classifier_model.pt
- Distilled teacher variant:
  - ./build/test_deit/test_deit_for_image_classification_with_teacher models/experimental/deit/deit_cpp/deit_model/deit_teacher_model.pt
- Test Perf:
  - ./build/test_deit/test_deit_e2e_optimized models/experimental/deit/deit_cpp/deit_model/deit_classifier_model.pt


## Result
- Test Perf:
  - ttnn_deit_224x224_batch_size_1. One inference iteration time (sec): 0.012335, FPS: 81.07, inference time (sec): 0.077563, sync output time(sec): 0.045783
