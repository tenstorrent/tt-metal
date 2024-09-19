# FADNetPP ttnn Demo

## Inputs required:
- For sample images, download the sample pack from this [site](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- Unzip the sample pack, in Monka->RGB_cleanpass place one left and right image in `models/experimental/functional_fadnetpp/demo/`.
- In demo.py, replace the images name on line no. 72 and 73.
- current branch  - `keerthanar/fadnetpp_bringup`.
- Test file available in `tests/ttnn/integration_tests/fadnetpp/`
- There is no pretrained weights for this model

## To run the demo:
1. Checkout to branch `keerthanar/fadnetpp_bringup`.
2. Run `pytest models/experimental/functional_fadnetpp/demo/demo.py`.
3. Result image will be generated at `tests/ttnn/integration_tests/fadnetpp/`.
