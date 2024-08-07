There are 7 conv's used in model_k model.

**model_k model: 256x256**
- To test the unit_test, run `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_k_256x256`
- Among 7 convs, 4 convs fail with Out of Memory issue.
- 3 convs having dilation > 1 not supported in new conv API, currently fails with L1 issue or output shape mismatch between torch and ttnn. Required support for dilation>1.

**model_k model: 128x128**
- Among 7 convs, 3 convs having dilation > 1 not supported, currently fails with L1 issue. Required support for dilation>1.
- Among the remaining 4 convs, 1 conv passed, 3 convs fail with Out of Memory issue.
- The command to test the unit test: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_k_128x128`

Note: For conv checking purpose batch_size=1 is used even though 32 is used in the model.
