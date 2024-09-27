There are 7 conv's used in model_k model.
**model_k model: 256x256**
- To test the unit_test, run `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_k_new_conv`
- Among 7 convs, 4 convs fail with Out of Memory issue.
- 3 convs having dilation > 1 passed.

**model_k model: 128x128**
- Among 7 convs, 3 convs having dilation > 1 passed.
- Remaining 4 convs are splitted into test_model_k_new_conv_128x128 (without padding) and test_model_k_new_conv_128x128_with_pad (with padding).
- 2 convs passes without padding. The command to test the unit test: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_k_new_conv_128x128`
- 2 convs passes by padding and unpadding the input before and after conv operation. The command to test the unit test: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_k_new_conv_128x128_with_pad`

- On experimenting the tests increasing batch sizes of 2,4,8,16.
- All conv passes with batch size 2 and 4.
- Increasing batch size to 8, 5 convs passed and 2 conv fails with L1 issue.
- Increasing batch sise to 16, all convs fails with L1 issue.

Note: For conv checking purpose batch_size=1 is used even though 32 is used in the model.
