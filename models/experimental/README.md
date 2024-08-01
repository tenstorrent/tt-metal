**Encoder block of model_net when usemaxpooling=True**
- Contains 4 convs and 4 maxpools.
- Unit tested different input resolutions of encoder block.

**model_net: 4094x510**
- To run Conv unit test of model_net 4094x510 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_4094x510`
- Among 4 convs, 1 conv passed. 3 convs fail with OOM issue
- To run MaxPool unit test of model_net 4094x510 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_maxpool2d.py::test_model_net_max_pool_4094x510`
- Among 4 maxpools, 2 maxpools fails with OOM issue, 2 maxpools fails with with valid_page_size error

**model_net: 2047x255**
- To run Conv unit test of model_net 2047x255 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_2047x255`
- Among 4 convs, 2 convs passed, 2 convs fails with OOM issue
- To run MaxPool unit test of model_net 2047x255 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_maxpool2d.py::test_model_net_max_pool_2047x255`
- Among 4 maxpools, 3 maxpools fails with with valid_page_size error, 1 maxpools fails with OOM issue

**model_net: 1024x128**
- To run Conv unit test of model_net 1024x128 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_1024x128`
- Among 4 convs, 3 convs passed, 1 conv fails with OOM issue
- To run MaxPool unit test of model_net 1024x128 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_maxpool2d.py::test_model_net_max_pool_1024x128`
- Among 4 maxpools, 1 maxpool passed with Bfloat16, 3 maxpools fails with with valid_page_size error

**model_net: 512x64**
- To run Conv unit test of model_net 512x64 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_512x64`
- Among 4 convs, 3 convs passed, 1 conv fails with OOM issue
- To run MaxPool unit test of model_net 512x64 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_maxpool2d.py::test_model_net_max_pool_512x64`
- Among 4 maxpools, 2 maxpool passed with Bfloat16, 2 maxpools fails with with valid_page_size error

**Encoder block of model_net usemaxpooling=False 2047x255**
- Among 4 convs, all the Convs passed.
- To run the Conv unit test of Encoder block 2047x255 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_encoder_2047x255_max_False`

**Decoder block of model_net usemaxpooling=False**
- Decoder block contains 4 Upsamples, 3 Concats, 1 Conv and 1 Sigmoid ops.
- To run the upsample unit tests of decoder block for 4094x510 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_upsample.py::test_model_net_upsample_4094x510`
- To run the upsample unit tests of decoder block for 2047x255 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_upsample.py::test_model_net_upsample_2047x255`
- To run the concat unit tests of decoder block for 4094x510 and 2047x255 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_concat.py::test_concat_model_net`
- To run the conv unit tests of decoder block for 4094x510 and 2047x255 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_decoder`
- To run the sigmoid unit tests of decoder block for 4094x510 and 2047x255 resolution, run the command: `pytest tests/ttnn/unit_tests/operations/test_activation.py::test_sigmoid_model_net`
