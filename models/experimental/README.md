**Unit tests of failing conv of each resolution**

**When Maxpooling=True, Encoder res: 4094x510**
- To reproduce the issue, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_encoder_4094x510_maxpooling_True`
- Among 4 convs, 1 conv passed. 3 convs fail with OOM issue

**When Maxpooling=True, Encoder res: 2047x255**
- To reproduce the issue, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_encoder_2047x255_maxpooling_True`
- Among 4 convs, 2 convs passed. 2 convs fail with OOM issue

**When Maxpooling=False, Encoder res: 4094x510**
- To reproduce the issue, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_encoder_4094x510_maxpooling_False`
- Among 4 convs, 1 conv passed. 3 convs fail with OOM issue

**When Maxpooling=False, Encoder res: 2047x255**
- To reproduce the issue, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_encoder_2047x255_maxpooling_False`
- Among 4 convs, 2 convs passed. 2 convs fail with OOM issue

**Decoder res: 4094x510 & 2047x255**
- To reproduce the issue, run the command: `pytest tests/ttnn/unit_tests/operations/test_new_conv2d.py::test_model_net_decoder`
- Among 2 convs, 2 convs fail with OOM issue
- Among 2 convs, 2 convs fail with OOM issue
