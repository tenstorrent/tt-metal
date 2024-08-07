**Unit tests of failing Maxpools of each resolution**

**When Maxpooling=True, Encoder res: 4094x510**
- To reproduce the issue, run the command: `pytest tests/ttnn/unit_tests/operations/test_max_pool2d.py::test_model_net_max_pool_4094x510`
- Among 4 maxpools, 2 maxpools fails with OOM issue
- 2 maxpools passed with Bfloat16, but the same configurations fails in Bfloat8_b

**When Maxpooling=True, Encoder res: 2047x255**
- To reproduce the issue, run the command: `pytest tests/ttnn/unit_tests/operations/test_max_pool2d.py::test_model_net_max_pool_2047x255`
- Among 4 maxpools, 1 maxpool fails with OOM issue
- 3 maxpools passed with Bfloat16, but the same configurations fails in Bfloat8_b
