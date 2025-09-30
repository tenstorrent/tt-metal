In scatter module of Pointpillars model, there is an operation: `indices = this_coors[:, 2] * self.nx + this_coors[:, 3]`
In ttnn the result of multiplication of sliced tensor `this_coors[:, 2]` and scalar `self.nx` is not equivalent to torch operation, instead getting garbage or zero values.

To test the unit test, run the command:
```
pytest models/experimental/pointpillars/test_case/test_pointscatter_unittest.py::test_case_multiply
```
