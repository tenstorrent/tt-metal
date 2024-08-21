# TTNN Sweep Tests

## Running all sweeps
```
python tests/ttnn/sweep_tests/run_sweeps.py
```

## Running a single sweep
```
python tests/ttnn/sweep_tests/run_sweeps.py --include add.py,matmul.py
```

## Printing report of all sweeps
```
python tests/ttnn/sweep_tests/print_report.py [--detailed]
```


## Using Pytest to run sweeps all the sweeps for one operation file
```
pytest <full-path-to-tt-metal>/tt-metal/tests/ttnn/sweep_tests/test_sweeps.py::test_<operation>
Example for matmul: pytest tests/ttnn/sweep_tests/test_sweeps.py::test_matmul
```

## Using Pytest to run a single sweep test by the index
```
pytest <full-path-to-tt-metal>/tt-metal/tests/ttnn/sweep_tests/test_sweeps.py::test_<operation>[<operation>.py-<index-of-test-instance>]
Example for matmul: TODO(arakhmati)
```

## Adding a new sweep test
In `tests/ttnn/sweep_tests/sweeps` add a new file `<new_file>.py`. (You can new folders as well)

The file must contain:
- `parameters` dictionary from a variable to the list of values to sweep
- optional `skip` function for filtering out unwanted combinations. It should return `Tuple[bool, Optional[str]]`. Second element of the tuple is the reason to skip the test.
- optional `xfail` function for marking the test as expected to fail. It should return `Tuple[bool, Optional[str]]`. Second element of the tuple is the expected exception.
- `run` function for running the test. It should return `Tuple[bool, Optional[str]]`. Second element of the tuple is the error message.

For example, let's add `tests/ttnn/sweep_tests/sweeps/to_and_from_device.py`:
```python

import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

parameters = {
    "height": [1, 32],
    "width": [1, 32],
}


def run(height, width, *, device) -> Tuple[bool, Optional[str]]:
    torch_tensor = torch.zeros((height, width))

    tensor = ttnn.from_torch(torch_tensor, device=device)
    tensor = ttnn.to_torch(tensor)

    return check_with_pcc(torch_tensor, tensor)

```
