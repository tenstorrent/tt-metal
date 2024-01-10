# TTNN Sweep Tests

## Running all sweeps
```
python tests/ttnn/sweep_tests/run_all_tests.py
```

## Printing report of all sweeps
```
python tests/ttnn/sweep_tests/print_report.py [--detailed]
```

## Debugging sweeps
```
python tests/ttnn/sweep_tests/run_failed_and_crashed_tests.py [--exclude add,linear] [--stepwise]
```

## Running a single test
```
python tests/ttnn/sweep_tests/run_single_test.py --test-name add --index 0
```

## Adding a new sweep test
In `tests/ttnn/sweep_tests/sweeps` add a new file `<new_file>.py`.

The file must contain:
- `parameters` dictionary from a variable to the list of values to sweep
- `skip` function for filtering out unwanted combinations. It should return `bool`
- `run` function for running the test. It should return `Tuple[bool, Optional[str]]`. Second element of the tuple is the error message

For example, let's add `tests/ttnn/sweep_tests/sweeps/to_and_from_device.py`:
```python

import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

parameters = {
    "height": [1, 32],
    "width": [1, 32],
}

def skip(height, width):
    if height == 1 and width == 1:
        return True
    return False

def run(height, width, *, device):
    torch_tensor = torch.zeros((height, width))

    tensor = ttnn.from_torch(torch_tensor, device=device)
    tensor = ttnn.to_torch(tensor)

    return check_with_pcc(torch_tensor, tensor)

```
