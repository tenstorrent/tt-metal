# Model Traced Sweep Files

Generated sweep files for 380 forward TTNN operations.

## File Structure

Each operation has its own file: `{operation_name}_model_traced.py`

## Suites

- `model_traced_sample`: Quick validation with basic configs
- `model_traced`: Full traced configurations from real models

## Operation Types

Files are automatically classified as:
- **Unary**: Single tensor input operations
- **Binary**: Two tensor input operations
- **Multi-input**: Three or more tensor input operations

## Usage

Run individual operation:
```bash
python3 tests/sweep_framework/sweeps_runner.py --module-name model_traced.{op_name}_model_traced --suite model_traced_sample
```

## Notes

- Multi-input operations may need manual adjustment of the run function
- Some operations may not be supported by the master config loader yet
