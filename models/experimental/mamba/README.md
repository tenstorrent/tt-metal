# Mamba Model Demo

## Run on CPU
```
python demo/demo.py "What is the capital of France?" "What is the capital of Germany?" --model cpu
```

## Run on Wormhole
```
python demo/demo.py "What is the capital of France?" "What is the capital of Germany?" --model wh
```
# Mamba Unit Tests
`model-version : mamba-370m`
## SSM
```
pytest -svv tests/test_mamba_ssm.py::test_mamba_ssm_inference[state-spaces/mamba-370m-1-0.99]
```
## Mamba Block
```
pytest -svv tests/test_mamba_block.py::test_mamba_block_inference[state-spaces/mamba-370m-1-0.99]
```
## Residual Block
```
pytest -svv tests/test_residual_block.py::test_residual_block_inference[state-spaces/mamba-370m-1-0.99]
```
## Full Model
Note : embedding layer amd LM head are on CPU
```
pytest -svv tests/test_full_model.py::test_mamba_model_inference[state-spaces/mamba-370m-1-0.99]
```
