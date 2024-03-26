# Mamba Model Demo

## Run on CPU
```
python demo/demo.py "What is the capital of France?" "What is the capital of Germany?" --model cpu
```

## Run on Wormhole
We have two functional model implementions on device, `wh` and `wh-opt`. `wh` is non-performant model implemented in ROW-MAJOR layout and `wh-opt` is the performant model in TILE layout. 

ROW-MAJOR layout implementation

```
python demo/demo.py "What is the capital of France?" "What is the capital of Germany?" --model wh
```
TILE layout implementation

```
python demo/demo.py "What is the capital of France?" "What is the capital of Germany?" --model wh-opt
```
# Mamba Unit Tests
ROW-MAJOR layout model implementation and tests can be found in `tt` and `tests` directory respectively. TILE layout model implementation and tests can be found in `tt_opt` and `tests_opt` directory respectively. In future, we will archive the ROW-MAJOR models and maintain a single directory for our performant version.
## SSM
ROW-MAJOR layout implementation
```
pytest -svv tests/test_mamba_ssm.py
```
TILE layout implementation
```
pytest -svv tests_opt/test_mamba_ssm.py
```
## Mamba Block
ROW-MAJOR layout implementation
```
pytest -svv tests/test_mamba_block.py
```
TILE layout implementation
```
pytest -svv tests_opt/test_mamba_block.py
```
## Residual Block
ROW-MAJOR layout implementation
```
pytest -svv tests/test_residual_block.py
```
TILE layout implementation
```
pytest -svv tests_opt/test_residual_block.py
```
## Full Model
Note : embedding layer amd LM head are on CPU
ROW-MAJOR layout implementation
```
pytest -svv tests/test_full_model.py
```
TILE layout implementation
```
pytest -svv tests_opt/test_full_model.py
```