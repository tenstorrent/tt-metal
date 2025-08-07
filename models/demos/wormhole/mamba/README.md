# Mamba-2.8b

## Platforms:
    Wormhole (n150, n300)

## Introduction
This demo is designed to run Mamba-2.8b and generate outputs for a set of prompts. It uses the finetuned version [state-spaces/mamba-2.8b-slimpj](https://huggingface.co/state-spaces/mamba-2.8b-slimpj) for quality outputs.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run
To run the demo using pre-written prompts for a batch of 32 users run:
```
pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/wormhole/mamba/demo/prompts.json' models/demos/wormhole/mamba/demo/demo.py
```

To run the demo using custom input prompts, you can provide a different path to the input prompts file for e.g.:

```
pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/wormhole/mamba/demo/demo.py
```

Any sequence length is supported. We currently only support JSON file with strictly 1 user prompt or 32 user prompts with same token length. Check the `models/demos/wormhole/mamba/demo/prompts.json` file for reference.

The prefill graph is not currently integrated into the demo. Therefore we currently process the prompt a single token at a time using the decode graph.

## Testing
### Unit Tests
These unit tests are designed to test the Mamba model and its components. The tests are written using the `pytest` framework.

**Navigate to the `tt-metal` directory**
```bash
cd tt-metal
```

#### SSM Block
```
pytest -svv models/demos/wormhole/mamba/tests/test_mamba_ssm.py
```

#### Mamba Block
```
pytest -svv models/demos/wormhole/mamba/tests/test_mamba_block.py
```

#### Residual Block
```
pytest -svv models/demos/wormhole/mamba/tests/test_residual_block.py
```

#### Full Model
Note : input embedding layer and TopK are on CPU

```
pytest -svv models/demos/wormhole/mamba/tests/test_mamba_model.py::test_inference
```

### Performance Tests
These tests are designed to evaluate device-side and host performance of Mamba model. The tests are written using the `pytest` framework.
**Navigate to the `tt-metal` directory**

#### End-to-End Model Performance
```bash
pytest -svv models/demos/wormhole/mamba/tests/test_mamba_perf.py -m models_performance_bare_metal
```

#### Device-Side Performance
Build with profiler support enabled (use the build script `./build_metal.sh --enable-profiler`) and run the following command to test device-side performance:
```
pytest -svv models/demos/wormhole/mamba/tests/test_mamba_perf.py -m models_device_performance_bare_metal
```

This will also generate device and host profiling logs in directory `generated/profiler/reports/ttnn_mamba`
