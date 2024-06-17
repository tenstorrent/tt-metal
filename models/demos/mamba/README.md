# Mamba Model

## Inference Demo

This demo is designed to run Mamba-2.8b on  a `wormhole` card and generate outputs for a set of prompts. We used finetuned version [state-spaces/mamba-2.8b-slimpj](https://huggingface.co/state-spaces/mamba-2.8b-slimpj) for quality outputs. Follow the instructions below to run the demo successfully.

### How to Run

To run the model for a single user you can use the command line input:
```
pytest --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/mamba/demo/demo.py
```

To run the demo using pre-written prompts for a batch of 32 users run:

```
pytest --disable-warnings -q -s --input-method=json --input-path='models/demos/mamba/demo/prompts.json' models/demos/mamba/demo/demo.py
```

To run the demo using custom input prompts, you can provide a different path to the input prompts file for e.g.:

```
pytest --disable-warnings -q -s --input-method=json --input-path='path_to_input_prompts.json' models/demos/mamba/demo/demo.py
```
Note: We currently only support json file with strictly 32 user prompts with same token length. Check the `models/demos/mamba/demo/prompts.json` file for reference.

---

Feel free to reach out if you have any questions or need further assistance!

## Mamba Unit Tests

These unit tests are designed to test the Mamba model and its components. The tests are written using the `pytest` framework.

**Navigate to the `tt-metal` directory**
```bash
cd tt-metal
```

### SSM Block

```
pytest -svv models/demos/mamba/tests/test_mamba_ssm.py
```

### Mamba Block

```
pytest -svv models/demos/mamba/tests/test_mamba_block.py
```

### Residual Block

```
pytest -svv models/demos/mamba/tests/test_residual_block.py
```

### Full Model

Note : input embedding layer amd TopK are on CPU

```
pytest -svv models/demos/mamba/tests/test_full_model.py::test_inference[state-spaces/mamba-2.8b-32-0.98-64-1]
```

## Mamba Model Performance Tests

These tests are designed to evaluate device-side and host performance of Mamba model. The tests are written using the `pytest` framework.
**Navigate to the `tt-metal` directory**

### End-to-End Model Performance

```bash
pytest -svv models/demos/mamba/tests/test_mamba_perf.py -m models_performance_bare_metal
```

### Device-Side Performance

Build with profiler support enabled (use the build script `./scripts/build_scripts/build_with_profiler_opt.sh`) and run the following command to test device-side performance:

```
pytest -svv models/demos/mamba/tests/test_mamba_perf.py -m models_device_performance_bare_metal

```

This will also generate device and host profiling logs in directory `generated/profiler/reports/ttnn_mamba`
