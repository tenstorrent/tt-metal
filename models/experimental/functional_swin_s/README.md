## Swin_S Model

## Platforms
    WH n150

## Test the Swin_S whole model:
- To run the Swin Transformer model pipeline, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_ttnn_swin_transformer.py`

## Test the Swin_S submodules:
- To run the Shifted Window Attention Submodule of Swin_s functional model, run the command: `pytest tests/ttnn/integration_tests/swin_s/- test_ttnn_shifted_window_attention.py`
- To run the MLP submodule, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_ttnn_mlp.py`
- To run the Swin Transformer Block submodule, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_ttnn_swin_transformer_block.py`
- To run the PatchMerging submodule, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_ttnn_patchmerging.py`

## Test the performance of Swin_S model:
- To run the inference of the model, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_swin_s_performant.py::test_run_swin_s_inference`
- To run the trace inference of the model, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_swin_s_performant.py::test_run_swin_s_trace_inference`
- To run the trace 2cq inference of the model, run the command: `pytest tests/ttnn/integration_tests/swin_s/test_swin_s_performant.py::test_run_swin_s_trace_2cq_inference`

## Test the demo of Swin_S model:
- To run the demo of Swin_S model by benchmarking with ImageNet dataset for classification, run the command: `pytest models/experimental/functional_swin_s/demo/demo.py`

### Torch ops: None

## Pending Issues:
- [#2795](https://github.com/tenstorrent/tt-metal/issues/2795) - Add support for masked_fill op
- [#12656](https://github.com/tenstorrent/tt-metal/issues/12656) - Unable to convert some input_tensors to shard
