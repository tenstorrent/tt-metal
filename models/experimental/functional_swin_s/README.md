To run the Shifted Window Attention Submodule of Swin_s functional model, run the command: pytest tests/ttnn/integration_tests/swin_s/test_ttnn_shifted_window_attention.py
To run the MLP submodule, run the command: pytest tests/ttnn/integration_tests/swin_s/test_ttnn_mlp.py
To run the Swin Transformer Block submodule, run the command: pytest tests/ttnn/integration_tests/swin_s/test_ttnn_swin_transformer_block.py
To run the PatchMerging submodule, run the command: pytest tests/ttnn/integration_tests/swin_s/test_ttnn_patchmerging.py
To run the Swin Transformer model pipeline, run the command: pytest tests/ttnn/integration_tests/swin_s/test_ttnn_swin_transformer.py
On testing the Swin Transformer model pipeline, facing the error:
RuntimeError: TT_FATAL @ ../tt_metal/impl/buffers/buffer.cpp:31: valid_page_size
info:
For valid non-interleaved buffers page size 6 must equal buffer size 1012. For interleaved-buffers page size should be divisible by buffer size
