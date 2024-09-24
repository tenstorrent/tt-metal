**Run Commands:**
Here are the commands to run the Tensor Parallel submodules of Llama3.1_8b model:
MLP Submodule: `pytest models/demos/wormhole/llama31_8b/tests/test_llama_mlp.py`
RMS Norm Submodule: `pytest models/demos/wormhole/llama31_8b/tests/test_llama_rms_norm.py`
Attention Submodule: `pytest models/demos/wormhole/llama31_8b/tests/test_llama_attention.py`

**Pending issues:**
- As suggested to use distributed RMS Norm, tried to implement Distributed RMS Norm submodule. Blocked with an error, raised issue. More details on issue [#12427](https://github.com/tenstorrent/tt-metal/issues/12427)
- The PCC of Attention module is 0.73. On improving PCC of module, replaced `nlp_create_qkv_heads` API with `nlp_create_qkv_heads_decode` API for creating qkv heads in multi-device. Blocked with an error, raised issue. More details on issue [#12428](https://github.com/tenstorrent/tt-metal/issues/12428)
- On implementing Tensor Parallel Decoder submodule, blocked with error. More details on issue [#12429](https://github.com/tenstorrent/tt-metal/issues/12429)

**PR Scope**
Solving the pending issues gives scope to proceed further in implementing Tensor Parallel Llama3.1_8b model.
