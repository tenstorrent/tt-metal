# Commands to test the sub_modules

- To run AdaLayerNormContinuous sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_ada_layernorm_continuous.py`
- To run AdaLayerNormZero sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_ada_layernorm_zero.py`
- To run Attention sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_attention.py`
- To run CombinedTimestepTextProjEmbeddings sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_combined_time_step_text_proj_emd.py`
- To run FeedForward sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_feed_forward.py`
- To run GELU sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_gelu.py`
- To run JointTransformerBlock sub_module -`pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_joint_transformer_block.py`
- To run PatchEmbed sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_patch_embed.py`
- To run PixArtAlphaTextProjection sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_pix_art_alpha_text_projection.py`
- To run RMSNorm sub_module -
- To run SD35AdaLayerNormZeroX sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_sd35_ada_layernorm_zerox.py`
- To run TimestepEmbedding sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_time_step_embeddings.py`
- To run Timesteps sub_module - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_time_steps.py`

# To run the whole model:
- Update the diffusers package using - `pip install -U diffusers`
- Make sure you have the saved tensors of model in the `models/experimental/functional_stable_diffusion3_5/reference/` path.
- Run command - `pytest tests/ttnn/integration_tests/stable_diffusion3_5/test_ttnn_sd3_transformer_2d_model.py`
- For real inputs the pcc is 0.94 and for random inputs the pcc is 0.98.
