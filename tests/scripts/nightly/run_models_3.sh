# wormhole_b0 only
env pytest tests/ttnn/integration_tests/unet                # -> failing: issue #7556
# env pytest tests/ttnn/integration_tests/stable_diffusion    # -> failing/hanging: issue #7560

env pytest models/experimental/mamba/tests/test_full_model.py       # -> failing with autoformat error: issue #7551
env pytest models/experimental/mamba/tests_opt/test_full_model.py
env pytest models/experimental/mamba/tests/test_benchmarks.py
env pytest models/experimental/mamba/tests/test_demo.py             # -> failing with autoformat error: issue #7551

env pytest models/demos/mistral7b/tests/test_mistral_embedding.py
env pytest models/demos/mistral7b/tests/test_mistral_rms_norm.py
env pytest models/demos/mistral7b/tests/test_mistral_mlp.py
env pytest models/demos/mistral7b/tests/test_mistral_attention.py
env pytest models/demos/mistral7b/tests/test_mistral_decoder.py     # -> failing: issue #7524
