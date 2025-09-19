# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from models.tt_transformers.tt.common import gather_cos_sin, precompute_freqs, rope_scaling_model_factory
from models.tt_transformers.tt.rope import RotaryEmbedding, rotary_embedding_factory


class TestRope:
    """Test suite to compare different RoPE implementations for consistency."""

    def test_basic_rope_vs_precompute_freqs(self):
        """
        Test that compares sin/cos matrices computed by RotaryEmbedding class
        vs precompute_freqs function to check for discrepancies.
        """
        # Test parameters
        dim = 128
        max_seq_len = 1024
        base = 10000.0
        device = torch.device("cpu")

        # Create RotaryEmbedding instance
        rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_seq_len, base=base, device=device)

        # Get cos/sin from RotaryEmbedding
        rope_cos, rope_sin = rope.cos_cached, rope.sin_cached

        # Get cos/sin from precompute_freqs
        precompute_cos, precompute_sin = precompute_freqs(
            dim=dim, end=2 * max_seq_len, theta=base, scale_factor=None, orig_context_len=None
        )
        precompute_cos, precompute_sin = gather_cos_sin(torch.arange(max_seq_len), precompute_cos, precompute_sin)

        print(f"RotaryEmbedding cos shape: {rope_cos.shape}")
        print(f"RotaryEmbedding sin shape: {rope_sin.shape}")
        print(f"precompute_freqs cos shape: {precompute_cos.shape}")
        print(f"precompute_freqs sin shape: {precompute_sin.shape}")

        # Compare shapes
        assert (
            rope_cos.shape == precompute_cos.shape
        ), f"Cos shapes don't match: {rope_cos.shape} vs {precompute_cos.shape}"
        assert (
            rope_sin.shape == precompute_sin.shape
        ), f"Sin shapes don't match: {rope_sin.shape} vs {precompute_sin.shape}"

        # Compare values with tolerance
        cos_diff = torch.abs(rope_cos - precompute_cos)
        sin_diff = torch.abs(rope_sin - precompute_sin)

        max_cos_diff = torch.max(cos_diff)
        max_sin_diff = torch.max(sin_diff)

        print(f"Max cos difference: {max_cos_diff}")
        print(f"Max sin difference: {max_sin_diff}")
        print(f"Mean cos difference: {torch.mean(cos_diff)}")
        print(f"Mean sin difference: {torch.mean(sin_diff)}")

        # Allow for small numerical differences
        tolerance = 1e-6
        assert max_cos_diff < tolerance, f"Cos values differ by more than {tolerance}: {max_cos_diff}"
        assert max_sin_diff < tolerance, f"Sin values differ by more than {tolerance}: {max_sin_diff}"

    def test_rope_llama3_scaling(self):
        """
        Test that the shape of the cos/sin matrices is correct for yarn scaling.
        """
        dim = 128
        max_seq_len = 1024
        base = 10000.0
        device = torch.device("cpu")

        rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_seq_len, base=base, device=device)
        rope_cos, rope_sin = rope.cos_cached, rope.sin_cached

        rope_llama_model = rope_scaling_model_factory(
            {"rope_type": "llama3", "factor": 32, "original_max_position_embeddings": 8192}
        )
        rope_llama_scaled = rotary_embedding_factory(
            dim=dim, max_position_embeddings=max_seq_len, base=base, rope_scaling=rope_llama_model
        )
        rope_llama_scaled_cos, rope_llama_scaled_sin = rope_llama_scaled.cos_cached, rope_llama_scaled.sin_cached

        assert rope_llama_scaled_cos.shape == rope_cos.shape == (1, 1, max_seq_len, dim)
        assert rope_llama_scaled_sin.shape == rope_sin.shape == (1, 1, max_seq_len, dim)

        cos_diff = torch.abs(rope_cos - rope_llama_scaled_cos)
        sin_diff = torch.abs(rope_sin - rope_llama_scaled_sin)

        max_cos_diff = torch.max(cos_diff)
        max_sin_diff = torch.max(sin_diff)

        print(f"Max cos difference: {max_cos_diff}")
        print(f"Max sin difference: {max_sin_diff}")
        print(f"Mean cos difference: {torch.mean(cos_diff)}")
        print(f"Mean sin difference: {torch.mean(sin_diff)}")

        # Make sure we actually ran the scaling
        assert max_cos_diff > 1e-6, f"Cos values are the same as non scaled. Max diff = {max_cos_diff}"
        assert max_sin_diff > 1e-6, f"Sin values are the same as non scaled. Max diff = {max_sin_diff}"

    def test_rope_yarn_scaling(self):
        """
        Test that the shape of the cos/sin matrices is correct for yarn scaling.
        """
        dim = 128
        max_seq_len = 1024
        base = 10000.0
        device = torch.device("cpu")

        rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_seq_len, base=base, device=device)
        rope_cos, rope_sin = rope.cos_cached, rope.sin_cached

        rope_yarn_model = rope_scaling_model_factory(
            {"rope_type": "yarn", "factor": 32, "original_max_position_embeddings": 8192}
        )
        rope_yarn_scaled = rotary_embedding_factory(
            dim=dim, max_position_embeddings=max_seq_len, base=base, rope_scaling=rope_yarn_model
        )
        rope_yarn_scaled_cos, rope_yarn_scaled_sin = rope_yarn_scaled.cos_cached, rope_yarn_scaled.sin_cached

        assert rope_yarn_scaled_cos.shape == rope_cos.shape == (1, 1, max_seq_len, dim)
        assert rope_yarn_scaled_sin.shape == rope_sin.shape == (1, 1, max_seq_len, dim)

        cos_diff = torch.abs(rope_cos - rope_yarn_scaled_cos)
        sin_diff = torch.abs(rope_sin - rope_yarn_scaled_sin)

        max_cos_diff = torch.max(cos_diff)
        max_sin_diff = torch.max(sin_diff)

        print(f"Max cos difference: {max_cos_diff}")
        print(f"Max sin difference: {max_sin_diff}")
        print(f"Mean cos difference: {torch.mean(cos_diff)}")
        print(f"Mean sin difference: {torch.mean(sin_diff)}")

        # Make sure we actually ran the scaling
        assert max_cos_diff > 1e-6, f"Cos values are the same as non scaled. Max diff = {max_cos_diff}"
        assert max_sin_diff > 1e-6, f"Sin values are the same as non scaled. Max diff = {max_sin_diff}"

    @pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": True, "trace_region_size": 30000000, "num_command_queues": 1}],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mesh_device",
        [
            {
                "N150": (1, 1),
                "N300": (1, 2),
                "N150x4": (1, 4),
                "T3K": (1, 8),
                "TG": (8, 4),
                "P150": (1, 1),
                "P300": (1, 2),
                "P150x4": (1, 4),
                "P150x8": (1, 8),
            }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
        ],
        indirect=True,
    )
    def test_rope_gemma(self, mesh_device):
        import copy

        from transformers import AutoConfig
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

        from models.demos.gemma3.tt.model_config import ModelArgs
        from models.tt_transformers.tt.model import Transformer

        config = AutoConfig.from_pretrained("/proj_sw/user_dev/google/gemma-3-27b-it").text_config
        gemma_rope = Gemma3RotaryEmbedding(config)
        local_config = copy.deepcopy(config)
        local_config.rope_theta = local_config.rope_local_base_freq
        local_config.rope_scaling = {"rope_type": "default"}
        gemma_rope_local = Gemma3RotaryEmbedding(local_config)
        max_seq_len = 4096
        tt_model_args = ModelArgs(
            mesh_device,
            instruct=False,
            max_batch_size=1,
            max_seq_len=max_seq_len,
        )
        state_dict = tt_model_args.load_state_dict()
        model = Transformer(
            args=tt_model_args,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            weight_cache_path=tt_model_args.weight_cache_path(ttnn.bfloat8_b),
        )

        x = torch.tensor([0.0])
        ttnn_cos_matrix = ttnn.to_torch(
            model.rope_setup.cos_matrix,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=(1, 8)),
        )[0]
        ttnn_sin_matrix = ttnn.to_torch(
            model.rope_setup.sin_matrix,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=(1, 8)),
        )[0]

        ttnn_local_cos_matrix = ttnn.to_torch(
            model.rope_local_setup.cos_matrix,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=(1, 8)),
        )[0]
        ttnn_local_sin_matrix = ttnn.to_torch(
            model.rope_local_setup.sin_matrix,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=(1, 8)),
        )[0]

        max_cos_diff = []
        max_sin_diff = []
        max_local_cos_diff = []
        max_local_sin_diff = []
        for i in range(max_seq_len):
            max_cos_diff.append(
                torch.abs(ttnn_cos_matrix[..., i, ::2] - gemma_rope(x, torch.tensor([[i]]))[0][..., :64]).max()
            )
            max_sin_diff.append(
                torch.abs(ttnn_sin_matrix[..., i, ::2] - gemma_rope(x, torch.tensor([[i]]))[1][..., :64]).max()
            )

            max_local_cos_diff.append(
                torch.abs(
                    ttnn_local_cos_matrix[..., i, ::2] - gemma_rope_local(x, torch.tensor([[i]]))[0][..., :64]
                ).max()
            )
            max_local_sin_diff.append(
                torch.abs(
                    ttnn_local_sin_matrix[..., i, ::2] - gemma_rope_local(x, torch.tensor([[i]]))[1][..., :64]
                ).max()
            )

        print(f"Max cos diff: {max(max_cos_diff)}")
        print(f"Max sin diff: {max(max_sin_diff)}")
        print(f"Max local cos diff: {max(max_local_cos_diff)}")
        print(f"Max local sin diff: {max(max_local_sin_diff)}")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(max_cos_diff, label="Global")
        ax[0].plot(max_local_cos_diff, label="Local")
        ax[0].legend()
        ax[1].plot(max_sin_diff, label="Global")
        ax[1].plot(max_local_sin_diff, label="Local")
        ax[1].legend()

        plt.savefig("max_cos_sin_diff.png")

    @pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": True, "trace_region_size": 30000000, "num_command_queues": 1}],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mesh_device",
        [
            {
                "N150": (1, 1),
                "N300": (1, 2),
                "N150x4": (1, 4),
                "T3K": (1, 8),
                "TG": (8, 4),
                "P150": (1, 1),
                "P300": (1, 2),
                "P150x4": (1, 4),
                "P150x8": (1, 8),
            }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
        ],
        indirect=True,
    )
    def test_rope_qwen(self, mesh_device):
        import copy

        from transformers import AutoConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding

        from models.demos.qwen25_vl.tt.model_config import ModelArgs
        from models.tt_transformers.tt.model import Transformer

        config = AutoConfig.from_pretrained("/proj_sw/user_dev/Qwen/Qwen2.5-VL-7B-Instruct")
        qwen_rope = Qwen2_5_VLRotaryEmbedding(config)
        local_config = copy.deepcopy(config)
        max_seq_len = 4096
        tt_model_args = ModelArgs(
            mesh_device,
            instruct=False,
            max_batch_size=1,
            max_seq_len=max_seq_len,
        )
        state_dict = tt_model_args.load_state_dict()
        model = Transformer(
            args=tt_model_args,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            weight_cache_path=tt_model_args.weight_cache_path(ttnn.bfloat8_b),
        )

        x = torch.tensor([0.0])
        ttnn_cos_matrix = ttnn.to_torch(
            model.rope_setup.cos_matrix,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=(1, 2)),
        )[0]
        ttnn_sin_matrix = ttnn.to_torch(
            model.rope_setup.sin_matrix,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=(1, 2)),
        )[0]

        max_cos_diff = []
        max_sin_diff = []
        for i in range(max_seq_len):
            max_cos_diff.append(
                torch.abs(ttnn_cos_matrix[..., i, ::2] - qwen_rope(x, torch.tensor([[[i]]]))[0][..., :64]).max()
            )
            max_sin_diff.append(
                torch.abs(ttnn_sin_matrix[..., i, ::2] - qwen_rope(x, torch.tensor([[[i]]]))[1][..., :64]).max()
            )

        print(f"Max cos diff: {max(max_cos_diff)}")
        print(f"Max sin diff: {max(max_sin_diff)}")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(max_cos_diff, label="Global")
        ax[0].legend()
        ax[1].plot(max_sin_diff, label="Global")
        ax[1].legend()

        plt.savefig("max_cos_sin_diff_qwen.png")


if __name__ == "__main__":
    # Run a quick test if executed directly
    test_instance = TestRope()
    test_instance.test_basic_rope_vs_precompute_freqs()
    test_instance.test_rope_llama3_scaling_shape()
    test_instance.test_rope_yarn_scaling_shape()
    print("All tests passed!")
