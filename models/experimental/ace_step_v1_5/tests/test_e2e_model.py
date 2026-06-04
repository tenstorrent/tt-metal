# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the end-to-end ACE-Step v1.5 pipeline.

TestDiTPipelineRandomInputs: TTNN smoke-tests with random weights.
TestDiTPipelinePCC:          Single-forward PCC between torch_ref and TTNN.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import ttnn
from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import PCC_THRESHOLD, assert_pcc_print
from models.experimental.ace_step_v1_5.torch_ref.full_pipeline import AceStepV15TorchPipeline
from models.experimental.ace_step_v1_5.ttnn_impl.full_pipeline import AceStepV15TTNNPipeline


def _make_random_safetensors(
    tmp_path,
    *,
    hidden_size=256,
    in_channels=192,
    patch_size=2,
    head_dim=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    audio_acoustic_hidden_dim=64,
    cond_dim=128,
):
    """Create a minimal safetensors file with random weights matching the real checkpoint layout."""
    D = hidden_size
    sd = {}

    # Patchify / de-patchify
    sd["decoder.proj_in.1.weight"] = torch.randn(D, in_channels, patch_size)
    sd["decoder.proj_in.1.bias"] = torch.randn(D)
    sd["decoder.proj_out.1.weight"] = torch.randn(D, audio_acoustic_hidden_dim, patch_size)
    sd["decoder.proj_out.1.bias"] = torch.randn(audio_acoustic_hidden_dim)

    # Output head
    sd["decoder.norm_out.weight"] = torch.randn(D)
    sd["decoder.scale_shift_table"] = torch.randn(1, 2, D)

    # Condition embedder
    sd["decoder.condition_embedder.weight"] = torch.randn(D, cond_dim)
    sd["decoder.condition_embedder.bias"] = torch.randn(D)

    # Timestep embeddings (linear_1, linear_2, time_proj)
    for te in ("time_embed", "time_embed_r"):
        sd[f"decoder.{te}.linear_1.weight"] = torch.randn(D, D)
        sd[f"decoder.{te}.linear_1.bias"] = torch.randn(D)
        sd[f"decoder.{te}.linear_2.weight"] = torch.randn(D, D)
        sd[f"decoder.{te}.linear_2.bias"] = torch.randn(D)
        sd[f"decoder.{te}.time_proj.weight"] = torch.randn(6 * D, D)
        sd[f"decoder.{te}.time_proj.bias"] = torch.randn(6 * D)

    intermediate_size = D * 3

    def _attn_keys(prefix):
        sd[f"{prefix}.q_proj.weight"] = torch.randn(num_heads * head_dim, D)
        sd[f"{prefix}.q_proj.bias"] = torch.randn(num_heads * head_dim)
        sd[f"{prefix}.k_proj.weight"] = torch.randn(num_kv_heads * head_dim, D)
        sd[f"{prefix}.k_proj.bias"] = torch.randn(num_kv_heads * head_dim)
        sd[f"{prefix}.v_proj.weight"] = torch.randn(num_kv_heads * head_dim, D)
        sd[f"{prefix}.v_proj.bias"] = torch.randn(num_kv_heads * head_dim)
        sd[f"{prefix}.o_proj.weight"] = torch.randn(D, num_heads * head_dim)
        sd[f"{prefix}.q_norm.weight"] = torch.randn(head_dim)
        sd[f"{prefix}.k_norm.weight"] = torch.randn(head_dim)

    for i in range(num_layers):
        pfx = f"decoder.layers.{i}"

        _attn_keys(f"{pfx}.self_attn")
        _attn_keys(f"{pfx}.cross_attn")

        sd[f"{pfx}.self_attn_norm.weight"] = torch.randn(D)
        sd[f"{pfx}.cross_attn_norm.weight"] = torch.randn(D)
        sd[f"{pfx}.mlp_norm.weight"] = torch.randn(D)

        sd[f"{pfx}.mlp.gate_proj.weight"] = torch.randn(intermediate_size, D)
        sd[f"{pfx}.mlp.up_proj.weight"] = torch.randn(intermediate_size, D)
        sd[f"{pfx}.mlp.down_proj.weight"] = torch.randn(D, intermediate_size)

        sd[f"{pfx}.scale_shift_table"] = torch.randn(6, D)

    ckpt_path = tmp_path / "model.safetensors"
    save_file(sd, str(ckpt_path))

    config = {
        "rms_norm_eps": 1e-6,
        "head_dim": head_dim,
        "sliding_window": None,
        "layer_types": ["full_attention"] * num_layers,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    return str(ckpt_path), {
        "hidden_size": D,
        "in_channels": in_channels,
        "patch_size": patch_size,
        "audio_acoustic_hidden_dim": audio_acoustic_hidden_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "cond_dim": cond_dim,
    }


class TestDiTPipelineRandomInputs:
    """Test the TTNN DiT pipeline with random weights and inputs."""

    @pytest.fixture(autouse=True)
    def setup(self, device, torch_seed):
        self.device = device

    def test_pipeline_forward_single_step(self, tmp_path):
        """Test a single forward pass through the DiT pipeline with random weights."""
        ckpt_path, dims = _make_random_safetensors(tmp_path)

        B, T, C = 1, 64, dims["in_channels"]
        cond_dim = dims["cond_dim"]
        timesteps = np.array([1.0, 0.5, 0.0], dtype=np.float32)

        pipe = AceStepV15TTNNPipeline(
            device=self.device,
            checkpoint_safetensors_path=ckpt_path,
            timesteps_host=timesteps,
            expected_input_length=T,
        )

        hidden_states = torch.randn(B, T, C, dtype=torch.bfloat16)
        enc_hs = torch.randn(B, 4, cond_dim, dtype=torch.bfloat16)

        hs_tt = ttnn.from_torch(hidden_states, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        enc_tt = ttnn.from_torch(enc_hs, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        out_tt = pipe.forward(
            hidden_states_btC=hs_tt,
            timestep_index=0,
            encoder_hidden_states_btd=enc_tt,
        )

        out = ttnn.to_torch(out_tt).float()
        assert out.ndim == 3, f"Expected rank 3, got {out.ndim}"
        assert out.shape[0] == B
        assert out.shape[2] == dims["audio_acoustic_hidden_dim"]
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_pipeline_forward_with_cfg_batch2(self, tmp_path):
        """Test forward pass with B=2 (CFG: cond + uncond)."""
        ckpt_path, dims = _make_random_safetensors(tmp_path)

        B, T, C = 2, 32, dims["in_channels"]
        cond_dim = dims["cond_dim"]
        timesteps = np.array([1.0, 0.0], dtype=np.float32)

        pipe = AceStepV15TTNNPipeline(
            device=self.device,
            checkpoint_safetensors_path=ckpt_path,
            timesteps_host=timesteps,
            expected_input_length=T,
        )

        hidden_states = torch.randn(B, T, C, dtype=torch.bfloat16)
        enc_hs = torch.randn(B, 4, cond_dim, dtype=torch.bfloat16)

        hs_tt = ttnn.from_torch(hidden_states, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        enc_tt = ttnn.from_torch(enc_hs, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        out_tt = pipe.forward(
            hidden_states_btC=hs_tt,
            timestep_index=0,
            encoder_hidden_states_btd=enc_tt,
        )

        out = ttnn.to_torch(out_tt).float()
        assert out.shape[0] == B
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_pipeline_forward_with_xt_and_context(self, tmp_path):
        """Test forward using split xt + context_latents inputs."""
        ckpt_path, dims = _make_random_safetensors(tmp_path)

        B, T = 1, 48
        timesteps = np.array([0.8, 0.4, 0.0], dtype=np.float32)

        pipe = AceStepV15TTNNPipeline(
            device=self.device,
            checkpoint_safetensors_path=ckpt_path,
            timesteps_host=timesteps,
            expected_input_length=T,
        )

        xt = torch.randn(B, T, 64, dtype=torch.bfloat16)
        ctx = torch.randn(B, T, 128, dtype=torch.bfloat16)
        enc_hs = torch.randn(B, 8, dims["cond_dim"], dtype=torch.bfloat16)

        xt_tt = ttnn.from_torch(xt, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ctx_tt = ttnn.from_torch(ctx, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        enc_tt = ttnn.from_torch(enc_hs, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        out_tt = pipe.forward(
            xt_bt64=xt_tt,
            context_latents_bt128=ctx_tt,
            timestep_index=1,
            encoder_hidden_states_btd=enc_tt,
        )

        out = ttnn.to_torch(out_tt).float()
        assert out.ndim == 3
        assert out.shape[0] == B
        assert out.shape[2] == dims["audio_acoustic_hidden_dim"]

    def test_multi_step_denoising_loop(self, tmp_path):
        """Test a multi-step denoising loop mimicking the full inference."""
        ckpt_path, dims = _make_random_safetensors(tmp_path)

        B, T = 1, 32
        num_steps = 3
        t_schedule = [float(i) / num_steps for i in range(num_steps, -1, -1)]
        timesteps = np.asarray(t_schedule + [0.0], dtype=np.float32)

        pipe = AceStepV15TTNNPipeline(
            device=self.device,
            checkpoint_safetensors_path=ckpt_path,
            timesteps_host=timesteps,
            expected_input_length=T,
        )

        xt = torch.randn(B, T, 64, dtype=torch.float32)
        ctx_lat = torch.randn(B, T, 128, dtype=torch.float32)
        enc_hs = torch.randn(B, 4, dims["cond_dim"], dtype=torch.float32)

        def _to_np(t):
            return t.detach().float().cpu().contiguous().numpy()

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG")
        act_dtype = ttnn.bfloat16

        for step_idx in range(num_steps):
            t_curr = t_schedule[step_idx]
            t_next = t_schedule[step_idx + 1] if step_idx < num_steps - 1 else 0.0
            dt = t_curr - t_next

            xt_tt = ttnn.as_tensor(
                _to_np(xt), device=self.device, dtype=act_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
            )
            ctx_tt = ttnn.as_tensor(
                _to_np(ctx_lat), device=self.device, dtype=act_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
            )
            enc_tt = ttnn.as_tensor(
                _to_np(enc_hs), device=self.device, dtype=act_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
            )

            acoustic = pipe.forward(
                xt_bt64=xt_tt,
                context_latents_bt128=ctx_tt,
                timestep_index=step_idx,
                encoder_hidden_states_btd=enc_tt,
            )

            vt = ttnn.to_torch(acoustic).to(torch.float32)
            xt = xt - vt * float(dt)

        assert xt.shape == (B, T, 64)
        assert torch.isfinite(xt).all(), "Denoised output contains non-finite values"


class TestDiTPipelinePCC:
    """Compare single-forward outputs between torch_ref and TTNN pipelines."""

    @pytest.fixture(autouse=True)
    def setup(self, device, torch_seed):
        self.device = device

    def _run_pcc_single_step(self, tmp_path, *, B, T, cond_seq_len, timestep_index, pcc_threshold):
        """Run both torch_ref and TTNN on identical inputs and assert PCC."""
        ckpt_path, dims = _make_random_safetensors(tmp_path)
        cond_dim = dims["cond_dim"]
        timesteps = np.array([1.0, 0.5, 0.0], dtype=np.float32)

        torch_pipe = AceStepV15TorchPipeline(
            checkpoint_safetensors_path=ckpt_path,
            timesteps_host=timesteps,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        ttnn_pipe = AceStepV15TTNNPipeline(
            device=self.device,
            checkpoint_safetensors_path=ckpt_path,
            timesteps_host=timesteps,
            expected_input_length=T,
        )

        # Shared inputs (bf16 to match pipeline precision)
        torch.manual_seed(42)
        xt = torch.randn(B, T, 64, dtype=torch.bfloat16)
        ctx = torch.randn(B, T, 128, dtype=torch.bfloat16)
        enc_hs = torch.randn(B, cond_seq_len, cond_dim, dtype=torch.bfloat16)

        # Torch reference forward
        with torch.inference_mode():
            torch_out = torch_pipe.forward(
                xt_bt64=xt,
                context_latents_bt128=ctx,
                timestep_index=timestep_index,
                encoder_hidden_states_btd=enc_hs,
            ).float()

        # TTNN forward
        def _to_np(t):
            return t.detach().float().cpu().contiguous().numpy()

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG")
        xt_tt = ttnn.as_tensor(
            _to_np(xt), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
        )
        ctx_tt = ttnn.as_tensor(
            _to_np(ctx), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
        )
        enc_tt = ttnn.as_tensor(
            _to_np(enc_hs), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
        )

        out_tt = ttnn_pipe.forward(
            xt_bt64=xt_tt,
            context_latents_bt128=ctx_tt,
            timestep_index=timestep_index,
            encoder_hidden_states_btd=enc_tt,
        )
        ttnn_out = ttnn.to_torch(out_tt).float()

        # Trim to matching shape (TTNN output may have tile-padding on the seq dim)
        min_seq = min(torch_out.shape[1], ttnn_out.shape[1])
        torch_out = torch_out[:, :min_seq, :]
        ttnn_out = ttnn_out[:, :min_seq, :]

        assert (
            torch_out.shape == ttnn_out.shape
        ), f"Shape mismatch: torch={tuple(torch_out.shape)} vs ttnn={tuple(ttnn_out.shape)}"
        assert_pcc_print(
            f"e2e_single_step_B{B}_T{T}_step{timestep_index}",
            torch_out,
            ttnn_out,
            pcc=pcc_threshold,
        )

    def test_pcc_single_step_b1(self, tmp_path):
        """PCC for single forward, B=1."""
        self._run_pcc_single_step(
            tmp_path,
            B=1,
            T=64,
            cond_seq_len=4,
            timestep_index=0,
            pcc_threshold=PCC_THRESHOLD,
        )

    def test_pcc_single_step_b2(self, tmp_path):
        """PCC for single forward, B=2 (CFG batch)."""
        self._run_pcc_single_step(
            tmp_path,
            B=2,
            T=32,
            cond_seq_len=8,
            timestep_index=1,
            pcc_threshold=PCC_THRESHOLD,
        )

    def test_pcc_multi_step_denoising(self, tmp_path):
        """PCC after a 3-step denoising loop (Torch vs TTNN)."""
        ckpt_path, dims = _make_random_safetensors(tmp_path)
        cond_dim = dims["cond_dim"]

        B, T = 1, 32
        num_steps = 3
        t_schedule = [float(i) / num_steps for i in range(num_steps, -1, -1)]
        timesteps = np.asarray(t_schedule + [0.0], dtype=np.float32)

        torch_pipe = AceStepV15TorchPipeline(
            checkpoint_safetensors_path=ckpt_path,
            timesteps_host=timesteps,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        ttnn_pipe = AceStepV15TTNNPipeline(
            device=self.device,
            checkpoint_safetensors_path=ckpt_path,
            timesteps_host=timesteps,
            expected_input_length=T,
        )

        torch.manual_seed(42)
        xt_torch = torch.randn(B, T, 64, dtype=torch.bfloat16)
        ctx = torch.randn(B, T, 128, dtype=torch.bfloat16)
        enc_hs = torch.randn(B, 4, cond_dim, dtype=torch.bfloat16)

        xt_ttnn = xt_torch.clone()

        def _to_np(t):
            return t.detach().float().cpu().contiguous().numpy()

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG")

        for step_idx in range(num_steps):
            t_curr = t_schedule[step_idx]
            t_next = t_schedule[step_idx + 1] if step_idx < num_steps - 1 else 0.0
            dt = t_curr - t_next

            # Torch step
            with torch.inference_mode():
                vt_torch = torch_pipe.forward(
                    xt_bt64=xt_torch,
                    context_latents_bt128=ctx,
                    timestep_index=step_idx,
                    encoder_hidden_states_btd=enc_hs,
                ).float()
            xt_torch = xt_torch.float() - vt_torch * float(dt)
            xt_torch = xt_torch.to(torch.bfloat16)

            # TTNN step
            xt_tt = ttnn.as_tensor(
                _to_np(xt_ttnn),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mem,
            )
            ctx_tt = ttnn.as_tensor(
                _to_np(ctx), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
            )
            enc_tt = ttnn.as_tensor(
                _to_np(enc_hs), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
            )

            vt_ttnn = ttnn.to_torch(
                ttnn_pipe.forward(
                    xt_bt64=xt_tt,
                    context_latents_bt128=ctx_tt,
                    timestep_index=step_idx,
                    encoder_hidden_states_btd=enc_tt,
                )
            ).float()
            xt_ttnn = xt_ttnn.float() - vt_ttnn * float(dt)
            xt_ttnn = xt_ttnn.to(torch.bfloat16)

        torch_final = xt_torch.float()
        ttnn_final = xt_ttnn.float()

        min_seq = min(torch_final.shape[1], ttnn_final.shape[1])
        torch_final = torch_final[:, :min_seq, :]
        ttnn_final = ttnn_final[:, :min_seq, :]

        assert_pcc_print(f"e2e_multi_step_denoising_{num_steps}steps", torch_final, ttnn_final)
