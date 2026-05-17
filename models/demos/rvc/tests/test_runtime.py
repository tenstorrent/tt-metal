# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime lifecycle validation tests.

Tests that persistent TTNN modules can:
1. Load weights once from checkpoint
2. Execute multiple forward passes without OOM
3. Produce correct results matching torch reference
4. Produce deterministic results across runs
5. Clean up device memory properly
"""

import os
import pytest
import torch
import ttnn

from safetensors.torch import load_file

from models.demos.rvc.tests.pcc_utils import compute_pcc, assert_pcc
from models.demos.rvc.ttnn.runtime import TTNNFlowDecoder, TTNNGeneratorNSF
from models.demos.rvc.torch_impl.reference import (
    load_flow_torch_modules, torch_flow_forward,
    build_torch_generator, torch_generator_forward,
)


CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "assets", "pretrained_v2", "f0G48k.safetensors"
)


@pytest.fixture(scope="module")
def checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    return load_file(CHECKPOINT_PATH)


class TestFlowRuntime:
    """Persistent flow decoder runtime validation."""

    def test_repeated_forward(self, device, checkpoint):
        """Run flow 5 times with same weights — NO OOM."""
        torch.manual_seed(42)
        flow = TTNNFlowDecoder.from_checkpoint(checkpoint, device)

        emb_g = torch.nn.Embedding(109, 256)
        emb_g.weight.data = checkpoint["emb_g.weight"].float()
        g = emb_g(torch.tensor([0])).unsqueeze(-1)
        z_p = torch.randn(1, 192, 30)

        results = []
        for i in range(5):
            z = flow(z_p, g)
            results.append(z.clone())
            print(f"    Run {i}: shape={z.shape}, mean={z.mean():.6f}")

        for i in range(1, 5):
            diff = (results[0] - results[i]).abs().max().item()
            assert diff == 0.0, f"Non-determinism at run {i}: max_diff={diff}"

        flow.deallocate()
        print(f"  Flow: 5/5 repeated runs, bit-identical, no OOM ✓")

    def test_correctness_vs_torch(self, device, checkpoint):
        """Compare persistent flow against torch reference."""
        torch.manual_seed(42)

        flow = TTNNFlowDecoder.from_checkpoint(checkpoint, device)

        emb_g = torch.nn.Embedding(109, 256)
        emb_g.weight.data = checkpoint["emb_g.weight"].float()
        g = emb_g(torch.tensor([0])).unsqueeze(-1)
        z_p = torch.randn(1, 192, 50)

        ref = torch_flow_forward(z_p, g, load_flow_torch_modules(checkpoint))
        out = flow(z_p, g)

        _, pcc = assert_pcc(ref, out, threshold=0.995, op_name="persistent_flow")
        max_err = (ref - out).abs().max().item()

        flow.deallocate()
        print(f"  Persistent flow vs torch: PCC={pcc:.6f}, max_err={max_err:.6f} ✓")


class TestGeneratorRuntime:
    """Persistent generator runtime validation."""

    def test_repeated_forward(self, device, checkpoint):
        """Run generator 3 times with same weights — NO OOM."""
        torch.manual_seed(42)
        gen = TTNNGeneratorNSF.from_checkpoint(checkpoint, device)

        emb_g = torch.nn.Embedding(109, 256)
        emb_g.weight.data = checkpoint["emb_g.weight"].float()
        g = emb_g(torch.tensor([0])).unsqueeze(-1)
        z = torch.randn(1, 192, 10)
        har = torch.randn(1, 1, 10 * 480)

        results = []
        for i in range(3):
            audio = gen(z, har, g)
            results.append(audio.clone())
            print(f"    Run {i}: shape={audio.shape}, range=[{audio.min():.4f}, {audio.max():.4f}]")

        for i in range(1, 3):
            diff = (results[0] - results[i]).abs().max().item()
            assert diff == 0.0, f"Non-determinism at run {i}: max_diff={diff}"

        gen.deallocate()
        print(f"  Generator: 3/3 repeated runs, bit-identical, no OOM ✓")

    def test_correctness_vs_torch(self, device, checkpoint):
        """Compare persistent generator against torch reference."""
        torch.manual_seed(42)

        gen = TTNNGeneratorNSF.from_checkpoint(checkpoint, device)

        emb_g = torch.nn.Embedding(109, 256)
        emb_g.weight.data = checkpoint["emb_g.weight"].float()
        g = emb_g(torch.tensor([0])).unsqueeze(-1)
        z = torch.randn(1, 192, 10)
        har = torch.randn(1, 1, 10 * 480)

        ref = torch_generator_forward(z, har, g, build_torch_generator(checkpoint))
        out = gen(z, har, g)

        assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
        _, pcc = assert_pcc(ref, out, threshold=0.950, op_name="persistent_generator")
        max_err = (ref - out).abs().max().item()

        gen.deallocate()
        print(f"  Persistent generator vs torch: PCC={pcc:.6f}, max_err={max_err:.4f} ✓")


class TestFullPipelineRuntime:
    """Full pipeline with persistent modules — repeated execution."""

    def test_pipeline_repeated_and_correct(self, device, checkpoint):
        """Run full flow+generator pipeline 3 times, compare 1st run to torch.

        Validates: correctness, determinism, and no OOM.
        """
        torch.manual_seed(42)

        flow = TTNNFlowDecoder.from_checkpoint(checkpoint, device)
        gen = TTNNGeneratorNSF.from_checkpoint(checkpoint, device)

        flow_mods = load_flow_torch_modules(checkpoint)
        gen_torch = build_torch_generator(checkpoint)

        emb_g = torch.nn.Embedding(109, 256)
        emb_g.weight.data = checkpoint["emb_g.weight"].float()
        g = emb_g(torch.tensor([0])).unsqueeze(-1)
        z_p = torch.randn(1, 192, 10)
        har = torch.randn(1, 1, 10 * 480)

        # Torch reference
        z_ref = torch_flow_forward(z_p, g, flow_mods)
        audio_ref = torch_generator_forward(z_ref, har, g, gen_torch)

        # Run 3 times
        results = []
        z_ttnn = None
        for i in range(3):
            z = flow(z_p, g)
            if z_ttnn is None:
                z_ttnn = z.clone()
            audio = gen(z, har, g)
            results.append(audio.clone())
            print(f"    Run {i}: shape={audio.shape}, range=[{audio.min():.4f}, {audio.max():.4f}]")

        # Correctness
        flow_pcc = compute_pcc(z_ref, z_ttnn)
        audio_pcc = compute_pcc(audio_ref, results[0])
        assert flow_pcc > 0.995, f"Flow PCC={flow_pcc}"
        assert audio_pcc > 0.950, f"Audio PCC={audio_pcc}"
        assert not torch.isnan(results[0]).any()
        assert results[0].abs().max() <= 1.0
        print(f"    Correctness: flow PCC={flow_pcc:.6f}, audio PCC={audio_pcc:.6f}")

        # Determinism
        for i in range(1, 3):
            diff = (results[0] - results[i]).abs().max().item()
            assert diff == 0.0, f"Non-determinism at run {i}: max_diff={diff}"

        flow.deallocate()
        gen.deallocate()
        print(f"  Full pipeline: 3/3 runs, correct + deterministic + no OOM ✓")
