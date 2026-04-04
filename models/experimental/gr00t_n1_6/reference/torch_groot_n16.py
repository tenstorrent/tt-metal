# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation for GR00T N1.6-3B.

Uses the upstream NVIDIA Isaac-GR00T library to run inference on CPU/GPU.
This provides the ground truth outputs for layer-by-layer PCC validation
against the TTNN implementation.

Requirements:
    pip install isaac-groot torch transformers
    # or clone https://github.com/NVIDIA/Isaac-GR00T
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def load_groot_reference_model(
    model_id: str = "nvidia/GR00T-N1.6-3B",
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load the upstream GR00T N1.6 model from HuggingFace.

    Returns the model and processor for running reference inference.
    """
    try:
        from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6Policy
        from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config as UpstreamConfig

        logger.info(f"Loading GR00T N1.6 from Isaac-GR00T library...")
        config = UpstreamConfig()
        model = Gr00tN1d6Policy(config)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        return model, config
    except ImportError:
        logger.warning("Isaac-GR00T library not available, trying HuggingFace...")

    try:
        from transformers import AutoModel, AutoConfig

        logger.info(f"Loading GR00T N1.6 from HuggingFace: {model_id}")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype,
        )
        model = model.to(device)
        model.eval()
        return model, config
    except Exception as e:
        logger.error(f"Failed to load GR00T reference model: {e}")
        raise


class Gr00tN16ReferenceRunner:
    """
    Wrapper for running reference GR00T N1.6 inference and extracting
    intermediate activations for layer-by-layer comparison.
    """

    def __init__(
        self,
        model_id: str = "nvidia/GR00T-N1.6-3B",
        device: str = "cpu",
    ):
        self.device = device
        self.model, self.config = load_groot_reference_model(model_id, device)
        self._hooks = []
        self._activations = {}

    def _register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks to capture intermediate activations."""
        self._activations = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._activations[name] = output.detach().cpu()
                elif isinstance(output, tuple):
                    self._activations[name] = output[0].detach().cpu()
            return hook

        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def run_vision_encoder(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run just the vision encoder and return features."""
        with torch.no_grad():
            # Access the vision encoder
            if hasattr(self.model, 'backbone'):
                vision = self.model.backbone.vision_encoder
            elif hasattr(self.model, 'vision_model'):
                vision = self.model.vision_model
            else:
                raise AttributeError("Cannot find vision encoder in model")

            features = vision(pixel_values.to(self.device))
            if isinstance(features, tuple):
                features = features[0]
            return features.cpu()

    def run_full_inference(
        self,
        pixel_values: torch.Tensor,
        text_tokens: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: int = 0,
        capture_layers: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run full inference and return all outputs + captured activations.

        Returns dict with:
            'actions': final predicted actions
            'backbone_features': VLM backbone output
            + any captured layer activations
        """
        if capture_layers:
            self._register_hooks(capture_layers)

        results = {}

        with torch.no_grad():
            t0 = time.time()

            # Run the model
            try:
                output = self.model(
                    pixel_values=pixel_values.to(self.device),
                    input_ids=text_tokens.to(self.device),
                    state=state.to(self.device),
                    embodiment_id=embodiment_id,
                )
                if isinstance(output, dict):
                    results['actions'] = output.get('actions', output.get('predicted_actions')).cpu()
                elif isinstance(output, torch.Tensor):
                    results['actions'] = output.cpu()
            except Exception as e:
                logger.error(f"Reference inference failed: {e}")
                raise
            finally:
                elapsed = time.time() - t0
                logger.info(f"Reference inference: {elapsed*1000:.1f}ms")

        # Collect captured activations
        results.update(self._activations)
        self._remove_hooks()

        return results

    def get_dummy_inputs(
        self,
        batch_size: int = 1,
        image_size: int = 384,
        text_seq_len: int = 32,
        state_dim: int = 29,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate dummy inputs for testing."""
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        text_tokens = torch.randint(0, 1000, (batch_size, text_seq_len))
        state = torch.randn(batch_size, state_dim)
        return pixel_values, text_tokens, state


def compute_pcc(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between reference and test tensors."""
    ref_flat = ref.float().flatten()
    test_flat = test.float().flatten()

    if ref_flat.shape != test_flat.shape:
        min_len = min(len(ref_flat), len(test_flat))
        ref_flat = ref_flat[:min_len]
        test_flat = test_flat[:min_len]

    ref_mean = ref_flat.mean()
    test_mean = test_flat.mean()

    ref_centered = ref_flat - ref_mean
    test_centered = test_flat - test_mean

    cov = (ref_centered * test_centered).sum()
    ref_std = (ref_centered ** 2).sum().sqrt()
    test_std = (test_centered ** 2).sum().sqrt()

    if ref_std == 0 or test_std == 0:
        return 1.0 if torch.allclose(ref_flat, test_flat) else 0.0

    return (cov / (ref_std * test_std)).item()


def compute_allclose(
    ref: torch.Tensor,
    test: torch.Tensor,
    rtol: float = 0.01,
    atol: float = 0.01,
) -> Tuple[bool, float]:
    """Check if tensors are close and return max absolute difference."""
    ref_f = ref.float()
    test_f = test.float()
    max_diff = (ref_f - test_f).abs().max().item()
    is_close = torch.allclose(ref_f, test_f, rtol=rtol, atol=atol)
    return is_close, max_diff
