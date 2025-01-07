from typing import Optional, Dict, Any
from pathlib import Path
import os

from genmo.mochi_preview.pipelines import ModelFactory, load_to_cpu
from models.experimental.mochi.asymm_dit_joint import TtAsymmDiTJoint
from models.experimental.mochi.common import get_cache_path, get_mochi_dir


class TtDiTModelFactory(ModelFactory):
    """Factory for creating TensorTorch DiT models."""

    def __init__(
        self,
        mesh_device,
        *,
        model_path: str,
        model_dtype: str,
        lora_path: Optional[str] = None,
        attention_mode: Optional[str] = None,
    ):
        """Initialize the TT DiT model factory.

        Args:
            model_path: Path to model weights
            model_dtype: Data type for model (e.g. "bf16")
            lora_path: Optional path to LoRA weights
            attention_mode: Optional attention implementation mode
        """
        attention_mode = "sdpa"

        super().__init__(
            model_path=model_path,
            lora_path=lora_path,
            model_dtype=model_dtype,
            attention_mode=attention_mode,
        )

        # TODO: parametrize based on inputs to get_model
        self.weight_cache_path = get_cache_path(os.environ.get("FAKE_DEVICE"))
        self.weights_path = os.path.join(get_mochi_dir(), "dit.safetensors")
        self.mesh_device = mesh_device

    def get_model(
        self,
        *,
        local_rank: int,
        device_id: Any,
        world_size: int,
        model_kwargs: Optional[Dict[str, Any]] = None,
        strict_load: bool = True,
        load_checkpoint: bool = True,
        fast_init: bool = True,
    ) -> TtAsymmDiTJoint:
        """Create and initialize a TT DiT model.

        Args:
            local_rank: Local process rank
            device_id: Device ID (ignored for TT implementation)
            world_size: Total number of processes
            model_kwargs: Optional additional model arguments
            strict_load: Whether to strictly enforce state dict loading
            load_checkpoint: Whether to load weights from checkpoint

        Returns:
            Initialized TT DiT model
        """
        assert load_checkpoint, "Checkpoint loading is required for TT DiT"
        if not model_kwargs:
            model_kwargs = {}

        # Load state dict if needed
        state_dict = {}

        print(f"Loading weights from {self.weights_path}")
        state_dict = load_to_cpu(self.weights_path)

        # Create model with standard arguments
        model = TtAsymmDiTJoint(
            mesh_device=self.mesh_device,
            state_dict=state_dict,
            weight_cache_path=self.weight_cache_path,
            depth=48,
            patch_size=2,
            num_heads=24,
            hidden_size_x=3072,
            hidden_size_y=1536,
            mlp_ratio_x=4.0,
            mlp_ratio_y=4.0,
            in_channels=12,
            qk_norm=True,
            qkv_bias=False,
            out_bias=True,
            patch_embed_bias=True,
            timestep_mlp_bias=True,
            timestep_scale=1000.0,
            t5_feat_dim=4096,
            t5_token_length=256,
            rope_theta=10000.0,
            attention_mode=self.kwargs["attention_mode"],
            **model_kwargs,
        )

        return model
