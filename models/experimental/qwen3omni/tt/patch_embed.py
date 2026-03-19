from typing import Optional
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

from models.experimental.tt_symbiote.core.module import TTNNModule


class TTNNQwen3OmniMoeVisionPatchEmbed(TTNNModule):
    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        embed_dim: int,
        mesh_device=None,
        init: bool = False,
        tp_mesh_axis: Optional[int] = None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mesh_device = mesh_device
        self.tp_mesh_axis = tp_mesh_axis

        assert not init, "Initialization not supported"
        if mesh_device is not None:
            self.to_device(mesh_device)

        # Compute kernel config (when device is available)
        self.compute_kernel_config = None
        if mesh_device is not None:
            self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

        # Weights set by load_state_dict via _prepare_torch_state; then preprocess/move to device
        self.proj_weight = None  # torch.Tensor (patch_volume, embed_dim) until preprocessed
        self.proj_bias = None  # torch.Tensor (1, embed_dim) until preprocessed
        self.tt_proj_weight = None  # ttnn.Tensor on device after move_weights_to_device
        self.tt_proj_bias = None

    # ------------------------------------------------------------
    # State loading
    # ------------------------------------------------------------
    def load_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True):
        """Load state; runs _prepare_torch_state then assigns proj_weight/proj_bias."""
        state = dict(state)
        self._prepare_torch_state(state)
        if "proj_weight" in state:
            self.proj_weight = state["proj_weight"]
        if "proj_bias" in state:
            self.proj_bias = state["proj_bias"]

    # ------------------------------------------------------------
    # Conv3D → Linear weight conversion
    # ------------------------------------------------------------
    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        conv_weight = state.pop("weight", None)

        if conv_weight is not None:
            out_c, in_c, kt, kh, kw = conv_weight.shape

            assert out_c == self.embed_dim
            assert in_c == self.in_channels
            assert kt == self.temporal_patch_size
            assert kh == self.patch_size
            assert kw == self.patch_size

            # (out_c, in_c, kt, kh, kw) -> (kt, kh, kw, in_c, out_c) -> (out_c, patch_volume)
            # preprocess_linear_weight does .T, so we pass (out_c, patch_volume) -> stored as (patch_volume, out_c) for ttnn.linear
            conv_weight = conv_weight.permute(2, 3, 4, 1, 0)
            conv_weight = conv_weight.reshape(kt * kh * kw * in_c, out_c)
            state["proj_weight"] = conv_weight.T.contiguous()  # (embed_dim, patch_volume) for preprocess_linear_weight

        conv_bias = state.pop("bias", None)
        if conv_bias is not None:
            state["proj_bias"] = conv_bias.reshape(1, -1)

    def preprocess_weights_impl(self):
        """Preprocess proj weight/bias for TTNN (linear layout)."""
        if self.proj_weight is None:
            return
        self._tt_proj_weight_host = preprocess_linear_weight(
            self.proj_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self._tt_proj_bias_host = None
        if self.proj_bias is not None:
            self._tt_proj_bias_host = preprocess_linear_bias(
                self.proj_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        if getattr(self, "_tt_proj_weight_host", None) is None:
            return
        self.tt_proj_weight = ttnn.to_device(self._tt_proj_weight_host, self.device)
        self.tt_proj_bias = (
            ttnn.to_device(self._tt_proj_bias_host, self.device) if self._tt_proj_bias_host is not None else None
        )

    # ------------------------------------------------------------
    # Forward (NO PATCHIFY HERE)
    # ------------------------------------------------------------
    def forward(
        self,
        latent_1BNI: ttnn.Tensor,  # (1, B, N, patch_volume)
    ) -> ttnn.Tensor:
        """
        Input:
            latent_1BNI: (1, B, N, K)

        Output:
            (1, B, N, embed_dim)
        """
        if self.tt_proj_weight is None:
            self.preprocess_weights()
            self.move_weights_to_device()

        latent_1BND = ttnn.linear(
            latent_1BNI,
            self.tt_proj_weight,
            bias=self.tt_proj_bias,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        return latent_1BND
