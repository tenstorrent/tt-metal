"""TTNN implementation of the Inworld TTS codec encoder.

Pipeline: AcousticEncoder + Wav2Vec2-BERT -> SemanticEncoder -> Fusion -> FSQ quantize.

TTNN accelerated:
- SemanticEncoder: Conv1d + ReLU + residual add on device
- AcousticEncoder: Conv1d on device where possible, SnakeBeta on host
- fc_prior: Linear(2048, 2048) on device
- Wav2Vec2-BERT: FFN/Linear/LayerNorm on device (via TtWav2Vec2Bert)

CPU boundaries:
- SnakeBeta activation (custom: x + 1/beta * sin^2(alpha*x))
- Anti-aliased resampling (FIR filters)
- FSQ quantize (codebook + rounding)
- Feature extraction (AutoFeatureExtractor)
"""

from typing import Dict, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.reference.functional import activation1d_forward, weight_norm_compute
from models.demos.inworld_tts.tt.model_config import ENCODER_CHANNELS, ENCODER_STRIDES, get_compute_kernel_config_hifi4
from models.demos.inworld_tts.tt.wav2vec2_bert import TtWav2Vec2Bert

L1 = ttnn.L1_MEMORY_CONFIG


def ttnn_snake_beta(x, alpha, beta):
    """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x).

    alpha, beta: [C] learnable parameters, broadcast over [B, C, T].
    """
    alpha = alpha.reshape([1, 1, 1, -1])
    beta = beta.reshape([1, 1, 1, -1])
    return x + (1.0 / beta) * ttnn.pow(ttnn.sin(alpha * x), 2)


class TtActivation1dTTNN(LightweightModule):
    """Anti-aliased SnakeBeta activation using TTNN operations.

    Pipeline: Upsample 2x (host zero-insertion + device FIR) -> SnakeBeta (device)
              -> Downsample 2x (device FIR conv with stride=2)

    The zero-insertion for upsampling stays on host (trivial tensor creation),
    but FIR filtering, SnakeBeta, and downsampling all run on device.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        up_filter: torch.Tensor,
        down_filter: torch.Tensor,
        device,
    ):
        super().__init__()
        self.device = device
        C = alpha.shape[0]
        self.C = C

        # SnakeBeta params on device [1, 1, 1, C] in L1 for broadcasting over [1, 1, T, C]
        # L1 keeps element-wise ops (sin, pow, mul, add) in L1 instead of falling to DRAM
        self.alpha_tt = ttnn.from_torch(
            alpha.reshape(1, 1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.beta_tt = ttnn.from_torch(
            beta.reshape(1, 1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # FIR filter kernels as ttnn host tensors for conv1d (depthwise: [C, 1, K])
        up_k = up_filter.squeeze(0)  # [1, K]
        down_k = down_filter.squeeze(0)  # [1, K]
        self.K_up = up_k.shape[-1]
        self.K_down = down_k.shape[-1]

        # Expand to depthwise [C, 1, K] and scale upsample by 2.0
        up_expanded = (up_k.expand(C, -1, -1) * 2.0).to(torch.bfloat16).to(torch.float32)
        down_expanded = down_k.expand(C, -1, -1).to(torch.bfloat16).to(torch.float32)

        self.up_weight = ttnn.from_torch(up_expanded, dtype=ttnn.float32)
        self.down_weight = ttnn.from_torch(down_expanded, dtype=ttnn.float32)

        # Padding for FIR filters
        self.up_pad_left = self.K_up // 2
        self.up_pad_right = self.K_up // 2 - 1 if self.K_up % 2 == 0 else self.K_up // 2
        self.down_pad_left = self.K_down // 2
        self.down_pad_right = self.K_down // 2 - 1 if self.K_down % 2 == 0 else self.K_down // 2

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.float32,
            deallocate_activation=True,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )

        # Cached device weights (populated on first forward, used for trace)
        self._up_cached_w = None
        self._up_cached_b = None
        self._down_cached_w = None
        self._down_cached_b = None

        # Pre-allocated zeros for trace compatibility (populated by prepare_for_trace)
        self._zeros_cache = {}  # (T, C) -> ttnn zeros tensor
        self._trace_mode = False

    def prepare_for_trace(self, T):
        """Prepare for trace capture: pre-allocate zeros and lock conv weights.

        Must be called after a warmup forward (to populate conv weight caches).
        """
        C = self.C
        # Pre-allocate the zeros tensor for the upsample zero-insertion
        self._zeros_cache[(T, C)] = ttnn.zeros(
            [1, T, 1, C], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        self._trace_mode = True

    def forward_ttnn(self, x_tt, C, T):
        """Anti-aliased SnakeBeta on TTNN tensor. All compute on device.

        Stays in ROW_MAJOR throughout. Trace-compatible when prepare_for_trace()
        has been called (uses pre-allocated zeros, cached conv weights).

        Args:
            x_tt: [1, 1, T, C] TTNN ROW_MAJOR on device
            C: number of channels
            T: time dimension length
        Returns:
            x_tt: [1, 1, T, C] TTNN ROW_MAJOR on device
        """
        B = 1
        # === UPSAMPLE 2x: zero-insertion + FIR filter ===
        x_reshaped = ttnn.reshape(x_tt, [1, T, 1, C])
        # Use pre-allocated zeros in trace mode, fresh allocation otherwise
        if self._trace_mode and (T, C) in self._zeros_cache:
            z = self._zeros_cache[(T, C)]
        else:
            z = ttnn.zeros([1, T, 1, C], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        x_interleaved = ttnn.concat([x_reshaped, z], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)  # [1, T, 2, C]
        x_up = ttnn.reshape(x_interleaved, [1, 1, T * 2, C])

        # FIR upsample filter (depthwise conv, groups=C, stride=1)
        # In trace mode: weights already cached, no host mutation
        uw = self._up_cached_w or self.up_weight
        ub = self._up_cached_b
        if self._trace_mode:
            x_tt, _, _ = ttnn.conv1d(
                input_tensor=x_up,
                weight_tensor=uw,
                in_channels=C,
                out_channels=C,
                device=self.device,
                bias_tensor=ub,
                kernel_size=self.K_up,
                stride=1,
                padding=(self.up_pad_left, self.up_pad_right),
                batch_size=B,
                input_length=T * 2,
                dtype=ttnn.bfloat16,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                groups=C,
                return_output_dim=True,
                return_weights_and_bias=False,
            )
        else:
            x_tt, _, [self._up_cached_w, self._up_cached_b] = ttnn.conv1d(
                input_tensor=x_up,
                weight_tensor=uw,
                in_channels=C,
                out_channels=C,
                device=self.device,
                bias_tensor=ub,
                kernel_size=self.K_up,
                stride=1,
                padding=(self.up_pad_left, self.up_pad_right),
                batch_size=B,
                input_length=T * 2,
                dtype=ttnn.bfloat16,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                groups=C,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
        x_tt = ttnn.sharded_to_interleaved(x_tt, ttnn.L1_MEMORY_CONFIG)
        if x_tt.layout != ttnn.ROW_MAJOR_LAYOUT:
            x_tt = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)

        # === SNAKEBETA on device (ROW_MAJOR, L1) ===
        x_tt = x_tt + (1.0 / self.beta_tt) * ttnn.pow(ttnn.sin(self.alpha_tt * x_tt), 2)

        # === DOWNSAMPLE: FIR lowpass + stride-2 decimation ===
        T_act = x_tt.shape[2]
        dw = self._down_cached_w or self.down_weight
        db = self._down_cached_b
        if self._trace_mode:
            x_tt, _, _ = ttnn.conv1d(
                input_tensor=x_tt,
                weight_tensor=dw,
                in_channels=C,
                out_channels=C,
                device=self.device,
                bias_tensor=db,
                kernel_size=self.K_down,
                stride=2,
                padding=(self.down_pad_left, self.down_pad_right),
                batch_size=B,
                input_length=T_act,
                dtype=ttnn.bfloat16,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                groups=C,
                return_output_dim=True,
                return_weights_and_bias=False,
            )
        else:
            x_tt, _, [self._down_cached_w, self._down_cached_b] = ttnn.conv1d(
                input_tensor=x_tt,
                weight_tensor=dw,
                in_channels=C,
                out_channels=C,
                device=self.device,
                bias_tensor=db,
                kernel_size=self.K_down,
                stride=2,
                padding=(self.down_pad_left, self.down_pad_right),
                batch_size=B,
                input_length=T_act,
                dtype=ttnn.bfloat16,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                groups=C,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
        x_tt = ttnn.sharded_to_interleaved(x_tt, ttnn.L1_MEMORY_CONFIG)
        if x_tt.layout != ttnn.ROW_MAJOR_LAYOUT:
            x_tt = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)

        return x_tt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply anti-aliased SnakeBeta activation. All compute on device.

        Args:
            x: [B, C, T] torch tensor
        Returns:
            [B, C, T] torch tensor
        """
        B, C, T = x.shape

        # === Convert input to device: [B, C, T] -> [1, 1, T, C] NHWC ===
        x_nhwc = x.permute(0, 2, 1).unsqueeze(0).to(torch.bfloat16)  # [1, 1, T, C]
        x_tt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

        x_tt = self.forward_ttnn(x_tt, C, T)

        # Back to torch [B, C, T]
        out = ttnn.to_torch(x_tt).float()  # [1, 1, T, C]
        return out.squeeze(0).permute(0, 2, 1)  # [B, C, T]


class TtActivation1d(LightweightModule):
    """Anti-aliased SnakeBeta activation using TTNN ops where possible.

    Pipeline: Upsample 2x -> SnakeBeta -> Downsample 2x to avoid aliasing.
    SnakeBeta: x + (1/beta) * sin^2(alpha * x)

    Due to the custom anti-aliasing with FIR filters and complex upsampling/downsampling,
    this currently falls back to CPU but accepts/returns tensors compatible with TTNN pipeline.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        up_filter: torch.Tensor,
        down_filter: torch.Tensor,
        device=None,
    ):
        """Initialize anti-aliased SnakeBeta activation.

        Args:
            alpha: [C] SnakeBeta alpha parameter
            beta: [C] SnakeBeta beta parameter
            up_filter: [1, 1, K] FIR upsampling filter
            down_filter: [1, 1, K] FIR lowpass/downsampling filter
            device: Optional TTNN device (for future acceleration)
        """
        super().__init__()
        # Store as torch tensors for CPU computation
        self.alpha = alpha
        self.beta = beta
        self.up_filter = up_filter
        self.down_filter = down_filter
        self.device = device

    def _snake_beta_ttnn(self, x: torch.Tensor) -> torch.Tensor:
        """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x).

        Args:
            x: [B, C, T] input tensor
        Returns:
            [B, C, T] activated tensor
        """
        # Reshape alpha and beta for broadcasting: [C] -> [1, C, 1]
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]

        # SnakeBeta: x + (1/beta) * sin^2(alpha * x)
        scaled = alpha * x
        sin_val = torch.sin(scaled)
        sin_sq = sin_val * sin_val
        result = x + (1.0 / beta) * sin_sq

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply anti-aliased SnakeBeta activation.

        Handles both torch.Tensor and ttnn.Tensor inputs.
        For ttnn inputs, converts to torch, processes, and converts back.

        Args:
            x: [B, C, T] input tensor (channels-first) - torch.Tensor or ttnn.Tensor
        Returns:
            [B, C, T] activated tensor in same format as input
        """
        # Check if input is TTNN tensor
        is_ttnn_input = hasattr(x, "device") and hasattr(x.device, "arch")

        # Convert TTNN to torch if needed
        if is_ttnn_input:
            x_torch = ttnn.to_torch(x)
        else:
            x_torch = x

        # Run CPU-based anti-aliased activation using reference implementation
        # (FIR filtering with zero insertion/decimation is complex to implement in TTNN)
        result = activation1d_forward(
            x_torch,
            self.alpha,
            self.beta,
            self.up_filter,
            self.down_filter,
        )

        # Convert back to TTNN if input was TTNN
        if is_ttnn_input and self.device is not None:
            result = ttnn.from_torch(
                result,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        return result


class TtConv1d(LightweightModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        kernel_size: int,
        padding,
        device,
        stride: int = 1,
    ):
        super().__init__()
        self.weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
        # Reshape bias to [1, 1, 1, out_channels] for conv1d (implemented as conv2d with height=1)
        if bias is not None:
            bias_reshaped = bias.reshape(1, 1, 1, -1)
            self.bias = ttnn.from_torch(bias_reshaped, dtype=ttnn.bfloat16)
        else:
            self.bias = None
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.float32,
            deallocate_activation=True,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.padding = padding
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )

    def prepare_for_trace(self):
        """Lock conv weights for trace. Must be called after warmup."""
        self._trace_mode = True

    def forward_ttnn(self, x_tt, T):
        """Conv1d on TTNN tensor. Input/output: [1, 1, T, C] ROW_MAJOR on device.

        Trace-compatible when prepare_for_trace() has been called.

        Args:
            x_tt: [1, 1, T, C] TTNN ROW_MAJOR on device
            T: input time dimension length
        Returns:
            (output_tt, out_length): output [1, 1, T_out, C_out] ROW_MAJOR on device, output time length
        """
        if getattr(self, "_trace_mode", False):
            output_tensor, out_length, _ = ttnn.conv1d(
                input_tensor=x_tt,
                weight_tensor=self.weight,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                device=self.device,
                bias_tensor=self.bias,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                batch_size=1,
                input_length=T,
                dtype=ttnn.bfloat16,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                groups=1,
                return_output_dim=True,
                return_weights_and_bias=False,
            )
        else:
            output_tensor, out_length, (self.weight, self.bias) = ttnn.conv1d(
                input_tensor=x_tt,
                weight_tensor=self.weight,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                device=self.device,
                bias_tensor=self.bias,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                batch_size=1,
                input_length=T,
                dtype=ttnn.bfloat16,
                conv_config=self.conv_config,
                compute_config=self.compute_config,
                groups=1,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if output_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        return output_tensor, out_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Conv1d on input tensor.

        Args:
            x: [B, C, T] torch tensor (channels-first format)
        Returns:
            [B, C_out, T_out] torch tensor
        """
        B, C, T = x.shape

        # Convert [B, C, T] -> [1, 1, T, C] NHWC ROW_MAJOR for conv1d
        x_nhwc = x.permute(0, 2, 1).unsqueeze(0).to(torch.bfloat16)
        x_tt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

        output_tt, out_length = self.forward_ttnn(x_tt, T)

        # Convert back to torch tensor [1, 1, T_out, C_out] -> [B, C_out, T_out]
        output_torch = ttnn.to_torch(output_tt).float()
        return output_torch.squeeze(0).permute(0, 2, 1)


class TtAcousticEncoder(LightweightModule):
    """AcousticEncoder -- Conv1d chains with SnakeBeta activation.

    Conv1d ops use precomputed weight-normed weights.
    All ops run on device via forward_ttnn() methods. Residual adds use ttnn.add.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor], device):
        super().__init__()
        self.device = device
        self.channels = ENCODER_CHANNELS
        self.strides = ENCODER_STRIDES

        # Precompute weight-normed initial conv
        self.initial_weight = weight_norm_compute(
            state_dict["conv_blocks.0.weight_g"],
            state_dict["conv_blocks.0.weight_v"],
        )
        self.initial_bias = state_dict["conv_blocks.0.bias"]
        self.initial_conv1d = TtConv1d(
            in_channels=1,
            out_channels=48,
            weight=self.initial_weight,
            bias=self.initial_bias,
            kernel_size=7,
            padding=3,  # k=7 Conv1d uses pad=3
            device=device,
        )

        # Precompute all encoder block weights and create TTNN modules
        self.block_weights = []
        self.block_activations = []  # list of lists: [block_idx][act_idx] -> TtActivation1d
        self.block_res_convs = []  # list of lists: [block_idx][(conv1, conv2), ...] per res unit
        self.block_final_act = []  # [block_idx] -> TtActivation1d (before downsample)
        self.block_downsample = []  # [block_idx] -> TtConv1d

        for block_idx in range(5):
            prefix = f"conv_blocks.{block_idx + 1}."
            from models.demos.inworld_tts.reference.functional import _extract_encoder_block_weights

            bw = _extract_encoder_block_weights(state_dict, prefix, self.channels[block_idx])
            self.block_weights.append(bw)

            c_in = self.channels[block_idx]
            c_out = self.channels[block_idx + 1]
            stride = self.strides[block_idx]

            # Create residual unit modules: 3 res units per block
            block_res_acts = []  # (act1, act2) per res unit
            block_res_cv = []  # (conv1, conv2) per res unit
            for res_idx in range(3):
                p = f"res_{res_idx}_"
                act1 = TtActivation1dTTNN(
                    alpha=bw[p + "act1_alpha"],
                    beta=bw[p + "act1_beta"],
                    up_filter=bw[p + "act1_up_filter"],
                    down_filter=bw[p + "act1_down_filter"],
                    device=device,
                )
                conv1 = TtConv1d(
                    in_channels=c_in,
                    out_channels=c_in,
                    weight=bw[p + "conv1_weight"],
                    bias=bw[p + "conv1_bias"],
                    kernel_size=7,
                    padding=3,
                    device=device,
                )
                act2 = TtActivation1dTTNN(
                    alpha=bw[p + "act2_alpha"],
                    beta=bw[p + "act2_beta"],
                    up_filter=bw[p + "act2_up_filter"],
                    down_filter=bw[p + "act2_down_filter"],
                    device=device,
                )
                conv2 = TtConv1d(
                    in_channels=c_in,
                    out_channels=c_in,
                    weight=bw[p + "conv2_weight"],
                    bias=bw[p + "conv2_bias"],
                    kernel_size=1,
                    padding=0,
                    device=device,
                )
                block_res_acts.append((act1, act2))
                block_res_cv.append((conv1, conv2))

            self.block_activations.append(block_res_acts)
            self.block_res_convs.append(block_res_cv)

            # Final activation before downsample
            final_act = TtActivation1dTTNN(
                alpha=bw["act_alpha"],
                beta=bw["act_beta"],
                up_filter=bw["act_up_filter"],
                down_filter=bw["act_down_filter"],
                device=device,
            )
            self.block_final_act.append(final_act)

            # Downsample conv: kernel_size = stride * 2
            kernel_size = stride * 2
            pad_total = kernel_size - stride
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            ds_conv = TtConv1d(
                in_channels=c_in,
                out_channels=c_out,
                weight=bw["downsample_weight"],
                bias=bw["downsample_bias"],
                kernel_size=kernel_size,
                padding=(pad_left, pad_right),
                device=device,
                stride=stride,
            )
            self.block_downsample.append(ds_conv)

        # Precompute final block weights
        final_prefix = "conv_final_block."
        self.final_activation = TtActivation1dTTNN(
            alpha=state_dict[final_prefix + "0.act.alpha"],
            beta=state_dict[final_prefix + "0.act.beta"],
            up_filter=state_dict[final_prefix + "0.upsample.filter"],
            down_filter=state_dict[final_prefix + "0.downsample.lowpass.filter"],
            device=device,
        )
        final_weight = weight_norm_compute(
            state_dict[final_prefix + "1.weight_g"],
            state_dict[final_prefix + "1.weight_v"],
        )
        final_bias = state_dict[final_prefix + "1.bias"]
        self.final_conv1d = TtConv1d(
            in_channels=1536,
            out_channels=1024,
            weight=final_weight,
            bias=final_bias,
            kernel_size=3,
            padding=1,
            device=device,
        )

    def prepare_for_trace(self, T):
        """Prepare all sub-components for trace capture.

        Must be called after a warmup forward to populate all conv weight caches.
        Pre-allocates zeros tensors and locks conv weights.

        Args:
            T: time dimension after initial conv (determines all downstream T values)
        """
        C = 48
        t = T
        for block_idx in range(5):
            # Residual unit activations + convs
            for res_idx in range(3):
                act1, act2 = self.block_activations[block_idx][res_idx]
                conv1, conv2 = self.block_res_convs[block_idx][res_idx]
                act1.prepare_for_trace(t)
                act2.prepare_for_trace(t)
                conv1.prepare_for_trace()
                conv2.prepare_for_trace()
            # Final activation + downsample
            self.block_final_act[block_idx].prepare_for_trace(t)
            self.block_downsample[block_idx].prepare_for_trace()
            # T changes after downsample
            stride = self.strides[block_idx]
            kernel_size = stride * 2
            pad_total = kernel_size - stride
            # Approximate T after conv with stride (matching conv1d output formula)
            t = (t + pad_total - kernel_size) // stride + 1
            C = self.channels[block_idx + 1]
        # Final block
        self.final_activation.prepare_for_trace(t)
        self.final_conv1d.prepare_for_trace()

    def forward_ttnn(self, x_tt: ttnn.Tensor, T: int) -> ttnn.Tensor:
        """Device-only forward: 5 encoder blocks + final activation + final conv.

        Accepts and returns ttnn tensors on device, avoiding PCIe round-trips.

        Args:
            x_tt: [1, 1, T, 48] ttnn tensor on device (ROW_MAJOR_LAYOUT)
            T: time dimension after initial conv
        Returns:
            [1, 1, T_final, 1024] ttnn tensor on device (ROW_MAJOR)
        """
        C = 48
        for block_idx in range(5):
            for res_idx in range(3):
                res = x_tt
                act1, act2 = self.block_activations[block_idx][res_idx]
                conv1, conv2 = self.block_res_convs[block_idx][res_idx]
                x_tt = act1.forward_ttnn(x_tt, C, T)
                x_tt, _ = conv1.forward_ttnn(x_tt, T)
                x_tt = act2.forward_ttnn(x_tt, C, T)
                x_tt, _ = conv2.forward_ttnn(x_tt, T)
                # Residual add on device (ROW_MAJOR, no layout conversion)
                x_tt = ttnn.add(res, x_tt)
            # Final activation + downsample
            x_tt = self.block_final_act[block_idx].forward_ttnn(x_tt, C, T)
            C_out = self.channels[block_idx + 1]
            x_tt, T = self.block_downsample[block_idx].forward_ttnn(x_tt, T)
            C = C_out

        # Final block: SnakeBeta(1536) + Conv1d(1536, 1024, k=3)
        x_tt = self.final_activation.forward_ttnn(x_tt, C, T)
        x_tt, _ = self.final_conv1d.forward_ttnn(x_tt, T)

        return x_tt

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass with all ops on device. Residual adds use ttnn.add.

        Wrapper around forward_ttnn that handles torch<->ttnn conversion.

        Args:
            waveform: [B, 1, samples] input audio
        Returns:
            [B, 1024, T] acoustic features
        """
        # Initial conv: Conv1d(1, 48, k=7, pad=3) -- still torch interface for channels=1
        x = self.initial_conv1d(waveform)  # [B, 48, T] torch

        # Convert to TTNN NHWC: [B, C, T] -> [1, 1, T, C]
        x_nhwc = x.permute(0, 2, 1).unsqueeze(0).to(torch.bfloat16)
        x_tt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

        T = x.shape[2]
        x_tt = self.forward_ttnn(x_tt, T)

        # Convert back: [1, 1, T, 1024] -> [B, 1024, T]
        out = ttnn.to_torch(x_tt).float()
        return out.squeeze(0).permute(0, 2, 1)


class TtSemanticEncoder(LightweightModule):
    """SemanticEncoder -- TTNN Conv1d + ReLU + residual add on device.

    Architecture: initial_conv -> N residual blocks (ReLU + Conv + ReLU + Conv) -> final_conv
    All Conv1d(1024, 1024, k=3, pad=1). Channels are tile-aligned (1024).
    """

    def __init__(self, device, state_dict: Dict[str, torch.Tensor], prefix: str = "SemanticEncoder_module."):
        super().__init__()
        self.device = device

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.float32,
            deallocate_activation=True,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        )

        def _load_conv(w_key, b_key):
            w = state_dict[w_key].to(torch.bfloat16).to(torch.float32)
            b = state_dict.get(b_key)
            w_tt = ttnn.from_torch(w, dtype=ttnn.float32)
            b_tt = None
            if b is not None:
                b = b.to(torch.bfloat16).to(torch.float32)
                b_tt = ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.float32)
            return w_tt, b_tt

        self.initial_w, self.initial_b = _load_conv(prefix + "initial_conv.weight", prefix + "initial_conv.bias")
        self.final_w, self.final_b = _load_conv(prefix + "final_conv.weight", prefix + "final_conv.bias")

        self.res_blocks = []
        block_idx = 1
        while prefix + f"residual_blocks.{block_idx}.weight" in state_dict:
            c1w, c1b = _load_conv(
                prefix + f"residual_blocks.{block_idx}.weight",
                prefix + f"residual_blocks.{block_idx}.bias",
            )
            c2w, c2b = _load_conv(
                prefix + f"residual_blocks.{block_idx + 2}.weight",
                prefix + f"residual_blocks.{block_idx + 2}.bias",
            )
            self.res_blocks.append((c1w, c1b, c2w, c2b))
            block_idx += 4

        # Device weight caches (populated on first forward)
        self._cached = {}

    def _conv1d(self, x, cache_key, w, b, T):
        """Conv1d(1024, 1024, k=3, pad=1) on device with weight caching."""
        cw = self._cached.get(cache_key + "_w", w)
        cb = self._cached.get(cache_key + "_b", b)
        out, out_len, [dw, db] = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=cw,
            in_channels=1024,
            out_channels=1024,
            device=self.device,
            bias_tensor=cb,
            kernel_size=3,
            stride=1,
            padding=1,
            batch_size=1,
            input_length=T,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self._cached[cache_key + "_w"] = dw
        self._cached[cache_key + "_b"] = db
        return ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)

    def forward_ttnn(self, x_tt: ttnn.Tensor, T: int) -> ttnn.Tensor:
        """Device-only forward: initial conv -> res blocks -> final conv.

        Accepts and returns ttnn tensors on device, avoiding PCIe round-trips.

        Args:
            x_tt: [1, 1, T, 1024] ttnn tensor on device (ROW_MAJOR_LAYOUT)
            T: time dimension length
        Returns:
            [1, 1, T, 1024] ttnn tensor on device (ROW_MAJOR, in DRAM)
        """
        # Initial conv
        x = self._conv1d(x_tt, "init", self.initial_w, self.initial_b, T)

        # Residual blocks: ReLU -> Conv -> ReLU -> Conv + skip
        for i, (c1w, c1b, c2w, c2b) in enumerate(self.res_blocks):
            res = x
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.relu(x)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = self._conv1d(x, f"res{i}_c1", c1w, c1b, T)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.relu(x)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = self._conv1d(x, f"res{i}_c2", c2w, c2b, T)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            res = ttnn.to_layout(res, ttnn.TILE_LAYOUT)
            x = ttnn.add(res, x)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Final conv
        x = self._conv1d(x, "final", self.final_w, self.final_b, T)
        return x

    def forward(self, semantic_features: torch.Tensor) -> torch.Tensor:
        """Forward pass on device.

        Wrapper around forward_ttnn that handles torch<->ttnn conversion.

        Args:
            semantic_features: [B, 1024, T] from Wav2Vec2-BERT (torch tensor)
        Returns:
            [B, 1024, T] (torch tensor)
        """
        B, C, T = semantic_features.shape

        # Convert [B, C, T] -> [1, 1, T, C] NHWC ROW_MAJOR for TTNN conv1d
        x_nhwc = semantic_features.permute(0, 2, 1).unsqueeze(0).to(torch.bfloat16)
        x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

        # Run device pipeline
        x = self.forward_ttnn(x, T)

        # Convert back [1, 1, T, C] -> [B, C, T]
        out = ttnn.to_torch(x).float()
        return out.squeeze(0).permute(0, 2, 1)


class TtCodecEncoder(LightweightModule):
    """Full codec encoder: Wav2Vec2-BERT + AcousticEncoder + SemanticEncoder + Fusion + FSQ quantize."""

    def __init__(
        self,
        device,
        state_dict: Dict[str, torch.Tensor],
        quantizer=None,
        dtype=ttnn.bfloat16,
        acoustic_prefix: str = "CodecEnc.",
        semantic_prefix: str = "SemanticEncoder_module.",
        w2v_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.device = device
        self.quantizer = quantizer

        # Acoustic encoder (CPU -- SnakeBeta + non-standard channel sizes)
        acoustic_sd = {k[len(acoustic_prefix) :]: v for k, v in state_dict.items() if k.startswith(acoustic_prefix)}
        self.acoustic_encoder = TtAcousticEncoder(acoustic_sd, device)

        # Wav2Vec2-BERT (TTNN -- FFN/Linear/LayerNorm on device)
        self.w2v_bert = TtWav2Vec2Bert(device, state_dict=w2v_state_dict, dtype=dtype)

        # Semantic encoder (TTNN -- Conv1d + ReLU + Add on device)
        self.semantic_encoder = TtSemanticEncoder(device, state_dict, prefix=semantic_prefix)

        # Feature extractor (lazy-loaded)
        self._feature_extractor = None

        # fc_prior: Linear(2048, 2048) -- TTNN
        fc_w = state_dict["fc_prior.weight"]
        fc_b = state_dict["fc_prior.bias"]
        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.fc_prior_weight = ttnn.from_torch(
            fc_w.T.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.fc_prior_bias = ttnn.from_torch(
            fc_b.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def _get_feature_extractor(self):
        """Lazy-load AutoFeatureExtractor for mel filterbank preprocessing."""
        if self._feature_extractor is None:
            from transformers import AutoFeatureExtractor

            self._feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        return self._feature_extractor

    def _extract_semantic_features(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Extract semantic features from waveform using Wav2Vec2-BERT.

        Args:
            waveform: [B, 1, samples] raw audio
        Returns:
            [B, 1024, T] semantic features (channels-first for SemanticEncoder)
        """
        fe = self._get_feature_extractor()
        audio_np = waveform.squeeze(1).numpy()
        inputs = fe(audio_np, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_features = inputs["input_features"].to(waveform.device)

        # Run Wav2Vec2-BERT (returns [B, T, 1024])
        hidden = self.w2v_bert(input_features)

        # Transpose to channels-first [B, 1024, T] for SemanticEncoder
        return hidden.transpose(1, 2)

    def forward_on_device(
        self,
        waveform: torch.Tensor,
        mel_features: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Full codec encoder: W2V → Semantic → Acoustic → Fusion → FSQ.

        Complete e2e pipeline where each block feeds into the next on device.
        Only CPU boundaries: initial_conv (channels=1), mel extraction, FSQ quantize.

        Data flow:
            mel → from_torch → W2V.forward_ttnn → Semantic.forward_ttnn ─┐
            wav → initial_conv → from_torch → Acoustic.forward_ttnn ─────┤
                                    ttnn.concat → fc_prior linear → to_torch → FSQ

        Args:
            waveform: [B, 1, samples] raw audio waveform
            mel_features: optional [B, T, 160] mel features. If None, extracted via AutoFeatureExtractor.
            sample_rate: audio sample rate (default 16000)
        Returns:
            [B, 1, T] integer VQ codes
        """
        if self.quantizer is None:
            raise ValueError("Quantizer required for FSQ quantization")

        # === Semantic path: W2V → SemanticEncoder (all on device) ===

        # Extract mel features if not provided
        if mel_features is None:
            fe = self._get_feature_extractor()
            audio_np = waveform.squeeze(1).numpy()
            inputs = fe(audio_np, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            mel_features = inputs["input_features"]

        # W2V on device: [B, T, 160] → [1, 1, T, 1024]
        mel_tt = ttnn.from_torch(
            mel_features.to(torch.bfloat16).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        w2v_tt = self.w2v_bert.forward_ttnn(mel_tt)  # [1, 1, T, 1024] TILE on device

        # Semantic encoder on device: needs ROW_MAJOR [1, 1, T, 1024]
        T_sem = w2v_tt.shape[2]
        semantic_tt = ttnn.to_layout(w2v_tt, ttnn.ROW_MAJOR_LAYOUT)
        semantic_tt = self.semantic_encoder.forward_ttnn(semantic_tt, T_sem)
        # semantic_tt: [1, 1, T, 1024] ROW_MAJOR on device

        # === Acoustic path: initial_conv (CPU) → encoder blocks (device) ===

        x_initial = self.acoustic_encoder.initial_conv1d(waveform)  # [B, 48, T_init] torch
        T_init = x_initial.shape[2]
        x_nhwc = x_initial.permute(0, 2, 1).unsqueeze(0).to(torch.bfloat16)
        acoustic_tt = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        acoustic_tt = self.acoustic_encoder.forward_ttnn(acoustic_tt, T_init)
        # acoustic_tt: [1, 1, T, 1024] ROW_MAJOR on device

        # === Fusion: concat → fc_prior → FSQ ===

        acoustic_tt = ttnn.to_layout(acoustic_tt, ttnn.TILE_LAYOUT)
        semantic_tt = ttnn.to_layout(semantic_tt, ttnn.TILE_LAYOUT)
        fused_tt = ttnn.concat(
            [acoustic_tt, semantic_tt], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG
        )  # [1, 1, T, 2048]

        projected_tt = ttnn.linear(
            fused_tt,
            self.fc_prior_weight,
            bias=self.fc_prior_bias,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        # FSQ quantize (CPU boundary)
        projected_torch = ttnn.to_torch(projected_tt).float()
        if projected_torch.dim() == 4:
            projected_torch = projected_torch.squeeze(0)

        _, indices = self.quantizer(projected_torch)
        vq_codes = indices.squeeze(-1).unsqueeze(1)  # [B, 1, T]

        return vq_codes

    def forward(
        self,
        waveform: torch.Tensor,
        semantic_features: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Full codec encoder forward.

        Args:
            waveform: [B, 1, samples] raw audio waveform
            semantic_features: optional [B, 1024, T] pre-computed. If None, extracted via Wav2Vec2-BERT.
            sample_rate: audio sample rate (default 16000)
        Returns:
            [B, 1, T] integer VQ codes
        """
        if self.quantizer is None:
            raise ValueError("Quantizer required for FSQ quantization")

        # Step 1: Acoustic encoder (CPU)
        acoustic_out = self.acoustic_encoder(waveform)  # [B, 1024, T]

        # Step 2: Extract or use provided semantic features
        if semantic_features is None:
            semantic_features = self._extract_semantic_features(waveform, sample_rate)

        # Step 3: Semantic encoder (TTNN Conv1d + ReLU)
        semantic_out = self.semantic_encoder(semantic_features)  # [B, 1024, T]

        # Step 4: Fuse acoustic + semantic
        fused = torch.cat([acoustic_out, semantic_out], dim=1)  # [B, 2048, T]
        fused = fused.transpose(1, 2)  # [B, T, 2048]

        # Step 5: fc_prior projection (TTNN)
        fused_ttnn = ttnn.from_torch(
            fused.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        projected = ttnn.linear(
            fused_ttnn,
            self.fc_prior_weight,
            bias=self.fc_prior_bias,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Back to CPU for FSQ quantization (needs float32)
        projected_torch = ttnn.to_torch(projected).float()
        if projected_torch.dim() == 4:
            projected_torch = projected_torch.squeeze(0)

        # Step 6: FSQ quantize (CPU)
        _, indices = self.quantizer(projected_torch)
        vq_codes = indices.squeeze(-1).unsqueeze(1)  # [B, 1, T]

        return vq_codes

    def forward_acoustic_only(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run only the acoustic encoder."""
        return self.acoustic_encoder(waveform)

    def forward_semantic_only(self, semantic_features: torch.Tensor) -> torch.Tensor:
        """Run only the semantic encoder."""
        return self.semantic_encoder(semantic_features)

    def forward_w2v_only(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run only the Wav2Vec2-BERT model."""
        return self.w2v_bert(input_features)
