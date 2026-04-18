# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TT-NN SuperPoint port.

On-device: all convolutions + ReLU + MaxPool + softmax + simple_nms + L2-normalize.
Host: threshold / border-remove / top-k / grid_sample (variable-shape post-processing).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import ttnn

ENCODER_OUT_CHANNELS = (64, 64, 128, 128)
INTEREST_HIDDEN = 256
KEYPOINT_DIM = 65
DESCRIPTOR_HIDDEN = 256
DESCRIPTOR_DIM = 256


def _to_device_weight(weight: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(weight, dtype=dtype)


def _to_device_bias(bias: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    # Conv bias must be shape [1, 1, 1, out_channels].
    bias = bias.reshape(1, 1, 1, -1)
    return ttnn.from_torch(bias, dtype=dtype)


class TtConv2D:
    """Thin wrapper around ttnn.conv2d with weight/bias caching."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        device,
        activation: str | None = None,
        weights_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        num_slices: int = 1,
        act_block_h_override: int | None = None,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en: bool = False,
    ) -> None:
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (1, 1)
        self.padding = (padding, padding)
        self.dilation = (1, 1)
        self.groups = 1

        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32)
        self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.float32)

        act = None
        if activation == "relu":
            act = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            activation=act,
            shard_layout=shard_layout,
            deallocate_activation=False,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if act_block_h_override is not None:
            self.conv_config.act_block_h_override = act_block_h_override
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )
        self.activation_dtype = activation_dtype
        if num_slices > 1:
            self.slice_config = ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dDRAMSliceHeight,
                num_slices=num_slices,
            )
        else:
            self.slice_config = ttnn.Conv2dL1FullSliceConfig

    def __call__(self, x, input_height: int, input_width: int, batch_size: int = 1):
        # DRAM-sliced conv requires the input tensor to live in DRAM.
        if self.slice_config is not ttnn.Conv2dL1FullSliceConfig:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x, [out_h, out_w], [self.weight, self.bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            conv_config=self.conv_config,
            slice_config=self.slice_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
        )
        return x, out_h, out_w


class TtSuperPoint:
    """On-device SuperPoint inference (encoder + score/descriptor heads).

    Post-processing (NMS output extraction + grid_sample) is performed on host
    because the number of keypoints is data-dependent.
    """

    def __init__(self, torch_model, device, input_height: int = 480, input_width: int = 640):
        self.device = device
        self.input_height = input_height
        self.input_width = input_width
        self.nms_radius = torch_model.config.nms_radius
        self.keypoint_threshold = torch_model.config.keypoint_threshold
        self.max_keypoints = torch_model.config.max_keypoints
        self.border_removal_distance = torch_model.config.border_removal_distance

        encoder = torch_model.encoder
        enc_hidden = torch_model.config.encoder_hidden_sizes

        # Encoder: 4 conv blocks of (conv_a -> relu -> conv_b -> relu [-> pool])
        # Per-block DRAM slicing to keep L1 circular buffers within budget at 480×640.
        # Block resolutions (H x W) with batch=1, image 480x640:
        #   block 0: 480x640, block 1: 240x320, block 2: 120x160, block 3: 60x80
        slice_per_block = (8, 4, 2, 1)
        # Encoder precision: bfloat8_b weights (2× smaller vs bfloat16 → better
        # DRAM BW) + HiFi2 math + fp32 accumulator. The accumulator carries the
        # precision through 8 conv layers; bfloat8 weights keep PCC ≥ 99%.
        enc_kwargs = dict(
            weights_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
        )
        in_ch = 1
        self.enc_convs = []
        for block_idx, block in enumerate(encoder.conv_blocks):
            add_pooling = block.pool is not None
            ns = slice_per_block[block_idx]
            self.enc_convs.append(
                (
                    TtConv2D(
                        block.conv_a.weight,
                        block.conv_a.bias,
                        in_channels=in_ch,
                        out_channels=enc_hidden[block_idx],
                        kernel_size=3,
                        padding=1,
                        device=device,
                        activation="relu",
                        num_slices=ns,
                        **enc_kwargs,
                    ),
                    TtConv2D(
                        block.conv_b.weight,
                        block.conv_b.bias,
                        in_channels=enc_hidden[block_idx],
                        out_channels=enc_hidden[block_idx],
                        kernel_size=3,
                        padding=1,
                        device=device,
                        activation="relu",
                        num_slices=ns,
                        **enc_kwargs,
                    ),
                    add_pooling,
                )
            )
            in_ch = enc_hidden[block_idx]

        # Head configs: keep precision higher on heads (softmax + L2-norm are
        # numerically sensitive vs. the encoder ReLU chain).
        head_kwargs = dict(
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
        )

        # Score decoder
        kp = torch_model.keypoint_decoder
        self.conv_score_a = TtConv2D(
            kp.conv_score_a.weight,
            kp.conv_score_a.bias,
            in_channels=enc_hidden[-1],
            out_channels=torch_model.config.decoder_hidden_size,
            kernel_size=3,
            padding=1,
            device=device,
            activation="relu",
            **head_kwargs,
        )
        self.conv_score_b = TtConv2D(
            kp.conv_score_b.weight,
            kp.conv_score_b.bias,
            in_channels=torch_model.config.decoder_hidden_size,
            out_channels=torch_model.config.keypoint_decoder_dim,
            kernel_size=1,
            padding=0,
            device=device,
            activation=None,
            **head_kwargs,
        )

        # Descriptor decoder
        desc = torch_model.descriptor_decoder
        self.conv_desc_a = TtConv2D(
            desc.conv_descriptor_a.weight,
            desc.conv_descriptor_a.bias,
            in_channels=enc_hidden[-1],
            out_channels=torch_model.config.decoder_hidden_size,
            kernel_size=3,
            padding=1,
            device=device,
            activation="relu",
            **head_kwargs,
        )
        self.conv_desc_b = TtConv2D(
            desc.conv_descriptor_b.weight,
            desc.conv_descriptor_b.bias,
            in_channels=torch_model.config.decoder_hidden_size,
            out_channels=torch_model.config.descriptor_decoder_dim,
            kernel_size=1,
            padding=0,
            device=device,
            activation=None,
            **head_kwargs,
        )

    @staticmethod
    def _preprocess_host(pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract one channel & convert to NHWC layout for tt-nn conv input."""
        # HF model: (B, 3, H, W) -> first channel only -> (B, 1, H, W)
        one_ch = pixel_values[:, 0:1, :, :]
        # tt-nn conv expects NHWC flattened: [1, 1, B*H*W, C]
        b, c, h, w = one_ch.shape
        nhwc = one_ch.permute(0, 2, 3, 1).reshape(1, 1, b * h * w, c)
        return nhwc

    def _encoder_forward(self, tt_input, h: int, w: int, batch_size: int):
        x = tt_input
        cur_h, cur_w = h, w
        for conv_a, conv_b, add_pooling in self.enc_convs:
            x, cur_h, cur_w = conv_a(x, cur_h, cur_w, batch_size)
            x, cur_h, cur_w = conv_b(x, cur_h, cur_w, batch_size)
            if add_pooling:
                channels = x.shape[-1]
                x = ttnn.max_pool2d(
                    input_tensor=x,
                    batch_size=batch_size,
                    input_h=cur_h,
                    input_w=cur_w,
                    channels=channels,
                    kernel_size=[2, 2],
                    stride=[2, 2],
                    padding=[0, 0],
                    dilation=[1, 1],
                )
                cur_h //= 2
                cur_w //= 2
        return x, cur_h, cur_w

    def _run_device(self, pixel_values: torch.Tensor):
        """Run conv-heavy portion on device, returning host tensors for post-proc."""
        b, _, h, w = pixel_values.shape
        nhwc = self._preprocess_host(pixel_values)
        tt_in = ttnn.from_torch(
            nhwc,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Encoder trunk
        encoded, enc_h, enc_w = self._encoder_forward(tt_in, h, w, b)

        # Score branch — conv on device, softmax on host (65 channels are not
        # tile-aligned so on-device softmax includes padding zeros and corrupts
        # the distribution).
        s, _, _ = self.conv_score_a(encoded, enc_h, enc_w, b)
        s, _, _ = self.conv_score_b(s, enc_h, enc_w, b)

        # Descriptor branch
        d, _, _ = self.conv_desc_a(encoded, enc_h, enc_w, b)
        d, _, _ = self.conv_desc_b(d, enc_h, enc_w, b)

        # Bring both back to host as torch tensors in NCHW.
        scores_nhwc = ttnn.to_torch(s).reshape(b, enc_h, enc_w, KEYPOINT_DIM)
        descriptors_nhwc = ttnn.to_torch(d).reshape(b, enc_h, enc_w, DESCRIPTOR_DIM)

        scores_nchw = scores_nhwc.permute(0, 3, 1, 2).contiguous().float()
        descriptors_nchw = descriptors_nhwc.permute(0, 3, 1, 2).contiguous().float()

        # Host-side softmax over channel dim (matches reference exactly)
        scores_nchw = torch.softmax(scores_nchw, dim=1)
        # Host-side L2 normalize over descriptor dim
        descriptors_nchw = F.normalize(descriptors_nchw, p=2, dim=1)

        ttnn.deallocate(encoded)
        ttnn.deallocate(s)
        ttnn.deallocate(d)
        ttnn.deallocate(tt_in)

        return scores_nchw, descriptors_nchw

    # --- Host-side post-processing (data-dependent; keypoint count varies) ---

    @staticmethod
    def _simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
        def max_pool(x):
            return F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    def _decode_keypoints(self, scores_nchw: torch.Tensor, apply_nms: bool = True):
        # Drop the dustbin (last channel), fold 8x8 -> full-res
        scores = scores_nchw[:, :-1]  # (B, 64, h, w)
        b, _, fh, fw = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, fh, fw, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, fh * 8, fw * 8)
        if apply_nms:
            scores = self._simple_nms(scores, self.nms_radius)
        return scores

    def _extract_keypoints_single(self, scores_1hw: torch.Tensor):
        _, height, width = scores_1hw.shape
        keypoints = torch.nonzero(scores_1hw[0] > self.keypoint_threshold)
        scores = scores_1hw[0][tuple(keypoints.t())]
        # Border removal
        border = self.border_removal_distance
        mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
        mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
        mask = mask_h & mask_w
        keypoints = keypoints[mask]
        scores = scores[mask]
        # Top-K
        if self.max_keypoints >= 0 and keypoints.shape[0] > self.max_keypoints:
            scores, idx = torch.topk(scores, self.max_keypoints, dim=0)
            keypoints = keypoints[idx]
        # (y, x) -> (x, y)
        keypoints = torch.flip(keypoints, [1]).to(scores.dtype)
        return keypoints, scores

    @staticmethod
    def _sample_descriptors(keypoints, descriptors, scale: int = 8):
        batch_size, num_channels, height, width = descriptors.shape
        keypoints = keypoints - scale / 2 + 0.5
        divisor = torch.tensor([[(width * scale - scale / 2 - 0.5), (height * scale - scale / 2 - 0.5)]])
        divisor = divisor.to(keypoints)
        keypoints = keypoints / divisor
        keypoints = keypoints * 2 - 1
        keypoints = keypoints.view(batch_size, 1, -1, 2)
        descriptors = F.grid_sample(descriptors, keypoints, mode="bilinear", align_corners=True)
        descriptors = descriptors.reshape(batch_size, num_channels, -1)
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return descriptors

    def forward(self, pixel_values: torch.Tensor):
        scores_nchw, descriptors_nchw = self._run_device(pixel_values)
        scores_pre_nms = self._decode_keypoints(scores_nchw, apply_nms=False)
        scores_full = self._simple_nms(scores_pre_nms, self.nms_radius)

        b, _, h, w = pixel_values.shape
        list_keypoints, list_scores, list_descriptors = [], [], []
        for i in range(b):
            kp, sc = self._extract_keypoints_single(scores_full[i : i + 1])
            list_keypoints.append(kp)
            list_scores.append(sc)
            d = self._sample_descriptors(kp[None], descriptors_nchw[i : i + 1], scale=8)[0]
            list_descriptors.append(d.transpose(0, 1))

        max_kp = max(k.shape[0] for k in list_keypoints)
        keypoints_t = torch.zeros((b, max_kp, 2))
        scores_t = torch.zeros((b, max_kp))
        descriptors_t = torch.zeros((b, max_kp, DESCRIPTOR_DIM))
        mask_t = torch.zeros((b, max_kp), dtype=torch.int)
        for i, (kp, sc, dc) in enumerate(zip(list_keypoints, list_scores, list_descriptors)):
            keypoints_t[i, : kp.shape[0]] = kp
            scores_t[i, : sc.shape[0]] = sc
            descriptors_t[i, : dc.shape[0]] = dc
            mask_t[i, : sc.shape[0]] = 1
        keypoints_t = keypoints_t / torch.tensor([w, h])
        return {
            "keypoints": keypoints_t,
            "scores": scores_t,
            "descriptors": descriptors_t,
            "mask": mask_t,
            "raw_scores_map": scores_full,
            "raw_scores_pre_nms": scores_pre_nms,
            "raw_descriptors_map": descriptors_nchw,
        }
