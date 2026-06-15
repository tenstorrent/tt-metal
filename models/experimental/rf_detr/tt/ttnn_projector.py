# SPDX-License-Identifier: Apache-2.0
"""TTNN port of RF-DETR's C2f projector (on device).

Input : 4 backbone feature maps, each channels-last [1, 1600, 384] (HW=40x40 row-major).
Output: source tokens channels-last [1, 1600, 256].

ConvX = Conv2d(bias=False) -> channels-first LayerNorm (over channels) -> SiLU.
1x1 convs are pointwise -> ttnn.linear over the channel dim; 3x3 convs -> ttnn.conv2d.
"""
import torch
import ttnn

H = W = 40


class TtProjector:
    def __init__(self, ref_model, device):
        self.device = device
        proj = ref_model.backbone[0].projector
        c2f = proj.stages[0][0]
        self.final_ln = self._ln(proj.stages[0][1], device)
        self.c = c2f.c  # hidden channels (128)

        self.cv1 = self._convx_1x1(c2f.cv1, device)            # 1536 -> 256
        self.cv2 = self._convx_1x1(c2f.cv2, device)            # 640  -> 256
        self.bottlenecks = [
            (self._convx_3x3(b.cv1, device), self._convx_3x3(b.cv2, device)) for b in c2f.m
        ]
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
        )

    # ---- weight extraction ----
    def _ln(self, cln, device):
        return {
            "w": ttnn.from_torch(cln.weight.detach().reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            "b": ttnn.from_torch(cln.bias.detach().reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            "eps": float(cln.eps),
        }

    def _convx_1x1(self, convx, device):
        w = convx.conv.weight.detach().squeeze(-1).squeeze(-1)  # [Cout,Cin]
        return {
            "w": ttnn.from_torch(w.t().contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device),
            "ln": self._ln(convx.bn, device),
        }

    def _convx_3x3(self, convx, device):
        return {
            "cin": convx.conv.in_channels,
            "cout": convx.conv.out_channels,
            "w": ttnn.from_torch(convx.conv.weight.detach(), dtype=ttnn.bfloat16),  # [Cout,Cin,3,3] host
            "ln": self._ln(convx.bn, device),
        }

    # ---- ops ----
    def _ln_silu(self, x, ln):
        x = ttnn.layer_norm(x, weight=ln["w"], bias=ln["b"], epsilon=ln["eps"])
        return ttnn.silu(x)

    def _conv1x1(self, x, p):
        return self._ln_silu(ttnn.linear(x, p["w"]), p["ln"])

    def _conv3x3(self, x, p):
        # x: [1, 1600, Cin] -> conv2d wants [1,1,N*H*W,Cin]
        xc = ttnn.reshape(x, (1, 1, H * W, p["cin"]))
        out, [pw, pb] = ttnn.conv2d(
            input_tensor=xc, weight_tensor=p["w"], bias_tensor=None, device=self.device,
            in_channels=p["cin"], out_channels=p["cout"], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
            batch_size=1, input_height=H, input_width=W,
            conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat8_b, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            compute_config=self.compute_config, groups=1,
            return_weights_and_bias=True, return_output_dim=False,
        )
        p["w"] = pw  # cache prepared weights
        out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG) if out.is_sharded() else out
        out = ttnn.reshape(ttnn.to_layout(out, ttnn.TILE_LAYOUT), (1, H * W, p["cout"]))
        return self._ln_silu(out, p["ln"])

    def __call__(self, feature_maps_cl):
        """feature_maps_cl: list of 4 ttnn tensors [1,1600,384] (channels-last). -> [1,1600,256]."""
        x = ttnn.concat(feature_maps_cl, dim=-1)          # [1,1600,1536]
        x = self._conv1x1(x, self.cv1)                    # -> [1,1600,256]
        parts = [x[:, :, : self.c], x[:, :, self.c :]]    # split 2 x 128
        cur = parts[-1]
        for cv1, cv2 in self.bottlenecks:
            cur = self._conv3x3(cur, cv1)
            cur = self._conv3x3(cur, cv2)
            parts.append(cur)
        x = ttnn.concat(parts, dim=-1)                    # [1,1600,640]
        x = self._conv1x1(x, self.cv2)                    # -> [1,1600,256]
        x = ttnn.layer_norm(x, weight=self.final_ln["w"], bias=self.final_ln["b"], epsilon=self.final_ln["eps"])
        return x
