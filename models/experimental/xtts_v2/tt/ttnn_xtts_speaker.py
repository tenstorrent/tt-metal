# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the XTTS-v2 ResNet speaker encoder (Block 2): logmel -> d-vector.

Mirrors reference/xtts_speaker_ref.py op-for-op, running ENTIRELY on device:

    logmel [1,64,T] --InstanceNorm(64, over T)--> unsqueeze -> [1,1,64,T] image
      --conv1(1->32,k3)+relu+bn--> layer1..4 (SE-ResNet, strides 1/2/2/2) --> [1,256,8,T/8]
      --fold freq into channels [1,2048,T']--> attentive stats pooling --> [1,4096]
      --fc(4096->512)--> L2-normalize --> d-vector [1,512]   (=speaker_embedding after unsqueeze(-1))

The d-vector conditions the HiFi-GAN vocoder (Block 4). One-shot per voice, so fidelity-first:
fp32 tensors + fp32 accumulation (HiFi3 + fp32_dest_acc), like Blocks 1 & 4.

Real 3x3 convs use ttnn.conv2d (input channels-last [1,1,H*W,C] ROW_MAJOR, weight [Cout,Cin,3,3]
as-is, output [1,1,Ho*Wo,Cout] TILE). The 1x1 convs (SE FCs, downsample, ASP attention) are matmuls
over channels. BatchNorms are eval-mode, so they fold to a per-channel affine (scale/shift). The
freq->channel fold ([1,8,T',256] -> [1,T',2048=(c*8+h)]) is a reshape+permute+reshape.

Validate + time vs the CPU reference:
    TT_METAL_HOME=<repo> PYTHONPATH=<repo> python models/experimental/xtts_v2/tt/ttnn_xtts_speaker.py
"""

import torch
import ttnn

from models.experimental.xtts_v2.reference.xtts_gpt_ref import DEFAULT_CKPT
from models.experimental.xtts_v2.reference.xtts_speaker_ref import load_speaker_state

DTYPE = ttnn.float32
BN_EPS = 1e-5  # BatchNorm2d/1d default
IN_EPS = 1e-5  # InstanceNorm1d default
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)

# (planes, num_blocks, first_stride, first_in_channels) per ResNet stage
LAYER_CFG = [(32, 3, 1, 32), (64, 4, 2, 32), (128, 6, 2, 64), (256, 3, 2, 128)]
OUTMAP = 8  # freq bins after 3 stride-2 stages: 64 / 8


class TtSpeakerEncoder:
    """On-device ResNet speaker encoder. __call__(logmel [1,64,T]) -> d-vector [1,512]."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT):
        self.device = device
        w = {k: v.float() for k, v in load_speaker_state(ckpt_path).items()}
        dev = lambda t: ttnn.from_torch(t.contiguous(), dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        host = lambda t: ttnn.from_torch(t.contiguous(), dtype=DTYPE)  # conv2d moves weights to device itself
        lin = lambda k: dev(w[k].t())  # Linear/1x1-conv weight [out,in] -> matmul weight [in,out]

        def bn(prefix, ndim=4):  # eval BatchNorm -> per-channel affine (scale, shift)
            g, b = w[prefix + ".weight"], w[prefix + ".bias"]
            m, v = w[prefix + ".running_mean"], w[prefix + ".running_var"]
            s = g / torch.sqrt(v + BN_EPS)
            shape = (1, 1, 1, -1) if ndim == 4 else (1, 1, -1)
            return dev(s.reshape(*shape)), dev((b - m * s).reshape(*shape))

        # stem
        self.conv1_w, self.conv1_b = host(w["conv1.weight"]), host(w["conv1.bias"].reshape(1, 1, 1, -1))
        self.bn1 = bn("bn1")
        # SE-ResNet blocks
        self.blocks = []  # list of dicts, in forward order
        for li, (planes, nb, s0, in0) in enumerate(LAYER_CFG, start=1):
            for bi in range(nb):
                p = f"layer{li}.{bi}."
                inc = in0 if bi == 0 else planes
                stride = s0 if bi == 0 else 1
                blk = {
                    "p": p, "inc": inc, "planes": planes, "stride": stride,
                    "c1": host(w[p + "conv1.weight"]), "bn1": bn(p + "bn1"),
                    "c2": host(w[p + "conv2.weight"]), "bn2": bn(p + "bn2"),
                    "se0w": lin(p + "se.fc.0.weight"), "se0b": dev(w[p + "se.fc.0.bias"]),
                    "se2w": lin(p + "se.fc.2.weight"), "se2b": dev(w[p + "se.fc.2.bias"]),
                    "ds": (p + "downsample.0.weight") in w,
                }
                if blk["ds"]:
                    blk["dsw"] = host(w[p + "downsample.0.weight"])
                    blk["dsbn"] = bn(p + "downsample.1")
                self.blocks.append(blk)
        # attentive statistics pooling (ASP): Conv1d k1 -> matmul over channels
        self.att0w = dev(w["attention.0.weight"].squeeze(-1).t())  # [2048,128]
        self.att0b = dev(w["attention.0.bias"])
        self.attbn = bn("attention.2", ndim=3)
        self.att3w = dev(w["attention.3.weight"].squeeze(-1).t())  # [128,2048]
        self.att3b = dev(w["attention.3.bias"])
        self.fcw, self.fcb = dev(w["fc.weight"].t()), dev(w["fc.bias"])  # [4096,512]

    # ---- primitives ----
    def _conv(self, x, wt, bt, in_c, out_c, k, stride, pad, H, W):  # -> (out [1,1,Ho*Wo,out_c] TILE, Ho, Wo)
        out, [Ho, Wo], _ = ttnn.conv2d(
            input_tensor=ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), weight_tensor=wt, bias_tensor=bt,
            in_channels=in_c, out_channels=out_c, device=self.device, kernel_size=(k, k),
            stride=(stride, stride), padding=(pad, pad), batch_size=1, input_height=H, input_width=W, groups=1,
            compute_config=COMPUTE_CONFIG, conv_config=ttnn.Conv2dConfig(weights_dtype=DTYPE),
            return_output_dim=True, return_weights_and_bias=True,
        )
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out)
        return out, Ho, Wo

    def _bn(self, x, sc_sh):
        return ttnn.add(ttnn.multiply(x, sc_sh[0]), sc_sh[1])

    def _se(self, x, b):  # x [1,1,HW,C] -> scaled by squeeze-excitation
        C = x.shape[3]
        y = ttnn.reshape(ttnn.mean(x, dim=2, keepdim=True), [1, C])  # global avg pool over HW
        y = ttnn.relu(ttnn.linear(y, b["se0w"], bias=b["se0b"], compute_kernel_config=COMPUTE_CONFIG))
        y = ttnn.sigmoid(ttnn.linear(y, b["se2w"], bias=b["se2b"], compute_kernel_config=COMPUTE_CONFIG))
        return ttnn.multiply(x, ttnn.reshape(y, [1, 1, 1, C]))

    def _block(self, x, H, W, b):
        residual = x
        out, Ho, Wo = self._conv(x, b["c1"], None, b["inc"], b["planes"], 3, b["stride"], 1, H, W)
        out = self._bn(ttnn.relu(out), b["bn1"])  # ReLU BEFORE BN (coqui's order)
        out, _, _ = self._conv(out, b["c2"], None, b["planes"], b["planes"], 3, 1, 1, Ho, Wo)
        out = self._se(self._bn(out, b["bn2"]), b)
        if b["ds"]:
            residual, _, _ = self._conv(x, b["dsw"], None, b["inc"], b["planes"], 1, b["stride"], 0, H, W)
            residual = self._bn(residual, b["dsbn"])
        return ttnn.relu(ttnn.add(out, residual)), Ho, Wo

    @torch.no_grad()
    def __call__(self, logmel):  # torch [1,64,T] -> torch d-vector [1,512]
        T = logmel.shape[2]
        # InstanceNorm1d(64, affine=False): per-freq normalize over time
        x = ttnn.from_torch(logmel.contiguous(), dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device)
        xc = ttnn.subtract(x, ttnn.mean(x, dim=2, keepdim=True))
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=2, keepdim=True)
        x = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, IN_EPS)))  # [1,64,T]
        x = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [1, 1, 64 * T, 1])  # image [H=64,W=T,C=1]

        x, H, W = self._conv(x, self.conv1_w, self.conv1_b, 1, 32, 3, 1, 1, 64, T)
        x = self._bn(ttnn.relu(x), self.bn1)
        for b in self.blocks:
            x, H, W = self._block(x, H, W, b)  # -> [1,1,H*W,256], H=8

        # fold freq into channels: [1,1,H*W,256] -> [1,H,W,256] -> [1,W,256,H] -> [1,W,2048=(c*8+h)]
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [1, H, W, 256])
        x = ttnn.permute(x, (0, 2, 3, 1))  # [1,W,256,H]
        x = ttnn.to_layout(ttnn.reshape(x, [1, W, 256 * H]), ttnn.TILE_LAYOUT)  # [1,T',2048]

        # ASP attention weights over the folded features
        a = ttnn.relu(ttnn.linear(x, self.att0w, bias=self.att0b, compute_kernel_config=COMPUTE_CONFIG))  # [1,T',128]
        a = self._bn(a, self.attbn)
        a = ttnn.linear(a, self.att3w, bias=self.att3b, compute_kernel_config=COMPUTE_CONFIG)  # [1,T',2048]
        xt = ttnn.transpose(x, 1, 2)  # [1,2048,T']
        wt = ttnn.softmax(ttnn.transpose(a, 1, 2), dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        mu = ttnn.sum(ttnn.multiply(xt, wt), dim=2, keepdim=True)  # [1,2048,1]
        e2 = ttnn.sum(ttnn.multiply(ttnn.multiply(xt, xt), wt), dim=2, keepdim=True)
        sg = ttnn.sqrt(ttnn.clip(ttnn.subtract(e2, ttnn.multiply(mu, mu)), 1e-5, 1e9))
        s = ttnn.reshape(ttnn.concat([mu, sg], dim=1), [1, 4096])  # [mean; std] -> [1,4096]
        s = ttnn.linear(s, self.fcw, bias=self.fcb, compute_kernel_config=COMPUTE_CONFIG)  # [1,512]
        s = ttnn.multiply(s, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(s, s), dim=-1, keepdim=True), 1e-12)))  # L2
        return ttnn.to_torch(s).float()


def main():
    import time

    from models.experimental.xtts_v2.reference import xtts_speaker_ref as ref
    from models.experimental.xtts_v2.reference.xtts_gpt_ref import pcc

    device = ttnn.open_device(device_id=0, l1_small_size=131072)
    try:
        core = ref.build_reference()
        enc = TtSpeakerEncoder(device)
        for T in (128, 505):
            logmel = ref.make_synthetic_logmel(n_frames=T)
            ref_dv = core(logmel, l2_norm=True)  # [1,512]
            got = enc(logmel)
            print(f"[speaker] T={T:4d}  d-vector PCC vs reference: {pcc(got, ref_dv):.5f}  {tuple(got.shape)}")
            enc(logmel)  # warm
            t0 = time.perf_counter()
            enc(logmel)
            print(f"[speaker] T={T:4d}  logmel -> d-vector: {time.perf_counter() - t0:.3f}s")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
