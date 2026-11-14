# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch reference of the Hy4 final head, written from vllm/models/hy_v4/nvidia/{hc.py HYV4HCHeadLayer, model.py HYV4Model.forward / HYV4ForCausalLM.compute_logits}. CPU only.

vLLM semantics (source-read, see lm-head/RESULT.md):
  streams x [T, 4, 6144] (bf16 or fp32 between layers)
  x = x.flatten(1).float(); rs = rsqrt(mean(x^2 over the 4*6144 values) + rms_norm_eps)          # eps 1e-5
  mixes = (x @ hc_head_fn^T [24576 -> 4], fp32) * rs
  pre = sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps                                     # hc_eps 1e-6, per stream
  y = sum_s pre[s] * x[s]  -> cast to the input dtype (bf16)                                       # [T, 6144]
  h = RMSNorm(6144, eps 1e-5)(y) -> bf16 (fp32 math inside, weight bf16)
  logits = torch.mm(h bf16, lm_head.weight^T bf16, out_dtype=float32)   (enable_lm_head_fp32 => ModelConfig.head_dtype float32: the WEIGHT STAYS bf16, only the accumulation / output is fp32)
  soft_logits_capping: not in the Hy4 checkpoint config (getattr default False).
"""
import torch

RMS_EPS = 1e-5
HC_EPS = 1e-6
VOCAB, DIM, HC = 120832, 6144, 4


def hc_head(x, fn, base, scale, rms_eps=RMS_EPS, hc_eps=HC_EPS, out_dtype=torch.bfloat16):
    """x [T, 4, D] -> [T, D]"""
    xf = x.float().reshape(x.shape[0], -1)
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + rms_eps)
    mixes = (xf @ fn.float().t()) * rs
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + hc_eps
    y = (pre.unsqueeze(-1) * xf.reshape(x.shape)).sum(1)
    return y.to(out_dtype)


def rmsnorm(y, w, eps=RMS_EPS, out_dtype=torch.bfloat16):
    yf = y.float()
    return ((yf * torch.rsqrt(yf.square().mean(-1, keepdim=True) + eps)).to(out_dtype) * w.to(out_dtype)).to(out_dtype)


def rmsnorm_fused(y, w, eps=RMS_EPS, out_dtype=torch.bfloat16):
    """CUDA-kernel style: fp32 math, one rounding at the end"""
    yf = y.float()
    return (yf * torch.rsqrt(yf.square().mean(-1, keepdim=True) + eps) * w.float()).to(out_dtype)


def head_hidden(x, W, norm="fused"):
    h = hc_head(x, W["hc_head.fn"], W["hc_head.base"], W["hc_head.scale"])
    return (rmsnorm_fused if norm == "fused" else rmsnorm)(h, W["final_norm"])


def logits_ref(h_bf16, lm_head_bf16):
    """fp32 accumulation of exact bf16 products == vLLM torch.mm(out_dtype=float32)"""
    return h_bf16.float() @ lm_head_bf16.float().t()
