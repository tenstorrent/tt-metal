# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What does *not* explain BFP4's synthetic-vs-real layer PCC gap.

The layer A/B measures BFP4 MLP weights against the HF reference at PCC
0.9939 / 0.9925 (sliding / full) on ``reference.synthetic_state_dict`` weights and
**0.9977 / 0.9970** on the released checkpoint
(``logs/layer_ab_precision.log``, ``logs/layer_ab_real_final.log``) -- a 2.6x
difference in error.  ``$optimize`` OPT-012 says to reconcile that with evidence,
so this probe tests the plausible mechanisms.  **All three are refuted**, and the
probe is committed for that negative result:

* ``RANGE`` -- per-block dynamic range ``max|w| / mean|w|`` over the 16-element
  blocks a ``bfloat4_b`` tile shares an exponent across.  Synthetic and real agree
  to within 4 % on every projection, so the natural "i.i.d. samples spread each
  block wider" story is simply not true here.
* ``QUANT`` -- the real on-device BFP4/BFP8 round-trip error of each weight
  tensor.  It is *larger* on the real weights, by 1x to 8x, because the real
  tensors have heavier tails and this statistic is set by outlier blocks.  The
  wrong direction.
* ``SNR`` -- the statistic PCC actually responds to: output correlation of a
  BFP4-weight matmul against an FP32-weight one, with the *same* unit-RMS
  activation.  Real and synthetic land within 1-9 % of each other
  (0.9930-0.9936 either way).  So BFP4 represents the two weight sets equally
  well *per projection*; the gap is not in the quantisation at all.

A fourth candidate was measured separately and is also too small: the
branch-to-residual ratio ``||y - x|| / ||x||`` of the HF layer is 0.943 (real) vs
1.042 (synthetic) on ``sliding`` and 1.093 vs 1.183 on ``full`` -- a 10 % effect
against a 2.6x one.

So the gap is an interaction *inside* the layer -- most plausibly error
cancellation across the SwiGLU product and the down projection when the weights
are structured -- which this stage did not isolate further, and deliberately does
not claim to have.

That does not change the policy decision, because the decision does not depend on
the mechanism: OPT-012's rule is that a synthetic-distribution PCC cannot veto a
policy that passes real-weight PCC under the disputed conditions, and the
real-weight evidence in ``tests/test_optimized_decoder.py`` is broad (six prefill
lengths including sub-tile and multi-chunk, an eight-step decode off the BFP8
cache, traced replay, batch 8, both layer kinds) and clears 0.995 with margin.
What the refutations *do* change is the README: the looser synthetic bar is
recorded as an unexplained measured gap, not dressed up in a mechanism.

    python .../bench/bfp_block_range_probe.py
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import reference_layer_indices

#: ``bfloat4_b`` / ``bfloat8_b`` share one exponent across this many elements.
BFP_BLOCK = 16

SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.gate_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def block_dynamic_range(w: torch.Tensor) -> tuple[float, float]:
    """``(mean, p99)`` of ``max|w| / mean|w|`` over 16-element blocks."""
    flat = w.detach().to(torch.float32).flatten()
    usable = flat.numel() - flat.numel() % BFP_BLOCK
    blocks = flat[:usable].reshape(-1, BFP_BLOCK).abs()
    ratio = blocks.amax(dim=1) / (blocks.mean(dim=1) + 1e-12)
    return float(ratio.mean()), float(torch.quantile(ratio, 0.99))


def roundtrip_error(w: torch.Tensor, dtype, mesh) -> float:
    """Max relative error of a real on-device quantise/dequantise round trip."""
    padded = w.detach().to(torch.float32)
    if padded.dim() == 2:
        padded = padded.unsqueeze(0).unsqueeze(0)
    tt = ttnn.from_torch(
        padded, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = ttnn.to_torch(tt).to(torch.float32)
    ttnn.deallocate(tt)
    scale = padded.abs().mean().clamp_min(1e-12)
    return float((got - padded).abs().max() / scale)


def projection_pcc(w: torch.Tensor, dtype, mesh, rows: int = 32, seed: int = 4242) -> float:
    """Output PCC of a ``dtype``-weight matmul against an FP32-weight reference.

    The activation is the same unit-RMS tensor the layer harness feeds in, and it
    is identical between the real and synthetic weight runs, so the only variable
    is the weight distribution.  This is the statistic PCC-based acceptance
    actually responds to.
    """
    weight = w.detach().to(torch.float32)
    if weight.dim() == 2:
        # HF stores [out, in]; the matmuls want [in, out].
        weight = weight.transpose(0, 1).contiguous()
    weight = weight.reshape(1, 1, weight.shape[-2], weight.shape[-1])
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, rows, weight.shape[-2], generator=generator)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True))
    reference = (x.to(torch.float64) @ weight.to(torch.float64)).float()

    tt_w = ttnn.from_torch(
        weight, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_x = ttnn.from_torch(
        x, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.linear(tt_x, tt_w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = ttnn.to_torch(out).to(torch.float32)
    for tensor in (tt_w, tt_x, out):
        ttnn.deallocate(tensor)
    a, b = got.flatten().to(torch.float64), reference.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="sliding,full")
    args = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    try:
        idxs = reference_layer_indices(R.hf_config())
        for kind in args.kinds.split(","):
            layer_idx = idxs[kind]
            real = R.real_state_dict(layer_idx)
            synth = R.synthetic_state_dict(layer_idx)
            prefix = f"{R.layer_prefix(layer_idx)}."
            for suffix in SUFFIXES:
                key = prefix + suffix
                rw, sw = real[key], synth[key]
                rm, rp = block_dynamic_range(rw)
                sm, sp = block_dynamic_range(sw)
                print(
                    f"RANGE kind={kind:8s} {suffix:28s} "
                    f"real mean={rm:6.3f} p99={rp:6.3f}   synth mean={sm:6.3f} p99={sp:6.3f}   "
                    f"synth/real mean={sm / rm:5.3f}",
                    flush=True,
                )
                for label, dtype in (("bfp8", ttnn.bfloat8_b), ("bfp4", ttnn.bfloat4_b)):
                    re_ = roundtrip_error(rw, dtype, mesh)
                    se_ = roundtrip_error(sw, dtype, mesh)
                    print(
                        f"QUANT kind={kind:8s} {suffix:28s} {label} "
                        f"real max_rel_err={re_:8.5f}  synth max_rel_err={se_:8.5f}  "
                        f"synth/real={se_ / max(re_, 1e-12):5.3f}",
                        flush=True,
                    )
                    rp = projection_pcc(rw, dtype, mesh)
                    sp = projection_pcc(sw, dtype, mesh)
                    print(
                        f"SNR   kind={kind:8s} {suffix:28s} {label} "
                        f"real out_pcc={rp:.6f}  synth out_pcc={sp:.6f}  "
                        f"real_err/synth_err={(1 - rp) / max(1 - sp, 1e-12):6.3f}",
                        flush=True,
                    )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
