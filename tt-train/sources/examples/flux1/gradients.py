# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Flux1 backward pass comparison: HuggingFace (PyTorch) vs ttml.

Compares per-parameter gradients between HuggingFace (CPU/bfloat16) and
ttml (Tenstorrent device/bfloat16) after a single forward+backward pass
with MSE loss on the noise prediction output.

Also records activations and true gradients at each boundary (after embeddings,
after each joint/double transformer block, after each single transformer block)
via HuggingFace hooks and ttml identity forward/backward ops, and writes a
tabular comparison plus CSV.

Usage:
    cd tt-train/sources/examples/flux1

    python3 gradients.py \
        --checkpoint black-forest-labs/FLUX.1-dev \
        --mesh_shape 1 8

    # Optional CSV paths (defaults: flux1_gradients_intermediates.csv, gradients_comparison.csv)
    python3 gradients.py --intermediates_csv my_intermediates.csv --param_gradients_csv my_params.csv
"""

from __future__ import annotations

import argparse
import csv
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

import ttnn
import ttml

from generate import (
    _sinusoidal_proj,
    _pack_latents,
    _latent_image_ids,
    encode_prompts,
    setup_device,
)
from model_flux_distributed import (
    DistributedFlux1Transformer,
    Flux1Config,
    create_flux1_config_from_hf,
    empty_init,
    load_weights_from_hf_distributed,
    _build_weight_mapping,
    _adaln,
    _chunk4d,
    _to_float32,
    _to_bfloat16,
    Typecast,
)


def _layer_sort_key(name):
    """Sort key that orders parameters by dataflow: embeddings → double blocks (0..N)
    → single blocks (0..N) → output layers, with sub-params in a stable order."""
    import re
    parts = name.split(".")
    if parts[0] == "time_text_embed":
        return (0, 0, name)
    if parts[0] == "x_embedder":
        return (1, 0, name)
    if parts[0] == "context_embedder":
        return (2, 0, name)
    m = re.match(r"transformer_blocks\.(\d+)\.", name)
    if m:
        return (3, int(m.group(1)), name)
    m = re.match(r"single_transformer_blocks\.(\d+)\.", name)
    if m:
        return (4, int(m.group(1)), name)
    if parts[0] in ("norm_out", "proj_out"):
        return (5, 0, name)
    return (6, 0, name)


def _all_boundary_names(num_double: int, num_single: int):
    keys = ["after_embed"]
    keys.extend(f"after_double_{i}" for i in range(num_double))
    keys.extend(f"after_single_{j}" for j in range(num_single))
    return keys


def _ttml_tensor_to_torch_flat(device, t):
    """Replicated ttml tensor / ttnn tensor → torch [S, D] or [1,S,D] for comparison."""
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    if hasattr(t, "get_value"):
        tt = t.get_value()
    else:
        tt = t
    x = ttnn.to_torch(tt, mesh_composer=composer).to(torch.float32)
    x = x[0]
    while x.dim() > 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.dim() == 3:
        x = x[0]
    return x.detach().clone()


def _finalize_ttml_boundary_grads(capture: dict, device):
    """After backward(), read ∂L/∂h at each boundary from mul-output tensor grads.

    Custom ``Function`` identity nodes are not used here: ``Function.apply`` skips
    registering user ``backward`` when outputs already carry autograd nodes
    (see ttml ``autograd/function.py``).  We use ``mul(x, 1.0)`` for a graph edge
    and pull grads from the resulting tensors after ``backward``.

    If a tensor never receives a gradient (e.g. ``prompt`` after the last single block:
    the head only uses ``spatial`` for ``norm_out``/``proj_out``), TTML may leave
    ``is_grad_initialized()`` false while HF hooks still deliver zeros.  We then
    store ``zeros_like`` the captured forward tensor so the comparison is 0 vs 0, not NaN.
    """
    refs = capture.pop("_boundary_tensor_refs", [])
    for key, ts, tp in refs:
        fwd_s = capture.get(f"{key}_spatial_fwd")
        fwd_p = capture.get(f"{key}_prompt_fwd")

        if ts is not None and hasattr(ts, "is_grad_initialized"):
            if ts.is_grad_initialized():
                capture[f"{key}_spatial_bwd"] = _ttml_tensor_to_torch_flat(device, ts.get_grad())
            elif fwd_s is not None:
                capture[f"{key}_spatial_bwd"] = torch.zeros_like(fwd_s)

        if tp is not None and hasattr(tp, "is_grad_initialized"):
            if tp.is_grad_initialized():
                capture[f"{key}_prompt_bwd"] = _ttml_tensor_to_torch_flat(device, tp.get_grad())
            elif fwd_p is not None:
                capture[f"{key}_prompt_bwd"] = torch.zeros_like(fwd_p)


def _make_block_boundary_apply(capture: dict, device, key: str):
    """Identity that creates a fresh autograd node (so we can read its grad after backward)
    while preserving the input dtype.  ``ttml.ops.binary.mul(x, 1.0)`` runs through the
    ttnn ``fast_and_approximate_mode`` path which silently downcasts to bfloat16, so we
    use a same-dtype typecast instead to keep the float32 residual streams float32.
    """

    def apply(spatial, prompt):
        s_in = spatial.get_value().dtype
        p_in = prompt.get_value().dtype
        print(f"[trace boundary {key}] in : spatial={s_in} prompt={p_in}")
        spatial = Typecast.apply(spatial, s_in)
        prompt = Typecast.apply(prompt, p_in)
        print(
            f"[trace boundary {key}] out: spatial={spatial.get_value().dtype} "
            f"prompt={prompt.get_value().dtype}"
        )
        capture[f"{key}_spatial_fwd"] = _ttml_tensor_to_torch_flat(device, spatial)
        capture[f"{key}_prompt_fwd"] = _ttml_tensor_to_torch_flat(device, prompt)
        capture.setdefault("_boundary_tensor_refs", []).append((key, spatial, prompt))
        return spatial, prompt

    return apply


def _patch_ttml_forward_with_boundaries(ttml_model, device, capture: dict):
    """Insert boundary capture ops between embed and each transformer block (same order as HF hooks)."""

    def forward_patched(
        self,
        spatial,
        prompt,
        timestep_proj,
        guidance_proj,
        pooled,
        spatial_rope_cos,
        spatial_rope_sin,
        prompt_rope_cos,
        prompt_rope_sin,
    ):
        time_embed = self.time_text_embed(timestep_proj, guidance_proj, pooled)
        time_embed = ttml.ops.unary.silu(time_embed)
        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)
        print(f"[trace] after embedders: spatial={spatial.get_value().dtype} prompt={prompt.get_value().dtype}")
        spatial = _to_float32(spatial)
        prompt = _to_float32(prompt)
        print(f"[trace] after _to_float32: spatial={spatial.get_value().dtype} prompt={prompt.get_value().dtype}")
        extra = (time_embed, spatial_rope_cos, spatial_rope_sin, prompt_rope_cos, prompt_rope_sin)

        apply_b = _make_block_boundary_apply(capture, device, "after_embed")
        spatial, prompt = apply_b(spatial, prompt)
        print(f"[trace] after boundary apply 'after_embed': spatial={spatial.get_value().dtype} prompt={prompt.get_value().dtype}")

        for i, block in enumerate(self.transformer_blocks):
            print(f"[trace] block {i} input: spatial={spatial.get_value().dtype} prompt={prompt.get_value().dtype}")
            spatial, prompt = block(spatial, prompt, *extra)
            apply_b = _make_block_boundary_apply(capture, device, f"after_double_{i}")
            spatial, prompt = apply_b(spatial, prompt)

        for j, block in enumerate(self.single_transformer_blocks):
            spatial, prompt = block(spatial, prompt, *extra)
            apply_b = _make_block_boundary_apply(capture, device, f"after_single_{j}")
            spatial, prompt = apply_b(spatial, prompt)

        spatial_time = self.norm_out(time_embed)
        scale, shift = _chunk4d(spatial_time, 2)
        spatial = _adaln(spatial, scale, shift)
        return self.proj_out(_to_bfloat16(spatial))

    ttml_model.forward = types.MethodType(forward_patched, ttml_model)


class HFBoundaryCapture:
    """HuggingFace: forward hooks on embedders / blocks; backward via ``Tensor.register_hook`` on outputs.

    ``register_full_backward_hook`` is unreliable for some module/return patterns; per-output
    tensor hooks match the gradient w.r.t. each block output (same as TTML boundary grads).
    """

    def __init__(self):
        self.capture: dict = {}
        self._handles = []

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def register(self, hf_model, num_double: int, num_single: int):
        def fwd_x(module, inp, out):
            self.capture["after_embed_spatial_fwd"] = out.detach().clone()

            def hook(grad):
                if grad is not None:
                    self.capture["after_embed_spatial_bwd"] = grad.detach().clone()
                return grad

            out.register_hook(hook)

        def fwd_ctx(module, inp, out):
            self.capture["after_embed_prompt_fwd"] = out.detach().clone()

            def hook(grad):
                if grad is not None:
                    self.capture["after_embed_prompt_bwd"] = grad.detach().clone()
                return grad

            out.register_hook(hook)

        self._handles.append(hf_model.x_embedder.register_forward_hook(fwd_x))
        self._handles.append(hf_model.context_embedder.register_forward_hook(fwd_ctx))

        for i in range(num_double):

            def fwd_d(module, inp, out, ii=i):
                self.capture[f"after_double_{ii}_prompt_fwd"] = out[0].detach().clone()
                self.capture[f"after_double_{ii}_spatial_fwd"] = out[1].detach().clone()

                def hook_p(grad):
                    if grad is not None:
                        self.capture[f"after_double_{ii}_prompt_bwd"] = grad.detach().clone()
                    return grad

                def hook_s(grad):
                    if grad is not None:
                        self.capture[f"after_double_{ii}_spatial_bwd"] = grad.detach().clone()
                    return grad

                out[0].register_hook(hook_p)
                out[1].register_hook(hook_s)

            self._handles.append(hf_model.transformer_blocks[i].register_forward_hook(fwd_d))

        for j in range(num_single):

            def fwd_s(module, inp, out, jj=j):
                self.capture[f"after_single_{jj}_prompt_fwd"] = out[0].detach().clone()
                self.capture[f"after_single_{jj}_spatial_fwd"] = out[1].detach().clone()

                def hook_p(grad):
                    if grad is not None:
                        self.capture[f"after_single_{jj}_prompt_bwd"] = grad.detach().clone()
                    return grad

                def hook_s(grad):
                    if grad is not None:
                        self.capture[f"after_single_{jj}_spatial_bwd"] = grad.detach().clone()
                    return grad

                out[0].register_hook(hook_p)
                out[1].register_hook(hook_s)

            self._handles.append(hf_model.single_transformer_blocks[j].register_forward_hook(fwd_s))


def _pair_stats(hf_t: torch.Tensor | None, ttml_t: torch.Tensor | None):
    if hf_t is None or ttml_t is None:
        return None
    a = hf_t.float().flatten()
    b = ttml_t.float().flatten()
    n = min(a.numel(), b.numel())
    if n == 0:
        return None
    a, b = a[:n], b[:n]
    diff = (a - b).abs()
    cos = (torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)).item()
    mrd = (diff / a.abs().clamp(min=1.0)).mean().item()
    return {
        "cos_sim": cos,
        "ad_mean": diff.mean().item(),
        "ad_max": diff.max().item(),
        "mrd": mrd,
        "hf_norm": a.norm().item(),
        "ttml_norm": b.norm().item(),
    }


def _compare_intermediates(hf_cap: dict, ttml_cap: dict, boundary_names: list[str], csv_path: str | None):
    """Print a table like debug_norm_k and optionally write CSV.

    ``boundary_names`` must be in **execution order** (as from ``_all_boundary_names``); rows are
    printed in that order (not lexicographic).
    """
    rows = []
    name_w = max(len("boundary"), max(len(b) for b in boundary_names))
    print(
        f"\n{'boundary':<{name_w}}  "
        f"{'sp_fwd_cs':>10}  {'sp_fwd_adm':>11}  {'sp_fwd_adx':>11}  {'sp_fwd_mrd':>11}  "
        f"{'pr_fwd_cs':>10}  {'pr_fwd_adm':>11}  {'pr_fwd_adx':>11}  {'pr_fwd_mrd':>11}  "
        f"{'sp_bwd_cs':>10}  {'pr_bwd_cs':>10}"
    )
    print("-" * (name_w + 126))

    fieldnames = [
        "boundary",
        "spatial_fwd_cos_sim",
        "spatial_fwd_ad_mean",
        "spatial_fwd_ad_max",
        "spatial_fwd_mrd",
        "prompt_fwd_cos_sim",
        "prompt_fwd_ad_mean",
        "prompt_fwd_ad_max",
        "prompt_fwd_mrd",
        "spatial_bwd_cos_sim",
        "spatial_bwd_ad_mean",
        "spatial_bwd_ad_max",
        "spatial_bwd_mrd",
        "prompt_bwd_cos_sim",
        "prompt_bwd_ad_mean",
        "prompt_bwd_ad_max",
        "prompt_bwd_mrd",
    ]

    for key in boundary_names:
        sp = _pair_stats(
            hf_cap.get(f"{key}_spatial_fwd"),
            ttml_cap.get(f"{key}_spatial_fwd"),
        )
        pp = _pair_stats(
            hf_cap.get(f"{key}_prompt_fwd"),
            ttml_cap.get(f"{key}_prompt_fwd"),
        )
        sb = _pair_stats(
            hf_cap.get(f"{key}_spatial_bwd"),
            ttml_cap.get(f"{key}_spatial_bwd"),
        )
        pb = _pair_stats(
            hf_cap.get(f"{key}_prompt_bwd"),
            ttml_cap.get(f"{key}_prompt_bwd"),
        )

        def _cs(x):
            return x["cos_sim"] if x else float("nan")

        def _adm(x):
            return x["ad_mean"] if x else float("nan")

        def _adx(x):
            return x["ad_max"] if x else float("nan")

        def _mrd(x):
            return x["mrd"] if x else float("nan")

        row = {
            "boundary": key,
            "spatial_fwd_cos_sim": _cs(sp),
            "spatial_fwd_ad_mean": _adm(sp),
            "spatial_fwd_ad_max": _adx(sp),
            "spatial_fwd_mrd": _mrd(sp),
            "prompt_fwd_cos_sim": _cs(pp),
            "prompt_fwd_ad_mean": _adm(pp),
            "prompt_fwd_ad_max": _adx(pp),
            "prompt_fwd_mrd": _mrd(pp),
            "spatial_bwd_cos_sim": _cs(sb),
            "spatial_bwd_ad_mean": _adm(sb),
            "spatial_bwd_ad_max": _adx(sb),
            "spatial_bwd_mrd": _mrd(sb),
            "prompt_bwd_cos_sim": _cs(pb),
            "prompt_bwd_ad_mean": _adm(pb),
            "prompt_bwd_ad_max": _adx(pb),
            "prompt_bwd_mrd": _mrd(pb),
        }
        rows.append(row)

        print(
            f"{key:<{name_w}}  "
            f"{_cs(sp):10.6f}  {_adm(sp):11.6f}  {_adx(sp):11.6f}  {_mrd(sp):11.6f}  "
            f"{_cs(pp):10.6f}  {_adm(pp):11.6f}  {_adx(pp):11.6f}  {_mrd(pp):11.6f}  "
            f"{_cs(sb):10.6f}  {_cs(pb):10.6f}"
        )

    print("-" * (name_w + 126))

    # Save all captured tensors for offline analysis
    diag_path = csv_path.replace(".csv", "_tensors.pt") if csv_path else "flux1_diag_tensors.pt"
    torch.save({"hf": dict(hf_cap), "ttml": dict(ttml_cap)}, diag_path)
    print(f"\n  All boundary tensors saved to {diag_path} for offline analysis")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\nIntermediate forward/backward comparison saved to {csv_path}")


def _compare_gradients(hf_grads, ttml_grads, mapping, csv_path=None):
    """Compare per-parameter gradients between HF and TTML."""
    rows = []
    cos_sim_sum = 0.0
    count = 0
    skipped = []

    for hf_name, ttml_name in sorted(mapping.items(), key=lambda kv: _layer_sort_key(kv[0])):
        if hf_name not in hf_grads:
            skipped.append((hf_name, "no HF grad"))
            continue
        if ttml_name not in ttml_grads:
            skipped.append((hf_name, "no TTML grad"))
            continue

        hf_grad = hf_grads[hf_name].float()
        ttml_grad = ttml_grads[ttml_name].float().squeeze()

        if hf_grad.shape != ttml_grad.shape:
            if hf_grad.dim() == 2 and ttml_grad.dim() == 2:
                r = min(hf_grad.shape[0], ttml_grad.shape[0])
                c = min(hf_grad.shape[1], ttml_grad.shape[1])
                hf_grad = hf_grad[:r, :c]
                ttml_grad = ttml_grad[:r, :c]
            elif hf_grad.dim() == 1 and ttml_grad.dim() == 1:
                d = min(hf_grad.shape[0], ttml_grad.shape[0])
                hf_grad = hf_grad[:d]
                ttml_grad = ttml_grad[:d]
            else:
                skipped.append((hf_name, f"shape: HF {hf_grad.shape} vs TTML {ttml_grad.shape}"))
                continue

        hf_flat = hf_grad.flatten()
        ttml_flat = ttml_grad.flatten()

        abs_diff = (hf_flat - ttml_flat).abs()
        ad_mean = abs_diff.mean().item()
        ad_max = abs_diff.max().item()

        hf_norm = hf_flat.norm()
        ttml_norm = ttml_flat.norm()
        cos_sim = (torch.dot(hf_flat, ttml_flat) / (hf_norm * ttml_norm + 1e-8)).item()
        cos_dist = 1.0 - cos_sim

        cos_sim_sum += cos_sim
        count += 1

        rows.append({
            "name": hf_name,
            "ad_mean": ad_mean,
            "ad_max": ad_max,
            "cos_sim": cos_sim,
            "cos_dist": cos_dist,
            "hf_norm": hf_norm.item(),
            "ttml_norm": ttml_norm.item(),
        })

    if not rows:
        if skipped:
            print(f"\n  Skipped {len(skipped)} parameters:")
            for name, reason in skipped:
                print(f"    {name}: {reason}")
        return

    name_w = max(max(len(r["name"]) for r in rows), 20)
    header = f"{'Parameter':>{name_w}}  {'CosSim':>10}  {'HF Norm':>12}  {'TTML Norm':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:>{name_w}}  "
            f"{r['cos_sim']:10.6f}  {r['hf_norm']:12.6f}  {r['ttml_norm']:12.6f}"
        )
    print("-" * len(header))
    if count > 0:
        print(f"{'Avg CosSim':>{name_w}}: {cos_sim_sum / count:.10f}")
        print(f"{'Parameters compared':>{name_w}}: {count}")
    if skipped:
        print(f"\n  Skipped {len(skipped)} parameters:")
        for name, reason in skipped:
            print(f"    {name}: {reason}")

    if csv_path:
        fieldnames = ["Parameter", "CosSim", "HF Norm", "TTML Norm"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    "Parameter": r["name"],
                    "CosSim": r["cos_sim"],
                    "HF Norm": r["hf_norm"],
                    "TTML Norm": r["ttml_norm"],
                })
        print(f"\nResults saved to {csv_path}")


def _has_partial_grad(ttml_name):
    """Return True if a replicated param has partial (per-device) gradients that
    need summing, False if the gradient is identical on all devices (take device 0).

    ColumnParallelLinear inserts broadcast(x) at its input, which all_reduces
    the input gradient in the backward pass.  This makes the gradient for every
    replicated tensor that feeds *only* through replicated operations identical
    on all devices.  The sole exceptions are norm_q / norm_k (and their
    "added" variants) whose RMSNorm backward runs on per-device head shards,
    producing genuinely partial gradients.
    """
    for tag in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        if tag in ttml_name:
            return True
    return False


def _extract_grad_distributed(tensor, device, shard_type, ttml_name=""):
    """Extract gradient from a ttml distributed tensor, gathering shards.

    For sharded params (col_w, col_b, row_w): concatenate shards along the
    appropriate dimension to reconstruct the full gradient.
    For replicated params (shard_type=None):
      - norm_q/k weights: each device holds a partial gradient from its head
        shard → sum across devices.
      - everything else (row_bias, time_text_embed, norm_out, proj_out, …):
        broadcast all_reduce makes the gradient identical on every device
        → take device 0.
    """
    if not tensor.is_grad_initialized():
        return None
    grad_tt = tensor.get_grad()
    if shard_type == "col_w":
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 2)
        return ttnn.to_torch(grad_tt, mesh_composer=composer).to(torch.float32)
    elif shard_type == "col_b":
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 3)
        return ttnn.to_torch(grad_tt, mesh_composer=composer).to(torch.float32)
    elif shard_type == "row_w":
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 3)
        return ttnn.to_torch(grad_tt, mesh_composer=composer).to(torch.float32)
    else:
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        grad_torch = ttnn.to_torch(grad_tt, mesh_composer=composer).to(torch.float32)
        if _has_partial_grad(ttml_name):
            return grad_torch.sum(dim=0, keepdim=True)
        return grad_torch[0:1]


def _un_refuse_proj_out_weight_grad(grad: torch.Tensor, attn_dim: int, mlp_dim: int, tp_size: int) -> torch.Tensor:
    """Undo the column reorder applied by _refuse_proj_out_weight so the gradient
    matches HF's original [attn_cols, mlp_cols] layout."""
    if tp_size <= 1:
        return grad
    rows = grad.shape[-2]
    attn_shard = attn_dim // tp_size
    mlp_shard = mlp_dim // tp_size
    g = grad.reshape(rows, tp_size, attn_shard + mlp_shard)
    g_attn = g[:, :, :attn_shard].reshape(rows, attn_dim)
    g_mlp = g[:, :, attn_shard:].reshape(rows, mlp_dim)
    return torch.cat([g_attn, g_mlp], dim=1)


def _fuse_qkv_grads(grads: dict, config: Flux1Config, tp_size: int) -> dict:
    """Fuse separate Q/K/V gradients into interleaved QKV, matching weight layout."""
    fused = dict(grads)
    dim = config.inner_dim
    shard = dim // tp_size

    def _interleave(q_w, k_w, v_w):
        chunks = []
        for i in range(tp_size):
            chunks.extend([
                q_w[i * shard : (i + 1) * shard],
                k_w[i * shard : (i + 1) * shard],
                v_w[i * shard : (i + 1) * shard],
            ])
        return torch.cat(chunks, dim=0)

    for i in range(config.num_layers):
        pfx = f"transformer_blocks.{i}.attn"
        for (q, k, v, out) in [
            ("to_q", "to_k", "to_v", "to_qkv"),
            ("add_q_proj", "add_k_proj", "add_v_proj", "add_qkv_proj"),
        ]:
            for suffix in ["weight", "bias"]:
                qn, kn, vn = f"{pfx}.{q}.{suffix}", f"{pfx}.{k}.{suffix}", f"{pfx}.{v}.{suffix}"
                if qn in fused:
                    fused[f"{pfx}.{out}.{suffix}"] = _interleave(
                        fused.pop(qn), fused.pop(kn), fused.pop(vn)
                    )

    for i in range(config.num_single_layers):
        pfx = f"single_transformer_blocks.{i}.attn"
        for suffix in ["weight", "bias"]:
            qn = f"{pfx}.to_q.{suffix}"
            kn = f"{pfx}.to_k.{suffix}"
            vn = f"{pfx}.to_v.{suffix}"
            if qn in fused:
                fused[f"{pfx}.to_qkv.{suffix}"] = _interleave(
                    fused.pop(qn), fused.pop(kn), fused.pop(vn)
                )

    return fused


def main():
    parser = argparse.ArgumentParser(description="Flux1: backward gradient comparison (HF vs ttml)")
    parser.add_argument("--checkpoint", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--prompt", type=str, default="A luxury sports car.")
    parser.add_argument("--mesh_shape", type=int, nargs=2, default=[1, 8], metavar=("ROWS", "COLS"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--intermediates_csv",
        type=str,
        default="flux1_gradients_intermediates.csv",
        help="CSV path for per-boundary forward/backward activation vs TTML comparison.",
    )
    parser.add_argument(
        "--param_gradients_csv",
        type=str,
        default="gradients_comparison.csv",
        help="CSV path for per-parameter gradient comparison.",
    )
    parser.add_argument(
        "--hf_dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help=(
            "Dtype for the HuggingFace reference model. Use ``float32`` for an "
            "accurate backward reference; ``bfloat16`` matches TTML's storage dtype "
            "but backward gradients deeper than ~4 blocks are dominated by bf16 "
            "rounding noise (fp32-vs-bf16 avg CosSim ≈ 0.51 already at 8+16 blocks, "
            "with some parameters going negative)."
        ),
    )
    args = parser.parse_args()

    dp_size, tp_size = args.mesh_shape
    checkpoint_name = args.checkpoint
    hf_dtype = torch.float32 if args.hf_dtype == "float32" else torch.bfloat16

    # ------------------------------------------------------------------
    # 1. Load HF transformer
    # ------------------------------------------------------------------
    from diffusers import FluxTransformer2DModel, AutoencoderKL

    print(f"Loading HF FluxTransformer2DModel: {checkpoint_name} (dtype={hf_dtype})")
    hf_transformer = FluxTransformer2DModel.from_pretrained(
        checkpoint_name, subfolder="transformer", torch_dtype=hf_dtype
    )
    hf_config_raw = hf_transformer.config
    config = create_flux1_config_from_hf(hf_config_raw)
    pos_embed_fn = hf_transformer.pos_embed
    hf_state_dict = hf_transformer.state_dict()

    # ------------------------------------------------------------------
    # 2. Encode prompts + prepare inputs
    # ------------------------------------------------------------------
    print("\nEncoding prompts ...")
    prompt_embeds, pooled_prompt_embeds = encode_prompts(
        args.prompt, checkpoint_name, config.joint_attention_dim
    )
    _, prompt_seq_len, _ = prompt_embeds.shape

    tmp_vae = AutoencoderKL.from_pretrained(checkpoint_name, subfolder="vae")
    vae_scale_factor = 2 ** len(tmp_vae.config.block_out_channels)
    num_channels_latents = config.in_channels // 4
    del tmp_vae

    latents_height = args.height // vae_scale_factor
    latents_width = args.width // vae_scale_factor

    torch.manual_seed(args.seed)
    latents_shape = [1, num_channels_latents, latents_height * 2, latents_width * 2]
    latents = _pack_latents(
        torch.randn(latents_shape, dtype=hf_dtype),
        1, num_channels_latents, latents_height, latents_width,
    )

    text_ids = torch.zeros([prompt_seq_len, 3])
    image_ids = _latent_image_ids(height=latents_height, width=latents_width)
    ids = torch.cat((text_ids, image_ids), dim=0)
    rope_cos, rope_sin = pos_embed_fn.forward(ids)

    spatial_rope_cos = rope_cos[prompt_seq_len:].to(hf_dtype)
    spatial_rope_sin = rope_sin[prompt_seq_len:].to(hf_dtype)
    prompt_rope_cos = rope_cos[:prompt_seq_len].to(hf_dtype)
    prompt_rope_sin = rope_sin[:prompt_seq_len].to(hf_dtype)

    timestep = torch.tensor([1.0], dtype=torch.float32)
    timestep_proj = _sinusoidal_proj(timestep)
    guidance = torch.full([1], fill_value=3.5, dtype=hf_dtype) if config.guidance_embeds else None
    guidance_proj = _sinusoidal_proj(guidance * 1000.0) if guidance is not None else None

    target = torch.randn_like(latents)

    # ------------------------------------------------------------------
    # 3. Setup device + create ttml model
    # ------------------------------------------------------------------
    ctx, device = setup_device(dp_size, tp_size, seed=args.seed)
    shard_dim = 1 if tp_size > 1 else None

    print(f"\nCreating ttml model (TP={tp_size}, use_checkpoint=False) ...")
    #config.num_layers = 8
    #config.num_single_layers = 8
    with empty_init():
        ttml_model = DistributedFlux1Transformer(config, shard_dim=shard_dim, use_checkpoint=True)

    print("Loading weights ...")
    load_weights_from_hf_distributed(ttml_model, dict(hf_state_dict), config, shard_dim=shard_dim)

    ttml_boundary_capture: dict = {}
    _patch_ttml_forward_with_boundaries(ttml_model, device, ttml_boundary_capture)

    # ------------------------------------------------------------------
    # 4. TTML forward + backward
    # ------------------------------------------------------------------
    print("\n[TTML] Forward + backward (with block-boundary capture) ...")
    ttml_ctx = ttml.autograd.AutoContext.get_instance()
    ttml_ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    ttml_model.train()

    def _to_ttml(arr, layout=ttnn.Layout.TILE):
        return ttml.autograd.Tensor.from_numpy(arr, layout, ttnn.bfloat16)

    tt_spatial = _to_ttml(latents.float().numpy().reshape(1, 1, *latents.shape[1:]))
    tt_prompt = _to_ttml(prompt_embeds.unsqueeze(0).float().numpy())
    tt_pooled = _to_ttml(pooled_prompt_embeds.unsqueeze(0).unsqueeze(0).float().numpy())
    tt_timestep = _to_ttml(timestep_proj.float().numpy().reshape(1, 1, 1, -1))
    tt_guidance = _to_ttml(guidance_proj.float().numpy().reshape(1, 1, 1, -1)) if guidance_proj is not None else None

    tt_rope_cos_s = _to_ttml(spatial_rope_cos.float().numpy().reshape(1, 1, *spatial_rope_cos.shape))
    tt_rope_sin_s = _to_ttml(spatial_rope_sin.float().numpy().reshape(1, 1, *spatial_rope_sin.shape))
    tt_rope_cos_p = _to_ttml(prompt_rope_cos.float().numpy().reshape(1, 1, *prompt_rope_cos.shape))
    tt_rope_sin_p = _to_ttml(prompt_rope_sin.float().numpy().reshape(1, 1, *prompt_rope_sin.shape))

    t0 = time.time()
    ttml_out = ttml_model(
        tt_spatial, tt_prompt, tt_timestep, tt_guidance, tt_pooled,
        tt_rope_cos_s, tt_rope_sin_s, tt_rope_cos_p, tt_rope_sin_p,
    )

    target_4d = target.float().numpy().reshape(1, 1, *target.shape[1:])
    tt_target = _to_ttml(target_4d)
    ttml_loss = ttml.ops.loss.mse_loss(ttml_out, tt_target)
    ttml_loss = ttml.ops.binary.mul(ttml_loss, 100)

    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    loss_val = ttnn.to_torch(ttml_loss.get_value(), mesh_composer=composer).to(torch.float32)[0].item()
    print(f"  Loss: {loss_val:.6f}")

    ttml_loss.backward(False)
    _finalize_ttml_boundary_grads(ttml_boundary_capture, device)
    ttml_time = time.time() - t0
    print(f"  Backward complete ({ttml_time:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Extract ttml gradients
    # ------------------------------------------------------------------
    ttml_params = ttml_model.parameters()
    root = next(iter(ttml_params)).split("/")[0]
    weight_mapping = _build_weight_mapping(config, root, tp_size)

    grad_mapping = {}
    ttml_grads = {}
    for hf_name, (ttml_name, shard_type) in weight_mapping.items():
        grad_mapping[hf_name] = ttml_name
        if ttml_name not in ttml_params:
            continue
        tensor = ttml_params[ttml_name]
        grad = _extract_grad_distributed(tensor, device, shard_type, ttml_name)
        if grad is not None:
            if "proj_out/weight" in ttml_name and "single_transformer_blocks" in ttml_name:
                grad = _un_refuse_proj_out_weight_grad(
                    grad.squeeze(0).squeeze(0), config.inner_dim, config.mlp_hidden_dim, tp_size
                ).unsqueeze(0).unsqueeze(0)
            ttml_grads[ttml_name] = grad

    print(f"  Gradients collected: {len(ttml_grads)}")

    ttml_ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    ttml_ctx.reset_graph()

    # ------------------------------------------------------------------
    # 6. HF forward + backward
    # ------------------------------------------------------------------
    print("\n[HF] Forward + backward (with block-boundary hooks) ...")
    hf_transformer.transformer_blocks = hf_transformer.transformer_blocks[:config.num_layers]
    hf_transformer.single_transformer_blocks = hf_transformer.single_transformer_blocks[:config.num_single_layers]

    hf_boundary = HFBoundaryCapture()
    hf_boundary.register(hf_transformer, config.num_layers, config.num_single_layers)

    hf_transformer.train()
    hf_transformer.zero_grad()

    for p in hf_transformer.parameters():
        p.requires_grad_(True)

    t0 = time.time()

    hf_out = hf_transformer(
        hidden_states=latents.to(hf_dtype),
        encoder_hidden_states=prompt_embeds.to(hf_dtype),
        pooled_projections=pooled_prompt_embeds.to(hf_dtype),
        timestep=timestep / 1000.0,
        img_ids=image_ids,
        txt_ids=text_ids,
        guidance=guidance.to(hf_dtype) if guidance is not None else None,
    ).sample

    hf_loss = F.mse_loss(hf_out.float(), target.float())
    hf_loss = hf_loss * 100
    hf_loss.backward()
    hf_time = time.time() - t0
    print(f"  Loss: {hf_loss.item():.6f}  ({hf_time:.1f}s)")
    print(f"  Output shape: {list(hf_out.shape)}")

    hf_grads = {}
    for name, param in hf_transformer.named_parameters():
        if param.grad is not None:
            hf_grads[name] = param.grad.float().clone()
    print(f"  Gradients collected: {len(hf_grads)}")

    hf_grads = _fuse_qkv_grads(hf_grads, config, tp_size)

    hf_boundary.remove_hooks()

    boundary_names = _all_boundary_names(config.num_layers, config.num_single_layers)

    # ------------------------------------------------------------------
    # 7. Compare block-boundary activations / backward grads
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Intermediate tensors at block boundaries (HF vs TTML)")
    print("=" * 70)
    _compare_intermediates(
        hf_boundary.capture,
        ttml_boundary_capture,
        boundary_names,
        csv_path=args.intermediates_csv,
    )

    # ------------------------------------------------------------------
    # 8. Compare per-parameter gradients
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Per-parameter gradient comparison (HF vs TTML)")
    print("=" * 70)
    _compare_gradients(hf_grads, ttml_grads, grad_mapping, csv_path=args.param_gradients_csv)

    ctx.close_device()


if __name__ == "__main__":
    main()
