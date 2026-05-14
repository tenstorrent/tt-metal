# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-decode-debug-3 — per-INDIVIDUAL-layer hooked PCC (single forward pass).

Captures the residual stream AFTER EACH OF the N layers in a single decode
forward pass, both for the TT model and the CPU reference.  Prints a per-layer
PCC table so the layer index where divergence becomes large can be identified.

This is the diagnostic tool for the exponential per-layer error amplification
observed in 64L decode (4L 0.9996 → 64L 0.30).  Pure precision compounding
would be √N noise; an exponential trajectory implies a per-layer multiplicative
error source that exists only in the decode forward path.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_eager_hooked_pcc.py \\
            -v -s
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_T_PREFILL = 128
_N_LAYERS_DEFAULT = 64


@pytest.fixture(scope="function")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _load_state_dict_for_layers(snapshot_dir: pathlib.Path, layer_indices: list[int]) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
    ] + [f"model.language_model.layers.{i}." for i in layer_indices]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() < 2:
        return float("nan")
    # Guard against constant tensors (corrcoef returns nan).
    if a.std().item() < 1e-12 or b.std().item() < 1e-12:
        return float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_tt_model(mesh, state_dict, n_layers: int):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = n_layers
    # Use the canonical [lin, lin, lin, full] * 16 pattern truncated to n_layers
    pattern_full = args.linear_attention_pattern
    if pattern_full is None or len(pattern_full) < n_layers:
        pattern_full = ["linear_attention"] * 3 + ["full_attention"]
        pattern_full = pattern_full * ((n_layers + 3) // 4)
    args.linear_attention_pattern = pattern_full[:n_layers]
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )
    return model, args


def _build_partial_rope_cos_sin_tt(mesh, positions: torch.Tensor):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    cos_tt = ttnn.from_torch(
        cos_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sin_tt = ttnn.from_torch(
        sin_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return cos_tt, sin_tt


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args):
    B, T, H = t.shape
    t_4d = t.reshape(1, 1, T, H)
    return ttnn.from_torch(
        t_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _gather_col_sharded_to_full_torch(tt_tensor, mesh, args, T: int) -> torch.Tensor:
    """Bring col-sharded [B, 1, T, H/4] back to torch [B, T, H]."""
    out = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
    )
    # dim 0 has the row-shards concatenated; we use a fresh copy of row 0.
    out = out[0:1]
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dim() == 3:
        out = out[:, :T, :]
    return out


def _cpu_reference_capture_per_layer(state_dict_hf: dict, layer_indices: list[int], x_full: torch.Tensor):
    """Run the CPU reference forward and capture hidden state AFTER each layer.

    Returns a list ``hidden_per_layer[layer_idx_in_indices] = torch.Tensor[B, T, H]``
    of length len(layer_indices).  The hidden at index i is the residual stream
    output of layer i (before the next layer's input projection).
    """
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer, Qwen36Config, build_mrope_cos_sin

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)

    T = x_full.shape[1]
    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    causal_mask = torch.zeros(1, 1, T, T)
    causal_mask = causal_mask.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))

    hidden = x_full.float()
    captures: list[torch.Tensor] = []
    for layer_idx in layer_indices:
        layer = HybridDecoderLayer(config, layer_idx).eval()
        pfx = f"model.language_model.layers.{layer_idx}."
        layer_sd: dict[str, torch.Tensor] = {}
        for k, v in state_dict_hf.items():
            if k.startswith(pfx):
                short = k[len(pfx) :]
                if short.startswith("self_attn."):
                    layer_sd["attention." + short[len("self_attn.") :]] = v.float()
                elif short.startswith("linear_attn."):
                    layer_sd["attention." + short[len("linear_attn.") :]] = v.float()
                else:
                    layer_sd[short] = v.float()
        layer.load_state_dict(layer_sd, strict=False)
        with torch.no_grad():
            hidden, _, _, _ = layer(hidden, cos, sin, attention_mask=causal_mask)
        captures.append(hidden.detach().clone())
    return captures


@pytest.mark.hardware
@pytest.mark.parametrize("n_layers", [_N_LAYERS_DEFAULT])
def test_qwen36_hooked_per_layer_decode_pcc_64(bh_glx_mesh, n_layers):
    _run_hooked(bh_glx_mesh, n_layers)


@pytest.mark.hardware
@pytest.mark.parametrize("n_layers", [16])
def test_qwen36_hooked_per_layer_decode_pcc_16(bh_glx_mesh, n_layers):
    _run_hooked(bh_glx_mesh, n_layers)


@pytest.mark.hardware
@pytest.mark.parametrize("n_layers", [32])
def test_qwen36_hooked_per_layer_decode_pcc_32(bh_glx_mesh, n_layers):
    _run_hooked(bh_glx_mesh, n_layers)


def _run_hooked(bh_glx_mesh, n_layers):
    """Capture per-layer residual hidden state in a single 64L decode forward.

    Sequence:
      1. Build N-layer TT model, run prefill T=128.
      2. CPU reference: full-sequence forward (T=129) with per-layer capture.
      3. TT decode: run one decode step (T=1) and hook each layer to capture
         its output residual (col-sharded H/4) → gather to full-H → store.
      4. Compute PCC(tt_after_layer_i, ref_after_layer_i) for i = 0..N-1.
    """
    print(f"\n=== [hooked-{n_layers}L] starting ===")
    layer_indices = list(range(n_layers))
    state_dict = _load_state_dict_for_layers(_SNAPSHOT, layer_indices)
    print(f"[hooked-{n_layers}L] loaded {len(state_dict)} weights")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, n_layers)
    print(f"[hooked-{n_layers}L] TT model built")

    torch.manual_seed(44)
    T_full = _T_PREFILL + 1
    x_full = torch.randn(1, T_full, args.dim, dtype=torch.bfloat16)

    # CPU reference: capture hidden AFTER each layer over the full sequence.
    print(f"[hooked-{n_layers}L] running CPU reference (full {T_full}-tok forward)...")
    ref_captures = _cpu_reference_capture_per_layer(state_dict, layer_indices, x_full)
    # The "decode step" hidden is at token index _T_PREFILL of each capture.
    decode_pos = _T_PREFILL
    ref_decode_per_layer: list[torch.Tensor] = [cap[:, decode_pos : decode_pos + 1, :].clone() for cap in ref_captures]
    print(f"[hooked-{n_layers}L] CPU ref done; {len(ref_decode_per_layer)} per-layer hiddens captured")

    # ---------- TT PREFILL on first _T_PREFILL tokens (seeds caches) ----------
    x_prefill = x_full[:, :_T_PREFILL, :]
    x_tt = _send_col_sharded_hidden(x_prefill, bh_glx_mesh, args)
    cos_tt_prefill, sin_tt_prefill = _build_partial_rope_cos_sin_tt(
        bh_glx_mesh, torch.arange(_T_PREFILL, dtype=torch.long)
    )
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    _ = model.forward(
        x_tt,
        current_pos=None,
        rot_mats=(cos_tt_prefill, sin_tt_prefill),
        user_id=0,
        mode="prefill",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=chunk_start_idx_tt,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )
    print(f"[hooked-{n_layers}L] prefill done; cache seeded")

    # ---------- HOOK each layer's forward to capture residual on decode ----------
    tt_decode_hiddens: list[torch.Tensor] = []

    original_forwards: list = []
    for i, layer in enumerate(model.layers):
        original_forwards.append(layer.forward)

    def make_hooked_forward(idx, orig_forward):
        def hooked(
            x,
            h,
            current_pos,
            rot_mats=None,
            user_id=0,
            mode="decode",
            page_table=None,
            chunk_page_table=None,
            chunk_start_idx=None,
            chunk_start_idx_tensor=None,
            kv_cache=None,
            batch_size=1,
        ):
            out_x, out_h = orig_forward(
                x,
                h,
                current_pos,
                rot_mats=rot_mats,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
            if mode == "decode":
                # out_x is col-sharded [B, 1, T=1, H/4]. Pull to torch full-H.
                # Note: ttnn.to_torch performs a device->host copy which is slow
                # but acceptable for a one-shot diagnostic over 64 layers.
                t = _gather_col_sharded_to_full_torch(out_x, bh_glx_mesh, args, T=1)
                # t is [B, 1, H] after squeeze. Detach + clone for safety.
                tt_decode_hiddens.append(t.detach().clone().float())
            return out_x, out_h

        return hooked

    for i, layer in enumerate(model.layers):
        layer.forward = make_hooked_forward(i, original_forwards[i])

    # ---------- TT DECODE step ----------
    x_decode = x_full[:, _T_PREFILL : _T_PREFILL + 1, :]
    x_decode_tt = _send_col_sharded_hidden(x_decode, bh_glx_mesh, args)
    cos_tt_decode, sin_tt_decode = _build_partial_rope_cos_sin_tt(
        bh_glx_mesh, torch.tensor([_T_PREFILL], dtype=torch.long)
    )
    tt_out = model.forward(
        x_decode_tt,
        current_pos=_T_PREFILL,
        rot_mats=(cos_tt_decode, sin_tt_decode),
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )

    # Restore.
    for i, layer in enumerate(model.layers):
        layer.forward = original_forwards[i]

    assert len(tt_decode_hiddens) == n_layers, f"expected {n_layers} hooked captures, got {len(tt_decode_hiddens)}"

    # ---------- Compute per-layer PCC ----------
    print(f"\n[hooked-{n_layers}L] === PER-LAYER PCC (residual stream after layer i) ===")
    print(
        f"{'lyr':>3}  {'pcc':>10}  {'Δpcc':>10}  {'kind':>16}  "
        f"{'tt_std':>10}  {'ref_std':>10}  {'rel_err':>10}  {'tt_max':>10}"
    )
    pcc_per_layer: list[float] = []
    prev_pcc = None
    for i in range(n_layers):
        tt_hi = tt_decode_hiddens[i].reshape(-1)
        ref_hi = ref_decode_per_layer[i].reshape(-1)
        pcc_i = _pcc(tt_hi, ref_hi)
        pcc_per_layer.append(pcc_i)
        d_pcc = (pcc_i - prev_pcc) if prev_pcc is not None else 0.0
        prev_pcc = pcc_i
        kind = args.linear_attention_pattern[i] if i < len(args.linear_attention_pattern) else "?"
        ref_std = ref_hi.std().item()
        tt_std = tt_hi.std().item()
        rel_err = (tt_hi - ref_hi).norm().item() / max(ref_hi.norm().item(), 1e-12)
        tt_max = tt_hi.abs().max().item()
        print(
            f"{i:>3}  {pcc_i:>10.6f}  {d_pcc:>10.6f}  {kind:>16}  "
            f"{tt_std:>10.4f}  {ref_std:>10.4f}  {rel_err:>10.4f}  {tt_max:>10.4f}"
        )

    # Acceptance: PCC at last layer must be > 0.99 to pass.
    final_pcc = pcc_per_layer[-1]
    print(f"\n[hooked-{n_layers}L] FINAL LAYER PCC = {final_pcc:.6f}")
    # First layer where PCC drops below 0.99
    first_below = next((i for i, p in enumerate(pcc_per_layer) if p < 0.99), None)
    print(f"[hooked-{n_layers}L] first layer with PCC < 0.99: {first_below}")

    # log only; sweep should always complete
    assert final_pcc > -2.0
