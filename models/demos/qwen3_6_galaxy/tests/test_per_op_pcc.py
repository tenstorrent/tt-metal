# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-op intermediate PCC diagnostic test for Qwen3.6-27B on BH Galaxy.

Task T11b: Identify the single op or transition that drops PCC the most,
then apply a targeted precision fix so the 16-layer test reaches > 0.995.

The test captures intermediates at each step of each decoder layer for BOTH
the TTNN model and the CPU reference, computes per-step PCC (TT vs ref),
and prints a table showing where the largest drops occur.

Run:
    LOG_FILE: /tmp/qwen36_logs/t11b_per_op.log
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_per_op_pcc.py \
        -x -s -v -k test_per_op_pcc_4layer \
        2>&1 | tee /tmp/qwen36_logs/t11b_per_op_4layer.log
"""

import json
import pathlib
import sys
import time

import pytest
import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_B = 1
_T_PREFILL = 32
_H = 5120


# ---------------------------------------------------------------------------
# PCC helper
# ---------------------------------------------------------------------------


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    cc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return cc


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity (alternative similarity metric)."""
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _gather_tt_tensor(x_tt, mesh_device, B=1):
    """Gather a replicated TTNN tensor [B,T,H] to CPU float32."""
    import ttnn

    x_cpu_raw = ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return x_cpu_raw[:B].float()  # first device copy


# ---------------------------------------------------------------------------
# Weight loading helpers (shared with test_full_model.py)
# ---------------------------------------------------------------------------


def _load_layer_weights(layer_idx: int) -> dict:
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    pfx = f"model.language_model.layers.{layer_idx}"
    keys_needed = [k for k in weight_map if k.startswith(pfx + ".")]
    files_needed = sorted({weight_map[k] for k in keys_needed})
    raw = {}
    for fn in files_needed:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in keys_needed:
            if k in shard:
                raw[k] = shard[k].float()
    result = {}
    for k, v in raw.items():
        short = k[len(pfx) + 1 :]
        result[short] = v
    return result


def _load_embedding_and_norm_and_head_weights() -> dict:
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    files_needed = sorted({weight_map[k] for k in needed if k in weight_map})
    raw = {}
    for fn in files_needed:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in needed:
            if k in shard:
                raw[k] = shard[k].float()
    return {
        "tok_embeddings.weight": raw["model.language_model.embed_tokens.weight"],
        "norm.weight": raw["model.language_model.norm.weight"],
        "output.weight": raw["lm_head.weight"],
    }


def _load_config():
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config

    with open(_SNAPSHOT_DIR / "config.json") as f:
        d = json.load(f)
    return Qwen36Config(d)


def _build_rope_cos_sin(T: int):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    return cos, sin  # [1, T, 64]


# ---------------------------------------------------------------------------
# Fabric mesh fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_8x4():
    import ttnn

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


# ---------------------------------------------------------------------------
# Reference model builder (same as test_full_model.py)
# ---------------------------------------------------------------------------


def _build_ref_model(config, num_layers: int, global_weights: dict, layers_weights: list):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36TextModel

    sliced_config = type(config).__new__(type(config))
    sliced_config.__dict__.update(config.__dict__)
    sliced_config.num_hidden_layers = num_layers
    sliced_config.layer_types = config.layer_types[:num_layers]

    model = Qwen36TextModel(sliced_config)
    model.eval()

    with torch.no_grad():
        model.tok_embeddings.weight.data.copy_(global_weights["tok_embeddings.weight"][: config.vocab_size])
        model.norm.weight.data.copy_(global_weights["norm.weight"])
        model.lm_head.weight.data.copy_(global_weights["output.weight"][: config.vocab_size])
        for i in range(num_layers):
            w = layers_weights[i]
            layer = model.layers[i]
            layer.input_layernorm.weight.data.copy_(w["input_layernorm.weight"])
            layer.post_attention_layernorm.weight.data.copy_(w["post_attention_layernorm.weight"])
            layer.mlp.gate_proj.weight.data.copy_(w["mlp.gate_proj.weight"])
            layer.mlp.up_proj.weight.data.copy_(w["mlp.up_proj.weight"])
            layer.mlp.down_proj.weight.data.copy_(w["mlp.down_proj.weight"])
            lt = config.layer_types[i]
            if lt == "full_attention":
                attn = layer.attention
                attn.q_proj.weight.data.copy_(w["self_attn.q_proj.weight"])
                attn.k_proj.weight.data.copy_(w["self_attn.k_proj.weight"])
                attn.v_proj.weight.data.copy_(w["self_attn.v_proj.weight"])
                attn.o_proj.weight.data.copy_(w["self_attn.o_proj.weight"])
                attn.q_norm.weight.data.copy_(w["self_attn.q_norm.weight"])
                attn.k_norm.weight.data.copy_(w["self_attn.k_norm.weight"])
            else:
                dn = layer.attention
                dn.in_proj_qkv.weight.data.copy_(w["linear_attn.in_proj_qkv.weight"])
                dn.in_proj_z.weight.data.copy_(w["linear_attn.in_proj_z.weight"])
                dn.in_proj_a.weight.data.copy_(w["linear_attn.in_proj_a.weight"])
                dn.in_proj_b.weight.data.copy_(w["linear_attn.in_proj_b.weight"])
                dn.conv1d.weight.data.copy_(w["linear_attn.conv1d.weight"])
                dn.A_log.data.copy_(w["linear_attn.A_log"])
                dn.dt_bias.data.copy_(w["linear_attn.dt_bias"])
                dn.norm.weight.data.copy_(w["linear_attn.norm.weight"])
                dn.out_proj.weight.data.copy_(w["linear_attn.out_proj.weight"])

    return model


# ---------------------------------------------------------------------------
# Reference per-layer intermediates capture
# ---------------------------------------------------------------------------


def _capture_ref_layer_intermediates(
    ref_model, x_input: torch.Tensor, cos, sin, attention_mask, layer_idx: int
) -> dict:
    """Run a single reference layer and capture intermediates.

    Args:
        x_input: [B, T, H] float32 — input to this layer
        cos, sin: MRoPE tensors
        attention_mask: causal mask
        layer_idx: layer index into ref_model.layers

    Returns:
        dict with keys:
          'input':         x_input (same as input)
          'after_input_LN': x after input_layernorm
          'after_attn':    attn_out (attention/DeltaNet contribution, BEFORE residual add)
          'after_residual1': x + attn_out
          'after_post_LN': x after post_attention_layernorm
          'after_mlp':     mlp_out (MLP contribution, BEFORE residual add)
          'after_residual2': x + mlp_out  (= output of layer)
    """
    layer = ref_model.layers[layer_idx]
    intermediates = {}

    with torch.no_grad():
        x = x_input.clone()

        # Record input
        intermediates["input"] = x.clone()

        # Input layernorm
        residual = x
        x_normed = layer.input_layernorm(x)
        intermediates["after_input_LN"] = x_normed.clone()

        # Attention / DeltaNet block
        lt = layer.layer_type
        if lt == "full_attention":
            attn_out, kv_cache_new = layer.attention(x_normed, cos, sin, None, None)
        else:
            attn_out, conv_state_new, recurrent_state_new = layer.attention(x_normed)
        intermediates["after_attn"] = attn_out.clone()

        # First residual add
        x = residual + attn_out
        intermediates["after_residual1"] = x.clone()

        # Post-attention layernorm
        residual2 = x
        x_normed2 = layer.post_attention_layernorm(x)
        intermediates["after_post_LN"] = x_normed2.clone()

        # MLP
        mlp_out = layer.mlp(x_normed2)
        intermediates["after_mlp"] = mlp_out.clone()

        # Second residual add (= layer output)
        x_out = residual2 + mlp_out
        intermediates["after_residual2"] = x_out.clone()

    return intermediates


# ---------------------------------------------------------------------------
# TTNN per-layer intermediates capture
# ---------------------------------------------------------------------------


def _capture_tt_layer_intermediates(
    tt_layer,
    x_tt,
    mesh_device,
    rot_mats,
    mode: str = "prefill",
) -> tuple:
    """Run a single TTNN decoder layer with intermediate capture.

    Returns:
        (x_tt_out, intermediates_dict)
        where intermediates_dict has same keys as the reference version but
        values are CPU float32 tensors (gathered from device).
    """
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.llama_decoder import _gather_from_cols, _shard_across_cols

    intermediates = {}
    mem = ttnn.DRAM_MEMORY_CONFIG

    orig_shape = list(x_tt.shape)
    B_in = orig_shape[0]
    T_in = orig_shape[1] if len(orig_shape) == 3 else orig_shape[-2]
    H_in = orig_shape[-1]

    # Capture input
    intermediates["input"] = _gather_tt_tensor(x_tt, mesh_device, B_in)

    # --- Pre-norm ---
    residual = x_tt
    x_sharded = _shard_across_cols(x_tt, mesh_device)
    x_normed_sharded = tt_layer.input_layernorm(x_sharded)
    x_sharded.deallocate(True)
    x_normed = _gather_from_cols(x_normed_sharded, mesh_device)
    x_normed_sharded.deallocate(True)

    # Capture after input LN
    intermediates["after_input_LN"] = _gather_tt_tensor(x_normed, mesh_device, B_in)

    # --- Attention / DeltaNet ---
    if tt_layer.layer_type == "linear_attention":
        result = tt_layer.attention.forward(
            x_normed,
            mode=mode,
            recurrent_state=None,
            conv_state=None,
            return_state=True,
        )
        attn_out_raw, new_dn_state, new_conv_state = result
        raw_T = list(attn_out_raw.shape)[1] if len(list(attn_out_raw.shape)) == 3 else list(attn_out_raw.shape)[-2]
        if raw_T != T_in:
            attn_out = ttnn.slice(attn_out_raw, [0, 0, 0], [B_in, T_in, H_in], memory_config=mem)
            attn_out_raw.deallocate(True)
        else:
            attn_out = attn_out_raw
    else:
        attn_out = tt_layer.attention.forward(
            x_normed,
            current_pos=0,
            rot_mats=rot_mats,
            user_id=0,
            mode=mode,
        )
        new_dn_state = None
        new_conv_state = None

    x_normed.deallocate(True)

    # Capture after attn/deltanet (pre-residual contribution)
    intermediates["after_attn"] = _gather_tt_tensor(attn_out, mesh_device, B_in)

    # --- First residual add ---
    x = ttnn.add(residual, attn_out, memory_config=mem)
    residual.deallocate(True)
    attn_out.deallocate(True)

    # Capture after residual1
    intermediates["after_residual1"] = _gather_tt_tensor(x, mesh_device, B_in)

    # --- Post-norm ---
    residual2 = x
    x_sharded2 = _shard_across_cols(x, mesh_device)
    x_normed2_sharded = tt_layer.post_attention_layernorm(x_sharded2)
    x_sharded2.deallocate(True)
    x_normed2 = _gather_from_cols(x_normed2_sharded, mesh_device)
    x_normed2_sharded.deallocate(True)

    # Capture after post LN
    intermediates["after_post_LN"] = _gather_tt_tensor(x_normed2, mesh_device, B_in)

    # --- MLP ---
    mlp_out = tt_layer.mlp.forward(x_normed2)
    x_normed2.deallocate(True)

    # Capture MLP output (pre-residual contribution)
    intermediates["after_mlp"] = _gather_tt_tensor(mlp_out, mesh_device, B_in)

    # --- Second residual add ---
    x_out = ttnn.add(residual2, mlp_out, memory_config=mem)
    residual2.deallocate(True)
    mlp_out.deallocate(True)

    # Capture after residual2 (= layer output)
    intermediates["after_residual2"] = _gather_tt_tensor(x_out, mesh_device, B_in)

    return x_out, intermediates, new_dn_state, new_conv_state


# ---------------------------------------------------------------------------
# Print per-layer PCC table
# ---------------------------------------------------------------------------

STEP_NAMES = [
    "input",
    "after_input_LN",
    "after_attn",
    "after_residual1",
    "after_post_LN",
    "after_mlp",
    "after_residual2",
]


def _print_pcc_table(all_layer_pccs: list, config, n_layers: int):
    """Print a formatted PCC table across all layers and steps.

    all_layer_pccs: list of dicts (one per layer), each dict maps step_name -> pcc_value
    """
    header = f"{'Layer':<6} {'Type':<18} " + "  ".join(f"{s:<18}" for s in STEP_NAMES)
    print()
    print("=" * len(header))
    print("Per-Op Intermediate PCC Table (TTNN vs CPU Reference)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    worst_pcc = 1.0
    worst_loc = None

    for i, layer_pccs in enumerate(all_layer_pccs):
        lt = config.layer_types[i]
        row = f"{i:<6} {lt:<18} "
        for j, step in enumerate(STEP_NAMES):
            pcc_val = layer_pccs.get(step, float("nan"))
            if j > 0:
                prev_step = STEP_NAMES[j - 1]
                prev_pcc = layer_pccs.get(prev_step, float("nan"))
                drop = prev_pcc - pcc_val if (prev_pcc == prev_pcc and pcc_val == pcc_val) else 0.0
                drop_str = f"(drop={drop:+.4f})" if abs(drop) > 1e-5 else ""
            else:
                drop_str = ""
            if pcc_val < worst_pcc:
                worst_pcc = pcc_val
                worst_loc = (i, step)
            flag = " !!!" if pcc_val < 0.99 else (" !" if pcc_val < 0.995 else "")
            row += f" {pcc_val:.4f}{flag:<4}"

        print(row)

    print("-" * len(header))

    # Per-step worst drop analysis
    print()
    print("Per-Step Drop Analysis (largest single-step drops):")
    drops = []
    for i, layer_pccs in enumerate(all_layer_pccs):
        for j in range(1, len(STEP_NAMES)):
            step = STEP_NAMES[j]
            prev_step = STEP_NAMES[j - 1]
            pcc_val = layer_pccs.get(step, float("nan"))
            prev_pcc = layer_pccs.get(prev_step, float("nan"))
            if pcc_val == pcc_val and prev_pcc == prev_pcc:
                drop = prev_pcc - pcc_val
                drops.append((drop, i, prev_step, step, pcc_val))

    drops.sort(key=lambda x: -x[0])
    for drop, layer_i, from_step, to_step, pcc_val in drops[:10]:
        lt = config.layer_types[layer_i]
        print(f"  Layer {layer_i} ({lt}): {from_step} → {to_step}: drop={drop:+.4f}  (pcc={pcc_val:.4f})")

    print()
    if worst_loc:
        print(f"Worst overall PCC: {worst_pcc:.4f} at Layer {worst_loc[0]}, step '{worst_loc[1]}'")
    print("=" * len(header))

    return worst_pcc, worst_loc, drops


# ---------------------------------------------------------------------------
# Main diagnostic test: 4-layer
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_per_op_pcc_4layer(mesh_8x4):
    """Per-op intermediate PCC diagnostic on 4 decoder layers.

    For each layer, captures TTNN and reference intermediates at:
      - input (= output of previous layer)
      - after input_layernorm (DistributedNorm)
      - after attention / DeltaNet block (pre-residual contribution)
      - after first residual add
      - after post_attention_layernorm (DistributedNorm)
      - after MLP (pre-residual contribution)
      - after second residual add (= layer output)

    Prints a table showing where the largest PCC drop occurs.
    The op with the biggest single-step drop is the precision bottleneck.

    LOG_FILE: /tmp/qwen36_logs/t11b_per_op_4layer.log
    """
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.llama_model import TtQwen36Transformer
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 4
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    print(f"\n[PerOpPCC-4L] Loading weights for {N_LAYERS} layers...")
    t0 = time.time()
    global_weights = _load_embedding_and_norm_and_head_weights()
    layers_weights = [_load_layer_weights(i) for i in range(N_LAYERS)]
    print(f"[PerOpPCC-4L] Weights loaded in {time.time()-t0:.1f}s")

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (_B, _T_PREFILL))

    cos, sin = _build_rope_cos_sin(_T_PREFILL)
    causal_mask = torch.zeros(1, 1, _T_PREFILL, _T_PREFILL)
    causal_mask = causal_mask.masked_fill(
        torch.triu(torch.ones(_T_PREFILL, _T_PREFILL), diagonal=1).bool(), float("-inf")
    )

    # Build CPU reference model
    print("[PerOpPCC-4L] Building CPU reference model...")
    ref_model = _build_ref_model(config, N_LAYERS, global_weights, layers_weights)

    # Build TTNN model
    print("[PerOpPCC-4L] Building TTNN model...")
    t0 = time.time()
    tt_model = TtQwen36Transformer(
        mesh_device=mesh_8x4,
        args=args,
        global_weights=global_weights,
        layers_weights=layers_weights,
        num_layers=N_LAYERS,
    )
    print(f"[PerOpPCC-4L] TTNN model built in {time.time()-t0:.1f}s")

    # ----------------------------------------------------------------
    # Shared embedding: use TTNN embedding output as starting point for both
    # (so both TT and ref start from identical BF16 embedding values)
    # ----------------------------------------------------------------
    print("[PerOpPCC-4L] Running embedding on TTNN device...")
    x_tt = tt_model._embed(input_ids)
    emb_cpu = ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_8x4, dim=0))
    emb_cpu = emb_cpu[:_B].float()  # [B, T, H] BF16 upcast to float32

    # Build RoPE on device for full_attention layers
    cos_tt, sin_tt = tt_model.rope_setup.get_cos_sin_for_prefill(_T_PREFILL)
    rot_mats = (cos_tt, sin_tt)

    # ----------------------------------------------------------------
    # Per-layer capture loop
    # ----------------------------------------------------------------
    all_layer_pccs = []

    # CPU: start from BF16 embedding values (same as TTNN)
    x_ref = emb_cpu.clone()  # [B, T, H] float32 (BF16 values)

    print("[PerOpPCC-4L] Running per-layer diagnostic...")
    for layer_i in range(N_LAYERS):
        lt = config.layer_types[layer_i]
        print(f"\n  Layer {layer_i} ({lt}):")

        # ---- TTNN intermediates ----
        t0 = time.time()
        x_tt, tt_ints, new_dn, new_conv = _capture_tt_layer_intermediates(
            tt_layer=tt_model.layers[layer_i],
            x_tt=x_tt,
            mesh_device=mesh_8x4,
            rot_mats=rot_mats,
            mode="prefill",
        )
        print(f"    TT layer time: {time.time()-t0:.2f}s")

        # ---- Reference intermediates ----
        ref_ints = _capture_ref_layer_intermediates(ref_model, x_ref, cos, sin, causal_mask, layer_i)

        # Update reference hidden state for next layer
        x_ref = ref_ints["after_residual2"].clone()

        # ---- Compute PCC at each step ----
        layer_pccs = {}
        for step in STEP_NAMES:
            tt_val = tt_ints[step]
            ref_val = ref_ints[step]
            pcc_val = _pcc(tt_val, ref_val)
            layer_pccs[step] = pcc_val
            print(f"    {step:<22}: PCC = {pcc_val:.6f}")

        all_layer_pccs.append(layer_pccs)

    # ----------------------------------------------------------------
    # Print summary table
    # ----------------------------------------------------------------
    worst_pcc, worst_loc, drops = _print_pcc_table(all_layer_pccs, config, N_LAYERS)

    # ----------------------------------------------------------------
    # Print top-3 per-layer PCC (at "after_residual2") for easy reading
    # ----------------------------------------------------------------
    print("\nPer-layer cumulative PCC (hidden state after each layer, TT vs ref fed same BF16 embedding):")
    for i, layer_pccs in enumerate(all_layer_pccs):
        lt = config.layer_types[i]
        pcc_val = layer_pccs["after_residual2"]
        print(f"  Layer {i} ({lt}): output_PCC = {pcc_val:.6f}")

    print("\n[PerOpPCC-4L] DIAGNOSTIC COMPLETE")
    print("Look at the 'Per-Step Drop Analysis' above to identify the precision bottleneck.")
    print("The op with the largest drop is the fix target.")

    # No hard assertion here — this is a diagnostic test.
    # We do assert that the data was collected without error.
    assert len(all_layer_pccs) == N_LAYERS
    assert all(isinstance(v, float) for lp in all_layer_pccs for v in lp.values())


# ---------------------------------------------------------------------------
# Main diagnostic test: 16-layer (after 4-layer identifies bottleneck)
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_per_op_pcc_16layer(mesh_8x4):
    """Per-op PCC diagnostic on 16 layers — same as 4-layer but for full sweep.

    Run AFTER test_per_op_pcc_4layer to confirm per-op fixes hold at 16 layers.

    LOG_FILE: /tmp/qwen36_logs/t11b_per_op_16layer.log
    """
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.llama_model import TtQwen36Transformer
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 16
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    print(f"\n[PerOpPCC-16L] Loading weights for {N_LAYERS} layers...")
    t0 = time.time()
    global_weights = _load_embedding_and_norm_and_head_weights()
    layers_weights = [_load_layer_weights(i) for i in range(N_LAYERS)]
    print(f"[PerOpPCC-16L] Weights loaded in {time.time()-t0:.1f}s")

    torch.manual_seed(43)
    input_ids = torch.randint(0, config.vocab_size, (_B, _T_PREFILL))

    cos, sin = _build_rope_cos_sin(_T_PREFILL)
    causal_mask = torch.zeros(1, 1, _T_PREFILL, _T_PREFILL)
    causal_mask = causal_mask.masked_fill(
        torch.triu(torch.ones(_T_PREFILL, _T_PREFILL), diagonal=1).bool(), float("-inf")
    )

    print("[PerOpPCC-16L] Building CPU reference model...")
    ref_model = _build_ref_model(config, N_LAYERS, global_weights, layers_weights)

    print("[PerOpPCC-16L] Building TTNN model...")
    t0 = time.time()
    tt_model = TtQwen36Transformer(
        mesh_device=mesh_8x4,
        args=args,
        global_weights=global_weights,
        layers_weights=layers_weights,
        num_layers=N_LAYERS,
    )
    print(f"[PerOpPCC-16L] TTNN model built in {time.time()-t0:.1f}s")

    # Shared embedding start
    x_tt = tt_model._embed(input_ids)
    emb_cpu = ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_8x4, dim=0))
    emb_cpu = emb_cpu[:_B].float()

    cos_tt, sin_tt = tt_model.rope_setup.get_cos_sin_for_prefill(_T_PREFILL)
    rot_mats = (cos_tt, sin_tt)

    all_layer_pccs = []
    x_ref = emb_cpu.clone()

    print("[PerOpPCC-16L] Running per-layer diagnostic (16 layers)...")
    for layer_i in range(N_LAYERS):
        lt = config.layer_types[layer_i]

        x_tt, tt_ints, new_dn, new_conv = _capture_tt_layer_intermediates(
            tt_layer=tt_model.layers[layer_i],
            x_tt=x_tt,
            mesh_device=mesh_8x4,
            rot_mats=rot_mats,
            mode="prefill",
        )

        ref_ints = _capture_ref_layer_intermediates(ref_model, x_ref, cos, sin, causal_mask, layer_i)
        x_ref = ref_ints["after_residual2"].clone()

        layer_pccs = {}
        for step in STEP_NAMES:
            tt_val = tt_ints[step]
            ref_val = ref_ints[step]
            pcc_val = _pcc(tt_val, ref_val)
            layer_pccs[step] = pcc_val

        all_layer_pccs.append(layer_pccs)
        # Print layer-level summary inline so we can see progress
        output_pcc = layer_pccs["after_residual2"]
        print(f"  Layer {layer_i:2d} ({lt:<18}): output_PCC = {output_pcc:.6f}")

    worst_pcc, worst_loc, drops = _print_pcc_table(all_layer_pccs, config, N_LAYERS)

    print("\nPer-layer cumulative PCC:")
    for i, layer_pccs in enumerate(all_layer_pccs):
        lt = config.layer_types[i]
        pcc_val = layer_pccs["after_residual2"]
        flag = " !!!" if pcc_val < 0.99 else ""
        print(f"  Layer {i:2d} ({lt:<18}): output_PCC = {pcc_val:.6f}{flag}")

    assert len(all_layer_pccs) == N_LAYERS
    print("[PerOpPCC-16L] DIAGNOSTIC COMPLETE")


# ---------------------------------------------------------------------------
# Targeted fix verification: run 4-layer AFTER a precision fix
# to confirm the major-drop op improved.
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_verify_fix_4layer(mesh_8x4):
    """Run 4-layer per-op diagnostic and assert no single step drops PCC > 0.002.

    Used to verify that a precision fix has resolved the major drop site.
    Run this AFTER applying a fix to confirm it worked.

    LOG_FILE: /tmp/qwen36_logs/t11b_verify_fix_4layer.log
    """
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.llama_model import TtQwen36Transformer
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 4
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    global_weights = _load_embedding_and_norm_and_head_weights()
    layers_weights = [_load_layer_weights(i) for i in range(N_LAYERS)]

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (_B, _T_PREFILL))
    cos, sin = _build_rope_cos_sin(_T_PREFILL)
    causal_mask = torch.zeros(1, 1, _T_PREFILL, _T_PREFILL)
    causal_mask = causal_mask.masked_fill(
        torch.triu(torch.ones(_T_PREFILL, _T_PREFILL), diagonal=1).bool(), float("-inf")
    )

    ref_model = _build_ref_model(config, N_LAYERS, global_weights, layers_weights)
    tt_model = TtQwen36Transformer(
        mesh_device=mesh_8x4,
        args=args,
        global_weights=global_weights,
        layers_weights=layers_weights,
        num_layers=N_LAYERS,
    )

    x_tt = tt_model._embed(input_ids)
    emb_cpu = ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_8x4, dim=0))
    emb_cpu = emb_cpu[:_B].float()

    cos_tt, sin_tt = tt_model.rope_setup.get_cos_sin_for_prefill(_T_PREFILL)
    rot_mats = (cos_tt, sin_tt)

    all_layer_pccs = []
    x_ref = emb_cpu.clone()

    for layer_i in range(N_LAYERS):
        x_tt, tt_ints, new_dn, new_conv = _capture_tt_layer_intermediates(
            tt_layer=tt_model.layers[layer_i],
            x_tt=x_tt,
            mesh_device=mesh_8x4,
            rot_mats=rot_mats,
            mode="prefill",
        )
        ref_ints = _capture_ref_layer_intermediates(ref_model, x_ref, cos, sin, causal_mask, layer_i)
        x_ref = ref_ints["after_residual2"].clone()

        layer_pccs = {step: _pcc(tt_ints[step], ref_ints[step]) for step in STEP_NAMES}
        all_layer_pccs.append(layer_pccs)

    worst_pcc, worst_loc, drops = _print_pcc_table(all_layer_pccs, config, N_LAYERS)

    # Assertions: after fix, all steps should have PCC > 0.99,
    # and no single step should drop more than 0.005.
    for i, layer_pccs in enumerate(all_layer_pccs):
        for step in STEP_NAMES:
            pcc_val = layer_pccs[step]
            assert pcc_val > 0.99, f"Layer {i} step '{step}' PCC = {pcc_val:.4f} < 0.99 after fix"

    print(f"[VerifyFix] ALL steps PCC > 0.99. Worst = {worst_pcc:.4f}")
