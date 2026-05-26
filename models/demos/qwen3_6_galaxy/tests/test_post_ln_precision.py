# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit test for post_attention_layernorm precision bug — Task T11c.

TDD: this test is written FIRST to reproduce the observed 0.062 PCC drop in
post_attention_layernorm (layer 0) that was identified in the T11b per-op
diagnostic log (t11b_per_op_4layer_v2.log).

The test isolates DistributedNorm with the ACTUAL ``residual + attn_out``
tensor from layer 0 of the Qwen3.6-27B DeltaNet stack, as captured from
the CPU reference model.

Expected initial result (before fix): PCC ≈ 0.935 (reproducing the bug).
Expected result after fix:           PCC > 0.999.

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    LOG_FILE: /tmp/qwen36_logs/t11c_post_ln_unit_test.log
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_post_ln_precision.py \\
        -x -s -v -k test_post_ln_precision_layer0 \\
        2>&1 | tee /tmp/qwen36_logs/t11c_post_ln_unit_test.log
"""

from __future__ import annotations

import json
import pathlib
import sys

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

_DIM = 5120
_EPS = 1e-6
_B, _T = 1, 32


# ---------------------------------------------------------------------------
# PCC helper
# ---------------------------------------------------------------------------


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    cc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return cc


# ---------------------------------------------------------------------------
# Weight loading helpers
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


def _load_embedding_weight() -> torch.Tensor:
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    key = "model.language_model.embed_tokens.weight"
    fn = weight_map[key]
    shard = load_st(str(_SNAPSHOT_DIR / fn))
    return shard[key].float()


# ---------------------------------------------------------------------------
# Fabric mesh fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_8x4():
    """Open the full 8×4 fabric mesh."""
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
# Helpers: shard/gather (same as llama_decoder.py)
# ---------------------------------------------------------------------------


def _shard_across_cols_torch(x_cpu: torch.Tensor, mesh_device, cluster_shape=(8, 4)):
    """Upload [B, T, H] to mesh sharded across dim=-1 (cols).

    Returns TTNN tensor [B, 1, T, H/4] per col.
    """
    import ttnn

    B, T, H = x_cpu.shape
    x_4d = x_cpu.unsqueeze(1)  # [B, 1, T, H]
    return ttnn.from_torch(
        x_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=list(cluster_shape)),
    )


def _gather_from_cols_to_torch(x_sharded, mesh_device, B=1, cluster_shape=(8, 4)):
    """Gather column-sharded TTNN tensor to [B, T, H] CPU float32."""
    import ttnn

    x_cpu = ttnn.to_torch(
        x_sharded,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=list(cluster_shape)),
    )
    # x_cpu: [8*B, 1, T, H]
    x_cpu = x_cpu[:B, 0, :, :]  # [B, T, H]
    return x_cpu.float()


# ---------------------------------------------------------------------------
# CPU reference RMSNorm (zero-centered)
# ---------------------------------------------------------------------------


def _ref_rmsnorm_zero_centered(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference zero-centered RMSNorm: output = (1 + weight) * x * rsqrt(mean(x^2) + eps)."""
    x_f32 = x.float()
    rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
    return x_f32 * rms * (1.0 + weight.float())


# ---------------------------------------------------------------------------
# Helper: capture layer-0 post_LN input from CPU reference
# ---------------------------------------------------------------------------


def _capture_post_ln_input_cpu(
    layer_weights: dict, embed_weight: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """Run layer-0 CPU reference up to the first residual add, returning the post_LN input.

    Steps:
      1. Embed input_ids → [B, T, H]
      2. Cast to BF16 and back to FP32 (mimicking the TTNN embedding round-trip)
      3. Run input_layernorm (zero-centered)
      4. Run DeltaNet attention block (layer 0 is linear_attention)
      5. Compute residual + attn_out  ← this is the post_LN input
      6. Cast to BF16 and back to FP32 (mimicking the TTNN round-trip before post_LN)

    Returns:
        [B, T, H] float32 tensor = (residual + attn_out).to(bfloat16).float()
        This matches what _shard_across_cols extracts from the TTNN device (device-0 copy).
    """
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config, Qwen36TextModel

    with open(_SNAPSHOT_DIR / "config.json") as f:
        cfg = Qwen36Config(json.load(f))

    # Build single-layer reference model (just enough to run layer 0)
    single_layer_cfg = type(cfg).__new__(type(cfg))
    single_layer_cfg.__dict__.update(cfg.__dict__)
    single_layer_cfg.num_hidden_layers = 1
    single_layer_cfg.layer_types = cfg.layer_types[:1]

    model = Qwen36TextModel(single_layer_cfg)
    model.eval()

    with torch.no_grad():
        # Load weights
        model.tok_embeddings.weight.data.copy_(embed_weight[: cfg.vocab_size])
        layer = model.layers[0]
        layer.input_layernorm.weight.data.copy_(layer_weights["input_layernorm.weight"])
        layer.post_attention_layernorm.weight.data.copy_(layer_weights["post_attention_layernorm.weight"])
        layer.mlp.gate_proj.weight.data.copy_(layer_weights["mlp.gate_proj.weight"])
        layer.mlp.up_proj.weight.data.copy_(layer_weights["mlp.up_proj.weight"])
        layer.mlp.down_proj.weight.data.copy_(layer_weights["mlp.down_proj.weight"])
        dn = layer.attention
        dn.in_proj_qkv.weight.data.copy_(layer_weights["linear_attn.in_proj_qkv.weight"])
        dn.in_proj_z.weight.data.copy_(layer_weights["linear_attn.in_proj_z.weight"])
        dn.in_proj_a.weight.data.copy_(layer_weights["linear_attn.in_proj_a.weight"])
        dn.in_proj_b.weight.data.copy_(layer_weights["linear_attn.in_proj_b.weight"])
        dn.conv1d.weight.data.copy_(layer_weights["linear_attn.conv1d.weight"])
        dn.A_log.data.copy_(layer_weights["linear_attn.A_log"])
        dn.dt_bias.data.copy_(layer_weights["linear_attn.dt_bias"])
        dn.norm.weight.data.copy_(layer_weights["linear_attn.norm.weight"])
        dn.out_proj.weight.data.copy_(layer_weights["linear_attn.out_proj.weight"])

        # Step 1: embed + BF16 round-trip (mimic TTNN embedding)
        x = model.tok_embeddings(input_ids).to(torch.bfloat16).float()  # [B, T, H]

        # Step 2: input_layernorm
        residual = x.clone()
        x_normed = layer.input_layernorm(x)

        # Step 3: DeltaNet (linear_attention), no cos/sin needed
        attn_out, conv_state_new, recurrent_state_new = layer.attention(x_normed)

        # Step 4: first residual add
        x_after_residual1 = residual + attn_out

        # Cast to BF16 and back — mimicking the TTNN device round-trip
        # (TTNN tensors are in BF16, so when we gather from device we get BF16 values)
        x_post_ln_input_bf16 = x_after_residual1.to(torch.bfloat16).float()

    return x_post_ln_input_bf16, x_after_residual1.float()


# ---------------------------------------------------------------------------
# Test 1: Reproduce the bug — DistributedNorm on real post_LN input
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_post_ln_precision_layer0(mesh_8x4):
    """Reproduce the post_attention_layernorm PCC drop on real layer-0 input.

    This test is written RED-first to confirm the bug is in DistributedNorm
    itself (not in the surrounding integration), then verifies the fix.

    Expected BEFORE fix: PCC ≈ 0.935 (matching the diagnostic observation).
    Expected AFTER fix:  PCC > 0.999.

    Steps:
      1. Load real post_attention_layernorm.weight for layer 0
      2. Build the actual residual + attn_out input from CPU reference
      3. Feed that input through TTNN DistributedNorm (col-sharded path)
      4. Compare to CPU reference (zero-centered RMSNorm on the same input)
      5. Assert PCC > 0.999

    If this test FAILS at PCC ≈ 0.935, the bug is in DistributedNorm itself.
    If it PASSES at PCC > 0.999, the bug is in the surrounding integration
    (residual add precision, layout conversion, etc.).
    """
    from models.demos.qwen3_6_galaxy.tt.distributed_norm import DistributedNorm

    print("\n[T11c] Loading weights for layer 0...")
    layer_weights = _load_layer_weights(0)
    embed_weight = _load_embedding_weight()
    post_ln_weight = layer_weights["post_attention_layernorm.weight"]  # [5120]
    input_ln_weight = layer_weights["input_layernorm.weight"]  # [5120]

    # -------------------------------------------------------------------
    # Step 1: Build input tensor that mimics the actual decode situation.
    # Use seed=42 to match test_full_model.py and test_per_op_pcc.py.
    # -------------------------------------------------------------------
    torch.manual_seed(42)
    input_ids = torch.randint(0, 248320, (_B, _T))

    print("[T11c] Running CPU reference to capture post_LN input (layer 0)...")
    post_ln_input_bf16, post_ln_input_f32 = _capture_post_ln_input_cpu(layer_weights, embed_weight, input_ids)
    # post_ln_input_bf16: [1, 32, 5120] float32 (BF16 values upcast)

    print(f"[T11c] post_LN input stats:")
    print(f"  std  = {post_ln_input_f32.std().item():.4f}")
    print(f"  mean = {post_ln_input_f32.mean().item():.4f}")
    print(f"  max  = {post_ln_input_f32.abs().max().item():.4f}")
    print(f"  min  = {post_ln_input_f32.abs().min().item():.6f}")

    # -------------------------------------------------------------------
    # Also capture input_LN's input (embedding output) for comparison
    # -------------------------------------------------------------------
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    emb_key = "model.language_model.embed_tokens.weight"
    emb_shard = load_st(str(_SNAPSHOT_DIR / weight_map[emb_key]))
    emb_w = emb_shard[emb_key].float()
    import torch.nn as nn

    emb = nn.Embedding(248320, _DIM)
    emb.weight.data.copy_(emb_w[:248320])
    with torch.no_grad():
        x_embed = emb(input_ids).to(torch.bfloat16).float()

    print(f"[T11c] input_LN input (embedding) stats:")
    print(f"  std  = {x_embed.std().item():.6f}")
    print(f"  mean = {x_embed.mean().item():.6f}")
    print(f"  max  = {x_embed.abs().max().item():.6f}")

    # -------------------------------------------------------------------
    # Step 2: Run TTNN DistributedNorm (post_LN) on the real post_LN input
    # -------------------------------------------------------------------
    print("\n[T11c] Building DistributedNorm (post_LN) on TTNN...")
    post_ln_norm = DistributedNorm(
        mesh_device=mesh_8x4,
        weight_torch=post_ln_weight,
        eps=_EPS,
        zero_centered=True,
    )

    print("[T11c] Sharding post_LN input and running DistributedNorm...")
    x_sharded = _shard_across_cols_torch(post_ln_input_bf16, mesh_8x4)
    x_normed_sharded = post_ln_norm(x_sharded)
    x_normed_cpu = _gather_from_cols_to_torch(x_normed_sharded, mesh_8x4, B=_B)
    x_sharded.deallocate(True)
    x_normed_sharded.deallocate(True)

    # -------------------------------------------------------------------
    # Step 3: CPU reference for post_LN (zero-centered RMSNorm)
    # -------------------------------------------------------------------
    ref_post_ln_out = _ref_rmsnorm_zero_centered(post_ln_input_bf16, post_ln_weight, eps=_EPS)

    # -------------------------------------------------------------------
    # Step 4: PCC comparison
    # -------------------------------------------------------------------
    pcc_post_ln = _pcc(x_normed_cpu, ref_post_ln_out)
    print(f"\n[T11c] post_LN PCC (TTNN vs CPU ref) = {pcc_post_ln:.6f}")
    print(f"[T11c] Expected > 0.999 (after fix), was ~0.935 before fix")

    # -------------------------------------------------------------------
    # Comparison: also run input_LN on the embedding input (should be ~0.9999)
    # -------------------------------------------------------------------
    print("\n[T11c] Running DistributedNorm (input_LN) on embedding input for comparison...")
    input_ln_norm = DistributedNorm(
        mesh_device=mesh_8x4,
        weight_torch=input_ln_weight,
        eps=_EPS,
        zero_centered=True,
    )
    x_embed_sharded = _shard_across_cols_torch(x_embed, mesh_8x4)
    x_embed_normed_sharded = input_ln_norm(x_embed_sharded)
    x_embed_normed_cpu = _gather_from_cols_to_torch(x_embed_normed_sharded, mesh_8x4, B=_B)
    x_embed_sharded.deallocate(True)
    x_embed_normed_sharded.deallocate(True)

    ref_input_ln_out = _ref_rmsnorm_zero_centered(x_embed, input_ln_weight, eps=_EPS)
    pcc_input_ln = _pcc(x_embed_normed_cpu, ref_input_ln_out)
    print(f"[T11c] input_LN PCC (TTNN vs CPU ref) = {pcc_input_ln:.6f}")
    print(f"[T11c] Expected > 0.999 (this is the baseline working case)")

    # -------------------------------------------------------------------
    # Report what was found
    # -------------------------------------------------------------------
    print(f"\n[T11c] === PCC SUMMARY ===")
    print(f"  input_LN on embedding input:    PCC = {pcc_input_ln:.6f}")
    print(f"  post_LN on residual+attn input: PCC = {pcc_post_ln:.6f}")
    pcc_drop = pcc_input_ln - pcc_post_ln
    print(f"  Drop between input_LN and post_LN: {pcc_drop:.6f}")

    if pcc_post_ln < 0.99:
        print(f"\n[T11c] BUG REPRODUCED: post_LN PCC = {pcc_post_ln:.4f} < 0.99")
        print("[T11c] DistributedNorm itself degrades on post_LN input distribution.")
        print("[T11c] Root cause: see test output and investigate stats dtype / magnitude.")
    else:
        print(f"\n[T11c] DistributedNorm passes in isolation (PCC = {pcc_post_ln:.4f} > 0.99).")
        print("[T11c] Bug is in the surrounding integration, not DistributedNorm itself.")

    # -------------------------------------------------------------------
    # Assertion (must pass after fix)
    # -------------------------------------------------------------------
    assert pcc_post_ln > 0.999, (
        f"post_LN PCC = {pcc_post_ln:.6f} < 0.999  "
        f"(input_LN baseline = {pcc_input_ln:.6f}). "
        f"The post_attention_layernorm precision bug was not fixed."
    )
    print("[T11c] PASSED: post_LN PCC > 0.999")


# ---------------------------------------------------------------------------
# Test 2: Investigate stats precision — compare BF16 vs FP32 stats
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_post_ln_stats_dtype_comparison(mesh_8x4):
    """Compare DistributedNorm precision with BF16 vs FP32 intermediate stats.

    This test runs DistributedNorm on the real post_LN input with two
    configurations:
      1. Current: rms_norm_pre_all_gather returns stats in bfloat16
      2. Modified: rms_norm_pre_all_gather returns stats in float32

    If the float32 stats version has significantly better PCC, the bug is the
    BF16 precision of the intermediate statistics in rms_norm_pre_all_gather.

    This is a diagnostic test — no assertion, just prints PCC comparison.
    """
    from models.demos.qwen3_6_galaxy.tt.distributed_norm import DistributedNorm

    print("\n[T11c-StatsTest] Loading weights...")
    layer_weights = _load_layer_weights(0)
    embed_weight = _load_embedding_weight()
    post_ln_weight = layer_weights["post_attention_layernorm.weight"]

    torch.manual_seed(42)
    input_ids = torch.randint(0, 248320, (_B, _T))

    print("[T11c-StatsTest] Capturing post_LN input from CPU reference...")
    post_ln_input_bf16, _ = _capture_post_ln_input_cpu(layer_weights, embed_weight, input_ids)

    # Reference CPU output
    ref_out = _ref_rmsnorm_zero_centered(post_ln_input_bf16, post_ln_weight, eps=_EPS)

    # ---- Current DistributedNorm (BF16 stats) ----
    norm_bf16_stats = DistributedNorm(
        mesh_device=mesh_8x4,
        weight_torch=post_ln_weight,
        eps=_EPS,
        zero_centered=True,
    )
    x_sharded = _shard_across_cols_torch(post_ln_input_bf16, mesh_8x4)
    x_normed_sharded = norm_bf16_stats(x_sharded)
    out_bf16_stats = _gather_from_cols_to_torch(x_normed_sharded, mesh_8x4, B=_B)
    x_sharded.deallocate(True)
    x_normed_sharded.deallocate(True)
    pcc_bf16 = _pcc(out_bf16_stats, ref_out)
    print(f"[T11c-StatsTest] BF16 stats: PCC = {pcc_bf16:.6f}")

    # ---- FP32 stats version (use DistributedNormFP32Stats if available) ----
    # Try using dtype=ttnn.float32 for rms_norm_pre_all_gather
    try:
        from models.demos.qwen3_6_galaxy.tt.distributed_norm import DistributedNormFP32Stats

        norm_fp32_stats = DistributedNormFP32Stats(
            mesh_device=mesh_8x4,
            weight_torch=post_ln_weight,
            eps=_EPS,
            zero_centered=True,
        )
        x_sharded2 = _shard_across_cols_torch(post_ln_input_bf16, mesh_8x4)
        x_normed_sharded2 = norm_fp32_stats(x_sharded2)
        out_fp32_stats = _gather_from_cols_to_torch(x_normed_sharded2, mesh_8x4, B=_B)
        x_sharded2.deallocate(True)
        x_normed_sharded2.deallocate(True)
        pcc_fp32 = _pcc(out_fp32_stats, ref_out)
        print(f"[T11c-StatsTest] FP32 stats: PCC = {pcc_fp32:.6f}")
        print(f"[T11c-StatsTest] Improvement: {pcc_fp32 - pcc_bf16:.6f}")
    except ImportError:
        print("[T11c-StatsTest] DistributedNormFP32Stats not available yet — run test_post_ln_precision_layer0 first")

    print(f"\n[T11c-StatsTest] Summary:")
    print(f"  BF16 stats PCC: {pcc_bf16:.6f}")
    print("  (FP32 stats test requires DistributedNormFP32Stats class to be added)")
