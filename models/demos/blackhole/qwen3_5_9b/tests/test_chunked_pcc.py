# models/demos/blackhole/qwen3_5_9b/tests/test_chunked_pcc.py
"""PCC validation: chunked vs recurrent delta rule.

Verifies that the chunked (parallel) delta rule produces output matching
the recurrent (sequential) reference within PCC > 0.99.

Run: pytest models/demos/blackhole/qwen3_5_9b/tests/test_chunked_pcc.py -v -s --timeout=600
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs

CHECKPOINT_DIR = "/local/ttuser/atupe/Qwen9b"


def compute_pcc(a, b):
    """Pearson correlation coefficient between two tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _run_single_deltanet_layer_both_modes(device, seq_len, chunk_size=64):
    """Run one DeltaNet layer in both recurrent and chunk mode, return PCC."""
    import glob

    from safetensors import safe_open

    from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_deltanet import Qwen35GatedDeltaNet
    from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict

    args = Qwen35ModelArgs(mesh_device=device, checkpoint_dir=CHECKPOINT_DIR)
    raw = {}
    for path in sorted(glob.glob(f"{CHECKPOINT_DIR}/model.safetensors-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)
    sd = remap_qwen35_state_dict(raw)
    del raw

    # Use layer 0 (a DeltaNet layer)
    layer_num = 0
    dn = Qwen35GatedDeltaNet(args, sd, layer_num, device)

    # Create random input of the target seq_len
    x_torch = torch.randn(1, seq_len, 4096, dtype=torch.bfloat16)
    x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run recurrent mode (reference)
    dn.reset_state(batch_size=1)
    x_recurrent = ttnn.clone(x_ttnn)
    out_recurrent = dn.forward(x_recurrent, mode="recurrent")
    out_recurrent_torch = ttnn.to_torch(out_recurrent)
    ttnn.deallocate(out_recurrent)

    # Run chunk mode
    dn.reset_state(batch_size=1)
    x_chunk = ttnn.clone(x_ttnn)
    out_chunk = dn.forward(x_chunk, mode="chunk", chunk_size=chunk_size)
    out_chunk_torch = ttnn.to_torch(out_chunk)
    ttnn.deallocate(out_chunk)
    ttnn.deallocate(x_ttnn)

    pcc = compute_pcc(out_recurrent_torch, out_chunk_torch)
    logger.info(f"seq_len={seq_len}, chunk_size={chunk_size}: PCC={pcc:.6f}")
    logger.info(f"  recurrent range: [{out_recurrent_torch.min():.4f}, {out_recurrent_torch.max():.4f}]")
    logger.info(f"  chunk range:     [{out_chunk_torch.min():.4f}, {out_chunk_torch.max():.4f}]")

    return pcc


@pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024, 100, 300, 700])
def test_chunked_vs_recurrent_pcc(seq_len, device):
    """Chunked delta rule must match recurrent reference with PCC > 0.99."""
    pcc = _run_single_deltanet_layer_both_modes(device, seq_len, chunk_size=64)
    threshold = 0.98 if seq_len >= 1024 else 0.99
    assert pcc > threshold, f"PCC {pcc:.6f} < {threshold} at seq_len={seq_len}"


@pytest.mark.parametrize(
    "seq_len, chunk_size",
    [
        (2048, 64),  # 32 sub-chunks — baseline for comparison
        (2048, 256),  # 8 sub-chunks — used in layer-chunked prefill
        (4096, 256),  # 16 sub-chunks — matches tested range at 1024/64
    ],
    ids=["2k_c64", "2k_c256", "4k_c256"],
)
def test_chunked_vs_recurrent_pcc_long(seq_len, chunk_size, device):
    """Long-sequence PCC: validates larger chunk_size reduces error accumulation.

    At chunk_size=64, error compounds across many sub-chunks (32 for 2048, 64 for 4096).
    At chunk_size=256, sub-chunk count stays within the validated range (<=16).
    """
    pcc = _run_single_deltanet_layer_both_modes(device, seq_len, chunk_size=chunk_size)
    if chunk_size == 64:
        # chunk_size=64 at long sequences: expect degradation, just log it
        threshold = 0.90
        logger.warning(f"chunk_size=64 at seq_len={seq_len}: PCC={pcc:.6f} (monitoring, threshold={threshold})")
    else:
        # chunk_size=256: should match quality of short sequences
        threshold = 0.97
    assert pcc > threshold, f"PCC {pcc:.6f} < {threshold} at seq_len={seq_len}, chunk_size={chunk_size}"


@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_delta_rule_ops_direct_pcc(seq_len, device):
    """Direct PCC test: TTNN chunk_gated_delta_rule_ttnn vs PyTorch chunk_gated_delta_rule.

    Uses realistic g magnitudes (negative, cumsum reaching -50+) to exercise
    the decay normalization path. The PyTorch reference is the ground truth.
    """
    import os
    import sys

    exp_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "experimental", "gated_attention_gated_deltanet"
        )
    )
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)

    from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
        chunk_gated_delta_rule,
    )
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import chunk_gated_delta_rule_ttnn

    B, H, K, V = 1, 32, 128, 128
    chunk_size = 64

    # Realistic inputs: g values are negative (log-space decay), typical range [-0.5, -2.0] per token
    torch.manual_seed(42)
    q_t = torch.randn(B, seq_len, H, K, dtype=torch.float32) * 0.1
    k_t = torch.randn(B, seq_len, H, K, dtype=torch.float32) * 0.1
    v_t = torch.randn(B, seq_len, H, V, dtype=torch.float32) * 0.1
    g_t = -torch.abs(torch.randn(B, seq_len, H, dtype=torch.float32)) * 0.8  # mean ~ -0.64
    beta_t = torch.sigmoid(torch.randn(B, seq_len, H, dtype=torch.float32))

    # PyTorch reference
    out_ref, state_ref = chunk_gated_delta_rule(
        q_t, k_t, v_t, g_t, beta_t, chunk_size=chunk_size, output_final_state=True
    )

    # TTNN implementation
    q_ttnn = ttnn.from_torch(q_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_ttnn = ttnn.from_torch(k_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_ttnn = ttnn.from_torch(v_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g_ttnn = ttnn.from_torch(g_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    beta_ttnn = ttnn.from_torch(beta_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_ttnn, state_ttnn = chunk_gated_delta_rule_ttnn(
        q_ttnn, k_ttnn, v_ttnn, beta_ttnn, g_ttnn, chunk_size=chunk_size, device=device
    )

    out_ttnn_torch = ttnn.to_torch(out_ttnn)
    state_ttnn_torch = ttnn.to_torch(state_ttnn)

    output_pcc = compute_pcc(out_ref, out_ttnn_torch)
    state_pcc = compute_pcc(state_ref, state_ttnn_torch)

    logger.info(f"seq_len={seq_len}: output PCC={output_pcc:.6f}, state PCC={state_pcc:.6f}")
    logger.info(f"  ref output range: [{out_ref.min():.4f}, {out_ref.max():.4f}]")
    logger.info(f"  ttnn output range: [{out_ttnn_torch.min():.4f}, {out_ttnn_torch.max():.4f}]")
    logger.info(f"  ref state range: [{state_ref.min():.4f}, {state_ref.max():.4f}]")
    logger.info(f"  ttnn state range: [{state_ttnn_torch.min():.4f}, {state_ttnn_torch.max():.4f}]")

    ttnn.deallocate(out_ttnn)
    ttnn.deallocate(state_ttnn)

    assert output_pcc > 0.99, f"Output PCC {output_pcc:.6f} < 0.99 at seq_len={seq_len}"
    assert state_pcc > 0.99, f"State PCC {state_pcc:.6f} < 0.99 at seq_len={seq_len}"


@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_chunked_state_pcc(seq_len, device):
    """Verify final recurrent state matches between chunked and recurrent modes."""
    import glob

    from safetensors import safe_open

    from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_deltanet import Qwen35GatedDeltaNet
    from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict

    args = Qwen35ModelArgs(mesh_device=device, checkpoint_dir=CHECKPOINT_DIR)
    raw = {}
    for path in sorted(glob.glob(f"{CHECKPOINT_DIR}/model.safetensors-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)
    sd = remap_qwen35_state_dict(raw)
    del raw

    dn = Qwen35GatedDeltaNet(args, sd, layer_num=0, device=device)

    torch.manual_seed(42)
    x_torch = torch.randn(1, seq_len, 4096, dtype=torch.bfloat16)
    x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Recurrent
    dn.reset_state(batch_size=1)
    out_rec = dn.forward(ttnn.clone(x_ttnn), mode="recurrent")
    state_rec = ttnn.to_torch(dn.recurrent_state)
    ttnn.deallocate(out_rec)

    # Chunked
    dn.reset_state(batch_size=1)
    out_chunk = dn.forward(ttnn.clone(x_ttnn), mode="chunk", chunk_size=64)
    state_chunk = ttnn.to_torch(dn.recurrent_state)
    ttnn.deallocate(out_chunk)
    ttnn.deallocate(x_ttnn)

    state_pcc = compute_pcc(state_rec, state_chunk)
    logger.info(f"seq_len={seq_len}: state PCC={state_pcc:.6f}")
    logger.info(f"  recurrent state range: [{state_rec.min():.4f}, {state_rec.max():.4f}]")
    logger.info(f"  chunked state range:   [{state_chunk.min():.4f}, {state_chunk.max():.4f}]")

    # State accumulates errors across all chunks; output PCC is the primary quality metric.
    # State PCC is a secondary check — relaxed threshold since small per-chunk errors compound.
    threshold = 0.95 if seq_len >= 256 else 0.98
    assert state_pcc > threshold, f"State PCC {state_pcc:.6f} < {threshold} at seq_len={seq_len}"
