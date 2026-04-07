# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test 4: Validate GDN prefill→decode state transition.

Verifies that TTNNQwen35GatedDeltaNet correctly transitions from prefill
(seq_len > 1) to decode (seq_len = 1), with conv/rec states carrying over.

The concern: during prefill, the GDN initializes and updates conv_states and
rec_states via ttnn.copy/recurrence ops. After prefill, these states must be
in the correct shape and values for the decode path to produce correct output.
"""

import pytest
import torch

from .conftest import (
    compute_pcc,
    get_config_attr,
    get_layer_type,
    skip_no_ttnn,
    skip_no_transformers,
    skip_no_symbiote,
    TTNN_AVAILABLE,
    TT_SYMBIOTE_AVAILABLE,
)

if TTNN_AVAILABLE:
    import ttnn

if TT_SYMBIOTE_AVAILABLE:
    from models.experimental.tt_symbiote.modules.qwen35_gated_deltanet import TTNNQwen35GatedDeltaNet
    from models.experimental.tt_symbiote.utils.device_management import set_device


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_gdn_prefill_then_decode_pcc(device, model_1_layer):
    """GDN prefill followed by decode vs PyTorch reference. PCC >= 0.90.

    Runs prefill (seq_len=32) then 3 decode steps (seq_len=1 each) through both
    PyTorch and TTNN, comparing decode outputs.
    """
    from transformers.cache_utils import DynamicCache

    model, config = model_1_layer
    layer_0 = model.model.layers[0]

    assert get_layer_type(layer_0) == "linear_attention"
    torch_gdn = layer_0.linear_attn

    hidden_size = get_config_attr(config, "hidden_size")
    batch_size = 1
    prefill_len = 32
    num_decode_steps = 3

    # Generate inputs
    x_prefill = torch.randn(batch_size, prefill_len, hidden_size, dtype=torch.bfloat16)
    x_decode_steps = [torch.randn(batch_size, 1, hidden_size, dtype=torch.bfloat16) for _ in range(num_decode_steps)]

    # ---- PyTorch reference: prefill + decode ----
    torch_cache = DynamicCache(config=config)
    with torch.no_grad():
        # Prefill
        torch_gdn(x_prefill, cache_params=torch_cache)

        # Decode steps
        torch_decode_outs = []
        for x_d in x_decode_steps:
            out_d = torch_gdn(x_d, cache_params=torch_cache)
            torch_decode_outs.append(out_d)

    # ---- TTNN: prefill + decode ----
    ttnn_gdn = TTNNQwen35GatedDeltaNet.from_torch(torch_gdn)
    set_device(ttnn_gdn, device)
    ttnn_gdn.preprocess_weights()
    ttnn_gdn.move_weights_to_device()

    # Use a dummy cache_params (just needs to be not-None for decode path)
    class DummyCache:
        """Minimal cache that satisfies cache_params is not None check."""

    dummy_cache = DummyCache()

    # Prefill on device
    tt_prefill = ttnn.from_torch(
        x_prefill,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gdn.forward(tt_prefill, cache_params=dummy_cache)

    # Verify states were initialized
    assert ttnn_gdn.conv_states is not None, "conv_states not initialized after prefill"
    assert ttnn_gdn.rec_states is not None, "rec_states not initialized after prefill"
    print(
        f"  After prefill: conv_states count={len(ttnn_gdn.conv_states)}, "
        f"rec_states shape={ttnn_gdn.rec_states.shape}"
    )

    # Decode steps on device
    ttnn_decode_outs = []
    for i, x_d in enumerate(x_decode_steps):
        tt_decode = ttnn.from_torch(
            x_d,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_d = ttnn_gdn.forward(tt_decode, cache_params=dummy_cache)
        ttnn_decode_outs.append(out_d)

    # Compare decode outputs
    for i, (torch_out, ttnn_out) in enumerate(zip(torch_decode_outs, ttnn_decode_outs)):
        torch_out_tensor = torch_out if isinstance(torch_out, torch.Tensor) else torch_out[0]
        ttnn_out_torch = ttnn.to_torch(ttnn_out).to(torch.float32)
        torch_out_f32 = torch_out_tensor.to(torch.float32)

        # Trim to matching shape (TTNN may pad)
        if ttnn_out_torch.shape != torch_out_f32.shape:
            slices = tuple(slice(0, s) for s in torch_out_f32.shape)
            ttnn_out_torch = ttnn_out_torch[slices]

        pcc_val = compute_pcc(torch_out_f32, ttnn_out_torch)
        print(f"  Decode step {i}: PCC = {pcc_val:.6f}")
        assert pcc_val >= 0.90, f"Decode step {i} PCC {pcc_val:.6f} < 0.90 — state transition may be broken"


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_gdn_decode_without_prefill_initializes_state(device, model_1_layer):
    """GDN decode with no prior prefill lazily initializes state.

    When cache_params is not None and seq_len=1 but no prefill has run,
    the GDN should initialize states on first call and produce valid output.
    """
    model, config = model_1_layer
    layer_0 = model.model.layers[0]

    assert get_layer_type(layer_0) == "linear_attention"
    torch_gdn = layer_0.linear_attn

    hidden_size = get_config_attr(config, "hidden_size")
    batch_size = 1

    ttnn_gdn = TTNNQwen35GatedDeltaNet.from_torch(torch_gdn)
    set_device(ttnn_gdn, device)
    ttnn_gdn.preprocess_weights()
    ttnn_gdn.move_weights_to_device()

    assert ttnn_gdn.conv_states is None, "States should be None before any forward"

    class DummyCache:
        pass

    x_decode = torch.randn(batch_size, 1, hidden_size, dtype=torch.bfloat16)
    tt_decode = ttnn.from_torch(
        x_decode,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Should not crash — states initialized lazily in _forward_decode
    out = ttnn_gdn.forward(tt_decode, cache_params=DummyCache())

    assert ttnn_gdn.conv_states is not None, "conv_states not initialized after decode"
    assert ttnn_gdn.rec_states is not None, "rec_states not initialized after decode"

    out_torch = ttnn.to_torch(out)
    assert not torch.isnan(out_torch).any(), "Output contains NaN after cold decode"
    assert not torch.isinf(out_torch).any(), "Output contains Inf after cold decode"
    print(
        f"  Cold decode output: shape={out_torch.shape}, "
        f"mean={out_torch.float().mean():.6f}, std={out_torch.float().std():.6f}"
    )


@pytest.fixture(scope="module")
def model_1_layer():
    """Load Qwen3.5-27B-FP8 with 1 hidden layer (module-scoped for reuse)."""
    from .conftest import load_model

    model, config = load_model(num_hidden_layers=1)
    return model, config
