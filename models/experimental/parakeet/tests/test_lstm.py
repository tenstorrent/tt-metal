"""
Test script to compare LSTMDropout from NeMo and TTNN implementation.

This script:
1. Loads pretrained NeMo model and extracts LSTMDropout
2. Extracts LSTM weights from NeMo LSTMDropout
3. Initializes TT_LSTMDropout with the same weights
4. Runs forward passes on both implementations
5. Compares outputs using PCC and other metrics
"""
import os
import sys
from typing import Optional, Dict, List
import nemo.collections.asr as nemo_asr

# Import NeMo LSTMDropout
from nemo.collections.common.parts.rnn import LSTMDropout
import torch
import ttnn
from models.common.metrics import compute_pcc, compute_max_abs_error, compute_mean_abs_error
from models.experimental.parakeet.tt.tt_lstm import TtLSTMCell

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def load_pretrained_nemo_lstm(
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
    map_location: str = "cpu",
) -> LSTMDropout:
    """
    Load pretrained NeMo ASR model and extract LSTMDropout module.

    Args:
        model_name: HuggingFace model name or path to checkpoint
        map_location: Device to load model on ('cpu' or 'cuda')

    Returns:
        LSTMDropout module with pretrained weights
    """

    print(f"Loading pretrained model: {model_name}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=map_location)
    asr_model.eval()

    # Try to find LSTMDropout module by searching
    lstm_module = None
    for name, module in asr_model.named_modules():
        if isinstance(module, LSTMDropout):
            lstm_module = module
            print(f"✅ Found LSTMDropout at: {name}")
            break

    if lstm_module is None:
        # If not found, try to find it in decoder
        if hasattr(asr_model, "decoder"):
            decoder = asr_model.decoder
            # Check if decoder has prediction network with LSTM
            if hasattr(decoder, "prediction"):
                pred = decoder.prediction
                if hasattr(pred, "dec_rnn"):
                    # Extract the underlying LSTM from LSTMDropout wrapper
                    lstm_wrapper = pred["dec_rnn"]
                    # Find the actual LSTM module
                    for name, module in lstm_wrapper.named_modules():
                        if isinstance(module, torch.nn.LSTM):
                            # Create a wrapper LSTMDropout with this LSTM
                            lstm_module = LSTMDropout(
                                input_size=module.input_size,
                                hidden_size=module.hidden_size,
                                num_layers=module.num_layers,
                                dropout=0.0,
                                forget_gate_bias=1.0,
                            )
                            # Copy weights
                            lstm_module.lstm = module
                            print(f"✅ Extracted LSTM from decoder prediction network")
                            break

    if lstm_module is None:
        raise ValueError("Could not find LSTMDropout module in the loaded model")

    print(f"   - Input size: {lstm_module.lstm.input_size}")
    print(f"   - Hidden size: {lstm_module.lstm.hidden_size}")
    print(f"   - Num layers: {lstm_module.lstm.num_layers}")

    return lstm_module


def extract_lstm_weights_from_nemo(
    nemo_lstm: LSTMDropout,
    layer_idx: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Extract LSTM weights from NeMo LSTMDropout for all layers or a specific layer.

    Args:
        nemo_lstm: NeMo LSTMDropout instance
        layer_idx: Layer index (None to extract all layers, or int for specific layer)

    Returns:
        List of dictionaries with weights and biases (one per layer), or single dict if layer_idx specified
        Each dict contains:
            - weight_ih: [4*hidden_size, input_size] for layer 0, [4*hidden_size, hidden_size] for others
            - weight_hh: [4*hidden_size, hidden_size]
            - bias_ih: [4*hidden_size]
            - bias_hh: [4*hidden_size]
            - input_size: input_size for layer 0, hidden_size for others
            - hidden_size: hidden_size
    """
    lstm = nemo_lstm.lstm
    num_layers = lstm.num_layers

    if layer_idx is not None:
        # Extract single layer
        if layer_idx >= num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range. LSTM has {num_layers} layers.")

        weight_ih = getattr(lstm, f"weight_ih_l{layer_idx}").data.clone()
        weight_hh = getattr(lstm, f"weight_hh_l{layer_idx}").data.clone()
        bias_ih = getattr(lstm, f"bias_ih_l{layer_idx}").data.clone()
        bias_hh = getattr(lstm, f"bias_hh_l{layer_idx}").data.clone()

        # Determine input size for this layer
        layer_input_size = lstm.input_size if layer_idx == 0 else lstm.hidden_size

        return {
            "weight_ih": weight_ih,
            "weight_hh": weight_hh,
            "bias_ih": bias_ih,
            "bias_hh": bias_hh,
            "input_size": layer_input_size,
            "hidden_size": lstm.hidden_size,
        }
    else:
        # Extract all layers
        weights_list = []

        for layer_idx in range(num_layers):
            weight_ih = getattr(lstm, f"weight_ih_l{layer_idx}").data.clone()
            weight_hh = getattr(lstm, f"weight_hh_l{layer_idx}").data.clone()
            bias_ih = getattr(lstm, f"bias_ih_l{layer_idx}").data.clone()
            bias_hh = getattr(lstm, f"bias_hh_l{layer_idx}").data.clone()

            # Determine input size for this layer
            layer_input_size = lstm.input_size if layer_idx == 0 else lstm.hidden_size

            weights_list.append(
                {
                    "weight_ih": weight_ih,  # [4*hidden_size, input_size] for layer 0, [4*hidden_size, hidden_size] for others
                    "weight_hh": weight_hh,  # [4*hidden_size, hidden_size]
                    "bias_ih": bias_ih,  # [4*hidden_size]
                    "bias_hh": bias_hh,  # [4*hidden_size]
                    "input_size": layer_input_size,
                    "hidden_size": lstm.hidden_size,
                }
            )

        return weights_list


def create_test_inputs(
    batch_size: int = 2,
    seq_len: int = 10,
    input_size: int = 640,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Create test inputs for LSTM comparison."""
    torch.manual_seed(seed)

    # Create input sequence: [seq_len, batch_size, input_size] (PyTorch LSTM format)
    x = torch.randn(seq_len, batch_size, input_size, dtype=torch.float32)

    return {
        "x": x,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "input_size": input_size,
    }


def test_lstm_dropout_accuracy(
    device,
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
    batch_size: int = 1,
    seq_len: int = 1,
    pcc_threshold: float = 0.99,
    use_pretrained: bool = True,
):
    """
    Test accuracy between NeMo LSTMDropout and TT_LSTMDropout.

    Args:
        device: TTNN device
        model_name: HuggingFace model name or checkpoint path
        batch_size: Batch size for test inputs
        seq_len: Sequence length
        pcc_threshold: Minimum PCC threshold for test to pass
        use_pretrained: If True, load pretrained weights; if False, use random weights
    """
    print("=" * 80)
    print("LSTMDropout Accuracy Comparison Test")
    print("=" * 80)

    if use_pretrained:
        # 1. Load pretrained NeMo LSTMDropout
        print("\n1. Loading pretrained NeMo LSTMDropout...")
        nemo_lstm = load_pretrained_nemo_lstm(model_name=model_name)

        # Extract weights for ALL layers
        print("\n2. Extracting LSTM weights for all layers...")
        weights_list = extract_lstm_weights_from_nemo(nemo_lstm, layer_idx=None)
        num_layers = len(weights_list)
        input_size = weights_list[0]["input_size"]
        hidden_size = weights_list[0]["hidden_size"]

        print(f"   ✅ Weights extracted for {num_layers} layers:")
        print(f"      - Input size: {input_size}")
        print(f"      - Hidden size: {hidden_size}")
        for layer_idx, weights in enumerate(weights_list):
            print(f"      - Layer {layer_idx}:")
            print(f"        * Weight IH shape: {weights['weight_ih'].shape}")
            print(f"        * Weight HH shape: {weights['weight_hh'].shape}")
    else:
        # Use random weights for testing
        print("\n1. Initializing NeMo LSTMDropout with random weights...")
        input_size = 640
        hidden_size = 640
        num_layers = 2
        torch.manual_seed(42)
        nemo_lstm = LSTMDropout(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=None,
            forget_gate_bias=1.0,
        )
        nemo_lstm.eval()

        weights_list = extract_lstm_weights_from_nemo(nemo_lstm, layer_idx=None)

    # 3. Initialize TTNN LSTM Cell with extracted weights for all layers
    print("\n3. Initializing TTNN LSTM Cell with all layers...")
    tt_lstm_cell = TtLSTMCell(
        input_size=input_size,
        hidden_size=hidden_size,
        device=device,
        num_layers=num_layers,
        dtype=ttnn.float32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weights=weights_list,  # Pass list of weights for all layers
    )
    print(f"   ✅ TTNN LSTM Cell initialized with {num_layers} layers")

    # 4. Create test inputs
    print("\n4. Creating test inputs...")
    inputs = create_test_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        input_size=input_size,
        seed=42,
    )
    print(f"   ✅ Inputs created:")
    print(f"      - x: {inputs['x'].shape} [seq_len, batch_size, input_size]")

    # 5. Run NeMo LSTM forward pass
    print("\n5. Running NeMo LSTM forward pass...")
    with torch.no_grad():
        # NeMo LSTMDropout expects [seq_len, batch_size, input_size]
        import pdb

        pdb.set_trace()
        nemo_output, nemo_states = nemo_lstm(inputs["x"])
        nemo_h, nemo_c = nemo_states

    print(f"   ✅ NeMo output shape: {nemo_output.shape} [seq_len, batch_size, hidden_size]")
    print(f"      - Hidden state shape: {nemo_h.shape} [num_layers, batch_size, hidden_size]")
    print(f"      - Cell state shape: {nemo_c.shape} [num_layers, batch_size, hidden_size]")

    # 6. Run TTNN LSTM forward pass (timestep by timestep)
    print("\n6. Running TTNN LSTM forward pass...")

    # Initialize states
    h_tt = ttnn.zeros(
        [num_layers, batch_size, hidden_size],
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    c_tt = ttnn.zeros(
        [num_layers, batch_size, hidden_size],
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Process sequence timestep by timestep
    tt_outputs = []
    for t in range(seq_len):
        # Convert input timestep to TTNN
        x_t = ttnn.from_torch(
            inputs["x"][t : t + 1].transpose(0, 1),  # [1, batch_size, input_size]
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Forward pass
        import pdb

        pdb.set_trace()
        output, (h_tt, c_tt) = tt_lstm_cell(x_t, h_tt, c_tt)
        tt_outputs.append(output)

    # Concatenate outputs
    tt_output = ttnn.concat(tt_outputs, dim=0)  # [seq_len, batch_size, hidden_size]
    # Convert TTNN output to PyTorch for comparison
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)
    tt_h_torch = ttnn.to_torch(h_tt).to(torch.float32)
    tt_c_torch = ttnn.to_torch(c_tt).to(torch.float32)

    print(f" ✅ TTNN output shape: {tt_output_torch.shape} [seq_len, batch_size, hidden_size]")
    print(f" ✅ TTNN hidden state shape: {tt_h_torch.shape} [num_layers, batch_size, hidden_size]")
    print(f" ✅ TTNN cell state shape: {tt_c_torch.shape} [num_layers, batch_size, hidden_size]")
    # 7. Compare outputs
    print("\n7. Comparing outputs...")

    # Compare sequence outputs
    nemo_output_flat = nemo_output.reshape(-1)
    tt_output_flat = tt_output_torch.reshape(-1)

    pcc_output = compute_pcc(tt_output_torch, nemo_output)
    max_error_output = compute_max_abs_error(tt_output_torch, nemo_output)
    mean_error_output = compute_mean_abs_error(tt_output_torch, nemo_output)

    print(f"   📊 Output Metrics:")
    print(f"      - PCC: {pcc_output:.6f}")
    print(f"      - Max Abs Error: {max_error_output:.6f}")
    print(f"      - Mean Abs Error: {mean_error_output:.6f}")

    if pcc_output >= pcc_threshold:
        print(f"   ✅ Output PCC >= {pcc_threshold} (PASS)")
    else:
        print(f"   ❌ Output PCC < {pcc_threshold} (FAIL)")

    # Compare final hidden states
    nemo_h_initial = nemo_h[0]  # [batch_size, hidden_size]
    pcc_h = compute_pcc(tt_h_torch[0].squeeze(0), nemo_h_initial)
    max_error_h = compute_max_abs_error(tt_h_torch.squeeze(0), nemo_h_initial)

    print(f"\n   📊 Hidden State Metrics:")
    print(f"      - PCC: {pcc_h:.6f}")
    print(f"      - Max Abs Error: {max_error_h:.6f}")

    if pcc_h >= pcc_threshold:
        print(f"   ✅ Hidden state PCC >= {pcc_threshold} (PASS)")
    else:
        print(f"   ❌ Hidden state PCC < {pcc_threshold} (FAIL)")

    # Compare final cell states
    nemo_c_final = nemo_c[layer_idx]  # [batch_size, hidden_size]
    pcc_c = compute_pcc(tt_c_torch[-1].squeeze(0), nemo_c_final)
    max_error_c = compute_max_abs_error(tt_c_torch.squeeze(0), nemo_c_final)

    print(f"\n   📊 Cell State Metrics:")
    print(f"      - PCC: {pcc_c:.6f}")
    print(f"      - Max Abs Error: {max_error_c:.6f}")

    if pcc_c >= pcc_threshold:
        print(f"   ✅ Cell state PCC >= {pcc_threshold} (PASS)")
    else:
        print(f"   ❌ Cell state PCC < {pcc_threshold} (FAIL)")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Output:     PCC = {pcc_output:.6f} {'✅' if pcc_output >= pcc_threshold else '❌'}")
    print(f"Hidden:      PCC = {pcc_h:.6f} {'✅' if pcc_h >= pcc_threshold else '❌'}")
    print(f"Cell:        PCC = {pcc_c:.6f} {'✅' if pcc_c >= pcc_threshold else '❌'}")

    all_passed = all(
        [
            pcc_output >= pcc_threshold,
            pcc_h >= pcc_threshold,
            pcc_c >= pcc_threshold,
        ]
    )

    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed!")

    return {
        "pcc_output": pcc_output,
        "pcc_hidden": pcc_h,
        "pcc_cell": pcc_c,
        "all_passed": all_passed,
    }


def main():
    """Main function to run LSTMDropout accuracy tests."""
    # Open TTNN device
    device = ttnn.open_device(device_id=0)

    try:
        # Run tests with pretrained model
        results = test_lstm_dropout_accuracy(
            device=device,
            model_name="nvidia/parakeet-tdt-0.6b-v2",
            batch_size=1,
            seq_len=1,
            pcc_threshold=0.99,
            use_pretrained=True,
        )
        # Exit with appropriate code
        exit(0 if results["all_passed"] else 1)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
