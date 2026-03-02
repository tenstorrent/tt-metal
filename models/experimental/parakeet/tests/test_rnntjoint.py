"""
Test script to compare PCC between TtRNNTJoint and RNNTJoint.

This script:
1. Initializes both models with the same weights
2. Creates test inputs
3. Runs forward pass on both models
4. Compares outputs using Pearson Correlation Coefficient (PCC)
"""
import sys
import os

# Import RNNTJoint from nemo
from nemo.collections.asr.modules.rnnt import RNNTJoint
import torch
import ttnn

from models.experimental.parakeet.tt.tt_rnnjoint import TtRNNTJoint
from models.common.metrics import compute_pcc

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def load_pretrained_rnnt_joint(model_name: str = "nvidia/parakeet-tdt-0.6b-v2", map_location: str = "cpu") -> RNNTJoint:
    """
    Load pretrained NeMo ASR model and extract the RNNTJoint module.

    Args:
        model_name: HuggingFace model name or path to checkpoint
        map_location: Device to load model on ('cpu' or 'cuda')

    Returns:
        RNNTJoint module with pretrained weights
    """
    import nemo.collections.asr as nemo_asr

    print(f"Loading pretrained model: {model_name}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=map_location)
    asr_model.eval()

    # Extract the joint module from the decoder
    # NeMo ASR models typically have the structure: decoder -> decoder_joint
    # if hasattr(asr_model, 'decoder') and hasattr(asr_model.decoder, 'decoder_joint'):
    #     joint_module = asr_model.decoder.decoder_joint.joint
    # elif hasattr(asr_model, 'decoder_joint'):
    #     joint_module = asr_model.decoder_joint.joint
    # else:
    # Try to find joint module by searching
    for name, module in asr_model.named_modules():
        if isinstance(module, RNNTJoint):
            joint_module = module
            break
    else:
        raise ValueError("Could not find RNNTJoint module in the loaded model")

    print(f"✅ Found RNNTJoint module")
    print(f"   - Encoder hidden: {joint_module.encoder_hidden}")
    print(f"   - Pred hidden: {joint_module.pred_hidden}")
    print(f"   - Joint hidden: {joint_module.joint_hidden}")
    print(f"   - Num classes: {joint_module._num_classes}")

    return joint_module


def load_weights_from_torch_to_tt(torch_joint: RNNTJoint, tt_joint: TtRNNTJoint):
    """Load weights from PyTorch RNNTJoint to TtRNNTJoint."""
    # Extract weights from PyTorch model
    pred_weight = torch_joint.pred.weight.data.clone()  # [joint_hidden, pred_hidden]
    pred_bias = torch_joint.pred.bias.data.clone()  # [joint_hidden]

    enc_weight = torch_joint.enc.weight.data.clone()  # [joint_hidden, enc_hidden]
    enc_bias = torch_joint.enc.bias.data.clone()  # [joint_hidden]

    # Extract from joint_net Sequential: [ReLU, Linear]
    joint_linear = torch_joint.joint_net[-1]  # Last layer is Linear
    joint_weight = joint_linear.weight.data.clone()  # [num_classes, joint_hidden]
    joint_bias = joint_linear.bias.data.clone()  # [num_classes]

    # Transpose weights to match TTNN format: [in_features, out_features]
    # PyTorch: [out_features, in_features] -> TTNN: [in_features, out_features]
    pred_weight_tt = pred_weight.transpose(0, 1)  # [pred_hidden, joint_hidden] = [640, 640]
    enc_weight_tt = enc_weight.transpose(0, 1)  # [encoder_hidden, joint_hidden] = [1024, 640]
    joint_weight_tt = joint_weight.transpose(0, 1)  # [joint_hidden, num_classes] = [640, 1030]

    # Update TTNN model weights
    tt_joint.pred_weight = pred_weight_tt
    tt_joint.pred_bias = pred_bias
    tt_joint.enc_weight = enc_weight_tt
    tt_joint.enc_bias = enc_bias
    tt_joint.joint_weight = joint_weight_tt
    tt_joint.joint_bias = joint_bias

    # Recreate TTNN layers with new weights
    tt_joint.pred = tt_joint._create_linear(tt_joint.pred["in_features"], tt_joint.pred["out_features"])
    tt_joint.enc = tt_joint._create_linear(tt_joint.enc["in_features"], tt_joint.enc["out_features"])
    tt_joint.joint_net = tt_joint._create_joint_net(
        tt_joint.joint_net["in_features"],
        tt_joint.joint_net["out_features"],
    )


def create_test_inputs(batch_size=2, encoder_seq_len=100, decoder_seq_len=50, seed=42):
    """Create test inputs for both models."""
    torch.manual_seed(seed)

    # RNNTJoint expects: encoder (B, D, T), decoder (B, D, U)
    encoder_outputs_torch = torch.randn(batch_size, encoder_seq_len, 1024)
    decoder_outputs_torch = torch.randn(batch_size, decoder_seq_len, 640)

    # TtRNNTJoint expects: encoder (B, T, D), decoder (B, U, D)
    encoder_outputs_tt = encoder_outputs_torch
    decoder_outputs_tt = decoder_outputs_torch

    return {
        "torch_encoder": encoder_outputs_torch,
        "torch_decoder": decoder_outputs_torch,
        "tt_encoder": encoder_outputs_tt,
        "tt_decoder": decoder_outputs_tt,
    }


def test_rnnt_joint_pcc(
    device,
    batch_size=2,
    encoder_seq_len=100,
    decoder_seq_len=50,
    use_pretrained: bool = False,
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
    seed=42,
    pcc_threshold=0.99,
):
    """
    Test PCC between TtRNNTJoint and RNNTJoint.

    Args:
        device: TTNN device
        batch_size: Batch size for test inputs
        encoder_seq_len: Encoder sequence length (T)
        decoder_seq_len: Decoder sequence length (U)
        use_pretrained: If True, load pretrained weights; if False, use random weights
        model_name: HuggingFace model name or checkpoint path
        seed: Random seed for reproducibility (only used if use_pretrained=False)
        pcc_threshold: Minimum PCC threshold for test to pass
    """
    print("=" * 80)
    print("RNNTJoint PCC Test")
    print("=" * 80)

    if use_pretrained:
        # Load pretrained model and extract joint module
        print("\n1. Loading pretrained RNNTJoint...")
        torch_joint = load_pretrained_rnnt_joint(model_name=model_name)

        # Extract dimensions from the loaded model
        encoder_hidden = torch_joint.encoder_hidden
        pred_hidden = torch_joint.pred_hidden
        joint_hidden = torch_joint.joint_hidden
        num_classes = torch_joint._num_classes - 1  # Subtract blank token

        print(f"   ✅ Pretrained model loaded")
        print(f"      - Encoder hidden: {encoder_hidden}")
        print(f"      - Pred hidden: {pred_hidden}")
        print(f"      - Joint hidden: {joint_hidden}")
        print(f"      - Num classes: {num_classes}")
    else:
        # Use random weights (original behavior)
        torch.manual_seed(seed)
        encoder_hidden = 1024
        pred_hidden = 640
        joint_hidden = 640
        num_classes = 1029

        print("\n1. Initializing PyTorch RNNTJoint with random weights...")
        jointnet_config = {
            "encoder_hidden": encoder_hidden,
            "pred_hidden": pred_hidden,
            "joint_hidden": joint_hidden,
            "activation": "relu",
        }

        torch_joint = RNNTJoint(
            jointnet=jointnet_config,
            num_classes=num_classes,
            log_softmax=False,
        )
        print(f"   ✅ PyTorch model initialized")

    torch_joint.log_softmax = False
    torch_joint.eval()

    if not next(torch_joint.parameters()).is_cuda:
        # Force log_softmax to False to prevent auto-application on CPU
        torch_joint.log_softmax = False

    # 2. Initialize TTNN TtRNNTJoint model
    print("\n2. Initializing TTNN TtRNNTJoint...")
    tt_joint = TtRNNTJoint(
        device=device,
        encoder_hidden=encoder_hidden,
        pred_hidden=pred_hidden,
        joint_hidden=joint_hidden,
        num_classes=num_classes,
        dtype=ttnn.float32,
    )
    print(f"   ✅ TTNN model initialized")

    # 3. Load weights from PyTorch to TTNN
    print("\n3. Loading weights from PyTorch to TTNN...")
    load_weights_from_torch_to_tt(torch_joint, tt_joint)
    print(f"   ✅ Weights loaded")

    # 4. Create test inputs
    print("\n4. Creating test inputs...")
    inputs = create_test_inputs(
        batch_size=batch_size, encoder_seq_len=encoder_seq_len, decoder_seq_len=decoder_seq_len, seed=seed
    )
    print(f"   ✅ Inputs created:")
    print(f"      - Encoder: {inputs['torch_encoder'].shape}")
    print(f"      - Decoder: {inputs['torch_decoder'].shape}")

    # 5. Run PyTorch forward pass
    print("\n5. Running PyTorch forward pass...")
    with torch.no_grad():
        torch_encoder_output = torch_joint.project_encoder(inputs["torch_encoder"])
        torch_decoder_output = torch_joint.project_prednet(inputs["torch_decoder"])
        torch_output = torch_joint.joint_after_projection(torch_encoder_output, torch_decoder_output)
    print(f"   ✅ PyTorch output shape: {torch_output.shape}")

    # 6. Convert inputs to TTNN format and run TTNN forward pass
    print("\n6. Running TTNN forward pass...")
    encoder_tt = ttnn.from_torch(
        inputs["tt_encoder"],
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    decoder_tt = ttnn.from_torch(
        inputs["tt_decoder"],
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_output = tt_joint.forward(encoder_tt, decoder_tt)

    # Convert TTNN output back to PyTorch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)
    print(f"   ✅ TTNN output shape: {tt_output_torch.shape}")

    # 7. Compute PCC
    print("\n7. Computing PCC...")
    pcc_value = compute_pcc(tt_output_torch, torch_output)
    print(f"   ✅ PCC: {pcc_value:.6f}")

    # 8. Print comparison statistics
    print("\n8. Comparison Statistics:")
    print(f"   - PyTorch output shape: {torch_output.shape}")
    print(f"   - TTNN output shape: {tt_output_torch.shape}")
    print(f"   - Mean absolute error: {torch.mean(torch.abs(tt_output_torch - torch_output)).item():.6f}")
    print(f"   - Max absolute error: {torch.max(torch.abs(tt_output_torch - torch_output)).item():.6f}")

    print("\n5.5. Diagnostic: Comparing intermediate outputs...")
    with torch.no_grad():
        # Get PyTorch intermediate outputs
        torch_enc_proj = torch_joint.project_encoder(inputs["torch_encoder"])
        torch_pred_proj = torch_joint.project_prednet(inputs["torch_decoder"])

        print(
            f"   PyTorch enc_proj shape: {torch_enc_proj.shape}, range: [{torch_enc_proj.min():.3f}, {torch_enc_proj.max():.3f}]"
        )
        print(
            f"   PyTorch pred_proj shape: {torch_pred_proj.shape}, range: [{torch_pred_proj.min():.3f}, {torch_pred_proj.max():.3f}]"
        )

    # Get TTNN intermediate outputs
    tt_enc_proj = tt_joint.project_encoder(encoder_tt)
    tt_pred_proj = tt_joint.project_prednet(decoder_tt)

    tt_enc_proj_torch = ttnn.to_torch(tt_enc_proj)
    tt_pred_proj_torch = ttnn.to_torch(tt_pred_proj)

    print(
        f"   TTNN enc_proj shape: {tt_enc_proj_torch.shape}, range: [{tt_enc_proj_torch.min():.3f}, {tt_enc_proj_torch.max():.3f}]"
    )
    print(
        f"   TTNN pred_proj shape: {tt_pred_proj_torch.shape}, range: [{tt_pred_proj_torch.min():.3f}, {tt_pred_proj_torch.max():.3f}]"
    )

    # Compare intermediate outputs
    enc_proj_pcc = compute_pcc(tt_enc_proj_torch, torch_enc_proj)
    pred_proj_pcc = compute_pcc(tt_pred_proj_torch, torch_pred_proj)
    print(f"   Encoder projection PCC: {enc_proj_pcc:.6f}")
    print(f"   Prediction projection PCC: {pred_proj_pcc:.6f}")

    # Check weight values
    print("\n5.6. Diagnostic: Checking weight values...")
    print(f"   PyTorch enc.weight shape: {torch_joint.enc.weight.shape}")
    print(f"   PyTorch enc.weight range: [{torch_joint.enc.weight.min():.6f}, {torch_joint.enc.weight.max():.6f}]")
    print(f"   TTNN enc.weight shape: {tt_joint.enc['weight'].shape}")
    tt_enc_weight_torch = ttnn.to_torch(tt_joint.enc["weight"])
    print(f"   TTNN enc.weight range: [{tt_enc_weight_torch.min():.6f}, {tt_enc_weight_torch.max():.6f}]")

    print("\n5.7. Diagnostic: Checking joint_net computation...")

    # Get activated tensor from both implementations
    with torch.no_grad():
        torch_f = torch_joint.project_encoder(inputs["torch_encoder"])
        torch_g = torch_joint.project_prednet(inputs["torch_decoder"])
        torch_f_exp = torch_f.unsqueeze(2)  # [B, T, 1, 640]
        torch_g_exp = torch_g.unsqueeze(1)  # [B, 1, U, 640]
        torch_joint_hidden = torch_f_exp + torch_g_exp  # [B, T, U, 640]
        torch_activated = torch.relu(torch_joint_hidden)

        print(
            f"   PyTorch activated shape: {torch_activated.shape}, range: [{torch_activated.min():.3f}, {torch_activated.max():.3f}]"
        )

        # Get TTNN activated tensor
    tt_f = tt_joint.project_encoder(encoder_tt)
    tt_g = tt_joint.project_prednet(decoder_tt)
    tt_f_exp = ttnn.unsqueeze(tt_f, dim=2)
    tt_g_exp = ttnn.unsqueeze(tt_g, dim=1)
    tt_joint_hidden = ttnn.add(tt_f_exp, tt_g_exp)
    tt_activated = ttnn.relu(tt_joint_hidden)

    tt_activated_torch = ttnn.to_torch(tt_activated)
    print(
        f"   TTNN activated shape: {tt_activated_torch.shape}, range: [{tt_activated_torch.min():.3f}, {tt_activated_torch.max():.3f}]"
    )

    activated_pcc = compute_pcc(tt_activated_torch, torch_activated)
    print(f"   Activated tensor PCC: {activated_pcc:.6f}")

    # Check joint_net weight
    print("\n5.8. Diagnostic: Checking joint_net weights...")
    joint_linear = torch_joint.joint_net[-1]
    print(f"   PyTorch joint.weight shape: {joint_linear.weight.shape}")
    print(f"   PyTorch joint.weight range: [{joint_linear.weight.min():.6f}, {joint_linear.weight.max():.6f}]")

    tt_joint_weight_torch = ttnn.to_torch(tt_joint.joint_net["weight"])
    print(f"   TTNN joint.weight shape: {tt_joint_weight_torch.shape}")
    print(f"   TTNN joint.weight range: [{tt_joint_weight_torch.min():.6f}, {tt_joint_weight_torch.max():.6f}]")

    # Verify joint weight transposition
    expected_joint_weight = joint_linear.weight.data.transpose(0, 1)  # [640, 1030]
    joint_weight_diff = torch.abs(tt_joint_weight_torch - expected_joint_weight)
    print(f"   Joint weight difference: mean={joint_weight_diff.mean():.6f}, max={joint_weight_diff.max():.6f}")

    # Test the final linear layer separately
    print("\n5.9. Diagnostic: Testing final linear layer...")
    with torch.no_grad():
        # PyTorch final linear
        torch_final_output = torch_joint.joint_net(torch_activated)
        print(
            f"   PyTorch final output shape: {torch_final_output.shape}, range: [{torch_final_output.min():.3f}, {torch_final_output.max():.3f}]"
        )

    # TTNN final linear
    tt_final_output = ttnn.linear(
        tt_activated,
        tt_joint.joint_net["weight"],
        bias=tt_joint.joint_net["bias"],
        memory_config=tt_joint.memory_config,
    )
    tt_final_output_torch = ttnn.to_torch(tt_final_output)
    print(
        f"   TTNN final output shape: {tt_final_output_torch.shape}, range: [{tt_final_output_torch.min():.3f}, {tt_final_output_torch.max():.3f}]"
    )

    final_pcc = compute_pcc(tt_final_output_torch, torch_final_output)
    print(f"   Final linear layer PCC: {final_pcc:.6f}")

    # Verify weight transposition is correct
    expected_enc_weight = torch_joint.enc.weight.data.transpose(0, 1)  # [1024, 640]
    weight_diff = torch.abs(tt_enc_weight_torch - expected_enc_weight)
    print(f"   Weight difference (should be ~0): mean={weight_diff.mean():.6f}, max={weight_diff.max():.6f}")
    # 9. Assert PCC threshold
    print(f"\n9. Validation:")
    if pcc_value >= pcc_threshold:
        print(f"   ✅ PASS: PCC ({pcc_value:.6f}) >= threshold ({pcc_threshold})")
        return True
    else:
        print(f"   ❌ FAIL: PCC ({pcc_value:.6f}) < threshold ({pcc_threshold})")
        return False


def main():
    """Main function to run the test."""
    # Initialize TTNN device
    print("Initializing TTNN device...")
    device = ttnn.open_device(device_id=0)
    print("✅ Device opened")

    try:
        # Run test with pretrained weights
        success = test_rnnt_joint_pcc(
            device=device,
            batch_size=1,
            encoder_seq_len=2,
            decoder_seq_len=1,
            use_pretrained=True,  # Set to True to use pretrained weights
            model_name="nvidia/parakeet-tdt-0.6b-v2",  # Or path to checkpoint
            pcc_threshold=0.99,
        )

        if success:
            print("\n" + "=" * 80)
            print("✅ TEST PASSED")
            print("=" * 80)
            return 0
        else:
            print("\n" + "=" * 80)
            print("❌ TEST FAILED")
            print("=" * 80)
            return 1

    finally:
        # Close device
        ttnn.close_device(device)
        print("\n✅ Device closed")


if __name__ == "__main__":
    exit(main())
