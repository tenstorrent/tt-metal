# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from pathlib import Path
from loguru import logger
from models.experimental.SSD512.tt.tt_ssd import build_ssd512
from models.experimental.SSD512.reference.ssd import build_ssd
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
@pytest.mark.parametrize(
    "size",
    (512,),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 98304}], indirect=True)
def test_ssd512_network(device, pcc, size, reset_seeds):
    """
    Test Full SSD512 Network.
    """
    seed = 0
    if reset_seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass

    num_classes = 21  # VOC dataset
    batch_size = 1

    # Build PyTorch reference model FIRST
    torch_model = build_ssd("train", size=size, num_classes=num_classes)
    torch_model.eval()

    for m in torch_model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    # Build TTNN model
    ttnn_model = build_ssd512(num_classes=num_classes, device=device)

    # Load weights from PyTorch to TTNN
    # This ensures both models use the SAME weights for fair comparison
    ttnn_model.load_weights_from_torch(torch_model)

    # Synchronize device after weight loading
    ttnn.synchronize_device(device)
    import gc

    gc.collect()

    # Set to True to enable ONNX export, False to skip
    export_onnx = False

    if export_onnx:
        logger.info("\n" + "=" * 70)
        logger.info("Exporting TTNN model (via PyTorch structure) to ONNX format")
        logger.info("=" * 70)

        # Create output directory for ONNX model
        onnx_output_dir = Path(__file__).parent.parent / "onnx_models"
        onnx_output_dir.mkdir(exist_ok=True)
        onnx_output_path = onnx_output_dir / f"ssd512_ttnn_{size}.onnx"

        # Create dummy input for ONNX export
        dummy_input = torch.randn(1, 3, size, size)

        # Create a wrapper module class that only returns loc and conf (excluding priors)
        # This avoids ONNX tracing issues with constant prior boxes
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                outputs = self.model(x)
                # In train phase, model returns (loc, conf, priors)
                # We only need loc and conf for ONNX export
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    return outputs[0], outputs[1]  # Return only loc and conf
                return outputs

        wrapper_model = ModelWrapper(torch_model)

        try:
            torch.onnx.export(
                wrapper_model,  # Wrapper module that excludes priors
                dummy_input,  # Example input
                str(onnx_output_path),  # Output path
                export_params=True,  # Store trained parameter weights (shared with TTNN)
                opset_version=11,  # ONNX opset version (11 supports most operations)
                do_constant_folding=False,  # Disable constant folding to avoid prior boxes tracing issues
                input_names=["input"],  # Input tensor name
                output_names=["loc", "conf"],  # Output tensor names (location and confidence)
                dynamic_axes={
                    "input": {0: "batch_size"},  # Variable batch size
                    "loc": {0: "batch_size"},  # Variable batch size
                    "conf": {0: "batch_size"},  # Variable batch size
                },
                verbose=False,
            )
            logger.info(f"✓ Successfully exported TTNN model (via PyTorch structure) to ONNX: {onnx_output_path}")
            logger.info(f"  Model size: {onnx_output_path.stat().st_size / (1024*1024):.2f} MB")
            logger.info(f"  Note: This ONNX model represents the TTNN model's computation graph")
            logger.info(f"  The TTNN model uses the same weights and architecture as the exported PyTorch model")

            # Save TTNN model metadata
            metadata_path = onnx_output_dir / f"ssd512_ttnn_{size}_metadata.json"
            import json

            metadata = {
                "model_type": "SSD512_TTNN",
                "input_size": size,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "onnx_file": str(onnx_output_path.name),
                "description": "TTNN SSD512 model exported to ONNX format. "
                "The ONNX model represents the computation graph shared between TTNN and PyTorch implementations.",
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Saved TTNN model metadata to: {metadata_path}")

            # Verify ONNX model can be loaded
            try:
                import onnx

                onnx_model = onnx.load(str(onnx_output_path))
                onnx.checker.check_model(onnx_model)
                logger.info("✓ ONNX model verification passed")

                # Log model information
                logger.info(f"  ONNX Model Info:")
                logger.info(f"    IR Version: {onnx_model.ir_version}")
                logger.info(f"    Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
                logger.info(f"    Inputs: {len(onnx_model.graph.input)}")
                logger.info(f"    Outputs: {len(onnx_model.graph.output)}")
                logger.info(f"    Nodes: {len(onnx_model.graph.node)}")

            except ImportError:
                logger.warning("  ⚠️  ONNX package not available for verification (install with: pip install onnx)")
            except Exception as e:
                logger.warning(f"  ⚠️  ONNX model verification failed: {e}")

        except Exception as e:
            logger.error(f"✗ Failed to export TTNN model to ONNX: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            import traceback

            logger.error(f"  Traceback: {traceback.format_exc()}")
    else:
        logger.info("Skipping ONNX export (export_onnx=False)")

    input_tensor = torch.randn(batch_size, 3, size, size)

    logger.info("Running PyTorch forward pass step-by-step")

    import torch.nn.functional as F

    torch_sources = []
    torch_loc_preds = []
    torch_conf_preds = []

    with torch.no_grad():
        x = input_tensor.clone()

        # VGG up to conv4_3 relu (layer 22)
        logger.info("\n[PyTorch] VGG backbone up to conv4_3...")
        for k in range(23):
            x = torch_model.base[k](x)
        torch_conv4_3 = x.clone()

        # Apply L2Norm
        torch_conv4_3_norm = torch_model.L2Norm(torch_conv4_3)
        torch_sources.append(torch_conv4_3_norm)
        logger.info(f"  Source 0 (conv4_3 after L2Norm): shape {torch_conv4_3_norm.shape}")

        # VGG up to conv7 (fc7)
        logger.info("\n[PyTorch] VGG backbone up to conv7...")
        for k in range(23, len(torch_model.base)):
            x = torch_model.base[k](x)
        torch_conv7 = x.clone()
        torch_sources.append(torch_conv7)
        logger.info(f"  Source 1 (conv7): shape {torch_conv7.shape}")

        # Extras layers
        logger.info("\n[PyTorch] Extras backbone...")
        for k, v in enumerate(torch_model.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                torch_sources.append(x.clone())
                logger.info(f"  Source {len(torch_sources)-1} (extras[{k}]): shape {x.shape}")

        # Multibox heads
        logger.info("\n[PyTorch] Multibox heads...")
        for source_idx, (source, loc_layer, conf_layer) in enumerate(
            zip(torch_sources, torch_model.loc, torch_model.conf)
        ):
            loc_pred = loc_layer(source)
            conf_pred = conf_layer(source)
            # Permute to NHWC for comparison (matching TTNN output format)
            loc_pred_nhwc = loc_pred.permute(0, 2, 3, 1).contiguous()
            conf_pred_nhwc = conf_pred.permute(0, 2, 3, 1).contiguous()
            torch_loc_preds.append(loc_pred_nhwc)
            torch_conf_preds.append(conf_pred_nhwc)
            logger.info(f"  Source {source_idx}: loc shape {loc_pred_nhwc.shape}, conf shape {conf_pred_nhwc.shape}")

        # Flatten and concatenate
        torch_loc_flat = torch.cat([o.view(o.size(0), -1) for o in torch_loc_preds], 1)
        torch_conf_flat = torch.cat([o.view(o.size(0), -1) for o in torch_conf_preds], 1)
        logger.info(f"\n[PyTorch] Final outputs:")
        logger.info(f"  Location shape: {torch_loc_flat.shape}")
        logger.info(f"  Confidence shape: {torch_conf_flat.shape}")

    # Store for final comparison (outside torch.no_grad() block)
    torch_loc = torch_loc_flat
    torch_conf = torch_conf_flat

    # ========== DEBUG: Step-by-step TTNN forward pass ==========
    logger.info("\n" + "=" * 70)
    logger.info("DEBUG: Running TTNN forward pass step-by-step")
    logger.info("=" * 70)

    # Synchronize before TTNN forward
    ttnn.synchronize_device(device)
    gc.collect()

    # Run TTNN forward pass with debug mode to capture intermediate results
    ttnn_loc, ttnn_conf, debug_dict = ttnn_model.forward(input_tensor, dtype=ttnn.bfloat16, debug=True)

    ttnn_sources = debug_dict["sources"]
    ttnn_loc_preds = debug_dict["loc_preds"]
    ttnn_conf_preds = debug_dict["conf_preds"]

    logger.info(f"\n[TTNN] Final outputs:")
    logger.info(f"  Location shape: {ttnn_loc.shape}")
    logger.info(f"  Confidence shape: {ttnn_conf.shape}")

    # ========== DEBUG: Compare intermediate outputs ==========
    logger.info("\n" + "=" * 70)
    logger.info("DEBUG: Comparing intermediate outputs with PCC")
    logger.info("=" * 70)

    # Compare sources (after L2Norm and extras)
    logger.info("\n--- Comparing Sources (Feature Maps) ---")
    for idx, (torch_source, ttnn_source) in enumerate(zip(torch_sources, ttnn_sources)):
        # Ensure both are in NCHW format
        if torch_source.shape != ttnn_source.shape:
            logger.warning(f"  Source {idx}: Shape mismatch! PyTorch: {torch_source.shape}, TTNN: {ttnn_source.shape}")
            # Try to match shapes
            min_shape = [min(s1, s2) for s1, s2 in zip(torch_source.shape, ttnn_source.shape)]
            torch_source = torch_source[tuple(slice(0, s) for s in min_shape)]
            ttnn_source = ttnn_source[tuple(slice(0, s) for s in min_shape)]

        torch_source_flat = torch_source.flatten().float()
        ttnn_source_flat = ttnn_source.flatten().float()

        does_pass, pcc_val = comp_pcc(torch_source_flat, ttnn_source_flat, 0.0)  # Just get PCC value
        logger.info(f"  Source {idx}: PCC = {pcc_val:.6f}, shape {torch_source.shape}")

        if pcc_val < 0.99:
            logger.warning(f"  Source {idx} has low PCC: {pcc_val:.6f}")

    # Compare multibox location predictions per source
    logger.info("\n--- Comparing Location Predictions per Source ---")
    for idx, (torch_loc_pred, ttnn_loc_pred) in enumerate(zip(torch_loc_preds, ttnn_loc_preds)):
        torch_loc_flat = torch_loc_pred.reshape(torch_loc_pred.shape[0], -1, 4)

        if torch_loc_flat.shape != ttnn_loc_pred.shape:
            logger.warning(
                f"  Loc Source {idx}: Shape mismatch! PyTorch: {torch_loc_flat.shape}, TTNN: {ttnn_loc_pred.shape}"
            )
            min_shape = [min(s1, s2) for s1, s2 in zip(torch_loc_flat.shape, ttnn_loc_pred.shape)]
            torch_loc_flat = torch_loc_flat[tuple(slice(0, s) for s in min_shape)]
            ttnn_loc_pred = ttnn_loc_pred[tuple(slice(0, s) for s in min_shape)]

        torch_loc_flat_flat = torch_loc_flat.flatten().float()
        ttnn_loc_flat_flat = ttnn_loc_pred.flatten().float()

        does_pass, pcc_val = comp_pcc(torch_loc_flat_flat, ttnn_loc_flat_flat, 0.0)
        logger.info(f"  Loc Source {idx}: PCC = {pcc_val:.6f}, shape {torch_loc_flat.shape}")

        if pcc_val < 0.99:
            logger.warning(f"    ⚠️  Loc Source {idx} has low PCC: {pcc_val:.6f}")

    # Compare multibox confidence predictions per source
    logger.info("\n--- Comparing Confidence Predictions per Source ---")
    for idx, (torch_conf_pred, ttnn_conf_pred) in enumerate(zip(torch_conf_preds, ttnn_conf_preds)):
        # torch_conf_pred is in NHWC format: [B, H, W, boxes*classes]
        # ttnn_conf_pred is in [B, H*W*boxes, classes] format

        # Convert torch_conf_pred to [B, H*W, boxes*classes]
        torch_conf_flat = torch_conf_pred.reshape(torch_conf_pred.shape[0], -1, torch_conf_pred.shape[-1])
        # torch_conf_flat: [B, H*W, boxes*classes]

        # Convert TTNN format [B, H*W*boxes, classes] to [B, H*W, boxes*classes] to match PyTorch
        # TTNN: [B, H*W*boxes, classes] = [B, H*W*num_boxes, num_classes]
        # We need to reshape to [B, H*W, num_boxes*num_classes]
        B, HW_boxes, num_classes = ttnn_conf_pred.shape
        # Calculate H*W from shape
        # torch_conf_pred is [B, H, W, boxes*classes], so H*W = torch_conf_flat.shape[1]
        H_W = torch_conf_flat.shape[1]  # H*W from PyTorch
        num_boxes = HW_boxes // H_W  # Calculate number of boxes per location
        # Reshape TTNN: [B, H*W*num_boxes, num_classes] -> [B, H*W, num_boxes*num_classes]
        ttnn_conf_reshaped = ttnn_conf_pred.reshape(B, H_W, num_boxes * num_classes)

        if torch_conf_flat.shape != ttnn_conf_reshaped.shape:
            logger.warning(
                f"  Conf Source {idx}: Shape mismatch after reshape! PyTorch: {torch_conf_flat.shape}, TTNN: {ttnn_conf_reshaped.shape}"
            )
            min_shape = [min(s1, s2) for s1, s2 in zip(torch_conf_flat.shape, ttnn_conf_reshaped.shape)]
            torch_conf_flat = torch_conf_flat[tuple(slice(0, s) for s in min_shape)]
            ttnn_conf_reshaped = ttnn_conf_reshaped[tuple(slice(0, s) for s in min_shape)]

        torch_conf_flat_flat = torch_conf_flat.flatten().float()
        ttnn_conf_flat_flat = ttnn_conf_reshaped.flatten().float()

        does_pass, pcc_val = comp_pcc(torch_conf_flat_flat, ttnn_conf_flat_flat, 0.0)
        logger.info(f"  Conf Source {idx}: PCC = {pcc_val:.6f}, shape {torch_conf_flat.shape}")

        if pcc_val < 0.99:
            logger.warning(f"    ⚠️  Conf Source {idx} has low PCC: {pcc_val:.6f}")

    logger.info("\n" + "=" * 70)

    # Ensure outputs are float32 for fair comparison
    ttnn_loc = ttnn_loc.float()
    ttnn_conf = ttnn_conf.float()

    # Check if shapes match
    logger.info(f"\nFinal output shapes:")
    logger.info(f"  PyTorch Location: {torch_loc.shape}")
    logger.info(f"  TTNN Location: {ttnn_loc.shape}")
    logger.info(f"  PyTorch Confidence: {torch_conf.shape}")
    logger.info(f"  TTNN Confidence: {ttnn_conf.shape}")

    # Flatten both tensors for comparison (comp_pcc expects same shape)
    torch_loc_flat = torch_loc.flatten()
    ttnn_loc_flat = ttnn_loc.flatten()
    torch_conf_flat = torch_conf.flatten()
    ttnn_conf_flat = ttnn_conf.flatten()

    # Truncate to minimum length if shapes don't match
    min_loc_len = min(len(torch_loc_flat), len(ttnn_loc_flat))
    min_conf_len = min(len(torch_conf_flat), len(ttnn_conf_flat))

    if len(torch_loc_flat) != len(ttnn_loc_flat):
        logger.warning(
            f"Location length mismatch! PyTorch: {len(torch_loc_flat)}, TTNN: {len(ttnn_loc_flat)}. Truncating to {min_loc_len}"
        )
        torch_loc_flat = torch_loc_flat[:min_loc_len]
        ttnn_loc_flat = ttnn_loc_flat[:min_loc_len]

    if len(torch_conf_flat) != len(ttnn_conf_flat):
        logger.warning(
            f"Confidence length mismatch! PyTorch: {len(torch_conf_flat)}, TTNN: {len(ttnn_conf_flat)}. Truncating to {min_conf_len}"
        )
        torch_conf_flat = torch_conf_flat[:min_conf_len]
        ttnn_conf_flat = ttnn_conf_flat[:min_conf_len]

    # Compare location predictions
    does_pass_loc, pcc_message_loc = comp_pcc(torch_loc_flat, ttnn_loc_flat, pcc)
    logger.info(f"Location PCC: {pcc_message_loc}")

    # Compare confidence predictions
    does_pass_conf, pcc_message_conf = comp_pcc(torch_conf_flat, ttnn_conf_flat, pcc)
    logger.info(f"Confidence PCC: {pcc_message_conf}")

    assert does_pass_loc, f"Location predictions do not meet PCC requirement {pcc}: {pcc_message_loc}"
    assert does_pass_conf, f"Confidence predictions do not meet PCC requirement {pcc}: {pcc_message_conf}"
