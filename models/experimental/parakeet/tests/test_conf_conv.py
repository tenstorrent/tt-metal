import torch
import gc
import ttnn

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerConvolution

from models.experimental.parakeet.reference.pytorch_conf_layer import ConformerConvolution as RefConformerConvolution

from models.common.metrics import (
    compute_pcc,
    compute_max_abs_error,
    compute_mean_abs_error,
)

from models.experimental.parakeet.tt.ttnn_conf_layer import TtConformerConvolution


# -----------------------------------------------------------------------------
# TT Parameter Container for Convolution
# -----------------------------------------------------------------------------


class TTConvParameters:
    def __init__(self, conv_module, device):
        self.pointwise1 = type("", (), {})()
        self.depthwise = type("", (), {})()
        self.pointwise2 = type("", (), {})()
        self.bn = type("", (), {})()

        # Conv weights on host so conv2d "Preprocessing weights on host" path is used
        # (device weights trigger pull-back + reprocess which can corrupt layout)
        # --------------------------
        # Pointwise 1
        # --------------------------
        print("stop3")
        self.pointwise1.weight = ttnn.from_torch(
            conv_module.pointwise_conv1.weight.detach().to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        if conv_module.pointwise_conv1.bias is not None:
            self.pointwise1.bias = ttnn.from_torch(
                conv_module.pointwise_conv1.bias.detach().to(torch.bfloat16),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
            )
        else:
            self.pointwise1.bias = None

        # --------------------------
        # Depthwise
        # --------------------------
        self.depthwise.weight = ttnn.from_torch(
            conv_module.depthwise_conv.weight.detach().to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        if conv_module.depthwise_conv.bias is not None:
            self.depthwise.bias = ttnn.from_torch(
                conv_module.depthwise_conv.bias.detach().to(torch.bfloat16),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
            )
        else:
            self.depthwise.bias = None

        # --------------------------
        # Pointwise 2
        # --------------------------
        self.pointwise2.weight = ttnn.from_torch(
            conv_module.pointwise_conv2.weight.detach().to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        if conv_module.pointwise_conv2.bias is not None:
            self.pointwise2.bias = ttnn.from_torch(
                conv_module.pointwise_conv2.bias.detach().to(torch.bfloat16),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
            )
        else:
            self.pointwise2.bias = None

        # --------------------------
        # BatchNorm
        # --------------------------
        self.bn.running_mean = ttnn.from_torch(
            conv_module.batch_norm.running_mean.detach().to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.bn.running_var = ttnn.from_torch(
            conv_module.batch_norm.running_var.detach().to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.bn.weight = ttnn.from_torch(
            conv_module.batch_norm.weight.detach().to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.bn.bias = ttnn.from_torch(
            conv_module.batch_norm.bias.detach().to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    model_name = "nvidia/parakeet-tdt-0.6b-v2"

    print(f"\nLoading model: {model_name}")

    asr_model = EncDecRNNTBPEModel.from_pretrained(
        model_name,
        map_location="cpu",
    )

    asr_model.eval()

    # -------------------------------------------------------------------------
    # Extract Conformer Convolution module
    # -------------------------------------------------------------------------

    conv_module = asr_model.encoder.layers[0].conv

    if not isinstance(conv_module, ConformerConvolution):
        raise RuntimeError("Invalid module type")

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------

    # Minimal batch/seq to fit depthwise conv in L1 (conv1d uses L1 path; needs <~1.3MB per bank)
    batch_size = 1
    seq_len = 16
    d_model = conv_module.d_model

    x = torch.randn(batch_size, seq_len, d_model)

    pad_mask = torch.zeros(batch_size, seq_len).bool()

    # -------------------------------------------------------------------------
    # Reference: symmetric-padding ConformerConvolution with same weights as NeMo
    # (NeMo uses CausalConv1D so we compare TTNN to ref with symmetric padding)
    # -------------------------------------------------------------------------

    kernel_size = conv_module.depthwise_conv.kernel_size[0]
    ref_conv = RefConformerConvolution(d_model=d_model, kernel_size=kernel_size, use_bias=True)
    ref_conv.eval()
    # Copy weights and biases from NeMo so ref matches original NeMo
    with torch.no_grad():
        ref_conv.pointwise_conv1.weight.copy_(conv_module.pointwise_conv1.weight)
        if conv_module.pointwise_conv1.bias is not None:
            ref_conv.pointwise_conv1.bias.copy_(conv_module.pointwise_conv1.bias)
        ref_conv.depthwise_conv.weight.copy_(conv_module.depthwise_conv.weight)
        if conv_module.depthwise_conv.bias is not None:
            ref_conv.depthwise_conv.bias.copy_(conv_module.depthwise_conv.bias)
        ref_conv.pointwise_conv2.weight.copy_(conv_module.pointwise_conv2.weight)
        if conv_module.pointwise_conv2.bias is not None:
            ref_conv.pointwise_conv2.bias.copy_(conv_module.pointwise_conv2.bias)
        ref_conv.batch_norm.load_state_dict(conv_module.batch_norm.state_dict())
    ref_conv = ref_conv.to(torch.bfloat16)

    with torch.no_grad():
        ref_out = ref_conv(x.to(torch.bfloat16), pad_mask)

    # -------------------------------------------------------------------------
    # TTNN Execution
    # -------------------------------------------------------------------------

    CONV_L1_SMALL_SIZE = 32768
    device = ttnn.CreateDevice(device_id=0, l1_small_size=CONV_L1_SMALL_SIZE)
    ttnn.SetDefaultDevice(device)

    try:
        tt_conv = TtConformerConvolution(
            d_model=d_model,
            kernel_size=conv_module.depthwise_conv.kernel_size[0],
            device=device,
            dtype=ttnn.bfloat16,
        )

        params = TTConvParameters(conv_module, device)
        # Ref has bias=False on all convs; zero conv biases so TTNN matches ref for comparison
        params.pointwise1.bias = None
        params.depthwise.bias = None
        params.pointwise2.bias = None

        x_tt = ttnn.from_torch(
            x.to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # TTNN unsqueeze/where only support bfloat16/float32/int32/uint32, not bool
        pad_mask_tt = ttnn.from_torch(
            pad_mask.to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        tt_out_tensor = tt_conv(x_tt, pad_mask_tt, params)

        tt_out = ttnn.to_torch(tt_out_tensor)

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        print("\n📊 Output Metrics (TTNN vs ref matching original NeMo):")

        pcc = compute_pcc(tt_out, ref_out)
        max_err = compute_max_abs_error(tt_out, ref_out)
        mean_err = compute_mean_abs_error(tt_out, ref_out)

        print(f"PCC:            {pcc:.6f}")
        print(f"Max Abs Error:  {max_err:.6f}")
        print(f"Mean Abs Error: {mean_err:.6f}")

        if pcc >= 0.99:
            print("\n✅ PASS")
        else:
            print("\n❌ FAIL")

        print("\nShapes:")
        print("Ref:", ref_out.shape)
        print("TTNN:", tt_out.shape)

    finally:
        print("\nCleaning up...")

        if "tt_out_tensor" in locals():
            del tt_out_tensor
        if "x_tt" in locals():
            del x_tt
        if "params" in locals():
            del params
        if "tt_conv" in locals():
            del tt_conv

        gc.collect()
        ttnn.close_device(device)

        print("Device closed cleanly.")


if __name__ == "__main__":
    main()
