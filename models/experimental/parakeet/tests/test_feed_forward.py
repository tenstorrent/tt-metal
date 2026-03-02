import torch
import gc
import ttnn

from nemo.collections.asr.parts.submodules.conformer_modules import ConformerFeedForward
from nemo.collections.asr.models import EncDecRNNTBPEModel

from models.common.metrics import (
    compute_pcc,
    compute_max_abs_error,
    compute_mean_abs_error,
)

from models.experimental.parakeet.tt.ttnn_conf_layer import TtConformerFeedForward


# -----------------------------------------------------------------------------
# TTNN Parameter Container
# -----------------------------------------------------------------------------


class TTParameters:
    def __init__(self, linear1_weight, linear2_weight, device):
        self.linear1 = type("", (), {})()
        self.linear2 = type("", (), {})()

        self.linear1.weight = ttnn.from_torch(
            linear1_weight.transpose(0, 1).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.linear2.weight = ttnn.from_torch(
            linear2_weight.transpose(0, 1).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------


def main():
    # -------------------------------------------------------------------------
    # Load NeMo model
    # -------------------------------------------------------------------------

    model_name = "nvidia/parakeet-tdt-0.6b-v2"

    print(f"\nLoading model: {model_name}")

    asr_model = EncDecRNNTBPEModel.from_pretrained(
        model_name,
        map_location="cpu",
    )

    asr_model.eval()

    # -------------------------------------------------------------------------
    # Extract FeedForward module
    # -------------------------------------------------------------------------

    ff_module = asr_model.encoder.layers[0].feed_forward1

    if not isinstance(ff_module, ConformerFeedForward):
        raise RuntimeError("Invalid module type")

    linear1_weight = ff_module.linear1.weight.detach()
    linear2_weight = ff_module.linear2.weight.detach()

    # -------------------------------------------------------------------------
    # Prepare input
    # -------------------------------------------------------------------------

    batch_size = 2
    seq_len = 32
    d_model = ff_module.d_model

    x = torch.randn(batch_size, seq_len, d_model)

    # -------------------------------------------------------------------------
    # Run NeMo reference
    # -------------------------------------------------------------------------

    with torch.no_grad():
        nemo_out = ff_module(x)

    # -------------------------------------------------------------------------
    # TTNN execution
    # -------------------------------------------------------------------------

    device = ttnn.open_device(device_id=0)

    try:
        tt_ff = TtConformerFeedForward(
            device=device,
            dtype=ttnn.bfloat16,
        )

        params = TTParameters(
            linear1_weight,
            linear2_weight,
            device,
        )

        x_tt = ttnn.from_torch(
            x.to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        tt_out_tensor = tt_ff(x_tt, params)

        tt_out = ttnn.to_torch(tt_out_tensor)

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        print("\n📊 Output Metrics:")

        pcc = compute_pcc(tt_out, nemo_out)
        max_err = compute_max_abs_error(tt_out, nemo_out)
        mean_err = compute_mean_abs_error(tt_out, nemo_out)

        print(f"PCC:            {pcc:.6f}")
        print(f"Max Abs Error:  {max_err:.6f}")
        print(f"Mean Abs Error: {mean_err:.6f}")

        if pcc >= 0.99:
            print("\n✅ PASS")
        else:
            print("\n❌ FAIL")

        print("\nShapes:")

        print("NeMo:", nemo_out.shape)
        print("TTNN:", tt_out.shape)

    finally:
        # ---------------------------------------------------------------------
        # CRITICAL CLEANUP SECTION
        # ---------------------------------------------------------------------

        print("\nCleaning up...")

        del tt_out_tensor
        del x_tt
        del params
        del tt_ff

        gc.collect()

        ttnn.close_device(device)

        print("Device closed cleanly.")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
