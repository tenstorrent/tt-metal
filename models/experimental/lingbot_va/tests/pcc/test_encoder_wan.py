# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Compare HuggingFace UMT5EncoderModel with TT-Metal UMT5Encoder.
Uses Pearson Correlation Coefficient (PCC) as the accuracy metric.
"""

import torch
import ttnn
from transformers import UMT5EncoderModel

from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder as TTUMT5Encoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.common.metrics import compute_pcc


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/text_encoder"
PCC_THRESHOLD = 0.99
BATCH_SIZE = 1
SEQ_LEN = 512  # adjust to your typical prompt length


# ─────────────────────────────────────────────
# Main test
# ─────────────────────────────────────────────


def test_umt5_encoder_comparison():
    """Compare TT-Metal UMT5Encoder with HuggingFace UMT5EncoderModel."""

    # ── 1. Load HuggingFace model (CPU to avoid CUDA→TTNN weight issues) ──
    print("Loading UMT5EncoderModel...")
    text_encoder = UMT5EncoderModel.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device="cpu")
    text_encoder.eval()  # keep on CPU deliberately

    # State dict on CPU — required for TTNN weight loading
    text_weights = {k: v.cpu() for k, v in text_encoder.state_dict().items()}

    # ── 2. Build a random tokenized input ──
    torch.manual_seed(42)
    input_ids = torch.randint(
        low=0,
        high=text_encoder.config.vocab_size,
        size=(BATCH_SIZE, SEQ_LEN),
        dtype=torch.long,
    )
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)

    # ── 3. UMT5EncoderModel forward ──
    print("Running UMT5EncoderModel forward pass...")
    with torch.no_grad():
        text_out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    text_embed = text_out.last_hidden_state.float()  # [B, seq, d_model]

    # ── 4. Open TT-Metal mesh device ──
    print("Setting up TT-Metal mesh device...")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    try:
        # ── 5. Parallel / CCL config ──
        encoder_parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
        ccl_manager = CCLManager(
            mesh_device=mesh_device,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        # ── 6. Build TTNN UMT5 config from HF config ──
        umt5_config = UMT5Config(
            vocab_size=text_encoder.config.vocab_size,
            embed_dim=text_encoder.config.d_model,
            ff_dim=text_encoder.config.d_ff,
            kv_dim=text_encoder.config.d_kv,
            num_heads=text_encoder.config.num_heads,
            num_hidden_layers=text_encoder.config.num_layers,
            max_prompt_length=512,
            layer_norm_eps=text_encoder.config.layer_norm_epsilon,
            relative_attention_num_buckets=text_encoder.config.relative_attention_num_buckets,
            relative_attention_max_distance=text_encoder.config.relative_attention_max_distance,
        )

        # ── 7. Create and load weights into TTNN encoder ──
        print("Creating TT-Metal UMT5Encoder and loading weights...")
        tt_encoder = TTUMT5Encoder(
            config=umt5_config,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=encoder_parallel_config,
        )
        tt_encoder.load_torch_state_dict(text_weights)

        # ── 8. TTNN forward ──
        print("Running TT-Metal forward pass...")
        tt_input = ttnn.from_torch(
            input_ids,
            dtype=ttnn.uint32,
            device=mesh_device,
        )
        tt_mask = ttnn.from_torch(
            attention_mask,
            dtype=ttnn.uint32,
            device=mesh_device,
        )
        tt_out = tt_encoder(tt_input, attention_mask=tt_mask)  # returns ttnn.Tensor [B, seq, d_model]
        # import pdb; pdb.set_trace()
        # ── 9. Convert TTNN output to torch, fix shape ──
        tt_out = tt_out[-1]
        tt_embed = ttnn.to_torch(tt_out).float()

        # ttnn.to_torch can add extra leading dims — squeeze to [B, seq, d_model]
        while tt_embed.dim() > 3:
            tt_embed = tt_embed.squeeze(0)

        assert tt_embed.shape == text_embed.shape, f"Shape mismatch: HF={text_embed.shape}, TT={tt_embed.shape}"

        # ── 10. Compute PCC ──
        pcc = compute_pcc(text_embed, tt_embed)
        max_err = (text_embed - tt_embed).abs().max().item()
        mean_err = (text_embed - tt_embed).abs().mean().item()

        # ── 11. Report ──
        print("\n" + "=" * 55)
        print("COMPARISON RESULTS")
        print("=" * 55)
        print(f"HuggingFace output shape : {text_embed.shape}")
        print(f"TT-Metal    output shape : {tt_embed.shape}")
        print(f"PCC                      : {pcc:.6f}  (threshold={PCC_THRESHOLD})")
        print(f"Max  absolute error      : {max_err:.6f}")
        print(f"Mean absolute error      : {mean_err:.6f}")
        print("=" * 55)

        passed = pcc >= PCC_THRESHOLD
        if passed:
            print("✅ TEST PASSED")
        else:
            print(f"❌ TEST FAILED  — PCC {pcc:.6f} < {PCC_THRESHOLD}")

    finally:
        ttnn.close_mesh_device(mesh_device)

    return passed, pcc, max_err, mean_err


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        passed, pcc, max_err, mean_err = test_umt5_encoder_comparison()
        exit(0 if passed else 1)
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
