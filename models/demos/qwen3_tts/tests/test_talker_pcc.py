# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests comparing TT Talker implementation against PyTorch reference.

These tests require:
  - TT device access (P150 / N150 / N300)
  - HF model weights (downloaded automatically on first run)

Test hierarchy:
  1. test_single_layer_pcc:  single TransformerBlock, PCC >= 0.99
  2. test_full_prefill_pcc:  full 28-layer prefill, PCC >= 0.98
  3. test_decode_step_pcc:   single decode step, PCC >= 0.98
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

MODEL_ID = os.environ.get("HF_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")


def _get_reference_model(dtype=torch.float32):
    """Load or create the PyTorch reference Talker."""
    from models.demos.qwen3_tts.reference.talker_ref import TalkerReference

    model = TalkerReference.from_pretrained(MODEL_ID)
    model.eval()
    return model.to(dtype)


def _build_tt_talker(mesh_device, args_override=None):
    """Build TT Talker with real weights."""
    from models.demos.qwen3_tts.tt.model_config import TalkerModelArgs
    from models.demos.qwen3_tts.tt.talker import TalkerTransformer

    args = TalkerModelArgs(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=256,
        use_hf_rope=True,
    )
    if args_override:
        for k, v in args_override.items():
            setattr(args, k, v)

    state_dict = args.load_state_dict()
    weight_cache_path = args.weight_cache_path(ttnn.bfloat16)

    talker = TalkerTransformer(
        args=args,
        dtype=ttnn.bfloat16,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
    )
    return talker, args


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "P150": (1, 1)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
class TestTalkerPCC:
    """PCC comparison tests for the Talker."""

    def test_reference_loads(self, mesh_device):
        """Sanity: verify the reference model loads and produces output."""
        ref = _get_reference_model()
        tokens = torch.randint(0, ref.text_vocab_size, (1, 8))
        logits, hidden = ref.forward_prefill(tokens)
        assert logits.shape == (1, 8, ref.codec_vocab_size)
        assert hidden.shape == (1, 8, ref.dim)
        logger.info("Reference model loaded and verified")

    def test_full_prefill_pcc(self, mesh_device):
        """Full 28-layer prefill: TT vs reference, PCC >= 0.98.

        Compares last-token logits from the TT Talker against the PyTorch reference.
        """
        ref = _get_reference_model()

        seq_len = 32
        tokens = torch.randint(0, ref.text_vocab_size, (1, seq_len))
        ref_logits, _ = ref.forward_prefill(tokens)
        ref_last_logits = ref_logits[:, -1, :]  # [1, vocab_size]

        tt_talker, args = _build_tt_talker(mesh_device)
        tt_logits_tt = tt_talker.prefill(tokens)

        last_token_idx = seq_len - 1
        tt_logits_torch = tt_talker.process_output_prefill(tt_logits_tt.cpu(), last_token_idx=last_token_idx % 32)
        tt_logits_torch = tt_logits_torch[: ref.codec_vocab_size]  # [vocab_size]

        pcc = comp_pcc(ref_last_logits.squeeze(0).float(), tt_logits_torch.float())
        logger.info(f"Full prefill PCC (last token logits): {pcc}")
        assert pcc >= 0.98, f"Full prefill PCC {pcc} < 0.98"

    def test_decode_step_pcc(self, mesh_device):
        """Single decode step after prefill: TT vs reference, PCC >= 0.98."""
        ref = _get_reference_model(dtype=torch.bfloat16)

        seq_len = 16
        tokens = torch.randint(0, ref.text_vocab_size, (1, seq_len))

        # Reference: prefill + one decode step
        ref.forward_prefill(tokens)
        kv_caches = ref.init_kv_caches(1, seq_len + 16, "cpu", dtype=torch.bfloat16)

        # Re-run prefill to populate KV caches
        x = ref.text_embedding(tokens)
        x = ref.text_projection_fc2(torch.nn.functional.silu(ref.text_projection_fc1(x)))
        cos = ref.rope_cos[:seq_len].unsqueeze(0)
        sin = ref.rope_sin[:seq_len].unsqueeze(0)
        mask = ref._causal_mask(seq_len, x.dtype, x.device)
        for i, layer in enumerate(ref.layers):
            x = layer(x, cos, sin, mask=mask, kv_cache=kv_caches[i], start_pos=0)

        # Decode one step
        codec_token = torch.randint(0, ref.codec_vocab_size, (1, 1))
        ref_decode_logits = ref.forward_decode(codec_token, kv_caches, start_pos=seq_len)
        ref_decode_logits = ref_decode_logits[:, -1, :]  # [1, vocab]

        # TT: prefill + decode
        tt_talker, args = _build_tt_talker(mesh_device)
        tt_talker.prefill(tokens)

        current_pos = torch.tensor([seq_len], dtype=torch.int64)
        padded_tokens = torch.nn.functional.pad(codec_token.squeeze(1), (0, args.max_batch_size - 1), value=0)
        padded_pos = torch.nn.functional.pad(current_pos, (0, args.max_batch_size - 1), value=0)

        tt_tokens, tt_pos, tt_rot_idxs, tt_page_table = tt_talker.prepare_inputs_decode(
            padded_tokens, padded_pos
        )
        tt_logits, _ = tt_talker.ttnn_decode_forward(
            tt_tokens, tt_pos, rot_mat_idxs=tt_rot_idxs, page_table=tt_page_table
        )

        tt_logits_torch = tt_talker.process_output_decode(tt_logits.cpu(), B=1)
        tt_logits_torch = tt_logits_torch[:, :, : ref.codec_vocab_size].squeeze()  # [vocab]

        pcc = comp_pcc(ref_decode_logits.float().squeeze(), tt_logits_torch.float())
        logger.info(f"Decode step PCC: {pcc}")
        assert pcc >= 0.98, f"Decode step PCC {pcc} < 0.98"
