# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer PCC debug for Voxtral TTS text backbone on QB (TP=4) vs CPU reference.

Localizes where the multi-device tensor-parallel path diverges from the single-device
golden path. Run on the QuietBox (auto-opens a 1x4 mesh with fabric) and compare the
per-layer prefill hidden states against the CPU reference.

    python models/experimental/voxtraltts/demo/pcc_debug_tp.py

Set VOXTRAL_PCC_TEXT="..." to override the probe text.
"""

from __future__ import annotations

import os

# Trace must be off: the debug path reads per-layer hiddens to host (incompatible with
# the captured decode trace, and trace replay would also skip the host collection).
os.environ["VOXTRAL_DECODE_TRACE"] = "0"

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL
from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

MODEL = os.environ.get("VOXTRAL_PCC_MODEL", DEFAULT_VOXTRAL_MODEL)
VOICE = os.environ.get("VOXTRAL_PCC_VOICE", "casual_male")
TEXT = os.environ.get(
    "VOXTRAL_PCC_TEXT",
    "The quick brown fox jumps over the lazy dog.",
)


def _open_mesh():
    n_devices = ttnn.get_num_devices()
    if n_devices >= 4:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), physical_device_ids=[0, 1, 2, 3])
        logger.info("Opened 1x4 mesh (QB, TP=4) with FABRIC_1D")
    else:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0])
        logger.info("Opened 1x1 mesh (single chip)")
    return mesh


def _pcc(label: str, ref: torch.Tensor, tt: torch.Tensor) -> float:
    ref_f = ref.reshape(-1).float()
    tt_f = tt.reshape(-1).float()
    n = min(ref_f.numel(), tt_f.numel())
    _, msg = comp_pcc(ref_f[:n], tt_f[:n], pcc=0.0)
    try:
        val = float(msg)
    except (TypeError, ValueError):
        # comp_pcc returns a string like "PCC: 0.9988"; extract the trailing float.
        val = float(str(msg).strip().split()[-1])
    return val


def main() -> None:
    mesh = _open_mesh()
    n_devices = mesh.get_num_devices()
    pipe = None
    try:
        logger.info(f"Loading TT pipeline from {MODEL!r} (n_devices={n_devices}) ...")
        pipe = VoxtralTTSPipeline.from_model_name(
            mesh,
            model_name_or_path=MODEL,
            text_max_seq_len=512,
            use_paged_kv_cache=True,
        )
        logger.info("Loading CPU reference ...")
        cpu = VoxtralCPUReference(model_name_or_path=MODEL, dtype="bfloat16", device="cpu")

        request = compose_speech_request(TEXT, MODEL, voice=VOICE)
        prompt_ids = request["prompt_token_ids"]
        logger.info(f"Prompt length: {len(prompt_ids)} tokens")

        # CPU golden: per-layer last-token hidden states. hidden_states[0] is the embedding
        # output; hidden_states[i+1] is the output of decoder layer i.
        _, cpu_embeds = cpu._prompt_embeddings(prompt_ids, VOICE)
        cpu_out = cpu.text_model(
            inputs_embeds=cpu_embeds.unsqueeze(0),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cpu_layers = [h[:, -1, :].squeeze(0).float() for h in cpu_out.hidden_states]
        cpu_final = cpu_out.hidden_states[-1][:, -1, :].squeeze(0).float()
        logger.info(f"CPU produced {len(cpu_layers)} hidden states (incl. embedding)")

        # TT path: prefill with per-layer hidden collection on the last token.
        # The pipeline was built with use_paged_kv_cache=True, so attention's KV update
        # needs a page_table (same as the demo's prefill call) — without it paged_update_cache
        # sees a batch-mismatched cache and asserts.
        tt_embeds = pipe._build_voice_injected_embeds(prompt_ids, VOICE)
        page_table = pipe._build_page_table(len(prompt_ids))
        last_hidden_tt, tt_layers = pipe.text.prefill_from_embeds(
            tt_embeds, start_pos=0, collect_layer_hiddens=True, page_table=page_table
        )

        n_layers = sum(1 for k in tt_layers if k.startswith("layer.") and k != "layer.final_norm")
        logger.info(f"TT collected {n_layers} decoder-layer hiddens + final_norm")
        logger.info("=" * 72)
        logger.info("Per-layer prefill hidden PCC (TT layer.j vs CPU hidden_states[j+1]):")

        first_low = None
        for j in range(n_layers):
            tt_h = tt_layers[f"layer.{j}"].float()
            cpu_h = cpu_layers[j + 1] if (j + 1) < len(cpu_layers) else cpu_layers[-1]
            pcc_val = _pcc(f"layer.{j}", cpu_h, tt_h)
            status = "PASS" if pcc_val >= 0.99 else "LOW"
            if pcc_val < 0.99 and first_low is None:
                first_low = j
            logger.info(f"  layer.{j:02d}: PCC={pcc_val:.5f}  [{status}]")

        # Final (post-norm) hidden, the actual input handed to the acoustic model.
        tt_final = pipe.text.hidden_tt_to_torch(last_hidden_tt).float()
        if last_hidden_tt.is_allocated():
            ttnn.deallocate(last_hidden_tt)
        final_pcc = _pcc("final_norm", cpu_final, tt_final)
        logger.info("-" * 72)
        logger.info(f"  final post-norm hidden: PCC={final_pcc:.5f}")
        logger.info("=" * 72)
        if first_low is None:
            logger.info("All layers >= 0.99 PCC. TP=4 text path matches reference within tolerance.")
        else:
            logger.warning(
                f"First diverging layer: layer.{first_low}. "
                f"The op introducing the error is inside decoder layer {first_low} "
                f"(attention all_reduce, MLP all_reduce, or the distributed norm gather)."
            )
    finally:
        if pipe is not None:
            try:
                pipe.cleanup_all()
            except Exception as exc:
                logger.warning(f"pipe cleanup failed: {exc}")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
