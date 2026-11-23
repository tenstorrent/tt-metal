# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ForcedAligner-0.6B on Tenstorrent: word timestamps for a known transcript.

The aligner is Qwen3-ASR with a smaller decoder (Qwen3-0.6B shape) and a classifier over
time bins in place of the LM head. One prefill, no decode: the prompt is the audio
followed by each word and two <timestamp> tokens, and the time bin predicted at a
<timestamp> token is that word boundary in units of timestamp_segment_time (80 ms).

It reuses the Qwen3-ASR audio encoder and on-device audio splice, and the shared
tt_transformers decoder, which returns every position's hidden state when no last token
is requested.
"""
import glob
import json
import os

import torch
from safetensors import safe_open

import ttnn
from models.demos.audio.qwen3_asr.reference.audio_encoder_ref import load_audio_tower_weights
from models.demos.audio.qwen3_asr.tt import audio_encoder as tt_enc
from models.demos.audio.qwen3_asr.tt.generator import Qwen3ASRGenerator
from models.demos.audio.qwen3_asr.tt.qwen3_asr_decoder import Qwen3ASRDecoder
from models.tt_transformers.tt.common import Mode, get_padded_prefill_len
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
CLASSIFIER_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
)
TOKENIZER_FILES = ["merges.txt", "vocab.json", "tokenizer_config.json"]


def find_aligner_snap():
    """The downloaded Qwen3-ForcedAligner-0.6B snapshot dir."""
    base = os.environ.get("QWEN3ALIGNER_SNAP_BASE") or os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots"
    )
    snaps = sorted(glob.glob(os.path.join(base, "*", "config.json")))
    if not snaps:
        raise FileNotFoundError(f"no {ALIGNER_REPO} snapshot under {base}")
    return os.path.dirname(snaps[0])


def extract_text_decoder(snap, out_dir):
    """Write the aligner's decoder as a plain Qwen3 checkpoint tt_transformers can load. Its
    lm_head is the token embedding: tt_transformers needs one, and the aligner never runs it."""
    from safetensors.torch import save_file

    text = json.load(open(os.path.join(snap, "config.json")))["thinker_config"]["text_config"]
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "rope_theta",
        "max_position_embeddings",
        "rms_norm_eps",
    )
    config = dict(
        architectures=["Qwen3ForCausalLM"],
        model_type="qwen3",
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        **{k: text[k] for k in keys},
    )
    sd = {}
    with safe_open(os.path.join(snap, "model.safetensors"), "pt") as h:
        for k in h.keys():
            if k.startswith("thinker.model."):
                sd["model." + k[len("thinker.model.") :]] = h.get_tensor(k)
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
    os.makedirs(out_dir, exist_ok=True)
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    for fn in TOKENIZER_FILES:
        with open(os.path.join(snap, fn), "rb") as src, open(os.path.join(out_dir, fn), "wb") as dst:
            dst.write(src.read())
    return out_dir


def find_text_decoder(snap):
    out_dir = os.environ.get("QWEN3ALIGNER_TEXT_DECODER") or os.path.expanduser("~/qwen3_forced_aligner_text_decoder")
    if os.path.isfile(os.path.join(out_dir, "model.safetensors")):
        return out_dir
    return extract_text_decoder(snap, out_dir)


class Qwen3ForcedAligner:
    def __init__(self, model, model_args, mesh_device, encoder_params, classifier, config):
        self.model = model
        self.args = model_args
        self.mesh_device = mesh_device
        self.encoder_params = encoder_params
        self.classifier = classifier
        self.classify_num = config["thinker_config"]["classify_num"]
        self.timestamp_token_id = config["timestamp_token_id"]
        self.timestamp_segment_time = config["timestamp_segment_time"]

    @classmethod
    def build(cls, mesh_device, max_seq_len=2048, snap_dir=None):
        snap_dir = snap_dir or find_aligner_snap()
        config = json.load(open(os.path.join(snap_dir, "config.json")))
        # ModelArgs takes its checkpoint only from HF_MODEL.
        os.environ["HF_MODEL"] = find_text_decoder(snap_dir)
        args = ModelArgs(
            mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            optimizations=lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
        )
        model = Qwen3ASRDecoder(
            args,
            ttnn.bfloat16,
            mesh_device,
            args.load_state_dict(),
            args.weight_cache_path(ttnn.bfloat16),
        )
        encoder_params = tt_enc.preprocess_weights(
            load_audio_tower_weights(snap_dir=snap_dir, dtype=torch.float32), mesh_device
        )
        tt_enc.warmup(encoder_params, mesh_device)
        with safe_open(os.path.join(snap_dir, "model.safetensors"), "pt") as h:
            weight = h.get_tensor("thinker.lm_head.weight").float()
        classifier = ttnn.from_torch(
            weight.t().contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        return cls(model, args, mesh_device, encoder_params, classifier, config)

    def timestamp_logits(self, input_ids, mel):
        """Time-bin logits (n_timestamps, classify_num), float, at each <timestamp> token.

        input_ids: (S,) aligner prompt with the audio placeholders expanded. mel: (128, T)
        features of the audio. The prompt is padded to a prefill length tt_transformers
        supports; the padding is causally masked from every real position."""
        if self.args.num_devices != 1:
            raise NotImplementedError("the aligner runs on one device")
        S = input_ids.shape[0]
        S_pad = get_padded_prefill_len(S)
        if S_pad > self.args.max_seq_len:
            raise ValueError(f"aligner prompt of {S} tokens exceeds max_seq_len {self.args.max_seq_len}")
        ((off, length),) = Qwen3ASRGenerator.audio_runs(input_ids)
        audio = tt_enc.encode_mel(mel.float(), self.encoder_params, self.mesh_device, to_host=False)
        if length != tt_enc.feat_out_len(mel.shape[-1]):
            raise ValueError(f"{length} audio placeholders for {tt_enc.feat_out_len(mel.shape[-1])} encoder outputs")
        tokens = torch.nn.functional.pad(input_ids, (0, S_pad - S)).reshape(1, -1)
        self.model.set_prefill_audio([[(off, length, audio)]])
        try:
            x, rot_global, rot_local = self.model.prepare_inputs_prefill(tokens, page_table=None)[:3]
        finally:
            self.model.set_prefill_audio(None)
            ttnn.deallocate(audio)
        hidden = self.model.ttnn_prefill_forward(
            x, rot_mats_global=rot_global, rot_mats_local=rot_local, user_id=0, get_last_token=-1
        )
        hidden = self.model.norm(
            hidden, mode=Mode.PREFILL, norm_config=self.args.get_norm_config("lm_head", Mode.PREFILL, None)
        )
        logits = ttnn.linear(hidden, self.classifier, compute_kernel_config=CLASSIFIER_KERNEL_CONFIG)
        ttnn.deallocate(hidden)
        out = ttnn.to_torch(logits).reshape(S_pad, -1)[:S, : self.classify_num].float()
        ttnn.deallocate(logits)
        return out[input_ids == self.timestamp_token_id]

    def timestamps_ms(self, input_ids, mel):
        """Raw boundary times in ms, two per word (start, end), before the monotonic fix-up
        the qwen-asr post-processing applies."""
        return self.timestamp_logits(input_ids, mel).argmax(-1) * self.timestamp_segment_time
