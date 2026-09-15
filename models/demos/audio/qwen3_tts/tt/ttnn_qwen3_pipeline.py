# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Qwen3-TTS CustomVoice on device: text in, waveform out.

Joins the three device blocks with the two host pieces that were missing: the dual-track
prompt assembly and the autoregressive loop that drives them.

Per frame the loop does:

    talker(prompt so far)      -> hidden state, then codec_head -> codebook 0
    code predictor(hidden, cb0) -> codebooks 1..15               (15 inner steps)
    sum the 16 codebook embeddings + tts_pad -> one new prompt position

and when codebook 0 comes back as `codec_eos_token_id` the utterance is over and the
collected frames go to the codec decoder.

**The prefill, which is the part that has to be exactly right.** For CustomVoice in
non-streaming mode it is `n_text + 11` positions, each the sum of a text-track and a
codec-track embedding:

    3          role, `<|im_start|>assistant\\n`, text track only
    4          codec think / think_bos / language / think_eos, against tts_pad
    1          the speaker id, against tts_pad
    1          codec_pad, against tts_bos
    n_text + 1 the text tokens then tts_eos, each against codec_pad
    1          codec_bos, against tts_pad

Language and speaker both enter as single codec-vocabulary tokens, which is why neither
appears in the text stream. In non-streaming mode `trailing_text_hidden` is a constant
`tts_pad_embed`, so after the prefill the model gets no further text signal; that is
upstream's design and the regime where its sampler is known to wander.

**What runs where.** On device: the 28-layer talker, the 5-layer code predictor, the codec
decoder, and the `codec_head` projection. On host: tokenisation, the embedding gathers, the
small `text_projection` MLP over the prompt, sampling, and the quantizer's table lookup.
That is 2.03B of the parameters on device and a few million of lookups on host.

Correctness first, and the cost is real: there is no KV cache, so every step recomputes the
whole prefix. Cost grows as the square of the utterance length, which is fine for a short
sentence and needs a cache before anything long.
"""

import torch
import torch.nn.functional as F

import ttnn
from models.demos.audio.qwen3_tts import frontend
from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor import (
    TtCodePredictor,
    preprocess_code_predictor_parameters,
)
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec import TtCodecDecoder, preprocess_codec_parameters
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import TtTalker, preprocess_talker_parameters

# Upstream's fixed slices into the prompt: three leading ids of role, five trailing.
ROLE_IDS = 3
TAIL_IDS = 5


class HostEmbeddings:
    """The lookup tables and the small projection that the prompt is built from."""

    def __init__(self, dtype=torch.float32):
        talker = checkpoint.load_prefixed("talker.", dtype=dtype, strip=True)
        self.text_table = talker["model.text_embedding.weight"]
        self.codec_table = talker["model.codec_embedding.weight"]
        self.fc1 = (talker["text_projection.linear_fc1.weight"], talker["text_projection.linear_fc1.bias"])
        self.fc2 = (talker["text_projection.linear_fc2.weight"], talker["text_projection.linear_fc2.bias"])
        self.codec_head = talker["codec_head.weight"]

        predictor = checkpoint.load_prefixed("talker.code_predictor.model.codec_embedding.", dtype=dtype)
        self.predictor_tables = [predictor[f"{index}.weight"] for index in range(len(predictor))]

        special = frontend.special_tokens()
        # Upstream builds these three together and chunks them in this order.
        stacked = self.project(self.text_table[[special["tts_bos"], special["tts_eos"], special["tts_pad"]]])
        self.tts_bos, self.tts_eos, self.tts_pad = (row.reshape(1, 1, -1) for row in stacked)

    def project(self, embedded):
        """`text_projection`: linear_fc2(silu(linear_fc1(x))). Tiny, so it stays on host."""
        return F.linear(F.silu(F.linear(embedded, *self.fc1)), *self.fc2)

    def text(self, ids):
        """Text ids -> projected embeddings [1, n, 2048]."""
        return self.project(self.text_table[torch.as_tensor(ids, dtype=torch.long)]).reshape(1, -1, 2048)

    def codec(self, ids):
        """Codec ids -> embeddings [1, n, 2048], no projection."""
        return self.codec_table[torch.as_tensor(ids, dtype=torch.long)].reshape(1, -1, 2048)


def build_custom_voice_prefill(text, speaker, language, tables=None):
    """The dual-track prompt for CustomVoice, non-streaming.

    Returns (embeddings [1, P, 2048], prompt_ids) where P is `n_text + 11`.
    """
    tables = tables or HostEmbeddings()
    talker_config = checkpoint.talker_config()
    speakers = talker_config["spk_id"]
    if not speakers:
        raise ValueError("this checkpoint defines no speakers; CustomVoice needs the CustomVoice weights")
    key = str(speaker).strip().lower()
    if key not in speakers:
        raise ValueError(f"unknown speaker {speaker!r}; this checkpoint offers {', '.join(sorted(speakers))}")

    prompt_ids = frontend.text_ids(text)
    text_ids = prompt_ids[ROLE_IDS:-TAIL_IDS]
    language_id = frontend.language_id(language)

    think = [talker_config["codec_think_id"], talker_config["codec_think_bos_id"]]
    think += [] if language_id is None else [language_id]
    think += [talker_config["codec_think_eos_id"]]

    # Codec track: think block, speaker, then codec_pad and codec_bos.
    codec_track = torch.cat(
        [
            tables.codec(think),
            tables.codec([speakers[key]]),
            tables.codec([talker_config["codec_pad_id"], talker_config["codec_bos_id"]]),
        ],
        dim=1,
    )

    # Text track opposite it: pads under the think block and the speaker, tts_bos under
    # codec_pad. Upstream drops the final codec_bos here and re-adds it after the text.
    text_track = torch.cat([tables.tts_pad.expand(-1, codec_track.shape[1] - 2, -1), tables.tts_bos], dim=1)
    head = torch.cat([tables.text(prompt_ids[:ROLE_IDS]), text_track + codec_track[:, :-1]], dim=1)

    # The text itself, then tts_eos, each against codec_pad; then codec_bos against tts_pad.
    body = torch.cat([tables.text(text_ids), tables.tts_eos], dim=1) + tables.codec(
        [talker_config["codec_pad_id"]] * (len(text_ids) + 1)
    )
    tail = tables.tts_pad + tables.codec([talker_config["codec_bos_id"]])

    return torch.cat([head, body, tail], dim=1), prompt_ids


class Qwen3TTSPipeline:
    """The three device blocks plus the host loop that drives them."""

    def __init__(self, device, num_layers=None):
        self.device = device
        self.tables = HostEmbeddings()
        self.talker = TtTalker(device, preprocess_talker_parameters(device, num_layers=num_layers))
        self.predictor = TtCodePredictor(device, preprocess_code_predictor_parameters(device))
        self.codec = TtCodecDecoder(device, preprocess_codec_parameters(device))
        self.talker_config = checkpoint.talker_config()
        self.groups = self.talker_config["code_predictor_config"]["num_code_groups"]
        self.eos = self.talker_config["codec_eos_token_id"]
        self.codec_head = ttnn.from_torch(
            self.tables.codec_head.t().contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _to_device(self, tensor):
        return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _talker_step(self, embeddings):
        """Run the talker over the prompt so far; return the last hidden state and cb0 logits."""
        length = embeddings.shape[1]
        cos, sin, mask = self.talker.host_inputs(length)
        hidden = self.talker(*(self._to_device(t) for t in (embeddings, cos, sin, mask)))
        last = ttnn.slice(hidden, [0, length - 1, 0], [1, length, hidden.shape[2]])
        logits = ttnn.linear(last, self.codec_head)
        return ttnn.to_torch(last).float().reshape(1, 1, -1), ttnn.to_torch(logits).float().reshape(-1)

    def _frame_embedding(self, frame):
        """The 16 codebooks of one frame, summed, which becomes the next prompt position.

        Codebook 0 reads the talker's table, the rest the predictor's own, indexed per step.
        """
        total = self.tables.codec([frame[0]])
        for index, code in enumerate(frame[1:]):
            total = total + self.tables.predictor_tables[index][code].reshape(1, 1, -1)
        return total + self.tables.tts_pad

    def generate(self, text, speaker="ryan", language="English", max_frames=400, on_frame=None):
        """text -> (waveform [1, N] at 24 kHz, codes [frames, 16]).

        Greedy throughout. Upstream samples by default, and at its shipped
        `repetition_penalty` of 1.05 that occasionally loops for dozens of frames before
        speaking; greedy is both reproducible and free of that.
        """
        embeddings, _ = build_custom_voice_prefill(text, speaker, language, self.tables)
        frames = []

        for step in range(max_frames):
            hidden, logits = self._talker_step(embeddings)
            first = int(logits.argmax())
            if first == self.eos:
                break
            rest = self.predictor.generate(hidden, first)
            frame = [first] + list(rest)
            frames.append(frame)
            if on_frame is not None:
                on_frame(step, frame)
            embeddings = torch.cat([embeddings, self._frame_embedding(frame)], dim=1)

        if not frames:
            raise RuntimeError("the talker emitted end-of-speech before any frame")

        codes = torch.tensor(frames, dtype=torch.long)  # [frames, 16]
        waveform = self.codec.decode(codes.t().unsqueeze(0)).reshape(1, -1)
        return waveform, codes
