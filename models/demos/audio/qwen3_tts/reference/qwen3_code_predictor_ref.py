# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the Qwen3-TTS code predictor, the 15-step inner loop.

The talker emits one frame per step and predicts only codebook 0 of the 16 that frame
needs. This module produces the other 15, autoregressively, from the talker's hidden state
and that first code. It is small (5 layers, hidden 1024) and runs 15 times per talker step,
so the model spends 12.5 talker steps and 187.5 of these per second of audio.

How the sequence is built, from `Qwen3TTSForConditionalGeneration.forward` and
`Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward`:

    position 0    the talker's hidden state                       [2048]
    position 1    talker.codec_embedding(codebook 0)              [2048]
    position i    code_predictor.codec_embedding[i-2](codebook i-1)   for i in 2..15

Every position is 2048 wide and `small_to_mtp_projection` takes it to the predictor's 1024.
Reading position i with `lm_head[i-1]` gives the logits for codebook i, so the same weights
serve a 2-position prefill followed by 14 single-position steps, or one teacher-forced pass
over all 16 positions. The teacher-forced form is what the PCC test uses: one pass, every
weight exercised, no sampling in the way.

Note which tables are which. Position 1 uses the **talker's** codec embedding; positions 2
onward use the **predictor's** own, indexed per step. Mixing them up produces a model that
runs and is wrong.
"""

import functools

import torch

from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.reference.qwen import transformers_compat  # noqa: F401  (registers rope "default")
from models.demos.audio.qwen3_tts.reference.qwen.talker import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    Qwen3TTSTalkerConfig,
)

CODE_PREDICTOR_PREFIX = "talker.code_predictor."
TALKER_CODEC_EMBEDDING = "talker.model.codec_embedding."


class CodePredictorReference:
    """The upstream code predictor, loaded from the checkpoint and frozen in eval mode."""

    def __init__(self, config=None, dtype=torch.float32):
        talker_cfg = dict(config or weights.talker_config())
        predictor_cfg = dict(talker_cfg["code_predictor_config"])
        # transformers 4.57.3 created `pad_token_id` on every config; 5.12.1 only sets what
        # it is given, and the vendored stack reads the attribute.
        talker_cfg.setdefault("pad_token_id", None)
        predictor_cfg.setdefault("pad_token_id", None)

        self.config = predictor_cfg
        self.talker_config = talker_cfg
        self.groups = predictor_cfg["num_code_groups"]

        self.model = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            Qwen3TTSTalkerCodePredictorConfig(**predictor_cfg),
            Qwen3TTSTalkerConfig(**talker_cfg),
        )
        self.model.load_state_dict(weights.load_prefixed(CODE_PREDICTOR_PREFIX, dtype=dtype), strict=True)
        self.model.to(dtype).eval()

    @torch.inference_mode()
    def teacher_forced(self, embeddings, return_intermediates=False):
        """embeddings [1, 16, 2048] -> logits [1, 15, 2048], one per codebook 1..15."""
        if not return_intermediates:
            return self.model.forward_finetune(inputs_embeds=embeddings).logits

        captured = {}
        handles = []

        def capture(name):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[name] = tensor.detach().clone()

            return hook

        modules = dict(self.model.named_modules())
        for index in range(self.config["num_hidden_layers"]):
            handles.append(modules[f"model.layers.{index}"].register_forward_hook(capture(f"layers.{index}")))
        handles.append(modules["model.norm"].register_forward_hook(capture("norm")))
        handles.append(modules["small_to_mtp_projection"].register_forward_hook(capture("projection")))
        try:
            logits = self.model.forward_finetune(inputs_embeds=embeddings).logits
        finally:
            for handle in handles:
                handle.remove()
        return logits, captured

    @torch.inference_mode()
    def generate_greedy(self, talker_hidden, first_code):
        """Greedy decode of codebooks 1 to 15 from a talker hidden state and codebook 0.

        Greedy rather than sampled so the result is a fixed target a port can be held to.
        Production sampling (temperature 0.9, top-k 50) sits on top of these same logits.
        Recomputes the prefix each step, matching what the TTNN loop does.
        """
        codes = [int(first_code)]
        for step in range(self.groups - 1):
            embeddings = build_input_embeddings(talker_hidden, codes)
            projected = self.model.small_to_mtp_projection(embeddings)
            hidden = self.model.model(inputs_embeds=projected).last_hidden_state
            codes.append(int(self.model.lm_head[step](hidden[:, -1]).argmax()))
        return codes[1:]


@functools.lru_cache(maxsize=1)
def reference_model(dtype=torch.float32):
    return CodePredictorReference(dtype=dtype)


@functools.lru_cache(maxsize=1)
def _embedding_tables(dtype=torch.float32):
    talker_table = weights.load_prefixed(TALKER_CODEC_EMBEDDING, dtype=dtype)["weight"]
    predictor = weights.load_prefixed(CODE_PREDICTOR_PREFIX + "model.codec_embedding.", dtype=dtype)
    ordered = [predictor[f"{index}.weight"] for index in range(len(predictor))]
    return talker_table, ordered


def build_input_embeddings(talker_hidden, codes, dtype=torch.float32):
    """Assemble the sequence the predictor reads, one position per known code plus the
    talker's hidden state at the front.

    `talker_hidden` is [1, 1, 2048] from the talker; `codes` holds the codebook ids already
    decided, starting at codebook 0. The full teacher-forced sequence passes all 15 of
    codebooks 0 to 14; a partial one mid-decode passes fewer. Position 1 uses the talker's
    table, later positions the predictor's own, indexed per step.
    """
    talker_table, predictor_tables = _embedding_tables(dtype)
    codes = torch.as_tensor(codes, dtype=torch.long).reshape(-1)
    if not 1 <= codes.numel() <= len(predictor_tables):
        raise ValueError(f"expected 1 to {len(predictor_tables)} codes, got {codes.numel()}")

    positions = [talker_hidden.reshape(1, 1, -1), talker_table[codes[0]].reshape(1, 1, -1)]
    for step, code in enumerate(codes[1:]):
        positions.append(predictor_tables[step][code].reshape(1, 1, -1))
    return torch.cat(positions, dim=1)
