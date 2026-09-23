# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Qwen3-TTS code predictor, the 15-step inner loop.

Reference: models/demos/audio/qwen3_tts/reference/qwen3_code_predictor_ref.py

The talker predicts codebook 0 of each frame; this produces the other 15. Five layers,
hidden 1024, attention in a 2048-wide head space (16 query heads over 8 key/value heads at
head_dim 128), intermediate 3072, vocabulary 2048. Identical at both model sizes.

Block boundary: embeddings [1, 16, talker hidden] in, logits [1, 15, 2048] out. Position i is read
by `lm_head[i-1]` to give codebook i, so the same weights serve a 2-position prefill
followed by 14 single-position steps, or one teacher-forced pass over all 16. This module
implements the teacher-forced pass and a greedy loop built on it.

**The decoder layers are the talker's, unchanged.** The two attention implementations differ
in exactly one line upstream: the talker rotates with interleaved MRoPE across three
position axes, this rotates with plain RoPE on one. Since both take cos and sin from the
host, the device code is identical and `TtTalkerDecoderLayer` is reused directly. Only the
table builder differs, which is `plain_rotary_tables` below.

Correctness first. The greedy loop recomputes the whole prefix at every step rather than
carrying a KV cache, which is 15 passes over at most 16 positions of a 5-layer model. That
is wasteful and deliberate: a cache is decode-time work and belongs with the talker's.
"""

import torch

import ttnn
from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import TtTalkerDecoderLayer, _compute_config, causal_mask

CODE_PREDICTOR_PREFIX = "talker.code_predictor."
TALKER_CODEC_EMBEDDING = "talker.model.codec_embedding."


def plain_rotary_tables(config, length):
    """Standard RoPE cos and sin as [1, 1, T, head_dim].

    Mirrors `Qwen3TTSRotaryEmbedding.forward` followed by `apply_rotary_pos_emb`'s
    unsqueeze. No multi-axis selection here: `rope_scaling` is null for this config, so all
    channels advance with the one position sequence.
    """
    head_dim = config["head_dim"]
    theta = config["rope_theta"]

    inverse_frequencies = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(length, dtype=torch.float32).reshape(1, 1, length)
    frequencies = (inverse_frequencies.reshape(1, -1, 1) @ positions).transpose(1, 2)
    embedded = torch.cat((frequencies, frequencies), dim=-1)
    return embedded.cos().unsqueeze(1), embedded.sin().unsqueeze(1)


def preprocess_projection(state, device, dtype):
    """`small_to_mtp_projection` as (weight, bias), or (None, None) where it is an identity.

    The 1.7B talker is 2048 wide against the predictor's 1024 and ships the projection, the
    one layer here with a bias. At 0.6B both are 1024 and upstream builds `nn.Identity`.
    """
    if "small_to_mtp_projection.weight" not in state:
        return None, None
    weight = ttnn.from_torch(
        state["small_to_mtp_projection.weight"].t().contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias = ttnn.from_torch(
        state["small_to_mtp_projection.bias"].reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    return weight, bias


def project(x, parameters, compute_config):
    """The predictor's input projection, which is nothing at all when the widths match."""
    if parameters["projection"] is None:
        return x
    return ttnn.linear(
        x, parameters["projection"], bias=parameters["projection_bias"], compute_kernel_config=compute_config
    )


def preprocess_code_predictor_parameters(device, config=None, dtype=ttnn.bfloat16):
    """Checkpoint weights -> the tensors the predictor runs with."""
    talker_cfg = dict(config or checkpoint.talker_config())
    cfg = dict(talker_cfg["code_predictor_config"])
    state = checkpoint.load_prefixed(CODE_PREDICTOR_PREFIX)

    def linear(name):
        return ttnn.from_torch(
            state[f"{name}.weight"].t().contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def norm(name):
        return ttnn.from_torch(
            state[f"{name}.weight"].reshape(1, 1, 1, -1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    layers = []
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{index}"
        layers.append(
            {
                "input_layernorm": norm(f"{prefix}.input_layernorm"),
                "post_attention_layernorm": norm(f"{prefix}.post_attention_layernorm"),
                "q_proj": linear(f"{prefix}.self_attn.q_proj"),
                "k_proj": linear(f"{prefix}.self_attn.k_proj"),
                "v_proj": linear(f"{prefix}.self_attn.v_proj"),
                "o_proj": linear(f"{prefix}.self_attn.o_proj"),
                "q_norm": norm(f"{prefix}.self_attn.q_norm"),
                "k_norm": norm(f"{prefix}.self_attn.k_norm"),
                "gate_proj": linear(f"{prefix}.mlp.gate_proj"),
                "up_proj": linear(f"{prefix}.mlp.up_proj"),
                "down_proj": linear(f"{prefix}.mlp.down_proj"),
            }
        )

    heads = cfg["num_code_groups"] - 1
    projection, projection_bias = preprocess_projection(state, device, dtype)
    return {
        "config": cfg,
        "talker_config": talker_cfg,
        "layers": layers,
        "norm": norm("model.norm"),
        "projection": projection,
        "projection_bias": projection_bias,
        "lm_head": [linear(f"lm_head.{index}") for index in range(heads)],
        # Host-side tables: position 1 reads the talker's codec embedding, positions 2
        # onward the predictor's own, indexed per step. Kept on host because a greedy loop
        # indexes them one row at a time.
        "talker_codec_embedding": checkpoint.load_prefixed(TALKER_CODEC_EMBEDDING)["weight"],
        "codec_embedding": [
            checkpoint.load_prefixed(CODE_PREDICTOR_PREFIX + "model.codec_embedding.")[f"{index}.weight"]
            for index in range(heads)
        ],
    }


class TtCodePredictor:
    """Five layers, then one output head per codebook."""

    def __init__(self, device, parameters):
        self.device = device
        self.p = parameters
        self.config = parameters["config"]
        self.groups = self.config["num_code_groups"]
        self.compute_config = _compute_config(device)
        self.layers = [
            TtTalkerDecoderLayer(device, layer, self.config, self.compute_config) for layer in parameters["layers"]
        ]

    def host_inputs(self, length):
        """Rotation tables and mask for a sequence of this length, as torch tensors."""
        cos, sin = plain_rotary_tables(self.config, length)
        return cos, sin, causal_mask(length)

    def __call__(self, embeddings, cos, sin, mask, return_intermediates=False):
        """embeddings [1, T, talker hidden] -> hidden states [1, T, 1024], plus logits when T is full.

        Returns hidden states; `logits` turns them into per-codebook predictions.
        """
        length = embeddings.shape[1]
        intermediates = {}

        x = project(embeddings, self.p, self.compute_config)
        if return_intermediates:
            intermediates["projection"] = x

        for index, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask, length)
            if return_intermediates:
                intermediates[f"layers.{index}"] = x

        x = ttnn.rms_norm(x, weight=self.p["norm"], epsilon=self.config["rms_norm_eps"])
        if return_intermediates:
            intermediates["norm"] = x
            return x, intermediates
        return x

    def logits(self, hidden, position):
        """Codebook logits from one position: `lm_head[position - 1]` reads position."""
        length = hidden.shape[1]
        row = ttnn.slice(hidden, [0, position, 0], [1, position + 1, hidden.shape[2]])
        return ttnn.linear(row, self.p["lm_head"][position - 1], compute_kernel_config=self.compute_config)

    def teacher_forced_logits(self, embeddings, cos, sin, mask):
        """All 15 codebook logits in one pass: [1, 15, 2048]."""
        hidden = self(embeddings, cos, sin, mask)
        rows = [self.logits(hidden, position) for position in range(1, self.groups)]
        return ttnn.concat(rows, dim=1)

    # ── the loop the model actually runs ────────────────────────────────────

    def build_embeddings(self, talker_hidden, codes):
        """Assemble the input sequence on host from a talker hidden state and known codes.

        `codes` holds the codebook ids already decided, starting at codebook 0. The result
        has one position per code plus the talker's hidden state at the front.
        """
        positions = [talker_hidden.reshape(1, 1, -1)]
        for index, code in enumerate(torch.as_tensor(codes, dtype=torch.long).reshape(-1).tolist()):
            table = self.p["talker_codec_embedding"] if index == 0 else self.p["codec_embedding"][index - 1]
            positions.append(table[code].reshape(1, 1, -1))
        return torch.cat(positions, dim=1)

    def generate(self, talker_hidden, first_code, pick=None):
        """Codebooks 1 to 15, from the talker's hidden state and codebook 0.

        `pick` maps one row of logits to an id, and defaults to argmax. See
        `sampling.sample` for why the shipped configuration does not use argmax here.

        Recomputes the prefix each step rather than carrying a KV cache. 15 passes over at
        most 16 positions of a 5-layer model, which is cheap enough to leave for later.
        """
        pick = pick or (lambda row: int(row.argmax()))
        codes = [int(first_code)]
        for step in range(self.groups - 1):
            embeddings = self.build_embeddings(talker_hidden, codes)
            length = embeddings.shape[1]
            cos, sin, mask = self.host_inputs(length)
            to_device = lambda tensor: ttnn.from_torch(
                tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            hidden = self(to_device(embeddings), to_device(cos), to_device(sin), to_device(mask))
            # The newest position is the last one, and step k reads it with lm_head[k].
            row = ttnn.slice(hidden, [0, length - 1, 0], [1, length, hidden.shape[2]])
            logits = ttnn.linear(row, self.p["lm_head"][step], compute_kernel_config=self.compute_config)
            codes.append(int(pick(ttnn.to_torch(logits).float().reshape(-1))))
        return codes[1:]
