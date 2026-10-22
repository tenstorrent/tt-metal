# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Qwen3-TTS talker, the 28-layer decoder.

Reference: models/demos/audio/qwen3_tts/reference/qwen3_talker_ref.py

Block boundary: input = embeddings [1, T, 2048], output = hidden states [1, T, 2048]. The
talker sums a text stream and a codec stream into those embeddings; building that prompt is
a separate job, and this block starts once it exists.

Shape, read from `talker_config`: 28 layers, hidden 2048, 16 query heads over 8 key/value
heads, head_dim 128, intermediate 6144, SiLU, RMS eps 1e-6, RoPE theta 1e6, MRoPE sections
[24, 20, 20], interleaved.

Correctness first. Nothing here is traced, sharded or fused, attention is written out
rather than calling a fused kernel, and the whole prefill runs in one pass. Perf work comes
later and should not start until this holds PCC.

Two details separate this from a stock decoder port:

  1. **QK-norm.** Each layer carries `q_norm` and `k_norm` of width 128, an RMSNorm over
     each head's channels applied to queries and keys before the rotation. Miss it and the
     model still runs, quietly wrong.
  2. **Interleaved MRoPE.** Position ids carry three axes. The 64 rotary pairs are assigned
     across them by `mrope_section`, interleaved rather than in contiguous blocks. That
     assignment is a host-side selection, so `rotary_tables` below builds the final cos and
     sin and the device does the plain rotation. `tests/pcc/test_talker_pcc.py` pins the
     host helper against the reference's own function.
"""

import torch

import ttnn
from models.demos.audio.qwen3_tts import weights as checkpoint

# A large negative number rather than -inf: masked positions are added to scores before the
# softmax, and -inf produces NaN wherever a row is entirely masked.
MASK_FILL = -1e9


def _compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


# ── host-side tables ────────────────────────────────────────────────────────


def rotary_tables(config, position_ids):
    """Build the cos and sin the rotation needs, as [1, 1, T, head_dim].

    Mirrors `Qwen3TTSTalkerRotaryEmbedding.forward` followed by the interleaved branch of
    `apply_multimodal_rotary_pos_emb`. Both run on host: the tables depend only on position
    ids, and the selection across the three MRoPE axes is indexing rather than arithmetic.

    `position_ids` is [3, batch, T], one row per axis.
    """
    head_dim = config["head_dim"]
    theta = config["rope_theta"]
    sections = config["rope_scaling"]["mrope_section"]
    interleaved = config["rope_scaling"].get("interleaved", False)

    inverse_frequencies = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32) / head_dim)
    )
    expanded = inverse_frequencies[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
    positions = position_ids[:, :, None, :].float()
    frequencies = (expanded.float() @ positions).transpose(2, 3)
    embedded = torch.cat((frequencies, frequencies), dim=-1)
    cos, sin = embedded.cos(), embedded.sin()

    if not interleaved:
        raise NotImplementedError("only the interleaved MRoPE branch is ported; this checkpoint uses it")

    def select(table):
        """Take channel c from axis `c % 3`, in the strided pattern upstream uses."""
        axes = len(sections)
        selected = table[0].clone()
        for axis, width in enumerate(sections[1:], 1):
            selected[..., axis : width * axes : axes] = table[axis][..., axis : width * axes : axes]
        return selected

    half = cos.shape[-1] // 2
    cos = torch.cat([select(cos[..., :half])] * 2, dim=-1).unsqueeze(1)
    sin = torch.cat([select(sin[..., :half])] * 2, dim=-1).unsqueeze(1)
    return cos, sin


def causal_mask(length):
    """Additive mask [1, 1, T, T]: 0 where a position may attend, MASK_FILL where it may not."""
    allowed = torch.ones(length, length, dtype=torch.bool).tril()
    return torch.where(allowed, 0.0, MASK_FILL).reshape(1, 1, length, length)


# ── parameters ──────────────────────────────────────────────────────────────


def preprocess_talker_parameters(device, config=None, state=None, num_layers=None, dtype=ttnn.bfloat16):
    """Checkpoint weights -> the tensors the decoder runs with.

    `num_layers` truncates the stack, which keeps component tests cheap; the full model
    leaves it None.
    """
    cfg = dict(config or checkpoint.talker_config())
    if num_layers is not None:
        cfg["num_hidden_layers"] = num_layers
    state = checkpoint.load_talker_state() if state is None else state

    def linear(name):
        """nn.Linear weight [out, in] -> device [in, out]; the talker carries no biases."""
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
        prefix = f"layers.{index}"
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

    return {"config": cfg, "layers": layers, "norm": norm("norm")}


# ── blocks ──────────────────────────────────────────────────────────────────


class TtTalkerMLP:
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, device, params, compute_config):
        self.p = params
        self.cc = compute_config

    def __call__(self, x):
        gate = ttnn.linear(x, self.p["gate_proj"], compute_kernel_config=self.cc)
        up = ttnn.linear(x, self.p["up_proj"], compute_kernel_config=self.cc)
        return ttnn.linear(ttnn.multiply(ttnn.silu(gate), up), self.p["down_proj"], compute_kernel_config=self.cc)


class TtTalkerAttention:
    """Grouped-query attention with QK-norm and a rotation supplied by the caller."""

    def __init__(self, device, params, config, compute_config):
        self.device = device
        self.p = params
        self.cc = compute_config
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.groups = self.heads // self.kv_heads
        self.scale = self.head_dim**-0.5
        self.eps = config["rms_norm_eps"]

    def _split_heads(self, x, heads, length):
        """[1, T, heads*head_dim] -> [1, heads, T, head_dim]."""
        x = ttnn.reshape(x, (1, length, heads, self.head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _apply_rotation(self, x, cos, sin):
        """Upstream's rotation in one kernel: `x * cos + rotate_half(x) * sin`.

        The HF layout `rotary_tables` already builds. The cached decoder rotates through the same
        op, which is what keeps the two graphs comparable in `test_decode_pcc.py`.
        """
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=False, compute_kernel_config=self.cc)

    def __call__(self, x, cos, sin, mask, length):
        query = self._split_heads(ttnn.linear(x, self.p["q_proj"], compute_kernel_config=self.cc), self.heads, length)
        key = self._split_heads(ttnn.linear(x, self.p["k_proj"], compute_kernel_config=self.cc), self.kv_heads, length)
        value = self._split_heads(
            ttnn.linear(x, self.p["v_proj"], compute_kernel_config=self.cc), self.kv_heads, length
        )

        # QK-norm runs per head over head_dim, before the rotation.
        query = ttnn.rms_norm(query, weight=self.p["q_norm"], epsilon=self.eps)
        key = ttnn.rms_norm(key, weight=self.p["k_norm"], epsilon=self.eps)

        query = self._apply_rotation(query, cos, sin)
        key = self._apply_rotation(key, cos, sin)

        # Grouped-query: each key/value head serves `groups` consecutive query heads.
        key = ttnn.repeat_interleave(key, self.groups, dim=1)
        value = ttnn.repeat_interleave(value, self.groups, dim=1)

        scores = ttnn.matmul(query, ttnn.permute(key, (0, 1, 3, 2)), compute_kernel_config=self.cc)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), mask)
        weights = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.cc)

        attended = ttnn.matmul(weights, value, compute_kernel_config=self.cc)
        attended = ttnn.permute(attended, (0, 2, 1, 3))
        attended = ttnn.reshape(attended, (1, length, self.heads * self.head_dim))
        return ttnn.linear(attended, self.p["o_proj"], compute_kernel_config=self.cc)


class TtTalkerDecoderLayer:
    def __init__(self, device, params, config, compute_config):
        self.p = params
        self.eps = config["rms_norm_eps"]
        self.attention = TtTalkerAttention(device, params, config, compute_config)
        self.mlp = TtTalkerMLP(device, params, compute_config)

    def __call__(self, x, cos, sin, mask, length):
        normed = ttnn.rms_norm(x, weight=self.p["input_layernorm"], epsilon=self.eps)
        x = ttnn.add(x, self.attention(normed, cos, sin, mask, length))
        normed = ttnn.rms_norm(x, weight=self.p["post_attention_layernorm"], epsilon=self.eps)
        return ttnn.add(x, self.mlp(normed))


class TtTalker:
    """The decoder stack: embeddings in, hidden states out."""

    def __init__(self, device, parameters):
        self.device = device
        self.p = parameters
        self.config = parameters["config"]
        self.compute_config = _compute_config(device)
        self.layers = [
            TtTalkerDecoderLayer(device, layer, self.config, self.compute_config) for layer in parameters["layers"]
        ]

    def host_inputs(self, length, position_ids=None):
        """The rotation tables and mask for a prompt of this length, as torch tensors."""
        if position_ids is None:
            positions = torch.arange(length, dtype=torch.long).reshape(1, 1, length)
            position_ids = positions.expand(3, 1, length).contiguous()
        cos, sin = rotary_tables(self.config, position_ids)
        return cos, sin, causal_mask(length)

    def __call__(self, embeddings, cos, sin, mask, return_intermediates=False):
        """embeddings [1, T, 2048] -> hidden states [1, T, 2048]."""
        length = embeddings.shape[1]
        intermediates = {}

        x = embeddings
        for index, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask, length)
            if return_intermediates:
                intermediates[f"layers.{index}"] = x

        x = ttnn.rms_norm(x, weight=self.p["norm"], epsilon=self.config["rms_norm_eps"])
        if return_intermediates:
            intermediates["norm"] = x
            return x, intermediates
        return x
