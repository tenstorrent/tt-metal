# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-Music3's RVQ depth decoder (the local LM) on one Blackhole chip.

Reference: diffusers ``models/transformers/minimax_music3_rvq_depth_decoder.py``
(``MiniMaxMusic3RVQDepthDecoder``): a 4-layer causal transformer (hidden 4096, 16 heads x 256,
SwiGLU 6144, RMSNorm eps 1e-6, learned position embedding of 16 slots) that, inside one audio
frame, predicts the seven residual RVQ codebooks c1..c7 from the backbone's hidden state and the
frame's semantic code. The reference has no KV cache: every depth step re-runs the whole (<= 9
position) sequence.

Device layout
-------------
The depth sequence of both CFG rows lives in ONE tile-aligned tensor ``seq`` of shape
``[1, 1, 64, 4096]`` (bf16, tile layout, DRAM): row ``b * 32 + s`` holds step ``s`` of batch row
``b``. Padding to 32 steps (one tile) keeps every op shape fixed regardless of the logical number
of steps (2..8 in the AR loop, up to 9 accepted), so the same program cache entries and the same
trace serve every step. Rows past the logical length are zero (plus the padded position
embedding); causal attention guarantees real rows never read them.

The sequence is built on device without host round-trips:

* ``place_step(seq, rows, step)``: ``seq + E[step] @ rows`` where ``rows`` is a ``[1, 1, 32, 4096]``
  tensor whose rows 0 / 1 are the two batch rows and ``E[step]`` is a ``[64, 32]`` one-hot
  scatter matrix (ones at ``(b * 32 + step, b)``).
* ``forward(seq, num_steps=n)``: the transformer stack over the padded sequence followed by
  ``SEL[n - 1] @ normed`` with ``SEL[s]`` a ``[32, 64]`` one-hot gather matrix, giving a
  ``[1, 1, 32, 4096]`` tensor whose rows 0 / 1 are the last logical step of the two rows - the
  "decode row" convention shared with ``MusicLLM``.

Because both selectors are plain tensors, ``DepthStepTrace`` captures one depth step (embed the
previous code, project, scatter, forward, all seven heads) as a single trace and replays it for
every step with only the small selector / code-id buffers rewritten; a second small trace seeds
steps 0 / 1 from two persistent buffers, so a whole frame allocates nothing on device.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

import ttnn
from models.autoports.minimaxai_minimax_music3.tt.constants import (
    AUDIO_VOCAB_SIZE,
    LLM_BATCH,
    LLM_HIDDEN,
    NUM_CODEBOOKS,
)
from models.common.lightweightmodule import LightweightModule

TILE = 32
DEPTH_HIDDEN = LLM_HIDDEN  # 4096
DEPTH_LAYERS = 4
DEPTH_HEADS = 16
DEPTH_HEAD_DIM = DEPTH_HIDDEN // DEPTH_HEADS  # 256
DEPTH_INTERMEDIATE = 6144
DEPTH_MAX_POSITIONS = 16
DEPTH_NORM_EPS = 1e-6
NUM_RESIDUAL_CODEBOOKS = NUM_CODEBOOKS - 1  # 7 heads, 7 embedding blocks
# The AR loop runs 2..8 steps (global hidden, semantic code, then c1..c6); 9 = one more than the
# longest sequence the pipeline ever builds, accepted so a caller can append c7 too.
MAX_STEPS = 9
SEQ_PAD = TILE
ROWS = LLM_BATCH * SEQ_PAD  # 64

TensorLike = Union[torch.Tensor, ttnn.Tensor]


def _one_hot_scatter(step: int) -> torch.Tensor:
    """``E[step]`` [64, 32]: ``E @ rows`` puts ``rows[b]`` at sequence row ``b * 32 + step``."""
    e = torch.zeros(ROWS, TILE)
    for b in range(LLM_BATCH):
        e[b * SEQ_PAD + step, b] = 1.0
    return e


def _one_hot_gather(step: int) -> torch.Tensor:
    """``SEL[step]`` [32, 64]: ``SEL @ seq`` gathers sequence row ``b * 32 + step`` into row ``b``."""
    s = torch.zeros(TILE, ROWS)
    for b in range(LLM_BATCH):
        s[b, b * SEQ_PAD + step] = 1.0
    return s


class DepthDecoder(LightweightModule):
    """``MiniMaxMusic3RVQDepthDecoder`` on device. See the module docstring for the layout."""

    def __init__(self, mesh_device, state_dict: Dict[str, torch.Tensor], *, dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.mem = ttnn.DRAM_MEMORY_CONFIG
        self.l1 = ttnn.L1_MEMORY_CONFIG
        # RMSNorm: the interleaved kernel parallelizes over tile rows and the padded sequence has
        # only two, so it ran on 2 cores (75 us). Width-sharding the [64, 4096] activation over a
        # 4x4 grid (shards [64, 256]) runs the norm on 16 cores in ~7 us plus two ~3 us resharding
        # ops (measured with Tracy; 8x4 / 8x8 grids were slower because of per-op dispatch gaps).
        norm_cores = 16
        self.norm_mem = ttnn.create_sharded_memory_config(
            shape=(ROWS, DEPTH_HIDDEN // norm_cores),
            core_grid=ttnn.CoreGrid(y=4, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[4, 4],
            subblock_w=4,
            block_h=ROWS // TILE,
            block_w=DEPTH_HIDDEN // norm_cores // TILE,
            inplace=False,
        )
        self.load_seconds = self._load_weights(state_dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(cls, mesh_device, weights_dir, *, dtype=ttnn.bfloat16) -> "DepthDecoder":
        """Load ``<weights_dir>/rvq_depth_decoder/diffusion_pytorch_model.safetensors``."""
        from safetensors.torch import load_file

        path = Path(weights_dir) / "rvq_depth_decoder" / "diffusion_pytorch_model.safetensors"
        return cls(mesh_device, load_file(str(path)), dtype=dtype)

    def _weight(self, w: torch.Tensor, *, transpose: bool = True, dtype=None) -> ttnn.Tensor:
        """A torch ``nn.Linear`` weight [out, in] -> device [in, out] tile tensor (or any 2D matrix)."""
        if transpose:
            w = w.transpose(0, 1)
        return ttnn.from_torch(
            w.contiguous(),
            dtype=dtype or self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.mem,
        )

    def _load_weights(self, sd: Dict[str, torch.Tensor]) -> float:
        t0 = time.time()
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
        expected = {
            "audio_embeddings.weight": (NUM_RESIDUAL_CODEBOOKS * AUDIO_VOCAB_SIZE, DEPTH_HIDDEN),
            "projection.weight": (DEPTH_HIDDEN, DEPTH_HIDDEN),
            "pos_embedding.weight": (DEPTH_MAX_POSITIONS, DEPTH_HIDDEN),
            "norm.weight": (DEPTH_HIDDEN,),
        }
        for k, shape in expected.items():
            assert tuple(sd[k].shape) == shape, (k, tuple(sd[k].shape), shape)

        # Embedding table stays row-major (ttnn.embedding contract).
        self.audio_embeddings = ttnn.from_torch(
            sd["audio_embeddings.weight"],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=self.mem,
        )
        self.projection = self._weight(sd["projection.weight"])
        # Position embedding pre-broadcast onto the padded [64, 4096] sequence layout: row b*32+s
        # is pos[s] for s < 16 and zero for the never-read padding rows.
        pos = torch.zeros(ROWS, DEPTH_HIDDEN, dtype=torch.bfloat16)
        for b in range(LLM_BATCH):
            pos[b * SEQ_PAD : b * SEQ_PAD + DEPTH_MAX_POSITIONS] = sd["pos_embedding.weight"]
        self.pos_embedding = ttnn.from_torch(
            pos.reshape(1, 1, ROWS, DEPTH_HIDDEN),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.mem,
        )

        def norm_w(name):
            return ttnn.from_torch(
                sd[name].reshape(1, 1, 1, DEPTH_HIDDEN),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=self.mem,
            )

        self.layers: List[dict] = []
        for i in range(DEPTH_LAYERS):
            p = f"layers.{i}."
            wqkv = torch.cat(
                [sd[p + "attn.to_q.weight"], sd[p + "attn.to_k.weight"], sd[p + "attn.to_v.weight"]], dim=0
            )  # [3*4096, 4096] -> transposed to [4096, 12288] = [Q | K | V] blocks (nlp_create_qkv_heads order)
            self.layers.append(
                {
                    "ln_in": norm_w(p + "input_layernorm.weight"),
                    "wqkv": self._weight(wqkv),
                    "wo": self._weight(sd[p + "attn.to_out.weight"]),
                    "ln_post": norm_w(p + "post_attention_layernorm.weight"),
                    "w_gate": self._weight(sd[p + "gate_proj.weight"]),
                    "w_up": self._weight(sd[p + "up_proj.weight"]),
                    "w_down": self._weight(sd[p + "down_proj.weight"]),
                }
            )
        self.norm_weight = norm_w("norm.weight")
        self.audio_heads = [self._weight(sd[f"audio_heads.{k}.weight"]) for k in range(NUM_RESIDUAL_CODEBOOKS)]
        # All seven heads fused: [4096, 7 * 1024]; column block k = audio_heads[k].
        self.audio_heads_fused = self._weight(
            torch.cat([sd[f"audio_heads.{k}.weight"] for k in range(NUM_RESIDUAL_CODEBOOKS)], dim=0)
        )

        # Scatter / gather selectors for every supported step index.
        self.scatter = [
            self._weight(_one_hot_scatter(s).reshape(1, 1, ROWS, TILE), transpose=False) for s in range(MAX_STEPS)
        ]
        self.gather = [
            self._weight(_one_hot_gather(s).reshape(1, 1, TILE, ROWS), transpose=False) for s in range(MAX_STEPS)
        ]
        self.scatter_none = self._weight(torch.zeros(1, 1, ROWS, TILE), transpose=False)
        ttnn.synchronize_device(self.mesh_device)
        return time.time() - t0

    # ------------------------------------------------------------------ small helpers
    def _linear(self, x: ttnn.Tensor, w: ttnn.Tensor, activation: Optional[str] = None) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            w,
            compute_kernel_config=self.compute_config,
            memory_config=self.mem,
            dtype=self.dtype,
            activation=activation,
        )

    def _rms_norm(self, x: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        xs = ttnn.interleaved_to_sharded(x, self.norm_mem)
        y = ttnn.rms_norm(
            xs,
            epsilon=DEPTH_NORM_EPS,
            weight=w,
            program_config=self.norm_program_config,
            compute_kernel_config=self.compute_config,
        )
        ttnn.deallocate(xs)
        out = ttnn.sharded_to_interleaved(y, self.mem)
        ttnn.deallocate(y)
        return out

    def rows_to_device(self, x: torch.Tensor) -> ttnn.Tensor:
        """Host [B, 4096] (B <= 32) -> the padded ``[1, 1, 32, 4096]`` decode-row tensor."""
        assert x.dim() == 2 and x.shape[0] <= TILE and x.shape[1] == DEPTH_HIDDEN, tuple(x.shape)
        pad = torch.zeros(TILE, DEPTH_HIDDEN, dtype=torch.bfloat16)
        pad[: x.shape[0]] = x.to(torch.bfloat16)
        return ttnn.from_torch(
            pad.reshape(1, 1, TILE, DEPTH_HIDDEN),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.mem,
        )

    @staticmethod
    def rows_to_host(x: ttnn.Tensor, batch: int = LLM_BATCH) -> torch.Tensor:
        """``[1, 1, 32, N]`` device tensor -> fp32 host ``[batch, N]`` (the logical rows)."""
        return ttnn.to_torch(x).reshape(TILE, -1)[:batch].float()

    def code_ids_to_device(self, codes: torch.Tensor, codebook: int) -> ttnn.Tensor:
        """Residual codes [B] of codebook ``codebook`` (1..7) -> ``[1, 32]`` uint32 table indices."""
        assert 1 <= codebook <= NUM_RESIDUAL_CODEBOOKS, codebook
        codes = codes.reshape(-1).to(torch.int64)
        assert codes.numel() <= TILE and int(codes.min()) >= 0 and int(codes.max()) < AUDIO_VOCAB_SIZE
        ids = torch.zeros(1, TILE, dtype=torch.int32)
        ids[0, : codes.numel()] = codes + (codebook - 1) * AUDIO_VOCAB_SIZE
        return ttnn.from_torch(
            ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device, memory_config=self.mem
        )

    # ------------------------------------------------------------------ public API
    def audio_embed(self, residual_codes: TensorLike, codebook: Optional[int] = None) -> ttnn.Tensor:
        """``audio_embeddings[code + (codebook - 1) * 1024]`` as a ``[1, 1, 32, 4096]`` decode-row tensor.

        ``residual_codes``: host ``[B]`` codes of codebook ``codebook`` (1..7), or a ``[1, 32]`` uint32
        device tensor of *already offset* table indices (``codebook`` ignored).
        """
        if isinstance(residual_codes, torch.Tensor):
            assert codebook is not None, "codebook index (1..7) is required for host codes"
            ids = self.code_ids_to_device(residual_codes, codebook)
        else:
            ids = residual_codes
        emb = ttnn.embedding(
            ids, self.audio_embeddings, layout=ttnn.TILE_LAYOUT, dtype=self.dtype, memory_config=self.mem
        )
        return ttnn.experimental.view(emb, (1, 1, TILE, DEPTH_HIDDEN))

    def residual_embedding_sum(self, frame_codes: torch.Tensor) -> ttnn.Tensor:
        """``sum_k audio_embeddings[c_k + (k-1)*1024]`` for ``frame_codes`` [B, 8] (column 0 ignored) -> ``[1, 1, 32, 4096]``.

        This is the residual half of the backbone's frame feedback (``MusicLLM.embed_frame``).
        """
        assert frame_codes.shape[-1] == NUM_CODEBOOKS, tuple(frame_codes.shape)
        total = None
        for k in range(1, NUM_CODEBOOKS):
            e = self.audio_embed(frame_codes[:, k], k)
            total = e if total is None else ttnn.add(total, e)
        return total

    def project(self, x: TensorLike) -> ttnn.Tensor:
        """``projection`` (4096 -> 4096, no bias) of a ``[B, 4096]`` host or ``[1, 1, 32, 4096]`` device row tensor."""
        if isinstance(x, torch.Tensor):
            x = self.rows_to_device(x)
        return self._linear(x, self.projection)

    def new_sequence(self) -> ttnn.Tensor:
        """An all-zero padded depth sequence ``[1, 1, 64, 4096]``."""
        return ttnn.from_torch(
            torch.zeros(1, 1, ROWS, DEPTH_HIDDEN, dtype=torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.mem,
        )

    def place_step(self, seq: ttnn.Tensor, rows: ttnn.Tensor, step: Union[int, ttnn.Tensor]) -> ttnn.Tensor:
        """Write ``rows`` (``[1, 1, 32, 4096]``, rows 0/1 = batch rows) into step ``step`` of ``seq`` (zero there before)."""
        e = self.scatter[step] if isinstance(step, int) else step
        placed = ttnn.matmul(
            e, rows, compute_kernel_config=self.compute_config, memory_config=self.mem, dtype=self.dtype
        )
        return ttnn.add(seq, placed, memory_config=self.mem)

    def sequence_from_host(self, inputs_embeds: torch.Tensor) -> ttnn.Tensor:
        """Host ``[B, steps, 4096]`` (already projected, as the reference's ``forward`` expects) -> padded device sequence."""
        batch, steps, dim = inputs_embeds.shape
        assert batch == LLM_BATCH and dim == DEPTH_HIDDEN and 1 <= steps <= MAX_STEPS, tuple(inputs_embeds.shape)
        seq = torch.zeros(LLM_BATCH, SEQ_PAD, DEPTH_HIDDEN, dtype=torch.bfloat16)
        seq[:, :steps] = inputs_embeds.to(torch.bfloat16)
        return ttnn.from_torch(
            seq.reshape(1, 1, ROWS, DEPTH_HIDDEN),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.mem,
        )

    def _attention(self, x: ttnn.Tensor, layer: dict) -> ttnn.Tensor:
        # The head split / merge ops also run on 2 cores (one per batch row's single tile row);
        # keeping their operands in L1 cuts them from 155 / 52 us to 113 / 38 us (Tracy).
        qkv = ttnn.linear(
            x, layer["wqkv"], compute_kernel_config=self.compute_config, memory_config=self.l1, dtype=self.dtype
        )  # [1, 1, 64, 12288]
        qkv = ttnn.experimental.view(qkv, (LLM_BATCH, 1, SEQ_PAD, 3 * DEPTH_HIDDEN))  # zero-copy tile-aligned row split
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=DEPTH_HEADS, num_kv_heads=DEPTH_HEADS, transpose_k_heads=False, memory_config=self.l1
        )
        ttnn.deallocate(qkv)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, compute_kernel_config=self.sdpa_compute_config, memory_config=self.l1
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=self.l1)  # [2, 1, 32, 4096]
        attn = ttnn.experimental.view(attn, (1, 1, ROWS, DEPTH_HIDDEN))
        return self._linear(attn, layer["wo"])

    def _mlp(self, x: ttnn.Tensor, layer: dict) -> ttnn.Tensor:
        gate = self._linear(x, layer["w_gate"], activation="silu")
        up = self._linear(x, layer["w_up"])
        h = ttnn.multiply(gate, up, memory_config=self.mem)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return self._linear(h, layer["w_down"])

    def hidden_states(self, seq: ttnn.Tensor) -> ttnn.Tensor:
        """The normed hidden of every (padded) position: ``[1, 1, 64, 4096]``."""
        x = ttnn.add(seq, self.pos_embedding, memory_config=self.mem)
        for layer in self.layers:
            x = ttnn.add(x, self._attention(self._rms_norm(x, layer["ln_in"]), layer), memory_config=self.mem)
            x = ttnn.add(x, self._mlp(self._rms_norm(x, layer["ln_post"]), layer), memory_config=self.mem)
        return self._rms_norm(x, self.norm_weight)

    def forward(self, inputs_embeds: TensorLike, *, num_steps: Union[int, ttnn.Tensor]) -> ttnn.Tensor:
        """Normed hidden of the LAST logical step as a ``[1, 1, 32, 4096]`` tensor (rows 0/1 = the batch rows).

        ``inputs_embeds``: host ``[2, num_steps, 4096]`` or the padded device sequence ``[1, 1, 64, 4096]``.
        ``num_steps``: the logical length (1..9) or a ``[1, 1, 32, 64]`` gather selector tensor (traced path).
        """
        if isinstance(inputs_embeds, torch.Tensor):
            assert isinstance(num_steps, int) and inputs_embeds.shape[1] == num_steps
            seq = self.sequence_from_host(inputs_embeds)
        else:
            seq = inputs_embeds
        if isinstance(num_steps, int):
            assert 1 <= num_steps <= MAX_STEPS, num_steps
            sel = self.gather[num_steps - 1]
        else:
            sel = num_steps
        normed = self.hidden_states(seq)
        return ttnn.matmul(
            sel, normed, compute_kernel_config=self.compute_config, memory_config=self.mem, dtype=self.dtype
        )

    def head(self, k: int, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """``audio_heads[k - 1]`` logits ``[1, 1, 32, 1024]`` (rows 0/1 = batch rows) for codebook ``k`` in 1..7."""
        assert 1 <= k <= NUM_RESIDUAL_CODEBOOKS, k
        return self._linear(hidden, self.audio_heads[k - 1])

    def heads_all(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """All seven heads at once: ``[1, 1, 32, 7 * 1024]``; column block ``k - 1`` = ``head(k)``."""
        return self._linear(hidden, self.audio_heads_fused)

    # ------------------------------------------------------------------ eager depth loop
    def teacher_forced_loop(
        self, global_hidden: torch.Tensor, semantic_embed: torch.Tensor, residual_codes: torch.Tensor
    ) -> Tuple[List[ttnn.Tensor], List[ttnn.Tensor]]:
        """The seven-step depth loop with the codes given (no sampling), everything on device.

        ``global_hidden`` / ``semantic_embed``: host ``[2, 4096]``; ``residual_codes``: host ``[2, 7]``.
        Returns the seven last-step hiddens and the seven head logits (device ``[1, 1, 32, *]`` tensors).
        """
        seq = self.new_sequence()
        seq = self.place_step(seq, self.project(global_hidden), 0)
        seq = self.place_step(seq, self.project(semantic_embed), 1)
        hiddens, logits = [], []
        for index in range(1, NUM_CODEBOOKS):
            hidden = self.forward(seq, num_steps=index + 1)
            hiddens.append(hidden)
            logits.append(self.head(index, hidden))
            if index < NUM_RESIDUAL_CODEBOOKS:
                emb = self.audio_embed(residual_codes[:, index - 1], index)
                seq = self.place_step(seq, self.project(emb), index + 1)
        return hiddens, logits


class DepthStepTrace:
    """One depth step as a single trace with fixed shapes, plus a seeding trace for step 0/1.

    Per replay the host rewrites three small buffers: the scatter selector ``E`` (``[64, 32]``; all
    zero for the first step, which appends nothing), the code-id row (``[1, 32]`` uint32, offset
    table indices) and the gather selector ``SEL`` (``[32, 64]``). The trace embeds + projects the
    code, scatters it into the persistent sequence buffer, runs the stack and all seven heads, and
    leaves the logits in a persistent output buffer. ``begin_frame`` writes the backbone hidden and
    the semantic-code embedding into two persistent seed buffers and replays a second, small trace
    that projects and scatters them into steps 0 / 1 of the sequence.

    Trace-lifetime contract: every device buffer this class touches is allocated in ``__init__``
    (persistent inputs / outputs) or inside a capture (trace-owned intermediates). Nothing is
    allocated after the captures, so replays never race a live post-capture buffer (tt-metal:
    "Allocating device buffers is unsafe due to the existence of an active trace" - a buffer
    allocated after capture must be dead before ``execute_trace``). Callers must keep the same
    rule: a device tensor allocated after ``DepthStepTrace`` was built must not be alive across
    ``step`` / ``begin_frame`` unless it is one of the persistent buffers; to seed from a device
    tensor (stage 04: the backbone hidden), pass it to ``begin_frame`` and it is copied into the
    persistent seed buffer with ``ttnn.copy`` (no allocation).
    """

    def __init__(self, decoder: DepthDecoder, cq_id: int = 0):
        self.d = decoder
        self.dev = decoder.mesh_device
        self.cq_id = cq_id
        d = decoder
        # Persistent buffers (allocated before any capture).
        self.seq = d.new_sequence()
        self.seed_hidden = d.rows_to_device(torch.zeros(LLM_BATCH, DEPTH_HIDDEN))
        self.seed_semantic = d.rows_to_device(torch.zeros(LLM_BATCH, DEPTH_HIDDEN))
        self.scatter_buf = ttnn.clone(d.scatter_none, memory_config=d.mem)
        self.gather_buf = ttnn.clone(d.gather[1], memory_config=d.mem)
        self.ids_buf = d.code_ids_to_device(torch.zeros(LLM_BATCH, dtype=torch.int64), 1)
        self._host_scatter = [
            _one_hot_scatter(s).reshape(1, 1, ROWS, TILE).to(torch.bfloat16) for s in range(MAX_STEPS)
        ]
        self._host_scatter_none = torch.zeros(1, 1, ROWS, TILE, dtype=torch.bfloat16)
        self._host_gather = [_one_hot_gather(s).reshape(1, 1, TILE, ROWS).to(torch.bfloat16) for s in range(MAX_STEPS)]
        self.seed_trace_id = None
        self.trace_id = None
        self.hidden_out = None
        self.logits_out = None
        self._capture()

    def _seed_graph(self):
        d = self.d
        first = ttnn.matmul(
            d.scatter[0],
            d.project(self.seed_hidden),
            compute_kernel_config=d.compute_config,
            memory_config=d.mem,
            dtype=d.dtype,
        )
        second = ttnn.matmul(
            d.scatter[1],
            d.project(self.seed_semantic),
            compute_kernel_config=d.compute_config,
            memory_config=d.mem,
            dtype=d.dtype,
        )
        seq = ttnn.add(first, second, memory_config=d.mem)
        ttnn.copy(seq, self.seq)

    def _step_graph(self):
        d = self.d
        emb = d.audio_embed(self.ids_buf)
        rows = d.project(emb)
        placed = ttnn.matmul(
            self.scatter_buf, rows, compute_kernel_config=d.compute_config, memory_config=d.mem, dtype=d.dtype
        )
        new_seq = ttnn.add(self.seq, placed, memory_config=d.mem)
        ttnn.copy(new_seq, self.seq)  # the sequence persists across replays in the same buffer
        hidden = d.forward(self.seq, num_steps=self.gather_buf)
        logits = d.heads_all(hidden)
        return hidden, logits

    def _capture(self):
        # 1. Compile runs (program cache) of BOTH exact op sequences - including the output copies -
        #    before any capture: an op first seen inside a capture would compile and load its
        #    kernels, which is a host write and fails the capture ("Writes are not supported during
        #    trace capture"); and a compile run between two captures would allocate while a trace
        #    is live (the allocator warning above).
        self._seed_graph()
        hidden, logits = self._step_graph()
        self.hidden_out = ttnn.clone(hidden, memory_config=self.d.mem)
        self.logits_out = ttnn.clone(logits, memory_config=self.d.mem)
        ttnn.copy(hidden, self.hidden_out)
        ttnn.copy(logits, self.logits_out)
        ttnn.synchronize_device(self.dev)
        ttnn.deallocate(hidden)
        ttnn.deallocate(logits)
        # 2. Captures. Allocations inside a capture are trace-owned and fine.
        self.seed_trace_id = ttnn.begin_trace_capture(self.dev, cq_id=self.cq_id)
        self._seed_graph()
        ttnn.end_trace_capture(self.dev, self.seed_trace_id, cq_id=self.cq_id)
        self.trace_id = ttnn.begin_trace_capture(self.dev, cq_id=self.cq_id)
        hidden, logits = self._step_graph()
        ttnn.copy(hidden, self.hidden_out)
        ttnn.copy(logits, self.logits_out)
        ttnn.end_trace_capture(self.dev, self.trace_id, cq_id=self.cq_id)
        ttnn.synchronize_device(self.dev)
        # The compile/capture runs used scatter_none, so self.seq is still the zero sequence; the
        # seed replay in begin_frame overwrites it anyway.

    def _write(self, buf: ttnn.Tensor, host: torch.Tensor, dtype, layout):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=layout), buf, cq_id=self.cq_id)

    def _seed(self, buf: ttnn.Tensor, value: TensorLike) -> None:
        if isinstance(value, torch.Tensor):
            pad = torch.zeros(TILE, DEPTH_HIDDEN, dtype=torch.bfloat16)
            pad[: value.shape[0]] = value.to(torch.bfloat16)
            self._write(buf, pad.reshape(1, 1, TILE, DEPTH_HIDDEN), self.d.dtype, ttnn.TILE_LAYOUT)
        else:
            ttnn.copy(value, buf)  # a persistent [1, 1, 32, 4096] device row tensor (e.g. the backbone hidden)

    def begin_frame(self, global_hidden: TensorLike, semantic_embed: TensorLike) -> None:
        """Seed steps 0 / 1 (projected backbone hidden and semantic-code embedding) for a new frame.

        Both arguments are host ``[2, 4096]`` tensors or ``[1, 1, 32, 4096]`` device row tensors.
        Allocation-free: two buffer writes and one trace replay.
        """
        self._seed(self.seed_hidden, global_hidden)
        self._seed(self.seed_semantic, semantic_embed)
        ttnn.execute_trace(self.dev, self.seed_trace_id, cq_id=self.cq_id, blocking=False)

    def step(self, index: int, prev_code: Optional[torch.Tensor]) -> ttnn.Tensor:
        """Depth step ``index`` (1..7): append ``prev_code`` (codebook ``index - 1``, None for index 1), return all-head logits.

        The returned ``[1, 1, 32, 7168]`` device tensor is the persistent output buffer; read it before the next step.
        """
        assert 1 <= index <= NUM_RESIDUAL_CODEBOOKS, index
        d = self.d
        if index == 1:
            assert prev_code is None
            self._write(self.scatter_buf, self._host_scatter_none, d.dtype, ttnn.TILE_LAYOUT)
        else:
            ids = torch.zeros(1, TILE, dtype=torch.int32)
            ids[0, :LLM_BATCH] = prev_code.reshape(-1).to(torch.int64)[:LLM_BATCH] + (index - 2) * AUDIO_VOCAB_SIZE
            self._write(self.ids_buf, ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            self._write(self.scatter_buf, self._host_scatter[index], d.dtype, ttnn.TILE_LAYOUT)
        self._write(self.gather_buf, self._host_gather[index], d.dtype, ttnn.TILE_LAYOUT)  # last logical step = index
        ttnn.execute_trace(self.dev, self.trace_id, cq_id=self.cq_id, blocking=False)
        return self.logits_out

    def logits_for(self, k: int, logits_all: Optional[ttnn.Tensor] = None) -> torch.Tensor:
        """Host fp32 ``[2, 1024]`` logits of head ``k`` (1..7) from the all-head output."""
        t = DepthDecoder.rows_to_host(logits_all if logits_all is not None else self.logits_out)
        return t[:, (k - 1) * AUDIO_VOCAB_SIZE : k * AUDIO_VOCAB_SIZE]

    def release(self):
        for attr in ("trace_id", "seed_trace_id"):
            tid = getattr(self, attr)
            if tid is not None:
                ttnn.release_trace(self.dev, tid)
                setattr(self, attr, None)
