# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""KV-cached autoregressive decode for the Qwen3-TTS code predictor.

The predictor is small but runs often: 15 steps per frame against the talker's one, so
187.5 steps per second of audio against 12.5. Measured before this existed, it cost 0.80 s
per second of audio against the talker's 0.16 s, which made it the dominant block.

Same shape of fix as the talker's. Prefill the two-position prompt (the talker's hidden
state and codebook 0) into a per-layer cache, then step one position at a time reading it.

One structural difference: **each step reads a different output head.** Step k predicts
codebook k+1 through `lm_head[k]`, so the traced graph covers the five transformer layers
and the head is applied outside it. The head is one [1024, 2048] matmul against five layers
of work, so leaving it out of the trace costs little and keeps a single graph serving all
15 steps.

The cache is tiny, 16 positions rounded up to the 64 that flash decode requires, but the
step count is what matters here: this is a dispatch-bound block, which is exactly what a
trace fixes.
"""

import torch

import ttnn
from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor import (
    CODE_PREDICTOR_PREFIX,
    TALKER_CODEC_EMBEDDING,
    preprocess_projection,
    project,
)
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import _compute_config, causal_mask
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker_decode import (
    CACHE_TILE_MULTIPLE,
    MLP_WEIGHT_DTYPE,
    NORM_SHARD_HEIGHT,
    ROPE_ROWS,
    decode_matmul_config,
    sharded_norm_plan,
)


def preprocess_cached_predictor_parameters(device, config=None, dtype=ttnn.bfloat16, mlp_dtype=MLP_WEIGHT_DTYPE):
    """Predictor weights with Q, K and V fused, mirroring the talker's repacking."""
    talker_cfg = dict(config or checkpoint.talker_config())
    cfg = dict(talker_cfg["code_predictor_config"])
    state = checkpoint.load_prefixed(CODE_PREDICTOR_PREFIX)

    def to_device(tensor, as_dtype=None):
        return ttnn.from_torch(
            tensor.contiguous(),
            dtype=as_dtype or dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def lookup_table(tensor):
        """A codebook for `ttnn.embedding`: row major and bf16, which the op requires."""
        return ttnn.from_torch(
            tensor.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def linear(name, as_dtype=None):
        return to_device(state[f"{name}.weight"].t(), as_dtype=as_dtype)

    def norm(name):
        return to_device(state[f"{name}.weight"].reshape(1, 1, 1, -1))

    layers = []
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{index}.self_attn"
        fused = torch.cat(
            [state[f"{prefix}.q_proj.weight"], state[f"{prefix}.k_proj.weight"], state[f"{prefix}.v_proj.weight"]],
            dim=0,
        )
        layers.append(
            {
                "input_layernorm": norm(f"model.layers.{index}.input_layernorm"),
                "post_attention_layernorm": norm(f"model.layers.{index}.post_attention_layernorm"),
                "qkv": to_device(fused.t()),
                "o_proj": linear(f"{prefix}.o_proj"),
                "q_norm": norm(f"{prefix}.q_norm"),
                "k_norm": norm(f"{prefix}.k_norm"),
                "gate_proj": linear(f"model.layers.{index}.mlp.gate_proj", mlp_dtype),
                "up_proj": linear(f"model.layers.{index}.mlp.up_proj", mlp_dtype),
                "down_proj": linear(f"model.layers.{index}.mlp.down_proj", mlp_dtype),
            }
        )

    heads = cfg["num_code_groups"] - 1
    talker_table = checkpoint.load_prefixed(TALKER_CODEC_EMBEDDING)["weight"]
    predictor_tables = [
        checkpoint.load_prefixed(CODE_PREDICTOR_PREFIX + "model.codec_embedding.")[f"{index}.weight"]
        for index in range(heads)
    ]
    projection, projection_bias = preprocess_projection(state, device, dtype)
    return {
        "config": cfg,
        # What each position arrives as: the talker's width, before any projection.
        "input_width": talker_cfg["hidden_size"],
        "layers": layers,
        "norm": norm("model.norm"),
        "projection": projection,
        "projection_bias": projection_bias,
        "lm_head": [linear(f"lm_head.{index}") for index in range(heads)],
        "talker_codec_embedding": talker_table,
        "codec_embedding": predictor_tables,
        # On device too, so a lookup costs an index write rather than 76 us of `from_torch`.
        "talker_codec_embedding_device": lookup_table(talker_table),
        "codec_embedding_device": [lookup_table(table) for table in predictor_tables],
    }


class TtCodePredictorCachedDecoder:
    """Prefill two positions, then 14 cached steps, one output head each."""

    def __init__(self, device, parameters):
        self.device = device
        self.p = parameters
        self.config = parameters["config"]
        self.compute_config = _compute_config(device)

        self.heads = self.config["num_attention_heads"]
        self.kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.config["head_dim"]
        self.hidden = self.config["hidden_size"]
        self.eps = self.config["rms_norm_eps"]
        self.groups = self.config["num_code_groups"]
        self.scale = self.head_dim**-0.5
        self.layers = len(parameters["layers"])

        self.max_seq = ((self.groups + CACHE_TILE_MULTIPLE - 1) // CACHE_TILE_MULTIPLE) * CACHE_TILE_MULTIPLE
        empty = torch.zeros(1, self.kv_heads, self.max_seq, self.head_dim)
        self.k_cache = [
            ttnn.from_torch(empty, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            for _ in range(self.layers)
        ]
        self.v_cache = [
            ttnn.from_torch(empty, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            for _ in range(self.layers)
        ]

        self._pos = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), device=device)
        self._in = ttnn.from_torch(
            torch.zeros(1, 1, parameters["input_width"]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self._cos = ttnn.from_torch(
            torch.zeros(1, 1, ROPE_ROWS, self.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self._sin = ttnn.from_torch(
            torch.zeros(1, 1, ROPE_ROWS, self.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self._k_shard = self._single_core_shard(0)
        self._v_shard = self._single_core_shard(1)
        self._zero_cache = ttnn.from_torch(empty, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        silu = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        first = parameters["layers"][0]
        self._matmul = {
            name: decode_matmul_config(device, *tuple(first[name].shape)[-2:], silu if name == "gate_proj" else None)
            for name in ("qkv", "o_proj", "gate_proj", "up_proj", "down_proj")
        }
        self._norm_plan = sharded_norm_plan(device, self.hidden)
        if self._norm_plan is not None:
            state = checkpoint.load_prefixed(CODE_PREDICTOR_PREFIX)
            expand = lambda name: ttnn.from_torch(
                state[name].reshape(1, 1, -1).expand(1, NORM_SHARD_HEIGHT, self.hidden).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            for index, layer in enumerate(parameters["layers"]):
                layer["input_layernorm_wide"] = expand(f"model.layers.{index}.input_layernorm.weight")
                layer["post_attention_layernorm_wide"] = expand(f"model.layers.{index}.post_attention_layernorm.weight")
            self._final_norm_wide = expand("model.norm.weight")

        # An index for the step's codebook lookups, and per-position rotary tables built once:
        # the predictor's positions are always 0 to 15, so nothing here changes per frame.
        self._index = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._position_inputs = {}

        self.trace_id = None
        self._out = None

    def _single_core_shard(self, column):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(column, 0), ttnn.CoreCoord(column, 0))}),
                (32, self.head_dim),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _norm(self, x, weights, key):
        if self._norm_plan is None or f"{key}_wide" not in weights or weights[f"{key}_wide"] is None:
            return ttnn.rms_norm(x, weight=weights[key], epsilon=self.eps)
        memory_config, program_config = self._norm_plan
        sharded = ttnn.interleaved_to_sharded(x, memory_config)
        out = ttnn.rms_norm(
            sharded,
            weight=weights[f"{key}_wide"],
            epsilon=self.eps,
            program_config=program_config,
            memory_config=memory_config,
            compute_kernel_config=self.compute_config,
        )
        return ttnn.sharded_to_interleaved(out)

    def _rotate(self, x, cos, sin):
        """The same fused rotation the talker uses, over 150 rotations a frame."""
        return ttnn.experimental.rotary_embedding_hf(
            x, cos, sin, is_decode_mode=False, compute_kernel_config=self.compute_config
        )

    def _mlp(self, x, layer):
        gate = ttnn.linear(
            x, layer["gate_proj"], compute_kernel_config=self.compute_config, program_config=self._matmul["gate_proj"]
        )
        if self._matmul["gate_proj"] is None or self._matmul["gate_proj"].fused_activation is None:
            gate = ttnn.silu(gate)
        up = ttnn.linear(
            x, layer["up_proj"], compute_kernel_config=self.compute_config, program_config=self._matmul["up_proj"]
        )
        return ttnn.linear(
            ttnn.multiply(gate, up),
            layer["down_proj"],
            compute_kernel_config=self.compute_config,
            program_config=self._matmul["down_proj"],
        )

    def _rotary(self, length, offset=0):
        inverse = 1.0 / (
            self.config["rope_theta"]
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).to(dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(offset, offset + length, dtype=torch.float32).reshape(1, 1, length)
        frequencies = (inverse.reshape(1, -1, 1) @ positions).transpose(1, 2)
        embedded = torch.cat((frequencies, frequencies), dim=-1)
        return embedded.cos().unsqueeze(1), embedded.sin().unsqueeze(1)

    def _host_position_inputs(self, position):
        """The three host tensors a position needs, built once and kept."""
        if position not in self._position_inputs:
            cos, sin = self._rotary(1, offset=int(position))
            rows = lambda table: ttnn.from_torch(
                table.reshape(1, 1, 1, -1).expand(1, 1, ROPE_ROWS, self.head_dim).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            self._position_inputs[position] = (
                ttnn.from_torch(torch.full((1,), int(position), dtype=torch.int32)),
                rows(cos),
                rows(sin),
            )
        return self._position_inputs[position]

    def set_position(self, position):
        pos, cos, sin = self._host_position_inputs(position)
        ttnn.copy_host_to_device_tensor(pos, self._pos)
        ttnn.copy_host_to_device_tensor(cos, self._cos)
        ttnn.copy_host_to_device_tensor(sin, self._sin)

    def prefill(self, embeddings):
        """Seed the cache from the two-position prompt and return its last hidden state."""
        length = embeddings.shape[1]
        cos, sin = self._rotary(length)
        to_device = lambda tensor: ttnn.from_torch(
            tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        x = to_device(embeddings)
        cos_tt, sin_tt, mask_tt = to_device(cos), to_device(sin), to_device(causal_mask(length))

        x = project(x, self.p, self.compute_config)
        for index, layer in enumerate(self.p["layers"]):
            normed = ttnn.rms_norm(x, weight=layer["input_layernorm"], epsilon=self.eps)
            qkv = ttnn.linear(normed, layer["qkv"], compute_kernel_config=self.compute_config)

            def split(start, heads, width):
                piece = ttnn.slice(qkv, [0, 0, start], [1, length, start + width])
                return ttnn.permute(ttnn.reshape(piece, (1, length, heads, self.head_dim)), (0, 2, 1, 3))

            q_width, kv_width = self.heads * self.head_dim, self.kv_heads * self.head_dim
            query = split(0, self.heads, q_width)
            key = split(q_width, self.kv_heads, kv_width)
            value = split(q_width + kv_width, self.kv_heads, kv_width)

            query = self._rotate(ttnn.rms_norm(query, weight=layer["q_norm"], epsilon=self.eps), cos_tt, sin_tt)
            key = self._rotate(ttnn.rms_norm(key, weight=layer["k_norm"], epsilon=self.eps), cos_tt, sin_tt)

            ttnn.fill_cache(self.k_cache[index], key, 0)
            ttnn.fill_cache(self.v_cache[index], value, 0)

            repeats = self.heads // self.kv_heads
            scores = ttnn.matmul(
                query,
                ttnn.permute(ttnn.repeat_interleave(key, repeats, dim=1), (0, 1, 3, 2)),
                compute_kernel_config=self.compute_config,
            )
            scores = ttnn.softmax(
                ttnn.add(ttnn.multiply(scores, self.scale), mask_tt), dim=-1, compute_kernel_config=self.compute_config
            )
            attended = ttnn.matmul(
                scores, ttnn.repeat_interleave(value, repeats, dim=1), compute_kernel_config=self.compute_config
            )
            attended = ttnn.reshape(ttnn.permute(attended, (0, 2, 1, 3)), (1, length, self.heads * self.head_dim))
            x = ttnn.add(x, ttnn.linear(attended, layer["o_proj"], compute_kernel_config=self.compute_config))
            normed = ttnn.rms_norm(x, weight=layer["post_attention_layernorm"], epsilon=self.eps)
            x = ttnn.add(x, self._mlp(normed, layer))

        hidden = ttnn.rms_norm(x, weight=self.p["norm"], epsilon=self.eps)
        return ttnn.slice(hidden, [0, length - 1, 0], [1, length, self.hidden])

    def _step_ops(self, x):
        """One position through the five layers. The output head stays outside the trace."""
        x = project(x, self.p, self.compute_config)
        for index, layer in enumerate(self.p["layers"]):
            normed = self._norm(x, layer, "input_layernorm")
            qkv = ttnn.linear(
                normed, layer["qkv"], compute_kernel_config=self.compute_config, program_config=self._matmul["qkv"]
            )
            query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.reshape(qkv, (1, 1, 1, qkv.shape[-1])), num_heads=self.heads, num_kv_heads=self.kv_heads
            )
            query = ttnn.sharded_to_interleaved(query)
            key = ttnn.sharded_to_interleaved(key)
            query = self._rotate(ttnn.rms_norm(query, weight=layer["q_norm"], epsilon=self.eps), self._cos, self._sin)
            key = self._rotate(ttnn.rms_norm(key, weight=layer["k_norm"], epsilon=self.eps), self._cos, self._sin)

            ttnn.experimental.paged_fused_update_cache(
                self.k_cache[index],
                ttnn.to_memory_config(key, self._k_shard),
                self.v_cache[index],
                ttnn.to_memory_config(value, self._v_shard),
                update_idxs_tensor=self._pos,
                page_table=None,
            )
            attended = ttnn.transformer.scaled_dot_product_attention_decode(
                query,
                self.k_cache[index],
                self.v_cache[index],
                cur_pos_tensor=self._pos,
                scale=self.scale,
                compute_kernel_config=self.compute_config,
            )
            attended = ttnn.reshape(ttnn.experimental.nlp_concat_heads(attended), (1, 1, self.heads * self.head_dim))
            x = ttnn.add(
                x,
                ttnn.linear(
                    attended,
                    layer["o_proj"],
                    compute_kernel_config=self.compute_config,
                    program_config=self._matmul["o_proj"],
                ),
            )
            normed = self._norm(x, layer, "post_attention_layernorm")
            x = ttnn.add(x, self._mlp(normed, layer))
        return self._norm(x, {"norm": self.p["norm"], "norm_wide": getattr(self, "_final_norm_wide", None)}, "norm")

    def warmup(self):
        """Compile the step eagerly, without capturing.

        Separate from `capture` so a caller driving two decoders can warm both before
        capturing either. The warmup allocates a buffer per op, and a buffer allocated
        while another trace exists is corrupt once that trace runs unless released first.
        Warms at the last cache slot, which no real decode reaches, and leaves the cache
        alone so this can follow `prefill`.
        """
        self.set_position(self.max_seq - 1)
        self._step_ops(self._in)
        ttnn.synchronize_device(self.device)
        self._warmed = True

    def capture(self):
        """Capture the step so later ones replay a single graph.

        Every trace must be captured before any of them executes. Capturing a second trace
        after the first has run hangs the device: measured, and it took a board reset.
        """
        if not getattr(self, "_warmed", False):
            self.warmup()  # a trace cannot compile new programs
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._out = self._step_ops(self._in)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

    def release(self):
        """Drop the captured trace so eager work is safe again.

        A prefill runs ops a trace never compiled and allocates as it goes, and buffers
        allocated while a trace is live are corrupt once it runs. So a second utterance
        has to release before it prefills, and capture again afterwards.
        """
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
            self._warmed = False

    def reset(self):
        """Clear the caches without allocating."""
        for cache in self.k_cache + self.v_cache:
            ttnn.copy(self._zero_cache, cache)

    def _head(self, hidden, step):
        """Codebook logits for this step, from its own output head."""
        return ttnn.linear(hidden, self.p["lm_head"][step], compute_kernel_config=self.compute_config)

    def _run(self, position):
        """One traced position, over whatever `_in` now holds."""
        self.set_position(position)
        if self.trace_id is not None:
            ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
            return self._out
        return self._step_ops(self._in)

    def _fill_from_host(self, embedding):
        """`_in` <- a host row. The slow way in, kept for callers holding torch tensors."""
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(embedding.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._in
        )

    def _fill_from_table(self, table, code):
        """`_in` <- row `code` of a device codebook, without the row touching the host.

        One index write instead of 76 us of `from_torch`, for identical values.
        """
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[int(code)]], dtype=torch.int32), dtype=ttnn.uint32), self._index
        )
        row = ttnn.embedding(self._index, table, layout=ttnn.TILE_LAYOUT)
        ttnn.copy(ttnn.reshape(row, (1, 1, self.p["input_width"])), self._in)
        ttnn.deallocate(row)

    def generate(self, talker_hidden, first_code, pick=None, watch=None):
        """Codebooks 1 to 15, from the talker's hidden state and codebook 0.

        `talker_hidden` may be the device tensor the talker returned, which is what the pipeline
        passes. `pick` defaults to argmax, but greedy sends this model into a silence code it
        never leaves. The two prompt positions go through the traced step like the rest: a
        per-frame prefill allocated buffers, and buffers allocated under a live trace are corrupt
        once it runs.
        """
        charge = watch.split if watch is not None else (lambda name: None)
        self.reset()
        if isinstance(talker_hidden, ttnn.Tensor):
            ttnn.copy(ttnn.reshape(talker_hidden, (1, 1, -1)), self._in)
        else:
            self._fill_from_host(talker_hidden.reshape(1, 1, -1))
        self._run(0)
        self._fill_from_table(self.p["talker_codec_embedding_device"], first_code)
        hidden = self._run(1)

        pick = pick or (lambda row: int(row.argmax()))
        codes = []
        for step in range(self.groups - 1):
            logits = self._head(hidden, step)
            row = ttnn.to_torch(logits).float().reshape(-1)
            ttnn.deallocate(logits)  # released before the next trace execution
            charge("predictor")
            code = int(pick(row))
            charge("predictor_sample")
            codes.append(code)
            if step == self.groups - 2:
                break
            self._fill_from_table(self.p["codec_embedding_device"][step], code)
            hidden = self._run(2 + step)
        return codes
