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

With `sample_on_device`, the whole frame is one trace instead: the predictor's positions
are 0 to 15 every frame, so the 16 steps, the 15 heads and their sampling unroll into a
single graph, and a frame costs one noise write, one replay and one 15-code read. The
draw is Gumbel-max over the top k: host-drawn Gumbel noise added to the scaled logits,
then the argmax. That is exactly a sample from the softmax over the top k, which
`ttnn.sampling` is not: its bf16 cumulative sum never reaches the tail of a 50-way draw.
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
    NORM_NAMES,
    NORM_SHARD_HEIGHT,
    TILE_ROWS,
    decode_mlp,
    linear_plans,
    sharded_norm_plan,
)


def preprocess_cached_predictor_parameters(device, config=None, dtype=ttnn.bfloat16, mlp_dtype=MLP_WEIGHT_DTYPE):
    """Predictor weights with Q, K and V fused, mirroring the talker's repacking, and the
    layer linears DRAM-sharded."""
    talker_cfg = dict(config or checkpoint.talker_config())
    cfg = dict(talker_cfg["code_predictor_config"])
    state = checkpoint.load_prefixed(CODE_PREDICTOR_PREFIX)
    plans = linear_plans(device, cfg)

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

    def sharded(key, name, as_dtype=None):
        return plans[key].weight(state[f"{name}.weight"].t(), device, as_dtype or dtype)

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
                "qkv": plans["qkv"].weight(fused.t(), device, dtype),
                "o_proj": sharded("o_proj", f"{prefix}.o_proj"),
                "q_norm": norm(f"{prefix}.q_norm"),
                "k_norm": norm(f"{prefix}.k_norm"),
                "gate_proj": sharded("gate_proj", f"model.layers.{index}.mlp.gate_proj", mlp_dtype),
                "up_proj": sharded("up_proj", f"model.layers.{index}.mlp.up_proj", mlp_dtype),
                "down_proj": sharded("down_proj", f"model.layers.{index}.mlp.down_proj", mlp_dtype),
            }
        )

    heads = cfg["num_code_groups"] - 1
    talker_table = checkpoint.load_prefixed(TALKER_CODEC_EMBEDDING)["weight"]
    predictor_tables = [state[f"model.codec_embedding.{index}.weight"] for index in range(heads)]
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
        # The norm weights again, on host, for the decoder's width-sharded copies.
        "norm_weights": {
            n: state[n]
            for n in [f"model.layers.{i}.{name}.weight" for i in range(cfg["num_hidden_layers"]) for name in NORM_NAMES]
            + ["model.norm.weight"]
        },
    }


# The candidates one device top-k call gives the frame trace. A top k of up to this samples
# on device; ids tied with the k-th value stay in, as upstream's top-k keeps them, as long
# as the tie ends within these candidates.
DEVICE_TOP_K_LIMIT = 64


class TtCodePredictorCachedDecoder:
    """Prefill two positions, then 14 cached steps, one output head each.

    A frame runs `batch` rows together, one request each. Every row is at the same position
    at every step, so the rows share the rotary tables and only the caches, the inputs and
    the samples are per row.
    """

    def __init__(self, device, parameters, batch=1):
        grid = device.compute_with_storage_grid_size()
        if not 1 <= batch <= grid.x:
            raise ValueError(f"a batch of {batch} rows: the step shards one row per core across {grid.x} columns")
        self.batch = batch
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
        empty = torch.zeros(batch, self.kv_heads, self.max_seq, self.head_dim)
        self.k_cache = [
            ttnn.from_torch(empty, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            for _ in range(self.layers)
        ]
        self.v_cache = [
            ttnn.from_torch(empty, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            for _ in range(self.layers)
        ]

        self._pos = ttnn.from_torch(torch.zeros(batch, dtype=torch.int32), device=device)
        self._in = ttnn.from_torch(
            torch.zeros(1, batch, parameters["input_width"]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._cos = ttnn.from_torch(
            torch.zeros(1, 1, TILE_ROWS, self.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self._sin = ttnn.from_torch(
            torch.zeros(1, 1, TILE_ROWS, self.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self._k_shard = self._row_shard(0)
        self._v_shard = self._row_shard(1)
        self._zero_cache = ttnn.from_torch(empty, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        self._linear = linear_plans(device, self.config)
        self._norm_plan = sharded_norm_plan(device, self.hidden)
        if self._norm_plan is not None:
            state = parameters["norm_weights"]
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
            torch.zeros(batch, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._position_inputs = {}

        self.trace_id = None
        self._out = None
        self._warmed = False
        # Set by `sample_on_device`: the frame trace's sampling settings and its inputs.
        self._device_sampling = None

    def _row_shard(self, row):
        """One tile of rows per batch row, on cores (0..batch-1, row): the cache update needs
        K and V sharded, and the fused update needs the two on different cores."""
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, row), ttnn.CoreCoord(self.batch - 1, row))}),
                (TILE_ROWS, self.head_dim),
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
        """The same fused rotation the talker's prefill uses, over 150 rotations a frame.

        A step's [1, batch, heads, head_dim] reads as `batch` heads of `heads` rows; every row
        of `cos` holds the one position all batch rows are at.
        """
        return ttnn.experimental.rotary_embedding_hf(
            x, cos, sin, is_decode_mode=False, compute_kernel_config=self.compute_config
        )

    def _project(self, x, layer, name):
        """One layer linear, DRAM-sharded; the output is width-sharded in L1."""
        return self._linear[name](x, layer[name])

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
                table.reshape(1, 1, 1, -1).expand(1, 1, TILE_ROWS, self.head_dim).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            self._position_inputs[position] = (
                ttnn.from_torch(torch.full((self.batch,), int(position), dtype=torch.int32)),
                rows(cos),
                rows(sin),
            )
        return self._position_inputs[position]

    def set_position(self, position):
        pos, cos, sin = self._host_position_inputs(position)
        ttnn.copy_host_to_device_tensor(pos, self._pos)
        ttnn.copy_host_to_device_tensor(cos, self._cos)
        ttnn.copy_host_to_device_tensor(sin, self._sin)

    def prefill(self, embeddings, row=0):
        """Seed batch row `row`'s cache from the two-position prompt and return its last
        hidden state."""
        length = embeddings.shape[1]
        cos, sin = self._rotary(length)
        to_device = lambda tensor: ttnn.from_torch(
            tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        x = to_device(embeddings)
        cos_tt, sin_tt, mask_tt = to_device(cos), to_device(sin), to_device(causal_mask(length))

        if length > TILE_ROWS:
            raise ValueError(f"a {length}-position prompt: the predictor's linears take up to {TILE_ROWS} rows")
        x = project(x, self.p, self.compute_config)
        for index, layer in enumerate(self.p["layers"]):
            normed = ttnn.rms_norm(x, weight=layer["input_layernorm"], epsilon=self.eps)
            qkv = ttnn.sharded_to_interleaved(self._project(normed, layer, "qkv"))

            def split(start, heads, width):
                piece = ttnn.slice(qkv, [0, 0, start], [1, length, start + width])
                return ttnn.permute(ttnn.reshape(piece, (1, length, heads, self.head_dim)), (0, 2, 1, 3))

            q_width, kv_width = self.heads * self.head_dim, self.kv_heads * self.head_dim
            query = split(0, self.heads, q_width)
            key = split(q_width, self.kv_heads, kv_width)
            value = split(q_width + kv_width, self.kv_heads, kv_width)

            query = self._rotate(ttnn.rms_norm(query, weight=layer["q_norm"], epsilon=self.eps), cos_tt, sin_tt)
            key = self._rotate(ttnn.rms_norm(key, weight=layer["k_norm"], epsilon=self.eps), cos_tt, sin_tt)

            ttnn.fill_cache(self.k_cache[index], key, row)
            ttnn.fill_cache(self.v_cache[index], value, row)

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
            x = ttnn.add(x, ttnn.sharded_to_interleaved(self._project(attended, layer, "o_proj")))
            normed = ttnn.rms_norm(x, weight=layer["post_attention_layernorm"], epsilon=self.eps)
            x = ttnn.add(x, decode_mlp(normed, layer, self._linear))

        hidden = ttnn.rms_norm(x, weight=self.p["norm"], epsilon=self.eps)
        return ttnn.slice(hidden, [0, length - 1, 0], [1, length, self.hidden])

    def _step_ops(self, x, position=None):
        """One position through the five layers. The output head stays outside the trace.

        `position` is a (pos, cos, sin) triple of device tensors; the default is the ones
        `set_position` writes.
        """
        pos, cos, sin = position or (self._pos, self._cos, self._sin)
        x = project(x, self.p, self.compute_config)
        for index, layer in enumerate(self.p["layers"]):
            normed = self._norm(x, layer, "input_layernorm")
            qkv = ttnn.sharded_to_interleaved(self._project(normed, layer, "qkv"))
            query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.reshape(qkv, (1, 1, self.batch, qkv.shape[-1])),
                num_heads=self.heads,
                num_kv_heads=self.kv_heads,
            )
            query = ttnn.sharded_to_interleaved(query)
            key = ttnn.sharded_to_interleaved(key)
            query = self._rotate(ttnn.rms_norm(query, weight=layer["q_norm"], epsilon=self.eps), cos, sin)
            key = self._rotate(ttnn.rms_norm(key, weight=layer["k_norm"], epsilon=self.eps), cos, sin)

            ttnn.experimental.paged_fused_update_cache(
                self.k_cache[index],
                ttnn.to_memory_config(key, self._k_shard),
                self.v_cache[index],
                ttnn.to_memory_config(value, self._v_shard),
                update_idxs_tensor=pos,
                page_table=None,
            )
            attended = ttnn.transformer.scaled_dot_product_attention_decode(
                query,
                self.k_cache[index],
                self.v_cache[index],
                cur_pos_tensor=pos,
                scale=self.scale,
                compute_kernel_config=self.compute_config,
            )
            attended = ttnn.reshape(attended, (1, self.batch, self.heads * self.head_dim))
            x = ttnn.add(x, ttnn.sharded_to_interleaved(self._project(attended, layer, "o_proj")))
            normed = self._norm(x, layer, "post_attention_layernorm")
            x = ttnn.add(x, decode_mlp(normed, layer, self._linear))
        return self._norm(x, {"norm": self.p["norm"], "norm_wide": getattr(self, "_final_norm_wide", None)}, "norm")

    def warmup(self):
        """Compile the step eagerly, without capturing.

        Separate from `capture` so a caller driving two decoders can warm both before
        capturing either. The warmup allocates a buffer per op, and a buffer allocated
        while another trace exists is corrupt once that trace runs unless released first.
        Warms at the last cache slot, which no real decode reaches, and leaves the cache
        alone so this can follow `prefill`.

        Once is enough: the programs stay in the device's program cache when the trace is
        released, so this does nothing again until `forget_programs`. A frame's warmup is
        a whole eager frame, about 65 ms.
        """
        if self._warmed:
            return
        if self._device_sampling is not None:
            self._frame_ops()
        else:
            self.set_position(self.max_seq - 1)
            self._step_ops(self._in)
        ttnn.synchronize_device(self.device)
        self._warmed = True

    def capture(self):
        """Capture the step so later ones replay a single graph.

        Every trace must be captured before any of them executes. Capturing a second trace
        after the first has run hangs the device: measured, and it took a board reset.
        """
        self.warmup()  # a trace cannot compile new programs
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._out = self._frame_ops() if self._device_sampling is not None else self._step_ops(self._in)
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

    def forget_programs(self):
        """Note that the device's program cache was cleared, so the next capture warms again."""
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

    def sample_on_device(self, temperature, top_k):
        """Sample every codebook inside one frame trace, with top-k and temperature.

        `top_k` of 1 is argmax. Must be called before `warmup` and `capture`.
        """
        if not 0 < top_k <= DEVICE_TOP_K_LIMIT:
            raise ValueError(f"device sampling takes 0 < top_k <= {DEVICE_TOP_K_LIMIT}, got {top_k}")
        if temperature <= 0:
            raise ValueError(f"device sampling needs a positive temperature, got {temperature}")
        if self.trace_id is not None:
            raise RuntimeError("release the predictor's trace before changing how it samples")
        steps = self.groups - 1
        # Every position's inputs on device, built once: the frame trace reads them directly.
        positions = []
        for position in range(self.groups):
            pos, cos, sin = self._host_position_inputs(position)
            positions.append(tuple(tensor.to(self.device) for tensor in (pos, cos, sin)))
        self._device_sampling = {
            "temperature": float(temperature),
            "top_k": int(top_k),
            "positions": positions,
            "noise": ttnn.from_torch(
                torch.zeros(1, steps, self.batch, DEVICE_TOP_K_LIMIT),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            ),
        }
        self._warmed = False

    def _sample(self, logits, step):
        """One code per row from a head's logits: argmax of the scaled top k plus this step's
        noise.

        Returns the codes as a float32 [1, 1, batch, 1], exact, as every codebook index is far
        below float32's 2^24. Of tied best scores, which only argmax meets, the lowest id
        wins, as `torch.argmax` picks. The top k is stable so that id stays a candidate
        however many logits tie.
        """
        settings = self._device_sampling
        k = settings["top_k"]
        values, indices = ttnn.topk(
            ttnn.reshape(logits, (1, 1, self.batch, -1)), k=DEVICE_TOP_K_LIMIT, dim=-1, stable=True
        )
        values = ttnn.typecast(values, ttnn.float32)
        floor = ttnn.slice(values, [0, 0, 0, k - 1], [1, 1, self.batch, k])
        noise = ttnn.slice(settings["noise"], [0, step, 0, 0], [1, step + 1, self.batch, DEVICE_TOP_K_LIMIT])
        scores = ttnn.add(ttnn.multiply(values, 1.0 / settings["temperature"]), noise)
        # Below the k-th value is out; -1e30 rather than -inf keeps the max and eq finite.
        scores = ttnn.where(ttnn.ge(values, floor), scores, -1e30)
        hit = ttnn.eq(scores, ttnn.max(scores, dim=-1, keepdim=True))
        return ttnn.min(ttnn.where(hit, ttnn.typecast(indices, ttnn.float32), 1e9), dim=-1, keepdim=True)

    def _embed_code(self, table, code):
        """`table`'s row for each batch row's code, which the device holds, shaped as a step input."""
        index = ttnn.reshape(ttnn.to_layout(ttnn.typecast(code, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (self.batch, 1))
        row = ttnn.embedding(index, table, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(row, (1, self.batch, self.p["input_width"]))

    def _frame_ops(self):
        """A whole frame per row from `_in` (the talker's hidden states) and `_index`
        (codebook 0).

        Returns codebooks 1 to 15 as a uint32 [1, 1, batch, 15].
        """
        positions = self._device_sampling["positions"]
        for cache in self.k_cache + self.v_cache:
            ttnn.copy(self._zero_cache, cache)
        self._step_ops(self._in, positions[0])
        first = ttnn.embedding(self._index, self.p["talker_codec_embedding_device"], layout=ttnn.TILE_LAYOUT)
        hidden = self._step_ops(ttnn.reshape(first, (1, self.batch, self.p["input_width"])), positions[1])
        codes = []
        for step in range(self.groups - 1):
            code = self._sample(self._head(hidden, step), step)
            codes.append(code)
            if step == self.groups - 2:
                break
            hidden = self._step_ops(self._embed_code(self.p["codec_embedding_device"][step], code), positions[2 + step])
        return ttnn.typecast(ttnn.concat(codes, dim=-1), ttnn.uint32)

    def generate_rows(self, talker_hidden, first_codes, watch=None, generator=None):
        """Codebooks 1 to 15 for every batch row: talker hidden states [1, batch, hidden] and
        one codebook 0 per row in, one list of 15 codes per row out. Needs `sample_on_device`."""
        if self._device_sampling is None:
            raise RuntimeError("a batch of rows samples on device; call sample_on_device first")
        if len(first_codes) != self.batch:
            raise ValueError(f"{len(first_codes)} first codes for a batch of {self.batch}")
        settings = self._device_sampling
        if settings["top_k"] == 1:
            gumbel = torch.zeros(1, self.groups - 1, self.batch, DEVICE_TOP_K_LIMIT)
        else:
            uniform = torch.rand(1, self.groups - 1, self.batch, DEVICE_TOP_K_LIMIT, generator=generator)
            # Clamped off 0 and 1, where the double log is infinite.
            gumbel = -torch.log(-torch.log(uniform.clamp_(1e-10, 1 - 1e-7)))
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(gumbel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), self._device_sampling["noise"]
        )
        firsts = torch.tensor([[int(code)] for code in first_codes], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(firsts, dtype=ttnn.uint32), self._index)
        if isinstance(talker_hidden, ttnn.Tensor):
            ttnn.copy(ttnn.reshape(talker_hidden, (1, self.batch, -1)), self._in)
        else:
            self._fill_from_host(talker_hidden.reshape(1, self.batch, -1))
        if self.trace_id is not None:
            ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
            codes = self._out
        else:
            codes = self._frame_ops()
        rows = ttnn.to_torch(codes).reshape(self.batch, -1).tolist()
        if watch is not None:
            watch.split("predictor")
        return [[int(code) for code in row] for row in rows]

    def generate(self, talker_hidden, first_code, pick=None, watch=None, generator=None):
        """Codebooks 1 to 15, from the talker's hidden state and codebook 0.

        `talker_hidden` may be the device tensor the talker returned, which is what the pipeline
        passes. `pick` defaults to argmax, but greedy sends this model into a silence code it
        never leaves. The two prompt positions go through the traced step like the rest: a
        per-frame prefill allocated buffers, and buffers allocated under a live trace are corrupt
        once it runs.

        Under `sample_on_device` the settings given there apply instead of `pick`, and
        `generator` draws the frame's noise, so a seeded one makes frames reproducible.
        """
        if self.batch != 1:
            raise RuntimeError(f"generate runs one row; a batch of {self.batch} runs through generate_rows")
        if self._device_sampling is not None:
            return self.generate_rows(talker_hidden, [first_code], watch, generator)[0]
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
