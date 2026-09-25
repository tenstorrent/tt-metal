# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""KV-cached autoregressive decode for the Qwen3-TTS talker.

`ttnn_qwen3_talker.py` recomputes the whole prompt every step, which is what made the
correctness work simple and what makes generation quadratic. This adds the two passes a
request actually runs: a one-shot prefill that seeds a per-layer on-device KV cache, and a
single-position step that reads it.

At step t only the new position's Q/K/V are computed. Its K/V append to a preallocated
cache and attention runs over 0..t through flash decode, so the prefix is never pushed
through the projections again.

Equivalence check: attention is causal, so the hidden state the cached step produces at
position t must equal position t of the full recompute. `tests/pcc/test_talker_decode.py`
holds it to that.

Three things differ from the same trick applied to a plain GPT:

  * **GQA.** 16 query heads over 8 key/value heads, so the cache holds 8 heads, not 16,
    and flash decode broadcasts them.
  * **QK-norm.** Each layer RMS-normalises Q and K over head_dim before the rotation, and
    the cache therefore stores post-norm, post-rotation K.
  * **A rotation at all.** Position enters through cos and sin tables rather than learned
    embeddings, so the step needs the current position's row. The talker's three MRoPE axes
    carry identical positions during generation, which is why plain tables suffice.

Two hazards inherited rather than rediscovered, both from the XTTS-v2 GPT port:

  * `scaled_dot_product_attention_decode` is wrong when the cache length is an odd number
    of 32-row tiles, so `max_seq` rounds up to a multiple of 64.
  * `ttnn.fill_cache` hands each core only its first cache address and walks forward, so a
    write run crossing a head boundary lands in the previous head and leaves positions
    silently zero, identically on every repeat. Safe only when every core gets at most one
    block or the blocks divide evenly; otherwise the write is chunked.
"""

import torch

import ttnn
from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import _compute_config, causal_mask, rotary_tables

# sdpa_decode miscomputes at an odd tile count; keep the cache an even number of tiles.
CACHE_TILE_MULTIPLE = 64


# The K reduction is cut into this many tiles per step so the weight stream pipelines
# against the math, and out_subblock_h * w must stay under this ceiling because
# fp32_dest_acc_en halves the register budget.
DECODE_IN0_BLOCK_W = 4

DECODE_MAX_SUBBLOCK = 4

# Output tiles per core, which sets each matmul's row count. Flat from 4 to 6, worse outside.
TARGET_PER_CORE_N = 5

# The MLP's matmuls only: a fifth off the step at no measurable cost. README has the rest.
MLP_WEIGHT_DTYPE = ttnn.bfloat8_b

# Rows in the step's rotary tables: the fused rotation pairs row i of `cos` with row i of
# its input, so one tile's worth covers every head count here.
ROPE_ROWS = 32


def decode_matmul_config(device, in_features, out_features, fused_activation=None):
    """1D-multicast matmul config for a single-position (M=1) decode linear.

    A few full grid rows, each core holding a wide slice of the output; None when the
    reduction will not chunk. Swept across all eight decode shapes: every winner spends 11 to
    22 cores, not the 64 an exact-division search picks, because at M=1 the multicast
    dominates. `down_proj` went 112.8 to 69.6 us. Dropping divisibility is what makes the
    11-wide grid reachable, since no width here divides by 11.

    `fused_activation` must travel in the config: `activation=` beside a program config runs
    a second kernel instead of fusing.
    """
    k_tiles, n_tiles = in_features // 32, out_features // 32
    grid = device.compute_with_storage_grid_size()

    # A long reduction pipelines better in bigger chunks: down_proj, 69.6 us at 8 against 74.3.
    in0_block_w = 8 if k_tiles % 8 == 0 and k_tiles > 64 else DECODE_IN0_BLOCK_W
    if k_tiles % in0_block_w:
        return None

    cols = grid.x
    rows = max(1, min(grid.y, round(n_tiles / (cols * TARGET_PER_CORE_N))))
    per_core_n = -(-n_tiles // (cols * rows))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=next(w for w in range(min(per_core_n, DECODE_MAX_SUBBLOCK), 0, -1) if per_core_n % w == 0),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def preprocess_cached_talker_parameters(
    device, config=None, state=None, num_layers=None, dtype=ttnn.bfloat16, mlp_dtype=MLP_WEIGHT_DTYPE
):
    """Weights for the cached decoder, with Q, K and V fused into one matmul.

    Purely a repacking: three matmuls become one per layer and one weight streams instead of
    three. `mlp_dtype` is the split precision; pass bf16 to measure against a uniform model.
    """
    cfg = dict(config or checkpoint.talker_config())
    if num_layers is not None:
        cfg["num_hidden_layers"] = num_layers
    state = checkpoint.load_talker_state() if state is None else state

    def to_device(tensor, layout=ttnn.TILE_LAYOUT, as_dtype=None):
        return ttnn.from_torch(
            tensor.contiguous(),
            dtype=as_dtype or dtype,
            layout=layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def linear(name, as_dtype=None):
        return to_device(state[f"{name}.weight"].t(), as_dtype=as_dtype)

    def norm(name):
        return to_device(state[f"{name}.weight"].reshape(1, 1, 1, -1))

    layers = []
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"layers.{index}.self_attn"
        fused = torch.cat(
            [state[f"{prefix}.q_proj.weight"], state[f"{prefix}.k_proj.weight"], state[f"{prefix}.v_proj.weight"]],
            dim=0,
        )  # [(heads + 2 * kv_heads) * head_dim, hidden]
        layers.append(
            {
                "input_layernorm": norm(f"layers.{index}.input_layernorm"),
                "post_attention_layernorm": norm(f"layers.{index}.post_attention_layernorm"),
                "qkv": to_device(fused.t()),
                "o_proj": linear(f"{prefix}.o_proj"),
                "q_norm": norm(f"{prefix}.q_norm"),
                "k_norm": norm(f"{prefix}.k_norm"),
                "gate_proj": linear(f"layers.{index}.mlp.gate_proj", mlp_dtype),
                "up_proj": linear(f"layers.{index}.mlp.up_proj", mlp_dtype),
                "down_proj": linear(f"layers.{index}.mlp.down_proj", mlp_dtype),
            }
        )

    return {"config": cfg, "layers": layers, "norm": norm("norm")}


def fill_step_tiles(kv_heads, cores):
    """Largest per-call tile count at which `ttnn.fill_cache` seeds every position.

    See the hazard note in the module docstring: no write run can straddle a head boundary
    when each core gets at most one block.
    """
    return max(1, cores // kv_heads)


# Width-sharded RMSNorm for the single-position step. An interleaved norm on one [1, 1, H]
# row runs effectively single-core, and the step does 57 of them at hidden 2048 (two per
# layer plus the final one), so sharding the reduction is worth it at equal PCC. The
# per-head QK-norms stay interleaved: they are only 128 wide, where sharding would cost
# more in resharding than the reduction saves.
NORM_SHARD_HEIGHT = 32


def sharded_norm_plan(device, dim, dtype=ttnn.bfloat16):
    """Memory and program config for a width-sharded norm over `dim`, or None if it will
    not tile cleanly onto this grid."""
    grid = device.compute_with_storage_grid_size()
    height = NORM_SHARD_HEIGHT
    if dim % height or (dim // height) % 32 or height % 8 or grid.x < 8 or grid.y < height // 8:
        return None
    width_per_core = dim // height
    core_grid = ttnn.CoreGrid(x=8, y=height // 8)
    memory_config = ttnn.create_sharded_memory_config(
        shape=(height, width_per_core),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[core_grid.x, core_grid.y],
        subblock_w=width_per_core // 32,
        block_h=height // 32,
        block_w=width_per_core // 32,
        inplace=False,
    )
    return memory_config, program_config


class TtTalkerCachedDecoder:
    """Prefill once into an on-device KV cache, then step one position at a time."""

    def __init__(self, device, parameters, max_seq=512):
        self.device = device
        self.p = parameters
        self.config = parameters["config"]
        self.compute_config = _compute_config(device)

        self.heads = self.config["num_attention_heads"]
        self.kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.config["head_dim"]
        self.hidden = self.config["hidden_size"]
        self.eps = self.config["rms_norm_eps"]
        self.scale = self.head_dim**-0.5
        self.layers = len(parameters["layers"])

        self.max_seq = ((max_seq + CACHE_TILE_MULTIPLE - 1) // CACHE_TILE_MULTIPLE) * CACHE_TILE_MULTIPLE
        empty = torch.zeros(1, self.kv_heads, self.max_seq, self.head_dim)
        self.k_cache = [self._cache_tensor(empty) for _ in range(self.layers)]
        self.v_cache = [self._cache_tensor(empty) for _ in range(self.layers)]

        # Stable per-step state, allocated once so a captured trace can read it. Allocating
        # under a live trace corrupts it, so everything the step touches exists up front.
        self._pos = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), device=device)
        self._in = ttnn.from_torch(
            torch.zeros(1, 1, self.hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        # One row per head, all identical: the fused rotation reads a row per row of its
        # input, and the step's Q carries `heads` of them. 32 covers both Q and K.
        self._cos = ttnn.from_torch(
            torch.zeros(1, 1, ROPE_ROWS, self.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self._sin = ttnn.from_torch(
            torch.zeros(1, 1, ROPE_ROWS, self.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        # `paged_update_cache` requires a sharded input, so the step hands it L1-sharded K and
        # V. They go on different cores because the fused K+V variant needs them not to
        # overlap, which keeps that swap available later.
        self._k_shard = self._single_core_shard(0)
        self._v_shard = self._single_core_shard(1)
        # One zero tensor to clear the caches from. `zeros_like` would allocate a fresh
        # device buffer per cache per utterance, and a buffer allocated while a trace is
        # live is corrupt once that trace runs unless it is released first.
        self._zero_cache = self._cache_tensor(empty)

        # Decode-only matmul configs. gate_proj carries the silu so the activation runs
        # inside the matmul rather than as a separate elementwise kernel.
        silu = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        first = parameters["layers"][0]
        self._matmul = {
            name: decode_matmul_config(device, *tuple(first[name].shape)[-2:], silu if name == "gate_proj" else None)
            for name in ("qkv", "o_proj", "gate_proj", "up_proj", "down_proj")
        }

        # The sharded norm wants its weight broadcast over the shard-height tile.
        self._norm_plan = sharded_norm_plan(device, self.hidden)
        if self._norm_plan is not None:
            state = checkpoint.load_talker_state()
            expand = lambda name: ttnn.from_torch(
                state[name].reshape(1, 1, -1).expand(1, NORM_SHARD_HEIGHT, self.hidden).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            for index, layer in enumerate(parameters["layers"]):
                layer["input_layernorm_wide"] = expand(f"layers.{index}.input_layernorm.weight")
                layer["post_attention_layernorm_wide"] = expand(f"layers.{index}.post_attention_layernorm.weight")
            self._final_norm_wide = expand("norm.weight")

        self.trace_id = None
        self._out = None

    def _norm(self, x, layer, key):
        """Width-sharded RMSNorm when the plan fits, otherwise the interleaved one."""
        if self._norm_plan is None:
            return ttnn.rms_norm(x, weight=layer[key], epsilon=self.eps)
        memory_config, program_config = self._norm_plan
        sharded = ttnn.interleaved_to_sharded(x, memory_config)
        out = ttnn.rms_norm(
            sharded,
            weight=layer[f"{key}_wide"],
            epsilon=self.eps,
            program_config=program_config,
            memory_config=memory_config,
            compute_kernel_config=self.compute_config,
        )
        return ttnn.sharded_to_interleaved(out)

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

    def _cache_tensor(self, empty):
        return ttnn.from_torch(empty, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    # ── shared pieces ───────────────────────────────────────────────────────

    def _rotate(self, x, cos, sin):
        """RoPE in one kernel, where the arithmetic spelled out took seven ops: 7.8 us against 26.7.

        `is_decode_mode=False` despite serving the step. Decode mode wants one row per batch slot
        and at batch 1 the step's rows are heads, so `cos` carries a row per head instead.
        """
        return ttnn.experimental.rotary_embedding_hf(
            x, cos, sin, is_decode_mode=False, compute_kernel_config=self.compute_config
        )

    def _mlp(self, x, layer, configs=None):
        configs = configs or {}
        gate = ttnn.linear(
            x, layer["gate_proj"], compute_kernel_config=self.compute_config, program_config=configs.get("gate_proj")
        )
        # The silu is already fused when a program config carries it; applying it again
        # would double up, and applying it never would be wrong.
        if configs.get("gate_proj") is None or configs["gate_proj"].fused_activation is None:
            gate = ttnn.silu(gate)
        up = ttnn.linear(
            x, layer["up_proj"], compute_kernel_config=self.compute_config, program_config=configs.get("up_proj")
        )
        return ttnn.linear(
            ttnn.multiply(gate, up),
            layer["down_proj"],
            compute_kernel_config=self.compute_config,
            program_config=configs.get("down_proj"),
        )

    def _split_qkv(self, qkv, length):
        """[1, L, 4096] -> q [1, 16, L, 128], k/v [1, 8, L, 128]."""
        q_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim

        def take(start, heads, width):
            piece = ttnn.slice(qkv, [0, 0, start], [1, length, start + width])
            piece = ttnn.reshape(piece, (1, length, heads, self.head_dim))
            return ttnn.permute(piece, (0, 2, 1, 3))

        return (
            take(0, self.heads, q_width),
            take(q_width, self.kv_heads, kv_width),
            take(q_width + kv_width, self.kv_heads, kv_width),
        )

    # ── prefill ─────────────────────────────────────────────────────────────

    def prefill(self, embeddings, position_ids=None):
        """Seed the cache for positions 0..P-1 in one pass and return the hidden states.

        Each layer's K/V weights are read once here rather than once per position, and the
        cache is written with `ttnn.fill_cache`, chunked when a single shot would let a
        write run cross a head boundary.
        """
        length = embeddings.shape[1]
        if length > self.max_seq:
            raise ValueError(f"prompt of {length} exceeds the cache's {self.max_seq}")

        cos, sin = rotary_tables(self.config, position_ids) if position_ids is not None else (None, None)
        if cos is None:
            positions = torch.arange(length, dtype=torch.long).reshape(1, 1, length)
            cos, sin = rotary_tables(self.config, positions.expand(3, 1, length).contiguous())

        to_device = lambda tensor: ttnn.from_torch(
            tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        x = to_device(embeddings) if not isinstance(embeddings, ttnn.Tensor) else embeddings
        cos_tt, sin_tt, mask_tt = to_device(cos), to_device(sin), to_device(causal_mask(length))

        grid = self.device.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        tiles = (length + 31) // 32
        single_shot = self.kv_heads * tiles <= cores or (self.kv_heads * tiles) % cores == 0
        step = fill_step_tiles(self.kv_heads, cores)

        for index, layer in enumerate(self.p["layers"]):
            normed = ttnn.rms_norm(x, weight=layer["input_layernorm"], epsilon=self.eps)
            qkv = ttnn.linear(normed, layer["qkv"], compute_kernel_config=self.compute_config)
            query, key, value = self._split_qkv(qkv, length)

            query = self._rotate(ttnn.rms_norm(query, weight=layer["q_norm"], epsilon=self.eps), cos_tt, sin_tt)
            key = self._rotate(ttnn.rms_norm(key, weight=layer["k_norm"], epsilon=self.eps), cos_tt, sin_tt)

            self._seed_cache(index, key, value, length, tiles, single_shot, step)

            repeats = self.heads // self.kv_heads
            attended = ttnn.matmul(
                query,
                ttnn.permute(ttnn.repeat_interleave(key, repeats, dim=1), (0, 1, 3, 2)),
                compute_kernel_config=self.compute_config,
            )
            attended = ttnn.softmax(
                ttnn.add(ttnn.multiply(attended, self.scale), mask_tt),
                dim=-1,
                compute_kernel_config=self.compute_config,
            )
            attended = ttnn.matmul(
                attended, ttnn.repeat_interleave(value, repeats, dim=1), compute_kernel_config=self.compute_config
            )
            attended = ttnn.reshape(ttnn.permute(attended, (0, 2, 1, 3)), (1, length, self.heads * self.head_dim))

            x = ttnn.add(x, ttnn.linear(attended, layer["o_proj"], compute_kernel_config=self.compute_config))
            normed = ttnn.rms_norm(x, weight=layer["post_attention_layernorm"], epsilon=self.eps)
            x = ttnn.add(x, self._mlp(normed, layer))

        return ttnn.rms_norm(x, weight=self.p["norm"], epsilon=self.eps)

    def _seed_cache(self, index, key, value, length, tiles, single_shot, step):
        """Write positions 0..length-1 of one layer's cache."""
        if single_shot:
            ttnn.fill_cache(self.k_cache[index], key, 0)
            ttnn.fill_cache(self.v_cache[index], value, 0)
            return
        for start in range(0, tiles, step):
            first, last = 32 * start, min(32 * (start + step), length)
            for cache, source in ((self.k_cache[index], key), (self.v_cache[index], value)):
                chunk = ttnn.slice(source, (0, 0, first, 0), (1, self.kv_heads, last, self.head_dim))
                ttnn.fill_cache(cache, chunk, 0, update_idx=first)
                ttnn.deallocate(chunk)

    # ── one cached step ─────────────────────────────────────────────────────

    def _step_ops(self, x):
        for index, layer in enumerate(self.p["layers"]):
            normed = self._norm(x, layer, "input_layernorm")
            qkv = ttnn.linear(
                normed, layer["qkv"], compute_kernel_config=self.compute_config, program_config=self._matmul["qkv"]
            )
            # Fused per-head split. Emits [1, 1, heads, head_dim] height-sharded in L1, which
            # is exactly the layout flash decode and the cache update want, so no permutes
            # appear in the step at all. It replaces 3 slices, 3 reshapes and 3 permutes.
            query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.reshape(qkv, (1, 1, 1, qkv.shape[-1])), num_heads=self.heads, num_kv_heads=self.kv_heads
            )
            # QK-norm cannot read height-sharded input, so Q and K come back interleaved for
            # the norm and the rotation, then K and V are resharded for the cache write.
            query = ttnn.sharded_to_interleaved(query)
            key = ttnn.sharded_to_interleaved(key)
            query = self._rotate(ttnn.rms_norm(query, weight=layer["q_norm"], epsilon=self.eps), self._cos, self._sin)
            key = self._rotate(ttnn.rms_norm(key, weight=layer["k_norm"], epsilon=self.eps), self._cos, self._sin)

            # The cache therefore stores post-norm, post-rotation K, which is what attention
            # over the prefix expects to find there. One kernel writes both, which needs them
            # on non-overlapping cores, hence the two single-core shards.
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
            # Fused head merge instead of permute + reshape.
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
            x = ttnn.add(x, self._mlp(normed, layer, self._matmul))
        return self._norm(x, {"norm": self.p["norm"], "norm_wide": getattr(self, "_final_norm_wide", None)}, "norm")

    def set_position(self, position):
        """Where the step writes and how far attention reads."""
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.full((1,), int(position), dtype=torch.int32)), self._pos)
        cos, sin = rotary_tables(self.config, torch.full((3, 1, 1), int(position), dtype=torch.long))
        for source, target in ((cos, self._cos), (sin, self._sin)):
            rows = source.reshape(1, 1, 1, -1).expand(1, 1, ROPE_ROWS, self.head_dim).contiguous()
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), target)

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

    def step(self, embedding, position):
        """One position: embedding [1, 1, hidden] -> hidden state [1, 1, hidden]."""
        # A host tensor copied into the stable input, rather than a fresh device buffer.
        if isinstance(embedding, ttnn.Tensor):
            ttnn.copy(embedding, self._in)
        else:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(embedding.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._in
            )
        self.set_position(position)
        if self.trace_id is not None:
            ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
            return self._out
        return self._step_ops(self._in)

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
