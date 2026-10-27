# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""KV-cached autoregressive decode for the Qwen3-TTS talker.

`ttnn_qwen3_talker.py` recomputes the whole prompt every step, which is what made the
correctness work simple and what makes generation quadratic. This adds the two passes a
request actually runs: a one-shot prefill that seeds a per-layer on-device KV cache, and a
single-position step that reads it.

The cache is paged, `[blocks, kv_heads, block_size, head_dim]` per layer, and every read and
write goes through a page table. The decoder allocates its own cache with an identity table
unless it is handed one, which is how a serving runtime that owns the KV blocks (vLLM) uses it.

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

A hazard inherited rather than rediscovered, from the XTTS-v2 GPT port:
`scaled_dot_product_attention_decode` is wrong when the cache length is an odd number of
32-row tiles, so `max_seq` rounds up to a multiple of 64.
"""

import torch

import ttnn
from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import _compute_config, causal_mask, rotary_tables

# sdpa_decode miscomputes at an odd tile count; keep the cache an even number of tiles.
CACHE_TILE_MULTIPLE = 64

# Positions per block of a cache the decoder allocates itself. A prefill pads the prompt to a
# 32-row tile and fills the padding too, so any block size must be a multiple of 32 for that
# fill to stay inside the request's last block.
OWNED_BLOCK_SIZE = 64

NORM_NAMES = ("input_layernorm", "post_attention_layernorm")


# The K reduction is cut into this many tiles per step so the weight stream pipelines
# against the math, and out_subblock_h * w must stay under this ceiling because
# fp32_dest_acc_en halves the register budget.
DECODE_IN0_BLOCK_W = 4

DECODE_MAX_SUBBLOCK = 4

# Output tiles per core, which sets each matmul's row count. Flat from 4 to 6, worse outside.
TARGET_PER_CORE_N = 5

# The MLP's matmuls only: a fifth off the step at no measurable cost. README has the rest.
MLP_WEIGHT_DTYPE = ttnn.bfloat8_b

# Rows a decode step's batch, or one batch row's heads, are padded to: one tile.
TILE_ROWS = 32


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


# Cores a DRAM-sharded linear spreads its activation over, at most. At M=1 the weight read
# is spread over the DRAM banks whatever the core count, so more cores only thin each one's
# K block: on every decode shape, 32 cores ran 1.3 to 2x slower than 8, and 4 matched 8.
DRAM_SHARDED_MAX_CORES = 8


def mlp_compute_config(device):
    """HiFi2 for the bfloat8_b MLP: a DRAM-sharded bf8 linear is compute-bound at HiFi4,
    20.3 against 28.0 us on the predictor's gate, at the same PCC to five places."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


class DramShardedPlan:
    """Configs for a DRAM-sharded decode linear of `in_features` by `out_features`.

    The weight is width-sharded across the DRAM banks, so every bank streams its own slice
    and none sits idle; an interleaved weight reaches well under the DRAM peak at M=1. The
    activation and the output are width-sharded in L1 over one grid of cores, which must
    divide both K and N in tiles.
    """

    def __init__(self, device, in_features, out_features, compute_config, fused_activation=None):
        k_tiles, n_tiles = in_features // TILE_ROWS, out_features // TILE_ROWS
        grid = device.compute_with_storage_grid_size()
        cols, rows = next(
            (cores, 1) if cores <= grid.x else (grid.x, cores // grid.x)
            for cores in range(min(DRAM_SHARDED_MAX_CORES, grid.x * grid.y), 0, -1)
            if k_tiles % cores == 0 and n_tiles % cores == 0 and (cores <= grid.x or cores % grid.x == 0)
        )
        cores = cols * rows
        core_grid = ttnn.CoreGrid(x=cols, y=rows)
        self.in_features, self.out_features = in_features, out_features
        self.compute_config = compute_config
        self.input_memory = self._width_sharded(in_features // cores, core_grid)
        self.output_memory = self._width_sharded(out_features // cores, core_grid)
        self.program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=next(w for w in range(8, 0, -1) if (k_tiles // cores) % w == 0),
            per_core_M=1,
            per_core_N=n_tiles // cores,
            fused_activation=fused_activation,
        )
        banks = device.dram_grid_size().x
        # Every bank holds an equal slice, so the last one's is ragged where N does not divide.
        padded_out = -(-out_features // (TILE_ROWS * banks)) * TILE_ROWS * banks
        self.weight_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                (in_features, padded_out // banks),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    @staticmethod
    def _width_sharded(width, core_grid):
        return ttnn.create_sharded_memory_config(
            shape=(TILE_ROWS, width),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def weight(self, tensor, device, dtype):
        """A [K, N] host weight on device in this plan's DRAM sharding."""
        return ttnn.from_torch(
            tensor.reshape(1, 1, self.in_features, self.out_features).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=self.weight_memory,
        )

    def __call__(self, x, weight):
        """x [.., rows <= 32, K], interleaved or already in `input_memory`; the output stays
        width-sharded in L1."""
        if not x.is_sharded():
            x = ttnn.interleaved_to_sharded(x, self.input_memory)
        return ttnn.linear(
            x,
            weight,
            compute_kernel_config=self.compute_config,
            program_config=self.program,
            memory_config=self.output_memory,
        )


def linear_plans(device, config):
    """A DRAM-sharded plan per decoder-layer linear; `gate_proj` fuses its SiLU."""
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    attention = config["num_attention_heads"] * config["head_dim"]
    qkv = attention + 2 * config["num_key_value_heads"] * config["head_dim"]
    silu = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    attention_config, mlp_config = _compute_config(device), mlp_compute_config(device)
    return {
        "qkv": DramShardedPlan(device, hidden, qkv, attention_config),
        "o_proj": DramShardedPlan(device, attention, hidden, attention_config),
        "gate_proj": DramShardedPlan(device, hidden, intermediate, mlp_config, silu),
        "up_proj": DramShardedPlan(device, hidden, intermediate, mlp_config),
        "down_proj": DramShardedPlan(device, intermediate, hidden, mlp_config),
    }


def decode_mlp(x, layer, plans):
    """The MLP over up to 32 rows, returned interleaved. Gate, up and their product stay
    sharded on one grid, which is down's input grid."""
    gate = plans["gate_proj"](x, layer["gate_proj"])
    up = plans["up_proj"](x, layer["up_proj"])
    product = ttnn.multiply(gate, up, memory_config=plans["down_proj"].input_memory)
    return ttnn.sharded_to_interleaved(plans["down_proj"](product, layer["down_proj"]))


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

    # The norm weights again, on host, for the decoder's width-sharded copies (68 small tensors).
    names = [f"layers.{i}.{n}.weight" for i in range(cfg["num_hidden_layers"]) for n in NORM_NAMES] + ["norm.weight"]
    return {"config": cfg, "layers": layers, "norm": norm("norm"), "norm_weights": {n: state[n] for n in names}}


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
    """Prefill once into an on-device KV cache, then step one position at a time.

    A step advances `batch` rows together, each its own request at its own position over
    its own row of the page table. Prefill fills one row at a time.
    """

    def __init__(self, device, parameters, max_seq=512, kv_cache=None, batch=1, prefill_lengths=()):
        """`kv_cache`, when given, is one `(k, v)` pair per layer of paged caches the caller
        owns; the caller then also supplies each request's blocks through `set_page_table`.

        `prefill_lengths` are the prompt lengths, multiples of 32, a prefill is traced at;
        `traced_prefill` pads a prompt to the shortest that holds it. Only with a caller's
        cache: their warmup writes block 0, which vLLM keeps empty."""
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
        self.scale = self.head_dim**-0.5
        self.layers = len(parameters["layers"])

        self.max_seq = ((max_seq + CACHE_TILE_MULTIPLE - 1) // CACHE_TILE_MULTIPLE) * CACHE_TILE_MULTIPLE
        self._owns_cache = kv_cache is None
        if self._owns_cache and batch > 1:
            raise ValueError("a cache this decoder allocates holds one request; pass kv_cache for a batch")
        if self._owns_cache:
            blocks = -(-self.max_seq // OWNED_BLOCK_SIZE)
            empty = torch.zeros(blocks, self.kv_heads, OWNED_BLOCK_SIZE, self.head_dim)
            kv_cache = [(self._cache_tensor(empty), self._cache_tensor(empty)) for _ in range(self.layers)]
        if len(kv_cache) != self.layers:
            raise ValueError(f"kv_cache has {len(kv_cache)} layers, the talker {self.layers}")
        self.k_cache = [k for k, _ in kv_cache]
        self.v_cache = [v for _, v in kv_cache]
        self.block_size = int(self.k_cache[0].shape[2])
        if self.block_size % 32 or tuple(self.k_cache[0].shape)[1:] != (self.kv_heads, self.block_size, self.head_dim):
            raise ValueError(
                f"cache blocks of shape {tuple(self.k_cache[0].shape)}: need [blocks, {self.kv_heads}, "
                f"a multiple of 32, {self.head_dim}]"
            )
        # Each row's request's blocks, in position order. Device-resident and rewritten in place,
        # so a captured step reads whichever requests' tables are current.
        self.max_blocks = -(-self.max_seq // self.block_size)
        if not self._owns_cache and self.max_blocks > int(self.k_cache[0].shape[0]):
            # A page table may not be wider than the cache it indexes, and no request can
            # hold more positions than the cache has, so the caller's cache bounds both.
            self.max_blocks = int(self.k_cache[0].shape[0])
            self.max_seq = self.max_blocks * self.block_size
        self._tables = torch.zeros(batch, self.max_blocks, dtype=torch.int32)
        self._page_table = ttnn.from_torch(self._tables, dtype=ttnn.int32, device=device)
        if self._owns_cache:
            self.set_page_table(torch.arange(self.max_blocks, dtype=torch.int32))

        # Stable per-step state, allocated once so a captured trace can read it. Allocating
        # under a live trace corrupts it, so everything the step touches exists up front.
        self._pos = ttnn.from_torch(torch.zeros(batch, dtype=torch.int32), device=device)
        self._in = ttnn.from_torch(
            torch.zeros(1, batch, self.hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        # Height-sharded, one batch row per core: the cache update needs its K and V sharded,
        # and the decode-mode rotation needs its input and tables sharded alike. K is rotated
        # straight into the cores the update reads it from. V goes on another row of cores,
        # since the fused K+V update needs the two not to overlap.
        self._k_shard = self._row_shard(0)
        self._v_shard = self._row_shard(1)
        # Each batch row's own position, [1, batch, 1, head_dim], laid out like K.
        self._cos = self._rope_table()
        self._sin = self._rope_table()
        # One zero tensor to clear the caches from. `zeros_like` would allocate a fresh
        # device buffer per cache per utterance, and a buffer allocated while a trace is
        # live is corrupt once that trace runs unless it is released first.
        self._zero_cache = self._cache_tensor(empty) if self._owns_cache else None

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
            state = parameters["norm_weights"]
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

        self.prefill_lengths = tuple(sorted(int(length) for length in prefill_lengths))
        if self.prefill_lengths and self._owns_cache:
            raise ValueError("a traced prefill needs a caller's cache, whose block 0 its warmup may write")
        for length in self.prefill_lengths:
            if length % TILE_ROWS or not 0 < length <= self.max_seq:
                raise ValueError(f"a traced prefill of {length}: need a multiple of {TILE_ROWS} up to {self.max_seq}")
        # One row of blocks, so one trace per length serves every batch row: `paged_fill_cache`
        # takes its row index as a compile-time argument.
        self._prefill_table = ttnn.from_torch(torch.zeros(1, self.max_blocks, dtype=torch.int32), device=device)
        self._prefills = {length: self._prefill_inputs(length) for length in self.prefill_lengths}
        self._prefill_traces = {}

    def _prefill_inputs(self, length):
        """A traced prefill's stable device inputs at one length. Positions are always 0..L-1,
        so the rotary tables and the mask are constants."""
        positions = torch.arange(length, dtype=torch.long).reshape(1, 1, length).expand(3, 1, length).contiguous()
        cos, sin = rotary_tables(self.config, positions)
        to_device = lambda tensor: ttnn.from_torch(
            tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        return {
            "x": to_device(torch.zeros(1, length, self.hidden)),
            "cos": to_device(cos),
            "sin": to_device(sin),
            "mask": to_device(causal_mask(length)),
            # Row 0 one-hot at the prompt's last position: picks the hidden state the next
            # frame needs out of the padded prompt.
            "select": to_device(torch.zeros(1, TILE_ROWS, length)),
            "out": None,
        }

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

    def _row_shard(self, row):
        """One tile of rows per batch row, on cores (0..batch-1, row)."""
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, row), ttnn.CoreCoord(self.batch - 1, row))}),
                (TILE_ROWS, self.head_dim),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _rope_table(self):
        empty = torch.zeros(1, self.batch, 1, self.head_dim)
        return ttnn.from_torch(
            empty, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=self._k_shard
        )

    def _cache_tensor(self, empty):
        return ttnn.from_torch(empty, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    @property
    def owns_cache(self):
        """Whether the cache was allocated here, and so may be cleared here."""
        return self._owns_cache

    def set_page_table(self, blocks, row=0):
        """Point batch row `row`, for prefill and step, at its request's cache blocks, in
        position order. The rest of the row reads block 0, vLLM's null block."""
        blocks = torch.as_tensor(blocks, dtype=torch.int32).reshape(-1)
        if blocks.numel() > self.max_blocks:
            raise ValueError(f"{blocks.numel()} blocks for a table of {self.max_blocks}")
        if not 0 <= row < self.batch:
            raise ValueError(f"row {row} of a batch of {self.batch}")
        self._tables[row] = 0
        self._tables[row, : blocks.numel()] = blocks
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(self._tables, dtype=ttnn.int32), self._page_table)

    def set_page_tables(self, tables):
        """Every row's blocks at once: `tables` [batch, blocks], each row in position order."""
        tables = torch.as_tensor(tables, dtype=torch.int32)
        if tables.shape[0] != self.batch or tables.shape[1] > self.max_blocks:
            raise ValueError(f"page tables of shape {tuple(tables.shape)} for {self.batch} rows of {self.max_blocks}")
        self._tables.zero_()
        self._tables[:, : tables.shape[1]] = tables
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(self._tables, dtype=ttnn.int32), self._page_table)

    # ── shared pieces ───────────────────────────────────────────────────────

    def _rotate(self, x, cos, sin, decode=False):
        """RoPE in one kernel, where the arithmetic spelled out took seven ops: 7.8 us against 26.7.

        Prefill rotates [1, heads, positions, head_dim] by a table row per position. A step
        rotates [1, batch, heads, head_dim], sharded a batch row per core, by its row's position.
        """
        return ttnn.experimental.rotary_embedding_hf(
            x, cos, sin, is_decode_mode=decode, compute_kernel_config=self.compute_config
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

    def prefill(self, embeddings, position_ids=None, row=0):
        """Seed batch row `row`'s cache for positions 0..P-1 in one pass and return the
        hidden states.

        Each layer's K/V weights are read once here rather than once per position, and the
        cache is written through that row of the page table with `paged_fill_cache`.
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
        x = self._prefill_ops(x, cos_tt, sin_tt, mask_tt, length, self._page_table, row)
        return ttnn.rms_norm(x, weight=self.p["norm"], epsilon=self.eps)

    def _prefill_ops(self, x, cos_tt, sin_tt, mask_tt, length, page_table, row):
        """The layers over a whole prompt, filling row `row` of `page_table`'s blocks. Returns
        the last layer's output, before the final norm."""
        for index, layer in enumerate(self.p["layers"]):
            normed = ttnn.rms_norm(x, weight=layer["input_layernorm"], epsilon=self.eps)
            qkv = ttnn.linear(normed, layer["qkv"], compute_kernel_config=self.compute_config)
            query, key, value = self._split_qkv(qkv, length)

            query = self._rotate(ttnn.rms_norm(query, weight=layer["q_norm"], epsilon=self.eps), cos_tt, sin_tt)
            key = self._rotate(ttnn.rms_norm(key, weight=layer["k_norm"], epsilon=self.eps), cos_tt, sin_tt)

            ttnn.experimental.paged_fill_cache(self.k_cache[index], key, page_table, batch_idx=row)
            ttnn.experimental.paged_fill_cache(self.v_cache[index], value, page_table, batch_idx=row)

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
        return x

    def _traced_prefill_ops(self, length):
        """A prefill at `length` from its stable inputs, ending at the selected row, normed."""
        inputs = self._prefills[length]
        x = self._prefill_ops(inputs["x"], inputs["cos"], inputs["sin"], inputs["mask"], length, self._prefill_table, 0)
        last = ttnn.matmul(inputs["select"], x, compute_kernel_config=self.compute_config)
        return ttnn.rms_norm(last, weight=self.p["norm"], epsilon=self.eps)

    def traced_prefill(self, embeddings, row):
        """Prefill a prompt [1, P, hidden] into batch row `row`, whose page table is set, by
        replaying the trace of the shortest `prefill_lengths` entry that holds it. Returns
        the hidden state at position P-1, [hidden], on host.

        The prompt is padded with zeros. Attention is causal, so the padding changes no
        position before P; its cache writes land past P, in the row's last block, which the
        row's own steps overwrite before reading, or in block 0.
        """
        prompt = embeddings.shape[1]
        length = next((length for length in self.prefill_lengths if length >= prompt), None)
        if length is None:
            raise ValueError(f"prompt of {prompt}: the longest traced prefill is {self.prefill_lengths[-1:]}")
        if not self._prefill_traces:
            raise RuntimeError("the prefill traces are not captured; call capture first")
        inputs = self._prefills[length]
        padded = torch.zeros(1, length, self.hidden)
        padded[:, :prompt] = embeddings
        select = torch.zeros(1, TILE_ROWS, length)
        select[0, 0, prompt - 1] = 1.0
        to_host = lambda tensor: ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(to_host(padded), inputs["x"])
        ttnn.copy_host_to_device_tensor(to_host(select), inputs["select"])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(self._tables[row : row + 1].clone()), self._prefill_table)
        ttnn.execute_trace(self.device, self._prefill_traces[length], cq_id=0, blocking=False)
        return ttnn.to_torch(inputs["out"]).float()[0, 0]

    # ── one cached step ─────────────────────────────────────────────────────

    def _step_ops(self, x):
        for index, layer in enumerate(self.p["layers"]):
            normed = self._norm(x, layer, "input_layernorm")
            qkv = ttnn.linear(
                normed, layer["qkv"], compute_kernel_config=self.compute_config, program_config=self._matmul["qkv"]
            )
            # Fused per-head split. Emits [1, batch, heads, head_dim] height-sharded in L1,
            # the layout flash decode and the cache update want, so no permutes appear in the
            # step at all. It replaces 3 slices, 3 reshapes and 3 permutes.
            query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
                ttnn.reshape(qkv, (1, 1, self.batch, qkv.shape[-1])),
                num_heads=self.heads,
                num_kv_heads=self.kv_heads,
            )
            # QK-norm cannot read height-sharded input, so Q and K come back interleaved for
            # the norm, then go back to a row per core for the rotation.
            query = ttnn.rms_norm(ttnn.sharded_to_interleaved(query), weight=layer["q_norm"], epsilon=self.eps)
            key = ttnn.rms_norm(ttnn.sharded_to_interleaved(key), weight=layer["k_norm"], epsilon=self.eps)
            query = ttnn.sharded_to_interleaved(
                self._rotate(ttnn.to_memory_config(query, self._k_shard), self._cos, self._sin, decode=True)
            )
            key = self._rotate(ttnn.to_memory_config(key, self._k_shard), self._cos, self._sin, decode=True)

            # The cache therefore stores post-norm, post-rotation K, which is what attention
            # over the prefix expects to find there. One kernel writes both, which needs them
            # on non-overlapping cores.
            ttnn.experimental.paged_fused_update_cache(
                self.k_cache[index],
                key,
                self.v_cache[index],
                ttnn.to_memory_config(value, self._v_shard),
                update_idxs_tensor=self._pos,
                page_table=self._page_table,
            )

            attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                query,
                self.k_cache[index],
                self.v_cache[index],
                page_table_tensor=self._page_table,
                cur_pos_tensor=self._pos,
                scale=self.scale,
                compute_kernel_config=self.compute_config,
            )
            attended = ttnn.reshape(attended, (1, self.batch, self.heads * self.head_dim))
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
        """Where each row's step writes and how far its attention reads: one position, or one
        per batch row."""
        positions = torch.as_tensor(position, dtype=torch.int32).reshape(-1).expand(self.batch).contiguous()
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(positions), self._pos)
        cos, sin = rotary_tables(self.config, positions.long().reshape(1, self.batch, 1).expand(3, -1, -1))
        for source, target in ((cos, self._cos), (sin, self._sin)):
            rows = source.reshape(1, self.batch, 1, self.head_dim)
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
        for trace_id in self._prefill_traces.values():
            ttnn.release_trace(self.device, trace_id)
        self._prefill_traces = {}

    def reset(self):
        """Clear the caches without allocating. Only for a cache this decoder owns."""
        if not self._owns_cache:
            raise RuntimeError("the KV cache belongs to the caller; it clears its own blocks")
        for cache in self.k_cache + self.v_cache:
            ttnn.copy(self._zero_cache, cache)

    def step(self, embedding, position):
        """One position per row: embeddings [1, batch, hidden] -> hidden states [1, batch, hidden].

        `position` is one position, or one per row.
        """
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
        alone so this can follow `prefill`; the traced prefills warm through block 0.
        """
        self.set_position(self.max_seq - 1)
        self._step_ops(self._in)
        # Through block 0 only, so no request's blocks are written.
        if self.prefill_lengths:
            empty = torch.zeros(1, self.max_blocks, dtype=torch.int32)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(empty), self._prefill_table)
        for length in self.prefill_lengths:
            self._traced_prefill_ops(length)
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
        for length in self.prefill_lengths:
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._prefills[length]["out"] = self._traced_prefill_ops(length)
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            self._prefill_traces[length] = trace_id
        ttnn.synchronize_device(self.device)
