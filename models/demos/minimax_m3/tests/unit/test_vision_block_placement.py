# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-only checks of where the vision encoder's blocks keep their activations: the rule (block_placement in
tt/vision/common.py), the L1 budget of the forward that follows it, that forward's calls, and the check of the device's
L1 before each forward (l1_free_for_activations). No device: the program configs are built with the real ttnn classes,
and M3VLVisionModel.forward_device runs at the checkpoint's widths on stand-ins for its device tensors and ops.

- The rule, at every patch count from 4 to 4096 (multiples of 4) on the grids of L1_GRIDS: "l1" and "l1_interleaved"
  keep the activations in L1 exactly where the image has at most L1_MAX_ROWS rows and linear_program_config gives each
  of the block's four linears a config, and those configs pass the matmul's checks. "l1" keeps the MLP's hidden
  activations in block shards exactly where fc1's and fc2's configs for that layout pass the matmul's checks, with
  fc1's output shards as fc2's input, in the layouts of SHARDED_MLP_CLASSES. The images of the device test of L1
  against DRAM in CI are among them, and with M3_VISION_ALL_GRIDS=1 its images take each of those layouts. "dram",
  another grid and every larger image get DRAM_PLACEMENT.
- The L1 budget, at every patch count the rule keeps in L1, with "l1" and "l1_interleaved": the forward of a 4-block
  tower is recorded, each op's output allocated before the op runs and freed where the forward frees it or drops its
  last reference, and replayed through a model of one L1 bank of PROOF_L1_BANK_BYTES (tensors from the top down, first
  fit). Every op's circular buffers end at least PROOF_L1_FOREIGN_BYTES below the lowest tensor, the interleaved
  LayerNorms keep the plan they have in an empty L1 with that much more taken, an op whose config ttnn picks runs with
  no tensor in L1, every matmul's config passes the matmul's checks for the operands the forward gives it, block 3
  starts from block 1's L1 state, and L1 is empty when the tower returns. At 2304 patches on 12x10 ten of the thirteen
  block ops have the margins a P150 measured with "l1_interleaved", and fc1 and fc2 the buffers it measured with the
  MLP's hidden activations in shards. With L1_MAX_ROWS lifted, the walk finds ops on 12x10 whose buffers would overlap
  a tensor above it.
- The calls: DRAM_PLACEMENT is pinned field by field. Each op gets the keywords of CALL_KEYWORDS and, where its field
  of the placement is not None, its memory_config, and none where it is None: so DRAM_PLACEMENT gives the LayerNorms,
  GELU and SDPA no memory_config. Each matmul gets linear_program_config's config for its operands. With the L1
  placements out_proj reads its input from DRAM, fc2 from DRAM or from fc1's shards, and the tower's output is in DRAM.
- The check before each forward: L1 memory views with another bank count, a smaller bank or a tensor lower in the bank
  make the forward use DRAM_PLACEMENT, with the reason logged once.

The 1-card leg selects these tests by name (test_vision_), as they have no mesh id.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_block_placement.py
"""
from __future__ import annotations

import collections
import itertools
import math
import weakref
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.tt.vision import common
from models.demos.minimax_m3.tt.vision.attention import M3VLAttention
from models.demos.minimax_m3.tt.vision.block import M3VLBlock
from models.demos.minimax_m3.tt.vision.common import (
    DRAM_PLACEMENT,
    L1_INTERLEAVED_PLACEMENT,
    L1_MAX_ROWS,
    L1_SHARDED_MLP_PLACEMENT,
    P150_L1_BANK_BYTES,
    PROOF_L1_BANK_BYTES,
    PROOF_L1_FOREIGN_BYTES,
    BlockPlacement,
    block_linears,
    block_placement,
    l1_free_for_activations,
    linear_program_config,
)
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm
from models.demos.minimax_m3.tt.vision.mlp import M3VLMLP
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel
from models.demos.minimax_m3.tt.vision.patch_embed import M3VLPatchEmbed

from .vision_helpers import (
    L1_GRID_IDS,
    L1_VS_DRAM_ALL_GRIDS,
    L1_VS_DRAM_CI_GRIDS,
    VISION_GRIDS,
    assert_matmul_config_valid,
    matmul_cb_bytes,
    matmul_config_problems,
    matmul_sharded_output,
)

CONFIG = MiniMaxM3VisionConfig()
PADDED_HEAD_DIM = nearest_32(CONFIG.head_dim)  # 96, as M3VLVisionModel.from_state_dict pads the heads
# Block 0, blocks 1 and 2, whose inputs lie at two L1 addresses in turn, and block 3, which starts from block 1's L1
# state: every later block of a deeper tower repeats block 1 or block 2, and the last, which writes its output to DRAM,
# takes no more room than the block it repeats.
NUM_LAYERS = 4
PATCHES = range(4, 4096 + 1, 4)
# The patch counts the rule keeps in L1 on each grid; at the others up to 2304 a block linear keeps ttnn's own config.
L1_PATCHES = {
    (11, 10): ((228, 320), (484, 640), (708, 2304)),
    (12, 10): ((228, 320), (452, 640), (676, 2304)),
    (13, 10): ((196, 320), (420, 2304)),
}
# Of those, the patch counts where "l1" keeps the MLP's hidden activations in block shards, by the layout of fc1's
# output: ("whole", columns) where every core of the split holds a whole block of rows, ("partial", columns) where the
# last column's holds fewer rows, with the split over that many of the grid's columns. A P150 (12x10) has run fc1's
# output in this layout at one size, 2304 patches (whole, 12 columns), with fc1 reading and fc2 writing block shards.
SHARDED_MLP_CLASSES = {
    (11, 10): {("whole", 11): 24, ("partial", 11): 112},
    (12, 10): {("whole", 12): 24, ("whole", 11): 24, ("partial", 12): 96, ("partial", 11): 96},
    (13, 10): {
        ("whole", 13): 16,
        ("whole", 12): 16,
        ("whole", 11): 8,
        ("partial", 13): 56,
        ("partial", 12): 32,
        ("partial", 11): 24,
    },
}

# The ops of a block in order, by role, and those of the tower before its blocks.
BLOCK_OPS = dict(
    ln1="layer_norm",
    qkv="linear",
    heads="nlp_create_qkv_heads",
    rope_q="rotary_embedding_llama",
    rope_k="rotary_embedding_llama",
    sdpa="scaled_dot_product_attention",
    out_proj="linear",
    add1="add",
    ln2="layer_norm",
    fc1="linear",
    gelu="gelu",
    fc2="linear",
    add2="add",
)
PRE_BLOCK_OPS = dict(
    patch_embed="linear", pre_layrnorm="layer_norm", cos_to_l1="to_memory_config", sin_to_l1="to_memory_config"
)
OPS = {**PRE_BLOCK_OPS, **BLOCK_OPS}
# (role, block) of each op of a tower forward, block None before the blocks.
TOWER_ROLES = [(role, None) for role in PRE_BLOCK_OPS] + [
    (role, block) for block in range(NUM_LAYERS) for role in BLOCK_OPS
]
# Each op's keywords besides memory_config, and how many of its first arguments are activations (the others are
# weights, the RoPE tables and memory configs).
CALL_KEYWORDS = {
    "linear": {"bias", "compute_kernel_config", "program_config"},
    "layer_norm": {"weight", "bias", "epsilon", "compute_kernel_config"},
    "nlp_create_qkv_heads": {"num_heads", "num_kv_heads", "transpose_k_heads"},
    "rotary_embedding_llama": {"is_decode_mode", "compute_kernel_config"},
    "scaled_dot_product_attention": {
        "is_causal",
        "scale",
        "program_config",
        "compute_kernel_config",
        "output_concat_heads",
    },
    "add": set(),
    "gelu": {"fast_and_approximate_mode"},
    "to_memory_config": set(),
}
ACTIVATION_ARGS = dict(
    linear=1,
    layer_norm=1,
    nlp_create_qkv_heads=1,
    rotary_embedding_llama=1,
    scaled_dot_product_attention=3,
    add=2,
    gelu=1,
)

TILE_BYTES, FP32_TILE_BYTES, BFP4_TILE_BYTES = 2048, 4096, 576
# The interleaved LayerNorm at width 1280, bf16 with fp32 statistics, weight and bias (layernorm_op_multi_core.cpp):
# input, weight and bias 40 tiles each, output 8, scaler and epsilon 2 each in bf16; centred and squared values 40
# tiles each, two intermediates of 8, mean and variance 2 each in fp32. It keeps this plan while these take less than
# 95% of the L1 below the lowest tensor, in whole bytes (program_cache_l1.hpp, layernorm_op_multi_core.cpp).
LAYERNORM_CB_BYTES = TILE_BYTES * (3 * 40 + 8 + 2 * 2) + FP32_TILE_BYTES * (2 * 40 + 2 * 8 + 2 * 2)
# The circular buffers per core of the ops other than the matmuls and SDPA, the same at every size of L1_PATCHES. All
# but the copy's equal what a P150 (12x10) measured at 2304 patches.
CB_BYTES = {
    "layer_norm": LAYERNORM_CB_BYTES,
    # 4 tiles for heads of 3 tiles (nlp_create_qkv_heads_program_factory.cpp).
    "nlp_create_qkv_heads": 4 * TILE_BYTES,
    # Input, cos, sin and output 6 tiles each, three intermediates and a scratch of 3 and the transformation matrix's
    # tile, for heads of 3 tiles and one sequence tile per core (rotary_embedding_llama_multi_core_program_factory.cpp).
    "rotary_embedding_llama": TILE_BYTES * (4 * 6 + 4 * 3 + 1),
    # 2 tiles of each input and of the output (binary_ng_program_factory.cpp).
    "add": 6 * TILE_BYTES,
    # 2 tiles in and 2 out for an input in L1 interleaved (unary_program_factory.cpp); on block shards both buffers
    # are the shards.
    "gelu": 4 * TILE_BYTES,
    # 2 pages (copy_default_tilized_program_factory.cpp).
    "to_memory_config": 2 * TILE_BYTES,
}
# At 2304 patches on 12x10, in a bank of P150_L1_BANK_BYTES (a device opened with no L1_SMALL region), the least margin
# of each block op over the blocks. A P150 measured these with every tensor of the block in L1 interleaved; SDPA's and
# GELU's outputs, in DRAM here, leave them the same.
MEASURED_MARGINS_2304 = dict(
    ln1=625280,
    qkv=199296,
    heads=940672,
    rope_q=873088,
    rope_k=873088,
    sdpa=516224,
    add1=1292928,
    ln2=625280,
    fc1=125568,
    add2=1243776,
)
# The other three, the walk's own: with those outputs in L1 the measured ones were less.
WALKED_MARGINS_2304 = dict(out_proj=821888, gelu=1100416, fc2=772736)
# The circular buffers per core a P150 measured at 2304 patches on 12x10 for fc1 writing block shards and fc2 reading
# them (and writing block shards there): one output block less than with an interleaved output, and the K-block
# buffers of an interleaved input.
MEASURED_SHARDED_CB_BYTES_2304 = dict(fc1=786432, fc2=434176)
# With "l1", as with "l1_interleaved" but for the MLP. There a block shard of fc1's output and the interleaved tensor
# take the same bytes per bank, 96 tiles, and so do the residual stream's shards and tensors, 24 tiles, so the tensors
# lie where they lay on that P150: fc1's margin is the one it measured with the residual stream in block shards too,
# fc2's the one it measured with every tensor in L1 interleaved. GELU's is the walk's own, with no buffers of its own.
SHARDED_MLP_MARGINS_2304 = dict(fc1=322176, gelu=912000, fc2=428672)
# The least margin of any op at any size of L1_PATCHES, in a bank of PROOF_L1_BANK_BYTES: (bytes, op).
LEAST_MARGIN = {
    ("l1", (11, 10)): (128512, "qkv"),
    ("l1", (12, 10)): (198144, "qkv"),
    ("l1", (13, 10)): (142848, "fc1"),
    ("l1_interleaved", (11, 10)): (124416, "fc1"),
    ("l1_interleaved", (12, 10)): (124416, "fc1"),
    ("l1_interleaved", (13, 10)): (142848, "fc1"),
}
# With L1_MAX_ROWS lifted, on 12x10 with activations "l1": the patch counts above 2304 where the rule would keep the
# activations in L1, those where some op's circular buffers would overlap a tensor, and the least margin, (bytes, op,
# patches).
UNCAPPED_12X10 = dict(l1_patches=((2308, 3200),), overlap_patches=((3076, 3200),), least=(-23040, "fc1", 3172))


def _ranges(values, step=4):
    """Consecutive runs of values, as (first, last) pairs."""
    runs = []
    for v in values:
        if runs and v == runs[-1][1] + step:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return tuple(tuple(r) for r in runs)


def _sdpa_cb_bytes(config, q_shape, cores):
    """SDPA's circular buffers per core for bf16 q, k and v with fp32 accumulation and no mask
    (sdpa_program_factory.cpp): the q chunk, double-buffered when a core runs more than one; the k and v chunks,
    double-buffered; a padding mask, double-buffered in bfp4, when the chunks do not divide the sequence; two scale
    tiles; the scores and two sums in fp32; two output accumulators, two maxima, their difference and the output."""
    heads, seq_len, head_tiles = q_shape[1], q_shape[2], q_shape[3] // ttnn.TILE_SIZE
    q, k = config.q_chunk_size // ttnn.TILE_SIZE, config.k_chunk_size // ttnn.TILE_SIZE
    q_buffers = 2 if math.ceil(heads * math.ceil(seq_len / config.q_chunk_size) / cores) > 1 else 1
    bf16_tiles = q * head_tiles * q_buffers + 4 * k * head_tiles + 2 + 2 * q * head_tiles + 3 * q + q * head_tiles
    padded = seq_len % config.q_chunk_size or seq_len % config.k_chunk_size
    mask = 2 * q * k * BFP4_TILE_BYTES if padded else 0
    return TILE_BYTES * bf16_tiles + FP32_TILE_BYTES * (q * k + 2 * q) + mask


class _Device:
    """Stands in for the mesh device: the code asks it for its compute grid, and ttnn.get_memory_view for its L1."""

    def __init__(self, grid):
        self.grid = grid

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(*self.grid)


def _memory_view(banks, bank_bytes, tensors=()):
    """An L1 memory view as ttnn.get_memory_view gives it: banks of bank_bytes, each with a tensor at each (offset from
    the bottom of the bank, bytes) of tensors, and a block table that lists those tensors and the free blocks between
    them."""
    blocks, end = [], 0
    for offset, size in sorted(tensors) + [(bank_bytes, 0)]:
        if offset > end:
            blocks.append((end, offset - end, "no"))
        if size:
            blocks.append((offset, size, "yes"))
        end = offset + size
    table = [
        dict(blockID=str(i), address=str(address), size=str(size), prevID=str(i - 1), nextID=str(i + 1), allocated=used)
        for i, (address, size, used) in enumerate(blocks)
    ]
    allocated = sum(size for _, size, used in blocks if used == "yes")
    return SimpleNamespace(
        num_banks=banks,
        total_bytes_per_bank=bank_bytes,
        total_bytes_allocated_per_bank=allocated,
        total_bytes_free_per_bank=bank_bytes - allocated,
        largest_contiguous_bytes_free_per_bank=max((size for _, size, used in blocks if used == "no"), default=0),
        block_table=table,
    )


def _empty_l1(monkeypatch, grid):
    """ttnn.get_memory_view gives an empty L1, one bank of PROOF_L1_BANK_BYTES per core of grid."""
    view = _memory_view(math.prod(grid), PROOF_L1_BANK_BYTES)
    monkeypatch.setattr(ttnn, "get_memory_view", lambda device, buffer_type: view)


def _layernorm_keeps_plan(margin):
    """Whether the interleaved LayerNorm keeps the plan LAYERNORM_CB_BYTES has in an empty L1, with its buffers margin
    bytes below the lowest tensor and PROOF_L1_FOREIGN_BYTES more taken at the top of the bank."""
    span = margin + LAYERNORM_CB_BYTES - PROOF_L1_FOREIGN_BYTES
    return LAYERNORM_CB_BYTES < span * 95 // 100


class _Tensor:
    """A device tensor stand-in: its logical and padded shapes, memory config and device, and an id."""

    def __init__(self, tensor_id, shape, padded_shape, memory_config, device):
        self.id, self.shape, self.padded_shape = tensor_id, tuple(shape), tuple(padded_shape)
        self._memory_config, self._device = memory_config, device

    def memory_config(self):
        return self._memory_config

    def device(self):
        return self._device


def _in_l1(memory_config):
    return memory_config.buffer_type == ttnn.BufferType.L1


class _Recorder:
    """Stands in for the ttnn ops of the encoder's forward and records, in order, each call, with its tensors by id
    and place, and each L1 tensor's allocation, before the call that makes it, and its release, at ttnn.deallocate or
    when its last reference goes. An op with no memory_config writes where ttnn's default puts its output."""

    def __init__(self, monkeypatch):
        self.events = []  # ("alloc", id, bytes per bank) | ("free", id) | ("call", op, args, kwargs)
        self._ids = itertools.count()
        self._live = set()
        monkeypatch.setattr(ttnn, "from_torch", self._from_torch)
        monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh_device: None)
        monkeypatch.setattr(ttnn, "deallocate", lambda t: self._release(t.id))
        monkeypatch.setattr(ttnn, "to_memory_config", self._to_memory_config)
        monkeypatch.setattr(ttnn, "linear", self._linear)
        monkeypatch.setattr(ttnn, "layer_norm", self._layer_norm)
        monkeypatch.setattr(ttnn, "add", self._add)
        monkeypatch.setattr(ttnn, "gelu", self._gelu)
        monkeypatch.setattr(ttnn.experimental, "nlp_create_qkv_heads", self._heads)
        monkeypatch.setattr(ttnn.experimental, "rotary_embedding_llama", self._rope)
        monkeypatch.setattr(ttnn.transformer, "scaled_dot_product_attention", self._sdpa)

    def _tensor(self, shape, memory_config, device, padded_shape=None):
        if padded_shape is None:
            padded_shape = (*shape[:-2], *(nearest_32(d) for d in shape[-2:]))
        t = _Tensor(next(self._ids), shape, padded_shape, memory_config, device)
        if _in_l1(memory_config):
            if memory_config.is_sharded():
                # One shard's range in every bank, the banks of cores without a shard included (allocator.cpp).
                height, width = memory_config.shard_spec.shape
                size = height * width // (ttnn.TILE_SIZE * ttnn.TILE_SIZE) * TILE_BYTES
            else:
                # An interleaved tensor takes the same range in every bank: its tiles spread over the banks.
                grid = device.compute_with_storage_grid_size()
                tiles = math.prod(padded_shape) // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)
                size = math.ceil(tiles / (grid.x * grid.y)) * TILE_BYTES
            self.events.append(("alloc", t.id, size))
            self._live.add(t.id)
            weakref.finalize(t, self._release, t.id)
        return t

    def _dram(self, shape, device):
        return self._tensor(shape, ttnn.DRAM_MEMORY_CONFIG, device)

    def _from_torch(self, t, **kwargs):
        # Default: DRAM (ttnn.from_torch).
        return self._tensor(t.shape, kwargs.get("memory_config", ttnn.DRAM_MEMORY_CONFIG), kwargs["device"])

    def _release(self, tensor_id):
        if tensor_id in self._live:
            self._live.remove(tensor_id)
            self.events.append(("free", tensor_id))

    def _call(self, op, args, kwargs, outputs):
        def ref(v):
            if not isinstance(v, _Tensor):
                return v
            return SimpleNamespace(
                id=v.id,
                shape=v.shape,
                padded_shape=v.padded_shape,
                l1=_in_l1(v.memory_config()),
                memory_config=v.memory_config(),
            )

        self.events.append(("call", op, tuple(map(ref, args)), {k: ref(v) for k, v in kwargs.items()}))
        return outputs

    def _linear(self, x, w, **kwargs):
        shape, padded_shape = (*x.shape[:-1], w.shape[-1]), (*x.padded_shape[:-1], w.padded_shape[-1])
        memory_config = kwargs["memory_config"]
        if memory_config.is_sharded():
            # The op lays the shards out by its program config.
            m_tiles, n_tiles = (padded_shape[-2] // ttnn.TILE_SIZE, padded_shape[-1] // ttnn.TILE_SIZE)
            memory_config = matmul_sharded_output(kwargs["program_config"], m_tiles, n_tiles)
        out = self._tensor(shape, memory_config, x.device(), padded_shape)
        return self._call("linear", (x, w), kwargs, out)

    def _layer_norm(self, x, **kwargs):
        # Default: the input's memory config (layernorm.cpp).
        out = self._tensor(x.shape, kwargs.get("memory_config", x.memory_config()), x.device(), x.padded_shape)
        return self._call("layer_norm", (x,), kwargs, out)

    def _heads(self, x, **kwargs):
        heads = kwargs["num_heads"]
        assert kwargs["num_kv_heads"] == heads
        shape = (1, heads, x.shape[-2], x.shape[-1] // (3 * heads))
        outs = tuple(self._tensor(shape, kwargs["memory_config"], x.device()) for _ in "qkv")
        return self._call("nlp_create_qkv_heads", (x,), kwargs, outs)

    def _rope(self, x, cos, sin, trans_mat, **kwargs):
        # Default: the input's memory config (rotary_embedding_llama_device_operation.cpp).
        out = self._tensor(x.shape, kwargs.get("memory_config", x.memory_config()), x.device(), x.padded_shape)
        return self._call("rotary_embedding_llama", (x, cos, sin, trans_mat), kwargs, out)

    def _sdpa(self, q, k, v, **kwargs):
        assert kwargs["output_concat_heads"]
        _, heads, rows, head_dim = q.shape
        # Default: DRAM (sdpa.cpp).
        out = self._tensor(
            (1, 1, rows, heads * head_dim), kwargs.get("memory_config", ttnn.DRAM_MEMORY_CONFIG), q.device()
        )
        return self._call("scaled_dot_product_attention", (q, k, v), kwargs, out)

    def _add(self, a, b, **kwargs):
        out = self._tensor(a.shape, kwargs["memory_config"], a.device(), a.padded_shape)
        return self._call("add", (a, b), kwargs, out)

    def _gelu(self, x, **kwargs):
        # Default: the input's memory config (unary.cpp).
        out = self._tensor(x.shape, kwargs.get("memory_config", x.memory_config()), x.device(), x.padded_shape)
        return self._call("gelu", (x,), kwargs, out)

    def _to_memory_config(self, x, memory_config):
        out = self._tensor(x.shape, memory_config, x.device(), x.padded_shape)
        return self._call("to_memory_config", (x, memory_config), {}, out)

    def forward(self, model, patches):
        """The calls, allocations and frees of model.forward_device on an image of 2 x patches / 2 patches, uploaded
        before. Checks that the calls are those of TOWER_ROLES and that the tower's output is in DRAM."""
        x_tt = self._dram((1, 1, patches, CONFIG.patch_dim), model.device)
        self.events.clear()
        out = model.forward_device(x_tt, torch.tensor([[1, 2, patches // 2]]))
        assert not _in_l1(out.memory_config()), f"the tower's output is in L1 at {patches} patches"
        events = list(self.events)
        ops = tuple(event[1] for event in events if event[0] == "call")
        assert ops == tuple(OPS[role] for role, _ in TOWER_ROLES), ops
        return events


def _model(recorder, grid, mlp_width=CONFIG.intermediate_size, hidden=CONFIG.hidden_size):
    """M3VLVisionModel at the checkpoint's widths, or with another MLP or hidden width, with NUM_LAYERS blocks and no
    projector, on a stand-in device of the grid, with stand-ins for its weights from recorder's ttnn.from_torch."""
    device = _Device(grid)
    heads = CONFIG.num_attention_heads
    qkv_width = 3 * heads * PADDED_HEAD_DIM

    def weights(*shape):
        return torch.empty(()).expand(shape)  # the shape, no memory

    def norm():
        return M3VLLayerNorm(device, hidden, weights(hidden), weights(hidden))

    def block():
        attention = M3VLAttention(
            device,
            heads,
            CONFIG.head_dim,
            wqkv_weight=weights(hidden, qkv_width),
            wo_weight=weights(heads * PADDED_HEAD_DIM, hidden),
            wqkv_bias=weights(qkv_width),
            wo_bias=weights(hidden),
        )
        mlp = M3VLMLP(
            device, weights(mlp_width, hidden), weights(mlp_width), weights(hidden, mlp_width), weights(hidden)
        )
        return M3VLBlock(norm(), attention, norm(), mlp)

    patch_embed = M3VLPatchEmbed(device, weights(hidden, CONFIG.patch_dim))
    blocks = [block() for _ in range(NUM_LAYERS)]
    return M3VLVisionModel(device, CONFIG, PADDED_HEAD_DIM, patch_embed, norm(), blocks)


def _calls(events):
    """(role, block, op, args, kwargs) of each recorded call, in order."""
    calls = [event[1:] for event in events if event[0] == "call"]
    return [(role, block, *call) for (role, block), call in zip(TOWER_ROLES, calls)]


def _walk(events, bank_bytes):
    """Replays a recorded forward through a model of one L1 bank: tensors from the top down, each at the top of the
    highest free range that holds it. Returns (role, block, margin) per call, the margin being the lowest tensor's
    offset less the op's circular buffers once its outputs exist (None where ttnn picks the config, which must then
    see no tensor in L1), the tensors' ranges at each call by (role, block), and whether L1 ends empty."""
    free, live, margins, layouts = [(0, bank_bytes)], {}, [], {}
    roles = iter(TOWER_ROLES)
    for event in events:
        if event[0] == "alloc":
            _, tensor_id, size = event
            i = max(i for i, (start, end) in enumerate(free) if end - start >= size)
            start, end = free[i]
            free[i : i + 1] = [(start, end - size)] if end - size > start else []
            live[tensor_id] = (end - size, end)
        elif event[0] == "free":
            free = sorted(free + [live.pop(event[1])])
            merged = [free[0]]
            for start, end in free[1:]:
                if merged[-1][1] == start:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((start, end))
            free = merged
        else:
            cb_bytes = _cb_bytes(*event[1:])
            lowest = min((start for start, _ in live.values()), default=bank_bytes)
            if cb_bytes is None:
                assert not live, f"{event[1]} plans by the free L1, with {len(live)} tensors in L1"
            role = next(roles)
            margins.append((*role, None if cb_bytes is None else lowest - cb_bytes))
            layouts[role] = sorted(live.values())
    return margins, layouts, not live


def _cb_bytes(op, args, kwargs):
    """The circular buffers per core of a recorded call, None for a matmul whose config ttnn picks."""
    if op == "linear":
        config = kwargs["program_config"]
        if config is None:
            return None
        fp32_dest_acc = kwargs["compute_kernel_config"].fp32_dest_acc_en
        k_tiles, bias = args[1].padded_shape[-2] // ttnn.TILE_SIZE, kwargs["bias"] is not None
        return matmul_cb_bytes(config, k_tiles, fp32_dest_acc, bias, out_sharded=kwargs["memory_config"].is_sharded())
    if op == "scaled_dot_product_attention":
        grid = kwargs["program_config"].compute_with_storage_grid_size
        return _sdpa_cb_bytes(kwargs["program_config"], args[0].shape, grid.x * grid.y)
    if op == "gelu":
        if args[0].memory_config.is_sharded():
            return 0  # its input and output buffers are the shards (unary_program_factory.cpp)
        assert args[0].l1, "CB_BYTES holds GELU's buffers for an input in L1"
    return CB_BYTES[op]


def _assert_linears_valid(events, grid):
    """Each recorded matmul with a program config passes the matmul's checks for the operands the forward gave it."""
    for event in events:
        if event[0] != "call" or event[1] != "linear" or event[3]["program_config"] is None:
            continue
        (x, w), kwargs = event[2], event[3]
        assert_matmul_config_valid(
            kwargs["program_config"],
            x.padded_shape[-2],
            *w.padded_shape[-2:],
            grid,
            bool(kwargs["compute_kernel_config"].fp32_dest_acc_en),
            in0_shards=x.memory_config if x.memory_config.is_sharded() else None,
            out_sharded=kwargs["memory_config"].is_sharded(),
        )


def _mlp_shards(device, mlp, rows, grid):
    """The layout of fc1's output in block shards, ("whole" or "partial", columns of its split), where fc1's config for
    that output and fc2's for it as its input pass the matmul's checks (GELU keeps its input's shards), else None."""
    fp32_dest_acc = bool(mlp.compute_kernel_config.fp32_dest_acc_en)
    (k1, n1), (k2, n2) = mlp.fc1_weight.padded_shape[-2:], mlp.fc2_weight.padded_shape[-2:]
    fc1 = linear_program_config(device, rows, k1, n1, mlp.compute_kernel_config, out_sharded=True)
    fc2 = linear_program_config(device, rows, k2, n2, mlp.compute_kernel_config, in0_sharded=True)
    m_tiles = rows // ttnn.TILE_SIZE
    if matmul_config_problems(fc1, rows, k1, n1, grid, fp32_dest_acc, out_sharded=True):
        return None
    shards = matmul_sharded_output(fc1, m_tiles, n1 // ttnn.TILE_SIZE)
    if matmul_config_problems(fc2, rows, k2, n2, grid, fp32_dest_acc, in0_shards=shards):
        return None
    return ("whole" if m_tiles % fc1.per_core_M == 0 else "partial", math.ceil(m_tiles / fc1.per_core_M))


@pytest.mark.parametrize("grid", list(L1_GRID_IDS.values()), ids=list(L1_GRID_IDS))
def test_vision_block_placement_rule(monkeypatch, grid, expect_error):
    """block_placement against linear_program_config at every patch count up to 4096: "l1" and "l1_interleaved" keep
    the activations in L1 exactly at the patch counts of L1_PATCHES, where the image has at most L1_MAX_ROWS rows and
    the four block linears get a config, padded rows or not; those configs pass the matmul's checks. There "l1" gives
    L1_SHARDED_MLP_PLACEMENT exactly where fc1's config with its output in block shards and fc2's with those shards as
    its input pass them, in the layouts of SHARDED_MLP_CLASSES, and L1_INTERLEAVED_PLACEMENT elsewhere, which
    "l1_interleaved" gives at every patch count of L1_PATCHES. The images of L1_VS_DRAM_CI_GRIDS are among them, and
    h36w36 keeps the MLP's hidden activations in block shards; the images the device test runs with
    M3_VISION_ALL_GRIDS=1 take every layout of SHARDED_MLP_CLASSES. "dram" and another grid give DRAM_PLACEMENT, the
    latter where the four block linears get a config, and another value raises."""
    model = _model(_Recorder(monkeypatch), grid)
    block = model.blocks[0]
    linears = block_linears(block)
    in_l1, sharded, classes, layout_of = [], [], collections.Counter(), {}
    for patches in PATCHES:
        rows = nearest_32(patches)
        configs = [
            (linear_program_config(model.device, rows, *w.padded_shape[-2:], compute), w.padded_shape[-2:])
            for w, compute in linears
        ]
        fits = rows <= L1_MAX_ROWS and all(config is not None for config, _ in configs)
        shards = _mlp_shards(model.device, block.mlp, rows, grid) if fits else None
        interleaved = L1_INTERLEAVED_PLACEMENT if fits else DRAM_PLACEMENT
        want = L1_SHARDED_MLP_PLACEMENT if shards else interleaved
        for size in (rows, patches):
            assert block_placement(block, size, "l1") is want, patches
            assert block_placement(block, size, "l1_interleaved") is interleaved, patches
        assert block_placement(block, rows, "dram") is DRAM_PLACEMENT, patches
        if fits:
            in_l1.append(patches)
            for config, (in_features, out_features) in configs:
                assert_matmul_config_valid(config, rows, in_features, out_features, grid, fp32_dest_acc=True)
        if shards:
            sharded.append(patches)
            classes[shards] += 1
            layout_of[patches] = shards
    assert _ranges(in_l1) == L1_PATCHES[grid], _ranges(in_l1)
    assert dict(classes) == SHARDED_MLP_CLASSES[grid], dict(classes)
    all_grids_layouts = {layout_of.get(math.prod(grid_thw)) for grid_thw in L1_VS_DRAM_ALL_GRIDS.values()}
    assert all_grids_layouts >= set(SHARDED_MLP_CLASSES[grid]), all_grids_layouts
    for grid_id in L1_VS_DRAM_CI_GRIDS:
        assert math.prod(VISION_GRIDS[grid_id]) in in_l1, grid_id
    assert math.prod(VISION_GRIDS["h36w36"]) in sharded
    model.device.grid = (8, 8)
    assert all(linear_program_config(model.device, 1024, *w.padded_shape[-2:], compute) for w, compute in linears)
    assert block_placement(block, 1024, "l1") is DRAM_PLACEMENT
    with expect_error(ValueError, "activations must be one of"):
        block_placement(block, 2304, "L1")


@pytest.mark.parametrize(
    "hidden, mlp_width, transposed_sizes",
    [(1280, 4800, 240), (1280, 5056, 240), (1408, 5120, 16)],
    ids=["hidden1280-mlp4800", "hidden1280-mlp5056", "hidden1408-mlp5120"],
)
def test_vision_block_placement_rule_mlp_width(monkeypatch, hidden, mlp_width, transposed_sizes):
    """The rule for the MLP's shards on 12x10 with widths other than the checkpoint's, where conditions bind that its
    widths meet at every size: with an MLP width of 4800 (150 tiles) fc2's K blocks of 6 tiles do not divide fc1's
    blocks of 15; with 5056 (158 tiles) fc1's width does not split into whole blocks of 16; with a hidden width of 1408
    (44 tiles) fc2's output takes 9 blocks of 5 tiles, so one of fc1's 10 blocks, fc2's K shards, would sit on a core
    with no block of fc2's output. fc1 and fc2 get transposed splits with the same rows per core at transposed_sizes
    patch counts, and "l1" keeps the MLP interleaved at every one, as fc1's and fc2's configs for the shards fail the
    matmul's checks there."""
    model = _model(_Recorder(monkeypatch), (12, 10), mlp_width=mlp_width, hidden=hidden)
    block = model.blocks[0]
    transposed = 0
    for patches in PATCHES:
        rows = nearest_32(patches)
        placement = block_placement(block, rows, "l1")
        assert placement is not L1_SHARDED_MLP_PLACEMENT, patches
        if placement is DRAM_PLACEMENT:
            continue
        assert _mlp_shards(model.device, block.mlp, rows, (12, 10)) is None, patches
        mlp = block.mlp
        fc1, fc2 = (
            linear_program_config(model.device, rows, *w.padded_shape[-2:], mlp.compute_kernel_config)
            for w in (mlp.fc1_weight, mlp.fc2_weight)
        )
        transposed += fc1.transpose_mcast and fc2.transpose_mcast and fc1.per_core_M == fc2.per_core_M
    assert transposed == transposed_sizes, transposed


@pytest.mark.parametrize("activations", ["l1", "l1_interleaved"])
@pytest.mark.parametrize("grid", list(L1_GRID_IDS.values()), ids=list(L1_GRID_IDS))
def test_vision_block_placement_l1_budget(monkeypatch, grid, activations):
    """The walk of the L1 budget at every patch count of L1_PATCHES, on the recorded forward of a 4-block tower: each
    op's circular buffers end PROOF_L1_FOREIGN_BYTES or more below the lowest tensor in a bank of PROOF_L1_BANK_BYTES,
    the LayerNorms keep their plan with that much more taken, an op whose config ttnn picks sees no tensor in L1, every
    matmul's config passes the matmul's checks for its operands, block 3 starts with the tensors of block 1 where block
    1 had them, and L1 is empty after the tower. The least margin is LEAST_MARGIN's; at 2304 patches on 12x10 the block
    ops' are the measured and walked ones, and with "l1" fc1's buffers and fc2's, as that P150 ran it, the measured
    ones."""
    recorder = _Recorder(monkeypatch)
    model = _model(recorder, grid)
    model.activations = activations
    _empty_l1(monkeypatch, grid)
    sizes, least = 0, None
    for patches in PATCHES:
        if block_placement(model.blocks[0], patches, activations) is DRAM_PLACEMENT:
            continue
        events = recorder.forward(model, patches)
        _assert_linears_valid(events, grid)
        margins, layouts, empty = _walk(events, PROOF_L1_BANK_BYTES)
        assert empty, f"L1 tensors left after the tower at {patches} patches"
        assert layouts[("ln1", 3)] == layouts[("ln1", 1)], patches
        for role, block, margin in margins:
            if margin is None:
                continue
            assert margin >= PROOF_L1_FOREIGN_BYTES, (patches, role, block, margin)
            if OPS[role] == "layer_norm":
                assert _layernorm_keeps_plan(margin), (patches, role, block, margin)
            if least is None or margin < least[0]:
                least = (margin, role)
        sizes += 1
        if grid == (12, 10) and patches == 2304:
            measured = {}
            for role, block, margin in _walk(events, P150_L1_BANK_BYTES)[0]:
                if block is not None:
                    measured[role] = min(margin, measured.get(role, margin))
            want = {**MEASURED_MARGINS_2304, **WALKED_MARGINS_2304}
            if activations == "l1":
                want.update(SHARDED_MLP_MARGINS_2304)
                # fc1's buffers as the walk took them, and fc2's as that P150 ran it, writing block shards.
                [fc1] = [call for call in _calls(events) if call[:2] == ("fc1", 0)]
                assert _cb_bytes(*fc1[2:]) == MEASURED_SHARDED_CB_BYTES_2304["fc1"]
                mlp = model.blocks[0].mlp
                k_tiles, n_tiles = (d // ttnn.TILE_SIZE for d in mlp.fc2_weight.padded_shape[-2:])
                fc2 = linear_program_config(
                    model.device,
                    2304,
                    k_tiles * ttnn.TILE_SIZE,
                    n_tiles * ttnn.TILE_SIZE,
                    mlp.compute_kernel_config,
                    in0_sharded=True,
                    out_sharded=True,
                )
                assert matmul_cb_bytes(fc2, k_tiles, True, out_sharded=True) == MEASURED_SHARDED_CB_BYTES_2304["fc2"]
            assert measured == want, measured
    assert sizes == sum((last - first) // 4 + 1 for first, last in L1_PATCHES[grid]), sizes
    assert least == LEAST_MARGIN[(activations, grid)], least


def test_vision_block_placement_l1_max_rows(monkeypatch):
    """Why the rule stops at L1_MAX_ROWS: with that limit lifted, "l1" would keep the activations in L1 at patch counts
    above it on 12x10 (UNCAPPED_12X10), and at some of them the walk of the L1 budget, as in
    test_vision_block_placement_l1_budget, finds an op whose circular buffers would overlap a tensor."""
    recorder = _Recorder(monkeypatch)
    model = _model(recorder, (12, 10))
    _empty_l1(monkeypatch, (12, 10))
    monkeypatch.setattr(common, "L1_MAX_ROWS", 4096)
    common._l1_block_placement.cache_clear()
    try:
        in_l1, overlap, least = [], [], None
        for patches in range(L1_MAX_ROWS + 4, 4096 + 1, 4):
            if block_placement(model.blocks[0], patches, "l1") is DRAM_PLACEMENT:
                continue
            in_l1.append(patches)
            margins = [m for m in _walk(recorder.forward(model, patches), PROOF_L1_BANK_BYTES)[0] if m[2] is not None]
            role, _, margin = min(margins, key=lambda m: m[2])
            if margin < 0:
                overlap.append(patches)
            if least is None or margin < least[0]:
                least = (margin, role, patches)
    finally:
        common._l1_block_placement.cache_clear()
    assert _ranges(in_l1) == UNCAPPED_12X10["l1_patches"], _ranges(in_l1)
    assert _ranges(overlap) == UNCAPPED_12X10["overlap_patches"], _ranges(overlap)
    assert least == UNCAPPED_12X10["least"], least


@pytest.mark.parametrize("patches", [2160, 2304])
def test_vision_block_placement_calls(monkeypatch, patches):
    """The recorded calls of a 4-block tower on 12x10, with activations "dram", "l1_interleaved" and "l1": each op gets
    the keywords of CALL_KEYWORDS and, where its field of the placement is not None, that memory_config, and no
    memory_config where it is None; each matmul gets linear_program_config's config for where its input and output
    are, so the configs of the interleaved placements but for fc1's output subblock and fc2's fuse_batch where the
    MLP's hidden activations are in block shards. DRAM_PLACEMENT is every activation in DRAM, with no memory_config for
    pre_layrnorm and GELU, which write where their inputs are, so with "dram" the RoPE tables' copies are the only
    tensors in L1. With the L1 placements out_proj reads its input from DRAM, fc2 from DRAM with "l1_interleaved" and
    from fc1's block shards, which GELU keeps, with "l1", QKV, fc1 and the other block ops from L1, and the last add
    writes the tower's output to DRAM."""
    recorder = _Recorder(monkeypatch)
    model = _model(recorder, (12, 10))
    _empty_l1(monkeypatch, (12, 10))
    dram = ttnn.DRAM_MEMORY_CONFIG
    assert DRAM_PLACEMENT == BlockPlacement(
        "dram", block_input=None, residual=dram, qkv=dram, fc1_out=dram, gelu_out=None
    )
    modes = (
        ("dram", DRAM_PLACEMENT),
        ("l1_interleaved", L1_INTERLEAVED_PLACEMENT),
        ("l1", L1_SHARDED_MLP_PLACEMENT),
    )
    for activations, placement in modes:
        model.activations = activations
        assert model.placement(SimpleNamespace(padded_shape=(1, 1, nearest_32(patches), 1184))) is placement
        events = recorder.forward(model, patches)
        memory_configs = dict(
            patch_embed=ttnn.DRAM_MEMORY_CONFIG,
            pre_layrnorm=placement.block_input,
            qkv=placement.qkv,
            heads=placement.qkv,
            out_proj=placement.residual,
            add1=placement.residual,
            fc1=placement.fc1_out,
            gelu=placement.gelu_out,
            fc2=placement.residual,
            add2=placement.residual,
        )
        # fc2 reads GELU's output, which is where fc1's is when GELU gets no memory config.
        fc2_input = placement.fc1_out if placement.gelu_out is None else placement.gelu_out
        reads_l1 = dict(patch_embed=False, pre_layrnorm=False, out_proj=False, fc2=_in_l1(fc2_input))
        for role, block, op, args, kwargs in _calls(events):
            case = (activations, role, block)
            memory_config = memory_configs.get(role)
            if role == "add2" and block == NUM_LAYERS - 1:
                memory_config = ttnn.DRAM_MEMORY_CONFIG
            assert set(kwargs) == CALL_KEYWORDS[op] | ({"memory_config"} if memory_config is not None else set()), case
            if memory_config is not None:
                assert kwargs["memory_config"] == memory_config, case
            if op == "to_memory_config":
                assert args[1] == ttnn.L1_MEMORY_CONFIG and not args[0].l1, case
                continue
            from_l1 = reads_l1.get(role, placement is not DRAM_PLACEMENT)
            assert {x.l1 for x in args[: ACTIVATION_ARGS[op]]} == {from_l1}, case
            sharded_input = role in ("gelu", "fc2") and placement.fc1_out.is_sharded()
            assert {x.memory_config.is_sharded() for x in args[: ACTIVATION_ARGS[op]]} == {sharded_input}, case
            if op == "rotary_embedding_llama":
                assert args[1].l1 and args[2].l1, case
            if op == "linear":
                x, w = args
                want = linear_program_config(
                    model.device,
                    x.padded_shape[-2],
                    *w.padded_shape[-2:],
                    kwargs["compute_kernel_config"],
                    in0_sharded=sharded_input,
                    out_sharded=role == "fc1" and placement.fc1_out.is_sharded(),
                )
                assert repr(kwargs["program_config"]) == repr(want), case
        if activations == "dram":
            assert sum(event[0] == "alloc" for event in events) == 2, events


def test_vision_l1_free_for_activations(monkeypatch):
    """l1_free_for_activations on 12x10 L1 memory views: a P150's as the tests open the device, empty, with tensors in
    the top PROOF_L1_FOREIGN_BYTES of the bank, and a larger bank pass; 110 or 130 banks, a bank 2 KB short of
    PROOF_L1_BANK_BYTES, a tensor 2 KB lower and a small tensor low in the bank, which leaves the largest free block
    above it, do not. M3VLVisionModel.placement then gives DRAM_PLACEMENT at 2304 patches with "l1" and
    "l1_interleaved", logs a warning the first time and every later fallback, with its reason, at debug level, and
    the forward puts only the RoPE tables' copies in L1. With "dram", or where the rule gives DRAM_PLACEMENT, it does
    not read the memory view."""
    recorder = _Recorder(monkeypatch)
    model = _model(recorder, (12, 10))
    bank, foreign = PROOF_L1_BANK_BYTES, PROOF_L1_FOREIGN_BYTES
    low_tensor = _memory_view(120, bank, [(32 * 1024, 2048)])
    l1_placements = (("l1", L1_SHARDED_MLP_PLACEMENT), ("l1_interleaved", L1_INTERLEAVED_PLACEMENT))
    views = [
        (_memory_view(120, bank), True),
        (_memory_view(120, bank, [(bank - foreign, foreign)]), True),
        (_memory_view(120, bank, [(bank - foreign, 2048), (bank - 4096, 4096)]), True),
        (_memory_view(120, P150_L1_BANK_BYTES), True),
        (_memory_view(110, bank), False),
        (_memory_view(130, bank), False),
        (_memory_view(120, bank - 2048), False),
        (_memory_view(120, bank, [(bank - foreign - 2048, 2048)]), False),
        (low_tensor, False),
    ]
    warnings, debugs = [], []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    debug_sink = logger.add(
        lambda message: debugs.append(message.record["message"]),
        level="DEBUG",
        filter=lambda record: record["level"].name == "DEBUG",
    )
    try:
        for view, free in views:

            def memory_view(device, buffer_type, view=view):
                assert device is model.device and buffer_type == ttnn.BufferType.L1
                return view

            monkeypatch.setattr(ttnn, "get_memory_view", memory_view)
            assert l1_free_for_activations(model.device)[0] is free, view
            for activations, l1_placement in l1_placements:
                model.activations = activations
                placement = model.placement(SimpleNamespace(padded_shape=(1, 1, 2304, 1184)))
                assert placement is (l1_placement if free else DRAM_PLACEMENT), (view, activations)
        monkeypatch.setattr(ttnn, "get_memory_view", lambda device, buffer_type: low_tensor)
        for activations, _ in l1_placements:
            model.activations = activations
            events = recorder.forward(model, 2304)
            assert sum(event[0] == "alloc" for event in events) == 2, (activations, events)
    finally:
        logger.remove(sink)
        logger.remove(debug_sink)
    assert len(warnings) == 1 and "110 L1 banks for 120 cores" in warnings[0], warnings
    fallbacks = [line for line in debugs if "keep their activations in DRAM" in line]
    for reason in ("110 L1 banks for 120 cores", "130 L1 banks for 120 cores", "per L1 bank, under", "below the top"):
        assert any(reason in line for line in fallbacks), (reason, fallbacks)

    def no_memory_view(device, buffer_type):
        raise AssertionError("read the L1 memory view")

    monkeypatch.setattr(ttnn, "get_memory_view", no_memory_view)
    for activations, _ in l1_placements:
        model.activations = activations
        assert model.placement(SimpleNamespace(padded_shape=(1, 1, 352, 1184))) is DRAM_PLACEMENT  # 324 patches
    model.activations = "dram"
    assert model.placement(SimpleNamespace(padded_shape=(1, 1, 2304, 1184))) is DRAM_PLACEMENT
