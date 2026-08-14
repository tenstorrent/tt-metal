# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full 48-layer Qwen3-Coder-30B-A3B-Instruct on the 4-die P300_X2 mesh.

Stage 05. This module is the *wrapper* around the stage-04 optimized multichip
decoder layer and it deliberately changes nothing about that layer's strategy:

* attention TP=4 (8 Q heads, 1 K head, 1 V head per die), experts EP=4 (32 of
  128 per die), router and both residual RMSNorms and the residual replicated;
* two all-reduces per layer, ``FABRIC_1D_RING``, 2 links prefill / 1 decode;
* expert weights ``bfloat4_b`` at LoFi with ``in0_block_w`` 16/12, attention
  projections ``bfloat8_b`` DRAM-sharded, paged KV cache ``bfloat16``;
* router top-k in fp32 logit space;
* **the inter-layer residual layout contract**: every layer takes and returns a
  replicated ``[1, 1, B, 2048]`` bfloat16 ``TILE`` ``DRAM_MEMORY_CONFIG``
  tensor, and there is no collective, gather, reshard or layout conversion
  between layers. ``prefill_hidden`` and ``decode_hidden`` below are literally a
  ``for`` loop over 48 layers with the residual threaded straight through.

What the wrapper adds, and where each new boundary lives:

``embed_tokens``
    **Replicated**, bf16, so the embedding output *is* the residual contract
    with no collective at all. A hidden-sharded embedding would be 4x smaller
    per die but would owe an all-gather on every prefill chunk and every decode
    token; at 0.622 GB/die against 22.35 GB of measured headroom
    (``doc/context_contract.json``) the replicated table is free and the
    collective is not. This is also the shape the stage-03 footprint probe
    allocated, so the published capacity numbers describe what actually runs.

``model.norm`` (final RMSNorm)
    Replicated, and shares the layer code: decode uses
    ``multichip_decoder.decode_residual_norm`` (width-sharded over 8 L1 cores,
    the same kernel and compute config as the two residual norms), prefill uses
    the interleaved ``ttnn.rms_norm``.

``lm_head``
    **Column-parallel over the vocabulary**: die *d* owns columns
    ``37984*d .. 37984*d+37983`` of ``[2048, 151936]``. 151936 = 4 * 37984 and
    37984 = 32 * 1187, so the split is exact and needs no vocabulary padding.
    **Logits never reach the host on the token-out path.** They do get gathered
    *on device*: the greedy strategy all-gathers the 37984-wide shard and takes
    a device argmax, and the top-k/top-p strategy all-gathers 32 candidate
    values and indices instead. Which of the two is faster here was measured,
    not assumed -- see ``sample_greedy_argmax``.

``rotary`` (decode only)
    ``ttnn.experimental.rotary_embedding_hf(is_decode_mode=True)`` reading a
    per-user cos/sin pair **gathered on device** by ``ttnn.embedding`` from a
    position tensor the trace advances with ``ttnn.plus_one``. The layer's
    shipped spelling, ``ttnn.experimental.rotary_embedding``, takes the position
    as a **Python int** compile-time argument and therefore cannot be replayed:
    a captured trace would rotate every subsequent token at the position it was
    captured at. Note this is the *HF* rotary, same ``rotate_half`` channel
    convention -- so unlike stage 04's rejected ``rotary_embedding_llama`` lever
    (README limitation 4) it needs no weight permutation, changes no KV-cache
    channel convention and leaves prefill untouched.
"""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig

from .functional_decoder import DecoderLayerConfig, KVCache
from .multichip_decoder import (
    MESH_SHAPE,
    NUM_DEVICES,
    TOPOLOGY,
    MeshContext,
    MeshDecoderConfig,
    MultichipWeights,
    _head_shard,
    _norm_compute_config,
    build_local_sparsity,
    decode_residual_norm,
    decoder_layer_decode_multichip,
    decoder_layer_prefill_multichip,
    fallback_audit,
    mesh_context,
    upload_multichip_weights,
)

HF_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
HF_REVISION = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"

HIDDEN_SIZE = 2048
VOCAB_SIZE = 151936
NUM_LAYERS = 48
HEAD_DIM = 128
MAX_CONTEXT = 262144
DEFAULT_PAGE_BLOCK_SIZE = 32
DEFAULT_MAX_BATCH_SIZE = 1
#: ``ttnn.sampling`` and ``nlp_create_qkv_heads_decode`` both address 32 fixed
#: user slots; decode is always one 32-row tile regardless of the active batch.
SAMPLING_SLOTS = 32
#: Trace region per device. Two traces (model decode + sampling) over 48 layers.
DEFAULT_TRACE_REGION_SIZE = 300_000_000
#: RoPE table rows materialised at construction; grown on demand to the request
#: horizon by ``ensure_rope_capacity`` so a short request never pays for 262144
#: rows (which would be 64 MB/die of cos plus 64 MB of sin).
DEFAULT_ROPE_CACHE_LEN = 8192

#: ``lm_head`` weight dtype. bfloat8_b halves the 155 MB/die bf16 read that a
#: decode step would otherwise make against a 2048x37984 weight.
LM_HEAD_WEIGHT_DTYPE = ttnn.bfloat8_b
#: The embedding table stays bf16: it is a gather, not a matmul, and bfloat8_b
#: would quantise every token's hidden state at the very top of the stack.
EMBED_WEIGHT_DTYPE = ttnn.bfloat16


class _WatcherCleanSampling1D(Sampling1D):
    """``Sampling1D`` with the force-argmax gather spelled the way this layer spells it.

    **Why this subclass exists.** ``ttnn.experimental.all_gather_async`` trips a
    BRISC ``ASSERT`` in ``minimal_default_writer.cpp`` when it is given
    ``topology=Topology::Linear`` **together with** ``num_workers_per_link=1``.
    Neither alone does it; the pair does, at any width. The full A/B matrix is
    ``doc/full_model/watcher_ab.log`` and the model-free reproducer is
    ``doc/full_model/probes/ccl_watcher_ab.py --leg linear_workers1``.

    ``Sampling1D._argmax_all_gather`` walks straight into that pair on any mesh
    smaller than T3K. Its first branch -- Ring, no barrier -- is guarded by
    ``default_topology(mesh) == Topology.Ring``, which is **False** on this 1x4
    Blackhole mesh, so the branch is unreachable here. The fallback then runs
    ``_get_argmax_all_gather_config``, which forces ``Topology.Linear`` for any
    mesh under 8 devices, and the call below it hardcodes
    ``num_workers_per_link=1``. Linear + 1 worker: exactly the tripping pair.

    The decoder layer's own two all-reduces have been watcher-clean for four
    stages, and the reason is visible in the same matrix: the layer never passes
    ``num_workers_per_link`` at all, so the op picks its default. This override
    does the same thing -- same op, same ``dim``, same semaphores, same
    ``Topology.Ring`` the layer uses, and **no tuning knobs pinned**. The
    matrix's ``sampler_shape_default_knobs`` leg is this exact call at this exact
    shape, and it is clean.

    This is a local workaround for an upstream bug, not a fix for it. Both
    reports (the op, and ``sampling_1d.py``'s unreachable Ring branch) still
    stand and should still be filed; this subclass just means stage 05 does not
    ship an unchecked-but-violated device invariant while they are open. When
    the op is fixed, delete this class and pass ``Sampling1D`` directly.

    Subclassing is the seam because ``Sampling1D.from_config`` builds through
    ``object.__new__(cls)`` and ``_bind_strategy`` binds
    ``self._pre_argmax_gather = self._argmax_all_gather`` by attribute lookup on
    the instance -- so the override is what gets bound. **No shared code is
    edited.**
    """

    def _argmax_all_gather(self, logits):
        cfg = self.config
        return ttnn.experimental.all_gather_async(
            logits,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=cfg.num_argmax_gather_links,
            memory_config=logits.memory_config(),
            topology=cfg.ag_topology,
            # Deliberately NOT passing chunks_per_sync / num_workers_per_link /
            # num_buffers_per_channel. Pinning num_workers_per_link=1 is the half
            # of the tripping pair we control. See the class docstring.
        )


def _lm_head_compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


class ShardedCheckpoint:
    """Read named tensors out of a sharded safetensors checkpoint on demand.

    The full checkpoint is 30.5B parameters, ~61 GB in bf16. Materialising it as
    one ``state_dict`` to build a model that uploads it layer by layer would
    need that whole 61 GB of host RAM at once; this reads only the tensors asked
    for, from only the shards that hold them, and holds nothing.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        index_path = self.path / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"checkpoint index is missing: {index_path}")
        self.weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(name)
        with safe_open(self.path / shard, framework="pt") as f:
            return f.get_tensor(name)

    def layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Every ``model.layers.<i>.*`` tensor, keyed layer-relative."""
        prefix = f"model.layers.{layer_idx}."
        by_shard: dict[str, list[str]] = {}
        for name, shard in self.weight_map.items():
            if name.startswith(prefix):
                by_shard.setdefault(shard, []).append(name)
        if not by_shard:
            raise KeyError(f"no tensors for layer {layer_idx}")
        out: dict[str, torch.Tensor] = {}
        for shard, names in by_shard.items():
            with safe_open(self.path / shard, framework="pt") as f:
                for name in names:
                    out[name[len(prefix) :]] = f.get_tensor(name)
        return out


def _validate_mesh(mesh_device) -> None:
    shape = tuple(int(v) for v in mesh_device.shape)
    if shape != MESH_SHAPE:
        raise ValueError(f"Qwen3CoderModel requires mesh {MESH_SHAPE}, got {shape}")
    if mesh_device.get_num_devices() != NUM_DEVICES:
        raise ValueError(f"Qwen3CoderModel requires exactly {NUM_DEVICES} devices")


def _rope_parameters(hf_config) -> dict:
    """``rope_parameters`` on current transformers, ``rope_theta`` on older ones.

    ``Qwen3MoeConfig`` no longer exposes a top-level ``rope_theta`` attribute --
    reading it raises ``AttributeError`` rather than returning ``None`` -- so
    the dict is the only spelling that works on both.
    """
    params = getattr(hf_config, "rope_parameters", None)
    if params:
        return dict(params)
    return {"rope_theta": hf_config.rope_theta, "rope_type": "default"}


def _rope_type(hf_config) -> str:
    return str(_rope_parameters(hf_config).get("rope_type", "default"))


def _rope_theta(hf_config) -> float:
    return float(_rope_parameters(hf_config)["rope_theta"])


def _rope_tables(hf_config, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The HF ``(cos, sin)`` tables for positions ``0..capacity-1``.

    Built here rather than through ``Qwen3MoeRotaryEmbedding`` so that no
    transformers model object is constructed at load time; the formula is the
    default rope (``rope_scaling`` is null in this checkpoint, which
    ``from_checkpoint`` asserts).
    """
    head_dim = int(getattr(hf_config, "head_dim", HEAD_DIM))
    theta = _rope_theta(hf_config)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    angles = torch.outer(torch.arange(capacity, dtype=torch.float64), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos().float(), angles.sin().float()


class Qwen3CoderModel:
    """The 48-layer causal LM over the stage-04 multichip decoder layer."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        checkpoint: ShardedCheckpoint,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_cache_len: int = MAX_CONTEXT,
        num_layers: int = NUM_LAYERS,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        rope_cache_len: int = DEFAULT_ROPE_CACHE_LEN,
    ) -> None:
        _validate_mesh(mesh_device)
        if not 1 <= int(max_batch_size) <= 32:
            # nlp_create_qkv_heads_decode_device_operation.cpp:51 asserts
            # num_users <= 32; a TTNN op limit, unchanged by TP.
            raise ValueError(f"max_batch_size must be in [1,32], got {max_batch_size}")
        if not 1 <= int(num_layers) <= int(hf_config.num_hidden_layers):
            raise ValueError(f"num_layers must be in [1,{hf_config.num_hidden_layers}]")
        if not 1 <= int(max_cache_len) <= int(hf_config.max_position_embeddings):
            raise ValueError(f"max_cache_len must be in [1,{hf_config.max_position_embeddings}]")
        if int(hf_config.hidden_size) != HIDDEN_SIZE or int(hf_config.vocab_size) != VOCAB_SIZE:
            raise ValueError("HF config does not match the Qwen3-Coder-30B-A3B full-model contract")
        if bool(hf_config.tie_word_embeddings):
            raise ValueError("this checkpoint has an untied lm_head; tied weights would be a different contract")
        if _rope_type(hf_config) != "default":
            raise ValueError(f"rope_type {_rope_type(hf_config)!r} is not supported by this port's rotary tables")

        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.max_batch_size = int(max_batch_size)
        self.max_cache_len = int(max_cache_len)
        self.num_layers = int(num_layers)
        self.page_block_size = int(page_block_size)
        self.hidden_size = HIDDEN_SIZE
        self.vocab_size = VOCAB_SIZE
        self.head_dim = int(getattr(hf_config, "head_dim", HEAD_DIM))
        self.rms_norm_eps = float(hf_config.rms_norm_eps)
        # Exact: 151936 = 4 * 37984 and 37984 = 32 * 1187.
        assert self.vocab_size % (32 * NUM_DEVICES) == 0, self.vocab_size
        self.local_vocab_size = self.vocab_size // NUM_DEVICES

        self.ctx: MeshContext = mesh_context(mesh_device)
        self.config = MeshDecoderConfig.from_hf(hf_config)
        self.global_config: DecoderLayerConfig = self.config.global_config

        self.embed_tokens = self._build_embedding(checkpoint)
        self.layers: list[MultichipWeights] = self._build_layers(checkpoint)
        self.final_norm, self.final_norm_rm = self._build_final_norm(checkpoint)
        self.lm_head = self._build_lm_head(checkpoint)

        self.sparsity = build_local_sparsity(mesh_device, self.config.local_moe)
        self.lm_head_compute_config = _lm_head_compute_config(mesh_device)
        self.norm_compute_config = _norm_compute_config(mesh_device)

        self.rope_cache_len = 0
        self.cos_table = None
        self.sin_table = None
        self.ensure_rope_capacity(min(int(rope_cache_len), self.max_cache_len))

        # ``_WatcherCleanSampling1D`` rather than ``Sampling1D``: same module,
        # same strategies, the force-argmax gather spelled without the pinned
        # ``num_workers_per_link`` that trips the watcher on this mesh. See the
        # class docstring above and ``doc/full_model/watcher_ab.log``.
        self.sampler = _WatcherCleanSampling1D.from_config(
            Sampling1DConfig(
                vocab_size=self.vocab_size,
                valid_vocab_size=self.vocab_size,
                mesh_device=mesh_device,
                tt_ccl=self.ctx.ccl,
                max_batch_size=32,
                max_top_k=32,
                num_gather_links=1,
                sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                allow_force_argmax=True,
                num_argmax_gather_links=1,
                ag_topology=TOPOLOGY,
                # **False, and that is a measurement.** ``Sampling1D``'s comment
                # calls the power-of-two pad a "big device-perf win for
                # non-power-of-2 vocab on the multi-device path". For a per-die
                # shard of 37984 it is the opposite: the pad is to 65536, a 1.73x
                # blow-up of the tensor ``ttnn.topk`` then scans, and
                # ``probes/sampler_probe.py`` measures the whole split path at
                # **11.006 ms padded against 6.151 ms unpadded**, 1.79x, at the
                # shipped logits shape with the sampled token unchanged.
                pad_to_power_of_2=False,
            )
        )
        self.sampler.load_device_buffers()
        self.kv_cache: list[KVCache] | None = None

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        mesh_device,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_cache_len: int = MAX_CONTEXT,
        num_layers: int = NUM_LAYERS,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        rope_cache_len: int = DEFAULT_ROPE_CACHE_LEN,
    ) -> "Qwen3CoderModel":
        checkpoint_path = Path(checkpoint_path)
        hf_config = AutoConfig.from_pretrained(checkpoint_path)
        checkpoint = ShardedCheckpoint(checkpoint_path)
        model = cls(
            mesh_device=mesh_device,
            hf_config=hf_config,
            checkpoint=checkpoint,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            num_layers=num_layers,
            page_block_size=page_block_size,
            rope_cache_len=rope_cache_len,
        )
        gc.collect()
        return model

    def _build_embedding(self, checkpoint: ShardedCheckpoint) -> ttnn.Tensor:
        host = checkpoint.get("model.embed_tokens.weight").float()
        tensor = ttnn.from_torch(
            host,
            dtype=EMBED_WEIGHT_DTYPE,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        del host
        gc.collect()
        return tensor

    def _build_layers(self, checkpoint: ShardedCheckpoint) -> list[MultichipWeights]:
        from .weight_mapping import convert_layer_weights

        layers = []
        for layer_idx in range(self.num_layers):
            sd = checkpoint.layer(layer_idx)
            torch_weights = convert_layer_weights(sd, self.hf_config)
            del sd
            layers.append(upload_multichip_weights(torch_weights, self.mesh_device, self.config))
            del torch_weights
            gc.collect()
        return layers

    def _build_final_norm(self, checkpoint: ShardedCheckpoint):
        host = checkpoint.get("model.norm.weight").float().reshape(-1)
        tiled = ttnn.from_torch(
            host.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # The layout the sharded rms_norm program factory reads; see
        # ``multichip_decoder.upload_multichip_weights.norm_row_major``.
        row_major = ttnn.from_torch(
            host.reshape(1, 1, host.numel() // 32, 32).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return tiled, row_major

    def _build_lm_head(self, checkpoint: ShardedCheckpoint) -> ttnn.Tensor:
        host = checkpoint.get("lm_head.weight").float().transpose(-2, -1).contiguous()
        assert tuple(host.shape) == (self.hidden_size, self.vocab_size), tuple(host.shape)
        tensor = ttnn.from_torch(
            host.reshape(1, 1, self.hidden_size, self.vocab_size),
            dtype=LM_HEAD_WEIGHT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        )
        del host
        gc.collect()
        return tensor

    # -- rotary ---------------------------------------------------------------

    def ensure_rope_capacity(self, required_len: int) -> bool:
        """Grow the device cos/sin tables to cover ``required_len`` positions."""
        required_len = int(required_len)
        if required_len <= self.rope_cache_len:
            return False
        if required_len > self.max_cache_len:
            raise ValueError(f"rotary capacity {required_len} exceeds context {self.max_cache_len}")
        capacity = min(self.max_cache_len, max(32, 1 << (required_len - 1).bit_length()))
        cos, sin = _rope_tables(self.hf_config, capacity)
        new = []
        for host in (cos, sin):
            new.append(
                ttnn.from_torch(
                    host.reshape(1, 1, capacity, self.head_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            )
        old = (self.cos_table, self.sin_table)
        self.cos_table, self.sin_table = new
        for tensor in old:
            if tensor is not None:
                ttnn.deallocate(tensor, True)
        self.rope_cache_len = capacity
        return True

    def rope_decode_tables(self, rotary_position: ttnn.Tensor):
        """Per-user ``(cos, sin)`` for one decode step, gathered **on device**.

        ``rotary_position`` is a ``[1, batch]`` uint32 device tensor. The gather
        is ``ttnn.embedding`` against the replicated cos/sin tables, so the
        position never leaves the device and the whole thing is capturable; the
        trace advances ``rotary_position`` itself with ``ttnn.plus_one``.

        Returns the height-sharded ``[1, batch, 1, head_dim]`` pair that
        ``rotary_embedding_hf(is_decode_mode=True)`` requires -- one core per
        user, the same ``_head_shard`` layout ``nlp_create_qkv_heads_decode``
        emits for Q and K.
        """
        batch = self.max_batch_size
        shard = _head_shard(32, self.head_dim, batch)
        out = []
        for table in (self.cos_table, self.sin_table):
            # [1, batch] -> [1, batch, head_dim] -> [1, 1, batch, head_dim]
            # -> [1, batch, 1, head_dim], the layout rotary_embedding_hf's decode
            # factory reads. Same sequence as ``RotarySetup1D.decode_forward``.
            gathered = ttnn.unsqueeze_to_4D(ttnn.embedding(rotary_position, table, layout=ttnn.TILE_LAYOUT))
            transposed = ttnn.transpose(gathered, 1, 2)
            if int(transposed.shape[1]) != batch:
                trimmed = ttnn.slice(transposed, [0, 0, 0, 0], [1, batch, 1, self.head_dim])
                ttnn.deallocate(transposed, True)
                transposed = trimmed
            out.append(ttnn.interleaved_to_sharded(transposed, shard))
            ttnn.deallocate(transposed, True)
        return out[0], out[1]

    def _rope_decode(self, tensor: ttnn.Tensor, cos_sharded, sin_sharded, _token_index):
        """The ``rope=`` seam handed to ``decoder_layer_decode_multichip``.

        ``_token_index`` is accepted and ignored: the position lives in the
        cos/sin pair, which is what makes this spelling replayable where the
        layer's default one is not.
        """
        shard = _head_shard(32, self.head_dim, self.max_batch_size)
        staged = ttnn.to_memory_config(tensor, shard)
        rotated = ttnn.experimental.rotary_embedding_hf(staged, cos_sharded, sin_sharded, is_decode_mode=True)
        ttnn.deallocate(staged, True)
        out = ttnn.to_memory_config(rotated, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rotated, True)
        return out

    # -- KV cache -------------------------------------------------------------

    def allocate_kv_cache(
        self,
        *,
        max_cache_len: int | None = None,
        num_blocks: int | None = None,
        page_table: ttnn.Tensor | None = None,
    ) -> list[KVCache]:
        """One paged ``KVCache`` per layer, 1 local KV head per die.

        512 B per token per layer per die -- a quarter of the single-die 2048 --
        which is what makes the advertised 262144 context fit; see
        ``doc/context_contract.json``.
        """
        cache_len = self.max_cache_len if max_cache_len is None else int(max_cache_len)
        blocks_per_seq = math.ceil(cache_len / self.page_block_size)
        total_blocks = self.max_batch_size * blocks_per_seq if num_blocks is None else int(num_blocks)
        local = self.config.local_attention
        caches = []
        for _ in range(self.num_layers):
            k, v = (
                ttnn.from_torch(
                    torch.zeros(total_blocks, local.num_key_value_heads, self.page_block_size, local.head_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                for _ in range(2)
            )
            caches.append(KVCache(k=k, v=v, page_table=page_table, block_size=self.page_block_size))
        return caches

    def ensure_internal_kv_cache(self, page_table: ttnn.Tensor | None = None) -> list[KVCache]:
        if self.kv_cache is None:
            self.kv_cache = self.allocate_kv_cache(page_table=page_table)
        return self.kv_cache

    @staticmethod
    def bind_page_table(kv_cache: Sequence[KVCache], page_table: ttnn.Tensor | None) -> list[KVCache]:
        """Point every layer's cache at ``page_table`` in place.

        The page table is a *persistent device tensor* owned by the caller (the
        generator, or vLLM later). Rebinding mutates the ``KVCache`` records
        rather than reallocating, so the tensor identity a captured trace
        recorded is preserved and an unchanged page table costs nothing.
        """
        for cache in kv_cache:
            cache.page_table = page_table
        return list(kv_cache)

    def reset_kv_cache(self, kv_cache: Sequence[KVCache] | None = None) -> None:
        selected = self.ensure_internal_kv_cache() if kv_cache is None else kv_cache
        for cache in selected:
            ttnn.fill(cache.k, 0.0, memory_config=cache.k.memory_config(), output_tensor=cache.k)
            ttnn.fill(cache.v, 0.0, memory_config=cache.v.memory_config(), output_tensor=cache.v)

    # -- forward: prefill -----------------------------------------------------

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, S]`` uint32 -> replicated ``[1, 1, S, 2048]``, no collective."""
        hidden = ttnn.embedding(
            tokens,
            self.embed_tokens,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        hidden = ttnn.unsqueeze_to_4D(hidden)
        return ttnn.reshape(hidden, (1, 1, int(hidden.shape[-2]), self.hidden_size))

    def prefill_hidden(
        self,
        tokens: ttnn.Tensor,
        *,
        kv_cache: Sequence[KVCache] | None = None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        """Run the whole stack over one user's prompt. ``S`` is arbitrary.

        Nothing here constrains ``S``: the collectives scatter on dim 3 (hidden,
        2048, fixed), ``attention_prefill`` slices RoPE's tile padding back, and
        ``moe_prefill_optimized`` pads to its chunk internally and slices back.
        """
        caches = self.ensure_internal_kv_cache() if kv_cache is None else kv_cache
        if len(caches) != self.num_layers:
            raise ValueError(f"kv_cache has {len(caches)} layers, expected {self.num_layers}")
        hidden = self.embed_prefill(tokens)
        seq_len = int(hidden.shape[-2])
        self.ensure_rope_capacity(seq_len)
        # Exactly ``seq_len`` rows, including non-tile-aligned lengths -- the
        # same shape the single-layer prefill gates pass at S = 33/100/257.
        cos = ttnn.slice(self.cos_table, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        sin = ttnn.slice(self.sin_table, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        for layer_idx in range(self.num_layers):
            hidden = decoder_layer_prefill_multichip(
                hidden,
                self.layers[layer_idx],
                self.config,
                self.ctx,
                cos,
                sin,
                self.sparsity,
                kv_cache=caches[layer_idx],
                user_id=user_id,
            )
        ttnn.deallocate(cos, True)
        ttnn.deallocate(sin, True)
        return hidden

    def prefill_norm(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            hidden,
            weight=self.final_norm,
            epsilon=self.rms_norm_eps,
            compute_kernel_config=self.norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def select_prefill_rows(self, hidden: ttnn.Tensor, rows: Sequence[int]) -> ttnn.Tensor:
        """Keep only ``rows`` of a ``[1, 1, S, H]`` prefill result."""
        seq_len = int(hidden.shape[-2])
        pieces = []
        for row in rows:
            if not 0 <= int(row) < seq_len:
                raise ValueError(f"prefill row {row} is outside [0,{seq_len})")
            if seq_len == 1:
                # At a **one-token prompt** the requested slice covers the whole
                # tensor, and ``ttnn.slice`` then hands back a view of its input
                # rather than a copy -- as a *different* Python object, so an
                # ``is`` guard does not catch it. The caller deallocates
                # ``hidden`` immediately afterwards, leaving the retained row
                # pointing at freed DRAM; that does not raise, it **segfaults**
                # in whatever reads it next (the final norm here). Copy instead.
                # `probes/prompt_len_1_repro.py` is the four-line reproduction.
                pieces.append(ttnn.clone(hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                continue
            pieces.append(
                ttnn.slice(
                    hidden,
                    [0, 0, int(row), 0],
                    [1, 1, int(row) + 1, self.hidden_size],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        if len(pieces) == 1:
            return pieces[0]
        out = ttnn.concat(pieces, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for piece in pieces:
            ttnn.deallocate(piece, True)
        return out

    def local_logits(self, normed: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, rows, 2048]`` -> this die's ``[1, 1, rows, 37984]`` logits."""
        return ttnn.linear(
            normed,
            self.lm_head,
            compute_kernel_config=self.lm_head_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    def gather_logits_to_torch(self, local_logits: ttnn.Tensor, *, valid_rows: int | None = None) -> torch.Tensor:
        """Host-side full-vocabulary logits. **Not** on the token-out path.

        Used by ``return_all_logits`` prefill checks and the host-sampling
        compatibility mode only; the measured decode path never calls this.
        """
        gathered = ttnn.all_gather(
            local_logits,
            dim=3,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPOLOGY,
        )
        host = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).float()
        ttnn.deallocate(gathered, True)
        if valid_rows is not None:
            host = host[..., : int(valid_rows), :]
        return host[..., : self.vocab_size]

    # -- forward: decode ------------------------------------------------------

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, 1, 32]`` uint32 -> replicated ``[1, 1, batch, 2048]``."""
        hidden = ttnn.embedding(
            tokens,
            self.embed_tokens,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        hidden = ttnn.unsqueeze_to_4D(hidden)
        flat = ttnn.reshape(hidden, (1, 1, int(hidden.shape[-2]), self.hidden_size))
        if int(flat.shape[-2]) == self.max_batch_size:
            return flat
        sliced = ttnn.slice(
            flat, [0, 0, 0, 0], [1, 1, self.max_batch_size, self.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(flat, True)
        return sliced

    def decode_hidden(
        self,
        tokens: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        rotary_position: ttnn.Tensor,
        kv_cache: Sequence[KVCache] | None = None,
    ) -> ttnn.Tensor:
        caches = self.ensure_internal_kv_cache() if kv_cache is None else kv_cache
        if len(caches) != self.num_layers:
            raise ValueError(f"kv_cache has {len(caches)} layers, expected {self.num_layers}")
        hidden = self.embed_decode(tokens)
        cos, sin = self.rope_decode_tables(rotary_position)
        for layer_idx in range(self.num_layers):
            hidden = decoder_layer_decode_multichip(
                hidden,
                self.layers[layer_idx],
                self.config,
                self.ctx,
                cos,
                sin,
                caches[layer_idx],
                current_pos,
                0,  # token_index: unused by the rope seam below, see _rope_decode
                rope=self._rope_decode,
            )
        ttnn.deallocate(cos, True)
        ttnn.deallocate(sin, True)
        return hidden

    def decode_terminal(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Final norm + column-parallel ``lm_head``, sampler-ready local logits.

        The norm is the layer's own width-sharded decode kernel, and the shard
        it emits is exactly the width-sharded L1 config the projections read, so
        crossing into the head costs one sharded-to-interleaved.
        """
        normed_sharded = decode_residual_norm(hidden, self.final_norm_rm, self.rms_norm_eps)
        normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed_sharded, True)
        # ``ttnn.sampling`` addresses 32 fixed user slots, and it compares the
        # *logical* shapes of its values and indices, so the logits handed to it
        # must be logically 32 rows and not ``batch`` rows padded to a tile.
        # The rows are already physically there -- ``batch <= 32`` and decode is
        # one 32-row tile -- so this only rewrites the logical shape.
        rows = int(normed.shape[-2])
        if rows < SAMPLING_SLOTS:
            padded = ttnn.pad(
                normed,
                [(0, 0), (0, 0), (0, SAMPLING_SLOTS - rows), (0, 0)],
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(normed, True)
            normed = padded
        logits = self.local_logits(normed)
        ttnn.deallocate(normed, True)
        return logits

    def decode_forward_from_ttnn_inputs(
        self,
        tokens: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        *,
        rotary_position: ttnn.Tensor,
        kv_cache: Sequence[KVCache] | None = None,
        advance_position: bool = True,
    ) -> ttnn.Tensor:
        """Token in -> sampler-ready local logits out, entirely on device.

        With ``advance_position`` the two position tensors are incremented
        **inside** this graph, so a captured trace steps its own positions on
        replay and the host never refreshes them per token.
        """
        hidden = self.decode_hidden(
            tokens,
            current_pos=current_pos,
            rotary_position=rotary_position,
            kv_cache=kv_cache,
        )
        logits = self.decode_terminal(hidden)
        if advance_position:
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(rotary_position)
        return logits

    # -- sampling -------------------------------------------------------------

    def sample_split(self, logits, *, k, p, temp, seeds=None, tt_out_tok=None):
        """Canonical split sampling: local top-32 -> all-gather -> ``ttnn.sampling``.

        ``k=1, p=0, temp=1`` is **semantically greedy**: the global argmax is by
        construction inside some die's local top-32, and the all-gather makes
        all four dies' candidates visible before the top-1 is taken.
        """
        return self.sampler.decode_forward(
            logits, k=k, p=p, temp=temp, seeds=seeds, tt_out_tok=tt_out_tok, enable_log_probs=False
        )[0]

    def sample_greedy_argmax(self, logits, *, tt_out_tok=None):
        """``Sampling1D``'s force-argmax path: all-gather the full vocabulary, argmax.

        Still the common module, still on device, still traced, still writes the
        sampled token straight into ``tt_out_tok`` -- it is a different strategy
        inside the same implementation, not a custom sampler.

        **This is what greedy uses**, because at this vocabulary it is 5.5x
        faster than the top-k/top-p split path (1.125 ms against 6.155 ms in the
        48-layer model, both rows of ``doc/full_model/probes/perf_full_model.csv``;
        the standalone sweep is ``doc/full_model/probes/sampler_probe.log``) and
        produces the same token. It gathers 151936 bf16 columns per die where
        the split path gathers 32 values plus 32 indices, so the trade is
        bandwidth against a 37984-wide ``ttnn.topk``, and on this mesh the
        bandwidth is cheaper. The moment any slot asks for ``top_k > 1`` or
        ``top_p > 0`` the generator switches back to ``sample_split``.
        """
        return self.sampler.decode_forward(logits, tt_out_tok=tt_out_tok, enable_log_probs=False)[0]

    # -- audit ----------------------------------------------------------------

    def runtime_fallback_audit(self, batch: int | None = None) -> dict:
        """The layer audit, plus the boundaries this wrapper owns."""
        batch = self.max_batch_size if batch is None else int(batch)
        audit = fallback_audit(self.layers[0], self.config, batch)
        audit.update(
            {
                "num_layers": self.num_layers,
                "embedding": "replicated_bf16_no_collective",
                "residual_contract": "replicated [1,1,B,2048] bf16 TILE DRAM, no inter-layer collective",
                "final_norm": "replicated, width-sharded decode kernel",
                "lm_head_parallelism": "column_parallel_over_vocab",
                "lm_head_local_vocab": self.local_vocab_size,
                "lm_head_weight_dtype": str(LM_HEAD_WEIGHT_DTYPE),
                "vocab_padding": 0,
                "decode_rope": "rotary_embedding_hf(is_decode_mode=True), device position gather",
                "decode_rope_position_source": "device tensor advanced by ttnn.plus_one inside the trace",
                "sampling_greedy": "Sampling1D force-argmax (all-gather vocab -> ttnn.argmax), traced, tt_out_tok",
                "sampling_topk_topp": "Sampling1D split (local topk -> all-gather 32 candidates -> ttnn.sampling)",
                "sampling_pad_to_power_of_2": False,
                "host_logit_readback_on_token_out_path": False,
                "host_argmax_on_token_out_path": False,
                "kv_cache_dtype": "bfloat16",
                "kv_cache_paged": True,
                "page_block_size": self.page_block_size,
                "collective_topology": str(TOPOLOGY),
                "prefill_num_links": self.ctx.num_links,
                "decode_num_links": self.ctx.decode_num_links,
            }
        )
        return audit

    def teardown(self) -> None:
        if self.kv_cache is not None:
            for cache in self.kv_cache:
                ttnn.deallocate(cache.k, True)
                ttnn.deallocate(cache.v, True)
            self.kv_cache = None


__all__ = [
    "DEFAULT_MAX_BATCH_SIZE",
    "DEFAULT_PAGE_BLOCK_SIZE",
    "DEFAULT_ROPE_CACHE_LEN",
    "DEFAULT_TRACE_REGION_SIZE",
    "HF_MODEL_ID",
    "HF_REVISION",
    "MAX_CONTEXT",
    "NUM_LAYERS",
    "Qwen3CoderModel",
    "ShardedCheckpoint",
]
