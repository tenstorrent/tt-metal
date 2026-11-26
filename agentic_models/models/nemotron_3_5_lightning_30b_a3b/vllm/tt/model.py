# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Nemotron-H full stack with the selected TP4/EP4 decoder contract.

Checkpoint conversion is setup-only. Embeddings produce the decoder's hidden
shards directly; only the terminal norm gathers hidden state. Logits remain
vocabulary-sharded for the common device sampler. Nemotron-H has no RoPE.
"""

import json
import math
import operator
import os
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig

import ttnn

from .multichip_decoder import DecoderState, MeshPolicy, MultichipDecoder
from .precision import layer_precision, load_precision
from .prefill_plan import prefill_plan

HF_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
HF_REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
DRAM = ttnn.DRAM_MEMORY_CONFIG
# perf/multiuser: Mamba recurrent/conv state lives in one slot-major tensor per
# layer; decode runs in place on a zero-copy first-B-rows view (decode bucket).
BATCHED_STATE = os.environ.get("NEMOTRON_BATCHED_STATE", "1") == "1"
# perf/multiuser hybrid prefix cache (NEMOTRON_PREFIX_CACHE=1): rows of saved per-layer Mamba state, keyed on the host by
# a hash of the token prefix they summarize (attention KV is cached by vLLM itself).
PREFIX_CACHE = os.environ.get("NEMOTRON_PREFIX_CACHE", "0") == "1"
# perf/multiuser round 3: Mamba conv state stored as rows (1,1,S,1536) instead of slot-major (S,1,1,1536) (which pads every
# slot to a 32-row tile), and batched decode keeps hidden in the (1,1,b,W) row layout through Mamba layers too.
MAMBA_ROWS = os.environ.get("NEMOTRON_MAMBA_ROWS", "1") == "1"
PREFIX_SNAPSHOTS = int(os.environ.get("NEMOTRON_PREFIX_SNAPSHOTS", "32")) if PREFIX_CACHE else 0


def batch_prefix_view(tensor, rows):
    return ttnn._ttnn.operations.experimental.batch_prefix_view(tensor, rows)


@dataclass
class BatchedSlot(DecoderState):
    """Placeholder for one slot of a slot-major Mamba state (see ModelCache.mamba)."""

    slot: int = 0


def mounted_snapshot():
    """Serving harnesses (tt-inference-server) place the pinned checkpoint outside $HF_HOME and export MODEL_WEIGHTS_DIR
    (or HF_MODEL) as a local snapshot directory. Prefer it when it holds config.json and the safetensors; a hub-cache
    snapshot directory must be the pinned revision."""
    for name in ("MODEL_WEIGHTS_DIR", "HF_MODEL"):
        value = os.environ.get(name)
        if value and (Path(value) / "config.json").is_file() and any(Path(value).glob("*.safetensors")):
            resolved = Path(value).resolve()
            if resolved.parent.name == "snapshots" and resolved.name != HF_REVISION:
                raise ValueError(f"{name}={value} is snapshot {resolved.name}, not the pinned revision {HF_REVISION}")
            return Path(value)
    return None


def checkpoint_path(path=None):
    if path is not None:
        return Path(path)
    mounted = mounted_snapshot()
    if mounted is not None:
        return mounted
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_MODEL, revision=HF_REVISION, local_files_only=True))


class Checkpoint:
    """Read one layer at a time, without retaining a full host state dict."""

    def __init__(self, path=None):
        self.path = checkpoint_path(path)
        self.config = AutoConfig.from_pretrained(self.path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def read(self, prefix):
        keys = [k for k in self.index if k.startswith(prefix)]
        result = {}
        for shard in sorted({self.index[k] for k in keys}):
            with safe_open(self.path / shard, framework="pt") as f:
                for key in keys:
                    if self.index[key] == shard:
                        result[key] = f.get_tensor(key)
        return result

    def tensor(self, key):
        with safe_open(self.path / self.index[key], framework="pt") as f:
            return f.get_tensor(key)


@dataclass(frozen=True)
class AttentionCacheSpec:
    """Resolved local-device pool geometry, in attention-layer order.

    ``attention_index`` is the plugin's attention-only index; ``layer_index``
    is the HF backbone index. K and V have separate buffers with this shape
    and dtype. Equal tensor_idx values share the pair, not recurrent state.
    """

    attention_index: int
    layer_index: int
    shape: tuple
    dtype: object
    tensor_idx: int

    @property
    def num_blocks(self):
        return self.shape[0]

    def as_tuple(self):
        return self.shape, self.dtype, self.tensor_idx


@dataclass
class ModelCache:
    """Caller-owned layer states. Attention pages are shared; Mamba slots aren't."""

    layers: list
    context_length: int
    num_slots: int
    block_size: int = 32
    attention_specs: tuple = ()
    # layer list index -> DecoderState whose tensors have num_slots rows (batched mode)
    mamba: dict = None
    # perf/multiuser prefix cache: layer list index -> DecoderState with PREFIX_SNAPSHOTS rows of saved Mamba state
    snap: dict = None

    @property
    def physical_blocks(self):
        """Exclusive physical page bounds, keyed by HF attention layer."""
        return {spec.layer_index: spec.num_blocks for spec in self.attention_specs}

    def tensors(self):
        seen = set()
        batched = list((self.mamba or {}).values())
        for layer in [*self.layers, batched]:
            for state in layer:
                for tensor in [state.key, state.value, *(state.conv or []), state.recurrent]:
                    if tensor is not None and id(tensor) not in seen:
                        seen.add(id(tensor))
                        yield tensor


@dataclass
class PrefillMetadata:
    """Preallocated per-slot scratch, refreshed between bounded prefill spans."""

    start: object
    positions: object
    fill_table: object
    unaligned_output: object
    absolute_start: int = -1


class NemotronModel:
    def __init__(
        self,
        mesh_device,
        *,
        checkpoint=None,
        optimized_terminal=True,
        precision_config=None,
        batched_prefill=False,
    ):
        self.mesh = mesh_device
        self.batched_prefill = batched_prefill
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("The full model requires the validated 1x4 mesh")
        weights = Checkpoint(checkpoint)
        self.config = weights.config
        self.checkpoint = weights.path
        self.layer_indices = tuple(range(self.config.num_hidden_layers))
        self.precision = load_precision(precision_config)
        self.policy = MeshPolicy(ccl_dtype=self.precision["ccl_dtype"])
        self.layers = []
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.head_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.precision["compute_fidelities"]["lm_head"]),
            math_approx_mode=self.precision["math_approx_mode"],
            fp32_dest_acc_en=self.precision["fp32_dest_acc_en"],
            packer_l1_acc=True,
        )
        self.embedding = self.tensor(
            weights.tensor("backbone.embeddings.weight"),
            dim=1,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        for index in self.layer_indices:
            layer = MultichipDecoder.from_state_dict(
                weights.read(f"backbone.layers.{index}."),
                hf_config=self.config,
                layer_idx=index,
                mesh_device=mesh_device,
                policy=self.policy,
                precision=layer_precision(self.precision, index),
            )
            self.layers.append(layer)
            print(f"MODEL_LAYER_LOADED {index} {layer.kind}", flush=True)
        self.norm = self.tensor(
            weights.tensor("backbone.norm_f.weight").reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT
        )
        head_key = "backbone.embeddings.weight" if self.config.tie_word_embeddings else "lm_head.weight"
        self.head = self.tensor(
            weights.tensor(head_key).T[None, None].contiguous(),
            dim=3,
            dtype=getattr(ttnn, self.precision["weight_groups"]["lm_head"]),
        )
        self.optimized_terminal = optimized_terminal
        self.terminal_policy = {
            "optimized": optimized_terminal,
            "weight_dtype": self.precision["weight_groups"]["lm_head"],
            "fidelity": self.precision["compute_fidelities"]["lm_head"],
        }
        if optimized_terminal:
            # Preserve the interleaved copy for multi-row prefill. The decode
            # copy is allocated before warmup/capture and shared by all slots.
            k, n = tuple(self.head.shape)[-2:]
            banks, readers = mesh_device.dram_grid_size().x, 3
            physical_n = math.ceil(n / (banks * readers * 32)) * banks * readers * 32
            bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
            head = self.head
            if physical_n != n:
                head = ttnn.pad(head, [(0, 0), (0, 0), (0, 0), (0, physical_n - n)], value=0.0)
            self.decode_head = ttnn.to_memory_config(
                head,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(bank_grid, (k, physical_n // banks), ttnn.ShardOrientation.ROW_MAJOR),
                ),
            )
            del head
            self.terminal_norm_memory = ttnn.create_sharded_memory_config(
                (32, k // 4),
                ttnn.CoreGrid(x=4, y=1),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.terminal_norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(4, 1),
                subblock_w=3,
                block_h=1,
                block_w=k // 128,
                inplace=False,
            )
            self.terminal_head_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=3,
                per_core_M=1,
                per_core_N=16,
                num_workers_per_dram_bank=readers,
                use_multicore_pipeline=True,
            )
            self.terminal_policy.update(
                readers=readers,
                in0_block_w=3,
                per_core_N=16,
                local_vocab=n,
                physical_vocab=physical_n,
                norm_cores=4,
                fp32_dest_acc=True,
                activation_dtype="bfloat16",
                output_dtype="bfloat16",
            )
        self.prefill_metadata = {}
        self._prefill_prepared = set()

    def _residual_transfer(self, tensors):
        if self.precision["residual_dtype"] == "bfloat16":
            return tensors
        # Norm and recurrent consumers require BF16. Quantize the inter-layer
        # transfer explicitly and restore their required input representation.
        return [
            ttnn.typecast(ttnn.typecast(x, getattr(ttnn, self.precision["residual_dtype"])), ttnn.bfloat16)
            for x in tensors
        ]

    def tensor(self, value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, dim=None):
        mapper = ttnn.ReplicateTensorToMesh(self.mesh) if dim is None else ttnn.ShardTensorToMesh(self.mesh, dim)
        return ttnn.from_torch(
            value.contiguous(), dtype=dtype, layout=layout, device=self.mesh, memory_config=DRAM, mesh_mapper=mapper
        )

    def cache_specs(self, *, context_length=262144, num_slots=1, per_layer_specs=None):
        """Describe/validate pools without allocating or importing vLLM.

        ``per_layer_specs`` is the pinned plugin's sequence of
        ``(shape, dtype, tensor_idx)`` triples. Physical capacity is independent
        of fixed-slot count; the scheduler must stay within each pool's bound.
        The selected default remains BFP8_B. An explicitly requested BF16 pool
        retains the decoder's existing caller-BF16 compatibility path.
        """
        import torch

        context_length, num_slots = operator.index(context_length), operator.index(num_slots)
        if not 1 <= context_length <= self.config.max_position_embeddings:
            raise ValueError("Cache context exceeds the checkpoint's physical position contract")
        if not 1 <= num_slots <= 32:
            raise ValueError("Expected between 1 and 32 request slots")
        attention = [layer for layer in self.layers if layer.kind == "attention"]
        minimum_blocks = (context_length + 31) // 32
        if per_layer_specs is None:
            per_layer_specs = [
                ((minimum_blocks * num_slots, 1, 32, 128), getattr(ttnn, self.precision["kv_cache_dtype"]), index)
                for index in range(len(attention))
            ]
        else:
            per_layer_specs = list(per_layer_specs)
        if len(per_layer_specs) != len(attention):
            raise ValueError("Cache specs must cover exactly the attention layers in model order")
        specs, pools = [], {}
        for index, (layer, entry) in enumerate(zip(attention, per_layer_specs, strict=True)):
            if len(entry) != 3:
                raise ValueError("Each cache spec must be a (shape, dtype, tensor_idx) triple")
            shape, dtype, tensor_idx = entry
            shape = tuple(operator.index(n) for n in shape)
            tensor_idx = operator.index(tensor_idx)
            if len(shape) != 4 or shape[1:] != (1, 32, 128):
                raise ValueError("TP4 attention pools require [physical_blocks, 1, 32, 128]")
            if not minimum_blocks <= shape[0] <= 2**31 - 1 or tensor_idx < 0:
                raise ValueError("Pool must hold one full context and use nonnegative int32 page addresses")
            if dtype is torch.bfloat16 or dtype in ("bfloat16", "torch.bfloat16"):
                dtype = ttnn.bfloat16
            elif dtype == "bfloat8_b":
                dtype = ttnn.bfloat8_b
            if dtype not in (ttnn.bfloat8_b, ttnn.bfloat16):
                raise ValueError("Preserved paged-cache policy supports only BFP8_B or explicit caller BF16")
            geometry = (shape, dtype)
            if tensor_idx in pools and pools[tensor_idx] != geometry:
                raise ValueError("Shared tensor_idx entries must have identical shape and dtype")
            pools[tensor_idx] = geometry
            specs.append(AttentionCacheSpec(index, layer.layer_idx, shape, dtype, tensor_idx))
        return tuple(specs)

    def bind_cache(self, cache, *, context_length, num_slots, per_layer_specs=None):
        """Validate and reuse caller storage; this method allocates no tensors.

        Binding is setup-only, before generator metadata preparation/capture.
        In particular it never replaces an existing pool to fit a new plan.
        """
        if not isinstance(cache, ModelCache):
            raise TypeError("Bind a ModelCache with both attention and recurrent state")
        if (cache.context_length, cache.num_slots, cache.block_size) != (context_length, num_slots, 32):
            raise ValueError("Existing cache geometry differs from the requested generator geometry")
        if per_layer_specs is None and cache.attention_specs:
            per_layer_specs = [spec.as_tuple() for spec in cache.attention_specs]
        specs = self.cache_specs(context_length=context_length, num_slots=num_slots, per_layer_specs=per_layer_specs)
        if len(cache.layers) != len(self.layers) or any(len(states) != num_slots for states in cache.layers):
            raise ValueError("Cache must contain every model layer and fixed slot")

        def allocation_id(tensor):
            # Views can be distinct Python objects sharing one device allocation.
            # Require independent backing storage, not just wrapper identities.
            get_id = getattr(tensor, "buffer_unique_id", None)
            identity = get_id() if callable(get_id) else None
            if identity is None:
                raise ValueError("Cache tensors must own allocated device storage")
            return identity

        pools, owners, recurrent = {}, {}, set()
        spec_iter = iter(specs)
        for layer, states in zip(self.layers, cache.layers, strict=True):
            if layer.kind == "attention":
                spec = next(spec_iter)
                pair = (states[0].key, states[0].value)
                if any(
                    tensor is None or tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype for tensor in pair
                ):
                    raise ValueError("Bound attention storage differs from its resolved shape/dtype")
                backing = tuple(allocation_id(tensor) for tensor in pair)
                if backing[0] == backing[1]:
                    raise ValueError("K and V must have separate storage")
                if any(state.key is not pair[0] or state.value is not pair[1] for state in states):
                    raise ValueError("All slots of one attention layer must share its physical pool")
                identity = tuple(map(id, pair))
                if spec.tensor_idx in pools and pools[spec.tensor_idx] != identity:
                    raise ValueError("Shared tensor_idx does not bind the same K/V storage")
                if any(root in owners and owners[root] != spec.tensor_idx for root in backing):
                    raise ValueError("Distinct tensor_idx values must not alias storage")
                if any(root in recurrent for root in backing):
                    raise ValueError("Attention and Mamba state must not alias storage")
                pools[spec.tensor_idx] = identity
                owners.update({root: spec.tensor_idx for root in backing})
            elif layer.kind == "mamba" and cache.mamba:
                continue
            elif layer.kind == "mamba":
                for state in states:
                    if state.conv is None or len(state.conv) != 3 or state.recurrent is None:
                        raise ValueError("Each Mamba slot requires three conv buffers and recurrent state")
                    tensors = [*state.conv, state.recurrent]
                    expected = [((1, 1, 1, 1536), ttnn.bfloat16)] * 3 + [((1, 16, 64, 128), ttnn.float32)]
                    for tensor, (shape, dtype) in zip(tensors, expected, strict=True):
                        root = allocation_id(tensor)
                        if tuple(tensor.shape) != shape or tensor.dtype != dtype or root in recurrent:
                            raise ValueError("Mamba state must preserve its dtype/shape and independent slot ownership")
                        if root in owners:
                            raise ValueError("Attention and Mamba state must not alias storage")
                        recurrent.add(root)
        cache.attention_specs = specs
        return cache

    def allocate_cache(self, *, context_length=262144, num_slots=1, per_layer_specs=None, existing_cache=None):
        """Allocate each resolved pool once, or bind existing storage unchanged."""
        import torch

        if existing_cache is not None:
            return self.bind_cache(
                existing_cache, context_length=context_length, num_slots=num_slots, per_layer_specs=per_layer_specs
            )
        specs = self.cache_specs(context_length=context_length, num_slots=num_slots, per_layer_specs=per_layer_specs)
        pools = {}
        for spec in specs:
            if spec.tensor_idx not in pools:
                pools[spec.tensor_idx] = DecoderState(
                    key=self.tensor(torch.zeros(spec.shape), dtype=spec.dtype),
                    value=self.tensor(torch.zeros(spec.shape), dtype=spec.dtype),
                )
        spec_iter = iter(specs)
        layers = []
        mamba = {}
        for index, layer in enumerate(self.layers):
            if layer.kind == "attention":
                state = pools[next(spec_iter).tensor_idx]
                layers.append([state] * num_slots)
            elif layer.kind == "mamba" and BATCHED_STATE:
                zero = lambda shape, dtype: self.tensor(torch.zeros(shape), dtype=dtype)
                conv_shape = (1, 1, num_slots, 1536) if MAMBA_ROWS else (num_slots, 1, 1, 1536)
                mamba[index] = DecoderState(
                    conv=[zero(conv_shape, ttnn.bfloat16) for _ in range(3)],
                    recurrent=zero((num_slots, 16, 64, 128), ttnn.float32),
                )
                layers.append([BatchedSlot(slot=s) for s in range(num_slots)])
            else:
                layers.append([layer.allocate_state(context_length=context_length) for _ in range(num_slots)])
        if mamba:
            self.mamba_zero_rows = (
                self.tensor(torch.zeros(1, 1, 1, 1536), dtype=ttnn.bfloat16),
                self.tensor(torch.zeros(1, 16, 64, 128), dtype=ttnn.float32),
            )
            if MAMBA_ROWS:
                eye = torch.eye(num_slots)
                # per-slot (1,1,S,1) column masks and inverses: row write = dst*inv + row*mask (binary ops only, exact
                # for 0/1 masks; eager ternary where/mac cost ~150 us vs ~40 us on the 1x4 mesh)
                self.conv_row_masks = [
                    (
                        self.tensor(eye[s].reshape(1, 1, num_slots, 1).contiguous(), dtype=ttnn.bfloat16),
                        self.tensor((1 - eye[s]).reshape(1, 1, num_slots, 1).contiguous(), dtype=ttnn.bfloat16),
                    )
                    for s in range(num_slots)
                ]
        snap = None
        if mamba and PREFIX_SNAPSHOTS > 0:
            k = PREFIX_SNAPSHOTS
            snap = {
                index: DecoderState(
                    conv=[zero((k, 1, 1, 1536), ttnn.bfloat16) for _ in range(3)],
                    recurrent=zero((k, 16, 64, 128), ttnn.float32),
                )
                for index in mamba
            }
        return ModelCache(layers, context_length, num_slots, attention_specs=specs, mamba=mamba or None, snap=snap)

    # ---- slot-major Mamba state helpers (batched mode) ----
    @staticmethod
    def _slot_row(t, slot):
        shape = list(t.shape)
        return ttnn.slice(t, [slot, 0, 0, 0], [slot + 1, shape[1], shape[2], shape[3]])

    @staticmethod
    def _conv_row(t, slot):
        """Row `slot` of a (1,1,S,1536) conv state as (1,1,1,1536)."""
        return ttnn.slice(t, [0, 0, slot, 0], [1, 1, slot + 1, 1536])

    def _conv_write(self, dst, row, slot):
        """Exact in-place write of a (1,1,1,1536) row into row `slot` of a (1,1,S,1536) conv state."""
        mask, inverse = self.conv_row_masks[slot]
        ttnn.add(ttnn.multiply(dst, inverse), ttnn.multiply(row, mask), output_tensor=dst)

    def _conv_get(self, t, slot):
        return self._conv_row(t, slot) if MAMBA_ROWS else self._slot_row(t, slot)

    def _conv_put(self, dst, row, slot):
        if MAMBA_ROWS:
            self._conv_write(dst, row, slot)
        else:
            ttnn.fill_cache(dst, row, slot)

    def mamba_load(self, cache, index, slot):
        """Per-slot working state for eager prefill (a copy unless the cache has one slot)."""
        batched = cache.mamba[index]
        if cache.num_slots == 1:
            return batched
        return DecoderState(
            conv=[self._conv_get(c, slot) for c in batched.conv], recurrent=self._slot_row(batched.recurrent, slot)
        )

    def mamba_store(self, cache, index, slot, state):
        if cache.num_slots == 1:
            return
        batched = cache.mamba[index]
        for dst, src in zip(batched.conv, state.conv, strict=True):
            self._conv_put(dst, src, slot)
        ttnn.fill_cache(batched.recurrent, state.recurrent, slot)

    def mamba_release(self, cache, slot):
        if not cache.mamba:
            return
        conv_zero, rec_zero = self.mamba_zero_rows
        for batched in cache.mamba.values():
            for dst in batched.conv:
                if MAMBA_ROWS:
                    ttnn.multiply(dst, self.conv_row_masks[slot][1], output_tensor=dst)  # zero row `slot`
                else:
                    self._conv_put(dst, conv_zero, slot)
            ttnn.fill_cache(batched.recurrent, rec_zero, slot)

    def mamba_remap(self, cache, remap):
        """Row d of every Mamba state takes old row remap[d] (handles cycles/duplicates)."""
        if not cache.mamba:
            return
        changed = [(d, s) for d, s in enumerate(remap) if d != s]
        if not changed:
            return
        for index in cache.mamba:
            batched = cache.mamba[index]
            for k, dst in enumerate([*batched.conv, batched.recurrent]):
                conv = k < len(batched.conv)
                get = self._conv_get if conv else self._slot_row
                rows = {s: get(dst, s) for s in sorted({s for _, s in changed})}
                for d, s in changed:
                    if conv:
                        self._conv_put(dst, rows[s], d)
                    else:
                        ttnn.fill_cache(dst, rows[s], d)
                del rows

    def mamba_snapshot(self, cache, slot, row):
        """Save every Mamba layer's state of `slot` into snapshot row `row` (prefix cache; pool is slot-major)."""
        for index, batched in cache.mamba.items():
            snap = cache.snap[index]
            for a, b in zip(batched.conv, snap.conv, strict=True):
                r = self._conv_get(a, slot)
                ttnn.fill_cache(b, r, row)
                del r
            r = self._slot_row(batched.recurrent, slot)
            ttnn.fill_cache(snap.recurrent, r, row)
            del r

    def mamba_restore(self, cache, row, slot):
        for index, batched in cache.mamba.items():
            snap = cache.snap[index]
            for a, b in zip(snap.conv, batched.conv, strict=True):
                r = self._slot_row(a, row)
                self._conv_put(b, r, slot)
                del r
            r = self._slot_row(snap.recurrent, row)
            ttnn.fill_cache(batched.recurrent, r, slot)
            del r

    def mamba_view(self, cache, index, rows):
        batched = cache.mamba[index]
        if rows == cache.num_slots:
            return batched
        return DecoderState(
            conv=list(batched.conv) if MAMBA_ROWS else [batch_prefix_view(c, rows) for c in batched.conv],
            recurrent=batch_prefix_view(batched.recurrent, rows),
        )

    def embed(self, tokens):
        out = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
        return ttnn.reshape(out, (1, 1, tokens.shape[-1], 672))

    def terminal(self, hidden, *, sampler_ready=False):
        # CCL program keys include logical shapes, even when tiled storage is
        # identical. Canonicalize rows before the collective: otherwise warming
        # every valid tail length creates hundreds of semaphore-owning programs
        # and exhausts L1_SMALL. Norm and LM head are token-local, so zero rows
        # neither influence valid rows nor change their precision policy.
        logical_rows = hidden.shape[2]
        padding = -logical_rows % 32
        if padding:
            hidden = ttnn.pad(hidden, [(0, 0), (0, 0), (0, padding), (0, 0)], value=0.0)
        hidden = ttnn.all_gather(hidden, dim=3, cluster_axis=1, memory_config=DRAM)
        optimized = self.optimized_terminal and hidden.shape[2] == 32
        if optimized:
            hidden = ttnn.to_memory_config(hidden, self.terminal_norm_memory)
        hidden = ttnn.rms_norm(
            hidden,
            weight=self.norm,
            epsilon=self.config.layer_norm_epsilon,
            compute_kernel_config=self.compute,
            memory_config=self.terminal_norm_memory if optimized else DRAM,
            **({"program_config": self.terminal_norm_program} if optimized else {}),
        )
        logits = ttnn.linear(
            hidden,
            self.decode_head if optimized else self.head,
            compute_kernel_config=self.head_compute,
            dtype=getattr(ttnn, self.precision["logits_dtype"]),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if optimized else DRAM,
            **({"program_config": self.terminal_head_program} if optimized else {}),
        )
        if optimized:
            logits = ttnn.to_memory_config(logits, DRAM)
            # Remove all invalid local-vocabulary columns before any sampler.
            # Zero-valued padding must never compete against real logits.
            logits = logits[:, :, :, : self.terminal_policy["local_vocab"]]
        # Decode's sampler consumes a physical 32-row tile. Keep that tile
        # instead of slicing logical rows only to pad them in the generator.
        # Padded hidden rows are zero and the head has no bias; their logits
        # remain zero. The sampler's persistent active mask owns row validity.
        return logits[:, :, :logical_rows, :] if padding and not sampler_ready else logits

    prefill_plan = staticmethod(prefill_plan)

    @staticmethod
    def _layer_items(items, layer):
        if not isinstance(items, dict):
            return items
        if layer.layer_idx in items:
            return items[layer.layer_idx]
        if layer.kind == "attention":
            raise ValueError(f"Missing page-table binding for attention layer {layer.layer_idx}")
        # Stateless MoE and recurrent Mamba do not consume attention pages.
        return next(iter(items.values()))

    def allocate_prefill_metadata(self, cache, *, layer_indices=None):
        """Call before any trace capture; retained for this configured cache."""
        import torch

        key = id(cache)
        if key not in self.prefill_metadata:
            integer = lambda x: self.tensor(x, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.prefill_metadata[key] = [
                PrefillMetadata(
                    start=integer(torch.zeros(1, dtype=torch.int32)),
                    positions=integer(torch.zeros(512, 1, dtype=torch.int32)),
                    fill_table=integer(torch.zeros(1, 64, dtype=torch.int32)),
                    unaligned_output=self.tensor(
                        torch.zeros(1, 1, 512, self.config.hidden_size), dim=3, layout=ttnn.ROW_MAJOR_LAYOUT
                    ),
                )
                for _ in range(cache.num_slots)
            ]
        if layer_indices is None:
            return self.prefill_metadata[key]
        for index in layer_indices:
            binding_key = (key, index)
            if binding_key not in self.prefill_metadata:
                # Start/relative positions and output scratch are shared across
                # layers of this slot: eager prefill consumes them sequentially.
                # Only fill mappings differ between layer cache groups.
                self.prefill_metadata[binding_key] = [
                    PrefillMetadata(
                        start=base.start,
                        positions=base.positions,
                        fill_table=self.tensor(
                            torch.zeros(1, 64, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                        ),
                        unaligned_output=base.unaligned_output,
                    )
                    for base in self.prefill_metadata[key]
                ]
        return {index: self.prefill_metadata[(key, index)] for index in layer_indices}

    def refresh_prefill_metadata(self, *, cache, slots, starts, page_host, layer_idx=None):
        """Host scheduler metadata only; no weights, activations, or logits cross here."""
        import torch

        if isinstance(page_host, dict):
            return {
                index: self.refresh_prefill_metadata(
                    cache=cache, slots=slots, starts=starts, page_host=pages, layer_idx=index
                )
                for index, pages in page_host.items()
            }
        binding_key = id(cache) if layer_idx is None else (id(cache), layer_idx)
        if binding_key not in self.prefill_metadata:
            raise RuntimeError("Allocate and prepare prefill metadata before capture")

        def copy(value, dst):
            host = ttnn.from_torch(value.contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, dst)

        rows = []
        for slot, start in zip(slots, starts, strict=True):
            metadata = self.prefill_metadata[binding_key][slot]
            pages = torch.as_tensor(page_host[slot], dtype=torch.int32).flatten()
            if not pages.numel():
                raise ValueError("An active slot needs a nonempty page table")
            first = start // cache.block_size
            selected = pages[first : first + 64]
            fill = pages[-1].repeat(1, 64)
            fill[0, : selected.numel()] = selected
            copy(fill, metadata.fill_table)
            copy(torch.tensor([start], dtype=torch.int32), metadata.start)
            copy((torch.arange(512, dtype=torch.int32) + start).reshape(512, 1), metadata.positions)
            metadata.absolute_start = start
            rows.append(metadata)
        return rows

    def prefill(
        self,
        tokens,
        *,
        cache,
        page_tables,
        slots,
        starts,
        positions=None,
        last_only=False,
        prefill_metadata=None,
    ):
        """Run one canonical bounded span per row; the generator owns chunking.

        Tokens contain exactly this span, not an absolute-position prefix. The
        generator refreshes metadata using the full scheduler page mapping and
        assembles returned diagnostic chunks on the host if all logits are asked
        for. This keeps device programs independent of total request length.
        """
        if prefill_metadata is None:
            prefill_metadata = (
                {index: [self.prefill_metadata[(id(cache), index)][slot] for slot in slots] for index in page_tables}
                if isinstance(page_tables, dict)
                else [self.prefill_metadata[id(cache)][slot] for slot in slots]
            )
        validation_metadata = (
            next(iter(prefill_metadata.values())) if isinstance(prefill_metadata, dict) else prefill_metadata
        )
        for row, start, metadata in zip(tokens, starts, validation_metadata, strict=True):
            length = row.shape[-1]
            if length not in range(1, 513) and length != 2048:
                raise ValueError("Split logical prompts with prefill_plan before the bounded model call")
            if start + length > cache.context_length:
                raise ValueError("Prefill exceeds configured cache context")
            if start % 512 and length > 512 - start % 512:
                raise ValueError("Split the initial unaligned span at the next 512-token boundary")
            if metadata.absolute_start != start:
                raise ValueError("Refresh this slot's prefill metadata before executing its span")
        hidden = [self.embed(row) for row in tokens]
        for index, (layer, states) in enumerate(zip(self.layers, cache.layers, strict=True)):
            layer_pages = self._layer_items(page_tables, layer)
            layer_metadata = self._layer_items(prefill_metadata, layer)
            if any(metadata.absolute_start != start for metadata, start in zip(layer_metadata, starts, strict=True)):
                raise ValueError("Layer-specific prefill metadata has stale absolute positions")
            if layer.kind == "moe":
                if self.batched_prefill:
                    hidden = layer.prefill_moe_prepared(hidden, state=states[slots[0]], page_table=layer_pages[0])
                else:
                    hidden = [
                        layer.prefill_moe_prepared([x], state=states[slot], page_table=page)[0]
                        for x, slot, page in zip(hidden, slots, layer_pages, strict=True)
                    ]
            elif layer.kind == "mamba" and cache.mamba:
                outputs = []
                for x, slot, page, start, metadata in zip(
                    hidden, slots, layer_pages, starts, layer_metadata, strict=True
                ):
                    state = self.mamba_load(cache, index, slot)
                    outputs.append(
                        layer.prefill_prepared(x, state=state, page_table=page, start_pos=start, metadata=metadata)
                    )
                    self.mamba_store(cache, index, slot, state)
                    del state
                hidden = outputs
            else:
                hidden = [
                    layer.prefill_prepared(x, state=states[slot], page_table=page, start_pos=start, metadata=metadata)
                    for x, slot, page, start, metadata in zip(
                        hidden, slots, layer_pages, starts, layer_metadata, strict=True
                    )
                ]
            hidden = self._residual_transfer(hidden)
        if last_only:
            hidden = [x[:, :, -1:, :] for x in hidden]
        return [self.terminal(x) for x in hidden]

    def prepare_prefill(self, *, cache, page_tables, page_host):
        """Compile the finite prefill family before model/sampler trace capture.

        One real layer of each kind supplies identical layer program signatures;
        the subsequent all-stack decode warmup supplies every layer's decode
        bindings. This mutates cache state: the generator MUST reset its actual
        cache and RNG after all warmup, before capture/request execution.
        """
        import torch

        table_signature = (
            (
                "per_layer",
                tuple((index, tuple(tuple(p.shape) for p in rows)) for index, rows in sorted(page_tables.items())),
            )
            if isinstance(page_tables, dict)
            else ("common", tuple(tuple(p.shape) for p in page_tables))
        )
        key = (id(cache), table_signature)
        if key in self._prefill_prepared:
            return self._prefill_prepared_report[key]
        self.allocate_prefill_metadata(cache, layer_indices=tuple(page_host) if isinstance(page_host, dict) else None)
        metadata_rows = self.refresh_prefill_metadata(cache=cache, slots=[0], starts=[0], page_host=page_host)
        representatives = {}
        attention_signatures = set()
        precision_signatures = set()
        for layer, states in zip(self.layers, cache.layers, strict=True):
            precision_signature = tuple(
                (role, layer.precision["weight_groups"][role], layer.precision["compute_fidelities"][role])
                for role in layer.precision["weight_groups"]
                if hasattr(layer, role)
            )
            if layer.kind == "attention":
                page = self._layer_items(page_tables, layer)[0]
                signature = (
                    tuple(page.shape),
                    tuple(states[0].key.shape),
                    states[0].key.dtype,
                    tuple(states[0].value.shape),
                    states[0].value.dtype,
                    precision_signature,
                )
                if signature in attention_signatures:
                    continue
                attention_signatures.add(signature)
                representatives[f"attention:{layer.layer_idx}"] = (layer, states[0])
            else:
                signature = (layer.kind, precision_signature)
                if signature not in precision_signatures:
                    name = layer.kind if layer.kind not in representatives else f"{layer.kind}:{layer.layer_idx}"
                    state = (
                        self.mamba_load(cache, self.layers.index(layer), 0)
                        if layer.kind == "mamba" and cache.mamba
                        else states[0]
                    )
                    representatives[name] = (layer, state)
                    precision_signatures.add(signature)
        lengths = list(range(1, min(512, cache.context_length) + 1))
        if cache.context_length >= 2048:
            lengths.append(2048)
        program_counts = [self.mesh.num_program_cache_entries()]
        for length in lengths:
            if length % 64 == 0 or length == 1:
                print(f"PREFILL_PREPARE length={length} programs={self.mesh.num_program_cache_entries()}", flush=True)
            tokens = self.tensor(
                torch.zeros(1, length, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            hidden = self.embed(tokens)
            for layer, state in representatives.values():
                page = self._layer_items(page_tables, layer)[0]
                metadata = self._layer_items(metadata_rows, layer)[0]
                if layer.kind == "moe":
                    out = layer.prefill_moe_prepared([hidden], state=state, page_table=page)[0]
                else:
                    out = layer.prefill_prepared(hidden, state=state, page_table=page, start_pos=0, metadata=metadata)
                transferred = self._residual_transfer([out])
                del transferred, out
                if layer.kind == "attention" and length < 512:
                    out = layer._prefill_unaligned_input(hidden)
                    del out
                    out = layer._prefill_unaligned_output(metadata, length)
                    del out
            # Last-token and all-logit diagnostics have different head shapes.
            out = self.terminal(hidden[:, :, -1:, :])
            del out
            out = self.terminal(hidden)
            del out
            del hidden, tokens
        # One longest initial unaligned prefix covers all scalar input slices,
        # position slices and RM output writes against their fixed 512 shapes.
        for layer, state in representatives.values():
            if layer.kind != "attention" or cache.context_length <= 1:
                continue
            count = min(511, cache.context_length - 1)
            metadata_rows = self.refresh_prefill_metadata(cache=cache, slots=[0], starts=[1], page_host=page_host)
            metadata = self._layer_items(metadata_rows, layer)[0]
            page = self._layer_items(page_tables, layer)[0]
            tokens = self.tensor(
                torch.zeros(1, count, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            hidden = self.embed(tokens)
            out = layer.prefill_prepared(hidden, state=state, page_table=page, start_pos=1, metadata=metadata)
            del out, hidden, tokens
        # Equal-bucket binary groups bound concat/split signatures independently
        # of logical row lengths, slot order, and request composition.
        group_buckets = []
        for layer, state in representatives.values():
            if layer.kind != "moe":
                continue
            page = self._layer_items(page_tables, layer)[0]
            physical_lengths = sorted({((n + 31) // 32) * 32 for n in lengths})
            for physical in physical_lengths:
                hidden = self.tensor(torch.zeros(1, 1, physical, self.config.hidden_size), dim=3)
                max_group = min(cache.num_slots, 32, 8192 // physical)
                # Concat keys include input aliasing. Real request rows own
                # distinct tensors, so repeated references to one dummy row
                # would compile an unusable alias-only warmup signature.
                rows = [hidden] + [ttnn.clone(hidden) for _ in range(max_group - 1)]
                count = 2
                while count <= max_group:
                    out = layer.prefill_moe_prepared(rows[:count], state=state, page_table=page)
                    del out
                    group_buckets.append([physical, count])
                    count *= 2
                del rows, hidden
        self.refresh_prefill_metadata(cache=cache, slots=[0], starts=[0], page_host=page_host)
        program_counts.append(self.mesh.num_program_cache_entries())
        report = {
            "logical_span_lengths": lengths,
            "mamba_prefill_chunk": {
                str(layer.layer_idx): layer.prefill_chunk for layer in self.layers if layer.kind == "mamba"
            },
            "projection_dispatch": "generic during prefill; tuned decode plans retained",
            "dispatch_counter_scope": "Python dispatches including warmup and capture; trace replays are counted by the generator",
            "unaligned_physical_rows": 512,
            "moe_group_buckets": group_buckets,
            "representative_layers": {kind: layer.layer_idx for kind, (layer, _) in representatives.items()},
            "program_cache_entries": program_counts,
            "cache_reset_required": True,
        }
        if not hasattr(self, "_prefill_prepared_report"):
            self._prefill_prepared_report = {}
        self._prefill_prepared_report[key] = report
        self._prefill_prepared.add(key)
        return report

    def decode(self, tokens, *, cache, page_tables, slots, current_positions, sampler_ready=False):
        hidden = [self.embed(row) for row in tokens]
        for layer, states in zip(self.layers, cache.layers, strict=True):
            layer_pages = self._layer_items(page_tables, layer)
            hidden = layer.decode_forward(
                hidden,
                state=[states[s] for s in slots],
                page_table=layer_pages,
                current_pos=current_positions,
                mask_inactive=True,
            )
            hidden = self._residual_transfer(hidden)
        packed = hidden[0] if len(hidden) == 1 else ttnn.concat(hidden, dim=2)
        return self.terminal(packed, sampler_ready=sampler_ready)

    def decode_batched(self, token_row, *, cache, page_tables, rows, current_positions, sampler_ready=False):
        """Decode the first ``rows`` slots as one packed batch (no per-slot split/concat).

        Numerically equivalent to ``decode`` over slots 0..rows-1: same per-layer ops,
        but hidden stays packed between layers and Mamba state is updated in place on a
        zero-copy prefix view of the slot-major state tensors.
        """
        if not cache.mamba:
            raise ValueError("decode_batched requires the slot-major Mamba cache (NEMOTRON_BATCHED_STATE=1)")
        b = rows
        width = self.config.hidden_size // 4
        hidden = self.embed(token_row)
        positions = current_positions[0] if b == 1 else ttnn.concat(current_positions, dim=0)
        active_flat = ttnn.typecast(ttnn.ge(ttnn.to_layout(positions, ttnn.TILE_LAYOUT), 0), ttnn.bfloat16)
        active_mask = ttnn.reshape(active_flat, (b, 1, 1, 1))
        if MAMBA_ROWS:
            return self._decode_batched_rows(
                hidden,
                cache=cache,
                page_tables=page_tables,
                b=b,
                positions=positions,
                active_mask=active_mask,
                active_flat=active_flat,
                sampler_ready=sampler_ready,
            )
        if b > 1:
            hidden = ttnn.reshape(hidden, (b, 1, 1, width))
        for index, (layer, states) in enumerate(zip(self.layers, cache.layers, strict=True)):
            if layer.kind == "mamba":
                state = self.mamba_view(cache, index, b)
                hidden = layer._mamba_decode(layer._norm(hidden), state=state, residual=hidden, active_mask=active_mask)
            else:
                x = hidden if b == 1 else ttnn.reshape(hidden, (1, 1, b, width))
                if layer.kind == "attention":
                    rows_pages = self._layer_items(page_tables, layer)[:b]
                    pages = rows_pages[0] if b == 1 else ttnn.concat(rows_pages, dim=0)
                    out = layer._attention_decode(
                        layer._norm(x), state=states[0], page_table=pages, current_pos=positions, residual=x
                    )
                else:
                    out = layer._finish(layer._moe(layer._norm(x)), x)
                hidden = out if b == 1 else ttnn.reshape(out, (b, 1, 1, width))
            hidden = self._residual_transfer([hidden])[0]
        packed = hidden if b == 1 else ttnn.reshape(hidden, (1, 1, b, width))
        return self.terminal(packed, sampler_ready=sampler_ready)

    def _decode_batched_rows(
        self, hidden, *, cache, page_tables, b, positions, active_mask, active_flat, sampler_ready
    ):
        """decode_batched with hidden kept as (1,1,b,W) rows through every layer (Mamba included)."""
        S = cache.num_slots
        active_rows = ttnn.reshape(active_flat, (1, 1, b, 1))
        if b != S:
            active_rows = ttnn.pad(active_rows, [(0, 0), (0, 0), (0, S - b), (0, 0)], 0.0)
        for index, (layer, states) in enumerate(zip(self.layers, cache.layers, strict=True)):
            if layer.kind == "mamba":
                state = self.mamba_view(cache, index, b)
                hidden = layer._mamba_decode_rows(
                    layer._norm(hidden), state=state, residual=hidden, active_mask=active_mask, active_rows=active_rows
                )
            elif layer.kind == "attention":
                rows_pages = self._layer_items(page_tables, layer)[:b]
                pages = rows_pages[0] if b == 1 else ttnn.concat(rows_pages, dim=0)
                hidden = layer._attention_decode(
                    layer._norm(hidden), state=states[0], page_table=pages, current_pos=positions, residual=hidden
                )
            else:
                hidden = layer._finish(layer._moe(layer._norm(hidden)), hidden)
            hidden = self._residual_transfer([hidden])[0]
        return self.terminal(hidden, sampler_ready=sampler_ready)

    def read_logits(self, logits):
        """Explicit correctness/host-sampling boundary, never used for token feedback."""
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, 3)).float()
