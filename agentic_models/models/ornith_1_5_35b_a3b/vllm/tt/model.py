# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Text-only Ornith autoregressive model over the selected TP4/EP4 decoder.

Checkpoint conversion and state allocation are explicit setup operations. All
forward methods consume device tensors, preserve the decoder residual contract,
and return vocabulary-sharded device logits. Sampling and trace lifetime belong
to the generator. The vision encoder and the optional MTP training head are not
part of the HF text autoregressive path.
"""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import ttnn

from .multichip_decoder import CollectiveWorkspace, MeshPolicy, MultichipDecoder
from .optimized_decoder import DecoderState, SharedCacheBatchPlan
from .precision import SAFE_POLICY, load_precision

DRAM = ttnn.DRAM_MEMORY_CONFIG
HF_MODEL = "ornith-ai/Ornith-1.5-35B-A3B"
HF_REVISION = "10fbf86fed7ecee4a061f8b499a618f46001cac1"


def measured_kv_capacity(precision=None):
    """Load the adjacent full-stack capacity measurement for the selected policy.

    This setup-only metadata check allocates no device storage. Wrapper-only
    pool selection does not change the explicitly measured allocation schedule.
    Changes to the model, generator, decoder or precision construction code
    invalidate the automatic default until its capacity evidence is refreshed.
    """
    root = Path(__file__).resolve().parents[1]
    path = root / "doc/full_model/kv_capacity.json"
    if not path.is_file():
        raise RuntimeError("Full-stack measured KV capacity is absent: doc/full_model/kv_capacity.json")
    record = json.loads(path.read_text())
    required = dict(
        status="measured",
        num_devices=4,
        mesh=[1, 4],
        layers=40,
        fixed_slot_capacity=32,
        logical_context_per_request=262144,
        page_size=32,
    )
    if any(record.get(key) != value for key, value in required.items()):
        raise RuntimeError("KV capacity evidence does not describe the selected full-stack TP4/EP4 policy")
    measured_policy = record.get("precision_policy", SAFE_POLICY)
    requested_policy = load_precision(precision)
    # IDs label experiments; actual runtime fields determine the memory contract.
    if {k: v for k, v in requested_policy.items() if k != "config_id"} != {
        k: v for k, v in measured_policy.items() if k != "config_id"
    }:
        raise RuntimeError(
            "Automatic KV capacity requires a matching measured precision policy; use an explicit budget for experiments"
        )
    blocks = record.get("physical_num_blocks")
    if (
        not isinstance(blocks, int)
        or blocks < 32
        or record.get("measured_largest_passing_blocks") != blocks
        or record.get("measured_smallest_failing_blocks") != blocks + 1
        or record.get("physical_token_capacity") != blocks * 32
        or record.get("max_tokens_all_users") != blocks * 32 - record.get("plugin_output_reservation_tokens", -1)
    ):
        raise RuntimeError("KV capacity requires a consistent adjacent physical pass/fail pair and plugin budget")
    evidence = record.get("evidence", [])
    if not evidence or any(not (root / name).is_file() for name in evidence):
        raise RuntimeError("Recorded KV capacity evidence is missing")
    for name in (
        "tt/model.py",
        "tt/generator.py",
        "tt/multichip_decoder.py",
        "tt/optimized_decoder.py",
        "tt/precision.py",
    ):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != record.get("source_sha256", {}).get(name):
            raise RuntimeError(f"Measured KV capacity requires review after runtime source change: {name}")
    return record


def mounted_snapshot():
    """Serving harnesses (tt-inference-server) place the pinned checkpoint outside $HF_HOME and export MODEL_WEIGHTS_DIR
    (or HF_MODEL) as a local snapshot directory. Prefer it when it holds config.json and the safetensors; a hub-cache
    snapshot directory must be the pinned revision."""
    revision = os.environ.get("HF_REVISION", HF_REVISION)
    for name in ("MODEL_WEIGHTS_DIR", "HF_MODEL"):
        value = os.environ.get(name)
        if value and (Path(value) / "config.json").is_file() and any(Path(value).glob("*.safetensors")):
            resolved = Path(value).resolve()
            if resolved.parent.name == "snapshots" and resolved.name != revision:
                raise ValueError(f"{name}={value} is snapshot {resolved.name}, not the pinned revision {revision}")
            return resolved
    return None


def resolve_checkpoint(checkpoint=None):
    """Resolve the exact local checkpoint; never download implicitly at runtime."""
    if checkpoint is not None:
        root = Path(checkpoint).expanduser().resolve()
    else:
        root = mounted_snapshot() or (
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            / "hub/models--ornith-ai--Ornith-1.5-35B-A3B/snapshots"
            / os.environ.get("HF_REVISION", HF_REVISION)
        )
    if not (root / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Indexed Ornith checkpoint is absent: {root}")
    return root


class CheckpointReader:
    """Read one layer at a time, releasing its host tensors after upload."""

    def __init__(self, root):
        self.root = Path(root)
        self.mapping = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]

    def read(self, keys, *, strip_prefix=""):
        from safetensors import safe_open

        keys = tuple(keys)
        result = {}
        for filename in sorted({self.mapping[key] for key in keys}):
            with safe_open(self.root / filename, framework="pt") as stream:
                for key in keys:
                    if self.mapping[key] == filename:
                        result[key[len(strip_prefix) :]] = stream.get_tensor(key)
        return result

    def tensor(self, key):
        return self.read([key])[key]

    def layer(self, index):
        prefix = f"model.language_model.layers.{index}."
        keys = [key for key in self.mapping if key.startswith(prefix)]
        if not keys:
            raise KeyError(f"Checkpoint has no layer {index}")
        return self.read(keys, strip_prefix=prefix)


@dataclass
class ModelState:
    """Stable layer-major state: ``layers[layer][slot]``.

    External callers may construct a state with their existing DecoderState
    pools. They must keep its backing storage alive for all bound traces. Shared
    page pools use the decoder's explicit shared-cache batch plans. The state
    factory defaults to request-local pools for B1 and a shared pool for B>1;
    both use the selected persistent K/V dtype with BF16 projected update inputs.
    """

    layers: tuple
    context: int
    batch_size: int
    page_size: int = 32
    shared_cache: bool = False
    reset_donors: dict | None = None
    shared_cache_plans: tuple | None = None
    page_table: object = None
    row_page_tables: tuple | None = None

    @property
    def layer_states(self):
        return self.layers


class OrnithModel:
    """HF embedding -> forty selected decoders -> Qwen norm -> TP4 LM head."""

    @classmethod
    def from_pretrained(
        cls, mesh_device, *, checkpoint=None, layer_indices=None, policy=None, rope_context=None, precision_config=None
    ):
        import torch
        from transformers import AutoConfig

        self = cls()
        self.device = mesh_device
        self.checkpoint = resolve_checkpoint(checkpoint)
        self.hf_config = AutoConfig.from_pretrained(self.checkpoint)
        self.config = self.hf_config.text_config
        self.policy = policy or MeshPolicy()
        self.precision = load_precision(precision_config)
        self.policy = replace(self.policy, ccl_dtype=self.precision["ccl_dtype"])
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("Ornith requires the selected 1x4 TP4/EP4 mesh")
        self.layer_indices = tuple(range(self.config.num_hidden_layers) if layer_indices is None else layer_indices)
        if (
            not self.layer_indices
            or len(set(self.layer_indices)) != len(self.layer_indices)
            or any(index < 0 or index >= self.config.num_hidden_layers for index in self.layer_indices)
        ):
            raise ValueError("Layer indices must be a nonempty sequence of distinct checkpoint layers")
        self.is_reduced_probe = self.layer_indices != tuple(range(self.config.num_hidden_layers))
        self.vocab_size = self.config.vocab_size
        self.padded_vocab_size = ((self.vocab_size + 4 * 65536 - 1) // (4 * 65536)) * 4 * 65536
        self.local_vocab_size = self.padded_vocab_size // 4
        self.workspace = CollectiveWorkspace(mesh_device, self.policy)
        reader = CheckpointReader(self.checkpoint)

        def upload(value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, dim=None, memory=DRAM):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=memory,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
                if dim is None
                else ttnn.ShardTensorToMesh(mesh_device, dim=dim),
            )

        embedding = reader.tensor("model.language_model.embed_tokens.weight")
        # Only the entry boundary gathers the TP4 hidden shards, matching the
        # selected replicated residual. No full embedding replication is needed.
        self.embedding_weight = upload(
            embedding[None, None],
            dtype=getattr(ttnn, self.precision["weight_groups"]["embedding"]),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dim=3,
        )
        self.tied_embeddings = bool(getattr(self.hf_config, "tie_word_embeddings", False))
        if self.tied_embeddings:
            head = embedding
        else:
            head = reader.tensor("lm_head.weight")
        head = torch.nn.functional.pad(head.T.contiguous(), (0, self.padded_vocab_size - self.vocab_size))
        head_device = upload(head[None, None], dtype=getattr(ttnn, self.precision["weight_groups"]["lm_head"]), dim=3)
        del head, embedding
        banks = mesh_device.dram_grid_size().x
        width = ((self.local_vocab_size + 32 * banks - 1) // (32 * banks)) * 32
        head_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                [self.config.hidden_size, width],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        self.lm_head_weight = ttnn.to_memory_config(head_device, head_memory)
        ttnn.deallocate(head_device)
        self.lm_head_input_memory = ttnn.create_sharded_memory_config(
            (32, self.config.hidden_size // 8),
            ttnn.CoreGrid(x=8, y=1),
            ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        self.lm_head_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            # BF16 doubles weight-CB bytes; the BFP8 block4 geometry overlaps
            # the persistent L1 sampler buffers. Block2 keeps that control legal.
            in0_block_w=2 if self.precision["weight_groups"]["lm_head"] == "bfloat16" else 4,
            per_core_M=1,
            per_core_N=width // 32,
            num_workers_per_dram_bank=2,
            use_multicore_pipeline=True,
        )
        self.lm_head_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.precision["compute_fidelities"]["lm_head"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.norm_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, self.precision["compute_fidelities"]["norm_recurrence"]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        gamma = (1.0 + reader.tensor("model.language_model.norm.weight").float()).reshape(1, 1, 1, -1)
        self.final_norm_weight = upload(gamma, dtype=getattr(ttnn, self.precision["weight_groups"]["norm"]))
        self.layers = []
        for index in self.layer_indices:
            state_dict = reader.layer(index)
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    state_dict,
                    hf_config=self.config,
                    layer_idx=index,
                    mesh_device=mesh_device,
                    policy=self.policy,
                    collective_workspace=self.workspace,
                    precision=self.precision,
                )
            )
            del state_dict
            print(f"ORNITH_LAYER_READY {index}", flush=True)
        self.layers = tuple(self.layers)

        # The same canonical full-width interleaved tables serve every full
        # attention layer. Decode does device lookup at its persistent position.
        self.rope_context = self.config.max_position_embeddings if rope_context is None else rope_context
        if not 1 <= self.rope_context <= self.config.max_position_embeddings:
            raise ValueError("RoPE table extent must lie inside the HF context contract")
        inv_freq = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2).float() / 64))
        phase = torch.arange(self.rope_context).float()[:, None] * inv_freq[None, :]
        self.rotary_tables = {}
        for name, values, identity in (("cos", phase.cos(), 1.0), ("sin", phase.sin(), 0.0)):
            expanded = torch.cat(
                [values.repeat_interleave(2, dim=-1), torch.full((self.rope_context, 192), identity)], dim=-1
            ).bfloat16()
            self.rotary_tables[name] = upload(expanded[None, None], layout=ttnn.ROW_MAJOR_LAYOUT)
        self.memory_after_weights = self.memory_view()
        return self

    def memory_view(self):
        """Measured allocator view; call after setup and again after cache setup."""
        result = {}
        fields = (
            "num_banks",
            "total_bytes_per_bank",
            "total_bytes_allocated_per_bank",
            "total_bytes_free_per_bank",
            "largest_contiguous_bytes_free_per_bank",
        )
        for name, kind in (("dram", ttnn.BufferType.DRAM), ("l1", ttnn.BufferType.L1)):
            view = ttnn.get_memory_view(self.device, kind)
            result[name] = {field: int(getattr(view, field)) for field in fields}
            result[name]["total_free_bytes"] = int(view.num_banks * view.total_bytes_free_per_bank)
        return result

    def _pool_blocks(self, *, extent, batch_size, page_size, shared_cache, num_blocks):
        """Resolve only the owned default shared pool; explicit budgets are exact."""
        if num_blocks is not None:
            if not isinstance(num_blocks, int) or num_blocks < 1:
                raise ValueError("num_blocks must be a positive physical KV page budget")
            return num_blocks
        requested = batch_size * extent // page_size
        if (
            shared_cache
            and self.layer_indices == tuple(range(40))
            and self.policy == replace(MeshPolicy(), ccl_dtype=self.precision["ccl_dtype"])
        ):
            return min(requested, measured_kv_capacity(self.precision)["physical_num_blocks"])
        return requested

    def create_state(self, *, context, batch_size=1, page_size=32, shared_cache=None, num_blocks=None):
        """Allocate logical context independently of an optional aggregate KV budget.

        ``num_blocks`` sizes each shared full-attention layer's physical pool;
        request-local caches use that many blocks per slot. The scheduler owns
        admission and page assignment when the pool is smaller than B*context.
        The selected all40 default shared pool is capped by the measured
        physical budget; explicit num_blocks and external storage are unchanged.
        This does not reduce the checkpoint's per-request context capability.
        """
        if not 1 <= batch_size <= 32 or not 1 <= context <= self.config.max_position_embeddings:
            raise ValueError("State requires 1..32 slots within the HF context contract")
        if page_size != 32:
            raise ValueError("The selected paged-cache policy requires page_size=32")
        if shared_cache is None:
            shared_cache = batch_size > 1
        extent = ((context + 255) // 256) * 256
        pool_blocks = self._pool_blocks(
            extent=extent, batch_size=batch_size, page_size=page_size, shared_cache=shared_cache, num_blocks=num_blocks
        )
        if shared_cache and pool_blocks < batch_size:
            raise ValueError("Shared KV pool must hold at least one page per fixed slot")
        layers = []
        for layer in self.layers:
            if shared_cache and layer.kind == "full_attention":
                shape = (pool_blocks, 1, page_size, 256)
                pools = [
                    ttnn.zeros(
                        shape, dtype=getattr(ttnn, layer.KV_CACHE_DTYPE), layout=ttnn.TILE_LAYOUT, device=self.device
                    )
                    for _ in range(2)
                ]
                layers.append(tuple(DecoderState(key=pools[0], value=pools[1]) for _ in range(batch_size)))
            elif num_blocks is not None and layer.kind == "full_attention":
                # Match the selected request-local constructor's BFP8 cache
                # policy; only physical page count differs from logical context.
                shape = (num_blocks, 1, page_size, 256)
                layers.append(
                    tuple(
                        DecoderState(
                            key=ttnn.zeros(
                                shape,
                                dtype=getattr(ttnn, layer.KV_CACHE_DTYPE),
                                layout=ttnn.TILE_LAYOUT,
                                device=self.device,
                            ),
                            value=ttnn.zeros(
                                shape,
                                dtype=getattr(ttnn, layer.KV_CACHE_DTYPE),
                                layout=ttnn.TILE_LAYOUT,
                                device=self.device,
                            ),
                        )
                        for _ in range(batch_size)
                    )
                )
            else:
                layers.append(tuple(layer.create_state(context=extent, page_size=page_size) for _ in range(batch_size)))
        result = ModelState(tuple(layers), context, batch_size, page_size, shared_cache)
        self.prepare_reset_state(result)
        return result

    def bind_page_tables(self, state, page_tables_host, *, valid_page_counts=None):
        """Prepare shared-cache plans over one stable mapping allocation.

        Call before generator warmup/capture. Every full-attention layer has its
        own K/V pools, but all consume the same mapping backing storage. Later
        scheduler changes copy to ``state.page_table`` without replacing it.
        The host mapping must assign disjoint physical pages to fixed slots.
        """
        self.validate_state(state)
        if not state.shared_cache:
            raise ValueError("bind_page_tables requires shared-cache state")
        if state.shared_cache_plans is not None:
            raise ValueError("Page bindings are immutable; refresh the existing page_table tensor")
        if valid_page_counts is not None:
            import torch

            if page_tables_host.dtype != torch.int32 or page_tables_host.ndim != 2:
                raise ValueError("Page mappings require a two-dimensional INT32 host tensor")
            if page_tables_host.shape[0] != state.batch_size or len(valid_page_counts) != state.batch_size:
                raise ValueError("One allocated-page count is required per fixed slot")
            if any(not 0 <= count <= page_tables_host.shape[1] for count in valid_page_counts):
                raise ValueError("Allocated page extent exceeds logical page-table width")
            used = torch.cat([page_tables_host[i, :count] for i, count in enumerate(valid_page_counts)])
            if torch.unique(used).numel() != used.numel():
                raise ValueError("Allocated shared KV page prefixes must be disjoint")
        plans = []
        first = None
        for index, layer in enumerate(self.layers):
            if layer.kind != "full_attention":
                plans.append(None)
                continue
            rows = state.layers[index]
            if first is None:
                if valid_page_counts is None:
                    first = layer.prepare_shared_cache_batch(states=rows, page_tables_host=page_tables_host)
                else:
                    signature = layer._shared_cache_pool_signature(rows[0].key, rows[0].value)
                    if any(layer._shared_cache_pool_signature(row.key, row.value) != signature for row in rows):
                        raise ValueError("Every slot must bind the same selected K/V pools")
                    if (page_tables_host < 0).any() or (page_tables_host >= rows[0].key.shape[0]).any():
                        raise ValueError("Page mapping exceeds the physical KV pool")
                    table = ttnn.from_torch(
                        page_tables_host.clone().contiguous(),
                        device=self.device,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    table_rows = tuple(table[i : i + 1, :] for i in range(state.batch_size))
                    first = SharedCacheBatchPlan(
                        rows[0].key, rows[0].value, table, table_rows, layer._shared_cache_owner
                    )
                plan = first
            else:
                signature = layer._shared_cache_pool_signature(rows[0].key, rows[0].value)
                if any(layer._shared_cache_pool_signature(row.key, row.value) != signature for row in rows):
                    raise ValueError("Every slot in a layer must bind the same K/V pools")
                if tuple(rows[0].key.shape) != tuple(first.key.shape):
                    raise ValueError("Shared-cache layers must use the same pool geometry")
                plan = SharedCacheBatchPlan(
                    rows[0].key,
                    rows[0].value,
                    first.page_table,
                    first.row_page_tables,
                    layer._shared_cache_owner,
                )
            plans.append(plan)
        if first is None:
            raise ValueError("No full-attention layer exists in this reduced model")
        state.shared_cache_plans = tuple(plans)
        state.page_table = first.page_table
        state.row_page_tables = first.row_page_tables
        return state.shared_cache_plans

    @staticmethod
    def _state_tensors(state, slots):
        seen = set()
        for rows in state.layers:
            for slot in slots:
                for name in ("conv", "recurrent", "key", "value"):
                    tensor = getattr(rows[slot], name)
                    if tensor is not None and tensor.buffer_address() not in seen:
                        seen.add(tensor.buffer_address())
                        yield name, tensor

    @staticmethod
    def _reset_signature(tensor):
        return (tuple(tensor.shape), str(tensor.dtype), str(tensor.layout), str(tensor.memory_config()))

    def prepare_reset_state(self, state):
        """Allocate one immutable zero donor per layout before any trace capture."""
        self.validate_state(state)
        if state.reset_donors is None:
            state.reset_donors = {}
        for _, tensor in self._state_tensors(state, range(state.batch_size)):
            key = self._reset_signature(tensor)
            if key not in state.reset_donors:
                state.reset_donors[key] = ttnn.zeros(
                    tensor.shape,
                    device=self.device,
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    memory_config=tensor.memory_config(),
                )

    def reset_state(self, state, *, slots=None, clear_kv=True):
        """Clear existing storage without host uploads or changing trace bindings.

        A partial reset of shared page pools must leave ``clear_kv=False`` and
        invalidate that request's positions/pages at the scheduler boundary;
        clearing an entire shared pool would overwrite other active requests.
        Recurrent and convolution state are always reset for selected slots.
        """
        self.validate_state(state)
        slots = tuple(range(state.batch_size) if slots is None else slots)
        if any(slot < 0 or slot >= state.batch_size for slot in slots):
            raise ValueError("Reset slot outside the configured request state")
        if clear_kv and state.shared_cache and set(slots) != set(range(state.batch_size)):
            raise ValueError("A partial shared-pool reset must preserve other requests' KV pages")
        if state.reset_donors is None:
            raise ValueError("External states require prepare_reset_state before capture")
        for name, tensor in self._state_tensors(state, slots):
            if clear_kv or name in ("conv", "recurrent"):
                ttnn.copy(state.reset_donors[self._reset_signature(tensor)], tensor)

    def validate_state(self, state):
        if not isinstance(state, ModelState) or len(state.layers) != len(self.layers):
            raise ValueError("Expected one state row for every loaded decoder layer")
        if not 1 <= state.batch_size <= 32 or any(len(rows) != state.batch_size for rows in state.layers):
            raise ValueError("All layers must expose the same 1..32 fixed request slots")

    def embed(self, tokens):
        """Replicated token IDs [1,1,1,N] -> selected BF16 residual layout."""
        value = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        value = ttnn.reshape(value, (1, 1, tokens.shape[-1], self.config.hidden_size // 4))
        return self.layers[0]._gather(value) if self.policy.residual_layout == "replicated" else value

    def rotary(self, positions):
        """Device lookup of absolute positions; no host RoPE rebuilding."""
        ids = positions if positions.dtype == ttnn.uint32 else ttnn.typecast(positions, ttnn.uint32)
        return {
            name: ttnn.reshape(
                ttnn.embedding(ids, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
                (1, 1, ids.shape[-1], 256),
            )
            for name, table in self.rotary_tables.items()
        }

    def decode_rotary(self, position):
        """One slot's current position -> fused decoder's two-head RoPE tile."""
        if position.shape[-1] != 1:
            raise ValueError("decode_rotary expects a single persistent position")
        return {name: ttnn.repeat(value, (1, 2, 32, 1)) for name, value in self.rotary(position).items()}

    def final_norm(self, hidden):
        if self.policy.residual_layout == "sharded":
            hidden = self.layers[0]._gather(ttnn.to_memory_config(hidden, DRAM))
        hidden = ttnn.to_memory_config(hidden, DRAM)
        normalized = ttnn.rms_norm(
            ttnn.typecast(hidden, ttnn.float32),
            weight=self.final_norm_weight,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.norm_compute,
            memory_config=DRAM,
        )
        return ttnn.typecast(normalized, ttnn.bfloat16)

    def logits(self, hidden):
        """Project 1..32 rows to local 65536-token shards; caller masks padding."""
        if not 1 <= hidden.shape[-2] <= 32:
            raise ValueError("Project terminal logits in physical row tiles of at most 32")
        normalized = self.final_norm(hidden)
        result = ttnn.linear(
            ttnn.to_memory_config(normalized, self.lm_head_input_memory),
            self.lm_head_weight,
            dtype=getattr(ttnn, self.precision["logits_sampling"]["logits_dtype"]),
            compute_kernel_config=self.lm_head_compute,
            program_config=self.lm_head_program,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        return ttnn.to_memory_config(result, DRAM)

    def terminal(self, hidden):
        """Generator-facing name for final norm and vocabulary-sharded logits."""
        return self.logits(hidden)

    def prefill_chunk(self, tokens, *, state, slot, plan_by_kind, return_hidden=False, last_token_only=False):
        """One prepared exact-length chunk; generator owns chunking and masking.

        Plans are prepared before the first capture and shared by layers of the
        same kind. The linear recurrent state is never advanced over padding.
        ``return_hidden`` lets the caller project only needed terminal tiles.
        """
        self.validate_state(state)
        if not 0 <= slot < state.batch_size:
            raise ValueError("Prefill slot is outside the configured request state")
        hidden = self.embed(tokens)
        for index, layer in enumerate(self.layers):
            hidden = layer.prefill_forward(hidden, state=state.layers[index][slot], plan=plan_by_kind[layer.kind])
            if hidden.dtype != getattr(ttnn, self.precision["residual_dtype"]):
                hidden = ttnn.typecast(hidden, getattr(ttnn, self.precision["residual_dtype"]))
        if return_hidden:
            return hidden
        if last_token_only:
            hidden = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)[:, :, -1:, :]
            hidden = ttnn.to_layout(hidden, ttnn.TILE_LAYOUT)
        return self.logits(hidden)

    def decode_forward(
        self,
        tokens,
        *,
        state,
        operands,
        active_slots=None,
        shared_cache_plans=None,
        prefill=False,
        return_hidden=False,
        state_write_indices=None,
    ):
        """Device-only decode over fixed slot bindings prepared by the generator.

        ``operands`` follows active-slot order and explicitly supplies current
        positions, page tables and RoPE. Inactive slots are not touched. A changed
        active-slot binding selects a prepared execution signature; it is not a
        reason to retire any existing trace. ``prefill`` is the accurate scalar
        continuation path and cannot silently substitute for ordinary decode.
        """
        self.validate_state(state)
        slots = tuple(range(state.batch_size) if active_slots is None else active_slots)
        if not slots or len(set(slots)) != len(slots) or any(slot < 0 or slot >= state.batch_size for slot in slots):
            raise ValueError("Active slots must be distinct configured request slots")
        if len(operands) != len(slots):
            raise ValueError("One explicit operand set is required per active slot")
        if prefill and len(slots) != 1:
            raise ValueError("Accurate scalar prefill must use one request slot")
        if state_write_indices is not None and (prefill or len(slots) < 2 or len(state_write_indices) != len(slots)):
            raise ValueError("Conditional state writes require explicit indices for batched decode")
        if shared_cache_plans is None and not prefill:
            shared_cache_plans = state.shared_cache_plans
        if state.shared_cache and shared_cache_plans is None and not prefill:
            raise ValueError("Shared-pool batch decode requires bind_page_tables during setup")
        if shared_cache_plans is not None and slots != tuple(range(state.batch_size)):
            raise ValueError("Shared plans preserve fixed slots; use device active masks for inactive rows")
        if tokens.shape[-1] < max(slots) + 1:
            raise ValueError("Token tensor must include every selected fixed slot")
        full_operands = []
        for operand in operands:
            values = {name: operand[name] for name in ("page_table", "current_pos", "cos", "sin") if name in operand}
            if "cos" not in values or "sin" not in values:
                values.update(self.decode_rotary(operand.get("rope_index", operand["current_pos"])))
            full_operands.append(values)
        batch_active_mask = (
            ttnn.concat([operand["active_mask"] for operand in operands], dim=2)
            if len(slots) > 1 and all("active_mask" in operand for operand in operands)
            else None
        )
        hidden = self.embed(tokens)
        rows = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        inputs = [ttnn.to_layout(rows[:, :, slot : slot + 1, :], ttnn.TILE_LAYOUT) for slot in slots]
        for index, layer in enumerate(self.layers):
            layer_states = [state.layers[index][slot] for slot in slots]
            kw = full_operands if layer.kind == "full_attention" else [{} for _ in slots]
            saved = []
            if layer.kind == "linear_attention" and state_write_indices is None:
                for layer_state, operand in zip(layer_states, operands, strict=True):
                    if "active_mask" in operand:
                        saved.append(
                            (
                                layer_state,
                                operand["active_mask"],
                                ttnn.clone(layer_state.conv),
                                ttnn.clone(layer_state.recurrent),
                            )
                        )
            if prefill:
                inputs = [layer.decode_forward(inputs[0], state=layer_states[0], prefill=True, **kw[0])]
            else:
                shared_plan = None if shared_cache_plans is None else shared_cache_plans[index]
                if shared_plan is not None:
                    kw = [{name: value for name, value in values.items() if name != "page_table"} for values in kw]
                inputs = layer.decode_batch(
                    inputs,
                    states=layer_states,
                    operands=kw,
                    shared_cache_plan=shared_plan,
                    **(
                        {"state_write_indices": state_write_indices}
                        if layer.kind == "linear_attention" and state_write_indices is not None
                        else {"batch_active_mask": batch_active_mask}
                        if layer.kind == "full_attention" and batch_active_mask is not None
                        else {}
                    ),
                )
            residual_dtype = getattr(ttnn, self.precision["residual_dtype"])
            inputs = [
                value if value.dtype == residual_dtype else ttnn.typecast(value, residual_dtype) for value in inputs
            ]
            # The fixed graph still computes inactive rows. Restore their
            # linear state on device; paged attention skips current_pos=-1.
            # Metadata presence selects a prepared graph, never a host read of
            # the dynamic mask. The all-active graph can omit these operations.
            for layer_state, active, previous_conv, previous_recurrent in saved:
                conv = ttnn.where(
                    active,
                    ttnn.to_layout(layer_state.conv, ttnn.TILE_LAYOUT),
                    ttnn.to_layout(previous_conv, ttnn.TILE_LAYOUT),
                )
                ttnn.copy(ttnn.to_layout(conv, layer_state.conv.layout), layer_state.conv)
                ttnn.copy(ttnn.where(active, layer_state.recurrent, previous_recurrent), layer_state.recurrent)
        hidden = inputs[0] if len(inputs) == 1 else ttnn.concat(inputs, dim=2)
        return hidden if return_hidden else self.logits(hidden)
