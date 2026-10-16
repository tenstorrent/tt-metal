# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B single-rank chunked-prefill runtime.

Satisfies the common/prefill runtime contract (ADDING_A_PREFILL_MODEL.md §2):
build model -> allocate KV cache -> compile -> prefill_chunk.

Mirrors GPT-OSS ``tt_prefill_runtime.py`` with Llama-specific differences:
  * Dense SwiGLU MLP (no MoE — no use_ep_moe / expert_weight_dtype knobs).
  * Llama3 RoPE frequency scaling (not YaRN).
  * n_kv_local = 2 (not 1): each TP column holds 2 KV heads (8 KV heads / 4 cols).
  * ``gather_layer`` reads back 2 local heads per TP column.
  * ``build_kv_chunk_table`` uses the Llama-specific multi-config builder that handles n_kv_local > 1.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions

from .attention.kv_cache import LlamaKVCache, allocate_kv_cache
from .rope import build_indexed_rope


def resolve_chunk_sizes(default_chunk_size: int, additional_chunk_sizes: tuple, max_seq_len: int) -> tuple:
    sizes = tuple(sorted({default_chunk_size, *additional_chunk_sizes}, reverse=True))
    for cs in sizes:
        if max_seq_len % cs != 0:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) must be a multiple of every supported chunk size; "
                f"{cs} does not divide it (supported: {sizes})"
            )
    return sizes


@dataclass
class TtPrefillRuntimeConfig:
    num_layers: int
    max_seq_len: int
    mesh_shape: tuple = (8, 4)  # (SP rows, TP cols) on the Blackhole galaxy
    default_chunk_size: int = 5120
    additional_chunk_sizes: tuple = ()
    num_users: int = 1
    sp_axis: int = 0
    tp_axis: int = 1
    topology: ttnn.Topology = ttnn.Topology.Ring
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[Path] = None
    owns_kv_cache: bool = True
    is_first_rank: bool = True
    is_last_rank: bool = True
    first_layer_idx: int = 0

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtPrefillRuntime:
    """Single-rank Llama 3.1 8B prefill lifecycle."""

    def __init__(self, mesh_device, hf_config, state_dict: dict, config: TtPrefillRuntimeConfig):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config

        self.chunk_sizes = resolve_chunk_sizes(
            config.default_chunk_size, config.additional_chunk_sizes, config.max_seq_len
        )
        self.max_chunk_size = self.chunk_sizes[0]
        assert config.topology in (ttnn.Topology.Ring, ttnn.Topology.Linear)

        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False
        self.kv_cache = None
        self._on_layer_complete = None
        self._layer_completion_sink = None
        self._prefill_request_id = -1

        self._build_model(state_dict)
        if config.owns_kv_cache:
            self._allocate_kv_cache()
        self._build_indexed_rope()

    def _build_model(self, state_dict: dict) -> None:
        from models.demos.llama_3_1_8b.tt.config import MeshConfig
        from models.demos.llama_3_1_8b.utils.general_utils import get_default_num_links

        from .ccl import CCLManager
        from .model import Model

        rows, cols = self.config.mesh_shape
        logger.info(
            f"Building Llama 3.1 8B TtPrefillRuntime model: num_layers={self.config.num_layers} "
            f"max_seq_len={self.config.max_seq_len} chunk_sizes={self.chunk_sizes} "
            f"num_users={self.config.num_users} mesh_shape={self.config.mesh_shape}"
        )
        mesh_config = MeshConfig((rows, cols), tp=cols)
        ccl = CCLManager(
            self.mesh_device, num_links=get_default_num_links(self.mesh_device), topology=self.config.topology
        )
        self.model = Model(
            mesh_device=self.mesh_device,
            hf_config=self.hf_config,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            tensor_cache_path=self.config.weight_cache_path,
            max_local_batch_size=1,
            max_seq_len=self.config.max_seq_len,
            sequence_parallel=True,
        )
        self.model_built = True

    def _allocate_kv_cache(self) -> None:
        self.kv_cache = allocate_kv_cache(
            self.mesh_device,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            sp_axis=self.config.sp_axis,
            num_users=self.config.num_users,
            head_dim=self.hf_config.head_dim,
            cache_dtype=self.config.cache_dtype,
        )
        self.kv_cache_allocated = True

    def _build_indexed_rope(self) -> None:
        rs = getattr(self.hf_config, "rope_scaling", None)
        if rs is None:
            rs_dict = {}
        elif isinstance(rs, dict):
            rs_dict = rs
        else:
            rs_dict = {
                "factor": getattr(rs, "factor", 8.0),
                "low_freq_factor": getattr(rs, "low_freq_factor", 1.0),
                "high_freq_factor": getattr(rs, "high_freq_factor", 4.0),
                "original_max_position_embeddings": getattr(rs, "original_max_position_embeddings", 8192),
            }
        self.rope_indexed = {
            cs: build_indexed_rope(
                self.mesh_device,
                head_dim=self.hf_config.head_dim,
                max_seq_len=self.config.max_seq_len,
                chunk_size=cs,
                sp_axis=self.config.sp_axis,
                rope_theta=getattr(self.hf_config, "rope_theta", 500000.0),
                factor=rs_dict.get("factor", 8.0),
                low_freq_factor=rs_dict.get("low_freq_factor", 1.0),
                high_freq_factor=rs_dict.get("high_freq_factor", 4.0),
                orig_max_pos=rs_dict.get("original_max_position_embeddings", 8192),
            )
            for cs in self.chunk_sizes
        }

    def _resolve_kv(self, kv_caches) -> LlamaKVCache:
        if kv_caches is None:
            assert self.kv_cache is not None, "runtime has no KV cache (owns_kv_cache=False): pass kv_caches"
            return self.kv_cache
        if isinstance(kv_caches, LlamaKVCache):
            return kv_caches
        return kv_caches[0]

    def make_chunk_input(self, token_ids: list, chunk_size: Optional[int] = None) -> ttnn.Tensor:
        chunk_size = chunk_size if chunk_size is not None else self.config.default_chunk_size
        assert chunk_size in self.rope_indexed, f"chunk_size={chunk_size} not in supported {tuple(self.rope_indexed)}"
        sp = self.config.sp_factor
        s_local = chunk_size // sp
        if not self.config.is_first_rank:
            emb = self.hf_config.hidden_size
            return ttnn.from_torch(
                torch.zeros(1, 1, s_local, emb),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        assert (
            len(token_ids) == chunk_size
        ), f"chunk input must be exactly chunk_size={chunk_size} tokens (pad the tail), got {len(token_ids)}"
        tok = torch.tensor(token_ids, dtype=torch.int32).reshape(sp, 1, s_local)
        return ttnn.from_torch(
            tok,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=self.config.mesh_shape, dims=(self.config.sp_axis, None)
            ),
        )

    def _embed_tokens(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.embedding(tokens, self.model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)
        return x

    def compile(self, kv_caches=None) -> None:
        assert self.model_built
        for chunk in self.chunk_sizes:
            ring = self.config.max_seq_len > chunk
            logger.info(
                f"Llama TtPrefillRuntime.compile() — warming up "
                f"{'2 cache-backed ring chunks' if ring else 'one all-gather fallback chunk'} of {chunk} tokens"
            )
            self.prefill_chunk(
                self.make_chunk_input([0] * chunk, chunk),
                kv_caches,
                slot_id=0,
                actual_start=0,
                actual_end=chunk,
                chunk_size=chunk,
            )
            if ring:
                self.prefill_chunk(
                    self.make_chunk_input([0] * chunk, chunk),
                    kv_caches,
                    slot_id=0,
                    actual_start=chunk,
                    actual_end=2 * chunk,
                    chunk_size=chunk,
                )
        ttnn.synchronize_device(self.mesh_device)
        self.compiled = True

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_caches=None,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        skip_lm_head: bool = True,
        get_last_token: int = -1,
        chunk_size: Optional[int] = None,
        request_id: int = -1,
        d2h_service=None,
        record_dev=None,
    ) -> Optional[ttnn.Tensor]:
        if d2h_service is not None:
            raise NotImplementedError("Llama prefill emits layer acks via set_layer_ack_channel, not the D2H path")
        assert self.model_built, "build the model before prefill_chunk()"
        chunk_size = chunk_size if chunk_size is not None else self.config.default_chunk_size
        assert chunk_size in self.rope_indexed
        kv = self._resolve_kv(kv_caches)
        assert 0 <= slot_id < self.config.num_users
        assert actual_start + chunk_size <= self.config.max_seq_len
        assert actual_start < actual_end <= actual_start + chunk_size

        self._prefill_request_id = request_id

        if self.config.is_first_rank:
            x_embd = self._embed_tokens(input_tensor)
            ttnn.deallocate(input_tensor)
        else:
            x_embd = input_tensor

        out = self.model.prefill_forward(
            x_embd,
            rot_mats_global=self.rope_indexed[chunk_size],
            kv_cache=kv,
            cached_len=actual_start,
            user_id=slot_id,
            get_last_token=get_last_token,
            skip_lm_head=skip_lm_head,
            indexed_rope=True,
            on_layer_complete=self._on_layer_complete,
        )
        if not self.config.is_last_rank:
            return out
        if skip_lm_head:
            if out is not None:
                out.deallocate(True)
            return None
        return out

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        assert self.compiled, "Call compile() before set_layer_ack_channel()"

        def on_layer_complete(layer_idx: int) -> None:
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete

    def set_layer_completion_sink(self, sink) -> None:
        assert self.compiled, "Call compile() before set_layer_completion_sink()"
        self._layer_completion_sink = sink
        first = self.config.first_layer_idx

        def on_layer_complete(layer_idx: int) -> None:
            sink(first + layer_idx, self._prefill_request_id)

        self._on_layer_complete = on_layer_complete

    def kv_migration_base_address(self, kv_caches) -> int:
        return int(self._resolve_kv(kv_caches).k.buffer_address())

    def kv_migration_stages(self, kv_cache, first_layer_idx=None, num_my_layers=None):
        from dataclasses import dataclass

        @dataclass
        class KvCacheStage:
            buffer_address: int
            first_layer_idx: int
            num_my_layers: int

        kv = self._resolve_kv(kv_cache)
        fli = first_layer_idx if first_layer_idx is not None else self.config.first_layer_idx
        nml = num_my_layers if num_my_layers is not None else self.config.num_layers
        return [
            KvCacheStage(buffer_address=int(kv.k.buffer_address()), first_layer_idx=fli, num_my_layers=nml),
            KvCacheStage(buffer_address=int(kv.v.buffer_address()), first_layer_idx=fli, num_my_layers=nml),
        ]

    def build_kv_chunk_table(
        self,
        kv_caches,
        path: str,
        *,
        first_layer_idx: int = 0,
        num_my_layers: Optional[int] = None,
        stage_layout=None,
    ) -> str:
        del first_layer_idx, num_my_layers, stage_layout
        from models.demos.llama_3_1_8b.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        kv = self._resolve_kv(kv_caches)
        c = self.config
        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            kv_cache=kv,
            seq_len=c.max_seq_len,
            num_layers=c.num_layers,
            mesh_shape=c.mesh_shape,
            sp_axis=c.sp_axis,
            num_users=c.num_users,
            chunk_size=c.default_chunk_size,
            num_kv_heads=self.hf_config.num_key_value_heads,
            head_dim=self.hf_config.head_dim,
            path=path,
        )

    def read_slot_kv(self, kv_caches, slot: int):
        kv = self._resolve_kv(kv_caches)
        mesh_device = self.mesh_device
        num_layers = self.config.num_layers

        def _block(tensor):
            s = list(tensor.shape)
            sl = ttnn.slice(
                tensor,
                [slot * num_layers, 0, 0, 0],
                [(slot + 1) * num_layers, s[1], s[2], s[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # n_kv_local=2: ConcatMesh2d dims=(2,1) -> SP on dim2, TP on dim1
            # Resulting shape: [num_layers, n_kv_local*cols, seq_local*sp, head_dim]
            # = [num_layers, 8, seq_local*sp, head_dim] = [num_layers, num_kv_heads, seq_cache, head_dim]
            block = ttnn.to_torch(
                sl, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
            ).float()
            ttnn.deallocate(sl)
            return block

        return [_block(kv.k), _block(kv.v)]

    def gather_layer(self, slot_id: int, layer_idx: int, n_tokens: int, kv_caches=None, chunk_size=None):
        """Read one layer's device K/V cache back to NATURAL token order. Returns (k, v) torch tensors
        [1, num_kv_heads, n_tokens, head_dim]. Handles n_kv_local=2 (2 heads per TP column)."""
        kv = self._resolve_kv(kv_caches)
        sp = self.config.sp_factor
        cols = self.config.tp_factor
        nkv = self.hf_config.num_key_value_heads
        n_kv_local = nkv // cols  # 2
        slot = slot_id * self.config.num_layers + layer_idx
        chunk_size = chunk_size if chunk_size is not None else self.config.default_chunk_size
        p = blockcyclic_positions(sp, chunk_size, self.config.max_seq_len)

        def gather(cache_tensor, col, local_head):
            dts = ttnn.get_device_tensors(cache_tensor)
            # Each device tensor has per-chip shape [B, n_kv_local, seq_local, head_dim].
            # Extract slot's data, then the specific local_head.
            dev = torch.cat([ttnn.to_torch(dts[r * cols + col])[slot, local_head].float() for r in range(sp)], dim=0)
            nat = torch.empty_like(dev)
            nat[p] = dev
            return nat[:n_tokens]

        heads_k = []
        heads_v = []
        for h in range(nkv):
            col = h // n_kv_local
            lh = h % n_kv_local
            heads_k.append(gather(kv.k, col, lh))
            heads_v.append(gather(kv.v, col, lh))
        k = torch.stack(heads_k, dim=0).unsqueeze(0)
        v = torch.stack(heads_v, dim=0).unsqueeze(0)
        return k, v

    def kv_cache_pcc_check(
        self,
        kv_caches=None,
        *,
        slot_id: int,
        n_chunks: int,
        trace_dir=None,
        first_layer_idx: int = 0,
        chunk_size=None,
        real_len=None,
        pt_path_override=None,
    ) -> float:
        from safetensors import safe_open

        from models.common.utility_functions import comp_pcc

        if pt_path_override is not None:
            raise NotImplementedError("Llama kv_cache_pcc_check has no per-slot .pt golden; use PREFILL_TRACE_DIR")
        from models.demos.common.prefill.runners.runner_utils import resolve_trace_dir

        raw_trace = trace_dir or os.environ.get("PREFILL_TRACE_DIR")
        assert raw_trace, "kv_cache_pcc_check needs PREFILL_TRACE_DIR or trace_dir="
        trace_dir = resolve_trace_dir(raw_trace)
        token_ids = list(json.load(open(Path(trace_dir) / "metadata.json"))["token_ids"])
        chunk_size = chunk_size if chunk_size is not None else self.config.default_chunk_size
        n_tokens = min(len(token_ids), n_chunks * chunk_size)
        if real_len is not None:
            n_tokens = min(n_tokens, int(real_len))
        assert n_tokens > 0

        head_dim = self.hf_config.head_dim
        rotary_dim = getattr(self.hf_config, "rotary_dim", head_dim)
        half = rotary_dim // 2
        # HF half-split -> Meta interleaved permutation
        src = list(range(head_dim))
        for m in range(rotary_dim):
            src[m] = half * (m % 2) + (m // 2)
        src = torch.tensor(src, dtype=torch.long)

        kv_dir = Path(trace_dir) / "kv_cache"
        logger.info(
            f"[kv-pcc] per-layer K / V vs golden ({trace_dir}) over [0,{n_tokens}) ({self.config.num_layers} layers):"
        )
        min_k, min_v = 1.0, 1.0
        for L in range(self.config.num_layers):
            gL = first_layer_idx + L
            dev_k, dev_v = self.gather_layer(
                slot_id=slot_id, layer_idx=L, n_tokens=n_tokens, kv_caches=kv_caches, chunk_size=chunk_size
            )
            with safe_open(str(kv_dir / f"layer_{gL}.safetensors"), framework="pt") as h:
                g_k = h.get_tensor(f"key_cache_layer_{gL}").float()[:, :, :n_tokens, :][..., src]
                g_v = h.get_tensor(f"value_cache_layer_{gL}").float()[:, :, :n_tokens, :]
            pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
            pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
            min_k, min_v = min(min_k, pcc_k), min(min_v, pcc_v)
            logger.info(f"  layer {gL:>2}: K={pcc_k:.5f} V={pcc_v:.5f}")
        logger.info(f"[kv-pcc] min PCC across {self.config.num_layers} layers: K={min_k:.5f} V={min_v:.5f}")
        return min(min_k, min_v)

    def kv_cache_pcc_per_layer(
        self,
        kv_caches=None,
        *,
        slot_id: int,
        n_chunks: int,
        trace_dir=None,
        first_layer_idx: int = 0,
        chunk_size=None,
        real_len=None,
    ) -> list:
        from safetensors import safe_open

        from models.common.utility_functions import comp_pcc
        from models.demos.common.prefill.runners.runner_utils import resolve_trace_dir

        raw_trace = trace_dir or os.environ.get("PREFILL_TRACE_DIR")
        assert raw_trace, "kv_cache_pcc_per_layer needs PREFILL_TRACE_DIR or trace_dir="
        trace_dir = resolve_trace_dir(raw_trace)
        token_ids = list(json.load(open(Path(trace_dir) / "metadata.json"))["token_ids"])
        chunk_size = chunk_size if chunk_size is not None else self.config.default_chunk_size
        n_tokens = min(len(token_ids), n_chunks * chunk_size)
        if real_len is not None:
            n_tokens = min(n_tokens, int(real_len))
        assert n_tokens > 0

        head_dim = self.hf_config.head_dim
        rotary_dim = getattr(self.hf_config, "rotary_dim", head_dim)
        half = rotary_dim // 2
        src = list(range(head_dim))
        for m in range(rotary_dim):
            src[m] = half * (m % 2) + (m // 2)
        src = torch.tensor(src, dtype=torch.long)

        kv_dir = Path(trace_dir) / "kv_cache"
        results = []
        for L in range(self.config.num_layers):
            gL = first_layer_idx + L
            dev_k, dev_v = self.gather_layer(
                slot_id=slot_id, layer_idx=L, n_tokens=n_tokens, kv_caches=kv_caches, chunk_size=chunk_size
            )
            with safe_open(str(kv_dir / f"layer_{gL}.safetensors"), framework="pt") as h:
                g_k = h.get_tensor(f"key_cache_layer_{gL}").float()[:, :, :n_tokens, :][..., src]
                g_v = h.get_tensor(f"value_cache_layer_{gL}").float()[:, :, :n_tokens, :]
            pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
            pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
            results.append({"layer": gL, "k": round(pcc_k, 5), "v": round(pcc_v, 5)})
        return results
