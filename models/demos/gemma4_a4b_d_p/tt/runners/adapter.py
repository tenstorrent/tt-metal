# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""``Gemma4A4BPrefillAdapter``: the common/prefill engine <-> Gemma-4 26B-A4B (text) boundary.

Single rank on a 1x4 mesh (sp = 1, tp = 4), FABRIC_2D. The KV cache the engine owns holds:
  * ``contract``: the migratable prefill-server cache (tt/runners/kv_contract.py: bf8, one 512-wide slab per chip for
    K and for V, [users * layers, 1, seq, 512], DRAM round-robin 32-token blocks, configs K chip 0..3, V chip 0..3)
  * ``attn[slot]``: per-user, per-layer bf16 attention caches the device attention reads its prefix from

The runtime takes the engine's device input as is (uint32 ROW_MAJOR [sp, 1, chunk / sp], tail padded past
actual_end) and runs the whole forward on the device (tt/model.py). The layer-completion sink fires after the layer's
KV writes have completed on the device (event sync per layer).

Import-light: all device / model imports happen inside the methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams


class Gemma4A4BConfig:
    """Static model-dimension constants (the prefill engine's ``model_config``)."""

    NUM_LAYERS = 30
    EMB_SIZE = 2816
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
    NUM_ATTENTION_HEADS = 16
    NUM_KEY_VALUE_HEADS = 8  # sliding layers
    HEAD_DIM = 256  # sliding layers
    NUM_GLOBAL_KEY_VALUE_HEADS = 2
    GLOBAL_HEAD_DIM = 512
    SLIDING_WINDOW = 1024
    GLOBAL_LAYERS = (5, 11, 17, 23, 29)
    KV_SLAB_WIDTH = 512  # per-chip K (and V) width per token in the migratable cache, both layer types
    INTERMEDIATE_SIZE = 2112
    MOE_INTERMEDIATE_SIZE = 704
    NUM_ROUTED_EXPERTS = 128
    NUM_EXPERTS_PER_TOKEN = 8
    VOCAB_SIZE = 262144
    HF_ID = "google/gemma-4-26B-A4B-it"


def resolve_model_path() -> str:
    """PREFILL_HF_MODEL, else the bring-up spec's checkpoint (BRINGUP_SPEC paths.hf), else the local HF hub cache."""
    env = os.environ.get("PREFILL_HF_MODEL")
    if env:
        return env
    if os.environ.get("BRINGUP_SPEC"):
        from models.demos.common.bringup.reference.golden import hf_path, load_spec

        return hf_path(load_spec())
    from huggingface_hub import snapshot_download

    return snapshot_download(Gemma4A4BConfig.HF_ID, local_files_only=True)


@dataclass
class Gemma4KvCaches(KvCaches):
    contract: object  # kv_contract.Gemma4ContractKV
    attn: list = field(default_factory=list)  # per slot: {layer: TtKVCacheSliding | TtKVCacheGlobal}


class Gemma4PrefillRuntime:
    """Structural runtime contract of ADDING_A_PREFILL_MODEL.md section 2 (stateless w.r.t. the KV cache)."""

    def __init__(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.gemma4_a4b_d_p.tt.model import TtGemma4Model

        assert params.is_first_rank and params.is_last_rank and params.first_layer_idx == 0, "single-rank only"
        assert tuple(params.mesh_shape) == (1, 4), f"Gemma-4 prefill is built for a 1x4 mesh, got {params.mesh_shape}"
        assert params.chunk_size % 64 == 0 and params.max_seq_len % params.chunk_size == 0
        self.mesh_device, self.config = mesh_device, params
        self.model = TtGemma4Model(
            mesh_device, resolve_model_path(), max_seq=params.max_seq_len, max_chunk=params.chunk_size
        )
        assert len(self.model.layer_ids) == params.num_layers
        self._sink = None

    def set_layer_completion_sink(self, sink) -> None:
        self._sink = sink

    def make_chunk_input(self, token_ids):
        """The engine's H2D layout: uint32 ROW_MAJOR [sp, 1, chunk / sp], tail padded with 0xFFFFFFFF."""
        import torch

        import ttnn

        c, sp = self.config.chunk_size, self.config.mesh_shape[0]
        ids = list(token_ids) + [0xFFFFFFFF] * (c - len(token_ids))
        t = torch.tensor(ids, dtype=torch.int64).reshape(sp, 1, c // sp).to(torch.uint32)
        return ttnn.from_torch(
            t,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(
                self.mesh_device,
                ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()]),
            ),
        )

    def _run(self, ids, kv_cache, slot, start, request_id, acks: bool):
        import ttnn

        sink = self._sink if acks else None
        mesh = self.mesh_device

        def on_layer(i, request_id=request_id):
            if sink is None:
                return
            # The ack promises the layer's KV is in the migratable cache: wait until the device has finished it.
            ttnn.event_synchronize(ttnn.record_event(mesh, 0))
            sink(i, request_id)  # global layer index (single rank: local == global)

        contract = kv_cache.contract
        h = self.model.prefill_chunk(
            ids,
            start,
            kv_cache.attn[slot],
            on_layer=on_layer,
            kv_sink_of=lambda i: contract.sink(i, start, slot),
        )
        ttnn.deallocate(h)

    def compile(self, kv_cache: Gemma4KvCaches) -> None:
        """Warm every chunk-offset program once in slot 0 (junk KV there; real requests overwrite it)."""
        import ttnn

        c = self.config.chunk_size
        for start in range(0, self.config.max_seq_len, c):
            ids = self.make_chunk_input([0] * c)
            self._run(ids, kv_cache, 0, start, 0, acks=False)
            ttnn.deallocate(ids)
        ttnn.synchronize_device(self.mesh_device)

    def prefill_chunk(
        self,
        input_tensor,
        kv_cache,
        *,
        slot_id,
        actual_start,
        actual_end,
        request_id=0,
        d2h_service=None,
        metadata_msg=None,
    ):
        c = self.config.chunk_size
        assert actual_start % 64 == 0, "chunk write offset must be 64-aligned (global cache block)"
        assert actual_start < actual_end <= actual_start + c, (actual_start, actual_end, c)
        assert actual_start + c <= self.config.max_seq_len, "chunk overruns the user slot"
        assert tuple(input_tensor.shape)[-1] * self.config.mesh_shape[0] == c, input_tensor.shape
        # Rows >= actual_end are pad (engine fills 0xFFFFFFFF): the embedding masks them in range, and being after
        # every real row they never reach a real position through causal attention.
        self._run(input_tensor, kv_cache, slot_id, actual_start, request_id, acks=True)
        return None  # last/single rank: the populated cache is the output

    # --- migration hooks ---
    def build_kv_chunk_table(self, kv_cache, path: str) -> str:
        import ttnn

        table = kv_cache.contract.address_table(seq_len=self.config.max_seq_len, chunk_size=self.config.chunk_size)
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
        return path

    def kv_migration_base_address(self, kv_cache) -> int:
        return int(kv_cache.contract.cache.k.buffer_address())


class Gemma4A4BPrefillAdapter(PrefillModelAdapter):
    """Gemma-4 26B-A4B-it prefill adapter (25 sliding + 5 global attention layers, dense MLP + 128-expert top-8 MoE)."""

    name = "gemma4_a4b_d_p"
    model_config = Gemma4A4BConfig
    hf_model_default = ""  # resolve_model_path(); PREFILL_HF_MODEL overrides
    ttnn_cache_default = ""  # tt/experts.py CACHE_ROOT (generated/gemma4_a4b_d_p/tt_cache)
    prefill_trace_default = ""  # bring-up golden dir; PREFILL_TRACE_DIR overrides
    default_gate_mode = "DEVICE_FP32"
    l1_small_size = 24576
    hf_repo_id = Gemma4A4BConfig.HF_ID

    def load_hf_config(self):
        from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import Gemma4TextConfig

        return Gemma4TextConfig.from_json(os.path.join(resolve_model_path(), "config.json"))

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        sp, tp = mesh_shape
        path = Path(env_cache) / f"{self.name}_bh_{int(sp) * int(tp)}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        from models.demos.gemma4_a4b_d_p.tt.model import new_attention_caches
        from models.demos.gemma4_a4b_d_p.tt.runners.kv_contract import Gemma4ContractKV

        assert tuple(params.mesh_shape) == (1, 4), f"Gemma-4 prefill is built for a 1x4 mesh, got {params.mesh_shape}"
        layers = list(range(params.first_layer_idx, params.first_layer_idx + params.num_layers))
        contract = Gemma4ContractKV(
            mesh_device, num_layers=params.num_layers, max_seq=params.max_seq_len, num_users=params.num_users
        )
        attn = [
            new_attention_caches(mesh_device, hf_config, layers, params.max_seq_len) for _ in range(params.num_users)
        ]
        return Gemma4KvCaches(contract=contract, attn=attn)

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        return Gemma4PrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, params=params)

    # Read-back of this layout (per-chip 512-wide K/V slabs, two layer types). The common producer has no branch for
    # it yet; see BREADCRUMBS (K.1): it needs `if ADAPTER.name == "gemma4_a4b_d_p": return
    # ADAPTER.read_slot_kv_and_check_pcc(...)` in prefill_producer._read_slot_kv_and_check_pcc.
    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir, num_layers):
        from models.demos.gemma4_a4b_d_p.tt.runners.kv_contract import read_slot_kv_and_check_pcc

        return read_slot_kv_and_check_pcc(
            table, device_map, slot_id, real_len, trace_dir, num_layers, self.load_hf_config()
        )
