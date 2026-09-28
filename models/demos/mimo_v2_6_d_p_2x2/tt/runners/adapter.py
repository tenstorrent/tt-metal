# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""``MiMo2x2PrefillAdapter``: the common/prefill engine <-> MiMo-V2.6-Flash-RL (text decoder) on a 2x2 mesh.

Port of the 1x4 prior's adapter (models/demos/mimo_v2_6_d_p/tt/runners/adapter.py). Single rank on a 2x2 Blackhole
mesh, FABRIC_2D, TP = 4 over the flattened mesh (chip d = 2*row + col), no sequence parallelism inside the model.
The KV cache the engine owns holds:
  * ``contract``: the migratable cache (tt/runners/kv_contract.py: bf8, one 384-wide slab per chip for K and for V,
    [users * layers, 1, max_seq, 384], DRAM round-robin 32-token blocks; configs K chip 0..3, V chip 0..3)
  * ``attn[slot]``: per-user, per-layer bf16 attention caches the device attention reads its prefix from

Served layers: ``PREFILL_MIMO_LAYERS`` (e.g. "0-5"), else the bring-up spec (BRINGUP_SPEC) when it is this model's,
else every layer of the rank; a contiguous run from the rank's first layer. The runtime acks exactly those layers.

Engine input: uint32 ROW_MAJOR [sp, 1, chunk / sp] with sp = mesh rows = 2, sharded over mesh axis 0 (row r holds
tokens [r*chunk/2, (r+1)*chunk/2)), replicated over the columns, tail padded with 0xFFFFFFFF past actual_end. The
runtime clamps pad ids into the vocab (``ttnn.minimum``, V = 152576 is not a power of two) and all-gathers the halves
over axis 0 on the device, giving every chip the whole chunk [1, 1, chunk] the model embeds. The forward runs
entirely on the device (tt/model.py); the layer-completion sink fires after an event sync per layer.

Import-light: all device / model imports happen inside the methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams

MODEL_NAME = "mimo_v2_6_d_p_2x2"
MESH_SHAPE = (2, 2)


class MiMoV26Config2x2:
    """Static model-dimension constants (the prefill engine's ``model_config``)."""

    NUM_LAYERS = 48
    EMB_SIZE = 4096
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 4  # full layers
    SWA_NUM_KEY_VALUE_HEADS = 8  # sliding layers
    HEAD_DIM = 192
    V_HEAD_DIM = 128
    SLIDING_WINDOW = 128
    KV_SLAB_WIDTH = 384
    INTERMEDIATE_SIZE = 16384
    MOE_INTERMEDIATE_SIZE = 2048
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 8
    VOCAB_SIZE = 152576
    HF_ID = "XiaomiMiMo/MiMo-V2.6-Flash-RL"


def _bringup_spec():
    if not os.environ.get("BRINGUP_SPEC"):
        return None
    from models.demos.common.bringup.reference.golden import load_spec

    s = load_spec()
    return s if s.model == MODEL_NAME else None


def resolve_model_path() -> str:
    """PREFILL_HF_MODEL, else the bring-up spec's checkpoint (paths.hf), else the local HF hub cache."""
    env = os.environ.get("PREFILL_HF_MODEL")
    if env:
        return env
    s = _bringup_spec()
    if s is not None:
        from models.demos.common.bringup.reference.golden import hf_path

        return hf_path(s)
    from huggingface_hub import snapshot_download

    return snapshot_download(MiMoV26Config2x2.HF_ID, local_files_only=True)


def served_layers(first_layer_idx: int, num_layers: int) -> list[int]:
    """Global indices of the layers this rank builds and acks (see module docstring)."""
    from models.demos.common.bringup.core.spec import parse_layers

    env = os.environ.get("PREFILL_MIMO_LAYERS")
    s = _bringup_spec()
    if env:
        layers = parse_layers(env, MiMoV26Config2x2.NUM_LAYERS)
    elif s is not None:
        layers = s.layers()
    else:
        layers = list(range(first_layer_idx, first_layer_idx + num_layers))
    layers = [i for i in layers if first_layer_idx <= i < first_layer_idx + num_layers]
    assert (
        layers == list(range(first_layer_idx, first_layer_idx + len(layers))) and layers
    ), f"served layers {layers} must be a non-empty contiguous run from the rank's first layer {first_layer_idx}"
    return layers


@dataclass
class MiMo2x2KvCaches(KvCaches):
    contract: object  # kv_contract.MiMoContractKV2x2
    layers: list = field(default_factory=list)
    attn: list = field(default_factory=list)  # per slot: {layer: TtKVCacheFull | TtKVCacheSliding}


class MiMo2x2PrefillRuntime:
    """Structural runtime contract of ADDING_A_PREFILL_MODEL.md section 2 (stateless w.r.t. the KV cache)."""

    def __init__(self, *, mesh_device, hf_config, params: PrefillRunParams):
        import ttnn
        from models.demos.mimo_v2_6_d_p_2x2.tt.model import TtMiMoModel

        assert params.is_first_rank and params.is_last_rank and params.first_layer_idx == 0, "single-rank only"
        assert tuple(params.mesh_shape) == MESH_SHAPE, f"built for a 2x2 mesh, got {params.mesh_shape}"
        sp = params.mesh_shape[0]
        assert params.chunk_size % (64 * sp) == 0 and params.max_seq_len % params.chunk_size == 0
        self.mesh_device, self.config = mesh_device, params
        self.layers = served_layers(params.first_layer_idx, params.num_layers)
        self.model = TtMiMoModel(
            mesh_device,
            resolve_model_path(),
            max_seq=params.max_seq_len,
            max_chunk=params.chunk_size,
            layers=self.layers,
        )
        self.vocab = int(self.model.cfg.vocab_size)
        self._ttnn = ttnn
        self._sink = None

    def set_layer_completion_sink(self, sink) -> None:
        self._sink = sink

    def make_chunk_input(self, token_ids):
        """The engine's H2D layout: uint32 ROW_MAJOR [sp, 1, chunk / sp], sharded over axis 0, tail 0xFFFFFFFF.
        Host side; used only by compile() warm-up (the engine supplies real inputs)."""
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

    def _device_ids(self, ids):
        """Engine input (per chip [1, 1, chunk/2], row r's half) -> every chip [1, 1, chunk] uint32 ROW_MAJOR, pad ids
        clamped to V - 1 (real ids are < V and pass unchanged). All on the device."""
        ttnn = self._ttnn
        t = ttnn.to_layout(ids, ttnn.TILE_LAYOUT)
        m = ttnn.minimum(t, self.vocab - 1)
        ttnn.deallocate(t)
        if self.config.mesh_shape[0] > 1:
            g = ttnn.all_gather(m, dim=-1, cluster_axis=0)
            ttnn.deallocate(m)
            m = g
        out = ttnn.to_layout(m, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(m)
        return out

    def _run(self, ids, kv_cache, slot, start, request_id, acks: bool):
        ttnn = self._ttnn
        sink = self._sink if acks else None
        mesh = self.mesh_device
        first = kv_cache.layers[0]

        def on_layer(i, request_id=request_id):
            if sink is None:
                return
            # The ack promises the layer's KV is in the migratable cache: wait until the device has finished it.
            ttnn.event_synchronize(ttnn.record_event(mesh, 0))
            sink(i - first, request_id)

        contract = kv_cache.contract
        clean = self._device_ids(ids)
        h = self.model.prefill_chunk(
            clean,
            start,
            kv_cache.attn[slot],
            on_layer=on_layer,
            kv_sink_of=lambda i: contract.sink(i - first, start, slot),
        )
        ttnn.deallocate(clean)
        ttnn.deallocate(h)

    def compile(self, kv_cache: MiMo2x2KvCaches) -> None:
        """Warm every chunk-offset program once in slot 0 (junk KV there; real requests overwrite it)."""
        ttnn = self._ttnn
        assert kv_cache.layers == self.layers, (kv_cache.layers, self.layers)
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
        assert actual_start % c == 0, "chunk write offset must be chunk-aligned (contract cache write)"
        assert actual_start < actual_end <= actual_start + c, (actual_start, actual_end, c)
        assert actual_start + c <= self.config.max_seq_len, "chunk overruns the user slot"
        assert tuple(input_tensor.shape)[-1] * self.config.mesh_shape[0] == c, input_tensor.shape
        self._run(input_tensor, kv_cache, slot_id, actual_start, request_id, acks=True)
        return None  # last/single rank: the populated cache is the output

    # --- migration hooks ---
    def build_kv_chunk_table(self, kv_cache, path: str) -> str:
        import ttnn

        table = kv_cache.contract.address_table(seq_len=self.config.max_seq_len, chunk_size=self.config.chunk_size)
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
        return path

    def kv_migration_base_address(self, kv_cache) -> int:
        return int(kv_cache.contract.k.buffer_address())


class MiMo2x2PrefillAdapter(PrefillModelAdapter):
    """MiMo-V2.6-Flash-RL prefill adapter on a 2x2 mesh (full / sliding-window attention with sinks, dense MLP +
    256-expert top-8 MoE, EP over 2 dispatch groups)."""

    name = MODEL_NAME
    model_config = MiMoV26Config2x2
    hf_model_default = ""  # resolve_model_path(); PREFILL_HF_MODEL overrides
    ttnn_cache_default = ""  # tt/experts.py CACHE_ROOT
    prefill_trace_default = ""  # bring-up golden dir; PREFILL_TRACE_DIR overrides
    default_gate_mode = "DEVICE_FP32"
    l1_small_size = 24576
    hf_repo_id = MiMoV26Config2x2.HF_ID

    def load_hf_config(self):
        from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoConfig

        return MiMoConfig.from_json(os.path.join(resolve_model_path(), "config.json"))

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        sp, tp = mesh_shape
        path = Path(env_cache) / f"{self.name}_bh_{int(sp) * int(tp)}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def num_kv_cache_layers(self, num_layers: int) -> int:
        """Acks the producer should expect per chunk: one per served layer (every MiMo layer writes KV)."""
        return len(served_layers(0, num_layers))

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        from models.demos.mimo_v2_6_d_p_2x2.tt.model import new_attention_caches
        from models.demos.mimo_v2_6_d_p_2x2.tt.runners.kv_contract import MiMoContractKV2x2

        assert tuple(params.mesh_shape) == MESH_SHAPE, f"built for a 2x2 mesh, got {params.mesh_shape}"
        layers = served_layers(params.first_layer_idx, params.num_layers)
        contract = MiMoContractKV2x2(
            mesh_device, num_layers=len(layers), max_seq=params.max_seq_len, num_users=params.num_users
        )
        attn = [
            new_attention_caches(mesh_device, hf_config, layers, params.max_seq_len) for _ in range(params.num_users)
        ]
        return MiMo2x2KvCaches(contract=contract, layers=layers, attn=attn)

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        return MiMo2x2PrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, params=params)

    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir, num_layers):
        """Read-back through the table: same configs and slab columns as the 1x4 prior, so its reader applies."""
        from models.demos.mimo_v2_6_d_p.tt.runners.kv_contract import read_slot_kv_and_check_pcc

        n = min(int(num_layers), len(served_layers(0, int(num_layers))))
        return read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir, n, self.load_hf_config())
