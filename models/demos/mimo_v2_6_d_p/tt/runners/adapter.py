# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""``MiMoPrefillAdapter``: the common/prefill engine <-> MiMo-V2.6-Flash-RL (text decoder) boundary.

Single rank on a 1x4 mesh (sp = 1, tp = 4), FABRIC_2D. The KV cache the engine owns holds:
  * ``contract``: the migratable prefill-server cache (tt/runners/kv_contract.py: bf8, one 384-wide slab per chip for
    K and for V, [users * layers, 1, seq, 384], DRAM round-robin 32-token blocks, configs K chip 0..3, V chip 0..3)
  * ``attn[slot]``: per-user, per-layer bf16 attention caches the device attention reads its prefix from

Served layers: the bring-up covers a layer subset (spec ``layers``, "0-5"); the full 48-layer model does not fit the
box. ``PREFILL_MIMO_LAYERS`` (e.g. "0-5") picks the layers, else the bring-up spec (BRINGUP_SPEC) when it is this
model's, else every layer of the rank. The subset must start at the rank's first layer and be contiguous; the runtime
acks exactly those layers (``num_kv_cache_layers`` tells the producer how many acks to expect).

The runtime takes the engine's device input as is (uint32 ROW_MAJOR [sp, 1, chunk / sp], tail padded with 0xFFFFFFFF
past actual_end), clamps the ids into the vocab on the device (``ttnn.minimum``; the vocab 152576 is not a power of
two), and runs the whole forward on the device (tt/model.py). The layer-completion sink fires after the layer's KV
writes have completed on the device (event sync per layer).

Import-light: all device / model imports happen inside the methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams

MODEL_NAME = "mimo_v2_6_d_p"


class MiMoV26Config:
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
    KV_SLAB_WIDTH = 384  # per-chip K (and V) width per token in the migratable cache, both layer types
    INTERMEDIATE_SIZE = 16384
    MOE_INTERMEDIATE_SIZE = 2048
    NUM_ROUTED_EXPERTS = 256
    NUM_EXPERTS_PER_TOKEN = 8
    VOCAB_SIZE = 152576
    HF_ID = "XiaomiMiMo/MiMo-V2.6-Flash-RL"


def _bringup_spec():
    """The bring-up spec when BRINGUP_SPEC names this model, else None."""
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

    return snapshot_download(MiMoV26Config.HF_ID, local_files_only=True)


def served_layers(first_layer_idx: int, num_layers: int) -> list[int]:
    """Global indices of the layers this rank builds and acks (see module docstring)."""
    from models.demos.common.bringup.core.spec import parse_layers

    env = os.environ.get("PREFILL_MIMO_LAYERS")
    s = _bringup_spec()
    if env:
        layers = parse_layers(env, MiMoV26Config.NUM_LAYERS)
    elif s is not None:
        layers = s.layers()
    else:
        layers = list(range(first_layer_idx, first_layer_idx + num_layers))
    layers = [i for i in layers if first_layer_idx <= i < first_layer_idx + num_layers]
    assert (
        layers == list(range(first_layer_idx, first_layer_idx + len(layers))) and layers
    ), f"served layers {layers} must be a non-empty contiguous run from the rank's first layer {first_layer_idx}"
    if len(layers) < num_layers:
        from loguru import logger

        logger.warning(f"[mimo] serving layers {layers[0]}-{layers[-1]} of the rank's {num_layers} (layer subset)")
    return layers


@dataclass
class MiMoKvCaches(KvCaches):
    contract: object  # kv_contract.MiMoContractKV
    layers: list = field(default_factory=list)  # served global layer ids (contract cache layer = i - layers[0])
    attn: list = field(default_factory=list)  # per slot: {layer: TtKVCacheFull | TtKVCacheSliding}


class MiMoPrefillRuntime:
    """Structural runtime contract of ADDING_A_PREFILL_MODEL.md section 2 (stateless w.r.t. the KV cache)."""

    def __init__(self, *, mesh_device, hf_config, params: PrefillRunParams):
        import ttnn
        from models.demos.mimo_v2_6_d_p.tt.model import TtMiMoModel

        assert params.is_first_rank and params.is_last_rank and params.first_layer_idx == 0, "single-rank only"
        assert tuple(params.mesh_shape) == (1, 4), f"MiMo prefill is built for a 1x4 mesh, got {params.mesh_shape}"
        assert params.chunk_size % 64 == 0 and params.max_seq_len % params.chunk_size == 0
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

    def _clamp_ids(self, ids):
        """Engine pad ids (0xFFFFFFFF) -> V - 1, on the device. Real ids are < V and pass unchanged; pad rows come
        after every real row, so under causal attention they never reach a real position."""
        ttnn = self._ttnn
        t = ttnn.to_layout(ids, ttnn.TILE_LAYOUT)
        m = ttnn.minimum(t, self.vocab - 1)
        ttnn.deallocate(t)
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
            sink(i, request_id)  # TtMiMoBlock.i is the global layer index

        contract = kv_cache.contract
        clean = self._clamp_ids(ids)
        h = self.model.prefill_chunk(
            clean,
            start,
            kv_cache.attn[slot],
            on_layer=on_layer,
            kv_sink_of=lambda i: contract.sink(i - first, start, slot),
        )
        ttnn.deallocate(clean)
        ttnn.deallocate(h)

    def compile(self, kv_cache: MiMoKvCaches) -> None:
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
        assert actual_start % 64 == 0, "chunk write offset must be 64-aligned (full-attention cache block)"
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
        return int(kv_cache.contract.cache.k.buffer_address())


class MiMoPrefillAdapter(PrefillModelAdapter):
    """MiMo-V2.6-Flash-RL prefill adapter (full / sliding-window attention with sinks, dense MLP + 256-expert top-8
    MoE with a noaux_tc sigmoid router)."""

    name = MODEL_NAME
    model_config = MiMoV26Config
    hf_model_default = ""  # resolve_model_path(); PREFILL_HF_MODEL overrides
    ttnn_cache_default = ""  # tt/experts.py CACHE_ROOT (generated/mimo_v2_6_d_p/tt_cache)
    prefill_trace_default = ""  # bring-up golden dir; PREFILL_TRACE_DIR overrides
    default_gate_mode = "DEVICE_FP32"
    l1_small_size = 24576
    hf_repo_id = MiMoV26Config.HF_ID

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
        from models.demos.mimo_v2_6_d_p.tt.model import new_attention_caches
        from models.demos.mimo_v2_6_d_p.tt.runners.kv_contract import MiMoContractKV

        assert tuple(params.mesh_shape) == (1, 4), f"MiMo prefill is built for a 1x4 mesh, got {params.mesh_shape}"
        layers = served_layers(params.first_layer_idx, params.num_layers)
        contract = MiMoContractKV(
            mesh_device, num_layers=len(layers), max_seq=params.max_seq_len, num_users=params.num_users
        )
        attn = [
            new_attention_caches(mesh_device, hf_config, layers, params.max_seq_len) for _ in range(params.num_users)
        ]
        return MiMoKvCaches(contract=contract, layers=layers, attn=attn)

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        return MiMoPrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, params=params)

    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir, num_layers):
        """Read-back of this layout (per-chip 384-wide K/V slabs, two layer types) over the served layers."""
        from models.demos.mimo_v2_6_d_p.tt.runners.kv_contract import read_slot_kv_and_check_pcc

        n = min(int(num_layers), len(served_layers(0, int(num_layers))))
        return read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir, n, self.load_hf_config())
