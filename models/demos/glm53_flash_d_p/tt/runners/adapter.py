# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""``Glm53FlashPrefillAdapter``: the common/prefill engine <-> GLM-5.3-Flash (text decoder) on a 2x2 mesh.

Single rank on a 2x2 Blackhole mesh, FABRIC_2D; the model is tt/model.py:TtGlmModel (KDA layers: TP 2 on axis 1,
SP 2 on axis 0; DSA layers: replicated latent / pooled-key caches, queries split by sequence; MoE over 2 dispatch
groups). The engine-owned cache (``GlmKvCaches``) holds:
  * ``contract``: the migratable copy of every served DSA layer's latent and pooled keys (tt/runners/kv_contract.py)
  * ``slots[slot][layer]``: the per-slot state the blocks read and write, bound before each chunk
    (DSA: the latent cache and the pooled-key cache; KDA: the recurrent / conv carries)

Served layers: ``PREFILL_GLM_LAYERS`` (e.g. "0-4"), else the bring-up spec (BRINGUP_SPEC) when it is this model's,
else every layer of the rank; a contiguous run from the rank's first layer. Never the MTP layer (45).

Acks: only the DSA layers own KV, so the runtime acks one per served DSA layer per chunk, with the layer's KV-slot
index (the address table's layer; ``kv_slot_layer_ids`` maps it to the global layer, ``acks_in_kv_slot_space``
tells the runner so), after the layer's contract copy is on the device (event sync).

Engine input: uint32 ROW_MAJOR [sp, 1, chunk / sp] with sp = mesh rows = 2, sharded over axis 0, replicated over the
columns, tail 0xFFFFFFFF past actual_end. On the device: ``ttnn.minimum`` clamps the pad ids into the vocab
(V = 154880 is not a power of two), all_gather over axis 0 gives every chip the whole chunk. actual_end (32-aligned)
goes to every KDA layer so the carries stop at the valid end; the DSA caches past it are never read for real rows.

Import-light: all device / model imports happen inside the methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams

MODEL_NAME = "glm53_flash_d_p"
MESH_SHAPE = (2, 2)
PAD_ID = 0xFFFFFFFF


class Glm53FlashConfig:
    """Static model-dimension constants (the prefill engine's ``model_config``)."""

    NUM_LAYERS = 45  # text decoder; the MTP layer (45) is not served
    EMB_SIZE = 4096
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
    NUM_ATTENTION_HEADS = 64
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 0
    INDEX_HEAD_DIM = 128
    INDEX_KPOOL = 4
    LINEAR_NUM_HEADS = 64
    LINEAR_HEAD_DIM = 128
    NUM_ROUTED_EXPERTS = 288
    NUM_EXPERTS_PER_TOKEN = 8
    VOCAB_SIZE = 154880
    HF_ID = "zai-org/GLM-5.3-Flash"


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

    return snapshot_download(Glm53FlashConfig.HF_ID, local_files_only=True)


def served_layers(first_layer_idx: int, num_layers: int) -> list[int]:
    """Global indices of the layers this rank builds (see module docstring)."""
    from models.demos.common.bringup.core.spec import parse_layers

    env = os.environ.get("PREFILL_GLM_LAYERS")
    s = _bringup_spec()
    if env:
        layers = parse_layers(env, Glm53FlashConfig.NUM_LAYERS)
    elif s is not None:
        layers = s.layers()
    else:
        layers = list(range(first_layer_idx, first_layer_idx + num_layers))
    layers = [
        i for i in layers if first_layer_idx <= i < first_layer_idx + num_layers and i < Glm53FlashConfig.NUM_LAYERS
    ]
    assert (
        layers == list(range(first_layer_idx, first_layer_idx + len(layers))) and layers
    ), f"served layers {layers} must be a non-empty contiguous run from the rank's first layer {first_layer_idx}"
    return layers


def _hf_config():
    from models.demos.glm53_flash_d_p.reference.glm_ref import GlmConfig

    return GlmConfig.from_json(os.path.join(resolve_model_path(), "config.json"))


def dsa_layers_of(cfg, layers) -> list[int]:
    """The served layers that own KV (DSA), in order: the KV-slot index k is dsa_layers_of(...)[k]."""
    return [i for i in layers if not cfg.is_kda(i)]


def new_block_state(mesh, cfg, layer: int, max_seq: int) -> dict:
    """One slot's state buffers for one block, with the shapes the block's modules allocate for themselves."""
    if cfg.is_kda(layer):
        from models.demos.glm53_flash_d_p.tt.kda_attention import kda_config, kda_state_zeros

        return {"kda": kda_state_zeros(mesh, kda_config(cfg))}
    from models.demos.glm53_flash_d_p.tt.indexer import pooled_key_cache
    from models.demos.glm53_flash_d_p.tt.mla_attention import latent_cache

    return {"kv_latent": latent_cache(mesh, cfg, max_seq), "index_key": pooled_key_cache(mesh, cfg, max_seq)}


@dataclass
class GlmKvCaches(KvCaches):
    contract: object  # kv_contract.GlmContractKV
    layers: list = field(default_factory=list)
    dsa_layers: list = field(default_factory=list)
    slots: list = field(default_factory=list)  # per slot: {layer: new_block_state(...)}


class Glm53FlashPrefillRuntime:
    """Structural runtime contract of ADDING_A_PREFILL_MODEL.md section 2 (stateless w.r.t. the KV cache)."""

    def __init__(self, *, mesh_device, hf_config, params: PrefillRunParams):
        import ttnn
        from models.demos.glm53_flash_d_p.tt.model import TtGlmModel

        assert params.is_first_rank and params.is_last_rank, "single-rank only"
        assert tuple(params.mesh_shape) == MESH_SHAPE, f"built for a 2x2 mesh, got {params.mesh_shape}"
        sp = params.mesh_shape[0]
        assert params.chunk_size % (128 * sp) == 0 and params.max_seq_len % params.chunk_size == 0, (
            params.chunk_size,
            params.max_seq_len,
        )
        self.mesh_device, self.config = mesh_device, params
        self.layers = served_layers(params.first_layer_idx, params.num_layers)
        self.model = TtGlmModel(
            mesh_device,
            resolve_model_path(),
            max_seq=params.max_seq_len,
            chunks=[params.chunk_size],
            layers=self.layers,
        )
        self.cfg = self.model.cfg
        self.dsa_layers = dsa_layers_of(self.cfg, self.layers)
        self.slot_of = {layer: k for k, layer in enumerate(self.dsa_layers)}
        self.blocks = {b.i: b for b in self.model.blocks}
        self.vocab = int(self.cfg.vocab_size)
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
        ids = list(token_ids) + [PAD_ID] * (c - len(token_ids))
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

    def _bind(self, kv_cache, slot: int) -> None:
        for layer in self.layers:
            self.blocks[layer].bind_state(kv_cache.slots[slot][layer])

    def _run(self, ids, kv_cache, slot, start, end, request_id, acks: bool):
        ttnn = self._ttnn
        sink = self._sink if acks else None
        mesh, c = self.mesh_device, self.config.chunk_size
        state = kv_cache.slots[slot]

        def on_layer(i, request_id=request_id):
            k = self.slot_of.get(i)
            if k is None:
                return  # KDA: fixed-size state only, no KV slab and no ack
            kv_cache.contract.write(slot, k, start, c, state[i]["kv_latent"], state[i]["index_key"])
            if sink is None:
                return
            # The ack promises the layer's KV is in the migratable cache: wait until the device has finished it.
            ttnn.event_synchronize(ttnn.record_event(mesh, 0))
            sink(k, request_id)

        self._bind(kv_cache, slot)
        clean = self._device_ids(ids)
        h = self.model.prefill_chunk(clean, start, on_layer=on_layer, end=end)
        ttnn.deallocate(clean)
        ttnn.deallocate(h)

    def compile(self, kv_cache: GlmKvCaches) -> None:
        """Warm every chunk-offset program once in slot 0 (junk state there; a request at 0 resets it)."""
        ttnn = self._ttnn
        assert kv_cache.layers == self.layers, (kv_cache.layers, self.layers)
        c = self.config.chunk_size
        for start in range(0, self.config.max_seq_len, c):
            ids = self.make_chunk_input([0] * c)
            self._run(ids, kv_cache, 0, start, start + c, 0, acks=False)
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
        assert actual_start % c == 0, "chunk write offset must be chunk-aligned"
        assert actual_start < actual_end <= actual_start + c, (actual_start, actual_end, c)
        assert actual_end % 32 == 0, f"actual_end {actual_end} must be 32-aligned (KDA's runtime end bound)"
        assert actual_start + c <= self.config.max_seq_len, "chunk overruns the user slot"
        assert tuple(input_tensor.shape)[-1] * self.config.mesh_shape[0] == c, input_tensor.shape
        self._run(input_tensor, kv_cache, slot_id, actual_start, actual_end, request_id, acks=True)
        return None  # last/single rank: the populated cache is the output

    # --- migration hooks ---
    def build_kv_chunk_table(self, kv_cache, path: str) -> str:
        import ttnn

        table = kv_cache.contract.address_table(seq_len=self.config.max_seq_len)
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
        return path

    def kv_migration_base_address(self, kv_cache) -> int:
        return kv_cache.contract.base_address()

    # --- fixed-size state (not in the table) ---
    def kda_state_torch(self, kv_cache, slot: int) -> dict:
        """{layer: {kda_recurrent, kda_conv}} of one slot in the reference layout (host read-back, test only)."""
        return {
            i: self.blocks[i].attn.state_torch(kv_cache.slots[slot][i]["kda"])
            for i in self.layers
            if self.cfg.is_kda(i)
        }


class Glm53FlashPrefillAdapter(PrefillModelAdapter):
    """GLM-5.3-Flash prefill adapter on a 2x2 mesh (KDA linear attention + DSA sparse MLA, mHC 4-stream residual,
    dense clamped SwiGLU / 288-expert top-8 MoE)."""

    name = MODEL_NAME
    model_config = Glm53FlashConfig
    hf_model_default = ""  # resolve_model_path(); PREFILL_HF_MODEL overrides
    ttnn_cache_default = ""  # tt/experts.py CACHE_ROOT
    prefill_trace_default = ""  # bring-up golden dir; PREFILL_TRACE_DIR overrides
    default_gate_mode = "DEVICE_FP32"
    l1_small_size = 24576
    hf_repo_id = Glm53FlashConfig.HF_ID
    acks_in_kv_slot_space = True  # the runtime acks with the KV-slot index (see prefill_runner)

    def load_hf_config(self):
        return _hf_config()

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        sp, tp = mesh_shape
        path = Path(env_cache) / f"{self.name}_bh_{int(sp) * int(tp)}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def kv_slot_layer_ids(self, num_layers: int) -> list[int]:
        """KV slot -> global layer: the served DSA layers (the only ones that own KV and ack)."""
        return dsa_layers_of(_hf_config(), served_layers(0, num_layers))

    def num_kv_cache_layers(self, num_layers: int) -> int:
        return len(self.kv_slot_layer_ids(num_layers))

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        from models.demos.glm53_flash_d_p.tt.runners.kv_contract import GlmContractKV

        assert tuple(params.mesh_shape) == MESH_SHAPE, f"built for a 2x2 mesh, got {params.mesh_shape}"
        layers = served_layers(params.first_layer_idx, params.num_layers)
        dsa = dsa_layers_of(hf_config, layers)
        assert dsa, f"served layers {layers} hold no DSA layer: nothing to migrate"
        contract = GlmContractKV(
            mesh_device, hf_config, num_dsa_layers=len(dsa), max_seq=params.max_seq_len, num_users=params.num_users
        )
        slots = [
            {i: new_block_state(mesh_device, hf_config, i, params.max_seq_len) for i in layers}
            for _ in range(params.num_users)
        ]
        return GlmKvCaches(contract=contract, layers=layers, dsa_layers=dsa, slots=slots)

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        return Glm53FlashPrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, params=params)

    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir, num_layers):
        """Read-back through the table (the DSA layers' latent and pooled keys) vs the golden; {name: min PCC}."""
        from models.demos.glm53_flash_d_p.tt.runners.kv_contract import read_slot_kv_and_check_pcc

        cfg = _hf_config()
        dsa = dsa_layers_of(cfg, served_layers(0, int(num_layers)))
        return read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir, dsa, cfg)


def contract_state_pcc(spec, runtime, kv, slot, length, golden) -> dict:
    """Bring-up contract hook (testing/contract.py): the slot's KDA carries after a request of ``length`` tokens vs the
    golden snapshot at ``length``; {kda_recurrent, kda_conv: min PCC over the served KDA layers}.
    The spec's hooks module re-exports this (bringup/hooks.py: contract_state_pcc)."""
    from loguru import logger

    from models.demos.glm53_flash_d_p.tt.runners.kv_contract import _pcc

    out = {}
    for layer, dev in runtime.kda_state_torch(kv, slot).items():
        gold = golden.state(layer, at=length)
        for name in ("kda_recurrent", "kda_conv"):
            p = _pcc(dev[name], gold[name].float().reshape(dev[name].shape))
            out[name] = min(out.get(name, 1.0), p)
            logger.info(f"  layer {layer:>2} {name} at {length}: PCC {p:.6f}")
    return out
