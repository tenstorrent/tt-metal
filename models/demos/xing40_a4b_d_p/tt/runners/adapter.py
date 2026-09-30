# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""``XingPrefillAdapter``: the common/prefill engine <-> Xing4.0-29B-A4B (text decoder, MTP layer 40 skipped).

Single rank on a 4x2 Blackhole mesh, FABRIC_2D, SP = 4 over axis 0 x TP = 2 over axis 1 (bringup/plan.md). The
model is the all-device tt/model.py:TtXingModel (the ladder's model, final norm unused). The KV cache the engine
owns (tt/runners/kv_contract.py) is the model's own state: every slot's MLA latent cache, which the attention writes
and gathers in place through ``TtMlaAttention.bind_cache``.

Served layers: ``PREFILL_XING_LAYERS`` (e.g. "0-5"), else the bring-up spec's subset (BRINGUP_SPEC, or this model's
bringup/spec.yaml), intersected with the rank's range; a contiguous run from the rank's first layer. The runtime acks
exactly those layers, with their global index.

Engine input: uint32 ROW_MAJOR [sp, 1, chunk / sp] with sp = mesh rows = 4, sharded over mesh axis 0 (row r holds
tokens [r chunk/4, (r+1) chunk/4)), replicated over the columns, tail padded with 0xFFFFFFFF past actual_end. That is
the embedding's input split (tt/model.py:TtEmbedding, contiguous rows over axis 0), so the runtime only clamps the pad
ids into the vocab (``ttnn.minimum``) and reshapes to [1, 1, 1, chunk/4], on the device. Pad tokens sit after every
real token, so causality keeps them out of every real row's KV (and MoE routing is per token). The layer-completion
sink fires after an event sync per layer (the layer's KV is then in the engine cache).

Import-light: all device / model imports happen inside the methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams

MODEL_NAME = "xing40_a4b_d_p"
MESH_SHAPE = (4, 2)
SPEC_PATH = Path(__file__).resolve().parents[2] / "bringup" / "spec.yaml"


class Xing40Config:
    """Static model-dimension constants (the prefill engine's ``model_config``)."""

    NUM_LAYERS = 40
    EMB_SIZE = 3584
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
    NUM_ATTENTION_HEADS = 32
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    QK_NOPE_HEAD_DIM = 128
    V_HEAD_DIM = 128
    HC_MULT = 4
    NUM_ROUTED_EXPERTS = 64
    NUM_EXPERTS_PER_TOKEN = 4
    VOCAB_SIZE = 131072
    HF_ID = "XingChen-AGI/Xing4.0-29B-A4B"


def bringup_spec():
    """The bring-up spec: BRINGUP_SPEC when it is this model's, else this model's own spec.yaml."""
    from models.demos.common.bringup.reference.golden import load_spec

    if os.environ.get("BRINGUP_SPEC"):
        s = load_spec()
        if s.model == MODEL_NAME:
            return s
    return load_spec(str(SPEC_PATH))


def resolve_model_path() -> str:
    """PREFILL_HF_MODEL, else the bring-up spec's checkpoint (paths.hf)."""
    env = os.environ.get("PREFILL_HF_MODEL")
    if env:
        return env
    from models.demos.common.bringup.reference.golden import hf_path

    return hf_path(bringup_spec())


def served_layers(first_layer_idx: int, num_layers: int) -> list[int]:
    """Global indices of the layers this rank builds and acks (see module docstring)."""
    from models.demos.common.bringup.core.spec import parse_layers

    env = os.environ.get("PREFILL_XING_LAYERS")
    layers = parse_layers(env, Xing40Config.NUM_LAYERS) if env else bringup_spec().layers()
    layers = [i for i in layers if first_layer_idx <= i < first_layer_idx + num_layers]
    assert (
        layers == list(range(first_layer_idx, first_layer_idx + len(layers))) and layers
    ), f"served layers {layers} must be a non-empty contiguous run from the rank's first layer {first_layer_idx}"
    return layers


@dataclass
class XingKvCaches(KvCaches):
    contract: object  # kv_contract.XingContractKV
    layers: list = field(default_factory=list)


class XingPrefillRuntime:
    """Structural runtime contract of ADDING_A_PREFILL_MODEL.md section 2 (stateless w.r.t. the KV cache)."""

    def __init__(self, *, mesh_device, hf_config, params: PrefillRunParams):
        import ttnn
        from models.demos.xing40_a4b_d_p.tt.model import TtXingModel

        assert params.is_first_rank and params.is_last_rank, "single-rank only"
        assert tuple(params.mesh_shape) == MESH_SHAPE, f"built for a 4x2 mesh, got {params.mesh_shape}"
        sp = params.mesh_shape[0]
        assert params.chunk_size % (32 * sp) == 0 and params.max_seq_len % params.chunk_size == 0
        self.mesh_device, self.config = mesh_device, params
        self.layers = served_layers(params.first_layer_idx, params.num_layers)
        c = params.chunk_size
        self.model = TtXingModel(mesh_device, resolve_model_path(), c // sp, c, layers=self.layers)
        self.vocab = int(self.model.cfg.vocab_size)
        # Geometry (RoPE tables, scratch, SDPA config, dispatch sizes) for (chunk, max_seq), built once; the KV lives
        # in the engine's cache (bind_cache per chunk), the modules' own caches are unused.
        self.model.setup(c, params.max_seq_len)
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
        """Engine input (per chip [1, 1, chunk/4], row r's quarter) -> the embedding's [1, 1, 1, chunk/4] uint32
        ROW_MAJOR with pad ids clamped to V - 1 (real ids are < V and pass unchanged). On the device."""
        ttnn = self._ttnn
        n = ids.shape[-1]
        t = ttnn.to_layout(ids, ttnn.TILE_LAYOUT)
        m = ttnn.minimum(t, self.vocab - 1)
        ttnn.deallocate(t)
        rm = ttnn.to_layout(m, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(m)
        return ttnn.reshape(rm, (1, 1, 1, n))

    def _bind(self, kv, slot: int) -> None:
        c = kv.contract
        for blk in self.model.blocks:
            blk.attn.bind_cache(c.kvpe, slot, c.kv_row(blk.i), len(c.layers))

    def _run(self, ids, kv_cache, slot, start, request_id, acks: bool):
        ttnn = self._ttnn
        sink = self._sink if acks else None
        mesh = self.mesh_device
        self._bind(kv_cache, slot)
        clean = self._device_ids(ids)
        h = self.model.embed(clean)
        ttnn.deallocate(clean)
        for blk in self.model.blocks:
            h2 = blk(h, start)
            ttnn.deallocate(h)
            h = h2
            if sink is not None:
                # The ack promises the layer's KV is in the engine cache: wait until the device has finished it.
                ttnn.event_synchronize(ttnn.record_event(mesh, 0))
                sink(blk.i, request_id)
        ttnn.deallocate(h)

    def compile(self, kv_cache: XingKvCaches) -> None:
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
        assert actual_start % c == 0, "chunk write offset must be chunk-aligned (block-cyclic cache write)"
        assert actual_start < actual_end <= actual_start + c, (actual_start, actual_end, c)
        assert actual_start + c <= self.config.max_seq_len, "chunk overruns the user slot"
        assert 0 <= slot_id < kv_cache.contract.num_users, (slot_id, kv_cache.contract.num_users)
        assert tuple(input_tensor.shape)[-1] * self.config.mesh_shape[0] == c, input_tensor.shape
        self._run(input_tensor, kv_cache, slot_id, actual_start, request_id, acks=True)
        return None  # last/single rank: the populated cache is the output

    # --- migration hooks ---
    def build_kv_chunk_table(self, kv_cache, path: str) -> str:
        import ttnn

        table = kv_cache.contract.address_table(seq_len=self.config.max_seq_len)
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
        return path

    def kv_migration_base_address(self, kv_cache) -> int:
        return int(kv_cache.contract.kvpe.buffer_address())


class XingPrefillAdapter(PrefillModelAdapter):
    """Xing4.0-29B-A4B prefill adapter on a 4x2 mesh (dense causal MLA, mHC 4-stream residual, dense SwiGLU /
    64-expert top-4 MoE with a shared expert, EP over the mesh columns' 4-chip dispatch groups)."""

    name = MODEL_NAME
    model_config = Xing40Config
    hf_model_default = ""  # resolve_model_path(); PREFILL_HF_MODEL overrides
    ttnn_cache_default = ""  # tt/experts.py CACHE_ROOT
    prefill_trace_default = ""  # bring-up golden dir; PREFILL_TRACE_DIR overrides
    default_gate_mode = "DEVICE_FP32"
    l1_small_size = 24576
    hf_repo_id = Xing40Config.HF_ID

    def load_hf_config(self):
        from models.demos.xing40_a4b_d_p.reference.xing_ref import XingConfig

        return XingConfig.from_json(os.path.join(resolve_model_path(), "config.json"))

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        sp, tp = mesh_shape
        path = Path(env_cache) / f"{self.name}_bh_{int(sp) * int(tp)}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def num_kv_cache_layers(self, num_layers: int) -> int:
        """Acks the producer should expect per chunk: one per served layer (every layer writes kv_latent)."""
        return len(served_layers(0, num_layers))

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        from models.demos.xing40_a4b_d_p.tt.runners.kv_contract import XingContractKV

        assert tuple(params.mesh_shape) == MESH_SHAPE, f"built for a 4x2 mesh, got {params.mesh_shape}"
        layers = served_layers(params.first_layer_idx, params.num_layers)
        contract = XingContractKV(
            mesh_device, layers, max_seq=params.max_seq_len, chunk=params.chunk_size, num_users=params.num_users
        )
        return XingKvCaches(contract=contract, layers=layers)

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        return XingPrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, params=params)

    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir, num_layers):
        """Read-back through the table vs the bring-up golden: {kv_latent: min PCC}."""
        from models.demos.xing40_a4b_d_p.tt.runners.kv_contract import read_slot_kv_and_check_pcc

        return read_slot_kv_and_check_pcc(
            table, device_map, slot_id, real_len, trace_dir, served_layers(0, int(num_layers))
        )
