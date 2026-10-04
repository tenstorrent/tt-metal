# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""``XingPrefillAdapter``: the common/prefill engine <-> Xing4.0-29B-A4B (text decoder, MTP layer 40 skipped).

Single rank on a 4x2 Blackhole mesh, FABRIC_2D, SP = 4 over axis 0 x TP = 2 over axis 1 (bringup/plan.md). The
model is the all-device tt/model.py:TtXingModel (the ladder's model, final norm unused). The KV cache the engine
owns (tt/runners/kv_contract.py) is the model's own state: every slot's MLA latent cache, which the attention writes
and gathers in place through ``TtMlaAttention.bind_cache``.

Served layers: every layer of the rank's range (num_kv_cache_layers(n) = n: every Xing layer writes KV);
``PREFILL_XING_LAYERS`` (e.g. "0-5") restricts it for debugging, a contiguous run from the rank's first layer. The
runtime acks exactly those layers, with their global index.

Engine input (bringup/serving_contract.md "Input"): uint32 ROW_MAJOR [sp, 1, chunk / sp] with sp = mesh rows = 4,
sharded over mesh axis 0, replicated over the columns, PAD_ID 0xFFFFFFFF past actual_end, already reshuffled by the
server (ring_sdpa_reshuffle with kv_offset = actual_start: absolute position g on row (g // 1280) % 4, rising within
a row). That is where the attention expects each token for a chunk at actual_start (update_padded_kv_cache,
rotary_embedding_indexed and ring_mla derive the placement from it on the device), so the runtime only clamps the pad
ids into the vocab and reshapes to [1, 1, 1, chunk/4], on the device. actual_start is any multiple of 32 (a follow-up
turn starts at the reused prefix); pad tokens sit after every real token, so causality keeps them out of every real
row's KV (MoE routing is per token).

Per layer: the attention writes only the records holding real tokens (valid_global = actual_end) and zeroes the pad
rows [actual_end, ceil32(actual_end)) of the last one; then the ack. Host sink (default): an event sync, then
``sink(global_layer, request_id)``. D2H (``d2h_service`` given, PREFILL_LAYER_ACK_D2H=1):
``outbound_socket_service_sync(d2h_service, metadata=metadata_msg)`` enqueued on the same CQ, no host sync.

Import-light: all device / model imports happen inside the methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.xing40_a4b_d_p.tt.settings import settings

MODEL_NAME = "xing40_a4b_d_p"
MESH_SHAPE = (4, 2)
SPEC_PATH = Path(__file__).resolve().parents[2] / "bringup" / "spec.yaml"


class Xing40Config:
    """Static model-dimension constants (the prefill engine's ``model_config``)."""

    NUM_LAYERS = 40
    EMB_SIZE = 3584
    # The runner opens the fabric with this packet payload (runner_utils.open_mesh_device). It must hold an fp32 tile
    # (4096 B: the mHC [S/4, 32] fp32 all_reduce), else reduce_scatter fits 0 pages per packet (SIGFPE); 4352 is the
    # fabric default every bring-up test ran with (tt_metal/fabric/erisc_datamover_builder.hpp).
    FABRIC_PAYLOAD_SIZE = 4352
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

    if settings.get("BRINGUP_SPEC_SET"):
        s = load_spec()
        if s.model == MODEL_NAME:
            return s
    return load_spec(str(SPEC_PATH))


def resolve_model_path() -> str:
    """PREFILL_HF_MODEL, else the bring-up spec's checkpoint (paths.hf)."""
    env = settings.get("HF_MODEL")
    if env:
        return env
    from models.demos.common.bringup.reference.golden import hf_path

    return hf_path(bringup_spec())


def served_layers(first_layer_idx: int, num_layers: int) -> list[int]:
    """Global indices of the layers this rank builds and acks (see module docstring)."""
    from models.demos.common.bringup.core.spec import parse_layers

    env = settings.get("LAYERS")
    layers = parse_layers(env, Xing40Config.NUM_LAYERS) if env else range(Xing40Config.NUM_LAYERS)
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
    """Runtime contract of ADDING_A_PREFILL_MODEL.md section 2 and bringup/serving_contract.md (stateless w.r.t. the
    KV cache: the engine's cache is bound into the attention modules per chunk)."""

    PAD_ID = 0xFFFFFFFF

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
        # in the engine's cache (bind_cache per chunk), so the modules' own latent caches are freed.
        self.model.setup(c, params.max_seq_len)
        for blk in self.model.blocks:
            blk.attn.drop_own_cache()
        self._ttnn = ttnn
        self._sink = None
        self._prepared = None
        # Opt-in (serve/runner.py): called as hidden_sink(h, slot, start, end) with the last layer's output of every
        # served chunk, after its acks, before h is freed. None (default) for the runner tests.
        self.hidden_sink = None

    def set_layer_completion_sink(self, sink) -> None:
        self._sink = sink

    def make_chunk_input(self, token_ids, start: int = 0):
        """The engine's H2D payload for a chunk at ``start``: uint32 ROW_MAJOR [sp, 1, chunk / sp], sharded over
        axis 0, PAD_ID tail, laid out on the rows as the server does (tt/layout.py:server_order). Host side; used
        only by compile() warm-up (the engine supplies real inputs)."""
        import torch

        import ttnn
        from models.demos.xing40_a4b_d_p.tt.layout import server_order

        c, sp = self.config.chunk_size, self.config.mesh_shape[0]
        ids = torch.tensor(list(token_ids) + [self.PAD_ID] * (c - len(token_ids)), dtype=torch.int64)
        t = ids[server_order(start, c, sp)].reshape(sp, 1, c // sp).to(torch.uint32)
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

    def _prepare(self, kv) -> None:
        """Load time: let every attention bind the engine's cache (a gather scratch of the cache's dtype)."""
        if self._prepared is kv.contract.kvpe:
            return
        for blk in self.model.blocks:
            blk.attn.prepare_cache(kv.contract.kvpe)
        self._prepared = kv.contract.kvpe

    def _bind(self, kv, slot: int) -> None:
        c = kv.contract
        for blk in self.model.blocks:
            blk.attn.bind_cache(c.kvpe, slot, c.kv_row(blk.i), len(c.layers))

    def _run(self, ids, kv_cache, slot, start, end, request_id, acks: bool, d2h_service=None, metadata_msg=None):
        ttnn = self._ttnn
        sink = self._sink if acks else None
        d2h = d2h_service if acks else None
        assert d2h is None or metadata_msg is not None, "metadata_msg is required with d2h_service"
        mesh = self.mesh_device
        self._bind(kv_cache, slot)
        clean = self.model.embed.clamp_ids(ids)
        h = self.model.embed(clean)
        ttnn.deallocate(clean)
        for blk in self.model.blocks:
            h2 = blk(h, start, end)  # writes the layer's KV records and zeroes the pad rows of the last one
            ttnn.deallocate(h)
            h = h2
            if d2h is not None:
                # Device-op ack on the same CQ, after the layer's cache write and pad zero: no host sync.
                ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h, metadata=metadata_msg)
            elif sink is not None:
                # The ack promises the layer's KV is in DRAM (the KV Manager reads it out of band): wait for it.
                ttnn.event_synchronize(ttnn.record_event(mesh, 0))
                sink(blk.i, request_id)
        if acks and self.hidden_sink is not None:
            self.hidden_sink(h, slot, start, end)
        ttnn.deallocate(h)

    def compile(self, kv_cache: XingKvCaches) -> None:
        """Warm the programs once in slot 0 (junk KV there; a request only reads rows it wrote itself): a cold full
        chunk, an unaligned follow-up start with a mid-record end (pad zero), and the last chunk of the slot."""
        ttnn = self._ttnn
        assert kv_cache.layers == self.layers, (kv_cache.layers, self.layers)
        self._prepare(kv_cache)
        c, m = self.config.chunk_size, self.config.max_seq_len
        for start, n in ((0, c), (2944 % c if m > c else 0, c - 2000), (m - c, c)):
            ids = self.make_chunk_input([0] * n, start)
            self._run(ids, kv_cache, 0, start, start + n, 0, acks=False)
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
        c, m = self.config.chunk_size, self.config.max_seq_len
        assert actual_start % 32 == 0, f"actual_start {actual_start} must be tile (32) aligned"
        assert actual_start < actual_end <= min(actual_start + c, m), (actual_start, actual_end, c, m)
        assert actual_start + c <= m, "chunk overruns the user slot (the server pulls the last chunk back)"
        assert 0 <= slot_id < kv_cache.contract.num_users, (slot_id, kv_cache.contract.num_users)
        assert tuple(input_tensor.shape)[-1] * self.config.mesh_shape[0] == c, input_tensor.shape
        self._prepare(kv_cache)
        self._run(
            input_tensor,
            kv_cache,
            int(slot_id),
            int(actual_start),
            int(actual_end),
            request_id,
            acks=True,
            d2h_service=d2h_service,
            metadata_msg=metadata_msg,
        )
        return None  # last/single rank: the populated cache is the output

    # --- migration hooks ---
    def build_kv_chunk_table(
        self, kv_cache, path: str, *, first_layer_idx=0, num_my_layers=None, stage_layout=None, stage_layouts=None
    ) -> str:
        """Export the address table (tt/runners/kv_contract.py). The runner's migration path passes the rank's layer
        range and the gathered stage layout; the mock-only path calls (kv, path=)."""
        import ttnn

        if stage_layout is None and stage_layouts:
            assert len(stage_layouts) == 1, "one cache, one stage layout"
            stage_layout = stage_layouts[0]
        n = len(kv_cache.layers)
        assert num_my_layers in (None, n), (num_my_layers, n)
        assert first_layer_idx == kv_cache.layers[0], (first_layer_idx, kv_cache.layers)
        table = kv_cache.contract.address_table(
            seq_len=self.config.max_seq_len, first_layer_idx=first_layer_idx, stage_layout=stage_layout
        )
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
        env_cache = self.ttnn_cache_default if settings.get("TTNN_CACHE") == "@default" else settings.get("TTNN_CACHE")
        if not env_cache:
            return None
        sp, tp = mesh_shape
        path = Path(env_cache) / f"{self.name}_bh_{int(sp) * int(tp)}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def num_kv_cache_layers(self, num_layers: int) -> int:
        """Acks the producer should expect per chunk: one per layer (every Xing layer writes kv_latent)."""
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
