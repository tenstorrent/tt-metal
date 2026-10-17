# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""``DS41PrefillAdapter``: the common/prefill engine <-> DeepSeek-V4.1-Flash (text decoder; vision and MTP skipped).

Single rank on a 2D Blackhole mesh (FABRIC_2D; SP = mesh rows over axis 0, TP = mesh columns over axis 1, both read
from mesh.shape). The model is the all-device tt/model.py:TtDS41Model, the ladder's model (lm_head off). The KV caches
the engine owns (tt/runners/kv_contract.py) are the model's served state: each slot's DS41DeviceState is bound to
its slabs of them, so the window / compressor modules write the served records in place (one format: bf16
ROW_MAJOR, what the ladder runs on).

Served layers: engine layer k = model layer ``layer_ids[k]``, from testing/contract.layer_ids_from_env
(PREFILL_LAYER_IDS when the bring-up contract serves a non-contiguous subset, F100; else
[first_layer_idx, first_layer_idx + num_layers)). Acks, the table and the read-back use the engine index. A subset
with gaps (a partial bring-up's representative layers) starts every run after a gap from the framework's feed
(testing/contract.gap_feed, F102: the golden block input of the run's first layer, as the ladder does); None for a
contiguous range (the served model).

Per chunk (prefill_chunk):
  1. the engine input (uint32 ROW_MAJOR [sp, 1, chunk / sp], sharded over axis 0, replicated over axis 1, the server's
     order for a chunk at actual_start, PAD_ID 0xFFFFFFFF past actual_end) is read back once: the token ids are the
     engram step's host input (components.yaml engram_rows, CPU: hash + memory-mapped lookup), and the model's rows are
     natural blocks (tt/model.py), so the ids are put in natural order with the load-time geometry
     (tt/attention.WindowGeometry.server_rows) and embedded (pad ids clamped into the vocabulary, K-72)
  2. the slot's id history takes [actual_start, actual_end); every engram layer's tail = the compressed ids of the 3
     positions before actual_start (a follow-up starts inside the slot's earlier prompt); begin_chunk sends the rows
  3. layer by layer: the layer writes its records (window rows, compressed / index rows, the last record's pad rows
     zero) through update_padded_kv_cache; then the ack: host sink = ttnn.synchronize_device, then
     sink(first_layer_idx + k, request_id) (K-71); D2H (d2h_service given) = outbound_socket_service_sync on the same
     command queue, no host sync
No host round trip inside a layer (the ladder gates host_transfers_per_layer at 0).

Import-light: all device / model imports happen inside the methods.
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams

MODEL_NAME = "deepseek_v41_flash_d_p"
SPEC_PATH = Path(__file__).resolve().parents[2] / "bringup" / "spec.yaml"
PAD_ID = 0xFFFFFFFF


class DS41Config:
    """Static model-dimension constants (the prefill engine's ``model_config``)."""

    NUM_LAYERS = 40
    EMB_SIZE = 5120
    # The runner opens the fabric with this packet payload (runner_utils.open_mesh_device; K-79): the fabric default
    # every bring-up test ran with (tt_metal/fabric/erisc_datamover_builder.hpp), one fp32 tile per packet.
    FABRIC_PAYLOAD_SIZE = 4352
    HEAD_DIM = 512
    INDEX_HEAD_DIM = 128
    HC_MULT = 4
    VOCAB_SIZE = 129280
    HF_ID = "deepseek-ai/DeepSeek-V4.1-Flash"


def bringup_spec():
    """The bring-up spec: BRINGUP_SPEC when set and this model's, else this model's own spec.yaml."""
    from models.demos.common.bringup.reference.golden import load_spec

    try:
        s = load_spec()
        if s.model == MODEL_NAME:
            return s
    except SystemExit:  # BRINGUP_SPEC unset
        pass
    return load_spec(str(SPEC_PATH))


def model_path() -> str:
    """The checkpoint the run uses (the spec's: the scaled one in a partial run on a smaller box, F75)."""
    from models.demos.common.bringup.reference.golden import hf_path

    return hf_path(bringup_spec())


def served_layer_ids(first_layer_idx: int, num_layers: int) -> list[int]:
    """Model layer of each engine layer (OV-14 / F100: testing/contract.layer_ids_from_env)."""
    from models.demos.common.bringup.testing.contract import layer_ids_from_env

    ids = layer_ids_from_env(first_layer_idx, num_layers)
    assert ids == sorted(set(ids)) and ids, f"served layers {ids} must be ascending and distinct"
    return ids


@dataclass
class DS41KvCaches(KvCaches):
    contract: object  # kv_contract.DS41ContractKV
    layer_ids: list = field(default_factory=list)
    by_config: dict = field(default_factory=dict)  # {config name: ttnn.Tensor}, table order


class DS41PrefillRuntime:
    """Runtime contract of ADDING_A_PREFILL_MODEL.md section 2 and bringup/serving_contract.md (stateless w.r.t. the
    KV cache: each slot's state is bound to its slabs of the engine's caches)."""

    def __init__(self, *, mesh_device, hf_config, params: PrefillRunParams):
        import ttnn
        from models.demos.deepseek_v41_flash_d_p.tt.attention import WindowGeometry
        from models.demos.deepseek_v41_flash_d_p.tt.model import TtDS41Model

        assert params.is_first_rank and params.is_last_rank, "single-rank only"
        assert tuple(params.mesh_shape) == tuple(mesh_device.shape), (params.mesh_shape, tuple(mesh_device.shape))
        c, m = params.chunk_size, params.max_seq_len
        assert m % c == 0, (m, c)
        self.mesh_device, self.config = mesh_device, params
        self.layer_ids = served_layer_ids(params.first_layer_idx, params.num_layers)
        self.model = TtDS41Model(mesh_device, model_path(), self.layer_ids, max_seq=m, chunk=c, lm_head=False)
        self.geo = WindowGeometry(mesh_device, c)
        from models.demos.common.bringup.testing.contract import gap_feed

        self.gap_feed = gap_feed(self.layer_ids, m, c)  # the bring-up's gapped subset only (F102); None when served
        self._ttnn = ttnn
        self._sink = None
        self._d2h = None
        self._states = {}  # (id of the KvCaches, slot) -> DS41DeviceState bound to the slot's slabs

    # ------------------------------------------------------------------ engine hooks
    def set_layer_completion_sink(self, sink) -> None:
        self._sink = sink

    def set_d2h_ack_service(self, service) -> None:
        self._d2h = service

    def state(self, kv, slot: int):
        """The slot's DS41DeviceState over the engine's caches (built once per slot; no device allocation)."""
        key = (id(kv), int(slot))
        if key not in self._states:
            self._states[key] = self.model.new_state(
                self.config.max_seq_len, slabs=kv.contract.slabs(int(slot)), keep_history=False
            )
        return self._states[key]

    def make_chunk_input(self, token_ids, start: int = 0):
        """The engine's H2D payload for a chunk at ``start`` (server order, PAD_ID tail), host side: compile()
        warm-up only (the engine supplies real inputs)."""
        import torch

        ttnn = self._ttnn
        c, sp, w = self.config.chunk_size, self.geo.sp, self.geo.w
        nat = torch.full((c,), PAD_ID, dtype=torch.int64)
        nat[: len(token_ids)] = torch.as_tensor(list(token_ids), dtype=torch.int64)
        srv = nat[self.geo.server_rows(start).reshape(-1)].reshape(sp, 1, w)
        return ttnn.from_torch(
            srv.to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(
                self.mesh_device,
                ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()]),
            ),
        )

    def _host_ids(self, inp, start: int, end: int):
        """Engine input -> the chunk's ids in natural order [chunk] int64 (PAD_ID past ``end``): one read of mesh
        column 0's rows, un-shuffled with the load-time server order."""
        import torch

        ttnn = self._ttnn
        g = self.geo
        dev = ttnn.get_device_tensors(inp)
        rows = [ttnn.to_torch(dev[r * g.tp]).reshape(-1).to(torch.int64) & 0xFFFFFFFF for r in range(g.sp)]
        srv = torch.stack(rows)  # [SP, W] the server's order
        nat = torch.empty(g.chunk, dtype=torch.int64)
        nat[g.server_rows(start).reshape(-1)] = srv.reshape(-1)
        nat[end - start :] = PAD_ID
        return nat

    def _run(self, inp, kv, slot, start, end, request_id, acks: bool, d2h_service=None, metadata_msg=None):
        ttnn = self._ttnn
        model = self.model
        sink = self._sink if acks else None
        d2h = (d2h_service if d2h_service is not None else self._d2h) if acks else None
        assert d2h is None or metadata_msg is not None, "metadata_msg is required with d2h_service"
        st = self.state(kv, slot)
        model.state = st  # begin_chunk reads (and advances) the active slot's engram tail
        ids = self._host_ids(inp, start, end)
        st.note_tokens(ids, start, end)
        model.begin_chunk(ids, start)
        h = None
        for k, i in enumerate(self.layer_ids):
            if k == 0 or (self.gap_feed is not None and i in self.gap_feed.starts):
                if h is not None:
                    model.free(h)
                h = model.embed(ids, start) if i == 0 else model.from_host(self.gap_feed.rows(i, start), start)
            h2 = model.layer(i, h, start, st, end=end)  # writes the layer's records, the last one's pad rows zero
            model.free(h)
            h = h2
            if d2h is not None:
                # device-op ack on the same command queue, after the layer's cache writes: no host sync
                ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h, metadata=metadata_msg)
            elif sink is not None:
                # the ack promises the layer's KV is in DRAM (the KV Manager reads it out of band): wait for it (K-71)
                ttnn.synchronize_device(self.mesh_device)
                sink(self.config.first_layer_idx + k, request_id)
        model.free(h)

    def compile(self, kv_cache: DS41KvCaches) -> None:
        """Warm the programs once in slot 0 (junk KV there; a request reads only rows it wrote itself): one cold full
        chunk. Other starts compile on first use."""
        ttnn = self._ttnn
        assert kv_cache.layer_ids == self.layer_ids, (kv_cache.layer_ids, self.layer_ids)
        c = self.config.chunk_size
        ids = self.make_chunk_input([0] * c, 0)
        self._run(ids, kv_cache, 0, 0, c, 0, acks=False)
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
        **_unused,
    ):
        c, m = self.config.chunk_size, self.config.max_seq_len
        start, end = int(actual_start), int(actual_end)
        assert start % 32 == 0, f"actual_start {start} must be tile (32) aligned"
        assert start < end <= min(start + c, m), (start, end, c, m)
        assert start + c <= m, "chunk overruns the slot (the server pulls the last chunk back)"
        assert 0 <= int(slot_id) < kv_cache.contract.num_users, (slot_id, kv_cache.contract.num_users)
        assert tuple(input_tensor.shape)[-1] * self.geo.sp == c, input_tensor.shape
        self._run(
            input_tensor,
            kv_cache,
            int(slot_id),
            start,
            end,
            request_id,
            acks=True,
            d2h_service=d2h_service,
            metadata_msg=metadata_msg,
        )
        return None  # last / single rank: the populated cache is the output

    # ------------------------------------------------------------------ migration hooks
    def kv_migration_stages(self, kv_cache, first_layer_idx, num_layers):
        """One stage per table config, in table order (the runner gathers each one's base address and chips)."""
        from models.demos.common.prefill.runners.migration import KvCacheStage

        assert int(num_layers) == len(kv_cache.layer_ids), (num_layers, kv_cache.layer_ids)
        return [KvCacheStage(b, int(first_layer_idx), int(num_layers)) for b in kv_cache.contract.stage_bases()]

    def build_kv_chunk_table(
        self, kv_cache, path: str, *, first_layer_idx=0, num_my_layers=None, stage_layout=None, stage_layouts=None
    ) -> str:
        """Export the address table (tt/runners/kv_contract.py). The runner's migration path passes the rank's layer
        range and the gathered stage layouts (one per config); the mock-only path calls (kv, path=)."""
        import ttnn

        n = len(kv_cache.layer_ids)
        assert num_my_layers in (None, n), (num_my_layers, n)
        if stage_layouts is None and stage_layout is not None:
            stage_layouts = [stage_layout]
        table = kv_cache.contract.address_table(
            seq_len=self.config.max_seq_len, first_layer_idx=int(first_layer_idx), stage_layouts=stage_layouts
        )
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
        return path

    def kv_migration_base_address(self, kv_cache) -> int:
        return kv_cache.contract.base_address()


class DS41PrefillAdapter(PrefillModelAdapter):
    """DeepSeek-V4.1-Flash prefill adapter on a 2D Blackhole mesh (CSA window / compressed sparse attention with a
    shared indexer, engram, mHC 4-stream residual, fp4 routed experts as bfp8 + a shared expert)."""

    name = MODEL_NAME
    model_config = DS41Config
    hf_model_default = ""  # the bring-up spec's checkpoint (model_path)
    ttnn_cache_default = ""
    prefill_trace_default = ""  # the bring-up golden dir; PREFILL_TRACE_DIR overrides
    default_gate_mode = "DEVICE_FP32"
    l1_small_size = 24576  # the spec's device_params
    hf_repo_id = DS41Config.HF_ID

    def load_hf_config(self):
        """The checkpoint's text config as attributes (the runner sets max_seq_len on it)."""
        raw = json.loads((Path(model_path()) / "config.json").read_text())
        return types.SimpleNamespace(**raw.get("text_config", raw))

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        return None  # weights load from the checkpoint (experts converted to bfp8 at load, tt/experts.py)

    def num_kv_cache_layers(self, num_layers: int) -> int:
        """Acks per chunk: one per served layer (every layer owns 0_window_kv)."""
        return int(num_layers)

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
        from models.demos.deepseek_v41_flash_d_p.tt.runners.kv_contract import DS41ContractKV

        layer_ids = served_layer_ids(params.first_layer_idx, params.num_layers)
        contract = DS41ContractKV(
            mesh_device,
            Config.from_dir(model_path()),
            layer_ids,
            max_seq=params.max_seq_len,
            chunk=params.chunk_size,
            num_users=params.num_users,
        )
        return DS41KvCaches(contract=contract, layer_ids=layer_ids, by_config=dict(contract.by_config))

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        return DS41PrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, params=params)

    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir, num_layers):
        """The producer's read-back (K-73): every served cache through the table vs the bring-up golden,
        {window_kv, compress_kv, index_k: min PCC}."""
        from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
        from models.demos.deepseek_v41_flash_d_p.tt.runners.kv_contract import read_slot_kv_and_check_pcc

        return read_slot_kv_and_check_pcc(
            table,
            device_map,
            slot_id,
            real_len,
            trace_dir,
            Config.from_dir(model_path()),
            served_layer_ids(0, int(num_layers)),
        )


def contract_state_pcc(spec, runtime, kv, slot, length, golden) -> dict:
    """Bring-up contract hook (testing/contract.py; bringup/hooks.py re-exports it): the slot's fixed-size state
    (spec state.fixed: the window ring, the compressor's partial group, the engram tail) after a request of
    ``length`` tokens vs the golden snapshot at ``length``; {name: min PCC over the served layers} (the engram tail,
    integer ids, scores 1.0 when equal, else 0.0)."""
    import torch
    from loguru import logger

    from models.demos.deepseek_v41_flash_d_p.tt.runners.kv_contract import _pcc

    st = runtime.state(kv, slot)
    out = {}
    for layer in runtime.layer_ids:
        gold = golden.state(layer, at=length)
        dev = st.to_torch(layer, length)
        for name in spec.state_fixed:
            if name not in gold or name not in dev:
                continue
            if name == "engram_tail":
                p = float(torch.equal(dev[name].long(), gold[name].long()))
            else:
                p = _pcc(dev[name].float(), gold[name].float().reshape(dev[name].shape))
            out[name] = min(out.get(name, 1.0), p)
            logger.info(f"  layer {layer:>2} {name} at {length}: {p:.6f}")
    return out
