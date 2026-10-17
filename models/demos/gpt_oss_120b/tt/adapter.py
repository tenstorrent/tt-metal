# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""gpt_oss_120b behind tt-metal's prefill runner (models/demos/common/prefill/adapter.py, ADAPTER_PATHS
"gpt_oss_120b"), as serving_contract.md "Adapter and table" says.

The engine-owned KV cache IS the device model's state (tt/model.py TtKvState: one K and one V tensor per chip, DRAM
ND-sharded [1, 1, 32, 64] ROUND_ROBIN_1D, batch slot = slot * layers + layer), so the runner, the ladder and the
contract tests all run on the same cache. The runtime drives the device model (TtModel.embed_ids, TtModel.layer) one
chunk at a time and acks each served layer to the host sink after a device event for that layer (ack_mode host).

Single rank: the runtime embeds the tokens and runs every served layer (no D2D pipeline stages, no trace).
Import-light: the model stack loads inside the methods.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams

RECORD_TOKENS = 32


class GptOss120bServeConfig:
    """The model constants the runner and the producer read (MODEL_CFG)."""

    NUM_LAYERS = 36
    NUM_KEY_VALUE_HEADS = 8
    HEAD_DIM = 64
    # the bring-up tests open the fabric with its default 4352 B packets; the fp32 CCLs need more than one fp32 tile
    # (4096 B) per packet (known issue "The prefill runner sizes the fabric packet from FABRIC_PAYLOAD_SIZE")
    FABRIC_PAYLOAD_SIZE = 4352


@dataclass(frozen=True)
class RuntimeConfig:
    chunk_size: int
    max_seq_len: int
    num_layers: int
    first_layer_idx: int
    num_users: int
    is_first_rank: bool = True
    is_last_rank: bool = True
    use_trace: bool = False


class GptOss120bRuntime:
    """What prefill_runner.py drives: compile, prefill_chunk, the layer-ack sink, the KV migration stage and table."""

    def __init__(self, mesh_device, model, cfg: dict, config: RuntimeConfig):
        self.mesh_device, self.model, self.cfg, self.config = mesh_device, model, cfg, config
        self.layers = list(range(config.first_layer_idx, config.first_layer_idx + config.num_layers))
        self._sink = None
        self.compiled = False

    # ---------------------------------------------------------------- serving
    def _run(self, tt_ids, kv, slot_id: int, start: int, end: int, request_id: int, ack: bool) -> None:
        import ttnn

        assert start % RECORD_TOKENS == 0, f"actual_start {start} must be tile-aligned"
        assert start < end <= start + self.config.chunk_size, f"[{start}, {end}) is not within one chunk"
        assert start + self.config.chunk_size <= kv.max_seq, f"chunk at {start} runs past the cache ({kv.max_seq})"
        assert 0 <= slot_id < kv.slots, f"slot {slot_id} out of range [0, {kv.slots})"
        kv.slot = slot_id
        h = self.model.embed_ids(tt_ids)
        tt_ids.deallocate(True)
        for i in self.layers:
            h2 = self.model.layer(i, h, start, kv, end=end)
            self.model.free(h)
            h = h2
            if ack and self._sink is not None:
                # the sink tells migration this layer's K / V rows of the chunk are in DRAM: wait for the device first
                ttnn.event_synchronize(ttnn.record_event(self.mesh_device, 0))
                self._sink(i, request_id)
        self.model.free(h)

    def prefill_chunk(
        self,
        input_tensor,
        kv_caches,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        d2h_service=None,
        metadata_msg=None,
    ):
        """One chunk as the engine sends it: ``input_tensor`` uint32 ROW_MAJOR [8, 1, 640] over the SP rows (each row in
        its positions' order, PAD_ID past ``actual_end``), written into ``slot_id`` at [actual_start, actual_end).
        Consumes ``input_tensor``; returns None (the cache is the output)."""
        if d2h_service is not None:
            raise NotImplementedError("gpt_oss_120b acks through the host sink; run with PREFILL_LAYER_ACK_D2H=0")
        self._run(input_tensor, kv_caches, slot_id, actual_start, actual_end, request_id, ack=True)
        return None

    def compile(self, kv_caches) -> None:
        """JIT-compile the chunk programs on a scratch cache of the same shape (the engine's cache stays zero): a
        cold chunk, then a chunk over a cached prefix ending mid-record (the pad mask). No acks."""
        import torch

        import ttnn

        from .model import new_kv_state

        kv = kv_caches  # the scratch cache has its shape, so the programs compiled here are the ones served
        scratch = new_kv_state(self.mesh_device, self.cfg, self.layers, kv.max_seq, kv.chunk, kv.slots)
        chunk = self.config.chunk_size
        for start, end in ((0, chunk), (chunk, 2 * chunk - 16)):
            if start + chunk > scratch.max_seq:
                break
            self._run(self._ids(torch.zeros(chunk, dtype=torch.int64)), scratch, 0, start, end, 0, ack=False)
        ttnn.synchronize_device(self.mesh_device)
        scratch.free()
        self.compiled = True

    def _ids(self, tokens):
        """Host token ids (start 0, natural order) as the engine's H2D input."""
        import ttnn

        mesh_shape = tuple(self.mesh_device.shape)
        return ttnn.from_torch(
            tokens.reshape(mesh_shape[0], 1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=mesh_shape, dims=(0, None)),
        )

    def set_layer_completion_sink(self, sink) -> None:
        """``sink(global_layer_idx, request_id)``, once per served layer per chunk, in layer order."""
        self._sink = sink

    # ---------------------------------------------------------------- migration
    def kv_migration_base_address(self, kv_caches) -> int:
        """The table's entry 0 (layer 0, position 0, slot 0, config k_h0): the K cache's address in bank 0."""
        return int(kv_caches.k.buffer_address())

    def kv_migration_stages(self, kv_caches, first_layer_idx=None, num_my_layers=None):
        from models.demos.common.prefill.runners.migration import KvCacheStage

        first = self.config.first_layer_idx if first_layer_idx is None else int(first_layer_idx)
        count = self.config.num_layers if num_my_layers is None else int(num_my_layers)
        return [KvCacheStage(self.kv_migration_base_address(kv_caches), first, count)]

    def build_kv_chunk_table(self, kv_caches, path: str, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        """The served cache's KV address table (TtKvState.build_kv_chunk_table), serialized to ``path``. Single rank:
        the table covers this rank's layers from the cache's own addresses."""
        import ttnn

        assert stage_layouts is None or all(len(s) == 1 for s in stage_layouts), "gpt_oss_120b serves one rank"
        table = kv_caches.build_kv_chunk_table(self.config.max_seq_len)
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, str(path))
        return str(path)


class GptOss120bPrefillAdapter(PrefillModelAdapter):
    name = "gpt_oss_120b"
    model_config = GptOss120bServeConfig
    # the bring-up box's checkpoint (spec paths.hf); PREFILL_HF_MODEL overrides, a hub id resolves in the local HF cache
    hf_model_default = (
        "/mnt/models/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    )
    ttnn_cache_default = ""  # weights load from the checkpoint; no TTNN weight cache
    prefill_trace_default = ""
    default_gate_mode = "DEVICE_FP32"

    hf_repo_id = "openai/gpt-oss-120b"
    num_layers_to_download = 36
    needs_flat_config_dir = False

    def hf_dir(self) -> Path:
        """The checkpoint directory: PREFILL_HF_MODEL, else the local HF cache snapshot of hf_model_default."""
        p = Path(self.hf_model_path())
        if p.is_dir():
            return p
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(str(p), local_files_only=True))

    def _cfg(self) -> dict:
        return json.loads((self.hf_dir() / "config.json").read_text())

    def load_hf_config(self):
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(str(self.hf_dir()))

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        return None

    @staticmethod
    def _layers(params: PrefillRunParams) -> list[int]:
        return list(range(params.first_layer_idx, params.first_layer_idx + params.num_layers))

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        """The served cache of this rank's layers for ``params.num_users`` slots, zeroed (tt/model.py new_kv_state)."""
        from .model import TtKvState, new_kv_state

        KvCaches.register(TtKvState)  # the engine treats it as an opaque KvCaches handle
        return new_kv_state(
            mesh_device, self._cfg(), self._layers(params), params.max_seq_len, params.chunk_size, params.num_users
        )

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams) -> GptOss120bRuntime:
        from .model import load_model

        assert params.is_first_rank and params.is_last_rank, "gpt_oss_120b serves on one rank (no D2D stages)"
        assert not params.use_trace, "gpt_oss_120b runs untraced (PREFILL_USE_TRACE=0)"
        assert tuple(params.mesh_shape) == tuple(mesh_device.shape), (params.mesh_shape, tuple(mesh_device.shape))
        layers = self._layers(params)
        model = load_model(mesh_device, self.hf_dir(), layers, params.max_seq_len, params.chunk_size, lm_head=False)
        config = RuntimeConfig(
            chunk_size=params.chunk_size,
            max_seq_len=params.max_seq_len,
            num_layers=params.num_layers,
            first_layer_idx=params.first_layer_idx,
            num_users=params.num_users,
        )
        return GptOss120bRuntime(mesh_device, model, self._cfg(), config)

    # ---------------------------------------------------------------- producer read-back
    def read_slot_kv_and_check_pcc(self, table, device_map: dict, slot_id: int, real_len: int, trace_dir, num_layers):
        """The KV of ``slot_id`` over [0, real_len) read through the table (over UMD, as the KV Manager reads it) vs
        the golden trace's kv_cache/layer_<i>.safetensors: {"k": min PCC, "v": min PCC} over the layers and heads.
        Configs 0..n-1 are the K heads (meta column order, so the golden's half-split K is permuted), n..2n-1 the V
        heads. PCC computed here in torch (the producer's comp_pcc may be stubbed in a precompile pass)."""
        import torch
        from safetensors import safe_open

        from models.demos.common.prefill.runners import prefill_producer as producer

        n_kv, hd = GptOss120bServeConfig.NUM_KEY_VALUE_HEADS, GptOss120bServeConfig.HEAD_DIM
        half = hd // 2
        perm = torch.tensor([half * (m % 2) + m // 2 for m in range(hd)])
        trace_dir = Path(trace_dir)
        meta = trace_dir / "metadata.json"
        golden_layers = json.loads(meta.read_text()).get("layers") if meta.is_file() else None
        if not golden_layers or len(golden_layers) != num_layers:
            golden_layers = list(range(num_layers))
        mins = {"k": 1.0, "v": 1.0}
        for li, layer in enumerate(golden_layers):
            with safe_open(str(trace_dir / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as f:
                gold = {n: f.get_tensor(f"{n}_cache_layer_{layer}").float() for n in ("key", "value")}
            for kind, base, name in (("k", 0, "key"), ("v", n_kv, "value")):
                g = gold[name]
                g = g[0] if g.dim() == 4 else g.reshape(g.shape[0], n_kv, hd).permute(1, 0, 2)  # [n_kv, seq, hd]
                g = g[:, :real_len]
                if kind == "k":
                    g = g[..., perm]
                dev = torch.stack(
                    [
                        producer._read_kv_slice(
                            table, device_map, base + h, li, slot_id, real_len, hd, producer._decode_bfp8_chunk
                        )
                        for h in range(n_kv)
                    ]
                ).float()
                mins[kind] = min(mins[kind], _pcc(dev, g))
        return mins


def _pcc(a, b) -> float:
    import torch

    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = torch.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float(torch.equal(a, b))
