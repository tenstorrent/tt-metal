# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""``Ernie45PrefillAdapter``: the common/prefill engine <-> ERNIE-4.5-21B-A3B boundary.

GQA (not MLA), so it subclasses ``PrefillModelAdapter`` directly, like GPT-OSS. Single rank on a 1x4 mesh
(sp=1, tp=4: one KV head per TP column). The KV cache the engine owns holds:
  * ``contract``: the migratable prefill-server cache (bf8, [users*layers, 1, seq, 128], DRAM round-robin)
  * ``attn[slot]``: the per-user bf16 attention cache the chunked SDPA reads the prefix from

Import-light: all device / model imports happen inside the methods.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.ernie45_d_p.config import Ernie45Config


@dataclass
class Ernie45KvCaches(KvCaches):
    contract: object  # tt.kv_contract.ErnieContractKV
    attn: list = field(default_factory=list)  # per-slot tt.attention.TtKVCache


class _SlotContract:
    """ErnieContractKV view bound to one user slot (what TtAttention writes through)."""

    def __init__(self, contract, slot):
        self.contract, self.slot = contract, slot

    def write(self, layer, k, v, start):
        self.contract.write(layer, k, v, start, slot=self.slot)


class Ernie45PrefillRuntime:
    """Structural runtime contract of ADDING_A_PREFILL_MODEL.md section 2 (stateless w.r.t. the KV cache)."""

    def __init__(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.ernie45_d_p.reference.ernie_ref import ErnieConfig, WeightLoader, resolve_model_path
        from models.demos.ernie45_d_p.tt.model import TtErnieModel

        assert params.is_first_rank and params.is_last_rank and params.first_layer_idx == 0, "single-rank only"
        self.mesh_device, self.config = mesh_device, params
        path = resolve_model_path(os.environ.get("PREFILL_HF_MODEL") or None)
        cfg = ErnieConfig.from_json(os.path.join(path, "config.json"))
        assert params.num_layers == cfg.num_hidden_layers
        self.model = TtErnieModel(mesh_device, WeightLoader(path), cfg, lm_head=False)
        self._sink = None

    def set_layer_completion_sink(self, sink) -> None:
        self._sink = sink

    def make_chunk_input(self, token_ids):
        import torch

        return torch.as_tensor(list(token_ids), dtype=torch.int64)

    def compile(self, kv_cache: Ernie45KvCaches) -> None:
        """Warm every chunk-shape program once (writes junk into slot 0 [0, chunk); real requests overwrite it)."""
        import ttnn

        c = self.config.chunk_size
        h = self.model.prefill_chunk(
            self.make_chunk_input([0] * c), 0, kv_cache.attn[0], contract_kv=_SlotContract(kv_cache.contract, 0)
        )
        ttnn.deallocate(h)
        if self.config.max_seq_len > c:  # also warm the chunked-SDPA (start > 0) path
            h = self.model.prefill_chunk(
                self.make_chunk_input([0] * c), c, kv_cache.attn[0], contract_kv=_SlotContract(kv_cache.contract, 0)
            )
            ttnn.deallocate(h)
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
        import ttnn

        assert actual_start % 32 == 0, "chunk write offset must be 32-aligned"
        assert actual_start + len(input_tensor) <= self.config.max_seq_len, "chunk overruns the user slot"
        sink = self._sink

        def on_layer(i, _h, request_id=request_id):
            if sink is not None:
                sink(i, request_id)  # global layer index (single rank: local == global)

        h = self.model.prefill_chunk(
            input_tensor,
            actual_start,
            kv_cache.attn[slot_id],
            on_layer=on_layer,
            contract_kv=_SlotContract(kv_cache.contract, slot_id),
        )
        ttnn.deallocate(h)
        return None  # last/single rank: the populated cache is the output

    # --- migration hooks ---
    def build_kv_chunk_table(self, kv_cache, path: str) -> str:
        import ttnn

        table = kv_cache.contract.address_table(seq_len=self.config.max_seq_len, chunk_size=self.config.chunk_size)
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
        return path

    def kv_migration_base_address(self, kv_cache) -> int:
        return int(kv_cache.contract.cache.k.buffer_address())


class Ernie45PrefillAdapter(PrefillModelAdapter):
    """ERNIE-4.5-21B-A3B prefill adapter (GQA 20/4, 64-expert top-6 MoE + 2 shared experts)."""

    name = "ernie45_d_p"
    model_config = Ernie45Config
    hf_model_default = ""  # resolved from the HF hub cache (baidu/ERNIE-4.5-21B-A3B-PT); PREFILL_HF_MODEL overrides
    ttnn_cache_default = ""
    prefill_trace_default = ""  # golden dir from reference/generate_golden.py; PREFILL_TRACE_DIR overrides
    default_gate_mode = "DEVICE_FP32"
    # Golden K (reference/generate_golden.py) is post-RoPE in ERNIE's native INTERLEAVED order, which is already
    # the device (Meta) order: the producer's GQA reader must not apply the HF->Meta rotary permutation.
    golden_k_rope_layout = "interleaved"

    hf_repo_id = "baidu/ERNIE-4.5-21B-A3B-PT"
    env_var = "ERNIE_MODEL_PATH"
    num_layers_to_download = 28

    def load_hf_config(self):
        from transformers import AutoConfig

        from models.demos.ernie45_d_p.reference.ernie_ref import resolve_model_path

        return AutoConfig.from_pretrained(resolve_model_path(os.environ.get("PREFILL_HF_MODEL") or None))

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None  # tt/common.py CACHE_ROOT (ERNIE_TT_CACHE) is used
        sp, tp = mesh_shape
        path = Path(env_cache) / f"{self.name}_bh_{int(sp) * int(tp)}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        from models.demos.ernie45_d_p.reference.ernie_ref import ErnieConfig
        from models.demos.ernie45_d_p.tt.attention import TtKVCache
        from models.demos.ernie45_d_p.tt.kv_contract import ErnieContractKV

        assert tuple(params.mesh_shape) == (1, 4), f"ERNIE prefill is built for a 1x4 mesh, got {params.mesh_shape}"
        cfg = ErnieConfig()
        layers = list(range(params.first_layer_idx, params.first_layer_idx + params.num_layers))
        contract = ErnieContractKV(
            mesh_device, num_layers=params.num_layers, max_seq=params.max_seq_len, num_users=params.num_users
        )
        attn = [TtKVCache(mesh_device, cfg, params.max_seq_len, layers) for _ in range(params.num_users)]
        return Ernie45KvCaches(contract=contract, attn=attn)

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        return Ernie45PrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, params=params)
