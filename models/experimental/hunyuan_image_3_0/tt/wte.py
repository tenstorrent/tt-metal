# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device token embedding (model.wte) for HunyuanImage-3.0.
# Mirrors HunyuanTtModel.embed — ROW_MAJOR weight table, TILE output.

from __future__ import annotations

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.hunyuan_image_3_0.tt.parallel_utils import resid_mem_config


class HunyuanTtWte(LightweightModule):
    """``model.wte``: input_ids [B, S] uint32 -> embeddings [B, S, H] TILE."""

    def __init__(
        self,
        device,
        wte_weight: torch.Tensor,
        *,
        weight_dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = int(wte_weight.shape[1])
        embed_dtype = weight_dtype
        if embed_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            embed_dtype = ttnn.bfloat16
        upload_kwargs = dict(
            dtype=embed_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if mesh_mapper is not None:
            upload_kwargs["mesh_mapper"] = mesh_mapper
        self.embed_weight = ttnn.from_torch(wte_weight, **upload_kwargs)

    def _upload_ids(self, input_ids: torch.Tensor, *, mesh_mapper=None) -> ttnn.Tensor:
        mapper = mesh_mapper
        if mapper is None and hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1:
            mapper = ttnn.ReplicateTensorToMesh(self.device)
        kwargs = dict(
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if mapper is not None:
            kwargs["mesh_mapper"] = mapper
        return ttnn.from_torch(input_ids.to(torch.int32), **kwargs)

    def embed(
        self,
        input_ids: torch.Tensor | ttnn.Tensor,
        *,
        mesh_mapper=None,
    ) -> ttnn.Tensor:
        """Embed token ids -> [B, S, H] TILE on device."""
        if isinstance(input_ids, torch.Tensor):
            bsz, seq = input_ids.shape[0], input_ids.shape[1]
            ids_tt = self._upload_ids(input_ids, mesh_mapper=mesh_mapper)
            owns_ids = True
        else:
            bsz, seq = input_ids.shape[0], input_ids.shape[-1]
            ids_tt = input_ids
            owns_ids = False
        # Match HunyuanTtModel.embed: DRAM above RESID_L1_MAX_SEQ avoids L1 residual
        # vs SDPA/matmul CB clashes at long prefill.
        emb = ttnn.embedding(ids_tt, self.embed_weight, layout=ttnn.TILE_LAYOUT, memory_config=resid_mem_config(seq))
        if owns_ids:
            ttnn.deallocate(ids_tt)
        return ttnn.reshape(emb, [bsz, seq, self.hidden_size])

    def to_torch(self, emb_tt: ttnn.Tensor, *, batch: int, seq: int) -> torch.Tensor:
        """Device TILE embeddings -> host float ``[batch, seq, H]``."""
        if hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1:
            out = ttnn.to_torch(emb_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            out = out[:batch]
        else:
            out = ttnn.to_torch(emb_tt)
        return out.reshape(batch, seq, self.hidden_size).float()

    def embedding_torch(
        self,
        input_ids: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        mesh_mapper=None,
    ) -> torch.Tensor:
        """On-device embed + download for host scatter paths (VAE/ViT token inject)."""
        emb_tt = self.embed(input_ids, mesh_mapper=mesh_mapper)
        out = self.to_torch(emb_tt, batch=input_ids.shape[0], seq=input_ids.shape[1]).to(dtype=dtype)
        ttnn.deallocate(emb_tt)
        return out


class BackboneWteAdapter:
    """Reuse ``HunyuanTtModel.embed_weight`` for trace decode (no duplicate wte upload)."""

    def __init__(self, model):
        if getattr(model, "embed_weight", None) is None:
            raise ValueError("backbone has no embed_weight; pass wte_weight for HunyuanTtWte")
        self.model = model

    def embed(self, input_ids: torch.Tensor | ttnn.Tensor) -> ttnn.Tensor:
        if isinstance(input_ids, torch.Tensor):
            kwargs = dict(
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.model.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if hasattr(self.model.device, "get_num_devices") and self.model.device.get_num_devices() > 1:
                kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.model.device)
            ids_tt = ttnn.from_torch(input_ids.to(torch.int32), **kwargs)
            hidden = self.model.embed(ids_tt)
            ttnn.deallocate(ids_tt)
            return hidden
        return self.model.embed(input_ids)

    def embedding_torch(
        self,
        input_ids: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        mesh_mapper=None,
    ) -> torch.Tensor:
        """On-device embed + download (reuses backbone embed_weight, no second upload)."""
        emb_tt = self.embed(input_ids)
        bsz, seq = input_ids.shape[0], input_ids.shape[1]
        if hasattr(self.model.device, "get_num_devices") and self.model.device.get_num_devices() > 1:
            out = ttnn.to_torch(emb_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.model.device, dim=0))
            out = out[:bsz]
        else:
            out = ttnn.to_torch(emb_tt)
        out = out.reshape(bsz, seq, self.model.hidden_size).float().to(dtype=dtype)
        ttnn.deallocate(emb_tt)
        return out
