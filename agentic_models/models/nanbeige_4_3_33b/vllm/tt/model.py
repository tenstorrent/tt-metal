"""Nanbeige full-attention TP4 model; all runtime arithmetic is TTNN."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

import ttnn

from .decoder import Decoder
from .precision import dtype, load_policy


class Model:
    def __init__(
        self,
        model_dir,
        mesh_device,
        *,
        max_model_len=131072,
        num_layers=None,
        num_pages=None,
        kv_cache=None,
    ):
        self.path = Path(model_dir)
        self.policy = load_policy()
        print("PRECISION_POLICY", self.policy["name"], self.policy["sha256"], flush=True)
        self.device = mesh_device
        self.config = SimpleNamespace(**json.loads((self.path / "config.json").read_text()))
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.max_model_len = max_model_len
        self.n_layers = num_layers or self.config.num_hidden_layers
        self.layers = []
        self.collective_semaphores = Decoder.make_collective_semaphores(mesh_device)
        for i in range(self.n_layers):
            sd = {name: self.weight(name) for name in self.index if name.startswith(f"model.layers.{i}.")}
            self.layers.append(
                Decoder.from_state_dict(
                    sd,
                    hf_config=self.config,
                    layer_idx=i,
                    mesh_device=mesh_device,
                    num_pages=None,
                    policy=self.policy,
                    collective_semaphores=self.collective_semaphores,
                )
            )
            del sd
            if i % 8 == 0:
                print(f"LOAD_LAYER {i}/{self.n_layers}", flush=True)
        self.embedding = self.put(
            self.weight("model.embed_tokens.weight"),
            dim=-1,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.norm = self.put(self.weight("model.norm.weight").reshape(1, 1, 1, -1).expand(1, 1, 32, -1))
        head_name = "model.embed_tokens.weight" if self.config.tie_word_embeddings else "lm_head.weight"
        self.head = self.put(
            self.weight(head_name).T,
            dim=-1,
            dtype=dtype(self.policy["weights"]["lm_head"]),
        )
        self.head = self.layers[0].prepare_linear(self.head, "head")
        inv = 1 / (self.config.rope_theta ** (torch.arange(0, self.config.head_dim, 2).float() / self.config.head_dim))
        phase = torch.arange(max_model_len + 1024).float()[:, None] * inv
        phase = torch.cat((phase, phase), -1)
        self.cos_table = self.put(phase.cos(), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.sin_table = self.put(phase.sin(), layout=ttnn.ROW_MAJOR_LAYOUT)
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        free = (view.total_bytes_per_bank - view.total_bytes_allocated_per_bank) * view.num_banks
        # get_memory_view(DRAM) already excludes the separately reserved trace
        # region (AllocatorImpl::init_one_bank_per_channel). Do not count it twice.
        reserve = 2 * 1024**3 + 256 * 1024**2 + 308 * 1024**2
        page_bytes = self.n_layers * 2 * 2 * 4 * (2048 if self.policy["kv_dtype"] == "bf16" else 1088)
        if kv_cache is not None and len(kv_cache) != self.n_layers:
            raise ValueError("External KV cache must contain every decoder layer")
        self.num_pages = (
            kv_cache[0][0].shape[0]
            if kv_cache is not None
            else (num_pages if num_pages is not None else self.policy["kv_pool_tokens"] // 32 + 64)
        )
        external_kv_bytes = self.num_pages * page_bytes if kv_cache is not None else 0
        self.memory_budget = dict(
            free_after_weights_bytes=int(free + external_kv_bytes),
            free_before_owned_kv_allocation_bytes=int(free),
            dram_allocator_bytes=int(view.total_bytes_per_bank * view.num_banks),
            allocated_dram_before_owned_kv_bytes=int(view.total_bytes_allocated_per_bank * view.num_banks),
            external_kv_bytes=external_kv_bytes,
            trace_region_requested_bytes=512 * 1024**2,
            reserved_workspace_headroom_bytes=reserve,
            bytes_per_page_all_layers=page_bytes,
            num_pages=self.num_pages,
            tokens=self.num_pages * 32,
            num_layers=self.n_layers,
            ownership="external" if kv_cache is not None else "generator",
            configured_request_pool_tokens=self.policy["kv_pool_tokens"],
            explicit_diagnostic_num_pages=num_pages,
            capacity_bound_pages=max(0, int(free + external_kv_bytes - reserve)) // page_bytes // 32 * 32,
            reserved_startup_pages=0 if kv_cache is not None else 64,
            reserves=dict(
                workspace_bytes=2 * 1024**3,
                ccl_sampling_bytes=256 * 1024**2,
                trace_bytes=0,
                safety_bytes=308 * 1024**2,
                snapshot_bytes=0,
                speculative_bytes=0,
            ),
        )
        if self.num_pages < 64:
            raise MemoryError(self.memory_budget)
        for i, layer in enumerate(self.layers):
            if kv_cache is None:
                layer.allocate_cache(self.num_pages)
            else:
                k, v = kv_cache[i]
                if any(
                    t.shape != [self.num_pages, 2, 32, 128] or t.dtype != dtype(self.policy["kv_dtype"]) for t in (k, v)
                ):
                    raise ValueError("External KV shape/dtype does not match production cache")
                layer.key, layer.value = k, v
        print("KV_BUDGET", self.memory_budget, flush=True)

    def weight(self, name):
        if name == "lm_head.weight" and self.policy.get("gptq_weights") and self.policy["weights"]["lm_head"] == "bfp4":
            with safe_open(
                Path(self.policy["gptq_weights"]) / "lm_head.safetensors",
                framework="pt",
            ) as f:
                return f.get_tensor("weight")
        with safe_open(self.path / self.index[name], framework="pt") as f:
            return f.get_tensor(name)

    def put(self, value, *, dim=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=(
                ttnn.ReplicateTensorToMesh(self.device) if dim is None else ttnn.ShardTensorToMesh(self.device, dim)
            ),
        )

    def embed(self, tokens):
        x = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, [1, 1, tokens.shape[-1], self.config.hidden_size // 4])
        return ttnn.typecast(
            ttnn.all_gather(x, dim=3, num_links=1, cluster_axis=1, topology=ttnn.Topology.Linear),
            ttnn.float32,
        )

    def rope(self, indices, *, batch=None):
        result = [ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT) for table in (self.cos_table, self.sin_table)]
        shape = [1, 1, indices.shape[-1], 128] if batch is None else [1, batch, 1, 128]
        return [ttnn.reshape(t, shape) for t in result]

    def logits(self, hidden):
        return self.project(self.normalize(hidden))

    def normalize(self, hidden):
        return self.layers[0]._norm(ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG), self.norm)

    def project(self, normalized, mode="prefill"):
        return self.layers[0]._linear(normalized, self.head, group="lm_head", mode=mode)

    def prefill(self, tokens, indices, start, metadata, page_table, batch=1, sdpa_chunks=(32, 32)):
        x = self.embed(tokens)
        cs, sn = self.rope(indices)
        for layer in self.layers:
            if batch == 1:
                x = layer.prefill_prepared(
                    x,
                    chunk_start=start,
                    fill_metadata=metadata,
                    page_table=page_table,
                    cos=cs,
                    sin=sn,
                    sdpa_chunks=sdpa_chunks,
                )
            else:
                x = layer.prefill_batch_prepared(
                    x,
                    batch=batch,
                    chunk_start=start,
                    fill_metadata=metadata,
                    page_table=page_table,
                    cos=cs,
                    sin=sn,
                    sdpa_chunks=sdpa_chunks,
                )
        return x

    def decode(self, tokens, positions, indices, active, page_table, batch):
        x = self.embed(tokens)[:, :, :batch, :]
        cs, sn = self.rope(indices, batch=batch)
        for layer in self.layers:
            x = layer.decode_forward(
                x,
                cur_pos=positions,
                page_table=page_table,
                active_rows=active,
                cos=cs,
                sin=sn,
            )
        return self.project(self.normalize(x), mode="decode")
