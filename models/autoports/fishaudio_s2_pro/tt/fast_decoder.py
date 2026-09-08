"""TTNN fast codebook decoder (Phase B): fish-speech's 4-layer "fast" tower on device, built from
models/common/modules 1D blocks (Attention1D, MLP1D, RMSNorm1D, RotarySetup1D, Embedding1D, LMHead1D).

Per frame: step 0 consumes the slow tower's post-norm hidden state (output discarded, it only fills the KV
cache), steps 1..9 consume the embedding of the previous codebook token and produce 4096-way logits. Each
layer keeps a 32-position KV cache (positions 0..9 are rewritten every frame; the causal decode only reads
<= cur_pos so no reset is needed). Weights are Meta-format (fish's own convention) so no permutation.

Sampling is on host in Phase B (logits row read back per step). Trace: two captured bodies (step0 / step),
warmed before capture; the position tensors advance on device with ttnn.plus_one and are reset per frame with
a host copy (outside the trace).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

import ttnn
from models.autoports.fishaudio_s2_pro.config import S2Config
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig, _dram_shard_core_grid_k_n
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig, _create_sharded_norm_program_config
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D, prepare_rot_idxs
from models.common.modules.tt_ccl import default_topology, get_tt_ccl
from models.common.tensor_utils import TILE_SIZE, get_padded_hidden_dim

KV_LEN = 32  # >= num_codebooks (10), multiple of 32 for SDPA decode
ROPE_LEN = 64


def _rope_tables(head_dim: int, theta: float, n: int = ROPE_LEN):
    """Meta/interleaved cos-sin tables [1,1,n,head_dim] (same math as fish's precompute_freqs_cis)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = torch.outer(torch.arange(n).float(), inv)
    mk = lambda t: torch.stack([t, t], -1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return mk(f.cos()), mk(f.sin())


def _meta_wqkv(wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, nd: int) -> torch.Tensor:
    """[1,1,dim, q+2kv] with per-device [q_i|k_i|v_i] blocks (weights already interleaved => no permute)."""
    q, k, v = wq.T, wk.T, wv.T
    per = [
        torch.cat([torch.chunk(q, nd, 1)[i], torch.chunk(k, nd, 1)[i], torch.chunk(v, nd, 1)[i]], -1) for i in range(nd)
    ]
    return torch.cat(per, -1).unsqueeze(0).unsqueeze(0).contiguous()


def _all_gather_if_tp(norm: RMSNorm1D, x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
    cfg = norm.config
    if cfg.mesh_device.get_num_devices() == 1 or x.shape[-1] == cfg.weight.source.numel():
        return x
    tt_ccl = cfg.tt_ccl or get_tt_ccl(cfg.mesh_device)
    return ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=1,
        topology=default_topology(cfg.mesh_device),
        memory_config=memory_config or x.memory_config(),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=24,
        num_workers_per_link=4,
        num_buffers_per_channel=2,
    )


class TTFastDecoder:
    def __init__(
        self,
        mesh_device,
        fast_sd: Dict[str, torch.Tensor],
        cfg: S2Config,
        *,
        weights_dtype=ttnn.bfloat8_b,
        cache_dir: Optional[str] = None,
        use_trace: bool = True,
        log: Callable[[str], None] = print,
    ):
        self.mesh, self.cfg, self.log = mesh_device, cfg, log
        fc = cfg.fast
        nd = self.nd = mesh_device.get_num_devices()
        assert fc.n_head % nd == 0 and fc.n_local_heads % nd == 0, (fc.n_head, fc.n_local_heads, nd)
        self.ccl = get_tt_ccl(mesh_device) if nd > 1 else None
        self.topo = default_topology(mesh_device) if nd > 1 else None
        self.use_trace = use_trace
        cache = (
            Path(cache_dir or os.environ.get("TT_DIT_CACHE_DIR", Path.home() / ".cache" / "fish_s2_pro"))
            / "fast_decoder"
            / f"{weights_dtype}".replace("DataType.", "")
            / f"{nd}dev"
        )
        cache.mkdir(parents=True, exist_ok=True)

        def lazy(t, name, dtype=weights_dtype):
            return LazyWeight(source=t.contiguous(), dtype=dtype, cache_dir_weight_name=(cache, name))

        w = fast_sd
        cos_t, sin_t = _rope_tables(fc.head_dim, fc.rope_base)
        self.rope = RotarySetup1D.from_config(
            Rope1DConfig(
                cos_matrix=LazyWeight(source=cos_t, dtype=ttnn.bfloat16),
                sin_matrix=LazyWeight(source=sin_t, dtype=ttnn.bfloat16),
                max_batch_size=1,
                head_dim=fc.head_dim,
                device=mesh_device,
                use_qk_fused=False,
                core_grid=ttnn.CoreCoord(8, 8),
            )
        )
        self.emb = Embedding1D.from_config(
            Embedding1DConfig(
                weights=LazyWeight(
                    source=w["embeddings.weight"].unsqueeze(0).unsqueeze(0).contiguous(), dtype=ttnn.bfloat16
                ),
                mesh_device=mesh_device,
            )
        )
        padded_hidden = get_padded_hidden_dim(fc.intermediate_size, nd, TILE_SIZE)
        ffn_grid = _dram_shard_core_grid_k_n(fc.dim, padded_hidden // nd)
        post_pc = _create_sharded_norm_program_config(fc.dim, ffn_grid, TILE_SIZE, TILE_SIZE)
        post_mc = ttnn.create_sharded_memory_config(
            (TILE_SIZE, fc.dim // ffn_grid.num_cores),
            ffn_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.layers = []
        for i in range(fc.n_layer):
            p = f"layers.{i}."
            attn = Attention1D.from_config(
                Attention1DConfig(
                    wqkv=lazy(
                        _meta_wqkv(
                            w[p + "attention.wq.weight"], w[p + "attention.wk.weight"], w[p + "attention.wv.weight"], nd
                        ),
                        f"l{i}_wqkv",
                    ),
                    wo=lazy(w[p + "attention.wo.weight"].T.unsqueeze(0).unsqueeze(0), f"l{i}_wo"),
                    mesh_device=mesh_device,
                    tt_ccl=self.ccl,
                    topology=self.topo,
                    n_heads=fc.n_head,
                    n_kv_heads=fc.n_local_heads,
                    head_dim=fc.head_dim,
                    max_batch_size=1,
                    max_seq_len=KV_LEN,
                    kv_cache_dtype=ttnn.bfloat8_b,
                    paged_attention_config=None,
                    use_vllm_paged_kv_cache=False,
                    q_norm_config=None,
                    k_norm_config=None,
                    use_qk_fused=False,
                )
            )
            mlp = MLP1D.from_config(
                MLP1DConfig(
                    w1=lazy(w[p + "feed_forward.w1.weight"].T, f"l{i}_w1"),
                    w2=lazy(w[p + "feed_forward.w2.weight"].T, f"l{i}_w2"),
                    w3=lazy(w[p + "feed_forward.w3.weight"].T, f"l{i}_w3"),
                    mesh_device=mesh_device,
                    tt_ccl=self.ccl,
                    topology=self.topo,
                    max_batch_size=1,
                )
            )
            n1 = RMSNorm1D.from_config(
                RMSNorm1DConfig(
                    weight=LazyWeight(source=w[p + "attention_norm.weight"], dtype=ttnn.bfloat16),
                    mesh_device=mesh_device,
                    tt_ccl=self.ccl,
                    eps=fc.norm_eps,
                    max_batch_size=1,
                )
            )
            n2 = RMSNorm1D.from_config(
                RMSNorm1DConfig(
                    weight=LazyWeight(source=w[p + "ffn_norm.weight"], dtype=ttnn.bfloat16),
                    mesh_device=mesh_device,
                    tt_ccl=self.ccl,
                    eps=fc.norm_eps,
                    max_batch_size=1,
                    decode_program_config=post_pc,
                    decode_memory_config=post_mc,
                )
            )
            self.layers.append((n1, attn, n2, mlp))
        self.final_norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=LazyWeight(source=w["norm.weight"], dtype=ttnn.bfloat16),
                mesh_device=mesh_device,
                tt_ccl=self.ccl,
                eps=fc.norm_eps,
                max_batch_size=1,
            )
        )
        self.lm_head = LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=[lazy(w["output.weight"].T, "output")],
                mesh_device=mesh_device,
                dim=fc.dim,
                max_batch_size=1,
            )
        )
        # persistent device inputs
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self.x0_dev = ttnn.from_torch(
            torch.zeros(1, 1, 32, fc.dim, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.tok_dev = ttnn.from_torch(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )
        self.cur_pos = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )
        self.rot_idxs = prepare_rot_idxs(self.rope.config, torch.zeros(1, dtype=torch.int64), on_host=False)
        self._pos0_host = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
        )
        self._rot0_host = prepare_rot_idxs(self.rope.config, torch.zeros(1, dtype=torch.int64), on_host=True)
        self.traces: Dict[str, tuple] = {}
        self.warmed = False

    # ------------------------------------------------------------------ graph pieces
    def _layers_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        rot = self.rope.decode_forward(self.rot_idxs)
        for n1, attn, n2, mlp in self.layers:
            r = ttnn.to_memory_config(x, n1.config.decode_memory_config)
            a = attn.decode_forward(n1.decode_forward(_all_gather_if_tp(n1, r)), self.cur_pos, rot, page_table=None)
            x = ttnn.add(x, ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            h = n2.decode_forward(_all_gather_if_tp(n2, ttnn.to_memory_config(x, n2.config.decode_memory_config)))
            x = ttnn.add(
                x,
                ttnn.to_memory_config(mlp.decode_forward(h), ttnn.DRAM_MEMORY_CONFIG),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return x

    def _advance(self):
        ttnn.plus_one(self.cur_pos, skip_negative_entries=True)
        ttnn.plus_one(self.rot_idxs)

    def _body_step0(self):
        x = self._layers_forward(self.x0_dev)
        self._advance()
        return x

    def _body_step(self):
        x = ttnn.to_memory_config(ttnn.unsqueeze_to_4D(self.emb.forward(self.tok_dev)), ttnn.DRAM_MEMORY_CONFIG)
        x = self._layers_forward(x)
        h = self.final_norm.decode_forward(
            _all_gather_if_tp(self.final_norm, ttnn.to_memory_config(x, self.final_norm.config.decode_memory_config))
        )
        logits = self.lm_head.forward(ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG))
        if self.nd > 1:
            logits = ttnn.experimental.all_gather_async(
                logits,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                memory_config=logits.memory_config(),
                topology=self.topo,
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
        logits = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._advance()
        return logits

    # ------------------------------------------------------------------ host <-> device
    def _reset_positions(self):
        ttnn.copy_host_to_device_tensor(self._pos0_host, self.cur_pos)
        ttnn.copy_host_to_device_tensor(self._rot0_host, self.rot_idxs)

    def _set_hidden(self, hidden: torch.Tensor):
        t = torch.zeros(1, 1, 32, self.cfg.fast.dim, dtype=torch.bfloat16)
        t[0, 0, 0] = hidden.to(torch.bfloat16)
        host = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(host, self.x0_dev)

    def _set_token(self, code: int):
        t = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        t[0, 0, 0, 0] = int(code)
        host = ttnn.from_torch(
            t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(host, self.tok_dev)

    def _read_logits(self, logits: ttnn.Tensor) -> torch.Tensor:
        t = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float()
        return t.reshape(-1, t.shape[-1])[0, : self.cfg.codebook_size]

    # ------------------------------------------------------------------ warmup / trace
    def warmup(self):
        """Compile every op (eager pass) and, if enabled, capture the two traces. Call once."""
        self._reset_positions()
        self._set_hidden(torch.zeros(self.cfg.fast.dim))
        self._set_token(0)
        self._body_step0()
        for _ in range(2):
            self._body_step()
        ttnn.synchronize_device(self.mesh)
        if self.use_trace:
            self._reset_positions()
            tid0 = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            out0 = self._body_step0()
            ttnn.end_trace_capture(self.mesh, tid0, cq_id=0)
            tid1 = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            out1 = self._body_step()
            ttnn.end_trace_capture(self.mesh, tid1, cq_id=0)
            ttnn.synchronize_device(self.mesh)
            self.traces = {"step0": (tid0, out0), "step": (tid1, out1)}
            self.log("fast decoder: traces captured (step0, step)")
        self.warmed = True

    def _run(self, name: str) -> ttnn.Tensor:
        if self.use_trace and self.traces:
            tid, out = self.traces[name]
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=True)
            return out
        return self._body_step0() if name == "step0" else self._body_step()

    # ------------------------------------------------------------------ public API (same as TorchFastDecoder)
    @torch.inference_mode()
    def frame(self, hidden: torch.Tensor, code0: int, choose, record: Optional[List] = None) -> List[int]:
        if not self.warmed:
            self.warmup()
        self._reset_positions()
        self._set_hidden(hidden.reshape(-1))
        self._run("step0")
        self._set_token(code0)
        out = []
        for i in range(1, self.cfg.num_codebooks):
            logits = self._read_logits(self._run("step"))
            if record is not None:
                record.append(logits)
            c = int(choose(logits, i))
            out.append(c)
            self._set_token(c)
        return out

    def release(self):
        for tid, _ in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
        self.traces = {}
