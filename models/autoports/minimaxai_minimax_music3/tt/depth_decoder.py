"""TTNN depth (local) LLM: 4 layers, hidden 4096, 16 MHA heads x 256, SwiGLU 6144, learned positions, 7 heads of 1024.

Per frame the reference runs 7 dependent steps over a sequence that grows from 2 to 8 tokens. Here every step
recomputes the whole (<= 8-token) sequence on ONE 32-row tile with causal SDPA, batch 2 (conditional / unconditional
rows): the graph is identical for all steps, so ONE trace is captured and replayed 7x per frame. Inputs are the RAW
(un-projected) embedding rows written by the host; the `projection`, positions, layers, final norm and all 7 heads
(stacked into one [4096, 7168] matmul) run on device. Sampling is on host in phase A.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import torch

import ttnn
from models.autoports.minimaxai_minimax_music3.config import (
    AUDIO_CODE_OFFSET,
    AUDIO_VOCAB_SIZE,
    NUM_CODEBOOKS,
    DepthConfig,
)
from models.autoports.minimaxai_minimax_music3.tt.weights import cache_root
from models.tt_dit.utils.matmul import get_matmul_config, get_matmul_core_grid
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

MATMUL_MODE = os.environ.get("MUSIC3_DEPTH_MATMUL", "minimal")  # minimal (tt_dit minimal_matmul, fused SwiGLU) | linear

ROWS = 32  # one tile; positions 0..7 are used


def _dtype_tag(dt) -> str:
    return str(dt).replace("DataType.", "").lower()


class TTDepthDecoder:
    def __init__(
        self,
        mesh_device,
        state: Dict[str, torch.Tensor],
        cfg: DepthConfig,
        *,
        weights_dtype=ttnn.bfloat16,
        full_embed: Optional[torch.Tensor] = None,
        use_trace: bool = True,
        log: Callable[[str], None] = print,
    ):
        self.mesh, self.cfg, self.log, self.use_trace = mesh_device, cfg, log, use_trace
        assert mesh_device.get_num_devices() == 1, "TTDepthDecoder runs on one chip"
        D, H, I = cfg.hidden_size, cfg.num_attention_heads, cfg.intermediate_size
        self.n_heads, self.head_dim = H, cfg.head_dim
        cache = cache_root() / "depth_decoder" / _dtype_tag(weights_dtype)
        cache.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        def dev(t: torch.Tensor, name: str, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=str(cache / name),
            )

        w = state
        self.w_proj = dev(w["projection.weight"].T, "projection")  # [D, D]
        pos = torch.zeros(1, 1, ROWS, D, dtype=torch.float32)
        pos[0, 0, : cfg.max_position_embeddings] = w["pos_embedding.weight"].float()
        self.pos_emb = dev(pos, "pos_emb", dtype=ttnn.bfloat16)
        self.layers = []
        for i in range(cfg.num_layers):
            p = f"layers.{i}."
            qkv = torch.cat(
                [w[p + "attn.to_q.weight"], w[p + "attn.to_k.weight"], w[p + "attn.to_v.weight"]], dim=0
            ).T  # [D, 3D]
            self.layers.append(
                {
                    "norm1": dev(
                        w[p + "input_layernorm.weight"].reshape(1, 1, 1, D), f"l{i}_norm1", dtype=ttnn.bfloat16
                    ),
                    "wqkv": dev(qkv, f"l{i}_wqkv"),
                    "wo": dev(w[p + "attn.to_out.weight"].T, f"l{i}_wo"),
                    "norm2": dev(
                        w[p + "post_attention_layernorm.weight"].reshape(1, 1, 1, D), f"l{i}_norm2", dtype=ttnn.bfloat16
                    ),
                    "w_gate": dev(w[p + "gate_proj.weight"].T, f"l{i}_wgate"),
                    "w_up": dev(w[p + "up_proj.weight"].T, f"l{i}_wup"),
                    # packed [up | gate] re-interleaved for the fused SwiGLU matmul epilogue: out = up * silu(gate)
                    "w_ffn_fused": dev(
                        prepare_for_fused_swiglu(
                            torch.cat([w[p + "up_proj.weight"], w[p + "gate_proj.weight"]], dim=0).T.float(),
                            ndev=1,
                            gate_is_first=False,
                        ),
                        f"l{i}_wffn_fused",
                    ),
                    "w_down": dev(w[p + "down_proj.weight"].T, f"l{i}_wdown"),
                }
            )
        self.norm_f = dev(w["norm.weight"].reshape(1, 1, 1, D), "norm_f", dtype=ttnn.bfloat16)
        heads = torch.cat([w[f"audio_heads.{i}.weight"] for i in range(NUM_CODEBOOKS - 1)], dim=0).T  # [D, 7*1024]
        self.w_heads = dev(heads, "heads")
        # host tables for the raw input rows
        self.audio_embeddings = w["audio_embeddings.weight"].to(torch.bfloat16)  # [7*1024, D]
        self.full_embed = full_embed.to(torch.bfloat16) if full_embed is not None else None
        self.eps = 1e-6
        arch = mesh_device.arch()
        self.ck_hifi4 = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ck_hifi2 = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        fid = os.environ.get("MUSIC3_DEPTH_FIDELITY", "hifi4" if weights_dtype == ttnn.bfloat16 else "hifi2")
        self.ck_mm = self.ck_hifi4 if fid == "hifi4" else self.ck_hifi2
        self.matmul_mode = MATMUL_MODE
        self.mm_grid = get_matmul_core_grid(mesh_device)
        # persistent I/O: the input stays ROW_MAJOR on device (host writes are a memcpy, tilize happens in the trace);
        # one trace per step index slices the needed row / head columns on device so the readback is ~20 KB
        self.x_raw = ttnn.from_torch(
            torch.zeros(2, 1, ROWS, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self._host_rows = torch.zeros(2, 1, ROWS, D, dtype=torch.bfloat16)
        self.traces: Dict[int, tuple] = {}  # step index -> (trace id, h_row, logits_row)
        self.warmed = False
        self.log(f"depth decoder weights on device in {time.time() - t0:.1f}s ({_dtype_tag(weights_dtype)})")

    # ------------------------------------------------------------------ graph
    def _lin(self, x, w, act=None, swiglu=False):
        if self.matmul_mode == "minimal":
            M, K, N = x.padded_shape[-2], x.padded_shape[-1], w.padded_shape[-1]
            return ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=w,
                config=get_matmul_config(M, K, N, self.mm_grid),
                fused_activation=act,
                compute_kernel_config=self.ck_mm,
                dtype=ttnn.bfloat16,
                fuse_swiglu=swiglu,
            )
        assert not swiglu
        return ttnn.linear(x, w, activation=act, compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16)

    def _forward(self, x_raw: ttnn.Tensor):
        x = ttnn.to_layout(x_raw, ttnn.TILE_LAYOUT)
        x = self._lin(x, self.w_proj)
        x = ttnn.add(x, self.pos_emb)
        for L in self.layers:
            h = ttnn.rms_norm(x, epsilon=self.eps, weight=L["norm1"])
            qkv = self._lin(h, L["wqkv"])
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=self.n_heads, num_kv_heads=self.n_heads, transpose_k_heads=False
            )
            a = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, compute_kernel_config=self.ck_hifi4
            )
            a = ttnn.experimental.nlp_concat_heads(a)
            x = ttnn.add(x, self._lin(a, L["wo"]))
            h = ttnn.rms_norm(x, epsilon=self.eps, weight=L["norm2"])
            if self.matmul_mode == "minimal":
                ff = self._lin(h, L["w_ffn_fused"], swiglu=True)  # up * silu(gate) in one matmul
            else:
                ff = ttnn.multiply(self._lin(h, L["w_gate"], act="silu"), self._lin(h, L["w_up"]))
            x = ttnn.add(x, self._lin(ff, L["w_down"]))
        h = ttnn.rms_norm(x, epsilon=self.eps, weight=self.norm_f)
        logits = self._lin(h, self.w_heads)
        return h, logits

    def _step_outputs(self, index: int):
        """Full forward + on-device row/head selection for step `index` (1..7): h_row [2,1,1,D], logits_row [2,1,1,1024]."""
        h, lg = self._forward(self.x_raw)
        h_rm, lg_rm = ttnn.untilize(h, use_multicore=True), ttnn.untilize(lg, use_multicore=True)
        h_row = ttnn.slice(h_rm, [0, 0, index, 0], [2, 1, index + 1, self.cfg.hidden_size])
        lg_row = ttnn.slice(
            lg_rm, [0, 0, index, (index - 1) * AUDIO_VOCAB_SIZE], [2, 1, index + 1, index * AUDIO_VOCAB_SIZE]
        )
        return h_row, lg_row

    def warmup(self):
        self._push_rows()
        for index in range(1, NUM_CODEBOOKS):
            self._step_outputs(index)  # compile pass (all 7 slice shapes)
        ttnn.synchronize_device(self.mesh)
        if self.use_trace:
            for index in range(1, NUM_CODEBOOKS):
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                outs = self._step_outputs(index)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                self.traces[index] = (tid, *outs)
            ttnn.synchronize_device(self.mesh)
            self.log("depth decoder: 7 step traces captured")
        self.warmed = True

    def _run_step(self, index: int):
        if self.traces:
            tid, h_row, lg_row = self.traces[index]
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=True)
            return h_row, lg_row
        return self._step_outputs(index)

    # ------------------------------------------------------------------ host <-> device
    def _push_rows(self):
        host = ttnn.from_torch(
            self._host_rows,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, self.x_raw)

    def _read_rows(self, h_row, lg_row):
        ht = ttnn.to_torch(ttnn.get_device_tensors(h_row)[0]).float().reshape(2, -1)  # [2, D]
        lt = ttnn.to_torch(ttnn.get_device_tensors(lg_row)[0]).float().reshape(2, -1)  # [2, 1024]
        return ht, lt

    # ------------------------------------------------------------------ public API (mirrors reference.pipeline.depth_codes)
    @torch.inference_mode()
    def frame(
        self,
        last_hidden: torch.Tensor,
        semantic_code: int,
        choose: Callable[[torch.Tensor, int], int],
        forced: Optional[List[int]] = None,
        record: Optional[dict] = None,
    ):
        """last_hidden [2, D] (cond, uncond) -> (codes [8] incl. c0, depth_hidden [7, D] (cond row)).
        `choose(guided_logits [1,1024], index) -> code` samples on host; `forced` overrides sampling (teacher forcing).
        """
        if not self.warmed:
            self.warmup()
        rows = self._host_rows
        rows.zero_()
        rows[:, 0, 0] = last_hidden.to(torch.bfloat16)
        rows[:, 0, 1] = self.full_embed[semantic_code + AUDIO_CODE_OFFSET]
        codes = [int(semantic_code)]
        hidden_parts = []
        for index in range(1, NUM_CODEBOOKS):
            self._push_rows()
            ht, logits = self._read_rows(*self._run_step(index))  # [2, D], [2, 1024] (head index-1)
            hidden_parts.append(ht[:1])
            if record is not None:
                record.setdefault("depth_logits", []).append(logits.clone())
                record.setdefault("depth_hidden", []).append(ht[:1].clone())
            from models.autoports.minimaxai_minimax_music3.reference.sampling import guided_depth_logits

            code = int(forced[index - 1]) if forced is not None else int(choose(guided_depth_logits(logits), index))
            codes.append(code)
            if index < NUM_CODEBOOKS - 1:
                rows[:, 0, index + 1] = self.audio_embeddings[code + (index - 1) * AUDIO_VOCAB_SIZE]
        return torch.tensor(codes, dtype=torch.int64), torch.cat(hidden_parts, dim=0)

    @torch.inference_mode()
    def forward_rows(self, inputs_embeds: torch.Tensor):
        """Unit-test entry: raw (un-projected) rows [2, steps, D] -> (hidden [2, steps, D], logits [7, 2, 1024] at the last step)."""
        if not self.warmed:
            self.warmup()
        steps = inputs_embeds.shape[1]
        self._host_rows.zero_()
        self._host_rows[:, 0, :steps] = inputs_embeds.to(torch.bfloat16)
        self._push_rows()
        h, lg = self._forward(self.x_raw)  # untraced full outputs
        ht = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float()[:, 0, :steps]
        lt = (
            ttnn.to_torch(ttnn.get_device_tensors(lg)[0])
            .float()[:, 0, steps - 1]
            .view(2, NUM_CODEBOOKS - 1, AUDIO_VOCAB_SIZE)
            .transpose(0, 1)
        )
        return ht, lt

    def release(self):
        for tid, *_ in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
        self.traces = {}
