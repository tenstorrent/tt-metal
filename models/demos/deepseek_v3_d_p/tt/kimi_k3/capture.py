# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Teacher capture for GPTQ calibration (Kimi-K3 / K3.1 prefill), capture format v0.

Off unless K3_CAPTURE_DIR is set: every tap below is then a no-op (one module-global check) and no tensor is read.
On, the eager prefill path (TtKimiK3Runtime.prefill_chunk) records, per layer of the stage:

- dense capture points: the Gram matrix sum_t x_t x_t^T of the activation x each GPTQ'd tensor multiplies, fp32
  per-chunk partials accumulated in fp64 on the host (a worker thread overlaps them with the device);
- routed experts: the latent rows the experts' gate/up consume (the MoE head's dispatch rows, bf16), the top-16
  expert ids (checkpoint numbering), the combine's bf16 routing weights (as float32) and the (token, slot) count of
  every expert.

    K3_CAPTURE_DIR      root; the run writes <root>/<run_id>/ (layer_LL.tmp/ renamed to layer_LL/ when complete)
    K3_CAPTURE_RUN      run id (default: UTC time)
    K3_CAPTURE_LAYERS   comma list of global layer indices (default: every layer of the stage)
    K3_CAPTURE_RAW      "1": also write every dense point's raw rows (fp32, validation only: large)
    K3_CAPTURE_MANIFEST / K3_CAPTURE_TOKENS_SHA256   recorded in run.json (the calibration driver sets them)

Points (capture_format.md): attn_in, kda_fb_in, kda_o_in, mla_qb_in, mla_kvb1_in (per head), mla_kvb2_in (per head),
mla_o_in, moe_in / mlp0_in, mlp0_down_in, latent_up_in (y_own RMS-normalised WITHOUT routed_expert_norm's gamma: both
stacks fold it into the up projection), shared_down_in (act(gate) * up of the shared experts recomputed with extra ttnn
ops from moe_in and the bf16 checkpoint weights: the fused MoE head never writes it), lm_head_in (final norm output).
Padding rows (past the chunk's actual_end) are excluded everywhere. Eager, unpacked forwards only; the MoE block
with one split and the SP axis on mesh axis 0.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from loguru import logger

import ttnn

VERSION = "v0"
CAPTURE = None  # the active TeacherCapture, None when K3_CAPTURE_DIR is unset
LM_HEAD = "lm_head"  # the lm_head_in point's "layer" (written to <run>/lm_head/)
PAUSED = False  # inside `paused()`: the runtime's warm-up, recorded by no hook (capture or logits)

# Layouts of a tapped device tensor (per chip: [..., rows, cols]):
RES = "res"  # [1, 1, S/SP, K/TP]: rows over SP, columns over TP
REP = "rep"  # [1, 1, S/SP, K]: rows over SP, replicated over TP
HEADS = "heads"  # [1, H/TP, S/SP, D]: heads over TP, rows over SP (a per-head point)

# Checkpoint tensor name (under model.layers.L.) -> capture point.
_KDA_MAP = {
    **{f"self_attn.{n}.weight": "attn_in" for n in ("q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "f_a_proj")},
    "self_attn.f_b_proj.weight": "kda_fb_in",
    "self_attn.o_proj.weight": "kda_o_in",
}
_MLA_MAP = {
    **{
        f"self_attn.{n}.weight": "attn_in"
        for n in ("q_a_proj", "kv_a_proj_with_mqa", "g_proj", "indexer.wk", "indexer.weights_proj")
    },
    "self_attn.q_b_proj.weight": "mla_qb_in",
    "self_attn.indexer.wq_b.weight": "mla_qb_in",
    "self_attn.kv_b_proj.weight[k_nope rows]": "mla_kvb1_in",
    "self_attn.kv_b_proj.weight[v rows]": "mla_kvb2_in",
    "self_attn.o_proj.weight": "mla_o_in",
}
_MOE_MAP = {
    "block_sparse_moe.routed_expert_down_proj.weight": "moe_in",
    "block_sparse_moe.shared_experts.gate_proj.weight": "moe_in",
    "block_sparse_moe.shared_experts.up_proj.weight": "moe_in",
    "block_sparse_moe.shared_experts.down_proj.weight": "shared_down_in",
    "block_sparse_moe.routed_expert_up_proj.weight": "latent_up_in",
    "block_sparse_moe.experts.*.w1/w3/w2": "moe_x + moe_ids + moe_w (routed)",
}
_MLP0_MAP = {
    "mlp.gate_proj.weight": "mlp0_in",
    "mlp.up_proj.weight": "mlp0_in",
    "mlp.down_proj.weight": "mlp0_down_in",
}


def active():
    """The active capture, or None (production)."""
    return CAPTURE


def start(runtime):
    """Turn the capture on for this runtime's stage when K3_CAPTURE_DIR is set (TtKimiK3Runtime._build_model)."""
    global CAPTURE
    root = os.environ.get("K3_CAPTURE_DIR")
    if not root:
        return None
    if CAPTURE is not None:
        CAPTURE.finish()
    CAPTURE = TeacherCapture(Path(root), runtime)
    return CAPTURE


class paused:
    """`with capture.paused():` - forwards inside record nothing (the runtime's warm-up)."""

    def __enter__(self):
        global PAUSED
        self.prev, PAUSED = PAUSED, True

    def __exit__(self, *exc):
        global PAUSED
        PAUSED = self.prev


def finish():
    """Write every captured layer and run.json (also at interpreter exit)."""
    global CAPTURE
    if CAPTURE is not None:
        CAPTURE.finish()
        CAPTURE = None


atexit.register(finish)


def _key(layer_idx):
    return layer_idx if layer_idx == LM_HEAD else int(layer_idx)


def _commit() -> str:
    try:
        home = os.environ.get("TT_METAL_HOME") or str(Path(__file__).resolve().parents[5])
        return subprocess.check_output(["git", "-C", home, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # (not a checkout)
        return "unknown"


class _Gram:
    """sum x^T x (fp64) of one point: [K, K], or [H, K, K] for a per-head point."""

    seconds = 0.0  # (worker time, all points)

    def __init__(self):
        self.g = None
        self.n = 0
        self.dtype = None

    def add(self, x: torch.Tensor, src_dtype: str):
        t0 = time.time()
        x = x.float()
        part = torch.einsum("thk,thl->hkl", x, x) if x.dim() == 3 else x.T @ x
        part = part.double()
        if self.g is None:
            self.g = part
        else:
            self.g += part
        self.n += int(x.shape[0])
        self.dtype = src_dtype
        _Gram.seconds += time.time() - t0


class TeacherCapture:
    def __init__(self, root: Path, runtime):
        cfg = runtime.config
        self.mesh = runtime.mesh_device
        self.shape = tuple(self.mesh.shape)
        self.sp_axis, self.tp_axis = int(cfg.sp_axis), int(cfg.tp_axis)
        assert self.sp_axis == 0, "K3_CAPTURE: v0 supports the SP axis on mesh axis 0 (the MoE block's owner rows)"
        self.chunk = int(cfg.chunk_size)
        self.first, self.num = int(cfg.first_layer_idx), int(cfg.num_layers)
        self.model_cfg = cfg.model_cfg
        want = [x.strip() for x in os.environ.get("K3_CAPTURE_LAYERS", "").split(",") if x.strip()]
        self.layers = {x if x == LM_HEAD else int(x) for x in want} if want else set(range(self.first, self.first + self.num))
        if not want and cfg.is_last_rank:
            self.layers.add(LM_HEAD)
        self.run_id = os.environ.get("K3_CAPTURE_RUN") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.dir = root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.raw = os.environ.get("K3_CAPTURE_RAW", "") not in ("", "0")
        self.grams: dict = {}  # (layer, point) -> _Gram
        self.moe_files: dict = {}  # layer -> {name: open file}
        self.counts: dict = {}  # layer -> int64 [E]
        self.ntok: dict = {}  # layer -> tokens captured
        self.valid = None  # this chunk's real rows
        self.n_tokens = self.n_sequences = self.n_chunks = 0
        self.t_device = self.t_host = 0.0
        self.started = time.time()
        self.moe_layer = None
        self._pool = ThreadPoolExecutor(max_workers=1)  # (fp64 Gram accumulation beside the device)
        self._pending = []
        self._lock = threading.Lock()
        self._perm = None
        self._y_own = None
        self._shared_w: dict = {}
        logger.warning(
            f"K3_CAPTURE {VERSION}: {self.dir}, layers {sorted(self.layers)}, raw {self.raw} (eager prefill only)"
        )

    # --- chunk bracket (TtKimiK3Runtime.prefill_chunk) ---

    def begin_chunk(self, actual_start, actual_end):
        assert actual_start is None or isinstance(actual_start, int), "K3_CAPTURE: eager prefill only (host scalars)"
        start = int(actual_start or 0)
        end = start + self.chunk if actual_end is None else int(actual_end)
        self.valid = max(0, min(self.chunk, end - start))
        if start == 0:
            self.n_sequences += 1
        self._t0 = time.time()
        self._t_host_chunk = 0.0

    def end_chunk(self):
        self.n_tokens += self.valid
        self.n_chunks += 1
        dt = time.time() - self._t0
        self.t_host += self._t_host_chunk
        self.t_device += dt - self._t_host_chunk
        self.valid = None

    def wants(self, layer_idx) -> bool:
        return self.valid is not None and layer_idx is not None and _key(layer_idx) in self.layers

    # --- device -> host ---

    def _chips(self, t):
        """(sp, tp, per-chip host tensor) of every chip of a mesh tensor."""
        C = self.shape[1]
        out = []
        for n, d in enumerate(ttnn.get_device_tensors(t)):
            rc = divmod(n, C)
            out.append((rc[self.sp_axis], rc[self.tp_axis], ttnn.to_torch(d)))
        return out

    def to_host(self, t, layout: str) -> torch.Tensor:
        """A tapped tensor as host [valid rows, K] (or [valid rows, H, D] for HEADS) in token order."""
        SP, TP = self.shape[self.sp_axis], self.shape[self.tp_axis]
        chips = self._chips(t)
        if layout == HEADS:
            per = {}
            for sp, tp, x in chips:
                x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [H/TP, S/SP, D]
                per[(sp, tp)] = x
            rows = torch.cat([torch.cat([per[(sp, tp)] for tp in range(TP)], 0) for sp in range(SP)], 1)
            return rows.permute(1, 0, 2)[: self.valid]  # [S, H, D]
        per = {}
        for sp, tp, x in chips:
            per[(sp, tp)] = x.reshape(-1, x.shape[-1])
        if layout == REP:
            rows = torch.cat([per[(sp, 0)] for sp in range(SP)], 0)
        else:
            rows = torch.cat([torch.cat([per[(sp, tp)] for tp in range(TP)], 1) for sp in range(SP)], 0)
        return rows[: self.valid]

    # --- taps ---

    def tap(self, layer_idx, point: str, t, layout: str = RES):
        """A dense capture point's device tensor: its Gram (and raw rows with K3_CAPTURE_RAW)."""
        if not self.wants(layer_idx):
            return
        t0 = time.time()
        x = self.to_host(t, layout)
        self._t_host_chunk += time.time() - t0
        self._add(_key(layer_idx), point, x, str(t.dtype))

    def tap_q_nope(self, layer_idx, tt_q, head_dim: int, nope_dim: int):
        """mla_kvb1_in: q_b's output [1, 1, S/SP, H/TP * head_dim] (each head's nope columns first) as the per-head
        q_nope [S, H, nope_dim] that kv_b1 (W_UK) multiplies."""
        if not self.wants(layer_idx):
            return
        t0 = time.time()
        q = self.to_host(tt_q, RES)
        q = q.reshape(q.shape[0], -1, head_dim)[..., :nope_dim]
        self._t_host_chunk += time.time() - t0
        self._add(int(layer_idx), "mla_kvb1_in", q, str(tt_q.dtype))

    def tap_host(self, layer_idx, point: str, x: torch.Tensor, src_dtype: str):
        """A capture point computed on the host (already [valid rows, K])."""
        if self.wants(layer_idx):
            self._add(_key(layer_idx), point, x, src_dtype)

    def _add(self, layer: int, point: str, x: torch.Tensor, src_dtype: str):
        key = (layer, point)
        g = self.grams.setdefault(key, _Gram())
        if self.raw:
            d = self._layer_tmp(layer)
            with open(d / f"raw_{point}.f32", "ab") as f:
                f.write(x.float().contiguous().numpy().tobytes())
        self._pending.append(self._pool.submit(g.add, x, src_dtype))
        if len(self._pending) > 8:  # (bounded host memory: wait for the oldest)
            self._pending.pop(0).result()

    def _name(self, layer) -> str:
        return LM_HEAD if layer == LM_HEAD else f"layer_{layer:02d}"

    def _layer_tmp(self, layer) -> Path:
        d = self.dir / f"{self._name(layer)}.tmp"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- the MoE block (TtMoEBlockFfn.__call__) ---

    def moe(self, ffn, block, x_in):
        """After a MoE block call of layer ffn.layer_idx: the routing rows, the dispatch rows (moe_x), the combine's
        routed sum (latent_up_in), and shared_down_in from x_in (moe_in's device tensor)."""
        layer = ffn.layer_idx
        if not self.wants(layer):
            return
        t0 = time.time()
        assert block.splits == 1, "K3_CAPTURE: the MoE block with one split only"
        head = block.last_head
        K = ffn.dims.top_k
        R, C = self.shape
        if self._perm is None:
            from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import block_order_perm

            self._perm = block_order_perm(self.mesh, ffn.dims.num_experts // (R * C), ffn.options.sp_axis)
        if os.environ.get("K3_CAPTURE_DEBUG") and not getattr(self, "_debug_done", set()) & {layer}:
            # (debug: the raw per-chip MoE head / combine outputs of this layer's first chunk)
            self._debug_done = getattr(self, "_debug_done", set()) | {layer}
            torch.save(
                {
                    "dispatch": [ttnn.to_torch(d) for d in ttnn.get_device_tensors(head.dispatch_buffer)],
                    "routing_rows": [ttnn.to_torch(d) for d in ttnn.get_device_tensors(head.routing_rows)],
                    "y_own": self._y_own_raw,
                    "row_blocks": int(block.row_blocks),
                    "pad_rows": int(block.pad_rows),
                    "mesh": self.shape,
                },
                self.dir / f"debug_moe_L{layer}.pt",
            )
        words = self._owner_rows(head.routing_rows, block, 2 * K)  # [chunk, 2K] uint16 (ids | bf16 weights)
        ids_b = words[:, :K].to(torch.int64) & 0xFFFF
        ids = self._perm[ids_b.clamp(max=self._perm.numel() - 1)].to(torch.int16)
        w = (words[:, K:].to(torch.int32) & 0xFFFF).to(torch.int16).view(torch.bfloat16).float()
        latent = self._dispatch_rows(head.dispatch_buffer, block, ffn.dims.routed_width)
        y_own, y_dtype = self._y_own
        self._y_own = None
        n = self.valid
        ids, w, latent, y_own = ids[:n], w[:n], latent[:n], y_own[:n]
        d = self._layer_tmp(layer)
        files = self.moe_files.setdefault(layer, {})
        for name, arr in (
            ("moe_x.bf16", latent.to(torch.bfloat16).view(torch.int16)),
            ("moe_ids.i16", ids),
            ("moe_w.f32", w.float()),
        ):
            if name not in files:
                files[name] = open(d / name, "ab")
            files[name].write(arr.contiguous().numpy().tobytes())
        cnt = self.counts.setdefault(layer, torch.zeros(ffn.dims.num_experts, dtype=torch.int64))
        cnt += torch.bincount(ids.long().flatten(), minlength=ffn.dims.num_experts)
        self.ntok[layer] = self.ntok.get(layer, 0) + n
        # latent_up_in: the routed sum RMS-normalised (fp64), gamma NOT applied (folded into the up weight)
        y = y_own.double()
        eps = float(getattr(self.model_cfg, "RMS_NORM_EPS", 1e-5))
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
        self._t_host_chunk += time.time() - t0
        self._add(layer, "latent_up_in", y.float(), "fp64 host norm of the combine output " + y_dtype)
        self._shared_down_in(ffn, x_in)

    def y_own(self, block, y_own):
        """MoEBlock._forward, before the tail: the combine's routed sum (owner rows) to the host."""
        if self.moe_layer is None or not self.wants(self.moe_layer):
            return
        t0 = time.time()
        if os.environ.get("K3_CAPTURE_DEBUG"):
            self._y_own_raw = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(y_own)]
        self._y_own = (self._owner_rows(y_own, block, int(y_own.shape[-1])), str(y_own.dtype))
        self._t_host_chunk += time.time() - t0

    def _dispatch_rows(self, t, block, width) -> torch.Tensor:
        """The MoE head's dispatch buffer as [chunk, width] in token order. Every chip's buffer is a landing buffer of
        the whole chunk (row = the token's position in the chunk): the rows dispatched to that chip are written, the
        rest hold earlier data; a token's owner chip (the owner order of _owner_rows) always holds its own row."""
        R, C = self.shape
        k = int(block.row_blocks)
        assert int(block.pad_rows) == 0, "K3_CAPTURE: moe_x with padded head rows is not mapped (v0)"
        tpc = self.chunk // R
        sub = tpc // k
        Sk = sub // C
        out = None
        for n, d in enumerate(ttnn.get_device_tensors(t)):
            rows = ttnn.to_torch(d).reshape(-1, width)
            if out is None:
                out = torch.zeros(self.chunk, width, dtype=rows.dtype)
            r, p = divmod(n, C)
            for i in range(k):
                lo = r * tpc + i * sub + p * Sk
                out[lo : lo + Sk] = rows[lo : lo + Sk]
        return out

    def _owner_rows(self, t, block, width) -> torch.Tensor:
        """An owner-order MoE head output as [chunk, width] in token order (tests/moe_block/test_moe_block_padding.py
        _owner_rows): chip (r, p) owns, of each of the head's k row blocks, Sk tokens at p Sk of the block's rows."""
        R, C = self.shape
        k = int(block.row_blocks)
        tpc = self.chunk // R
        tpp = tpc + int(block.pad_rows)
        sub = tpp // k
        Sk = sub // C
        out = None
        for n, d in enumerate(ttnn.get_device_tensors(t)):
            rows = ttnn.to_torch(d).reshape(-1, width)
            if out is None:
                out = torch.zeros(self.chunk, width, dtype=rows.dtype)
            r, p = divmod(n, C)
            for i in range(k):
                for t_ in range(Sk):
                    pos = i * sub + p * Sk + t_
                    if pos < tpc:
                        out[r * tpc + pos] = rows[i * Sk + t_]
        return out

    def _shared_down_in(self, ffn, x_in):
        """act(gate(x)) * up(x) of the shared experts, recomputed on device from moe_in's tensor with the bf16
        checkpoint weights (HiFi4, fp32 DEST, fp32 output): the fused MoE head never writes it."""
        layer = ffn.layer_idx
        t0 = time.time()
        w = self._shared_w.get(layer)
        if w is None:
            host = ffn.adapter.moe_block_nonrouted_weights(layer, ffn.hf_config)
            cols = [None, None]
            cols[self.tp_axis] = 3  # (N over TP: each chip computes its shared hidden columns)
            mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=self.shape, dims=tuple(cols))
            w = {
                k: ttnn.from_torch(
                    host[k].T.contiguous().reshape(1, 1, *host[k].T.shape).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )
                for k in ("gate", "up")
            }
            for old in self._shared_w.values():  # (one layer's at a time)
                for t in old.values():
                    ttnn.deallocate(t)
            self._shared_w = {layer: w}
        cfg = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        x = ttnn.all_gather(x_in, dim=3, cluster_axis=self.tp_axis)  # (moe_in is TP-column-sharded)
        g = ttnn.linear(x, w["gate"], compute_kernel_config=cfg, dtype=ttnn.float32)
        u = ttnn.linear(x, w["up"], compute_kernel_config=cfg, dtype=ttnn.float32)
        ttnn.deallocate(x)
        mc = self.model_cfg
        act = getattr(mc, "SHARED_EXPERT_ACTIVATION", getattr(mc, "ROUTED_EXPERT_ACTIVATION", "silu"))
        if act == "situ":
            from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import situ_glu

            a = situ_glu(g, u, mc.ACTIVATION_SITU_BETA, mc.ACTIVATION_SITU_LINEAR_BETA)
        else:
            assert act == "silu", f"K3_CAPTURE: shared expert activation {act!r}"
            a = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            ttnn.deallocate(g)
            ttnn.deallocate(u)
        self._t_host_chunk += time.time() - t0
        self.tap(layer, "shared_down_in", a, RES)
        ttnn.deallocate(a)

    # --- output ---

    def finish(self):
        t_fin = time.time()
        for f in self._pending:
            f.result()
        self._pending.clear()
        self._pool.shutdown(wait=True)
        for files in self.moe_files.values():
            for f in files.values():
                f.close()
        layers = sorted({k[0] for k in self.grams} | set(self.moe_files), key=lambda l: (isinstance(l, str), str(l).zfill(3)))
        written = []
        for layer in layers:
            d = self._layer_tmp(layer)
            meta = {
                "format": VERSION,
                "layer": layer,
                "kind": self._kind(layer),
                "n_tokens": self.ntok.get(layer) or max((g.n for (l, _), g in self.grams.items() if l == layer), default=0),
                "files": {},
                "tensor_map": self._tensor_map(layer),
                "notes": {
                    "latent_up_in": "routed sum (combine output) RMS-normalised in fp64, routed_expert_norm gamma NOT applied",
                    "shared_down_in": "recomputed with extra ttnn ops (bf16 checkpoint gate/up, HiFi4 fp32) from moe_in",
                    "moe_x": "the MoE head's dispatch rows (bf16; owner chip's landing buffer row) = the routed experts' gate/up input",
                    "moe_w": "the combine's routing weights (bf16 in the routing rows), as float32",
                    "moe_ids": "checkpoint expert ids",
                },
            }
            for (l, point), g in sorted(self.grams.items()):
                if l != layer or g.g is None:
                    continue
                g.g.numpy().tofile(d / f"gram_{point}.f64")
                K = int(g.g.shape[-1])
                (d / f"gram_{point}.json").write_text(
                    json.dumps({"n_tokens": g.n, "K": K, "heads": int(g.g.shape[0]) if g.g.dim() == 3 else None,
                                "source_dtype": g.dtype})
                )
                meta["files"][f"gram_{point}.f64"] = {"shape": list(g.g.shape), "dtype": "float64"}
                if self.raw:
                    meta["files"][f"raw_{point}.f32"] = {"shape": [g.n] + list(g.g.shape[:-2]) + [K], "dtype": "float32"}
            if layer in self.moe_files:
                n = self.ntok[layer]
                self.counts[layer].numpy().tofile(d / "expert_counts.i64")
                meta["files"].update(
                    {
                        "moe_x.bf16": {"shape": [n, None], "dtype": "bfloat16"},
                        "moe_ids.i16": {"shape": [n, 16], "dtype": "int16"},
                        "moe_w.f32": {"shape": [n, 16], "dtype": "float32"},
                        "expert_counts.i64": {"shape": [int(self.counts[layer].numel())], "dtype": "int64"},
                    }
                )
                width = (d / "moe_x.bf16").stat().st_size // (2 * max(n, 1))
                meta["files"]["moe_x.bf16"]["shape"] = [n, int(width)]
            (d / "meta.json").write_text(json.dumps(meta, indent=1))
            final = self.dir / self._name(layer)
            if final.exists():
                shutil.rmtree(final)
            d.rename(final)
            written.append(layer)
        run = {
            "run_id": self.run_id,
            "format": VERSION,
            "tt_metal_commit": _commit(),
            "env": {k: v for k, v in os.environ.items() if k.startswith(("K3_ACC", "K3_CAPTURE", "MOE_IMPL"))},
            "mesh": list(self.shape),
            "sp_axis": self.sp_axis,
            "chunk": self.chunk,
            "calibration_manifest": os.environ.get("K3_CAPTURE_MANIFEST"),
            "tokens_sha256": os.environ.get("K3_CAPTURE_TOKENS_SHA256"),
            "n_sequences": self.n_sequences,
            "n_tokens": self.n_tokens,
            "n_chunks": self.n_chunks,
            "layers": written,
            "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds_device_side": round(self.t_device, 2),
            "seconds_host_reads": round(self.t_host, 2),
            "seconds_gram_worker": round(_Gram.seconds, 2),
            "seconds_finish_wait": round(time.time() - t_fin, 2),
        }
        (self.dir / "run.json").write_text(json.dumps(run, indent=1))
        logger.warning(f"K3_CAPTURE: wrote layers {written} to {self.dir}")

    def _kind(self, layer) -> str:
        if layer == LM_HEAD:
            return LM_HEAD
        if layer in self.model_cfg.mla_layer_ids():
            return "mla"
        return "kda_dense0" if layer < self.model_cfg.NUM_DENSE_LAYERS else "kda"

    def _tensor_map(self, layer) -> dict:
        if layer == LM_HEAD:
            return {"lm_head.weight": "lm_head_in"}
        m = dict(_MLA_MAP if self._kind(layer) == "mla" else _KDA_MAP)
        m.update(_MLP0_MAP if layer < self.model_cfg.NUM_DENSE_LAYERS else _MOE_MAP)
        return {f"model.layers.{layer}.{k}": v for k, v in m.items()}
