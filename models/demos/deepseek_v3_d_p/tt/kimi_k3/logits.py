# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Logits of every prompt position (Kimi-K3 / K3.1 prefill, last stage), for end-to-end KLD against a CPU reference.

Off unless K3_LOGITS_DIR is set. On, the last rank's eager prefill_chunk applies the LM head to the model's output (the
final norm's) at every real position: x [S/SP, H/TP] all-gathered over TP, times lm_head^T [H, V/TP] (vocab over TP,
replicated over SP) at HiFi4 with fp32 DEST, fp32 logits read back and appended in token order:

    <K3_LOGITS_DIR>/logits.f32   raw little-endian float32 [T, V] (padding rows excluded)
    <K3_LOGITS_DIR>/meta.json    vocab, n_tokens, the chunks' (actual_start, actual_end), weight dtype, commit, env

The LM head weight is the checkpoint's bf16 lm_head.weight (the model's adapter's moe_block_checkpoint), bf16 on device
under K3_ACC_BF16_WEIGHTS, else bf8 (decode's production dtype). Eager forwards only.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

import ttnn

LOGITS = None  # the active LogitsHook, None when K3_LOGITS_DIR is unset


def start(runtime):
    """Build the hook on the last rank when K3_LOGITS_DIR is set (TtKimiK3Runtime._build_model)."""
    global LOGITS
    root = os.environ.get("K3_LOGITS_DIR")
    if not root or not runtime.config.is_last_rank:
        return None
    if LOGITS is not None:
        LOGITS.finish()
    LOGITS = LogitsHook(Path(root), runtime)
    return LOGITS


def finish():
    global LOGITS
    if LOGITS is not None:
        LOGITS.finish()
        LOGITS = None


atexit.register(finish)


class LogitsHook:
    def __init__(self, root: Path, runtime):
        from models.demos.deepseek_v3_d_p.tt.kimi_k3 import accuracy
        from models.demos.deepseek_v3_d_p.tt.moe_impl import model_adapter

        cfg = runtime.config
        self.mesh = runtime.mesh_device
        self.shape = tuple(self.mesh.shape)
        self.sp_axis, self.tp_axis = int(cfg.sp_axis), int(cfg.tp_axis)
        self.chunk = int(cfg.chunk_size)
        self.dir = root
        self.dir.mkdir(parents=True, exist_ok=True)
        ckpt = Path(model_adapter(cfg.model_cfg).moe_block_checkpoint())
        wm = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
        key = "lm_head.weight" if "lm_head.weight" in wm else "language_model.lm_head.weight"
        with safe_open(str(ckpt / wm[key]), framework="pt", device="cpu") as f:
            w = f.get_tensor(key)  # [V, H] bf16
        self.vocab = int(w.shape[0])
        TP = self.shape[self.tp_axis]
        pad = -self.vocab % (TP * ttnn.TILE_SIZE)
        wt = torch.nn.functional.pad(w.T.contiguous(), (0, pad)).reshape(1, 1, w.shape[1], self.vocab + pad)
        dims = [None, None]
        dims[self.tp_axis] = 3
        self.wdtype = accuracy.weight_dtype(ttnn.bfloat8_b)
        self.weight = ttnn.from_torch(
            wt,
            dtype=self.wdtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=self.shape, dims=tuple(dims)),
        )
        del w, wt
        self.cfg = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.file = open(self.dir / "logits.f32", "ab")
        self.chunks = []
        self.n_tokens = 0
        self.valid = None
        self.seconds = 0.0
        self.ckpt, self.key = str(ckpt), key
        logger.warning(f"K3_LOGITS: {self.dir}, vocab {self.vocab}, LM head {self.wdtype} (eager prefill only)")

    def begin_chunk(self, actual_start, actual_end):
        assert actual_start is None or isinstance(actual_start, int), "K3_LOGITS: eager prefill only (host scalars)"
        start = int(actual_start or 0)
        end = start + self.chunk if actual_end is None else int(actual_end)
        self.valid = max(0, min(self.chunk, end - start))
        self.chunks.append([start, start + self.valid])

    def write(self, out):
        """The model's output (the final norm's, [1, 1, S/SP, H/TP]) -> fp32 logits of the chunk's real rows."""
        if self.valid is None or out is None:
            return
        t0 = time.time()
        x = ttnn.all_gather(out, dim=3, cluster_axis=self.tp_axis)
        logits = ttnn.linear(x, self.weight, dtype=ttnn.float32, compute_kernel_config=self.cfg)
        ttnn.deallocate(x)
        SP, TP = self.shape[self.sp_axis], self.shape[self.tp_axis]
        C = self.shape[1]
        per = {}
        for n, d in enumerate(ttnn.get_device_tensors(logits)):
            rc = divmod(n, C)
            per[(rc[self.sp_axis], rc[self.tp_axis])] = ttnn.to_torch(d).reshape(-1, d.shape[-1])
        ttnn.deallocate(logits)
        rows = torch.cat([torch.cat([per[(sp, tp)] for tp in range(TP)], 1) for sp in range(SP)], 0)
        rows = rows[: self.valid, : self.vocab].float().contiguous()
        self.file.write(rows.numpy().tobytes())
        self.n_tokens += self.valid
        self.valid = None
        self.seconds += time.time() - t0

    def finish(self):
        from models.demos.deepseek_v3_d_p.tt.kimi_k3.capture import _commit

        self.file.close()
        ttnn.deallocate(self.weight)
        meta = {
            "vocab": self.vocab,
            "n_tokens": self.n_tokens,
            "dtype": "float32",
            "shape": [self.n_tokens, self.vocab],
            "chunks": self.chunks,
            "lm_head": {"checkpoint": self.ckpt, "key": self.key, "device_dtype": str(self.wdtype),
                        "math": "HiFi4, fp32 DEST, fp32 output"},
            "input": "the stage's output (final norm applied), all-gathered over TP",
            "tt_metal_commit": _commit(),
            "env": {k: v for k, v in os.environ.items() if k.startswith(("K3_ACC", "K3_LOGITS", "K3_CAPTURE"))},
            "seconds": round(self.seconds, 2),
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=1))
        logger.warning(f"K3_LOGITS: wrote {self.n_tokens} x {self.vocab} logits to {self.dir}")
