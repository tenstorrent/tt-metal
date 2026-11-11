# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 vision encoder for the media module: the PUP head-parallel tower (tt/vision/dp_vision.py) on the encoder
service's own (1, N) mesh, N in {4, 8}.

Every chip holds the whole tower (bf8 weights, replicated). A batch of images runs data-parallel (whole images per
chip; images left over are split over all chips) and a single large image sequence-parallel with its attention
head-parallel (16 heads / N per chip). Each chip returns its own merged rows [rows, 4096] (Q_DPV_SELECT=rows: no
output gather), read back once and scattered into the items' stored values in image-token order.

The service owns the mesh: load() builds weights and traces on it and close() frees them; nothing here opens, closes
or resets devices.
"""

import os
import time

import numpy as np
import torch
from loguru import logger

# The deployed tower configuration (PUP session 5: worst/s5/BEST_CONFIG_S5.sh, the Q_DPV_* part). DPVision reads
# these from the environment until its M2 port turns them into constants; load() sets them (an explicit environment
# value wins, except Q_DPV_SELECT, which the encoder needs as rows).
DEPLOYED_TOWER = {
    "Q_DPV_KV_BF8": "1",
    "Q_DPV_FASTHOST": "1",
    "Q_DPV_MIXED": "1",
    "Q_DPV_L1": "2",
    "Q_DPV_SDPA": "128,512",
    "Q_DPV_EXP_APPROX": "1",
    "Q_DPV_SDPA_FID": "hifi2",
    "Q_DPV_FC1_ACT": "gelu_tanh",
    "Q_DPV_SP_MIN": "1024",
    "Q_DPV_SMALLM": "512",
    "Q_DPV_WDT": "bfp8",
    "Q_DPV_SDPA_SP": "64,1024",
    "Q_DPV_SP_HEADS": "1",
    "Q_DPV_SP_HEADS_BF8": "1",
    "Q_DPV_SP_HEADS_SDPA": "64,1024",
    "Q_DPV_SP_HEADS_SDPA_SMALL": "64,1024",
    "Q_DPV_MIXED_HP": "1",
    "Q_DPV_ATT_L1": "2",
    "Q_DPV_QKV_L1": "1",
    "Q_DPV_FC1_L1": "1",
    "Q_DPV_PIX_T1": "1",
    "Q_DPV_GRID": "1",
}


def load_reference_visual(weights_path: str):
    """The HF vision tower alone (no text weights): Qwen3_5VisionModel from config.json + the model.visual.* tensors."""
    import glob

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = AutoConfig.from_pretrained(weights_path)
    visual = Qwen3_5VisionModel._from_config(config.vision_config).float()
    state = {}
    for path in sorted(glob.glob(os.path.join(weights_path, "*.safetensors"))):
        with safe_open(path, "pt") as f:
            for key in f.keys():
                for prefix in ("model.visual.", "visual."):
                    if key.startswith(prefix):
                        state[key[len(prefix) :]] = f.get_tensor(key).float()
    missing, unexpected = visual.load_state_dict(state, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]
    if missing or unexpected:
        raise ValueError(f"vision weights in {weights_path}: missing {missing[:4]}, unexpected {unexpected[:4]}")
    return visual.eval()


class _TowerShim:
    """What DPVision reads from the stock TP tower, without building the stock tower's device weights."""

    def __init__(self, args, reference):
        self.model_args = args
        self.reference_model = reference
        attention = type("Attention", (), {"configuration": args})()
        block = type("Block", (), {"attention": attention})()
        self.tt_model = type("Tower", (), {"blocks": [block]})()


class VisionEncoder:
    def __init__(self, mesh_device, weights_path: str, options: dict):
        shape = tuple(mesh_device.shape)
        if len(shape) != 2 or shape[0] != 1 or shape[1] not in (1, 2, 4, 8, 16):
            raise ValueError(f"the Qwen3.5 vision encoder runs on a (1, N) mesh with N dividing 16 heads, got {shape}")
        self.mesh = mesh_device
        self.weights_path = weights_path
        self.trace = bool(options.get("trace", True))
        self.warm_grids = [[tuple(g) for g in grids] for grids in options.get("warm_grids", ())]
        self.dv = None
        self._grids = {}
        self.last = {}

    def load(self):
        import ttnn

        from models.demos.blackhole.qwen36.tt.vision.dp_vision import DPVision
        from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs

        for k, v in DEPLOYED_TOWER.items():
            os.environ.setdefault(k, v)
        os.environ["Q_DPV_SELECT"] = "rows"
        if self.weights_path:
            os.environ["HF_MODEL"] = self.weights_path
        t0 = time.perf_counter()
        args = VisionModelArgs(self.mesh, max_batch_size=1, max_seq_len=8192)
        reference = load_reference_visual(self.weights_path or args.CKPT_DIR)
        self.dv = DPVision(_TowerShim(args, reference), ttnn)
        logger.info(f"Qwen3.5 vision encoder on {tuple(self.mesh.shape)}: weights in {time.perf_counter() - t0:.1f} s")
        self._ensure([torch.tensor([list(g) for g in grids], dtype=torch.long) for grids in self.warm_grids])

    @staticmethod
    def _key(grid):
        return tuple(tuple(int(v) for v in g) for g in grid.tolist())

    def _ensure(self, grids):
        """Every grid set in grids ready to replay. Trace safety: a trace's intermediates are freed when its capture
        ends, so a buffer allocated AFTER a capture (another grid set's pixel buffer, constants or output) can sit in
        memory that trace overwrites on every replay. All grid sets are therefore allocated (prepare_trace) before any
        is captured; a new grid set at request time releases the traces, allocates, and captures every set again."""
        new = [g for g in grids if self._key(g) not in self._grids]
        if not new:
            return
        t0 = time.perf_counter()
        if self.trace:
            self._release_traces()
        for g in new:
            self._grids[self._key(g)] = g
            self.dv.prepare_trace(g)
        if self.trace:
            for g in self._grids.values():
                self.dv.capture(g)
        logger.info(
            f"vision grid sets {[self._key(g) for g in new]}: compiled + (re)captured {len(self._grids)} set(s) in "
            f"{time.perf_counter() - t0:.1f} s"
        )

    def _release_traces(self):
        import ttnn

        for c in self.dv.cache.values():
            if c.get("tid") is not None:
                ttnn.release_trace(self.mesh, c["tid"])
                c["tid"] = None
                if c.get("out") is not None:
                    ttnn.deallocate(c["out"])
                    c["out"] = None

    def _entry(self, grid):
        self._ensure([grid])
        return self.dv._entry(grid)

    def run(self, items):
        """Device pass for items: [n_tokens_total, hidden] bf16 host rows in image-token order (+ timings in .last)."""
        import ttnn

        from models.demos.blackhole.qwen36.vision.media import CHANNELS, PATCH_SIZE

        grid = torch.tensor([[1, it.grid_h, it.grid_w] for it in items], dtype=torch.long)
        self._entry(grid)
        t0 = time.perf_counter()
        patches = np.concatenate([it.patches for it in items]) if len(items) > 1 else items[0].patches
        rows = patches.shape[0]
        lut = ((torch.arange(256, dtype=torch.float32) - 127.5) / 127.5).to(torch.bfloat16)
        one = lut[torch.from_numpy(patches).long()].reshape(rows, CHANNELS, 1, PATCH_SIZE * PATCH_SIZE)
        pix = one.expand(rows, CHANNELS, 2, PATCH_SIZE * PATCH_SIZE)  # HF pixel_values as a view (2 equal frames)
        t1 = time.perf_counter()
        split = []
        out = self.dv(pix, grid, split=split)
        ttnn.synchronize_device(self.mesh)
        t2 = time.perf_counter()
        host = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        t3 = time.perf_counter()
        emb = host[self.dv.row_order(grid)]
        t4 = time.perf_counter()
        self.last = {
            "pixels_ms": (t1 - t0) * 1e3,
            "tower_host_ms": split[0][0] * 1e3,
            "device_ms": (t2 - t1 - split[0][0]) * 1e3,
            "d2h_ms": (t3 - t2) * 1e3,
            "reorder_ms": (t4 - t3) * 1e3,
            "rows": int(emb.shape[0]),
        }
        return emb

    def encode(self, items, rows_out):
        """rows_out[i]: item i's [n_tokens, hidden] uint16 (bf16 bits) view of its stored value."""
        emb = self.run(items)
        t0 = time.perf_counter()
        bits = emb.view(torch.int16).numpy().view(np.uint16)
        o = 0
        for it, dst in zip(items, rows_out):
            dst[:] = bits[o : o + it.n_tokens]
            o += it.n_tokens
        assert o == bits.shape[0], (o, bits.shape)
        self.last["write_ms"] = (time.perf_counter() - t0) * 1e3

    def close(self):
        import ttnn

        dv, self.dv = self.dv, None
        if dv is None:
            return
        for c in dv.cache.values():
            if c.get("tid") is not None:
                ttnn.release_trace(self.mesh, c["tid"])
            for k in ("xbuf", "out"):
                if c.get(k) is not None:
                    ttnn.deallocate(c[k])
        dv.cache.clear()
