# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One exact-ternary projection for single-chip decode: y = W x with W from the PQ2_0 GGUF.

forward(x): x [.., 1(32), K] bf16 32x32 tiles, row 0 live -> y [.., 1(32), N] 32x32 tiles in
`out_mem`, row 0 live, rest zero. N is padded on the weight side to a whole number of output
tiles per A3 worker; the padding columns are zero and never written.
"""

from __future__ import annotations

import os

import numpy as np

import ttnn

from . import op as a3
from .mlp import Shared


class TernaryLinear:
    def __init__(self, shared: Shared, name: str, cache_dir=None, out_mem=ttnn.L1_MEMORY_CONFIG, rows=None, shape=None):
        """name: GGUF tensor (and cache key). rows: optional callable returning (codes [N, K],
        scales [N, K/128]) already reordered by the caller (e.g. the GDN v-head regrouping), with
        shape = (N, K). The GGUF codes are only decoded on a cache miss."""
        from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

        self.sh = shared
        self.out_mem = out_mem
        if rows is None:
            rows = lambda: (lambda t: (t.codes, t.scales))(read_ternary(shared.gguf, name))  # noqa: E731
            shape = tuple(shared.gguf.tensors[name].shape)
        n, k = shape
        self.n, self.k = n, k
        assert k in shared.m, f"no Hadamard tensor for width {k}"
        per = 32 * len(shared.workers)  # output columns per round over all workers
        n_pad = -(-n // per) * per

        def build():
            codes_nk, scales_ng = rows()
            codes = np.zeros((k, n_pad), np.float32)
            codes[:, :n] = codes_nk.T
            scales = np.zeros((k // 128, n_pad), np.float32)
            scales[:, :n] = scales_ng.T
            return codes, scales

        cache = os.path.join(cache_dir, name) if cache_dir else None
        self.w, self.s = a3.weights_for_device(shared.device, build, cache=cache, kn=(k, n_pad))

    def forward(self, x, out_mem=None):
        sh = self.sh
        x_in = x
        if x.dtype != ttnn.bfloat16:  # prep reads bf16 rows; the GDN out-proj input is fp32
            x = ttnn.typecast(x, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        elif x.is_sharded() or x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x8 = a3.masked_input_buffer(sh.device, sh.workers, self.k)
        a3.prep(x, sh.m[self.k], x8, sh.workers, src_tile32=True)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)[:-1] + [self.n]), ttnn.bfloat16,
                                             ttnn.TILE_LAYOUT, sh.device, out_mem or self.out_mem)
        a3.ternary_linear(x8, self.w, self.s, out, sh.workers, cache=False, n_real=self.n)
        ttnn.deallocate(x8)
        if x is not x_in:
            ttnn.deallocate(x)
        return out


# Sites that passed their e2e gate; each joins in its own commit.
VALIDATED = {"mlp", "gdn_out", "gdn_in", "attn_qkv", "attn_o", "lm_head"}
WIRED = {"mlp", "gdn_out", "gdn_in", "attn_o", "attn_qkv", "lm_head"}


def sites():
    """BONSAI_NATIVE_SITES: comma list of the projections that run exact-ternary ("all" = every wired
    site); unset = the validated ones."""
    v = os.environ.get("BONSAI_NATIVE_SITES")
    if v is None:
        return VALIDATED
    return WIRED if v == "all" else set(v.split(","))


def gdn_in_shape(shared: Shared, layer: int):
    g = shared.gguf.tensors
    return (g[f"blk.{layer}.attn_qkv.weight"].shape[0] + g[f"blk.{layer}.attn_gate.weight"].shape[0],
            g[f"blk.{layer}.attn_qkv.weight"].shape[1])


def attn_qkv_shape(shared: Shared, layer: int):
    g = shared.gguf.tensors
    return (sum(g[f"blk.{layer}.attn_{p}.weight"].shape[0] for p in "qkv"), g[f"blk.{layer}.attn_q.weight"].shape[1])


def gdn_in_rows(shared: Shared, layer: int):
    """GDN in-projection [qkv | z] rows in the order the dense TP=1 path uses: the HF converter's
    vperm_rows for attn_qkv (v part) and attn_gate (bonsai_conversion_report: vperm_rows in all 48
    GDN layers); prepare_gdn_qkv is the identity at TP=1."""
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary, v_head_index

    kv = shared.gguf.kv
    nv, nk = int(kv["qwen35.ssm.time_step_rank"]), int(kv["qwen35.ssm.group_count"])
    hd = int(kv["qwen35.ssm.inner_size"]) // nv
    hk = int(kv["qwen35.ssm.state_size"])
    qk = 2 * nk * hk
    v = v_head_index(nv, nk, hd)
    qkv = read_ternary(shared.gguf, f"blk.{layer}.attn_qkv.weight")
    z = read_ternary(shared.gguf, f"blk.{layer}.attn_gate.weight")
    perm_qkv = np.concatenate([np.arange(qk), qk + v])
    codes = np.concatenate([qkv.codes[perm_qkv], z.codes[v]])
    scales = np.concatenate([qkv.scales[perm_qkv], z.scales[v]])
    return codes, scales


def attn_qkv_rows(shared: Shared, layer: int):
    """Full-attention [q+gate | k | v] rows as QWEN36_1C_FUSED_QKV concatenates the HF q/k/v_proj
    (the converter keeps attn_q/k/v as_is in all 16 attention layers)."""
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

    ts = [read_ternary(shared.gguf, f"blk.{layer}.attn_{p}.weight") for p in ("q", "k", "v")]
    return np.concatenate([t.codes for t in ts]), np.concatenate([t.scales for t in ts])
