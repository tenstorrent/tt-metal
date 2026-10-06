# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One DeepSeek-V4.1-Flash decoder layer (decode), mirroring ``Block.forward`` of the checkpoint's model.py.

    residual = x
    attn_pre, attn_post, attn_comb = mixes(x)           # attention-side mHC coefficients, from the stream x
    h = collapse(x, pre_in)                              # pre_in = the PREVIOUS layer's ffn `pre`
    x = expand(attn(attn_norm(h)), residual, attn_post, attn_comb)
    residual = x
    ffn_pre, ffn_post, ffn_comb = mixes(x)
    h = collapse(x, attn_pre)                            # note: attn_pre, produced above
    x = expand(ffn(ffn_norm(h)), residual, ffn_post, ffn_comb)
    return x, ffn_pre

Streams are fp32 [1, 1, T, 4*5120] (tokens sharded over mesh rows, replicated over columns); sub-blocks run in
bf16 and return replicated [1, 1, T, 5120].
"""

import os
import time

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_v2 import (
    DSV41SharedExpertV2 as DSV41SharedExpert,  # 155 -> 112 us/layer (tuned 1D mcast configs)
)

_STREAM_BF16 = os.environ.get("DSV41_STREAM_BF16") == "1"


def _rnd_bf16(t):
    return ttnn.typecast(ttnn.typecast(t, ttnn.bfloat16), ttnn.float32)


class DSV41Layer:
    def __init__(
        self,
        mesh_device,
        mesh_config,
        ccl,
        attention,
        norms,
        mhc_params,
        moe_weights,
        gate_bias_shift,
        users_per_row=4,
        eps=1e-20,
        moe_buffers=None,
        expert_state=None,
    ):
        self.mesh_device, self.mesh_config, self.ccl = mesh_device, mesh_config, ccl
        self.T = users_per_row
        _d = lambda tag: self._l1_diag(tag)
        _d("layer init start")
        self._bt("layer init start")
        self.eps = eps
        self.debug = None  # set to a dict to capture sub-block outputs (error-budget tests)
        self.attention = attention
        _d("before mhc")
        self._bt("(attention built before this)")
        self.mhc_attn = DSV41MHC(mesh_device, *mhc_params["attn"])
        self.mhc_ffn = DSV41MHC(mesh_device, *mhc_params["ffn"])
        _d("after mhc x2")
        self._bt("mhc x2")
        self.moe = DSV41MoEBlock(
            mesh_device,
            moe_weights,
            topology=ttnn.Topology.Linear,
            batch_per_device=users_per_row,
            gate_bias_shift=gate_bias_shift,
            buffers=moe_buffers,
            expert_state=expert_state,
        )
        # norm weights as fp32 tile rows [1, 1, 1, D]: the norm is composed from small ops (see _norm)
        _d("after moe block (gate+experts+buffers)")
        self._bt("moe block (gate+experts)")
        self.moe.warmup()  # before anything else touches L1: see DSV41MoEBlock.warmup
        _d("after moe warmup")
        self._bt("moe warmup (compile pass)")
        sid = next(iter(moe_weights["shared_w0"]))
        if (
            os.environ.get("DSV41_SHARED", "v2") == "v1"
        ):  # accuracy study: the simple v1 shared expert, optionally bf16 weights
            from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert import DSV41SharedExpert as _V1

            self.shared = _V1(
                mesh_device,
                moe_weights["shared_w0"][sid],
                moe_weights["shared_w1"][sid],
                moe_weights["shared_w2"][sid],
                dtype=ttnn.bfloat16 if os.environ.get("DSV41_SHARED_DT", "bfp8") == "bf16" else ttnn.bfloat8_b,
            )
        else:
            self.shared = DSV41SharedExpert(
                mesh_device, moe_weights["shared_w0"][sid], moe_weights["shared_w1"][sid], moe_weights["shared_w2"][sid]
            )
        up = lambda t: ttnn.from_torch(
            t.reshape(1, 1, 1, -1).float(),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.attn_norm_w, self.ffn_norm_w = up(norms["attn_norm"]), up(norms["ffn_norm"])
        self._bt("shared expert + norms")

    def _bt(
        self, tag
    ):  # DSV41_BUILD_PROFILE=1: wall time since the previous mark (device synchronised) while building a layer
        import os as _os

        if _os.environ.get("DSV41_BUILD_PROFILE") == "1":
            ttnn.synchronize_device(self.mesh_device)
            now = time.perf_counter()
            print(f"BUILDT {tag:34s} {now - getattr(self, '_bt_last', now):6.2f} s", flush=True)
            self._bt_last = time.perf_counter()

    def _l1_diag(self, tag):
        import os

        if os.environ.get("DSV41_L1_DIAG") == "1":
            ttnn.synchronize_device(self.mesh_device)
            mv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.L1)
            print(f"L1BUILD {tag:42s} allocated/bank {mv.total_bytes_allocated_per_bank:8d}", flush=True)

    def _norm(self, h32, w):
        """RMSNorm over the last dim, composed from small ops. The fused ``ttnn.rms_norm`` kernel statically
        reserves ~1 MB of L1 per core for a 5120-wide row; a persistent L1 buffer left in the middle of L1 by the
        MoE decode ops (owned by cached programs) then makes it fail on the second forward call.
        Like the reference: round the input to bf16, normalise in fp32, return bf16."""
        if os.environ.get("DSV41_FUSED_NORM", "1") == "1":
            return ttnn.rms_norm(ttnn.typecast(h32, ttnn.bfloat16), epsilon=self.eps, weight=w)
        x = ttnn.typecast(ttnn.typecast(h32, ttnn.bfloat16), ttnn.float32)
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        y = ttnn.multiply(ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, self.eps))), w)
        return ttnn.typecast(y, ttnn.bfloat16)

    def _to_tok(self, t):  # [1,1,T,D] <-> [T,1,1,D]
        return ttnn.reshape(t, [self.T, 1, 1, t.shape[-1]])

    def _to_row(self, t):
        return ttnn.reshape(t, [1, 1, self.T, t.shape[-1]])

    def forward(self, x, pre_in, st, profile=None, forced_routing=None):
        """x [T,1,4,D] fp32, pre_in [T,1,1,4] fp32, st: attention step inputs -> (x_new, ffn_pre).

        ``profile``: optional dict; when given, every section is bracketed by a device sync and its wall time (s)
        accumulated under its name (this serialises the host, so the sum is an upper bound of the eager latency).
        """
        T = self.T
        t0 = [time.perf_counter()]

        def mark(name):
            if profile is not None:
                ttnn.synchronize_device(self.mesh_device)
                t1 = time.perf_counter()
                profile[name] = profile.get(name, 0.0) + (t1 - t0[0])
                if profile.get("_l1_trace") is not None:  # debug: L1 allocator state after each section
                    mv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.L1)
                    profile["_l1_trace"].append(
                        (name, mv.total_bytes_allocated_per_bank, mv.largest_contiguous_bytes_free_per_bank)
                    )
                t0[0] = time.perf_counter()

        attn_pre, attn_post, attn_comb = self.mhc_attn.mixes(x)
        mark("mhc_attn_mixes")
        h = self.mhc_attn.collapse_norm(
            x, pre_in, self.attn_norm_w, self.eps
        )  # fused collapse + RMSNorm -> bf16 [1,1,T,D]
        mark("mhc_attn_collapse+norm")
        a = self.attention.forward(h, st)  # [1,1,T,D] bf16, replicated
        mark("attention")
        # a: bf16 [1,1,T,D], consumed as is; expand + the FFN-side mixes are one program with DSV41_MHC_EP=1
        if (
            _STREAM_BF16
        ):  # accuracy study: round the stream to bf16 at every hc_post like the bf16 reference / GPU model
            x2 = _rnd_bf16(self.mhc_attn.expand(a, x, attn_post, attn_comb, None))
            ffn_pre, ffn_post, ffn_comb = self.mhc_ffn.mixes(x2)
        else:
            x2, (ffn_pre, ffn_post, ffn_comb) = self.mhc_attn.expand_mixes(
                a, x, attn_post, attn_comb, None, self.mhc_ffn
            )
        mark("mhc_attn_expand+ffn_mixes")
        h, h_tok = self.mhc_ffn.collapse_norm_rm(
            x2, attn_pre, self.ffn_norm_w, self.eps
        )  # h [1,1,T,D] bf16, h_tok [T,1,1,D] bf16 row-major
        mark("mhc_ffn_collapse+norm")
        m = self.moe.forward(h, h_tok, forced_routing)  # [1,1,T,D/cols] per device
        mark("moe")
        m = self.mesh_config.allgather(m, self.ccl, axis=1, dim=3)  # -> [1,1,T,D] replicated
        mark("moe_allgather")
        sh = self.shared.forward(h)  # shared expert, replicated bfp8 (see shared_expert.py)
        mark("shared_expert")
        if self.debug is not None:
            self.debug.update(
                attn_out=a, ffn_in=h, ffn_out=ttnn.add(ttnn.typecast(m, ttnn.float32), sh), routed=m, shared=sh
            )
        x3 = self.mhc_ffn.expand(
            m, x2, ffn_post, ffn_comb, sh
        )  # post * (routed + shared) + comb^T x, summed inside the kernel
        if _STREAM_BF16:
            x3 = _rnd_bf16(x3)
        mark("mhc_ffn_expand")
        # hand DRAM-resident tensors to the caller (the next layer): an L1-sharded `ffn_pre` kept alive across
        # calls overlaps the static buffers of the next norm kernel.
        return ttnn.to_memory_config(x3, ttnn.DRAM_MEMORY_CONFIG), ttnn.to_memory_config(
            ffn_pre, ttnn.DRAM_MEMORY_CONFIG
        )
