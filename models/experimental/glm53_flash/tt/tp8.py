# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14 (PRODUCTION_PATH.md M3): tensor-parallel shards of a dense_kda layer (layers 0-2 kind) on a mesh of TP
chips (TP8 on the 2x4 mesh; TP2 on the card0 pair for development), built at load time from the single-chip Stage 00c
weight cache (doc/ttnn/14-production-tp8-megakernel/shard_contract.md is the contract):

  rank r = the mesh device index in row-major mesh-coordinate order (= ttnn.ShardTensorToMesh's order);
  heads   h in [H_r * r, H_r * (r + 1)), H_r = 64 / TP (8 at TP8); channels c in [128 H_r r, 128 H_r (r + 1))
  w_qkv   [4096, 3 * 128 H_r]   = [q[:, c] | k[:, c] | v[:, c]]                         bfloat8_b (column slices)
  w_lowrank [4096, 256 + H_r]   = [f_a | g_a | b[:, heads]]  (f_a / g_a replicated)    bfloat16
  w_fb, w_gb [128, 128 H_r]     = f_b[:, c], g_b[:, c]                                  bfloat16
  taps    [4, 3 * 128 H_r]      = [q_conv[:, c] | k_conv[:, c] | v_conv[:, c]]          bfloat16
  dt_bias [128 H_r], A_log [H_r] (decay rate exp(A_log) per channel)                    float32
  w_o     [128 H_r, 4096]       = o_proj[c, :]                                          bfloat8_b (row slice)
  gate_up [4096, 2 I_r]         = [gate[:, i] | up[:, i]], i in [I_r r, I_r (r + 1)), I_r = 12288 / TP
  down    [I_r, 4096]           = down[i, :]                                            bfloat8_b (row slice)
  replicated: mHC (fn, scale, base) of both sublayers, input / post-attention RMSNorm weights, o_norm
  state   S [B, H_r, 128, 128] fp32 (heads), conv history [1, 3, B, 3 * 128 H_r] bf16 (the qkv columns above)

Every slice is tile-aligned (32-row / 32-column boundaries) and bfloat8_b blocks are 16-value face rows, so the
re-quantised shard holds the cache's exact values (`check_lossless`). The per-chip layer then is the single-chip code
on these tensors: FusedKDA -> OptimizedKDA (DS linears of the shards) and SwiGLUMLP -> OptimizedSwiGLU, the 7a
megakernel programs (MkStepB step D, MkFFN) on the shard objects, the fabric all-reduce phase after o_proj and down.
"""

from __future__ import annotations

import torch

import ttnn

TILE = 32


class TPShards:
    """Load-time TP slices of one layer from a TileCacheSource (`host(layer, name)` -> full torch tensor)."""

    def __init__(self, mesh, source, layer: int, cfg, *, vtp: int | None = None):
        """vtp (timing only): shard as TP = vtp but place only ranks 0 .. ndev - 1 (e.g. TP8-sized shards on the 2-chip
        pair: every chip runs exactly a TP8 chip's program; the all-reduce sums 2 of the 8 partials)."""
        self.mesh, self.source, self.layer, self.cfg = mesh, source, layer, cfg
        self.ndev = int(mesh.get_num_devices())
        self.tp = int(vtp or self.ndev)
        assert self.ndev <= self.tp, (self.ndev, self.tp)
        self.virtual = self.tp != self.ndev
        self.heads = int(cfg.linear_num_heads)
        self.hd = int(cfg.linear_head_dim)
        assert self.heads % self.tp == 0, (self.heads, self.tp)
        self.h = self.heads // self.tp  # heads per chip
        self.w = self.h * self.hd  # channels per chip
        self._host = {}

    # ------------------------------------------------------------------ host side
    def host(self, name: str) -> torch.Tensor:
        if name not in self._host:
            self._host[name] = self.source.host(self.layer, name)
        return self._host[name]

    def heads_of(self, r: int) -> slice:
        return slice(self.h * r, self.h * (r + 1))

    def chans_of(self, r: int) -> slice:
        return slice(self.w * r, self.w * (r + 1))

    def drop(self):
        self._host = {}

    # ------------------------------------------------------------------ device side
    def sharded(self, blocks: list, dtype, dim: int, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        """blocks[r] = rank r's tensor (same shape) -> one mesh tensor, device r holding blocks[r]."""
        assert len(blocks) == self.ndev
        full = torch.cat([b.contiguous() for b in blocks], dim=dim)
        return ttnn.from_torch(full, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                               memory_config=memory_config, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=dim))  # fmt: skip

    def replicated(self, t: torch.Tensor, dtype) -> ttnn.Tensor:
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))  # fmt: skip

    def col(self, name: str, sl_fn, dtype) -> ttnn.Tensor:
        full = self.host(name)
        return self.sharded([full[..., sl_fn(r)] for r in range(self.ndev)], dtype, dim=-1)

    def row(self, name: str, sl_fn, dtype) -> ttnn.Tensor:
        full = self.host(name)
        return self.sharded([full[sl_fn(r)] for r in range(self.ndev)], dtype, dim=0)

    # ------------------------------------------------------------------ the layer's pieces
    def kda_tensors(self) -> dict:
        p = "self_attn."
        bf8, bf16 = ttnn.bfloat8_b, ttnn.bfloat16
        cat3 = lambda names, r: torch.cat(
            [self.host(p + n)[..., self.chans_of(r)] for n in names], dim=-1
        )  # noqa: E731
        qkv = self.sharded([cat3(("q_proj.weight", "k_proj.weight", "v_proj.weight"), r) for r in range(self.ndev)],
                           bf8, dim=-1)  # fmt: skip
        fa, ga, b = (self.host(p + n) for n in ("f_a_proj.weight", "g_a_proj.weight", "b_proj.weight"))
        lowrank = self.sharded([torch.cat([fa, ga, b[:, self.heads_of(r)]], dim=-1) for r in range(self.ndev)], bf16,
                               dim=-1)  # fmt: skip
        taps = self.sharded([cat3(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"), r)
                             for r in range(self.ndev)], bf16, dim=-1)  # fmt: skip
        return dict(
            w_qkv=qkv,
            w_lowrank=lowrank,
            w_fb=self.col(p + "f_b_proj.weight", self.chans_of, bf16),
            w_gb=self.col(p + "g_b_proj.weight", self.chans_of, bf16),
            w_o=self.row(p + "o_proj.weight", self.chans_of, bf8),
            taps=taps,
            o_norm=self.replicated(self.host(p + "o_norm.weight"), bf16),
        )

    def kda_rate_bias(self) -> tuple:
        """(decay_rate, dt_bias) per chip [1, 1, 1, 128 H_r] fp32 (KDA's constructor would replicate them)."""
        p = "self_attn."
        a_log = self.host(p + "A_log").float().reshape(-1)
        dt = self.host(p + "dt_bias").float().reshape(-1)
        rate = [torch.exp(a_log[self.heads_of(r)]).repeat_interleave(self.hd).reshape(1, 1, 1, -1)
                for r in range(self.ndev)]  # fmt: skip
        bias = [dt[self.chans_of(r)].reshape(1, 1, 1, -1) for r in range(self.ndev)]
        return self.sharded(rate, ttnn.float32, dim=-1), self.sharded(bias, ttnn.float32, dim=-1)

    def mlp_tensors(self) -> dict:
        i_full = int(self.host("mlp.gate_proj.weight").shape[-1])
        assert i_full % (self.tp * TILE) == 0
        ir = i_full // self.tp
        isl = lambda r: slice(ir * r, ir * (r + 1))  # noqa: E731
        return dict(
            gate=self.col("mlp.gate_proj.weight", isl, ttnn.bfloat8_b),
            up=self.col("mlp.up_proj.weight", isl, ttnn.bfloat8_b),
            down=self.row("mlp.down_proj.weight", isl, ttnn.bfloat8_b),
            intermediate=ir,
        )

    def shared_tensors(self) -> dict:
        """Stage 14 (GLM53_TP_SHARED): the MoE layer's shared expert TP-sharded like the dense MLP — gate / up output
        columns and down input rows of chip r = intermediate slice r (the down output is a partial sum)."""
        i_full = int(self.host("mlp.shared_experts.gate_proj.weight").shape[-1])
        assert i_full % (self.tp * TILE) == 0, i_full
        ir = i_full // self.tp
        isl = lambda r: slice(ir * r, ir * (r + 1))  # noqa: E731
        return dict(
            gate=self.col("mlp.shared_experts.gate_proj.weight", isl, ttnn.bfloat8_b),
            up=self.col("mlp.shared_experts.up_proj.weight", isl, ttnn.bfloat8_b),
            down=self.row("mlp.shared_experts.down_proj.weight", isl, ttnn.bfloat8_b),
            intermediate=ir,
        )

    def mhc_args(self, sub: str) -> dict:
        """mhc_pre arguments of sublayer `sub` ("attn" / "ffn"), replicated on every chip."""
        cfg, n = self.cfg, self.cfg.hc_mult
        scale = self.host(f"hc_{sub}_scale").float().reshape(-1)
        norm = "input_layernorm.weight" if sub == "attn" else "post_attention_layernorm.weight"
        return dict(
            fn_t=self.host(f"hc_{sub}_fn").reshape(n * cfg.hidden_size, (2 + n) * n),
            scale=torch.cat([scale[0].repeat(n), scale[1].repeat(n), scale[2].repeat(n * n)]),
            bias=self.host(f"hc_{sub}_base").float().reshape(-1),
            norm_w=self.host(norm).float().reshape(-1),
            hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters,
            hidden=cfg.hidden_size, n=n,
        )  # fmt: skip

    # ------------------------------------------------------------------ state
    def shard_state(self, S: torch.Tensor, C: torch.Tensor):
        """Full single-chip KDA decode state (S [B, 64, 128, 128] fp32, C [1, 3, B, 24576] bf16) -> per-chip mesh
        tensors (S heads, C the chip's q | k | v columns)."""
        width = self.heads * self.hd
        Sm = self.sharded([S[:, self.heads_of(r)] for r in range(self.ndev)], ttnn.float32, dim=1)
        cs = lambda r: torch.cat(
            [C[..., k * width + self.chans_of(r).start : k * width + self.chans_of(r).stop] for k in range(3)], dim=-1
        )  # fmt: skip  # noqa: E731
        Cm = self.sharded([cs(r) for r in range(self.ndev)], ttnn.bfloat16, dim=-1)
        return Sm, Cm

    def shard_state_zeros(self, B: int):
        """shard_state of all-zero S / C for B rows, the host side done ONCE per B: the sharded zero tensors are tilized
        into host multi-device tensors on the first call (cached on this object) and every call uploads a copy
        (ttnn.to_device: same spec, mapper and memory config as shard_state's from_torch). A served cache allocates
        68 such states (34 KDA layers x (the state chunk + the prefix-cache snapshot chunk)): 68 host tilizes of
        [32, 64, 128, 128] fp32 + [1, 3, 32, 24576] bf16 were 13.6 s of the KV phase (int27c mock profile)."""
        cache = self.__dict__.setdefault("_zero_state_host", {})
        key = int(B)
        if key not in cache:
            S = torch.zeros(B, 64, 128, 128)
            C = torch.zeros(1, 3, B, 24576, dtype=torch.bfloat16)
            width = self.heads * self.hd
            cs = lambda r: torch.cat(
                [C[..., k * width + self.chans_of(r).start : k * width + self.chans_of(r).stop] for k in range(3)],
                dim=-1,
            )  # fmt: skip  # noqa: E731
            host = lambda blocks, dtype, dim: ttnn.from_torch(  # noqa: E731
                torch.cat([b.contiguous() for b in blocks], dim=dim), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=dim))  # fmt: skip
            cache[key] = (host([S[:, self.heads_of(r)] for r in range(self.ndev)], ttnn.float32, 1),
                          host([cs(r) for r in range(self.ndev)], ttnn.bfloat16, -1))  # fmt: skip
        Sh, Ch = cache[key]
        return (ttnn.to_device(Sh, self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                ttnn.to_device(Ch, self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG))  # fmt: skip

    def gather_state(self, Sm: ttnn.Tensor, Cm: ttnn.Tensor):
        """Inverse of shard_state (host): per-chip S / C -> the full single-chip layout."""
        Ss = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(Sm)]
        Cs = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(Cm)]
        S = torch.cat(Ss, dim=1)
        C = torch.cat(
            [torch.cat([c[..., k * self.w : (k + 1) * self.w] for c in Cs], dim=-1) for k in range(3)], dim=-1
        )
        return S, C


def check_lossless(shards: TPShards, name: str, dtype=ttnn.bfloat8_b) -> bool:
    """Host check: a tile-aligned slice of a cache tensor re-quantised to `dtype` reproduces its values exactly."""
    full = shards.host(name).float()
    sl = full[..., : 2 * TILE * 16] if full.shape[-1] >= 2 * TILE * 16 else full
    t = ttnn.from_torch(sl.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return bool(torch.equal(ttnn.to_torch(t).float(), sl))
