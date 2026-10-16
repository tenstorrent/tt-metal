# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""pplx-embed-v1-4B MLP with fused SwiGLU on the batched prefill path.

The stock MLP runs FF1 (gate) and FF3 (up) as two separate matmuls and then a
third op to combine them::

    w1_out = matmul(x, w1)                     # gate
    w3_out = matmul(x, w3)                     # up
    w2_in  = mul(w1_out, w3_out, act=[silu])   # BinaryNg

``ttnn.experimental.minimal_matmul`` can do all three in one op via
``fuse_swiglu=True``, which collapses gate/up tile-pairs to ``silu(gate)*up``
as it packs. At batch>=8 that matters: BinaryNg is 78.5 ms of the 502 ms
bs=32 device total (15.6%, the largest non-matmul cost), the SwiGLU multiply is
one of its three ops per layer, and the two intermediates are DRAM-resident, so
fusing also removes their round-trips and one matmul launch.

Weight layout
-------------
The op's contract (minimal_matmul_device_operation.cpp): "The weight packs
gate/up tile-pairs along N, so its width must be an even number of tiles; the
matmul output collapses each pair to one tile (silu(gate)*up)."

That is *tile-pair interleaved*, not a plain ``[gate | up]`` concat — 32-column
tiles alternate g0,u0,g1,u1,... Verified at real 4B shapes (K=2560, N=9728,
M=512): interleaved gives PCC 0.9997 against
``silu(x @ gate) * (x @ up)`` while plain concat gives -0.0002, i.e. the wrong
layout is silent numerical garbage rather than an error.

Scope
-----
Only the prefill path that already routes to ``minimal_matmul`` is fused. Decode,
galaxy/multi-device, the legacy MatmulMultiCoreReuseMultiCast path used at bs=1,
and anything whose shape fails the op's constraints fall through to the stock
implementation. Opt out entirely with ``QWEN_FUSE_SWIGLU=0``.
"""

import os

import torch

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.silu_mul import silu_mul
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.silu_mul import supported as _silu_mul_supported
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import Mode, OpGroup

TILE = 32


def _pack_gate_up_tile_pairs(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Interleave gate/up along N in ``TILE``-column groups.

    gate, up: ``[K, N]`` -> ``[K, 2N]`` ordered g0,u0,g1,u1,... per 32-col tile.
    """
    k, n = gate.shape
    assert n % TILE == 0, f"N={n} must be tile-aligned"
    g = gate.reshape(k, n // TILE, TILE)
    u = up.reshape(k, n // TILE, TILE)
    return torch.stack([g, u], dim=2).reshape(k, 2 * n)


def _wrap_silu_mul(original_mul, min_rows):
    """Route the stock ``ttnn.mul(a, b, input_tensor_a_activations=[SILU])`` (the SwiGLU product
    on the unfused path) to ``custom_ops.silu_mul`` for large M. Standalone at the bs32 shape
    (``[1,32,512,9728]`` bfp8): 1591 -> 1373 us (-13.7%) and closer to torch (PCC 0.99941 vs
    0.99897). Below ``min_rows`` (bs1, [1,1,512,9728] in L1) the op runs mode 3, minimal_matmul's single-pass SwiGLU
    with its bfp8-sized sigmoid: 52.1 -> 30.2 us device per call and closer to an fp32 SwiGLU (rel. RMSE 0.0113 vs
    the stock 0.0176; perf_tools/bench_bs1_swiglu.py). The product is SFPU-bound there, so the block-sharded path on
    FF1 / FF3's 96 cores is slower (33.6) than 120 cores reading interleaved tiles. QWEN_SILU_MUL_BS1=0: stock mul."""

    def wrapper(a, b, *args, **kwargs):
        acts = kwargs.get("input_tensor_a_activations")
        rows = None
        if (
            not args
            and acts is not None
            and len(acts) == 1
            and acts[0] == ttnn.UnaryOpType.SILU
            and kwargs.get("input_tensor_b_activations") is None
            and kwargs.get("activations") is None
            and hasattr(a, "padded_shape")
            and _silu_mul_supported(a, b)
        ):
            rows = int(a.padded_shape[-2]) * int(a.padded_shape[-3]) * int(a.padded_shape[0])
            if rows < min_rows and os.getenv("QWEN_SILU_MUL_BS1", "1") != "1":
                rows = None
        if rows is not None:
            out = silu_mul(
                a,
                b,
                out_dtype=kwargs.get("dtype"),
                memory_config=kwargs.get("memory_config"),
                mode=None if rows >= min_rows else 3,
            )
            if os.getenv("QWEN_SILU_MUL_VERIFY", "0") == "1":
                import torch

                ref = original_mul(a, b, *args, **kwargs)
                o, r = ttnn.to_torch(out).float().flatten(), ttnn.to_torch(ref).float().flatten()
                aq, bq = ttnn.to_torch(a).float().flatten(), ttnn.to_torch(b).float().flatten()
                gold = torch.nn.functional.silu(aq) * bq
                pcc = lambda x, y: torch.corrcoef(torch.stack([x, y]))[0, 1].item()
                print(
                    f"[verify silu_mul] pcc(fused,stock)={pcc(o, r):.6f} pcc(fused,torch)={pcc(o, gold):.6f} "
                    f"pcc(stock,torch)={pcc(r, gold):.6f}",
                    flush=True,
                )
                ttnn.deallocate(ref)
            return out
        return original_mul(a, b, *args, **kwargs)

    return wrapper


def _wrap_ff13_1d(original_linear, w1, w3, device):
    """bs1 FF1 / FF3 ([512, 2560] x [2560, 9728], bfp4 weights DRAM width-sharded over the 8 banks) on the 1D
    in0-multicast matmul over 12x10 instead of the 2D multicast on 12x8: every core owns all 16 M tiles and 3 N tiles
    (102 cores working instead of 96) and reads its weight slice straight from its DRAM bank(s) (the 1D factory's
    IN1_DRAM_WIDTH_SHARDED path). in0 must be width-sharded: the norm's 10x8 block-sharded output is resharded once
    onto 10 cores (512 x 256 each, 1.6 us) and both projections read it. Standalone 69.6 -> 61.9 us per call
    (perf_tools/bench_bs1_mm_ablate.py). QWEN_BS1_FF13_1D=0: the 2D path."""
    grid = device.compute_with_storage_grid_size()
    ws10 = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 0))}),
            [512, 256],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=3,
        out_block_h=16,
        out_block_w=3,
        per_core_M=16,
        per_core_N=3,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    state = {"src": None, "x": None, "uses": 0}

    def eligible(x, w):
        if w is not w1 and w is not w3 or grid.x < 12 or grid.y < 10:
            return False
        mc = x.memory_config()
        return (
            mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
            and list(x.padded_shape)[-2:] == [512, 2560]
            and list(w.padded_shape)[-2:] == [2560, 9728]
            and w.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            and w.memory_config().buffer_type == ttnn.BufferType.DRAM
        )

    def wrapper(x, w, *args, **kwargs):
        if args or not eligible(x, w):
            return original_linear(x, w, *args, **kwargs)
        if state["src"] is not x:
            state["src"], state["x"], state["uses"] = x, ttnn.to_memory_config(x, ws10), 0
        kwargs = dict(kwargs, program_config=pc, core_grid=None)
        out = original_linear(state["x"], w, **kwargs)
        state["uses"] += 1
        if state["uses"] == 2:  # FF1 and FF3 both done: free the resharded copy
            ttnn.deallocate(state["x"])
            state["src"], state["x"] = None, None
        return out

    return wrapper


class PplxFusedSwigluMLP(MLP):
    """MLP whose batched-prefill FF1/FF3/mul collapse into one fused matmul."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default off: the demo's apply_workload_env opts in for the batch sizes
        # where it measured faster (see its comment). Anything constructing this
        # class directly gets stock behaviour unless it asks for the fusion.
        self.fuse_swiglu_enabled = os.getenv("QWEN_FUSE_SWIGLU", "0") == "1"
        self.w_gate_up = None
        if self.fuse_swiglu_enabled:
            self._build_packed_gate_up(kwargs.get("state_dict"), kwargs.get("weight_cache_path"))

    # ------------------------------------------------------------------ #
    def _build_packed_gate_up(self, state_dict, weight_cache_path):
        """Pack w1/w3 into one tile-pair-interleaved device tensor.

        Packing happens once at construction. If anything about the checkpoint
        layout does not match expectations we leave ``w_gate_up`` as None and the
        forward path silently keeps using the stock two-matmul implementation.
        """
        try:
            # Must match the base class exactly: it derives the prefix from
            # self.__class__.__name__, which is why this class is renamed to "MLP"
            # at the bottom of this file. Ask for "MLP" explicitly here so the
            # lookup cannot drift if that alias is ever removed.
            prefix = self.args.get_state_dict_prefix("MLP", self.layer_num)
            w1 = state_dict[f"{prefix}.w1.weight"]
            w3 = state_dict[f"{prefix}.w3.weight"]
        except (KeyError, TypeError, AttributeError):
            return

        # Checkpoint stores [out_features, in_features]; the matmul wants [K, N].
        gate, up = w1.transpose(-2, -1), w3.transpose(-2, -1)
        if gate.shape != up.shape or gate.shape[-1] % TILE != 0:
            return

        packed = _pack_gate_up_tile_pairs(gate, up).unsqueeze(0).unsqueeze(0)
        # The base __init__ has already resolved this for w1/w3; reuse it so the
        # packed tensor cannot drift from the unpacked ones.
        self.w_gate_up = ttnn.as_tensor(
            packed,
            dtype=self.ff1_3_dtype or ttnn.bfloat8_b,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=(
                None if weight_cache_path is None else str(weight_cache_path / f"{prefix}.w_gate_up_swiglu")
            ),
        )

    # ------------------------------------------------------------------ #
    def _can_fuse(self, x, mode, seq_len) -> bool:
        if not self.fuse_swiglu_enabled or self.w_gate_up is None:
            return False
        if mode != Mode.PREFILL:
            return False
        if self.args.is_galaxy or self.args.num_devices > 1:
            return False
        # Only where the stock path would already have used minimal_matmul; the
        # legacy 2D path (bs=1 short-seq on this model) is faster left alone --
        # except that at bs=1 the three ops it replaces (FF1 104 us + FF3 104 us
        # + SwiGLU mul 156 us = 364 us/layer, the mul alone being 3.8 ms/iter)
        # make one fused minimal_matmul worth trying. QWEN_FUSE_SWIGLU_BS1=1
        # opts the legacy-path shapes in; forward() then builds a bs=1-tuned
        # MinimalMatmulConfig since get_mlp_ff1_3_prg_config returns the legacy
        # config there.
        if not self.args.use_minimal_prefill_matmul(seq_len) and os.getenv("QWEN_FUSE_SWIGLU_BS1", "0") != "1":
            return False
        # fuse_swiglu needs the packed width to be an even number of tiles.
        return (2 * self.args.hidden_dim) % (2 * TILE) == 0

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        seq_len = x.shape[-2]
        if not self._can_fuse(x, mode, seq_len):
            if self._can_silu_in_ff1(mode, seq_len):
                return self._forward_silu_in_ff1(x, mode, seq_len)
            if mode == Mode.PREFILL and os.getenv("QWEN_SILU_MUL", "1") == "1":
                # Unfused SwiGLU path (bs32 opt-out, bs1): the silu(a)*b product as one model-local generic_op,
                # and at bs1 FF1 / FF3 on the 1D matmul over 120 cores.
                _orig_mul, _orig_linear = ttnn.mul, ttnn.linear
                ttnn.mul = _wrap_silu_mul(_orig_mul, int(os.getenv("QWEN_SILU_MUL_MIN_ROWS", "8192")))
                if os.getenv("QWEN_BS1_FF13_1D", "1") == "1":
                    ttnn.linear = _wrap_ff13_1d(_orig_linear, self.w1, self.w3, self.mesh_device)
                try:
                    return super().forward(x, mode)
                finally:
                    ttnn.mul, ttnn.linear = _orig_mul, _orig_linear
            return super().forward(x, mode)

        layer = max(self.layer_num, 0)
        ckc = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )
        pc = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        if not isinstance(pc, ttnn.MinimalMatmulConfig):
            if os.getenv("QWEN_FUSE_SWIGLU_BS1", "0") != "1":
                return super().forward(x, mode)
            pc = self._bs1_fused_config(ckc)

        prefill_mem_seq = int(seq_len)
        w2_in = ttnn.experimental.minimal_matmul(
            x,
            self.w_gate_up,
            config=pc,
            compute_kernel_config=ckc,
            memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher, prefill_seq_len=prefill_mem_seq),
            fuse_swiglu=True,
        )
        ttnn.deallocate(x)
        return self._down_project(w2_in, mode, seq_len)

    def _bs1_fused_config(self, ckc) -> "ttnn.MinimalMatmulConfig":
        """MinimalMatmulConfig for the fused FF1/FF3 at bs=1 (M=512).

        Defaults are the best of perf_tools/bench_ff13_sweep.py 1 with the pack-thread SwiGLU kernel (2,20,8 1x2,
        214 us standalone); the fused path still loses to the legacy FF1 + FF3 + mul at bs1 e2e (15.7 -> 16.0 ms,
        NEGATIVE_RESULTS 58), so it stays opt-in. A subblock is capped at the DST half (8 tiles, 4 with
        fp32_dest_acc_en). K_block 40, which hides the SwiGLU batched (§60), is slower here: at M=512 the SwiGLU is
        only 3% exposed and the kernel is data-movement-bound (NEGATIVE_RESULTS 61). Needs QWEN_FUSE_SWIGLU=1 as well
        (the demo defaults it off at bs1, and without it no packed weight is built). Grid is the full device, clamped
        for harvested parts. QWEN_MM_BLOCK_FF13 / QWEN_MM_SUBBLOCK_FF13 are honoured if set so the config can be swept
        like the others.
        """
        gx, gy = self.args._clamp_grid_to_device((12, 10))
        mb, kb, nb = self.args._resolve_mm_blocks("QWEN_MM_BLOCK_FF13", default=(2, 20, 8))
        fp32 = bool(getattr(ckc, "fp32_dest_acc_en", False))
        sbh, sbw = self.args._resolve_mm_subblocks("QWEN_MM_SUBBLOCK_FF13", default=(1, 2))
        # The resolver falls back to the global QWEN_MM_SUBBLOCK (1,8 in the demo): clamp to the DST budget and keep
        # the op's N_block_size % subblock_w == 0 invariant.
        cap = 4 if fp32 else 8
        sbw = min(sbw, cap)
        while sbw > 1 and nb % sbw:
            sbw -= 1
        return ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sbh,
            subblock_w=sbw,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        )

    # ------------------------------------------------------------------ #
    def _can_silu_in_ff1(self, mode, seq_len) -> bool:
        """Unfused FF1/FF3 with the SiLU moved into FF1's matmul epilogue.

        Where fuse_swiglu is off (bs=32: the fused matmul lost more than the mul
        saved) the SwiGLU multiply is a BinaryNg with a SiLU on input A. Standalone
        that SiLU is ~55% of the op (75 vs 34 us at bs=1 shapes) and at bs=32 the
        mul is ~50 ms/iter of kernel time. minimal_matmul exposes
        ``fused_activation``, so FF1 can emit silu(x@w1) directly and the mul
        becomes a plain multiply. Opt-in: QWEN_SILU_IN_FF1=1.
        """
        if os.getenv("QWEN_SILU_IN_FF1", "0") != "1" or mode != Mode.PREFILL:
            return False
        if self.args.is_galaxy or self.args.num_devices > 1:
            return False
        return self.args.use_minimal_prefill_matmul(seq_len)

    def _forward_silu_in_ff1(self, x, mode, seq_len):
        layer = max(self.layer_num, 0)
        ckc = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )
        pc = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        if not isinstance(pc, ttnn.MinimalMatmulConfig):
            return super().forward(x, mode)
        mem = self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher, prefill_seq_len=int(seq_len))
        w1_out = ttnn.experimental.minimal_matmul(
            x,
            self.w1,
            config=pc,
            compute_kernel_config=ckc,
            memory_config=mem,
            dtype=self.ff1_3_dtype,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )
        w3_out = ttnn.experimental.minimal_matmul(
            x, self.w3, config=pc, compute_kernel_config=ckc, memory_config=mem, dtype=self.ff1_3_dtype
        )
        ttnn.deallocate(x)
        w2_in = ttnn.mul(w1_out, w3_out, dtype=self.ff1_3_dtype or ttnn.bfloat8_b, memory_config=w1_out.memory_config())
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        return self._down_project(w2_in, mode, seq_len)

    # ------------------------------------------------------------------ #
    def _down_project(self, w2_in, mode, seq_len):
        """FF2 plus the tail the stock forward does after the SwiGLU combine."""
        layer = max(self.layer_num, 0)
        li_ff2_ckc = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer, op=OpGroup.LI_FF2, configuration=self.args
        )
        prefill_mem_seq = int(seq_len)
        pc_2 = self.args.get_mlp_ff2_prg_config(mode, seq_len, self.prefetcher)
        out_mem = self.args.get_mlp_ff2_mem_config(mode, self.prefetcher, prefill_seq_len=prefill_mem_seq)

        if isinstance(pc_2, ttnn.MinimalMatmulConfig):
            w2_out = ttnn.experimental.minimal_matmul(
                w2_in, self.w2, compute_kernel_config=li_ff2_ckc, config=pc_2, memory_config=out_mem
            )
        else:
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_ckc,
                program_config=pc_2,
                memory_config=out_mem,
                core_grid=None,
            )
        ttnn.deallocate(w2_in)

        original_shape = w2_out.shape
        w2_out = ttnn.reshape(
            w2_out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        return w2_out


# The base MLP derives its state_dict prefix from ``self.__class__.__name__``
# (args.get_state_dict_prefix(self.__class__.__name__, layer_num)), so a subclass
# with a different name would look up weights under the wrong prefix and fail to
# load. tt/attention.py does the same for PplxBidirectionalAttention.
PplxFusedSwigluMLP.__name__ = "MLP"
