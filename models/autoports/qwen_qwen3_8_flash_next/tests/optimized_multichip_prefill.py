"""Real-shape mesh prefill 2D/minimal kernel families; decode is untouched."""

import argparse
import math
import sys
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--prefill-kernel", choices=["2d", "minimal", "l1"], default="2d")
    p.add_argument("--grid-x", type=int, default=10)
    p.add_argument("--k-block", type=int, default=8)
    p.add_argument("--packed-k-cap", type=int, default=None)
    p.add_argument("--subblock-h", type=int, default=1)
    p.add_argument("--n-block", type=int, default=4)
    p.add_argument("--cast-input", action="store_true")
    p.add_argument("--prefill-role", default="all")
    p.add_argument("--large-outer-only", action="store_true")
    p.add_argument("--stage-prefill-output", action="store_true")
    p.add_argument("--preserve-memory", action="store_true")
    a, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    original = ttnn.linear
    if a.prefill_kernel == "l1":
        read = target.ParallelHyperConnection.read
        write = target.ParallelHyperConnection.write

        def l1_read(self, x, **kw):
            if not kw.get("decode", False):
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            return read(self, x, **kw)

        def l1_write(self, x, update, inject):
            b, s = tuple(x.shape)[1:3]
            if s == 1:
                return write(self, x, update, inject)
            scaled = ttnn.multiply(
                ttnn.reshape(update, (1, b * s, 1, 640)),
                ttnn.reshape(inject, (1, b * s, 4, 1)),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            result = ttnn.add(ttnn.reshape(x, (1, b * s, 4, 640)), scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
            return ttnn.reshape(result, x.shape)

        target.ParallelHyperConnection.read = l1_read
        target.ParallelHyperConnection.write = l1_write
    large_outer = False
    stage_output = False
    outer = target.MultichipDecoder._chunk
    prefill_outer = target.MultichipDecoder.prefill_forward

    def prefill(self, x, **kw):
        nonlocal stage_output
        prior = stage_output
        stage_output = x.shape[2] > self.chunk_size
        try:
            return prefill_outer(self, x, **kw)
        finally:
            stage_output = prior

    def chunk(self, x, **kw):
        nonlocal large_outer
        prior = large_outer
        large_outer = not kw["decode"] and x.shape[2] >= 256
        try:
            result = outer(self, x, **kw)
            if stage_output and not kw["decode"]:
                result = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)
            return result
        finally:
            large_outer = prior

    if a.large_outer_only or a.stage_prefill_output:
        target.MultichipDecoder._chunk = chunk
    if a.stage_prefill_output:
        target.MultichipDecoder.prefill_forward = prefill
    build = target.MultichipDecoder.from_state_dict.__func__
    selected = {}
    rows = {}

    def construct(cls, *args, **kw):
        obj = build(cls, *args, **kw)
        groups = {
            "hc": [(h, n) for h in (obj.attn_hyper, obj.mlp_hyper) for n in ("read_weight", "up")],
            "attn": [
                (obj.attention, n)
                for n in ("packed_weight", "out_weight", "packed_projection", "wo")
                if hasattr(obj.attention, n)
            ],
            "shared": [(obj.mlp, n) for n in ("router", "shared_gu", "shared_down")],
            "ple": [(obj.ple, "projection")] if obj.ple else [],
        }
        for role, entries in groups.items():
            if a.prefill_role not in ("all", role):
                continue
            for owner, name in entries:
                selected[getattr(owner, name).buffer_address()] = role + "/" + name
        return obj

    def linear(x, w, **kw):
        if (
            (a.large_outer_only and not large_outer)
            or w.buffer_address() not in selected
            or x.shape[-2] < 2
            or math.prod(tuple(x.shape)[:-2]) != 1
        ):
            return original(x, w, **kw)
        m = math.ceil(x.shape[-2] / 32)
        k = math.ceil(w.shape[-2] / 32)
        n = math.ceil(w.shape[-1] / 32)
        block = min(a.k_block, k)
        if a.packed_k_cap and selected[w.buffer_address()] in ("attn/packed_projection", "attn/packed_weight"):
            block = min(block, a.packed_k_cap)
        while k % block:
            block -= 1
        gx = min(a.grid_x, n)
        gy = min(8, m)
        pn = math.ceil(n / gx)
        pm = math.ceil(m / gy)
        sub_h = min(a.subblock_h, pm, 4)
        while pm % sub_h:
            sub_h -= 1
        sub = min(4 // sub_h, pn)
        while pn % sub:
            sub -= 1
        dtype = kw.get("dtype") or x.dtype
        compute = kw.get("compute_kernel_config")
        key = (selected[w.buffer_address()], m, k, n)
        rows[str(key)] = dict(
            role=key[0],
            m=m * 32,
            k=k * 32,
            n=n * 32,
            logical_m=x.shape[-2],
            input_dtype=str(x.dtype),
            weight_dtype=str(w.dtype),
            output_dtype=str(dtype),
            compute=str(compute),
            grid=[gx, gy],
            k_block=block,
            per_core_m=pm,
            per_core_n=pn,
            subblock=[sub_h, sub],
        )
        if a.prefill_kernel == "l1":
            rows[str(key)] = dict(
                role=key[0],
                logical_m=x.shape[-2],
                input_dtype=str(x.dtype),
                weight_dtype=str(w.dtype),
                output_dtype=str(dtype),
                compute=str(compute),
                program_config=str(kw.get("program_config")),
                input_memory="L1_INTERLEAVED",
                output_memory="L1_INTERLEAVED",
            )
            xx = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            return original(xx, w, **dict(kw, memory_config=ttnn.L1_MEMORY_CONFIG))
        if a.prefill_kernel == "minimal":
            xx = ttnn.typecast(x, w.dtype) if a.cast_input else x
            pc = ttnn.MinimalMatmulConfig(
                M_block_size=1,
                K_block_size=block,
                N_block_size=a.n_block,
                subblock_h=1,
                subblock_w=min(4, a.n_block),
                compute_with_storage_grid_size=ttnn.CoreCoord(10, 8),
            )
            return ttnn.experimental.minimal_matmul(
                xx, w, config=pc, dtype=dtype, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=block,
            out_subblock_h=sub_h,
            out_subblock_w=sub,
            per_core_M=pm,
            per_core_N=pn,
            transpose_mcast=False,
            fuse_batch=False,
        )
        return original(
            x,
            w,
            program_config=pc,
            compute_kernel_config=compute,
            dtype=dtype,
            memory_config=(
                kw.get("memory_config", ttnn.DRAM_MEMORY_CONFIG) if a.preserve_memory else ttnn.DRAM_MEMORY_CONFIG
            ),
        )

    target.MultichipDecoder.from_state_dict = classmethod(construct)
    ttnn.linear = linear
    from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench

    bench.CANDIDATE = {"prefill": vars(a), "rows": rows}
    bench.main()


if __name__ == "__main__":
    main()
