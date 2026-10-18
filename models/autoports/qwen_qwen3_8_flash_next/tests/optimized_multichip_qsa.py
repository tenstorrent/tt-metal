"""Mesh-local KV precision and explicit sparse-SDPA chunk candidates."""

import argparse
import sys
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--cache-fp8", action="store_true")
    p.add_argument("--decode-k-chunk", type=int, default=32)
    p.add_argument("--prefill-k-chunk", type=int, default=128)
    a, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    build = target.ParallelQSA.build.__func__

    def construct(cls, *args, **kw):
        obj = build(cls, *args, **kw)
        if obj.cache_dtype != (ttnn.fp8_e4m3 if a.cache_fp8 else ttnn.bfloat16):
            shape = tuple(obj.packed_cache.shape)
            del obj.packed_cache
            obj.cache_dtype = ttnn.fp8_e4m3 if a.cache_fp8 else ttnn.bfloat16
            obj.packed_cache = ttnn.empty(
                shape,
                dtype=obj.cache_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=obj.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        obj.decode_chunk = a.decode_k_chunk
        obj.sparse_chunk = a.prefill_k_chunk
        return obj

    target.ParallelQSA.build = classmethod(construct)
    if a.validate:
        from models.autoports.qwen_qwen3_8_flash_next.tests import validate_multichip_decoder as bench
    else:
        from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench
    bench.CANDIDATE = vars(a)
    bench.main()


if __name__ == "__main__":
    main()
